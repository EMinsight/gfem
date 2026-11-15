"""Post-hoc measurements on result fields.

A result carries every quantity that needs no extra input. A quantity that
needs extra physics input (a probe point, an interface surface with its
thickness and permittivity, a voltage path) is a measurement: a function
called on a result that carries its fields, after the solve, also on a
result reloaded with `gfem.load_result`.

- `Field.at`: a field's value at points.
- `interface_participation`: the electric energy of a mode in a thin
  dielectric layer on a surface, and its share of ``1 / Q``.
- `characteristic_impedance`: the power-voltage impedance of a port mode
  along a voltage path.
- `power_flow`: the net power a driven field carries out of a set of
  elements, exact for the solved field (a flux of point-sampled
  ``E x H*`` is not, see the measurements recipe).
- `electric_energy`: the time-averaged electric energy a field stores in
  each volume region, with the model's materials.

Measurements run on the CPU and work under `jax.jit` and `jax.vmap`; under
`jax.grad` they differentiate with respect to the field coefficients (so
material-field gradients flow through the solve that made the field) and
the measurement's own numbers. That includes the normalized mode fields of
an eigenmode solve, which differentiate with respect to `gfem.MaterialField`
properties and `gfem.SheetField` fills for an isolated mode (see the
"Differentiate eigenfrequencies, Q and participation ratios" recipe).
Mesh-node gradients raise `gfem.NonDifferentiableError`: for a shape
derivative of a measurement, take finite differences of it over meshes
moved by `gfem.shape.boundary_morph`.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _mpi, _objective, _probe
from gfem._checks import check_fixed, is_traced
from gfem._materials import Material, _is_law, field_principal_values
from gfem._pytree import register_fields
from gfem.conventions import MODE_ENERGY
from gfem.errors import MeshRefinementUnsupportedError, NonDifferentiableError
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import EigenmodeResult, PortModeResult

__all__ = [
    "INTERFACE_KINDS",
    "InterfaceParticipation",
    "characteristic_impedance",
    "electric_energy",
    "interface_participation",
    "power_flow",
]

# --- interface participation -------------------------------------------------------

#: The thin-layer models of `interface_participation` (Wenner et al., Appl.
#: Phys. Lett. 99, 113513 (2011)).
InterfaceKind = Literal["default", "MA", "MS", "SA"]
INTERFACE_KINDS = get_args(InterfaceKind)
#: A medium is vacuum when its slowest light speed is within this of c.
_VACUUM = 1.0 - 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class InterfaceParticipation:
    """The share of each mode's electric energy in a thin interface layer.

    Attributes:
        participation: ``(n_modes,)`` the layer's electric energy over the
            mode's electric energy ``E`` (see `gfem.EigenmodeResult`).
        inverse_q: ``(n_modes,)`` the layer's dielectric-loss contribution to
            ``1 / Q``, ``participation * tan_delta``; ``None`` when no loss
            tangent was given.
    """

    participation: jax.Array
    inverse_q: jax.Array | None = None

    @property
    def q(self) -> jax.Array:
        """The Q the layer's loss alone would give, ``1 / inverse_q``.

        Raises:
            ValueError: No loss tangent was given.
        """
        if self.inverse_q is None:
            raise ValueError(
                "This interface participation has no loss: pass tan_delta= to "
                "gfem.interface_participation."
            )
        return 1.0 / self.inverse_q

    def __repr__(self) -> str:
        try:
            p = np.asarray(self.participation)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "InterfaceParticipation(traced)"
        text = ", ".join(f"{x:.4g}" for x in p)
        if self.inverse_q is not None:
            text += "; 1/Q " + ", ".join(f"{x:.4g}" for x in np.asarray(self.inverse_q))
        return f"InterfaceParticipation([{text}])"


register_fields(InterfaceParticipation, ("participation", "inverse_q"))


def interface_participation(
    modes: EigenmodeResult,
    surface: str | Sequence[str],
    thickness: Any,
    eps_r: Any,
    *,
    kind: InterfaceKind = "default",
    tan_delta: Any = None,
) -> InterfaceParticipation:
    """Electric energy participation of a thin dielectric layer on a surface.

    The layer (an oxide on a metal, a residue on a substrate) is too thin to
    mesh: it has thickness ``t`` and relative permittivity ``eps_r`` and
    sits on ``surface``. Its electric energy follows from the mode's field
    at the surface, for each kind of interface (Wenner et al. 2011):

    - ``"MA"`` (metal-air): ``t / eps_r * |E_n|^2``, ``E`` on the vacuum
      side of the surface.
    - ``"MS"`` (metal-substrate): ``t / eps_r * |(eps_s E)_n|^2``, ``E``
      and the substrate permittivity ``eps_s`` (real part, a tensor if
      anisotropic) on the non-vacuum side: the layer carries the
      substrate's normal flux.
    - ``"SA"`` (substrate-air): ``t * (eps_r |E_t|^2 + |E_n|^2 / eps_r)``,
      ``E`` on the vacuum side.
    - ``"default"``: ``t * eps_r * |E|^2``, ``E`` averaged over the two
      sides of an interior surface. Not for a film with the same medium on
      both faces: there the two faces' normal fields are opposite and the
      mean cancels.

    each times ``eps0 / 2`` and integrated over the surface. A medium is
    vacuum when no wave in it is slower than in vacuum: its slowest light
    speed, from its largest principal ``eps_r`` and ``mu_r``, is that of
    vacuum (to 1e-6), so an anisotropic side such as ``eps_r = (1, 1, 4)`` is
    not vacuum. The layer itself is isotropic. For ``"MA"`` and ``"MS"``
    each side of the wanted kind carries its own layer and the layers'
    energies add: a film with vacuum on both faces has one oxide per face,
    and its ``"MA"`` participation is the sum of the two faces'. Asked on a
    surface that is not a film and has the same medium on both sides, they
    count the layer twice, which is a misuse. For ``"SA"`` and ``"default"``
    the field is averaged over the sides of the wanted kind. Where no side
    is of the wanted kind (``"MA"`` on a plate under substrate), the surface
    adds nothing. Sides inside volumes the model excludes from the solve
    (metal) do not count. The participation is that energy over the mode's
    electric energy ``E`` (1 J, see `gfem.conventions`), and
    ``participation * tan_delta`` is the layer's contribution to ``1 / Q``.

    On a zero-thickness film the field at the film edge is singular, and the
    participation of a layer that reaches the edge grows without bound as
    the mesh is refined there: by about 6 to 7 % of its value each time the
    edge cells are halved, on a 10 um strip. It is not a mesh-converged
    number. Compare designs on meshes with the same edge cell size, and take
    that size near the real film thickness for an absolute value.

    Differentiable with respect to ``thickness``, ``eps_r``, ``tan_delta``,
    the mode fields and the substrate's `MaterialField` permittivity (the
    ``"MS"`` flux), per principal axis for a field with ``axes``. Through
    the eigenmode solve, the mode fields differentiate with respect to
    `gfem.MaterialField` properties and `gfem.SheetField` fills of an
    isolated mode (see the "Differentiate eigenfrequencies, Q and
    participation ratios" recipe; a mode of a degenerate cluster raises
    `gfem.DegenerateModeError`). Mesh-node gradients raise
    `gfem.NonDifferentiableError`: take finite differences of the
    participation over meshes moved by `gfem.shape.boundary_morph` instead.

    Args:
        modes: An eigenmode result.
        surface: A surface region, or several (their layers add up).
        thickness: The layer thickness ``t``, m, > 0.
        eps_r: The layer's relative permittivity, > 0.
        kind: One of `INTERFACE_KINDS`.
        tan_delta: The layer's loss tangent, >= 0, or ``None``.

    Returns:
        Per-mode participation and, with ``tan_delta``, its ``1 / Q``.

    Raises:
        TypeError: ``modes`` is not a `gfem.EigenmodeResult`.
        RegionNotFoundError: The mesh has no region of that name.
        TypeError: ``eps_r`` is a `gfem.PermittivityLaw` (the layer is a
            number).
        ValueError: A name is a volume region or a surface of a
            `gfem.Periodic` pair (interior to the periodic cell), a number is
            out of range, the result does not carry its materials (saved
            by an older gfem), or a side of the surface lies in a region
            with a permittivity law.
    """
    from gfem.results import EigenmodeResult

    if not isinstance(modes, EigenmodeResult):
        raise TypeError(
            "interface_participation needs a gfem.EigenmodeResult, got "
            f"{type(modes).__name__}."
        )
    if kind not in INTERFACE_KINDS:
        raise ValueError(
            f"Unknown interface kind {kind!r}; expected one of "
            f"{', '.join(repr(k) for k in INTERFACE_KINDS)}."
        )
    _positive(thickness, "thickness", strict=True)
    if _is_law(eps_r):
        raise TypeError(
            "The interface layer's eps_r must be a number; dispersive layers are "
            f"not supported (got {type(eps_r).__name__})."
        )
    _positive(eps_r, "eps_r", strict=True)
    if tan_delta is not None:
        _positive(tan_delta, "tan_delta", strict=False)
    if not modes.e_fields:
        raise ValueError("This eigenmode result holds no mode fields.")
    if not modes.materials:
        raise ValueError(
            "This eigenmode result does not carry the materials it was solved "
            "with (it was saved by an older gfem): solve the model again."
        )
    field = modes.e_fields[0]
    surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
    # read from the result, never from a cache: the pairs of this model only
    for pair in modes.periodic_surfaces:
        for name in surfaces:
            if name in pair:
                raise ValueError(
                    f"{name!r} is a surface of periodic pair {pair!r}, which "
                    "is interior to the periodic cell: it holds no interface "
                    "layer."
                )
    geometry = _surface_geometry(field, surfaces)
    _refuse_law_sides(modes.materials, field.mesh, geometry, surfaces)
    coefficients = jnp.stack([f.coefficients for f in modes.e_fields])
    energy = _layer_energy(
        geometry,
        kind,
        coefficients=coefficients,
        nodes=field.mesh.nodes,
        materials=modes.materials,
        thickness=jnp.asarray(thickness, float),
        eps_l=jnp.asarray(eps_r, float),
    )
    participation = energy / MODE_ENERGY
    inverse_q = None if tan_delta is None else participation * jnp.asarray(tan_delta)
    return InterfaceParticipation(participation, inverse_q)


class _SurfaceGeometry:
    """The topology of a measurement surface: sides, quadrature points.

    Hashed by identity: one object per (mesh topology, surfaces, field
    layout), so `_layer_energy` compiles once per surface.
    """

    def __init__(self, mesh: Mesh, surfaces: tuple[str, ...], layout: Any) -> None:
        triangles = []
        for name in surfaces:
            region = mesh.region(name)
            if region.dim != 2:
                raise ValueError(
                    f"{name!r} is a volume region; interface_participation needs "
                    "a surface region."
                )
            triangles.append(mesh.elements(name))
        engine = _probe.engine()
        tri = np.concatenate(triangles)
        sides = mesh.triangle_neighbours[tri]
        inside = engine.in_support(mesh, layout)
        self.topology = mesh._topology  # noqa: SLF001
        self.layout = layout
        self.triangles = tri
        self.sides = sides
        self.exists = (sides >= 0) & inside[np.maximum(sides, 0)]
        bary, self.weights = engine.triangle_rule(2 * layout.order)
        self.n_q = len(bary)
        # the evaluation points: the quadrature points seen from each side
        self.pairs = [np.flatnonzero(self.exists[:, s]) for s in (0, 1)]
        parts = [
            engine.face_points(mesh, tri[k], sides[k, s], bary)
            for s, k in enumerate(self.pairs)
        ]
        self.located = engine.Located(
            np.concatenate([p.tetrahedra for p in parts]),
            np.concatenate([p.barycentric for p in parts]),
        )


_GEOMETRIES: dict[tuple, _SurfaceGeometry] = {}
_MAX_GEOMETRIES = 8


def _surface_geometry(field: Field, surfaces: tuple[str, ...]) -> _SurfaceGeometry:
    mesh = field.mesh
    layout = _probe.layout(field)
    key = (id(mesh._topology), surfaces, layout)  # noqa: SLF001
    found = _GEOMETRIES.get(key)
    if found is not None and found.topology is mesh._topology:  # noqa: SLF001
        return found
    made = _SurfaceGeometry(mesh, surfaces, layout)
    _GEOMETRIES[key] = made
    while len(_GEOMETRIES) > _MAX_GEOMETRIES:
        _GEOMETRIES.pop(next(iter(_GEOMETRIES)))
    return made


@functools.partial(jax.jit, static_argnames=("geometry", "kind"))
def _layer_energy(
    geometry: _SurfaceGeometry,
    kind: InterfaceKind,
    *,
    coefficients: jax.Array,
    nodes: jax.Array,
    materials: dict[str, Any],
    thickness: jax.Array,
    eps_l: jax.Array,
) -> jax.Array:
    """The layer's electric energy per mode, J (see `interface_participation`)."""
    g = geometry
    mesh = Mesh(nodes, g.topology)
    m, n_t, n_q = coefficients.shape[0], len(g.triangles), g.n_q
    values = _probe.evaluate(coefficients, mesh, g.layout, g.located)
    e = jnp.zeros((m, n_t, 2, n_q, 3), dtype=values.dtype)
    start = 0
    for s, k in enumerate(g.pairs):
        block = values[:, start : start + len(k) * n_q]
        e = e.at[:, k, s].set(block.reshape(m, len(k), n_q, 3))
        start += len(k) * n_q

    v = nodes[mesh.triangles[g.triangles]]
    cross = jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    twice_area = jnp.linalg.norm(cross, axis=1)
    normal = cross / twice_area[:, None]
    weights = 0.5 * twice_area[:, None] * g.weights[None, :]  # (n_t, q), m^2

    eps_side, vacuum = _media(materials, mesh, g.sides, g.exists)
    ex = jnp.asarray(g.exists)
    if kind == "default":
        density = eps_l * _norm2(_average(e, ex))
    elif kind == "MS":
        # one layer per selected side (ADR 0106): density per side, then summed
        d = jnp.einsum("tsij,mtsqj->mtsqi", eps_side, e)
        dn = jnp.einsum("mtsqi,ti->mtsq", d, normal)
        density = _sum_sides(_sq(dn), ex & ~vacuum) / eps_l
    elif kind == "MA":
        en = jnp.einsum("mtsqi,ti->mtsq", e, normal)
        density = _sum_sides(_sq(en), ex & vacuum) / eps_l
    else:  # SA
        selected = _average(e, ex & vacuum)
        en = jnp.einsum("mtqi,ti->mtq", selected, normal)
        et = selected - en[..., None] * normal[None, :, None, :]
        density = eps_l * _norm2(et) + _sq(en) / eps_l
    return 0.5 * const.epsilon_0 * thickness * jnp.einsum("mtq,tq->m", density, weights)


def _refuse_law_sides(
    materials: Any, mesh: Mesh, geometry: _SurfaceGeometry, surfaces: tuple[str, ...]
) -> None:
    """Raise if a side of a measured surface lies in a permittivity law region.

    The ``"MS"`` and ``"SA"`` layers read a side's Re eps_r, which a law does
    not have (ADR 0109). A law region that touches no measured face is
    skipped by `_media`.

    Raises:
        ValueError: A law region has a tetrahedron on a side of ``surfaces``.
    """
    sides = geometry.sides[geometry.exists]
    for region, material in materials.items():
        if not (isinstance(material, Material) and _is_law(material.eps_r)):
            continue
        if np.isin(sides, mesh.elements(region)).any():
            raise ValueError(
                f"interface_participation on {', '.join(map(repr, surfaces))}: "
                f"region {region!r} on a side of the surface holds a permittivity "
                "law, which has no one permittivity for the layer's field. Measure "
                "an interface between constant materials."
            )


def _positive(value: Any, name: str, *, strict: bool) -> None:
    if is_traced(value):
        return
    x = np.asarray(value)
    if x.shape != () or not np.isrealobj(x) or not np.isfinite(x):
        raise ValueError(f"{name} must be one finite real number, got {value!r}.")
    if (x <= 0) if strict else (x < 0):
        bound = "> 0" if strict else ">= 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}.")


def _sq(x: jax.Array) -> jax.Array:
    """``|x|^2`` elementwise."""
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2


def _norm2(x: jax.Array) -> jax.Array:
    """``|x|^2`` of vectors along the last axis."""
    return _sq(x).sum(axis=-1)


def _average(x: jax.Array, selected: jax.Array) -> jax.Array:
    """Mean over the selected sides: ``(m, t, 2, q, ...)`` -> ``(m, t, q, ...)``."""
    w = selected.astype(float)
    count = jnp.maximum(w.sum(axis=1), 1.0)
    w = w / count[:, None]
    shape = (1, *w.shape, 1) + (1,) * (x.ndim - 4)
    return (x * w.reshape(shape)).sum(axis=2)


def _sum_sides(x: jax.Array, selected: jax.Array) -> jax.Array:
    """Sum over the selected sides: ``(m, t, 2, q)`` -> ``(m, t, q)``."""
    w = selected.astype(float)
    return (x * w[None, :, :, None]).sum(axis=2)


def _media(
    materials: dict[str, Any], mesh: Mesh, sides: np.ndarray, exists: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Real permittivity tensors ``(t, 2, 3, 3)`` and vacuum flags ``(t, 2)``."""
    from gfem._engine.coefficients import _tensor

    tets = np.maximum(sides, 0)
    eps = jnp.zeros((*sides.shape, 3, 3)) + jnp.eye(3)
    # (c_min / c0)^2 >= 1 / (eps_max mu_max): no wave slower than in vacuum
    # means the side's permittivity is eps0 I, which the layers need
    speed2 = jnp.ones(sides.shape)
    for region, material in materials.items():
        members = mesh.elements(region)
        position = np.full(len(mesh.tetrahedra), -1)
        position[members] = np.arange(len(members))
        inside = jnp.asarray((position[tets] >= 0) & exists)
        if isinstance(material, Material) and _is_law(material.eps_r):
            # on no measured side (`_refuse_law_sides`): never read
            continue
        if isinstance(material, Material):
            tensor = jnp.asarray(_tensor(material).real)
            high = float(np.max(material.eps_r)) * float(np.max(material.mu_r))
            value = jnp.broadcast_to(tensor, eps.shape)
            high = jnp.full(sides.shape, high)
        elif material.axes is not None:
            # principal values along fixed axes: the tensor per element, and
            # the bound from the largest principal values
            index = np.maximum(position[tets], 0)
            e_r, m_r = (
                jnp.real(jnp.asarray(field_principal_values(material, name, index)))
                for name in ("eps_r", "mu_r")
            )
            a = jnp.asarray(material.axes)
            value = jnp.einsum("...k,ki,kj->...ij", e_r, a, a)
            value = jnp.broadcast_to(value, eps.shape)
            high = jnp.max(e_r, axis=-1) * jnp.max(m_r, axis=-1)
            high = jnp.broadcast_to(high, sides.shape)
        else:
            e_r = 1.0 if material.eps_r is None else material.eps_r
            m_r = 1.0 if material.mu_r is None else material.mu_r
            e_r = jnp.real(jnp.asarray(e_r))
            m_r = jnp.real(jnp.asarray(m_r))
            index = np.maximum(position[tets], 0)
            e_r = e_r[index] if e_r.ndim else jnp.full(sides.shape, e_r)
            m_r = m_r[index] if m_r.ndim else jnp.full(sides.shape, m_r)
            value = e_r[..., None, None] * jnp.eye(3)
            high = e_r * m_r
        eps = jnp.where(inside[..., None, None], value, eps)
        speed2 = jnp.where(inside, 1.0 / high, speed2)
    vacuum = speed2 >= _VACUUM**2
    return eps, vacuum


# --- characteristic impedance ------------------------------------------------------


def characteristic_impedance(mode: PortModeResult, voltage_path: Any) -> jax.Array:
    """Power-voltage characteristic impedance of a port mode along a path.

    ``Z_PV = |V|^2 / (2 P)`` with ``V`` the line integral of the mode's
    electric field along ``voltage_path`` and ``P = 1 W`` the mode's power
    For a wave port declared with a ``voltage_path``
    the result already holds it as ``Z_PV``; this measures it along any
    path, after the solve. A constant of the port mode solve: port modes
    are not differentiable.

    Args:
        mode: A port mode result.
        voltage_path: Two or more points (m) on the port surface; the path
            runs through them in order, e.g. from a ground to the signal
            conductor.

    Returns:
        ``(n_freq,)`` ohm.

    Raises:
        TypeError: ``mode`` is not a `gfem.PortModeResult`.
        ValueError: The path is malformed or leaves the port plane, or
            ``mode`` carries no field (a driven result computed inside
            `jax.jit`, `jax.grad` or `jax.vmap`).
        NonDifferentiableError: ``voltage_path`` is computed inside
            `jax.grad`, `jax.jit` or `jax.vmap` from their arguments; pass a
            fixed value.
    """
    from gfem._sweep.port_mode import path_voltage
    from gfem.results import PortModeResult

    if not isinstance(mode, PortModeResult):
        raise TypeError(
            "characteristic_impedance needs a gfem.PortModeResult, got "
            f"{type(mode).__name__}."
        )
    check_fixed(voltage_path, "voltage_path must be fixed numbers")
    path = np.asarray(voltage_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(
            "voltage_path must be two or more points (m), each a 3-vector, got "
            f"shape {path.shape}."
        )
    if not np.isfinite(path).all():
        raise ValueError("voltage_path must hold finite numbers.")
    field = mode.e_field
    if field is None:
        raise ValueError(
            f"The port mode of {mode.port!r} carries no field: the driven result "
            "was computed inside jax.jit, jax.grad or jax.vmap, where a wave "
            "port's mode field is not available. Port modes are constants of the "
            "solve: solve gfem.PortMode for the port outside the transformation."
        )
    volts = path_voltage(
        field._space,  # noqa: SLF001
        field._basis,  # noqa: SLF001
        field._coords,  # noqa: SLF001
        path,
        mode.port,
    )
    return jnp.asarray(np.abs(volts) ** 2 / 2.0)


# --- power flow --------------------------------------------------------------------


def power_flow(
    field: Field,
    side: str | Sequence[str] | Callable[[np.ndarray], np.ndarray],
    *,
    model: Model,
) -> jax.Array:
    """Net time-averaged power leaving ``side`` through its boundary, W.

    ``side`` is a set of elements: a volume region, several (their union),
    or a predicate ``inside(centroids) -> bool`` on the ``(n_tetrahedra,
    3)`` element centroids in m, e.g. ``lambda c: c[:, 0] < x0`` for the
    power flowing in +x across ``x = x0``. Its surface is made of the
    element faces between the selected elements of the field's support and
    the unselected ones (a staircase where a plane cuts through elements);
    faces on the boundary of the solve domain are not on it, since what
    crosses them is counted as absorbed inside ``side``. Positive means net
    power leaves ``side``. A driven field is the peak phasor for 1 W
    incident at its excited port (`gfem.conventions`), so the number reads
    as a fraction of the incident power.

    The power comes from the residual of the discrete equations on the
    outside element layer, not from an integral of ``1/2 Re(E x H*)``, so
    it is exact for the solved field (to the solver's residual) and the
    same through every surface that encloses the same ports and losses.
    In lossy media it is the power entering ``side`` through its ports,
    minus what its elements and faces absorb (dielectric loss,
    conductivity, `Conductor` and `Impedance` sheets, its absorbing faces).
    A face term belongs to ``side`` only when every solve-domain element
    beside it does, so a sheet lying on the surface counts as outside.
    A side may cross or touch a `gfem.Periodic` pair.

    A field of an adaptive sweep (`gfem.ReducedFields`) is exact at the
    solved frequencies and accurate to the sweep's tolerance in between.

    Differentiable through the field: inside `jax.grad`, `jax.jit` or
    `jax.vmap` the kept field of a solve (lumped or wave ports) carries the
    gradient, so absorbed, radiated and transmitted power are objectives.
    The model's leaves are read only in the element layer just outside
    ``side``; a traced material field there has no gradient here and
    raises, and any other leaf gets an exact zero from this measurement
    (its influence arrives through the field).

    Args:
        field: A driven electric field (quantity ``"E"``, in H(curl),
            complex, with a real ``omega``), e.g. ``result.e_field(f)``.
        side: The volume region name(s), or the centroid predicate.
        model: The model the field was solved with: the materials and
            boundaries the residual reads.

    Returns:
        A 0-d float64 array, W.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, ``model`` is not a
            `gfem.Model`, or ``side`` is neither names nor a callable.
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface region; the predicate does not
            return one boolean per tetrahedron; ``side`` selects no element
            of the support, or all of them; the field is not a driven
            electric field, or was not solved on ``model``; or the
            surface is within one element of a port's surface.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: While `jax.grad` traces it: a material
            field with an element in the element layer just outside
            ``side`` (or beside a face that layer's terms read), or the fill
            of a sheet with a face there, is differentiated (move the
            surface one element away from the design region, or wrap the
            value in `jax.lax.stop_gradient`); the node coordinates or the
            field's ``omega`` are differentiated; the field is one of an
            adaptive sweep (`gfem.ReducedFields`) and is differentiated. And
            whenever the node coordinates are traced and ``side`` is a
            predicate: name its regions.
    """
    from gfem._field import Field
    from gfem.model import Model

    if not isinstance(field, Field):
        raise TypeError(f"power_flow needs a gfem.Field, got {type(field).__name__}.")
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("power_flow")
    if not isinstance(model, Model):
        raise TypeError(
            f"power_flow needs model= a gfem.Model, got {type(model).__name__}."
        )
    names = _side_names(side)
    _check_driven(field)
    recorder = _objective.recording()
    if recorder is not None:
        return _recorded_power_flow(recorder, field, names, model)
    if not (is_traced(field) or is_traced(model)):
        engine = importlib.import_module("gfem._engine.power_flow")
        return jnp.asarray(engine.power_flow(field, names, model), dtype=float)
    if callable(names) and is_traced((field.mesh.nodes, model.mesh.nodes)):
        raise NonDifferentiableError(
            "power_flow's side is a predicate on element centroids, which needs "
            "the node coordinates as numbers, but they are computed inside "
            "jax.grad, jax.jit or jax.vmap here: name the side's volume regions "
            "instead, so that its surface is known before the transformation runs."
        )
    from gfem._power_flow import power_flow as traced_power_flow

    return traced_power_flow(field, names, model)


def _recorded_power_flow(
    recorder: _objective.Recorder,
    field: Field,
    names: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> jax.Array:
    """`power_flow` while an objective is classified: one recorded output.

    Its side is checked as the absorbed-power estimate needs it (ADR 0143,
    ticket 11's raise table): named regions, no port surface in or on it,
    only volume losses, isotropic non-law materials.
    """
    label = recorder.label(field, "power_flow")
    if callable(names):
        raise MeshRefinementUnsupportedError(
            "power_flow side predicate",
            "The objective calls power_flow on a centroid predicate: its "
            "staircase surface changes as the mesh is refined, so the power "
            "it selects drifts with the mesh, an error the estimate does not "
            "describe. Name the side's volume regions.",
        )
    if recorder.discover:
        _check_estimated_side(names, model)
    concrete = recorder.field(label)
    return recorder.output(
        f"power_flow|{label}|{'+'.join(names)}",
        (),
        jnp.float64,
        lambda: power_flow(concrete, list(names), model=model),
        measurement="power_flow",
        label=label,
        side=names,
    )


def _check_estimated_side(names: tuple[str, ...], model: Model) -> None:
    """Raise for a `power_flow` side the absorbed-power estimate has no case for."""
    from gfem._engine import dispersion
    from gfem._engine.coefficients import anisotropic_regions
    from gfem.model import Absorbing, Conductor, Impedance, surfaces_of

    mesh = model.mesh
    tetrahedra = np.concatenate([mesh.elements(n) for n in names])
    nodes = np.unique(np.asarray(mesh.tetrahedra)[tetrahedra])

    def touches(surface: str) -> bool:
        triangles = np.asarray(mesh.triangles)[mesh.elements(surface)]
        return bool(np.isin(triangles, nodes).any())

    for port in model.ports:
        if any(touches(s) for s in surfaces_of(port)):
            raise MeshRefinementUnsupportedError(
                "power through a port",
                f"The objective's power_flow side touches port {port.name!r}: "
                "the power through a port is |S|^2, so write the objective on "
                "result.s.",
            )
    for boundary in model.boundaries:
        if isinstance(boundary, (Impedance, Conductor, Absorbing)) and any(
            touches(s) for s in surfaces_of(boundary)
        ):
            raise MeshRefinementUnsupportedError(
                "surface loss in power_flow",
                "The objective's power_flow side holds the lossy surface "
                f"{boundary.surface!r}, whose absorbed power has no estimate: "
                "use gfem.FieldEstimator().",
            )
    laws = {lr.region for lr in dispersion.law_regions(model)}
    odd = [n for n in names if n in laws] + list(anisotropic_regions(model, names))
    if odd:
        raise MeshRefinementUnsupportedError(
            "anisotropic or dispersive power_flow",
            "The objective's power_flow side holds the anisotropic or "
            f"dispersive region {odd[0]!r}, where the absorbed-power estimate "
            "was not verified: use gfem.FieldEstimator().",
        )


# --- electric energy ---------------------------------------------------------------


def electric_energy(field: Field, *, model: Model) -> dict[str, jax.Array]:
    """Time-averaged electric energy of a field in each volume region, J.

    ``W_e[r] = 1/4 eps0 ∫_r E* . Re(eps_r) E dV`` over the region's elements
    in the solve domain, with the full real permittivity tensor in an
    anisotropic medium. In a region with a `gfem.PermittivityLaw` the
    weight is the Brillouin one, ``Re d(omega eps)/d omega`` at the field's
    real frequency (the time-averaged electric energy of a dispersive
    medium, exact for a lossless law and the weak-loss value otherwise).
    The field is a peak phasor, hence the 1/4. A driven field is the one for
    1 W incident at its excited port
    (`gfem.conventions`), so in a region with one loss tangent the power
    dielectric loss absorbs is ``2 * omega * tan_delta * W_e[r]`` W, the
    number `power_flow` gives for that region when nothing else in it
    absorbs.

    A mode field of an `gfem.EigenmodeResult` is accepted too. The mode
    normalization counts ``1/2 Re ∫ D* . E``, the total energy of a lossless
    mode and twice this time average: ``2 * W_e[r]`` equals
    ``bulk_participation[r] * gfem.conventions.MODE_ENERGY`` in a region
    without a permittivity law (the mode energy weighs a law region with its
    energy permittivity instead, see `gfem.conventions`).

    The integral is exact for the solved field and the model's element-wise
    permittivity. A field of an adaptive sweep (`gfem.ReducedFields`) is
    exact at the solved frequencies and accurate to the sweep's tolerance in
    between. Not differentiable in this version.

    Args:
        field: A time-harmonic electric field (quantity ``"E"``, in H(curl),
            with an ``omega``), e.g. ``result.e_field(f)``.
        model: The model the field was solved with: the permittivities and
            regions the integral reads.

    Returns:
        A 0-d float64 array per volume region with elements in the solve
        domain, by region name, J.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, or ``model`` is not a
            `gfem.Model`.
        ValueError: The field is not a time-harmonic electric field (another
            quantity, or static), or was not solved on ``model``; or a
            permittivity law gives ``Re d(omega eps)/d omega <= 0`` in its
            region at the field's frequency (loss dominates there: a metal
            below its damping frequency, a Lorentz pole inside its
            linewidth).
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: ``field`` or ``model`` is computed inside
            `jax.grad`, `jax.jit` or `jax.vmap` from their arguments; call
            `electric_energy` outside them.
    """
    from gfem._field import Field
    from gfem.model import Model

    if not isinstance(field, Field):
        raise TypeError(
            f"electric_energy needs a gfem.Field, got {type(field).__name__}."
        )
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("electric_energy")
    if not isinstance(model, Model):
        raise TypeError(
            f"electric_energy needs model= a gfem.Model, got {type(model).__name__}."
        )
    from gfem._engine.dispersion import refuse_blends

    if _objective.recording() is not None:
        raise MeshRefinementUnsupportedError(
            "electric_energy",
            "The objective reads gfem.electric_energy, which has no field "
            "cotangent, so its error is not estimated. Read the energy of an "
            "electrostatic result from result.energy.",
        )
    refuse_blends(model, "electric_energy")
    if is_traced(field) or is_traced(model):
        raise NonDifferentiableError(
            "gfem.electric_energy is not differentiable in this version: its "
            "field and model must be fixed values, not computed inside jax.grad, "
            "jax.jit or jax.vmap. Call it outside the function JAX transforms."
        )
    _check_time_harmonic(
        field,
        "electric_energy",
        static="an electrostatic result has its energy per region in result.energy",
    )
    engine = importlib.import_module("gfem._engine.energy")
    return {
        region: jnp.asarray(energy, dtype=float)
        for region, energy in engine.electric_energy(field, model).items()
    }


def _side_names(
    side: Any,
) -> tuple[str, ...] | Callable[[np.ndarray], np.ndarray]:
    """``side`` as a predicate or a tuple of region names.

    Raises:
        TypeError: ``side`` is neither.
    """
    if isinstance(side, str):
        return (side,)
    if callable(side):
        return side
    if isinstance(side, Sequence) and side and all(isinstance(s, str) for s in side):
        return tuple(side)
    raise TypeError(
        "power_flow's side must be a volume region name, a sequence of names or "
        f"a predicate on element centroids, got {side!r}."
    )


def _check_time_harmonic(field: Field, who: str, *, static: str) -> None:
    """Raise unless ``field`` is a time-harmonic electric field.

    ``who`` names the measurement in the messages, ``static`` is the remedy
    for a static field.

    Raises:
        ValueError: Another quantity or space, or no ``omega``.
    """
    if field.quantity != "E" or field.space != "hcurl":
        raise ValueError(
            f"{who} needs an electric field (quantity 'E' in 'hcurl'), got "
            f"quantity {field.quantity!r} in {field.space!r}: pass the E field "
            "the B field or potential came from."
        )
    if field.omega is None:
        raise ValueError(
            f"{who} needs a time-harmonic field; this one is static (omega is "
            f"None): {static}."
        )


def _check_driven(field: Field) -> None:
    """Raise unless ``field`` is a driven electric field.

    Raises:
        ValueError: Another quantity or space, real coefficients, or an
            ``omega`` that is missing or complex.
    """
    _check_time_harmonic(field, "power_flow", static="solve a gfem.Driven problem")
    if jnp.iscomplexobj(field.omega):
        raise ValueError(
            "power_flow needs a driven field with a real omega; eigenmode fields "
            "have no time-averaged power flow (complex omega, arbitrary "
            "normalization)."
        )
    if not field.is_complex:
        raise ValueError(
            "power_flow needs a complex phasor field, got real coefficients: "
            "solve a gfem.Driven problem."
        )
