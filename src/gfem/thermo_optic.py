r"""Temperature in optics: the first-order thermo-optic shift of a wave port's mode.

A material's ``thermo_optic_coefficient`` (:math:`dn/dT`, 1/K) acts on the
temperature rise (the temperature above the source-free temperature, a
`gfem.SteadyThermalResult.rise`) about the index of the material's ``eps_r``
(ADR 0160). Two ways read it:

- `thermo_optic_shift`: the change of a wave port mode's effective index to
  first order in the rise, from one mode solve (ADR 0174):

  .. math::

      \delta n_\mathrm{eff} = \frac{c \varepsilon_0}{4} \sum_e 2 n_{0,e}
          (dn/dT)_e\, \Delta T_e \int_e |E_0|^2 \, dA

  over the triangles of the port's section, :math:`E_0` the mode at 1 W and
  :math:`\Delta T_e` the mean rise over the tetrahedron behind each
  triangle. It is linear in the rise and in :math:`dn/dT`, and its phase per
  length is :math:`k_0\, \delta n_\mathrm{eff}`.
- `gfem.Model.with_temperature_rise`: the model with
  :math:`\varepsilon_r = (n_0 + (dn/dT)\, \Delta T_e)^2` per element, for any
  optical solve (a `gfem.PortMode` re-solve checks the first order).

The rise may live on another mesh than the optical model (a window around
one core, in the same coordinates): each element's rise is then the
quadrature mean of the rise at points located in its mesh (ADR 0176).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._adjoint.measure import measure
from gfem._checks import is_traced
from gfem._field import Field
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._pytree import register_fields
from gfem.thermal import Trace

if TYPE_CHECKING:
    from gfem.mesh import Mesh
    from gfem.model import Model

__all__ = ["RoutePhase", "ThermoOpticShift", "route_phase", "thermo_optic_shift"]


def _section(model: Model, port: Any) -> tuple[np.ndarray, np.ndarray]:
    """The gfem triangles of a wave port's section and the tetrahedra behind them.

    The triangles of the port's surface with the electromagnetic solve
    domain on exactly one side, as the port mode problem takes them.
    """
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    tag = mesh.region(port.surface).tag
    nb = np.asarray(mesh.triangle_neighbours)
    faces = np.flatnonzero(np.asarray(mesh.triangle_tags) == tag)
    side = (nb[faces] >= 0) & active[np.maximum(nb[faces], 0)]
    one = side.sum(axis=1) == 1
    faces, side = faces[one], side[one]
    tets = np.where(side[:, 0], nb[faces, 0], nb[faces, 1])
    return faces, tets


def _region_rows(mesh: Mesh, tets: np.ndarray) -> dict[str, np.ndarray]:
    """The positions in ``tets`` of each volume region's tetrahedra."""
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    tags = np.asarray(mesh.tetrahedron_tags)[tets]
    return {names[int(t)]: np.flatnonzero(tags == t) for t in np.unique(tags)}


def _values(value: Any, pos: np.ndarray) -> Any:
    """A material value at element positions ``pos`` of its region."""
    if is_traced(value):
        arr = jnp.asarray(value)
        return jnp.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]
    arr = np.asarray(value)
    return np.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]


def _lossy(material: Material | MaterialField, pos: np.ndarray) -> str | None:
    """Which loss a region's material has on elements ``pos``, or ``None``."""
    if isinstance(material, Material):
        if isinstance(material.eps_r, PermittivityLaw):
            return "a permittivity law"
        eps = np.asarray(material.eps_r, complex)
        tan, sigma = material.tan_delta, material.conductivity
    else:
        if material.blend_ends is not None:
            return "a permittivity law (a density blend's end)"
        eps, tan, sigma = material.eps_r, material.tan_delta, material.conductivity
        eps = None if eps is None or is_traced(eps) else _values(eps, pos)
        tan = None if tan is None or is_traced(tan) else _values(tan, pos)
        sigma = None if sigma is None or is_traced(sigma) else _values(sigma, pos)
    if eps is not None and np.any(np.asarray(eps).imag != 0):
        return "a complex eps_r"
    if tan is not None and np.any(np.asarray(tan) != 0):
        return "a nonzero tan_delta"
    if sigma is not None and np.any(np.asarray(sigma) != 0):
        return "a nonzero conductivity"
    return None


def _coefficients(
    model: Model, port: str, tets: np.ndarray, owner: str = "thermo_optic_shift"
) -> jax.Array:
    """dn/dT of each tetrahedron behind the section (0 where none is given).

    Raises:
        ValueError: No region behind the section has a coefficient.
        NotImplementedError: The section is lossy, or a region with a
            coefficient is anisotropic.
    """
    mesh = model.mesh
    out = jnp.zeros(len(tets))
    given = False
    for region, rows in _region_rows(mesh, tets).items():
        material = model.materials[region]
        pos = np.searchsorted(mesh.elements(region), tets[rows])
        loss = _lossy(material, pos)
        if loss is not None:
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s "
                f"section has {loss}. The first-order shift reads |E|^2, which "
                "holds for a lossless section only: give the section's regions a "
                "real eps_r with no tan_delta or conductivity (an optical window "
                "or EM-excluded metal keeps the loss out of the section)."
            )
        coefficient = material.thermo_optic_coefficient
        if coefficient is None:
            continue
        anisotropic = (
            not material.isotropic
            if isinstance(material, Material)
            else material.axes is not None
        )
        if anisotropic:
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s "
                "section is anisotropic and has a thermo_optic_coefficient: a "
                "scalar dn/dT on a permittivity tensor is not defined in this "
                "version. Give the region an isotropic eps_r."
            )
        given = True
        out = out.at[rows].set(_values(coefficient, pos))
    if not given:
        regions = ", ".join(repr(r) for r in _region_rows(mesh, tets))
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section has a "
            f"thermo_optic_coefficient (its regions are {regions}), so the shift "
            "is zero. Set thermo_optic_coefficient on the core's material."
        )
    return out


def _element_rise(rise: Any, mesh: Mesh, tets: np.ndarray, owner: str) -> jax.Array:
    """The mean of ``rise`` over each of ``tets`` of ``mesh``.

    On the rise's own mesh, the exact element mean (`gfem.element_mean`'s
    kernel). On another mesh, the quadrature mean of the rise at points of
    each tetrahedron located in the rise's mesh, exact to the rise's order
    when the meshes coincide. Linear in the rise's coefficients.

    Raises:
        TypeError: ``rise`` is not a scalar `gfem.Field`.
        ValueError: An element is outside the rise's support or mesh.
    """
    from gfem import _probe
    from gfem.measurements import _element_means

    if not isinstance(rise, Field) or _probe.COMPONENTS[rise.space] != 1:
        raise TypeError(
            f"{owner}: the rise must be a scalar gfem.Field (a thermal result's "
            f"rise), got {type(rise).__name__}."
        )
    if rise.mesh._topology is mesh._topology:  # noqa: SLF001
        return _element_means(
            rise,
            tets,
            f"{owner}: {{n}} of its elements are outside the rise's support (the "
            "thermal domain), where the rise has no value.",
        )
    bary, weights = _probe.engine().tetrahedron_rule(rise.order)
    corners = np.asarray(mesh.nodes, float)[np.asarray(mesh.tetrahedra)[tets]]
    points = np.einsum("qk,nkd->nqd", bary, corners).reshape(-1, 3)
    try:
        values = rise.at(points)
    except ValueError as err:
        raise ValueError(f"{owner}: {err}") from err
    return jnp.reshape(values, (len(tets), len(weights))) @ jnp.asarray(weights)


@dataclasses.dataclass(frozen=True, eq=False)
class _Sampling:
    """Where a shift trace reads the rise: tetrahedra of the optical mesh."""

    mesh: Any
    tetrahedra: np.ndarray
    port: str


@dataclasses.dataclass(frozen=True, eq=False)
class _ShiftTrace(Trace):
    """The thermo-optic phase per length of a section, rad/m, on the rise."""

    _weights: jax.Array
    _sampling: _Sampling
    unit: str = "rad/m"

    def _sample(self, temperature: Field | None, rise: Field) -> jax.Array:
        """``k0 sum_t w_t (dn/dT)_t rise_t``, the phase per length."""
        del temperature
        s = self._sampling
        owner = f"the thermo-optic shift of wave port {s.port!r}"
        return jnp.sum(self._weights * _element_rise(rise, s.mesh, s.tetrahedra, owner))

    def __repr__(self) -> str:
        return (
            f"Trace(thermo-optic phase per length of wave port "
            f"{self._sampling.port!r}, rad/m)"
        )


register_fields(_ShiftTrace, ("_weights",))


@dataclasses.dataclass(frozen=True, eq=False)
class ThermoOpticShift:
    r"""The first-order thermo-optic shift of a wave port's mode.

    The mode is solved once on the optical model and held constant: the
    shift is exactly linear in the rise and in the thermo-optic coefficient
    (ADR 0174), and differentiable with respect to both (through the rise,
    the thermal inputs).

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: The frequency, Hz.
        n_eff: The unperturbed effective index (real; the section is
            lossless), a constant.
        delta_n_eff: The change of the effective index for the rise given,
            ``None`` without one.
        phase_per_length: ``k0 delta_n_eff``, rad/m, ``None`` without a
            rise: the extra phase delay per unit length of guide, positive
            for a positive coefficient and a positive rise. A transmission
            through length ``L`` turns its argument by minus
            ``phase_per_length * L`` (`gfem.conventions`). With the heater's
            power per length :math:`P'`, :math:`P_\pi = \pi P' /`
            ``phase_per_length``.
    """

    port: str
    mode: int
    frequency: float
    n_eff: jax.Array
    delta_n_eff: jax.Array | None
    phase_per_length: jax.Array | None
    _trace: _ShiftTrace

    def trace(self) -> Trace:
        """The phase per length as a trace of a `gfem.TransientThermal` solve.

        A linear functional of the rise in rad/m (``phase_per_length`` at
        each step); its weights depend on the mode and dn/dT only, not on
        the rise this shift was given.
        """
        return self._trace

    def __repr__(self) -> str:
        try:
            n = float(np.asarray(self.n_eff))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"ThermoOpticShift(port {self.port!r}, mode {self.mode})"
        head = (
            f"ThermoOpticShift(port {self.port!r}, mode {self.mode}, "
            f"{self.frequency:.6g} Hz, n_eff {n:.6f}"
        )
        if self.phase_per_length is None:
            return head + ", no rise)"
        try:
            dn = float(np.asarray(self.delta_n_eff))
            phi = float(np.asarray(self.phase_per_length))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return head + f", delta_n_eff {dn:.6g}, phase_per_length {phi:.6g} rad/m)"


register_fields(
    ThermoOpticShift, ("n_eff", "delta_n_eff", "phase_per_length", "_trace")
)


def _frequency(frequency: Any, owner: str = "thermo_optic_shift") -> float:
    if np.ndim(frequency) != 0:
        raise ValueError(
            f"{owner} takes one frequency, got "
            f"{np.size(frequency)} values: call it once per frequency in a loop."
        )
    if is_traced(frequency):
        raise TypeError(
            f"{owner}: the frequency is a fixed number, not a differentiable input."
        )
    f = float(frequency)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{owner}: frequency must be > 0 Hz, got {f}.")
    return f


def _wave_port(model: Model, port: str) -> Any:
    from gfem._sweep.port_mode import _wave_port as find

    return find(model, port)


def _check_model(model: Any, owner: str) -> None:
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(f"{owner} needs a gfem.Model, got {type(model).__name__}.")


def _mode_weights(
    model: Model, port: str, frequency: float, mode: int | None, faces: np.ndarray
) -> tuple[int, jax.Array, jax.Array, float]:
    """The port's mode on ``model`` and its first-order weight per section triangle.

    Returns:
        The mode number, ``n_eff`` (a constant), the weights ``w_t`` (``(n_tri,)``,
        in the order of ``faces``; ``delta_n_eff = sum_t w_t (dn/dT)_t rise_t``)
        and ``k0`` (1/m).
    """
    from gfem._adjoint import solve_vjp
    from gfem._engine.thermo_optic import ShiftWeightsRule
    from gfem.options import SolverOptions
    from gfem.problems import PortMode

    problem = PortMode(port, frequency, mode=mode)
    rule = ShiftWeightsRule(model, problem, SolverOptions(), faces)
    if is_traced(model):
        outputs = solve_vjp(rule, model, problem)
    else:
        outputs, _ = rule.forward(model, keep=False)
    k0 = 2 * math.pi * frequency / scipy.constants.c
    gamma = jnp.asarray(outputs["gamma"])[0]
    n_eff = jnp.real(gamma / (1j * k0))
    return rule.mode, n_eff, jnp.asarray(outputs["weights"]), k0


def thermo_optic_shift(
    model: Model,
    rise: Field | None,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
) -> ThermoOpticShift:
    r"""The first-order thermo-optic shift of a wave port's mode.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one frequency)
    and gives the change of its effective index caused by ``rise`` through
    each element's thermo-optic coefficient, to first order:

    .. math::

        \delta n_\mathrm{eff} = \frac{c \varepsilon_0}{4} \sum_e 2 n_{0,e}
            (dn/dT)_e\, \Delta T_e \int_e |E_0|^2 \, dA

    :math:`E_0` is the mode at 1 W, :math:`e` runs over the triangles of the
    port's section, :math:`n_{0,e}` and :math:`(dn/dT)_e` are the material's
    of the tetrahedron behind it, and :math:`\Delta T_e` is the mean rise
    over that tetrahedron. Elements without a coefficient contribute nothing.

    ``model`` is the optical model, which carries ``eps_r`` and the
    coefficient: the thermal model with the substrate and heater excluded
    from electromagnetic solves (``excluded={"em": [...]}``), or a separate
    window model around one core in the same coordinates. The mode is the
    port's (``mode=None``) or the one given, as `gfem.PortMode` picks it;
    exclusions or a window make the core's mode the first.

    Under `jax.grad` the shift differentiates with respect to the rise (the
    thermal inputs behind it) and the thermo-optic coefficient. The mode is
    held constant, so the optical model's ``eps_r``, ``tan_delta``,
    ``mu_r``, ``conductivity`` and node coordinates raise
    `gfem.NonDifferentiableError`.

    Example:
        P_pi of a heater driven at power per length ``P1`` (W/m)::

            steady = gfem.solve(model, gfem.SteadyThermal(heat))
            shift = gfem.thermo_optic_shift(
                model, steady.rise, port="p", frequency=299792458.0 / 1.55e-6
            )
            p_pi = np.pi * steady.power_per_length / shift.phase_per_length

    Args:
        model: The optical model.
        rise: The temperature rise, a scalar `gfem.Field` on any mesh that
            holds the section's elements (``SteadyThermalResult.rise``), or
            ``None`` for the mode and `ThermoOpticShift.trace` only.
        port: The wave port's name.
        frequency: One frequency, Hz.
        mode: Which mode (1 = the most propagating), or ``None`` for the
            wave port's own.

    Returns:
        The shift: ``n_eff``, ``delta_n_eff``, ``phase_per_length`` and
        ``trace()``.

    Raises:
        ValueError: ``port`` is not a wave port of the model, a sequence of
            frequencies is given, no element of the section has a
            thermo_optic_coefficient, or a section element lies outside the
            rise's mesh or support (the message names the point).
        NotImplementedError: The section is lossy (a complex ``eps_r``, a
            ``tan_delta`` or ``conductivity`` other than 0, or a permittivity
            law), or a region with a coefficient is anisotropic.
        TypeError: ``rise`` is not a scalar `gfem.Field`.
    """
    _check_model(model, "thermo_optic_shift")
    f = _frequency(frequency)
    faces, tets = _section(model, _wave_port(model, port))
    coefficients = _coefficients(model, port, tets)
    owner = f"thermo_optic_shift (wave port {port!r})"
    if rise is not None:
        rise_e = _element_rise(rise, model.mesh, tets, owner)
    found, n_eff, weights, k0 = _mode_weights(model, port, f, mode, faces)
    weights = weights * coefficients
    trace = _ShiftTrace(
        _weights=k0 * weights,
        _sampling=_Sampling(model.mesh, tets, port),
    )
    delta = phase = None
    if rise is not None:
        delta = jnp.sum(weights * rise_e)
        phase = k0 * delta
    return ThermoOpticShift(
        port=port,
        mode=found,
        frequency=f,
        n_eff=n_eff,
        delta_n_eff=delta,
        phase_per_length=phase,
        _trace=trace,
    )


def with_temperature_rise(model: Model, rise: Field) -> Model:
    """`gfem.Model.with_temperature_rise` (see there)."""
    if rise is None:
        raise TypeError("Model.with_temperature_rise needs a rise Field, got None.")
    mesh = model.mesh
    active = np.asarray(model._domain.active)  # noqa: SLF001
    materials = dict(model.materials)
    for region, material in model.materials.items():
        coefficient = material.thermo_optic_coefficient
        if coefficient is None:
            continue
        owner = f"Model.with_temperature_rise: region {region!r}"
        tets = np.asarray(mesh.elements(region))
        read = active[tets]
        if not read.any():  # not in any electromagnetic solve
            continue
        if isinstance(material, Material):
            if isinstance(material.eps_r, PermittivityLaw) or not material.isotropic:
                raise NotImplementedError(
                    f"{owner} has a thermo_optic_coefficient and "
                    + (
                        "a permittivity law"
                        if isinstance(material.eps_r, PermittivityLaw)
                        else "an anisotropic material"
                    )
                    + ": eps_r = (n0 + dn/dT rise)^2 needs one isotropic eps_r."
                )
            field = MaterialField.uniform(region, material)
        else:
            if material.axes is not None or material.blend_ends is not None:
                raise NotImplementedError(
                    f"{owner} has a thermo_optic_coefficient on an anisotropic or "
                    "blended material field: eps_r = (n0 + dn/dT rise)^2 needs one "
                    "isotropic eps_r."
                )
            field = material
        rise_e = (
            jnp.zeros(len(tets))
            .at[np.flatnonzero(read)]
            .set(_element_rise(rise, mesh, tets[read], owner))
        )
        eps = 1.0 if field.eps_r is None else field.eps_r
        n0 = jnp.sqrt(jnp.asarray(eps))
        eps_r = (n0 + jnp.asarray(coefficient) * rise_e) ** 2
        materials[region] = dataclasses.replace(field, eps_r=eps_r, region=region)
    return model.replace(materials=materials)


# --- route phase (ADR 0183) -------------------------------------------------

#: Window points located per block of stations: bounds the location's memory.
_POINTS_PER_BLOCK = 2_000_000


def _stations(
    name: str, route: np.ndarray, spacing: float, up: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The stations of one route: evenly spaced in arc length, ends included.

    The tangent is the segment's direction (at a vertex a station falls on,
    the mean of its two segments'); left is ``up x tangent`` and the
    section's y the unit vector of ``up`` square to the tangent.

    Returns:
        Arc lengths ``(n,)`` (m), points, tangents, lefts and ups ``(n, 3)``.

    Raises:
        ValueError: Fewer than two distinct points, or a segment parallel to
            ``up`` (the message names the route and the segment).
    """
    route = np.asarray(route, float)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    d = np.diff(route, axis=0)
    lengths = np.linalg.norm(d, axis=1)
    keep = lengths > 1e-12 * lengths.sum()
    if not keep.any():
        raise ValueError(
            f"route_phase: route {name!r} needs at least two distinct points."
        )
    segments = np.flatnonzero(keep)
    starts, lengths = route[segments], lengths[keep]
    dirs = d[keep] / lengths[:, None]
    sideways = np.linalg.norm(np.cross(up, dirs), axis=1)
    if (sideways < 1e-9).any():
        k = int(segments[np.flatnonzero(sideways < 1e-9)[0]])
        raise ValueError(
            f"route_phase: route {name!r}: segment {k} (points {k} to {k + 1}) is "
            f"parallel to up {tuple(up.tolist())}, so the window's left (up x "
            "tangent) is not defined there. Give a route that never runs along "
            "up, or another up."
        )
    s_v = np.concatenate(([0.0], np.cumsum(lengths)))
    total = s_v[-1]
    n = max(2, int(np.ceil(total / spacing - 1e-9)) + 1)
    s = np.linspace(0.0, total, n)
    k = np.clip(np.searchsorted(s_v, s, side="right") - 1, 0, len(dirs) - 1)
    points = starts[k] + (s - s_v[k])[:, None] * dirs[k]
    points[-1] = starts[-1] + lengths[-1] * dirs[-1]
    tangent = dirs[k].copy()
    if len(dirs) > 1:  # stations on an interior vertex
        inner = s_v[1:-1]
        near = np.clip(np.searchsorted(inner, s), 0, len(inner) - 1)
        for cand in (near, np.maximum(near - 1, 0)):
            hit = np.abs(s - inner[cand]) <= 1e-12 * total
            m = cand[hit] + 1  # the vertex between segments m - 1 and m
            mean = dirs[m - 1] + dirs[m]
            size = np.linalg.norm(mean, axis=1)
            ok = size > 1e-12
            rows = np.flatnonzero(hit)
            tangent[rows[ok]] = mean[ok] / size[ok, None]
            points[rows] = starts[m]
    left = np.cross(up, tangent)
    size = np.linalg.norm(left, axis=1)
    if (size < 1e-9).any():  # pragma: no cover - a vertex mean along up
        j = int(np.flatnonzero(size < 1e-9)[0])
        raise ValueError(
            f"route_phase: route {name!r}: at arc length {s[j]:.6g} m the route "
            f"runs parallel to up {tuple(up.tolist())}."
        )
    left /= size[:, None]
    return s, points, tangent, left, np.cross(tangent, left)


@dataclasses.dataclass(frozen=True, eq=False)
class _RouteSampler:
    """Where a route phase reads the rise: the window placed at each station.

    The window's section triangles (``corners``, section x and y in m) sit at
    every station with x along ``left`` and y along ``up``. The rise's mean
    over each placed triangle, a fixed linear map of the rise coefficients,
    is located once per rise mesh and layout and kept as a sparse matrix.
    """

    name: str
    arc_length: np.ndarray
    points: np.ndarray
    left: np.ndarray
    up: np.ndarray
    trapezoid: np.ndarray
    corners: np.ndarray
    _cache: dict = dataclasses.field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """``(stations, window triangles)``."""
        return len(self.arc_length), len(self.corners)

    def _placed(self, xy: np.ndarray, j0: int, j1: int) -> np.ndarray:
        """``(points, 3)`` m: section points ``xy`` placed at stations j0..j1."""
        p, l, u = self.points[j0:j1], self.left[j0:j1], self.up[j0:j1]
        x, y = xy[..., 0], xy[..., 1]
        out = (
            p[:, None, None, :]
            + x[None, ..., None] * l[:, None, None, :]
            + y[None, ..., None] * u[:, None, None, :]
        )
        return out.reshape(-1, 3)

    def _outside(self, mesh: Mesh, layout: Any, xy: np.ndarray, j0: int, j1: int):
        """Raise for the first station in j0..j1 whose window leaves the rise."""
        from gfem._engine import probe

        lo, hi = j0, j1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try:
                probe.locate(mesh, layout, self._placed(xy, lo, mid))
            except ValueError:
                hi = mid
            else:
                lo = mid
        try:
            probe.locate(mesh, layout, self._placed(xy, lo, lo + 1))
        except ValueError as err:
            raise ValueError(
                f"route_phase: route {self.name!r}: the window at the station at "
                f"arc length {self.arc_length[lo]:.6g} m samples outside the rise: "
                f"{err}"
            ) from err
        raise RuntimeError("a window point failed to locate")  # pragma: no cover

    def matrices(self, mesh: Mesh, layout: Any) -> tuple[Any, Any]:
        """``(M, total)``: per station and triangle, and trapezoid-summed.

        ``M @ c`` is the rise's mean over each placed triangle, ``(stations *
        triangles,)`` station-major; ``total @ c`` its trapezoid integral over
        the arc length, per triangle (m K).
        """
        import scipy.sparse

        from gfem._engine import probe

        nodes = np.ascontiguousarray(mesh.nodes)
        key = (id(mesh._topology), hash(nodes.tobytes()), layout)  # noqa: SLF001
        found = self._cache.get(key)
        if found is not None and found[0] is mesh._topology:  # noqa: SLF001
            return found[1]
        bary, wq = probe.triangle_rule(2 * max(layout.order, 1))
        xy = np.einsum("qk,tkd->tqd", bary, self.corners)
        n_st, n_tri = self.shape
        q = len(wq)
        block = max(1, _POINTS_PER_BLOCK // (n_tri * q))
        blocks = []
        for j0 in range(0, n_st, block):
            j1 = min(n_st, j0 + block)
            try:
                located = probe.locate(mesh, layout, self._placed(xy, j0, j1))
            except ValueError:
                self._outside(mesh, layout, xy, j0, j1)
                raise  # pragma: no cover
            rows = np.repeat(np.arange((j1 - j0) * n_tri), q)
            weights = np.tile(wq, (j1 - j0) * n_tri)
            p = probe.Probe(mesh, layout, located)
            blocks.append(p.rows(weights, rows, (j1 - j0) * n_tri))
        full = scipy.sparse.vstack(blocks).tocsr()
        summing = scipy.sparse.kron(
            self.trapezoid[None, :], scipy.sparse.identity(n_tri), format="csr"
        )
        made = (full, (summing @ full).tocsr())
        self._cache.clear()  # one rise mesh at a time
        self._cache[key] = (mesh._topology, made)  # noqa: SLF001
        return made

    def sample(self, rise: Any, *, total: bool) -> jax.Array:
        """The placed-triangle means of ``rise`` (``total``: route-integrated)."""
        _check_rise(rise)
        kernel = _RouteKernel(self, rise, total=total)
        inputs = {"coefficients": rise.coefficients, "nodes": rise.mesh.nodes}
        return measure(kernel, inputs)["values"]


def _check_rise(rise: Any) -> None:
    from gfem import _probe

    if not isinstance(rise, Field) or _probe.COMPONENTS[rise.space] != 1:
        raise TypeError(
            "route_phase: the rise must be a scalar gfem.Field (a thermal result's "
            f"rise), got {type(rise).__name__}."
        )
    if rise._communicator is not None:  # noqa: SLF001
        raise NotImplementedError(
            "route_phase: the rise is split across ranks; a route phase reads a "
            "rise solved on one process in this version."
        )


class _RouteKernel:
    """A route's placed-triangle rise means: ``M c`` and its transpose."""

    name = "a route phase"
    pieces = frozenset({"coefficients"})
    comm = None

    def __init__(self, sampler: _RouteSampler, rise: Field, *, total: bool) -> None:
        from gfem import _probe

        self.sampler = sampler
        self.topology = rise.mesh._topology  # noqa: SLF001
        self.layout = _probe.layout(rise)
        self.total = total
        n_st, n_tri = sampler.shape
        n = n_tri if total else n_st * n_tri
        dtype = rise.coefficients.dtype
        self.output_specs = {"values": jax.ShapeDtypeStruct((n,), dtype)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        from gfem import _probe

        return _probe.NODES_REASON if name == "nodes" else None

    def describe(self, name: str) -> str:
        return {"nodes": "the mesh node coordinates (field.mesh.nodes)"}.get(
            name, f"the field {name}"
        )

    def _matrix(self, nodes: np.ndarray) -> Any:
        from gfem.mesh import Mesh

        full, total = self.sampler.matrices(Mesh(nodes, self.topology), self.layout)
        return total if self.total else full

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"values": self._matrix(values["nodes"]) @ values["coefficients"]}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no route phase gradient for {wanted}")
        matrix = self._matrix(values["nodes"])
        return {"coefficients": matrix.T @ np.asarray(cotangents["values"])}


@dataclasses.dataclass(frozen=True, eq=False)
class _RouteTrace(Trace):
    """The thermo-optic phase of one route, rad, on the rise."""

    _weights: jax.Array
    _sampler: _RouteSampler
    unit: str = "rad"

    def _sample(self, temperature: Field | None, rise: Field) -> jax.Array:
        """``sum_t (k0 w_t (dn/dT)_t) int rise_t ds``, the route's phase."""
        del temperature
        return jnp.sum(self._weights * self._sampler.sample(rise, total=True))

    def __repr__(self) -> str:
        return f"Trace(thermo-optic phase of route {self._sampler.name!r}, rad)"


register_fields(_RouteTrace, ("_weights",))


@dataclasses.dataclass(frozen=True, eq=False)
class RoutePhase:
    """The thermo-optic phase along named routes through a 3D rise.

    One optical window's first-order weights (its mode held constant, ADR
    0174) swept along each route: exactly linear in the rise and in the
    window's thermo-optic coefficient, and differentiable with respect to
    both (ADR 0183).

    Attributes:
        port: The window's wave port.
        mode: Which mode (1 = the most propagating).
        frequency: The frequency, Hz.
        n_eff: The window's unperturbed effective index, a constant.
        phase: Per route name, the extra phase delay along the route, rad
            (positive for a positive coefficient and rise); ``None`` without
            a rise. ``P_pi = pi * steady.power / phase[name]``.
        phase_per_length: Per route name, the phase per length at each
            station, rad/m (``None`` without a rise).
        arc_length: Per route name, the stations' arc lengths, m (constants).
        length: Per route name, the route's length, m.
    """

    port: str
    mode: int
    frequency: float
    n_eff: jax.Array
    phase: dict[str, jax.Array] | None
    phase_per_length: dict[str, jax.Array] | None
    arc_length: dict[str, np.ndarray]
    length: dict[str, float]
    _traces: dict[str, _RouteTrace]

    def trace(self, name: str) -> Trace:
        """Route ``name``'s phase as a trace of a `gfem.TransientThermal` solve.

        A linear functional of the rise in rad (``phase[name]`` at each step);
        its weights depend on the mode, dn/dT and the route only.

        Raises:
            KeyError: No route of that name.
        """
        if name not in self._traces:
            raise KeyError(
                f"route_phase has no route {name!r}: its routes are "
                f"{list(self._traces)}."
            )
        return self._traces[name]

    def __repr__(self) -> str:
        names = ", ".join(repr(n) for n in self._traces)
        return (
            f"RoutePhase(port {self.port!r}, mode {self.mode}, "
            f"{self.frequency:.6g} Hz, routes {names})"
        )


register_fields(
    RoutePhase,
    ("n_eff", "phase", "phase_per_length", "arc_length", "length", "_traces"),
)


def _fixed(value: Any, what: str) -> np.ndarray:
    if is_traced(value):
        raise TypeError(
            f"route_phase: {what} is traced: route points, up and spacing are fixed "
            "numbers, not differentiable inputs."
        )
    return np.asarray(value, float)


def _routes(routes: Any) -> dict[str, np.ndarray]:
    if not isinstance(routes, Mapping) or not routes:
        raise TypeError(
            "route_phase: routes must be a non-empty mapping of name to an (N, 3) "
            f"polyline in m, got {type(routes).__name__}."
        )
    out = {}
    for name, points in routes.items():
        if not isinstance(name, str):
            raise TypeError(f"route_phase: routes names must be str, got {name!r}.")
        p = _fixed(points, f"routes[{name!r}]")
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
            raise ValueError(
                f"route_phase: routes[{name!r}] must be an (N, 3) polyline in m with "
                f"N >= 2, got shape {p.shape}."
            )
        if not np.isfinite(p).all():
            raise ValueError(f"route_phase: routes[{name!r}] has non-finite points.")
        out[name] = p
    return out


def _window_corners(model: Model, faces: np.ndarray, port: str) -> np.ndarray:
    """``(n_tri, 3, 2)`` section x, y (m) of the window's section triangles.

    Raises:
        ValueError: The port's surface is not in a plane z = constant.
    """
    mesh = model.mesh
    if is_traced(mesh.nodes):
        raise TypeError(
            "route_phase: the window's node coordinates are traced; they place "
            "the window along the routes and must be fixed numbers."
        )
    xyz = np.asarray(mesh.nodes, float)[np.asarray(mesh.triangles)[faces]]
    extent = np.ptp(xyz.reshape(-1, 3), axis=0)
    if extent[2] > 1e-9 * max(extent[0], extent[1]):
        raise ValueError(
            f"route_phase reads wave port {port!r}'s section in the window's own "
            "x and y (x to the left of travel, y along up, the station at the "
            "origin): its surface must lie in a plane z = constant, as a "
            "cross-section slab's front does."
        )
    return xyz[..., :2]


def route_phase(
    window: Model,
    rise: Field | None,
    *,
    routes: Mapping[str, Any],
    port: str,
    frequency: float,
    up: Any = (0.0, 0.0, 1.0),
    spacing: float = 0.5e-6,
    mode: int | None = None,
) -> RoutePhase:
    """The thermo-optic phase along routes through a 3D temperature rise.

    Solves the mode of an optical window (a model of one isolated guide's
    cross-section with a wave port, ``port``) once, and sweeps its first-order
    weights along each route (ADR 0183)::

        phase = int sum_t k0 w_t (dn/dT)_t rise_t(s) ds

    ``w_t`` are `thermo_optic_shift`'s weights on the window's section
    triangles, the same everywhere (straight-guide weights, also in bends),
    and ``rise_t(s)`` is the quadrature mean of ``rise`` over triangle ``t``
    placed at the station at arc length ``s``. Stations are spaced evenly
    along each polyline, ends included, at most ``spacing`` apart, and the
    integral is the trapezoid rule.

    The window lives in route-local coordinates: its section is its port's
    surface in a plane z = constant, read in its own x and y; at each station
    the section's origin sits on the route, x points to the left of travel
    (``up x tangent``) and y along ``up`` (made square to the route). The
    thermo-optic coefficient is the window's; the 3D model's is not read.

    Under `jax.grad` the phase differentiates with respect to the rise (the
    thermal inputs behind it) and the window's thermo-optic coefficient;
    the window's permittivity, losses, permeability and node coordinates
    raise `gfem.NonDifferentiableError` (the mode is held constant).

    Example:
        P_pi of a spiral and the phase of one pass::

            steady = gfem.solve(thermal, gfem.SteadyThermal(heat))
            route = gfem.route_phase(
                window,
                steady.rise,
                routes={"spiral": centreline, "pass": cut},
                port="core",
                frequency=299792458.0 / 1.55e-6,
            )
            p_pi = np.pi * steady.power / route.phase["spiral"]

    Args:
        window: The optical window model.
        rise: The temperature rise, a scalar `gfem.Field` on the 3D thermal
            mesh (``SteadyThermalResult.rise``), or ``None`` for the mode, the
            stations and `RoutePhase.trace` only.
        routes: Name to an ``(N, 3)`` polyline in m (the guide's reference
            line, at least two distinct points). Fixed numbers.
        port: The window's wave port.
        frequency: One frequency, Hz.
        up: The wafer normal (any length), a fixed vector.
        spacing: The largest station spacing, m.
        mode: Which mode (1 = the most propagating), or ``None`` for the
            wave port's own.

    Returns:
        The phase per route: ``phase``, ``phase_per_length``,
        ``arc_length``, ``length``, ``n_eff`` and ``trace(name)``.

    Raises:
        TypeError: ``routes``, ``up`` or ``spacing`` is traced or malformed,
            or ``rise`` is not a scalar `gfem.Field`.
        ValueError: A route has fewer than two distinct points or a segment
            parallel to ``up`` (named), a placed window samples outside the
            rise's mesh or support (the route and the station's arc length
            named), the port's surface is not in a plane z = constant, no
            section element has a thermo_optic_coefficient, or a sequence of
            frequencies is given.
        NotImplementedError: The window's section is lossy, a region with a
            coefficient is anisotropic, or the rise is split across ranks.
    """
    _check_model(window, "route_phase")
    f = _frequency(frequency, "route_phase")
    paths = _routes(routes)
    up_v = _fixed(up, "up")
    if up_v.shape != (3,) or not np.isfinite(up_v).all() or not up_v.any():
        raise ValueError(f"route_phase: up must be a nonzero 3-vector, got {up!r}.")
    step = float(_fixed(spacing, "spacing"))
    if not (math.isfinite(step) and step > 0):
        raise ValueError(f"route_phase: spacing must be > 0 m, got {spacing!r}.")
    if rise is not None:
        _check_rise(rise)
    faces, tets = _section(window, _wave_port(window, port))
    coefficients = _coefficients(window, port, tets, "route_phase")
    # first: a perturbed window input raises by name before its nodes are read
    found, n_eff, weights, k0 = _mode_weights(window, port, f, mode, faces)
    corners = _window_corners(window, faces, port)
    samplers = {}
    for name, points in paths.items():
        s, p, _, left, upward = _stations(name, points, step, up_v)
        h = np.diff(s)
        trapezoid = np.zeros(len(s))
        trapezoid[:-1] += h / 2
        trapezoid[1:] += h / 2
        samplers[name] = _RouteSampler(name, s, p, left, upward, trapezoid, corners)
    weights = k0 * weights * coefficients
    phase = per_length = None
    if rise is not None:
        phase, per_length = {}, {}
        for name, sampler in samplers.items():
            rho = jnp.reshape(sampler.sample(rise, total=False), sampler.shape)
            per_length[name] = rho @ weights
            phase[name] = jnp.sum(jnp.asarray(sampler.trapezoid) * per_length[name])
    return RoutePhase(
        port=port,
        mode=found,
        frequency=f,
        n_eff=n_eff,
        phase=phase,
        phase_per_length=per_length,
        arc_length={n: sp.arc_length for n, sp in samplers.items()},
        length={n: float(sp.arc_length[-1]) for n, sp in samplers.items()},
        _traces={
            n: _RouteTrace(_weights=weights, _sampler=sp) for n, sp in samplers.items()
        },
    )
