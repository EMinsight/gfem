r"""`gfem.far_field`: the far-field amplitude and the IEEE 145 radiation outputs.

The amplitude is the reciprocity integral of `gfem._engine.far_field`
over the element layer just outside a side, a linear function of the field
coefficients and a closed-form function of the layer's node coordinates.
The radiated power is `gfem.power_flow`'s residual through the same side,
so the directivity integrates to :math:`4\pi` to the discretization of
:math:`U` alone. One host kernel computes both, behind gfem's gradient
wrapper (`gfem._adjoint.measure`):

- the field coefficients: differentiable (both outputs), unless the field
  is formed from an adaptive sweep's frozen basis;
- the field's node coordinates: differentiable for the amplitude; the
  radiated power has no node gradient in this version and raises once it
  carries a cotangent;
- ``omega``: read, not differentiable;
- the model's leaves: as `gfem.power_flow` reads them (a design in its
  layer raises; the far-field layer refuses a design before that).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _mpi, _pml
from gfem._adjoint.measure import measure
from gfem._adjoint.solve_vjp import input_kind
from gfem._checks import check_fixed, is_traced
from gfem._dim2 import refuse_entry
from gfem._power_flow import _model_role, _Role
from gfem._pytree import register_fields
from gfem._sources import DIPOLES, INCIDENT, PlaneWave
from gfem.errors import NonDifferentiableError
from gfem.mesh import Mesh, _at_imported_nodes

if TYPE_CHECKING:
    from gfem._engine.far_field import Layer
    from gfem._engine.power_flow import Surface
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["FarField", "far_field"]

_FIELD_INPUTS = {"coefficients": "coefficients", "mesh": "nodes", "omega": "omega"}
#: Why the radiated power (and what divides by it) has no node gradient.
POWER_NODES_REASON = (
    "the radiated power is read from power_flow's residual, which this version "
    "does not differentiate with respect to the node coordinates; differentiate "
    "the far-field amplitude or the radiation intensity instead, or wrap the "
    "radiated power in jax.lax.stop_gradient"
)


def engine() -> Any:
    """The NGSolve far-field module, imported on first use."""
    return importlib.import_module("gfem._engine.far_field")


@dataclasses.dataclass(frozen=True, eq=False)
class FarField:
    r"""The far field of one excitation on a grid of directions, per kept frequency.

    Under :math:`e^{+i\omega t}` the field far from the sources is
    :math:`E(r\hat r) \approx F(\hat r)\,e^{-ikr}/r`, with :math:`k` and
    :math:`\eta` of the exterior medium and the phase referred to the
    model's origin (`gfem.conventions`, Far fields). Every array has the
    kept frequencies first, then the broadcast shape of ``theta`` and
    ``phi``.

    Attributes:
        frequency: The kept frequencies of the column, Hz, ``(n_freq,)``.
        theta: The polar angles from :math:`+z`, rad.
        phi: The azimuths from :math:`+x`, rad.
        e_theta: :math:`F_\theta`, V (m times V/m).
        e_phi: :math:`F_\phi`, V.
        radiated_power: :math:`P_\mathrm{rad}`, W, ``(n_freq,)``: the power
            flowing out of the side, from the residual of the discrete
            equations (`gfem.power_flow`).
        accepted_power: :math:`P_\mathrm{acc}`, W, ``(n_freq,)``: a port's
            :math:`1 - |S_{ii}|^2` (1 W incident), a dipole's emitted power;
            ``None`` for an incident field.
        incident_power: :math:`P_\mathrm{inc}`, W, ``(n_freq,)``: a port's
            1 W, a dipole's emitted power (no mismatch); ``None`` for an
            incident field.
        incident_amplitude: :math:`|E_0|`, V/m, ``(n_freq,)``, of a
            `gfem.PlaneWave` column, ``None`` otherwise.
        impedance: :math:`\eta`, the exterior medium's wave impedance, ohm.
    """

    frequency: tuple[float, ...]
    theta: Any
    phi: Any
    e_theta: jax.Array
    e_phi: jax.Array
    radiated_power: jax.Array
    accepted_power: jax.Array | None
    incident_power: jax.Array | None
    incident_amplitude: jax.Array | None
    impedance: float

    def _per_frequency(self, x: Any) -> Any:
        """``x`` of shape ``(n_freq,)`` broadcast against the direction grid."""
        return jnp.reshape(x, (-1,) + (1,) * (jnp.ndim(self.e_theta) - 1))

    @property
    def intensity(self) -> jax.Array:
        r"""The radiation intensity :math:`U = |F|^2 / (2\eta)`, W/sr."""
        return (jnp.abs(self.e_theta) ** 2 + jnp.abs(self.e_phi) ** 2) / (
            2 * self.impedance
        )

    @property
    def directivity(self) -> jax.Array:
        r"""The directivity :math:`D = 4\pi U / P_\mathrm{rad}`."""
        return 4 * math.pi * self.intensity / self._per_frequency(self.radiated_power)

    def _driven(self, what: str) -> tuple[jax.Array, jax.Array]:
        if self.accepted_power is None or self.incident_power is None:
            raise ValueError(
                f"The {what} is an antenna's (a port or a dipole column): an "
                "incident field's column has no accepted power; read rcs."
            )
        return self.accepted_power, self.incident_power

    @property
    def gain(self) -> jax.Array:
        r"""The gain :math:`G = 4\pi U / P_\mathrm{acc}` (IEEE 145)."""
        accepted, _ = self._driven("gain")
        return 4 * math.pi * self.intensity / self._per_frequency(accepted)

    @property
    def realized_gain(self) -> jax.Array:
        r"""The realized gain :math:`G_r = 4\pi U / P_\mathrm{inc}` (IEEE 145).

        :math:`4\pi U` for a port column, fed 1 W.
        """
        _, incident = self._driven("realized gain")
        return 4 * math.pi * self.intensity / self._per_frequency(incident)

    @property
    def radiation_efficiency(self) -> jax.Array:
        r""":math:`P_\mathrm{rad} / P_\mathrm{acc}`, ``(n_freq,)``."""
        accepted, _ = self._driven("radiation efficiency")
        return self.radiated_power / accepted

    @property
    def e_right(self) -> jax.Array:
        r"""The right-hand circular component :math:`(F_\theta + iF_\phi)/\sqrt2`, V."""
        return (self.e_theta + 1j * self.e_phi) / math.sqrt(2)

    @property
    def e_left(self) -> jax.Array:
        r"""The left-hand circular component :math:`(F_\theta - iF_\phi)/\sqrt2`, V."""
        return (self.e_theta - 1j * self.e_phi) / math.sqrt(2)

    @property
    def axial_ratio(self) -> jax.Array:
        r"""The axial ratio :math:`(|F_R| + |F_L|)/\big||F_R| - |F_L|\big|`.

        1 for circular polarization, ``inf`` for linear.
        """
        r, left = jnp.abs(self.e_right), jnp.abs(self.e_left)
        return (r + left) / jnp.abs(r - left)

    @property
    def rcs(self) -> jax.Array:
        r"""The bistatic radar cross section :math:`4\pi|F|^2/|E_0|^2`, m².

        Of a `gfem.PlaneWave` column's scattered field.

        Raises:
            ValueError: The column is not a plane wave's.
        """
        if self.incident_amplitude is None:
            raise ValueError(
                "The radar cross section is a plane wave's (a gfem.PlaneWave "
                "column): it divides by the incident amplitude |E_0|; compute "
                "4 pi |F|^2 / |E_0|^2 for another incident field."
            )
        f2 = jnp.abs(self.e_theta) ** 2 + jnp.abs(self.e_phi) ** 2
        return 4 * math.pi * f2 / self._per_frequency(self.incident_amplitude) ** 2

    def __repr__(self) -> str:
        shape = tuple(np.shape(self.e_theta))
        return f"FarField(frequency={self.frequency!r}, shape={shape})"


register_fields(
    FarField,
    (
        "theta",
        "phi",
        "e_theta",
        "e_phi",
        "radiated_power",
        "accepted_power",
        "incident_power",
        "incident_amplitude",
    ),
)


def far_field(
    result: DrivenResult,
    name: str,
    theta: Any,
    phi: Any,
    *,
    model: Model,
    side: str | Sequence[str] | Callable[[np.ndarray], np.ndarray] | None = None,
) -> FarField:
    """See `gfem.measurements.far_field`."""
    from gfem.model import Model
    from gfem.results import DrivenResult

    refuse_entry(getattr(model, "mesh", None), "far_field")
    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"far_field needs a gfem.DrivenResult, got {type(result).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"far_field needs model= a gfem.Model, got {type(model).__name__}."
        )
    check_fixed(theta, "far_field's theta must be fixed numbers")
    check_fixed(phi, "far_field's phi must be fixed numbers")
    try:
        theta_b, phi_b = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
    except ValueError:
        raise ValueError(
            f"far_field's theta {np.shape(theta)} and phi {np.shape(phi)} do not "
            "broadcast together."
        ) from None
    if not (np.isfinite(theta_b).all() and np.isfinite(phi_b).all()):
        raise ValueError("far_field's theta and phi must be finite, in rad.")
    names = _side(side)
    source = result.sources.get(name)
    if source is None and name not in result.s.excited:
        columns = ", ".join(map(repr, result._excitations)) or "none"  # noqa: SLF001
        raise KeyError(
            f"{name!r} is neither an excited port nor a source of the result; its "
            f"columns are {columns}."
        )
    background = model.background
    if background is None:
        raise ValueError(
            "far_field reads the exterior medium from the model's background: "
            "give Model(background=gfem.Material(...)), the medium the field "
            "radiates into."
        )
    frequencies = [f for f, c in result.kept if c == name]
    if not frequencies:
        raise ValueError(
            f"No field of {name!r} was kept: solve with Driven(..., "
            "keep_fields=True) to read its far field."
        )
    incident = isinstance(source, INCIDENT)
    part = "scattered" if incident else "total"
    fields = [result.e_field(f, name, part=part) for f in frequencies]
    if fields[0]._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("far_field")
    traced_nodes = is_traced((fields[0].mesh.nodes, model.mesh.nodes))
    if callable(names) and traced_nodes:
        raise NonDifferentiableError(
            "far_field's side is a predicate on element centroids, which needs "
            "the node coordinates as numbers, but they are computed inside "
            "jax.grad, jax.jit or jax.vjp here: name the side's volume regions "
            "instead, so that its surface is known before the transformation runs."
        )
    concrete = _concrete(fields[0])
    ff = engine()
    layer = ff.exterior_layer(
        concrete,
        model,
        names,
        background,
        nodes=np.asarray(concrete.mesh.nodes),
        incident=incident,
    )
    pf = importlib.import_module("gfem._engine.power_flow")
    where = pf.surface_of(concrete, layer.inside, model)
    rhat, that, phat = ff.directions(theta_b.reshape(-1), phi_b.reshape(-1))
    if layer.ground is not None:
        _check_half_space(rhat, layer.ground)
    amplitude = []
    power = []
    for field in fields:
        a, p = _measure(field, model, layer, where, (rhat, that, phat))
        amplitude.append(a)
        power.append(p)
    amp = jnp.stack(amplitude)  # (n_freq, n_dir, 2)
    shape = (len(fields), *theta_b.shape)
    accepted, offered, e0 = _normalization(result, name, source, frequencies)
    eta = const.mu_0 * const.c * math.sqrt(layer.mu_r / layer.eps_r)
    return FarField(
        frequency=tuple(float(f) for f in frequencies),
        theta=theta_b,
        phi=phi_b,
        e_theta=amp[..., 0].reshape(shape),
        e_phi=amp[..., 1].reshape(shape),
        radiated_power=jnp.stack(power),
        accepted_power=accepted,
        incident_power=offered,
        incident_amplitude=e0,
        impedance=eta,
    )


def _side(side: Any) -> Any:
    """``None``, the names as a tuple, or the predicate."""
    if side is None or callable(side):
        return side
    if isinstance(side, str):
        return (side,)
    if isinstance(side, Sequence) and all(isinstance(s, str) for s in side):
        return tuple(side)
    raise TypeError(
        "far_field's side must be None (the layer touching the truncation), "
        "volume region names, or a predicate on element centroids, got "
        f"{type(side).__name__}."
    )


def _concrete(field: Field) -> Field:
    """``field`` at the imported node positions if its nodes are traced."""
    if not is_traced(field.mesh.nodes):
        return field
    return dataclasses.replace(
        field,
        mesh=_at_imported_nodes(field.mesh),
        coefficients=np.zeros(field.coefficients.shape, field.coefficients.dtype),
        omega=None if field.omega is None else 0.0,
    )


def _check_half_space(rhat: np.ndarray, ground: Any) -> None:
    """Raise for a direction below the ground plane."""
    below = rhat @ ground.normal < -1e-12
    if below.any():
        k = int(np.flatnonzero(below)[0])
        raise ValueError(
            f"The model ends on a {ground.kind} ground plane (normal "
            f"{tuple(np.round(ground.normal, 12))} into the domain), and the far "
            f"field exists on its side only, but direction {tuple(rhat[k])} points "
            "below it: give theta and phi in the half-space above the plane."
        )


def _normalization(
    result: DrivenResult, name: str, source: Any, frequencies: list[float]
) -> tuple[Any, Any, Any]:
    """``(P_acc, P_inc, |E_0|)`` of the column at the kept frequencies."""
    rows = [
        int(np.argmin(np.abs(np.asarray(result.frequency) - f))) for f in frequencies
    ]
    if source is None:  # a port fed 1 W
        s = result.s[name, name]
        accepted = 1.0 - jnp.abs(jnp.asarray(s)[np.asarray(rows)]) ** 2
        return accepted, jnp.ones(len(rows)), None
    if isinstance(source, DIPOLES):
        emitted = jnp.asarray(result.emitted_power[name])[np.asarray(rows)]
        return emitted, emitted, None
    if isinstance(source, PlaneWave):
        return None, None, jnp.full(len(rows), abs(source.amplitude))
    return None, None, None


def _measure(
    field: Field,
    model: Model,
    layer: Layer,
    where: Surface,
    basis: tuple[np.ndarray, ...],
) -> tuple[jax.Array, jax.Array]:
    """``(amplitude (n_dir, 2), radiated power)`` of one field."""
    f_paths, f_def = jax.tree_util.tree_flatten_with_path(field)
    m_paths, m_def = jax.tree_util.tree_flatten_with_path(model)
    inputs: dict[str, Any] = {}
    roles: dict[str, _Role] = {}
    for path, leaf in f_paths:
        key = _FIELD_INPUTS[path[0].name]
        inputs[key] = leaf
        roles[key] = _field_role(key, field)
    for path, leaf in m_paths:
        key = "model" + jax.tree_util.keystr(tuple(path))
        inputs[key] = leaf
        kind = input_kind(path, "model")
        roles[key] = (
            _Role("read", None, "the mesh node coordinates (model.mesh.nodes)")
            if kind[0] == "nodes"
            else _model_role(kind, model, where)
        )
    kernel = _FarFieldKernel(
        roles=roles,
        field_def=(f_def, tuple(_FIELD_INPUTS[p[0].name] for p, _ in f_paths)),
        model_def=(m_def, tuple(n for n in inputs if n.startswith("model"))),
        topology=field.mesh._topology,  # noqa: SLF001
        layer=layer,
        where=where,
        order=field.order,
        basis=basis,
    )
    out = measure(kernel, inputs)
    return out["amplitude"], out["power"]


def _field_role(name: str, field: Field) -> _Role:
    """The role of a field input."""
    if name == "nodes":
        return _Role("read", None, "the mesh node coordinates (field.mesh.nodes)")
    if name == "omega":
        return _Role(
            "read",
            (
                "a field's angular frequency is fixed: differentiate the solve's "
                "frequency through the kept field instead"
            ),
            "the field's omega",
        )
    if field._reduced:  # noqa: SLF001
        return _Role(
            "read",
            (
                "the field is formed from an adaptive sweep's frozen reduced "
                "basis, which holds no adjoint of the far field, so the "
                "frozen-basis gradient is not the solve's: solve with "
                'sweep="uniform" at the frequency whose far field you differentiate'
            ),
            "the coefficients of a field of an adaptive sweep (gfem.ReducedFields)",
        )
    return _Role("read", None, "the field coefficients")


class _FarFieldKernel:
    """The far-field amplitude and the radiated power of one field."""

    name = "gfem.far_field"
    pieces: ClassVar[frozenset[str]] = frozenset()
    comm = None

    def __init__(
        self,
        *,
        roles: dict[str, _Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        topology: Any,
        layer: Layer,
        where: Surface,
        order: int,
        basis: tuple[np.ndarray, ...],
    ) -> None:
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.topology = topology
        self.layer = layer
        self.where = where
        self.order = order
        self.basis = basis
        self.output_specs = {
            "amplitude": jax.ShapeDtypeStruct((len(basis[0]), 2), jnp.complex128),
            "power": jax.ShapeDtypeStruct((), jnp.float64),
        }
        self._flux: tuple[bytes, Any] | None = None

    def role(self, name: str) -> str:
        return self.roles[name].role

    def unsupported(self, name: str) -> str | None:
        return self.roles[name].unsupported

    def describe(self, name: str) -> str:
        return self.roles[name].description

    def check_backward(self, wanted: tuple[str, ...], live: tuple[str, ...]) -> None:
        """Raise at trace time for a node gradient of the radiated power."""
        nodes = [n for n in wanted if n == "nodes" or n.startswith("model.mesh")]
        if "power" in live and nodes:
            raise NonDifferentiableError(
                f"gfem cannot differentiate {self.name}'s radiated power (and "
                "the directivity, gain and efficiency divided by it) with respect "
                f"to {self.describe(nodes[0])}: {POWER_NODES_REASON}."
            )

    # --- host -------------------------------------------------------------------

    def _unflatten(self, values: dict[str, np.ndarray]) -> tuple[Field, Model]:
        f_def, f_names = self.field_def
        m_def, m_names = self.model_def
        field = jax.tree.unflatten(f_def, [values[n] for n in f_names])
        model = jax.tree.unflatten(m_def, [values[n] for n in m_names])
        return field, model

    def _probe(self, nodes: np.ndarray, field: Field) -> Any:
        from gfem._engine import probe
        from gfem._probe import layout

        ff = engine()
        bary, _ = ff.rule(self.order)
        mesh = Mesh(nodes, self.topology)
        return probe.probe(mesh, layout(field), ff.located(self.layer, bary))

    def _flux_of(self, values: dict[str, np.ndarray], model: Model, field: Field) -> Any:
        pf = importlib.import_module("gfem._engine.power_flow")
        m_names = self.model_def[1]
        key = b"".join(np.ascontiguousarray(values[n]).tobytes() for n in m_names)
        if self._flux is None or self._flux[0] != key:
            stretch = _pml.field_stretch(model, field)
            self._flux = (key, pf.build(self.order, model, self.where, stretch=stretch))
        return self._flux[1]

    def _function(
        self, nodes: np.ndarray, omega: float
    ) -> tuple[Callable[..., Any], np.ndarray]:
        """``(F(corners, E, curl E) -> (n_dir, 2), corners at the nodes)``."""
        ff = engine()
        mesh = Mesh(nodes, self.topology)
        corners0 = ff.corners(mesh, self.layer, nodes)
        chi = ff.chi_corners(mesh, self.layer)
        bary, weights = ff.rule(self.order)
        rhat, that, phat = self.basis
        k = omega / const.c * self.layer.index
        layer = self.layer

        def function(corners: Any, e: Any, curl: Any) -> Any:
            e_ref, c_ref = ff.reference_values(corners0, e, curl)
            v = ff.amplitudes(
                corners,
                e_ref,
                c_ref,
                chi=chi,
                bary=bary,
                weights=weights,
                rhat=rhat,
                k=k,
                ground=layer.ground,
            )
            v = v / (4 * math.pi)
            return jnp.stack(
                [jnp.sum(v * that, axis=1), jnp.sum(v * phat, axis=1)], axis=1
            )

        return function, corners0

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        field, model = self._unflatten(values)
        pf = importlib.import_module("gfem._engine.power_flow")
        pf.check_nodes(field, model)
        nodes = np.asarray(values["nodes"], float)
        omega = float(np.asarray(values["omega"]).real)
        e = np.asarray(values["coefficients"])
        flux = self._flux_of(values, model, field)
        pf.check_size(e, flux.op, self.order)
        prb = self._probe(nodes, field)
        ev = prb.values(e)[0]
        cv = prb.values(e, curl=True)[0]
        function, corners = self._function(nodes, omega)
        amplitude = np.asarray(function(corners, ev, cv))
        return {"amplitude": amplitude, "power": np.asarray(flux.power(e, omega))}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        field, model = self._unflatten(values)
        nodes = np.asarray(values["nodes"], float)
        omega = float(np.asarray(values["omega"]).real)
        e = np.asarray(values["coefficients"])
        out: dict[str, np.ndarray] = {}
        for name in wanted:
            out[name] = np.zeros_like(values[name])
        ct_a = cotangents.get("amplitude")
        if ct_a is not None and ("coefficients" in wanted or "nodes" in wanted):
            prb = self._probe(nodes, field)
            ev = prb.values(e)[0]
            cv = prb.values(e, curl=True)[0]
            function, corners = self._function(nodes, omega)
            _, pull = jax.vjp(function, corners, ev, cv)
            g_corners, g_e, g_c = (np.asarray(g) for g in pull(jnp.asarray(ct_a)))
            if "coefficients" in wanted:
                g = prb.transpose(g_e[None]) + prb.transpose(g_c[None], curl=True)
                out["coefficients"] = out["coefficients"] + g[0]
            if "nodes" in wanted:
                tets = np.asarray(Mesh(nodes, self.topology).tetrahedra)
                grad = np.zeros_like(nodes)
                np.add.at(
                    grad, tets[self.layer.tetrahedra], np.real(g_corners)
                )
                out["nodes"] = grad
        ct_p = cotangents.get("power")
        if ct_p is not None and "coefficients" in wanted:
            flux = self._flux_of(values, model, field)
            out["coefficients"] = out["coefficients"] + float(ct_p) * flux.pullback(
                e, omega
            )
        unknown = [
            n
            for n in wanted
            if n not in ("coefficients", "nodes") and not n.startswith("model.mesh")
        ]
        if unknown:  # the measure wrapper raised for the others
            raise NotImplementedError(f"no far-field gradient for {unknown}")
        return out
