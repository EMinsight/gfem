"""Temperature in optics: the first-order thermo-optic shift of a wave port's mode.

A material's `thermo_optic_coefficient` (dn/dT, 1/K) acts on the
temperature rise (the temperature above the source-free temperature, a
`gfem.SteadyThermalResult.rise`) about the index of the material's ``eps_r``
(ADR 0160). Two ways read it:

- `thermo_optic_shift`: the change of a wave port mode's effective index to
  first order in the rise, from one mode solve (ADR 0174)::

      delta_n_eff = (c eps0 / 4) sum_e 2 n0_e (dn/dT)_e rise_e int_e |E0|^2 dA

  over the triangles of the port's section, ``E0`` the mode at 1 W and
  ``rise_e`` the mean rise over the tetrahedron behind each triangle. It is
  linear in the rise and in dn/dT, and its phase per length is
  ``k0 delta_n_eff``.
- `gfem.Model.with_temperature_rise`: the model with ``eps_r = (n0 + dn/dT
  rise_e)^2`` per element, for any optical solve (a `gfem.PortMode`
  re-solve checks the first order).

The rise may live on another mesh than the optical model (a window around
one core, in the same coordinates): each element's rise is then the
quadrature mean of the rise at points located in its mesh (ADR 0176).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._field import Field
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._pytree import register_fields
from gfem.thermal import Trace

if TYPE_CHECKING:
    from gfem.mesh import Mesh
    from gfem.model import Model

__all__ = ["ThermoOpticShift", "thermo_optic_shift"]


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


def _coefficients(model: Model, port: str, tets: np.ndarray) -> jax.Array:
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
                f"thermo_optic_shift: region {region!r} of wave port {port!r}'s "
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
                f"thermo_optic_shift: region {region!r} of wave port {port!r}'s "
                "section is anisotropic and has a thermo_optic_coefficient: a "
                "scalar dn/dT on a permittivity tensor is not defined in this "
                "version. Give the region an isotropic eps_r."
            )
        given = True
        out = out.at[rows].set(_values(coefficient, pos))
    if not given:
        regions = ", ".join(repr(r) for r in _region_rows(mesh, tets))
        raise ValueError(
            f"thermo_optic_shift: no element of wave port {port!r}'s section has a "
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
    """The first-order thermo-optic shift of a wave port's mode.

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
            power per length ``P'``, ``P_pi = pi * P' / phase_per_length``.
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


def _frequency(frequency: Any) -> float:
    if np.ndim(frequency) != 0:
        raise ValueError(
            "thermo_optic_shift takes one frequency, got "
            f"{np.size(frequency)} values: call it once per frequency in a loop."
        )
    if is_traced(frequency):
        raise TypeError(
            "thermo_optic_shift: the frequency is a fixed number, not a "
            "differentiable input."
        )
    f = float(frequency)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"thermo_optic_shift: frequency must be > 0 Hz, got {f}.")
    return f


def thermo_optic_shift(
    model: Model,
    rise: Field | None,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
) -> ThermoOpticShift:
    """The first-order thermo-optic shift of a wave port's mode.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one frequency)
    and gives the change of its effective index caused by ``rise`` through
    each element's thermo-optic coefficient, to first order::

        delta_n_eff = (c eps0 / 4) sum_e 2 n0_e (dn/dT)_e rise_e int_e |E0|^2 dA

    ``E0`` is the mode at 1 W, ``e`` runs over the triangles of the port's
    section, ``n0_e`` and ``(dn/dT)_e`` are the material's of the
    tetrahedron behind it, and ``rise_e`` is the mean rise over that
    tetrahedron. Elements without a coefficient contribute nothing.

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
                model, steady.rise, port="p", frequency=gfem.c / 1.55e-6
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
    from gfem._adjoint import solve_vjp
    from gfem._engine.thermo_optic import ShiftWeightsRule
    from gfem._sweep.port_mode import _wave_port
    from gfem.model import Model
    from gfem.options import SolverOptions
    from gfem.problems import PortMode

    if not isinstance(model, Model):
        raise TypeError(
            f"thermo_optic_shift needs a gfem.Model, got {type(model).__name__}."
        )
    f = _frequency(frequency)
    wave_port = _wave_port(model, port)
    faces, tets = _section(model, wave_port)
    coefficients = _coefficients(model, port, tets)
    owner = f"thermo_optic_shift (wave port {port!r})"
    if rise is not None:
        rise_e = _element_rise(rise, model.mesh, tets, owner)
    problem = PortMode(port, f, mode=mode)
    rule = ShiftWeightsRule(model, problem, SolverOptions(), faces)
    if is_traced(model):
        outputs = solve_vjp(rule, model, problem)
    else:
        outputs, _ = rule.forward(model, keep=False)
    k0 = 2 * math.pi * f / scipy.constants.c
    gamma = jnp.asarray(outputs["gamma"])[0]
    n_eff = jnp.real(gamma / (1j * k0))
    weights = jnp.asarray(outputs["weights"]) * coefficients
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
        mode=rule.mode,
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
