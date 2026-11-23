r"""The linear electro-optic (Pockels) shift of a wave port's mode.

A material's ``electro_optic_tensor`` (:math:`r_{Ik}`, m/V) changes its
optical permittivity under the static field :math:`E_\mathrm{s} = -\nabla V`
of an electrostatic potential:

.. math::

    \Delta\varepsilon_r = -\varepsilon_r\,(r \cdot E_\mathrm{s})\,\varepsilon_r,
    \qquad (r \cdot E_\mathrm{s})_{ij} = \sum_k r_{ijk} E_{\mathrm{s},k}

`electro_optic_shift` gives the change of a wave port mode's effective index
to first order in that change, from one mode solve, per volt on one
terminal's potential.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._dim2 import refuse_entry
from gfem._field import Field, FieldMap
from gfem._materials import (
    _VOIGT_ROW,
    Material,
    electro_optic_in_frame,
    principal_tensor,
)
from gfem._optical_shift import (
    _check_model,
    _frequency,
    _lossy,
    _mode_weights,
    _region_rows,
    _section,
    _wave_port,
)
from gfem._pytree import register_fields
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem.mesh import Mesh
    from gfem.model import Model

__all__ = ["ElectroOpticShift", "electro_optic_shift"]

_OWNER = "electro_optic_shift"


@dataclasses.dataclass(frozen=True, eq=False)
class ElectroOpticShift:
    r"""The first-order electro-optic shift of a wave port's mode, per volt.

    The mode is solved once on the optical model and held constant: the
    shift is exactly linear in the potential and in the electro-optic
    tensor. Per volt means per volt on the terminal whose potential was
    given (1 V there, 0 V on the other terminals and the ground); a drive
    :math:`V` multiplies every per-volt value by :math:`V`.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: The frequency, Hz.
        n_eff: The unperturbed effective index (real; the section is
            lossless), a constant.
        delta_n_eff_per_volt: :math:`\partial n_\mathrm{eff}/\partial V`,
            1/V.
        phase_per_length_per_volt: :math:`k_0\,\partial n_\mathrm{eff}/
            \partial V`, rad/(V m): the extra phase delay per unit length of
            guide and per volt. A transmission through length :math:`L`
            turns its argument by :math:`-\phi V L`, :math:`\phi` this value
            (`gfem.conventions`).
    """

    port: str
    mode: int
    frequency: float
    n_eff: jax.Array
    delta_n_eff_per_volt: jax.Array
    phase_per_length_per_volt: jax.Array

    @property
    def v_pi_l(self) -> jax.Array:
        r"""The voltage-length product of this arm, V m.

        :math:`V_\pi L = \lambda / (2\,|\partial n_\mathrm{eff}/\partial V|)`,
        :math:`\lambda = c/f`: the drive times length that turns the phase by
        :math:`\pi`; infinite for a zero shift. :math:`V_\pi` of a length
        :math:`L` is ``v_pi_l / L``; ``v_pi_l * 100`` is in V cm.
        """
        wavelength = scipy.constants.c / self.frequency
        return wavelength / (2 * jnp.abs(self.delta_n_eff_per_volt))

    def __repr__(self) -> str:
        try:
            n = float(np.asarray(self.n_eff))
            dn = float(np.asarray(self.delta_n_eff_per_volt))
            vl = float(np.asarray(self.v_pi_l))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"ElectroOpticShift(port {self.port!r}, mode {self.mode})"
        return (
            f"ElectroOpticShift(port {self.port!r}, mode {self.mode}, "
            f"{self.frequency:.6g} Hz, n_eff {n:.6f}, delta_n_eff_per_volt "
            f"{dn:.6g} /V, v_pi_l {vl:.6g} V m)"
        )


register_fields(
    ElectroOpticShift, ("n_eff", "delta_n_eff_per_volt", "phase_per_length_per_volt")
)


def _check_potential(potential: Any) -> None:
    """Raise unless ``potential`` is one terminal's potential on one process."""
    remedy = (
        "Pass one terminal's potential, e.g. es.potential['signal'] (pick one "
        "terminal; the shift of several terminals is the sum of each "
        "terminal's shift times its drive)."
    )
    if isinstance(potential, FieldMap):
        raise TypeError(
            f"{_OWNER} takes one terminal's potential, got the whole mapping of "
            f"terminals {list(potential)}. {remedy}"
        )
    if not isinstance(potential, Field):
        raise TypeError(
            f"{_OWNER} takes one terminal's potential (a gfem.Field), got "
            f"{type(potential).__name__}. {remedy}"
        )
    if potential.quantity != "V" or potential.space != "h1":
        raise TypeError(
            f"{_OWNER} takes a potential in volts (quantity 'V' on h1), got "
            f"a field of quantity {potential.quantity!r} on {potential.space}. "
            f"{remedy}"
        )
    if potential._communicator is not None:  # noqa: SLF001
        raise NotImplementedError(
            f"{_OWNER}: the potential is split across ranks; gather on one rank "
            "or solve on one process."
        )


def _permittivity(material: Any, pos: np.ndarray) -> Any:
    """The optical permittivity of elements ``pos``, ``(n, 3, 3)``, global frame."""
    if isinstance(material, Material):
        eps = principal_tensor(np.real(material.eps_r), material.axes)
        return jnp.broadcast_to(jnp.asarray(eps), (len(pos), 3, 3))
    eps = jnp.real(jnp.asarray(1.0 if material.eps_r is None else material.eps_r))
    if material.axes is None:
        eps = eps if eps.ndim == 0 else eps[pos]
        return jnp.broadcast_to(eps, (len(pos),))[:, None, None] * jnp.eye(3)
    eps = jnp.broadcast_to(eps if eps.ndim == 1 else eps[pos], (len(pos), 3))
    a = jnp.asarray(material.axes, dtype=jnp.float64)
    return jnp.einsum("nk,ki,kj->nij", eps, a, a)


def _tensor(material: Any, pos: np.ndarray) -> Any:
    """The contracted electro-optic tensor of elements ``pos``, global frame."""
    r = material.electro_optic_tensor
    r = np.asarray(r) if isinstance(material, Material) else jnp.asarray(r)
    if r.ndim == 3:
        r = r[pos]
    r = jnp.asarray(electro_optic_in_frame(r, material.axes))
    return jnp.broadcast_to(r, (len(pos), 6, 3))


def _crystal(
    model: Model, port: str, tets: np.ndarray
) -> tuple[np.ndarray, jax.Array, jax.Array]:
    """The section's elements with a tensor: rows into ``tets``, eps and r.

    Raises:
        ValueError: No element behind the section has a tensor.
        NotImplementedError: The section is lossy.
    """
    mesh = model.mesh
    rows, eps, r = [], [], []
    regions = _region_rows(mesh, tets)
    for region, at in regions.items():
        material = model.materials[region]
        pos = np.searchsorted(mesh.elements(region), tets[at])
        loss = _lossy(material, pos)
        if loss is not None:
            raise NotImplementedError(
                f"{_OWNER}: region {region!r} of wave port {port!r}'s section has "
                f"{loss}. The first-order shift weighs Re(E E^H), which holds for "
                "a lossless section only: give the section's regions a real eps_r "
                "with no tan_delta or conductivity (an optical window or "
                "EM-excluded electrodes keep the loss out of the section)."
            )
        if material.electro_optic_tensor is None:
            continue
        rows.append(at)
        eps.append(_permittivity(material, pos))
        r.append(_tensor(material, pos))
    if not rows:
        names = ", ".join(repr(name) for name in regions)
        raise ValueError(
            f"{_OWNER}: no element of wave port {port!r}'s section has an "
            f"electro_optic_tensor (its regions are {names}), so the shift is "
            "zero. Set electro_optic_tensor on the crystal's material."
        )
    return np.concatenate(rows), jnp.concatenate(eps), jnp.concatenate(r)


def _static_field(potential: Field, mesh: Mesh, tets: np.ndarray, owner: str) -> Any:
    r"""The mean of :math:`-\nabla V` over each of ``tets`` of ``mesh``, ``(n, 3)``.

    The quadrature mean at points of each tetrahedron (a rule exact for the
    gradient of the potential's order), located in the potential's mesh: on
    the potential's own mesh, at its nodes, the exact element mean. Linear
    in the potential's coefficients.

    Raises:
        ValueError: A point is outside the potential's mesh or support.
    """
    from gfem import _probe
    from gfem._adjoint.measure import measure
    from gfem._engine.electro_optic import StaticFieldKernel

    engine = _probe.engine()
    layout = _probe.layout(potential)
    bary, weights = engine.tetrahedron_rule(max(potential.order - 1, 0))
    corners = np.asarray(mesh.nodes, float)[np.asarray(mesh.tetrahedra)[tets]]
    points = np.einsum("qk,nkd->nqd", bary, corners).reshape(-1, 3)
    static = potential.mesh
    located = nodes = None
    if not is_traced(static.nodes):
        nodes = np.asarray(static.nodes)
        same = static._topology is mesh._topology and np.array_equal(  # noqa: SLF001
            nodes, np.asarray(mesh.nodes)
        )
        try:
            if same:
                located = _same_mesh(static, layout, tets, bary, points)
            else:
                located = engine.locate(static, layout, points)
        except ValueError as err:
            raise ValueError(f"{owner}: {err}") from err
    kernel = StaticFieldKernel(
        static._topology,  # noqa: SLF001
        layout,
        points,
        weights,
        located=located,
        nodes=nodes,
    )
    inputs = {"coefficients": potential.coefficients, "nodes": static.nodes}
    return measure(kernel, inputs)["field"]


def _same_mesh(
    mesh: Mesh, layout: Any, tets: np.ndarray, bary: np.ndarray, points: np.ndarray
) -> Any:
    """Points of ``tets`` located in the mesh they were placed in.

    Raises:
        ValueError: A tetrahedron is outside the potential's support.
    """
    from gfem import _probe

    engine = _probe.engine()
    q = len(bary)
    held = engine.in_support(mesh, layout)[tets]
    if not held.all():
        k = int(np.flatnonzero(~held)[0])
        raise engine.outside_support(mesh, layout, int(tets[k]), points[k * q])
    return engine.Located(np.repeat(tets, q), np.tile(bary, (len(tets), 1)))


def electro_optic_shift(
    model: Model,
    potential: Field,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
) -> ElectroOpticShift:
    r"""The first-order electro-optic shift of a wave port's mode, per volt.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one frequency)
    and gives the change of its effective index caused by the static field
    of ``potential`` through each element's ``electro_optic_tensor``, to
    first order:

    .. math::

        \delta n_\mathrm{eff} = \sum_t W_t : \Delta\varepsilon_t, \qquad
        W_t = \frac{c\varepsilon_0}{4}\,\mathrm{Re}\int_t E_0 E_0^H\,dA,
        \qquad
        \Delta\varepsilon_t = -\varepsilon_t\,(r_t \cdot E_{\mathrm{s},t})\,
            \varepsilon_t

    :math:`E_0` is the mode at 1 W, :math:`t` runs over the triangles of
    the port's section, :math:`\varepsilon_t` and :math:`r_t` are the
    optical permittivity and the electro-optic tensor of the tetrahedron
    behind it (in the global frame, rotated from the material's ``axes``),
    and :math:`E_{\mathrm{s},t}` is the mean of :math:`-\nabla V` over that
    tetrahedron. Elements without a tensor contribute nothing. The full
    tensor counts, off-diagonal entries included.

    ``model`` is the optical model: ``eps_r`` at the optical frequency and
    the tensor, with the electrodes excluded from electromagnetic solves
    (``excluded={"em": [...]}``), or a separate window model around one
    core in the same coordinates. Take an optical window out to the
    electrodes' inner edges: the shift weighs the mode's lateral tail,
    where the static field is strongest. ``potential`` is one terminal's
    potential of a separate electrostatic model (``eps_r`` the static
    permittivity) on any mesh that holds the section's crystal elements (a
    DC-conduction potential serves too: any potential in volts on H1); give
    the static and the optical crystal the same ``axes``, which the
    shift cannot check. The mode is the port's (``mode=None``) or the one
    given, as `gfem.PortMode` picks it; exclusions or a window make the
    core's mode the first.

    `jax.grad` reaches the electrostatic mesh's node coordinates (an
    electrode gap through a `gfem.shape.boundary_morph` of the electrostatic
    model): the static field is read at fixed points of the optical model,
    so its derivative counts the potential's change and the static mesh
    moving under those points. In a gap morph, name the crystal's interfaces
    in the electrostatic mesh so they slide, and hold the waveguide with
    ``fixed=``; an unnamed interface deforms with the electrode. With the
    optical model on another mesh, the shift jumps slightly when a point
    crosses a face of the moving static mesh: the gradient is that of the
    smooth piece the design is on.

    The shift is linear in the potential: several terminals' shifts add,
    each times its drive. A push-pull modulator's two arms, ``a`` and ``b``
    (two ports on one potential), give
    :math:`V_\pi L = \lambda / (2\,|s_a - s_b|)` with :math:`s` the
    ``delta_n_eff_per_volt`` of each arm.

    Example:
        V_pi L of each arm and of the push-pull pair::

            problem = gfem.Electrostatic(terminals=["signal"], ground="ground")
            es = gfem.solve(static, problem)
            f = scipy.constants.c / 1.55e-6
            a = gfem.electro_optic_shift(
                optical, es.potential["signal"], port="a", frequency=f
            )
            b = gfem.electro_optic_shift(
                optical, es.potential["signal"], port="b", frequency=f
            )
            print(a.v_pi_l * 100, "V cm")
            push_pull = (
                scipy.constants.c
                / f
                / (2 * abs(a.delta_n_eff_per_volt - b.delta_n_eff_per_volt))
            )

    Args:
        model: The optical model.
        potential: One terminal's potential, a scalar `gfem.Field`
            (``ElectrostaticResult.potential["name"]``), at 1 V on that
            terminal.
        port: The wave port's name.
        frequency: One frequency, Hz.
        mode: Which mode (1 = the most propagating), or ``None`` for the
            wave port's own.

    Returns:
        The shift: ``n_eff``, ``delta_n_eff_per_volt``,
        ``phase_per_length_per_volt`` and ``v_pi_l``.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`; ``potential`` is not one
            terminal's potential (a mapping of terminals, a result, or a
            field other than a potential).
        ValueError: ``port`` is not a wave port of the model, a sequence of
            frequencies is given, no element of the section has an
            ``electro_optic_tensor``, or a crystal element lies outside the
            potential's mesh or support (the message names the point).
        NotImplementedError: The model or the potential is on a 2D mesh, the
            potential is split across ranks, or the section is lossy (a
            complex ``eps_r``, a ``tan_delta`` or ``conductivity`` other than
            0, or a permittivity law).
        NonDifferentiableError: Under `jax.grad`, an input of the optical
            model other than the tensor (its ``eps_r``, ``tan_delta``,
            ``mu_r``, ``conductivity`` or node coordinates): the mode is held
            constant.
    """
    refuse_entry(getattr(model, "mesh", None), _OWNER)
    refuse_entry(getattr(potential, "mesh", None), _OWNER)
    _check_model(model, _OWNER)
    _check_potential(potential)
    f = _frequency(frequency, _OWNER)
    faces, tets = _section(model, _wave_port(model, port))
    rows, eps, r = _crystal(model, port, tets)
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{_OWNER}: gfem cannot differentiate the electro-optic shift with "
            "respect to the optical model's node coordinates: the mode and the "
            "points the static field is read at are held constant. Wrap them in "
            "jax.lax.stop_gradient to use their value without their derivative."
        )
    owner = f"{_OWNER} (wave port {port!r})"
    field = _static_field(potential, model.mesh, tets[rows], owner)
    found, n_eff, weights, k0 = _mode_weights(
        model, port, f, mode, faces, tensor=True, rule=_weights_rule()
    )
    eta = jnp.einsum("nIk,nk->nI", r, field)[:, _VOIGT_ROW]
    delta_eps = -eps @ eta @ eps
    delta = jnp.sum(weights[rows] * delta_eps)
    return ElectroOpticShift(
        port=port,
        mode=found,
        frequency=f,
        n_eff=n_eff,
        delta_n_eff_per_volt=delta,
        phase_per_length_per_volt=k0 * delta,
    )


def _weights_rule() -> type:
    from gfem._engine.electro_optic import ElectroOpticWeightsRule

    return ElectroOpticWeightsRule
