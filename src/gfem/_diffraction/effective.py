r"""Effective permittivity and permeability of a slab from Floquet-port S.

The Nicolson-Ross-Weir retrieval in the robust-sign form, for the
``e^{+i\omega t}`` convention. `retrieve` is the pure ``jax.numpy`` math;
`effective_medium` reads the S, the media and the reference planes of a solved
periodic cell and calls it once per polarization column.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._diffraction._polarization import JONES_ORDER
from gfem._diffraction.outputs import (
    cell_of,
    check_offsets,
    check_surface,
    jones_block,
)
from gfem._floquet import C0
from gfem._pytree import register_fields
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.results import DrivenResult

#: |Re z| below which the sign of the impedance root is decided by passivity.
_SIGN_BAND = 5e-3
#: Cross-polarized and bianisotropy diagnostics: amplitude ratio above which
#: the model of a diagonal, reciprocal slab is flagged (1e-4 in power).
_DIAGNOSTIC = 1e-2


def retrieve(  # noqa: PLR0917  (the six inputs of the retrieval, called positionally)
    s11: jax.Array,
    s21: jax.Array,
    k0d: jax.Array,
    eps_b: float,
    mu_b: float,
    branch: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Retrieve :math:`(\varepsilon, \mu, n, Z/Z_0)` of a slab from its S.

    Pure ``jax.numpy``, so ``jit`` and ``grad`` apply. With
    :math:`z = Z/Z_b` the slab's impedance relative to the background and
    :math:`\Gamma = (z-1)/(z+1)`, the slab of optical thickness
    :math:`k_0 d` has :math:`T = e^{-i n k_0 d}`,
    :math:`S_{11} = \Gamma(1-T^2)/(1-\Gamma^2T^2)` and
    :math:`S_{21} = T(1-\Gamma^2)/(1-\Gamma^2T^2)`. Inverting,
    :math:`z = \sqrt{((1+S_{11})^2-S_{21}^2)/((1-S_{11})^2-S_{21}^2)}` with
    :math:`\operatorname{Re} z \ge 0`; where :math:`|\operatorname{Re} z|` is
    below :math:`5\times10^{-3}` the sign that gives :math:`|T|\le1` is taken
    instead. Then :math:`T = S_{21}/(1-S_{11}\Gamma)` and
    :math:`n = (2\pi m - \arg T + i\ln|T|)/(k_0 d)`, :math:`m` the branch and
    :math:`\arg T` unwrapped along axis 0.

    Args:
        s11: Reflection, ``(n_freq,)`` or ``(n_freq, n_col)``, complex.
        s21: Transmission, the shape of ``s11``.
        k0d: :math:`k_0 d`, ``(n_freq,)``, vacuum wave number times the
            thickness (rad).
        eps_b: Background relative permittivity.
        mu_b: Background relative permeability.
        branch: Integer :math:`m` of the phase sheet at the lowest frequency.

    Returns:
        ``(permittivity, permeability, index, impedance)`` of the shape of
        ``s11``, the impedance relative to :math:`Z_0`.
    """
    s11 = jnp.asarray(s11, complex)
    s21 = jnp.asarray(s21, complex)
    k0d = jnp.asarray(k0d, float)
    if s11.ndim == 2:
        k0d = k0d[:, None]
    num = (1 + s11) ** 2 - s21**2
    den = (1 - s11) ** 2 - s21**2
    z0 = jnp.sqrt(num / den)

    def transmission(z):
        return s21 / (1 - s11 * (z - 1) / (z + 1))

    near_zero = jnp.abs(jnp.real(z0)) < _SIGN_BAND
    keep = jnp.abs(transmission(z0)) <= jnp.abs(transmission(-z0))
    z = jnp.where(near_zero & ~keep, -z0, z0)
    gamma = (z - 1) / (z + 1)
    t = s21 / (1 - s11 * gamma)
    phase = jnp.unwrap(jnp.angle(t), axis=0)
    n = (2 * jnp.pi * branch - phase + 1j * jnp.log(jnp.abs(t))) / k0d
    z_rel = z * jnp.sqrt(mu_b / eps_b)
    return n / z_rel, n * z_rel, n, z_rel


@dataclasses.dataclass(frozen=True, eq=False)
class EffectiveMedium:
    r"""The effective diagonal permittivity and permeability of a slab.

    Relative to vacuum, in the lattice frame :math:`(\hat e_1, \hat e_2)`,
    per frequency. Column :math:`i` is the wave with :math:`\mathbf{E}` along
    :math:`\hat e_{i+1}`, so :math:`\varepsilon_{11} = n_0/z_0`,
    :math:`\mu_{22} = n_0 z_0`, :math:`\varepsilon_{22} = n_1/z_1` and
    :math:`\mu_{11} = n_1 z_1`.

    The values describe the slab only while :math:`|n| k_0 a \ll 1`, :math:`a`
    the period along the propagation; near the Bragg condition they are not
    material constants.

    Attributes:
        permittivity: ``(n_freq, 2)`` complex :math:`(\varepsilon_{11},
            \varepsilon_{22})`.
        permeability: ``(n_freq, 2)`` complex :math:`(\mu_{11}, \mu_{22})`.
        index: ``(n_freq, 2)`` complex refractive index of each column, with
            :math:`\operatorname{Im} n \le 0` for a passive slab.
        impedance: ``(n_freq, 2)`` complex impedance of each column relative
            to :math:`Z_0`.
        thickness: The slab thickness between the reference planes, m.
        frequency: ``(n_freq,)`` Hz.
        branch: The phase sheet used at the lowest frequency.

    Raises:
        ValueError: An array field is not ``(n_freq, 2)``, or ``frequency``
            does not hold one value per row.
    """

    permittivity: jax.Array
    permeability: jax.Array
    index: jax.Array
    impedance: jax.Array
    thickness: float
    frequency: np.ndarray
    branch: int

    def __post_init__(self) -> None:
        frequency = np.atleast_1d(np.asarray(self.frequency, float))
        for name in ("permittivity", "permeability", "index", "impedance"):
            shape = tuple(np.shape(getattr(self, name)))
            if len(shape) != 2 or shape[1] != 2:
                raise ValueError(
                    f"EffectiveMedium: {name} must have shape (n_freq, 2), got {shape}."
                )
            if frequency.ndim != 1 or len(frequency) != shape[0]:
                raise ValueError(
                    f"EffectiveMedium: frequency must hold {shape[0]} frequencies, "
                    f"one per row of {name}, got {len(frequency)}."
                )
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "thickness", float(self.thickness))
        object.__setattr__(self, "branch", int(self.branch))

    def __repr__(self) -> str:
        return (
            f"EffectiveMedium({len(self.frequency)} frequencies, "
            f"thickness {self.thickness:g} m, branch {self.branch})"
        )


register_fields(EffectiveMedium, ("permittivity", "permeability", "index", "impedance"))


def _check_branch(branch: Any, who: str) -> int:
    if is_traced(branch):
        raise NonDifferentiableError(
            f"{who}: branch is an integer sheet of the phase, a fixed value that "
            "gfem does not differentiate with respect to. Pass a concrete int."
        )
    if isinstance(branch, (bool, np.bool_)) or not isinstance(
        branch, (int, np.integer)
    ):
        raise TypeError(f"{who}: branch must be an int, got {branch!r}.")
    return int(branch)


def _check_normal(who: str, cell: Any, names: dict[str, list[str]]) -> None:
    """Raise unless every port's wave vector is exactly zero (normal incidence)."""
    for group in names.values():
        for name in group:
            wv = cell.info[name].wave_vector
            if is_traced(wv):
                raise NonDifferentiableError(
                    f"{who}: the result's diffraction orders are traced by JAX "
                    "(the result entered jax.jit or jax.vmap as an argument), so "
                    "normal incidence cannot be checked; they are constants of "
                    f"the solve. Call {who} on the result inside the function "
                    "that calls gfem.solve, or outside the transformation."
                )
            if np.any(np.asarray(wv) != 0.0):
                raise ValueError(
                    f"{who}: Floquet port {name!r} has a wave vector with a "
                    "tangential part (oblique incidence); the retrieval is for "
                    "normal incidence. Solve with Driven(...) without wave_vector."
                )


def effective_medium(
    result: DrivenResult,
    incident: str,
    transmitted: str,
    *,
    model: Model,
    branch: int = 0,
) -> EffectiveMedium:
    r"""The effective permittivity and permeability of a slab between two faces.

    A slab of a periodic cell, seen at normal incidence through the specular
    Floquet ports of two faces, behaves for these waves as a homogeneous slab
    of diagonal :math:`\varepsilon` and :math:`\mu` in the lattice frame. The
    reference planes are those of the ports,
    :math:`\mathbf{p} = \mathbf{p}_\text{face} + \text{offset}\,\hat n` with
    :math:`\hat n` the face normal into the domain; put the offsets on the
    slab faces, and the slab is what lies between the planes (thickness
    :math:`d`). The reflection :math:`S_{11}` and transmission
    :math:`S_{21}` of each column feed a Nicolson-Ross-Weir inversion (see
    `gfem.diffraction`'s ``e^{+i\omega t}`` convention): the impedance root
    :math:`z = \sqrt{((1+S_{11})^2-S_{21}^2)/((1-S_{11})^2-S_{21}^2)}` with
    :math:`\operatorname{Re} z \ge 0` (passivity decides it within
    :math:`5\times10^{-3}` of zero), :math:`T = e^{-i n k_0 d}`, and
    :math:`n = (2\pi m - \arg T + i \ln|T|)/(k_0 d)` with :math:`\arg T`
    unwrapped along frequency from the principal value at the lowest one and
    :math:`m` = ``branch``. Then :math:`\varepsilon = n / (Z/Z_0)` and
    :math:`\mu = n\,Z/Z_0`.

    Both polarizations of the specular order need a Floquet port on both
    faces and both incident columns must be excited. Several frequencies must
    increase strictly (the phase is unwrapped along them). Differentiable
    through S (``jax.grad``, ``jax.jit``).

    Warns when the cross-polarized reflection or transmission exceeds 1 % of
    the co-polarized amplitude (the principal axes are not along
    :math:`\hat e_1, \hat e_2`), and, when the ``transmitted`` columns are
    excited too, when :math:`|S_{11} - S_{22}|` exceeds 1e-2 (a bianisotropic
    or asymmetric cell). These are model diagnostics, not accuracy claims. They
    are skipped while S is traced.

    Args:
        result: A `gfem.DrivenResult` of a model with Floquet ports.
        incident: The Floquet face the wave enters through.
        transmitted: The Floquet face on the other side of the slab.
        model: The model the result was solved with.
        branch: Integer phase sheet of :math:`n k_0 d` at the lowest
            frequency (0 for a slab thinner than half a wavelength in the
            medium).

    Returns:
        An `EffectiveMedium`.

    Raises:
        TypeError: ``result`` or ``model`` has the wrong type, ``branch`` is
            not an int.
        NonDifferentiableError: ``branch`` or the result's wave vectors are
            traced.
        NotImplementedError: The two faces have different media behind them.
        ValueError: An angle sweep (a sequence of results), a wave vector with
            a tangential part (oblique incidence), a face without Floquet
            ports, a missing or unexcited port, ``incident`` equal to
            ``transmitted``, two orders' ports at different offsets, reference
            planes that meet or cross, several frequencies not strictly
            increasing.

    Warns:
        UserWarning: Cross-polarized transmission or reflection, or a
            reflection that depends on the side, above the diagnostics above.
    """
    who = "effective_medium"
    branch = _check_branch(branch, who)
    cell = cell_of(result, model, who)
    if cell.sequence:
        raise ValueError(
            f"{who}: an angle sweep (a sequence of results) is not retrieved; "
            "the retrieval is for normal incidence. Pass the one result of the "
            "normal-incidence solve."
        )
    check_surface(cell, incident)
    check_surface(cell, transmitted)
    if incident == transmitted:
        raise ValueError(
            f"{who}: incident and transmitted are both {incident!r}; name the two "
            "faces on either side of the slab."
        )
    order = (0, 0)
    names: dict[str, list[str]] = {}
    for surface in (incident, transmitted):
        names[surface] = []
        for pol in JONES_ORDER:
            name = cell.port(surface, order, pol)
            if name is None:
                raise ValueError(
                    f"{who}: no Floquet port reads order {order} in {pol} on "
                    f"{surface!r} (the retrieval needs both polarizations of the "
                    f"specular order on both faces). Add gfem.FloquetPort(name, "
                    f"{surface!r}, order={order}, polarization={pol!r}) to the "
                    "model."
                )
            names[surface].append(name)
    for name in names[incident]:
        if name not in cell.s.excited:
            d = cell.info[name]
            raise ValueError(
                f"{who}: Floquet port {name!r} (order {d.order}, {d.polarization} "
                f"on {d.surface!r}) was not excited, so S has no column for it. "
                f"Solve with Driven(..., excite=None) or add {name!r} to "
                "Driven.excite."
            )
    check_offsets(cell, incident, order)
    check_offsets(cell, transmitted, order)
    _check_normal(who, cell, names)
    f_in, f_tr = cell.face(incident), cell.face(transmitted)
    if f_in.eps_r != f_tr.eps_r or f_in.mu_r != f_tr.mu_r:
        raise NotImplementedError(
            f"{who}: the media behind {incident!r} (eps_r {f_in.eps_r:g}, mu_r "
            f"{f_in.mu_r:g}) and {transmitted!r} (eps_r {f_tr.eps_r:g}, mu_r "
            f"{f_tr.mu_r:g}) differ; the retrieval needs the same medium on both "
            "sides of the slab."
        )
    offset_in = cell.offsets.get(names[incident][0], 0.0)
    offset_tr = cell.offsets.get(names[transmitted][0], 0.0)
    plane_in = f_in.origin + offset_in * f_in.normal
    plane_tr = f_tr.origin + offset_tr * f_tr.normal
    thickness = float((plane_tr - plane_in) @ f_in.normal)
    if not thickness > 0.0:
        raise ValueError(
            f"{who}: the reference planes of {incident!r} and {transmitted!r} "
            f"are {thickness:g} m apart along the propagation (meeting or "
            "crossed), so the slab has no thickness. Give the Floquet ports the "
            "offsets that put the planes on the slab faces, "
            "FloquetPort(offset=...) or result.with_offsets(...)."
        )
    freq = cell.frequency
    if len(freq) > 1 and not np.all(np.diff(freq) > 0):
        raise ValueError(
            f"{who}: the frequencies must increase strictly (the phase is "
            "unwrapped along them); solve them in ascending order."
        )
    s = cell.s
    refl = jones_block(s, names[incident], names[incident])
    trans = jones_block(s, names[transmitted], names[incident])
    _diagnose(who, cell, names, incident, transmitted, refl, trans)
    k0d = 2.0 * np.pi * freq / C0 * thickness
    s11 = jnp.stack([refl[:, 0, 0], refl[:, 1, 1]], axis=1)
    s21 = jnp.stack([trans[:, 0, 0], trans[:, 1, 1]], axis=1)
    eps, mu, n, z = retrieve(s11, s21, k0d, f_in.eps_r, f_in.mu_r, branch)
    # column 0 (E along e1) gives (eps11, mu22); column 1 (E along e2) (eps22, mu11)
    return EffectiveMedium(
        permittivity=eps,
        permeability=mu[:, ::-1],
        index=n,
        impedance=z,
        thickness=thickness,
        frequency=freq,
        branch=branch,
    )


def _diagnose(  # noqa: PLR0917  (internal: labels and the two blocks)
    who: str,
    cell: Any,
    names: dict[str, list[str]],
    incident: str,
    transmitted: str,
    refl: jax.Array,
    trans: jax.Array,
) -> None:
    """Warn on cross-polarization and on side-dependent reflection (concrete S)."""
    if is_traced(refl) or is_traced(trans):
        return
    for label, block in (("reflection", refl), ("transmission", trans)):
        b = np.abs(np.asarray(block))
        co = np.maximum(b[:, 0, 0], b[:, 1, 1])
        cross = np.maximum(b[:, 0, 1], b[:, 1, 0])
        if np.any(cross > _DIAGNOSTIC * co):
            warnings.warn(
                f"{who}: the cross-polarized {label} reaches "
                f"{float(np.max(cross / np.maximum(co, 1e-300))):.2g} of the "
                "co-polarized amplitude, so the principal axes of the slab are not "
                "along the lattice vectors e1 and e2 and the diagonal tensor "
                "describes it only approximately. Align the lattice with the "
                "principal axes.",
                UserWarning,
                stacklevel=3,
            )
    if all(n in cell.s.excited for n in names[transmitted]):
        back = jones_block(cell.s, names[transmitted], names[transmitted])
        diff = np.abs(np.asarray(refl) - np.asarray(back))
        diff = np.maximum(diff[:, 0, 0], diff[:, 1, 1])
        if np.any(diff > _DIAGNOSTIC):
            warnings.warn(
                f"{who}: the reflection from {incident!r} and from "
                f"{transmitted!r} differ by {float(np.max(diff)):.2g}, so the cell "
                "is not the symmetric, reciprocal slab the retrieval assumes "
                "(a bianisotropic or asymmetric cell); the effective parameters "
                "describe only the wave from the first face.",
                UserWarning,
                stacklevel=3,
            )


__all__ = ["EffectiveMedium", "effective_medium", "retrieve"]
