r"""Emissivity of a periodic emitter by Kirchhoff's law, and the hemisphere rule.

`emissivity` reads the absorptance of an angle sweep and labels it with the
emission direction; `hemisphere` builds the quadrature of the hemispherical
average and `hemispherical_emissivity` applies it. Every output is
`jax.numpy` of ``result.s.values`` with static weights, so it differentiates
through S.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_jacobi

from gfem._diffraction.outputs import balance_of
from gfem._diffraction.sweep import AngleSweepResult
from gfem._pytree import register_fields

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = [
    "Emissivity",
    "HemisphereRule",
    "emissivity",
    "hemisphere",
    "hemispherical_emissivity",
]

_POLARIZATIONS = ("TE", "TM", "RCP", "LCP", "unpolarized")
#: Absolute tolerance, rad, between a sweep's incidences and a rule's nodes.
_NODE_TOL = 1e-12


def _integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"hemisphere: {name} must be an integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"hemisphere: {name} must be >= 1, got {value}.")
    return int(value)


@dataclasses.dataclass(frozen=True, eq=False)
class HemisphereRule:
    r"""A quadrature of the hemispherical average over emission directions.

    The nodes are ordered by polar angle, then azimuth. The weights sum to 1,
    so a constant emissivity :math:`e` averages to :math:`e`:

    .. math::
        e_h = \frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^{\pi/2}
        e(\theta,\varphi)\cos\theta\sin\theta\,d\theta\,d\varphi
        \approx \sum_k w_k\,e(\theta_k,\varphi_k).

    Build it with `hemisphere`; the arrays are read-only.

    Attributes:
        theta: ``(n,)`` polar angles, rad, ascending below :math:`\pi/2`.
        phi: ``(n,)`` azimuths, rad.
        weight: ``(n,)`` weights, summing to 1.
        mirror: Whether the azimuths are folded by a mirror symmetry.
    """

    theta: np.ndarray
    phi: np.ndarray
    weight: np.ndarray
    mirror: bool

    def __post_init__(self) -> None:
        for name in ("theta", "phi", "weight"):
            a = np.array(getattr(self, name), float)
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    def __repr__(self) -> str:
        return f"HemisphereRule({len(self.theta)} nodes, mirror={self.mirror})"


def hemisphere(
    n_theta: int, n_phi: int, *, rotation: int = 1, mirror: bool = False
) -> HemisphereRule:
    r"""A Gauss-Jacobi rule in :math:`\cos\theta` times an azimuth trapezoid.

    With :math:`\mu = \cos\theta` the hemispherical average is
    :math:`2\int_0^1 e\,\mu\,d\mu` for an emissivity :math:`e` without
    azimuth dependence. The polar nodes are the roots of the Jacobi
    polynomial :math:`P^{(0,1)}_{n_\theta}`, :math:`\mu_i = (1 + x_i)/2`, with
    weights :math:`w_i/2`: exact for :math:`e` a polynomial of degree
    :math:`2n_\theta - 1` in :math:`\mu`. The azimuths are
    :math:`2\pi j/n_\varphi` with weight :math:`1/n_\varphi`, exact for the
    harmonics below :math:`n_\varphi`. :math:`\theta = \pi/2` is never a node.

    ``rotation`` and ``mirror`` declare the symmetry of the emitter, which is
    yours to assert: ``rotation=r`` (a rotation by :math:`2\pi/r` leaves the
    cell invariant) folds the azimuths onto :math:`[0, 2\pi/r)`, and
    ``mirror=True`` onto :math:`[0, \pi/r]`. ``mirror=True`` asserts the
    reflection :math:`\varphi \to -\varphi`: a mirror plane that contains the
    lattice vector :math:`\hat e_1` and the surface normal (with
    ``rotation=r``, the planes at :math:`\varphi = j\pi/r` follow). A cell
    mirrored only about another plane, say the one containing
    :math:`\hat e_2` (:math:`\varphi \to \pi - \varphi`), is folded wrongly and
    gives a wrong :math:`e_h` without a message: use ``mirror=False``, or
    ``rotation=2, mirror=True`` when the cell has both mirror planes. Equal
    azimuths merge and add their weights.

    Args:
        n_theta: Polar nodes.
        n_phi: Azimuths before folding.
        rotation: Order :math:`r` of the rotation symmetry; ``n_phi`` must be
            a multiple of it.
        mirror: Whether a mirror plane is a symmetry.

    Returns:
        A `HemisphereRule`.

    Raises:
        TypeError: An argument is not an integer (``mirror``: a bool).
        ValueError: An argument is below 1, or ``n_phi`` is not a multiple of
            ``rotation``.
    """
    n_theta = _integer("n_theta", n_theta)
    n_phi = _integer("n_phi", n_phi)
    rotation = _integer("rotation", rotation)
    if not isinstance(mirror, (bool, np.bool_)):
        raise TypeError(f"hemisphere: mirror must be True or False, got {mirror!r}.")
    if n_phi % rotation:
        raise ValueError(
            f"hemisphere: n_phi={n_phi} is not a multiple of rotation={rotation}, "
            "so the azimuths are not invariant under a rotation by "
            f"2*pi/{rotation}. Use n_phi = "
            f"{rotation * max(1, round(n_phi / rotation))}."
        )
    x, w = roots_jacobi(n_theta, 0, 1)
    mu, wt = ((1 + x) / 2)[::-1], (w / 2)[::-1]  # theta ascending
    sector = n_phi // rotation  # azimuth steps per sector, integers
    counts: dict[int, int] = {}
    for j in range(n_phi):
        k = j % sector
        if mirror:
            k = min(k, sector - k)
        counts[k] = counts.get(k, 0) + 1
    steps = sorted(counts)
    az = 2 * math.pi * np.asarray(steps, float) / n_phi
    az_w = np.asarray([counts[k] for k in steps], float) / n_phi
    return HemisphereRule(
        theta=np.repeat(np.arccos(mu), len(steps)),
        phi=np.tile(az, n_theta),
        weight=np.repeat(wt, len(steps)) * np.tile(az_w, n_theta),
        mirror=bool(mirror),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class Emissivity:
    r"""Spectral emissivity per emission direction.

    Attributes:
        values: ``(n_incidence, n_freq)`` real.
        frequency: ``(n_freq,)`` Hz.
        theta: Emission polar angle of each row, rad (the sweep's
            :math:`\theta`).
        phi: Emission azimuth of each row, rad: the sweep's :math:`\varphi +
            \pi` wrapped to :math:`(-\pi, \pi]`.
        surface: The Floquet surface the light is emitted through.
        polarization: The polarization of the emitted light.
    """

    values: jax.Array
    frequency: np.ndarray
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    surface: str
    polarization: str

    def __repr__(self) -> str:
        return (
            f"Emissivity({len(self.theta)} directions on {self.surface!r}, "
            f"{self.polarization}, {len(self.frequency)} frequencies)"
        )


register_fields(Emissivity, ("values",))


def _check_sweep(sweep: Any, who: str) -> None:
    if not isinstance(sweep, AngleSweepResult):
        raise TypeError(
            f"{who} needs a gfem.diffraction.AngleSweepResult (from "
            f"gfem.diffraction.angle_sweep), got {type(sweep).__name__}: the "
            "emission directions are the sweep's incidences reversed. For one "
            "direction, sweep one angle: angle_sweep(..., theta=t, phi=p)."
        )


def _check_polarization(polarization: Any, who: str) -> None:
    if isinstance(polarization, str) and polarization in _POLARIZATIONS:
        return
    if isinstance(polarization, (tuple, list)) and len(polarization) == 2:
        raise ValueError(
            f"{who}: polarization={polarization!r} is a Jones pair; emissivity "
            "is defined here for 'TE', 'TM', 'RCP', 'LCP' and 'unpolarized'. "
            "gfem.diffraction.power_balance gives the absorptance of a Jones pair."
        )
    raise ValueError(
        f"{who}: polarization must be 'TE', 'TM', 'RCP', 'LCP' or "
        f"'unpolarized', got {polarization!r}."
    )


def _emit(sweep: Any, model: Model, polarization: Any, who: str) -> Emissivity:
    _check_sweep(sweep, who)
    _check_polarization(polarization, who)
    balance = balance_of(
        who,
        sweep,
        sweep.surface,
        polarization=polarization,
        model=model,
        order=(0, 0),
    )
    phi = []
    for p in sweep.phi:
        q = p + math.pi
        q -= 2 * math.pi * math.ceil((q - math.pi) / (2 * math.pi))
        phi.append(float(q))
    return Emissivity(
        values=balance.absorptance,
        frequency=balance.frequency,
        theta=tuple(sweep.theta),
        phi=tuple(phi),
        surface=sweep.surface,
        polarization=polarization,
    )


def emissivity(
    sweep: AngleSweepResult,
    *,
    model: Model,
    polarization: Literal["TE", "TM", "RCP", "LCP", "unpolarized"] = "unpolarized",
) -> Emissivity:
    r"""Spectral emissivity of a periodic emitter by Kirchhoff's law.

    For a body in thermal equilibrium that is reciprocal (no gyrotropic
    tensor, no gain), the emissivity into a direction and polarization equals
    the absorptance of a wave incident from that direction and polarization:

    .. math:: e(\theta, \varphi + \pi) = A(\theta, \varphi) = 1 - R - T -
       \mathrm{coupled},

    where the incidence :math:`(\theta, \varphi)` of the sweep is the
    reverse of the emission direction, :math:`(\theta, \varphi + \pi)`: the
    returned labels carry that azimuth. For a cell without a symmetry that
    maps :math:`\varphi` to :math:`\varphi + \pi` the two differ, so the label
    matters. Power delivered to lumped or wave ports (``coupled``) is not
    counted as absorbed in the body, so it is not emitted. TE
    maps to TE, TM to TM, RCP to RCP as seen along the emission direction;
    ``"unpolarized"`` is their mean. Small negative values from round-off are
    not clipped.

    The balance is complete only if every propagating order has a port, as
    for `power_balance`.

    Args:
        sweep: The angle sweep of the cell, whose incidences are the emission
            directions reversed.
        model: The model the sweep was solved with.
        polarization: ``"TE"``, ``"TM"``, ``"RCP"``, ``"LCP"`` or
            ``"unpolarized"``.

    Returns:
        An `Emissivity`.

    Raises:
        TypeError: ``sweep`` is not an `AngleSweepResult`, or as for
            `power_balance`.
        ValueError: ``polarization`` is not one of the five names (a Jones
            pair is refused), or as for `power_balance`.
        NonDifferentiableError: As for `power_balance`.
    """
    return _emit(sweep, model, polarization, "emissivity")


def hemispherical_emissivity(
    sweep: AngleSweepResult,
    rule: HemisphereRule,
    *,
    model: Model,
    polarization: Literal["TE", "TM", "RCP", "LCP", "unpolarized"] = "unpolarized",
) -> jax.Array:
    r"""The hemispherical emissivity :math:`e_h(f)` over a `HemisphereRule`.

    .. math:: e_h = \sum_k w_k\,e(\theta_k, \varphi_k),

    with static weights :math:`w_k`; the gradient needs one adjoint per
    incidence and frequency. The sweep must be solved at the rule's nodes, in
    order: ``angle_sweep(..., theta=rule.theta, phi=rule.phi)``.

    Args:
        sweep: The angle sweep over the rule's nodes.
        rule: The rule, from `hemisphere`.
        model: The model the sweep was solved with.
        polarization: As for `emissivity`; ``"RCP"`` and ``"LCP"`` are
            refused for a rule folded by a mirror, which turns one into the
            other.

    Returns:
        ``(n_freq,)`` real.

    Raises:
        TypeError: ``sweep`` or ``rule`` has the wrong type.
        ValueError: The polarization is invalid, the rule is mirror-folded and
            the polarization circular, or the sweep's incidences are not the
            rule's nodes.
    """
    who = "hemispherical_emissivity"
    _check_sweep(sweep, who)
    if not isinstance(rule, HemisphereRule):
        raise TypeError(
            "hemispherical_emissivity needs rule= a gfem.diffraction."
            "HemisphereRule (from gfem.diffraction.hemisphere), got "
            f"{type(rule).__name__}."
        )
    _check_polarization(polarization, who)
    if rule.mirror and polarization in ("RCP", "LCP"):
        raise ValueError(
            f"hemispherical_emissivity: polarization={polarization!r} with a "
            "rule folded by mirror=True: a mirror turns RCP into LCP, so the "
            f"folded nodes do not carry {polarization}. Build the rule with "
            "mirror=False, or use 'TE', 'TM' or 'unpolarized'."
        )
    n_s, n_r = len(sweep.theta), len(rule.theta)
    detail = None
    if n_s != n_r:
        detail = f"the sweep has {n_s} incidences, the rule {n_r} nodes"
    else:
        for i in range(n_s):
            ts, ps = sweep.theta[i], sweep.phi[i]
            tr, pr = float(rule.theta[i]), float(rule.phi[i])
            if abs(ts - tr) > _NODE_TOL or abs(ps - pr) > _NODE_TOL:
                detail = (
                    f"incidence {i} is (theta, phi) = ({ts:.6g}, {ps:.6g}) rad, "
                    f"node {i} is ({tr:.6g}, {pr:.6g}) rad"
                )
                break
    if detail is not None:
        raise ValueError(
            "hemispherical_emissivity: the sweep's incidences are not the "
            f"rule's nodes: {detail}. Sweep the rule's nodes: "
            "angle_sweep(..., theta=rule.theta, phi=rule.phi)."
        )
    e = _emit(sweep, model, polarization, who)
    return jnp.tensordot(jnp.asarray(rule.weight), e.values, axes=1)
