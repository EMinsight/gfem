r"""The Gaussian beam's field: the Huygens complex-focus beam, on the host.

In the frame of the beam (focus at the origin, axis :math:`\hat z`,
polarization :math:`\hat x`), in a homogeneous lossless background of
wavenumber :math:`k` and impedance :math:`\eta`, the field is the electric
:math:`\hat x` plus magnetic :math:`\hat y` dipole field of
:math:`\psi = j_0(kR)` at the complex point :math:`-ib\hat z`:

.. math::

    E = C e^{-a}\left[k^2\hat x\,\psi + \nabla\partial_x\psi
        - ik\,\nabla\times(\hat y\,\psi)\right],
    \qquad R^2 = x^2 + y^2 + (z + ib)^2,

with :math:`b = k w_0^2/2` the Rayleigh range and :math:`a = kb`. It is an
exact solution of Maxwell's equations, entire in :math:`R^2` (no branch
cut, no singular point), and exactly the superposition of real propagating
plane waves

.. math::

    E = \frac{C k^2}{4\pi}\int_{S^2} (1 + \cos\theta)\,e^{a(\cos\theta - 1)}
        (\cos\phi\,\hat\theta - \sin\phi\,\hat\phi)\,
        e^{-ik\hat s\cdot r}\,d\Omega,

whose paraxial limit is the fundamental Gaussian beam of waist :math:`w_0`.
A paraxial beam is not used: without its longitudinal field it leaves a
residual source of order :math:`1/(k w_0)` in the background.

With :math:`F_c(u) = e^{c(u-1)}[(1+u)^2/c - 2(1+u)/c^2 + 2/c^3]`, the power
through any plane normal to the axis and the focal field are

.. math::

    P = \frac{\pi k^2 C^2}{4\eta} J, \quad
    J = F_{2a}(1) - 2F_{2a}(0) + F_{2a}(-1), \qquad
    E_x(0) = \frac{C k^2}{4} I, \quad I = F_a(1) - F_a(-1),

so :math:`C = \sqrt{4\eta P/(\pi k^2 J)}` carries :math:`P` = 1 W. Below
:math:`k w_0 = 0.2` the terms of :math:`J` cancel to round-off, so such
waists are refused.

Evaluation: with :math:`u = R^2`, :math:`X = k\sqrt u` (either branch, the
results are even in :math:`X`), :math:`\psi' = -\tfrac{k^2}{2}g_1`,
:math:`\psi'' = \tfrac{k^4}{4}g_2`, :math:`g_l = j_l(X)/X^l`,

.. math::

    E_x = k^2 g_0 + 2\psi' + 4x^2\psi'' + 2ik(z+ib)\psi', \quad
    E_y = 4xy\psi'', \quad
    E_z = 4x(z+ib)\psi'' - 2ikx\psi'.

The factor :math:`e^{-a}` goes inside the sines and cosines,
:math:`e^{-a}\sin X = (e^{iX-a} - e^{-iX-a})/(2i)`, which never exceed 1
since :math:`|\operatorname{Im}X| \le a`; near the ring :math:`R = 0`,
where the closed forms are :math:`0/0`, the :math:`g_l` are their Taylor
series in :math:`X^2`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.constants as const

from gfem._sources import jones, propagation_basis
from gfem.conventions import INCIDENT_POWER

if TYPE_CHECKING:
    from gfem._background import Background
    from gfem._sources import GaussianBeam

__all__ = [
    "check_background",
    "check_waist",
    "focal_field",
    "impedance",
    "values",
]

#: The smallest :math:`k w_0` a beam is evaluated at: below it the power
#: integral loses more than :math:`10^{-12}` to cancellation.
SMALLEST_KW0 = 0.2
#: Where the closed forms switch to the series: :math:`|X^2|` below it.
_SERIES = 0.25
_TERMS = 10


def _series(order: int) -> np.ndarray:
    r"""Taylor coefficients of :math:`j_l(X)/X^l` in :math:`X^2`."""
    return np.array(
        [
            (-1) ** n
            / (2**n * math.factorial(n) * math.prod(range(1, 2 * n + 2 * order + 2, 2)))
            for n in range(_TERMS)
        ]
    )


_COEFFICIENTS = tuple(_series(order) for order in range(3))


def check_background(source: GaussianBeam, background: Background) -> Any:
    """The beam's medium: the background's one lossless layer.

    Raises:
        NotImplementedError: A background of several layers or with a
            ground plane, or a lossy one.
    """
    if len(background.layers) > 1 or background.ground is not None:
        raise NotImplementedError(
            f"The GaussianBeam {source.name!r} is solved in a homogeneous "
            "background; a beam in a layered background (several layers or a "
            "ground plane) is not solved in this gfem version. Give its field "
            "as a gfem.IncidentField, or read the coupling of a beam with "
            "gfem.beam_coupling."
        )
    material = background.layers[0]
    if np.any(material.tan_delta) or np.any(material.conductivity):
        raise NotImplementedError(
            f"The GaussianBeam {source.name!r} travels in a lossy background "
            f"({material!r}); its 1 W is a power through planes across its "
            "axis, conserved in a lossless background only. Give the background "
            "without tan_delta or conductivity."
        )
    return material


def _medium(
    source: GaussianBeam, background: Background, frequency: float
) -> tuple[float, float]:
    """The wavenumber (rad/m) and impedance (ohm) of the beam's medium."""
    material = check_background(source, background)
    eps, mu = float(material.eps_r), float(material.mu_r)
    k = 2 * math.pi * frequency / const.c * math.sqrt(eps * mu)
    return k, const.mu_0 * const.c * math.sqrt(mu / eps)


def impedance(source: GaussianBeam, background: Background) -> float:
    """The wave impedance of the beam's medium, ohm."""
    return _medium(source, background, 1.0)[1]


def check_waist(source: GaussianBeam, background: Background, frequency: float) -> None:
    """Raise if the beam's waist is too small to normalize at ``frequency``.

    Raises:
        NotImplementedError: As `check_background`.
        ValueError: :math:`k w_0` below `SMALLEST_KW0`.
    """
    k, _ = _medium(source, background, frequency)
    kw0 = k * source.waist
    if not kw0 >= SMALLEST_KW0:
        raise ValueError(
            f"The GaussianBeam {source.name!r} has k w0 = {kw0:.3g} at "
            f"{frequency:.6g} Hz (waist {source.waist:.6g} m), below "
            f"{SMALLEST_KW0}: its power cannot be normalized to round-off. Give "
            "a waist of at least 0.2 / k, or the field as a gfem.IncidentField."
        )


def _antiderivative(c: float, u: float) -> float:
    r""":math:`F_c(u)`, the antiderivative of :math:`(1+u)^2 e^{c(u-1)}`."""
    return math.exp(c * (u - 1)) * ((1 + u) ** 2 / c - 2 * (1 + u) / c**2 + 2 / c**3)


def _amplitude(k: float, eta: float, waist: float) -> tuple[float, float]:
    """:math:`C` for 1 W and the focal field :math:`C k^2 I / 4`, V/m."""
    a = (k * waist) ** 2 / 2
    j = (
        _antiderivative(2 * a, 1)
        - 2 * _antiderivative(2 * a, 0)
        + _antiderivative(2 * a, -1)
    )
    i = _antiderivative(a, 1) - _antiderivative(a, -1)
    c = math.sqrt(4 * eta * INCIDENT_POWER / (math.pi * k**2 * j))
    return c, c * k**2 * i / 4


def focal_field(
    source: GaussianBeam, background: Background, frequency: float
) -> float:
    """The peak field at the focus, V/m (`GaussianBeam.focal_field`)."""
    check_waist(source, background, frequency)
    k, eta = _medium(source, background, frequency)
    return _amplitude(k, eta, source.waist)[1]


def _scaled(x2: np.ndarray, a: float) -> tuple[np.ndarray, ...]:
    r""":math:`e^{-a} j_l(X)/X^l`, :math:`l = 0, 1, 2`, entire in :math:`X^2`."""
    small = np.abs(x2) < _SERIES
    big = np.where(small, 1.0, x2)  # the closed forms away from the ring
    x = np.sqrt(big)
    plus, minus = np.exp(1j * x - a), np.exp(-1j * x - a)
    sin, cos = (plus - minus) / 2j, (plus + minus) / 2
    closed = (
        sin / x,
        (sin / x - cos) / big,
        ((3 / big - 1) * sin / x - 3 * cos / big) / big,
    )
    scale = math.exp(-a)
    return tuple(
        np.where(small, scale * np.polyval(coefficients[::-1], x2), value)
        for coefficients, value in zip(_COEFFICIENTS, closed, strict=True)
    )


def _frame_field(points: np.ndarray, k: float, waist: float) -> np.ndarray:
    r"""The beam along :math:`+z`, along :math:`\hat x`, focused at the origin.

    With :math:`C = 1`.
    """
    x, y, z = points.T
    b = k * waist**2 / 2
    zc = z + 1j * b
    g0, g1, g2 = _scaled(k**2 * (x * x + y * y + zc * zc), k * b)
    d1 = -(k**2) / 2 * g1
    d2 = k**4 / 4 * g2
    return np.stack(
        [
            k**2 * g0 + 2 * d1 + 4 * x * x * d2 + 2j * k * zc * d1,
            4 * x * y * d2,
            4 * x * zc * d2 - 2j * k * x * d1,
        ],
        axis=1,
    )


def values(
    source: GaussianBeam, background: Background, points: np.ndarray, frequency: float
) -> np.ndarray:
    """The beam's electric field at ``points`` (``(n, 3)``, m), ``(n, 3)`` V/m.

    Raises:
        NotImplementedError: As `check_background`.
        ValueError: As `check_waist`.
    """
    check_waist(source, background, frequency)
    k, eta = _medium(source, background, frequency)
    c, _ = _amplitude(k, eta, source.waist)
    k_hat, theta_hat, phi_hat = propagation_basis(source.theta, source.phi)
    # columns: the beam frame's x (TM), y (TE) and axis in model coordinates
    frame = np.stack([theta_hat, phi_hat, k_hat], axis=1)
    local = (np.asarray(points, float).reshape(-1, 3) - np.array(source.focus)) @ frame
    a_tm, a_te = jones(source.polarization)
    field = np.zeros((len(local), 3), complex)
    if a_tm:
        field += a_tm * _frame_field(local, k, source.waist)
    if a_te:
        # the TE beam is the TM one turned by 90 degrees about the axis
        turned = _frame_field(
            np.stack([local[:, 1], -local[:, 0], local[:, 2]], axis=1), k, source.waist
        )
        field += a_te * np.stack([-turned[:, 1], turned[:, 0], turned[:, 2]], axis=1)
    return c * field @ frame.T
