r"""Closed forms and cases for the multipole decomposition; nothing here solves.

Bohren and Huffman's Mie coefficients in gfem's convention
(:math:`e^{+i\omega t}`: the conjugate of their :math:`a_n, b_n` at the index
:math:`m = \sqrt{\bar\varepsilon\mu}`), gfem's map from them to the spherical
coefficients of a sphere at the origin under
``PlaneWave(theta=0, phi=0, polarization="TM")`` (:math:`E_0\hat x e^{-ikz}`),

.. math::

    \alpha_{l,\pm1} = \pm(-i)^l\sqrt{\pi(2l+1)}\,\bar a_l, \qquad
    \beta_{l,\pm1} = -(-i)^l\sqrt{\pi(2l+1)}\,\bar b_l,

all other :math:`m` zero, the error metric the gates use, and the case table
of the reference spheres.

The metric is the degree error over the whole expansion:
:math:`E_l = \lVert(\alpha_l, \beta_l) - \text{Mie}\rVert / \lVert(\alpha,
\beta)\rVert_\text{all}`, both types and all :math:`m` of degree :math:`l`
over the norm of the reference expansion; the leak is the same norm of the
:math:`m \ne \pm1` entries of degree :math:`\le 2`. A relative error per
type fails on a coefficient that is nearly zero (the anapole's
:math:`a_1`); this does not.

Measured floors (order 3, the default free-space sphere and the prototype
meshes) are recorded here as the verification tests measure them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import spherical_jn, spherical_yn

#: The case table's reference tolerances live with the tests that use them.


@dataclass(frozen=True)
class SphereCase:
    """A reference sphere: relative permittivity, size parameter, name.

    ``x`` is :math:`ka` of the sphere's volume-equivalent radius; ``eps`` is
    gfem's relative permittivity (negative imaginary part for loss) and
    ``mu`` its relative permeability.
    """

    name: str
    eps: complex | None
    x: float
    mu: float = 1.0


#: A: the default free-space sphere (:math:`a = 0.25` um, :math:`\lambda_0 = 1`
#: um); C: the first Kerker point (:math:`a_1 = b_1`) of an index-3.5 sphere;
#: D': the first anapole (:math:`a_1 = 0`) of an index-4 sphere. C and D' are
#: the prior-art reference values; the lossy sphere L is the default tier's.
CASES = {
    "A": SphereCase("A", 1.5**2 + 0j, 1.0),
    "L": SphereCase("L", 4.0 - 1.0j, math.pi / 2),
    "C": SphereCase("C", 3.5**2 + 0j, 0.783916362855),
    "D'": SphereCase("D'", 4.0**2 + 0j, 1.165407223827),
    "PEC": SphereCase("PEC", None, 1.0),
    "MU": SphereCase("MU", 2.0 + 0j, 0.8, mu=2.0),
}

#: The long-wavelength moments at the anapole D', in units of
#: :math:`6\pi\varepsilon_0 E_0 / k^3` (Mie internal field quadrature):
#: :math:`p_{0,x}` and :math:`-ikT_x`; the exact :math:`p_x` is zero.
ANAPOLE_LONG_WAVELENGTH = {"D'": (0.287088, -0.306525), "D": (0.417177, -0.454942)}


def _riccati(n: np.ndarray, z: complex):
    """:math:`\\psi_n, \\psi_n'` of the Riccati-Bessel function :math:`z j_n(z)`."""
    return z * spherical_jn(n, z), spherical_jn(n, z) + z * spherical_jn(
        n, z, derivative=True
    )


def mie_coefficients(
    eps: complex | None, x: float, lmax: int, *, mu: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""gfem's Mie coefficients :math:`\bar a_n, \bar b_n`, :math:`n = 1..` ``lmax``.

    Bohren and Huffman 4.53 with the sphere's permeability :math:`\mu_1 =` ``mu``
    (the background's is 1) at :math:`m = \sqrt{\bar\varepsilon\,\mu}`, then
    conjugated into :math:`e^{+i\omega t}`. ``eps=None`` is the perfectly
    conducting sphere, :math:`a_n = \psi_n'(x)/\xi_n'(x)`,
    :math:`b_n = \psi_n(x)/\xi_n(x)`.
    """
    n = np.arange(1, lmax + 1)
    psi, dpsi = _riccati(n, x)
    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    dh = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    xi, dxi = x * h, h + x * dh
    if eps is None:
        a, b = dpsi / dxi, psi / xi
    else:
        m = np.sqrt(np.conj(complex(eps)) * mu)
        psim, dpsim = _riccati(n, m * x)
        a = (m * psim * dpsi - mu * psi * dpsim) / (m * psim * dxi - mu * xi * dpsim)
        b = (mu * psim * dpsi - m * psi * dpsim) / (mu * psim * dxi - m * xi * dpsim)
    return np.conj(a), np.conj(b)


def mie_multipoles(
    eps: complex | None, x: float, lmax: int, *, mu: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""gfem's :math:`\alpha_{lm}, \beta_{lm}` of a Mie sphere, flat.

    The sphere sits at the model origin under :math:`E_0\hat x e^{-ikz}`;
    the coefficients are per :math:`E_0`. A sphere centred at :math:`r_o` has
    them times :math:`e^{-ik\hat k\cdot r_o}`.
    """
    a, b = mie_coefficients(eps, x, lmax, mu=mu)
    alpha = np.zeros(lmax * (lmax + 2), complex)
    beta = np.zeros_like(alpha)
    for l in range(1, lmax + 1):
        c = (-1j) ** l * math.sqrt(math.pi * (2 * l + 1))
        for m, sign in ((1, 1), (-1, -1)):
            j = l * l + l + m - 1
            alpha[j] = sign * c * a[l - 1]
            beta[j] = -c * b[l - 1]
    return alpha, beta


def degree_errors(
    alpha: np.ndarray, beta: np.ndarray, ref_alpha: np.ndarray, ref_beta: np.ndarray
) -> np.ndarray:
    r""":math:`E_l` for :math:`l = 1..` ``lmax``, ``(lmax,)``.

    ``alpha`` and ``beta`` are one frequency's flat coefficients; the
    reference may be longer or shorter in degree (the common ones are used,
    the norm is the reference's whole).
    """
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    n = min(alpha.size, ref_alpha.size)
    lmax = round(math.sqrt(n + 1) - 1)
    norm = math.sqrt(np.sum(np.abs(ref_alpha) ** 2) + np.sum(np.abs(ref_beta) ** 2))
    out = np.empty(lmax)
    for l in range(1, lmax + 1):
        s = slice((l - 1) * (l + 1), l * (l + 2))
        out[l - 1] = (
            math.sqrt(
                np.sum(np.abs(alpha[s] - ref_alpha[s]) ** 2)
                + np.sum(np.abs(beta[s] - ref_beta[s]) ** 2)
            )
            / norm
        )
    return out


def leak(
    alpha: np.ndarray, beta: np.ndarray, ref_alpha: np.ndarray, ref_beta: np.ndarray
) -> float:
    r"""The :math:`m \ne \pm1` content of degrees :math:`\le 2` over the reference norm.

    The Mie map is zero there for :math:`E_0\hat x e^{-ikz}`.
    """
    norm = math.sqrt(np.sum(np.abs(ref_alpha) ** 2) + np.sum(np.abs(ref_beta) ** 2))
    off = [
        j
        for l in (1, 2)
        for m in range(-l, l + 1)
        if abs(m) != 1
        for j in [l * l + l + m - 1]
    ]
    return (
        math.sqrt(
            np.sum(np.abs(np.asarray(alpha)[off]) ** 2)
            + np.sum(np.abs(np.asarray(beta)[off]) ** 2)
        )
        / norm
    )
