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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pml_cases as pc
import scipy.constants as const
import sphere_mesh
from scipy.special import spherical_jn, spherical_yn

import gfem

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


# --- meshes and models --------------------------------------------------------------

#: The cube's permittivity in `portless_scatterer_model`.
CUBE_EPS = 3.0
#: The centre of the one-cell scatterer of `cube` before any translation, um.
CUBE_CENTRE = 0.375


def _cube_volume(c: np.ndarray) -> str:
    if ((c > 0.25) & (c < 0.5)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _cube_feed(c: np.ndarray) -> str:
    on = abs(c[0] - 0.5) < 1e-9 and 0.25 < c[1] < 0.5 and 0.25 < c[2] < 0.5
    return "feed" if on else ""


def _cube_shell(c: np.ndarray) -> str:
    inside = ((c > 0.25 - 1e-9) & (c < 0.5 + 1e-9)).all()
    on = (abs(c - 0.25) < 1e-9) | (abs(c - 0.5) < 1e-9)
    return "feed" if inside and on.any() else ""


def cube(
    path: str | Path,
    *,
    centred: bool = False,
    cells: int = 4,
    centre: tuple[float, float, float] | None = None,
    feed: bool = False,
    shell: bool = False,
) -> Path:
    """A one-cell cube "scat" in air "air" wrapped in a layer "pml", um.

    The cube is the cell :math:`[0.25, 0.5]^3` of the unit box (side
    :math:`0.25`, circumradius :math:`0.2165` um) on `pml_cases.box`'s grid
    with a layer of a quarter of a micrometre on every side. ``centred``
    translates the whole mesh so that the cube's centre is the origin;
    ``centre`` puts it at the given point (um). The same mesh, translated.
    With ``feed`` the cube's :math:`+x` face is the surface region "feed", to
    hang a port on; with ``shell`` all its faces are.
    """
    target = (0.0, 0.0, 0.0) if centred else centre
    mapping: Callable[[np.ndarray], np.ndarray] | None = None
    if target is not None:
        shift = np.asarray(target, float) - CUBE_CENTRE
        mapping = lambda p: p + shift  # noqa: E731
    return pc.box(
        path,
        pc.SIDES,
        thickness=0.25,
        cells=cells,
        volume=_cube_volume,
        surface=_cube_shell if shell else _cube_feed if feed else None,
        mapping=mapping,
    )


def portless_scatterer_model(
    path: str | Path, *, field: bool = False, eps: float = CUBE_EPS
) -> gfem.Model:
    """The model of `cube`'s mesh: air background, the cube of permittivity ``eps``.

    ``eps=1`` leaves nothing of the scatterer (the empty-scatterer cases);
    with ``field`` the cube is a `gfem.MaterialField` of that value, which is
    a scatterer region whatever its value.
    """
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    air = gfem.Material()
    scat = (
        gfem.MaterialField(eps_r=np.full(len(mesh.elements("scat")), eps))
        if field
        else gfem.Material(eps_r=eps)
    )
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air, "scat": scat},
        pml=gfem.PML("pml"),
        background=air,
    )


def sphere_case(
    path: str | Path,
    case: SphereCase,
    *,
    maxh: float = 0.25,
    sphere_maxh: float = 0.05,
    thickness: float = 0.35,
) -> tuple[gfem.Mesh, float, float]:
    r"""The case's mirrored sphere: mesh, :math:`a_\mathrm{vol}` (um), frequency (Hz).

    The sphere of radius :math:`x/(2\pi)` um in an air cube of half-side
    :math:`a + 0.4` with its refinement box at :math:`a + 0.12`, the layer
    ``thickness`` thick; the frequency puts :math:`k a_\mathrm{vol}` of the
    meshed sphere's volume-equivalent radius on the case's :math:`x`.
    """
    a = case.x / (2 * math.pi)
    volume = sphere_mesh.write(
        path,
        maxh=maxh,
        sphere_maxh=sphere_maxh,
        mirrored=True,
        radius=a,
        half_side=a + 0.4,
        inner=a + 0.12,
        thickness=thickness,
    )
    a_vol = (3 * volume / (4 * math.pi)) ** (1 / 3)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    return mesh, a_vol, case.x * const.c / (2 * math.pi * a_vol * 1e-6)
