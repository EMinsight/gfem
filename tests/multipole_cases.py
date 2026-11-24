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

Measured floors (order 3; the gates of `tests/verification/test_multipoles.py`
are at least three times these). The lossy sphere L at sphere maxh 0.06 in a
layer 0.5 um thick (maxh 0.25, 7590 tetrahedra). On the free-space mesh
(sphere maxh 0.2, layer 0.35) :math:`E_1 = 9.9\times10^{-3}` and the partial
cross sections are 1 % off. Two floors, each measured on its own:

- The layer. At 0.35 um (under two elements of maxh 0.25) its reflection,
  scattered back by the sphere, leaves about :math:`5\times10^{-3}` of the
  norm in degree 1 whatever the sphere's mesh (sphere maxh 0.1 to 0.05):
  order 4 on the same mesh does not lower it, the target reflection does
  move it (1e-3, 1e-6, 1e-8 on one mesh: :math:`E_1` of the axial wave
  6.2e-3, 4.8e-3, 1.9e-3); 0.5 um brings it to 5e-4 to 1e-3.
- The faceted sphere. :math:`E_2` is 1.6e-3 at sphere maxh 0.1, 1e-3 at
  0.07, 4e-4 at 0.06 and 0.05, whatever the layer.

The relative error of the electric partial cross section of degree 2 (5.7 %
of the total; its coefficients are 0.24 of the norm) is up to eight times
:math:`E_2`, and only the part of the error in phase with the coefficients
counts, which depends on the incidence: on the free-space layer the axial wave
polarized along :math:`y` is off by 1.1 %, along :math:`x` by 0.4 %, and over
the TE and TM waves at 0, 30 and 90 degrees the worst is 1.9 %, with the
projection exact (the sum matches `gfem.cross_sections`, itself off by the
same amount against Mie) and its error spread evenly over :math:`m`. On the
mesh here:

- :math:`E_1, E_2, E_3`: 5.2e-4, 4.0e-4, 1.5e-4; leak 3.6e-4; exact
  :math:`p_x` and :math:`m_y` against the Mie closed form 3.4e-4, 7.2e-4.
- Sum of the partial cross sections against `gfem.cross_sections`: -2.1e-4
  (z), -4.8e-4 (te), -2.0e-4 (tm).
- Partial cross sections against Mie at 30 degrees and on the axis, where at
  least 1 % of the total: up to 1.5e-3; over the TE and TM waves at 0, 30
  and 90 degrees, 1.6e-3; sphere maxh 0.058 to 0.062 with layers 0.5 to
  0.65 um stay under 4.7e-3.
- Complex amplitude against the real one: 1.1e-14. Wide beam against the
  axial wave: :math:`E_1` 6.3e-6, sum -1.4e-3 (the beam's non-uniformity
  over the sphere, the same on every mesh).
- Origin shift (30, -20, 50) nm: total 3e-14 at the default degree; the
  electric dipole share moves by -12 %.
- PEC sphere: :math:`E_1, E_2` 1.1e-3, 6.9e-4, leak 5.6e-4, sum -2.2e-4.
  :math:`\varepsilon_r = \mu_r = 2`: 7.9e-4, 6.1e-4, 6.2e-4, -1.3e-5.
- Dipoles in the box of 7 cells per um, layer 3/7 um, order 3: moments back
  1.2e-4 (p), 1.4e-4 (m) in vacuum, 2.6e-4, 3.1e-4 in
  :math:`\varepsilon_b = 2, \mu_b = 3`; content of degree 2 and above 7.7e-4,
  9.4e-4 (1.4e-3, 1.8e-3 in the medium); total power about the shifted
  origin 9e-9; against the far field's radiated power +1.0e-4, -1.4e-4. The
  content of degree 2 and above is the air's discretization around the
  dipole: 1.9e-3 at 6 cells per um, 1.3e-4 at order 4 on them, and the same
  digits with layers of 2 to 4 cells (the projection's layer integral keeps
  only what radiates from inside it). The radiated power does see the
  layer: 4.4e-4 with 2 cells of 1/7 um.
- Kerker sphere C (maxh 0.25, sphere 0.05): :math:`E_1, E_2` 7.2e-3,
  5.8e-4, leak 1.8e-3, sum 1.6e-3, :math:`|p_x - m_y/c|/|p_x|` 5.2e-3.

- Induced-current route against the projection, lossy sphere, degree 1, both
  types over the degree-1 norm: 2.6e-3 on the free-space default mesh
  (maxh 0.4, sphere 0.2), 6.6e-4 at maxh 0.25, 2.4e-4 at maxh 0.2 with the
  sphere at 0.1 (the mesh V6 uses; gate 1e-3). The difference is the far-field
  layer's discretization, not the volume integral.
- Toroidal correction, same sphere: :math:`|p_0 - ikT - p| / |p_0 - p| =
  0.066`, gate 0.2.
- Anapole D' (maxh 0.25, sphere 0.05, order 3): exact dipole 2.3e-3 of the
  norm; :math:`p_{0,x}` 0.28661 and :math:`-ikT_x` -0.30678 against 0.287088
  and -0.306525; :math:`|p_0 - ikT| / |p_0| = 0.070`. Gates 1e-2.
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
    r""":math:`\psi_n, \psi_n'` of the Riccati-Bessel function :math:`z j_n(z)`."""
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
