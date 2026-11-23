r"""Free-space models shared by the source tests: meshes and closed forms.

`dipole_box` writes a cube of "air" wrapped in perfectly matched layer cells
on every side; `image_power` is the emitted power of a dipole above a
perfect conductor or a perfect magnetic wall by image theory, `bulk_power`
the closed form gfem's Purcell factor divides by, written out here as an
independent check. Every builder takes the file to write; nothing here
solves. Lengths in um (the mesh unit), the vacuum wavelength 1 um.

For the scattering cases, `sphere` writes a sphere ("sphere", radius
`RADIUS`) in a cube of air wrapped in a perfectly matched layer
(`sphere_mesh.write`, netgen: unstructured, straight tetrahedra), read back
through `gfem.Mesh.from_gmsh`; `mie` gives the Bohren-Huffman series it is
compared with.

Straight tetrahedra cut the sphere's surface, so the closed forms are
evaluated at the volume-equivalent radius :math:`a_\mathrm{vol}` of the
meshed sphere: against the nominal radius the faceted sphere is 2-3 % off,
against :math:`a_\mathrm{vol}` it matches a curved mesh to 0.1 % once the
sphere is resolved. Order 2 is not usable at these sizes (-4 % lossy, -10 %
lossless: :math:`\varepsilon = 4` halves the wavelength inside), so the
gates run at order 3.

Measured errors against Mie at :math:`a_\mathrm{vol}` (extinction,
absorption, scattering), plane wave at 30 degrees, order 3, on the mirrored
mesh (`sphere_mesh.write`):

- `MAXH`, sphere at `SPHERE_MAXH` (about 2000 tetrahedra): lossy
  :math:`\varepsilon = 4 - i` TE +0.07 / -0.08 / +0.16 %, TM -0.22 / -0.43 /
  -0.10 %; lossless :math:`\varepsilon = 4` -0.25 % (TE), -0.56 % (TM); PEC
  +1.99 % (TE), +1.44 % (TM). Gates `TOLERANCE` and `PEC_TOLERANCE`. The half
  model with its mirror wall equals the whole to :math:`10^{-13}`.
- Gaussian beams of waist 3 um focused on the lossy sphere, same directions,
  over their focal intensity, against the plane waves: -0.24 / -0.47 /
  -0.11 % (TE and TM alike), the beam's non-uniformity over the sphere.
  Gate `BEAM_TOLERANCE`.
- Sphere refined to `REFINED_MAXH` (:math:`\lambda_\mathrm{in}/6`): gate
  `REFINED_TOLERANCE`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pml_cases as pc
import sphere_mesh
from scipy.special import spherical_jn, spherical_yn

import gfem

#: Speed of light and vacuum permeability, SI (CODATA 2018: gfem's may be a
#: later value, 1e-9 apart).
C0 = 299792458.0
MU0 = 1.25663706212e-6
#: The vacuum wavelength (um) and its frequency (Hz).
WAVELENGTH = 1.0
F0 = C0 / (WAVELENGTH * 1e-6)
OMEGA0 = 2 * math.pi * F0
K0 = 2 * math.pi / (WAVELENGTH * 1e-6)
UM = 1e-6

#: The sphere's nominal radius, um: :math:`ka = \pi/2`.
RADIUS = sphere_mesh.RADIUS
#: The global and the sphere's maximal edge lengths, um.
MAXH = sphere_mesh.MAXH
SPHERE_MAXH = sphere_mesh.SPHERE_MAXH
REFINED_MAXH = 0.08
#: Relative gates on each cross section (see the module docstring).
TOLERANCE = 2e-2
PEC_TOLERANCE = 3e-2
REFINED_TOLERANCE = 1e-2
#: Relative gate on a Gaussian beam's cross sections (waist 3 um, focused on
#: the sphere) against the plane wave's (see the module docstring).
BEAM_TOLERANCE = 2e-2
#: The lossy and lossless spheres' materials, :math:`\varepsilon = 4 - i`
#: and 4 under :math:`e^{+i\omega t}`.
LOSSY = gfem.Material(eps_r=4.0, tan_delta=0.25)
LOSSLESS = gfem.Material(eps_r=4.0)
#: The incidence: 30 degrees from +z in the xz plane.
THETA = math.radians(30.0)
SOURCES = (
    gfem.PlaneWave("te", theta=THETA, phi=0.0, polarization="TE"),
    gfem.PlaneWave("tm", theta=THETA, phi=0.0, polarization="TM"),
)


def dipole_box(
    path: str | Path,
    *,
    cells: int = 4,
    thickness: float = 0.5,
    volume=None,
    surface=None,
) -> Path:
    """The unit cube "air" ([0, 1]^3 um) in "pml" cells on all six sides.

    Args:
        path: The gmsh 2.2 file to write.
        cells: Cells per um.
        thickness: The layer thickness, um, a multiple of ``1 / cells``.
        volume: Region name of a cell from its centroid (``None`` removes
            it); by default "air" inside the cube and "pml" outside.
        surface: Name of a boundary face from its centroid (see
            `pml_cases.box`).

    Returns:
        ``path``.
    """
    return pc.box(
        path,
        sides=pc.SIDES,
        thickness=thickness,
        cells=cells,
        volume=volume,
        surface=surface,
    )


def bulk_power(
    omega: float,
    moment: np.ndarray,
    *,
    magnetic: bool = False,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> float:
    r"""A dipole's power in the unbounded lossless medium, W.

    Electric: :math:`\mu_0\mu_r n \omega^4 |p|^2 / (12\pi c)`; magnetic:
    :math:`\mu_0\varepsilon_r n\omega^4 |m|^2 / (12\pi c^3)`.
    """
    n = math.sqrt(eps_r * mu_r)
    size = float(np.vdot(moment, moment).real)
    if magnetic:
        return MU0 * eps_r * n * omega**4 * size / (12 * math.pi * C0**3)
    return MU0 * mu_r * n * omega**4 * size / (12 * math.pi * C0)


def image_power(height: float, *, normal: bool, sign: int) -> float:
    r"""An electric dipole's power above a mirror over its vacuum power.

    The image at :math:`-h` has the same normal and the opposite tangential
    moment over a perfect conductor (``sign = +1``), the opposite ones over
    a perfect magnetic wall (``sign = -1``). With :math:`x = 2 k h`, the
    mutual term of two parallel dipoles is (normal, along the line between
    them) :math:`3 (\sin x - x\cos x) / x^3`, (tangential, across it)
    :math:`\tfrac{3}{2}[(1/x - 1/x^3)\sin x + \cos x / x^2]`.

    Args:
        height: The dipole's height above the mirror, in wavelengths.
        normal: The moment is normal to the mirror (else tangential).
        sign: +1 over a perfect conductor, -1 over a perfect magnetic wall.
    """
    x = 2 * 2 * math.pi * height
    if normal:
        mutual = 3 * (math.sin(x) - x * math.cos(x)) / x**3
        return 1 + sign * mutual
    mutual = 1.5 * ((1 / x - 1 / x**3) * math.sin(x) + math.cos(x) / x**2)
    return 1 - sign * mutual


#: The dielectric cube's permittivity in `scatterer_model`.
SCATTERER_EPS = 3.0
#: In the air of `scatterer_box`, outside the cube, off every grid face.
DIPOLE_POSITION = (0.13e-6, 0.41e-6, 0.87e-6)


def _scatterer_volume(c: np.ndarray) -> str:
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _scatterer_feed(c: np.ndarray) -> str:
    """The cube's +x face, a lumped port's surface."""
    on = abs(c[0] - 0.75) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75
    return "feed" if on else ""


def scatterer_box(path: str | Path, *, cells: int = 4, thickness: float = 0.5) -> Path:
    """A dielectric cube "scat" in the middle of `dipole_box`, its +x face "feed".

    With ``cells=8`` a shell of air elements, untouched by the layer, holds
    the cube: the truncation layer is background (``thickness`` as in
    `dipole_box`).
    """
    return pc.box(
        path,
        sides=pc.SIDES,
        thickness=thickness,
        cells=cells,
        volume=_scatterer_volume,
        surface=_scatterer_feed,
    )


def scatterer_model(path: str | Path, *, field: bool = False):
    """The model of `scatterer_box`: air background, a 50 ohm port on "feed".

    With ``field`` the cube is a `gfem.MaterialField` of the same value.
    """
    mesh = gfem.Mesh.from_gmsh(path)
    air = gfem.Material()
    n = len(mesh.elements("scat"))
    scat = (
        gfem.MaterialField(eps_r=np.full(n, SCATTERER_EPS))
        if field
        else gfem.Material(eps_r=SCATTERER_EPS)
    )
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air, "scat": scat},
        ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
        pml=gfem.PML("pml"),
        background=air,
    )


def sphere(
    path: str | Path,
    *,
    maxh: float = MAXH,
    sphere_maxh: float = SPHERE_MAXH,
    half: bool = False,
    mirrored: bool = False,
) -> tuple[gfem.Mesh, float]:
    r"""The sphere in air in a layer, and its volume-equivalent radius (um).

    The sphere's surface is the surface region "shell". With ``half`` only
    :math:`y \ge 0` is meshed and the plane :math:`y = 0` is the surface
    region "mirror": a TE wave in the xz plane (E along y) satisfies a PEC
    wall there, a TM wave a PMC wall; with ``mirrored`` that half and its
    image make the whole mesh (`sphere_mesh.write`).
    """
    volume = sphere_mesh.write(
        path, maxh=maxh, sphere_maxh=sphere_maxh, half=half, mirrored=mirrored
    )
    radius = (3 * volume / (4 * math.pi)) ** (1 / 3)
    return gfem.Mesh.from_gmsh(path, length_unit="um"), radius


def model(
    mesh: gfem.Mesh, material: gfem.Material | None, *, mirror: str | None = None
) -> gfem.Model:
    """The sphere's model: ``material`` inside it, or a PEC sphere for ``None``.

    ``mirror`` is the half model's wall at y = 0: ``"PEC"`` or ``"PMC"``.
    """
    air = gfem.Material()
    materials = {"air": air, "pml": air}
    boundaries: list = []
    excluded: tuple = ()
    if material is None:
        boundaries.append(gfem.PEC("shell"))
        excluded = ("sphere",)
    else:
        materials["sphere"] = material
    if mirror == "PEC":
        boundaries.append(gfem.PEC("mirror"))
    elif mirror == "PMC":
        boundaries.append(gfem.PMC("mirror"))
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries,
        excluded=excluded,
        pml=gfem.PML("pml"),
        background=air,
    )


def mie(eps: complex | None, x: float) -> tuple[float, float, float]:
    r"""Extinction, absorption and scattering efficiencies of a sphere.

    Bohren and Huffman's series (Absorption and Scattering of Light by
    Small Particles, 1983, ch. 4) for size parameter :math:`x = ka`,
    :math:`Q = \sigma / (\pi a^2)`. Their :math:`e^{-i\omega t}` index is
    :math:`m = n + i\kappa`, so gfem's :math:`\varepsilon = \varepsilon' -
    i\varepsilon''` (:math:`e^{+i\omega t}`) is :math:`m =
    \sqrt{\varepsilon' + i\varepsilon''}`. ``eps=None`` is the perfectly
    conducting sphere: :math:`a_n = \psi_n'(x)/\xi_n'(x)`,
    :math:`b_n = \psi_n(x)/\xi_n(x)`.
    """
    n = np.arange(1, math.ceil(x + 4 * x ** (1 / 3) + 2) + 6)

    def psi(z: complex) -> np.ndarray:
        return z * spherical_jn(n, z)

    def dpsi(z: complex) -> np.ndarray:
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    dh = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    xi, dxi = x * h, h + x * dh
    if eps is None:
        a, b = dpsi(x) / dxi, psi(x) / xi
    else:
        m = np.sqrt(np.conj(complex(eps)))
        mx = m * x
        a = (m * psi(mx) * dpsi(x) - psi(x) * dpsi(mx)) / (
            m * psi(mx) * dxi - xi * dpsi(mx)
        )
        b = (psi(mx) * dpsi(x) - m * psi(x) * dpsi(mx)) / (
            psi(mx) * dxi - m * xi * dpsi(mx)
        )
    ext = 2 / x**2 * float(np.sum((2 * n + 1) * np.real(a + b)))
    sca = 2 / x**2 * float(np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)))
    return ext, ext - sca, sca


def mie_cross_sections(eps: complex | None, radius_um: float) -> np.ndarray:
    """Extinction, absorption, scattering in m² at `WAVELENGTH` (``mie``)."""
    k = 2 * math.pi / WAVELENGTH
    area = math.pi * (radius_um * 1e-6) ** 2
    return np.array(mie(eps, k * radius_um)) * area


def half_space_power(height: float, eps: complex | None, *, normal: bool) -> float:
    r"""An electric dipole's power above a half-space over its vacuum power.

    The dipole sits in vacuum at ``height`` wavelengths above the plane of a
    half-space of permittivity ``eps`` (``None``: a perfect conductor).
    Sommerfeld's reflection integrals (Novotny and Hecht, Principles of
    Nano-Optics, 2012, ch. 10), with :math:`s = k_\rho/k`,
    :math:`s_z = \sqrt{1 - s^2}`, :math:`s_{z2} = \sqrt{\varepsilon - s^2}`
    (non-negative imaginary parts, their :math:`e^{-i\omega t}`; the power
    ratio is real, so the sign convention does not enter) and
    :math:`x = 2kh`:

    .. math::

        \frac{P_\perp}{P_0} = 1 + \tfrac32\operatorname{Re}\int_0^\infty
            \frac{s^3}{s_z} r_p\, e^{i x s_z}\,ds, \qquad
        \frac{P_\parallel}{P_0} = 1 + \tfrac34\operatorname{Re}\int_0^\infty
            \frac{s}{s_z}\left(r_s - s_z^2 r_p\right) e^{i x s_z}\,ds,

    :math:`r_s = (s_z - s_{z2})/(s_z + s_{z2})`, :math:`r_p = (\varepsilon s_z
    - s_{z2})/(\varepsilon s_z + s_{z2})`. The propagating part
    (:math:`s = \sin t`) and the evanescent part (:math:`s = \cosh u`,
    :math:`s_z = i\sinh u`) are integrated separately, with no singularity.
    """
    from scipy.integrate import quad

    x = 2 * 2 * math.pi * height

    def reflections(s: complex, sz: complex) -> tuple[complex, complex]:
        if eps is None:
            return -1.0, 1.0
        sz2 = np.sqrt(complex(eps) - s * s)
        sz2 = sz2 if sz2.imag >= 0 else -sz2
        return (sz - sz2) / (sz + sz2), (eps * sz - sz2) / (eps * sz + sz2)

    def term(s: float, sz: complex) -> complex:
        rs, rp = reflections(s, sz)
        kernel = s**2 * rp if normal else (rs - sz * sz * rp) / 2
        return s * kernel * np.exp(1j * x * sz)

    # ds / s_z = dt on s = sin t; ds / s_z = -i du on s = cosh u
    propagating = quad(
        lambda t: term(math.sin(t), complex(math.cos(t))).real,
        0,
        math.pi / 2,
        limit=200,
    )[0]
    u_max = math.asinh(40 / x)
    kinks = []
    if eps is not None and complex(eps).real > 1:
        kinks = [math.acosh(math.sqrt(complex(eps).real))]
    evanescent = quad(
        lambda u: (-1j * term(math.cosh(u), 1j * math.sinh(u))).real,
        0,
        u_max,
        points=kinks or None,
        limit=400,
    )[0]
    return 1 + 1.5 * (propagating + evanescent)
