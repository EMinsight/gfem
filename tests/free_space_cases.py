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


# --- a wide box for Gaussian beams ----------------------------------------------

#: The beam box's extent, um, centred on the origin, inside its layer.
BEAM_BOX = (2.8, 2.8, 1.2)


def beam_box(path: str | Path, *, cells: int = 8) -> gfem.Model:
    """A box of air, `BEAM_BOX` um centred on the origin, in a layer on all sides.

    ``cells`` per side, the layer a quarter of the box thick: wide enough
    to hold a beam of waist 0.5 um focused on its top face (2.2 waists from
    the axis to the side at the top face).
    """
    scale = np.array(BEAM_BOX)
    path = pc.box(
        path,
        pc.SIDES,
        thickness=0.25,
        cells=cells,
        mapping=lambda p: (p - 0.5) * scale,
    )
    air = gfem.Material()
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"air": air, "pml": air},
        pml=gfem.PML("pml"),
        background=air,
    )


# --- a port-fed strip in a perfectly matched layer (gradient gates) -------------


def _strip_volume(c: np.ndarray) -> str:
    """A core (1/6, 5/6)^3 um split at x = 0.5 um, a shell, the layer."""
    if not ((c > 0) & (c < 1)).all():
        return "pml"
    if not ((c > 1 / 6) & (c < 5 / 6)).all():
        return "shell"
    if ((c > 2 / 6) & (c < 4 / 6)).all() and c[0] > 0.5:
        return "design"
    return "left" if c[0] < 0.5 else "right"


def _strip_surface(c: np.ndarray) -> str:
    """A strip in the plane x = 0.5 um: its arms "wire" and its feed."""
    eps = 1e-9
    if abs(c[0] - 0.5) > eps or not 0.5 < c[1] < 4 / 6:
        return ""
    if 3 / 6 < c[2] < 4 / 6:
        return "feed"
    return "wire" if 1 / 6 < c[2] < 5 / 6 else ""


def strip(path: str | Path) -> gfem.Model:
    """A PEC strip fed by lumped port "p1", beside a design block, in a layer.

    The unit cube of air in a 2/6 um "pml" on every side; inside it a
    "shell", and a core split at x = 0.5 um into "left" and "right", with
    the "design" block (a `gfem.MaterialField`, 2) in the right half.
    """
    mesh = gfem.Mesh.from_gmsh(
        pc.box(
            path,
            sides=pc.SIDES,
            thickness=2 / 6,
            cells=6,
            volume=_strip_volume,
            surface=_strip_surface,
        )
    )
    air = gfem.Material()
    n = len(mesh.elements("design"))
    return gfem.Model(
        mesh,
        materials={
            "left": air,
            "right": air,
            "shell": air,
            "pml": air,
            "design": gfem.MaterialField(eps_r=np.full(n, 2.0)),
        },
        boundaries=[gfem.PEC("wire")],
        ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
        pml=gfem.PML("pml"),
        background=air,
    )


def assert_layer_moves(model: gfem.Model, morph, side) -> None:
    """The layer outside ``side`` (``None``: the one touching the layer) moves.

    So that a node gradient through ``morph`` exercises the kernel's node pass.
    """
    import jax.numpy as jnp

    moved = np.asarray(morph(jnp.ones(morph.n_params) * 1e-8).nodes) - np.asarray(
        model.mesh.nodes
    )
    mesh = model.mesh
    tets = np.asarray(mesh.tetrahedra)
    regions = ("left", "right", "design", "shell") if side is None else side
    inside = np.zeros(len(tets), bool)
    for r in regions:
        inside[mesh.elements(r)] = True
    if side is None:
        cut = np.zeros(len(mesh.nodes), bool)
        cut[np.unique(tets[mesh.elements("pml")])] = True
        chi = ~cut
    else:
        chi = np.zeros(len(mesh.nodes), bool)
        chi[np.unique(tets[inside])] = True
    at = chi[tets]
    layer = at.any(axis=1) & ~at.all(axis=1)
    assert np.abs(moved[np.unique(tets[layer])]).max() > 0


# --- a block under a plane, lit from above (beam-coupling gradients) -------------

#: The plane the block's top face lies in, um: the side below it is open.
LID = 4 / 6
#: A dipole below the block, in "below", off every face of the grid, m.
LID_DIPOLE = np.array([0.43, 0.55, 0.25]) * UM


def _lid_volume(c: np.ndarray) -> str:
    if not ((c > 0) & (c < 1)).all():
        return "pml"
    if ((c > 2 / 6) & (c < LID)).all():
        return "block"
    return "below" if c[2] < LID else "above"


def _lid_surface(c: np.ndarray) -> str:
    inside = ((c[:2] > 2 / 6) & (c[:2] < LID)).all()
    return "lid" if abs(c[2] - LID) < 1e-9 and inside else ""


def lid_box(path: str | Path, *, field: bool = False) -> gfem.Model:
    """A dielectric "block" whose top face "lid" lies in the plane z = `LID`.

    The unit cube of air in a 2/6 um "pml" on every side, split at that
    plane into "below" and "above"; the block (2/6, 4/6)^3 um, of
    permittivity 2 (a `gfem.MaterialField` with ``field``). The regions
    "below" and "block" are a side whose surface is the plane: open, it
    ends on the layer, and the elements above it touch the lid.
    """
    mesh = gfem.Mesh.from_gmsh(
        pc.box(
            path,
            sides=pc.SIDES,
            thickness=2 / 6,
            cells=6,
            volume=_lid_volume,
            surface=_lid_surface,
        )
    )
    air = gfem.Material()
    n = len(mesh.elements("block"))
    block = (
        gfem.MaterialField(eps_r=np.full(n, 2.0)) if field else gfem.Material(eps_r=2.0)
    )
    return gfem.Model(
        mesh,
        materials={"below": air, "above": air, "pml": air, "block": block},
        pml=gfem.PML("pml"),
    )


# --- a grating coupler on a slab (beam coupling) ---------------------------------

#: The grating slab's wavelength (um) and frequency (Hz), and its indices.
GRATING_WAVELENGTH = 1.55
GRATING_F0 = C0 / (GRATING_WAVELENGTH * UM)
N_SI, N_OX = 3.476, 1.444
SILICON = gfem.Material(eps_r=N_SI**2)
OXIDE = gfem.Material(eps_r=N_OX**2)
#: The section, um: x along the guide, y up; the slab is invariant along z.
GRATING_X = (-5.0, 7.5)
GRATING_Y = (-1.2, 1.6)
#: The guide (220 nm), its 70 nm trenches and the port window.
T_SI, ETCH = 0.22, 0.07
PERIOD, FILL, TEETH = 0.63, 0.5, 5
WINDOW = (-0.8, 1.0)
#: The oxide perfectly matched layers above and below, um thick.
GRATING_PML = 0.6


def grating_slab(
    path: str | Path, *, h_si: float = 0.08, h_ox: float = 0.25
) -> gfem.Mesh:
    """A silicon guide with `TEETH` trenches in oxide, as a slab mesh.

    Regions "si" (the guide and its teeth), "trench" (the oxide in the
    trenches), "clad" (the oxide around), and `GRATING_PML` thick oxide
    layers "pml_top" and "pml_bottom" above and below; edges "port" (the
    window on the left end, across the guide), "left" (the rest of that
    end), "right"; the slab's end faces "front" and "back". The guide runs
    from the port to the right end, where it meets an absorbing face.
    """
    import thermal_cases

    x0, x1 = GRATING_X
    y0, y1 = GRATING_Y
    lo, hi = WINDOW
    base = T_SI - ETCH
    rects: list = [
        (x0, y0, x1, lo, "clad", h_ox),
        (x0, lo, x1, 0.0, "clad", h_ox),
        (x0, 0.0, x1, base, "si", h_si),
        (x0, T_SI, x1, hi, "clad", h_ox),
        (x0, hi, x1, y1, "clad", h_ox),
        (x0, y1, x1, y1 + GRATING_PML, "pml_top", h_ox),
        (x0, y0 - GRATING_PML, x1, y0, "pml_bottom", h_ox),
        (x0, base, 0.0, T_SI, "si", h_si),
    ]
    for i in range(TEETH):
        a = i * PERIOD
        rects.append((a, base, a + FILL * PERIOD, T_SI, "si", h_si))
        rects.append((a + FILL * PERIOD, base, a + PERIOD, T_SI, "trench", h_si))
    rects.append((TEETH * PERIOD, base, x1, T_SI, "si", h_si))
    eps = 1e-9
    edges = [
        (lambda c: abs(c[0] - x0) < eps and lo < c[1] < hi, "port"),
        (lambda c: abs(c[0] - x0) < eps and y0 < c[1] < y1, "left"),
        (lambda c: abs(c[0] - x1) < eps and y0 < c[1] < y1, "right"),
    ]
    return thermal_cases.section(Path(path), rects, h_ox, edges)


def grating_model(mesh: gfem.Mesh, **materials: gfem.Material) -> gfem.Model:
    """The slab's model: wave port "p1", absorbing ends, PML above and below.

    The slab's walls are PEC: a TE field (E along z) meets them.
    """
    layers = {"pml_top": OXIDE, "pml_bottom": OXIDE}
    return gfem.Model(
        mesh,
        materials={
            "clad": OXIDE,
            "trench": OXIDE,
            "si": SILICON,
            **layers,
            **materials,
        },
        ports=[gfem.WavePort("p1", "port")],
        boundaries=[
            gfem.Absorbing("left"),
            gfem.Absorbing("right"),
            gfem.PEC("front"),
            gfem.PEC("back"),
        ],
        pml=gfem.PML(tuple(layers)),
        background=OXIDE,
    )


class Fiber:
    r"""A z-invariant TE Gaussian beam coming down in the oxide, 1 W through the slab.

    :math:`E = \hat z\int A(k_x)\,e^{-i(k_x(x - x_0) - k_y(y - y_0))}\,dk_x`
    over the propagating spectrum, :math:`k_y = \sqrt{k^2 - k_x^2}`, with
    :math:`A = e^{-(\alpha - \alpha_0)^2 (k w_0)^2/4}`, :math:`\alpha =
    \arcsin(k_x/k)`: a beam of waist :math:`w_0` focused at
    :math:`(x_0, y_0)`, travelling down at ``angle`` from :math:`-y` toward
    :math:`+x`. Its power per length is :math:`\pi\int|A|^2k_y\,dk_x /
    (\omega\mu_0)`; it is scaled to 1 W through the slab's ``thickness``.
    A module-level class, so that the function pickles.
    """

    def __init__(
        self,
        *,
        waist: float,
        focus: tuple[float, float],
        angle: float,
        thickness: float,
        samples: int = 801,
    ) -> None:
        self.waist = waist
        self.focus = focus
        self.angle = angle
        self.thickness = thickness
        self.samples = samples

    def __call__(self, points: np.ndarray, frequency: float) -> np.ndarray:
        omega = 2 * math.pi * frequency
        k = omega / C0 * N_OX
        kx = np.linspace(-k, k, self.samples)[1:-1]
        ky = np.sqrt(k**2 - kx**2)
        a = np.exp(-((np.arcsin(kx / k) - self.angle) ** 2) * (k * self.waist) ** 2 / 4)
        dk = kx[1] - kx[0]
        per_length = math.pi * np.sum(np.abs(a) ** 2 * ky) * dk / (omega * MU0)
        a = a * dk / math.sqrt(per_length * self.thickness)
        out = np.zeros((len(points), 3), complex)
        for start in range(0, len(points), 4096):
            p = points[start : start + 4096]
            x = p[:, :1] - self.focus[0]
            y = p[:, 1:2] - self.focus[1]
            out[start : start + 4096, 2] = np.exp(-1j * (kx * x - ky * y)) @ a
        return out
