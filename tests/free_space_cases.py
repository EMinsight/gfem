r"""Scattering cases with closed forms, shared by the free-space tests.

`sphere` writes a sphere ("sphere", radius `RADIUS`) in a cube of air
("air", half-side `HALF`) wrapped in a perfectly matched layer ("pml",
`THICKNESS` thick), meshed by netgen (unstructured, straight tetrahedra) and
read back through `gfem.Mesh.from_gmsh`, lengths in micrometres; `mie`
gives the Bohren-Huffman series it is compared with. Nothing here solves.

Straight tetrahedra cut the sphere's surface, so the closed forms are
evaluated at the volume-equivalent radius :math:`a_\mathrm{vol}` of the
meshed sphere: against the nominal radius the faceted sphere is 2-3 % off,
against :math:`a_\mathrm{vol}` it matches a curved mesh to 0.1 % once the
sphere is resolved. Order 2 is not usable at these sizes (-4 % lossy, -10 %
lossless: :math:`\varepsilon = 4` halves the wavelength inside), so the
gates run at order 3.

Measured errors against Mie at :math:`a_\mathrm{vol}` (extinction,
absorption, scattering), plane wave at 30 degrees, order 3:

- `MAXH`, sphere at `SPHERE_MAXH` (about 50k DOF): lossy
  :math:`\varepsilon = 4 - i` +0.48 / -1.27 / +1.53 %, lossless
  :math:`\varepsilon = 4` -0.56 %, PEC +1.42 %. Gates `TOLERANCE` and
  `PEC_TOLERANCE`.
- Sphere refined to `REFINED_MAXH` (:math:`\lambda_\mathrm{in}/6`, about
  80k DOF): lossy +0.38 / +0.23 / +0.46 %, lossless +0.22 %, the exterior
  mesh and layer floor. Gate `REFINED_TOLERANCE`.
"""

from __future__ import annotations

import math
from pathlib import Path

import ngsolve as ngs
import numpy as np
import structured_meshes as sm
from scipy.special import spherical_jn, spherical_yn

import gfem

#: Speed of light, m/s.
C0 = 299792458.0
#: The vacuum wavelength, um (the mesh unit), and its frequency, Hz.
WAVELENGTH = 1.0
F0 = C0 / (WAVELENGTH * 1e-6)
#: The sphere's nominal radius, um: :math:`ka = \pi/2`.
RADIUS = 0.25
#: Half-side of the air cube and the layer's thickness, um.
HALF = 0.8
THICKNESS = 0.35
#: The global and the sphere's maximal edge lengths, um.
MAXH = 0.4
SPHERE_MAXH = 0.2
REFINED_MAXH = 0.08
#: Relative gates on each cross section (see the module docstring).
TOLERANCE = 2e-2
PEC_TOLERANCE = 3e-2
REFINED_TOLERANCE = 1e-2
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


def sphere(
    path: str | Path,
    *,
    maxh: float = MAXH,
    sphere_maxh: float = SPHERE_MAXH,
    half: bool = False,
) -> tuple[gfem.Mesh, float]:
    r"""The sphere in air in a layer, and its volume-equivalent radius (um).

    The sphere's surface is the surface region "shell". With ``half`` only
    :math:`y \ge 0` is meshed and the plane :math:`y = 0` is the surface
    region "mirror": a TE wave in the xz plane (E along y) satisfies a PEC
    wall there, a TM wave a PMC wall.
    """
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere

    ball = Sphere(Pnt(0, 0, 0), RADIUS)
    ball.mat("sphere")
    ball.maxh = sphere_maxh
    ball.solids.maxh = sphere_maxh
    ball.faces.maxh = sphere_maxh
    ball.faces.name = "shell"
    h, t = HALF, HALF + THICKNESS
    low = 0.0 if half else -h
    air = Box(Pnt(-h, low, -h), Pnt(h, h, h)) - ball
    air.mat("air")
    if half:
        ball = ball * Box(Pnt(-t, 0, -t), Pnt(t, t, t))
        ball.mat("sphere")
        ball.faces.name = "shell"
    # the layer as boxes split along the interior box's planes, as the layer
    # needs: no element straddles one of them
    cuts = [(-t, -h), (-h, h), (h, t)]
    y_cuts = [(0.0, h), (h, t)] if half else cuts
    pml = []
    for x0, x1 in cuts:
        for y0, y1 in y_cuts:
            for z0, z1 in cuts:
                if (x0, z0) == (-h, -h) and y0 in (-h, 0.0):
                    continue
                cell = Box(Pnt(x0, y0, z0), Pnt(x1, y1, z1))
                cell.mat("pml")
                pml.append(cell)
    shape = Glue([ball, air, *pml])
    if half:
        for face in shape.faces:
            if abs(face.center.y) < 1e-9:
                face.name = "mirror"
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    tet_names = [el.mat for el in mesh.Elements(ngs.VOL)]
    named = ("shell", "mirror")
    bnd = [el for el in mesh.Elements(ngs.BND) if el.mat in named]
    tris = np.array([[v.nr for v in el.vertices] for el in bnd]).reshape(-1, 3)
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    inside = np.array([n == "sphere" for n in tet_names])
    volume = np.abs(det[inside]).sum() / 6 * (2 if half else 1)
    sm.write_msh22(path, (nodes, tets, tet_names, tris, [el.mat for el in bnd]))
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
    n = np.arange(1, int(math.ceil(x + 4 * x ** (1 / 3) + 2)) + 6)

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
