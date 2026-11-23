r"""Free-space models shared by the source tests: meshes and closed forms.

`dipole_box` writes a cube of "air" wrapped in perfectly matched layer cells
on every side; `image_power` is the emitted power of a dipole above a
perfect conductor or a perfect magnetic wall by image theory, `bulk_power`
the closed form gfem's Purcell factor divides by, written out here as an
independent check. Every builder takes the file to write; nothing here
solves. Lengths in um (the mesh unit), the vacuum wavelength 1 um.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pml_cases as pc

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


def scatterer_box(path: str | Path, *, cells: int = 4) -> Path:
    """A dielectric cube "scat" in the middle of `dipole_box`, its +x face "feed"."""
    return pc.box(
        path,
        sides=pc.SIDES,
        cells=cells,
        volume=_scatterer_volume,
        surface=_scatterer_feed,
    )


def scatterer_model(path: str | Path, *, field: bool = False):
    """The model of `scatterer_box`: air background, a 50 ohm port on "feed".

    With ``field`` the cube is a `gfem.MaterialField` of the same value.
    """
    import gfem

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
