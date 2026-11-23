"""Silicon-on-insulator strip waveguides on structured meshes, for the solver tests.

`straight` writes the SOI strip (a 0.5 x 0.22 um silicon core in silica)
running straight along x through a square box, as the mirror half model
z >= 0, on the structured Kuhn grid of `structured_meshes`; `model` makes
the `gfem.Model` of such a mesh: second-order absorbing outer faces, a PMC
mirror at z = 0 and a TE0 wave port on each end face (the whole face, its
edges PEC). The box is sized in cladding wavelengths at 1.55 um, so a case
names its electrical size. Nothing here solves.

    path = soi_cases.straight(tmp / "soi.msh", 1.0)
    model = soi_cases.model(path)
    problem = gfem.Driven(frequency=soi_cases.F0, excite=None)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import structured_meshes as sm

#: Speed of light, m/s.
C0 = 299792458.0
#: The vacuum wavelength (um, the mesh unit) and its frequency (Hz).
WAVELENGTH = 1.55
F0 = C0 / (WAVELENGTH * 1e-6)
#: Silicon at 1.55 um, and fused silica from Malitson's Sellmeier formula.
N_CORE = 3.4757
N_CLADDING = 1.444024
#: The cladding wavelength, um: the unit of a box's electrical size.
LAMBDA_CLADDING = WAVELENGTH / N_CLADDING
#: The core's width and thickness, um.
CORE_W, CORE_T = 0.5, 0.22
#: The top of the oxide above the core's mid-plane, um.
Z_TOP = CORE_T / 2 + 1.0
#: The in-plane cell size, um.
H = 0.1


def half_width(scale: float) -> float:
    """The box half-width (um, to 1 nm) of a box ``scale`` cladding wavelengths wide."""
    return round(scale * LAMBDA_CLADDING / 2, 3)


def _axis(lines: Iterable[float], h: float) -> np.ndarray:
    """Grid along one axis: every line a node, each segment cut at <= ``h``."""
    pts = sorted(set(lines))
    out = [np.array([pts[0]])]
    for a, b in itertools.pairwise(pts):
        out.append(np.linspace(a, b, math.ceil((b - a) / h - 1e-9) + 1)[1:])
    return np.concatenate(out)


def _z_edges(h: float, growth: float = 1.25) -> np.ndarray:
    """The z grid from 0: the half core in even cells, then graded oxide to `Z_TOP`.

    The core takes ``max(2, ceil(0.11 / h))`` cells; the oxide cells grow by
    ``growth`` up to ``min(0.2, 2 h)``, then stay at that size.
    """
    n_core = max(2, math.ceil(CORE_T / 2 / h - 1e-9))
    hz_max = min(0.2, 2 * h)
    size, z, sizes = CORE_T / 2 / n_core, CORE_T / 2, []
    while size * growth < hz_max and z + size * growth < Z_TOP:
        size *= growth
        sizes.append(size)
        z += size
    rest = Z_TOP - CORE_T / 2 - sum(sizes)
    n_rest = math.ceil(rest / hz_max - 1e-9)
    sizes += [rest / n_rest] * n_rest
    up = np.concatenate(
        [np.linspace(0.0, CORE_T / 2, n_core + 1), CORE_T / 2 + np.cumsum(sizes)]
    )
    up[-1] = Z_TOP
    return up


def straight(
    path: str | Path,
    scale: float,
    *,
    lines: Iterable[float] = (),
    h: float = H,
) -> Path:
    """The SOI strip straight along x through the box ``[-B, B]^2 x [0, Z_TOP]``.

    ``B`` is `half_width` of ``scale``. Regions: "core" and "cladding";
    surfaces: "port1" at x = -B, "port2" at x = B, "mirror" at z = 0 and
    "outer" on the y faces and the top.

    Args:
        path: The gmsh 2.2 file to write.
        scale: The box width in cladding wavelengths.
        lines: Extra grid lines at these ``|y|`` (um), mirrored to ``-y``:
            a line at ``B - g`` lays a layer of ``g``-thin cells along both
            outer y faces.
        h: The largest in-plane cell size, um.

    Returns:
        ``path``.
    """
    box = half_width(scale)
    ys = {CORE_W / 2, box, *lines}
    x = _axis({-box, 0.0, box}, h)
    y = _axis({-v for v in ys} | ys, h)
    z = _z_edges(h)
    eps = 1e-9

    def volume(c):
        if abs(c[2]) < CORE_T / 2 and abs(c[1]) < CORE_W / 2:
            return "core"
        return "cladding"

    def surface(c):
        x_, y_, z_ = c
        if abs(z_) < eps:
            return "mirror"
        if abs(x_ + box) < eps:
            return "port1"
        if abs(x_ - box) < eps:
            return "port2"
        if abs(abs(y_) - box) < eps or abs(abs(z_) - Z_TOP) < eps:
            return "outer"
        return ""

    # the grid runs z from -Z_TOP to 0 and is reflected: the Kuhn split's
    # diagonals then match the half model the solver tests were measured on
    mesh = sm.structured(
        (x, y, -z[::-1]), volume, surface, lambda p: p * [1.0, 1.0, -1.0]
    )
    return sm.write_msh22(path, mesh)


def model(path: str | Path):
    """The `gfem.Model` of a `straight` mesh: wave ports "p1" and "p2"."""
    import gfem

    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "cladding": gfem.Material(eps_r=N_CLADDING**2, name="SiO2"),
            "core": gfem.Material(eps_r=N_CORE**2, name="Si"),
        },
        boundaries=[gfem.Absorbing("outer", order=2), gfem.PMC("mirror")],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
