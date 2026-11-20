"""Meshes with perfectly matched layers shared by the layer tests.

`box` writes the unit cube [0, 1]^3 ("air") wrapped in layer cells on the
named sides, on the structured Kuhn grid of `structured_meshes`;
`tem_line` and `te10_guide` end a straight guide with a PEC-backed layer,
layered along z, for the closed-form reflection cases, whose measured
floors and errors are recorded next to their gates. Every builder takes
the file to write; nothing here solves.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import structured_meshes as sm

#: The six sides of the interior box, as they are written in messages.
SIDES = ("-x", "+x", "-y", "+y", "-z", "+z")


def _edges(sides: tuple[str, ...], thickness: float, cells: int) -> tuple:
    out = []
    for axis in "xyz":
        lo = -thickness if f"-{axis}" in sides else 0.0
        hi = 1.0 + thickness if f"+{axis}" in sides else 1.0
        out.append(np.linspace(lo, hi, round((hi - lo) * cells) + 1))
    return tuple(out)


def box(
    path: str | Path,
    sides: tuple[str, ...] = ("+x",),
    *,
    thickness: float = 0.5,
    cells: int = 2,
    volume: Callable[[np.ndarray], str | None] | None = None,
    surface: Callable[[np.ndarray], str] | None = None,
    mapping: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Path:
    """The unit cube "air" with layer cells "pml" beyond the given sides.

    Args:
        path: The gmsh 2.2 file to write.
        sides: Sides of the cube that carry a layer (`SIDES`); the edge and
            corner cells between two layer sides are layer cells too.
        thickness: The layer thickness, a multiple of ``1 / cells``.
        cells: Cells per unit length.
        volume: Region name of a cell from its centroid, ``None`` to remove
            the cell; by default "air" inside the cube and "pml" outside.
        surface: Name of a boundary face from its centroid, ``""`` for the
            default: "outer" on the grid's bounding box, nothing elsewhere.
        mapping: Maps grid points ``(n, 3)`` to coordinates; the
            classifiers see grid points.

    Returns:
        ``path``.
    """
    edges = _edges(tuple(sides), thickness, cells)
    lo = np.array([e[0] for e in edges])
    hi = np.array([e[-1] for e in edges])
    eps = 1e-9

    def default_volume(c):
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    def named(c):
        name = surface(c) if surface is not None else ""
        if name:
            return name
        if (np.abs(c - lo) < eps).any() or (np.abs(c - hi) < eps).any():
            return "outer"
        return ""

    mesh = sm.carve(edges, volume or default_volume, named, mapping)
    return sm.write_msh22(path, mesh)


# --- closed-form cases ----------------------------------------------------------

#: Speed of light, m/s.
C0 = 299792458.0
#: The cases' vacuum wavelength, um (the mesh unit), and its frequency, Hz.
WAVELENGTH = 1.0
F0 = C0 / (WAVELENGTH * 1e-6)
#: Air before the layer and the layer's thickness, along z (um).
AIR_LENGTH, THICKNESS = 1.5, 0.5
#: The TEM line's square section and the TE10 guide's (um).
LINE_SECTION = (0.2, 0.2)
GUIDE_SECTION = (0.75, 0.3)

#: Reflection floor of a PEC-backed layer on these layered meshes at order
#: 3, h = lambda / 10: measured 9.3e-7 for R0 = 1e-6 (the TEM line,
#: normal incidence; 4.3e-6 for the steeper R0 = 1e-8). A closed form is
#: asserted only where it sits 100 times above it.
FLOOR = 1e-6
#: TE10 floor at h = 1/14 with the reference frequency where k0 = beta:
#: measured 7.2e-5 (h = 0.1 in the raw probe), so V3c asserts 2e-4.
GUIDE_FLOOR = 2e-4
#: Relative gate on a closed-form reflection above its floor. Measured
#: through gfem (order 3, one thread), relative to the closed form: TEM
#: R0 = 1e-2: 4.8e-5; R0 = 1e-3 at f_ref: 6.0e-4 (and 1.5 f_ref: 2.9e-5
#: against 3.2e-5, 2.7e-6 off, under the 3-floor bound); pinned f_ref = 2f:
#: 6.0e-4; eps_r = 2.25: 3.1e-3; TE10 at lambda = 1: 2.2e-5; near cutoff:
#: 8.1e-5; f_ref where k0 = beta: |S11| = 3.2e-6.
REL = 1e-2


def _section_axis(n: float, length: float) -> np.ndarray:
    return np.linspace(0.0, length, max(1, round(length * n)) + 1)


def waveguide(
    path: str | Path,
    section: tuple[float, float],
    *,
    cells: float = 10,
    lateral: float | None = None,
    pec_x: bool = True,
) -> Path:
    """A straight guide along z ended by a PEC-backed layer: "air" then "pml".

    The guide spans ``section`` in x and y, ``AIR_LENGTH`` of "air" from
    z = 0 and ``THICKNESS`` of "pml" beyond. Surfaces: "port" at z = 0,
    "pec" on the y walls, on the x walls when ``pec_x`` (else they stay
    unnamed: PMC), and on the back face of the layer.

    Args:
        path: The gmsh 2.2 file to write.
        section: ``(a, b)``, um.
        cells: Cells per um along z (every z grid plane at a multiple of
            ``1 / cells``, so the layer's face is one of them).
        lateral: Cells per um across the section; ``cells / 2`` by default,
            as the floors were measured (the fields vary slowly across).
        pec_x: Whether the x walls are PEC.

    Returns:
        ``path``.
    """
    a, b = section
    lateral = cells / 2 if lateral is None else lateral
    total = AIR_LENGTH + THICKNESS
    edges = (
        _section_axis(lateral, a),
        _section_axis(lateral, b),
        np.linspace(0.0, total, round(total * cells) + 1),
    )
    eps = 1e-9

    def volume(c):
        return "air" if c[2] < AIR_LENGTH else "pml"

    def surface(c):
        if abs(c[2]) < eps:
            return "port"
        if abs(c[2] - total) < eps or abs(c[1]) < eps or abs(c[1] - b) < eps:
            return "pec"
        if pec_x and (abs(c[0]) < eps or abs(c[0] - a) < eps):
            return "pec"
        return ""

    return sm.write_msh22(path, sm.structured(edges, volume, surface))


def tem_line(path: str | Path, *, cells: float = 10) -> Path:
    """The TEM line: PEC y walls, PMC x walls (a plane wave E_y), `waveguide`."""
    return waveguide(path, LINE_SECTION, cells=cells, pec_x=False)


def te10_guide(path: str | Path, *, cells: float = 14) -> Path:
    """The TE10 guide (a = 0.75, b = 0.3 um, PEC walls), `waveguide`."""
    return waveguide(path, GUIDE_SECTION, cells=cells)
