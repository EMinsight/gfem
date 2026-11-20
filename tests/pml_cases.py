"""Meshes with perfectly matched layers shared by the layer tests.

`box` writes the unit cube [0, 1]^3 ("air") wrapped in layer cells on the
named sides, on the structured Kuhn grid of `structured_meshes`. Every
builder takes the file to write; nothing here solves.
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
