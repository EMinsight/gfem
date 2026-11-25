"""The optical windows of the acousto-optic tutorial, built with netgen.

The waveguide of an acousto-optic modulator is a rib of x-cut lithium niobate
on the film the acoustic wave runs in. `optical_window` meshes the optical
cross-section around one rib of `transducer_section`'s device: the film, the
rib, the buried oxide below and air above, in the same (x, y) coordinates as
the acoustic section, so that the acoustic fields are read at the optical
points with no offset. Like the transducer helper, it writes the mesh as a gmsh
2.2 file and reads it back through `gfem.Mesh.from_gmsh_cross_section`, and
netgen meshes it on one thread, so it meshes the same way on every run.

The window, in um, centred on the rib's axis ``x_c``:

- ``"box"``: the oxide, ``-BOTTOM < y < -SLAB``;
- ``"slab"``: the film's slab, ``-SLAB < y < 0``;
- ``"rib"``: the rib on it, as in the transducer section;
- ``"air"``: ``0 < y < TOP``, around the rib.

The windows hold no metal: the fingers are 5.5 um away from the ribs and the
mode does not reach them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen import occ
from transducer_section import (
    RIB_ANGLE,
    RIB_HEIGHT,
    RIB_WIDTH,
    SLAB,
    UM,
    _rect,
    _write_gmsh,
)

import gfem

HALF, TOP, BOTTOM = 3.0, 2.0, 2.0  # um: half-width, air above, oxide below
H_FILM, H_OUTER = 0.08, 0.4  # um: edge lengths in the film and rib, and elsewhere
THICKNESS = 0.1e-6  # m, the slab the 2D mesh is extruded into for the mode solve


def optical_window(
    x_centre: float,
    *,
    directory: str | Path | None = None,
    h_film: float = H_FILM,
    h_outer: float = H_OUTER,
) -> gfem.Mesh:
    """Mesh the optical window around the rib centred at ``x_centre`` (m).

    ``h_film`` and ``h_outer`` (um) are the edge lengths in the film and rib and
    elsewhere. ``directory`` receives ``window.msh`` (``None``: a temporary
    directory, removed once the mesh is read).
    """
    if directory is None:
        with tempfile.TemporaryDirectory() as temporary:
            return optical_window(
                x_centre, directory=temporary, h_film=h_film, h_outer=h_outer
            )
    xc = x_centre / UM
    foot = RIB_HEIGHT / np.tan(np.deg2rad(RIB_ANGLE))
    half_base, half_top = RIB_WIDTH / 2 + foot, RIB_WIDTH / 2
    faces = [
        _rect(xc - HALF, -SLAB - BOTTOM, xc + HALF, -SLAB, "box", maxh=h_outer),
        _rect(xc - HALF, -SLAB, xc + HALF, 0.0, "slab", maxh=h_film),
    ]
    trapezoid = (
        occ.WorkPlane()
        .MoveTo(xc - half_base, 0)
        .LineTo(xc + half_base, 0)
        .LineTo(xc + half_top, RIB_HEIGHT)
        .LineTo(xc - half_top, RIB_HEIGHT)
        .Close()
        .Face()
    )
    trapezoid.name, trapezoid.maxh = "rib", h_film
    air = _rect(xc - HALF, 0.0, xc + HALF, TOP, "air", maxh=h_outer) - trapezoid
    air.name = "air"
    ngmesh = ngs.Mesh(
        occ.OCCGeometry(occ.Glue([*faces, trapezoid, air]), dim=2).GenerateMesh(
            maxh=h_outer
        )
    )
    nodes = np.array([v.point for v in ngmesh.vertices])
    tris = [(el.mat, [v.nr for v in el.vertices]) for el in ngmesh.Elements(ngs.VOL)]
    path = Path(directory) / "window.msh"
    _write_gmsh(path, nodes, tris, [])
    return gfem.Mesh.from_gmsh_cross_section(path, thickness=THICKNESS)
