r"""Write the nanosphere tutorial's mesh: a sphere in an air cube in a layer.

A sphere ("sphere", radius `RADIUS`) in a cube of air ("air", half-side
`HALF`) wrapped in a perfectly matched layer ("pml", `THICKNESS` thick),
meshed by netgen (unstructured, straight tetrahedra) and written as a gmsh
2.2 file, lengths in micrometres. The dipole tutorial puts its emitters on
the :math:`+z` axis just outside the sphere, where the field of a dipole
close to a metal varies on the scale of its distance: a box around that
stretch of the axis (`CAP`), sphere and air alike, is meshed at `CAP_MAXH`,
and the sphere's surface at `SURFACE_MAXH` elsewhere, so that its faceted
volume stays within a percent of the round one. The layer is made of boxes
split along the cube's planes, so that no element straddles one of them.

Usage: ``python tools/nanosphere_mesh.py docs/tutorials/data``
(``just nanosphere-mesh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import ngsolve as ngs
import numpy as np
import structured_meshes as sm

#: The sphere's radius, the cube's half-side and the layer's thickness, um.
RADIUS = 0.15
HALF = 0.45
THICKNESS = 0.35
#: Half-side of a box of air around the sphere, meshed at `NEAR_MAXH`, um.
INNER = 0.35
#: The refined box on the :math:`+z` axis: its half-width, and how far it
#: reaches into the sphere and out of it, um.
CAP = (0.04, 0.025, 0.045)
#: The maximal edge lengths, um: global, near box, the sphere's volume and
#: surface, and the refined box.
MAXH = 0.3
NEAR_MAXH = 0.12
SPHERE_MAXH = 0.08
SURFACE_MAXH = 0.045
CAP_MAXH = 0.018


def write(path: str | Path) -> float:
    """Write the mesh to ``path``; return the meshed sphere's volume, um³."""
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere

    ball = Sphere(Pnt(0, 0, 0), RADIUS)
    width, below, above = CAP
    cap = Box(Pnt(-width, -width, RADIUS - below), Pnt(width, width, RADIUS + above))
    parts = []
    for solid, name in ((ball * cap, "sphere"), (cap - ball, "air")):
        solid.mat(name)
        solid.solids.maxh = CAP_MAXH
        solid.faces.maxh = CAP_MAXH
        parts.append(solid)
    rest = ball - cap
    rest.mat("sphere")
    rest.solids.maxh = SPHERE_MAXH
    rest.faces.maxh = SURFACE_MAXH
    h, t = HALF, HALF + THICKNESS
    near = Box(Pnt(-INNER, -INNER, -INNER), Pnt(INNER, INNER, INNER))
    inner = near - ball - cap
    inner.mat("air")
    inner.solids.maxh = NEAR_MAXH
    air = Box(Pnt(-h, -h, -h), Pnt(h, h, h)) - near
    air.mat("air")
    # the layer as boxes split along the cube's planes, as the layer needs:
    # no element straddles one of them
    cuts = [(-t, -h), (-h, h), (h, t)]
    pml = []
    for x0, x1 in cuts:
        for y0, y1 in cuts:
            for z0, z1 in cuts:
                if (x0, y0, z0) == (-h, -h, -h):
                    continue
                cell = Box(Pnt(x0, y0, z0), Pnt(x1, y1, z1))
                cell.mat("pml")
                pml.append(cell)
    shape = Glue([*parts, rest, inner, air, *pml])
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=MAXH))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    tet_names = [el.mat for el in mesh.Elements(ngs.VOL)]
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    inside = np.array([n == "sphere" for n in tet_names])
    sm.write_msh22(path, (nodes, tets, tet_names, np.empty((0, 3), int), []))
    return float(np.abs(det[inside]).sum() / 6)


if __name__ == "__main__":
    out = Path(sys.argv[1]) / "nanosphere.msh"
    print(f"{out}: sphere volume {write(out):.6f} um^3")
