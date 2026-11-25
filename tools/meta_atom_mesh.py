r"""Write the meta-atom tutorial's mesh: a design cube in air in a layer.

A cube ("design", side `SIDE`) at the origin, the region whose density the
tutorial optimizes, in a cube of air ("air", half-side `HALF`) wrapped in a
perfectly matched layer ("pml", `THICKNESS` thick), meshed by netgen
(unstructured, straight tetrahedra) and written as a gmsh 2.2 file, lengths
in micrometres. The design cube is meshed at `DESIGN_MAXH`, fine enough for
a density that varies over a few elements and for the wavelength in
silicon, and the air at `MAXH`. The layer is made of boxes split along the
cube's planes, so that no element straddles one of them.

Usage: ``python tools/meta_atom_mesh.py docs/tutorials/data``
(``just meta-atom-mesh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import ngsolve as ngs
import numpy as np
import structured_meshes as sm

#: The design cube's side, the air cube's half-side and the layer's thickness, um.
SIDE = 0.35
HALF = 0.4
THICKNESS = 0.3
#: Half-side of a box of air around the design, meshed at `NEAR_MAXH`, um.
INNER = 0.28
#: The maximal edge lengths, um: global, near box and design cube.
MAXH = 0.4
NEAR_MAXH = 0.2
DESIGN_MAXH = 0.08


def write(path: str | Path) -> float:
    """Write the mesh to ``path``; return the meshed design's volume, um³."""
    from netgen.occ import Box, Glue, OCCGeometry, Pnt

    s = SIDE / 2
    design = Box(Pnt(-s, -s, -s), Pnt(s, s, s))
    design.mat("design")
    design.solids.maxh = DESIGN_MAXH
    h, t = HALF, HALF + THICKNESS
    near = Box(Pnt(-INNER, -INNER, -INNER), Pnt(INNER, INNER, INNER))
    inner = near - design
    inner.mat("air")
    inner.solids.maxh = NEAR_MAXH
    air = Box(Pnt(-h, -h, -h), Pnt(h, h, h)) - near
    air.mat("air")
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
    shape = Glue([design, inner, air, *pml])
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=MAXH))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    tet_names = [el.mat for el in mesh.Elements(ngs.VOL)]
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    inside = np.array([n == "design" for n in tet_names])
    sm.write_msh22(path, (nodes, tets, tet_names, np.empty((0, 3), int), []))
    return float(np.abs(det[inside]).sum() / 6)


if __name__ == "__main__":
    out = Path(sys.argv[1]) / "meta_atom.msh"
    print(f"{out}: design volume {write(out):.6f} um^3")
