"""The two silicon slabs of the optical force recipe, meshed with netgen.

The section in the (x, y) plane: two slabs ``"si1"`` and ``"si2"``, each
``width`` wide and as tall as the section, ``gap`` apart (the gap centred on
x = 0), in a box of ``"air"`` that reaches ``clearance`` beyond them. Its four
sides are the surface ``"wall"``. A wave port on the section's end face
``"front"`` and a perfectly conducting ``"wall"`` make the slabs' TE mode
(its electric field along y) the mode of two slab waveguides, whatever the
height. Netgen meshes on one thread here, so a section meshes the same way on
every run and the page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6


def slab_pair_section(
    gap: float,
    width: float = 0.22 * UM,
    height: float = 0.1 * UM,
    clearance: float = 1.2 * UM,
    *,
    h_slab: float = 0.03 * UM,
    h_air: float = 0.25 * UM,
) -> gfem.Mesh:
    """Mesh the two slabs: every length in m, edge lengths ``h_slab`` and ``h_air``."""
    g, w, t, c = (v / UM for v in (gap, width, height, clearance))
    half = g / 2 + w + c
    box = Rectangle(2 * half, t).Face().Move((-half, 0, 0))
    left = Rectangle(w, t).Face().Move((-g / 2 - w, 0, 0))
    right = Rectangle(w, t).Face().Move((g / 2, 0, 0))
    box.faces.name, left.faces.name, right.faces.name = "air", "si1", "si2"
    for slab in (left, right):
        slab.faces.maxh = h_slab / UM
    shape = Glue([box, left, right])
    for edge in shape.edges:
        c0 = edge.center
        if min(abs(abs(c0.x) - half), abs(c0.y), abs(c0.y - t)) < 1e-9:
            edge.name = "wall"
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_air / UM))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat == "wall"
    ]
    tags: dict[tuple[int, str], int] = {}
    for dim, items in ((1, lines), (2, triangles)):
        for name, _ in items:
            tags.setdefault((dim, name), len(tags) + 1)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(tags))]
    out += [f'{dim} {k} "{name}"' for (dim, name), k in tags.items()]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(triangles))]
    k = 0
    for dim, kind, items in ((1, 1, lines), (2, 2, triangles)):
        for name, vertices in items:
            k += 1
            tag = tags[dim, name]
            out.append(
                f"{k} {kind} 2 {tag} {tag} " + " ".join(str(v + 1) for v in vertices)
            )
    out.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "slabs.msh"
        path.write_text("\n".join(out) + "\n")
        return gfem.Mesh.from_gmsh_cross_section(
            path, length_unit="um", thickness=0.05 * UM
        )
