"""The thin-film lithium niobate ridge of the second-harmonic recipe, meshed with netgen.

The section in the (x, y) plane, y up, y = 0 at the bottom of the film: a
film ``"film"`` of ``film`` thickness etched ``etch`` deep into a ridge
``width`` wide at its top, centred on x = 0, on ``"box"`` oxide, under air
``"air"``. The section reaches ``margin`` beyond the ridge on every side.
`ridge_section` meshes it, writes the mesh as a gmsh 2.2 file and reads it
back through `gfem.Mesh.from_gmsh_cross_section`, as a mesh of your own
would be; its end face ``"front"`` carries a wave port.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6


def ridge_section(
    width: float,
    film: float = 0.6 * UM,
    etch: float = 0.3 * UM,
    margin: float = 1.5 * UM,
    *,
    h_film: float = 0.15 * UM,
    h_outer: float = 0.6 * UM,
) -> gfem.Mesh:
    """Mesh the ridge: every length in m, edge lengths ``h_film`` and ``h_outer``."""
    w, t, e, g = (v / UM for v in (width, film, etch, margin))
    half = w / 2 + g
    box = Rectangle(2 * half, g).Face().Move((-half, -g, 0))
    slab = Rectangle(2 * half, t - e).Face().Move((-half, 0, 0))
    ridge = Rectangle(w, e).Face().Move((-w / 2, t - e, 0))
    air = Rectangle(2 * half, e + g).Face().Move((-half, t - e, 0))
    # the air block overlaps the ridge: cut it out
    air = air - ridge
    for face, name in ((box, "box"), (slab, "film"), (ridge, "film"), (air, "air")):
        face.faces.name = name
    slab.faces.maxh = ridge.faces.maxh = h_film / UM
    geometry = OCCGeometry(Glue([box, slab, ridge, air]), dim=2)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=h_outer / UM))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    tags = {name: k for k, name in enumerate(dict.fromkeys(n for n, _ in triangles), 1)}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines += [str(len(tags)), *(f'2 {k} "{name}"' for name, k in tags.items())]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    lines += ["$EndNodes", "$Elements", str(len(triangles))]
    lines += [
        f"{k} 2 2 {tags[name]} {tags[name]} " + " ".join(str(v + 1) for v in vertices)
        for k, (name, vertices) in enumerate(triangles, 1)
    ]
    lines.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ridge.msh"
        path.write_text("\n".join(lines) + "\n")
        return gfem.Mesh.from_gmsh_cross_section(path, length_unit="um")
