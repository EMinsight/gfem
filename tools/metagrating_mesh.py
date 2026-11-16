"""Build the metagrating tutorial's meshes: one period of a silicon grating on glass.

Usage (gmsh is not in the lock; ``just metagrating-mesh`` runs this with
``uv run --with gmsh==4.15.2 --with numpy --no-project``):
    python tools/metagrating_mesh.py docs/tutorials/data

Writes ``metagrating.msh`` and ``metagrating_fine.msh`` (gmsh 2.2, um). The
committed files are this script's output with gmsh 4.15.2 on one thread; another
gmsh version can move nodes, so the script refuses it, and a regenerated mesh
means re-measuring the tutorial's values.

Geometry (um): one period of a grating whose lines run along y, as a thin cell
periodic in x and y.

- x in [0, PERIOD], PERIOD = 1.55 / sin(50 deg) = 2.0234: the -1 transmitted
  order of a 1.55 um wave at normal incidence leaves into air at 50 deg.
- y in [0, 0.08]: the cell's thickness along the lines.
- z: volume "glass" in [0, 0.5], the two design layers "lower" in
  [0.5, 0.8] and "upper" in [0.8, 1.1], "air" in [1.1, 1.7].

Surfaces: "bottom" (z = 0, under the glass) and "top" (z = 1.7, over the
air), the open faces; "x_min", "x_max", "y_min", "y_max", the periodic pairs.

The mesh is a prism mesh cut into tetrahedra. gmsh triangulates the (x, z)
cross-section: the glass and the air with sides of at most h_max
(frontal-Delaunay), the design layers as a structured grid of ``PIXELS``
columns, one per pixel of the tutorial's layer grid (pitch PERIOD / 81), and
``rows`` rows per layer. Every design element then lies inside one pixel
column, so meshes with the same columns sample the same pixel image, and
refining them changes the field, not the design. The x = PERIOD edges are
copies of the x = 0 edges (``setPeriodic``). The triangulation is
extruded along y in ``Y_LAYERS`` layers (a periodic pair needs two elements
across), and each prism is cut into three tetrahedra by its vertices' rank
in (z, x): every quadrilateral face takes the diagonal from its higher-ranked
bottom vertex to its lower-ranked top vertex, so neighbouring prisms conform
and the faces x = 0 and x = PERIOD, like y = 0 and y = 0.08, carry translated
copies of one triangulation, as `gfem.Periodic` requires.

==================== ===== ============ ===== ==========
mesh                 h_max design rows  nodes tetrahedra
==================== ===== ============ ===== ==========
metagrating.msh      0.08  6 (50 nm)    4683  18084
metagrating_fine.msh 0.06  12 (25 nm)   8646  33648
==================== ===== ============ ===== ==========

Physical groups (gmsh number, name): volumes 1 "glass", 2 "lower",
3 "upper", 4 "air"; surfaces 5 "bottom", 6 "top", 7 "x_min", 8 "x_max",
9 "y_min", 10 "y_max".
"""

import math
import sys
from pathlib import Path

import numpy as np
from structured_meshes import write_msh22

GMSH_VERSION = "4.15.2"

PERIOD = 1.55 / math.sin(math.radians(50.0))  # um, along x
DEPTH = 0.08  # um, along y (the lines)
LAYERS = (("glass", 0.5), ("lower", 0.3), ("upper", 0.3), ("air", 0.6))  # bottom up
NAMES = [name for name, _ in LAYERS]
TOPS = np.cumsum([height for _, height in LAYERS])  # each volume's top face, um
Y_LAYERS = 2  # tetrahedron layers along y: a periodic pair needs two across

PIXELS = 81  # the tutorial's pixels per period: pitch PERIOD / 81 = 24.98 nm
DESIGN = ("lower", "upper")
#: (h_max in um, rows per design layer): 50 nm and 25 nm rows
MESHES = {"metagrating.msh": (0.08, 6), "metagrating_fine.msh": (0.06, 12)}


def _section(h_max, rows):
    """The (x, z) cross-section's triangulation with gmsh.

    Returns the nodes ``(n, 2)`` as (x, z), the triangles ``(m, 3)`` and
    each triangle's volume name.
    """
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    occ = gmsh.model.occ
    # the section in gmsh's (x, y) plane: gmsh y is the cell's z
    rectangles = [
        (2, occ.addRectangle(0, top - height, 0, PERIOD, height))
        for (_, height), top in zip(LAYERS, TOPS, strict=True)
    ]
    occ.fragment(rectangles[:1], rectangles[1:])
    occ.synchronize()
    sides = {0.0: [], PERIOD: []}
    for _, tag in gmsh.model.getEntities(1):
        c = occ.getCenterOfMass(1, tag)
        for x, found in sides.items():
            if abs(c[0] - x) < 1e-9:
                found.append((c[1], tag))
    low, high = ([t for _, t in sorted(found)] for found in sides.values())
    # affine map (4 x 4, row-major) from the x = 0 edges to their periodic copies
    affine = [1, 0, 0, PERIOD, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, high, low, affine)
    # the design layers: a structured grid of PIXELS columns, one per pixel of
    # the tutorial's layer grid, and ``rows`` rows per layer
    for _, surface in gmsh.model.getEntities(2):
        zc = occ.getCenterOfMass(2, surface)[1]
        if NAMES[int(np.searchsorted(TOPS, zc))] not in DESIGN:
            continue
        for _, curve in gmsh.model.getBoundary([(2, surface)], oriented=False):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, curve)
            along_x = x1 - x0 > y1 - y0
            gmsh.model.mesh.setTransfiniteCurve(
                curve, PIXELS + 1 if along_x else rows + 1
            )
        gmsh.model.mesh.setTransfiniteSurface(surface)
    for k in ("ExtendFromBoundary", "FromPoints", "FromCurvature"):
        gmsh.option.setNumber("Mesh.MeshSize" + k, 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal-Delaunay
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    nodes = np.asarray(coords).reshape(-1, 3)[:, :2]
    triangles, names = [], []
    for _, surface in gmsh.model.getEntities(2):
        zc = occ.getCenterOfMass(2, surface)[1]
        name = NAMES[int(np.searchsorted(TOPS, zc))]
        _, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
        tris = [index[int(t)] for t in element_nodes[0]]
        triangles.append(np.array(tris, dtype=np.int64).reshape(-1, 3))
        names += [name] * (len(tris) // 3)
    gmsh.finalize()
    return nodes, np.concatenate(triangles), names


def _extrude(nodes, triangles, names):
    """The cell's tetrahedra and named boundary triangles, as `write_msh22` takes."""
    n = len(nodes)
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((nodes[:, 0], nodes[:, 1]))] = np.arange(n)  # by z, then x
    ys = np.linspace(0.0, DEPTH, Y_LAYERS + 1)
    points = np.array([(x, y, z) for y in ys for x, z in nodes])
    tets, tet_names = [], []
    for name in NAMES:  # volumes in LAYERS order: physical numbers 1 to 4
        for tri in triangles[[k for k, t in enumerate(names) if t == name]]:
            a, b, c = tri[np.argsort(rank[tri])]
            for j in range(Y_LAYERS):
                lo, hi = j * n, (j + 1) * n
                for tet in (
                    [a + lo, b + lo, c + lo, a + hi],
                    [b + lo, c + lo, a + hi, b + hi],
                    [c + lo, a + hi, b + hi, c + hi],
                ):
                    p = points[tet]
                    if np.linalg.det(p[1:] - p[0]) < 0:  # positive orientation
                        tet[0], tet[1] = tet[1], tet[0]
                    tets.append(tet)
                    tet_names.append(name)

    tris, tri_names = [], []
    # the cross-section's boundary edges: on z = 0, z = top, x = 0, x = PERIOD
    edges = np.sort(np.concatenate([triangles[:, [i, (i + 1) % 3]] for i in range(3)]))
    unique, count = np.unique(edges, axis=0, return_counts=True)
    sides = {
        "bottom": (1, 0.0),
        "top": (1, TOPS[-1]),
        "x_min": (0, 0.0),
        "x_max": (0, PERIOD),
    }
    for face, (axis, value) in sides.items():
        for edge in unique[count == 1]:
            if np.abs(nodes[edge, axis] - value).max() > 1e-9:
                continue
            p, q = edge[np.argsort(rank[edge])]
            for j in range(Y_LAYERS):
                lo, hi = j * n, (j + 1) * n
                tris += [(p + lo, q + lo, p + hi), (q + lo, p + hi, q + hi)]
                tri_names += [face, face]
    for face, j in (("y_min", 0), ("y_max", Y_LAYERS)):
        tris += [tuple(t + j * n) for t in triangles]
        tri_names += [face] * len(triangles)
    return points, np.array(tets), tet_names, np.array(tris), tri_names


def build(out, h_max, rows):
    """Mesh the cell with sides of at most ``h_max`` um and write it to ``out``.

    Returns the node and tetrahedron counts.
    """
    mesh = _extrude(*_section(h_max, rows))
    write_msh22(out, mesh)
    return len(mesh[0]), len(mesh[1])


def main(outdir):
    """Write every mesh of ``MESHES`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, sizes in MESHES.items():
        nodes, tets = build(outdir / name, *sizes)
        print(f"{outdir / name}: {nodes} nodes, {tets} tetrahedra")


if __name__ == "__main__":
    main(sys.argv[1])
