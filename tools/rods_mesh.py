r"""Build the photonic-crystal tutorial's meshes: one cell of a square lattice of rods.

Usage (gmsh is not in the lock; ``just rods-mesh`` runs this with
``uv run --with gmsh==4.15.2 --with numpy --no-project``):
    python tools/rods_mesh.py docs/tutorials/data

Writes ``rods.msh`` and ``rods_fine.msh`` (gmsh 2.2, mm, triangles only, read
by `gfem.Mesh.from_gmsh_cross_section`, which extrudes them into a slab one
element thick). The committed files are this script's output with gmsh 4.15.2
on one thread; another gmsh version can move nodes, so the script refuses it,
and a regenerated mesh means re-measuring the tutorial's values.

Geometry (mm), the section in the (x, y) plane: the square cell of side
CELL = 10 with a corner at the origin, and a rod of radius RADIUS = 2
(0.2 CELL) at its centre. The rod is a regular N-gon with the disk's area,
of circumradius

.. math:: R = r \sqrt{\frac{2\pi}{N \sin(2\pi/N)}},

with :math:`r` the rod's radius, and a vertex on the +x ray from the centre;
N is a multiple of 8, so the polygon keeps the square's symmetry. A
straight-sided mesh facets a round rod, and the faceting is the whole error
of the bands; the equal-area polygon keeps the cell's mean permittivity exact.

gmsh meshes one eighth of the cell, the wedge between the centre's 0 and 45
degree rays out to the edge x = CELL (frontal-Delaunay, sides of at most
h_max), and the wedge is mirrored under the eight symmetries of the square,
with coincident nodes merged. The section therefore has the crystal's
in-plane point group (C4v), and the edges x = 0 and x = CELL, like y = 0 and
y = CELL, are exact translates of one another without ``setPeriodic``. Nodes
are numbered in (x, y) order and triangles run counterclockwise, for stable
diffs.

============= ===== == ===== =========
mesh          h_max N  nodes triangles
============= ===== == ===== =========
rods.msh      1.25  24 161   288
rods_fine.msh 0.625 48 441   816
============= ===== == ===== =========

Physical groups (gmsh number, name): areas 1 "rod", 2 "air"; lines 3
"x_min", 4 "x_max", 5 "y_min", 6 "y_max".
"""

import math
import sys
from pathlib import Path

import numpy as np
from structured_meshes import write_msh22

GMSH_VERSION = "4.15.2"

CELL = 10.0  # mm, the lattice constant
RADIUS = 0.2 * CELL  # mm, the rod's radius
#: The tutorial's meshes: file name -> (h_max in mm, the rod polygon's sides).
MESHES = {"rods.msh": (1.25, 24), "rods_fine.msh": (0.625, 48)}

# the eight symmetries of the square about its centre, as integer matrices: the
# four rotations by quarter turns, each also after the swap x <-> y
QUARTER_TURN = np.array([[0, -1], [1, 0]])
SWAP = np.array([[0, 1], [1, 0]])
SYMMETRIES = [
    np.linalg.matrix_power(QUARTER_TURN, k) @ s
    for k in range(4)
    for s in (np.eye(2, dtype=int), SWAP)
]


def polygon_radius(n):
    """The circumradius of a regular ``n``-gon with the area of the rod's disk."""
    return RADIUS * math.sqrt(2 * math.pi / (n * math.sin(2 * math.pi / n)))


def _wedge(h_max, n):
    """The wedge's triangulation with gmsh, 0 to 45 degrees about the centre.

    The wedge runs out to the cell's edge x = CELL.

    Returns the nodes ``(k, 2)``, the triangles ``(m, 3)`` and each
    triangle's area name.
    """
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    geo = gmsh.model.geo
    c, rc = CELL / 2, polygon_radius(n)
    sides = n // 8  # the polygon's sides inside the wedge

    def point(x, y):
        return geo.addPoint(x, y, 0)

    centre = point(c, c)
    angles = [2 * math.pi * i / n for i in range(sides + 1)]
    poly = [point(c + rc * math.cos(t), c + rc * math.sin(t)) for t in angles]
    edge_low, edge_high = point(CELL, c), point(CELL, CELL)
    rim = [geo.addLine(poly[i], poly[i + 1]) for i in range(sides)]
    axis_in, axis_out = geo.addLine(centre, poly[0]), geo.addLine(poly[0], edge_low)
    edge = geo.addLine(edge_low, edge_high)
    diagonal_out = geo.addLine(edge_high, poly[-1])
    diagonal_in = geo.addLine(poly[-1], centre)
    rod = geo.addPlaneSurface([geo.addCurveLoop([axis_in, *rim, diagonal_in])])
    outside = [axis_out, edge, diagonal_out, *[-line for line in reversed(rim)]]
    air = geo.addPlaneSurface([geo.addCurveLoop(outside)])
    geo.synchronize()
    for k in ("ExtendFromBoundary", "FromPoints", "FromCurvature"):
        gmsh.option.setNumber("Mesh.MeshSize" + k, 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal-Delaunay
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {int(t): i for i, t in enumerate(tags)}
    nodes = np.asarray(coords).reshape(-1, 3)[:, :2]
    triangles, names = [], []
    for surface, name in ((rod, "rod"), (air, "air")):
        _, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
        tris = [index[int(t)] for t in element_nodes[0]]
        triangles.append(np.array(tris, dtype=np.int64).reshape(-1, 3))
        names += [name] * (len(tris) // 3)
    gmsh.finalize()
    return nodes, np.concatenate(triangles), names


def _mirror(nodes, triangles, names):
    """The cell: the wedge under every symmetry of the square, equal nodes merged."""
    c = CELL / 2
    merged, points = {}, []

    def number(p):
        key = tuple(np.round(p * 1e9).astype(np.int64))  # 1e-9 mm apart: one node
        if key not in merged:
            merged[key] = len(points)
            points.append(p)
        return merged[key]

    cell_triangles, cell_names = [], []
    for op in SYMMETRIES:
        image = [number(p) for p in (nodes - c) @ op.T + c]
        cell_triangles += [[image[v] for v in tri] for tri in triangles]
        cell_names += names
    return np.array(points), np.array(cell_triangles), cell_names


def section(h_max, n):
    """The cell's section, as `write_msh22` takes a triangle-only mesh.

    Returns the nodes ``(k, 3)`` (z = 0) in (x, y) order, the triangles
    counterclockwise with their area names (rod first), and the boundary
    segments with their line names.
    """
    nodes, triangles, names = _mirror(*_wedge(h_max, n))
    order = np.lexsort((nodes[:, 1], nodes[:, 0]))  # by x, then y
    rank = np.empty(len(nodes), dtype=np.int64)
    rank[order] = np.arange(len(nodes))
    nodes, triangles = nodes[order], rank[triangles]
    p = nodes[triangles]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    clockwise = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] < 0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    tris, tri_names = [], []
    for name in ("rod", "air"):
        mine = triangles[[k for k, t in enumerate(names) if t == name]]
        mine = mine[np.lexsort(np.sort(mine, axis=1).T[::-1])]
        tris += mine.tolist()
        tri_names += [name] * len(mine)
    edges = np.sort(np.concatenate([triangles[:, [i, (i + 1) % 3]] for i in range(3)]))
    unique, count = np.unique(edges, axis=0, return_counts=True)
    segments, segment_names = [], []
    for name, axis, value in (
        ("x_min", 0, 0.0),
        ("x_max", 0, CELL),
        ("y_min", 1, 0.0),
        ("y_max", 1, CELL),
    ):
        for e in unique[count == 1]:
            if np.abs(nodes[e, axis] - value).max() < 1e-9:
                segments.append(e.tolist())
                segment_names.append(name)
    points = np.column_stack([nodes, np.zeros(len(nodes))])
    return points, np.array(tris), tri_names, np.array(segments), segment_names


def build(out, h_max, n):
    """Mesh the cell (sides of at most ``h_max`` mm, an ``n``-gon rod); write ``out``.

    Returns the node and triangle counts.
    """
    mesh = section(h_max, n)
    write_msh22(out, mesh)
    return len(mesh[0]), len(mesh[1])


def main(outdir):
    """Write every mesh of ``MESHES`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, sizes in MESHES.items():
        nodes, triangles = build(outdir / name, *sizes)
        print(f"{outdir / name}: {nodes} nodes, {triangles} triangles")


if __name__ == "__main__":
    main(sys.argv[1])
