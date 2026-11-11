"""Structured tetrahedral meshes written as gmsh 2.2 files, for tests.

No gmsh needed: a box of hexahedra is cut into six tetrahedra each (the Kuhn
split, conforming across hexahedra), optionally mapped to another shape, and
every boundary face is named by the caller from its centroid in the box's
parameter space.

    path = write_msh22(tmp / "plate.msh", plate(1.0, 1.0, (("oxide", 0.1),), (2, 2, 3)))
"""

from __future__ import annotations

import itertools
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Kuhn split of the unit cube: one tetrahedron per permutation of the axes.
_CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))[:, ::-1]


def _kuhn() -> np.ndarray:
    tets = []
    for perm in itertools.permutations(range(3)):
        path = [np.zeros(3, int)]
        for axis in perm:
            step = path[-1].copy()
            step[axis] = 1
            path.append(step)
        tets.append([int(np.flatnonzero((p == _CORNERS).all(1))[0]) for p in path])
    return np.array(tets)


Classify = Callable[[np.ndarray], str]


def structured(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume: Classify,
    surface: Classify,
    mapping: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    mirror: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    """Tetrahedra of the box grid with axis ``edges``, named by centroid.

    Args:
        edges: Grid coordinates along each parameter axis.
        volume: Region name of a tetrahedron from its parameter centroid.
        surface: Region name of a boundary or interface triangle from its
            parameter centroid; ``""`` leaves the triangle out.
        mapping: Maps parameter points ``(n, 3)`` to coordinates.
        mirror: Per axis, whether the cells of the upper half are the
            mirror images of the lower half's (an even number of cells and
            edges symmetric about their midpoint give a mesh symmetric
            under that reflection; the Kuhn split alone is not).

    Returns:
        nodes, tetrahedra, tetrahedron region names, triangles, triangle
        region names. Interface triangles (between two volume regions) are
        offered to ``surface`` too.
    """
    nx, ny, nz = (len(e) for e in edges)
    grid = np.stack(np.meshgrid(*edges, indexing="ij"), axis=-1).reshape(-1, 3)
    ids = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    cubes = []
    halves = [
        (n - 1) // 2 if m else n for n, m in zip((nx, ny, nz), mirror, strict=True)
    ]
    for i, j, k in itertools.product(range(nx - 1), range(ny - 1), range(nz - 1)):
        flip = [c >= h for c, h in zip((i, j, k), halves, strict=True)]
        cubes.append(
            [
                ids[
                    i + (1 - a if flip[0] else a),
                    j + (1 - b if flip[1] else b),
                    k + (1 - c if flip[2] else c),
                ]
                for a, b, c in _CORNERS
            ]
        )
    tets = np.array(cubes)[:, _kuhn()].reshape(-1, 4)
    nodes = grid if mapping is None else mapping(grid)
    # positive orientation in mapped coordinates
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    tet_names = [volume(c) for c in grid[tets].mean(axis=1)]

    faces = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    uniq, first, inverse, counts = np.unique(
        key, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    owner = np.repeat(np.arange(len(tets)), 4)
    region_of = {}
    for f, t in zip(inverse.ravel(), owner, strict=True):
        region_of.setdefault(int(f), set()).add(tet_names[t])
    tris, tri_names = [], []
    for f in range(len(uniq)):
        if counts[f] == 2 and len(region_of[f]) == 1:
            continue
        name = surface(grid[uniq[f]].mean(axis=0))
        if name:
            tris.append(faces[first[f]])
            tri_names.append(name)
    return nodes, tets, tet_names, np.array(tris).reshape(-1, 3), tri_names


def write_msh22(path: str | Path, mesh: tuple) -> Path:
    """Write a gmsh 2.2 ASCII file; physical numbers follow first appearance.

    Args:
        path: The file to write.
        mesh: ``(nodes, tets, tet_names, tris, tri_names)`` from `structured`.
    """
    path = Path(path)
    nodes, tets, tet_names, tris, tri_names = mesh
    vols = list(dict.fromkeys(tet_names))
    surfs = list(dict.fromkeys(tri_names))
    tag = {(3, n): i + 1 for i, n in enumerate(vols)}
    tag |= {(2, n): len(vols) + i + 1 for i, n in enumerate(surfs)}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines.append(str(len(tag)))
    lines += [f'{dim} {t} "{name}"' for (dim, name), t in tag.items()]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [
        f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)
    ]
    lines += ["$EndNodes", "$Elements", str(len(tris) + len(tets))]
    k = 0
    for tri, name in zip(tris, tri_names, strict=True):
        k += 1
        t = tag[2, name]
        lines.append(f"{k} 2 2 {t} {t} " + " ".join(str(v + 1) for v in tri))
    for tet, name in zip(tets, tet_names, strict=True):
        k += 1
        t = tag[3, name]
        lines.append(f"{k} 4 2 {t} {t} " + " ".join(str(v + 1) for v in tet))
    lines.append("$EndElements")
    # by rename: every rank of a launched test may write this same file, and rank 0
    # must not read another rank's half
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "w") as out:
        out.write("\n".join(lines) + "\n")
    Path(partial).chmod(0o644)
    Path(partial).replace(path)
    return path


def plate(
    width: float,
    depth: float,
    layers: tuple[tuple[str, float], ...],
    n: tuple[int, int, int],
):
    """A parallel-plate stack: ``layers`` of (name, thickness) from z = 0.

    Surfaces: "bottom" (z = 0), "top" (z = total), "side" (x, y walls),
    and "<lower>__<upper>" between layers. ``n`` cells along x, y and each
    layer's z.
    """
    tops = np.cumsum([t for _, t in layers])
    zs = [np.zeros(1)]
    for (_, t), z1 in zip(layers, tops, strict=True):
        zs.append(z1 - t + np.linspace(0, t, n[2] + 1)[1:])
    edges = (
        np.linspace(0, width, n[0] + 1),
        np.linspace(0, depth, n[1] + 1),
        np.concatenate(zs),
    )
    eps = 1e-9 * max(width, depth, tops[-1])

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - tops[-1]) < eps:
            return "top"
        for (lo, _), (hi, _), z in zip(layers, layers[1:], tops, strict=False):
            if abs(c[2] - z) < eps:
                return f"{lo}__{hi}"
        return "side"

    return structured(edges, volume, surface)


def coax_quarter(
    inner: float, outer: float, length: float, *, n_r: int, n_theta: int, n_z: int = 1
):
    """A quarter of a coaxial line, a < r < b, 0 < theta < pi/2, 0 < z < length.

    Volume "dielectric". Surfaces "inner" (r = a), "outer" (r = b),
    "symmetry" (theta = 0 and pi/2) and "ends" (z = 0 and length). The
    polygonal conductors have their nodes on the circles.
    """
    edges = (
        np.linspace(inner, outer, n_r + 1),
        np.linspace(0, np.pi / 2, n_theta + 1),
        np.linspace(0, length, n_z + 1),
    )
    tol = 1e-9

    def surface(c):
        if abs(c[0] - inner) < tol * outer:
            return "inner"
        if abs(c[0] - outer) < tol * outer:
            return "outer"
        if abs(c[2]) < tol * length or abs(c[2] - length) < tol * length:
            return "ends"
        return "symmetry"

    def mapping(p):
        r, t, z = p.T
        return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)

    return structured(edges, lambda _: "dielectric", surface, mapping)


def strip_line(
    length: float,
    width: float,
    height: float,
    n: tuple[int, int, int],
    *,
    layers: tuple[tuple[str, float], ...] | None = None,
    split_ports: bool = False,
    core: tuple[float, float] | None = None,
):
    """A parallel-plate line along x: plates at z = 0 and z = height.

    Volume "dielectric" (or ``layers`` of (name, thickness) from z = 0).
    Surfaces "bottom" (z = 0), "top" (z = height), "port1" (x = 0),
    "port2" (x = length) and "side" (y walls); with ``split_ports`` each
    port face is cut at y = width / 2 into "<port>a" (y below) and
    "<port>b". Layer interfaces are "<lower>__<upper>". ``n`` cells along
    x, y and z (per layer). With ``core = (x0, x1)`` the cells between x0
    and x1 (grid planes) form one volume "core" across every layer; its
    faces towards the layers are left out.
    """
    layers = (("dielectric", height),) if layers is None else layers
    tops = np.cumsum([t for _, t in layers])
    zs = [np.zeros(1)]
    for (_, t), z1 in zip(layers, tops, strict=True):
        zs.append(z1 - t + np.linspace(0, t, n[2] + 1)[1:])
    edges = (
        np.linspace(0, length, n[0] + 1),
        np.linspace(0, width, n[1] + 1),
        np.concatenate(zs),
    )
    eps = 1e-9 * max(length, width, tops[-1])

    def volume(c):
        if core is not None and core[0] < c[0] < core[1]:
            return "core"
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        for x, name in ((0.0, "port1"), (length, "port2")):
            if abs(c[0] - x) < eps:
                if split_ports:
                    return name + ("a" if c[1] < width / 2 else "b")
                return name
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - tops[-1]) < eps:
            return "top"
        for (lo, _), (hi, _), z in zip(layers, layers[1:], tops, strict=False):
            if abs(c[2] - z) < eps:
                return f"{lo}__{hi}"
        return "side"

    def named(c):
        # the core's faces towards the layers are interior: left out
        if core is not None and min(abs(c[0] - x) for x in core) < eps:
            return ""
        return surface(c)

    return structured(edges, volume, named)


def sheet_line(
    length: float,
    width: float,
    height: float,
    n: tuple[int, int, int],
    *,
    span: tuple[float, float] | None = None,
):
    """A parallel-plate line along x glued from two halves at z = height / 2.

    The plates are the y walls, "plate1" (y = 0) and "plate2" (y = width),
    so the TEM field is along y, tangential to the horizontal interior
    surface "sheet" between the volumes "lower" and "upper". Surfaces
    "port1" (x = 0), "port2" (x = length) and "side" (z = 0, height). With
    ``span = (x0, x1)`` only the interface faces between x0 and x1 are
    "sheet" (the rest carries no surface; the node layout is unchanged).
    ``n`` cells along x, y and z (per half; even counts keep the halves
    apart).
    """
    edges = (
        np.linspace(0, length, n[0] + 1),
        np.linspace(0, width, n[1] + 1),
        np.linspace(0, height, 2 * n[2] + 1),
    )
    eps = 1e-9 * max(length, width, height)

    def volume(c):
        return "lower" if c[2] < height / 2 else "upper"

    def surface(c):
        if abs(c[2] - height / 2) < eps:
            if span is not None and not span[0] < c[0] < span[1]:
                return ""
            return "sheet"
        for x, name in ((0.0, "port1"), (length, "port2")):
            if abs(c[0] - x) < eps:
                return name
        if abs(c[1]) < eps:
            return "plate1"
        if abs(c[1] - width) < eps:
            return "plate2"
        return "side"

    return structured(edges, volume, surface)


def periodic_cell(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume: Classify = lambda _: "cell",
    *,
    sheets: tuple[tuple[str, float], ...] = (),
    surface: Classify | None = None,
    paired: str = "xyz",
    mirror: tuple[bool, bool, bool] = (False, False, False),
):
    """A box cell whose opposite faces match under translation, for periodic pairs.

    The Kuhn split makes every pair of opposite faces identical up to the
    translation across the box, so they can be paired by `gfem.Periodic`.

    Args:
        edges: Grid coordinates along x, y and z.
        volume: Region name of a tetrahedron from its centroid.
        sheets: ``(name, x)`` interior surfaces on the grid planes x
            (lumped-port sheets): the volume is split at each plane, so
            every face of the plane is named, whatever the regions on its
            two sides (the tetrahedra keep the names ``volume`` gives).
        surface: Renames boundary faces: a name from the face centroid, or
            ``""`` to keep the default name.
        paired: The axes whose opposite faces are meant to be paired.
        mirror: As in `structured`; refused on a paired axis.

    Surfaces: "x_min", "x_max", "y_min", "y_max", "z_min", "z_max" (the box
    faces, unless ``surface`` renames them) and the ``sheets``.

    Raises:
        ValueError: ``mirror`` on a paired axis.
    """
    for axis, flipped in zip("xyz", mirror, strict=True):
        if flipped and axis in paired:
            raise ValueError(
                f"periodic_cell: mirror on the paired axis {axis!r} is not "
                "supported; a periodic cell needs the plain Kuhn split along "
                "every paired axis."
            )
    lo = np.array([e[0] for e in edges], dtype=float)
    hi = np.array([e[-1] for e in edges], dtype=float)
    eps = 1e-9 * float(np.max(hi - lo))
    planes = np.sort([x for _, x in sheets])

    def split(c):
        return f"{volume(c)}\0{int(np.searchsorted(planes, c[0]))}"

    def named(c):
        for name, x in sheets:
            if abs(c[0] - x) < eps:
                return name
        for axis, a in enumerate("xyz"):
            if abs(c[axis] - lo[axis]) < eps or abs(c[axis] - hi[axis]) < eps:
                side = "min" if abs(c[axis] - lo[axis]) < eps else "max"
                renamed = surface(c) if surface is not None else ""
                return renamed or f"{a}_{side}"
        return ""

    nodes, tets, tet_names, tris, tri_names = structured(
        edges, split, named, mirror=mirror
    )
    return nodes, tets, [n.split("\0")[0] for n in tet_names], tris, tri_names


_DROPPED = "\0dropped"


def carve(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume: Callable[[np.ndarray], str | None],
    surface: Classify,
    mapping: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    """`structured`, with the cells ``volume`` names `None` removed.

    The faces between a kept and a removed tetrahedron are boundary faces
    offered to ``surface``; triangles that touch no kept tetrahedron and
    nodes no kept tetrahedron uses are dropped, and the nodes renumbered.
    """
    nodes, tets, names, tris, tri_names = structured(
        edges, lambda c: volume(c) or _DROPPED, surface, mapping
    )
    keep = np.array([n != _DROPPED for n in names])
    tets = tets[keep]
    names = [n for n, k in zip(names, keep, strict=True) if k]
    faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], axis=2)
    faces = np.unique(faces.reshape(-1, 3), axis=0)
    n = len(nodes)

    def key(f: np.ndarray) -> np.ndarray:
        f = f.astype(np.int64)
        return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]

    on_kept = np.isin(key(np.sort(tris, axis=1)), key(faces))
    tris = tris[on_kept]
    tri_names = [t for t, k in zip(tri_names, on_kept, strict=True) if k]
    used = np.unique(tets)
    renumber = np.full(n, -1)
    renumber[used] = np.arange(len(used))
    return nodes[used], renumber[tets], names, renumber[tris], tri_names


def _graded(lo: float, hi: float, corner: float, n: int, grade: float) -> np.ndarray:
    """Grid lines on [lo, hi], ``n`` cells per unit length, a node at ``corner``.

    ``grade`` > 1 packs them towards ``corner`` as (k / m) ** grade.
    """
    parts = [np.array([corner])]
    for end in (lo, hi):
        length = abs(end - corner)
        if length == 0:
            continue
        m = max(1, round(length * n))
        parts.append(
            corner + np.sign(end - corner) * length * (np.arange(1, m + 1) / m) ** grade
        )
    x = np.unique(np.concatenate(parts))
    x[0], x[-1] = lo, hi
    return x


def square_coax_quarter(n: int, *, grade: float = 1.0):
    """A quarter of a square coax: an L-shaped section with a 270 degree corner.

    Inner conductor of half-side 1, outer of half-side 2, extruded by 0.5 (one
    cell): the section [0, 2]^2 minus [0, 1]^2, whose re-entrant corner is the
    inner conductor's corner (1, 1). ``n`` cells per unit length in x and y;
    ``grade`` > 1 packs them towards the corner.

    Volume "dielectric". Surfaces "inner", "outer", "ends" (z = 0 and 0.5) and
    "symmetry" (x = 0 and y = 0).
    """
    a, b, t, tol = 1.0, 2.0, 0.5, 1e-9
    xy = _graded(0, b, a, n, grade)
    edges = (xy, xy.copy(), np.array([0.0, t]))

    def volume(c):
        return None if c[0] < a and c[1] < a else "dielectric"

    def surface(c):
        x, y, z = c
        if abs(x - b) < tol or abs(y - b) < tol:
            return "outer"
        if (abs(x - a) < tol and y < a + tol) or (abs(y - a) < tol and x < a + tol):
            return "inner"
        if abs(z) < tol or abs(z - t) < tol:
            return "ends"
        return "symmetry"

    return carve(edges, volume, surface)


def step_line(n: int, *, grade: float = 1.0):
    """A parallel-plate line along x whose top plate steps down halfway.

    Length 10, width 2 (one y cell), height 1; for x > 5 the top plate is at
    z = 0.5, so the step's lower edge is a 270 degree re-entrant edge along y at
    (5, 0.5). ``n`` cells per unit length in x and z; ``grade`` > 1 packs them
    towards the edge.

    Volume "dielectric". Surfaces "bottom" (z = 0), "top" (the stepped plate),
    "port1" (x = 0, full height), "port2" (x = 10, half height) and "side"
    (the y walls).
    """
    length, width, height, tol = 10.0, 2.0, 1.0, 1e-9
    edges = (
        _graded(0, length, length / 2, n, grade),
        np.array([0.0, width]),
        _graded(0, height, height / 2, n, grade),
    )

    def volume(c):
        return None if c[0] > length / 2 and c[2] > height / 2 else "dielectric"

    def surface(c):
        x, _, z = c
        if abs(x) < tol:
            return "port1"
        if abs(x - length) < tol:
            return "port2"
        if abs(z) < tol:
            return "bottom"
        if (
            abs(z - height) < tol
            or (abs(x - length / 2) < tol and z > height / 2 - tol)
            or (abs(z - height / 2) < tol and x > length / 2 - tol)
        ):
            return "top"
        return "side"

    return carve(edges, volume, surface)


def l_cavity(n: int, *, grade: float = 1.0):
    """A cavity of L-shaped section: [-1, 1]^2 minus [0, 1] x [-1, 0], height 0.5.

    The re-entrant corner is at the origin. ``n`` cells per unit length in x
    and y, one cell in z; ``grade`` > 1 packs them towards the corner.

    Volume "air". Surface "wall" (the whole boundary).
    """
    xy = _graded(-1, 1, 0, n, grade)
    edges = (xy, xy.copy(), np.array([0.0, 0.5]))

    def volume(c):
        return None if c[0] > 0 and c[1] < 0 else "air"

    return carve(edges, volume, lambda _: "wall")
