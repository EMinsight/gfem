"""Unstructured meshes shared by the thermal tests (tests/unit, tests/verification).

Netgen (shipped with NGSolve) meshes the geometry, so the elements are
unstructured; the meshes are written as gmsh 2.2 files and read back
through gfem's readers, the path a user's mesh takes. Every builder takes
the file to write (in a pytest ``tmp_path_factory.mktemp`` directory).
Lengths are in micrometres.

- `section`: a 2D cross-section of glued rectangles, triangles and named
  edges, read by `gfem.Mesh.from_gmsh_cross_section`.
- `brick`: a 3D box of stacked layers, tetrahedra and named faces, read by
  `gfem.Mesh.from_gmsh`.
- `extruded`: a cross-section extruded along z in prism layers split into
  tetrahedra (a z-invariant 3D twin of `section`), read by
  `gfem.Mesh.from_gmsh`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import ngsolve as ngs
import numpy as np
import structured_meshes as sm

import gfem

#: A rectangle: (x0, y0, x1, y1, region name, local maximal edge length or None).
Rect = tuple[float, float, float, float, str, float | None]
#: An edge (or face) namer: (predicate on its centre (x, y, z), name).
Namer = tuple[Callable[[tuple[float, float, float]], bool], str]


def _named(item: object, namers: Sequence[Namer]) -> str | None:
    c = item.center  # ty: ignore[unresolved-attribute]
    point = (c.x, c.y, c.z)
    return next((name for test, name in namers if test(point)), None)


def _mesh2d(rects: Sequence[Rect], maxh: float, edges: Sequence[Namer]):
    """Netgen's triangles of glued rectangles: nodes, (name, vertices) tris, lines."""
    from netgen.occ import Glue, OCCGeometry, Rectangle

    faces = []
    for x0, y0, x1, y1, name, h in rects:
        face = Rectangle(x1 - x0, y1 - y0).Face().Move((x0, y0, 0))
        face.faces.name = name
        if h is not None:
            face.faces.maxh = h
        faces.append(face)
    shape = Glue(faces)
    for edge in shape.edges:
        name = _named(edge, edges)
        if name is not None:
            edge.name = name
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    names = {name for _, name in edges}
    tris = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat in names
    ]
    return nodes, tris, lines


def section(
    path: Path,
    rects: Sequence[Rect],
    maxh: float,
    edges: Sequence[Namer],
    *,
    thickness: float | None = None,
) -> gfem.Mesh:
    """A cross-section of glued rectangles, named edges only, as a slab mesh."""
    nodes, tris, lines = _mesh2d(rects, maxh, edges)
    _write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh_cross_section(path, thickness=thickness)


def extruded(
    path: Path,
    rects: Sequence[Rect],
    maxh: float,
    edges: Sequence[Namer],
    *,
    length: float,
    layers: int,
    region: Callable[[int, str], str] | None = None,
) -> gfem.Mesh:
    """A cross-section extruded along z over ``length`` in ``layers`` prism layers.

    Each prism splits into three tetrahedra by the smallest-index rule, so
    the layers conform; a named edge becomes a named face along the
    extrusion. ``region(layer, name)`` renames a triangle's region in a layer
    (a heater in some layers only). Read by `gfem.Mesh.from_gmsh`.
    """
    xy, tris, lines = _mesh2d(rects, maxh, edges)
    n = len(xy)
    nodes = np.zeros(((layers + 1) * n, 3))
    for i in range(layers + 1):
        nodes[i * n : (i + 1) * n, :2] = xy[:, :2]
        nodes[i * n : (i + 1) * n, 2] = length * i / layers
    tets, tet_names, faces, face_names = [], [], [], []
    for i in range(layers):
        o = i * n
        for name, tri in tris:
            a, b, c = sorted(tri)
            a, b, c = a + o, b + o, c + o
            big_a, big_b, big_c = a + n, b + n, c + n
            for q in ((a, b, c, big_c), (a, b, big_b, big_c), (a, big_a, big_b, big_c)):
                tets.append(q)
                tet_names.append(name if region is None else region(i, name))
        for name, ends in lines:
            p, q = (v + o for v in sorted(ends))
            faces += [(p, q, q + n), (p, p + n, q + n)]
            face_names += [name, name]
    tets = np.array(tets)
    v = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    sm.write_msh22(path, (nodes, tets, tet_names, np.array(faces), face_names))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _write_section(path: Path, nodes: np.ndarray, tris: list, lines: list) -> None:
    """Triangles and lines as gmsh 2.2; physical numbers by first appearance."""
    tags: dict[tuple[int, str], int] = {}
    for dim, items in ((1, lines), (2, tris)):
        for name, _ in items:
            tags.setdefault((dim, name), len(tags) + 1)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    out.append(str(len(tags)))
    out += [f'{dim} {t} "{name}"' for (dim, name), t in tags.items()]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(tris))]
    k = 0
    for dim, kind, items in ((1, 1, lines), (2, 2, tris)):
        for name, vertices in items:
            k += 1
            t = tags[dim, name]
            out.append(
                f"{k} {kind} 2 {t} {t} " + " ".join(str(v + 1) for v in vertices)
            )
    out.append("$EndElements")
    path.write_text("\n".join(out) + "\n")


def brick(
    path: Path,
    size: tuple[float, float],
    layers: Sequence[tuple[str, float]],
    maxh: float,
    faces: Sequence[Namer],
) -> gfem.Mesh:
    """A ``size[0]`` x ``size[1]`` box of ``layers`` (name, thickness) up z."""
    from netgen.occ import Box, Glue, OCCGeometry

    boxes, z = [], 0.0
    for name, t in layers:
        box = Box((0, 0, z), (size[0], size[1], z + t))
        box.mat(name)
        boxes.append(box)
        z += t
    shape = Glue(boxes)
    for face in shape.faces:
        name = _named(face, faces)
        if name is not None:
            face.name = name
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    tet_names = [el.mat for el in mesh.Elements(ngs.VOL)]
    names = {name for _, name in faces}
    bnd = [el for el in mesh.Elements(ngs.BND) if el.mat in names]
    tris = np.array([[v.nr for v in el.vertices] for el in bnd]).reshape(-1, 3)
    # positive orientation, as gmsh writes it
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    sm.write_msh22(path, (nodes, tets, tet_names, tris, [el.mat for el in bnd]))
    return gfem.Mesh.from_gmsh(path, length_unit="um")
