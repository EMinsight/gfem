r"""Write the sphere-scattering mesh: a sphere in an air cube in a box layer.

A sphere ("sphere", radius `RADIUS`, surface "shell") in a cube of air
("air", half-side `HALF`) wrapped in a perfectly matched layer ("pml",
`THICKNESS` thick), meshed by netgen (unstructured, straight tetrahedra)
and written as a gmsh 2.2 file, lengths in micrometres. The layer is made
of boxes split along the cube's planes, so that no element straddles one
of them. The cross-sections recipe downloads `docs/tutorials/data/sphere.msh`
(the defaults); the free-space tests build their variants.

Usage: ``python tools/sphere_mesh.py docs/tutorials/data`` (``just sphere-mesh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import ngsolve as ngs
import numpy as np
import structured_meshes as sm

#: The sphere's radius, the cube's half-side and the layer's thickness, um.
RADIUS = 0.25
HALF = 0.8
THICKNESS = 0.35
#: Half-side of a box of air around the sphere, um: its faces hold the mesh
#: near the sphere finer than the global size.
INNER = 0.42
#: The global and the sphere's maximal edge lengths, um.
MAXH = 0.4
SPHERE_MAXH = 0.15


def write(
    path: str | Path,
    *,
    maxh: float = MAXH,
    sphere_maxh: float = SPHERE_MAXH,
    half: bool = False,
    mirrored: bool = False,
) -> float:
    r"""Write the mesh to ``path``; return the meshed sphere's volume, um³.

    With ``half`` only :math:`y \ge 0` is meshed, and the plane
    :math:`y = 0` is the surface region "mirror"; the volume returned is
    then the whole sphere's (twice the meshed half). With ``mirrored`` the
    half is meshed and reflected across :math:`y = 0`: a whole mesh whose
    every element has its mirror image, so that a field symmetric about the
    plane is the half model's.
    """
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere

    half = half or mirrored

    ball = Sphere(Pnt(0, 0, 0), RADIUS)
    ball.mat("sphere")
    ball.maxh = sphere_maxh
    ball.solids.maxh = sphere_maxh
    ball.faces.maxh = sphere_maxh
    ball.faces.name = "shell"
    h, t = HALF, HALF + THICKNESS
    low = 0.0 if half else -h
    near = Box(Pnt(-INNER, 0.0 if half else -INNER, -INNER), Pnt(INNER, INNER, INNER))
    air = Box(Pnt(-h, low, -h), Pnt(h, h, h)) - near
    air.mat("air")
    inner = near - ball
    inner.mat("air")
    if half:
        ball = ball * Box(Pnt(-t, 0, -t), Pnt(t, t, t))
        ball.mat("sphere")
        ball.faces.name = "shell"
    # the layer as boxes split along the interior box's planes, as the layer
    # needs: no element straddles one of them
    cuts = [(-t, -h), (-h, h), (h, t)]
    y_cuts = [(0.0, h), (h, t)] if half else cuts
    pml = []
    for x0, x1 in cuts:
        for y0, y1 in y_cuts:
            for z0, z1 in cuts:
                if (x0, z0) == (-h, -h) and y0 in (-h, 0.0):
                    continue
                cell = Box(Pnt(x0, y0, z0), Pnt(x1, y1, z1))
                cell.mat("pml")
                pml.append(cell)
    shape = Glue([ball, inner, air, *pml])
    if half:
        for face in shape.faces:
            if abs(face.center.y) < 1e-9:
                face.name = "mirror"
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
    tet_names = [el.mat for el in mesh.Elements(ngs.VOL)]
    named = ("shell", "mirror")
    bnd = [el for el in mesh.Elements(ngs.BND) if el.mat in named]
    tris = np.array([[v.nr for v in el.vertices] for el in bnd]).reshape(-1, 3)
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    inside = np.array([n == "sphere" for n in tet_names])
    volume = np.abs(det[inside]).sum() / 6 * (2 if half else 1)
    if mirrored:
        nodes, tets, tet_names, tris, tri_names = _reflect(
            nodes, tets, tet_names, tris, [el.mat for el in bnd]
        )
        sm.write_msh22(path, (nodes, tets, tet_names, tris, tri_names))
        return float(volume)
    sm.write_msh22(path, (nodes, tets, tet_names, tris, [el.mat for el in bnd]))
    return float(volume)


def _reflect(
    nodes: np.ndarray,
    tets: np.ndarray,
    tet_names: list,
    tris: np.ndarray,
    tri_names: list,
) -> tuple:
    """The half mesh and its image across y = 0, glued on the plane."""
    on_plane = np.abs(nodes[:, 1]) < 1e-12
    image = np.arange(len(nodes)) + len(nodes)
    image[on_plane] = np.flatnonzero(on_plane)
    mirror = nodes * np.array([1.0, -1.0, 1.0])
    all_nodes = np.concatenate([nodes, mirror])
    # a reflection flips orientation: swap two vertices back
    images = image[tets][:, [0, 2, 1, 3]]
    keep = [k for k, name in enumerate(tri_names) if name != "mirror"]
    tris = tris[keep]
    names = [tri_names[k] for k in keep]
    used = np.unique(np.concatenate([tets, images]))
    renumber = np.full(len(all_nodes), -1)
    renumber[used] = np.arange(len(used))
    return (
        all_nodes[used],
        renumber[np.concatenate([tets, images])],
        list(tet_names) * 2,
        renumber[np.concatenate([tris, image[tris]])],
        names * 2,
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]) / "sphere.msh"
    print(f"{out}: sphere volume {write(out):.6f} um^3")
