"""The periodic seam check on a received mesh, against plain loops (ADR 0071).

A received mesh is mimicked on one process: a random half of a periodic box's tets
and triangles, with all of its identifications, so edges and faces go missing.
The loops are mpi-support probe 17's, with faces added to the reverse count.
"""

import itertools

import mpi_cases
import netgen.meshing as ngm
import numpy as np
import pytest

from gfem._engine.mesh import to_ngsolve
from gfem._engine.seam_check import (
    expected_surface_counts,
    seam_counts,
    surface_counts,
)
from gfem._periodic import pairing

TET_EDGES = list(itertools.combinations(range(4), 2))
TET_FACES = list(itertools.combinations(range(4), 3))


def _key(*v):
    return tuple(sorted(v))


def _local_topology(ngmesh):
    edges, faces = set(), set()
    for el in ngmesh.Elements3D():
        v = [p.nr0 for p in el.vertices]
        edges.update(_key(v[a], v[b]) for a, b in TET_EDGES)
        faces.update(_key(v[a], v[b], v[c]) for a, b, c in TET_FACES)
    for el in ngmesh.Elements2D():
        v = [p.nr0 for p in el.vertices]
        edges.update(_key(v[a], v[b]) for a, b in itertools.combinations(range(3), 2))
        faces.add(_key(*v))
    for el in ngmesh.Elements1D():
        edges.add(_key(*[p.nr0 for p in el.vertices]))
    return edges, faces


def _maps(ngmesh):
    out = {}
    for a, b, nr in ngmesh.GetIdentifications():
        out.setdefault(int(nr), {})[b.nr0] = a.nr0
    return out


def _loops(ngmesh):
    """(missing edges, missing faces, reverse edges, reverse faces), by loops."""
    edges, faces = _local_topology(ngmesh)
    me, mf, re, rf = set(), set(), set(), set()
    for m in _maps(ngmesh).values():
        inv = {}
        for b, a in m.items():
            inv.setdefault(a, []).append(b)
        for found, entities, missing in ((edges, edges, me), (faces, faces, mf)):
            for e in entities:
                if all(x in m for x in e):
                    t = _key(*(m[x] for x in e))
                    if t not in found:
                        missing.add(t)
        for found, entities, reverse in ((edges, edges, re), (faces, faces, rf)):
            for e in entities:
                if all(x in inv for x in e):
                    for t in itertools.product(*(inv[x] for x in e)):
                        if _key(*t) not in found:
                            reverse.add(_key(*t))
    return len(me), len(mf), len(re), len(rf)


def _sub_mesh(full, rng):
    """``full``'s points and identifications, a random half of its elements."""
    tets = full.Elements3D().NumPy()["nodes"][:, :4]
    tris = full.Elements2D().NumPy()["nodes"][:, :3]
    m = ngm.Mesh(dim=3)  # ty: ignore[unresolved-attribute]
    for p in full.Points():
        m.Add(ngm.MeshPoint(ngm.Pnt(*p.p)))  # ty: ignore[unresolved-attribute]
    m.Add(ngm.FaceDescriptor(surfnr=1, domin=1, bc=1))  # ty: ignore[unresolved-attribute]
    point = ngm.PointId  # ty: ignore[unresolved-attribute]
    for t in tets[rng.random(len(tets)) < 0.5]:
        m.Add(ngm.Element3D(1, [point(int(v)) for v in t]))  # ty: ignore[unresolved-attribute]
    for t in tris[rng.random(len(tris)) < 0.5]:
        m.Add(ngm.Element2D(1, [point(int(v)) for v in t]))  # ty: ignore[unresolved-attribute]
    for a, b, k in full.GetIdentifications():
        m.AddPointIdentification(point(a.nr), point(b.nr), k, 2)
    return m


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("axes", ["x", "xy"])
def test_the_seam_check_counts_what_the_loops_count(n, axes):
    mesh = mpi_cases.box(n)
    pairs = (pairing(mesh, "xmin", "xmax"),)
    if axes == "xy":
        pairs += (pairing(mesh, "ymin", "ymax"),)
    full = to_ngsolve(mesh, pairs).mesh.ngmesh
    nonzero = np.zeros(4, int)
    for seed in range(4):
        sub = _sub_mesh(full, np.random.default_rng(seed))
        expected = _loops(sub)
        assert tuple(seam_counts(sub)) == expected, seed
        nonzero += np.array(expected) > 0
    # every count is exercised
    assert (nonzero > 0).all()


def test_a_whole_mesh_misses_nothing():
    mesh = mpi_cases.box(3)
    full = to_ngsolve(mesh, (pairing(mesh, "xmin", "xmax"),)).mesh.ngmesh
    assert seam_counts(full).tolist() == [0, 0, 0, 0]


def test_a_mesh_with_no_elements_counts_zero():
    m = ngm.Mesh(dim=3)  # ty: ignore[unresolved-attribute]
    assert seam_counts(m).tolist() == [0, 0, 0, 0]


def test_paired_surfaces_are_expected_twice_and_others_once():
    mesh = mpi_cases.box(2)
    tags = np.unique(mesh.triangle_tags)
    names = {r.tag: r.name for r in mesh.regions}
    paired = {mesh.region("xmin").tag, mesh.region("xmax").tag}
    want = dict(
        zip(
            [names[int(t)] for t in tags],
            expected_surface_counts(mesh.triangle_tags, tags, paired).tolist(),
            strict=True,
        )
    )
    # a side wall of a box of 2 cells a side holds 2 x 2 squares of 2 triangles
    assert want["xmin"] == want["xmax"] == 16
    assert want["ymin"] == want["ymax"] == want["port"] == want["abc"] == 8


def test_local_surface_counts_count_each_held_triangle():
    mesh = mpi_cases.box(2)
    tags = np.unique(mesh.triangle_tags)
    xmin = mesh.region("xmin").tag
    held = np.flatnonzero(mesh.triangle_tags == xmin)[:3]
    counts = surface_counts(mesh.triangle_tags, np.r_[held, held[:1]], tags)
    assert counts[tags == xmin].tolist() == [4]
    assert counts.sum() == 4
