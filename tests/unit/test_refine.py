"""`gfem._refine.bisect`: longest-edge bisection of marked tetrahedra."""

from pathlib import Path

import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm

import gfem
from gfem._refine import bisect

DATA = Path(__file__).parent / "data"
EPS0 = scipy.constants.epsilon_0
FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


@pytest.fixture(scope="module")
def cpw():
    return gfem.Mesh.from_gmsh(DATA / "cpw_section.msh", length_unit="um")


def _near_line(mesh, fraction=0.1):
    """Mark the tetrahedra closest to a vertical line: a local, singular-like mark."""
    x = np.asarray(mesh.imported_nodes)
    c = x[mesh.tetrahedra].mean(axis=1)
    lo, span = x.min(axis=0), np.ptp(x, axis=0)
    axis = lo + np.array([0.37, 0.41, 0.0]) * span
    d = np.linalg.norm((c - axis)[:, :2], axis=1)
    marked = np.zeros(len(c), bool)
    marked[np.argsort(d, kind="stable")[: max(1, int(fraction * len(c)))]] = True
    return marked


def _signed_volumes(mesh):
    x = np.asarray(mesh.imported_nodes)[mesh.tetrahedra]
    return np.linalg.det(x[:, 1:] - x[:, :1]) / 6


def _boundary_faces(tets):
    faces = np.sort(tets[:, FACES].reshape(-1, 3), axis=1)
    u, counts = np.unique(faces, axis=0, return_counts=True)
    return u[counts == 1], counts


def _area(x, tris):
    a, b, c = (x[tris[:, k]] for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def test_refined_mesh_is_conforming_and_fills_the_same_volume(cpw):
    fine, _ = bisect(cpw, _near_line(cpw))
    assert len(fine.tetrahedra) > len(cpw.tetrahedra)
    once, counts = _boundary_faces(fine.tetrahedra)
    assert counts.max() == 2
    assert np.all(np.abs(_signed_volumes(fine)) > 0)
    np.testing.assert_allclose(
        np.abs(_signed_volumes(fine)).sum(),
        np.abs(_signed_volumes(cpw)).sum(),
        rtol=1e-13,
    )
    old_once, _ = _boundary_faces(cpw.tetrahedra)
    np.testing.assert_allclose(
        _area(np.asarray(fine.imported_nodes), once),
        _area(np.asarray(cpw.imported_nodes), old_once),
        rtol=1e-13,
    )


def _barycentric(x, tets, points):
    """Barycentric coordinates of each point in its tetrahedron."""
    p = x[tets]
    t = np.transpose(p[:, 1:] - p[:, :1], (0, 2, 1))
    lam = np.linalg.solve(t, (points - p[:, 0])[..., None])[..., 0]
    return np.column_stack([1 - lam.sum(axis=1), lam])


def test_children_lie_in_their_parent_and_keep_its_tags_and_orientation(cpw):
    fine, ref = bisect(cpw, _near_line(cpw))
    parents = ref.tetrahedron_parents
    assert parents.shape == (len(fine.tetrahedra),)
    x = np.asarray(fine.imported_nodes)
    centroids = x[fine.tetrahedra].mean(axis=1)
    lam = _barycentric(x, cpw.tetrahedra[parents], centroids)
    assert lam.min() > 0
    v, v0 = _signed_volumes(fine), _signed_volumes(cpw)
    np.testing.assert_array_equal(np.sign(v), np.sign(v0[parents]))
    np.testing.assert_allclose(
        np.bincount(parents, weights=v, minlength=len(v0)), v0, rtol=1e-12
    )
    np.testing.assert_array_equal(fine.tetrahedron_tags, cpw.tetrahedron_tags[parents])
    np.testing.assert_array_equal(
        fine.tetrahedron_entities, cpw.tetrahedron_entities[parents]
    )


def _normals(x, tris):
    a, b, c = (x[tris[:, k]] for k in range(3))
    return 0.5 * np.cross(b - a, c - a)


def test_triangles_split_with_their_faces_and_keep_tags_and_normals(cpw):
    fine, ref = bisect(cpw, _near_line(cpw))
    parents = ref.triangle_parents
    assert len(fine.triangles) > len(cpw.triangles)
    x, x0 = np.asarray(fine.imported_nodes), np.asarray(cpw.imported_nodes)
    n, n0 = _normals(x, fine.triangles), _normals(x0, cpw.triangles)
    # same direction, and the children's vector areas add up to the parent's
    assert np.all(np.einsum("ij,ij->i", n, n0[parents]) > 0)
    summed = np.stack(
        [np.bincount(parents, weights=n[:, k], minlength=len(n0)) for k in range(3)],
        axis=1,
    )
    np.testing.assert_allclose(summed, n0, atol=1e-12 * np.abs(n0).max())
    np.testing.assert_array_equal(fine.triangle_tags, cpw.triangle_tags[parents])
    np.testing.assert_array_equal(
        fine.triangle_entities, cpw.triangle_entities[parents]
    )
    # every triangle is a face of the refined tetrahedra
    faces = {tuple(f) for f in np.sort(fine.tetrahedra[:, FACES].reshape(-1, 3), 1)}
    assert all(tuple(t) in faces for t in np.sort(fine.triangles, axis=1))


def test_triangle_neighbours_are_recomputed_on_the_same_sides(cpw):
    fine, ref = bisect(cpw, _near_line(cpw))
    nb, nb0 = fine.triangle_neighbours, cpw.triangle_neighbours
    coarse_side = nb0[ref.triangle_parents]
    np.testing.assert_array_equal(nb < 0, coarse_side < 0)
    inside = nb >= 0
    np.testing.assert_array_equal(
        ref.tetrahedron_parents[nb[inside]], coarse_side[inside]
    )
    for surface in cpw.surfaces:
        assert fine.adjacency(surface) == cpw.adjacency(surface)


def test_every_marked_tetrahedron_is_bisected_and_the_result_is_deterministic(cpw):
    marked = _near_line(cpw)
    fine, ref = bisect(cpw, marked)
    again, ref_again = bisect(cpw, marked.copy())
    children = np.bincount(ref.tetrahedron_parents, minlength=len(marked))
    assert children[marked].min() >= 2
    assert children.min() >= 1
    for name in ("imported_nodes", "tetrahedra", "triangles", "triangle_neighbours"):
        np.testing.assert_array_equal(getattr(fine, name), getattr(again, name))
    np.testing.assert_array_equal(
        ref.tetrahedron_parents, ref_again.tetrahedron_parents
    )
    np.testing.assert_array_equal(ref.triangle_parents, ref_again.triangle_parents)


def _unrefined(coarse, fine, parents, kind):
    """Rows of the fine mesh that are coarse elements left whole, and their parents."""
    whole = np.bincount(parents, minlength=len(getattr(coarse, kind))) == 1
    rows = np.flatnonzero(whole[parents])
    np.testing.assert_array_equal(
        getattr(fine, kind)[rows], getattr(coarse, kind)[parents[rows]]
    )
    return rows, parents[rows]


def test_old_nodes_and_unrefined_elements_keep_their_indices_and_order(cpw):
    fine, ref = bisect(cpw, _near_line(cpw))
    n = len(cpw.imported_nodes)
    np.testing.assert_array_equal(fine.imported_nodes[:n], cpw.imported_nodes)
    np.testing.assert_array_equal(np.asarray(fine.nodes), fine.imported_nodes)
    for kind, parents in (
        ("tetrahedra", ref.tetrahedron_parents),
        ("triangles", ref.triangle_parents),
    ):
        rows, old = _unrefined(cpw, fine, parents, kind)
        assert len(rows) > 0
        assert np.all(np.diff(old) > 0)
    # the refined elements' new vertices are all midpoints, appended
    assert fine.tetrahedra.max() == len(fine.imported_nodes) - 1


def test_refined_mesh_is_a_new_in_memory_topology(cpw):
    fine, _ = bisect(cpw, _near_line(cpw))
    assert fine.source is None
    assert fine.length_unit == cpw.length_unit
    assert fine.regions == cpw.regions
    assert fine._topology is not cpw._topology
    assert not fine.tetrahedra.flags.writeable


def test_an_empty_mark_changes_nothing_but_the_source(cpw):
    fine, ref = bisect(cpw, np.zeros(len(cpw.tetrahedra), bool))
    np.testing.assert_array_equal(fine.tetrahedra, cpw.tetrahedra)
    np.testing.assert_array_equal(fine.triangles, cpw.triangles)
    np.testing.assert_array_equal(
        ref.tetrahedron_parents, np.arange(len(cpw.tetrahedra))
    )
    assert fine.source is None


@pytest.mark.parametrize(
    "marked",
    [np.zeros(3, bool), np.zeros(5829, int), np.zeros((5829, 1), bool)],
    ids=["length", "dtype", "shape"],
)
def test_a_mark_that_is_not_a_boolean_mask_over_the_tetrahedra_raises(cpw, marked):
    with pytest.raises(ValueError, match=r"boolean array of shape \(5829,\)"):
        bisect(cpw, marked)


def _min_dihedral_deg(x, tets):
    """The smallest dihedral angle of the mesh, in degrees."""
    p = x[tets]
    normals = []
    for k, face in enumerate(FACES):
        a, b, c = (p[:, i] for i in face)
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        away = np.sign(np.einsum("ij,ij->i", n, p[:, k] - a))
        normals.append(-n * away[:, None])  # outward
    largest_cos = max(
        (-np.einsum("ij,ij->i", normals[i], normals[j])).max()
        for i in range(4)
        for j in range(i + 1, 4)
    )
    return np.degrees(np.arccos(min(largest_cos, 1.0)))


def _off_plane(x, tris):
    """Largest distance of a surface's nodes from their plane, over its size."""
    p = x[np.unique(tris)]
    c = p - p.mean(axis=0)
    normal = np.linalg.svd(c, full_matrices=False)[2][-1]
    return np.abs(c @ normal).max() / np.ptp(p, axis=0).max()


def test_five_cycles_keep_ports_planar_and_dihedral_angles_bounded(cpw):
    start = _min_dihedral_deg(np.asarray(cpw.imported_nodes), cpw.tetrahedra)
    mesh = cpw
    for _ in range(5):
        mesh, _ = bisect(mesh, _near_line(mesh))
    x = np.asarray(mesh.imported_nodes)
    assert len(mesh.tetrahedra) > 2 * len(cpw.tetrahedra)
    assert _min_dihedral_deg(x, mesh.tetrahedra) >= 0.75 * start
    for port in ("port1", "port2"):
        tris = mesh.triangles[mesh.elements(port)]
        assert len(tris) > len(cpw.elements(port))
        assert _off_plane(x, tris) < 1e-12


def test_refined_mesh_builds_in_the_engine_and_solves(tmp_path):
    path = tmp_path / "plate.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, (("a", 0.5), ("b", 0.25)), (2, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    for _ in range(2):
        mesh, _ = bisect(mesh, _near_line(mesh, fraction=0.3))
    model = gfem.Model(
        mesh, materials={"a": gfem.Material(eps_r=3.9), "b": gfem.Material(eps_r=7.5)}
    )
    cap = gfem.solve(model, gfem.Electrostatic(terminals="top", ground="bottom"))
    # two plate capacitors in series; the potential is linear, exact on any mesh
    series = EPS0 * 12e-12 / (0.5e-6 / 3.9 + 0.25e-6 / 7.5)
    np.testing.assert_allclose(cap.capacitance[0, 0], series, rtol=1e-8)
