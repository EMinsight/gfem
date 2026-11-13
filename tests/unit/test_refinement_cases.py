"""The singular benchmark cases of the refinement tests (tests/refinement_cases.py).

Their meshes come from `structured_meshes` without gmsh; an order-2 uniform
solve at n = 2 reproduces the errors measured when the cases were chosen
(coax 8.54e-3, step S21 at 3 GHz 1.22e-3, cavity f1 5.66e-3, relative).
"""

import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

BUILDERS = {
    "coax": (
        sm.square_coax_quarter,
        {"dielectric"},
        {"inner", "outer", "ends", "symmetry"},
    ),
    "step": (
        sm.step_line,
        {"dielectric"},
        {"port1", "port2", "bottom", "top", "side"},
    ),
    "cavity": (sm.l_cavity, {"air"}, {"wall"}),
}


def _volumes(nodes, tets):
    p = nodes[tets]
    return np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )


@pytest.mark.parametrize("case", list(BUILDERS))
@pytest.mark.parametrize("grade", [1.0, 3.0])
def test_singular_builders_are_deterministic_positive_and_named(case, grade):
    build, volumes, surfaces = BUILDERS[case]
    first, second = build(2, grade=grade), build(2, grade=grade)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    nodes, tets, tet_names, tris, tri_names = first
    assert (_volumes(nodes, tets) > 0).all()
    assert set(tet_names) == volumes
    assert set(tri_names) == surfaces
    # every node is used, and every triangle is a face of a kept tetrahedron
    assert np.array_equal(np.unique(tets), np.arange(len(nodes)))
    faces = np.sort(tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], axis=2)
    faces = {tuple(f) for f in faces.reshape(-1, 3)}
    assert all(tuple(t) in faces for t in np.sort(tris, axis=1))


def test_carve_drops_the_cells_named_none():
    edges = (np.linspace(0, 2, 3), np.linspace(0, 2, 3), np.linspace(0, 1, 2))

    def volume(c):
        return None if c[0] < 1 and c[1] < 1 else "kept"

    def surface(c):
        on_wall = (abs(c[0] - 1) < 1e-9 and c[1] < 1) or (
            abs(c[1] - 1) < 1e-9 and c[0] < 1
        )
        return "hole" if on_wall else "box"

    nodes, tets, names, _, tri_names = sm.carve(edges, volume, surface)
    assert len(tets) == 3 * 6
    assert names == ["kept"] * len(tets)
    centroids = nodes[tets].mean(axis=1)
    assert not ((centroids[:, 0] < 1) & (centroids[:, 1] < 1)).any()
    # the hole's two walls, two triangles each; its own outer faces are gone
    assert tri_names.count("hole") == 4
    assert len(nodes) == 18 - 2


def test_grade_packs_cells_towards_the_corner():
    nodes, *_ = sm.square_coax_quarter(4, grade=3.0)
    xs = np.unique(nodes[:, 0])
    corner = int(np.flatnonzero(np.isclose(xs, 1.0))[0])
    assert xs[corner] - xs[corner - 1] < 0.1 * (xs[1] - xs[0])
    assert xs[corner + 1] - xs[corner] < 0.1 * (xs[-1] - xs[-2])


def test_uniform_coax_reproduces_the_measured_error(tmp_path):
    c = rc.coax_capacitance(rc.solve(rc.coax_model(tmp_path, 2), rc.COAX_PROBLEM))
    error = abs(c - rc.COAX_C) / rc.COAX_C
    assert error == pytest.approx(8.54e-3, rel=0.02)


def test_uniform_step_line_reproduces_the_measured_error(tmp_path):
    s = rc.step_s(rc.solve(rc.step_model(tmp_path, 2), rc.step_problem((3e9,))))
    reference = rc.STEP_S[3e9]
    error = abs(s[0, 1, 0] - reference["S21"]) / abs(reference["S21"])
    assert error == pytest.approx(1.22e-3, rel=0.02)


def test_uniform_cavity_reproduces_the_measured_error(tmp_path):
    f = rc.cavity_frequency(rc.solve(rc.cavity_model(tmp_path, 2), rc.CAVITY_PROBLEM))
    error = abs(f - rc.CAVITY_F1) / rc.CAVITY_F1
    assert error == pytest.approx(5.66e-3, rel=0.02)


def test_the_references_agree_with_their_recorded_forms():
    # the quarter coax model's C, recorded with the reference
    quarter = rc.COAX_C * 8.8541878128e-12 * 0.5e-6 / 4
    assert quarter == pytest.approx(1.1326822e-17, rel=1e-7)
    # f1 = c sqrt(lambda1) / (2 pi), the L's side in mm
    f1 = 299792458 * np.sqrt(rc.CAVITY_LAMBDA1) * 1e3 / (2 * np.pi)
    assert f1 == pytest.approx(rc.CAVITY_F1, rel=1e-9)
    assert all(
        entry[f"{s}_bound"] > 0 for entry in rc.STEP_S.values() for s in ("S21", "S11")
    )
