r"""The reciprocal lattice and the wave vector path of a periodic cell.

`Model.reciprocal_lattice` is dual to the periodic translations,
$\mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}$. `gfem.wave_vector_path`
recognises the lattice from the translations, places its high-symmetry
points (the hexagonal K as the zone corner, whatever the basis angle) and
samples straight segments between them with exactly ``n_points`` wave
vectors, every high-symmetry point among them. Geometry only: nothing here
solves.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest

import gfem

LATTICES = ("line", "rect", "square", "cubic", "hex60", "hex120", "oblique75")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """One model per lattice case, built lazily."""
    built = {}
    builders = {
        "line": lambda d: pc.c3(pc.c3_mesh(d)),
        "rect": lambda d: pc.c2(pc.c2_mesh(d, cells=(2, 2, 1))),
        "square": lambda d: pc.c2(pc.c2_square_mesh(d, cells=(2, 2, 1))),
        "cubic": lambda d: pc.c4(pc.c4_mesh(d, cells=(2, 2, 2))),
        "hex60": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 60)),
        "hex120": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 120)),
        "oblique75": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 75)),
    }

    def get(name):
        if name not in built:
            built[name] = builders[name](tmp_path_factory.mktemp(name))
        return built[name]

    return get


@pytest.mark.parametrize("case", LATTICES)
def test_reciprocal_lattice_is_dual_to_the_translations(models, case):
    model = models(case)
    a = np.asarray(model.periodic_translations)
    b = np.asarray(model.reciprocal_lattice)
    assert b.shape == a.shape
    np.testing.assert_allclose(
        b @ a.T, 2 * math.pi * np.eye(len(a)), rtol=0, atol=1e-12
    )
    # inside the span of the translations
    basis, _ = np.linalg.qr(a.T)
    np.testing.assert_allclose(basis @ (basis.T @ b.T), b.T, rtol=0, atol=1e-9)


def test_reciprocal_lattice_is_empty_without_a_pair(tmp_path):
    mesh = pc.c2_mesh(tmp_path, cells=(2, 2, 1))
    model = gfem.Model(mesh, materials={"cell": gfem.Material()})
    assert model.reciprocal_lattice == ()


# --- lattices and their named points -------------------------------------------------

#: The lattice each case is recognised as, and a path through its named points.
RECOGNISED = {
    "line": ("line", ["Γ", "X"]),
    "rect": ("rectangular", ["Γ", "X", "S", "Y"]),
    "square": ("square", ["Γ", "X", "M"]),
    "cubic": ("cubic", ["Γ", "X", "M", "R"]),
    "hex60": ("hexagonal", ["Γ", "M", "K"]),
    "hex120": ("hexagonal", ["Γ", "M", "K"]),
    "oblique75": ("other", ["Γ", "W"]),
}


def _corners(model, points, **kwargs):
    """The path through ``points`` with one wave vector per point: the corners."""
    return gfem.wave_vector_path(model, points, n_points=len(points), **kwargs)


@pytest.mark.parametrize("case", LATTICES)
def test_the_lattice_is_recognised_from_the_translations(models, case):
    lattice, points = RECOGNISED[case]
    named = {"W": (0.5, 0.5)} if lattice == "other" else None
    path = _corners(models(case), points, named_points=named)
    assert path.lattice == lattice
    assert path.labels == tuple(points)


@pytest.mark.parametrize("case", ["hex60", "hex120"])
def test_hexagonal_k_is_the_zone_corner_for_both_bases(models, case):
    model = models(case)
    a = np.linalg.norm(model.periodic_translations[0])
    b1, b2 = np.asarray(model.reciprocal_lattice)
    gamma, m, k = _corners(model, ["Γ", "M", "K"]).wave_vectors
    np.testing.assert_array_equal(gamma, 0.0)
    np.testing.assert_allclose(m, b1 / 2, rtol=0, atol=1e-12)
    assert abs(np.linalg.norm(k) - 4 * math.pi / (3 * a)) < 1e-12 * np.linalg.norm(k)
    # the corner of the zone: equidistant from Γ, b1 and the ±b2 at 60° to b1
    near = b2 if b1 @ b2 > 0 else -b2
    assert math.isclose(b1 @ near, 0.5 * (b1 @ b1), rel_tol=1e-12)
    distances = [np.linalg.norm(k - g) for g in (np.zeros(3), b1, near)]
    np.testing.assert_allclose(distances, distances[0], rtol=1e-12)


def test_named_points_of_each_lattice(models):
    c3 = pc.C3_A * pc.MM
    ax, ay = pc.C2_AX * pc.MM, pc.C2_AY * pc.MM
    c4 = pc.C4_L * pc.MM
    pi = math.pi
    expected = {
        "line": {"X": (pi / c3, 0, 0)},
        "square": {"X": (pi / ax, 0, 0), "M": (pi / ax, pi / ax, 0)},
        "rect": {
            "X": (pi / ax, 0, 0),
            "Y": (0, pi / ay, 0),
            "S": (pi / ax, pi / ay, 0),
        },
        # X along the first pair
        "cubic": {
            "X": (pi / c4, 0, 0),
            "M": (pi / c4, pi / c4, 0),
            "R": (pi / c4, pi / c4, pi / c4),
        },
    }
    for case, points in expected.items():
        path = _corners(models(case), ["Γ", *points])
        np.testing.assert_array_equal(path.wave_vectors[0], 0.0)
        for name, k in points.items():
            got = path.wave_vectors[path.index(name)]
            scale = np.linalg.norm(k)
            np.testing.assert_allclose(got, k, rtol=0, atol=1e-14 * scale)
