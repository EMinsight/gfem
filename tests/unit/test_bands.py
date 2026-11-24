r"""The reciprocal lattice and the wave vector path of a periodic cell.

`Model.reciprocal_lattice` is dual to the periodic translations,
$\mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}$. `gfem.wave_vector_path`
recognises the lattice from the translations, places its high-symmetry
points (the hexagonal K as the zone corner, whatever the basis angle) and
samples straight segments between them with exactly ``n_points`` wave
vectors, every high-symmetry point among them. Geometry only: nothing here
solves.
"""

import itertools
import math

import numpy as np
import periodic_cases as pc
import pytest
from strip2d import write_strip

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


# --- sampling the path ----------------------------------------------------------------

PATHS = {
    "square": ["Γ", "X", "M", "Γ"],
    "rect": ["Γ", "X", "S", "Y", "Γ"],
    "hex60": ["Γ", "M", "K", "Γ"],
    "cubic": ["Γ", "X", "M", "Γ", "R"],
}


@pytest.mark.parametrize("n_points", [None, 7, 31, 100])
@pytest.mark.parametrize("case", PATHS)
def test_n_points_is_exact_and_hits_every_corner(models, case, n_points):
    model, points = models(case), PATHS[case]
    n_points = n_points or len(points)
    path = gfem.wave_vector_path(model, points, n_points=n_points)
    corners = _corners(model, points).wave_vectors
    k = path.wave_vectors
    assert len(path) == len(k) == len(path.distance) == n_points
    assert path.labels == tuple(points)
    index = np.asarray(path.label_index)
    assert index[0] == 0
    assert index[-1] == n_points - 1
    np.testing.assert_array_equal(k[index], corners)
    # the intervals shared by segment length, at least one each
    lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    counts = np.diff(index)
    quota = (n_points - 1) * lengths / lengths.sum()
    assert (counts >= 1).all()
    assert (np.abs(counts - quota) <= 1).all(), (counts, quota)
    # evenly spaced inside each segment, no joint repeated
    steps = np.linalg.norm(np.diff(k, axis=0), axis=1)
    for s, (start, stop) in enumerate(itertools.pairwise(index)):
        np.testing.assert_allclose(
            steps[start:stop], lengths[s] / counts[s], rtol=1e-12
        )
    assert (steps > 0).all()
    np.testing.assert_allclose(np.diff(path.distance), steps, rtol=1e-12)
    assert path.distance[0] == 0
    assert math.isclose(path.distance[-1], lengths.sum(), rel_tol=1e-12)
    # plotting and lookup
    positions, names = path.ticks
    np.testing.assert_array_equal(positions, path.distance[index])
    assert names == points
    assert path.index("Γ") == path.index("G") == path.index("Gamma") == 0
    assert path.index(points[2]) == index[2]


def test_named_points_add_and_override(models):
    square = models("square")
    b1, b2 = np.asarray(square.reciprocal_lattice)
    path = _corners(
        square, ["Γ", "X", "Q"], named_points={"X": (0.25, 0.0), "Q": (0.1, 0.2)}
    )
    np.testing.assert_array_equal(path.wave_vectors[1], 0.25 * b1)
    np.testing.assert_allclose(path.wave_vectors[2], 0.1 * b1 + 0.2 * b2, rtol=1e-15)
    assert path.lattice == "square"
    # fcc-style points on the cubic cell, beside its own X
    cubic = models("cubic")
    g = 2 * math.pi / (pc.C4_L * pc.MM)
    fcc = {"W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5)}
    path = _corners(cubic, ["Γ", "X", "W", "L", "Γ"], named_points=fcc)
    expected = [(0, 0, 0), (0.5, 0, 0), *fcc.values(), (0, 0, 0)]
    np.testing.assert_allclose(path.wave_vectors, g * np.asarray(expected), atol=1e-12)


def test_a_path_of_given_wave_vectors_computes_its_distance():
    k = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 12.0)]
    path = gfem.WaveVectorPath(k, labels=("Γ", "Z"), label_index=(0, 2))
    np.testing.assert_array_equal(path.distance, [0.0, 5.0, 17.0])
    assert path.lattice == "other"
    assert path.index("Z") == 2
    assert repr(path) == "WaveVectorPath(Γ-Z, 3 wave vectors, other lattice)"
    assert repr(gfem.WaveVectorPath(k[:1])) == (
        "WaveVectorPath(1 wave vector, other lattice)"
    )
    with pytest.raises(ValueError, match="read-only"):
        path.wave_vectors[0, 0] = 1.0
    with pytest.raises(ValueError, match=r"shape \(n_k, 3\).*got shape \(3,\)"):
        gfem.WaveVectorPath(k[0])
    with pytest.raises(ValueError, match="'K' is not a label of this path"):
        path.index("K")


def test_a_wave_vector_path_repr(models):
    path = gfem.wave_vector_path(models("square"), ["Γ", "X", "M", "Γ"], n_points=31)
    assert repr(path) == "WaveVectorPath(Γ-X-M-Γ, 31 wave vectors, square lattice)"


# --- refusals -------------------------------------------------------------------------


def _model(models, tmp_path, which):
    """A lattice case of `models`, or one of the models that cannot have a path."""
    if which == "non-model":
        return "cell"
    if which == "2D":
        flat = write_strip(tmp_path / "strip.msh")
        mesh = gfem.Mesh.from_gmsh(flat, length_unit="um")
        return gfem.Model(mesh, materials={k: gfem.Material() for k in "abc"})
    if which == "no pair":
        mesh = pc.c2_mesh(tmp_path, cells=(2, 2, 1))
        return gfem.Model(mesh, materials={"cell": gfem.Material()})
    return models(which)


#: (model, points, keywords, error, message) per refusal.
REFUSALS = {
    "non-model": (
        "non-model",
        ["Γ", "X"],
        {},
        TypeError,
        r"^wave_vector_path: needs a gfem\.Model, got str\.$",
    ),
    "2D": (
        "2D",
        ["Γ", "X"],
        {},
        NotImplementedError,
        (
            r"^gfem\.wave_vector_path is not available on a 2D mesh\. .*PEC faces "
            r"give the Ez bands, natural faces the Hz bands; .*from_gmsh_cross_section"
        ),
    ),
    "no pair": (
        "no pair",
        ["Γ", "X"],
        {},
        ValueError,
        r"no gfem\.Periodic pair, so it has no Bloch wave vector.*Pair the cell's",
    ),
    "unknown name": (
        "square",
        ["Γ", "K"],
        {},
        ValueError,
        (
            r"'K' is not a named point of this lattice \(square: Γ, X, M\); give "
            r"its reduced coordinates in named_points=\{'K': \(…\)\}"
        ),
    ),
    "wrong length": (
        "cubic",
        ["Γ", "W"],
        {"named_points": {"W": (0.5, 0.25)}},
        ValueError,
        r"named_points\['W'\] has 2 coordinates, but the model has 3 periodic pairs",
    ),
    "one point": (
        "square",
        ["Γ"],
        {},
        ValueError,
        r"a path needs at least 2 points, got 1",
    ),
    "repeated point": (
        "square",
        ["Γ", "X", "X", "M"],
        {},
        ValueError,
        r"points 1 and 2 \('X', 'X'\) are the same wave vector",
    ),
    "same wave vector": (
        "square",
        ["Γ", "G"],
        {},
        ValueError,
        r"points 0 and 1 \('Γ', 'Γ'\) are the same wave vector",
    ),
    "too few n_points": (
        "square",
        ["Γ", "X", "M", "Γ", "X"],
        {"n_points": 4},
        ValueError,
        (
            r"n_points=4 is fewer than the 5 points of the path: every "
            r"high-symmetry point is a wave vector"
        ),
    ),
    "one string": ("square", "ΓXM", {}, TypeError, r"points is a sequence of names"),
    "float n_points": (
        "square",
        ["Γ", "X"],
        {"n_points": 7.0},
        TypeError,
        r"n_points must be an integer, got 7\.0",
    ),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_wave_vector_path_refusals(models, tmp_path, case):
    which, points, kwargs, error, message = REFUSALS[case]
    model = _model(models, tmp_path, which)
    with pytest.raises(error, match=message):
        gfem.wave_vector_path(model, points, **({"n_points": 7} | kwargs))
