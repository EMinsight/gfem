r"""The reciprocal lattice and the wave vector path of a periodic cell.

`Model.reciprocal_lattice` is dual to the periodic translations,
:math:`\mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}`. `gfem.wave_vector_path`
recognises the lattice from the translations, places its high-symmetry
points (the hexagonal K as the zone corner, whatever the basis angle) and
samples straight segments between them with exactly ``n_points`` wave
vectors, every high-symmetry point among them: geometry only. The sweep
`gfem.band_diagram` is tested below: its refusals, its band order, the
Γ fill, the connections and the flags on a cheap C2 cell.
"""

import dataclasses
import itertools
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import scipy.sparse as sp
from strip2d import write_strip

import gfem
from gfem import _band_tracking as tracking
from gfem import bands

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
    path = gfem.WaveVectorPath(k, labels=("Gamma", "Z"), label_index=(0, 2))
    np.testing.assert_array_equal(path.distance, [0.0, 5.0, 17.0])
    assert path.labels == ("Γ", "Z")
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


# --- the band diagram sweep ---------------------------------------------------------

TARGET = 2e9  # below half the highest frequency found on C2's lowest bands
EG_TARGET = 0.5e9  # likewise on the eigen gradient cell, whose lowest band is 1.5-4 GHz


class SolveRanError(AssertionError):
    """A refusal came after a solve."""


@pytest.fixture
def no_solve(monkeypatch):
    """Every per-wave-vector solve fails the test: a refusal must come first."""

    def refuse(*_args, **_kwargs):
        raise SolveRanError("a solve ran")

    monkeypatch.setattr(bands, "_solve_wave_vector", refuse)


@pytest.fixture(scope="module")
def c2_cell(tmp_path_factory):
    return pc.c2(pc.c2_mesh(tmp_path_factory.mktemp("c2_cell"), cells=(4, 4, 1)))


@pytest.fixture(scope="module")
def c2_path(c2_cell):
    return gfem.wave_vector_path(c2_cell, ["Γ", "X"], n_points=3)


def _diagram(model, path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return gfem.band_diagram(
            model, path, **({"target": TARGET, "n_bands": 3} | kwargs)
        )


def _independent(model, k, n=3):
    r"""The solve a band diagram must equal at ``k``, by :math:`\mathrm{Re}\,\omega`."""
    result = gfem.solve(
        model, gfem.Eigenmode(TARGET, n_modes=n, wave_vector=tuple(map(float, k)))
    )
    return result, np.argsort(np.asarray(result.omega).real, kind="stable")


BAND_REFUSALS = {
    "non-model": (
        "non-model",
        {},
        TypeError,
        r"^band_diagram: needs a gfem\.Model, got str\.$",
    ),
    "2D": (
        "2D",
        {},
        NotImplementedError,
        (
            r"^gfem\.band_diagram is not available on a 2D mesh\. .*PEC faces give "
            r"the Ez bands, natural faces the Hz bands"
        ),
    ),
    "no pair": (
        "no pair",
        {},
        ValueError,
        (
            r"^band_diagram: the model has no gfem\.Periodic pair, so it has no "
            r"Bloch wave vector and no bands\. Pair the cell's opposite surfaces "
            r"with gfem\.Periodic\.$"
        ),
    ),
    "path shape": (
        "rect",
        {"path": np.zeros((3, 2))},
        ValueError,
        (
            r"the path must be a gfem\.WaveVectorPath or wave vectors of shape "
            r"\(n_k, 3\) in rad/m with n_k ≥ 1, got shape \(3, 2\)\.$"
        ),
    ),
    "empty path": (
        "rect",
        {"path": np.zeros((0, 3))},
        ValueError,
        r"with n_k ≥ 1, got shape \(0, 3\)\.$",
    ),
    "nan path": (
        "rect",
        {"path": np.array([[np.nan, 0.0, 0.0]])},
        ValueError,
        r"the path must be a gfem\.WaveVectorPath",
    ),
    "off span": (
        "rect",
        {"path": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 5.0]])},
        ValueError,
        (
            r"wave vector 1 \(10, 0, 5\) rad/m has a component of 5 rad/m outside "
            r"the span of the periodic translations .* Give it in that span, e\.g\. "
            r"\(10, 0, 0\) rad/m"
        ),
    ),
    "target zero": (
        "rect",
        {"target": 0.0},
        ValueError,
        r"^band_diagram: target must be a finite frequency > 0 Hz, got 0\.0\.$",
    ),
    "target nan": (
        "rect",
        {"target": math.nan},
        ValueError,
        r"target must be a finite frequency > 0 Hz, got nan",
    ),
    "n_bands zero": (
        "rect",
        {"n_bands": 0},
        ValueError,
        r"^band_diagram: n_bands must be an integer ≥ 1, got 0\.$",
    ),
    "n_bands float": (
        "rect",
        {"n_bands": 2.5},
        ValueError,
        r"n_bands must be an integer ≥ 1, got 2\.5",
    ),
    "keep_fields": (
        "rect",
        {"keep_fields": [0, 5]},
        IndexError,
        (
            r"^band_diagram: keep_fields names wave vector 5, but the path has 2, "
            r"numbered from 0\.$"
        ),
    ),
}


@pytest.mark.usefixtures("no_solve")
@pytest.mark.parametrize("case", BAND_REFUSALS)
def test_band_diagram_refusals_come_before_the_first_solve(models, tmp_path, case):
    which, kwargs, error, message = BAND_REFUSALS[case]
    model = _model(models, tmp_path, which)
    path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    arguments = {"path": path, "target": TARGET, "n_bands": 2} | kwargs
    with pytest.raises(error, match=message):
        gfem.band_diagram(model, **arguments)


def test_the_suggested_wave_vector_prints_no_round_off_or_negative_zero():
    # an oblique pair of translations: the projection leaves -5e-17 on y
    translations = np.array([[0.01, 0.003, 0.0], [0.002, 0.007, 0.0]])
    with pytest.raises(ValueError, match=r"e\.g\. \(10, 0, 0\) rad/m"):
        bands._check_span(np.array([[10.0, 0.0, 5.0]]), translations)
    printable = bands._printable(np.array([10.0, -0.0, -3e-16, 1e-5, -2.5]))
    assert bands._vec(printable) == "(10, 0, 0, 1e-05, -2.5)"
    assert bands._vec(bands._printable(np.array([-0.0, 0.0]))) == "(0, 0)"


@pytest.fixture(scope="module")
def eigen_gradient_cell(tmp_path_factory):
    mesh = pc.eigen_cell_mesh(tmp_path_factory.mktemp("eigen_cell"))
    n = len(mesh.elements("cell"))
    return mesh, jnp.linspace(*pc.EG_EPS, n)


def _ramp_cell(mesh, eps):
    return pc.eigen_cell(mesh, eps, tan_delta=pc.EG_TAN_DELTA)


def _jit_case(name, eigen_gradient_cell):
    mesh, eps = eigen_gradient_cell
    k = np.array([[0.0, 0.0, 0.0], [0.5 * pc.EG_K[0], 0.0, 0.0]])

    def gap(e, path=k, target=EG_TARGET):
        bd = gfem.band_diagram(_ramp_cell(mesh, e), path, target=target, n_bands=2)
        return bd.frequency.sum()

    return {
        "jit": (lambda: jax.jit(gap)(eps), r"not inside jax\.jit or jax\.vmap"),
        "vmap": (
            lambda: jax.vmap(gap)(jnp.stack([eps, eps])),
            r"not inside jax\.jit or jax\.vmap",
        ),
        "jit_grad": (
            lambda: jax.jit(jax.grad(gap))(eps),
            r"not inside jax\.jit or jax\.vmap",
        ),
        "traced_k": (
            lambda: jax.grad(lambda s: gap(eps, path=k * s))(1.0),
            r"the wave vectors and the target must be fixed numbers",
        ),
        "traced_target": (
            lambda: jax.grad(lambda t: gap(eps, target=t))(EG_TARGET),
            r"the wave vectors and the target must be fixed numbers",
        ),
    }[name]


@pytest.mark.usefixtures("no_solve")
@pytest.mark.parametrize(
    "case", ["jit", "vmap", "jit_grad", "traced_k", "traced_target"]
)
def test_band_diagram_inside_jit_or_vmap_raises_before_a_solve(
    eigen_gradient_cell, case
):
    run, message = _jit_case(case, eigen_gradient_cell)
    with pytest.raises(gfem.NonDifferentiableError, match=message):
        run()


# --- band connections from overlaps -------------------------------------------------

#: (overlap, w_prev, w_cur) -> (link, by_overlap): the matrices of the probe
#: that fixed the thresholds (clean continuations 0.90-1.00, doublets 0.707,
#: leakage at most 0.36, a mode with no continuation at most 0.08).
CONNECTIONS = {
    "clean": (
        [[0.97, 0.14], [0.12, 0.95]],
        [1.0, 2.0],
        [1.1, 2.1],
        [0, 1],
        [True, True],
    ),
    "crossing": (
        [[0.05, 0.98], [0.97, 0.04]],
        [1.0, 2.0],
        [1.1, 1.9],
        [1, 0],
        [True, True],
    ),
    "leakage": (
        [[0.95, 0.36], [0.30, 0.90]],
        [1.0, 2.0],
        [1.0, 2.0],
        [0, 1],
        [True, True],
    ),
    "doublet": (
        [[0.707, 0.707], [0.707, 0.707]],
        [1.0, 1.0],
        [2.0, 2.0],
        [0, 1],
        [False, False],
    ),
    "doublet in frequency order": (
        [[0.707, 0.707], [0.707, 0.707]],
        [1.0, 2.0],
        [2.1, 1.9],
        [1, 0],
        [False, False],
    ),
    "split pair, margin met": (
        [[0.73, 0.67], [0.56, 0.78]],
        [1.0, 2.0],
        [1.0, 2.0],
        [0, 1],
        [True, True],
    ),
    "split pair, margin missed": (
        [[0.73, 0.70], [0.56, 0.78]],
        [1.0, 2.0],
        [1.0, 2.0],
        [0, 1],
        [False, False],
    ),
    "one rival under the threshold": (
        [[0.51, 0.49]],
        [1.0],
        [1.0, 2.0],
        [0, -1],
        [False, False],
    ),
    "one rival under the threshold, the other order": (
        [[0.49, 0.51]],
        [1.0],
        [1.0, 2.0],
        [-1, 0],
        [False, False],
    ),
    "two rows wanting one column": (
        [[0.90, 0.10], [0.80, 0.10]],
        [1.0, 2.0],
        [1.0, 2.0],
        [0, -1],
        [True, False],
    ),
    "below the threshold": (
        [[0.40, 0.10], [0.10, 0.45]],
        [1.0, 2.0],
        [1.0, 2.0],
        [-1, -1],
        [False, False],
    ),
    # a lone mode rivals a split pair's span, and the partner of its own
    # pair joins the set: no mode gets two partners
    "a lone mode rivalling a pair": (
        [
            [0.62, 0.778, 0.0, 0.102],
            [0.655, 0.438, 0.941, 0.537],
            [0.655, 0.438, 0.941, 0.537],
            [0.376, 0.4, 0.0, 0.764],
        ],
        [16.455, 18.042, 18.042, 20.418],
        [14.784, 17.456, 18.465, 22.585],
        [0, 1, 2, 3],
        [False, False, False, True],
    ),
    # three members of a group cut by n_bands continue into a pair: its two
    # lowest, in frequency order
    "unequal groups": (
        [[0.853, 0.853], [0.853, 0.853], [0.853, 0.853]],
        [1.0, 1.0, 1.0009],
        [2.0, 2.001],
        [0, 1],
        [False, False],
    ),
}


@pytest.mark.parametrize("case", CONNECTIONS)
def test_band_connections_from_overlaps(case):
    overlap, w_prev, w_cur, link, by_overlap = CONNECTIONS[case]
    got, by = tracking.connect(np.array(overlap), np.array(w_prev), np.array(w_cur))
    assert got.tolist() == link
    assert by.tolist() == by_overlap


def test_the_modes_going_to_zero_at_gamma_are_filled_in_and_connected():
    # three wave vectors, Γ in the middle: the lowest mode of each neighbour
    # goes to zero frequency there, the solve at Γ finds the three above it
    w = np.array([[0.5, 1.0, 2.0], [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]])
    into = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float) + 0.02
    out = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float) + 0.02
    tracked = tracking.track(
        [into, out], w, np.array([True] * 3), np.array([False, True, False])
    )
    assert tracked.static.tolist() == [0, 1, 0]
    assert tracked.connection.tolist() == [[-1, -1, -1], [0, 1, 2], [0, 1, 2]]
    assert tracked.by_overlap.tolist() == [
        [False] * 3,
        [False, True, True],
        [False, True, True],
    ]
    assert not tracked.lost.any()


def test_a_mode_with_no_partner_inside_the_window_is_lost_not_one_at_the_top():
    w = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.5]])
    # band 1 continues nowhere, band 2 continues into the top one
    o = np.array([[0.99, 0.02, 0.0], [0.0, 0.03, 0.01], [0.0, 0.0, 0.97]])
    tracked = tracking.track([o], w, np.array([True, True]), np.array([False] * 2))
    assert tracked.lost.tolist() == [[False, True, False], [False, True, False]]
    assert tracked.connection.tolist() == [[-1, -1, -1], [0, -1, 2]]
    # the same mode left at the top of the window is an edge, not lost
    o = np.array([[0.99, 0.0, 0.0], [0.0, 0.97, 0.0], [0.0, 0.0, 0.03]])
    tracked = tracking.track([o], w, np.array([True, True]), np.array([False] * 2))
    assert not tracked.lost.any()
    # an unanchored window's bottom edge is open too
    o = np.array([[0.03, 0.0, 0.0], [0.0, 0.97, 0.0], [0.0, 0.0, 0.99]])
    tracked = tracking.track([o], w, np.array([False, False]), np.array([False] * 2))
    assert not tracked.lost.any()
    tracked = tracking.track([o], w, np.array([True, True]), np.array([False] * 2))
    assert tracked.lost.tolist() == [[True, False, False], [True, False, False]]


def _pair_basis(angle: float) -> np.ndarray:
    """A pair's two fields in coordinates 0 and 1, rotated by ``angle``, complex."""
    c, s = math.cos(angle), math.sin(angle)
    out = np.zeros((4, 2), dtype=complex)
    out[:2] = [[c, -s * 1j], [s, c * 1j]]
    return out * np.exp(0.7j)


@pytest.mark.parametrize("angle", [0.0, 0.4, math.pi / 4, 1.3])
def test_overlaps_with_a_split_pair_take_its_span_whatever_its_basis(angle):
    mass = sp.diags([1.0, 2.0, 1.0, 3.0]).tocsr()
    root = np.sqrt([1.0, 2.0, 1.0, 3.0])[:, None]
    pair = _pair_basis(angle) / root
    # a lone mode with 0.6 of its norm in the pair's span, split evenly
    # between the members at angle 0 (0.42 each), and a lone mode alone
    lone = np.array([[0.6 / math.sqrt(2)], [0.6 / math.sqrt(2)], [0.8], [0.0]]) / root
    a = np.hstack([lone, np.eye(4)[:, [3]] / root])
    b = np.hstack([pair, np.eye(4)[:, [3]] / root * 2j])
    o = tracking.overlaps(
        mass, a, b, np.array([1.0, 3.0]), np.array([2.0, 2.0002, 3.0])
    )
    np.testing.assert_allclose(o, [[0.6, 0.6, 0.0], [0.0, 0.0, 1.0]], atol=1e-14)
    # pair to pair: one span, in two bases
    other = _pair_basis(0.9) / root
    o = tracking.overlaps(mass, pair, other, np.array([2.0, 2.0]), np.array([2.0, 2.0]))
    np.testing.assert_allclose(o, np.ones((2, 2)), atol=1e-14)


def test_a_gamma_whose_neighbours_disagree_is_not_filled_and_is_lost():
    # one mode goes to zero from the left, none from the right
    w = np.array([[0.5, 1.0, 2.0], [1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])
    into = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float) + 0.02
    out = np.eye(3) * 0.97
    tracked = tracking.track(
        [into, out], w, np.array([True] * 3), np.array([False, True, False])
    )
    assert tracked.static.tolist() == [0, 0, 0]
    assert tracked.lost[1].all()
    assert {k for k, _, _ in tracked.lost_at} == {0, 1}


def test_no_band_continues_a_mode_the_static_bands_pushed_out():
    # Γ first: one band goes to zero there, so its highest mode is band 3,
    # outside the diagram; the mode it continues into connects to none
    w = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]])
    o = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=float) * 0.97 + 0.02
    tracked = tracking.track([o], w, np.array([True] * 2), np.array([True, False]))
    assert tracked.static.tolist() == [1, 0]
    assert tracked.connection.tolist() == [[-1, -1, -1], [0, 1, -1]]
    # Γ's band 2 (its second mode) continues nowhere: lost; band 2 beyond is not
    assert np.argwhere(tracked.lost).tolist() == [[0, 2]]


# --- a small diagram: order, Γ fill, fields and flags -------------------------------


@pytest.fixture(scope="module")
def c2_diagram(c2_cell, c2_path):
    return _diagram(c2_cell, c2_path, keep_fields=[1])


def test_band_diagram_principal_participation_has_its_own_axis(c2_diagram):
    # the principal axes follow a material's ``axes`` order, not x, y, z: they
    # sit on ``axis``, beside the Cartesian ``direction`` of v_g and k
    shape = np.asarray(c2_diagram.omega).shape
    bd = dataclasses.replace(
        c2_diagram, principal_participation={"m": jnp.zeros((*shape, 3))}
    )
    ds = bd.to_xarray()
    assert ds.principal_participation.dims == (
        "wave_vector",
        "band",
        "anisotropic_region",
        "axis",
    )
    assert ds.axis.values.tolist() == [0, 1, 2]
    assert ds.direction.values.tolist() == ["x", "y", "z"]


def test_bands_are_ascending_and_gamma_reports_static_bands(
    c2_cell, c2_path, c2_diagram
):
    bd = c2_diagram
    assert np.asarray(bd.omega).shape == (3, 3)
    assert bd.omega[0, 0] == 0
    assert bool(bd.static[0, 0])
    assert int(bd.static.sum()) == 1
    for name in ("error", "v_g"):
        assert np.isnan(np.asarray(getattr(bd, name))[0, 0]).all()
    assert np.isnan(np.asarray(bd.bulk_participation["cell"])[0, 0])
    assert (np.diff(np.asarray(bd.omega).real, axis=1) >= 0).all()
    for i, k in enumerate(c2_path.wave_vectors):
        result, order = _independent(c2_cell, k)
        found = np.asarray(result.omega)[order]
        s = int(bd.static[i].sum())
        np.testing.assert_allclose(
            np.asarray(bd.omega)[i, s:], found[: 3 - s], rtol=1e-10, atol=0
        )
    assert bd.lost.sum() == 0
    assert bool(bd.anchored.all())


def test_band_diagram_to_xarray(c2_diagram):
    ds = c2_diagram.to_xarray()
    assert ds.frequency.dims == ("wave_vector", "band")
    assert ds.v_g.dims == ("wave_vector", "band", "direction")
    assert ds.v_g.attrs["units"] == "m/s"
    assert ds.omega.attrs["units"] == "rad/s"
    assert ds.anchored.dims == ("wave_vector",)
    assert ds.distance.attrs["units"] == "rad/m"
    assert ds.k.attrs["units"] == "rad/m"
    assert ds.label.values.tolist() == ["Γ", "", "X"]
    assert ds.direction.values.tolist() == ["x", "y", "z"]
    assert ds.k.dims == ("wave_vector", "direction")
    assert ds.bulk_participation.dims == ("wave_vector", "band", "region")
    for name in ("connection", "connection_by_overlap", "static", "lost", "cluster"):
        assert ds[name].dims == ("wave_vector", "band")
    assert ds.attrs["target"] == TARGET
    assert ds.attrs["lattice"] == "rectangular"
    assert not any("field" in name for name in ds.variables)
    np.testing.assert_allclose(
        ds.frequency.values, np.asarray(c2_diagram.omega).real / (2 * np.pi)
    )
    with pytest.raises(TypeError, match="needs concrete values"):
        jax.jit(lambda b: b.to_xarray())(c2_diagram)


def test_kept_fields_and_e_field(c2_cell, c2_path, c2_diagram):
    bd = c2_diagram
    result, order = _independent(c2_cell, c2_path.wave_vectors[1])
    for band in range(3):
        mine = np.asarray(bd.e_field(1, band).coefficients)
        theirs = np.asarray(result.e_fields[int(order[band])].coefficients)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-10 * abs(theirs).max())
    with pytest.raises(
        ValueError,
        match=r"fields were not kept at wave vector 2: "
        r"pass keep_fields=\[2\] \(or True\)",
    ):
        bd.e_field(2, 0)
    with pytest.raises(IndexError, match="Wave vector 3 does not exist"):
        bd.e_field(3, 0)
    with pytest.raises(IndexError, match="Band 3 does not exist"):
        bd.e_field(1, 3)
    both = _diagram(c2_cell, c2_path, keep_fields=[0], n_bands=2)
    with pytest.raises(
        ValueError,
        match=r"band 0 at wave vector 0 \(Γ\) is a static band: it has no field",
    ):
        both.e_field(0, 0)
    assert both.e_field(0, 1).quantity == "E"


def test_band_diagram_is_a_pytree_with_a_repr(c2_diagram):
    leaves, tree = jax.tree_util.tree_flatten(c2_diagram)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(np.asarray(again.omega), np.asarray(c2_diagram.omega))
    assert again.path is c2_diagram.path
    assert again.target == TARGET
    assert repr(c2_diagram) == "BandDiagram(Γ-X, 3 wave vectors, 3 bands, 0-32 GHz)"
    seen = []
    jax.jit(lambda bd: seen.append(repr(bd)) or bd.omega)(c2_diagram)
    assert seen == ["BandDiagram(3 wave vectors, 3 bands)"]


def test_a_reversed_path_gives_the_same_bands(c2_cell, c2_path, c2_diagram):
    reverse = bands.WaveVectorPath(c2_path.wave_vectors[::-1])
    back = _diagram(c2_cell, reverse)
    forward = c2_diagram
    assert bool(back.static[2, 0])
    assert not back.static[:2].any()
    np.testing.assert_allclose(
        np.asarray(back.omega)[::-1], np.asarray(forward.omega), rtol=1e-10, atol=0
    )
    np.testing.assert_array_equal(
        np.asarray(back.cluster)[::-1], np.asarray(forward.cluster)
    )
    np.testing.assert_allclose(
        np.asarray(back.v_g)[::-1],
        np.asarray(forward.v_g),
        rtol=0,
        atol=1e-9 * pc.C0,
    )


def test_k_and_k_plus_g_give_the_same_bands(c2_cell):
    b1 = np.asarray(c2_cell.reciprocal_lattice)[0]
    path = np.array([0.3 * b1, 1.3 * b1, b1])
    bd = _diagram(c2_cell, path)
    omega = np.asarray(bd.omega)
    np.testing.assert_allclose(omega[1], omega[0], rtol=1e-10, atol=0)
    np.testing.assert_allclose(
        np.asarray(bd.v_g)[1], np.asarray(bd.v_g)[0], rtol=0, atol=1e-9 * pc.C0
    )
    assert bd.connection[1].tolist() == [0, 1, 2]
    assert bd.connection_by_overlap[1].all()
    assert bd.static.tolist() == [[False] * 3, [False] * 3, [True, False, False]]
    assert omega[2, 0] == 0


def test_a_high_target_flags_and_warns_once_unanchored(c2_cell):
    path = gfem.wave_vector_path(c2_cell, ["Γ", "X"], n_points=2)
    with pytest.warns(gfem.UnanchoredBandWindowWarning) as record:
        bd = gfem.band_diagram(c2_cell, path, target=20e9, n_bands=2)
    ours = [w for w in record if w.category is gfem.UnanchoredBandWindowWarning]
    assert len(ours) == 1
    assert ours[0].filename == __file__
    message = str(ours[0].message)
    assert message.startswith("band_diagram: at wave vectors 0, 1 (Γ, X) the target")
    assert "20 GHz is not below half the highest band found there" in message
    assert bd.anchored.tolist() == [False, False]
    assert not bd.static.any()
    assert bd.omega[0, 0] != 0


def test_a_band_missing_at_one_wave_vector_is_lost_and_warns_once(c2_cell):
    path = gfem.wave_vector_path(c2_cell, ["Γ", "X"], n_points=5)
    calls = []
    real = bands._solve_wave_vector

    def without_band_0(model, problem, options, comm, cache):
        calls.append(None)
        if len(calls) != 3:
            return real(model, problem, options, comm, cache)
        wider = dataclasses.replace(problem, n_modes=problem.n_modes + 1)
        result = real(model, wider, options, comm, cache)
        keep = np.argsort(np.asarray(result.omega).real)[1:]
        return dataclasses.replace(
            result,
            omega=result.omega[keep],
            error=result.error[keep],
            cluster=result.cluster[keep],
            v_g=result.v_g[keep],
            e_fields=tuple(result.e_fields[int(i)] for i in keep),
            bulk_participation={
                k: v[keep] for k, v in result.bulk_participation.items()
            },
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bands, "_solve_wave_vector", without_band_0)
        with pytest.warns(gfem.LostBandWarning) as record:
            bd = gfem.band_diagram(c2_cell, path, target=TARGET, n_bands=3)
    lost = [w for w in record if w.category is gfem.LostBandWarning]
    assert len(lost) == 1
    message = str(lost[0].message)
    assert "band 0 at wave vector 1" in message
    assert "band 0 at wave vector 3" in message
    assert np.argwhere(np.asarray(bd.lost)).tolist() == [[1, 0], [3, 0]]


@pytest.mark.filterwarnings("error::gfem.UnanchoredBandWindowWarning")
def test_the_band_group_velocity_of_a_band_diagram_raises_under_grad(
    eigen_gradient_cell,
):
    mesh, eps = eigen_gradient_cell
    k = np.array([[0.5 * pc.EG_K[0], 0.0, 0.0]])

    def velocity(e):
        bd = gfem.band_diagram(_ramp_cell(mesh, e), k, target=EG_TARGET, n_bands=1)
        return bd.v_g.real.sum()

    with pytest.raises(gfem.NonDifferentiableError, match="'v_g'"):
        jax.grad(velocity)(eps)


@pytest.mark.filterwarnings("error::gfem.UnanchoredBandWindowWarning")
def test_eager_grad_through_a_band_diagram_is_the_sum_of_per_wave_vector_gradients(
    eigen_gradient_cell,
):
    mesh, eps = eigen_gradient_cell
    model = _ramp_cell(mesh, eps)
    path = gfem.wave_vector_path(model, ["Γ", "X"], n_points=3)
    weights = jnp.array([1.0, 0.37])
    target = EG_TARGET

    def swept(e):
        bd = gfem.band_diagram(_ramp_cell(mesh, e), path, target=target, n_bands=2)
        return jnp.sum(weights * bd.frequency), bd.frequency[0, 0]

    def apart(e, static):
        total = 0.0
        for i, k in enumerate(path.wave_vectors):
            r = gfem.solve(
                _ramp_cell(mesh, e),
                gfem.Eigenmode(target, n_modes=2, wave_vector=tuple(map(float, k))),
            )
            w = jax.lax.stop_gradient(r.omega).real
            order = np.argsort(np.asarray(w))
            s = static[i]
            total += jnp.sum(weights[s:] * (r.omega.real[order[: 2 - s]] / (2 * np.pi)))
        return total

    static = np.asarray(
        gfem.band_diagram(model, path, target=target, n_bands=2).static.sum(axis=1)
    )
    assert static.tolist() == [1, 0, 0]
    together = jax.grad(lambda e: swept(e)[0])(eps)
    separate = jax.grad(lambda e: apart(e, static))(eps)
    np.testing.assert_allclose(
        np.asarray(together),
        np.asarray(separate),
        rtol=1e-9,
        atol=1e-9 * abs(np.asarray(separate)).max(),
    )
    assert float(abs(np.asarray(together)).max()) > 0
    assert np.all(np.asarray(jax.grad(lambda e: swept(e)[1])(eps)) == 0)
