r"""Band diagrams against closed forms and published values.

- The empty square-lattice plate (`periodic_cases.c2_square_mesh`): its bands
  along Γ-X are the empty lattice, cross where the lattice says, fill the
  static band at Γ, share one cluster at X, and have the group velocities
  $(c/n)(\mathbf k + \mathbf G)/|\mathbf k + \mathbf G|$ where a band is alone
  and none where bands touch with different slopes.
- The square lattice of alumina rods of Joannopoulos et al. (chapter 5) on
  the tutorial mesh: the Ez and Hz bands at Γ, X, M, the Ez gap of 31.4 % of
  its midgap, the group velocities at the segment midpoints, the symmetry of
  the zone-edge points, and the sweep against independent solves.
- A lossy crystal sweep that must find every band (a lossy refinement used to
  drop the nearest mode).
"""

import math
import warnings

import numpy as np
import periodic_cases as pc
import pytest

import gfem


@pytest.fixture(autouse=True)
def _warnings_are_errors():
    """A diagram that warns is a failure here, whatever the pytest config."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --- the empty square lattice -------------------------------------------------------

SQUARE_TARGET = 2e9
#: Seven bands reach $|\mathbf k + \mathbf G| = 1.6\cdot 2\pi/a$, where the band
#: error grows as $(|\mathbf k + \mathbf G|\,h)^{2p}$: 8x8x1 at order 2 misses
#: `C2_BAND_RTOL` (7.9e-4 on band 6 next to Γ); 6x6x1 at order 3 is within
#: 5.5e-5 on every band, at 1.5 times its dofs.
SQUARE_CELLS = (6, 6, 1)
SQUARE_ORDER = 3


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    mesh = pc.c2_square_mesh(tmp_path_factory.mktemp("square"), cells=SQUARE_CELLS)
    model = pc.c2(mesh)
    path = gfem.wave_vector_path(model, ["Γ", "X"], n_points=6)
    options = gfem.SolverOptions(order=SQUARE_ORDER)
    diagram = gfem.band_diagram(
        model, path, target=SQUARE_TARGET, n_bands=7, options=options
    )
    return path, diagram


def _lattice(k, n_modes):
    """The ``n_modes`` lowest (frequency, group velocity / c) of the empty lattice.

    A group velocity is ``None`` where the lattice's bands of that frequency
    touch with different slopes (or a group ``n_modes`` cuts): NaN.
    """
    a = pc.C2_AX * pc.MM
    rows = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            q = np.asarray(k) + 2 * math.pi * np.array([i / a, j / a, 0.0])
            norm = np.linalg.norm(q)
            f = pc.C0 * norm / (2 * math.pi * math.sqrt(pc.C2_EPS))
            rows.append((f, q / norm / math.sqrt(pc.C2_EPS)))
    rows.sort(key=lambda r: r[0])
    out = []
    for f, v in rows[: n_modes + 4]:
        same = [w for g, w in rows if abs(g - f) <= 1e-3 * f]
        alone = all(np.allclose(w, same[0], atol=1e-9) for w in same)
        cut = sum(1 for g, _ in rows[:n_modes] if abs(g - f) <= 1e-3 * f)
        out.append((f, v if alone and cut == len(same) else None))
    return out[:n_modes]


def test_empty_square_lattice_bands_match_the_empty_lattice(square):
    path, bd = square
    expected = [
        pc.empty_lattice(k, (pc.C2_AX * pc.MM, pc.C2_AX * pc.MM, None), pc.C2_EPS, 7)
        for k in path.wave_vectors
    ]
    np.testing.assert_allclose(
        np.asarray(bd.frequency), np.stack(expected), rtol=pc.C2_BAND_RTOL, atol=0
    )


def test_empty_square_lattice_bands_cross_and_fill_gamma(square):
    _, bd = square
    assert bd.static.sum(axis=1).tolist() == [1, 0, 0, 0, 0, 0]
    assert np.isnan(np.asarray(bd.v_g)[0, 0]).all()
    assert bd.anchored.all()
    assert not bd.lost.any()
    # the crossing between κ = 0.2 and 0.3: band 6 there is band 4 before it
    assert int(bd.connection[3, 6]) == 4
    assert bool(bd.connection_by_overlap[3, 6])
    # the X doublet is one cluster, connected by frequency order
    cluster = np.asarray(bd.cluster)[5]
    assert cluster[0] == cluster[1]
    assert not bd.connection_by_overlap[5, :2].any()
    assert bd.connection[5, :2].tolist() == [0, 1]


def test_empty_square_lattice_group_velocities_are_the_lattice_slopes(square):
    path, bd = square
    v = np.asarray(bd.v_g) / pc.C0
    for i in range(1, 6):
        for band, (_, want) in enumerate(_lattice(path.wave_vectors[i], 7)):
            if want is None:
                assert np.isnan(v[i, band]).all(), (i, band)
            else:
                np.testing.assert_allclose(
                    v[i, band].real,
                    want,
                    rtol=0,
                    atol=pc.C2_VG_TOL,
                    err_msg=str((i, band)),
                )


# --- the alumina rod crystal --------------------------------------------------------

#: The frequency of a normalized $\nu$ band value and back.
POLARIZATIONS = ("Ez", "Hz")
RODS_TARGET = pc.rods_frequency(0.1)
RODS_BAND_RTOL = 1e-3
RODS_GAP_POINT = 0.1  # percent of the midgap
RODS_VG_TOL = {"Ez": 5e-4, "Hz": 2e-3}  # of c, at the segment midpoints
SEGMENT_DIRECTION = {
    "Γ-X": np.array([1.0, 0.0, 0.0]),
    "X-M": np.array([0.0, 1.0, 0.0]),
    "M-Γ": -np.array([1.0, 1.0, 0.0]) / math.sqrt(2),
}


@pytest.fixture(scope="module")
def rod_models():
    mesh = pc.rods_tutorial_mesh()
    return {
        "Ez": pc.rods_slab(mesh, ez=True),
        "Hz": pc.rods_slab(mesh, ez=False),
    }


@pytest.fixture(scope="module")
def rod_diagrams(rod_models):
    options = gfem.SolverOptions(order=2)
    out = {}
    for name, model in rod_models.items():
        path = gfem.wave_vector_path(model, ["Γ", "X", "M", "Γ"], n_points=7)
        out[name] = (
            path,
            gfem.band_diagram(
                model, path, target=RODS_TARGET, n_bands=4, options=options
            ),
        )
    return out


def _normalized(bd):
    return np.asarray(bd.frequency) * pc.ROD_A * pc.MM / pc.C0


@pytest.mark.parametrize("pol", POLARIZATIONS)
def test_rods_crystal_bands_gap_and_group_velocity(rod_diagrams, pol):
    path, bd = rod_diagrams[pol]
    nu = _normalized(bd)
    points = {"Γ": 0, "X": 2, "M": 4}
    for name, i in points.items():
        np.testing.assert_allclose(
            nu[i], pc.ROD_BANDS[pol][name], rtol=RODS_BAND_RTOL, atol=0
        )
    assert path.label_index == (0, 2, 4, 6)
    for i in (0, 6):  # Γ at both ends: band 0 is the static one
        assert bool(bd.static[i, 0])
        assert int(bd.static[i].sum()) == 1
    if pol == "Ez":
        gap = nu[:, 1].min() - nu[:, 0].max()
        middle = 0.5 * (nu[:, 1].min() + nu[:, 0].max())
        assert abs(100 * gap / middle - 31.4) <= RODS_GAP_POINT
    v = np.asarray(bd.v_g) / pc.C0
    for segment, midpoint in zip(("Γ-X", "X-M", "M-Γ"), (1, 3, 5), strict=True):
        want = pc.ROD_V_G[pol].get(segment)
        if want is None:
            continue
        along = v[midpoint].real @ SEGMENT_DIRECTION[segment]
        np.testing.assert_allclose(along, want, rtol=0, atol=RODS_VG_TOL[pol])
    assert bd.anchored.all()
    assert not bd.lost.any()


@pytest.mark.parametrize("pol", POLARIZATIONS)
def test_rods_crystal_zone_edge_symmetry_and_doublets(rod_diagrams, pol):
    _, bd = rod_diagrams[pol]
    v = np.asarray(bd.v_g) / pc.C0
    nu = _normalized(bd)
    # a singlet at a corner of the zone has no group velocity; a doublet
    # shares one value (a cluster the mesh keeps together)
    singlets = {2: range(4), 4: (0, 3)}
    for i, bands in singlets.items():
        for band in bands:
            assert np.linalg.norm(v[i, band]) <= 1e-8, (i, band)
    for i, band in ((0, 2), (0, 3), (4, 1), (4, 2)):
        assert not np.isnan(v[i, band]).any(), (i, band)
        assert np.linalg.norm(v[i, band]) <= 1e-4, (i, band)
    assert nu[0, 2] == pytest.approx(nu[0, 3], rel=1e-3)


@pytest.fixture(scope="module")
def independent_rods(rod_models, rod_diagrams):
    """Each wave vector of the Ez diagram solved alone on a fresh discretization."""
    path, _ = rod_diagrams["Ez"]
    options = gfem.SolverOptions(order=2)
    out = []
    for k in path.wave_vectors:
        gfem.release_memory()
        problem = gfem.Eigenmode(RODS_TARGET, n_modes=4, wave_vector=tuple(k))
        result = gfem.solve(rod_models["Ez"], problem, options=options)
        order = np.argsort(np.asarray(result.omega).real, kind="stable")
        out.append((result, order))
    return out


def test_a_band_diagram_equals_independent_solves(rod_diagrams, independent_rods):
    _, bd = rod_diagrams["Ez"]
    for i, (result, order) in enumerate(independent_rods):
        s = int(bd.static[i].sum())
        n = 4 - s
        assert np.asarray(result.omega).shape == (4,)
        np.testing.assert_allclose(
            np.asarray(bd.omega)[i, s:],
            np.asarray(result.omega)[order][:n],
            rtol=1e-10,
            atol=0,
        )
        mine = np.asarray(bd.cluster)[i, s:]
        theirs = np.asarray(result.cluster)[order][:n]
        same = mine[:, None] == mine[None, :]
        assert (same == (theirs[:, None] == theirs[None, :])).all()
        np.testing.assert_allclose(
            np.asarray(bd.v_g)[i, s:],
            np.asarray(result.v_g)[order][:n],
            rtol=0,
            atol=1e-9 * pc.C0,
        )


# --- a lossy crystal ---------------------------------------------------------------


def test_a_lossy_crystal_sweep_finds_every_band(tmp_path):
    mesh = pc.rods_mesh(tmp_path, 8)
    target = pc.rods_frequency(0.02)
    a = math.pi / (pc.ROD_A * pc.MM)
    path = np.array([[a * t, 0.0, 0.0] for t in (0.0, 0.25, 0.5, 0.75, 1.0)])
    lossy = gfem.band_diagram(
        pc.rods(mesh, tan_delta=0.01), path, target=target, n_bands=3
    )
    lossless = gfem.band_diagram(pc.rods(mesh), path, target=target, n_bands=3)
    assert not lossy.lost.any()
    omega, twin = np.asarray(lossy.omega), np.asarray(lossless.omega)
    static = np.asarray(lossy.static)
    np.testing.assert_allclose(omega.real, twin.real, rtol=1e-3, atol=0)
    assert (omega.imag[~static] > 0).all()
    assert np.isfinite(np.asarray(lossy.v_g)[2]).all()
