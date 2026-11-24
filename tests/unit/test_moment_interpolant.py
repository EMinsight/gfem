"""The moment interpolant of an incident field given as values at the moment points."""

import time
from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest

import gfem
from gfem._engine.cache import discretization
from gfem._engine.domain import hcurl_space
from gfem._engine.moments import incident_values, moment_interpolant

DATA = Path(__file__).parent / "data"
#: The two-box mesh is in micrometres: a 2 x 2 x 2 box, substrate below air.
UM = 1e-6
K = (
    2
    * np.pi
    / (1.7 * UM)
    * np.array([0.3, -0.5, 0.81])
    / np.linalg.norm([0.3, -0.5, 0.81])
)
POLARIZATION = np.array([1.0, 0.7j, 0.0])
POLARIZATION[2] = -(K[0] * POLARIZATION[0] + K[1] * POLARIZATION[1]) / K[2]


def _model():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    return gfem.Model(
        mesh, materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()}
    )


def _plane_wave(points):
    """A plane wave in V/m at ``(n, 3)`` points in metres."""
    return np.exp(-1j * points @ K)[:, None] * POLARIZATION[None, :]


def _dual_set(model, order, field_cf):
    """NGSolve's moment interpolant of a closed form given in file units."""
    built = discretization(model.mesh).built
    space = hcurl_space(model, built, order)
    gf = ngs.GridFunction(space)
    gf.Set(field_cf, dual=True)
    return gf.vec.FV().NumPy().copy(), space


@pytest.mark.parametrize("order", [1, 2, 3])
def test_equals_ngsolve_dual_set_of_the_closed_form(order):
    model = _model()
    interpolant = moment_interpolant(model, order)
    w = interpolant(_plane_wave(interpolant.points))
    # the engine value of an H(curl) field is its SI value times the scale
    x, y, z = (UM * c for c in (ngs.x, ngs.y, ngs.z))
    phase = ngs.exp(-1j * (K[0] * x + K[1] * y + K[2] * z))
    cf = ngs.CF(tuple(UM * complex(e) * phase for e in POLARIZATION))
    reference, _ = _dual_set(model, order, cf)
    assert np.linalg.norm(w - reference) <= 1e-12 * np.linalg.norm(reference)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_wall_the_field_satisfies_keeps_a_zero_tangential_trace(order):
    # n x E = 0 on the box's bottom face z = 0: the field of a PEC image pair
    def standing(points):
        kx, ky, kz = K
        z = points[:, 2]
        lateral = np.exp(-1j * (kx * points[:, 0] + ky * points[:, 1]))
        return lateral[:, None] * np.stack(
            [
                np.sin(kz * z) * POLARIZATION[0],
                np.sin(kz * z) * POLARIZATION[1],
                np.cos(kz * z) * POLARIZATION[2],
            ],
            axis=1,
        )

    model = _model()
    interpolant = moment_interpolant(model, order)
    built = discretization(model.mesh).built
    gf = ngs.GridFunction(hcurl_space(model, built, order))
    gf.vec.FV().NumPy()[:] = interpolant(standing(interpolant.points))
    grid = np.linspace(0.05, 1.95, 15)
    xs, ys = (a.ravel() for a in np.meshgrid(grid, grid))
    on_wall = np.asarray(gf(built.mesh(xs, ys, np.zeros_like(xs))))
    inside = np.asarray(gf(built.mesh(xs, ys, np.full_like(xs, 0.3))))
    assert np.abs(on_wall[:, :2]).max() <= 1e-12 * np.abs(inside).max()


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (lambda p: np.zeros((len(p), 2)), "shape"),
        (lambda p: np.zeros(3 * len(p)), "shape"),
        (lambda p: np.full((len(p), 3), "a"), "dtype"),
        (lambda p: np.where(p[:, :1] > 0.5e-6, np.nan, 1.0) * np.ones(3), "non-finite"),
        (lambda p: np.full((len(p), 3), np.inf * 1j), "non-finite"),
    ],
)
def test_an_incident_field_function_breaking_its_contract_is_refused(returned, message):
    points = np.random.default_rng(0).uniform(0, 1e-6, (10, 3))
    with pytest.raises(ValueError, match=rf"IncidentField 'beam'.*{message}"):
        incident_values(lambda p, _: returned(p), points, 1e14, "IncidentField 'beam'")


def test_an_incident_field_function_gets_the_points_and_frequency():
    points = np.random.default_rng(0).uniform(0, 1e-6, (10, 3))
    seen = []

    def function(p, f):
        seen.append((p, f))
        return p.astype(int)  # integer V/m is accepted

    values = incident_values(function, points, 2e14, "IncidentField 'beam'")
    assert seen[0][0] is points
    assert seen[0][1] == 2e14
    assert values.dtype == complex
    np.testing.assert_array_equal(values, points.astype(int))


def test_an_element_set_takes_the_coefficients_of_its_own_entities():
    model = _model()
    sub = np.zeros(len(model.mesh.tetrahedra), bool)
    sub[model.mesh.elements("sub")] = True
    whole = moment_interpolant(model, 2)
    part = moment_interpolant(model, 2, elements=sub)
    w_whole = whole(_plane_wave(whole.points))
    w_part = part(_plane_wave(part.points))
    held = np.flatnonzero(np.diff(part.matrix.indptr))
    assert 0 < len(held) < len(w_whole)
    np.testing.assert_allclose(
        w_part[held], w_whole[held], rtol=0, atol=1e-13 * np.abs(w_whole).max()
    )
    assert not np.diff(part.matrix.indptr)[
        np.setdiff1d(np.arange(len(w_whole)), held)
    ].any()
    top = model.mesh.nodes[model.mesh.tetrahedra[sub]][..., 2].max()  # metres
    assert part.points[:, 2].max() <= top * (1 + 1e-12)


def test_is_built_once_per_discretization_order_and_element_set():
    model = _model()
    sub = np.zeros(len(model.mesh.tetrahedra), bool)
    sub[model.mesh.elements("sub")] = True
    first = moment_interpolant(model, 2)
    assert moment_interpolant(model, 2) is first
    assert moment_interpolant(model, 3) is not first
    assert moment_interpolant(model, 2, elements=sub) is moment_interpolant(
        model, 2, elements=sub.copy()
    )
    assert moment_interpolant(model, 2, elements=sub) is not first


def test_reports_setup_and_right_hand_side_time():
    """Observations only: setup per discretization, cost per right-hand side."""
    model = _model()
    for order in (1, 2, 3):
        interpolant = moment_interpolant(model, order)
        start = time.perf_counter()
        interpolant(_plane_wave(interpolant.points))
        per_rhs = time.perf_counter() - start
        n = interpolant.matrix.shape[0]
        print(  # noqa: T201
            f"moment interpolant p={order}: {n} DOFs, "
            f"{len(interpolant.points)} points, nnz/DOF "
            f"{interpolant.matrix.nnz / n:.1f}, setup {interpolant.setup_time:.3f} s, "
            f"per right-hand side {per_rhs * 1e3:.2f} ms"
        )


def test_an_empty_element_set_has_no_moment_points():
    model = _model()
    none = np.zeros(len(model.mesh.tetrahedra), bool)
    interpolant = moment_interpolant(model, 3, elements=none)
    assert interpolant.points.shape == (0, 3)
    assert not interpolant(interpolant.points.astype(complex)).any()
