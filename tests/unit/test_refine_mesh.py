"""`gfem.refine_mesh` and `gfem.error_estimate` on electrostatic problems.

Most cases run the square coax of `refinement_cases` at n = 2 (72
tetrahedra), whose corner singularity keeps its field error estimate near
0.1, far above any tolerance these tests ask for.
"""

import sys
import types

import jax
import jax.numpy as jnp
import mpi_cases
import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

import gfem
from gfem.refinement import dorfler

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
COAX_ESTIMATE = 0.1085  # the n = 2 coax's field error estimate, for orientation


@pytest.fixture(scope="module")
def coax(tmp_path_factory):
    return rc.coax_model(tmp_path_factory.mktemp("coax"), 2)


def _refine(model, **kwargs):
    kwargs.setdefault("options", OPTIONS)
    return gfem.refine_mesh(model, rc.COAX_PROBLEM, **kwargs)


def _barycentric(nodes, tets, points):
    """Barycentric coordinates of ``points[i]`` in tetrahedron ``tets[i]``."""
    p = nodes[tets]
    t = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    lam = np.linalg.solve(t, (points - p[:, 0])[..., None])[..., 0]
    return np.column_stack([1 - lam.sum(axis=1), lam])


def _plate(tmp_path, materials):
    """Layers a and b; ``materials`` may be a function of the mesh."""
    layers = (("a", 0.5), ("b", 1.0))
    path = sm.write_msh22(tmp_path / "plate.msh", sm.plate(3.0, 2.0, layers, (2, 2, 2)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    if callable(materials):
        materials = materials(mesh)
    return gfem.Model(mesh, materials=materials)


PLATE = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


# --- the estimate ------------------------------------------------------------


@pytest.mark.parametrize(
    "materials",
    [
        pytest.param(
            {"a": gfem.Material(eps_r=2.0), "b": gfem.Material(eps_r=5.0)},
            id="isotropic",
        ),
        pytest.param(
            {
                "a": gfem.Material(eps_r=(7.0, 3.0, 2.0)),
                "b": gfem.Material(eps_r=(1.5, 9.0, 4.0)),
            },
            id="anisotropic",
        ),
        pytest.param(
            lambda mesh: {
                "a": gfem.MaterialField(eps_r=np.full(len(mesh.elements("a")), 3.0)),
                "b": gfem.Material(eps_r=4.0),
            },
            id="field",
        ),
    ],
)
def test_a_flux_the_mesh_holds_exactly_has_no_error(tmp_path, materials):
    # Layered plates: V is linear in each layer and eps grad V constant and
    # normal to the layers, so the recovered flux is the flux.
    model = _plate(tmp_path, materials)
    result = gfem.solve(model, PLATE, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert estimate.indicators.shape == (len(model.mesh.tetrahedra),)
    assert estimate.values == (estimate.value,)
    assert estimate.value < 1e-6


@pytest.mark.parametrize("eps_r", [4.0, (4.0, 4.0, 4.0)], ids=["scalar", "tensor"])
def test_the_estimate_is_relative_to_the_field_energy(coax, eps_r):
    # a uniform permittivity scales the flux and the energy alike
    model = coax.replace(materials={"dielectric": gfem.Material(eps_r=eps_r)})
    result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert estimate.value == pytest.approx(COAX_ESTIMATE, rel=1e-3)


def test_the_coax_corner_holds_the_largest_indicators(coax):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=coax)
    assert estimate.value == pytest.approx(COAX_ESTIMATE, rel=1e-3)
    assert estimate.value == pytest.approx(np.sqrt((estimate.indicators**2).sum()))
    # the re-entrant corner is the edge x = y = 1 um
    nodes = np.asarray(coax.mesh.nodes)
    corner = np.linalg.norm(nodes[:, :2] - 1e-6, axis=1) < 1e-12
    worst = np.argmax(estimate.indicators)
    assert corner[coax.mesh.tetrahedra[worst]].any()


def test_terminals_are_combined_by_root_mean_square(coax):
    # With no ground, the outer terminal's potential is 1 - the inner one's:
    # the same error, so the mean over two terminals is the one-terminal value.
    single = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    both = gfem.solve(
        coax, gfem.Electrostatic(terminals=["inner", "outer"]), options=OPTIONS
    )
    one = gfem.error_estimate(single, gfem.FieldEstimator(), model=coax)
    two = gfem.error_estimate(both, gfem.FieldEstimator(), model=coax)
    assert two.value == pytest.approx(one.value, rel=1e-9)
    np.testing.assert_allclose(two.indicators, one.indicators, rtol=1e-8, atol=0)


def test_an_estimate_needs_the_model_of_its_result(coax, tmp_path):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    other = rc.coax_model(tmp_path, 4)
    with pytest.raises(ValueError, match="model the result was solved on"):
        gfem.error_estimate(result, gfem.FieldEstimator(), model=other)


# --- Dörfler marking --------------------------------------------------------


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.5, [0]), (0.7, [0, 2]), (1.0, [0, 1, 2])],
)
def test_dorfler_marks_the_fewest_tetrahedra_reaching_the_fraction(fraction, expected):
    # squares 9, 1, 4, 0 of 14: half needs 9; 70 % needs 9 + 4; all needs 14
    marked = dorfler(np.array([3.0, 1.0, 2.0, 0.0]), fraction)
    assert np.flatnonzero(marked).tolist() == expected


def test_a_cycle_marks_the_fewest_tetrahedra_holding_the_error_fraction(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = _refine(coax, max_cycles=1, error_fraction=0.3, tolerance=1e-3)
    first = gfem.error_estimate(
        gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS),
        gfem.FieldEstimator(),
        model=coax,
    )
    squared = np.sort(first.indicators**2)[::-1]
    k = ref.cycles[0].marked
    assert squared[:k].sum() >= 0.3 * squared.sum() > squared[: k - 1].sum()


# --- the loop and its stops ---------------------------------------------------


def test_max_cycles_zero_is_error_estimate(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = _refine(coax, max_cycles=0)
    estimate = gfem.error_estimate(ref.result, gfem.FieldEstimator(), model=coax)
    assert ref.model is coax
    assert ref.stop == "max_cycles"
    assert not ref.converged
    # the same numbers, to the round-off of threaded assembly
    np.testing.assert_allclose(ref.estimate.values, estimate.values, rtol=1e-12)
    np.testing.assert_allclose(
        ref.estimate.indicators, estimate.indicators, rtol=1e-12, atol=0
    )
    (cycle,) = ref.cycles
    assert cycle == gfem.RefinementCycle(
        tetrahedra=72,
        dofs=len(ref.result.potential["inner"].coefficients),
        estimate=ref.estimate.value,
        value=None,
        marked=0,
    )
    np.testing.assert_array_equal(ref.tetrahedron_parents, np.arange(72))


def test_the_loop_stops_at_tolerance_without_a_warning(tmp_path, recwarn):
    model = _plate(tmp_path, {"a": gfem.Material(), "b": gfem.Material(eps_r=3.0)})
    ref = gfem.refine_mesh(model, PLATE, options=OPTIONS)
    assert ref.stop == "tolerance"
    assert ref.converged
    assert len(ref.cycles) == 1
    assert ref.cycles[0].marked == 0
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_max_cycles_refines_that_many_times_and_warns(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning, match="max_cycles=2"):
        ref = _refine(coax, max_cycles=2, tolerance=1e-3)
    assert ref.stop == "max_cycles"
    assert not ref.converged
    assert len(ref.cycles) == 3
    tets = [c.tetrahedra for c in ref.cycles]
    assert tets[0] == 72 < tets[1] < tets[2] == len(ref.model.mesh.tetrahedra)
    assert [c.marked > 0 for c in ref.cycles] == [True, True, False]
    estimates = [c.estimate for c in ref.cycles]
    assert estimates[0] > estimates[1] > estimates[2] == ref.estimate.value
    assert ref.cycles[-1].dofs == len(ref.result.potential["inner"].coefficients)


def test_max_tetrahedra_discards_the_mesh_over_the_cap(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning, match="max_tetrahedra"):
        ref = _refine(coax, max_tetrahedra=80)
    assert ref.stop == "max_tetrahedra"
    assert ref.model is coax
    assert len(ref.cycles) == 1
    assert ref.cycles[0].marked > 0
    assert ref.estimate.value == pytest.approx(COAX_ESTIMATE, rel=1e-3)


class _Failing:
    """`gfem.solve` that raises ``error`` on its ``fail_on``-th call."""

    def __init__(self, error, fail_on):
        self.error, self.fail_on, self.calls = error, fail_on, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise self.error
        return gfem.solve(*args, **kwargs)


@pytest.mark.parametrize(
    "error",
    [
        gfem.LinearSolverConvergenceError("GMRES stopped at 1e-3"),
        gfem.EigenmodeConvergenceError("mode 0 did not converge"),
    ],
)
def test_a_failed_solve_ends_the_loop_on_the_last_solved_mesh(coax, monkeypatch, error):
    monkeypatch.setattr("gfem.refinement.solve", _Failing(error, fail_on=3))
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning, match=str(error)):
        ref = _refine(coax, tolerance=1e-3)
    assert ref.stop == "solve failed"
    assert len(ref.cycles) == 2
    assert len(ref.model.mesh.tetrahedra) == ref.cycles[1].tetrahedra
    assert len(ref.tetrahedron_parents) == ref.cycles[1].tetrahedra
    assert ref.estimate.value == ref.cycles[1].estimate


def test_any_other_solve_error_propagates(coax, monkeypatch):
    monkeypatch.setattr("gfem.refinement.solve", _Failing(MemoryError(), fail_on=2))
    with pytest.raises(MemoryError):
        _refine(coax, tolerance=1e-3)


def test_a_failed_first_solve_raises(coax, monkeypatch):
    error = gfem.LinearSolverConvergenceError("GMRES stopped")
    monkeypatch.setattr("gfem.refinement.solve", _Failing(error, fail_on=1))
    with pytest.raises(gfem.LinearSolverConvergenceError):
        _refine(coax)


# --- parents and transfer ----------------------------------------------------


@pytest.fixture(scope="module")
def refined(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        return _refine(coax, max_cycles=3, tolerance=1e-3)


def test_composed_parents_contain_their_children(coax, refined):
    fine = refined.model.mesh
    coarse_nodes = np.asarray(coax.mesh.nodes)
    centroids = np.asarray(fine.nodes)[fine.tetrahedra].mean(axis=1)
    parents = np.asarray(coax.mesh.tetrahedra)[refined.tetrahedron_parents]
    assert (_barycentric(coarse_nodes, parents, centroids) > -1e-12).all()
    # a triangle's centroid lies in its parent triangle's plane and inside it
    tri_centroids = np.asarray(fine.nodes)[fine.triangles].mean(axis=1)
    p = coarse_nodes[np.asarray(coax.mesh.triangles)[refined.triangle_parents]]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    offset = np.einsum("ij,ij->i", tri_centroids - p[:, 0], normal)
    np.testing.assert_allclose(offset, 0, atol=1e-24)
    assert len(refined.cycles) == 4


@pytest.mark.parametrize("region", ["dielectric", "inner"])
def test_transfer_gathers_a_region_from_the_input_mesh(coax, refined, region):
    old = coax.mesh.elements(region)
    values = np.arange(len(old), dtype=float)
    out = refined.transfer(values, region)
    parents = (
        refined.tetrahedron_parents
        if coax.mesh.region(region).dim == 3
        else refined.triangle_parents
    )
    new = refined.model.mesh.elements(region)
    assert out.shape == (len(new),)
    np.testing.assert_array_equal(old[out.astype(int)], parents[new])
    pairs = refined.transfer(np.column_stack([values, -values]), region)
    np.testing.assert_array_equal(pairs[:, 1], -out)


def test_transfer_is_differentiable(coax, refined):
    n = len(coax.mesh.elements("dielectric"))
    weights = jnp.arange(len(refined.model.mesh.elements("dielectric")), dtype=float)
    grad = jax.grad(lambda v: (refined.transfer(v, "dielectric") * weights).sum())(
        jnp.ones(n)
    )
    assert float(grad.sum()) == pytest.approx(float(weights.sum()))


# --- refusals, before any solve ----------------------------------------------


@pytest.fixture
def no_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement.solve", solve)


@pytest.mark.parametrize(
    ("key", "feature"),
    [
        ("adaptive_sweep", "adaptive sweep"),
        ("periodic", "periodic"),
        ("bloch", "bloch"),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_unsupported_problems_raise_before_any_solve(tmp_path, key, feature):
    model, problem, options = mpi_cases.refused(key, tmp_path)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(model, problem, options=options)
    assert caught.value.feature == feature
    assert isinstance(caught.value, NotImplementedError)


@pytest.mark.usefixtures("no_solve")
def test_a_port_mode_problem_raises_before_any_solve(tmp_path):
    model, _, options = mpi_cases.refused("adaptive_sweep", tmp_path)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(model, gfem.PortMode("p1", 5e9), options=options)
    assert caught.value.feature == "port mode"


@pytest.mark.usefixtures("no_solve")
def test_the_adaptive_sweep_refusal_gives_the_remedy(tmp_path):
    model, problem, options = mpi_cases.refused("adaptive_sweep", tmp_path)
    with pytest.raises(gfem.MeshRefinementUnsupportedError, match="uniform sweep"):
        gfem.refine_mesh(model, problem, options=options)


@pytest.mark.usefixtures("no_solve")
def test_more_than_one_rank_raises_before_any_solve(coax, monkeypatch):
    class Stub:
        rank = 0

        def __init__(self, size):
            self.size = size

    mpi = types.SimpleNamespace(
        COMM_WORLD=Stub(4),
        COMM_SELF=Stub(1),
        get_vendor=lambda: ("Open MPI", (0, 0, 0)),
        Is_initialized=lambda: True,
        Is_finalized=lambda: False,
    )
    module = types.ModuleType("mpi4py")
    module.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", module)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        _refine(coax, comm=Stub(4))
    assert caught.value.feature == "mesh refinement"
    assert "COMM_SELF" in str(caught.value)


@pytest.mark.usefixtures("no_solve")
def test_a_tolerance_below_ten_solve_tolerances_raises(coax):
    with pytest.raises(ValueError, match=r"at least 10 x SolverOptions\.tolerance"):
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, tolerance=5e-6)
    gfem_options = gfem.SolverOptions(tolerance=1e-9)
    eigen = gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=1)
    with pytest.raises(ValueError, match="eigen_tolerance"):
        gfem.refine_mesh(coax, eigen, tolerance=5e-10, options=gfem_options)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tolerance": 0.0}, ValueError),
        ({"tolerance": "1e-2"}, TypeError),
        ({"error_fraction": 0.0}, ValueError),
        ({"error_fraction": 1.5}, ValueError),
        ({"max_cycles": -1}, ValueError),
        ({"max_cycles": 2.0}, TypeError),
        ({"max_tetrahedra": 0}, ValueError),
        ({"estimator": "field"}, TypeError),
        ({"options": {"order": 2}}, TypeError),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_bad_parameters_raise_before_any_solve(coax, kwargs, error):
    with pytest.raises(error):
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, **kwargs)


@pytest.mark.usefixtures("no_solve")
def test_a_quantity_estimator_is_not_available_yet(coax):
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, estimator=estimator)
    assert caught.value.feature == "quantity estimator"


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"capacitance": ("a", "b"), "omega": 0}, {"s": "p2"}, {"omega": True}],
)
def test_a_quantity_estimator_names_exactly_one_quantity(kwargs):
    with pytest.raises(TypeError):
        gfem.QuantityEstimator(**kwargs)


def test_a_quantity_estimator_holds_its_key():
    assert gfem.QuantityEstimator(s=["p2", "p1"]).s == ("p2", "p1")
    assert gfem.QuantityEstimator(omega=np.int64(1)).omega == 1
    with pytest.raises(ValueError):
        gfem.QuantityEstimator(omega=-1)


# --- JAX ---------------------------------------------------------------------


def test_a_traced_design_raises_nondifferentiable(coax):
    n = len(coax.mesh.elements("dielectric"))

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        model = coax.replace(materials={"dielectric": field})
        return _refine(model, max_cycles=0).estimate.value

    with pytest.raises(gfem.NonDifferentiableError, match="refine_mesh"):
        jax.grad(objective)(jnp.ones(n))


def test_refine_mesh_and_error_estimate_raise_inside_jit(coax):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    calls = {
        "refine_mesh": lambda: _refine(coax, max_cycles=0),
        "error_estimate": lambda: gfem.error_estimate(
            result, gfem.FieldEstimator(), model=coax
        ),
    }
    for name, call in calls.items():
        with pytest.raises(gfem.NonDifferentiableError, match=name):
            jax.jit(lambda x, call=call: (call(), x)[1])(1.0)


def test_refine_mesh_runs_under_grad_with_concrete_inputs(coax):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        plain = _refine(coax, max_cycles=1, tolerance=1e-3).estimate.value

    def objective(x):
        return x * _refine(coax, max_cycles=1, tolerance=1e-3).estimate.value

    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        assert float(jax.grad(objective)(2.0)) == pytest.approx(plain)


def test_the_public_names_are_exported():
    for name in (
        "refine_mesh",
        "error_estimate",
        "FieldEstimator",
        "QuantityEstimator",
        "MeshRefinement",
        "RefinementCycle",
        "ErrorEstimate",
        "MeshRefinementNotConvergedWarning",
        "MeshRefinementUnsupportedError",
    ):
        assert name in gfem.__all__
        assert hasattr(gfem, name)
