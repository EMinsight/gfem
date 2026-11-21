"""`gfem.QuantityEstimator`: key checks, refusals, mode tracking, sheets.

The effectivity of the estimate is verified in
`tests/verification/test_quantity_estimate.py`; these cases check what is
asked of it before any solve, and how it reads a result.
"""

import dataclasses
import itertools
import logging
import math
import types

import contrast_cases as cc
import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._engine import estimate as kernels
from gfem.refinement import _dorfler

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture
def no_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement._solve", solve)


@pytest.fixture(scope="module")
def coax(tmp_path_factory):
    return rc.coax_model(tmp_path_factory.mktemp("coax"), 2)


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    return rc.step_model(tmp_path_factory.mktemp("step"), 2)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    return rc.cavity_model(tmp_path_factory.mktemp("cavity"), 2)


def _with(model, **changes):
    """``model`` rebuilt with other materials, boundaries or ports."""
    kwargs = {
        "materials": model.materials,
        "boundaries": model.boundaries,
        "ports": model.ports,
    }
    return gfem.Model(model.mesh, **{**kwargs, **changes})


def _waves(*, offset=0.0):
    return [
        gfem.WavePort("p1", "port1", offset=offset),
        gfem.WavePort("p2", "port2"),
    ]


# --- key checks, before any solve --------------------------------------------


@pytest.mark.usefixtures("no_solve")
def test_an_unknown_terminal_or_port_raises_key_error(coax, step):
    with pytest.raises(KeyError, match="'outer'"):
        gfem.refine_mesh(
            coax,
            rc.COAX_PROBLEM,
            estimator=gfem.QuantityEstimator(capacitance=("inner", "outer")),
        )
    with pytest.raises(KeyError, match="'p3'"):
        gfem.refine_mesh(
            step,
            rc.step_problem(),
            estimator=gfem.QuantityEstimator(s=("p3", "p1")),
        )


@pytest.mark.usefixtures("no_solve")
def test_an_unexcited_source_port_raises_with_the_remedy(step):
    # The entry's column is the source port's field: it must be excited.
    problem = gfem.Driven([3e9], excite="p1")
    with pytest.raises(ValueError, match=r"Driven\.excite"):
        gfem.refine_mesh(
            step, problem, estimator=gfem.QuantityEstimator(s=("p1", "p2"))
        )


def _reactive_p2(step):
    """``step`` with port p2 an inductance alone: its row of S is undefined."""
    return _with(
        step,
        ports=[step.ports[0], gfem.LumpedPort("p2", "port2", "+z", inductance=1e-9)],
    )


@pytest.mark.usefixtures("no_solve")
def test_an_undefined_read_row_raises_before_any_solve(step):
    # p2 has no resistance: S[p2, p1] is NaN, so it has nothing to estimate
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            _reactive_p2(step),
            gfem.Driven([3e9], excite="p1"),
            estimator=gfem.QuantityEstimator(s=("p2", "p1")),
        )
    assert caught.value.feature == "undefined S row"
    assert "resistance" in str(caught.value)


def test_error_estimate_of_an_undefined_read_row_raises(step):
    model = _reactive_p2(step)
    problem = gfem.Driven([3e9], excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, gfem.QuantityEstimator(s=("p2", "p1")), model=model)
    assert caught.value.feature == "undefined S row"


@pytest.mark.usefixtures("no_solve")
def test_a_mode_index_beyond_n_modes_raises(cavity):
    with pytest.raises(ValueError, match="n_modes=1"):
        gfem.refine_mesh(
            cavity, rc.CAVITY_PROBLEM, estimator=gfem.QuantityEstimator(omega=1)
        )


@pytest.mark.parametrize(
    "estimator",
    [
        gfem.QuantityEstimator(s=("inner", "inner")),
        gfem.QuantityEstimator(omega=0),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_a_keyword_of_another_problem_raises_type_error(coax, estimator):
    with pytest.raises(TypeError, match="Electrostatic"):
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, estimator=estimator)


def test_error_estimate_checks_the_key_against_the_result(coax):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    with pytest.raises(KeyError):
        gfem.error_estimate(
            result, gfem.QuantityEstimator(capacitance=("x", "x")), model=coax
        )
    with pytest.raises(TypeError):
        gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=coax)


# --- lossy wave ports: the S entry's adjoint ---------------------------------


@pytest.fixture
def prepares(monkeypatch):
    """The operators the linear solver prepares (factorizes, for "direct")."""
    from gfem import _linalg

    prepared = []
    original = _linalg.prepare

    def counting(*args, **kwargs):
        prepared.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_linalg, "prepare", counting)
    return prepared


@pytest.fixture
def factorizations(caplog):
    """The solve steps that factorize (or refactorize) a matrix, so far."""
    gfem.release_memory()
    caplog.set_level(logging.INFO, logger="gfem.solver")

    def count():
        return sum(
            r.name == "gfem.solver"
            and getattr(r, "action", None) in {"factorized", "refactorized"}
            for r in caplog.records
        )

    yield count
    gfem.release_memory()


def _forced_adjoint(model, result, port, frequency=None):
    """Port ``port``'s S-functional adjoint, solved whether parallel or not.

    Returns ``(z, c x)``: the adjoint field at ``frequency`` (the result's
    first by default), and the port's own field times `parallel_scale`
    (``None`` where the functional is not parallel).
    """
    from gfem._engine.driven import DrivenRule
    from gfem._engine.port_network import parallel_scale

    if frequency is None:
        frequency = float(result.s.frequency[0])
    omega = 2 * math.pi * frequency
    rule = DrivenRule(model, gfem.Driven([frequency]), OPTIONS)
    system, ports = rule.system(model)
    i = model.port_names.index(port)
    with system.sweep() as sweep:
        functional = ports.functional(omega)[:, i]
        z = sweep.solve(omega, functional[:, None])[0][:, 0]
        c = parallel_scale(functional, ports.rhs(omega)[:, i])
    z = system.operator.expand(z)
    x = dict(zip(result.kept, result.e_fields, strict=True))[frequency, port]
    own = None if c is None else c * np.asarray(x.coefficients)
    return dataclasses.replace(x, coefficients=z), own


@pytest.mark.parametrize("tan_delta", [0.0, 0.05])
def test_a_parallel_port_s_estimate_is_its_adjoint_estimate(tmp_path, tan_delta):
    # A uniform guide's TE10 (lossy or not) is real up to one phase: its S
    # functional is parallel to its excitation, so the adjoint is the port's
    # own field scaled and the free-dual estimate is the adjoint estimate.
    model = rc.lossy_guide_model(tmp_path, 2, "uniform", tan_delta, offset=1e-3)
    problem = gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    z, own = _forced_adjoint(model, result, "p2")
    assert own is not None
    np.testing.assert_allclose(
        z.coefficients, own, rtol=0, atol=1e-12 * np.abs(own).max()
    )
    free = kernels.s_entry(model, result, "p2", "p1")
    adjoint = kernels.s_entry(
        model, result, "p2", "p1", adjoints={(rc.LOSSY_GUIDE_FREQUENCY, "p2"): z}
    )
    assert adjoint[0].value == pytest.approx(free[0].value, rel=1e-12)
    np.testing.assert_allclose(adjoint[0].weights, free[0].weights, rtol=1e-10)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_refine_mesh_solves_the_adjoint_with_the_forward(tmp_path, prepares):
    # Port p2 is not excited, so the S entry needs its adjoint. refine_mesh
    # solves it as one more right-hand side of its own solve (one
    # factorization); error_estimate on the finished result factorizes once
    # more, for the same estimate.
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05, offset=1e-3)
    problem = gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY], excite="p1")
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    refined = gfem.refine_mesh(
        model, problem, estimator=estimator, max_cycles=0, options=OPTIONS
    )
    assert len(prepares) == 1
    again = gfem.error_estimate(refined.result, estimator, model=refined.model)
    assert len(prepares) == 2
    assert again.value == pytest.approx(refined.estimate.value, rel=1e-8)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_a_parallel_lossy_port_takes_no_adjoint_solve(tmp_path, prepares):
    model = rc.lossy_guide_model(tmp_path, 2, "uniform", 0.05)
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    refined = gfem.refine_mesh(
        model,
        gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY]),
        estimator=estimator,
        max_cycles=0,
        options=OPTIONS,
    )
    assert len(prepares) == 1
    gfem.error_estimate(refined.result, estimator, model=refined.model)
    assert len(prepares) == 1


# --- an unexcited read port: its solved adjoint --------------------------------


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_an_unexcited_read_port_is_estimated_through_its_solved_adjoint(
    step, factorizations
):
    # A lumped functional is parallel to its excitation, so the solved
    # adjoint of an unexcited p2 is p2's own field scaled: the estimate is
    # the all-excited one. It rides on the forward's factorization in
    # refine_mesh; error_estimate factorizes once more per frequency.
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    every = gfem.refine_mesh(
        step, rc.step_problem(), estimator=estimator, max_cycles=0, options=OPTIONS
    )
    forward = factorizations()
    assert forward == len(rc.STEP_S)
    gfem.release_memory()  # no handle carries over to the next solve
    one = gfem.refine_mesh(
        step,
        gfem.Driven(list(rc.STEP_S), excite="p1"),
        estimator=estimator,
        max_cycles=0,
        options=OPTIONS,
    )
    assert factorizations() == 2 * forward
    assert one.estimate.value == pytest.approx(every.estimate.value, rel=1e-12)
    np.testing.assert_allclose(
        one.estimate.indicators, every.estimate.indicators, rtol=1e-10
    )
    gfem.release_memory()
    again = gfem.error_estimate(one.result, estimator, model=one.model)
    assert factorizations() == 3 * forward
    assert again.value == pytest.approx(one.estimate.value, rel=1e-12)


def test_an_adjoint_load_callback_sees_each_frequency_expanded_solution(step):
    # The callback is called per frequency with that frequency's solution on
    # the fields' space and returns keyed loads; each key's adjoint comes
    # back per (frequency, key). Truth: p2's S-functional adjoint is p2's
    # own field times its parallel scale (a lumped port).
    from gfem.solve import _solve

    seen = {}

    def loads(context):
        seen[context.frequency] = np.array(context.solution)
        assert context.excite == ("p1",)
        return {("dual", "p2"): context.s_functional("p2")}

    frequencies = list(rc.STEP_S)
    result, adjoints = _solve(
        step,
        gfem.Driven(frequencies, excite="p1", keep_fields=True),
        options=OPTIONS,
        adjoint_loads=loads,
    )
    full = gfem.solve(step, gfem.Driven(frequencies, keep_fields=True), options=OPTIONS)
    fields = dict(zip(result.kept, result.e_fields, strict=True))
    assert sorted(adjoints) == [(f, ("dual", "p2")) for f in frequencies]
    for f in frequencies:
        x = np.asarray(fields[f, "p1"].coefficients)
        np.testing.assert_allclose(seen[f][:, 0], x, rtol=0, atol=1e-14 * abs(x).max())
        _, own = _forced_adjoint(step, full, "p2", f)
        z = adjoints[f, ("dual", "p2")]
        np.testing.assert_allclose(z, own, rtol=0, atol=1e-10 * abs(own).max())


def test_an_empty_adjoint_load_solves_nothing_more(step, factorizations):
    from gfem.solve import _solve

    _, adjoints = _solve(
        step,
        gfem.Driven([3e9], excite="p1"),
        options=OPTIONS,
        adjoint_loads=lambda _context: {},
    )
    assert adjoints == {}
    assert factorizations() == 1


class _Unconverged:
    """A handle whose solve stops above its tolerance."""

    def solve_counted(self, *_args, **_kwargs):
        raise gfem.LinearSolverConvergenceError("GMRES stopped at 1e-3.")


def test_an_unconverged_adjoint_is_not_reported_as_a_forward_miss(
    tmp_path, monkeypatch
):
    # Only the adjoint solve fails: the forward solved every frequency, so
    # the error is the adjoint's, with no partial sweep.
    from gfem._engine.system_operator import SystemSweep

    monkeypatch.setattr(SystemSweep, "reuse_at", lambda _self, _omega: _Unconverged())
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    frequencies = [rc.LOSSY_GUIDE_FREQUENCY, 1.01 * rc.LOSSY_GUIDE_FREQUENCY]
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        gfem.refine_mesh(
            model,
            gfem.Driven(frequencies, excite="p1"),  # p2's row takes an adjoint
            estimator=gfem.QuantityEstimator(s=("p2", "p1")),
            max_cycles=0,
            options=OPTIONS,
        )
    message = str(caught.value)
    assert "adjoint" in message
    assert "'p2'" in message
    assert "GMRES stopped at 1e-3." in message
    assert "uniform sweep stopped" not in message
    assert caught.value.frequency == frequencies[0]
    assert caught.value.unsolved is None
    assert caught.value.result is None


def _adjoint_rule(model, problem):
    from gfem._engine.driven import DrivenRule, port_loads
    from gfem._engine.solve_request import SolveRequest

    rule = DrivenRule(model, problem, OPTIONS)
    return rule, SolveRequest(adjoint_loads=port_loads(("p2",)))


def test_adjoint_ports_never_reach_a_differentiated_solve(tmp_path):
    # A backward pass keeps no adjoint columns: the forward refuses them.
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    rule, request = _adjoint_rule(model, gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY]))
    with pytest.raises(RuntimeError, match="differentiated"):
        rule.forward(model, keep=True, request=request)


def test_adjoint_ports_never_reach_an_adaptive_sweep(tmp_path):
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    f = rc.LOSSY_GUIDE_FREQUENCY
    problem = gfem.Driven(np.linspace(f, 1.1 * f, 5), sweep="adaptive")
    rule, request = _adjoint_rule(model, problem)
    with pytest.raises(RuntimeError, match="adaptive"):
        rule.forward(model, keep=False, request=request)


@pytest.mark.usefixtures("no_solve")
def test_frequency_workers_refuse_an_s_estimate_on_a_lossy_wave_port(tmp_path):
    # Frequency workers take wave ports but solve no adjoint:
    # refine_mesh raises before any solve rather than drop the adjoint ports.
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    f = rc.LOSSY_GUIDE_FREQUENCY
    with pytest.raises(
        gfem.MeshRefinementUnsupportedError, match="frequency_workers=1"
    ) as caught:
        gfem.refine_mesh(
            model,
            gfem.Driven([f, 1.01 * f]),
            estimator=gfem.QuantityEstimator(s=("p2", "p1")),
            max_cycles=0,
            options=dataclasses.replace(OPTIONS, frequency_workers=2),
        )
    assert caught.value.feature == "frequency workers"


def test_the_s_estimate_scales_with_the_lossy_de_embedding(tmp_path):
    # S_ij = (O_ij - delta_ij) d_i d_j, |d| = exp(alpha offset) > 1 on a
    # lossy port: the fields stay, the estimate grows by |d_i d_j|.
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    values, d = [], 1.0
    for offset in (0.0, 1e-3):
        model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05, offset=offset)
        result = gfem.solve(
            model,
            gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY], keep_fields=True),
            options=OPTIONS,
        )
        values.append(gfem.error_estimate(result, estimator, model=model).value)
        alpha = [float(np.real(result.port_modes[p].gamma[0])) for p in ("p1", "p2")]
        d = math.exp(sum(alpha) * offset)
    assert d > 1.01
    assert values[1] == pytest.approx(values[0] * d, rel=1e-10)


@pytest.mark.parametrize(
    ("case", "estimator"),
    [
        ("coax-anisotropic", ("capacitance", ("inner", "inner"))),
        ("step-mu", ("s", ("p2", "p1"))),
        ("step-anisotropic-mu", ("s", ("p2", "p1"))),
    ],
)
def test_anisotropic_permittivity_and_permeability_are_estimated(
    coax, step, *, case, estimator
):
    # Verified in tests/verification/test_quantity_estimate.py (anisotropic
    # coax, permeability and anisotropic lines and boxes).
    rotated = gfem.Material(
        eps_r=(3.0, 5.0, 2.0), mu_r=(1.5, 2.5, 2.0), axes=tuple(map(tuple, rc.ROT3.T))
    )
    model, problem = {
        "coax-anisotropic": (
            _with(coax, materials={"dielectric": gfem.Material(eps_r=(1, 2, 3))}),
            rc.COAX_PROBLEM,
        ),
        "step-mu": (
            _with(step, materials={"dielectric": gfem.Material(eps_r=4.0, mu_r=2.0)}),
            gfem.Driven([3e9], keep_fields=True),
        ),
        "step-anisotropic-mu": (
            _with(step, materials={"dielectric": rotated}),
            gfem.Driven([3e9], keep_fields=True),
        ),
    }[case]
    keyword, key = estimator
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(**{keyword: key}), model=model
    )
    assert np.all(np.isfinite(estimate.values))
    assert np.all(np.asarray(estimate.values) > 0)


def test_a_lossless_wave_port_s_entry_is_estimated(step):
    model = _with(step, ports=_waves())
    result = gfem.solve(model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS)
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=model
    )
    assert 0 < estimate.value < 1e-2


def test_a_de_embedding_offset_leaves_the_s_estimate_unchanged(step):
    # The offset multiplies S by exp(i beta offset) after the solve, of
    # modulus 1 on a lossless port: the fields and the estimate stay.
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    values = []
    for offset in (0.0, 1e-3):
        model = _with(step, ports=_waves(offset=offset))
        result = gfem.solve(
            model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS
        )
        values.append(gfem.error_estimate(result, estimator, model=model).value)
    assert values[1] == pytest.approx(values[0], rel=1e-12)


# --- reading a result --------------------------------------------------------


def test_the_capacitance_estimate_is_the_squared_field_estimate(coax):
    # C_h - C = a(e, e) for a terminal at 1 V: the relative capacitance
    # estimate is the relative energy estimate squared.
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=coax)
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    quantity = gfem.error_estimate(result, estimator, model=coax)
    assert quantity.values == pytest.approx((field.value**2,), rel=1e-10)
    assert quantity.estimator == estimator
    # and it marks the same tetrahedra: its indicators are the field estimator's
    np.testing.assert_allclose(quantity.indicators, field.indicators, rtol=1e-10)


def test_the_diagonal_capacitance_kernel_is_the_field_kernel(coax):
    # The (i, j) kernel at i = j is the diagonal estimate it replaced: the
    # field estimator's squared indicators of that terminal's potential, to
    # round-off (threaded integrals sum in another order on the two paths).
    problem = gfem.Electrostatic(terminals=["inner", "outer"])
    result = gfem.solve(coax, problem, options=OPTIONS)
    (group,) = kernels.electrostatic(coax, result)
    for index, terminal in enumerate(("inner", "outer")):
        terms = kernels.capacitance(coax, result, terminal, terminal)
        assert terms.value == pytest.approx(float(group[index].sum()), rel=1e-13)
        np.testing.assert_allclose(terms.weights, group[index], rtol=1e-13, atol=0)


def test_an_off_diagonal_capacitance_is_estimated(tmp_path):
    model = rc.sheet_model(tmp_path, 2)
    result = gfem.solve(model, rc.SHEET_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    assert 0 < estimate.value < 1
    assert float(np.sum(estimate.indicators**2)) == pytest.approx(
        estimate.value, rel=1e-12
    )
    # the entry is symmetric, and so is its estimate
    swapped = gfem.QuantityEstimator(capacitance=("b", "a"))
    assert gfem.error_estimate(result, swapped, model=model).value == pytest.approx(
        estimate.value, rel=1e-12
    )


def test_an_s_estimate_has_one_value_per_frequency(step):
    result = gfem.solve(
        step, gfem.Driven([3e9, 8e9], keep_fields=True), options=OPTIONS
    )
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    assert len(estimate.values) == 2
    assert estimate.value == max(estimate.values)
    assert all(0 < v < 1e-2 for v in estimate.values)


@pytest.mark.parametrize("frequency", [3e9, 8e9])
def test_an_s_estimate_marks_as_the_field_estimator(step, frequency):
    # The indicator is |a_K(e~_1, e~_2)|, the curl and mass terms of one
    # tetrahedron together: on the step line it follows the field estimator's
    # (correlation 0.9999997 at n = 4 in the DWR prototype) and marks the same
    # tetrahedra.
    result = gfem.solve(step, gfem.Driven(frequency, keep_fields=True), options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=step)
    quantity = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    assert np.corrcoef(quantity.indicators**2, field.indicators**2)[0, 1] > 0.99999
    np.testing.assert_array_equal(
        _dorfler(quantity.indicators, 0.5), _dorfler(field.indicators, 0.5)
    )


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_cycles_record_the_quantity(coax, step, cavity):
    ref = gfem.refine_mesh(
        coax,
        rc.COAX_PROBLEM,
        estimator=gfem.QuantityEstimator(capacitance=("inner", "inner")),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == float(ref.result.capacitance[0, 0])
    assert ref.cycles[0].estimate == ref.estimate.value
    ref = gfem.refine_mesh(
        step,
        rc.step_problem(),
        estimator=gfem.QuantityEstimator(s=("p2", "p1")),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == tuple(complex(v) for v in ref.result.s["p2", "p1"])
    ref = gfem.refine_mesh(
        cavity,
        rc.CAVITY_PROBLEM,
        estimator=gfem.QuantityEstimator(omega=0),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == complex(ref.result.omega[0])


def test_each_quantity_estimate_names_its_kind(coax, step, cavity):
    cases = [
        (coax, rc.COAX_PROBLEM, {"capacitance": ("inner", "inner")}, "capacitance"),
        (step, rc.step_problem(), {"s": ("p2", "p1")}, "s"),
        (cavity, rc.CAVITY_PROBLEM, {"omega": 0}, "omega"),
    ]
    for model, problem, key, kind in cases:
        if isinstance(problem, gfem.Driven):
            kept = dataclasses.replace(problem, keep_fields=True)
        else:
            kept = problem
        result = gfem.solve(model, kept, options=OPTIONS)
        estimator = gfem.QuantityEstimator(**key)
        assert gfem.error_estimate(result, estimator, model=model).kind == kind


# --- the kernels: complex sums, per-tetrahedron terms, one recovery ------------

#: Each kernel's value before its terms were exposed (b5ca8d5, order 2,
#: direct, n = 2): the coax's C_inner,inner, the step line's S21 and S11 at 3
#: and 8 GHz, the L cavity's first omega.
KERNEL_VALUES = {
    "capacitance": (0.013389066116039919,),
    ("p2", "p1"): (0.001994273976810584, 0.005197056917579232),
    ("p1", "p1"): (0.0014252233956337092, 0.002609020921160659),
    "omega": (0.021384095777234734,),
}


def test_the_kernels_keep_their_values(coax, step, cavity):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    terms = kernels.capacitance(coax, result, "inner", "inner")
    assert (terms.value,) == pytest.approx(KERNEL_VALUES["capacitance"], rel=1e-12)
    problem = dataclasses.replace(rc.step_problem(), keep_fields=True)
    result = gfem.solve(step, problem, options=OPTIONS)
    for entry in (("p2", "p1"), ("p1", "p1")):
        values = tuple(t.value for t in kernels.s_entry(step, result, *entry))
        assert values == pytest.approx(KERNEL_VALUES[entry], rel=1e-12)
    result = gfem.solve(cavity, rc.CAVITY_PROBLEM, options=OPTIONS)
    terms = kernels.omega(cavity, result, 0)
    assert (terms.value,) == pytest.approx(KERNEL_VALUES["omega"], rel=1e-12)


def test_a_kernel_returns_its_complex_sums_and_terms(step, cavity):
    problem = gfem.Driven([3e9], keep_fields=True)
    result = gfem.solve(step, problem, options=OPTIONS)
    (terms,) = kernels.s_entry(step, result, "p2", "p1")
    n = len(step.mesh.tetrahedra)
    assert terms.curl.shape == terms.mass.shape == (n,)
    assert np.iscomplexobj(terms.curl)
    assert np.iscomplexobj(terms.mass)
    assert terms.curl_sum == pytest.approx(terms.curl.sum(), rel=1e-12)
    assert terms.mass_sum == pytest.approx(terms.mass.sum(), rel=1e-12)
    lam = abs(terms.lam)
    assert terms.value == pytest.approx(
        terms.scale * (abs(terms.curl_sum) + lam * abs(terms.mass_sum)), rel=1e-14
    )
    np.testing.assert_allclose(
        terms.weights, terms.scale * np.abs(terms.curl - terms.lam * terms.mass)
    )
    result = gfem.solve(cavity, rc.CAVITY_PROBLEM, options=OPTIONS)
    terms = kernels.omega(cavity, result, 0)
    assert isinstance(terms.lam, complex)


@pytest.fixture
def recoveries(monkeypatch):
    """The `FluxRecovery` objects built."""
    built = []
    original = kernels.FluxRecovery.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(kernels.FluxRecovery, "__init__", counting)
    return built


def test_two_quantities_on_one_mesh_build_one_recovery(step, recoveries):
    # One cycle's estimates share one recovery and one set of recovered
    # errors per field: S21 and S11 at two frequencies read the fields of p1
    # and p2 at each, through one `RecoveryCache`.
    from gfem.refinement import _estimate

    problem = dataclasses.replace(rc.step_problem(), keep_fields=True)
    result = gfem.solve(step, problem, options=OPTIONS)
    cache = kernels.RecoveryCache()
    s21 = _estimate(
        step, result, gfem.QuantityEstimator(s=("p2", "p1")), threads=None, cache=cache
    )
    s11 = _estimate(
        step, result, gfem.QuantityEstimator(s=("p1", "p1")), threads=None, cache=cache
    )
    assert len(recoveries) == 1
    assert s21.values == pytest.approx(KERNEL_VALUES["p2", "p1"], rel=1e-12)
    assert s11.values == pytest.approx(KERNEL_VALUES["p1", "p1"], rel=1e-12)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_one_refine_mesh_cycle_builds_one_recovery(step, recoveries):
    gfem.refine_mesh(
        step,
        rc.step_problem(),
        estimator=gfem.QuantityEstimator(s=("p2", "p1")),
        tolerance=1e-5,
        max_cycles=1,
        options=OPTIONS,
    )
    assert len(recoveries) == 2  # one per solved mesh, for both frequencies


# --- eigenmode tracking ------------------------------------------------------


def test_a_mode_in_a_degenerate_cluster_raises(cavity):
    result = gfem.solve(
        cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2), options=OPTIONS
    )
    degenerate = dataclasses.replace(result, cluster=jnp.zeros(2, int))
    estimator = gfem.QuantityEstimator(omega=0)
    gfem.error_estimate(result, estimator, model=cavity)
    with pytest.raises(gfem.DegenerateModeError, match="cluster_rtol"):
        gfem.error_estimate(degenerate, estimator, model=cavity)


def test_a_mode_within_cluster_rtol_of_another_raises(cavity):
    # refine_mesh's rule, with the default SolverOptions().cluster_rtol = 1e-6
    result = gfem.solve(
        cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2), options=OPTIONS
    )
    omega = np.asarray(result.omega)
    estimator = gfem.QuantityEstimator(omega=0)
    apart = dataclasses.replace(
        result, omega=jnp.asarray([omega[0], omega[0] * 1.00001])
    )
    gfem.error_estimate(apart, estimator, model=cavity)
    close = dataclasses.replace(
        result, omega=jnp.asarray([omega[0], omega[0] * 1.0000001])
    )
    with pytest.raises(gfem.DegenerateModeError, match="cluster_rtol=1e-06"):
        gfem.error_estimate(close, estimator, model=cavity)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_mode_is_followed_by_nearest_omega(cavity):
    # Mode 1 of three around the first resonance: on every refined mesh the
    # estimate stays with the mode nearest the previous omega.
    problem = gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=3)
    ref = gfem.refine_mesh(
        cavity,
        problem,
        estimator=gfem.QuantityEstimator(omega=1),
        tolerance=1e-9,
        max_cycles=2,
        options=OPTIONS,
    )
    omegas = [cycle.value for cycle in ref.cycles]
    first = np.asarray(gfem.solve(cavity, problem, options=OPTIONS).omega)
    assert omegas[0] == pytest.approx(complex(first[1]), rel=1e-13)
    for before, after in itertools.pairwise(omegas):
        assert abs(after - before) < abs(first[1] - first[0]) / 2


# --- cluster means -----------------------------------------------------------

#: The 1 x 1 x 1.5 mm box's TE101/TE011 pair, exactly degenerate (Hz).
PAIR_F = const.c * 1e3 * math.sqrt(1 + 1 / 2.25) / 2


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """The pair box at n = 2, its three lowest modes: the pair, then TM110."""
    model = rc.cluster_box_model(tmp_path_factory.mktemp("pair"), (1.0, 1.0, 1.5), 2)
    result = gfem.solve(
        model, gfem.Eigenmode(target=PAIR_F, n_modes=3), options=OPTIONS
    )
    return model, result


def _fake_modes(*omegas, cluster=None):
    """A stand-in eigenmode result: its omegas, each mode its own cluster."""
    labels = np.arange(len(omegas)) if cluster is None else np.asarray(cluster)
    return types.SimpleNamespace(omega=np.asarray(omegas, complex), cluster=labels)


def test_a_cluster_names_two_or_more_distinct_modes():
    assert gfem.QuantityEstimator(omega=(1, 0)).omega == (0, 1)
    assert gfem.QuantityEstimator(omega=[2, 0, 1]) == gfem.QuantityEstimator(
        omega=(0, 1, 2)
    )
    with pytest.raises(ValueError, match="at least two modes"):
        gfem.QuantityEstimator(omega=(0,))
    with pytest.raises(ValueError, match="names a mode twice"):
        gfem.QuantityEstimator(omega=(0, 0))
    with pytest.raises(ValueError, match=">= 0"):
        gfem.QuantityEstimator(omega=(0, -1))
    with pytest.raises(TypeError, match="mode indices"):
        gfem.QuantityEstimator(omega=(0, 1.0))
    with pytest.raises(TypeError, match="Q of a cluster"):
        gfem.QuantityEstimator(q=(0, 1))


@pytest.mark.usefixtures("no_solve")
def test_a_cluster_without_a_guard_mode_raises(cavity):
    for n_modes, match in ((2, r"gfem\.Eigenmode\(n_modes=3\)"), (1, "at or above")):
        with pytest.raises(ValueError, match=match):
            gfem.refine_mesh(
                cavity,
                gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=n_modes),
                estimator=gfem.QuantityEstimator(omega=(0, 1)),
            )


@pytest.mark.usefixtures("no_solve")
@pytest.mark.parametrize("other", [{"omega": 1}, {"q": 0}, {"omega": (1, 2)}])
def test_a_mode_of_a_cluster_cannot_be_named_again(cavity, other):
    with pytest.raises(ValueError, match="followed as a whole"):
        gfem.refine_mesh(
            cavity,
            gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=4),
            estimator=(
                gfem.QuantityEstimator(omega=(0, 1)),
                gfem.QuantityEstimator(**other),
            ),
            tolerance=1e-3,
        )


def test_an_error_estimate_of_a_cluster_needs_its_guard_mode(pair):
    model, result = pair
    with pytest.raises(ValueError, match=r"n_modes=4"):
        gfem.error_estimate(
            result, gfem.QuantityEstimator(omega=(0, 1, 2)), model=model
        )


def test_a_single_mode_in_a_degenerate_cluster_names_its_members(pair):
    model, result = pair
    assert np.asarray(result.cluster)[0] == np.asarray(result.cluster)[1]
    with pytest.raises(
        gfem.DegenerateModeError,
        match=r"mode 0 at .* GHz is in a degenerate cluster with mode 1 \(.* GHz\).*"
        r"QuantityEstimator\(omega=\(0, 1\)\), or use gfem\.FieldEstimator\(\)",
    ):
        gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)


def test_a_cluster_naming_part_of_a_degenerate_cluster_raises(pair):
    model, result = pair
    triple = dataclasses.replace(result, cluster=jnp.zeros(3, int))
    with pytest.raises(
        gfem.DegenerateModeError,
        match=r"mode 0 at .* is in a degenerate cluster with modes 1 \(.*\) and "
        r"2 \(.*\).*name all of them, QuantityEstimator\(omega=\(0, 1, 2\)\)",
    ):
        gfem.error_estimate(triple, gfem.QuantityEstimator(omega=(0, 1)), model=model)


def test_a_cluster_too_near_another_mode_raises_on_the_first_mesh(pair):
    model, result = pair
    w = complex(np.asarray(result.omega)[:2].mean())
    near = [w * (1 - 1e-3), w * (1 + 1e-3), w * (1 + 1.5e-3)]
    crowded = dataclasses.replace(
        result, omega=jnp.asarray(near), cluster=jnp.arange(3)
    )
    with pytest.raises(gfem.DegenerateModeError, match="not clearly apart from mode 2"):
        gfem.error_estimate(crowded, gfem.QuantityEstimator(omega=(0, 1)), model=model)


def test_a_cluster_mean_is_estimated_and_followed(pair):
    model, result = pair
    estimator = gfem.QuantityEstimator(omega=(0, 1))
    estimate = gfem.error_estimate(result, estimator, model=model)
    assert estimate.kind == "cluster mean"
    assert len(estimate.values) == 1
    assert estimate.indicators.shape == (len(model.mesh.tetrahedra),)
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            model,
            gfem.Eigenmode(target=PAIR_F, n_modes=3),
            estimator=estimator,
            tolerance=1e-9,
            max_cycles=2,
            options=OPTIONS,
        )
    assert ref.estimate.kind == "cluster mean"
    assert ref.cycles[0].estimate == pytest.approx(estimate.value, rel=1e-9)
    omegas = np.asarray(ref.result.omega)
    last = ref.cycles[-1]
    assert sorted(last.modes) == [0, 1]
    assert last.value == pytest.approx(omegas[[last.modes[0], last.modes[1]]].mean())
    exact = 2 * math.pi * PAIR_F
    errors = [abs(cycle.value / exact - 1) for cycle in ref.cycles]
    assert errors[-1] < errors[0]


def test_the_cluster_kernel_of_one_mode_is_the_omega_kernel(cavity):
    result = gfem.solve(
        cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2), options=OPTIONS
    )
    for mode in (0, 1):
        single = kernels.omega(cavity, result, mode)
        cluster = kernels.omega_cluster(cavity, result, (mode,))
        assert cluster.value == pytest.approx(single.value, rel=1e-12)
        np.testing.assert_allclose(cluster.weights, single.weights, rtol=1e-10)


def test_the_cluster_estimate_is_unchanged_by_a_basis_of_the_cluster(pair):
    # Any invertible mix of the pair's fields spans the same cluster: the trace
    # form, its value and its indicators, does not see the basis.
    model, result = pair
    fields = list(result.e_fields)
    x = np.stack([np.asarray(f.coefficients) for f in fields[:2]], axis=1)
    mix = np.array([[0.8, -0.9], [0.6, 1.7]])
    mixed = x @ mix
    rebased = dataclasses.replace(
        result,
        e_fields=[
            dataclasses.replace(fields[0], coefficients=jnp.asarray(mixed[:, 0])),
            dataclasses.replace(fields[1], coefficients=jnp.asarray(mixed[:, 1])),
            fields[2],
        ],
    )
    before = kernels.omega_cluster(model, result, (0, 1))
    after = kernels.omega_cluster(model, rebased, (0, 1))
    assert after.value == pytest.approx(before.value, rel=1e-9)
    np.testing.assert_allclose(after.weights, before.weights, rtol=1e-7, atol=0)


def test_a_cluster_follows_the_modes_nearest_its_previous_mean():
    from gfem.refinement import _track_cluster

    # the pair split apart and swapped order with a far mode below
    result = _fake_modes(10.002, 7.0, 9.999, 13.0)
    assert _track_cluster(result, (0, 1), 10.0, 1e-3) == (0, 2)


def test_a_cluster_whose_mean_moved_too_far_raises():
    from gfem.refinement import _track_cluster

    with pytest.raises(gfem.DegenerateModeError, match=r"moved their mean.*more modes"):
        _track_cluster(_fake_modes(10.5, 10.6, 14.0), (0, 1), 10.0, 1e-3)


def test_a_cluster_too_near_another_mode_raises_on_a_refined_mesh():
    from gfem.refinement import _track_cluster

    # a member 0.01 from the previous mean, the third mode 0.015 from it
    with pytest.raises(gfem.DegenerateModeError, match="not clearly apart from mode 2"):
        _track_cluster(_fake_modes(9.99, 10.01, 10.015), (0, 1), 10.0, 1.0)


# --- lossy eigenmodes --------------------------------------------------------


def _box(directory, material=None, top=None):
    """The 1 x 0.7 x 0.4 mm box, h = 0.2 mm, PEC but for ``top``."""
    path = sm.write_msh22(
        directory / "box.msh", sm.plate(1.0, 0.7, (("air", 0.4),), (5, 4, 2))
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": material or gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("side"), top or gfem.PEC("top")],
    )


#: Near the box's 110 mode, Hz.
BOX_110 = 3e8 * math.sqrt(1 / 1e-6 + 1 / 0.49e-6) / 2


def test_a_lossless_mode_keeps_its_estimate(tmp_path):
    # Measured with the lossless kernel (before the lossy form): on a
    # lossless mode the lossy form is the same number.
    model = _box(tmp_path)
    problem = gfem.Eigenmode(target=BOX_110, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    assert estimate.value == pytest.approx(0.0016711308802387496, rel=1e-12)
    assert np.sum(estimate.indicators**2) == pytest.approx(
        0.0008213829986877815, rel=1e-12
    )


@pytest.mark.parametrize(
    ("material", "top"),
    [
        (gfem.Material(tan_delta=0.1, conductivity=5.0), None),
        (gfem.MaterialField(eps_r=1 - 0.1j), None),
        (None, gfem.Impedance("top", resistance=377.0)),
        (None, gfem.Absorbing("top")),
        (None, gfem.Conductor("top", conductivity=1e5)),
    ],
    ids=["volume", "complex-eps", "impedance", "absorbing", "conductor"],
)
def test_a_lossy_mode_is_estimated(tmp_path, material, top):
    model = _box(tmp_path, material, top)
    problem = gfem.Eigenmode(target=BOX_110, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    assert abs(complex(result.omega[0]).imag) > 0
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    assert 0 < estimate.value < 1e-2
    assert np.all(np.isfinite(estimate.indicators))


#: Each eigen term the omega estimate admits: (material, boundary).
LIFTED = {
    "second-order-absorbing": (None, gfem.Absorbing("side", order=2)),
    "lossy-mu": (gfem.MaterialField(eps_r=2.0, mu_r=2 * (1 - 0.1j)), None),
    "lossy-anisotropic": (
        gfem.Material(eps_r=(2.0, 3.0, 4.0), mu_r=(1.5, 2.0, 2.5), tan_delta=0.1),
        None,
    ),
}


def _lifted_box(directory, name):
    material, boundary = LIFTED[name]
    model = _box(directory, material)
    if boundary is not None:
        model = _with(model, boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), boundary])
    return model


@pytest.mark.parametrize("name", list(LIFTED))
def test_a_lifted_lossy_mode_is_estimated(tmp_path, name):
    # second-order absorbing walls, a complex mu_r and loss inside
    # an anisotropic material no longer refuse the omega estimate.
    model = _lifted_box(tmp_path, name)
    problem = gfem.Eigenmode(target=BOX_110, n_modes=2)
    result = gfem.solve(model, problem, options=OPTIONS)
    assert abs(complex(result.omega[0]).imag) > 0
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    assert estimate.kind == "omega"
    assert 0 < estimate.value < 1e-1
    assert np.all(np.isfinite(estimate.indicators))


LOSSY_TENSOR = gfem.Material(eps_r=(1, 2, 3), tan_delta=0.01)


@pytest.mark.parametrize(
    ("fixture", "problem", "estimator"),
    [
        ("cavity", rc.CAVITY_PROBLEM, gfem.QuantityEstimator(omega=0)),
        ("cavity", rc.CAVITY_PROBLEM, gfem.FieldEstimator()),
        (
            "step",
            gfem.Driven([5e9]),
            gfem.QuantityEstimator(s=("p2", "p1")),
        ),
    ],
    ids=["omega", "field", "s"],
)
def test_loss_in_an_anisotropic_material_is_estimated(
    request, fixture, problem, estimator
):
    # every tensor gfem represents is R D R^T with real axes, so
    # the modulus weights of the complex eps_eff are exact; S, omega and the
    # field estimate it.
    model = request.getfixturevalue(fixture)
    lossy = _with(model, materials=dict.fromkeys(model.materials, LOSSY_TENSOR))
    ref = gfem.refine_mesh(
        lossy, problem, estimator=estimator, tolerance=10.0, options=OPTIONS
    )
    assert ref.stop == "tolerance"
    assert np.all(np.isfinite(ref.estimate.values))
    assert np.all(np.isfinite(ref.estimate.indicators))
    assert ref.estimate.value > 0


# --- driven refusals ----------------------------------------------------------


class _SolvedError(Exception):
    """Raised in place of the first solve: every check before it passed."""


@pytest.fixture
def reaches_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise _SolvedError

    monkeypatch.setattr("gfem.refinement._solve", solve)


S21 = gfem.QuantityEstimator(s=("p2", "p1"))


def _refused(model, problem, estimator=S21):
    """The `MeshRefinementUnsupportedError` refine_mesh raises before any solve."""
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(model, problem, estimator=estimator, options=OPTIONS)
    assert "FieldEstimator()" in str(caught.value)  # the remedy
    return caught.value


def _admitted(model, problem, estimator=S21):
    """Every check of refine_mesh passes: it reaches the first solve."""
    with pytest.raises(_SolvedError):
        gfem.refine_mesh(model, problem, estimator=estimator, options=OPTIONS)


@pytest.mark.parametrize(
    "surface",
    [
        gfem.Impedance("side", resistance=100.0),
        gfem.Impedance("side", inductance=1e-9),
        gfem.Absorbing("side"),
        gfem.Absorbing("side", order=2),
    ],
    ids=["resistive-impedance", "inductive-impedance", "absorbing", "absorbing-2"],
)
@pytest.mark.usefixtures("no_solve")
def test_an_impedance_or_absorbing_surface_refuses_the_s_estimate(step, surface):
    model = _with(step, boundaries=[*step.boundaries, surface])
    error = _refused(model, gfem.Driven([5e9]))
    assert error.feature == "impedance or absorbing surface"
    assert "'side'" in str(error)


def test_an_estimate_of_s_by_an_impedance_surface_is_refused(step):
    model = _with(step, boundaries=[*step.boundaries, gfem.Absorbing("side")])
    result = gfem.solve(model, gfem.Driven([5e9], keep_fields=True), options=OPTIONS)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, S21, model=model)
    assert caught.value.feature == "impedance or absorbing surface"
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert field.value > 0  # unclaimed there, not refused


@pytest.mark.usefixtures("reaches_solve")
def test_a_conductor_surface_admits_the_s_estimate(step):
    conductor = gfem.Conductor("side", conductivity=1e5)
    _admitted(_with(step, boundaries=[*step.boundaries, conductor]), gfem.Driven([5e9]))


@pytest.mark.parametrize(
    "estimator",
    [gfem.QuantityEstimator(omega=0), gfem.FieldEstimator()],
    ids=["omega", "field"],
)
@pytest.mark.usefixtures("reaches_solve")
def test_an_impedance_surface_admits_omega_and_the_field(cavity, estimator):
    wall = gfem.Impedance("wall", resistance=377.0)
    _admitted(_with(cavity, boundaries=[wall]), rc.CAVITY_PROBLEM, estimator)


@pytest.mark.usefixtures("reaches_solve")
def test_an_impedance_surface_admits_the_field_estimate_of_a_driven_model(step):
    model = _with(step, boundaries=[*step.boundaries, gfem.Absorbing("side")])
    _admitted(model, gfem.Driven([5e9]), gfem.FieldEstimator())


MU_HALVES = (gfem.Material(), gfem.Material(mu_r=2.5))


@pytest.mark.usefixtures("no_solve")
def test_a_lumped_port_on_a_permeability_contrast_refuses_the_s_estimate(tmp_path):
    model = cc.split_line_model(tmp_path, "beside", MU_HALVES, 2)
    error = _refused(model, gfem.Driven([15e9]))
    assert error.feature == "lumped port spanning a permeability contrast"
    assert "'p1'" in str(error)


@pytest.mark.usefixtures("no_solve")
def test_a_lumped_port_across_a_permeability_interface_refuses_too(tmp_path):
    model = cc.split_line_model(tmp_path, "across", MU_HALVES, 2)
    error = _refused(model, gfem.Driven([3e9]))
    assert error.feature == "lumped port spanning a permeability contrast"


def test_an_estimate_of_s_by_a_permeability_contrast_is_refused(tmp_path):
    model = cc.split_line_model(tmp_path, "beside", MU_HALVES, 2)
    result = gfem.solve(model, gfem.Driven([15e9], keep_fields=True), options=OPTIONS)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, S21, model=model)
    assert caught.value.feature == "lumped port spanning a permeability contrast"


@pytest.mark.parametrize(
    "fill",
    [
        gfem.Material(eps_r=2.0, mu_r=2.0),
        gfem.Material(mu_r=(1.5, 2.0, 2.5)),
        gfem.Material(mu_r=(1.5, 2.0, 2.5), axes=tuple(map(tuple, rc.ROT3.T))),
    ],
    ids=["uniform", "diagonal", "rotated"],
)
@pytest.mark.usefixtures("reaches_solve")
def test_a_uniform_permeability_at_a_lumped_port_is_admitted(tmp_path, fill):
    # The tensors are compared whole: one anisotropic fill is no contrast.
    _admitted(
        cc.split_line_model(tmp_path, "beside", (fill, fill), 2), gfem.Driven([3e9])
    )


@pytest.mark.usefixtures("reaches_solve")
def test_a_wave_port_on_a_permeability_contrast_is_admitted(tmp_path):
    model = cc.split_line_model(tmp_path, "beside", MU_HALVES, 2, ports="wave")
    _admitted(model, gfem.Driven([15e9]))


@pytest.mark.usefixtures("reaches_solve")
def test_a_lumped_port_on_a_permittivity_contrast_is_admitted(tmp_path):
    eps = (gfem.Material(eps_r=3.0), gfem.Material())
    _admitted(cc.split_line_model(tmp_path, "across", eps, 2), gfem.Driven([3e9]))


@pytest.mark.usefixtures("reaches_solve")
def test_a_lumped_port_on_a_conductivity_contrast_is_admitted(tmp_path):
    model = cc.contrast_line_model(tmp_path, "sigma", 1, ports="lumped")
    _admitted(model, gfem.Driven([10e9]))


@pytest.mark.usefixtures("reaches_solve")
def test_a_lumped_port_on_a_permeability_contrast_admits_the_field_estimate(tmp_path):
    model = cc.split_line_model(tmp_path, "beside", MU_HALVES, 2)
    _admitted(model, gfem.Driven([15e9]), gfem.FieldEstimator())


#: The Drude core of the plasmonic tests: Re eps -29.8 at 3 GHz, -4.9 at 8 GHz.
DRUDE_CORE = gfem.PermittivityLaw(
    terms=[gfem.Drude(plasma_frequency=20e9, damping=2e9)]
)


@pytest.fixture(scope="module")
def drude_line(tmp_path_factory):
    """A 10 x 2 x 1 mm line, eps_r 4, with a `DRUDE_CORE` at 4 < x < 6 mm."""
    path = tmp_path_factory.mktemp("drude") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "dielectric": gfem.Material(eps_r=4.0),
            "core": gfem.Material(eps_r=DRUDE_CORE),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=30.0),
        ],
    )


@pytest.mark.usefixtures("no_solve")
def test_a_law_in_the_plasmonic_window_refuses_the_s_estimate(drude_line):
    error = _refused(drude_line, gfem.Driven([3e9, 8e9]))
    assert error.feature == "plasmonic permittivity"
    message = str(error)
    assert "'core'" in message
    assert "8 GHz" in message
    assert "Re eps = -4.88" in message


def test_an_estimate_of_s_in_the_plasmonic_window_is_refused(drude_line):
    problem = gfem.Driven([8e9], keep_fields=True)
    result = gfem.solve(drude_line, problem, options=OPTIONS)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, S21, model=drude_line)
    assert caught.value.feature == "plasmonic permittivity"
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=drude_line)
    assert field.value > 0  # unclaimed there, not refused


@pytest.mark.usefixtures("reaches_solve")
def test_a_deep_metal_law_is_admitted(drude_line):
    _admitted(drude_line, gfem.Driven([3e9]))


@pytest.mark.usefixtures("reaches_solve")
def test_a_law_in_the_plasmonic_window_admits_the_field_estimate(drude_line):
    _admitted(drude_line, gfem.Driven([8e9]), gfem.FieldEstimator())


# --- electrostatic sheets ----------------------------------------------------


def test_an_interior_sheet_terminal_does_not_smear_the_estimate(tmp_path):
    # A sheet terminal between two layers, ground above and below: V is
    # linear in each layer, so the solve is exact and its flux jumps across
    # the sheet. Recovered across the sheet, the flux would be smeared.
    layers = (("a", 0.5), ("b", 1.0))
    path = sm.write_msh22(tmp_path / "plate.msh", sm.plate(3.0, 2.0, layers, (2, 2, 2)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"a": gfem.Material(eps_r=2.0), "b": gfem.Material(eps_r=5.0)},
    )
    problem = gfem.Electrostatic(terminals={"sheet": "a__b"}, ground=["bottom", "top"])
    result = gfem.solve(model, problem, options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    estimator = gfem.QuantityEstimator(capacitance=("sheet", "sheet"))
    quantity = gfem.error_estimate(result, estimator, model=model)
    assert field.value < 1e-6
    assert quantity.value < 1e-12


# --- Q -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"q": (0, 1)}, TypeError),
        ({"q": True}, TypeError),
        ({"q": 0, "omega": 0}, TypeError),
        ({"q": -1}, ValueError),
    ],
    ids=["cluster", "bool", "two-keywords", "negative"],
)
def test_q_takes_one_mode_index(kwargs, error):
    with pytest.raises(error) as caught:
        gfem.QuantityEstimator(**kwargs)
    if kwargs == {"q": (0, 1)}:
        assert "cluster" in str(caught.value)
    assert gfem.QuantityEstimator(q=np.int64(1)).q == 1


@pytest.mark.usefixtures("no_solve")
def test_q_names_a_mode_of_an_eigenmode_problem(coax, cavity):
    with pytest.raises(TypeError, match="Eigenmode"):
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, estimator=gfem.QuantityEstimator(q=0))
    with pytest.raises(ValueError, match=r"q=1.*n_modes=1"):
        gfem.refine_mesh(
            cavity, rc.CAVITY_PROBLEM, estimator=gfem.QuantityEstimator(q=1)
        )


def _resistive_line(directory, n):
    """The 10 mm parallel-plate line, eps_r 4, closed by 50 and 30 ohm ports."""
    path = sm.write_msh22(
        directory / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=50.0),
            gfem.LumpedPort("p2", "port2", "+z", resistance=30.0),
        ],
    )


def test_q_estimate_of_the_resistive_line(tmp_path):
    # Its value and indicators as measured by an independent
    # implementation of the Q estimate (line, n = 4, first resonance, Q 1.78).
    model = _resistive_line(tmp_path, 4)
    problem = gfem.Eigenmode(target=7.5e9, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(q=0), model=model)
    assert estimate.kind == "q"
    assert estimate.value == pytest.approx(0.0010167422563164418, rel=1e-8)
    assert np.sum(estimate.indicators**2) == pytest.approx(
        0.0007770395510489388, rel=1e-8
    )


def test_q_of_a_lossless_mode_raises(cavity):
    result = gfem.solve(cavity, rc.CAVITY_PROBLEM, options=OPTIONS)
    with pytest.raises(ValueError, match=r"lossless.*omega=0"):
        gfem.error_estimate(result, gfem.QuantityEstimator(q=0), model=cavity)


def _lossy_cavity(cavity):
    return _with(cavity, materials={"air": gfem.Material(eps_r=4.0, conductivity=5.0)})


#: The lossy cavity's first mode, Hz.
LOSSY_CAVITY_F = rc.CAVITY_F1 / 2


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_q_cycles_record_the_followed_mode_q(cavity):
    ref = gfem.refine_mesh(
        _lossy_cavity(cavity),
        gfem.Eigenmode(target=LOSSY_CAVITY_F, n_modes=1),
        estimator=gfem.QuantityEstimator(q=0),
        tolerance=1e-5,
        max_cycles=1,
        options=OPTIONS,
    )
    assert ref.estimate.kind == "q"
    assert len(ref.cycles) == 2
    assert ref.cycles[-1].value == float(ref.result.q[0])
    assert 3.2 < ref.cycles[0].value < 3.4  # Q 3.30 (sigma 5 S/m, eps_r 4)


def test_q_tolerance_below_its_floor_raises(cavity):
    # Q's relative error is cot(arg omega) times omega's (about 2Q): the
    # floor is 10 x eigen_tolerance x cot(arg omega_h), here about 6.3e-7.
    options = dataclasses.replace(OPTIONS, eigen_tolerance=1e-8)
    with pytest.raises(ValueError, match=r"eigen_tolerance") as caught:
        gfem.refine_mesh(
            _lossy_cavity(cavity),
            gfem.Eigenmode(target=LOSSY_CAVITY_F, n_modes=1),
            estimator=gfem.QuantityEstimator(q=0),
            tolerance=5e-7,
            max_cycles=0,
            options=options,
        )
    assert "cot" in str(caught.value)


@pytest.mark.parametrize(
    ("material", "boundary", "feature", "cause"),
    [
        (None, gfem.Absorbing("wall", order=2), "lossy eigenmode", "second-order"),
        (gfem.MaterialField(mu_r=1 - 0.01j), None, "lossy eigenmode", "complex mu_r"),
        (LOSSY_TENSOR, None, "lossy eigenmode", "anisotropic"),
        (
            gfem.Material(
                eps_r=gfem.PermittivityLaw(
                    eps_inf=2.0,
                    terms=[gfem.Debye(delta_eps=1.0, relaxation_frequency=1e11)],
                )
            ),
            None,
            "dispersive eigenmode",
            "permittivity law",
        ),
    ],
    ids=["second-order-absorbing", "lossy-mu", "anisotropic-loss", "law"],
)
@pytest.mark.usefixtures("no_solve")
def test_q_keeps_the_lossy_eigenmode_refusals(
    cavity, material, boundary, feature, cause
):
    # the omega estimate alone admits these: Q has no case for them.
    changes = {}
    if material is not None:
        changes["materials"] = {"air": material}
    if boundary is not None:
        changes["boundaries"] = [boundary]
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            _with(cavity, **changes),
            rc.CAVITY_PROBLEM,
            estimator=gfem.QuantityEstimator(q=0),
            options=OPTIONS,
        )
    assert caught.value.feature == feature
    assert cause in str(caught.value)
    assert "q=" in str(caught.value)


# --- several quantities --------------------------------------------------------

S11 = gfem.QuantityEstimator(s=("p1", "p1"))


@pytest.mark.parametrize(
    ("estimator", "tolerance", "error", "match"),
    [
        ((), None, ValueError, "at least one"),
        ((S21, S21), None, ValueError, "twice"),
        ((S21, gfem.FieldEstimator()), None, TypeError, "FieldEstimator"),
        ([S21, S11], None, TypeError, "tuple"),
        ((S21, S11), (1e-3,), ValueError, "2 estimators"),
        ((S21, S11), (1e-3, 1e-3, 1e-3), ValueError, "2 estimators"),
        (S21, (1e-3,), TypeError, "tuple of estimators"),
        ((S21, S11), (1e-3, "1e-3"), TypeError, "number"),
        ((S21, S11), (1e-3, 1e-9), ValueError, "at least 10"),
    ],
    ids=[
        "empty",
        "duplicate",
        "field-estimator",
        "list",
        "short",
        "long",
        "tuple-tolerance-single",
        "string",
        "floor",
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_a_bad_tuple_of_estimators_raises_before_any_solve(
    step, estimator, tolerance, error, match
):
    with pytest.raises(error, match=match):
        gfem.refine_mesh(
            step,
            rc.step_problem(),
            estimator=estimator,
            tolerance=tolerance,
            options=OPTIONS,
        )


@pytest.mark.usefixtures("no_solve")
def test_each_estimator_of_a_tuple_is_checked(step):
    with pytest.raises(KeyError, match="p9"):
        gfem.refine_mesh(
            step,
            rc.step_problem(),
            estimator=(S21, gfem.QuantityEstimator(s=("p9", "p1"))),
            options=OPTIONS,
        )
    with pytest.raises(TypeError, match="Eigenmode"):
        gfem.refine_mesh(
            step,
            rc.step_problem(),
            estimator=(S21, gfem.QuantityEstimator(omega=0)),
            options=OPTIONS,
        )


def test_error_estimate_takes_one_estimator(step):
    with pytest.raises(TypeError, match="one estimator"):
        gfem.error_estimate(object(), (S21, S11), model=step)


def _refine_step(step, estimator, tolerance, max_cycles, problem=None):
    return gfem.refine_mesh(
        step,
        problem or rc.step_problem([3e9]),
        estimator=estimator,
        tolerance=tolerance,
        max_cycles=max_cycles,
        options=OPTIONS,
    )


def test_a_tuple_of_estimators_records_tuples(step):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning) as caught:
        ref = _refine_step(step, (S21, S11), (1e-5, 2e-5), 1)
    message = str(caught[0].message)
    assert "s=('p2', 'p1')" in message
    assert "s=('p1', 'p1')" in message
    assert "tolerance=2e-05" in message
    assert not ref.converged
    assert ref.stop == "max_cycles"
    first, second = ref.estimate
    assert (first.estimator, second.estimator) == (S21, S11)
    assert ref.cycles[-1].estimate == (first.value, second.value)
    for cycle in ref.cycles:
        assert len(cycle.estimate) == 2
        s21, s11 = cycle.value
        assert len(s21) == len(s11) == 1
        assert cycle.modes == {}
    assert ref.cycles[-1].value == (
        tuple(complex(v) for v in ref.result.s["p2", "p1"]),
        tuple(complex(v) for v in ref.result.s["p1", "p1"]),
    )


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_a_one_tuple_refines_as_its_estimator(step):
    one = _refine_step(step, (S21,), (1e-5,), 1)
    alone = _refine_step(step, S21, 1e-5, 1)
    assert [c.tetrahedra for c in one.cycles] == [c.tetrahedra for c in alone.cycles]
    for joint, single in zip(one.cycles, alone.cycles, strict=True):
        assert joint.estimate == pytest.approx((single.estimate,), rel=1e-10)
    (estimate,) = one.estimate
    assert estimate.values == pytest.approx(alone.estimate.values, rel=1e-12)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_a_quantity_at_its_tolerance_leaves_the_marking(step):
    # S11 meets its loose tolerance on the first mesh: the marking is S21's.
    joint = _refine_step(step, (S21, S11), (1e-5, 1.0), 1)
    alone = _refine_step(step, S21, 1e-5, 1)
    assert [c.marked for c in joint.cycles] == [c.marked for c in alone.cycles]
    assert [c.tetrahedra for c in joint.cycles] == [c.tetrahedra for c in alone.cycles]


def test_the_loop_stops_when_every_quantity_meets_its_tolerance(step, recwarn):
    ref = _refine_step(step, (S21, S11), 1.0, 3)
    assert ref.stop == "tolerance"
    assert ref.converged
    assert len(ref.cycles) == 1
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_a_tuple_takes_the_union_of_the_adjoints(step, factorizations):
    # S21 with p2 unexcited solves p2's adjoint with the forward; S11 needs
    # none. Together they factorize as S21 alone does: the adjoint rides once.
    problem = gfem.Driven([3e9], excite=["p1"])
    joint = _refine_step(step, (S21, S11), 1.0, 0, problem)
    together = factorizations()
    gfem.release_memory()  # no handle carries over to the next run
    single = []
    for estimator in (S21, S11):
        before = factorizations()
        single.append(_refine_step(step, estimator, 1.0, 0, problem))
        single_count = factorizations() - before
        if estimator is S21:
            assert single_count == together
        gfem.release_memory()
    for estimate, alone in zip(joint.estimate, single, strict=True):
        assert estimate.values == pytest.approx(alone.estimate.values, rel=1e-12)


@pytest.mark.usefixtures("no_solve")
def test_frequency_workers_refuse_a_tuple_with_a_solved_dual(step):
    options = dataclasses.replace(OPTIONS, frequency_workers=2)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            step,
            gfem.Driven([3e9, 8e9], excite=["p1"]),
            estimator=(S11, S21),
            options=options,
        )
    assert caught.value.feature == "frequency workers"
    assert "s=('p2', 'p1')" in str(caught.value)


# --- mode tracks ---------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_estimators_of_one_mode_share_its_track(cavity):
    ref = gfem.refine_mesh(
        _lossy_cavity(cavity),
        gfem.Eigenmode(target=LOSSY_CAVITY_F, n_modes=2),
        estimator=(gfem.QuantityEstimator(omega=0), gfem.QuantityEstimator(q=0)),
        tolerance=1e-5,
        max_cycles=1,
        options=OPTIONS,
    )
    for cycle in ref.cycles:
        assert dict(cycle.modes) == {0: 0}
        omega, q = cycle.value
        assert q == pytest.approx(abs(omega) / (2 * omega.imag), rel=1e-9)
    assert [e.kind for e in ref.estimate] == ["omega", "q"]


def test_two_tracks_landing_on_one_mode_raise(cavity, monkeypatch):
    # The second mesh's modes are made to sit nearest one mode for both
    # tracks: 0.5 % from the first mesh's two, the other far.
    from gfem.solve import _solve

    calls = []

    def solve(*args, **kwargs):
        result, adjoints = _solve(*args, **kwargs)
        w = complex(np.asarray(result.omega)[0])
        omegas = [w, 1.01 * w] if not calls else [1.005 * w, 3 * w]
        calls.append(1)
        return dataclasses.replace(result, omega=jnp.asarray(omegas)), adjoints

    monkeypatch.setattr("gfem.refinement._solve", solve)
    with pytest.raises(gfem.DegenerateModeError, match=r"named 0 and 1.*mode 0"):
        gfem.refine_mesh(
            cavity,
            gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2),
            estimator=(
                gfem.QuantityEstimator(omega=0),
                gfem.QuantityEstimator(omega=1),
            ),
            tolerance=1e-5,
            max_cycles=1,
            options=OPTIONS,
        )
    assert len(calls) == 2


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_each_track_records_its_mode(cavity):
    problem = gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=3)
    ref = gfem.refine_mesh(
        cavity,
        problem,
        estimator=(gfem.QuantityEstimator(omega=2), gfem.QuantityEstimator(omega=0)),
        tolerance=1e-9,
        max_cycles=1,
        options=OPTIONS,
    )
    assert dict(ref.cycles[0].modes) == {0: 0, 2: 2}
    last = ref.cycles[-1]
    omegas = np.asarray(ref.result.omega)
    assert last.value == (
        complex(omegas[last.modes[2]]),
        complex(omegas[last.modes[0]]),
    )


@pytest.mark.usefixtures("no_solve")
def test_a_tuple_keeps_each_estimators_driven_refusal(step):
    model = _with(step, boundaries=[*step.boundaries, gfem.Absorbing("side")])
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model, gfem.Driven([5e9]), estimator=(S11, S21), options=OPTIONS
        )
    assert caught.value.feature == "impedance or absorbing surface"
