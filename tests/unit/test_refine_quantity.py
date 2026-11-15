"""`gfem.QuantityEstimator`: key checks, refusals, mode tracking, sheets.

The effectivity of the estimate is verified in
`tests/verification/test_quantity_estimate.py`; these cases check what is
asked of it before any solve, and how it reads a result.
"""

import dataclasses
import itertools
import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
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
    system, ports = rule._ports(model, warn=False)
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
    # A slab's hybrid lossy port mode is not parallel to its excitation: the
    # S entry needs port p2's adjoint. refine_mesh solves it as one more
    # right-hand side of its own solve (one factorization); error_estimate
    # on the finished result factorizes once more, for the same estimate.
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05, offset=1e-3)
    problem = gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY])
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    refined = gfem.refine_mesh(
        model, problem, estimator=estimator, max_cycles=0, options=OPTIONS
    )
    assert len(prepares) == 1
    again = gfem.error_estimate(refined.result, estimator, model=refined.model)
    assert len(prepares) == 2
    assert again.value == pytest.approx(refined.estimate.value, rel=1e-8)
    _, own = _forced_adjoint(model, refined.result, "p2")
    assert own is None
    free = kernels.s_entry(model, refined.result, "p2", "p1")[0].value
    assert abs(free / refined.estimate.value - 1) > 1e-6


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
            gfem.Driven(frequencies),
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

    rule = DrivenRule(model, problem, OPTIONS)
    rule.adjoint_loads = port_loads(("p2",))
    return rule


def test_adjoint_ports_never_reach_a_differentiated_solve(tmp_path):
    # A backward pass keeps no adjoint columns: the forward refuses them.
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    rule = _adjoint_rule(model, gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY]))
    with pytest.raises(RuntimeError, match="differentiated"):
        rule.forward(model, keep=True)


def test_adjoint_ports_never_reach_an_adaptive_sweep(tmp_path):
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    f = rc.LOSSY_GUIDE_FREQUENCY
    problem = gfem.Driven(np.linspace(f, 1.1 * f, 5), sweep="adaptive")
    rule = _adjoint_rule(model, problem)
    with pytest.raises(RuntimeError, match="adaptive"):
        rule.forward(model, keep=False)


@pytest.mark.usefixtures("no_solve")
def test_frequency_workers_refuse_an_s_estimate_on_a_lossy_wave_port(tmp_path):
    # Frequency workers take wave ports (ADR 0099) but solve no adjoint:
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
    assert omegas[0] == complex(first[1])
    for before, after in itertools.pairwise(omegas):
        assert abs(after - before) < abs(first[1] - first[0]) / 2


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


@pytest.mark.parametrize(
    ("material", "boundary", "cause"),
    [
        (None, gfem.Absorbing("wall", order=2), "second-order absorbing"),
        (gfem.MaterialField(mu_r=1 - 0.01j), None, "complex mu_r"),
    ],
    ids=["second-order-absorbing", "lossy-mu"],
)
@pytest.mark.usefixtures("no_solve")
def test_the_lossy_eigenmode_refusal_names_its_cause(cavity, material, boundary, cause):
    changes = {}
    if material is not None:
        changes["materials"] = {"air": material}
    if boundary is not None:
        changes["boundaries"] = [boundary]
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            _with(cavity, **changes),
            rc.CAVITY_PROBLEM,
            estimator=gfem.QuantityEstimator(omega=0),
        )
    message = str(caught.value)
    assert caught.value.feature == "lossy eigenmode"
    assert cause in message
    assert "wave port" not in message


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
@pytest.mark.usefixtures("no_solve")
def test_loss_in_an_anisotropic_material_is_one_refusal(
    request, fixture, problem, estimator
):
    # No estimator has a case for the complex tensor eps_eff: S, omega and
    # the field refuse it under one feature name (spec 5.5).
    model = request.getfixturevalue(fixture)
    lossy = _with(model, materials=dict.fromkeys(model.materials, LOSSY_TENSOR))
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(lossy, problem, estimator=estimator, options=OPTIONS)
    assert caught.value.feature == "lossy anisotropic material"
    assert "loss in the anisotropic material of" in str(caught.value)
    assert "lossless" in str(caught.value)  # the remedy


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
