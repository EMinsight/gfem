"""`gfem.QuantityEstimator(objective=J)`: classification, raises and checks.

An objective ``J(result, model)`` is classified by tracing it (ADR 0143 and
its amendment): its estimate kind, refusals and mixtures from what it
structurally reads, its read entries from NaN reach and nonzero cotangents.
The effectivity of the objective estimates is verified in
`tests/verification/test_objective_estimate.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc

import gfem
from gfem import _objective

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
STEP_F = (3e9, 8e9)


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    return rc.step_model(tmp_path_factory.mktemp("step"), 2)


@pytest.fixture(scope="module")
def step_result(step):
    problem = gfem.Driven(list(STEP_F), keep_fields=True)
    return gfem.solve(step, problem, options=OPTIONS)


def _s21_band(result, _model):
    return jnp.mean(1 - jnp.abs(result.s["p2", "p1"]) ** 2)


def _reads(objective):
    return {(r.leaf, r.index): r for r in objective.reads}


# --- classification ------------------------------------------------------------------


def test_a_band_mean_of_s21_reads_s21_at_each_frequency(step, step_result):
    objective = _objective.classify(_s21_band, step_result, step)
    assert objective.kind == "objective of s"
    s21 = np.asarray(step_result.s["p2", "p1"])
    assert objective.value == pytest.approx(float(np.mean(1 - np.abs(s21) ** 2)))
    reads = _reads(objective)
    assert set(reads) == {(".s.values", (0, 1, 0)), (".s.values", (1, 1, 0))}
    # gfem's cotangent contract: dJ/dx - i dJ/dy, here -2 conj(S21) / 2
    for f in range(2):
        assert reads[".s.values", (f, 1, 0)].cotangent == pytest.approx(
            -np.conj(s21[f])
        )


def test_an_objective_at_its_target_keeps_its_read_with_zero_cotangent(
    step, step_result
):
    target = float(np.abs(np.asarray(step_result.s["p2", "p1"])[0]) ** 2)

    def at_target(result, _model):
        return (jnp.abs(result.s["p2", "p1"][0]) ** 2 - target) ** 2

    objective = _objective.classify(at_target, step_result, step)
    (read,) = objective.reads
    assert (read.leaf, read.index, read.cotangent) == (".s.values", (0, 1, 0), 0)
    # J = (|S|^2 - t)^2: its Hessian in (Re, Im) S is 8 S S^T at the target
    s = complex(np.asarray(step_result.s["p2", "p1"])[0])
    assert objective.curvature[0, 0] == pytest.approx(8 * abs(s) ** 2)


def test_a_stop_gradient_read_has_the_cotangent_of_the_value(step, step_result):
    def weighted(result, _model):
        s11 = jax.lax.stop_gradient(jnp.abs(result.s["p1", "p1"][0]) ** 2)
        return jnp.abs(result.s["p2", "p1"][0]) ** 2 * s11

    reads = _reads(_objective.classify(weighted, step_result, step))
    s21 = complex(np.asarray(step_result.s["p2", "p1"])[0])
    s11 = complex(np.asarray(step_result.s["p1", "p1"])[0])
    expected = 2 * np.conj(s11) * abs(s21) ** 2
    assert reads[".s.values", (0, 0, 0)].cotangent == pytest.approx(expected)


def test_an_entry_read_with_a_zero_weight_is_read(step, step_result):
    def zero_weight(result, _model):
        return jnp.abs(result.s["p2", "p1"][1]) + 0 * result.s["p1", "p1"][1].real

    reads = _reads(_objective.classify(zero_weight, step_result, step))
    assert set(reads) == {(".s.values", (1, 1, 0)), (".s.values", (1, 0, 0))}
    assert reads[".s.values", (1, 0, 0)].cotangent == 0


def test_db_reads_its_entry(step, step_result):
    def mean_db(result, _model):
        return jnp.mean(result.s.db("p2", "p1"))

    objective = _objective.classify(mean_db, step_result, step)
    assert objective.kind == "objective of s"
    assert {r.index for r in objective.reads} == {(0, 1, 0), (1, 1, 0)}


def test_y_reads_the_entries_of_its_frequency(step, step_result):
    def admittance(result, _model):
        return result.s.y()[1, 0, 1].real

    objective = _objective.classify(admittance, step_result, step)
    assert objective.kind == "objective of s"
    assert {r.index for r in objective.reads} <= {
        (1, i, j) for i in range(2) for j in range(2)
    }
    assert {r.index[0] for r in objective.reads} == {1}


# --- refusals of the classification --------------------------------------------------


def _raises(feature, objective, result, model):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        _objective.classify(objective, result, model)
    assert caught.value.feature == feature
    return caught.value


def _robust_max(result, _model):
    return jnp.max(1 - jnp.abs(result.s["p2", "p1"]) ** 2)


def _where(result, _model):
    s21 = result.s["p2", "p1"][0]
    return jnp.where(jnp.abs(s21) > 0.5, jnp.abs(s21) ** 2, 0.0)


def _cond(result, _model):
    s21 = result.s["p2", "p1"][0]
    return jax.lax.cond(jnp.abs(s21) > 0.5, lambda x: jnp.abs(x) ** 2, jnp.abs, s21)


@jax.custom_jvp
def _square(x):
    return x**2


@_square.defjvp
def _square_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return x**2, 2 * x * t


def _custom(result, _model):
    return _square(jnp.abs(result.s["p2", "p1"][0]))


def _point(result, _model):
    e = result.e_field(STEP_F[0], "p1").at(np.array([5e-3, 1e-3, 0.5e-3]))
    return jnp.sum(jnp.abs(e) ** 2)


def _coefficients(result, _model):
    return jnp.sum(jnp.abs(result.e_fields[0].coefficients) ** 2)


def _predicate_side(result, model):
    field = result.e_field(STEP_F[0], "p1")
    return gfem.power_flow(field, lambda c: c[:, 0] < 5e-3, model=model)


def _port_side(result, model):
    return gfem.power_flow(result.e_field(STEP_F[0], "p1"), "dielectric", model=model)


def _mixed(result, model):
    return 1 - jnp.abs(result.s["p1", "p1"][0]) ** 2 + _port_side(result, model)


def _energy(result, model):
    return gfem.electric_energy(result.e_field(STEP_F[0], "p1"), model=model)[
        "dielectric"
    ]


@pytest.mark.parametrize(
    ("objective", "feature"),
    [
        (_robust_max, "non-smooth objective"),
        (_where, "non-smooth objective"),
        (_cond, "non-smooth objective"),
        (_custom, "custom derivative in objective"),
        (_point, "point value"),
        (_coefficients, "field coefficients"),
        (_predicate_side, "power_flow side predicate"),
        (_port_side, "power through a port"),
        (_energy, "electric_energy"),
    ],
)
def test_an_objective_reading_what_has_no_estimate_raises(
    step, step_result, objective, feature
):
    _raises(feature, objective, step_result, step)


def test_a_field_from_outside_the_result_raises(step, step_result):
    foreign = step_result.e_field(STEP_F[0], "p1")

    def closure(_result, model):
        return gfem.power_flow(foreign, "dielectric", model=model)

    _raises("field coefficients", closure, step_result, step)


def test_a_mixture_of_s_and_measured_power_raises(step, step_result, monkeypatch):
    # with the port check out of the way, S plus a power is a mixture
    monkeypatch.setattr("gfem.measurements._check_estimated_side", lambda *_: None)
    error = _raises("mixed objective", _mixed, step_result, step)
    assert "objective of absorbed power" in str(error)
    assert "objective of s" in str(error)


def test_an_undefined_s_row_raises(tmp_path):
    model = rc.step_model(tmp_path, 2)
    model = gfem.Model(
        model.mesh,
        materials=model.materials,
        boundaries=model.boundaries,
        ports=[model.ports[0], gfem.LumpedPort("p2", "port2", "+z", inductance=1e-9)],
    )
    problem = gfem.Driven([3e9], excite=["p1"], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)

    def s21(result, _model):
        return jnp.abs(result.s["p2", "p1"][0]) ** 2

    _raises("undefined S row", s21, result, model)


def test_a_port_mode_output_raises_and_with_offsets_reads_s(tmp_path):
    model = rc.step_model(tmp_path, 2)
    model = gfem.Model(
        model.mesh,
        materials=model.materials,
        boundaries=model.boundaries,
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    result = gfem.solve(model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS)

    def gamma(result, _model):
        return jnp.abs(result.port_modes["p1"].gamma[0])

    def shifted(result, _model):
        moved = result.with_offsets({"p1": 1e-3})
        return jnp.angle(moved.s["p2", "p1"][0])

    _raises("port mode output", gamma, result, model)
    objective = _objective.classify(shifted, result, model)
    assert objective.kind == "objective of s"
    assert [r.index for r in objective.reads] == [(0, 1, 0)]


# --- the objective's type and value --------------------------------------------------


def test_a_complex_objective_raises_type_error(step, step_result):
    with pytest.raises(TypeError, match="one real number"):
        _objective.classify(lambda r, _: r.s["p2", "p1"][0] * 1, step_result, step)


def test_an_objective_of_several_numbers_raises_type_error(step, step_result):
    with pytest.raises(TypeError, match="one real number"):
        _objective.classify(lambda r, _: jnp.abs(r.s["p2", "p1"]), step_result, step)


def test_an_objective_reading_nothing_raises_value_error(step, step_result):
    with pytest.raises(ValueError, match="reads nothing gfem solved"):
        _objective.classify(lambda _r, _m: jnp.asarray(1.0), step_result, step)


def test_a_non_finite_objective_raises_value_error(step, step_result):
    def infinite(result, _model):
        return jnp.abs(result.s["p2", "p1"][0]) / 0.0

    with pytest.raises(ValueError, match="not a finite number"):
        _objective.classify(infinite, step_result, step)


def test_a_python_branch_on_a_value_raises_type_error(step, step_result):
    def branch(result, _model):
        s21 = jnp.abs(result.s["p2", "p1"][0])
        return s21 if s21 > 0.5 else 1 - s21

    with pytest.raises(TypeError, match="no Python branch"):
        _objective.classify(branch, step_result, step)


def test_an_objective_under_an_outer_grad_raises(step, step_result):
    def outer(p):
        def objective(result, _model):
            return p * jnp.abs(result.s["p2", "p1"][0]) ** 2

        return _objective.classify(objective, step_result, step).value

    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(outer)(1.0)


# --- the classification table (JAX pin) ----------------------------------------------


@pytest.fixture(scope="module")
def bars(tmp_path_factory):
    model = rc.bars_model(tmp_path_factory.mktemp("bars"), "near", 2)
    return model, gfem.solve(model, rc.BARS_PROBLEM, options=OPTIONS)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    model = rc.cavity_model(tmp_path_factory.mktemp("cavity"), 2)
    problem = gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=3)
    return model, gfem.solve(model, problem, options=OPTIONS)


#: (case, objective, kind, read (leaf, index)s): the classification the
#: estimators rely on, at the pinned JAX. A JAX upgrade that changes how
#: jax.numpy lowers these (and so what dead-code elimination, the NaN reach
#: or the cotangents see) fails here first.
CLASSIFICATION_TABLE = [
    (
        "step",
        lambda r, _: jnp.abs(r.s["p2", "p1"][1]) ** 2,
        "objective of s",
        {(".s.values", (1, 1, 0))},
    ),
    (
        "step",
        lambda r, _: jnp.mean(jnp.abs(r.s.values[:, :, 0]) ** 2),
        "objective of s",
        {(".s.values", (f, i, 0)) for f in range(2) for i in range(2)},
    ),
    (
        "bars",
        lambda r, _: r.mutual[0, 1],
        "objective of capacitance",
        {(".capacitance", (0, 1))},
    ),
    (
        "bars",
        lambda r, _: r.capacitance[1, 1],
        "objective of capacitance",
        {(".capacitance", (1, 1))},
    ),
    (
        "bars",
        lambda r, _: r.energy["dielectric"][0],
        "objective of electrostatic energy",
        {(".energy['dielectric']", (0,))},
    ),
    (
        "cavity",
        lambda r, _: r.frequency[0] * 1e-9,
        "objective of omega",
        {(".omega", (0,))},
    ),
    (
        "cavity",
        lambda r, _: jnp.mean(r.omega[1:].real) * 1e-9,
        "objective of omega",
        {(".omega", (1,)), (".omega", (2,))},
    ),
]


@pytest.mark.parametrize(
    ("case", "objective", "kind", "reads"),
    CLASSIFICATION_TABLE,
    ids=[f"{c}-{k}-{i}" for i, (c, _, k, _) in enumerate(CLASSIFICATION_TABLE)],
)
def test_the_classification_table(request, case, objective, kind, reads):
    if case == "step":
        model, result = (request.getfixturevalue(n) for n in ("step", "step_result"))
    else:
        model, result = request.getfixturevalue(case)
    classified = _objective.classify(objective, result, model)
    assert classified.kind == kind
    assert set(_reads(classified)) == reads


def test_the_mode_aligned_view_puts_each_followed_mode_at_its_first_index(cavity):
    _, result = cavity
    omega = np.asarray(result.omega)
    view = _objective.aligned(result, {0: 2})
    np.testing.assert_array_equal(np.asarray(view.omega), omega[[2, 0, 1]])
    assert view.e_fields[0] is result.e_fields[2]
    assert _objective.aligned(result, {0: 0, 1: 1}) is result


# --- the estimator, refine_mesh and error_estimate -----------------------------------


@pytest.fixture
def no_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement._solve", solve)


def test_an_objective_is_one_callable_keyword():
    with pytest.raises(TypeError, match="takes a function"):
        gfem.QuantityEstimator(objective=1.0)
    with pytest.raises(TypeError, match="exactly one of"):
        gfem.QuantityEstimator(objective=_s21_band, s=("p2", "p1"))


def test_error_estimate_of_an_objective_of_s(step, step_result):
    estimator = gfem.QuantityEstimator(objective=_s21_band)
    estimate = gfem.error_estimate(step_result, estimator, model=step)
    # the step line is lossless with lumped ports: its rewrite, never above C
    assert estimate.kind == "objective of s, lossless rewrite"
    assert len(estimate.values) == 1
    assert estimate.value > 0
    assert estimate.indicators.shape == (len(step.mesh.tetrahedra),)
    # its value is the combined dual's: never above the chain-rule sum of the
    # entries' own estimates (ADR 0129), plus its remainder
    s21 = np.abs(np.asarray(step_result.s["p2", "p1"]))
    entry = gfem.error_estimate(
        step_result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    chain = sum(a * eta for a, eta in zip(s21, entry.values, strict=True))
    remainder = 0.5 * sum(eta**2 for eta in entry.values)
    assert estimate.value <= chain + remainder * (1 + 1e-9)


def test_an_objective_of_another_kind_raises_by_its_kind(bars):
    model, result = bars
    estimator = gfem.QuantityEstimator(objective=lambda r, _: r.mutual[0, 1])
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, estimator, model=model)
    assert caught.value.feature == "objective of capacitance"


def test_an_unexcited_read_row_gives_the_all_excited_estimate(step):
    # S11 rides on its free dual and S21 on p2's solved adjoint when p2 is
    # not excited: their combined dual must equal the all-excited one, whose
    # duals are both free (the free dual's phase, -i / (4 k0 eta0), matters)
    def objective(result, _model):
        return jnp.mean(
            jnp.abs(result.s["p2", "p1"]) ** 2
            - 0.3 * jnp.abs(result.s["p1", "p1"]) ** 2
        )

    estimator = gfem.QuantityEstimator(objective=objective)
    values = []
    for excite in (None, ["p1"]):
        problem = gfem.Driven(list(STEP_F), excite=excite, keep_fields=True)
        result = gfem.solve(step, problem, options=OPTIONS)
        values.append(gfem.error_estimate(result, estimator, model=step).value)
    assert values[1] == pytest.approx(values[0], rel=1e-10)


@pytest.mark.usefixtures("no_solve")
def test_an_objective_needs_its_tolerance(step):
    estimator = gfem.QuantityEstimator(objective=_s21_band)
    problem = gfem.Driven(list(STEP_F))
    with pytest.raises(TypeError, match=r"pass refine_mesh\(tolerance=\.\.\.\)"):
        gfem.refine_mesh(step, problem, estimator=estimator)
    with pytest.raises(TypeError, match=r"pass refine_mesh\(tolerance=\.\.\.\)"):
        gfem.refine_mesh(
            step, problem, estimator=(estimator, gfem.QuantityEstimator(s=("p2", "p1")))
        )


@pytest.mark.usefixtures("no_solve")
def test_an_objective_on_an_impedance_surface_raises_before_any_solve(step):
    model = gfem.Model(
        step.mesh,
        materials=step.materials,
        boundaries=[*step.boundaries[:1], gfem.Impedance("top", resistance=100.0)],
        ports=step.ports,
    )
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model,
            gfem.Driven(list(STEP_F)),
            estimator=gfem.QuantityEstimator(objective=_s21_band),
            tolerance=1e-3,
        )
    assert caught.value.feature == "impedance or absorbing surface"
    assert "FieldEstimator()" in str(caught.value)


def test_an_objective_tolerance_below_its_sensitivity_floor_raises(step):
    # J = 1e6 |S21|^2: its sensitivity is about 2e6 |S21|, so 1e-3 lies below
    # 10 x the solve tolerance 1e-10 carried through it
    def scaled(result, _model):
        return 1e6 * jnp.abs(result.s["p2", "p1"][0]) ** 2

    with pytest.raises(ValueError, match="first-order sensitivity"):
        gfem.refine_mesh(
            step,
            gfem.Driven([STEP_F[0]]),
            estimator=gfem.QuantityEstimator(objective=scaled),
            tolerance=1e-3,
            options=gfem.SolverOptions(
                order=2, linear_solver="direct", tolerance=1e-10
            ),
            max_cycles=0,
        )


def test_an_objective_with_a_solved_dual_refuses_frequency_workers(step):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            step,
            gfem.Driven(list(STEP_F), excite=["p1"]),
            estimator=gfem.QuantityEstimator(objective=_s21_band),
            tolerance=1e-4,
            options=gfem.SolverOptions(
                order=2, linear_solver="direct", frequency_workers=2
            ),
            max_cycles=0,
        )
    assert caught.value.feature == "frequency workers"


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_objective_dual_factorizes_once_then_rides_on_the_solves(step, monkeypatch):
    from gfem import refinement

    calls = []
    adjoints = refinement._adjoints

    def counted(*args, **kwargs):
        calls.append(args[2])
        return adjoints(*args, **kwargs)

    monkeypatch.setattr(refinement, "_adjoints", counted)
    estimator = gfem.QuantityEstimator(objective=_s21_band)
    refinement_ = gfem.refine_mesh(
        step,
        gfem.Driven(list(STEP_F), excite=["p1"]),
        estimator=estimator,
        tolerance=2e-5,
        options=OPTIONS,
        max_cycles=2,
    )
    assert calls == [("p2",)]  # cycle 0 only
    assert len(refinement_.cycles) == 3
    for cycle in refinement_.cycles:
        assert isinstance(cycle.value, float)
    # the ridden dual gives the estimate error_estimate gives on the last mesh
    again = gfem.error_estimate(refinement_.result, estimator, model=refinement_.model)
    assert again.value == pytest.approx(refinement_.estimate.value, rel=1e-8)


# --- the lossless rewrite (ADR 0135) --------------------------------------------


def test_a_lossless_lumped_port_model_estimates_through_its_lossless_rewrite(
    step, step_result
):
    estimator = gfem.QuantityEstimator(objective=_s21_band)
    estimate = gfem.error_estimate(step_result, estimator, model=step)
    assert estimate.kind == "objective of s, lossless rewrite"
    assert estimate.indicators.shape == (len(step.mesh.tetrahedra),)
    # lam = 0 is a choice of the rewrite, so it is never above C (ADR 0129)
    _, _, c = _c_form(step, step_result, _s21_band)
    assert 0 < estimate.value <= c.value * (1 + 1e-12)


def _c_form(model, result, objective, **kwargs):
    from gfem._engine import estimate as kernels

    classified = _objective.classify(objective, result, model)
    s = result.s
    reads = [
        (float(s.frequency[f]), s.ports[i], s.excited[j], r.cotangent)
        for r in classified.reads
        for f, i, j in [r.index]
    ]
    return (
        classified,
        reads,
        kernels.objective_of_s(model, result, reads, classified.curvature, **kwargs),
    )


def test_the_lossless_rewrite_at_lam_zero_is_the_c_form(step, step_result):
    from gfem._engine import estimate as kernels

    def sheet(result, _model):
        s21, s11 = result.s["p2", "p1"], result.s["p1", "p1"]
        return jnp.mean((jnp.abs(s21) ** 2 - 0.5) ** 2 + jnp.abs(s11) ** 2)

    classified, reads, c = _c_form(step, step_result, sheet)
    columns = len({(f, source) for f, _, source, _ in reads})
    at_zero = kernels.lossless_rewrite(
        step,
        step_result,
        reads,
        classified.curvature,
        classified.diagonal,
        lam=np.zeros(columns),
    )
    assert at_zero.linear == pytest.approx(c.linear, rel=1e-12)
    assert at_zero.remainder == pytest.approx(c.remainder, rel=1e-12)
    np.testing.assert_allclose(at_zero.weights, c.weights, rtol=1e-12, atol=0)
    # the minimum is at most the value at lam = 0, and adds |lam . D|
    best = kernels.lossless_rewrite(
        step, step_result, reads, classified.curvature, classified.diagonal
    )
    assert best.value <= at_zero.value
    assert best.value == pytest.approx(
        best.linear + best.remainder + abs(best.lam @ best.defect), rel=1e-14
    )


def test_the_lossless_rewrite_moves_transmission_onto_reflection(step, step_result):
    # |S21|^2 at one frequency: on a lossless two-port its error follows
    # |S11|^2's, so the rewrite estimates it far below C
    def s21(result, _model):
        return jnp.abs(result.s["p2", "p1"][0]) ** 2

    estimate = gfem.error_estimate(
        step_result, gfem.QuantityEstimator(objective=s21), model=step
    )
    _, _, c = _c_form(step, step_result, s21)
    assert estimate.value < 0.8 * c.value


def _variant(model, **changes):
    fields = {
        "materials": model.materials,
        "boundaries": model.boundaries,
        "ports": model.ports,
    }
    return gfem.Model(model.mesh, **{**fields, **changes})


@pytest.mark.parametrize(
    "variant",
    [
        "lossy dielectric",
        "conductor plate",
        "wave ports",
    ],
)
def test_no_lossless_rewrite_where_power_may_not_be_conserved(variant, step, tmp_path):
    if variant == "wave ports":
        model = rc.lossy_guide_model(tmp_path, 2, "uniform", 0.0)
        frequencies = [rc.LOSSY_GUIDE_FREQUENCY]
    else:
        changes = {
            "lossy dielectric": {
                "materials": {"dielectric": gfem.Material(eps_r=4.0, tan_delta=0.1)}
            },
            "conductor plate": {
                "boundaries": [
                    gfem.PEC("bottom"),
                    gfem.Conductor("top", conductivity=5.8e7),
                ]
            },
        }[variant]
        model = _variant(step, **changes)
        frequencies = [STEP_F[0]]
    result = gfem.solve(
        model, gfem.Driven(frequencies, keep_fields=True), options=OPTIONS
    )
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=_s21_band), model=model
    )
    assert estimate.kind == "objective of s"
    _, _, c = _c_form(model, result, _s21_band)
    assert estimate.value == pytest.approx(c.value, rel=1e-12)
