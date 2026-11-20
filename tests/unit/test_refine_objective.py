"""`gfem.QuantityEstimator(objective=J)`: classification, raises and checks.

An objective ``J(result, model)`` is classified by tracing it: its
estimate kind, refusals and mixtures from what it
structurally reads, its read entries from NaN reach and nonzero cotangents.
The effectivity of the objective estimates is verified in
`tests/verification/test_objective_estimate.py`.
"""

import dataclasses

import contrast_cases as cc
import field_objective_cases as foc
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

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
    assert (read.leaf, read.index) == (".s.values", (0, 1, 0))
    # |S|^2 - t is zero up to the round-off of reading |S|^2 twice
    assert read.cotangent == pytest.approx(0, abs=1e-14)
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


def _scan_max(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"]) ** 2

    def body(c, x):
        return jnp.maximum(c, x), None

    return jax.lax.scan(body, 0.0, s21)[0]


def _while_on_read(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"][0])
    return jax.lax.while_loop(lambda x: x < 1.0, lambda x: 2 * x, s21)


def _cond_in_scan(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"]) ** 2

    def body(c, x):
        return c + jax.lax.cond(x > 0.5, lambda y: y, lambda y: 2 * y, x), None

    return jax.lax.scan(body, 0.0, s21)[0]


def _custom_in_scan(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"])
    return jax.lax.scan(lambda c, x: (c + _square(x), None), 0.0, s21)[0]


def _relu(result, _model):
    return jax.nn.relu(jnp.abs(result.s["p2", "p1"][0]) - 0.5)


@jax.custom_batching.custom_vmap
def _vmapped(x):
    return x**2


@_vmapped.def_vmap
def _vmapped_rule(_axis_size, in_batched, x):
    return x**2, in_batched[0]


def _unknown(result, _model):
    # custom_vmap_call: a higher-order primitive the classification does not know
    return _vmapped(jnp.abs(result.s["p2", "p1"][0]))


@pytest.mark.parametrize(
    ("objective", "feature"),
    [
        (_scan_max, "non-smooth objective"),
        (_while_on_read, "non-smooth objective"),
        (_cond_in_scan, "non-smooth objective"),
        (_custom_in_scan, "custom derivative in objective"),
        (_relu, "non-smooth objective"),
        (_unknown, "non-smooth objective"),
    ],
)
def test_an_objective_branching_inside_a_higher_order_primitive_raises(
    step, step_result, objective, feature
):
    _raises(feature, objective, step_result, step)


def test_a_stop_gradient_inside_a_scan_has_the_cotangent_of_the_value(
    step, step_result
):
    def scanned(result, _model):
        s21 = jnp.abs(result.s["p2", "p1"][0])
        body = lambda c, _: (jax.lax.stop_gradient(c) * c, None)  # noqa: E731
        return jax.lax.scan(body, s21, None, length=1)[0]

    reads = _reads(_objective.classify(scanned, step_result, step))
    s21 = complex(np.asarray(step_result.s["p2", "p1"][0]))
    # the cotangent of |s21|^2 (dJ/dx - i dJ/dy), not of |s21| sg(|s21|)
    assert reads[".s.values", (0, 1, 0)].cotangent == pytest.approx(2 * np.conj(s21))


def test_a_fixed_count_loop_over_a_read_is_admitted(step, step_result):
    def looped(result, _model):
        s21 = jnp.abs(result.s["p2", "p1"][0]) ** 2
        return jax.lax.fori_loop(0, 3, lambda _i, x: x + s21, 0.0)

    objective = _objective.classify(looped, step_result, step)
    assert objective.kind == "objective of s"


def _logsumexp(result, _model):
    return jax.scipy.special.logsumexp(jnp.abs(result.s["p2", "p1"]) ** 2)


def _softplus(result, _model):
    return jnp.sum(jax.nn.softplus(jnp.abs(result.s["p2", "p1"]) ** 2))


def _logaddexp(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"]) ** 2
    return jnp.logaddexp(s21[0], s21[1])


def _normalized_by_a_stopped_max(result, _model):
    s21 = jnp.abs(result.s["p2", "p1"]) ** 2
    return jnp.sum(s21) / jax.lax.stop_gradient(jnp.max(s21))


@pytest.mark.parametrize(
    "objective", [_logsumexp, _softplus, _logaddexp, _normalized_by_a_stopped_max]
)
def test_jax_smooth_rules_and_a_stopped_max_are_admitted(step, step_result, objective):
    classified = _objective.classify(objective, step_result, step)
    assert classified.kind == "objective of s"
    assert {(r.leaf, r.index) for r in classified.reads} == {
        (".s.values", (0, 1, 0)),
        (".s.values", (1, 1, 0)),
    }


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


def test_an_objective_beside_an_undefined_s_row_keeps_the_c_form(tmp_path):
    # Lossless, lumped ports only, but p2 has no resistance: its row of S is
    # undefined, so the lossless rewrite (over every row) does not apply.
    model = rc.step_model(tmp_path, 2)
    model = gfem.Model(
        model.mesh,
        materials=model.materials,
        boundaries=model.boundaries,
        ports=[model.ports[0], gfem.LumpedPort("p2", "port2", "+z", inductance=1e-9)],
    )
    problem = gfem.Driven([3e9], excite=["p1"], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)

    def s11(result, _model):
        return jnp.abs(result.s["p1", "p1"][0]) ** 2

    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=s11), model=model
    )
    assert estimate.kind == "objective of s"
    assert np.isfinite(estimate.value)
    assert estimate.value > 0


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
        "bars",
        lambda r, _: gfem.region_mean(r.potential["a"], "dielectric"),
        "objective of potential means",
        {("region_mean|.potential['a']|dielectric", (0,))},
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
    # entries' own estimates, plus its remainder
    s21 = np.abs(np.asarray(step_result.s["p2", "p1"]))
    entry = gfem.error_estimate(
        step_result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    chain = sum(a * eta for a, eta in zip(s21, entry.values, strict=True))
    remainder = 0.5 * sum(eta**2 for eta in entry.values)
    assert estimate.value <= chain + remainder * (1 + 1e-9)


# --- objectives of driven fields ---------------------------------------------------

LINE_F = foc.STEPPED_LINE_F


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return foc.stepped_line_model(tmp_path_factory.mktemp("line"), 2)


@pytest.fixture(scope="module")
def line_result(line):
    problem = gfem.Driven([LINE_F, 1.25 * LINE_F], keep_fields=True)
    return gfem.solve(line, problem, options=OPTIONS)


def _layer_power(result, model, f=LINE_F, port="p1"):
    return -gfem.power_flow(result.e_field(f, port), ["lsub", "lossy"], model=model)


def _box_mean(result, _model, f=LINE_F, port="p1"):
    return gfem.region_mean(result.e_field(f, port), "rbox")


def _with(model, **changes):
    fields = {
        "materials": model.materials,
        "boundaries": model.boundaries,
        "ports": model.ports,
    }
    return gfem.Model(model.mesh, **(fields | changes))


def test_the_absorbed_power_is_the_volume_form_of_power_flow(line, line_result):
    from gfem._engine import field_duals

    field = line_result.e_field(LINE_F, "p1")
    layout = (field.support, tuple(field._excluded), field.order)
    mass = field_duals.power_mass(line, layout, ("lsub", "lossy"), LINE_F)
    x = np.asarray(field.coefficients)
    volume = float(np.real(np.vdot(x, mass @ x)))
    assert volume > 0
    assert volume == pytest.approx(float(_layer_power(line_result, line)), rel=1e-9)


def test_a_read_measurement_runs_once_and_an_unread_one_never(
    line, line_result, monkeypatch
):
    from gfem import measurements

    calls = []
    real = measurements.region_mean

    def counted(field, regions):
        if _objective.recording() is None:
            calls.append(tuple(regions))
        return real(field, regions)

    monkeypatch.setattr(measurements, "region_mean", counted)

    def objective(result, _model):
        field = result.e_field(LINE_F, "p1")
        measurements.region_mean(field, "lsub")  # read by nothing
        return jnp.abs(measurements.region_mean(field, "rbox")[2]) ** 2

    classified = _objective.classify(objective, line_result, line)
    assert classified.kind == "objective of field means"
    assert calls == [("rbox",)]


def test_a_classification_again_is_the_same_classification(step, step_result):
    first = _objective.classify(_s21_band, step_result, step)
    again = _objective.classify(_s21_band, step_result, step)
    assert again.value == first.value
    assert again.reads == first.reads
    np.testing.assert_array_equal(again.curvature, first.curvature)
    for a, b in zip(again.diagonal, first.diagonal, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("objective", "kind"),
    [
        (_layer_power, "objective of absorbed power"),
        (lambda r, m: jnp.real(_box_mean(r, m)[2]), "objective of field means"),
        (
            lambda r, m: jnp.sum(jnp.abs(_box_mean(r, m)) ** 2),
            "objective of field means",
        ),
    ],
)
def test_error_estimate_of_an_objective_of_fields(line, line_result, objective, kind):
    estimate = gfem.error_estimate(
        line_result, gfem.QuantityEstimator(objective=objective), model=line
    )
    assert estimate.kind == kind
    assert estimate.value > 0
    assert estimate.indicators.shape == (len(line.mesh.tetrahedra),)
    assert np.sum(estimate.indicators**2) >= estimate.value * (1 - 1e-9)


def test_a_square_of_absorbed_power_adds_its_curvature_to_the_chain_rule(
    line, line_result
):
    # J = P^2: its linear term is 2 |P| eta_P, its remainder 1/2 ||H|| eta_P^2
    # with H = 2, where eta_P is P's own estimate (linear term and remainder)
    power = float(_layer_power(line_result, line))
    eta = gfem.error_estimate(
        line_result, gfem.QuantityEstimator(objective=_layer_power), model=line
    ).value
    squared = gfem.error_estimate(
        line_result,
        gfem.QuantityEstimator(objective=lambda r, m: _layer_power(r, m) ** 2),
        model=line,
    ).value
    assert squared == pytest.approx(2 * power * eta + eta**2, rel=1e-9)


def test_a_field_objective_over_frequencies_and_excitations_sums_its_reads(
    line, line_result
):
    # each read's combined dual is its own: a sum of reads at different
    # frequencies is estimated as the sum of their estimates
    def at(f, port):
        return lambda r, m: jnp.real(_box_mean(r, m, f, port)[2])

    reads = [at(LINE_F, "p1"), at(1.25 * LINE_F, "p2")]
    parts = [
        gfem.error_estimate(
            line_result, gfem.QuantityEstimator(objective=j), model=line
        ).value
        for j in reads
    ]
    total = gfem.error_estimate(
        line_result,
        gfem.QuantityEstimator(objective=lambda r, m: reads[0](r, m) + reads[1](r, m)),
        model=line,
    )
    assert total.kind == "objective of field means"
    assert total.value == pytest.approx(sum(parts), rel=1e-9)


@pytest.mark.parametrize(
    ("change", "feature"),
    [
        (
            lambda m: {
                "boundaries": [gfem.Conductor("bottom", 5.8e7), m.boundaries[1]]
            },
            "surface loss in power_flow",
        ),
        (
            lambda m: {
                "materials": m.materials
                | {"lsub": gfem.Material(eps_r=(4.0, 4.0, 5.0), tan_delta=0.05)}
            },
            "anisotropic or dispersive power_flow",
        ),
        (
            lambda m: {
                "materials": m.materials | {"lsub": gfem.Material(eps_r=cc.DRUDE_METAL)}
            },
            "anisotropic or dispersive power_flow",
        ),
    ],
    ids=["conductor", "anisotropic", "law"],
)
def test_a_power_flow_side_the_estimate_has_no_case_for_raises(
    line, line_result, change, feature
):
    # the side's checks read the model only: the result's mesh is the same
    error = _raises(feature, _layer_power, line_result, _with(line, **change(line)))
    assert "FieldEstimator()" in str(error)


def test_a_power_flow_side_touching_a_port_raises(line, line_result):
    def through(result, model):
        return -gfem.power_flow(result.e_field(LINE_F, "p1"), "diel", model=model)

    error = _raises("power through a port", through, line_result, line)
    assert "result.s" in str(error)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_field_dual_factorizes_once_then_rides_on_the_solves(line, monkeypatch):
    from gfem import refinement

    calls = []
    duals = refinement._field_duals

    def counted(*args, **kwargs):
        calls.append(args[2])
        return duals(*args, **kwargs)

    monkeypatch.setattr(refinement, "_field_duals", counted)
    estimator = gfem.QuantityEstimator(objective=_layer_power)
    refinement_ = gfem.refine_mesh(
        line,
        gfem.Driven([LINE_F]),
        estimator=estimator,
        tolerance=1e-7,
        options=OPTIONS,
        max_cycles=2,
    )
    assert calls == [[("power", LINE_F, "p1", ("lsub", "lossy"))]]  # cycle 0 only
    assert len(refinement_.cycles) == 3
    assert refinement_.estimate.kind == "objective of absorbed power"
    # the ridden dual gives the estimate error_estimate gives on the last mesh
    again = gfem.error_estimate(refinement_.result, estimator, model=refinement_.model)
    assert again.value == pytest.approx(refinement_.estimate.value, rel=1e-8)


def test_a_field_objective_refuses_frequency_workers(line):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            line,
            gfem.Driven([LINE_F, 1.25 * LINE_F]),
            estimator=gfem.QuantityEstimator(objective=_layer_power),
            tolerance=1e-4,
            options=gfem.SolverOptions(
                order=2, linear_solver="direct", frequency_workers=2
            ),
            max_cycles=0,
        )
    assert caught.value.feature == "frequency workers"


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


# --- objectives of eigenmode results ----------------------------------------------


def _with_leaf(result, name, value):
    return dataclasses.replace(result, **{name: {"x": jnp.asarray(value)}})


@pytest.mark.parametrize(
    ("leaf", "shape"),
    [
        ("q_port", (3,)),
        ("participation", (3,)),
        ("sheet_participation", (3,)),
        ("principal_participation", (3, 3)),
    ],
)
def test_an_eigen_objective_reading_a_refused_leaf_raises(cavity, leaf, shape):
    model, result = cavity
    fabricated = _with_leaf(result, leaf, np.full(shape, 0.5))

    def objective(r, _model):
        return jnp.sum(getattr(r, leaf)["x"][0]) + r.frequency[0] * 1e-9

    _raises(leaf, objective, fabricated, model)


def test_an_eigen_objective_reading_a_mode_field_raises(cavity):
    model, result = cavity

    def coefficients(r, _model):
        return jnp.sum(jnp.abs(r.e_fields[0].coefficients) ** 2)

    def point(r, _model):
        e = r.e_fields[0].at(np.array([0.2e-3, 0.2e-3, 0.1e-3]))
        return jnp.sum(jnp.abs(e) ** 2)

    _raises("normalized mode field", coefficients, result, model)
    _raises("normalized mode field", point, result, model)


def test_error_estimate_of_an_eigen_objective_of_omega(cavity):
    model, result = cavity

    def objective(r, _model):
        return (r.frequency[1].real / r.frequency[0].real - 1.1) ** 2

    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    assert estimate.kind == "objective of omega"
    assert estimate.value > 0
    assert estimate.indicators.shape == (len(model.mesh.tetrahedra),)
    assert estimate._modes == (0, 1)


def test_an_eigen_objective_reading_a_degenerate_mode_raises(cavity):
    model, result = cavity
    clustered = dataclasses.replace(result, cluster=jnp.asarray([0, 0, 1]))
    estimator = gfem.QuantityEstimator(objective=lambda r, _: r.frequency[0].real)
    with pytest.raises(gfem.DegenerateModeError, match="degenerate cluster"):
        gfem.error_estimate(clustered, estimator, model=model)


def test_an_eigen_objective_keeps_the_lossy_eigenmode_refusal(tmp_path):
    lossless = rc.cavity_model(tmp_path, 2)
    model = gfem.Model(
        lossless.mesh,
        materials={"air": gfem.MaterialField(mu_r=2.0 * (1 - 0.1j))},
        boundaries=lossless.boundaries,
    )
    result = gfem.solve(
        model, gfem.Eigenmode(target=rc.CAVITY_F1 / 1.4, n_modes=1), options=OPTIONS
    )
    estimator = gfem.QuantityEstimator(objective=lambda r, _: r.frequency[0].real)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(result, estimator, model=model)
    assert caught.value.feature == "lossy eigenmode"


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_an_eigen_objective_follows_each_mode_it_reads(cavity):
    model, _ = cavity

    def objective(r, _model):
        return r.frequency[1].real * 1e-9

    refined = gfem.refine_mesh(
        model,
        gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=3),
        estimator=gfem.QuantityEstimator(objective=objective),
        tolerance=1e-6,
        options=OPTIONS,
        max_cycles=1,
    )
    assert len(refined.cycles) == 2
    for cycle in refined.cycles:
        assert 1 in cycle.modes
        assert cycle.estimate > 0
    # the objective's value is mode 1 of the first mesh, followed
    last = refined.cycles[-1]
    omega = np.asarray(refined.result.omega)[last.modes[1]]
    assert last.value == pytest.approx(omega.real / (2 * np.pi) * 1e-9, rel=1e-12)


# --- objectives of participation ---------------------------------------------------


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    """A 1 x 0.7 x 0.2 mm PEC box, eps_r 4 slab at 0.3 < x < 0.6: two modes."""
    edges = tuple(np.linspace(0, a, round(a * 10) + 1) for a in (1.0, 0.7, 0.2))

    def volume(c):
        return "block" if 0.3 < c[0] < 0.6 else "air"

    edges_end = [e[-1] for e in edges]

    def surface_of(c):
        on = [abs(c[a]) < 1e-9 or abs(c[a] - edges_end[a]) < 1e-9 for a in range(3)]
        return "walls" if any(on) else ""

    path = sm.write_msh22(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        sm.structured(edges, volume, surface_of),
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": gfem.Material(), "block": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("walls")],
    )
    problem = gfem.Eigenmode(target=140e9, n_modes=2)
    return model, problem, gfem.solve(model, problem, options=OPTIONS)


def _block_share(result, _model):
    return result.bulk_participation["block"][0]


def test_error_estimate_of_a_participation(slab):
    model, _, result = slab
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=_block_share), model=model
    )
    assert estimate.kind == "objective of participation"
    assert estimate.value > 0
    assert estimate.indicators.shape == (len(model.mesh.tetrahedra),)
    assert estimate._dual_modes == ((0, "block"),)


def test_a_participation_times_its_frequency_is_one_participation_objective(slab):
    model, _, result = slab

    def objective(r, _model):
        return r.bulk_participation["block"][0] * r.frequency[0].real * 1e-11

    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    assert estimate.kind == "objective of participation"
    assert estimate._modes == (0,)


def test_a_participation_of_a_mode_near_another_raises(slab):
    model, _, result = slab
    w = complex(np.asarray(result.omega)[0])
    crowded = dataclasses.replace(
        result, omega=jnp.asarray([w, w * (1 + 1e-5)]), cluster=jnp.arange(2)
    )
    with pytest.raises(gfem.DegenerateModeError, match="mode 1"):
        gfem.error_estimate(
            crowded, gfem.QuantityEstimator(objective=_block_share), model=model
        )


@pytest.mark.parametrize(
    "options",
    [
        gfem.SolverOptions(order=2, linear_solver="bddc_gmres"),
        gfem.SolverOptions(order=2, linear_solver="direct", eigen_tolerance=1e-8),
    ],
)
def test_a_participation_out_of_mode_field_reach_raises(slab, options):
    model, problem, _ = slab
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model,
            problem,
            estimator=gfem.QuantityEstimator(objective=_block_share),
            tolerance=1e-3,
            options=options,
            max_cycles=0,
        )
    assert caught.value.feature == "mode-field dual"


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_participation_dual_is_solved_once_then_rides_on_the_forward(
    slab, monkeypatch
):
    from gfem import refinement

    calls = []
    solve = refinement._participation_duals

    def counted(*args, **kwargs):
        calls.append(args[2])
        return solve(*args, **kwargs)

    monkeypatch.setattr(refinement, "_participation_duals", counted)
    model, problem, _ = slab
    estimator = gfem.QuantityEstimator(objective=_block_share)
    refined = gfem.refine_mesh(
        model,
        problem,
        estimator=estimator,
        tolerance=1e-9,
        options=OPTIONS,
        max_cycles=2,
    )
    assert calls == [0]  # the first mesh only
    assert len(refined.cycles) == 3
    for cycle in refined.cycles:
        assert 0 in cycle.modes
    # the ridden dual gives the estimate error_estimate gives on the last mesh
    view = _objective.aligned(refined.result, refined.cycles[-1].modes)
    again = gfem.error_estimate(view, estimator, model=refined.model)
    assert again.value == pytest.approx(refined.estimate.value, rel=1e-6)


# --- the lossless rewrite -------------------------------------------------------


def test_a_lossless_lumped_port_model_estimates_through_its_lossless_rewrite(
    step, step_result
):
    estimator = gfem.QuantityEstimator(objective=_s21_band)
    estimate = gfem.error_estimate(step_result, estimator, model=step)
    assert estimate.kind == "objective of s, lossless rewrite"
    assert estimate.indicators.shape == (len(step.mesh.tetrahedra),)
    # lam = 0 is a choice of the rewrite, so it is never above C
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
    # the rewrite adds zero-cotangent rows to each read column, so tiny
    # per-tetrahedron weights move by round-off: compare to the largest
    np.testing.assert_allclose(
        at_zero.weights, c.weights, rtol=1e-12, atol=1e-12 * np.max(c.weights)
    )
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


# --- electrostatic objectives --------------------------------------------------------


@pytest.mark.parametrize(
    ("objective", "feature"),
    [
        (
            lambda r, _: r.potential["a"].at(np.array([2e-6, 1e-6, 0.25e-6])) ** 2,
            "point value",
        ),
        (
            lambda r, _: r.capacitance[0, 1] + r.energy["dielectric"][0],
            "mixed objective",
        ),
        (
            lambda r, _: (
                gfem.region_mean(r.potential["a"], "dielectric")
                + r.energy["dielectric"][0] * 1e17
            ),
            "mixed objective",
        ),
    ],
)
def test_an_electrostatic_objective_reading_what_has_no_estimate_raises(
    bars, objective, feature
):
    model, result = bars
    _raises(feature, objective, result, model)


@pytest.fixture(scope="module")
def split_bars(tmp_path_factory):
    model = rc.bars_split_model(tmp_path_factory.mktemp("split"), "near", 2)
    return model, gfem.solve(model, rc.BARS_PROBLEM, options=OPTIONS)


@pytest.mark.parametrize(
    ("objective", "kind"),
    [
        (lambda r, _: r.inverse[0, 1], "objective of capacitance"),
        (lambda r, _: r.energy["left"][1] * 1e18, "objective of electrostatic energy"),
        (
            lambda r, _: (
                gfem.region_mean(r.potential["a"], "left")
                - gfem.region_mean(r.potential["a"], ["right"])
            ),
            "objective of potential means",
        ),
    ],
)
def test_error_estimate_of_an_electrostatic_objective(split_bars, objective, kind):
    model, result = split_bars
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    assert estimate.kind == kind
    assert len(estimate.values) == 1
    assert estimate.value > 0
    assert estimate.indicators.shape == (len(model.mesh.tetrahedra),)


def test_a_capacitance_entry_objective_is_the_entry_estimate_in_farads(split_bars):
    # J = C_ab: the combined form is the entry's own, scaled by |C_ab|
    model, result = split_bars
    objective = gfem.error_estimate(
        result,
        gfem.QuantityEstimator(objective=lambda r, _: r.capacitance[0, 1]),
        model=model,
    )
    entry = gfem.error_estimate(
        result, gfem.QuantityEstimator(capacitance=("a", "b")), model=model
    )
    c_ab = abs(float(result.capacitance[0, 1]))
    assert objective.value == pytest.approx(entry.value * c_ab, rel=1e-10)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_electrostatic_dual_factorizes_once_then_rides_on_the_solves(
    split_bars, monkeypatch
):
    from gfem._engine import electrostatic

    calls = []
    solve_duals = electrostatic.solve_duals

    def counted(*args, **kwargs):
        calls.append(1)
        return solve_duals(*args, **kwargs)

    monkeypatch.setattr(electrostatic, "solve_duals", counted)
    model, _ = split_bars

    estimator = gfem.QuantityEstimator(
        objective=lambda r, _: r.energy["left"][0] * 1e17
    )
    refinement_ = gfem.refine_mesh(
        model,
        rc.BARS_PROBLEM,
        estimator=estimator,
        tolerance=1e-6,
        options=gfem.SolverOptions(order=2, linear_solver="direct", tolerance=1e-10),
        max_cycles=2,
    )
    assert len(calls) == 1  # cycle 0 only
    assert len(refinement_.cycles) == 3
    again = gfem.error_estimate(refinement_.result, estimator, model=refinement_.model)
    assert again.value == pytest.approx(refinement_.estimate.value, rel=1e-8)
