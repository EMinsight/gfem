"""`gfem.tolerance_report`: linearized spread, effects and yield of named quantities.

Analytic cases run on JAX functions with no solve, so every expected value is a
closed form. One tiny real solve checks the Jacobian against finite differences.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem.shape import boundary_morph

# The Monte Carlo of the joint yield draws N = 2**18 samples; a yield Y is
# estimated with standard error sqrt(Y (1 - Y) / N). The tests accept 4
# standard errors (a false failure about 6e-5, and the sampler is seeded).
N_SAMPLES = 2**18


def standard_error(y: float) -> float:
    return math.sqrt(y * (1 - y) / N_SAMPLES)


def phi(x: float) -> float:
    """The standard normal distribution function, from the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# A linear model: a = 2 x + 3 y + 1, b = x - 4 y, nominal (1, 2). Tolerances
# (0.3, 0.6) are three standard deviations, so sigma = (0.1, 0.2).
J_LIN = np.array([[2.0, 3.0], [1.0, -4.0]])
F0_LIN = np.array([9.0, -7.0])
SIGMA = np.array([0.1, 0.2])
NOMINAL = {"x": 1.0, "y": 2.0}
TOLERANCE = {"x": 0.3, "y": 0.6}


def linear(p):
    return {"a": 2 * p["x"] + 3 * p["y"] + 1, "b": p["x"] - 4 * p["y"]}


def report(**kwargs):
    kwargs.setdefault("tolerance", TOLERANCE)
    return gfem.tolerance_report(linear, NOMINAL, **kwargs)


# Round-off of a few float64 operations on numbers of order 10 is below
# 1e-14; the closed forms agree with the report to 1e-12 relative.
RTOL = 1e-12


def test_values_and_jacobian_of_a_linear_model():
    r = report()
    assert r.parameters == ("x", "y")
    assert r.quantities == ("a", "b")
    assert r.values == pytest.approx({"a": 9.0, "b": -7.0}, rel=RTOL)
    np.testing.assert_allclose(r.jacobian, J_LIN, rtol=RTOL)
    assert r.jacobian.shape == (2, 2)


def test_effect_is_the_change_at_plus_tolerance():
    r = report()
    t = np.array([0.3, 0.6])
    np.testing.assert_allclose(r.effect, J_LIN * t, rtol=RTOL)


def test_spread_is_the_root_sum_of_squares_of_the_independent_effects():
    r = report()
    expected = np.sqrt(((J_LIN * SIGMA) ** 2).sum(axis=1))
    assert [r.std["a"], r.std["b"]] == pytest.approx(list(expected), rel=RTOL)


def test_variance_shares_sum_to_one_and_have_no_correlation_part():
    r = report()
    np.testing.assert_allclose(r.share.sum(axis=1), 1.0, rtol=RTOL)
    np.testing.assert_allclose(
        r.share[0], (J_LIN[0] * SIGMA) ** 2 / (0.04 + 0.36), rtol=RTOL
    )
    assert r.correlation_share is None


def test_correlation_changes_the_spread_and_adds_a_remainder():
    rho = 0.5
    r = report(correlation={("y", "x"): rho})
    # var a = (2 * 0.1)^2 + (3 * 0.2)^2 + 2 * rho * (2 * 0.1) * (3 * 0.2)
    var_a = 0.04 + 0.36 + 2 * rho * 0.2 * 0.6
    assert r.std["a"] == pytest.approx(math.sqrt(var_a), rel=RTOL)
    assert r.correlation_share["a"] == pytest.approx(2 * rho * 0.12 / var_a, rel=RTOL)
    # b has J = (1, -4): the cross term is negative
    var_b = 0.01 + 0.64 + 2 * rho * 0.1 * (-0.8)
    assert r.correlation_share["b"] == pytest.approx(
        2 * rho * 0.1 * (-0.8) / var_b, rel=RTOL
    )
    total = r.share.sum(axis=1) + np.array(
        [r.correlation_share["a"], r.correlation_share["b"]]
    )
    np.testing.assert_allclose(total, 1.0, rtol=RTOL)


def test_the_order_within_a_correlation_pair_is_free_and_a_repeat_may_agree():
    one = report(correlation={("x", "y"): 0.3})
    other = report(correlation={("y", "x"): 0.3})
    twice = report(correlation={("x", "y"): 0.3, ("y", "x"): 0.3})
    assert one.std == pytest.approx(other.std, rel=RTOL)
    assert one.std == pytest.approx(twice.std, rel=RTOL)


def test_worst_case_is_the_sum_of_absolute_effects():
    r = report()
    for i, name in enumerate(("a", "b")):
        half = (np.abs(J_LIN[i]) * np.array([0.3, 0.6])).sum()
        assert r.worst_case[name] == pytest.approx(
            (F0_LIN[i] - half, F0_LIN[i] + half), rel=RTOL
        )


def test_single_spec_yield_and_cpk_are_the_normal_closed_forms():
    s = math.sqrt(0.4)
    r = report(specs={"a": (8.5, None)})
    exact = 1 - phi((8.5 - 9.0) / s)
    assert r.yields["a"] == pytest.approx(exact, abs=1e-12)
    assert r.cpk["a"] == pytest.approx((9.0 - 8.5) / (3 * s), rel=RTOL)
    # one spec: the joint yield is that exact yield, with no sampling error
    assert r.joint_yield == pytest.approx(exact, abs=1e-12)
    assert r.joint_yield_error == 0.0
    assert r.joint_yield_bounds == pytest.approx((exact, exact), abs=1e-12)
    assert set(r.yields) == {"a"}


def test_two_sided_window_and_the_upper_one_sided_form():
    s = math.sqrt(0.4)
    r = report(specs={"a": (8.0, 10.5), "b": (None, -6.0)})
    exact_a = phi((10.5 - 9.0) / s) - phi((8.0 - 9.0) / s)
    exact_b = phi((-6.0 + 7.0) / math.sqrt(0.65))
    assert r.yields["a"] == pytest.approx(exact_a, abs=1e-12)
    assert r.yields["b"] == pytest.approx(exact_b, abs=1e-12)
    # the margin is the nearer finite side
    assert r.cpk["a"] == pytest.approx(min(1.5, 1.0) / (3 * s), rel=RTOL)


def test_a_far_lower_limit_keeps_its_accuracy_in_the_tail():
    # Y = Phi(-a) - Phi(-b) for two tail limits: no cancellation to 0 or below
    r = report(specs={"a": (9.0 + 7 * math.sqrt(0.4), 9.0 + 8 * math.sqrt(0.4))})
    exact = 0.5 * math.erfc(7 / math.sqrt(2)) - 0.5 * math.erfc(8 / math.sqrt(2))
    assert r.yields["a"] > 0
    assert r.yields["a"] == pytest.approx(exact, rel=1e-9)


def test_joint_yield_of_independent_quantities_is_the_product():
    # u = x, v = y with sigma 1 and no correlation: the windows factor
    def uv(p):
        return {"u": p["x"], "v": p["y"]}

    r = gfem.tolerance_report(
        uv,
        {"x": 0.0, "y": 0.0},
        tolerance={"x": 3.0, "y": 3.0},
        specs={"u": (-1.0, 1.0), "v": (-2.0, None)},
    )
    exact = (phi(1) - phi(-1)) * (1 - phi(-2))
    assert abs(r.joint_yield - exact) <= 4 * standard_error(exact)
    assert r.joint_yield_error == pytest.approx(standard_error(r.joint_yield), rel=1e-9)
    lower, upper = r.joint_yield_bounds
    assert lower == pytest.approx(max(0.0, 1 - (1 - r.yields["u"]) - (1 - r.yields["v"])))
    assert upper == pytest.approx(min(r.yields["u"], r.yields["v"]))
    assert lower <= exact <= upper


def test_joint_yield_of_two_correlated_quantities_matches_the_orthant_formula():
    # P(u > 0, v > 0) = 1/4 + arcsin(rho) / (2 pi) for a correlated pair
    rho = 0.6

    def uv(p):
        return {"u": p["x"], "v": p["y"]}

    r = gfem.tolerance_report(
        uv,
        {"x": 0.0, "y": 0.0},
        tolerance={"x": 3.0, "y": 3.0},
        specs={"u": (0.0, None), "v": (0.0, None)},
        correlation={("x", "y"): rho},
    )
    exact = 0.25 + math.asin(rho) / (2 * math.pi)
    assert abs(r.joint_yield - exact) <= 4 * standard_error(exact)
    assert r.yields["u"] == pytest.approx(0.5, abs=1e-12)
    lower, upper = r.joint_yield_bounds
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.5, abs=1e-12)


def test_the_joint_yield_is_reproducible_between_calls():
    kwargs = {"specs": {"a": (8.0, None), "b": (None, -6.0)}}
    assert report(**kwargs).joint_yield == report(**kwargs).joint_yield


def test_zero_tolerance_holds_the_parameter():
    r = report(tolerance={"x": 0.3, "y": 0.0}, specs={"a": (8.0, None)})
    np.testing.assert_array_equal(r.effect[:, 1], 0.0)
    np.testing.assert_array_equal(r.share[:, 1], 0.0)
    assert r.std["a"] == pytest.approx(0.2, rel=RTOL)
    assert np.isfinite(r.share).all()
    assert "y has no effect" not in str(r)  # the tolerance, not the function, is 0


def test_all_tolerances_zero_gives_a_deterministic_pass_or_fail():
    zero = {"x": 0.0, "y": 0.0}
    r = report(tolerance=zero, specs={"a": (8.0, 10.0), "b": (-6.0, None)})
    assert r.std == {"a": 0.0, "b": 0.0}
    assert r.yields == {"a": 1.0, "b": 0.0}  # b = -7 lies below -6
    assert r.cpk["a"] == math.inf
    assert r.joint_yield == 0.0
    assert np.isfinite(r.share).all()
    assert r.checked == ()


def test_a_quantity_without_a_spec_has_no_yield():
    r = report(specs={"a": (8.0, None)})
    assert "b" not in r.yields
    assert "b" not in r.cpk
    assert r.specs == {"a": (8.0, None)}
    assert report().joint_yield is None
    assert report().joint_yield_bounds is None
    assert report().joint_yield_error is None
    assert report().yields == {}


# --- the linearity check ------------------------------------------------------------


def quadratic(c):
    """q = p + c p^2 at p = 0: slope 1, curvature 2 c."""
    return lambda p: {"q": p["p"] + c * p["p"] ** 2}


T = 0.1


def test_linear_error_and_mean_shift_of_a_quadratic_are_exact():
    # at +-T: q = +-T + c T^2, so the error against the slope is c T^2 on
    # both sides and the curvature is 2 c; sigma = T / 3
    c = 0.5
    r = gfem.tolerance_report(quadratic(c), {"p": 0.0}, tolerance={"p": T})
    assert r.checked == ("p",)
    assert r.linear_error.shape == (1, 1)
    # the three values are of order 0.1: round-off of a few 1e-17
    assert r.linear_error[0, 0] == pytest.approx(c * T**2, abs=1e-15)
    assert r.mean_shift["q"] == pytest.approx(0.5 * 2 * c * (T / 3) ** 2, abs=1e-15)


def test_a_quantity_is_flagged_when_the_error_exceeds_a_tenth_of_three_sigma():
    # error c T^2 against 0.1 * 3 s = 0.1 T (s = T/3, slope 1): flag when c T > 0.1
    flagged = gfem.tolerance_report(quadratic(2.0), {"p": 0.0}, tolerance={"p": T})
    assert flagged.nonlinear == ("q",)
    assert "nonlinear within" in str(flagged)
    assert "not reliable" in str(flagged)
    fine = gfem.tolerance_report(quadratic(0.5), {"p": 0.0}, tolerance={"p": T})
    assert fine.nonlinear == ()
    assert "within a tenth of" in str(fine)


def test_check_linearity_zero_skips_the_pair():
    calls = []

    def counting(p):
        calls.append(1)
        return {"q": p["p"] ** 2 + p["p"]}

    gfem.tolerance_report(counting, {"p": 1.0}, tolerance={"p": 0.1}, check_linearity=0)
    off = len(calls)
    calls.clear()
    r = gfem.tolerance_report(counting, {"p": 1.0}, tolerance={"p": 0.1})
    # the Jacobian trace calls the function once; the check adds the pair
    assert len(calls) == off + 2
    assert r.checked == ("p",)
    off_report = gfem.tolerance_report(
        counting, {"p": 1.0}, tolerance={"p": 0.1}, check_linearity=0
    )
    assert off_report.checked == ()
    assert off_report.linear_error.shape == (1, 0)
    assert off_report.mean_shift == {}
    assert off_report.nonlinear == ()
    assert "Linearity not checked." in str(off_report)


def test_the_check_takes_the_parameters_with_the_largest_variance_share():
    def three(p):
        return {"q": p["a"] + 5 * p["b"] + 2 * p["c"]}

    nominal = {"a": 0.0, "b": 0.0, "c": 0.0}
    tol = {"a": 0.3, "b": 0.3, "c": 0.3}
    assert gfem.tolerance_report(three, nominal, tolerance=tol).checked == ("b",)
    two = gfem.tolerance_report(three, nominal, tolerance=tol, check_linearity=2)
    assert two.checked == ("b", "c")
    assert two.linear_error.shape == (1, 2)
    # a linear function has no linear error beyond round-off at 5 * 0.3
    assert np.abs(two.linear_error).max() < 1e-14


def test_a_held_parameter_is_not_checked():
    def two(p):
        return {"q": p["a"] ** 2 + p["b"]}

    r = gfem.tolerance_report(
        two, {"a": 1.0, "b": 1.0}, tolerance={"a": 0.0, "b": 0.1}, check_linearity=2
    )
    assert r.checked == ("b",)


def test_a_limit_beyond_three_sigma_is_marked_tail_extrapolated():
    far = report(specs={"a": (4.0, None)})
    assert "*" in str(far)
    assert "beyond" in str(far)
    near = report(specs={"a": (8.5, None)})
    assert "%*" not in str(near)


def test_a_parameter_the_function_does_not_read_is_named():
    def partial(p):
        return {"q": p["a"]}

    r = gfem.tolerance_report(
        partial, {"a": 1.0, "z": 2.0}, tolerance={"a": 0.1, "z": 0.1}
    )
    assert "z has no effect on any quantity: check that the function reads it." in str(r)
    held = gfem.tolerance_report(
        partial, {"a": 1.0, "z": 2.0}, tolerance={"a": 0.1, "z": 0.0}
    )
    assert "no effect" not in str(held)


# --- the text -----------------------------------------------------------------------


def test_text_tables_carry_the_rows_and_columns():
    r = report(specs={"a": (6.0, None), "b": (None, 0.0)})
    text = str(r)
    assert text.startswith("Tolerance report: 2 quantities, 2 parameters\n")
    assert "3 standard deviations" in text
    assert "parameters independent" in text
    for word in ("Parameter", "nominal", "tolerance", "worst case", "Cpk", "yield"):
        assert word in text
    assert ">= 6" in text
    assert "<= 0" in text
    assert "±0.3" in text
    assert "pass" in text  # a: worst case [6.6, 11.4] meets >= 6
    assert "All specs:" in text
    assert "Monte Carlo" in text
    assert "Effect at +tolerance" in text
    assert "(" in text.split("Effect at +tolerance")[1]
    assert "Linearity, checked on" in text


def test_a_fail_verdict_and_the_correlation_note():
    r = report(specs={"a": (8.5, None)}, correlation={("x", "y"): 0.5})
    text = str(r)
    assert "fail" in text  # worst case [6.6, 11.4] reaches below 8.5
    assert "correlation +" in text
    assert "correlated" in text


def test_largest_share_comes_first_in_the_effects():
    r = report()
    line = next(
        row for row in str(r).splitlines() if row.startswith("b ") and " %)" in row
    )
    # b: shares x 0.01/0.65, y 0.64/0.65
    assert line.index("y ") < line.index("x ")


def test_one_spec_prints_an_exact_yield():
    text = str(report(specs={"a": (8.5, None)}))
    assert "exact" in text
    assert "Monte Carlo" not in text


def test_markdown_is_pipe_tables():
    r = report(specs={"a": (8.5, None), "b": (None, 0.0)})
    md = r.to_markdown()
    rows = [line for line in md.splitlines() if line.startswith("|")]
    assert any(set(line) <= set("|- :") for line in rows)  # separator rows
    assert any("a" in line and "±" in line for line in rows)
    assert "Tolerance report" in md
    # every table row has the same number of cells as its header
    block: list[str] = []
    for line in [*md.splitlines(), ""]:
        if line.startswith("|"):
            block.append(line)
        elif block:
            assert len({row.count("|") for row in block}) == 1
            block = []


def test_repr_is_one_line():
    r = report(specs={"a": (8.5, None), "b": (None, 0.0)})
    assert repr(r).startswith("ToleranceReport(2 quantities, 2 parameters, joint yield ")
    assert "\n" not in repr(r)
    assert repr(report()) == "ToleranceReport(2 quantities, 2 parameters)"


def test_the_report_is_a_frozen_value():
    r = report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.joint_yield = 1.0  # ty: ignore[invalid-assignment]


# --- refusals -----------------------------------------------------------------------


def call(quantities=linear, nominal=None, **kwargs):
    kwargs.setdefault("tolerance", TOLERANCE)
    return gfem.tolerance_report(quantities, nominal or NOMINAL, **kwargs)


@pytest.mark.parametrize(
    ("quantities", "match"),
    [
        (lambda p: p["x"], "mapping"),
        (lambda p: {1: p["x"]}, "str"),
        (lambda p: {"c": p["x"] + 1j * p["y"]}, "complex.*jnp.abs"),
        (lambda p: {"c": jnp.asarray(1j) * p["x"]}, "complex"),
    ],
)
def test_bad_quantities_raise_type_error(quantities, match):
    with pytest.raises(TypeError, match=match):
        call(quantities)


def test_a_traced_nominal_raises_type_error():
    def outer(x):
        return gfem.tolerance_report(
            linear, {"x": x, "y": 2.0}, tolerance=TOLERANCE
        ).jacobian.sum()

    with pytest.raises(TypeError, match="not differentiable"):
        jax.grad(outer)(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantities": lambda p: {"v": jnp.ones(3) * p["x"]}}, "'v'.*one name per"),
        ({"quantities": lambda p: {}}, "no quantity"),
        ({"nominal": {"x": jnp.ones(2), "y": 2.0}}, "scalar.*FabricationModel"),
        ({"nominal": {"x": float("nan"), "y": 2.0}}, "finite"),
        ({"tolerance": {"x": 0.3}}, "keys.*'y'"),
        ({"tolerance": {"x": 0.3, "y": 0.6, "z": 1.0}}, "keys.*'z'"),
        ({"tolerance": {"x": -0.3, "y": 0.6}}, "negative"),
        ({"tolerance": {"x": float("inf"), "y": 0.6}}, "finite"),
        ({"specs": {"nope": (0.0, 1.0)}}, "'nope'"),
        ({"specs": {"a": (None, None)}}, "both sides"),
        ({"specs": {"a": (float("nan"), 1.0)}}, "NaN"),
        ({"specs": {"a": (2.0, 1.0)}}, "low < high"),
        ({"specs": {"a": (1.0, 1.0)}}, "low < high"),
        ({"correlation": {("x", "nope"): 0.1}}, "'nope'"),
        ({"correlation": {("x", "x"): 0.1}}, "itself"),
        ({"correlation": {("x", "y"): 0.1, ("y", "x"): 0.2}}, "twice"),
        ({"correlation": {("x", "y"): 1.5}}, r"\[-1, 1\]"),
        ({"correlation": {("x", "y"): float("nan")}}, r"\[-1, 1\]"),
        ({"check_linearity": -1}, "check_linearity"),
        ({"check_linearity": 3}, "check_linearity"),
    ],
)
def test_bad_inputs_raise_value_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        call(**kwargs)


def test_a_correlation_matrix_that_is_not_positive_semidefinite_is_refused():
    # three parameters with pairwise rho = -0.9: the smallest eigenvalue is
    # 1 + 2 * (-0.9) = -0.8 < 0
    def three(p):
        return {"q": p["a"] + p["b"] + p["c"]}

    names = ("a", "b", "c")
    with pytest.raises(ValueError, match="positive semidefinite"):
        gfem.tolerance_report(
            three,
            dict.fromkeys(names, 0.0),
            tolerance=dict.fromkeys(names, 1.0),
            correlation={("a", "b"): -0.9, ("a", "c"): -0.9, ("b", "c"): -0.9},
        )


def test_a_singular_correlation_is_allowed():
    # rho = 1: y follows x, and q = x - y is exactly constant: s = 0 up to round-off
    def diff(p):
        return {"d": p["x"] + p["y"]}

    r = gfem.tolerance_report(
        diff,
        {"x": 0.0, "y": 0.0},
        tolerance={"x": 3.0, "y": 3.0},
        correlation={("x", "y"): 1.0},
        specs={"d": (-6.0, None)},
    )
    # fully correlated: s = 2, so the limit is at -3 s
    assert r.std["d"] == pytest.approx(2.0, rel=RTOL)
    assert r.yields["d"] == pytest.approx(1 - phi(-3.0), abs=1e-12)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf")],
)
def test_a_non_finite_quantity_names_the_quantity(value):
    with pytest.raises(ValueError, match="'a'.*finite"):
        gfem.tolerance_report(
            lambda p: {"a": p["x"] * value}, {"x": 1.0}, tolerance={"x": 0.1}
        )


def test_a_non_finite_check_value_names_the_quantity_and_the_parameter():
    def edge(p):
        # finite and smooth at the nominal point, infinite at the +tolerance corner
        return {"q": 1.0 / (p["p"] - 1.1)}

    with pytest.raises(ValueError, match="'q'.*'p'.*finite"):
        gfem.tolerance_report(edge, {"p": 1.0}, tolerance={"p": 0.1})


# --- the scalar boundary morph ------------------------------------------------------


def test_a_scalar_morph_call_is_the_uniform_offset(tmp_path):
    import gradient_cases as gc

    mesh = gc.wave_mesh(tmp_path)
    model = gc.wave_line(mesh, "lossless")
    morph = boundary_morph(model, "top")
    h = 1e-5
    scalar = morph(h)
    vector = morph(h * jnp.ones(morph.n_params))
    # the scalar is broadcast before the same linear map: the nodes agree to
    # the round-off of the solve, below 1e-9 of the 1e-5 m offset
    np.testing.assert_allclose(
        np.asarray(scalar.nodes), np.asarray(vector.nodes), rtol=0, atol=1e-14
    )
    moved = np.asarray(scalar.nodes) - np.asarray(mesh.nodes)
    assert np.abs(moved).max() == pytest.approx(h, rel=1e-9)
    with pytest.raises(ValueError, match="normal displacements"):
        morph(jnp.ones(morph.n_params + 1))
    # a scalar is also traceable
    g = jax.grad(lambda t: jnp.sum(morph(t).nodes[:, 2]))(0.0)
    assert np.isfinite(float(g))


# --- one real solve -----------------------------------------------------------------


def test_the_jacobian_of_a_real_solve_matches_central_differences(tmp_path):
    """Edge bias of the top plate and the permittivity of one layer, driven S."""
    import gradient_cases as gc

    mesh = gc.wave_mesh(tmp_path)
    model = gc.wave_line(mesh, "lossless")
    morph = boundary_morph(model, "top")
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    problem = gfem.Driven(frequency=(12e9, 18e9))

    def quantities(p):
        m = model.replace(mesh=morph(p["edge bias"]))
        m = m.with_material("lo", gfem.MaterialField(eps_r=p["eps_lo"]))
        s = gfem.solve(m, problem, options=options).s
        return {
            "|S21|^2 at 12 GHz": jnp.abs(s["p2", "p1"][0]) ** 2,
            "|S11|^2 at 18 GHz": jnp.abs(s["p1", "p1"][1]) ** 2,
        }

    nominal = {"edge bias": 0.0, "eps_lo": 4.0}
    tolerance = {"edge bias": 5e-6, "eps_lo": 0.2}
    r = gfem.tolerance_report(
        quantities,
        nominal,
        tolerance=tolerance,
        specs={"|S21|^2 at 12 GHz": (0.0, None)},
    )
    assert r.jacobian.shape == (2, 2)
    assert np.isfinite(r.jacobian).all()
    assert r.checked in (("edge bias",), ("eps_lo",))
    assert np.isfinite(r.linear_error).all()

    def values(x):
        out = quantities(dict(zip(nominal, x, strict=True)))
        return np.array([float(v) for v in out.values()])

    # Central differences at h and h / 2. Truncation error is c h^2, so
    # FD(h/2) is off by (FD(h) - FD(h/2)) / 3; the bound takes that estimate
    # three times over. The round-off term is d f / h with d f the
    # error of a direct solve of this small system, taken as 1e-13 of
    # quantities of order 1 (100 times the double-precision unit round-off).
    steps = {"edge bias": 2e-6, "eps_lo": 1e-4}
    x0 = np.array(list(nominal.values()))
    for j, name in enumerate(nominal):
        e = np.zeros(2)
        e[j] = 1.0
        h = steps[name]
        coarse = (values(x0 + h * e) - values(x0 - h * e)) / (2 * h)
        fine = (values(x0 + h / 2 * e) - values(x0 - h / 2 * e)) / h
        bound = np.abs(coarse - fine) + 4 * 1e-13 / (h / 2)
        assert (np.abs(r.jacobian[:, j] - fine) <= bound).all(), (name, r.jacobian[:, j], fine)
