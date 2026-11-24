"""Thermal and DC conduction direct solves take their handles from the plan pool.

A serial CPU direct solve of a steady or transient thermal problem, or of a
DC conduction problem, refactorizes a pooled handle of the same operator
instead of planning again, as a driven solve does. A design that leaves the
operator unchanged (a heat-source amount, the drive) solves on the held
factors. ``retain_memory=0`` keeps nothing pooled and gives the same bits.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import thermal_cases as tc

import gfem
from gfem._engine import cache
from gfem._linalg.direct import DirectHandle, UpdateFactors
from gfem._linalg.factor_reuse import AmortizedRule
from gfem._linalg.pardiso import PardisoFactors

LOGGER = "gfem.solver"
W, D = 2.0, 1.0
LAYERS = (("clad", 2.0), ("core", 1.0))
# One solve thread: threaded assembly and reductions sum in a varying order, so
# the pooled-versus-plain comparisons below hold to the last bit only on one.
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)
NO_POOL = gfem.SolverOptions(
    order=2, linear_solver="direct", retain_memory=0, threads=1
)


@pytest.fixture(scope="module")
def brick(tmp_path_factory):
    def at(z):
        return lambda p: abs(p[2] - z) < 1e-9

    faces = [(at(0.0), "bottom"), (at(2.0), "clad_top"), (at(3.0), "top")]
    path = tmp_path_factory.mktemp("pool") / "brick.msh"
    return tc.brick(path, (W, D), LAYERS, 0.8, faces)


@pytest.fixture
def records(caplog):
    gfem.release_memory()
    caplog.set_level(logging.INFO, logger=LOGGER)

    def read(problem=None, direction=None):
        return [
            r
            for r in caplog.records
            if r.name == LOGGER
            and hasattr(r, "direction")
            and (problem is None or r.problem == problem)
            and (direction is None or r.direction == direction)
        ]

    yield read
    gfem.release_memory()


def _model(mesh, k, boundaries):
    return gfem.Model(
        mesh,
        materials={
            "clad": gfem.MaterialField(
                thermal_conductivity=k[mesh.elements("clad")],
                mass_density=jnp.full(len(mesh.elements("clad")), 2200.0),
                specific_heat=jnp.full(len(mesh.elements("clad")), 700.0),
            ),
            "core": gfem.Material(
                thermal_conductivity=2.0, mass_density=2200.0, specific_heat=700.0
            ),
        },
        boundaries=boundaries,
    )


EQUAL = [gfem.FixedTemperature("bottom", temperature=300.0)]
UNEQUAL = [
    gfem.FixedTemperature("bottom", temperature=300.0),
    gfem.FixedTemperature("top", temperature=320.0),
]
HEAT = gfem.HeatSource.uniform("core", power_density=1e11)


def _design(mesh, i):
    n = len(mesh.elements("clad"))
    return jnp.linspace(1.0, 1.0 + 0.1 * (i + 1), n)


def _loop(mesh, options, boundaries=EQUAL, designs=3):
    def objective(k):
        result = gfem.solve(
            _model(mesh, k, boundaries), gfem.SteadyThermal(HEAT), options=options
        )
        return jnp.sum(result.temperature.coefficients**2) * 1e-6

    out = []
    for i in range(designs):
        value, grad = jax.value_and_grad(objective)(_design(mesh, i))
        out.append((np.asarray(value), np.asarray(grad)))
    return out


def test_a_steady_design_loop_refactorizes_from_design_two_and_matches_no_pool(
    brick, records
):
    pooled = _loop(brick, DIRECT)
    actions = [r.action for r in records("steady thermal", "forward")]
    assert actions[0] == "factorized"
    assert actions[1:] == ["refactorized"] * 2
    gfem.release_memory()
    plain = _loop(brick, NO_POOL)
    for (v, g), (v0, g0) in zip(pooled, plain, strict=True):
        assert np.array_equal(v, v0)
        assert np.array_equal(g, g0)


def _source_loop(mesh, options, designs=3):
    """Designs that change the heat-source amount only: the operator stays."""
    model = _model(mesh, jnp.full(len(mesh.elements("clad")), 1.3), EQUAL)

    def objective(amount):
        heat = gfem.HeatSource.uniform("core", power_density=amount)
        result = gfem.solve(model, gfem.SteadyThermal(heat), options=options)
        return jnp.sum(result.temperature.coefficients**2) * 1e-6

    return [
        tuple(
            np.asarray(v)
            for v in jax.value_and_grad(objective)(jnp.asarray(1e11 * (i + 1.0)))
        )
        for i in range(designs)
    ]


def test_a_source_only_steady_loop_holds_the_factors_and_matches_no_pool(
    brick, records
):
    pooled = _source_loop(brick, DIRECT)
    actions = [r.action for r in records("steady thermal", "forward")]
    assert actions == ["factorized", "held", "held"]
    gfem.release_memory()
    plain = _source_loop(brick, NO_POOL)
    for a, b in zip(pooled, plain, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_a_forced_gate_reuses_the_factors_within_the_tolerance(
    brick, records, monkeypatch
):
    exact = _loop(brick, NO_POOL)
    before = len(records("steady thermal", "forward"))
    gfem.release_memory()
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
    # a test-sized R/C gives a cap of a few iterations
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 100))
    reused = _loop(brick, DIRECT)
    actions = [r.action for r in records("steady thermal", "forward")][before:]
    # the second design starts the chain, the third reuses it
    assert actions[:2] == ["factorized", "refactorized"]
    assert actions[2] == "reused"
    for (v, g), (v0, g0) in zip(reused, exact, strict=True):
        assert v == pytest.approx(v0, rel=1e-5)
        np.testing.assert_allclose(g, g0, rtol=1e-4, atol=1e-6 * np.abs(g0).max())


def test_unequal_boundaries_and_a_k_gradient_prepare_the_steady_operator_once(
    brick, monkeypatch
):
    prepared = []
    prepare = gfem._linalg.ResolvedLinearSolver.prepare

    def counted(self, *args, **kwargs):
        prepared.append(1)
        return prepare(self, *args, **kwargs)

    monkeypatch.setattr(gfem._linalg.ResolvedLinearSolver, "prepare", counted)
    gfem.release_memory()
    _loop(brick, DIRECT, UNEQUAL, designs=1)
    assert len(prepared) == 1


def test_a_pooled_handle_leaves_with_release_memory_and_retain_memory_zero(brick):
    model = _model(brick, jnp.full(len(brick.elements("clad")), 1.3), EQUAL)
    problem = gfem.SteadyThermal(HEAT)
    gfem.release_memory()
    gfem.solve(model, problem, options=DIRECT)
    assert cache.ENTRIES.pooled == 1
    gfem.release_memory()
    assert cache.ENTRIES.pooled == 0
    gfem.solve(model, problem, options=NO_POOL)
    assert cache.ENTRIES.pooled == 0
    _loop(brick, NO_POOL, designs=2)
    gfem.solve(model, problem, options=NO_POOL)
    assert cache.ENTRIES.pooled == 0


def test_a_differentiated_thermal_solve_names_its_problem_in_the_timing_records(
    brick, caplog
):
    caplog.set_level(logging.INFO, logger="gfem.timing")
    _loop(brick, DIRECT, designs=1)
    timing = [r for r in caplog.records if r.name == "gfem.timing"]
    assert timing
    assert {r.problem for r in timing} == {"steady_thermal"}


# --- DC conduction ------------------------------------------------------------


def _dc_loop(mesh, options, *, design, designs=3):
    n = len(mesh.elements("clad"))

    def objective(sigma):
        model = gfem.Model(
            mesh,
            materials={
                "clad": gfem.MaterialField(conductivity=sigma),
                "core": gfem.Material(conductivity=2.0),
            },
        )
        problem = gfem.DCConduction(terminals={"a": "top"}, ground="bottom")
        result = gfem.solve(model, problem, options=options)
        return jnp.sum(result.conductance) * 1e6

    out = []
    for i in range(designs):
        sigma = jnp.linspace(1.0, 1.0 + 0.1 * (i + 1) * design, n)
        out.append(tuple(np.asarray(v) for v in jax.value_and_grad(objective)(sigma)))
    return out


def test_a_dc_design_loop_refactorizes_and_an_unchanged_operator_is_held(
    brick, records, caplog
):
    for design, expected in ((1, ["factorized"] + ["refactorized"] * 2), (0, None)):
        gfem.release_memory()
        caplog.clear()
        pooled = _dc_loop(brick, DIRECT, design=design)
        forward = [r.action for r in records(direction="forward")]
        assert forward == (expected or ["factorized", "held", "held"])
        gfem.release_memory()
        plain = _dc_loop(brick, NO_POOL, design=design)
        for a, b in zip(pooled, plain, strict=True):
            assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


# --- transient ----------------------------------------------------------------


def _transient(mesh, options, boundaries=UNEQUAL, designs=3):
    def objective(k):
        problem = gfem.TransientThermal(
            HEAT,
            duration=1e-6,
            steps=3,
            initial_temperature=310.0,
            keep_fields=True,
        )
        result = gfem.solve(_model(mesh, k, boundaries), problem, options=options)
        field = result.temperature(1e-6).coefficients
        return jnp.sum(field**2) * 1e-6

    return [
        tuple(np.asarray(v) for v in jax.value_and_grad(objective)(_design(mesh, i)))
        for i in range(designs)
    ]


def test_a_transient_loop_takes_its_t0_operator_from_the_pool(brick, records):
    pooled = _transient(brick, DIRECT)
    backward = [r.action for r in records("transient thermal", "backward")]
    # per design: the stepping solves, then the T0 pair, which finds the
    # forward's source-free handle (equal values) in the pool
    designs = [backward[i : i + 7] for i in range(0, len(backward), 7)]
    assert len(designs) == 3
    assert [d[-1] for d in designs] == ["held"] * 3
    gfem.release_memory()
    plain = _transient(brick, NO_POOL)
    for a, b in zip(pooled, plain, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_the_gate_stays_closed_for_a_stepping_handle_above_the_floor(
    brick, records, monkeypatch
):
    for factors in (UpdateFactors, PardisoFactors):
        monkeypatch.setattr(factors, "above_floor", lambda _: True)
    _transient(brick, DIRECT)
    actions = [r.action for r in records("transient thermal", "forward")]
    assert actions.count("refactorized") >= 1
    assert not [a for a in actions if a.startswith("reused")]


# --- traced boundary values ---------------------------------------------------


def _traced_boundaries(top):
    return [
        gfem.FixedTemperature("bottom", temperature=300.0),
        gfem.FixedTemperature("top", temperature=top),
    ]


def test_traced_boundary_values_and_a_k_gradient_prepare_the_steady_operator_once(
    brick, monkeypatch
):
    prepared = []
    prepare = gfem._linalg.ResolvedLinearSolver.prepare

    def counted(self, *args, **kwargs):
        prepared.append(1)
        return prepare(self, *args, **kwargs)

    monkeypatch.setattr(gfem._linalg.ResolvedLinearSolver, "prepare", counted)
    gfem.release_memory()

    def objective(k, top):
        model = _model(brick, k, _traced_boundaries(top))
        result = gfem.solve(model, gfem.SteadyThermal(HEAT), options=DIRECT)
        return jnp.sum(result.temperature.coefficients**2) * 1e-6

    gk, gt = jax.grad(objective, argnums=(0, 1))(_design(brick, 0), jnp.asarray(320.0))
    assert np.abs(gk).max() > 0
    assert abs(float(gt)) > 0
    assert len(prepared) == 1


def test_a_transient_with_traced_boundary_values_holds_the_t0_operator(brick, records):
    def objective(k, top):
        problem = gfem.TransientThermal(
            HEAT, duration=1e-6, steps=3, initial_temperature=310.0, keep_fields=True
        )
        model = _model(brick, k, _traced_boundaries(top))
        result = gfem.solve(model, problem, options=DIRECT)
        return jnp.sum(result.temperature(1e-6).coefficients ** 2) * 1e-6

    for i in range(2):
        jax.grad(objective, argnums=(0, 1))(_design(brick, i), jnp.asarray(320.0))
    backward = [r.action for r in records("transient thermal", "backward")]
    per_design = len(backward) // 2
    # the T0 pair and the boundary adjoint come last in a design's backward,
    # and both find the source-free handle the forward returned to the pool
    assert backward[per_design - 2 : per_design] == ["held", "held"]
    assert backward[-2:] == ["held", "held"]


def test_a_pooled_handle_taken_without_a_tolerance_solves_at_the_options(monkeypatch):
    # a pooled handle keeps its last user's tolerance (a backward's, maybe
    # looser than the forward's): a caller that names none gets the options'
    from types import SimpleNamespace

    from gfem._engine import scalar_plans

    held = SimpleNamespace(tolerance=1e-3)
    monkeypatch.setattr(scalar_plans, "Refactorable", object)
    monkeypatch.setattr(scalar_plans.ENTRIES, "take_plan", lambda _: held)
    monkeypatch.setattr(
        scalar_plans, "refactorize_or_prepare", lambda pooled, *_, **__: pooled
    )
    linear = SimpleNamespace(tolerance=1e-9)
    site = scalar_plans.PlanSite(("key",), None, 0)
    scalar_plans.prepare_pooled(linear, site, None)
    assert held.tolerance == 1e-9
    scalar_plans.prepare_pooled(linear, site, None, tolerance=1e-5)
    assert held.tolerance == 1e-5
