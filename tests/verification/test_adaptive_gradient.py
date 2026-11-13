"""Gradients through adaptive sweeps: the reduced basis held fixed.

The two-layer line of the uniform-sweep cells (`gradient_cases.layered_line`:
random per-element lossy permittivity and conductivity, second-order
absorbing side walls) swept over 10-30 GHz with a frozen sample set. At a
sample frequency the full solution lies in the reduced basis, so S and its
frozen-basis gradient equal the full solve's to solver tolerance (the
matrix cells and that check are in test_gradient_matrix.py). Between
samples the gradient error follows the reduced model's field error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from gradient_cases import layered_line, line_design, s_functional, tight, with_fields

import gfem
from gfem._engine.cache import ENTRIES
from gfem._linalg.direct import DirectHandle
from gfem._linalg.factor_reuse import (
    FIRST,
    OTHER_FREQUENCY,
    SAME_DESIGN,
    AmortizedRule,
)
from gfem._linalg.pardiso import pardiso_available

GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
SAMPLES = (GRID[0], GRID[20], GRID[10], GRID[4], GRID[16])
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def _s_objective(line, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        s = gfem.solve(
            with_fields(line, {**values, name: p}), problem, options=options
        ).s.values[jnp.asarray(rows)]
        return s_functional(s)

    return objective


def _relative(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# --- between samples: the error follows the field error --------------------------

OFF = [2, 7, 13, 18]


def _field_errors(line, values, options):
    """Relative field error of the reduced model at each off-sample index."""
    model = with_fields(line, values)
    kept = [GRID[i] for i in OFF]
    full = gfem.solve(
        model, gfem.Driven(frequency=kept, keep_fields=True), options=options
    )
    problem = gfem.Driven(
        frequency=GRID, sweep="adaptive", samples=SAMPLES, keep_fields=kept
    )
    reduced = gfem.solve(model, problem, options=options)
    errors = []
    for f in kept:
        worst = 0.0
        for port in ("p1", "p2"):
            x = full.e_field(f, port).coefficients
            y = reduced.e_field(f, port).coefficients
            worst = max(worst, _relative(y, x))
        errors.append(worst)
    return errors


@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_gradient_error_between_samples_follows_the_field_error(line, name):
    values, _ = line_design(line, seed=1)
    options = tight("direct")
    errors = _field_errors(line, values, options)
    for i, error in zip(OFF, errors, strict=True):
        adaptive = _s_objective(line, values, name, options, ADAPTIVE, [i])
        full = _s_objective(
            line, values, name, options, gfem.Driven(frequency=GRID[i]), [0]
        )
        grad = jax.grad(adaptive)(values[name])
        full_grad = jax.grad(full)(values[name])
        assert 0 < error < 1e-1
        assert _relative(grad, full_grad) <= 10 * error


# --- kept fields: their cotangents take the same reduced adjoint as S ------------


def test_a_port_voltage_read_from_a_field_differentiates_like_s(line):
    # Port 2's voltage functional applied to port 1's field is S21 (times
    # sqrt(2 R)): through the kept field's reduced coordinates it must give
    # the S gradient, on and between the samples alike.
    from gfem._engine import terms
    from gfem._engine.driven import driven_operator

    values, _ = line_design(line, seed=4)
    options = tight("direct")
    op = driven_operator(with_fields(line, values), options.order)
    voltage = terms.port_vectors(op.space, op.built, op.order, op.ports[1])[1]
    c = (0.3 - 0.7j) / np.sqrt(2 * 50.0)
    kept = [GRID[4], GRID[7]]  # a sample and a frequency between samples
    problem = gfem.Driven(
        frequency=GRID, sweep="adaptive", samples=SAMPLES, keep_fields=kept
    )

    def from_field(eps):
        result = gfem.solve(
            with_fields(line, {**values, "eps_r": eps}), problem, options=options
        )
        return sum(
            jnp.real(c * jnp.sum(voltage * result.e_field(f, "p1").coefficients))
            for f in kept
        )

    def from_s(eps):
        s = gfem.solve(
            with_fields(line, {**values, "eps_r": eps}), problem, options=options
        ).s
        return jnp.sum(jnp.real((0.3 - 0.7j) * s["p2", "p1"][jnp.asarray([4, 7])]))

    eps = values["eps_r"]
    assert float(from_field(eps)) == pytest.approx(float(from_s(eps)), rel=1e-10)
    np.testing.assert_allclose(
        jax.grad(from_field)(eps), jax.grad(from_s)(eps), rtol=1e-9
    )


# --- transformations and the kept reduced model ----------------------------------
# solve-vs-solve differs by about 1 ulp (threaded assembly)
RTOL = 1e-9


def _eps_objective(line, options):
    values, _ = line_design(line, seed=5)
    return _s_objective(line, values, "eps_r", options, ADAPTIVE, range(len(GRID)))


def _designs(line, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    return jnp.asarray(rng.uniform(2.0, 6.0, (k, n)) * (1 - 0.02j))


def test_the_backward_keeps_and_needs_no_full_size_solver(line, monkeypatch):
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1)[0]
    objective = _eps_objective(line, tight("direct"))
    value, pullback = jax.vjp(objective, eps)
    entry = ENTRIES.get(max(ENTRIES._entries))
    assert entry.handle is None
    assert sorted(entry.fields) == [
        "basis",
        "coordinates",
        "deembedding",
        "functionals",
        "matrices",
    ]

    def refuse(*_, **__):
        raise AssertionError("the frozen-basis backward prepared a full solver")

    monkeypatch.setattr("gfem._linalg.prepare", refuse)
    grad = pullback(jnp.ones_like(value))[0]
    monkeypatch.undo()
    np.testing.assert_allclose(grad, jax.grad(objective)(eps), rtol=RTOL)


def test_vmap_of_grad_equals_a_python_loop(line):
    designs = _designs(line, 2, seed=1)
    grad = jax.grad(_eps_objective(line, tight("direct")))
    looped = jnp.stack([grad(d) for d in designs])
    np.testing.assert_allclose(jax.vmap(grad)(designs), looped, rtol=RTOL)
    np.testing.assert_allclose(jax.jit(jax.vmap(grad))(designs), looped, rtol=RTOL)
    assert not np.allclose(looped[0], looped[1], rtol=1e-3)


def test_pullbacks_twice_and_after_eviction_equal_fresh_gradients(line):
    from gfem._engine.cache import ENTRIES

    eps = _designs(line, 1, seed=2)[0]
    objective = _eps_objective(line, tight("direct"))
    fresh = jax.grad(objective)(eps)
    _, pullback = jax.vjp(objective, eps)
    np.testing.assert_allclose(pullback(1.0)[0], fresh, rtol=RTOL)
    np.testing.assert_allclose(pullback(2.0)[0], 2 * fresh, rtol=RTOL)
    ENTRIES.clear()
    np.testing.assert_allclose(pullback(1.0)[0], fresh, rtol=RTOL)
    options = gfem.SolverOptions(
        order=2, linear_solver="direct", tolerance=1e-12, retain_memory=0
    )
    recompute = jax.grad(_eps_objective(line, options))
    np.testing.assert_allclose(recompute(eps), fresh, rtol=RTOL)


# --- factors reused across designs at each sample (ADR 0121) ---------------------

REUSE_TOLERANCE = 1e-6
SAMPLE_ROWS = [GRID.index(f) for f in SAMPLES]


def _reuse_run(line, monkeypatch, *, forced, retain_memory=None):
    """S at the samples, gradients and each design's handle steps, 4 designs.

    A step is ``(frequency, action, reason)``. The designs move by 1e-4 of a
    random direction, so a reused solve converges in a few iterations.
    """
    values, rng = line_design(line, seed=3)
    direction = jnp.asarray(rng.standard_normal(values["eps_r"].shape))
    options = gfem.SolverOptions(
        order=2,
        linear_solver="direct",
        tolerance=REUSE_TOLERANCE,
        retain_memory=retain_memory,
    )
    rows = jnp.asarray(SAMPLE_ROWS)

    def objective(eps):
        model = with_fields(line, {**values, "eps_r": eps})
        s = gfem.solve(model, ADAPTIVE, options=options).s.values[rows]
        return s_functional(s), s

    steps: list = []
    stepped = DirectHandle._stepped

    def recorded(self, action, reason):
        steps[-1].append((self._frequency, action, reason))
        stepped(self, action, reason)

    out = []
    with monkeypatch.context() as patch:
        patch.setattr(DirectHandle, "_stepped", recorded)
        if forced:
            patch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
            patch.setattr(AmortizedRule, "cap", property(lambda _: 100))
        step = jax.value_and_grad(objective, has_aux=True)
        ENTRIES.clear()
        try:
            for k in range(4):
                steps.append([])
                (_, s), grad = step(values["eps_r"] + 1e-4 * k * direction)
                out.append((np.asarray(s), np.asarray(grad)))
        finally:
            ENTRIES.clear()
    return out, steps


@pytest.mark.skipif(not pardiso_available(), reason="PARDISO's reuse (the mkl extra)")
def test_each_frozen_sample_reuses_its_own_factors_across_designs(line, monkeypatch):
    exact, _ = _reuse_run(line, monkeypatch, forced=False, retain_memory=0)
    reused, steps = _reuse_run(line, monkeypatch, forced=True)
    omegas = sorted(2 * np.pi * f for f in SAMPLES)
    # design 0 meets each sample for the first time: one base, refactorized
    # from sample to sample; a sample met again gets its own base (ADR 0121,
    # amended), whose chain starts at its next refactorization; by design 3
    # every sample reuses its own, and a backward's steps refactorize nothing
    assert not [step for step in steps[0] if step[1] == "reused"]
    assert {step[2] for step in steps[0]} <= {FIRST, OTHER_FREQUENCY}
    last = steps[3]
    assert sorted(step[0] for step in last if step[1] == "reused") == omegas
    others = {step[1:] for step in last if step[1] != "reused"}
    assert others <= {("held", SAME_DESIGN)}
    for (s, grad), (s_exact, grad_exact) in zip(reused, exact, strict=True):
        assert np.max(np.abs(s - s_exact)) <= 0.05 * REUSE_TOLERANCE
        error = np.linalg.norm(grad - grad_exact) / np.linalg.norm(grad_exact)
        assert error <= REUSE_TOLERANCE


# --- a port that is not excited: its S functional joins the basis ----------------

LUMPED_GRID = np.linspace(1e9, 20e9, 58)
LUMPED_AT = [0, 57, 30]


@pytest.fixture(scope="module")
def lumped_line(tmp_path_factory):
    path = tmp_path_factory.mktemp("lumped") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.mark.parametrize("excite", [("p1",), ("p2",)])
def test_an_unexcited_lumped_port_row_is_exact_at_the_samples(lumped_line, excite):
    # the adjoint of the unexcited port's row is not in the span of the
    # excited solutions: each sample also solves that port's S functional
    options = gfem.SolverOptions(linear_solver="direct")
    n = len(lumped_line.mesh.tetrahedra)
    rng = np.random.default_rng(1)
    p0 = jnp.asarray(rng.uniform(3.0, 5.0, n))
    d = rng.standard_normal(n)
    d /= np.linalg.norm(d)
    frozen = gfem.Driven(
        frequency=LUMPED_GRID,
        sweep="adaptive",
        samples=[LUMPED_GRID[i] for i in LUMPED_AT],
        excite=excite,
    )
    uniform = gfem.Driven(frequency=LUMPED_GRID[LUMPED_AT], excite=excite)

    def jacobian(problem, rows):
        def s_column(p):
            field = gfem.MaterialField(eps_r=p * (1 - 1e-3j))
            model = lumped_line.with_material("dielectric", field)
            s = gfem.solve(model, problem, options=options).s.values
            s = s[jnp.asarray(rows)]
            return jnp.stack([s.real, s.imag])

        # one forward and its tangent along d: every entry's directional gradient
        return np.asarray(jax.jacrev(lambda t: s_column(p0 + t * d))(0.0))

    got = jacobian(frozen, LUMPED_AT)
    want = jacobian(uniform, [0, 1, 2])
    assert got.shape == (2, 3, 2, 1)
    np.testing.assert_allclose(got, want, rtol=1e-9)
