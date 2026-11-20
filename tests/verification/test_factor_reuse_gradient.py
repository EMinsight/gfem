"""Gradients on direct factors reused across designs, against the exact path.

The gate (`DirectHandle._reuse_pays`) forced on a small 3D line: three nearby
designs in a row through ``jax.value_and_grad`` on the plan pool's base. Each
reused design's S stays within a twentieth of the tolerance of the exact
run's, and its gradient within the tolerance, relative.
With the rule's cap at 1 every reused column passes it, refactorizes mid-step
and finishes direct: S and the gradient are then the exact run's. The same
holds on sparse Cholesky, the factorization without MKL, and a script run
twice at one thread reuses at the same designs (the rule reads counts).

Over several frequencies each frequency gets its own base the second time
the sweep meets it: from then on every frequency's
forward reuses, and a backward that solves finds each base holding its
design's values and refactorizes nothing (``held``, ``same design``).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_cases import layered_line, line_design, s_functional, with_fields

import gfem
from gfem._engine import cache
from gfem._engine.system_operator import SystemSweep
from gfem._linalg import direct
from gfem._linalg.direct import DirectHandle
from gfem._linalg.factor_reuse import (
    FIRST,
    OTHER_FREQUENCY,
    SAME_DESIGN,
    AmortizedRule,
)
from gfem._linalg.pardiso import pardiso_available

TOLERANCE = 1e-6
PROBLEM = gfem.Driven(frequency=24e9)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", tolerance=TOLERANCE)

needs_pardiso = pytest.mark.skipif(
    not pardiso_available(), reason="PARDISO's reuse (the mkl extra)"
)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def _designs(line):
    values, rng = line_design(line, seed=3)
    eps = values["eps_r"]
    direction = jnp.asarray(rng.standard_normal(eps.shape))
    return values, [eps + 1e-4 * k * direction for k in range(4)]


def _run(line, values, designs, monkeypatch, *, forced, cap=100, options=OPTIONS):
    """S, gradients and each design's column iterations, on a fresh pool.

    ``cap`` replaces the rule's cap on a reused column's iterations when the
    gate is ``forced``.
    """
    counts: list = []
    solve = SystemSweep.solve

    def counted(self, *args, **kwargs):
        x, its = solve(self, *args, **kwargs)
        counts[-1].append(its)
        return x, its

    with monkeypatch.context() as patch:
        patch.setattr(SystemSweep, "solve", counted)
        if forced:
            patch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
            # a test-sized R/C gives a cap of a few iterations
            patch.setattr(AmortizedRule, "cap", property(lambda _: cap))

        def objective(eps):
            model = with_fields(line, {**values, "eps_r": eps})
            s = gfem.solve(model, PROBLEM, options=options).s.values
            return s_functional(s), s

        step = jax.value_and_grad(objective, has_aux=True)
        cache.ENTRIES.clear()
        out = []
        try:
            for eps in designs:
                counts.append([])
                (_, s), grad = step(eps)
                out.append((np.asarray(s), np.asarray(grad)))
        finally:
            cache.ENTRIES.clear()
    return out, counts


def _reused(counts):
    """Whether each design's forward solves ran on reused factors."""
    return [any(its and min(its) > 0 for its in step) for step in counts]


def _within_bounds(reused, exact):
    for (s, grad), (s_exact, grad_exact) in zip(reused, exact, strict=True):
        assert np.max(np.abs(s - s_exact)) <= 0.05 * TOLERANCE
        error = np.linalg.norm(grad - grad_exact) / np.linalg.norm(grad_exact)
        assert error <= TOLERANCE


@needs_pardiso
def test_reused_factors_keep_s_and_the_gradient_within_the_tolerance(line, monkeypatch):
    values, designs = _designs(line)
    exact, plain = _run(line, values, designs, monkeypatch, forced=False)
    reused, counts = _run(line, values, designs, monkeypatch, forced=True)
    # the gate off: every forward solve is direct
    assert all(its == () for step in plain for its in step)
    # step 0 prepares, step 1 refactorizes (a chain starts); later ones reuse
    assert any(_reused(counts[2:]))
    _within_bounds(reused, exact)


@needs_pardiso
def test_a_column_past_its_cap_refactorizes_and_gives_the_exact_gradient(
    line, monkeypatch
):
    values, designs = _designs(line)
    exact, _ = _run(line, values, designs, monkeypatch, forced=False)
    capped, counts = _run(line, values, designs, monkeypatch, forced=True, cap=1)
    # step 0 prepares, step 1 refactorizes (a chain starts); from then on the
    # first column stops at its cap and refactorizes, the second is direct
    assert counts[:2] == [[()], [()]]
    assert counts[2:] == [[(1, 0)], [(1, 0)]]
    for (s, grad), (s_exact, grad_exact) in zip(capped, exact, strict=True):
        assert np.max(np.abs(s - s_exact)) <= 1e-12 * np.max(np.abs(s_exact))
        error = np.linalg.norm(grad - grad_exact) / np.linalg.norm(grad_exact)
        assert error <= 1e-12


def test_sparse_cholesky_reuses_factors_within_the_tolerance(line, monkeypatch):
    # the factorization without the mkl extra
    monkeypatch.setattr(direct, "pardiso_inverse", lambda *_, **__: None)
    values, designs = _designs(line)
    keep_nothing = dataclasses.replace(OPTIONS, retain_memory=0)
    exact, _ = _run(
        line, values, designs, monkeypatch, forced=False, options=keep_nothing
    )
    reused, counts = _run(line, values, designs, monkeypatch, forced=True)
    assert any(_reused(counts[2:]))
    _within_bounds(reused, exact)


@needs_pardiso
def test_a_script_run_twice_at_one_thread_reuses_at_the_same_designs(line, monkeypatch):
    values, designs = _designs(line)
    one = dataclasses.replace(OPTIONS, threads=1)
    exact, _ = _run(line, values, designs, monkeypatch, forced=False, options=one)
    first, counts = _run(line, values, designs, monkeypatch, forced=True, options=one)
    again, recount = _run(line, values, designs, monkeypatch, forced=True, options=one)
    assert any(_reused(counts))
    assert _reused(recount) == _reused(counts)
    assert recount == counts
    _within_bounds(first, exact)
    _within_bounds(again, exact)


FREQUENCIES = (16e9, 20e9, 24e9)


def _run_frequencies(line, values, designs, monkeypatch, *, forced):
    """S, gradients and, per design, the forward's and backward's handle steps.

    The objective adds a field term at the line's centre, so each backward
    solves. A step is ``(frequency, action, reason)``.
    """
    problem = gfem.Driven(frequency=FREQUENCIES, keep_fields=True)
    point = np.asarray(line.mesh.nodes).mean(axis=0)[None, :]
    steps: list = []
    stepped = DirectHandle._stepped

    def recorded(self, action, reason):
        steps[-1].append((self._frequency, action, reason))
        stepped(self, action, reason)

    def objective(eps):
        result = gfem.solve(
            with_fields(line, {**values, "eps_r": eps}), problem, options=OPTIONS
        )
        value = s_functional(result.s.values)
        for f in FREQUENCIES:
            field = result.e_field(f, line.port_names[0]).at(point)
            value = value + 1e-20 * jnp.sum(jnp.abs(field) ** 2)
        return value, result.s.values

    out, forward, backward = [], [], []
    with monkeypatch.context() as patch:
        patch.setattr(DirectHandle, "_stepped", recorded)
        if forced:
            patch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
            patch.setattr(AmortizedRule, "cap", property(lambda _: 100))
        cache.ENTRIES.clear()
        try:
            for eps in designs:
                steps.append([])
                _, pullback, s = jax.vjp(objective, eps, has_aux=True)
                forward.append(steps[-1])
                steps.append([])
                (grad,) = pullback(1.0)
                backward.append(steps[-1])
                out.append((np.asarray(s), np.asarray(grad)))
        finally:
            cache.ENTRIES.clear()
    return out, forward, backward


@needs_pardiso
def test_each_frequency_reuses_its_own_factors_and_a_backward_refactorizes_nothing(
    line, monkeypatch
):
    values, designs = _designs(line)
    exact, _, _ = _run_frequencies(line, values, designs, monkeypatch, forced=False)
    reused, forward, backward = _run_frequencies(
        line, values, designs, monkeypatch, forced=True
    )
    omegas = [2 * np.pi * f for f in FREQUENCIES]
    # design 0 meets each frequency for the first time, its backward too (the
    # same visit): one base, refactorized from frequency to frequency
    assert forward[0] == [
        (omegas[1], "refactorized", FIRST),
        (omegas[2], "refactorized", OTHER_FREQUENCY),
    ]
    assert backward[0] == [
        (omegas[1], "refactorized", OTHER_FREQUENCY),
        (omegas[0], "refactorized", OTHER_FREQUENCY),
    ]
    # design 1 plans the other frequencies' bases (a new handle steps
    # nothing); from then on each base stays at its frequency, and every
    # frequency reuses its own factors
    for steps in forward[1:]:
        frequencies = [step[0] for step in steps]
        assert frequencies == [omega for omega in omegas if omega in frequencies]
        assert all(step[2] != OTHER_FREQUENCY for step in steps)
    reusing = {step[0] for steps in forward[1:] for step in steps if step[2] is None}
    assert reusing == set(omegas)
    assert [step[0] for step in forward[-1]] == omegas
    # from the last frequency down: the forward left the base there, and each
    # other base holds this design's values
    held = [(omegas[1], "held", SAME_DESIGN), (omegas[0], "held", SAME_DESIGN)]
    assert backward[1:] == [held] * (len(designs) - 1)
    _within_bounds(reused, exact)
