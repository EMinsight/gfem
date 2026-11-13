"""Gradients on direct factors reused across designs, against the exact path.

The gate (`DirectHandle._reuse_pays`) forced on a small 3D line: three nearby
designs in a row through ``jax.value_and_grad`` on the plan pool's base. Each
reused design's S stays within a twentieth of the tolerance of the exact
run's, and its gradient within the tolerance, relative (ADR 0094, amended).
With the rule's cap at 1 every reused column passes it, refactorizes mid-step
and finishes direct: S and the gradient are then the exact run's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_cases import layered_line, line_design, s_functional, with_fields

import gfem
from gfem._engine import cache
from gfem._engine.system_operator import SystemSweep
from gfem._linalg.direct import DirectHandle
from gfem._linalg.factor_reuse import AmortizedRule
from gfem._linalg.pardiso import pardiso_available

TOLERANCE = 1e-6
PROBLEM = gfem.Driven(frequency=24e9)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", tolerance=TOLERANCE)

pytestmark = pytest.mark.skipif(
    not pardiso_available(), reason="reuse is PARDISO's (the mkl extra)"
)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def _run(line, values, designs, monkeypatch, *, forced, cap=100):
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
            # a test-sized R/C gives a cap of a few iterations, set by timing
            patch.setattr(AmortizedRule, "cap", property(lambda _: cap))

        def objective(eps):
            model = with_fields(line, {**values, "eps_r": eps})
            s = gfem.solve(model, PROBLEM, options=OPTIONS).s.values
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


def test_reused_factors_keep_s_and_the_gradient_within_the_tolerance(line, monkeypatch):
    values, rng = line_design(line, seed=3)
    eps = values["eps_r"]
    direction = jnp.asarray(rng.standard_normal(eps.shape))
    designs = [eps + 1e-4 * k * direction for k in range(4)]
    exact, plain = _run(line, values, designs, monkeypatch, forced=False)
    reused, counts = _run(line, values, designs, monkeypatch, forced=True)
    # the gate off: every forward solve is direct
    assert all(its == () for step in plain for its in step)
    # step 0 prepares, step 1 refactorizes (a chain starts); later ones reuse
    assert any(its and min(its) > 0 for step in counts[2:] for its in step)
    for (s, grad), (s_exact, grad_exact) in zip(reused, exact, strict=True):
        assert np.max(np.abs(s - s_exact)) <= 0.05 * TOLERANCE
        error = np.linalg.norm(grad - grad_exact) / np.linalg.norm(grad_exact)
        assert error <= TOLERANCE


def test_a_column_past_its_cap_refactorizes_and_gives_the_exact_gradient(
    line, monkeypatch
):
    values, rng = line_design(line, seed=3)
    eps = values["eps_r"]
    direction = jnp.asarray(rng.standard_normal(eps.shape))
    designs = [eps + 1e-4 * k * direction for k in range(4)]
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
