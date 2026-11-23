"""Gradients of iterative solves that start from the previous design's solution.

The design initial guess: on one process each ``bddc_gmres`` or ``pmg_gmres``
design's first frequency starts from the previous design's solution there.
Four designs in a row on the small 3D two-layer line
(`gradient_cases.layered_line`), through ``jax.value_and_grad`` at tolerance
1e-6, against the same designs solved at 1e-10 (the reference) and at 1e-6
with no kept state between designs (no guess). The guess saves iterations;
S stays within the tolerance of the reference, and the gradient's relative
difference from it at most twice the guess-free run's (1.5x was
measured on the 1.0M-DOF 3D bend).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradient_cases import layered_line, line_design, s_functional, with_fields

import gfem
from gfem._engine.system_operator import SystemSweep

TOLERANCE = 1e-6
PROBLEM = gfem.Driven(frequency=(20e9, 24e9))


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def _designs(line):
    """Four designs a few percent apart, as an optimizer's early steps."""
    values, rng = line_design(line, seed=5)
    eps = values["eps_r"]
    direction = jnp.asarray(rng.uniform(-1.0, 1.0, eps.shape))
    return values, [eps * (1 + 0.03 * k * direction) for k in range(4)]


def _run(line, values, designs, monkeypatch, *, solver, tolerance, guess):
    """S, gradients and each design's first-frequency iterations.

    Without ``guess`` every design starts from no kept state.
    """
    options = gfem.SolverOptions(order=2, linear_solver=solver, tolerance=tolerance)
    first: list = []
    solve = SystemSweep.solve

    def counted(self, omega, rhs, **kwargs):
        x, its = solve(self, omega, rhs, **kwargs)
        if kwargs.get("x0") is None or kwargs.get("guess") is not None:
            first.append((max(its), kwargs.get("guess")))
        return x, its

    def objective(eps):
        model = with_fields(line, {**values, "eps_r": eps})
        s = gfem.solve(model, PROBLEM, options=options).s.values
        return s_functional(s), s

    step = jax.value_and_grad(objective, has_aux=True)
    out = []
    gfem.release_memory()
    with monkeypatch.context() as patch:
        patch.setattr(SystemSweep, "solve", counted)
        try:
            for eps in designs:
                if not guess:
                    gfem.release_memory()
                (_, s), grad = step(eps)
                out.append((np.asarray(s), np.asarray(grad)))
        finally:
            gfem.release_memory()
    return out, first


def _gradient_error(run, reference):
    """The largest relative gradient difference over the designs after the first.

    The first design is solved alike with and without a guess.
    """
    return max(
        np.linalg.norm(grad - grad_ref) / np.linalg.norm(grad_ref)
        for (_, grad), (_, grad_ref) in zip(run[1:], reference[1:], strict=True)
    )


@pytest.mark.parametrize("solver", ["bddc_gmres", "pmg_gmres"])
def test_a_design_guess_keeps_s_and_the_gradient_within_the_guess_free_band(
    line, monkeypatch, solver
):
    values, designs = _designs(line)
    run = functools.partial(_run, line, values, designs, monkeypatch, solver=solver)
    reference, _ = run(tolerance=1e-10, guess=False)
    plain, cold = run(tolerance=TOLERANCE, guess=False)
    guessed, warm = run(tolerance=TOLERANCE, guess=True)
    # the first design has nothing to start from; the others start from the
    # previous design's solution, in fewer iterations
    assert [g for _, g in cold] == [None] * 4
    assert [g for _, g in warm] == [None] + ["previous design"] * 3
    assert all(w < c for (w, _), (c, _) in zip(warm[1:], cold[1:], strict=True))
    for (s, _), (s_ref, _) in zip(guessed, reference, strict=True):
        assert np.max(np.abs(s - s_ref)) <= TOLERANCE
    assert _gradient_error(guessed, reference) <= 2 * _gradient_error(plain, reference)
