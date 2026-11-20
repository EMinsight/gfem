"""A permittivity law across ranks: driven S and the law pass.

The `poles` law of the gradient matrix's cell D1 (Drude and Lorentz terms, Re eps
< 0 at its first frequency) on the core of `gradient_cases.dispersive_line`, on two
ranks against one process: S to `S_GATE`, and the gradient in the law's normalized
parameters to `GRADIENT_GATE`, with differences on the ranks. The law pass's
per-pair integrals are each rank's elements' share, summed over the ranks with the
other passes' (`_adjoint.solve_vjp`), so a lost sum shows as a gradient off by
about half. BDDC runs on either stack; `direct` is PETSc MUMPS, on the MPI stack
only.
"""

from pathlib import Path

import gradient_cases as cases
import jax
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from ranks import on_ranks

import gfem
from gfem import _petsc

#: Relative to the largest entry of the one-process value.
S_GATE = 1e-12
GRADIENT_GATE = 1e-10
RANKS = 2
SOLVERS = (
    "bddc_gmres",
    pytest.param(
        "direct",
        marks=pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack"),
    ),
)
LAW = cases.LAWS["poles"]
FREQUENCIES = cases.LAW_FREQUENCIES["poles"]


def _s(directory, solver):
    """S of the strip line with the `poles` law on its core, one thread a rank."""
    model = cases.dispersive_line(cases.dispersive_mesh(Path(directory)), LAW)
    problem = gfem.Driven(frequency=list(FREQUENCIES))
    options = cases.tight(solver, threads=1)
    return np.asarray(gfem.solve(model, problem, options=options).s.values)


def _gradient(directory, solver, *, differences=True):
    """The value and the gradient in the law's normalized parameters p0 (1 + t)."""
    mesh = cases.dispersive_mesh(Path(directory))
    problem = gfem.Driven(frequency=list(FREQUENCIES))
    options = cases.tight(solver, threads=1)

    def objective(t):
        model = cases.dispersive_line(mesh, cases.scaled_law(LAW, t))
        return cases.s_functional(gfem.solve(model, problem, options=options).s.values)

    t = np.zeros(len(jax.tree.leaves(LAW)))
    value, grad = jax.value_and_grad(objective)(t)
    grad = np.asarray(grad)
    if differences:
        directions = unit_directions(np.random.default_rng(13), t.shape, 1)
        assert_gradient_matches(objective, t, directions, step=1e-3, grad=grad)
    return np.asarray(value), grad


def _relative(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max())


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_law_models_s_on_two_ranks_matches_one_process(tmp_path, solver):
    one = _s(tmp_path, "direct")
    s, *_ = on_ranks(_s, tmp_path, solver, ranks=RANKS, limit=300)
    error = _relative(s, one)
    print(f"{solver} ranks={RANKS} S error={error:.3g}")  # noqa: T201
    assert error < S_GATE, f"S off by {error:.3g} relative"


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_law_parameter_gradient_on_two_ranks_matches_one_process(tmp_path, solver):
    _, one = _gradient(tmp_path, "direct", differences=False)
    (_, grad), *_ = on_ranks(_gradient, tmp_path, solver, ranks=RANKS, limit=600)
    assert np.isfinite(grad).all()
    error = _relative(grad, one)
    print(f"{solver} ranks={RANKS} law gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"law gradient off by {error:.3g} relative"
