"""gfem's CG: NGSolve's ``CGSolver`` iteration on the layout's products."""

import ngsolve as ngs
import numpy as np
import pytest
from netgen.geom2d import unit_square

from gfem._linalg.cg import Cg

pytestmark = pytest.mark.usefixtures("ngsolve_threads")


@pytest.fixture(scope="module")
def poisson():
    """A real Poisson problem on H1 order 3 with a Jacobi preconditioner."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.1))
    space = ngs.H1(mesh, order=3, dirichlet="left|bottom")
    u, v = space.TnT()
    form = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx + u * v * ngs.dx)
    pre = ngs.Preconditioner(form, "local")
    form.Assemble()
    rhs = ngs.LinearForm(ngs.x * v * ngs.dx).Assemble()
    return form.mat, pre.mat, rhs.vec


def _solve(solver, rhs):
    sol = rhs.CreateVector()
    solver.Solve(rhs=rhs, sol=sol)
    return np.array(sol.FV().NumPy()), solver.iterations, list(solver.residuals)


@pytest.mark.parametrize("stop", [{"tol": 1e-10}, {"atol": 1e-9}])
def test_cg_matches_ngsolve_cg(poisson, stop):
    mat, pre, rhs = poisson
    ours = _solve(Cg(mat, pre, maxiter=500, **stop), rhs)
    theirs = _solve(ngs.solvers.CGSolver(mat, pre, maxiter=500, **stop), rhs)
    assert ours[1] == theirs[1] > 10
    # late residuals are square roots of products that cancel: summation order shows
    np.testing.assert_allclose(ours[2], theirs[2], rtol=0, atol=1e-7 * theirs[2][0])
    scale = np.abs(theirs[0]).max()
    # both stop at the tolerance; their round-off differs
    assert np.abs(ours[0] - theirs[0]).max() <= 1e-8 * scale


def test_cg_stops_at_maxiter(poisson):
    mat, pre, rhs = poisson
    _, iterations, _ = _solve(Cg(mat, pre, tol=1e-14, maxiter=3), rhs)
    assert iterations == 3
