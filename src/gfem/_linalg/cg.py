r"""gfem's conjugate gradients: NGSolve's ``CGSolver`` on the layout's products.

`Cg` is NGSolve 6.2.2608's preconditioned CG (the ``tol``/``atol`` test of
``CheckResidual`` on the square root of :math:`|w^\mathsf{H} d|`, :math:`w` the
preconditioned residual :math:`d`) with its two inner products taken by gfem.

Under ranks the vectors keep NGSolve's statuses: the residual :math:`d` and the
operator's product :math:`A s` are loads, the preconditioned residual :math:`w` and
the search direction :math:`s` are states. Each product pairs a state with a load,
so it is the rank sum of the local products, reduced to rank 0 and broadcast
(`gfem._mpi.reduce_bcast`): every rank reads the same :math:`\alpha`, :math:`\beta`
and residual, and stops at the same iteration (ADR 0058, ADR 0500). NGSolve's own
``InnerProduct`` is an ``MPI_Allreduce``, which MPI does not promise to hold bitwise
across ranks. A one-process handle keeps NGSolve's ``CGSolver``.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import ngsolve as ngs
import numpy as np
from ngsolve.krylovspace import LinearSolver

from gfem import _mpi
from gfem._linalg.layout import ONE_PROCESS, DofLayout

__all__ = ["Cg"]


class Cg(LinearSolver):
    """NGSolve's preconditioned CG, its inner products agreed across ranks.

    ``Solve``, ``residuals``, ``iterations``, ``maxiter`` and ``tol``/``atol``
    (read on every solve) are NGSolve's ``LinearSolver``'s.
    """

    name = "CG"

    def __init__(
        self,
        mat: ngs.BaseMatrix,
        pre: ngs.BaseMatrix,
        *,
        tol: float | None = None,
        atol: float | None = None,
        maxiter: int = 100,
        conjugate: bool = False,
        layout: DofLayout = ONE_PROCESS,
    ) -> None:
        """Set up the solver; ``conjugate`` as ``CGSolver``'s (Hermitian products)."""
        # the stub types pre as a Preconditioner and tol/atol as float; both work
        super().__init__(mat, pre, tol=tol, atol=atol, maxiter=maxiter)  # ty: ignore[invalid-argument-type]
        self.conjugate = conjugate
        self._layout = layout

    def _product(self, state: ngs.BaseVector, load: ngs.BaseVector) -> Any:
        """``state . load`` (``state`` conjugated with ``conjugate``), agreed."""
        comm = self._layout.comm
        if comm is not None:
            # no-ops on vectors already in these statuses, as NGSolve leaves them
            state.Cumulate()
            load.Distribute()
        a, b = state.FV().NumPy(), load.FV().NumPy()
        local = np.vdot(a, b) if self.conjugate else a @ b
        value = _mpi.reduce_bcast(comm, np.atleast_1d(local))[0]
        return complex(value) if np.iscomplexobj(value) else float(value)

    def _SolveImpl(  # noqa: N802 (NGSolve's name)
        self, rhs: ngs.BaseVector, sol: ngs.BaseVector
    ) -> None:
        """NGSolve 6.2.2608's ``CGSolver._SolveImpl``, with gfem's products."""
        A, pre = self.mat, self.pre  # ty: ignore[unresolved-attribute]
        d, w, s = (sol.CreateVector() for _ in range(3))
        d.data = rhs - A * sol
        w.data = pre * d
        s.data = w
        wdn = self._product(w, d)
        if self.CheckResidual(sqrt(abs(wdn))):
            return
        while True:
            w.data = A * s
            wd = wdn
            as_s = self._product(s, w)
            if as_s == 0 or wd == 0:
                break
            alpha = wd / as_s
            sol.data += alpha * s
            d.data += (-alpha) * w
            w.data = pre * d
            wdn = self._product(w, d)
            if self.CheckResidual(sqrt(abs(wdn))):
                return
            beta = wdn / wd
            s *= beta
            s.data += w
