"""How a space's DOFs lie across ranks: the products of load and state columns.

A column is a NumPy array over a rank's DOFs, shared ones included, and its role
names its status. A **load column** (``solve``'s ``B``, ``apply``'s
result, a port's excitation or functional) is ``DISTRIBUTED``: a shared DOF's value
is the sum over the ranks that hold it. A **state column** (``solve``'s result,
``x0``, ``apply``'s ``X``) is ``CUMULATED``: each of those ranks holds the value.

A load paired with a state is a rank sum of local products. Two columns of one role
need an exchange and a sum over owned DOFs, since a plain rank sum depends on how a
load is split. NGSolve's vectors carry the status, and `mark_load` and `mark_state`
tell them what a NumPy write just put there: a state in a column vector is wrong by
far more than its size, with no error.

Two states multiply over owned DOFs only (`DofLayout.state_inner`,
`DofLayout.state_gram`, `DofLayout.state_norm`), and a few state columns are made
orthonormal by Cholesky QR twice on their Gram matrix (`DofLayout.state_qr`). A DOF
mask set by any of its holders becomes one every holder agrees on through one
exchange (`DofLayout.held_by_any`).

With one process (`ONE_PROCESS`) the roles collapse: `pair` is literally
``load.T @ state``, each state product is today's NumPy expression, bit for bit,
and nothing communicates. Each product returns an agreed value, reduced to rank 0
and broadcast.
"""

from __future__ import annotations

import contextlib
from math import sqrt
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.sparse

from gfem import _mpi
from gfem._linalg import blas

__all__ = ["ONE_PROCESS", "DofLayout", "RankLayout"]


class DofLayout:
    """One process's layout, where every product is the plain one.

    Attributes:
        comm: The communicator; ``None`` on one process.
        owners: Per local DOF, whether this rank owns it; ``None`` on one process,
            where it owns them all.
    """

    comm: Any = None
    owners: np.ndarray | None = None

    def pair(
        self, load: np.ndarray | scipy.sparse.csc_matrix, state: np.ndarray
    ) -> np.ndarray:
        """``load.T @ state``: over all ranks, the same for every split of ``load``."""
        return load.T @ state

    def inner(self, a: np.ndarray, b: np.ndarray) -> complex:
        """``vdot(a, b)`` of two load columns."""
        return np.vdot(a, b)

    def norm(self, load: np.ndarray) -> float:
        """The 2-norm of a load column."""
        return float(np.linalg.norm(load))

    def exchange(self, load: np.ndarray) -> np.ndarray:
        """The state column of a load's summed values."""
        return load

    def held_by_any(self, mask: np.ndarray) -> np.ndarray:
        """Per local DOF, whether some rank holding it set ``mask`` there."""
        return np.asarray(mask, dtype=bool)

    def state_inner(self, a: np.ndarray, b: np.ndarray) -> Any:
        """The owned-entry product ``vdot(a, b)`` of two state columns."""
        return np.vdot(a, b)

    def state_gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""Owned-entry products :math:`A^\mathsf{H} B` of two sets of state columns."""
        return A.conj().T @ B

    def state_norm(self, state: np.ndarray | ngs.BaseVector) -> float:
        """The 2-norm of a state column, or of an NGSolve vector holding one."""
        if isinstance(state, ngs.BaseVector):
            return ngs.Norm(state)
        return float(np.linalg.norm(state))

    def state_qr(self, A: np.ndarray) -> np.ndarray:
        """Orthonormal state columns spanning those of ``A``: ``np.linalg.qr``'s Q."""
        return np.linalg.qr(A)[0]

    def mark_load(self, vector: ngs.BaseVector) -> None:
        """Tell ``vector`` that a load was written into its NumPy view."""

    def mark_state(self, vector: ngs.BaseVector) -> None:
        """Tell ``vector`` that a state was written into its NumPy view."""

    def blas_threads(self, threads: int) -> contextlib.AbstractContextManager[None]:  # noqa: ARG002
        """A phase's BLAS threads; one process leaves the libraries as they are."""
        return contextlib.nullcontext()


#: The layout of every serial handle.
ONE_PROCESS = DofLayout()


class RankLayout(DofLayout):
    """A space's DOFs across the ranks of a communicator.

    A rank that holds no DOFs takes part in every product with empty columns.
    """

    def __init__(self, space: ngs.FESpace, comm: Any) -> None:
        """The layout of ``space`` on the resolved ``comm``."""
        self.comm = comm
        from gfem._linalg.base import free_mask  # base imports this module

        self.pardofs = space.ParallelDofs()
        self.owners = free_mask(self.pardofs.MasterDofs(), space.ndof)
        self._dtype = np.complex128 if space.is_complex else np.float64
        self._work = ngs.GridFunction(space).vec.CreateVector()

    def _sum(self, local: np.ndarray) -> np.ndarray:
        """The rank sum of ``local``, in this space's scalar type and its shape.

        A product of two columns is 0-d, which the reduction's buffer is not.
        """
        local = np.asarray(local, self._dtype)
        return _mpi.reduce_bcast(self.comm, local).reshape(local.shape)

    def pair(
        self, load: np.ndarray | scipy.sparse.csc_matrix, state: np.ndarray
    ) -> np.ndarray:
        """The rank sum of the local products ``load.T @ state``."""
        return self._sum(load.T @ state)

    def inner(self, a: np.ndarray, b: np.ndarray) -> complex:
        """``vdot`` over owned DOFs of both loads, exchanged."""
        owned = self.owners
        local = np.vdot(self.exchange(a)[owned], self.exchange(b)[owned])
        return self._sum(np.array([local]))[0].item()

    def norm(self, load: np.ndarray) -> float:
        """The 2-norm over owned DOFs of the exchanged load."""
        return _owned_norm(self.comm, self.exchange(load)[self.owners])

    def exchange(self, load: np.ndarray) -> np.ndarray:
        """The sum over a shared DOF's holders, on each of them: one neighbour exchange.

        Args:
            load: ``(ndof,)`` or ``(ndof, k)``.
        """
        load = np.asarray(load)
        if load.ndim == 1:
            return self._exchange_column(load)
        out = np.zeros(load.shape, self._dtype)
        for j in range(load.shape[1]):
            out[:, j] = self._exchange_column(load[:, j])
        return out

    def _exchange_column(self, column: np.ndarray) -> np.ndarray:
        work = self._work
        work.FV().NumPy()[:] = column
        self.mark_load(work)
        work.Cumulate()
        # a copy: the next exchange writes here
        return np.array(work.FV().NumPy())

    def held_by_any(self, mask: np.ndarray) -> np.ndarray:
        """The OR of ``mask`` over each DOF's holders: one neighbour exchange."""
        return np.real(self.exchange(np.asarray(mask, dtype=self._dtype))) > 0

    def state_inner(self, a: np.ndarray, b: np.ndarray) -> Any:
        """The owned-entry product of two states: owned DOFs' ``vdot``, rank-summed."""
        owned = self.owners
        local = np.atleast_1d(np.vdot(a[owned], b[owned]))
        return _mpi.reduce_bcast(self.comm, local)[0]

    def state_gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""Owned-entry products :math:`A^\mathsf{H} B` of state columns, summed."""
        owned = self.owners
        return _mpi.reduce_bcast(self.comm, A[owned].conj().T @ B[owned])

    def state_norm(self, state: np.ndarray | ngs.BaseVector) -> float:
        """The 2-norm over owned DOFs: agreed, unlike ``ngs.Norm``, an ``Allreduce``."""
        if isinstance(state, ngs.BaseVector):
            state = state.FV().NumPy()
        return _owned_norm(self.comm, np.asarray(state)[self.owners])

    def state_qr(self, A: np.ndarray) -> np.ndarray:
        r"""Orthonormal state columns spanning ``A``'s: Cholesky QR twice.

        With :math:`G = A^\mathsf{H} A = R^\mathsf{H} R`, :math:`Q = A R^{-1}`, done
        twice, each Gram matrix agreed. Its columns may differ from one process's
        Householder Q by a unit factor each; the span is the same.

        Raises:
            numpy.linalg.LinAlgError: ``A``'s columns are numerically dependent, on
                every rank alike (the Gram matrix is agreed).
        """
        Q = np.asarray(A)
        for _ in range(2):
            upper = np.linalg.cholesky(self.state_gram(Q, Q)).conj().T
            Q = np.linalg.solve(upper.T, Q.T).T
        return Q

    def mark_load(self, vector: ngs.BaseVector) -> None:
        """``vector``'s status becomes ``DISTRIBUTED``."""
        vector.SetParallelStatus(ngs.la.PARALLEL_STATUS.DISTRIBUTED)

    def mark_state(self, vector: ngs.BaseVector) -> None:
        """``vector``'s status becomes ``CUMULATED``."""
        vector.SetParallelStatus(ngs.la.PARALLEL_STATUS.CUMULATED)

    def blas_threads(self, threads: int) -> contextlib.AbstractContextManager[None]:
        """Hold every loaded BLAS at ``threads`` for a phase."""
        return blas.held(threads)


def _owned_norm(comm: Any, owned: np.ndarray) -> float:
    """The 2-norm of a column cut to each rank's owned entries."""
    squares = np.array([float(np.sum(owned.real**2 + owned.imag**2))])
    return sqrt(_mpi.reduce_bcast(comm, squares)[0])
