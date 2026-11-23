"""The ``bddc_gmres`` backend: NGSolve BDDC preconditioner and a Krylov solver.

BDDC splits each element's DOFs into a wirebasket (vertices and low-order
edges) and the rest, and solves a global coarse problem on the wirebasket.
That coarse problem is factorized by an explicitly named inverse, never
NGSolve's default: gfem's flagged MKL PARDISO with the ``mkl`` extra (see
`gfem._linalg.pardiso`), ``sparsecholesky`` without it, or when a setup is
expected to serve enough right-hand-side columns for its threads: 16 or more
on 16 threads or more, 6 or more on 56 threads or more (`_coarse_inverse`).
A general operator (a Bloch phase, A^T != A) is never given sparse Cholesky:
its coarse problem is factored by PARDISO LU, or UMFPACK without MKL.
The Krylov method follows the operator: conjugate gradients for a Hermitian
(real symmetric positive definite) operator, NGSolve's on one process, otherwise gfem's
GMRES (`gfem._linalg.gmres`), which orthogonalizes its Krylov basis by CGS2
on gfem's threads.

Across a sweep the handle keeps its preconditioner, and sets it up again by
the iteration-count rules of `gfem._linalg.krylov_handle`, which also says
what `BddcHandle.last_action` and `last_reason` report.

Under ranks (an operator with a layout) BDDC's coarse problem is factored by MUMPS
across ranks when the MPI PETSc stack is installed (`gfem._linalg.coarse`), otherwise
on rank 0 by the same rule; GMRES runs on owned entries (`gfem._linalg.gmres`), CG
is gfem's on agreed products (`gfem._linalg.cg`), and every reuse decision
reads an agreed value: the iterations, the residuals and the norms of
`gfem._linalg.layout`. Each rank's BLAS runs at its threads in a setup and at
one thread in a solve, whose coarse apply gains nothing from them.
"""

from __future__ import annotations

import ngsolve as ngs

from gfem import _mpi, _petsc, _timing
from gfem._linalg import Operator, Symmetry, coarse
from gfem._linalg.base import (
    equals_its_transpose,
    host_bytes,
    local_matrix,
    same_free_dofs,
)
from gfem._linalg.direct import GENERAL_INVERSE, INVERSE
from gfem._linalg.krylov_handle import (
    KrylovHandle,
    _slack,  # noqa: F401  (the per-column slack of BDDC's reuse, read here by name)
)
from gfem._linalg.pardiso import pardiso_inverse

#: Threads from which a setup serving many columns factors its coarse problem
#: with sparse Cholesky: at 16 threads PARDISO's one-column solve broke even
#: with sparse Cholesky's after 154-365 applies of one factorization on the
#: Modal CPUs, 306-429 on MareNostrum 5's Xeon 8480+; up to 8 threads
#: PARDISO's solve was no slower, and 9-15 threads are unmeasured.
SPARSE_COARSE_THREADS = 16
#: Right-hand-side columns a setup is expected to serve (each about 60 applies
#: cold, 33 warm) from which it factors its coarse problem with sparse
#: Cholesky: 16 columns are about 555 applies, 1.3x the largest break-even
#: at 16 threads (MareNostrum 5); every measured single solve (1-4 ports)
#: stays below it.
SPARSE_COARSE_COLUMNS = 16
#: Threads from which a setup factors its coarse problem with sparse Cholesky
#: at fewer columns (`SPARSE_COARSE_WIDE_COLUMNS`): on MareNostrum 5 (Xeon
#: 8480+) the break-even fell to 95-133 applies at 56 and 112 threads, against
#: 156-216 at 32 and 306-429 at 16. 33-55 threads are unmeasured.
SPARSE_COARSE_WIDE_THREADS = 56
#: Columns from which a setup on `SPARSE_COARSE_WIDE_THREADS` threads or more
#: factors with sparse Cholesky: 6 columns are at least about 225 applies,
#: 1.7x the largest break-even there. A 2-port single solve (2 columns, about
#: 128 applies, at the break-even) stays on PARDISO.
SPARSE_COARSE_WIDE_COLUMNS = 6


def _coarse_inverse(
    symmetry: Symmetry, *, is_complex: bool, threads: int, columns: int
) -> str:
    """The inverse that factors BDDC's coarse problem.

    ``sparsecholesky`` when the setup runs on `SPARSE_COARSE_THREADS` threads
    or more and is expected to serve `SPARSE_COARSE_COLUMNS` right-hand-side
    columns or more, or on `SPARSE_COARSE_WIDE_THREADS` or more and
    `SPARSE_COARSE_WIDE_COLUMNS` or more: there its many applies outweigh
    PARDISO's faster factorization, and the solve with a sparse Cholesky
    coarse factorization scales with the threads where PARDISO's one-column
    solve does not.
    Otherwise gfem's PARDISO (`pardiso_inverse`) when MKL is installed,
    symmetric when the operator is ``A^T = A`` (complex symmetric, or
    ``"hermitian"`` on a real space; a complex Hermitian one is not), sparse
    Cholesky without it. The rule counts columns, never times, so a run's
    choice repeats exactly.

    An operator without ``A^T = A`` (a Bloch phase) skips the rule: sparse
    Cholesky would factor it as ``L D L^T``, silently wrong, so its coarse
    problem is PARDISO's LU, or UMFPACK (`GENERAL_INVERSE`) without MKL.
    """
    if not equals_its_transpose(symmetry, is_complex=is_complex):
        return pardiso_inverse(symmetry, is_complex=is_complex) or GENERAL_INVERSE
    if (threads >= SPARSE_COARSE_THREADS and columns >= SPARSE_COARSE_COLUMNS) or (
        threads >= SPARSE_COARSE_WIDE_THREADS and columns >= SPARSE_COARSE_WIDE_COLUMNS
    ):
        return INVERSE
    return pardiso_inverse(symmetry, is_complex=is_complex) or INVERSE


def _set_up(
    form: ngs.BilinearForm, symmetry: Symmetry, *, columns: int, comm=None
) -> ngs.Preconditioner:
    """Register BDDC on ``form`` and assemble it, which sets BDDC up.

    The coarse (wirebasket) problem's solver is an explicitly named direct
    factorization (`_coarse_inverse`), chosen from NGSolve's threads and the
    ``columns`` the setup is expected to serve. (``coarsetype="direct"`` is
    not set: on the driven H(curl) reference it made the setup 5x slower and
    GMRES take 68 iterations, not 49.) Across ranks, all name one inverse, or on the
    MPI PETSc stack gfem's own ``coarsetype`` (`gfem._linalg.coarse`).
    """
    is_complex = bool(form.space.is_complex)
    if _coarse_on_mumps(comm, is_complex=is_complex):
        preconditioner = ngs.Preconditioner(
            form,
            "bddc",
            coarsetype=coarse.coarse_type(
                comm, equals_its_transpose(symmetry, is_complex=is_complex)
            ),
        )
        with _timing.phase("assemble"):
            form.Assemble()
        return preconditioner
    inverse = _coarse_inverse(
        symmetry,
        is_complex=is_complex,
        threads=_mpi.agree_min(comm, ngs.GetNumThreads()),
        columns=columns,
    )
    preconditioner = ngs.Preconditioner(form, "bddc", inverse=inverse)
    with _timing.phase("assemble"):
        form.Assemble()
    return preconditioner


def _coarse_on_mumps(comm, *, is_complex: bool) -> bool:
    """Whether BDDC's coarse problem leaves rank 0: on ranks, with the MPI stack.

    A wheel-only install keeps rank 0, and so does a stack gfem cannot use. The route
    never changes iterations or S, so no error is worth a solve that can still run.
    """
    if comm is None or not is_complex:
        return False
    return _mpi.agree_all(comm, flag=_petsc.mpi_stack_usable())


class BddcHandle(KrylovHandle):
    """One BDDC preconditioner, reused by a Krylov solve per right-hand side.

    BDDC reads the fixed DOFs from the space: the operator's free DOFs must
    be the space's ``FreeDofs()`` (build the space with ``dirichlet=``).

    The Krylov solve, its warm start and the preconditioner reuse across a
    sweep are `gfem._linalg.krylov_handle.KrylovHandle`'s; this handle
    registers BDDC on the form before assembly, which sets it up. Every setup
    chooses its coarse factorization (`_coarse_inverse`) from the columns one
    setup is expected to serve.
    """

    kind = "bddc_gmres"
    preconditioner_name = "BDDC"

    def __init__(
        self,
        operator: Operator,
        symmetry: Symmetry,
        *,
        tolerance: float,
        maxiter: int,
        reuse: bool = True,
        width: int = 1,
        solves: int = 1,
    ) -> None:
        """Register BDDC on the form, assemble, and set up the Krylov solver.

        ``reuse=False`` sets BDDC up again at every `refactorize`. ``width``
        right-hand-side columns per solve and ``solves`` solve calls are
        expected; they choose the coarse factorization only.
        """
        self._coarse = None
        super().__init__(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            reuse=reuse,
            width=width,
            solves=solves,
        )

    def _check_operator(self, operator: Operator) -> None:
        """Refuse free DOFs other than the space's: BDDC reads the space's."""
        same = same_free_dofs(operator.freedofs, operator.form.space.FreeDofs())
        if not _mpi.agree_all(self.layout.comm, flag=same):
            raise ValueError(
                "BDDC fixes the Dirichlet DOFs of the space, so the operator's free "
                "DOFs must be the space's FreeDofs(); build the space with "
                "dirichlet= on the fixed boundaries."
            )

    def _set_up(self, form: ngs.BilinearForm) -> ngs.Preconditioner:
        """`_set_up` on ``form``, every BLAS at this rank's threads under ranks."""
        comm = self.layout.comm
        with self.layout.blas_threads(ngs.GetNumThreads()):
            preconditioner = _set_up(
                form, self.symmetry, columns=self._expected_columns, comm=comm
            )
            self._coarse = coarse.finish(comm, columns=self._expected_columns)
        return preconditioner

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix, the preconditioner and the coarse factors."""
        # read once, without the lock: a concurrent free gives 0 or the old count
        held, pre, coarse = self._matrix, self._preconditioner, self._coarse
        if self._solver is None or held is None or pre is None:
            return 0
        matrix = local_matrix(held, self.layout.comm)
        factors = 0 if coarse is None else coarse.nbytes
        return host_bytes(matrix) + host_bytes(pre.mat) + factors

    def _drop_setup(self) -> None:
        """Give a coarse factorization back to its pool, every rank together."""
        if self._coarse is not None:
            self._coarse.free()
            self._coarse = None
