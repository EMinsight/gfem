"""BDDC's coarse problem across ranks on PETSc MUMPS.

On rank 0 the coarse factorization holds about a third of a mesh's DOFs in one
process and every rank waits for its apply. NGSolve lets a registered ``coarsetype``
take its place, so with the MPI PETSc stack `CoarseMumps` factors the wirebasket
across ranks, with no change to BDDC: iterations and S stay those of the rank 0 route.

PT-Scotch's fill grows with the ranks it orders on (r10M: 556 M entries at 104 k
wirebasket DOF per rank, 4520 M at 52 k; r10M_void_split, 4 threads per rank: 7.3 G on
64 ranks, 85 G on 180), so ``PCTELESCOPE`` in front of MUMPS runs it on at most one
rank per `DOFS_PER_RANK` and `MAX_RANKS` ranks, without building a communicator by hand.

MUMPS factors it by :math:`L D L^T` where the operator equals its transpose and by LU
otherwise (a Bloch phase). NGSolve hands the creator only the matrix and the free
DOFs, so each symmetry has its own registered ``coarsetype`` (`coarse_type`).
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, Final, Literal

import ngsolve as ngs
import numpy as np

from gfem import _mpi, _petsc
from gfem._linalg.petsc import (
    MUMPS_RETRIES,
    MUMPS_WORKSPACE_ERROR,
    RANK_MUMPS_OPTIONS,
    FreeNumbering,
    distributed_aij,
)
from gfem._linalg.petsc import defaults as factor_defaults

#: Wirebasket DOFs per coarse rank: PT-Scotch's fill is fine at 66 k and 104 k and
#: jumps at 52 k and 33 k on r0.5 and r10M; "about 100 k" errs on the fine side.
DOFS_PER_RANK = 100_000
#: Coarse ranks at most under PT-Scotch: with an even count per node it orders on the
#: largest power of two of them, so 56 orders on 32 and 64 on 64, where the fill jumps
#: (B2_void_split: 8.87 G entries on 56 ranks, 10.8-11.3 G on 64).
MAX_RANKS = 56

#: The two coarse orderings `rule` picks from.
PT_SCOTCH: Final = "pt-scotch"
METIS: Final = "metis"
type Ordering = Literal["pt-scotch", "metis"]
#: METIS's coarse ranks are every this-many-th rank: 7 per node at 28 ranks per node.
METIS_EVERY = 4
#: Wirebasket DOFs per METIS coarse rank at least: r0.5 on 224 ranks, 31.5 k each,
#: still applied faster than on 56.
METIS_DOFS_PER_RANK = 30_000
#: Columns from which one METIS set-up repays its serial analysis: about 300-370
#: applies at B2's 64 iterations per column.
METIS_COLUMNS = 6
#: MUMPS reads it through ``MUMPS_CONFIG_FILE``: METIS's ``NITER=1`` takes about
#: 15 % off the serial analysis for 2 % more fill, and PETSc has no option for it.
MUMPS_CONFIG = Path(__file__).with_name("mumps_metis.cfg")
#: The variable MUMPS reads its configuration file from.
CONFIG_VARIABLE = "MUMPS_CONFIG_FILE"
#: MUMPS's ordering options under METIS: a sequential analysis (ICNTL(28) 1) with
#: METIS (ICNTL(7) 5), so ICNTL(29), PT-Scotch's choice, is left out.
METIS_MUMPS_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("mat_mumps_icntl_28", 1),
    ("mat_mumps_icntl_7", 5),
)

#: Failures a larger workspace may cure: behind telescoping every failure reads 0.
RETRIED = (MUMPS_WORKSPACE_ERROR, 0)

#: Per communicator, the factorizations NGSolve asked for since the last `finish`,
#: of either symmetry.
_made: dict[int | None, list[CoarseMumps]] = {}
#: The ``coarsetype`` names registered with NGSolve in this process.
_registered: set[str] = set()


def reduction(dofs: int, size: int) -> int:
    """``PCTELESCOPE``'s factor, under `DOFS_PER_RANK` and `MAX_RANKS`.

    Args:
        dofs: The coarse problem's free DOFs.
        size: The ranks of the communicator.

    Returns:
        1 for no reduction. PETSc keeps every factor-th rank, ``ceil(size / factor)``
        of them.
    """
    return max(1, math.ceil(size / max(1, min(dofs // DOFS_PER_RANK, MAX_RANKS))))


def rule(dofs: int, size: int, columns: int, *, seen: bool) -> tuple[Ordering, int]:
    r"""The coarse ordering and ``PCTELESCOPE``'s factor of one set-up.

    With :math:`n` the coarse DOFs and :math:`P` the ranks, PT-Scotch takes
    :math:`c_P = \max(1, \min(P, \lfloor n / 100\,000 \rfloor, 56))` ranks and
    sequential METIS :math:`c_M = \max(1, \min(\lceil P / 4 \rceil,
    \lfloor n / 30\,000 \rfloor))`. METIS orders when :math:`c_M > c_P` and the
    set-up serves at least `METIS_COLUMNS` columns or repeats a pattern: its serial
    analysis costs several PT-Scotch set-ups, which only a faster apply over many
    columns, or an analysis kept across set-ups, pays back. The rule reads counts
    only, so every rank and every rerun picks the same.

    Args:
        dofs: The coarse problem's free DOFs.
        size: The ranks of the communicator.
        columns: The columns the set-up is expected to serve.
        seen: This process has set up the same pattern before.

    Returns:
        The ordering and the factor: PETSc keeps every factor-th rank.
    """
    scotch = max(1, min(size, dofs // DOFS_PER_RANK, MAX_RANKS))
    metis = max(1, min(math.ceil(size / METIS_EVERY), dofs // METIS_DOFS_PER_RANK))
    if metis > scotch and (columns >= METIS_COLUMNS or seen):
        return METIS, math.ceil(size / metis)
    return PT_SCOTCH, reduction(dofs, size)


@contextlib.contextmanager
def mumps_config(ordering: Ordering) -> Generator[None]:
    """``MUMPS_CONFIG_FILE`` on `MUMPS_CONFIG` for a METIS set-up, under `_petsc.LOCK`.

    MUMPS reads the variable at each job, so rank-local factors set up outside the
    scope never see it, while the coarse factor keeps ``NITER=1`` for its later
    jobs. A value the user set is left as it is: it applies to every MUMPS in the
    process.
    """
    with _petsc.LOCK:
        if ordering != METIS or CONFIG_VARIABLE in os.environ:
            yield
            return
        os.environ[CONFIG_VARIABLE] = str(MUMPS_CONFIG)
        try:
            yield
        finally:
            os.environ.pop(CONFIG_VARIABLE, None)


def options(
    symmetric: bool,
    *,
    workspace: int,
    factor: int,
    ordering: Ordering = PT_SCOTCH,
) -> tuple[tuple[str, Any], ...]:
    """PETSc's options: an exact MUMPS factorization, behind telescoping if it reduces.

    Args:
        symmetric: The operator has :math:`A^T = A`: :math:`L D L^T`, else LU.
        workspace: MUMPS's ICNTL(14).
        factor: ``PCTELESCOPE``'s reduction (`rule`); 1 for none.
        ordering: `PT_SCOTCH`, gfem's options under ranks, or `METIS`
            (`METIS_MUMPS_OPTIONS` in their place).
    """
    mumps = dict(RANK_MUMPS_OPTIONS)
    if ordering == METIS:
        del mumps["mat_mumps_icntl_29"]
        mumps.update(METIS_MUMPS_OPTIONS)
    mumps["mat_mumps_icntl_14"] = workspace
    factorization = (*factor_defaults(symmetric)[1:], *mumps.items())
    if factor == 1:
        return (("ksp_type", "preonly"), *factorization)
    return (
        ("ksp_type", "preonly"),
        ("pc_type", "telescope"),
        ("pc_telescope_reduction_factor", factor),
        ("telescope_ksp_type", "preonly"),
        *((f"telescope_{name}", value) for name, value in factorization),
    )


class CoarseMumps(ngs.BaseMatrix):
    """The coarse factorization: a ``ParallelMatrix`` factored by MUMPS across ranks.

    NGSolve calls the creator twice per setup, first with the form's whole matrix,
    which is not assembled yet, and last with the wirebasket, so nothing is factored
    until `set_up`. The PETSc objects are destroyed by `free`, which every rank calls
    together, never by a collector that runs at another time on each rank.
    """

    def __init__(
        self,
        matrix: Any,
        freedofs: Any,
        comm: Any,
        *,
        symmetric: bool,
    ) -> None:
        """Hold what NGSolve gives; nothing is read from it yet.

        ``symmetric``: the operator has :math:`A^T = A`, so MUMPS factors it by
        :math:`L D L^T` on a matrix flagged symmetric; LU otherwise.
        """
        super().__init__()
        self.symmetric = symmetric
        self._source = matrix
        self._freedofs = freedofs
        self._comm = comm
        self._ksp: Any = None
        self._aij: Any = None
        self._x: Any = None
        self._b: Any = None
        self._freed = False
        #: The columns the set-up is expected to serve, for `rule`.
        self.columns = 1

    def set_up(self) -> None:
        """Factor the wirebasket, on a new KSP while MUMPS's workspace is too small.

        Raises:
            gfem._mpi.AgreedError: The factorization failed, on every rank.
        """
        PETSc = _petsc.require_petsc()
        comm = self._comm
        with _petsc.LOCK:
            self._work = self._source.CreateColVector()
            self._numbering = FreeNumbering(
                self._source.row_pardofs, self._freedofs, self._work, comm
            )
            self._aij, _ = distributed_aij(
                self._source.local_mat, self._numbering, comm, symmetric=self.symmetric
            )
            self._x, self._b = self._aij.createVecs()
            self._ordering, self._factor = rule(
                self._numbering.total, comm.size, self.columns, seen=False
            )
            workspace = dict(RANK_MUMPS_OPTIONS)["mat_mumps_icntl_14"]
            for attempt in range(MUMPS_RETRIES + 1):
                self._ksp = PETSc.KSP().create(comm)
                prefix = _petsc.next_prefix()
                self._ksp.setOptionsPrefix(prefix)
                self._ksp.setOperators(self._aij)
                pairs = options(
                    self.symmetric,
                    workspace=workspace,
                    factor=self._factor,
                    ordering=self._ordering,
                )
                with (
                    _petsc.scoped_options(prefix, pairs),
                    mumps_config(self._ordering),
                ):
                    self._ksp.setFromOptions()
                    self._ksp.setUp()
                    if self._factor > 1:
                        # the inner factorization happens at the first apply
                        self._b.set(1.0)
                        self._ksp.solve(self._b, self._x)
                failure = self._agreed_failure()
                if failure is None:
                    return
                self._ksp.destroy()
                self._ksp = None
                if failure not in RETRIED or attempt == MUMPS_RETRIES:
                    self.free()
                    raise _mpi.AgreedError(
                        "The factorization of BDDC's coarse problem failed (MUMPS "
                        f"INFOG(1)={failure or 'unread behind PCTELESCOPE'}): the "
                        "operator is singular at this frequency, or MUMPS needs more "
                        f"workspace than ICNTL(14) {workspace} gives."
                    )
                workspace *= 2

    def _agreed_failure(self) -> int | None:
        """``None`` if the set-up succeeded, else INFOG(1) on every rank.

        Behind ``PCTELESCOPE`` it is 0: petsc4py 3.26 has no way to the inner factor,
        so what shows there is the first apply's failed preconditioner and a solution
        that is not finite.
        """
        PETSc = _petsc.require_petsc()
        pc = self._ksp.getPC()
        failed = pc.getFailedReason() != PETSc.PC.FailedReason.NOERROR
        if self._factor > 1:
            failed = failed or not np.isfinite(self._x.array).all()
        if not _mpi.agree_any(self._comm, flag=failed):
            return None
        if self._factor > 1:
            return 0
        return _mpi.agree_min(self._comm, int(pc.getFactorMatrix().getMumpsInfog(1)))

    @property
    def nbytes(self) -> int:
        """This rank's factors, by MUMPS's own count (INFO(22)).

        0 behind ``PCTELESCOPE``, whose inner factor cannot be read
        (`_agreed_failure`): the cache's count then leaves these factors out.
        """
        if self._ksp is None or self._factor > 1:
            return 0
        return 10**6 * int(self._ksp.getPC().getFactorMatrix().getMumpsInfo(22))

    def IsComplex(self) -> bool:  # noqa: N802 - NGSolve's name
        """Only complex operators take this route (`gfem._linalg.bddc`)."""
        return True

    def Height(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.height

    def Width(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.width

    def CreateRowVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateRowVector()

    def CreateColVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateColVector()

    def Mult(self, x: Any, y: Any) -> None:  # noqa: N802 - NGSolve's name
        """``y`` = the inverse on ``x``: exchange, owned, MUMPS, owned, exchange."""
        if self._freed:
            raise RuntimeError("This coarse factorization was freed.")
        with _petsc.LOCK:
            if self._ksp is None:
                self.set_up()
            work = self._work
            work.data = x
            work.Cumulate()
            self._b.array[:] = self._numbering.state_rows(work.FV().NumPy())
            self._ksp.solve(self._b, self._x)
            y.FV().NumPy()[:] = self._numbering.state(self._x.array)
            y.SetParallelStatus(ngs.la.PARALLEL_STATUS.CUMULATED)

    def free(self) -> None:
        """Destroy the PETSc objects; every rank calls this together."""
        with _petsc.LOCK:
            self._freed = True
            for obj in (self._ksp, self._aij, self._x, self._b):
                if obj is not None:
                    obj.destroy()
            self._ksp = self._aij = self._x = self._b = None


def coarse_type(comm: Any, symmetric: bool) -> str:
    """The ``coarsetype`` flag of BDDC on ``comm``, registered on first use.

    One per communicator and symmetry, ``gfem_mumps_{key}_{lu|cholesky}``: the
    creator closes over its factorization, since NGSolve gives it no symmetry. Both
    keep their factorizations in one list, for `finish`.
    """
    key = _mpi.comm_key(comm)
    name = f"gfem_mumps_{key}_{'cholesky' if symmetric else 'lu'}"
    _made.setdefault(key, [])
    if name not in _registered:

        def create(matrix: Any, freedofs: Any, *_: Any) -> CoarseMumps:
            made = CoarseMumps(matrix, freedofs, comm, symmetric=symmetric)
            _made.setdefault(key, []).append(made)
            return made

        ngs.comp.RegisterPreconditioner(name, create)
        _registered.add(name)
    return name


def finish(comm: Any) -> CoarseMumps | None:
    """After a setup: the factorization it uses, factored; ``None`` on the rank 0 route.

    BDDC keeps the last one NGSolve asked for, and an earlier one was never factored.
    """
    made = _made.get(_mpi.comm_key(comm)) if comm is not None else None
    if not made:
        return None
    last = made[-1]
    made.clear()
    last.set_up()
    return last
