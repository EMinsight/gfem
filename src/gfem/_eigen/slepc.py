"""Shift-invert Krylov-Schur (SLEPc) on the eigenmode seed operator.

SLEPc's EPS runs on a petsc4py Python shell matrix that applies the shared
seed operator (`gfem._eigen.seed`): ``T(sigma)^-1`` through the eigen
operator's own linear solver (any ``linear_solver``), the divergence-free
projection on every output, and the companion form when the seed is damped.
The problem is a standard non-Hermitian one, its largest eigenvalues sought,
and the spectral transform is SLEPc's default ``shift`` at 0, so SLEPc sets
up no linear solver of its own. The start vector is the seed's fixed,
projected vector, so a solve is deterministic; no deflation space is set.
A solve stops after ``seed._SEED_RESTARTS`` restarts, unless the user's
``eps_max_it`` says otherwise (ADR 0122).

Under ranks (ADR 0502) the shell, its vectors and the EPS live on the mesh
communicator, and a vector's rows are the owned free DOFs of the eigen operator
(`EigenOperator.numbering`), ``[x rows; v rows]`` for the companion form: each
apply maps the rows to a state, applies the seed operator unchanged and maps its
state back, so the Krylov process is one process's, permuted. The converged count,
the reason and every eigenvalue are rank 0's, broadcast before gfem reads them
(ADR 0058).

petsc4py and slepc4py come from `gfem._petsc` on the first solve; nothing is
imported at module scope.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Generator, Mapping
from typing import Any, cast

import numpy as np

from gfem import _mpi, _petsc
from gfem._eigen import seed as _seed
from gfem._eigen.operator import EigenOperator
from gfem._eigen.seed import SeedBoundReached, SeedOperator, seed_operator
from gfem.options import PetscOptions, PetscValue, split_petsc_options

__all__ = ["shift_invert"]


class _Shell:
    """The Python context of the shell matrix: ``y = apply(x)``.

    An exception raised in a callback would interrupt SLEPc's solve, and
    SLEPc cannot destroy an EPS interrupted there. So the first exception is
    kept in `error` and the shell answers zero from then on; the stopping
    test (`_stopping_test`) ends the solve at its next check, and the caller
    raises `error` as it was.

    Under ranks (``distributed``) only an agreed error is kept: every rank
    raises it at the same apply, so every rank's stopping test ends the solve
    together. Any other error propagates and ends the job (ADR 0058): kept, it
    would leave the other ranks inside a collective solve.
    """

    def __init__(
        self, apply: Callable[[np.ndarray], np.ndarray], *, distributed: bool = False
    ) -> None:
        self._apply = apply
        self._distributed = distributed
        self.error: BaseException | None = None
        #: How many times the operator was applied.
        self.applies = 0

    def mult(self, mat, x, y) -> None:  # noqa: ARG002 - petsc4py passes the matrix
        if self.error is None:
            self.applies += 1
            try:
                y.array = self._apply(x.array_r.copy())
            except BaseException as err:
                if self._distributed and not _agreed(err):
                    raise
                self.error = err
            else:
                return
        y.array = 0.0


def _agreed(error: BaseException) -> bool:
    """Whether every rank raises ``error`` together (`gfem._mpi.section`'s test)."""
    return isinstance(error, _mpi.AGREED_ERRORS) or getattr(
        error, "_gfem_agreed", False
    )


class _Rows:
    """A seed vector as the shell's rows: all its entries, or owned free rows.

    On one process the rows are the vector. Under ranks (``numbering``, a
    `gfem._linalg.petsc.FreeNumbering`) each block of ``n`` local DOFs (one, or
    two for the companion form) is its owned free rows.
    """

    def __init__(self, numbering, blocks: int, n: int) -> None:
        self._numbering = numbering
        self._blocks = blocks
        self._n = n

    @property
    def distributed(self) -> bool:
        return self._numbering is not None

    def local(self, size: int) -> int:
        """This rank's rows of a vector of ``size`` local entries."""
        if self._numbering is None:
            return size
        return self._blocks * self._numbering.count

    def rows(self, state: np.ndarray) -> np.ndarray:
        """A state's rows."""
        if self._numbering is None:
            return state
        n = self._n
        return np.concatenate(
            [
                self._numbering.state_rows(state[b * n : (b + 1) * n])
                for b in range(self._blocks)
            ]
        )

    def state(self, rows: np.ndarray) -> np.ndarray:
        """The state of rows."""
        if self._numbering is None:
            return rows
        count = self._numbering.count
        return np.concatenate(
            [
                self._numbering.state(rows[b * count : (b + 1) * count])
                for b in range(self._blocks)
            ]
        )


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Where a seed solve runs: its shell, PETSc communicator and shapes.

    Attributes:
        shell: The shell matrix's context.
        comm: The PETSc communicator: ``COMM_SELF``, or the mesh's.
        shape: The shell's PETSc size.
        rows: The map between seed vectors and the shell's rows.
        mpi_comm: The mesh communicator, ``None`` on one process.
        size: A seed vector's local entries.
    """

    shell: _Shell
    comm: Any
    shape: list
    rows: _Rows
    mpi_comm: Any
    size: int

    @classmethod
    def of(cls, op: EigenOperator, seed: SeedOperator, PETSc) -> _Plan:
        """The plan of ``seed`` on ``op``'s layout."""
        rows = _Rows(op.numbering, 2 if seed.damped else 1, seed.n)
        if not rows.distributed:
            shape = [seed.size, seed.size]
            return cls(
                _Shell(seed.apply), PETSc.COMM_SELF, shape, rows, None, seed.size
            )

        def apply(r: np.ndarray) -> np.ndarray:
            return rows.rows(seed.apply(rows.state(r)))

        comm = op.layout.comm
        local = rows.local(seed.size)
        shape = [[local, seed.total], [local, seed.total]]
        shell = _Shell(apply, distributed=True)
        return cls(shell, PETSc.Comm(comm), shape, rows, comm, seed.size)

    def start(self, mat, start: np.ndarray, PETSc):
        """The PETSc start vector of the seed's ``start``."""
        if self.mpi_comm is None:
            return PETSc.Vec().createWithArray(start.copy(), comm=self.comm)
        vector = mat.createVecRight()
        vector.array = self.rows.rows(start)
        return vector

    @contextlib.contextmanager
    def replicated(self) -> Generator[None]:
        """A `ValueError` from the replicated options is raised by every rank."""
        try:
            yield
        except ValueError as err:
            if self.mpi_comm is not None:
                _mpi.agreed(err)
            raise


def _stopping_test(shell: _Shell):
    """SLEPc's basic stopping test, which also ends a solve whose shell failed."""
    reason = _petsc.require_slepc().EPS.ConvergedReason

    def test(eps, its: int, max_it: int, nconv: int, nev: int):  # noqa: ARG001
        if shell.error is not None:
            return reason.CONVERGED_USER
        if nconv >= nev:
            return reason.CONVERGED_TOL
        if its >= max_it:
            return reason.DIVERGED_ITS
        return reason.CONVERGED_ITERATING

    return test


def _ncv(value: PetscValue, nev: int, size: int) -> int:
    """The user's ``eps_ncv``, at least ``nev + 1`` and at most ``size``.

    `gfem.SolverOptions` checked that it is an integer >= 1.
    """
    return min(size, max(cast("int", value), nev + 1))


def _dimensions(
    seed: SeedOperator, nev: int, options: Mapping[str, PetscValue] | PetscOptions
) -> tuple[int, int, tuple]:
    """``(nev, ncv)`` of a seed solve, and the SLEPc options gfem passes on.

    ``eps_ncv`` is gfem's to read (`_ncv`), not SLEPc's.
    """
    nev, ncv = seed.dimensions(nev)
    _, eigen = split_petsc_options(tuple(dict(options).items()))
    pairs = tuple(p for p in eigen if p[0] != "eps_ncv")
    if len(pairs) < len(eigen):
        ncv = _ncv(dict(eigen)["eps_ncv"], nev, seed.total)
    return nev, ncv, pairs


def _check_read(eps, names: list[str]) -> None:
    """Raise if SLEPc did not read some of the user's ``names``.

    Call after the solve, with the options still in the database.
    """
    unread = _petsc.unread(eps.getOptionsPrefix(), names)
    if not unread:
        return
    keys = ", ".join(f"SolverOptions.petsc_options[{n!r}]" for n in unread)
    verb = "was" if len(unread) == 1 else "were"
    raise ValueError(
        f"{keys} {verb} not read by SLEPc's eigen solver (eps_type {eps.getType()}): "
        "check the name, or remove options this configuration does not use."
    )


def _configure(eps, mat, prefix: str, *, nev: int, ncv: int, tol: float) -> None:
    """Set what gfem decides on ``eps``: Krylov-Schur on ``mat``, its prefix.

    ``max_it`` is the seed's restart bound (`gfem._eigen.seed._SEED_RESTARTS`);
    ``eps_max_it`` overrides it, as ``setFromOptions`` runs after.
    """
    SLEPc = _petsc.require_slepc()
    eps.setOptionsPrefix(prefix)
    eps.setOperators(mat)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setDimensions(nev=nev, ncv=ncv)
    eps.setTolerances(tol=tol, max_it=_seed._SEED_RESTARTS)  # noqa: SLF001


def _set_from_options(eps) -> None:
    """``eps.setFromOptions()``; a refused option raises `ValueError`.

    slepc4py reports the refusal as a `SystemError` around PETSc's error.
    """
    PETSc = _petsc.require_petsc()
    try:
        eps.setFromOptions()
    except (PETSc.Error, SystemError) as err:
        raise ValueError(
            "SolverOptions.petsc_options was refused by SLEPc: "
            f"{_petsc.error_text(err)}."
        ) from _petsc.unwrap(err)


def _eigenpairs(eps, vector, nev: int, plan: _Plan) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``nev`` converged ``(theta, vectors)`` of a solved ``eps``.

    A negative converged reason is not an error: the pairs it converged, as
    ARPACK's early stop. Under ranks (``comm``) the count and the eigenvalues
    are rank 0's, and the vectors states.
    """
    comm = plan.mpi_comm
    m = min(_root(comm, eps.getConverged()), nev)
    theta = np.empty(m, complex)
    vectors = np.empty((plan.size, m), complex)
    for i in range(m):
        theta[i] = eps.getEigenpair(i, vector)
        vectors[:, i] = plan.rows.state(vector.array_r.copy())
    if comm is not None:
        comm.Bcast(theta, root=0)
    return theta, vectors


def _root(comm, value: int) -> int:
    """Rank 0's ``value`` on every rank; ``value`` itself on one process."""
    return int(value) if comm is None else int(comm.bcast(int(value), root=0))


def _bound(
    eps, seed: SeedOperator, theta: np.ndarray, vectors: np.ndarray, plan: _Plan
) -> SeedBoundReached | None:
    """What a solved ``eps`` that stopped at its ``max_it`` raises, else ``None``."""
    reason = _petsc.require_slepc().EPS.ConvergedReason
    if _root(plan.mpi_comm, eps.getConvergedReason()) != reason.DIVERGED_ITS:
        return None
    k, x = seed.pairs(theta, vectors)
    applies = plan.shell.applies
    return SeedBoundReached(k, x, restarts=eps.getTolerances()[1], applies=applies)


def shift_invert(
    op: EigenOperator,
    project,
    nev: int,
    *,
    tol: float,
    options: Mapping[str, PetscValue] | PetscOptions = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``nev`` eigenpairs ``(k, x)`` of ``T`` nearest ``op.shift``.

    Args:
        op: The operator with ``T(sigma)`` prepared.
        project: The divergence-free projection, ``x -> x``.
        nev: How many pairs to ask SLEPc for.
        tol: SLEPc's relative tolerance on the shift-inverted eigenvalues.
        options: ``SolverOptions.petsc_options``. Its ``eps_ncv`` sets the
            subspace size (at least ``nev + 1``); its other SLEPc options
            (`gfem.options.split_petsc_options`) reach the EPS, in PETSc's
            options database until the solve ends.

    Returns:
        Wavenumbers ``(m,)`` (per file unit, ``m <= nev``; may include
        pairs with ``Re k <= 0``) and eigenvectors ``(n, m)``.

    Raises:
        SeedBoundReached: SLEPc stopped at its restart bound (``max_it``);
            it carries the pairs it converged.
        gfem.SolverUnavailableError: slepc4py cannot be used.
        ValueError: SLEPc refused the options, or did not read one of them
            (after a solve that succeeded).
        RuntimeError: SLEPc failed (PETSc's reason quoted). An exception
            raised by the shift solve itself is raised as it is.
    """
    SLEPc = _petsc.require_slepc()
    PETSc = _petsc.require_petsc("eigen_solver")
    seed = seed_operator(op, project)
    nev, ncv, pairs = _dimensions(seed, nev, options)

    plan = _Plan.of(op, seed, PETSc)
    shell, comm = plan.shell, plan.comm
    # LOCK is held across the whole solve, not only around the options: the
    # shell's `mult` runs gfem's linear solves (PETSc calls, which take the
    # reentrant LOCK again) on this same thread, and PETSc must never be
    # called from another thread while SLEPc is inside it
    with _petsc.LOCK:
        created: list = []  # destroyed in reverse in the finally
        failed = True
        try:
            mat = PETSc.Mat().createPython(plan.shape, shell, comm=comm)
            created.append(mat)
            start = plan.start(mat, seed.start, PETSc)
            created.append(start)
            vector = mat.createVecRight()
            created.append(vector)
            eps = SLEPc.EPS().create(comm)
            created.append(eps)
            mat.setUp()
            prefix = _petsc.next_prefix()
            _configure(eps, mat, prefix, nev=nev, ncv=ncv, tol=tol)
            eps.setStoppingTest(_stopping_test(shell))
            # SLEPc reads its viewer options (eps_view, eps_error_*, ...) at
            # the end of the solve: the options stay in the database across it
            with _petsc.scoped_options(prefix, pairs):
                with plan.replicated():
                    _set_from_options(eps)
                eps.setInitialSpace([start])
                eps.solve()
                if shell.error is not None:
                    raise shell.error
                with plan.replicated():
                    _check_read(eps, [name for name, _ in pairs])
            theta, vectors = _eigenpairs(eps, vector, nev, plan)
            bound = _bound(eps, seed, theta, vectors, plan)
            failed = False
        except (PETSc.Error, SystemError) as err:
            if shell.error is not None:
                raise shell.error from None
            raise RuntimeError(
                "SLEPc's eigen solve failed (SolverOptions.petsc_options = "
                f"{dict(options)}): {_petsc.error_text(err)}"
            ) from _petsc.unwrap(err)
        finally:
            for obj in reversed(created):
                try:
                    obj.destroy()
                except (PETSc.Error, SystemError):
                    # a destroy that fails never hides the solve's own error
                    if not failed:
                        raise
    if bound is not None:
        raise bound
    return seed.pairs(theta, vectors)
