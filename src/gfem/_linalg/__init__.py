"""Every linear solve goes through one handle seam: prepare / solve / free.

``prepare(operator, symmetry=..., kind=...)`` assembles the operator's form
and sets up a backend (a factorization, a preconditioner) into a `Handle`.
``Handle.solve(B)`` solves for one or many right-hand sides, as often as
needed: terminal solves, port solves and adjoint solves reuse the same
factors. ``Handle.solve_counted(B)`` also returns the Krylov iterations of
each column of ``B`` in this call. ``Handle.free()`` releases the backend's
memory.

A handle may also refactorize (`Refactorable`): ``handle.refactorize(operator)``
assembles another operator with the same sparsity pattern (the next
frequency of a sweep) and factorizes its values on the existing solver
plan, so a sweep plans once. ``"direct"``, ``"cudss"`` and ``"petsc"``
handles can;
``"bddc_gmres"`` keeps its preconditioner for the new operator instead, caps
each warm solve on it, and sets it up again when reusing it costs more than a
setup (preconditioner reuse; ``prepare(..., reuse_preconditioner=False)`` sets
it up at every refactorization). `refactorize_or_prepare` refactorizes a
handle that can, and prepares a new one in place of any other.

An operator may also carry its piece sum (``Operator.piece_sum``, see
`gfem._linalg.pieces`): pieces assembled once and their weights, which the
``direct`` and ``petsc`` linear solvers sum instead of assembling the form.

Vectors are NumPy arrays over all DOFs of the operator's space. ``solve``
reads only the free rows of ``B`` and returns zeros in the fixed (Dirichlet)
rows, so a Dirichlet lifting ``u_D`` is completed by
``u = u_D + handle.solve(-handle.apply(u_D))``. Under ranks they are arrays over
a rank's DOFs, and each one's role names its status: ``solve`` maps a load to a
state, ``apply`` a state to a load (`gfem._linalg.layout`).

Backends:

- ``"direct"``: an LDL^T factorization, always named explicitly, never
  NGSolve's default inverse: MKL PARDISO with METIS ordering when the ``mkl``
  extra is installed, NGSolve sparse Cholesky otherwise. One factorization
  serves every right-hand side.
- ``"bddc_gmres"``: NGSolve BDDC, its coarse problem factored by MKL PARDISO
  with gfem's flags when the ``mkl`` extra is installed, by sparse Cholesky
  otherwise, and conjugate gradients (Hermitian operators) or GMRES (the
  others) to the relative preconditioned residual ``tolerance``. GMRES, cold
  or warm, stops at ``|P(b - Ax)| <= tolerance |P b|`` (the residual estimate
  of its left-preconditioned iteration); CG at the P-norm of the residual
  over its first value. The solution is accurate to about ``tolerance``; the
  unpreconditioned ``|b - Ax| / |b|`` can be orders larger at low frequency
  and does not measure the error. A solve that stops early above the
  tolerance (a Krylov breakdown) restarts once from its iterate, and raises
  `gfem.LinearSolverConvergenceError` if it stops short again. GMRES takes
  an initial guess (``solve(B, x0=...)``, e.g. the previous frequency's
  solution, or the previous design's at a design's first frequency); CG
  ignores it. BDDC reads the fixed DOFs from the space, so the operator's
  free DOFs must be the space's ``FreeDofs()``.
- ``"pmg_gmres"``: GMRES on one V-cycle of a two-level p-multigrid
  (`gfem._linalg.pmg`), to the same relative preconditioned residual as
  ``"bddc_gmres"``, with the same preconditioner reuse. Its direct level is
  factored as ``"direct"`` factors; its smoother needs ``A^T = A``. It reads
  the solve domain's cells from ``Operator.cells``.
- ``"cudss"``: the ``direct`` solver on a GPU (``device="cuda"`` or
  ``"cuda:N"``), NVIDIA cuDSS. The host holds all data; the factors stay on
  the GPU. Symmetric and Hermitian operators only. Needs the cuda extra; an
  unusable device raises `gfem.DeviceUnavailableError`, never falls back to
  the CPU.
- ``"petsc"``: a PETSc KSP / PC on the free block (`gfem._linalg.petsc`),
  from complex petsc4py, which gfem does not install. By default exact, a
  MUMPS LDL^T factorization; ``petsc_options`` configure any other KSP / PC,
  and a Krylov method stops at ``tolerance`` times ``|b|`` in the true
  residual. Real symmetric and complex symmetric operators only. Without a
  usable petsc4py it raises `gfem.SolverUnavailableError`, never falls back.

Under ranks ``"direct"`` and ``"petsc"`` are both `gfem._linalg.petsc.RankPetscHandle`
on the MPI PETSc stack, whose absence raises `gfem.SolverUnavailableError`.
``"bddc_gmres"`` needs no PETSc: with the MPI stack its coarse problem moves off rank
0 (`gfem._linalg.coarse`).

`low_rank_update` wraps a handle of ``A`` into one of ``A + U C Vᵀ``
(Woodbury, the update never assembled): wave ports' modal corrections.

`factorize` gives a ``direct`` handle on a real symmetric SciPy matrix, for
auxiliary systems assembled outside NGSolve.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    get_args,
    runtime_checkable,
)

import ngsolve as ngs
import numpy as np

from gfem import _timing

if TYPE_CHECKING:
    from gfem._linalg.layout import DofLayout
    from gfem._linalg.pieces import PieceSum
    from gfem._linalg.pmg import Cells, DirectSetCount
    from gfem.options import PetscOptions, PetscValue, SolverOptions

__all__ = [
    "KINDS",
    "AutoChoice",
    "Handle",
    "Kind",
    "Operator",
    "PmgInputs",
    "Refactorable",
    "ResolvedLinearSolver",
    "Symmetry",
    "auto_linear_solver",
    "factorize",
    "low_rank_update",
    "prepare",
    "refactorize_or_prepare",
    "setup_facts",
]

Symmetry = Literal["complex_symmetric", "hermitian", "general"]
#: A linear solver a problem can use (`ResolvedLinearSolver.kind`).
Kind = Literal["direct", "bddc_gmres", "petsc", "pmg_gmres"]
KINDS: tuple[Kind, ...] = get_args(Kind)
#: Most Krylov iterations of an iterative solve.
MAXITER = 400
#: Bytes per unknown of a direct factorization, above every measured peak
#: (46.9 KiB, sparse Cholesky at order 3; PARDISO 16.7-21.7 KiB):
#: `auto_linear_solver`.
DIRECT_BYTES_PER_UNKNOWN = 48 * 2**10


#: Bytes per unknown of the p-multigrid by order, its direct set's
#: higher-order DOFs aside: the operator and its symmetric copy, the
#: vertex-star factors, the Whitney factor and the Krylov vectors, at or
#: above every measured peak at order 2 (`pmg_estimate`).
PMG_BYTES_PER_UNKNOWN = {2: 8 * 2**10, 3: 16 * 2**10}


class PmgInputs(NamedTuple):
    """What `auto_linear_solver` reads to weigh ``pmg_gmres``.

    Attributes:
        order: The element order, 2 or 3.
        limit: The memory the process may use, in bytes: the p-multigrid
            pools no factor, so its estimate is held to all of it.
        count: Counts the direct set from the mesh
            (`gfem._linalg.pmg.direct_set_count`); called only where direct
            does not fit.
    """

    order: int
    limit: int
    count: Callable[[], DirectSetCount]


class AutoChoice(NamedTuple):
    """What `auto_linear_solver` resolved, and why.

    Attributes:
        kind: The linear solver.
        reason: Why, in words, with the numbers.
        estimate: Direct's estimate, in bytes.
        pmg_estimate: The p-multigrid's estimate, in bytes, where the rule
            weighed it; else ``None``.
        direct_set: The direct set's DOFs it counted; else ``None``.
    """

    kind: Kind
    reason: str
    estimate: int
    pmg_estimate: int | None = None
    direct_set: int | None = None


def auto_linear_solver(
    default: Kind,
    *,
    unknowns: int,
    in_flight: int,
    general: bool,
    budget: int,
    pmg: PmgInputs | None = None,
    reason: str | None = None,
) -> AutoChoice:
    """``"direct"`` while it fits, else ``"pmg_gmres"`` where it fits, else ``default``.

    Direct's estimate is `DIRECT_BYTES_PER_UNKNOWN` times ``unknowns`` per
    factorization ``in_flight``, doubled for a ``general`` operator (LU
    stores L and U). Where it exceeds ``budget`` and ``pmg`` is given (a
    model ``pmg_gmres`` solves), the direct set is counted and
    `pmg_estimate` held to ``pmg.limit``. Equal to the bound fits. Reads no
    clock and no installed library.

    Args:
        default: The linear solver where neither fits.
        unknowns: The system size, an upper bound.
        in_flight: The factorizations held at once (frequency workers).
        general: Whether the operator is general (not ``A^T = A``).
        budget: Direct's bytes.
        pmg: The p-multigrid's inputs, or ``None`` where it does not apply.
        reason: Why ``default`` is the fallback, appended where it is chosen
            past direct's budget.
    """
    estimate = direct_estimate(unknowns, in_flight=in_flight, general=general)
    fits = estimate <= budget
    gib = 2**30
    why = (
        f"{unknowns:,} unknowns, est. {estimate / gib:.1f} GiB factorization "
        f"{'<=' if fits else '>'} {budget / gib:.1f} GiB budget "
        f"(half of {2 * budget / gib:.3g} GiB)"
    )
    if fits:
        return AutoChoice("direct", why, estimate)
    pmg_bytes = direct = None
    if pmg is not None:
        count = pmg.count()
        direct = count.direct
        pmg_bytes = pmg_estimate(
            unknowns,
            direct=direct,
            whitney=count.whitney,
            order=pmg.order,
            in_flight=in_flight,
        )
        below = pmg_bytes <= pmg.limit
        why += (
            f"; p-multigrid est. {pmg_bytes / gib:.1f} GiB (direct set "
            f"{100 * direct / max(unknowns, 1):.1f} %) {'<=' if below else '>'} "
            f"{pmg.limit / gib:.3g} GiB memory"
        )
        if below:
            return AutoChoice("pmg_gmres", why, estimate, pmg_bytes, direct)
    if reason is not None:
        why += f"; {reason}"
    return AutoChoice(default, why, estimate, pmg_bytes, direct)


def pmg_estimate(
    unknowns: int, *, direct: int, whitney: int, order: int, in_flight: int
) -> int:
    r"""Bytes of ``in_flight`` p-multigrid solves of ``unknowns``.

    $n\,(c_p N + c_d (D - W))$: `PMG_BYTES_PER_UNKNOWN` $c_p$ of ``order``
    per unknown, and `DIRECT_BYTES_PER_UNKNOWN` $c_d$ per higher-order DOF of
    the ``direct`` set $D$ (its ``whitney`` DOFs $W$ aside), which is a
    direct factor.
    """
    per_unknown = PMG_BYTES_PER_UNKNOWN[order]
    higher = DIRECT_BYTES_PER_UNKNOWN * (direct - whitney)
    return in_flight * (per_unknown * unknowns + higher)


def direct_estimate(unknowns: int, *, in_flight: int, general: bool) -> int:
    """Bytes of ``in_flight`` direct factorizations of ``unknowns``."""
    return DIRECT_BYTES_PER_UNKNOWN * unknowns * in_flight * (2 if general else 1)


@dataclasses.dataclass(frozen=True, eq=False)
class Operator:
    """A linear operator to prepare: an unassembled form and its free DOFs.

    The form is assembled by `prepare`, because some backends (BDDC) must
    see the element matrices during assembly.

    Attributes:
        form: The NGSolve bilinear form, not yet assembled. Symmetric
            symmetries need ``BilinearForm(..., symmetric=True)``.
        freedofs: The DOFs solved for; the others are fixed (Dirichlet).
        piece_sum: The same operator as a piece sum ``Σ θ_k A_k`` of pieces
            assembled once (`gfem._linalg.pieces`), or ``None``. A linear
            solver may sum it instead of assembling ``form`` (``direct`` and
            ``petsc`` do); the others assemble ``form``.
        layout: How the form's space lies across ranks, on which `prepare`
            dispatches; ``None`` on one process. It cannot be read off the form,
            which is not assembled yet.
        cells: Builds the solve domain's cells (`gfem._linalg.pmg.Cells`),
            which ``pmg_gmres`` reads for its direct set; ``None`` for every
            volume element of the form's mesh, none of them in a perfectly
            matched layer. The other linear solvers ignore it.
        frequency: The frequency the operator is at, in Hz, for a linear
            solver's log; ``None`` when it has none.
    """

    form: ngs.BilinearForm
    freedofs: ngs.BitArray
    piece_sum: PieceSum | None = None
    layout: DofLayout | None = None
    cells: Callable[[], Cells] | None = None
    frequency: float | None = None


class Handle(Protocol):
    """A prepared operator: solve many right-hand sides with one setup.

    ``kind`` names the backend: a `Kind`, or ``"cudss"`` (the ``direct``
    linear solver on a GPU).
    """

    kind: str
    symmetry: Symmetry
    ndof: int
    #: The products of this handle's columns; `gfem._linalg.layout.ONE_PROCESS`
    #: unless it runs across ranks, where ``ndof`` is the rank's.
    layout: DofLayout

    @property
    def nbytes(self) -> int:
        """Host memory held by the matrix and the backend, in bytes.

        Under ranks, this rank's.
        """
        ...

    def solve_counted(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Solve ``A X = B`` on the free DOFs; ``B`` is ``(ndof,)`` or ``(ndof, k)``.

        ``transpose`` solves ``A^T X = B``; of a general operator, only on the
        ``direct`` backend (its factors). ``tolerance`` overrides the prepared
        relative residual of iterative backends for this solve (adjoint
        solves); ``x0`` (the shape of ``B``) is an initial guess for GMRES
        (``bddc_gmres`` on a non-Hermitian operator). CG and the direct
        backends ignore ``x0``; direct backends ignore ``tolerance`` too,
        except that a solve on factors reused across designs reads it.

        Returns ``X`` and the Krylov iterations of each column of ``B``, in
        order: every iteration the column cost in this call (restarts and
        solves again on a preconditioner set up anew included). ``()`` for a
        solve with no Krylov iteration (direct, cudss, PETSc ``preonly``). A
        solve that raises returns no count; its error message holds it.
        """
        ...

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """``X`` alone: ``solve_counted(B, ...)[0]``."""
        ...

    def apply(self, X: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """``A X`` with the full assembled matrix (fixed rows included).

        ``transpose``: ``A^T X``.
        """
        ...

    def free(self) -> None:
        """Release the backend; later calls to `solve` raise."""
        ...


@runtime_checkable
class Refactorable(Handle, Protocol):
    """A handle that refactorizes: a new operator on the same solver plan.

    ``isinstance(handle, Refactorable)`` tells whether a handle can.
    """

    def refactorize(
        self,
        operator: Operator,
        *,
        solves: int | None = None,
        frequency: float | None = None,
        sweep: bool = False,
    ) -> None:
        """Assemble ``operator`` and factorize its values in place.

        A backend may sum the operator's piece sum (``operator.piece_sum``)
        instead of assembling its form. The operator has the handle's
        symmetry and free DOFs (the same space). A changed sparsity pattern
        is planned again, never refused. Later solves use the new operator.
        ``solves``: the solve calls the new factorization is expected to
        serve, when they differ from the count the handle was prepared with
        (`prepare`'s ``solves``); ``None`` keeps it. Only BDDC's coarse
        choice reads it. ``frequency``: the ω the operator is at, when the
        caller knows it; only the direct solver reads it, to solve a design on
        factors made at the same ω for an earlier one. ``sweep``:
        the operator is a frequency sweep's next one; only BDDC reads it, to
        solve without an initial guess on its reused preconditioner
        (otherwise it sets BDDC up again first).

        Raises:
            RuntimeError: The handle was freed.
        """
        ...


def setup_facts(handle: Handle) -> dict[str, Any]:
    """What a ``"setup"`` timing record says of ``handle``: solver, action, bytes.

    The action is the handle's ``last_action`` on a backend that tracks one,
    else ``"factorized"``.
    """
    action = getattr(handle, "last_action", None)
    return {
        "solver": handle.kind,
        "action": action or "factorized",
        "nbytes": int(handle.nbytes),
    }


def _timed_setup[**P](function: Callable[P, Handle]) -> Callable[P, Handle]:
    """``function`` timed as a ``"setup"`` phase on ``gfem.timing``.

    The record names the handle's solver, what it did and its bytes.
    """

    @functools.wraps(function)
    def timed(*args: P.args, **kwargs: P.kwargs) -> Handle:
        with _timing.phase("setup") as span:
            handle = function(*args, **kwargs)
            if span.enabled:
                span.set(**setup_facts(handle))
        return handle

    return timed


@_timed_setup
def refactorize_or_prepare(
    handle: Handle | None,
    operator: Operator,
    prepare_operator: Callable[[Operator], Handle],
    *,
    solves: int | None = None,
    frequency: float | None = None,
    sweep: bool = False,
) -> Handle:
    """A handle of ``operator`` that reuses ``handle``'s solver plan when it can.

    A `Refactorable` ``handle`` refactorizes ``operator`` and is returned;
    any other is freed, and ``prepare_operator(operator)`` is returned.

    Args:
        handle: The handle of the previous operator, or ``None``.
        operator: The new operator (the handle's symmetry and free DOFs).
        prepare_operator: Prepares a new handle (`prepare` with the
            handle's options).
        solves: The solves the refactorization is expected to serve
            (`Refactorable.refactorize`), or ``None`` to keep the handle's
            count; ``prepare_operator`` states its own.
        frequency: The ω ``operator`` is at, or ``None``
            (`Refactorable.refactorize`).
        sweep: ``operator`` is a frequency sweep's next one
            (`Refactorable.refactorize`).

    Raises:
        ValueError: ``operator``'s free-DOF set is not the one ``handle`` was
            prepared with: a refactorization keeps the free DOFs.
    """
    if isinstance(handle, Refactorable):
        handle.refactorize(operator, solves=solves, frequency=frequency, sweep=sweep)
        return handle
    if handle is not None:
        handle.free()
    return prepare_operator(operator)


@_timed_setup
def prepare(
    operator: Operator,
    *,
    symmetry: Symmetry,
    kind: str,
    device: str = "cpu",
    tolerance: float = 1e-6,
    maxiter: int = MAXITER,
    width: int = 1,
    solves: int = 1,
    threads: int | None = None,
    reuse_preconditioner: bool = True,
    petsc_options: Mapping[str, PetscValue] | PetscOptions = (),
) -> Handle:
    """Assemble ``operator`` and set up the ``kind`` backend.

    Args:
        operator: The form and free DOFs.
        symmetry: ``"hermitian"`` (real symmetric or complex Hermitian),
            ``"complex_symmetric"`` (A^T = A) or ``"general"``.
        kind: The backend.
        device: ``"cpu"``, or ``"cuda"`` / ``"cuda:N"`` for the ``direct``
            backend on a GPU (cuDSS). The GPU is checked before assembly.
        tolerance: Relative preconditioned residual of iterative backends
            (``|P(b - Ax)| / |P b|`` for BDDC + GMRES, the P-norm of the
            residual over its first value for CG). The direct backend reads
            it only on factors reused across designs.
        maxiter: Most iterations of iterative backends; a solve that does
            not converge in them raises `gfem.LinearSolverConvergenceError`.
        width: Right-hand-side columns of a usual solve (the block width:
            terminals, ports). The GPU solver solves that many at once; wider
            blocks go in chunks, narrower ones are zero-padded. BDDC chooses
            its coarse factorization from it (see ``solves``); the other CPU
            backends ignore it.
        solves: Solve calls the handle is expected to serve: a uniform
            sweep's frequencies, an eigen shift's seed solve. BDDC chooses
            its coarse factorization from ``width * solves`` (``width``
            without preconditioner reuse); other backends ignore it.
        threads: ``SolverOptions.threads``. ``1`` keeps the GPU solver's
            host-side planning on one thread; otherwise it is multithreaded
            when the cuDSS threading library is installed. CPU backends
            ignore it.
        reuse_preconditioner: Iterative backends: keep the preconditioner
            across `refactorize` (preconditioner reuse); False sets it up
            again at every refactorization. Direct backends ignore it.
        petsc_options: The linear solver's PETSc options (the linear half of
            ``SolverOptions.petsc_options``, `ResolvedLinearSolver`):
            ``kind="petsc"`` reads them over gfem's defaults; other linear
            solvers ignore them.

    Raises:
        NotImplementedError: The backend is not available, or the GPU
            solver gets a general operator.
        ValueError: The backend cannot handle this symmetry, a GPU is
            asked for an iterative backend, or PETSc refused or did not read
            some of ``petsc_options``.
        RuntimeError: PETSc's set-up failed (``kind="petsc"``: a singular
            operator, or an option PETSc cannot apply).
        gfem.DeviceUnavailableError: The GPU cannot be used.
        gfem.SolverUnavailableError: ``kind="petsc"`` and PETSc cannot be
            used (checked before assembly), or ``"direct"`` or ``"petsc"`` under
            ranks without the MPI PETSc stack.
        gfem._mpi.AgreedError: Under ranks, MUMPS failed on every retry.
    """
    if operator.layout is not None:
        return _prepare_on_ranks(
            operator,
            operator.layout,
            symmetry=symmetry,
            kind=kind,
            tolerance=tolerance,
            maxiter=maxiter,
            width=width,
            solves=solves,
            reuse_preconditioner=reuse_preconditioner,
            petsc_options=tuple(dict(petsc_options).items()),
        )
    if device != "cpu":
        from gfem import _cuda

        if kind != "direct":
            raise ValueError(
                f"device={device!r} needs kind='direct': only the direct solver "
                f"runs on a GPU, not {kind!r}."
            )
        device_id = _cuda.require(device)
        _cuda.warn_if_jax_preallocates()
        from gfem._linalg.cudss import CudssHandle

        return CudssHandle.from_operator(
            operator, symmetry, device_id=device_id, width=width, threads=threads
        )
    if kind == "petsc":
        from gfem import _petsc

        _petsc.require_petsc("linear_solver")
        from gfem._linalg.petsc import PetscHandle

        return PetscHandle.from_operator(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            options=tuple(dict(petsc_options).items()),
            reuse=reuse_preconditioner,
        )
    if kind == "direct":
        from gfem._linalg.direct import DirectHandle

        return DirectHandle.from_operator(operator, symmetry, tolerance=tolerance)
    if kind == "bddc_gmres":
        from gfem._linalg.bddc import BddcHandle

        return BddcHandle(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            reuse=reuse_preconditioner,
            width=width,
            solves=solves,
        )
    if kind == "pmg_gmres":
        from gfem._linalg.pmg import PmgHandle

        return PmgHandle(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            reuse=reuse_preconditioner,
            width=width,
            solves=solves,
        )
    raise _unavailable(kind)


def _unavailable(kind: str) -> NotImplementedError:
    """The error of a ``kind`` that is no backend."""
    return NotImplementedError(
        f"linear_solver={kind!r} is not available in this version of gfem; use "
        f"one of {', '.join(repr(k) for k in KINDS)}."
    )


def _driven_only(problem: str, default: Kind) -> str:
    """The refusal of ``pmg_gmres`` for a ``problem`` problem, naming its ``"auto"``."""
    if default == "direct":
        remedy = "Use linear_solver='direct' (or 'auto', which picks it)."
    else:
        remedy = f"Use linear_solver='auto', which picks {default!r}, or 'direct'."
    return (
        "linear_solver='pmg_gmres' solves driven problems only, not "
        f"{problem} problems. {remedy}"
    )


def _prepare_on_ranks(
    operator: Operator,
    layout: DofLayout,
    *,
    symmetry: Symmetry,
    kind: str,
    tolerance: float,
    maxiter: int,
    width: int,
    solves: int,
    reuse_preconditioner: bool,
    petsc_options: PetscOptions,
) -> Handle:
    """`prepare` across ranks: a backend that takes a layout.

    ``direct`` and ``bddc_gmres`` take an operator without :math:`A^T = A` (MUMPS LU,
    and BDDC with an LU coarse problem); ``petsc`` refuses it as on one process.

    Raises:
        NotImplementedError: No backend of ``kind``, on every rank.
        gfem.DistributedUnsupportedError: ``pmg_gmres``, which runs on one
            process only, on every rank.
        ValueError: ``petsc`` and an operator without :math:`A^T = A`, on every rank.
        gfem.SolverUnavailableError: ``direct`` or ``petsc`` without the MPI PETSc
            stack.
    """
    from gfem import _mpi
    from gfem._linalg.bddc import BddcHandle

    if kind == "pmg_gmres":
        # `gfem._mpi.refuse` stops it before any mesh; a caller below that meets
        # the same refusal, on every rank
        _mpi.refuse_feature("pmg_gmres")
    if kind not in KINDS:
        raise _mpi.agreed(_unavailable(kind))
    if kind in ("direct", "petsc"):
        from gfem import _petsc

        _petsc.require_mpi_petsc(kind, layout.comm)
        from gfem._linalg.petsc import RankPetscHandle

        return RankPetscHandle.from_operator(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            layout=layout,
            kind=kind,
            options=petsc_options if kind == "petsc" else (),
            reuse=reuse_preconditioner,
        )
    return BddcHandle(
        operator,
        symmetry,
        tolerance=tolerance,
        maxiter=maxiter,
        reuse=reuse_preconditioner,
        width=width,
        solves=solves,
    )


@dataclasses.dataclass(frozen=True)
class ResolvedLinearSolver:
    """A problem's linear solver, resolved once from its `gfem.SolverOptions`.

    A rule resolves it at construction, before any assembly, and prepares
    every operator of its solve with it; callers pass only what changes per
    call.

    Attributes:
        kind: The linear solver, never ``"auto"``.
        device: ``SolverOptions.device``.
        threads: ``SolverOptions.threads``.
        tolerance: ``SolverOptions.tolerance``, the prepared default.
        petsc_options: The linear half of ``SolverOptions.petsc_options``
            (`gfem.options.split_petsc_options`), split once.
    """

    kind: Kind
    device: str
    threads: int | None
    tolerance: float
    petsc_options: PetscOptions

    @classmethod
    def resolve(
        cls,
        options: SolverOptions,
        *,
        default: Kind,
        problem: str,
        comm: Any = None,
        unknowns: int | None = None,
        in_flight: int = 1,
        general: bool = False,
        budget: int | None = None,
        reason: str | None = None,
        pmg: PmgInputs | None = None,
    ) -> Self:
        """The linear solver of ``options`` for a ``problem`` problem.

        With ``unknowns``, ``"auto"`` on one CPU process is `auto_linear_solver`
        (``"direct"`` while it fits ``budget``, else ``"pmg_gmres"`` where
        ``pmg`` is given and fits, else ``default``). A resolved
        ``"auto"`` is logged on ``gfem.solver`` (`report.auto_choice`); an
        explicit choice is not.

        Args:
            options: The solver options.
            default: The problem type's linear solver for ``"auto"`` on the
                CPU; on a GPU ``"auto"`` is ``"direct"``.
            problem: The problem type, for the error message.
            comm: The resolved communicator of a distributed solve.
            unknowns: The system size, an upper bound, for the rule.
            in_flight: The factorizations held at once (frequency workers).
            general: Whether the operator is general (not ``A^T = A``).
            budget: The rule's bytes; required with ``unknowns``.
            reason: Why ``"auto"`` takes ``default`` without ``unknowns``, or
                past ``budget``, for the log.
            pmg: The p-multigrid's inputs for the rule, or ``None`` where
                ``pmg_gmres`` does not solve the model.

        Raises:
            ValueError: ``linear_solver="pmg_gmres"`` for a problem other than
                ``"driven"``.
            NotImplementedError: The linear solver is not available.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used, or ``"direct"`` or ``"petsc"`` under ``comm``
                without the MPI PETSc stack.
        """
        from gfem._linalg import report
        from gfem.options import split_petsc_options

        name = options.linear_solver
        if name == "pmg_gmres" and problem != "driven":
            raise ValueError(_driven_only(problem, default))
        if name != "auto":
            kind = name
        elif options.device != "cpu":
            kind = "direct"  # the only solver that runs on a GPU
            report.auto_choice(kind, problem=problem)
        elif comm is not None or unknowns is None:
            kind = default
            report.auto_choice(kind, problem=problem, reason=reason)
        else:
            if budget is None:  # pragma: no cover - a caller error
                raise TypeError("resolve needs a budget with unknowns.")
            choice = auto_linear_solver(
                default,
                unknowns=unknowns,
                in_flight=in_flight,
                general=general,
                budget=budget,
                pmg=pmg,
                reason=reason,
            )
            kind = choice.kind
            report.auto_choice(
                kind,
                problem=problem,
                reason=choice.reason,
                unknowns=unknowns,
                estimate=choice.estimate,
                budget=budget,
                pmg_estimate=choice.pmg_estimate,
                direct_set=choice.direct_set,
            )
        if kind not in KINDS:  # pragma: no cover - SolverOptions validates
            raise NotImplementedError(
                f"linear_solver={name!r} is not available for {problem} problems "
                "in this version of gfem; use one of "
                f"{', '.join(repr(k) for k in KINDS)}."
            )
        if kind == "petsc" or (comm is not None and kind == "direct"):
            # before any assembly, and before any worker
            from gfem import _petsc

            if comm is None:
                _petsc.require_petsc("linear_solver")
            else:
                _petsc.require_mpi_petsc(kind, comm)
        pairs = tuple(dict(options.petsc_options).items())
        return cls(
            kind=kind,
            device=options.device,
            threads=options.threads,
            tolerance=options.tolerance,
            petsc_options=split_petsc_options(pairs)[0],
        )

    def prepare(
        self,
        operator: Operator,
        *,
        symmetry: Symmetry,
        width: int = 1,
        solves: int = 1,
        tolerance: float | None = None,
        reuse_preconditioner: bool = True,
    ) -> Handle:
        """Assemble ``operator`` and set up this linear solver (`prepare`).

        ``tolerance`` overrides the prepared default (adjoint solves).
        """
        return prepare(
            operator,
            symmetry=symmetry,
            kind=self.kind,
            device=self.device,
            tolerance=self.tolerance if tolerance is None else tolerance,
            width=width,
            solves=solves,
            threads=self.threads,
            reuse_preconditioner=reuse_preconditioner,
            petsc_options=self.petsc_options,
        )


def factorize(matrix) -> Handle:
    """A ``direct`` handle on a real symmetric positive definite SciPy matrix.

    Every DOF is free. The ``direct`` backend's factorization, as for `prepare`.

    Raises:
        ValueError: ``matrix`` is not square.
    """
    from gfem._linalg.direct import DirectHandle

    return DirectHandle.from_scipy(matrix)


def low_rank_update(
    handle: Handle,
    vectors: np.ndarray,
    coefficients,
    *,
    right: np.ndarray | None = None,
    infinite: np.ndarray | None = None,
    row_scaled: bool = False,
    borrow: bool = False,
) -> Handle:
    """A handle of ``A + U C Vᵀ`` from a handle of ``A`` (Woodbury).

    See `gfem._linalg.lowrank`. The result owns ``handle`` unless it borrows
    it.

    Args:
        handle: The prepared base operator ``A``.
        vectors: ``U``, ``(ndof, p)``, zero on the fixed DOFs.
        coefficients: ``C``, ``(p, p)``.
        right: ``V``, as ``U``; ``None`` for ``U`` (``A + U C Uᵀ``).
        infinite: ``(p,)`` bools: the rows whose diagonal shift is
            infinite, each an exact constraint row (their row and column of
            ``C`` zero); ``None`` for none.
        row_scaled: Divide each capacitance row with a large shift by it.
        borrow: Leave ``handle`` to its owner: freeing the result releases
            only the update's own columns.
    """
    from gfem._linalg.lowrank import LowRankUpdate

    return LowRankUpdate(
        handle,
        vectors,
        coefficients,
        right=right,
        infinite=infinite,
        row_scaled=row_scaled,
        borrow=borrow,
    )
