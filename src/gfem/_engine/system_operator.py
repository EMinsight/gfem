"""The system operator of a driven solve: ``A(ω) + W(ω)`` and every handle of it.

``A(ω) = Σ θ_k(ω) A_k`` is the driven operator with the wave ports' Robin
pieces, and ``W(ω) = U C Uᵀ`` the wave ports' rank-two modal corrections
(`gfem._engine.wave_ports`), never assembled: a handle of ``A + W`` is a
Woodbury update of a handle of ``A`` (`gfem._linalg.low_rank_update`), for
forward and adjoint solves alike (``A + W`` is complex symmetric without a
Bloch wave vector; with one, the operator's symmetry is ``"general"``). The
``direct`` and ``petsc`` linear solvers sum ``A(ω)`` from the pieces ``A_k``
assembled once (the piece sum, `gfem._linalg.pieces`), kept until `free`.

A `SystemOperator` answers two questions: a forward sweep's right-hand
sides at each ω, warm-started, through one base handle (`SystemSweep`), and
a backward pass's adjoint loads at frequency ``n`` (`solve_adjoint`), at the
adjoint tolerance. The adjoint solve is a solve with ``(A + W)^T``: the
operator itself when it is complex symmetric, and under a Bloch wave vector
k the transposed system, the system operator at -k (``A(k)^T = A(-k)`` on
the same DOFs, and ``W`` is symmetric and off every pair), built on the
first adjoint solve. Adjoint solves refactorize one base handle at each
frequency, so memory holds one factorization whatever the number of
frequencies: on a GPU, and, starting from the forward's base handle
(`keep_base`, ADR 0065), under ranks and on one process with the CPU
``direct`` and ``bddc_gmres`` solvers. Elsewhere on the CPU (the transposed
system, the ``petsc`` solver, a backward without its forward's system) an
adjoint solve keeps one handle per frequency for later backward passes.

On one process with the CPU ``direct`` solver, each volume piece of an operator
with material-field regions (`FixedPieces`) is summed as its part on those
regions, assembled every solve, plus its part on the other elements, kept per
discretization behind a digest of its coefficient there (ADR 0082). S moves by
summation round-off only.

There, too, a forward sweep takes its base handle from the plan pool
(`gfem._engine.cache`) and refactorizes it, rather than planning again
(`PlanSlot`). A differentiated forward's base moves to its entry and returns to
the pool when the consumed entry is freed (ADR 0081); any other returns when
the sweep ends.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Any, Self

import ngsolve as ngs
import numpy as np
import scipy.linalg

from gfem._engine.cache import ENTRIES
from gfem._engine.wave_ports import PortSurfaceCoupling
from gfem._linalg import (
    Handle,
    Operator,
    ResolvedLinearSolver,
    low_rank_update,
    refactorize_or_prepare,
)
from gfem._linalg.base import free_mask
from gfem._linalg.pieces import (
    AssembledPieces,
    FixedPart,
    Integrand,
    KeptParts,
    PieceSum,
)

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenOperator
    from gfem._engine.terms import Piece

__all__ = ["FixedPieces", "PlanSlot", "SystemOperator", "SystemSweep"]


@dataclasses.dataclass(frozen=True, eq=False)
class FixedPieces:
    """How a driven operator's volume pieces split at its material-field regions.

    Attributes:
        store: The kept fixed parts of the operator's space (per discretization).
        outside: Per gfem tetrahedron, whether it lies outside every
            `gfem.MaterialField` region: the fixed parts' elements.
        varying: Builds the pieces restricted to the material-field regions
            (`gfem._engine.terms.pieces` with ``elements``), on demand.
    """

    store: KeptParts
    outside: np.ndarray
    varying: Callable[[], list[Piece]]

    def split(
        self, pieces: Sequence[Piece]
    ) -> tuple[list[Integrand], list[FixedPart | None]]:
        """Per piece, the integrand to assemble at every build and its fixed part.

        A volume piece (one with coefficient values) gives its varying part
        and a `FixedPart` keyed on a digest of its coefficient outside the
        material-field regions; every other piece itself and ``None``.
        """
        varying = {p.name: p for p in self.varying() if p.values is not None}
        integrands: list[Integrand] = []
        fixed: list[FixedPart | None] = []
        for piece in pieces:
            part = varying.get(piece.name)
            if piece.values is None or part is None:
                integrands.append(piece.integrand)
                fixed.append(None)
                continue
            values = np.ascontiguousarray(piece.values[self.outside])
            digest = hashlib.blake2b(digest_size=16)
            digest.update(f"{values.dtype.str}{values.shape}".encode())
            digest.update(values.tobytes())
            integrands.append(part.integrand)
            fixed.append(
                FixedPart(self.store, piece.name, digest.hexdigest(), piece.integrand)
            )
        return integrands, fixed


class SystemOperator:
    """The system operator ``A(ω) + W(ω)`` of one driven solve, and every handle of it.

    Owns the wave ports' Robin pieces, the piece sum, the modal corrections
    ``W``, the adjoint handles (one refactorized base, else one per
    frequency on the CPU), the transposed system under a Bloch wave vector,
    and their memory. A forward sweep's base handle lives in its
    `SystemSweep`.

    Attributes:
        operator: The driven operator, wave-port Robin pieces included.
        surfaces: The wave port surfaces' couplings.
    """

    def __init__(
        self,
        operator: DrivenOperator,
        surfaces: Sequence[PortSurfaceCoupling],
        linear: ResolvedLinearSolver,
        *,
        width: int,
        adjoint_tolerance: float,
        transposed: Callable[[], DrivenOperator] | None = None,
    ) -> None:
        """Add each wave port surface's Robin piece to ``operator``.

        Args:
            operator: The driven operator.
            surfaces: The wave port surfaces' couplings.
            linear: The rule's linear solver.
            width: The right-hand sides per solve (the excited ports).
            adjoint_tolerance: The relative residual of adjoint solves.
            transposed: Builds the driven operator of ``operator``'s
                transpose (the operator at -k under a Bloch wave vector), for
                the adjoint solves; ``None`` when ``operator`` is complex
                symmetric.
        """
        if surfaces:
            operator = dataclasses.replace(
                operator, pieces=[*operator.pieces, *(c.piece for c in surfaces)]
            )
        self.operator = operator
        self.surfaces = list(surfaces)
        self._linear = linear
        self._width = width
        self._adjoint_tolerance = adjoint_tolerance
        # assembled at the first direct prepare (the piece sum); a backward
        # pass reuses them, `free` drops them
        integrands = [piece.integrand for piece in operator.pieces]
        fixed = None
        if (
            operator.fixed is not None
            and operator.built.comm is None
            and (linear.kind, linear.device) == ("direct", "cpu")
        ):
            integrands, fixed = operator.fixed.split(operator.pieces)
        self._pieces = AssembledPieces(
            operator.space,
            integrands,
            symmetric=operator.symmetric,
            comm=operator.built.comm,
            fixed=fixed,
        )
        self._handles: dict[int, Handle] = {}
        self._adjoint_sweep: SystemSweep | None = None
        self._transposed = transposed
        self._transposed_system: SystemOperator | None = None

    def sweep(
        self,
        *,
        solves: int = 1,
        reuse_preconditioner: bool = True,
        tolerance: float | None = None,
        pool_budget: int | None = None,
    ) -> SystemSweep:
        """One base handle for a sweep's frequencies (see `SystemSweep`).

        ``reuse_preconditioner=False`` sets an iterative backend's
        preconditioner up at every frequency, ``solves`` (a uniform sweep's
        frequencies) is the solve calls the base is expected to serve and
        ``tolerance`` overrides the linear solver's (adjoint solves)
        (`gfem._linalg.ResolvedLinearSolver.prepare`). With a ``pool_budget``
        (``retain_memory``'s bytes) a serial CPU direct sweep takes its base
        from the plan pool and returns it there (`PlanSlot`).
        """
        return SystemSweep(
            self.operator_at,
            functools.partial(self._system_handle, borrow=True),
            functools.partial(
                self._linear.prepare,
                symmetry=self.operator.symmetry,
                width=self._width,
                solves=solves,
                tolerance=tolerance,
                reuse_preconditioner=reuse_preconditioner,
            ),
            plans=None if pool_budget is None else self._plan_slot(pool_budget),
        )

    def _plan_slot(self, budget: int) -> PlanSlot | None:
        """Where this operator's base handles pool, or ``None`` off the pool.

        Serial CPU direct solves only (ADR 0082). The key holds the topology,
        the space key, a digest of the free DOFs and the symmetry; the sweep
        adds the thread count.
        """
        op = self.operator
        if (
            op.built.comm is not None
            or op.topology is None
            or (self._linear.kind, self._linear.device) != ("direct", "cpu")
        ):
            return None
        free = free_mask(op.space.FreeDofs(), op.space.ndof)
        digest = hashlib.blake2b(np.packbits(free).tobytes(), digest_size=16)
        site = (id(op.topology), op.space_key, digest.hexdigest(), op.symmetry)
        return PlanSlot(site, op.topology, budget)

    def solve_adjoint(self, n: int, omega: float, loads: np.ndarray) -> np.ndarray:
        """Solve the adjoint loads ``(ndof, q)`` of frequency ``n`` at ``ω``.

        At the adjoint tolerance, with ``(A + W)^T``: ``A + W`` itself when
        complex symmetric (the adjoint solve is a forward solve), else the
        transposed system's.

        Raises:
            gfem.LinearSolverConvergenceError: The linear solve did not
                converge.
        """
        handle = self._adjoint_system()._adjoint_handle(n, omega)  # noqa: SLF001
        return handle.solve(loads, tolerance=self._adjoint_tolerance)

    @property
    def nbytes(self) -> int:
        """Host memory held by the adjoint handles, the pieces and ``W``, in bytes.

        The transposed system's handles and pieces included.
        """
        held = self._held_nbytes()
        if self._transposed_system is not None:
            held += self._transposed_system._held_nbytes()  # noqa: SLF001
        return held + sum(int(c.vectors.nbytes) for c in self.surfaces)

    def free(self, *, return_plans: bool = False) -> None:
        """Release the adjoint handles and the pieces (a prepare reassembles them).

        The transposed system goes too (the next adjoint solve builds it). With
        ``return_plans`` the kept base handle returns to the plan pool, where it
        came from one (`free_base`).
        """
        for handle in self._handles.values():
            handle.free()
        self._handles = {}
        self.free_base(return_plan=return_plans)
        self._adjoint_sweep = None
        self._pieces.free()
        if self._transposed_system is not None:
            self._transposed_system.free()
        self._transposed_system = None

    def keep_base(self, sweep: SystemSweep) -> None:
        """Move a finished forward sweep's base handle to the adjoint solves.

        Under ranks, where one factorization per frequency does not fit and a
        backward must not prepare what its forward just freed (ADR 0065), and on
        one process with the CPU ``direct`` and ``bddc_gmres`` solvers when the
        operator is its own adjoint's, so that the backward refactorizes it rather
        than preparing a handle per frequency. A base it no longer holds is
        prepared again at the adjoint tolerance. A base from the plan pool moves
        in any case, so that it returns to the pool only when the entry holding it
        is freed (ADR 0082); any other sweep keeps its base, and ``with`` frees
        it.
        """
        if (
            self.operator.built.comm is None
            and not sweep.pooled
            and not self._shares_base()
        ):
            return
        adjoint = self.sweep(tolerance=self._adjoint_tolerance)
        sweep.move_base(adjoint)
        # each adjoint frequency is one solve call: BDDC's coarse choice reads it
        adjoint.expect_solves(1)
        self._adjoint_sweep = adjoint

    def _shares_base(self) -> bool:
        """Whether one-process adjoint solves refactorize the forward's base.

        The CPU ``direct`` and ``bddc_gmres`` solvers, without a transposed
        system (whose adjoint solves are on another operator).
        """
        return (
            self._transposed is None
            and self._linear.device == "cpu"
            and self._linear.kind in ("direct", "bddc_gmres")
        )

    def free_base(self, *, return_plan: bool = False) -> None:
        """Release the adjoint solves' base handle; the sweep prepares it again.

        With ``return_plan`` a base from the plan pool returns there instead.
        """
        if self._adjoint_sweep is not None:
            self._adjoint_sweep.free(return_plan=return_plan)

    def _held_nbytes(self) -> int:
        """The adjoint handles' and the pieces' bytes (not ``W``'s)."""
        held = sum(int(h.nbytes) for h in self._handles.values())
        if self._adjoint_sweep is not None:  # the adjoint base
            base = self._adjoint_sweep._base  # noqa: SLF001
            held += 0 if base is None else int(base.nbytes)
        return held + self._pieces.nbytes

    def _adjoint_system(self) -> SystemOperator:
        """The system whose handles solve with ``(A + W)^T``.

        ``self`` without a transposed operator; otherwise the transposed
        system, built on first use with the same wave ports, solver and
        tolerances.
        """
        if self._transposed is None:
            return self
        if self._transposed_system is None:
            self._transposed_system = SystemOperator(
                self._transposed(),
                self.surfaces,
                self._linear,
                width=self._width,
                adjoint_tolerance=self._adjoint_tolerance,
            )
        return self._transposed_system

    def operator_at(self, omega: float) -> Operator:
        """``A(ω)``, the operator without the modal corrections, to prepare.

        Its form, and its piece sum for the ``direct`` and ``petsc`` linear
        solvers.
        """
        op = self.operator
        weights = tuple(complex(piece.theta(omega)) for piece in op.pieces)
        return Operator(
            op.form(omega),
            op.space.FreeDofs(),
            piece_sum=PieceSum(self._pieces, weights),
            layout=op.layout,
        )

    def corrections(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """``W(ω) = U C Uᵀ``: ``U`` ``(ndof, 2 n_wave)``, ``C`` block diagonal.

        One block per wave port surface, two columns per wave port on it
        (see `PortSurfaceCoupling.correction`).
        """
        columns, blocks = [], []
        for c in self.surfaces:
            p, d = c.correction(omega)
            columns.append(c.vectors @ p)
            blocks.append(d)
        return np.concatenate(columns, axis=1), scipy.linalg.block_diag(*blocks)

    def _system_handle(self, base: Handle, omega: float, *, borrow: bool) -> Handle:
        """A handle of ``A(ω) + W(ω)`` from a handle ``base`` of ``A(ω)``.

        ``base`` itself without wave ports; otherwise a Woodbury update that
        owns ``base`` or ``borrow``s it.
        """
        if not self.surfaces:
            return base
        return low_rank_update(base, *self.corrections(omega), borrow=borrow)

    def _adjoint_handle(self, n: int, omega: float) -> Handle:
        """The adjoint handle of ``A(ω) + W(ω)`` of frequency ``n``.

        On a GPU the adjoint base refactorized at ``ω``, valid until the next
        call. Under ranks, and on one process once the forward's base was kept
        (`keep_base`), the same, except that a base already at ``ω`` serves as
        it is: a backward starting from the base's frequency factorizes once
        fewer. Otherwise on the CPU prepared once per ``n`` and kept.
        """
        if self.operator.built.comm is not None:
            if self._adjoint_sweep is None:
                self._adjoint_sweep = self.sweep(tolerance=self._adjoint_tolerance)
            return self._adjoint_sweep.reuse_at(omega)
        if self._linear.device != "cpu":
            if self._adjoint_sweep is None:
                self._adjoint_sweep = self.sweep(tolerance=self._adjoint_tolerance)
            return self._adjoint_sweep.at(omega)
        if self._adjoint_sweep is not None:
            return self._adjoint_sweep.reuse_at(omega)
        if n not in self._handles:
            handle = self._linear.prepare(
                self.operator_at(omega),
                symmetry=self.operator.symmetry,
                width=self._width,
                tolerance=self._adjoint_tolerance,
            )
            self._handles[n] = self._system_handle(handle, omega, borrow=False)
        return self._handles[n]


class SystemSweep:
    """One base handle of ``A(ω)`` for a whole sweep: one solver plan.

    `at` gives a handle of the system operator ``A(ω) + W(ω)`` at each
    frequency. The first call prepares the base handle; later ones
    refactorize it at the new ω when it can, or else free it and prepare a
    new one (`gfem._linalg.refactorize_or_prepare`). An iterative base keeps
    its preconditioner across frequencies (preconditioner reuse) unless
    prepared with ``reuse_preconditioner=False``, as the adaptive sweep's
    is. Wave ports wrap the base per frequency in a borrowing Woodbury
    update. A handle from `at` is valid until the next call. `free` (or
    leaving the ``with`` block) releases the base, also when the sweep
    raises, and the sweep prepares it again if asked.

    With a `PlanSlot` the first call takes the base from the plan pool when a
    handle of its key is free there, and refactorizes it; leaving the ``with``
    block without an error returns the base to the pool.
    """

    def __init__(
        self,
        operator_at: Callable[[float], Operator],
        system_handle: Callable[[Handle, float], Handle],
        prepare: Callable[[Operator], Handle],
        plans: PlanSlot | None = None,
    ) -> None:
        """Hold ``ω -> A(ω)``, the wrap into ``A(ω) + W(ω)`` and the base's prepare.

        The wrap borrows the base handle. ``plans``: where the base pools.
        """
        self._operator_at = operator_at
        self._system_handle = system_handle
        self._prepare = prepare
        self._plans = plans
        # the pool key of the base, set when the base was taken or prepared
        self._key: Hashable | None = None
        self._base: Handle | None = None
        # the system handle `at` gave last, and its frequency
        self._last: tuple[float, Handle] | None = None

    @property
    def pooled(self) -> bool:
        """Whether the base returns to the plan pool."""
        return self._plans is not None

    def at(self, omega: float) -> Handle:
        """A handle of ``A(ω) + W(ω)``, borrowing the sweep's base handle."""
        operator = self._operator_at(omega)
        if self._base is None and self._plans is not None:
            self._key = self._plans.key()
            self._base = ENTRIES.take_plan(self._key)
        self._base = refactorize_or_prepare(self._base, operator, self._prepare)
        handle = self._system_handle(self._base, omega)
        self._last = (omega, handle)
        return handle

    def reuse_at(self, omega: float) -> Handle:
        """As `at`, but the last handle if it is still at ``ω``, Woodbury state too."""
        if self._last is not None and self._last[0] == omega:
            return self._last[1]
        return self.at(omega)

    def expect_solves(self, solves: int) -> None:
        """Restate the solve calls the base's next setups serve, if it reads them.

        A base moved from a sweep that expected another count (BDDC's coarse
        choice, `gfem._linalg.bddc.BddcHandle.expect_solves`).
        """
        expect = getattr(self._base, "expect_solves", None)
        if expect is not None:
            expect(solves)

    def move_base(self, other: SystemSweep) -> None:
        """Give ``other`` this sweep's base handle; this one holds none after.

        The base's place in the plan pool goes with it.
        """
        other._base, other._last = self._base, self._last
        other._plans, other._key = self._plans, self._key
        self._base, self._last = None, None

    def solve(
        self, omega: float, rhs: np.ndarray, *, x0: np.ndarray | None = None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Solve at ``ω``: the solution and the Krylov iterations per column."""
        return self.at(omega).solve_counted(rhs, x0=x0)

    def free(self, *, return_plan: bool = False) -> None:
        """Release the base handle.

        With ``return_plan`` a pooled sweep's base returns to the plan pool
        instead, its factors released.
        """
        base, plans, key = self._base, self._plans, self._key
        self._base, self._last = None, None
        if base is None:
            return
        if return_plan and plans is not None and key is not None:
            ENTRIES.give_plan(key, plans.topology, base, plans.budget)
        else:
            base.free()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, error: type[BaseException] | None, *exc: object) -> None:
        # a sweep that raised may leave its base mid-refactorization
        self.free(return_plan=error is None)


@dataclasses.dataclass(frozen=True, eq=False)
class PlanSlot:
    """Where a serial CPU direct sweep's base handle pools (ADR 0082).

    Attributes:
        site: The pool key but the thread count: topology identity, space
            key, free-DOF digest, symmetry. Node positions are not in it.
        topology: The mesh topology (the pool frees its handles once no
            discretization of it is left).
        budget: ``retain_memory``'s bytes, the walk's budget on return.
    """

    site: tuple
    topology: Any
    budget: int

    def key(self) -> tuple:
        """The pool key: ``site`` and the thread count of a plan made now.

        A PARDISO handle runs on the thread count of its analysis (ADR 0032),
        so another count plans anew.
        """
        return (*self.site, max(1, ngs.GetNumThreads()))
