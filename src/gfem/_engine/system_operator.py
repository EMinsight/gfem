"""The system operator of a driven solve: ``A(ω) + W(ω)`` and every handle of it.

``A(ω) = Σ θ_k(ω) A_k`` is the driven operator with the port surfaces' Robin
pieces, and ``W(ω) = U C Vᵀ`` the port surfaces' low-rank corrections, one
`LowRankTerm` per surface coupling, never assembled: a handle of ``A + W`` is a
Woodbury update of a handle of ``A`` (`gfem._linalg.low_rank_update`), for
forward and adjoint solves alike (``A + W`` is complex symmetric without a
Bloch wave vector; with one, the operator's symmetry is ``"general"``). A wave
port surface's rank-two modal correction (`gfem._engine.wave_ports`) is ``U C
Uᵀ`` with ``C`` complex symmetric (``V = U``); a coupling may instead give its
own term with a right factor ``V`` (a Floquet surface's ``U C Uᴴ``, ``V =
conj(U)``), infinite diagonal shifts and a row-scaled capacitance
(`gfem._linalg.lowrank`). The ``direct`` and ``petsc`` linear solvers sum
``A(ω)`` from the pieces ``A_k`` assembled once (the piece sum,
`gfem._linalg.pieces`), kept until `free`.

A `SystemOperator` answers two questions: a forward sweep's right-hand
sides at each ω, warm-started, through one base handle (`SystemSweep`), and
a backward pass's adjoint loads at frequency ``n`` (`solve_adjoint`), at the
adjoint tolerance. The adjoint solve is a solve with ``(A + W)^T``: the
operator itself when it is complex symmetric, and under a Bloch wave vector
k the transposed system, the system operator at -k (``A(k)^T = A(-k)`` on
the same DOFs), built on the first adjoint solve. Its corrections are the
transposes of the forward's: a wave port's ``W`` is symmetric and off every
pair, so its coupling serves both systems; a coupling that is not symmetric
gives its transposed form (``transposed()``, a Floquet surface's ``conj(U)
Cᵀ Uᵀ``, read off the forward vectors), which the transposed system takes in
its place. Adjoint solves refactorize one base handle at each
frequency, so memory holds one factorization whatever the number of
frequencies: on a GPU, and, starting from the forward's base handle
(`keep_base`, ADR 0065), under ranks and on one process with the CPU
``bddc_gmres`` solver. On one process with the CPU ``direct`` solver they take
each frequency's base from the plan pool instead (frequency bases, below), the
transposed system's under the -k key (ADR 0092). Elsewhere on the CPU (the
transposed system off that solver, the ``petsc`` solver, a backward without
its forward's system) an adjoint solve keeps one handle per frequency for
later backward passes.

On one process with the CPU ``direct`` solver, each volume piece of an operator
with material-field regions (`FixedPieces`) is summed as its part on those
regions, assembled every solve, plus its part on the other elements, kept per
discretization behind a digest of its coefficient there and of the regions'
layout (ADR 0082). S moves by summation round-off only.

There, too, a forward sweep takes its base handle from the plan pool
(`gfem._engine.cache`) and refactorizes it, rather than planning again
(`PlanSlot`). It holds one base per frequency (frequency bases, ADR 0121): at
each new frequency the base returns to the pool under the frequency it was
at, and the base of the new one is taken, which still holds that frequency's
factors of the last design. A frequency gets its own base only the second
time the site solves at it: the first time the sweep refactorizes the base it
holds, so a one-shot sweep keeps one base, and a design loop plans its
frequency bases at its second design. A backward's adjoint solves take them the same
way, each still holding its design's values, and refactorize nothing. Where
``retain_memory`` holds fewer bases than frequencies, a returning base gives up
its own factors first, so a fixed set of frequencies keeps theirs. A
differentiated forward's last base moves to its entry and returns to
the pool once the entry no longer holds it: when the consumed entry is freed
(ADR 0081), or when the entry gives it up under ``retain_memory`` (ADR 0093),
after which its backward takes a plan back from the pool. Any other returns
when the sweep ends. The transposed system's base returns the same way.

Not thread-safe: one solve at a time per process (`gfem._engine.workers`). A
handle's own lock only orders a `free` against a running solve.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Self

import ngsolve as ngs
import numpy as np
import scipy.linalg

from gfem import _timing
from gfem._engine.cache import ENTRIES
from gfem._engine.wave_ports import PortSurfaceCoupling
from gfem._linalg import (
    Handle,
    Operator,
    ResolvedLinearSolver,
    low_rank_update,
    refactorize_or_prepare,
    report,
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
    from gfem._engine.floquet_ports import FloquetSurfaceCoupling
    from gfem._engine.terms import Piece

__all__ = ["FixedPieces", "LowRankTerm", "PlanSlot", "SystemOperator", "SystemSweep"]


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
        material-field regions and of those regions' layout (which elements
        lie outside); every other piece itself and ``None``.
        """
        varying = {p.name: p for p in self.varying() if p.values is not None}
        layout = np.packbits(np.asarray(self.outside, bool)).tobytes()
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
            digest.update(layout)
            digest.update(values.tobytes())
            integrands.append(part.integrand)
            fixed.append(
                FixedPart(self.store, piece.name, digest.hexdigest(), piece.integrand)
            )
        return integrands, fixed


@dataclasses.dataclass(frozen=True, eq=False)
class LowRankTerm:
    """A low-rank correction ``U C Vᵀ`` and how its Woodbury solve treats it.

    Attributes:
        left: ``U``, ``(ndof, p)``, zero on the fixed DOFs.
        coefficients: ``C``, ``(p, p)``.
        right: ``V``, as ``U``; ``None`` for ``U``.
        infinite: ``(p,)`` bools: the rows whose diagonal shift is infinite
            (their row and column of ``C`` zero); ``None`` for none.
        row_scaled: Whether the capacitance is row-scaled.
    """

    left: np.ndarray
    coefficients: np.ndarray
    right: np.ndarray | None = None
    infinite: np.ndarray | None = None
    row_scaled: bool = False

    def transposed(self) -> LowRankTerm:
        """``(U C Vᵀ)ᵀ = V Cᵀ Uᵀ``, with the same flags."""
        if self.right is None:
            return dataclasses.replace(self, coefficients=self.coefficients.T)
        return dataclasses.replace(
            self, left=self.right, coefficients=self.coefficients.T, right=self.left
        )

    @classmethod
    def concatenate(cls, terms: Sequence[LowRankTerm]) -> LowRankTerm:
        """The block-diagonal sum of ``terms``: columns side by side, ``C`` blocks.

        ``right`` is ``None`` when every term's is; ``infinite`` likewise;
        the capacitance is row-scaled when any term's is.
        """
        left = np.concatenate([t.left for t in terms], axis=1)
        coefficients = scipy.linalg.block_diag(*(t.coefficients for t in terms))
        right = None
        if any(t.right is not None for t in terms):
            right = np.concatenate(
                [t.left if t.right is None else t.right for t in terms], axis=1
            )
        infinite = None
        if any(t.infinite is not None for t in terms):
            infinite = np.concatenate(
                [
                    np.zeros(t.left.shape[1], bool)
                    if t.infinite is None
                    else t.infinite
                    for t in terms
                ]
            )
        return cls(
            left,
            coefficients,
            right=right,
            infinite=infinite,
            row_scaled=any(t.row_scaled for t in terms),
        )


def _term(coupling: Any, omega: float) -> LowRankTerm:
    """A surface coupling's correction at ω.

    Its own `LowRankTerm` (``term``), else a wave port's ``W = vectors P D
    Pᵀ vectorsᵀ`` (`PortSurfaceCoupling.correction`).
    """
    term = getattr(coupling, "term", None)
    if term is not None:
        return term(omega)
    p, d = coupling.correction(omega)
    return LowRankTerm(coupling.vectors @ p, d)


def _transposed(coupling: Any) -> Any:
    """A coupling for the transposed system: its ``transposed()``, else itself."""
    transposed = getattr(coupling, "transposed", None)
    return coupling if transposed is None else transposed()


class SystemOperator:
    """The system operator ``A(ω) + W(ω)`` of one driven solve, and every handle of it.

    Owns the wave ports' Robin pieces, the piece sum, the modal corrections
    ``W``, the adjoint handles (one refactorized base, else one per
    frequency on the CPU), the transposed system under a Bloch wave vector,
    and their memory. A forward sweep's base handle lives in its
    `SystemSweep`.

    Attributes:
        operator: The driven operator, port surfaces' Robin pieces included.
        surfaces: The port surfaces' couplings: each has a Robin ``piece``,
            its ``vectors`` and either a wave port's ``correction(ω)`` or its
            own ``term(ω)`` (a `LowRankTerm`), and may give a
            ``transposed()`` coupling for the transposed system.
    """

    def __init__(
        self,
        operator: DrivenOperator,
        surfaces: Sequence[PortSurfaceCoupling | FloquetSurfaceCoupling],
        linear: ResolvedLinearSolver,
        *,
        width: int,
        adjoint_tolerance: float,
        transposed: Callable[[], DrivenOperator] | None = None,
        pool_budget: int | None = None,
    ) -> None:
        """Add each port surface's Robin piece to ``operator``.

        Args:
            operator: The driven operator.
            surfaces: The port surfaces' couplings (see ``surfaces``).
            linear: The rule's linear solver.
            width: The right-hand sides per solve (the excited ports).
            adjoint_tolerance: The relative residual of adjoint solves.
            transposed: Builds the driven operator of ``operator``'s
                transpose (the operator at -k under a Bloch wave vector), for
                the adjoint solves; ``None`` when ``operator`` is complex
                symmetric.
            pool_budget: ``retain_memory``'s bytes: with it the transposed
                system's adjoint solves refactorize one base from the plan
                pool where they can (`_adjoint_system`).
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
        self._pool_budget = pool_budget

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
        plans = None if pool_budget is None else self._plan_slot(pool_budget)
        return self._sweep(
            solves=solves,
            reuse_preconditioner=reuse_preconditioner,
            tolerance=tolerance,
            plans=plans,
        )

    def _sweep(
        self,
        *,
        solves: int,
        reuse_preconditioner: bool,
        tolerance: float | None,
        plans: PlanSlot | None,
    ) -> SystemSweep:
        """`sweep` on the plan slot ``plans`` (or off the pool)."""
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
            plans=plans,
            tolerance=self._linear.tolerance if tolerance is None else tolerance,
        )

    def free_digest(self) -> str:
        """A digest of the operator's free DOFs: a key's, where they may change."""
        op = self.operator
        free = free_mask(op.space.FreeDofs(), op.space.ndof)
        return hashlib.blake2b(np.packbits(free).tobytes(), digest_size=16).hexdigest()

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
        site = (id(op.topology), op.space_key, self.free_digest(), op.symmetry)
        store = None if op.fixed is None else op.fixed.store
        return PlanSlot(site, op.topology, budget, store)

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
        x, its = handle.solve_counted(loads, tolerance=self._adjoint_tolerance)
        report.solve_step(
            getattr(handle, "base", handle),
            problem="driven",
            direction="backward",
            iterations=its,
            frequency=omega / (2 * math.pi),
        )
        return x

    def adjoint_order(self, omegas: Sequence[float]) -> list[int]:
        """The frequency indices of ``omegas`` in the order a backward solves them.

        From the adjoint base's frequency: ascending when the base sits at the
        first (up to round-off), else descending (a forward leaves it at the
        last). Repeated backward passes so alternate, and each refactorizes
        once fewer than there are frequencies, where one base serves them all
        (off the plan pool). Under a Bloch wave vector the adjoint base is the
        transposed system's.
        """
        system = self if self._transposed is None else self._transposed_system
        sweep = None if system is None else system._adjoint_sweep  # noqa: SLF001
        at = None if sweep is None else sweep.frequency
        indices = list(range(len(omegas)))
        first = (
            bool(omegas)
            and at is not None
            and bool(np.isclose(at, omegas[0], rtol=1e-12, atol=0.0))
        )
        return indices if first else indices[::-1]

    @property
    def nbytes(self) -> int:
        """Host memory held by the adjoint handles and the pieces, in bytes.

        The transposed system's handles and pieces included. What others may
        hold too (``W``'s vectors, kept fixed parts, patterns) is in `shared`.
        """
        held = self._held_nbytes()
        if self._transposed_system is not None:
            held += self._transposed_system._held_nbytes()  # noqa: SLF001
        return held

    def shared(self) -> dict[int, int]:
        """Bytes of ``W``'s vectors and of the handles' and pieces' shared objects.

        By object identity, so that the entry cache counts each once.
        """
        held = {id(c.vectors): int(c.vectors.nbytes) for c in self.surfaces}
        systems = [self]
        if self._transposed_system is not None:
            systems.append(self._transposed_system)
        for system in systems:
            handles = list(system._handles.values())  # noqa: SLF001
            sweep = system._adjoint_sweep  # noqa: SLF001
            if sweep is not None and sweep._base is not None:  # noqa: SLF001
                handles.append(sweep._base)  # noqa: SLF001
            for handle in handles:
                shared = getattr(handle, "shared", None)
                if shared is not None:
                    held.update(shared())
            held.update(system._pieces.shared())  # noqa: SLF001
        return held

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
            self._transposed_system.free(return_plans=return_plans)
        self._transposed_system = None

    def keep_base(self, sweep: SystemSweep) -> None:
        """Move a finished forward sweep's base handle to the adjoint solves.

        Under ranks, where one factorization per frequency does not fit and a
        backward must not prepare what its forward just freed (ADR 0065), and on
        one process with the CPU ``direct`` and ``bddc_gmres`` solvers when the
        operator is its own adjoint's, so that the backward refactorizes it rather
        than preparing a handle per frequency (on the plan pool, takes each
        frequency's base in turn: `SystemSweep.at`). A base it no longer holds is
        prepared again at the adjoint tolerance, taking a plan from the pool where
        it can. A shared base from the plan pool returns to the pool once no
        entry holds it (ADRs 0082, 0093). A
        base the adjoint solves do not share (the transposed system's solves,
        which hold a base of their own) stays with its sweep: ``with`` returns it
        to the pool or frees it.
        """
        if self.operator.built.comm is None and not self._shares_base():
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

    def free_base(self, *, return_plan: bool = False, walk: bool = True) -> None:
        """Release the adjoint solves' base handle; the sweep prepares it again.

        The transposed system's too. With ``return_plan`` a base from the plan
        pool returns there instead (`SystemSweep.free`).
        """
        if self._adjoint_sweep is not None:
            self._adjoint_sweep.free(return_plan=return_plan, walk=walk)
        if self._transposed_system is not None:
            self._transposed_system.free_base(return_plan=return_plan, walk=walk)

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
        system, built on first use with the same solver and tolerances and
        each surface coupling's transposed form where it gives one (a wave
        port's coupling serves both). With a pool budget, where the plan pool
        serves (one process, CPU ``direct``), its adjoint solves run one sweep
        whose base comes from the pool under the -k key and returns there with
        the entry.
        """
        if self._transposed is None:
            return self
        if self._transposed_system is None:
            system = SystemOperator(
                self._transposed(),
                [_transposed(c) for c in self.surfaces],
                self._linear,
                width=self._width,
                adjoint_tolerance=self._adjoint_tolerance,
            )
            budget = self._pool_budget
            slot = None if budget is None else system._plan_slot(budget)
            if slot is not None:
                system._adjoint_sweep = system._sweep(
                    solves=1,
                    reuse_preconditioner=True,
                    tolerance=self._adjoint_tolerance,
                    plans=slot,
                )
            self._transposed_system = system
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

    def corrections(self, omega: float) -> LowRankTerm:
        """``W(ω) = U C Vᵀ``, block diagonal over the surface couplings.

        A wave port surface gives two columns per wave port on it (see
        `PortSurfaceCoupling.correction`), with ``V = U``; a coupling with its
        own ``term`` gives that.
        """
        return LowRankTerm.concatenate([_term(c, omega) for c in self.surfaces])

    def _system_handle(self, base: Handle, omega: float, *, borrow: bool) -> Handle:
        """A handle of ``A(ω) + W(ω)`` from a handle ``base`` of ``A(ω)``.

        ``base`` itself without port surfaces; otherwise a Woodbury update that
        owns ``base`` or ``borrow``s it.
        """
        if not self.surfaces:
            return base
        term = self.corrections(omega)
        return low_rank_update(
            base,
            term.left,
            term.coefficients,
            right=term.right,
            infinite=term.infinite,
            row_scaled=term.row_scaled,
            borrow=borrow,
        )

    def _adjoint_handle(self, n: int, omega: float) -> Handle:
        """The adjoint handle of ``A(ω) + W(ω)`` of frequency ``n``.

        On a GPU the adjoint base refactorized at ``ω``, valid until the next
        call. Under ranks, and on one process once the forward's base was kept
        (`keep_base`), the same, except that a base already at ``ω`` serves as
        it is: a backward starting from the base's frequency factorizes once
        fewer; on the plan pool ``ω``'s own base, which refactorizes nothing.
        Otherwise on the CPU prepared once per ``n`` and kept.
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
    """One base handle of ``A(ω)`` at a time for a sweep: one solver plan.

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

    With a `PlanSlot` the base is a frequency base (ADR 0121): each new ω
    returns the held base to the plan pool under the ω it was at and takes
    ω's (`gfem._engine.cache.EntryCache.take_plan`), or prepares one; leaving
    the ``with`` block without an error returns the base to the pool. A taken
    base solves at the sweep's ``tolerance``, not at the one of the sweep that
    prepared it (an adjoint's): a solve on factors reused across designs reads
    it.
    """

    def __init__(
        self,
        operator_at: Callable[[float], Operator],
        system_handle: Callable[[Handle, float], Handle],
        prepare: Callable[[Operator], Handle],
        plans: PlanSlot | None = None,
        tolerance: float | None = None,
    ) -> None:
        """Hold ``ω -> A(ω)``, the wrap into ``A(ω) + W(ω)`` and the base's prepare.

        The wrap borrows the base handle. ``plans``: where the base pools.
        ``tolerance``: the one ``prepare`` prepares at, given to a base taken
        from the pool.
        """
        self._operator_at = operator_at
        self._system_handle = system_handle
        self._prepare = prepare
        self._plans = plans
        self._tolerance = tolerance
        # the pool key of the base, set when the base was taken or prepared
        self._key: tuple | None = None
        self._base: Handle | None = None
        # the system handle `at` gave last, and its frequency
        self._last: tuple[float, Handle] | None = None

    @property
    def pooled(self) -> bool:
        """Whether the base returns to the plan pool."""
        return self._plans is not None

    @property
    def base(self) -> Handle | None:
        """The base handle held, or ``None``: what a step's record reads."""
        return self._base

    @property
    def frequency(self) -> float | None:
        """The ω of the handle `at` gave last, or ``None``."""
        return None if self._last is None else self._last[0]

    def at(self, omega: float) -> Handle:
        """A handle of ``A(ω) + W(ω)``, borrowing the sweep's base handle.

        A pooled sweep holds a base per frequency (ADR 0121): at another ω the
        base returns to the pool under its last frequency, and ω's is taken
        (`gfem._engine.cache.EntryCache.take_plan`), or the same base again
        where the site meets ω for the first time. It refactorizes as a
        frequency ``sweep`` (`gfem._linalg.Refactorable.refactorize`).
        """
        with _timing.phase("setup", frequency=omega / (2 * math.pi)) as span:
            operator = self._operator_at(omega)
            plans = self._plans
            if plans is not None and (
                self._base is None
                or (self._last is not None and self._last[0] != omega)
            ):
                self._take_base(plans, omega)
            base = self._base = refactorize_or_prepare(
                self._base, operator, self._prepare, frequency=omega, sweep=True
            )
            handle = self._system_handle(base, omega)
            if span.enabled:
                span.set(
                    solver=getattr(base, "kind", None),
                    action=getattr(base, "last_action", None) or "factorized",
                    nbytes=int(base.nbytes),
                )
        self._last = (omega, handle)
        return handle

    def _take_base(self, plans: PlanSlot, omega: float) -> None:
        """Hold ω's pooled base, returning the one held; ``None`` if none serves.

        The key keeps the thread count of the sweep's first take: its bases
        were planned on it, whatever the caller's count now.
        """
        base, key = self._base, self._key
        if base is None or key is None:
            self._key = plans.key(omega)
            self._base = ENTRIES.take_plan(
                self._key, budget=plans.budget, visit=plans.visit
            )
        else:
            self._key = (*key[:-1], float(omega))
            self._base = ENTRIES.switch_plan(
                self._returned_key(),
                plans.topology,
                base,
                plans.budget,
                kept=plans.kept(),
                to=self._key,
                visit=plans.visit,
            )
        if self._base is not None and self._tolerance is not None:
            self._base.tolerance = self._tolerance

    def _returned_key(self) -> tuple | None:
        """The pool key of the held base: its last frequency's."""
        key, last = self._key, self._last
        if key is None or last is None:
            return key
        return (*key[:-1], float(last[0]))

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
        self,
        omega: float,
        rhs: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        guess: str | None = None,
        reference: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Solve at ``ω``: the solution and the Krylov iterations per column.

        Logs the step (`gfem._linalg.report`), as the base handle tells it;
        ``guess`` names where ``x0`` came from. A ``reference`` iteration count
        stands in for the one a fresh BDDC base would measure on this solve
        (`gfem._linalg.bddc.BddcHandle.assume_reference`).
        """
        handle = self.at(omega)
        assume = getattr(self._base, "assume_reference", None)
        if reference is not None and assume is not None:
            assume(reference)
        frequency = omega / (2 * math.pi)
        with _timing.phase("solve", frequency=frequency) as span:
            x, its = handle.solve_counted(rhs, x0=x0)
            span.set(iterations=tuple(int(i) for i in its))
        report.solve_step(
            self._base,
            problem="driven",
            direction="forward",
            iterations=its,
            frequency=frequency,
            guess=guess,
        )
        return x, its

    @property
    def reference_iterations(self) -> int | None:
        """The base's reference iteration count, if it keeps one (BDDC's ``n_ref``)."""
        return getattr(self._base, "reference_iterations", None)

    @property
    def reused(self) -> bool:
        """Whether the last solve ran on a preconditioner built for another matrix.

        As the base handle's ``last_action`` tells it (``reused``, or
        ``reused, capped, refactorized``: it started on one); ``False`` for a
        backend that tracks none.
        """
        action = getattr(self._base, "last_action", None)
        return action is not None and action.startswith("reused")

    def free(self, *, return_plan: bool = False, walk: bool = True) -> None:
        """Release the base handle.

        With ``return_plan`` a pooled sweep's base returns to the plan pool
        instead, with its factors (the pool releases them under pressure,
        where it may ``walk``: `EntryCache.give_plan`).
        """
        base, plans, key = self._base, self._plans, self._returned_key()
        self._base, self._last = None, None
        if base is None:
            return
        if return_plan and plans is not None and key is not None:
            ENTRIES.give_plan(
                key,
                plans.topology,
                base,
                plans.budget,
                plans.kept(),
                walk=walk,
                visit=plans.visit,
            )
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
        budget: ``retain_memory``'s bytes, the walk's budget on return (a
            `gfem._engine.cache.DefaultBudget` under the default).
        store: The kept fixed parts of the operator's space, or ``None``
            without them (`FixedPieces`).
        visit: This design's sweeps on the slot: a forward's, and the backward
            it hands its base to (`SystemSweep.move_base`). The pool counts a
            frequency they return a base under as met only for later visits
            (`gfem._engine.cache.EntryCache.take_plan`).
    """

    site: tuple
    topology: Any
    budget: int
    store: KeptParts | None = None
    visit: object = dataclasses.field(default_factory=object)

    def kept(self) -> list[Any]:
        """The patterns of the kept fixed parts: what a pooled base may keep."""
        return (
            [] if self.store is None else [k.pattern for k in self.store.kept.values()]
        )

    def key(self, omega: float) -> tuple:
        """The pool key of a base at ``ω``: ``site``, the thread count, ``ω``.

        The thread count of a plan made now: a PARDISO handle runs on the
        thread count of its analysis (ADR 0032), so another count plans anew.
        A sweep holds one base per frequency it met before (ADR 0121).
        """
        return (*self.site, max(1, ngs.GetNumThreads()), float(omega))
