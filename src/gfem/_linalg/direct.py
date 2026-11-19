"""The ``direct`` backend: an LDL^T factorization on the CPU, LU when A^T != A.

With the ``gfem[mkl]`` extra installed, NGSolve's MKL PARDISO factors with a
METIS nested-dissection ordering and the symmetric matrix type; it fills
about a third less than NGSolve's sparse Cholesky (minimum degree) and
factors several times faster. Without it, NGSolve's sparse Cholesky. Both
give the same answers to the direct solve's accuracy.

A general operator (a Bloch phase, A^T != A) or a complex Hermitian one is
factored by LU: PARDISO's unsymmetric matrix type with the ``mkl`` extra,
NGSolve's UMFPACK (built into the NGSolve wheels) without it. Sparse
Cholesky is never given one: it factors A = L D L^T whatever the matrix,
silently wrong when A^T != A. The same LU factors solve the transposed
system A^T X = B (``solve(..., transpose=True)``).

Installing ``mkl`` switches NGSolve's process-wide default inverse to PARDISO
(an LU with minimum degree), so every inverse gfem creates names its type.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Protocol, cast

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._linalg import Symmetry
from gfem._linalg.base import (
    PlannedHandle,
    equals_its_transpose,
    host_bytes,
)
from gfem._linalg.factor_reuse import (
    CHOLESKY_FLOOR_NONZEROS,
    FIRST,
    GATE,
    RELEASED,
    REUSED_TOLERANCE,
    SAME_DESIGN,
    AmortizedRule,
    cholesky_ratio,
)
from gfem._linalg.gmres import Gmres, StopNorm
from gfem._linalg.pardiso import (  # noqa: F401 (the first two re-exported)
    PARDISO_FLAGS,
    PardisoFactors,
    pardiso_available,
    pardiso_inverse,
)

#: NGSolve's sparse Cholesky: the factorization without MKL, also the BDDC
#: coarse factorization's.
INVERSE = "sparsecholesky"
#: NGSolve's UMFPACK (LU): the factorization of a general operator without
#: MKL, also the BDDC coarse factorization's then.
GENERAL_INVERSE = "umfpack"


@functools.cache
def _real_sparse_matrix() -> Any:
    """NGSolve's ``SparseMatrix<double>`` class.

    NGSolve names it after the C++ ``typeid``: ``SparseMatrixd`` with GCC and
    Clang, another name with MSVC (Windows), so it is found by what its
    ``CreateFromCOO`` returns.
    """
    for cls in vars(ngs.la).values():
        if not (isinstance(cls, type) and "CreateFromCOO" in vars(cls)):
            continue
        signature = cls.CreateFromCOO.__doc__ or ""
        if signature.rstrip().endswith("SparseMatrixTM<double>"):
            return cls
    raise RuntimeError("ngsolve.la has no real SparseMatrix with CreateFromCOO")


class Factors(Protocol):
    """A direct handle's inverse, and how it factorizes new values on its plan."""

    #: The inverse a solve applies.
    inverse: Any
    #: The factors were released (`release`): no solve until `new_values`.
    released: bool

    def release(self) -> None:
        """Drop the factors and keep the plan, where the backend can."""
        ...

    def new_values(
        self, matrix: ngs.BaseMatrix, write: Callable[[ngs.BaseMatrix], None]
    ) -> None:
        """``write(matrix)``, then factorize its new values on the plan."""
        ...

    @property
    def nbytes(self) -> int:
        """Host memory of the factors and whatever else this holds, in bytes."""
        ...

    @property
    def factor_bytes(self) -> int:
        """The bytes of `nbytes` that `release` frees."""
        ...

    def model_ratio(self) -> float:
        """One refactorization over one direct column, from the plan's counts."""
        ...

    def above_floor(self) -> bool:
        """Whether the plan is large enough for the factor-reuse rule."""
        ...


class UpdateFactors:
    """An inverse NGSolve refactorizes itself (``Update()``): Cholesky, UMFPACK."""

    def __init__(self, inverse: ngs.BaseMatrix, free: int) -> None:
        """Hold ``inverse``, factored on ``free`` DOFs."""
        self.inverse = inverse
        #: Never set: these factors cannot be released without their ordering.
        self.released = False
        self._free = free

    @functools.cached_property
    def _nonzeros(self) -> float:
        """Sparse Cholesky's factor nonzeros (``nze``; UMFPACK raises on it)."""
        return float(self.inverse.nze)

    def model_ratio(self) -> float:
        """Sparse Cholesky's model ratio (`cholesky_ratio`), no thread term."""
        return cholesky_ratio(self._nonzeros, self._free)

    def above_floor(self) -> bool:
        """Whether the factors hold at least `CHOLESKY_FLOOR_NONZEROS` nonzeros."""
        return self._nonzeros >= CHOLESKY_FLOOR_NONZEROS

    def release(self) -> None:
        """Keep everything: NGSolve frees no factors and keeps the ordering."""

    def new_values(
        self, matrix: ngs.BaseMatrix, write: Callable[[ngs.BaseMatrix], None]
    ) -> None:
        """``write(matrix)``, then ``Update()`` the inverse."""
        write(matrix)
        self.inverse.Update()

    @property
    def nbytes(self) -> int:
        """Host memory NGSolve reports for the factors, in bytes."""
        return host_bytes(self.inverse)

    @property
    def factor_bytes(self) -> int:
        """0: `release` frees nothing."""
        return 0


class DirectHandle(PlannedHandle):
    """One factorization, reused for every right-hand side: LDL^T, or LU.

    MKL PARDISO when the ``mkl`` extra is installed, NGSolve's sparse
    Cholesky otherwise (see the module docstring), factor ``A = L D L^T``
    when ``A^T = A``: a real symmetric operator (``"hermitian"`` on a real
    space) or a complex symmetric one. A general or a complex Hermitian
    operator is factored by LU: PARDISO's unsymmetric type, or UMFPACK
    without MKL (`GENERAL_INVERSE`); a transposed solve of a general operator
    runs on the same factors.

    An operator with a piece sum (``Operator.piece_sum``) is summed, never
    assembled: into a new matrix at creation, into the handle's own matrix at
    `refactorize`.

    `refactorize` factorizes another operator's values in place: a
    `gfem._linalg.Refactorable` handle. PARDISO gathers them straight into
    its wrapper's buffer and runs its numeric factorization (phase 22);
    sparse Cholesky, or a wrapper without the private attributes PARDISO's
    `Factors` adapter (`gfem._linalg.pardiso.PardisoFactors`) reads, runs
    NGSolve's ``Update()``. The gather index is built at the first
    refactorization, so a handle factored once never builds it.

    A PARDISO handle refactorizes and solves on the thread count it was
    factored with, whatever TaskManager the caller is in: a solve outside the
    TaskManager it was factored in stays parallel, and one factored outside
    any TaskManager stays on one thread (`gfem._linalg.pardiso`).

    `release_factors` drops PARDISO's factors and keeps its plan, for a
    handle held between solves (a plan pool); it refuses to solve or apply
    until `refactorize` factorizes new values. Sparse Cholesky and UMFPACK
    keep their factors.

    A handle (PARDISO or sparse Cholesky) of an operator equal to its
    transpose may solve a refactorization on the factors it holds instead
    (preconditioner reuse across designs, ADR 0094): `refactorize` at the
    ``frequency`` its factors were made at writes the new values into the
    held matrix and leaves the factors, and its solves run `Gmres` on the
    held matrix, left-preconditioned by them, to `REUSED_TOLERANCE` times the
    tolerance. The handle's `AmortizedRule` decides each refactorization from
    counts, never a clock: the plan's model ratio and floor (`Factors`, read
    once per plan) and the iterations of reused columns; its gate is
    `_reuse_pays`. Sparse Cholesky frees nothing at `release_factors`, which
    still ends the rule's chain. A column past the rule's cap, or one that
    breaks down above its tolerance, refactorizes at the held matrix and
    finishes that solve direct: a reused solve never raises. Reused columns
    report their iterations (`solve_counted`); direct ones none.

    `last_action` and `last_reason` say what the last step did and why
    (`gfem._linalg.report`): ``factorized`` (a new plan, ``first``),
    ``refactorized`` (the rule's `AmortizedRule.reason`; ``None`` on a handle
    without the rule), ``reused``, ``reused, capped, refactorized``
    (``capped``), or ``held`` (``same design``: the values it holds, nothing
    refactorized); `last_details` the reason's numbers. Under the rule a
    refactorization on the held values is no step and changes none of them.
    """

    kind = "direct"
    sums_pieces = True
    transposes_general = True

    def __init__(
        self,
        matrix: ngs.BaseMatrix,
        freedofs: ngs.BitArray | None,
        symmetry: Symmetry,
        *,
        is_complex: bool,
        tolerance: float = 1e-6,
    ) -> None:
        """Factorize an assembled matrix (see `prepare` and `factorize`).

        ``symmetry`` is refused before (`from_operator`; `from_scipy`'s real
        symmetric matrix always passes). ``tolerance`` is the prepared
        tolerance, read only by solves on reused factors.
        """
        super().__init__(matrix, freedofs, symmetry, is_complex=is_complex)
        self.tolerance = tolerance
        factors = self._factor(matrix, freedofs)
        self._factors: Factors | None = factors
        self._solver = factors.inverse
        #: The reuse rule: a handle with ``A^T = A`` only.
        self._rule: AmortizedRule | None = None
        if equals_its_transpose(symmetry, is_complex=is_complex):
            self._rule = AmortizedRule(
                factors.model_ratio(), above_floor=factors.above_floor()
            )
        # the frequency of the refactorization under way, if told
        self._frequency: float | None = None
        # the held values' stop norm, for reused-factor GMRES (`_hold_stop_norm`)
        self._stop: StopNorm | None = None
        #: What the last step did, and why (see the class docstring).
        self.last_action: str | None = "factorized"
        self.last_reason: str | None = FIRST
        #: The numbers of `last_reason`, by name.
        self.last_details: dict[str, float] = {}
        # bytes and budget of a release for retain_memory (`note_release`)
        self._release_note: dict[str, float] = {}

    def _factor(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> Factors:
        free = matrix.height if freedofs is None else freedofs.NumSet()
        # gfem's registered PARDISO type, which applies `PARDISO_FLAGS` and
        # runs its calls (phases 12, 22 and 33) at OpenMP block time 0; never
        # on an empty matrix, which PARDISO refuses
        pardiso = (
            pardiso_inverse(self.symmetry, is_complex=self._complex)
            if free > 0
            else None
        )
        if pardiso is not None:
            inverse = pardiso
        elif equals_its_transpose(self.symmetry, is_complex=self._complex):
            inverse = INVERSE
        else:
            inverse = GENERAL_INVERSE
        if freedofs is None or free == matrix.height:
            # every DOF free: NGSolve factors the matrix itself instead of
            # projecting it on the free DOFs first
            solver = matrix.Inverse(inverse=inverse)
        else:
            solver = matrix.Inverse(freedofs, inverse=inverse)
        if pardiso is None:
            return UpdateFactors(solver, free)
        return PardisoFactors(solver, self._free, is_complex=self._complex)

    @classmethod
    def from_scipy(cls, matrix: sp.sparray | sp.spmatrix) -> DirectHandle:
        """Factorize a real symmetric SciPy matrix (see `gfem._linalg.factorize`)."""
        coo = sp.coo_array(matrix)
        n, m = coo.shape
        if n != m:
            raise ValueError(f"factorize needs a square matrix, got shape {coo.shape}.")
        assembled = _real_sparse_matrix().CreateFromCOO(
            coo.row.astype(np.int64),
            coo.col.astype(np.int64),
            np.asarray(coo.data, dtype=float),
            n,
            n,
        )
        return cls(assembled, None, "hermitian", is_complex=False)

    def release_factors(self) -> None:
        """Drop the factors and keep the plan until the next `refactorize`.

        PARDISO releases L and U and keeps its ordering and symbolic
        analysis; `solve`, `solve_counted` and `apply` then raise until
        `refactorize` factorizes new values. Sparse Cholesky and UMFPACK keep
        their factors. The pieces' pattern goes too, so a held handle does not
        keep its last operator's pieces alive: the next `refactorize` assembles
        (sums) into a new matrix and copies its values in. Waits for a running
        solve or refactorization.

        Raises:
            RuntimeError: The handle was freed.
        """
        with self._lock:
            self._live()
            cast("Factors", self._factors).release()
            self._pattern = None
            if self._rule is not None:
                self._rule.end_chain(released=True)

    @property
    def released(self) -> bool:
        """Whether its factors were released and not factorized since."""
        factors = self._factors
        return factors is not None and factors.released

    def note_release(self, *, base_bytes: int, budget_bytes: int) -> None:
        """A cache walk releases this handle's factors for ``retain_memory``.

        The next refactorized step's record carries the two numbers.
        """
        self._release_note = {"base_bytes": base_bytes, "budget_bytes": budget_bytes}

    def _check_solvable(self) -> None:
        """Refuse a handle whose factors were released; under the lock."""
        if self._factors is not None and self._factors.released:
            raise RuntimeError(
                "This direct handle's factors were released: refactorize it "
                "before solving."
            )

    def _refactorizing_at(self, frequency: float | None) -> None:
        """Hold the frequency of the refactorization under way; under the lock."""
        self._frequency = frequency

    def _hold_stop_norm(self, stop: tuple[Any, np.ndarray] | None) -> None:
        r"""Hold the held values' stop norm ``(T, d)``, if any; under the lock.

        Reused-factor GMRES then stops on :math:`\|T (d \circ r)\|` of its
        preconditioned residual :math:`r` on the free DOFs (ADR 0094, 0216);
        without one, on :math:`\|r\|`.
        """
        if stop is None:
            self._stop = None
            return
        T, d = stop
        rows = np.flatnonzero(self._free)
        if T.shape[1] != len(rows) or len(d) != len(rows):
            raise ValueError(
                f"A stop norm of {T.shape[1]} columns for {len(rows)} free DOFs."
            )
        self._stop = StopNorm(T, np.asarray(d), rows)

    def _reuse_pays(self) -> bool:
        """The rule's gate (`AmortizedRule.pays`); tests force it."""
        return self._rule is not None and self._rule.pays()

    def _new_values(
        self, write: Callable[[ngs.BaseMatrix], None], same: Callable[[], bool]
    ) -> None:
        """Write the held matrix and factorize it on the plan; under the lock.

        Or only write it, when the rule reuses the held factors: PARDISO's
        own copy of the values stays the factored matrix's. Values equal to
        the held ones (``same()``: the same design again, a backward taking
        its plan back) are no new step and are not written: the step in
        progress goes on, on the factors it solves with, and the step reads
        ``held`` (``same design``): it refactorized nothing.
        """
        matrix = self._live()
        # a live handle holds its factors (`_release` drops them)
        factors = cast("Factors", self._factors)
        rule = self._rule
        if rule is not None and not factors.released and same():
            self._stepped("held", SAME_DESIGN)
            return
        if rule is not None and rule.next_step(
            frequency=self._frequency,
            held=not factors.released,
            pays=self._reuse_pays,
        ):
            write(matrix)
            self._stepped("reused", None)
            return
        factors.new_values(matrix, write)
        if rule is not None:
            rule.refactorized(self._frequency)
        self._stepped("refactorized", None if rule is None else rule.reason)

    def _stepped(self, action: str, reason: str | None) -> None:
        """Hold the step's action, reason and its numbers; under the lock."""
        details: dict[str, float] = {}
        rule = self._rule
        if reason == GATE and rule is not None and rule.ratio is not None:
            details = {"ratio": rule.ratio, "first_iterations": rule.first_iterations}
        elif reason == RELEASED:
            details = self._release_note
        self._release_note = {}
        self.last_action, self.last_reason, self.last_details = action, reason, details

    def _new_plan(self, matrix: ngs.BaseMatrix, freedofs: ngs.BitArray | None) -> None:
        """Factorize ``matrix`` anew and hold it; under the lock.

        Factors before it holds anything: a failure leaves the handle as it was.
        """
        factors = self._factor(matrix, freedofs)
        self._matrix, self._solver, self._factors = matrix, factors.inverse, factors
        if self._rule is not None:
            self._rule.end_chain()
            self._rule.model_ratio = factors.model_ratio()
            self._rule.above_floor = factors.above_floor()
        self._stepped("factorized", FIRST)

    @classmethod
    def _admit(cls, symmetry: Symmetry, *, is_complex: bool) -> None:
        """Every symmetry: L D L^T when A^T = A, LU otherwise (`_factor`)."""

    @property
    def _inverse(self):
        return self._solver

    def _inverse_map(
        self,
        B: np.ndarray,
        tolerance: float | None,
        x0: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """The direct solve, or GMRES on reused factors (see the class docstring).

        ``x0`` is ignored. Returns each column's iterations on a reused
        step, ``()`` on a direct one.
        """
        rule = self._rule
        # the rule's counters are read and written with the solve they count
        with self._lock:
            if rule is not None and rule.reusing:
                return self._reused_map(rule, B, tolerance)
            X = self._map(lambda _: self._solver, B)
            if rule is not None:
                rule.solved_direct(1 if B.ndim == 1 else B.shape[1])
            return X, ()

    def _reused_map(
        self, rule: AmortizedRule, B: np.ndarray, tolerance: float | None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Each column by `Gmres` on the held matrix, preconditioned by the factors.

        Each stops at `REUSED_TOLERANCE` times ``tolerance`` (``None``: the
        prepared one) relative to its preconditioned load, within the rule's
        cap, both measured in the held values' stop norm when they have one
        (`_hold_stop_norm`). A column that does not refactorizes at the held
        matrix, and it and the later ones are solved direct (counted 0
        iterations after it).
        """
        tight = REUSED_TOLERANCE * (self.tolerance if tolerance is None else tolerance)
        cols, flat = self._columns(B)
        out = np.zeros(cols.shape, dtype=np.complex128 if self._complex else float)
        counts: list[int] = []
        with self._lock:
            matrix = self._live()
            b, x = matrix.CreateColVector(), matrix.CreateRowVector()
            for j in range(cols.shape[1]):
                b.FV().NumPy()[:] = cols[:, j]
                if not rule.reusing:
                    x.data = self._solver * b
                    rule.solved_direct(1)
                    counts.append(0)
                    out[:, j] = x.FV().NumPy()
                    continue
                solver = Gmres(
                    matrix,
                    self._solver,
                    tol=tight,
                    maxiter=rule.cap,
                    stop_norm=self._stop,
                )
                solver.Solve(rhs=b, sol=x, initialize=True)
                its = int(solver.iterations)  # ty: ignore[unresolved-attribute]
                counts.append(its)
                # in the stop norm's coordinates, when the values have one
                residuals = solver.residuals  # ty: ignore[unresolved-attribute]
                if not residuals or residuals[-1] <= tight * residuals[0]:
                    rule.solved_reused(1, its)
                else:
                    # capped: the column counts once, as the direct one below
                    self._refactorize_held(rule)
                    x.data = self._solver * b
                    rule.solved_direct(1)
                out[:, j] = x.FV().NumPy()
        return (out[:, 0] if flat else out), tuple(counts)

    def _refactorize_held(self, rule: AmortizedRule) -> None:
        """Factorize the held matrix's values (the new design); under the lock."""
        cast("Factors", self._factors).new_values(self._live(), lambda _: None)
        rule.refactorized(self._frequency)
        self._stepped("reused, capped, refactorized", rule.reason)

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the factors, in bytes.

        PARDISO's factors count the gather index once built (at the first
        `refactorize`). The pieces' pattern, which kept fixed parts and pieces
        may hold too, is in `shared`.
        """
        # read once, without the lock: a cache walk must not wait for a
        # factorization, and a concurrent free gives 0 or the old count
        matrix, factors = self._matrix, self._factors
        if factors is None or matrix is None:
            return 0
        return host_bytes(matrix) + factors.nbytes

    @property
    def factor_bytes(self) -> int:
        """The bytes of `nbytes` that `release_factors` frees (ADR 0123)."""
        factors = self._factors
        return 0 if factors is None else factors.factor_bytes

    def _release(self) -> None:
        """Drop the factors; under the lock."""
        self._factors = None
