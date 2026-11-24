"""A Krylov solve on a held preconditioner, kept across a sweep.

`KrylovHandle` is what the ``bddc_gmres`` and ``pmg_gmres`` handles
(`gfem._linalg.bddc`, `gfem._linalg.pmg`) do apart from building their
preconditioner: the Krylov method, the initial guess,
the preconditioner reuse across a sweep's frequencies and its iteration-count
rules. A subclass builds and holds one preconditioner (`KrylovHandle._set_up`).

The Krylov method follows the operator: conjugate gradients for a Hermitian
(real symmetric positive definite) operator, NGSolve's on one process,
otherwise gfem's GMRES (`gfem._linalg.gmres`).

Across a sweep the handle keeps its preconditioner (`KrylovHandle.refactorize`)
once a solve on it measured a reference iteration count, solves on the reused
preconditioner to a tolerance `REUSED_TOLERANCE` times tighter, caps every
warm solve there from that count, and sets it up again when GMRES iterations
grow past one setup's worth. A GMRES solve without an initial guess on a
preconditioner reused across a sweep's frequencies (an adjoint solve of a
multi-frequency backward) runs that warm path from a zero guess;
after a refactorization at no stated frequency (an eigen reshift) it sets
the preconditioner up at the new operator first. A handle prepared without
reuse sets it up at each `KrylovHandle.refactorize`.

`KrylovHandle.last_action` and `last_reason` say what the last step did and
why (`gfem._linalg.report`): ``factorized`` (set up: ``first``,
``rebuild rule``, or ``cold solve`` for a CG solve, or a GMRES solve after
an eigen reshift, on a reused preconditioner), ``reused``, or ``reused,
capped, refactorized`` (``capped``: a warm column set the preconditioner up
inside the solve).

Under ranks GMRES runs on owned entries (`gfem._linalg.gmres`), CG is gfem's
on agreed products (`gfem._linalg.cg`), and every decision above reads an
agreed value: the iterations, the residuals and the norms of
`gfem._linalg.layout`.
"""

from __future__ import annotations

import math
from typing import ClassVar, Protocol

import ngsolve as ngs
import numpy as np

from gfem import _timing
from gfem._linalg import Operator, Symmetry
from gfem._linalg.base import NgsHandle, host_bytes, local_matrix
from gfem._linalg.cg import Cg
from gfem._linalg.gmres import Gmres
from gfem.errors import LinearSolverConvergenceError

#: Krylov iterations a preconditioner setup is worth, summed over the right-hand
#: sides of a solve: on the driven references (order 2) one BDDC setup took as long as
#: about 30 GMRES iterations of one right-hand side.
SETUP_ITERATIONS = 30
#: Factor on the tolerance of a solve on a reused preconditioner. GMRES stops
#: at ``tolerance * |P b|``, and a preconditioner set up at another frequency
#: measures the error less well: on the driven references (r8_abc1, r4_abc1,
#: r8_wp; 1e-6) reused solves ended 10-20x less accurate in S than fresh ones
#: (up to 1.1e-5 against 3e-7 to 6e-7). One more decade gave fresh accuracy
#: for about 10 more iterations per column.
REUSED_TOLERANCE = 0.1


def _slack(columns: int) -> int:
    """Extra iterations per column a reused preconditioner may cost: ``Δ``."""
    return max(1, round(SETUP_ITERATIONS / columns))


def _reused_reference(n_ref: int, measured: float, tolerance: float) -> int:
    """``n_ref``, measured at ``measured``, scaled to a reused solve's tolerance.

    GMRES gains about the same digits per iteration, so ``n_ref`` iterations
    to ``measured`` are ``n_ref * log(REUSED_TOLERANCE * tolerance) /
    log(measured)`` to the tighter tolerance of a reused solve at
    ``tolerance``.
    """
    if not 0 < measured < 1:
        return n_ref
    scale = math.log(REUSED_TOLERANCE * tolerance) / math.log(measured)
    return math.ceil(n_ref * max(scale, 1.0))


class Preconditioner(Protocol):
    """What a setup gives a Krylov solve: an NGSolve preconditioner, or its like."""

    @property
    def mat(self) -> ngs.BaseMatrix:
        """The preconditioner as a matrix: its product applies it."""
        ...


class KrylovHandle(NgsHandle):
    """One preconditioner, reused by a Krylov solve per right-hand side.

    A subclass builds the preconditioner (`_set_up`), and may check the
    operator first (`_check_operator`) and drop what a setup holds besides
    the preconditioner (`_drop_setup`).

    `refactorize` swaps in another operator and keeps the preconditioner
    (preconditioner reuse across a sweep) when a solve on it measured a
    reference iteration count, setting it up again when reusing it costs
    more than a setup: a `gfem._linalg.Refactorable` handle. With
    ``reuse=False`` it sets the preconditioner up at every refactorization.
    Every setup is told the columns one setup is expected to serve:
    ``width * solves`` with reuse, ``width`` without it; a `refactorize` may
    state its own ``solves``, which later setups count until another
    `refactorize` states one.
    """

    #: The preconditioner's name in a convergence error.
    preconditioner_name: ClassVar[str]
    #: The remedies a convergence error ends with.
    convergence_remedy: ClassVar[str] = (
        "Use linear_solver='direct', or a looser tolerance."
    )

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
        """Set the preconditioner up on the form, and the Krylov solver on it.

        ``reuse=False`` sets the preconditioner up again at every
        `refactorize`. ``width`` right-hand-side columns per solve and
        ``solves`` solve calls are expected; they reach the setups only.
        """
        form = operator.form
        if operator.layout is not None:
            self.layout = operator.layout
        self._check_operator(operator)
        self._width = width
        self.reuse = reuse
        self._expected_columns = self._columns_per_setup(solves)
        self.symmetry = symmetry
        # the held operator's frequency (Hz), which a setup may log
        self._frequency = operator.frequency
        preconditioner = self._set_up(form)
        is_complex = bool(form.space.is_complex)
        super().__init__(form.mat, operator.freedofs, symmetry, is_complex=is_complex)
        self.tolerance = tolerance
        self.maxiter = maxiter
        self._hold(form, preconditioner)
        #: What the last step did, and why (see the module docstring).
        self.last_action: str | None = "factorized"
        self.last_reason: str | None = "first"

    def _check_operator(self, operator: Operator) -> None:
        """Hook: refuse an operator the preconditioner cannot serve."""

    def _set_up(self, form: ngs.BilinearForm) -> Preconditioner:
        """Assemble ``form`` and set the preconditioner up on it.

        Returns the preconditioner: its ``mat`` applies it. The setup may
        read ``self._expected_columns``, the columns it is expected to serve.
        """
        raise NotImplementedError

    def _drop_setup(self) -> None:
        """Hook: drop what a setup holds besides the preconditioner."""

    def expect_solves(self, solves: int) -> None:
        """Restate the solve calls the next setups serve, as `refactorize` does.

        For the setups only; the held preconditioner stays.
        """
        with self._lock:
            self._expected_columns = self._columns_per_setup(solves)

    @property
    def reference_iterations(self) -> int | None:
        """``n_ref`` of the held preconditioner (see `refactorize`), or ``None``."""
        return self._reference_iterations

    def assume_reference(self, iterations: int) -> None:
        """Take ``iterations`` as ``n_ref`` of a fresh preconditioner not yet measured.

        For a solve from the previous design's solution: it takes
        fewer iterations than a cold solve, and would set ``n_ref`` so low
        that the next frequencies' reused solves cap and set the
        preconditioner up again. The ``n_ref`` the previous design measured
        stands in.
        """
        with self._lock:
            if not self._reused and self._reference_iterations is None:
                self._reference_iterations = int(iterations)

    def _columns_per_setup(self, solves: int) -> int:
        """The columns one setup serves when ``solves`` solve calls are expected.

        Without reuse every refactorization sets the preconditioner up for one solve.
        """
        return self._width * (solves if self.reuse else 1)

    def _hold(self, form: ngs.BilinearForm, preconditioner) -> None:
        """Solve ``form``'s matrix on a preconditioner just set up from it."""
        self._form = form
        self._preconditioner = preconditioner
        self._matrix = form.mat
        self._solver = self._krylov(form.mat, preconditioner)
        self._reused = False
        #: Whether the reuse came from a refactorization at a sweep's frequency.
        self._swept = False
        self._reference_iterations: int | None = None
        self._rebuild = False

    def _build(self, form: ngs.BilinearForm) -> None:
        """Set the preconditioner up again on ``form`` (assembled again); hold it."""
        with _timing.phase("setup") as span:
            # release the old setup first
            self._solver = self._preconditioner = self._matrix = self._form = None
            self._drop_setup()
            self._hold(form, self._set_up(form))
            if span.enabled:
                span.set(solver=self.kind, action="factorized", nbytes=self.nbytes)

    def _held(self) -> tuple[ngs.BaseMatrix, Preconditioner, ngs.BilinearForm]:
        """The matrix, its preconditioner and its form.

        Raises:
            RuntimeError: The handle was freed.
        """
        matrix, pre, form = self._matrix, self._preconditioner, self._form
        if self._solver is None or matrix is None or pre is None or form is None:
            raise RuntimeError("This solver handle was freed.")
        return matrix, pre, form

    def _krylov(
        self,
        matrix: ngs.BaseMatrix,
        pre: Preconditioner,
        *,
        maxiter: int | None = None,
        reference: float | None = None,
    ):
        """The Krylov solver of ``matrix`` on the preconditioner ``pre``.

        ``maxiter`` defaults to the handle's; ``reference`` is as for
        `_gmres`.
        """
        maxiter = self.maxiter if maxiter is None else maxiter
        if self.symmetry != "hermitian":
            return self._gmres(matrix, pre, maxiter=maxiter, reference=reference)
        if reference is None:
            stop = {"tol": self.tolerance}
        else:
            stop = {"atol": self.tolerance * reference}
        if self.layout.comm is not None:
            # products agreed across ranks
            return Cg(
                matrix,
                pre.mat,
                maxiter=maxiter,
                conjugate=self._complex,
                layout=self.layout,
                **stop,
            )
        return ngs.solvers.CGSolver(
            matrix, pre.mat, maxiter=maxiter, conjugate=self._complex, **stop
        )

    def _gmres(
        self,
        matrix: ngs.BaseMatrix,
        pre: Preconditioner,
        *,
        maxiter: int,
        reference: float | None = None,
        tolerance: float | None = None,
    ):
        """GMRES of ``matrix`` on ``pre``, to ``tolerance`` times ``reference``.

        ``reference=None`` stops relative to the first residual instead;
        ``tolerance`` defaults to the handle's.
        """
        tolerance = self.tolerance if tolerance is None else tolerance
        if reference is None:
            return Gmres(
                matrix, pre.mat, tol=tolerance, maxiter=maxiter, layout=self.layout
            )
        return Gmres(
            matrix,
            pre.mat,
            atol=tolerance * reference,
            maxiter=maxiter,
            layout=self.layout,
        )

    def refactorize(
        self,
        operator: Operator,
        *,
        solves: int | None = None,
        frequency: float | None = None,  # noqa: ARG002
        sweep: bool = False,
    ) -> None:
        """Assemble ``operator`` and solve it on the held preconditioner.

        Preconditioner reuse (see `gfem._linalg.Refactorable`): only the
        matrix is assembled, and the Krylov solver is rebuilt on it; the
        preconditioner is reused. ``n_ref`` is the most iterations per column
        of the first solve on a fresh preconditioner (cold or warm, at the
        prepared tolerance), or, after a capped column set the preconditioner
        up inside a solve, of the iterations spent after that setup; a solve
        on a reused preconditioner never changes it. The preconditioner is
        set up again here instead when the handle does not reuse, when
        ``n_ref`` is unset, or when a warm solve on the reused preconditioner,
        at the prepared tolerance, went past ``n_ref + Δ`` iterations per
        column (``n_ref`` scaled to the reused solve's tighter tolerance,
        `_reused_reference`), with ``Δ`` `SETUP_ITERATIONS` shared by the
        solve's columns. So every warm solve on a reused preconditioner is
        capped (see `_warm`). A GMRES solve without an initial guess on a
        reused preconditioner runs that warm path from zero when this
        refactorization is a frequency ``sweep``'s; otherwise (an eigen
        reshift, far from the operator the preconditioner was set up at) and
        for a CG solve, it sets the preconditioner up again first. The rule
        counts iterations, never times, so runs repeat exactly.

        ``solves`` restates the solve calls the next setups are expected to
        serve (``prepare``'s ``solves``), for the setups only;
        ``None`` keeps the count. A setup may come later than this call,
        inside a solve (a capped column, or a cold solve). ``sweep`` tells
        whether a cold GMRES solve may reuse the preconditioner;
        ``frequency`` is ignored (only the direct solver reads it).
        """
        form = operator.form
        with self._lock:
            _, pre, _ = self._held()
            if solves is not None:
                self._expected_columns = self._columns_per_setup(solves)
            self._frequency = operator.frequency
            if self._rebuild or not self.reuse or self._reference_iterations is None:
                reason = "rebuild rule" if self.reuse and self._rebuild else "first"
                self._build(form)
                self.last_action, self.last_reason = "factorized", reason
                return
            with _timing.phase("assemble"):
                form.Assemble()
            self._form = form
            self._matrix = form.mat
            self._solver = self._krylov(form.mat, pre)
            self._reused = True
            self._swept = sweep
            self.last_action, self.last_reason = "reused", None

    def _converged(
        self, solver, reference: float | None, tolerance: float | None = None
    ) -> bool:
        """Whether ``solver``'s last residual estimate is at the tolerance.

        ``reference`` is the residual the tolerance is relative to; ``None``
        is the first residual of the iteration. ``tolerance`` defaults to the
        handle's. A stop at ``maxiter`` or before it (a breakdown) above the
        tolerance is not converged.
        """
        residuals = solver.residuals
        if not residuals:
            return True
        start = residuals[0] if reference is None else reference
        tolerance = self.tolerance if tolerance is None else tolerance
        return residuals[-1] <= tolerance * start

    def _finish(
        self,
        solver,
        b: ngs.BaseVector,
        x: ngs.BaseVector,
        reference: float | None = None,
        iterations: int = 0,
    ) -> int:
        """Finish a column's solve: restart it if needed, count, raise.

        ``solver`` solved ``b`` into ``x``; a restart updates ``x`` in place.
        A Krylov solve that stops before its ``maxiter`` above the tolerance
        (CG's silent exit, or any solver's) restarts once from ``x``, with
        the iterations left, to ``tolerance`` times the same start. (gfem's
        GMRES returns the right iterate on a lucky breakdown, so it meets the
        tolerance there.) ``reference`` is as for `_converged`;
        ``iterations`` were spent on the column before ``solver`` (a capped
        solve on a reused preconditioner). Returns the column's iterations,
        these included.
        """
        spent = iterations + int(solver.iterations)
        residuals = solver.residuals
        if not residuals:
            return spent
        start = residuals[0] if reference is None else reference
        early = int(solver.iterations) < solver.maxiter
        if early and not self._converged(solver, start):
            matrix, pre, _ = self._held()
            left = solver.maxiter - solver.iterations
            solver = self._krylov(matrix, pre, maxiter=left, reference=start)
            solver.Solve(rhs=b, sol=x, initialize=False)
            spent += int(solver.iterations)
            early = int(solver.iterations) < solver.maxiter
        if self._converged(solver, start):
            return spent
        name = "CG" if self.symmetry == "hermitian" else "GMRES"
        reached = f"{solver.residuals[-1] / start:.3e}"
        if early:
            what = (
                f": it stopped short of the tolerance after {spent} iterations, "
                f"restart included (reached {reached})"
            )
        else:
            what = f" in {self.maxiter} iterations (reached {reached})"
        raise LinearSolverConvergenceError(
            f"{self.preconditioner_name} + {name} did not converge to the relative "
            f"tolerance {self.tolerance:g}{what}. {self.convergence_remedy}"
        )

    def _inverse_map(
        self, B: np.ndarray, tolerance: float | None, x0: np.ndarray | None = None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        _, _, form = self._held()
        with self._lock, self.layout.blas_threads(1):  # reentrant: _map, _warm
            if self._reused and (
                self.symmetry == "hermitian" or (x0 is None and not self._swept)
            ):
                # CG has no warm path, and an eigen reshift's shift-invert
                # solves run on a preconditioner set up at the new shift
                self._build(form)
                self.last_action, self.last_reason = "factorized", "cold solve"
            elif self._reused and x0 is None:
                # GMRES on the reused preconditioner from zero
                x0 = np.zeros(B.shape, dtype=np.complex128 if self._complex else float)
            solver = self._solver
            prepared = self.tolerance
            # only a solve at the prepared tolerance measures n_ref
            measure = tolerance is None or tolerance == prepared
            if tolerance is not None:
                # NGSolve's Krylov solvers read .tol on every Solve
                solver.tol = self.tolerance = tolerance
            try:
                if x0 is not None and self.symmetry != "hermitian":
                    return self._warm(B, x0, measure=measure, prepared=prepared)
                counts: list[int] = []
                out = self._map(
                    lambda _: solver,
                    B,
                    after=lambda *column: counts.append(self._finish(*column)),
                )
                # a cold solve runs on a fresh preconditioner
                self._set_reference(counts, measure=measure)
                return out, tuple(counts)
            finally:
                solver.tol = self.tolerance = prepared

    def _set_reference(self, iterations: list[int], *, measure: bool) -> None:
        """Set ``n_ref`` to the most ``iterations`` of a column, when unset.

        ``iterations`` ran on a fresh preconditioner, one entry per column.
        Only a solve at the prepared tolerance (``measure``) sets ``n_ref``.
        """
        if measure and iterations and self._reference_iterations is None:
            self._reference_iterations = max(iterations)

    def _drop_a_bad_guess(self, matrix, pre, b, x, r) -> float:
        """``|P b|``, after zeroing ``x`` if its residual exceeds that (worse than 0).

        ``r`` is scratch space.
        """
        layout = self.layout
        r.data = pre.mat * b
        reference = layout.state_norm(r)
        r.data = b - matrix * x
        b_res = r.CreateVector()
        b_res.data = pre.mat * r
        if layout.state_norm(b_res) >= reference:
            x[:] = 0
            layout.mark_state(x)
        return reference

    def _warm(
        self, B: np.ndarray, x0: np.ndarray, *, measure: bool, prepared: float
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """GMRES from the initial guess ``x0``, to ``tolerance * |P b|``.

        The stopping residual is relative to the preconditioned right-hand
        side, not to the first residual, so a good guess saves iterations.
        A guess whose residual exceeds ``|P b|`` (worse than zero) is
        dropped.

        On a reused preconditioner (see `refactorize`) the tolerance is
        `REUSED_TOLERANCE` times tighter, since a preconditioner set up at
        another matrix measures the error less well, and ``n_ref`` (measured
        at the ``prepared`` tolerance) is scaled to it (`_reused_reference`).
        Each column there stops at
        ``min(maxiter, n_ref + 2Δ)`` iterations; a column that stops there
        unconverged sets the preconditioner up again at this matrix and goes
        on from where it stopped. A breakdown above the tolerance on the reused
        preconditioner (a stop before the cap) sets it up again the same
        way: that is the warm path's recovery, and only a breakdown on the
        fresh preconditioner meets the restart rule of `_finish`. Only a
        solve with ``measure`` true (at the prepared tolerance) counts: the
        iterations spent on a fresh preconditioner set ``n_ref`` when it is
        unset, and those spent on the reused one decide the rebuild rule,
        when every column stayed on it. Returns the solution and each
        column's iterations.
        """
        cols, flat = self._columns(B)
        guesses, _ = self._columns(x0)
        matrix, _, _ = self._held()
        out = np.zeros(cols.shape, dtype=np.complex128 if self._complex else float)
        # a load, a state, and what the first preconditioned residual overwrites
        b, x, r = (
            matrix.CreateColVector(),
            matrix.CreateRowVector(),
            matrix.CreateColVector(),
        )
        layout = self.layout
        slack = _slack(cols.shape[1])
        stale: list[int] = []  # per column, on the reused preconditioner
        fresh: list[int] = []  # per column, after a setup at this matrix
        counts: list[int] = []  # per column, both of these
        with self._lock:
            # refactorize reuses a preconditioner only with n_ref set, and a
            # reused one solves to a tighter tolerance, in more iterations
            tight = REUSED_TOLERANCE * self.tolerance
            n_ref = _reused_reference(
                self._reference_iterations or 0, prepared, self.tolerance
            )
            for j in range(cols.shape[1]):
                matrix, pre, form = self._held()
                on_reused = self._reused
                cap = self.maxiter
                tolerance = self.tolerance
                if on_reused:
                    cap = min(cap, n_ref + 2 * slack)
                    tolerance = tight
                b.FV().NumPy()[:] = cols[:, j]
                layout.mark_load(b)
                x.FV().NumPy()[:] = guesses[:, j]
                layout.mark_state(x)
                reference = self._drop_a_bad_guess(matrix, pre, b, x, r)
                solver = self._gmres(
                    matrix, pre, maxiter=cap, reference=reference, tolerance=tolerance
                )
                solver.Solve(rhs=b, sol=x, initialize=False)
                spent = 0
                if on_reused and not self._converged(solver, reference, tolerance):
                    # set up at this matrix, and go on from the partial iterate
                    spent = int(solver.iterations)
                    self._build(form)
                    self.last_action = "reused, capped, refactorized"
                    self.last_reason = "capped"
                    matrix, pre, _ = self._held()
                    r.data = pre.mat * b
                    reference = layout.state_norm(r)
                    solver = self._gmres(
                        matrix, pre, maxiter=self.maxiter, reference=reference
                    )
                    solver.Solve(rhs=b, sol=x, initialize=False)
                    on_reused = False
                total = self._finish(solver, b, x, reference, spent)
                (stale if on_reused else fresh).append(total - spent)
                counts.append(total)
                out[:, j] = x.FV().NumPy()
            # n_ref only from a fresh preconditioner; the rebuild rule, by
            # iteration count, only when the whole solve stayed reused; both
            # only at the prepared tolerance
            self._set_reference(fresh, measure=measure)
            if measure and self._reused and max(stale) > n_ref + slack:
                self._rebuild = True
        return (out[:, 0] if flat else out), tuple(counts)

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the preconditioner, in bytes."""
        # read once, without the lock: a concurrent free gives 0 or the old count
        held, pre = self._matrix, self._preconditioner
        if self._solver is None or held is None or pre is None:
            return 0
        matrix = local_matrix(held, self.layout.comm)
        return host_bytes(matrix) + host_bytes(pre.mat)

    def _release(self) -> None:
        """Drop the preconditioner and the form; under the lock."""
        self._preconditioner = None
        self._form = None
        self._drop_setup()
