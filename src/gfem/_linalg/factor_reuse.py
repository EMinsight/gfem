"""When a direct handle solves on an earlier design's factors: the amortized rule.

Preconditioner reuse for the direct solver: a serial
direct handle (PARDISO, or sparse Cholesky without MKL) of an operator equal
to its transpose, held in the plan pool, keeps design j's factors and solves
the next design's matrix by GMRES, left-preconditioned by them, instead of
refactorizing. A step is one refactorization call on the handle with new
values, and every solve until the next (the same values again are no step).
In a sweep each frequency base carries its own rule.

`AmortizedRule` decides each step from counts, never a clock, in
units of one direct-solve column: the plan's model ratio ``R`` (one
refactorization over one direct column, `pardiso_ratio` or `cholesky_ratio`,
taken once per plan after its analysis) and the GMRES iterations of reused
columns. After a refactorization (a chain starts), reuse while the last
reused step's total iterations are below the mean cost per step of the
chain, the refactorized step counted as ``R + C``; at or above it,
refactorize. The gate: reuse only while ``R / C >= I1`` and the plan is
above its floor (`PARDISO_FLOOR_MFLOP`, `CHOLESKY_FLOOR_NONZEROS`), with
``C`` the columns of the last step and ``I1`` the iterations per column of
the last chain's first reused step (`FIRST_ITERATIONS` before any). A gate
closed by ``I1`` alone re-probes: after `REPROBE_STEPS` refactorized steps
with it closed, the next step reuses once, and its iterations per column
become ``I1``; a probe that leaves it closed doubles the wait before the next
one, so a model where reuse never pays probes only a few times in a run.
Only at the frequency the factors were made at, and only while they are
held. The decisions are a function of the counts: a script repeats them at a
fixed thread count (``kappa`` and PARDISO's parallel ordering depend on it).

A refactorized step names its `AmortizedRule.reason`: the first condition
of `next_step` that failed, or ``capped`` for a reused step a capped
column turned into a refactorized one.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable

#: A solve on reused factors stops at this fraction of the tolerance.
REUSED_TOLERANCE = 0.05
#: ``I1`` before any chain has had a reused step: the fewest iterations per
#: column measured at lag 1 on any workload (CPW, 5).
FIRST_ITERATIONS = 5.0
#: Refactorized steps with the gate closed by ``I1`` before one reused step
#: measures it again: early designs move fastest, and the iterations settle.
REPROBE_STEPS = 5

#: PARDISO's refactorization over one direct column is ``kappa * F / nnz(L)``
#: (factorization flops over factor nonzeros); ``kappa`` by thread count,
#: measured on MN5 (Sapphire Rapids) on the 571k- and 2.49M-DOF 3D bends:
#: factorization scales with threads, the triangular solve does not.
KAPPA = (
    (1, 0.0113),
    (2, 0.0110),
    (4, 0.0075),
    (8, 0.0060),
    (16, 0.0037),
    (32, 0.0025),
    (56, 0.0018),
)
#: No reuse on a PARDISO plan of fewer factorization Mflop: about 0.35 s at 8
#: MN5 threads, 2.3 s at one; small models, test models included, stay exact.
PARDISO_FLOOR_MFLOP = 1e5
#: Sparse Cholesky's refactorization over one direct column is this times its
#: factor nonzeros per free DOF, at any thread count (it scales little).
CHOLESKY_RATIO = 0.14
#: No reuse on a sparse Cholesky plan of fewer factor nonzeros (about 1 s).
CHOLESKY_FLOOR_NONZEROS = 2e7

#: The reasons a step refactorizes, in the order `AmortizedRule.next_step`
#: checks them; then a capped column's.
FIRST = "first"
RELEASED = "factors released (retain_memory)"
OTHER_FREQUENCY = "other frequency"
BELOW_FLOOR = "below floor"
GATE = "gate"
CHAIN = "chain"
CAPPED = "capped"
#: Why a step refactorized nothing: its values are the ones held.
SAME_DESIGN = "same design"


def kappa(threads: int) -> float:
    """`KAPPA` at ``threads``, linear in ``log(threads)`` between its entries."""
    counts = [n for n, _ in KAPPA]
    if threads <= counts[0]:
        return KAPPA[0][1]
    if threads >= counts[-1]:
        return KAPPA[-1][1]
    i = bisect.bisect_right(counts, threads)
    (n0, k0), (n1, k1) = KAPPA[i - 1], KAPPA[i]
    return k0 + (k1 - k0) * math.log(threads / n0) / math.log(n1 / n0)


def pardiso_ratio(mflop: float, factor_nonzeros: int, threads: int) -> float:
    """PARDISO's model ratio: ``kappa(threads) F / nnz(L)``, ``F`` in flops."""
    if factor_nonzeros <= 0:
        return 0.0
    return kappa(threads) * mflop * 1e6 / factor_nonzeros


def cholesky_ratio(factor_nonzeros: float, free: int) -> float:
    """Sparse Cholesky's model ratio: `CHOLESKY_RATIO` ``nze / n``."""
    return CHOLESKY_RATIO * factor_nonzeros / free if free > 0 else 0.0


class AmortizedRule:
    """One handle's reuse decisions and the counts they read.

    The handle gives its plan's counts (`model_ratio`, `above_floor`, again
    for each new plan), reports each refactorization (`refactorized`) and
    each solve (`solved_direct`, `solved_reused`), and asks at each new step
    whether to reuse (`next_step`). The state travels with the handle
    through the pool.
    """

    def __init__(self, model_ratio: float, *, above_floor: bool) -> None:
        """A rule on a plan of ``model_ratio`` (``R``), above its floor or not."""
        #: One refactorization over one direct column (``R``), of the plan.
        self.model_ratio = model_ratio
        #: Whether the plan is large enough to reuse at all.
        self.above_floor = above_floor
        #: Columns of the last finished step (``C``).
        self.columns = 0
        #: Iterations per column of the last chain's first reused step (``I1``).
        self.first_iterations = FIRST_ITERATIONS
        # the frequency the held factors were made at
        self._frequency: float | None = None
        # the cost in columns of the refactorized step that started the
        # chain, and the iterations of each reused step since; None: no chain
        self._start: float | None = None
        self._chain: list[int] = []
        # the step in progress: reused, its columns and iterations
        self._reused = False
        self._step = (0, 0)
        self._open = False
        # refactorized steps since the gate closed on ``I1`` alone, the steps
        # to wait before the next probe, and whether the last open was a probe
        self._closed_steps = 0
        self._wait = REPROBE_STEPS
        self._probing = False
        #: Why the last step refactorized, ``None`` while it reuses.
        self.reason: str | None = FIRST
        # why the chain ended: a new plan, or factors released
        self._ended = FIRST

    @property
    def reusing(self) -> bool:
        """Whether the step in progress solves on reused factors."""
        return self._reused

    def pays(self) -> bool:
        """The gate: ``R / C >= I1`` on a plan above its floor."""
        ratio = self._open_ratio()
        return ratio is not None and ratio >= self.first_iterations

    def _open_ratio(self) -> float | None:
        """``R / C`` on a plan above its floor, once a step had columns."""
        return self._ratio() if self.above_floor else None

    @property
    def cap(self) -> int:
        """Most iterations of a reused column: ``ceil(R / C) + 1``."""
        ratio = self._ratio()
        return 1 if ratio is None else math.ceil(ratio) + 1

    @property
    def ratio(self) -> float | None:
        """``R / C``, or ``None`` before a step had columns."""
        return self._ratio()

    def _ratio(self) -> float | None:
        if not self.columns:
            return None
        return self.model_ratio / self.columns

    def next_step(
        self, *, frequency: float | None, held: bool, pays: Callable[[], bool]
    ) -> bool:
        """Close the step in progress; whether the next one reuses the factors.

        ``frequency`` is the next step's (``None``: not told), ``held``
        whether the factors are still held, ``pays`` the gate, read once the
        step is closed. A step that reuses is in progress after this; one
        that does not names the first failing condition in `reason`.
        """
        self._close()
        reason = None
        if self._start is None:
            reason = self._ended
        elif not held:
            reason = RELEASED
        elif (
            frequency is None
            or self._frequency is None
            or not math.isclose(frequency, self._frequency, rel_tol=1e-12, abs_tol=0.0)
        ):
            reason = OTHER_FREQUENCY
        elif not self._gate(pays):
            reason = GATE if self.above_floor else BELOW_FLOOR
        elif self._chain and self._chain[-1] >= self._mean(self._start):
            reason = CHAIN
        reuse = reason is None
        self.reason = reason
        self._reused = reuse
        self._step = (0, 0)
        self._open = reuse
        return reuse

    def _gate(self, pays: Callable[[], bool]) -> bool:
        """``pays()``, or a re-probe of a gate closed by ``I1`` for long enough."""
        if pays():
            self._closed_steps = 0
            self._wait = REPROBE_STEPS
            self._probing = False
            return True
        if self._probing:
            # the probe left the gate closed: wait twice as long next time
            self._probing = False
            self._wait *= 2
        if self._open_ratio() is None:
            # closed by the floor: no probe
            self._closed_steps = 0
            return False
        if self._closed_steps < self._wait:
            self._closed_steps += 1
            return False
        self._closed_steps = 0
        self._probing = True
        return True

    def _mean(self, start: float) -> float:
        """Mean cost per step, in columns, of the chain started at ``start``."""
        return (start + sum(self._chain)) / (1 + len(self._chain))

    def refactorized(self, frequency: float | None) -> None:
        """A refactorization at ``frequency``: a step of it.

        Also inside a reused step (a capped column): the step becomes a
        refactorized one, and its chain starts.
        """
        if self._reused:
            self.reason = CAPPED
        self._frequency = frequency
        self._reused = False
        # a capped step keeps its columns, not its reused iterations
        self._step = (self._step[0] if self._open else 0, 0)
        self._open = True

    def solved_direct(self, columns: int) -> None:
        """``columns`` direct columns were solved."""
        n, iterations = self._step
        self._step = (n + columns, iterations)

    def solved_reused(self, columns: int, iterations: int) -> None:
        """``columns`` columns on reused factors took ``iterations`` in all."""
        n, its = self._step
        self._step = (n + columns, its + iterations)

    def end_chain(self, *, released: bool = False) -> None:
        """The factors went (``released``, or planned anew).

        The next step refactorizes, and names the cause (`reason`).
        """
        self._ended = RELEASED if released else FIRST
        self._close()
        self._start, self._chain = None, []
        self._frequency = None
        self._reused = False

    def _close(self) -> None:
        """Record the step in progress in the chain."""
        if not self._open:
            return
        self._open = False
        columns, iterations = self._step
        if columns:
            self.columns = columns
        if not self._reused:
            self._start = self.model_ratio + self.columns
            self._chain = []
            return
        if not self._chain and columns:
            self.first_iterations = iterations / columns
        self._chain.append(iterations)
