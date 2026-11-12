"""When a direct handle solves on an earlier design's factors: the amortized rule.

Preconditioner reuse for the direct solver (ADR 0094, amended): a serial
PARDISO handle held in the plan pool keeps design j's factors and solves the
next design's matrix by GMRES, left-preconditioned by them, instead of
refactorizing. A step is one refactorization call on the handle, and every
solve until the next.

`AmortizedRule` decides each step from the costs the handle times: after a
refactorization (a chain starts), reuse while the last reused step's solver
seconds are below the mean solver seconds per step of the chain, the
refactorized step counted as ``t22 + C t33``; at or above it, refactorize.
The gate: reuse only while ``R / C >= I1`` and the last refactorization took
at least `REUSE_MIN_REFACTOR_SECONDS`, with ``R = t22 / t33`` (one
refactorization over one direct column), ``C`` the columns of the last step
and ``I1`` the iterations per column of the last chain's first reused step
(`FIRST_ITERATIONS` before any). Only at the frequency the factors were made
at, and only while they are held. It reads no clock: the handle reports.
"""

from __future__ import annotations

import math
from collections.abc import Callable

#: A solve on reused factors stops at this fraction of the tolerance.
REUSED_TOLERANCE = 0.05
#: No reuse unless the last refactorization took at least this long, in
#: seconds: timing noise on a small model never turns reuse on.
REUSE_MIN_REFACTOR_SECONDS = 1.0
#: ``I1`` before any chain has had a reused step.
FIRST_ITERATIONS = 10.0


class AmortizedRule:
    """One handle's reuse decisions and the costs they read.

    The handle reports each refactorization (`refactorized`) and each solve
    (`solved_direct`, `solved_reused`), and asks at each new step whether to
    reuse (`next_step`). The state travels with the handle through the pool.
    """

    def __init__(self) -> None:
        #: Seconds of the last refactorization (``t22``), once one was timed.
        self.t22: float | None = None
        #: Seconds of one direct column (``t33``), once one was timed.
        self.t33: float | None = None
        #: Columns of the last finished step (``C``).
        self.columns = 0
        #: Iterations per column of the last chain's first reused step (``I1``).
        self.first_iterations = FIRST_ITERATIONS
        # the frequency the held factors were made at
        self._frequency: float | None = None
        # solver seconds of the refactorized step that started the chain, and
        # of each reused step since; None: no chain
        self._start: float | None = None
        self._chain: list[float] = []
        # the step in progress: reused, its columns, iterations and seconds
        self._reused = False
        self._step = (0, 0, 0.0)
        self._open = False

    @property
    def reusing(self) -> bool:
        """Whether the step in progress solves on reused factors."""
        return self._reused

    def pays(self) -> bool:
        """The gate: ``R / C >= I1`` and the last refactorization at the floor."""
        ratio = self._ratio()
        return (
            ratio is not None
            and self.t22 is not None
            and self.t22 >= REUSE_MIN_REFACTOR_SECONDS
            and ratio >= self.first_iterations
        )

    @property
    def cap(self) -> int:
        """Most iterations of a reused column: ``ceil(R / C) + 1``."""
        ratio = self._ratio()
        return 1 if ratio is None else math.ceil(ratio) + 1

    def _ratio(self) -> float | None:
        """``R / C``, or ``None`` before both were measured."""
        if not self.t22 or not self.t33 or not self.columns:
            return None
        return self.t22 / self.t33 / self.columns

    def next_step(
        self, *, frequency: float | None, held: bool, pays: Callable[[], bool]
    ) -> bool:
        """Close the step in progress; whether the next one reuses the factors.

        ``frequency`` is the next step's (``None``: not told), ``held``
        whether the factors are still held, ``pays`` the gate, read once the
        step is closed. A step that reuses is in progress after this.
        """
        self._close()
        reuse = (
            self._start is not None
            and held
            and frequency is not None
            and self._frequency is not None
            and math.isclose(frequency, self._frequency, rel_tol=1e-12, abs_tol=0.0)
            and pays()
            and (not self._chain or self._chain[-1] < self._mean(self._start))
        )
        self._reused = reuse
        self._step = (0, 0, 0.0)
        self._open = reuse
        return reuse

    def _mean(self, start: float) -> float:
        """Mean solver seconds per step of the chain started at ``start`` seconds."""
        return (start + sum(self._chain)) / (1 + len(self._chain))

    def refactorized(self, seconds: float, frequency: float | None) -> None:
        """A refactorization took ``seconds`` at ``frequency``: a step of it.

        Also inside a reused step (a capped column): the step becomes a
        refactorized one, and its chain starts.
        """
        self.t22 = seconds
        self._frequency = frequency
        self._reused = False
        # a capped step keeps its columns, not its reused iterations
        self._step = (self._step[0] if self._open else 0, 0, 0.0)
        self._open = True

    def solved_direct(self, columns: int, seconds: float) -> None:
        """``columns`` direct columns took ``seconds``."""
        if columns:
            self.t33 = seconds / columns
        n, iterations, spent = self._step
        self._step = (n + columns, iterations, spent + seconds)

    def solved_reused(self, columns: int, iterations: int, seconds: float) -> None:
        """``columns`` columns on reused factors took ``iterations`` and ``seconds``."""
        n, its, spent = self._step
        self._step = (n + columns, its + iterations, spent + seconds)

    def end_chain(self) -> None:
        """The factors went (released, or planned anew): the next step refactorizes."""
        self._close()
        self._start, self._chain = None, []
        self._frequency = None
        self._reused = False

    def _close(self) -> None:
        """Record the step in progress in the chain."""
        if not self._open:
            return
        self._open = False
        columns, iterations, seconds = self._step
        if columns:
            self.columns = columns
        if not self._reused:
            if self.t22 is not None:
                self._start = self.t22 + self.columns * (self.t33 or 0.0)
                self._chain = []
            return
        if not self._chain and columns:
            self.first_iterations = iterations / columns
        self._chain.append(seconds)
