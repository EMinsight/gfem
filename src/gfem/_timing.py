"""The ``gfem.timing`` logger: how long each phase of a solve took.

One INFO record per phase occurrence on each rank. Silent by default: Python's
root level is WARNING, and a record is built only when the logger is enabled
for INFO (the clock is read either way). The attribute names below are the
contract a handler reads; they are record attributes, as ``extra`` sets them.

Phases, disjoint (a phase's seconds exclude every other phase timed inside it):

- ``"discretize"``: the spaces, the partition across ranks and the ports.
  Adds ``ndof``, the DOFs of the solve's H(curl) space this rank owns (the
  space's count in one process; summed over the ranks, the problem's).
- ``"assemble"``: matrix assembly, including the sum of the assembled pieces.
  With ``linear_solver="bddc_gmres"`` NGSolve builds the preconditioner while
  it assembles the first matrix of a setup, so that assembly holds the
  preconditioner's build.
- ``"setup"``: a factorization or a preconditioner, without its assembly.
  Adds ``solver`` (the linear solver), ``action`` (``"factorized"``,
  ``"refactorized"``, ``"reused"``, ``"held"``, as on ``gfem.solver``) and
  ``nbytes`` (bytes the solver holds after the setup).
- ``"solve"``: the solve at one frequency, or a whole eigenmode search. Adds
  ``iterations``: Krylov iterations per excited port, ``()`` for a direct
  solve and for an eigenmode search.

Every record carries ``phase``, ``seconds`` (wall time on this rank),
``problem`` (``"driven"``, ``"eigenmode"``, ...), ``frequency`` (Hz, or
``None`` outside a frequency), ``rank`` and ``ranks``. gfem sums nothing and
runs no collective for these records: a handler sums them and gathers them
across ranks. A phase nested in the same phase is part of the outer one.

Frequency workers time their blocks in their own processes; the parent logs
their records (`gfem._linalg.report.replay`). The workers run side by side,
so their seconds can add up to more than the parent's wall time.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import re
import time
from collections.abc import Callable, Generator
from typing import Any

LOGGER = logging.getLogger("gfem.timing")


@dataclasses.dataclass(frozen=True)
class _Scope:
    problem: str | None
    rank: int
    ranks: int


@dataclasses.dataclass
class Span:
    """One open phase: its attributes, and the seconds of other phases inside it.

    ``enabled``: the phase will log a record (an outermost span, the logger
    enabled for INFO); a caller reads costly attributes only then.
    """

    phase: str
    attrs: dict[str, Any]
    enabled: bool
    inner: float = 0.0

    def set(self, **attrs: Any) -> None:
        """Add attributes to the record."""
        self.attrs.update(attrs)


_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "gfem_timing_scope", default=None
)
_OPEN: contextvars.ContextVar[tuple[Span, ...]] = contextvars.ContextVar(
    "gfem_timing_open", default=()
)


def problem_name(problem: Any) -> str | None:
    """The name a problem's records carry: its class name in snake case."""
    if problem is None:
        return None
    name = type(problem).__name__
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


@contextlib.contextmanager
def scope(problem: str | None, comm: Any = None) -> Generator[None]:
    """Name the problem and the ranks of the records logged inside the block."""
    rank, ranks = (0, 1) if comm is None else (int(comm.rank), int(comm.size))
    token = _SCOPE.set(_Scope(problem, rank, ranks))
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextlib.contextmanager
def phase(name: str, **attrs: Any) -> Generator[Span]:
    """Time the block as phase ``name`` and log its record (see the module).

    ``attrs`` and those set on the span go on the record; ``frequency``
    defaults to the enclosing span's. A span inside an open span of the same
    phase logs nothing: the outer one holds its time. A block that raises
    logs nothing.
    """
    outer = _OPEN.get()
    if any(s.phase == name for s in outer):
        yield Span(name, {}, enabled=False)
        return
    if "frequency" not in attrs:
        attrs["frequency"] = outer[-1].attrs.get("frequency") if outer else None
    span = Span(name, attrs, enabled=LOGGER.isEnabledFor(logging.INFO))
    token = _OPEN.set((*outer, span))
    start = time.perf_counter()
    try:
        yield span
    finally:
        seconds = time.perf_counter() - start
        _OPEN.reset(token)
        if outer:
            outer[-1].inner += seconds
    if span.enabled:
        _log(span, seconds - span.inner)


def _log(span: Span, seconds: float) -> None:
    current = _SCOPE.get() or _Scope(None, 0, 1)
    attrs: dict[str, Any] = {"problem": current.problem, **span.attrs}
    frequency = attrs["frequency"]
    at = "" if frequency is None else f" at {frequency / 1e9:g} GHz"
    of = "" if attrs["problem"] is None else f"{attrs['problem']} "
    on = "" if current.ranks == 1 else f" on rank {current.rank}"
    LOGGER.info(
        "%s%s%s%s: %.3g s",
        of,
        span.phase,
        at,
        on,
        seconds,
        extra={
            **attrs,
            "phase": span.phase,
            "seconds": seconds,
            "rank": current.rank,
            "ranks": current.ranks,
        },
    )


def timed[**P, R](name: str, function: Callable[P, R]) -> Callable[P, R]:
    """``function``, each call timed as phase ``name`` (`phase`)."""

    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        with phase(name):
            return function(*args, **kwargs)

    return call
