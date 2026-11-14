"""The ``gfem.solver`` logger: what each solve step did (ADR 0120).

One INFO record per solve step, forward and backward, on the driven and
electrostatic paths, under ``jax.grad`` and ``jax.jit`` too (the solves run
in host callbacks). Silent by default: Python's root level is WARNING. The
fields are also record attributes (``extra``), for a handler or ``caplog``:
``problem``, ``frequency`` (Hz, ``None`` for a static problem),
``direction`` (``forward``/``backward``), ``solver``, ``action``
(``factorized``, ``refactorized``, ``reused``, ``reused, capped,
refactorized``, ``held``: a direct step on the values it holds),
``iterations`` (per column, ``()`` for a direct solve on fresh factors),
``reason`` (why a step set up or refactorized, else ``None``), ``guess``
(``"previous design"`` for a solve that started from the previous design's
solution (ADR 0126), else ``None``) and ``columns``
(``"adjoint port"`` for a uniform sweep's adjoint port columns, else
``None``), plus the reason's numbers where it has some.

Resolving ``linear_solver="auto"`` logs one record of the choice
(`auto_choice`).

A frequency worker logs in its own process: `collected` keeps its records
and the parent logs them with `replay`.

`_linalg` owns the logger so that it imports nothing from `_engine`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Sequence
from typing import Any

LOGGER = logging.getLogger("gfem.solver")


def solve_step(
    handle: Any,
    *,
    problem: str,
    direction: str,
    iterations: tuple[int, ...],
    frequency: float | None = None,
    guess: str | None = None,
    reused: bool = False,
    columns: str | None = None,
) -> None:
    """Log one solve step on ``handle``, the base handle that solved it.

    The action, reason and the reason's numbers are the handle's
    (``last_action``, ``last_reason``, ``last_details``; ``None`` on a
    backend that tracks none), or with ``reused`` the action ``"reused"``:
    the step solved on a handle an earlier step prepared and left as it was.
    ``guess`` names the initial guess's source; ``columns`` what the
    columns are when not the excitations' (``"adjoint port"``).
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    solver = getattr(handle, "kind", None)
    action = getattr(handle, "last_action", None)
    reason = getattr(handle, "last_reason", None)
    details = dict(getattr(handle, "last_details", None) or {})
    if reused:
        action, reason, details = "reused", None, {}
    iterations = tuple(int(i) for i in iterations)
    at = "" if frequency is None else f" at {frequency / 1e9:g} GHz"
    what = f"{solver} {action}" if action is not None else f"{solver}"
    why = "" if reason is None else f" ({reason})"
    numbers = "".join(f", {k}={v:g}" for k, v in details.items())
    start = "" if guess is None else f", from the {guess}'s solution"
    of = "" if columns is None else f" of the {columns} columns"
    LOGGER.info(
        "%s %s solve%s%s: %s%s%s, iterations %s%s",
        problem,
        direction,
        of,
        at,
        what,
        why,
        numbers,
        list(iterations),
        start,
        extra={
            "problem": problem,
            "frequency": frequency,
            "direction": direction,
            "solver": solver,
            "action": action,
            "reason": reason,
            "iterations": iterations,
            "guess": guess,
            "columns": columns,
            **details,
        },
    )


def auto_choice(
    kind: str,
    *,
    problem: str,
    reason: str | None = None,
    unknowns: int | None = None,
    estimate: int | None = None,
    budget: int | None = None,
) -> None:
    """Log what ``linear_solver="auto"`` resolved to for a ``problem`` problem.

    ``reason`` states the driven rule's numbers (ADR 0124); a fixed choice
    has none. Attributes ``linear_solver``, ``unknowns``, ``estimate`` and
    ``budget`` (bytes; ``None`` for a fixed choice).
    """
    why = "" if reason is None else f": {reason}"
    LOGGER.info(
        "linear_solver='auto' chose %r for this %s problem%s",
        kind,
        problem,
        why,
        extra={
            "problem": problem,
            "linear_solver": kind,
            "unknowns": unknowns,
            "estimate": estimate,
            "budget": budget,
        },
    )


class _Keep(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def collected(keep: bool) -> Generator[list[logging.LogRecord]]:
    """The records logged inside the block, not handled here (``keep``).

    For a process whose parent logs them (`replay`); without ``keep`` the
    list stays empty and the logger is left as it is.
    """
    if not keep:
        yield []
        return
    handler = _Keep()
    level, propagate = LOGGER.level, LOGGER.propagate
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    try:
        yield handler.records
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(level)
        LOGGER.propagate = propagate


def replay(records: Sequence[logging.LogRecord]) -> None:
    """Log another process's ``records`` (`collected`) here, in order."""
    for record in records:
        LOGGER.handle(record)
