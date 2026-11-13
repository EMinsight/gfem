"""The ``gfem.solver`` logger: what each solve step did (ADR 0120).

One INFO record per solve step, forward and backward, on the driven and
electrostatic paths, under ``jax.grad`` and ``jax.jit`` too (the solves run
in host callbacks). Silent by default: Python's root level is WARNING. The
fields are also record attributes (``extra``), for a handler or ``caplog``:
``problem``, ``frequency`` (Hz, ``None`` for a static problem),
``direction`` (``forward``/``backward``), ``solver``, ``action``
(``factorized``, ``refactorized``, ``reused``, ``reused, capped,
refactorized``), ``iterations`` (per column, ``()`` for a direct solve on
fresh factors) and ``reason`` (why a step set up or refactorized, else
``None``), plus the reason's numbers where it has some.

`_linalg` owns the logger so that it imports nothing from `_engine`.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("gfem.solver")


def solve_step(
    handle: Any,
    *,
    problem: str,
    direction: str,
    iterations: tuple[int, ...],
    frequency: float | None = None,
) -> None:
    """Log one solve step on ``handle``, the base handle that solved it.

    The action, reason and the reason's numbers are the handle's
    (``last_action``, ``last_reason``, ``last_details``; ``None`` on a
    backend that tracks none).
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    solver = getattr(handle, "kind", None)
    action = getattr(handle, "last_action", None)
    reason = getattr(handle, "last_reason", None)
    details = dict(getattr(handle, "last_details", None) or {})
    iterations = tuple(int(i) for i in iterations)
    at = "" if frequency is None else f" at {frequency / 1e9:g} GHz"
    what = f"{solver} {action}" if action is not None else f"{solver}"
    why = "" if reason is None else f" ({reason})"
    numbers = "".join(f", {k}={v:g}" for k, v in details.items())
    LOGGER.info(
        "%s %s solve%s: %s%s%s, iterations %s",
        problem,
        direction,
        at,
        what,
        why,
        numbers,
        list(iterations),
        extra={
            "problem": problem,
            "frequency": frequency,
            "direction": direction,
            "solver": solver,
            "action": action,
            "reason": reason,
            "iterations": iterations,
            **details,
        },
    )
