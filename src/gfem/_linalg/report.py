"""The ``gfem.solver`` logger: what each solve step did.

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
solution, else ``None``) and ``columns``
(``"adjoint load"`` for a uniform sweep's adjoint load columns, else
``None``), plus the reason's numbers where it has some.

Resolving ``linear_solver="auto"`` logs one record of the choice
(`auto_choice`). A driven rule logs how it resolved ``SolverOptions.gauge``
(`gauge_choice`), and each tree-cotree gauge basis built logs its sizes and
setup times (`gauge_basis`).

A port mode sweep logs one record when every output frequency passes its
guard (`port_mode_sweep`): ``ports``, ``frequencies``, ``exact_solves`` and
``rounds``. An eigenmode mode search logs one record per seed solve
(`seed_solve`), so a retry shows: ``problem``, ``attempt``, ``pairs``
(asked), ``returned``, ``modes`` (selected), ``wanted`` and ``bound`` (the
seed solve stopped at its restart bound). Mesh refinement logs its
refinement cycles on its own logger, ``gfem.mesh_refinement``
(`gfem.refinement`).

A frequency worker logs in its own process: `collected` keeps its records,
on this logger and on ``gfem.timing`` (`gfem._timing`), and the parent logs
them with `replay`, each on its own logger.

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
    columns are when not the excitations' (``"adjoint load"``).
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

    ``reason`` states the driven rule's numbers; a fixed choice
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


def gauge_choice(*, problem: str, on: bool, reason: str | None) -> None:
    """Log whether a ``problem`` solve runs through the tree-cotree gauge.

    ``reason`` says why it does not (``None`` when it does). Attributes
    ``gauge`` (``"tree_cotree"``, or ``None`` for an ungauged solve) and
    ``reason``.
    """
    state = "on" if on else f"off ({reason})"
    LOGGER.info(
        "%s: tree-cotree gauge %s",
        problem.capitalize(),
        state,
        extra={
            "problem": problem,
            "gauge": "tree_cotree" if on else None,
            "reason": reason,
        },
    )


def gauge_basis(
    *,
    v_unknowns: int,
    y_unknowns: int,
    conductors: int,
    floating: int,
    nnz: int,
    ungauged_nnz: int,
    basis_seconds: float,
    template_seconds: float,
) -> None:
    """Log one build of a tree-cotree gauge basis and its gauge template.

    The free unknowns of the driven space (``v_unknowns``) and of the gauge
    space (``y_unknowns``), the PEC conductors and their floating
    potentials, the gauged pattern's entries against the ungauged element
    graph's (``nnz``, ``ungauged_nnz``), and the setup seconds of the basis
    (gradient and spanning forest) and of the template. The attributes carry
    the same names.
    """
    LOGGER.info(
        "Tree-cotree gauge basis: %d -> %d free unknowns, %d conductors "
        "(%d floating), template %d entries (ungauged %d), basis %.3g s, "
        "template %.3g s",
        v_unknowns,
        y_unknowns,
        conductors,
        floating,
        nnz,
        ungauged_nnz,
        basis_seconds,
        template_seconds,
        extra={
            "v_unknowns": v_unknowns,
            "y_unknowns": y_unknowns,
            "conductors": conductors,
            "floating": floating,
            "nnz": nnz,
            "ungauged_nnz": ungauged_nnz,
            "basis_seconds": basis_seconds,
            "template_seconds": template_seconds,
        },
    )


def seed_solve(
    *,
    attempt: int,
    pairs: int,
    returned: int,
    modes: int,
    wanted: int,
    bound: bool,
) -> None:
    """Log one seed solve of an eigenmode mode search, once it is selected.

    ``attempt`` counts from 1 (a retry asks for twice the pairs);
    ``pairs`` is how many pairs it asked for, ``returned`` how many it
    converged, ``modes`` how many of them were selected as modes, against
    the ``wanted`` count; ``bound`` says it stopped at its restart bound.
    Attributes ``problem`` (``"eigenmode"``) and the same names.
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    LOGGER.info(
        "eigenmode seed solve %d: %d pairs asked, %d returned, "
        "%d of %d modes selected%s",
        attempt,
        pairs,
        returned,
        modes,
        wanted,
        ", restart bound reached" if bound else "",
        extra={
            "problem": "eigenmode",
            "attempt": attempt,
            "pairs": pairs,
            "returned": returned,
            "modes": modes,
            "wanted": wanted,
            "bound": bound,
        },
    )


def port_mode_sweep(
    *, ports: Sequence[str], frequencies: int, exact_solves: int, rounds: int
) -> None:
    """Log one port mode sweep, once every output frequency passes its guard.

    ``ports`` names the wave ports of the swept surface; ``frequencies`` is
    the output frequency count, ``exact_solves`` the exact port mode solves
    the sweep made (retries after a rebuild included) and ``rounds`` the
    times it checked the reduced model over the grid. The attributes carry
    the same names, ``ports`` as a tuple.
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    ports = tuple(ports)
    LOGGER.info(
        "port mode sweep for %s over %d frequencies: %d exact solves, %d rounds",
        ", ".join(ports),
        frequencies,
        exact_solves,
        rounds,
        extra={
            "ports": ports,
            "frequencies": frequencies,
            "exact_solves": exact_solves,
            "rounds": rounds,
        },
    )


class _Keep(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


#: The loggers whose records a frequency worker hands its parent.
COLLECTED = (LOGGER, logging.getLogger("gfem.timing"))


@contextlib.contextmanager
def collected(keep: bool) -> Generator[list[logging.LogRecord]]:
    """The records logged inside the block, not handled here (``keep``).

    For a process whose parent logs them (`replay`): the records of every
    logger in `COLLECTED`, in the order they were logged. Without ``keep``
    the list stays empty and the loggers are left as they are.
    """
    if not keep:
        yield []
        return
    handler = _Keep()
    saved = [(logger, logger.level, logger.propagate) for logger in COLLECTED]
    for logger in COLLECTED:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    try:
        yield handler.records
    finally:
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


def replay(records: Sequence[logging.LogRecord]) -> None:
    """Log another process's ``records`` (`collected`) here, in order.

    Each on the logger that logged it, when that logger is enabled for it.
    """
    for record in records:
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)
