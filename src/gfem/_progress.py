"""Progress rows on the log display's console.

A long loop opens one row and advances it once per unit::

    with _progress.task("uniform sweep", total=len(frequencies)) as row:
        for f in frequencies:
            ...
            row.advance(frequency=f, iterations=its)

and `gfem._timing.phase` reports each outermost phase it enters (`phase`),
shown in the innermost open row, or as a spinner row of its own when no row is
open. The keywords are values the loop already holds; the display formats
them.

`gfem.log.enable` calls `start` with the display's rich console when progress
is asked for, and `disable` (or the next `enable`) calls `stop`. While no
progress is shown, ``_LIVE`` is ``None`` and `task` and `phase` return the one
shared no-op `_NULL`: a loop pays one global read per call, builds nothing
and imports nothing. A frequency worker has no display: while the parent
shows progress, `forward_to` makes the worker's advances the parent's.

Rows nest per context, as timing phases do: a row opened while a row with the
same label is open joins it, so a fixed-angle sweep's one-frequency solves
advance their parent's row. The display itself is process-wide, since the
backward of a differentiated solve may run on a thread JAX made, where the
caller's context is not set. A row lives on the display that opened it until
it closes, even past `stop`.
"""

from __future__ import annotations

import contextvars
import importlib.util
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol, Self

#: The live display, or ``None`` while progress is off.
_LIVE: Live | None = None


class _Null:
    """The row and phase of a display that is off: every call does nothing."""

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def advance(self, n: int = 1, **fields: Any) -> None:
        """Count ``n`` more units done."""

    def update(self, completed: int | None = None, **fields: Any) -> None:
        """Set the units done."""

    def note(self, **fields: Any) -> None:
        """Show ``fields`` on the row."""


_NULL = _Null()


def task(label: str, total: int | None = None) -> Any:
    """A context manager for a loop's row: ``total`` units, ``None`` if unknown.

    Entering it gives the row, whose ``advance``, ``update`` and ``note`` take
    the keywords the display shows beside the count.
    """
    live = _LIVE
    if live is None:
        return _NULL
    return live.task(label, total)


def phase(name: str) -> Any:
    """A context manager around a timing phase ``name``, shown while it runs."""
    live = _LIVE
    if live is None:
        return _NULL
    return live.phase(name)


def shown(
    console: Any,
    *,
    rank: int,
    find_spec: Callable[[str], object] = importlib.util.find_spec,
) -> bool:
    """Whether ``console`` can draw progress rows on world rank ``rank``.

    Only rank 0 draws: the rows of several ranks would interleave in the
    launcher's output. In Jupyter rows need ipywidgets, and rich reports a
    notebook as not interactive; elsewhere they need a terminal that is
    interactive, since a transient row still writes blank lines into a pipe
    and an interactive console without cursor control glues the row onto the
    next log line.

    Args:
        console: The display's rich ``Console``, or ``None`` without rich.
        rank: This process's world rank.
        find_spec: `importlib.util.find_spec`, to look ipywidgets up.
    """
    if console is None or rank != 0:
        return False
    if console.is_jupyter:
        return find_spec("ipywidgets") is not None
    return bool(console.is_terminal and console.is_interactive)


def start(console: Any) -> None:
    """Show progress rows on ``console``, a rich ``Console`` on stderr, if it can."""
    from gfem import _mpi

    global _LIVE  # noqa: PLW0603 - the one switch every loop reads
    _LIVE = (
        Live(_RichBoard(console)) if shown(console, rank=_mpi.world_rank()) else None
    )


def forward_to(queue: Any) -> None:
    """Send this frequency worker's advances to the parent through ``queue``.

    A worker pool's initializer: a worker runs one counted loop, its block of
    the uniform sweep, and each ``advance`` of it puts ``(n, fields)`` on
    ``queue`` for the parent to advance its own row. Phases stay in the worker:
    the parent's row shows the parent's, and a phase per worker would flicker.
    """
    global _LIVE  # noqa: PLW0603 - the one switch every loop reads
    _LIVE = Live(_QueueBoard(queue))


def stop() -> None:
    """Stop showing progress; rows still open finish where they were opened."""
    global _LIVE  # noqa: PLW0603 - the one switch every loop reads
    _LIVE = None


class Board(Protocol):
    """What draws the rows: one key per open row."""

    def open(self, label: str, total: int | None, depth: int, *, spinner: bool) -> Any:
        """Draw a new row at nesting ``depth``; a spinner row has no count."""
        ...

    def advance(self, key: Any, n: int, fields: Mapping[str, Any]) -> None:
        """Count ``n`` more units done on row ``key``."""
        ...

    def update(
        self, key: Any, completed: int | None, fields: Mapping[str, Any]
    ) -> None:
        """Set the units done on row ``key`` (``None`` keeps them)."""
        ...

    def phase(self, key: Any, name: str) -> None:
        """Show phase ``name`` on row ``key`` (``""`` for none)."""
        ...

    def close(self, key: Any) -> None:
        """Remove row ``key``."""
        ...


#: The rows open in this context, outermost first.
_OPEN: contextvars.ContextVar[tuple[Row, ...]] = contextvars.ContextVar(
    "gfem_progress_open", default=()
)


class Live:
    """A live display: rows on ``board``, nested per context."""

    def __init__(self, board: Board) -> None:
        self.board = board

    def task(self, label: str, total: int | None) -> Row:
        """A row for a loop (`task`)."""
        return Row(self, label, total)

    def phase(self, name: str) -> _Phase:
        """The hook of a timing phase (`phase`)."""
        return _Phase(self, name)


class Row:
    """One row: entering it opens it, or joins an open row of the same label."""

    def __init__(
        self, live: Live, label: str, total: int | None, *, spinner: bool = False
    ) -> None:
        self.live, self.label, self.total, self.spinner = live, label, total, spinner
        self.key: Any = None
        #: The phase shown on the row.
        self.phase = ""
        self._joined: Row | None = None
        self._token: contextvars.Token | None = None

    def __enter__(self) -> Row:  # noqa: PYI034 - a joined row is the outer one
        outer = _OPEN.get()
        same = [row for row in outer if row.label == self.label]
        if same:
            # a nested loop of the same kind is part of the outer one
            self._joined = same[-1]
            return self._joined
        self.key = self.live.board.open(
            self.label, self.total, len(outer), spinner=self.spinner
        )
        self._token = _OPEN.set((*outer, self))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._joined is not None or self._token is None:
            return
        _OPEN.reset(self._token)
        self.live.board.close(self.key)

    def advance(self, n: int = 1, **fields: Any) -> None:
        """Count ``n`` more units done; ``fields`` show beside the count."""
        self.live.board.advance(self.key, n, fields)

    def update(self, completed: int | None = None, **fields: Any) -> None:
        """Set the units done (it may go down); ``fields`` as in `advance`."""
        self.live.board.update(self.key, completed, fields)

    def note(self, **fields: Any) -> None:
        """Show ``fields`` on the row, the count unchanged."""
        self.live.board.update(self.key, None, fields)


class _Phase:
    """A phase in the innermost open row, or its own spinner row with none open."""

    def __init__(self, live: Live, name: str) -> None:
        self.live, self.name = live, name
        self._row: Row | None = None
        self._previous = ""
        self._spinner: Row | None = None

    def __enter__(self) -> Self:
        outer = _OPEN.get()
        if outer:
            row = self._row = outer[-1]
            self._previous, row.phase = row.phase, self.name
            row.live.board.phase(row.key, self.name)
        else:
            self._spinner = Row(self.live, self.name, None, spinner=True)
            self._spinner.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._row is not None:
            self._row.phase = self._previous
            self._row.live.board.phase(self._row.key, self._previous)
        elif self._spinner is not None:
            self._spinner.__exit__(*exc)


class _QueueBoard:
    """A frequency worker's board: advances go to the parent, nothing is drawn.

    The arguments it ignores are those of `Board`.
    """

    def __init__(self, queue: Any) -> None:
        self.queue = queue

    def open(self, label: str, total: int | None, depth: int, *, spinner: bool) -> Any:  # noqa: ARG002
        return None

    def advance(self, key: Any, n: int, fields: Mapping[str, Any]) -> None:  # noqa: ARG002
        self.queue.put((n, dict(fields)))

    def update(
        self,
        key: Any,  # noqa: ARG002
        completed: int | None,  # noqa: ARG002
        fields: Mapping[str, Any],  # noqa: ARG002
    ) -> None:
        return None

    def phase(self, key: Any, name: str) -> None:  # noqa: ARG002
        return None

    def close(self, key: Any) -> None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------- the rich board


def count(completed: float, total: float | None) -> str:
    """A row's count: ``k/n``, or ``k/?`` when the total is not known."""
    return f"{int(completed)}/{'?' if total is None else int(total)}"


def detail(fields: Mapping[str, Any]) -> str:
    """``fields`` as the row shows them: a frequency in GHz, numbers short."""
    parts = []
    for name, value in fields.items():
        if value is None or value == ():
            continue
        if name == "frequency":
            parts.append(f"{value / 1e9:g} GHz")
            continue
        if isinstance(value, tuple | list):
            text = ", ".join(str(v) for v in value)
        elif isinstance(value, float):
            text = f"{value:.3g}"
        else:
            text = str(value)
        parts.append(f"{name.replace('_', ' ')} {text}")
    return "  ".join(parts)


class _RichBoard:
    """Rows drawn by one rich ``Progress`` on the display's console.

    Transient, and live only while a row is open: between solves nothing
    refreshes and stdout is left alone.
    """

    def __init__(self, console: Any) -> None:
        self.progress = _rich_progress(console)
        self._lock = threading.Lock()
        self._rows = 0
        self._fields: dict[Any, dict[str, Any]] = {}

    def open(self, label: str, total: int | None, depth: int, *, spinner: bool) -> Any:
        key = self.progress.add_task(
            "  " * depth + label, total=total, phase="", detail="", spinner=spinner
        )
        with self._lock:
            self._fields[key] = {}
            self._rows += 1
            if self._rows == 1:
                self.progress.start()
        return key

    def advance(self, key: Any, n: int, fields: Mapping[str, Any]) -> None:
        self.progress.update(key, advance=n, detail=self._detail(key, fields))

    def update(
        self, key: Any, completed: int | None, fields: Mapping[str, Any]
    ) -> None:
        self.progress.update(key, completed=completed, detail=self._detail(key, fields))

    def phase(self, key: Any, name: str) -> None:
        self.progress.update(key, phase=name)

    def close(self, key: Any) -> None:
        self.progress.remove_task(key)
        with self._lock:
            self._fields.pop(key, None)
            self._rows -= 1
            if self._rows == 0:
                self.progress.stop()

    def _detail(self, key: Any, fields: Mapping[str, Any]) -> str:
        with self._lock:
            shown_fields = self._fields.setdefault(key, {})
            shown_fields.update(fields)
            return detail(shown_fields)


def _rich_progress(console: Any) -> Any:
    """The rich ``Progress`` of the rows, its columns built here (rich imported)."""
    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.text import Text

    class Bar(BarColumn):
        def render(self, task: Any) -> Any:
            return Text("") if task.fields["spinner"] else super().render(task)

    class Count(ProgressColumn):
        def render(self, task: Any) -> Any:
            if task.fields["spinner"]:
                return Text("")
            return Text(count(task.completed, task.total), style="progress.download")

    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", markup=False),
        Bar(bar_width=24),
        Count(),
        TextColumn("{task.fields[phase]}", style="dim", markup=False),
        TextColumn("{task.fields[detail]}", markup=False),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        refresh_per_second=4,
    )
