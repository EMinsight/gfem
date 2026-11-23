"""Show gfem's log records on stderr while it solves.

gfem reports what its solvers do through the standard `logging` module, on
loggers under ``gfem`` (``gfem.solver``, ``gfem.timing``, ``gfem.solve_cache``,
...), and prints nothing of it by default. `enable` turns on the log display:
one line per record on stderr,

.. code-block:: text

    14:02:31.207 INFO  solver driven forward solve at 2 GHz: direct factorized

with the wall-clock time, the level, the logger's short name, the process's
rank when several were launched, then the record's message unchanged. The
display uses rich when it is installed (``pip install "gfem[log]"``) and plain
text with ANSI colours otherwise; the lines read the same either way. `disable`
turns it off again.

Users who want their own handlers attach them to the ``gfem`` logger and skip
`enable`: the records and their ``extra`` attributes are the same either way.
The "Watch a solve run" recipe shows the display on a sweep, its progress
rows, and its use under MPI.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

from gfem import _mpi, _progress

__all__ = ["disable", "enable"]

_NAME = "gfem.log"
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

#: Guards the handler swap, so two threads enabling at once leave one handler.
_LOCK = threading.Lock()


def enable(
    level: int | str = "INFO",
    *,
    timing: bool = False,
    rank0_only: bool = True,
    progress: bool = True,
    rich: bool | None = None,
) -> None:
    """Show gfem's log records at ``level`` and above on stderr.

    A second call replaces the first one's settings. While the display is on,
    the ``gfem`` logger stops propagating, so a root handler (``basicConfig``,
    pytest's ``caplog``) does not see gfem's records and nothing prints twice.

    Args:
        level: The lowest level shown: a name such as ``"info"``, in any case,
            or an int such as ``logging.DEBUG``.
        timing: Also show the phase records of ``gfem.timing``, whatever
            ``level`` is: ``enable("WARNING", timing=True)`` shows phase records
            and warnings only.
        rank0_only: On a process launched as a rank other than 0, show only
            warnings and above.
        progress: Show progress alongside the records where the console can
            draw it.
        rich: ``None`` uses rich if it is installed, ``False`` plain text, and
            ``True`` requires rich.

    Raises:
        TypeError: An argument has the wrong type; a bool is not a level.
        ValueError: ``level`` is an unknown name or a negative int.
        ImportError: ``rich=True`` and rich is not installed.
    """
    threshold = _level(level)
    for name, value in (
        ("timing", timing),
        ("rank0_only", rank0_only),
        ("progress", progress),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"gfem.log.enable {name} must be a bool, got {value!r}.")
    if rich is not None and not isinstance(rich, bool):
        raise TypeError(f"gfem.log.enable rich must be None or a bool, got {rich!r}.")
    console = None if rich is False else _console(required=rich is True)

    with _LOCK:
        logger = logging.getLogger("gfem")
        old = _installed(logger)
        if old is None:
            saved = (logger.level, logger.propagate)
        else:
            saved = old.saved
            _progress.stop()
            logger.removeHandler(old)
            old.close()
        handler = _Display(console, saved)
        handler.set_name(_NAME)
        handler.addFilter(_Gate(threshold, timing=timing, rank0_only=rank0_only))
        logger.addHandler(handler)
        # The logger's level is the union of what the filter can show, since
        # frequency workers collect a record only when its logger is enabled
        # for it. Level 0 would defer to root's level, hence at least 1.
        lowest = min(threshold, logging.INFO) if timing else threshold
        logger.setLevel(max(lowest, 1))
        logger.propagate = False
        if progress and console is not None:
            _progress.start(console)


def disable() -> None:
    """Turn the log display off; a no-op when it is off.

    Restores the ``gfem`` logger's level and ``propagate`` as they were before
    the first `enable`, overwriting any level set on it in between.
    """
    with _LOCK:
        logger = logging.getLogger("gfem")
        handler = _installed(logger)
        if handler is None:
            return
        _progress.stop()
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(handler.saved[0])
        logger.propagate = handler.saved[1]


def _installed(logger: logging.Logger) -> _Display | None:
    """The log display's handler on ``logger``, if it is on."""
    for handler in logger.handlers:
        if isinstance(handler, _Display):
            return handler
    return None


def _level(level: object) -> int:
    """``level`` as an int, or `TypeError` / `ValueError`."""
    if isinstance(level, bool) or not isinstance(level, (int, str)):
        raise TypeError(
            f"gfem.log.enable level must be a level name or an int, got {level!r}."
        )
    if isinstance(level, str):
        value = logging.getLevelNamesMapping().get(level.upper())
        if value is None:
            raise ValueError(
                "gfem.log.enable level must be one of "
                f"{', '.join(repr(name) for name in _LEVELS)}, got {level!r}."
            )
        return value
    if level < 0:
        raise ValueError(
            f"gfem.log.enable level must be a non-negative int, got {level!r}."
        )
    return level


def _console(*, required: bool) -> Any:
    """A rich ``Console`` on stderr, or ``None`` without rich unless ``required``.

    Built now, not at import: rich reads ``NO_COLOR``, ``FORCE_COLOR`` and
    ``TTY_COMPATIBLE`` when a console is made.
    """
    try:
        from rich.console import Console
    except ImportError as err:
        if required:
            raise ImportError(
                'gfem.log.enable(rich=True) needs rich: pip install "gfem[log]".'
            ) from err
        return None
    return Console(stderr=True)


def _world() -> tuple[int, int]:
    """This process's world rank and the number of launched tasks."""
    return _mpi.world_rank(), _mpi.launched_tasks()


class _Gate(logging.Filter):
    """What the display shows: ``level``, ``timing`` and the rank rule."""

    def __init__(self, level: int, *, timing: bool, rank0_only: bool) -> None:
        super().__init__()
        self.level, self.timing, self.rank0_only = level, timing, rank0_only

    def filter(self, record: logging.LogRecord) -> bool:
        below = record.levelno < logging.WARNING
        if below and _is_timing(record):
            shown = self.timing
        else:
            shown = record.levelno >= self.level
        # rank-local warnings show from every rank
        if shown and below and self.rank0_only:
            return _mpi.world_rank() == 0
        return shown


def _is_timing(record: logging.LogRecord) -> bool:
    return record.name == "gfem.timing" or record.name.startswith("gfem.timing.")


# ---------------------------------------------------------------- the log line

#: A span of a log line: its text and a rich style name ("" for none).
Span = tuple[str, str]

_LEVEL_TAG = {
    "DEBUG": ("DEBUG", "dim"),
    "INFO": ("INFO", "cyan"),
    "WARNING": ("WARN", "bold yellow"),
    "ERROR": ("ERROR", "bold red"),
    "CRITICAL": ("CRIT", "white on red"),
}
_SHORT = {"solve_cache": "cache", "_linalg.cudss": "cudss"}
_NAME_STYLE = {
    "solver": "blue",
    "timing": "dim",
    "cache": "cyan",
    "mpi": "yellow",
    "mesh": "yellow",
    "cudss": "blue",
}
#: The style of a solver step's ``"{solver} {action}"``: a factorization is the
#: costly step, a reuse the cheap one.
_ACTION_STYLE = {
    "factorized": "magenta",
    "refactorized": "magenta",
    "reused": "green",
    "reused, capped, refactorized": "yellow",
}
#: SGR codes of the styles above, for the plain sink.
_SGR = {
    "dim": "2",
    "bold": "1",
    "cyan": "36",
    "blue": "34",
    "yellow": "33",
    "green": "32",
    "magenta": "35",
    "bold yellow": "1;33",
    "bold red": "1;31",
    "white on red": "37;41",
}


def _short_name(name: str) -> str:
    tail = name.removeprefix("gfem.")
    return _SHORT.get(tail, tail.rsplit(".", 1)[-1].lstrip("_"))


def _spans(record: logging.LogRecord, rank: int, size: int) -> list[Span]:
    """``record`` as one log line of styled spans, its traceback below it."""
    clock = time.strftime("%H:%M:%S", time.localtime(record.created))
    tag, tag_style = _LEVEL_TAG.get(record.levelname, (record.levelname, ""))
    name = _short_name(record.name)
    spans: list[Span] = [
        (f"{clock}.{int(record.msecs):03d}", "dim"),
        (" ", ""),
        (f"{tag:<5}", tag_style),
        (" ", ""),
        (f"{name:<6}", _NAME_STYLE.get(name, "")),
        (" ", ""),
    ]
    if size > 1:
        spans += [(f"r{rank}/{size}", "bold"), (" ", "")]
    message = record.getMessage()
    if _is_timing(record):
        spans.append((message, "dim"))
    else:
        spans += _message_spans(record, message)
    if record.exc_info:
        spans.append(("\n" + _PLAIN.formatException(record.exc_info), ""))
    if record.stack_info:
        spans.append(("\n" + _PLAIN.formatStack(record.stack_info), ""))
    return spans


def _message_spans(record: logging.LogRecord, message: str) -> list[Span]:
    """The message, with a solver step's ``"{solver} {action}"`` styled."""
    solver = getattr(record, "solver", None)
    action = getattr(record, "action", None)
    style = _ACTION_STYLE.get(action, "") if isinstance(action, str) else ""
    start = message.find(f"{solver} {action}") if style and solver else -1
    if start < 0:
        return [(message, "")]
    end = start + len(f"{solver} {action}")
    return [(message[:start], ""), (message[start:end], style), (message[end:], "")]


_PLAIN = logging.Formatter()


class _Display(logging.Handler):
    """The log display's handler: `_spans` to a rich console or to stderr."""

    def __init__(self, console: Any, saved: tuple[int, bool]) -> None:
        super().__init__()
        #: ``gfem``'s level and ``propagate`` from before the first `enable`.
        self.saved = saved
        self.console = console
        self.text: Any = None
        if console is not None:
            from rich.text import Text

            self.text = Text

    def emit(self, record: logging.LogRecord) -> None:
        try:
            spans = _spans(record, *_world())
            if self.console is not None:
                line = self.text.assemble(
                    *(span if span[1] else span[0] for span in spans)
                )
                self.console.print(line, soft_wrap=True, highlight=False, markup=False)
            else:
                _write_ansi(spans)
        except Exception:  # noqa: BLE001 - logging's rule: report, never raise
            self.handleError(record)


def _write_ansi(spans: list[Span]) -> None:
    """Write the line to the current ``sys.stderr``, coloured where it suits."""
    stream = sys.stderr
    if stream is None:
        return
    if _colour(stream):
        line = "".join(
            f"\x1b[{_SGR[style]}m{text}\x1b[0m" if style and text else text
            for text, style in spans
        )
    else:
        line = "".join(text for text, _ in spans)
    stream.write(line + "\n")
    stream.flush()


def _colour(stream: Any) -> bool:
    """Whether the plain sink colours ``stream``, by rich's own rules."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False
