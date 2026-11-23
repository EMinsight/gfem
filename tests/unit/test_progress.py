"""Progress rows: the seam loops call, the gate, and the rows of a uniform sweep.

The seam is read through a recording display installed as ``_progress._LIVE``,
so labels, totals and advances are checked without a terminal. One smoke test
draws real rows into captured stderr: ``TTY_COMPATIBLE=1`` set before
`gfem.log.enable` builds the display's console makes rich treat it as a
terminal.
"""

import ast
import pathlib
import re

import pytest

import gfem
import gfem.log
from gfem import _progress

_ENV = (
    "FORCE_COLOR",
    "NO_COLOR",
    "TTY_COMPATIBLE",
    "TTY_INTERACTIVE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "SLURM_PROCID",
    "SLURM_STEP_NUM_TASKS",
)


@pytest.fixture(autouse=True)
def _plain_environment(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- off


def test_off_every_call_is_the_one_shared_no_op():
    assert _progress._LIVE is None
    row = _progress.task("uniform sweep", total=3)
    assert row is _progress._NULL
    assert _progress.phase("solve") is _progress._NULL
    with row as inside:
        assert inside is _progress._NULL
        inside.advance(frequency=1e9)
        inside.update(completed=2, exact_solves=3)
        inside.note(cycle=1)


def test_enable_on_a_pipe_draws_nothing():
    # captured stderr is not a terminal
    gfem.log.enable()
    assert _progress._LIVE is None


def test_enable_on_a_terminal_starts_progress_and_disable_stops_it(monkeypatch):
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    gfem.log.enable()
    assert _progress._LIVE is not None
    gfem.log.disable()
    assert _progress._LIVE is None


@pytest.mark.parametrize(
    "keywords", [{"progress": False}, {"rich": False}], ids=["progress", "rich"]
)
def test_no_progress_when_asked_for_none_or_without_rich(monkeypatch, keywords):
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    gfem.log.enable(**keywords)
    assert _progress._LIVE is None


def test_rank_one_draws_no_progress_even_showing_its_records(monkeypatch):
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    gfem.log.enable(rank0_only=False)
    assert _progress._LIVE is None


# ---------------------------------------------------------------- the gate


class _Console:
    """What the gate reads off a rich console."""

    def __init__(self, *, jupyter=False, terminal=False, interactive=False):
        self.is_jupyter = jupyter
        self.is_terminal = terminal
        self.is_interactive = interactive


def _find(present):
    return lambda name: object() if name in present else None


TERMINAL = _Console(terminal=True, interactive=True)
NOTEBOOK = _Console(jupyter=True)

GATE = {
    "no rich": (None, 0, (), False),
    "rank 1": (TERMINAL, 1, (), False),
    "pipe": (_Console(), 0, (), False),
    # TTY_INTERACTIVE alone: interactive but no cursor control
    "interactive only": (_Console(interactive=True), 0, (), False),
    "notebook without ipywidgets": (NOTEBOOK, 0, (), False),
    "terminal": (TERMINAL, 0, (), True),
    "notebook with ipywidgets": (NOTEBOOK, 0, ("ipywidgets",), True),
}


@pytest.mark.parametrize("case", GATE)
def test_the_gate(case):
    console, rank, present, shown = GATE[case]
    assert _progress.shown(console, rank=rank, find_spec=_find(present)) is shown


# ---------------------------------------------------------------- rich import


@pytest.mark.parametrize("module", ["log.py", "_progress.py"])
def test_rich_is_imported_only_inside_functions(module):
    path = pathlib.Path(gfem.__file__).parent / module
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    inside = {
        id(node)
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(re.match(r"rich(\.|$)", name) for name in names):
            assert id(node) in inside, f"{module}:{node.lineno} imports rich"


# ---------------------------------------------------------------- rows


class Board:
    """A board that records what it is asked to draw, as ``(event, ...)`` tuples."""

    def __init__(self):
        self.events = []
        self.rows = {}  # key -> [label, total, depth, completed, closed]

    def open(self, label, total, depth, *, spinner):
        key = len(self.rows)
        self.rows[key] = {
            "label": label,
            "total": total,
            "depth": depth,
            "spinner": spinner,
            "done": 0,
            "closed": False,
        }
        self.events.append(("open", label, total, depth, spinner))
        return key

    def advance(self, key, n, fields):
        self.rows[key]["done"] += n
        self.events.append(("advance", self.rows[key]["label"], n, dict(fields)))

    def update(self, key, completed, fields):
        if completed is not None:
            self.rows[key]["done"] = completed
        self.events.append(("update", self.rows[key]["label"], completed, dict(fields)))

    def phase(self, key, name):
        self.events.append(("phase", self.rows[key]["label"], name))

    def close(self, key):
        self.rows[key]["closed"] = True
        self.events.append(("close", self.rows[key]["label"]))

    def summary(self):
        """Each row opened: label, total, units done, closed."""
        return [
            (r["label"], r["total"], r["done"], r["closed"]) for r in self.rows.values()
        ]


@pytest.fixture
def board(monkeypatch):
    out = Board()
    monkeypatch.setattr(_progress, "_LIVE", _progress.Live(out))
    return out


def test_rows_nest_and_phases_show_in_the_innermost(board):
    with _progress.task("mesh refinement") as outer:
        with _progress.task("uniform sweep", total=2) as inner:
            with _progress.phase("solve"):
                inner.advance(frequency=1e9)
            inner.advance(frequency=2e9)
        outer.advance(cycle=0)
    assert board.events == [
        ("open", "mesh refinement", None, 0, False),
        ("open", "uniform sweep", 2, 1, False),
        ("phase", "uniform sweep", "solve"),
        ("advance", "uniform sweep", 1, {"frequency": 1e9}),
        ("phase", "uniform sweep", ""),
        ("advance", "uniform sweep", 1, {"frequency": 2e9}),
        ("close", "uniform sweep"),
        ("advance", "mesh refinement", 1, {"cycle": 0}),
        ("close", "mesh refinement"),
    ]


def test_a_nested_phase_hands_the_row_back_its_outer_phase(board):
    with (
        _progress.task("uniform sweep", total=1),
        _progress.phase("solve"),
        _progress.phase("setup"),
    ):
        pass
    phases = [e[2] for e in board.events if e[0] == "phase"]
    assert phases == ["solve", "setup", "solve", ""]


def test_a_phase_with_no_row_open_is_a_spinner_row_of_its_own(board):
    with _progress.phase("discretize"), _progress.phase("assemble"):
        pass
    assert board.events == [
        ("open", "discretize", None, 0, True),
        ("phase", "discretize", "assemble"),
        ("phase", "discretize", ""),
        ("close", "discretize"),
    ]


def test_a_row_of_an_open_label_joins_it(board):
    with _progress.task("uniform sweep", total=3) as parent:
        for _ in range(3):
            with _progress.task("uniform sweep", total=1) as child:
                assert child is parent
                child.advance()
    assert board.summary() == [("uniform sweep", 3, 3, True)]


def test_an_unknown_total_counts_up_with_a_question_mark(board):
    with _progress.task("adaptive sweep") as row:
        row.advance()
        row.advance()
    assert board.summary() == [("adaptive sweep", None, 2, True)]
    assert _progress.count(2, None) == "2/?"
    assert _progress.count(2, 5) == "2/5"


def test_an_exception_closes_the_row(board):
    def interrupted():
        with _progress.task("uniform sweep", total=3) as row:
            row.advance()
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert board.summary() == [("uniform sweep", 3, 1, True)]
    # the context holds no row any more
    with _progress.task("uniform sweep", total=1):
        pass
    assert board.events[-2][:4] == ("open", "uniform sweep", 1, 0)


def test_a_row_opened_before_stop_finishes_on_its_display(board):
    with _progress.task("uniform sweep", total=2) as row:
        row.advance()
        _progress.stop()
        row.advance()
        # a new loop opened now draws nothing
        assert _progress.task("adaptive sweep") is _progress._NULL
    assert board.summary() == [("uniform sweep", 2, 2, True)]


def test_a_row_open_through_disable_finishes_on_the_real_display(monkeypatch):
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    gfem.log.enable()
    live = _progress._LIVE
    with _progress.task("uniform sweep", total=2) as row:
        row.advance()
        gfem.log.disable()
        assert live.board.progress.live.is_started
        row.advance()
    assert not live.board.progress.live.is_started


def test_detail_shows_a_frequency_in_ghz_and_skips_empty_values():
    shown = _progress.detail({"frequency": 21e9, "iterations": (), "error": 1.23456e-3})
    assert shown == "21 GHz  error 0.00123"
    assert _progress.detail({"iterations": (12, 14)}) == "iterations 12, 14"


def test_timing_phases_show_whether_or_not_their_records_are_logged(board):
    from gfem import _timing

    # a phase nested in the same phase is part of the outer one
    with (
        _progress.task("uniform sweep", total=1),
        _timing.phase("solve"),
        _timing.phase("solve"),
        _timing.phase("setup"),
    ):
        pass
    phases = [e[2] for e in board.events if e[0] == "phase"]
    assert phases == ["solve", "setup", "solve", ""]
