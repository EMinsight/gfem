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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from recording_board import RecordingBoard
from solver_misses import cannot_converge_at

import gfem
import gfem.log
from gfem import _progress
from gfem._engine import driven
from gfem._engine.cache import Entry
from gfem._engine.driven import DrivenRule, port_loads
from gfem._engine.driven_sweep import (
    ADJOINT_ROW,
    UNIFORM_ROW,
    FixedAngleChildren,
    SweepSplit,
)
from gfem._engine.solve_request import SolveRequest
from gfem._engine.sweep_outputs import SweepOutputs

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


@pytest.fixture
def board(monkeypatch):
    out = RecordingBoard()
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


# ---------------------------------------------------------------- uniform sweep

F = (21e9, 22e9, 23e9)
DIRECT = gfem.SolverOptions(linear_solver="direct")


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=2.0)},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def loops(board):
    """The loop rows the board drew, spinner rows of phases left out."""
    return [
        (r["label"], r["total"], r["done"], r["closed"])
        for r in board.rows.values()
        if not r["spinner"]
    ]


def test_a_uniform_sweep_advances_its_row_once_per_frequency(line, board):
    gfem.release_memory()
    gfem.solve(line, gfem.Driven(F), options=DIRECT)
    assert loops(board) == [("uniform sweep", 3, 3, True)]
    advances = [e[3] for e in board.events if e[:2] == ("advance", "uniform sweep")]
    assert [a["frequency"] for a in advances] == list(F)
    # a direct solve has no Krylov iterations
    assert all(a["iterations"] == () for a in advances)
    # the solve's phases ran in the row
    phases = {e[2] for e in board.events if e[:2] == ("phase", "uniform sweep")}
    assert {"assemble", "setup", "solve"} <= phases
    # discretizing ran before the row opened, as a spinner
    assert ("open", "discretize", None, 0, True) in board.events


def test_a_sweep_stopping_at_an_unsolved_frequency_closes_its_row(
    line, board, monkeypatch
):
    gfem.release_memory()
    cannot_converge_at(monkeypatch, F[1:2])
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    with pytest.raises(gfem.errors.LinearSolverConvergenceError):
        gfem.solve(line, gfem.Driven(F), options=options)
    assert loops(board) == [("uniform sweep", 3, 1, True)]
    assert all(r["closed"] for r in board.rows.values())


def test_a_gradient_adds_the_adjoint_sweep_s_row(line, board):
    gfem.release_memory()
    n = len(line.mesh.tetrahedra)

    def objective(eps):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        s = gfem.solve(model, gfem.Driven(F[:2]), options=DIRECT).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    jax.grad(objective)(jnp.full(n, 2.0))
    assert loops(board) == [
        ("uniform sweep", 2, 2, True),
        ("uniform sweep (adjoint)", 2, 2, True),
    ]
    advances = [
        e[3] for e in board.events if e[:2] == ("advance", "uniform sweep (adjoint)")
    ]
    assert sorted(a["frequency"] for a in advances) == list(F[:2])


def test_an_adjoint_only_pass_has_the_adjoint_sweep_s_row(line, board):
    gfem.release_memory()
    rule = DrivenRule(line, gfem.Driven(F[:2], excite="p1"), DIRECT)
    rule.solve_adjoints(line, port_loads(("p2",)))
    assert loops(board) == [("uniform sweep (adjoint)", 2, 2, True)]


def test_a_2d_sweep_has_the_uniform_sweep_s_row(board):
    data = pathlib.Path(__file__).parents[2] / "docs" / "tutorials" / "data"
    mesh = gfem.Mesh.from_gmsh(data / "line2d.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=4.0)},
        boundaries=(gfem.PEC("side"),),
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )
    problem = gfem.Driven((4e9, 5e9), excite="p1", polarization="Hz")
    gfem.solve(model, problem, options=DIRECT)
    assert loops(board) == [("uniform sweep", 2, 2, True)]


class _Child:
    """A fixed-angle sweep's one-frequency rule: its rows as the real rule's."""

    def __init__(self, model, problem, options, **_):  # noqa: ARG002
        (self.f,) = problem.frequency

    def forward(self, model, *, keep, request):  # noqa: ARG002
        with _progress.task(UNIFORM_ROW, total=1) as row:
            row.advance(frequency=self.f)
        out = SweepOutputs.uniform(
            (0,),
            s=np.zeros((1, 1, 1), complex),
            z_ref=np.full((1, 1), 50.0),
            iterations=[],
            reused=[],
            e_field=None,
            waves=[],
        )
        return out.outputs(), Entry(physics="em", fields={"solution": np.ones((2, 1))})

    def backward(self, model, entry, cotangents, *, wanted, port_modes):  # noqa: ARG002
        with _progress.task(ADJOINT_ROW, total=1) as row:
            row.advance(frequency=self.f)
        return []

    def forget_operator(self):
        pass


def test_a_fixed_angle_sweep_s_children_advance_their_parent_s_rows(board, monkeypatch):
    monkeypatch.setattr(driven, "DrivenRule", _Child)
    problem = gfem.Driven(F, wave_vector=(10.0, 0.0, 0.0), wave_vector_frequency=F[0])
    rule = SimpleNamespace(
        adaptive=False,
        problem=problem,
        options=gfem.SolverOptions(threads=1),
        linear=SimpleNamespace(kind="direct"),
        kept=(),
        excite=("p1",),
        gauge=None,
        comm=None,
        order_frequency=F,
        wave_ports=(),
        port_names=("p1",),
        physics="em",
        stretch=None,
    )
    rule.children = FixedAngleChildren(rule)
    split = SweepSplit.of(rule)
    _, entry = split.forward(None, keep=True, request=SolveRequest())
    # the cotangent reaches the first and last frequencies only
    cotangent = np.zeros((3, 1, 1))
    cotangent[[0, 2]] = 1.0
    split.runner.backward(None, entry, {"s": cotangent}, wanted=None)
    assert loops(board) == [(UNIFORM_ROW, 3, 3, True), (ADJOINT_ROW, 3, 3, True)]


# ---------------------------------------------------------------- drawn rows

#: rich's cursor, erase and style codes.
CONTROL = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def test_a_sweep_draws_its_row_among_intact_log_lines(line, capsys, monkeypatch):
    gfem.release_memory()
    # rich treats the console as a terminal: rows draw into captured stderr
    monkeypatch.setenv("TTY_COMPATIBLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    gfem.log.enable()
    gfem.solve(line, gfem.Driven(F), options=DIRECT)
    gfem.log.disable()
    err = capsys.readouterr().err
    assert "uniform sweep" in err
    assert "\x1b[2K" in err  # the row was erased, not left in the scroll
    # a carriage return then an erase wipes the row off its line
    text = CONTROL.sub("", err.replace("\r", "\n"))
    solver = re.findall(
        r"^\d\d:\d\d:\d\d\.\d{3} INFO  solver driven forward .* at (\d+) GHz: direct",
        text,
        re.MULTILINE,
    )
    assert solver == ["21", "22", "23"]
