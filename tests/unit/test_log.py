"""The log display: `gfem.log.enable` / `disable` and the log lines they print.

Everything is read from captured stderr, which is not a terminal, so neither
sink colours its output unless ``FORCE_COLOR`` says so. Records are built by
hand and handed to their logger, so a clock and a message are known exactly.
"""

import io
import logging
import os
import re
import subprocess
import sys

import pytest

import gfem
import gfem.log

#: The environment that decides colour and the world rank, cleared for each test
#: so a CI runner's ``FORCE_COLOR`` or a Slurm step's task id cannot leak in.
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
LINE = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} (DEBUG|INFO |WARN |ERROR|CRIT ) (\S+)\s+")


@pytest.fixture(autouse=True)
def _plain_environment(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def record(name="gfem.solver", level=logging.INFO, msg="hello", args=(), **extra):
    """A record as ``logger.log`` would build it, with ``extra`` attributes."""
    out = logging.LogRecord(name, level, __file__, 1, msg, args, None)
    out.__dict__.update(extra)
    return out


def emit(*records):
    for r in records:
        logging.getLogger(r.name).handle(r)


def gfem_state():
    logger = logging.getLogger("gfem")
    return list(logger.handlers), logger.level, logger.propagate


# ---------------------------------------------------------------- surface


def test_public_names():
    assert "log" in gfem.__all__
    assert gfem.log.__all__ == ["disable", "enable"]


def test_import_leaves_gfem_silent_and_disable_brings_last_resort_back():
    code = (
        "import logging, sys, gfem, gfem.log\n"
        "g = logging.getLogger('gfem')\n"
        "print(g.handlers, g.level, g.propagate)\n"
        "logging.getLogger('gfem.mpi').warning('before')\n"
        "logging.getLogger('gfem.solver').info('unseen')\n"
        "gfem.log.enable(); gfem.log.disable()\n"
        "print(g.handlers, g.level, g.propagate)\n"
        "logging.getLogger('gfem.mpi').warning('after')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert done.stdout.splitlines() == ["[] 0 True", "[] 0 True"]
    assert done.stderr.splitlines() == ["before", "after"]


def test_two_enables_leave_one_handler_and_the_latest_settings(capsys):
    gfem.log.enable()
    gfem.log.enable("WARNING")
    handlers = logging.getLogger("gfem").handlers
    assert [h.name for h in handlers] == ["gfem.log"]
    assert logging.getHandlerByName("gfem.log") is handlers[0]
    emit(record(msg="info"), record(level=logging.WARNING, msg="warning"))
    lines = capsys.readouterr().err.splitlines()
    assert [line.split()[-1] for line in lines] == ["warning"]


def test_records_print_once_beside_a_root_handler(capsys):
    root = io.StringIO()
    handler = logging.StreamHandler(root)
    logging.getLogger().addHandler(handler)
    try:
        gfem.log.enable()
        emit(record(msg="shown once", level=logging.WARNING))
        gfem.log.disable()
        assert capsys.readouterr().err.count("shown once") == 1
        assert root.getvalue() == ""
        emit(record(msg="back to root", level=logging.WARNING))
        assert root.getvalue() == "back to root\n"
    finally:
        logging.getLogger().removeHandler(handler)


def test_disable_restores_level_and_propagate_and_adds_no_handler():
    logger = logging.getLogger("gfem")
    logger.setLevel(logging.ERROR)
    try:
        gfem.log.enable("DEBUG")
        gfem.log.enable("INFO")
        assert logger.propagate is False
        gfem.log.disable()
        assert gfem_state() == ([], logging.ERROR, True)
        gfem.log.disable()
        assert gfem_state() == ([], logging.ERROR, True)
    finally:
        logger.setLevel(logging.NOTSET)


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        ((True,), {}, TypeError, "level"),
        ((2.5,), {}, TypeError, "level"),
        (("LOUD",), {}, ValueError, r"DEBUG.*CRITICAL.*got 'LOUD'\."),
        ((-1,), {}, ValueError, r"got -1\."),
        ((), {"timing": 1}, TypeError, "timing"),
        ((), {"rank0_only": "no"}, TypeError, "rank0_only"),
        ((), {"progress": None}, TypeError, "progress"),
        ((), {"rich": "auto"}, TypeError, "rich"),
    ],
)
def test_bad_arguments_raise_before_any_change(args, kwargs, error, match):
    before = gfem_state()
    with pytest.raises(error, match=match):
        gfem.log.enable(*args, **kwargs)
    assert gfem_state() == before
    gfem.log.enable()
    enabled = gfem_state()
    with pytest.raises(error, match=match):
        gfem.log.enable(*args, **kwargs)
    assert gfem_state() == enabled


def test_level_names_ignore_case_and_ints_pass_through():
    gfem.log.enable("debug")
    assert logging.getLogger("gfem").level == logging.DEBUG
    gfem.log.enable(logging.ERROR)
    assert logging.getLogger("gfem").level == logging.ERROR


# ---------------------------------------------------------------- filters


def test_warning_with_timing_shows_phase_records_and_warnings(capsys):
    gfem.log.enable("WARNING", timing=True)
    # frequency workers collect records only while gfem.solver is on for INFO
    assert logging.getLogger("gfem.solver").isEnabledFor(logging.INFO)
    emit(
        record(msg="solver-info"),
        record("gfem.timing", msg="phase-info"),
        record("gfem.solve_cache", logging.DEBUG, msg="cache-debug"),
        record("gfem.mpi", logging.WARNING, msg="mpi-warning"),
    )
    shown = [line.split()[-1] for line in capsys.readouterr().err.splitlines()]
    assert shown == ["phase-info", "mpi-warning"]


def test_phase_records_hide_by_default(capsys):
    gfem.log.enable("DEBUG")
    emit(record("gfem.timing", msg="phase-info"), record(msg="solver-info"))
    shown = [line.split()[-1] for line in capsys.readouterr().err.splitlines()]
    assert shown == ["solver-info"]


def launched(monkeypatch, rank, size):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", str(rank))
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", str(size))


def test_rank0_only_drops_info_on_other_ranks_and_keeps_warnings(monkeypatch, capsys):
    launched(monkeypatch, 1, 2)
    gfem.log.enable(timing=True)
    emit(
        record(msg="solver-info"),
        record("gfem.timing", msg="phase-info"),
        record("gfem.mpi", logging.WARNING, msg="mpi-warning"),
    )
    lines = capsys.readouterr().err.splitlines()
    assert [line.split()[-1] for line in lines] == ["mpi-warning"]
    assert " r1/2 " in lines[0]
    gfem.log.enable(rank0_only=False)
    emit(record(msg="solver-info"))
    assert " r1/2 solver-info" in capsys.readouterr().err


def test_rank_tag_shows_on_rank_0_when_several_tasks_launched(monkeypatch, capsys):
    launched(monkeypatch, 0, 2)
    gfem.log.enable()
    emit(record(msg="solved"))
    assert re.search(r" solver r0/2 solved$", capsys.readouterr().err.rstrip())


# ---------------------------------------------------------------- layout


@pytest.fixture(params=[None, False], ids=["rich", "ansi"])
def sink(request):
    """``enable``'s ``rich`` argument for each sink."""
    return request.param


def test_line_layout_and_short_names(sink, capsys):
    gfem.log.enable("DEBUG", timing=True, rich=sink)
    names = {
        "gfem.solver": "solver",
        "gfem.timing": "timing",
        "gfem.solve_cache": "cache",
        "gfem.mpi": "mpi",
        "gfem.mesh": "mesh",
        "gfem._linalg.cudss": "cudss",
        "gfem.mesh_refinement": "mesh_refinement",
    }
    levels = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO ",
        logging.WARNING: "WARN ",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRIT ",
    }
    emit(*(record(name, msg="m") for name in names))
    emit(*(record(level=level, msg="m") for level in levels))
    lines = capsys.readouterr().err.splitlines()
    matches = [LINE.match(line) for line in lines]
    assert all(matches), lines
    assert [m[2] for m in matches[: len(names)]] == list(names.values())
    assert [m[1] for m in matches[len(names) :]] == list(levels.values())
    # the name column pads to six and never truncates
    assert lines[0][13:].startswith("INFO  solver m")
    assert lines[2][13:].startswith("INFO  cache  m")
    assert lines[6][13:].startswith("INFO  mesh_refinement m")


def test_clock_is_the_record_s_wall_time(sink, capsys):
    gfem.log.enable(rich=sink)
    r = record()
    r.created, r.msecs = 1_700_000_000.25, 250.0
    emit(r)
    import time

    clock = time.strftime("%H:%M:%S", time.localtime(r.created)) + ".250"
    assert capsys.readouterr().err.startswith(clock + " INFO  solver hello")


def test_message_is_get_message_unchanged(sink, capsys):
    gfem.log.enable(rich=sink)
    r = record(
        msg="%s [bold]not markup[/bold] %r at 1.0e9 Hz",
        args=("a" * 200, [1, 2]),
    )
    emit(r)
    line = capsys.readouterr().err
    assert line.endswith(" " + r.getMessage() + "\n")
    assert line.count("\n") == 1


def _fail():
    raise ValueError("node 17 has no periodic image")


def test_traceback_is_the_stdlib_s_plain_rendering(sink, capsys):
    gfem.log.enable(rich=sink)
    try:
        _fail()
    except ValueError:
        r = record("gfem.mesh", logging.WARNING, msg="the check raised:")
        r.exc_info = sys.exc_info()
    emit(r)
    err = capsys.readouterr().err
    expected = logging.Formatter().formatException(r.exc_info)
    first, rest = err.split("\n", 1)
    assert first.endswith(" WARN  mesh   the check raised:")
    assert rest == expected + "\n"


# ---------------------------------------------------------------- colour


def test_no_escape_codes_off_a_terminal_and_the_same_text_on_both_sinks(capsys):
    records = [
        record(msg="solved", solver="direct", action="factorized"),
        record("gfem.timing", msg="assemble 0.1 s"),
        record("gfem.mpi", logging.WARNING, msg="waiting"),
    ]
    out = {}
    for sink in (None, False):
        gfem.log.enable("DEBUG", timing=True, rich=sink)
        emit(*records)
        out[sink] = capsys.readouterr().err
    assert "\x1b[" not in out[None]
    assert out[None] == out[False]
    assert len(out[None].splitlines()) == 3


def styled(sink, monkeypatch, capsys, *records):
    monkeypatch.setenv("FORCE_COLOR", "1")
    gfem.log.enable("DEBUG", timing=True, rich=sink)
    emit(*records)
    return capsys.readouterr().err


def test_force_color_and_no_color_on_the_fallback(monkeypatch, capsys):
    text = styled(False, monkeypatch, capsys, record())
    assert "\x1b[36mINFO \x1b[0m" in text
    assert "\x1b[34msolver\x1b[0m" in text
    monkeypatch.setenv("NO_COLOR", "1")
    emit(record())
    assert "\x1b[" not in capsys.readouterr().err


def test_dumb_terminal_has_no_colour_on_the_fallback(monkeypatch, capsys):
    monkeypatch.setenv("TERM", "dumb")
    gfem.log.enable(rich=False)
    emit(record())
    assert "\x1b[" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("action", "sgr"),
    [
        ("factorized", "35"),
        ("refactorized", "35"),
        ("reused", "32"),
        ("reused, capped, refactorized", "33"),
    ],
)
def test_the_solver_action_is_styled_and_its_text_kept(
    sink, action, sgr, monkeypatch, capsys
):
    r = record(
        msg="driven forward solve at 1 GHz: %s %s, iterations [1]",
        args=("direct", action),
        solver="direct",
        action=action,
    )
    text = styled(sink, monkeypatch, capsys, r)
    assert f"\x1b[{sgr}mdirect {action}\x1b[0m" in text
    assert re.sub(r"\x1b\[[0-9;]*m", "", text).endswith(r.getMessage() + "\n")


def test_an_action_absent_from_the_message_styles_nothing(monkeypatch, capsys):
    r = record(msg="plain message", solver="direct", action="factorized")
    text = styled(False, monkeypatch, capsys, r)
    assert text.endswith(" plain message\n")


# ---------------------------------------------------------------- without rich


class TestWithoutRich:
    @pytest.fixture(autouse=True)
    def _no_rich(self, monkeypatch):
        for name in [m for m in sys.modules if m == "rich" or m.startswith("rich.")]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "rich", None)

    def test_the_fallback_keeps_the_layout(self, capsys):
        gfem.log.enable()
        emit(record(msg="solved"))
        line = capsys.readouterr().err
        assert LINE.match(line)
        assert line.endswith(" INFO  solver solved\n")

    def test_requiring_rich_names_the_extra(self):
        before = gfem_state()
        with pytest.raises(ImportError, match=r"gfem\[log\]"):
            gfem.log.enable(rich=True)
        assert gfem_state() == before
