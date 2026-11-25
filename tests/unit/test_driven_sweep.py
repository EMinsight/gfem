"""`SweepSplit`: a driven sweep's partition, block rows, join and misses.

Mesh-free: a stub rule chooses the split, and a stub `BlockRunner` (or stub
single-frequency rules behind `FixedAngleChildren`) returns canned outputs.
"""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

import gfem
from gfem._engine import driven, driven_sweep
from gfem._engine.cache import Entry
from gfem._engine.driven_sweep import (
    Block,
    BlockOutcome,
    FixedAngleChildren,
    FrequencyWorkers,
    SweepSplit,
)
from gfem._engine.solve_request import SolveRequest
from gfem._engine.sweep_outputs import SweepOutputs
from gfem._sweep.port_mode import _PER_MODE, rows_of
from gfem.errors import LinearSolverConvergenceError

SIX = (1e9, 2e9, 3e9, 4e9, 5e9, 6e9)


def _rule(frequency=SIX, *, workers=1, fixed_angle=False, wave_ports=()):
    """A stub of what `SweepSplit.of` and `FixedAngleChildren` read off a rule."""
    if fixed_angle:
        problem = gfem.Driven(
            frequency, wave_vector=(10.0, 0.0, 0.0), wave_vector_frequency=frequency[0]
        )
    else:
        problem = gfem.Driven(frequency)
    rule = SimpleNamespace(
        adaptive=False,
        problem=problem,
        options=gfem.SolverOptions(frequency_workers=workers, threads=1),
        linear=SimpleNamespace(kind="direct"),
        kept=(),
        excite=("p1",),
        gauge=None,
        comm=None,
        order_frequency=tuple(frequency),
        wave_ports=wave_ports,
        port_names=("p1",),
        physics="em",
        children=None,
        stretch=None,
        sources=(),
        dipoles=(),
        emitting=(),
    )
    if fixed_angle:
        rule.children = FixedAngleChildren(rule)
    return rule


def _outputs(rows, n_block):
    """A uniform sweep's outputs over a block of ``n_block`` at ``rows``."""
    s = np.arange(n_block, dtype=complex).reshape(n_block, 1, 1)
    return SweepOutputs.uniform(
        tuple(rows),
        s=s,
        z_ref=np.full((n_block, 1), 50.0),
        iterations=[],
        reused=[],
        e_field=None,
        waves=[],
    )


class _Runner:
    """A `BlockRunner` that records its blocks and returns canned outcomes."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.blocks = []

    def run(self, model, blocks, *, keep, request):  # noqa: ARG002
        self.blocks.extend(blocks)
        return self.outcomes


def _with(split, runner):
    return SweepSplit(**{**split.__dict__, "runner": runner})


# --- partition -----------------------------------------------------------------


def test_workers_split_the_grid_in_contiguous_blocks():
    split = SweepSplit.of(_rule(workers=4))
    assert isinstance(split.runner, FrequencyWorkers)
    assert [list(b) for b in split.blocks] == [[0, 1], [2, 3], [4], [5]]


def test_more_workers_than_frequencies_drop_the_empty_blocks():
    split = SweepSplit.of(_rule((1e9, 2e9), workers=4))
    assert [list(b) for b in split.blocks] == [[0], [1]]


def test_a_fixed_angle_sweep_takes_one_block_per_frequency():
    rule = _rule((1e9, 2e9, 3e9), fixed_angle=True)
    split = SweepSplit.of(rule)
    assert split.runner is rule.children
    assert [list(b) for b in split.blocks] == [[0], [1], [2]]


def test_a_one_pass_or_adaptive_sweep_does_not_split():
    assert SweepSplit.of(_rule()) is None
    adaptive = _rule(workers=4)
    adaptive.adaptive = True
    assert SweepSplit.of(adaptive) is None


# --- join and misses -----------------------------------------------------------


def test_two_worker_misses_raise_once_at_the_lowest():
    split = SweepSplit.of(_rule(workers=2))
    first, second = split.blocks
    runner = _Runner(
        [
            BlockOutcome(first, _outputs((0,), 3), None, (1, "no luck."), None),
            BlockOutcome(second, _outputs((0, 1), 3), None, (5, "bad."), None),
        ]
    )
    request = SolveRequest(keeps_partial=True)
    with pytest.raises(LinearSolverConvergenceError) as info:
        _with(split, runner).forward(None, keep=False, request=request)
    err = info.value
    assert str(err) == (
        "At 2e+09 Hz, no luck. The uniform sweep stopped at 2e+09 Hz and 6e+09 Hz "
        "(one frequency worker each), with 3 of 6 frequencies solved."
    )
    assert err.frequency == 2e9
    assert err.unsolved == (2e9, 3e9, 6e9)
    assert err.__cause__ is None
    assert err.__suppress_context__
    assert err._partial.outputs.rows == (0, 3, 4)
    assert err._partial.reason == "no luck."


def test_a_fixed_angle_miss_at_the_first_frequency_chains_its_cause():
    split = SweepSplit.of(_rule((1e9, 2e9), fixed_angle=True))
    cause = LinearSolverConvergenceError("GMRES stalled.", frequency=1e9)
    runner = _Runner([BlockOutcome([0], None, None, (0, "GMRES stalled."), cause)])
    with pytest.raises(LinearSolverConvergenceError) as info:
        _with(split, runner).forward(None, keep=False, request=SolveRequest())
    err = info.value
    assert str(err) == (
        "At 1e+09 Hz, GMRES stalled. The uniform sweep stopped there, with 0 of "
        "2 frequencies solved."
    )
    assert err.__cause__ is cause
    assert err.unsolved == (1e9, 2e9)
    assert err._partial is None


def test_the_blocks_join_in_frequency_order_with_their_solutions():
    split = SweepSplit.of(_rule((1e9, 2e9, 3e9), workers=2))
    first, second = split.blocks
    runner = _Runner(
        [
            BlockOutcome(first, _outputs((0, 1), 2), np.ones((4, 2)), None, None),
            BlockOutcome(second, _outputs((0,), 1), np.zeros((4, 1)), None, None),
        ]
    )
    outputs, entry = _with(split, runner).forward(
        None, keep=True, request=SolveRequest()
    )
    np.testing.assert_array_equal(np.asarray(outputs["s"]).ravel(), [0, 1, 0])
    assert entry.handle is None
    np.testing.assert_array_equal(
        entry.fields["solution"], np.concatenate([np.ones((4, 2)), np.zeros((4, 1))], 1)
    )


# --- block rows ----------------------------------------------------------------


def _arrays(n):
    """`swept_once`-shaped arrays of one mode over ``n`` frequencies."""
    arrays = {
        name: np.arange(n, dtype=float)[None, :] + k for k, name in enumerate(_PER_MODE)
    }
    arrays["solved"] = np.arange(n) % 2 == 0
    arrays["basis"] = np.eye(3)
    return arrays


def _assert_rows(blocks, arrays):
    for block in blocks:
        (rows,) = block.port_modes
        expected = rows_of(arrays, block.indices)
        assert rows.keys() == expected.keys()
        for name in rows:
            np.testing.assert_array_equal(rows[name], expected[name])


def test_each_block_takes_its_rows_of_the_parent_s_sweep(monkeypatch):
    arrays = _arrays(len(SIX))
    swept = SimpleNamespace(arrays=arrays, ports=(), sweeps=[])
    monkeypatch.setattr(driven_sweep, "port_mode_sweeps", lambda *_, **__: [swept])
    split = SweepSplit.of(_rule(workers=4, wave_ports=("p1",)))
    runner = _Runner()
    _with(split, runner).forward(None, keep=False, request=SolveRequest())
    assert len(runner.blocks) == 4
    _assert_rows(runner.blocks, arrays)


def test_each_block_takes_its_rows_of_the_handed_ones(monkeypatch):
    def refuse(*_, **__):
        raise AssertionError("a split handed rows swept again")

    monkeypatch.setattr(driven_sweep, "port_mode_sweeps", refuse)
    arrays = _arrays(3)
    split = SweepSplit.of(_rule((1e9, 2e9, 3e9), fixed_angle=True, wave_ports=("p1",)))
    runner = _Runner()
    request = SolveRequest(port_modes=[arrays])
    _with(split, runner).forward(None, keep=False, request=request)
    assert [list(b.indices) for b in runner.blocks] == [[0], [1], [2]]
    _assert_rows(runner.blocks, arrays)


# --- fixed-angle children ------------------------------------------------------


class _Child:
    """A stub single-frequency rule; ``log`` records what each child did."""

    log: ClassVar[list] = []
    misses: ClassVar[frozenset] = frozenset()

    def __init__(self, model, problem, options, **_):  # noqa: ARG002
        (self.f,) = problem.frequency
        self.log.append(("built", self.f))

    def forward(self, model, *, keep, request):  # noqa: ARG002
        self.log.append(("forward", self.f, request.keeps_partial))
        request.adjoints[self.f, "load"] = self.f
        if self.f in self.misses:
            raise LinearSolverConvergenceError("GMRES stalled.", frequency=self.f)
        entry = Entry(physics="em", fields={"solution": np.full((2, 1), self.f)})
        return _outputs((0,), 1).outputs(), entry if keep else None

    def backward(self, model, entry, cotangents, *, wanted, port_modes):  # noqa: ARG002
        self.log.append(("backward", self.f, port_modes))
        return []

    def forget_operator(self):
        self.log.append(("forget", self.f))


@pytest.fixture
def child(monkeypatch):
    monkeypatch.setattr(_Child, "log", [])
    monkeypatch.setattr(_Child, "misses", frozenset())
    monkeypatch.setattr(driven, "DrivenRule", _Child)
    return _Child


def test_fixed_angle_children_serve_forward_and_backward_once_each(child):
    frequency = (1e9, 2e9, 3e9)
    split = SweepSplit.of(_rule(frequency, fixed_angle=True))
    request = SolveRequest()
    _, entry = split.forward(None, keep=True, request=request)
    assert entry.fields["solution"].shape == (2, 3)
    # each frequency's adjoints land in the parent request's reply dict
    assert request.adjoints == {(f, "load"): f for f in frequency}
    split.runner.backward(None, entry, {"s": np.zeros((3, 1, 1))}, wanted=None)
    forward = [(("built", f), ("forward", f, True), ("forget", f)) for f in frequency]
    backward = [(("backward", f, None), ("forget", f)) for f in frequency]
    # one child per frequency, built once; its operator goes after each
    # frequency, forward and backward
    assert child.log == [e for steps in forward + backward for e in steps]


def test_fixed_angle_children_forget_the_operator_on_a_miss(child):
    child.misses = frozenset({2e9})
    split = SweepSplit.of(_rule((1e9, 2e9, 3e9), fixed_angle=True))
    with pytest.raises(LinearSolverConvergenceError) as info:
        split.forward(None, keep=False, request=SolveRequest(keeps_partial=True))
    assert info.value.frequency == 2e9
    assert info.value._partial.outputs.rows == (0,)
    assert [e for e in child.log if e[0] != "built"] == [
        ("forward", 1e9, True),
        ("forget", 1e9),
        ("forward", 2e9, True),
        ("forget", 2e9),
    ]


def test_a_child_s_backward_stands_on_the_rows_its_forward_was_handed(child):
    arrays = _arrays(2)
    split = SweepSplit.of(_rule((1e9, 2e9), fixed_angle=True, wave_ports=("p1",)))
    request = SolveRequest(port_modes=[arrays])
    _, entry = split.forward(None, keep=True, request=request)
    split.runner.backward(None, entry, {"s": np.zeros((2, 1, 1))}, wanted=None)
    handed = [e[2] for e in child.log if e[0] == "backward"]
    _assert_rows([Block([n], rows) for n, rows in enumerate(handed)], arrays)
