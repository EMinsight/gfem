"""Frequency workers: a uniform driven sweep spread over spawned processes.

`SolverOptions.frequency_workers=k` splits the frequencies into k
contiguous blocks, each solved by a frequency worker; the parent joins the
blocks back in frequency order. Every test compares with the sequential
sweep (``frequency_workers=1``) on a small parallel-plate line. Each
worker sweep pays the workers' start-up (seconds), so the tests are few.
"""

import logging
import multiprocessing
import os
import queue
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from gradient_cases import (
    core_design,
    hybrid_line,
    hybrid_mesh,
    ramp,
    s_functional,
    tight,
    wave_line,
    wave_mesh,
)
from recording_board import RecordingBoard
from solver_misses import cannot_converge_at

import gfem
from gfem import _progress
from gfem._engine import cache, driven, runtime, workers
from gfem._engine.driven import DrivenRule
from gfem._engine.driven_sweep import solve_block
from gfem._linalg import prepare
from gfem._linalg.direct import DirectHandle

DATA = Path(__file__).parent / "data"
FREQUENCIES = (3e9, 3.5e9, 4e9, 4.5e9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _plates(path=None):
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=path),
            gfem.WavePort("p2", "port2"),
        ],
    )


def test_two_workers_with_wave_ports_give_the_sequential_sweep():
    # the parent sweeps the port modes once over the grid and hands each
    # worker its rows; a kept frequency in each block
    # THz on this micrometre line: at GHz it sits near the low-frequency
    # floor, where a kept field keeps fewer digits than S
    path = ((20e-6, 0.0, 0.0), (20e-6, 10e-6, 0.0))
    model = _plates(path)
    grid = (1.5e12, 2.0e12, 2.5e12, 3.0e12)
    problem = gfem.Driven(grid, keep_fields=(grid[1], grid[3]))
    one = gfem.solve(
        model, problem, options=gfem.SolverOptions(order=1, linear_solver="direct")
    )
    two = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(
            order=1, linear_solver="direct", frequency_workers=2
        ),
    )
    np.testing.assert_allclose(two.s.values, one.s.values, rtol=0, atol=1e-12)
    # the workers solve on the parent's port modes exactly; the port mode
    # sweeps of two solves agree to rounding, bit for bit only on one thread
    np.testing.assert_allclose(two.s.z_ref, one.s.z_ref, rtol=1e-12)
    for name in ("p1", "p2"):
        mode, expected = two.port_modes[name], one.port_modes[name]
        np.testing.assert_allclose(mode.gamma, expected.gamma, rtol=1e-12)
        np.testing.assert_array_equal(mode.solved, expected.solved)
    assert two.kept == one.kept
    for f, port in one.kept:
        np.testing.assert_allclose(
            two.e_field(f, port).coefficients,
            one.e_field(f, port).coefficients,
            rtol=0,
            atol=1e-10 * np.abs(one.e_field(f, port).coefficients).max(),
        )
    mode = two.port_modes["p1"]
    z = gfem.characteristic_impedance(mode, path)
    np.testing.assert_allclose(z, mode.Z_PV, rtol=1e-13)


def test_one_frequency_with_wave_ports_runs_no_worker():
    model = _plates()
    options = gfem.SolverOptions(order=1, frequency_workers=2)
    s = gfem.solve(model, gfem.Driven(3e9), options=options).s.values
    ref = gfem.solve(model, gfem.Driven(3e9), options=gfem.SolverOptions(order=1))
    np.testing.assert_allclose(s, ref.s.values, rtol=0, atol=1e-9)


WAVE_GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 4))


def test_a_lossy_wave_port_beside_a_lumped_port_over_two_workers(tmp_path):
    # a block that swept its own port modes would move S by about 1e-3 here
    mesh = wave_mesh(tmp_path)
    model = wave_line(mesh, "mixed")
    values, _ = core_design(mesh, seed=4)

    def objective(eps, workers):
        core = gfem.MaterialField(**{**values, "eps_r": eps})
        m = model.with_material("core", core)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        options = tight("direct", frequency_workers=workers)
        s = gfem.solve(m, gfem.Driven(WAVE_GRID), options=options).s.values
        return s_functional(s), jax.lax.stop_gradient(s)

    eps = values["eps_r"]
    one, s_one = jax.grad(objective, has_aux=True)(eps, 1)
    two, s_two = jax.grad(objective, has_aux=True)(eps, 2)
    np.testing.assert_allclose(s_two, s_one, rtol=0, atol=1e-12)
    assert np.abs(np.asarray(one)).max() > 0
    np.testing.assert_allclose(two, one, rtol=0, atol=1e-9 * np.abs(one).max())


def test_a_face_design_gradient_over_two_workers_equals_one_process(tmp_path):
    # the lower layer touches both port faces, so the modes follow it: the
    # parent's backward rebuilds the whole-grid sweep and runs the port pass
    mesh = wave_mesh(tmp_path)
    model = wave_line(mesh, "lossy")
    n = len(mesh.elements("lo"))
    values = {"eps_r": ramp(mesh, "lo", 3.5, 4.5), "tan_delta": jnp.full(n, 0.05)}
    weights = jnp.array([1.0, -2.0, 0.5, 3.0])[:, None, None]

    def objective(eps, workers):
        field = gfem.MaterialField(**{**values, "eps_r": eps})
        m = model.with_material("lo", field)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        options = tight("direct", frequency_workers=workers)
        problem = gfem.Driven(WAVE_GRID, excite="p1")
        result = gfem.solve(m, problem, options=options)
        gamma = result.port_modes["p1"].gamma
        return jnp.sum(weights * jnp.abs(result.s.values) ** 2) + jnp.sum(
            jnp.imag(gamma * weights[:, 0, 0])
        )

    eps = values["eps_r"]
    one = jax.grad(objective)(eps, 1)
    two = jax.grad(objective)(eps, 2)
    assert np.abs(np.asarray(one)).max() > 0
    np.testing.assert_allclose(two, one, rtol=1e-10, atol=1e-14 * np.abs(one).max())


def test_workers_on_shared_port_surfaces_give_the_sequential_sweep(tmp_path):
    # two shared port surfaces, two workers: the same S as one process
    model = hybrid_line(hybrid_mesh(tmp_path))
    grid = (60e9, 70e9)
    options = gfem.SolverOptions(order=2, linear_solver="direct", frequency_workers=2)
    rule = DrivenRule(model, gfem.Driven(grid), options)
    two, _ = rule.forward(model, keep=False)
    one, _ = _forward(model, gfem.Driven(grid), order=2, linear_solver="direct")
    np.testing.assert_allclose(two["s"], one["s"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(two["port_gamma"], one["port_gamma"], rtol=1e-12)


def test_a_worker_does_not_repeat_the_parents_port_mode_warnings():
    # a lone mode=2 skips mode 1: the parent's rule warns, a worker's does not
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1", 2), gfem.WavePort("p2", "port2", 2)],
    )
    problem = gfem.Driven(2.5e12)
    options = gfem.SolverOptions(order=1)
    with pytest.warns(gfem.SkippedPortModeWarning):
        DrivenRule(model, problem, options)
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.SkippedPortModeWarning)
        outputs, _, reason, _ = solve_block(
            *workers.model_parts(model), problem, options, keep=False
        )
    assert reason is None
    assert outputs.rows == (0,)


def _forward(model, problem, *, keep=False, **options):
    rule = DrivenRule(model, problem, gfem.SolverOptions(**options))
    return rule.forward(model, keep=keep)


def _count_prepares(monkeypatch) -> list[int]:
    """Spy on this process's driven `prepare`: workers solve elsewhere."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr("gfem._linalg.prepare", spy)
    return calls


@pytest.mark.parametrize(
    "linear_solver", ["direct", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_two_exact_workers_give_the_sequential_sweep(line, monkeypatch, linear_solver):
    # a kept frequency in each block: the fields join in frequency order
    problem = gfem.Driven(FREQUENCIES, keep_fields=(3.5e9, 4.5e9))
    calls = _count_prepares(monkeypatch)
    cache.ENTRIES.clear()  # no pooled base of an earlier test serves
    one, one_entry = _forward(line, problem, keep=True, linear_solver=linear_solver)
    # one base: direct meets each frequency for the first time
    planned = 1
    assert len(calls) == planned
    two, two_entry = _forward(
        line, problem, keep=True, linear_solver=linear_solver, frequency_workers=2
    )
    assert len(calls) == planned  # the parent solved nothing
    np.testing.assert_allclose(two["s"], one["s"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(two["z_ref"], one["z_ref"])
    assert two["iterations"] == one["iterations"] == [()] * 4
    assert two["e_field"].shape == one["e_field"].shape == (one["e_field"].shape[0], 4)
    np.testing.assert_allclose(two["e_field"], one["e_field"], rtol=0, atol=1e-12)
    solution, sequential = two_entry.fields["solution"], one_entry.fields["solution"]
    assert solution.shape == sequential.shape
    np.testing.assert_allclose(solution, sequential, rtol=0, atol=1e-12)
    two_entry.free()
    one_entry.free()


def test_two_bddc_workers_agree_to_the_solver_tolerance(line, caplog):
    # a kept frequency in each block: the fields join in frequency order
    problem = gfem.Driven(FREQUENCIES, keep_fields=(3.5e9, 4.5e9))
    one, _ = _forward(line, problem, linear_solver="bddc_gmres")
    caplog.set_level(logging.INFO, logger="gfem.solver")
    two, _ = _forward(line, problem, linear_solver="bddc_gmres", frequency_workers=2)
    # each block sets BDDC up at its first frequency and reuses it at its second
    assert two["reused"] == [False, True, False, True]
    # the workers' records, logged here in frequency order
    logged = [r for r in caplog.records if hasattr(r, "direction")]
    assert [r.frequency for r in logged] == list(FREQUENCIES)
    assert [r.action for r in logged] == ["factorized", "reused"] * 2
    assert [r.iterations for r in logged] == two["iterations"]
    # each block's GMRES starts cold, so S moves within the tolerance
    np.testing.assert_allclose(two["s"], one["s"], rtol=0, atol=1e-6)
    assert len(two["iterations"]) == 4
    assert all(len(its) == 2 for its in two["iterations"])  # per excited port
    assert two["e_field"].shape == one["e_field"].shape == (one["e_field"].shape[0], 4)
    scale = np.abs(one["e_field"]).max()
    # the kept fields too, to the solver tolerance (about 3e-6 of the largest)
    np.testing.assert_allclose(
        two["e_field"], one["e_field"], rtol=0, atol=1e-5 * scale
    )


def test_the_workers_timing_records_are_logged_here_by_their_logger(line, caplog):
    # only gfem.timing is on: the workers' gfem.solver records stay silent
    caplog.set_level(logging.INFO, logger="gfem.timing")
    options = gfem.SolverOptions(linear_solver="direct", frequency_workers=2)
    gfem.solve(line, gfem.Driven(FREQUENCIES), options=options)
    assert not [r for r in caplog.records if r.name == "gfem.solver"]
    logged = [r for r in caplog.records if r.name == "gfem.timing"]
    solves = [r for r in logged if r.phase == "solve"]
    assert [r.frequency for r in solves] == list(FREQUENCIES)
    # each worker discretizes its own copy of the model
    discretized = [r for r in logged if r.phase == "discretize"]
    assert len(discretized) == 2
    assert len({r.ndof for r in discretized}) == 1
    for r in logged:
        assert r.problem == "driven"
        assert (r.rank, r.ranks) == (0, 1)


SIX = (3e9, 3.5e9, 4e9, 4.5e9, 5e9, 5.5e9)  # two blocks of three


def _in_process(monkeypatch):
    """Run each worker's block in this process, so test seams reach it."""

    def run(function, arguments, _labels, *, threads, keywords=None, row=None):  # noqa: ARG001
        keywords = keywords or [{}] * len(arguments)
        return [function(*a, **k) for a, k in zip(arguments, keywords, strict=True)]

    monkeypatch.setattr(workers, "run", run)


def test_a_worker_keeps_nothing_and_plans_once(line, monkeypatch):
    handed = []

    def run(function, arguments, _labels, *, threads, keywords=None, row=None):  # noqa: ARG001
        handed.extend(a[-2].retain_memory for a in arguments)
        keywords = keywords or [{}] * len(arguments)
        return [function(*a, **k) for a, k in zip(arguments, keywords, strict=True)]

    monkeypatch.setattr(workers, "run", run)
    calls = _count_factorizations(monkeypatch)
    cache.ENTRIES.clear()
    options = gfem.SolverOptions(linear_solver="direct", frequency_workers=2)
    gfem.solve(line, gfem.Driven(SIX), options=options)
    # each block of three frequencies refactorizes one base
    assert handed == [0, 0]
    assert len(calls) == 2
    assert cache.ENTRIES.pooled == 0


def _count_factorizations(monkeypatch) -> list[int]:
    """Spy on direct handles' new plans in this process."""
    calls = []
    factor = DirectHandle._factor

    def counted(self, *args, **kwargs):
        calls.append(1)
        return factor(self, *args, **kwargs)

    monkeypatch.setattr(DirectHandle, "_factor", counted)
    return calls


@pytest.mark.parametrize("fits", [True, False])
def test_workers_solve_with_the_parents_auto_choice(line, monkeypatch, fits):
    handed = []

    def run(function, arguments, _labels, *, threads, keywords=None, row=None):  # noqa: ARG001
        handed.extend(a[-2].linear_solver for a in arguments)
        keywords = keywords or [{}] * len(arguments)
        return [function(*a, **k) for a, k in zip(arguments, keywords, strict=True)]

    monkeypatch.setattr(workers, "run", run)
    if not fits:
        monkeypatch.setattr(driven, "default_retain_memory", lambda: 1)
    rule = DrivenRule(
        line, gfem.Driven(FREQUENCIES), gfem.SolverOptions(frequency_workers=2)
    )
    kind = "direct" if fits else "pmg_gmres"
    assert rule.linear.kind == kind
    outputs, _ = rule.forward(line, keep=False)
    assert handed == [kind, kind]
    direct = all(its == () for its in outputs["iterations"])
    assert direct == fits


@pytest.mark.parametrize(
    ("misses", "solved"),
    [
        pytest.param((4,), (0, 1, 2, 3), id="second-block"),
        pytest.param((1, 4), (0, 3), id="both-blocks"),
    ],
)
def test_each_worker_keeps_the_frequencies_it_solved_before_a_miss(
    line, monkeypatch, misses, solved
):
    _in_process(monkeypatch)
    options = gfem.SolverOptions(linear_solver="bddc_gmres", frequency_workers=2)
    problem = gfem.Driven(SIX, keep_fields=(SIX[0], SIX[3]))
    clean = gfem.solve(line, problem, options=options)
    cannot_converge_at(monkeypatch, [SIX[i] for i in misses])
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        gfem.solve(line, problem, options=options)
    err = caught.value
    assert err.frequency == SIX[misses[0]]
    assert err.unsolved == tuple(f for i, f in enumerate(SIX) if i not in solved)
    for i in misses:
        assert f"{SIX[i]:g} Hz" in str(err)
    assert err.result.s.frequency == tuple(SIX[i] for i in solved)
    np.testing.assert_allclose(
        err.result.s.values,
        np.asarray(clean.s.values)[list(solved)],
        rtol=0,
        atol=1e-12,
    )
    assert err.result.kept == (
        (SIX[0], "p1"),
        (SIX[0], "p2"),
        (SIX[3], "p1"),
        (SIX[3], "p2"),
    )


@pytest.mark.petsc
def test_workers_that_miss_at_their_first_frequencies_raise_one_error(line):
    # spawned workers: Jacobi with one GMRES iteration misses everywhere
    options = gfem.SolverOptions(
        linear_solver="petsc",
        petsc_options={"ksp_type": "gmres", "pc_type": "jacobi", "ksp_max_it": 1},
        frequency_workers=2,
    )
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        gfem.solve(line, gfem.Driven(FREQUENCIES), options=options)
    assert caught.value.result is None
    assert caught.value.unsolved == FREQUENCIES
    assert caught.value.frequency == FREQUENCIES[0]


def test_a_killed_worker_names_its_frequencies():
    with pytest.raises(
        RuntimeError, match=r"4e\+09 to 4.5e\+09 Hz.*memory.*loading its inputs.*fewer"
    ):
        workers.run(os._exit, [(1,)], ["4e+09 to 4.5e+09 Hz"], threads=1)


@pytest.mark.skipif(
    not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)"
)
def test_a_worker_starts_with_its_share_of_threads(monkeypatch):
    # NumPy and the BLAS read their thread variables when imported, before a
    # pool initializer would run: the worker process must start with them
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    (environ,) = workers.run(
        Path.read_bytes, [(Path("/proc/self/environ"),)], ["a probe"], threads=3
    )
    start = dict(v.split(b"=", 1) for v in environ.split(b"\0") if b"=" in v)
    for name in (b"OMP_NUM_THREADS", b"MKL_NUM_THREADS", b"OPENBLAS_NUM_THREADS"):
        assert start[name] == b"3"
    # the parent's own environment is left as it was
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "MKL_NUM_THREADS" not in os.environ


# ---------------------------------------------------------------- progress


def test_a_worker_forwards_its_rows_advances_and_not_its_phases(monkeypatch):
    monkeypatch.setattr(_progress, "_LIVE", None)
    forwarded = queue.Queue()
    _progress.forward_to(forwarded)
    with _progress.task("uniform sweep", total=2) as row:
        with _progress.phase("solve"):
            row.advance(frequency=3e9, iterations=(4,))
        row.advance(frequency=3.5e9)
    items = []
    while not forwarded.empty():
        items.append(forwarded.get_nowait())
    assert items == [
        (1, {"frequency": 3e9, "iterations": (4,)}),
        (1, {"frequency": 3.5e9}),
    ]


def test_a_worker_forwards_only_its_uniform_sweep_rows_advances(monkeypatch):
    # the parent advances its uniform sweep row by what it gets: a row of
    # another loop run in the worker would count units of another kind
    monkeypatch.setattr(_progress, "_LIVE", None)
    forwarded = queue.Queue()
    _progress.forward_to(forwarded)
    with _progress.task("mode search") as other:
        other.advance(attempt=1)
        with _progress.task("uniform sweep", total=1) as row:
            row.advance(frequency=3e9)
        other.advance(attempt=2)
    assert forwarded.get_nowait() == (1, {"frequency": 3e9})
    assert forwarded.empty()


def test_two_workers_advance_the_parents_row_once_per_frequency(line, monkeypatch):
    problem = gfem.Driven(FREQUENCIES)
    # off: no queue is made and the pools start without an initializer
    context = multiprocessing.get_context("spawn")
    queues = []
    make_queue = context.Queue

    def spy_queue(*args, **kwargs):
        queues.append(1)
        return make_queue(*args, **kwargs)

    monkeypatch.setattr(context, "Queue", spy_queue)
    initializers = []
    pool = workers.ProcessPoolExecutor

    def spy_pool(*args, **kwargs):
        initializers.append(kwargs.get("initializer"))
        return pool(*args, **kwargs)

    monkeypatch.setattr(workers, "ProcessPoolExecutor", spy_pool)
    off, _ = _forward(line, problem, linear_solver="direct", frequency_workers=2)
    assert queues == []
    assert initializers == [None, None]
    board = RecordingBoard()
    monkeypatch.setattr(_progress, "_LIVE", _progress.Live(board))
    on, _ = _forward(line, problem, linear_solver="direct", frequency_workers=2)
    assert board.summary() == [("uniform sweep", 4, 4, True)]
    frequencies = sorted(e[3]["frequency"] for e in board.events if e[0] == "advance")
    assert frequencies == list(FREQUENCIES)
    # threaded factorizations are not bitwise reproducible between two runs, so
    # the progress feed is held to the same tolerance as the other worker checks
    np.testing.assert_allclose(on["s"], off["s"], rtol=0, atol=1e-12)


def test_with_progress_the_first_blocks_error_is_raised(monkeypatch):
    board = RecordingBoard()
    monkeypatch.setattr(_progress, "_LIVE", _progress.Live(board))
    # the second block fails first; the error raised is still the first block's
    arguments = [
        ("import time; time.sleep(1); raise ValueError('first')",),
        ("raise KeyError('second')",),
    ]
    # the queue the workers forward through is closed even so
    context = multiprocessing.get_context("spawn")
    made = []
    make_queue = context.Queue

    def spy_queue(*args, **kwargs):
        made.append(make_queue(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(context, "Queue", spy_queue)
    with (
        _progress.task("uniform sweep", total=2) as row,
        pytest.raises(ValueError, match="first"),
    ):
        workers.run(exec, arguments, ["block 0", "block 1"], threads=1, row=row)
    assert board.summary() == [("uniform sweep", 2, 0, True)]
    (forwarded,) = made
    assert forwarded._closed


def test_worker_threads_split_the_cores_this_process_may_use(monkeypatch):
    # an affinity mask (taskset, a container's cpuset) limits the CPUs; a CPU with no
    # topology directory is a core of its own
    monkeypatch.delenv("NGS_NUM_THREADS", raising=False)
    monkeypatch.delenv("PYTHON_CPU_COUNT", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    mask = {100000 + cpu for cpu in range(4)}
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: mask, raising=False)
    assert workers.worker_threads(None, 2) == 2
    assert workers.worker_threads(None, 8) == 1
    assert workers.worker_threads(12, 4) == 3  # given threads win


def test_worker_threads_count_the_cores_of_a_mask_holding_both_siblings(monkeypatch):
    monkeypatch.delenv("NGS_NUM_THREADS", raising=False)
    monkeypatch.delenv("PYTHON_CPU_COUNT", raising=False)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    mask = set(range(16))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: mask, raising=False)
    read_text = Path.read_text

    def siblings(path, *args, **kwargs):
        if path.name == "thread_siblings_list":
            return str(int(path.parent.parent.name.removeprefix("cpu")) // 2)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", siblings)
    assert workers.worker_threads(None, 2) == 4 == runtime.solve_threads(None) // 2


def test_worker_threads_split_ngs_num_threads_when_it_is_set(monkeypatch):
    monkeypatch.delenv("PYTHON_CPU_COUNT", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setenv("NGS_NUM_THREADS", "8")
    assert workers.worker_threads(None, 2) == 4 == runtime.solve_threads(None) // 2
    assert workers.worker_threads(None, 16) == 1
    monkeypatch.setenv("NGS_NUM_THREADS", "abc")  # falls back to the mask, as a solve
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    mask = {100000 + cpu for cpu in range(6)}
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: mask, raising=False)
    assert workers.worker_threads(None, 2) == 3


def test_a_gradient_through_two_workers_matches_one(line):
    # p2 is not excited, so its S row takes an adjoint solve in the parent;
    # weights that differ per frequency catch blocks joined out of order
    n = len(line.mesh.tetrahedra)
    eps = jnp.asarray(np.random.default_rng(5).uniform(2.0, 5.0, n))
    weights = jnp.array([1.0, -2.0, 0.5, 3.0])[:, None, None]

    def objective(eps, workers):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        options = gfem.SolverOptions(linear_solver="direct", frequency_workers=workers)
        problem = gfem.Driven(FREQUENCIES, excite="p1")
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(weights * jnp.abs(s) ** 2)

    one = jax.grad(objective)(eps, 1)
    two = jax.grad(objective)(eps, 2)
    assert np.abs(np.asarray(one)).max() > 0
    np.testing.assert_allclose(two, one, rtol=1e-10, atol=1e-14)


def test_a_backward_that_rebuilds_port_state_walks_it_down(line):
    # the workers solve in their own processes: here the entry holds the
    # solutions only, and the backward builds the system and the port state
    eps = jnp.full(len(line.mesh.tetrahedra), 3.0)
    problem = gfem.Driven(FREQUENCIES, excite="p1")

    def objective(retain_memory):
        options = gfem.SolverOptions(
            linear_solver="direct", frequency_workers=2, retain_memory=retain_memory
        )

        def s(eps):
            model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
            return jnp.sum(
                jnp.abs(gfem.solve(model, problem, options=options).s.values)
            )

        return s

    cache.ENTRIES.clear()
    cache._DISCRETIZATIONS.clear()
    _, vjp = jax.vjp(objective(None), eps)
    (entry,) = cache.ENTRIES._entries.values()
    budget = entry.nbytes  # the entry and nothing the backward adds
    vjp(1.0)
    assert any(d.ports is not None for d in cache._DISCRETIZATIONS.values())
    cache.ENTRIES.clear()
    cache._DISCRETIZATIONS.clear()
    _, vjp = jax.vjp(objective(budget), eps)
    (entry,) = cache.ENTRIES._entries.values()
    vjp(1.0)
    assert entry.handle is not None  # the backward built its system here
    assert all(d.ports is None for d in cache._DISCRETIZATIONS.values())
    cache.ENTRIES.clear()
