"""`benchmarks.adapters.gfem`: plan answers, the rendered run, the worker end to end.

The pure parts read only the case files. The end-to-end tests run the worker as the
harness's ``run --local`` path does (one rank, the bare program pinned by the caller)
on the committed tiny parallel-plate case; they check status, DOF count, phases and
quantity shapes, never timings.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import adapters, stacks
from benchmarks.adapters import CommandResult, Layout, Measured, Run, RunRefusedError
from benchmarks.adapters import gfem as gfem_adapter
from benchmarks.harness import config, data
from benchmarks.harness.case import load_case

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "benchmarks" / "cases"
TINY = Path(__file__).parent / "data" / "bench" / "tiny"
TINY_NDOF = 938  # order 2 on the 6 kB parallel plate, one process
COMMIT = "858bc66" + "0" * 33
SITE = """
name = "test"
[slurm]
account = "A"
[slurm.qos]
debug = "d"
[cluster]
root = "{root}"
[stacks.wheels]
base_venv = "{root}/venv"
wheel_dirs = ["{root}/wheels"]
petsc_wheelhouse = "{root}/petsc"
palace_wheelhouse = "{root}/palace"
python = "{root}/python"
[data]
dir = "{root}/data"
[output]
dir = "{root}/out"
"""
ADAPTER = gfem_adapter.ADAPTER


@pytest.fixture
def stack(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    (root / "site.toml").write_text(SITE.replace("{root}", str(root)))
    site = config.load_site(root / "site.toml")
    return stacks.resolve("wheels", site, commit=COMMIT)


def _taskset(stack):
    """The stack with the one-rank taskset fallback (as the stack check may set)."""
    description = stack.description
    codes = {
        **description.codes,
        "gfem": replace(description.codes["gfem"], launch_r1="taskset"),
    }
    return replace(stack, description=replace(description, codes=codes))


def _run(case, solver, stack, workdir, *, layout="1x16", tolerance=None, **kw):
    case = load_case(case if isinstance(case, Path) else CASES / case)
    lay = Layout.parse(layout)
    return Run(
        case=case,
        solver=solver,
        tolerance=tolerance,
        layout=lay,
        core_budget=lay.cores,
        repeat=1,
        stack=stack,
        workdir=workdir,
        mesh=kw.pop("mesh", Path("/staged/mesh.msh")),
        **kw,
    )


# --- plan answers ------------------------------------------------------------------


def test_gfem_is_registered():
    assert adapters.ADAPTERS["gfem"] is ADAPTER
    assert ADAPTER.code == "gfem"
    assert {k: v.klass for k, v in ADAPTER.kinds.items()} == {
        "direct": "direct",
        "petsc-mumps": "direct",
        "bddc": "iterative",
    }


@pytest.mark.parametrize(
    ("case", "kind", "layout", "refused"),
    [
        ("microstrip", "bddc", "4x4", None),
        ("microstrip", "petsc-mumps", "16x1", None),
        ("microstrip", "direct", "1x16", None),
        ("microstrip", "direct", "2x8", "one process"),
        ("lcavity", "bddc", "1x16", None),
        ("lcavity", "bddc", "2x8", "eigen"),
        ("bragg", "petsc-mumps", "2x8", "eigen"),
        ("bragg", "direct", "1x16", None),
        ("r10M_void", "bddc", "28x4", None),
        ("r10M_void", "bddc", "1x224", "one node"),
        ("microstrip", "mumps", "1x16", "no kind"),
    ],
)
def test_supports_refuses_what_gfem_cannot_run(case, kind, layout, refused):
    reason = ADAPTER.supports(load_case(CASES / case), kind, Layout.parse(layout))
    if refused is None:
        assert reason is None
    else:
        assert reason is not None
        assert refused in reason


def test_supports_refuses_an_adaptive_sweep_and_order_1_bddc_above_one_rank():
    case = load_case(CASES / "microstrip")
    adaptive = replace(case, problem=replace(case.problem, sweep="adaptive"))
    assert "adaptive" in ADAPTER.supports(adaptive, "bddc", Layout(4, 4))
    assert ADAPTER.supports(adaptive, "bddc", Layout(1, 16)) is None
    order1 = replace(case, order=1)
    assert "order 1" in ADAPTER.supports(order1, "bddc", Layout(4, 4))
    assert ADAPTER.supports(order1, "petsc-mumps", Layout(4, 4)) is None


@pytest.mark.parametrize(
    ("case", "kind", "budget", "layout"),
    [
        ("microstrip", "bddc", 16, "1x16"),
        ("microstrip", "direct", 112, "1x112"),
        ("lcavity", "bddc", 112, "1x112"),
        ("r10M_void", "bddc", 224, "56x4"),
        ("r10M_void", "petsc-mumps", 448, "112x4"),
        ("r10M_void", "direct", 224, "1x224"),
        ("bragg", "bddc", 224, "1x224"),
    ],
)
def test_default_layout_is_one_process_on_one_node(case, kind, budget, layout):
    got = ADAPTER.default_layout(load_case(CASES / case), kind, budget)
    assert str(got) == layout


# --- prepare -------------------------------------------------------------------------


def test_prepare_writes_run_json_with_the_kind_pins(stack, tmp_path):
    run = _run("microstrip", "gfem:bddc", stack, tmp_path, tolerance=1e-8)
    run = replace(run, pinned_ndof=315870)
    launch = ADAPTER.prepare(run)
    rendered = json.loads((tmp_path / "run.json").read_text())
    assert rendered["mesh"] == "/staged/mesh.msh"
    assert rendered["solver"] == "gfem:bddc"
    assert rendered["options"] == {
        "order": 2,
        "linear_solver": "bddc_gmres",
        "tolerance": 1e-8,
        "eigen_tolerance": 1e-10,
    }
    assert rendered["layout"] == {"ranks": 1, "threads": 16}
    assert rendered["pinned_ndof"] == 315870
    assert load_case(CASES / "microstrip").to_dict() == rendered["case"]
    (command,) = launch.commands
    assert command.program == (
        "python",
        "-m",
        "benchmarks.adapters.gfem_worker",
        "run.json",
    )
    assert command.launch == "srun"
    assert command.argv[: command.argv.index("python")] == (
        *adapters.srun_argv(Layout(1, 16), "none"),
    )


def test_direct_kinds_carry_no_tolerance(stack, tmp_path):
    ADAPTER.prepare(_run("lcavity", "gfem:petsc-mumps", stack, tmp_path))
    options = json.loads((tmp_path / "run.json").read_text())["options"]
    assert options == {"order": 2, "linear_solver": "petsc", "eigen_tolerance": 1e-10}


def test_ranks_use_the_stack_mpi_plugin_and_one_rank_may_fall_back_to_taskset(
    stack, tmp_path
):
    launch = ADAPTER.prepare(
        _run("microstrip", "gfem:bddc", stack, tmp_path, layout="4x4", tolerance=1e-8)
    )
    assert "--mpi=pmix" in launch.commands[0].argv
    assert launch.commands[0].argv[2:4] == ("-n", "4")
    launch = ADAPTER.prepare(
        _run("microstrip", "gfem:direct", _taskset(stack), tmp_path)
    )
    (command,) = launch.commands
    assert command.launch == "taskset"
    assert command.argv == command.program


def test_prepare_refuses_an_unsupported_run(stack, tmp_path):
    with pytest.raises(RunRefusedError, match="eigen"):
        ADAPTER.prepare(_run("lcavity", "gfem:bddc", stack, tmp_path, layout="4x4"))


def test_version_query_is_the_bare_worker(stack):
    (command,) = ADAPTER.version(stack).commands
    assert command.launch == "taskset"
    assert command.program[-1] == "--version"


# --- collect on written outputs -------------------------------------------------------


def _record(phase, seconds, rank, ranks, **attrs):
    return {
        "phase": phase,
        "seconds": seconds,
        "problem": "driven",
        "frequency": 1e9,
        "rank": rank,
        "ranks": ranks,
        **attrs,
    }


def _two_ranks(workdir, *, ndof_rank1=500):
    """Outputs of a two-rank run: rank 0 owns no DOFs; rank 1 logs two discretizes."""
    workdir.mkdir(parents=True, exist_ok=True)
    rank0 = [
        _record("discretize", 1.0, 0, 2, ndof=0),
        _record("assemble", 2.0, 0, 2),
        _record(
            "setup", 3.0, 0, 2, solver="bddc_gmres", action="factorized", nbytes=10
        ),
        _record("solve", 0.5, 0, 2, iterations=(7, 8)),
    ]
    rank1 = [
        _record("discretize", 1.5, 1, 2, ndof=ndof_rank1),
        _record("discretize", 0.5, 1, 2, ndof=ndof_rank1),
        _record("assemble", 1.0, 1, 2),
        _record(
            "setup", 4.0, 1, 2, solver="bddc_gmres", action="factorized", nbytes=30
        ),
        _record("solve", 0.5, 1, 2, iterations=(7, 8)),
    ]
    for rank, records in enumerate((rank0, rank1)):
        lines = "".join(json.dumps(r) + "\n" for r in records)
        (workdir / f"phases.rank{rank}.jsonl").write_text(lines)
    outcome = {
        "status": "ok",
        "quantities": {
            "frequencies_hz": [1e9],
            "s": {"p1,p1": [[0.1, 0.0]], "p2,p1": [[0.9, 0.1]]},
        },
        "ranks": [
            {
                "rank": 0,
                "mesh_read_s": 0.2,
                "span_s": 9.0,
                "vmhwm_kib": 2**20,
                "host": "n1",
            },
            {
                "rank": 1,
                "mesh_read_s": 0.3,
                "span_s": 9.5,
                "vmhwm_kib": 3 * 2**20,
                "host": "n1",
            },
        ],
        "threads": [4, 4],
        "versions": {"gfem": "0.1.0", "ngsolve": "6.2.2608"},
    }
    (workdir / "outcome.json").write_text(json.dumps(outcome))
    (workdir / "stdout.txt").write_text("")


def _r10m(stack, workdir, **kw):
    return _run(
        "r10M_void", "gfem:bddc", stack, workdir, layout="2x2", tolerance=1e-8, **kw
    )


def _measured(workdir, code=0):
    return Measured(results=(CommandResult(code, workdir / "stdout.txt", 12.0),))


def test_ndof_is_one_discretize_per_rank_summed_over_ranks(stack, tmp_path):
    _two_ranks(tmp_path)
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path))
    assert outcome.status == "ok"
    assert outcome.ndof == 500
    assert outcome.ranks_seen == 2
    assert outcome.threads_seen == 4


def test_phases_are_max_over_ranks_with_rank_0_beside(stack, tmp_path):
    _two_ranks(tmp_path)
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path))
    assert outcome.rank_reduction == "max"
    assert outcome.phases["discretize_s"] == pytest.approx(2.0)
    assert outcome.phases["setup_s"] == pytest.approx(4.0)
    assert outcome.phases["mesh_read_s"] == pytest.approx(0.3)
    assert outcome.rank0["assemble_s"] == pytest.approx(2.0)
    assert outcome.rank0["other_s"] == pytest.approx(9.0 - 0.2 - 1.0 - 2.0 - 3.0 - 0.5)
    for bucket in ("estimate_s", "postpro_s", "rom_s"):
        assert outcome.phases[bucket] is None
    assert outcome.work_span_s == pytest.approx(9.5)
    assert outcome.setups == 1
    assert outcome.solves == 1
    assert outcome.iterations == [15]
    assert outcome.self_peak_mem_gib == pytest.approx(4.0)
    assert outcome.solver_nbytes == 40


def test_peak_memory_is_summed_with_the_rank_max_and_largest_node_beside(
    stack, tmp_path
):
    _two_ranks(tmp_path)
    written = json.loads((tmp_path / "outcome.json").read_text())
    written["ranks"][1]["host"] = "n2"
    (tmp_path / "outcome.json").write_text(json.dumps(written))
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path))
    assert outcome.self_peak_mem_gib == pytest.approx(4.0)
    assert outcome.self_peak_rank_max_gib == pytest.approx(3.0)
    assert outcome.self_peak_largest_node_gib == pytest.approx(3.0)


def test_ranks_on_one_node_sum_into_its_peak(stack, tmp_path):
    _two_ranks(tmp_path)
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path))
    assert outcome.self_peak_largest_node_gib == pytest.approx(4.0)


def test_bddc_structural_note_says_the_preconditioner_build_is_in_assemble(
    stack, tmp_path
):
    _two_ranks(tmp_path)
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path))
    assert "assemble" in outcome.structural["bddc_build"]
    assert outcome.structural["excitations"] == 1


def test_summed_ndof_above_one_rank_is_the_dof_guard(stack, tmp_path):
    _two_ranks(tmp_path, ndof_rank1=499)
    run = _r10m(stack, tmp_path, pinned_ndof=500)
    outcome = ADAPTER.collect(run, _measured(tmp_path))
    assert outcome.status == "failed"
    assert outcome.reason == "dof_mismatch: gfem 0.1.0 499 != pinned 500"
    assert outcome.checks["dof"] == "fail"


def test_a_worker_that_left_no_outcome_failed(stack, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "stdout.txt").write_text("Killed\n")
    outcome = ADAPTER.collect(_r10m(stack, tmp_path), _measured(tmp_path, -9))
    assert outcome.status == "failed"
    assert "exit -9" in outcome.reason
    assert outcome.checks["complete"] == "fail"
    assert all(v is None for v in outcome.phases.values())


# --- the worker, end to end ---------------------------------------------------------


def _local(stack, tmp_path, name, **kw):
    """Prepare and run the tiny case as ``run --local`` does at one rank."""
    workdir = tmp_path / name
    workdir.mkdir()
    case = load_case(TINY)
    mesh = data.stage(case, tmp_path / "data", ())
    run = _run(TINY, "gfem:direct", _taskset(stack), workdir, layout="1x1", mesh=mesh)
    run = replace(run, **kw)
    (command,) = ADAPTER.prepare(run).commands
    env = {**os.environ, **command.env, "OMP_NUM_THREADS": "1", "NGS_NUM_THREADS": "1"}
    env["MKL_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (command.env["PYTHONPATH"], os.environ.get("PYTHONPATH")) if p
    )
    program = (sys.executable, *command.program[1:])
    stdout = workdir / command.stdout
    with stdout.open("w") as out:
        code = subprocess.run(
            program,
            check=False,
            cwd=workdir,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode
    measured = Measured(results=(CommandResult(code, stdout, None),))
    return run, ADAPTER.collect(run, measured)


def test_the_worker_solves_the_tiny_case(stack, tmp_path):
    run, outcome = _local(stack, tmp_path, "ok", pinned_ndof=TINY_NDOF)
    assert outcome.status == "ok", (run.workdir / "stdout.txt").read_text()
    assert outcome.ndof == TINY_NDOF
    assert outcome.checks == {"dof": "pass", "converged": "pass", "complete": "pass"}
    assert outcome.ranks_seen == 1
    assert outcome.threads_seen == 1
    for bucket in ("mesh_read_s", "discretize_s", "assemble_s", "setup_s", "solve_s"):
        assert outcome.phases[bucket] is not None
        assert outcome.rank0[bucket] == outcome.phases[bucket]
    assert outcome.setups == 2
    assert outcome.solves == 2
    assert outcome.solver_nbytes > 0
    assert outcome.self_peak_mem_gib > 0
    quantities = outcome.quantities
    assert quantities["frequencies_hz"] == [1e12, 1.5e12]
    assert set(quantities["s"]) == {"p1,p1", "p2,p1", "p1,p2", "p2,p2"}
    for values in quantities["s"].values():
        assert len(values) == 2
        assert all(len(z) == 2 for z in values)
    assert abs(complex(*quantities["s"]["p2,p1"][0])) > 0.9
    assert "job.json" in outcome.files
    assert (run.workdir / "job.json").exists()


def test_the_worker_stops_at_a_dof_count_off_the_pin(stack, tmp_path):
    _, outcome = _local(stack, tmp_path, "dof", pinned_ndof=TINY_NDOF + 1)
    assert outcome.status == "failed"
    assert outcome.reason.startswith("dof_mismatch: gfem ")
    assert outcome.reason.endswith(f" {TINY_NDOF} != pinned {TINY_NDOF + 1}")
    assert outcome.checks["dof"] == "fail"
    assert outcome.phases["assemble_s"] is None
