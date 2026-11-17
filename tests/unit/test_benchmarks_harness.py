"""`benchmarks.harness.{measure,slurm,budget,ship}`: launch, measure, Slurm and ship.

Pure parts run on synthetic ``/proc``, ``/sys`` and sacct text; the job-level parts on
planned run directories of the tiny fixture campaign with hand-written records. One
local end-to-end test runs the tiny case through ``run --local`` (one gfem worker).
No test talks to Slurm or a cluster.
"""

import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import stacks
from benchmarks.__main__ import main
from benchmarks.adapters import Layout, gfem, palace
from benchmarks.adapters import Run as AdapterRun
from benchmarks.harness import budget, config, matrix, measure, records, ship, slurm
from benchmarks.harness.case import load_case

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "unit" / "data" / "bench"
TINY_CAMPAIGN = DATA / "tiny" / "campaign.toml"
COMMIT = "858bc66" + "0" * 33
GIT = shutil.which("git") or "git"
SITE = """
name = "test-cluster"
[slurm]
account = "acct42"
constraint = "fast"
[slurm.qos]
debug = "q_debug"
long = "q_long"
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
[budget]
ceiling_core_h = 100
prior_core_h = 40
"""


def _done(stdout: str = "", code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    path = root / "site.toml"
    path.write_text(SITE.replace("{root}", str(root)))
    real = root / "envs/palace-0.19.0/lib/python3.12/site-packages/palace_solver/bin"
    real.mkdir(parents=True)
    (real / "palace-real").touch()
    return config.load_site(path, environ={})


def _plan(site, tmp_path, *, solvers=None, repeats=None, name="run"):
    campaign = matrix.override(
        config.load_campaign(TINY_CAMPAIGN), solvers=solvers, repeats=repeats
    )
    plan = matrix.expand(campaign, site, commit=matrix.source_commit()[0])
    rundir = tmp_path / name
    matrix.write(plan, rundir, site=site, commit_utc="2026-11-17T00:00:00Z")
    slurm.copy_site(site, rundir)
    return slurm.load_rundir(rundir)


def _ctx(**kw):
    kw.setdefault("environ", {"USER": "someone"})
    return slurm.Context(local=kw.pop("local", False), job_id=kw.pop("job", "77"), **kw)


def _record(rd, row, status="ok", **fields):
    record = slurm._planned(rd, row, _ctx())
    record.status = status
    for key, value in fields.items():
        if key == "ndof":
            record.case.ndof = value
        elif key == "converged":
            record.checks["converged"] = value
        else:
            setattr(record, key, value)
    slurm.write(rd, record, {"USER": "someone"})
    return record


def _status(rd, name):
    return records.load_record(rd.path / "records" / f"{name}.json")


# --- launch lines ---------------------------------------------------------------------


def _resolved(site):
    return stacks.resolve("wheels", site, commit=COMMIT)


def _adapter_run(site, tmp_path, solver, layout, *, tolerance=None):
    lay = Layout.parse(layout)
    return AdapterRun(
        case=load_case(REPO / "benchmarks" / "cases" / "microstrip"),
        solver=solver,
        tolerance=tolerance,
        layout=lay,
        core_budget=lay.cores,
        repeat=1,
        stack=_resolved(site),
        workdir=tmp_path / "w",
        mesh=Path("/staged/mesh.msh"),
    )


SRUN = (
    "srun",
    "--exact",
    "-n",
    "{R}",
    "-c",
    "{T}",
    "--cpu-bind=cores",
    "--distribution=block:block",
    "--hint=nomultithread",
    "--ear=off",
    "--mpi={mpi}",
)


@pytest.mark.parametrize(
    ("solver", "layout", "mpi"),
    [
        ("gfem:direct", "1x16", "none"),
        ("gfem:bddc", "4x4", "pmix"),
        ("palace:mumps", "16x1", "pmi2"),
        ("palace:pmg", "1x16", "pmi2"),
    ],
)
def test_every_launch_is_the_one_srun_line(site, tmp_path, solver, layout, mpi):
    code = solver.partition(":")[0]
    adapter = {"gfem": gfem.ADAPTER, "palace": palace.ADAPTER}[code]
    tolerance = 1e-8 if solver.endswith(("bddc", "pmg")) else None
    run = _adapter_run(site, tmp_path, solver, layout, tolerance=tolerance)
    (command,) = adapter.prepare(run).commands
    lay = Layout.parse(layout)
    expected = tuple(s.format(R=lay.ranks, T=lay.threads, mpi=mpi) for s in SRUN)
    assert command.argv[: len(expected)] == expected
    assert command.launch == "srun"


def test_the_taskset_fallback_pins_the_bare_program_to_a_block(site, tmp_path):
    run = _adapter_run(site, tmp_path, "gfem:direct", "1x4")
    command = gfem.ADAPTER.prepare(run).commands[0]
    command = type(command)(
        argv=command.program, program=command.program, launch="taskset"
    )
    argv, launch = slurm._argv(command, run, _ctx())
    assert launch == "taskset"
    assert argv[:2] == ("taskset", "-c")
    assert argv[3:] == command.program
    assert len(measure.parse_cpu_list(argv[2])) == 4


def test_local_runs_wrap_gfem_in_taskset_and_refuse_ranks(site, tmp_path):
    run = _adapter_run(site, tmp_path, "gfem:direct", "1x1")
    (command,) = gfem.ADAPTER.prepare(run).commands
    argv, launch = slurm._argv(command, run, _ctx(local=True))
    assert launch == "taskset"
    assert argv[3] == sys.executable
    wide = _adapter_run(site, tmp_path, "gfem:bddc", "2x1", tolerance=1e-8)
    (command,) = gfem.ADAPTER.prepare(wide).commands
    with pytest.raises(slurm.RunRefusedError, match="one rank"):
        slurm._argv(command, wide, _ctx(local=True))


def test_hydra_is_bound_under_taskset():
    argv = measure.hydra_argv(
        ("palace-real", "c.json"), Layout(4, 1), [0, 1, 2, 3], "mx"
    )
    assert argv == (
        "taskset",
        "-c",
        "0-3",
        "mx",
        "-n",
        "4",
        "-launcher",
        "fork",
        "-bind-to",
        "core",
        "palace-real",
        "c.json",
    )


# --- env ------------------------------------------------------------------------------


def test_affinity_variables_are_refused_in_any_env_table(tmp_path):
    text = TINY_CAMPAIGN.read_text().replace(
        "[stacks.wheels]", '[stacks.wheels]\nenv = { OMP_PLACES = "threads" }'
    )
    path = tmp_path / "c.toml"
    path.write_text(text.replace('cases_dir = ".."', f'cases_dir = "{DATA}"'))
    with pytest.raises(config.ConfigError, match="OMP_PLACES"):
        config.load_campaign(path)


def test_the_local_env_drops_the_callers_threads_and_affinity(site):
    env = slurm.local_env(
        _resolved(site),
        "gfem",
        4,
        {"OMP_PROC_BIND": "spread", "KMP_AFFINITY": "x", "MKL_NUM_THREADS": "9"},
    )
    assert env["OMP_PROC_BIND"] == "close"
    assert env["OMP_PLACES"] == "cores"
    assert "KMP_AFFINITY" not in env
    assert env["MKL_NUM_THREADS"] == env["NGS_NUM_THREADS"] == "4"


def test_the_env_script_is_sourced_and_the_command_env_merged_over_it(tmp_path):
    script = tmp_path / "env.sh"
    script.write_text(
        "module purge >/dev/null 2>&1\nunset OMP_PLACES\nexport A=1 PYTHONPATH=/env\n"
    )
    sourced = measure.sourced_env(
        script, {"PATH": os.environ["PATH"], "OMP_PLACES": "x"}
    )
    assert sourced["A"] == "1"
    assert "OMP_PLACES" not in sourced
    merged = measure.command_env(sourced, {"PYTHONPATH": "/tree", "B": "2"})
    assert merged["PYTHONPATH"] == os.pathsep.join(["/tree", "/env"])
    assert merged["B"] == "2"


# --- placement ------------------------------------------------------------------------


@pytest.fixture
def sysfs(tmp_path):
    """Two-way SMT like an MN5 GPP node: CPU 112-223 are siblings of 0-111."""
    root = tmp_path / "cpu"
    for cpu in range(224):
        core = cpu % 112
        topo = root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "thread_siblings_list").write_text(f"{core},{core + 112}\n")
    return root


def test_a_sibling_hyperthread_maps_to_its_physical_core(sysfs):
    assert measure.core_of(112, sysfs) == 0
    assert measure.core_of(223, sysfs) == 111
    assert measure.core_of(5, sysfs) == 5


@pytest.mark.parametrize(
    ("masks", "threads", "status"),
    [
        (["0-3", "4-7", "8-11", "12-15"], 4, "pass"),
        (["0,112", "1,113", "2,114", "3,115"], 1, "pass"),  # Hydra -bind-to core
        (["0-3", "0-3", "0-3", "0-3"], 1, "fail"),  # probe 13: one shared mask
        (["0-1", "1-2"], 2, "fail"),
        (["0-15"], 16, "pass"),
        (["0,112"], 2, "fail"),  # two hyperthreads of one core are one core
    ],
)
def test_placement_wants_t_physical_cores_per_rank_disjoint(
    sysfs, masks, threads, status
):
    assert measure.check_placement(masks, threads, sysfs).status == status


def test_the_shared_mask_failure_names_the_ranks(sysfs):
    found = measure.check_placement(["0-3"] * 4, 1, sysfs)
    assert "rank 0 mask 0-3 holds 4 cores" in found.reason
    assert "rank 1 shares core 0 with rank 0" in found.reason


def test_busy_threads_count_hyperthreads_on_their_core(sysfs):
    ps = """    PID     TID PSR %CPU COMMAND
    100     100   0 99.0 palace-real
    100     101 112 98.0 palace-real
    100     102   1  0.1 palace-real
    200     200   0 99.0 other
"""
    assert measure.busy_threads_per_core(ps, [100], sysfs) == 2


def test_listpids_takes_the_newest_step_and_its_tasks_in_order():
    text = """PID      JOBID    STEPID   LOCALID GLOBALID
1000     77       batch    0       0
2000     77       3        -       -
2001     77       3        1       1
2002     77       3        0       0
1500     77       2        0       0
"""
    found = measure.listpids(text)
    assert found.step == "3"
    assert found.ranks == (2002, 2001)
    assert set(found.all) == {2000, 2001, 2002}


def test_pss_and_masks_come_from_proc(tmp_path):
    pid = tmp_path / "proc" / "42"
    pid.mkdir(parents=True)
    (pid / "smaps_rollup").write_text("Rss:  10 kB\nPss:  2048 kB\n")
    (pid / "status").write_text("Name:\tx\nCpus_allowed_list:\t0-3\n")
    assert measure.pss_kib(42, tmp_path / "proc") == 2048
    assert measure.cpus_allowed(42, tmp_path / "proc") == "0-3"
    assert measure.pss_kib(43, tmp_path / "proc") == 0


# --- the watchdog ---------------------------------------------------------------------


def _measure(tmp_path, argv, **kw):
    sampler = measure.Sampler(sample_s=0.05, snapshot_s=100.0)
    return measure.measure(
        argv,
        cwd=tmp_path,
        env=dict(os.environ),
        stdout=tmp_path / "out.txt",
        layout=Layout(1, 1),
        sampler=sampler,
        **kw,
    )


def test_the_wall_cap_ends_a_run_as_timeout(tmp_path):
    m = _measure(tmp_path, ["sleep", "30"], cap_s=0.3)
    assert m.ended_by == "timeout"
    assert m.returncode is None
    assert m.wall_s < 15


def test_a_dof_count_off_the_pin_ends_the_run(tmp_path):
    def watch(line):
        return int(line.split()[1]) if line.startswith("DOF ") else None

    argv = [
        sys.executable,
        "-c",
        "import time; print('DOF 12', flush=True); time.sleep(30)",
    ]
    m = _measure(tmp_path, argv, cap_s=20, watch=watch, pinned_ndof=13)
    assert m.ended_by == "dof"
    assert m.dof_seen == 12


def test_a_finished_run_is_measured(tmp_path):
    m = _measure(tmp_path, [sys.executable, "-c", "print('hi')"], cap_s=20)
    assert m.returncode == 0
    assert m.ended_by is None
    assert (tmp_path / "out.txt").read_text() == "hi\n"


def test_a_short_run_still_has_its_placement_checked(tmp_path):
    m = measure.measure(
        ["taskset", "-c", str(min(os.sched_getaffinity(0))), "sleep", "0.4"],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout=tmp_path / "out.txt",
        layout=Layout(1, 1),
        cap_s=20,
    )
    assert m.returncode == 0
    assert m.placement.status == "pass"


def test_sacct_step_facts_wait_for_a_final_state():
    answers = iter([_done("RUNNING|\n"), _done("OUT_OF_MEMORY by 0|2.5G\n")])
    state, rss = measure.step_facts(
        "77", "3", lambda *_a, **_k: next(answers), wait_s=0.0
    )
    assert state == "OUT_OF_MEMORY"
    assert rss == pytest.approx(2.5)


# --- steps, skip rules, supersede -----------------------------------------------------


def test_the_campaign_name_comes_from_the_manifest(site, tmp_path):
    rd = _plan(site, tmp_path)
    assert rd.campaign.name == rd.manifest.name == "campaign"
    assert rd.campaign.cases_dir == DATA


def test_a_rerun_supersedes_the_earlier_record(site, tmp_path):
    rd = _plan(site, tmp_path, repeats=1)
    row = rd.steps("tiny-wheels")[0]
    _record(rd, row, "failed", reason="first")
    _record(rd, row, "ok")  # what step does: supersede, then write
    path = rd.path / "records" / f"{row['name']}.json"
    records.supersede(path)
    slurm.write(rd, slurm._planned(rd, row, _ctx()), {})
    moved = sorted(p.name for p in path.parent.glob("*.superseded-*.json"))
    assert len(moved) == 1
    assert [p.name for p in records.record_paths(rd.path)] == [path.name]


def test_a_refused_plan_row_is_written_refused(site, tmp_path):
    rd = _plan(site, tmp_path, repeats=1)
    steps_file = rd.path / "jobs" / "tiny-wheels.steps"
    rows = rd.steps("tiny-wheels")
    rows[0]["refused"] = "a reason"
    steps_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert slurm.step(rd, "tiny-wheels", 1, _ctx()) == 0
    found = _status(rd, rows[0]["name"])
    assert (found.status, found.reason) == ("refused", "a reason")


def test_a_timeout_on_repeat_1_skips_the_later_repeats(site, tmp_path):
    rd = _plan(site, tmp_path, solvers=["gfem:direct"])
    rows = rd.steps("tiny-wheels")
    first = next(r for r in rows if r["repeat"] == 1)
    second = next(r for r in rows if r["repeat"] == 2)
    _record(rd, first, "timeout")
    slurm.step(rd, "tiny-wheels", second["line"], _ctx())
    found = _status(rd, second["name"])
    assert (found.status, found.reason) == ("not_started", "earlier_repeat_timeout")


def _bddc_rows(rd):
    """A second, non-default bddc cell beside the default one (as a grid has)."""
    rows = rd.steps("tiny-wheels")
    default = next(r for r in rows if r["solver"] == "gfem:bddc" and r["repeat"] == 1)
    other = dict(default, name=default["name"].replace("default", "1x1"), asked="1x1")
    other["default"] = False
    other["line"] = len(rows) + 1
    rows.append(other)
    (rd.path / "jobs" / "tiny-wheels.steps").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return default, other


def test_a_default_cell_not_converging_skips_the_kinds_other_cells(site, tmp_path):
    rd = _plan(site, tmp_path)
    default, other = _bddc_rows(rd)
    _record(rd, default, "failed", converged="fail", failure_class="solver")
    slurm.step(rd, "tiny-wheels", other["line"], _ctx())
    found = _status(rd, other["name"])
    assert (found.status, found.reason) == ("not_started", "kind_not_converging")
    assert slurm.skip_reason(rd, default, rd.steps("tiny-wheels")) is None


def test_a_converging_default_cell_skips_nothing(site, tmp_path):
    rd = _plan(site, tmp_path)
    default, other = _bddc_rows(rd)
    _record(rd, default, "ok", converged="pass")
    assert slurm.skip_reason(rd, other, rd.steps("tiny-wheels")) is None


def test_too_little_job_time_left_is_not_started(site, tmp_path):
    rd = _plan(site, tmp_path, repeats=1)
    row = rd.steps("tiny-wheels")[0]
    slurm.step(rd, "tiny-wheels", 1, _ctx(job_end=100.0, now=lambda: 50.0))
    found = _status(rd, row["name"])
    assert found.status == "not_started"
    assert found.reason.startswith("job_time_left")


def test_a_harness_error_leaves_a_harness_failure(site, tmp_path, monkeypatch):
    rd = _plan(site, tmp_path, repeats=1)

    def boom(*_args, **_kw):
        raise RuntimeError("broken")

    monkeypatch.setattr(slurm, "_execute", boom)
    slurm.step(rd, "tiny-wheels", 1, _ctx())
    found = _status(rd, rd.steps("tiny-wheels")[0]["name"])
    assert (found.status, found.failure_class) == ("failed", "harness")
    assert "broken" in found.reason


# --- close ----------------------------------------------------------------------------


def _two_codes(site, tmp_path):
    rd = _plan(site, tmp_path, solvers=["gfem:direct", "palace:mumps"], repeats=1)
    return rd, rd.steps("tiny-wheels")


def test_close_passes_a_complete_campaign_and_fills_the_manifest(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    for row in rows:
        _record(rd, row, "ok", ndof=938)
    assert slurm.close(rd, environ={"USER": "someone"}) == 0
    manifest = records.load_campaign(rd.path / "campaign.json")
    assert manifest.checks["complete"] == "pass"
    assert manifest.checks["dof_equal"] == "pass"
    assert set(manifest.records) == {f"records/{r['name']}.json" for r in rows}
    path = rd.path / next(iter(manifest.records))
    assert manifest.records[path.relative_to(rd.path).as_posix()] == (
        records.file_sha256(path)
    )
    assert manifest.closed_utc


def test_close_scores_ok_records_once(site, tmp_path):
    rd = _plan(site, tmp_path, solvers=["gfem:direct", "gfem:bddc"], repeats=1)
    rows = rd.steps("tiny-wheels")
    tiny = rd.case("tiny")
    freqs = list(tiny.problem.frequencies())
    keys = [f"{t},{f}" for t, f in tiny.quantities.s]
    for row in rows:
        shift = 1e-9 if row["solver"] == "gfem:bddc" else 0.0
        outputs = {
            "frequencies_hz": freqs,
            "s": {k: [[0.5 + shift, 0.0] for _ in freqs] for k in keys},
        }
        _record(rd, row, "ok", ndof=938, outputs=outputs)
    assert slurm.close(rd, environ={}) == 0
    found = {r["solver"]: _status(rd, r["name"]) for r in rows}
    bddc = found["gfem:bddc"]
    assert bddc.accuracy is None
    assert bddc.extra["accuracy_reason"] == "no_reference"
    assert bddc.checks["reference_matches"] == "skipped"
    assert bddc.same_code_error["value"] == pytest.approx(1e-9)
    assert bddc.same_code_error["against"] == found["gfem:direct"].name
    manifest = records.load_campaign(rd.path / "campaign.json")
    path = rd.path / "records" / f"{bddc.name}.json"
    assert manifest.records[f"records/{bddc.name}.json"] == records.file_sha256(path)
    before = path.read_bytes()
    assert slurm.close(rd, environ={}) == 0
    assert path.read_bytes() == before


def test_close_fails_on_a_dof_mismatch_between_codes(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    _record(rd, rows[0], "ok", ndof=938)
    _record(rd, rows[1], "ok", ndof=940)
    assert slurm.close(rd, environ={}) == 1
    checks = records.load_campaign(rd.path / "campaign.json").checks
    assert checks["dof_equal"] == "fail"
    assert checks["complete"] == "pass"


def test_dof_equal_counts_a_run_failed_on_its_dof_count(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    _record(rd, rows[0], "ok", ndof=938)
    _record(
        rd,
        rows[1],
        "failed",
        ndof=940,
        failure_class="solver",
        reason="dof_mismatch: palace 0.19.0 940 != pinned 938",
    )
    slurm.close(rd, environ={})
    checks = records.load_campaign(rd.path / "campaign.json").checks
    assert checks["dof_equal"] == "fail"
    assert checks["dof_equal_detail"]["tiny"]["ndof"] == {
        "gfem": [938],
        "palace": [940],
    }


def test_dof_equal_counts_the_dof_the_guard_saw_in_stdout(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    _record(rd, rows[0], "ok", ndof=938)
    _record(
        rd,
        rows[1],
        "failed",
        failure_class="solver",
        extra={"observations": {"dof_seen_in_stdout": 940}},
    )
    slurm.close(rd, environ={})
    checks = records.load_campaign(rd.path / "campaign.json").checks
    assert checks["dof_equal"] == "fail"


def test_dof_equal_fails_on_a_dof_mismatch_even_within_one_code(site, tmp_path):
    rd = _plan(site, tmp_path, solvers=["gfem:direct", "gfem:bddc"], repeats=1)
    rows = rd.steps("tiny-wheels")
    _record(rd, rows[0], "ok", ndof=938)
    _record(rd, rows[1], "failed", ndof=940, failure_class="solver")
    slurm.close(rd, environ={})
    checks = records.load_campaign(rd.path / "campaign.json").checks
    assert checks["dof_equal"] == "fail"


def test_close_fails_on_a_missing_row_but_not_on_a_solver_failure(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    _record(rd, rows[0], "failed", failure_class="solver")
    assert slurm.close(rd, environ={}) == 1
    _record(rd, rows[1], "timeout")
    assert slurm.close(rd, environ={}) == 0


def test_close_fails_on_a_mixed_campaign(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    for row in rows:
        record = _record(rd, row, "ok")
    record.provenance.gfem = {"commit": "f" * 40}
    slurm.write(rd, record, {})
    assert slurm.close(rd, environ={}) == 1
    checks = records.load_campaign(rd.path / "campaign.json").checks
    assert checks["provenance"] == "fail"


def test_close_turns_stubs_killed_and_unreached_rows_not_started(site, tmp_path):
    rd, rows = _two_codes(site, tmp_path)
    _record(rd, rows[0], "started")
    (rd.path / "submitted.tsv").write_text(
        budget.SUBMITTED_HEADER + "\ntiny-wheels\t4242\t2026\tq\t0:20:00\t1.0\n"
    )
    for entry in rd.manifest.jobs:
        entry.extra["slurm_ids"] = ["4242"]
    sacct = "4242|TIMEOUT|1200|268800|gs01|q_debug\n"
    code = slurm.close(rd, run=lambda *_a, **_k: _done(sacct), environ={})
    assert code == 0
    assert _status(rd, rows[0]["name"]).status == "killed"
    assert _status(rd, rows[0]["name"]).slurm_state == "TIMEOUT"
    assert _status(rd, rows[1]["name"]).status == "not_started"
    (job,) = records.load_campaign(rd.path / "campaign.json").jobs
    assert job.billed_core_h == pytest.approx(268800 / 2 / 3600, abs=1e-3)
    assert job.state == "TIMEOUT"


# --- budget and submit ----------------------------------------------------------------


def test_the_budget_refuses_past_the_ceiling_with_sacct_spend(tmp_path):
    out = tmp_path / "out" / "c1"
    out.mkdir(parents=True)
    (out / "submitted.tsv").write_text(
        budget.SUBMITTED_HEADER + "\nj\t11\tt\tq\t1:00:00\t5\nk\t12\tt\tq\t1:00:00\t5\n"
    )
    sacct = "11|COMPLETED|10|36000|n|q\n12|COMPLETED|10|72000|n|q\n"
    files = budget.submitted_files(tmp_path / "out")
    spent = budget.spend(40.0, files, lambda *_a, **_k: _done(sacct))
    assert spent.jobs_core_h == pytest.approx(15.0)
    assert spent.total == pytest.approx(55.0)
    budget.check(spent, 45.0, 100.0)
    with pytest.raises(budget.BudgetError, match="ceiling 100"):
        budget.check(spent, 45.5, 100.0)


def test_unfinished_jobs_count_at_their_time_limit_until_they_finish(tmp_path):
    (tmp_path / "submitted.tsv").write_text(
        budget.SUBMITTED_HEADER
        + "\nj\t11\tt\tq\t1:00:00\t5\nk\t12\tt\tq\t1:30:00\t5"
        + "\nl\t13\tt\tq\t1:00:00\t5\n"
    )
    sacct = (
        "11|COMPLETED|10|36000|n|q|60|1\n"
        "12|PENDING|0|0|None assigned|q|90|2\n"
        "13|RUNNING|600|1000|n|q|60|1\n"
    )
    spent = budget.spend(
        0.0, [tmp_path / "submitted.tsv"], lambda *_a, **_k: _done(sacct)
    )
    # 5 billed + 1.5 h x 2 nodes x 112 + 1 h x 1 node x 112
    assert spent.jobs_core_h == pytest.approx(5.0 + 336.0 + 112.0)


def test_submit_refuses_a_plan_past_the_ceiling(site, tmp_path):
    rd = _plan(site, tmp_path)
    with pytest.raises(budget.BudgetError):
        slurm.submit(rd, dry_run=True, ceiling=41.0, run=lambda *_a, **_k: _done())


def test_submit_dry_run_prints_sbatch_lines_and_submits_nothing(site, tmp_path, capsys):
    rd = _plan(site, tmp_path)

    def no_call(*_args, **_kw):
        raise AssertionError("dry run called a command")

    assert slurm.submit(rd, dry_run=True, run=no_call) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("sbatch "))
    assert "--account=acct42" in line
    assert "--qos=q_debug" in line
    assert "--constraint=fast" in line
    assert "--exclusive" in line
    assert line.endswith(f"{rd.path.resolve()} tiny-wheels")
    assert not (rd.path / "submitted.tsv").exists()


def test_the_job_constraint_joins_the_sites():
    assert slurm.constraint("highmem", "fast") == "highmem&fast"
    assert slurm.constraint(None, "fast") == "fast"
    assert slurm.constraint("highmem", None) == "highmem"
    assert slurm.constraint(None, None) is None


def test_submit_records_slurm_ids_and_pins_the_env(site, tmp_path, monkeypatch):
    rd = _plan(site, tmp_path)
    found = {
        "gfem_manifest": {"sha256": "e" * 64},
        "palace_manifest": {"sha256": "p" * 64},
    }
    monkeypatch.setattr(slurm.stacks, "provenance", lambda _resolved: found)
    calls = []

    def run(argv, **_kw):
        calls.append(argv)
        return _done("31337;cluster\n")

    code = slurm.submit(rd, run=run, prepare=lambda _resolved: tmp_path)
    assert code == 0
    assert calls[0][0] == "sbatch"
    assert slurm.submitted(rd.path)[0]["slurm_id"] == "31337"
    manifest = records.load_campaign(rd.path / "campaign.json")
    assert manifest.jobs[0].extra["slurm_ids"] == ["31337"]
    assert manifest.pins.stacks["wheels"]["env_manifest_sha256"] == "e" * 64


# --- job files ------------------------------------------------------------------------


def test_pip_freeze_keeps_wheel_names_never_paths():
    text = (
        "gfem @ file:///gpfs/scratch/acct/u/gfem/refs/abc/dist/gfem-0.1.0-py3-none-any"
        ".whl#sha256=8c7d\nnumpy==2.5.4\n"
    )
    out = records.scrub_freeze(text, {})
    assert out == "gfem @ file:gfem-0.1.0-py3-none-any.whl#sha256=8c7d\nnumpy==2.5.4\n"
    assert records.scrub(out, {}) == out


def test_the_machine_benchmark_covers_the_jobs_threads_and_widest_layout():
    steps = [
        {"threads": 16, "ranks": 1},
        {"threads": 4, "ranks": 4},
        {"threads": 1, "ranks": 16},
        {"threads": 2, "ranks": 8, "refused": "x"},
    ]
    assert slurm.bench_options(steps) == ([1, 4, 16], "16x1")


# --- ship -----------------------------------------------------------------------------


def test_ship_dry_run_lists_only_committed_files(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            [GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout

    git("init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "b.txt").write_text("b\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    (repo / "untracked.txt").write_text("u\n")
    cases = tmp_path / "cases"
    cases.mkdir()
    code = ship.ship(
        "HEAD",
        ship.Target.parse("host:/r"),
        repo=repo,
        cases_dir=cases,
        dry_run=True,
        run=subprocess.run,
    )
    assert code == 0
    out = capsys.readouterr().out
    listed = [ln.strip() for ln in out.splitlines() if ln.startswith("  ")]
    assert listed[:-1] == ["COMMIT", "COMMIT_UTC", "a.py", "sub/b.txt"]
    assert "untracked.txt" not in out
    sha = git("rev-parse", "HEAD").strip()
    assert f"host:/r/refs/{sha[:7]}/" in out


def test_ship_of_a_tree_another_tool_shipped_adds_the_missing_commit_date(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            [GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout

    git("init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    sha, when = ship.commit_of(repo, "HEAD")
    cases = tmp_path / "cases"
    cases.mkdir()
    calls = []

    def run(argv, **_kw):
        calls.append(list(argv))
        out = "/r/refs/x/dist/gfem-0.1.0-py3-none-any.whl\n" if "ls" in argv[-1] else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    code = ship.ship(
        "HEAD", ship.Target.parse("host:/r"), repo=repo, cases_dir=cases, run=run
    )
    assert code == 0
    remote = [c[-1] for c in calls if c[0] == "ssh"]
    dest = f"/r/refs/{sha[:7]}"
    assert any(
        f"{dest}/COMMIT_UTC" in cmd and when in cmd and "||" in cmd for cmd in remote
    )


def test_ship_refuses_a_store_mesh_with_another_hash(tmp_path):
    case = load_case(REPO / "benchmarks" / "cases" / "r10M_void")
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{case.mesh.sha256}.msh.xz").write_bytes(lzma.compress(b"not the mesh"))
    with pytest.raises(ship.ShipError, match="mesh hash"):
        ship.mesh_sources([case], [store])
    with pytest.raises(ship.ShipError, match="no local store"):
        ship.mesh_sources([case], [tmp_path])


# --- run --local, end to end ----------------------------------------------------------


def test_run_local_leaves_a_record_for_every_row(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    site = tmp_path / "local.toml"
    site.write_text(
        f'name = "ws"\n[cluster]\nroot = "{local}"\n[data]\ndir = "{{root}}/data"\n'
        f'[output]\ndir = "{{root}}/out"\n'
    )
    rundir = tmp_path / "t"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NGS_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    args = ["plan", str(TINY_CAMPAIGN), "--site", str(site), "--out", str(rundir)]
    assert main([*args, "--solver", "gfem:direct", "--repeats", "1"]) == 0
    assert main(["run", str(rundir), "--local"]) == 0
    assert main(["close", str(rundir)]) == 0
    rd = slurm.load_rundir(rundir)
    rows = [row for job in rd.jobs() for row in rd.steps(job)]
    assert rows
    for row in rows:
        found = _status(rd, row["name"])
        assert found.status == "ok", found.reason
        assert found.run.launch == "taskset"
        assert found.provenance.machine.name == "local"
        assert found.case.ndof == 938
        assert found.checks["placement"] == "pass"
        assert found.checks["layout"] == "pass"
        assert found.timings.work_s is not None
        memory = found.memory
        assert 0 < memory.peak_rss_rank_max_gib <= memory.peak_rss_sum_gib
        assert memory.peak_rss_largest_node_gib == pytest.approx(
            memory.peak_rss_sum_gib
        )
