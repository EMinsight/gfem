"""``submit``, ``step``, ``run --local``, ``status``, ``close`` (ADRs 0401, 0410-0414).

A planned run directory (``plan``) holds one steps file per job; each line is one run.

- ``submit`` (login node) prepares the stacks and stages the meshes, refuses past the
  effort's core-hour ceiling (:mod:`benchmarks.harness.budget`), then ``sbatch``es
  ``benchmarks/job.sbatch RUNDIR JOB`` per job and lists the ids in ``submitted.tsv``.
- ``job.sbatch`` runs :func:`job_start` (``jobs/<jobid>/{node.txt, pip-freeze.txt,
  machine.json}``, ``module -t list`` and the BLAS ``ldd`` line once) and then
  ``python -m benchmarks step RUNDIR JOB LINE`` for every line.
- :func:`step` writes a ``started`` stub (an earlier record of the name moves aside
  to ``NAME.superseded-N.json``), applies the skip rules, checks provenance against
  the manifest's pins and re-verifies the staged mesh, then adapter ``prepare`` ->
  :func:`benchmarks.harness.measure.measure` -> ``collect``, and writes the record.
  Every planned row leaves a record: ``refused`` (plan, provenance, mesh, adapter),
  ``not_started`` (skip rules, too little job time left), else what ran.
- ``run --local`` runs the same lines without Slurm (gfem at one rank under taskset,
  Palace under Hydra), records marked ``machine: local``.
- ``close`` turns ``started`` stubs into ``killed`` and missing rows of ended jobs
  into ``not_started`` (from sacct), scores the ok records once against their
  case's ``reference.json`` (:mod:`benchmarks.harness.score`), fills the manifest's
  jobs (billed core-h), records (sha256) and campaign checks, and exits non-zero
  only when ``complete``, ``provenance`` or ``dof_equal`` fails.

Skip rules (ADR 0414): repeat 1 of a cell ending ``timeout`` makes its later repeats
``not_started`` (``earlier_repeat_timeout``); the default-layout cell of an
iterative (case, kind) failing ``converged`` in round 1 makes the other cells of that
(case, kind) ``not_started`` (``kind_not_converging``).

Standard library only (ADR 0400).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks import stacks
from benchmarks.adapters import (
    ADAPTERS,
    Adapter,
    Command,
    CommandResult,
    Layout,
    Measured,
    Outcome,
    Run,
    RunRefusedError,
)
from benchmarks.harness import budget, config, data, matrix, measure, records, score
from benchmarks.harness.case import Case, definition_sha256, load_case
from benchmarks.harness.config import ConfigError, Runner, Site

STEP_STOP = 3
"""``step``'s exit status when Slurm disabled step creation: the job stops."""
MARGIN_S = 120.0
"""A run starts only with at least this much job time left; its cap keeps 60 s."""
STOP_MARGIN_S = 60.0
SITE_COPY = "site.toml"
LOCAL = "local"
REFERENCE_JSON = "reference.json"
HIGHMEM_KIB = 600 * 2**20
"""Above this ``MemTotal`` a node is ``highmem`` (MN5 GPP: 256 GB vs 1 TB)."""
JOB_FILES = ("node.txt", "pip-freeze.txt", "machine.json")
ENDED_STATES = (
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
)
CAMPAIGN_GATES = ("complete", "provenance", "dof_equal")
"""Campaign checks that make ``close`` exit non-zero (ADR 0414)."""


class HarnessError(Exception):
    """A step the harness itself could not carry out (failure class ``harness``)."""


# --- run directory --------------------------------------------------------------------


@dataclass(frozen=True)
class RunDir:
    """A planned run directory, read back."""

    path: Path
    manifest: records.Campaign
    campaign: config.Campaign
    site: Site

    @property
    def records_dir(self) -> Path:
        """Where the suite records go."""
        return self.path / "records"

    @property
    def cases_dir(self) -> Path:
        """Where the campaign's case directories are."""
        return self.campaign.cases_dir or config.CASES_DIR

    def jobs(self) -> list[str]:
        """The planned jobs, by name."""
        return sorted(p.stem for p in (self.path / "jobs").glob("*.steps"))

    def steps(self, job: str) -> list[dict[str, Any]]:
        """The lines of job ``job``."""
        path = self.path / "jobs" / f"{job}{matrix.STEPS_SUFFIX}"
        if not path.is_file():
            raise ConfigError(f"{self.path}: no job {job!r} (jobs: {self.jobs()})")
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def case(self, name: str) -> Case:
        """The case file of ``name``."""
        return load_case(self.cases_dir / name)

    def roots(self) -> dict[str, str]:
        """Site roots by placeholder, for the record scrub."""
        return records.site_roots(self.site)


def load_rundir(rundir: Path, site: Path | None = None) -> RunDir:
    """Read ``rundir``: manifest, resolved campaign file and the site file.

    The site file is ``site`` if given, else ``RUNDIR/site.toml`` (copied by ``plan``
    and ``submit``), else ``$GFEM_BENCH_SITE``. The campaign's name is the
    manifest's: the resolved copy is always called ``campaign.toml``.
    """
    manifest = records.load_campaign(rundir / "campaign.json")
    if site is None and (rundir / SITE_COPY).is_file():
        site = rundir / SITE_COPY
    loaded_site = config.load_site(site)
    campaign = config.load_campaign(rundir / "campaign.toml")
    campaign = dataclasses.replace(campaign, name=manifest.name)
    return RunDir(rundir, manifest, campaign, loaded_site)


def copy_site(site: Site, rundir: Path) -> None:
    """Keep the site file beside the run directory's plan (never in the store)."""
    if site.path.resolve() != (rundir / SITE_COPY).resolve():
        shutil.copyfile(site.path, rundir / SITE_COPY)


# --- context of a step ----------------------------------------------------------------


@dataclass
class Context:
    """Where a step runs: inside a Slurm job, or ``run --local``."""

    local: bool
    job_id: str
    """Slurm job id, or ``local``."""
    environ: Mapping[str, str]
    job_end: float | None = None
    """Epoch seconds the job ends (``BENCH_JOB_END``), if known."""
    run: Runner = subprocess.run
    sampler: measure.Sampler | None = None
    now: Callable[[], float] = time.time

    @classmethod
    def from_env(cls, *, local: bool, environ: Mapping[str, str] | None = None):
        """The context of this process."""
        environ = dict(os.environ if environ is None else environ)
        job = environ.get("SLURM_JOB_ID")
        end = environ.get("BENCH_JOB_END")
        return cls(
            local=local or job is None,
            job_id=LOCAL if local or job is None else job,
            environ=environ,
            job_end=float(end) if end else None,
        )

    def left_s(self) -> float | None:
        """Seconds of job time left, if the job's end is known."""
        return None if self.job_end is None else self.job_end - self.now()

    def job_dir(self, rundir: Path) -> Path:
        """``jobs/<jobid>/``: job-level files."""
        return rundir / "jobs" / self.job_id


# --- small helpers --------------------------------------------------------------------


def _utc(when: float | None = None) -> str:
    moment = (
        dt.datetime.now(dt.UTC)
        if when is None
        else dt.datetime.fromtimestamp(when, dt.UTC)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _reference(case: Case) -> tuple[dict[str, Any] | None, Path | None]:
    if case.directory is None:
        return None, None
    path = case.directory / REFERENCE_JSON
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text()), path
    except json.JSONDecodeError:
        return None, path


def pinned_ndof(case: Case) -> int | None:
    """The DOF pin of the case's mesh at its order (``reference.json``), if any."""
    reference, _ = _reference(case)
    if reference is None:
        return None
    try:
        value = reference["meshes"][case.mesh.sha256]["dof"][str(case.order)]
    except (KeyError, TypeError):
        return None
    return int(value) if value is not None else None


def _record_path(rundir: Path, name: str) -> Path:
    return rundir / "records" / f"{name}.json"


def _load(path: Path) -> records.SuiteRecord | None:
    try:
        return records.load_record(path)
    except (OSError, records.RecordError):
        return None


def _planned(rd: RunDir, row: Mapping[str, Any], ctx: Context) -> records.SuiteRecord:
    run = records.Run(
        case=row["case"],
        solver=row["solver"],
        stack=row["stack"],
        layout=records.Layout(row["asked"], row["ranks"], row["threads"]),
        core_budget=row["core_budget"],
        repeat=row["repeat"],
        tolerance=row["tolerance"],
        order_in_job=row["line"],
        job=ctx.job_id,
        added_for=row.get("added_for"),
    )
    machine = records.Machine(name=LOCAL if ctx.local else rd.site.name)
    return records.SuiteRecord(
        campaign=rd.manifest.id,
        name=row["name"],
        run=run,
        status="started",
        provenance=records.Provenance(machine=machine),
    )


def write(rd: RunDir, record: records.SuiteRecord, environ: Mapping[str, str]) -> Path:
    """Scrub and write ``record``; a record that leaks becomes a harness failure."""
    secrets = records.site_secrets(rd.site, environ)
    try:
        return records.write_record(
            record, rd.records_dir, roots=rd.roots(), secrets=secrets
        )
    except records.ScrubError as error:
        print(f"benchmarks step {record.name}: {error}", file=sys.stderr)
        bare = _bare(record, "failed", "scrub: the record named private text")
        bare.failure_class = "harness"
        return records.write_record(
            bare, rd.records_dir, roots=rd.roots(), secrets=secrets
        )


def _bare(
    record: records.SuiteRecord, status: str, reason: str | None
) -> records.SuiteRecord:
    """``record`` reduced to its plan, with a new status and reason."""
    return records.SuiteRecord(
        campaign=record.campaign,
        name=record.name,
        run=record.run,
        status=status,
        reason=reason,
        provenance=records.Provenance(machine=record.provenance.machine),
    )


# --- skip rules -----------------------------------------------------------------------


def _rep_name(name: str, repeat: int) -> str:
    base, _, _ = name.rpartition("__rep")
    return f"{base}__rep{repeat}"


def skip_reason(
    rd: RunDir, row: Mapping[str, Any], steps: Sequence[Mapping]
) -> str | None:
    """``earlier_repeat_timeout``, ``kind_not_converging`` or ``None`` (ADR 0414)."""
    if row["repeat"] > 1:
        first = _load(_record_path(rd.path, _rep_name(row["name"], 1)))
        if first is not None and first.status == "timeout":
            return "earlier_repeat_timeout"
    code, _, kind = row["solver"].partition(":")
    adapter = ADAPTERS.get(code)
    if adapter is None or adapter.kinds[kind].klass != "iterative":
        return None
    cell = _rep_name(row["name"], 1)
    for other in steps:
        if (
            other["case"] == row["case"]
            and other["solver"] == row["solver"]
            and other["stack"] == row["stack"]
            and other.get("default")
            and other["repeat"] == 1
            and other["tolerance"] == rd.campaign.anchor_tolerance
            and other["name"] != cell
            and other.get("added_for") is None
        ):
            found = _load(_record_path(rd.path, other["name"]))
            if found is not None and found.checks.get("converged") == "fail":
                return "kind_not_converging"
    return None


# --- provenance -----------------------------------------------------------------------


def provenance_problems(
    rd: RunDir, resolved: stacks.Resolved, *, local: bool
) -> list[str]:
    """How the env a step is about to run disagrees with the campaign's pins."""
    pins = rd.manifest.pins
    problems = []
    commit, _ = matrix.source_commit()
    pinned = pins.gfem.get("commit")
    if pinned and commit != pinned:
        problems.append(f"gfem commit {commit} != pinned {pinned}")
    for code, version in pins.solvers.items():
        have = resolved.versions.get(code)
        if have and have != version:
            problems.append(f"{code} {have} != pinned {version}")
    if not local:
        stack_pins = pins.stacks.get(resolved.name, {})
        try:
            found = stacks.provenance(resolved)
        except (OSError, KeyError, IndexError):
            found = {}
        for key, part in (
            ("env_manifest_sha256", "gfem_manifest"),
            ("palace_manifest_sha256", "palace_manifest"),
        ):
            want = stack_pins.get(key)
            have = (found.get(part) or {}).get("sha256")
            if want and have != want:
                problems.append(f"stack {resolved.name} {key} {have} != pinned {want}")
    return problems


# --- the machine ----------------------------------------------------------------------


def _meminfo_kib(proc: Path = measure.PROC) -> int | None:
    try:
        text = (proc / "meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def _cpu_model(proc: Path = measure.PROC) -> str | None:
    try:
        text = (proc / "cpuinfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return None


def node_class(mem_total_kib: int | None) -> str | None:
    """``standard`` or ``highmem`` from ``MemTotal`` (an observation)."""
    if mem_total_kib is None:
        return None
    return "highmem" if mem_total_kib > HIGHMEM_KIB else "standard"


def _machine(
    rd: RunDir, ctx: Context, placement: measure.Placement | None
) -> records.Machine:
    mem = _meminfo_kib()
    masks = placement.masks if placement else []
    cpus = sorted({c for mask in masks for c in measure.parse_cpu_list(mask)})
    env = ctx.environ
    return records.Machine(
        name=LOCAL if ctx.local else rd.site.name,
        nodes=[platform.node()],
        node_class=node_class(mem),
        cpu=_cpu_model(),
        mem_total_kb=[mem] if mem else [],
        cpus_allowed=len(cpus) or None,
        pinned_cpus=measure.cpu_list(cpus) if cpus else None,
        exclusive=not ctx.local,
        partition=env.get("SLURM_JOB_PARTITION"),
        qos=env.get("BENCH_QOS"),
        constraint=env.get("BENCH_CONSTRAINT") or None,
    )


# --- one step -------------------------------------------------------------------------


@dataclass
class _Launched:
    measurement: measure.Measurement
    launch: str
    env: dict[str, str]
    script: str
    cap_s: float
    cap_reason: str
    stdout: Path


def _available_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover (not Linux)
        return list(range(os.cpu_count() or 1))


def local_env(
    resolved: stacks.Resolved, code: str, threads: int, environ: Mapping[str, str]
) -> dict[str, str]:
    """``run --local``'s env: the caller's, thread and affinity variables replaced.

    What the stack's env script exports is added; modules and ``PATH`` are not.
    """
    drop = (*stacks.THREAD_VARIABLES, *config.AFFINITY_VARIABLES)
    env = {k: v for k, v in environ.items() if k not in drop}
    section = resolved.description.codes[code]
    env.update(section.env)
    env.update(resolved.env)
    env.update({k: str(v) for k, v in stacks.thread_exports(section, threads).items()})
    env.update(stacks.PLACES)
    return env


def _argv(command: Command, run: Run, ctx: Context) -> tuple[tuple[str, ...], str]:
    """The argv the harness starts and the record's ``run.launch``."""
    if not ctx.local:
        if command.launch == "srun":
            return command.argv, "srun"
        cpus = measure.block(run.layout.threads, _available_cpus())
        return measure.taskset_argv(command.program, cpus), "taskset"
    program = command.program
    if program and program[0] == "python":
        program = (sys.executable, *program[1:])
    cpus = measure.block(run.layout.cores, _available_cpus())
    if run.layout.ranks == 1:
        return measure.taskset_argv(program, cpus), "taskset"
    if run.code == "gfem":
        raise RunRefusedError("run --local runs gfem at one rank only")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        raise RunRefusedError("run --local: no mpiexec (Hydra) on PATH")
    return measure.hydra_argv(program, run.layout, cpus, mpiexec), "hydra"


@dataclass(frozen=True)
class _Job:
    """One run being executed: where, how, and with what."""

    rd: RunDir
    ctx: Context
    run: Run
    resolved: stacks.Resolved
    adapter: Adapter


def _base_env(x: _Job) -> tuple[dict[str, str], str]:
    run = x.run
    if x.ctx.local:
        env = local_env(x.resolved, run.code, run.layout.threads, x.ctx.environ)
        return env, ""
    script = stacks.render(x.resolved, run.code, run.layout.threads)
    path = run.workdir / "env.sh"
    path.write_text(script)
    return measure.sourced_env(path, x.ctx.environ), script


def _version(x: _Job, env: Mapping[str, str]) -> str | None:
    """The adapter's version query, run once per job and code and kept."""
    job_dir = x.ctx.job_dir(x.rd.path)
    job_dir.mkdir(parents=True, exist_ok=True)
    keep = job_dir / f"{x.adapter.code}-version.txt"
    if keep.is_file():
        return keep.read_text()
    (command,) = x.adapter.version(x.resolved).commands
    program = command.program
    if x.ctx.local and program and program[0] == "python":
        program = (sys.executable, *program[1:])
    done = subprocess.run(
        list(program),
        cwd=job_dir,
        env=measure.command_env(env, command.env),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if done.returncode != 0:
        return None
    keep.write_text(done.stdout)
    return done.stdout


def _launch(x: _Job, command: Command, base: Mapping[str, str], script: str):
    run, ctx = x.run, x.ctx
    cap, why = run.wall_cap_s or float("inf"), "wall cap"
    left = ctx.left_s()
    if left is not None and left - STOP_MARGIN_S < cap:
        cap, why = left - STOP_MARGIN_S, "job time left"
    argv, launch = _argv(command, run, ctx)
    env = measure.command_env(base, command.env)
    sampler = ctx.sampler or measure.Sampler(
        slurm_job=None if ctx.local or launch != "srun" else ctx.job_id
    )
    m = measure.measure(
        argv,
        cwd=run.workdir,
        env=env,
        stdout=run.workdir / command.stdout,
        layout=run.layout,
        cap_s=cap,
        watch=command.watch,
        pinned_ndof=run.pinned_ndof,
        mem_cap_gib=run.mem_cap_gib,
        sampler=sampler,
        threads_log=run.workdir / "threads.txt",
    )
    return _Launched(m, launch, env, script, cap, why, run.workdir / command.stdout)


def _timings(
    case: Case, outcome: Outcome, wall_s: float, adapter: Adapter
) -> records.Timings:
    t = records.Timings(wall_s=round(wall_s, 3))
    for bucket in records.BUCKETS:
        setattr(t, bucket, outcome.phases.get(bucket))
    span = outcome.work_span_s
    if span is not None:
        t.startup_s = max(wall_s - span, 0.0)
        t.work_s = wall_s - t.startup_s
    parts = [t.mesh_read_s, t.discretize_s, t.assemble_s, t.setup_s]
    present = [p for p in parts if p is not None]
    t.setup_total_s = sum(present) if present else None
    t.solve_total_s = t.solve_s
    if case.problem.kind == "driven" and t.work_s is not None:
        samples = len(case.problem.frequencies())
        t.per_sample_s = t.work_s / samples if samples else None
    t.rank_reduction = outcome.rank_reduction
    t.source = outcome.native.get("source") or adapter.timing_source
    t.rank0 = dict(outcome.rank0) if outcome.rank0 else None
    return t


def _memory(
    outcome: Outcome, m: measure.Measurement, sacct_rss: float | None, adapter: Adapter
) -> records.Memory:
    return records.Memory(
        peak_rss_sum_gib=outcome.self_peak_mem_gib,
        peak_rss_rank_max_gib=outcome.self_peak_rank_max_gib,
        peak_rss_largest_node_gib=outcome.self_peak_largest_node_gib,
        metric=adapter.memory_metric if outcome.self_peak_mem_gib is not None else None,
        pss_peak_sum_gib=m.pss_peak_sum_gib,
        sacct_max_rss_gib=sacct_rss,
        self_reports=(
            {"solver_nbytes": outcome.solver_nbytes}
            if outcome.solver_nbytes is not None
            else {}
        ),
    )


def _layout_check(run: Run, outcome: Outcome) -> str:
    seen = (outcome.ranks_seen, outcome.threads_seen)
    if None in seen:
        return "skipped"
    return "pass" if seen == (run.layout.ranks, run.layout.threads) else "fail"


def _job_text(rd: RunDir, ctx: Context, name: str) -> str:
    path = ctx.job_dir(rd.path) / name
    return path.read_text() if path.is_file() else ""


def _case_facts(case: Case) -> records.CaseFacts:
    facts = records.CaseFacts(
        definition_sha256=definition_sha256(case),
        mesh_sha256=case.mesh.sha256,
        order=case.order,
    )
    reference, path = _reference(case)
    if reference is not None and path is not None:
        facts.reference = records.Reference(
            id=str(reference.get("id")), sha256=records.file_sha256(path)
        )
    return facts


def _resolve(rd: RunDir, name: str) -> stacks.Resolved:
    commit = rd.manifest.pins.gfem.get("commit") or matrix.source_commit()[0]
    return stacks.resolve(
        name,
        rd.site,
        rd.campaign.stack_versions.get(name),
        rd.campaign.stack_env.get(name),
        commit=commit,
    )


def _precheck(
    rd: RunDir, ctx: Context, case: Case, resolved: stacks.Resolved, record
) -> Path | None:
    """Provenance against the pins and the staged mesh; ``None`` when refused."""
    pins = rd.manifest.pins.stacks.get(resolved.name, {})
    record.provenance.stack = {
        "name": resolved.name,
        "description_sha256": pins.get("description_sha256"),
        "env_manifest_sha256": pins.get("env_manifest_sha256"),
        "fingerprint": resolved.fingerprint,
    }
    problems = provenance_problems(rd, resolved, local=ctx.local)
    if problems:
        record.status, record.reason = "refused", "provenance: " + "; ".join(problems)
        record.checks["provenance"] = "fail"
        return None
    staged = data.staged_path(Path(rd.site.data_dir), case.mesh.sha256)
    try:
        data.verify_staged(staged, case.mesh.sha256)
    except data.Refused as error:
        record.status, record.reason = "refused", str(error)
        record.checks["mesh_hash"] = "fail"
        return None
    record.checks["mesh_hash"] = "pass"
    return staged


def _run_commands(x: _Job) -> tuple[list[_Launched], str | None]:
    """Prepare, then launch each command until one does not end cleanly."""
    launch = x.adapter.prepare(x.run)
    base, script = _base_env(x)
    version = _version(x, base)
    launched = []
    for command in launch.commands:
        done = _launch(x, command, base, script)
        launched.append(done)
        m = done.measurement
        if m.ended_by is not None or m.returncode not in {0, None}:
            break
    return launched, version


CRASH_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGFPE", "SIGILL")
    if hasattr(signal, name)
)
"""Signals of a task that crashed: srun then cancels the step itself."""


def slurm_verdict(
    state: str | None, returncode: int | None, ended_by: str | None
) -> tuple[str, str | None, str] | None:
    """``(status, failure_class, reason)`` Slurm's step state imposes, if any.

    A step ``CANCELLED`` because its task died of a crash signal (exit 128 + N) is a
    ``solver`` failure, not a kill; any other killed state is ``killed``. The
    watchdog's own ending wins.
    """
    if ended_by is not None or state not in measure.KILLED_STATES:
        return None
    if state == "CANCELLED" and returncode is not None and returncode > 128:
        number = returncode - 128
        crash = {int(s): s.name for s in CRASH_SIGNALS}
        if number in crash:
            return (
                "failed",
                "solver",
                f"crashed: signal {number} ({crash[number]}); slurm step {state}",
            )
    return "killed", None, f"slurm: {state}"


def _status(
    x: _Job, record: records.SuiteRecord, launched: Sequence[_Launched], wall: float
) -> float | None:
    """Slurm's and the watchdog's word over the adapter's; the sacct MaxRSS."""
    last = launched[-1]
    m = last.measurement
    sacct_rss = None
    if not x.ctx.local and last.launch == "srun" and m.step is not None:
        state, rss = measure.step_facts(x.ctx.job_id, m.step, x.ctx.run)
        record.slurm_state = state
        sacct_rss = rss if wall > measure.SACCT_MIN_S else None
        verdict = slurm_verdict(state, m.returncode, m.ended_by)
        if verdict is not None:
            record.status, record.failure_class, reason = verdict
            if record.status == "failed" and record.reason:
                reason = f"{reason}; {record.reason}"
            record.reason = reason
    if m.ended_by == "timeout":
        record.status, record.failure_class = "timeout", None
        record.reason = f"{last.cap_reason} {last.cap_s:.0f} s"
    elif m.ended_by == "memory_watchdog":
        record.status, record.failure_class = "memory_watchdog", None
        record.reason = f"memory cap {x.run.mem_cap_gib} GiB"
    return sacct_rss


def _fill(
    x: _Job,
    record: records.SuiteRecord,
    outcome: Outcome,
    launched: Sequence[_Launched],
) -> None:
    """Write what ran into ``record``."""
    run = x.run
    first, last = launched[0].measurement, launched[-1].measurement
    record.run.launch = launched[0].launch
    record.run.step = last.step
    record.run.started_utc = first.started_utc
    record.run.finished_utc = last.finished_utc
    record.returncode = last.returncode
    record.status, record.reason = outcome.status, outcome.reason
    record.failure_class = outcome.failure_class
    wall = sum(d.measurement.wall_s for d in launched)
    sacct_rss = _status(x, record, launched, wall)
    record.case.ndof = outcome.ndof
    record.outputs = {**outcome.quantities, "iterations": outcome.iterations}
    record.timings = _timings(run.case, outcome, wall, x.adapter)
    record.memory = _memory(outcome, last, sacct_rss, x.adapter)
    record.structural = outcome.structural
    observed = outcome.provenance.get("version")
    pinned = x.rd.manifest.pins.solvers.get(run.code)
    record.checks.update(outcome.checks)
    drift = bool(observed and pinned and observed != pinned)
    record.checks["provenance"] = "fail" if drift else "pass"
    record.checks["layout"] = _layout_check(run, outcome)
    record.checks["placement"] = first.placement.status
    record.extra["observations"] = {
        "busy_threads_per_core_max": first.busy_threads_per_core,
        "placement": dataclasses.asdict(first.placement),
        "dof_seen_in_stdout": first.dof_seen,
    }
    _fill_provenance(x, record, outcome, launched)


def _fill_provenance(
    x: _Job,
    record: records.SuiteRecord,
    outcome: Outcome,
    launched: Sequence[_Launched],
) -> None:
    versions = dict(outcome.provenance.get("versions") or {})
    solver = {k: v for k, v in outcome.provenance.items() if k != "versions"}
    solver["code"] = x.run.code
    solver["version"] = outcome.provenance.get("version") or versions.get(x.run.code)
    record.provenance = records.Provenance(
        gfem={"commit": x.resolved.commit, "version": versions.get("gfem")},
        harness={"commit": matrix.source_commit()[0]},
        solver=solver,
        stack=record.provenance.stack,
        versions={k: str(v) for k, v in versions.items()},
        env=records.allowed_env(launched[0].env),
        machine=_machine(x.rd, x.ctx, launched[0].measurement.placement),
    )
    found: dict[str, Any] = {}
    if not x.ctx.local:
        with contextlib.suppress(OSError, KeyError, IndexError):
            found = stacks.provenance(x.resolved)
    record.stack = stacks.record(
        x.resolved,
        env_script=launched[0].script,
        provenance=found,
        modules=_job_text(x.rd, x.ctx, "modules.txt"),
        ldd=_job_text(x.rd, x.ctx, "ldd.txt"),
    )
    if x.ctx.local:
        record.stack["local"] = "run --local: the harness's own interpreter and env"


def _execute(
    rd: RunDir,
    row: Mapping[str, Any],
    ctx: Context,
    record: records.SuiteRecord,
) -> tuple[records.SuiteRecord, bool]:
    """Run one planned row; return its record and whether the job must stop."""
    case = rd.case(row["case"])
    record.case = _case_facts(case)
    resolved = _resolve(rd, row["stack"])
    staged = _precheck(rd, ctx, case, resolved, record)
    if staged is None:
        return record, False
    workdir = rd.path / "work" / row["name"]
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    run = Run(
        case=case,
        solver=row["solver"],
        tolerance=row["tolerance"],
        layout=Layout(row["ranks"], row["threads"]),
        core_budget=row["core_budget"],
        repeat=row["repeat"],
        stack=resolved,
        workdir=workdir,
        mesh=staged,
        pinned_ndof=pinned_ndof(case),
        wall_cap_s=row["wall_cap_s"],
    )
    x = _Job(rd, ctx, run, resolved, ADAPTERS[run.code])
    try:
        launched, version = _run_commands(x)
    except RunRefusedError as error:
        record.status, record.reason = "refused", str(error)
        return record, False
    results = tuple(
        CommandResult(d.measurement.returncode, d.stdout, d.measurement.wall_s)
        for d in launched
    )
    outcome = x.adapter.collect(run, Measured(results, version))
    _fill(x, record, outcome, launched)
    return record, any(d.measurement.step_disabled for d in launched)


def step(rd: RunDir, job: str, line: int, ctx: Context) -> int:
    """Run line ``line`` of job ``job``; always leave a record.

    Exit status 0, or :data:`STEP_STOP` when Slurm disabled step creation.
    """
    steps = rd.steps(job)
    if not 1 <= line <= len(steps):
        raise ConfigError(f"job {job} has lines 1-{len(steps)}, not {line}")
    row = steps[line - 1]
    records.supersede(_record_path(rd.path, row["name"]))
    record = _planned(rd, row, ctx)
    if row.get("refused"):
        record.status, record.reason = "refused", row["refused"]
        write(rd, record, ctx.environ)
        return 0
    skip = skip_reason(rd, row, steps)
    left = ctx.left_s()
    if skip is None and left is not None and left < MARGIN_S:
        skip = f"job_time_left: {max(left, 0):.0f} s"
    if skip is not None:
        record.status, record.reason = "not_started", skip
        write(rd, record, ctx.environ)
        return 0
    record.run.started_utc = _utc()
    write(rd, record, ctx.environ)  # the STARTED stub
    stop = False
    try:
        record, stop = _execute(rd, row, ctx, record)
    except Exception as error:  # noqa: BLE001 (every planned row leaves a record)
        record = _bare(record, "failed", f"harness: {type(error).__name__}: {error}")
        record.failure_class = "harness"
    write(rd, record, ctx.environ)
    return STEP_STOP if stop else 0


# --- run --local ----------------------------------------------------------------------


def stage_meshes(rd: RunDir, cases: Sequence[str]) -> dict[str, str]:
    """Stage every case's mesh; return the refusal reason per case that failed."""
    refused = {}
    stores = [Path(s) for s in rd.site.data_stores]
    for name in cases:
        try:
            data.stage(rd.case(name), Path(rd.site.data_dir), stores)
        except data.Refused as error:
            refused[name] = str(error)
    return refused


def run_local(rd: RunDir, job: str | None = None, ctx: Context | None = None) -> int:
    """``run --local``: every line of every job (or of ``job``) without Slurm."""
    ctx = ctx or Context.from_env(local=True)
    jobs = [job] if job else rd.jobs()
    for name in jobs:
        steps = rd.steps(name)
        stage_meshes(rd, sorted({s["case"] for s in steps}))
        for line in range(1, len(steps) + 1):
            row = steps[line - 1]
            print(f"--- {name} {line}/{len(steps)} {row['name']}", flush=True)
            step(rd, name, line, ctx)
            found = _load(_record_path(rd.path, row["name"]))
            if found is not None:
                print(f"    {found.status} {found.reason or ''}".rstrip(), flush=True)
    return 0


# --- job start (inside job.sbatch) ----------------------------------------------------


def _ldd_lines(paths: Sequence[Path]) -> str:
    out = []
    for path in paths:
        done = subprocess.run(
            ["ldd", str(path)], capture_output=True, text=True, check=False
        )
        out += [
            f"{path.name}: {line.strip()}"
            for line in done.stdout.splitlines()
            if any(k in line.lower() for k in ("blas", "lapack", "mkl"))
        ]
    return "\n".join(out) + ("\n" if out else "")


def job_start(rd: RunDir, job: str, ctx: Context) -> int:
    """Write the job-level facts once, at the job's start.

    ``node.txt``, ``pip-freeze.txt``, ``modules.txt``, ``ldd.txt`` and the machine
    benchmark ``machine.json`` (whose failure does not stop the job).
    """
    out = ctx.job_dir(rd.path)
    out.mkdir(parents=True, exist_ok=True)
    roots = rd.roots()
    secrets = records.site_secrets(rd.site, ctx.environ)

    if ctx.environ.get("TMPDIR"):
        roots = {**roots, "{tmpdir}": ctx.environ["TMPDIR"]}

    def keep(name: str, text: str) -> None:
        try:
            records.scrub(text, roots, secrets)
        except records.ScrubError as error:
            print(f"job start: {name}: {error}", file=sys.stderr)
            return
        (out / name).write_text(records.replace_roots(text, roots))

    node = [f"host {platform.node()}"]
    for argv in (["lscpu"], ["numactl", "-H"]):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False)
            node.append(done.stdout)
        except OSError:
            node.append(f"({argv[0]} not available)")
    mem = _meminfo_kib()
    node.append(f"MemTotal: {mem} kB")
    keep("node.txt", "\n".join(node) + "\n")
    steps = rd.steps(job)
    stack_names = sorted({s["stack"] for s in steps})
    commit = rd.manifest.pins.gfem.get("commit") or matrix.source_commit()[0]
    for name in stack_names:
        resolved = stacks.resolve(
            name,
            rd.site,
            rd.campaign.stack_versions.get(name),
            rd.campaign.stack_env.get(name),
            commit=commit,
        )
        python = Path(resolved.code_env.get("gfem", "")) / "bin" / "python"
        if python.is_file():
            done = subprocess.run(
                [str(python), "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=False,
            )
            text = records.scrub_freeze(done.stdout, roots)
            keep("pip-freeze.txt", text)
        script = out / f"env-{name}-gfem.sh"
        script.write_text(stacks.render(resolved, "gfem"))
        done = subprocess.run(
            [
                "bash",
                "-c",
                '. "$1" >/dev/null 2>&1; module -t list 2>&1',
                "bash",
                str(script),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (out / "modules.txt").write_text(done.stdout)
        libs = sorted(
            Path(resolved.code_env.get("gfem", "/")).glob(
                "lib/python3*/site-packages/ngsolve/libngla*.so"
            )
        )
        libs += sorted(
            Path(resolved.code_env.get("palace", "/")).glob(
                "lib/python3*/site-packages/palace_solver/bin/palace-real"
            )
        )
        (out / "ldd.txt").write_text(_ldd_lines(libs))
        machine_benchmark(rd, job=job, ctx=ctx, python=python, out=out, keep=keep)
    return 0


def bench_options(steps: Sequence[Mapping[str, Any]]) -> tuple[list[int], str]:
    """``--threads`` (distinct T of the job) and ``--concurrent`` (most ranks)."""
    threads = sorted({int(s["threads"]) for s in steps if not s.get("refused")})
    widest = max(
        (s for s in steps if not s.get("refused")),
        key=lambda s: (s["ranks"], -s["threads"]),
        default=None,
    )
    concurrent = f"{widest['ranks']}x{widest['threads']}" if widest else "1x1"
    return threads or [1], concurrent


def machine_benchmark(
    rd: RunDir,
    *,
    job: str,
    ctx: Context,
    python: Path,
    out: Path,
    keep: Callable[[str, str], None],
) -> None:
    """``tools/scale_bench.py node`` as one srun step; scrubbed to ``machine.json``."""
    steps = rd.steps(job)
    threads, concurrent = bench_options(steps)
    budget_cores = max(int(s["core_budget"]) for s in steps)
    raw = out / "machine.raw.json"
    tool = matrix.ROOT / "tools" / "scale_bench.py"
    argv = [
        *measure_srun(Layout(1, budget_cores)),
        str(python),
        str(tool),
        "node",
        "--threads",
        *map(str, threads),
        "--concurrent",
        concurrent,
        "--build-dir",
        ctx.environ.get("TMPDIR", str(out)),
        "--out",
        str(raw),
    ]
    log = out / "machine.log"
    with log.open("w") as stream:
        done = subprocess.run(
            argv, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=900
        )
    if done.returncode != 0 or not raw.is_file():
        print(
            f"machine benchmark exited {done.returncode} (see {log})", file=sys.stderr
        )
        return
    keep("machine.json", raw.read_text())


def measure_srun(layout: Layout) -> list[str]:
    """The suite's srun line for a one-process harness step (``--mpi=none``)."""
    from benchmarks.adapters import srun_argv

    return list(srun_argv(layout, "none"))


# --- submit ---------------------------------------------------------------------------


def _plan_rows(rundir: Path) -> dict[str, dict[str, str]]:
    lines = (rundir / "plan.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    return {
        row[0]: dict(zip(header, row, strict=False))
        for row in (line.split("\t") for line in lines[1:] if line)
    }


def constraint(job: str | None, site: str | None) -> str | None:
    """The job's constraint (``highmem``) and the site's, joined with ``&``."""
    parts = [c for c in (job, site) if c]
    return "&".join(dict.fromkeys(parts)) or None


def sbatch_argv(
    rd: RunDir,
    job: str,
    row: Mapping[str, str],
    *,
    qos: str | None = None,
    python: str = sys.executable,
) -> list[str]:
    """``sbatch --parsable ... benchmarks/job.sbatch RUNDIR JOB``."""
    site = rd.site
    queue = qos or site.qos.get(row["qos"], row["qos"])
    argv = ["sbatch", "--parsable"]
    if site.account:
        argv.append(f"--account={site.account}")
    argv += [
        f"--qos={queue}",
        f"--time={row['time_limit']}",
        f"--nodes={row['nodes']}",
        "--exclusive",
        "--ear=off",
        f"--job-name=bench-{job}",
    ]
    joined = constraint(row.get("constraint") or None, site.constraint or None)
    if joined:
        argv.append(f"--constraint={joined}")
    rundir = rd.path.resolve()
    argv += [
        f"--output={rundir}/logs/job-%j.out",
        f"--export=ALL,BENCH_TREE={matrix.ROOT},BENCH_PYTHON={python}",
        str(matrix.ROOT / "benchmarks" / "job.sbatch"),
        str(rundir),
        job,
    ]
    return argv


def _wheel(root: Path = matrix.ROOT) -> Path:
    found = sorted((root / "dist").glob("gfem-*.whl"))
    if not found:
        raise ConfigError(f"no gfem wheel under {root}/dist: ship the commit first")
    return found[-1]


def limit_seconds(value: str) -> int:
    """Seconds of an ``H:MM:SS`` time limit; :class:`ConfigError` if malformed."""
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", value)
    seconds = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) if match else 0
    if seconds <= 0:
        raise ConfigError(f"--time {value!r}: expected a positive H:MM:SS")
    return seconds


def with_time_limit(row: Mapping[str, str], value: str) -> dict[str, str]:
    """A ``plan.tsv`` row at time limit ``value`` (``H:MM:SS``).

    The QOS purpose and the estimate follow the new limit as ``plan`` sets them:
    ``debug`` up to 2 h, else ``long``; estimate = limit / 1.5.
    """
    seconds = limit_seconds(value)
    hours, rest = divmod(seconds, 3600)
    est_s = seconds / matrix.LIMIT_FACTOR
    nodes = int(row["nodes"])
    return {
        **row,
        "time_limit": f"{hours}:{rest // 60:02d}:{rest % 60:02d}",
        "qos": "debug" if seconds <= matrix.DEBUG_LIMIT_S else "long",
        "est_s": f"{est_s:.0f}",
        "est_core_h": f"{nodes * matrix.NODE_CORES * est_s / 3600:.1f}",
    }


def submit(
    rd: RunDir,
    *,
    jobs: Sequence[str] | None = None,
    qos: str | None = None,
    time_limit: str | None = None,
    dry_run: bool = False,
    ceiling: float | None = None,
    run: Runner = subprocess.run,
    prepare: Callable[[stacks.Resolved], Path] | None = None,
) -> int:
    """Prepare, check the budget, ``sbatch`` each job; print the lines on a dry run.

    ``time_limit`` (``H:MM:SS``) replaces the planned time limit of every submitted job,
    with its estimate and QOS purpose (:func:`with_time_limit`), for a rerun after a
    job ran out of time; ``submitted.tsv`` records the limit used.
    """
    names = list(jobs) if jobs else rd.jobs()
    plan = _plan_rows(rd.path)
    unknown = [j for j in names if j not in plan]
    if unknown:
        raise ConfigError(f"no planned job {', '.join(unknown)} (jobs: {rd.jobs()})")
    if time_limit is not None:
        plan = {j: with_time_limit(plan[j], time_limit) for j in names}
    commit, _ = matrix.source_commit()
    try:
        records.check_extends(
            rd.manifest,
            records.Pins(
                gfem={"commit": commit},
                harness={"commit": commit},
                solvers=rd.manifest.pins.solvers,
            ),
        )
    except records.RecordError as error:
        raise ConfigError(str(error)) from error
    estimate = sum(float(plan[j]["est_core_h"]) for j in names)
    files = budget.submitted_files(Path(rd.site.output_dir), [rd.path])
    spent = budget.spend(rd.site.prior_core_h, files, run)
    limit = rd.site.ceiling_core_h if ceiling is None else ceiling
    budget.check(spent, estimate, limit)
    print(
        f"budget: spend {spent.total:.1f} + estimate {estimate:.1f} core-h "
        f"<= ceiling {limit:.0f}"
    )
    lines = {j: sbatch_argv(rd, j, plan[j], qos=qos) for j in names}
    if dry_run:
        for argv in lines.values():
            print(shlex.join(argv))
        return 0
    _prepare(rd, names, prepare)
    copy_site(rd.site, rd.path)
    (rd.path / "logs").mkdir(exist_ok=True)
    tsv = rd.path / budget.SUBMITTED
    if not tsv.is_file():
        tsv.write_text(budget.SUBMITTED_HEADER + "\n")
    for job, argv in lines.items():
        done = run(argv, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            print(f"sbatch {job}: {done.stderr.strip()}", file=sys.stderr)
            return 1
        slurm_id = done.stdout.strip().splitlines()[-1].split(";")[0]
        queue = next(a for a in argv if a.startswith("--qos=")).removeprefix("--qos=")
        with tsv.open("a") as out:
            out.write(
                "\t".join(
                    [
                        job,
                        slurm_id,
                        _utc(),
                        queue,
                        plan[job]["time_limit"],
                        plan[job]["est_core_h"],
                    ]
                )
                + "\n"
            )
        for entry in rd.manifest.jobs:
            if entry.job == job:
                entry.qos = queue
                ids = entry.extra.setdefault("slurm_ids", [])
                ids.append(slurm_id)
                entry.constraint = constraint(
                    entry.constraint, rd.site.constraint or None
                )
        print(f"submitted {job}: {slurm_id} ({queue}, {plan[job]['time_limit']})")
    records.write_campaign(
        rd.manifest,
        rd.path,
        roots=rd.roots(),
        secrets=records.site_secrets(rd.site, os.environ),
    )
    return 0


def _prepare(
    rd: RunDir,
    jobs: Sequence[str],
    prepare: Callable[[stacks.Resolved], Path] | None,
) -> None:
    """Prepare each stack (pin its env manifests) and stage each case's mesh."""
    rows = [row for job in jobs for row in rd.steps(job)]
    commit = rd.manifest.pins.gfem["commit"]
    for name in sorted({r["stack"] for r in rows}):
        resolved = stacks.resolve(
            name,
            rd.site,
            rd.campaign.stack_versions.get(name),
            rd.campaign.stack_env.get(name),
            commit=commit,
        )
        if prepare is None:
            stacks.prepare(
                resolved, wheel=_wheel(), codes=matrix.campaign_codes(rd.campaign)
            )
        else:
            prepare(resolved)
        found = stacks.provenance(resolved)
        pins = rd.manifest.pins.stacks.setdefault(name, {})
        for key, part in (
            ("env_manifest_sha256", "gfem_manifest"),
            ("palace_manifest_sha256", "palace_manifest"),
        ):
            have = (found.get(part) or {}).get("sha256")
            if pins.get(key) and pins[key] != have:
                raise ConfigError(
                    f"stack {name}: {key} {have} != pinned {pins[key]}: the env "
                    "changed since this campaign's first submit"
                )
            pins[key] = have
    refused = stage_meshes(rd, sorted({r["case"] for r in rows}))
    for case, reason in refused.items():
        print(f"mesh of {case} refused (its runs will be refused): {reason}")


# --- status and close -----------------------------------------------------------------


def submitted(rundir: Path) -> list[dict[str, str]]:
    """Rows of ``submitted.tsv`` (job, slurm_id, ...)."""
    path = rundir / budget.SUBMITTED
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"), strict=False)) for ln in lines[1:] if ln]


def _latest_ids(rundir: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    for row in submitted(rundir):
        latest[row["job"]] = row["slurm_id"]
    return latest


def status(rd: RunDir, run: Runner = subprocess.run) -> int:
    """Print per job its Slurm state and its records by status."""
    ids = _latest_ids(rd.path)
    try:
        facts = budget.sacct_jobs(ids.values(), run) if ids else {}
    except budget.BudgetError:
        facts = {}
    for job in rd.jobs():
        steps = rd.steps(job)
        counts: dict[str, int] = {}
        for row in steps:
            found = _load(_record_path(rd.path, row["name"]))
            key = found.status if found else "missing"
            counts[key] = counts.get(key, 0) + 1
        slurm = ids.get(job)
        state = facts.get(slurm or "", {}).get("state", "") if slurm else ""
        where = f"{slurm} {state}".strip() if slurm else "not submitted"
        summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"{job:<24} {where:<24} {len(steps)} runs: {summary}")
    return 0


def _observed_ndof(r: records.SuiteRecord) -> int | None:
    """The DOF count a run reported, else the one the DOF guard read in stdout."""
    if r.case.ndof is not None:
        return r.case.ndof
    seen = (r.extra.get("observations") or {}).get("dof_seen_in_stdout")
    return seen if isinstance(seen, int) else None


def _dof_equal(rows: Sequence[records.SuiteRecord]) -> tuple[str, dict[str, Any]]:
    """Per case, every DOF count a run observed, whatever its status.

    Any two counts that differ on a case fail it, a ``dof_mismatch`` run included;
    a case passes once two codes observed the same count, else it is skipped.
    """
    per_case: dict[str, dict[str, set[int]]] = {}
    for r in rows:
        ndof = _observed_ndof(r)
        if ndof is not None:
            code = r.run.solver.partition(":")[0]
            per_case.setdefault(r.run.case, {}).setdefault(code, set()).add(ndof)
    detail: dict[str, Any] = {}
    verdicts = []
    for case, codes in sorted(per_case.items()):
        same = len({n for v in codes.values() for n in v}) == 1
        if same and len(codes) < 2:
            continue
        counts = {c: sorted(v) for c, v in codes.items()}
        detail[case] = {"result": "pass" if same else "fail", "ndof": counts}
        verdicts.append(detail[case]["result"])
    verdict = "fail" if "fail" in verdicts else "pass" if verdicts else "skipped"
    return verdict, detail


def _stamp(
    rd: RunDir,
    ids: Mapping[str, str],
    facts: Mapping[str, Mapping[str, str]],
    environ: Mapping[str, str],
) -> tuple[list[records.SuiteRecord], list[str]]:
    """Every planned row's record, and the rows still without one.

    Stubs of ended jobs become ``killed``, rows they never reached ``not_started``.
    """
    rows: list[records.SuiteRecord] = []
    missing: list[str] = []
    for job in rd.jobs():
        slurm = ids.get(job)
        state = facts.get(slurm, {}).get("state") if slurm else None
        ended = state in ENDED_STATES
        for row in rd.steps(job):
            found = _load(_record_path(rd.path, row["name"]))
            if found is not None and found.status == "started" and ended:
                found.status, found.slurm_state = "killed", state
                found.reason = f"slurm: job ended {state} during the run"
                write(rd, found, environ)
            elif found is None and ended:
                ctx = Context(local=False, job_id=slurm or LOCAL, environ=environ)
                found = _planned(rd, row, ctx)
                found.status, found.slurm_state = "not_started", state
                found.reason = f"job ended {state} before this line"
                write(rd, found, environ)
            if found is None:
                missing.append(row["name"])
            else:
                rows.append(_load(_record_path(rd.path, row["name"])) or found)
    return rows, missing


def campaign_checks(
    rd: RunDir, rows: Sequence[records.SuiteRecord], missing: Sequence[str]
) -> dict[str, Any]:
    """The campaign checks.

    Gates ``complete``, ``provenance`` and ``dof_equal``; observations ``node_class``
    and ``harness_failures``.
    """
    checks: dict[str, Any] = {}
    unfinished = [r.name for r in rows if r.status not in records.FINAL_STATUSES]
    checks["complete"] = "fail" if missing or unfinished else "pass"
    if missing or unfinished:
        checks["complete_detail"] = {"missing": list(missing), "unfinished": unfinished}
    problems = records.mixed(rd.manifest, rows)
    bad = [r.name for r in rows if r.checks.get("provenance") == "fail"]
    checks["provenance"] = "fail" if problems or bad else "pass"
    if problems or bad:
        checks["provenance_detail"] = {"mixed": problems, "records": bad}
    checks["dof_equal"], detail = _dof_equal(rows)
    if detail:
        checks["dof_equal_detail"] = detail
    classes = sorted({c for r in rows if (c := r.provenance.machine.node_class)})
    checks["node_class"] = {"observed": classes, "planned": rd.campaign.node_class}
    checks["harness_failures"] = [r.name for r in rows if r.failure_class == "harness"]
    return checks


def _fill_jobs(
    rd: RunDir, ids: Mapping[str, str], facts: Mapping[str, Mapping[str, str]]
) -> None:
    """Each submitted job's state, elapsed time, nodes and billed core-hours."""
    by_name = {j.job: j for j in rd.manifest.jobs}
    for job, slurm in ids.items():
        entry = by_name.get(job)
        fact = facts.get(slurm)
        if entry is None or fact is None:
            continue
        entry.state = fact["state"]
        entry.elapsed_s = float(fact["elapsed_raw"]) if fact["elapsed_raw"] else None
        entry.nodes = [fact["nodelist"]] if fact["nodelist"] else []
        entry.qos = fact["qos"] or entry.qos
        billed = [
            budget.billed_core_h(int(facts[i]["cputime_raw"]))
            for i in entry.extra.get("slurm_ids", [slurm])
            if i in facts and facts[i]["cputime_raw"].isdigit()
        ]
        entry.billed_core_h = round(sum(billed), 3) if billed else None


def close(
    rd: RunDir, run: Runner = subprocess.run, environ: Mapping[str, str] | None = None
) -> int:
    """Stamp unfinished rows, fill the manifest, check the campaign gates.

    Exits 1 when ``complete``, ``provenance`` or ``dof_equal`` fails, else 0.
    """
    environ = dict(os.environ if environ is None else environ)
    ids = _latest_ids(rd.path)
    all_ids = [row["slurm_id"] for row in submitted(rd.path)]
    facts = budget.sacct_jobs(all_ids, run) if all_ids else {}
    rows, missing = _stamp(rd, ids, facts, environ)
    for scored in score.score_campaign(rows, rd.case):
        write(rd, scored, environ)
    checks = campaign_checks(rd, rows, missing)
    _fill_jobs(rd, ids, facts)
    rd.manifest.records = records.record_hashes(rd.path)
    rd.manifest.checks = checks
    rd.manifest.closed_utc = _utc()
    records.write_campaign(
        rd.manifest,
        rd.path,
        roots=rd.roots(),
        secrets=records.site_secrets(rd.site, environ),
    )
    billed = sum(j.billed_core_h or 0.0 for j in rd.manifest.jobs)
    print(f"campaign {rd.manifest.id}: {len(rows)} records, {len(missing)} missing")
    for gate in CAMPAIGN_GATES:
        print(f"  {gate}: {checks[gate]}")
    harness = len(checks["harness_failures"])
    print(f"  harness failures: {harness}; billed {billed:.1f} core-h")
    return 1 if any(checks[g] == "fail" for g in CAMPAIGN_GATES) else 0


# --- the job script's entry -----------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m benchmarks.harness.slurm start RUNDIR JOB`` (job.sbatch only)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] != "start":
        print("usage: python -m benchmarks.harness.slurm start RUNDIR JOB")
        return 2
    rd = load_rundir(Path(args[1]))
    return job_start(rd, args[2], Context.from_env(local=False))


if __name__ == "__main__":
    sys.exit(main())
