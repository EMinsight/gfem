"""Launch one command of a run and measure it (ADRs 0410-0412).

Every run is one process tree started here: on the cluster the adapter's srun line
(each run its own Slurm step, one placement rule for every code and rank count), at
one rank the stack's ``taskset`` fallback, and under ``run --local`` taskset or
Hydra (``-launcher fork -bind-to core`` under taskset), recorded ``machine: local``.

- **Env.** The step's env script is sourced in a subshell (:func:`sourced_env`), and
  the adapter's ``Command.env`` is merged over it (:func:`command_env`; a
  ``PYTHONPATH`` is put in front of one already there). Nothing else of the caller's
  shell reaches a run: the script unsets thread and affinity variables itself.
- **Watchdog.** A wall cap (the run's ``wall_cap_s``, cut to the job's time left)
  ends the run as ``timeout``; a memory cap, when set, as ``memory_watchdog``. The
  process group gets SIGTERM, then SIGKILL 10 s later.
- **Sampler.** Every 0.1 s over the first 5 s, then every second, the run's pids
  (``scontrol listpids <job>`` on Slurm, the newest numbered step; else the process
  tree) have their ``Pss`` summed from
  ``/proc/<pid>/smaps_rollup`` (``pss_peak_sum_gib``); once every rank is up, the
  placement check reads each thread's ``Cpus_allowed_list`` and CPU time (each thread
  counted at its latest read); every 10 s a ``ps -eLo``
  snapshot of the run's threads is logged and summarised as the most busy threads
  seen on one physical core (an observation). stdout is tailed for the adapter's
  DOF guard (``Command.watch``): a count off the pin ends the run at once.

Standard library only (ADR 0400).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.adapters import Layout
from benchmarks.harness.config import Runner

PROC = Path("/proc")
SYSFS_CPU = Path("/sys/devices/system/cpu")
KIB_PER_GIB = 2**20
SAMPLE_S = 1.0
START_SAMPLE_S = 0.1
"""Sampling period over a run's first :data:`START_S`, so short runs get checked."""
START_S = 5.0
SNAPSHOT_S = 10.0
KILL_GRACE_S = 10.0
WORKING_CPU_S = 0.1
"""CPU time from which a thread counts as working (idle MPI helpers stay near 0)."""
BUSY_PCPU = 50.0
"""A thread above this ``pcpu`` counts as busy in the thread snapshot."""
SACCT_MIN_S = 60.0
"""sacct samples every 60 s: its ``MaxRSS`` is kept only for longer runs."""
STEP_DISABLED = "step creation still disabled"
"""srun's message when Slurm stopped the job's steps: the job stops."""
KILLED_STATES = ("OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED", "BOOT_FAIL")
"""Slurm step states that make a record ``killed``."""
HYDRA_ARGS = ("-launcher", "fork", "-bind-to", "core")


# --- env ------------------------------------------------------------------------------


def sourced_env(script: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment after ``. script`` in a bash subshell started with ``base``.

    The script's own output is discarded; a failing line (a module the node does
    not have) does not stop it, as in a batch script.
    """
    done = subprocess.run(
        ["bash", "-c", '. "$1" >/dev/null 2>&1; env -0', "bash", str(script)],
        env=dict(base),
        capture_output=True,
        check=True,
    )
    out = {}
    for item in done.stdout.decode(errors="replace").split("\0"):
        name, sep, value = item.partition("=")
        if sep and name and name != "_":
            out[name] = value
    return out


def command_env(sourced: Mapping[str, str], extra: Mapping[str, str]) -> dict[str, str]:
    """``extra`` (the adapter's ``Command.env``) over ``sourced``.

    A ``PYTHONPATH`` goes in front of one already set, so the shipped tree comes first
    and a path the caller needs (a source checkout under ``run --local``) stays.
    """
    env = dict(sourced)
    for name, value in extra.items():
        if name == "PYTHONPATH" and env.get(name):
            env[name] = os.pathsep.join([value, env[name]])
        else:
            env[name] = value
    return env


# --- launch lines ---------------------------------------------------------------------


def cpu_list(cpus: Sequence[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    out: list[str] = []
    ordered = sorted(set(cpus))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        out.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(out)


def parse_cpu_list(text: str) -> list[int]:
    """``"0-2,5"`` (or ``0,112``) -> ``[0, 1, 2, 5]``."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def block(cores: int, available: Sequence[int], sysfs: Path = SYSFS_CPU) -> list[int]:
    """The first ``cores`` physical cores of ``available``, one CPU each."""
    seen: dict[int, int] = {}
    for cpu in sorted(available):
        seen.setdefault(core_of(cpu, sysfs), cpu)
    if len(seen) < cores:
        raise ValueError(f"{cores} cores asked, {len(seen)} available")
    return [seen[c] for c in sorted(seen)[:cores]]


def taskset_argv(program: Sequence[str], cpus: Sequence[int]) -> tuple[str, ...]:
    """``taskset -c <block> program``: the one-rank fallback and ``run --local``."""
    return ("taskset", "-c", cpu_list(cpus), *program)


def hydra_argv(
    program: Sequence[str], layout: Layout, cpus: Sequence[int], mpiexec: str
) -> tuple[str, ...]:
    """Hydra under taskset with rank binding: ``run --local`` only (ADR 0412)."""
    return (
        *taskset_argv((mpiexec, "-n", str(layout.ranks), *HYDRA_ARGS), cpus),
        *program,
    )


# --- placement ------------------------------------------------------------------------


def core_of(cpu: int, sysfs: Path = SYSFS_CPU) -> int:
    """The physical core of logical CPU ``cpu``: the lowest of its siblings.

    On an MN5 GPP node CPU 112-223 are the hyperthread siblings of 0-111.
    """
    path = sysfs / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        return min(parse_cpu_list(path.read_text()))
    except (OSError, ValueError):
        return cpu


@dataclass
class Placement:
    """The placement check of one run (record ``checks.placement``)."""

    status: str
    """``pass``, ``fail`` or ``skipped`` (no rank was seen)."""
    masks: list[str] = field(default_factory=list)
    """Each rank's ``Cpus_allowed_list``, rank order."""
    reason: str | None = None


def check_placement(
    masks: Sequence[str], threads: int, sysfs: Path = SYSFS_CPU
) -> Placement:
    """Each rank's mask holds exactly ``threads`` physical cores, pairwise disjoint."""
    if not masks:
        return Placement("skipped", reason="no rank seen")
    problems = []
    taken: dict[int, int] = {}
    for rank, mask in enumerate(masks):
        cores = {core_of(cpu, sysfs) for cpu in parse_cpu_list(mask)}
        if len(cores) != threads:
            problems.append(f"rank {rank} mask {mask} holds {len(cores)} cores")
        for core in sorted(cores):
            if core in taken:
                problems.append(
                    f"rank {rank} shares core {core} with rank {taken[core]}"
                )
                break
            taken[core] = rank
    if problems:
        return Placement("fail", list(masks), "; ".join(problems))
    return Placement("pass", list(masks))


@dataclass(frozen=True)
class ThreadSeen:
    """One thread of a rank as the sampler last read it."""

    mask: str
    """Its ``Cpus_allowed_list``."""
    cpu_s: float
    """User plus system CPU time so far."""


def rank_placement(
    process: Sequence[str],
    threads: Sequence[Mapping[int, ThreadSeen]],
    threads_per_rank: int,
    sysfs: Path,
) -> Placement:
    """The placement check on the process masks, else on the working threads.

    A rank's process (main-thread) mask is its allotment unless ``OMP_PROC_BIND``
    narrowed the main thread to one core; then the masks of its working threads
    (``threads``, tid to the latest read, with at least :data:`WORKING_CPU_S`) are.
    Idle helper threads (MPI's sit on the main thread's core, other libraries' carry
    stray masks) are left out. Either view passing passes.

    Ranks whose working threads share a core, or span more cores than their share,
    fail; so does a rank with ``threads_per_rank`` working threads on fewer cores.
    A rank with fewer cores only because fewer of its threads were seen working
    (a run over before its OpenMP threads were sampled) is ``skipped`` (unverified).
    """
    first = check_placement(process, threads_per_rank, sysfs)
    if first.status == "pass":
        return first
    working = [
        {tid: t for tid, t in rank.items() if t.cpu_s >= WORKING_CPU_S}
        for rank in threads
    ]
    union = [
        cpu_list(sorted({c for t in rank.values() for c in parse_cpu_list(t.mask)}))
        for rank in working
    ]
    second = check_placement(union, threads_per_rank, sysfs)
    if second.status == "pass":
        return second
    problems, unseen = [], []
    taken: dict[int, int] = {}
    for rank, (mask, seen) in enumerate(zip(union, working, strict=True)):
        cores = {core_of(cpu, sysfs) for cpu in parse_cpu_list(mask)}
        for core in sorted(cores):
            if core in taken:
                problems.append(
                    f"rank {rank} shares core {core} with rank {taken[core]}"
                )
                break
            taken[core] = rank
        if len(cores) > threads_per_rank:
            problems.append(f"rank {rank}: working threads span {len(cores)} cores")
        elif len(cores) < threads_per_rank and len(seen) >= threads_per_rank:
            problems.append(
                f"rank {rank}: {len(seen)} working threads on {len(cores)} cores"
            )
        elif len(cores) < threads_per_rank:
            unseen.append(
                f"rank {rank}: saw {len(seen)} of {threads_per_rank} working threads"
            )
    if problems:
        return Placement("fail", union, "; ".join(problems))
    return Placement("skipped", union, "; ".join(unseen) + " (run too short to verify)")


def busy_threads_per_core(
    ps_text: str, pids: Sequence[int], sysfs: Path = SYSFS_CPU
) -> int:
    """Most busy threads of ``pids`` on one physical core in a ``ps -eLo`` snapshot.

    Columns: ``pid tid psr pcpu comm``.
    """
    wanted = set(pids)
    per_core: dict[int, int] = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit() or int(parts[0]) not in wanted:
            continue
        try:
            psr, pcpu = int(parts[2]), float(parts[3])
        except ValueError:
            continue
        if pcpu > BUSY_PCPU:
            core = core_of(psr, sysfs)
            per_core[core] = per_core.get(core, 0) + 1
    return max(per_core.values(), default=0)


# --- pids -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StepPids:
    """The pids of one Slurm step on this node."""

    step: str | None
    ranks: tuple[int, ...]
    """Task pids in task order."""
    all: tuple[int, ...]


def listpids(text: str) -> StepPids:
    """The newest numbered step of ``scontrol listpids <job>`` output.

    Columns: ``PID JOBID STEPID LOCALID GLOBALID``; ``LOCALID`` is ``-`` for the
    step daemon's own helpers.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit() and parts[2].isdigit():
            rows.append(parts)
    if not rows:
        return StepPids(None, (), ())
    step = max(int(r[2]) for r in rows)
    mine = [r for r in rows if int(r[2]) == step]
    tasks = sorted((int(r[4]), int(r[0])) for r in mine if r[4].isdigit())
    return StepPids(
        str(step), tuple(pid for _, pid in tasks), tuple(int(r[0]) for r in mine)
    )


def _children(proc: Path) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # comm may hold spaces and parentheses: fields resume after the last ")"
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 1:
            kids.setdefault(int(fields[1]), []).append(int(entry.name))
    return kids


def process_tree(root: int, proc: Path = PROC) -> list[int]:
    """``root`` and its descendants (the local runner's process-tree poll)."""
    kids = _children(proc)
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo += kids.get(pid, [])
    return out


def _comm(pid: int, proc: Path) -> str:
    try:
        return (proc / str(pid) / "comm").read_text().strip()
    except OSError:
        return ""


def pss_kib(pid: int, proc: Path = PROC) -> int:
    """``Pss`` of ``pid`` in KiB (0 when it is gone or unreadable)."""
    try:
        text = (proc / str(pid) / "smaps_rollup").read_text()
    except OSError:
        return 0
    match = re.search(r"^Pss:\s+(\d+) kB", text, re.MULTILINE)
    return int(match[1]) if match else 0


def _mask(status: Path) -> list[int] | None:
    try:
        text = status.read_text()
    except OSError:
        return None
    match = re.search(r"^Cpus_allowed_list:\s*(\S+)", text, re.MULTILINE)
    return parse_cpu_list(match[1]) if match else None


def process_mask(pid: int, proc: Path = PROC) -> str | None:
    """``Cpus_allowed_list`` of ``pid``'s main thread."""
    own = _mask(proc / str(pid) / "status")
    return None if own is None else cpu_list(own)


def _cpu_s(stat: Path, ticks_per_s: float) -> float | None:
    try:
        text = stat.read_text()
    except OSError:
        return None
    # comm may hold spaces and parentheses: fields resume after the last ")"
    fields = text[text.rfind(")") + 2 :].split()
    try:
        return (int(fields[11]) + int(fields[12])) / ticks_per_s
    except (IndexError, ValueError):
        return None


def thread_states(
    pid: int, proc: Path = PROC, ticks_per_s: float | None = None
) -> dict[int, ThreadSeen]:
    """Each thread of ``pid``: its ``Cpus_allowed_list`` and CPU time (empty if gone).

    ``OMP_PROC_BIND`` narrows each thread (the main one to the first core), so the
    rank's allotment is what its working threads together may use.
    """
    ticks = ticks_per_s or float(os.sysconf("SC_CLK_TCK"))
    found = {}
    for task in sorted((proc / str(pid) / "task").glob("*")):
        mask = _mask(task / "status")
        cpu_s = _cpu_s(task / "stat", ticks)
        if task.name.isdigit() and mask is not None and cpu_s is not None:
            found[int(task.name)] = ThreadSeen(cpu_list(mask), cpu_s)
    return found


# --- one command ----------------------------------------------------------------------


@dataclass
class Measurement:
    """What the harness saw of one command."""

    returncode: int | None
    """``None`` when the harness ended it."""
    wall_s: float
    started_utc: str
    finished_utc: str
    ended_by: str | None = None
    """``timeout``, ``memory_watchdog`` or ``dof`` when the harness ended it."""
    dof_seen: int | None = None
    pss_peak_sum_gib: float | None = None
    placement: Placement = field(default_factory=lambda: Placement("skipped"))
    busy_threads_per_core: int | None = None
    step: str | None = None
    """The Slurm step id of an srun launch."""
    step_disabled: bool = False


@dataclass
class Sampler:
    """Where the sampler finds a run's pids, and how it reads them."""

    slurm_job: str | None = None
    """Look the pids up with ``scontrol listpids`` (an srun launch in a job)."""
    proc: Path = PROC
    sysfs: Path = SYSFS_CPU
    run: Runner = subprocess.run
    sample_s: float = SAMPLE_S
    start_sample_s: float = START_SAMPLE_S
    start_s: float = START_S
    snapshot_s: float = SNAPSHOT_S
    exclude_comm: tuple[str, ...] = ("mpiexec", "hydra_pmi_proxy", "taskset")
    """Launcher processes that are no rank (local Hydra)."""

    def pids(self, root: int) -> StepPids:
        """The run's step id, rank pids and every pid."""
        if self.slurm_job is not None:
            done = self.run(
                ["scontrol", "listpids", self.slurm_job],
                capture_output=True,
                text=True,
                check=False,
            )
            return listpids(done.stdout)
        tree = process_tree(root, self.proc)
        ranks = [p for p in tree if _comm(p, self.proc) not in self.exclude_comm]
        return StepPids(None, tuple(sorted(ranks)), tuple(tree))

    def snapshot(self) -> str:
        """``ps -eLo pid,tid,psr,pcpu,comm``."""
        done = self.run(
            ["ps", "-eLo", "pid,tid,psr,pcpu,comm"],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.stdout


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _kill(process: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            continue
        return


class _Tail:
    """New complete lines of a growing file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.rest = ""

    def lines(self) -> list[str]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return []
        self.offset += len(data)
        text = self.rest + data.decode(errors="replace")
        *done, self.rest = text.split("\n")
        return done


@dataclass
class _Limits:
    """When the harness ends a run, and what it looks at while it runs."""

    layout: Layout
    cap_s: float
    watch: Callable[[str], int | None] | None = None
    pinned_ndof: int | None = None
    mem_cap_gib: float | None = None
    threads_log: Path | None = None


class _Watch:
    """Sampling passes over a running process (see :meth:`period`)."""

    def __init__(self, sampler: Sampler, limits: _Limits, stdout: Path) -> None:
        self.sampler = sampler
        self.limits = limits
        self.tail = _Tail(stdout)
        self.t0 = time.monotonic()
        self.next_snapshot = self.t0
        self.peak_kib = 0
        self.threads: dict[int, dict[int, ThreadSeen]] = {}
        started = _utc()
        self.m = Measurement(None, 0.0, started, started)

    def period(self) -> float:
        """Seconds to the next sample: :data:`START_SAMPLE_S` early in the run."""
        sampler = self.sampler
        if time.monotonic() - self.t0 < sampler.start_s:
            return min(sampler.start_sample_s, sampler.sample_s)
        return sampler.sample_s

    def sample(self, root: int) -> None:
        """Pids, PSS, placement once, a thread snapshot every 10 s."""
        sampler, m = self.sampler, self.m
        pids = sampler.pids(root)
        m.step = pids.step or m.step
        pss = sum(pss_kib(p, sampler.proc) for p in pids.all)
        self.peak_kib = max(self.peak_kib, pss)
        layout = self.limits.layout
        if m.placement.status != "pass" and len(pids.ranks) == layout.ranks:
            # each thread's latest read: OMP threads start after the main thread
            # binds, and a task read before srun bound it shows the step's cpuset
            main = [process_mask(p, sampler.proc) for p in pids.ranks]
            if all(main):
                for rank, pid in enumerate(pids.ranks):
                    seen = self.threads.setdefault(rank, {})
                    seen.update(thread_states(pid, sampler.proc))
                m.placement = rank_placement(
                    [x for x in main if x],
                    [self.threads[r] for r in range(len(main))],
                    layout.threads,
                    sampler.sysfs,
                )
        now = time.monotonic()
        if now >= self.next_snapshot and pids.all:
            text = sampler.snapshot()
            busy = busy_threads_per_core(text, pids.all, sampler.sysfs)
            m.busy_threads_per_core = max(m.busy_threads_per_core or 0, busy)
            if self.limits.threads_log is not None:
                with self.limits.threads_log.open("a") as log:
                    log.write(f"# +{now - self.t0:.0f} s\n{text}")
            self.next_snapshot = now + sampler.snapshot_s

    def read(self, lines: Sequence[str]) -> None:
        """Feed stdout lines to the DOF guard."""
        watch, pinned = self.limits.watch, self.limits.pinned_ndof
        if watch is None:
            return
        for line in lines:
            n = watch(line)
            if n is not None:
                self.m.dof_seen = n
                if pinned is not None and n != pinned:
                    self.m.ended_by = self.m.ended_by or "dof"

    def verdict(self) -> str | None:
        """Why the harness ends the run now, if it does."""
        limits = self.limits
        if self.m.ended_by is not None:
            return self.m.ended_by
        if time.monotonic() - self.t0 > limits.cap_s:
            return "timeout"
        cap = limits.mem_cap_gib
        if cap is not None and self.peak_kib / KIB_PER_GIB > cap:
            return "memory_watchdog"
        return None

    def finish(self, stdout: Path) -> Measurement:
        """Close the measurement once the process is gone."""
        m = self.m
        m.wall_s = time.monotonic() - self.t0
        m.finished_utc = _utc()
        m.pss_peak_sum_gib = self.peak_kib / KIB_PER_GIB if self.peak_kib else None
        if m.ended_by is None:
            self.read([*self.tail.lines(), self.tail.rest])
        with contextlib.suppress(OSError):
            m.step_disabled = STEP_DISABLED in stdout.read_text(errors="replace")
        return m


def measure(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: Path,
    layout: Layout,
    cap_s: float,
    watch: Callable[[str], int | None] | None = None,
    pinned_ndof: int | None = None,
    mem_cap_gib: float | None = None,
    sampler: Sampler | None = None,
    threads_log: Path | None = None,
) -> Measurement:
    """Run ``argv`` in ``cwd`` with ``env``, stdout and stderr to ``stdout``; measure.

    ``cap_s`` ends it as ``timeout``. ``watch`` maps a stdout line to a DOF count; a
    count other than ``pinned_ndof`` ends it at once (``ended_by = "dof"``).
    """
    sampler = sampler or Sampler()
    limits = _Limits(layout, cap_s, watch, pinned_ndof, mem_cap_gib, threads_log)
    state = _Watch(sampler, limits, stdout)
    with stdout.open("w") as out:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        while True:
            try:
                state.m.returncode = process.wait(timeout=state.period())
                break
            except subprocess.TimeoutExpired:
                pass
            state.sample(process.pid)
            state.read(state.tail.lines())
            state.m.ended_by = state.verdict()
            if state.m.ended_by is not None:
                _kill(process)
                break
    return state.finish(stdout)


# --- sacct ----------------------------------------------------------------------------


def _gib(text: str) -> float | None:
    match = re.match(r"^([\d.]+)([KMGT]?)$", text.strip())
    if not match:
        return None
    scale = {"": 2**-30, "K": 2**-20, "M": 2**-10, "G": 1.0, "T": 2**10}[match[2]]
    return float(match[1]) * scale


def step_facts(
    job: str,
    step: str,
    run: Runner = subprocess.run,
    tries: int = 6,
    wait_s: float = 5.0,
) -> tuple[str | None, float | None]:
    """``(State, MaxRSS GiB)`` of step ``job.step`` from sacct, once it settled."""
    state, rss = None, None
    for i in range(tries):
        done = run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                f"{job}.{step}",
                "--format=State,MaxRSS",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        line = next((ln for ln in done.stdout.splitlines() if ln.strip()), "")
        fields = line.split("|")
        state = fields[0].split()[0] if fields and fields[0].strip() else None
        rss = _gib(fields[1]) if len(fields) > 1 and fields[1].strip() else None
        if state not in (None, "RUNNING", "COMPLETING", "PENDING"):
            break
        if i + 1 < tries:
            time.sleep(wait_s)
    return state, rss
