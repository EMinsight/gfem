"""The gfem solver adapter (``gfem:direct``, ``gfem:petsc-mumps``, ``gfem:bddc``).

gfem runs as ``python -m benchmarks.adapters.gfem_worker run.json`` in the stack's
gfem env, under the suite's srun line at every rank count (``--mpi=none`` at one
rank, the stack's plugin above; ADR 0412), or bare when the stack's one-rank launch
path is ``taskset``. :meth:`GfemAdapter.prepare` writes ``run.json``; the worker
writes ``outcome.json`` (status, quantities, per-rank mesh read, span, ``VmHWM`` and
threads) and one ``phases.rank<k>.jsonl`` of ``gfem.timing`` records per rank (ADR
0190); :meth:`GfemAdapter.collect` reduces them per bucket by max over ranks, with
rank 0's values beside.

Pins (ADR 0411): ``linear_solver`` per kind, ``tolerance`` on the iterative kind
only, ``eigen_tolerance`` 1e-10, no field output; everything else at gfem's
defaults (``threads=None`` follows the bound mask, one frequency worker).

Standard library only (ADR 0400): the worker is the only module that imports gfem.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.adapters import (
    Command,
    Kind,
    Launch,
    Layout,
    Measured,
    Outcome,
    Run,
    RunRefusedError,
    launched,
)
from benchmarks.harness.case import Case
from benchmarks.stacks import Resolved

CODE = "gfem"
WORKER = "benchmarks.adapters.gfem_worker"
ROOT = Path(__file__).resolve().parents[2]
"""The tree holding ``benchmarks/``: the worker's ``PYTHONPATH``."""
RUN_JSON = "run.json"
OUTCOME_JSON = "outcome.json"
JOB_JSON = "job.json"
STDOUT = "stdout.txt"
VERSION_STDOUT = "gfem-version.txt"
PHASE_FILE = "phases.rank{rank}.jsonl"
NODE_CORES = 112
"""Cores of one node: one gfem process never spans two."""
RANK_THREADS = 4
"""Threads per rank of the default layout above one node (28 x 4 per node)."""
EIGEN_TOLERANCE = 1e-10
BOUNDARY_KINDS = ("pec", "conductor", "absorbing", "periodic")
PORT_KINDS = ("lumped",)
TIMED = ("discretize", "assemble", "setup", "solve")
"""``gfem.timing`` phases; each fills the bucket ``<phase>_s``."""
BUCKETS = (
    "mesh_read_s",
    "discretize_s",
    "assemble_s",
    "setup_s",
    "solve_s",
    "estimate_s",
    "postpro_s",
    "rom_s",
    "other_s",
)
KIB_PER_GIB = 2**20


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_phases(workdir: Path) -> dict[int, list[dict[str, Any]]]:
    """The ``gfem.timing`` records of each rank, from ``phases.rank<k>.jsonl``."""
    ranks: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(workdir.glob(PHASE_FILE.format(rank="*"))):
        rank = int(path.name.removeprefix("phases.rank").removesuffix(".jsonl"))
        lines = path.read_text().splitlines()
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a rank killed mid-line
        ranks[rank] = records
    return ranks


def rank_ndof(records: list[dict[str, Any]]) -> int | None:
    """DOFs one rank owns: its largest ``discretize`` record, never their sum.

    Every solve logs one ``discretize``; forward and backward solves, Bloch vectors
    and frequency workers each log their own over the same space.
    """
    counts = [r["ndof"] for r in records if r.get("phase") == "discretize"]
    return max(counts) if counts else None


def total_ndof(ranks: Mapping[int, list[dict[str, Any]]]) -> int | None:
    """The problem's DOFs: the ranks' owned counts summed (rank 0 may own none)."""
    counts = [n for n in map(rank_ndof, ranks.values()) if n is not None]
    return sum(counts) if counts else None


def _buckets(records: list[dict[str, Any]], info: Mapping[str, Any] | None) -> dict:
    """One rank's buckets: phase sums, its mesh read and the rest of its span."""
    out: dict[str, float | None] = dict.fromkeys(BUCKETS)
    for phase in TIMED:
        seconds = [r["seconds"] for r in records if r.get("phase") == phase]
        out[f"{phase}_s"] = math.fsum(seconds) if seconds else None
    if info is not None:
        out["mesh_read_s"] = info.get("mesh_read_s")
        span = info.get("span_s")
        if span is not None:
            timed = [out[f"{p}_s"] for p in TIMED] + [out["mesh_read_s"]]
            out["other_s"] = span - math.fsum(s for s in timed if s is not None)
    return out


def _rank_refusal(case: Case, kind: str) -> str | None:
    """Why gfem cannot run ``kind`` on ``case`` above one rank, if it cannot."""
    periodic = any(b.kind == "periodic" for b in case.boundaries)
    refusals = (
        (kind == "direct", "gfem:direct (PARDISO) runs in one process only"),
        (case.problem.kind == "eigen", "an eigen solve has no distributed gfem path"),
        (
            periodic or case.problem.wave_vector is not None,
            "a periodic model has no distributed gfem path",
        ),
        (
            case.problem.sweep == "adaptive",
            "an adaptive sweep has no distributed gfem path",
        ),
        (
            kind == "bddc" and case.order == 1,
            "BDDC at order 1 is refused above one rank",
        ),
    )
    return next((reason for refused, reason in refusals if refused), None)


def _max(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _peaks(outcome: Outcome, ranks: list[dict[str, Any]]) -> None:
    """Per-rank ``VmHWM``: summed, the rank max, and the largest node's sum."""
    if not ranks or any(r.get("vmhwm_kib") is None for r in ranks):
        return
    peaks = [float(r["vmhwm_kib"]) / KIB_PER_GIB for r in ranks]
    nodes: dict[str, float] = {}
    for r, peak in zip(ranks, peaks, strict=True):
        host = str(r.get("host") or "")
        nodes[host] = nodes.get(host, 0.0) + peak
    outcome.self_peak_mem_gib = sum(peaks)
    outcome.self_peak_rank_max_gib = max(peaks)
    outcome.self_peak_largest_node_gib = max(nodes.values())


@dataclass(frozen=True)
class GfemAdapter:
    """The gfem solver adapter."""

    code: str = CODE
    kinds: Mapping[str, Kind] = field(
        default_factory=lambda: {
            "direct": Kind("direct", {"linear_solver": "direct"}),
            "petsc-mumps": Kind("direct", {"linear_solver": "petsc"}),
            "bddc": Kind("iterative", {"linear_solver": "bddc_gmres"}),
        }
    )
    direct_kind: str = "direct"
    """PARDISO, one process."""
    pins: tuple[str, ...] = (
        "linear_solver per kind",
        "the run tolerance",
        "eigen_tolerance 1e-10",
        "no field output",
    )
    timing_source: str = "gfem.timing"
    memory_metric: str = "VmHWM_sum"
    stack_keys: tuple[str, ...] = ()

    # --- plan time -------------------------------------------------------------------

    def supports(self, case: Case, kind: str, layout: Layout) -> str | None:
        """Refuses unknown kinds and inputs, and what gfem cannot run.

        That is a rank path gfem lacks, and any kind but direct on a Bloch-periodic
        eigenproblem (a non-symmetric operator; gfem raises and names direct).
        """
        if kind not in self.kinds:
            return f"gfem has no kind {kind!r} (known: {', '.join(self.kinds)})"
        unknown = sorted({b.kind for b in case.boundaries} - set(BOUNDARY_KINDS))
        if unknown:
            return f"boundary kind(s) {', '.join(unknown)} not mapped for gfem"
        ports = sorted({p.kind for p in case.ports} - set(PORT_KINDS))
        if ports:
            return f"port kind(s) {', '.join(ports)} not mapped for gfem"
        if layout.threads > NODE_CORES:
            return f"one gfem process runs on one node ({NODE_CORES} cores)"
        problem = case.problem
        bloch = problem.kind == "eigen" and problem.wave_vector is not None
        if bloch and kind != self.direct_kind:
            return (
                "a Bloch-periodic eigen solve is non-symmetric; gfem solves it "
                "with direct only"
            )
        return None if layout.ranks == 1 else _rank_refusal(case, kind)

    def default_layout(self, case: Case, kind: str, budget: int) -> Layout:
        """One process (1 x c) on one node; above a node 28 x 4 per node.

        Kinds and problems without a rank path stay 1 x c at every budget, which
        :meth:`supports` refuses above a node (the cell is then ``refused``).
        """
        if budget <= NODE_CORES or budget % RANK_THREADS:
            return Layout(1, budget)
        wide = Layout(budget // RANK_THREADS, RANK_THREADS)
        if self.supports(case, kind, wide) is not None:
            return Layout(1, budget)
        return wide

    # --- run time --------------------------------------------------------------------

    def _command(self, stack: Resolved, layout: Layout, *args: str, **kw: Any):
        section = stack.description.codes[CODE]
        mpi = "none" if layout.ranks == 1 else section.mpi or "pmix"
        program = ("python", "-m", WORKER, *args)
        argv, launch = launched(program, layout, mpi, section.launch_r1)
        return Command(
            argv=argv,
            program=program,
            env={"PYTHONPATH": str(ROOT)},
            launch=launch,
            **kw,
        )

    def version(self, stack: Resolved) -> Launch:
        """The worker's ``--version`` (gfem and its libraries as JSON), run bare."""
        del stack  # the env script puts the stack's gfem env first on PATH
        program = ("python", "-m", WORKER, "--version")
        command = Command(
            argv=program,
            program=program,
            env={"PYTHONPATH": str(ROOT)},
            phase="version",
            stdout=VERSION_STDOUT,
            launch="taskset",
        )
        return Launch(commands=(command,))

    def render(self, run: Run) -> dict[str, Any]:
        """The ``run.json`` the worker reads."""
        options: dict[str, Any] = {
            "order": run.case.order,
            **self.kinds[run.kind].settings,
        }
        if self.kinds[run.kind].klass == "iterative":
            if run.tolerance is None:
                raise RunRefusedError(f"{run.solver} needs a run tolerance")
            options["tolerance"] = run.tolerance
        options["eigen_tolerance"] = EIGEN_TOLERANCE
        return {
            "format": 1,
            "solver": run.solver,
            "case": run.case.to_dict(),
            "mesh": str(run.mesh),
            "options": options,
            "layout": {"ranks": run.layout.ranks, "threads": run.layout.threads},
            "pinned_ndof": run.pinned_ndof,
        }

    def prepare(self, run: Run) -> Launch:
        """Write ``run.json`` and return the worker's launch line."""
        reason = self.supports(run.case, run.kind, run.layout)
        if reason is not None:
            raise RunRefusedError(reason)
        run.workdir.mkdir(parents=True, exist_ok=True)
        rendered = self.render(run)
        (run.workdir / RUN_JSON).write_text(json.dumps(rendered, indent=1) + "\n")
        command = self._command(run.stack, run.layout, RUN_JSON, stdout=STDOUT)
        return Launch(commands=(command,))

    def collect(self, run: Run, measured: Measured) -> Outcome:
        """Reduce the worker's outputs: max over ranks per bucket, rank 0 beside."""
        workdir = run.workdir
        written = _read_json(workdir / OUTCOME_JSON)
        ranks = read_phases(workdir)
        code = measured.results[0].returncode if measured.results else None
        info = {r["rank"]: r for r in (written or {}).get("ranks", [])}
        per_rank = {
            k: _buckets(ranks.get(k, []), info.get(k))
            for k in sorted(set(ranks) | set(info))
        }
        outcome = Outcome(status="ok", ndof=total_ndof(ranks))
        outcome.phases = {b: _max([p[b] for p in per_rank.values()]) for b in BUCKETS}
        outcome.rank0 = dict(per_rank.get(0, dict.fromkeys(BUCKETS)))
        outcome.rank_reduction = "max"
        outcome.native = {"source": "gfem.timing", "per_rank": per_rank}
        outcome.work_span_s = _max([i.get("span_s") for i in info.values()])
        rank0 = ranks.get(0, [])
        solves = [r for r in rank0 if r.get("phase") == "solve"]
        outcome.setups = sum(r.get("phase") == "setup" for r in rank0)
        outcome.solves = len(solves)
        outcome.iterations = [sum(r.get("iterations") or ()) for r in solves]
        nbytes = [
            _max([r.get("nbytes") for r in records if r.get("phase") == "setup"])
            for records in ranks.values()
        ]
        if any(n is not None for n in nbytes):
            outcome.solver_nbytes = sum(n for n in nbytes if n is not None)
        _peaks(outcome, list(info.values()))
        outcome.ranks_seen = len(info) or len(ranks) or None
        threads = (written or {}).get("threads") or []
        outcome.threads_seen = max(threads) if threads else None
        versions = dict((written or {}).get("versions") or {})
        if measured.version:
            try:
                versions = {**json.loads(measured.version), **versions}
            except json.JSONDecodeError:
                versions.setdefault("query", measured.version.strip())
        outcome.provenance = {"code": CODE, "versions": versions}
        outcome.quantities = (written or {}).get("quantities") or {}
        outcome.structural = self._structural(run, outcome)
        outcome.files = [
            name
            for name in (RUN_JSON, OUTCOME_JSON, STDOUT, JOB_JSON)
            if (workdir / name).exists()
        ] + [PHASE_FILE.format(rank=k) for k in sorted(ranks)]
        self._status(run, outcome, written, code)
        return outcome

    @staticmethod
    def _structural(run: Run, outcome: Outcome) -> dict[str, Any]:
        structural: dict[str, Any] = {
            "setups": outcome.setups,
            "solves": outcome.solves,
        }
        if run.case.problem.kind == "driven":
            structural["excitations"] = len(run.case.problem.excite)
        if run.kind == "direct":
            structural["direct_arm"] = "PARDISO (MKL), one process"
        elif run.kind == "petsc-mumps":
            structural["direct_arm"] = "PETSc MUMPS"
        else:
            structural["bddc_build"] = (
                "NGSolve builds the BDDC preconditioner while it assembles the "
                "first matrix of a setup, so the build is timed inside assemble"
            )
        return structural

    @staticmethod
    def _status(
        run: Run, outcome: Outcome, written: dict[str, Any] | None, code: int | None
    ) -> None:
        version = outcome.provenance["versions"].get("gfem", "?")
        pinned = run.pinned_ndof
        shape = _shape(run.case, outcome.quantities)
        complete = written is not None and written.get("status") == "ok" and not shape
        outcome.checks["complete"] = "pass" if complete else "fail"
        if shape and outcome.quantities:
            outcome.native["output_shape"] = shape
        if pinned is not None and outcome.ndof is not None:
            ok = outcome.ndof == pinned
            outcome.checks["dof"] = "pass" if ok else "fail"
            if not ok:
                outcome.status, outcome.failure_class = "failed", "solver"
                outcome.reason = (
                    f"dof_mismatch: gfem {version} {outcome.ndof} != pinned {pinned}"
                )
                return
        if written is None:
            outcome.status, outcome.failure_class = "failed", "solver"
            outcome.reason = f"incomplete: no {OUTCOME_JSON} (exit {code})"
            return
        converged = written.get("converged")
        if converged is not None:
            outcome.checks["converged"] = converged
        if written.get("status") != "ok":
            outcome.status = "failed"
            outcome.failure_class = written.get("failure_class") or "solver"
            outcome.reason = written.get("reason") or f"exit {code}"
        elif shape:
            outcome.status, outcome.failure_class = "failed", "harness"
            outcome.reason = f"output shape: {'; '.join(shape)}"
        elif code not in (0, None):
            outcome.status, outcome.failure_class = "failed", "solver"
            outcome.reason = f"exit {code}"


def _shape(case: Case, quantities: Mapping[str, Any]) -> list[str]:
    """What the quantities miss against the case (empty when complete)."""
    if not quantities:
        return ["no quantities"]
    if case.problem.kind == "eigen":
        modes = case.quantities.modes or 0
        found = len(quantities.get("f_hz") or [])
        return [] if found >= modes else [f"{found} modes < {modes}"]
    frequencies = case.problem.frequencies()
    problems = []
    if len(quantities.get("frequencies_hz") or []) != len(frequencies):
        problems.append("frequency count differs from the case")
    s = quantities.get("s") or {}
    for to, frm in case.quantities.s:
        values = s.get(f"{to},{frm}")
        if values is None or len(values) != len(frequencies):
            problems.append(f"S {to},{frm} missing or short")
    return problems


ADAPTER = GfemAdapter()
