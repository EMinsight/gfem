"""``plan``: a campaign file expanded into jobs, an estimate and a run directory.

A campaign's axes (case x solver x stack x layout x tolerance x repeats) become
**cells**: one (case, solver, stack, layout, tolerance) each, run ``repeats`` times.

- ``best`` expands to the layout grid (:func:`grid`) minus what the adapter refuses;
  ``default`` is the adapter's default layout; ``RxT`` is taken as written. A refused
  ``default`` or ``RxT`` cell stays in the plan with its reason, once, so ``step``
  writes its ``refused`` record (ADR 0411).
- Iterative kinds run at the anchor tolerance; ``[[ladder]]`` rungs off the anchor
  run at the default layout (or the ``RxT`` the ladder names; ADR 0414). Direct kinds
  carry no tolerance.
- For each (stack, case, code) with an iterative cell and no direct one, one run of
  the code's direct kind at its default layout is added (``added_for =
  "same_code_error"``, 1 repeat; ADR 0409), unless the campaign sets
  ``same_code_direct = false`` (reference campaigns whose finest rung no direct
  solver takes).
- Cells are grouped into jobs (one per case x stack, or one per stack with ``[jobs]
  group = "stack"``) and run in rounds, round r rotated by r * ceil(|L| / repeats),
  so repeats of a cell never run back to back (:func:`rotate`).
- The estimate uses the case files' ``est_s`` hints (``"<solver>@<RxT>"``, then
  ``"<solver>"``, else the wall cap; ``staging`` is per job): a cell whose hint
  reaches its wall cap counts once (later repeats are skipped after a timeout), each
  job adds :data:`JOB_OVERHEAD_S`. Core-hours = nodes x 112 x hours (ADR 0188); the
  time limit is 1.5x the estimate, QOS ``debug`` up to 2 h, else ``long``.

:func:`write` lays out ``RUNDIR/{campaign.toml, COMMIT, plan.tsv, jobs/<job>.steps,
campaign.json}``; each steps line is one run as JSON. Standard library only (ADR 0400).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks import stacks
from benchmarks.adapters import ADAPTERS, Adapter, Layout
from benchmarks.harness import case as case_mod
from benchmarks.harness import config, records
from benchmarks.harness.case import Case
from benchmarks.harness.config import Campaign, ConfigError, Site

NODE_CORES = 112
"""Cores of one MN5 GPP node; an exclusive node bills all of them (ADR 0188)."""
SOCKET_CORES = 56
JOB_OVERHEAD_S = 420.0
"""Per job: env check, staging, sampler warm-up (300 s), machine benchmark (120 s)."""
LIMIT_FACTOR = 1.5
DEBUG_LIMIT_S = 2 * 3600.0
"""Jobs whose limit is at most this go to the ``debug`` QOS, others to ``long``."""
STAGING_KEY = "staging"
SAME_CODE_ERROR = "same_code_error"
STEPS_SUFFIX = ".steps"
ROOT = Path(__file__).resolve().parents[2]


class PlanError(ConfigError):
    """A campaign ``plan`` refuses to expand or write."""


# --- layouts ---------------------------------------------------------------------


def grid(budget: int) -> tuple[Layout, ...]:
    """Every R x T = ``budget``, ranks ascending; T divides 56 when ``budget`` > 56.

    The one-process layout 1 x c is always in; no rank straddles a socket under
    ``block:block`` (ADR 0412). Refusals per code are the adapters' (:func:`expand`).
    """
    out = []
    for ranks in range(1, budget + 1):
        if budget % ranks:
            continue
        threads = budget // ranks
        if ranks > 1 and budget > SOCKET_CORES and SOCKET_CORES % threads:
            continue
        out.append(Layout(ranks, threads))
    return tuple(out)


# --- cells, steps, jobs ------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (case, solver, stack, layout, tolerance), run ``repeats`` times."""

    case: str
    solver: str
    stack: str
    layout: str
    """Resolved ``RxT``."""
    asked: str
    """``default``, ``grid`` or ``RxT`` (the record's ``layout.asked``)."""
    tolerance: float | None
    repeats: int
    wall_cap_s: float
    est_s: float = 0.0
    """Estimated seconds of one run (0 for a refused cell)."""
    default: bool = False
    """The layout is the code's default layout for this case and kind."""
    rung: bool = False
    """An off-anchor ladder rung (its tolerance is in the run name)."""
    added_for: str | None = None
    refused: str | None = None

    @property
    def code(self) -> str:
        """The solver's code."""
        return self.solver.partition(":")[0]

    @property
    def est_total_s(self) -> float:
        """Estimated seconds of all repeats; a run reaching its cap counts once."""
        if self.refused:
            return 0.0
        if self.est_s >= self.wall_cap_s:
            return self.wall_cap_s
        return self.est_s * self.repeats

    def name(self, repeat: int) -> str:
        """The suite-record name of repeat ``repeat`` (1-based)."""
        layout = "default" if self.asked == "default" else self.layout
        return records.record_name(
            self.case,
            self.solver,
            self.stack,
            layout,
            repeat,
            rung=self.tolerance if self.rung else None,
        )


@dataclass(frozen=True)
class Step:
    """One run: a cell's repeat at its place in a job."""

    cell: Cell
    repeat: int
    job: str
    line: int
    """1-based line of ``jobs/<job>.steps``: ``step RUNDIR JOB LINE``."""
    core_budget: int

    @property
    def name(self) -> str:
        """The run's suite-record name."""
        return self.cell.name(self.repeat)

    def to_json(self) -> dict[str, Any]:
        """The steps-file line."""
        c = self.cell
        layout = Layout.parse(c.layout)
        return {
            "line": self.line,
            "job": self.job,
            "name": self.name,
            "case": c.case,
            "solver": c.solver,
            "stack": c.stack,
            "layout": c.layout,
            "asked": c.asked,
            "ranks": layout.ranks,
            "threads": layout.threads,
            "default": c.default,
            "tolerance": c.tolerance,
            "repeat": self.repeat,
            "core_budget": self.core_budget,
            "wall_cap_s": c.wall_cap_s,
            "est_s": c.est_s,
            "added_for": c.added_for,
            "refused": c.refused,
        }


@dataclass(frozen=True)
class Job:
    """One exclusive Slurm job of a plan."""

    name: str
    group: str
    cases: tuple[str, ...]
    stack: str
    steps: tuple[Step, ...]
    nodes: int
    est_s: float
    constraint: str | None = None

    @property
    def cells(self) -> int:
        """Distinct cells of the job."""
        return len({s.cell for s in self.steps})

    @property
    def est_core_h(self) -> float:
        """Billed core-hours of the estimate: nodes x 112 x hours."""
        return self.nodes * NODE_CORES * self.est_s / 3600

    @property
    def limit_s(self) -> float:
        """The job's time limit, 1.5x its estimate."""
        return LIMIT_FACTOR * self.est_s

    @property
    def limit_core_h(self) -> float:
        """Core-hours the time limit can bill."""
        return self.nodes * NODE_CORES * self.limit_s / 3600

    @property
    def qos(self) -> str:
        """QOS purpose by time limit: ``debug`` (at most 2 h) or ``long``."""
        return "debug" if self.limit_s <= DEBUG_LIMIT_S else "long"

    @property
    def time_limit(self) -> str:
        """``H:MM:00`` for sbatch ``--time``, whole minutes rounded up."""
        minutes = math.ceil(self.limit_s / 60)
        return f"{minutes // 60}:{minutes % 60:02d}:00"


@dataclass(frozen=True)
class Plan:
    """A campaign expanded at one commit."""

    campaign: Campaign
    commit: str
    cells: tuple[Cell, ...]
    jobs: tuple[Job, ...]
    resolved: dict[str, stacks.Resolved] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def runs(self) -> int:
        """Planned runs (suite records) of every job."""
        return sum(len(job.steps) for job in self.jobs)

    @property
    def est_core_h(self) -> float:
        """Estimated billed core-hours of every job."""
        return sum(job.est_core_h for job in self.jobs)

    @property
    def limit_core_h(self) -> float:
        """Core-hours the jobs' time limits can bill."""
        return sum(job.limit_core_h for job in self.jobs)

    def solver_versions(self) -> dict[str, str]:
        """Pinned solver versions: Palace from the stacks (gfem is the commit)."""
        if not any(s.startswith("palace:") for s in self.campaign.solvers):
            return {}
        versions = sorted(
            v for r in self.resolved.values() if (v := r.versions.get("palace"))
        )
        return {"palace": versions[0]} if versions else {}


# --- command-line overrides --------------------------------------------------------


def override(
    campaign: Campaign,
    *,
    cases: Sequence[str] | None = None,
    solvers: Sequence[str] | None = None,
    stacks: Sequence[str] | None = None,
    layouts: Sequence[str] | None = None,
    repeats: int | None = None,
) -> Campaign:
    """``campaign`` with axes replaced by ``plan``'s flags, validated again.

    Ladders keep only their cases and solvers still planned; a stack not in the
    campaign file runs at its description's defaults.
    """
    try:
        return _override(
            campaign,
            cases=cases,
            solvers=solvers,
            stack_names=stacks,
            layouts=layouts,
            repeats=repeats,
        )
    except PlanError:
        raise
    except ConfigError as error:
        raise PlanError(str(error)) from error


def _override(
    campaign: Campaign,
    *,
    cases: Sequence[str] | None,
    solvers: Sequence[str] | None,
    stack_names: Sequence[str] | None,
    layouts: Sequence[str] | None,
    repeats: int | None,
) -> Campaign:
    where = f"{campaign.path} (command line)"
    changes: dict[str, Any] = {}
    if cases is not None:
        known = config.known_cases(campaign.cases_dir or config.CASES_DIR)
        _registry(cases, known, "case", where)
        missing = [c for c in cases if c not in campaign.wall_cap_s]
        if missing:
            raise PlanError(f"{where}: no wall_cap_s for {', '.join(missing)}")
        changes["cases"] = tuple(cases)
        changes["wall_cap_s"] = {c: campaign.wall_cap_s[c] for c in cases}
    if solvers is not None:
        changes["solvers"] = _registry(solvers, config.SOLVERS, "solver", where)
    if stack_names is not None:
        changes["stacks"] = _registry(stack_names, config.STACKS, "stack", where)
    if layouts is not None:
        for layout in layouts:
            config.check_layout(layout, campaign.core_budget, where=where)
        changes["layouts"] = tuple(layouts)
    if repeats is not None:
        if repeats < 1:
            raise PlanError(f"{where}: --repeats must be positive")
        changes["repeats"] = repeats
    new = dataclasses.replace(campaign, **changes)
    ladder = []
    for rung in new.ladder:
        rung_cases = tuple(c for c in rung.cases if c in new.cases)
        rung_solvers = tuple(s for s in rung.solvers if s in new.solvers)
        if rung_cases and rung_solvers:
            ladder.append(
                dataclasses.replace(rung, cases=rung_cases, solvers=rung_solvers)
            )
    return dataclasses.replace(new, ladder=tuple(ladder))


def _registry(
    values: Sequence[str], registry: Sequence[str], axis: str, where: str
) -> tuple[str, ...]:
    unknown = [v for v in values if v not in registry]
    if unknown or not values:
        raise PlanError(
            f"{where}: unknown {axis} {', '.join(unknown) or '(none)'} "
            f"(known: {', '.join(registry)})"
        )
    return tuple(values)


# --- expansion ---------------------------------------------------------------------


def load_cases(names: Sequence[str], cases_dir: Path = config.CASES_DIR) -> dict:
    """The case files of ``names``."""
    out = {}
    for name in names:
        try:
            out[name] = case_mod.load_case(cases_dir / name / case_mod.CASE_FILE)
        except case_mod.CaseError as error:
            raise PlanError(str(error)) from error
    return out


def estimate(case: Case, solver: str, layout: str, wall_cap_s: float) -> float:
    """Seconds of one run: the case's ``est_s`` hint, capped by the wall cap."""
    hints = case.est_s
    for key in (f"{solver}@{layout}", solver):
        if key in hints:
            return min(hints[key], wall_cap_s)
    return wall_cap_s


def expand(
    campaign: Campaign,
    site: Site,
    *,
    commit: str,
    cases: Mapping[str, Case] | None = None,
    adapters: Mapping[str, Adapter] = ADAPTERS,
) -> Plan:
    """Expand ``campaign`` into cells and jobs; refuse an unlaunchable stack."""
    if cases is None:
        cases = load_cases(campaign.cases, campaign.cases_dir or config.CASES_DIR)
    resolved = _resolve_stacks(campaign, site, commit)
    notes: list[str] = []
    cells: list[Cell] = []
    for stack in campaign.stacks:
        for name in campaign.cases:
            found = _case_cells(campaign, cases[name], stack, adapters)
            cells += found
            if not campaign.same_code_direct:
                continue
            cells += _same_code_direct(
                campaign,
                cases[name],
                stack=stack,
                found=found,
                adapters=adapters,
                notes=notes,
            )
    jobs = _jobs(campaign, cases, cells)
    plan = Plan(campaign, commit, tuple(cells), jobs, resolved, tuple(notes))
    if len({r.versions.get("palace") for r in resolved.values()}) > 1 and any(
        s.startswith("palace:") for s in campaign.solvers
    ):
        raise PlanError(
            "stacks resolve different Palace versions; a campaign pins one version "
            "per solver"
        )
    return plan


def _resolve_stacks(
    campaign: Campaign, site: Site, commit: str
) -> dict[str, stacks.Resolved]:
    resolved = {}
    for name in campaign.stacks:
        try:
            stack = stacks.resolve(
                name,
                site,
                campaign.stack_versions.get(name),
                campaign.stack_env.get(name),
                commit=commit,
            )
        except ConfigError as error:
            raise PlanError(f"stack {name}: {error}") from error
        reason = stacks.refusal(stack)
        if reason is not None:
            raise PlanError(f"stack {name} refused: {reason}")
        resolved[name] = stack
    return resolved


def _kind(adapters: Mapping[str, Adapter], solver: str) -> tuple[Adapter, str]:
    code, _, kind = solver.partition(":")
    if code not in adapters:
        raise PlanError(
            f"no solver adapter for {code!r} (known: {', '.join(adapters)})"
        )
    return adapters[code], kind


def _anchor_layouts(
    campaign: Campaign, case: Case, adapter: Adapter, kind: str, default: Layout
) -> list[tuple[Layout, str]]:
    """``(layout, asked)`` of the campaign's layouts, grid expanded, no duplicate."""
    out: dict[str, tuple[Layout, str]] = {}
    for asked in campaign.layouts:
        if asked != "best":
            layout = default if asked == "default" else Layout.parse(asked)
            out.setdefault(str(layout), (layout, asked))
            continue
        if case.size == "large":
            raise PlanError(
                f"{case.name} is a large case: list its layouts, best "
                "expands only on small cases"
            )
        for layout in grid(campaign.core_budget):
            if adapter.supports(case, kind, layout) is None:
                out.setdefault(str(layout), (layout, "grid"))
    return list(out.values())


def _case_cells(
    campaign: Campaign, case: Case, stack: str, adapters: Mapping[str, Adapter]
) -> list[Cell]:
    cap = campaign.wall_cap_s[case.name]
    cells: list[Cell] = []
    for solver in campaign.solvers:
        adapter, kind = _kind(adapters, solver)
        if kind not in adapter.kinds:
            raise PlanError(f"{solver}: {adapter.code} has no kind {kind!r}")
        iterative = adapter.kinds[kind].klass == "iterative"
        default = adapter.default_layout(case, kind, campaign.core_budget)
        anchor = campaign.anchor_tolerance if iterative else None
        wanted = [
            (layout, asked, anchor, False)
            for layout, asked in _anchor_layouts(campaign, case, adapter, kind, default)
        ]
        seen = {str(layout) for layout, *_ in wanted}
        for rung in campaign.ladder:
            if case.name not in rung.cases or solver not in rung.solvers:
                continue
            if not iterative:
                raise PlanError(f"ladder lists {solver}, a direct kind")
            layout = default if rung.layout == "default" else Layout.parse(rung.layout)
            wanted += [
                (layout, rung.layout, tolerance, True)
                for tolerance in rung.tolerances
                if not (tolerance == anchor and str(layout) in seen)
            ]
        for layout, asked, tolerance, rung in wanted:
            reason = adapter.supports(case, kind, layout)
            cells.append(
                Cell(
                    case=case.name,
                    solver=solver,
                    stack=stack,
                    layout=str(layout),
                    asked=asked,
                    tolerance=tolerance,
                    repeats=1 if reason else campaign.repeats,
                    wall_cap_s=cap,
                    est_s=0.0 if reason else estimate(case, solver, str(layout), cap),
                    default=layout == default,
                    rung=rung,
                    refused=reason,
                )
            )
    return cells


def _same_code_direct(
    campaign: Campaign,
    case: Case,
    *,
    stack: str,
    found: Sequence[Cell],
    adapters: Mapping[str, Adapter],
    notes: list[str],
) -> list[Cell]:
    """The direct runs ADR 0409 needs for the same-code error, when none is planned."""
    added = []
    budget = campaign.core_budget
    cap = campaign.wall_cap_s[case.name]
    codes = dict.fromkeys(c.code for c in found)
    for code in codes:
        adapter = adapters[code]
        klass = {
            c.solver: adapter.kinds[c.solver.partition(":")[2]].klass
            for c in found
            if c.code == code and not c.refused
        }
        if "iterative" not in klass.values() or "direct" in klass.values():
            continue
        for kind, spec in adapter.kinds.items():
            if spec.klass != "direct":
                continue
            layout = adapter.default_layout(case, kind, budget)
            if adapter.supports(case, kind, layout) is not None:
                continue
            solver = f"{code}:{kind}"
            added.append(
                Cell(
                    case=case.name,
                    solver=solver,
                    stack=stack,
                    layout=str(layout),
                    asked="default",
                    tolerance=None,
                    repeats=1,
                    wall_cap_s=cap,
                    est_s=estimate(case, solver, str(layout), cap),
                    default=True,
                    added_for=SAME_CODE_ERROR,
                )
            )
            break
        else:
            notes.append(
                f"{case.name} {code}: no direct kind runs; same_code_error null"
            )
    return added


# --- jobs and rotation -------------------------------------------------------------


def rotate[T](rows: Sequence[T], repeats: int | Sequence[int]) -> list[T]:
    """Rows in rounds, round r rotated by r * ceil(|rows| / R) (ADR 0411).

    ``repeats`` is one count for every row or one per row; a row runs in the first
    ``repeats`` rounds. A repeat that would follow the same row is swapped with the
    next different row of its round, so no row runs twice back to back (unless it is
    the only row left).
    """
    counts = [repeats] * len(rows) if isinstance(repeats, int) else list(repeats)
    rounds = max(counts, default=0)
    n = len(rows)
    shift = math.ceil(n / rounds) if rounds else 0
    order: list[int] = []
    for r in range(rounds):
        k = (r * shift) % n if n else 0
        members = [i for i in (*range(k, n), *range(k)) if counts[i] > r]
        if order and members and members[0] == order[-1]:
            swap = next((j for j, i in enumerate(members) if i != order[-1]), None)
            if swap is not None:
                members[0], members[swap] = members[swap], members[0]
        order += members
    return [rows[i] for i in order]


def _jobs(
    campaign: Campaign, cases: Mapping[str, Case], cells: Sequence[Cell]
) -> tuple[Job, ...]:
    keys: dict[tuple[str, ...], list[Cell]] = {}
    for c in cells:
        key = (c.stack,) if campaign.group == "stack" else (c.case, c.stack)
        keys.setdefault(key, []).append(c)
    nodes = math.ceil(campaign.core_budget / NODE_CORES)
    constraint = "highmem" if campaign.node_class == "highmem" else None
    jobs = []
    for key, members in keys.items():
        name = "-".join(key)
        order = rotate(members, [c.repeats for c in members])
        seen: dict[Cell, int] = {}
        steps = []
        for line, c in enumerate(order, start=1):
            seen[c] = seen.get(c, 0) + 1
            steps.append(Step(c, seen[c], name, line, campaign.core_budget))
        job_cases = tuple(dict.fromkeys(c.case for c in members))
        est = JOB_OVERHEAD_S + math.fsum(c.est_total_s for c in members)
        est += math.fsum(cases[n].est_s.get(STAGING_KEY, 0.0) for n in job_cases)
        jobs.append(
            Job(
                name=name,
                group=campaign.group,
                cases=job_cases,
                stack=members[0].stack,
                steps=tuple(steps),
                nodes=nodes,
                est_s=est,
                constraint=constraint,
            )
        )
    return tuple(jobs)


# --- run directory -------------------------------------------------------------------


def source_commit(root: Path = ROOT) -> tuple[str, str | None]:
    """``(commit, commit_utc)`` of the tree.

    A shipped tree has ``COMMIT`` and ``COMMIT_UTC`` (written by ``ship``); a checkout
    asks ``git``.
    """
    commit_file = root / "COMMIT"
    if commit_file.is_file():
        utc = root / "COMMIT_UTC"
        return (
            commit_file.read_text().strip(),
            utc.read_text().strip() if utc.is_file() else None,
        )
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "show", "-s", "--format=%H %cI", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PlanError(f"{root}: no COMMIT file and not a git checkout") from error
    when = dt.datetime.fromisoformat(out[1]).astimezone(dt.UTC)
    return out[0], when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _campaign_toml(plan: Plan) -> dict[str, Any]:
    c = plan.campaign
    raw: dict[str, Any] = {
        "schema": 1,
        "cases": list(c.cases),
        "solvers": list(c.solvers),
        "layouts": list(c.layouts),
        "core_budget": c.core_budget,
        "repeats": c.repeats,
        "node_class": c.node_class,
        "anchor_tolerance": c.anchor_tolerance,
        "jobs": {"group": c.group},
        "wall_cap_s": dict(c.wall_cap_s),
    }
    if c.cases_dir is not None:
        raw["cases_dir"] = str(c.cases_dir)
    if not c.same_code_direct:
        raw["same_code_direct"] = False
    if c.ladder:
        raw["ladder"] = [
            {
                "cases": list(r.cases),
                "solvers": list(r.solvers),
                "tolerances": list(r.tolerances),
                "layout": r.layout,
            }
            for r in c.ladder
        ]
    raw["stacks"] = {
        name: {
            "versions": dict(plan.resolved[name].versions),
            **({"env": dict(c.stack_env[name])} if name in c.stack_env else {}),
        }
        for name in c.stacks
    }
    return raw


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    plan: Plan, site: Site, commit_utc: str | None, campaign_id: str, now: dt.datetime
) -> records.Campaign:
    c = plan.campaign
    return records.Campaign(
        id=campaign_id,
        name=c.name,
        planned_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        campaign_file=records.CampaignFile(
            path=_relative(c.path), sha256=_sha256(c.path)
        ),
        pins=records.Pins(
            gfem={"commit": plan.commit, "commit_utc": commit_utc},
            harness={"commit": plan.commit},
            solvers=plan.solver_versions(),
            stacks={
                name: {
                    "description_sha256": hashlib.sha256(
                        r.description.text.encode()
                    ).hexdigest(),
                    "env_manifest_sha256": None,
                    "fingerprint": r.fingerprint,
                }
                for name, r in plan.resolved.items()
            },
        ),
        machine=site.name,
        core_budget=c.core_budget,
        jobs=[
            records.Job(
                job=job.name,
                group=job.group,
                qos=site.qos.get(job.qos, job.qos),
                constraint=job.constraint,
                extra={
                    "cases": list(job.cases),
                    "stack": job.stack,
                    "runs": len(job.steps),
                    "nodes": job.nodes,
                    "est_s": round(job.est_s, 1),
                    "est_core_h": round(job.est_core_h, 1),
                    "time_limit": job.time_limit,
                },
            )
            for job in plan.jobs
        ],
        extra={
            "node_class": c.node_class,
            "repeats": c.repeats,
            "runs": plan.runs,
            "est_core_h": round(plan.est_core_h, 1),
            "limit_core_h": round(plan.limit_core_h, 1),
            "notes": list(plan.notes),
        },
    )


def _existing_id(plan: Plan, rundir: Path) -> str | None:
    path = rundir / "campaign.json"
    if not path.is_file():
        return None
    old = records.load_campaign(path)
    if old.name != plan.campaign.name:
        raise PlanError(
            f"{rundir} holds campaign {old.name}, not {plan.campaign.name}: "
            "a run directory holds one campaign"
        )
    pins = records.Pins(
        gfem={"commit": plan.commit},
        harness={"commit": plan.commit},
        solvers=plan.solver_versions(),
    )
    try:
        records.check_extends(old, pins)
    except records.RecordError as error:
        raise PlanError(str(error)) from error
    return old.id


def write(
    plan: Plan,
    rundir: Path,
    *,
    site: Site,
    commit_utc: str | None,
    now: dt.datetime | None = None,
) -> records.Campaign:
    """Write the run directory of ``plan``; return its manifest.

    An existing run directory is planned again only for the same campaign at the
    same commit and solver versions (its id is kept).
    """
    now = dt.datetime.now(dt.UTC) if now is None else now
    campaign_id = _existing_id(plan, rundir)
    if campaign_id is None:
        siblings = (
            rundir.parent.glob("*/campaign.json") if rundir.parent.is_dir() else ()
        )
        taken = [records.load_campaign(p).id for p in siblings]
        campaign_id = records.campaign_id(
            now.date(), plan.commit, plan.campaign.name, taken
        )
    jobs_dir = rundir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for old in jobs_dir.glob(f"*{STEPS_SUFFIX}"):
        old.unlink()
    (rundir / "COMMIT").write_text(plan.commit + "\n")
    if commit_utc:
        (rundir / "COMMIT_UTC").write_text(commit_utc + "\n")
    (rundir / "campaign.toml").write_text(case_mod.dumps(_campaign_toml(plan)))
    header = "job\tcases\tstack\truns\tcells\tnodes\test_s\test_core_h\ttime_limit"
    lines = [header + "\tqos\tconstraint"]
    for job in plan.jobs:
        (jobs_dir / f"{job.name}{STEPS_SUFFIX}").write_text(
            "".join(json.dumps(s.to_json()) + "\n" for s in job.steps)
        )
        lines.append(
            "\t".join(
                [
                    job.name,
                    ",".join(job.cases),
                    job.stack,
                    str(len(job.steps)),
                    str(job.cells),
                    str(job.nodes),
                    f"{job.est_s:.0f}",
                    f"{job.est_core_h:.1f}",
                    job.time_limit,
                    job.qos,
                    job.constraint or "",
                ]
            )
        )
    (rundir / "plan.tsv").write_text("\n".join(lines) + "\n")
    manifest = _manifest(plan, site, commit_utc, campaign_id, now)
    records.write_campaign(manifest, rundir, roots=records.site_roots(site))
    return manifest


def summary(plan: Plan, site: Site, manifest: records.Campaign) -> str:
    """What ``plan`` prints: one line per job and the total."""
    rows = [f"campaign {manifest.id}"]
    rows.append(
        f"{'job':<24} {'cells':>5} {'runs':>5} {'est core-h':>10} {'limit':>8}  qos"
    )
    for job in plan.jobs:
        qos = site.qos.get(job.qos, job.qos)
        extra = f" {job.constraint}" if job.constraint else ""
        rows.append(
            f"{job.name:<24} {job.cells:>5} {len(job.steps):>5} "
            f"{job.est_core_h:>10.0f} {job.time_limit:>8}  {job.qos} ({qos}){extra}"
        )
    rows.append(
        f"total: {len(plan.jobs)} jobs, {plan.runs} runs, ~{plan.est_core_h:.0f} "
        f"core-h (time limits {plan.limit_core_h:.0f})"
    )
    rows += [f"note: {n}" for n in plan.notes]
    return "\n".join(rows)
