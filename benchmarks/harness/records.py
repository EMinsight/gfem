"""Suite records, campaign manifests and the records store (ADRs 0401, 0402).

A **suite record** (``suite-record/1``) is one JSON file per planned run, named
``<case>/<code>-<kind>__<stack>__<layout>__rep<n>`` under the run directory's
``records/``. Its blocks: ``schema``, ``campaign``, ``name``; ``run`` (case, solver,
stack, layout asked and resolved, core budget, repeat, tolerance, order in job, Slurm
job and step, start and finish UTC, ``added_for``, ``launch``); ``status``, ``reason``,
``failure_class``, ``slurm_state``, ``returncode``; ``case`` (definition and mesh
sha256, order, DOF count, converged reference); ``outputs``; ``accuracy`` and
``same_code_error``; ``timings`` (flat ``<bucket>_s`` headline buckets, a phase a code
does not report is ``null`` never 0, ``rank_reduction``, ``source``, ``rank0``);
``memory``; ``checks``; ``structural``; ``stack``; ``provenance``.

``status`` says only whether the run executed (``STATUSES``); correctness lives in
``checks`` (``pass``, ``fail``, ``skipped``, ``observed``). A failure reason carries a
class, ``solver`` or ``harness``. A rerun of a name moves the earlier file aside to
``NAME.superseded-N.json``; superseded records never reach the store or a report.

A **campaign manifest** (``suite-campaign/1``, ``campaign.json``) is written by
``plan`` and completed by ``close``: id ``YYYYMMDD-<sha7>-<name>`` (``-2``, ``-3`` ...
on a same-day rerun), pins (gfem commit and its UTC date, harness commit, one version
per solver, per software stack the description and env-manifest sha256), the campaign
file and its sha256, machine, core budget, ``planned_utc``; later ``jobs``,
``records`` (relative path to sha256) and campaign ``checks``. Records whose observed
provenance disagrees with the pins make a mixed campaign, refused unless asked.

Records are scrubbed when written: each site root is replaced by its key placeholder
(``{cluster.root}``, ``{stacks.root}``, ...), and any account, user name or remaining
home or ``/gpfs`` path fails the scrub. Schema majors bump only on incompatible change;
one loader per major; an unknown major is refused; stored records are never rewritten.
Unknown keys of a known major are kept as they are (an optional key added later).

The **records store** is a worktree of the orphan branch ``bench-data``. Per campaign
it holds ``campaign.json``, ``records/<case>/*.json`` and
``jobs/<jobid>/{node.txt, pip-freeze.txt, machine.json}``; ``fetch --records-only``
writes exactly that set. Standard library only (ADR 0400).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from benchmarks.harness import config

RECORD_SCHEMA = "suite-record"
CAMPAIGN_SCHEMA = "suite-campaign"
RECORD_MAJOR = 1
CAMPAIGN_MAJOR = 1

STATUSES = (
    "ok",
    "failed",
    "timeout",
    "memory_watchdog",
    "killed",
    "not_started",
    "started",
    "refused",
)
"""Whether a run executed; ``ok``, ``failed`` and ``refused`` are an adapter's."""
FINAL_STATUSES = tuple(s for s in STATUSES if s != "started")
FAILURE_CLASSES = ("solver", "harness")
SKIP_REASONS = ("earlier_repeat_timeout", "kind_not_converging")
"""Reasons of ``not_started`` records skipped inside a job (ticket 12)."""
CHECK_VALUES = ("pass", "fail", "skipped", "observed")
LAYOUTS_ASKED = ("default", "best", "grid")
"""Named values of ``run.layout.asked``; an explicit ``RxT`` is the other form."""
LAUNCHES = ("srun", "taskset", "hydra")
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
"""Phase buckets of ``timings`` (and of ``timings.rank0``)."""

ENV_PREFIXES = (
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "NGS_",
    "JAX_",
    "I_MPI_",
    "SLURM_CPUS_",
    "SLURM_NTASKS",
    "UCX_",
    "SLURM_EAR",
)
"""Prefixes of the environment variables a record keeps (``provenance.env``)."""

_LAYOUT = re.compile(r"^[1-9]\d*x[1-9]\d*$")
_SUPERSEDED = re.compile(r"\.superseded-\d+\.json$")
_PRIVATE_PATH = re.compile(r"(?<![\w.}])/(?:gpfs|home|scratch|Users|root)/[^\s\"']*")


class RecordError(ValueError):
    """A record or manifest the suite refuses to read or write."""


class ScrubError(RecordError):
    """Text left after the scrub still names an account, a user or a private path."""


# --- JSON <-> dataclasses ---------------------------------------------------------


def _dataclass_type(hint: object) -> type | None:
    """The dataclass in ``hint`` (``X`` or ``X | None``), else None."""
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return hint
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(inner) == 1:
            return _dataclass_type(inner[0])
    return None


def _list_item_type(hint: object) -> type | None:
    if typing.get_origin(hint) is list:
        (item,) = typing.get_args(hint)
        return _dataclass_type(item)
    return None


def _decode(cls: Any, data: object, where: str) -> Any:
    if not isinstance(data, dict):
        raise RecordError(f"{where}: expected an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.name != "extra"}
    values: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name == "extra":
            continue
        if f.name not in data:
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            ):
                raise RecordError(f"{where}: missing key {f.name}")
            continue
        value = data[f.name]
        nested = _dataclass_type(hints[f.name])
        item = _list_item_type(hints[f.name])
        if nested is not None and value is not None:
            value = _decode(nested, value, f"{where}.{f.name}")
        elif item is not None:
            if not isinstance(value, list):
                raise RecordError(f"{where}.{f.name}: expected a list")
            value = [
                _decode(item, v, f"{where}.{f.name}[{i}]") for i, v in enumerate(value)
            ]
        values[f.name] = value
    extra = {k: v for k, v in data.items() if k not in names}
    try:
        return cls(**values, extra=extra)
    except (TypeError, ValueError) as error:
        raise RecordError(f"{where}: {error}") from error


def _encode(obj: object) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        if hasattr(obj, "schema"):
            out["schema"] = obj.schema
        for f in dataclasses.fields(obj):
            if f.name != "extra":
                out[f.name] = _encode(getattr(obj, f.name))
        out.update(getattr(obj, "extra", {}))
        return out
    if isinstance(obj, list | tuple):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


class _Block:
    """``to_json``/``from_json`` for a block dataclass; unknown keys go to ``extra``."""

    def to_json(self) -> dict[str, Any]:
        """The block as plain JSON values, fields in declaration order."""
        return _encode(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, where: str | None = None) -> Self:
        """Read the block from plain JSON values; refuse a missing required key."""
        return _decode(cls, copy.deepcopy(dict(data)), where or cls.__name__)


def _check(value: object, allowed: Sequence[object], what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} {value!r} not in {', '.join(map(str, allowed))}")


# --- suite record -----------------------------------------------------------------


@dataclass
class Layout(_Block):
    """Layout asked (``default``, ``best``, ``grid`` or ``RxT``) and resolved."""

    asked: str
    ranks: int | None = None
    threads: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asked not in LAYOUTS_ASKED and not _LAYOUT.match(self.asked):
            raise ValueError(
                f"layout.asked {self.asked!r} is not default/best/grid/RxT"
            )


@dataclass
class Run(_Block):
    """What was planned and where it ran."""

    case: str
    solver: str
    """``code:kind``."""
    stack: str
    layout: Layout
    core_budget: int
    repeat: int
    tolerance: float | None = None
    order_in_job: int | None = None
    job: str | None = None
    step: str | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    added_for: str | None = None
    """``same_code_error`` for a direct run ``plan`` added, else None."""
    launch: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.launch is not None:
            _check(self.launch, LAUNCHES, "run.launch")


@dataclass
class Reference(_Block):
    """The converged reference a record was scored against."""

    id: str
    sha256: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseFacts(_Block):
    """The case as run: definition and mesh pins, order, DOF count."""

    definition_sha256: str | None = None
    mesh_sha256: str | None = None
    order: int | None = None
    ndof: int | None = None
    reference: Reference | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Accuracy(_Block):
    """Error against the converged reference (harness scorer only)."""

    metric: str | None = None
    value: float | None = None
    per_item: list[float] | None = None
    bound: float | None = None
    """The reference's own bound."""
    resolved: bool | None = None
    """True when the error is at least ten times the bound."""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Timings(_Block):
    """Phase buckets (headline reduction), derived figures and rank 0's buckets."""

    wall_s: float | None = None
    startup_s: float | None = None
    work_s: float | None = None
    mesh_read_s: float | None = None
    discretize_s: float | None = None
    assemble_s: float | None = None
    setup_s: float | None = None
    solve_s: float | None = None
    estimate_s: float | None = None
    postpro_s: float | None = None
    rom_s: float | None = None
    other_s: float | None = None
    setup_total_s: float | None = None
    solve_total_s: float | None = None
    per_sample_s: float | None = None
    rank_reduction: str | None = None
    source: str | None = None
    rank0: dict[str, float | None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory(_Block):
    """Memory figures; each code's self-report kept beside, never mixed in."""

    peak_rss_sum_gib: float | None = None
    peak_rss_rank_max_gib: float | None = None
    peak_rss_largest_node_gib: float | None = None
    metric: str | None = None
    pss_peak_sum_gib: float | None = None
    sacct_max_rss_gib: float | None = None
    self_reports: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Machine(_Block):
    """Where the run executed."""

    name: str | None = None
    nodes: list[str] = field(default_factory=list)
    node_class: str | None = None
    cpu: str | None = None
    mem_total_kb: list[int] = field(default_factory=list)
    cpus_allowed: int | None = None
    pinned_cpus: str | None = None
    exclusive: bool | None = None
    partition: str | None = None
    qos: str | None = None
    constraint: str | None = None
    mpi_library: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance(_Block):
    """What the run observed: commits, solver build, stack hashes, env, machine."""

    gfem: dict[str, Any] = field(default_factory=dict)
    """``commit``, ``version``."""
    harness: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    """``code``, ``version``, ``build``, ``wheel_sha256``, ``build_flags``."""
    stack: dict[str, Any] = field(default_factory=dict)
    """``name``, ``description_sha256``, ``env_manifest_sha256``."""
    versions: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    """Allow-listed variables only (``ENV_PREFIXES``)."""
    machine: Machine = field(default_factory=Machine)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteRecord(_Block):
    """One planned run of a campaign (``suite-record/1``)."""

    campaign: str
    name: str
    run: Run
    status: str
    schema: str = f"{RECORD_SCHEMA}/{RECORD_MAJOR}"
    reason: str | None = None
    failure_class: str | None = None
    slurm_state: str | None = None
    returncode: int | None = None
    case: CaseFacts = field(default_factory=CaseFacts)
    outputs: dict[str, Any] = field(default_factory=dict)
    accuracy: Accuracy | None = None
    same_code_error: dict[str, Any] | None = None
    timings: Timings = field(default_factory=Timings)
    memory: Memory = field(default_factory=Memory)
    checks: dict[str, str] = field(default_factory=dict)
    structural: dict[str, Any] = field(default_factory=dict)
    stack: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self.status, STATUSES, "status")
        if self.failure_class is not None:
            _check(self.failure_class, FAILURE_CLASSES, "failure_class")
        for name, value in self.checks.items():
            _check(value, CHECK_VALUES, f"checks.{name}")


# --- campaign manifest ------------------------------------------------------------


@dataclass
class CampaignFile(_Block):
    """The committed campaign file a campaign was planned from."""

    path: str
    sha256: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pins(_Block):
    """One gfem commit, one harness commit, one version per solver, stack hashes."""

    gfem: dict[str, Any]
    """``commit``, ``commit_utc``, ``version``."""
    harness: dict[str, Any]
    solvers: dict[str, str] = field(default_factory=dict)
    """Solver code to version."""
    stacks: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Stack name to ``description_sha256`` and ``env_manifest_sha256``."""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job(_Block):
    """One Slurm job of a campaign, filled by ``close``."""

    job: str
    group: str | None = None
    qos: str | None = None
    constraint: str | None = None
    nodes: list[str] = field(default_factory=list)
    elapsed_s: float | None = None
    billed_core_h: float | None = None
    """sacct ``CPUTimeRAW`` / 2 / 3600 (ADR 0188)."""
    state: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Campaign(_Block):
    """A campaign manifest (``suite-campaign/1``, ``campaign.json``)."""

    id: str
    name: str
    planned_utc: str
    campaign_file: CampaignFile
    pins: Pins
    machine: str
    core_budget: int
    schema: str = f"{CAMPAIGN_SCHEMA}/{CAMPAIGN_MAJOR}"
    jobs: list[Job] = field(default_factory=list)
    records: dict[str, str] = field(default_factory=dict)
    """Record path relative to the campaign directory, to its sha256."""
    checks: dict[str, Any] = field(default_factory=dict)
    closed_utc: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --- loaders, one per major --------------------------------------------------------


def _major(data: Mapping[str, Any], family: str, where: str) -> int:
    schema = data.get("schema")
    prefix = f"{family}/"
    if not isinstance(schema, str) or not schema.startswith(prefix):
        raise RecordError(f"{where}: schema {schema!r} is not {family}/<major>")
    try:
        return int(schema.removeprefix(prefix))
    except ValueError as error:
        raise RecordError(f"{where}: schema {schema!r} has no integer major") from error


def _read_json(source: Path | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(source, Mapping):
        return dict(source), "<record>"
    try:
        data = json.loads(Path(source).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RecordError(f"{source}: {error}") from error
    if not isinstance(data, dict):
        raise RecordError(f"{source}: not a JSON object")
    return data, str(source)


_RECORD_LOADERS = {1: SuiteRecord.from_json}
_CAMPAIGN_LOADERS = {1: Campaign.from_json}


def load_record(source: Path | Mapping[str, Any]) -> SuiteRecord:
    """Read a suite record of any known major, normalized to the current shape."""
    data, where = _read_json(source)
    major = _major(data, RECORD_SCHEMA, where)
    loader = _RECORD_LOADERS.get(major)
    if loader is None:
        raise RecordError(
            f"{where}: schema {data['schema']} unknown (this harness reads "
            f"{RECORD_SCHEMA}/{', '.join(map(str, _RECORD_LOADERS))})"
        )
    return loader(data, where=where)


def load_campaign(source: Path | Mapping[str, Any]) -> Campaign:
    """Read a campaign manifest of any known major."""
    data, where = _read_json(source)
    major = _major(data, CAMPAIGN_SCHEMA, where)
    loader = _CAMPAIGN_LOADERS.get(major)
    if loader is None:
        raise RecordError(
            f"{where}: schema {data['schema']} unknown (this harness reads "
            f"{CAMPAIGN_SCHEMA}/{', '.join(map(str, _CAMPAIGN_LOADERS))})"
        )
    return loader(data, where=where)


# --- scrub and write --------------------------------------------------------------


def site_roots(site: config.Site) -> dict[str, str]:
    """Each absolute site root by its placeholder, e.g. ``{cluster.root}``.

    The names agree with the stack kinds' (``{stacks.<name>.<key>}``,
    ``{stacks.<name>.wheel_dirs[i]}``, ``{stacks.root}``), so a ``stack`` block
    made by ``benchmarks.stacks.record`` passes through this scrub unchanged.
    """
    roots: dict[str, str] = {"{cluster.root}": site.cluster_root}
    if site.stacks_root:
        roots["{stacks.root}"] = site.stacks_root
    roots["{data.dir}"] = site.data_dir
    roots["{output.dir}"] = site.output_dir
    for i, store in enumerate(site.data_stores):
        roots[f"{{data.stores[{i}]}}"] = store
    if site.ship_mesh_store:
        roots["{ship.mesh_store}"] = site.ship_mesh_store
    for name, inputs in site.stacks.items():
        for key, value in inputs.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    roots[f"{{stacks.{name}.{key}[{i}]}}"] = item
            else:
                roots[f"{{stacks.{name}.{key}}}"] = value
    return roots


def replace_roots(text: str, roots: Mapping[str, str]) -> str:
    """``text`` with each root path replaced by its placeholder, longest first.

    A root matches only as a whole path prefix: ``/r/gfem`` is replaced in
    ``/r/gfem/envs`` but not in ``/r/gfem2``, which is left for the leak check.
    """
    for placeholder, path in sorted(roots.items(), key=lambda kv: -len(kv[1])):
        root = path.rstrip("/")
        if root:
            text = re.sub(re.escape(root) + r"(?=/|$|[^\w.-])", placeholder, text)
    return text


def site_secrets(site: config.Site, environ: Mapping[str, str]) -> tuple[str, ...]:
    """Names a record must never hold: the Slurm account and the user name."""
    names = [site.account, environ.get("USER") or environ.get("LOGNAME")]
    return tuple(n for n in names if n)


def scrub(value: Any, roots: Mapping[str, str], secrets: Iterable[str] = ()) -> Any:
    """Replace site roots by their placeholders; refuse what still leaks.

    ``roots`` maps placeholder to path (``site_roots``). Roots are replaced longest
    first (``replace_roots``), so a wheelhouse under the cluster root becomes
    ``{stacks.wheels.palace_wheelhouse}``, not ``{cluster.root}/...``. Keys and values
    are scrubbed. Raises ``ScrubError`` naming each account or user name and each
    home, scratch or ``/gpfs`` path left.
    """
    # a name inside a path or a hyphenated word (data/bench/, bench-<sha7>) is not
    # the account or user; private paths are caught by _PRIVATE_PATH
    names = [
        re.compile(rf"(?<![\w./-]){re.escape(s)}(?![\w./-])") for s in secrets if s
    ]
    leaks: list[str] = []

    def text(s: str, where: str) -> str:
        s = replace_roots(s, roots)
        leaks.extend(f"{where}: {m.group(0)}" for m in _PRIVATE_PATH.finditer(s))
        leaks.extend(f"{where}: names {p.pattern}" for p in names if p.search(s))
        return s

    def walk(v: Any, where: str) -> Any:
        if isinstance(v, str):
            return text(v, where)
        if isinstance(v, dict):
            return {
                text(k, f"{where}.{k}"): walk(item, f"{where}.{k}")
                for k, item in v.items()
            }
        if isinstance(v, list | tuple):
            return [walk(item, f"{where}[{i}]") for i, item in enumerate(v)]
        return v

    out = walk(value, "$")
    if leaks:
        raise ScrubError("scrub failed: " + "; ".join(leaks))
    return out


def dumps(data: Mapping[str, Any]) -> str:
    """The stored text of a record or manifest (one-space indent, final newline)."""
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


def write_record(
    record: SuiteRecord,
    records_dir: Path,
    *,
    roots: Mapping[str, str],
    secrets: Iterable[str] = (),
) -> Path:
    """Scrub ``record`` and write it to ``records_dir/<name>.json``; return the path."""
    data = scrub(record.to_json(), roots, secrets)
    return _write(records_dir / f"{record.name}.json", dumps(data))


def write_campaign(
    campaign: Campaign,
    rundir: Path,
    *,
    roots: Mapping[str, str],
    secrets: Iterable[str] = (),
) -> Path:
    """Scrub ``campaign`` and write it to ``rundir/campaign.json``; return the path."""
    data = scrub(campaign.to_json(), roots, secrets)
    return _write(rundir / "campaign.json", dumps(data))


def allowed_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The allow-listed part of ``environ`` (``ENV_PREFIXES``), sorted by name."""
    return {k: environ[k] for k in sorted(environ) if k.startswith(ENV_PREFIXES)}


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes (the manifest's ``records`` values)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_paths(rundir: Path) -> list[Path]:
    """Every current record under ``rundir/records`` (superseded ones left out)."""
    root = rundir / "records"
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*.json") if p.is_file() and not is_superseded(p)
    )


def record_hashes(rundir: Path) -> dict[str, str]:
    """``records/<case>/<name>.json`` -> sha256 of every current record."""
    return {
        p.relative_to(rundir).as_posix(): file_sha256(p) for p in record_paths(rundir)
    }


_FILE_URL = re.compile(r"file://(/[^\s#]*/)?([^/\s#]+)")


def scrub_freeze(text: str, roots: Mapping[str, str]) -> str:
    """``pip freeze`` text fit for the store: roots replaced, local paths dropped.

    A ``name @ file:///<path>/<wheel>#sha256=..`` line keeps the wheel's file name
    and hash, never the path (``jobs/<jobid>/pip-freeze.txt``, ADR 0402).
    """
    text = replace_roots(text, roots)
    return _FILE_URL.sub(lambda m: f"file:{m[2]}", text)


# --- names ------------------------------------------------------------------------


def record_name(
    case: str,
    solver: str,
    stack: str,
    layout: str,
    repeat: int,
    *,
    rung: float | None = None,
) -> str:
    """``<case>/<code>-<kind>__<stack>__<layout>[__tol<t>]__rep<n>``.

    ``layout`` is ``default`` or the resolved ``RxT`` (a grid run names its ``RxT``).
    ``rung`` is the tolerance of an off-anchor ladder rung (``tol1e-06``), which
    would otherwise share its name with the anchor run at the same layout.
    """
    tol = "" if rung is None else f"__tol{rung:g}"
    return f"{case}/{solver.replace(':', '-')}__{stack}__{layout}{tol}__rep{repeat}"


def is_superseded(path: Path) -> bool:
    """Whether ``path`` is a record moved aside by a rerun."""
    return _SUPERSEDED.search(path.name) is not None


def supersede(path: Path) -> Path | None:
    """Move the record at ``path`` to the first free ``NAME.superseded-N.json``.

    Returns the new path, or None when there was nothing to move.
    """
    if not path.exists():
        return None
    stem = path.name.removesuffix(".json")
    n = 1
    while (target := path.with_name(f"{stem}.superseded-{n}.json")).exists():
        n += 1
    path.rename(target)
    return target


def campaign_id(day: dt.date, commit: str, name: str, existing: Iterable[str]) -> str:
    """``YYYYMMDD-<sha7>-<name>``, then ``-2``, ``-3`` ... if ``existing`` has it."""
    base = f"{day:%Y%m%d}-{commit[:7]}-{name}"
    taken = set(existing)
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


# --- mixed campaigns ----------------------------------------------------------------


def mixed(campaign: Campaign, rows: Iterable[SuiteRecord]) -> list[str]:
    """How ``rows`` disagree with ``campaign``'s id and pins (empty: not mixed).

    Only what a record observed is compared: a record that never ran has no
    provenance to disagree with.
    """
    pins = campaign.pins
    problems: list[str] = []
    for row in rows:
        seen = row.provenance
        if row.campaign != campaign.id:
            problems.append(f"{row.name}: campaign {row.campaign} != {campaign.id}")
        for what, observed, pinned in (
            ("gfem commit", seen.gfem.get("commit"), pins.gfem.get("commit")),
            ("harness commit", seen.harness.get("commit"), pins.harness.get("commit")),
        ):
            if observed and pinned and observed != pinned:
                problems.append(f"{row.name}: {what} {observed} != pinned {pinned}")
        code, version = seen.solver.get("code"), seen.solver.get("version")
        pinned_version = pins.solvers.get(code) if code else None
        if version and pinned_version and version != pinned_version:
            problems.append(f"{row.name}: {code} {version} != pinned {pinned_version}")
        stack = pins.stacks.get(seen.stack.get("name") or row.run.stack, {})
        for key in ("description_sha256", "env_manifest_sha256"):
            observed, pinned = seen.stack.get(key), stack.get(key)
            if observed and pinned and observed != pinned:
                problems.append(
                    f"{row.name}: stack {row.run.stack} {key} {observed} "
                    f"!= pinned {pinned}"
                )
    return problems


def refuse_mixed(
    campaign: Campaign, rows: Iterable[SuiteRecord], *, allow_mixed: bool = False
) -> list[str]:
    """Raise on a mixed campaign unless ``allow_mixed``; return the disagreements."""
    problems = mixed(campaign, rows)
    if problems and not allow_mixed:
        raise RecordError(
            f"campaign {campaign.id} is mixed (pass --allow-mixed to look): "
            + "; ".join(problems)
        )
    return problems


def check_extends(campaign: Campaign, pins: Pins) -> None:
    """Refuse adding runs at another gfem commit or solver version to ``campaign``."""
    old = campaign.pins
    problems = [
        f"{what} {new} != pinned {pinned}"
        for what, new, pinned in (
            ("gfem commit", pins.gfem.get("commit"), old.gfem.get("commit")),
            ("harness commit", pins.harness.get("commit"), old.harness.get("commit")),
        )
        if new != pinned
    ]
    for code in sorted(set(pins.solvers) | set(old.solvers)):
        new, pinned = pins.solvers.get(code), old.solvers.get(code)
        if new != pinned:
            problems.append(f"{code} {new} != pinned {pinned}")
    if problems:
        raise RecordError(
            f"campaign {campaign.id}: {'; '.join(problems)}; "
            "a new commit or solver version is a new campaign"
        )


# --- fetch and the records store ----------------------------------------------------

STORE_FILTERS = (
    "--include=/campaign.json",
    "--exclude=*.superseded-*.json",
    "--include=/records/",
    "--include=/records/*/",
    "--include=/records/*/*.json",
    "--include=/jobs/",
    "--include=/jobs/*/",
    "--include=/jobs/*/node.txt",
    "--include=/jobs/*/pip-freeze.txt",
    "--include=/jobs/*/machine.json",
    "--exclude=*",
)
"""rsync filters of the records-store set of one campaign run directory."""


def fetch_argv(source: str, dest: Path, *, records_only: bool = False) -> list[str]:
    """The rsync command copying run directory ``source`` (``[HOST:]PATH``)."""
    filters = [*STORE_FILTERS, "--prune-empty-dirs"] if records_only else []
    return ["rsync", "-a", *filters, source.rstrip("/") + "/", f"{dest}/"]


def store_leaks(directory: Path) -> list[str]:
    """Home, scratch or ``/gpfs`` paths left in the files under ``directory``."""
    leaks = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        text = path.read_text(errors="replace")
        leaks.extend(
            f"{path.relative_to(directory).as_posix()}: {m.group(0)}"
            for m in _PRIVATE_PATH.finditer(text)
        )
    return leaks


def fetch(
    source: str,
    dest: Path,
    *,
    records_only: bool = False,
    run: Any = subprocess.run,
) -> int:
    """Copy a run directory back; with ``records_only``, only the records-store set.

    The records-store set is ``campaign.json``, ``records/<case>/*.json`` without
    superseded records, and ``jobs/<jobid>/{node.txt, pip-freeze.txt, machine.json}``;
    no logs, work directories or job scripts. A store set that still names a private
    path is reported and the exit status is 1: fix the writer, never the store. The
    command never commits.
    """
    dest.mkdir(parents=True, exist_ok=True)
    done = run(fetch_argv(source, dest, records_only=records_only), check=False)
    if done.returncode != 0:
        print(f"benchmarks fetch: rsync exited {done.returncode}", file=sys.stderr)
        return done.returncode
    if records_only:
        leaks = store_leaks(dest)
        if leaks:
            print(
                "benchmarks fetch: the store set names private paths:\n  "
                + "\n  ".join(leaks),
                file=sys.stderr,
            )
            return 1
    return 0
