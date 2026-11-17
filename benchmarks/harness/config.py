"""Site files and campaign files of the benchmark suite.

A **site file** says where and as whom the suite runs: Slurm account, QOS per
purpose, node constraint, cluster and stack roots, the inputs of the ``wheels``
stack kind, data and mesh-store directories, the transfer host, the output
directory and the effort's core-hour ceiling. It lives outside the repository and is
found through ``--site`` or ``$GFEM_BENCH_SITE``; only ``benchmarks/site.example.toml``
is committed. The one placeholder is ``{root}``, replaced by ``[cluster] root``.

A **campaign file** (``benchmarks/campaigns/*.toml``) lists the matrix of suite
records: cases, solvers (``code:kind``), software stacks, layouts, the core budget,
repeats, node class, anchor tolerance, wall caps per case, tolerance ladders and job
grouping. The software stacks are the ``[stacks.<name>]`` tables (an empty
``[stacks.wheels]`` selects a stack at its defaults); each may hold ``versions``
overrides (``[stacks.wheels.versions] palace = "0.19.0"``) and an ``env`` table.

Both loaders refuse unknown keys, missing keys and values outside the registries,
and every ``env`` table refuses the affinity variables the harness sets itself.
Standard library only (ADR 0400).
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[1]
CASES_DIR = PACKAGE / "cases"
SITE_ENV = "GFEM_BENCH_SITE"

SOLVERS = ("gfem:direct", "gfem:petsc-mumps", "gfem:bddc", "palace:mumps", "palace:pmg")
STACKS = ("wheels", "native-oss", "native-intel")
NAMED_LAYOUTS = ("default", "best")
NODE_CLASSES = ("standard", "highmem")
JOB_GROUPS = ("case", "stack")
VERSIONED = ("palace", "ngsolve", "petsc")
AFFINITY_VARIABLES = (
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
)
DEFAULT_CEILING_CORE_H = 2000.0

_LAYOUT = re.compile(r"^([1-9]\d*)x([1-9]\d*)$")
_PLACEHOLDER = re.compile(r"\{[^}]*\}")


class ConfigError(ValueError):
    """A site or campaign file the suite refuses."""


# --- shared checks ---------------------------------------------------------------


def check_env(env: Mapping[str, object], *, where: str) -> None:
    """Refuse affinity variables and non-string values in an ``env`` table.

    The harness exports ``OMP_PLACES=cores OMP_PROC_BIND=close`` itself and unsets
    the other affinity variables, so no stack, site or campaign may set them.
    """
    bad = sorted(set(env) & set(AFFINITY_VARIABLES))
    if bad:
        raise ConfigError(
            f"{where}: env sets {', '.join(bad)}; the harness owns thread affinity"
        )
    for name, value in env.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where}: env {name} must be a string")


def _keys(table: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)} "
            f"(allowed: {', '.join(allowed)})"
        )


def _required(table: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{where}: missing key {key}")
    return table[key]


def _table(table: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = table.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{where}{key} must be a table")
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    return value


def _strs(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of strings")
    return tuple(value)


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: no such file") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error


# --- site file -------------------------------------------------------------------

_SITE_KEYS = ("name", "slurm", "cluster", "stacks", "data", "ship", "output", "budget")
_SITE_KEYS_ENV = (*_SITE_KEYS, "env")
_SLURM_KEYS = ("account", "qos", "constraint")
_STACKS_KEYS_FIXED = ("root",)
_WHEELS_KEYS = (
    "base_venv",
    "wheel_dirs",
    "petsc_wheelhouse",
    "palace_wheelhouse",
    "python",
)


@dataclass(frozen=True)
class Site:
    """A loaded site file, ``{root}`` filled in."""

    path: Path
    name: str
    cluster_root: str
    data_dir: str
    output_dir: str
    stacks_root: str | None = None
    stacks: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Inputs per wheels-kind stack name (``[stacks.<name>]``)."""
    data_stores: tuple[str, ...] = ()
    ship_host: str | None = None
    ship_mesh_store: str | None = None
    account: str | None = None
    qos: dict[str, str] = field(default_factory=dict)
    """QOS per purpose, e.g. ``debug`` and ``long``."""
    constraint: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    ceiling_core_h: float = DEFAULT_CEILING_CORE_H
    prior_core_h: float = 0.0


def site_path(path: Path | None, environ: Mapping[str, str] | None = None) -> Path:
    """The site file named by ``--site``, else by ``$GFEM_BENCH_SITE``."""
    environ = os.environ if environ is None else environ
    if path is not None:
        return path
    if environ.get(SITE_ENV):
        return Path(environ[SITE_ENV])
    raise ConfigError(f"no site file: pass --site FILE or set ${SITE_ENV}")


def load_site(path: Path | None, environ: Mapping[str, str] | None = None) -> Site:
    """Load and validate a site file (``--site`` path, else ``$GFEM_BENCH_SITE``)."""
    path = site_path(path, environ)
    raw = _read(path)
    where = str(path)
    _keys(raw, _SITE_KEYS_ENV, where)

    cluster = _table(raw, "cluster", f"{where}: ")
    _keys(cluster, ("root",), f"{where} [cluster]")
    root = _str(_required(cluster, "root", f"{where} [cluster]"), "cluster.root")

    def fill(value: object, key: str) -> str:
        text = _str(value, key)
        for placeholder in _PLACEHOLDER.findall(text):
            if placeholder != "{root}":
                raise ConfigError(
                    f"{where}: {key} uses {placeholder}; only {{root}} is a placeholder"
                )
        return text.replace("{root}", root)

    slurm = _table(raw, "slurm", f"{where}: ")
    _keys(slurm, _SLURM_KEYS, f"{where} [slurm]")
    qos = _table(slurm, "qos", f"{where}: slurm.")
    qos = {
        purpose: _str(value, f"slurm.qos.{purpose}") for purpose, value in qos.items()
    }

    stacks = _table(raw, "stacks", f"{where}: ")
    stacks_root = stacks.get("root")
    wheels: dict[str, dict[str, Any]] = {}
    for name, inputs in stacks.items():
        if name in _STACKS_KEYS_FIXED:
            continue
        if not isinstance(inputs, dict):
            raise ConfigError(f"{where}: stacks.{name} must be a table")
        _keys(inputs, _WHEELS_KEYS, f"{where} [stacks.{name}]")
        filled: dict[str, Any] = {}
        for key, value in inputs.items():
            label = f"stacks.{name}.{key}"
            if isinstance(value, list):
                filled[key] = [fill(v, label) for v in _strs(value, label)]
            else:
                filled[key] = fill(value, label)
        wheels[name] = filled

    data = _table(raw, "data", f"{where}: ")
    _keys(data, ("dir", "stores"), f"{where} [data]")
    ship = _table(raw, "ship", f"{where}: ")
    _keys(ship, ("host", "mesh_store"), f"{where} [ship]")
    output = _table(raw, "output", f"{where}: ")
    _keys(output, ("dir",), f"{where} [output]")
    budget = _table(raw, "budget", f"{where}: ")
    _keys(budget, ("ceiling_core_h", "prior_core_h"), f"{where} [budget]")
    env = _table(raw, "env", f"{where}: ")
    check_env(env, where=f"{where} [env]")

    return Site(
        path=path,
        name=_str(_required(raw, "name", where), "name"),
        cluster_root=root,
        stacks_root=None if stacks_root is None else fill(stacks_root, "stacks.root"),
        stacks=wheels,
        data_dir=fill(_required(data, "dir", f"{where} [data]"), "data.dir"),
        data_stores=tuple(
            fill(v, "data.stores") for v in _strs(data.get("stores", []), "data.stores")
        ),
        output_dir=fill(_required(output, "dir", f"{where} [output]"), "output.dir"),
        ship_host=None if "host" not in ship else _str(ship["host"], "ship.host"),
        ship_mesh_store=None
        if "mesh_store" not in ship
        else fill(ship["mesh_store"], "ship.mesh_store"),
        account=None
        if "account" not in slurm
        else _str(slurm["account"], "slurm.account"),
        qos=qos,
        constraint=None
        if "constraint" not in slurm
        else _str(slurm["constraint"], "slurm.constraint"),
        env=dict(env),
        ceiling_core_h=_number(
            budget.get("ceiling_core_h", DEFAULT_CEILING_CORE_H),
            "budget.ceiling_core_h",
        ),
        prior_core_h=_number(budget.get("prior_core_h", 0.0), "budget.prior_core_h"),
    )


# --- campaign file ---------------------------------------------------------------

_CAMPAIGN_KEYS = (
    "schema",
    "cases",
    "solvers",
    "stacks",
    "layouts",
    "core_budget",
    "repeats",
    "node_class",
    "anchor_tolerance",
    "wall_cap_s",
    "jobs",
    "ladder",
    "cases_dir",
)
_LADDER_KEYS = ("cases", "solvers", "tolerances", "layout")
_STACK_OVERRIDE_KEYS = ("versions", "env")


@dataclass(frozen=True)
class Ladder:
    """Off-anchor tolerance rungs for some (case, solver) pairs, at one layout."""

    cases: tuple[str, ...]
    solvers: tuple[str, ...]
    tolerances: tuple[float, ...]
    layout: str = "default"
    """``default`` or an explicit ``RxT``; never ``best`` (chosen post hoc)."""


@dataclass(frozen=True)
class Campaign:
    """A loaded campaign file."""

    path: Path
    name: str
    """The file stem, part of the campaign id."""
    cases: tuple[str, ...]
    solvers: tuple[str, ...]
    stacks: tuple[str, ...]
    layouts: tuple[str, ...]
    core_budget: int
    repeats: int
    node_class: str
    anchor_tolerance: float
    wall_cap_s: dict[str, float]
    group: str = "case"
    ladder: tuple[Ladder, ...] = ()
    stack_versions: dict[str, dict[str, str]] = field(default_factory=dict)
    stack_env: dict[str, dict[str, str]] = field(default_factory=dict)
    cases_dir: Path | None = None
    """Where the campaign's case directories are; ``None`` = ``benchmarks/cases``.

    Set by ``cases_dir`` in the file (relative to it), for harness fixtures only.
    """


def known_cases(cases_dir: Path = CASES_DIR) -> tuple[str, ...]:
    """Suite case names: directories of ``cases_dir`` holding a ``case.toml``."""
    if not cases_dir.is_dir():
        return ()
    return tuple(
        sorted(p.parent.name for p in cases_dir.glob("*/case.toml") if p.is_file())
    )


def check_layout(layout: str, budget: int, *, where: str) -> None:
    """Refuse a layout other than ``default``, ``best`` or ``RxT`` within budget."""
    if layout in NAMED_LAYOUTS:
        return
    match = _LAYOUT.match(layout)
    if match is None:
        raise ConfigError(
            f"{where}: layout {layout!r} is not default, best or RxT (e.g. 4x4)"
        )
    ranks, threads = int(match[1]), int(match[2])
    if ranks * threads > budget:
        raise ConfigError(
            f"{where}: layout {layout} uses {ranks * threads} cores "
            f"> core budget {budget}"
        )


def _members(
    values: tuple[str, ...], registry: tuple[str, ...], axis: str, where: str
) -> tuple[str, ...]:
    if not values:
        raise ConfigError(f"{where}: {axis} is empty")
    unknown = [v for v in values if v not in registry]
    if unknown:
        raise ConfigError(
            f"{where}: unknown {axis} {', '.join(unknown)} "
            f"(known: {', '.join(registry) or 'none'})"
        )
    return values


def load_campaign(path: Path, *, cases: tuple[str, ...] | None = None) -> Campaign:
    """Load and validate a campaign file; ``cases`` defaults to the committed cases.

    A ``cases_dir`` key (relative to the file) takes the cases from that directory
    instead of ``benchmarks/cases``: the harness's own test fixtures use it.
    """
    raw = _read(path)
    where = str(path)
    _keys(raw, _CAMPAIGN_KEYS, where)
    cases_dir = _cases_dir(raw, path)

    schema = _required(raw, "schema", where)
    if schema != 1:
        raise ConfigError(f"{where}: schema {schema!r} unknown (this harness reads 1)")

    budget = _positive_int(_required(raw, "core_budget", where), "core_budget")
    case_names = _members(
        _strs(_required(raw, "cases", where), "cases"),
        known_cases(cases_dir or CASES_DIR) if cases is None else cases,
        "case",
        where,
    )
    solvers = _members(
        _strs(_required(raw, "solvers", where), "solvers"), SOLVERS, "solver", where
    )
    stack_tables = _table(raw, "stacks", f"{where}: ")
    if "stacks" not in raw or not stack_tables:
        raise ConfigError(f"{where}: no [stacks.<name>] table")
    stacks = _members(tuple(stack_tables), STACKS, "stack", where)
    layouts = _strs(_required(raw, "layouts", where), "layouts")
    if not layouts:
        raise ConfigError(f"{where}: layouts is empty")
    for layout in layouts:
        check_layout(layout, budget, where=where)

    node_class = _str(_required(raw, "node_class", where), "node_class")
    if node_class not in NODE_CLASSES:
        raise ConfigError(
            f"{where}: node_class {node_class!r} not in {', '.join(NODE_CLASSES)}"
        )

    caps = _table(raw, "wall_cap_s", f"{where}: ")
    if "wall_cap_s" not in raw:
        raise ConfigError(f"{where}: missing key wall_cap_s")
    _keys(caps, case_names, f"{where} [wall_cap_s]")
    missing = [c for c in case_names if c not in caps]
    if missing:
        raise ConfigError(f"{where}: [wall_cap_s] has no cap for {', '.join(missing)}")
    wall_cap_s = {c: _number(caps[c], f"wall_cap_s.{c}") for c in case_names}

    jobs = _table(raw, "jobs", f"{where}: ")
    _keys(jobs, ("group",), f"{where} [jobs]")
    group = _str(jobs.get("group", "case"), "jobs.group")
    if group not in JOB_GROUPS:
        raise ConfigError(
            f"{where}: jobs.group {group!r} not in {', '.join(JOB_GROUPS)}"
        )

    ladder = tuple(
        _ladder(rung, case_names, solvers, budget, f"{where} [[ladder]] {i + 1}")
        for i, rung in enumerate(raw.get("ladder", []))
    )

    versions: dict[str, dict[str, str]] = {}
    env: dict[str, dict[str, str]] = {}
    for name, override in stack_tables.items():
        label = f"{where} [stacks.{name}]"
        if not isinstance(override, dict):
            raise ConfigError(f"{label}: must be a table")
        _keys(override, _STACK_OVERRIDE_KEYS, label)
        if "versions" in override:
            pinned = _table(override, "versions", f"{label}: ")
            _keys(pinned, VERSIONED, f"{label}.versions")
            versions[name] = {
                k: _str(v, f"stacks.{name}.versions.{k}") for k, v in pinned.items()
            }
        if "env" in override:
            table = _table(override, "env", f"{label}: ")
            check_env(table, where=f"{label}.env")
            env[name] = dict(table)

    return Campaign(
        path=path,
        name=path.stem,
        cases=case_names,
        solvers=solvers,
        stacks=stacks,
        layouts=layouts,
        core_budget=budget,
        repeats=_positive_int(_required(raw, "repeats", where), "repeats"),
        node_class=node_class,
        anchor_tolerance=_number(
            _required(raw, "anchor_tolerance", where), "anchor_tolerance"
        ),
        wall_cap_s=wall_cap_s,
        group=group,
        ladder=ladder,
        stack_versions=versions,
        stack_env=env,
        cases_dir=cases_dir,
    )


def _cases_dir(raw: Mapping[str, Any], path: Path) -> Path | None:
    if "cases_dir" not in raw:
        return None
    found = (path.parent / _str(raw["cases_dir"], "cases_dir")).resolve()
    if not found.is_dir():
        raise ConfigError(f"{path}: cases_dir {found} is not a directory")
    return found


def _ladder(
    rung: object,
    cases: tuple[str, ...],
    solvers: tuple[str, ...],
    budget: int,
    where: str,
) -> Ladder:
    if not isinstance(rung, dict):
        raise ConfigError(f"{where}: must be a table")
    _keys(rung, _LADDER_KEYS, where)
    tolerances = _required(rung, "tolerances", where)
    if not isinstance(tolerances, list) or not tolerances:
        raise ConfigError(f"{where}: tolerances must be a non-empty list of numbers")
    layout = _str(rung.get("layout", "default"), "ladder.layout")
    if layout == "best":
        raise ConfigError(
            f"{where}: ladder layout best is chosen after the run; use default or RxT"
        )
    check_layout(layout, budget, where=where)
    return Ladder(
        cases=_members(
            _strs(_required(rung, "cases", where), "ladder.cases"),
            cases,
            "case",
            where,
        ),
        solvers=_members(
            _strs(_required(rung, "solvers", where), "ladder.solvers"),
            solvers,
            "solver",
            where,
        ),
        tolerances=tuple(_number(t, "ladder.tolerances") for t in tolerances),
        layout=layout,
    )
