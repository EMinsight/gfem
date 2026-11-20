"""Software stacks of the benchmark suite: committed descriptions resolved per site.

A **stack description** is one TOML file beside this module (``<name>.toml``). It
says what a software stack is and never where: its kind (``wheels`` or ``native``),
closed-source parts, default package versions, and per code it runs (a ``[<code>]``
section for each code of :data:`benchmarks.adapters.ADAPTERS`, e.g. ``[gfem]``,
``[palace]``) the modules or packages to load, fixed environment variables, a table
of thread variables (``"threads"`` = the layout's threads per rank, or a literal
count, clamped by ``caps``), the srun MPI plugin, and the keys the code's adapter
asks for (``stack_keys``: Palace's launcher and binary).

The site file says where the parts live, and a campaign may override versions.
:func:`resolve` (pure, at ``plan``) fills both in and fingerprints the result;
:func:`render` writes the env script a step sources; :func:`refusal` says why a
resolved stack cannot launch; :func:`prepare` (login node, ``submit`` and
``run --local``) builds or validates the envs; :func:`record` is the scrubbed
``stack`` block of a suite record. Each kind is one module of this package
(:mod:`benchmarks.stacks.wheels`, :mod:`benchmarks.stacks.native`) with
``inputs``, ``activation``, ``refusal`` and ``prepare``.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmarks.harness import records
from benchmarks.harness.config import (
    AFFINITY_VARIABLES,
    VERSIONED,
    ConfigError,
    Site,
    check_env,
    known_codes,
    versioned,
)

STACKS_DIR = Path(__file__).resolve().parent
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NGS_NUM_THREADS",
)
"""Unset before a code's own thread table is exported."""
PLACES = {"OMP_PLACES": "cores", "OMP_PROC_BIND": "close"}
"""Exported last for every code and layout."""
LAUNCHERS = ("srun",)
MPI_PLUGINS = ("pmix", "pmi2")
LAUNCH_R1 = ("srun", "taskset")


class StackRefusedError(ConfigError):
    """A resolved software stack that cannot be prepared or launched."""


@dataclass(frozen=True)
class CodeSection:
    """What one code (``gfem``, ``palace``...) of a stack loads and exports."""

    modules: tuple[str, ...] = ()
    """``wheels`` kind: modules loaded after ``module purge``, one per line."""
    packages: tuple[str, ...] = ()
    """``native`` kind: packages whose ``activate.sh`` is sourced, in order."""
    env: dict[str, str] = field(default_factory=dict)
    threads: dict[str, str | int] = field(default_factory=dict)
    """Thread variable -> ``"threads"`` or a literal count."""
    caps: dict[str, int] = field(default_factory=dict)
    mpi: str | None = None
    """srun ``--mpi`` plugin above one rank."""
    launcher: str | None = None
    entry: str | None = None
    python: str | None = None
    """``native`` gfem: the package whose prefix holds the Python."""
    launch_r1: str = "srun"
    """The one-rank launch path: ``srun`` or the ``taskset`` fallback."""


@dataclass(frozen=True)
class Description:
    """A loaded stack description."""

    path: Path
    text: str
    """The file's bytes as text: part of the fingerprint."""
    name: str
    kind: str
    closed: tuple[str, ...]
    versions: dict[str, str]
    codes: dict[str, CodeSection]


@dataclass(frozen=True)
class Resolved:
    """A stack description resolved against a site file and a campaign."""

    description: Description
    commit: str
    versions: dict[str, str]
    inputs: dict[str, Any]
    """Resolved site inputs of the kind (wheels: the ``[stacks.<name>]`` table)."""
    roots: dict[str, str]
    """Placeholder -> site path, replaced back by :func:`scrub`."""
    env: dict[str, str]
    """Site and campaign ``env`` tables, exported after the code's own."""
    fingerprint: str
    """sha256 over the description, versions, site inputs and extra env."""
    env_dir: str
    """``{cluster.root}/envs/bench-<sha7>-<stack>-<fp8>``: gfem's env, env scripts."""
    code_env: dict[str, str] = field(default_factory=dict)
    """``wheels`` kind: the venv each code's ``bin`` comes from."""
    prefixes: dict[str, str] = field(default_factory=dict)
    """``native`` kind: package -> install prefix."""

    @property
    def name(self) -> str:
        """The stack's name."""
        return self.description.name

    @property
    def kind(self) -> str:
        """The stack's kind (``wheels`` or ``native``)."""
        return self.description.kind


# --- descriptions ----------------------------------------------------------------


def names(stacks_dir: Path | None = None) -> tuple[str, ...]:
    """The committed software stacks: the ``*.toml`` files of ``stacks_dir``."""
    stacks_dir = STACKS_DIR if stacks_dir is None else stacks_dir
    return tuple(sorted(p.stem for p in stacks_dir.glob("*.toml")))


_TOP_KEYS = ("schema", "name", "kind", "closed", "versions")
_SECTION_KEYS = {
    "wheels": ("modules", "env", "threads", "mpi", "launcher", "entry", "launch_r1"),
    "native": (
        "packages",
        "env",
        "threads",
        "mpi",
        "launcher",
        "entry",
        "python",
        "launch_r1",
    ),
}


def _keys(table: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)} "
            f"(allowed: {', '.join(allowed)})"
        )


def _strs(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of strings")
    return tuple(value)


def _choice(value: object, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{where} must be one of {', '.join(allowed)}")
    return str(value)


def _count(value: object, where: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f'{where} must be "threads" or a positive count')
    return value


def _threads(table: object, where: str) -> tuple[dict[str, str | int], dict[str, int]]:
    if not isinstance(table, dict):
        raise ConfigError(f"{where} must be a table")
    caps_raw = table.get("caps", {})
    if not isinstance(caps_raw, dict):
        raise ConfigError(f"{where}.caps must be a table")
    threads: dict[str, str | int] = {}
    for var, value in table.items():
        if var == "caps":
            continue
        if var in AFFINITY_VARIABLES:
            raise ConfigError(f"{where}: {var} is set by the harness, not a stack")
        threads[var] = (
            "threads" if value == "threads" else _count(value, f"{where}.{var}")
        )
    caps = {var: _count(cap, f"{where}.caps.{var}") for var, cap in caps_raw.items()}
    stray = sorted(set(caps) - set(threads))
    if stray:
        raise ConfigError(f"{where}: caps {', '.join(stray)} not in the thread table")
    return threads, caps


def _stack_keys(code: str) -> tuple[str, ...]:
    from benchmarks.adapters import ADAPTERS  # the adapters import this module

    return ADAPTERS[code].stack_keys


def _section(raw: object, kind: str, code: str, where: str) -> CodeSection:
    where = f"{where} [{code}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: must be a table")
    _keys(raw, _SECTION_KEYS[kind], where)
    env = raw.get("env", {})
    if not isinstance(env, dict):
        raise ConfigError(f"{where}: env must be a table")
    check_env(env, where=where)
    clash = sorted(set(env) & set(THREAD_VARIABLES))
    if clash:
        raise ConfigError(f"{where}: env sets {', '.join(clash)}; use the thread table")
    if "threads" not in raw:
        raise ConfigError(f"{where}: missing table [{code}.threads]")
    threads, caps = _threads(raw["threads"], f"{where}.threads")
    if "mpi" not in raw:
        raise ConfigError(f"{where}: missing key mpi")
    section = CodeSection(
        modules=_strs(raw.get("modules", []), f"{where} modules"),
        packages=_strs(raw.get("packages", []), f"{where} packages"),
        env=dict(env),
        threads=threads,
        caps=caps,
        mpi=_choice(raw["mpi"], MPI_PLUGINS, f"{where} mpi"),
        launcher=None
        if "launcher" not in raw
        else _choice(raw["launcher"], LAUNCHERS, f"{where} launcher"),
        entry=None if "entry" not in raw else str(raw["entry"]),
        python=None if "python" not in raw else str(raw["python"]),
        launch_r1=_choice(
            raw.get("launch_r1", "srun"), LAUNCH_R1, f"{where} launch_r1"
        ),
    )
    missing = [k for k in _stack_keys(code) if getattr(section, k) is None]
    if missing:
        raise ConfigError(f"{where}: {code} needs {' and '.join(missing)}")
    if kind == "native" and not section.packages:
        raise ConfigError(f"{where}: a native stack names its packages")
    return section


def load(name: str, stacks_dir: Path | None = None) -> Description:
    """Load and validate the committed description of software stack ``name``.

    A stack runs the codes it has a ``[<code>]`` section for.
    """
    stacks_dir = STACKS_DIR if stacks_dir is None else stacks_dir
    path = stacks_dir / f"{name}.toml"
    if not path.is_file():
        raise ConfigError(
            f"unknown stack {name!r} (known: {', '.join(names(stacks_dir))})"
        )
    text = path.read_text()
    where = str(path)
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{where}: {error}") from error
    _keys(raw, (*_TOP_KEYS, *known_codes()), where)
    if raw.get("schema") != 1:
        raise ConfigError(f"{where}: schema {raw.get('schema')!r} unknown (reads 1)")
    if raw.get("name") != name:
        raise ConfigError(f"{where}: name {raw.get('name')!r} is not the file stem")
    kind = _choice(raw.get("kind"), tuple(KINDS), f"{where} kind")
    versions = raw.get("versions", {})
    if not isinstance(versions, dict):
        raise ConfigError(f"{where}: versions must be a table")
    _keys(versions, versioned(), f"{where} [versions]")
    missing = [p for p in VERSIONED if p not in versions]
    if missing:
        raise ConfigError(f"{where}: [versions] lacks {', '.join(missing)}")
    sections = {
        code: _section(raw[code], kind, code, where)
        for code in known_codes()
        if code in raw
    }
    for code, section in sections.items():
        unversioned = [p for p in section.packages if p not in versions]
        if unversioned:
            raise ConfigError(
                f"{where} [{code}]: packages {', '.join(unversioned)} have no version"
            )
    return Description(
        path=path,
        text=text,
        name=name,
        kind=kind,
        closed=_strs(raw.get("closed", []), f"{where} closed"),
        versions={k: str(v) for k, v in versions.items()},
        codes=sections,
    )


# --- resolve, render, refuse, prepare ---------------------------------------------


def _kind(resolved: Resolved | Description) -> ModuleType:
    return KINDS[resolved.kind]


def resolve(
    name: str,
    site: Site,
    versions: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    commit: str,
    stacks_dir: Path | None = None,
) -> Resolved:
    """Resolve stack ``name`` against ``site`` with a campaign's overrides.

    ``versions`` and ``env`` are the campaign's ``[stacks.<name>]`` tables; the
    site's ``[env]`` is exported before the campaign's. Pure: reads only the
    committed description.
    """
    description = load(name, stacks_dir)
    versions = dict(versions or {})
    _keys(versions, versioned(), f"campaign [stacks.{name}.versions]")
    merged = {**description.versions, **versions}
    extra = {**site.env, **dict(env or {})}
    check_env(extra, where=f"site and campaign env of stack {name}")
    kind = _kind(description)
    inputs, roots = kind.inputs(description, merged, site)
    roots = {**roots, "{cluster.root}": site.cluster_root}
    fingerprint = hashlib.sha256(
        json.dumps([description.text, merged, inputs, extra], sort_keys=True).encode()
    ).hexdigest()
    env_dir = f"{site.cluster_root}/envs/bench-{commit[:7]}-{name}-{fingerprint[:8]}"
    return Resolved(
        description=description,
        commit=commit,
        versions=merged,
        inputs=inputs,
        roots=roots,
        env=extra,
        fingerprint=fingerprint,
        env_dir=env_dir,
        **kind.paths(description, merged, site, env_dir),
    )


def thread_exports(section: CodeSection, threads: int) -> dict[str, int]:
    """The thread variables of ``section`` at ``threads`` per rank, caps applied."""
    if threads < 1:
        raise ValueError(f"threads per rank must be positive, not {threads}")
    values = {}
    for var, value in section.threads.items():
        count = threads if value == "threads" else int(value)
        values[var] = min(count, section.caps.get(var, count))
    return values


def render(resolved: Resolved, code: str, threads: int | None = None) -> str:
    """The env script ``code`` sources; with ``threads``, the step's full script.

    Order: ``module purge``; the kind's activation (modules and ``PATH``, or each
    package's ``activate.sh``); unset thread and affinity variables; the code's
    ``env``, then the site's and campaign's; thread exports for ``threads`` per rank;
    ``OMP_PLACES=cores OMP_PROC_BIND=close``. Without ``threads`` it stops after the
    env exports: the static part written at prepare time. Never ``LD_LIBRARY_PATH``.
    """
    section = resolved.description.codes.get(code)
    if section is None:
        known = ", ".join(resolved.description.codes)
        raise ConfigError(
            f"stack {resolved.name} has no [{code}] section (runs: {known})"
        )
    lines = ["module purge >/dev/null 2>&1"]
    lines += _kind(resolved).activation(resolved, code)
    lines.append("unset " + " ".join((*THREAD_VARIABLES, *AFFINITY_VARIABLES)))
    for var, value in {**section.env, **resolved.env}.items():
        lines.append(f"export {var}={shlex.quote(value)}")
    if threads is not None:
        for var, count in thread_exports(section, threads).items():
            lines.append(f"export {var}={count}")
        lines.append(" ".join(["export", *(f"{k}={v}" for k, v in PLACES.items())]))
    return "\n".join(lines) + "\n"


def refusal(resolved: Resolved) -> str | None:
    """Why ``resolved`` cannot launch now, or ``None``; reads the site's files."""
    return _kind(resolved).refusal(resolved)


def prepare(resolved: Resolved, **options: Any) -> Path:
    """Prepare ``resolved`` on the login node; idempotent and offline.

    Returns the env directory, which holds ``env-<code>.sh`` (the static scripts).
    Raises :class:`StackRefusedError` when the stack cannot be prepared. Options are
    the kind's (``wheels``: ``wheel``, the gfem wheel of the commit).
    """
    env_dir = _kind(resolved).prepare(resolved, **options)
    for code in resolved.description.codes:
        (env_dir / f"env-{code}.sh").write_text(render(resolved, code))
    return env_dir


# --- records -----------------------------------------------------------------------


def scrub(text: str, resolved: Resolved) -> str:
    """``text`` with each resolved site root replaced by its key placeholder.

    Longer roots go first, so ``{stacks.wheels.base_venv}`` wins over
    ``{cluster.root}`` for a path under both. Paths outside the roots are left for
    the record scrub to refuse (one implementation:
    :func:`benchmarks.harness.records.replace_roots`).
    """
    return records.replace_roots(text, resolved.roots)


def description_json(resolved: Resolved) -> dict[str, Any]:
    """The resolved description: stack file after overrides, site inputs filled."""
    description = resolved.description
    return {
        "name": description.name,
        "kind": description.kind,
        "closed": list(description.closed),
        "versions": dict(resolved.versions),
        "codes": {
            code: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in vars(section).items()
            }
            for code, section in description.codes.items()
        },
        "inputs": resolved.inputs,
        "env": dict(resolved.env),
        "env_dir": resolved.env_dir,
        "code_env": dict(resolved.code_env),
        "prefixes": dict(resolved.prefixes),
    }


def record(
    resolved: Resolved,
    *,
    env_script: str,
    provenance: Mapping[str, Any] | None = None,
    modules: str = "",
    ldd: str = "",
) -> dict[str, Any]:
    """The scrubbed ``stack`` block of a suite record.

    ``env_script`` is the script the step sourced, ``provenance`` what
    :func:`provenance` read after prepare, ``modules`` and ``ldd`` the job's
    ``module -t list`` and BLAS line. Every string is scrubbed of site roots.
    """
    block = {
        "name": resolved.name,
        "kind": resolved.kind,
        "closed": list(resolved.description.closed),
        "versions": dict(resolved.versions),
        "fingerprint": resolved.fingerprint,
        "description": description_json(resolved),
        "provenance": dict(provenance or {}),
        "env": env_script,
        "modules": modules,
        "ldd": ldd,
    }
    return json.loads(scrub(json.dumps(block), resolved))


def provenance(resolved: Resolved) -> dict[str, Any]:
    """What the prepared stack's own files say it is (see each kind's docs)."""
    return _kind(resolved).provenance(resolved)


from benchmarks.stacks import native, wheels  # noqa: E402  (kinds use the types above)

KINDS: dict[str, ModuleType] = {"wheels": wheels, "native": native}
"""Stack kind -> module with ``inputs``, ``paths``, ``activation``, ``refusal``,
``prepare`` and ``provenance``."""
