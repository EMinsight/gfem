"""The Palace solver adapter and its version dialects (ADRs 0404, 0406).

Everything that depends on the Palace version lives in :data:`DIALECTS`, keyed by
``major.minor``: one frozen :class:`Dialect` per version with three parts.

- **render** (:class:`Render`): the case and run become ``config.json``. Both
  dialects pin ``Refinement`` off, ``Absorbing.Order`` from the case (only order 1
  is supported, the order-2 sign differs between versions), ``Solver.Linear.Type``
  always, ``MGMaxLevels 1`` and ``ComplexCoarseSolve`` for the direct kind, the
  estimator lightened (``EstimatorTol 1e-2``, ``EstimatorMG``), ``MaxIts 2000``,
  eigen ``Tol 1e-10`` with inner linear ``Tol 1e-12``, and no field output. They
  never write ``AbsTol`` (0.19 only, default 0) or ``NSamples`` (gone in 0.19).
  0.18 adds ``Model.RefineCrackElements: false``; 0.19 writes no crack key.
- **launch** (:class:`LaunchRules`): the stack's ``entry`` (``palace-real``, never
  the wheel's wrapper, whose thread policy changed in 0.19) under the suite's one
  srun line; thread variables come from the stack's env script.
- **parse** (:class:`Parse`): ``palace.json`` ``ElapsedTime.Durations`` through an
  explicit key table into record buckets (an unknown key fails the parse; 0.19
  added three wave-port rows), the stdout "Elapsed Time Report" for the max over
  ranks of each row when there is more than one rank (``rank_reduction:
  "max-of-rows"``, an upper bound; rank 0's buckets kept in ``rank0``), the
  ``Level k (p = n): N unknowns`` DOF line, GMRES (non-)convergence lines and the
  ``--version`` text.

Palace's timer rows are exclusive, so rank 0's buckets partition ``Total``; the
untimed rest goes to ``other_s``. A run is complete when ``ElapsedTime`` exists.
Peak memory is ``PeakNodeMemoryMegabytes.Total``, a sum of per-rank peaks; the rank
max is ``PeakMemoryMegabytes.Max``, the largest node ``PeakNodeMemoryMegabytes.Max``.
``inputs_match`` reads ``config_resolved.json`` back through the case importer and
compares it with the case. Standard library only (ADR 0400).
"""

from __future__ import annotations

import cmath
import csv
import functools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
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
from benchmarks.harness.case import UNITS, Case, CaseError, import_palace
from benchmarks.stacks import Resolved

CODE = "palace"
CONFIG = "config.json"
OUTPUT = "output"
STDOUT = "palace.stdout"
VERSION_STDOUT = "palace-version.txt"
MIB_PER_GIB = 1024.0
#: Boundary kinds of a case this adapter renders.
BOUNDARY_KINDS = ("pec", "conductor", "absorbing", "impedance", "periodic")
#: Pinned solver settings (ADR 0411): direct and eigen tolerances, caps, estimator.
DIRECT_TOL = 1e-10
EIGEN_TOL = 1e-10
EIGEN_LINEAR_TOL = 1e-12
MAX_ITS = 2000
ESTIMATOR = {"EstimatorTol": 1e-2, "EstimatorMG": True}
PMG_TYPE = "SuperLU"
"""What the wheel resolves ``Type`` to when it is omitted: pinned, so unchanged."""
REL_TOL = 1e-12
"""Relative tolerance of the case comparisons (frequencies, targets, values)."""


class PalaceParseError(ValueError):
    """Palace output the dialect cannot read (an unknown timer key...)."""


# --- dialects ------------------------------------------------------------------


@dataclass(frozen=True)
class Render:
    """What a dialect adds to the rendered config."""

    model: Mapping[str, Any] = field(default_factory=dict)
    """Extra ``Model`` keys (0.18: ``RefineCrackElements: false``)."""


@dataclass(frozen=True)
class LaunchRules:
    """How a dialect's binary is started."""

    entry: str = "palace-real"
    """Default binary when the stack names none."""
    version_flag: str = "--version"


@dataclass(frozen=True)
class Parse:
    """Where a dialect's outputs keep what the suite reads."""

    buckets: Mapping[str, str]
    """``ElapsedTime.Durations`` key -> record bucket (``Total`` apart)."""
    table_title: str = "Elapsed Time Report (s)"


@dataclass(frozen=True)
class Dialect:
    """One Palace version's render, launch and parse rules."""

    name: str
    render: Render
    launch: LaunchRules
    parse: Parse


_BUCKETS_0_18 = {
    "Initialization": "mesh_read_s",
    "MeshPreprocessing": "mesh_read_s",
    "OperatorConstruction": "assemble_s",
    "WavePorts": "assemble_s",
    "Setup": "setup_s",
    "LinearSolve": "solve_s",
    "Preconditioner": "solve_s",
    "CoarseSolve": "solve_s",
    "TimeStepping": "solve_s",
    "EigenvalueSolve": "solve_s",
    "Div.-FreeProjection": "solve_s",
    "PROMConstruction": "rom_s",
    "PROMSolve": "rom_s",
    "Estimation": "estimate_s",
    "Construction": "estimate_s",
    "Solve": "estimate_s",
    "Adaptation": "other_s",
    "Rebalancing": "other_s",
    "Postprocessing": "postpro_s",
    "FarFields": "postpro_s",
    "Paraview": "postpro_s",
    "Gridfunction": "postpro_s",
    "DiskIO": "postpro_s",
}
_BUCKETS_0_19 = {
    **_BUCKETS_0_18,
    # wave-port sub-timers carved out of WavePorts (research D1)
    "Eigenproblem": "assemble_s",
    "Fieldreconstruction": "assemble_s",
    "Modepostprocessing": "assemble_s",
}

DIALECTS: dict[str, Dialect] = {
    "0.18": Dialect(
        name="0.18",
        render=Render(model={"RefineCrackElements": False}),
        launch=LaunchRules(),
        parse=Parse(buckets=_BUCKETS_0_18),
    ),
    "0.19": Dialect(
        name="0.19",
        render=Render(),
        launch=LaunchRules(),
        parse=Parse(buckets=_BUCKETS_0_19),
    ),
}

#: Record buckets a Palace run fills; ``discretize_s`` stays ``None`` (space
#: construction is inside ``OperatorConstruction``).
PHASES = (
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
_FILLED = tuple(p for p in PHASES if p != "discretize_s")


_RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def release(version: str) -> str:
    """``"0.18.1.post5"`` -> ``"0.18.1"``; refuses text that is not a version."""
    match = _RELEASE.match(version.strip())
    if match is None:
        raise RunRefusedError(f"Palace version {version!r} is not X.Y.Z")
    return ".".join(match.groups())


def dialect_of(version: str) -> Dialect:
    """The dialect of a Palace version; refused when none matches."""
    key = ".".join(release(version).split(".")[:2])
    if key not in DIALECTS:
        known = ", ".join(DIALECTS)
        raise RunRefusedError(f"no Palace dialect for {version} (known: {known})")
    return DIALECTS[key]


def stack_dialect(stack: Resolved) -> Dialect:
    """The dialect of the Palace version a stack declares."""
    if "palace" not in stack.versions:
        raise RunRefusedError(f"stack {stack.name} declares no Palace version")
    return dialect_of(stack.versions["palace"])


# --- version ---------------------------------------------------------------------

_VERSION_LINE = re.compile(r"^Palace version:\s*(\S+)", re.MULTILINE)
_SCHEMA_LINE = re.compile(r"^Schema version:\s*(\S+)", re.MULTILINE)
_BUILD_LINE = re.compile(r"^Build:\s*(.+)$", re.MULTILINE)
_DEPENDENCY = re.compile(r"^\s+([^:]+):\s*(.+)$")


def parse_version(text: str) -> dict[str, Any]:
    """``--version`` text -> ``version`` (tag), ``schema``, ``build``, ``dependencies``.

    0.18 prints the tag and schema; 0.19 adds a build id and every dependency.
    """
    tag = _VERSION_LINE.search(text)
    if tag is None:
        raise PalaceParseError("no 'Palace version:' line in the --version output")
    out: dict[str, Any] = {"version": tag[1], "text": text}
    schema = _SCHEMA_LINE.search(text)
    if schema is not None:
        out["schema"] = schema[1]
    build = _BUILD_LINE.search(text)
    if build is not None:
        out["build"] = build[1].strip()
    lines = text.splitlines()
    if "Dependencies:" in lines:
        deps = {}
        for line in lines[lines.index("Dependencies:") + 1 :]:
            match = _DEPENDENCY.match(line)
            if match is None:
                break
            deps[match[1].strip()] = match[2].strip()
        out["dependencies"] = deps
    return out


def version_refusal(stack: Resolved, text: str) -> str | None:
    """Why the installed Palace (its ``--version`` text) does not match the stack."""
    declared = stack.versions.get("palace")
    if declared is None:
        return f"stack {stack.name} declares no Palace version"
    try:
        installed = parse_version(text)["version"]
        if release(installed) != release(declared):
            return (
                f"Palace {installed} installed, stack {stack.name} declares {declared}"
            )
        dialect_of(installed)
    except (PalaceParseError, RunRefusedError) as error:
        return str(error)
    return None


# --- entry and launch ------------------------------------------------------------


def entry(stack: Resolved) -> str:
    """The Palace binary of a stack.

    ``wheels``: ``palace_solver/bin/<entry>`` inside the stack's Palace venv (the
    wrapper in ``bin/`` is never used); ``native``: the bare name, on ``PATH``
    after the packages' ``activate.sh``.
    """
    section = stack.description.codes.get(CODE)
    if section is None:
        raise RunRefusedError(f"stack {stack.name} has no [palace] section")
    name = section.entry or stack_dialect(stack).launch.entry
    if stack.kind != "wheels":
        return name
    venv = Path(stack.code_env[CODE])
    found = sorted(venv.glob(f"lib/python3*/site-packages/palace_solver/bin/{name}"))
    if len(found) != 1:
        what = "no" if not found else f"{len(found)}"
        raise RunRefusedError(f"stack {stack.name}: {what} {name} under {venv}")
    return str(found[0])


def _level_ndof(order: int, line: str) -> int | None:
    match = _LEVEL.match(line)
    if match is None or int(match[2]) != order:
        return None
    return int(match[3])


_LEVEL = re.compile(r"^\s*Level (\d+) \(p = (\d+)\): (\d+) unknowns\s*$")


def dof_watch(order: int) -> Callable[[str], int | None]:
    """The DOF guard's line parser: the non-auxiliary ``Level`` line at ``order``."""
    return functools.partial(_level_ndof, order)


def stdout_ndof(text: str, order: int) -> int | None:
    """The last DOF count stdout reports at ``order`` (``None`` if none)."""
    counts = [n for line in text.splitlines() if (n := _level_ndof(order, line))]
    return counts[-1] if counts else None


# --- render ------------------------------------------------------------------------


def _tags(case: Case, regions: Sequence[str]) -> list[int]:
    return [case.regions[r][1] for r in regions]


def _material(case: Case, index: int) -> dict[str, Any]:
    m = case.materials[index]

    def value(v: float | tuple[float, ...]) -> float | list[float]:
        return list(v) if isinstance(v, tuple) else v

    out: dict[str, Any] = {
        "Attributes": _tags(case, [m.region]),
        "Permittivity": value(m.eps_r),
    }
    if m.mu_r is not None:
        out["Permeability"] = value(m.mu_r)
    if m.tan_delta is not None:
        out["LossTan"] = value(m.tan_delta)
    if m.conductivity is not None:
        out["Conductivity"] = m.conductivity
    if m.axes is not None:
        out["MaterialAxes"] = [list(a) for a in m.axes]
    return out


def _boundaries(case: Case) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_kind: dict[str, list[Any]] = {}
    for b in case.boundaries:
        by_kind.setdefault(b.kind, []).append(b)
    if "pec" in by_kind:
        out["PEC"] = {"Attributes": _tags(case, [b.region for b in by_kind["pec"]])}
    if "conductor" in by_kind:
        out["Conductivity"] = [
            {
                "Attributes": _tags(case, [b.region]),
                "Conductivity": b.settings["conductivity"],
                "Thickness": b.settings["thickness"],
            }
            for b in by_kind["conductor"]
        ]
    if "absorbing" in by_kind:
        out["Absorbing"] = {
            "Attributes": _tags(case, [b.region for b in by_kind["absorbing"]]),
            "Order": by_kind["absorbing"][0].settings.get("order", 1),
        }
    if "impedance" in by_kind:
        out["Impedance"] = [
            {
                "Attributes": _tags(case, [b.region]),
                **{
                    key: b.settings[name]
                    for name, key in (("rs_ohm", "Rs"), ("ls_h", "Ls"), ("cs_f", "Cs"))
                    if name in b.settings
                },
            }
            for b in by_kind["impedance"]
        ]
    if "periodic" in by_kind:
        periodic: dict[str, Any] = {
            "BoundaryPairs": [
                {
                    "DonorAttributes": _tags(case, [b.settings["donor"]]),
                    "ReceiverAttributes": _tags(case, [b.settings["receiver"]]),
                    "Translation": list(b.settings["translation"]),
                }
                for b in by_kind["periodic"]
            ]
        }
        if case.problem.wave_vector is not None:
            length = UNITS[case.mesh.length_unit]
            # the case keeps rad/m; Palace reads rad per mesh length unit
            periodic["FloquetWaveVector"] = [
                k * length for k in case.problem.wave_vector
            ]
        out["Periodic"] = periodic
    _ports(case, out)
    return out


def _ports(case: Case, out: dict[str, Any]) -> None:
    excite = set(case.problem.excite)
    lumped, wave, pec = [], [], []
    for i, port in enumerate(case.ports, start=1):
        entry: dict[str, Any] = {"Index": i}
        if port.name in excite:
            entry["Excitation"] = i
        if port.kind == "lumped":
            for name, key in (
                ("impedance_ohm", "R"),
                ("inductance_h", "L"),
                ("capacitance_f", "C"),
            ):
                if name in port.settings:
                    entry[key] = port.settings[name]
            entry["Elements"] = [
                {
                    "Attributes": _tags(case, [e.region]),
                    "Direction": e.direction.upper(),
                }
                for e in port.elements
            ]
            lumped.append(entry)
        else:
            entry["Attributes"] = _tags(case, [port.settings["region"]])
            entry["Mode"] = port.settings.get("mode", 1)
            entry["Offset"] = port.settings.get("offset", 0.0)
            wave.append(entry)
            pec += [r for r in port.settings.get("pec_regions", []) if r not in pec]
    if lumped:
        out["LumpedPort"] = lumped
    if wave:
        out["WavePort"] = wave
    if pec:
        out["WavePortPEC"] = {"Attributes": _tags(case, pec)}


def _linear(case: Case, kind: str, tolerance: float | None) -> dict[str, Any]:
    eigen = case.problem.kind == "eigen"
    if kind == "mumps":
        linear: dict[str, Any] = {
            "Type": "MUMPS",
            "MGMaxLevels": 1,
            "ComplexCoarseSolve": True,
            "Tol": EIGEN_LINEAR_TOL if eigen else DIRECT_TOL,
        }
    else:
        if not eigen and tolerance is None:
            raise RunRefusedError("palace:pmg needs a run tolerance")
        linear = {"Type": PMG_TYPE, "Tol": EIGEN_LINEAR_TOL if eigen else tolerance}
    return {**linear, "MaxIts": MAX_ITS, **ESTIMATOR}


def render(
    case: Case,
    kind: str,
    dialect: Dialect,
    *,
    mesh: Path,
    output: Path,
    tolerance: float | None,
) -> dict[str, Any]:
    """The Palace config of one run of ``case`` with ``kind`` at ``dialect``."""
    problem = case.problem
    solver: dict[str, Any] = {"Order": case.order}
    if problem.kind == "driven":
        freqs = [f / 1e9 for f in problem.frequencies()]
        solver["Driven"] = {
            "Samples": [{"Type": "Point", "Freq": freqs, "SaveStep": 0}],
            "AdaptiveTol": 0.0,
        }
    else:
        assert problem.target_hz is not None  # noqa: S101 (validated eigen case)
        solver["Eigenmode"] = {
            "Target": problem.target_hz / 1e9,
            "N": problem.n_modes,
            "Tol": EIGEN_TOL,
            "Save": 0,
        }
    solver["Linear"] = _linear(case, kind, tolerance)
    return {
        "Problem": {
            "Type": "Driven" if problem.kind == "driven" else "Eigenmode",
            "Verbose": 1,
            "Output": str(output),
            "OutputFormats": {"Paraview": False, "GridFunction": False},
        },
        "Model": {
            "Mesh": str(mesh),
            "L0": UNITS[case.mesh.length_unit],
            "Refinement": {"UniformLevels": 0, "MaxIts": 0},
            **dialect.render.model,
        },
        "Domains": {
            "Materials": [_material(case, i) for i in range(len(case.materials))]
        },
        "Boundaries": _boundaries(case),
        "Solver": solver,
    }


# --- parse -------------------------------------------------------------------------


def _bucket(key: str, dialect: Dialect) -> str:
    if key not in dialect.parse.buckets:
        raise PalaceParseError(
            f"unknown Palace {dialect.name} timer key {key!r}: map it in the dialect"
        )
    return dialect.parse.buckets[key]


def buckets(
    durations: Mapping[str, float], dialect: Dialect
) -> dict[str, float | None]:
    """Rank 0's ``ElapsedTime.Durations`` as record buckets partitioning ``Total``."""
    out: dict[str, float | None] = dict.fromkeys(PHASES)
    for phase in _FILLED:
        out[phase] = 0.0
    rows = 0.0
    for key, seconds in durations.items():
        if key == "Total":
            continue
        phase = _bucket(key, dialect)
        out[phase] = (out[phase] or 0.0) + seconds
        rows += seconds
    if "Total" not in durations:
        raise PalaceParseError("ElapsedTime.Durations has no Total")
    out["other_s"] = (out["other_s"] or 0.0) + durations["Total"] - rows
    return out


_ROW = re.compile(r"^(\s*\S.*?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def time_table(text: str, dialect: Dialect) -> dict[str, tuple[float, float, float]]:
    """The last "Elapsed Time Report" of stdout: key -> (min, max, avg); ``{}``."""
    lines = text.splitlines()
    starts = [
        i for i, line in enumerate(lines) if line.startswith(dialect.parse.table_title)
    ]
    if not starts:
        return {}
    rows: dict[str, tuple[float, float, float]] = {}
    for line in lines[starts[-1] + 2 :]:
        if set(line.strip()) <= {"-"}:
            if line.strip():
                continue
            break
        match = _ROW.match(line)
        if match is None:
            break
        key = re.sub(r"\s+", "", match[1])
        rows[key] = (float(match[2]), float(match[3]), float(match[4]))
        if key == "Total":
            break
    if "Total" not in rows:
        return {}
    return rows


def max_of_rows(
    table: Mapping[str, tuple[float, float, float]], dialect: Dialect
) -> dict[str, float | None]:
    """Buckets as the sum of each row's maximum over ranks (an upper bound).

    ``other_s`` adds the part of the slowest ``Total`` no row's maximum covers.
    """
    out: dict[str, float | None] = dict.fromkeys(PHASES)
    for phase in _FILLED:
        out[phase] = 0.0
    rows = 0.0
    for key, (_low, high, _avg) in table.items():
        if key == "Total":
            continue
        phase = _bucket(key, dialect)
        out[phase] = (out[phase] or 0.0) + high
        rows += high
    out["other_s"] = (out["other_s"] or 0.0) + max(0.0, table["Total"][1] - rows)
    return out


_SUMMARY = re.compile(
    r"\b(?:F?GMRES|P?CG) solver (converged|did NOT converge) in (\d+) iterations?"
)
_WARNING = "Linear solver did not converge"


def convergence(text: str) -> tuple[str | None, list[int]]:
    """``converged`` check (``pass``/``fail``/``None``) and iterations per solve."""
    iterations, failed = [], _WARNING in text
    for match in _SUMMARY.finditer(text):
        iterations.append(int(match[2]))
        failed = failed or match[1] != "converged"
    if failed:
        return "fail", iterations
    return ("pass" if iterations else None), iterations


def port_s(path: Path, names: Sequence[str]) -> dict[str, Any]:
    """``port-S.csv`` (dB, degrees) -> frequencies and complex S as ``[re, im]``.

    ``S[i][j]`` is to port ``i`` from port ``j`` (Palace indices, 1-based, are the
    case's port order); keys are ``"to,from"`` port names.
    """
    with path.open() as stream:
        rows = [[c.strip() for c in row] for row in csv.reader(stream) if row]
    header, data = rows[0], rows[1:]
    out: dict[str, Any] = {"frequencies_hz": [float(r[0]) * 1e9 for r in data], "s": {}}
    pair = re.compile(r"^\|S\[(\d+)\]\[(\d+)\]\| \(dB\)$")
    for col, title in enumerate(header):
        match = pair.match(title)
        if match is None:
            continue
        key = f"{names[int(match[1]) - 1]},{names[int(match[2]) - 1]}"
        values = []
        for r in data:
            magnitude = 10.0 ** (float(r[col]) / 20.0)
            z = cmath.rect(magnitude, math.radians(float(r[col + 1])))
            values.append([z.real, z.imag])
        out["s"][key] = values
    return out


def eig(path: Path) -> dict[str, Any]:
    """``eig.csv`` -> modes' ``f_hz`` (``[re, im]``), ``q`` and backward error."""
    with path.open() as stream:
        rows = [[c.strip() for c in row] for row in csv.reader(stream) if row]
    header, data = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}
    return {
        "f_hz": [
            [float(r[col["Re{f} (GHz)"]]) * 1e9, float(r[col["Im{f} (GHz)"]]) * 1e9]
            for r in data
        ],
        "q": [float(r[col["Q"]]) for r in data],
        "error_bkwd": [float(r[col["Error (Bkwd.)"]]) for r in data],
    }


# --- inputs_match ------------------------------------------------------------------

_DEFAULTS = {"mu_r": 1.0, "tan_delta": 0.0, "conductivity": 0.0}


def _same(a: object, b: object) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-300)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _materials(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for m in raw.get("materials", []):
        entry = {k: v for k, v in m.items() if k != "region" and _DEFAULTS.get(k) != v}
        out[m["region"]] = entry
    return out


def _boundary_key(b: Mapping[str, Any]) -> str:
    return json.dumps([b["kind"], b.get("region"), b.get("donor"), b.get("receiver")])


def _boundary_set(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {_boundary_key(b): dict(b) for b in raw.get("boundaries", [])}


def _port_list(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for p in raw.get("ports", []):
        entry = {k: v for k, v in p.items() if k != "name" and v != 0.0}
        out.append(entry)
    return out


def _resolved_cleaned(config: Mapping[str, Any]) -> dict[str, Any]:
    """A resolved config without the empty boundary lists Palace writes."""
    cleaned = dict(config)
    cleaned["Boundaries"] = {
        k: v for k, v in config.get("Boundaries", {}).items() if v not in ([], {})
    }
    return cleaned


def inputs_match(case: Case, resolved: Mapping[str, Any]) -> list[str]:
    """Differences between a run's ``config_resolved.json`` and its case (``[]``).

    The config goes back through the case importer; the order, materials, boundary
    kinds and values, absorbing order, ports (R, signed directions), excitations,
    frequencies (relative 1e-12 after the GHz conversion), the eigen target and
    mode count and the wave vector are compared.
    """
    try:
        back = import_palace(
            _resolved_cleaned(resolved), case.regions, name=case.name, voided=True
        )
    except (CaseError, KeyError, TypeError, ValueError) as error:
        return [f"config_resolved.json does not import: {error}"]
    mine = case.to_dict()
    diffs = []
    if back["order"] != case.order:
        diffs.append(f"order {back['order']} != case {case.order}")
    if not _same(_materials(back), _materials(mine)):
        diffs.append(f"materials {_materials(back)} != case {_materials(mine)}")
    got, want = _boundary_set(back), _boundary_set(mine)
    if got.keys() != want.keys():
        diffs.append(f"boundaries {sorted(got)} != case {sorted(want)}")
    else:
        diffs += [
            f"boundary {k}: {got[k]} != case {want[k]}"
            for k in want
            if not _same(got[k], want[k])
        ]
    if not _same(_port_list(back), _port_list(mine)):
        diffs.append(f"ports {_port_list(back)} != case {_port_list(mine)}")
    diffs += _problem_diffs(case, back["problem"])
    return diffs


def _problem_diffs(case: Case, back: Mapping[str, Any]) -> list[str]:
    problem = case.problem
    diffs = []
    if back["kind"] != problem.kind:
        return [f"problem kind {back['kind']} != case {problem.kind}"]
    if problem.kind == "driven":
        names = [p.name for p in case.ports]
        excite = [names[int(p[1:]) - 1] for p in back["excite"]]
        if excite != list(problem.excite):
            diffs.append(f"excitations {excite} != case {list(problem.excite)}")
        freqs = back["frequencies_hz"]
        if not isinstance(freqs, list) or not _same(freqs, list(problem.frequencies())):
            diffs.append(f"frequencies {freqs} != case {list(problem.frequencies())}")
    else:
        if not _same(back["target_hz"], problem.target_hz):
            diffs.append(f"target {back['target_hz']} != case {problem.target_hz}")
        if back["n_modes"] != problem.n_modes:
            diffs.append(f"modes {back['n_modes']} != case {problem.n_modes}")
    wave = back.get("wave_vector")
    want = None if problem.wave_vector is None else list(problem.wave_vector)
    if (wave is None) != (want is None) or (wave is not None and not _same(wave, want)):
        diffs.append(f"wave vector {wave} != case {want}")
    return diffs


def output_shape(case: Case, quantities: Mapping[str, Any]) -> list[str]:
    """Differences between the quantities' shape and the case's (``[]``)."""
    if case.problem.kind == "eigen":
        want = case.quantities.modes or 0
        got = len(quantities.get("f_hz", []))
        return [] if got >= want else [f"{got} modes < case {want}"]
    diffs = []
    freqs = quantities.get("frequencies_hz", [])
    if not _same(freqs, list(case.problem.frequencies())):
        diffs.append(f"frequencies {freqs} != case {list(case.problem.frequencies())}")
    missing = [
        f"{t},{f}"
        for t, f in case.quantities.s
        if f"{t},{f}" not in quantities.get("s", {})
    ]
    if missing:
        diffs.append(f"S entries missing: {', '.join(missing)}")
    return diffs


# --- the adapter -----------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _memory(outcome: Outcome, meta: Mapping[str, Any]) -> None:
    """Peak memory: summed over ranks, the largest node and the largest rank."""
    node = meta.get("PeakNodeMemoryMegabytes")
    if node is not None:
        outcome.self_peak_mem_gib = node["Total"] / MIB_PER_GIB
        outcome.self_peak_largest_node_gib = node["Max"] / MIB_PER_GIB
    rank = meta.get("PeakMemoryMegabytes")
    if rank is not None:
        outcome.self_peak_rank_max_gib = rank["Max"] / MIB_PER_GIB


@dataclass(frozen=True)
class PalaceAdapter:
    """The Palace solver adapter (``palace:mumps``, ``palace:pmg``)."""

    code: str = CODE
    kinds: Mapping[str, Kind] = field(
        default_factory=lambda: {
            "mumps": Kind(
                "direct",
                {"Type": "MUMPS", "MGMaxLevels": 1, "ComplexCoarseSolve": True},
            ),
            "pmg": Kind("iterative", {"Type": PMG_TYPE}),
        }
    )
    direct_kind: str = "mumps"
    pins: tuple[str, ...] = (
        "Refinement 0",
        "Absorbing.Order 1",
        "Linear.Type pinned per kind",
        "MUMPS kind: MGMaxLevels 1, ComplexCoarseSolve, Tol 1e-10",
        "iterative Tol = the run tolerance, MaxIts 2000",
        "EstimatorTol 1e-2, EstimatorMG",
        "eigen Tol 1e-10, inner Linear.Tol 1e-12",
        "no field output",
    )
    timing_source: str = "palace.json"
    memory_metric: str = "PeakNodeMemoryMegabytes.Total"
    stack_keys: tuple[str, ...] = ("launcher", "entry")

    def supports(self, case: Case, kind: str, layout: Layout) -> str | None:
        """Refuses unknown kinds, exclusions, unmapped boundaries and sweeps."""
        del layout  # every RxT on one node runs
        if kind not in self.kinds:
            return f"palace has no kind {kind!r} (known: {', '.join(self.kinds)})"
        if case.excluded:
            return "case has excluded volumes: DOF equality impossible with Palace"
        unknown = sorted({b.kind for b in case.boundaries} - set(BOUNDARY_KINDS))
        if unknown:
            return f"boundary kind(s) {', '.join(unknown)} not rendered for Palace"
        orders = {
            b.settings.get("order", 1) for b in case.boundaries if b.kind == "absorbing"
        }
        if orders - {1}:
            return "absorbing order 2 differs between Palace versions; only order 1"
        if case.problem.kind == "driven" and case.problem.sweep != "uniform":
            return "adaptive sweeps are not rendered for Palace"
        return None

    def default_layout(self, case: Case, kind: str, budget: int) -> Layout:
        """Palace's default: one rank per core (c x 1)."""
        del case, kind
        return Layout(budget, 1)

    def version(self, stack: Resolved) -> Launch:
        """``<entry> --version``, run bare once per job."""
        program = (entry(stack), stack_dialect(stack).launch.version_flag)
        command = Command(
            argv=program,
            program=program,
            phase="version",
            stdout=VERSION_STDOUT,
            launch="taskset",
        )
        return Launch(commands=(command,))

    def prepare(self, run: Run) -> Launch:
        """Write ``config.json`` and return the srun line around the binary."""
        reason = self.supports(run.case, run.kind, run.layout)
        if reason is not None:
            raise RunRefusedError(reason)
        dialect = stack_dialect(run.stack)
        config = render(
            run.case,
            run.kind,
            dialect,
            mesh=run.mesh,
            output=run.workdir / OUTPUT,
            tolerance=run.tolerance,
        )
        run.workdir.mkdir(parents=True, exist_ok=True)
        path = run.workdir / CONFIG
        path.write_text(json.dumps(config, indent=2) + "\n")
        section = run.stack.description.codes[CODE]
        program = (entry(run.stack), str(path))
        argv, launch = launched(
            program, run.layout, section.mpi or "pmi2", section.launch_r1
        )
        command = Command(
            argv=argv,
            program=program,
            stdout=STDOUT,
            launch=launch,
            watch=dof_watch(run.case.order),
        )
        return Launch(commands=(command,))

    def collect(self, run: Run, measured: Measured) -> Outcome:
        """Parse ``palace.json``, the CSVs, stdout and ``config_resolved.json``."""
        dialect = stack_dialect(run.stack)
        provenance: dict[str, Any] = {
            "code": CODE,
            "dialect": dialect.name,
            "stack_version": run.stack.versions.get("palace"),
        }
        if measured.version is not None:
            refused = version_refusal(run.stack, measured.version)
            if refused is not None:
                return Outcome(status="refused", reason=refused, provenance=provenance)
            provenance.update(parse_version(measured.version))
        try:
            return self._collect(run, measured, dialect, provenance)
        except PalaceParseError as error:
            return Outcome(
                status="failed",
                reason=f"parse: {error}",
                failure_class="harness",
                provenance=provenance,
            )

    def _collect(
        self,
        run: Run,
        measured: Measured,
        dialect: Dialect,
        provenance: dict[str, Any],
    ) -> Outcome:
        out = run.workdir / OUTPUT
        meta = _read_json(out / "palace.json") or {}
        problem = meta.get("Problem", {})
        stdout = run.workdir / STDOUT
        text = stdout.read_text(errors="replace") if stdout.exists() else ""
        outcome = Outcome(
            status="ok",
            ndof=problem.get("DegreesOfFreedom") or stdout_ndof(text, run.case.order),
            ranks_seen=problem.get("MPISize"),
            threads_seen=problem.get("OpenMPThreads"),
            provenance={**provenance, "git_tag": meta.get("GitTag")},
            files=[
                CONFIG,
                STDOUT,
                *(
                    f"{OUTPUT}/{name}"
                    for name in (
                        "palace.json",
                        "config_resolved.json",
                        "port-S.csv",
                        "eig.csv",
                    )
                    if (out / name).exists()
                ),
            ],
        )
        outcome.phases = dict.fromkeys(PHASES)
        outcome.rank0 = dict.fromkeys(PHASES)
        elapsed = meta.get("ElapsedTime")
        if elapsed is not None:
            durations = elapsed["Durations"]
            outcome.rank0 = buckets(durations, dialect)
            outcome.work_span_s = durations["Total"]
            outcome.native = {"durations": durations, "counts": elapsed["Counts"]}
            table = time_table(text, dialect) if (outcome.ranks_seen or 1) > 1 else {}
            if table:
                outcome.phases = max_of_rows(table, dialect)
                outcome.rank_reduction = "max-of-rows"
                outcome.native["table"] = {k: list(v) for k, v in table.items()}
                outcome.work_span_s = table["Total"][1]
            else:
                outcome.phases = dict(outcome.rank0)
                outcome.rank_reduction = "rank0"
            outcome.setups = elapsed["Counts"].get("Setup")
        solver = meta.get("LinearSolver", {})
        outcome.solves = solver.get("TotalSolves")
        _memory(outcome, meta)
        resolved = _read_json(out / "config_resolved.json")
        if resolved is not None:
            outcome.native["config_resolved"] = resolved
            diffs = inputs_match(run.case, resolved)
            outcome.checks["inputs_match"] = "fail" if diffs else "pass"
            if diffs:
                outcome.native["inputs_match"] = diffs
        converged, iterations = convergence(text)
        outcome.iterations = iterations
        if converged is not None:
            outcome.checks["converged"] = converged
        outcome.quantities = self._quantities(run.case, out)
        outcome.structural = self._structural(run, outcome)
        shape = (
            output_shape(run.case, outcome.quantities)
            if outcome.quantities
            else ["no quantities"]
        )
        complete = elapsed is not None and not shape
        outcome.checks["complete"] = "pass" if complete else "fail"
        if shape:
            outcome.native["output_shape"] = shape
        self._status(run, measured, outcome, elapsed is not None)
        return outcome

    @staticmethod
    def _quantities(case: Case, out: Path) -> dict[str, Any]:
        if case.problem.kind == "eigen":
            path = out / "eig.csv"
            return eig(path) if path.exists() else {}
        path = out / "port-S.csv"
        if not path.exists():
            return {}
        names = [p.name for p in case.ports]
        parsed = port_s(path, names)
        wanted = {f"{t},{f}" for t, f in case.quantities.s}
        parsed["s"] = {k: v for k, v in parsed["s"].items() if k in wanted}
        return parsed

    def _structural(self, run: Run, outcome: Outcome) -> dict[str, Any]:
        structural: dict[str, Any] = {
            "setups": outcome.setups,
            "solves": outcome.solves,
            "estimator": "always on, lightened (EstimatorTol 1e-2, EstimatorMG)",
        }
        if run.case.problem.kind == "driven":
            structural["excitations"] = len(run.case.problem.excite)
        if run.kind == "pmg":
            structural["fine_level"] = "matrix-free (partial assembly)"
            structural["pmg_coarse"] = f"{PMG_TYPE} on the real-valued coarse matrix"
        else:
            structural["direct_arm"] = "GMRES around complex MUMPS (MGMaxLevels 1)"
        return structural

    @staticmethod
    def _status(run: Run, measured: Measured, outcome: Outcome, complete: bool) -> None:
        code = measured.results[0].returncode if measured.results else None
        version = outcome.provenance.get("version") or run.stack.versions.get("palace")
        if (
            run.pinned_ndof is not None
            and outcome.ndof is not None
            and outcome.ndof != run.pinned_ndof
        ):
            outcome.checks["dof"] = "fail"
            outcome.status, outcome.failure_class = "failed", "solver"
            pinned = run.pinned_ndof
            outcome.reason = (
                f"dof_mismatch: palace {version} {outcome.ndof} != pinned {pinned}"
            )
            return
        if run.pinned_ndof is not None and outcome.ndof is not None:
            outcome.checks["dof"] = "pass"
        if outcome.checks.get("converged") == "fail":
            outcome.status, outcome.failure_class = "failed", "solver"
            outcome.reason = "not_converged: a linear solve stopped above tolerance"
        elif not complete:
            outcome.status, outcome.failure_class = "failed", "solver"
            outcome.reason = f"incomplete: no ElapsedTime in palace.json (exit {code})"
        elif code not in (0, None):
            outcome.status, outcome.failure_class = "failed", "solver"
            outcome.reason = f"exit {code}"


ADAPTER = PalaceAdapter()
