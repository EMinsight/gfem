"""Suite cases: the case file, its mesh pin and the Palace-config importer.

A **case file** (``benchmarks/cases/<name>/case.toml``, ``format = 1``) is the
solver-neutral definition of a suite case, keyed by region names: the pinned mesh
(``[mesh]``: sha256 of the uncompressed ``.msh``, ``bytes``, ``length_unit``, ``file``
when committed beside the case, ``[mesh.source]`` provenance), the region table
(``[regions]`` ``name = [dim, tag]`` from the mesh's ``$PhysicalNames``), materials,
boundaries by ``kind``, ports, exclusions, the element ``order``, the ``[problem]``
and the ``[quantities]`` compared. Lengths are in the mesh's unit, everything else
SI. Tolerance, solver, layout and threads belong to a run, never to a case.

``case pin`` hashes a mesh, writes ``[mesh]`` and ``[regions]`` and puts the bytes
where the size rule says: ``mesh.msh.xz`` beside the case when its xz file is at most
1 MiB, else ``<sha256>.msh.xz`` in the first mesh store. ``case import`` turns a gsim
Palace ``config.json`` into a case file once, at authoring time, and pins its mesh
with the same code. Standard library only (ADR 0400).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import math
import os
import re
import shutil
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any

FORMAT = 1
CASE_FILE = "case.toml"
COMMITTED_MESH = "mesh.msh.xz"
MAX_COMMITTED_BYTES = 1 << 20
"""Largest xz file committed beside a case (1 MiB); larger meshes are external."""
XZ_PRESET = 9
UNITS = {"um": 1e-6, "mm": 1e-3, "m": 1.0}
SIZES = ("small", "large")
SWEEPS = ("uniform", "adaptive")
PORT_KINDS = ("lumped", "wave")
_CHUNK = 1 << 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIRECTION = re.compile(r"^[+-][xyz]$")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_FLOOR_SLACK = 1e-9
#: Keys of a boundary that name regions (checked against ``[regions]``).
_REGION_KEYS = ("region", "donor", "receiver")


class CaseError(ValueError):
    """A case file, mesh or pin the suite refuses."""


# --- mesh bytes ---------------------------------------------------------------------


def open_mesh(path: Path) -> IO[bytes]:
    """Open a mesh for reading its uncompressed bytes (``.xz`` decompressed)."""
    return lzma.open(path) if path.suffix == ".xz" else path.open("rb")


def _blocks(stream: IO[bytes]) -> Iterator[bytes]:
    yield from iter(lambda: stream.read(_CHUNK), b"")


def mesh_digest(path: Path) -> tuple[str, int]:
    """sha256 and size of the uncompressed mesh bytes (what a solver reads)."""
    digest = hashlib.sha256()
    size = 0
    with open_mesh(path) as stream:
        for block in _blocks(stream):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def physical_names(path: Path) -> dict[str, tuple[int, int]]:
    """``name -> (dim, tag)`` from a mesh's ``$PhysicalNames`` block.

    The block is ASCII in every MSH 2.2 and 4.1 file, binary ones included, so no
    gmsh or gfem import is needed. An ``.xz`` file is read decompressed.
    """
    names: dict[str, tuple[int, int]] = {}
    with open_mesh(path) as stream:
        for raw in stream:
            if raw.strip() == b"$PhysicalNames":
                for _ in range(int(next(stream))):
                    dim, tag, name = next(stream).decode().split(maxsplit=2)
                    names[name.strip().strip('"')] = (int(dim), int(tag))
                return names
            if raw.startswith((b"$Nodes", b"$Entities", b"$Elements")):
                break
    msg = f"{path}: no $PhysicalNames block before the mesh data"
    raise CaseError(msg)


# --- the case file ------------------------------------------------------------------


@dataclass(frozen=True)
class MeshSource:
    """Where a pinned mesh came from: provenance, never executed (ADR 0405)."""

    recipe: str
    commit: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class MeshPin:
    """A case's ``[mesh]``: the pin is the sha256 of the uncompressed ``.msh``."""

    sha256: str
    bytes: int
    length_unit: str
    file: str | None = None
    """Relative to the case directory when committed; ``None`` = external."""
    source: MeshSource | None = None


@dataclass(frozen=True)
class Material:
    """A volume region's material; ``eps_r``/``tan_delta`` scalar or principal."""

    region: str
    eps_r: float | tuple[float, ...]
    mu_r: float | tuple[float, ...] | None = None
    tan_delta: float | tuple[float, ...] | None = None
    conductivity: float | None = None
    axes: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class Boundary:
    """One boundary condition; ``kind`` is open, adapters refuse unknown kinds.

    ``pec``, ``conductor`` (``conductivity``, ``thickness``), ``absorbing``
    (``order``), ``impedance`` and ``periodic`` (``donor``, ``receiver``,
    ``translation``) are in use; every key but ``kind`` and ``region`` is in
    ``settings``.
    """

    kind: str
    region: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortElement:
    """A face of a lumped port with its signed axis (``+x``..``-z``)."""

    region: str
    direction: str


@dataclass(frozen=True)
class Port:
    """A lumped port (``elements``) or a wave port (``region``, ``mode``...)."""

    kind: str
    name: str
    elements: tuple[PortElement, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    """``impedance_ohm`` etc. for lumped; ``region``, ``mode``, ``offset``,
    ``pec_regions`` for wave."""


@dataclass(frozen=True)
class Problem:
    """``driven`` (frequencies, sweep, excite) or ``eigen`` (target, modes)."""

    kind: str
    frequencies_hz: tuple[float, ...] | dict[str, float] = ()
    sweep: str = "uniform"
    excite: tuple[str, ...] = ()
    target_hz: float | None = None
    n_modes: int | None = None
    wave_vector: tuple[float, float, float] | None = None
    """Bloch wave vector in rad/m (SI), periodic cases only."""

    def frequencies(self) -> tuple[float, ...]:
        """The driven frequencies in Hz.

        A ``{start, stop, step}`` range expands to every ``start + k step`` up to
        ``stop`` (floor with slack, as Palace's grid).
        """
        if not isinstance(self.frequencies_hz, dict):
            return self.frequencies_hz
        start = self.frequencies_hz["start"]
        stop = self.frequencies_hz["stop"]
        step = self.frequencies_hz["step"]
        count = math.floor((stop - start) / step + _FLOOR_SLACK) + 1
        return tuple(start + k * step for k in range(count))


@dataclass(frozen=True)
class Quantities:
    """What is measured: S entries ``(to, from)`` or the lowest ``modes``."""

    s: tuple[tuple[str, str], ...] = ()
    modes: int | None = None


@dataclass(frozen=True)
class Case:
    """A loaded, validated case file."""

    name: str
    size: str
    family: str
    order: int
    mesh: MeshPin
    regions: dict[str, tuple[int, int]]
    problem: Problem
    quantities: Quantities
    materials: tuple[Material, ...] = ()
    boundaries: tuple[Boundary, ...] = ()
    ports: tuple[Port, ...] = ()
    excluded: tuple[tuple[str, ...], ...] = ()
    """Volumes removed from the domain, each as the surfaces enclosing it."""
    est_s: dict[str, float] = field(default_factory=dict)
    """Run-time hints for ``plan``'s estimate (keys set by the plan)."""
    directory: Path | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """The case as the TOML-shaped dict :func:`dumps` writes."""
        return _case_dict(self)


_TOP = (
    "format",
    "name",
    "size",
    "family",
    "order",
    "mesh",
    "regions",
    "problem",
    "quantities",
    "est_s",
    "materials",
    "boundaries",
    "ports",
    "excluded",
)
_MESH_KEYS = ("sha256", "bytes", "length_unit", "file", "source")
_SOURCE_KEYS = ("recipe", "commit", "note")
_MATERIAL_KEYS = ("region", "eps_r", "mu_r", "tan_delta", "conductivity", "axes")
_PROBLEM_KEYS = {
    "driven": ("kind", "frequencies_hz", "sweep", "excite", "wave_vector"),
    "eigen": ("kind", "target_hz", "n_modes", "wave_vector"),
}
_LUMPED_KEYS = (
    "kind",
    "name",
    "elements",
    "impedance_ohm",
    "inductance_h",
    "capacitance_f",
)
_WAVE_KEYS = ("kind", "name", "region", "mode", "offset", "pec_regions")


def _keys(table: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        msg = f"{where}: unknown key(s) {', '.join(unknown)}"
        raise CaseError(msg)


def _get(table: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in table:
        msg = f"{where}: missing key {key}"
        raise CaseError(msg)
    return table[key]


def _table(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{where} must be a table"
        raise CaseError(msg)
    return value


def _tables(value: object, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        msg = f"{where} must be an array of tables"
        raise CaseError(msg)
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        msg = f"{where} must be a string"
        raise CaseError(msg)
    return value


def _strs(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f"{where} must be a list of strings"
        raise CaseError(msg)
    return tuple(value)


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{where} must be a number"
        raise CaseError(msg)
    return float(value)


def _int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{where} must be a positive integer"
        raise CaseError(msg)
    return value


def _scalar_or_vector(value: object, where: str) -> float | tuple[float, ...]:
    if isinstance(value, list):
        return tuple(_float(v, where) for v in value)
    return _float(value, where)


def _choice(value: object, choices: Sequence[str], where: str) -> str:
    text = _str(value, where)
    if text not in choices:
        msg = f"{where} {text!r} not in {', '.join(choices)}"
        raise CaseError(msg)
    return text


def load_case(path: Path) -> Case:
    """Load and validate a case file (a ``case.toml`` or its directory).

    The ``name`` must equal the directory's name; every region a material,
    boundary, port or exclusion names must be in ``[regions]``.
    """
    file = path / CASE_FILE if path.is_dir() else path
    try:
        raw = tomllib.loads(file.read_text())
    except FileNotFoundError as error:
        msg = f"{file}: no such file"
        raise CaseError(msg) from error
    except tomllib.TOMLDecodeError as error:
        msg = f"{file}: {error}"
        raise CaseError(msg) from error
    loaded = case_from_dict(raw, where=str(file))
    if loaded.name != file.parent.name:
        msg = f"{file}: name {loaded.name!r} differs from its directory"
        msg += f" {file.parent.name!r}"
        raise CaseError(msg)
    return replace(loaded, directory=file.parent)


def case_from_dict(raw: Mapping[str, Any], *, where: str = "case") -> Case:
    """Validate a parsed case file into a :class:`Case`."""
    _keys(raw, _TOP, where)
    if raw.get("format") != FORMAT:
        msg = f"{where}: format {raw.get('format')!r} unknown (this harness reads 1)"
        raise CaseError(msg)
    regions = _regions(_table(_get(raw, "regions", where), f"{where}: regions"), where)

    def region(value: object, label: str, dim: int | None = None) -> str:
        name = _str(value, label)
        if name not in regions:
            msg = f"{where}: {label} names {name!r}, not in [regions]"
            raise CaseError(msg)
        if dim is not None and regions[name][0] != dim:
            msg = f"{where}: {label} {name!r} is not of dimension {dim}"
            raise CaseError(msg)
        return name

    materials = tuple(
        _material(m, f"{where} materials[{i}]", region)
        for i, m in enumerate(_tables(raw.get("materials", []), "materials"))
    )
    boundaries = tuple(
        _boundary(b, f"{where} boundaries[{i}]", region)
        for i, b in enumerate(_tables(raw.get("boundaries", []), "boundaries"))
    )
    ports = tuple(
        _port(p, f"{where} ports[{i}]", region)
        for i, p in enumerate(_tables(raw.get("ports", []), "ports"))
    )
    names = [p.name for p in ports]
    if len(set(names)) != len(names):
        msg = f"{where}: port names repeat: {names}"
        raise CaseError(msg)
    excluded = []
    for i, e in enumerate(_tables(raw.get("excluded", []), "excluded")):
        label = f"excluded[{i}].enclosed_by"
        _keys(e, ("enclosed_by",), f"{where} excluded[{i}]")
        surfaces = _strs(_get(e, "enclosed_by", label), label)
        excluded.append(tuple(region(s, label, 2) for s in surfaces))
    problem = _problem(_table(_get(raw, "problem", where), "problem"), where, names)
    quantities = _quantities(
        _table(_get(raw, "quantities", where), "quantities"), where, names
    )
    est = _table(raw.get("est_s", {}), "est_s")
    return Case(
        name=_str(_get(raw, "name", where), "name"),
        size=_choice(_get(raw, "size", where), SIZES, f"{where}: size"),
        family=_str(_get(raw, "family", where), "family"),
        order=_int(_get(raw, "order", where), "order"),
        mesh=_mesh(_table(_get(raw, "mesh", where), "mesh"), where),
        regions=regions,
        problem=problem,
        quantities=quantities,
        materials=materials,
        boundaries=boundaries,
        ports=ports,
        excluded=tuple(excluded),
        est_s={k: _float(v, f"est_s.{k}") for k, v in est.items()},
    )


def _regions(table: Mapping[str, Any], where: str) -> dict[str, tuple[int, int]]:
    regions = {}
    for name, value in table.items():
        ok = (
            isinstance(value, list)
            and len(value) == 2  # (dim, tag)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            and value[0] in (0, 1, 2, 3)
        )
        if not ok:
            msg = f"{where}: regions.{name} must be [dim, tag]"
            raise CaseError(msg)
        regions[name] = (value[0], value[1])
    return regions


def _mesh(table: Mapping[str, Any], where: str) -> MeshPin:
    _keys(table, _MESH_KEYS, f"{where} [mesh]")
    sha = _str(_get(table, "sha256", f"{where} [mesh]"), "mesh.sha256")
    if not _SHA256.match(sha):
        msg = f"{where}: mesh.sha256 {sha!r} is not a lowercase sha256"
        raise CaseError(msg)
    source = None
    if "source" in table:
        src = _table(table["source"], "mesh.source")
        _keys(src, _SOURCE_KEYS, f"{where} [mesh.source]")
        source = MeshSource(
            recipe=_str(_get(src, "recipe", "mesh.source"), "mesh.source.recipe"),
            commit=None if "commit" not in src else _str(src["commit"], "commit"),
            note=None if "note" not in src else _str(src["note"], "note"),
        )
    return MeshPin(
        sha256=sha,
        bytes=_int(_get(table, "bytes", f"{where} [mesh]"), "mesh.bytes"),
        length_unit=_choice(
            _get(table, "length_unit", f"{where} [mesh]"),
            tuple(UNITS),
            f"{where}: mesh.length_unit",
        ),
        file=None if "file" not in table else _str(table["file"], "mesh.file"),
        source=source,
    )


def _material(table: Mapping[str, Any], where: str, region: Any) -> Material:
    _keys(table, _MATERIAL_KEYS, where)
    axes = table.get("axes")
    if axes is not None:
        if not isinstance(axes, list) or not all(isinstance(a, list) for a in axes):
            msg = f"{where}: axes must be a list of vectors"
            raise CaseError(msg)
        axes = tuple(tuple(_float(x, f"{where} axes") for x in a) for a in axes)
    optional = {
        key: _scalar_or_vector(table[key], f"{where} {key}")
        for key in ("mu_r", "tan_delta")
        if key in table
    }
    return Material(
        region=region(_get(table, "region", where), f"{where} region", 3),
        eps_r=_scalar_or_vector(_get(table, "eps_r", where), f"{where} eps_r"),
        conductivity=None
        if "conductivity" not in table
        else _float(table["conductivity"], f"{where} conductivity"),
        axes=axes,
        **optional,
    )


def _boundary(table: Mapping[str, Any], where: str, region: Any) -> Boundary:
    kind = _str(_get(table, "kind", where), f"{where} kind")
    for key in _REGION_KEYS:
        if key in table:
            region(table[key], f"{where} {key}", 2)
    return Boundary(
        kind=kind,
        region=table.get("region"),
        settings={k: v for k, v in table.items() if k not in ("kind", "region")},
    )


def _port(table: Mapping[str, Any], where: str, region: Any) -> Port:
    kind = _choice(_get(table, "kind", where), PORT_KINDS, f"{where} kind")
    name = _str(_get(table, "name", where), f"{where} name")
    if kind == "wave":
        _keys(table, _WAVE_KEYS, where)
        region(_get(table, "region", where), f"{where} region", 2)
        for pec in _strs(table.get("pec_regions", []), f"{where} pec_regions"):
            region(pec, f"{where} pec_regions", 2)
        settings = {k: v for k, v in table.items() if k not in ("kind", "name")}
        return Port(kind=kind, name=name, settings=settings)
    _keys(table, _LUMPED_KEYS, where)
    elements = []
    for i, e in enumerate(_tables(_get(table, "elements", where), "elements")):
        label = f"{where} elements[{i}]"
        _keys(e, ("region", "direction"), label)
        direction = _str(_get(e, "direction", label), f"{label} direction")
        if not _DIRECTION.match(direction):
            msg = f"{label}: direction {direction!r} is not a signed axis like +z"
            raise CaseError(msg)
        elements.append(
            PortElement(region(_get(e, "region", label), label, 2), direction)
        )
    if not elements:
        msg = f"{where}: a lumped port needs elements"
        raise CaseError(msg)
    settings = {
        k: _float(v, f"{where} {k}")
        for k, v in table.items()
        if k not in ("kind", "name", "elements")
    }
    return Port(kind=kind, name=name, elements=tuple(elements), settings=settings)


def _problem(table: Mapping[str, Any], where: str, ports: list[str]) -> Problem:
    kind = _choice(_get(table, "kind", where), tuple(_PROBLEM_KEYS), f"{where} kind")
    _keys(table, _PROBLEM_KEYS[kind], f"{where} [problem]")
    wave = table.get("wave_vector")
    wave_vector = None
    if wave is not None:
        if not isinstance(wave, list) or len(wave) != 3:
            msg = f"{where}: problem.wave_vector must be 3 numbers (rad/m)"
            raise CaseError(msg)
        x, y, z = (_float(v, "problem.wave_vector") for v in wave)
        wave_vector = (x, y, z)
    if kind == "eigen":
        return Problem(
            kind=kind,
            target_hz=_float(_get(table, "target_hz", where), "problem.target_hz"),
            n_modes=_int(_get(table, "n_modes", where), "problem.n_modes"),
            wave_vector=wave_vector,
        )
    freqs = _get(table, "frequencies_hz", where)
    if isinstance(freqs, dict):
        _keys(freqs, ("start", "stop", "step"), f"{where} problem.frequencies_hz")
        frequencies: tuple[float, ...] | dict[str, float] = {
            k: _float(_get(freqs, k, "problem.frequencies_hz"), k)
            for k in ("start", "stop", "step")
        }
    elif isinstance(freqs, list) and freqs:
        frequencies = tuple(_float(f, "problem.frequencies_hz") for f in freqs)
    else:
        msg = f"{where}: problem.frequencies_hz must be a list or {{start, stop, step}}"
        raise CaseError(msg)
    excite = _strs(_get(table, "excite", where), "problem.excite")
    unknown = [p for p in excite if p not in ports]
    if unknown or not excite:
        msg = f"{where}: problem.excite names {unknown or 'no port'}"
        msg += f" (ports: {', '.join(ports)})"
        raise CaseError(msg)
    return Problem(
        kind=kind,
        frequencies_hz=frequencies,
        sweep=_choice(table.get("sweep", "uniform"), SWEEPS, f"{where} sweep"),
        excite=excite,
        wave_vector=wave_vector,
    )


def _quantities(table: Mapping[str, Any], where: str, ports: list[str]) -> Quantities:
    _keys(table, ("s", "modes"), f"{where} [quantities]")
    if "modes" in table:
        return Quantities(modes=_int(table["modes"], "quantities.modes"))
    pairs = []
    for pair in _get(table, "s", f"{where} [quantities]"):
        to, frm = _strs(pair, "quantities.s entry")
        for port in (to, frm):
            if port not in ports:
                msg = f"{where}: quantities.s names port {port!r}"
                msg += f" (ports: {', '.join(ports)})"
                raise CaseError(msg)
        pairs.append((to, frm))
    return Quantities(s=tuple(pairs))


def _case_dict(c: Case) -> dict[str, Any]:
    mesh: dict[str, Any] = {
        "sha256": c.mesh.sha256,
        "bytes": c.mesh.bytes,
        "length_unit": c.mesh.length_unit,
    }
    if c.mesh.file is not None:
        mesh["file"] = c.mesh.file
    if c.mesh.source is not None:
        mesh["source"] = _drop_none(c.mesh.source.__dict__)
    problem: dict[str, Any] = {"kind": c.problem.kind}
    if c.problem.kind == "driven":
        freqs = c.problem.frequencies_hz
        problem["frequencies_hz"] = (
            dict(freqs) if isinstance(freqs, dict) else list(freqs)
        )
        problem["sweep"] = c.problem.sweep
        problem["excite"] = list(c.problem.excite)
    else:
        problem["target_hz"] = c.problem.target_hz
        problem["n_modes"] = c.problem.n_modes
    if c.problem.wave_vector is not None:
        problem["wave_vector"] = list(c.problem.wave_vector)
    quantities: dict[str, Any] = (
        {"modes": c.quantities.modes}
        if c.quantities.modes is not None
        else {"s": [list(p) for p in c.quantities.s]}
    )
    out: dict[str, Any] = {
        "format": FORMAT,
        "name": c.name,
        "size": c.size,
        "family": c.family,
        "order": c.order,
        "mesh": mesh,
        "regions": {k: list(v) for k, v in c.regions.items()},
        "problem": problem,
        "quantities": quantities,
    }
    if c.est_s:
        out["est_s"] = dict(c.est_s)
    if c.materials:
        out["materials"] = [_drop_none(_lists(m.__dict__)) for m in c.materials]
    if c.boundaries:
        out["boundaries"] = [
            _drop_none({"kind": b.kind, "region": b.region, **b.settings})
            for b in c.boundaries
        ]
    if c.ports:
        out["ports"] = [_port_dict(p) for p in c.ports]
    if c.excluded:
        out["excluded"] = [{"enclosed_by": list(e)} for e in c.excluded]
    return out


def _port_dict(p: Port) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": p.kind, "name": p.name}
    if p.kind == "lumped":
        out.update(p.settings)
        out["elements"] = [e.__dict__.copy() for e in p.elements]
    else:
        out.update(p.settings)
    return out


def _lists(table: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: object) -> object:
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        return value

    return {k: convert(v) for k, v in table.items()}


def _drop_none(table: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in table.items() if v is not None}


# --- TOML writer --------------------------------------------------------------------


def _key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)  # also TOML's inf, -inf and nan
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    msg = f"cannot write {type(value).__name__} to TOML"
    raise CaseError(msg)


def _is_tables(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, dict) for v in value)
    )


def _emit(table: Mapping[str, Any], prefix: str, lines: list[str]) -> None:
    lines.extend(
        f"{_key(k)} = {_value(v)}"
        for k, v in table.items()
        if not isinstance(v, dict) and not _is_tables(v)
    )
    for k, v in table.items():
        if isinstance(v, dict):
            lines += ["", f"[{prefix}{_key(k)}]"]
            _emit(v, f"{prefix}{_key(k)}.", lines)
    for k, v in table.items():
        if _is_tables(v):
            for item in v:
                lines += ["", f"[[{prefix}{_key(k)}]]"]
                # tables nested inside an array of tables are written inline
                lines.extend(f"{_key(a)} = {_value(b)}" for a, b in item.items())


def dumps(raw: Mapping[str, Any]) -> str:
    """A case dict as TOML that ``tomllib`` reads back equal.

    Top-level keys come in case-file order, then tables, then arrays of tables
    (tables nested inside those are written inline).
    """
    order = {k: i for i, k in enumerate(_TOP)}
    ordered = dict(sorted(raw.items(), key=lambda kv: order.get(kv[0], len(order))))
    lines: list[str] = []
    _emit(ordered, "", lines)
    return "\n".join(lines) + "\n"


def write_case(directory: Path, raw: Mapping[str, Any]) -> Path:
    """Write ``raw`` as ``directory/case.toml`` (atomically); return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CASE_FILE
    tmp = path.with_name(f".{CASE_FILE}.{os.getpid()}.part")
    tmp.write_text(dumps(raw))
    tmp.replace(path)
    return path


# --- pin ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pinned:
    """What :func:`pin` wrote: the pin and where the mesh bytes went."""

    sha256: str
    bytes: int
    committed: bool
    path: Path
    """``mesh.msh.xz`` in the case directory, or the store file."""


def _compress(source: Path, target: Path, limit: int | None) -> bool:
    """xz-compress ``source``'s mesh bytes into ``target``.

    Returns ``False``, writing nothing, as soon as the output passes ``limit``.
    """
    tmp = target.with_name(f".{target.name}.{os.getpid()}.part")
    compressor = lzma.LZMACompressor(preset=XZ_PRESET)
    written = 0
    complete = False
    try:
        with open_mesh(source) as src, tmp.open("wb") as dst:
            for block in itertools.chain(_blocks(src), [None]):
                out = (
                    compressor.flush() if block is None else compressor.compress(block)
                )
                written += len(out)
                if limit is not None and written > limit:
                    break
                dst.write(out)
            else:
                complete = True
    finally:
        if not complete:
            tmp.unlink(missing_ok=True)
    if complete:
        tmp.replace(target)
    return complete


def _store_external(source: Path, sha: str, stores: Sequence[Path]) -> Path:
    for store in stores:
        existing = store / f"{sha}.msh.xz"
        if existing.exists():
            got, _ = mesh_digest(existing)
            if got != sha:
                msg = f"mesh store file {existing} gives {got[:12]}, not {sha[:12]}"
                raise CaseError(msg)
            return existing
    if not stores:
        msg = (
            f"mesh {sha[:12]} compresses to more than {MAX_COMMITTED_BYTES} bytes: "
            "external, but no mesh store is given (pass --site)"
        )
        raise CaseError(msg)
    store = stores[0]
    store.mkdir(parents=True, exist_ok=True)
    target = store / f"{sha}.msh.xz"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.part")
    if source.suffix == ".xz":
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    else:
        _compress(source, target, None)
    target.chmod(0o444)
    return target


def pin(case_dir: Path, mesh: Path, unit: str, stores: Sequence[Path] = ()) -> Pinned:
    """Pin ``mesh`` (``.msh`` or ``.msh.xz``) as the mesh of the case in ``case_dir``.

    Writes ``[mesh]`` and ``[regions]`` and places the bytes by the size rule.

    Creates ``case.toml`` (``format``, ``name``) if missing and keeps an existing
    ``[mesh.source]``. Refuses a case that already has a pin: changing a pin is a
    deliberate edit of the case file, never a command (ADR 0405).
    """
    if unit not in UNITS:
        msg = f"unit {unit!r} not in {', '.join(UNITS)}"
        raise CaseError(msg)
    file = case_dir / CASE_FILE
    raw: dict[str, Any] = (
        tomllib.loads(file.read_text())
        if file.exists()
        else {"format": FORMAT, "name": case_dir.name}
    )
    old = raw.get("mesh", {})
    if "sha256" in old:
        msg = (
            f"{file}: already pinned to {old['sha256'][:12]}; a pin is changed by "
            "editing the case file deliberately, never overwritten"
        )
        raise CaseError(msg)
    if not mesh.is_file():
        msg = f"{mesh}: no such mesh"
        raise CaseError(msg)
    regions = physical_names(mesh)
    sha, size = mesh_digest(mesh)
    case_dir.mkdir(parents=True, exist_ok=True)
    committed_path = case_dir / COMMITTED_MESH
    committed = _compress(mesh, committed_path, MAX_COMMITTED_BYTES)
    path = committed_path if committed else _store_external(mesh, sha, stores)
    table: dict[str, Any] = {"sha256": sha, "bytes": size, "length_unit": unit}
    if committed:
        table["file"] = COMMITTED_MESH
    if "source" in old:
        table["source"] = old["source"]
    raw["mesh"] = table
    raw["regions"] = {k: list(v) for k, v in sorted(regions.items(), key=_by_tag)}
    write_case(case_dir, raw)
    return Pinned(sha256=sha, bytes=size, committed=committed, path=path)


def _by_tag(item: tuple[str, tuple[int, int]]) -> tuple[int, int]:
    dim, tag = item[1]
    return (-dim, tag)


# --- Palace config importer ---------------------------------------------------------

_PALACE_BOUNDARIES = (
    "PEC",
    "Conductivity",
    "Absorbing",
    "Impedance",
    "LumpedPort",
    "WavePort",
    "WavePortPEC",
    "Periodic",
    "Postprocessing",
)
#: Provenance files gsim writes beside ``config.json``, copied into ``source/``.
PROVENANCE = ("config.json", "build.json", "metadata.json", "port_information.json")


def _axis(text: str) -> str:
    text = text.strip().lower()
    return text if text[0] in "+-" else "+" + text


def _unit(length: float) -> str:
    for name, value in UNITS.items():
        if math.isclose(length, value, rel_tol=1e-12):
            return name
    msg = f"Model.L0 {length} is not one of {', '.join(UNITS)}"
    raise CaseError(msg)


def _attributes(entry: Mapping[str, Any]) -> list[int]:
    return list(entry.get("Attributes", []))


class _Tags:
    """Region names of a config's integer attributes."""

    def __init__(self, regions: Mapping[str, tuple[int, int]]) -> None:
        self.by_tag = {v: k for k, v in regions.items()}

    def __call__(self, dim: int, tag: int) -> str:
        if (dim, tag) not in self.by_tag:
            msg = f"config attribute {tag} (dim {dim}) is not a physical group"
            raise CaseError(msg)
        return self.by_tag[dim, tag]


def import_palace(
    config: Mapping[str, Any],
    regions: Mapping[str, tuple[int, int]],
    *,
    name: str,
    voided: bool,
    frequencies_hz: Sequence[float] | None = None,
) -> dict[str, Any]:
    """A case dict (no ``[mesh]`` pin yet) from a gsim/Palace ``config.json``.

    Merges the rules of the repository's config-to-model builders: single-face
    and ``Elements`` lumped ports, unsigned axes read as positive, materials'
    ``Permeability``, per-axis ``LossTan``, ``MaterialAxes`` and ``Conductivity``,
    PEC, Conductivity, Absorbing, Impedance, WavePort with WavePortPEC, Periodic.
    Unless ``voided``, each ``*_xy`` conductor shell and its ``*_z`` partner
    enclose an excluded volume, closed by the absorbing faces when wave ports
    exist. Lengths stay in ``Model.L0`` units (the mesh's). ``frequencies_hz``
    replaces the config's driven frequencies (required when it has none).
    """
    region = _Tags(regions)
    bnd = config.get("Boundaries", {})
    unknown = sorted(set(bnd) - set(_PALACE_BOUNDARIES))
    if unknown:
        msg = f"config Boundaries {', '.join(unknown)}: not imported; write by hand"
        raise CaseError(msg)
    boundaries = _import_boundaries(bnd, region)
    ports, excite = _import_ports(bnd, region)
    excluded = []
    if not voided:
        ends = (
            [b["region"] for b in boundaries if b["kind"] == "absorbing"]
            if bnd.get("WavePort")
            else []
        )
        excluded = [
            {"enclosed_by": [b["region"], b["region"].replace("_xy", "_z"), *ends]}
            for b in boundaries
            if b["kind"] == "conductor" and b["region"].endswith("_xy")
        ]
    solver = config.get("Solver", {})
    problem, quantities = _import_problem(
        config, ports, excite, frequencies_hz=frequencies_hz
    )
    raw: dict[str, Any] = {
        "format": FORMAT,
        "name": name,
        "size": "small",
        "family": name,
        "order": solver.get("Order", 1),
        "problem": problem,
        "quantities": quantities,
        "materials": _import_materials(config, region),
        "boundaries": boundaries,
        "ports": ports,
        "excluded": excluded,
    }
    return {k: v for k, v in raw.items() if v != []}


def _import_materials(config: Mapping[str, Any], region: _Tags) -> list[dict]:
    materials = []
    for m in config.get("Domains", {}).get("Materials", []):
        for tag in _attributes(m):
            entry: dict[str, Any] = {
                "region": region(3, tag),
                "eps_r": m["Permittivity"],
            }
            for key, out in (
                ("Permeability", "mu_r"),
                ("LossTan", "tan_delta"),
                ("Conductivity", "conductivity"),
                ("MaterialAxes", "axes"),
            ):
                if key in m:
                    entry[out] = m[key]
            materials.append(entry)
    return materials


def _import_boundaries(bnd: Mapping[str, Any], region: _Tags) -> list[dict]:
    boundaries: list[dict[str, Any]] = [
        {"kind": "pec", "region": region(2, t)} for t in _attributes(bnd.get("PEC", {}))
    ]
    boundaries += [
        {
            "kind": "conductor",
            "region": region(2, t),
            "conductivity": c["Conductivity"],
            "thickness": c["Thickness"],
        }
        for c in bnd.get("Conductivity", [])
        for t in _attributes(c)
    ]
    absorbing = bnd.get("Absorbing", {})
    boundaries += [
        {"kind": "absorbing", "region": region(2, t), "order": absorbing["Order"]}
        for t in _attributes(absorbing)
    ]
    for imp in bnd.get("Impedance", []):
        values = {
            out: imp[key]
            for key, out in (("Rs", "rs_ohm"), ("Ls", "ls_h"), ("Cs", "cs_f"))
            if key in imp
        }
        boundaries += [
            {"kind": "impedance", "region": region(2, t), **values}
            for t in _attributes(imp)
        ]
    for pair in bnd.get("Periodic", {}).get("BoundaryPairs", []):
        donors, receivers = pair["DonorAttributes"], pair["ReceiverAttributes"]
        boundaries += [
            {
                "kind": "periodic",
                "donor": region(2, d),
                "receiver": region(2, r),
                "translation": pair["Translation"],
            }
            for d, r in zip(donors, receivers, strict=True)
        ]
    return boundaries


def _import_ports(
    bnd: Mapping[str, Any], region: _Tags
) -> tuple[list[dict[str, Any]], list[str]]:
    ports: list[dict[str, Any]] = []
    excite: list[str] = []
    for p in sorted(bnd.get("LumpedPort", []), key=lambda p: p["Index"]):
        port: dict[str, Any] = {"kind": "lumped", "name": f"p{p['Index']}"}
        for key, out in (
            ("R", "impedance_ohm"),
            ("L", "inductance_h"),
            ("C", "capacitance_f"),
        ):
            if key in p:
                port[out] = float(p[key])
        port["elements"] = [
            {
                "region": region(2, e["Attributes"][0]),
                "direction": _axis(e["Direction"]),
            }
            for e in p.get("Elements", [p])
        ]
        ports.append(port)
        if p.get("Excitation"):
            excite.append(port["name"])
    pec_edges = [region(2, t) for t in _attributes(bnd.get("WavePortPEC", {}))]
    for p in sorted(bnd.get("WavePort", []), key=lambda p: p["Index"]):
        port = {
            "kind": "wave",
            "name": f"p{p['Index']}",
            "region": region(2, p["Attributes"][0]),
            "mode": p.get("Mode", 1),
            "offset": p.get("Offset", 0.0),
        }
        if pec_edges:
            port["pec_regions"] = pec_edges
        ports.append(port)
        if p.get("Excitation"):
            excite.append(port["name"])
    return ports, excite


def _import_problem(
    config: Mapping[str, Any],
    ports: list[dict[str, Any]],
    excite: list[str],
    *,
    frequencies_hz: Sequence[float] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    solver = config.get("Solver", {})
    kind = config.get("Problem", {}).get("Type")
    l0 = float(config["Model"]["L0"])
    wave = config.get("Boundaries", {}).get("Periodic", {}).get("FloquetWaveVector")
    extra = {} if wave is None else {"wave_vector": [k / l0 for k in wave]}
    if kind == "Eigenmode":
        eigen = solver.get("Eigenmode", {})
        n = int(eigen.get("N", 1))
        problem = {"kind": "eigen", "target_hz": eigen["Target"] * 1e9, "n_modes": n}
        return {**problem, **extra}, {"modes": n}
    if kind != "Driven":
        msg = f"config Problem.Type {kind!r}: only Driven and Eigenmode import"
        raise CaseError(msg)
    driven = solver.get("Driven", {})
    frequencies: list[float] | dict[str, float]
    if frequencies_hz is not None:
        frequencies = [float(f) for f in frequencies_hz]
    else:
        frequencies = _config_frequencies(driven)
    names = [p["name"] for p in ports]
    excite = excite or names[:1]
    problem = {
        "kind": "driven",
        "frequencies_hz": frequencies,
        "sweep": "adaptive"
        if frequencies_hz is None and driven.get("AdaptiveTol", 0) > 0
        else "uniform",
        "excite": excite,
        **extra,
    }
    return problem, {"s": [[to, frm] for frm in excite for to in names]}


def _config_frequencies(driven: Mapping[str, Any]) -> dict[str, float]:
    samples = driven.get("Samples")
    if samples:
        if len(samples) != 1 or samples[0].get("Type", "Linear") != "Linear":
            msg = "config Driven.Samples: only one Linear range imports"
            raise CaseError(msg)
        sample = samples[0]
    elif "MinFreq" in driven:
        sample = driven
    else:
        msg = "config has no driven frequencies: pass them (--frequencies-ghz)"
        raise CaseError(msg)
    return {
        out: sample[key] * 1e9
        for key, out in (
            ("MinFreq", "start"),
            ("MaxFreq", "stop"),
            ("FreqStep", "step"),
        )
    }


def is_voided(config_path: Path) -> bool:
    """A gsim case is voided when its ``build.json`` lists excluded entities."""
    build = config_path.with_name("build.json")
    return build.exists() and "excluded_entities" in json.loads(build.read_text())


def import_case(
    config_path: Path,
    mesh: Path,
    case_dir: Path,
    stores: Sequence[Path] = (),
    frequencies_hz: Sequence[float] | None = None,
) -> Pinned:
    """``case import``: write ``case_dir/case.toml`` from a Palace config.

    Keeps the config and its sibling gsim provenance files in ``source/`` and pins
    the mesh.

    Refuses an existing case file. The import is a starting point: problem
    settings a suite case changes (frequencies, ABC order) are edited after.
    """
    if (case_dir / CASE_FILE).exists():
        msg = f"{case_dir / CASE_FILE} exists; import writes new cases only"
        raise CaseError(msg)
    for path in (config_path, mesh):
        if not path.is_file():
            msg = f"{path}: no such file"
            raise CaseError(msg)
    config = json.loads(config_path.read_text())
    unit = _unit(float(config["Model"]["L0"]))
    raw = import_palace(
        config,
        physical_names(mesh),
        name=case_dir.name,
        voided=is_voided(config_path),
        frequencies_hz=frequencies_hz,
    )
    source = case_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, source / "config.json")
    for name in PROVENANCE[1:]:
        sibling = config_path.with_name(name)
        if sibling.exists():
            shutil.copyfile(sibling, source / name)
    write_case(case_dir, raw)
    pinned = pin(case_dir, mesh, unit, stores)
    if not pinned.committed:
        raw = tomllib.loads((case_dir / CASE_FILE).read_text())
        raw["size"] = "large"
        write_case(case_dir, raw)
    load_case(case_dir)
    return pinned
