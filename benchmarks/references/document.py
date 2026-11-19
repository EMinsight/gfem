"""The ``reference.json`` file (``suite-reference/1``): fields, checks and writer.

Fields (union of tickets 08 and 15, spec gap G1): ``format``, ``id``
(``<case>-<YYYYMMDD>-<sha7>``), ``family``, ``computed_utc``, ``case``
(``name``, ``definition_sha256``, ``mesh_sha256`` of the suite case it was computed
for), ``physics_sha256`` (the case minus mesh, order, name, size, ``reference``
and ``est_s``; what the scorer matches), ``meshes`` (``<sha256>`` -> ``level``
and ``dof`` per order: the DOF guard's pins), ``method`` (``closed_form``,
``semi_analytic``, ``richardson``, ``overkill`` or ``none``, then ``reason``),
``quantities`` (per quantity ``values``, ``bound``, ``bound_kind``, ``metric``),
``ceiling`` and ``same_mesh_error`` per order, ``cross_checks``, ``inputs``,
``provenance``, ``runs``, ``cost_core_h``.

A file is replaced only with a new identity (``id``, ``computed_utc``): rewriting an
unchanged reference keeps its bytes, so ``reference CASE`` is idempotent.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.harness import case as case_mod

FORMAT = "suite-reference/1"
FILE = "reference.json"
METHODS = ("closed_form", "semi_analytic", "richardson", "overkill", "none")
REQUIRED = (
    "format",
    "id",
    "family",
    "computed_utc",
    "case",
    "physics_sha256",
    "meshes",
    "method",
    "quantities",
    "ceiling",
    "same_mesh_error",
    "cross_checks",
    "inputs",
    "provenance",
    "runs",
    "cost_core_h",
)
OPTIONAL = ("reason", "target")
QUANTITY_KEYS = ("values", "bound", "bound_kind", "metric")
BOUND_KINDS = ("absolute", "relative")


def problems(doc: Mapping[str, Any]) -> list[str]:
    """What makes ``doc`` not a ``suite-reference/1`` file (empty when valid)."""
    found = [f"missing {k}" for k in REQUIRED if k not in doc]
    found += [f"unknown key {k}" for k in doc if k not in (*REQUIRED, *OPTIONAL)]
    if doc.get("format") != FORMAT:
        found.append(f"format {doc.get('format')!r} is not {FORMAT}")
    if doc.get("method") not in METHODS:
        found.append(f"method {doc.get('method')!r} not in {', '.join(METHODS)}")
    if doc.get("method") == "none" and not doc.get("reason"):
        found.append("method none without a reason")
    for key in ("definition_sha256", "mesh_sha256"):
        if not isinstance(doc.get("case"), dict) or key not in doc["case"]:
            found.append(f"case.{key} missing")
    for sha, mesh in (doc.get("meshes") or {}).items():
        if not isinstance(mesh, dict) or "dof" not in mesh or "level" not in mesh:
            found.append(f"meshes.{sha[:12]} needs level and dof")
    for name, quantity in (doc.get("quantities") or {}).items():
        missing = [k for k in QUANTITY_KEYS if k not in quantity]
        if missing:
            found.append(f"quantities.{name} misses {', '.join(missing)}")
        elif quantity["bound_kind"] not in BOUND_KINDS:
            found.append(f"quantities.{name}.bound_kind {quantity['bound_kind']!r}")
    return found


def load(path: Path) -> dict[str, Any] | None:
    """The reference file at ``path``, or ``None`` when absent or not JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def case_block(case: case_mod.Case) -> dict[str, str]:
    """The ``case`` block: the suite case and its current pins."""
    return {
        "name": case.name,
        "definition_sha256": case_mod.definition_sha256(case),
        "mesh_sha256": case.mesh.sha256,
    }


def skeleton(case: case_mod.Case, *, commit: str, now: dt.datetime) -> dict[str, Any]:
    """Every field in order, identity set, the case's own mesh at level 0."""
    return {
        "format": FORMAT,
        "id": f"{case.name}-{now:%Y%m%d}-{commit[:7]}",
        "family": case.family,
        "computed_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case": case_block(case),
        "physics_sha256": case_mod.physics_sha256(case),
        "meshes": {case.mesh.sha256: {"level": 0, "dof": {}}},
        "method": "none",
        "quantities": {},
        "ceiling": {},
        "same_mesh_error": {},
        "cross_checks": [],
        "inputs": {"meshes": [case.mesh.sha256]},
        "provenance": {"gfem_commit": commit},
        "runs": None,
        "cost_core_h": 0.0,
    }


def dumps(doc: Mapping[str, Any]) -> str:
    """The file's text: indented JSON in field order, one trailing newline."""
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def _identity(doc: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        doc.get("id"),
        doc.get("computed_utc"),
        (doc.get("provenance") or {}).get("gfem_commit"),
        doc.get("runs"),
    )


def _essence(doc: Mapping[str, Any]) -> dict[str, Any]:
    """``doc`` without its identity: what decides whether it changed.

    ``runs`` is identity too: it names the archive by the reference's own id.
    """
    out = copy.deepcopy(dict(doc))
    for key in ("id", "computed_utc", "runs"):
        out.pop(key, None)
    out.get("provenance", {}).pop("gfem_commit", None)
    return json.loads(json.dumps(out))


def keep_dof_pins(doc: dict[str, Any], old: Mapping[str, Any] | None) -> None:
    """Carry over DOF pins of ``old`` that ``doc`` does not set (hand-filled pins)."""
    if not old:
        return
    for sha, mesh in (old.get("meshes") or {}).items():
        if sha not in doc["meshes"]:
            continue
        dof = doc["meshes"][sha]["dof"]
        for order, count in (mesh.get("dof") or {}).items():
            dof.setdefault(order, count)
        doc["meshes"][sha]["dof"] = dict(sorted(dof.items()))


def write(path: Path, doc: dict[str, Any]) -> bool:
    """Write ``doc`` to ``path``; return False when the file already says the same.

    An unchanged reference keeps its identity and its bytes; a changed one is
    written with the identity of ``doc``.
    """
    found = problems(doc)
    if found:
        raise ValueError(f"{path}: not a reference: {'; '.join(found)}")
    old = load(path)
    keep_dof_pins(doc, old)
    if old is not None and _essence(old) == _essence(doc):
        doc["id"], doc["computed_utc"], commit, runs = _identity(old)
        doc["provenance"]["gfem_commit"] = commit
        doc["runs"] = runs
        if path.read_text() == dumps(doc):
            return False
    path.write_text(dumps(doc))
    return True
