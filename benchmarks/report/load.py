"""Read a report's inputs: manifests, the records they list, references (ADR 0413).

The inputs are exactly each campaign's ``campaign.json``, the records it lists
(``records`` {path: sha256}, each checked before it is read), the per-job machine
benchmarks ``jobs/<job>/machine.json`` beside them, and the archived references of
the store (``references/<id>/reference.json``) of the cases the records ran. No log,
no clock, no gfem import.

Every input path is relative to the **store root**: the directory holding
``campaigns/`` when a campaign directory sits at ``<root>/campaigns/<id>``, else the
campaign directory's parent. ``inputs.sha256`` lists one ``sha256  relpath`` line per
input, sorted by path; its own sha256 is the report's digest.

Refused: a record whose sha256 differs from its manifest entry, an unknown schema
major, a mixed campaign (unless ``allow_mixed``), and records of ``run --local``
mixed with cluster records. Standard library only (ADR 0400).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.harness import config, records

REFERENCES = "references"
REFERENCE_FILE = "reference.json"
LOCAL = "local"


class ReportError(records.RecordError):
    """Inputs a report refuses to read."""


@dataclass
class LoadedCampaign:
    """One campaign: its manifest, its records and its jobs' machine benchmarks."""

    rel: str
    """The campaign directory relative to the store root."""
    manifest: records.Campaign
    rows: list[records.SuiteRecord]
    machines: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Slurm job id -> ``machine.json``."""
    mixed: list[str] = field(default_factory=list)


@dataclass
class Reference:
    """An archived converged reference."""

    rel: str
    sha256: str
    doc: dict[str, Any]


@dataclass
class Inputs:
    """Everything a report reads, and the inputs manifest."""

    root: Path
    campaigns: list[LoadedCampaign]
    references: dict[str, Reference]
    files: dict[str, str]
    """Relative path -> sha256 of every input."""
    local_only: bool = False

    def manifest(self) -> str:
        """``inputs.sha256``: ``sha256  relpath`` per input, sorted by path."""
        return "".join(f"{self.files[p]}  {p}\n" for p in sorted(self.files))

    def digest(self) -> str:
        """sha256 of :meth:`manifest`."""
        return hashlib.sha256(self.manifest().encode()).hexdigest()


def store_root(directory: Path) -> Path:
    """The store root of a campaign directory."""
    directory = directory.resolve()
    if directory.parent.name == "campaigns":
        return directory.parent.parent
    return directory.parent


def resolve(names: Sequence[str], out: Path | None = None) -> list[Path]:
    """Campaign directories of ``names`` (a directory, or a campaign id).

    An id is looked up as ``campaigns/<id>`` in the store holding ``out``
    (``<store>/reports/<report id>``), then in ``../gfem-bench-data`` beside the
    repository (the README's store worktree).
    """
    stores = []
    if out is not None:
        stores.append(Path(out).resolve().parent.parent)
    stores.append(config.PACKAGE.parent.parent / "gfem-bench-data")
    found = []
    for name in names:
        path = Path(name)
        if (path / "campaign.json").is_file():
            found.append(path)
            continue
        hits = [s / "campaigns" / name for s in stores]
        hit = next((h for h in hits if (h / "campaign.json").is_file()), None)
        if hit is None:
            raise ReportError(
                f"{name}: no campaign.json there nor in "
                + ", ".join(str(h) for h in hits)
            )
        found.append(hit)
    return found


def _sha(path: Path) -> str:
    return records.file_sha256(path)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _campaign(directory: Path, root: Path, files: dict[str, str]) -> LoadedCampaign:
    path = directory / "campaign.json"
    manifest = records.load_campaign(path)
    files[_rel(path, root)] = _sha(path)
    rows = []
    for rel, want in sorted(manifest.records.items()):
        p = directory / rel
        if not p.is_file():
            raise ReportError(f"{p}: listed in {path} but missing")
        got = _sha(p)
        if got != want:
            raise ReportError(
                f"{p}: sha256 {got[:12]} differs from the manifest's {want[:12]}"
            )
        rows.append(records.load_record(p))
        files[_rel(p, root)] = got
    machines = {}
    for p in sorted((directory / "jobs").glob("*/machine.json")):
        machines[p.parent.name] = json.loads(p.read_text())
        files[_rel(p, root)] = _sha(p)
    return LoadedCampaign(_rel(directory, root), manifest, rows, machines)


def _references(
    root: Path, families: set[str], named: set[str], files: dict[str, str]
) -> dict[str, Reference]:
    """References named by the records or of their cases' families, by id."""
    out = {}
    for p in sorted((root / REFERENCES).glob(f"*/{REFERENCE_FILE}")):
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as error:
            raise ReportError(f"{p}: {error}") from error
        rid = doc.get("id", p.parent.name)
        if rid not in named and doc.get("family") not in families:
            continue
        rel = _rel(p, root)
        files[rel] = _sha(p)
        out[rid] = Reference(rel, files[rel], doc)
    return out


def load(
    directories: Sequence[Path],
    *,
    allow_mixed: bool = False,
    rescore: Mapping[str, str] | None = None,
) -> Inputs:
    """Read and check every input of a report over ``directories``."""
    if not directories:
        raise ReportError("no campaign given")
    roots = {store_root(d) for d in directories}
    if len(roots) > 1:
        raise ReportError(
            "campaigns from different stores: " + ", ".join(map(str, sorted(roots)))
        )
    root = roots.pop()
    files: dict[str, str] = {}
    campaigns = [_campaign(Path(d), root, files) for d in directories]
    ids = [c.manifest.id for c in campaigns]
    if len(set(ids)) != len(ids):
        raise ReportError(f"a campaign is given twice: {', '.join(ids)}")
    for c in campaigns:
        c.mixed = records.refuse_mixed(c.manifest, c.rows, allow_mixed=allow_mixed)
    rows = [r for c in campaigns for r in c.rows]
    local = {
        r.provenance.machine.name == LOCAL
        for r in rows
        if r.provenance.machine.name is not None
    }
    if local == {True, False}:
        raise ReportError(
            "records of run --local mixed with cluster records: a local run is "
            "never compared with a cluster run"
        )
    named = {r.case.reference.id for r in rows if r.case.reference is not None}
    named |= set((rescore or {}).values())
    families = {r.run.case for r in rows}
    references = _references(root, families, named, files)
    for case, rid in (rescore or {}).items():
        if rid not in references:
            raise ReportError(
                f"--rescore {case}={rid}: no {REFERENCES}/{rid}/{REFERENCE_FILE} "
                f"in the store {root}"
            )
    return Inputs(root, campaigns, references, files, local_only=local == {True})
