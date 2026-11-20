"""Campaign reports: a deterministic Markdown and SVG directory (ADR 0413).

``python -m benchmarks report <campaign ids|dirs> --out DIR`` reads only the campaign
manifests, the records they list and the store's references (:mod:`.load`), computes
every printed number (:mod:`.views`), and writes ``DIR/index.md`` (:mod:`.markdown`),
``fig-*.svg`` (:mod:`.figures`, matplotlib imported lazily), ``report.json``
(``suite-report/1``) and ``inputs.sha256``. ``--no-figures`` needs the standard
library only.

``report --check DIR`` regenerates the report from the inputs ``DIR/report.json``
names (relative to ``DIR``) into a temporary directory and compares bytes. Figures
made with another matplotlib version, or with none importable, are not compared;
tables and JSON must still match exactly. ``report.json`` names the harness commit
that rendered it (``harness_revision``, ``-dirty`` when the checkout had changes);
``--check`` keeps the stored one and says when it differs from the checking tree's.
``report.json`` also records the reference families read per case
(``reference_families``, from the case files' ``[reference]`` key); ``--check``
reuses that map instead of the checking tree's case files.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.harness import matrix
from benchmarks.report import load, markdown, views
from benchmarks.report.load import ReportError

INDEX = "index.md"
JSON = "report.json"
INPUTS = "inputs.sha256"


def matplotlib_version() -> str | None:
    """The importable matplotlib's version, else None."""
    try:
        import matplotlib as mpl
    except ImportError:
        return None
    return str(mpl.__version__)


def harness_revision(root: Path = matrix.ROOT) -> str | None:
    """The commit of the harness tree ``root``, else None.

    A git checkout with changes (``git status --porcelain`` not empty) gets
    ``-dirty`` appended; a shipped tree (``COMMIT`` file, no ``.git``) is clean by
    construction.
    """
    try:
        commit = matrix.source_commit(root)[0]
    except matrix.PlanError:
        return None
    if not (root / ".git").exists():
        return commit
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return commit
    return f"{commit}-dirty" if status.strip() else commit


def _dumps(data: Any) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def build(
    directories: Sequence[Path],
    out: Path,
    flags: Mapping[str, Any],
    *,
    matplotlib: str | None,
    revision: str | None = None,
    families: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], load.Inputs]:
    """Load and check the inputs; the report as it would be written to ``out``.

    ``families`` is a stored report's reference-family map; without it the map
    comes from the case files of this tree.
    """
    inputs = load.load(
        directories,
        allow_mixed=bool(flags.get("allow_mixed")),
        rescore=flags.get("rescore") or {},
        families=families,
    )
    root = Path(os.path.relpath(inputs.root, Path(out).resolve())).as_posix()
    report = views.build(
        inputs,
        flags=flags,
        inputs_root=root,
        matplotlib=matplotlib,
        harness_revision=revision,
    )
    return report, inputs


def write(
    report: Mapping[str, Any], inputs: load.Inputs, out: Path, *, draw: bool
) -> None:
    """Write the report directory (figures only when ``draw``)."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fig-*.svg"):
        old.unlink()
    (out / INPUTS).write_text(inputs.manifest())
    (out / JSON).write_text(_dumps(report))
    (out / INDEX).write_text(markdown.render(report))
    if draw:
        from benchmarks.report import figures

        figures.render(report, out)


def generate(directories: Sequence[Path], out: Path, flags: Mapping[str, Any]) -> str:
    """Write a report of ``directories`` into ``out``; return its id."""
    version = None
    if not flags.get("no_figures"):
        version = matplotlib_version()
        if version is None:
            raise ReportError(
                "matplotlib is not importable: run with `uv run --group docs` or "
                "pass --no-figures"
            )
    report, inputs = build(
        directories, out, flags, matplotlib=version, revision=harness_revision()
    )
    write(report, inputs, out, draw=version is not None)
    return str(report["id"])


def check(directory: Path) -> tuple[bool, list[str]]:
    """Regenerate ``directory``'s report and compare; ``(same, messages)``."""
    directory = Path(directory)
    try:
        stored = json.loads((directory / JSON).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{directory / JSON}: {error}") from error
    root = (directory / stored["inputs_root"]).resolve()
    dirs = [root / rel for rel in stored["inputs"]]
    flags = stored["flags"]
    messages = []
    made_with = stored.get("matplotlib")
    current = matplotlib_version() if made_with else None
    compare_figures = made_with is not None and current == made_with
    if made_with and current is None:
        messages.append("figures not compared: matplotlib is not importable")
    elif made_with and current != made_with:
        messages.append(
            f"figures regenerated with matplotlib {current} (made with {made_with}); "
            "not compared"
        )
    here = harness_revision()
    rendered_at = stored.get("harness_revision") or here
    if rendered_at != here:
        messages.append(f"rendered at harness {rendered_at}; regenerated at {here}")
    report, inputs = build(
        dirs,
        directory,
        flags,
        matplotlib=made_with,
        revision=rendered_at,
        families=stored.get("reference_families"),
    )
    with tempfile.TemporaryDirectory(prefix="bench-report-") as tmp:
        fresh = Path(tmp)
        write(report, inputs, fresh, draw=compare_figures)
        names = {p.name for p in fresh.iterdir()}
        have = {p.name for p in directory.iterdir() if p.is_file()}
        if not compare_figures:
            have = {n for n in have if not n.startswith("fig-")}
        drift = sorted(names ^ have)
        drift += sorted(
            n
            for n in names & have
            if (fresh / n).read_bytes() != (directory / n).read_bytes()
        )
    if drift:
        messages.append("drift: " + ", ".join(drift))
    return not drift, messages
