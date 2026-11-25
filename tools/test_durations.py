"""Fold junit XML from CI runs into ``tests/durations.json``.

Each test's key is ``<junit classname>::<name>``; its value is the median of
its durations over the given files, rounded to 0.1 s. Skipped testcases are
left out, and only observed keys are written, so tests that no longer exist
drop out of the file. Commit the result by hand.

From junit files already on disk::

    python tools/test_durations.py --junit unit-a.xml unit-b.xml

From a CI run (default: the latest completed ``schedule`` run of ``CI``),
downloading its ``durations-*`` artifacts with the ``gh`` CLI::

    python tools/test_durations.py
    python tools/test_durations.py --run 123456789

``--out`` sets the output file (default ``tests/durations.json``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "tests" / "durations.json"


def merge(junit_paths: Iterable[Path | str]) -> dict[str, float]:
    """Median seconds per ``classname::name`` over junit files, skips excluded."""
    seen: dict[str, list[float]] = {}
    for path in junit_paths:
        # Junit files come from our own CI runs, not untrusted input.
        root = ET.parse(path).getroot()  # noqa: S314
        for case in root.iter("testcase"):
            if case.find("skipped") is not None:
                continue
            key = f"{case.get('classname', '')}::{case.get('name', '')}"
            seen.setdefault(key, []).append(float(case.get("time", 0.0)))
    return {k: round(statistics.median(v), 1) for k, v in seen.items()}


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args],  # noqa: S607  # gh is resolved from PATH on purpose
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _latest_scheduled_run() -> str:
    out = _gh(
        "run", "list", "--workflow", "CI", "--event", "schedule",
        "--status", "completed", "--limit", "1", "--json", "databaseId",
        "--jq", ".[0].databaseId",
    )  # fmt: skip
    run = out.strip()
    # A nightly with one flaky failure is never "success", yet it uploads its
    # durations artifacts (if: always()), so any completed run will do.
    if not run or run == "null":
        raise SystemExit("no completed scheduled CI run found; pass --run ID")
    return run


def _download(run: str, tmp: Path) -> list[Path]:
    try:
        _gh("run", "download", run, "-p", "durations-*", "-D", str(tmp))
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        raise SystemExit(
            f"could not download the durations-* artifacts of run {run} "
            f"(expired, or none uploaded?): {detail}"
        ) from error
    found = sorted(tmp.rglob("*.xml"))
    if not found:
        raise SystemExit(f"run {run} has no durations-* junit files")
    return found


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--junit", nargs="+", type=Path, metavar="FILE")
    source.add_argument("--run", metavar="ID", help="CI run id to download from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.junit:
        durations = merge(args.junit)
    else:
        run = args.run or _latest_scheduled_run()
        with tempfile.TemporaryDirectory() as tmp:
            durations = merge(_download(run, Path(tmp)))
    args.out.write_text(
        json.dumps(durations, indent=0, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(durations)} durations to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
