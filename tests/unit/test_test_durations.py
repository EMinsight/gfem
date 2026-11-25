import json
import subprocess
import sys
from pathlib import Path

import test_durations as td

TOOL = Path(td.__file__)


def _junit(path: Path, cases) -> Path:
    """Write a junit file from ``(classname, name, seconds, skipped)`` rows."""
    rows = []
    for cls, name, t, skipped in cases:
        body = "<skipped/>" if skipped else ""
        rows.append(
            f'<testcase classname="{cls}" name="{name}" time="{t}">{body}</testcase>'
        )
    path.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        + "".join(rows)
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


def _two(tmp_path):
    a = _junit(
        tmp_path / "a.xml",
        [
            ("tests.unit.test_x", "test_a", 1.04, False),
            ("tests.unit.test_x", "test_b[p-1]", 10.0, False),
            ("tests.unit.test_x", "test_skip", 0.0, True),
            ("tests.unit.test_x.TestC", "test_c", 2.26, False),
        ],
    )
    b = _junit(
        tmp_path / "b.xml",
        [
            ("tests.unit.test_x", "test_a", 1.36, False),
            ("tests.unit.test_x", "test_b[p-1]", 20.0, False),
            ("tests.unit.test_x", "test_skip", 3.0, True),
        ],
    )
    return a, b


def test_merge_median_rounding_skips_and_key_format(tmp_path):
    a, b = _two(tmp_path)
    assert td.merge([a, b]) == {
        "tests.unit.test_x::test_a": 1.2,
        "tests.unit.test_x::test_b[p-1]": 15.0,
        "tests.unit.test_x.TestC::test_c": 2.3,
    }


def test_merge_median_of_three_and_single(tmp_path):
    cases = lambda t: [("m", "t", t, False)]  # noqa: E731
    paths = [
        _junit(tmp_path / f"{i}.xml", cases(t)) for i, t in enumerate([1.0, 9.0, 2.0])
    ]
    assert td.merge(paths) == {"m::t": 2.0}
    assert td.merge(paths[:1]) == {"m::t": 1.0}


def test_merge_skipped_in_one_file_only_uses_others(tmp_path):
    a = _junit(tmp_path / "a.xml", [("m", "t", 0.0, True)])
    b = _junit(tmp_path / "b.xml", [("m", "t", 4.0, False)])
    assert td.merge([a, b]) == {"m::t": 4.0}


def test_cli_junit_writes_sorted_file(tmp_path):
    a, b = _two(tmp_path)
    out = tmp_path / "durations.json"
    subprocess.run(
        [sys.executable, str(TOOL), "--junit", str(a), str(b), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    data = {
        "tests.unit.test_x::test_a": 1.2,
        "tests.unit.test_x::test_b[p-1]": 15.0,
        "tests.unit.test_x.TestC::test_c": 2.3,
    }
    assert (
        out.read_text(encoding="utf-8")
        == json.dumps(data, indent=0, sort_keys=True) + "\n"
    )
