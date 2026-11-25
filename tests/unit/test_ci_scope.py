import subprocess
import sys
from pathlib import Path

import ci_scope
import pytest

TOOL = Path(ci_scope.__file__)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["whitepaper/main.typ"], "none"),
        (["CHANGELOG.md"], "none"),
        (["whitepaper/a.typ", "NOTES.md"], "none"),
        (["README.md"], "docs"),
        (["docs/index.rst"], "docs"),
        (["docs/x.py"], "docs"),
        (["docs/a.md", "whitepaper/b.typ", "CHANGELOG.md"], "docs"),
        (["src/gfem/a.py"], "full"),
        (["tools/x.py"], "full"),
        (["tests/unit/test_a.py"], "full"),
        (["pyproject.toml"], "full"),
        ([".github/workflows/ci.yml"], "full"),
        (["docs/a.md", "tools/x.py"], "full"),
        (["sub/NOTES.md"], "full"),
        ([], "full"),
    ],
)
def test_scope(paths, expected):
    assert ci_scope.scope(paths) == expected


def _run(stdin):
    return subprocess.run(
        [sys.executable, str(TOOL)],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_cli_prints_scope_and_ignores_blank_lines():
    assert _run("whitepaper/a.typ\n\nCHANGELOG.md\n\n") == "scope=none\n"
    assert _run("docs/a.md\n  \nREADME.md\n") == "scope=docs\n"
    assert _run("src/gfem/a.py\n") == "scope=full\n"


def test_cli_empty_stdin_is_full():
    assert _run("") == "scope=full\n"
    assert _run("\n\n") == "scope=full\n"
