"""Docstrings that hold math are raw strings.

``ruff check`` with the project's ``fix = true`` turns ``\\omega`` into
``\\\\omega`` in a docstring that is not raw, and MathJax then shows a line
break and "omega". A raw docstring keeps the LaTeX as written.
"""

import ast
import tokenize
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "gfem"
FILES = sorted(SRC.rglob("*.py"))


def docstring_tokens(path: Path):
    """Each docstring's (line, source text) as written, prefix and quotes included."""
    lines = {
        node.body[0].value.lineno
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    with path.open("rb") as f:
        for tok in tokenize.tokenize(f.readline):
            if tok.type == tokenize.STRING and tok.start[0] in lines:
                yield tok.start[0], tok.string


def problems(path: Path) -> list[str]:
    out = []
    for line, text in docstring_tokens(path):
        if ":math:" not in text and ".. math::" not in text:
            continue
        where = f"{path.name}:{line}"
        prefix = text[: len(text) - len(text.lstrip("rRbBuUfF"))]
        if "\\" in text and "r" not in prefix.lower():
            out.append(f"{where}: math docstring with a backslash is not a raw string")
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_a_math_docstring_is_raw(path):
    assert problems(path) == []


def test_a_plain_docstring_with_math_is_reported(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('def f():\n    """Under :math:`e^{+i\\\\omega t}`."""\n')
    assert len(problems(bad)) == 1
