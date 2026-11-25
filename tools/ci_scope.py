"""Classify the paths a change touches into a CI scope.

Reads changed paths on standard input, one per line (blank lines ignored),
and prints ``scope=<s>`` so the output can be appended to ``$GITHUB_OUTPUT``::

    git diff --name-only BEFORE SHA | python tools/ci_scope.py

``none``: every path is under ``whitepaper/`` or is a root Markdown file other
than ``README.md`` (no test reads them). ``docs``: every path is in ``none``,
under ``docs/``, or a root ``*.md`` (README.md included, unit tests read it).
``full``: anything else, and an empty list (a diff that could not be computed
must not skip tests).

It uses the standard library only: the CI ``changes`` job runs it with a bare
interpreter, before any environment is synced.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable


def _root_markdown(path: str) -> bool:
    return "/" not in path and path.endswith(".md")


def _is_none(path: str) -> bool:
    return path.startswith("whitepaper/") or (
        _root_markdown(path) and path != "README.md"
    )


def _is_docs(path: str) -> bool:
    return _is_none(path) or path.startswith("docs/") or _root_markdown(path)


def scope(paths: Iterable[str]) -> str:
    """Return ``"none"``, ``"docs"`` or ``"full"`` for the changed ``paths``."""
    paths = list(paths)
    if not paths:
        return "full"
    if all(_is_none(p) for p in paths):
        return "none"
    if all(_is_docs(p) for p in paths):
        return "docs"
    return "full"


def main() -> int:
    """Print ``scope=<s>`` for the paths on standard input."""
    paths = [line.strip() for line in sys.stdin if line.strip()]
    print(f"scope={scope(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
