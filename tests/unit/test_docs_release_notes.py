"""No docs page ships a placeholder.

A number not yet measured is written between angle brackets while a change is in
flight (``⟨0.78 → 0.46⟩``). It must be measured or its sentence deleted before the
docs are built for a release.
"""

from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
PLACEHOLDER = ("⟨", "⟩")


def test_no_docs_page_holds_a_placeholder():
    pages = sorted(DOCS.rglob("*.md")) + sorted(DOCS.rglob("*.rst"))
    assert pages
    found = [
        f"{page.relative_to(DOCS)}:{number}"
        for page in pages
        for number, line in enumerate(page.read_text().splitlines(), start=1)
        if any(mark in line for mark in PLACEHOLDER)
    ]
    assert not found, f"placeholders left in: {found}"
