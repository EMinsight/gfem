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


def test_the_thermal_block_flags_the_density_rename_as_the_code_does():
    import warnings

    import pytest

    import gfem

    notes = (DOCS / "project" / "release-notes.md").read_text(encoding="utf-8")
    unreleased = notes.split("\n## ")[1]
    assert "\nThermal:\n" in unreleased
    block = " ".join(unreleased.split("\nThermal:\n")[1].split("\n\n")[0].split())
    assert block.startswith("- **Behaviour change: `density` is now `mass_density`.**")
    for said in ("`DeprecationWarning`", "`AttributeError`", "Job files"):
        assert said in block, said
    # what the block says the old name does
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        material = gfem.Material(density=2203.0)
    assert [w.category for w in caught] == [DeprecationWarning]
    assert material.mass_density == 2203.0
    with pytest.raises(AttributeError, match="mass_density"):
        _ = material.density
