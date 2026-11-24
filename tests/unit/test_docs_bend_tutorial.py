"""The bend tutorial ends at the drawn pattern, checked, written to GDS, solved again.

The docs build executes no notebook, so the committed outputs are what the site
shows; the CI tutorials job re-executes the page. Its design is the drawn
pattern of the last iterate (polygons), not a thresholded pixel image, and the
cell that solves it asserts that it transmits what the optimizer's last
iterate promised. The GDS file goes to a temporary directory, so a run leaves
nothing in the checkout. This reads the committed notebook only.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "bend.ipynb"
#: The largest gap between the drawn pattern's |S21|^2 and the last iterate's:
#: the solve reads the pattern's pixels bilinearly, the iterate's at beta = 32.
GAP = "0.02"


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def _code(cells):
    return [c for c in cells if c["cell_type"] == "code"]


def _prose(cells):
    return " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")


def test_the_committed_run_has_no_error_and_no_stderr(cells):
    code = _code(cells)
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
    streams = [o for c in code for o in c["outputs"] if o.get("name") == "stderr"]
    assert streams == [], "stderr in the committed outputs fails the docs build"


def test_the_design_is_the_drawn_pattern_not_a_thresholded_image(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert "> 0.5).astype" not in code
    assert "contourpy" not in code
    assert re.search(r"\bdrawn = grid\.pattern\(", code)
    assert re.search(r"\bfinal = grid\.area_fraction\(drawn\)", code)
    assert "LayerPattern.from_polygons" in code


def test_the_pattern_solve_asserts_the_gap_to_the_last_iterate(cells):
    (cell,) = [c for c in _code(cells) if "t_pattern - t_last" in _source(c)]
    assert re.search(
        rf"assert_array_less\(abs\(t_pattern - t_last\), {re.escape(GAP)}\)",
        _source(cell),
    )
    assert GAP in _prose(cells), "the asserted gap is not quoted in the prose"


def test_the_pattern_is_checked_against_an_example_rule(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert re.search(r"drawn\.check\(min_width=.+, min_spacing=.+\)", code)
    printed = " ".join(
        "".join(o.get("text", ""))
        for c in _code(cells)
        if ".check(" in _source(c)
        for o in c["outputs"]
    )
    assert "PatternCheck(" in printed
    assert "example" in _prose(cells)


def test_the_gds_file_goes_to_a_temporary_directory(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert re.search(r"\bout = Path\(tempfile\.mkdtemp\(\)\)", code)
    (call,) = re.findall(r"design\.write_gds\(([^,]+),", code)
    assert call.strip().startswith("out /"), call
    assert list(NOTEBOOK.parent.rglob("*.gds")) == []


def test_the_page_links_the_layout_recipe(cells):
    assert "{doc}`../recipes/fabricable-layout`" in _prose(cells)
