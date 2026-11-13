"""The transmon tutorial quotes only values its own closing cell asserts.

The page's prose rounds the numbers its cells print (frequencies, the
dispersive shift, its gradient). The docs build executes no notebook, so a
quoted number that drifted from the code would stay on the site; the CI
tutorials job re-executes the page, and its last code cell asserts each quoted
value against the computed one. This reads the committed notebook only: every
number the closing cell asserts is quoted in the prose, and the committed
outputs are those of a run that reached that cell without an error.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "transmon.ipynb"
# a decimal literal: -2.496, 5.643, -1.109e-2 (integers are counts, not values)
NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def _closing(cells):
    return [c for c in cells if c["cell_type"] == "code"][-1]


def test_the_closing_cell_asserts_the_quoted_values(cells):
    closing = _source(_closing(cells))
    assert len(re.findall(r"\bassert_\w+\(", closing)) >= 8, "too few checks"
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    quoted = set(NUMBER.findall(prose.replace("\u2212", "-")))
    tolerances = set(re.findall(r"(?:atol|rtol)=([-\d.e]+)", closing))
    asserted = set(NUMBER.findall(closing)) - tolerances
    assert sorted(asserted - quoted) == [], "asserted but not quoted in the prose"


def test_the_committed_run_reached_the_closing_cell(cells):
    code = [c for c in cells if c["cell_type"] == "code"]
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
