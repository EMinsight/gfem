"""The metalens tutorial's committed run checks its physics in its closing cell.

The docs build executes no notebook, so the committed outputs are what the site
shows; the CI tutorials job re-executes the page, and its last code cell
asserts the closed-form and conservation checks the page relies on. This reads
the committed notebook only.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "metalens.ipynb"
MESH_DATA = NOTEBOOK.parent / "data"
NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def _code(cells):
    return [c for c in cells if c["cell_type"] == "code"]


def test_the_committed_run_reached_the_closing_cell(cells):
    code = _code(cells)
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
    streams = [o for c in code for o in c["outputs"] if o.get("name") == "stderr"]
    assert streams == [], "stderr in the committed outputs fails the docs build"


def test_the_closing_cell_asserts_the_physics_checks(cells):
    closing = _source(_code(cells)[-1])
    for name in (
        "balance_cell",  # the cell's power balance
        "cross_polarization",  # TE stays TE
        "order3_change",  # the table's mesh error against third order
        "derivative_error",  # the stored derivative against a central difference
        "hermite_error",  # held-out widths against the interpolant
        "ideal_focus",  # the ideal lens against the stationary-phase focus
        "matched",  # the phase-matched lens at both colours
        "green_focus",  # its green focus at the chromatic shift
        "optimized",  # the two-colour optimum
        "fd_error",  # the gradient against a central difference
        "supercell_balance",  # the supercell's power balance
        "deflected",  # the deflected order, full solve against the approximation
    ):
        assert re.search(rf"\b{name}\b", closing), name


def test_the_numbers_the_closing_cell_asserts_are_quoted(cells):
    closing = _source(_code(cells)[-1])
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    quoted = set(NUMBER.findall(prose.replace("\u2212", "-")))
    tolerances = set(re.findall(r"(?:atol|rtol)=([-\d.e]+)", closing))
    limits = set(re.findall(r"assert_array_less\([^,]+,\s*([-\d.e]+)\)", closing))
    asserted = set(NUMBER.findall(closing)) - tolerances - limits
    assert sorted(asserted - quoted) == [], "asserted but not quoted in the prose"


def test_the_page_reads_its_meshes_from_the_data_folder(cells):
    code = " ".join(_source(c) for c in _code(cells))
    for name in ("metalens_cell.msh", "metalens_supercell.msh"):
        assert f'"data/{name}"' in code
        assert (MESH_DATA / name).is_file()


def test_the_page_names_no_other_solver(cells):
    text = " ".join(_source(c) for c in cells).lower()
    assert "palace" not in text
    assert "gdsfactory" not in text
