"""The grating-coupler tutorial's committed run checks its physics in its closing cell.

The docs build executes no notebook, so the committed outputs are what the site
shows; the CI tutorials job re-executes the page, and its last code cell
asserts the checks the page relies on: the design's gain over the grating as
meshed, the adjoint gradient against a finite difference on the tutorial's
mesh, the power balance of the guide's watt, the port-fed coupling against
the beam-lit port wave (reciprocity) and the coupling read on other planes.
This reads the committed notebook only.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "grating.ipynb"
MESH_DATA = NOTEBOOK.parent / "data"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


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
        "gain",  # the design couples more than the grating as meshed
        "gradient_error",  # adjoint against finite differences, on this mesh
        "balance",  # reflected + passed on + up + down = 1
        "reciprocity",  # port-fed coupling = beam-lit port wave
        "planes",  # the coupling read on other planes
    ):
        assert re.search(rf"\b{name}\b", closing), name


def test_the_numbers_the_closing_cell_asserts_are_quoted(cells):
    closing = _source(_code(cells)[-1])
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    quoted = set(NUMBER.findall(prose))
    asserted = set(NUMBER.findall(closing))
    assert sorted(asserted - quoted) == [], "asserted but not quoted in the prose"


def test_the_page_designs_through_the_morph_and_the_coupling(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert "boundary_morph" in code
    assert "beam_coupling" in code
    assert "IncidentField" in code  # the z-invariant fiber, written on the page
    assert "port_wave" in code  # the reciprocal route


def test_the_page_reads_its_mesh_from_the_data_folder(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert '"data/grating.msh"' in code
    assert (MESH_DATA / "grating.msh").is_file()


def test_the_page_names_no_other_solver(cells):
    text = " ".join(_source(c) for c in cells).lower()
    assert "palace" not in text
    assert "gdsfactory" not in text
