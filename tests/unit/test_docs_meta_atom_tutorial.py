"""The meta-atom tutorial's committed run checks its physics in its closing cell.

The docs build executes no notebook, so the committed outputs are what the site
shows. The page solves nothing: it reads the run record of
tools/meta_atom_run.py (docs/tutorials/data/meta_atom.{json,npz}), refuses a
record whose script hashes differ, and its last code cell asserts the gradient
against the finite difference, the order-3 re-solve, the sum of the partial
cross sections against the cross sections and the gain over the start. This
reads the committed notebook and record only.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "docs" / "tutorials" / "meta-atom.ipynb"
DATA = NOTEBOOK.parent / "data"
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


def test_the_closing_cell_asserts_the_checks(cells):
    closing = _source(_code(cells)[-1])
    for name in (
        "fd_error",  # the gradient against the finite difference
        "order_gap",  # the order-3 re-solve of the final design
        "sum_gap",  # partial cross sections against cross_sections
        "gain",  # the gain over the starting design
        "single_channel",  # the sphere's 6 pi / k^2, a reference
    ):
        assert re.search(rf"\b{name}\b", closing), name


def test_the_numbers_the_closing_cell_asserts_are_quoted(cells):
    closing = _source(_code(cells)[-1])
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    quoted = set(NUMBER.findall(prose))
    tolerances = set(re.findall(r"(?:atol|rtol)=([-\d.e]+)", closing))
    limits = set(re.findall(r"assert_array_less\([^,]+,\s*([-\d.e]+)\)", closing))
    asserted = set(NUMBER.findall(closing)) - tolerances - limits
    assert sorted(asserted - quoted) == [], "asserted but not quoted in the prose"


def test_the_record_is_a_full_run_of_the_committed_scripts():
    record = json.loads((DATA / "meta_atom.json").read_text())
    assert record["profile"] == "full"
    for script, recorded in record["provenance"]["sha256"].items():
        current = hashlib.sha256((ROOT / script).read_bytes()).hexdigest()
        assert current == recorded, f"{script} changed since the record: rerun it"
    assert (DATA / "meta_atom.npz").is_file()


def test_the_record_holds_the_numbers_the_page_quotes(cells):
    settings = json.loads((DATA / "meta_atom.json").read_text())["settings"]
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    for value, digits in (
        (settings["start_um2"], 3),
        (settings["final_um2"], 3),
        (settings["order3"]["md"], 3),
        (settings["single_channel_um2"], 3),
    ):
        assert f"{value:.{digits}f}" in prose, value
    assert settings["gain"] > 5
    assert settings["fd"]["error"] < 1e-6


def test_the_page_never_calls_the_sphere_value_a_bound(cells):
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    assert "not a bound" in prose
    assert "upper bound" not in prose


def test_the_page_calls_multipoles_outside_jit_and_reads_its_mesh_data(cells):
    text = " ".join(_source(c) for c in cells)
    assert "outside {py:obj}`jax.jit`" in text
    assert (DATA / "meta_atom.msh").is_file()


def test_the_page_names_no_other_solver(cells):
    text = " ".join(_source(c) for c in cells).lower()
    assert "palace" not in text
    assert "gdsfactory" not in text
