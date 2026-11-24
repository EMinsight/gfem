"""The modulator tutorial's committed run checks its physics in its closing cell.

The docs build executes no notebook, so the committed outputs are what the site
shows; the CI tutorials job re-executes the page, and its last code cell
asserts the checks the page relies on: the arms' shifts equal and opposite,
the push-pull V_pi L at the published section, the tensor's shares, the gap
gradient against a difference, and an SLSQP optimum on the clearance
constraint inside the gap and width bounds. This reads the committed notebook, and
builds the page's section with its helper, docs/tutorials/tfln_section.py,
which the page imports and offers for download.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "modulator.ipynb"
HELPER = NOTEBOOK.with_name("tfln_section.py")
NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def _code(cells):
    return [c for c in cells if c["cell_type"] == "code"]


def _printed(cells) -> str:
    return "".join(
        "".join(o.get("text", ""))
        for c in _code(cells)
        for o in c["outputs"]
        if o.get("name") == "stdout"
    )


def test_the_committed_run_reached_the_closing_cell(cells):
    code = _code(cells)
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
    streams = [o for c in code for o in c["outputs"] if o.get("name") == "stderr"]
    assert streams == [], "stderr in the committed outputs fails the docs build"
    assert "page run in" in _printed(cells)


def test_the_printed_push_pull_v_pi_l_is_finite_and_positive(cells):
    (line,) = [
        line
        for line in _printed(cells).splitlines()
        if line.startswith("V_pi L push-pull")
    ]
    (value,) = (float(n) for n in NUMBER.findall(line))
    assert np.isfinite(value)
    assert value > 0


def test_the_closing_cell_asserts_the_physics_checks(cells):
    closing = _source(_code(cells)[-1])
    for name in (
        "s_b / s_a",  # the mirror symmetry of the push-pull section
        "v_pi_l",  # the push-pull V_pi L at the published section
        "density",  # the integrand's sum is the shift
        "shares",  # the tensor's shares sum to 1
        "gamma",  # the overlap factor
        "fd_error",  # the gap gradient against a difference
        "result.success",  # the optimizer stopped
        "constraint_opt",  # on the clearance constraint
        "g_opt",  # inside the gap bounds
        "w_opt",  # inside the width bounds
        "v_pi_l_opt",  # below the start
    ):
        assert name in closing, name
    assert "1.62" in closing
    assert "atol=1e-4" in closing  # s_b / s_a = -1 to 1e-4
    assert "less(fd_error, 1e-3)" in closing  # measured about 2e-5


def test_the_asserted_v_pi_l_is_quoted(cells):
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    assert "1.62 V·cm" in prose
    assert "1.8 V·cm" in prose  # the measured value, an observation only


def test_the_page_takes_lithium_niobate_from_the_library(cells):
    code = " ".join(_source(c) for c in _code(cells))
    assert "materials.lithium_niobate(" in code
    assert "materials.lithium_niobate_static(" in code
    text = " ".join(_source(c) for c in cells)
    # the library's tensor is the clamped set the page's prose relies on
    assert "unclamped" not in text
    assert "Turner 1966" in text


def test_the_page_offers_its_section_builder(cells):
    text = " ".join(_source(c) for c in cells)
    assert "from tfln_section import tfln_section" in text
    assert "{download}`tfln_section.py`" in text
    assert HELPER.is_file()


def test_the_section_builder_names_what_the_page_reads(tmp_path):
    sys.path.insert(0, str(HELPER.parent))
    try:
        from tfln_section import tfln_section
    finally:
        sys.path.remove(str(HELPER.parent))
    section = tfln_section(3.5e-6, 0.9e-6, directory=tmp_path, coarsen=2.0)
    static = section.static
    assert set(static.volumes) == {
        "air", "cladding", "box", "slab", "ridge_a", "ridge_b",
        "signal", "ground_r", "ground_l",
    }  # fmt: skip
    for name in ("signal_s", "ground_s", "ridge_a_s", "ridge_b_s", "ln_if", "front"):
        assert name in static.surfaces, name
    assert section.ridge_centre == pytest.approx(2e-6 + 3.5e-6 / 2)
    assert section.ground_edge - section.signal_edge == pytest.approx(3.5e-6)
    for window, sign in ((section.window_a, 1), (section.window_b, -1)):
        assert set(window.volumes) == {"ridge", "slab", "box", "cladding"}
        # the window lies inside the static section, out to the electrodes
        nodes = np.asarray(window.nodes)
        x = sign * nodes[:, 0]
        assert x.min() == pytest.approx(section.signal_edge)
        assert x.max() == pytest.approx(section.ground_edge)
        assert window.section_thickness == static.section_thickness
