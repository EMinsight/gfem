"""The photonic-crystal tutorial's committed run checks its physics in its closing cell.

The docs build executes no notebook, so the committed outputs are what the site
shows; the CI tutorials job re-executes the page, and its last code cell asserts
the checks the page relies on: the Ez gap against the published 31.4 %, the band
edges' places, the Hz overlap, the doublet at M, the band group velocity against
its long-wave limit, the band edges and the bands' slope, and the gap's gradient
against perturbation theory and a central difference. This reads the committed
notebook.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).parents[2] / "docs"
NOTEBOOK = DOCS / "tutorials" / "photonic-crystal.ipynb"
NUMBER = re.compile(r"-?\d+\.\d+")


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


def test_the_printed_ez_gap_is_the_published_one(cells):
    (line,) = [
        line for line in _printed(cells).splitlines() if line.startswith("Ez gap: ")
    ]
    (percent,) = re.findall(r"centre (\d+\.\d+) %", line)
    assert float(percent) == pytest.approx(31.4, abs=0.1)


def test_the_gap_edges_and_the_hz_overlap_are_printed(cells):
    printed = _printed(cells)
    assert "band 0 tops out at M, band 1 bottoms out at X" in printed
    (line,) = [x for x in printed.splitlines() if x.startswith("Hz band 0 reaches")]
    assert float(NUMBER.findall(line)[-1]) > 0  # the overlap, no Hz gap
    assert "anchored at every wave vector: True" in printed
    assert "lost bands: False" in printed


def test_the_printed_slowing_is_finite_and_above_one(cells):
    (line,) = [line for line in _printed(cells).splitlines() if "light slowed" in line]
    (factor,) = re.findall(r"slowed (\d+)-fold", line)
    assert np.isfinite(float(factor))
    assert float(factor) > 1


def test_the_gradient_cell_is_the_decisive_one(cells):
    (cell,) = [c for c in _code(cells) if "jax.value_and_grad" in _source(c)]
    assert "decisive" in cell["metadata"]["tags"]
    assert "gfem.band_diagram" in _source(cell)


def test_the_closing_cell_asserts_the_physics_checks(cells):
    closing = _source(_code(cells)[-1])
    for name in (
        "gap_ratio",  # the Ez gap against the published value
        "edge_k",  # band 0 tops out at M, band 1 bottoms out at X
        "hz_overlap",  # no Hz gap
        "doublet_split",  # the mesh's asymmetry at M
        "anchored",  # the target holds the bands' numbers
        "lost",  # the path is fine enough
        "v_long",  # the long-wave limit of the group velocity
        "v_edge",  # standing waves at the band edges
        "slope_error",  # the group velocity against the bands' slope
        "perturbation_error",  # the gradient against perturbation theory
        "fd_error",  # the gradient against a central difference
        "delta_mesh",  # the finer mesh
    ):
        assert name in closing, name
    assert "0.314" in closing
    assert "1e-3" in closing  # gap_ratio, v_long and v_edge
    assert "(perturbation_error, 1e-8)" in closing
    assert "(fd_error, 1e-6)" in closing


def test_the_ez_gap_is_quoted_in_the_prose(cells):
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    assert "31.4 %" in prose


def test_the_page_names_its_polarizations_ez_and_hz(cells):
    text = " ".join(_source(c) for c in cells)
    assert not re.search(r"\bTE\b", text)
    # the textbook's name appears once, to translate it
    assert len(re.findall(r"\bTM\b", text)) == 1


def test_the_page_offers_its_mesh_and_the_file_exists(cells):
    text = " ".join(_source(c) for c in cells)
    assert "{download}`data/rods.msh`" in text
    assert (DOCS / "tutorials" / "data" / "rods.msh").is_file()
    assert (DOCS / "tutorials" / "data" / "rods_fine.msh").is_file()
    for name in ("Palace", "Tidy3D", "Flexcompute", "GDSFactory"):
        assert name not in text, name
