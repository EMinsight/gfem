"""The transducer tutorial's committed run reports what its prose says.

The docs build executes no notebook, so the committed outputs are what the
site shows; the CI tutorials job re-executes the page. This reads the
committed notebook: every code cell ran without an error or a stderr line, the
printed numbers are physical (a positive radiation conductance peaking at the
SH0 speed, the same strain at the two arms in opposite phase, a capacitive
transducer), and the page quotes the paper's measured values only. It also
builds the page's section with its helper, docs/tutorials/transducer_section.py,
which the page imports and offers for download.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).parents[2] / "docs" / "tutorials" / "transducer.ipynb"
HELPER = NOTEBOOK.with_name("transducer_section.py")
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


def _line(cells, start: str) -> list[float]:
    (line,) = [line for line in _printed(cells).splitlines() if line.startswith(start)]
    return [float(n) for n in NUMBER.findall(line)]


def test_the_committed_run_executed_every_cell_cleanly(cells):
    code = _code(cells)
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
    streams = [o for c in code for o in c["outputs"] if o.get("name") == "stderr"]
    assert streams == [], "stderr in the committed outputs fails the docs build"


def test_the_radiation_conductance_is_positive_and_peaks_at_the_shear_plate_speed(
    cells,
):
    assert "positive at all 25 frequencies" in _printed(cells)
    f_peak, speed = _line(cells, "Re Y' peaks at")
    assert 0.855 < f_peak < 0.875  # GHz, at the 2.2 um pitch
    assert 3.7 < speed < 3.9  # km/s, v = 2 p f


def test_the_clamped_capacitance_is_the_susceptance_s_scale(cells):
    (c0,) = _line(cells, "C0' =")
    assert 600 < c0 < 800  # pF/m
    low, high = _line(cells, "Im Y' / (omega C0')")
    assert 0.5 < low < 1.5  # the page's loose bound
    assert 0.5 < high < 1.5


def test_the_effective_coupling_follows_its_definition(cells):
    f_s, f_p, k2 = _line(cells, "f_s =")
    assert f_s < f_p
    assert k2 == pytest.approx((f_p**2 - f_s**2) / f_p**2, abs=2e-3)


def test_the_arms_strain_the_same_in_opposite_phase(cells):
    _f, first, second = _line(cells, "|S_xz| at")
    assert first == pytest.approx(second, rel=1e-3)
    low, high = _line(cells, "phase between the arms")
    assert 179.0 < low <= high <= 180.0 + 1e-9


def test_the_short_transducer_is_a_capacitor(cells):
    (r_s,) = _line(cells, "R_s =")
    assert r_s == pytest.approx(26.7, abs=0.1)  # ohm, 400 um of 1.1 um x 50 nm Al
    *_, z_re, z_im, s11 = _line(cells, "at 0.8")
    assert z_re > 0 > z_im  # capacitive
    assert -1.0 < s11 < 0.0  # almost total reflection
    low, high = _line(cells, "S11 between")
    assert -1.0 < low < high < 0.0


def test_the_page_quotes_the_papers_measurement_and_no_other_source(cells):
    prose = " ".join(_source(c) for c in cells if c["cell_type"] == "markdown")
    assert "arXiv 2606.05337" in prose
    assert "\u221234.3 dB" in prose  # the measured dip, an observation only
    for name in ("Palace", "Flexcompute", "Tidy3D", "tidy3d"):
        assert name.lower() not in " ".join(_source(c) for c in cells).lower()


def test_the_page_offers_its_section_builder(cells):
    text = " ".join(_source(c) for c in cells)
    assert "from transducer_section import transducer_section" in text
    assert "{download}`transducer_section.py`" in text
    assert HELPER.is_file()


@pytest.fixture(scope="module")
def helper():
    sys.path.insert(0, str(HELPER.parent))
    try:
        import transducer_section
    finally:
        sys.path.remove(str(HELPER.parent))
    return transducer_section


def test_the_section_builder_names_what_the_page_reads(helper, tmp_path):
    section = helper.transducer_section(
        outer=2, between=3, edge_scale=3.0, directory=tmp_path
    )
    mesh = section.mesh
    assert mesh.dim == 2
    assert set(mesh.volumes) == {
        "air", "ln", "ln_frame", "box", "box_frame", "si", "si_frame",
        "signal", "ground",
    }  # fmt: skip
    assert set(mesh.surfaces) == {"bottom"}
    assert section.fingers == (3, 2)
    assert section.pitch == pytest.approx(2.2e-6)
    # two ribs, five pitches... here three apart, at the film's mid-depth
    assert section.arms.shape == (2, 2)
    assert np.diff(section.arms[:, 0]) == pytest.approx(3 * section.pitch)
    assert section.arms[:, 1] == pytest.approx(-0.09e-6)
    x_a, x_b = section.interior
    assert x_a == pytest.approx(17.6e-6)
    nodes = np.asarray(mesh.nodes)
    assert nodes[:, 0].min() == pytest.approx(0.0, abs=1e-12)
    assert nodes[:, 0].max() == pytest.approx(x_a + x_b)


def test_the_page_s_section_has_the_documented_fingers_and_a_pitch_of_2p2_um(helper):
    # 4 fingers beyond each rib and 5 between, signal first: 6 signal, 5 ground
    section = helper.transducer_section()
    assert section.fingers == (6, 5)
    assert section.arms[:, 0] == pytest.approx([29.7e-6, 40.7e-6])
    assert np.diff(section.arms[:, 0]) == pytest.approx(5 * section.pitch)


def test_line_electrodes_name_their_lines_and_no_ribs_leave_no_arms(helper, tmp_path):
    section = helper.transducer_section(
        outer=2, between=3, ribs=False, electrodes="lines", edge_scale=3.0,
        directory=tmp_path,
    )  # fmt: skip
    assert set(section.mesh.surfaces) == {"bottom", "signal", "ground"}
    assert "signal" not in section.mesh.volumes
    assert section.arms.shape == (0, 2)


def test_the_section_builder_refuses_what_it_cannot_build(helper):
    with pytest.raises(ValueError, match="electrodes"):
        helper.transducer_section(electrodes="foil")
    with pytest.raises(ValueError, match="at least 1"):
        helper.transducer_section(outer=0)
