"""The waveguide dispersion recipe runs and prints what the page shows.

docs/recipes/waveguide-dispersion.md runs nothing at docs build time. This runs
its blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: the group index and dispersion of a 500 x 220
nm silicon wire at 1.55 um, the two zero-dispersion widths, the clearance a
dispersion needs, and the slope of the group index.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "waveguide-dispersion.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("waveguide_dispersion")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_band_has_the_wires_group_index_and_anomalous_dispersion(ran):
    lines = _lines(ran, "Over a band")[1:6]
    rows = [dp.numbers(line) for line in lines]
    n_g_1550, d_1550 = rows[2][-2:]
    assert n_g_1550 == pytest.approx(4.196, abs=2e-3)
    assert d_1550 == pytest.approx(487, rel=0.02)
    assert all(row[-1] > 0 for row in rows)  # anomalous across the band


def test_constant_indices_lose_the_materials_dispersion(ran):
    n_g_frozen = dp.numbers(_lines(ran, "Material and waveguide dispersion")[0])[-2]
    n_g = dp.numbers(_lines(ran, "Over a band")[3])[-2]
    assert n_g_frozen < 0.98 * n_g


def test_the_zero_dispersion_widths_bracket_the_anomalous_window(ran):
    last = _lines(ran, "Widths of zero dispersion")[-1]
    first, second = dp.numbers(last)
    assert 400 < first < 450
    assert 600 < second < 650


def test_a_thin_cladding_reads_the_wrong_dispersion(ran):
    rows = [dp.numbers(line) for line in _lines(ran, "Check your results")[:3]]
    d_thin, d_one, d_more = (row[-1] for row in rows)
    assert d_thin < -50
    assert d_one == pytest.approx(d_more, abs=1.0)
    # the effective index is within 3e-4 where the dispersion is wrong
    assert abs(rows[0][-2] - rows[1][-2]) < 4e-4


def test_d_is_the_slope_of_the_group_index(ran):
    (slope,) = dp.numbers(_lines(ran, "Check your results", 1)[0])
    assert slope < 1e-2
