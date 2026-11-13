"""The conventions page shows every convention of `gfem.conventions` on a solve.

docs/conventions.md runs nothing at docs build time. This checks that the
page names every constant of the module (so a constant added there without a
line on the page fails here) with its value, and runs the page's blocks on
the repository's own docs/tutorials/data/line.msh: what they print is what
the page shows, and is what each convention says it must be.
"""

import docs_pages as dp
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "conventions.md"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("conventions"))


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def _table(page: str) -> dict[str, str]:
    """The page's table of constants: name -> the text of its value cell."""
    rows = {}
    for line in page.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("`gfem.conventions."):
            rows[cells[0].strip("`").removeprefix("gfem.conventions.")] = cells[1]
    return rows


@pytest.mark.parametrize("name", gfem.conventions.__all__)
def test_the_page_states_every_constant_with_its_value(page, name):
    value = getattr(gfem.conventions, name)
    table = _table(page)
    assert name in table, f"no row for gfem.conventions.{name} in the page's table"
    if isinstance(value, str):
        assert table[name] == f"`{value}`"
    else:
        (shown,) = dp.numbers(table[name])
        assert shown == value
    # Beyond the table, a section says what the constant means on a solve.
    assert f"`{name}`" in dp._FENCE.sub("", page.split("## Units")[1])


def test_the_table_has_no_constant_the_module_lacks(page):
    assert set(_table(page)) == set(gfem.conventions.__all__)


def test_the_page_links_the_module(page):
    assert "{py:obj}`gfem.conventions`" in page


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_mesh_coordinates_are_in_metres(ran):
    (example,) = [example for example in ran if example.section == "Units"]
    assert example.printed.splitlines()[0] == "mm"
    assert dp.numbers(example.printed) == pytest.approx([40e-3, 6e-3, 1e-3])


def test_the_phase_of_a_transmission_falls_with_frequency(ran):
    phase = _printed(ran, "Time dependence")
    assert len(phase) == 7
    assert np.all(np.diff(phase) < 0)
    assert phase[[0, -1]] == pytest.approx([3.05, -7.83], abs=5e-3)


def test_a_driven_field_carries_what_the_port_does_not_reflect(ran):
    through, unreflected = _printed(ran, "S-parameters and power")
    # 1 W incident: the transmitted fraction of the matched-line formula.
    assert unreflected == pytest.approx(dp.line_formula(5e9) ** 2, abs=1e-4)
    assert through == pytest.approx(unreflected, abs=2e-6)


def test_a_decaying_mode_has_a_positive_imaginary_frequency(ran):
    (example,) = [example for example in ran if example.section == "Sign of loss"]
    assert example.printed.split()[1] == "True"
    frequency, q, from_omega = dp.numbers(example.printed)
    assert frequency == pytest.approx(2.14197, abs=1e-5)
    assert q == pytest.approx(1251.7, abs=0.05)
    assert from_omega == pytest.approx(q, rel=1e-12)


def test_a_mode_field_stores_the_mode_energy(ran):
    energy, shares = _printed(ran, "Eigenmode fields")
    assert energy == pytest.approx(gfem.conventions.MODE_ENERGY, rel=1e-9)
    assert shares == pytest.approx(1.0, rel=1e-9)


def test_a_lossy_law_has_a_negative_imaginary_part_and_a_static_value(ran):
    (example,) = [e for e in ran if e.section == "Dispersive materials"]
    real, imag, static = dp.numbers(example.printed)
    # 2 + 1 * f0^2 / (f0^2 - f^2 + i gamma f) at f = 9, f0 = 10, gamma = 1 GHz
    assert complex(real, imag) == pytest.approx(2 + 100 / (19 + 9j), rel=1e-12)
    assert imag < 0
    assert static == 3.0


def test_the_dispersive_section_names_the_two_derived_permittivities(page):
    section = dp._prose(dp.section(page, "Dispersive materials"))
    assert "{term}`static permittivity`" in section
    assert "{term}`energy permittivity`" in section
    assert "{doc}`recipes/dispersive-materials`" in section
