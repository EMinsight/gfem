r"""The point-dipole recipe runs, and its words follow gfem.

docs/recipes/point-dipoles.md runs nothing at docs build time. This runs its
executed blocks on the repository's own docs/tutorials/data/pml_line.msh and
checks what they print against the page and against the physics: a dipole
in the middle of a TEM line sends half its power each way (beta = 1/2), the
power split adds up, and the best orientation is along the line's field.
It checks the page's bulk power formulas against gfem's, its public names
and glossary links, and ties each item of its Limits to the refusal tests
that hold it, so a new refusal without a page line fails, and so does a page
line without one.
"""

import math
import re

import docs_pages as dp
import numpy as np
import pytest

import gfem
from gfem import _purcell

PAGE = dp.DOCS / "recipes" / "point-dipoles.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_dipoles.py"
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "A position inside one lossless, isotropic material": (
        "test_a_dipole_refuses_a_position_without_a_bulk_power",
        "test_a_dipole_refuses_a_material_field",
        "test_a_dipole_refuses_an_excluded_volume",
        "test_a_magnetic_dipole_refuses_like_an_electric_one",
    ),
    "No boundary or sheet face": (
        "test_a_dipole_refuses_a_boundary_or_sheet_face",
        "test_a_dipole_refuses_a_port_face",
        "test_a_dipole_refuses_the_boundary_of_the_solve_domain",
    ),
    "Node gradients of electric dipoles only": (
        "test_a_magnetic_dipole_refuses_node_gradients",
    ),
    "A power side clear of the dipole": (
        "test_power_flow_refuses_a_side_whose_element_layer_holds_a_dipole",
    ),
    "Port waves alone under `source_fields=False`": (
        "test_source_fields_false_refuses_an_incident_field",
        "test_source_fields_false_needs_a_port_with_a_defined_wave",
        "test_source_fields_false_without_an_excited_port_refuses_keep_fields",
        "test_a_dipole_without_its_field_refuses_purcell_e_field_and_far_field",
    ),
    "Point dipoles only": (),
}
#: Refusal tests of the measurement's input, not limits of the dipole.
INPUT_CHECKS = (
    "test_the_purcell_factor_needs_a_dipole",
    "test_the_purcell_factor_refuses_an_incident_field",
)
#: The user glossary terms the page links.
TERMS = (
    "Point dipole",
    "Source",
    "Emitted power",
    "Bulk power",
    "Purcell factor",
    "Beta factor",
    "Radiative efficiency",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("dipole_recipe"))


def test_the_page_downloads_the_layer_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    url = dp.MESH_URL.replace("line.msh", "pml_line.msh")
    assert f'url = "{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def _printed(ran, section):
    (example,) = [e for e in ran if e.section == section]
    return dp.numbers(example.printed)


def test_the_line_takes_half_the_power_each_way(ran):
    power, _, guided, beta = _printed(ran, "Place a dipole")
    assert beta == pytest.approx(0.5, abs=1e-3)
    assert guided == pytest.approx(power / 2, rel=1e-3)


def test_the_power_split_adds_up(ran):
    numbers = _printed(ran, "Where the power goes")
    assert numbers[-2] == pytest.approx(1.0, abs=1e-7)  # sum / P
    assert numbers[-1] == 1.0  # lossless: every watt leaves


def test_the_best_orientation_is_along_the_line_field(ran):
    numbers = _printed(ran, "The best orientation")
    np.testing.assert_allclose(numbers[:3], [0.0, 1.0, 0.0], atol=1e-3)
    power = _printed(ran, "Place a dipole")[0]
    assert numbers[3] == pytest.approx(power, rel=1e-3)


def test_the_page_formulas_are_gfem_s_bulk_power(page):
    # the page's P0 for each kind, written out, against gfem's
    omega, size = 2 * math.pi * 3e14, 1e-30
    mu0, c = 4e-7 * math.pi, 299_792_458.0
    eps_r, mu_r = 2.25, 1.0
    n = math.sqrt(eps_r * mu_r)
    electric = gfem.ElectricDipole("p", position=(0, 0, 0), moment=(0, 0, size))
    magnetic = gfem.MagneticDipole("m", position=(0, 0, 0), moment=(0, 0, size))
    written = {
        electric: mu0 * mu_r * n * omega**4 * size**2 / (12 * math.pi * c),
        magnetic: mu0 * eps_r * n * omega**4 * size**2 / (12 * math.pi * c**3),
    }
    assert r"\frac{\mu_0 \mu_r n\, \omega^4 |p|^2}{12 \pi c}" in page
    assert r"\frac{\mu_0 \varepsilon_r n\, \omega^4 |m|^2}{12 \pi c^3}" in page
    for dipole, value in written.items():
        got = float(_purcell.bulk_power(dipole, omega, eps_r=eps_r, mu_r=mu_r))
        assert got == pytest.approx(value, rel=1e-8)


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"ElectricDipole", "MagneticDipole", "purcell_factor"} <= names
    assert names - {"shape", "conventions"} <= set(gfem.__all__)


def test_the_page_passes_the_static_check(page):
    assert not dp.static_problems(page), dp.static_problems(page)


@pytest.mark.parametrize("term", TERMS)
def test_the_page_links_each_dipole_term(page, term):
    linked = set()
    for role in re.findall(r"\{term\}`([^`]+)`", page):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        linked.add(" ".join((explicit[1] if explicit else role).split()).lower())
    assert term.lower() in linked


def _limits(page):
    section = dp.section(page, "Limits")
    return re.findall(r"^- \*\*(.+?)\*\*", section, re.MULTILINE)


def test_the_limits_are_the_ones_listed_here(page):
    assert _limits(page) == list(LIMITS)


def test_every_refusal_test_has_a_limit_and_every_limit_its_tests():
    text = REFUSALS.read_text(encoding="utf-8")
    tests = set(re.findall(r"^def (test_\w*(?:refuse|needs)\w*)", text, re.MULTILINE))
    tied = {t for names in LIMITS.values() for t in names}
    assert tied <= tests | set(INPUT_CHECKS), tied - tests
    assert tests - tied == set(INPUT_CHECKS), tests - tied
