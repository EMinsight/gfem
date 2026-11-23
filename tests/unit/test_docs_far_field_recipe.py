r"""The far-field recipe runs, and its words follow gfem.

docs/recipes/far-field.md runs nothing at docs build time. This runs its
executed blocks on the repository's own docs/tutorials/data/antenna_box.msh
and checks what they print against the page and against the physics: the
intensity integrates to the radiated power, the realized gain is the
directivity times the radiated power, two crossed dipoles in quadrature radiate
circular polarization, and a strip lit face-on scatters back. It checks the page's
public names and glossary links, and ties each item of its Limits to the
refusal tests that hold it.
"""

import re

import docs_pages as dp
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "far-field.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_far_field.py"
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "A radiating truncation": ("test_far_field_refuses_a_closed_model",),
    "One lossless exterior medium": (
        "test_far_field_needs_the_background",
        "test_far_field_refuses_a_layered_or_lossy_exterior",
        "test_far_field_refuses_a_layer_that_is_not_the_exterior_medium",
        "test_far_field_refuses_a_design_in_its_layer",
        "test_far_field_refuses_a_side_whose_layer_is_the_pml",
    ),
    "Every source inside the side": (
        "test_far_field_refuses_a_source_outside_the_side",
        "test_far_field_refuses_a_source_in_its_layer",
        "test_far_field_refuses_a_port_in_its_layer",
    ),
    "Ground planes flat, PEC or PMC": (
        "test_a_direction_below_the_ground_plane_is_refused",
    ),
    "No node gradient of the radiated power": (),
}
#: Refusal tests of the measurement's input, not limits of the transform.
INPUT_CHECKS = ("test_far_field_needs_a_kept_field_of_a_column",)
TERMS = (
    "Far-field amplitude",
    "Far-field layer",
    "Exterior medium",
    "Radiation intensity",
    "Directivity",
    "Ground plane",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("far_field_recipe"))


def test_the_page_downloads_the_antenna_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    url = dp.MESH_URL.replace("line.msh", "antenna_box.msh")
    assert f'url = "{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def _printed(ran, section):
    (example,) = [e for e in ran if e.section == section]
    return dp.numbers(example.printed)


def test_the_strip_radiates_like_a_dipole(ran):
    radiated, accepted, total, directivity, realized = _printed(
        ran, "Read the pattern of an antenna"
    )
    assert total == pytest.approx(radiated, rel=1e-2)
    assert accepted == pytest.approx(radiated, rel=5e-2)
    # D = 4 pi U / P_rad and G_r = 4 pi U / (1 W)
    assert realized == pytest.approx(directivity * radiated, rel=1e-2)


def test_the_quadrature_pair_is_nearly_circular(ran):
    (ratio,) = _printed(ran, "Circular polarization")
    assert ratio < 0.1


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"far_field", "FarField", "PlaneWave"} <= names
    assert names - {"shape", "conventions"} <= set(gfem.__all__)


def test_the_page_names_each_far_field_attribute(page):
    for name in (
        "e_theta",
        "e_phi",
        "intensity",
        "radiated_power",
        "directivity",
        "gain",
        "realized_gain",
        "radiation_efficiency",
        "e_right",
        "e_left",
        "axial_ratio",
        "rcs",
    ):
        assert hasattr(gfem.FarField, name) or name in {
            f.name for f in gfem.FarField.__dataclass_fields__.values()
        }
        assert f"`{name}`" in page, name


def test_the_page_passes_the_static_check(page):
    assert not dp.static_problems(page), dp.static_problems(page)


@pytest.mark.parametrize("term", TERMS)
def test_the_page_links_each_far_field_term(page, term):
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
    pattern = r"^def (test_\w*(?:refuse|needs)\w*)"
    tests = set(re.findall(pattern, text, re.MULTILINE))
    tied = {t for names in LIMITS.values() for t in names}
    assert tied <= tests | set(INPUT_CHECKS), tied - tests
    assert tests - tied == set(INPUT_CHECKS), tests - tied


def test_the_lit_strip_scatters_back(ran):
    rcs = _printed(ran, "Radar cross section")[0]  # then the 2 of um^2
    assert rcs > 0.01  # um^2
