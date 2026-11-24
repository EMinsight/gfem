"""The cross-sections recipe runs, and its words follow gfem.

docs/recipes/cross-sections.md runs nothing at docs build time. This runs
its executed blocks on the repository's own docs/tutorials/data/sphere.msh
and compares what they print with the page; checks its Mie series against
the tests' own; and ties each item of its Limits to the refusal tests of
tests/unit/test_cross_sections.py, so a new refusal without a page line
fails, and so does a page line without one.
"""

import ast
import re

import docs_pages as dp
import free_space_cases as fsc
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "cross-sections.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_cross_sections.py"
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "Incident-field columns only": ("test_a_port_column_has_no_cross_sections",),
    "Kept fields": ("test_cross_sections_need_kept_fields",),
    "One intensity per source": (
        "test_an_incident_field_needs_its_intensity",
        "test_a_plane_wave_takes_no_intensity",
    ),
    "A background truncation layer": (
        "test_a_scatterer_in_the_truncation_layer_is_refused",
    ),
    "Lossless backgrounds": ("test_a_lossy_background_is_not_measured",),
    "Wave ports outside the side": ("test_a_side_reaching_a_wave_port_is_refused",),
    "Power flow of an incident column": (
        "test_power_flow_refuses_the_incident_part",
        "test_power_flow_refuses_a_side_in_the_layer_of_an_incident_column",
    ),
}
#: What makes a test name a refusal.
REFUSING = re.compile(r"refuse|needs?_|takes_no|has_no|not_measured")
TERMS = (
    "Cross sections",
    "Incident intensity",
    "Truncation layer",
    "Scatterer",
    "Incident field",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("xsec_recipe"))


def test_the_page_downloads_the_sphere_mesh(page):
    first = dp.examples(page)[0]
    url = dp.MESH_URL.replace("line.msh", "sphere.msh")
    assert f'"{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


@pytest.mark.parametrize(
    ("eps", "x"), [(4 - 1j, 1.5), (4.0, 1.57), (2.25 - 0.1j, 3.0), (1.55**2, 5.2)]
)
def test_the_page_s_mie_series_is_the_tests_own(page, eps, x):
    (block,) = [b for b in dp.blocks(page, "python") if "def mie" in b]
    namespace = {"np": np}
    exec(compile(block.split("\n\n\np = ")[0], str(PAGE), "exec"), namespace)  # noqa: S102
    np.testing.assert_allclose(
        namespace["mie"](eps, x), fsc.mie(eps, x), rtol=1e-12, atol=1e-12
    )


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"cross_sections", "power_flow", "PlaneWave"} <= names
    assert names <= set(gfem.__all__)
    assert not dp.static_problems(page), dp.static_problems(page)


def test_the_page_quotes_no_test(page):
    assert "tests/" not in page
    assert "free_space_cases" not in page


@pytest.mark.parametrize("term", TERMS)
def test_the_page_links_each_term(page, term):
    linked = set()
    for role in re.findall(r"\{term\}`([^`]+)`", page):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        linked.add(" ".join((explicit[1] if explicit else role).split()).lower())
    assert term.lower() in linked


def _limits(page):
    return re.findall(r"^- \*\*([^*]+)\*\*", dp.section(page, "Limits"), re.MULTILINE)


def _refusal_tests():
    tree = ast.parse(REFUSALS.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def test_each_limit_on_the_page_has_its_refusal_test(page):
    assert sorted(_limits(page)) == sorted(LIMITS)
    tests = _refusal_tests()
    for limit, names in LIMITS.items():
        assert set(names) <= tests, (limit, sorted(set(names) - tests))


def test_each_refusal_test_has_its_limit_on_the_page():
    refusals = {name for name in _refusal_tests() if REFUSING.search(name)}
    mapped = {name for names in LIMITS.values() for name in names}
    assert sorted(refusals - mapped) == []
