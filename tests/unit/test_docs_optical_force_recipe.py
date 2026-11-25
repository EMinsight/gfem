"""The optical-force recipe runs, and its words follow gfem.

docs/recipes/optical-force.md runs nothing at docs build time. This runs its
executed blocks on the repository's own docs/tutorials/data/sphere.msh and
docs/recipes/slab_pair_section.py and compares what they print with the page;
and ties each item of its Limits to the refusal tests of
tests/unit/test_optical_force.py, so a new refusal without a page line fails,
and so does a page line without one. One section is not run: the eigenmode one
reads a cavity mesh the page does not ship (tests/unit/test_optical_force.py
runs that case).
"""

import ast
import re

import docs_pages as dp
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "optical-force.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_optical_force.py"
#: The sections whose blocks are not run, and why (see the module docstring).
NOT_RUN = ("Eigenmodes at 1 J",)
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "A 3D electric field": (
        "test_a_field_that_is_not_a_3d_time_harmonic_electric_field_is_refused",
        "test_a_2d_mesh_is_refused_by_name",
    ),
    "The field's own model": ("test_a_field_not_solved_on_the_model_is_refused",),
    "A body with a shell": (
        "test_a_body_that_selects_no_element_or_has_no_shell_is_refused",
    ),
    "One plain medium in the shell": (
        "test_a_shell_that_is_not_one_plain_medium_is_refused",
    ),
    "Nothing special beside the body": (
        "test_a_port_an_absorbing_surface_or_a_layer_beside_the_shell_is_refused",
        "test_a_body_node_on_a_periodic_pair_is_refused",
        "test_a_dipole_in_the_shell_is_refused",
    ),
    "Total fields only": (
        "test_the_layer_and_the_parts_of_an_incident_column_are_refused",
    ),
    "One process, no refinement": (
        "test_a_distributed_field_the_recorder_and_a_2d_mesh_are_refused",
    ),
    "No gradient through the geometry or the shell": (
        "test_the_node_coordinates_the_shell_material_and_a_reduced_field_raise_by_name",
    ),
    "Typed arguments": (
        "test_wrong_argument_types_raise_type_errors",
        "test_an_unknown_region_raises_by_name",
    ),
}
#: What makes a test name a refusal.
REFUSING = re.compile(r"refused|raise")
TERMS = ("Optical force", "Stress shell")


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("force_recipe")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir, skip=NOT_RUN)


def test_the_page_downloads_the_sphere_mesh_and_the_slab_section(page):
    code = "\n".join(e.code for e in dp.examples(page))
    for url in (
        dp.MESH_URL.replace("line.msh", "sphere.msh"),
        dp.MESH_URL.replace("tutorials/data/line.msh", "recipes/slab_pair_section.py"),
    ):
        assert f'"{url}"' in code or url in code
        assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"optical_force", "power_flow", "PortMode", "MaterialField"} <= names
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
