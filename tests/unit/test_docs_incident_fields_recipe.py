"""The incident-fields recipe runs, and its words follow gfem.

docs/recipes/incident-fields.md runs nothing at docs build time. This runs
its executed blocks on the repository's own docs/tutorials/data/sphere.msh
and compares what they print with the page; checks its quoted contact limit,
its angle convention and public names; and ties each item of its Limits to
the refusal tests that hold it, so a new refusal without a page line fails,
and so does a page line without one.
"""

import ast
import re
import shutil
import urllib.request

import docs_pages as dp
import numpy as np
import pytest

import gfem
from gfem._engine import incident

PAGE = dp.DOCS / "recipes" / "incident-fields.md"
REFUSALS = (
    dp.ROOT / "tests" / "unit" / "test_incident_solve.py",
    dp.ROOT / "tests" / "unit" / "test_driven_result_sources.py",
    dp.ROOT / "tests" / "unit" / "test_layered_solve.py",
)
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "A radiating truncation": (
        "test_a_closed_model_is_refused",
        "test_an_exterior_lossy_or_lumped_face_is_refused",
    ),
    "Scatterers clear of the truncation": (
        "test_the_incident_contact_above_its_limit_is_refused_with_its_ratio",
        "test_a_scatterer_on_an_absorbing_face_is_refused",
    ),
    "Walls the incident field satisfies": (
        "test_a_pec_wall_the_incident_field_crosses_is_refused",
        "test_a_default_pmc_wall_the_incident_field_crosses_is_refused",
    ),
    "Ports outside the incident field": (
        "test_an_incident_field_on_a_wave_port_face_is_refused",
    ),
    "Lossless incidence": (
        "test_sources_this_version_does_not_solve_are_refused",
        "test_a_lossy_incidence_layer_is_refused",
        "test_a_gaussian_beam_on_several_layers_is_refused",
    ),
    "Stacks matched to the mesh": (
        "test_a_tetrahedron_across_a_stack_plane_is_refused",
        "test_grazing_incidence_on_a_stack_is_refused",
        "test_incidence_from_below_a_ground_plane_is_refused",
        "test_a_ground_plane_needs_a_pec_face_on_it",
        "test_a_ground_plane_off_the_mesh_bottom_is_refused",
        "test_elements_below_the_ground_plane_are_refused",
        "test_a_layer_region_off_the_stack_is_refused_by_its_contact",
    ),
    "One process per solve": (
        "test_frequency_workers_do_not_take_incident_fields_yet",
    ),
    "Material gradients": (
        "test_node_coordinates_are_not_differentiable_with_an_incident_field",
        "test_a_design_touching_the_layer_is_not_differentiable",
    ),
    "Functions are not written": (
        "test_a_result_of_an_incident_function_cannot_be_written",
        "test_an_incident_field_cannot_be_written",
        "test_a_job_with_an_incident_field_cannot_be_written",
    ),
}
#: What makes a test name a refusal.
REFUSING = re.compile(r"refused|not_differentiable|cannot_be_written|do_not_take")
TERMS = (
    "Incident field",
    "Source",
    "Background stack",
    "Total field",
    "Scattered field",
    "Scatterer",
    "Incident contact",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("incident_recipe"))


def test_the_page_downloads_the_sphere_mesh(page):
    first = dp.examples(page)[0]
    url = dp.MESH_URL.replace("line.msh", "sphere.msh")
    assert f'"{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_absorbed_power_is_the_cross_section_s(ran, tmp_path, monkeypatch):
    def urlretrieve(url, filename):
        shutil.copy(dp.MESHES[url], tmp_path / filename)

    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for example in ran:
        exec(compile(example.code, str(PAGE), "exec"), namespace)  # noqa: S102
    xs = gfem.cross_sections(namespace["result"], "te", model=namespace["model"])
    absorbed = float(xs.absorption[0] * xs.intensity[0])
    assert float(namespace["absorbed"]) == pytest.approx(absorbed, rel=1e-10)


def test_the_page_s_function_is_its_plane_wave(page):
    (block,) = [b for b in dp.blocks(page, "python") if "def te_wave" in b]
    namespace = {"np": np, "gfem": gfem, "theta": np.radians(30)}
    exec(compile(block, str(PAGE), "exec"), namespace)  # noqa: S102
    wave = gfem.PlaneWave("te", theta=np.radians(30), phi=0.0, polarization="TE")
    points = np.random.default_rng(1).uniform(-1e-6, 1e-6, (7, 3))
    background = gfem.Background([gfem.Material()])
    f = 3e14
    np.testing.assert_allclose(
        namespace["te_wave"](points, f),
        incident._incident.values(wave, background, points, f),
        rtol=1e-12,
    )


def test_the_page_states_the_contact_limit(page):
    exponent = round(np.log10(incident.CONTACT))
    assert 10.0**exponent == incident.CONTACT
    assert f"$10^{{{exponent}}}$ of its peak" in " ".join(page.split())


def test_the_page_states_the_angle_convention(page):
    k_hat = gfem._sources.propagation_basis(np.pi - np.radians(30), 0.0)[0]
    assert k_hat[2] < 0  # coming down
    assert "`theta=np.pi - np.radians(30)`" in page


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"PlaneWave", "IncidentField", "power_flow", "Background"} <= names
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


def _tests():
    names = set()
    for path in REFUSALS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names |= {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
    return names


def test_each_limit_on_the_page_has_its_refusal_test(page):
    assert sorted(_limits(page)) == sorted(LIMITS)
    tests = _tests()
    for limit, names in LIMITS.items():
        assert set(names) <= tests, (limit, sorted(set(names) - tests))


def test_each_refusal_test_has_its_limit_on_the_page():
    refusals = set()
    for path in (REFUSALS[0], REFUSALS[2]):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        refusals |= {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and REFUSING.search(node.name)
        }
    mapped = {name for names in LIMITS.values() for name in names}
    assert sorted(refusals - mapped) == []
