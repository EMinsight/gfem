"""The multipoles recipe runs, and its words follow gfem.

docs/recipes/multipoles.md runs nothing at docs build time. This runs its
executed blocks on the repository's own docs/tutorials/data/sphere.msh with
one solve and compares what they print with the page; checks its Mie series
against the tests' own; keeps the readings it states (the electric and
magnetic dipoles nearly equal, the multipoles' scattering next to the cross
sections') within bounds; and ties each item of its Limits to the refusal
tests of tests/unit/test_multipoles.py, so a new refusal without a page line
fails, and so does a page line without one.
"""

import ast
import re

import docs_pages as dp
import multipole_cases as mc
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "multipoles.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_multipoles.py"
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "Incident-field columns only": (
        "test_u7_checks_are_in_order_and_refusals_are_named",
        "test_u7_a_column_without_kept_fields_or_incident_part",
        "test_u7_current_moments_checks_are_in_order_and_refusals_are_named",
        "test_u7_current_moments_has_no_intensity_and_reads_a_field_without_incident_part",
    ),
    "A homogeneous lossless background": (
        "test_u7_checks_are_in_order_and_refusals_are_named",
        "test_u7_walls_are_refused_before_the_far_field_layer",
        "test_u7_current_moments_walls_truncation_layer_and_wave_port",
    ),
    "3D, a truncation layer and the degree cap": (
        "test_u7_checks_are_in_order_and_refusals_are_named",
        "test_u7_a_scatterer_in_the_truncation_layer_or_at_a_port",
        "test_u7_current_moments_walls_truncation_layer_and_wave_port",
        "test_u12_lmax_and_its_bounds",
    ),
    "Volume currents only": (
        "test_u13_current_moments_refuses_a_surface_current_that_multipoles_accepts",
        "test_u13_current_moments_refuses_a_permeability_contrast",
    ),
    "Not differentiable": ("test_current_moments_is_refused_under_jax_transforms",),
}
#: What makes a test of the file a refusal test.
REFUSING = re.compile(r"^test_u(7|12|13)_|refused_under")
TERMS = (
    "Multipole decomposition",
    "Multipole coefficients",
    "Multipole degree",
    "Multipole origin",
    "Incident amplitude",
    "Partial scattering cross section",
    "Exact multipole moments",
    "Induced current",
    "Current moments",
    "Long-wavelength dipole",
    "Toroidal dipole",
    "Kerker condition",
    "Anapole",
    "Scatterer",
    "Incident field",
    "Incident intensity",
    "Distributed solve",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("multipoles_recipe"))


def _printed(ran, heading):
    return dp.numbers(next(e.printed for e in ran if e.section == heading))


def test_the_page_downloads_the_sphere_mesh(page):
    first = dp.examples(page)[0]
    url = dp.MESH_URL.replace("line.msh", "sphere.msh")
    assert f'"{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


@pytest.mark.parametrize(
    ("eps", "x"), [(12.25 + 0j, 0.78), (4.0 - 1.0j, 1.5), (2.25, 3.0)]
)
def test_the_page_s_mie_series_is_the_tests_own(page, eps, x):
    block = next(b for b in dp.blocks(page, "python") if "def mie" in b)
    namespace = {"np": np}
    exec(  # noqa: S102
        compile(
            "from scipy.special import spherical_jn, spherical_yn\n"
            + block.split("\n\nk = ")[0].split("\n\n\n", 1)[1],
            str(PAGE),
            "exec",
        ),
        namespace,
    )
    a, b = namespace["mie"](eps, x, 5)
    ref_a, ref_b = mc.mie_coefficients(eps, x, 5)
    np.testing.assert_allclose(np.conj(a), ref_a, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.conj(b), ref_b, rtol=1e-12, atol=1e-14)


def test_the_two_dipoles_are_equal_at_the_kerker_frequency(ran):
    assert abs(_printed(ran, "Partial cross sections")[-1] - 1) < 0.05
    kerker = next(e.printed for e in ran if e.section == "Kerker and Huygens")
    ratio = complex(re.search(r"p_x = (\S+j)$", kerker.splitlines()[1])[1])
    assert abs(ratio - 1) < 0.05


def test_the_multipoles_scattering_is_next_to_the_cross_sections(ran):
    gap = _printed(ran, "Check against the cross sections")[-1]
    assert abs(gap) < 1.0


def test_the_toroidal_correction_gives_the_exact_dipole(ran):
    assert _printed(ran, "Anapoles: two calls")[-1] < 0.01


def test_the_origin_moves_the_shares_and_not_their_sum(ran):
    text = next(e.printed for e in ran if e.section == "Origin and degree cap")
    moved, sums, tail = text.strip().splitlines()
    ed, md = (float(x) for x in re.findall(r"(?:ED|MD) ([\d.]+)", moved))
    assert abs(ed / md - 1) > 0.05
    here, there = (float(x) for x in re.findall(r"([\d.]+) um\^2", sums))
    assert here == pytest.approx(there, rel=1e-3)
    ten, seven = (float(x) for x in re.findall(r"(\d\.\d+) um", tail))
    assert ten == pytest.approx(seven, rel=1e-3)


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"multipoles", "current_moments", "cross_sections", "PlaneWave"} <= names
    assert names <= set(gfem.__all__) | {"conventions"}
    assert not dp.static_problems(page), dp.static_problems(page)


def test_the_page_quotes_no_test_or_local_note(page):
    assert "tests/" not in page
    assert "multipole_cases" not in page
    assert ".scratch" not in page
    assert not re.search(r"\bADR\b", page)


@pytest.mark.parametrize("term", TERMS)
def test_the_page_links_each_term(page, term):
    linked = set()
    for role in re.findall(r"\{term\}`([^`]+)`", page):
        explicit = re.search(r"<([^>]+)>\s*$", " ".join(role.split()))
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
