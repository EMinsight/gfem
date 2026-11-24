r"""The perfectly matched layer recipe runs, and its words follow gfem.

docs/recipes/perfectly-matched-layer.md runs nothing at docs build time.
This runs its executed blocks on the repository's own
docs/tutorials/data/pml_line.msh and compares each printed :math:`|S_{11}|`
with the hand formula :math:`R_0^{f/f_\text{ref}}`; checks its profile
formulas against gfem's own, its quoted default, warning and public names;
and ties each item of its Limits to the refusal test that holds it, so a new
refusal without a page line fails, and so does a page line without one.
"""

import ast
import math
import re
import types
import warnings

import docs_pages as dp
import pml_cases as pc
import pytest

import gfem
from gfem import _floquet, _pml

PAGE = dp.DOCS / "recipes" / "perfectly-matched-layer.md"
REFUSALS = dp.ROOT / "tests" / "unit" / "test_pml.py"
#: The sections whose code needs the reader's own mesh: read, never run.
STATIC = (
    "Oblique incidence on a periodic cell",
    "Wave ports through side layers",
    "Radiation Q of a resonator",
)
#: The target reflection the page's line is solved with.
TARGET = 1e-2
#: How far a printed |S11| may sit from its formula: the line's reflection
#: floor. Measured 4.4e-7 at f0 and 4.5e-6 at 1.5 f0 (lambda / 6.7 elements).
LINE_FLOOR = 1e-5
#: Gates of the layer's verification cases (the floors that are not also a
#: physical value the page states: the default R0 and its example target).
TOLERANCES = (pc.GUIDE_FLOOR, pc.CELL_FLOOR, pc.GUIDE_MOVE)
#: Each item of the page's Limits (its bold lead) -> the tests that refuse it.
LIMITS = {
    "No error estimates or mesh refinement for wave problems": (
        "test_driven_error_estimates_refuse_the_layer",
        "test_eigenmode_error_estimates_refuse_the_layer",
        "test_refine_mesh_refuses_the_layer_before_any_solve",
    ),
    "Box layers only": (
        "test_a_layer_cell_inside_the_interior_box_is_refused",
        "test_a_layer_that_straddles_a_face_of_the_interior_box_is_refused",
    ),
    "One thickness per side": (
        "test_two_layer_regions_ending_on_two_planes_of_one_side_are_refused",
        "test_one_layer_region_with_a_stepped_outer_face_is_refused",
    ),
    "Constant backgrounds": (
        "test_a_layer_background_must_be_constant_and_axis_aligned",
    ),
    "No absorber or sheet on the layer": (
        "test_absorbing_on_a_face_of_the_layer_is_refused",
        "test_a_sheet_in_the_layer_is_refused",
    ),
    "Ports clear of the layer": (
        "test_a_lumped_port_touching_the_layer_is_refused",
        "test_a_floquet_face_touching_the_layer_is_refused",
        "test_a_wave_port_normal_along_a_stretch_axis_is_refused",
        "test_a_wave_port_oblique_in_a_layer_is_refused",
    ),
    "No stretch along a periodic axis": (
        "test_a_stretch_along_a_periodic_axis_is_refused",
    ),
    "Paths and surfaces in the physical part": (
        "test_an_impedance_path_through_the_layer_is_refused",
        "test_interface_participation_refuses_a_surface_all_in_the_layer",
    ),
    "Moved meshes": (
        "test_a_moved_mesh_pushing_air_past_a_stretched_face_is_refused",
        "test_a_moved_mesh_pushing_air_past_a_stretched_face_is_refused_under_grad",
        "test_boundary_morph_of_a_surface_in_the_layer_is_refused",
        "test_boundary_morph_of_a_surface_in_the_truncation_layer_is_refused",
    ),
    "No iterative solver": ("test_an_explicit_bddc_gmres_refuses_the_layer",),
    "Thin layers": ("test_a_thin_layer_warns",),
}
#: Refusal tests of input mistakes, not limits of the layer: no page line.
INPUT_CHECKS = (
    "test_a_layer_refuses_bad_values",
    "test_a_layer_region_excluded_from_electromagnetic_solves_is_refused",
)
#: The user glossary terms the page links.
TERMS = (
    "Perfectly matched layer",
    "Target reflection",
    "Reference frequency",
    "Interior box",
    "Physical region",
    "PML mode",
    "Clearance",
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("pml_recipe"), skip=STATIC)


def test_the_page_downloads_the_layer_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    url = dp.MESH_URL.replace("line.msh", "pml_line.msh")
    assert f'url = "{url}"' in first.code
    assert dp.MESHES[url].is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_line_reflects_its_target_reflection(ran):
    (line,) = [e for e in ran if e.section == "Close a line with a layer"]
    rows = [dp.numbers(row) for row in line.printed.splitlines()]
    assert [row[0] for row in rows] == [1.0, 1.5]
    for ratio, s11, formula in rows:
        expected = _pml.reflection(TARGET, ratio, 1.0, 1.0)
        assert formula == pytest.approx(expected, rel=1e-3)
        assert abs(s11 - expected) <= LINE_FLOOR, (ratio, s11, expected)


@pytest.fixture(scope="module")
def profile(page):
    """The page's ``reflection`` and ``stretch_maximum``, run from its own block."""
    (block,) = [b for b in dp.blocks(page, "python") if "def reflection" in b]
    namespace = {}
    exec(compile(block.split("\n\n\nprint")[0], str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    ("target", "f", "cos_theta"),
    [(1e-6, 1.0, 1.0), (1e-3, 1.5, 0.36), (1e-8, 0.7, 0.8)],
)
def test_the_page_formulas_are_the_layer_s_own(profile, target, f, cos_theta):
    f_ref, thickness = 2e14, 0.6e-6
    k_ref = 2 * math.pi * f_ref / _pml.C0
    assert profile["reflection"](target, f * f_ref, f_ref, cos_theta) == (
        pytest.approx(_pml.reflection(target, f, 1.0, cos_theta), rel=1e-12)
    )
    assert profile["stretch_maximum"](target, f_ref, thickness) == pytest.approx(
        _pml.stretch_maximum(target, k_ref, thickness), rel=1e-12
    )


def test_the_page_states_the_default_target_reflection(page):
    default = gfem.PML("pml").target_reflection
    exponent = round(math.log10(default))
    assert default == 10.0**exponent
    assert f"`target_reflection` is $R_0 = 10^{{{exponent}}}$" in " ".join(page.split())


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert {"PML", "LayerClearanceWarning", "ObliqueAbsorberWarning"} <= names
    assert names - {"shape", "refine_mesh"} <= set(gfem.__all__)


def test_the_page_quotes_no_test(page):
    assert "tests/" not in page
    assert "pml_cases" not in page
    for value in TOLERANCES:
        mantissa, exponent = f"{value:.0e}".split("e")
        written = f"{mantissa}e{int(exponent)}"
        assert not re.search(rf"(?<![\d.e-]){re.escape(written)}(?!\d)", page), value
        assert f"{mantissa}\\times10^{{{int(exponent)}}}" not in page, value


@pytest.mark.parametrize("term", TERMS)
def test_the_page_links_each_layer_term(page, term):
    linked = set()
    for role in re.findall(r"\{term\}`([^`]+)`", page):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        linked.add((explicit[1] if explicit else role).strip().lower())
    assert term.lower() in linked


def test_the_quoted_remedy_is_gfem_s_own_warning(page, monkeypatch):
    section = dp.section(page, "Oblique incidence on a periodic cell")
    (quoted,) = [b for b in dp.blocks(section, "text") if b.startswith("Absorbing")]
    # the page's numbers: 60 degrees, the specular order, a 1 um wavelength
    monkeypatch.setattr(
        _floquet,
        "_worst_incidence",
        lambda *_: (math.radians(60.0), (0, 0), pc.F0),
    )
    model = types.SimpleNamespace(
        _periodic=[object()], boundaries=[gfem.Absorbing("bottom")]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _floquet.warn_oblique_absorbers(model, [pc.F0], None, stacklevel=1)
    (warning,) = caught
    assert warning.category is gfem.ObliqueAbsorberWarning
    assert " ".join(quoted.split()) in str(warning.message)


def _limits(page: str) -> list[str]:
    return re.findall(r"^- \*\*([^*]+)\*\*", dp.section(page, "Limits"), re.MULTILINE)


def _refusal_tests() -> set[str]:
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
    mapped = {name for names in LIMITS.values() for name in names}
    refusals = {
        name for name in _refusal_tests() if "refuse" in name or name.endswith("_warns")
    }
    # the gain-sign warning is a symptom the page explains, not a refusal
    refusals.discard("test_a_mode_with_gain_sign_warns")
    assert sorted(refusals - mapped - set(INPUT_CHECKS)) == []
    assert set(INPUT_CHECKS) <= refusals
