"""The diffraction-efficiency recipe names public objects and leaves test evidence out.

docs/recipes/diffraction-efficiency.md runs nothing at docs build time: its
meshes are the reader's own. This reads its code without solving: blocks parse,
`gfem.diffraction` names are public, keywords bind to the signatures. It also
checks the page against quoting test evidence or another solver.
"""

import ast
import re

import docs_pages as dp
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

PAGE = dp.DOCS / "recipes" / "diffraction-efficiency.md"

#: Every tolerance the Floquet and diffraction verification cases assert.
TOLERANCES = (pc.F1_COARSE_TOL, pc.F1_S_TOL, pc.F1_CROSS_TOL, pc.BREWSTER_TOL)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def test_the_blocks_parse_and_bind_to_the_signatures(page):
    assert dp.blocks(page, "python")
    assert dp.static_problems(page) == []


def test_every_diffraction_name_on_the_page_is_public(page):
    names = set(re.findall(r"\b(?:dif|gfem\.diffraction)\.([A-Za-z_]\w*)", page))
    assert {"floquet_ports", "efficiencies", "power_balance", "jones"} <= names
    assert "angle_sweep" in names
    assert names <= set(dif.__all__)


def test_every_keyword_of_a_diffraction_call_is_a_parameter(page):
    import inspect

    calls = 0
    for block in dp.blocks(page, "python"):
        for node in ast.walk(ast.parse(block)):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == "dif":
                parameters = inspect.signature(getattr(dif, node.func.attr)).parameters
                calls += 1
                assert {k.arg for k in node.keywords} <= set(parameters)
    assert calls >= 6


def test_the_page_quotes_no_case_id(page):
    assert not re.search(r"\*\*[CFR]\d\b", page)


@pytest.mark.parametrize("value", TOLERANCES)
def test_the_page_quotes_no_test_tolerance(page, value):
    mantissa, exponent = f"{value:.0e}".split("e")
    written = f"{mantissa}e{int(exponent)}"
    assert not re.search(rf"(?<![\d.e-]){re.escape(written)}(?!\d)", page)


def test_the_page_names_no_test_file_and_no_other_solver(page):
    lower = page.lower()
    assert "tests/" not in page
    for word in ("palace", "flexcompute", "tidy3d"):
        assert word not in lower


@pytest.mark.parametrize(
    "text",
    [
        "{term}`Jones matrix`",
        "{term}`absorptance`",
        "{term}`angle sweep`",
        "{term}`conical <Conical incidence>`",
        "retain_memory=0",
        "cache=",
        "NonDifferentiableError",
        "{doc}`emissivity`",
        "{doc}`floquet-ports`",
    ],
)
def test_the_page_carries(page, text):
    assert text in page or text in " ".join(page.split()), text


@pytest.mark.parametrize("heading", ["Check your results", "Limits", "Next"])
def test_the_page_has_its_closing_sections(page, heading):
    assert dp.section(page, heading).strip()


def test_the_names_the_page_calls_exist_in_gfem(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names - {"conventions", "diffraction"} <= set(gfem.__all__)
