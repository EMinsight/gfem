"""The emissivity recipe names public objects, and its Planck total is right.

docs/recipes/emissivity.md runs nothing at docs build time. Its code is read
without solving (blocks parse, `gfem.diffraction` names are public, keywords
bind), its `planck_total` block runs against the formula written here, and the
page is checked against quoting test evidence or another solver.
"""

import ast
import inspect
import re

import docs_pages as dp
import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

PAGE = dp.DOCS / "recipes" / "emissivity.md"

TOLERANCES = (pc.F1_COARSE_TOL, pc.F1_S_TOL, pc.F1_CROSS_TOL, pc.BREWSTER_TOL)
H, KB, C = 6.62607015e-34, 1.380649e-23, 299_792_458.0


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def planck_total(page):
    """The page's ``planck_total``, run from its own block."""
    (block,) = [b for b in dp.blocks(page, "python") if "def planck_total" in b]
    head = block.split("\n\n\ntotal = ")[0]  # the function, not its call
    namespace: dict = {}
    exec(compile(head, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace["planck_total"]


def reference(frequency, e_h, temperature):
    """The Planck-weighted emissivity, written with NumPy and no helper."""
    m = (
        2
        * np.pi
        * H
        * frequency**3
        / C**2
        / np.expm1(H * frequency / (KB * temperature))
    )
    return np.trapezoid(m * e_h, frequency) / np.trapezoid(m, frequency)


@pytest.mark.parametrize("temperature", [300.0, 600.0, 1500.0])
def test_the_planck_total_is_the_weighted_trapezoid(planck_total, temperature):
    nu = np.linspace(1e12, 2e14, 400)
    e_h = 0.3 + 0.2 * np.sin(nu / 3e13) ** 2
    got = float(planck_total(nu, e_h, temperature))
    assert got == pytest.approx(reference(nu, e_h, temperature), rel=1e-12)


def test_a_constant_emissivity_is_its_own_planck_total(planck_total):
    nu = np.linspace(1e12, 3e14, 300)
    assert float(planck_total(nu, np.full(nu.size, 0.37), 800.0)) == pytest.approx(
        0.37, rel=1e-12
    )


def test_the_peak_frequency_remark_is_wien(page):
    nu = np.linspace(1e10, 2e13, 2_000_000)
    m = nu**3 / np.expm1(H * nu / (KB * 100.0))
    peak = nu[np.argmax(m)] / 100.0
    assert peak == pytest.approx(5.88e10, rel=1e-3)
    assert "5.88" in page


def test_the_blocks_parse_and_bind_to_the_signatures(page):
    assert dp.blocks(page, "python")
    assert dp.static_problems(page) == []


def test_every_diffraction_name_on_the_page_is_public(page):
    names = set(re.findall(r"\b(?:dif|gfem\.diffraction)\.([A-Za-z_]\w*)", page))
    assert {"emissivity", "hemisphere", "hemispherical_emissivity"} <= names
    assert names <= set(dif.__all__)


def test_every_keyword_of_a_diffraction_call_is_a_parameter(page):
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


def test_the_page_states_the_reversed_azimuth(page):
    assert r"\varphi + \pi" in page


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
        "{term}`emissivity`",
        "{term}`hemispherical emissivity`",
        "{term}`absorptance`",
        "Kirchhoff",
        "mirror=True",
        "`cache=`",
        "{doc}`diffraction-efficiency`",
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
