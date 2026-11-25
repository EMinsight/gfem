"""The metasurface recipe names public objects and its closed form is right.

docs/recipes/metasurface-from-cells.md runs nothing at docs build time: its
meshes are the reader's own. This reads its code without solving: blocks parse,
`gfem.diffraction` names are public, keywords bind to the signatures. The page's
`deflection_efficiency` function is run here against the closed form of a
phase ramp that fits the Fourier grid exactly. It also checks the page against
quoting test evidence or another solver.
"""

import ast
import inspect
import math
import re

import docs_pages as dp
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

PAGE = dp.DOCS / "recipes" / "metasurface-from-cells.md"

#: Every tolerance the Floquet and diffraction verification cases assert.
TOLERANCES = (pc.F1_COARSE_TOL, pc.F1_S_TOL, pc.F1_CROSS_TOL, pc.BREWSTER_TOL)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def deflection_efficiency(page):
    """The page's ``deflection_efficiency`` function, run from its own block."""
    (block,) = [
        b for b in dp.blocks(page, "python") if b.startswith("def deflection_eff")
    ]
    namespace = {"np": np, "jnp": jnp, "c": pc.C0}
    exec(compile(block, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace["deflection_efficiency"]


def ramp(n, period, polarization, wavelength=0.633e-6, pitch=0.3e-6):
    """A unit-amplitude ramp of ``period`` cells, leaving at ``theta`` toward +e1."""
    f = pc.C0 / wavelength
    x = (np.arange(n) - (n - 1) / 2) * pitch
    sin_theta = wavelength / (period * pitch)
    t = jnp.asarray(np.exp(-2j * np.pi / wavelength * sin_theta * x))[None]
    zero = jnp.zeros_like(t)
    values = jnp.stack([t, zero] if polarization == "TM" else [zero, t], axis=1)
    return dif.ApertureField(values, np.array([f]), x), math.asin(sin_theta)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_a_ramp_on_the_fourier_grid_carries_its_flux_factor(
    deflection_efficiency, polarization
):
    field, theta = ramp(40, 10, polarization)
    got = deflection_efficiency(field, theta, 0.02, pad=1)
    factor = math.cos(theta) if polarization == "TE" else 1 / math.cos(theta)
    np.testing.assert_allclose(got, [factor], rtol=1e-12)


def test_a_ramp_leaves_toward_plus_e1_only(deflection_efficiency):
    field, theta = ramp(40, 10, "TE")
    assert float(deflection_efficiency(field, -theta, 0.02, pad=1)[0]) < 1e-12


def test_the_blocks_parse_and_bind_to_the_signatures(page):
    assert dp.blocks(page, "python")
    assert dp.static_problems(page) == []


def test_every_diffraction_name_on_the_page_is_public(page):
    names = set(re.findall(r"\b(?:dif|gfem\.diffraction)\.([A-Za-z_]\w*)", page))
    assert {"cell_response", "aperture_field", "propagate"} <= names
    assert {"floquet_ports", "ApertureField", "CellResponse"} <= names
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
    assert calls >= 5


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


def test_the_page_requires_the_sigmoid_map_and_the_validity_table(page):
    assert "lo + (hi - lo) * jax.nn.sigmoid" in page
    table = dp.section(page, "When the approximation holds")
    assert "40°" in table
    assert "| largest phase step" in table


@pytest.mark.parametrize(
    "text",
    [
        "{term}`locally periodic approximation`",
        "{term}`cell response`",
        "{term}`aperture field`",
        "what the literature calls a unit-cell or phase library",
        "boundary_morph",
        "gfem.shape.quality",
        "out=",
        "to_json",
        "from_json",
        "NonDifferentiableError",
        "jnp.fft",
        "{doc}`floquet-ports`",
        "{doc}`diffraction-efficiency`",
    ],
)
def test_the_page_carries(page, text):
    assert text in page or text in " ".join(page.split()), text


def test_the_literature_phrase_appears_once(page):
    flat = " ".join(page.split())
    assert flat.count("unit-cell or phase library") == 1


@pytest.mark.parametrize("heading", ["Check your results", "Limits", "Next"])
def test_the_page_has_its_closing_sections(page, heading):
    assert dp.section(page, heading).strip()


def test_the_names_the_page_calls_exist_in_gfem(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names - {"conventions", "diffraction"} <= set(gfem.__all__)
