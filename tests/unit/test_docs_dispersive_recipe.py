"""The dispersive-materials recipe runs what can run without a mesh.

docs/recipes/dispersive-materials.md solves nothing at docs build time and
prints no solved number (ADR 0017, as amended): the metal-bend tutorial is
where solved values live, and how gfem checks them is on docs/verification.md.
This checks that the page's law block runs and builds what the page says, that
the page shows the pattern that keeps a law's frequencies of order one for an
optimizer, states the advice and the limits a reader must keep, and links the
conventions and the tutorial.
"""

import re

import docs_pages as dp
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "dispersive-materials.md"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def laws(page):
    """The namespace of the page's first block, which needs no mesh."""
    (block, *_) = dp.blocks(page, "python")
    assert "Mesh" not in block
    assert "solve(" not in block
    namespace: dict = {}
    exec(compile(block, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace


def test_the_page_prints_no_solved_number(page):
    assert dp.blocks(page, "text") == []
    for block in dp.blocks(page, "python"):
        assert "print(" not in block


def test_the_law_block_builds_gold_and_a_datasheet_substrate(laws):
    gold = laws["gold"]
    assert isinstance(gold, gfem.Material)
    assert isinstance(gold.eps_r, gfem.PermittivityLaw)
    # gold at 1.55 µm: Re eps far below 0, a little loss (the tutorial's value)
    eps = complex(gold.eps_r(299792458.0 / 1.55e-6))
    assert eps.real == pytest.approx(-125.83, abs=0.01)
    assert eps.imag == pytest.approx(-8.39, abs=0.01)
    fr4 = laws["fr4"].eps_r
    # the datasheet value is met at the datasheet frequency
    assert complex(fr4(1e9)) == pytest.approx(4.3 * (1 - 0.02j), rel=1e-12)
    assert fr4.eps_inf == pytest.approx(3.92, abs=5e-3)
    assert fr4.terms[0].delta_eps == pytest.approx(1.135, abs=5e-3)


def test_every_term_class_appears_on_the_page(page):
    for name in ("Drude", "Lorentz", "Debye", "DjordjevicSarkar"):
        assert f"gfem.{name}(" in page, name


def test_the_gradient_keeps_law_frequencies_of_order_one(page):
    section = dp.section(page, "Gradients")
    (block, *_) = dp.blocks(section, "python")
    assert re.search(r"resonance_frequency=\w+ \* ", block)
    assert "per hertz" in dp._prose(section)


def test_a_metal_blend_starts_from_a_binary_design(page):
    section = dp.section(page, "Metals in a topology optimization")
    assert 'scheme="index"' in section
    assert "binary design" in dp._prose(section)
    assert "{doc}`../tutorials/metal-bend`" in section


def test_the_problems_section_names_each_problem_and_its_refusal(page):
    section = dp._prose(dp.section(page, "Which problems take a law"))
    for name in ("Driven", "Eigenmode", "PortMode", "Electrostatic"):
        assert f"gfem.{name}" in section, name
    assert "{term}`unresolved band`" in section
    assert "{term}`energy permittivity`" in section
    assert "{term}`static permittivity`" in section


def test_the_limits_say_where_nearest_is_ill_defined(page):
    limits = dp._prose(dp.section(page, "Limits"))
    assert "{term}`modal overlap`" in limits
    assert "100 restarts" in limits
    assert "gfem.EigenmodeConvergenceError" in limits


def test_the_page_links_the_conventions(page):
    assert "{py:obj}`gfem.conventions`" in page


def test_the_page_quotes_no_tolerance_of_the_metal_bend_test(page):
    source = (dp.ROOT / "tests" / "verification" / "test_metal_bend.py").read_text()
    for value in set(re.findall(r"(?:rtol|atol|abs|rel)=([\d.]+e-\d+)", source)):
        assert value not in page, value


def test_the_check_section_compares_with_a_constant_material(page):
    check = dp.section(page, "Check your results")
    (block, *_) = dp.blocks(check, "python")
    assert "tan_delta=" in block
    assert "conductivity=" in block
