"""The fit recipe runs and prints what the page shows.

docs/recipes/fit-measured-data.md runs nothing at docs build time. This runs
its blocks in order on the repository's copy of the gold film table the page
downloads (McPeak et al. 2015, CC0), so the printed `PermittivityFit` on the
page is the one the fit returns. The block that builds a model needs a mesh
the repository does not ship and is left out; the static page check covers
it. The page must also state the convention a table is read in, the remedies
of each refusal and warning, and link the terms and the conventions; it quotes
none of the thresholds the verification test uses on the same table.
"""

import re

import docs_pages as dp
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "fit-measured-data.md"
#: The section whose block needs a mesh the repository does not ship.
MODEL_SECTION = "Give the law to a model"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("fit_measured_data")
    return dp.run_page(PAGE, workdir, skip=(MODEL_SECTION,))


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_fit_prints_its_terms_errors_and_range(ran):
    printed = "".join(example.printed for example in ran)
    assert "PermittivityFit(Drude + " in printed
    assert "Hz)" in printed


def test_the_model_block_gives_the_fitted_law_to_a_region(page):
    (block,) = dp.blocks(dp.section(page, MODEL_SECTION), "python")
    assert "gfem.Model(" in block
    assert "gold" in block


@pytest.mark.parametrize(
    "term",
    [
        "extinction coefficient",
        "fitted range",
        "permittivity fit",
        "permittivity law",
    ],
)
def test_the_page_links_its_terms(page, term):
    linked = {
        (re.search(r"<([^>]+)>\s*$", role) or [None, role])[1].strip().lower()
        for role in re.findall(r"\{term\}`([^`]+)`", page)
    }
    assert term in linked


def test_extinction_is_always_the_extinction_coefficient(page):
    for match in re.finditer(r"extinction", page, re.IGNORECASE):
        assert re.match(r"\s+coefficient", page[match.end() :]), page[
            match.start() - 40 : match.end() + 40
        ]


def test_the_page_states_the_convention_and_the_remedies(page):
    for text in (
        "np.conj",
        "drude=True",
        r"\mathrm{Im}\,\varepsilon = -2nk",
        "(n - ik)^2",
        "gfem.FittedRangeWarning",
        "gfem.PermittivityFitWarning",
        "frequency_range=None",
        "max_lorentz_terms",
    ):
        assert text in page, text


def test_the_page_links_the_conventions_and_the_library(page):
    assert "{py:obj}`gfem.conventions`" in page
    assert "{py:mod}`gfem.materials`" in page
    assert "{doc}`dispersive-materials`" in page


def test_the_page_quotes_no_threshold_of_the_data_test(page):
    source = (
        dp.ROOT / "tests" / "verification" / "test_fit_permittivity_data.py"
    ).read_text(encoding="utf-8")
    number = r"(\d+e-\d+|\d*\.\d+)"
    thresholds = set(re.findall(rf"[<>]=? {number}", source))
    thresholds |= set(re.findall(rf"{number} <", source))
    assert thresholds  # the pattern still reads the test
    for value in thresholds:
        assert not re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", page), value


def test_the_api_lists_the_library_page():
    index = (dp.DOCS / "api" / "index.md").read_text(encoding="utf-8")
    tree = dp.toctree_entries(index)
    assert tree.index("gfem/materials/index") == tree.index("gfem/model/index") + 1


def test_the_library_holds_only_its_functions():
    """A re-export of `Material` here would move its API page to this module."""
    import gfem.materials

    assert all(callable(getattr(gfem.materials, n)) for n in gfem.materials.__all__)
    assert not set(gfem.materials.__all__) & set(gfem.model.__all__)


def test_the_conventions_state_the_extinction_coefficient():
    doc = gfem.conventions.__doc__
    assert "extinction coefficient" in doc
    assert "(n - ik)^2" in doc
    assert r"\mathrm{Im}\,\varepsilon = -2nk" in doc
    assert "fit_permittivity" in doc


def test_the_conventions_page_states_the_extinction_coefficient():
    page = (dp.DOCS / "conventions.md").read_text(encoding="utf-8")
    section = dp._prose(dp.section(page, "Dispersive materials"))
    assert "{term}`extinction coefficient`" in section
    assert "(n - ik)^2" in section
    assert "{py:obj}`gfem.fit_permittivity`" in section
