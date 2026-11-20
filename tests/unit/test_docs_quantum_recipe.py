"""The quantum-metrics and loss-budget recipes run what can run without a mesh.

docs/recipes/quantum-metrics.md and docs/recipes/loss-budget.md solve nothing
at docs build time and print no solved number (user pages quote no test evidence): the
transmon tutorial is where solved values live, and how gfem checks them is on
docs/verification.md. This checks that the pages name public objects only,
quote no tolerance of the dispersive-shift verification, keep the default
``eigen_tolerance`` (forward polishing makes it sufficient), state the
contracts a reader must keep (one ``L_J`` for the port and the dressed modes,
what ``overlap`` does under differentiation, one junction), and that the
mesh-free `gfem.quantum.dressed_modes` call on the page runs.
"""

import re
from pathlib import Path

import docs_pages as dp
import numpy as np
import pytest

import gfem

RECIPES = dp.DOCS / "recipes"
QUANTUM = RECIPES / "quantum-metrics.md"
BUDGET = RECIPES / "loss-budget.md"
#: The dispersive shift's verification, whose tolerances no page quotes.
CHI_TESTS = Path(__file__).parents[1] / "verification" / "test_transmon_chi.py"


@pytest.fixture(scope="module")
def quantum():
    return QUANTUM.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def budget():
    return BUDGET.read_text(encoding="utf-8")


def _tolerances() -> set[str]:
    source = CHI_TESTS.read_text(encoding="utf-8")
    values = {float(v) for v in re.findall(r"(?:rtol|atol)=([\d.]+e-\d+)", source)}
    assert values
    return {f"{v:.0e}".replace("e-0", "e-") for v in values}


# -- Both pages -----------------------------------------------------------------


@pytest.mark.parametrize("path", [QUANTUM, BUDGET], ids=lambda p: p.name)
def test_the_page_quotes_no_verification_tolerance(path):
    page = path.read_text(encoding="utf-8")
    for written in _tolerances():
        assert not re.search(rf"(?<![\d.e-]){re.escape(written)}(?!\d)", page), written


@pytest.mark.parametrize("path", [QUANTUM, BUDGET], ids=lambda p: p.name)
def test_the_page_names_no_test_file(path):
    assert "tests/" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [QUANTUM, BUDGET], ids=lambda p: p.name)
def test_every_gfem_name_on_the_page_is_public(path):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", path.read_text(encoding="utf-8")))
    assert names
    assert names - {"conventions", "shape", "quantum"} <= set(gfem.__all__)


@pytest.mark.parametrize("path", [QUANTUM, BUDGET], ids=lambda p: p.name)
def test_the_page_keeps_the_default_eigen_tolerance(path):
    """Only the raise above 1e-10 is named; no page asks for a tighter solve."""
    page = path.read_text(encoding="utf-8")
    assert "eigen_tolerance=" not in page
    quoted = re.findall(r"eigen_tolerance`?[^.\n]{0,30}?(\de-\d+)", page)
    assert set(quoted) <= {"1e-10"}


# -- The quantum-metrics recipe -------------------------------------------------


def test_the_mesh_free_call_runs(quantum):
    (block,) = [
        b
        for b in dp.blocks(quantum, "python")
        if "dressed_modes(" in b and "gfem.solve(" not in b
    ]
    namespace: dict = {}
    exec(compile(block, str(QUANTUM), "exec"), namespace)  # noqa: S102
    dressed = namespace["dressed"]
    assert isinstance(dressed, gfem.quantum.DressedModes)
    shift = np.asarray(dressed.dispersive_shift)
    assert shift.shape == (2, 2)
    assert shift[0, 1] < 0  # a transmon pulls its resonator down
    assert float(dressed.overlap) > 0.5


def test_one_inductance_feeds_the_port_and_the_dressed_modes(quantum):
    section = dp.section(quantum, "The junction inductance as a design input")
    (block,) = [b for b in dp.blocks(section, "python") if "model_at(" in b]
    assert re.search(r"LumpedPort\([^)]*inductance=L_J", block)
    assert re.search(r"dressed_modes\([^)]*inductance=L_J", block, re.DOTALL)
    assert "cannot detect" in dp._prose(section)


def test_the_overlap_is_read_not_differentiated(quantum):
    refusals = dp._prose(dp.section(quantum, "When it refuses"))
    assert "overlap" in refusals
    assert "NaN" in refusals
    assert "gfem.NonDifferentiableError" in refusals
    assert "gfem.ModeLabelError" in refusals


def test_the_sheet_section_drops_the_target_advice(quantum):
    sheet = dp.section(quantum, "Kinetic inductance of a film: sheet participation")
    assert "sheet_participation" in sheet
    assert "sheet-plasmon" in sheet
    assert not re.search(r"`target`\s+close", quantum)


def test_the_limits_name_one_junction_and_the_target(quantum):
    limits = dp.section(quantum, "Limits")
    assert "ValueError" in limits
    assert "Eigenmode.target" in limits
    assert "{doc}`eigenmode-gradient`" in limits


def test_the_page_links_the_conventions(quantum):
    assert "{py:obj}`gfem.conventions`" in quantum


# -- The loss-budget recipe -----------------------------------------------------


def test_the_budget_adds_the_three_kinds_of_term(budget):
    (block, *_) = dp.blocks(budget, "python")
    assert "1 / modes.q" in block
    assert ".inverse_q" in block
    assert "bulk_participation" in block


def test_the_budget_states_the_t1_conversion(budget):
    t1 = dp._prose(dp.section(budget, "T1 limits"))
    assert "Q_i / Re(omega)" in t1
    assert "q_port" in t1  # the Purcell limit
