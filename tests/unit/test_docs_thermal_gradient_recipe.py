"""The thermal gradient recipe runs and its gradients meet their closed forms.

docs/recipes/thermal-gradient.md runs nothing at docs build time. This runs
its blocks on the repository's own heater section
(docs/tutorials/data/heater_w2.msh): one jax.grad through DC conduction,
Joule heating, the steady temperature and the thermo-optic shift, whose
gradients times their inputs must be 2, -1, -1 (to the convection's share)
and 1 times the phase.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "thermal-gradient.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("thermal_gradient"))


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_phase_is_near_pi(ran):
    (phase,) = dp.numbers(ran[0].printed)
    assert phase == pytest.approx(3.14, abs=0.02)


def test_each_gradient_is_its_power_law(ran):
    (example,) = [
        e for e in ran if e.section == "Check the gradient against closed forms"
    ]
    lines = [dp.numbers(line)[-1] for line in example.printed.splitlines()]
    current, sigma, k, scale, *regions = lines
    assert current == pytest.approx(2.0, abs=1e-6)
    assert sigma == pytest.approx(-1.0, abs=1e-6)
    assert scale == pytest.approx(1.0, abs=1e-6)
    # the convection on top does not scale with k: a few 1e-5 (2.1e-5 measured)
    assert k == pytest.approx(-1.0, abs=5e-5)
    assert k != pytest.approx(-1.0, abs=1e-6)
    assert all(share < 0 for share in regions)
    # the neighbour and the neighbour's window are left out (1.8e-3 measured)
    assert sum(regions) == pytest.approx(k, abs=3e-3)


def test_the_page_lists_what_raises():
    page = PAGE.read_text(encoding="utf-8")
    section = dp._prose(dp.section(page, "What raises"))
    for name in ("gfem.NonDifferentiableError", "gfem.rise_time", "gfem.time_constant"):
        assert name in section, name
