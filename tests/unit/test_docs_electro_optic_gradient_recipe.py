"""The electro-optic gradient recipe runs and its gradients meet their closed forms.

docs/recipes/electro-optic-gradient.md runs nothing at docs build time. This
runs its blocks on the modulator tutorial's section, built by the
repository's own docs/tutorials/tfln_section.py (the page downloads it): one
jax.grad of an arm's phase with respect to the tensor, the static
permittivities and the drive, whose gradients times their inputs sum to 1, 0
and 1 times the phase; the gap gradient through two summed morphs against a
difference at 1 nm; the width by a difference over remeshes; and the
optical permittivity raising under jax.grad.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "electro-optic-gradient.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("electro_optic_gradient")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str) -> list:
    (example,) = [example for example in ran if example.section == section]
    return [dp.numbers(line) for line in example.printed.splitlines()]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_arm_a_phase_is_a_negative_delay(ran):
    ((*_, phase),) = _lines(ran, "The phase as a function of its inputs")
    # 2 mm at 1 V of the tutorial's arm a: k0 * 2.41e-5 * 2e-3 = 0.195 rad
    assert phase == pytest.approx(-0.195, rel=0.02)


def test_each_gradient_is_its_closed_form(ran):
    lines = [n[-1] for n in _lines(ran, "Check the gradient against closed forms")]
    tensor, r13, r22, r33, r51, eps, x, y, z, oxide, air, drive = lines
    assert tensor == pytest.approx(1.0, abs=1e-6)
    assert r13 + r22 + r33 + r51 == pytest.approx(1.0, abs=2e-4)  # printed to 1e-4
    assert r33 > 0.9
    assert abs(r22) < 1e-3  # crystal y is along the guide
    assert eps == pytest.approx(0.0, abs=1e-9)
    assert x + y + z + oxide + air == pytest.approx(0.0, abs=3e-4)
    assert z < 0 < oxide
    assert drive == pytest.approx(1.0, abs=1e-6)


def test_the_gap_gradient_matches_a_small_difference(ran):
    per_um, one_nm, _two_nm = (n[-1] for n in _lines(ran, "The electrode gap"))
    # the tutorial's finer section: -0.296 per um
    assert per_um == pytest.approx(-0.296, abs=0.01)
    assert one_nm < 1e-4


def test_the_width_difference_is_small_next_to_the_gap(ran):
    ((per_um,),) = _lines(ran, "The ridge width, by a difference over remeshes")
    assert abs(per_um) < 0.1


def test_the_page_lists_what_raises(ran):
    (example,) = [e for e in ran if e.section == "What raises"]
    assert example.printed.strip() == "NonDifferentiableError"
    section = dp._prose(dp.section(PAGE.read_text(encoding="utf-8"), "What raises"))
    for name in ("gfem.NonDifferentiableError", "jax.lax.stop_gradient", "eps_r"):
        assert name in section, name


def test_the_page_names_no_other_tool():
    text = PAGE.read_text(encoding="utf-8").lower()
    for name in ("palace", "tidy3d", "flexcompute", "gdsfactory"):
        assert name not in text, name
