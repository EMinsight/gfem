"""The electro-optic phase recipe runs and prints what the page shows.

docs/recipes/electro-optic-phase.md runs nothing at docs build time. This runs
its blocks in order on the modulator tutorial's section, built by the
repository's own docs/tutorials/tfln_section.py (the page downloads it), and
checks the printed numbers: the two arms' shifts are equal and opposite by the
section's mirror symmetry, V_pi L of one arm is lambda / (2 |dn/dV|), the
push-pull pair needs half of it, the capacitance per length is the solve's,
and the TE mode is the second.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "electro-optic-phase.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("electro_optic_phase")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str) -> dp.Example:
    (example,) = [example for example in ran if example.section == section]
    return example


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_arms_shift_equal_and_opposite(ran):
    lines = _example(ran, "The shift per volt").printed.splitlines()
    numbers = dp.numbers(lines[0])
    n_eff, dn, v_pi_l = numbers[-3:]
    assert 1.85 < n_eff < 1.95
    assert dn < 0  # arm a: a field along the crystal's +z lowers the index
    assert v_pi_l == pytest.approx(1.55e-6 / (2 * abs(dn)), rel=1e-5)
    (ratio,) = dp.numbers(lines[1])
    # the coarse meshes are not mirror images (3.3e-3 measured)
    assert ratio == pytest.approx(-1.0, abs=5e-3)
    (v_pi_l_cm,) = dp.numbers(lines[2])
    assert v_pi_l_cm == pytest.approx(100 * v_pi_l, rel=1e-3)


def test_push_pull_needs_half_an_arm(ran):
    shift = _example(ran, "The shift per volt").printed.splitlines()
    ((ratio,), (arm,)) = (dp.numbers(line) for line in shift[1:])
    lines = _example(ran, "Push-pull and several terminals").printed.splitlines()
    (push_pull,) = dp.numbers(lines[0])
    length_mm, v_pi = dp.numbers(lines[1])
    # lambda / (2 |s_a - s_b|), with s_b = ratio * s_a
    assert push_pull == pytest.approx(arm / (1 - ratio), rel=1e-3)
    # the tutorial's finer section gives 1.61 V cm
    assert push_pull == pytest.approx(1.61, rel=0.05)
    assert v_pi == pytest.approx(push_pull * 1e-2 / (length_mm * 1e-3), rel=2e-2)


def test_the_capacitance_per_length_is_the_solves(ran):
    printed = _example(ran, "The electrodes' capacitance per length").printed
    *_, per_length = dp.numbers(printed)
    assert "capacitance per length in pF/m" in printed  # a section's repr
    # two gaps of 3.5 um on a film of permittivity tens: tens to hundreds of pF/m
    assert 30 < per_length < 300


def test_the_te_mode_is_the_second(ran):
    lines = _example(ran, "Choose the window and the mode").printed.splitlines()
    first, second = (dp.numbers(line) for line in lines)
    assert first[-1] < 0.1  # TM
    assert second[-1] > 0.9  # TE
    assert first[-2] > second[-2]  # numbered from the most propagating


def test_the_page_names_no_other_tool():
    text = PAGE.read_text(encoding="utf-8").lower()
    for name in ("palace", "tidy3d", "flexcompute", "gdsfactory"):
        assert name not in text, name
