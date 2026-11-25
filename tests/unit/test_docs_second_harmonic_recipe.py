"""The second-harmonic recipe runs and prints what the page shows.

docs/recipes/second-harmonic-generation.md runs nothing at docs build time. This
runs its blocks in order, on the ridge section built by the repository's own
docs/recipes/ridge_section.py (the page downloads it), and checks the numbers
against what the physics fixes: Miller's rule from the indices, the selection
rule that leaves one harmonic mode coupled to the pump, the coherence length and
the poling period as two forms of the same mismatch, the poling factors, the
equality of the two entries of the coupling, and the quadratic dependence on the
tensor.
"""

import math

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "second-harmonic-generation.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("second_harmonic")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_crystal_has_the_x_cut_indices_and_the_cited_tensor(ran):
    (line,) = _lines(ran, "Lithium niobate and its $d$ tensor")[:1]
    n_x, n_y, n_z = dp.numbers(line)[:3]
    assert n_x == n_y > n_z  # ordinary, ordinary, extraordinary (Zelmon, 1.55 um)
    assert n_z == pytest.approx(2.13756, abs=1e-5)
    assert dp.numbers(line)[-1] == pytest.approx(2.5e-11, rel=1e-12)  # d33


def test_miller_rule_lowers_d33_from_1064_to_1550_nm(ran):
    lines = _lines(ran, "Lithium niobate and its $d$ tensor", 1)
    factor, d33 = dp.numbers(lines[0])[:2]
    assert factor == pytest.approx(0.8996, abs=1e-4)
    assert d33 == pytest.approx(25.2 * factor, abs=0.05)


def test_only_one_harmonic_mode_is_coupled_to_the_pump(ran):
    lines = _lines(ran, "The guide's modes", 1)
    n_eff = [dp.numbers(line)[1] for line in lines]
    eta = [dp.numbers(line)[2] for line in lines]
    assert n_eff == sorted(n_eff, reverse=True)  # ranked by index
    # the others are quasi-TM or hybrid: orthogonal to the pump's squared field
    assert eta[2] > 100 * eta[1] > 1e4 * eta[0]


def test_the_phase_mismatch_gives_the_poling_period_and_the_coherence_length(ran):
    lines = _lines(ran, "The figures of the pair")
    db = dp.numbers(lines[1])[0]
    period = dp.numbers(lines[3])[0]
    assert db > 0  # normal dispersion
    assert period == pytest.approx(2 * math.pi / db * 1e6, rel=1e-3)
    assert 3.5 < period < 4.5  # the thin-film order: Wang et al. pole at 4 um
    coherence = dp.numbers(_lines(ran, "The harmonic along a poled guide")[0])[0]
    assert coherence == pytest.approx(period / 2, rel=1e-3)


def test_poling_factors_and_the_mismatch_of_a_wrong_period(ran):
    lines = _lines(ran, "The harmonic along a poled guide")
    unpoled, half, d39, off = (dp.numbers(line)[-2] for line in lines[1:5])
    # |G_1|^2 of a duty cycle D against D = 1/2
    assert d39 / half == pytest.approx(math.sin(0.39 * math.pi) ** 2, rel=2e-3)
    assert half > 1e5 * unpoled
    assert off < 1e-4 * half  # a period 1 % long
    assert 0.1 < half < 2  # per cent: the undepleted pump is a fair assumption


def test_the_band_is_matched_at_its_centre_and_falls_off_it(ran):
    lines = _lines(ran, "Efficiency over a band")
    ratio = [dp.numbers(line)[-1] for line in lines]
    assert ratio[2] == max(ratio)
    assert ratio[0] == pytest.approx(ratio[4], rel=5e-2)  # a symmetric sinc^2 peak
    assert 0.5 * ratio[2] < ratio[0] < ratio[2]
    delta_beta = [dp.numbers(line)[1] for line in lines]
    assert delta_beta[0] > delta_beta[4]  # beta_2w - 2 beta_w falls with the wavelength


def test_the_coupling_tensor_holds_the_second_harmonic_entry(ran):
    lines = _lines(ran, "Sum and difference frequency, and parametric processes")
    assert dp.numbers(lines[1])[0] == pytest.approx(dp.numbers(lines[2])[0], rel=1e-3)
    assert lines[3].endswith("True")


def test_the_efficiency_is_quadratic_in_the_tensor(ran):
    (ratio,) = dp.numbers(_lines(ran, "Check your results")[0])
    assert ratio == pytest.approx(4.0, abs=1e-5)
