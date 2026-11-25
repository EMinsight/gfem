"""The waveguide nonlinearity recipe runs and prints what the page shows.

docs/recipes/waveguide-nonlinearity.md runs nothing at docs build time. This
runs its blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: the figure of merit of the material, the
exact phase mismatch against its dispersion estimate, the self-phase entry of
the coupling tensor against the coefficient of the same row, and the
linearity in the Kerr index.
"""

import itertools

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "waveguide-nonlinearity.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("waveguide_nonlinearity")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_coefficient_of_a_silicon_wire_falls_with_the_wavelength(ran):
    lines = _lines(ran, "The nonlinear coefficient over a band")[1:6]
    gamma = [dp.numbers(line)[-3] for line in lines]
    assert gamma[2] == pytest.approx(177, rel=0.02)
    assert all(a > b for a, b in itertools.pairwise(gamma))
    # the weak-guidance area of a wire this small is a fraction of a um^2
    assert 0.05 < dp.numbers(lines[2])[-2] < 0.2


def test_the_figure_of_merit_is_the_materials(ran):
    lines = _lines(ran, "Two-photon absorption and the figure of merit")
    guide, material = dp.numbers(lines[1])[-2:]
    assert guide == pytest.approx(material, abs=1e-3)
    assert material == pytest.approx(0.32, abs=0.01)
    losses = [dp.numbers(line)[-1] for line in lines[2:5]]
    assert losses[1] == pytest.approx(10 * losses[0], rel=0.02)  # square of power


def test_the_exact_phase_mismatch_is_near_its_dispersion_estimate(ran):
    numbers = dp.numbers(_lines(ran, "Three waves: phase mismatch and couplings")[0])
    exact, estimate = numbers[0], numbers[-1]  # the 2 of Omega^2 between them
    assert exact > 0  # anomalous guide
    assert exact == pytest.approx(estimate, rel=0.05)
    assert exact != pytest.approx(estimate, rel=1e-4)  # beta_4 is there


def test_the_coupling_tensor_has_the_self_phase_on_its_diagonal(ran):
    lines = _lines(ran, "Three waves: phase mismatch and couplings", 1)
    spm, xpm, fwm = dp.numbers(lines[0])[-3:]
    assert 1.7 < xpm / spm < 2.0
    assert 0.9 < fwm / spm < 1.0
    matched = dp.numbers(lines[1])[-1]  # W
    assert 1 < matched < 3
    # the tensor's entry and the coefficient of the same row, both printed
    entry, row = dp.numbers(lines[2])[-4:-2], dp.numbers(lines[2])[-2:]
    assert entry == pytest.approx(row, abs=5e-3)


def test_the_ring_parameters_are_those_of_a_ten_micron_ring(ran):
    first, second = _lines(ran, "Parameters of a microring")
    free_spectral_range, d2 = dp.numbers(first)[0], dp.numbers(first)[-1]
    assert free_spectral_range == pytest.approx(1137, rel=0.02)
    assert d2 > 0  # anomalous dispersion
    g0, threshold = dp.numbers(second)[-3:-1]
    assert g0 == pytest.approx(293, rel=0.03)
    assert threshold == pytest.approx(10.3, rel=0.05)


def test_the_coefficient_converges_in_the_clearance(ran):
    rows = [dp.numbers(line)[-1] for line in _lines(ran, "Check your results")[:3]]
    assert rows[1] == pytest.approx(rows[2], rel=1e-4)
    assert rows[0] == pytest.approx(rows[1], rel=5e-3)


def test_the_coefficient_is_linear_in_the_kerr_index(ran):
    (rise,) = dp.numbers(_lines(ran, "Check your results", 1)[0])
    assert rise == pytest.approx(1.0, abs=1e-6)
