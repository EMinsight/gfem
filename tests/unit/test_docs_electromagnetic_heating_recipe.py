"""The electromagnetic heating recipe runs and prints what the page shows.

docs/recipes/electromagnetic-heating.md runs nothing at docs build time. This
runs its blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, and checks every printed number against a hand
value: the losses add up to 1 - |S11|^2 - |S21|^2, the ceramic's loss is near
the lossy section's closed form, the heat is the drive times the loss and all
of it reaches the sink, the sheet heat splits between the top and the sink,
separate tones heat as the mean of the two coherent phases, and the gradient
matches its central difference.
"""

import docs_pages as dp
import numpy as np
import pytest
from loaded_line import absorbed

PAGE = dp.DOCS / "recipes" / "electromagnetic-heating.md"
FREQUENCY, EPS, TAN_DELTA = 5e9, 10.0, 0.02  # the page's line and ceramic


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("electromagnetic_heating"))


def _lines(ran, section: str) -> list[np.ndarray]:
    """The numbers of each line ``section``'s block printed."""
    (example,) = [example for example in ran if example.section == section]
    return [dp.numbers(line) for line in example.printed.splitlines()]


def test_the_page_downloads_the_line(page):
    first = dp.examples(page)[0]
    assert dp.MESH_URL in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_losses_add_up_to_what_s_does_not_account_for(ran):
    feed, design, top, total, unaccounted = (n[-1] for n in _lines(ran, "Loss split"))
    assert feed == 0.0  # the air is lossless
    assert design + top == pytest.approx(total, abs=2e-6)
    assert total == pytest.approx(unaccounted, abs=2e-6)
    # the copper top takes a little of the section's closed-form loss
    hand = absorbed(FREQUENCY, "Ez", EPS * (1 - 1j * TAN_DELTA))
    assert design == pytest.approx(hand, rel=1e-2)
    assert 0 < top < 0.05 * design


def test_the_heat_is_the_drive_times_the_loss_and_reaches_the_sink(ran):
    lines = _lines(ran, "Heat and temperature")
    total = _lines(ran, "Loss split")[3][-1]
    repr_total, volume, sheet = lines[0][-3:]
    assert repr_total == pytest.approx(total, rel=1e-5)  # 1 W in at p1
    assert volume + sheet == pytest.approx(repr_total, rel=1e-5)
    (power,), (sink,) = lines[1], lines[2]
    assert power == pytest.approx(total, rel=1e-5)
    assert sink == pytest.approx(power, rel=1e-5)
    design, feed = lines[3][-1], lines[4][-1]
    assert feed > design > 0  # the copper's heat crosses the air


def test_a_cooled_top_takes_a_small_share_of_the_heat(ran):
    (top,), (bottom,) = (n[-1:] for n in _lines(ran, "Where sheet heat goes"))
    power = _lines(ran, "Heat and temperature")[1][0]
    assert top + bottom == pytest.approx(power, rel=1e-5)
    assert 0 < top < 0.01 * power


def test_separate_tones_heat_as_the_mean_of_the_two_phases(ran):
    in_phase, opposed, (apart,) = (
        n[-3:] if i < 2 else n[-1:]
        for i, n in enumerate(_lines(ran, "Coherent and incoherent drives"))
    )
    assert in_phase[0] > apart > opposed[0] > 0
    assert apart == pytest.approx((in_phase[0] + opposed[0]) / 2, rel=1e-5)
    # 0.5 W into each port, apart: the heat of 1 W into one port
    total = _lines(ran, "Loss split")[3][-1]
    assert apart == pytest.approx(total, rel=1e-5)


def test_the_gradient_matches_its_central_difference(ran):
    (rise,), (summed,), (fd,) = (n[-1:] for n in _lines(ran, "Gradients"))
    design = _lines(ran, "Heat and temperature")[3][-1]
    assert rise == pytest.approx(design, abs=1e-4)
    assert summed == pytest.approx(fd, rel=1e-5)
    # more loss, more heat: the ceramic warms
    assert summed > 0
