"""The acousto-optic recipe runs and prints what the page shows.

docs/recipes/acousto-optic-shift.md runs nothing at docs build time. This runs
its blocks in order on the tutorial's section at coarse edges, built by the
repository's own docs/tutorials helpers (the page downloads them), and checks
the printed numbers: the arms respond alike, the photoelastic and electro-optic
parts have opposite signs, V_pi L of an arm is lambda / (2 |dn|), the incident
referral multiplies the push-pull product by the impedance ratio, and the
gradient in the slide agrees with its central difference.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "acousto-optic-shift.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("acousto_optic_shift")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str) -> dp.Example:
    (example,) = [example for example in ran if example.section == section]
    return example


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_arms_respond_alike_and_v_pi_l_is_the_shifts(ran):
    lines = _example(ran, "The shift per volt").printed.splitlines()
    numbers = dp.numbers(lines[0])  # mode, f, n_eff, count, f min, f max, V m min, max
    assert 1.83 < numbers[2] < 1.85
    (ratio,) = dp.numbers(lines[1])
    # the two ribs differ by the meshes' asymmetry and the strain's gradient (1.3e-2)
    assert ratio == pytest.approx(1.0, abs=3e-2)
    (v_pi_l_cm,) = dp.numbers(lines[2])[-1:]
    # the middle frequency's V_pi L lies between the band's extremes
    assert 100 * numbers[-2] < v_pi_l_cm < 100 * numbers[-1]


def test_the_two_parts_are_of_opposite_sign(ran):
    lines = _example(ran, "The photoelastic and the electro-optic part").printed
    _, second, third = lines.splitlines()
    ratio, away = dp.numbers(second)
    (total,) = dp.numbers(third)
    assert 0.25 < ratio < 0.40
    assert abs(away) < 45  # degrees from exactly opposed
    assert 0.5 < total < 0.9  # the sum is smaller than the strain's part


def test_the_incident_product_is_the_push_pull_one_times_the_referral(ran):
    rows = [
        dp.numbers(line)
        for line in _example(
            ran, "Push-pull and the incident wave"
        ).printed.splitlines()
    ]
    assert len(rows) == 3
    for _, on_fingers, incident in rows:
        # a capacitive transducer: the fingers see about twice the incident wave
        assert 0.4 < incident / on_fingers < 0.7
    assert all(row[1] > 0 for row in rows)


def test_the_slide_gradient_matches_its_central_difference(ran):
    lines = _example(
        ran, "Where the windows sit, and the gradient"
    ).printed.splitlines()
    grad, fd = dp.numbers(lines[-1])[-2], dp.numbers(lines[-1])[-1]
    # noise of the two solves (~1e-14 of the value) over 2 h |f'| at h = 1e-12 m
    # is 5e-7; the printed figures carry 5 digits
    assert grad == pytest.approx(fd, rel=1e-4)
