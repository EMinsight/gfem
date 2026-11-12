"""The "Solve a device" recipe runs and prints what the page shows.

docs/recipes/solve-a-device.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, and compares every printed number with a hand
formula coded here (the matched-line transmission, the plate attenuation, the
open-ended line's resonances, the plate capacitance) and with the output
block the page shows under the code.
"""

import docs_pages as dp
import numpy as np
import pytest
from scipy.optimize import brentq

PAGE = dp.DOCS / "recipes" / "solve-a-device.md"
C0 = 299792458.0
ETA0 = 376.73
BAND = np.linspace(2e9, 8e9, 7)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("solve_a_device"))


def _printed(ran, section: str) -> list[np.ndarray]:
    """The numbers each block of ``section`` printed, one array per block."""
    found = [
        dp.numbers(example.printed) for example in ran if example.section == section
    ]
    assert found, f"no Python block under {section!r}"
    return found


def _open_line_resonances(count: int) -> np.ndarray:
    """Resonances (Hz) of 10 mm air, 20 mm of eps_r = 10, 10 mm air, open ends.

    A TEM line with a magnetic wall at each end; by symmetry the middle plane
    is a magnetic or an electric wall, which gives one equation each.
    """
    n, a = np.sqrt(10.0), 10e-3

    def magnetic(f):
        k = 2 * np.pi * f / C0
        return np.tan(k * a) + n * np.tan(k * n * a)

    def electric(f):
        k = 2 * np.pi * f / C0
        return np.tan(k * a) * np.tan(k * n * a) - n

    grid = np.linspace(0.2e9, 9e9, 4000)
    roots = []
    for equation in (magnetic, electric):
        values = equation(grid)
        for lo, hi, at_lo, at_hi in zip(
            grid[:-1], grid[1:], values[:-1], values[1:], strict=True
        ):
            # A sign change between two small values is a root, not a pole.
            if at_lo * at_hi < 0 and abs(at_lo) < 50 and abs(at_hi) < 50:
                roots.append(brentq(equation, lo, hi))
    return np.sort(roots)[:count]


def test_the_page_downloads_the_tutorial_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    assert f'url = "{dp.MESH_URL}"' in first.code
    assert dp.MESH.is_file()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_every_section_with_a_number_prints_it(ran):
    printing = {example.section for example in ran if example.printed.strip()}
    assert printing == {
        "S-parameters over a band",
        "Walls and metals",
        "Resonances and Q",
        "Coupling Q of a port",
        "Capacitance",
        "Check your results",
    }


def test_the_band_is_the_matched_line_transmission(ran):
    (printed,) = _printed(ran, "S-parameters over a band")
    frequency, s21, db, z_ref = printed[:7], printed[7:14], printed[14:21], printed[21:]
    assert frequency * 1e9 == pytest.approx(BAND)
    # The page says "within 0.00002".
    assert s21 == pytest.approx(dp.line_formula(BAND), abs=2e-5)
    assert s21[[0, 3, 6]] == pytest.approx([0.8306, 0.9013, 0.6052], abs=1e-4)
    assert db == pytest.approx(20 * np.log10(s21), abs=1e-6)
    assert z_ref == pytest.approx([ETA0 / 6, ETA0 / 6])


def test_an_unnamed_surface_is_a_magnetic_wall(ran):
    walls, _ = _printed(ran, "Walls and metals")
    unnamed, named, formula = walls[:2], walls[2:4], walls[4]
    assert unnamed == pytest.approx(named, abs=1e-12)
    k = 2 * np.pi * 5e9 / C0
    assert formula == pytest.approx(np.angle(np.exp(-1j * k * 40e-3)))
    assert named == pytest.approx([1.0, formula], abs=1e-6)


def test_a_copper_plate_absorbs_what_the_attenuation_formula_says(ran):
    _, (solved, formula) = _printed(ran, "Walls and metals")
    r_s = np.sqrt(np.pi * 5e9 * 4e-7 * np.pi / 5.8e7)
    alpha = r_s / (2 * ETA0 * 1e-3)
    assert formula == pytest.approx(np.exp(-2 * alpha * 40e-3), rel=1e-12)
    assert formula == pytest.approx(0.998043, abs=1e-6)
    assert solved == pytest.approx(formula, abs=2e-6)


def test_the_resonances_are_the_open_ended_line_and_q_the_ceramic_loss(ran):
    (printed,) = _printed(ran, "Resonances and Q")
    frequency, q, share, from_share = (printed[i : i + 3] for i in (0, 3, 6, 9))
    roots = _open_line_resonances(3)
    assert roots / 1e9 == pytest.approx([2.14197, 4.19063, 5.96308], abs=1e-5)
    # The page says "within 5 kHz".
    assert frequency * 1e9 == pytest.approx(roots, abs=5e3)
    assert q == pytest.approx([1251.7, 1379.5, 1748.3], abs=0.05)
    assert from_share == pytest.approx(1 / (1e-3 * share), rel=1e-7)
    assert q == pytest.approx(from_share, rel=1e-6)
    assert "True" in ran[3].printed


def test_two_equal_ports_halve_the_coupling_q(ran):
    (printed,) = _printed(ran, "Coupling Q of a port")
    q_p1, q_p2, q, parallel = (printed[i : i + 2] for i in (0, 2, 4, 6))
    assert q_p1 == pytest.approx([665.28, 792.96], abs=0.01)
    assert q_p2 == pytest.approx(q_p1, rel=1e-9)
    assert parallel == pytest.approx(q_p1 / 2, rel=1e-9)
    assert q == pytest.approx(parallel, rel=1e-5)


def test_the_capacitance_is_two_plate_capacitors(ran):
    (printed,) = _printed(ran, "Capacitance")
    # eps_0 is the page's own constant, written 8.8541878128e-12.
    solved, formula = printed[-2:]
    hand = 8.8541878128e-12 * (6e-3 / 1e-3) * (20e-3 + 10.0 * 20e-3) * 1e12
    assert formula == pytest.approx(hand, rel=1e-12)
    assert formula == pytest.approx(11.68753, abs=1e-5)
    assert solved == pytest.approx(formula, rel=1e-8)


def test_the_checks_hold_on_the_first_solve(ran):
    (printed,) = _printed(ran, "Check your results")
    balance, (order_2, order_3) = printed[:7], printed[-2:]
    assert balance == pytest.approx(1.0, abs=1e-6)  # the page: "better than 1e-6"
    (band,) = _printed(ran, "S-parameters over a band")
    assert order_2 == pytest.approx(band[7 + 3], abs=1e-8)
    # The page: "differ in the sixth digit".
    assert 1e-7 < abs(order_3 - order_2) < 1e-5
    (check,) = [e for e in ran if e.section == "Check your results"]
    assert check.printed.splitlines()[-2] == "True"
