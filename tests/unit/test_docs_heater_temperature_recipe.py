"""The heater-temperature recipe runs and prints what the page shows.

docs/recipes/heater-temperature.md runs nothing at docs build time. This runs
its blocks in order, as a reader does, on the repository's own
docs/tutorials/data/heater_w2.msh, and checks every printed number against a
hand value: the strip resistance 1/(sigma w t), the Joule power I^2 R', a
uniform source equal to the Joule heat of a straight strip, and a rise that
does not move when only a boundary's ambient does.
"""

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "heater-temperature.md"
SIGMA, WIDTH, THICKNESS = 2.3e6, 2e-6, 0.14e-6  # the page's TiN strip
CURRENT, LENGTH = 6.74e-3, 320e-6


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("heater_temperature"))


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def _lines(ran, section: str) -> list[np.ndarray]:
    """The numbers of each line ``section``'s block printed."""
    (example,) = [example for example in ran if example.section == section]
    return [dp.numbers(line) for line in example.printed.splitlines()]


def test_the_page_downloads_the_heater_section(page):
    first = dp.examples(page)[0]
    assert dp.MESH_URL.replace("line.msh", "heater_w2.msh") in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_end_faces_are_insulated_and_the_mapping_leaves_thermal_whole(ran):
    (example,) = [e for e in ran if e.section == "Materials and boundaries"]
    lines = example.printed.splitlines()
    assert lines[1:] == ["('front', 'back')", "()"]


def test_the_strip_resistance_is_one_over_sigma_times_its_area(ran):
    solved, formula, total = (n[-1] for n in _lines(ran, "The heater's resistance"))
    hand = 1 / (SIGMA * WIDTH * THICKNESS)
    assert formula == pytest.approx(hand, rel=1e-6)
    assert solved == pytest.approx(hand, rel=1e-6)
    assert total == pytest.approx(hand * LENGTH, abs=0.005)


def test_the_joule_power_is_i_squared_r_and_the_rise_is_absolute_minus_300(ran):
    _, (power,), (_, i2r), (core, heater), (temperature,) = _lines(
        ran, "The temperature at a drive"
    )
    hand = CURRENT**2 / (SIGMA * WIDTH * THICKNESS)
    assert power == pytest.approx(hand, rel=1e-5)
    assert i2r == pytest.approx(hand, rel=1e-5)
    assert temperature == pytest.approx(300.0 + core, abs=2e-3)
    # the heater sits above the core, nearer the heat: it is warmer
    assert heater > 2 * core > 0


def test_a_uniform_source_is_the_joule_heat_of_a_straight_strip(ran):
    (core,) = _printed(ran, "Heat without a current")[-1:]
    (joule,) = _printed(ran, "The temperature at a drive")[-3:-2]
    assert core == pytest.approx(joule, abs=1e-3)


def test_a_warmer_ambient_moves_the_temperature_not_the_rise(ran):
    rise, warmer_mk = _printed(
        ran, "A boundary that sets the temperature, not the rise"
    )
    (joule,) = _printed(ran, "The temperature at a drive")[-3:-2]
    assert rise == pytest.approx(joule, abs=1e-3)
    assert 0 < warmer_mk < 1.0


def test_the_heat_leaves_through_the_sinks_and_crosses_the_cut(ran):
    (sink, up), (*_, balance), (*_, down) = _lines(ran, "Where the heat goes")
    (power,) = _printed(ran, "The temperature at a drive")[-6:-5]
    # read from the discrete balance: the sinks close to the solver's residual
    assert balance < 1e-9
    assert sink + up == pytest.approx(power, rel=1e-5)
    assert 0 < up < 1e-4 * power  # the air takes a few 1e-5 of the heat
    # no heat is made below the cut, so the sink's heat crosses it
    assert down == pytest.approx(sink, rel=1e-9)
