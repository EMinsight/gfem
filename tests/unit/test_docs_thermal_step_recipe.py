"""The thermal step-response recipe runs and prints what the page shows.

docs/recipes/thermal-step-response.md runs nothing at docs build time. This
runs its blocks in order on the repository's own heater section
(docs/tutorials/data/heater_w2.msh) and checks what a linear step response
must do: the fall mirrors the rise, the run returns to the sink temperature,
a single slow mode gives rise time / time constant near ln 9, a short trace
warns, and backward Euler lags TR-BDF2 at the same step.
"""

import math

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "thermal-step-response.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("thermal_step"))


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def _lines(ran, section: str) -> list[np.ndarray]:
    """The numbers of each line ``section``'s block printed."""
    (example,) = [example for example in ran if example.section == section]
    return [dp.numbers(line) for line in example.printed.splitlines()]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_run_heats_to_near_pi_and_cools_back_to_the_sink(ran):
    shape, (core_on, core_end), (heater_on,), (phase_on,) = _lines(
        ran, "A step on and off"
    )
    assert shape[0] == 321  # steps + 1 times
    assert core_end == pytest.approx(300.0, abs=0.01)
    assert heater_on > core_on > 300.0
    assert phase_on == pytest.approx(math.pi, rel=0.02)  # 70 W/m is about P_pi


def test_the_kept_field_is_the_rise_part_way_up(ran):
    kept, (*_, halfway) = _lines(ran, "Fields at chosen times")
    assert list(kept) == [40.0, 160.0]
    core_on = _lines(ran, "A step on and off")[1][0]
    assert 0.9 * (core_on - 300.0) < halfway < core_on - 300.0


def test_the_fall_mirrors_the_rise_and_one_mode_dominates(ran):
    (tau_on, tau_off), (*_, rise, ratio) = _lines(ran, "Rise time and time constant")
    assert tau_off == pytest.approx(tau_on, rel=2e-3)
    assert rise / tau_on == pytest.approx(ratio, abs=0.01)
    assert 2.0 < ratio < math.log(9.0)


def test_a_short_trace_warns(ran):
    (example,) = [e for e in ran if e.section == "A trace that has not settled"]
    assert example.printed.strip() == "['UnsettledTraceWarning']"


def test_backward_euler_lags_tr_bdf2(ran):
    (tau_euler,) = _printed(ran, "The time scheme")
    tau = _lines(ran, "Rise time and time constant")[0][0]
    assert tau < tau_euler < 1.1 * tau
