"""The thermo-optic phase recipe runs and prints what the page shows.

docs/recipes/thermo-optic-phase.md runs nothing at docs build time. This runs
its blocks in order on the repository's own heater section
(docs/tutorials/data/heater_w2.msh) and its extrusion
(docs/tutorials/data/heater_straight.msh, tools/heater_section_mesh.py), and
checks the printed numbers: P_pi = pi / (phase per watt), the 3D strip's
resistance R' L, and the route phase of a z-invariant rise against the
section's phase per length times the length.
"""

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "thermo-optic-phase.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("thermo_optic_phase"))


def _printed(ran, section: str) -> np.ndarray:
    (example,) = [example for example in ran if example.section == section]
    return dp.numbers(example.printed)


def _lines(ran, section: str) -> list[np.ndarray]:
    """The numbers of each line ``section``'s block printed."""
    (example,) = [example for example in ran if example.section == section]
    return [dp.numbers(line) for line in example.printed.splitlines()]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_p_pi_is_pi_over_the_phase_per_watt(ran):
    numbers = _printed(ran, "The phase per watt on a cross-section")
    per_length, per_watt, p_pi_mw = numbers[-3:]
    assert per_watt == pytest.approx(per_length, rel=1e-3)  # at 1 W/m
    assert p_pi_mw == pytest.approx(1e3 * np.pi / per_watt, rel=1e-3)


def test_the_neighbour_picks_up_a_small_positive_share(ran):
    (crosstalk,) = _printed(ran, "Thermal crosstalk on a second waveguide")
    assert 0 < crosstalk < 0.2


def test_a_z_invariant_route_is_the_section_phase_times_its_length(ran):
    lines = _lines(ran, "Along a route through a 3D heater")
    (r_3d, r_section), route, section, p_pi_mw = (
        lines[0],
        *(n[-1] for n in lines[1:]),
    )
    assert r_3d == pytest.approx(r_section, rel=1e-4)
    # the 3D mesh is coarser across the section (1.4e-4 measured)
    assert route == pytest.approx(section, rel=1e-3)
    (_, _, p_pi_section) = _printed(ran, "The phase per watt on a cross-section")[-3:]
    assert p_pi_mw == pytest.approx(p_pi_section, rel=1e-3)
