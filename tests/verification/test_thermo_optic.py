"""The first-order thermo-optic shift against mode re-solves (ADR 0174, 0176).

A heater cross-section shaped like femwell's TiN heater (a silicon wafer
under a 2 um oxide box, a 500 x 220 nm silicon core, a 2 um x 140 nm heater
2 um above it, oxide cladding): the wafer sinks the heat, and the wafer and
heater are excluded from electromagnetic solves so the port's first mode is
the core's.

- At the drive that gives a pi shift, the first-order shift is within 1e-3
  of a `gfem.PortMode` re-solve on `gfem.Model.with_temperature_rise`
  (1.4e-4 on the femwell mesh, thermal probes/08).
- A separate optical window around the core, in the same coordinates, with
  the rise sampled across meshes, gives the same shift within 1e-3 as the
  EM-excluded section (1e-4 on the femwell mesh).
"""

import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem

WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
N_SI, N_OX = 3.4777, 1.444
HALF = 6.0  # um, half width of the section
#: One slab thickness for both meshes: the window's elements lie in the section's.
THICKNESS = 0.2e-6
#: The heater length of thermal probes/08: P_pi over it is 70 W/m on femwell's heater.
LENGTH = 320e-6
CORE = (-0.25, 0.0, 0.25, 0.22, "core", 0.04)
SECTION = [
    (-HALF, -2.5, HALF, -2.0, "wafer", None),
    (-HALF, -2.0, HALF, 0.0, "box", None),
    (-HALF, 0.0, HALF, 2.8, "clad", None),
    CORE,
    (-1.0, 2.22, 1.0, 2.36, "heater", 0.1),
]
WINDOW = [(-1.75, -1.0, 1.75, 1.22, "clad", None), CORE]
OXIDE = {
    "eps_r": N_OX**2,
    "thermal_conductivity": 1.38,
    "thermo_optic_coefficient": 1e-5,
}
MATERIALS = {
    "wafer": gfem.Material(thermal_conductivity=148.0),
    "box": gfem.Material(**OXIDE),
    "clad": gfem.Material(**OXIDE),
    "core": gfem.Material(
        eps_r=N_SI**2, thermal_conductivity=90.0, thermo_optic_coefficient=1.86e-4
    ),
    "heater": gfem.Material(thermal_conductivity=28.0),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("heater") / "section.msh"
    bottom = [(lambda p: abs(p[1] + 2.5) < 1e-9, "bottom")]
    mesh = tc.section(path, SECTION, 0.6, bottom, thickness=THICKNESS)
    return gfem.Model(
        mesh,
        materials=MATERIALS,
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        ports=[gfem.WavePort("p", "front")],
        excluded={"em": ["wafer", "heater"]},
    )


def _steady(model, power_per_length):
    heat = gfem.HeatSource.uniform("heater", power_per_length=power_per_length)
    return gfem.solve(model, gfem.SteadyThermal(heat))


@pytest.fixture(scope="module")
def per_watt(model):
    """The rise and the shift at 1 W/m."""
    steady = _steady(model, 1.0)
    shift = gfem.thermo_optic_shift(model, steady.rise, port="p", frequency=FREQUENCY)
    return steady, shift


def test_first_order_matches_a_re_solve_at_the_p_pi_drive(model, per_watt):
    steady, shift = per_watt
    p_pi = float(np.pi * steady.power_per_length / shift.phase_per_length)
    # W; femwell's heater (a wider section) needs 22.4 mW
    assert 15e-3 < p_pi < 35e-3
    # a heater of length LENGTH turns the phase by pi at P_pi / LENGTH per length
    driven = _steady(model, p_pi / LENGTH)
    first = gfem.thermo_optic_shift(model, driven.rise, port="p", frequency=FREQUENCY)
    np.testing.assert_allclose(
        float(first.phase_per_length) * LENGTH, np.pi, rtol=1e-10
    )
    cold = gfem.solve(model, gfem.PortMode("p", FREQUENCY))
    heated = model.with_temperature_rise(driven.rise)
    hot = gfem.solve(heated, gfem.PortMode("p", FREQUENCY))
    resolved = float(np.real(hot.n_eff[0] - cold.n_eff[0]))
    # the second-order term: -1.4e-4 (-1.42e-4 measured, P_pi 21.31 mW)
    assert float(first.delta_n_eff) == pytest.approx(resolved, rel=1e-3)


def test_a_window_model_matches_the_excluded_section(tmp_path_factory, per_watt):
    steady, shift = per_watt
    path = tmp_path_factory.mktemp("window") / "window.msh"
    mesh = tc.section(path, WINDOW, 0.6, [], thickness=THICKNESS)
    window = gfem.Model(
        mesh,
        materials={k: MATERIALS[k] for k in ("clad", "core")},
        ports=[gfem.WavePort("p", "front")],
    )
    other = gfem.thermo_optic_shift(window, steady.rise, port="p", frequency=FREQUENCY)
    # (1.2e-6 measured)
    assert float(other.phase_per_length) == pytest.approx(
        float(shift.phase_per_length), rel=1e-3
    )
