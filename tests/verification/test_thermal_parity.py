"""Parity with gsim's 2D thermal path on gsim's own mesh (ADR 0180).

gsim's femwell TiN heater (femwell's metal heater phase shifter example,
after Jacques et al. 2019): a 500 x 220 nm silicon core (k 90 W/(m K)) on a
2 um buried oxide over a 0.5 um silicon wafer (k 148) held at the heat sink,
a 2 um x 140 nm TiN heater (k 28, sigma 2.3e6 S/m) 2 um above the core,
oxide (k 1.38) up to 2.8 um, 16 um wide. `data/gsim_femwell_tin.msh` is
the cross-section mesh gsim writes for it (msh 2.2, 5,219 triangles), so
both codes solve on the same triangles; gfem reads it as a slab.

- P_pi through an optical window around the core (gsim's window, the core
  plus 1.5 um aside and 1 um above and below) against gsim's frozen value
  at 1e-3 (6e-6 measured on this mesh with a mode re-solve, thermal
  probes/13); against femwell's own example output at 3 %.
- The phase step response at P_pi, theta = 1 over gsim's time grid: its
  10-90 % rise time and its time constant at 1e-4 (ADR 0180's 2e-3,
  tightened: 2e-6 and 7e-6 measured).

Measured (ostau, 2 threads): P_pi 22.3920 mW (2.3e-4 from gsim's re-solve,
gsim's own first order sits 1.9e-4 off it), rise time 18.78643 us, time
constant 8.48886 us; the whole file about 20 s.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem

MESH = Path(__file__).parent / "data" / "gsim_femwell_tin.msh"

# Frozen gsim numbers on MESH: gsim 5ecbb6a (femwell 0.1.12, scikit-fem
# 12.0.2), gsim's tests/thermal femwell_sim() at its default precision
# (thermal degree 2, optical order 2, thermal window -8..8 um, heater
# length 320 um), run by .scratch/thermal/probes/13/gsim_femwell_tin.py
# (gsim_femwell_tin.json). P_pi from gsim's mode re-solve; the transient
# steps 400 times over 60 us at the P_pi drive, first-order phase.
GSIM_P_PI = 22.38686e-3  # W
GSIM_RISE_TIME_THETA_1 = 18.7864e-6  # s, 10-90 %
GSIM_TIME_CONSTANT_THETA_1 = 8.4888e-6  # s, 1 - 1/e
# theta = 0.5 on the same grid: frozen, not gated (the slab gates the
# schemes' accuracy, tests/verification/test_thermal_closed_forms.py)
GSIM_RISE_TIME_THETA_HALF = 18.6126e-6
GSIM_TIME_CONSTANT_THETA_HALF = 8.3968e-6
#: femwell's example output, 3.786 rad at 7.4 mA over 320 um.
FEMWELL_P_PI = 22.6e-3  # W

WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
DURATION, STEPS = 60e-6, 400
#: gsim's heater length: the phase over it at P_pi is pi.
LENGTH = 320e-6
#: One slab thickness for the section and the window: the window's
#: elements lie in the section's.
THICKNESS = 0.5e-6

SILICON = {"mass_density": 2330.0, "specific_heat": 711.0}
OXIDE = {
    "eps_r": 1.444**2,
    "thermal_conductivity": 1.38,
    "thermo_optic_coefficient": 1e-5,
    "mass_density": 2203.0,
    "specific_heat": 709.0,
}
CORE = {
    "eps_r": 3.4777**2,
    "thermal_conductivity": 90.0,
    "thermo_optic_coefficient": 1.86e-4,
    **SILICON,
}
MATERIALS = {
    "core": gfem.Material(**CORE),
    "oxide": gfem.Material(**OXIDE),
    "si_wafer": gfem.Material(thermal_conductivity=148.0, **SILICON),
    "heater": gfem.Material(
        conductivity=2.3e6,
        thermal_conductivity=28.0,
        mass_density=5240.0,
        specific_heat=598.0,
    ),
}
#: gsim's optical window (um): the core with 1.5 um aside, 1 um above and below.
WINDOW = [
    (-1.75, -1.0, 1.75, 1.22, "oxide", None),
    (-0.25, 0.0, 0.25, 0.22, "core", 0.04),
]


@pytest.fixture(scope="module")
def section():
    mesh = gfem.Mesh.from_gmsh_cross_section(MESH, thickness=THICKNESS)
    model = gfem.Model(
        mesh,
        materials=MATERIALS,
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    dc = gfem.solve(model, gfem.DCConduction({"heater": "front"}, ground="back"))
    return model, dc


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    path = tmp_path_factory.mktemp("parity") / "window.msh"
    mesh = tc.section(path, WINDOW, 0.6, [], thickness=THICKNESS)
    return gfem.Model(
        mesh,
        materials={k: MATERIALS[k] for k in ("oxide", "core")},
        ports=[gfem.WavePort("p", "front")],
    )


@pytest.fixture(scope="module")
def p_pi(section, window):
    model, dc = section
    heat = gfem.joule_heating(dc, power_per_length=1.0)
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    shift = gfem.thermo_optic_shift(window, steady.rise, port="p", frequency=FREQUENCY)
    return float(np.pi * steady.power_per_length / shift.phase_per_length)


def test_p_pi_matches_gsim_on_its_own_mesh(p_pi):
    # 2.3e-4 measured
    assert p_pi == pytest.approx(GSIM_P_PI, rel=1e-3)


def test_p_pi_is_within_femwells_band(p_pi):
    assert p_pi == pytest.approx(FEMWELL_P_PI, rel=3e-2)


def test_the_step_response_matches_gsims_rise_time_and_time_constant(
    section, window, p_pi
):
    model, dc = section
    shift = gfem.thermo_optic_shift(window, None, port="p", frequency=FREQUENCY)
    problem = gfem.TransientThermal(
        gfem.joule_heating(dc, power_per_length=p_pi / LENGTH),
        duration=DURATION,
        steps=STEPS,
        traces={"phase": shift.trace()},
    )
    options = gfem.SolverOptions(time_scheme="theta", theta=1.0)
    result = gfem.solve(model, problem, options=options)
    t = np.asarray(result.time)
    phase = np.asarray(result.traces["phase"]) * LENGTH
    # 18.78643 and 8.48886 us measured
    rise = gfem.rise_time(t, phase)
    tau = gfem.time_constant(t, phase)
    assert rise == pytest.approx(GSIM_RISE_TIME_THETA_1, rel=1e-4)
    assert tau == pytest.approx(GSIM_TIME_CONSTANT_THETA_1, rel=1e-4)
