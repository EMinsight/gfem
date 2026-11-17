"""Thermal solves against convergent closed forms (ADR 0180).

A strip heater on top of a slab of depth H, with a heat sink at the bottom
and an insulated top, heats the top surface as (per unit length, strip of
width w, flux F, conductivity k)

    T(x) - T_sink = F / (pi k) int_{-w/2}^{w/2} ln coth(pi |x - s| / (4 H)) ds,

which decays far from the strip as exp(-pi |x| / (2 H)): decay length
2 H / pi. The strip is an interior-free `gfem.HeatFlux` surface on the top
edge, so the cross-section slab carries no thickness-model error.
"""

import math

import numpy as np
import pytest
import thermal_cases as tc
from scipy.integrate import quad

import gfem

UM = 1e-6
H, L, W, K = 4.5, 40.0, 0.2, 1.0  # um, um, um, W/(m K)
FLUX = 1e6  # W/m^2 on the strip
XS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])  # um


def _exact(x: float) -> float:
    """The closed-form rise at x um on the top surface, K."""

    def integrand(s):
        return math.log(1 / math.tanh(math.pi * abs(x - s) / (4 * H)))

    value = quad(
        integrand, -W / 2, W / 2, limit=200, points=[x] if abs(x) < W else None
    )
    return FLUX * value[0] * UM / (math.pi * K)


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    path = tmp_path_factory.mktemp("strip") / "strip.msh"
    rects = [
        (-L, -H, -W / 2, 0.0, "oxide", None),
        (-W / 2, -H, W / 2, 0.0, "oxide", 0.05),
        (W / 2, -H, L, 0.0, "oxide", None),
    ]
    edges = [
        (lambda p: abs(p[1]) < 1e-9 and abs(p[0]) < W / 2, "heater"),
        (lambda p: abs(p[1] + H) < 1e-9, "bottom"),
    ]
    mesh = tc.section(path, rects, 0.5, edges)
    model = gfem.Model(
        mesh,
        materials={"oxide": gfem.Material(thermal_conductivity=K)},
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.HeatFlux("heater", flux=FLUX),
        ],
    )
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    result = gfem.solve(model, gfem.SteadyThermal(), options=options)
    z = mesh.section_thickness / 2
    points = np.stack([XS * UM, np.zeros_like(XS), np.full_like(XS, z)], axis=1)
    return np.asarray(result.temperature.at(points)) - 300.0


def test_a_strip_heater_follows_the_ln_coth_profile(strip):
    exact = np.array([_exact(x) for x in XS])
    # gate 2e-3 (ADR 0180; 2.4e-4 measured)
    np.testing.assert_allclose(strip, exact, rtol=2e-3)


def test_the_far_field_decay_length_is_2h_over_pi(strip):
    # the log slope between 8 and 12 um (8.4e-4 measured); gate 2 %, gsim's
    # (a fit-window model error, not a discretisation one)
    far = XS >= 8.0
    slope = np.polyfit(XS[far], np.log(strip[far]), 1)[0]
    assert -1 / slope == pytest.approx(2 * H / math.pi, rel=2e-2)
