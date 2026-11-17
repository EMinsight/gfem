"""Thermal solves against convergent closed forms (ADR 0180).

Steady: a strip heater on a slab. Transient: gsim's 1D slab (below).

A strip heater on top of a slab of depth H, with a heat sink at the bottom
and an insulated top, heats the top surface as (per unit length, strip of
width w, flux F, conductivity k)

    T(x) - T_sink = F / (pi k) int_{-w/2}^{w/2} ln coth(pi |x - s| / (4 H)) ds,

which decays far from the strip as exp(-pi |x| / (2 H)): decay length
2 H / pi. The strip is an interior-free `gfem.HeatFlux` surface on the top
edge, so the cross-section slab carries no thickness-model error.
"""

import math
import warnings

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


# --- transient: gsim's 1D slab (ADRs 0164, 0166, 0180) -------------------------
#
# Depth SL, sink at the bottom, insulated top, uniform heat SQ switched on at
# t = 0: the top rises as T_s - sum_n b_n sin(l_n SL) exp(-a l_n^2 t), with
# l_n = (2n + 1) pi / (2 SL), b_n = 2 SQ / (SL SK l_n^3), a = SK / (rho c_p),
# T_s = SQ SL^2 / (2 SK), and the slowest mode decays with
# tau1 = 4 rho c_p SL^2 / (pi^2 SK).

SL, SK, RHO, CP, SQ = 10.0 * UM, 1.4, 1600.0, 1000.0, 1e11
TAU1 = 4 * RHO * CP * SL**2 / (math.pi**2 * SK)  # 46.318 us
T_SINK = 300.0
T_STEADY = SQ * SL**2 / (2 * SK)  # top rise, 3.5714 K


def _analytic_top(t: np.ndarray, terms: int = 400) -> np.ndarray:
    a = SK / (RHO * CP)
    out = np.full(np.shape(t), T_STEADY)
    for n in range(terms):
        ln = (2 * n + 1) * math.pi / (2 * SL)
        bn = 2 * SQ / (SL * SK * ln**3)
        out = out - bn * math.sin(ln * SL) * np.exp(-a * ln**2 * np.asarray(t))
    return out


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    edges = [
        (lambda p: abs(p[1]) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - SL / UM) < 1e-9, "top"),
    ]
    mesh = tc.section(path, [(0.0, 0.0, 1.0, SL / UM, "slab", None)], 0.5, edges)
    material = gfem.Material(
        thermal_conductivity=SK, mass_density=RHO, specific_heat=CP
    )
    return gfem.Model(
        mesh,
        materials={"slab": material},
        boundaries=[gfem.FixedTemperature("bottom", temperature=T_SINK)],
    )


def _top(model):
    return (0.5 * UM, SL, model.mesh.section_thickness / 2)


def _step(model, duration, steps, **options):
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("slab", power_density=SQ),
        duration=duration,
        steps=steps,
        traces={"top": gfem.Trace.point(_top(model))},
    )
    options = gfem.SolverOptions(order=2, linear_solver="direct", **options)
    result = gfem.solve(model, problem, options=options)
    return np.asarray(result.time), np.asarray(result.traces["top"]) - T_SINK, result


def test_the_first_eigenvalue_gives_tau1(slab):
    from scipy.sparse.linalg import eigsh

    from gfem._engine.thermal import TransientThermalRule

    problem = gfem.TransientThermal(duration=1.0, steps=1)
    rule = TransientThermalRule(slab, problem, gfem.SolverOptions(order=2))
    stiffness, capacity, free = rule.matrices(slab)
    k = stiffness[free][:, free]
    c = capacity[free][:, free]
    lowest = eigsh(k, k=1, M=c, sigma=0.0, which="LM", return_eigenvectors=False)
    # gate 1e-6 (ADR 0180; 2.8e-8 measured, 588 DOF)
    assert 1 / lowest[0] == pytest.approx(TAU1, rel=1e-6)


def _decay_tau(t, rise):
    window = (t >= 2 * TAU1) & (t <= 4 * TAU1)
    slope = np.polyfit(t[window], np.log(T_STEADY - rise[window]), 1)[0]
    return -1 / slope


def test_the_late_decay_slope_of_tr_bdf2_gives_tau1(slab):
    t, rise, _ = _step(slab, 5 * TAU1, 100)
    # gate 1e-3 (ADR 0180; 1.0e-4 measured)
    assert _decay_tau(t, rise) == pytest.approx(TAU1, rel=1e-3)


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_the_late_decay_slope_of_the_theta_scheme_gives_tau1(slab, theta):
    t, rise, _ = _step(slab, 5 * TAU1, 500, time_scheme="theta", theta=theta)
    # gate 1 % (gsim's, ADR 0180; 5.0e-3 measured at theta 1, backward
    # Euler's first-order lag, and 8.4e-6 at theta 0.5)
    assert _decay_tau(t, rise) == pytest.approx(TAU1, rel=1e-2)


def test_the_trace_times_match_the_analytic_trace(slab):
    t, rise, _ = _step(slab, 5 * TAU1, 500)
    fine = np.linspace(0.0, 5 * TAU1, 200001)
    exact = _analytic_top(fine)
    with warnings.catch_warnings():
        # 5 tau1 is short of settled: both are normalised over the same window
        warnings.simplefilter("ignore", gfem.UnsettledTraceWarning)
        pairs = [
            (gfem.rise_time(t, rise), gfem.rise_time(fine, exact)),
            (gfem.time_constant(t, rise), gfem.time_constant(fine, exact)),
        ]
    for measured, analytic in pairs:
        # gate 1e-3 (ADR 0180; 6.9e-8 and 1.9e-6 measured)
        assert measured == pytest.approx(analytic, rel=1e-3)


def test_the_final_temperature_is_the_steady_one(slab):
    _, _, result = _step(slab, 10 * TAU1, 100)
    steady = gfem.solve(
        slab,
        gfem.SteadyThermal(gfem.HeatSource.uniform("slab", power_density=SQ)),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    top = _top(slab)
    final = float(result.rise(result.kept_times[-1]).at(top))
    # gate 5e-3 (ADR 0180; 4.7e-5 measured)
    assert final == pytest.approx(float(steady.rise.at(top)), rel=5e-3)
