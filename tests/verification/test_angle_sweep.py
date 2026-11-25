"""Angle sweeps against closed forms and frozen RCWA, p = 3, direct solver.

F1 fine through one sweep over a plain and three Bloch incidences (planar and
conical) against the Airy form, and F2 fine over the G0, G20 and G20c
incidences against the frozen RCWA efficiencies, ports declared for every
incidence with `floquet_ports(..., wave_vector=ks)`.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

OPTS = gfem.SolverOptions(order=3, linear_solver="direct")
TOP = "z_max"
F0 = pc.F1_FREQUENCY
DEG = math.pi / 180.0
#: Planar specular cross-polarization (1.1e-5 measured); conical at least 1e-2
#: (0.117 measured).
PLANAR_CROSS = 1e-4
CONICAL_CROSS = 1e-2

S7_INCIDENCES = ((0.0, 0.0), (20.0, 30.0), (40.0, 30.0), (40.0, 120.0))


@pytest.fixture(scope="module")
def f1_sweep(tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("s7"), pc.F1_FINE_CELLS)
    model = pc.f1(mesh)
    theta = [t * DEG for t, _ in S7_INCIDENCES]
    phi = [p * DEG for _, p in S7_INCIDENCES]
    return model, dif.angle_sweep(
        model, [F0], surface=TOP, theta=theta, phi=phi, options=OPTS
    )


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_s7_each_member_matches_airy(f1_sweep, pol):
    _, sweep = f1_sweep
    other = "TM" if pol == "TE" else "TE"
    for member, theta in zip(sweep, sweep.theta, strict=True):
        kt = 2 * math.pi * F0 / pc.C0 * math.sin(theta)
        r, t = pc.f1_airy(pol, kt=kt)
        s = member.s
        assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_S_TOL
        assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_S_TOL
        for side in ("R", "T"):
            cross = abs(complex(s[f"{side}_{other}", f"R_{pol}"][0]))
            assert cross <= pc.F1_CROSS_TOL
        column = np.asarray(s.values)[0, :, s.excited.index(f"R_{pol}")]
        assert abs(1 - np.sum(np.abs(column) ** 2)) <= pc.F1_POWER_TOL


def test_s7_power_balance_closes_per_member(f1_sweep):
    model, sweep = f1_sweep
    bal = dif.power_balance(sweep, TOP, polarization="unpolarized", model=model)
    assert np.abs(np.asarray(bal.absorptance)).max() <= pc.F1_POWER_TOL


def test_s7_unpolarized_efficiencies_do_not_depend_on_phi(f1_sweep):
    model, sweep = f1_sweep
    eff = dif.efficiencies(sweep, TOP, polarization="unpolarized", model=model)
    # members 2 and 3 are 40 degrees at phi = 30 and 120
    difference = np.abs(np.asarray(eff.values[2]) - np.asarray(eff.values[3]))
    assert difference.max() <= 1e-5


F2_CASE_NAMES = ("G0", "G20", "G20c")
F2_ORDERS = tuple((m, 0) for m in (-2, -1, 0, 1, 2))


@pytest.fixture(scope="module")
def f2_sweep(tmp_path_factory):
    mesh = pc.f2_mesh(tmp_path_factory.mktemp("s8"), pc.F2_FINE_CELLS)
    base = pc.f2(mesh)
    theta = [pc.F2_CASES[c][0] * DEG for c in F2_CASE_NAMES]
    phi = [pc.F2_CASES[c][1] * DEG for c in F2_CASE_NAMES]
    ks = [
        base.bloch_wave_vector(TOP, F0, theta=a, phi=b)
        for a, b in zip(theta, phi, strict=True)
    ]
    ports = [
        *dif.floquet_ports(base, TOP, F0, wave_vector=ks, wave_vector_frequency=F0),
        *dif.floquet_ports(base, "z_min", F0, wave_vector=ks, wave_vector_frequency=F0),
    ]
    model = base.replace(ports=ports)
    sweep = dif.angle_sweep(
        model, [F0], surface=TOP, theta=theta, phi=phi, options=OPTS
    )
    return model, sweep


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("index", range(len(F2_CASE_NAMES)))
def test_s8_efficiencies_match_rcwa_and_the_balance_closes(f2_sweep, pol, index):
    model, sweep = f2_sweep
    case = F2_CASE_NAMES[index]
    eff = dif.efficiencies(sweep, TOP, polarization=pol, model=model)
    reference = pc.F2_RCWA[case][pol]
    error = 0.0
    for (face, m), value in reference.items():
        surface = TOP if face == "R" else "z_min"
        got = float(np.asarray(eff[surface, (m, 0)])[index, 0])
        error = max(error, abs(got - value))
    assert error <= pc.F2_EFFICIENCY_TOL
    bal = dif.power_balance(sweep, TOP, polarization=pol, model=model)
    assert abs(float(np.asarray(bal.absorptance)[index, 0])) <= pc.F2_POWER_TOL


@pytest.mark.parametrize("index", range(len(F2_CASE_NAMES)))
def test_s8_the_specular_cross_polarization_is_planar_or_conical(f2_sweep, index):
    model, sweep = f2_sweep
    jones = dif.jones(sweep, TOP, TOP, model=model)
    values = np.asarray(jones.values)[index, 0]
    cross = max(abs(values[0, 1]), abs(values[1, 0]))
    if F2_CASE_NAMES[index] == "G20c":
        assert cross >= CONICAL_CROSS
    else:
        assert cross <= PLANAR_CROSS
