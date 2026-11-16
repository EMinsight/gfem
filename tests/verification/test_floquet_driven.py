"""Floquet ports against closed forms: the Fresnel slab F1, normal and oblique.

F1 (`periodic_cases.f1`: an eps 4 slab on an eps 2.25 substrate, air above,
Floquet ports on both open faces) at p = 3 with h about lambda/10 in each
medium, TE and TM, against the tangential-E Airy form in ADR 0147's lattice
frame: the complex S entries, the cross-polarized entries and the power
defect of each excited column. At normal incidence, at conical incidence
(40 degrees, azimuth 30) under the Bloch wave vector, and on the Brewster
cell (air / eps 4 / air at atan 2), whose TM reflection vanishes; the
reference-plane offset against moving the faces. The lamellar grating F2
(`periodic_cases.f2`) at normal, oblique, conical and lossy incidence
against the frozen RCWA efficiencies, with its power balance and absorption.
"""

import dataclasses
import math
import warnings

import numpy as np
import periodic_cases as pc
import pytest

import gfem


@pytest.fixture(scope="module")
def f1_normal(tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("f1"), pc.F1_FINE_CELLS)
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"])
    return gfem.solve(pc.f1(mesh), problem, options=gfem.SolverOptions(order=3))


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_f1_at_normal_incidence_matches_airy(f1_normal, pol):
    other = "TM" if pol == "TE" else "TE"
    r, t = pc.f1_airy(pol)
    s = f1_normal.s
    assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_S_TOL
    assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_S_TOL
    for side in ("R", "T"):
        assert abs(complex(s[f"{side}_{other}", f"R_{pol}"][0])) <= pc.F1_CROSS_TOL
    column = np.asarray(s.values)[0, :, s.excited.index(f"R_{pol}")]
    assert abs(1 - np.sum(np.abs(column) ** 2)) <= pc.F1_POWER_TOL


# --- Bloch (oblique) incidence ---------------------------------------------------


def _oblique(eps_sub, theta, phi, cells, tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("f1"), cells)
    model = pc.f1(mesh, eps_sub=eps_sub)
    k = model.bloch_wave_vector("z_max", pc.F1_FREQUENCY, theta=theta, phi=phi)
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"], wave_vector=k)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))
    return result, float(np.linalg.norm(k))


@pytest.fixture(scope="module")
def f1_conical(tmp_path_factory):
    return _oblique(
        pc.F1_EPS_SUB, pc.F1_THETA, pc.F1_PHI, pc.F1_FINE_CELLS, tmp_path_factory
    )


@pytest.fixture(scope="module")
def brewster(tmp_path_factory):
    return _oblique(
        pc.BREWSTER_EPS_SUB, pc.BREWSTER_THETA, 0.0, pc.BREWSTER_CELLS, tmp_path_factory
    )


def _check_fresnel(result, kt, pol, eps_sub):
    other = "TM" if pol == "TE" else "TE"
    r, t = pc.f1_airy(pol, kt=kt, eps_sub=eps_sub)
    s = result.s
    assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_S_TOL
    assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_S_TOL
    for side in ("R", "T"):
        assert abs(complex(s[f"{side}_{other}", f"R_{pol}"][0])) <= pc.F1_CROSS_TOL
    column = np.asarray(s.values)[0, :, s.excited.index(f"R_{pol}")]
    assert abs(1 - np.sum(np.abs(column) ** 2)) <= pc.F1_POWER_TOL


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_f1_at_conical_incidence_matches_airy(f1_conical, pol):
    result, kt = f1_conical
    _check_fresnel(result, kt, pol, pc.F1_EPS_SUB)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_the_brewster_cell_matches_airy(brewster, pol):
    result, kt = brewster
    _check_fresnel(result, kt, pol, pc.BREWSTER_EPS_SUB)


def test_the_brewster_cell_reflects_no_tm(brewster):
    result, _ = brewster
    assert abs(complex(result.s["R_TM", "R_TM"][0])) <= pc.BREWSTER_TOL
    # TE does reflect there: |r|^2 = 0.155 (Airy)
    assert abs(complex(result.s["R_TE", "R_TE"][0])) > 0.3


# --- reference-plane offsets ------------------------------------------------------

#: F1 with its faces 0.40 um from the slab (model B), its ports offset 0.15
#: um back to the planes of the 0.25 um cell (model A); cells keep h. Their
#: S agree to 1.4e-6 (measured) under `pc.F1_S_TOL`.
F1_FAR_D = 0.40
F1_FAR_CELLS = (4, 6, 6, 5)


def test_offsets_move_the_reference_planes_as_the_geometry_does(
    f1_conical, tmp_path_factory
):
    near, _ = f1_conical
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("f1far"), F1_FAR_CELLS, d=F1_FAR_D)
    offset = (F1_FAR_D - pc.F1_D) * pc.UM
    ports = [dataclasses.replace(p, offset=offset) for p in pc.f1_ports()]
    model = pc.f1(mesh, ports)
    k = model.bloch_wave_vector(
        "z_max", pc.F1_FREQUENCY, theta=pc.F1_THETA, phi=pc.F1_PHI
    )
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"], wave_vector=k)
    far = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))
    assert far.s.ports == near.s.ports
    error = np.abs(np.asarray(far.s.values) - np.asarray(near.s.values))
    assert error.max() <= pc.F1_S_TOL


# --- F2: the lamellar grating against RCWA ----------------------------------------

#: Every order propagating in some F2 case, declared on both faces in both
#: polarizations (rows of orders evanescent on a face read 0).
F2_ORDERS = tuple((m, 0) for m in (-2, -1, 0, 1, 2))
#: max |eta - eta_ref| over every propagating order and incident
#: polarization, p = 3 on `pc.F2_FINE_CELLS` (TE 9e-6 - 8e-5, TM 1.3e-4
#: normal - 5.8e-4 oblique measured: TM is held by the field singularity at
#: the ridge edges, 4.3e-4 at G20 with 8 x cells and 4 in the ridge).
F2_EFFICIENCY_TOL = 1e-3
#: 1 - sum |S|^2 of each column on the lossless cases (2.8e-7 - 1.7e-6
#: measured).
F2_POWER_TOL = 1e-4
#: |(1 - sum |S|^2) - A_ref| on G20L (9e-6 TE, 1.6e-4 TM measured).
F2_ABSORPTION_TOL = 1e-3
#: |power_flow + sum_T |S|^2| through the substrate (4e-7 - 1.3e-6
#: measured on G0 and G20).
F2_POWER_FLOW_TOL = 1e-3


@pytest.fixture(scope="module")
def f2_fine(tmp_path_factory):
    return pc.f2_mesh(tmp_path_factory.mktemp("f2"), pc.F2_FINE_CELLS)


@pytest.fixture(scope="module", params=list(pc.F2_CASES))
def grating(request, f2_fine):
    theta, phi, tan_delta = pc.F2_CASES[request.param]
    model = pc.f2(f2_fine, pc.f2_ports(F2_ORDERS), tan_delta=tan_delta)
    k = model.bloch_wave_vector(
        "z_max", pc.F1_FREQUENCY, theta=math.radians(theta), phi=math.radians(phi)
    )
    problem = gfem.Driven(
        [pc.F1_FREQUENCY],
        excite=["R0,0_TE", "R0,0_TM"],
        wave_vector=k,
        keep_fields=[pc.F1_FREQUENCY],
    )
    with warnings.catch_warnings():
        # every propagating order is declared: the sums are complete
        warnings.simplefilter("error", gfem.UnrequestedOrderWarning)
        result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))
    return request.param, result, model


def _efficiencies(result, pol):
    """|S|^2 of column R0,0_{pol}, summed over output polarizations, by (face, m)."""
    s = result.s
    column = np.abs(np.asarray(s.values)[0, :, s.excited.index(f"R0,0_{pol}")]) ** 2
    out = {}
    for name, value in zip(s.ports, column, strict=True):
        m = int(name[1:].split(",")[0])
        out[name[0], m] = out.get((name[0], m), 0.0) + float(value)
    return out


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_the_grating_matches_rcwa(grating, pol):
    case, result, _ = grating
    eta = _efficiencies(result, pol)
    reference = pc.F2_RCWA[case][pol]
    # the RCWA table lists every propagating order: the others read 0
    propagating = {
        (name[0], order.order[0])
        for name, order in result.diffraction_orders.items()
        if bool(np.asarray(order.propagating)[0])
    }
    assert propagating == set(reference)
    error = max(abs(eta[key] - value) for key, value in reference.items())
    assert error <= F2_EFFICIENCY_TOL
    defect = 1 - sum(eta.values())
    if case == "G20L":
        assert abs(defect - pc.F2_ABSORPTION[pol]) <= F2_ABSORPTION_TOL
    else:
        assert abs(defect) <= F2_POWER_TOL


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_the_power_into_the_substrate_is_the_transmitted_efficiency(grating, pol):
    # the substrate holds the T face: the power leaving it is minus every
    # transmitted order's |S|^2 of the 1 W incident wave
    _, result, model = grating
    eta = _efficiencies(result, pol)
    transmitted = sum(v for (side, _), v in eta.items() if side == "T")
    field = result.e_field(pc.F1_FREQUENCY, f"R0,0_{pol}")
    power = float(gfem.power_flow(field, ("sub",), model=model))
    assert abs(power + transmitted) <= F2_POWER_FLOW_TOL
