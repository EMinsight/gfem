"""Floquet ports against closed forms: the Fresnel slab F1, normal and oblique.

F1 (`periodic_cases.f1`: an eps 4 slab on an eps 2.25 substrate, air above,
Floquet ports on both open faces) at p = 3 with h about lambda/10 in each
medium, TE and TM, against the tangential-E Airy form in ADR 0147's lattice
frame: the complex S entries, the cross-polarized entries and the power
defect of each excited column. At normal incidence, at conical incidence
(40 degrees, azimuth 30) under the Bloch wave vector, and on the Brewster
cell (air / eps 4 / air at atan 2), whose TM reflection vanishes.
"""

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
