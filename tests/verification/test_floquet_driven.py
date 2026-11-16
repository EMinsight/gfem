"""Floquet ports against closed forms: the Fresnel slab F1 at normal incidence.

F1 (`periodic_cases.f1`: an eps 4 slab on an eps 2.25 substrate, air above,
Floquet ports on both open faces) at p = 3 with h about lambda/10 in each
medium, TE and TM, against the tangential-E Airy form in ADR 0147's lattice
frame: the complex S entries, the cross-polarized entries and the power
defect of each excited column.
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
