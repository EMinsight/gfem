"""The cell response against held-out solves.

R8: the ridge of `periodic_cases` on one mesh, its width moved by a boundary
morph; 5 samples of the width over 0.17 to 0.33 um at f0, the substrate's TE
and TM excited, interpolated at 4 held-out midpoints and compared with a direct
solve there. The error of the 3-sample subset on the same midpoints shows the
stored derivative at work.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

UM = 1e-6
W0 = pc.RIDGE_W0
SAMPLES = [0.17, 0.21, 0.25, 0.29, 0.33]
HELD = [0.19, 0.23, 0.27, 0.31]
EXCITE = ["sub TE", "sub TM"]
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=2)
#: Max over the 8 specular entries of |S_interp - S_solve|.
TOL = 2e-3
#: The 3-sample subset's error over the 5-sample one.
RATIO = 8.0


@pytest.fixture(scope="module")
def ridge(tmp_path_factory):
    mesh = pc.ridge_mesh(tmp_path_factory.mktemp("ridge"))
    base = pc.ridge(mesh)
    morph = gfem.shape.boundary_morph(base, "sidewall", grows="atom")
    wm = jnp.asarray(pc.ridge_width_map(morph))

    def make_cell(p, dp):
        return base.replace(mesh=morph((p - W0 + dp) * UM * wm))

    return make_cell


def test_r8_hermite_interpolation_matches_held_out_solves(ridge):
    response = dif.cell_response(
        ridge, SAMPLES, frequency=[pc.RIDGE_FREQUENCY], excite=EXCITE, options=OPTIONS
    )
    keep = [0, 2, 4]
    coarse = dataclasses.replace(
        response,
        parameter=response.parameter[keep],
        values=response.values[keep],
        derivative=response.derivative[keep],
    )
    problem = gfem.Driven([pc.RIDGE_FREQUENCY], excite=EXCITE)
    err, err3 = 0.0, 0.0
    for w in HELD:
        s = gfem.solve(ridge(w, 0.0), problem, options=OPTIONS).s
        assert s.ports == response.ports
        err = max(err, float(np.abs(response(w).values - s.values).max()))
        err3 = max(err3, float(np.abs(coarse(w).values - s.values).max()))
    assert err <= TOL, err
    assert err3 >= RATIO * err, (err, err3)
