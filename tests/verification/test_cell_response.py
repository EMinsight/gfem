r"""The cell response against held-out solves.

R8: the ridge of `periodic_cases` on one mesh, its width moved by a boundary
morph; 9 samples of the width over 0.17 to 0.33 um at f0, the substrate's TE
and TM excited. The 5-sample subset (every other sample) is checked at its
interval midpoints, the 4 samples it leaves out, against the accuracy bound;
the 9 samples are checked at their 8 midpoints, solved directly.

The Hermite error at :math:`s = (p - p_k)/h` in an interval is
:math:`\tfrac{1}{4!} S^{(4)}(\xi)\, s^2 (1 - s)^2 h^4`, so halving the spacing
divides the error at the midpoints by :math:`2^4 = 16` once :math:`S^{(4)}` is
about constant over an interval. The convergence check compares midpoints
with midpoints: a grid of twice the spacing, read at the midpoints of the
finer one, sits at a quarter of its intervals, where :math:`s^2 (1 - s)^2` is
:math:`9/16` of its midpoint value, so that ratio tends to 9, not 16. A
stored derivative in the wrong units, or a missing one, leaves an error of
order :math:`h` and fails the check.
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
SAMPLES = [0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33]
#: The midpoints of SAMPLES, solved directly.
HELD = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
EXCITE = ["sub TE", "sub TM"]
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=2)
#: Max over the 8 specular entries of |S_interp - S_solve| with 5 samples, at
#: their midpoints. Measured 1.18e-3 (sub TM <- sub TM).
TOL = 2e-3
#: The order log2(err_5 / err_9) of the max Hermite error at the midpoints,
#: at least halfway from 3 to 4 (cubic Hermite converges at order 4).
#: Measured 3.90 (ratio 14.9, err_9 = 7.9e-5).
MIN_ORDER = 3.5


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
    every_other = dataclasses.replace(
        response,
        parameter=response.parameter[::2],
        values=np.asarray(response.values)[::2],
        derivative=np.asarray(response.derivative)[::2],
    )
    # the 5-sample subset at its midpoints: the samples it leaves out
    err5 = max(
        float(np.abs(every_other(w).values - response.values[i]).max())
        for i, w in enumerate(SAMPLES)
        if i % 2
    )
    problem = gfem.Driven([pc.RIDGE_FREQUENCY], excite=EXCITE)
    err9 = 0.0
    for w in HELD:
        s = gfem.solve(ridge(w, 0.0), problem, options=OPTIONS).s
        assert s.ports == response.ports
        err9 = max(err9, float(np.abs(response(w).values - s.values).max()))
    assert err5 <= TOL, err5
    assert np.log2(err5 / err9) >= MIN_ORDER, (err5, err9)
