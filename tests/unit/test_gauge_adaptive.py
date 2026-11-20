"""The adaptive sweep through the tree-cotree gauge.

On the parallel-plate line of test_gauge_solve.py, whose S has a closed form,
an adaptive sweep through the gauge keeps every digit below the
low-frequency floor: its snapshots are gauged solves, and its reduced pieces,
projected by the matrix-free ``DrivenOperator.apply``, lose none. Above the
floor it matches the uniform gauged sweep to the sweep tolerance.

The gauged solves ask for the gauge (``gauge="tree_cotree"``), the ungauged
ones turn it off (``gauge="none"``).
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
R = 50.0
ETA0 = const.mu_0 * const.c
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
BELOW = np.geomspace(1e4, 1e6, 9)
ABOVE = np.linspace(1e9, 20e9, 30)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _closed_form(f):
    """S of the line between two R loads, from its ABCD matrix."""
    beta = 2 * math.pi * f / const.c * math.sqrt(EPS)
    z0 = ETA0 * H / (W * math.sqrt(EPS))
    length = L * 1e-3
    a = d = math.cos(beta * length)
    b = 1j * z0 * math.sin(beta * length)
    c = 1j * math.sin(beta * length) / z0
    den = a + b / R + c * R + d
    return np.array(
        [
            [(a + b / R - c * R - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / R - c * R + d) / den],
        ]
    )


def _solve(model, problem, *, gauged):
    options = dataclasses.replace(OPTIONS, gauge="tree_cotree" if gauged else "none")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.LowFrequencyWarning)
        return gfem.solve(model, problem, options=options)


def test_gauged_adaptive_sweep_matches_the_closed_form_below_the_floor(line):
    problem = gfem.Driven(frequency=BELOW, sweep="adaptive")
    s = np.asarray(_solve(line, problem, gauged=True).s.values)
    expected = np.stack([_closed_form(f) for f in BELOW])
    assert np.abs(s - expected).max() <= 1e-12


def test_gauged_adaptive_sweep_matches_the_uniform_gauged_sweep_above_the_floor(
    line,
):
    adaptive = _solve(line, gfem.Driven(frequency=ABOVE, sweep="adaptive"), gauged=True)
    uniform = _solve(line, gfem.Driven(frequency=ABOVE), gauged=True)
    assert adaptive.sweep.converged
    diff = np.abs(np.asarray(adaptive.s.values) - np.asarray(uniform.s.values)).max()
    assert diff <= OPTIONS.adaptive_tolerance
