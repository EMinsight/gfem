"""Driven solves through the tree-cotree gauge (ADR 0215, 0216).

On the parallel-plate line of test_driven.py, whose S has a closed form (ABCD
matrices), the gauged direct solve keeps every digit far below the
low-frequency floor where the ungauged one loses them, in S, in its
reciprocity and in the adjoint gradient. A wave-port model solves the same
gauged and ungauged above the floor, and a uniform sweep refactorizes its
pooled gauged plan.

The gauge is switched on by the internal `driven.tree_cotree_gauge` context
until `SolverOptions.gauge` exists.
"""

import logging
import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._engine import cache, driven
from gfem._linalg.direct import DirectHandle

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
R = 50.0
ETA0 = const.mu_0 * const.c
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
DATA = Path(__file__).parent / "data"


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


def _closed_form(f, eps=EPS):
    """S of the line between two R loads, from its ABCD matrix (jnp: differentiable)."""
    beta = 2 * math.pi * f / const.c * jnp.sqrt(eps)
    z0 = ETA0 * H / (W * jnp.sqrt(eps))
    length = L * 1e-3
    a = d = jnp.cos(beta * length)
    b = 1j * z0 * jnp.sin(beta * length)
    c = 1j * jnp.sin(beta * length) / z0
    den = a + b / R + c * R + d
    return jnp.array(
        [
            [(a + b / R - c * R - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / R - c * R + d) / den],
        ]
    )


def _solve(model, problem, *, gauged, options=OPTIONS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.LowFrequencyWarning)
        if not gauged:
            return gfem.solve(model, problem, options=options)
        with driven.tree_cotree_gauge():
            return gfem.solve(model, problem, options=options)


FREQUENCIES = (1e4, 1e6, 1e8)


@pytest.fixture(scope="module")
def gauged_s(line):
    problem = gfem.Driven(frequency=FREQUENCIES)
    return np.asarray(_solve(line, problem, gauged=True).s.values)


@pytest.mark.parametrize("n", range(len(FREQUENCIES)))
def test_gauged_s_matches_the_closed_form_below_the_floor(gauged_s, n):
    expected = np.asarray(_closed_form(FREQUENCIES[n]))
    assert np.abs(gauged_s[n] - expected).max() <= 1e-13


def test_ungauged_s_loses_digits_at_10_khz(line):
    s = np.asarray(_solve(line, gfem.Driven(frequency=1e4), gauged=False).s.values)
    assert np.abs(s[0] - np.asarray(_closed_form(1e4))).max() > 1e-10


def test_gauged_s_is_reciprocal(gauged_s):
    assert np.abs(gauged_s[:, 0, 1] - gauged_s[:, 1, 0]).max() <= 1e-14


def test_gauged_adjoint_gradient_matches_the_closed_form(line):
    # p1 alone is excited, so S11's gradient takes an adjoint solve
    problem = gfem.Driven(frequency=1e4, excite="p1")

    def part(eps, take):
        # one value per tetrahedron, all equal: the gradient's sum is dS/dε
        field = gfem.MaterialField(eps_r=jnp.full(len(line.mesh.tetrahedra), eps))
        model = line.with_material("dielectric", field)
        return take(_solve(model, problem, gauged=True).s.values[0, 0, 0])

    eps = jnp.float64(EPS)
    grad = jax.grad(part)(eps, jnp.real) + 1j * jax.grad(part)(eps, jnp.imag)
    expected = jax.jacfwd(lambda e: _closed_form(1e4, e)[0, 0].real)(eps) + 1j * (
        jax.jacfwd(lambda e: _closed_form(1e4, e)[0, 0].imag)(eps)
    )
    assert abs(complex(grad) - complex(expected)) <= 1e-10 * abs(complex(expected))


def test_gauged_and_ungauged_wave_port_s_agree_above_the_floor():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    width, gap = 40e-6, 10e-6
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort(
                "p1",
                "port1",
                voltage_path=((width / 2, 0.0, 0.0), (width / 2, gap, 0.0)),
            ),
            gfem.WavePort("p2", "port2"),
        ],
    )
    problem = gfem.Driven(frequency=1e9)
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    gauged = np.asarray(_solve(model, problem, gauged=True, options=options).s.values)
    ungauged = np.asarray(
        _solve(model, problem, gauged=False, options=options).s.values
    )
    assert np.abs(gauged - ungauged).max() <= 1e-9


@pytest.fixture
def plans(monkeypatch):
    """Counts the direct handles' analyses (a new plan, PARDISO phase 11)."""
    calls = []
    factor = DirectHandle._factor

    def counted(self, *args, **kwargs):
        calls.append(1)
        return factor(self, *args, **kwargs)

    monkeypatch.setattr(DirectHandle, "_factor", counted)
    cache.release_memory()
    yield calls
    cache.release_memory()


def test_a_uniform_sweep_refactorizes_its_pooled_gauged_plan(line, plans, caplog):
    with caplog.at_level(logging.INFO, logger="gfem.solver"):
        _solve(line, gfem.Driven(frequency=(1e4, 2e4, 3e4)), gauged=True)
        # new frequencies: the pooled base again
        _solve(line, gfem.Driven(frequency=(4e4, 5e4)), gauged=True)
    assert len(plans) == 1
    actions = [r.action for r in caplog.records if hasattr(r, "action")]
    assert actions == ["factorized"] + ["refactorized"] * 4
