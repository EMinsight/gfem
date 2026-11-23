r"""Gradients of free-space measurements, against finite differences.

A lossy dielectric cube (a `gfem.MaterialField`, "scat") in an air box
wrapped in a perfectly matched layer, at order 2, under plane waves. The
cross sections and the corrected power flow read the background pieces of
an element layer, the load :math:`b` and the incident interpolant, none of
which depends on the design: the design reaches them through the solved
field alone, which the checks with a frozen field assert. Fourth-order
differences to :math:`10^{-6}` (`finite_differences`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem

pytestmark = pytest.mark.verification

AIR = gfem.Material()
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0 / 2
TE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")
TM = gfem.PlaneWave("tm", theta=0.4, phi=0.3, polarization="TM")
#: The cube's permittivity: the design is its real part.
EPS, LOSS = 3.0, 0.6j


def _volume(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("xsec") / "box.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        cells=4,
        volume=_volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("scat"))
    return gfem.Model(
        mesh,
        materials={
            "air": AIR,
            "pml": AIR,
            "scat": gfem.MaterialField(eps_r=np.full(n, EPS - LOSS)),
        },
        pml=gfem.PML("pml"),
        background=AIR,
    )


def _design(model, eps):
    return model.with_material("scat", gfem.MaterialField(eps_r=eps - LOSS))


def _solve(model, eps):
    moved = _design(model, eps)
    problem = gfem.Driven(F, sources=(TE, TM), keep_fields=True)
    return moved, gfem.solve(moved, problem, options=OPTIONS)


def _cross_sections(model, result):
    te = gfem.cross_sections(result, "te", model=model)
    tm = gfem.cross_sections(result, "tm", model=model)
    scale = 1e12  # m² to um²
    return scale * (te.scattering[0] + 0.7 * te.absorption[0] + 0.4 * tm.extinction[0])


def _start(model):
    return jnp.full(len(model.mesh.elements("scat")), EPS)


def test_cross_sections_by_the_scatterer_permittivity(model):
    def objective(eps):
        return _cross_sections(*_solve(model, eps))

    eps = _start(model)
    directions = unit_directions(np.random.default_rng(5), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_cross_sections_read_no_scatterer_material(model):
    _, result = _solve(model, _start(model))
    frozen = jax.lax.stop_gradient(result)

    def objective(eps):
        return _cross_sections(_design(model, eps), frozen)

    gradient = jax.grad(objective)(_start(model))
    assert np.all(np.asarray(gradient) == 0)


def test_corrected_power_flow_by_the_absorber_permittivity(model):
    def objective(eps):
        moved, result = _solve(model, eps)
        total = result.e_field(F, "tm")
        return -1e15 * gfem.power_flow(total, "scat", model=moved)

    eps = _start(model)
    directions = unit_directions(np.random.default_rng(7), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)
