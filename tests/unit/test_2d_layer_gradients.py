"""Layer-grid designs of a 2D model, sampled at the solve's quadrature points.

U1 checks the sampled permittivity against the bilinear pixel map at the
pinned rule's points. The gradient cases differentiate an S functional by
the pixels through `gfem.solve`, at orders 2 and 3, against fourth-order
central differences (relative error below 1e-6) and a Taylor remainder of
slope 2.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)

import gfem
from gfem import design

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
F = 5e9
GHZ = 1e9
MM = 1e-3
#: The Taylor steps, along a unit direction of pixel densities.
STEPS = (4e-3, 2e-3, 1e-3)

#: A Drude metal at 5 GHz: Re eps about -15.
DRUDE = gfem.PermittivityLaw(
    eps_inf=1.0,
    terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=2 * GHZ)],
)
DIELECTRIC = gfem.Material(eps_r=2.0)
LOSSY = gfem.Material(eps_r=12.0, tan_delta=0.05)


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


@pytest.fixture(scope="module")
def grid(mesh):
    return design.LayerGrid(mesh, "design", pitch=1 * MM)


def line(mesh, design_material, *, boundaries=()):
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": design_material},
        boundaries=boundaries,
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )


def rule_points(built, region, order):
    """The pinned rule's points on ``region``'s triangles, in metres, per element.

    NGSolve's reference triangle has the vertices (1, 0), (0, 1), (0, 0).
    """
    rule = ngs.IntegrationRule(ngs.ET.TRIG, 2 * order)
    ref = np.array([p[:2] for p in rule.points])
    out = {}
    for el in built.mesh.Elements(ngs.VOL):
        if el.mat != region:
            continue
        v = np.array([built.mesh[w].point[:2] for w in el.vertices]) * built.scale
        out[el.nr] = (
            ref[:, :1] * v[0] + ref[:, 1:] * v[1] + (1 - ref.sum(1))[:, None] * v[2]
        )
    return out


@pytest.mark.parametrize("order", [2, 3])
def test_u1_the_sampled_permittivity_is_the_bilinear_map_at_the_rule_points(
    mesh, grid, order
):
    from gfem._engine.cache import discretization
    from gfem._engine.driven2d import driven2d_operator

    rho = np.random.default_rng(0).uniform(0.0, 1.0, grid.shape)
    field = gfem.MaterialField.interpolate(
        grid.to_elements(jnp.asarray(rho)), DIELECTRIC, LOSSY, scheme="linear"
    )
    op = driven2d_operator(line(mesh, field), order, "Ez")
    (piece,) = (p for p in op.pieces if p.name == "points mass design")
    values = piece.coefficient.vec.FV().NumPy()
    built = discretization(mesh).built
    eps_a, eps_b = 2.0, 12.0 * (1 - 0.05j)
    worst = 0.0
    for nr, points in rule_points(built, "design", order).items():
        index, weight = design._bilinear(points, grid.origin, grid.pitch, grid.shape)
        expected = eps_a + (rho.ravel()[index] * weight).sum(axis=1) * (eps_b - eps_a)
        got = values[nr * len(points) : (nr + 1) * len(points)]
        worst = max(worst, np.abs(got - expected).max())
    assert worst < 1e-12


def functional(s):
    """A real functional of every S entry, S (2, 2)."""
    weights = jnp.array([[1.0, 0.5], [2.0, -1.0]])
    return jnp.sum(weights * jnp.abs(s) ** 2) + jnp.real((0.3 - 0.7j) * s[1, 0])


def check(objective, p, rng, step):
    """FD along two unit directions, and a Taylor remainder of slope 2."""
    grad = jax.grad(objective)(p)
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=step, grad=grad)
    assert_taylor_second_order(
        objective, p, directions[0], STEPS, grad=grad, tolerance=0.05
    )


#: Per case: the polarization, the ends, the scheme, the side lines.
CASES = {
    # a Drude metal end: the law's eps_inf limit and susceptibility pieces
    "Ez index Drude": ("Ez", DIELECTRIC, gfem.Material(eps_r=DRUDE), "index", ()),
    # constant ends whose permeability and conductivity blend too
    "Ez simp mu sigma": (
        "Ez",
        DIELECTRIC,
        gfem.Material(eps_r=6.0, mu_r=1.5, conductivity=0.05),
        "simp",
        (),
    ),
    # constant ends only in Hz: the stiffness 1 / eps at the points
    "Hz index lossy": ("Hz", DIELECTRIC, LOSSY, "index", (gfem.PEC("side"),)),
}


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("case", CASES)
def test_pixel_gradients_at_the_quadrature_points(mesh, grid, case, order):
    polarization, void, solid, scheme, sides = CASES[case]
    rng = np.random.default_rng(order)
    rho = jnp.asarray(rng.uniform(0.05, 0.25, grid.shape))
    problem = gfem.Driven(F, polarization=polarization)
    options = gfem.SolverOptions(order=order, linear_solver="direct")

    def objective(rho):
        field = gfem.MaterialField.interpolate(
            grid.to_elements(rho), void, solid, scheme=scheme
        )
        model = line(mesh, field, boundaries=sides)
        return functional(gfem.solve(model, problem, options=options).s.values[0])

    check(objective, rho, rng, step=1e-2)


#: The layer densities an Hz solve refuses, and what the message names.
HZ_REFUSED = {
    "law end": (gfem.Material(eps_r=DRUDE), "permittivity-law"),
    "conductivity": (gfem.Material(eps_r=4.0, conductivity=0.05), "conductivity"),
}


@pytest.mark.parametrize("refused", HZ_REFUSED)
def test_hz_refuses_a_frequency_dependent_layer_density(mesh, grid, refused):
    solid, what = HZ_REFUSED[refused]
    field = gfem.MaterialField.interpolate(
        grid.to_elements(jnp.full(grid.shape, 0.1)), DIELECTRIC, solid
    )
    model = line(mesh, field, boundaries=(gfem.PEC("side"),))
    with pytest.raises(NotImplementedError, match=rf"'design'.*{what}.*Hz"):
        gfem.solve(model, gfem.Driven(F, polarization="Hz"))


def test_an_absorbing_line_on_a_layer_grid_raises(mesh, grid):
    field = gfem.MaterialField.interpolate(
        grid.to_elements(jnp.full(grid.shape, 0.1)), DIELECTRIC, LOSSY
    )
    model = line(mesh, field, boundaries=(gfem.Absorbing("side"),))
    with pytest.raises(NotImplementedError, match="'design', a layer-grid design"):
        gfem.solve(model, gfem.Driven(F, polarization="Ez"))
