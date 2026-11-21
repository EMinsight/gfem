"""Gradients of a 2D driven solve by per-element material fields.

Each gradient case differentiates an S functional through `gfem.solve` and
checks it against fourth-order central differences (relative error below
1e-6) and a Taylor remainder of slope 2; the last cases check the inputs
that still raise.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
F = 5e9
GHZ = 1e9
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
#: The Taylor steps, along a unit direction of densities or permittivities.
STEPS = (2e-2, 1e-2, 5e-3)

#: A Drude metal at 5 GHz: Re eps about -15.
DRUDE = gfem.PermittivityLaw(
    eps_inf=1.0,
    terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=2 * GHZ)],
)


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


def line(mesh, design, *, boundaries=()):
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": design},
        boundaries=boundaries,
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )


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


def test_ez_index_blend_with_a_drude_end(mesh):
    rng = np.random.default_rng(0)
    n = len(mesh.elements("design"))
    rho = jnp.asarray(rng.uniform(0.05, 0.2, n))

    def objective(rho):
        field = gfem.MaterialField.interpolate(
            rho, gfem.Material(eps_r=2.0), gfem.Material(eps_r=DRUDE), scheme="index"
        )
        problem = gfem.Driven(F, polarization="Ez")
        s = gfem.solve(line(mesh, field), problem, options=OPTIONS).s.values[0]
        return functional(s)

    check(objective, rho, rng, step=1e-2)


HZ_BLENDS = {
    # a metal's constant permittivity at one frequency, Re eps about -15: the
    # index blend written out, as a field of complex eps_r
    "metal": lambda rho: gfem.MaterialField(
        eps_r=((1 - rho) * np.sqrt(2.0) + rho * np.sqrt(-15.0 - 3.0j)) ** 2
    ),
    # the user's spelling, two constant ends (its conductivity is a traced zero)
    "interpolate": lambda rho: gfem.MaterialField.interpolate(
        rho,
        gfem.Material(eps_r=2.0),
        gfem.Material(eps_r=12.0, tan_delta=0.05),
        scheme="index",
    ),
}


@pytest.mark.parametrize("blend", HZ_BLENDS)
def test_hz_constant_permittivity_index_blend(mesh, blend):
    # the two ports' Hz signs differ (inward normals +x and -x), so S21
    # carries the sign convention into the functional
    rng = np.random.default_rng(1)
    n = len(mesh.elements("design"))
    rho = jnp.asarray(rng.uniform(0.05, 0.2, n))

    def objective(rho):
        model = line(mesh, HZ_BLENDS[blend](rho), boundaries=(gfem.PEC("side"),))
        problem = gfem.Driven(F, polarization="Hz")
        s = gfem.solve(model, problem, options=OPTIONS).s.values[0]
        return functional(s)

    check(objective, rho, rng, step=1e-2)


#: Per property, its base value and a difference step.
PROPERTIES = {"tan_delta": (0.05, 1e-2), "mu_r": (1.5, 3e-2)}


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
@pytest.mark.parametrize("prop", PROPERTIES)
def test_a_field_s_loss_tangent_and_permeability(mesh, polarization, prop):
    rng = np.random.default_rng(3)
    n = len(mesh.elements("design"))
    base, step = PROPERTIES[prop]
    p = jnp.asarray(base * rng.uniform(0.8, 1.2, n))

    def objective(p):
        values = {"eps_r": jnp.full(n, 3.0), "tan_delta": jnp.full(n, 0.02), prop: p}
        field = gfem.MaterialField(**values)
        # the sides carry the TEM line in both polarizations: PMC (the
        # default) in Ez, PEC in Hz
        sides = (gfem.PEC("side"),) if polarization == "Hz" else ()
        model = line(mesh, field, boundaries=sides)
        problem = gfem.Driven(F, polarization=polarization)
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    check(objective, p, rng, step=step)


def test_hz_conductivity_gradient_at_zero(mesh):
    """A field's conductivity, zero in Hz, against a region's uniform one.

    A field takes no nonzero conductivity in Hz, so the difference is taken
    on a `Material` region's conductivity (one-sided, fourth order) along
    the uniform direction.
    """
    n = len(mesh.elements("design"))
    problem = gfem.Driven(F, polarization="Hz")
    side = (gfem.PEC("side"),)

    def objective(sigma):
        field = gfem.MaterialField(eps_r=jnp.full(n, 4.0), conductivity=sigma)
        model = line(mesh, field, boundaries=side)
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    def uniform(sigma):
        model = line(
            mesh, gfem.Material(eps_r=4.0, conductivity=sigma), boundaries=side
        )
        return float(
            functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])
        )

    grad = np.asarray(jax.grad(objective)(jnp.zeros(n)))
    h = 1e-3  # S/m, against omega eps0 = 0.28 S/m
    f = [uniform(k * h) for k in range(5)]
    fd = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * h)
    assert abs(grad.sum() - fd) < 1e-6 * abs(fd)


#: A frequency above the 25 GHz cutoff of the ports' line mode between
#: absorbing ends.
F_ABSORBING = 30e9


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_design_owned_second_order_absorbing_line(mesh, polarization):
    # the line "side" runs along the design region: its segments there read
    # the design's permittivity, in the first- and second-order terms
    rng = np.random.default_rng(2)
    n = len(mesh.elements("design"))
    eps = jnp.asarray(rng.uniform(2.0, 4.0, n))
    absorbing = (gfem.Absorbing("side", order=2),)

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps, tan_delta=jnp.full(n, 0.02))
        model = line(mesh, field, boundaries=absorbing)
        problem = gfem.Driven(F_ABSORBING, polarization=polarization)
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    check(objective, eps, rng, step=3e-2)


def test_hz_law_blend_raises(mesh):
    n = len(mesh.elements("design"))

    def objective(rho):
        field = gfem.MaterialField.interpolate(
            rho, gfem.Material(eps_r=2.0), gfem.Material(eps_r=DRUDE), scheme="index"
        )
        model = line(mesh, field, boundaries=(gfem.PEC("side"),))
        problem = gfem.Driven(F, polarization="Hz")
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    with pytest.raises(NotImplementedError, match=r"'design'.*permittivity-law.*Hz"):
        jax.grad(objective)(jnp.full(n, 0.1))


def test_node_coordinates_still_raise(mesh):
    model = line(mesh, gfem.Material(eps_r=2.0))
    leaves, tree = jax.tree_util.tree_flatten(mesh)

    def objective(nodes):
        moved = jax.tree_util.tree_unflatten(tree, [nodes])
        problem = gfem.Driven(F, polarization="Ez")
        result = gfem.solve(model.replace(mesh=moved), problem, options=OPTIONS)
        return functional(result.s.values[0])

    # traced nodes count as moved at the solve entry
    with pytest.raises(NotImplementedError, match=r"nodes moved.*2D mesh"):
        jax.grad(objective)(leaves[0])
