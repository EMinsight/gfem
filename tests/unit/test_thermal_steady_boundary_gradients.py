"""Steady thermal gradients with respect to the four boundary values.

A fixed temperature, a heat transfer coefficient, an ambient and a flux are
differentiable in a steady solve: the temperature, the rise and the kept
fields. The rise reads only the coefficient, so its gradient with respect to
the other three is exactly zero.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches

import gfem

L, W, T = 4.0, 1.0, 0.2  # bar, um
METAL = gfem.Material(
    conductivity=1e6,
    eps_r=3.9,
    thermal_conductivity=28.0,
    mass_density=5240.0,
    specific_heat=598.0,
)
OXIDE = gfem.Material(
    eps_r=3.9, thermal_conductivity=1.4, mass_density=2200.0, specific_heat=700.0
)
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
PROBLEM = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-1))
VALUES = np.array([300.0, 25.0, 290.0, 5.0])
STEPS = (20.0, 1.0, 20.0, 2000.0)  # the linear values take any step; h a few percent
NAMES = ("temperature", "coefficient", "ambient", "flux")


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    """A metal bar in an oxide box: contacts "end_a", "end_b", a "floor" sink."""
    edges = (
        np.linspace(0.0, L, 5),
        np.array([-1.0, 0.0, W, 2.0]),
        np.array([0.0, T, 0.6]),
    )
    tol = 1e-9

    def volume(c):
        return "bar" if c[2] < T and 0 < c[1] < W else "oxide"

    def surface(c):
        in_bar = c[2] < T and 0 < c[1] < W
        if abs(c[0]) < tol and in_bar:
            return "end_a"
        if abs(c[0] - L) < tol and in_bar:
            return "end_b"
        return "floor" if abs(c[2]) < tol and not in_bar else ""

    path = tmp_path_factory.mktemp("steady") / "bar.msh"
    return gfem.Mesh.from_gmsh(
        sm.write_msh22(path, sm.structured(edges, volume, surface))
    )


def _model(mesh, t, h, amb, q):
    return gfem.Model(
        mesh,
        materials={"bar": METAL, "oxide": OXIDE},
        boundaries=[
            gfem.FixedTemperature("floor", temperature=t),
            gfem.Convection("end_a", coefficient=h, ambient=amb),
            gfem.HeatFlux("end_b", flux=q),
        ],
    )


def _weights(mesh):
    n = gfem.solve(_model(mesh, *VALUES), PROBLEM, options=DIRECT).temperature
    rng = np.random.default_rng(3)
    size = n.coefficients.shape[0]
    return rng.uniform(0.5, 1.5, size) / size, rng.uniform(0.5, 1.5, size) / size


def _objective(mesh, field):
    w_t, w_r = _weights(mesh)

    def objective(values):
        result = gfem.solve(_model(mesh, *values), PROBLEM, options=DIRECT)
        total = 0.0
        if "temperature" in field:
            total = total + jnp.dot(w_t, result.temperature.coefficients)
        if "rise" in field:
            total = total + jnp.dot(w_r, result.rise.coefficients)
        return total

    return objective


@pytest.mark.parametrize(
    "field", [("temperature",), ("rise",), ("temperature", "rise")]
)
@pytest.mark.parametrize("which", range(4))
def test_each_value_matches_a_fourth_order_difference(mesh, field, which):
    objective = _objective(mesh, field)
    grad = np.asarray(jax.grad(objective)(jnp.asarray(VALUES)))
    if field == ("rise",) and which != 1:
        assert grad[which] == 0.0
        return
    direction = np.zeros(4)
    direction[which] = 1.0
    assert_gradient_matches(
        objective,
        jnp.asarray(VALUES),
        [direction],
        step=STEPS[which],
        grad=grad,
    )


def test_the_rise_does_not_move_with_ambient_flux_or_fixed_temperature(mesh):
    objective = _objective(mesh, ("rise",))
    grad = np.asarray(jax.grad(objective)(jnp.asarray(VALUES)))
    assert grad[[0, 2, 3]].tolist() == [0.0, 0.0, 0.0]
    assert grad[1] != 0.0


def test_a_flux_traced_at_zero_has_a_gradient(mesh):
    objective = _objective(mesh, ("temperature",))
    values = jnp.asarray([300.0, 25.0, 290.0, 0.0])
    grad = np.asarray(jax.grad(objective)(values))
    assert grad[3] != 0.0
    assert_gradient_matches(
        objective,
        values,
        [np.array([0.0, 0.0, 0.0, 1.0])],
        step=2000.0,
        grad=grad,
    )


def test_touching_fixed_surfaces_sum_to_a_common_shift(mesh):
    w_t, _ = _weights(mesh)

    def build(a, b):
        return gfem.Model(
            mesh,
            materials={"bar": METAL, "oxide": OXIDE},
            boundaries=[
                gfem.FixedTemperature("floor", temperature=a),
                gfem.FixedTemperature("end_a", temperature=b),
                gfem.Convection("end_b", coefficient=25.0, ambient=300.0),
            ],
        )

    def objective(ab):
        result = gfem.solve(build(ab[0], ab[1]), PROBLEM, options=DIRECT)
        return jnp.dot(w_t, result.temperature.coefficients)

    def common(t):
        return objective(jnp.stack([t, t]))

    ab = jnp.asarray([300.0, 300.0])
    grad = np.asarray(jax.grad(objective)(ab))
    assert_gradient_matches(
        common, jnp.asarray(300.0), [np.asarray(1.0)], step=1.0, grad=grad.sum()
    )
