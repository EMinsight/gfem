"""Transient thermal solves differentiate in the four boundary values.

A fixed temperature, a heat transfer coefficient, an ambient and a flux, with
unequal boundaries and a float initial temperature, through the traces, the
kept fields and the source-free temperature, finite differences of fourth order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import thermal_cases as tc
from finite_differences import assert_gradient_matches, derivative, unit_directions

import gfem

L = 4.0
STEPS = 6
DURATION = 4e-6
MATERIAL = gfem.Material(
    thermal_conductivity=1.4, mass_density=2200.0, specific_heat=700.0
)
#: temperature, coefficient, ambient, flux: the reference values
REFERENCE = np.array([300.0, 1e5, 290.0, 5e4])
EDGES = [
    (lambda p: abs(p[1]) < 1e-9 and p[0] < 1.5, "bottom"),
    (lambda p: abs(p[1] - L) < 1e-9 and p[0] < 1.5, "top"),
    (lambda p: abs(p[0]) < 1e-9, "side"),
]


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    path = tmp_path_factory.mktemp("boundary-transient")
    one = tc.section(path / "one.msh", [(0.0, 0.0, 1.0, L, "a", None)], 2.0, EDGES)
    # a second piece, apart, touches no sink
    rects = [(0.0, 0.0, 1.0, L, "a", None), (2.0, 0.0, 3.0, L, "b", None)]
    two = tc.section(path / "two.msh", rects, 2.0, EDGES)
    return {"one": one, "two": two}


def _objective(mesh, regions, options):
    materials = dict.fromkeys(regions, MATERIAL)
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("a", power_density=1e12),
        duration=DURATION,
        steps=STEPS,
        initial_temperature=295.0,
        traces={name: gfem.Trace.mean(name) for name in regions},
        keep_fields=True,
    )
    weights = np.linspace(1.0, 2.0, STEPS + 1)

    def objective(x):
        t, h, amb, q = x * REFERENCE
        model = gfem.Model(
            mesh,
            materials=materials,
            boundaries=[
                gfem.FixedTemperature("bottom", temperature=t),
                gfem.Convection("top", coefficient=h, ambient=amb),
                gfem.HeatFlux("side", flux=q),
            ],
        )
        result = gfem.solve(model, problem, options=options)
        out = jnp.sum(result.traces["a"] * weights)
        out += 0.01 * jnp.sum(result.temperature(DURATION).coefficients ** 2) / 1e4
        out += 1e-3 * jnp.sum(result.rise(DURATION / 2).coefficients)
        for name in regions[1:]:
            out += jnp.sum(result.traces[name])
        return out

    return objective


@pytest.mark.parametrize(
    ("scheme", "theta", "mesh", "regions"),
    [
        ("tr_bdf2", 1.0, "one", ("a",)),
        ("theta", 1.0, "one", ("a",)),
        ("tr_bdf2", 1.0, "two", ("a", "b")),
    ],
)
def test_the_four_values_differentiate_by_finite_differences(
    meshes, scheme, theta, mesh, regions
):
    options = gfem.SolverOptions(
        order=2, time_scheme=scheme, theta=theta, linear_solver="direct"
    )
    objective = _objective(meshes[mesh], regions, options)
    x = jnp.ones(4)
    rng = np.random.default_rng(1)
    grad = np.asarray(jax.grad(objective)(x))
    assert np.all(np.abs(grad) > 0)
    # each value alone, then a mix
    basis = [np.eye(4)[i] for i in range(4)]
    assert_gradient_matches(
        objective, x, [*basis, *unit_directions(rng, (4,), 2)], step=1e-2, grad=grad
    )


def test_a_value_the_trace_does_not_read_still_matches_one_sided(meshes):
    # the derivative along one value agrees with a plain difference too
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    objective = _objective(meshes["one"], ("a",), options)
    x = jnp.ones(4)
    grad = np.asarray(jax.grad(objective)(x))
    for i in range(4):
        d = np.eye(4)[i]
        assert float(grad[i]) == pytest.approx(
            derivative(objective, x, d, 1e-2), rel=1e-6
        )


@pytest.mark.parametrize(
    ("scheme", "theta"), [("tr_bdf2", 1.0), ("theta", 1.0), ("theta", 0.5)]
)
def test_heat_out_traces_differentiate_in_the_four_values(meshes, scheme, theta):
    sides = {"out_top": "top", "out_bottom": "bottom", "out_side": "side"}
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("a", power_density=1e12),
        duration=DURATION,
        steps=STEPS,
        initial_temperature=295.0,
        traces={k: gfem.Trace.heat_out_per_length(v) for k, v in sides.items()},
    )
    options = gfem.SolverOptions(
        order=2, time_scheme=scheme, theta=theta, linear_solver="direct"
    )
    weights = np.linspace(1.0, 2.0, STEPS + 1)

    def objective(x):
        t, h, amb, q = x * REFERENCE
        model = gfem.Model(
            meshes["one"],
            materials={"a": MATERIAL},
            boundaries=[
                gfem.FixedTemperature("bottom", temperature=t),
                gfem.Convection("top", coefficient=h, ambient=amb),
                gfem.HeatFlux("side", flux=q),
            ],
        )
        result = gfem.solve(model, problem, options=options)
        total = 0.0
        for k, name in enumerate(sides):
            total += (k + 1) * jnp.sum(result.traces[name] * weights)
        return total / 1e-6

    x = jnp.ones(4)
    grad = np.asarray(jax.grad(objective)(x))
    assert np.all(np.abs(grad) > 0)
    basis = [np.eye(4)[i] for i in range(4)]
    rng = np.random.default_rng(2)
    assert_gradient_matches(
        objective, x, [*basis, *unit_directions(rng, (4,), 2)], step=1e-2, grad=grad
    )
