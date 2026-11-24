"""Gradients of the cell outputs of `gfem.diffraction` against finite differences.

G1: the lamellar grating F2 (`periodic_cases.floquet_gradient_model`, a lossy
per-element permittivity, every propagating order declared) under the conical
Bloch incidence of 20 degrees and azimuth 30, at f0 and 0.8 f0, with an
objective that reads every view of a cell output: the circular efficiency rows,
the unpolarized absorptance, a circular entry of a Jones matrix across the
cell, a Mueller entry of the reflection and the power of an order that does
not propagate. Fourth-order central differences of the same objective on the
direct solver, `jax.jit` of the gradient against the eager gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import tight

import gfem
from gfem import diffraction as dif

TOP, BOT = "z_max", "z_min"
#: The finite-difference step in the permittivity of each element.
STEP = 1e-2
#: `jax.jit(jax.grad(J))` against `jax.grad(J)`, relative in the 2-norm (the
#: measured difference is round-off).
JIT_TOL = 1e-12


def _objective(mesh, problem, weights, options):
    def objective(eps):
        model = pc.floquet_gradient_model(mesh, eps)
        result = gfem.solve(model, problem, options=options)
        rows = dif.efficiencies(
            result, TOP, polarization="RCP", model=model, basis="circular"
        )
        absorbed = dif.power_balance(
            result, TOP, polarization="unpolarized", model=model
        ).absorptance
        across = dif.jones(result, BOT, TOP, model=model).circular
        reflection = dif.jones(result, TOP, TOP, model=model).mueller
        closed = dif.jones(result, TOP, TOP, model=model, order=(1, 0)).values
        return (
            jnp.sum(weights * rows.values)
            + 2.0 * jnp.sum(absorbed)
            + jnp.sum(jnp.real((0.7 - 0.3j) * across[:, 1, 0]))
            + jnp.sum(reflection[:, 2, 3])
            + jnp.sum(jnp.abs(closed) ** 2)
        )

    return objective


def test_g1_the_cell_outputs_match_finite_differences(tmp_path):
    mesh = pc.f2_mesh(tmp_path)
    eps = pc.floquet_gradient_design(mesh, np.random.default_rng(30))
    model = pc.floquet_gradient_model(mesh, eps)
    problem = pc.floquet_gradient_problem(model, bloch=True, fixed_angle=False)
    options = tight("direct")
    result = gfem.solve(model, problem, options=options)
    rows = dif.efficiencies(
        result, TOP, polarization="RCP", model=model, basis="circular"
    )
    weights = np.random.default_rng(1).uniform(0.5, 1.5, rows.values.shape)
    objective = _objective(mesh, problem, weights, options)

    grad = jax.grad(objective)(eps)
    directions = unit_directions(np.random.default_rng(0), eps.shape, 3)
    directions[2] = 1j * directions[2]  # the imaginary part of eps
    assert_gradient_matches(objective, eps, directions, step=STEP, rel=1e-6, grad=grad)
    jitted = jax.jit(jax.grad(objective))(eps)
    assert float(jnp.linalg.norm(jitted - grad) / jnp.linalg.norm(grad)) <= JIT_TOL
