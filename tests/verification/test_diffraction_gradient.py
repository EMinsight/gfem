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


# --- G2: emissivity ------------------------------------------------------------

#: The weights of the two directional emissivities in G2's objective.
G2_WEIGHTS = jnp.asarray([[0.3], [0.7]])
#: The gradient with ``retain_memory=0`` against the default one (2-norm).
G2_RETAIN_TOL = 1e-9


def _g2_objective(mesh, options):
    f0 = pc.F1_FREQUENCY
    frequencies = (f0, 0.9 * f0)
    rule = dif.hemisphere(1, 1)

    def objective(eps):
        model = pc.f1(mesh).with_material("slab", gfem.MaterialField(eps_r=eps))
        hemispherical = dif.angle_sweep(
            model,
            frequencies,
            surface=TOP,
            theta=rule.theta,
            phi=rule.phi,
            options=options,
        )
        e_h = dif.hemispherical_emissivity(hemispherical, rule, model=model)
        directional = dif.angle_sweep(
            model,
            frequencies,
            surface=TOP,
            theta=[0.0, 0.0],
            phi=[0.0, np.pi / 2],
            options=options,
        )
        e = dif.emissivity(directional, model=model).values
        return jnp.sum(e_h) + jnp.sum(G2_WEIGHTS * e)

    return objective


def test_g2_emissivity_matches_finite_differences(tmp_path):
    mesh = pc.f1_mesh(tmp_path)
    n = len(mesh.elements("slab"))
    scale = np.random.default_rng(30).uniform(0.9, 1.1, n)
    eps = jnp.asarray(4.0 * (1 - 0.1j) * scale)
    objective = _g2_objective(mesh, tight("direct"))
    grad = jax.grad(objective)(eps)
    directions = unit_directions(np.random.default_rng(0), eps.shape, 3)
    directions[2] = 1j * directions[2]
    assert_gradient_matches(objective, eps, directions, step=STEP, rel=1e-6, grad=grad)
    frugal = jax.grad(_g2_objective(mesh, tight("direct", retain_memory=0)))(eps)
    assert (
        float(jnp.linalg.norm(frugal - grad) / jnp.linalg.norm(grad)) <= G2_RETAIN_TOL
    )


# --- G3: the cell response's stored derivative -------------------------------------

#: The finite-difference step in the ridge width, um (1 nm).
G3_STEP = 1e-3
#: Max |D_fd - D| over max |D|.
G3_TOL = 1e-6


def test_g3_the_stored_derivative_matches_fourth_order_differences(tmp_path):
    um = 1e-6
    w0 = pc.RIDGE_W0
    mesh = pc.ridge_mesh(tmp_path)
    base = pc.ridge(mesh)
    morph = gfem.shape.boundary_morph(base, "sidewall", grows="atom")
    wm = jnp.asarray(pc.ridge_width_map(morph))

    def make_cell(p, dp):
        return base.replace(mesh=morph((p - w0 + dp) * um * wm))

    freqs = [pc.RIDGE_FREQUENCY, 1.05 * pc.RIDGE_FREQUENCY]
    excite = ["sub TE", "sub TM"]
    options = gfem.SolverOptions(
        order=2, linear_solver="direct", threads=2, tolerance=1e-12
    )
    response = dif.cell_response(
        make_cell, [w0], frequency=freqs, excite=excite, options=options
    )
    problem = gfem.Driven(freqs, excite=excite)

    def s_of(k):
        return np.asarray(
            gfem.solve(
                make_cell(w0 + k * G3_STEP, 0.0), problem, options=options
            ).s.values
        )

    f = {k: s_of(k) for k in (-2, -1, 1, 2)}
    fd = (8 * (f[1] - f[-1]) - (f[2] - f[-2])) / (12 * G3_STEP)
    d = np.asarray(response.derivative[0])
    assert d.shape == fd.shape == (2, 4, 2)
    assert np.abs(fd - d).max() <= G3_TOL * np.abs(d).max()
