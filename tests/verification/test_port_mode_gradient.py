"""Port mode gradients that the matrix does not reach.

- Reduced sweep rows are differentiated exactly: the gradient at the default
  ``port_mode_tolerance`` (where most rows of a band come from the reduced
  model) equals central differences of solves at ``1e-12`` to 1e-7.
- A near-degenerate pair keeps exact gradients (the backward only fails when
  the adjoint solve itself does).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gfem

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).parents[1] / "unit" / "data"
A, B = 22.86e-3, 10.16e-3
TIGHT = gfem.SolverOptions(port_mode_tolerance=1e-12)


def central(f, x, h):
    """Fourth-order central difference."""
    return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h)


def slab_model(eps, tan_delta=0.05):
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    n = len(mesh.elements("slab"))
    return gfem.Model(
        mesh,
        materials={
            "slab": gfem.MaterialField(
                eps_r=eps * jnp.ones(n), tan_delta=tan_delta * jnp.ones(n)
            ),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


POINTS = np.array(
    [[x * A, y * B, 0.0] for x, y in ((0.2, 0.3), (0.5, 0.5), (0.7, 0.4), (0.9, 0.8))]
)


def profile(result, k0):
    """Gamma and a phase-carrying field term at every row."""
    e = result.e_field.at(POINTS)
    scale = jnp.sqrt(jnp.sum(jnp.abs(e) ** 2, axis=(1, 2)))
    return (
        jnp.sum(result.gamma.imag / k0)
        + 0.3 * jnp.sum(result.gamma.real / k0)
        + jnp.sum(jnp.abs(e) ** 2 / scale[:, None, None] ** 2).real
        + jnp.sum(jnp.real(e[:, :, 1] / scale[:, None] * (0.3 - 0.7j)))
    )


def test_reduced_sweep_rows_have_the_exact_modes_gradient():
    grid = np.linspace(10e9, 20e9, 21)
    k0 = 2 * np.pi * grid / 299792458.0
    problem = gfem.PortMode("p1", grid)
    solved = gfem.solve(slab_model(4.0), problem).solved
    assert int(np.sum(~np.asarray(solved))) >= len(grid) // 2

    def objective(options):
        return lambda eps: profile(
            gfem.solve(slab_model(eps), problem, options=options), k0
        )

    gradient = float(jax.grad(objective(gfem.SolverOptions()))(4.0))
    reference = central(objective(TIGHT), 4.0, 1e-2)
    assert abs(gradient - reference) <= 1e-7 * abs(reference), (gradient, reference)


def test_a_cotangent_on_gamma_alone_needs_no_polish_and_matches():
    grid = np.linspace(10e9, 20e9, 21)
    k0 = 2 * np.pi * grid / 299792458.0
    problem = gfem.PortMode("p1", grid)

    def objective(options):
        return lambda eps: jnp.sum(
            gfem.solve(slab_model(eps), problem, options=options).gamma.imag / k0
        )

    gradient = float(jax.grad(objective(gfem.SolverOptions()))(4.0))
    reference = central(objective(TIGHT), 4.0, 1e-2)
    assert abs(gradient - reference) <= 1e-7 * abs(reference)


def square_model(delta, eps):
    """The square guide stretched by ``1 + delta`` along y; corner triangles ``eps``."""
    mesh = gfem.Mesh.from_gmsh(DATA / "square_waveguide.msh", length_unit="mm")
    nodes = np.asarray(mesh.nodes)
    stretched = nodes.copy()
    mid = nodes[:, 1].mean()
    stretched[:, 1] = mid + (1 + delta) * (nodes[:, 1] - mid)
    mesh = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(mesh), [jnp.asarray(stretched)]
    )
    tets = np.asarray(mesh.tetrahedra)[np.asarray(mesh.elements("air"))]
    centre = nodes[tets].mean(axis=1)
    corner = (centre[:, 0] > nodes[:, 0].mean()) & (centre[:, 1] > mid)
    field = jnp.where(jnp.asarray(corner), eps, 1.0)
    return gfem.Model(
        mesh,
        materials={"air": gfem.MaterialField(eps_r=field)},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def test_a_near_degenerate_pair_keeps_exact_gamma_and_profile_gradients():
    problem = gfem.PortMode("p1", 20e9)
    side = 10e-3
    pts = np.array([[0.3 * side, 0.4 * side, 0.0], [0.6 * side, 0.7 * side, 0.0]])

    def objective(eps):
        r = gfem.solve(square_model(1e-3, eps), problem, options=TIGHT)
        e = r.e_field.at(pts)[0]
        scale = jnp.sqrt(jnp.sum(jnp.abs(e) ** 2))
        return r.gamma[0].imag / 1e3 + jnp.real(jnp.sum(e[:, 1]) * (0.4 - 0.2j)) / scale

    gap = 2.1e-3
    gradient = float(jax.grad(objective)(1.0))
    reference = central(objective, 1.0, 1e-2 * gap)
    assert abs(gradient - reference) <= 1e-6 * abs(reference), (gradient, reference)
