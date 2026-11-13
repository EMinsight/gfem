"""The dispersive flagship: the bend tutorial's design square in gold and oxide.

The metal-bend tutorial's model (docs/tutorials/metal-bend.ipynb) on
``docs/tutorials/data/bend.msh``: the 2D slab bend, its design square a
refractive-index blend of the cladding (n = 1.444) and a gold-like Drude metal
(plasma frequency 2.183e15 Hz, damping 1.28e13 Hz), one driven uniform sweep
at 1.50, 1.55 and 1.60 um. The objective is the band mean of 1 - |S21|^2.

The 45 degree mirror (gold where x + y > 0.6 um) pins the forward values; its
band objective's gradient with respect to the pixels, through the tutorial's
chain (mirror about y = x, Helmholtz filter, projection at beta = 4), is
checked against a fourth-order difference. The design gets a little pixel
noise first, so that no filtered density sits on the projection's clamp at 0
or 1, where the clamp's derivative is zero and a difference is not.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches
from scipy.constants import c

import gfem
from gfem import design

MESH = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "bend.msh"
um = 1e-6
FREQUENCIES = [c / (w * um) for w in (1.50, 1.55, 1.60)]
CLADDING = gfem.Material(eps_r=1.444**2)
GOLD = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        terms=[gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)]
    )
)
PROBLEM = gfem.Driven(frequency=FREQUENCIES, excite="p1")
OPTIONS = gfem.SolverOptions(linear_solver="direct")


@pytest.fixture(scope="module")
def bend():
    """The model, its layer grid, and the 45 degree mirror's pixels."""
    mesh = gfem.Mesh.from_gmsh(MESH, length_unit="um")
    model = gfem.Model(
        mesh,
        materials={
            "cladding": CLADDING,
            "core": gfem.Material(eps_r=2.25**2),
            "design": CLADDING,
        },
        boundaries=[gfem.PEC("plates"), gfem.Absorbing("outer", order=2)],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
    grid = design.LayerGrid(mesh, "design", pitch=0.05 * um)
    # Area fraction of gold per pixel, 8 x 8 samples.
    offsets = ((np.arange(8) + 0.5) / 8 - 0.5) * grid.pitch
    xs = grid.x[None, :, None, None] + offsets[None, None, None, :]
    ys = grid.y[:, None, None, None] + offsets[None, None, :, None]
    mirror = ((xs + ys) > 0.6 * um).mean(axis=(2, 3))
    return model, grid, mirror


def _band(model, grid, density):
    field = gfem.MaterialField.interpolate(
        grid.to_elements(density), CLADDING, GOLD, scheme="index"
    )
    result = gfem.solve(model.with_material("design", field), PROBLEM, options=OPTIONS)
    return jnp.abs(result.s["p2", "p1"]) ** 2


def test_mirror_band_transmission(bend):
    model, grid, mirror = bend
    t = np.asarray(_band(model, grid, jnp.asarray(mirror)))
    np.testing.assert_allclose(t, [0.2243, 0.2108, 0.1928], atol=1e-3)


def test_band_objective_gradient_through_the_blend(bend):
    model, grid, mirror = bend
    blur = design.helmholtz_filter(grid, radius=0.3 * um)

    def missed(rho):
        density = design.project(blur(0.5 * (rho + rho.T)), 4.0)
        return jnp.mean(1 - _band(model, grid, density))

    noise = 0.05 * np.random.default_rng(0).standard_normal(mirror.shape)
    start = jnp.asarray(np.clip(mirror + noise, 0.0, 1.0))
    direction = np.random.default_rng(1).standard_normal(mirror.shape)
    direction /= np.linalg.norm(direction)
    assert_gradient_matches(missed, start, [direction], step=1e-4, rel=1e-6)
