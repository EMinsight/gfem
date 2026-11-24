"""The dispersive flagship: the bend tutorial's design square in gold and oxide.

The metal-bend tutorial's bend, its design square a refractive-index blend of
the cladding (n = 1.444) and a gold-like Drude metal (plasma frequency
2.183e15 Hz, damping 1.28e13 Hz), one driven uniform sweep at 1.50, 1.55 and
1.60 um. The objective is the band mean of 1 - |S21|^2.
It is posed twice: as the tutorial's 2D model with E along z on
``docs/tutorials/data/bend2d.msh``, and as a slab one cell thick between PEC
plates on ``docs/tutorials/data/bend.msh``, its z = 0 face extruded.

The 45 degree mirror (gold where x + y > 0.6 um) pins the forward values of
each; the two differ by up to 6e-3 because the slab reads the pixels at its
element centroids and the 2D model at several points per triangle. Its
band objective's gradient with respect to the pixels, through the tutorial's
chain (mirror about y = x, Helmholtz filter, projection at beta = 4), is
checked against a fourth-order difference. The design gets a little pixel
noise first, so that no filtered density sits on the projection's clamp at 0
or 1, where the clamp's derivative is zero and a difference is not.

With the design square all gold (a plain law material, no blend), the heat
it dissipates is the power the field brings into it: `gfem.dissipated_power`
against `gfem.power_flow`. The absorbing outer boundary keeps S out of this
identity, and the blended bend is the dissipated power's refusal case.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches
from scipy.constants import c

import gfem
from gfem import design

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
MESH = DATA / "bend.msh"
MESH_2D = DATA / "bend2d.msh"
um = 1e-6
FREQUENCIES = [c / (w * um) for w in (1.50, 1.55, 1.60)]
CLADDING = gfem.Material(eps_r=1.444**2)
GOLD = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        terms=[gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)]
    )
)
PROBLEM = gfem.Driven(frequency=FREQUENCIES, excite="p1")
PROBLEM_2D = gfem.Driven(frequency=FREQUENCIES, excite="p1", polarization="Ez")
OPTIONS = gfem.SolverOptions(linear_solver="direct")


def _model(mesh, boundaries):
    return gfem.Model(
        mesh,
        materials={
            "cladding": CLADDING,
            "core": gfem.Material(eps_r=2.25**2),
            "design": CLADDING,
        },
        boundaries=boundaries,
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )


def _mirror(grid):
    # Area fraction of gold per pixel, 8 x 8 samples.
    offsets = ((np.arange(8) + 0.5) / 8 - 0.5) * grid.pitch
    xs = grid.x[None, :, None, None] + offsets[None, None, None, :]
    ys = grid.y[:, None, None, None] + offsets[None, None, :, None]
    return ((xs + ys) > 0.6 * um).mean(axis=(2, 3))


@pytest.fixture(scope="module")
def bend():
    """The slab model, its layer grid, the 45 degree mirror's pixels, the problem."""
    mesh = gfem.Mesh.from_gmsh(MESH, length_unit="um")
    model = _model(mesh, [gfem.PEC("plates"), gfem.Absorbing("outer", order=2)])
    grid = design.LayerGrid(mesh, "design", pitch=0.05 * um)
    return model, grid, _mirror(grid), PROBLEM


@pytest.fixture(scope="module")
def bend_2d():
    """The 2D model, its layer grid, the 45 degree mirror's pixels, the problem."""
    mesh = gfem.Mesh.from_gmsh(MESH_2D, length_unit="um")
    model = _model(mesh, [gfem.Absorbing("outer", order=2)])
    grid = design.LayerGrid(mesh, "design", pitch=0.05 * um)
    return model, grid, _mirror(grid), PROBLEM_2D


def _band(model, grid, density, problem):
    field = gfem.MaterialField.interpolate(
        grid.to_elements(density), CLADDING, GOLD, scheme="index"
    )
    result = gfem.solve(model.with_material("design", field), problem, options=OPTIONS)
    return jnp.abs(result.s["p2", "p1"]) ** 2


def test_mirror_band_transmission(bend):
    model, grid, mirror, problem = bend
    t = np.asarray(_band(model, grid, jnp.asarray(mirror), problem))
    np.testing.assert_allclose(t, [0.2243, 0.2108, 0.1928], atol=1e-3)


def test_mirror_band_transmission_on_the_2d_model(bend_2d):
    model, grid, mirror, problem = bend_2d
    t = np.asarray(_band(model, grid, jnp.asarray(mirror), problem))
    np.testing.assert_allclose(t, [0.2283, 0.2161, 0.1988], atol=1e-3)


@pytest.mark.parametrize("posed", ["bend", "bend_2d"])
def test_band_objective_gradient_through_the_blend(posed, request):
    model, grid, mirror, problem = request.getfixturevalue(posed)
    blur = design.helmholtz_filter(grid, radius=0.3 * um)

    def missed(rho):
        density = design.project(blur(0.5 * (rho + rho.T)), 4.0)
        return jnp.mean(1 - _band(model, grid, density, problem))

    noise = 0.05 * np.random.default_rng(0).standard_normal(mirror.shape)
    start = jnp.asarray(np.clip(mirror + noise, 0.0, 1.0))
    direction = np.random.default_rng(1).standard_normal(mirror.shape)
    direction /= np.linalg.norm(direction)
    assert_gradient_matches(missed, start, [direction], step=1e-4, rel=1e-6)


def test_a_gold_design_dissipates_what_power_flow_says_enters_it(bend):
    # the design square all gold, a plain Drude law: the heat kernel reads the
    # law's complex permittivity at the frequency. The absorbing outer
    # boundary keeps S out of this identity; the blended bend is refused.
    model, _, _, _ = bend
    gold = model.with_material("design", GOLD)
    problem = gfem.Driven(frequency=FREQUENCIES[1], excite="p1", keep_fields=True)
    result = gfem.solve(gold, problem, options=OPTIONS)
    field = result.e_field(FREQUENCIES[1], "p1")
    lost = gfem.dissipated_power(field, model=gold)
    entering = -float(gfem.power_flow(field, "design", model=gold))
    assert entering > 1e-3
    # gate 1e-12 (9e-15 measured)
    assert float(lost["design"]) == pytest.approx(entering, rel=1e-12)
    assert float(lost["core"]) == 0.0
