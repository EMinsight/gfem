r"""Layer patterns and the fabrication chain, end to end.

V1 and V2 use the bend tutorial's 2D model (``docs/tutorials/data/bend2d.msh``,
E along z, 1.55 um, its design square on a 50 nm grid, filter radius 0.3 um)
with the quarter-circle arc of the tutorial drawn in the square: a guide 0.55 um
wide bent at a 1.5 um radius about the square's lower-left corner, joining the
two feed guides. Its filtered density is the design the optimizer would end at.

- V1: the pattern of that design (its :math:`\eta = 0.5` contour) is solved
  back as area fractions of the 50 nm pixels. Its :math:`T = |S_{21}|^2`
  matches the binary pattern drawn on a 6.25 nm grid on the same mesh, and the
  :math:`\beta = 32` projection the optimizer solves, to 0.02.
- V2: the gradient of a worst-case loss over the three printed designs of a
  fabrication model (filter, projection, Gaussian blur and threshold at three
  edge biases, then a solve each) matches a fourth-order difference.

V3 needs no solve: a 64 x 64 design matched by Adam to a target whose bars,
slots and bridge are 0.3 to 0.7 :math:`\ell` wide (below the rule
:math:`\ell`), first unconstrained, then with the length-scale constraints
asked at :math:`1.1\ell` as a quadratic penalty. The penalty brings the
pattern's narrowest width and spacing up to about the rule.
"""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from scipy.constants import c

import gfem
from gfem import design

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
um = 1e-6
CLAD = gfem.Material(eps_r=1.444**2)
CORE = gfem.Material(eps_r=2.25**2)
PROBLEM = gfem.Driven(frequency=c / (1.55 * um), excite="p1", polarization="Ez")
OPTIONS = gfem.SolverOptions(linear_solver="direct")
PITCH = 0.05 * um
RADIUS = 0.3 * um  # the filter's
WIDTH, BEND = 0.55 * um, 1.5 * um  # the guide, and the arc's radius
CORNER = -1.5 * um  # the arc's centre, the square's lower-left corner (x = y)


@pytest.fixture(scope="module")
def bend():
    """The 2D bend, its layer grid and the drawn arc's pixel area fractions."""
    mesh = gfem.Mesh.from_gmsh(DATA / "bend2d.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"cladding": CLAD, "core": CORE, "design": CLAD},
        boundaries=[gfem.Absorbing("outer", order=2)],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
    grid = design.LayerGrid(mesh, "design", pitch=PITCH)
    angles = np.linspace(0.0, 0.5 * np.pi, 65)
    outer = (BEND + WIDTH / 2) * np.stack([np.cos(angles), np.sin(angles)], 1)
    inner = (BEND - WIDTH / 2) * np.stack([np.cos(angles), np.sin(angles)], 1)
    arc = design.LayerPattern.from_polygons([CORNER + np.vstack([outer, inner[::-1]])])
    return mesh, model, grid, grid.area_fraction(arc)


def transmission(model, grid, pixels):
    r""":math:`|S_{21}|^2` of the design square holding ``pixels`` of ``grid``."""
    field = gfem.MaterialField.interpolate(grid.to_elements(pixels), CLAD, CORE)
    result = gfem.solve(model.with_material("design", field), PROBLEM, options=OPTIONS)
    return jnp.abs(result.s["p2", "p1"][0]) ** 2


def bilinear(rho, grid, x, y):
    """``rho`` read at points ``(x, y)`` as the grid reads it: bilinear between
    pixel centres, the edge pixels' values beyond the outermost centres."""
    fx = np.clip((x - grid.x[0]) / grid.pitch, 0, grid.shape[1] - 1)
    fy = np.clip((y - grid.y[0]) / grid.pitch, 0, grid.shape[0] - 1)
    i = np.minimum(np.floor(fx).astype(int), grid.shape[1] - 2)
    j = np.minimum(np.floor(fy).astype(int), grid.shape[0] - 2)
    tx, ty = fx - i, fy - j
    return (
        rho[j, i] * (1 - tx) * (1 - ty)
        + rho[j, i + 1] * tx * (1 - ty)
        + rho[j + 1, i] * (1 - tx) * ty
        + rho[j + 1, i + 1] * tx * ty
    )


def test_v1_the_pattern_solved_back_transmits_as_the_fine_pattern(bend):
    mesh, model, grid, arc = bend
    filtered = np.asarray(design.helmholtz_filter(grid, radius=RADIUS)(arc))
    raster = grid.area_fraction(grid.pattern(filtered, eta=0.5))
    t_raster = float(transmission(model, grid, raster))

    # The pattern itself, independent of the contouring: where the bilinear
    # field reaches 0.5, on pixels an eighth of the pitch (unsampled ones warn).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnsampledPixelsWarning)
        fine = design.LayerGrid(mesh, "design", pitch=PITCH / 8)
    x, y = np.meshgrid(fine.x, fine.y)
    layout = (bilinear(filtered, grid, x, y) >= 0.5).astype(float)
    t_fine = float(transmission(model, fine, layout))

    t_32 = float(transmission(model, grid, design.project(filtered, 32.0)))

    assert abs(t_raster - t_fine) <= 0.02, (t_raster, t_fine)
    assert abs(t_raster - t_32) <= 0.02, (t_raster, t_32)


def test_v2_worst_printed_design_gradient_matches_finite_differences(bend):
    _, model, grid, arc = bend
    blur = design.helmholtz_filter(grid, radius=RADIUS)
    fab = design.FabricationModel(grid, sigma=2 * PITCH)

    def worst(rho):
        printed = fab.robust(
            design.project(blur(rho), 8.0), beta=8.0, tolerance=0.02 * um
        )
        missed = jnp.stack([1 - transmission(model, grid, p) for p in printed])
        return 0.01 * jax.nn.logsumexp(missed / 0.01)

    # Pixel noise keeps every value off the projections' clamps, where a
    # difference sees a kink the derivative does not.
    noise = 0.05 * np.random.default_rng(0).standard_normal(arc.shape)
    start = jnp.asarray(np.clip(arc + noise, 0.0, 1.0))
    directions = unit_directions(np.random.default_rng(1), arc.shape, 2)
    assert_gradient_matches(worst, start, directions, step=1e-3)


# --- V3: the toy constrained optimization ------------------------------------------

MM = 1e-3
ELL = 0.1 * MM  # the rule, and the filter radius
THIN = 0.7  # the target's features, in ell


@pytest.fixture(scope="module")
def toy_grid(tmp_path_factory):
    """A 64 x 64 grid of pitch ell / 8 on a single-layer plate."""
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path_factory.mktemp("toy") / "toy.msh",
        sm.plate(0.8, 0.8, (("design", 0.05),), (64, 64, 1)),
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return design.LayerGrid(mesh, "design", pitch=ELL / 8)


def toy_target(grid):
    """Four bars 0.7 ell wide, 0.3 ell apart, and two blocks joined by a bridge
    0.56 ell wide: none can be matched with the rules held."""
    x, y = np.meshgrid(grid.x - grid.bounds[0], grid.y - grid.bounds[1])
    target = np.zeros(grid.shape)
    for k in range(4):
        bar = (np.abs(x - (1 + k) * ELL) < THIN / 2 * ELL) & (y > ELL) & (y < 7 * ELL)
        target[bar] = 1
    for lo, hi in ((0.5, 3.0), (5.0, 7.5)):
        target[(x > 5 * ELL) & (x < 7.5 * ELL) & (y > lo * ELL) & (y < hi * ELL)] = 1
    bridge = (np.abs(x - 6.25 * ELL) < 0.8 * THIN / 2 * ELL) & (y > 3 * ELL)
    target[bridge & (y < 5 * ELL)] = 1
    return jnp.asarray(target)


def adam(value_and_grad, rho, steps, rate, state=None):
    """``steps`` Adam steps on ``rho``, clipped to [0, 1]; returns (rho, state)."""
    m, v, t = state or (jnp.zeros_like(rho), jnp.zeros_like(rho), 0)
    for _ in range(steps):
        _, g = value_and_grad(rho)
        t += 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        step = rate * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        rho = jnp.clip(rho - step, 0.0, 1.0)
    return rho, (m, v, t)


def test_v3_length_scale_penalty_widens_the_narrowest_features(toy_grid):
    grid = toy_grid
    blur = design.helmholtz_filter(grid, radius=ELL)
    rules = design.length_scale_constraints(
        blur, min_width=1.1 * ELL, min_spacing=1.1 * ELL
    )
    target = toy_target(grid)

    def mismatch(rho, beta):  # scaled so its values lie in about 1-100
        return 1e4 * jnp.mean((design.project(blur(rho), beta) - target) ** 2)

    # Stage 1: unconstrained, continuation to beta = 32.
    rho = jnp.full(grid.shape, 0.5)
    for beta in (4.0, 8.0, 16.0, 32.0):
        step = jax.jit(jax.value_and_grad(lambda r, b=beta: mismatch(r, b)))
        rho, _ = adam(step, rho, 80, 0.05)
    first = grid.pattern(np.asarray(blur(rho)), eta=0.5).check(
        min_width=ELL, min_spacing=ELL
    )
    assert not first.passed  # the target's features are below the rule

    # Stage 2: the rules asked 10 % above ell, penalty weight raised.
    state = None
    for mu in (1.0, 10.0, 100.0):

        def penalized(r, mu=mu):
            held = jnp.stack(rules(blur(r)))
            return mismatch(r, 32.0) + mu * jnp.sum(jnp.maximum(held, 0.0) ** 2)

        rho, state = adam(jax.jit(jax.value_and_grad(penalized)), rho, 166, 0.02, state)
    # A penalty ends slightly infeasible: measured (0.44, 0.19).
    values = rules(blur(rho))
    assert float(values.solid) <= 0.5
    assert float(values.void) <= 0.5
    second = grid.pattern(np.asarray(blur(rho)), eta=0.5).check(
        min_width=ELL, min_spacing=ELL
    )
    # Unconstrained the narrowest are 0.47 ell (width) and 0.24 ell (spacing);
    # with the penalty, measured 1.051 ell and 0.906 ell. Held to 0.05 ell below
    # those, and in any case to 0.6 ell.
    width = second.narrowest_width / ELL
    spacing = second.narrowest_spacing / ELL
    assert width >= max(0.6, 1.051 - 0.05), width
    assert spacing >= max(0.6, 0.906 - 0.05), spacing
