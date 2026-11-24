"""The fabrication model: a Gaussian blur and a threshold on the drawn design."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import gfem
from gfem import design

MM = 1e-3
#: Pixel side of the plates: one mesh cell per pixel.
PITCH = 0.025 * MM
#: The blur, four pixels: the pitch rule's sigma / 4.
SIGMA = 4 * PITCH


def _face(tmp_path_factory, name, side, cells):
    """A 2D plate (the z = 0 face of a structured plate), area "design"."""
    import structured_meshes as sm

    face = sm.z0_face(sm.plate(side[0], side[1], (("design", 0.1),), (*cells, 1)))
    path = sm.write_msh22(tmp_path_factory.mktemp(name) / f"{name}.msh", face)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    """A 1 mm x 1 mm 2D plate, 40 x 40 cells."""
    return _face(tmp_path_factory, "square", (1.0, 1.0), (40, 40))


@pytest.fixture(scope="module")
def grid(square):
    return design.LayerGrid(square, "design", pitch=PITCH)


def _step(grid, edge):
    """A binary drawn design, solid at x > edge."""
    return np.broadcast_to((grid.x > edge).astype(float), grid.shape).copy()


def test_model_lives_in_gfem_design_and_keeps_its_parameters(grid):
    from gfem import _fabrication

    assert design.FabricationModel is _fabrication.FabricationModel
    assert "FabricationModel" in design.__all__
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=-0.5 * SIGMA)
    assert (fab.grid, fab.sigma, fab.bias) == (grid, SIGMA, -0.5 * SIGMA)
    assert repr(fab) == f"FabricationModel({grid!r}, sigma=0.0001 m, bias=-5e-05 m)"
    assert design.FabricationModel(grid, sigma=SIGMA).bias == 0.0


# --- bias is the edge shift -----------------------------------------------------------


@pytest.mark.parametrize("b", [-1.5, -0.5, 0.0, 0.5, 1.5])
def test_printed_edge_moves_out_by_the_bias(grid, b):
    _, ymin, xmax, ymax = grid.bounds
    edge = 0.5 * MM  # on a pixel boundary
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=b * SIGMA)
    pattern = fab.pattern(_step(grid, edge))
    assert len(pattern) == 1
    printed_edge = xmax - pattern.area / (ymax - ymin)
    assert abs(printed_edge - (edge - b * SIGMA)) <= 0.02 * SIGMA


def test_threshold_is_the_gaussian_tail_at_the_bias(grid):
    from scipy.stats import norm

    fab = design.FabricationModel(grid, sigma=SIGMA, bias=0.5 * SIGMA)
    # a straight edge prints where the blurred step, the normal CDF, crosses eta
    assert fab.threshold() == pytest.approx(norm.sf(0.5), rel=1e-12)
    for b in (-2.0, -1.0, 0.0, 0.3, 1.5, 2.0):
        assert fab.threshold(b * SIGMA) == pytest.approx(norm.sf(b), rel=1e-12)
    assert fab.threshold(0.0) == 0.5
    assert isinstance(fab.threshold(), float)


def test_printed_design_crosses_one_half_at_the_moved_edge(grid):
    edge, b = 0.5 * MM, 0.5 * SIGMA
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=b)
    printed = np.asarray(fab(_step(grid, edge), beta=64.0))
    row = printed[grid.shape[0] // 2]
    # solid from the pixel holding the moved edge on, void before it
    first = np.searchsorted(grid.x, edge - b)
    assert row[first] > 0.5 > row[first - 1]
    assert row[first - 4] < 1e-3
    assert row[first + 3] > 1 - 1e-3
    assert np.ptp(printed, axis=0).max() < 1e-12  # constant along y
    # a bias per call overrides the model's
    moved = np.asarray(fab(_step(grid, edge), beta=64.0, bias=-b))[0]
    first = np.searchsorted(grid.x, edge + b)
    assert moved[first] > 0.5 > moved[first - 1]


def _blob(grid):
    """A drawn design with curved edges, lines and gaps: a disk and a ring."""
    x, y = np.meshgrid(grid.x, grid.y)
    r = np.hypot(x - 0.5 * MM, y - 0.5 * MM)
    return ((r < 0.12 * MM) | ((r > 0.25 * MM) & (r < 0.33 * MM))).astype(float)


def test_robust_triple_is_nested_and_its_bias_spread_symmetric(grid):
    b, t = -0.25 * SIGMA, 0.5 * SIGMA
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=b)
    drawn = _blob(grid)
    triple = fab.robust(drawn, beta=16.0, tolerance=t)
    assert isinstance(triple, design.Robust)
    eroded, nominal, dilated = (np.asarray(d) for d in triple)
    assert (eroded <= nominal + 1e-15).all()
    assert (nominal <= dilated + 1e-15).all()
    assert (dilated - eroded).max() > 0.5
    np.testing.assert_allclose(nominal, fab(drawn, beta=16.0), atol=1e-15)
    # eroded and dilated are the printed designs at bias - t and bias + t
    np.testing.assert_allclose(eroded, fab(drawn, beta=16.0, bias=b - t), atol=1e-15)
    np.testing.assert_allclose(dilated, fab(drawn, beta=16.0, bias=b + t), atol=1e-15)
    # their straight edges sit t either side of the nominal one
    edges = []
    for shift in (-t, 0.0, t):
        pattern = fab.pattern(_step(grid, 0.5 * MM), bias=b + shift)
        edges.append(grid.bounds[2] - pattern.area / (grid.bounds[3] - grid.bounds[1]))
    assert edges[0] - edges[1] == pytest.approx(t, abs=0.02 * SIGMA)
    assert edges[1] - edges[2] == pytest.approx(t, abs=0.02 * SIGMA)


def test_robust_takes_a_bias_per_call_as_the_printed_design_does(grid, small):
    t = 0.25 * SIGMA
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=-0.25 * SIGMA)
    drawn = _blob(grid)
    for bias in (0.5 * SIGMA, None):
        own = fab.bias if bias is None else bias
        triple = fab.robust(drawn, beta=16.0, tolerance=t, bias=bias)
        for printed, shift in zip(triple, (-t, 0.0, t), strict=True):
            np.testing.assert_array_equal(
                printed, fab(drawn, beta=16.0, bias=own + shift)
            )
    # |b + t| beyond 2 sigma is refused for the bias given, not the model's
    with pytest.raises(ValueError, match=r"(?s)bias \+ tolerance.*2 sigma"):
        fab.robust(drawn, beta=16.0, tolerance=t, bias=1.9 * SIGMA)
    # a traced bias differentiates, the dilated design growing with it
    model = design.FabricationModel(small, sigma=2 * PITCH)
    rho = jax.random.uniform(jax.random.key(4), small.shape)
    grad = jax.grad(
        lambda b: jnp.sum(model.robust(rho, beta=8.0, tolerance=PITCH, bias=b).dilated)
    )(0.0)
    assert grad > 0


def test_without_bias_void_prints_as_solid_does(grid):
    # eta(-b) = 1 - eta(b) and the blur keeps constants: eroding the solid is
    # dilating the void
    fab = design.FabricationModel(grid, sigma=SIGMA)
    drawn = _blob(grid)
    solid = fab.robust(drawn, beta=16.0, tolerance=SIGMA)
    void = fab.robust(1 - drawn, beta=16.0, tolerance=SIGMA)
    np.testing.assert_allclose(void.dilated, 1 - solid.eroded, atol=1e-12)
    np.testing.assert_allclose(void.nominal, 1 - solid.nominal, atol=1e-12)


# --- the blur -------------------------------------------------------------------------


def test_blur_kernel_is_a_non_negative_unit_mass_of_variance_sigma_squared(grid):
    sigma = 2 * PITCH  # the pixel is ten blurs from the walls
    fab = design.FabricationModel(grid, sigma=sigma)
    i, j = 20, 20
    point = np.zeros(grid.shape)
    point[i, j] = 1.0
    kernel = np.asarray(fab.image(point))
    assert kernel.min() > -1e-15
    assert kernel.sum() == pytest.approx(1.0, abs=1e-12)
    x, y = np.meshgrid(grid.x - grid.x[j], grid.y - grid.y[i])
    assert np.sum(kernel * x) == pytest.approx(0.0, abs=1e-12 * sigma)
    assert np.sum(kernel * x**2) == pytest.approx(sigma**2, rel=1e-9)
    assert np.sum(kernel * y**2) == pytest.approx(sigma**2, rel=1e-9)
    assert np.sum(kernel * x * y) == pytest.approx(0.0, abs=1e-12 * sigma**2)


@pytest.mark.parametrize("b", [-0.5, 0.5])
def test_a_guide_meeting_a_wall_prints_its_width_plus_twice_the_bias(grid, b):
    # the reflecting blur continues a guide normal to the wall as itself
    xmin, _, xmax, _ = grid.bounds
    w = 4 * SIGMA
    guide = (np.abs(grid.y - 0.5 * MM) < w / 2).astype(float)
    drawn = np.broadcast_to(guide[:, None], grid.shape)
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=b * SIGMA)
    ((exterior, holes),) = fab.pattern(drawn)
    assert holes == ()
    for wall in (xmin, xmax):
        ys = exterior[np.isclose(exterior[:, 0], wall, rtol=0, atol=1e-12), 1]
        assert np.ptp(ys) == pytest.approx(w + 2 * b * SIGMA, abs=0.02 * SIGMA)


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    """The square plate as one period of a cell along x."""
    return _face(tmp_path_factory, "cell", (1.0, 1.0), (40, 40))


@pytest.fixture(scope="module")
def three_cells(tmp_path_factory):
    """Three periods of the cell side by side."""
    return _face(tmp_path_factory, "three", (3.0, 1.0), (120, 40))


def test_periodic_blur_wraps_as_a_tiled_grid_blurs(cell, three_cells):
    period = design.LayerGrid(cell, "design", pitch=PITCH, periodic=(True, False))
    tiled = design.LayerGrid(three_cells, "design", pitch=PITCH)
    drawn = np.roll(_blob(period), 17, axis=1)  # across the seam
    n = period.shape[1]
    wrapped = design.FabricationModel(period, sigma=SIGMA).image(drawn)
    middle = design.FabricationModel(tiled, sigma=SIGMA).image(np.tile(drawn, 3))
    np.testing.assert_allclose(wrapped, middle[:, n : 2 * n], atol=1e-12)
    walled = design.LayerGrid(cell, "design", pitch=PITCH)
    reflected = design.FabricationModel(walled, sigma=SIGMA).image(drawn)
    assert np.abs(np.asarray(wrapped - reflected)).max() > 0.1


# --- gradients and transforms ---------------------------------------------------------


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A 0.3 mm x 0.25 mm 2D plate, 12 x 10 cells."""
    mesh = _face(tmp_path_factory, "small", (0.3, 0.25), (12, 10))
    return design.LayerGrid(mesh, "design", pitch=PITCH)


def test_printed_design_has_second_derivatives_in_the_drawn_design_and_the_bias(
    small,
):
    fab = design.FabricationModel(small, sigma=2 * PITCH, bias=0.3 * PITCH)
    drawn = jax.random.uniform(jax.random.key(0), small.shape)
    check_grads(jax.jit(lambda rho: fab(rho, beta=8.0)), (drawn,), order=2)
    # the bias in pixels, so the finite-difference step is a fraction of one
    in_pixels = jax.jit(lambda u: fab(drawn, beta=8.0, bias=u * PITCH))
    check_grads(in_pixels, (0.3,), order=2)
    grad = jax.grad(lambda b: jnp.sum(fab(drawn, beta=8.0, bias=b)))(0.0)
    assert grad > 0  # a positive bias grows the solid


def test_worst_of_the_robust_triple_differentiates_through_the_whole_chain(small):
    blur = design.helmholtz_filter(small, radius=4 * PITCH)
    fab = design.FabricationModel(small, sigma=2 * PITCH, bias=-0.2 * PITCH)
    weights = 0.1 * jax.random.normal(jax.random.key(1), small.shape)

    def worst(rho):
        drawn = design.project(blur(rho), 8.0)
        printed = jnp.stack(fab.robust(drawn, beta=8.0, tolerance=0.5 * PITCH))
        losses = jnp.sum(printed * weights, axis=(1, 2))
        return 0.1 * jax.nn.logsumexp(losses / 0.1)

    rho = jax.random.uniform(jax.random.key(2), small.shape)
    check_grads(worst, (rho,), order=1, modes=("rev",))


def test_model_closes_over_into_jit_and_passes_as_a_static_argument(small):
    fab = design.FabricationModel(small, sigma=2 * PITCH, bias=0.4 * PITCH)
    drawn = jax.random.uniform(jax.random.key(3), small.shape)
    eager = fab(drawn, beta=8.0)
    np.testing.assert_allclose(jax.jit(lambda r: fab(r, beta=8.0))(drawn), eager)
    np.testing.assert_allclose(
        jax.jit(lambda f, r: f(r, beta=8.0))(fab, drawn), eager, rtol=1e-14
    )
    traced = jax.jit(lambda r, b: fab(r, beta=8.0, bias=b))(drawn, 0.4 * PITCH)
    np.testing.assert_allclose(traced, eager, rtol=1e-14)
    triple = jax.jit(lambda r: fab.robust(r, beta=8.0, tolerance=PITCH))(drawn)
    np.testing.assert_allclose(triple.nominal, eager, rtol=1e-14)


# --- errors ---------------------------------------------------------------------------


def test_blur_narrower_than_a_pixel_is_refused(grid):
    with pytest.raises(ValueError, match=r"(?s)sigma.*pitch.*sigma / 4"):
        design.FabricationModel(grid, sigma=0.9 * PITCH)
    design.FabricationModel(grid, sigma=PITCH)  # one pixel: resolved
    for sigma in (0.0, -SIGMA, float("nan")):
        with pytest.raises(ValueError, match="sigma must be"):
            design.FabricationModel(grid, sigma=sigma)


def test_bias_beyond_two_blurs_is_refused(grid):
    with pytest.raises(ValueError, match=r"(?s)bias.*2 sigma.*tail"):
        design.FabricationModel(grid, sigma=SIGMA, bias=2.1 * SIGMA)
    design.FabricationModel(grid, sigma=SIGMA, bias=-2 * SIGMA)
    fab = design.FabricationModel(grid, sigma=SIGMA, bias=1.5 * SIGMA)
    drawn = _step(grid, 0.5 * MM)
    with pytest.raises(ValueError, match="2 sigma"):
        fab(drawn, beta=8.0, bias=-2.5 * SIGMA)
    with pytest.raises(ValueError, match="2 sigma"):
        fab.threshold(3 * SIGMA)
    with pytest.raises(ValueError, match="2 sigma"):
        fab.pattern(drawn, bias=2.5 * SIGMA)
    with pytest.raises(ValueError, match=r"(?s)bias \+ tolerance.*2 sigma"):
        fab.robust(drawn, beta=8.0, tolerance=SIGMA)
    fab.robust(drawn, beta=8.0, tolerance=0.5 * SIGMA)  # dilated at 2 sigma


def test_tolerance_must_be_positive(grid):
    fab = design.FabricationModel(grid, sigma=SIGMA)
    for tolerance in (0.0, -0.1 * SIGMA):
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            fab.robust(_step(grid, 0.5 * MM), beta=8.0, tolerance=tolerance)


def test_model_takes_a_layer_grid_not_a_mesh(square):
    with pytest.raises(TypeError, match="takes a LayerGrid"):
        design.FabricationModel(square, sigma=SIGMA)


def test_model_takes_the_pixel_array_not_a_layer_density(grid):
    fab = design.FabricationModel(grid, sigma=SIGMA)
    density = grid.to_elements(jnp.asarray(_step(grid, 0.5 * MM)))
    assert isinstance(density, design.LayerDensity)
    for call in (
        fab.image,
        lambda d: fab(d, beta=8.0),
        lambda d: fab.robust(d, beta=8.0, tolerance=SIGMA),
        fab.pattern,
    ):
        with pytest.raises(TypeError, match=r"density\.pixels"):
            call(density)


def test_drawn_design_of_the_wrong_shape_or_complex_is_refused(grid):
    fab = design.FabricationModel(grid, sigma=SIGMA)
    drawn = _step(grid, 0.5 * MM)
    with pytest.raises(ValueError, match=r"(?s)shape.*\(40, 40\)"):
        fab(drawn[:-1], beta=8.0)
    with pytest.raises(ValueError, match="real"):
        fab.image(drawn + 0j)
    with pytest.raises(ValueError, match="beta must be > 0"):
        fab(drawn, beta=0.0)


def test_printed_pattern_is_computed_outside_transforms(grid):
    fab = design.FabricationModel(grid, sigma=SIGMA)
    drawn = jnp.asarray(_step(grid, 0.5 * MM))
    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.jit(lambda r: fab.pattern(r).area)(drawn)
    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.jit(lambda b: fab.pattern(drawn, bias=b).area)(0.0)
