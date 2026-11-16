"""Design helpers: projection, the robust triple, filters and the layer grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import gfem
from gfem import design

# --- project -----------------------------------------------------------------------


def test_project_keeps_void_solid_and_threshold():
    rho = jnp.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(design.project(rho, beta=8.0), [0.0, 0.5, 1.0])
    # an off-centre threshold still maps 0 -> 0 and 1 -> 1
    np.testing.assert_allclose(design.project(rho[::2], beta=8.0, eta=0.3), [0, 1])


def test_project_sharpens_towards_a_step_as_beta_grows():
    rho = jnp.array([0.4, 0.6])
    soft = design.project(rho, beta=1.0)
    hard = design.project(rho, beta=64.0)
    assert 0.39 < soft[0] < 0.41  # nearly linear at beta = 1 (tanh x ~ x)
    np.testing.assert_allclose(hard, [0.0, 1.0], atol=1e-5)


def test_project_rejects_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError, match="eta"):
        design.project(jnp.zeros(2), beta=8.0, eta=1.2)
    with pytest.raises(ValueError, match="beta"):
        design.project(jnp.zeros(2), beta=0.0)


def test_project_gradients():
    rho = jnp.linspace(0.05, 0.95, 7)
    check_grads(lambda r: design.project(r, beta=6.0, eta=0.4), (rho,), order=2)
    check_grads(lambda b: design.project(rho, beta=b), (4.0,), order=1)


@pytest.mark.parametrize("beta", [16.0, 32.0, 48.0])
def test_project_stays_in_the_unit_interval_against_round_off(beta):
    rho = jnp.concatenate(
        [jnp.linspace(0, 0.05, 100001), jnp.linspace(0.95, 1, 100001)]
    )
    p = np.asarray(design.project(rho, beta=beta))
    assert p.min() >= 0.0
    assert p.max() <= 1.0
    assert p[0] == 0.0
    assert p[-1] == 1.0


def test_project_gradient_at_void_and_solid_is_the_tanh_slope():
    # A clamp that halves the slope at a tie (jnp.clip) fails this.
    grad = jax.grad(lambda r: design.project(r, beta=8.0).sum())(jnp.array([0.0, 1.0]))
    slope = 8 / np.cosh(4.0) ** 2 / (2 * np.tanh(4.0))
    np.testing.assert_allclose(grad, [slope, slope], rtol=1e-12)


def test_sheet_field_accepts_a_projected_density():
    fill = design.project(jnp.linspace(0, 0.05, 100001), beta=32.0)
    field = gfem.SheetField.interpolate(fill)
    assert np.asarray(field.fill).min() == 0.0


# --- robust ------------------------------------------------------------------------


def test_robust_eroded_uses_the_higher_threshold():
    rho = jnp.linspace(0, 1, 11)
    r = design.robust(rho, beta=8.0, delta=0.2)
    np.testing.assert_allclose(r.eroded, design.project(rho, beta=8.0, eta=0.7))
    np.testing.assert_allclose(r.nominal, design.project(rho, beta=8.0, eta=0.5))
    np.testing.assert_allclose(r.dilated, design.project(rho, beta=8.0, eta=0.3))
    # eroded has less solid than nominal, dilated more
    assert jnp.sum(r.eroded) < jnp.sum(r.nominal) < jnp.sum(r.dilated)


def test_robust_triple_stacks_for_vmap():
    rho = jnp.linspace(0, 1, 5)
    stacked = jnp.stack(design.robust(rho, beta=4.0, delta=0.1))
    assert stacked.shape == (3, 5)
    sums = jax.vmap(jnp.sum)(stacked)
    assert sums.shape == (3,)


def test_robust_rejects_a_delta_that_leaves_the_unit_interval():
    with pytest.raises(ValueError, match="delta"):
        design.robust(jnp.zeros(3), beta=4.0, delta=0.6)
    with pytest.raises(ValueError, match="delta"):
        design.robust(jnp.zeros(3), beta=4.0, delta=0.0)


def test_robust_gradients():
    rho = jnp.linspace(0.05, 0.95, 6)

    def worst(r):
        return jnp.max(jnp.stack(design.robust(r, beta=5.0, delta=0.1)) @ r)

    check_grads(lambda r: design.robust(r, beta=5.0, delta=0.1), (rho,), order=2)
    check_grads(worst, (rho,), order=1, modes=["rev"])


# --- LayerGrid ---------------------------------------------------------------------

MM = 1e-3


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A 1 mm x 0.5 mm stack: substrate, a two-cell-thick design layer, air."""
    import structured_meshes as sm

    import gfem

    layers = (("substrate", 0.2), ("design", 0.1), ("air", 0.3))
    path = tmp_path_factory.mktemp("stack") / "stack.msh"
    sm.write_msh22(path, sm.plate(1.0, 0.5, layers, (8, 4, 2)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _centroids(mesh, region):
    nodes = np.asarray(mesh.nodes)
    return nodes[mesh.tetrahedra[mesh.elements(region)]].mean(axis=1)


def test_layer_grid_covers_the_region_footprint(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    assert grid.shape == (2, 4)  # (ny, nx)
    assert grid.region == "design"
    assert grid.pitch == pytest.approx(0.25 * MM)
    np.testing.assert_allclose(grid.origin, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(grid.x, np.array([0.125, 0.375, 0.625, 0.875]) * MM)
    np.testing.assert_allclose(grid.y, np.array([0.125, 0.375]) * MM)
    assert grid.n_elements == len(stack.elements("design"))


def test_layer_grid_reproduces_a_linear_density_inside_the_pixel_centres(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    X, Y = np.meshgrid(grid.x, grid.y)
    rho = 0.2 + 0.4 * X / MM + 0.6 * Y / MM  # bilinear interpolation is exact
    values = np.asarray(grid.to_elements(jnp.asarray(rho)))
    c = _centroids(stack, "design")
    inside = (
        (c[:, 0] >= grid.x[0]) & (c[:, 0] <= grid.x[-1])
        & (c[:, 1] >= grid.y[0]) & (c[:, 1] <= grid.y[-1])
    )  # fmt: skip
    assert inside.sum() > 20
    np.testing.assert_allclose(
        values[inside], 0.2 + 0.4 * c[inside, 0] / MM + 0.6 * c[inside, 1] / MM
    )
    # outside the span of pixel centres the edge pixels' values hold
    assert values.min() >= rho.min() - 1e-12
    assert values.max() <= rho.max() + 1e-12


def test_layer_grid_is_constant_through_the_thickness(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    rho = jax.random.uniform(jax.random.key(0), grid.shape)
    values = np.asarray(grid.to_elements(rho))
    c = _centroids(stack, "design")
    lower = c[:, 2] < 0.25 * MM  # the design layer is two cells thick
    key = np.round(c[:, :2] / MM, 9)
    by_xy = {tuple(k): v for k, v in zip(key[lower], values[lower], strict=True)}
    above = [by_xy[tuple(k)] for k in key[~lower]]
    np.testing.assert_array_equal(values[~lower], above)


def test_layer_grid_warns_about_pixels_no_element_samples(stack):
    with pytest.warns(
        gfem.UnsampledPixelsWarning, match=r"(?s)\d+ of 800 pixels.*coarsen"
    ):
        grid = design.LayerGrid(stack, "design", pitch=0.025 * MM)
    assert grid.shape == (20, 40)
    assert grid.sampled.shape == grid.shape
    assert 0 < grid.sampled.sum() < grid.sampled.size
    g = jax.grad(lambda r: jnp.sum(grid.to_elements(r) ** 2))(jnp.ones(grid.shape))
    np.testing.assert_array_equal(np.asarray(g)[~grid.sampled], 0.0)
    assert (np.asarray(g)[grid.sampled] > 0).all()


def test_layer_grid_rejects_a_density_of_the_wrong_shape(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        grid.to_elements(jnp.zeros(8))


def test_layer_grid_on_an_unknown_region_or_bad_pitch_raises(stack):
    import gfem

    with pytest.raises(gfem.RegionNotFoundError):
        design.LayerGrid(stack, "desing", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="pitch"):
        design.LayerGrid(stack, "design", pitch=-1.0)


# --- LayerGrid on a sheet (a horizontal surface) ----------------------------------


def _face_centroids(mesh, surface):
    nodes = np.asarray(mesh.nodes)
    return nodes[mesh.triangles[mesh.elements(surface)]].mean(axis=1)


def test_surface_grid_covers_the_sheet_and_maps_pixels_to_faces(stack):
    sheet = "substrate__design"
    grid = design.LayerGrid(stack, sheet, pitch=0.25 * MM)
    assert grid.shape == (2, 4)
    assert grid.n_faces == len(stack.elements(sheet))
    assert grid.n_elements is None
    X, Y = np.meshgrid(grid.x, grid.y)
    rho = 0.2 + 0.4 * X / MM + 0.6 * Y / MM
    values = np.asarray(grid.to_faces(jnp.asarray(rho)))
    c = _face_centroids(stack, sheet)
    inside = (
        (c[:, 0] >= grid.x[0]) & (c[:, 0] <= grid.x[-1])
        & (c[:, 1] >= grid.y[0]) & (c[:, 1] <= grid.y[-1])
    )  # fmt: skip
    assert inside.sum() > 20
    np.testing.assert_allclose(
        values[inside], 0.2 + 0.4 * c[inside, 0] / MM + 0.6 * c[inside, 1] / MM
    )
    check_grads(grid.to_faces, (jnp.asarray(rho),), order=2)
    assert "faces" in repr(grid)


def test_surface_and_volume_grids_name_each_others_map(stack):
    faces = design.LayerGrid(stack, "substrate__design", pitch=0.25 * MM)
    volume = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="to_faces"):
        faces.to_elements(jnp.zeros(faces.shape))
    with pytest.raises(ValueError, match="to_elements"):
        volume.to_faces(jnp.zeros(volume.shape))
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        faces.to_faces(jnp.zeros(8))


def test_surface_grid_needs_a_horizontal_surface(stack):
    with pytest.raises(ValueError, match=r"(?s)not horizontal.*\d+ of"):
        design.LayerGrid(stack, "side", pitch=0.25 * MM)


def test_surface_grid_filters_on_its_pixels(stack):
    grid = design.LayerGrid(stack, "substrate__design", pitch=0.25 * MM)
    blur = design.helmholtz_filter(grid, radius=0.5 * MM)
    rho = jax.random.uniform(jax.random.key(2), grid.shape)
    assert blur(rho).shape == grid.shape


def test_layer_grid_gradients(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    rho = jax.random.uniform(jax.random.key(1), grid.shape)
    check_grads(grid.to_elements, (rho,), order=2)
    jitted = jax.jit(grid.to_elements)
    np.testing.assert_allclose(jitted(rho), grid.to_elements(rho))


# --- helmholtz_filter --------------------------------------------------------------
# The filter solves -r^2 lap(u) + u = rho with Neumann edges, r = radius / (2
# sqrt(3)) (the cone filter of the same radius has the same second moment). On a
# cosine of wavenumber k that fits the Neumann edges the exact attenuation is
# 1 / (1 + r^2 k^2).


def _attenuation(radius, k):
    return 1 / (1 + (radius / (2 * np.sqrt(3))) ** 2 * k**2)


@pytest.fixture(scope="module")
def fine(tmp_path_factory):
    """A 100 x 50 pixel grid on a 1 mm x 0.5 mm single-layer plate."""
    import structured_meshes as sm

    import gfem

    path = sm.write_msh22(
        tmp_path_factory.mktemp("fine") / "fine.msh",
        sm.plate(1.0, 0.5, (("design", 0.1),), (100, 50, 1)),
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return design.LayerGrid(mesh, "design", pitch=0.01 * MM)


def test_grid_filter_attenuates_a_cosine_as_the_pde_says(fine):
    k = np.pi / (1.0 * MM)  # half a period across the 1 mm footprint
    rho = np.cos(k * fine.x)[None, :] * np.ones((fine.shape[0], 1))
    radius = 0.3 * MM
    out = np.asarray(design.helmholtz_filter(fine, radius=radius)(jnp.asarray(rho)))
    np.testing.assert_allclose(out, _attenuation(radius, k) * rho, atol=2e-3)
    np.testing.assert_allclose(
        design.helmholtz_filter(fine, radius=radius)(jnp.ones(fine.shape)), 1.0
    )  # constants pass


def test_grid_filter_conserves_the_total_and_is_symmetric(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    blur = design.helmholtz_filter(grid, radius=0.4 * MM)
    a, b = jax.random.uniform(jax.random.key(2), (2, *grid.shape))
    np.testing.assert_allclose(jnp.sum(blur(a)), jnp.sum(a))
    np.testing.assert_allclose(jnp.vdot(a, blur(b)), jnp.vdot(blur(a), b))
    assert jnp.ptp(blur(a)) < jnp.ptp(a)


def test_mesh_filter_attenuates_a_cosine_as_the_pde_says(tmp_path):
    import structured_meshes as sm

    import gfem

    path = sm.write_msh22(
        tmp_path / "bar.msh", sm.plate(1.0, 0.2, (("design", 0.2),), (24, 2, 2))
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    k = np.pi / (1.0 * MM)
    c = _centroids(mesh, "design")
    radius = 0.3 * MM
    out = design.helmholtz_filter(mesh, "design", radius=radius)(np.cos(k * c[:, 0]))
    np.testing.assert_allclose(
        out, _attenuation(radius, k) * np.cos(k * c[:, 0]), atol=1e-2
    )


def test_mesh_filter_keeps_constants_and_the_volume_weighted_total(stack):
    blur = design.helmholtz_filter(stack, "design", radius=0.2 * MM)
    n = len(stack.elements("design"))
    np.testing.assert_allclose(blur(jnp.ones(n)), 1.0)
    nodes = np.asarray(stack.nodes)
    p = nodes[stack.tetrahedra[stack.elements("design")]]
    vol = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6
    rho = jax.random.uniform(jax.random.key(3), (n,))
    np.testing.assert_allclose(vol @ blur(rho), vol @ rho)
    assert jnp.ptp(blur(rho)) < jnp.ptp(rho)


def test_a_grid_needs_fixed_mesh_nodes(stack):
    def build(mesh):
        design.LayerGrid(mesh, "design", pitch=0.25 * MM)
        return mesh.nodes.sum()

    with pytest.raises(
        ValueError, match=r"computed inside jax\.grad, jax\.jit or jax\.vmap"
    ):
        jax.jit(build)(stack)


def test_filter_arguments_are_checked(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="LayerGrid"):
        design.helmholtz_filter(stack, "substrate__design", radius=1e-4)
    with pytest.raises(ValueError, match="region"):
        design.helmholtz_filter(stack, radius=1e-4)
    with pytest.raises(ValueError, match="region"):
        design.helmholtz_filter(grid, "design", radius=1e-4)
    with pytest.raises(ValueError, match="radius"):
        design.helmholtz_filter(grid, radius=0.0)
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        design.helmholtz_filter(grid, radius=1e-4)(jnp.zeros(3))


@pytest.mark.parametrize("carrier", ["mesh", "grid"])
def test_filter_gradients_and_transforms(stack, carrier):
    if carrier == "grid":
        grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
        blur, shape = design.helmholtz_filter(grid, radius=0.3 * MM), grid.shape
    else:
        blur = design.helmholtz_filter(stack, "design", radius=0.3 * MM)
        shape = (len(stack.elements("design")),)
    rho = jax.random.uniform(jax.random.key(4), shape)
    check_grads(blur, (rho,), order=2)
    np.testing.assert_allclose(jax.jit(blur)(rho), blur(rho), rtol=1e-12)
    batch = jax.random.uniform(jax.random.key(5), (3, *shape))
    np.testing.assert_allclose(
        jax.vmap(blur)(batch), np.stack([blur(b) for b in batch]), rtol=1e-12
    )


# --- periodic LayerGrid -------------------------------------------------------------
# On a periodic axis the grid's edges are the seams of a periodic cell: the filter
# is circulant there (an FFT instead of the cosine transform) and the bilinear map
# wraps instead of clamping, so the length scale holds across the seam.


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """A 1 mm x 0.5 mm single-layer plate, 40 x 20 cells."""
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path_factory.mktemp("plate") / "plate.msh",
        sm.plate(1.0, 0.5, (("design", 0.1),), (40, 20, 1)),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _seam_bump(grid, centre=0.0):
    """A Gaussian bump across the x = 0 seam, at x = centre, over the period."""
    period = grid.shape[1] * grid.pitch
    d = (grid.x - grid.origin[0] - centre + period / 2) % period - period / 2
    return np.exp(-0.5 * (d / (0.08 * MM)) ** 2)[None, :] * np.ones((grid.shape[0], 1))


@pytest.mark.parametrize("periodic", [(True, False), (True, True)])
def test_periodic_filter_is_invariant_under_a_shift_across_the_seam(plate, periodic):
    grid = design.LayerGrid(plate, "design", pitch=0.05 * MM, periodic=periodic)
    assert grid.periodic == periodic
    blur = design.helmholtz_filter(grid, radius=0.3 * MM)
    rho = (
        _seam_bump(grid, 0.07 * MM)
        * (1 + 0.5 * np.cos(2 * np.pi * grid.y / (0.5 * MM)))[:, None]
    )
    half = grid.shape[1] // 2
    shifted = np.roll(np.asarray(blur(jnp.roll(rho, half, axis=1))), -half, axis=1)
    np.testing.assert_allclose(blur(rho), shifted, atol=1e-14)
    if periodic[1]:
        half_y = grid.shape[0] // 2
        shifted = np.roll(np.asarray(blur(jnp.roll(rho, half_y, 0))), -half_y, 0)
        np.testing.assert_allclose(blur(rho), shifted, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(blur(rho)), rho.sum())
    # a bump centred on the seam spreads as much on both sides of it
    out = np.asarray(blur(_seam_bump(grid)))
    np.testing.assert_allclose(out[:, :half], out[:, ::-1][:, :half], atol=1e-14)


def test_neumann_filter_is_not_invariant_across_the_seam(plate):
    grid = design.LayerGrid(plate, "design", pitch=0.05 * MM)
    assert grid.periodic == (False, False)
    blur = design.helmholtz_filter(grid, radius=0.3 * MM)
    rho = _seam_bump(grid, 0.07 * MM)
    half = grid.shape[1] // 2
    shifted = np.roll(np.asarray(blur(jnp.roll(rho, half, axis=1))), -half, axis=1)
    assert np.abs(np.asarray(blur(rho)) - shifted).max() > 1e-2


def test_periodic_filter_attenuates_a_full_period_cosine_as_the_pde_says(plate):
    grid = design.LayerGrid(plate, "design", pitch=0.025 * MM, periodic=(True, True))
    k = 2 * np.pi / (1.0 * MM)  # one period across the 1 mm cell
    rho = np.cos(k * grid.x)[None, :] + np.sin(2 * k * grid.y)[:, None]
    radius = 0.3 * MM
    out = np.asarray(design.helmholtz_filter(grid, radius=radius)(jnp.asarray(rho)))
    expected = (
        _attenuation(radius, k) * np.cos(k * grid.x)[None, :]
        + _attenuation(radius, 2 * k) * np.sin(2 * k * grid.y)[:, None]
    )
    np.testing.assert_allclose(out, expected, atol=5e-3)


def test_default_grid_filter_and_map_are_bit_identical_to_explicit_neumann(stack):
    plain = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    neumann = design.LayerGrid(
        stack, "design", pitch=0.25 * MM, periodic=(False, False)
    )
    rho = jax.random.uniform(jax.random.key(7), plain.shape)
    np.testing.assert_array_equal(
        design.helmholtz_filter(plain, radius=0.3 * MM)(rho),
        design.helmholtz_filter(neumann, radius=0.3 * MM)(rho),
    )
    np.testing.assert_array_equal(plain.to_elements(rho), neumann.to_elements(rho))
    # the cosine-transform filter of today, by scipy
    import scipy.fft

    h, length = plain.pitch, 0.3 * MM * design.PDE_LENGTH_PER_RADIUS
    ky, kx = (
        (2 / h * np.sin(np.pi * np.arange(n) / (2 * n))) ** 2 for n in plain.shape
    )
    gain = 1 / (1 + length**2 * (ky[:, None] + kx[None, :]))
    reference = scipy.fft.idctn(
        gain * scipy.fft.dctn(np.asarray(rho), type=2, norm="ortho"), norm="ortho"
    )
    np.testing.assert_allclose(
        design.helmholtz_filter(plain, radius=0.3 * MM)(rho), reference, atol=1e-14
    )
    assert "periodic" not in repr(plain)


def test_wrapped_map_is_continuous_across_the_seam(plate):
    grid = design.LayerGrid(plate, "design", pitch=0.025 * MM, periodic=(True, False))
    period, h = grid.shape[1] * grid.pitch, grid.pitch
    c = _centroids(plate, "design")
    row = np.asarray(jax.random.uniform(jax.random.key(8), (grid.shape[1],)))
    rho = np.broadcast_to(row, grid.shape)
    values = np.asarray(grid.to_elements(jnp.asarray(rho)))
    # within half a pixel of the seam, on either side, the map is the linear
    # interpolation between the last and the first pixel: it meets itself at
    # x = 0 and x = period with the value (rho[-1] + rho[0]) / 2
    s = (c[:, 0] - grid.origin[0] + h / 2) % period  # distance past the last centre
    near = s < h
    assert near.sum() > 10
    assert (c[near, 0] < grid.x[0]).any()
    assert (c[near, 0] > grid.x[-1]).any()
    t = s[near] / h
    np.testing.assert_allclose(values[near], (1 - t) * row[-1] + t * row[0])
    # a smooth periodic density is reproduced to O(h^2) everywhere, seam included
    smooth = np.sin(2 * np.pi * grid.x / period)[None, :] * np.ones((grid.shape[0], 1))
    exact = np.sin(2 * np.pi * c[:, 0] / period)
    wrapped = np.asarray(grid.to_elements(jnp.asarray(smooth)))
    clamped = design.LayerGrid(plate, "design", pitch=0.025 * MM)
    clamped = np.asarray(clamped.to_elements(jnp.asarray(smooth)))
    assert np.abs(wrapped - exact).max() < (2 * np.pi * h / period) ** 2 / 8
    assert np.abs(clamped - exact).max() > 10 * np.abs(wrapped - exact).max()


def test_periodic_filter_and_map_gradient_matches_finite_differences(plate):
    grid = design.LayerGrid(plate, "design", pitch=0.05 * MM, periodic=(True, True))
    blur = design.helmholtz_filter(grid, radius=0.15 * MM)
    weights = jax.random.normal(jax.random.key(9), (grid.n_elements,))

    def objective(rho):
        density = grid.to_elements(design.project(blur(rho), beta=4.0))
        return jnp.sum(weights * density**2)

    rho = jax.random.uniform(jax.random.key(10), grid.shape)
    direction = jax.random.normal(jax.random.key(11), grid.shape)
    step = 1e-5
    fd = (objective(rho + step * direction) - objective(rho - step * direction)) / (
        2 * step
    )
    ad = jnp.vdot(jax.grad(objective)(rho), direction)
    np.testing.assert_allclose(ad, fd, rtol=1e-6)
    check_grads(blur, (rho,), order=2)
    check_grads(grid.to_elements, (rho,), order=2)
    np.testing.assert_allclose(jax.jit(blur)(rho), blur(rho), rtol=1e-12)
    batch = jax.random.uniform(jax.random.key(12), (3, *grid.shape))
    np.testing.assert_allclose(
        jax.vmap(blur)(batch), np.stack([blur(b) for b in batch]), rtol=1e-12
    )
    a, b = jax.random.uniform(jax.random.key(13), (2, *grid.shape))
    np.testing.assert_allclose(jnp.vdot(a, blur(b)), jnp.vdot(blur(a), b))


def test_periodic_grid_needs_whole_pixels_and_a_bool_per_axis(plate):
    with pytest.raises(ValueError, match=r"(?s)periodic along y.*Use pitch=0\.00025 m"):
        design.LayerGrid(plate, "design", pitch=0.3 * MM, periodic=(False, True))
    # along a Neumann axis the grid may overhang the footprint as before
    design.LayerGrid(plate, "design", pitch=0.3 * MM, periodic=(False, False))
    for bad in [True, (True,), ("x", "y"), (1, 0)]:
        with pytest.raises(ValueError, match=r"one bool per axis \(x, y\)"):
            design.LayerGrid(plate, "design", pitch=0.05 * MM, periodic=bad)
    grid = design.LayerGrid(plate, "design", pitch=0.05 * MM, periodic=(True, False))
    assert "periodic=(True, False)" in repr(grid)
