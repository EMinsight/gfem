"""Layer patterns: contours of a filtered density, user polygons, area fractions."""

import ast
import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem import _pattern, design

MM = 1e-3
#: The square plate's pixel side: 40 x 40 pixels over 1 mm x 1 mm.
PITCH = 0.025 * MM


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    """A 1 mm x 1 mm single-layer plate, 40 x 40 cells: one cell per pixel."""
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path_factory.mktemp("square") / "square.msh",
        sm.plate(1.0, 1.0, (("design", 0.1),), (40, 40, 1)),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def square2d(tmp_path_factory):
    """The z = 0 face of the square plate: a 2D mesh with an area "design"."""
    import structured_meshes as sm

    face = sm.z0_face(sm.plate(1.0, 1.0, (("design", 0.1),), (40, 40, 1)))
    path = sm.write_msh22(tmp_path_factory.mktemp("square2d") / "face.msh", face)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def grid(square):
    return design.LayerGrid(square, "design", pitch=PITCH)


def _xy(grid):
    return np.meshgrid(grid.x, grid.y)


def _signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _rings(pattern):
    for polygon in pattern:
        yield polygon.exterior
        yield from polygon.holes


def _disk(grid, radius, centre, width, eta=0.5):
    """A smooth field whose eta-isoline is the circle of radius ``radius``."""
    X, Y = _xy(grid)
    r = np.hypot(X - centre[0], Y - centre[1])
    edge = radius + width * np.arctanh(2 * eta - 1)  # the 0.5-isoline's radius
    return 0.5 + 0.5 * np.tanh((edge - r) / width)


# --- LayerGrid.pattern ----------------------------------------------------------------

CENTRE = (0.5013 * MM, 0.4979 * MM)  # off the pixel lattice: no symmetry


@pytest.mark.parametrize("eta", [0.5, 0.6])
def test_a_disk_contours_to_its_circle_within_a_twentieth_of_a_pixel(grid, eta):
    radius = 10 * PITCH
    rho = _disk(grid, radius, CENTRE, 2 * PITCH, eta)
    pattern = grid.pattern(rho, eta=eta)
    assert len(pattern) == 1
    (exterior, holes) = pattern.polygons[0]
    assert holes == ()
    error = np.hypot(exterior[:, 0] - CENTRE[0], exterior[:, 1] - CENTRE[1]) - radius
    assert np.abs(error).max() < 0.05 * PITCH
    assert abs(pattern.area - np.pi * radius**2) < 2 * np.pi * radius * 0.05 * PITCH
    perimeter = np.sum(np.hypot(*np.diff(exterior, axis=0).T))
    assert perimeter == pytest.approx(2 * np.pi * radius, rel=5e-3)


def test_a_bar_keeps_its_width_within_a_twentieth_of_a_pixel(grid):
    width, centre = 6.3 * PITCH, 0.4871 * MM
    X, _ = _xy(grid)
    rho = 0.5 + 0.5 * np.tanh((width / 2 - np.abs(X - centre)) / (2 * PITCH))
    (exterior, _), *rest = grid.pattern(rho)
    assert rest == []
    # a straight bar across the footprint: its long edges are single segments
    assert len(exterior) == 5
    assert abs(np.ptp(exterior[:, 0]) - width) < 0.05 * PITCH


def test_a_half_plane_closes_exactly_on_the_footprint(grid):
    x0 = 0.3017 * MM
    X, _ = _xy(grid)
    rho = 0.5 + 0.5 * np.tanh((X - x0) / (2 * PITCH))
    (exterior, holes), *rest = grid.pattern(rho)
    assert rest == []
    assert holes == ()
    _, ymin, xmax, ymax = grid.bounds
    assert exterior[:, 0].max() == xmax
    assert exterior[:, 1].min() == ymin
    assert exterior[:, 1].max() == ymax
    assert np.abs(exterior[:, 0] - x0).min() < 0.05 * PITCH
    # lossless clean-up leaves one vertex per corner of the box
    assert len(exterior) == 5
    assert _signed_area(exterior) == pytest.approx((xmax - x0) * (ymax - ymin), 1e-3)


def test_a_grid_overhanging_its_footprint_is_clipped_to_bounds(square):
    grid = design.LayerGrid(square, "design", pitch=0.03 * MM)  # 34 x 34, 1.02 mm
    xmin, ymin, xmax, ymax = grid.bounds
    assert grid.origin[0] < xmin
    pattern = grid.pattern(np.ones(grid.shape))
    (exterior, holes), *rest = pattern
    assert rest == []
    assert holes == ()
    np.testing.assert_array_equal(np.sort(np.unique(exterior[:, 0])), [xmin, xmax])
    np.testing.assert_array_equal(np.sort(np.unique(exterior[:, 1])), [ymin, ymax])
    assert pattern.bounds == grid.bounds


def test_a_periodic_bar_is_cut_at_the_seam_with_matching_vertices(square):
    grid = design.LayerGrid(square, "design", pitch=PITCH, periodic=(True, False))
    X, Y = _xy(grid)
    xmin, _, xmax, _ = grid.bounds
    across = np.minimum(np.abs(X - xmin), np.abs(X - xmax))  # distance to the seam
    rho = (0.5 + 0.5 * np.tanh((0.1 * MM - across) / (2 * PITCH))) * (
        0.5 + 0.5 * np.tanh((0.2 * MM - np.abs(Y - 0.5 * MM)) / (2 * PITCH))
    )
    pattern = grid.pattern(rho)
    assert pattern.periodic == (True, False)
    assert len(pattern) == 2
    left, right = sorted((p.exterior for p in pattern), key=lambda e: e[:, 0].min())
    on_left = np.sort(left[:-1][left[:-1, 0] == xmin, 1])
    on_right = np.sort(right[:-1][right[:-1, 0] == xmax, 1])
    assert len(on_left) == 2
    # one seam value on both lines; contourpy walks them in opposite directions,
    # so the interpolated vertices agree to round-off
    np.testing.assert_allclose(on_left, on_right, rtol=0, atol=1e-12 * PITCH)


@pytest.mark.parametrize(("off_diagonal", "count"), [(0.05, 2), (0.3, 1)])
def test_a_saddle_is_resolved_by_the_cell_centre_mean(grid, off_diagonal, count):
    # two solid pixels on a diagonal; the cell-centre mean (2 + 2v) / 4 against
    # eta = 0.6 decides whether they connect: 0.525 apart, 0.65 joined
    rho = np.zeros(grid.shape)
    rho[20, 20] = rho[21, 21] = 1.0
    rho[20, 21] = rho[21, 20] = off_diagonal
    assert len(grid.pattern(rho, eta=0.6)) == count


def test_rings_are_closed_outer_counterclockwise_holes_clockwise(grid):
    X, Y = _xy(grid)
    r = np.hypot(X - CENTRE[0], Y - CENTRE[1])
    rho = 0.5 + 0.5 * np.tanh((0.1 * MM - np.abs(r - 0.25 * MM)) / (2 * PITCH))
    pattern = grid.pattern(rho)
    assert len(pattern) == 1
    (exterior, holes) = pattern.polygons[0]
    assert len(holes) == 1
    assert _signed_area(exterior) > 0
    assert _signed_area(holes[0]) < 0
    for ring in _rings(pattern):
        assert ring.dtype == np.float64
        assert ring.shape[1] == 2
        np.testing.assert_array_equal(ring[0], ring[-1])
        assert np.all(np.any(np.diff(ring, axis=0) != 0, axis=1))  # no repeats


def test_empty_gives_no_polygon_and_full_gives_the_footprint(grid):
    assert len(grid.pattern(np.zeros(grid.shape))) == 0
    assert grid.pattern(np.zeros(grid.shape)).area == 0.0
    full = grid.pattern(np.ones(grid.shape))
    (exterior, holes), *rest = full
    assert rest == []
    assert holes == ()
    xmin, ymin, xmax, ymax = grid.bounds
    assert len(exterior) == 5
    assert {tuple(v) for v in exterior} == {
        (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    }  # fmt: skip
    assert full.area == pytest.approx((xmax - xmin) * (ymax - ymin), rel=1e-12)


def test_solid_includes_the_threshold(grid):
    assert len(grid.pattern(np.full(grid.shape, 0.5))) == 1


def test_the_pattern_records_its_grid(grid):
    pattern = grid.pattern(np.ones(grid.shape), eta=0.4)
    assert pattern.region == "design"
    assert pattern.eta == 0.4
    assert pattern.bounds == grid.bounds
    assert pattern.periodic == (False, False)


def test_volume_sheet_and_2d_grids_give_the_same_pattern(square, square2d):
    grids = [
        design.LayerGrid(square, "design", pitch=PITCH),
        design.LayerGrid(square, "bottom", pitch=PITCH),
        design.LayerGrid(square2d, "design", pitch=PITCH),
    ]
    rho = _disk(grids[0], 10 * PITCH, CENTRE, 2 * PITCH)
    patterns = [g.pattern(rho) for g in grids]
    for other in patterns[1:]:
        assert len(other) == len(patterns[0]) == 1
        np.testing.assert_allclose(
            other.polygons[0].exterior, patterns[0].polygons[0].exterior, atol=1e-15
        )


def test_pattern_refuses_bad_input(grid, square2d):
    rho = np.full(grid.shape, 0.7)
    with pytest.raises(ValueError, match=r"(?s)shape.*\(40, 40\)"):
        grid.pattern(rho[:-1])
    bad = rho.copy()
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        grid.pattern(bad)
    for eta in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match=r"eta must be in \(0, 1\)"):
            grid.pattern(rho, eta=eta)
    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.grad(lambda r: grid.pattern(r).area)(jnp.asarray(rho))
    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.jit(lambda r: grid.pattern(r).area)(jnp.asarray(rho))
    flat = design.LayerGrid(square2d, "design", pitch=PITCH)
    with pytest.raises(TypeError, match="pixels"):
        flat.pattern(flat.to_elements(jnp.asarray(rho)))


# --- LayerGrid.area_fraction ----------------------------------------------------------


def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_a_full_pattern_covers_every_pixel_exactly_overhang_included(square):
    grid = design.LayerGrid(square, "design", pitch=0.03 * MM)
    fraction = grid.area_fraction(grid.pattern(np.ones(grid.shape)))
    assert fraction.shape == grid.shape
    assert fraction.dtype == np.float64
    np.testing.assert_allclose(fraction, 1.0, rtol=0, atol=1e-12)


def test_a_half_plane_gives_the_analytic_edge_fractions(grid):
    x0 = 0.30173 * MM
    _, ymin, xmax, ymax = grid.bounds
    pattern = design.LayerPattern.from_polygons([_box(x0, ymin, xmax, ymax)])
    fraction = grid.area_fraction(pattern)
    left = grid.x - PITCH / 2
    expected = np.clip((left + PITCH - x0) / PITCH, 0, 1)
    np.testing.assert_allclose(
        fraction, np.broadcast_to(expected, grid.shape), atol=1e-12
    )


def test_a_disk_covers_its_area_exactly(grid):
    pattern = grid.pattern(_disk(grid, 10 * PITCH, CENTRE, 2 * PITCH))
    fraction = grid.area_fraction(pattern)
    assert fraction.min() >= 0.0
    assert fraction.max() <= 1.0
    assert fraction.sum() * PITCH**2 == pytest.approx(pattern.area, rel=1e-12)


def test_a_polygon_past_the_footprint_is_clipped(square):
    grid = design.LayerGrid(square, "design", pitch=0.03 * MM)  # overhangs 0.01 mm
    xmin, ymin, xmax, ymax = grid.bounds
    x0, y0 = 0.6123 * MM, 0.2071 * MM
    guide = design.LayerPattern.from_polygons([_box(x0, y0, xmax + MM, ymax + MM)])
    fraction = grid.area_fraction(guide)
    # each pixel's area inside the footprint, the denominator
    xe = np.clip(grid.origin[0] + np.arange(grid.shape[1] + 1) * grid.pitch, xmin, xmax)
    ye = np.clip(grid.origin[1] + np.arange(grid.shape[0] + 1) * grid.pitch, ymin, ymax)
    inside = np.outer(np.diff(ye), np.diff(xe))
    covered = np.sum(fraction * inside)
    assert covered == pytest.approx((xmax - x0) * (ymax - y0), rel=1e-12)
    assert fraction[-1, -1] == pytest.approx(1.0, abs=1e-12)  # overhanging corner pixel


def test_a_pattern_missing_the_footprint_raises_with_a_metres_hint(grid):
    far = design.LayerPattern.from_polygons(
        [_box(2.0, 2.0, 3.0, 3.0)]
    )  # in mm by mistake
    with pytest.raises(ValueError, match="metres"):
        grid.area_fraction(far)


def test_an_empty_pattern_gives_zeros(grid):
    fraction = grid.area_fraction(grid.pattern(np.zeros(grid.shape)))
    np.testing.assert_array_equal(fraction, np.zeros(grid.shape))


def test_a_periodic_pattern_covers_its_area(square):
    grid = design.LayerGrid(square, "design", pitch=PITCH, periodic=(True, True))
    X, _ = _xy(grid)
    xmin, _, xmax, _ = grid.bounds
    across = np.minimum(np.abs(X - xmin), np.abs(X - xmax))
    pattern = grid.pattern(0.5 + 0.5 * np.tanh((0.1 * MM - across) / (2 * PITCH)))
    assert len(pattern) == 2
    fraction = grid.area_fraction(pattern)
    assert fraction.sum() * PITCH**2 == pytest.approx(pattern.area, rel=1e-12)


def test_area_fraction_refuses_a_density_array(grid):
    with pytest.raises(TypeError, match="LayerPattern"):
        grid.area_fraction(np.ones(grid.shape))


def test_a_margin_band_pins_the_guides_area_fractions(grid):
    # the real neighbour, a feed guide, enters a band of pinned pixels
    xmin = grid.bounds[0]
    guide = _box(xmin - MM, 0.4537 * MM, xmin + 0.1 * MM, 0.5537 * MM)
    guides = design.LayerPattern.from_polygons([guide])
    fixed = grid.area_fraction(guides)
    in_margin = np.zeros(grid.shape, bool)
    in_margin[:, :4] = True  # 4 pixels = 0.1 mm
    free = jax.random.uniform(jax.random.key(0), grid.shape)
    rho = jnp.where(in_margin, fixed, free)
    np.testing.assert_array_equal(np.asarray(rho)[in_margin], fixed[in_margin])
    # rows crossed by the guide's long edges are partly covered
    row = np.clip(
        (np.minimum(grid.y + PITCH / 2, 0.5537 * MM)
         - np.maximum(grid.y - PITCH / 2, 0.4537 * MM)) / PITCH, 0, 1
    )  # fmt: skip
    np.testing.assert_allclose(fixed[:, :4], np.repeat(row[:, None], 4, 1), atol=1e-12)
    np.testing.assert_array_equal(fixed[:, 4:], 0.0)


# --- LayerPattern ---------------------------------------------------------------------


def _frame():
    """A 2 x 2 square with a 1 x 1 hole: solid area 3."""
    return design.Polygon(_box(0, 0, 2, 2), (_box(0.5, 0.5, 1.5, 1.5)[::-1],))


def test_area_is_the_outer_rings_minus_the_holes():
    pattern = design.LayerPattern.from_polygons([_frame(), _box(3, 0, 4, 1)])
    assert pattern.area == pytest.approx(4.0, rel=1e-15)


def test_a_pattern_is_a_sequence_of_polygons():
    pattern = design.LayerPattern.from_polygons([_frame(), _box(3, 0, 4, 1)])
    assert len(pattern) == 2
    polygons = list(pattern)
    assert all(isinstance(p, design.Polygon) for p in polygons)
    exterior, holes = polygons[0]
    assert exterior.shape == (5, 2)
    assert len(holes) == 1


def test_repr_counts_polygons_holes_and_vertices(grid):
    pattern = design.LayerPattern.from_polygons([_frame(), _box(3, 0, 4, 1)])
    text = repr(pattern)
    assert "2 polygons, 1 hole, 12 vertices" in text
    assert "None" not in text
    assert "eta" not in text
    contour = repr(grid.pattern(np.ones(grid.shape)))
    assert "'design'" in contour
    assert "eta=0.5" in contour
    assert "1 polygon, 0 holes, 4 vertices" in contour
    assert "100 %" in contour


def test_a_pattern_is_frozen():
    pattern = design.LayerPattern.from_polygons([_box(0, 0, 1, 1)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        pattern.eta = 0.3


def test_a_pattern_refuses_traced_input():
    def build(ring):
        return design.LayerPattern.from_polygons([ring]).area

    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.jit(build)(jnp.asarray(_box(0.0, 0.0, 1.0, 1.0)))

    def direct(ring):
        return design.LayerPattern((design.Polygon(ring, ()),), None).area

    with pytest.raises(ValueError, match="outside the transformed function"):
        jax.jit(direct)(jnp.asarray(_box(0.0, 0.0, 1.0, 1.0)))


# --- LayerPattern.from_polygons -------------------------------------------------------


def test_from_polygons_closes_open_rings():
    ((exterior, _),) = design.LayerPattern.from_polygons([_box(0, 0, 1, 1)])
    np.testing.assert_array_equal(exterior[0], exterior[-1])
    assert exterior.shape == (5, 2)
    assert exterior.dtype == np.float64


def test_from_polygons_orients_outer_rings_ccw_and_holes_cw():
    clockwise = _box(0, 0, 2, 2)[::-1]
    hole = _box(0.5, 0.5, 1.5, 1.5)  # counter-clockwise
    ((exterior, (inner,)),) = design.LayerPattern.from_polygons(
        [design.Polygon(clockwise, (hole,))]
    )
    assert _signed_area(exterior) == pytest.approx(4.0)
    assert _signed_area(inner) == pytest.approx(-1.0)


def test_from_polygons_drops_repeated_vertices():
    ring = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 0]], float)
    ((exterior, _),) = design.LayerPattern.from_polygons([ring])
    assert exterior.shape == (5, 2)


def test_from_polygons_keeps_bounds_and_periodic_and_no_region():
    pattern = design.LayerPattern.from_polygons(
        [_box(0, 0, 1, 1)], bounds=(0, 0, 2, 2), periodic=(True, False)
    )
    assert pattern.bounds == (0.0, 0.0, 2.0, 2.0)
    assert all(type(v) is float for v in pattern.bounds)
    assert pattern.periodic == (True, False)
    assert pattern.region is None
    assert pattern.eta is None
    assert design.LayerPattern.from_polygons([]).polygons == ()


@pytest.mark.parametrize(
    ("polygons", "kwargs", "match"),
    [
        ([np.array([[0, 0], [1, np.inf], [1, 1]])], {}, r"polygon 0, exterior.*finite"),
        ([_box(0, 0, 1, 1), np.array([[0, 0], [1, 0], [1, 0]])], {},
         r"polygon 1, exterior.*3"),
        ([np.array([[0, 0], [1, 1], [2, 2]])], {}, r"polygon 0, exterior.*area"),
        ([np.array([[0, 0], [2, 2], [2, 0], [0, 1]])], {},
         r"polygon 0, exterior.*(?:crosses|simple)"),
        ([design.Polygon(_box(0, 0, 2, 2), (_box(1.5, 0.5, 2.5, 1.5),))], {},
         r"polygon 0, hole 0.*polygon 0, exterior"),
        ([design.Polygon(_box(0, 0, 2, 2), (_box(3, 0.5, 4, 1.5),))], {},
         r"polygon 0, hole 0.*inside"),
        ([_box(0, 0, 2, 2), _box(1.5, 0.5, 2.5, 1.5)], {},
         r"polygon 1, exterior.*polygon 0, exterior"),
        ([_box(0, 0, 2, 2), _box(0.5, 0.5, 1, 1)], {},
         r"polygon 1.*solid of polygon 0"),
        ([_box(0, 0, 3, 1)], {"bounds": (0, 0, 2, 2)}, r"polygon 0, exterior.*bounds"),
        ([_box(0, 0, 1, 1)], {"periodic": (True, False)}, "bounds"),
        ([_box(-0.5, 0, 1, 1)], {"bounds": (0, 0, 2, 2), "periodic": (True, False)},
         r"polygon 0, exterior.*seam"),
    ],
)  # fmt: skip
def test_from_polygons_names_the_polygon_and_ring_at_fault(polygons, kwargs, match):
    with pytest.raises(ValueError, match=match):
        design.LayerPattern.from_polygons(polygons, **kwargs)


def test_a_polygon_inside_another_ones_hole_is_allowed():
    pattern = design.LayerPattern.from_polygons([_frame(), _box(0.8, 0.8, 1.2, 1.2)])
    assert pattern.area == pytest.approx(3.16)


def test_from_polygons_refuses_bad_types_and_shapes():
    with pytest.raises(TypeError, match="Polygon"):
        design.LayerPattern.from_polygons([None])
    with pytest.raises(TypeError, match="in a list"):
        design.LayerPattern.from_polygons(_box(0, 0, 1, 1))
    with pytest.raises(ValueError, match=r"polygon 0, exterior.*\(n, 2\)"):
        design.LayerPattern.from_polygons([[0, 0, 1, 1]])
    with pytest.raises(ValueError, match=r"polygon 0, exterior.*\(n, 2\)"):
        design.LayerPattern.from_polygons([np.zeros((4, 3))])
    with pytest.raises(ValueError, match="bounds"):
        design.LayerPattern.from_polygons([_box(0, 0, 1, 1)], bounds=(0, 0, -1, 1))


# --- module ---------------------------------------------------------------------------

NAMES = ("LayerPattern", "Polygon")


def test_the_new_names_live_in_gfem_design_only():
    for name in NAMES:
        assert name in design.__all__
        assert getattr(design, name) is getattr(_pattern, name)
        assert name not in gfem.__all__


def test_design_imports_no_private_module_at_its_bottom():
    tree = ast.parse(inspect.getsource(design))
    first = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    )  # fmt: skip
    late = [
        node for node in tree.body[first:]
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]  # fmt: skip
    assert late == []
