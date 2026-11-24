"""Feature checks: minimum width and spacing of a layer pattern."""

from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem import _feature_check, design

UM = 1e-6
#: The footprint is 2 um x 2 um in 100 x 100 pixels.
PITCH = 0.02 * UM
#: The rule of the known shapes.
RULE = 0.25 * UM


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """A 2 um x 2 um single-layer plate, one cell per pixel."""
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path_factory.mktemp("plate") / "plate.msh",
        sm.plate(2.0, 2.0, (("design", 0.1),), (100, 100, 1)),
    )
    return gfem.Mesh.from_gmsh(path, length_unit="um")


@pytest.fixture(scope="module")
def grid(plate):
    return design.LayerGrid(plate, "design", pitch=PITCH)


# Shapes as signed distances (um, negative inside) about the footprint's centre
# (1, 1) um; the grid contours 0.5 - sdf, so the pattern is the sdf's zero set.


def _box(x, y, centre, half):
    """A rectangle of centre ``(cx, cy)`` and half sides ``(hx, hy)``."""
    (cx, cy), (hx, hy) = centre, half
    qx, qy = np.abs(x - cx) - hx, np.abs(y - cy) - hy
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0)


def _disk(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def _pattern(grid, sdf):
    """The pattern of ``sdf <= 0``, coordinates in um about the centre."""
    X, Y = np.meshgrid(grid.x / UM - 1, grid.y / UM - 1)
    return grid.pattern(0.5 - sdf(X, Y))


def _dumbbell(x, y):
    bells = np.minimum(_disk(x, y, -0.5, 0, 0.35), _disk(x, y, 0.5, 0, 0.35))
    return np.minimum(bells, _box(x, y, (0, 0), (0.5, 0.075)))


# --- LayerPattern.check ---------------------------------------------------------------


def _two_bars(x, y):
    return np.minimum(
        _box(x, y, (-0.375, 0), (0.3, 0.4)), _box(x, y, (0.375, 0), (0.3, 0.4))
    )


def _annulus(x, y):
    return np.maximum(_disk(x, y, 0, 0, 0.5), -_disk(x, y, 0, 0, 0.3))


def _notch(x, y):
    return np.maximum(
        _box(x, y, (0, 0), (0.6, 0.5)), -_box(x, y, (0, 0.3), (0.075, 0.4))
    )


#: Narrow shapes: (signed distance, kind, true distance (um), KLayout's
#: projection-metric distance on the same polygons (um, 1e-4 um database unit)).
NARROW = {
    "bar 0.20": (lambda x, y: _box(x, y, (0, 0), (0.6, 0.1)), "width", 0.20, 0.2000),
    "disk d=0.20": (lambda x, y: _disk(x, y, 0, 0, 0.1), "width", 0.20, 0.1982),
    "dumbbell bridge 0.15": (_dumbbell, "width", 0.15, 0.1500),
    "two bars, 0.15 apart": (_two_bars, "spacing", 0.15, 0.1500),
    "annulus wall 0.20": (_annulus, "width", 0.20, 0.1999),
    "notch 0.15": (_notch, "spacing", 0.15, 0.1500),
}


@pytest.mark.parametrize("name", NARROW)
def test_a_narrow_shape_is_one_violation_at_its_true_distance(grid, name):
    sdf, kind, true, klayout = NARROW[name]
    check = _pattern(grid, sdf).check(min_width=RULE, min_spacing=RULE)
    (violation,) = check.violations
    assert violation.kind == kind
    assert getattr(check, f"narrowest_{kind}") == violation.distance
    # A faceted circle reads short by its facets' sagitta.
    tolerance = (0.1 if name.startswith("disk") else 0.05) * PITCH
    assert abs(violation.distance - true * UM) < tolerance
    # KLayout's distances are whole database units on snapped vertices.
    assert abs(violation.distance - klayout * UM) < 1.5e-4 * UM


#: Shapes wider than the rule everywhere: (signed distance, narrowest width (um)).
WIDE = {
    # 1.2 rules across: a curve's facets never pair (they meet at under 90 degrees)
    "disk d=0.30": (lambda x, y: _disk(x, y, 0, 0, 0.15), 0.30),
    # 90 degree corners are never measured
    "box 0.6": (lambda x, y: _box(x, y, (0, 0), (0.3, 0.3)), 0.6),
    "L-shape": (
        lambda x, y: np.minimum(
            _box(x, y, (0, -0.3), (0.6, 0.15)), _box(x, y, (-0.45, 0.15), (0.15, 0.6))
        ),
        0.3,
    ),
}


@pytest.mark.parametrize("name", WIDE)
def test_curves_and_right_angles_pass(grid, name):
    sdf, width = WIDE[name]
    check = _pattern(grid, sdf).check(min_width=RULE, min_spacing=RULE)
    assert check.passed
    assert check.violations == ()
    assert abs(check.narrowest_width - width * UM) < 0.05 * PITCH


# --- footprint and seams --------------------------------------------------------------


def test_a_stub_at_the_footprint_edge_passes(grid):
    # 0.10 deep where a guide enters: the footprint edge is not a layout edge
    check = _pattern(grid, lambda x, y: _box(x, y, (-0.96, 0), (0.06, 0.3))).check(
        min_width=RULE
    )
    assert check.passed
    assert check.narrowest_width > RULE


def test_a_feature_entering_the_footprint_is_measured_across_only(grid):
    pattern = _pattern(grid, lambda x, y: _box(x, y, (-0.5, 0), (0.5, 0.15)))
    # Where the field equals the threshold on the footprint edge, contourpy
    # gives the bar's corners twice, one rounding apart: both corners stay.
    assert pattern.area == pytest.approx(0.3 * UM**2, rel=1e-3, abs=0)
    check = pattern.check(min_width=RULE)
    assert check.passed
    assert abs(check.narrowest_width - 0.30 * UM) < 0.05 * PITCH


def test_all_solid_and_empty_have_nothing_to_measure(grid):
    for rho in (np.ones(grid.shape), np.zeros(grid.shape)):
        check = grid.pattern(rho).check(min_width=RULE, min_spacing=RULE)
        assert check.passed
        assert check.narrowest_width == np.inf
        assert check.narrowest_spacing == np.inf


@pytest.fixture(scope="module")
def periodic_grid(plate):
    return design.LayerGrid(plate, "design", pitch=PITCH, periodic=(True, False))


def _across_the_seam(x, y, half_width):
    """A bar about x = -1 um (the seam), cut in two by it."""
    return np.minimum(
        _box(x, y, (-1, 0), (half_width, 0.5)), _box(x, y, (1, 0), (half_width, 0.5))
    )


def test_a_bar_cut_by_the_seam_is_measured_once(grid, periodic_grid):
    def cut(x, y):
        return _across_the_seam(x, y, 0.1)

    check = _pattern(periodic_grid, cut).check(min_width=RULE)
    (violation,) = check.violations
    assert abs(violation.distance - 0.20 * UM) < 0.05 * PITCH
    # as many edge pairs as the same bar whole, in the middle of the footprint
    whole = _pattern(grid, lambda x, y: _box(x, y, (0, 0), (0.1, 0.5))).check(
        min_width=RULE
    )
    assert len(violation.markers) == len(whole.violations[0].markers)
    # without the period each half has only the seam edge facing it
    assert _pattern(grid, cut).check(min_width=RULE).passed


def test_a_void_across_the_seam_is_one_violation(grid, periodic_grid):
    # solid from x = -0.925 to -0.075 and 0.075 to 0.925 um: two 0.15 um voids,
    # one in the middle and one across the seam
    def bars(x, y):
        return np.minimum(
            _box(x, y, (-0.5, 0), (0.425, 0.5)), _box(x, y, (0.5, 0), (0.425, 0.5))
        )

    check = _pattern(periodic_grid, bars).check(min_spacing=RULE)
    assert [v.kind for v in check.violations] == ["spacing", "spacing"]
    for violation in check.violations:
        assert abs(violation.distance - 0.15 * UM) < 0.05 * PITCH
    assert len(_pattern(grid, bars).check(min_spacing=RULE).violations) == 1


def test_a_doubly_periodic_square_is_one_spacing_violation(plate):
    grid = design.LayerGrid(plate, "design", pitch=PITCH, periodic=(True, True))
    # a 1.85 um square: 0.15 um of void around it, across both seams
    check = _pattern(grid, lambda x, y: _box(x, y, (0, 0), (0.925, 0.925))).check(
        min_spacing=RULE
    )
    (violation,) = check.violations
    assert violation.kind == "spacing"
    assert abs(violation.distance - 0.15 * UM) < 0.05 * PITCH


def test_a_sliver_against_a_margin_band_is_measured(grid):
    # The footprint's left edge: a 4 pixel (0.08 um) solid sliver along it is
    # gfem's cut, not measured...
    sliver = np.zeros(grid.shape)
    sliver[:, :4] = 1.0
    assert grid.pattern(sliver).check(min_width=RULE).passed
    # ...but against a margin band of pinned pixels (here the cladding, void,
    # with a feed guide entering) it is a layout edge, and measured.
    xmin, ymin = grid.bounds[:2]
    guide = design.LayerPattern.from_polygons(
        [_rectangle(xmin - UM, ymin + 0.8 * UM, xmin + 0.3 * UM, ymin + 1.2 * UM)]
    )
    in_margin = np.zeros(grid.shape, bool)
    in_margin[:, :15] = True  # 0.3 um, more than the rule
    free = np.zeros(grid.shape)
    free[:, 15:19] = 1.0
    rho = np.where(in_margin, grid.area_fraction(guide), free)
    check = grid.pattern(rho).check(min_width=RULE)
    assert {v.kind for v in check.violations} == {"width"}
    assert abs(check.narrowest_width - 4 * PITCH) < 0.05 * PITCH
    assert all(v.location[0] > xmin + 0.3 * UM for v in check.violations)


def _rectangle(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# --- the bend tutorial's arc ----------------------------------------------------------

BEND = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "bend2d.msh"
#: The arc's drawn width, the filter radius and the grid pitch.
ARC, RADIUS, BEND_PITCH = 0.55 * UM, 0.3 * UM, 0.05 * UM


def _line_width(eta):
    r"""The printed width of a straight line ARC wide through the filter at eta.

    In units of the filter radius, the 1D Helmholtz filter's response to a
    line of width :math:`h` is :math:`1 - e^{-\sqrt3 h}\cosh(2\sqrt3 x)`
    inside it, :math:`\lvert x \rvert < h/2`.
    """
    h = ARC / RADIUS
    root3 = np.sqrt(3)
    x = np.arccosh((1 - eta) * np.exp(root3 * h)) / (2 * root3)
    assert x < h / 2  # the threshold edge lies inside the drawn line
    return 2 * x * RADIUS


@pytest.fixture(scope="module")
def bend():
    """The bend's design grid and its Helmholtz-filtered drawn arc."""
    mesh = gfem.Mesh.from_gmsh(BEND, length_unit="um")
    grid = design.LayerGrid(mesh, "design", pitch=BEND_PITCH)
    angle = np.linspace(0, np.pi / 2, 401)
    centre = np.array([-1.5, -1.5]) * UM
    outer = centre + (1.5 * UM + ARC / 2) * np.stack([np.cos(angle), np.sin(angle)], 1)
    inner = centre + (1.5 * UM - ARC / 2) * np.stack([np.cos(angle), np.sin(angle)], 1)
    arc = design.LayerPattern.from_polygons([np.concatenate([outer, inner[::-1]])])
    blur = design.helmholtz_filter(grid, radius=RADIUS)
    return grid, np.asarray(blur(grid.area_fraction(arc)))


def test_the_bend_arc_prints_the_straight_line_width(bend):
    grid, filtered = bend
    check = grid.pattern(filtered, eta=0.5).check(min_width=0.5 * UM)
    assert check.passed
    assert abs(check.narrowest_width - _line_width(0.5)) < 0.1 * BEND_PITCH


def test_the_eroded_bend_arc_is_one_violation(bend):
    grid, filtered = bend
    check = grid.pattern(filtered, eta=0.7).check(min_width=0.5 * UM)
    (violation,) = check.violations
    assert violation.kind == "width"
    assert abs(violation.distance - _line_width(0.7)) < 0.1 * BEND_PITCH


# --- the report -----------------------------------------------------------------------


def _report():
    """Bars 0.10 and 0.18 um wide, the second 0.12 um from a third: three places."""
    bars = [
        _rectangle(0, 0, 0.10 * UM, 2 * UM),
        _rectangle(1 * UM, 0, 1.18 * UM, 2 * UM),
        _rectangle(1.30 * UM, 0, 2 * UM, 2 * UM),
    ]
    return design.LayerPattern.from_polygons(bars).check(
        min_width=0.2 * UM, min_spacing=0.15 * UM
    )


def test_violations_come_narrowest_first():
    check = _report()
    assert not check.passed
    assert [(v.kind, round(v.distance / UM, 9)) for v in check.violations] == [
        ("width", 0.10),
        ("spacing", 0.12),
        ("width", 0.18),
    ]
    assert check.narrowest_width == check.violations[0].distance
    assert check.narrowest_spacing == check.violations[1].distance


def test_a_violation_unpacks_into_its_markers():
    kind, _, location, markers = _report().violations[1]
    assert kind == "spacing"
    assert location.shape == (2,)
    assert markers.ndim == 3
    assert markers.shape[1:] == (2, 2)
    # each marker runs across the void between the facing edges
    np.testing.assert_allclose(
        np.sort(markers[:, :, 0], axis=1), [[1.18 * UM, 1.30 * UM]] * len(markers)
    )
    assert location[0] == pytest.approx(1.24 * UM, rel=1e-12)


def test_a_passing_check_reports_its_margin():
    check = design.LayerPattern.from_polygons([_rectangle(0, 0, UM, UM)]).check(
        min_width=0.2 * UM
    )
    assert check.passed
    assert check.violations == ()
    assert check.narrowest_width == pytest.approx(1 * UM, rel=1e-12)
    assert check.min_spacing is None
    assert check.narrowest_spacing is None
    assert check.region is None


def test_a_check_is_a_frozen_report():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        _report().min_width = 1.0


def test_repr_gives_each_rule_its_count_and_narrowest(grid):
    # (along parallel edges every marker is narrowest: y is any of theirs)
    text = repr(_report())
    assert text.startswith(
        "PatternCheck(min_width 200 nm: 2 violations, narrowest 100 nm at (0.05, "
    )
    assert "um; min_spacing 150 nm: 1 violation, narrowest 120 nm at (1.24, " in text
    check = _pattern(grid, _dumbbell).check(min_spacing=RULE)
    assert repr(check).startswith(
        "PatternCheck('design', min_spacing 250 nm: passed, narrowest "
    )
    assert "min_width" not in repr(check)
    empty = grid.pattern(np.zeros(grid.shape)).check(min_width=RULE)
    assert "passed, no facing edges" in repr(empty)


@pytest.mark.parametrize(
    ("rules", "error", "match"),
    [
        ({}, ValueError, "give min_width, min_spacing or both"),
        ({"min_width": 0.0}, ValueError, "min_width must be a finite length > 0"),
        ({"min_spacing": -1e-7}, ValueError, "min_spacing must be a finite length > 0"),
        ({"min_width": np.inf}, ValueError, "finite"),
        ({"min_width": np.nan}, ValueError, "finite"),
        ({"min_width": True}, TypeError, "min_width is a length in metres"),
        ({"min_spacing": "150 nm"}, TypeError, "min_spacing is a length in metres"),
        ({"min_width": [1e-7]}, TypeError, "a real number"),
        ({"min_width": 1e-7 + 0j}, TypeError, "a real number"),
    ],
)
def test_check_refuses_bad_rules(rules, error, match):
    pattern = design.LayerPattern.from_polygons([_rectangle(0, 0, UM, UM)])
    with pytest.raises(error, match=match):
        pattern.check(**rules)


def test_check_refuses_positional_rules():
    pattern = design.LayerPattern.from_polygons([_rectangle(0, 0, UM, UM)])
    with pytest.raises(TypeError):
        pattern.check(1e-7)


def test_the_report_names_live_in_gfem_design():
    for name in ("PatternCheck", "Violation"):
        assert name in design.__all__
        assert name not in gfem.__all__
        assert getattr(design, name) is getattr(_feature_check, name)
