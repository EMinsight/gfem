"""The racetrack double spiral and its heater (tools/heater_geometry.py): no mesh."""

import heater_geometry as hg
import numpy as np
import pytest
from scipy.spatial import cKDTree

SPIRAL = hg.Spiral()


def _length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def test_the_route_length_has_a_closed_form():
    # 17 straights of 30 um (8 per side and the centre one), 7 bends of radii
    # 10..22 um at each end, and the centre's two 4.5 um semicircles:
    # 510 + 2 pi (10 + 12 + ... + 22) + 2 pi 4.5 = 510 + 233 pi um
    assert hg.route_length(SPIRAL) == pytest.approx(510 + 233 * np.pi, rel=1e-14)
    fine = hg.centreline(SPIRAL, max_step=0.01)
    assert _length(fine) == pytest.approx(510 + 233 * np.pi, rel=1e-7)
    other = hg.Spiral(inner_radius=6.0, straight=20.0, straights_per_side=4)
    # 9 x 20 + 2 pi (6 + 8 + 10) + 2 pi 2.5
    assert hg.route_length(other) == pytest.approx(180 + 53 * np.pi, rel=1e-14)
    assert _length(hg.centreline(other, max_step=0.01)) == pytest.approx(
        180 + 53 * np.pi, rel=1e-7
    )


def test_the_straights_sit_at_a_uniform_2_um_pitch():
    passes = hg.passes(SPIRAL)
    top = sorted(p.y for p in passes.values() if p.side == "top")
    bottom = sorted(p.y for p in passes.values() if p.side == "bottom")
    assert top == pytest.approx([10, 12, 14, 16, 18, 20, 22, 24])
    assert bottom == pytest.approx([-22, -20, -18, -16, -14, -12, -10, -8])
    (centre,) = (p for p in passes.values() if p.side == "centre")
    assert centre.y == pytest.approx(1.0)
    # each pass is a stretch of the centreline itself, 0 <= x <= 30 um
    line = hg.centreline(SPIRAL, max_step=0.1)
    tree = cKDTree(line)
    for p in passes.values():
        assert sorted([p.start[0], p.end[0]]) == pytest.approx([0.0, 30.0])
        assert tree.query([p.start, p.end])[0].max() < 1e-9
        assert [p.start[1], p.end[1]] == pytest.approx([p.y, p.y], abs=1e-12)


def test_the_core_never_comes_near_itself():
    # Points of the centreline closer than the pitch less a margin are
    # neighbours along the route, not two passes: no two cores overlap, and
    # the closest cores are 2 - 0.5 = 1.5 um apart.
    line = hg.centreline(SPIRAL, max_step=0.05)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    pairs = np.array(sorted(cKDTree(line).query_pairs(r=1.9)))
    assert np.abs(s[pairs[:, 0]] - s[pairs[:, 1]]).max() < 2.0
    polygon = hg.core_polygon(SPIRAL, max_step=0.05)
    # the core polygon is the centreline offset by +-0.25 um: its area is w L
    x, y = polygon.T
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(0.5 * hg.route_length(SPIRAL), rel=1e-4)


def test_three_passes_are_centred_under_each_heater_straight():
    passes = hg.passes(SPIRAL)
    half = SPIRAL.heater_width / 2
    centres = hg.heater_straights(SPIRAL)
    assert centres == pytest.approx((12.0, -10.0))
    for yc in centres:
        under = sorted(
            p.y
            for p in passes.values()
            # the whole core width under the heater
            if abs(p.y - yc) + SPIRAL.core_width / 2 <= half
        )
        assert under == pytest.approx([yc - 2, yc, yc + 2])
    heater = hg.heater_polygon(SPIRAL)
    # the straights run from the pads at x = -10 um past the 30 um straights
    assert heater[:, 0].min() == pytest.approx(-10.0)
    assert heater[:, 0].max() == pytest.approx(30 + 11 + half, rel=1e-6)
    for pad, yc in zip(hg.pad_polygons(SPIRAL), centres, strict=True):
        assert np.ptp(pad, axis=0) == pytest.approx([11.0, 11.0])
        assert pad[:, 0].max() == pytest.approx(-10.0)
        assert pad[:, 1].mean() == pytest.approx(yc)
