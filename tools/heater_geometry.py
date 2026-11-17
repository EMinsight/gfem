"""A racetrack double spiral of silicon waveguide under a TiN heater, in numpy.

The 3D heater tutorial's device (docs/tutorials/heater3d.ipynb, built by
tools/heater3d_run.py), in um in the wafer plane. One centreline gives both
the waveguide core's footprint (the centreline offset by half the core width)
and the routes the thermo-optic phase is integrated along, so the meshed core
and the route cannot disagree.

The spiral is two interleaved racetrack spirals joined at the centre. Its
straights run along x over 0 <= x <= ``straight``; every bend is a circular
half turn about one of two centres, (``straight``, 0) on the right and (0,
``pitch``) on the left, with radii ``inner_radius``, ``inner_radius +
pitch``, ... . The second spiral is the first turned by 180 degrees about
(``straight``/2, ``pitch``/2), so their straights interleave at a uniform
``pitch``: ``straights_per_side`` above the centre and as many below. At the
centre, two semicircles of radius (``inner_radius`` - ``pitch``/2)/2 and a
straight through the centre point join the two spirals. With constant
curvature per bend, the route's length has a closed form (`route_length`).

The heater is a U of TiN, ``heater_width`` wide: two straights, each centred
on a pass so that three passes lie under it, from x = ``heater_start`` to x
= ``straight``, joined by a semicircle on the right, and ending at the left
on square pads.
"""

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass(frozen=True)
class Spiral:
    """The spiral and heater, in um.

    Attributes:
        inner_radius: The radius of the spiral's tightest half turns (the
            centre's two semicircles are tighter).
        straight: The length of each straight.
        pitch: The distance between neighbouring passes, centre to centre.
        straights_per_side: The straights above the centre, and below it.
        core_width: The silicon core's width.
        core_thickness: The silicon core's thickness.
        heater_width: The width of the heater's straights and bend.
        heater_thickness: The heater's thickness.
        heater_z: The height of the heater's underside above the core's.
        heater_pass: Which pass each heater straight is centred on,
            counted from the centre (0 = innermost): 1 puts the innermost
            three passes of each side under the heater.
        heater_start: The x where the heater straights meet the pads.
        pad: The side of the square pads.
    """

    inner_radius: float = 10.0
    straight: float = 30.0
    pitch: float = 2.0
    straights_per_side: int = 8
    core_width: float = 0.5
    core_thickness: float = 0.22
    heater_width: float = 5.0
    heater_thickness: float = 0.1
    heater_z: float = 2.2
    heater_pass: int = 1
    heater_start: float = -10.0
    pad: float = 11.0

    @property
    def centre_radius(self):
        """The radius of the two semicircles at the centre."""
        return (self.inner_radius - self.pitch / 2) / 2


class Pass(typing.NamedTuple):
    """One straight of the spiral, in the direction of travel.

    Attributes:
        side: ``"top"`` (y above the centre), ``"bottom"`` or ``"centre"``.
        index: The pass's rank from the centre on its side (0 = innermost).
        y: Its centreline's y.
        start: Where the route enters it (x, y).
        end: Where the route leaves it (x, y).
    """

    side: str
    index: int
    y: float
    start: tuple[float, float]
    end: tuple[float, float]


def _pieces(spiral):
    """The route as ("line", length) and ("arc", radius, +-pi) moves, and its start.

    The route starts at the outer end of the lower spiral, (0, y0), heading
    +x; "arc" turns by the signed angle (+ counterclockwise).
    """
    n, s, r0, length = (
        spiral.straights_per_side,
        spiral.pitch,
        spiral.inner_radius,
        spiral.straight,
    )
    radii = [r0 + k * s for k in range(n - 1)]  # the tightest first
    moves = []
    for radius in radii[::-1]:  # inwards, counterclockwise
        moves += [("line", length), ("arc", radius, np.pi)]
    moves += [("line", length)]
    rc = spiral.centre_radius
    moves += [("arc", rc, np.pi), ("line", length), ("arc", rc, -np.pi)]
    for radius in radii:  # outwards, clockwise
        moves += [("line", length), ("arc", radius, -np.pi)]
    moves += [("line", length)]
    return moves, (0.0, -(r0 + (n - 2) * s))


def _walk(spiral, max_step):
    """Points along the centreline (N, 2), their headings (N,) and move labels.

    Straights contribute their ends only; arcs are sampled with chords at
    most ``max_step`` long. ``labels[i]`` is the index of the move that
    ends at point ``i`` (-1 for the first point).
    """
    moves, (x, y) = _pieces(spiral)
    heading = 0.0
    points, headings, labels = [(x, y)], [heading], [-1]
    for k, move in enumerate(moves):
        if move[0] == "line":
            x += move[1] * np.cos(heading)
            y += move[1] * np.sin(heading)
            points.append((x, y))
            headings.append(heading)
            labels.append(k)
            continue
        _, radius, sweep = move
        side = np.sign(sweep)
        cx = x - side * radius * np.sin(heading)
        cy = y + side * radius * np.cos(heading)
        steps = max(1, int(np.ceil(abs(sweep) * radius / max_step)))
        start = np.arctan2(y - cy, x - cx)
        for t in np.linspace(0, sweep, steps + 1)[1:]:
            points.append(
                (cx + radius * np.cos(start + t), cy + radius * np.sin(start + t))
            )
            headings.append(heading + t)
            labels.append(k)
        heading += sweep
        # snap the end of the half turn onto its exact value
        x, y = cx + radius * np.cos(start + sweep), cy + radius * np.sin(start + sweep)
        points[-1] = (x, y)
    return np.array(points), np.array(headings), np.array(labels)


def centreline(spiral, *, max_step=0.1):
    """The route's centreline, (N, 2) um, in the direction of travel.

    Straights are given by their ends; arcs by chords at most ``max_step``
    um long (a chord's sagitta is ``max_step**2 / (8 r)``).
    """
    return _walk(spiral, max_step)[0]


def route_length(spiral):
    """The centreline's length in um, in closed form.

    ``2n + 1`` straights, ``n - 1`` half turns of radii ``r0 + k s`` at each
    end, and the centre's two semicircles of radius ``rc``::

        (2n + 1) L + 2 pi sum_k (r0 + k s) + 2 pi rc
    """
    n, s, r0 = spiral.straights_per_side, spiral.pitch, spiral.inner_radius
    bends = (n - 1) * r0 + s * (n - 1) * (n - 2) / 2
    return float(
        (2 * n + 1) * spiral.straight
        + 2 * np.pi * bends
        + 2 * np.pi * spiral.centre_radius
    )


def passes(spiral):
    """Every straight of the spiral by name, in route order.

    Names: ``top_k`` and ``bottom_k`` (k = 0 innermost) and ``centre``.
    """
    points, _, labels = _walk(spiral, max_step=spiral.straight)
    moves, _ = _pieces(spiral)
    centre_y = spiral.pitch / 2
    found = []
    for k, move in enumerate(moves):
        if move[0] != "line":
            continue
        i = int(np.flatnonzero(labels == k)[0])
        start, end = tuple(points[i - 1]), tuple(points[i])
        found.append((start, end))
    ys = sorted({round(s[1], 9) for s, _ in found})
    top = [y for y in ys if y > centre_y + 1e-9]
    bottom = [y for y in ys if y < centre_y - 1e-9][::-1]
    out = {}
    for start, end in found:
        y = round(start[1], 9)
        if y in top:
            side, index = "top", top.index(y)
        elif y in bottom:
            side, index = "bottom", bottom.index(y)
        else:
            side, index = "centre", 0
        name = "centre" if side == "centre" else f"{side}_{index}"
        out[name] = Pass(side, index, float(y), start, end)
    return out


def heater_straights(spiral):
    """The y of the heater's two straights: centred on pass ``heater_pass``."""
    top = spiral.inner_radius + spiral.heater_pass * spiral.pitch
    return float(top), float(spiral.pitch - top)


def heater_polygon(spiral, *, max_step=0.25):
    """The heater U's outline, (M, 2) um, counterclockwise, not closed.

    Two straights from ``heater_start`` to ``straight`` and a semicircle on
    the right about (``straight``, centre y); arcs by chords of at most
    ``max_step`` um.
    """
    top, bottom = heater_straights(spiral)
    x0, x1, half = spiral.heater_start, spiral.straight, spiral.heater_width / 2
    yc, radius = (top + bottom) / 2, (top - bottom) / 2

    def arc(r, a0, a1):
        steps = max(1, int(np.ceil(abs(a1 - a0) * r / max_step)))
        a = np.linspace(a0, a1, steps + 1)
        return np.c_[x1 + r * np.cos(a), yc + r * np.sin(a)]

    return np.vstack(
        [
            [(x0, bottom - half)],
            arc(radius + half, -np.pi / 2, np.pi / 2),  # outer edge, upwards
            [(x0, top + half), (x0, top - half)],
            arc(radius - half, np.pi / 2, -np.pi / 2),  # inner edge, downwards
            [(x0, bottom + half)],
        ]
    )


def pad_polygons(spiral):
    """The two square pads, each (4, 2) um, left of the heater's straights."""
    x0, a = spiral.heater_start, spiral.pad
    return [
        np.array(
            [(x0 - a, y - a / 2), (x0, y - a / 2), (x0, y + a / 2), (x0 - a, y + a / 2)]
        )
        for y in heater_straights(spiral)
    ]


def core_polygon(spiral, *, max_step=0.5):
    """The waveguide core's footprint, (M, 2) um: the centreline offset by +-w/2.

    Each centreline point moves along its exact normal, so arcs stay
    concentric; chords on the arcs are at most ``max_step`` um long.
    """
    points, headings, _ = _walk(spiral, max_step)
    normal = np.c_[-np.sin(headings), np.cos(headings)] * spiral.core_width / 2
    left, right = points + normal, points - normal
    polygon = np.vstack([right, left[::-1]])
    keep = np.r_[True, np.linalg.norm(np.diff(polygon, axis=0), axis=1) > 1e-9]
    return polygon[keep]
