r"""Layer patterns: the polygons of a layer-grid design, and their area fractions.

A layer pattern is the part of a layer grid's footprint where the filtered
density reaches the projection threshold :math:`\eta`: the
:math:`\beta \to \infty` limit of ``project(filtered, beta, eta)``, the drawn
design. `LayerGrid.pattern` draws it as polygons with sub-pixel edges,
`LayerPattern.from_polygons` builds one from user polygons (feed guides,
hand-drawn designs), and `LayerGrid.area_fraction` rasterizes one exactly back
onto the pixels.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gfem._checks import is_traced, metres_hint

if TYPE_CHECKING:
    from gfem._feature_check import PatternCheck
    from gfem.design import LayerGrid

__all__ = ["LayerPattern", "Polygon"]


class Polygon(NamedTuple):
    """One polygon of a layer pattern: an outer ring and its holes.

    Rings are closed (the first vertex repeated at the end, so
    ``ax.plot(*ring.T)`` draws one whole), float64, in metres in the mesh's
    xy frame. The outer ring runs counter-clockwise and the holes clockwise
    (the right-hand rule), so the signed areas of all rings add up to the
    solid area.

    Attributes:
        exterior: The outer ring, ``(n, 2)``.
        holes: The holes, each ``(m, 2)``.
    """

    exterior: np.ndarray
    holes: tuple[np.ndarray, ...] = ()


#: Why a pattern cannot be built inside a JAX transform.
_TRACED = (
    "the polygons' vertex count depends on the values, so a pattern is never "
    "traced; compute the pattern outside the transformed function, from the "
    "concrete filtered density"
)


@dataclasses.dataclass(frozen=True, eq=False)
class LayerPattern:
    """The polygons where a layer is solid.

    Three routes build one: `gfem.design.LayerGrid.pattern` contours a
    filtered density at the projection threshold; `from_polygons` takes user
    polygons (feed guides, hand-drawn designs) and validates them; and
    `gfem.design.FabricationModel.pattern` predicts the printed pattern of a
    drawn design. The constructor itself does not validate.
    A sequence of `Polygon`: ``len(pattern)``, ``for exterior, holes in
    pattern``. Not a JAX pytree: its vertex count depends on the values.

    Attributes:
        polygons: The polygons, disjoint.
        bounds: The footprint rectangle the pattern closes on,
            ``(xmin, ymin, xmax, ymax)`` (m), or ``None`` (user polygons
            with no footprint).
        periodic: Per axis ``(x, y)``, whether the pattern is one period of a
            periodic cell, cut at the seam (the edges of `bounds`).
        region: The layer grid's region, ``None`` for user polygons.
        eta: The threshold the pattern was contoured at, ``None`` for user
            polygons.
    """

    polygons: tuple[Polygon, ...]
    bounds: tuple[float, float, float, float] | None
    periodic: tuple[bool, bool] = (False, False)
    region: str | None = None
    eta: float | None = None

    def __post_init__(self) -> None:
        if is_traced(self.polygons) or is_traced(self.bounds):
            raise ValueError(f"LayerPattern: {_TRACED}.")

    @classmethod
    def from_polygons(
        cls,
        polygons: Iterable[Polygon | Any],
        *,
        bounds: tuple[float, float, float, float] | None = None,
        periodic: tuple[bool, bool] = (False, False),
    ) -> LayerPattern:
        """A layer pattern from user polygons, normalized and validated.

        Each ring is closed if open, repeated vertices are dropped, outer
        rings are turned counter-clockwise and holes clockwise. Nothing else
        moves: no vertex is snapped, merged or simplified.

        Args:
            polygons: Each a `Polygon` ``(exterior, holes)`` or a bare
                ``(n, 2)`` array (a polygon with no hole), in metres.
            bounds: The footprint rectangle ``(xmin, ymin, xmax, ymax)`` (m)
                every vertex must lie in; ``None`` for none.
            periodic: Per axis ``(x, y)``, whether the pattern is one period
                of a periodic cell (needs ``bounds``, whose edges are the
                seam). Polygons must come cut at the seam.

        Returns:
            The pattern, with ``region`` and ``eta`` ``None``.

        Raises:
            TypeError: ``polygons`` is a single ring (wrap it in a list), or
                an item is neither a `Polygon` nor an array.
            ValueError: Naming the polygon and ring: a non-finite coordinate,
                fewer than 3 distinct vertices, no enclosed area, a ring
                crossing itself or another ring, a hole outside its outer
                ring, a polygon inside another one's solid, a vertex outside
                ``bounds`` or past a periodic seam. Also bad ``bounds`` or
                ``periodic``, a periodic axis without ``bounds``, or values
                computed inside `jax.grad`, `jax.jit` or `jax.vmap`.
        """
        if is_traced(polygons) or is_traced(bounds):
            raise ValueError(f"LayerPattern.from_polygons: {_TRACED}.")
        if getattr(polygons, "ndim", None) == 2:  # one ring, NumPy or JAX
            raise TypeError(
                "LayerPattern.from_polygons takes a sequence of polygons; wrap a "
                "single (n, 2) ring in a list: from_polygons([ring])."
            )
        periodic = _check_periodic(periodic)
        box = None if bounds is None else _check_bounds(bounds)
        if any(periodic) and box is None:
            raise ValueError(
                "LayerPattern.from_polygons: a periodic pattern needs bounds, the "
                "rectangle whose edges are the seam: bounds=(xmin, ymin, xmax, ymax) m."
            )
        out = []
        for k, item in enumerate(polygons):
            if isinstance(item, Polygon):
                exterior, holes = item
            elif item is None or isinstance(item, str | bytes | dict):
                raise TypeError(_not_a_polygon(k, item))
            else:
                exterior, holes = item, ()
            rings = [_user_ring(exterior, k, "exterior", +1)]
            rings += [_user_ring(h, k, f"hole {i}", -1) for i, h in enumerate(holes)]
            if box is not None:
                for ring, name in zip(rings, _names(k, len(rings)), strict=True):
                    _check_within(ring, name, box, periodic)
            out.append(Polygon(rings[0], tuple(rings[1:])))
        polygons_ = tuple(out)
        _check_crossings(polygons_)
        _check_nesting(polygons_)
        return cls(polygons_, box, periodic)

    def check(
        self, *, min_width: float | None = None, min_spacing: float | None = None
    ) -> PatternCheck:
        r"""Measure the pattern against a minimum width and a minimum spacing.

        Finds the places where solid (a width) or void (a spacing) is
        narrower than the rule, by the facing edges of the polygons, as a
        foundry's width and space rules do: two edges that run opposite ways
        with solid between them (or void), at a distance measured
        perpendicular to one of them. Two edges meeting at a corner of 90° or
        more are never measured against each other; an acute corner is. A
        distance strictly below the rule is a violation, and too-narrow edge
        pairs within one rule of each other are one violation. A notch is a
        spacing. The narrowest width and spacing are reported even when the
        check passes, as the margin::

            check = pattern.check(min_width=150e-9, min_spacing=150e-9)
            print(check)
            if check.passed:
                gfem.design.write_gds("design.gds", {(1, 0): pattern})

        The check is exact on the polygons, which follow the
        :math:`\beta \to \infty` design's edge to a small fraction of the
        pitch. Edges on the footprint rectangle are not measured: what lies
        beyond is the rest of the layout, so a sliver along the footprint
        facing void outside is left to the assembled layout's checks (pin a
        margin band of the real neighbours inside the grid to measure it).
        On a periodic axis features are measured across the seam. To check a
        robust design, check the eroded pattern's width and the dilated
        pattern's spacing. A foundry's checks may measure differently (along
        the shortest line on Manhattan layouts), and GDS snapping moves a
        distance by up to :math:`\sqrt2` database units: this is a pre-check,
        not foundry sign-off.

        Args:
            min_width: The minimum width (m), > 0, or ``None`` to skip it.
            min_spacing: The minimum spacing (m), > 0, or ``None`` to skip it.

        Returns:
            A `PatternCheck`: the narrowest width and spacing, the violations
            (narrowest first, each with markers to plot) and ``passed``.

        Raises:
            TypeError: A rule is not a real number (``bool`` included).
            ValueError: Neither rule given, or a rule not finite or not > 0.
        """
        # imported here: the check is a module of its own, loaded when used
        from gfem._feature_check import check

        return check(self, min_width=min_width, min_spacing=min_spacing)

    @property
    def area(self) -> float:
        """The solid area in m², the outer rings' areas minus the holes'."""
        return float(sum(_signed_area(r) for r in _rings(self.polygons)))

    def __len__(self) -> int:
        return len(self.polygons)

    def __iter__(self) -> Iterator[Polygon]:
        return iter(self.polygons)

    def __repr__(self) -> str:
        n = len(self.polygons)
        holes = sum(len(p.holes) for p in self.polygons)
        vertices = sum(len(r) - 1 for r in _rings(self.polygons))
        parts = [] if self.region is None else [repr(self.region)]
        parts.append(
            f"{n} polygon{'' if n == 1 else 's'}, {holes} "
            f"hole{'' if holes == 1 else 's'}, {vertices} vertices"
        )
        if self.bounds is None:
            parts.append(f"area {self.area:.4g} m^2")
        else:
            xmin, ymin, xmax, ymax = self.bounds
            footprint = (xmax - xmin) * (ymax - ymin)
            parts.append(
                f"solid {100 * self.area / footprint:.3g} % of {footprint:.4g} m^2"
            )
        if any(self.periodic):
            parts.append(f"periodic={self.periodic}")
        if self.eta is not None:
            parts.append(f"eta={self.eta:g}")
        return f"LayerPattern({', '.join(parts)})"


# --- contouring -----------------------------------------------------------------------

#: Collinear vertices dropped by the clean-up: |cross product| at most this times
#: pitch², far below any edge shift (contourpy keeps one vertex per pixel centre along
#: a straight footprint edge).
_COLLINEAR = 1e-9


def contour(
    grid: LayerGrid,
    rho: Any,
    eta: Any,
    sample: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> LayerPattern:
    """The pattern of a filtered pixel density at threshold ``eta``.

    The sample nodes run from footprint edge to footprint edge: the pixel
    centres strictly inside it plus its two edges per axis, valued through
    the grid's own bilinear map (``sample`` gives its indices and weights),
    so solid touching the footprint closes exactly along it and a grid
    overhanging the footprint is clipped to it. On a periodic axis both seam
    lines take one value, so a feature cut there ends on matching vertices.

    Args:
        grid: The layer grid.
        rho: The filtered pixel density, ``grid.shape``.
        eta: The threshold, in (0, 1); solid is ``rho >= eta``.
        sample: Points ``(n, 2)`` to the bilinear map's flat pixel indices
            and weights, each ``(n, 4)``.
    """
    name = "LayerGrid.pattern"
    if is_traced(rho) or is_traced(eta):
        raise ValueError(f"{name}: {_TRACED}.")
    rho = np.asarray(grid._pixels(rho, name, real=True), dtype=np.float64)  # noqa: SLF001
    if not np.all(np.isfinite(rho)):
        raise ValueError(
            f"{name}: rho holds {int(np.sum(~np.isfinite(rho)))} NaN or infinite "
            "values; contour the finite filtered density."
        )
    eta = float(eta)
    if not 0 < eta < 1:
        raise ValueError(f"{name}: the threshold eta must be in (0, 1), got {eta}.")

    from contourpy import FillType, contour_generator  # light, only needed here

    xmin, ymin, xmax, ymax = grid.bounds
    xs = _nodes(grid.x, xmin, xmax)
    ys = _nodes(grid.y, ymin, ymax)
    X, Y = np.meshgrid(xs, ys)
    index, weight = sample(np.stack([X.ravel(), Y.ravel()], axis=1))
    z = np.sum(rho.ravel()[index] * weight, axis=1).reshape(X.shape)
    # Both seam lines are one line: give them bitwise one value.
    if grid.periodic[0]:
        z[:, -1] = z[:, 0]
    if grid.periodic[1]:
        z[-1, :] = z[0, :]
    generator = contour_generator(
        xs, ys, z, name="serial", fill_type=FillType.OuterOffset
    )
    # contourpy fills z > lower; one ulp below eta makes solid exactly z >= eta.
    points, offsets = generator.filled(np.nextafter(eta, -np.inf), np.inf)
    tolerance = _COLLINEAR * grid.pitch**2
    polygons = []
    for p, o in zip(points, offsets, strict=True):
        vertices = np.asarray(p, dtype=np.float64)
        starts = np.asarray(o, dtype=np.int64)
        rings = [_clean(vertices[a:b], tolerance) for a, b in pairwise(starts)]
        if rings[0] is None:
            continue
        holes = tuple(r for r in rings[1:] if r is not None)
        polygons.append(Polygon(rings[0], holes))
    return LayerPattern(tuple(polygons), grid.bounds, grid.periodic, grid.region, eta)


def _nodes(centres: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The pixel centres strictly inside ``(lo, hi)``, and the two edges."""
    inner = centres[(centres > lo) & (centres < hi)]
    return np.concatenate([[lo], inner, [hi]])


def _clean(ring: np.ndarray, tolerance: float) -> np.ndarray | None:
    """A closed ring without repeated or collinear vertices; ``None`` if degenerate."""
    ring = drop_repeats(np.asarray(ring, dtype=np.float64))
    while len(ring) >= 3:
        previous, following = np.roll(ring, 1, axis=0), np.roll(ring, -1, axis=0)
        a, b = ring - previous, following - ring
        flat = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) <= tolerance
        if not flat.any():
            return np.concatenate([ring, ring[:1]])
        if flat.all():
            return None
        # Drop every other vertex of each run of flat ones, then look again:
        # two copies of a corner one rounding apart both read flat, and
        # dropping both at once would cut the corner off.
        start = int(np.argmin(flat))  # a vertex that stays
        run = np.roll(flat, -start)
        index = np.arange(len(run))
        position = index - np.maximum.accumulate(np.where(run, 0, index))
        ring = ring[~np.roll(run & (position % 2 == 1), start)]
    return None


# --- area fraction --------------------------------------------------------------------


def area_fraction(grid: LayerGrid, pattern: LayerPattern) -> np.ndarray:
    r"""Each pixel's share of its in-footprint area covered by the pattern.

    The exact area fraction: the rings' edges are split at the pixel lines
    (clipped to the footprint) and each piece adds its signed trapezoid, the
    area between it and the pixel's lower edge, to its pixel and a full column
    height to the pixels below it (Green's theorem, area :math:`-\oint y\,dx` per
    column).
    """
    if not isinstance(pattern, LayerPattern):
        raise TypeError(
            "LayerGrid.area_fraction takes a LayerPattern (from grid.pattern or "
            f"LayerPattern.from_polygons), got {type(pattern).__name__}."
        )
    ny, nx = grid.shape
    xmin, ymin, xmax, ymax = grid.bounds
    xe = np.clip(grid.origin[0] + np.arange(nx + 1) * grid.pitch, xmin, xmax)
    ye = np.clip(grid.origin[1] + np.arange(ny + 1) * grid.pitch, ymin, ymax)
    rings = list(_rings(pattern.polygons))
    if not rings:
        return np.zeros(grid.shape)
    p0 = np.concatenate([r[:-1] for r in rings])
    p1 = np.concatenate([r[1:] for r in rings])
    start, end = _split_at_lines(p0, p1, xe, ye)
    dx = end[:, 0] - start[:, 0]
    xm = 0.5 * (start[:, 0] + end[:, 0])
    ym = 0.5 * (start[:, 1] + end[:, 1])
    use = (dx != 0) & (xm > xmin) & (xm < xmax) & (ym > ymin)
    dx, xm, ym = dx[use], xm[use], ym[use]
    column = np.searchsorted(xe, xm, side="right") - 1
    row = np.minimum(np.searchsorted(ye, ym, side="right") - 1, ny)  # ny: above
    within = row < ny
    partial = np.bincount(
        row[within] * nx + column[within],
        -(ym[within] - ye[row[within]]) * dx[within],
        minlength=ny * nx,
    ).reshape(ny, nx)
    # Each piece also covers its whole column below its own row.
    full = np.bincount(row * nx + column, -dx, minlength=(ny + 1) * nx)
    below = np.cumsum(full.reshape(ny + 1, nx)[::-1], axis=0)[::-1][1:]
    height, width = np.diff(ye), np.diff(xe)
    covered = partial + below * height[:, None]
    fraction = np.clip(covered / np.outer(height, width), 0.0, 1.0)
    if not fraction.any():
        lo = p0.min(axis=0)
        hi = p0.max(axis=0)
        raise ValueError(
            f"LayerGrid.area_fraction: no polygon of the pattern overlaps the "
            f"footprint of {grid.region!r}, x in [{xmin:g}, {xmax:g}] m, y in "
            f"[{ymin:g}, {ymax:g}] m; the pattern spans x in [{lo[0]:g}, "
            f"{hi[0]:g}], y in [{lo[1]:g}, {hi[1]:g}].{metres_hint(np.abs(p0).max())}"
        )
    return fraction


def _split_at_lines(
    p0: np.ndarray, p1: np.ndarray, xe: np.ndarray, ye: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The segments ``p0 -> p1`` split where they cross the lines ``xe``, ``ye``."""
    n = len(p0)
    segment = [np.arange(n), np.arange(n)]
    t = [np.zeros(n), np.ones(n)]
    for axis, lines in ((0, xe), (1, ye)):
        a, b = p0[:, axis], p1[:, axis]
        first = np.searchsorted(lines, np.minimum(a, b), side="right")
        count = np.maximum(
            np.searchsorted(lines, np.maximum(a, b), side="left") - first, 0
        )
        s = np.repeat(np.arange(n), count)
        k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        segment.append(s)
        t.append((lines[first[s] + k] - a[s]) / (b - a)[s])
    s, t_ = np.concatenate(segment), np.concatenate(t)
    order = np.lexsort((t_, s))
    s, t_ = s[order], t_[order]
    points = p0[s] + t_[:, None] * (p1 - p0)[s]
    same = s[:-1] == s[1:]
    return points[:-1][same], points[1:][same]


# --- user polygons --------------------------------------------------------------------

#: A ring encloses no area when its |signed area| is at most this times its
#: bounding box's squared diagonal (round-off of collinear vertices).
_FLAT = 1e-12
#: Vertices may lie this far (relative to the bounds' larger side) outside bounds.
_BOUNDS_SLACK = 1e-12


def _not_a_polygon(k: int, item: Any) -> str:
    return (
        f"LayerPattern.from_polygons: polygon {k} is a {type(item).__name__}; each "
        "polygon is a Polygon(exterior, holes) or an (n, 2) array of vertices (m)."
    )


def _user_ring(ring: Any, k: int, name: str, sign: int) -> np.ndarray:
    """A user ring closed, without repeats, oriented by ``sign``, or raise."""
    where = f"LayerPattern.from_polygons: polygon {k}, {name}"
    try:
        array = np.array(ring, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(_not_a_polygon(k, ring)) from None
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{where}: a ring is an (n, 2) array of (x, y) vertices (m), got "
            f"shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{where}: coordinates must be finite.")
    open_ = drop_repeats(array)
    if len(open_) < 3:
        raise ValueError(
            f"{where}: a ring needs at least 3 distinct vertices, got {len(open_)}."
        )
    closed = np.concatenate([open_, open_[:1]])
    area = _signed_area(closed)
    if abs(area) <= _FLAT * np.sum(np.ptp(open_, axis=0) ** 2):
        raise ValueError(f"{where}: the ring encloses no area (collinear vertices).")
    return closed if area * sign > 0 else closed[::-1].copy()


def _check_bounds(bounds: Any) -> tuple[float, float, float, float]:
    try:
        box = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError):
        box = np.full(1, np.nan)
    if (
        box.shape != (4,)
        or not np.all(np.isfinite(box))
        or not (box[0] < box[2] and box[1] < box[3])
    ):
        raise ValueError(
            "LayerPattern.from_polygons: bounds is the footprint rectangle "
            "(xmin, ymin, xmax, ymax) in m, with xmin < xmax and ymin < ymax; "
            f"got {bounds!r}."
        )
    xmin, ymin, xmax, ymax = (float(v) for v in box)
    return (xmin, ymin, xmax, ymax)


def _check_periodic(periodic: Any) -> tuple[bool, bool]:
    try:
        ok = len(periodic) == 2 and all(
            isinstance(p, bool | np.bool_) for p in periodic
        )
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(
            "LayerPattern.from_polygons: periodic takes one bool per axis (x, y), "
            f"e.g. periodic=(True, False), got {periodic!r}."
        )
    return (bool(periodic[0]), bool(periodic[1]))


def _check_within(
    ring: np.ndarray,
    name: str,
    box: tuple[float, float, float, float],
    periodic: tuple[bool, bool],
) -> None:
    slack = _BOUNDS_SLACK * max(box[2] - box[0], box[3] - box[1])
    for axis, label in enumerate("xy"):
        lo, hi = box[axis], box[axis + 2]
        out = (ring[:, axis] < lo - slack) | (ring[:, axis] > hi + slack)
        if not out.any():
            continue
        x, y = ring[np.argmax(out)]
        where = f"LayerPattern.from_polygons: {name}"
        if periodic[axis]:
            raise ValueError(
                f"{where}: the vertex ({x:g}, {y:g}) m lies past the periodic seam "
                f"{label} = {lo:g} / {hi:g} m. The pattern holds one period: cut "
                "the polygon at the seam into its pieces on each side."
            )
        raise ValueError(
            f"{where}: the vertex ({x:g}, {y:g}) m lies outside the bounds "
            f"{label} in [{lo:g}, {hi:g}] m.{metres_hint(max(abs(x), abs(y)))}"
        )


def _names(k: int, n_rings: int) -> list[str]:
    return [f"polygon {k}, exterior"] + [
        f"polygon {k}, hole {i}" for i in range(n_rings - 1)
    ]


def _check_crossings(polygons: tuple[Polygon, ...]) -> None:
    """Raise if a ring crosses itself or two rings cross.

    Within one ring any contact between non-adjacent edges breaks simplicity;
    between rings only a proper crossing is refused (touching at a vertex, or
    two guides abutting along an edge, is allowed). Candidate pairs: edges
    whose midpoints lie within the longer edge's length (a k-d tree query from
    each edge with its own length finds every pair from its longer edge).
    """
    from scipy.spatial import KDTree  # only needed here

    rings, names = [], []
    for k, polygon in enumerate(polygons):
        rings += [polygon.exterior, *polygon.holes]
        names += _names(k, 1 + len(polygon.holes))
    if not rings:
        return
    a = np.concatenate([r[:-1] for r in rings])
    b = np.concatenate([r[1:] for r in rings])
    sizes = np.array([len(r) - 1 for r in rings])
    ring = np.repeat(np.arange(len(rings)), sizes)
    position = np.arange(len(a)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    length = np.linalg.norm(b - a, axis=1)
    near = KDTree(0.5 * (a + b)).query_ball_point(0.5 * (a + b), r=length)
    i = np.repeat(np.arange(len(a)), [len(n) for n in near])
    j = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    i, j = np.minimum(i, j), np.maximum(i, j)
    pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    i, j = pairs[:, 0], pairs[:, 1]
    same = ring[i] == ring[j]
    gap = np.abs(position[i] - position[j])
    adjacent = same & ((gap == 1) | (gap == sizes[ring[i]] - 1))
    i, j, same = (
        i[~adjacent & (i != j)],
        j[~adjacent & (i != j)],
        same[~adjacent & (i != j)],
    )

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                       - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))  # fmt: skip

    o1, o2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
    o3, o4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)

    def on(p: np.ndarray, q: np.ndarray, r: np.ndarray, o: np.ndarray) -> np.ndarray:
        """``r`` collinear with and within the box of segment ``p -> q``."""
        lo, hi = np.minimum(p, q), np.maximum(p, q)
        return (o == 0) & np.all((r >= lo) & (r <= hi), axis=1)

    touch = (
        on(a[i], b[i], a[j], o1) | on(a[i], b[i], b[j], o2)
        | on(a[j], b[j], a[i], o3) | on(a[j], b[j], b[i], o4)
    )  # fmt: skip
    bad = np.flatnonzero(proper | (same & touch))
    if not len(bad):
        return
    m = bad[0]
    first, second = int(ring[i[m]]), int(ring[j[m]])
    x, y = 0.5 * (a[i[m]] + b[i[m]])
    if first == second:
        raise ValueError(
            f"LayerPattern.from_polygons: {names[first]} is not simple: it crosses "
            f"itself near ({x:g}, {y:g}) m."
        )
    raise ValueError(
        f"LayerPattern.from_polygons: {names[second]} crosses {names[first]} near "
        f"({x:g}, {y:g}) m; rings must not cross."
    )


def _check_nesting(polygons: tuple[Polygon, ...]) -> None:
    """Raise for a hole outside its outer ring or a polygon in another's solid.

    Rings do not cross (checked first), so a ring lies wholly on one side of
    another up to shared boundary points; a majority vote of a few of its
    vertices settles which, whatever the boundary points report.
    """
    for k, (exterior, holes) in enumerate(polygons):
        for h, hole in enumerate(holes):
            if not _mostly(_inside(_probes(hole), [exterior])):
                raise ValueError(
                    f"LayerPattern.from_polygons: polygon {k}, hole {h} is not "
                    f"inside polygon {k}, exterior; a hole lies within its outer ring."
                )
    if len(polygons) < 2:
        return
    lo = np.array([p.exterior.min(axis=0) for p in polygons])
    hi = np.array([p.exterior.max(axis=0) for p in polygons])
    # Only a polygon whose box lies in another's box can lie in its solid.
    boxed = np.all(lo[:, None] >= lo[None, :], axis=2) & np.all(
        hi[:, None] <= hi[None, :], axis=2
    )
    np.fill_diagonal(boxed, False)
    for other in np.flatnonzero(boxed.any(axis=0)):
        inner = np.flatnonzero(boxed[:, other])
        probes = [_probes(polygons[k].exterior) for k in inner]
        # even-odd over the exterior and the holes together: inside the solid
        solid = _inside(np.concatenate(probes), list(_rings([polygons[other]])))
        ends = np.cumsum([len(p) for p in probes])
        for k, flags in zip(inner, np.split(solid, ends[:-1]), strict=True):
            if _mostly(flags):
                raise ValueError(
                    f"LayerPattern.from_polygons: polygon {k} lies inside the solid "
                    f"of polygon {other}; polygons are disjoint (an island goes in a "
                    "hole, or merge the two)."
                )


#: Vertices of a ring voting on which side of another ring it lies.
_PROBES = 5


def _probes(ring: np.ndarray) -> np.ndarray:
    """Up to `_PROBES` vertices spread along a closed ring."""
    n = len(ring) - 1
    return ring[np.unique(np.linspace(0, n - 1, min(n, _PROBES)).astype(np.int64))]


def _mostly(flags: np.ndarray) -> bool:
    return 2 * int(np.sum(flags)) > len(flags)


#: Point-edge pairs per block of the point-in-polygon test (memory bound).
_POINT_EDGE_BLOCK = 2**22


def _inside(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Whether each point is inside the closed rings (even-odd rule)."""
    a = np.concatenate([r[:-1] for r in rings])
    b = np.concatenate([r[1:] for r in rings])
    dy = np.where(b[:, 1] == a[:, 1], 1.0, b[:, 1] - a[:, 1])
    out = np.empty(len(points), dtype=bool)
    step = max(1, _POINT_EDGE_BLOCK // len(a))
    for s in range(0, len(points), step):
        px, py = points[s : s + step, :1], points[s : s + step, 1:]
        crosses = (a[:, 1] > py) != (b[:, 1] > py)
        x = a[:, 0] + (py - a[:, 1]) * (b[:, 0] - a[:, 0]) / dy
        out[s : s + step] = np.sum(crosses & (px < x), axis=1) % 2 == 1
    return out


# --- shared ---------------------------------------------------------------------------


def _rings(polygons: Iterable[Polygon]) -> Iterator[np.ndarray]:
    for polygon in polygons:
        yield polygon.exterior
        yield from polygon.holes


def _signed_area(ring: np.ndarray) -> float:
    """The shoelace area of a closed ring, positive counter-clockwise."""
    return 0.5 * float(twice_area(ring[:-1]))


def twice_area(ring: np.ndarray) -> Any:
    """Twice the signed area of an open ring, in the ring's own arithmetic.

    The shoelace sum, positive counter-clockwise: exact on Python integers
    (an object array), rounded on floats.
    """
    x, y = ring[:, 0], ring[:, 1]
    return np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def cut_edges(
    a: np.ndarray, b: np.ndarray, longest: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges ``a -> b`` cut into equal collinear pieces of at most ``longest``.

    Returns the pieces' starts and ends and each piece's edge index. Two
    pieces that touch have midpoints within ``longest`` of each other, so a
    k-d tree of the midpoints finds every contact among them, however long
    the edges.
    """
    count = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / longest)).astype(
        np.int64
    )
    edge = np.repeat(np.arange(len(a)), count)
    k = np.arange(len(edge)) - np.repeat(np.cumsum(count) - count, count)
    step = (b - a)[edge] / count[edge, None]
    start = a[edge] + k[:, None] * step
    return start, start + step, edge


def drop_repeats(ring: np.ndarray) -> np.ndarray:
    """An open ring without consecutive repeats (the last against the first too).

    Of consecutive repeats the cyclically last copy stays, so a closed ring
    loses its closing repeat and keeps its first vertex; a ring of one point
    repeated keeps one.
    """
    keep = np.any(ring != np.roll(ring, -1, axis=0), axis=1)
    return ring[keep] if keep.any() else ring[:1]
