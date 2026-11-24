r"""Feature checks: a layer pattern's minimum width and minimum spacing.

The check measures pairs of polygon edges that face each other, the way a
foundry's width and space rules do. Two edges face each other when they run
opposite ways (:math:`\hat u_i \cdot \hat u_j < 0`) and the other edge lies on
the solid side of each (a width) or on the void side of each (a spacing).
Their distance is measured perpendicular to one edge, from an endpoint of the
other whose foot falls on it (the projection metric): the least of the four
endpoint projections. So two edges meeting at a corner of 90° or more never
pair, and an acute corner does. Measuring along the shortest straight line
(the Euclidean metric) would instead pair facets of one pixel-faceted curve
and fail a disk wider than the rule.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gfem._checks import is_traced

if TYPE_CHECKING:
    from gfem._pattern import LayerPattern

__all__ = ["PatternCheck", "Violation"]


class Violation(NamedTuple):
    """One place where a layer pattern is narrower than a rule.

    Too-narrow edge pairs whose markers' midpoints lie within the rule of one
    another are one place. Each marker runs from the foot on one edge to the
    endpoint of the other edge it was measured from, so ``ax.plot(*m.T)``
    draws marker ``m``.

    Attributes:
        kind: ``"width"`` (across solid) or ``"spacing"`` (across void).
        distance: The narrowest distance here (m).
        location: The midpoint of the narrowest marker, ``(2,)`` (m).
        markers: Every too-narrow edge pair here, ``(n, 2, 2)`` (m).
    """

    kind: str
    distance: float
    location: np.ndarray
    markers: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class PatternCheck:
    """A layer pattern measured against a minimum width and spacing.

    A report, not a verdict to raise on: `passed` is true when no violation
    was found; ``narrowest_width`` and ``narrowest_spacing`` give the margin
    even then.

    Attributes:
        region: The pattern's layer-grid region, ``None`` for user polygons.
        min_width: The minimum width checked (m), ``None`` when not checked.
        min_spacing: The minimum spacing checked (m), ``None`` when not
            checked.
        narrowest_width: The narrowest width measured anywhere (m):
            ``math.inf`` when no two edges face each other across solid,
            ``None`` when not checked.
        narrowest_spacing: The narrowest spacing, likewise.
        violations: Every place narrower than its rule, narrowest first.
    """

    region: str | None
    min_width: float | None
    min_spacing: float | None
    narrowest_width: float | None
    narrowest_spacing: float | None
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        """Whether the pattern keeps every rule checked."""
        return not self.violations

    def __repr__(self) -> str:
        parts = [] if self.region is None else [repr(self.region)]
        rules = []
        for kind, rule, narrowest in (
            ("width", self.min_width, self.narrowest_width),
            ("spacing", self.min_spacing, self.narrowest_spacing),
        ):
            if rule is None or narrowest is None:
                continue
            found = [v for v in self.violations if v.kind == kind]
            text = f"min_{kind} {_nm(rule)}: "
            if not found:
                text += "passed, " + (
                    f"narrowest {_nm(narrowest)}"
                    if math.isfinite(narrowest)
                    else "no facing edges"
                )
            else:
                x, y = found[0].location / _UM
                text += (
                    f"{len(found)} violation{'' if len(found) == 1 else 's'}, "
                    f"narrowest {_nm(narrowest)} at ({x:.4g}, {y:.4g}) um"
                )
            rules.append(text)
        parts.append("; ".join(rules))
        return f"PatternCheck({', '.join(parts)})"


_UM = 1e-6


def _nm(length: float) -> str:
    return f"{length / 1e-9:.4g} nm"


#: A pair closer than this times the rule is two edges sharing a vertex, not a
#: feature: adjacent edges of a ring meet at distance zero.
_SHARED = 1e-9
#: Edges whose ends lie this close (times the footprint's larger side) to a
#: footprint line lie on it: contoured seam vertices match only to round-off.
_ON_FOOTPRINT = 1e-9
#: Candidate edge pairs evaluated at a time (bounds the temporary arrays).
_BLOCK = 2**18


def check(
    pattern: LayerPattern, min_width: Any = None, min_spacing: Any = None
) -> PatternCheck:
    """The report of `LayerPattern.check`, which documents it."""
    where = "LayerPattern.check"
    if min_width is None and min_spacing is None:
        raise ValueError(
            f"{where}: give min_width, min_spacing or both (m); a check of no "
            "rule would always pass."
        )
    rules = {
        kind: _rule(value, f"min_{kind}", where)
        for kind, value in (("width", min_width), ("spacing", min_spacing))
        if value is not None
    }
    edges = _Edges(pattern)
    violations: list[Violation] = []
    narrowest: dict[str, float] = {}
    for kind, rule in rules.items():
        markers, distance = edges.pairs(kind, rule, _SHARED * rule)
        violations += edges.sites(kind, markers, distance, rule)
        narrowest[kind] = (
            float(distance.min()) if len(distance) else edges.narrowest(kind, rule)
        )
    violations.sort(key=lambda v: (v.distance, v.kind, *v.location))
    return PatternCheck(
        region=pattern.region,
        min_width=rules.get("width"),
        min_spacing=rules.get("spacing"),
        narrowest_width=narrowest.get("width"),
        narrowest_spacing=narrowest.get("spacing"),
        violations=tuple(violations),
    )


def _rule(value: Any, name: str, where: str) -> float:
    """A rule as a finite length > 0 (m), or raise."""
    if (
        is_traced(value)
        or isinstance(value, bool | np.bool_)
        or np.ndim(value) != 0
        or np.asarray(value).dtype.kind not in "iuf"
    ):
        raise TypeError(
            f"{where}: {name} is a length in metres, a real number, got {value!r}."
        )
    out = float(value)
    if not (math.isfinite(out) and out > 0):
        raise ValueError(f"{where}: {name} must be a finite length > 0 m, got {out}.")
    return out


class _Edges:
    """A pattern's layout edges, directed with the solid on their left.

    Outer rings run counter-clockwise and holes clockwise, so the solid lies on
    the left of every edge. Edges on the footprint rectangle are gfem's cut,
    not the layout's: what lies beyond is the user's layout (or the next
    period, across a seam), so they are never measured.
    """

    def __init__(self, pattern: LayerPattern) -> None:
        rings = [r for p in pattern.polygons for r in (p.exterior, *p.holes)]
        if rings:
            a = np.concatenate([r[:-1] for r in rings])
            b = np.concatenate([r[1:] for r in rings])
        else:
            a = b = np.zeros((0, 2))
        self.bounds = pattern.bounds
        self.periodic = pattern.periodic
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            tolerance = _ON_FOOTPRINT * max(xmax - xmin, ymax - ymin)
            on = np.zeros(len(a), dtype=bool)
            for axis, line in ((0, xmin), (0, xmax), (1, ymin), (1, ymax)):
                on |= (np.abs(a[:, axis] - line) <= tolerance) & (
                    np.abs(b[:, axis] - line) <= tolerance
                )
            a, b = a[~on], b[~on]
            extent = np.array([xmax - xmin, ymax - ymin])
        else:
            points = np.concatenate([a, b]) if len(a) else np.zeros((1, 2))
            extent = np.ptp(points, axis=0)
        self.a, self.b = a, b
        self.diagonal = float(np.hypot(*extent))

    def narrowest(self, kind: str, rule: float) -> float:
        """The least distance of any facing pair, ``inf`` if none.

        Searches with a radius doubling from twice the rule: the first search
        that finds a pair has found every closer one.
        """
        radius = 2 * rule
        while True:
            _, distance = self.pairs(kind, radius, _SHARED * rule)
            if len(distance):
                return float(distance.min())
            if radius > self.diagonal:
                return math.inf
            radius *= 2

    def pairs(
        self, kind: str, rule: float, floor: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The facing pairs closer than ``rule``: markers ``(n, 2, 2)``, distances.

        Edges longer than half the rule are cut into collinear pieces, so the
        midpoints of a pair closer than the rule lie within the rule plus the
        longest piece of each other.
        """
        from scipy.spatial import KDTree  # only needed here

        a, b = _pieces(self.a, self.b, rule / 2)
        n = len(a)
        if n == 0:
            return np.zeros((0, 2, 2)), np.zeros(0)
        length = np.linalg.norm(b - a, axis=1)
        reach = rule + float(length.max())
        a, b, piece, shift = self._tiled(a, b, reach)
        length = np.linalg.norm(b - a, axis=1)
        u = (b - a) / length[:, None]
        normal = np.stack([-u[:, 1], u[:, 0]], axis=1)  # towards the solid
        side = 1.0 if kind == "width" else -1.0
        candidates = KDTree(0.5 * (a + b)).query_pairs(reach, output_type="ndarray")
        markers, distances, found = [], [], []
        for start in range(0, len(candidates), _BLOCK):
            i, j = candidates[start : start + _BLOCK].T
            # Keep pairs with an edge of this period; then the edges must run
            # opposite ways (corners of 90 degrees or more never pair).
            keep = (i < n) | (j < n)
            i, j = i[keep], j[keep]
            keep = np.einsum("ij,ij->i", u[i], u[j]) < 0
            i, j = i[keep], j[keep]
            best = np.full(len(i), np.inf)
            marker = np.zeros((len(i), 2, 2))
            for e, f in ((i, j), (j, i)):  # the endpoints of f onto e
                for q in (a[f], b[f]):
                    t = np.einsum("ij,ij->i", q - a[e], u[e])
                    s = side * np.einsum("ij,ij->i", q - a[e], normal[e])
                    on = (t >= -floor) & (t <= length[e] + floor)
                    ok = on & (s > floor) & (s < best)
                    best = np.where(ok, s, best)
                    marker[ok, 0] = a[e][ok] + t[ok, None] * u[e][ok]
                    marker[ok, 1] = q[ok]
            hit = best < rule
            markers.append(marker[hit])
            distances.append(best[hit])
            found.append(np.stack([i[hit], j[hit]], axis=1))
        markers_ = np.concatenate(markers) if markers else np.zeros((0, 2, 2))
        distance = np.concatenate(distances) if distances else np.zeros(0)
        if any(self.periodic) and len(distance):
            pair = np.concatenate(found)
            once = _once(piece[pair[:, 0]], piece[pair[:, 1]], shift[pair[:, 0]],
                         shift[pair[:, 1]])  # fmt: skip
            markers_, distance = markers_[once], distance[once]
        return markers_, distance

    def sites(
        self, kind: str, markers: np.ndarray, distance: np.ndarray, rule: float
    ) -> list[Violation]:
        """Markers grouped into places: midpoints within the rule are one place.

        On a periodic axis the midpoints are compared modulo the period, so a
        place across the seam is one place whichever side its markers lie on.
        """
        if not len(distance):
            return []
        from scipy.sparse import coo_matrix  # only needed here
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import KDTree

        mid = markers.mean(axis=1)
        wrapped, box = mid.copy(), np.zeros(2)
        if self.bounds is not None:
            low = np.array(self.bounds[:2])
            period = np.array(self.bounds[2:]) - low
            for axis in np.flatnonzero(self.periodic):
                box[axis] = period[axis]
                w = np.mod(mid[:, axis] - low[axis], period[axis])
                wrapped[:, axis] = np.where(w < period[axis], w, 0.0)
        near = KDTree(wrapped, boxsize=box).query_pairs(rule, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(len(mid),) * 2
        )
        _, label = connected_components(graph, directed=False)
        out = []
        for site in range(label.max() + 1):
            members = np.flatnonzero(label == site)
            best = members[np.argmin(distance[members])]
            out.append(
                Violation(kind, float(distance[best]), mid[best], markers[members])
            )
        return out

    def _tiled(
        self, a: np.ndarray, b: np.ndarray, reach: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The pieces and, on periodic axes, their copies within reach.

        Returns the endpoints with the period's own pieces first, each piece's
        index and the period shift ``(n, 2)`` of its copy.
        """
        n = len(a)
        piece = np.arange(n)
        if not any(self.periodic) or self.bounds is None:
            return a, b, piece, np.zeros((n, 2), dtype=np.int64)
        xmin, ymin, xmax, ymax = self.bounds
        period = np.array([xmax - xmin, ymax - ymin])
        ranges = [
            range(-k, k + 1) if p else range(1)
            for p, k in zip(
                self.periodic, np.ceil(reach / period).astype(int), strict=True
            )
        ]
        shifts = sorted(itertools.product(*ranges), key=lambda s: s != (0, 0))
        low, high = np.array([xmin, ymin]) - reach, np.array([xmax, ymax]) + reach
        tiles = []
        for s in shifts:
            offset = np.asarray(s) * period
            mid = 0.5 * (a + b) + offset
            # Only copies within reach of the footprint can pair with its edges.
            near = np.all((mid >= low) & (mid <= high), axis=1) | (s == (0, 0))
            tiles.append((a[near] + offset, b[near] + offset, piece[near], s))
        return (
            np.concatenate([t[0] for t in tiles]),
            np.concatenate([t[1] for t in tiles]),
            np.concatenate([t[2] for t in tiles]),
            np.concatenate([np.tile(t[3], (len(t[2]), 1)) for t in tiles]),
        )


def _pieces(
    a: np.ndarray, b: np.ndarray, longest: float
) -> tuple[np.ndarray, np.ndarray]:
    """Edges cut into equal collinear pieces of at most ``longest``."""
    count = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / longest)).astype(
        np.int64
    )
    edge = np.repeat(np.arange(len(a)), count)
    k = np.arange(len(edge)) - np.repeat(np.cumsum(count) - count, count)
    step = (b - a)[edge] / count[edge, None]
    start = a[edge] + k[:, None] * step
    return start, start + step


def _once(pi: np.ndarray, pj: np.ndarray, si: np.ndarray, sj: np.ndarray) -> np.ndarray:
    """One index per pair found from both sides of a periodic seam.

    A pair of a piece and another's copy is the same pair as the copy's
    piece and the first piece's opposite copy: key each by its pieces and
    their relative shift, in a canonical order. Of the two, keep the one
    whose first piece is in this period, so that the markers of one place
    all lie by the same piece.
    """
    d = sj - si
    swap = (pi > pj) | ((pi == pj) & ((d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0))))
    first, second = np.where(swap, pj, pi), np.where(swap, pi, pj)
    d = np.where(swap[:, None], -d, d)
    copied = np.any(np.where(swap[:, None], sj, si) != 0, axis=1)
    order = np.lexsort((copied, d[:, 1], d[:, 0], second, first))
    key = np.stack([first, second, d[:, 0], d[:, 1]], axis=1)[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = np.any(key[1:] != key[:-1], axis=1)
    return np.sort(order[new])
