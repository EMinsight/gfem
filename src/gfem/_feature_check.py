r"""Feature checks: a layer pattern's minimum width and minimum spacing.

The check measures pairs of polygon edges that face each other, the way a
foundry's width and space rules do. Two edges face each other when they run
opposite ways (:math:`\hat u_i \cdot \hat u_j < 0`) and the other edge lies on
the solid side of each (a width) or on the void side of each (a spacing).
Their distance is measured perpendicular to one edge, from an endpoint of the
other whose foot falls on it (the projection metric): the least of the four
endpoint projections, on the whole edges, so it does not depend on the rule:
a rule fails exactly when it exceeds the narrowest distance. So two edges
meeting at a corner of 90° or more never pair, and an acute corner does.
Measuring along the shortest straight line
(the Euclidean metric) would instead pair facets of one pixel-faceted curve
and fail a disk wider than the rule.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gfem._checks import positive_length
from gfem._pattern import cut_edges

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
        location: The midpoint of the narrowest edge pair's shortest
            marker, ``(2,)`` (m).
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


#: Two edges sharing a vertex meet at distance zero: a distance at most this
#: times the pattern's extent is such a vertex, not a feature.
_SHARED = 1e-9
#: Edges whose ends lie this close (times the footprint's larger side) to a
#: footprint line lie on it: contoured seam vertices match only to round-off.
_ON_FOOTPRINT = 1e-9
#: How to differentiate a rule (it cannot be: a check is not differentiable).
_FIXED_RULE = "A rule is fixed: give it as a number."
#: Candidate edge pairs evaluated at a time (bounds the temporary arrays).
_PAIR_BLOCK = 2**18


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
        kind: positive_length(value, f"min_{kind}", where, _FIXED_RULE)
        for kind, value in (("width", min_width), ("spacing", min_spacing))
        if value is not None
    }
    edges = _Edges(pattern)
    violations: list[Violation] = []
    narrowest: dict[str, float] = {}
    for rule in sorted(set(rules.values())):
        kinds = tuple(k for k, r in rules.items() if r == rule)
        least, involved, _ = edges.distances(kinds, rule)
        for kind in kinds:
            if least[kind] < rule:  # then the least of all pairs
                narrowest[kind] = least[kind]
                markers, distance, local = edges.markers(kind, rule, involved[kind])
                violations += edges.sites(kind, markers, distance, local, rule)
    unknown = tuple(k for k in rules if k not in narrowest)
    if unknown:
        narrowest |= edges.narrowest(unknown, min(rules[k] for k in unknown))
    violations.sort(key=lambda v: (v.distance, v.kind, *v.location))
    return PatternCheck(
        region=pattern.region,
        min_width=rules.get("width"),
        min_spacing=rules.get("spacing"),
        narrowest_width=narrowest.get("width"),
        narrowest_spacing=narrowest.get("spacing"),
        violations=tuple(violations),
    )


#: Edge pairs as ``(i, j, shift_i, shift_j)``: both edges' indices and the
#: period shifts ``(n, 2)`` of their copies.
_Pairs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class _Edges:
    """A pattern's polygon edges, directed with the solid on their left.

    Outer rings run counter-clockwise and holes clockwise, so the solid lies on
    the left of every edge. Edges on the footprint rectangle are gfem's cut,
    not the pattern's: what lies beyond is the user's layout (or the next
    period, across a seam), so they are never measured.

    A pair's distance is measured on the two edges themselves, never on
    pieces of them, so it does not depend on the rule: a rule above the
    narrowest distance always finds it, and one below never does. Edges are
    cut into pieces to find candidate pairs, and to draw a violation's
    markers along it.
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
        self.period = np.zeros(2)
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
            self.period = extent
        else:
            points = np.concatenate([a, b]) if len(a) else np.zeros((1, 2))
            extent = np.ptp(points, axis=0)
        self.a, self.b = a, b
        length = np.linalg.norm(b - a, axis=1)
        self.u = (b - a) / length[:, None]
        # Candidate pieces need not be shorter than a typical edge: the search
        # then stays local whatever the rule, a tiny one included.
        self.typical = float(np.median(length)) if len(a) else 0.0
        self.diagonal = float(np.hypot(*extent))
        self.floor = _SHARED * self.diagonal

    def narrowest(self, kinds: tuple[str, ...], rule: float) -> dict[str, float]:
        """The least distance of any facing pair per kind, ``inf`` if none.

        Searches with a radius doubling from twice the rule: the first search
        that finds a pair closer than its radius has found every closer one,
        and a search that saw every pair has found the least.
        """
        out: dict[str, float] = {}
        radius = 2 * rule
        while True:
            left = tuple(k for k in kinds if k not in out)
            least, _, complete = self.distances(left, radius)
            for kind in left:
                if complete or least[kind] < radius:
                    out[kind] = least[kind]
            if len(out) == len(kinds):
                return out
            if radius > self.diagonal:
                return out | {k: math.inf for k in kinds if k not in out}
            radius *= 2

    def distances(
        self, kinds: tuple[str, ...], rule: float
    ) -> tuple[dict[str, float], dict[str, np.ndarray], bool]:
        """The least distance of the facing pairs seen, per kind.

        Every pair closer than ``rule`` is seen; whether every pair was seen
        is the third value. Candidates are pieces whose midpoints lie within
        the rule plus the longest piece: an endpoint and its foot closer than
        the rule lie in such pieces. Also returns, per kind, the edges of the
        pairs closer than the rule.
        """
        from scipy.spatial import KDTree  # only needed here

        least = dict.fromkeys(kinds, math.inf)
        close: dict[str, list[np.ndarray]] = {k: [np.zeros(0, np.int64)] for k in kinds}
        a, b, edge = cut_edges(self.a, self.b, max(rule / 2, self.typical))
        n = len(a)
        complete = True
        if n:
            reach = rule + float(np.linalg.norm(b - a, axis=1).max())
            mid, piece, shift = self._tiled(0.5 * (a + b), reach)
            candidates = KDTree(mid).query_pairs(reach, output_type="ndarray")
            complete = not any(self.periodic) and len(candidates) == n * (n - 1) // 2
            for start in range(0, len(candidates), _PAIR_BLOCK):
                block = candidates[start : start + _PAIR_BLOCK]
                i, j, si, sj = self._facing(block, n, edge[piece], shift)
                found = _project(kinds, self.floor, *self._ends(i, j, si, sj))
                for kind in kinds:
                    d = found[kind][0]
                    if len(d):
                        least[kind] = min(least[kind], float(d.min()))
                    hit = d < rule
                    close[kind] += [i[hit], j[hit]]
        involved = {k: np.unique(np.concatenate(close[k])) for k in kinds}
        return least, involved, complete

    def markers(
        self, kind: str, rule: float, involved: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The markers of the edge pairs closer than ``rule``, among ``involved``.

        The edges are cut into pieces of at most half the rule, and each pair
        of pieces closer than the rule, of an edge pair closer than the rule,
        gives a marker: markers run along a too-narrow stretch. Returns the
        markers ``(n, 2, 2)``, their edge pair's distance and their own.
        """
        from scipy.spatial import KDTree  # only needed here

        a, b, edge = cut_edges(self.a[involved], self.b[involved], rule / 2)
        edge = involved[edge]
        n = len(a)
        reach = rule + float(np.linalg.norm(b - a, axis=1).max())
        mid, piece, shift = self._tiled(0.5 * (a + b), reach)
        candidates = KDTree(mid).query_pairs(reach, output_type="ndarray")
        found: list[tuple[np.ndarray, ...]] = [
            (np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0))
            + (np.zeros(0, dtype=np.int64),) * 2
            + (np.zeros((0, 2), dtype=np.int64),) * 2
        ]
        for start in range(0, len(candidates), _PAIR_BLOCK):
            p, q = candidates[start : start + _PAIR_BLOCK].T
            keep = (p < n) | (q < n)
            p, q = p[keep], q[keep]
            e, f = edge[piece[p]], edge[piece[q]]
            keep = np.einsum("ij,ij->i", self.u[e], self.u[f]) < 0
            p, q, e, f = p[keep], q[keep], e[keep], f[keep]
            op, oq = shift[p] * self.period, shift[q] * self.period
            ends = (a[piece[p]] + op, b[piece[p]] + op, a[piece[q]] + oq,
                    b[piece[q]] + oq)  # fmt: skip
            local, which = _project((kind,), self.floor, *ends)[kind]
            ends_ = self._ends(e, f, shift[p], shift[q])
            whole = _project((kind,), self.floor, *ends_)[kind][0]
            hit = (local < rule) & (whole < rule)
            found.append(
                (
                    _markers(which[hit], *(x[hit] for x in ends)),
                    whole[hit],
                    local[hit],
                    piece[p[hit]],
                    piece[q[hit]],
                    shift[p[hit]],
                    shift[q[hit]],
                )
            )
        markers, distance, local, pi, pj, si, sj = (
            np.concatenate([f[k] for f in found]) for k in range(7)
        )
        if any(self.periodic):
            once = _once(pi, pj, si, sj)
            markers, distance, local = markers[once], distance[once], local[once]
        return markers, distance, local

    def sites(
        self,
        kind: str,
        markers: np.ndarray,
        distance: np.ndarray,
        local: np.ndarray,
        rule: float,
    ) -> list[Violation]:
        """Markers grouped into places: midpoints within the rule are one place.

        On a periodic axis the midpoints are compared modulo the period, so a
        place across the seam is one place whichever side its markers lie on.
        A place's location is the shortest marker of its narrowest edge pair.
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
            for axis in np.flatnonzero(self.periodic):
                box[axis] = self.period[axis]
                w = np.mod(mid[:, axis] - low[axis], self.period[axis])
                wrapped[:, axis] = np.where(w < self.period[axis], w, 0.0)
        near = KDTree(wrapped, boxsize=box).query_pairs(rule, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(len(mid),) * 2
        )
        _, label = connected_components(graph, directed=False)
        out = []
        for site in range(label.max() + 1):
            members = np.flatnonzero(label == site)
            best = members[np.lexsort((local[members], distance[members]))[0]]
            out.append(
                Violation(kind, float(distance[best]), mid[best], markers[members])
            )
        return out

    def _ends(
        self, i: np.ndarray, j: np.ndarray, si: np.ndarray, sj: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The endpoints of edges ``i`` and ``j``, each moved by its period shift."""
        if not any(self.periodic):
            return self.a[i], self.b[i], self.a[j], self.b[j]
        oi, oj = si * self.period, sj * self.period
        return self.a[i] + oi, self.b[i] + oi, self.a[j] + oj, self.b[j] + oj

    def _facing(
        self, candidates: np.ndarray, n: int, edge: np.ndarray, shift: np.ndarray
    ) -> _Pairs:
        """Candidate piece pairs as pairs of edges running opposite ways.

        Pairs keep a piece of this period; corners of 90 degrees or more never
        pair. ``edge`` and ``shift`` are each tiled piece's edge and shift. An
        edge pair found by several piece pairs measures the same each time.
        """
        p, q = candidates.T
        keep = (p < n) | (q < n)
        p, q = p[keep], q[keep]
        i, j = edge[p], edge[q]
        keep = np.einsum("ij,ij->i", self.u[i], self.u[j]) < 0
        return i[keep], j[keep], shift[p[keep]], shift[q[keep]]

    def _tiled(
        self, mid: np.ndarray, reach: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pieces' midpoints and, on periodic axes, their copies within reach.

        Returns the midpoints with the period's own pieces first, each one's
        piece index and the period shift ``(n, 2)`` of its copy.
        """
        n = len(mid)
        piece = np.arange(n)
        if not any(self.periodic) or self.bounds is None:
            return mid, piece, np.zeros((n, 2), dtype=np.int64)
        xmin, ymin, xmax, ymax = self.bounds
        ranges = [
            range(-k, k + 1) if p else range(1)
            for p, k in zip(
                self.periodic, np.ceil(reach / self.period).astype(int), strict=True
            )
        ]
        shifts = sorted(itertools.product(*ranges), key=lambda s: s != (0, 0))
        low, high = np.array([xmin, ymin]) - reach, np.array([xmax, ymax]) + reach
        tiles = []
        for s in shifts:
            copy = mid + np.asarray(s) * self.period
            # Only copies within reach of the footprint can pair with its edges.
            near = np.all((copy >= low) & (copy <= high), axis=1) | (s == (0, 0))
            tiles.append((copy[near], piece[near], s))
        return (
            np.concatenate([t[0] for t in tiles]),
            np.concatenate([t[1] for t in tiles]),
            np.concatenate([np.tile(t[2], (len(t[1]), 1)) for t in tiles]),
        )


def _orientations(
    e0: np.ndarray, e1: np.ndarray, f0: np.ndarray, f1: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """The four endpoint projections: (edge start, edge end, endpoint)."""
    return (e0, e1, f0), (e0, e1, f1), (f0, f1, e0), (f0, f1, e1)


def _project(
    kinds: tuple[str, ...], floor: float, *ends: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The projection distance of segment pairs ``(e0, e1, f0, f1)``, per kind.

    The least of the four endpoint projections whose foot falls on the other
    segment and that lie on its solid side (a width) or void side (a
    spacing), ``inf`` if none; and which projection it is (0 to 3).
    """
    out = {k: (np.full(len(ends[0]), np.inf), np.zeros(len(ends[0]), np.int64))
           for k in kinds}  # fmt: skip
    for n, (p0, p1, q) in enumerate(_orientations(*ends)):
        d = p1 - p0
        length = np.hypot(d[:, 0], d[:, 1])
        ux, uy = d[:, 0] / length, d[:, 1] / length
        w = q - p0
        t = w[:, 0] * ux + w[:, 1] * uy
        s = w[:, 1] * ux - w[:, 0] * uy  # along the normal towards the solid
        on = (t >= -floor) & (t <= length + floor)
        for kind in kinds:
            best, which = out[kind]
            side = s if kind == "width" else -s
            ok = on & (side > floor) & (side < best)
            out[kind] = np.where(ok, side, best), np.where(ok, n, which)
    return out


def _markers(which: np.ndarray, *ends: np.ndarray) -> np.ndarray:
    """Each pair's marker, foot to endpoint, for the projection ``which``."""
    out = np.zeros((len(which), 2, 2))
    for k, (p0, p1, q) in enumerate(_orientations(*ends)):
        m = which == k
        d = p1[m] - p0[m]
        u = d / np.hypot(d[:, 0], d[:, 1])[:, None]
        t = np.einsum("ij,ij->i", q[m] - p0[m], u)
        out[m, 0] = p0[m] + t[:, None] * u
        out[m, 1] = q[m]
    return out


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
