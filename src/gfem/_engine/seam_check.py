"""Whether a received periodic mesh can be wrapped: its seam and surface counts.

``MeshAccess::UpdateBuffers`` maps each second-side edge and face of a rank to its
first-side twin (last identification wins) and fails, or numbers DOFs wrongly,
when the twin is not on that rank. `seam_counts` predicts that from the
netgen mesh a rank received, before ``ngs.Mesh``: the missing twins in the declared
orientation, and the reverse ones, which are what would be missing with the pair
swapped on the same partition. A rank also holds the partner triangles of its
paired surfaces, so the rank sum of local triangles per surface tag is exactly
twice the mesh's on paired surfaces and exactly once elsewhere
(`expected_surface_counts`).

Only entities whose vertices all lie on an identified surface are built: memory is
booleans per local element and integers per surface vertex.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_TET_FACES = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
_TRI_EDGES = np.array([(0, 1), (0, 2), (1, 2)])
_SEGMENT = np.array([(0, 1)])
_TRIANGLE = np.array([(0, 1, 2)])


class SeamCounts(NamedTuple):
    """A rank's counts of distinct twins, in the order a reduction packs them."""

    #: Second-side edges whose first-side twin, declared orientation, is not held.
    missing_edges: int
    #: Second-side faces whose first-side twin, declared orientation, is not held.
    missing_faces: int
    #: The edges that would be missing with the pair swapped.
    reverse_edges: int
    #: The faces that would be missing with the pair swapped.
    reverse_faces: int


def seam_counts(ngmesh: Any) -> SeamCounts:
    """Missing edges, missing faces, reverse edges and reverse faces on this rank.

    Each count is of distinct twins. A mesh with no elements counts zero.

    Args:
        ngmesh: A netgen mesh, as received from ``Distribute``.
    """
    second, first, number = _identifications(ngmesh)
    tets = _nodes(ngmesh.Elements3D(), 4)
    tris = _nodes(ngmesh.Elements2D(), 3)
    segments = _nodes(ngmesh.Elements1D(), 2)
    out = np.zeros(4, dtype=np.int64)
    if len(second) == 0 or len(tets) + len(tris) + len(segments) == 0:
        return SeamCounts(*out.tolist())
    n_points = int(max(second.max(), first.max())) + 1
    for array in (tets, tris, segments):
        if len(array):
            n_points = max(n_points, int(array.max()))
    # NumPy() numbers nodes from 1
    on = np.zeros(n_points + 1, dtype=bool)
    on[first + 1] = True
    on[second + 1] = True
    edges = (
        np.concatenate(
            [
                _entities(tets, _TET_EDGES, on),
                _entities(tris, _TRI_EDGES, on),
                _entities(segments, _SEGMENT, on),
            ]
        )
        - 1
    )
    faces = np.concatenate(
        [_entities(tets, _TET_FACES, on), _entities(tris, _TRIANGLE, on)]
    )
    faces -= 1
    # a compact numbering of the surface vertices keeps a face's key in int64
    surface = np.flatnonzero(on) - 1
    base = max(len(surface), 1)
    if float(base) ** 3 >= 2.0**63:
        raise ValueError(f"{base} surface vertices: a face key overflows int64.")
    edge_keys = np.unique(_keys(np.searchsorted(surface, edges), base))
    face_keys = np.unique(_keys(np.searchsorted(surface, faces), base))
    edges = np.unique(edges, axis=0) if len(edges) else edges
    faces = np.unique(faces, axis=0) if len(faces) else faces
    for j, forward in enumerate((True, False)):
        lost: list[list[np.ndarray]] = [[], []]
        for k in np.unique(number):
            chosen = number == k
            sec, fir = _last_pair(second[chosen], first[chosen])
            source, target = (sec, fir) if forward else (fir, sec)
            for i, (rows, keys) in enumerate(((edges, edge_keys), (faces, face_keys))):
                images = _images(rows, source, target)
                if len(images):
                    found = _keys(np.searchsorted(surface, images), base)
                    lost[i].append(found[~np.isin(found, keys)])
        for i in range(2):
            out[2 * j + i] = len(np.unique(np.concatenate(lost[i]))) if lost[i] else 0
    return SeamCounts(*out.tolist())


def surface_counts(
    triangle_tags: np.ndarray, facet_order: np.ndarray, tags: np.ndarray
) -> np.ndarray:
    """How many local triangles carry each of ``tags``.

    Args:
        triangle_tags: The `gfem.Mesh`'s triangle tags.
        facet_order: This rank's triangles, as gfem triangle numbers.
        tags: The surface tags counted, sorted.
    """
    local = np.asarray(triangle_tags)[np.asarray(facet_order, dtype=np.int64)]
    return _count(local, tags)


def expected_surface_counts(
    triangle_tags: np.ndarray, tags: np.ndarray, paired: set[int]
) -> np.ndarray:
    """The rank sum `surface_counts` must reach: paired surfaces twice, others once.

    Args:
        triangle_tags: The `gfem.Mesh`'s triangle tags.
        tags: The surface tags counted, sorted.
        paired: The tags of the paired surfaces.
    """
    counts = _count(np.asarray(triangle_tags), tags)
    twice = np.isin(tags, np.fromiter(paired, dtype=np.int64, count=len(paired)))
    return np.where(twice, 2 * counts, counts)


def _count(values: np.ndarray, tags: np.ndarray) -> np.ndarray:
    position = np.searchsorted(tags, values)
    return np.bincount(position, minlength=len(tags))[: len(tags)].astype(np.int64)


def _identifications(ngmesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(second, first, number)``, 0-based: one Python pass over the local pairs."""
    pairs = ngmesh.GetIdentifications()
    n = len(pairs)
    first = np.fromiter((p[0].nr0 for p in pairs), np.int64, n)
    second = np.fromiter((p[1].nr0 for p in pairs), np.int64, n)
    number = np.fromiter((p[2] for p in pairs), np.int64, n)
    return second, first, number


def _nodes(elements: Any, k: int) -> np.ndarray:
    """The 1-based vertices of ``elements``: an empty list comes as (0,) or (0, 0)."""
    nodes = np.asarray(elements.NumPy()["nodes"])
    if nodes.ndim != 2 or len(nodes) == 0:
        return np.zeros((0, k), dtype=np.int64)
    return nodes[:, :k].astype(np.int64)


def _entities(cells: np.ndarray, local: np.ndarray, on: np.ndarray) -> np.ndarray:
    """The entities ``cells[:, local]`` with all vertices ``on``, rows sorted."""
    if len(cells) == 0:
        return np.zeros((0, local.shape[1]), dtype=np.int64)
    rows = [cells[:, loc][on[cells[:, loc]].all(axis=1)] for loc in local]
    return np.sort(np.concatenate(rows), axis=1)


def _keys(rows: np.ndarray, base: int) -> np.ndarray:
    keys = np.zeros(len(rows), dtype=np.int64)
    for j in range(rows.shape[1]):
        keys = keys * base + rows[:, j]
    return keys


def _last_pair(second: np.ndarray, first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One first vertex per second vertex, the last given, as a dict built in order."""
    order = np.argsort(second, kind="stable")
    s, f = second[order], first[order]
    last = np.r_[s[1:] != s[:-1], True]
    return s[last], f[last]


def _images(rows: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Every image of ``rows`` under the relation ``source`` to ``target``, rows sorted.

    A row has images when each of its vertices has one; several targets of a vertex
    give the cartesian product.
    """
    if len(rows) == 0 or len(source) == 0:
        return np.zeros((0, rows.shape[1]), dtype=np.int64)
    order = np.lexsort((target, source))
    s, t = source[order], target[order]
    low = np.searchsorted(s, rows, "left")
    count = np.searchsorted(s, rows, "right") - low
    has = (count > 0).all(axis=1)
    low, count = low[has], count[has]
    total = count.prod(axis=1)
    if (total == 1).all():
        return np.sort(t[low], axis=1)
    row = np.repeat(np.arange(len(low)), total)
    position = np.arange(total.sum()) - np.repeat(np.cumsum(total) - total, total)
    out = np.empty((len(row), rows.shape[1]), dtype=np.int64)
    for j in range(rows.shape[1] - 1, -1, -1):
        c = count[row, j]
        out[:, j] = t[low[row, j] + position % c]
        position = position // c
    return np.sort(out, axis=1)
