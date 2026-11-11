"""Bisection refinement of a gfem mesh (ADR 0083).

Longest-edge (Rivara) bisection with conforming closure, on gfem's arrays in
NumPy:

- every marked tetrahedron puts its longest edge in the set of split edges;
- each pass, every tetrahedron holding a split edge bisects on its own longest
  edge (which joins the set), until no tetrahedron holds one;
- triangles then bisect the same way: a face's first bisected edge is always
  its longest, so faces follow their tetrahedra.

Edges are ordered globally by length, then by their sorted node ids, so the
result does not depend on anything but the mesh and the mark. A child keeps
its parent's node order with one vertex replaced by the midpoint, so it keeps
the parent's orientation; unrefined elements keep their relative order, old
nodes keep their indices and midpoints are appended.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gfem._pytree import readonly
from gfem.mesh import _TET_EDGES, Mesh, _Topology, _triangle_neighbours

_TET_EDGE_PAIRS = np.array(_TET_EDGES)
_TRI_EDGE_PAIRS = np.array([(0, 1), (0, 2), (1, 2)])
# Edge key: the smaller node id times this, plus the larger.
_KEY_BASE = np.int64(1) << 31


@dataclasses.dataclass(frozen=True)
class Refinement:
    """Where each element of a bisected mesh comes from, one level up.

    Attributes:
        tetrahedron_parents: For each tetrahedron of the refined mesh, the row
            of the input mesh's `tetrahedra` it lies in.
        triangle_parents: For each triangle of the refined mesh, the row of the
            input mesh's `triangles` it lies in.
    """

    tetrahedron_parents: np.ndarray
    triangle_parents: np.ndarray


def bisect(mesh: Mesh, marked) -> tuple[Mesh, Refinement]:
    """Bisect the marked tetrahedra, and their neighbours until the mesh conforms.

    The mesh is bisected at its current node coordinates, which become the
    refined mesh's imported coordinates. Every marked tetrahedron is bisected
    at least once; children keep their parent's tags, entities and
    orientation, and surface triangles split with the faces they lie on.

    Args:
        mesh: The mesh to refine (concrete node coordinates).
        marked: Boolean mask over the mesh's tetrahedra, shape ``(n_tet,)``.

    Returns:
        The refined mesh (a new topology with no source file) and the one-level
        parent indices of its tetrahedra and triangles.

    Raises:
        ValueError: ``marked`` is not a boolean mask over the tetrahedra.
    """
    tets = np.asarray(mesh.tetrahedra, np.int64)
    marked = np.asarray(marked)
    if marked.dtype != bool or marked.shape != (len(tets),):
        raise ValueError(
            f"marked must be a boolean array of shape ({len(tets)},), one entry "
            f"per tetrahedron; got {marked.dtype} of shape {marked.shape}."
        )
    splits = _Splits(np.asarray(mesh.nodes, dtype=float))
    first = tets[marked]
    splits.add(np.unique(_longest_edge_keys(splits.x, first, _TET_EDGE_PAIRS)))
    tets, tet_parents = _bisect_cells(
        splits, tets, np.arange(len(tets)), _TET_EDGE_PAIRS, grow=True
    )
    tris = np.asarray(mesh.triangles, np.int64)
    tris, tri_parents = _bisect_cells(
        splits, tris, np.arange(len(tris)), _TRI_EDGE_PAIRS, grow=False
    )
    x = splits.x
    old = mesh._topology  # noqa: SLF001
    topology = _Topology(
        length_unit=old.length_unit,
        imported_nodes=readonly(x),
        regions=old.regions,
        tetrahedra=readonly(tets),
        tet_tags=readonly(old.tet_tags[tet_parents]),
        tet_entities=readonly(old.tet_entities[tet_parents]),
        triangles=readonly(tris),
        tri_tags=readonly(old.tri_tags[tri_parents]),
        tri_entities=readonly(old.tri_entities[tri_parents]),
        tri_neighbours=readonly(
            _triangle_neighbours(x, tets, tris, Path("refined mesh"))
        ),
        source=None,
    )
    refinement = Refinement(
        tetrahedron_parents=readonly(tet_parents),
        triangle_parents=readonly(tri_parents),
    )
    return Mesh(jnp.asarray(x), topology), refinement


class _Splits:
    """The split edges, by key, with their midpoint node ids; and all nodes."""

    def __init__(self, nodes: np.ndarray) -> None:
        self.x = nodes
        self.keys = np.empty(0, np.int64)
        self.mids = np.empty(0, np.int64)

    def has(self, keys: np.ndarray) -> np.ndarray:
        if not len(self.keys):
            return np.zeros(keys.shape, bool)
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return self.keys[i] == keys

    def midpoint(self, keys: np.ndarray) -> np.ndarray:
        return self.mids[np.searchsorted(self.keys, keys)]

    def add(self, keys: np.ndarray) -> None:
        """Split these new edges (unique, sorted keys): append their midpoints."""
        a, b = keys // _KEY_BASE, keys % _KEY_BASE
        mids = len(self.x) + np.arange(len(keys))
        self.x = np.concatenate([self.x, 0.5 * (self.x[a] + self.x[b])])
        all_keys = np.concatenate([self.keys, keys])
        order = np.argsort(all_keys, kind="stable")
        self.keys = all_keys[order]
        self.mids = np.concatenate([self.mids, mids])[order]


def _keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b) * _KEY_BASE + np.maximum(a, b)


def _longest_local_edge(x: np.ndarray, cells: np.ndarray, pairs: np.ndarray):
    """Per cell, its longest local edge: by length, then the larger edge key."""
    a, b = cells[:, pairs[:, 0]], cells[:, pairs[:, 1]]
    length = np.linalg.norm(x[a] - x[b], axis=-1)
    tied = length == length.max(axis=1, keepdims=True)
    return np.argmax(np.where(tied, _keys(a, b), -1), axis=1)


def _longest_edge_keys(x, cells, pairs) -> np.ndarray:
    k = _longest_local_edge(x, cells, pairs)
    rows = np.arange(len(cells))
    return _keys(cells[rows, pairs[k, 0]], cells[rows, pairs[k, 1]])


def _bisect_cells(splits: _Splits, cells, parents, pairs, *, grow: bool):
    """Bisect every cell holding a split edge, on its longest edge, until none does.

    With ``grow``, a longest edge not yet split joins the split set (the
    tetrahedra's closure); without it, it must already be split (triangles
    follow their tetrahedra).
    """
    while True:
        edge_keys = _keys(cells[:, pairs[:, 0]], cells[:, pairs[:, 1]])
        active = splits.has(edge_keys).any(axis=1)
        if not active.any():
            return cells, parents
        hit = cells[active]
        k = _longest_local_edge(splits.x, hit, pairs)
        rows = np.arange(len(hit))
        i, j = pairs[k, 0], pairs[k, 1]
        keys = _keys(hit[rows, i], hit[rows, j])
        new = ~splits.has(keys)
        if new.any():
            if not grow:
                raise RuntimeError(
                    "Bisection left a surface triangle split off its longest edge."
                )
            splits.add(np.unique(keys[new]))
        mid = splits.midpoint(keys)
        first, second = hit.copy(), hit.copy()
        first[rows, j] = mid
        second[rows, i] = mid
        cells = np.concatenate([cells[~active], first, second])
        parents = np.concatenate([parents[~active], parents[active], parents[active]])
