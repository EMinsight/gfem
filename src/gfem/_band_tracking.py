r"""Band connections between neighbouring wave vectors, from the modes' fields.

Each wave vector of a band diagram reports its bands in ascending
:math:`\mathrm{Re}\,\omega`. This module adds the connections: for each band
the band at the previous wave vector it continues, from the overlap

.. math::

    O_{ij} = \frac{|x_i^H M y_j|}{\lVert x_i\rVert_M \lVert y_j\rVert_M}

of the two modes' fields in the mass :math:`M` weighted by
:math:`\mathrm{Re}\,\varepsilon_r` on the unpaired space. A modulus makes the
arbitrary global phase of each mode irrelevant. Modes within a relative
frequency of ``GROUP_RTOL`` form a group whose basis the solver picks at
random (a degenerate cluster, or a pair the mesh splits): between groups the
overlap is taken between their spans, so it does not depend on that basis.
Everything here reads host numbers: the fields and frequencies of the solves
with their gradients stopped.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import scipy.optimize
import scipy.sparse as sp

from gfem._eigen.velocity import GROUP_RTOL, groups

#: A mode continues another when their overlap is at least this: more than a
#: quarter of its norm.
CONTINUES = 0.5
#: A connection is decided by overlap when it beats every other candidate in
#: its row and column by at least this.
MARGIN = 0.05
#: A wave vector component :math:`\mathbf k\cdot\mathbf a_j / 2\pi` this close
#: to an integer is one (for the Γ test).
GAMMA_ATOL = 1e-9


def overlaps(
    mass: sp.csr_matrix,
    a: np.ndarray,
    b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> np.ndarray:
    r"""The overlaps between the modes ``a`` and ``b``, group by group.

    The modes (columns) of each side form groups within ``GROUP_RTOL`` of
    their frequencies ``w_a`` and ``w_b``. For groups :math:`P` and
    :math:`Q`, the singular values of

    .. math::

        K_{PQ} = G_P^{-1/2} A_P^H M B_Q G_Q^{-1/2}

    with :math:`G` each group's Gram matrix in :math:`M`, are the cosines
    of the angles between the two spans, whatever basis the solver picked
    inside either group. Every pair of members gets their root mean square,
    :math:`\lVert K_{PQ}\rVert_F / \sqrt{\min(p, q)}`. Between two lone modes
    it is :math:`O_{ij}`; between a lone mode and a group, the norm of the
    mode's projection onto the group's span.
    """
    cross = a.conj().T @ (mass @ b)
    out = np.zeros(cross.shape)
    sides = [
        [
            (g, _inverse_sqrt(x[:, g].conj().T @ (mass @ x[:, g])))
            for g in groups(w, GROUP_RTOL)
        ]
        for x, w in ((a, w_a), (b, w_b))
    ]
    for p, root_p in sides[0]:
        for q, root_q in sides[1]:
            k = root_p @ cross[np.ix_(p, q)] @ root_q
            out[np.ix_(p, q)] = np.linalg.norm(k) / math.sqrt(min(len(p), len(q)))
    return out


def _inverse_sqrt(gram: np.ndarray) -> np.ndarray:
    """:math:`G^{-1/2}` of a Hermitian positive definite Gram matrix."""
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, values.max() * np.finfo(float).eps)
    return (vectors / np.sqrt(values)) @ vectors.conj().T


def gamma_equivalent(wave_vector: np.ndarray, translations: np.ndarray) -> bool:
    """Whether every translation's Bloch phase at ``wave_vector`` is 1."""
    turns = translations @ np.asarray(wave_vector, dtype=float) / (2 * np.pi)
    return bool(np.all(np.abs(turns - np.round(turns)) <= GAMMA_ATOL))


def connect(
    overlap: np.ndarray, w_prev: np.ndarray, w_cur: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Connect the modes of a wave vector to the modes of the previous one.

    A one-to-one pairing maximising :math:`\sum O_{ij}^2` over the pairs
    with :math:`O_{ij} \ge 1/2`. A pair is decided by overlap when it beats
    every other entry of its row and of its column by at least 0.05. Rival
    pairs, with each member's partner, form a set (members of one group of
    `overlaps` always do, their overlaps tie): it keeps as many pairs as the
    assignment made, between its lowest modes on each side in ascending
    frequency.

    Args:
        overlap: ``(n_prev, n_cur)``, the modes in ascending
            :math:`\mathrm{Re}\,\omega`.
        w_prev: ``(n_prev,)`` :math:`\mathrm{Re}\,\omega` of the previous
            modes.
        w_cur: ``(n_cur,)`` :math:`\mathrm{Re}\,\omega` of the modes.

    Returns:
        ``(link, by_overlap)``, one entry per current mode: the previous mode
        it continues (-1 for none) and whether the overlap decided it.
    """
    n_prev, n_cur = overlap.shape
    qualified = np.where(overlap >= CONTINUES, overlap**2, 0.0)
    rows, cols = scipy.optimize.linear_sum_assignment(-qualified)
    link = np.full(n_cur, -1, dtype=int)
    by_overlap = np.zeros(n_cur, dtype=bool)
    for i, j in zip(rows, cols, strict=True):
        if overlap[i, j] >= CONTINUES:
            link[j] = i
            by_overlap[j] = True
    # rival pairs: the entries within the margin of a pair in its row or column
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in np.flatnonzero(link >= 0):
        i = int(link[j])
        cols_ = [("c", int(c)) for c in range(n_cur) if c != j]
        rows_ = [("r", int(r)) for r in range(n_prev) if r != i]
        rivals = [x for x in cols_ if overlap[i, x[1]] > overlap[i, j] - MARGIN]
        rivals += [x for x in rows_ if overlap[x[1], j] > overlap[i, j] - MARGIN]
        if rivals:
            for x in [("r", i), ("c", int(j)), *rivals]:
                parent[find(x)] = find(("r", i))
    rivals_sets: dict[tuple[str, int], tuple[set[int], set[int]]] = {}
    for kind, idx in list(parent):
        members = rivals_sets.setdefault(find((kind, idx)), (set(), set()))
        members[0 if kind == "r" else 1].add(idx)
    for rows_in, cols_in in rivals_sets.values():
        _close(link, rows_in, cols_in)
        cols = sorted(cols_in, key=lambda j: (w_cur[j], j))
        n_pairs = int((link[cols] >= 0).sum())
        by_overlap[cols] = False
        link[cols] = -1
        # only modes reaching the threshold with the other side take part
        rows_ok = [i for i in rows_in if (overlap[i, cols] >= CONTINUES).any()]
        cols_ok = [j for j in cols if (overlap[sorted(rows_in), j] >= CONTINUES).any()]
        rows = sorted(rows_ok, key=lambda i: (w_prev[i], i))[:n_pairs]
        for j, i in zip(cols_ok, rows, strict=False):
            link[j] = i
    return link, by_overlap


def _close(link: np.ndarray, rows: set[int], cols: set[int]) -> None:
    """Add to a rival set the partner of each of its members, until none is new."""
    while True:
        new_rows = {int(link[j]) for j in cols if link[j] >= 0} - rows
        new_cols = {int(j) for j in np.flatnonzero(link >= 0) if link[j] in rows}
        new_cols -= cols
        if not (new_rows or new_cols):
            return
        rows |= new_rows
        cols |= new_cols


@dataclasses.dataclass(frozen=True)
class Tracking:
    """The connections of a whole path, in band numbers (static bands first).

    Attributes:
        static: ``(n_k,)`` the number of static bands filled at each wave vector.
        connection: ``(n_k, n_bands)`` the band at the previous wave vector
            each band continues, -1 for none.
        by_overlap: ``(n_k, n_bands)`` whether the overlap decided it.
        lost: ``(n_k, n_bands)`` bands inside the window with no partner.
        lost_at: ``(k, band, neighbour)`` of every lost band, for the warning.
    """

    static: np.ndarray
    connection: np.ndarray
    by_overlap: np.ndarray
    lost: np.ndarray
    lost_at: tuple[tuple[int, int, int], ...]


def _interior(matched: np.ndarray, anchored: bool) -> list[int]:
    """The unmatched modes that are not at an edge of the window.

    The top edge is any unmatched mode with no matched mode above it. The
    bottom edge counts only when the window is not anchored.
    """
    out = []
    for j in np.flatnonzero(~matched):
        top = not matched[j + 1 :].any()
        bottom = not anchored and not matched[:j].any()
        if not (top or bottom):
            out.append(int(j))
    return out


def _static_bands(
    w: np.ndarray,
    matched_prev: list[np.ndarray],
    matched_cur: list[np.ndarray],
    use: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], list[int]], dict[int, int]]:
    """The static bands of each wave vector in ``use``, from its neighbours.

    The number is that of unmatched modes of a neighbour that lie below the
    lowest mode found there. The neighbours must agree, else no fill: the
    band numbers there are off by a count unknown.

    Returns:
        The count per wave vector; per ``(wave vector, neighbour)``, the
        neighbour's modes that go to zero; per wave vector whose neighbours
        disagree, one of them.
    """
    n_k, n = w.shape
    s = np.zeros(n_k, dtype=int)
    below: dict[tuple[int, int], list[int]] = {}
    disagree: dict[int, int] = {}
    for g in map(int, np.flatnonzero(use)):
        sides = {}
        if g > 0:  # the previous wave vector's modes
            sides[g - 1] = ~matched_prev[g - 1]
        if g < n_k - 1:  # the next wave vector's modes
            sides[g + 1] = ~matched_cur[g]
        low = {
            q: [int(j) for j in np.flatnonzero(free) if w[q, j] < w[g, 0]]
            for q, free in sides.items()
        }
        counts = {len(modes) for modes in low.values()}
        if len(counts) == 1:
            s[g] = min(counts.pop(), n)
            for q, modes in low.items():
                below[g, q] = modes
        elif counts:
            disagree[g] = min(low)
    return s, below, disagree


def track(
    overlap: Sequence[np.ndarray],
    w: np.ndarray,
    anchored: np.ndarray,
    gamma: np.ndarray,
) -> Tracking:
    r"""Connect every wave vector's modes and fill the static bands at Γ.

    Args:
        overlap: ``n_k - 1`` matrices, step ``i`` between the modes of wave
            vectors ``i`` and ``i + 1``.
        w: ``(n_k, n_bands)`` :math:`\mathrm{Re}\,\omega` of the modes
            found, ascending.
        anchored: ``(n_k,)`` the target is below half the highest mode found.
        gamma: ``(n_k,)`` the wave vector is Γ-equivalent.
    """
    n_k, n = w.shape
    links, bys, matched_prev, matched_cur = [], [], [], []
    for i, o in enumerate(overlap):
        link, by = connect(o, w[i], w[i + 1])
        links.append(link)
        bys.append(by)
        matched_cur.append(link >= 0)
        taken = np.zeros(n, bool)
        taken[link[link >= 0]] = True
        matched_prev.append(taken)
    s, below, disagree = _static_bands(w, matched_prev, matched_cur, gamma & anchored)
    connection = np.full((n_k, n), -1, dtype=int)
    by_overlap = np.zeros((n_k, n), dtype=bool)
    lost = np.zeros((n_k, n), dtype=bool)
    lost_at: dict[tuple[int, int], int] = {}

    def flag(k: int, mode: int, other: int) -> None:
        if mode + s[k] < n:  # not a mode the static bands pushed out
            lost[k, mode + s[k]] = True
            lost_at.setdefault((k, mode + s[k]), other)

    for i, (link, by) in enumerate(zip(links, bys, strict=True)):
        cur, prev = i + 1, i
        for j in np.flatnonzero(link >= 0):
            # a partner the static bands pushed out is not in the diagram
            if j + s[cur] < n and link[j] + s[prev] < n:
                connection[cur, j + s[cur]] = link[j] + s[prev]
                by_overlap[cur, j + s[cur]] = by[j]
        # a static band continues from (or into) the modes that go to zero
        into = sorted(below.get((prev, cur), []), key=lambda j: w[cur, j])
        for band, j in enumerate(into):
            connection[cur, j + s[cur]] = band
        out_of = sorted(below.get((cur, prev), []), key=lambda j: w[prev, j])
        for band, j in enumerate(out_of):
            connection[cur, band] = j + s[prev]
        for j in _interior(matched_cur[i], bool(anchored[cur])):
            if j not in into:
                flag(cur, j, prev)
        for j in _interior(matched_prev[i], bool(anchored[prev])):
            if j not in out_of:
                flag(prev, j, cur)
    for g, other in disagree.items():
        for band in range(n):
            flag(g, band, other)
    return Tracking(
        s,
        connection,
        by_overlap,
        lost,
        tuple((k, b, o) for (k, b), o in sorted(lost_at.items())),
    )
