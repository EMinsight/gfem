r"""Band connections between neighbouring wave vectors, from the modes' fields.

Each wave vector of a band diagram reports its bands in ascending
:math:`\mathrm{Re}\,\omega`. This module adds the connections: for each band
the band at the previous wave vector it continues, from the overlap

.. math::

    O_{ij} = \frac{|x_i^H M y_j|}{\lVert x_i\rVert_M \lVert y_j\rVert_M}

of the two modes' fields in the mass :math:`M` weighted by
:math:`\mathrm{Re}\,\varepsilon_r` on the unpaired space. A modulus makes the
arbitrary global phase of each mode irrelevant. Everything here reads host
numbers: the fields and frequencies of the solves with their gradients
stopped.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import ngsolve as ngs
import numpy as np
import scipy.optimize
import scipy.sparse as sp

from gfem.model import Model

#: A mode continues another when their overlap is at least this: more than a
#: quarter of its norm.
CONTINUES = 0.5
#: A connection is decided by overlap when it beats every other candidate in
#: its row and column by at least this.
MARGIN = 0.05
#: A wave vector component of ``k . a_j / 2 pi`` this close to an integer is
#: one (for the Γ test).
GAMMA_ATOL = 1e-9


def mass_matrix(model: Model, order: int) -> sp.csr_matrix:
    r"""The mass :math:`\int \mathrm{Re}\,\varepsilon_r\, u\cdot v`, unpaired space.

    ``model`` carries concrete values (its leaves under
    `jax.lax.stop_gradient`); ``order`` is the solve's H(curl) order. A
    perfectly matched layer holds no weight, as the mode energy.
    """
    from gfem._engine.coefficients import coefficient
    from gfem._engine.driven import driven_operator
    from gfem._engine.eigen import _physical_eps

    n_tets = len(model.mesh.tetrahedra)
    n_tris = len(model.mesh.triangles)
    op = driven_operator(
        model,
        order,
        elements=np.zeros(n_tets, bool),
        faces=np.zeros(n_tris, bool),
        paired=False,
    )
    eps = coefficient(op.built, np.real(_physical_eps(model)))
    u, v = op.space.TnT()
    form = ngs.BilinearForm(op.space, check_unused=False)
    form += (eps * u) * v * ngs.dx(bonus_intorder=2)
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    n = op.space.ndof
    return sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
    )


def overlaps(mass: sp.csr_matrix, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|<a_i, b_j>_M| / (|a_i| |b_j|)`` for the columns of ``a`` and ``b``."""
    num = np.abs(a.conj().T @ (mass @ b))
    na = np.sqrt(np.real(np.sum(a.conj() * (mass @ a), axis=0)))
    nb = np.sqrt(np.real(np.sum(b.conj() * (mass @ b), axis=0)))
    return num / np.outer(na, nb)


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
    every other entry of its row and of its column by at least 0.05; the
    modes of a group of rival pairs (a degenerate cluster, a near-degenerate
    pair the mesh splits) are paired in ascending frequency instead, when
    both sides hold as many.

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
    groups: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    for kind, idx in list(parent):
        members = groups.setdefault(find((kind, idx)), ([], []))
        members[0 if kind == "r" else 1].append(idx)
    for rows_in, cols_in in groups.values():
        by_overlap[cols_in] = False
        if len(rows_in) == len(cols_in):
            by_frequency = sorted(rows_in, key=lambda i: (w_prev[i], i))
            link[sorted(cols_in, key=lambda j: (w_cur[j], j))] = by_frequency
    return link, by_overlap


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
) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
    """The static bands of each wave vector in ``use``, from its neighbours.

    The number is that of unmatched modes of a neighbour that lie below the
    lowest mode found there. The neighbours must agree, else no fill and
    those modes are lost.

    Returns:
        The count per wave vector, and, per ``(wave vector, neighbour)``, the
        neighbour's modes that go to zero.
    """
    n_k, n = w.shape
    s = np.zeros(n_k, dtype=int)
    below: dict[tuple[int, int], list[int]] = {}
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
    return s, below


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
    s, below = _static_bands(w, matched_prev, matched_cur, gamma & anchored)
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
            if j + s[cur] < n:
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
    return Tracking(
        s,
        connection,
        by_overlap,
        lost,
        tuple((k, b, o) for (k, b), o in sorted(lost_at.items())),
    )
