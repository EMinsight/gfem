"""Mode-field reach: which found modes are isolated.

A found mode's mode fields differentiate, and the forward polishes it, only
when it is isolated: alone in its cluster (`clusters`, frequencies within
``SolverOptions.cluster_rtol``), its cluster not cut short by the guard mode
(`_incomplete`), and its relative gap to its gap set (`GapSet`) at least
`_ISOLATED` (``cluster_rtol`` if larger). The mode search builds the gap
set; the eigenmode rule and mesh refinement read the answers.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["GapSet", "clusters", "isolated"]

#: The smallest relative gap of a mode in mode-field reach (with
#: ``SolverOptions.cluster_rtol`` if larger): an isolated mode,
#: polished in the forward and given mode-field gradients.
_ISOLATED = 1e-4


@dataclasses.dataclass(frozen=True)
class GapSet:
    """The modes a mode's relative gap is measured to.

    Attributes:
        pool: The ``k`` of every converged, distinct pair the selection
            accepted, those cut beyond ``n_modes`` included, and of a mode
            found after the count was met: the modes themselves among them.
        guard: The guard mode's ``k``, or ``None`` when none was found.
        ritz: The seed solve's unrefined Ritz values (polynomial operators
            only): the gap set when the pool and the guard leave it empty.
    """

    pool: np.ndarray
    guard: complex | None
    ritz: np.ndarray

    def gap(self, k: complex) -> float:
        """``min |k - k_j| / max(|k|, |k_j|)`` over ``k``'s neighbours.

        The neighbours are the pool and the guard, ``k``'s own entry (the
        pool's nearest to it) left out; without any, the Ritz values but
        the nearest to ``k`` (its own seed). ``inf`` when there are none:
        the mode is isolated.
        """
        near = _others(self.pool, k)
        if self.guard is not None:
            near = np.append(near, self.guard)
        if not len(near):
            near = _others(self.ritz, k)
        if not len(near):
            return math.inf
        return float(np.min(np.abs(near - k) / np.maximum(abs(k), np.abs(near))))


def _others(values: np.ndarray, k: complex) -> np.ndarray:
    """``values`` without the one nearest ``k`` (``k``'s own)."""
    if not len(values):
        return values
    return np.delete(values, int(np.argmin(np.abs(values - k))))


def _close(a: complex, b: complex, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def _incomplete(
    ks: list[complex], labels: np.ndarray, guard: complex | None, rtol: float
) -> np.ndarray:
    """Whether each mode's cluster reaches the guard mode ``guard`` (cut short)."""
    incomplete = np.zeros(len(ks), dtype=bool)
    if guard is None:
        return incomplete
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if any(_close(guard, ks[j], rtol) for j in members):
            incomplete[members] = True
    return incomplete


def isolated(
    labels: np.ndarray, incomplete: np.ndarray, gap: np.ndarray, rtol: float
) -> np.ndarray:
    """Which modes are in mode-field reach.

    A mode is when it is the only member of its cluster (``labels``), its
    cluster is not cut short by the guard mode (``incomplete``), and its
    relative gap to its gap set is at least `_ISOLATED`, or ``rtol``
    (``SolverOptions.cluster_rtol``) when that is larger. An entry stores a
    gap of 0 for a mode whose forward polish failed (its ``polish_failed``
    field), so such a mode is never isolated.
    """
    labels = np.asarray(labels)
    single = np.array([np.count_nonzero(labels == v) == 1 for v in labels], dtype=bool)
    return (
        single
        & ~np.asarray(incomplete, dtype=bool)
        & (np.asarray(gap) >= max(_ISOLATED, rtol))
    )


def clusters(ks: list[complex], rtol: float) -> np.ndarray:
    """A label per eigenvalue: chains of eigenvalues within ``rtol`` share one.

    Labels count from 0 in order of first appearance.
    """
    labels = list(range(len(ks)))
    for i in range(len(ks)):
        for j in range(i):
            if _close(ks[i], ks[j], rtol) and labels[i] != labels[j]:
                old, new = labels[i], labels[j]
                labels = [new if v == old else v for v in labels]
    renumber: dict[int, int] = {}
    for v in labels:
        renumber.setdefault(v, len(renumber))
    return np.array([renumber[v] for v in labels], dtype=np.int64)
