"""Mode-field reach: the options that leave mode fields out, the isolated modes.

A solve's mode fields differentiate only on the CPU with the direct linear
solver, without a Bloch wave vector or a periodic pair, and at
``SolverOptions.eigen_tolerance`` at most `FIELD_TOLERANCE`; `blocker` names
the reach blocker, the first of these conditions a solve fails (the polish
subset leaves the tolerance out).

A found mode's mode fields differentiate, and the forward polishes it, only
when it is isolated: alone in its cluster (`clusters`, frequencies within
``SolverOptions.cluster_rtol``), its cluster not cut short by the guard mode
(`incomplete`), and its relative gap to its gap set (`GapSet`) at least
`ISOLATED` (``cluster_rtol`` if larger). `ModeReach` holds these answers per
mode. The mode search builds the gap set; the eigenmode rule and mesh
refinement read the answers, each in its own words.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gfem.options import SolverOptions

__all__ = [
    "FIELD_TOLERANCE",
    "ISOLATED",
    "Blocker",
    "GapSet",
    "ModeReach",
    "blocker",
    "clusters",
    "incomplete",
    "isolated",
    "threshold",
]

#: The smallest relative gap of a mode in mode-field reach (with
#: ``SolverOptions.cluster_rtol`` if larger): an isolated mode,
#: polished in the forward and given mode-field gradients.
ISOLATED = 1e-4
#: The largest ``SolverOptions.eigen_tolerance`` of a solve whose mode
#: fields differentiate: it keeps the forward polish's Newton start close.
FIELD_TOLERANCE = 1e-10

#: A condition that leaves mode fields out of reach.
Blocker = Literal["wave_vector", "periodic", "device", "linear", "tolerance"]


def blocker(
    options: SolverOptions,
    linear: str,
    *,
    wave_vector: object,
    periodic: bool,
    tolerance: bool = True,
) -> Blocker | None:
    """The reach blocker: the first condition leaving mode fields out, or ``None``.

    In this order: a Bloch ``wave_vector`` (not ``None``), a ``periodic``
    pair, a device other than the CPU, a ``linear`` solver kind other than
    ``"direct"`` (as resolved by the caller), and an
    ``options.eigen_tolerance`` above `FIELD_TOLERANCE`. With ``tolerance``
    false, the last is left out: the polish subset, where the forward
    polishes its isolated modes.
    """
    if wave_vector is not None:
        return "wave_vector"
    if periodic:
        return "periodic"
    if options.device != "cpu":
        return "device"
    if linear != "direct":
        return "linear"
    if tolerance and options.eigen_tolerance > FIELD_TOLERANCE:
        return "tolerance"
    return None


def threshold(rtol: float) -> float:
    """The isolation threshold: `ISOLATED`, or ``rtol`` when that is larger."""
    return max(ISOLATED, rtol)


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


def incomplete(
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
    relative gap to its gap set is at least `ISOLATED`, or ``rtol``
    (``SolverOptions.cluster_rtol``) when that is larger. An entry stores a
    gap of 0 for a mode whose forward polish failed (its ``polish_failed``
    field), so such a mode is never isolated.
    """
    labels = np.asarray(labels)
    single = np.array([np.count_nonzero(labels == v) == 1 for v in labels], dtype=bool)
    return (
        single
        & ~np.asarray(incomplete, dtype=bool)
        & (np.asarray(gap) >= threshold(rtol))
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


@dataclasses.dataclass(frozen=True)
class ModeReach:
    """Which found modes are in mode-field reach, and the evidence.

    Attributes:
        labels: The modes' cluster labels (`clusters`).
        incomplete: Whether each mode's cluster reaches the guard mode.
        gap: Each mode's relative gap to its gap set (`GapSet.gap`, ``inf``
            without a neighbour), 0 for a mode whose polish failed.
        polish_failed: Whether each mode's polish in the forward failed.
        rtol: ``SolverOptions.cluster_rtol``.
    """

    labels: np.ndarray
    incomplete: np.ndarray
    gap: np.ndarray
    polish_failed: np.ndarray
    rtol: float

    @property
    def isolated(self) -> np.ndarray:
        """Which modes are isolated (`isolated`)."""
        return isolated(self.labels, self.incomplete, self.gap, self.rtol)

    @property
    def threshold(self) -> float:
        """The isolation threshold (`threshold`) at ``rtol``."""
        return threshold(self.rtol)

    @classmethod
    def of(
        cls,
        ks: list[complex],
        guard: complex | None,
        gaps: GapSet,
        polish_failed: np.ndarray,
        rtol: float,
    ) -> Self:
        """The reach of modes ``ks``, the guard mode's ``guard``, gap set ``gaps``."""
        labels = clusters(ks, rtol)
        gap = np.array([gaps.gap(k) for k in ks])
        failed = np.asarray(polish_failed, dtype=bool)
        gap[failed] = 0.0
        return cls(labels, incomplete(ks, labels, guard, rtol), gap, failed, rtol)

    @classmethod
    def from_fields(cls, fields: Mapping[str, np.ndarray], rtol: float) -> Self:
        """The reach an entry's fields record (as `ModeReach.fields` gives them).

        An entry without ``polish_failed`` had no polish fail.
        """
        failed = fields.get("polish_failed", np.zeros(len(fields["k"]), dtype=bool))
        return cls(fields["cluster"], fields["incomplete"], fields["gap"], failed, rtol)

    def fields(self) -> dict[str, np.ndarray]:
        """The entry fields ``cluster``, ``incomplete``, ``gap``, ``polish_failed``.

        The arrays themselves, not copies.
        """
        return {
            "cluster": self.labels,
            "incomplete": self.incomplete,
            "gap": self.gap,
            "polish_failed": self.polish_failed,
        }
