r"""The Bloch wave vectors of a periodic cell's band diagram.

A model with `gfem.Periodic` pairs is a periodic cell of translations
:math:`\mathbf a_j`. A Bloch mode of wave vector :math:`\mathbf k` repeats
from cell to cell up to a phase:

.. math::

    \mathbf E(\mathbf r + \mathbf a_j)
        = e^{-i\mathbf k\cdot\mathbf a_j}\,\mathbf E(\mathbf r)

The reciprocal lattice `gfem.Model.reciprocal_lattice` holds the
:math:`\mathbf b_i` with

.. math::

    \mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}

inside the span of the translations. :math:`\mathbf k` and
:math:`\mathbf k + \mathbf b_i` give every pair the same phase, so the first
Brillouin zone holds every Bloch mode once. `wave_vector_path` samples
straight segments between its high-symmetry points (Γ, X, M, K, R…), given
in reduced coordinates :math:`c_i` of :math:`\mathbf k = \sum_i c_i
\mathbf b_i`.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Mapping, Sequence

import numpy as np

from gfem._dim2 import refuse_entry
from gfem.model import Model

__all__ = ["WaveVectorPath", "wave_vector_path"]

_PATH = "wave_vector_path"
_GAMMA = "Γ"
_ALIASES = {"G": _GAMMA, "Gamma": _GAMMA}
#: Relative tolerance on the lengths and angles of the translations when the
#: lattice is recognised.
LATTICE_RTOL = 1e-6
#: The named points of each lattice besides Γ, in reduced coordinates; X lies
#: along the first pair. The hexagonal K depends on the basis angle and is
#: placed by `_named_points`.
_NAMED: dict[str, dict[str, tuple[float, ...]]] = {
    "line": {"X": (0.5,)},
    "square": {"X": (0.5, 0.0), "M": (0.5, 0.5)},
    "rectangular": {"X": (0.5, 0.0), "Y": (0.0, 0.5), "S": (0.5, 0.5)},
    "hexagonal": {"M": (0.5, 0.0)},
    "cubic": {"X": (0.5, 0.0, 0.0), "M": (0.5, 0.5, 0.0), "R": (0.5, 0.5, 0.5)},
    "other": {},
}


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class WaveVectorPath:
    r"""The Bloch wave vectors a band diagram samples, in path order.

    Built by `wave_vector_path`; the constructor wraps any wave vectors
    (``WaveVectorPath(k)``) and computes ``distance``. Host arrays: nothing
    in it is differentiable.

    Attributes:
        wave_vectors: ``(n_k, 3)`` Cartesian Bloch wave vectors, rad/m
            (read-only).
        labels: The high-symmetry points' names in path order (a name may
            repeat: Γ … Γ).
        label_index: The wave-vector index of each label.
        lattice: ``"line"``, ``"square"``, ``"rectangular"``,
            ``"hexagonal"``, ``"cubic"`` or ``"other"`` (the default, also
            for wave vectors given directly).
        distance: ``(n_k,)`` cumulative path length
            :math:`\sum\lvert\Delta\mathbf k\rvert` from 0, rad/m: the
            abscissa of a band diagram (read-only).
    """

    wave_vectors: np.ndarray
    labels: tuple[str, ...] = ()
    label_index: tuple[int, ...] = ()
    lattice: str = "other"
    distance: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        k = np.array(self.wave_vectors, dtype=float)
        if k.ndim != 2 or k.shape[1] != 3 or not len(k) or not np.isfinite(k).all():
            raise ValueError(
                "WaveVectorPath: the wave vectors must be finite numbers of shape "
                f"(n_k, 3) in rad/m with n_k ≥ 1, got shape {k.shape}."
            )
        labels = tuple(self.labels)
        if not all(isinstance(name, str) for name in labels):
            raise TypeError(f"WaveVectorPath: labels are names, got {labels!r}.")
        index = tuple(operator.index(i) for i in self.label_index)
        if len(index) != len(labels):
            raise ValueError(
                f"WaveVectorPath: {len(labels)} labels but {len(index)} "
                "label_index entries: give one wave-vector index per label."
            )
        for name, i in zip(labels, index, strict=True):
            if not 0 <= i < len(k):
                raise IndexError(
                    f"WaveVectorPath: label {name!r} is at wave vector {i}, but "
                    f"the path has {len(k)}, numbered from 0."
                )
        if self.lattice not in _NAMED:
            raise ValueError(
                f"WaveVectorPath: lattice is one of {', '.join(map(repr, _NAMED))}, "
                f"got {self.lattice!r}."
            )
        distance = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))]
        )
        k.setflags(write=False)
        distance.setflags(write=False)
        object.__setattr__(self, "wave_vectors", k)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_index", tuple(int(i) for i in index))
        object.__setattr__(self, "distance", distance)

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def ticks(self) -> tuple[np.ndarray, list[str]]:
        """``(positions, names)`` of the labels on the ``distance`` axis.

        ``plt.xticks(*path.ticks)`` marks the high-symmetry points.
        """
        return self.distance[list(self.label_index)], list(self.labels)

    def index(self, label: str) -> int:
        """The index of the first wave vector carrying ``label``.

        ``"G"`` and ``"Gamma"`` find Γ.

        Raises:
            ValueError: No wave vector carries ``label``.
        """
        name = _canonical(label)
        if name not in self.labels:
            raise ValueError(
                f"{label!r} is not a label of this path "
                f"({', '.join(dict.fromkeys(self.labels)) or 'none'})."
            )
        return self.label_index[self.labels.index(name)]

    def __repr__(self) -> str:
        n = len(self)
        parts = ["-".join(self.labels)] if self.labels else []
        parts += [
            f"{n} wave vector{'s' if n != 1 else ''}",
            f"{self.lattice} lattice",
        ]
        return f"WaveVectorPath({', '.join(parts)})"


def wave_vector_path(
    model: Model,
    points: Sequence[str],
    *,
    n_points: int,
    named_points: Mapping[str, Sequence[float]] | None = None,
) -> WaveVectorPath:
    r"""The Bloch wave vectors along straight segments between high-symmetry points.

    The lattice is recognised from the model's `gfem.Model.periodic_translations`
    (lengths and angles to a relative ``LATTICE_RTOL``), and names its
    high-symmetry points in reduced coordinates :math:`c_i` of
    :math:`\mathbf k = \sum_i c_i \mathbf b_i`, the :math:`\mathbf b_i` of
    `gfem.Model.reciprocal_lattice`. Besides Γ:

    - ``"line"``, one pair: X (1/2);
    - ``"square"``, two pairs, equal and orthogonal: X (1/2, 0), M (1/2, 1/2);
    - ``"rectangular"``, two pairs, orthogonal and unequal: X (1/2, 0),
      Y (0, 1/2), S (1/2, 1/2);
    - ``"hexagonal"``, two pairs, equal, at 60° or 120°: M (1/2, 0), K;
    - ``"cubic"``, three pairs, equal and orthogonal: X (1/2, 0, 0),
      M (1/2, 1/2, 0), R (1/2, 1/2, 1/2);
    - ``"other"``, anything else: none.

    X lies along the first `gfem.Periodic` in ``boundaries`` order. The
    hexagonal K is the corner of the zone, the circumcentre of Γ,
    :math:`\mathbf b_1` and whichever of :math:`\pm\mathbf b_2` makes 60° with
    :math:`\mathbf b_1`: :math:`(1/3, -1/3)` for translations at 60°,
    :math:`(1/3, 1/3)` at 120°. Γ is spelt ``"Γ"``, ``"G"`` or ``"Gamma"``
    and labelled ``"Γ"``. Other points (fcc, bcc, oblique or tetragonal
    cells) come from ``named_points``.

    The ``n_points - 1`` intervals are shared between the segments in
    proportion to their lengths in :math:`\mathbf k` by largest remainder, at
    least one per segment, so every high-symmetry point is a wave vector,
    joints are not repeated, and the spacing is even within each segment.

    Example:
        Γ-X-M-Γ of a square lattice, for `matplotlib` ticks::

            path = gfem.wave_vector_path(model, ["Γ", "X", "M", "Γ"], n_points=31)
            plt.xticks(*path.ticks)
            k_m = path.wave_vectors[path.index("M")]

    Args:
        model: A periodic cell: a `gfem.Model` with at least one
            `gfem.Periodic` pair, on a 3D mesh.
        points: The high-symmetry points' names, in path order, at least
            two, no name twice in a row.
        n_points: The exact number of wave vectors, at least
            ``len(points)``. The cost of a band diagram grows with it.
        named_points: Names to reduced coordinates, one number per pair,
            e.g. ``{"W": (0.5, 0.25, 0.75)}``; they add to the lattice's
            names or override them.

    Returns:
        The path: ``wave_vectors``, ``distance``, ``labels``,
        ``label_index``, ``lattice``, ``ticks`` and ``index``.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`, ``points`` is one string
            or holds a non-string, or ``n_points`` is not an integer.
        NotImplementedError: ``model`` is on a 2D mesh.
        ValueError: The model has no `gfem.Periodic` pair; a point is not
            a named point of the lattice; a ``named_points`` entry has not
            one finite number per pair; fewer than two points, or two
            consecutive points at the same wave vector; ``n_points`` below
            the number of points.
    """
    if not isinstance(model, Model):
        raise TypeError(f"{_PATH}: needs a gfem.Model, got {type(model).__name__}.")
    refuse_entry(model.mesh, _PATH)
    if not model.periodic_translations:
        raise ValueError(
            f"{_PATH}: the model has no gfem.Periodic pair, so it has no Bloch "
            "wave vector and no bands. Pair the cell's opposite surfaces with "
            "gfem.Periodic."
        )
    if isinstance(points, str):
        raise TypeError(
            f"{_PATH}: points is a sequence of names, e.g. ['Γ', 'X', 'M'], got "
            f"the string {points!r}."
        )
    if isinstance(n_points, bool):
        raise TypeError(f"{_PATH}: n_points must be an integer, got {n_points!r}.")
    try:
        n_points = operator.index(n_points)
    except TypeError:
        raise TypeError(
            f"{_PATH}: n_points must be an integer, got {n_points!r}."
        ) from None
    a = np.asarray(model.periodic_translations, dtype=float)
    b = np.asarray(model.reciprocal_lattice, dtype=float)
    lattice = _lattice(a)
    known = _named_points(lattice, b) | _given_points(named_points, len(a))
    labels = []
    for point in points:
        if not isinstance(point, str):
            raise TypeError(
                f"{_PATH}: points are names of high-symmetry points, got "
                f"{point!r}; give its reduced coordinates a name in named_points."
            )
        name = _canonical(point)
        if name not in known:
            raise ValueError(
                f"{_PATH}: {point!r} is not a named point of this lattice "
                f"({lattice}: {', '.join(known)}); give its reduced coordinates "
                f"in named_points={{{point!r}: (…)}}."
            )
        labels.append(name)
    if len(labels) < 2:
        raise ValueError(f"{_PATH}: a path needs at least 2 points, got {len(labels)}.")
    if n_points < len(labels):
        raise ValueError(
            f"{_PATH}: n_points={n_points} is fewer than the {len(labels)} "
            "points of the path: every high-symmetry point is a wave vector."
        )
    corners = np.stack([known[name] @ b for name in labels])
    lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    for i, length in enumerate(lengths):
        if length == 0:
            raise ValueError(
                f"{_PATH}: points {i} and {i + 1} ({labels[i]!r}, "
                f"{labels[i + 1]!r}) are the same wave vector, a segment of "
                "length 0. Remove one of them."
            )
    counts = _intervals(lengths, n_points - 1)
    rows = [
        corners[s] + (np.arange(m) / m)[:, None] * (corners[s + 1] - corners[s])
        for s, m in enumerate(counts)
    ]
    return WaveVectorPath(
        np.concatenate([*rows, corners[-1:]]),
        labels=tuple(labels),
        label_index=tuple(int(i) for i in np.concatenate([[0], np.cumsum(counts)])),
        lattice=lattice,
    )


def _canonical(name: str) -> str:
    """Γ for its spellings, any other name unchanged."""
    return _ALIASES.get(name, name)


def _lattice(a: np.ndarray) -> str:
    """The lattice the translations ``a`` (rows, m) form, a key of `_NAMED`."""
    if len(a) == 1:
        return "line"
    lengths = np.linalg.norm(a, axis=1)
    equal = np.ptp(lengths) <= LATTICE_RTOL * lengths.max()
    angles = [
        math.acos(np.clip(a[i] @ a[j] / (lengths[i] * lengths[j]), -1.0, 1.0))
        for i in range(len(a))
        for j in range(i + 1, len(a))
    ]

    def at(angle: float, degrees: float) -> bool:
        target = math.radians(degrees)
        return abs(angle - target) <= LATTICE_RTOL * target

    orthogonal = all(at(t, 90) for t in angles)
    if len(a) == 3:
        return "cubic" if equal and orthogonal else "other"
    if orthogonal:
        return "square" if equal else "rectangular"
    if equal and (at(angles[0], 60) or at(angles[0], 120)):
        return "hexagonal"
    return "other"


def _named_points(lattice: str, b: np.ndarray) -> dict[str, np.ndarray]:
    """Γ and the lattice's named points, reduced coordinates."""
    named = {_GAMMA: np.zeros(len(b))}
    named |= {name: np.array(c) for name, c in _NAMED[lattice].items()}
    if lattice == "hexagonal":
        # The zone corner is the circumcentre of Γ, b1 and the ±b2 at 60° to
        # b1: an equilateral triangle, so its centroid (b1 ± b2) / 3. b2 is at
        # 60° to b1 when the translations are at 120°, at 120° to b1 when they
        # are at 60°.
        sign = 1.0 if b[0] @ b[1] > 0 else -1.0
        named["K"] = np.array([1 / 3, sign / 3])
    return named


def _given_points(
    named_points: Mapping[str, Sequence[float]] | None, n_pairs: int
) -> dict[str, np.ndarray]:
    """The user's ``named_points``, checked, reduced coordinates."""
    if named_points is None:
        return {}
    if not isinstance(named_points, Mapping):
        raise TypeError(
            f"{_PATH}: named_points maps names to reduced coordinates, e.g. "
            f"{{'W': (0.5, 0.25, 0.75)}}, got {type(named_points).__name__}."
        )
    out = {}
    for name, value in named_points.items():
        if not isinstance(name, str):
            raise TypeError(f"{_PATH}: named_points keys are names, got {name!r}.")
        c = np.asarray(value, dtype=float).reshape(-1)
        if len(c) != n_pairs:
            raise ValueError(
                f"{_PATH}: named_points[{name!r}] has {len(c)} coordinates, but "
                f"the model has {n_pairs} periodic pair{'s' if n_pairs != 1 else ''}"
                ": give one reduced coordinate per pair, in boundaries order."
            )
        if not np.isfinite(c).all():
            raise ValueError(
                f"{_PATH}: named_points[{name!r}] = {tuple(c.tolist())} is not finite."
            )
        out[_canonical(name)] = c
    return out


def _intervals(lengths: np.ndarray, total: int) -> np.ndarray:
    """``total`` intervals shared by segment length, largest remainder, ≥ 1 each.

    ``total`` is at least ``len(lengths)``.
    """
    quota = total * lengths / lengths.sum()
    counts = np.maximum(1, np.floor(quota)).astype(int)
    while counts.sum() < total:
        counts[np.argmax(quota - counts)] += 1
    while counts.sum() > total:  # the segments raised to one interval
        counts[np.argmax(np.where(counts > 1, counts - quota, -np.inf))] -= 1
    return counts
