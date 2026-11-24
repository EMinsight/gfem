r"""The Bloch wave vectors and bands of a periodic cell.

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

`band_diagram` solves the cell's Bloch eigenmodes at each wave vector of a
path, independently, at one fixed target frequency, and reports the
:math:`n` lowest bands in ascending :math:`\mathrm{Re}\,\omega` with the
connections from each band to the previous wave vector's.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import operator
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _band_tracking as tracking
from gfem import _mpi
from gfem._checks import is_staging, is_traced
from gfem._dim2 import refuse_entry
from gfem._field import Field
from gfem._periodic import SPAN_RTOL
from gfem._pytree import register_fields
from gfem._text import hertz, hertz_band
from gfem.errors import (
    LostBandWarning,
    NonDifferentiableError,
    UnanchoredBandWindowWarning,
)
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Eigenmode
from gfem.results import _variable, _xarray
from gfem.solve import solve as _solve

__all__ = ["BandDiagram", "WaveVectorPath", "band_diagram", "wave_vector_path"]

_PATH = "wave_vector_path"
_GAMMA = "Γ"
_ALIASES = {"G": _GAMMA, "Gamma": _GAMMA}
#: Relative tolerance on the lengths and angles of the translations when the
#: lattice is recognised.
_LATTICE_RTOL = 1e-6
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
            repeat: Γ … Γ); ``"G"`` and ``"Gamma"`` are stored as ``"Γ"``.
        label_index: The wave-vector index of each label.
        lattice: ``"line"``, ``"square"``, ``"rectangular"``,
            ``"hexagonal"``, ``"cubic"`` or ``"other"`` (the default, also
            for wave vectors given directly).
        distance: ``(n_k,)`` cumulative path length
            :math:`\sum\lvert\Delta\mathbf k\rvert` from 0, rad/m: the
            abscissa of a band diagram (read-only).

    Raises:
        ValueError: The wave vectors are not finite numbers of shape
            ``(n_k, 3)`` with ``n_k`` ≥ 1, ``labels`` and ``label_index``
            differ in length, or ``lattice`` is not one of the names above.
        TypeError: A label is not a string.
        IndexError: A label's index is outside the path.
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
        if not all(isinstance(name, str) for name in self.labels):
            raise TypeError(f"WaveVectorPath: labels are names, got {self.labels!r}.")
        labels = tuple(_canonical(name) for name in self.labels)
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
    (lengths and angles to a relative :math:`10^{-6}`), and names its
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
    equal = np.ptp(lengths) <= _LATTICE_RTOL * lengths.max()
    angles = [
        math.acos(np.clip(a[i] @ a[j] / (lengths[i] * lengths[j]), -1.0, 1.0))
        for i in range(len(a))
        for j in range(i + 1, len(a))
    ]

    def at(angle: float, degrees: float) -> bool:
        target = math.radians(degrees)
        return abs(angle - target) <= _LATTICE_RTOL * target

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


_BAND = "band_diagram"
_AXES = ["x", "y", "z"]
#: Wave vectors a warning lists before "and n more".
_LISTED = 5


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class BandDiagram:
    r"""The bands of a periodic cell along a wave vector path.

    Built by `band_diagram`. Every per-band array has the wave vector first
    and the band second: ``(n_k, n_bands, ...)``, band 0 the lowest, bands
    in ascending :math:`\mathrm{Re}\,\omega` at each wave vector. The
    connections from one wave vector's bands to the previous one's, ``lost``
    and ``anchored`` say how far to trust that order; none of them changes
    it.

    Differentiation: ``omega`` differentiates with respect to the model's
    materials, sheet fill and node coordinates as `EigenmodeResult.omega`
    does, so `frequency` and the gap of two bands do under `jax.grad`. Take
    the extremum of a gap over a cluster's mean, or at a non-degenerate band
    edge: a gradient through one member of a degenerate cluster raises
    `gfem.DegenerateModeError`, as a single solve does. The static bands
    are a constant 0 with a zero gradient. ``v_g`` and the participations
    follow `EigenmodeResult`: a gradient through ``v_g`` raises
    `gfem.NonDifferentiableError`.

    Attributes:
        path: The wave vectors, `WaveVectorPath` (a raw array is wrapped).
        target: The fixed target frequency of every solve, Hz.
        omega: Complex angular frequencies, rad/s, ``(n_k, n_bands)``;
            exactly 0 for a static band.
        error: Backward error of each mode, ``(n_k, n_bands)``; NaN for a
            static band.
        v_g: The band group velocity :math:`\partial\omega/\partial\mathbf k`,
            m/s, ``(n_k, n_bands, 3)`` complex, as `EigenmodeResult.v_g`;
            NaN for a static band, a group of bands touching with different
            slopes, and a group cut by ``n_bands``. The group index is
            :math:`c / |v_g|`.
        cluster: Integer degeneracy label of each band, ``(n_k, n_bands)``,
            as `EigenmodeResult.cluster`, counted from 0 in band order; the
            static bands of a Γ share one label.
        connection: ``(n_k, n_bands)`` the band at the previous wave vector
            that this band continues; -1 for none (a mode entering the
            window, and every band of the first wave vector).
        connection_by_overlap: ``(n_k, n_bands)`` True where the field
            overlap decided the connection; False where frequency order
            did (a band of a degenerate or near-degenerate group, whose
            overlap is taken with the group's span), or where there is
            none. A static band connects to the neighbouring band that goes
            to zero by frequency order.
        static: ``(n_k, n_bands)`` a band filled in at :math:`\omega = 0`
            at a wave vector equivalent to Γ, where the solve drops the
            cell's static fields: the bands that go to zero there.
        lost: ``(n_k, n_bands)`` a band inside the window with no partner at
            a neighbouring wave vector: the path is too coarse there, or the
            solve missed a mode (`gfem.LostBandWarning`); and every band of a
            Γ whose neighbours disagree on how many bands go to zero there.
        anchored: ``(n_k,)`` the band window is anchored: the target is below
            half the highest frequency found there, so the bands found are
            the lowest ones and band ``n`` is the ``n``-th lowest
            (`gfem.UnanchoredBandWindowWarning`).
        q_port: Coupling Q of each port by name, ``(n_k, n_bands)``, as
            `EigenmodeResult.q_port`; NaN for a static band.
        participation: As `EigenmodeResult.participation`, ``(n_k, n_bands)``
            by port name.
        bulk_participation: As `EigenmodeResult.bulk_participation`,
            ``(n_k, n_bands)`` by region name.
        sheet_participation: As `EigenmodeResult.sheet_participation`,
            ``(n_k, n_bands)`` by sheet name.
        principal_participation: As `EigenmodeResult.principal_participation`,
            ``(n_k, n_bands, 3)`` by region name.
        e_fields: The mode fields of the wave vectors ``keep_fields`` named:
            for each wave vector a tuple with one `Field` per band
            (``None`` for a static band), ``()`` where not kept. Read them
            with `e_field`.
    """

    path: WaveVectorPath
    target: float
    omega: jax.Array
    error: jax.Array
    v_g: jax.Array
    cluster: jax.Array
    connection: jax.Array
    connection_by_overlap: jax.Array
    static: jax.Array
    lost: jax.Array
    anchored: jax.Array
    q_port: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    bulk_participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    sheet_participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    principal_participation: dict[str, jax.Array] = dataclasses.field(
        default_factory=dict
    )
    e_fields: tuple[tuple[Field | None, ...], ...] = ()

    @property
    def frequency(self) -> jax.Array:
        r"""Band frequencies :math:`\mathrm{Re}\,\omega / (2\pi)`, Hz."""
        return jnp.real(self.omega) / (2 * math.pi)

    @property
    def q(self) -> jax.Array:
        r"""Quality factors :math:`|\omega| / (2 |\mathrm{Im}\,\omega|)`.

        ``inf`` for a lossless band, NaN for a static one.
        """
        return jnp.abs(self.omega) / (2 * jnp.abs(jnp.imag(self.omega)))

    @property
    def n_bands(self) -> int:
        """The number of bands per wave vector."""
        return int(np.shape(self.omega)[1])

    @property
    def wave_vectors(self) -> np.ndarray:
        """The path's wave vectors, rad/m, ``(n_k, 3)``."""
        return self.path.wave_vectors

    @property
    def distance(self) -> np.ndarray:
        """The path's cumulative distance, rad/m, ``(n_k,)``: the abscissa."""
        return self.path.distance

    def e_field(self, index: int, band: int) -> Field:
        """The electric field of band ``band`` at wave vector ``index``.

        Band 0 is the lowest. Needs ``keep_fields`` to have named ``index``.

        Raises:
            IndexError: No such wave vector or band.
            ValueError: The fields were not kept at this wave vector, or the
                band is a static one, which has no field.
        """
        n_k = len(self.path)
        if not -n_k <= index < n_k:
            raise IndexError(
                f"Wave vector {index} does not exist: the path has {n_k}, "
                "numbered from 0."
            )
        if not -self.n_bands <= band < self.n_bands:
            raise IndexError(
                f"Band {band} does not exist: the diagram has {self.n_bands} "
                "band(s), numbered from 0 (the lowest)."
            )
        index, band = index % n_k, band % self.n_bands
        if not self.e_fields[index]:
            raise ValueError(
                f"fields were not kept at wave vector {index}: pass "
                f"keep_fields=[{index}] (or True)."
            )
        field = self.e_fields[index][band]
        if field is None:
            where = (
                f" ({self.path.labels[self.path.label_index.index(index)]})"
                if (index in self.path.label_index)
                else ""
            )
            raise ValueError(
                f"band {band} at wave vector {index}{where} is a static band: "
                "it has no field."
            )
        return field

    def to_xarray(self) -> Any:
        """The diagram as an `xarray.Dataset` over ``("wave_vector", "band")``.

        Coordinates: ``wave_vector`` and ``band`` (numbered from 0, the
        lowest), ``axis`` (x, y, z), ``distance`` (rad/m), ``label`` (the
        high-symmetry point's name, ``""`` elsewhere) and ``k`` (rad/m).
        Data variables: ``omega`` (rad/s), ``frequency`` (Hz), ``q``,
        ``error``, ``cluster``, ``v_g`` (m/s, over the ``axis`` too),
        ``connection``, ``connection_by_overlap``, ``static``, ``lost``,
        ``anchored`` (over ``wave_vector``), and the participations over
        ``port``, ``region``, ``anisotropic_region`` and ``sheet``, as
        `EigenmodeResult.to_xarray` (NaN where a port has no value). The
        attributes are ``target`` (Hz) and ``lattice``. The fields are not
        exported. Needs the ``xarray`` extra: ``pip install "gfem[xarray]"``.
        ``ds.frequency.plot.line(x="distance", hue="band")`` draws the bands.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The diagram holds JAX tracers.
        """
        xr = _xarray(self)
        n_k = len(self.path)
        labels = np.full(n_k, "", dtype=object)
        for name, i in zip(self.path.labels, self.path.label_index, strict=True):
            labels[i] = name
        d = ("wave_vector", "band")
        coords: dict[str, Any] = {
            "wave_vector": np.arange(n_k),
            "band": np.arange(self.n_bands),
            "axis": _AXES,
            "distance": _variable(("wave_vector",), self.path.distance, "rad/m"),
            "label": _variable(("wave_vector",), labels.astype(str)),
            "k": _variable(("wave_vector", "axis"), self.path.wave_vectors, "rad/m"),
        }
        data = {
            "omega": _variable(d, self.omega, "rad/s"),
            "frequency": _variable(d, self.frequency, "Hz"),
            "q": _variable(d, self.q),
            "error": _variable(d, self.error),
            "cluster": _variable(d, self.cluster),
            "v_g": _variable((*d, "axis"), self.v_g, "m/s"),
            "connection": _variable(d, self.connection),
            "connection_by_overlap": _variable(d, self.connection_by_overlap),
            "static": _variable(d, self.static),
            "lost": _variable(d, self.lost),
            "anchored": _variable(("wave_vector",), self.anchored),
        }
        ports = list(dict.fromkeys([*self.q_port, *self.participation]))
        if ports:
            coords["port"] = ports
            for name, values in (
                ("q_port", self.q_port),
                ("participation", self.participation),
            ):
                table = np.full((n_k, self.n_bands, len(ports)), np.nan)
                for p, port in enumerate(ports):
                    if port in values:
                        table[:, :, p] = np.asarray(values[port])
                data[name] = _variable((*d, "port"), table)
        for name, dim, values in (
            ("bulk_participation", "region", self.bulk_participation),
            ("sheet_participation", "sheet", self.sheet_participation),
        ):
            if values:
                coords[dim] = list(values)
                data[name] = _variable(
                    (*d, dim), np.stack([np.asarray(v) for v in values.values()], 2)
                )
        if self.principal_participation:
            coords["anisotropic_region"] = list(self.principal_participation)
            data["principal_participation"] = _variable(
                (*d, "anisotropic_region", "axis"),
                np.stack(
                    [np.asarray(v) for v in self.principal_participation.values()], 2
                ),
            )
        return xr.Dataset(
            data,
            coords=coords,
            attrs={"target": self.target, "lattice": self.path.lattice},
        )

    def __repr__(self) -> str:
        n_k = len(self.path)
        head = f"{n_k} wave vector{'s' if n_k != 1 else ''}, {self.n_bands} bands"
        if is_traced(self.omega):
            return f"BandDiagram({head})"
        f = np.asarray(self.frequency)
        f = f[np.isfinite(f)]
        parts = ["-".join(self.path.labels)] if self.path.labels else []
        parts.append(head)
        if f.size:
            parts.append(hertz_band(float(f.min()), float(f.max()), 3))
        return f"BandDiagram({', '.join(parts)})"


register_fields(
    BandDiagram,
    (
        "omega",
        "error",
        "v_g",
        "cluster",
        "connection",
        "connection_by_overlap",
        "static",
        "lost",
        "anchored",
        "q_port",
        "participation",
        "bulk_participation",
        "sheet_participation",
        "principal_participation",
        "e_fields",
    ),
)


@dataclasses.dataclass
class _Found:
    """One wave vector's solve, its modes in ascending Re ω."""

    omega: jax.Array
    error: jax.Array
    v_g: jax.Array
    cluster: np.ndarray
    w: np.ndarray
    ratios: dict[str, dict[str, jax.Array]]
    fields: tuple[Field, ...]


def band_diagram(
    model: Model,
    path: WaveVectorPath | Any,
    *,
    target: float,
    n_bands: int,
    keep_fields: bool | Sequence[int] = False,
    options: SolverOptions | None = None,
    comm: Any = None,
    cache: str | os.PathLike | None = None,
) -> BandDiagram:
    r"""The lowest bands of a periodic cell along a path of Bloch wave vectors.

    Each wave vector is one independent
    ``gfem.solve(model, Eigenmode(target, n_modes=n_bands, wave_vector=k))``,
    all at the same ``target``: no start vector, no target continued from the
    previous wave vector. The sweep then orders each wave vector's modes by
    :math:`\mathrm{Re}\,\omega` (band 0 the lowest), fills in the static bands
    at a wave vector equivalent to Γ, and connects each band to the
    previous wave vector's by field overlap.

    Choose ``target`` below half the lowest frequency of band ``n_bands - 1``
    along the path: then the ``n_bands`` modes nearest the target are the
    lowest ones at every wave vector, and band ``n`` is the ``n``-th lowest.
    A starting point is :math:`c\,|\mathbf b|_\mathrm{min} / (4\, n_\mathrm{max})`
    for the shortest reciprocal lattice vector and the highest refractive
    index. At a wave vector where the test fails, ``anchored`` is False and
    `gfem.UnanchoredBandWindowWarning` fires once; a window around a higher
    target is a legitimate look at higher bands. With ``n_bands=1`` on a
    slab (at most 2 in 3D) the test can fail falsely at Γ, where every
    requested band is static.

    At Γ the solve drops the cell's static fields, so the bands that go to
    zero frequency there are missing. They are filled in at
    :math:`\omega = 0`, as many as the neighbouring wave vectors have
    modes below the lowest one found at Γ, flagged ``static`` and with a
    zero gradient. A band inside the window with no partner at a
    neighbouring wave vector is flagged ``lost`` and warns once with
    `gfem.LostBandWarning`: the bands keep frequency order, but beyond
    that wave vector it may put one band's values in another's place.

    Runs under `jax.grad` (eagerly, one solve and one backward pass per wave
    vector): the bands are ordered and connected from the modes' values, and
    the frequencies carry the gradient. It raises
    `gfem.NonDifferentiableError` inside `jax.jit` or `jax.vmap`: loop over
    designs in Python. The fields stay on the host: ``keep_fields`` names
    the wave vectors whose fields the result keeps.

    Example:
        The TM bands of a square lattice of rods and the gap between the
        first two, with its gradient with respect to the rod permittivity::

            path = gfem.wave_vector_path(cell(8.9), ["Γ", "X", "M", "Γ"], n_points=31)


            def gap(eps_rod):
                bd = gfem.band_diagram(cell(eps_rod), path, target=1e9, n_bands=4)
                return bd.frequency[:, 1].min() - bd.frequency[:, 0].max()


            dgap = jax.grad(gap)(8.9)

    Args:
        model: A periodic cell: a `gfem.Model` with at least one
            `gfem.Periodic` pair, on a 3D mesh.
        path: A `WaveVectorPath`, or wave vectors ``(n_k, 3)`` in rad/m in
            the span of the periodic translations.
        target: The target frequency of every solve, Hz, a finite number
            above 0, fixed (not computed inside `jax.grad`).
        n_bands: The number of bands per wave vector, at least 1.
        keep_fields: ``False`` keeps no field, ``True`` those of every wave
            vector, a sequence of wave-vector indices those wave vectors'.
        options: Solver options of every solve.
        comm: An MPI communicator: a Bloch eigen solve runs on one process,
            so give ``MPI.COMM_SELF`` to run independent copies.
        cache: A solve cache directory, passed to every solve.

    Returns:
        The `BandDiagram`.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`.
        NotImplementedError: ``model`` is on a 2D mesh.
        ValueError: The model has no `gfem.Periodic` pair; ``path`` is not
            wave vectors of shape ``(n_k, 3)``; a wave vector lies outside
            the span of the translations; ``target`` or ``n_bands`` is out of
            range.
        IndexError: ``keep_fields`` names a wave vector outside the path.
        NonDifferentiableError: Inside `jax.jit` or `jax.vmap`, or the wave
            vectors or the target are computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`.
        DistributedUnsupportedError: ``comm`` has more than one rank.
    """
    if not isinstance(model, Model):
        raise TypeError(f"{_BAND}: needs a gfem.Model, got {type(model).__name__}.")
    refuse_entry(model.mesh, _BAND)
    translations = np.asarray(model.periodic_translations, dtype=float)
    if not len(translations):
        raise ValueError(
            f"{_BAND}: the model has no gfem.Periodic pair, so it has no Bloch "
            "wave vector and no bands. Pair the cell's opposite surfaces with "
            "gfem.Periodic."
        )
    if is_traced(path) or is_traced(target):
        raise NonDifferentiableError(
            f"{_BAND}: the wave vectors and the target must be fixed numbers, "
            "not values computed inside jax.grad, jax.jit or jax.vmap: gfem "
            "does not differentiate with respect to them. The band group "
            "velocity ∂ω/∂k is v_g."
        )
    wave_path = _wave_path(path)
    _check_span(wave_path.wave_vectors, translations)
    target = _check_target(target)
    n_bands = _check_n_bands(n_bands)
    n_k = len(wave_path)
    kept = _kept(keep_fields, n_k)
    fixed = jax.lax.stop_gradient(model)
    if is_staging() or is_traced(fixed):
        raise NonDifferentiableError(
            f"{_BAND} runs under jax.grad, but not inside jax.jit or jax.vmap: "
            "it orders and connects the bands from each wave vector's modes as "
            "numbers. Call it outside jax.jit, and loop over designs in Python "
            "instead of jax.vmap."
        )
    if _mpi.resolve_comm(comm) is not None:
        _mpi.refuse_feature("bloch")
    found: list[_Found] = []
    overlap: list[np.ndarray] = []
    mass = None
    previous = None
    for i, k in enumerate(wave_path.wave_vectors):
        problem = Eigenmode(
            target,
            n_modes=n_bands,
            wave_vector=(float(k[0]), float(k[1]), float(k[2])),
        )
        result = _solve_wave_vector(model, problem, options, comm, cache)
        one = _order(result)
        found.append(one)
        if mass is None:
            mass = tracking.mass_matrix(fixed, result.e_fields[0].order)
        x = np.stack(
            [
                np.asarray(jax.lax.stop_gradient(f.coefficients), dtype=complex)
                for f in one.fields
            ],
            axis=1,
        )
        if previous is not None:
            overlap.append(
                tracking.overlaps(mass, previous, x, found[-2].w.real, one.w.real)
            )
        previous = x
        if i not in kept:
            one.fields = ()
    w = np.stack([f.w.real for f in found])
    omega_target = 2 * math.pi * target
    anchored = np.asarray(omega_target < w.max(axis=1) / 2)
    gamma = np.array(
        [tracking.gamma_equivalent(k, translations) for k in wave_path.wave_vectors]
    )
    tracked = tracking.track(overlap, w, anchored, gamma)
    stacklevel = 2
    if not anchored.all():
        warnings.warn(
            _unanchored_message(wave_path, target, anchored, w),
            UnanchoredBandWindowWarning,
            stacklevel=stacklevel,
        )
    if tracked.lost.any():
        warnings.warn(
            _lost_message(tracked, w),
            LostBandWarning,
            stacklevel=stacklevel,
        )
    return _assemble(found, tracked, anchored, path=wave_path, target=target)


def _solve_wave_vector(
    model: Model,
    problem: Eigenmode,
    options: SolverOptions | None,
    comm: Any,
    cache: str | os.PathLike | None,
):
    """One wave vector's Bloch eigen solve: `gfem.solve`."""
    return _solve(model, problem, options=options, comm=comm, cache=cache)


def _order(result) -> _Found:
    """``result``'s modes in ascending Re ω, the host values apart."""
    w = np.asarray(jax.lax.stop_gradient(result.omega))
    order = np.argsort(w.real, kind="stable")
    v_g = result.v_g
    if v_g is None:
        v_g = jnp.full((len(w), 3), complex(np.nan, np.nan))
    ratios = {
        name: {key: value[order] for key, value in getattr(result, name).items()}
        for name in (
            "q_port",
            "participation",
            "bulk_participation",
            "sheet_participation",
            "principal_participation",
        )
    }
    return _Found(
        omega=result.omega[order],
        error=result.error[order],
        v_g=v_g[order],
        cluster=np.asarray(result.cluster)[order],
        w=w[order],
        ratios=ratios,
        fields=tuple(result.e_fields[int(i)] for i in order),
    )


def _wave_path(path: Any) -> WaveVectorPath:
    """``path`` as a `WaveVectorPath`."""
    if isinstance(path, WaveVectorPath):
        return path
    shape = "unknown"
    try:
        k = np.asarray(path, dtype=float)
        shape = str(k.shape)
        if k.ndim == 2 and k.shape[1] == 3 and len(k) and np.isfinite(k).all():
            return WaveVectorPath(k)
    except (TypeError, ValueError):
        pass
    raise ValueError(
        f"{_BAND}: the path must be a gfem.WaveVectorPath or wave vectors of "
        f"shape (n_k, 3) in rad/m with n_k ≥ 1, got shape {shape}."
    )


def _check_span(wave_vectors: np.ndarray, translations: np.ndarray) -> None:
    """Every wave vector lies in the span of the translations."""
    basis, _ = np.linalg.qr(translations.T)
    for i, k in enumerate(wave_vectors):
        projected = basis @ (basis.T @ k)
        off = float(np.linalg.norm(k - projected))
        if off > SPAN_RTOL * float(np.linalg.norm(k)):
            raise ValueError(
                f"{_BAND}: wave vector {i} {_vec(k)} rad/m has a component of "
                f"{off:.6g} rad/m outside the span of the periodic translations "
                f"{', '.join(_vec(a) for a in translations)} m, which no pair's "
                "phase reads. Give it in that span, e.g. "
                f"{_vec(projected)} rad/m (gfem.wave_vector_path builds such "
                "paths)."
            )


def _vec(x: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(v):.6g}" for v in x) + ")"


def _check_target(target: Any) -> float:
    try:
        value = float(target)
    except (TypeError, ValueError):
        value = math.nan
    if isinstance(target, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{_BAND}: target must be a finite frequency > 0 Hz, got {target!r}."
        )
    return value


def _check_n_bands(n_bands: Any) -> int:
    value = 0
    if not isinstance(n_bands, bool):
        with contextlib.suppress(TypeError):
            value = operator.index(n_bands)
    if value < 1:
        raise ValueError(f"{_BAND}: n_bands must be an integer ≥ 1, got {n_bands!r}.")
    return value


def _kept(keep_fields: bool | Sequence[int], n_k: int) -> set[int]:
    """The wave-vector indices whose fields the diagram keeps."""
    if isinstance(keep_fields, (bool, np.bool_)):
        return set(range(n_k)) if keep_fields else set()
    out = set()
    for i in keep_fields:
        if not 0 <= operator.index(i) < n_k:
            raise IndexError(
                f"{_BAND}: keep_fields names wave vector {i}, but the path has "
                f"{n_k}, numbered from 0."
            )
        out.add(operator.index(i))
    return out


def _indices(indices: Sequence[int]) -> str:
    shown = ", ".join(str(i) for i in indices[:_LISTED])
    more = len(indices) - _LISTED
    return shown + (f" and {more} more" if more > 0 else "")


def _unanchored_message(
    path: WaveVectorPath, target: float, anchored: np.ndarray, w: np.ndarray
) -> str:
    bad = [int(i) for i in np.flatnonzero(~anchored)]
    names = {i: n for n, i in zip(path.labels, path.label_index, strict=True)}
    where = ""
    if all(i in names for i in bad[:_LISTED]):
        where = f" ({', '.join(names[i] for i in bad[:_LISTED])})"
    highest = hertz(float(w[bad[0]].max()) / (2 * math.pi))
    return (
        f"{_BAND}: at wave vectors {_indices(bad)}{where} the target "
        f"{hertz(target)} is not below half the highest band found there "
        f"({highest}), so the bands found may not be the lowest and their "
        "order there is not the band number. Lower the target below half the "
        "lowest frequency of the highest requested band along the path. "
        "BandDiagram.anchored marks them."
    )


def _lost_message(tracked: tracking.Tracking, w: np.ndarray) -> str:
    s = tracked.static
    items = [
        f"band {band} at wave vector {k} "
        f"({hertz(float(w[k, band - s[k]]) / (2 * math.pi))}) → {other}"
        for k, band, other in tracked.lost_at[:_LISTED]
    ]
    more = len(tracked.lost_at) - _LISTED
    tail = f", and {more} more" if more > 0 else ""
    m = len(tracked.lost_at)
    return (
        f"{_BAND}: {m} band{'s' if m != 1 else ''} found inside the window "
        f"{'have' if m != 1 else 'has'} no partner at a neighbouring wave "
        f"vector: {', '.join(items)}{tail}. The path is too coarse there, or "
        "the solve missed a mode; frequency order beyond that wave vector may "
        "put one band's values in another. Use more n_points, a target nearer "
        "the bands, or more n_bands. BandDiagram.lost marks them."
    )


def _padded(rows: jax.Array, s: int, fill: Any) -> jax.Array:
    """``rows`` (one per mode) with ``s`` bands of ``fill`` below, as many rows."""
    if not s:
        return rows
    head = jnp.full((s, *rows.shape[1:]), fill, dtype=rows.dtype)
    return jnp.concatenate([head, rows[: rows.shape[0] - s]])


def _labels(cluster: np.ndarray, s: int) -> np.ndarray:
    """Cluster labels counted from 0 in band order; the static bands share one."""
    kept = cluster[: len(cluster) - s]
    seen: dict[int, int] = {}
    out = [seen.setdefault(int(c), len(seen) + (1 if s else 0)) for c in kept]
    return np.array([0] * s + out, dtype=int)


def _assemble(
    found: list[_Found],
    tracked: tracking.Tracking,
    anchored: np.ndarray,
    *,
    path: WaveVectorPath,
    target: float,
) -> BandDiagram:
    """Stack the wave vectors' bands, the static ones filled in."""
    n = len(found[0].w)
    s = [int(k) for k in tracked.static]
    nan = math.nan

    def stack(rows: list[jax.Array], fill: Any) -> jax.Array:
        return jnp.stack([_padded(r, k, fill) for r, k in zip(rows, s, strict=True)])

    omega = stack([f.omega for f in found], complex(0.0))
    error = stack([f.error for f in found], nan)
    v_g = stack([f.v_g for f in found], complex(nan, nan))
    cluster = np.stack([_labels(f.cluster, k) for f, k in zip(found, s, strict=True)])
    static = np.arange(n)[None, :] < np.array(s)[:, None]
    ratios = {
        name: {key: stack([f.ratios[name][key] for f in found], nan) for key in group}
        for name, group in found[0].ratios.items()
    }
    fields = tuple(
        ((None,) * k + f.fields[: n - k] if f.fields else ())
        for f, k in zip(found, s, strict=True)
    )
    return BandDiagram(
        path=path,
        target=target,
        omega=omega,
        error=error,
        v_g=v_g,
        cluster=jnp.asarray(cluster),
        connection=jnp.asarray(tracked.connection),
        connection_by_overlap=jnp.asarray(tracked.by_overlap),
        static=jnp.asarray(static),
        lost=jnp.asarray(tracked.lost),
        anchored=jnp.asarray(anchored),
        e_fields=fields,
        **ratios,
    )
