r"""Affine pieces assembled once, summed per operator: the piece sum.

An operator :math:`A = \sum_k \theta_k A_k` whose pieces :math:`A_k` do not
change (a driven sweep's frequencies) need not be assembled again for each set
of weights :math:`\theta_k`: `AssembledPieces` assembles each piece once, with
:math:`\theta_k = 1`, and `PieceSum` writes :math:`\sum_k \theta_k A_k` into
a matrix of the full sparsity pattern.
`PieceSum` reads its pieces through the `Pieces` protocol, of which
`AssembledPieces` is one adapter.
That skips NGSolve's allocation of a new matrix and graph, the part of an
assembly that does not scale with threads.

The pieces with the full pattern (the volume terms) are summed as NGSolve
vectors over the matrix values; each other piece (a surface term, whose
graph couples its surface DOFs only) keeps its compact values and their
positions in the full pattern, and is added there.

Under ranks each of this runs on a rank's local matrix. No value crosses ranks: a
local matrix of a rank without elements is empty, and a rank's surface piece may
share the full pattern where another's does not. Two things are agreed, because
they decide what every rank does next: the piece the pattern comes from, by summed
nonzeros, and whether some piece lies outside the pattern.

On one process a piece may also have a fixed part (`FixedPart`, ADR 0082): the
piece is then its varying part, assembled at every build, plus the fixed part,
whose values in the full pattern outlive the pieces in a `KeptParts` store,
behind a digest of what it reads. The fixed part is kept the second time it is
seen at one digest, as the whole piece minus the varying part; the first time
the whole piece is assembled as without one, so a store that never sees a
digest twice (node moves, a new discretization at every call) costs nothing.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import ngsolve as ngs
import numpy as np
from numpy.typing import ArrayLike

from gfem import _mpi
from gfem._linalg.base import host_bytes, local_matrix, same_pattern

try:  # private to scipy: the piece sum falls back to numpy without it
    from scipy.sparse._sparsetools import (  # ty: ignore[unresolved-import]
        csc_matvec as _csc_matvec,
    )
except ImportError:  # pragma: no cover - scipy's layout
    _csc_matvec = None

Integrand = Callable[[Any, Any], Any]


@dataclasses.dataclass(frozen=True, eq=False)
class _Kept:
    """A fixed part's values, kept in a `KeptParts` store.

    Attributes:
        digest: The digest the values were kept at.
        pattern: A matrix of the whole piece's (full) pattern; its own values
            are not read.
        values: The fixed part's values in ``pattern``.
        varying: The varying part's nonzeros and their positions in ``pattern``.
    """

    digest: str
    pattern: Any
    values: ngs.BaseVector
    varying: tuple[int, np.ndarray]

    def shared(self) -> dict[int, int]:
        """Bytes of the pattern, the values and the varying positions, by identity.

        Pieces and handles may hold the same objects (`AssembledPieces.shared`).
        """
        return {
            id(self.pattern): host_bytes(self.pattern),
            id(self.values): int(self.values.FV().NumPy().nbytes),
            id(self.varying[1]): int(self.varying[1].nbytes),
        }


class KeptParts:
    """The fixed parts kept on one space, by piece name, and the digests seen.

    Holds one kept part per name: a new digest replaces it.
    """

    def __init__(self) -> None:
        self.kept: dict[str, _Kept] = {}
        self.seen: dict[str, str] = {}

    def shared(self) -> dict[int, int]:
        """The kept parts' bytes, by the identity of each object held."""
        held: dict[int, int] = {}
        for kept in self.kept.values():
            held.update(kept.shared())
        return held

    def clear(self) -> None:
        """Drop every kept part and every digest seen (memory pressure).

        Nothing kept is changed: pieces that hold a part keep reading it. The
        next build assembles plainly and keeps at the second sighting.
        """
        self.kept = {}
        self.seen = {}


@dataclasses.dataclass(frozen=True, eq=False)
class FixedPart:
    """The part of a piece that `AssembledPieces` keeps across builds.

    The piece's integrand in `AssembledPieces` is its varying part; this is
    the rest, known by a digest of what it reads (its coefficient on its
    elements): a store entry at the same digest holds its values.

    Attributes:
        store: Where the kept values live (one store per space).
        name: The piece's name, its key in ``store``.
        digest: The digest of what the fixed part reads.
        whole: The whole piece's integrand.
    """

    store: KeptParts
    name: str
    digest: str
    whole: Integrand

    def found(self) -> _Kept | None:
        """The kept values at this digest, or ``None``."""
        kept = self.store.kept.get(self.name)
        return kept if kept is not None and kept.digest == self.digest else None

    def first_seen(self) -> bool:
        """Whether this digest is new to the store; marks it seen."""
        new = self.store.seen.get(self.name) != self.digest
        self.store.seen[self.name] = self.digest
        return new

    def keep(
        self,
        space: ngs.FESpace,
        symmetric: bool,
        varying: Any,
        pattern: Any = None,
    ) -> _Kept | None:
        """Assemble the whole piece and keep it minus the ``varying`` part.

        The values share ``pattern``'s matrix when the whole piece has its
        pattern. ``None`` (nothing kept) when the varying part lies outside
        the whole piece's pattern.
        """
        whole = _assemble(space, symmetric, self.whole)
        if pattern is None or not same_pattern(whole, pattern):
            pattern = whole
        _, cols, rowptr = pattern.CSR()
        data, c, r = varying.CSR()
        at = _positions(rowptr, cols, r, c)
        if at is None:
            return None
        # the volume pieces' varying parts share their elements, so mostly
        # their positions: one array is held for them all
        for other in self.store.kept.values():
            if np.array_equal(other.varying[1], at):
                at = other.varying[1]
                break
        values = ngs.BaseVector(whole.nze, True)
        out = values.FV().NumPy()
        out[:] = whole.AsVector().FV().NumPy()
        out[at] -= np.asarray(data)
        kept = _Kept(self.digest, pattern, values, (varying.nze, at))
        self.store.kept[self.name] = kept
        return kept


def _assemble(space: ngs.FESpace, symmetric: bool, integrand: Integrand) -> Any:
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=symmetric)
    form += integrand(u, v)
    form.Assemble()
    return form.mat


class AssembledPieces:
    r"""The pieces :math:`A_k` of a complex space, assembled on first use.

    Each piece is assembled with :math:`\theta_k = 1`.

    Holds the full-pattern matrix of the piece with the most nonzeros (its
    pattern is the operator's), the values of the other full-pattern pieces,
    and each remaining piece's compact values with their positions in that
    pattern. A piece with a `FixedPart` is summed as its varying part and its
    kept fixed part once kept (on one process only). `free` drops them; a
    later use assembles them again (the fixed parts stay in their store).
    """

    def __init__(
        self,
        space: ngs.FESpace,
        integrands: Sequence[Integrand],
        *,
        symmetric: bool = True,
        comm: Any = None,
        fixed: Sequence[FixedPart | None] | None = None,
    ) -> None:
        """Hold ``space`` and each piece's integrand ``(trial, test) -> integrals``.

        ``symmetric`` is the forms' flag: False for a general operator (a
        Bloch phase); the full matrix is stored either way. ``comm`` is the
        resolved communicator of a distributed ``space``. ``fixed`` holds,
        per piece, its `FixedPart` (its integrand is then the varying part's)
        or ``None``; only without ``comm``.
        """
        self._comm = comm
        self._space = space
        self._symmetric = symmetric
        self._integrands = tuple(integrands)
        self._fixed = (None,) * len(self._integrands) if fixed is None else tuple(fixed)
        if len(self._fixed) != len(self._integrands) or (
            comm is not None and any(self._fixed)
        ):
            raise ValueError("One fixed part or None per piece, on one process only.")
        self._built = False
        self._template: ngs.BaseMatrix | None = None
        # the template's own values among the parts, and the split pieces
        self._template_values: Any = None
        self._split: frozenset[int] = frozenset()
        #: (values, positions or None, piece index, shared): shared values are
        #: a kept fixed part's or the template's own, counted in `shared`
        self._parts: list[tuple[Any, np.ndarray | None, int, bool]] = []
        # new at each assembly: freed pieces assemble anew
        self._generation: object | None = None
        #: Serializes the first use (assembled once) and `free`.
        self._lock = threading.Lock()

    @property
    def pattern(self) -> ngs.BaseMatrix | None:
        """The full-pattern matrix every sum is shaped as, or ``None``.

        ``None`` when a piece couples DOFs that the full pattern does not:
        the pieces cannot be summed, the operator must be assembled.
        """
        return self._assembled()[0]

    @property
    def generation(self) -> object:
        """Who holds these values: a new object at each assembly (after `free`)."""
        with self._lock:
            if not self._built:
                self._build()
            return self._generation

    def matrix(self, weights: Sequence[complex]) -> ngs.BaseMatrix | None:
        r"""A new matrix of `pattern` holding :math:`\sum_k \theta_k A_k`.

        ``None`` without a pattern.
        """
        template = self.pattern
        if template is None:
            return None
        matrix = template.CreateMatrix()
        self.sum_into(weights, matrix)
        return matrix

    def sum_into(self, weights: Sequence[complex], matrix: ngs.BaseMatrix) -> None:
        r"""Overwrite the values of ``matrix`` (of `pattern`).

        They become :math:`\sum_k \theta_k A_k`.
        """
        template, parts = self._assembled()
        if template is None:  # pragma: no cover - callers check the pattern
            raise ValueError("These pieces have no common sparsity pattern.")
        weights = [complex(w) for w in weights]
        if len(weights) != len(self._integrands):
            raise ValueError(f"{len(self._integrands)} weights, got {len(weights)}.")
        target = local_matrix(matrix, self._comm).AsVector()
        full = [(weights[k], v) for v, at, k, _ in parts if at is None]
        (w0, v0), *rest = full
        target.data = w0 * v0
        for w, values in rest:
            target.data += w * values
        scattered = [(weights[k], v, at) for v, at, k, _ in parts if at is not None]
        out = target.FV().NumPy()
        if not _scatter_compiled(out, scattered):
            _scatter_numpy(out, scattered)

    @property
    def nbytes(self) -> int:
        """Host memory of the assembled pieces, in bytes (0 before first use).

        The objects others may hold too are in `shared`, not here.
        """
        with self._lock:
            parts = self._parts
        held = 0
        for values, at, k, shared in parts:
            if shared:
                continue
            if at is None:
                held += int(values.FV().NumPy().nbytes)
                continue
            held += int(values.nbytes)
            # a split piece's positions are its kept part's, in `shared`
            if k not in self._split:
                held += int(at.nbytes)
        return held

    def shared(self) -> dict[int, int]:
        """Bytes of what a handle or a `KeptParts` store may hold too, by identity.

        The template (a handle's pieces' pattern, or a kept part's), the kept
        fixed parts' values and their varying parts' positions.
        """
        with self._lock:
            template, parts = self._template, self._parts
        if template is None:
            return {}
        held = {id(template): host_bytes(local_matrix(template, self._comm))}
        for values, at, k, shared in parts:
            if shared and at is None and values is not self._template_values:
                held[id(values)] = int(values.FV().NumPy().nbytes)
            if at is not None and k in self._split:
                held[id(at)] = int(at.nbytes)
        return held

    def free(self) -> None:
        """Drop the assembled pieces."""
        with self._lock:
            self._built = False
            self._generation = None
            self._template = None
            self._template_values = None
            self._parts = []
            self._split = frozenset()

    def _assembled(self) -> tuple[ngs.BaseMatrix | None, list]:
        """The template and the parts, assembled once on first use."""
        with self._lock:
            if not self._built:
                self._build()
            return self._template, self._parts

    def _assemble_pieces(
        self,
    ) -> tuple[list[tuple[int, Any]], list[tuple[int, FixedPart, _Kept, Any]], Any]:
        """The pieces assembled whole, the split ones and a kept full pattern.

        Each whole piece as ``(index, matrix)``, each split one as ``(index,
        fixed part, its kept values, varying part's matrix)``; the pattern is ``None``
        without a split piece.
        """
        space, symmetric = self._space, self._symmetric
        whole: list[tuple[int, Any]] = []
        split: list[tuple[int, FixedPart, _Kept, Any]] = []
        pattern = None
        for k, (integrand, fixed) in enumerate(
            zip(self._integrands, self._fixed, strict=True)
        ):
            kept = None if fixed is None else fixed.found()
            if fixed is not None and kept is None and fixed.first_seen():
                whole.append((k, _assemble(space, symmetric, fixed.whole)))
                continue
            if fixed is None:
                whole.append((k, _assemble(space, symmetric, integrand)))
                continue
            varying = _assemble(space, symmetric, integrand)
            if kept is None or kept.varying[0] != varying.nze:
                kept = fixed.keep(space, symmetric, varying, pattern)
            if kept is None:  # pragma: no cover - a varying part outside the whole
                whole.append((k, _assemble(space, symmetric, fixed.whole)))
                continue
            if pattern is None:
                pattern = kept.pattern
            split.append((k, fixed, kept, varying))
        return whole, split, pattern

    def _build(self) -> None:
        """Assemble each piece and place its values in the full pattern."""
        space, symmetric = self._space, self._symmetric
        whole, split, pattern = self._assemble_pieces()
        comm = self._comm
        local = [(k, local_matrix(m, comm)) for k, m in whole]
        if pattern is None:
            nonzeros = _mpi.reduce_bcast(comm, np.array([m.nze for _, m in local]))
            first = int(np.argmax(nonzeros))
            # a distributed matrix's own, its local matrix gives the pattern
            matrix, template = whole[first][1], local[first][1]
        else:
            matrix = template = pattern
        _, cols, rowptr = template.CSR()
        parts: list[tuple[Any, np.ndarray | None, int, bool]] = []
        for k, fixed, kept, varying in split:
            if kept.pattern is not template and not same_pattern(
                kept.pattern, template
            ):  # pragma: no cover - one space, one full pattern
                local.append((k, _assemble(space, symmetric, fixed.whole)))
                continue
            parts.append((kept.values, None, k, True))
            parts.append((np.array(varying.CSR()[0]), kept.varying[1], k, False))
        outside = False
        template_values = None
        for k, m in local:
            if m is template:
                template_values = template.AsVector()
                parts.append((template_values, None, k, True))
            elif same_pattern(m, template):
                # a copy, so that the piece's own matrix and graph are released
                values = ngs.BaseVector(m.nze, True)
                values.FV().NumPy()[:] = m.AsVector().FV().NumPy()
                parts.append((values, None, k, False))
            else:
                data, c, r = m.CSR()
                at = _positions(rowptr, cols, r, c)
                if at is None:
                    outside = True
                    break
                parts.append((np.array(data), at, k, False))
        self._built = True
        self._generation = object()
        if not _mpi.agree_any(comm, flag=outside):
            self._template, self._parts = matrix, parts
            self._template_values = template_values
            self._split = frozenset(k for k, *_ in split)


#: Scattered parts of a piece sum: (weight, compact values, their positions).
_Scattered = list[tuple[complex, np.ndarray, np.ndarray]]


def _scatter_compiled(out: np.ndarray, scattered: _Scattered) -> bool:
    """Add each ``w * values`` at its positions of ``out``, in place.

    One compiled pass per part, without temporaries: scipy's CSC matrix-vector
    product of the one-column matrix of ``values`` in rows ``at`` with ``[w]``.
    ``False`` (nothing added) without scipy's private routine.
    """
    if _csc_matvec is None:
        return False
    one = np.zeros(2, np.int64)
    for i, (w, values, at) in enumerate(scattered):
        one[1] = len(at)
        try:
            _csc_matvec(len(out), 1, one, at, values, np.array([w]), out)
        # scipy refuses unsupported dtypes (TypeError) and an output dtype the
        # inputs do not cast to (ValueError) before it writes: the rest by numpy
        except (TypeError, ValueError):
            _scatter_numpy(out, scattered[i:])
            break
    return True


def _scatter_numpy(out: np.ndarray, scattered: _Scattered) -> None:
    """`_scatter_compiled` by numpy; parts at one positions array scatter once."""
    summed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for w, values, at in scattered:
        held = summed.get(id(at))
        if held is None:
            summed[id(at)] = (at, w * values)
        else:
            np.add(held[1], w * values, out=held[1])
    for at, values in summed.values():
        out[at] += values


def _positions(
    rowptr: ArrayLike, cols: ArrayLike, sub_rowptr: ArrayLike, sub_cols: ArrayLike
) -> np.ndarray | None:
    """The positions of a sub-pattern's entries in the CSR value vector of a pattern.

    Takes CSR index arrays of any integer type (NGSolve's are unsigned) and
    computes in int64. Reads only the pattern's rows that the sub-pattern
    touches. Columns are sorted within each row (NGSolve's graphs are).
    ``None`` when an entry is not in the pattern.
    """
    rowptr, cols, sub_rowptr, sub_cols = (
        np.asarray(a).astype(np.int64) for a in (rowptr, cols, sub_rowptr, sub_cols)
    )
    counts = np.diff(sub_rowptr)
    rows = np.flatnonzero(counts)
    starts = rowptr[rows]
    lengths = rowptr[rows + 1] - starts
    # the positions of the touched rows' entries, row after row
    first = np.cumsum(lengths) - lengths
    positions = np.repeat(starts - first, lengths) + np.arange(lengths.sum())
    width = len(rowptr)  # above every column index
    keys = np.repeat(np.arange(len(rows)), lengths) * width + cols[positions]
    wanted = np.repeat(np.arange(len(rows)), counts[rows]) * width + sub_cols
    if not len(wanted):
        return np.zeros(0, np.int64)
    found = np.searchsorted(keys, wanted)
    if found.max() >= len(keys) or not np.array_equal(keys[found], wanted):
        return None
    return positions[found]


class Pieces(Protocol):
    """Pieces :math:`A_k` a `PieceSum` sums with its weights (`AssembledPieces`)."""

    @property
    def pattern(self) -> ngs.BaseMatrix | None:
        """The matrix every sum is shaped as, or ``None`` (assemble instead)."""
        ...

    @property
    def generation(self) -> object:
        """Who holds the values: a new object whenever they change."""
        ...

    def matrix(self, weights: Sequence[complex]) -> ngs.BaseMatrix | None:
        r"""A new matrix of `pattern` holding :math:`\sum_k \theta_k A_k`.

        ``None`` without a pattern.
        """
        ...

    def sum_into(self, weights: Sequence[complex], matrix: ngs.BaseMatrix) -> None:
        r"""Overwrite the values of ``matrix`` (of `pattern`).

        They become :math:`\sum_k \theta_k A_k`.
        """
        ...

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes, but `shared`."""
        ...

    def shared(self) -> dict[int, int]:
        """Bytes of what others may hold too, by identity."""
        ...

    def free(self) -> None:
        """Drop what is held; a later use builds it again."""
        ...


@dataclasses.dataclass(frozen=True, eq=False)
class PieceSum:
    r"""One operator as a piece sum: :math:`\sum_k \theta_k A_k` of assembled pieces.

    Attributes:
        pieces: The pieces :math:`A_k` (`AssembledPieces` or another adapter).
        weights: :math:`\theta_k`, one per piece.
    """

    pieces: Pieces
    weights: tuple[complex, ...]

    @property
    def pattern(self) -> ngs.BaseMatrix | None:
        """See `Pieces.pattern`."""
        return self.pieces.pattern

    @property
    def key(self) -> tuple[object, tuple[complex, ...]]:
        """What the sum's values are made of: equal keys, equal values.

        The pieces' assembly (`Pieces.generation`) and the weights;
        it holds neither the pieces nor their values.
        """
        return self.pieces.generation, self.weights

    def matrix(self) -> ngs.BaseMatrix | None:
        """See `Pieces.matrix`."""
        return self.pieces.matrix(self.weights)

    def sum_into(self, matrix: ngs.BaseMatrix) -> None:
        """See `Pieces.sum_into`."""
        self.pieces.sum_into(self.weights, matrix)

    @property
    def stop_norm(self) -> tuple[Any, np.ndarray] | None:
        r"""The map and scaling an iterative stop reads the sum's solutions by.

        ``(T, d)`` when the pieces have a ``stop_norm(weights)`` (the gauged
        piece sum, `gfem._linalg.gauge.GaugedPieces.stop_norm`): a solution
        :math:`\hat y` of the sum stands for the field
        :math:`T (d \circ \hat y)`. ``None`` for every other piece sum, whose
        solutions are the field itself.
        """
        stop_norm = getattr(self.pieces, "stop_norm", None)
        return None if stop_norm is None else stop_norm(self.weights)
