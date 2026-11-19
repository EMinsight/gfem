r"""The gauged piece sum and the handle that solves through it (ADR 0215, 0216).

Two adapters on existing protocols carry a driven solve through the
tree-cotree gauge :math:`x = T y` (`gfem._engine.gauge.GaugeBasis`):

- `GaugedPieces` is a `gfem._linalg.pieces.Pieces` whose matrices are the
  gauged piece sum on the gauge space :math:`Y`. It forms each gauged piece
  :math:`\hat A_k = T^\mathsf{T} A_k T` once from the assembled pieces (then
  frees those), and at each set of weights writes

  .. math::

      D \Big( \sum_k \theta_k \hat A_k \Big) D, \qquad
      D = \operatorname{diag}(d), \quad
      d_i = \Big| \sum_k \theta_k (\hat A_k)_{ii} \Big|^{-1/2}

  (:math:`d_i = 1` where that diagonal is zero) into a matrix of the gauge
  template. A direct handle prepared on it refactorizes in place, pooled and
  reused as any other.
- `GaugedHandle` is a `gfem._linalg.Handle` in the driven space's
  coordinates over a base handle of that matrix:
  :math:`\hat B = d \circ T^\mathsf{T} B`, :math:`X = T (d \circ \hat Y)`.
  Since :math:`T` is square and nonsingular from the free DOFs of :math:`Y`
  to those of :math:`V`, :math:`T \hat A^{-1} T^\mathsf{T} = A^{-1}` on the
  free DOFs: callers (Woodbury updates, adjoint solves) see the ungauged
  operator's inverse.

Only one process, and only complex symmetric operators (the gauge's scope):
`gfem._engine.system_operator.SystemOperator` is the only place that builds
them.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ngsolve as ngs
import numpy as np

from gfem._linalg.base import csr_bytes, host_bytes
from gfem._linalg.pieces import Pieces, _scatter_compiled, _scatter_numpy

if TYPE_CHECKING:
    import scipy.sparse as sp

    from gfem._engine.gauge import GaugeBasis
    from gfem._linalg import Handle, Symmetry
    from gfem._linalg.layout import DofLayout

__all__ = ["GaugedHandle", "GaugedPieces"]


class GaugedPieces:
    r"""The gauged piece sum: `Pieces` on the gauge space :math:`Y`.

    Wraps the driven space's pieces :math:`A_k` (an `AssembledPieces`). At
    first use it forms each gauged piece
    :math:`\hat A_k = T^\mathsf{T} A_k T` (`GaugeBasis.transform`), keeps
    its values, their positions in the gauge template and its diagonal, and
    frees the wrapped pieces. `free` drops the gauged values; the next use
    assembles and transforms again (kept fixed parts stay in their store).

    The pattern is the gauge template itself, for every build: a handle
    prepared on one design sums the next one's values in place.
    """

    def __init__(self, pieces: Pieces, basis: GaugeBasis, kinds: Sequence[str]) -> None:
        """Wrap ``pieces`` (on the driven space) under the gauge ``basis``.

        ``kinds`` holds each piece's kind (`gfem._engine.terms.Piece.kind`):
        a curl piece is formed on the cotree block only.
        """
        self._inner = pieces
        self._basis = basis
        self._kinds = tuple(kinds)
        # (values, positions in the template) per piece, and the diagonals
        self._parts: list[tuple[np.ndarray, np.ndarray]] = []
        self._diagonals: np.ndarray | None = None
        self._generation: object | None = None
        # the gauged pattern's row of each template value (Y-free numbering)
        self._rows: np.ndarray | None = None
        #: Serializes the first use (formed once) and `free`.
        self._lock = threading.Lock()

    @property
    def basis(self) -> GaugeBasis:
        """The gauge basis :math:`T`."""
        return self._basis

    @property
    def pattern(self) -> ngs.BaseMatrix:
        """The gauge template: every sum is a matrix of its pattern.

        Never written into: `matrix` gives a new one.
        """
        return self._basis.template

    @property
    def generation(self) -> object:
        """Who holds these values: a new object at each build (after `free`)."""
        with self._lock:
            self._build()
            return self._generation

    def scaling(self, weights: Sequence[complex]) -> np.ndarray:
        r"""The symmetric diagonal scaling :math:`d` at ``weights``, Y-free numbering.

        .. math::

            d_i = \Big| \sum_k \theta_k (\hat A_k)_{ii} \Big|^{-1/2},

        and 1 where that diagonal is zero. A pure function of the weights:
        `sum_into` and `GaugedHandle` read the same one.
        """
        with self._lock:
            self._build()
            diagonals = self._diagonals
        assert diagonals is not None  # noqa: S101 - built above
        theta = np.asarray([complex(w) for w in weights])
        if len(theta) != len(self._kinds):
            raise ValueError(f"{len(self._kinds)} weights, got {len(theta)}.")
        size = np.abs(theta @ diagonals)
        return np.where(size > 0, 1.0 / np.sqrt(np.where(size > 0, size, 1.0)), 1.0)

    def stop_norm(self, weights: Sequence[complex]) -> tuple[sp.csr_matrix, np.ndarray]:
        r"""The gauge basis :math:`T` and the scaling :math:`d` at ``weights``.

        A solution :math:`\hat y` of the scaled sum stands for the field
        :math:`T (d \circ \hat y)` (Y-free numbering): a direct handle's
        reused-factor GMRES stops on that field's norm (ADR 0094, 0216).
        """
        return self._basis.T, self.scaling(weights)

    def matrix(self, weights: Sequence[complex]) -> ngs.BaseMatrix:
        r"""A new matrix of `pattern` holding :math:`D (\sum_k \theta_k \hat A_k) D`."""
        matrix = self._basis.template.CreateMatrix()
        self.sum_into(weights, matrix)
        return matrix

    def sum_into(self, weights: Sequence[complex], matrix: ngs.BaseMatrix) -> None:
        r"""Overwrite ``matrix`` (of `pattern`) with the scaled gauged sum.

        That is :math:`D (\sum_k \theta_k \hat A_k) D`, with :math:`D` from
        `scaling`.
        """
        d = self.scaling(weights)
        with self._lock:
            self._build()
            parts, rows = self._parts, self._rows
        assert rows is not None  # noqa: S101 - built above
        out = matrix.AsVector().FV().NumPy()
        out[:] = 0
        scattered = [
            (complex(w), values, at)
            for w, (values, at) in zip(weights, parts, strict=True)
        ]
        if not _scatter_compiled(out, scattered):
            _scatter_numpy(out, scattered)
        out *= d[rows]
        out *= d[self._basis.pattern.indices]

    @property
    def nbytes(self) -> int:
        """Host memory of the gauged values, positions and diagonals, in bytes.

        The template is in `shared`.
        """
        with self._lock:
            parts, diagonals, rows = self._parts, self._diagonals, self._rows
        held = sum(int(values.nbytes) for values, _ in parts)
        held += sum(int(at.nbytes) for at in {id(at): at for _, at in parts}.values())
        held += 0 if diagonals is None else int(diagonals.nbytes)
        return held + (0 if rows is None else int(rows.nbytes))

    def shared(self) -> dict[int, int]:
        """The gauge template's bytes, by identity: handles and the pool hold it too."""
        template = self._basis.template
        return {id(template): host_bytes(template)}

    def free(self) -> None:
        """Drop the gauged values and the wrapped pieces."""
        with self._lock:
            self._parts = []
            self._diagonals = None
            self._rows = None
            self._generation = None
            self._inner.free()

    def _build(self) -> None:
        """Form each gauged piece once; under the lock."""
        if self._generation is not None:
            return
        basis, n = self._basis, len(self._kinds)
        parts: list[tuple[np.ndarray, np.ndarray]] = []
        diagonals = np.zeros((n, basis.T.shape[1]), complex)
        for k, kind in enumerate(self._kinds):
            unit = np.zeros(n, complex)
            unit[k] = 1.0
            piece = self._inner.matrix(tuple(unit))
            if piece is None:  # pragma: no cover - one space, one full pattern
                raise RuntimeError("The driven pieces have no common pattern.")
            hat = basis.transform(piece, kind)
            del piece
            at = basis.positions(hat)
            # the volume pieces share their pattern: one positions array
            for _, other in parts:
                if np.array_equal(other, at):
                    at = other
                    break
            parts.append((np.asarray(hat.data, complex), at))
            diagonals[k] = hat.diagonal()
        # the gauged values hold the pieces now
        self._inner.free()
        indptr = basis.pattern.indptr
        self._rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        self._parts, self._diagonals = parts, diagonals
        self._generation = object()


class GaugedHandle:
    r"""A handle in the driven space's coordinates over a gauged base handle.

    ``base`` solves :math:`D \hat A D \hat Z = \hat B` on the gauge space
    :math:`Y`; this handle solves :math:`A X = B` on the free DOFs of
    :math:`V`: :math:`\hat B = d \circ T^\mathsf{T} B` (fixed rows of
    :math:`B` unread), :math:`X = T (d \circ \hat Z)` (zero on the fixed
    DOFs). It never refactorizes: its owner refactorizes ``base`` and wraps
    it again with the new scaling.

    Attributes:
        kind: The base handle's backend.
        symmetry: The base handle's symmetry.
        ndof: The driven space's DOFs.
        layout: The base handle's layout (one process).
    """

    def __init__(
        self,
        base: Handle,
        basis: GaugeBasis,
        scaling: np.ndarray,
        *,
        ndof: int,
        borrow: bool = False,
    ) -> None:
        """Wrap ``base`` (of the gauged piece sum scaled by ``scaling``).

        ``ndof``: the driven space's DOFs. ``borrow`` leaves ``base`` to its
        owner: `free` does not free it.
        """
        self.kind: str = base.kind
        self.symmetry: Symmetry = base.symmetry
        self.ndof = int(ndof)
        self.layout: DofLayout = base.layout
        self._base: Handle | None = base
        self._basis = basis
        self._scaling = np.asarray(scaling)
        self._owner = not borrow

    @property
    def base(self) -> Handle | None:
        """The gauged base handle this wraps, ``None`` once freed."""
        return self._base

    @property
    def last_action(self) -> str | None:
        """The base handle's last action (`gfem._linalg.report.solve_step`)."""
        return getattr(self._base, "last_action", None)

    @property
    def last_reason(self) -> str | None:
        """The base handle's last reason."""
        return getattr(self._base, "last_reason", None)

    @property
    def last_details(self) -> dict[str, float] | None:
        """The base handle's last reason's numbers."""
        return getattr(self._base, "last_details", None)

    @property
    def nbytes(self) -> int:
        """The base handle's host memory (the basis is in `shared`)."""
        return 0 if self._base is None else int(self._base.nbytes)

    def shared(self) -> dict[int, int]:
        """The base handle's shared objects and the gauge basis, by identity."""
        shared = getattr(self._base, "shared", None)
        held = {} if shared is None else dict(shared())
        for matrix in (self._basis.T, self._basis.TT):
            held[id(matrix)] = csr_bytes(matrix)
        return held

    def solve(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """``X`` alone: ``solve_counted(B, ...)[0]``."""
        return self.solve_counted(B, transpose=transpose, tolerance=tolerance, x0=x0)[0]

    def solve_counted(
        self,
        B: np.ndarray,
        *,
        transpose: bool = False,
        tolerance: float | None = None,
        x0: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        r"""Solve :math:`A X = B` through the gauged base.

        As `gfem._linalg.Handle.solve_counted`. ``transpose`` and
        ``tolerance`` go to the base solve; ``x0`` is dropped (a direct base
        ignores it).
        """
        base = self._base
        if base is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.asarray(B)
        if B.shape[0] != self.ndof or B.ndim not in (1, 2):
            raise ValueError(
                f"Expected an array of shape ({self.ndof},) or ({self.ndof}, k), "
                f"got {B.shape}."
            )
        flat = B.ndim == 1
        columns = B[:, None] if flat else B
        basis, d = self._basis, self._scaling[:, None]
        loads = np.zeros((basis.ndof_y, columns.shape[1]), complex)
        loads[basis.y_dofs] = d * (basis.TT @ columns[basis.v_dofs])
        out, its = base.solve_counted(loads, transpose=transpose, tolerance=tolerance)
        X = np.zeros((self.ndof, columns.shape[1]), complex)
        X[basis.v_dofs] = basis.T @ (d * out[basis.y_dofs])
        return (X[:, 0] if flat else X), its

    def apply(self, X: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Not available: no driven caller applies a gauged system handle.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("A gauged handle has no product.")

    def free(self) -> None:
        """Release the base handle unless borrowed."""
        if self._base is not None and self._owner:
            self._base.free()
        self._base = None
