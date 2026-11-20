"""A prepared operator plus a low-rank term: ``A + U C Vᵀ`` by Woodbury.

``low_rank_update(handle, U, C, right=V)`` wraps a prepared handle of ``A``
into a handle of ``A + U C Vᵀ`` (``U`` and ``V`` ``(ndof, p)`` with few
columns, ``C`` ``(p, p)``), never assembling the update: a wave port's modal
correction is dense on the port face, so it stays matrix-free.
``V`` defaults to ``U`` (a wave port's ``U C Uᵀ``); a Floquet surface under a
Bloch wave vector has ``V = conj(U)``. A solve uses the Sherman-Morrison-Woodbury
identity

    (A + U C Vᵀ)⁻¹ B = Y - Z (I + C Vᵀ Z)⁻¹ C Vᵀ Y,   Y = A⁻¹ B,  Z = A⁻¹ U,

so it costs ``p`` extra columns of the base solve, once per handle (``Z``
is kept for later solves, per direction and tolerance). The first solve
puts ``B`` and ``U`` in one call of the base handle. ``C`` need not be
invertible. A transposed solve is one with ``Aᵀ + V Cᵀ Uᵀ``: the same
identity with ``U`` and ``V`` swapped and ``Cᵀ``.

The capacitance ``I + C Vᵀ Z`` may be row-scaled (``row_scaled``): a row
``i`` with ``|C_ii| max_j |(Vᵀ Z)_ij| > 1`` is divided by ``C_ii``, so a
large diagonal shift (a Floquet order near grazing, whose TM shift grows as
``1/gamma``) leaves the small system well conditioned; the solution is the same
up to round-off. A row may be flagged ``infinite``: its diagonal shift is
infinite (a TM order exactly at grazing), and its row of the capacitance is
the exact constraint ``(Vᵀ X)_i = 0`` (``(Uᵀ X)_i = 0`` transposed), the
limit of a growing shift, never ``inf`` or NaN arithmetic. A flagged row
and column of ``C`` must be zero (its value is the flag), and such an update
has no product (`LowRankUpdate.apply` raises).

``U`` and ``V`` must vanish on the fixed DOFs (the update acts on the free
ones, like the base solve).

The update owns its base handle unless it borrows it (``borrow=True``): a
borrowing update's `free` drops only its own columns, and the base stays
usable (a sweep's base handle, wrapped at each frequency).
"""

from __future__ import annotations

import numpy as np

from gfem._linalg import Handle

__all__ = ["LowRankUpdate"]


class LowRankUpdate:
    """A `gfem._linalg.Handle` of ``A + U C Vᵀ`` (see the module docstring).

    Attributes:
        kind: The base handle's backend.
        symmetry: The base handle's symmetry (the update is the caller's).
        ndof: The number of DOFs.
        vectors: ``U``.
        coefficients: ``C``.
        right: ``V``; ``None`` for ``U``.
        infinite: Per row of ``C``, whether its diagonal shift is infinite;
            ``None`` for none.
        row_scaled: Whether the capacitance is row-scaled.
    """

    def __init__(
        self,
        handle: Handle,
        vectors: np.ndarray,
        coefficients,
        *,
        right: np.ndarray | None = None,
        infinite: np.ndarray | None = None,
        row_scaled: bool = False,
        borrow: bool = False,
    ) -> None:
        """Wrap ``handle`` (of ``A``) with the update ``U C Vᵀ``.

        ``right`` is ``V`` (``None``: ``U``); ``infinite`` flags the rows of
        ``C`` whose diagonal shift is infinite (their row and column of ``C``
        zero); ``row_scaled`` divides the capacitance's large rows by their
        shift. ``borrow`` leaves ``handle`` to its owner: `free` does not
        free it.
        """
        vectors = np.asarray(vectors)
        coefficients = np.asarray(coefficients)
        p = vectors.shape[1]
        if vectors.shape[0] != handle.ndof or coefficients.shape != (p, p):
            raise ValueError(
                f"Expected U of shape ({handle.ndof}, p) and C of shape (p, p), "
                f"got {vectors.shape} and {coefficients.shape}."
            )
        if right is not None:
            right = np.asarray(right)
            if right.shape != vectors.shape:
                raise ValueError(
                    f"Expected V of the shape of U {vectors.shape}, got {right.shape}."
                )
        if infinite is not None:
            infinite = np.asarray(infinite, bool)
            if infinite.shape != (p,):
                raise ValueError(
                    f"Expected one infinite flag per column ({p},), "
                    f"got {infinite.shape}."
                )
            if np.any(coefficients[infinite]) or np.any(coefficients[:, infinite]):
                raise ValueError(
                    "The row and column of C of an infinite shift must be zero."
                )
            if not infinite.any():
                infinite = None
        self.kind = handle.kind
        self.symmetry = handle.symmetry
        self.ndof = handle.ndof
        self.layout = handle.layout
        self.vectors = vectors
        self.coefficients = coefficients
        self.right = right
        self.infinite = infinite
        self.row_scaled = row_scaled
        self._base: Handle | None = handle
        self._owner = not borrow
        self._z: dict[tuple[bool, float | None], np.ndarray] = {}

    @property
    def base(self) -> Handle | None:
        """The handle of ``A`` this wraps, ``None`` once freed."""
        return self._base

    @property
    def nbytes(self) -> int:
        """Host memory of the base handle, ``U``, ``V`` and the kept ``A⁻¹ U``."""
        if self._base is None:
            return 0
        kept = sum(z.nbytes for z in self._z.values())
        right = 0 if self.right is None else self.right.nbytes
        return int(self._base.nbytes) + self.vectors.nbytes + right + kept

    def shared(self) -> dict[int, int]:
        """The base handle's shared objects' bytes, by identity (not in ``nbytes``)."""
        shared = getattr(self._base, "shared", None)
        return {} if shared is None else shared()

    def _factors(self, transpose: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The left factor, right factor and ``C`` of the solve's direction.

        ``(U, V, C)``, transposed ``(V, U, Cᵀ)``.
        """
        U = self.vectors
        V = U if self.right is None else self.right
        if transpose:
            return V, U, self.coefficients.T
        return U, V, self.coefficients

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
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Solve ``(A + U C Vᵀ) X = B`` (``transpose``: ``(Aᵀ + V Cᵀ Uᵀ) X = B``).

        As `gfem._linalg.Handle.solve_counted`; ``x0`` is an initial guess of
        ``X`` and goes to the base solve of ``Y``. The counts are those of
        the columns of ``B``, never of the update's own columns.
        """
        base = self._base
        if base is None:
            raise RuntimeError("This solver handle was freed.")
        B = np.asarray(B)
        flat = B.ndim == 1
        columns = B[:, None] if flat else B
        guess = None if x0 is None else np.asarray(x0).reshape(columns.shape)
        k = columns.shape[1]
        U, V, C = self._factors(transpose)
        key = (transpose, tolerance)
        if key in self._z:
            Z = self._z[key]
            Y, its = base.solve_counted(
                columns, transpose=transpose, tolerance=tolerance, x0=guess
            )
        else:
            both = np.concatenate([columns, U], axis=1)
            if guess is not None:
                guess = np.concatenate([guess, np.zeros(U.shape, guess.dtype)], axis=1)
            out, its = base.solve_counted(
                both, transpose=transpose, tolerance=tolerance, x0=guess
            )
            Y, Z = out[:, :k], out[:, k:]
            self._z[key] = Z
        # U and V are loads, Z and Y states; a rank sum makes the small system agreed
        pair = self.layout.pair
        if not self.row_scaled and self.infinite is None:
            small = np.eye(C.shape[0]) + C @ pair(V, Z)
            X = Y - Z @ np.linalg.solve(small, C @ pair(V, Y))
        else:
            small, rhs = _scaled(C, pair(V, Z), pair(V, Y), self.infinite)
            X = Y - Z @ np.linalg.solve(small, rhs)
        return (X[:, 0] if flat else X), its[:k]

    def apply(self, X: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """``(A + U C Vᵀ) X`` with the base handle's full matrix.

        ``transpose``: ``(Aᵀ + V Cᵀ Uᵀ) X``.

        Raises:
            ValueError: A shift is infinite.
        """
        if self._base is None:
            raise RuntimeError("This solver handle was freed.")
        if self.infinite is not None:
            raise ValueError("An update with an infinite shift has no product.")
        X = np.asarray(X)
        U, V, C = self._factors(transpose)
        return self._base.apply(X, transpose=transpose) + U @ (
            C @ self.layout.pair(V, X)
        )

    def free(self) -> None:
        """Release the kept columns, and the base handle unless borrowed."""
        if self._base is not None and self._owner:
            self._base.free()
        self._base = None
        self._z = {}


def _scaled(
    C: np.ndarray, G: np.ndarray, R: np.ndarray, infinite: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """The row-scaled capacitance ``I + C G`` and right-hand side ``C R``.

    A row ``i`` with ``|C_ii| max_j |G_ij| > 1`` is divided by ``C_ii``; a
    row flagged ``infinite`` is the limit, ``G_i`` and ``R_i``.
    """
    small = np.eye(C.shape[0]) + C @ G
    rhs = C @ R
    shift = np.diagonal(C)
    rows = np.abs(shift) * np.abs(G).max(axis=1, initial=0.0) > 1.0
    small[rows] /= shift[rows, None]
    rhs[rows] /= shift[rows, None]
    if infinite is not None:
        small[infinite] = G[infinite]
        rhs[infinite] = R[infinite]
    return small, rhs
