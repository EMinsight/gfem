"""SciPy SuperLU + ARPACK shift-invert for small indefinite eigenproblems.

The internal backend of the 2D port mode problem. Its shifted operator
``A - sigma B`` is indefinite (and complex symmetric with losses), which
NGSolve's ``sparsecholesky`` cannot factor (it has no pivoting and returns
NaN), so SuperLU with partial pivoting factors it on the host. Port
cross-sections are 1e4-1e5 unknowns, so this needs neither the solver-handle
registry nor a GPU.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: ARPACK's default relative tolerance on the shifted eigenvalues: two port
#: mode solves agree to about this, not to rounding.
EIGS_TOLERANCE = 1e-10


class ShiftInvert(spla.LinearOperator):
    """``x -> (A - sigma B)^-1 B x`` with ``A - sigma B`` factored by SuperLU.

    One factor serves ARPACK (`eigs`) and, through `solve`, any further
    solve near the shift, such as a preconditioner.
    """

    def __init__(self, lu: spla.SuperLU, b: sp.csr_matrix, sigma: complex) -> None:
        super().__init__(dtype=np.dtype(b.dtype), shape=b.shape)
        self._lu = lu
        self._b = b
        self.sigma = sigma

    @classmethod
    def factor(cls, a: sp.csr_matrix, b: sp.csr_matrix, sigma: complex) -> ShiftInvert:
        """Factor ``A - sigma B``; a real shift keeps a real pencil real."""
        shift = complex(sigma)
        sigma = shift.real if shift.imag == 0 else shift
        dtype = np.result_type(a.dtype, b.dtype, type(sigma))
        bb = sp.csr_matrix(b, dtype=dtype)
        lu = spla.splu(sp.csc_matrix(sp.csr_matrix(a, dtype=dtype) - sigma * bb))
        return cls(lu, bb, sigma)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._lu.solve(self._b @ x)

    def eigs(
        self, k: int, *, tol: float = EIGS_TOLERANCE
    ) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` eigenpairs nearest the shift; see `shift_invert_eigs`."""
        n = self.shape[0]
        k = max(1, min(k, n - 2))
        ncv = min(n, max(2 * k + 1, 20))
        v0 = np.ones(n, dtype=self._b.dtype)
        try:
            theta, vecs = spla.eigs(self, k=k, which="LM", tol=tol, ncv=ncv, v0=v0)
        except spla.ArpackNoConvergence as err:
            theta, vecs = err.eigenvalues, err.eigenvectors
        keep = np.abs(theta) > 0
        return self.sigma + 1 / theta[keep], vecs[:, keep]

    def solve(self, r: np.ndarray) -> np.ndarray:
        """``(A - sigma B)^-1 r``; a complex ``r`` on a real factor by parts."""
        if np.iscomplexobj(r) and not np.iscomplexobj(self._b):
            return self._lu.solve(np.ascontiguousarray(r.real)) + 1j * self._lu.solve(
                np.ascontiguousarray(r.imag)
            )
        return self._lu.solve(r)


def shift_invert_eigs(
    a: sp.csr_matrix,
    b: sp.csr_matrix,
    sigma: complex,
    k: int,
    *,
    tol: float = EIGS_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` eigenpairs of ``A x = lam B x`` with ``lam`` nearest ``sigma``.

    SuperLU factors ``A - sigma B`` once; ARPACK runs on
    ``(A - sigma B)^-1 B``, whose largest eigenvalues ``1 / (lam - sigma)``
    are the wanted ones. The start vector is fixed, so the result is
    deterministic. If ARPACK stops early, the pairs it converged are returned.

    Args:
        a: Sparse ``A`` (n x n).
        b: Sparse ``B`` (n x n).
        sigma: The shift.
        k: How many pairs; capped at ``n - 2``.
        tol: ARPACK's relative tolerance on the shifted eigenvalues.

    Returns:
        Eigenvalues ``(m,)`` and eigenvectors ``(n, m)``, ``m <= k``.
    """
    return ShiftInvert.factor(a, b, sigma).eigs(k, tol=tol)


def backward_error(
    a: sp.csr_matrix,
    b: sp.csr_matrix,
    lam: complex,
    x: np.ndarray,
) -> float:
    """``|A x - lam B x| / (|A x| + |lam| |B x|)`` of one eigenpair."""
    ax, bx = a @ x, b @ x
    den = np.linalg.norm(ax) + abs(lam) * np.linalg.norm(bx)
    return float(np.linalg.norm(ax - lam * bx) / den) if den > 0 else 0.0
