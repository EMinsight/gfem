"""Shift-invert Arnoldi (SciPy ARPACK) on the eigenmode seed operator.

ARPACK's implicitly restarted Arnoldi finds the largest eigenvalues of the
shared seed operator (`gfem._eigen.seed`, which derives the shift-invert
and companion forms): ``T(sigma)^-1`` through the eigen operator's linear
solver, the divergence-free projection on every vector, and a fixed,
projected start vector, so a solve is deterministic. A solve stops after
``seed._SEED_RESTARTS`` restarts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse.linalg as spla

from gfem._eigen import seed as _seed
from gfem._eigen.operator import EigenOperator
from gfem._eigen.seed import SeedBoundReached, seed_operator

__all__ = ["shift_invert"]


class _Operator(spla.LinearOperator):
    """A complex ``size x size`` operator from its matrix-vector product."""

    def __init__(self, size: int, apply: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(dtype=np.dtype(complex), shape=(size, size))
        self._apply = apply
        #: How many times the operator was applied.
        self.applies = 0

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        self.applies += 1
        return self._apply(np.asarray(x).reshape(-1))


def shift_invert(
    op: EigenOperator,
    project,
    nev: int,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``nev`` eigenpairs ``(k, x)`` of ``T`` nearest ``op.shift``.

    Args:
        op: The operator with ``T(sigma)`` prepared.
        project: The divergence-free projection, ``x -> x``.
        nev: How many pairs to ask ARPACK for.
        tol: ARPACK's relative tolerance on the shift-inverted eigenvalues.

    Returns:
        Wavenumbers ``(m,)`` (per file unit, ``m <= nev``; may include
        pairs with ``Re k <= 0``) and eigenvectors ``(n, m)``.

    Raises:
        SeedBoundReached: ARPACK stopped at ``_SEED_RESTARTS`` restarts; it
            carries the pairs it converged.
    """
    seed = seed_operator(op, project)
    nev, ncv = seed.dimensions(nev)
    restarts = _seed._SEED_RESTARTS  # noqa: SLF001 - read now: tests patch it
    operator = _Operator(seed.size, seed.apply)
    try:
        theta, vectors = spla.eigs(
            operator,
            k=nev,
            which="LM",
            tol=tol,
            ncv=ncv,
            v0=seed.start,
            maxiter=restarts,
        )
    except spla.ArpackNoConvergence as err:
        k, x = seed.pairs(err.eigenvalues, err.eigenvectors)
        raise SeedBoundReached(
            k, x, restarts=restarts, applies=operator.applies
        ) from None
    return seed.pairs(theta, vectors)
