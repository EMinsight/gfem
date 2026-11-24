r"""The reverse-mode derivative of a port mode with respect to its 2D pencil.

A port mode is an eigenpair of the complex symmetric pencil
:math:`A(k_0)\,x = \lambda B(k_0)\,x`, :math:`\lambda = \gamma^2`, so
:math:`x^T` is the left eigenvector and
:math:`\mathrm{d}\lambda = x^T(\mathrm{d}A - \lambda\,\mathrm{d}B)\,x / x^TBx`.

Take a real objective of :math:`\lambda` and of the normalized mode
coordinates, which are invariant under :math:`x \to cx` for every complex
:math:`c \ne 0` (power, phase gauge and sign fix modulus and phase). With
the cotangents :math:`\bar\lambda` and :math:`\bar x` in the convention
:math:`\mathrm{d}L = \operatorname{Re}(\bar\lambda\,\mathrm{d}\lambda
+ \bar x^T\mathrm{d}x)`, one deflated solve

.. math::

    (A - \lambda B)\,w = \bar x - Bx\,\frac{x^T\bar x}{x^TBx},
    \qquad x^TBw = 0,

gives :math:`q = \bar\lambda\,x / (x^TBx) - w` and
:math:`\mathrm{d}L = \operatorname{Re}\,q^T(\mathrm{d}A - \lambda\,\mathrm{d}B)\,x`.
The cotangent of the piece :math:`A_j` is :math:`\theta_j\,q x^T`, that of
:math:`B_j` is :math:`-\lambda\theta_j\,q x^T`, and that of the scalar
:math:`\theta_j` is :math:`q^T(A_j - \lambda B_j)\,x`. A cotangent on
:math:`\lambda` alone needs no solve.

The solve is GMRES on :math:`\Pi_l(A - \lambda B)\Pi_r`, with
:math:`\Pi_l = I - Bx\,x^T/x^TBx` and :math:`\Pi_r = I - x\,x^TB/x^TBx`,
preconditioned by a shift-invert factor of :math:`A - \sigma B`. The
projection keeps an inexact pair (a reduced row) from amplifying the
right-hand side's inconsistency by :math:`1/(\lambda_\mathrm{exact} -
\lambda)`; a plain solve of the nearly singular :math:`A - \lambda B` cannot
recover from that.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gfem.errors import PortModeError

# GMRES target and the level accepted once the residual stops falling: at
# the attainable accuracy of an ill-conditioned section (a coplanar line's
# low-frequency floor) the residual stalls above the target for
# hundreds of iterations without improving the gradient.
RTOL = 1e-10
ACCEPT = 1e-8
POLISH_RTOL = 1e-4
# Iterations over which the residual must fall by `_FALL` to count as falling.
_WINDOW = 10
_FALL = 2.0
_MAX_ITER = 300
_RESTART = 60


class Factor(Protocol):
    r"""A factorization of :math:`A - \sigma B` (`scipy.sparse.linalg.splu`)."""

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """The solution of the factored system for ``rhs``."""
        ...


class Pieces(Protocol):
    """The affine pieces of a pencil (`gfem._engine.port_mode.Pencil`)."""

    a: tuple[sp.csr_matrix, ...]
    b: tuple[sp.csr_matrix, ...]


@dataclasses.dataclass(frozen=True)
class AdjointSolve:
    r"""One deflated solve.

    Attributes:
        w: The solution, with :math:`x^TBw = 0`.
        residual: :math:`|\Pi_l((A - \lambda B)w - r)| / |r|` with the
            projected right-hand side :math:`r`.
        iterations: GMRES iterations spent.
        direct: Whether the direct fallback produced ``w``.
    """

    w: np.ndarray
    residual: float
    iterations: int
    direct: bool


def shift_invert_factor(a: sp.spmatrix, b: sp.spmatrix, sigma: complex) -> Factor:
    r"""The sparse LU of :math:`A - \sigma B`."""
    return spla.splu(sp.csc_matrix(a - sigma * b, dtype=complex))


def degenerate_error(who: str | None, residual: float) -> PortModeError:
    """The failure of an adjoint solve, in the forward's degenerate wording."""
    from gfem._sweep.port_mode import _DEGENERATE_REMEDY

    return PortModeError(
        f"{who + ': ' if who else ''}the port mode's derivative could not be "
        f"computed (the adjoint solve stalled at a relative residual of "
        f"{residual:.1e}): another mode has nearly the same propagation "
        "constant, so the port's field is not defined. " + _DEGENERATE_REMEDY
    )


def deflated_solve(
    a: sp.spmatrix,
    b: sp.spmatrix,
    lam: complex,
    x: np.ndarray,
    rhs: np.ndarray,
    *,
    factor: Factor,
    rtol: float = RTOL,
    accept: float | None = ACCEPT,
    max_iter: int = _MAX_ITER,
    fallback: bool = True,
    who: str | None = None,
) -> AdjointSolve:
    r"""Solve :math:`(A - \lambda B)w = r`, :math:`x^TBw = 0` for the projected ``rhs``.

    GMRES stops at ``rtol``, or below ``accept`` once the residual has
    stopped falling (less than a factor 2 over 10 iterations; ``None``
    keeps only ``rtol``). A result with a true residual above ``accept``
    (at least ``100 rtol``) goes to a direct LU of
    :math:`A - \lambda B` plus projection (unless ``fallback`` is off).

    Raises:
        PortModeError: Neither the iteration nor the fallback reached that
            residual, the mark of a degenerate mode.
    """
    bx = b @ x
    xbx = x @ bx
    limit = max(accept or 0.0, 100 * rtol)
    rhs = rhs - bx * ((x @ rhs) / xbx)
    norm = float(np.linalg.norm(rhs))
    if norm == 0.0:
        return AdjointSolve(np.zeros_like(x), 0.0, 0, False)
    t = sp.csr_matrix(a - lam * b)

    def pi_l(v: np.ndarray) -> np.ndarray:
        return v - bx * ((x @ v) / xbx)

    def pi_r(v: np.ndarray) -> np.ndarray:
        return v - x * ((bx @ v) / xbx)

    def residual(w: np.ndarray) -> float:
        return float(np.linalg.norm(pi_l(t @ w - rhs)) / norm)

    w, iterations = _gmres(
        lambda v: pi_l(t @ pi_r(v)),
        lambda v: pi_r(factor.solve(pi_l(v))),
        rhs,
        rtol=rtol,
        accept=accept,
        max_iter=max_iter,
    )
    w = pi_r(w)
    res = residual(w)
    if res <= limit:
        return AdjointSolve(w, res, iterations, False)
    if fallback:
        w = pi_r(spla.splu(sp.csc_matrix(t, dtype=complex)).solve(rhs))
        res = residual(w)
        if res <= limit:
            return AdjointSolve(w, res, iterations, True)
    raise degenerate_error(who, res)


def _gmres(op, pre, rhs, *, rtol, accept, max_iter):
    """Restarted right-preconditioned GMRES from zero: ``(solution, iterations)``.

    Within a cycle the Givens recursion estimates the residual; every
    restart replaces the estimate by the true residual, which is where an
    ill-conditioned section's floor shows. The stop reads that history.
    """
    beta = float(np.linalg.norm(rhs))
    n = len(rhs)
    m = min(_RESTART, max_iter)
    basis = np.empty((m + 1, n), dtype=complex)
    pre_basis = np.empty((m, n), dtype=complex)
    w = np.zeros(n, dtype=complex)
    r = rhs
    history = [1.0]
    total = 0
    while total < max_iter:
        hess = np.zeros((m + 1, m), dtype=complex)
        cs = np.zeros(m, dtype=complex)
        sn = np.zeros(m, dtype=complex)
        g = np.zeros(m + 1, dtype=complex)
        g[0] = np.linalg.norm(r)
        basis[0] = r / g[0]
        k = 0
        stop = False
        while k < m and total < max_iter:
            pre_basis[k] = pre(basis[k])
            v = op(pre_basis[k])
            for _ in range(2):  # Gram-Schmidt twice, for orthogonality
                for i in range(k + 1):
                    c = np.vdot(basis[i], v)
                    hess[i, k] += c
                    v = v - c * basis[i]
            hess[k + 1, k] = vnorm = np.linalg.norm(v)
            for i in range(k):
                top = np.conj(cs[i]) * hess[i, k] + np.conj(sn[i]) * hess[i + 1, k]
                hess[i + 1, k] = -sn[i] * hess[i, k] + cs[i] * hess[i + 1, k]
                hess[i, k] = top
            h = np.hypot(abs(hess[k, k]), abs(hess[k + 1, k]))
            cs[k], sn[k] = (
                (1.0, 0.0) if h == 0.0 else (hess[k, k] / h, hess[k + 1, k] / h)
            )
            hess[k, k], hess[k + 1, k] = h, 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = np.conj(cs[k]) * g[k]
            k += 1
            total += 1
            history.append(abs(g[k]) / beta)
            stop = _stops(history, rtol, accept)
            if stop or vnorm == 0.0:  # converged, or an invariant subspace
                break
            basis[k] = v / vnorm
        y = np.linalg.solve(np.triu(hess[:k, :k]), g[:k])
        w = w + y @ pre_basis[:k]
        r = rhs - op(w)
        history[-1] = float(np.linalg.norm(r)) / beta  # the true residual
        if _stops(history, rtol, accept) or history[-1] == 0.0:
            break
    return w, total


def _stops(history: list[float], rtol: float, accept: float | None) -> bool:
    """Whether the residual reached ``rtol``, or ``accept`` and stopped falling."""
    if history[-1] <= rtol:
        return True
    if accept is None or history[-1] > accept or len(history) <= _WINDOW:
        return False
    return history[-1] > history[-1 - _WINDOW] / _FALL


def polish(
    a: sp.spmatrix,
    b: sp.spmatrix,
    lam: complex,
    x: np.ndarray,
    *,
    factor: Factor,
    rtol: float = POLISH_RTOL,
    who: str | None = None,
) -> tuple[complex, np.ndarray]:
    r"""One Newton step on an inexact pair, then the Rayleigh quotient.

    Solves :math:`\Pi_l(A - \lambda B)\Pi_r t = -r` with the residual
    :math:`r = Ax - \lambda Bx` (the Jacobi-Davidson correction equation)
    and returns the quotient :math:`x^TAx / x^TBx` of :math:`x + t` with that
    vector. A row of a reduced port mode model, with a backward error up to
    the sweep's tolerance, comes out at the exact discrete pair's accuracy.
    """
    residual = a @ x - lam * (b @ x)
    step = deflated_solve(
        a, b, lam, x, -residual, factor=factor, rtol=rtol, accept=None, who=who
    )
    y = x + step.w
    return complex((y @ (a @ y)) / (y @ (b @ y))), y


def pencil_cotangent(
    a: sp.spmatrix,
    b: sp.spmatrix,
    lam: complex,
    x: np.ndarray,
    *,
    lam_bar: complex,
    x_bar: np.ndarray | None,
    factor: Factor,
    **options,
) -> np.ndarray:
    r"""The vector :math:`q` that carries the pair's cotangent to the pencil.

    :math:`\mathrm{d}L = \operatorname{Re}\,q^T(\mathrm{d}A - \lambda\,\mathrm{d}B)x`.

    ``x_bar`` is the cotangent of the pair's vector (``None`` or zero for a
    cotangent on :math:`\lambda` alone, which needs no solve). The extra
    keyword arguments go to `deflated_solve`.
    """
    q = lam_bar * x / (x @ (b @ x))
    if x_bar is not None and np.any(x_bar):
        q = q - deflated_solve(a, b, lam, x, x_bar, factor=factor, **options).w
    return q


def piece_weights(
    theta: tuple[complex, ...], lam: complex
) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    r"""The weights of :math:`q x^T` in the cotangents of ``A`` and ``B`` pieces.

    The cotangent of :math:`A_j` is :math:`\theta_j\,qx^T` and that of
    :math:`B_j` is :math:`-\lambda\theta_j\,qx^T`.
    """
    return tuple(theta), tuple(-lam * t for t in theta)


def scalar_cotangents(
    pencil: Pieces, lam: complex, q: np.ndarray, x: np.ndarray
) -> np.ndarray:
    r"""The cotangent :math:`q^T(A_j - \lambda B_j)\,x` of each :math:`\theta_j`."""
    return np.array(
        [
            q @ (aj @ x) - lam * (q @ (bj @ x))
            for aj, bj in zip(pencil.a, pencil.b, strict=True)
        ]
    )
