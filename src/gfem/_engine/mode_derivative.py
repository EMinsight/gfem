r"""The derivative of a port mode in the wavenumber: group index and dispersion.

A port mode solves :math:`A(k_0)x = \lambda B(k_0)x` with
:math:`\lambda = \gamma^2` and the affine pencil
:math:`A(k_0 + h) = \sum_n h^n A_n`, :math:`A_n = \sum_j c_{jn}A_j` from
the pencil's Taylor coefficients (`gfem._engine.port_mode.Pencil.taylor`),
:math:`B` likewise. The pieces are complex symmetric, so :math:`x^T` is the
left eigenvector and Hellmann-Feynman gives

.. math::

    \lambda' = \frac{x^T(A_1 - \lambda B_1)x}{x^TB_0x}.

The mode derivative :math:`x'` solves

.. math::

    (A_0 - \lambda B_0)x' = -(A_1 - \lambda'B_0 - \lambda B_1)x,
    \qquad x^TB_0x' = 0,

and the Rayleigh quotient on :math:`x + hx'` matches :math:`\lambda(k_0 + h)`
to :math:`O(h^4)`, so it gives :math:`\lambda''` and :math:`\lambda'''` with
no further solve. With :math:`\beta = \sqrt{-\lambda}`, the group index is
:math:`n_g = \beta'(k_0)` and the dispersion follows from
:math:`\beta''` and :math:`\beta'''` (`ModeDerivative.dispersion`).

The same formulas serve a full-space mode and a reduced one; only the solve
for :math:`x'` differs, so it is passed in (`gmres_on` for a full-space mode).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import scipy.constants
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gfem._engine.taylor import ORDER, Taylor

__all__ = [
    "ModeDerivative",
    "gmres_on",
    "mode_derivative",
    "piece_series",
    "rayleigh_series",
    "residual",
    "right_hand_side",
    "slope",
]

# GMRES on the shift-invert factor converged within 24 iterations on every
# section tried, anticrossings included, so 4 restarts of 50 are a wide margin
_RESTART = 50
_CYCLES = 4
# a true residual this many times the GMRES tolerance means the preconditioned
# residual misled it: solve directly instead
_DIRECT_SOLVE_RATIO = 100.0


class Solve(Protocol):
    """The solve for a mode derivative (`gmres_on`)."""

    def __call__(
        self,
        a0: Any,
        b0: Any,
        lam: complex,
        rhs: np.ndarray,
        x: np.ndarray,
        *,
        size: float,
    ) -> tuple[np.ndarray, float]:
        r""":math:`x'` with :math:`x^TB_0x' = 0`, and its relative residual.

        ``size`` is the right-hand side's uncancelled size (`residual`).
        """
        ...


class Preconditioner(Protocol):
    """A solve near the pencil's shift (`gfem._linalg.superlu.ShiftInvert`)."""

    def solve(self, r: np.ndarray) -> np.ndarray:
        r"""An approximate :math:`(A - \sigma B)^{-1}r`."""
        ...


@dataclasses.dataclass(frozen=True)
class ModeDerivative:
    r"""One mode of one exact solve and its derivative in :math:`k_0`.

    Attributes:
        k0: The wavenumber, per local unit.
        lam: :math:`\lambda = \gamma^2` of the mode.
        x: The mode.
        x1: Its derivative :math:`x'` in :math:`k_0`, with
            :math:`x^TB_0x' = 0`.
        coefficients: The pencil's Taylor coefficients at ``k0``, shape (J, 4).
        lam_taylor: :math:`\lambda^{(n)}(k_0)/n!` for :math:`n \le 3`.
        residual: The relative residual of the :math:`x'` equation
            (`residual`).
    """

    k0: float
    lam: complex
    x: np.ndarray
    x1: np.ndarray
    coefficients: np.ndarray
    lam_taylor: np.ndarray
    residual: float

    def dispersion(self, scale: float) -> tuple[complex, complex, complex]:
        r"""Group index, :math:`\beta_2` (s²/m) and :math:`\beta_3` (s³/m).

        :math:`\beta = \sqrt{-\lambda}` (principal root, as
        :math:`\gamma = i\beta`) per local unit of ``scale`` metres, so
        :math:`n_g = \beta'(k_0)`, :math:`\beta_2 = \beta''(k_0)\,s/c^2` and
        :math:`\beta_3 = \beta'''(k_0)\,s^2/c^3` with :math:`s` the scale.
        """
        c = scipy.constants.c
        beta = (-Taylor(self.lam_taylor)).sqrt().c
        return (
            complex(beta[1]),
            complex(2 * beta[2] * scale / c**2),
            complex(6 * beta[3] * scale**2 / c**3),
        )


def mode_derivative(
    a: Sequence[Any],
    b: Sequence[Any],
    coefficients: np.ndarray,
    lam: complex,
    x: np.ndarray,
    *,
    solve: Solve,
    k0: float,
) -> ModeDerivative:
    r"""The mode derivative of the eigenpair :math:`(\lambda, x)` at ``k0``.

    Its series of :math:`\lambda` holds ``lam`` itself, :math:`\lambda'` by
    Hellmann-Feynman, then :math:`\lambda''/2` and :math:`\lambda'''/6` from
    the Rayleigh series (`rayleigh_series`).

    Args:
        a: The pencil's pieces :math:`A_j` (sparse or dense).
        b: Its pieces :math:`B_j`.
        coefficients: :math:`c_{jn}`, shape (J, 4), as
            `gfem._engine.port_mode.Pencil.taylor` at ``k0``.
        lam: The eigenvalue.
        x: The eigenvector.
        solve: The solve for :math:`x'` (`Solve`).
        k0: The wavenumber, per local unit.
    """
    a_n = piece_series(a, coefficients)
    b_n = piece_series(b, coefficients)
    bx, a1x, b1x = b_n[0] @ x, a_n[1] @ x, b_n[1] @ x
    lam1 = slope(x, a1x, bx, b1x, lam)
    rhs, size = right_hand_side(a1x, bx, b1x, lam, lam1)
    x1, eta = solve(a_n[0], b_n[0], lam, rhs, x, size=size)
    series = rayleigh_series(a_n, b_n, x, x1)
    return ModeDerivative(
        k0=k0,
        lam=lam,
        x=x,
        x1=x1,
        coefficients=coefficients,
        lam_taylor=np.array([lam, lam1, series[2], series[3]], dtype=complex),
        residual=eta,
    )


def piece_series(pieces: Sequence[Any], coefficients: np.ndarray) -> list[Any]:
    r""":math:`M_n = \sum_j c_{jn}M_j` for each order :math:`n \le 3`.

    ``pieces`` holds :math:`M_j` (sparse or dense, any shape) and
    ``coefficients`` the matching rows :math:`c_{jn}`, shape (J, 4).
    """
    return [
        sum(c * m for c, m in zip(coefficients[:, n], pieces, strict=True))
        for n in range(ORDER + 1)
    ]


def slope(
    x: np.ndarray, a1x: np.ndarray, b0x: np.ndarray, b1x: np.ndarray, lam: complex
) -> complex:
    r"""Hellmann-Feynman: :math:`\lambda' = x^T(A_1 - \lambda B_1)x / x^TB_0x`.

    From the products :math:`A_1x`, :math:`B_0x` and :math:`B_1x`.
    """
    return complex((x @ a1x - lam * (x @ b1x)) / (x @ b0x))


def right_hand_side(
    a1x: np.ndarray, b0x: np.ndarray, b1x: np.ndarray, lam: complex, lam1: complex
) -> tuple[np.ndarray, float]:
    r"""The :math:`x'` equation's right-hand side and its uncancelled size.

    :math:`r = -(A_1 - \lambda'B_0 - \lambda B_1)x` and
    :math:`s = \|A_1x\| + |\lambda'|\,\|B_0x\| + |\lambda|\,\|B_1x\|`
    (`residual`), from the products :math:`A_1x`, :math:`B_0x` and
    :math:`B_1x`.
    """
    norm = np.linalg.norm
    rhs = -(a1x - lam1 * b0x - lam * b1x)
    size = float(norm(a1x) + abs(lam1) * norm(b0x) + abs(lam) * norm(b1x))
    return rhs, size


def rayleigh_series(
    a_n: Sequence[Any], b_n: Sequence[Any], x: np.ndarray, x1: np.ndarray
) -> np.ndarray:
    r"""The Taylor coefficients of the Rayleigh quotient on :math:`x + hx'`.

    :math:`R(h) = (x + hx')^TA(h)(x + hx') / (x + hx')^TB(h)(x + hx')`, with
    :math:`A(h) = \sum_n h^n A_n` from ``a_n`` and :math:`B(h)` from ``b_n``;
    returns :math:`R^{(n)}(0)/n!` for :math:`n \le 3`.
    """
    vectors = (x, x1)

    def quadratic(pieces: Sequence[Any]) -> list[complex]:
        products = [[m @ v for v in vectors] for m in pieces]
        out = []
        for n in range(ORDER + 1):
            acc = 0j
            for p in range(2):
                for r in range(2):
                    q = n - p - r
                    if 0 <= q < len(pieces):
                        acc += vectors[p] @ products[q][r]
            out.append(acc)
        return out

    num, den = quadratic(a_n), quadratic(b_n)
    out: list[complex] = []
    for n in range(ORDER + 1):
        acc = num[n] - sum(out[m] * den[n - m] for m in range(n))
        out.append(acc / den[0])
    return np.array(out, dtype=complex)


def gmres_on(factor: Preconditioner, rtol: float) -> Solve:
    r"""Solve for :math:`x'` by GMRES preconditioned by the exact solve's factor.

    :math:`x'` comes out of GMRES (``rtol``, 4 restarts of 50), less its
    :math:`x` component so that :math:`x^TB_0x' = 0`. The right-hand side
    loses its :math:`B_0x` component first: it has none in exact arithmetic
    (Hellmann-Feynman), but where the mode does not change with :math:`k_0`
    (a homogeneously filled guide, :math:`x' = 0`) it is rounding, and that
    component would keep the singular system from converging. When GMRES does
    not converge, or the residual (`residual`) exceeds 100 times ``rtol``, a
    direct LU of :math:`A_0 - \lambda B_0` replaces it (the matrix is
    singular in exact arithmetic only along :math:`x`, which the projection
    removes).
    Never raises: a factorization that fails keeps the GMRES result.

    Args:
        factor: The shift-invert factor of the exact solve.
        rtol: GMRES's relative tolerance.
    """

    def solve(
        a0: Any, b0: Any, lam: complex, rhs: np.ndarray, x: np.ndarray, *, size: float
    ) -> tuple[np.ndarray, float]:
        t = sp.csr_matrix(a0 - lam * b0)
        bx = b0 @ x
        rhs = rhs - bx * ((x @ rhs) / (x @ bx))
        precondition = _FactorOperator(factor, t.shape)
        z, info = spla.gmres(
            t,
            rhs,
            M=precondition,
            rtol=rtol,
            atol=0.0,
            restart=_RESTART,
            maxiter=_CYCLES,
        )
        x1 = _orthogonal(z, x, bx)
        eta = residual(a0, b0, lam, x1, rhs, size=size)
        if info == 0 and eta <= _DIRECT_SOLVE_RATIO * rtol:
            return x1, eta
        try:
            z = spla.splu(sp.csc_matrix(t)).solve(rhs)
        except RuntimeError:  # exactly singular: keep what GMRES gave
            return x1, eta
        x1 = _orthogonal(z, x, bx)
        return x1, residual(a0, b0, lam, x1, rhs, size=size)

    return solve


def residual(
    a0: Any, b0: Any, lam: complex, x1: np.ndarray, rhs: np.ndarray, *, size: float
) -> float:
    r"""The relative residual of the :math:`x'` equation.

    .. math::

        \eta' = \frac{\|(A_0 - \lambda B_0)x' - r\|}
        {\|A_0x'\| + |\lambda|\,\|B_0x'\| + s},

    with :math:`r = -(A_1 - \lambda'B_0 - \lambda B_1)x` the right-hand side
    and :math:`s = \|A_1x\| + |\lambda'|\,\|B_0x\| + |\lambda|\,\|B_1x\|`
    (``size``) its uncancelled size. Not :math:`\|r\|`: where the mode does
    not change with :math:`k_0` (:math:`x' = 0`), :math:`r` is rounding of
    those terms, and a residual relative to it would read
    :math:`10^{-2}`-:math:`10^{-1}` for a mode derivative exact to rounding.
    """
    ax, bx = a0 @ x1, b0 @ x1
    den = np.linalg.norm(ax) + abs(lam) * np.linalg.norm(bx) + size
    return float(np.linalg.norm(ax - lam * bx - rhs) / den) if den > 0 else 0.0


class _FactorOperator(spla.LinearOperator):
    """GMRES's preconditioner: ``factor.solve``."""

    def __init__(self, factor: Preconditioner, shape: tuple[int, int]) -> None:
        super().__init__(dtype=np.dtype(complex), shape=shape)
        self._factor = factor

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._factor.solve(x)


def _orthogonal(z: np.ndarray, x: np.ndarray, bx: np.ndarray) -> np.ndarray:
    r"""``z`` less its ``x`` component in the bilinear form :math:`u^TB_0v`."""
    return z - x * ((bx @ z) / (x @ bx))
