"""Deflated GMRES on an eigen operator's live factors, and the mode polish.

At an eigenvalue ``k`` of ``T`` with right eigenvector ``x`` and left one
``y`` (``x`` when complex symmetric), ``T(k)`` is singular: ``T(k) z = b``
has a solution only for ``y^T b = 0``, and it is defined up to a multiple
of ``x``. `deflated_solve` solves it with the two rank-one projections of
quantum-metrics ticket 03, ``t = T'(k) x``:

- ``P v = v - t (y^T v) / (y^T t)`` on right-hand sides and residuals: onto
  the range of ``T(k)``, along ``t``;
- ``Q v = v - x (t^T v) / (t^T x)`` on iterates: the gauge ``t^T z = 0``.

GMRES runs on ``Q T(sigma)^-1 P T(k) Q``, left-preconditioned by the
factors the eigen operator holds at its current shift ``sigma`` near ``k``
(no refactorization): only `EigenOperator.solve`, ``apply`` and
``derivative`` are used. Its stopping rule is ADR 0104's (amendment of
ticket 19): an Arnoldi stop at ``stop |P b|``, then one more solve gives
``rho = |P(b - T z)| / |z|``, accepted at ``rho <= accept``; otherwise it
restarts from the iterate with the absolute tolerance ``stop |z|``. After
`APPLY_CAP` applies (factor solves) without acceptance the operator reshifts
once to ``k (1 + d)``, ``d = min(1e-3, 1e-2 gap)``, never exactly at ``k``,
and stays there; after another `APPLY_CAP`, `LinearSolverConvergenceError`.
No call passes a frequency to the refactorization, so ADR 0094's factor reuse
never serves it.

`polish` takes Newton steps on a converged mode with the same solve: mode
polishing, the forward's correction of every mode in mode-field reach (ADR
0104, amendment of ticket 19).

Columns keep the roles of `gfem._linalg.layout`: ``x``, ``y``, the iterates
and GMRES's preconditioned basis are states; ``b``, ``t`` and residuals are
loads. :math:`y^T v` and :math:`t^T v` pair a load with a state, and the basis's
products and norms are the eigen operator's layout's state products, so every
value a branch reads is agreed across ranks.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from gfem import _mpi
from gfem._eigen.refine import Refined
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem.errors import EigenmodeConvergenceError, LinearSolverConvergenceError

if TYPE_CHECKING:
    from gfem._eigen.operator import EigenOperator

__all__ = ["APPLY_CAP", "Deflated", "deflated_solve", "polish", "reshift_offset"]

#: The adjoint solve's Arnoldi stop (relative to ``|P b|``, then the
#: absolute restart tolerance relative to ``|z|``) and its acceptance of
#: ``rho``.
STOP, ACCEPT = 1e-10, 1e-8
#: The polish's stop and acceptance.
POLISH_STOP, POLISH_ACCEPT = 1e-6, 1e-4
#: A polish step's correction (relative to the mode's norm) that ends the
#: polish, and the most steps taken.
POLISH_LIMIT, POLISH_STEPS = 1e-4, 3
#: Factor solves on one factorization before a reshift, and after it
#: before the error.
APPLY_CAP = 60
#: The largest reshift offset (relative): a mode without a neighbour's.
RESHIFT = 1e-3
#: The reshift offset's fraction of the mode's relative gap.
GAP_FRACTION = 1e-2


@dataclasses.dataclass(frozen=True)
class Deflated:
    """An accepted deflated solve.

    Attributes:
        z: The solution, with ``t^T z = 0``.
        rho: ``|P(b - T z)| / |z|``, preconditioned (0 for ``P b = 0``).
        applies: Factor solves spent.
        reshifted: Whether the operator was reshifted (it stays there).
    """

    z: np.ndarray
    rho: float
    applies: int
    reshifted: bool


def reshift_offset(gap: float) -> float:
    """``d = min(1e-3, 1e-2 gap)``: 1e-3 for a mode without a neighbour."""
    return min(RESHIFT, GAP_FRACTION * gap)


class _System:
    """``T(k)`` deflated at the mode ``(k, x, y)``, on the operator's factors."""

    def __init__(
        self, eop: EigenOperator, k: complex, x: np.ndarray, y: np.ndarray | None
    ) -> None:
        self.eop = eop
        self.k = complex(k)
        self.x = np.where(eop.free, x, 0)
        self.y = self.x if y is None else np.where(eop.free, y, 0)
        self.layout = eop.layout
        self.t = eop.derivative(self.k, self.x)
        self.yt = self.layout.pair(self.t, self.y)
        self.tx = self.layout.pair(self.t, self.x)
        self.applies = 0

    def residual(self, v: np.ndarray) -> np.ndarray:
        """``P v``: onto the range of ``T(k)``, along ``t``."""
        return v - self.t * (self.layout.pair(v, self.y) / self.yt)

    def gauge(self, v: np.ndarray) -> np.ndarray:
        """``Q v``: ``t^T (Q v) = 0``, along ``x``."""
        return v - self.x * (self.layout.pair(self.t, v) / self.tx)

    def precondition(self, v: np.ndarray) -> np.ndarray:
        """``Q T(sigma)^-1 P v``: one factor solve."""
        self.applies += 1
        return self.gauge(self.eop.solve(self.residual(v)))

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """The left-preconditioned deflated operator on ``v``."""
        return self.precondition(self.eop.apply(self.k, self.gauge(v)))


def deflated_solve(
    eop: EigenOperator,
    x: np.ndarray,
    y: np.ndarray | None,
    b: np.ndarray,
    *,
    k: complex,
    gap: float,
    mode: int,
    stop: float = STOP,
    accept: float = ACCEPT,
    reshift: bool = True,
    purpose: str = "adjoint solve",
) -> Deflated:
    """``T(k) z = P b`` with ``t^T z = 0`` at the mode ``(k, x, y)``.

    Args:
        eop: The eigen operator, factorized at a shift near ``k``.
        x: The mode's right eigenvector (any scale and phase).
        y: Its left eigenvector, ``None`` for ``x`` (complex symmetric).
        b: The right-hand side (its component along ``t`` is dropped).
        k: The mode's eigenvalue.
        gap: The mode's relative gap to its neighbours (``inf`` without
            any): the reshift offset's scale.
        mode: The mode's index, for the error message.
        stop: The Arnoldi stop, relative to ``|P b|``, and the restart's
            absolute tolerance, relative to ``|z|``.
        accept: The largest ``rho`` accepted.
        reshift: Whether the mode may still reshift the operator (each mode
            reshifts it at most once).
        purpose: What the solve is for, for the error message.

    Raises:
        LinearSolverConvergenceError: Not accepted after `APPLY_CAP` applies,
            the reshift and `APPLY_CAP` more (or `APPLY_CAP` without
            ``reshift``).
    """
    system = _System(eop, k, x, y)
    b = np.where(eop.free, b, 0).astype(complex)
    z = np.zeros_like(b)
    reshifted = False
    while True:
        z, rho, accepted = _cycles(system, b, z, stop=stop, accept=accept)
        if accepted:
            return Deflated(z, rho, system.applies, reshifted)
        if reshifted or not reshift:
            break
        eop.reshift(system.k * (1 + reshift_offset(gap)))
        reshifted = True
    after = (
        f", a reshift to k (1 + {reshift_offset(gap):g}) and {APPLY_CAP} more"
        if reshifted
        else ""
    )
    raise LinearSolverConvergenceError(
        f"The deflated GMRES of the {purpose} of mode {mode} was not accepted: "
        f"rho = {rho:.3g} > {accept:g} after {APPLY_CAP} applies{after}. The mode "
        "may be nearly degenerate with another (its relative gap is "
        f"{gap:.3g})."
    )


def _cycles(
    system: _System, b: np.ndarray, z: np.ndarray, *, stop: float, accept: float
) -> tuple[np.ndarray, float, bool]:
    """Restarted GMRES from ``z`` on the current factors, `APPLY_CAP` applies."""
    eop, k = system.eop, system.k
    norm = system.layout.state_norm
    start = system.applies
    # z is a state: a DOF's holders agree, but a rank may hold none of z's
    nonzero = _mpi.agree_any(system.layout.comm, flag=bool(z.any()))
    r = system.precondition(b - eop.apply(k, z) if nonzero else b)
    size = norm(r)
    if size == 0:
        return z, 0.0, True
    tolerance = stop * size
    rho = math.inf
    while system.applies - start < APPLY_CAP:
        left = APPLY_CAP - (system.applies - start)
        z = system.gauge(
            z + _gmres(system.matvec, r, tolerance, left, layout=system.layout)
        )
        r = system.precondition(b - eop.apply(k, z))
        size = norm(z)
        rho = float(norm(r) / size) if size > 0 else math.inf
        if rho <= accept:
            return z, rho, True
        tolerance = stop * size
    return z, rho, False


def _gmres(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    tolerance: float,
    steps: int,
    *,
    layout: DofLayout = ONE_PROCESS,
) -> np.ndarray:
    """GMRES from zero, stopped when its Arnoldi estimate is at most ``tolerance``.

    Classical Gram-Schmidt twice; the least-squares problem is solved anew
    each step (at most `APPLY_CAP` columns). ``b`` and ``matvec``'s values are
    states: the basis's products are ``layout``'s.
    """
    norm = layout.state_norm
    beta = norm(b)
    if beta <= tolerance:
        return np.zeros_like(b)
    basis = [b / beta]
    h = np.zeros((steps + 1, steps), dtype=complex)
    e1 = np.zeros(steps + 1, dtype=complex)
    e1[0] = beta
    coefficients = np.zeros(1, dtype=complex)
    for j in range(steps):
        w = matvec(basis[j])
        q = np.array(basis)
        before = norm(w)
        for _ in range(2):
            c = layout.state_gram(q.T, w)  # q.conj() @ w
            w = w - c @ q
            h[: j + 1, j] += c
        h[j + 1, j] = norm(w)
        coefficients = np.linalg.lstsq(h[: j + 2, : j + 1], e1[: j + 2], rcond=None)[0]
        estimate = np.linalg.norm(h[: j + 2, : j + 1] @ coefficients - e1[: j + 2])
        if estimate <= tolerance or abs(h[j + 1, j]) <= 1e-14 * before:
            break
        basis.append(w / h[j + 1, j])
    return coefficients @ np.array(basis[: len(coefficients)])


def polish(eop: EigenOperator, mode: Refined, gap: float, *, index: int) -> Refined:
    """Mode polishing: Newton corrections of a converged mode on live factors.

    Each step solves ``T(k) d = -T(k) x`` (`deflated_solve`, stop
    `POLISH_STOP`, accept `POLISH_ACCEPT`), sets ``x <- x + d`` and takes
    ``k`` from the Rayleigh functional. Steps repeat while ``|d| > 1e-4
    |x|``, at most `POLISH_STEPS`. The mode keeps its refinement's step
    count; its error is the polished pair's backward error.

    Raises:
        EigenmodeConvergenceError: The last step's correction is still
            above ``1e-4 |x|``.
        LinearSolverConvergenceError: A step's solve was not accepted.
    """
    norm = eop.layout.state_norm
    x = np.where(eop.free, mode.x, 0)
    x = x / norm(x)
    k = complex(mode.k)
    reshift = True
    size = math.inf
    for _ in range(POLISH_STEPS):
        step = deflated_solve(
            eop,
            x,
            None,
            -eop.apply(k, x),
            k=k,
            gap=gap,
            mode=index,
            stop=POLISH_STOP,
            accept=POLISH_ACCEPT,
            reshift=reshift,
            purpose="polish",
        )
        reshift = reshift and not step.reshifted
        size = float(norm(step.z))  # relative: |x| = 1
        x = x + step.z
        x = x / norm(x)
        k = eop.rayleigh(x, k)
        if size <= POLISH_LIMIT:
            return dataclasses.replace(mode, k=k, x=x, error=eop.backward_error(k, x))
    raise EigenmodeConvergenceError(
        f"Polishing mode {index} did not converge: its Newton correction is "
        f"still {size:.3g} of its norm after {POLISH_STEPS} steps (at most "
        f"{POLISH_LIMIT:g} ends the polish). The mode may be nearly degenerate "
        f"with another (its relative gap is {gap:.3g}); tighten "
        "SolverOptions.eigen_tolerance or move Eigenmode.target."
    )
