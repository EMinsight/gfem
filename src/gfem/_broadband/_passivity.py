r"""Passivity of a rational model of S: the Hamiltonian test and enforcement.

A model :math:`S(s) = D + C (sI - A)^{-1} B` on a real reference is passive
when no singular value of :math:`S(i\omega)` exceeds 1 at any
:math:`\omega \ge 0`. With :math:`\sigma_{\max}(D) < 1`, the frequencies
where a singular value crosses 1 are the imaginary eigenvalues
:math:`i\omega` of the Hamiltonian matrix (size :math:`2NP` for :math:`N`
poles and :math:`P` ports)

.. math::

    M = \begin{pmatrix}
        A - B R^{-1} D^T C & -B R^{-1} B^T \\
        C^T Q^{-1} C & -A^T + C^T D R^{-1} B^T
    \end{pmatrix},
    \qquad R = D^T D - I, \quad Q = D D^T - I,

and :math:`\sigma_{\max}` at one point between consecutive crossings labels
each interval. The full matrix is used, not the half-size one, which holds
only for symmetric S. When :math:`\sigma_{\max}(D)` is near 1 and S near
unitary over the band (a lossless line), :math:`R` and :math:`Q` are nearly
singular and the eigenvalues can lose crossings, so samples of
:math:`\sigma_{\max}` resolving every pole's peak are added to the edges.

Enforcement keeps the poles (so stability) and perturbs the residues by the
least change of S on the data frequencies (weighted as the fit was). The
singular values of :math:`D` are first clipped to :math:`1 - \delta`. Then,
at the :math:`\sigma_{\max}` peak of each violation, every singular triple
:math:`(\sigma_k, u_k, v_k)` near 1 gives a linear constraint

.. math::

    \sigma_k + \mathrm{Re}\left(u_k^H \Delta S\, v_k\right) \le 1 - \delta.

The least change subject to them is a least-distance problem, solved
exactly by nonnegative least squares (Lawson and Hanson). The peaks of
earlier iterations stay constrained, and the test is repeated until no
violation remains. The minimum-norm change of a symmetric model is
symmetric, so a reciprocal model stays reciprocal.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.optimize

from gfem._broadband._poles import (
    Poles,
    basis,
    coefficients,
    response,
    stack,
    state_matrix,
)
from gfem._linalg.gmres import _ONE_BLAS_THREAD, scipy_openblas
from gfem._text import hertz, hertz_band
from gfem.errors import PassivityEnforcementError

__all__ = ["MARGIN", "enforce", "violations", "worst_violation"]

#: Margin below 1 that enforcement leaves on the singular values.
MARGIN = 1e-4
#: Enforcement iterations (perturbation and test) at most.
_ITERATIONS = 30
#: Singular values of D within this of 1 are tested against 1 plus twice it,
#: where the Hamiltonian at exactly 1 is singular (a short circuit, D = -I).
_SLACK = 1e-9
#: Eigenvalues whose real part is below this share of their modulus are
#: candidate crossings. Generous: a candidate where the label of the
#: neighbouring intervals does not change is not an edge.
_IMAGINARY = 1e-4
#: Constraints are set on the singular values above 1 minus this many margins.
_ACTIVE = 10
#: Samples of the largest singular value in a violation, then around its peak.
_PEAK_SAMPLES = (64, 33)
#: Evenly spaced samples of the passivity test's guard to twice the largest pole.
_EVEN_SAMPLES = 4001
#: The guard's samples around each pair's resonance, in damping widths.
_AROUND = np.linspace(-4.0, 4.0, 17)


def violations(
    poles: np.ndarray, residues: np.ndarray, d: np.ndarray
) -> tuple[tuple[float, float], ...]:
    r"""Intervals (rad/s) where the largest singular value of S exceeds 1.

    Args:
        poles: ``(n_poles,)`` stable poles, rad/s; real ones and conjugate
            pairs, each pair adjacent with its positive-imaginary pole first.
        residues: ``(n_poles, P, P)``, conjugate for a pair.
        d: ``(P, P)`` real.

    Returns:
        The intervals in increasing order; the last one ends at infinity
        when :math:`\sigma_{\max}(D) > 1`.
    """
    model = _Model.of(poles, residues, d)
    with _ONE_BLAS_THREAD.held(scipy_openblas()):
        bands = model.violations()
    return tuple((lo * model.scale, hi * model.scale) for lo, hi in bands)


def worst_violation(
    poles: np.ndarray,
    residues: np.ndarray,
    d: np.ndarray,
    bands: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """The largest singular value of S over ``bands`` (rad/s), and where.

    An interval that ends at infinity is searched over ten times its start
    and at infinity (where S is ``d``); the place is then ``inf``.
    """
    model = _Model.of(poles, residues, d)
    best = (-math.inf, math.nan)
    for band in bands:
        lo, hi = band[0] / model.scale, band[1] / model.scale
        if math.isinf(hi):
            sigma_d = float(scipy.linalg.svdvals(model.d, check_finite=False)[0])
            best = max(best, (sigma_d, math.inf))
        w = model.peak(lo, 10 * (lo + 1) if math.isinf(hi) else hi)
        best = max(best, (float(model.sigma_max(np.array([w]))[0]), w * model.scale))
    return best


def enforce(
    poles: np.ndarray,
    residues: np.ndarray,
    d: np.ndarray,
    omega: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Residues and ``d`` of a passive model, closest to the given one.

    Args:
        poles, residues, d: The model, as for `violations`.
        omega: The data frequencies, rad/s, on which the change of S is
            minimized; fewer basis functions than twice their number.
        weights: Positive weights of the data frequencies.

    Returns:
        The new residues (same pole order) and ``d``.

    Raises:
        PassivityEnforcementError: Violations remain after the iterations,
            or the constraints cannot be met; the message names the
            remaining intervals in Hz.
    """
    model = _Model.of(poles, residues, d)
    x = 1j * np.asarray(omega, float) / model.scale
    with _ONE_BLAS_THREAD.held(scipy_openblas()):
        model = model.with_d(_clip(model.d))
        # the change of S on the data is |Phi dc| per element: with Phi = QR,
        # y = R dc turns the least change into a least-norm problem
        (r,) = scipy.linalg.qr(
            stack(basis(x, model.poles) * np.asarray(weights, float)[:, None]),
            mode="r",
            check_finite=False,
        )
        r = r[: model.c.shape[1]]
        peaks = np.zeros(0)
        for _ in range(_ITERATIONS):
            bands = model.violations()
            if not bands:
                break
            peaks = np.union1d(peaks, [model.peak(lo, hi) for lo, hi in bands])
            g, h = model.constraints(peaks)
            if not h.size:
                break
            n_el, n = model.c.shape
            gy = scipy.linalg.solve_triangular(
                r, g.reshape(-1, n).T, trans="T", check_finite=False
            ).T.reshape(h.size, n_el * n)
            y = _least_distance(gy, h)
            if y is None:
                break
            dc = scipy.linalg.solve_triangular(
                r, y.reshape(n_el, n).T, check_finite=False
            ).T
            model = model.with_c(model.c + dc)
        bands = model.violations()
    if bands:
        found = ", ".join(
            _hz_band(lo * model.scale, hi * model.scale) for lo, hi in bands
        )
        raise PassivityEnforcementError(
            "Passivity enforcement could not make the model passive in "
            f"{_ITERATIONS} iterations: the largest singular value of S still "
            f"exceeds 1 over {found}. Fit with enforce_passivity=False to inspect "
            "the model (its passivity_violations and network()), and check the "
            "data's reference impedance and sampling."
        )
    return model.residues(), model.d


def _hz_band(lo: float, hi: float) -> str:
    """An interval in rad/s, written in Hz."""
    lo, hi = lo / (2 * math.pi), hi / (2 * math.pi)
    return f"above {hertz(lo)}" if math.isinf(hi) else hertz_band(lo, hi)


def _clip(d: np.ndarray) -> np.ndarray:
    """``d`` with its singular values clipped to 1 minus the margin."""
    u, sv, vt = scipy.linalg.svd(d, check_finite=False)
    if sv.max(initial=0.0) <= 1 - MARGIN:
        return d
    return (u * np.minimum(sv, 1 - MARGIN)) @ vt


def _least_distance(g: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    r"""The least-norm :math:`y` with :math:`G y \le h`, or None if none.

    Lawson and Hanson's reduction: with :math:`E = -G`, :math:`f = -h`, the
    nonnegative least-squares solution :math:`u` of
    :math:`\begin{pmatrix} E^T \\ f^T \end{pmatrix} u = e_{n+1}` has residual
    :math:`r`, and :math:`y = -r_{1:n} / r_{n+1}`; :math:`r_{n+1} = 0`
    means the constraints are inconsistent.
    """
    a = np.vstack([-g.T, -h[None, :]])
    e = np.zeros(a.shape[0])
    e[-1] = 1.0
    u, _ = scipy.optimize.nnls(a, e, maxiter=50 * h.size + 100)
    res = a @ u - e
    if abs(res[-1]) < 1e-14:
        return None
    return -res[:-1] / res[-1]


class _Model:
    """A pole-residue model as real coefficients, frequencies scaled to O(1).

    ``c[el, k]`` is the coefficient of basis function ``k`` (real poles,
    then the real and imaginary parts of each pair's residue) in element
    ``el = i * P + j`` of S.
    """

    def __init__(
        self,
        poles: Poles,
        c: np.ndarray,
        d: np.ndarray,
        scale: float,
        layout: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        self.poles = poles
        self.c = c
        self.d = d
        self.scale = scale
        self._layout = layout

    @classmethod
    def of(cls, poles: np.ndarray, residues: np.ndarray, d: np.ndarray) -> _Model:
        """The model of physical poles (rad/s), residues and ``d``."""
        poles = np.asarray(poles, complex).reshape(-1)
        residues = np.asarray(residues, complex)
        d = np.asarray(d, float)
        scale = float(np.abs(poles).max(initial=0.0)) or 1.0
        pole_set, c = coefficients(poles, residues, scale)
        real = np.flatnonzero(poles.imag == 0)
        pairs = np.flatnonzero(poles.imag > 0)
        return cls(pole_set, c, d, scale, (real, pairs, poles.size))

    def with_c(self, c: np.ndarray) -> _Model:
        """The same model with other coefficients."""
        return _Model(self.poles, c, self.d, self.scale, self._layout)

    def with_d(self, d: np.ndarray) -> _Model:
        """The same model with another ``d``."""
        return _Model(self.poles, self.c, d, self.scale, self._layout)

    def residues(self) -> np.ndarray:
        """Physical residues ``(n_poles, P, P)``, in the original pole order."""
        real, pairs, n_poles = self._layout
        n_ports = self.d.shape[0]
        c = self.c * self.scale
        out = np.zeros((n_poles, n_ports * n_ports), complex)
        out[real] = c[:, : real.size].T
        pair = c[:, real.size :].T.reshape(pairs.size, 2, n_ports * n_ports)
        out[pairs] = pair[:, 0] + 1j * pair[:, 1]
        out[pairs + 1] = np.conj(out[pairs])
        return out.reshape(n_poles, n_ports, n_ports)

    def s(self, w: np.ndarray) -> np.ndarray:
        """S ``(len(w), P, P)`` at scaled angular frequencies ``w``."""
        return response(1j * np.asarray(w, float), self.poles, self.c, self.d)

    def sigma_max(self, w: np.ndarray) -> np.ndarray:
        """The largest singular value of S at each of ``w``."""
        if not np.size(w):
            return np.zeros(0)
        return np.linalg.svd(self.s(w), compute_uv=False)[:, 0]

    def violations(self) -> list[tuple[float, float]]:
        """Intervals of scaled frequency where the largest singular value > 1."""
        sd = scipy.linalg.svdvals(self.d, check_finite=False)
        gamma = 1 + 2 * _SLACK if np.any(np.abs(sd - 1) <= _SLACK) else 1.0
        # the samples catch the crossings the eigenvalues lose when the
        # Hamiltonian is ill-conditioned (sigma(D) near 1, S near unitary
        # over the band, as for a lossless line); between true crossings
        # they only split an interval of one label
        inner = np.union1d(self._crossings(gamma), self._samples())
        edges = np.concatenate([[0.0], inner[inner > 0], [math.inf]])
        mids = 0.5 * (edges[:-2] + edges[1:-1])
        over = np.append(self.sigma_max(mids) > gamma, sd.max(initial=0.0) > gamma)
        bands: list[tuple[float, float]] = []
        for k in np.flatnonzero(over):
            lo, hi = float(edges[k]), float(edges[k + 1])
            if bands and bands[-1][1] == lo:
                bands[-1] = (bands[-1][0], hi)
            else:
                bands.append((lo, hi))
        return bands

    def _samples(self) -> np.ndarray:
        """Scaled frequencies that resolve every pole's peak.

        Evenly spaced samples to twice the largest pole (the scale), and around
        each pair's resonance a few points per damping width.
        """
        if not self.poles.size:
            return np.zeros(0)
        pairs = self.poles.pairs
        around = pairs.imag[:, None] + np.abs(pairs.real)[:, None] * _AROUND
        return np.union1d(np.linspace(0.0, 2.0, _EVEN_SAMPLES), around.reshape(-1))

    def _crossings(self, gamma: float) -> np.ndarray:
        """Candidate frequencies where a singular value of S equals ``gamma``."""
        if not self.poles.size:
            return np.zeros(0)
        n_ports = self.d.shape[0]
        a_p, b_p = state_matrix(self.poles)
        n = a_p.shape[0]
        a = np.kron(np.eye(n_ports), a_p)
        b = np.kron(np.eye(n_ports), b_p[:, None])
        # row i of C holds the coefficients of S[i, j] for j = 0, 1, ... in turn
        c = self.c.reshape(n_ports, n_ports * n) / gamma
        d = self.d / gamma
        eye = np.eye(n_ports)
        r_inv = scipy.linalg.inv(d.T @ d - eye, check_finite=False)
        q_inv = scipy.linalg.inv(d @ d.T - eye, check_finite=False)
        m = np.block(
            [
                [a - b @ r_inv @ d.T @ c, -b @ r_inv @ b.T],
                [c.T @ q_inv @ c, -a.T + c.T @ d @ r_inv @ b.T],
            ]
        )
        lam = scipy.linalg.eigvals(m, check_finite=False, overwrite_a=True)
        near = np.abs(lam.real) <= _IMAGINARY * np.abs(lam) + 1e-12
        w = np.unique(np.abs(lam[near].imag))
        return w[w > 0]

    def peak(self, lo: float, hi: float) -> float:
        """Where the largest singular value peaks in a finite interval."""
        coarse, fine = _PEAK_SAMPLES
        w = np.linspace(lo, hi, coarse + 2)[1:-1]
        k = int(np.argmax(self.sigma_max(w)))
        w = np.linspace(w[max(k - 1, 0)], w[min(k + 1, w.size - 1)], fine)
        return float(w[np.argmax(self.sigma_max(w))])

    def constraints(self, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Linearized bounds :math:`g \cdot \Delta c \le h` at ``peaks``.

        One row per singular value within the active range of 1 at each
        peak; ``g`` is ``(rows, n_el * N)``.
        """
        n_ports = self.d.shape[0]
        phi = basis(1j * peaks, self.poles)
        u, sv, vh = np.linalg.svd(self.s(peaks))
        rows, rhs = [], []
        for f, k in zip(*np.nonzero(sv > 1 - _ACTIVE * MARGIN), strict=True):
            coef = np.outer(u[f, :, k].conj(), vh[f, k].conj()).reshape(-1)
            rows.append(np.real(coef[:, None] * phi[f][None, :]).reshape(-1))
            rhs.append(1 - MARGIN - sv[f, k])
        size = n_ports * n_ports * self.c.shape[1]
        return np.asarray(rows).reshape(-1, size), np.asarray(rhs, float)
