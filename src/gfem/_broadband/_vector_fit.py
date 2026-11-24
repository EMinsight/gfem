r"""Fast relaxed vector fitting of a multiport S on its data frequencies.

One pole set is shared by every element of S (no symmetry assumed), with
real poles and conjugate pairs carrying conjugate residues, a fitted
constant :math:`D` and no proportional term:

.. math::

    S_{ij}(s) = d_{ij} + \sum_n \frac{r_{ij,n}}{s - p_n}, \qquad s = i\omega.

Each relocation step solves for a scaling function
:math:`\sigma(s) = \tilde d + \sum_n \tilde r_n / (s - a_n)` on the current
poles :math:`a_n` with :math:`\sigma S` rational on the same poles; the new
poles are the zeros of :math:`\sigma`. :math:`\tilde d` is free (relaxed
vector fitting) and one extra equation, :math:`\mathrm{Re} \sum_k
\sigma(s_k) = K`, excludes the zero solution. The element unknowns drop out
by a QR factorization of each element's block (fast vector fitting), so the
shared unknowns come from one small least-squares problem. Unstable poles
are flipped into the left half-plane. Frequencies are scaled by the highest
one, so the arithmetic is on numbers of order one.

The order rises in steps of four poles until the largest per-element RMS
error is within the tolerance; poles whose in-band contribution is below a
tenth of the tolerance are then removed and the residues refitted, since an
order above the need pushes spurious poles far outside the band.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg

from gfem._broadband._reference import constant_real
from gfem._checks import is_staging, is_traced, static_float
from gfem._linalg.gmres import _ONE_BLAS_THREAD, scipy_openblas
from gfem._text import hertz, hertz_band
from gfem.broadband import RationalModel
from gfem.errors import (
    NonDifferentiableError,
    NonPassiveDataWarning,
    RationalFitNotConvergedWarning,
)

if TYPE_CHECKING:
    from gfem.results import NetworkParameters

__all__ = ["PoleResidueFit", "fit_rational", "vector_fit"]

#: Fewest distinct data frequencies a fit accepts.
_MIN_FREQUENCIES = 6
#: Default cap on the pole count (also capped below the frequency count).
_MAX_POLES = 100
_NOT_DIFFERENTIABLE = (
    "NetworkParameters.fit_rational relocates poles iteratively and picks its "
    "order in steps, so gfem does not differentiate it: S, z_ref, z0, "
    "weights and the options must be concrete values, and the fit cannot be "
    "staged by jax.jit. Fit outside jax.grad, jax.jit or jax.vmap, or wrap S "
    "in jax.lax.stop_gradient."
)

#: Pole relocations per order at most.
_ITERATIONS = 20
#: An order's relocation stops once the error changes by less than this share.
_STALL = 1e-3
#: Smallest relaxed constant of the scaling function, below which it is fixed.
_RELAX_GUARD = 1e-8
#: Band ratio from which the initial poles are spaced geometrically.
_GEOMETRIC_SPAN = 100.0
#: Poles whose largest in-band contribution is below this share of the
#: tolerance are removed after the fit.
_PRUNE = 0.1


def fit_rational(
    net: NetworkParameters,
    *,
    tolerance: Any,
    max_poles: Any,
    weights: Any,
    enforce_passivity: bool,
    z0: Any,
) -> RationalModel:
    """`NetworkParameters.fit_rational`: checks, reference, fit and model."""
    for name, value in (
        ("max_poles", max_poles),
        ("enforce_passivity", enforce_passivity),
    ):
        if is_traced(value):
            raise NonDifferentiableError(
                f"fit_rational {name} is a fixed value and gfem does not "
                f"differentiate with respect to it. {_NOT_DIFFERENTIABLE}"
            )
    tol = static_float(tolerance, "fit_rational tolerance", _NOT_DIFFERENTIABLE)
    # under jax.jit even a closed-over network's S becomes a tracer once
    # indexed, so staging is refused as such
    if is_staging() or is_traced((net.values, net.z_ref, z0, weights)):
        raise NonDifferentiableError(_NOT_DIFFERENTIABLE)
    if tol <= 0:
        raise ValueError(f"fit_rational tolerance must be > 0, got {tol!r}.")
    s = np.asarray(net._square("fit_rational()"))  # noqa: SLF001
    f = np.asarray(net.frequency, dtype=float).reshape(-1)
    _check_frequencies(f)
    n_max = _max_poles(max_poles, f.size)
    w = _weights(weights, f.size)
    s, z_ref, nominal = _reference(net, s, z0)
    if not np.isfinite(s).all():
        raise ValueError(
            "fit_rational needs finite S; the network has NaN or infinite values."
        )
    from gfem._broadband import _passivity

    order = np.argsort(f)
    omega, s = 2 * math.pi * f[order], s[order]
    # the factorizations are small: a threaded BLAS made them tens of times
    # slower on a loaded host, so SciPy's OpenBLAS runs them on one thread
    with _ONE_BLAS_THREAD.held(scipy_openblas()):
        fit = vector_fit(omega, s, tolerance=tol, max_poles=n_max, weights=w[order])
        bands = _passivity.violations(fit.poles, fit.residues, fit.d)
    band = (float(f.min()), float(f.max()))
    if not fit.converged:
        warnings.warn(
            f"fit_rational reached max_poles={n_max} over {hertz_band(*band)} "
            f"with an error of {fit.error:.3g}, above the tolerance {tol:.3g} "
            f"({fit.poles.size} poles kept). Raise max_poles, fit a narrower "
            "band, or remove a known delay first: a long line needs about one "
            "pole pair per phase wrap in the band, and a wave port's reference "
            "plane moves with DrivenResult.with_offsets.",
            RationalFitNotConvergedWarning,
            stacklevel=3,
        )
    residues, d, error = fit.residues, fit.d, fit.error
    if enforce_passivity:
        residues, d = _passivity.enforce(fit.poles, residues, d, omega, w[order])
        enforced = _evaluate(omega, fit.poles, residues, d)
        error = _rms(enforced - s)
        moved = _rms(enforced - _evaluate(omega, fit.poles, fit.residues, fit.d))
        if moved > tol:
            sigma, where = _passivity.worst_violation(
                fit.poles, fit.residues, fit.d, bands
            )
            if not bands:  # moved by the clip of d alone: name the data's
                sv = np.linalg.svd(s, compute_uv=False)[:, 0]
                sigma, where = float(sv.max()), float(omega[np.argmax(sv)])
            _warn_non_passive(sigma, where / (2 * math.pi), moved, tol, error)
    return RationalModel(
        poles=fit.poles,
        residues=residues,
        d=d,
        z_ref=z_ref,
        ports=net.ports,
        frequency_range=band,
        error=error,
        fit_error=fit.error,
        passivity_violations=tuple(
            (lo / (2 * math.pi), hi / (2 * math.pi)) for lo, hi in bands
        ),
        nominal_reference=nominal,
    )


def _evaluate(
    omega: np.ndarray, poles: np.ndarray, residues: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """S ``(K, P, P)`` of a pole-residue model at angular frequencies."""
    terms = 1 / (1j * omega[:, None] - poles[None, :])
    return d[None] + np.einsum("kn,nij->kij", terms, residues)


def _rms(diff: np.ndarray) -> float:
    """Largest per-element RMS over frequency of ``diff`` ``(K, P, P)``."""
    return float(np.sqrt(np.mean(np.abs(diff) ** 2, axis=0)).max())


def _warn_non_passive(
    sigma: float, where: float, moved: float, tol: float, error: float
) -> None:
    """`NonPassiveDataWarning`: enforcement moved S beyond the tolerance.

    ``sigma`` is the fit's largest singular value, at ``where`` Hz.
    """
    place = "at high frequency" if math.isinf(where) else f"at {hertz(where)}"
    warnings.warn(
        f"fit_rational: making the model passive changed S by {moved:.3g}, "
        f"more than the tolerance {tol:.3g}: the data are not passive on this "
        f"reference, and their fit reaches a largest singular value of "
        f"{sigma:.4g} {place}. The model returned is passive, with an error of "
        f"{error:.3g} against the data. Usual causes: a reference plane moved "
        "past a lossy line by DrivenResult.with_offsets, a nominal reference "
        "(z0= on wave ports without voltage_path) on a lossy mode, or a mesh "
        "too coarse for S.",
        NonPassiveDataWarning,
        stacklevel=4,
    )


def _check_frequencies(f: np.ndarray) -> None:
    """Finite, non-negative, distinct, and enough of them."""
    if not (np.isfinite(f).all() and (f >= 0).all()):
        raise ValueError(
            "fit_rational needs frequencies >= 0 Hz; got "
            f"{f[~(np.isfinite(f) & (f >= 0))].tolist()}."
        )
    values, counts = np.unique(f, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            "fit_rational needs distinct frequencies; the network repeats "
            f"{values[counts > 1].tolist()} Hz."
        )
    if f.size < _MIN_FREQUENCIES:
        raise ValueError(
            f"fit_rational needs at least {_MIN_FREQUENCIES} frequencies, got {f.size}."
        )


def _max_poles(max_poles: Any, n_freq: int) -> int:
    """``max_poles`` checked, or its default ``min(100, n_freq - 1)``."""
    if max_poles is None:
        return min(_MAX_POLES, n_freq - 1)
    if (
        isinstance(max_poles, bool)
        or not isinstance(max_poles, (int, np.integer))
        or not 4 <= max_poles <= n_freq - 1
    ):
        raise ValueError(
            "fit_rational max_poles must be an integer from 4 to the number of "
            f"frequencies minus one ({n_freq - 1}), got {max_poles!r}."
        )
    return int(max_poles)


def _weights(weights: Any, n_freq: int) -> np.ndarray:
    """Per-frequency weights checked, or uniform ones."""
    if weights is None:
        return np.ones(n_freq)
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"fit_rational weights must be real numbers, got {weights!r}."
        ) from None
    if w.shape != (n_freq,) or not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError(
            f"fit_rational weights must be {n_freq} finite values > 0, one per "
            f"frequency in the network's order; got shape {w.shape}."
        )
    return w


def _reference(
    net: NetworkParameters, s: np.ndarray, z0: Any
) -> tuple[np.ndarray, np.ndarray, bool]:
    """S on one constant real reference per port, and that reference.

    The flag says whether the reference is a nominal label. The rules are
    those of `NetworkParameters.to_touchstone`, except that a reference real
    and constant to round-off counts as constant, at its mean real value.
    """
    z_ref = np.asarray(net.z_ref)
    if np.isnan(z_ref).all() and np.isfinite(s).all():
        if z0 is None:
            raise ValueError(
                "No port has a reference impedance (z_ref is NaN: wave ports "
                "without voltage_path), so S is normalized to unit modal power "
                "and has no impedance to fit it on. Pass z0= (e.g. z0=50.0) to "
                "fit S unchanged under that nominal label, or give each WavePort "
                "a voltage_path."
            )
        return s, _z0(z0, net.ports), True
    missing = [
        p for p, bad in zip(net.ports, np.isnan(z_ref).any(axis=0), strict=True) if bad
    ]
    if missing:
        raise ValueError(
            f"Port {', '.join(map(repr, missing))} has no reference impedance "
            "(z_ref is NaN): a LumpedPort without a resistance, a WavePort "
            "without a voltage_path, or a FloquetPort whose order is evanescent "
            "somewhere in the band. S cannot be fitted on a reference without "
            "one; give the port a resistance or a voltage_path, or for a "
            "FloquetPort drop the port or narrow the grid."
        )
    if z0 is None:
        constant = constant_real(z_ref)
        if constant is None:
            found = np.unique(np.round(z_ref, 12))
            raise ValueError(
                "A rational model needs one constant real reference impedance "
                f"per port, but z_ref takes the values {found.tolist()} ohm. "
                "Pass z0= (e.g. z0=50.0) to renormalize S to it before the fit."
            )
        return s, constant, False
    z = _z0(z0, net.ports)
    return np.asarray(net.renormalize(z).values), z, False


def _z0(z0: Any, ports: tuple[str, ...]) -> np.ndarray:
    """``z0`` checked: one real value > 0, or one per port; ``(n_ports,)``."""
    try:
        z = np.asarray(z0)
        ok = (
            not np.iscomplexobj(z)
            and z.dtype != bool
            and z.ndim <= 1
            and (z.ndim == 0 or z.shape == (len(ports),))
        )
        z = z.astype(float) if ok else z
    except (TypeError, ValueError):
        ok = False
    if not ok or not (np.isfinite(z).all() and (z > 0).all()):
        raise ValueError(
            "fit_rational z0 must be one real reference impedance > 0 ohm, or "
            f"one per port ({', '.join(map(repr, ports))}); got {z0!r}."
        )
    return np.broadcast_to(z, (len(ports),)).copy()


@dataclasses.dataclass(frozen=True)
class PoleResidueFit:
    """A fitted pole-residue model of S, in physical units.

    Attributes:
        poles: ``(n_poles,)`` complex, rad/s: real poles, then each
            conjugate pair with its positive-imaginary pole first.
        residues: ``(n_poles, n_ports, n_ports)`` complex, rad/s.
        d: ``(n_ports, n_ports)`` real.
        error: Largest per-element RMS of the fit's error on the data.
        converged: Whether ``error`` is within the tolerance.
    """

    poles: np.ndarray
    residues: np.ndarray
    d: np.ndarray
    error: float
    converged: bool


@dataclasses.dataclass(frozen=True)
class _Poles:
    """Normalized poles: real ones, and one pole per pair (``Im > 0``)."""

    real: np.ndarray
    pairs: np.ndarray

    @property
    def size(self) -> int:
        """The number of poles, counting both of each pair."""
        return self.real.size + 2 * self.pairs.size


@dataclasses.dataclass(frozen=True)
class _Fit:
    """Poles, real residue coefficients ``(n_el, size)``, ``d`` and the error."""

    poles: _Poles
    c: np.ndarray
    d: np.ndarray
    error: float


def vector_fit(
    omega: np.ndarray,
    s: np.ndarray,
    *,
    tolerance: float,
    max_poles: int,
    weights: np.ndarray,
) -> PoleResidueFit:
    """Fit S by fast relaxed vector fitting, raising the order as needed.

    Args:
        omega: Angular frequencies, rad/s, ``(K,)``, increasing, distinct,
            at least one positive.
        s: S on them, ``(K, n_ports, n_ports)``, finite.
        tolerance: Target of the largest per-element RMS error.
        max_poles: The most poles tried, at least 4 and below ``K``.
        weights: Positive per-frequency weights, ``(K,)``.

    Returns:
        The first fit within ``tolerance`` (pruned), or the most accurate
        one tried when no order up to ``max_poles`` reaches it.
    """
    n_freq, n_ports, _ = s.shape
    scale = float(omega[-1])
    x = 1j * omega / scale
    h = s.reshape(n_freq, n_ports * n_ports).T
    tried = []
    for n_poles in range(4, max_poles + 1, 4):
        fit = _fit_order(x, h, weights, _initial_poles(omega / scale, n_poles // 2))
        if fit.error <= tolerance:
            pruned = _prune(x, h, weights, fit, tolerance)
            return _physical(pruned, scale, n_ports, converged=True)
        tried.append(fit)
    best = min(tried, key=lambda fit: fit.error)
    return _physical(best, scale, n_ports, converged=False)


def _initial_poles(x: np.ndarray, n_pairs: int) -> _Poles:
    r"""Lightly damped pairs spread over the band, :math:`-\beta/100 \pm i\beta`."""
    positive = x[x > 0]
    lo, hi = float(positive[0]), float(positive[-1])
    if lo == hi:
        beta = np.full(n_pairs, hi)
    elif hi / lo >= _GEOMETRIC_SPAN:
        beta = np.geomspace(lo, hi, n_pairs)
    else:
        beta = np.linspace(lo, hi, n_pairs)
    return _Poles(np.zeros(0), -beta / 100 + 1j * beta)


def _fit_order(x: np.ndarray, h: np.ndarray, w: np.ndarray, poles: _Poles) -> _Fit:
    """Relocate the poles until the error stalls; the best iterate."""
    floor = 1e-15 * float(np.abs(h).max(initial=0.0))
    best = fit = _residues(x, h, w, _relocate(x, h, w, poles))
    for _ in range(_ITERATIONS - 1):
        previous = fit.error
        fit = _residues(x, h, w, _relocate(x, h, w, fit.poles))
        if fit.error < best.error:
            best = fit
        if abs(previous - fit.error) <= _STALL * fit.error + floor:
            break
    return best


def _basis(x: np.ndarray, poles: _Poles) -> np.ndarray:
    """Real-coefficient basis ``(K, size)``: real poles, then (Re, Im) per pair."""
    a = 1 / (x[:, None] - poles.pairs[None, :])
    b = 1 / (x[:, None] - np.conj(poles.pairs)[None, :])
    pair = np.empty((x.size, 2 * poles.pairs.size), complex)
    pair[:, 0::2] = a + b
    pair[:, 1::2] = 1j * (a - b)
    return np.hstack([1 / (x[:, None] - poles.real[None, :]), pair])


def _stack(m: np.ndarray) -> np.ndarray:
    """Real and imaginary rows of ``m`` stacked along the second-to-last axis."""
    return np.concatenate([m.real, m.imag], axis=-2)


def _relocate(x: np.ndarray, h: np.ndarray, w: np.ndarray, poles: _Poles) -> _Poles:
    """One fast relaxed relocation: the zeros of the scaling function."""
    phi = _basis(x, poles)
    n_freq, n = phi.shape
    n_el = h.shape[0]
    wphi = w[:, None] * phi
    # each element's block [phi, 1, -h phi, -h]; only the rows of its R that
    # act on the shared unknowns (the scaling function's) are kept
    rows = []
    for he in h:
        block = np.hstack([wphi, w[:, None], -he[:, None] * wphi, -(he * w)[:, None]])
        (r,) = scipy.linalg.qr(
            _stack(block), mode="r", overwrite_a=True, check_finite=False
        )
        rows.append(r[n + 1 : 2 * n + 2, n + 1 :])
    a = np.vstack(rows)
    weight = float(np.linalg.norm(w[None, :] * h)) / (n_freq * math.sqrt(n_el))
    relax = np.append(phi.real.sum(axis=0), n_freq) * weight
    x_sigma = _solve(
        np.vstack([a, relax]), np.append(np.zeros(a.shape[0]), n_freq * weight)
    )
    c, d = x_sigma[:-1], x_sigma[-1]
    if abs(d) < _RELAX_GUARD:
        d = math.copysign(_RELAX_GUARD, d)
        c = _solve(a[:, :n], -a[:, n] * d)
    state, b = _state_matrix(poles)
    zeros = scipy.linalg.eigvals(state - np.outer(b, c) / d)
    return _split(zeros)


def _solve(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Least squares with unit-norm columns, for conditioning."""
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    return scipy.linalg.lstsq(a / norms, rhs, check_finite=False)[0] / norms


def _state_matrix(poles: _Poles) -> tuple[np.ndarray, np.ndarray]:
    """Real block-diagonal state matrix and input vector of the pole set."""
    n_real = poles.real.size
    a = np.zeros((poles.size, poles.size))
    b = np.zeros(poles.size)
    a[np.arange(n_real), np.arange(n_real)] = poles.real
    b[:n_real] = 1.0
    for k, p in enumerate(poles.pairs):
        i = n_real + 2 * k
        a[i, i] = a[i + 1, i + 1] = p.real
        a[i, i + 1] = p.imag
        a[i + 1, i] = -p.imag
        b[i] = 2.0
    return a, b


def _split(zeros: np.ndarray) -> _Poles:
    """Stable real poles and pairs from the eigenvalues of a real matrix."""
    # flip unstable poles; keep every pole strictly inside the left half-plane
    re = -np.maximum(np.abs(zeros.real), 1e-12 * np.maximum(np.abs(zeros), 1.0))
    zeros = re + 1j * zeros.imag
    real = np.abs(zeros.imag) <= 1e-9 * np.abs(zeros)
    return _Poles(
        np.sort(zeros[real].real), np.sort_complex(zeros[~real & (zeros.imag > 0)])
    )


def _residues(x: np.ndarray, h: np.ndarray, w: np.ndarray, poles: _Poles) -> _Fit:
    """Residues and ``d`` of every element on fixed poles, and the error."""
    phi = _basis(x, poles)
    m = np.hstack([phi, np.ones((x.size, 1))]) * w[:, None]
    coef = _solve_many(_stack(m), _stack((h * w[None, :]).T))
    c, d = coef[:-1].T, coef[-1]
    error = _error(phi, c, d, h)
    return _Fit(poles, c, d, error)


def _solve_many(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """`_solve` for several right-hand sides (columns of ``rhs``)."""
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    return scipy.linalg.lstsq(a / norms, rhs, check_finite=False)[0] / norms[:, None]


def _error(phi: np.ndarray, c: np.ndarray, d: np.ndarray, h: np.ndarray) -> float:
    """Largest per-element RMS of the model's error over the data."""
    model = _terms(phi, c) + d[:, None]
    return float(np.sqrt(np.mean(np.abs(model - h) ** 2, axis=1)).max())


def _terms(phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``c @ phi.T`` (complex ``(n_el, K)``) by SciPy's BLAS, on real arrays."""
    gemm = scipy.linalg.get_blas_funcs("gemm", dtype=np.float64)
    stacked = gemm(1.0, _stack(phi), c, trans_b=True)
    n_freq = phi.shape[0]
    return (stacked[:n_freq] + 1j * stacked[n_freq:]).T


def _prune(
    x: np.ndarray, h: np.ndarray, w: np.ndarray, fit: _Fit, tolerance: float
) -> _Fit:
    """Drop poles contributing below a tenth of the tolerance in band; refit."""
    phi = _basis(x, fit.poles)
    n_real = fit.poles.real.size
    threshold = _PRUNE * tolerance
    groups = [[k] for k in range(n_real)] + [
        [n_real + 2 * k, n_real + 2 * k + 1] for k in range(fit.poles.pairs.size)
    ]
    keep = np.array(
        [_contribution(phi[:, g], fit.c[:, g]) >= threshold for g in groups], bool
    )
    keep_real, keep_pairs = keep[:n_real], keep[n_real:]
    if keep.all():
        return fit
    pruned = _residues(
        x, h, w, _Poles(fit.poles.real[keep_real], fit.poles.pairs[keep_pairs])
    )
    return pruned if pruned.error <= tolerance else fit


def _contribution(phi: np.ndarray, c: np.ndarray) -> float:
    """Largest magnitude, over elements and data, of some poles' terms."""
    return float(np.abs(_terms(phi, c)).max(initial=0.0))


def _physical(
    fit: _Fit, scale: float, n_ports: int, *, converged: bool
) -> PoleResidueFit:
    """Poles and residues in rad/s, residues as ``(n_poles, P, P)`` matrices."""
    n_real = fit.poles.real.size
    poles = [complex(p) for p in fit.poles.real]
    residues = [fit.c[:, k].astype(complex) for k in range(n_real)]
    for k, p in enumerate(fit.poles.pairs):
        r = fit.c[:, n_real + 2 * k] + 1j * fit.c[:, n_real + 2 * k + 1]
        poles += [complex(p), complex(np.conj(p))]
        residues += [r, np.conj(r)]
    shape = (len(poles), n_ports, n_ports)
    return PoleResidueFit(
        poles=np.asarray(poles, complex).reshape(-1) * scale,
        residues=np.asarray(residues, complex).reshape(shape) * scale,
        d=fit.d.reshape(n_ports, n_ports),
        error=fit.error,
        converged=converged,
    )
