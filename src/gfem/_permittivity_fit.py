r"""Fit a tabulated permittivity to a passive `PermittivityLaw`.

The fit is bounded least squares in the law's own parameters
(:math:`\varepsilon_\infty`, an optional `Drude` term and :math:`N` `Lorentz`
terms) on the relative error of the complex permittivity, with every strength,
resonance and damping held :math:`\ge 0`, so every iterate is a passive law.
The term count grows one Lorentz term at a time from fixed starting resonances,
which makes the result repeatable. `scipy.optimize` is imported only when a fit
runs, so importing gfem does not load it.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem._checks import is_traced
from gfem._materials import Drude, Lorentz, PermittivityLaw
from gfem.errors import NonDifferentiableError, PermittivityFitWarning

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["PermittivityFit", "fit_permittivity"]

_OWNER = "fit_permittivity"
#: The default cap on the Lorentz term count: beyond 6 terms dense tables gain
#: a few tenths of a percent at a much longer run time.
_DEFAULT_TERMS = 6
#: Starting resonances per added term, on a log grid over
#: :math:`[0.5 f_\mathrm{min}, 3 f_\mathrm{max}]`.
_SEEDS = 8
_SEED_LOW, _SEED_HIGH = 0.5, 3.0
#: A resonance stays above this fraction of the lowest data frequency.
_RESONANCE_FLOOR = 1e-3
#: The linewidth floor, in median data spacings: below it, noise in a table
#: turns into narrow spurious resonances.
_LINEWIDTH_SPACINGS = 2.0
_SOLVER = {
    "method": "trf",
    "x_scale": "jac",
    "max_nfev": 800,
    "ftol": 1e-10,
    "xtol": 1e-10,
    "gtol": 1e-15,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class PermittivityFit:
    r"""A permittivity law fitted to tabulated data, with its errors.

    Returned by `fit_permittivity`; a user does not build one. Each error is
    the largest relative error over the data points, and holds over the data's
    frequencies only, which the law carries as ``law.frequency_range``. A
    fit is not saved: save the law, through a `gfem.Material` in a model.

    Attributes:
        law: The fitted law, passive at every real frequency: ``eps_inf``,
            then a `gfem.Drude` term (if any), then `gfem.Lorentz` terms by
            rising resonance. ``law.frequency_range`` is the lowest and
            highest data frequency.
        eps_error: :math:`\max_i |\varepsilon_\mathrm{law}(f_i) -
            \varepsilon_i| / |\varepsilon_i|`, the quantity the fit
            minimizes.
        n_error: :math:`\max_i |n_\mathrm{law}(f_i) - n_i| / n_i`, with
            :math:`n - ik = \sqrt{\varepsilon}` on the branch
            :math:`k \ge 0`. A point where :math:`n_i = 0` counts 0 if the
            law's :math:`n` is 0 there and :math:`\infty` otherwise.
        k_error: :math:`\max_i |k_\mathrm{law}(f_i) - k_i| / k_i`, counted
            the same way. It is :math:`\infty` when the data have
            :math:`k = 0` where the law has loss.
    """

    law: PermittivityLaw
    eps_error: float
    n_error: float
    k_error: float

    def __repr__(self) -> str:
        k = _percent(self.k_error)
        if math.isinf(self.k_error):
            k += " (loss where the data have k = 0)"
        lo, hi = self.law.frequency_range or (math.nan, math.nan)
        return (
            f"PermittivityFit({_describe(self.law)}; max relative error "
            f"eps {_percent(self.eps_error)}, n {_percent(self.n_error)}, k {k}; "
            f"{lo:.3g}-{hi:.3g} Hz)"
        )


def fit_permittivity(
    frequency: ArrayLike,
    *,
    n: ArrayLike | None = None,
    k: ArrayLike | None = None,
    eps: ArrayLike | None = None,
    drude: bool = False,
    tolerance: float = 1e-2,
    max_lorentz_terms: int | None = None,
) -> PermittivityFit:
    r"""Fit tabulated optical constants or permittivity to a passive law.

    The law is ``eps_inf`` plus, with ``drude=True``, one `gfem.Drude`
    term, plus Lorentz terms, added one at a time until the largest relative
    error in :math:`\varepsilon` over the data is at most ``tolerance``. If
    no term count up to the cap gets there, the law with the smallest error
    is returned and `gfem.PermittivityFitWarning` says so. Every strength
    and damping is held :math:`\ge 0`, so the law is passive at every real
    frequency, and :math:`\varepsilon_\infty \ge 1`.

    Give the data one of two ways:

    - ``n`` and ``k``: the tabulated index :math:`n + ik`, where ``k`` is
      the extinction coefficient :math:`k \ge 0` of an absorbing medium. As
      a magnitude it does not depend on the time convention, so tables
      under either convention are passed as they are; gfem's
      :math:`e^{+i\omega t}` convention gives
      :math:`\varepsilon = (n - ik)^2`, so
      :math:`\mathrm{Im}\,\varepsilon = -2nk`.
    - ``eps``: the complex relative permittivity in gfem's convention, with
      :math:`\mathrm{Im}\,\varepsilon \le 0` for loss (`gfem.conventions`).
      A table written under :math:`e^{-i\omega t}`
      (:math:`\varepsilon_1 + i\varepsilon_2`) is passed as
      ``np.conj(eps)``.

    When the data have no loss (``k`` left out or all 0, or a real ``eps``)
    every damping is fixed at 0 and the law is lossless, with no resonance
    inside the data's range. A lossless fit reaches errors of about
    :math:`10^{-6}` cheaply, so a tighter ``tolerance`` suits phase-sensitive
    work. With loss, every Lorentz damping is at least twice the median
    spacing of the data frequencies, so that noise in a table does not turn
    into narrow resonances. A term whose strength fits to 0 is dropped:
    ``drude=True`` allows a Drude term, it does not force one.

    The law holds over the data's frequencies only: its ``frequency_range``
    is their lowest and highest value, and a solve that reads it outside
    warns `gfem.FittedRangeWarning`. A sum of damped terms has loss at every
    frequency, so a fit across an absorption edge leaks loss below the edge:
    the ``k`` error shows it, and warns. For a metal, :math:`n` is small
    where :math:`k` is large, and the ``n`` error, which is the error of its
    absorption (:math:`\mathrm{Im}\,\varepsilon = -2nk`), can be about
    twenty times the :math:`\varepsilon` error near the index minimum (gold
    near 0.7 µm) whatever the term count. It is reported, never warned.

    The fit is deterministic: the same data give the same law to the last
    bit on one machine, in any order of rows. It runs on the host, in
    seconds to about a minute for up to a few hundred points; the time
    grows with the points and the terms, which ``max_lorentz_terms`` bounds.
    It is not collective: every MPI rank may call it and gets the same law.

    Args:
        frequency: The data frequencies in Hz: 1-D, real, > 0 and distinct,
            in any order (wavelength tables convert with ``c / wavelength``).
        n: The refractive index, real, one per frequency.
        k: The extinction coefficient :math:`k \ge 0`, one per frequency;
            with ``n`` only. Left out, the data are lossless.
        eps: The complex relative permittivity under
            :math:`e^{+i\omega t}`, instead of ``n`` and ``k``.
        drude: Also fit one `gfem.Drude` term, for a metal.
        tolerance: The target for the largest relative error in
            :math:`\varepsilon`, > 0.
        max_lorentz_terms: The most Lorentz terms to try. ``None`` takes
            :math:`\min(6, N_\mathrm{pts})`, where the point cap
            :math:`N_\mathrm{pts}` is the most terms whose free real
            parameters number at most half the data points (more would
            follow the noise between them).

    Returns:
        The fitted law with its largest relative errors in
        :math:`\varepsilon`, :math:`n` and :math:`k` over the data.

    Raises:
        TypeError: Neither or both of ``n`` and ``eps`` are given, ``k``
            comes without ``n``, or ``drude`` is not a bool.
        ValueError: The data are unusable: not 1-D or of unequal lengths,
            not real (``frequency``, ``n``, ``k``) or not numbers, NaN or
            infinite, a frequency <= 0 or repeated, :math:`n < 0`, gain
            (:math:`k < 0`, or :math:`\mathrm{Im}\,\varepsilon > 0`), an
            ``eps`` of 0, too few points for the smallest law, or a
            ``max_lorentz_terms`` or ``tolerance`` out of range. Nothing is
            clipped; the message names the remedy.
        NonDifferentiableError: An input is traced by JAX
            (`jax.grad`, `jax.jit`): the fit picks a term count and
            starting resonances, so it has no derivative. Fit outside the
            transformation; the returned law's parameters differentiate.

    Warns:
        PermittivityFitWarning: The error in :math:`\varepsilon` or in
            :math:`k` is above ``tolerance``.

    Example:
        A measured gold table, wavelength in µm::

            from scipy.constants import c

            wavelength, n, k = np.loadtxt("gold.csv", delimiter=",", unpack=True)
            fit = gfem.fit_permittivity(c / (wavelength * 1e-6), n=n, k=k, drude=True)
            gold = gfem.Material(eps_r=fit.law, name="gold (our film)")
    """
    data = _Data.check(
        frequency,
        n=n,
        k=k,
        eps=eps,
        drude=drude,
        tolerance=tolerance,
        max_lorentz_terms=max_lorentz_terms,
    )
    scale = 10.0 ** round(math.log10(data.f[-1]))
    steps = _greedy(data, scale)
    best = _choose(steps, data.tolerance)
    law = _build(best.params, data, scale)
    fit = _errors(law, data)
    _warn(fit, data, best.terms)
    return fit


# --- input checks ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Data:
    """Checked data, sorted by rising frequency, and the fit's settings."""

    f: np.ndarray
    eps: np.ndarray
    n: np.ndarray
    k: np.ndarray
    drude: bool
    lossless: bool
    tolerance: float
    cap: int
    cap_reason: str
    point_cap: int

    @classmethod
    def check(
        cls,
        frequency: Any,
        *,
        n: Any,
        k: Any,
        eps: Any,
        drude: Any,
        tolerance: Any,
        max_lorentz_terms: Any,
    ) -> _Data:
        """Refuse unusable input before any fitting (the caller's arrays stay)."""
        if is_traced((frequency, n, k, eps, tolerance)):
            raise NonDifferentiableError(
                f"{_OWNER} prepares data and is not differentiable (it picks a "
                "term count and starting resonances); call it outside "
                "jax.grad/jit and differentiate through the returned law's "
                "parameters."
            )
        if not isinstance(drude, (bool, np.bool_)):
            raise TypeError(f"{_OWNER}: drude takes True or False, got {drude!r}.")
        tol = _tolerance(tolerance)
        if k is not None and n is None:
            raise TypeError(f"{_OWNER}: k goes with n; eps already holds the loss.")
        if (n is None) == (eps is None):
            raise TypeError(
                f"{_OWNER}: give the data as n= (and k=) or as eps=, not both."
            )
        given = {"frequency": frequency, "n": n, "k": k, "eps": eps}
        arrays = {
            name: _array(name, value)
            for name, value in given.items()
            if value is not None
        }
        _same_shape(arrays)
        for name, arr in arrays.items():
            _finite(name, arr)
        f, order = _frequencies(arrays["frequency"])
        e, nn, kk = _permittivity(arrays, order, f)
        lossless = not np.any(e.imag)
        drude = bool(drude)
        base, per = _parameters(drude=drude, lossless=lossless)
        m = f.size
        smallest = base + (0 if drude else per)
        if m < 2 * smallest:
            law = "a Drude term" if drude else "one Lorentz term"
            raise ValueError(
                f"{_OWNER} needs at least {2 * smallest} data points for {law} "
                f"(twice its {smallest} free parameters), got {m}: with fewer "
                "the law follows the noise between them."
            )
        point_cap = int((m / 2 - base) // per)
        cap, reason = _cap(
            max_lorentz_terms,
            point_cap=point_cap,
            points=m,
            base=base,
            per=per,
            drude=drude,
        )
        return cls(
            f=f,
            eps=e,
            n=nn,
            k=kk,
            drude=drude,
            lossless=lossless,
            tolerance=tol,
            cap=cap,
            cap_reason=reason,
            point_cap=point_cap,
        )


def _frequencies(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The frequencies sorted rising, and the order that sorts the rows."""
    f = frequency.astype(float)
    if f.size and f.min() <= 0:
        i = int(np.argmin(f))
        raise ValueError(
            f"{_OWNER}: frequencies must be > 0 Hz, got {f[i]:.6g} Hz at index {i}."
        )
    order = np.argsort(f, kind="stable")
    f = f[order]
    repeated = np.flatnonzero(np.diff(f) == 0)
    if repeated.size:
        raise ValueError(
            f"{_OWNER}: the frequency {f[repeated[0]]:.9g} Hz appears more "
            "than once; average or drop the repeated rows (repeats make the "
            "median spacing, and so the linewidth floor, 0)."
        )
    return f, order


def _permittivity(
    arrays: dict[str, np.ndarray], order: np.ndarray, f: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eps, n and k of each sorted row, from n and k or from eps."""
    if "eps" in arrays:
        e = arrays["eps"].astype(complex)[order]
        _eps_signs(f, e)
        n, k = _index(e)
    else:
        n = arrays["n"].astype(float)[order]
        k = arrays["k"].astype(float)[order] if "k" in arrays else np.zeros_like(n)
        _index_signs(f, n, k)
        e = (n - 1j * k) ** 2
        if not np.any(k):
            e = e.real + 0j  # (n - 0j)**2 can carry a -0j
    if np.any(e == 0):
        raise ValueError(
            f"{_OWNER}: relative error is undefined where eps = 0 (n = k = 0); "
            "drop those rows."
        )
    return e, n, k


def _tolerance(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{_OWNER}: tolerance must be > 0, got {value!r}.")
    arr = np.asarray(value)
    if arr.dtype.kind not in "iuf" or not np.isfinite(arr) or arr <= 0:
        raise ValueError(f"{_OWNER}: tolerance must be > 0, got {value!r}.")
    return float(arr)


def _array(name: str, value: Any) -> np.ndarray:
    arr = np.array(value)  # a copy: the caller's array is never modified
    kinds = "iufc" if name == "eps" else "iuf"
    if name != "eps" and np.iscomplexobj(arr):
        raise ValueError(f"{_OWNER}: {name} must be real, got a complex array.")
    if arr.dtype.kind not in kinds:
        what = "numbers" if name == "eps" else "real"
        raise ValueError(f"{_OWNER}: {name} must be {what}, got dtype {arr.dtype}.")
    return arr


def _same_shape(arrays: dict[str, np.ndarray]) -> None:
    f = arrays["frequency"]
    if all(a.ndim == 1 for a in arrays.values()) and all(
        a.shape == f.shape for a in arrays.values()
    ):
        return
    shapes = ", ".join(f"{name} {a.shape}" for name, a in arrays.items())
    raise ValueError(
        f"{_OWNER}: {', '.join(arrays)} must be 1-D arrays of one length "
        f"(one row per frequency), got shapes {shapes}."
    )


def _finite(name: str, arr: np.ndarray) -> None:
    bad = np.flatnonzero(~np.isfinite(arr))
    if bad.size:
        raise ValueError(
            f"{_OWNER}: {name} has {bad.size} NaN or infinite value(s), the "
            f"first at index {bad[0]}; drop those rows, e.g. "
            "ok = np.isfinite(n) & np.isfinite(k)."
        )


def _index_signs(f: np.ndarray, n: np.ndarray, k: np.ndarray) -> None:
    negative = np.flatnonzero(n < 0)
    if negative.size:
        i = negative[np.argmin(n[negative])]
        raise ValueError(
            f"{_OWNER}: n < 0 at {negative.size} rows (smallest {n[i]:.6g} at "
            f"{f[i]:.6g} Hz): a negative index is not a passive Drude-Lorentz "
            "material."
        )
    gain = np.flatnonzero(k < 0)
    if not gain.size:
        return
    if not np.any(k > 0):
        raise ValueError(
            f"{_OWNER}: k < 0 at {gain.size} rows: k is the extinction "
            "coefficient, >= 0 for an absorbing medium under any time "
            "convention; if you passed Im(n - ik), pass k=-k."
        )
    i = gain[np.argmin(k[gain])]
    raise ValueError(
        f"{_OWNER}: k < 0 at {gain.size} rows (smallest {k[i]:.6g} at "
        f"{f[i]:.6g} Hz) is gain, which a passive law cannot fit; if these are "
        "rounding of k = 0, pass k=np.clip(k, 0, None)."
    )


def _eps_signs(f: np.ndarray, eps: np.ndarray) -> None:
    gain = np.flatnonzero(eps.imag > 0)
    if not gain.size:
        return
    if not np.any(eps.imag < 0):
        raise ValueError(
            f"{_OWNER}: eps has Im > 0 at {gain.size} rows: gfem's convention "
            "is e^{+iωt}, where loss is Im eps < 0 (gfem.conventions); a table "
            "written under e^{-iωt} (eps1 + i eps2) is its conjugate: pass "
            "eps=np.conj(eps)."
        )
    i = gain[np.argmax(eps.imag[gain])]
    raise ValueError(
        f"{_OWNER}: Im eps > 0 at {gain.size} rows (largest {eps.imag[i]:.6g} "
        f"at {f[i]:.6g} Hz) is gain, which a passive law cannot fit; if these "
        "are rounding of Im eps = 0, pass "
        "eps=eps.real + 1j * np.minimum(eps.imag, 0)."
    )


def _parameters(*, drude: bool, lossless: bool) -> tuple[int, int]:
    """Free real parameters: the base law's, and each Lorentz term's."""
    if lossless:
        return (2 if drude else 1), 2
    return (3 if drude else 1), 3


def _cap(
    value: Any, *, point_cap: int, points: int, base: int, per: int, drude: bool
) -> tuple[int, str]:
    """The Lorentz term cap, and why it is that."""
    m = points
    if value is None:
        if point_cap < _DEFAULT_TERMS:
            return point_cap, f"the point cap for {m} points"
        return _DEFAULT_TERMS, f"the default {_DEFAULT_TERMS}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(  # noqa: TRY004 (every max_lorentz_terms refusal is one class)
            f"{_OWNER}: max_lorentz_terms takes an int >= 0 or None, got {value!r}."
        )
    terms = int(value)
    if terms < 0:
        raise ValueError(
            f"{_OWNER}: max_lorentz_terms takes an int >= 0 or None, got {terms}."
        )
    if terms == 0 and not drude:
        raise ValueError(
            f"{_OWNER}: max_lorentz_terms=0 without drude=True: the law would "
            "have no term."
        )
    if terms > point_cap:
        need = 2 * (base + per * terms)
        raise ValueError(
            f"{_OWNER}: max_lorentz_terms={terms} needs at least {need} points, "
            f"got {m}: use max_lorentz_terms<={point_cap} or more data."
        )
    return terms, "max_lorentz_terms"


# --- the fit -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Step:
    """The law after one greedy step, in scaled units."""

    terms: int
    params: np.ndarray
    error: float  # max relative eps error; inf for a law with no term


def _layout(params: np.ndarray, drude: bool, lossless: bool) -> tuple:
    """Split a parameter vector into eps_inf, the Drude term and Lorentz terms."""
    per = 2 if lossless else 3
    i = 1
    drude_term = None
    if drude:
        drude_term = (params[1], 0.0 if lossless else params[2])
        i = 2 if lossless else 3
    lorentz = []
    while i < len(params):
        d, f0 = params[i], params[i + 1]
        lorentz.append((d, f0, 0.0 if lossless else params[i + 2]))
        i += per
    return params[0], drude_term, lorentz


def _model(
    params: np.ndarray, f: np.ndarray, drude: bool, lossless: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The law's permittivity at ``f`` and its Jacobian in ``params``."""
    eps_inf, drude_term, lorentz = _layout(params, drude, lossless)
    value = np.full(f.shape, eps_inf, dtype=complex)
    cols = [np.ones(f.shape, dtype=complex)]
    if drude_term is not None:
        fp, g = drude_term
        den = f * (f - 1j * g)
        value += -(fp**2) / den
        cols.append(-2 * fp / den)
        if not lossless:
            cols.append(-1j * fp**2 / (f * (f - 1j * g) ** 2))
    for d, f0, g in lorentz:
        den = f0**2 - f**2 + 1j * g * f
        value += d * f0**2 / den
        cols.append(f0**2 / den)
        cols.append(d * 2 * f0 * (-(f**2) + 1j * g * f) / den**2)
        if not lossless:
            cols.append(-1j * d * f * f0**2 / den**2)
    return value, np.array(cols).T


def _solve(
    params: np.ndarray,
    *,
    low: list[float],
    high: list[float],
    data: _Data,
    f: np.ndarray,
) -> np.ndarray:
    """Bounded least squares on the relative error, from ``params``."""
    eps, drude, lossless = data.eps, data.drude, data.lossless
    # imported here so that importing gfem does not load scipy.optimize
    from scipy.optimize import least_squares

    weight = 1 / np.abs(eps)

    def stack(z: np.ndarray) -> np.ndarray:
        return z.real if lossless else np.concatenate([z.real, z.imag])

    def residual(p: np.ndarray) -> np.ndarray:
        return stack((_model(p, f, drude, lossless)[0] - eps) * weight)

    def jacobian(p: np.ndarray) -> np.ndarray:
        jac = _model(p, f, drude, lossless)[1] * weight[:, None]
        return jac.real if lossless else np.vstack([jac.real, jac.imag])

    return least_squares(
        residual, params, jac=jacobian, bounds=(low, high), **_SOLVER
    ).x


def _relative(params: np.ndarray, data: _Data, f: np.ndarray) -> np.ndarray:
    value = _model(params, f, data.drude, data.lossless)[0]
    return np.abs(value - data.eps) / np.abs(data.eps)


def _greedy(data: _Data, scale: float) -> list[_Step]:
    """Add one Lorentz term at a time, from fixed starting resonances.

    Each step tries every starting resonance with the previous law's
    parameters as the start of a joint fit, and keeps the fit with the
    smallest RMS relative error. It stops at the first law within the
    tolerance, or at the cap.
    """
    f = data.f / scale
    fmin, fmax = f[0], f[-1]
    floor = 0.0 if data.lossless else _LINEWIDTH_SPACINGS * float(np.median(np.diff(f)))
    params, low, high = _start(data, fmin)
    p = _solve(np.array(params), low=low, high=high, data=data, f=f)
    steps = []
    if data.drude:
        steps.append(_Step(0, p, _max_error(p, data, f)))
        if steps[-1].error <= data.tolerance:
            return steps
    for terms in range(1, data.cap + 1):
        best = None
        for f0 in np.geomspace(_SEED_LOW * fmin, _SEED_HIGH * fmax, _SEEDS):
            if data.lossless:
                start = np.concatenate([p, [0.5, f0]])
                low_t = [*low, 0.0, _RESONANCE_FLOOR * fmin]
                high_t = [*high, np.inf, np.inf]
            else:
                start = np.concatenate([p, [0.5, f0, max(0.3 * f0, floor)]])
                low_t = [*low, 0.0, _RESONANCE_FLOOR * fmin, floor]
                high_t = [*high, np.inf, np.inf, np.inf]
            x = _solve(start, low=low_t, high=high_t, data=data, f=f)
            if data.lossless and _resonance_in_range(x, data, fmin, fmax):
                continue  # transparent data hold no pole: it would diverge between rows
            rms = float(np.sqrt(np.mean(_relative(x, data, f) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, x, low_t, high_t)
        if best is None:
            break
        _, p, low, high = best
        steps.append(_Step(terms, p, _max_error(p, data, f)))
        if steps[-1].error <= data.tolerance:
            break
    return steps


def _start(data: _Data, fmin: float) -> tuple[list[float], list[float], list[float]]:
    """The base law's starting parameters and bounds: eps_inf, then Drude."""
    if not data.drude:
        return [max(1.0, float(data.eps.real.max()))], [1.0], [np.inf]
    fp = fmin * math.sqrt(max(1.0, -data.eps[0].real))
    if data.lossless:
        return [1.0, fp], [1.0, 0.0], [np.inf, np.inf]
    return [1.0, fp, 0.01 * fp], [1.0, 0.0, 0.0], [np.inf] * 3


def _resonance_in_range(
    params: np.ndarray, data: _Data, fmin: float, fmax: float
) -> bool:
    _, _, lorentz = _layout(params, data.drude, data.lossless)
    return any(d > 0 and fmin <= f0 <= fmax for d, f0, _ in lorentz)


def _max_error(params: np.ndarray, data: _Data, f: np.ndarray) -> float:
    _, drude_term, lorentz = _layout(params, data.drude, data.lossless)
    has_term = (drude_term is not None and drude_term[0] > 0) or any(
        d > 0 for d, _, _ in lorentz
    )
    if not has_term:
        return math.inf  # a constant is not a law
    return float(np.max(_relative(params, data, f)))


def _choose(steps: list[_Step], tolerance: float) -> _Step:
    """The first law within the tolerance, else the smallest error.

    Of equal errors the law with fewer terms wins.
    """
    if not steps:
        # only a lossless fit without a Drude term discards seeds, and it
        # discarded every one at the first term
        raise ValueError(
            f"{_OWNER}: no lossless law fits these data: every Lorentz term "
            "tried put its resonance between the data's frequencies, where a "
            "lossless law diverges. Remedies: give k where the material "
            "absorbs, fit a narrower range of rows on one side of the "
            "resonance, or pass drude=True for a metal."
        )
    for step in steps:
        if step.error <= tolerance:
            return step
    best = min(steps, key=lambda s: s.error)  # min keeps the first of equal errors
    if math.isinf(best.error):
        raise ValueError(
            f"{_OWNER}: every fitted term came out with zero strength: the data "
            "show no dispersion a Drude-Lorentz law can follow."
        )
    return best


def _build(params: np.ndarray, data: _Data, scale: float) -> PermittivityLaw:
    eps_inf, drude_term, lorentz = _layout(params, data.drude, data.lossless)
    terms: list[Drude | Lorentz] = []
    if drude_term is not None and drude_term[0] > 0:
        fp, g = drude_term
        terms.append(
            Drude(plasma_frequency=float(fp) * scale, damping=float(g) * scale)
        )
    for d, f0, g in sorted(lorentz, key=lambda t: t[1]):
        if d > 0:
            terms.append(
                Lorentz(
                    delta_eps=float(d),
                    resonance_frequency=float(f0) * scale,
                    damping=float(g) * scale,
                )
            )
    return PermittivityLaw(
        eps_inf=float(eps_inf),
        terms=tuple(terms),
        frequency_range=(float(data.f[0]), float(data.f[-1])),
    )


# --- the report --------------------------------------------------------------------


def _index(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``n`` and ``k`` from ``n - ik = sqrt(eps)`` on the branch ``k >= 0``."""
    s = np.sqrt(eps.astype(complex))
    s = np.where(s.imag > 0, -s, s)
    return s.real, -s.imag


def _relative_index(model: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Relative error per point; a 0 in the data counts 0 if matched, else inf."""
    err = np.empty(data.shape)
    zero = data == 0
    err[~zero] = np.abs(model[~zero] - data[~zero]) / data[~zero]
    err[zero] = np.where(model[zero] == 0, 0.0, np.inf)
    return err


def _errors(law: PermittivityLaw, data: _Data) -> PermittivityFit:
    value = np.asarray(law(data.f))
    if data.lossless:
        value = value.real + 0j
    n_law, k_law = _index(value)
    return PermittivityFit(
        law=law,
        eps_error=float(np.max(np.abs(value - data.eps) / np.abs(data.eps))),
        n_error=float(np.max(_relative_index(n_law, data.n))),
        k_error=float(np.max(_relative_index(k_law, data.k))),
    )


def _warn(fit: PermittivityFit, data: _Data, terms: int) -> None:
    lines = []
    tol = data.tolerance
    if fit.eps_error > tol:
        remedies = []
        if terms >= data.cap and data.cap < data.point_cap:
            remedies.append(
                f"raise max_lorentz_terms (up to {data.point_cap} for these points)"
            )
        if not data.drude and np.any(data.eps.real < 0):
            remedies.append("pass drude=True (the data have Re eps < 0, as a metal)")
        remedies += [
            "narrow the band (fit a slice of the rows)",
            "raise tolerance if the table's own scatter is that large",
        ]
        lines.append(
            f"the fit reached a relative error of {_percent(fit.eps_error)} in eps "
            f"with {_describe(fit.law)} (cap {data.cap} Lorentz terms: "
            f"{data.cap_reason}), above tolerance {tol:.3g}. Remedies: "
            + "; ".join(remedies)
            + "."
        )
    if fit.k_error > tol:
        value = np.asarray(fit.law(data.f))
        _, k_law = _index(value)
        remedy = (
            "Fit only where the material absorbs, or fit the transparent part "
            "alone without k, which gives a lossless law."
        )
        if math.isinf(fit.k_error):
            zero = np.flatnonzero(data.k == 0)
            i = zero[np.argmax(k_law[zero])]
            lines.append(
                f"the law's k reaches {k_law[i]:.3g} at {data.f[i]:.4g} Hz where "
                "the data have k = 0: a sum of damped terms has loss at every "
                f"frequency and leaks it below an absorption edge. {remedy}"
            )
        else:
            err = _relative_index(k_law, data.k)
            i = int(np.argmax(err))
            lines.append(
                f"the law's k is off by {_percent(fit.k_error)} at {data.f[i]:.4g} Hz "
                f"(law {k_law[i]:.3g}, data {data.k[i]:.3g}): a sum of damped terms "
                "has loss at every frequency and leaks it below an absorption "
                f"edge. {remedy}"
            )
    if lines:
        warnings.warn(
            f"{_OWNER}: " + "\n".join(lines), PermittivityFitWarning, stacklevel=3
        )


def _percent(x: float) -> str:
    return "inf" if math.isinf(x) else f"{100 * x:.2g} %"


def _describe(law: PermittivityLaw) -> str:
    drude = sum(isinstance(t, Drude) for t in law.terms)
    lorentz = sum(isinstance(t, Lorentz) for t in law.terms)
    parts = (["Drude"] if drude else []) + ([f"{lorentz} Lorentz"] if lorentz else [])
    return " + ".join(parts)
