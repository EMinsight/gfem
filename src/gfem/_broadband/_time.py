r"""Time responses of S: windowed inverse transforms in `jax.numpy`.

Low pass (impulse and step): S on a harmonic grid :math:`f_k = k\,\Delta f`,
:math:`k = 1..N`, with DC added, times the upper half of a Kaiser window,
through `irfft` with no conjugation (gfem's phasors are
:math:`e^{+i\omega t}`, so the plain inverse transform puts a delay at
positive time). The record is two-sided, :math:`t_m = (m - n_t/2) /
(n_t\,\Delta f)`, because the window's kernel reaches into negative time; the
step is the cumulative sum over the whole record (summing from
:math:`t = 0` would drop half of the kernel of an edge at 0), and the impulse
is divided by the kernel's peak, so an isolated reflection :math:`\Gamma`
peaks at :math:`\Gamma` whatever the padding.

Band pass (impulse only): S on any evenly spaced band times a full Kaiser
window, complex inverse transform, magnitude.

The window, grid and time axis are host constants; S and the reference
impedance flow through `jax.numpy` only, so the responses differentiate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._broadband._reference import constant_real
from gfem._checks import is_traced, static_float
from gfem._pytree import readonly
from gfem._text import hertz, hertz_band
from gfem.broadband import TimeResponse
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem.results import NetworkParameters

#: Relative tolerance of the grid checks, on :math:`f_k / \Delta f - k`.
_GRID_TOL = 1e-6

#: The lowest number of positive frequencies a time response takes.
_MIN_POINTS = 3

_MODES = ("low_pass", "band_pass")


def impulse_response(
    net: NetworkParameters,
    *,
    mode: Any,
    kaiser_beta: Any,
    n_time: Any,
    z0: Any,
) -> TimeResponse:
    """`NetworkParameters.impulse_response`."""
    if is_traced(mode):
        raise NonDifferentiableError(
            "impulse_response mode is a fixed choice; pass a plain string."
        )
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError(
            f"impulse_response mode is 'low_pass' or 'band_pass', got {mode!r}."
        )
    beta = _kaiser_beta(kaiser_beta, "impulse_response")
    if mode == "band_pass":
        return _band_pass(net, beta, n_time, z0)
    return _low_pass(net, "impulse", beta, n_time, z0)


def step_response(
    net: NetworkParameters, *, kaiser_beta: Any, n_time: Any, z0: Any
) -> TimeResponse:
    """`NetworkParameters.step_response`."""
    beta = _kaiser_beta(kaiser_beta, "step_response")
    return _low_pass(net, "step", beta, n_time, z0)


def _low_pass(
    net: NetworkParameters, kind: str, beta: float, n_time: Any, z0: Any
) -> TimeResponse:
    zero, positive, df = _harmonic_grid(net.frequency, kind)
    n = _n_time(n_time, len(positive), f"{kind}_response")
    s, z_ref, excited = _reference(net, z0, f"{kind}_response")
    s_pos = s[jnp.asarray(positive)]
    if zero is None:
        # S is Hermitian, so Re S is even in f: the polynomial in f^2 through
        # the first three points, at 0 (exact for 1, f^2, f^4)
        dc = jnp.real(15 * s_pos[0] - 6 * s_pos[1] + s_pos[2]) / 10
    else:
        dc = s[zero]
    x = jnp.concatenate([dc[None].astype(s_pos.dtype), s_pos])
    window = _half_kaiser(len(positive) + 1, beta)
    h = jnp.fft.irfft(x * window[:, None, None], n=n, axis=0)
    h = jnp.fft.fftshift(h, axes=0)
    if kind == "step":
        values = jnp.cumsum(h, axis=0)
    else:
        values = h / np.fft.irfft(window, n=n)[0]
    return TimeResponse(
        time=_time_axis(n, df),
        values=values,
        z_ref=z_ref,
        ports=net.ports,
        excited=excited,
        kind=kind,  # ty: ignore[invalid-argument-type]
        mode="low_pass",
    )


def _band_pass(
    net: NetworkParameters, beta: float, n_time: Any, z0: Any
) -> TimeResponse:
    order, df = _even_grid(net.frequency)
    n = _n_time(n_time, len(order), "impulse_response")
    s, z_ref, excited = _reference(net, z0, "impulse_response")
    window = np.kaiser(len(order), beta)
    h = jnp.fft.ifft(s[jnp.asarray(order)] * window[:, None, None], n=n, axis=0)
    # the kernel's peak is at t = 0: sum(w) / n
    values = jnp.fft.fftshift(jnp.abs(h), axes=0) / (window.sum() / n)
    return TimeResponse(
        time=_time_axis(n, df),
        values=values,
        z_ref=z_ref,
        ports=net.ports,
        excited=excited,
        kind="impulse",
        mode="band_pass",
    )


def _kaiser_beta(value: Any, what: str) -> float:
    beta = static_float(
        value,
        f"{what} kaiser_beta",
        "Pass a plain number (0 to 13; 6 is the usual network-analyzer default).",
    )
    if beta < 0:
        raise ValueError(f"{what} kaiser_beta must be >= 0, got {beta}.")
    return beta


def _half_kaiser(m: int, beta: float) -> np.ndarray:
    r"""The upper half of a Kaiser window of length :math:`2m`, from its peak.

    :math:`w_k = I_0(\beta \sqrt{1 - (k/m)^2}) / I_0(\beta)`, :math:`k =
    0..m-1`: the periodic window's samples from the centre on, so the
    two-sided spectrum :math:`S(-f) = S(f)^*` sees one symmetric window.
    """
    k = np.arange(m) / m
    return np.i0(beta * np.sqrt(1 - k**2)) / np.i0(beta)


def _n_time(value: Any, n_freq: int, what: str) -> int:
    """The record's sample count: given (even, >= 2N), or the default."""
    if value is None:
        # Δt <= 1 / (20 f_max), below a tenth of the rise time at any β
        return 1 << math.ceil(math.log2(20 * n_freq))
    if is_traced(value):
        raise NonDifferentiableError(
            f"{what} n_time is a fixed sample count; pass a plain integer."
        )
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value % 2
        or value < 2 * n_freq
    ):
        raise ValueError(
            f"{what} n_time must be an even integer >= 2N = {2 * n_freq} (N = "
            f"{n_freq} frequencies), got {value!r}; None picks the smallest "
            "power of two >= 20N."
        )
    return int(value)


def _time_axis(n: int, df: float) -> np.ndarray:
    r""":math:`t_m = (m - n/2) / (n\,\Delta f)`, the two-sided record."""
    return readonly((np.arange(n) - n // 2) / (n * df))


def _harmonic_grid(
    frequency: tuple[float, ...], kind: str
) -> tuple[int | None, np.ndarray, float]:
    r"""The zero-frequency index, the positive indices in increasing order, Δf.

    Raises:
        ValueError: The frequencies are not :math:`k\,\Delta f`, :math:`k =
            1..N`, :math:`N \ge 3`, with at most one zero; the message gives
            the grid to solve on and the explicit resampling call.
    """
    f = np.asarray(frequency, dtype=float)
    zeros = np.flatnonzero(f == 0)
    positive = np.flatnonzero(f > 0)
    positive = positive[np.argsort(f[positive], kind="stable")]
    fp = f[positive]
    problem = None
    if (f < 0).any() or not np.isfinite(f).all():
        problem = "it holds negative or non-finite frequencies"
    elif zeros.size > 1:
        problem = "0 Hz appears more than once"
    elif fp.size < _MIN_POINTS:
        problem = f"it has {fp.size} positive frequencies, fewer than {_MIN_POINTS}"
    else:
        k = fp / fp[0]
        bad = np.flatnonzero(np.abs(k - np.arange(1, fp.size + 1)) > _GRID_TOL)
        if bad.size:
            problem = (
                f"{hertz(fp[bad[0]])} is not {bad[0] + 1} times the lowest "
                f"frequency {hertz(fp[0])}"
            )
    if problem is not None:
        df = fp[0] if fp.size else 1.0
        n = max(_MIN_POINTS, math.ceil(fp[-1] / df)) if fp.size else _MIN_POINTS
        raise ValueError(
            f"A low-pass {kind} response needs S on a harmonic grid, f_k = k df "
            f"for k = 1..N (N >= {_MIN_POINTS}, an optional 0 Hz point), because "
            f"the inverse transform is periodic in 1/df; here {problem}. Solve "
            "on such a grid: df = 1 / duration, n = ceil(f_max / df), "
            "frequency = df * np.arange(1, n + 1) (here df = "
            f"{df:.6g}, n = {n}). Or resample S explicitly through a rational "
            "model: s.fit_rational().network(df * np.arange(0, n + 1))."
        )
    return (int(zeros[0]) if zeros.size else None), positive, float(fp[0])


def _even_grid(frequency: tuple[float, ...]) -> tuple[np.ndarray, float]:
    """The indices in increasing frequency and the spacing of an even band.

    Raises:
        ValueError: The frequencies are fewer than three, repeated, or not
            evenly spaced; the message gives the resampling call.
    """
    f = np.asarray(frequency, dtype=float)
    order = np.argsort(f, kind="stable")
    fs = f[order]
    if fs.size >= _MIN_POINTS and np.isfinite(fs).all() and fs[-1] > fs[0]:
        df = (fs[-1] - fs[0]) / (fs.size - 1)
        k = (fs - fs[0]) / df
        if (np.abs(k - np.arange(fs.size)) <= _GRID_TOL).all():
            return order, float(df)
    band = hertz_band(float(fs.min()), float(fs.max())) if fs.size else "no frequencies"
    raise ValueError(
        "A band-pass impulse response needs S on at least "
        f"{_MIN_POINTS} evenly spaced frequencies; got {fs.size} over {band}, "
        "not evenly spaced. Solve on np.linspace(f_lo, f_hi, n), or resample S "
        "explicitly through a rational model: "
        "s.fit_rational().network(np.linspace(f_lo, f_hi, n))."
    )


def _reference(
    net: NetworkParameters, z0: Any, what: str
) -> tuple[jax.Array, jax.Array, tuple[str, ...]]:
    """S on one real reference per port, that reference ``(n_ports,)``, excited.

    ``z0`` given: S renormalized to it (every port excited). Otherwise
    ``z_ref`` must be one real value per port over frequency, to round-off
    (the port's mean real value), or NaN for every port (S on unit modal
    power; the reference stays NaN).
    """
    n_ports = len(net.ports)
    z_ref = net.z_ref
    if not is_traced(z_ref):
        _check_defined(net, np.asarray(z_ref), z0, what)
    if z0 is not None:
        if not is_traced(z0):
            arr = np.asarray(z0)
            if (
                np.iscomplexobj(arr)
                or arr.dtype == bool
                or arr.ndim > 1
                or (arr.ndim == 1 and arr.shape != (n_ports,))
                or not np.isfinite(arr.astype(float)).all()
                or not (arr.astype(float) > 0).all()
            ):
                raise ValueError(
                    f"{what} z0 is one real reference impedance > 0 ohm, or one "
                    f"per port ({n_ports}: {', '.join(map(repr, net.ports))}); "
                    f"got {z0!r}."
                )
        z = jnp.broadcast_to(jnp.asarray(z0, dtype=float), (n_ports,))
        renormalized = net.renormalize(z)
        return renormalized.values, z, renormalized.excited
    if is_traced(z_ref):
        # inside a transformation the reference cannot be inspected; it is
        # read as real and constant, as it is checked to be outside one, and
        # takes the same mean value
        return net.values, jnp.real(z_ref).mean(axis=0), net.excited
    arr = np.asarray(z_ref)
    if np.isnan(arr).all():
        return net.values, jnp.full(n_ports, jnp.nan), net.excited
    constant = constant_real(arr)
    if constant is None:
        found = np.unique(np.round(arr, 12))
        raise ValueError(
            f"{what} needs one real reference impedance per port, constant over "
            f"frequency, but z_ref takes the values {found.tolist()} ohm (a "
            "WavePort's Z_PV is complex and frequency dependent). Pass z0= "
            "(e.g. z0=50.0) to renormalize S to one reference first."
        )
    return net.values, jnp.asarray(constant), net.excited


def _check_defined(
    net: NetworkParameters, z_ref: np.ndarray, z0: Any, what: str
) -> None:
    """Refuse a port without a reference impedance where one is needed.

    Every port NaN (with finite S) is a modal network: fine without ``z0``,
    which cannot renormalize it. Some ports NaN: never.
    """
    nan = np.isnan(z_ref).any(axis=0)
    if not nan.any():
        return
    finite = is_traced(net.values) or np.isfinite(np.asarray(net.values)).all()
    if np.isnan(z_ref).all() and finite:
        if z0 is None:
            return
        raise ValueError(
            "No port has a reference impedance (z_ref is NaN: wave ports "
            "without voltage_path), so S cannot be renormalized to z0. Call "
            f"{what} without z0 for the response of S on unit modal power, or "
            "give each WavePort a voltage_path."
        )
    missing = [p for p, bad in zip(net.ports, nan, strict=True) if bad]
    where = np.isnan(z_ref[:, nan]).any(axis=1)
    freqs = ", ".join(
        hertz(float(f))
        for f, bad in zip(np.asarray(net.frequency), where, strict=True)
        if bad
    )
    raise ValueError(
        f"Port {', '.join(map(repr, missing))} has no reference impedance "
        "(z_ref is NaN): a LumpedPort without a resistance, a WavePort "
        "without a voltage_path beside ports with one, or a FloquetPort whose "
        f"order is evanescent at {freqs}. Its waves are undefined, so {what} "
        "has no response for it; give the port a resistance or a "
        "voltage_path, or for a FloquetPort drop the port or narrow the grid."
    )
