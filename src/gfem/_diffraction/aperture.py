"""A field sampled per cell on a metasurface's outgoing plane, and its propagation."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import readonly, register_fields
from gfem.errors import NonDifferentiableError

#: Vacuum speed of light in m/s.
_C0 = 299792458.0
#: Relative spread of the sample spacings that still counts as evenly spaced.
_UNIFORM_RTOL = 1e-6

_CONSTANTS = (
    "ApertureField: {name} must be fixed numbers: sample positions, frequencies, "
    "index and distance are constants of the field. Build them outside "
    "jax.grad, jax.jit or jax.vmap."
)


def _axis(name: str, value: Any) -> np.ndarray:
    """A read-only evenly spaced increasing 1D float axis, or a C3 error."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"ApertureField: {name} must be evenly spaced and increasing (the "
            "propagator samples a uniform grid); it needs at least two finite "
            f"samples in one dimension, got shape {arr.shape}."
        )
    steps = np.diff(arr)
    mean = float(np.mean(steps))
    if mean <= 0 or float(np.max(np.abs(steps - mean))) > _UNIFORM_RTOL * mean:
        raise ValueError(
            f"ApertureField: {name} must be evenly spaced and increasing (the "
            f"propagator samples a uniform grid); its spacings range over "
            f"{steps.min():g} to {steps.max():g} m."
        )
    return readonly(arr)


@dataclasses.dataclass(frozen=True, eq=False)
class ApertureField:
    r"""A tangential electric field sampled once per cell on a parallel plane.

    A metasurface sends out one plane wave per cell, so its outgoing field
    is a grid of samples, one at the centre of each cell. `propagate` carries
    it to another plane by the angular spectrum. The constructor builds
    analytic fields too: an ideal lens, a deflector, a Strehl reference.

    Args:
        values: Complex tangential field in V/m, shape ``(n_freq, 2, n1)``
            for a one-axis aperture or ``(n_freq, 2, n1, n2)``. Axis 1 holds
            the components along $\hat e_1$ and $\hat e_2$ (TM, TE).
        frequency: ``(n_freq,)`` frequencies in Hz.
        x: ``(n1,)`` sample positions along $\hat e_1$ in m, evenly spaced,
            measured from the aperture centre.
        y: ``(n2,)`` positions along $\hat e_2$ in m, or ``None`` for a
            one-axis aperture.
        index: Refractive index of the half-space the field propagates into.
        distance: Distance in m from the outgoing reference plane (0 at the
            aperture).

    Raises:
        ValueError: The shapes, the sample spacing, the frequencies, the
            index or the distance are not valid.
        NonDifferentiableError: A position, frequency, index or distance is
            traced by JAX (only ``values`` is a design-dependent input).
    """

    values: jax.Array
    frequency: np.ndarray
    x: np.ndarray
    y: np.ndarray | None = None
    index: float = 1.0
    distance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("frequency", "x", "y", "index", "distance"):
            if is_traced(getattr(self, name)):
                raise NonDifferentiableError(_CONSTANTS.format(name=name))
        values = jnp.asarray(self.values)
        if not jnp.issubdtype(values.dtype, jnp.complexfloating):
            values = values.astype(jnp.result_type(values.dtype, 1j))
        if values.ndim not in (3, 4) or values.shape[1] != 2:
            raise ValueError(
                "ApertureField: values must have shape (n_freq, 2, n1) or "
                f"(n_freq, 2, n1, n2), got {tuple(values.shape)}."
            )
        one_axis = values.ndim == 3
        n_freq, n1 = values.shape[0], values.shape[2]
        nx = np.size(self.x)
        ny = None if self.y is None else np.size(self.y)
        n2 = None if one_axis else values.shape[3]
        if nx != n1 or ny != n2:
            want = "y=None" if one_axis else f"y of length {n2}"
            ydesc = "y=None" if ny is None else f"y of length {ny}"
            raise ValueError(
                f"ApertureField: values of shape {tuple(values.shape)} need x of "
                f"length {n1} and {want}, got x of length {nx} and {ydesc}."
            )
        x = _axis("x", self.x)
        y = None if self.y is None else _axis("y", self.y)
        freq = np.asarray(self.frequency, dtype=float)
        if freq.shape != (n_freq,) or not np.all(np.isfinite(freq) & (freq > 0)):
            raise ValueError(
                f"ApertureField: frequency must hold {n_freq} frequencies > 0 Hz, "
                f"one per leading entry of values, got {self.frequency!r}."
            )
        index = float(self.index)
        if not (math.isfinite(index) and index > 0):
            raise ValueError(
                f"ApertureField: index must be a finite number > 0, got {self.index!r}."
            )
        distance = float(self.distance)
        if not (math.isfinite(distance) and distance >= 0):
            raise ValueError(
                "ApertureField: distance must be a finite length >= 0 in m, "
                f"got {self.distance!r}."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "frequency", readonly(freq))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "distance", distance)

    @property
    def intensity(self) -> jax.Array:
        r"""$|E_1|^2 + |E_2|^2$, shape ``(n_freq, n1)`` or ``(n_freq, n1, n2)``."""
        return jnp.sum(jnp.abs(self.values) ** 2, axis=1)


register_fields(ApertureField, ("values",))


def _wavenumber_z(q2: np.ndarray, k: float) -> np.ndarray:
    r"""$k_z$: real where propagating, $-i\sqrt{q^2-k^2}$ where evanescent."""
    d = k**2 - q2
    return np.where(
        d >= 0, np.sqrt(np.maximum(d, 0.0)), -1j * np.sqrt(np.maximum(-d, 0.0))
    )


def _transfer(field: ApertureField, distance: float, padded: tuple[int, ...]):
    """Band-limited transfer function, shape ``(n_freq, *padded)``, host NumPy."""
    axes = (field.x,) if field.y is None else (field.x, field.y)
    pitches = [float(a[1] - a[0]) for a in axes]
    qs = [
        2 * np.pi * np.fft.fftfreq(n, p) for n, p in zip(padded, pitches, strict=True)
    ]
    grids = np.meshgrid(*qs, indexing="ij")
    q2 = np.sum([g**2 for g in grids], axis=0)
    out = np.empty((len(field.frequency), *padded), dtype=complex)
    for i, f in enumerate(field.frequency):
        k = 2 * np.pi * field.index * float(f) / _C0
        kz = _wavenumber_z(q2, k)
        h = np.exp(-1j * kz * distance)
        propagating = np.real(kz) > 0
        keep = np.ones(q2.shape, dtype=bool)
        for g, n, p in zip(grids, padded, pitches, strict=True):
            keep &= ~propagating | (
                distance * np.abs(g) <= 0.5 * n * p * np.maximum(np.real(kz), 1e-300)
            )
        out[i] = np.where(keep, h, 0.0)
    return out


def propagate(
    field: ApertureField, distance: float, *, padding: float = 2.0
) -> ApertureField:
    r"""Propagate an aperture field to a parallel plane by the angular spectrum.

    The field is zero-padded to ``padding`` times the aperture's width along
    each axis, every plane-wave component is multiplied by $e^{-ik_zd}$ with
    $k_z=\sqrt{k^2-q^2}$ (propagating) or $-i\sqrt{q^2-k^2}$ (evanescent,
    $e^{-|k_z|d}$) and $k = 2\pi n f/c_0$, the components whose walk-off
    $d|q_i|/k_z$ exceeds half the padded window $L_i$ along an axis are
    dropped (Matsushima and Shimobaba's band limit, which keeps the
    periodic window's wrap-around out of the result), and the field is
    cropped back to the input grid. A one-axis aperture propagates along
    $\hat e_1$ only.

    The transfer function is a host NumPy constant, so the whole map is a
    linear JAX operation on ``values``: `jax.jit`, `jax.grad` and `jax.vmap`
    work through it. Padding 2 keeps the focal spot of a lens within
    $1.4\times10^{-4}$ of the peak and the field across the footprint
    within $10^{-3}$; padding 3 gives $4\times10^{-5}$.

    Args:
        field: The field to propagate.
        distance: Propagation distance in m, $\ge 0$; a fixed number.
        padding: FFT window as a multiple of the aperture's width, $\ge 1$;
            a fixed number.

    Returns:
        The field on the plane ``field.distance + distance``, on the input
        grid.

    Raises:
        TypeError: ``field`` is not an `ApertureField`.
        ValueError: ``distance`` or ``padding`` is not valid.
        NonDifferentiableError: ``distance`` or ``padding`` is traced.
    """
    if not isinstance(field, ApertureField):
        raise TypeError(
            "propagate needs a gfem.diffraction.ApertureField, got "
            f"{type(field).__name__}."
        )
    if is_traced(distance) or is_traced(padding):
        raise NonDifferentiableError(
            "propagate: distance and padding must be fixed numbers: the transfer "
            "function is built once, outside the trace. Build them outside "
            "jax.grad, jax.jit or jax.vmap."
        )
    d = _real(distance)
    if d is None or d < 0:
        raise ValueError(
            f"propagate: distance must be a finite length >= 0 in m, got {distance!r}."
        )
    pad = _real(padding)
    if pad is None or pad < 1:
        raise ValueError(
            "propagate: padding must be a finite number >= 1 (the FFT window as "
            f"a multiple of the aperture's width), got {padding!r}."
        )
    nd = field.values.ndim - 2
    shape = field.values.shape[2:]
    padded = tuple(math.ceil(n * pad) for n in shape)
    h = _transfer(field, d, padded)[:, None]
    pads = [(0, 0), (0, 0)] + [
        ((b - s) // 2, b - s - (b - s) // 2) for s, b in zip(shape, padded, strict=True)
    ]
    axes = tuple(range(-nd, 0))
    spec = jnp.fft.fftn(jnp.pad(field.values, pads), axes=axes)
    out = jnp.fft.ifftn(spec * h, axes=axes)
    crop = (
        slice(None),
        slice(None),
        *(slice(p[0], p[0] + s) for p, s in zip(pads[2:], shape, strict=True)),
    )
    return ApertureField(
        out[crop], field.frequency, field.x, field.y, field.index, field.distance + d
    )


def _real(value: Any) -> float | None:
    """``value`` as a finite float, or ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
