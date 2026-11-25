"""A field sampled per cell on a metasurface's outgoing plane, and its propagation."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._diffraction._polarization import JONES_ORDER
from gfem._diffraction.outputs import (
    Cell,
    _runs,
    check_offsets,
    check_surface,
    incidence_of,
    jones_block,
)
from gfem._pytree import readonly, register_fields
from gfem._text import hertz
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gfem._diffraction.response import CellResponse

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
    """A read-only evenly spaced increasing 1D float axis, else `ValueError`."""
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
            the components along :math:`\hat e_1` and :math:`\hat e_2` (TM, TE).
        frequency: ``(n_freq,)`` frequencies in Hz.
        x: ``(n1,)`` sample positions along :math:`\hat e_1` in m, evenly spaced,
            measured from the aperture centre.
        y: ``(n2,)`` positions along :math:`\hat e_2` in m, or ``None`` for a
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
        r""":math:`|E_1|^2 + |E_2|^2`.

        Shape ``(n_freq, n1)`` or ``(n_freq, n1, n2)``.
        """
        return jnp.sum(jnp.abs(self.values) ** 2, axis=1)


register_fields(ApertureField, ("values",))


def _wavenumber_z(q2: np.ndarray, k: float) -> np.ndarray:
    r""":math:`k_z`, real where propagating, :math:`-i\sqrt{q^2-k^2}` where not."""
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
    each axis, every plane-wave component is multiplied by :math:`e^{-ik_zd}` with
    :math:`k_z=\sqrt{k^2-q^2}` (propagating) or :math:`-i\sqrt{q^2-k^2}` (evanescent,
    :math:`e^{-|k_z|d}`) and :math:`k = 2\pi n f/c_0`, the components whose walk-off
    :math:`d|q_i|/k_z` exceeds half the padded window :math:`L_i` along an axis are
    dropped (Matsushima and Shimobaba's band limit, which keeps the
    periodic window's wrap-around out of the result), and the field is
    cropped back to the input grid. A one-axis aperture propagates along
    :math:`\hat e_1` only.

    The transfer function is a host NumPy constant, so the whole map is a
    linear JAX operation on ``values``: `jax.jit`, `jax.grad` and `jax.vmap`
    work through it. Padding 2 keeps the focal spot of a lens within
    :math:`1.4\times10^{-4}` of the peak and the field across the footprint
    within :math:`1.1\times10^{-3}`; padding 3 gives :math:`4\times10^{-5}`.

    Args:
        field: The field to propagate.
        distance: Propagation distance in m, :math:`\ge 0`; a fixed number.
        padding: FFT window as a multiple of the aperture's width, :math:`\ge 1`;
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


# --- the aperture field of a cell response ---------------------------------------

_WHO = "aperture_field"
#: How far from a right angle (|cos|) the two lattice vectors may sit.
_RECTANGULAR_ATOL = 1e-9


def _check_lattice_rectangular(
    response: CellResponse, outgoing: str
) -> tuple[float, float]:
    """The pitches along the two lattice vectors of ``outgoing`` (rectangular only)."""
    a1, a2 = (np.asarray(v, dtype=float) for v in response.lattice[outgoing])
    n1, n2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    cos = float(a1 @ a2) / (n1 * n2)
    if abs(cos) > _RECTANGULAR_ATOL:
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        raise ValueError(
            f"{_WHO}: the lattice vectors of {outgoing!r} make {angle:.6g}°; "
            "the aperture places cells on a rectangular grid along ê1 and "
            "ê2."
        )
    return n1, n2


def _check_not_diffracting(
    response: CellResponse, outgoing: str, index: float, pitches: tuple[float, float]
) -> None:
    """Raise `ValueError` if a non-specular order propagates behind ``outgoing``."""
    freqs = np.asarray(response.frequency, dtype=float)
    k = 2 * np.pi * index * freqs / _C0
    b = 2 * np.pi / np.asarray(pitches)
    reach = int(np.ceil(np.max(k) / np.min(b))) + 1
    found: dict[tuple[int, int], list[int]] = {}
    for m in range(-reach, reach + 1):
        for n in range(-reach, reach + 1):
            if (m, n) == (0, 0):
                continue
            kappa = math.hypot(m * b[0], n * b[1])
            idx = [i for i in range(len(freqs)) if kappa < k[i]]
            if idx:
                found[(m, n)] = idx
    if not found:
        return
    order = min(found, key=lambda o: (abs(o[0]) + abs(o[1]), o))
    fmax = float(np.max(freqs[found[order]]))
    raise ValueError(
        f"{_WHO}: the cells diffract: order {order} propagates behind "
        f"{outgoing!r} at {_runs(found[order], freqs)}, so a cell does not send "
        "out one plane wave and the locally periodic approximation does not "
        "hold. Keep the pitch below the wavelength in that half-space "
        f"({_C0 / (index * fmax):g} m at {hertz(fmax)})."
    )


def _outgoing_ports(cell: Cell, outgoing: str) -> list[str]:
    """The (0, 0) TM and TE ports of ``outgoing``."""
    check_surface(cell, outgoing)
    names = []
    for pol in JONES_ORDER:
        name = cell.port(outgoing, (0, 0), pol)
        if name is None:
            raise ValueError(
                f"{_WHO}: no Floquet port reads order (0, 0) in {pol} on "
                f"{outgoing!r} (an aperture field needs both polarizations of "
                f"the outgoing order). Add gfem.FloquetPort(name, {outgoing!r}, "
                f"order=(0, 0), polarization={pol!r}) to the model."
            )
        names.append(name)
    check_offsets(cell, outgoing, (0, 0))
    return names


def _check_range(response: CellResponse, parameter: np.ndarray) -> None:
    """Raise `ValueError` when a concrete parameter map leaves the sampled range."""
    lo, hi = float(response.parameter[0]), float(response.parameter[-1])
    outside = ~((parameter >= lo) & (parameter <= hi))
    if np.any(outside):
        raise ValueError(
            f"{_WHO}: {int(outside.sum())} of {parameter.size} cells have a "
            f"parameter outside the sampled range [{lo:g}, {hi:g}] (min "
            f"{np.nanmin(parameter):g}, max {np.nanmax(parameter):g}); the cell "
            "response does not extrapolate. Map an unbounded design variable "
            "onto the range, e.g. lo + (hi - lo) * jax.nn.sigmoid(x)."
        )


def aperture_field(
    response: CellResponse,
    parameter: ArrayLike,
    outgoing: str,
    incident: str,
    *,
    polarization: Any,
    amplitude: ArrayLike = 1.0,
) -> ApertureField:
    r"""The field a metasurface sends out, in the locally periodic approximation.

    Each cell is replaced by the periodic array of itself and radiates the
    one plane wave of the infinite array; the field behind the metasurface is
    one sample per cell, at the cell's centre,

    .. math::

        \mathbf E_\text{out}(\mathbf x_j) = a_j\,J\big(p_j\big)\,
        \mathbf e_\text{in},

    with :math:`p_j` the meta-atom parameter of cell :math:`j`,
    :math:`a_j` the complex envelope of the incident light, :math:`J(p)` the
    :math:`2\times2` (TM, TE) block of the cell response between the
    specular orders of ``incident`` and ``outgoing`` (the Jones matrix
    `gfem.diffraction.jones` reads from a solve; the response interpolates it
    at :math:`p_j`, as `CellResponse` does), and
    :math:`\mathbf e_\text{in}` the incident polarization (circular states
    through :math:`U_{h_\text{in}}`, as for `gfem.diffraction.jones`). At
    normal incidence the TM and TE references are :math:`\hat e_1` and
    :math:`\hat e_2` in every cell, so the two components are the tangential
    components of the field and each obeys the scalar Helmholtz equation:
    `propagate` carries them separately. Reflection (a reflectarray) is
    ``outgoing == incident``; the field then propagates back into the
    incident half-space.

    The approximation neglects the coupling between neighbouring cells. It
    holds where the parameter varies slowly over the aperture: the error
    grows with the phase step between neighbours, about 0.01 to 0.02 in
    amplitude up to steps of :math:`40^\circ`, 0.1 for a :math:`65^\circ` ramp
    with one reset, and it fails for steps of :math:`100^\circ` or more, such as the
    :math:`2\pi` reset of a deflector's phase ramp. A uniform map is exact.
    Samples sit at the cell centres, not on a staircase of uniform patches:
    the sampled field is band-limited to :math:`|q_i| \le \pi/a_i`, and
    the sampling matches a full solve of a few cells better than the
    staircase for resonant meta-atoms.

    Cell :math:`j` of an axis of :math:`n` cells sits at
    :math:`x_j = (j - \tfrac{n-1}{2})\,a`, measured from the aperture centre
    (the lattice pitch :math:`a` is read from the response): with an even
    count the centre falls between two samples. The field is on the reference
    plane of the outgoing port (its offset, where the phase of S is read),
    which is distance 0 of the result.

    Args:
        response: A `CellResponse` of a rectangular lattice at normal
            incidence whose cells send out the specular order only.
        parameter: The meta-atom parameter of each cell, in the units of
            ``response.parameter``: shape ``(n1,)`` (a row of cells along
            :math:`\hat e_1`, uniform along :math:`\hat e_2`: deflectors,
            cylindrical lenses) or ``(n1, n2)`` (along :math:`\hat e_1`,
            :math:`\hat e_2`, ``indexing="ij"``). Differentiable. A concrete
            map outside the sampled range raises; a traced one (``jit``,
            ``vmap``) gives NaN in the cells outside it.
        outgoing: The Floquet surface the field leaves through.
        incident: The Floquet surface the light enters through.
        polarization: ``"TE"``, ``"TM"``, ``"RCP"``, ``"LCP"`` or a Jones
            pair ``(a_TM, a_TE)``; a fixed value.
        amplitude: Complex envelope of the incident light per cell
            (Gaussian illumination, an aperture stop), broadcast to the shape
            of ``parameter``. Differentiable.

    Returns:
        The `ApertureField`, ``values`` in the power-wave units of the
        incident plane wave: :math:`|E|^2` of a normally leaving wave is the
        intensity relative to the incident one.

    Raises:
        TypeError: ``response`` is not a `CellResponse`.
        ValueError: ``parameter`` is not one or two axes of at least two cells,
            ``amplitude`` does not broadcast to it, a concrete parameter lies
            outside the sampled range, the response was sampled at oblique
            incidence, its lattice is not rectangular, a non-specular order
            propagates behind ``outgoing``, ``polarization`` is
            ``"unpolarized"`` (an incoherent mean, not one field) or not
            valid, or a Floquet port the block needs is missing, not excited
            or at a different offset from its partner.
        NonDifferentiableError: ``polarization`` is traced.
    """
    from gfem._diffraction.response import CellResponse, interpolate

    if not isinstance(response, CellResponse):
        raise TypeError(
            "aperture_field needs a gfem.diffraction.CellResponse, got "
            f"{type(response).__name__}."
        )
    if isinstance(polarization, str) and polarization == "unpolarized":
        raise ValueError(
            "aperture_field: polarization='unpolarized' is an incoherent mean, "
            "not one field; build the aperture field for 'TE' and for 'TM', "
            "propagate both and average their intensities."
        )
    shape = tuple(np.shape(parameter))
    if len(shape) not in (1, 2) or min(shape, default=0) < 2:
        raise ValueError(
            "aperture_field: the parameter map must have one axis (cells along "
            "ê1) or two (cells along ê1, ê2), with at least two "
            f"cells along each, got shape {shape}."
        )
    try:
        broadcast_ok = np.broadcast_shapes(np.shape(amplitude), shape) == shape
    except ValueError:
        broadcast_ok = False
    if not broadcast_ok:
        raise ValueError(
            f"aperture_field: amplitude of shape {np.shape(amplitude)} does not "
            f"broadcast to the parameter map's shape {shape}."
        )
    if response.wave_vector is not None and any(response.wave_vector):
        kx, ky, kz = response.wave_vector
        raise ValueError(
            "aperture_field: the cell response was sampled at oblique incidence "
            f"(wave vector ({kx:g}, {ky:g}, {kz:g}) rad/m); the aperture field "
            "assumes normal incidence, where every cell's TM and TE references "
            "are ê1 and ê2."
        )
    cell = Cell(
        who=_WHO,
        results=(_Labels(response.ports, response.excited),),
        sequence=False,
        model=None,
        floquet=response.ports,
        others=(),
        info=dict(response.diffraction_orders),
        offsets=dict(response.offsets),
        _directions=dict(response.direction),
    )
    rows = _outgoing_ports(cell, outgoing)
    pitches = _check_lattice_rectangular(response, outgoing)
    index = float(response.diffraction_orders[rows[0]].index)
    _check_not_diffracting(response, outgoing, index, pitches)
    inc = incidence_of(cell, incident, polarization, (0, 0))
    ((_, pair),) = inc.combos
    cols = [pol for pol, a in zip(JONES_ORDER, pair, strict=True) if a != 0]
    # `incidence_of` has refused any column without a port
    names = [str(cell.port(incident, (0, 0), pol)) for pol in cols]
    weights = jnp.asarray([pair[JONES_ORDER.index(pol)] for pol in cols])
    if not is_traced(parameter):
        arr = np.asarray(parameter, dtype=float)
        _check_range(response, arr)
        parameter = arr
    p = jnp.asarray(parameter, dtype=float)

    # interpolate the entries the block reads only: 2 rows x the used columns,
    # cut out by the extractor `jones` uses
    def block(a):
        return jax.vmap(
            lambda v: jones_block(
                _Labels(response.ports, response.excited, v), rows, names
            )
        )(a)

    values = block(response.values)
    derivative = block(response.derivative)
    per_cell = jax.vmap(lambda q: interpolate(response, q, values, derivative))(
        p.ravel()
    )
    # (cells, n_freq, 2, n_columns) -> (cells, n_freq, 2)
    field = jnp.einsum("cfrk,k->cfr", per_cell, weights)
    field = jnp.moveaxis(field, 0, -1).reshape(field.shape[1], 2, *shape)
    field = field * jnp.broadcast_to(jnp.asarray(amplitude), shape)
    axes = [
        (np.arange(n) - (n - 1) / 2) * a for n, a in zip(shape, pitches, strict=False)
    ]
    return ApertureField(
        field,
        response.frequency,
        axes[0],
        axes[1] if len(shape) == 2 else None,
        index,
        0.0,
    )


@dataclasses.dataclass(frozen=True)
class _Labels:
    """The port labels of a cell response, shaped as a result's ``s``."""

    ports: tuple[str, ...]
    excited: tuple[str, ...]
    values: Any = None

    @property
    def s(self) -> _Labels:
        return self
