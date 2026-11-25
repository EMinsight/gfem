r"""The port mode's normalization as a pure function of its rows, with its pullback.

A port mode's pencil vector :math:`y` (reduced coordinates) and propagation
constant :math:`\gamma` give the field coordinates :math:`z_0 = (y_t/\gamma,
y_n)`. The sweep fixes their scale and phase row by row, over the sorted
frequency grid:

1. **Power**: :math:`z_1 = z_0/\sqrt{|P|}` with the complex power :math:`P`
   of :math:`z_0`; :math:`P/|P|` is the row's power phase.
2. **Phase**: at the first row the quadrant gauge :math:`g = -i(l_n\cdot z_1 +
   \gamma\,l_t\cdot z_1)` (or the largest entry of the field when :math:`g`
   vanishes), after it the overlap with the previous row's final
   coordinates :math:`o = \bar z_{i-1}^T M z_1`; with a polarity functional
   :math:`r\cdot z_1` at every row instead. :math:`z_2 = z_1\,|g|/g`.
3. **Sign**: along a voltage path, :math:`z = \pm z_2` with :math:`\operatorname{Re}
   V \ge 0`.

`normalize_rows` is that map. `normalize_vjp` is its reverse mode in the
convention :math:`\mathrm{d}L = \operatorname{Re}\sum \bar a\,\mathrm{d}a`
(plain, not conjugated, cotangents), written by hand: the overlap step makes
each row depend on every lower row, so the rows are pulled back in reverse
order and each row's cotangent on its final coordinates collects the next
row's overlap term first. The normalization is invariant under :math:`y \to
cy` for every complex :math:`c \ne 0`, so :math:`\bar y^T y = 0`.

The pullback returns, besides :math:`(\bar y, \bar\gamma)` per row, the
cotangents on the forms the normalization reads, as `FormPair` values: the
power forms and the gauge functionals depend on the inverse permeability.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.constants

from gfem._engine.port_mode_sensitivity import FormPair
from gfem._sign import largest_entry

if TYPE_CHECKING:
    from gfem._engine.line_mode_sensitivity import LinePair

#: Relative size of the gauge below which the quadrant gauge is replaced by
#: the field's largest entry.
GAUGE_RTOL = 1e-12


@dataclasses.dataclass(eq=False)
class _Row:
    """What one row's forward leaves for its pullback."""

    z0: np.ndarray
    power: complex
    z1: np.ndarray
    kind: str  # "gauge", "largest", "overlap" or "polarity"
    value: complex  # the complex number the phase is fixed by
    entry: int
    sign: float


@dataclasses.dataclass(eq=False)
class Normalized:
    """The normalized rows of one mode.

    Attributes:
        z: ``(n, nz)`` the field coordinates, 1 W (per metre on a line).
        voltage: ``(n,)`` the voltage of each row, NaN without a path.
        phase: ``(n,)`` the power phase :math:`P/|P|` of each row.
    """

    z: np.ndarray
    voltage: np.ndarray
    phase: np.ndarray
    rows: list[_Row] = dataclasses.field(repr=False)


@dataclasses.dataclass(eq=False)
class NormalizationCotangents:
    """The pullback of `normalize_rows`.

    Attributes:
        y: ``(n, m)`` the cotangent of each row's pencil vector.
        gamma: ``(n,)`` the cotangent of each row's propagation constant.
        reference: The cotangent of the ``reference`` the first row was
            fixed against, or ``None``.
        pairs: The cotangents on the forms the normalization reads (a line
            port's are `LinePair`).
    """

    y: np.ndarray
    gamma: np.ndarray
    reference: np.ndarray | None
    pairs: Sequence[FormPair | LinePair]


def normalize_rows(
    fields: Any,
    mass: Any,
    y: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    *,
    volts: np.ndarray | None = None,
    polarity: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> Normalized:
    r"""The normalized coordinates of one mode's rows.

    Args:
        fields: The space's reduction on the sweep's basis
            (`gfem._engine.port_mode.SectionReduction`: ``w``, ``coords``,
            ``power`` and ``gauge``).
        mass: ``(nz, nz)`` the real mass of the field coordinates, which
            tracks the mode across the grid.
        y: ``(n, m)`` the pencil vector of each row, in the order of the grid.
        gamma: ``(n,)`` the propagation constants, per local unit.
        omega: ``(n,)`` the angular frequencies, rad/s.
        volts: ``(nz,)`` the voltage functional of a mode with a path.
        polarity: ``(nz,)`` a polarity functional :math:`r` (the field at a
            polarity reference's point along its arrow): each row's phase is
            made :math:`r\cdot z` real and positive on its own, with no chain.
        reference: The previous row's coordinates, to continue a grid; the
            first row is fixed by the quadrant gauge without it.
    """
    n = len(y)
    z = np.zeros((n, fields.w.shape[1]), complex)
    voltage = np.full(n, complex(math.nan, math.nan))
    phase = np.ones(n, complex)
    rows: list[_Row] = []
    for i in range(n):
        zi = fields.coords(y[i], gamma[i])
        z0 = zi.copy()
        power = fields.power(zi, gamma[i], omega[i])
        zi /= math.sqrt(abs(power))
        phase[i] = power / abs(power)
        z1 = zi.copy()
        previous = reference if i == 0 else z[i - 1]
        entry = -1
        if polarity is not None:
            kind, value = "polarity", polarity @ zi
            zi *= abs(value) / value
        elif previous is None:
            kind, value = "gauge", fields.gauge(zi, gamma[i], omega[i])
            c = fields.w @ zi
            if abs(value) > GAUGE_RTOL * np.abs(c).max():
                zi *= abs(value) / value
            else:
                entry = largest_entry(c)
                kind, value = "largest", c[entry]
                zi *= abs(value) / value
        else:
            kind, value = "overlap", previous.conj() @ (mass @ zi)
            zi *= abs(value) / value
        sign = 1.0
        if volts is not None:
            voltage[i] = volts @ zi
            if voltage[i].real < 0:
                zi, voltage[i], sign = -zi, -voltage[i], -1.0
        z[i] = zi
        rows.append(_Row(z0, power, z1, kind, value, entry, sign))
    return Normalized(z, voltage, phase, rows)


def normalize_vjp(  # noqa: PLR0915, PLR0917
    fields: Any,
    mass: Any,
    norm: Normalized,
    y: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    *,
    z_bar: np.ndarray,
    voltage_bar: np.ndarray | None = None,
    phase_bar: np.ndarray | None = None,
    volts: np.ndarray | None = None,
    polarity: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> NormalizationCotangents:
    r"""The reverse mode of `normalize_rows` at the rows it returned.

    ``fields``, ``mass``, ``y``, ``gamma``, ``omega``, ``volts``, ``polarity``
    and ``reference`` are the forward's. The cotangents are those of
    :math:`z`, of the voltage and of the power phase (zeros by default;
    ``fields`` must expose its forms, `SectionReduction.forms`).
    """
    n = len(y)
    g_form, t_form, ln, lt = fields.forms()
    scale = fields.space.section.scale
    mu0 = scipy.constants.mu_0
    w = fields.w
    src = fields.source
    tangential = fields.tangential
    z_bar = np.array(z_bar, dtype=complex)  # accumulates the next rows' terms
    voltage_bar = np.zeros(n, complex) if voltage_bar is None else voltage_bar
    phase_bar = np.zeros(n, complex) if phase_bar is None else phase_bar
    y_bar = np.zeros(y.shape, complex)
    gamma_bar = np.zeros(n, complex)
    reference_bar = None
    pairs: list[FormPair] = []
    for i in reversed(range(n)):
        row = norm.rows[i]
        gam = gamma[i]
        zb = z_bar[i]
        if volts is not None:
            zb = zb + voltage_bar[i] * volts
        zb = row.sign * zb
        # the phase factor u = |g| / g: z2 = u z1, dphi = Im(dg / g)
        u = abs(row.value) / row.value
        s = u * (zb @ row.z1)
        g_bar = -1j * s.imag / row.value
        zb1 = u * zb
        if row.kind == "gauge":
            zb1 = zb1 - 1j * g_bar * (ln + gam * lt)
            gamma_bar[i] += -1j * g_bar * (lt @ row.z1)
            lifted = w @ row.z1
            pairs.append(FormPair("gauge_n", None, lifted, -1j * g_bar))
            pairs.append(FormPair("gauge_t", None, lifted, -1j * g_bar * gam))
        elif row.kind == "largest":
            zb1 = zb1 + g_bar * _row(w, row.entry)
        elif row.kind == "polarity":
            zb1 = zb1 + g_bar * polarity
        else:
            previous = norm.z[i - 1] if i else reference
            assert previous is not None  # noqa: S101 - an overlap row has one
            zb1 = zb1 + g_bar * (mass.T @ previous.conj())
            lower = np.conj(g_bar * (mass @ row.z1))
            if i == 0:
                reference_bar = lower
            else:
                z_bar[i - 1] += lower
        # the power: z1 = z0 / sqrt|P|, phase = P / |P|
        modulus = abs(row.power)
        direction = row.power / modulus
        zb0 = zb1 / math.sqrt(modulus)
        slope = -0.5 * modulus**-1.5 * float((zb1 @ row.z0).real)
        p_bar = slope * direction.conjugate()
        p_bar += phase_bar[i] / modulus
        p_bar -= (
            float((phase_bar[i] * direction).real) * direction.conjugate() / modulus
        )
        pref = 0.5 * scale * 1j / (omega[i] * mu0)
        z0 = row.z0
        zc = z0.conj()
        gc = gam.conjugate()
        gz, tz = g_form @ zc, t_form @ zc
        zb0 = zb0 + p_bar * pref * (gz + gc * tz)
        zb0 = zb0 + np.conj(p_bar * pref * (g_form.T @ z0 + gc * (t_form.T @ z0)))
        gamma_bar[i] += np.conj(p_bar * pref * (z0 @ tz))
        lifted0 = w @ z0
        pairs.append(FormPair("power_g", lifted0, lifted0.conj(), p_bar * pref))
        pairs.append(FormPair("power_t", lifted0, lifted0.conj(), p_bar * pref * gc))
        # the coordinates are (y_t / gamma, y_n)
        np.add.at(y_bar[i], src, np.where(tangential, zb0 / gam, zb0))
        gamma_bar[i] += -np.sum(zb0[tangential] * y[i][src[tangential]]) / gam**2
    return NormalizationCotangents(y_bar, gamma_bar, reference_bar, pairs)


def _row(w: Any, e: int) -> np.ndarray:
    """Row ``e`` of the field basis ``w`` as a dense vector."""
    row = w[e]
    return np.asarray(row.todense()).ravel() if hasattr(row, "todense") else row
