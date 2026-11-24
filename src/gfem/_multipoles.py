r"""Multipole decomposition of a scattered field: result types and the projection.

The scattered field of an incident-field column outside a sphere that holds
every source is expanded in outgoing vector spherical waves,

.. math::

    E_s = E_\mathrm{inc} \sum_{l\ge1}\sum_{m=-l}^{l}
    \big[\alpha_{lm} N^{(2)}_{lm} + \beta_{lm} M^{(2)}_{lm}\big],

with the conventions of `gfem.conventions` (Multipoles). This module holds
what needs no solved field:

- `Multipoles` and `CurrentMoments`, the frozen results and their
  properties (the partial cross sections, the exact moments as fixed linear
  maps of the :math:`l\le2` coefficients);
- `_gauss_grid` and `_project`, the closed-form projection of far-field
  amplitudes sampled on a Gauss-Legendre by trapezoid grid onto the
  coefficients, a constant linear map written in `jax.numpy`;
- `_wiscombe`, the default degree.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import numbers
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import sph_harm_y

from gfem._pytree import readonly, register_fields

__all__ = ["CurrentMoments", "Multipoles"]


class _Grid(NamedTuple):
    """Directions and quadrature weights, each ``(n_theta, 2 * n_theta)``."""

    theta: np.ndarray
    phi: np.ndarray
    weight: np.ndarray


def _gauss_grid(n_theta: int) -> _Grid:
    r"""Gauss-Legendre in :math:`\cos\theta` by trapezoid in :math:`\phi`.

    ``n_theta`` polar nodes (never on a pole) and ``2 * n_theta`` azimuths
    from 0; the weights sum to :math:`4\pi`. The rule is exact for every
    spherical harmonic product of total degree up to ``2 * n_theta - 1``.
    """
    if n_theta < 1:
        raise ValueError(f"n_theta must be at least 1, got {n_theta}.")
    x, w = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 2 * n_theta
    phi = np.arange(n_phi) * (2 * math.pi / n_phi)
    theta_g, phi_g = np.meshgrid(np.arccos(x), phi, indexing="ij")
    weight = np.outer(w, np.full(n_phi, 2 * math.pi / n_phi))
    return _Grid(theta_g, phi_g, weight)


def _wiscombe(x: float) -> int:
    r"""Wiscombe's degree count :math:`\max(2, \lceil x + 4x^{1/3} + 2\rceil)`."""
    return max(2, math.ceil(x + 4 * x ** (1 / 3) + 2))


def _flat_lm(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    ls = [l for l in range(1, lmax + 1) for _ in range(-l, l + 1)]
    ms = [m for l in range(1, lmax + 1) for m in range(-l, l + 1)]
    return np.array(ls, dtype=int), np.array(ms, dtype=int)


def _y(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Condon-Shortley :math:`Y_{lm}`, zero for :math:`|m| > l`."""
    if abs(m) > l:
        return np.zeros(np.broadcast(theta, phi).shape, complex)
    return sph_harm_y(l, m, theta, phi)


def _x_lm(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Jackson's :math:`X_{lm}` as its :math:`(\hat\theta, \hat\phi)` components."""
    s = math.sqrt(l * (l + 1))
    y = _y(l, m, theta, phi)
    dy = m / np.tan(theta) * y + math.sqrt((l - m) * (l + m + 1)) * np.exp(
        -1j * phi
    ) * _y(l, m + 1, theta, phi)
    return -m * y / (np.sin(theta) * s), -1j * dy / s


@functools.lru_cache(maxsize=4)
def _basis(n_theta: int, lmax: int) -> tuple[np.ndarray, ...]:
    r"""The projection matrices for ``n_theta`` and ``lmax``.

    With :math:`F` the far-field amplitude, the absolute coefficients are
    :math:`c^E_{lm} = k\,i^{-l}\oint\overline{\hat r\times X_{lm}}\cdot F\,d\Omega`
    and :math:`c^M_{lm} = k\,i^{-(l+1)}\oint\overline{X_{lm}}\cdot F\,d\Omega`;
    the four matrices hold everything but :math:`k`, ``(n_lm, n_theta * n_phi)``
    each: electric on :math:`F_\theta`, electric on :math:`F_\phi`, magnetic
    on :math:`F_\theta`, magnetic on :math:`F_\phi`.
    """
    grid = _gauss_grid(n_theta)
    th, ph, w = (a.ravel() for a in grid)
    ls, ms = _flat_lm(lmax)
    out = [np.empty((len(ls), th.size), complex) for _ in range(4)]
    for j, (l, m) in enumerate(zip(ls, ms, strict=True)):
        xt, xp = _x_lm(int(l), int(m), th, ph)
        # rhat x X has components (-X_phi, X_theta).
        ce, cm = (1j) ** (-int(l)), (1j) ** (-int(l) - 1)
        out[0][j] = w * ce * np.conj(-xp)
        out[1][j] = w * ce * np.conj(xt)
        out[2][j] = w * cm * np.conj(xt)
        out[3][j] = w * cm * np.conj(xp)
    for a in out:
        a.flags.writeable = False
    return tuple(out)


def _project(
    e_theta: Any,
    e_phi: Any,
    grid: _Grid,
    *,
    k: Any,
    lmax: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Absolute spherical coefficients of far-field amplitudes on a Gauss grid.

    Args:
        e_theta: :math:`F_\theta` (V) on ``grid``, ``(n_freq, *grid.theta.shape)``.
        e_phi: :math:`F_\phi` (V), same shape.
        grid: The directions and weights of `_gauss_grid`.
        k: The background wavenumber per frequency, rad/m, ``(n_freq,)``.
        lmax: The highest degree.

    Returns:
        ``(c_alpha, c_beta)``, :math:`E_\mathrm{inc}\alpha_{lm}` and
        :math:`E_\mathrm{inc}\beta_{lm}` in V/m, each ``(n_freq, lmax * (lmax + 2))``
        in the flat order :math:`j = l^2 + l + m - 1`. Linear in the
        amplitudes, so a gradient flows through it.
    """
    if grid.theta.ndim != 2 or grid.theta.shape[1] != 2 * grid.theta.shape[0]:
        raise ValueError("grid must come from _gauss_grid.")
    ft = jnp.reshape(jnp.asarray(e_theta, complex), (jnp.shape(e_theta)[0], -1))
    fp = jnp.reshape(jnp.asarray(e_phi, complex), (jnp.shape(e_phi)[0], -1))
    if ft.shape[1] != grid.theta.size or fp.shape != ft.shape:
        raise ValueError(
            f"The amplitudes must have {grid.theta.size} directions per "
            f"frequency (the grid's), got {ft.shape[1]} and {fp.shape[1]}."
        )
    ket, kep, kmt, kmp = (jnp.asarray(a) for a in _basis(grid.theta.shape[0], lmax))
    kk = jnp.reshape(jnp.asarray(k, float), (-1, 1))
    c_alpha = kk * (ft @ ket.T + fp @ kep.T)
    c_beta = kk * (ft @ kmt.T + fp @ kmp.T)
    return c_alpha, c_beta


def _spherical_unit_vectors() -> np.ndarray:
    r""":math:`\hat e_{-1}, \hat e_0, \hat e_{+1}` as rows."""
    return np.array(
        [
            [1, -1j, 0],
            [0, 0, 1.0],
            [-1, -1j, 0],
        ]
    ) / np.array([[math.sqrt(2)], [1.0], [math.sqrt(2)]])


def _clebsch_gordan_11(m1: int, m2: int) -> float:
    r""":math:`\langle 1\,m_1\,1\,m_2\,|\,2\,(m_1+m_2)\rangle`."""
    if m1 == m2 != 0:
        return 1.0
    if {m1, m2} in ({1, 0}, {-1, 0}):
        return 1 / math.sqrt(2)
    if {m1, m2} == {1, -1}:
        return 1 / math.sqrt(6)
    return math.sqrt(2 / 3)  # m1 == m2 == 0


def _rank2_tensors() -> np.ndarray:
    r""":math:`T_{2M}`, :math:`M = -2..2`, ``(5, 3, 3)``."""
    e = _spherical_unit_vectors()  # index m + 1
    out = np.zeros((5, 3, 3), complex)
    for big_m in range(-2, 3):
        for m1 in (-1, 0, 1):
            m2 = big_m - m1
            if abs(m2) <= 1:
                out[big_m + 2] += _clebsch_gordan_11(m1, m2) * np.outer(
                    e[m1 + 1], e[m2 + 1]
                )
    return out


_E_HAT = _spherical_unit_vectors()
_T2 = _rank2_tensors()
# Flat index j = l^2 + l + m - 1: the l = 1 block is 0..2, the l = 2 block 3..7.
_DEGREE_ONE = slice(0, 3)
_DEGREE_TWO = slice(3, 8)


def _format_origin(origin: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{x:.4g}" for x in origin) + ") m"


@dataclasses.dataclass(frozen=True, eq=False)
class Multipoles:
    r"""The spherical multipoles of a scattered field, per kept frequency.

    Under :math:`e^{+i\omega t}` the scattered field outside a sphere that
    holds every source is

    .. math::

        E_s = E_\mathrm{inc} \sum_{l\ge1}\sum_{m=-l}^{l}
        \big[\alpha_{lm} N^{(2)}_{lm} + \beta_{lm} M^{(2)}_{lm}\big],

    with :math:`M^{(2)}_{lm} = h^{(2)}_l(kr) X_{lm}`, :math:`N^{(2)}_{lm} =
    k^{-1}\nabla\times M^{(2)}_{lm}`, Jackson's :math:`X_{lm}` and
    Condon-Shortley harmonics (`gfem.conventions`, Multipoles), about
    ``origin``. The coefficients are dimensionless and power-normalized: the
    partial scattering cross section of an electric (magnetic) multipole of
    degree :math:`l` is :math:`k^{-2}\sum_m|\alpha_{lm}|^2`
    (:math:`k^{-2}\sum_m|\beta_{lm}|^2`). Every array has the kept
    frequencies first, and the coefficients are stored :math:`l`-major,
    :math:`m` ascending: flat index :math:`j = l^2 + l + m - 1`.

    Attributes:
        frequency: The kept frequencies of the column, Hz.
        origin: The multipole origin, m, in model coordinates.
        lmax: The highest degree, the same at every frequency.
        wavenumber: The background :math:`k`, rad/m, one per frequency.
        impedance: The background :math:`\eta`, ohm.
        incident_amplitude: :math:`E_\mathrm{inc}`, V/m, ``(n_freq,)``,
            complex: a plane wave's amplitude, a beam's focal field
            :math:`\sqrt{2\eta I_0}`, :math:`\sqrt{2\eta I}` of an
            `gfem.IncidentField`. Not `gfem.FarField.incident_amplitude`,
            which is the real :math:`|E_0|` of a plane wave and ``None`` of a
            beam.
        alpha: :math:`\alpha_{lm}` on :math:`N^{(2)}_{lm}`, the electric
            (transverse magnetic) coefficients, ``(n_freq, lmax * (lmax + 2))``.
        beta: :math:`\beta_{lm}` on :math:`M^{(2)}_{lm}`, the magnetic
            (transverse electric) coefficients, same shape.
    """

    frequency: tuple[float, ...]
    origin: tuple[float, float, float]
    lmax: int
    wavenumber: tuple[float, ...]
    impedance: float
    incident_amplitude: jax.Array
    alpha: jax.Array
    beta: jax.Array

    def __post_init__(self) -> None:
        if not isinstance(self.lmax, int) or self.lmax < 2:
            raise ValueError(
                f"lmax must be an integer of at least 2, got {self.lmax!r}."
            )
        n_freq, n_lm = len(self.frequency), self.lmax * (self.lmax + 2)
        if len(self.wavenumber) != n_freq or len(self.origin) != 3:
            raise ValueError(
                "frequency and wavenumber must have one entry per kept "
                "frequency and origin three."
            )
        for name in ("incident_amplitude", "alpha", "beta"):
            want = (n_freq,) if name == "incident_amplitude" else (n_freq, n_lm)
            if tuple(jnp.shape(getattr(self, name))) != want:
                raise ValueError(
                    f"{name} must have shape {want}, got "
                    f"{tuple(jnp.shape(getattr(self, name)))}."
                )

    # -- indices -----------------------------------------------------------
    @property
    def l(self) -> np.ndarray:  # noqa: E743
        """The degree of each flat index, a read-only integer array."""
        return _flat_index_arrays(self.lmax)[0]

    @property
    def m(self) -> np.ndarray:
        """The azimuthal index of each flat index, a read-only integer array."""
        return _flat_index_arrays(self.lmax)[1]

    def index(self, l: int, m: int) -> int:
        r"""The flat index :math:`j = l^2 + l + m - 1` of :math:`(l, m)`.

        ``mp.alpha[:, mp.index(2, -1)]`` is :math:`\alpha_{2,-1}`;
        ``mp.alpha[:, mp.l == 2]`` is the whole degree.

        Raises:
            TypeError: ``l`` or ``m`` is not an integer (``bool`` included).
            ValueError: :math:`(l, m)` is not in the expansion.
        """
        for name, v in (("l", l), ("m", m)):
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise TypeError(
                    f"Multipoles.index: {name} must be an int, got {type(v).__name__}."
                )
        if not (1 <= l <= self.lmax and -l <= m <= l):
            raise ValueError(
                f"Multipoles.index: (l, m) = ({l}, {m}) is not a multipole of "
                f"this result: 1 <= l <= lmax = {self.lmax} and -l <= m <= l."
            )
        return int(l * l + l + m - 1)

    # -- derived outputs ---------------------------------------------------
    def _k(self) -> np.ndarray:
        return np.asarray(self.wavenumber, float)

    def _per_frequency(self, x: Any, ndim: int) -> Any:
        return jnp.reshape(x, (-1,) + (1,) * (ndim - 1))

    @property
    def intensity(self) -> jax.Array:
        r"""The incident intensity :math:`|E_\mathrm{inc}|^2/(2\eta)`, W/m²."""
        e = self.incident_amplitude
        return (e.real**2 + e.imag**2) / (2 * self.impedance)

    def _partial(self, c: jax.Array) -> jax.Array:
        ls = _flat_index_arrays(self.lmax)[0]
        onehot = jnp.asarray(ls[:, None] == np.arange(1, self.lmax + 1)[None])
        k2 = jnp.asarray(self._k() ** 2)[:, None]
        return ((c.real**2 + c.imag**2) @ onehot.astype(float)) / k2

    @property
    def electric_scattering(self) -> jax.Array:
        r""":math:`\sigma^E_l = k^{-2}\sum_m|\alpha_{lm}|^2`, m², ``(n_freq, lmax)``.

        Column :math:`l - 1`: dipole, quadrupole, octupole for :math:`l = 1,
        2, 3`. Equal to the cross sections of the exact :math:`p` and
        :math:`Q^e` for :math:`l = 1, 2`.
        """
        return self._partial(self.alpha)

    @property
    def magnetic_scattering(self) -> jax.Array:
        r""":math:`\sigma^M_l = k^{-2}\sum_m|\beta_{lm}|^2`, m², ``(n_freq, lmax)``."""
        return self._partial(self.beta)

    @property
    def scattering(self) -> jax.Array:
        r"""The sum of the partial scattering cross sections, m², ``(n_freq,)``.

        The far-field power over :math:`I` up to ``lmax``. It is a different
        discretization from `gfem.cross_sections`' scattering cross
        section and differs from it by a fraction of a percent at order 3.
        """
        return jnp.sum(self.electric_scattering, axis=1) + jnp.sum(
            self.magnetic_scattering, axis=1
        )

    def _epsilon(self) -> np.ndarray:
        omega = 2 * math.pi * np.asarray(self.frequency, float)
        return self._k() / (omega * self.impedance)

    @property
    def electric_dipole(self) -> jax.Array:
        r"""The exact electric dipole :math:`p`, C·m, ``(n_freq, 3)``.

        :math:`p = -\sqrt{6\pi}\,\varepsilon E_\mathrm{inc} k^{-3}
        \sum_m\alpha_{1m}\hat e_m`, in the model's :math:`x, y, z`.
        """
        s = jnp.asarray(_E_HAT)
        scale = -math.sqrt(6 * math.pi) * jnp.asarray(self._epsilon() / self._k() ** 3)
        return (
            scale[:, None]
            * self.incident_amplitude[:, None]
            * (self.alpha[:, _DEGREE_ONE] @ s)
        )

    @property
    def magnetic_dipole(self) -> jax.Array:
        r"""The exact magnetic dipole :math:`m`, A·m², ``(n_freq, 3)``.

        :math:`m = -i\sqrt{6\pi}\,E_\mathrm{inc}(k^{3}\eta)^{-1}
        \sum_m\beta_{1m}\hat e_m`.
        """
        s = jnp.asarray(_E_HAT)
        scale = (
            -1j
            * math.sqrt(6 * math.pi)
            / (jnp.asarray(self._k() ** 3) * self.impedance)
        )
        return (
            scale[:, None]
            * self.incident_amplitude[:, None]
            * (self.beta[:, _DEGREE_ONE] @ s)
        )

    @property
    def electric_quadrupole(self) -> jax.Array:
        r"""The exact electric quadrupole :math:`Q^e`, C·m², ``(n_freq, 3, 3)``.

        Symmetric and traceless, with Jackson's normalization
        :math:`\int\rho(3r_\alpha r_\beta - r^2\delta_{\alpha\beta})\,dV` in
        the long-wavelength limit.
        """
        t = jnp.asarray(_T2)
        scale = (
            -12 * math.sqrt(5 * math.pi) * jnp.asarray(self._epsilon() / self._k() ** 4)
        )
        return (
            scale[:, None, None]
            * self.incident_amplitude[:, None, None]
            * (jnp.einsum("fM,Mab->fab", self.alpha[:, _DEGREE_TWO], t))
        )

    @property
    def magnetic_quadrupole(self) -> jax.Array:
        r"""The exact magnetic quadrupole :math:`Q^m`, A·m³, ``(n_freq, 3, 3)``.

        Symmetric and traceless; long-wavelength limit
        :math:`\int[r_\alpha(r\times J)_\beta + r_\beta(r\times J)_\alpha]\,dV`,
        three times the magnetic quadrupole of Evlyukhin and the anapole
        review.
        """
        t = jnp.asarray(_T2)
        scale = (
            -12j
            * math.sqrt(5 * math.pi)
            / (jnp.asarray(self._k() ** 4) * self.impedance)
        )
        return (
            scale[:, None, None]
            * self.incident_amplitude[:, None, None]
            * (jnp.einsum("fM,Mab->fab", self.beta[:, _DEGREE_TWO], t))
        )

    def __repr__(self) -> str:
        try:
            parts = [
                np.asarray(x)
                for x in (
                    self.electric_scattering,
                    self.magnetic_scattering,
                )
            ]
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "Multipoles(traced)"
        se, sm = parts
        rows = "; ".join(
            f"{f:.6g} Hz: ED {e[0]:.4g}, MD {m[0]:.4g}, EQ {e[1]:.4g}, MQ {m[1]:.4g} m²"
            for f, e, m in zip(self.frequency, se, sm, strict=True)
        )
        return (
            f"Multipoles(lmax={self.lmax}, origin={_format_origin(self.origin)}; "
            f"{rows})"
        )


@functools.lru_cache(maxsize=16)
def _flat_index_arrays(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    ls, ms = _flat_lm(lmax)
    return readonly(ls), readonly(ms)


register_fields(Multipoles, ("incident_amplitude", "alpha", "beta"))


@dataclasses.dataclass(frozen=True, eq=False)
class CurrentMoments:
    r"""The long-wavelength electric dipole and the toroidal dipole of a scatterer.

    Integrals of the induced current :math:`J = +i\omega\varepsilon_0
    (\varepsilon_r - \varepsilon_b)E` over the scatterer, with :math:`r`
    measured from ``origin``. They approximate the exact dipole of
    `gfem.Multipoles` as :math:`p \approx p_0 - ikT`; they are not
    orthogonal multipoles, so no partial cross section is defined for them.

    Attributes:
        frequency: The kept frequencies of the column, Hz.
        origin: The origin of :math:`r`, m, in model coordinates.
        wavenumber: The background :math:`k`, rad/m, one per frequency.
        long_wavelength_dipole: :math:`p_0 = (i\omega)^{-1}\int J\,dV`, C·m,
            ``(n_freq, 3)``.
        toroidal_dipole: :math:`T = (10v)^{-1}\int[(r\cdot J)r - 2r^2J]\,dV`,
            C·m², ``(n_freq, 3)``, with :math:`v = \omega/k`.
    """

    frequency: tuple[float, ...]
    origin: tuple[float, float, float]
    wavenumber: tuple[float, ...]
    long_wavelength_dipole: jax.Array
    toroidal_dipole: jax.Array

    def __repr__(self) -> str:
        try:
            p0 = np.linalg.norm(np.asarray(self.long_wavelength_dipole), axis=1)
            kt = np.linalg.norm(np.asarray(self.toroidal_dipole), axis=1) * np.asarray(
                self.wavenumber
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "CurrentMoments(traced)"
        rows = "; ".join(
            f"{f:.6g} Hz: |p0| {a:.4g}, |kT| {b:.4g} C·m"
            for f, a, b in zip(self.frequency, p0, kt, strict=True)
        )
        return f"CurrentMoments(origin={_format_origin(self.origin)}; {rows})"


register_fields(CurrentMoments, ("long_wavelength_dipole", "toroidal_dipole"))
