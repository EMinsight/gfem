r"""A plane wave's field in a background stack, by a stable transfer matrix.

Under :math:`e^{+i\omega t}`, layer :math:`j` of a `gfem.Background` (from
the bottom, :math:`j = 0`) holds a constant :math:`\varepsilon_j`,
:math:`\mu_j` at each frequency,

.. math::

    \varepsilon_j = \varepsilon_r(1 - i\tan\delta) - \frac{i\sigma}{\omega
    \varepsilon_0},

and one up-going and one down-going plane wave with the incident wave's
lateral wave vector :math:`k_t`, which is real and set by the incidence
layer (the bottom for :math:`\theta < \pi/2`, the top for
:math:`\theta > \pi/2`). With :math:`k_j^2 = \omega^2\varepsilon_j\mu_j /
c^2` and :math:`k_{z,j} = \sqrt{k_j^2 - |k_t|^2}` on the branch
:math:`\operatorname{Im} k_{z,j} \le 0`, the field in layer :math:`j` is

.. math::

    E = \bigl[U_j\,\hat e^+_j\,e^{-i k_{z,j}(z - z^+_j)}
        + D_j\,\hat e^-_j\,e^{+i k_{z,j}(z - z^-_j)}\bigr]
        e^{-i k_t\cdot r},

each wave referenced at the plane it leaves (:math:`z^+_j` the bottom of
the layer, :math:`z^-_j` its top). No exponential then grows across a
layer, so evanescent, thick and lossy layers stay finite. TE and TM
decouple; per polarization the amplitudes solve one square system: the
continuity of tangential :math:`E` and :math:`H` at every interface, the
incident amplitude, no wave arriving from infinity on the far side, and,
over a ground plane, :math:`E_t = 0` there. The polarization vectors are
:math:`\hat\phi` (TE) and :math:`\hat\theta_j^\pm = (k_{z,j}^\pm/k_j)
\hat\kappa - (|k_t|/k_j)\hat z` (TM), so the incident wave is exactly the
`gfem.PlaneWave`'s, :math:`E_0` and its phase given in its own layer at the
model origin; :math:`H = k\times E/(\omega\mu_0\mu_j)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.constants as const

from gfem._background import Background
from gfem._sources import PlaneWave, jones

__all__ = [
    "StackWave",
    "incidence_layer",
    "layer_constants",
    "layer_index",
    "number",
]


def number(value: Any) -> float:
    """A background layer's property as a float (layers are constant, isotropic)."""
    return float(value)


def layer_constants(
    background: Background, frequency: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Each layer's complex :math:`\varepsilon_r` and :math:`\mu_r` at a frequency."""
    omega = 2 * math.pi * frequency
    eps = np.array(
        [
            number(m.eps_r) * (1 - 1j * number(m.tan_delta))
            - 1j * number(m.conductivity) / (omega * const.epsilon_0)
            for m in background.layers
        ],
        complex,
    )
    mu = np.array([number(m.mu_r) for m in background.layers], complex)
    return eps, mu


def layer_index(background: Background, z: np.ndarray) -> np.ndarray:
    """The stack layer holding each height ``z`` (m); a plane counts below."""
    return np.searchsorted(np.asarray(background.interfaces, float), z, side="left")


def incidence_layer(background: Background, theta: float) -> int:
    """The index of the layer a plane wave of polar angle ``theta`` comes from."""
    return 0 if math.cos(theta) > 0 else len(background.layers) - 1


def _branch(q: np.ndarray) -> np.ndarray:
    kz = np.sqrt(q.astype(complex))
    return np.where(kz.imag > 0, -kz, kz)


class StackWave:
    """The field of a `gfem.PlaneWave` in a background stack at one frequency.

    The source is assumed solvable there (`gfem._incident.check_solvable`):
    a lossless incidence layer, no grazing incidence, and incidence from
    above over a ground plane.
    """

    def __init__(self, background: Background, wave: PlaneWave, frequency: float):
        """Solve the amplitudes (see the module docstring)."""
        eps, mu = layer_constants(background, frequency)
        n = len(eps)
        planes = np.asarray(background.interfaces, float)
        self.background = background
        self.mu = mu
        self.omega = 2 * math.pi * frequency
        k0 = self.omega / const.c
        self.k = k0 * np.sqrt(eps * mu)
        up = math.cos(wave.theta) > 0
        j_in = 0 if up else n - 1
        k_in = float(self.k[j_in].real)
        self.kt = k_in * math.sin(wave.theta)
        self.kappa = np.array([math.cos(wave.phi), math.sin(wave.phi), 0.0])
        self.phi_hat = np.array([-math.sin(wave.phi), math.cos(wave.phi), 0.0])
        self.kz = _branch(self.k**2 - self.kt**2)
        # the plane each wave leaves: the up wave the layer's bottom, the
        # down wave its top; the infinite layers use their one plane
        ground = background.ground
        bottom = planes[0] if n > 1 else 0.0
        top = planes[-1] if n > 1 else 0.0
        self.z_up = np.r_[bottom if ground is None else ground, planes]
        self.z_down = np.r_[planes, top]
        if n == 1 and ground is not None:
            self.z_down = np.array([ground])
        a_tm, a_te = jones(wave.polarization) * complex(wave.amplitude)
        # incident amplitude at its plane, its phase given at the origin
        kz_in = self.kz[j_in]
        if up:
            incident = np.exp(-1j * kz_in * self.z_up[0])
        else:
            incident = np.exp(1j * kz_in * self.z_down[-1])
        self.amplitudes = {
            pol: self._solve(pol, a * incident, up, planes)
            for pol, a in (("TM", a_tm), ("TE", a_te))
        }

    def _tangential(self, pol: str, j: int, sign: int) -> tuple[complex, complex]:
        r"""Unit wave's tangential :math:`E` and :math:`\omega\mu_0 H`, scalars.

        TE: :math:`E` along :math:`\hat\phi`, :math:`H` along
        :math:`\hat\kappa`; TM: :math:`E` along :math:`\hat\kappa`, :math:`H`
        along :math:`\hat\phi`. ``sign`` is +1 for the up wave.
        """
        kz, k, mu = sign * self.kz[j], self.k[j], self.mu[j]
        if pol == "TM":
            return kz / k, k / mu
        return 1.0, -kz / mu

    def _solve(
        self, pol: str, incident: complex, up: bool, planes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.k)
        # unknowns: U_0..U_{n-1}, D_0..D_{n-1}
        size = 2 * n
        m = np.zeros((size, size), complex)
        rhs = np.zeros(size, complex)
        row = 0
        for i, z in enumerate(planes):  # layer i below, i + 1 above
            for j, side in ((i, 1.0), (i + 1, -1.0)):
                e_up = np.exp(-1j * self.kz[j] * (z - self.z_up[j]))
                e_down = np.exp(1j * self.kz[j] * (z - self.z_down[j]))
                for comp in (0, 1):
                    m[row + comp, j] += side * self._tangential(pol, j, 1)[comp] * e_up
                    m[row + comp, n + j] += (
                        side * self._tangential(pol, j, -1)[comp] * e_down
                    )
            row += 2
        # the incident wave, and nothing from infinity on the far side
        if up:
            m[row, 0], rhs[row] = 1.0, incident
            m[row + 1, 2 * n - 1] = 1.0
        else:
            m[row, 2 * n - 1], rhs[row] = 1.0, incident
            ground = self.background.ground
            if ground is None:
                m[row + 1, 0] = 1.0
            else:  # E_t = 0 on the ground plane
                e_down = np.exp(1j * self.kz[0] * (ground - self.z_down[0]))
                m[row + 1, 0] = self._tangential(pol, 0, 1)[0]
                m[row + 1, n] = self._tangential(pol, 0, -1)[0] * e_down
        x = np.linalg.solve(m, rhs)
        return x[:n], x[n:]

    def _waves(self, j: int, sign: int) -> tuple[np.ndarray, np.ndarray]:
        """The wave vector and the TM unit vector of a wave in layer ``j``."""
        kz = sign * self.kz[j]
        k_vec = self.kt * self.kappa + kz * np.array([0.0, 0.0, 1.0])
        theta_hat = (kz * self.kappa - self.kt * np.array([0.0, 0.0, 1.0])) / self.k[j]
        return k_vec, theta_hat

    def _fields(self, points: np.ndarray, magnetic: bool) -> np.ndarray:
        points = np.asarray(points, float).reshape(-1, 3)
        z = points[:, 2]
        layer = layer_index(self.background, z)
        lateral = np.exp(-1j * self.kt * (points @ self.kappa))
        out = np.zeros((len(points), 3), complex)
        for j in np.unique(layer):
            on = layer == j
            for sign, ref, which in ((1, self.z_up[j], 0), (-1, self.z_down[j], 1)):
                k_vec, theta_hat = self._waves(j, sign)
                e = (
                    self.amplitudes["TM"][which][j] * theta_hat
                    + self.amplitudes["TE"][which][j] * self.phi_hat
                )
                if magnetic:
                    e = np.cross(k_vec, e) / (self.omega * const.mu_0 * self.mu[j])
                phase = np.exp(-1j * sign * self.kz[j] * (z[on] - ref)) * lateral[on]
                out[on] += phase[:, None] * e[None, :]
        return out

    def electric(self, points: np.ndarray) -> np.ndarray:
        """The electric field at ``points`` (``(n, 3)``, m), V/m."""
        return self._fields(points, magnetic=False)

    def magnetic(self, points: np.ndarray) -> np.ndarray:
        """The magnetic field at ``points`` (``(n, 3)``, m), A/m."""
        return self._fields(points, magnetic=True)
