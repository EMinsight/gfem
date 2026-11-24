r"""Pole sets of a rational model of S, and the model's S on them.

Shared by the fit and the passivity test. A pole set holds the real poles
and one pole of each conjugate pair (positive imaginary part); on it, an
element of S with real coefficients :math:`c_k` is

.. math::

    S_{ij}(s) = d_{ij} + \sum_k c_{ij,k}\, \phi_k(s),

with :math:`\phi = 1/(s - p)` for a real pole and, for a pair :math:`a`,
:math:`1/(s - a) + 1/(s - a^*)` and :math:`i/(s - a) - i/(s - a^*)`, the
real and imaginary parts of the pair's residue. The same model in physical
form, :math:`D + \sum_n R_n / (s - p_n)` over every pole, is `evaluate`.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Poles:
    """Poles: real ones, and one pole per conjugate pair (``Im > 0``)."""

    real: np.ndarray
    pairs: np.ndarray

    @property
    def size(self) -> int:
        """The number of poles, counting both of each pair."""
        return self.real.size + 2 * self.pairs.size


def basis(x: np.ndarray, poles: Poles) -> np.ndarray:
    """Real-coefficient basis ``(K, size)``: real poles, then (Re, Im) per pair."""
    a = 1 / (x[:, None] - poles.pairs[None, :])
    b = 1 / (x[:, None] - np.conj(poles.pairs)[None, :])
    pair = np.empty((x.size, 2 * poles.pairs.size), complex)
    pair[:, 0::2] = a + b
    pair[:, 1::2] = 1j * (a - b)
    return np.hstack([1 / (x[:, None] - poles.real[None, :]), pair])


def stack(m: np.ndarray) -> np.ndarray:
    """Real and imaginary rows of ``m`` stacked along the second-to-last axis."""
    return np.concatenate([m.real, m.imag], axis=-2)


def state_matrix(poles: Poles) -> tuple[np.ndarray, np.ndarray]:
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


def coefficients(
    poles: np.ndarray, residues: np.ndarray, scale: float = 1.0
) -> tuple[Poles, np.ndarray]:
    """The pole set and real coefficients ``(P * P, size)`` of a physical model.

    Args:
        poles: ``(n_poles,)`` complex: real poles, and conjugate pairs whose
            residues are conjugate.
        residues: ``(n_poles, P, P)`` complex, real for a real pole.
        scale: Frequency scale: poles and residues are divided by it.
    """
    real = np.flatnonzero(poles.imag == 0)
    pairs = np.flatnonzero(poles.imag > 0)
    n_ports = residues.shape[-1]
    flat = residues.reshape(poles.size, n_ports * n_ports) / scale
    # (n_el, 2 * n_pairs): the real and imaginary parts of each pair's
    # residue, side by side
    pair = np.stack([flat[pairs].real, flat[pairs].imag], axis=-1)
    c = np.hstack(
        [flat[real].real.T, pair.transpose(1, 0, 2).reshape(flat.shape[1], -1)]
    )
    return Poles(poles[real].real / scale, poles[pairs] / scale), c


def response(x: np.ndarray, poles: Poles, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """S ``(K, P, P)`` at Laplace values ``x`` of the model on a pole set."""
    n_ports = d.shape[0]
    return d[None] + (basis(x, poles) @ c.T).reshape(-1, n_ports, n_ports)


def evaluate(
    s: np.ndarray, poles: np.ndarray, residues: np.ndarray, d: np.ndarray
) -> np.ndarray:
    r""":math:`D + \sum_n R_n / (s - p_n)`, ``(K, P, P)``, at Laplace values ``s``.

    Args:
        s: ``(K,)`` complex, e.g. :math:`2\pi i f`.
        poles: ``(n_poles,)`` complex, every pole; pairs as for `coefficients`.
        residues: ``(n_poles, P, P)`` complex.
        d: ``(P, P)`` real.
    """
    pole_set, c = coefficients(poles, residues)
    return response(s, pole_set, c, d)
