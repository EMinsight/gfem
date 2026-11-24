"""Plane-wave polarization algebra of the cell outputs, written out for the tests.

Everything here is NumPy on a single Jones matrix, built from the definitions
(not from `gfem.diffraction`): the circular basis, the Stokes vector of a beam
and the Mueller matrix measured by sending four probe beams through a Jones
matrix. The components are ``(b_TM, b_TE)`` in a beam's Floquet references and
``h = +1`` or ``-1`` is its direction along the cell's third axis.
"""

import math

import numpy as np

PROBES = ("H", "V", "D", "RCP")


def circular_unitary(h: int) -> np.ndarray:
    """Columns RCP and LCP (IEEE, looking into the beam), rows TM and TE."""
    return np.array([[1.0, 1.0], [-1j * h, 1j * h]]) / math.sqrt(2)


def rotation(psi: float) -> np.ndarray:
    """The rotation of the (TM, TE) plane by ``psi``."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


def stokes(b: np.ndarray, h: int) -> np.ndarray:
    """The Stokes vector of a beam on its ``(theta, phi) = (h b_TM, b_TE)`` pair."""
    b_theta, b_phi = h * b[0], b[1]
    cross = b_theta * np.conj(b_phi)
    return np.array(
        [
            abs(b_theta) ** 2 + abs(b_phi) ** 2,
            abs(b_theta) ** 2 - abs(b_phi) ** 2,
            2 * cross.real,
            2 * cross.imag,
        ]
    )


def probe(name: str, h: int) -> np.ndarray:
    """The ``(b_TM, b_TE)`` of a probe beam of direction ``h``.

    Horizontal and vertical (the theta and phi directions), diagonal, and
    right-circular.
    """
    r = math.sqrt(0.5)
    return {
        "H": np.array([h, 0.0]),
        "V": np.array([0.0, 1.0]),
        "D": np.array([h * r, r]),
        "RCP": circular_unitary(h)[:, 0],
    }[name]


def mueller_measured(j: np.ndarray, h_out: int, h_in: int) -> np.ndarray:
    """The Mueller matrix of ``j`` from the Stokes vectors of the four probes."""
    s_in = np.stack([stokes(probe(n, h_in), h_in) for n in PROBES], axis=1)
    s_out = np.stack([stokes(j @ probe(n, h_in), h_out) for n in PROBES], axis=1)
    return s_out @ np.linalg.inv(s_in)
