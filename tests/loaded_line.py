r"""The loaded line in closed form: a section between two feeds, as three lines.

The topology tutorial's line (``docs/tutorials/data/line.msh`` and
``line2d.msh``, mm): 40 mm long and :math:`w = 6` mm wide, a section of
permittivity ``eps`` (complex for a lossy one) over :math:`10 < x < 30`
between air feeds. The section is uniform across the width, so each port's
mode is the same function of :math:`y` in every part of the line and no other
mode is excited: S is that of a three-part transmission line.
"""

import math

import numpy as np
from scipy.constants import c

LENGTH, WIDTH, SECTION, EPS = 40.0, 6.0, (10.0, 30.0), 10.0


def loaded_line(frequency, polarization, *, sine=False, eps=EPS):
    r"""``(S11, S21)`` of the loaded line, closed form.

    The mode is uniform (TEM) or :math:`\sin(\pi y / w)`, so each part carries
    :math:`\beta^2 = k_0^2\varepsilon_r - q^2` with :math:`q = 0` or
    :math:`\pi / w`. At each interface :math:`u` and :math:`w\,\partial_x u`
    are continuous, :math:`w = 1` for Ez and :math:`1/\varepsilon_r` for Hz.
    S is read from the electric field across the line (along :math:`y`). In
    Hz that field is :math:`\partial_x H_z / (i\omega\varepsilon)`: a
    reflected wave's has the opposite sign of its :math:`H_z`, a transmitted
    wave's the same, so S11 is minus the :math:`H_z` reflection and S21 the
    :math:`H_z` transmission. A lossy section's :math:`\varepsilon_r` is
    complex, :math:`\varepsilon'(1 - i\tan\delta)`, and its :math:`\beta`
    the root with a negative imaginary part (a forward wave decays).
    """
    k0 = 2 * math.pi * frequency / c * 1e-3  # per mm
    q2 = (math.pi / WIDTH) ** 2 if sine else 0.0
    parts = (1.0, eps, 1.0)
    beta = [np.sqrt(k0**2 * e - q2 + 0j) for e in parts]
    weight = [1 / e if polarization == "Hz" else 1.0 for e in parts]
    starts = (0.0, *SECTION)

    def value(j, x):  # u of (A_j, B_j) at x, A_j forward, B_j backward
        d = x - starts[j]
        return np.array([np.exp(-1j * beta[j] * d), np.exp(1j * beta[j] * d)])

    def flux(j, x):
        return weight[j] * 1j * beta[j] * value(j, x) * np.array([-1, 1])

    # unknowns B_0, A_1, B_1, A_2; A_0 = 1 and B_2 = 0
    m = np.zeros((4, 4), complex)
    r = np.zeros(4, complex)
    for row, part in ((0, value), (1, flux)):
        left, right = part(0, SECTION[0]), part(1, SECTION[0])
        m[row, :3] = left[1], -right[0], -right[1]
        r[row] = -left[0]
    for row, part in ((2, value), (3, flux)):
        left, right = part(1, SECTION[1]), part(2, SECTION[1])
        m[row, 1:] = left[0], left[1], -right[0]
    b0, _, _, a2 = np.linalg.solve(m, r)
    s11, s21 = b0, a2 * np.exp(-1j * beta[2] * (LENGTH - SECTION[1]))
    return (-s11 if polarization == "Hz" else s11), s21


def absorbed(frequency, polarization, eps):
    """``1 - |S11|^2 - |S21|^2`` of the loaded line: the section's loss per 1 W."""
    s11, s21 = loaded_line(frequency, polarization, eps=eps)
    return 1 - abs(s11) ** 2 - abs(s21) ** 2
