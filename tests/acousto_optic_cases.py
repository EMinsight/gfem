"""Lithium niobate's photoelastic tensor, shared by the acousto-optic tests."""

import numpy as np

#: Photoelastic constants p^E of lithium niobate (dimensionless), the set the
#: push-pull modulator reference uses; p14 is negative with a positive e22 and r22.
P11, P12, P13, P14 = -0.026, 0.090, 0.133, -0.075
P31, P33, P41, P44 = 0.179, 0.071, -0.151, 0.146


def tensor_3m_p(
    *, p11=P11, p12=P12, p13=P13, p14=P14, p31=P31, p33=P33, p41=P41, p44=P44
) -> np.ndarray:
    """The contracted 6x6 tensor of class 3m, crystal frame, Voigt xx yy zz yz xz xy.

    Not symmetric: p14 and p41 differ.
    """
    return np.array(
        [
            [p11, p12, p13, p14, 0, 0],
            [p12, p11, p13, -p14, 0, 0],
            [p31, p31, p33, 0, 0, 0],
            [p41, -p41, 0, p44, 0, 0],
            [0, 0, 0, 0, p44, p41],
            [0, 0, 0, 0, p14, (p11 - p12) / 2],
        ]
    )


P_LN = tensor_3m_p()
