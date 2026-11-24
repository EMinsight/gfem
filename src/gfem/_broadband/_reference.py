r"""The one real reference impedance per port that the fit and time views need.

A reference impedance counts as real and constant over frequency when its
imaginary part and its spread across frequency are both within a relative
:math:`10^{-9}` of its magnitude: a TEM wave port's :math:`Z_{PV}`, real and
frequency independent in theory, differs between frequencies by about
:math:`10^{-13}` relative from the mode solve's round-off, and exact equality
would refuse it. The port's reference is then its mean real value.
"""

from __future__ import annotations

import numpy as np

#: Relative imaginary part and spread over frequency still read as constant.
RTOL = 1e-9


def constant_real(z_ref: np.ndarray) -> np.ndarray | None:
    r"""Each port's mean real reference ``(n_ports,)``, or None when not constant.

    Args:
        z_ref: Reference impedances ``(n_freq, n_ports)``, no NaN.

    Returns:
        :math:`\overline{\operatorname{Re} Z_p}` over frequency for each port
        :math:`p`, if :math:`\max_f |\operatorname{Im} Z_p|` and
        :math:`\max_f \operatorname{Re} Z_p - \min_f \operatorname{Re} Z_p`
        are both at most :math:`10^{-9} \max_f |Z_p|`; None otherwise.
    """
    z = np.asarray(z_ref)
    real = np.real(z)
    scale = RTOL * np.abs(z).max(axis=0)
    spread = real.max(axis=0) - real.min(axis=0)
    if (np.abs(np.imag(z)).max(axis=0) > scale).any() or (spread > scale).any():
        return None
    return real.mean(axis=0).astype(float)
