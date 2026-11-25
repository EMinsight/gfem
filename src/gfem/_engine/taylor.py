r"""Truncated Taylor series to third order in one real variable.

A `Taylor` holds :math:`c_n = F^{(n)}(x_0)/n!` for :math:`n \le 3`, so that
:math:`F(x_0 + h) = \sum_n c_n h^n + O(h^4)`. Sums, products, quotients and
the few functions the port mode pencil's scalars need compose these series
exactly to that order, so each scalar is written once as an expression and
its chain rules are never written by hand.

`Taylor.real` takes the real part of each coefficient: for a real step
:math:`h` it is the series of :math:`\operatorname{Re}F`, which is not
analytic, so no complex step or contour can give it.
"""

from __future__ import annotations

import cmath

import numpy as np

__all__ = ["ORDER", "Taylor"]

ORDER = 3


class Taylor:
    r""":math:`\sum_{n \le 3} c_n h^n`, complex coefficients ``c``."""

    __slots__ = ("c",)

    def __init__(self, c) -> None:
        """The series of coefficients ``c`` (length 4)."""
        self.c = np.asarray(c, dtype=complex)

    @classmethod
    def variable(cls, x0: complex) -> Taylor:
        """The variable itself at ``x0``: :math:`x_0 + h`."""
        return cls([x0, 1.0, 0.0, 0.0])

    @classmethod
    def constant(cls, a: complex) -> Taylor:
        """A constant."""
        return cls([a, 0.0, 0.0, 0.0])

    @staticmethod
    def _lift(other: Taylor | complex) -> Taylor:
        return other if isinstance(other, Taylor) else Taylor.constant(other)

    def __add__(self, other: Taylor | complex) -> Taylor:
        return Taylor(self.c + self._lift(other).c)

    __radd__ = __add__

    def __sub__(self, other: Taylor | complex) -> Taylor:
        return Taylor(self.c - self._lift(other).c)

    def __rsub__(self, other: Taylor | complex) -> Taylor:
        return Taylor(self._lift(other).c - self.c)

    def __neg__(self) -> Taylor:
        return Taylor(-self.c)

    def __mul__(self, other: Taylor | complex) -> Taylor:
        if not isinstance(other, Taylor):
            return Taylor(self.c * other)
        a, b = self.c, other.c
        return Taylor(
            [sum(a[i] * b[n - i] for i in range(n + 1)) for n in range(ORDER + 1)]
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Taylor | complex) -> Taylor:
        if not isinstance(other, Taylor):
            return Taylor(self.c / other)
        return self * other.reciprocal()

    def __rtruediv__(self, other: complex) -> Taylor:
        return self.reciprocal() * other

    def apply(self, f0: complex, f1: complex, f2: complex, f3: complex) -> Taylor:
        r""":math:`F(\cdot)` of this series from :math:`F^{(n)}` at :math:`c_0`.

        With :math:`d = \sum_{n \ge 1} c_n h^n`,
        :math:`F = f_0 + f_1 d + f_2 d^2/2 + f_3 d^3/6`.
        """
        d = Taylor([0.0, *self.c[1:]])
        d2 = d * d
        return Taylor.constant(f0) + d * f1 + d2 * (f2 / 2) + d2 * d * (f3 / 6)

    def reciprocal(self) -> Taylor:
        """:math:`1/F`."""
        a = complex(self.c[0])
        return self.apply(1 / a, -1 / a**2, 2 / a**3, -6 / a**4)

    def sqrt(self) -> Taylor:
        r""":math:`\sqrt F`, principal root."""
        a = cmath.sqrt(complex(self.c[0]))
        return self.apply(a, 0.5 / a, -0.25 / a**3, 0.375 / a**5)

    def tanh(self) -> Taylor:
        r""":math:`\tanh F`."""
        t = cmath.tanh(complex(self.c[0]))
        s = 1 - t * t
        return self.apply(t, s, -2 * t * s, s * (6 * t * t - 2))

    @property
    def real(self) -> Taylor:
        r"""The series of :math:`\operatorname{Re}F` for a real step.

        The real part of each coefficient.
        """
        return Taylor(self.c.real)
