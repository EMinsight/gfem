"""Diffraction orders of a periodic cell: the order helpers of a Floquet surface.

The module is the home of the public helpers that list, declare and read the
diffraction orders a periodic cell radiates into the half-space behind a
Floquet surface.
"""

from __future__ import annotations

__all__: list[str] = []

from gfem._diffraction.aperture import ApertureField, propagate

__all__ += ["ApertureField", "propagate"]

from gfem._diffraction.outputs import (  # noqa: E402
    Efficiencies,
    PowerBalance,
    efficiencies,
    floquet_ports,
    power_balance,
)

__all__ += [
    "Efficiencies",
    "PowerBalance",
    "efficiencies",
    "floquet_ports",
    "power_balance",
]
