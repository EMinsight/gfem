"""Diffraction orders of a periodic cell: the order helpers of a Floquet surface.

The module is the home of the public helpers that list, declare and read the
diffraction orders a periodic cell radiates into the half-space behind a
Floquet surface.
"""

from __future__ import annotations

from gfem._diffraction.aperture import ApertureField, propagate
from gfem._diffraction.outputs import (
    Efficiencies,
    Jones,
    PowerBalance,
    efficiencies,
    floquet_ports,
    jones,
    power_balance,
)
from gfem._diffraction.sweep import AngleSweepResult, angle_sweep

__all__ = [
    "AngleSweepResult",
    "ApertureField",
    "Efficiencies",
    "Jones",
    "PowerBalance",
    "angle_sweep",
    "efficiencies",
    "floquet_ports",
    "jones",
    "power_balance",
    "propagate",
]
