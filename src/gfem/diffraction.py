"""Diffraction orders of a periodic cell: the order helpers of a Floquet surface.

The module is the home of the public helpers that list, declare and read the
diffraction orders a periodic cell radiates into the half-space behind a
Floquet surface.
"""

from __future__ import annotations

from gfem._diffraction.aperture import ApertureField, propagate
from gfem._diffraction.emissivity import (
    Emissivity,
    HemisphereRule,
    emissivity,
    hemisphere,
    hemispherical_emissivity,
)
from gfem._diffraction.outputs import (
    Efficiencies,
    Jones,
    PowerBalance,
    efficiencies,
    floquet_ports,
    jones,
    power_balance,
)
from gfem._diffraction.response import CellResponse, cell_response
from gfem._diffraction.sweep import AngleSweepResult, angle_sweep

__all__ = [
    "AngleSweepResult",
    "ApertureField",
    "CellResponse",
    "Efficiencies",
    "Emissivity",
    "HemisphereRule",
    "Jones",
    "PowerBalance",
    "angle_sweep",
    "cell_response",
    "efficiencies",
    "emissivity",
    "floquet_ports",
    "hemisphere",
    "hemispherical_emissivity",
    "jones",
    "power_balance",
    "propagate",
]
