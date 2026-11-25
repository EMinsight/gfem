"""Periodic surfaces: orders, efficiencies, sweeps, emissivity and metasurfaces.

A periodic cell with Floquet ports on its faces radiates into diffraction
orders. This module reads those orders, in five areas:

- **Orders and efficiencies.** `floquet_ports` declares and lists the
  orders of a surface; `efficiencies` gives the power in each order,
  `power_balance` the reflected, transmitted, coupled and absorbed shares, and
  `jones` the polarization-resolved Jones matrix of one order with its
  circular and Mueller views. `effective_medium` retrieves the effective
  permittivity and permeability of a slab between two faces.
- **Angle sweeps.** `angle_sweep` solves one cell over a grid of incidence
  angles and returns an `AngleSweepResult` that the readers above accept.
- **Emissivity.** `emissivity` applies Kirchhoff's law to a sweep, and
  `hemispherical_emissivity` integrates it with a `hemisphere` quadrature rule
  (`HemisphereRule`), optionally folded by the cell's symmetry.
- **Cell response.** `cell_response` samples a cell's S-matrix over a design
  parameter (a width, a radius) once, as a `CellResponse` whose Hermite
  interpolant is exact in value and in derivative at the samples.
- **Metasurfaces.** `aperture_field` places one interpolated cell per
  position (the locally periodic approximation) as an `ApertureField`,
  and `propagate` carries it to another plane by the angular spectrum.

Everything that reads results is differentiable with ``jax.grad``, ``jax.jit``
and ``jax.vmap`` in the solved design. The recipes on diffraction efficiency,
emissivity and metasurfaces from cells walk through each area, and the metalens
tutorial designs a lens from a cell library.
"""

from __future__ import annotations

from gfem._diffraction.aperture import ApertureField, aperture_field, propagate
from gfem._diffraction.effective import EffectiveMedium, effective_medium
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
    "EffectiveMedium",
    "Efficiencies",
    "Emissivity",
    "HemisphereRule",
    "Jones",
    "PowerBalance",
    "angle_sweep",
    "aperture_field",
    "cell_response",
    "effective_medium",
    "efficiencies",
    "emissivity",
    "floquet_ports",
    "hemisphere",
    "hemispherical_emissivity",
    "jones",
    "power_balance",
    "propagate",
]
