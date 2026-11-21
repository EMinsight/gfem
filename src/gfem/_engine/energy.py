"""The electric energy a solved field stores in each volume region.

The host kernel of `gfem.electric_energy`: ``1/4 eps0 ∫ Re(eps_r) E .
conj(E)`` over each region's tetrahedra in the solve domain, at quadrature
order ``2 order``, exact for the discrete field and an element-wise
permittivity. A permittivity law region reads the Brillouin weight ``Re
d(omega eps)/d omega`` at the field's real frequency instead. A perfectly
matched layer's regions get no entry.
The field's coefficients are on the unpaired space of the driven operator,
also on a model with `gfem.Periodic` pairs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem._engine import dispersion
from gfem._engine.driven import driven_operator
from gfem._engine.eigen import element_energy, law_scale, scale_per_tetrahedron
from gfem._engine.power_flow import check_field, check_size
from gfem._pml import physical_regions

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["electric_energy"]


def electric_energy(field: Field, model: Model) -> dict[str, float]:
    """Time-averaged electric energy per region, J (see `gfem.electric_energy`).

    Raises:
        ValueError: The field was not solved on ``model``, or a law
            region's Brillouin weight is not positive at its frequency.
    """
    mesh = model.mesh
    support = check_field(field, model)
    # only the space is read: no piece is assembled
    op = driven_operator(
        model,
        field.order,
        elements=np.zeros(len(mesh.tetrahedra), bool),
        faces=np.zeros(len(mesh.triangles), bool),
        paired=False,
    )
    e = np.asarray(field.coefficients)
    check_size(e, op, field.order)
    density = element_energy(model, op)
    per_tet = np.zeros(len(mesh.tetrahedra))
    per_tet[op.built.cell_order] = (
        0.25 * const.epsilon_0 * op.built.scale * np.real(density(e))
    )
    if dispersion.law_regions(model):
        # a law region's eps_inf to its Brillouin weight at the field's
        # frequency
        omega = float(np.real(np.asarray(field.omega)))
        dispersion.check_energy(
            model, omega / (2 * math.pi), "field at", brillouin=True
        )
        per_tet *= scale_per_tetrahedron(
            model, law_scale(model, omega, dispersion.brillouin_eps)
        )
    out = {}
    # a layer region holds no energy of the device (`element_energy`)
    for region in physical_regions(mesh.volumes, model.pml):
        tets = mesh.elements(region)
        tets = tets[support[tets]]
        if len(tets):
            out[region] = float(per_tet[tets].sum())
    return out
