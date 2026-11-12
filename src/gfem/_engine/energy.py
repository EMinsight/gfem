"""The electric energy a solved field stores in each volume region.

The host kernel of `gfem.electric_energy`: ``1/4 eps0 ∫ Re(eps_r) E .
conj(E)`` over each region's tetrahedra in the solve domain, at quadrature
order ``2 order``, exact for the discrete field and an element-wise
permittivity. The field's coefficients are on the unpaired space of the
driven operator, also on a model with `gfem.Periodic` pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem._engine.driven import driven_operator
from gfem._engine.eigen import element_energy
from gfem._engine.power_flow import _check_field

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["electric_energy"]


def electric_energy(field: Field, model: Model) -> dict[str, float]:
    """Time-averaged electric energy per region, J (see `gfem.electric_energy`).

    Raises:
        ValueError: The field was not solved on ``model``.
    """
    mesh = model.mesh
    support = _check_field(field, model)
    # only the space is read: no piece is assembled
    op = driven_operator(
        model,
        field.order,
        elements=np.zeros(len(mesh.tetrahedra), bool),
        faces=np.zeros(len(mesh.triangles), bool),
        paired=False,
    )
    e = np.asarray(field.coefficients)
    if e.shape[0] != op.space.ndof:
        raise ValueError(
            f"The field has {e.shape[0]} coefficients but the model's order-"
            f"{field.order} driven space has {op.space.ndof} DOFs: the field "
            "was not solved on this model."
        )
    density = element_energy(model, op)
    per_tet = np.zeros(len(mesh.tetrahedra))
    per_tet[op.built.tetrahedron_order] = (
        0.25 * const.epsilon_0 * op.built.scale * np.real(density(e))
    )
    out = {}
    for region in mesh.volumes:
        tets = mesh.elements(region)
        tets = tets[support[tets]]
        if len(tets):
            out[region] = float(per_tet[tets].sum())
    return out
