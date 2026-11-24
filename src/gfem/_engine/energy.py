"""The electric energy a solved field stores in each region, as unit-weight integrals.

The host half of `gfem.electric_energy` (`gfem._energy`): per physical
region ``r``, ``1/4 eps0 ∫_r w E . conj(E)`` over the region's tetrahedra
in the solve domain, at quadrature order ``2 order``, exact for the discrete
field and an element-wise permittivity. The weight ``w`` is ``Re(eps_r)``
(the full real tensor) off a permittivity law region and 1 on one: the
region's Brillouin weight ``Re d(omega eps)/d omega`` is a function of the
law and the field's frequency, which JAX multiplies in, so that the host
reads no law parameter. A perfectly matched layer's regions get no entry.
The field's coefficients are on the unpaired space of the driven operator,
also on a model with `gfem.Periodic` pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem._engine import dispersion
from gfem._engine.driven import driven_operator
from gfem._engine.eigen import (
    EnergyPart,
    element_energy,
    law_scale,
    real_eps,
    scale_per_tetrahedron,
)
from gfem._pml import physical_regions

if TYPE_CHECKING:
    from gfem._engine.driven import DrivenOperator
    from gfem.model import Model

__all__ = ["EnergyParts", "build", "regions"]


def regions(model: Model, support: np.ndarray) -> tuple[str, ...]:
    """The physical volume regions with an element in ``support``, in mesh order.

    A layer region holds no energy of the device (`element_energy`).
    """
    mesh = model.mesh
    out = []
    for region in physical_regions(mesh.volumes, model.pml):
        tets = mesh.elements(region)
        if support[tets].any():
            out.append(region)
    return tuple(out)


class EnergyParts:
    """The unit-weight energy integrals of a model's regions.

    Attributes:
        op: The driven operator whose unpaired space the field lives on
            (nothing assembled).
        unit: ``1/4 eps0`` times the operator's scale: the integral of a
            part's form is ``unit`` times the energy in J.
        regions: The regions, in the output's order.
        parts: Each region's `EnergyPart`: the form ``∫ w u . v`` on its
            tetrahedra of the support.
        laws: The regions with a permittivity law (their weight is 1).
    """

    def __init__(
        self,
        model: Model,
        order: int,
        support: np.ndarray,
        names: tuple[str, ...],
    ) -> None:
        mesh = model.mesh
        self.model = model
        # only the space is read: no piece is assembled
        self.op: DrivenOperator = driven_operator(
            model,
            order,
            elements=np.zeros(len(mesh.tetrahedra), bool),
            faces=np.zeros(len(mesh.triangles), bool),
            paired=False,
        )
        self.unit = 0.25 * const.epsilon_0 * self.op.built.scale
        self.regions = names
        self.laws = frozenset(lr.region for lr in dispersion.law_regions(model))
        eps = real_eps(model)
        if self.laws:
            # real_eps holds eps_inf on a law region, whose weight is 1
            unit_law = scale_per_tetrahedron(
                model, law_scale(model, 1.0, lambda *_: 1.0)
            )
            eps = eps * unit_law.reshape((-1,) + (1,) * (eps.ndim - 1))
        else:
            unit_law = np.ones(len(mesh.tetrahedra))
        self._unit_law = unit_law
        shape = (-1,) + (1,) * (eps.ndim - 1)
        self.parts = []
        for region in names:
            inside = np.zeros(len(mesh.tetrahedra), bool)
            tets = mesh.elements(region)
            inside[tets[support[tets]]] = True
            self.parts.append(
                EnergyPart(self.op, np.where(inside.reshape(shape), eps, 0.0), inside)
            )
        self._density = element_energy(model, self.op)

    def energies(self, e: np.ndarray) -> np.ndarray:
        """The regions' unit-weight energies of the coefficients ``e``, J."""
        per_tet = np.zeros(len(self.model.mesh.tetrahedra))
        per_tet[self.op.built.cell_order] = self.unit * np.real(self._density(e))
        per_tet *= self._unit_law
        return np.array([per_tet[part.elements].sum() for part in self.parts])


def build(
    model: Model, order: int, support: np.ndarray, names: tuple[str, ...]
) -> EnergyParts:
    """The `EnergyParts` of ``names`` on ``model`` and a field's ``support``."""
    return EnergyParts(model, order, support, names)
