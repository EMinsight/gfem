r"""The power balance of an incident-field column, read on a side's surface.

A driven solve with an incident field solves :math:`Au = b` with the
load :math:`b = A_\mathrm{bg}(\chi\Pi w)` (`gfem._engine.incident`), and
the scattered field is :math:`e_s = u - \chi\Pi w`. For a set :math:`S`
of elements whose closure holds every scatterer piece and whose outside
element layer is background, with :math:`\operatorname{flux}_S` the
residual form of `gfem._engine.power_flow` and :math:`c = 2\omega\mu_0`
(times the forms' scale), the discrete equations tested with
:math:`\chi_S u` and :math:`\chi_S e_s` split exactly into

.. math::

    P_\mathrm{sca} = \operatorname{flux}_S(e_s), \qquad
    P_\mathrm{abs} = \operatorname{Im}(u_S^H b)/c - \operatorname{flux}_S(u),
    \qquad P_\mathrm{ext} = P_\mathrm{abs} + P_\mathrm{sca},

with :math:`u_S` the field on the DOFs of :math:`S`'s cells, faces and
edges. The flux reads only the background pieces of the outside layer, and
:math:`b` and :math:`\Pi w` read no scatterer material, so each power
depends on a material design only through :math:`u`; the three are the
same on every admissible :math:`S`. The flux of the total field alone is
biased by the pairing :math:`\operatorname{Im}(u_S^H b)/c`, which the
corrected `gfem.power_flow` subtracts.

:math:`b` is rebuilt here without the wave ports' terms, which couple the
DOFs of a port's surface only: a side whose closure reaches a wave port's
surface is refused, so the rows it reads are exact.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from gfem import _incident
from gfem._background import Background, lossless
from gfem._boundaries import (
    PEC,
    Conductor,
    Impedance,
    LumpedPort,
    WavePort,
    surfaces_of,
)
from gfem._checks import is_traced
from gfem._engine import power_flow as pf
from gfem._engine.driven import driven_operator
from gfem._engine.incident import (
    background_model,
    incident_coefficients,
    scatterer_elements,
    scatterer_regions,
)
from gfem._linalg.base import free_mask
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._power_flow import LAYER_REMEDY
from gfem.errors import NonDifferentiableError
from gfem.mesh import _TET_EDGES

if TYPE_CHECKING:
    from gfem._engine.power_flow import Flux, Surface
    from gfem._field import Field
    from gfem._incident import IncidentPart
    from gfem._pml import Stretch
    from gfem.model import Model

__all__ = [
    "IncidentReading",
    "check_lossless",
    "check_side",
    "check_truncation_layer",
    "reading",
]


def check_lossless(background: Background) -> None:
    r"""Raise unless the background is one lossless medium.

    Raises:
        NotImplementedError: A lossy background: the identity needs
            :math:`\operatorname{Im}((\Pi w)^H A_\mathrm{bg}\Pi w) = 0`, and the
            absorption would hold the background's own loss of the incident
            field.
    """
    for j, material in enumerate(background.layers):
        if not lossless(material):
            where = (
                f"layer {j} of the background stack"
                if _incident.layered(background)
                else "the background"
            )
            raise NotImplementedError(
                f"cross_sections in a lossy background ({where} is "
                f"{material!r}) is not measured in this gfem version: the "
                "absorption would include the background's own loss of the "
                "incident field. Give lossless layers."
            )


#: The surfaces that are part of the scatterer: none may reach the truncation
#: layer, which must be background.
_SCATTERER_SURFACES = (Conductor, Impedance, LumpedPort, PEC)


def check_truncation_layer(
    model: Model,
    background: Background,
    where: Surface,
    *,
    entry: str = "cross_sections",
    outcome: str = "finite cross section",
    remedy: str | None = None,
) -> None:
    """Raise if a scatterer piece is in the truncation layer.

    ``entry``, ``outcome`` and ``remedy`` word the messages of another
    measurement with the same refusal (`gfem.multipoles`).

    Raises:
        NonDifferentiableError: A traced material of such a region.
        ValueError: A scatterer region with an element in the layer, or a
            sheet, lumped port or perfect conductor with a face beside it.
    """
    mesh = model.mesh
    read = where.read_elements(mesh)
    remedy = remedy or (
        "Leave at least one background element between every scatterer and "
        "the PML layer or Absorbing face, or measure per region with "
        "power_flow and port_wave."
    )
    scatterer = scatterer_elements(model, background)
    for region in scatterer_regions(model, background):
        if not (read & scatterer)[mesh.elements(region)].any():
            continue
        if is_traced(model.materials[region]):
            raise NonDifferentiableError(
                f"gfem cannot differentiate {entry} with respect to the "
                f"material of region {region!r}: it has elements in the "
                "truncation layer, whose materials the measurement reads; "
                f"{LAYER_REMEDY}, or wrap it in jax.lax.stop_gradient."
            )
        raise ValueError(
            f"{entry}: the scatterer region {region!r} reaches the "
            "truncation layer (the elements touching the PML layer or an "
            "Absorbing face), which must be background: a body reaching the "
            f"truncation has no {outcome}. {remedy}"
        )
    neighbours = mesh.triangle_neighbours
    for item in (*model.boundaries, *model.ports):
        if not isinstance(item, _SCATTERER_SURFACES):
            continue
        for surface in surfaces_of(item):
            nb = neighbours[mesh.elements(surface)]
            beside = (nb >= 0) & where.layer[np.maximum(nb, 0)]
            # an exterior wall (one element beside it) is a symmetry plane
            sheet = (nb >= 0).all(axis=1)
            if (beside.any(axis=1) & sheet).any():
                raise ValueError(
                    f"{entry}: the {type(item).__name__} surface "
                    f"{surface!r} reaches the truncation layer (the elements "
                    "touching the PML layer or an Absorbing face), which must "
                    f"be background. {remedy}"
                )


def check_side(model: Model, where: Surface, who: str) -> None:
    """Raise if the side's closure reaches a wave port's surface.

    The load's rows there hold the port's terms, which need its modes.

    Raises:
        ValueError: Naming the port.
    """
    mesh = model.mesh
    keys = pf.edge_keys(where.tie, mesh.tetrahedra, np.array(_TET_EDGES))
    closure = np.unique(keys[where.inside])
    for port in model.ports:
        if not isinstance(port, WavePort):
            continue
        for name in surfaces_of(port):
            tris = mesh.triangles[mesh.elements(name)]
            edges = pf.edge_keys(where.tie, tris, pf.TRI_EDGES)
            if np.isin(edges, closure).any():
                raise ValueError(
                    f"{who}: the side reaches wave port {port.name!r}'s surface "
                    f"{name!r}, where an incident field's load holds the port's "
                    "terms. Keep the port outside the side, at least one element "
                    "away, or read the scattered field (part='scattered')."
                )


@dataclasses.dataclass(frozen=True)
class IncidentReading:
    r"""What the balance of one incident-field column reads besides the flux.

    Under ranks :math:`\chi\Pi w` is a rank's piece of a state and
    :math:`b` its part of a load (its own elements' terms), so the load
    pairing is the layout's, reduced and broadcast.

    Attributes:
        incident: :math:`\chi\Pi w` on the field's DOFs.
        pairing: :math:`b` on the DOFs of the side's closure, zero elsewhere.
        scale: :math:`c`, the flux's divisor at the field's ω.
        layout: How the DOFs lie across ranks.
    """

    incident: np.ndarray
    pairing: np.ndarray
    scale: float
    layout: DofLayout = ONE_PROCESS

    def load(self, e: np.ndarray) -> float:
        r""":math:`\operatorname{Im}(e_S^H b)/c`."""
        paired = self.layout.pair(self.pairing, np.conj(e))
        return float(np.imag(paired)) / self.scale

    def load_pullback(self) -> np.ndarray:
        """JAX's gradient of `load` in ``e`` (``dP/dx - i dP/dy``)."""
        return 1j * np.conj(self.pairing) / self.scale


def reading(
    field: Field,
    part: IncidentPart,
    model: Model,
    where: Surface,
    flux: Flux,
    *,
    stretch: Stretch | None,
) -> IncidentReading:
    """The `IncidentReading` of ``field`` on ``where`` (see the module docstring)."""
    omega = float(np.real(np.asarray(field.omega)))
    comm = field._communicator  # noqa: SLF001
    bg = driven_operator(
        background_model(model, part.background),
        field.order,
        paired=False,
        comm=comm,
        stretch=stretch,
    )
    w = pf.local_coefficients(np.asarray(incident_coefficients(field, part)), bg, comm)
    b = np.zeros_like(w, dtype=complex)
    for k, piece in enumerate(bg.pieces):
        if piece.kind != "port":
            b += complex(piece.theta(omega)) * bg.apply(k, w)
    b[~free_mask(bg.space.FreeDofs(), bg.space.ndof)] = 0
    mask = pf.dofs(bg, where.tie, where.inside, where.inside)
    return IncidentReading(
        incident=w,
        pairing=np.where(mask, b, 0.0),
        scale=flux._scale(omega),  # noqa: SLF001
        layout=flux.layout,
    )
