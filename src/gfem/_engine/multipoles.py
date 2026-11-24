r"""The induced current of a solved field, at quadrature points of the scatterer.

The host side of `gfem.current_moments`. With the time dependence
:math:`e^{+i\omega t}` the polarization current of a scatterer in a lossless
background of relative permittivity :math:`\varepsilon_b` is

.. math::

    J = i\omega\varepsilon_0(\varepsilon_\mathrm{eff} - \varepsilon_b)\,E,
    \qquad
    \varepsilon_\mathrm{eff} = \varepsilon_r(1 - i\tan\delta)
    - \frac{i\sigma}{\varepsilon_0\omega},

the total field :math:`E` read in every scatterer element (the elements of
the regions that are not the background, `gfem._engine.incident`) outside the
perfectly matched layer. The permittivity is one value (or one tensor) per
element at the frequency of the field, a permittivity law included
(`gfem._engine.dispersion.eps_at`); the conduction current is the
:math:`\sigma E` of the imaginary part.

The field is read by the probe, which gathers the values of any element on
every rank, so a distributed field gives every rank the same points and the
same current: the sums over the elements need no reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem import _incident, _probe
from gfem._engine import far_field as ff
from gfem._engine import probe
from gfem._engine.coefficients import element_tensors
from gfem._engine.dispersion import eps_at
from gfem._engine.incident import layer_regions, scatterer_elements

if TYPE_CHECKING:
    from gfem._background import Background
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["Current", "current", "elements"]


class Current:
    """The induced current at the quadrature points of the scatterer.

    Attributes:
        points: ``(n, 3)``, m.
        weights: ``(n,)``, m³, the quadrature weights.
        current: ``(n, 3)``, A/m², complex.
    """

    def __init__(self, points: np.ndarray, weights: np.ndarray, current: np.ndarray):
        self.points = points
        self.weights = weights
        self.current = current


def elements(model: Model, background: Background) -> np.ndarray:
    """The scatterer's elements outside the perfectly matched layer, as indices."""
    mask = scatterer_elements(model, background).copy()
    mesh = model.mesh
    for region in layer_regions(model):
        mask[mesh.elements(region)] = False
    return np.flatnonzero(mask)


def _contrast(
    model: Model, background: Background, omega: float, rows: np.ndarray
) -> np.ndarray:
    r""":math:`\varepsilon_\mathrm{eff} - \varepsilon_b` of ``rows``.

    ``(n,)``, or ``(n, 3, 3)`` when any material is a tensor.
    """
    eps = eps_at(model, omega)
    sigma = element_tensors(model, "conductivity").real
    if eps.ndim != sigma.ndim:
        eye = np.eye(3)
        eps, sigma = (
            v if v.ndim == 3 else v[:, None, None] * eye for v in (eps, sigma)
        )
    eps = eps - 1j * sigma / (const.epsilon_0 * omega)
    eps_b = float(_incident.background_material(background).eps_r)
    out = eps[rows]
    if out.ndim == 3:
        return out - eps_b * np.eye(3)
    return out - eps_b


def current(
    field: Field,
    model: Model,
    background: Background,
    rows: np.ndarray,
    omega: float,
) -> Current:
    """The induced current of ``field`` (the total field) in elements ``rows``.

    The quadrature degree is that of the far-field layer (:math:`2p + 4`): the
    moments weight the current by spherical Bessel functions of the position.
    """
    mesh = field.mesh
    nodes = np.asarray(mesh.nodes, float)
    corners = nodes[np.asarray(mesh.tetrahedra)[rows]]
    bary, weights = ff.rule(field.order)
    jac = np.stack([corners[:, i] - corners[:, 3] for i in range(3)], axis=-1)
    points = corners[:, 3][:, None] + np.einsum("tij,pj->tpi", jac, bary[:, :3])
    w = np.abs(np.linalg.det(jac))[:, None] / 6 * weights[None]
    located = probe.Located(
        tetrahedra=np.repeat(rows, len(bary)),
        barycentric=np.tile(bary, (len(rows), 1)),
    )
    prb = probe.probe(mesh, _probe.layout(field), located, field._communicator)  # noqa: SLF001
    e = prb.values(np.asarray(field.coefficients))[0].reshape(len(rows), len(bary), 3)
    contrast = _contrast(model, background, omega, rows)
    if contrast.ndim == 3:
        d = np.einsum("tij,tpj->tpi", contrast, e)
    else:
        d = contrast[:, None, None] * e
    return Current(
        points.reshape(-1, 3),
        w.reshape(-1),
        (1j * omega * const.epsilon_0 * d).reshape(-1, 3),
    )
