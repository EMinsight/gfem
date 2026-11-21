r"""Layer-grid designs of a 2D mesh, sampled at the pinned quadrature points.

A `gfem.MaterialField` built from a `gfem.design.LayerDensity` holds pixels,
not values per element. A solve samples them at the points of its pinned rule
of degree :math:`2p` (`gfem._engine.quadrature`): the density at a point is
the bilinear map of the pixels, :math:`\rho_q = \sum_k w_k(x_q)\rho_k`, a fixed
sparse matrix :math:`P_q` (`point_map`, with the grid's periodic wrap), and
the blend is applied per point, :math:`\varepsilon_r(\rho_q)`.

The per-point values live on NGSolve's ``IntegrationRuleSpace`` of order
:math:`p`, whose degrees of freedom are exactly the rule's points, element by
element in rule order: a coefficient on it assembles a form on the pinned rule
as the analytic coefficient would, and a linear form with its test function
gives :math:`\omega_q |J_e|` times the integrand at each point. The sensitivity
kernel's point pass integrates each field pair that way, and :math:`P_q^T`
carries the per-point cotangents back to the pixels.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from ngsolve.comp import IntegrationRuleSpace

from gfem._engine.cache import discretization
from gfem._materials import Blend, MaterialField, PixelGeometry, refractive_index
from gfem.design import _bilinear
from gfem.model import Model

__all__ = [
    "PointMap",
    "constant_permittivity",
    "function",
    "geometry_of",
    "other",
    "point_blends",
    "point_map",
    "rule_space",
]


@dataclasses.dataclass(frozen=True, eq=False)
class PointMap:
    """The points of a layer grid's region and the bilinear map onto them.

    Attributes:
        dofs: The region's points as degrees of freedom of the rule space
            (`rule_space`): NGSolve element by element, in rule order.
        cells: The region's NGSolve elements, for ``definedonelements``.
        elements: One boolean per gfem element: the region's.
        matrix: :math:`P_q`, ``(points, pixels)``: the density at the points
            from the flat pixel array.
    """

    dofs: np.ndarray
    cells: Any
    elements: np.ndarray
    matrix: sp.csr_array


def point_blends(model: Model) -> dict[str, MaterialField]:
    """The regions whose field holds a layer grid's pixels, in materials order."""
    return {
        region: m
        for region, m in model.materials.items()
        if isinstance(m, MaterialField)
        and m.blend_ends is not None
        and m.blend_ends.pixels is not None
    }


def geometry_of(model: Model, region: str) -> PixelGeometry:
    """The pixel geometry of ``region``'s layer grid.

    Raises:
        ValueError: The region's material is no layer grid's (an internal
            error: callers take regions from `point_blends`).
    """
    field = point_blends(model).get(region)
    geometry = None if field is None else getattr(field.blend_ends, "pixels", None)
    if geometry is None:  # pragma: no cover - callers take point_blends' regions
        msg = f"Region {region!r} holds no layer grid."
        raise ValueError(msg)
    return geometry


def rule_space(model: Model, order: int, *, is_complex: bool) -> ngs.FESpace:
    """The space of values at the points of the pinned rule of ``order``.

    One per discretization, order and number field.
    """
    disc = discretization(model.mesh, model._periodic)  # noqa: SLF001
    return disc.space(
        ("rule points", order, is_complex),
        lambda: IntegrationRuleSpace(disc.built.mesh, order=order, complex=is_complex),
    )


def point_map(model: Model, region: str, order: int) -> PointMap:
    """The `PointMap` of ``region``'s layer grid for a solve of ``order``.

    Cached with the discretization (the mesh at its node coordinates), by the
    grid's geometry and the order.
    """
    geometry = geometry_of(model, region)
    disc = discretization(model.mesh, model._periodic)  # noqa: SLF001

    def build() -> PointMap:
        built = disc.built
        space = rule_space(model, order, is_complex=False)
        n_cells = len(built.cell_order)
        per_cell = space.ndof // max(n_cells, 1)
        elements = np.zeros(len(model.mesh.volume_elements), bool)
        elements[model.mesh.elements(region)] = True
        mine = elements[built.cell_order]
        dofs = (np.flatnonzero(mine)[:, None] * per_cell + np.arange(per_cell)).ravel()
        xy = []
        for coordinate in (ngs.x, ngs.y):
            gf = ngs.GridFunction(space)
            gf.Set(coordinate)
            xy.append(gf.vec.FV().NumPy()[dofs] * built.scale)
        index, weight = _bilinear(
            np.stack(xy, axis=1),
            np.asarray(geometry.origin),
            geometry.pitch,
            geometry.shape,
            geometry.periodic,
        )
        n_points = len(dofs)
        rows = np.repeat(np.arange(n_points), index.shape[1])
        matrix = sp.csr_array(
            (weight.ravel(), (rows, index.ravel())),
            shape=(n_points, geometry.shape[0] * geometry.shape[1]),
        )
        return PointMap(dofs, ngs.BitArray(mine.tolist()), elements, matrix)

    return disc.space(("layer points", geometry, order), build)


def function(model: Model, order: int, points: PointMap, values: np.ndarray) -> Any:
    """``values`` at ``points`` as a coefficient on the rule space (zero elsewhere)."""
    is_complex = np.iscomplexobj(values)
    gf = ngs.GridFunction(rule_space(model, order, is_complex=is_complex))
    gf.vec.FV().NumPy()[points.dofs] = values
    return gf


def constant_permittivity(
    ends: Blend, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""The permittivity of a blend of two constant ends, and its slope in rho.

    ``"index"``: :math:`((1 - \rho) N_a + \rho N_b)^2`; a weight scheme:
    :math:`\varepsilon_a + w(\rho)(\varepsilon_b - \varepsilon_a)`.
    """
    a, b = complex(ends.void), complex(ends.solid)  # ty: ignore[no-matching-overload]
    if ends.scheme == "index":
        na, nb = refractive_index(a), refractive_index(b)
        n = (1 - rho) * na + rho * nb
        return n**2, 2 * n * (nb - na)
    return a + ends.weight(rho) * (b - a), ends.slope(rho) * (b - a)


def other(
    ends: Blend, name: str, rho: np.ndarray, default: float
) -> tuple[np.ndarray, np.ndarray]:
    """Property ``name`` of the blend at the densities ``rho``, and its slope.

    ``default`` with a zero slope when the ends do not both define it.
    """
    found = ends.other(name, rho)
    if found is None:
        return np.full(len(rho), default), np.zeros(len(rho))
    return np.asarray(found[0]), np.asarray(found[1])
