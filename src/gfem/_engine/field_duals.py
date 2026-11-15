"""The duals of objectives of driven fields (ADRs 0132, 0133, 0144).

An objective reads a driven field only through a recognized measurement,
whose outputs each get a dual solved for a unit cotangent on that output
(ADR 0143): ``A z = g``, ``g`` the output's pullback load on the field's
coefficients, at the output's frequency, on the forward's factorization
(`gfem._engine.driven.AdjointLoads`). Two outputs, keyed:

- ``("power", f, excitation, side)``: `gfem.power_flow` over a side with no
  port and only volume losses, in its volume form (ADR 0132): the net power
  leaving ``side`` is ``-x^H M_w x``, M_w the H(curl) mass matrix weighted
  by `absorbed_weights`; its load is ``-2 conj(M_w x)``.
- ``("mean", f, excitation, regions, c)``: component ``c`` of
  `gfem.region_mean` over ``regions`` (ADR 0133); its load is the averaging
  row (`gfem._engine.fields.mean_rows`).

Each load is the assembly of a continuous volume load ``g`` (`load_parts`),
which the estimate adds to the dual's flux: ``eps_eff z + g / k0^2`` is
divergence free (the equilibrated dual flux, ADR 0144).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._engine import quadrature
from gfem._engine.coefficients import coefficient
from gfem._engine.fields import LENGTH_POWER, fe_space, mean_rows
from gfem._engine.mesh import EngineMesh
from gfem.model import Model

__all__ = ["absorbed_weights", "field_loads", "load_parts", "power_mass"]

#: A dual's key (see the module docstring).
FieldKey = tuple[Any, ...]


def absorbed_weights(model: Model, built: EngineMesh, side: Sequence[str], f: float):
    """Per tetrahedron, w with the power ``side`` absorbs = sum_K w_K int_K |E|^2.

    ``w = 1/2 omega eps0 eps'' scale``: eps'' = -Im eps_eff (tan_delta and
    conductivity of an isotropic material), E the field's view in the mesh's
    length unit (`gfem._engine.fields.LENGTH_POWER`), the integral over the
    mesh's coordinates; zero off ``side``.
    """
    from gfem._engine.estimate import _eps_eff_values

    omega = 2 * math.pi * f
    k = omega / scipy.constants.c * built.scale
    eps = _eps_eff_values(model, built, k)
    if eps.ndim == 3:  # isotropic on the side (`measurements._check_estimated_side`)
        eps = eps[:, 0, 0]
    mask = np.zeros(len(model.mesh.tetrahedra))
    for name in side:
        mask[model.mesh.elements(name)] = 1.0
    scale = built.scale ** (3 + 2 * LENGTH_POWER["hcurl"])
    return 0.5 * omega * scipy.constants.epsilon_0 * scale * (-eps.imag) * mask


def power_mass(
    model: Model, layout: tuple, side: Sequence[str], f: float
) -> sp.csr_matrix:
    """M_w on the fields' space of ``layout`` (support, excluded, order)."""
    support, excluded, order = layout
    built, fes, region = fe_space(model.mesh, support, excluded, "hcurl", order)
    weight = coefficient(built, absorbed_weights(model, built, side, f))
    u, v = fes.TnT()
    form = ngs.BilinearForm(fes)
    form += weight * u * v * quadrature.dx(order, definedon=region)
    form.Assemble()
    rows, cols, values = form.mat.COO()
    n = fes.ndof
    return sp.csr_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )


def field_loads(keys: Sequence[FieldKey]) -> Any:
    """The `gfem._engine.driven.AdjointLoads` of the duals ``keys``.

    At each frequency, the loads of the keys at that frequency, formed from
    the forward's solution there.
    """
    keys = tuple(keys)

    def loads(context: Any) -> dict[Hashable, np.ndarray]:
        out: dict[Hashable, np.ndarray] = {}
        support, excluded, order = context.layout
        for key in keys:
            if key[1] != context.frequency:
                continue
            if key[0] == "power":
                _, f, excitation, side = key
                x = context.solution[:, context.excite.index(excitation)]
                mass = power_mass(context.model, context.layout, side, f)
                g = -2 * np.conj(mass @ x)
            else:
                _, _, _, regions, c = key
                rows = mean_rows(
                    context.model.mesh, (support, excluded, "hcurl", order), regions
                )
                g = rows[c].astype(complex)
            out[key] = context.restrict(g)
        return out

    return loads


def load_parts(
    model: Model,
    built: EngineMesh,
    key: FieldKey,
    ct: complex,
    field_parts: Sequence[ngs.GridFunction],
) -> list[ngs.CoefficientFunction]:
    """``ct g``, the continuous load of ``key`` times ``ct``, as (real, imaginary).

    ``field_parts`` is the excitation's field (real and imaginary parts) at
    the key's frequency, which a power's load reads.
    """
    if key[0] == "power":
        _, f, _, side = key
        weight = coefficient(built, absorbed_weights(model, built, side, f))
        er, ei = field_parts
        # -2 ct w conj(E), ct real
        return [-2 * ct.real * weight * er, 2 * ct.real * weight * ei]
    _, _, _, regions, c = key
    mask = np.zeros(len(model.mesh.tetrahedra))
    for name in regions:
        mask[model.mesh.elements(name)] = 1.0
    chi = coefficient(built, mask)
    volume = ngs.Integrate(chi, built.mesh)
    unit = [0.0, 0.0, 0.0]
    unit[c] = built.scale ** LENGTH_POWER["hcurl"] / volume
    direction = ngs.CF(tuple(unit))
    return [ct.real * chi * direction, ct.imag * chi * direction]
