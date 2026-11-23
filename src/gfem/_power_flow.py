"""`gfem.power_flow` under JAX transformations: its host kernel and gradient.

The power through a side's surface is ``P = Im[conj(D e) . M e] / c0``
(`gfem._engine.power_flow`), a real quadratic form of the field
coefficients ``e`` on the element layer just outside the side. Its
gradient in ``e`` is two matrix-free applies of the layer's pieces, and
`gfem._adjoint.measure` gives it the solve's JAX semantics, as for
`gfem.Field.at`.

The inputs are the field's coefficients, node coordinates and ``omega``,
and the model's leaves. The surface is fixed at trace time from the mesh
topology, so the inputs the measurement reads are known then:

- the coefficients: differentiable, unless the field is formed from an
  adaptive sweep's frozen basis;
- the node coordinates and ``omega``: read, not differentiable;
- a material field whose region has an element in the layer (or beside a
  face the layer's terms read), and the fill of a sheet with a face there:
  read, not differentiable (the restricted operator has no sensitivity
  kernel);
- every other model leaf: unread, an exact zero.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _pml
from gfem._adjoint.measure import measure
from gfem._boundaries import surfaces_of
from gfem._checks import is_traced
from gfem._field_kernels import (
    Role,
    RoleKernel,
    design_key,
    engine,
    field_inputs,
    field_role,
    model_inputs,
    rebuild,
    reduced_reason,
)
from gfem._probe import NODES_REASON

if TYPE_CHECKING:
    from gfem._engine.power_flow import Surface
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["LAYER_REMEDY", "power_flow"]

#: How to keep a design out of the surface's element layer (the measure
#: wrapper adds jax.lax.stop_gradient as the other way out).
LAYER_REMEDY = "move the surface one element away from the design region"
#: Why a field's frequency has no gradient here.
_FREQUENCY_REASON = (
    "a field's angular frequency is fixed: differentiate the solve's "
    "frequency through the kept field instead"
)


def power_flow(
    field: Field,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> jax.Array:
    """`gfem.power_flow` on a field or model that may hold JAX tracers."""
    flow_engine = engine("power_flow")
    where = flow_engine.surface(field, side, model)
    if not is_traced((field.mesh.nodes, model.mesh.nodes)):
        flow_engine.check_nodes(field, model)
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_def = field_inputs(
        field,
        inputs,
        roles,
        lambda name: field_role(
            name, field, omega=_FREQUENCY_REASON, reduced=reduced_reason("power")
        ),
    )
    model_def, _ = model_inputs(
        model, inputs, roles, lambda kind: _model_role(kind, model, where)
    )
    kernel = _PowerFlowKernel(
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        where=where,
        order=field.order,
    )
    return measure(kernel, inputs)["power"]


def _model_role(kind: tuple[Any, ...], model: Model, where: Surface) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    mesh = model.mesh
    if kind[0] == "nodes":
        return Role(
            "read", NODES_REASON, "the mesh node coordinates (model.mesh.nodes)"
        )
    if kind[0] == "material":
        region = kind[1]
        described = f"model.materials[{region!r}].{kind[2]}"
        if where.read_elements(mesh)[mesh.elements(region)].any():
            return Role(
                "read",
                f"region {region!r} has elements in the element layer just "
                "outside the side, whose materials the power reads; " + LAYER_REMEDY,
                described,
            )
        return Role("unread", None, described)
    if kind[0] == "boundary":
        boundary = model.boundaries[kind[1]]
        described = f"model.boundaries[{kind[1]}]" + "".join(f".{k}" for k in kind[2:])
        for name in surfaces_of(boundary):
            if where.faces[mesh.elements(name)].any():
                return Role(
                    "read",
                    f"surface {name!r} has faces in the element layer just "
                    "outside the side, whose terms the power reads; " + LAYER_REMEDY,
                    described,
                )
        return Role("unread", None, described)
    return Role("unread", None, "model." + ".".join(str(k) for k in kind))


class _PowerFlowKernel(RoleKernel):
    """The power through a fixed surface, and its pullback on the field."""

    name = "gfem.power_flow"
    output_specs: ClassVar[dict[str, jax.ShapeDtypeStruct]] = {
        "power": jax.ShapeDtypeStruct((), jnp.float64)
    }

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        where: Surface,
        order: int,
    ) -> None:
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.where = where
        self.order = order
        # the operator of the last design state (forward and backward share it)
        self._cached: tuple[bytes, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, np.ndarray, float]:
        """The flux, the coefficients and omega of one design state."""
        flow_engine = engine("power_flow")
        field = rebuild(self.field_def, values)
        model = rebuild(self.model_def, values)
        flow_engine.check_nodes(field, model)
        key = design_key(values, self.model_def[1])
        if self._cached is None or self._cached[0] != key:
            stretch = _pml.field_stretch(model, field)
            self._cached = (
                key,
                flow_engine.build(self.order, model, self.where, stretch=stretch),
            )
        flux = self._cached[1]
        e = np.asarray(values["coefficients"])
        flow_engine.check_size(e, flux.op, self.order)
        return flux, e, float(np.asarray(values["omega"]).real)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        flux, e, omega = self._host(values)
        return {"power": np.asarray(flux.power(e, omega))}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no power flow gradient for {wanted}")
        flux, e, omega = self._host(values)
        ct = float(np.asarray(cotangents["power"]))
        return {"coefficients": ct * flux.pullback(e, omega)}
