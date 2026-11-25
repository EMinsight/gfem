r"""`gfem.optical_force` under JAX transformations: its host kernel and gradient.

The force and the torque about the centroid are a real quadratic form of the
field coefficients :math:`e` on the stress shell of the body
(`gfem._engine.optical_force`). `gfem._adjoint.measure` gives them the solve's
JAX semantics, as for `gfem.power_flow`. The inputs are the field's
coefficients, node coordinates and :math:`\omega`, and the model's leaves. The
body and its shell are fixed at trace time from the mesh topology, so the
inputs the measurement reads are known then:

- the coefficients: differentiable, unless the field is formed from an
  adaptive sweep's frozen basis (the cotangent is one linear form of the
  shell, `gfem._engine.sensitivity._field_shape`);
- :math:`\omega`: differentiable (the real part), as an eigenmode's frequency
  depends on the design;
- the node coordinates, and the material of a region with an element in the
  shell: read, not differentiable (the force is already a first derivative of
  the discrete Lagrangian by the nodes);
- every other model leaf, the body's materials among them: unread, an exact
  zero (their influence arrives through the field).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _pml
from gfem._adjoint.measure import measure
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
from gfem._power_flow import concrete_field
from gfem.mesh import _at_imported_nodes

if TYPE_CHECKING:
    from gfem._engine.sides import SideSpec
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["evaluate"]

#: The material properties the stress shell reads.
_READ = ("eps_r", "mu_r", "tan_delta", "conductivity")

#: Why the node coordinates have no gradient.
NODES_REASON = (
    "the force is the derivative of the discrete Lagrangian by the body's "
    "nodes, and the derivative of that by the nodes is not available"
)

_VECTOR = jax.ShapeDtypeStruct((3,), jnp.float64)


def evaluate(
    field: Field, body: SideSpec, model: Model
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The force, the torque about the centroid and the centroid of a body.

    Raises:
        NonDifferentiableError: A differentiated input is the node
            coordinates, the material of a region in the stress shell or the
            coefficients of an adaptive sweep's field.
    """
    traced = is_traced((field, model))
    shell_regions: set[str] = set()
    if traced:
        # the topology only: the shell never reads a traced value
        flat_model = model.replace(mesh=_at_imported_nodes(model.mesh))
        where = engine("optical_force").shell(concrete_field(field), body, flat_model)
        mesh = model.mesh
        shell_regions = {n for n in mesh.volumes if where.shell[mesh.elements(n)].any()}
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_def = field_inputs(
        field,
        inputs,
        roles,
        lambda name: _field_role(name, field),
    )
    model_def, _ = model_inputs(
        model,
        inputs,
        roles,
        lambda kind: _model_role(kind, shell_regions),
    )
    kernel = _OpticalForceKernel(
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        body=body,
        order=field.order,
    )
    out = measure(kernel, inputs)
    return out["force"], out["torque"], out["centroid"]


def _field_role(name: str, field: Field) -> Role:
    """The role of a field leaf (a `FIELD_INPUTS` value)."""
    if name == "nodes":
        return Role(
            "read", NODES_REASON, "the mesh node coordinates (field.mesh.nodes)"
        )
    if name == "omega":
        return Role("read", None, "the field's omega")
    return field_role(name, field, reduced=reduced_reason("force and torque"))


def _model_role(kind: tuple[Any, ...], shell_regions: set[str]) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    if kind[0] == "nodes":
        return Role(
            "read", NODES_REASON, "the mesh node coordinates (model.mesh.nodes)"
        )
    if kind[0] == "material" and kind[1] in shell_regions and kind[2] in _READ:
        return Role(
            "read",
            f"region {kind[1]!r} has elements in the stress shell around the "
            "body, whose medium the force reads; keep the design out of the "
            "shell (the body's own material is differentiable)",
            f"model.materials[{kind[1]!r}].{kind[2]}",
        )
    return Role("unread", None, "model." + ".".join(str(k) for k in kind))


class _OpticalForceKernel(RoleKernel):
    """The force and torque of a body, and their pullback on the field."""

    name = "gfem.optical_force"
    output_specs = {"force": _VECTOR, "torque": _VECTOR, "centroid": _VECTOR}  # noqa: RUF012

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        body: SideSpec,
        order: int,
    ) -> None:
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.body = body
        self.order = order
        # the shell and its operator of the last design state (forward and
        # backward share it)
        self._cached: tuple[bytes, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, Any, Any, np.ndarray]:
        """The shell, its operator, the model and the coefficients."""
        of = engine("optical_force")
        field = rebuild(self.field_def, values)
        model = rebuild(self.model_def, values)
        key = design_key(values, self.model_def[1])
        key += np.ascontiguousarray(values["omega"]).tobytes()
        if self._cached is None or self._cached[0] != key:
            where = of.shell(field, self.body, model)
            stress = of.build(
                field, model, where, stretch=_pml.field_stretch(model, field)
            )
            self._cached = (key, (where, stress))
        where, stress = self._cached[1]
        e = np.asarray(values["coefficients"])
        of.check_size(e, stress.op, self.order)
        return where, stress, model, e

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        of = engine("optical_force")
        where, stress, model, e = self._host(values)
        omega = float(np.asarray(values["omega"]).real)
        force, torque = of.evaluate(stress, model, where, e, omega)
        return {
            "force": np.where(where.valid[:3], force, np.nan),
            "torque": np.where(where.valid[3:], torque, np.nan),
            "centroid": where.centroid,
        }

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        of = engine("optical_force")
        where, stress, model, e = self._host(values)
        omega = float(np.asarray(values["omega"]).real)
        zero = np.zeros(3)
        force = np.asarray(cotangents.get("force", zero), float)
        torque = np.asarray(cotangents.get("torque", zero), float)
        grad_e, grad_omega = of.pullback(
            stress, model, where, e, omega=omega, force=force, torque=torque
        )
        out: dict[str, np.ndarray] = {}
        if "coefficients" in wanted:
            out["coefficients"] = grad_e
        if "omega" in wanted:
            out["omega"] = np.asarray(grad_omega)
        return out
