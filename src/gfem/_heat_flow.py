r"""`gfem.heat_flow`'s host kernel and its gradient.

The flow is an affine function of the temperature's coefficients and the
source density (`gfem._engine.heat_flow.Balance`), whose temperature
weights read the conductivity on the side's elements. `gfem._adjoint.measure`
gives it the solve's JAX semantics:

- the temperature coefficients: differentiable, the weights of `Balance`
  (the solve carries them on to its own inputs);
- the heat field's coefficients (the source density) and the result's
  face heat: differentiable, linear;
- a material leaf of the thermal conductivity: differentiable through the
  explicit pair,
  :math:`\partial Q / \partial k_e = s \int_e \nabla w \cdot \nabla T`
  on the side's elements (:math:`s` the mesh scale), contracted by the
  sensitivity kernel's material pass;
- a thermal boundary value (a convection coefficient or ambient, a flux):
  differentiable through the explicit terms of the affine flow
  (`gfem._engine.heat_flow.BoundaryTerms`); a fixed temperature reaches it
  through the temperature's coefficients alone, an exact zero here;
- the node coordinates: read, not differentiable;
- every other model leaf: unread, an exact zero.

The side's weights are fixed at trace time from the mesh topology and the
fixed DOFs, so no input moves them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from gfem._adjoint.measure import measure
from gfem._checks import is_traced
from gfem._field_kernels import (
    Role,
    RoleKernel,
    design_key,
    engine,
    model_inputs,
    rebuild,
)
from gfem._probe import NODES_REASON

if TYPE_CHECKING:
    from gfem._engine.heat_flow import Side
    from gfem._engine.sides import SideSpec
    from gfem._engine.thermal import SteadyThermalRule
    from gfem.model import Model
    from gfem.results import SteadyThermalResult

__all__ = ["heat_flow", "steady_rule"]


def steady_rule(model: Model, order: int) -> SteadyThermalRule:
    """A steady thermal rule on ``model``: its space, boundaries and forms."""
    from gfem._engine.thermal import SteadyThermalRule
    from gfem.options import SolverOptions
    from gfem.problems import SteadyThermal

    options = SolverOptions(order=order, linear_solver="direct")
    return SteadyThermalRule(model, SteadyThermal(), options)


def heat_flow(
    result: SteadyThermalResult,
    side: SideSpec,
    model: Model,
    *,
    who: str,
) -> jax.Array:
    """The heat leaving ``side``, W (a slab's on a cross-section).

    ``result`` was checked to be solved on ``model`` (``gfem.measurements._heat_side``).
    """
    order = result.temperature.order
    rule = steady_rule(model, order)
    # the side is topology (a predicate side has concrete nodes, checked):
    # traced nodes are read at their imported positions
    if is_traced(model.mesh.nodes):
        model_at = model.replace(mesh=rule.still_mesh())
    else:
        model_at = model
    weights = engine("heat_flow").side(rule, model_at, side, who=who)
    inputs: dict[str, Any] = {
        "temperature": result.temperature.coefficients,
        "heat": result.heat.coefficients,
        "nodes": result.temperature.mesh.nodes,
    }
    if result._face_heat is not None:  # noqa: SLF001
        inputs["face_heat"] = result._face_heat  # noqa: SLF001
    roles = {
        "face_heat": Role("read", None, "the heat sources' face heat"),
        "temperature": Role("read", None, "the temperature's coefficients"),
        "heat": Role("read", None, "the heat sources' power density"),
        "nodes": Role(
            "read", NODES_REASON, "the mesh node coordinates (result's mesh.nodes)"
        ),
    }
    model_def, kinds = model_inputs(
        model, inputs, roles, lambda kind: _model_role(kind, rule)
    )
    kernel = _HeatFlowKernel(
        roles=roles,
        kinds=kinds,
        model_def=model_def,
        weights=weights,
        order=order,
        heat_order=rule.heat_order(),
    )
    return measure(kernel, inputs)["flow"]


def _model_role(kind: tuple, rule: SteadyThermalRule) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    from gfem._adjoint.solve_vjp import _describe
    from gfem._engine.sensitivity import is_thermal_value, unsupported

    described = _describe(kind)
    if kind[0] == "nodes":
        return Role("read", NODES_REASON, described)
    if is_thermal_value(kind) and rule.role(kind) == "read":
        # the measured flow reads the boundary values itself, outside the solve
        return Role("read", None, described)
    if kind[0] == "material" and rule.role(kind) == "read":
        return Role("read", unsupported(rule.form, kind), described)
    return Role("unread", None, described)


class _HeatFlowKernel(RoleKernel):
    """The heat through a fixed side, and its pullback."""

    name = "gfem.heat_flow"
    output_specs: ClassVar[dict[str, jax.ShapeDtypeStruct]] = {
        "flow": jax.ShapeDtypeStruct((), jnp.float64)
    }

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        kinds: dict[str, tuple],
        model_def: tuple[Any, tuple[str, ...]],
        weights: Side,
        order: int,
        heat_order: np.ndarray,
    ) -> None:
        self.roles = roles
        self.kinds = kinds
        self.model_def = model_def
        self.weights = weights
        self.order = order
        self.heat_order = heat_order
        # the balance of the last design state (forward and backward share it)
        self._cached: tuple[bytes, Any, Any, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, Any, Any]:
        """The model, the rule and the balance of one design state."""
        key = design_key(values, self.model_def[1])
        if self._cached is None or self._cached[0] != key:
            model = rebuild(self.model_def, values)
            rule = steady_rule(model, self.order)
            self._cached = (
                key,
                model,
                rule,
                engine("heat_flow").balance(rule, model, self.weights),
            )
        return self._cached[1:]

    def _density(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """The power density per tetrahedron (zero off the thermal domain)."""
        density = np.zeros(len(self.weights.elements))
        density[self.heat_order] = np.asarray(values["heat"], dtype=float)
        return density

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        _, _, balance = self._host(values)
        t = np.asarray(values["temperature"], dtype=float)
        face = values.get("face_heat")
        flow = balance.flow(t, self._density(values), face)
        return {"flow": np.asarray(flow)}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        from gfem._engine.sensitivity import is_thermal_value, sensitivity

        model, rule, balance = self._host(values)
        ct = float(np.asarray(cotangents["flow"]))
        out: dict[str, np.ndarray] = {}
        if "temperature" in wanted:
            out["temperature"] = ct * balance.temperature
        if "heat" in wanted:
            out["heat"] = ct * balance.source[self.heat_order]
        if "face_heat" in wanted:
            out["face_heat"] = ct * balance.face_source
        wanted_kinds = {n: self.kinds[n] for n in wanted if n in self.kinds}
        t = np.asarray(values["temperature"], dtype=float)
        boundary = {n: k for n, k in wanted_kinds.items() if is_thermal_value(k)}
        materials = {n: k for n, k in wanted_kinds.items() if n not in boundary}
        if boundary:
            terms = engine("heat_flow").boundary_terms(rule, model, self.weights)
            for name, kind in boundary.items():
                out[name] = np.asarray(
                    ct * terms.derivative(model, kind[1], kind[2], t)
                )
        if materials:
            pair = balance.conductivity(rule, model, t)
            grads = sensitivity(
                model,
                [pair],
                set(materials.values()),
                form=rule.form,
                terms=lambda nodes: rule.terms(model, nodes=nodes),
            )
            for name, kind in materials.items():
                leaf = np.asarray(values[name])
                g = ct * np.asarray(grads[kind])
                out[name] = g.sum() if not leaf.ndim else g.reshape(leaf.shape)
        return out
