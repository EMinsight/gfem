"""`gfem.heat_flow`'s host kernel and its gradient.

The flow is an affine function of the temperature's coefficients and the
source density (`gfem._engine.heat_flow.Balance`), whose temperature
weights read the conductivity on the side's elements. `gfem._adjoint.measure`
gives it the solve's JAX semantics:

- the temperature coefficients: differentiable, the weights of `Balance`
  (the solve carries them on to its own inputs);
- the heat field's coefficients (the source density): differentiable,
  linear;
- a material leaf of the thermal conductivity: differentiable through the
  explicit pair, ``dQ/dk_e = scale * int_e grad(w) . grad(T)`` on the side's
  elements, contracted by the sensitivity kernel's material pass;
- the node coordinates: read, not differentiable;
- every other model leaf: unread, an exact zero.

The side's weights are fixed at trace time from the mesh topology and the
fixed DOFs, so no input moves them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gfem._adjoint.measure import measure
from gfem._adjoint.solve_vjp import input_kind
from gfem._probe import NODES_REASON

if TYPE_CHECKING:
    from gfem._engine.heat_flow import Side
    from gfem._engine.thermal import SteadyThermalRule
    from gfem.model import Model
    from gfem.results import SteadyThermalResult

__all__ = ["heat_flow", "steady_rule"]


class _Role(NamedTuple):
    """What the flow does with one input."""

    role: str
    unsupported: str | None
    description: str


def engine() -> Any:
    """The NGSolve heat flow module, imported on first use."""
    return importlib.import_module("gfem._engine.heat_flow")


def steady_rule(model: Model, order: int) -> SteadyThermalRule:
    """A steady thermal rule on ``model``: its space, boundaries and forms."""
    from gfem._engine.thermal import SteadyThermalRule
    from gfem.options import SolverOptions
    from gfem.problems import SteadyThermal

    options = SolverOptions(order=order, linear_solver="direct")
    return SteadyThermalRule(model, SteadyThermal(), options)


def heat_flow(
    result: SteadyThermalResult,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
    *,
    who: str,
) -> jax.Array:
    """The heat leaving ``side``, W (a slab's on a cross-section).

    ``result`` was checked to be solved on ``model`` (`check_solved`).
    """
    order = result.temperature.order
    rule = steady_rule(model, order)
    # the side is topology: read on the mesh at its imported nodes
    still = model.replace(mesh=rule._still())  # noqa: SLF001
    weights = engine().side(rule, still, side, who=who)
    m_paths, m_def = jax.tree_util.tree_flatten_with_path(model)
    inputs: dict[str, Any] = {
        "temperature": result.temperature.coefficients,
        "heat": result.heat.coefficients,
        "nodes": result.temperature.mesh.nodes,
    }
    roles = {
        "temperature": _Role("read", None, "the temperature's coefficients"),
        "heat": _Role("read", None, "the heat sources' power density"),
        "nodes": _Role(
            "read", NODES_REASON, "the mesh node coordinates (result's mesh.nodes)"
        ),
    }
    kinds: dict[str, tuple] = {}
    for path, leaf in m_paths:
        name = "model" + jax.tree_util.keystr(tuple(path))
        kind = input_kind(path, "model")
        inputs[name] = leaf
        kinds[name] = kind
        roles[name] = _model_role(kind, rule)
    kernel = _HeatFlowKernel(
        roles=roles,
        kinds=kinds,
        model_def=(m_def, tuple(kinds)),
        weights=weights,
        order=order,
        heat_order=rule._heat_order(),  # noqa: SLF001
    )
    return measure(kernel, inputs)["flow"]


def _model_role(kind: tuple, rule: SteadyThermalRule) -> _Role:
    """The role of a model leaf, by its `input_kind`."""
    from gfem._adjoint.solve_vjp import _describe
    from gfem._engine.sensitivity import unsupported

    described = _describe(kind)
    if kind[0] == "nodes":
        return _Role("read", NODES_REASON, described)
    if kind[0] == "material" and rule.role(kind) == "read":
        return _Role("read", unsupported(rule.form, kind), described)
    return _Role("unread", None, described)


class _HeatFlowKernel:
    """The heat through a fixed side, and its pullback."""

    name = "gfem.heat_flow"
    pieces: ClassVar[frozenset[str]] = frozenset()
    comm = None
    output_specs: ClassVar[dict[str, jax.ShapeDtypeStruct]] = {
        "flow": jax.ShapeDtypeStruct((), jnp.float64)
    }

    def __init__(
        self,
        *,
        roles: dict[str, _Role],
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

    def role(self, name: str) -> str:
        return self.roles[name].role

    def unsupported(self, name: str) -> str | None:
        return self.roles[name].unsupported

    def describe(self, name: str) -> str:
        return self.roles[name].description

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, Any, Any]:
        """The model, the rule and the balance of one design state."""
        m_def, m_names = self.model_def
        key = b"".join(np.ascontiguousarray(values[n]).tobytes() for n in m_names)
        if self._cached is None or self._cached[0] != key:
            model = jax.tree.unflatten(m_def, [values[n] for n in m_names])
            rule = steady_rule(model, self.order)
            self._cached = (
                key,
                model,
                rule,
                engine().balance(rule, model, self.weights),
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
        return {"flow": np.asarray(balance.flow(t, self._density(values)))}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        from gfem._engine.sensitivity import sensitivity

        model, rule, balance = self._host(values)
        ct = float(np.asarray(cotangents["flow"]))
        out: dict[str, np.ndarray] = {}
        if "temperature" in wanted:
            out["temperature"] = ct * balance.temperature
        if "heat" in wanted:
            out["heat"] = ct * balance.source[self.heat_order]
        materials = {n: self.kinds[n] for n in wanted if n in self.kinds}
        if materials:
            t = np.asarray(values["temperature"], dtype=float)
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
