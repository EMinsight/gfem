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

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gfem._adjoint.measure import measure
from gfem._adjoint.solve_vjp import input_kind
from gfem._boundaries import surfaces_of
from gfem._checks import is_traced
from gfem._probe import NODES_REASON

if TYPE_CHECKING:
    from gfem._engine.power_flow import Surface
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["LAYER_REMEDY", "power_flow"]

#: How to keep a design out of the surface's element layer (the measure
#: wrapper adds jax.lax.stop_gradient as the other way out).
LAYER_REMEDY = "move the surface one element away from the design region"
_FIELD_INPUTS = {"coefficients": "coefficients", "mesh": "nodes", "omega": "omega"}


class _Role(NamedTuple):
    """What the power does with one input."""

    #: ``"read"`` or ``"unread"`` (an exact zero).
    role: str
    #: Why a read input is not differentiable; ``None`` if it is.
    unsupported: str | None
    #: The input, as the user knows it.
    description: str


def engine() -> Any:
    """The NGSolve power flow module, imported on first use."""
    return importlib.import_module("gfem._engine.power_flow")


def power_flow(
    field: Field,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> jax.Array:
    """`gfem.power_flow` on a field or model that may hold JAX tracers."""
    pf = engine()
    where = pf.surface(field, side, model)
    if not is_traced((field.mesh.nodes, model.mesh.nodes)):
        pf.check_nodes(field, model)
    f_paths, f_def = jax.tree_util.tree_flatten_with_path(field)
    m_paths, m_def = jax.tree_util.tree_flatten_with_path(model)
    inputs: dict[str, Any] = {}
    roles: dict[str, _Role] = {}
    for path, leaf in f_paths:
        name = _FIELD_INPUTS[path[0].name]
        inputs[name] = leaf
        roles[name] = _field_role(name, field)
    for path, leaf in m_paths:
        name = "model" + jax.tree_util.keystr(tuple(path))
        inputs[name] = leaf
        roles[name] = _model_role(input_kind(path, "model"), model, where)
    kernel = _PowerFlowKernel(
        roles=roles,
        field_def=(f_def, tuple(_FIELD_INPUTS[p[0].name] for p, _ in f_paths)),
        model_def=(m_def, tuple(n for n in inputs if n.startswith("model"))),
        where=where,
        order=field.order,
    )
    return measure(kernel, inputs)["power"]


def _field_role(name: str, field: Field) -> _Role:
    """The role of a field input."""
    if name == "nodes":
        return _Role(
            "read", NODES_REASON, "the mesh node coordinates (field.mesh.nodes)"
        )
    if name == "omega":
        return _Role(
            "read",
            (
                "a field's angular frequency is fixed: differentiate the solve's "
                "frequency through the kept field instead"
            ),
            "the field's omega",
        )
    if field._reduced:  # noqa: SLF001
        return _Role(
            "read",
            (
                "the field is formed from an adaptive sweep's frozen reduced "
                "basis, which holds no adjoint of the power, so the frozen-basis "
                'gradient is not the solve\'s: solve with sweep="uniform" at the '
                "frequency whose power you differentiate"
            ),
            "the coefficients of a field of an adaptive sweep (gfem.ReducedFields)",
        )
    return _Role("read", None, "the field coefficients")


def _model_role(kind: tuple[Any, ...], model: Model, where: Surface) -> _Role:
    """The role of a model leaf, by its `input_kind`."""
    mesh = model.mesh
    if kind[0] == "nodes":
        return _Role(
            "read", NODES_REASON, "the mesh node coordinates (model.mesh.nodes)"
        )
    if kind[0] == "material":
        region = kind[1]
        described = f"model.materials[{region!r}].{kind[2]}"
        if where.read_elements(mesh)[mesh.elements(region)].any():
            return _Role(
                "read",
                f"region {region!r} has elements in the element layer just "
                "outside the side, whose materials the power reads; " + LAYER_REMEDY,
                described,
            )
        return _Role("unread", None, described)
    if kind[0] == "boundary":
        boundary = model.boundaries[kind[1]]
        described = f"model.boundaries[{kind[1]}]" + "".join(f".{k}" for k in kind[2:])
        for name in surfaces_of(boundary):
            if where.faces[mesh.elements(name)].any():
                return _Role(
                    "read",
                    f"surface {name!r} has faces in the element layer just "
                    "outside the side, whose terms the power reads; " + LAYER_REMEDY,
                    described,
                )
        return _Role("unread", None, described)
    return _Role("unread", None, "model." + ".".join(str(k) for k in kind))


class _PowerFlowKernel:
    """The power through a fixed surface, and its pullback on the field."""

    name = "gfem.power_flow"
    pieces: ClassVar[frozenset[str]] = frozenset()
    comm = None
    output_specs: ClassVar[dict[str, jax.ShapeDtypeStruct]] = {
        "power": jax.ShapeDtypeStruct((), jnp.float64)
    }

    def __init__(
        self,
        *,
        roles: dict[str, _Role],
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

    def role(self, name: str) -> str:
        return self.roles[name].role

    def unsupported(self, name: str) -> str | None:
        return self.roles[name].unsupported

    def describe(self, name: str) -> str:
        return self.roles[name].description

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, np.ndarray, float]:
        """The flux, the coefficients and omega of one design state."""
        pf = engine()
        f_def, f_names = self.field_def
        m_def, m_names = self.model_def
        field = jax.tree.unflatten(f_def, [values[n] for n in f_names])
        model = jax.tree.unflatten(m_def, [values[n] for n in m_names])
        pf.check_nodes(field, model)
        key = b"".join(np.ascontiguousarray(values[n]).tobytes() for n in m_names)
        if self._cached is None or self._cached[0] != key:
            self._cached = (key, pf.build(self.order, model, self.where))
        flux = self._cached[1]
        e = np.asarray(values["coefficients"])
        pf.check_size(e, flux.op, self.order)
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
