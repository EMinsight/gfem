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
- the node coordinates: differentiable, through the sensitivity kernel's
  node pass on the layer's pieces (`gfem._engine.power_flow.node_pullback`),
  except for the corrected power and the cross sections of an incident
  column, whose load reads the incident field's interpolant on the mesh;
- ``omega``: read, not differentiable;
- a material field whose region has an element in the layer (or beside a
  face the layer's terms read), and the fill of a sheet with a face there:
  read, not differentiable (the restricted operator has no sensitivity
  kernel);
- every other model leaf: unread, an exact zero.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, overload

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _pml
from gfem._adjoint.measure import measure
from gfem._boundaries import THERMAL_VALUES, surfaces_of
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
from gfem.mesh import _at_imported_nodes

if TYPE_CHECKING:
    from gfem._engine.power_flow import Surface
    from gfem._engine.sides import SideSpec
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["LAYER_REMEDY", "balance", "concrete_field", "power_flow"]

#: How to keep a design out of the surface's element layer (the measure
#: wrapper adds jax.lax.stop_gradient as the other way out).
LAYER_REMEDY = "move the surface one element away from the design region"


@overload
def side_names(side: Any, who: str) -> SideSpec: ...
@overload
def side_names(side: Any, who: str, *, default: str) -> SideSpec | None: ...
def side_names(side: Any, who: str, *, default: str | None = None) -> SideSpec | None:
    """A measurement's ``side`` as a tuple of region names or a predicate.

    Args:
        side: The argument as given.
        who: The measurement, for the message.
        default: What ``side=None`` means, for a measurement that takes it
            (it is then returned as is); ``None`` refuses it.

    Raises:
        TypeError: ``side`` is neither a volume region name, a non-empty
            sequence of names, a predicate on element centroids, nor an
            accepted ``None``.
    """
    if side is None and default is not None:
        return None
    if isinstance(side, str):
        return (side,)
    if callable(side):
        return side
    if isinstance(side, Sequence) and side and all(isinstance(s, str) for s in side):
        return tuple(side)
    none = "" if default is None else f"None ({default}), "
    raise TypeError(
        f"{who}'s side must be {none}a volume region name, a sequence of names "
        f"or a predicate on element centroids, got {side!r}."
    )


def power_flow(
    field: Field,
    side: SideSpec,
    model: Model,
) -> jax.Array:
    """`gfem.power_flow` on a field or model that may hold JAX tracers.

    The total field of an incident-field column (one carrying its load
    reference) has its load pairing subtracted (`corrected_where`).
    """
    pf = engine("power_flow")
    where = pf.surface(concrete_field(field), side, model)
    part = field._incident  # noqa: SLF001
    if part is not None:
        _check_corrected(field, where, model)
        return balance(field, where, model, mode="corrected")["power"]
    return balance(field, where, model)["power"]


def concrete_field(field: Field) -> Field:
    """``field`` at the imported node positions if its nodes are traced.

    For what a measurement fixes at trace time from the topology (its
    surface, its layer, the element of a source), with zero coefficients.
    """
    if not is_traced(field.mesh.nodes):
        return field
    return dataclasses.replace(
        field,
        mesh=_at_imported_nodes(field.mesh),
        coefficients=np.zeros(field.coefficients.shape, field.coefficients.dtype),
        omega=None if field.omega is None else 0.0,
    )


def _check_corrected(field: Field, where: Surface, model: Model) -> None:
    """Raise where the corrected power flow of an incident column has no meaning.

    Raises:
        ValueError: The field is an incident part, or the side holds
            perfectly-matched-layer elements or reaches a wave port.
    """
    part = field._incident  # noqa: SLF001
    assert part is not None  # noqa: S101 - the caller checked
    if part.is_incident:
        raise ValueError(
            "power_flow of the incident part of an incident-field column "
            "balances no load: read the total field (the default part) for the "
            "net power out of a side, part='scattered' for the scattered power, "
            "or gfem.cross_sections."
        )
    mesh = model.mesh
    for region in part.layers:
        if where.inside[mesh.elements(region)].any():
            raise ValueError(
                f"power_flow: the side holds elements of the PML layer region "
                f"{region!r}, where an incident-field column's kept field is the "
                "scattered field, not the total: keep the side outside the "
                "layer, or use gfem.cross_sections or part='scattered'."
            )
    importlib.import_module("gfem._engine.cross_sections").check_side(
        model, where, "power_flow"
    )


def balance(
    field: Field,
    where: Surface,
    model: Model,
    *,
    mode: str = "power",
) -> dict[str, jax.Array]:
    """The kernel's outputs on a surface (see `_PowerFlowKernel`)."""
    pf = engine("power_flow")
    if not is_traced((field.mesh.nodes, model.mesh.nodes)):
        pf.check_nodes(field, model)
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_def = field_inputs(
        field,
        inputs,
        roles,
        lambda name: (
            _nodes_role("field", mode, model)
            if name == "nodes"
            else field_role(
                name,
                field,
                reduced=reduced_reason("power"),
            )
        ),
    )
    model_def, _ = model_inputs(
        model,
        inputs,
        roles,
        lambda kind: (
            _nodes_role("model", mode, model)
            if kind[0] == "nodes"
            else _model_role(kind, model, where)
        ),
    )
    kernel = _PowerFlowKernel(
        comm=field._communicator,  # noqa: SLF001
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        where=where,
        order=field.order,
        mode=mode,
    )
    return measure(kernel, inputs)


#: Why the corrected power and the cross sections have no node gradient.
INCIDENT_NODES_REASON = (
    "the incident load they subtract reads the incident field's interpolant on "
    "the mesh, whose derivative by the node coordinates is not available"
)


def _nodes_role(owner: str, mode: str, model: Model) -> Role:
    """The role of the node coordinates of the field or of the model."""
    described = f"the mesh node coordinates ({owner}.mesh.nodes)"
    if mode != "power":
        return Role("read", INCIDENT_NODES_REASON, described)
    terms = importlib.import_module("gfem._engine.terms")
    return Role("read", terms.absorbing_shape_unsupported(model), described)


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
    if kind[0] == "boundary" and kind[2] not in THERMAL_VALUES:
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


_SCALAR = jax.ShapeDtypeStruct((), jnp.float64)
#: The kernel's outputs by mode: ``"power"`` the flux of the field,
#: ``"corrected"`` the flux of an incident column's total field minus its
#: load pairing, ``"cross_sections"`` the absorbed and scattered powers of
#: such a field (`gfem._engine.cross_sections`).
_OUTPUTS = {
    "power": ("power",),
    "corrected": ("power",),
    "cross_sections": ("absorption", "scattering"),
}


class _PowerFlowKernel(RoleKernel):
    """The power through a fixed surface, and its pullback on the field.

    Under ranks (``comm``) a collective host call: the coefficients are a
    rank's piece, the outputs agreed, and the pullback a rank's part of a load.
    """

    pieces: ClassVar[frozenset[str]] = frozenset({"coefficients"})

    def __init__(
        self,
        *,
        comm: Any,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        where: Surface,
        order: int,
        mode: str = "power",
    ) -> None:
        self.comm = comm
        self.mode = mode
        self.name = (
            "gfem.cross_sections" if mode == "cross_sections" else "gfem.power_flow"
        )
        self.output_specs = dict.fromkeys(_OUTPUTS[mode], _SCALAR)
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.where = where
        self.order = order
        # the operator of the last design state (forward and backward share it)
        self._cached: tuple[bytes, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, Any, np.ndarray]:
        """The flux, the incident reading (or ``None``) and the coefficients."""
        pf = engine("power_flow")
        field = rebuild(self.field_def, values)
        model = rebuild(self.model_def, values)
        pf.check_nodes(field, model)
        key = design_key(values, self.model_def[1])
        key += np.ascontiguousarray(values["omega"]).tobytes()
        if self._cached is None or self._cached[0] != key:
            stretch = _pml.field_stretch(model, field)
            flux = pf.build(
                self.order, model, self.where, stretch=stretch, comm=self.comm
            )
            reading = None
            part = field._incident  # noqa: SLF001
            if self.mode != "power" and part is not None:
                xs = importlib.import_module("gfem._engine.cross_sections")
                reading = xs.reading(
                    field, part, model, self.where, flux, stretch=stretch
                )
            self._cached = (key, (flux, reading))
        flux, reading = self._cached[1]
        e = pf.local_coefficients(
            np.asarray(values["coefficients"]), flux.op, self.comm
        )
        pf.check_size(e, flux.op, self.order)
        return flux, reading, e

    def _padded(self, grad: np.ndarray, values: dict[str, np.ndarray]) -> dict:
        """The coefficients' cotangent at the field's shape (a rank's piece padded)."""
        out = np.zeros(np.shape(values["coefficients"]), complex)
        out[: len(grad)] = grad
        return {"coefficients": out}

    def _nodes(
        self,
        values: dict[str, np.ndarray],
        flux: Any,
        e: np.ndarray,
        omega: float,
        *,
        ct: float,
        names: list[str],
    ) -> dict[str, np.ndarray]:
        """The node coordinates' cotangents: on the first wanted, zero on the rest.

        The field's and the model's node coordinates are one position (the
        host refuses others), so the gradient is given once.
        """
        model = rebuild(self.model_def, values)
        grad = ct * engine("power_flow").node_pullback(
            flux, model, self.where, e, omega, comm=self.comm
        )
        out = {n: np.zeros(np.shape(values[n])) for n in names}
        out[names[0]] = grad.reshape(np.shape(values[names[0]]))
        return out

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        flux, reading, e = self._host(values)
        omega = float(np.asarray(values["omega"]).real)
        if self.mode == "power":
            return {"power": np.asarray(flux.power(e, omega))}
        if self.mode == "corrected":
            return {"power": np.asarray(flux.power(e, omega) - reading.load(e))}
        scattered = e - reading.incident
        return {
            "absorption": np.asarray(reading.load(e) - flux.power(e, omega)),
            "scattering": np.asarray(flux.power(scattered, omega)),
        }

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        nodes = [n for n in wanted if n == "nodes" or n.startswith("model.mesh")]
        others = [n for n in wanted if n != "coefficients" and n not in nodes]
        if others or (nodes and self.mode != "power"):
            # the measure wrapper raised for these
            raise NotImplementedError(f"no power flow gradient for {wanted}")
        flux, reading, e = self._host(values)
        omega = float(np.asarray(values["omega"]).real)

        def ct(name: str) -> float:
            return float(np.asarray(cotangents.get(name, 0.0)))

        if self.mode == "power":
            out = {}
            if "coefficients" in wanted:
                out = self._padded(ct("power") * flux.pullback(e, omega), values)
            if nodes:
                out |= self._nodes(values, flux, e, omega, ct=ct("power"), names=nodes)
            return out
        if self.mode == "corrected":
            grad = flux.pullback(e, omega) - reading.load_pullback()
            return self._padded(ct("power") * grad, values)
        grad = np.zeros(e.shape, complex)
        if "absorption" in cotangents:
            grad += ct("absorption") * (
                reading.load_pullback() - flux.pullback(e, omega)
            )
        if "scattering" in cotangents:
            grad += ct("scattering") * flux.pullback(e - reading.incident, omega)
        return self._padded(grad, values)
