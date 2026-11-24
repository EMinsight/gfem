r"""The loss matrices of driven fields under JAX transformations.

`loss_matrices` gives, for the kept fields of the ports driven together at
one frequency, the per-tetrahedron and per-triangle Hermitian
:math:`P \times P` loss matrices (`gfem._engine.heat`): the power each
element and sheet face dissipates is
:math:`\sum_{pq} \overline{a_p} a_q D[p, q]` W for incident waves
:math:`a_p` (:math:`|a_p|^2` W). `gfem._adjoint.measure` gives the host
kernel the solve's JAX semantics, as for `gfem.power_flow`.

The inputs are the fields' coefficients, node coordinates and ``omega``,
and the model's leaves:

- the coefficients: differentiable (the field term), unless a field is
  formed from an adaptive sweep's frozen basis;
- the node coordinates and ``omega``: read, not differentiable;
- the permittivity, loss tangent and conductivity of a region with heated
  elements, its permittivity law's parameters, the fill of a `Conductor`
  or `Impedance` sheet, an `Impedance`'s resistance and a law blend's
  density: read and differentiable (the explicit term, linear in each);
- every other leaf (a real ``mu_r``, an `Impedance`'s inductance or
  capacitance, a lumped port's values, thermal properties): unread, an
  exact zero here; its influence arrives through the fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._adjoint.measure import measure
from gfem._boundaries import Conductor, Impedance
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
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["FREQUENCY_REASON", "REDUCED_REASON", "heated_regions", "loss_matrices"]

#: Why a field's frequency has no derivative here.
FREQUENCY_REASON = (
    "the frequency is a fixed input of the driven solve, so the heat it "
    "dissipates has no derivative with respect to it"
)
#: Why a field of an adaptive sweep has no gradient here.
REDUCED_REASON = reduced_reason("dissipated power")
#: The outputs: real and imaginary parts of the tetrahedra's and the
#: triangles' matrices.
OUTPUTS = ("tetrahedra_re", "tetrahedra_im", "triangles_re", "triangles_im")


def loss_matrices(fields: Sequence[Field], model: Model) -> dict[str, jax.Array]:
    """The loss matrices of ``fields`` (one frequency, one model), W.

    Returns ``tetrahedra_re`` / ``tetrahedra_im`` of shape ``(n_tetrahedra,
    P, P)`` and ``triangles_re`` / ``triangles_im`` of shape
    ``(n_triangles, P, P)``, in gfem order: the real and imaginary parts of
    Hermitian matrices, zero off the heated tetrahedra and the sheet faces.

    The caller has checked the fields (driven, solved on ``model``, one
    frequency) and the refusals by name.
    """
    heat = engine("heat")
    support = _support(fields, model)
    heated = heat.heat_elements(model, support)
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_defs = tuple(
        field_inputs(
            field,
            inputs,
            roles,
            lambda name, field=field: field_role(
                name, field, omega=FREQUENCY_REASON, reduced=REDUCED_REASON
            ),
            suffix=f"[{i}]",
        )
        for i, field in enumerate(fields)
    )
    model_def, _ = model_inputs(
        model, inputs, roles, lambda kind: _model_role(kind, model, heated)
    )
    kernel = _HeatKernel(
        roles=roles,
        field_defs=field_defs,
        model_def=model_def,
        support=support,
        order=fields[0].order,
        n_triangles=len(model.mesh.triangles),
        n_tetrahedra=len(model.mesh.tetrahedra),
    )
    return measure(kernel, inputs)


def heated_regions(field: Field, model: Model) -> dict[str, np.ndarray]:
    """The volume regions whose loss is heat, with their elements in the support.

    The regions outside a perfectly matched layer that hold an element of
    the field's support, in mesh order, each with those elements.

    Raises:
        ValueError: The field's mesh topology or support is not the model's.
    """
    from gfem._pml import physical_regions

    support = engine("power_flow").check_support(field, model)
    mesh = model.mesh
    out = {}
    for region in physical_regions(mesh.volumes, model.pml):
        elements = mesh.elements(region)
        elements = elements[support[elements]]
        if len(elements):
            out[region] = elements
    return out


def _support(fields: Sequence[Field], model: Model) -> np.ndarray:
    """The fields' common support, after checking each was solved on ``model``.

    With traced node coordinates only the topology is checked.

    Raises:
        ValueError: A field was not solved on ``model``, or the fields have
            different supports.
    """
    flow_engine = engine("power_flow")
    concrete = not is_traced((*(f.mesh.nodes for f in fields), model.mesh.nodes))
    check = flow_engine.check_field if concrete else flow_engine.check_support
    supports = [check(field, model) for field in fields]
    if any(not np.array_equal(s, supports[0]) for s in supports[1:]):
        raise ValueError(
            "The fields have different supports: the loss matrices couple "
            "fields of one solve domain only."
        )
    return supports[0]


def _model_role(kind: tuple, model: Model, heated: np.ndarray) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    if kind[0] == "nodes":
        return Role(
            "read", NODES_REASON, "the mesh node coordinates (model.mesh.nodes)"
        )
    if kind[0] in ("material", "law"):
        rest = "".join(f".{k}" for k in kind[2:])
        described = f"model.materials[{kind[1]!r}]{rest}"
        read = heated[model.mesh.elements(kind[1])].any() and (
            kind[0] == "law" or kind[2] in engine("heat").LOSSY_PROPERTIES
        )
    elif kind[0] == "boundary":
        described = f"model.boundaries[{kind[1]}]" + "".join(f".{k}" for k in kind[2:])
        boundary = model.boundaries[kind[1]]
        read = isinstance(boundary, Conductor | Impedance) and (
            tuple(kind[2:]) == ("fill", "fill")
            or (isinstance(boundary, Impedance) and kind[2:] == ("resistance",))
        )
    else:
        described, read = "model." + ".".join(str(k) for k in kind), False
    if read:
        return Role("read", None, described, kind)
    return Role("unread", None, described)


class _HeatKernel(RoleKernel):
    """The loss matrices of P fields at one frequency, and their pullback.

    The forward gives `OUTPUTS`; the backward the fields' coefficient
    cotangents (field term) and the read model leaves' (explicit term).
    """

    name = "gfem.dissipated_power"

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        field_defs: tuple[tuple[Any, tuple[str, ...]], ...],
        model_def: tuple[Any, tuple[str, ...]],
        support: np.ndarray,
        order: int,
        n_tetrahedra: int,
        n_triangles: int,
    ) -> None:
        self.roles = roles
        self.field_defs = field_defs
        self.model_def = model_def
        self.support = support
        self.order = order
        p = len(field_defs)
        self.output_specs = {
            "tetrahedra_re": jax.ShapeDtypeStruct((n_tetrahedra, p, p), jnp.float64),
            "tetrahedra_im": jax.ShapeDtypeStruct((n_tetrahedra, p, p), jnp.float64),
            "triangles_re": jax.ShapeDtypeStruct((n_triangles, p, p), jnp.float64),
            "triangles_im": jax.ShapeDtypeStruct((n_triangles, p, p), jnp.float64),
        }
        # the lossy pieces of the last design state and frequency
        self._cached: tuple[bytes, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, np.ndarray, Any]:
        """The losses, the fields' coefficients ``(ndof, P)`` and the model."""
        flow_engine = engine("power_flow")
        model = rebuild(self.model_def, values)
        columns, omegas = [], []
        for definition in self.field_defs:
            field = rebuild(definition, values)
            flow_engine.check_nodes(field, model)
            columns.append(np.asarray(field.coefficients))
            omegas.append(float(np.real(np.asarray(field.omega))))
        if any(w != omegas[0] for w in omegas):
            raise ValueError(
                "The fields have different frequencies: the loss matrices "
                "couple fields of one frequency only."
            )
        key = design_key(values, self.model_def[1]) + np.float64(omegas[0]).tobytes()
        if self._cached is None or self._cached[0] != key:
            self._cached = (
                key,
                engine("heat").build(model, self.order, self.support, omegas[0]),
            )
        losses = self._cached[1]
        e = np.stack(columns, axis=1)
        flow_engine.check_size(e[:, 0], losses.op, self.order)
        return losses, e, model

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        losses, e, _ = self._host(values)
        tets, faces = losses.matrices(e)
        return {
            "tetrahedra_re": tets.real,
            "tetrahedra_im": tets.imag,
            "triangles_re": faces.real,
            "triangles_im": faces.imag,
        }

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        losses, e, _ = self._host(values)
        shapes = {k: s.shape for k, s in self.output_specs.items()}

        def ct(name: str) -> np.ndarray:
            if name in cotangents:
                return np.asarray(cotangents[name], dtype=float)
            return np.zeros(shapes[name])

        g_tets = ct("tetrahedra_re") + 1j * ct("tetrahedra_im")
        g_faces = ct("triangles_re") + 1j * ct("triangles_im")
        out: dict[str, np.ndarray] = {}
        fields = [n for n in wanted if n.startswith("coefficients[")]
        if fields:
            grads = losses.pullback(e, g_tets, g_faces)
            for name in fields:
                out[name] = grads[:, int(name[len("coefficients[") : -1])]
        kinds = {self.roles[n].kind: n for n in wanted if n.startswith("model")}
        explicit = losses.explicit(e, g_tets, g_faces, set(kinds))
        for kind, name in kinds.items():
            g = np.asarray(explicit[kind])
            leaf = np.asarray(values[name])
            if not leaf.ndim:
                g = g.sum()
            elif g.ndim > leaf.ndim:  # a uniform leaf of principal values, (3,)
                g = g.sum(axis=0)
            out[name] = g.reshape(leaf.shape)
        return out
