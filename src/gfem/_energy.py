r"""The electric energy of fields under JAX transformations.

`electric_energy` gives, per volume region, the time-averaged energy
:math:`W_e[r] = \frac{1}{4} \varepsilon_0 \int_r w\, |E|^2\, dV` of a field.
`gfem._adjoint.measure` gives the host kernel the solve's JAX semantics, as
for `gfem.power_flow`. The kernel outputs, per region, the integral with the
weight :math:`w = \mathrm{Re}(\varepsilon_r)` (the real tensor) off a
permittivity law region and :math:`w = 1` on one, so that it reads no law
parameter; JAX multiplies a law region by its Brillouin weight
:math:`b = \mathrm{Re}\,[\varepsilon(f) + f \varepsilon'(f)]` at the field's
real frequency :math:`f = \mathrm{Re}(\omega) / 2\pi`, computed from the law
itself. The law's parameters and a mode field's complex :math:`\omega` then
differentiate exactly.

The inputs are the field's coefficients, node coordinates and ``omega`` and
the model's leaves:

- the coefficients: differentiable (the field term), unless the field is
  formed from an adaptive sweep's frozen basis;
- the permittivity of a region off a law (scalar, per element, per axis),
  and the node coordinates: read and differentiable (the explicit term of
  the energy, and the sensitivity kernel's node pass);
- the field's ``omega``: read by JAX only, through the Brillouin weight;
- every other leaf (loss tangent, conductivity, ``mu_r``, boundaries, ports,
  a law's parameters): unread by the kernel, an exact zero here; their
  influence arrives through the field, or for a law through JAX.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

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
    field_inputs,
    field_role,
    model_inputs,
    rebuild,
    reduced_reason,
)
from gfem._materials import PermittivityLaw

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["REDUCED_REASON", "brillouin_weight", "electric_energy"]

#: Why a field of an adaptive sweep has no gradient here.
REDUCED_REASON = reduced_reason("electric energy")


def brillouin_weight(law: PermittivityLaw, omega: jax.Array) -> jax.Array:
    r"""The Brillouin weight :math:`\mathrm{Re}\,[\varepsilon + f \varepsilon']`.

    Taken at :math:`f = \mathrm{Re}(\omega) / 2\pi`, in JAX.
    """
    f = jnp.real(omega) / (2 * math.pi)
    eps, slope = jax.jvp(law, (f,), (jnp.ones_like(f),))
    return jnp.real(eps + f * slope)


def electric_energy(field: Field, model: Model) -> dict[str, jax.Array]:
    """The time-averaged electric energy per region, J (see `gfem.electric_energy`).

    The caller has checked the field and the model (a time-harmonic E field,
    solved on ``model``) and the refusals by name.

    Raises:
        ValueError: A law region's Brillouin weight is not positive at the
            field's frequency.
    """
    energy = engine("energy")
    power_flow = engine("power_flow")
    concrete = not is_traced((field.mesh.nodes, model.mesh.nodes))
    support = (
        power_flow.check_field(field, model)
        if concrete
        else power_flow.check_support(field, model)
    )
    names = energy.regions(model, support)
    laws: dict[str, PermittivityLaw] = {}
    for region, material in model.materials.items():
        law = getattr(material, "eps_r", None)
        if isinstance(law, PermittivityLaw):
            laws[region] = law
    if laws and not is_traced((model, field.omega)):
        # a concrete frequency and model: raise here, not from the callback
        engine("dispersion").check_energy(
            model,
            float(np.real(np.asarray(field.omega))) / (2 * math.pi),
            "field at",
            brillouin=True,
        )
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_def = field_inputs(
        field,
        inputs,
        roles,
        lambda name: _field_role(name, field),
    )
    model_def, _ = model_inputs(
        model, inputs, roles, lambda kind: _model_role(kind, laws)
    )
    kernel = _EnergyKernel(
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        support=support,
        regions=names,
        order=field.order,
    )
    omega = jnp.asarray(field.omega)
    unit_weighted = measure(kernel, inputs)["energy"]
    out = {}
    for i, region in enumerate(names):
        value = unit_weighted[i]
        if region in laws:
            value = value * brillouin_weight(laws[region], omega)
        out[region] = value
    return out


def _field_role(name: str, field: Field) -> Role:
    """The role of a field input: ``omega`` is JAX's, not the kernel's."""
    if name == "omega":
        return Role("unread", None, "the field's omega")
    return field_role(name, field, omega="", reduced=REDUCED_REASON, nodes=None)


def _model_role(kind: tuple, laws: dict[str, PermittivityLaw]) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    if kind[0] == "nodes":
        return Role("read", None, "the mesh node coordinates (model.mesh.nodes)", kind)
    if kind[0] == "material" and kind[2] == "eps_r" and kind[1] not in laws:
        return Role("read", None, f"model.materials[{kind[1]!r}].eps_r", kind)
    return Role("unread", None, "model." + ".".join(str(k) for k in kind))


class _EnergyKernel(RoleKernel):
    """The regions' unit-weight energies and their pullback.

    The forward gives the energy of each region (J, ``w = 1`` on a law
    region); the backward the field's coefficient cotangent (field term) and
    the read model leaves' (explicit term).
    """

    name = "gfem.electric_energy"

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        support: np.ndarray,
        regions: tuple[str, ...],
        order: int,
    ) -> None:
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.support = support
        self.regions = regions
        self.order = order
        self.output_specs = {
            "energy": jax.ShapeDtypeStruct((len(regions),), jnp.float64)
        }
        # the parts of the last design state
        self._cached: tuple[bytes, Any] | None = None

    def _host(self, values: dict[str, np.ndarray]) -> tuple[Any, np.ndarray, Any]:
        """The energy parts, the coefficients and the model."""
        model = rebuild(self.model_def, values)
        field = rebuild(self.field_def, values)
        engine("power_flow").check_nodes(field, model)
        key = design_key(values, self.model_def[1])
        if self._cached is None or self._cached[0] != key:
            self._cached = (
                key,
                engine("energy").build(model, self.order, self.support, self.regions),
            )
        parts = self._cached[1]
        e = np.asarray(field.coefficients)
        engine("power_flow").check_size(e, parts.op, self.order)
        return parts, e, (model, field)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        parts, e, (model, field) = self._host(values)
        if parts.laws:
            # a law region's weight must be positive at the field's frequency
            omega = float(np.real(np.asarray(field.omega)))
            engine("dispersion").check_energy(
                model, omega / (2 * math.pi), "field at", brillouin=True
            )
        return {"energy": parts.energies(e)}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        parts, e, (model, _) = self._host(values)
        sens = engine("sensitivity")
        ct = np.real(np.asarray(cotangents["energy"], dtype=float))
        out: dict[str, np.ndarray] = {}
        if "coefficients" in wanted:
            g = np.zeros(e.shape, complex)
            for c, part in zip(ct, parts.parts, strict=True):
                if c:
                    g += 2 * c * parts.unit * part.apply(np.conj(e))
            out["coefficients"] = g
        # the field's and the model's nodes are one position (the host
        # refuses others): the gradient goes to the first wanted
        nodes = [n for n in wanted if n == "nodes" or n.startswith("model.mesh")]
        kinds = {
            self.roles[n].kind: n
            for n in wanted
            if n.startswith("model") and self.roles[n].kind and n not in nodes
        }
        if nodes:
            kinds[("nodes",)] = nodes[0]
        if kinds:
            names = [f"energy {r}" for r in self.regions]
            # a law region's weight 1 has no material derivative: its term
            # moves with the nodes only
            terms = [
                sens.energy_term(n, p.coefficient, p.elements)
                for n, p in zip(names, parts.parts, strict=True)
            ]
            pair = sens.FieldPair(
                space=parts.op.space,
                left=np.conj(e)[:, None],
                right=e[:, None],
                weights={n: parts.unit * c for n, c in zip(names, ct, strict=True)},
                order=parts.op.order,
                domain=engine("domain").em_region(model, parts.op.built),
            )
            got = sens.sensitivity(
                model, [pair], set(kinds), form="driven", terms=lambda _: terms
            )
            for kind, name in kinds.items():
                g = np.real(np.asarray(got[kind]))
                leaf = np.asarray(values[name])
                if not leaf.ndim:
                    g = g.sum()
                elif g.ndim > leaf.ndim:  # a uniform leaf of principal values
                    g = g.sum(axis=0)
                out[name] = g.reshape(leaf.shape)
        for name in wanted:
            out.setdefault(name, np.zeros(np.shape(values[name])))
        return out
