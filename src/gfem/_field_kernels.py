"""What the host kernels of field measurements share under `gfem._adjoint.measure`.

`gfem.power_flow` (`gfem._power_flow`), the loss matrices of
`gfem.dissipated_power` and the electromagnetic heat (`gfem._heat`) and
`gfem.heat_flow` (`gfem._heat_flow`) each name their inputs, give each one
a `Role` (read or unread, and why a read one has no gradient), and rebuild
the field and the model from the named host values. This module holds that
bookkeeping once:

- `Role` and `RoleKernel`, the ``role`` / ``unsupported`` / ``describe``
  trio the measure wrapper calls;
- `field_inputs` and `model_inputs`, the loops that flatten a field or a
  model into named inputs with their roles, and `rebuild`, their inverse;
- `field_role`, the roles of a driven field's coefficients, node
  coordinates and ``omega``;
- `design_key`, the bytes of a design state that a kernel caches on.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import jax
import numpy as np

from gfem._adjoint.solve_vjp import input_kind
from gfem._probe import NODES_REASON

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "FIELD_INPUTS",
    "Role",
    "RoleKernel",
    "design_key",
    "engine",
    "field_inputs",
    "field_role",
    "model_inputs",
    "rebuild",
    "reduced_reason",
]

#: A field leaf's input name, by its attribute.
FIELD_INPUTS = {"coefficients": "coefficients", "mesh": "nodes", "omega": "omega"}


class Role(NamedTuple):
    """What a measurement does with one input."""

    #: ``"read"`` or ``"unread"`` (an exact zero).
    role: str
    #: Why a read input is not differentiable; ``None`` if it is.
    unsupported: str | None
    #: The input, as the user knows it.
    description: str
    #: The input kind (`gfem._adjoint.solve_vjp.input_kind`) of a model leaf
    #: whose explicit term the kernel computes, else ``None``.
    kind: tuple | None = None


class RoleKernel:
    """The measure wrapper's view of a kernel's inputs, from their `Role`.

    A subclass sets ``roles`` (input name -> `Role`) and gives ``name``,
    ``output_specs``, ``forward`` and ``backward``.
    """

    pieces: ClassVar[frozenset[str]] = frozenset()
    comm = None
    roles: dict[str, Role]

    def role(self, name: str) -> str:
        return self.roles[name].role

    def unsupported(self, name: str) -> str | None:
        return self.roles[name].unsupported

    def describe(self, name: str) -> str:
        return self.roles[name].description


def engine(name: str) -> Any:
    """The NGSolve module ``gfem._engine.<name>``, imported on first use."""
    return importlib.import_module(f"gfem._engine.{name}")


def reduced_reason(quantity: str) -> str:
    """Why a field of an adaptive sweep has no gradient of ``quantity``."""
    return (
        "the field is formed from an adaptive sweep's frozen reduced basis, "
        f"which holds no adjoint of the {quantity}, so the frozen-basis "
        'gradient is not the solve\'s: solve with sweep="uniform" at the '
        f"frequency whose {quantity} you differentiate"
    )


#: Why a field's frequency has no gradient.
FREQUENCY_REASON = (
    "a field's angular frequency is fixed: differentiate the solve's "
    "frequency through the kept field instead"
)


def field_role(
    name: str,
    field: Field,
    *,
    omega: str = FREQUENCY_REASON,
    reduced: str,
    nodes: str | None = NODES_REASON,
) -> Role:
    """The role of a driven field's input ``name`` (a `FIELD_INPUTS` value).

    ``omega`` is why the field's frequency has no gradient, ``reduced`` why
    an adaptive sweep's field has none (`reduced_reason`), ``nodes`` why its
    node coordinates have none (``None`` for a measurement that
    differentiates them).
    """
    if name == "nodes":
        return Role("read", nodes, "the mesh node coordinates (field.mesh.nodes)")
    if name == "omega":
        return Role("read", omega, "the field's omega")
    if field._reduced:  # noqa: SLF001
        return Role(
            "read",
            reduced,
            "the coefficients of a field of an adaptive sweep (gfem.ReducedFields)",
        )
    return Role("read", None, "the field coefficients")


def field_inputs(
    field: Field,
    inputs: dict[str, Any],
    roles: dict[str, Role],
    role_of: Callable[[str], Role],
    *,
    suffix: str = "",
) -> tuple[Any, tuple[str, ...]]:
    """Add ``field``'s leaves to ``inputs`` and ``roles``.

    Each leaf is named by `FIELD_INPUTS` plus ``suffix`` and its role is
    ``role_of(base name)``.

    Returns:
        The field's tree definition and its input names, for `rebuild`.
    """
    paths, definition = jax.tree_util.tree_flatten_with_path(field)
    names = []
    for path, leaf in paths:
        base = FIELD_INPUTS[path[0].name]
        name = base + suffix
        inputs[name] = leaf
        roles[name] = role_of(base)
        names.append(name)
    return definition, tuple(names)


def model_inputs(
    model: Model,
    inputs: dict[str, Any],
    roles: dict[str, Role],
    role_of: Callable[[tuple], Role],
) -> tuple[tuple[Any, tuple[str, ...]], dict[str, tuple]]:
    """Add ``model``'s leaves to ``inputs`` and ``roles``, by input kind.

    Each leaf is named ``"model"`` plus its key path and its role is
    ``role_of(kind)`` for its `gfem._adjoint.solve_vjp.input_kind`.

    Returns:
        The model's tree definition and input names, for `rebuild`, and
        each input's kind by name.
    """
    paths, definition = jax.tree_util.tree_flatten_with_path(model)
    kinds: dict[str, tuple] = {}
    for path, leaf in paths:
        name = "model" + jax.tree_util.keystr(tuple(path))
        kind = input_kind(path, "model")
        inputs[name] = leaf
        roles[name] = role_of(kind)
        kinds[name] = kind
    return (definition, tuple(kinds)), kinds


def rebuild(definition: tuple[Any, tuple[str, ...]], values: dict[str, Any]) -> Any:
    """The field or model of a (tree definition, input names) pair at ``values``."""
    tree, names = definition
    return jax.tree.unflatten(tree, [values[n] for n in names])


def design_key(values: dict[str, Any], names: Iterable[str]) -> bytes:
    """The bytes of the named inputs: equal keys, equal design states."""
    return b"".join(np.ascontiguousarray(values[n]).tobytes() for n in names)
