"""`gfem.optical_force` on a field or model that may hold JAX tracers.

The force and torque are a real quadratic form of the field coefficients
``e`` on the stress shell of the body (`gfem._engine.optical_force`).

Here the host kernel runs on concrete fields and models only: a field or
model computed inside `jax.grad`, `jax.jit` or `jax.vmap` has no gradient
yet and raises, so the measurement's public signature is already final.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from gfem._checks import is_traced
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem._engine.sides import SideSpec
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["evaluate"]


def evaluate(
    field: Field, body: SideSpec, model: Model
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The force, the torque about the centroid and the centroid of a body.

    Raises:
        NonDifferentiableError: The field or the model is traced.
    """
    if is_traced(field) or is_traced(model):
        raise NonDifferentiableError(
            "optical_force has no gradient yet: it was called on a field or "
            "model computed inside jax.grad, jax.jit or jax.vmap. Call it on "
            "the concrete field of a finished solve."
        )
    engine = importlib.import_module("gfem._engine.optical_force")
    force, torque, centroid = engine.force_torque(field, body, model)
    return tuple(jnp.asarray(a, dtype=float) for a in (force, torque, centroid))  # ty: ignore[invalid-return-type]
