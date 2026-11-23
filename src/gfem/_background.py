r"""The background stack: the planar layers an incident field solves.

Declared on the model (``Model(background=...)``): every source of a problem
and every measurement that needs the exterior medium read the same stack.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from gfem._checks import is_traced, static_float
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem.errors import NonDifferentiableError

__all__ = ["Background"]

_FIXED = "Build the background from concrete values: it is a constant of the solve."


def _layer(value: Any, k: int) -> Material:
    head = f"Background layer {k}"
    if isinstance(value, MaterialField):
        raise NotImplementedError(
            f"{head} is a MaterialField; the background stack's layers are "
            "constant. Give it a Material; a varying material belongs to the "
            "scatterer, in the model's materials."
        )
    if not isinstance(value, Material):
        raise TypeError(
            f"Background layers are gfem.Material values, from bottom to top; "
            f"{head.lower()} is a {type(value).__name__}."
        )
    if isinstance(value.eps_r, PermittivityLaw):
        raise NotImplementedError(
            f"{head} has a permittivity law; the background stack's layers are "
            "constant. Give it the constant permittivity of the band."
        )
    if not value.isotropic:
        raise NotImplementedError(
            f"{head} is anisotropic; the background stack's layers are "
            "isotropic. An anisotropic material belongs to the scatterer, in "
            "the model's materials."
        )
    return value


@dataclasses.dataclass(frozen=True, init=False)
class Background:
    r"""The planar layers, constant and isotropic, whose solution is the incident field.

    The layers are normal to :math:`z`, from bottom (:math:`-z`) to top
    (:math:`+z`), separated by the planes ``interfaces``; the bottom and top
    layers extend to infinity, unless a PEC ground plane at
    :math:`z = z_g` ends the stack below. One `Material` and no interfaces
    is a homogeneous background. ``Model(background=material)`` is
    shorthand for ``Model(background=gfem.Background(material))``.

    An incident field (`PlaneWave`, `GaussianBeam`, `IncidentField`) is the
    field of its source in this stack alone; where the model's material
    differs from the stack at the same height is the scatterer.

    Example:
        Silicon on a 2 µm buried oxide over a silicon substrate, under air::

            stack = gfem.Background(
                [si, oxide, si, air], interfaces=[-2e-6, 0.0, 220e-9]
            )
            model = gfem.Model(mesh, materials=..., pml=..., background=stack)

    Args:
        layers: One `Material`, or a sequence of them from bottom to top:
            constant, isotropic, possibly lossy.
        interfaces: The :math:`z` of the planes between the layers, m,
            ascending, one fewer than the layers.
        ground: The :math:`z` of a PEC ground plane ending the bottom layer,
            m, below the lowest interface; ``None``: the bottom layer extends
            to :math:`z = -\infty`.

    Raises:
        TypeError: A layer is not a `Material`.
        ValueError: No layer, a count of interfaces that is not one fewer
            than the layers, interfaces that are not finite or do not
            ascend, or a ground plane that is not finite or not below them.
        NotImplementedError: A layer is a `MaterialField`, has a
            `PermittivityLaw`, or is anisotropic.
        NonDifferentiableError: A plane is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`; the background is a fixed value.
    """

    layers: tuple[Material, ...]
    interfaces: tuple[float, ...]
    ground: float | None

    def __init__(
        self,
        layers: Material | Sequence[Material],
        *,
        interfaces: Sequence[float] = (),
        ground: float | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        items = (
            tuple(layers)
            if isinstance(layers, Sequence) and not isinstance(layers, str)
            else (layers,)
        )
        if not items:
            raise ValueError("A Background needs at least one layer.")
        items = tuple(_layer(m, k) for k, m in enumerate(items))
        if is_traced(interfaces):
            raise NonDifferentiableError(
                "Background.interfaces is a fixed value and gfem does not "
                f"differentiate with respect to it. {_FIXED}"
            )
        try:
            planes = np.asarray(interfaces, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(
                f"Background.interfaces must be z values in m, got {interfaces!r}."
            ) from None
        n = len(items)
        if len(planes) != n - 1:
            layer = "layer needs" if n == 1 else "layers need"
            plural = "interface" if n == 2 else "interfaces"
            raise ValueError(
                f"{n} {layer} {n - 1} {plural} between them, got "
                f"{len(planes)}: {tuple(planes.tolist())}."
            )
        if not np.isfinite(planes).all():
            raise ValueError(
                f"Background.interfaces must be finite, got {tuple(planes.tolist())}."
            )
        if (np.diff(planes) <= 0).any():
            raise ValueError(
                "Background.interfaces must ascend, from the bottom layer up, got "
                f"{tuple(planes.tolist())}."
            )
        z_g = (
            None
            if ground is None
            else static_float(ground, "Background.ground", _FIXED)
        )
        if z_g is not None and len(planes) and z_g >= planes[0]:
            raise ValueError(
                f"Background.ground at z = {z_g} m must lie below the lowest "
                f"interface, z = {planes[0]} m: it ends the bottom layer."
            )
        object.__setattr__(self, "layers", items)
        object.__setattr__(self, "interfaces", tuple(float(z) for z in planes))
        object.__setattr__(self, "ground", z_g)

    def __repr__(self) -> str:
        layers = (
            repr(self.layers[0]) if len(self.layers) == 1 else repr(list(self.layers))
        )
        interfaces = f", interfaces={self.interfaces!r}" if self.interfaces else ""
        ground = "" if self.ground is None else f", ground={self.ground!r}"
        return f"Background({layers}{interfaces}{ground})"


jax.tree_util.register_static(Background)


def background_of(value: Any) -> Background | None:
    """``Model(background=...)`` as a `Background`, or ``None``."""
    if value is None or isinstance(value, Background):
        return value
    return Background(value)
