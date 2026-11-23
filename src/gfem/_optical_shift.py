"""Section, mode and sampling helpers shared by the optical shifts of a wave port.

A first-order shift of a port mode's effective index reads the mode once on
the port's section (the triangles of the port's surface and the
tetrahedra of the optical model behind them), weighs each triangle by the
mode at 1 W, and samples a perturbing field on the tetrahedra behind it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._materials import Material, MaterialField, PermittivityLaw

if TYPE_CHECKING:
    from gfem.mesh import Mesh
    from gfem.model import Model


def _section(model: Model, port: Any) -> tuple[np.ndarray, np.ndarray]:
    """The gfem triangles of a wave port's section and the tetrahedra behind them.

    The triangles of the port's surface with the electromagnetic solve
    domain on exactly one side, as the port mode problem takes them.
    """
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    tag = mesh.region(port.surface).tag
    nb = np.asarray(mesh.triangle_neighbours)
    faces = np.flatnonzero(np.asarray(mesh.triangle_tags) == tag)
    side = (nb[faces] >= 0) & active[np.maximum(nb[faces], 0)]
    one = side.sum(axis=1) == 1
    faces, side = faces[one], side[one]
    tets = np.where(side[:, 0], nb[faces, 0], nb[faces, 1])
    return faces, tets


def _region_rows(mesh: Mesh, tets: np.ndarray) -> dict[str, np.ndarray]:
    """The positions in ``tets`` of each volume region's tetrahedra."""
    names = mesh._region_names()  # noqa: SLF001
    tags = np.asarray(mesh.tetrahedron_tags)[tets]
    return {names[int(t)]: np.flatnonzero(tags == t) for t in np.unique(tags)}


def _values(value: Any, pos: np.ndarray) -> Any:
    """A material value at element positions ``pos`` of its region."""
    if is_traced(value):
        arr = jnp.asarray(value)
        return jnp.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]
    arr = np.asarray(value)
    return np.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]


def _field_values(material: MaterialField, name: str, pos: np.ndarray) -> Any:
    """A material field's property on elements ``pos`` (``None``: unset or traced).

    A region-wide value (one value, or three principal values) stays whole.
    """
    value = getattr(material, name)
    if value is None or is_traced(value):
        return None
    arr = np.asarray(value)
    return arr if arr.ndim == material.has_principal_values(name) else arr[pos]


def _lossy(material: Material | MaterialField, pos: np.ndarray) -> str | None:
    """Which loss a region's material has on elements ``pos``, or ``None``."""
    if isinstance(material, Material):
        if isinstance(material.eps_r, PermittivityLaw):
            return "a permittivity law"
        eps = np.asarray(material.eps_r, complex)
        tan, sigma = material.tan_delta, material.conductivity
    else:
        if material.blend_ends is not None:
            return "a permittivity law (a density blend's end)"
        eps, tan, sigma = (
            _field_values(material, name, pos)
            for name in ("eps_r", "tan_delta", "conductivity")
        )
    if eps is not None and np.any(np.asarray(eps).imag != 0):
        return "a complex eps_r"
    if tan is not None and np.any(np.asarray(tan) != 0):
        return "a nonzero tan_delta"
    if sigma is not None and np.any(np.asarray(sigma) != 0):
        return "a nonzero conductivity"
    return None


@dataclasses.dataclass(frozen=True, eq=False)
class _Sampling:
    """Where a shift trace samples a field: tetrahedra of the optical mesh."""

    mesh: Any
    tetrahedra: np.ndarray
    port: str


def _frequency(frequency: Any, owner: str) -> float:
    if np.ndim(frequency) != 0:
        raise ValueError(
            f"{owner} takes one frequency, got "
            f"{np.size(frequency)} values: call it once per frequency in a loop."
        )
    if is_traced(frequency):
        raise TypeError(
            f"{owner}: the frequency is a fixed number, not a differentiable input."
        )
    f = float(frequency)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{owner}: frequency must be > 0 Hz, got {f}.")
    return f


def _wave_port(model: Model, port: str) -> Any:
    from gfem._sweep.port_mode import _wave_port as find

    return find(model, port)


def _check_model(model: Any, owner: str) -> None:
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(f"{owner} needs a gfem.Model, got {type(model).__name__}.")


def _mode_weights(
    model: Model,
    port: str,
    frequency: float,
    mode: int | None,
    faces: np.ndarray,
    *,
    tensor: bool = False,
    rule: type | None = None,
) -> tuple[int, jax.Array, jax.Array, float]:
    r"""The port's mode on ``model`` and its first-order weight per section triangle.

    Args:
        model: The optical model.
        port: The wave port's name.
        frequency: The frequency, Hz.
        mode: The mode number, or ``None`` for the port's.
        faces: The gfem triangles of the port's section, the order of the
            weights.
        tensor: The real symmetric 3x3 weight
            :math:`W_t = \tfrac{c\varepsilon_0}{4}\mathrm{Re}\int_t E_0E_0^H\,dA`
            (global frame) instead of the scalar one; its trace is the
            scalar weight over :math:`2 n_{0,t}`.
        rule: The weights rule (a `ShiftWeightsRule` whose refusals name
            the shift), ``None`` for the thermo-optic one.

    Returns:
        The mode number, ``n_eff`` (a constant), the weights (``(n_tri,)``
        :math:`w_t`, with
        :math:`\delta n_\mathrm{eff} = \sum_t w_t (dn/dT)_t \Delta T_t`, or
        ``(n_tri, 3, 3)`` :math:`W_t` with ``tensor``, in the order of
        ``faces``) and ``k0`` (1/m).
    """
    from gfem._adjoint import solve_vjp
    from gfem._engine.thermo_optic import ShiftWeightsRule
    from gfem.options import SolverOptions
    from gfem.problems import PortMode

    problem = PortMode(port, frequency, mode=mode)
    made = (rule or ShiftWeightsRule)(
        model, problem, SolverOptions(), faces, tensor=tensor
    )
    if is_traced(model):
        outputs = solve_vjp(made, model, problem)
    else:
        outputs, _ = made.forward(model, keep=False)
    k0 = 2 * math.pi * frequency / scipy.constants.c
    gamma = jnp.asarray(outputs["gamma"])[0]
    n_eff = jnp.real(gamma / (1j * k0))
    return made.mode, n_eff, jnp.asarray(outputs["weights"]), k0
