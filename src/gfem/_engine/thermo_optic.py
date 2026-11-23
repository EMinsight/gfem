"""The first-order weights of a thermo-optic shift on a wave port's section.

`ShiftWeightsRule` runs the port's mode solve (`gfem.PortMode`) and turns
its mode at 1 W into one weight per triangle of the section::

    w_t = (c eps0 / 4) 2 n0_t int_t |E0|^2 dA

so that ``delta_n_eff = sum_t w_t (dn/dT)_t rise_t``. ``n0_t`` is the
square root of the real permittivity of the tetrahedron behind triangle
``t``, as the port mode problem reads it. With ``tensor`` it gives instead
the real symmetric 3x3 weight per triangle, in the global frame::

    W_t = (c eps0 / 4) Re int_t E0 E0^H dA

whose trace is ``w_t / (2 n0_t)``, for a tensor perturbation
``delta_n_eff = sum_t W_t : delta_eps_t``. The weights are constants of the
mode: no output has a gradient, and every input the port mode
problem reads (the optical model's ``eps_r``, ``tan_delta``, ``mu_r``,
``conductivity``, node coordinates, boundary and port values) raises
`gfem.NonDifferentiableError` under `jax.grad`; the thermal properties and
the thermo-optic coefficient are not read, so they get exact zeros here and
differentiate where the shift multiplies them in JAX.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._engine.solve_request import SolveRequest
from gfem._sweep.port_mode import PortModeRule
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode

__all__ = ["ShiftWeightsRule"]


class ShiftWeightsRule(PortModeRule):
    """The port mode solve of a thermo-optic shift, and its section weights.

    Outputs ``gamma`` (``(1,)``, the mode's propagation constant, 1/m) and
    ``weights`` (``(n_tri,)``, or ``(n_tri, 3, 3)`` with ``tensor``,
    dimensionless, in the order of ``faces``).
    """

    def __init__(
        self,
        model: Model,
        problem: PortMode,
        options: SolverOptions,
        faces: np.ndarray,
        *,
        tensor: bool = False,
    ) -> None:
        """The port mode rule, and the section's mesh triangles in order.

        Args:
            model: The optical model.
            problem: The port mode problem at one frequency.
            options: Solver options (the element order).
            faces: The gfem triangles of the port's section, the order of
                ``weights``.
            tensor: The 3x3 weight per triangle instead of the scalar one.
        """
        super().__init__(model, problem, options)
        self.faces = np.asarray(faces)
        self.tensor = tensor

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """The propagation constant and one weight per section triangle."""
        del model
        shape = (len(self.faces), 3, 3) if self.tensor else (len(self.faces),)
        return {
            "gamma": jax.ShapeDtypeStruct((1,), jnp.complex128),
            "weights": jax.ShapeDtypeStruct(shape, jnp.float64),
        }

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every input the port mode problem reads: the mode is held constant."""
        del kind
        return (
            "a thermo-optic shift holds the port mode constant (first order in "
            "the rise), and its derivative with respect to the optical model's "
            "permittivity, losses, permeability, node coordinates or boundary "
            "values would need the mode's own derivative, which this version "
            "does not have (a frozen mode gives a wrong value). Differentiate "
            "through the rise or the thermo_optic_coefficient"
        )

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the thermo-optic shift's mode output "
            f"{name!r} with respect to the optical model: the mode is a constant "
            "of the shift. Wrap the input in jax.lax.stop_gradient to use its "
            "value without its derivative."
        )

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict, None]:
        """Solve the mode and integrate its energy on each section triangle."""
        outputs, _ = super().forward(model, keep=keep, request=request)
        sweep = outputs["sweep"]
        weigh = section_tensor_weights if self.tensor else section_weights
        weights = weigh(sweep.space, sweep.basis @ sweep.coords[0])
        faces = np.asarray(sweep.space.section.faces)
        order = np.argsort(faces)
        where = order[np.searchsorted(faces, self.faces, sorter=order)]
        if not np.array_equal(faces[where], self.faces):  # pragma: no cover
            raise RuntimeError("the port section's triangles changed")
        return {"gamma": sweep.gamma, "weights": weights[where]}, None


def section_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    """``(c eps0 / 4) 2 n0_t int_t |E|^2 dA`` per section triangle.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_tri,)`` in the section's triangle order (the 2D mesh's element
        order).
    """
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    et, en = gf.components
    density = ngs.Norm(et) ** 2 + ngs.Norm(en) ** 2
    energy = ngs.Integrate(
        density, space.mesh, order=2 * space.order, element_wise=True
    )
    section = space.section
    energy = np.asarray(energy).real * section.scale**2
    n0 = np.sqrt(np.asarray(section.eps_nn).real)
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    return c_eps0 / 4 * 2 * n0 * energy


def section_tensor_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    """``(c eps0 / 4) Re int_t E E^H dA`` per section triangle, global frame.

    Real symmetric: the imaginary part is antisymmetric, so it meets nothing
    in a symmetric permittivity change. Its trace is ``section_weights``
    over ``2 n0_t``.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_tri, 3, 3)`` in the section's triangle order (the 2D mesh's
        element order).
    """
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    et, en = gf.components
    local = [et[0], et[1], en]
    section = space.section
    w = np.empty((len(section.triangles), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            product = ngs.Integrate(
                local[i] * ngs.Conj(local[j]),
                space.mesh,
                order=2 * space.order,
                element_wise=True,
            )
            w[:, i, j] = w[:, j, i] = np.asarray(product).real
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    w *= c_eps0 / 4 * section.scale**2
    frame = np.asarray(section.frame, float)
    w = np.einsum("ai,tab,bj->tij", frame, w, frame)
    return (w + np.swapaxes(w, 1, 2)) / 2
