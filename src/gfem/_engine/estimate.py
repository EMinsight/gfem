"""Flux recovery: the field estimator's error indicators (ADR 0087).

A solved field's flux (eps grad V in electrostatics; eps E, and curl E, for
driven and eigenmode fields) is projected into a continuous recovery space of
degree p - 1 by a global L2 projection, and the element-wise distance between
the recovered and the discrete flux is the field's error indicator:

    eta_K^2 = || R(eps grad V) - eps grad V ||^2_{eps^-1, K}

The recovery space is NGSolve's ``HDiv(order=p-1)`` (all of P_{p-1}, normal
components continuous, so a jump of eps across a material interface is not
smeared), ``definedon`` the solve domain and compressed. Its mass matrix is
factorized once per mesh (`FluxRecovery`) and the factor reused for every
field. NGSolve's local ``GridFunction.Set`` is never used: it does not
converge at a corner singularity.

Each field's indicators are divided by its energy norm squared; `combine`
merges the normalized indicators of several fields. Element arrays are in
gfem tetrahedron order.
"""

from __future__ import annotations

from collections.abc import Sequence

import ngsolve as ngs
import numpy as np

from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import coefficient, element_tensors
from gfem._engine.domain import em_region
from gfem._engine.fields import fe_space
from gfem._engine.runtime import task_manager
from gfem._field import Field
from gfem.model import Model
from gfem.results import ElectrostaticResult

__all__ = ["FluxRecovery", "combine", "electrostatic"]


class FluxRecovery:
    """Recovery into ``HDiv(order - 1)`` on one model's solve domain.

    Builds the space and factorizes its mass matrix once; `squared_error`
    then costs one right-hand side, one back substitution and two element
    passes per field.
    """

    def __init__(self, model: Model, order: int) -> None:
        """Build the recovery space of an order-``order`` solution on ``model``."""
        self.model = model
        self.order = order
        self.built = discretization(model.mesh).built
        self.region = em_region(model, self.built)
        self.dx = quadrature.dx(order, definedon=self.region)
        space = ngs.HDiv(
            self.built.mesh, order=max(order - 1, 0), definedon=self.region
        )
        self.space = ngs.Compress(space)
        u, v = self.space.TnT()
        mass = ngs.BilinearForm(self.space, symmetric=True)
        mass += u * v * self.dx
        mass.Assemble()
        self._mass = mass
        self._inverse = mass.mat.Inverse(
            self.space.FreeDofs(), inverse="sparsecholesky"
        )
        eps = element_tensors(model, "eps_r").real
        inverse = 1.0 / eps if eps.ndim == 1 else np.linalg.inv(eps)
        self.eps = coefficient(self.built, eps)
        self.eps_inverse = coefficient(self.built, inverse)
        self._p0 = ngs.L2(self.built.mesh, order=0)

    def recover(self, flux: ngs.CoefficientFunction) -> ngs.GridFunction:
        """The global L2 projection of ``flux`` into the recovery space."""
        rhs = ngs.LinearForm(self.space)
        rhs += flux * self.space.TestFunction() * self.dx
        rhs.Assemble()
        recovered = ngs.GridFunction(self.space)
        recovered.vec.data = self._inverse * rhs.vec
        return recovered

    def element_integrals(self, density: ngs.CoefficientFunction) -> np.ndarray:
        """``density`` integrated over each tetrahedron of the solve domain."""
        form = ngs.LinearForm(self._p0)
        form += density * self._p0.TestFunction() * self.dx
        form.Assemble()
        out = np.zeros(len(self.model.mesh.tetrahedra))
        out[self.built.tetrahedron_order] = form.vec.FV().NumPy()
        return out

    def flux_error(self, flux: ngs.CoefficientFunction) -> np.ndarray:
        """``|| R(flux) - flux ||^2_{eps^-1, K}`` per tetrahedron."""
        diff = self.recover(flux) - flux
        return self.element_integrals(diff * (self.eps_inverse * diff))


def _grid_function(field: Field, real_part: bool = True) -> ngs.GridFunction:
    """A real NGSolve view of a field (its real or imaginary part)."""
    excluded = field._excluded  # noqa: SLF001
    _, space, _ = fe_space(
        field.mesh, field.support, excluded, field.space, field.order
    )
    values = np.asarray(field.coefficients)
    gf = ngs.GridFunction(space)
    gf.vec.FV().NumPy()[:] = values.real if real_part else values.imag
    return gf


def electrostatic(
    model: Model, result: ElectrostaticResult, threads: int | None = None
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each terminal's potential.

    One solve (the whole capacitance matrix), holding one field per
    terminal: eta_K^2 / || grad V ||^2_eps per tetrahedron.
    """
    fields = list(result.potential.values())
    with task_manager(threads):
        recovery = FluxRecovery(model, fields[0].order)
        group = []
        for field in fields:
            gf = _grid_function(field)
            flux = recovery.eps * ngs.grad(gf)
            energy = recovery.element_integrals(ngs.grad(gf) * flux).sum()
            group.append(recovery.flux_error(flux) / energy)
    return [group]


def combine(
    solves: Sequence[Sequence[np.ndarray]],
) -> tuple[tuple[float, ...], np.ndarray]:
    """Per-solve estimates and per-tetrahedron indicators from normalized eta^2.

    ``solves`` holds, per solve (a frequency, a mode, or the one electrostatic
    solve), the normalized squared indicators of each of its fields. A
    solve's estimate is sqrt(sum over its tetrahedra and fields / its field
    count); a tetrahedron's indicator is the root mean square over the solves
    of its indicator in each (itself the root mean square over the solve's
    fields).
    """
    per_solve = [np.mean(np.asarray(fields), axis=0) for fields in solves]
    values = tuple(float(np.sqrt(eta2.sum())) for eta2 in per_solve)
    indicators = np.sqrt(np.mean(per_solve, axis=0))
    return values, indicators
