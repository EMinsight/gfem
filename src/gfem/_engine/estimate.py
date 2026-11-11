"""Flux recovery: the field estimator's error indicators (ADR 0087).

A solved field's flux (eps grad V in electrostatics; eps E, and curl E, for
driven and eigenmode fields) is projected into a continuous recovery space of
degree p - 1 by a global L2 projection, and the element-wise distance between
the recovered and the discrete flux is the field's error indicator:

    eta_K^2 = || R(eps grad V) - eps grad V ||^2_{eps^-1, K}

    eta_K^2 = || R(mu^-1 curl E) - mu^-1 curl E ||^2_{mu, K}
              + k^2 || R(eps E) - eps E ||^2_{eps^-1, K}

with k = Re(omega) / c in the mesh's length unit. eps E goes into NGSolve's
``HDiv(order=p-1)`` (all of P_{p-1}, normal components continuous, so a jump
of eps across a material interface is not smeared); mu^-1 curl E into
``HCurl(order=p-1, type1=False)`` (all of P_{p-1}; gfem's own ``type1``
kind is not, and its recovery error does not converge). Both are
``definedon`` the solve domain and compressed, and cut along interior PEC
sheets, across which the solved flux jumps. Each mass matrix is factorized
once per mesh (`FluxRecovery`) and the factor reused for every field, real
and imaginary part. NGSolve's local ``GridFunction.Set`` is never used: it
does not converge at a corner singularity.

Each field's indicators are divided by its energy norm squared; `combine`
merges the normalized indicators of several fields. Element arrays are in
gfem tetrahedron order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import coefficient, element_tensors
from gfem._engine.domain import em_region, pec_surfaces
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._field import Field
from gfem.model import Model
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult

__all__ = ["FluxRecovery", "combine", "driven", "eigenmode", "electrostatic"]


class _Projection:
    """Global L2 projection into one recovery space of a model's solve domain.

    The space is ``make(**flags)`` compressed. Its mass matrix is factorized
    once (sparse Cholesky) and the factor reused for every projection. With
    ``cracks`` (NGSolve face numbers of interior PEC sheets) the space is cut
    along them: each side of a sheet face, and each group of the elements
    around a sheet edge that no face off the sheet joins, gets its own copy
    of the face's or edge's DOFs, so the recovered field may jump across the
    sheet as the solved one does. That space is reached through the broken
    (element-wise) space, and solved with SciPy.
    """

    def __init__(
        self,
        make: Callable[..., ngs.FESpace],
        dx: ngs.comp.DifferentialSymbol,
        cracks: frozenset[int],
        region: ngs.Region,
    ) -> None:
        self._dx = dx
        conforming = ngs.Compress(make())
        if not cracks:
            self.space = conforming
            u, v = conforming.TnT()
            mass = ngs.BilinearForm(conforming, symmetric=True)
            mass += u * v * dx
            mass.Assemble()
            self._mass = mass
            self._inverse = mass.mat.Inverse(
                conforming.FreeDofs(), inverse="sparsecholesky"
            )
            self._cut = None
            return
        self.space = ngs.Compress(make(discontinuous=True))
        self._cut = _cut_space(conforming, self.space, cracks, region)
        u, v = self.space.TnT()
        mass = ngs.BilinearForm(self.space, symmetric=True)
        mass += u * v * dx
        mass.Assemble()
        rows, cols, values = mass.mat.COO()
        broken = sp.csr_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
            shape=(self.space.ndof, self.space.ndof),
        )
        self._solve = spla.factorized((self._cut.T @ broken @ self._cut).tocsc())

    def project(self, flux: ngs.CoefficientFunction) -> ngs.GridFunction:
        """The L2 projection of ``flux``."""
        rhs = ngs.LinearForm(self.space)
        rhs += flux * self.space.TestFunction() * self._dx
        rhs.Assemble()
        recovered = ngs.GridFunction(self.space)
        if self._cut is None:
            recovered.vec.data = self._inverse * rhs.vec
        else:
            b = self._cut.T @ rhs.vec.FV().NumPy()
            recovered.vec.FV().NumPy()[:] = self._cut @ self._solve(b)
        return recovered


def _cut_space(
    conforming: ngs.FESpace,
    broken: ngs.FESpace,
    cracks: frozenset[int],
    region: ngs.Region,
) -> sp.csr_matrix:
    """``broken`` DOFs from the DOFs of ``conforming`` cut along ``cracks``.

    Both spaces are the same element on the same domain, one continuous and
    one element-wise, so an element's local DOFs correspond one to one.
    """
    mesh = conforming.mesh
    inside = region.Mask()

    def elements(node: ngs.NodeId) -> list[int]:
        return [e.nr for e in mesh[node].elements if inside[mesh[e].index]]

    side: dict[tuple[int, int], int] = {}  # (conforming DOF, element) -> copy
    copies = 0
    nodes: list[tuple[ngs.NodeId, list[list[int]]]] = []
    edges = set()
    for f in cracks:
        face = ngs.NodeId(ngs.FACE, f)
        nodes.append((face, [[e] for e in elements(face)]))
        edges.update(v.nr for v in mesh[face].edges)
    for nr in sorted(edges):
        edge = ngs.NodeId(ngs.EDGE, nr)
        joins = [
            pair
            for face in mesh[edge].faces
            if face.nr not in cracks and len(pair := elements(face)) == 2
        ]
        nodes.append((edge, _connected(elements(edge), joins)))
    for node, parts in nodes:
        dofs = [d for d in conforming.GetDofNrs(node) if d >= 0]
        for part in parts[1:]:
            for k, d in enumerate(dofs):
                for e in part:
                    side[d, e] = conforming.ndof + copies + k
            copies += len(dofs)
    rows, cols = [], []
    for el in mesh.Elements(ngs.VOL):
        for c, b in zip(conforming.GetDofNrs(el), broken.GetDofNrs(el), strict=True):
            if b >= 0:
                rows.append(b)
                cols.append(side.get((c, el.nr), c))
    return sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(broken.ndof, conforming.ndof + copies),
    )


def _connected(items: list[int], joins: list[list[int]]) -> list[list[int]]:
    """``items`` grouped into the connected parts of the pairs ``joins``."""
    group = {e: e for e in items}

    def root(e: int) -> int:
        while group[e] != e:
            e = group[e]
        return e

    for a, b in joins:
        group[root(a)] = root(b)
    parts: dict[int, list[int]] = {}
    for e in items:
        parts.setdefault(root(e), []).append(e)
    return list(parts.values())


class FluxRecovery:
    """Recovery of a model's solved fields on its solve domain.

    eps E (or eps grad V) goes into ``HDiv(order - 1)``; mu^-1 curl E into
    ``HCurl(order - 1, type1=False)``, built on first use. Each space is
    built and its mass matrix factorized once; `flux_error` and
    `curl_error` then cost one right-hand side, one back substitution and
    two element passes per field. With ``cracks`` the spaces are cut along
    the model's interior PEC sheets.
    """

    def __init__(self, model: Model, order: int, *, cracks: bool = False) -> None:
        """Build the recovery of an order-``order`` solution on ``model``."""
        self.model = model
        self.order = order
        self.built = discretization(model.mesh).built
        self.region = em_region(model, self.built)
        self.dx = quadrature.dx(order, definedon=self.region)
        self._cracks = _sheet_faces(model, self.built) if cracks else frozenset()
        self._hdiv = self._projection(ngs.HDiv)
        self._hcurl: _Projection | None = None
        self.eps, self.eps_inverse = self._weights("eps_r")
        self.mu, self.mu_inverse = self._weights("mu_r")
        self._p0 = ngs.L2(self.built.mesh, order=0)

    def _projection(
        self, kind: Callable[..., ngs.FESpace], **flags: Any
    ) -> _Projection:
        mesh, order, region = self.built.mesh, max(self.order - 1, 0), self.region

        def make(**more: Any) -> ngs.FESpace:
            return kind(mesh, order=order, definedon=region, **flags, **more)

        return _Projection(make, self.dx, self._cracks, region)

    def _weights(
        self, name: str
    ) -> tuple[ngs.CoefficientFunction, ngs.CoefficientFunction]:
        """A real material property and its inverse, scalar or tensor, per element."""
        values = element_tensors(self.model, name).real
        inverse = 1.0 / values if values.ndim == 1 else np.linalg.inv(values)
        return coefficient(self.built, values), coefficient(self.built, inverse)

    def recover(self, flux: ngs.CoefficientFunction) -> ngs.GridFunction:
        """The global L2 projection of ``flux`` into ``HDiv(order - 1)``."""
        return self._hdiv.project(flux)

    def element_integrals(self, density: ngs.CoefficientFunction) -> np.ndarray:
        """``density`` integrated over each tetrahedron of the solve domain."""
        form = ngs.LinearForm(self._p0)
        form += density * self._p0.TestFunction() * self.dx
        form.Assemble()
        out = np.zeros(len(self.model.mesh.tetrahedra))
        out[self.built.tetrahedron_order] = form.vec.FV().NumPy()
        return out

    def flux_residual(self, flux: ngs.CoefficientFunction) -> ngs.CoefficientFunction:
        """The recovered error of a flux, ``R(flux) - flux`` (eps E, eps grad V)."""
        return self.recover(flux) - flux

    def curl_residual(self, curl: ngs.CoefficientFunction) -> ngs.CoefficientFunction:
        """The recovered error of mu^-1 curl E, ``R(mu^-1 curl) - mu^-1 curl``."""
        if self._hcurl is None:
            self._hcurl = self._projection(ngs.HCurl, type1=False)
        field = self.mu_inverse * curl
        return self._hcurl.project(field) - field

    def flux_error(self, flux: ngs.CoefficientFunction) -> np.ndarray:
        """``|| R(flux) - flux ||^2_{eps^-1, K}`` per tetrahedron."""
        diff = self.flux_residual(flux)
        return self.element_integrals(diff * (self.eps_inverse * diff))

    def curl_error(self, curl: ngs.CoefficientFunction) -> np.ndarray:
        """``|| R(mu^-1 curl) - mu^-1 curl ||^2_{mu, K}`` per tetrahedron."""
        diff = self.curl_residual(curl)
        return self.element_integrals(diff * (self.mu * diff))


def _sheet_faces(model: Model, built: EngineMesh) -> frozenset[int]:
    """NGSolve face numbers of the model's PEC surfaces inside the solve domain."""
    domain = model._domain  # noqa: SLF001
    names = {name for name in pec_surfaces(model) if domain.interior(name)}
    if not names:
        return frozenset()
    return frozenset(
        el.faces[0].nr for el in built.mesh.Elements(ngs.BND) if el.mat in names
    )


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


def driven(
    model: Model, result: DrivenResult, threads: int | None = None
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each kept field, grouped by frequency.

    One group per frequency, holding one field per excited port, estimated
    one at a time.
    """
    groups: dict[float, list[np.ndarray]] = {}
    with task_manager(threads):
        recovery = None
        for (frequency, _), field in zip(result.kept, result.e_fields, strict=True):
            if recovery is None:
                recovery = FluxRecovery(model, field.order, cracks=True)
            groups.setdefault(frequency, []).append(_wave(recovery, field))
    return list(groups.values())


def eigenmode(
    model: Model, result: EigenmodeResult, threads: int | None = None
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each mode, one group per mode."""
    fields = list(result.e_fields)
    with task_manager(threads):
        recovery = FluxRecovery(model, fields[0].order, cracks=True)
        return [[_wave(recovery, field)] for field in fields]


def _wave(recovery: FluxRecovery, field: Field) -> np.ndarray:
    """eta_K^2 / energy of one time-harmonic electric field.

    eta_K^2 = || R(mu^-1 curl E) - mu^-1 curl E ||^2_{mu,K}
    + k^2 || R(eps E) - eps E ||^2_{eps^-1,K}, with k = Re(omega) / c in the
    mesh's length unit, summed over the real and imaginary parts; the energy
    is || curl E ||^2_{mu^-1} + k^2 || E ||^2_eps over the solve domain.
    """
    omega = float(np.real(np.asarray(field.omega)))
    k = omega / scipy.constants.c * recovery.built.scale
    parts = [True, False] if field.is_complex else [True]
    eta2 = np.zeros(len(recovery.model.mesh.tetrahedra))
    energy = 0.0
    for real_part in parts:
        gf = _grid_function(field, real_part)
        curl = ngs.curl(gf)
        flux = recovery.eps * gf
        eta2 += recovery.curl_error(curl) + k**2 * recovery.flux_error(flux)
        energy += recovery.element_integrals(
            curl * (recovery.mu_inverse * curl) + k**2 * (gf * flux)
        ).sum()
    return eta2 / energy


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
