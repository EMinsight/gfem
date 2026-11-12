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
sheets (and, for a potential, the interior terminal and ground sheets),
across which the solved flux jumps. Both are constrained on the solve
domain's natural surfaces, n . R(eps E) = 0 and n x R(mu^-1 curl E) = 0
(zero trace DOFs on their exterior faces; on a cut space every copy of
one): for a potential every exterior surface not held at a potential, for a
wave field the PMC and default PMC surfaces. Unconstrained, the recovery
there is free to miss the solved field's natural condition, and a tensor eps
then hides the error (ADR 0087). Each mass matrix is factorized
once per mesh (`FluxRecovery`) and the factor reused for every field, real
and imaginary part. NGSolve's local ``GridFunction.Set`` is never used: it
does not converge at a corner singularity.

Each field's indicators are divided by its energy norm squared; `combine`
merges the normalized indicators of several fields. Element arrays are in
gfem tetrahedron order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from gfem.model import PEC, PMC, Conductor, Model
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult

__all__ = [
    "FluxRecovery",
    "capacitance",
    "combine",
    "driven",
    "eigenmode",
    "electrostatic",
    "omega",
    "s_entry",
]


class _Projection:
    """Global L2 projection into one recovery space of a model's solve domain.

    The space is ``make(**flags)`` compressed. Its mass matrix is factorized
    once (sparse Cholesky) and the factor reused for every projection. With
    ``cracks`` (NGSolve face numbers of interior PEC sheets) the space is cut
    along them: each side of a sheet face, and each group of the elements
    around a sheet edge that no face off the sheet joins, gets its own copy
    of the face's or edge's DOFs, so the recovered field may jump across the
    sheet as the solved one does. That space is reached through the broken
    (element-wise) space, and solved with SciPy. The DOFs of the
    ``constrained`` boundary elements (their trace: normal for HDiv,
    tangential for HCurl) are held at zero, and on the cut path every copy of
    one too.
    """

    def __init__(
        self,
        make: Callable[..., ngs.FESpace],
        dx: ngs.comp.DifferentialSymbol,
        cracks: frozenset[int],
        region: ngs.Region,
        constrained: Sequence[ngs.ElementId] = (),
    ) -> None:
        self._dx = dx
        conforming = ngs.Compress(make())
        free = ngs.BitArray(conforming.FreeDofs())
        for el in constrained:
            for d in conforming.GetDofNrs(el):
                if d >= 0:
                    free.Clear(d)
        if not cracks:
            self.space = conforming
            u, v = conforming.TnT()
            mass = ngs.BilinearForm(conforming, symmetric=True)
            mass += u * v * dx
            mass.Assemble()
            self._mass = mass
            self._inverse = mass.mat.Inverse(free, inverse="sparsecholesky")
            self._cut = None
            return
        self.space = ngs.Compress(make(discontinuous=True))
        cut, origin = _cut_space(conforming, self.space, cracks, region)
        kept = np.array([free[d] for d in range(len(free))], dtype=bool)[origin]
        self._cut = cut[:, np.flatnonzero(kept)]
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
) -> tuple[sp.csr_matrix, np.ndarray]:
    """``broken`` DOFs from the DOFs of ``conforming`` cut along ``cracks``.

    Both spaces are the same element on the same domain, one continuous and
    one element-wise, so an element's local DOFs correspond one to one.
    Returns the matrix and, per column, the ``conforming`` DOF it copies.
    """
    mesh = conforming.mesh
    inside = region.Mask()

    def elements(node: ngs.NodeId) -> list[int]:
        return [e.nr for e in mesh[node].elements if inside[mesh[e].index]]

    side: dict[tuple[int, int], int] = {}  # (conforming DOF, element) -> copy
    origin = list(range(conforming.ndof))
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
                    side[d, e] = len(origin) + k
            origin.extend(dofs)
    rows, cols = [], []
    for el in mesh.Elements(ngs.VOL):
        for c, b in zip(conforming.GetDofNrs(el), broken.GetDofNrs(el), strict=True):
            if b >= 0:
                rows.append(b)
                cols.append(side.get((c, el.nr), c))
    cut = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(broken.ndof, len(origin))
    )
    return cut, np.asarray(origin)


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
    the model's interior PEC sheets. Both are constrained on the
    ``natural`` surfaces' exterior faces.

    Attributes:
        natural: The natural surfaces: with ``cracks`` the model's `PMC` and
            default PMC surfaces, otherwise every exterior surface not in
            ``sheets``.
    """

    def __init__(
        self,
        model: Model,
        order: int,
        *,
        cracks: bool = False,
        sheets: Sequence[str] = (),
    ) -> None:
        """Build the recovery of an order-``order`` solution on ``model``.

        ``cracks`` cuts the spaces along the model's interior PEC sheets,
        ``sheets`` along these surfaces too (where they are interior).
        """
        self.model = model
        self.order = order
        self.built = discretization(model.mesh).built
        self.region = em_region(model, self.built)
        self.dx = quadrature.dx(order, definedon=self.region)
        names = set(pec_surfaces(model)) if cracks else set()
        self._cracks = _sheet_faces(model, self.built, names | set(sheets))
        self.natural = _natural_surfaces(model, cracks=cracks, sheets=sheets)
        self._constrained = _exterior_elements(
            self.built, self.region, set(self.natural)
        )
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

        return _Projection(make, self.dx, self._cracks, region, self._constrained)

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


def _natural_surfaces(
    model: Model, *, cracks: bool, sheets: Sequence[str]
) -> tuple[str, ...]:
    """The solve domain's natural surfaces, where the recovery is constrained.

    For a wave field (``cracks``) the model's `PMC` and default PMC surfaces;
    for a potential every exterior surface not held at a potential (not in
    ``sheets``). Sorted names; some may be interior in part.
    """
    domain = model._domain  # noqa: SLF001
    if cracks:
        names = set(model.default_pmc) | {
            b.surface for b in model.boundaries if isinstance(b, PMC)
        }
    else:
        fixed = set(sheets)
        names = {s for s in model.mesh.surfaces if domain.exterior(s)} - fixed
    return tuple(sorted(names))


def _exterior_elements(
    built: EngineMesh, region: ngs.Region, names: set[str]
) -> list[ngs.ElementId]:
    """Boundary elements of the surfaces ``names`` on the exterior of ``region``."""
    if not names:
        return []
    mesh = built.mesh
    inside = region.Mask()
    out = []
    for el in mesh.Elements(ngs.BND):
        if el.mat not in names:
            continue
        face = mesh[el.faces[0]]
        if sum(bool(inside[mesh[e].index]) for e in face.elements) == 1:
            out.append(ngs.ElementId(el))
    return out


def _sheet_faces(model: Model, built: EngineMesh, names: set[str]) -> frozenset[int]:
    """NGSolve face numbers of the surfaces ``names`` inside the solve domain."""
    domain = model._domain  # noqa: SLF001
    names = {name for name in names if domain.interior(name)}
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
        gfs = [_grid_function(field) for field in fields]
        recovery = _potential_recovery(model, result)
        group = []
        for gf in gfs:
            flux = recovery.eps * ngs.grad(gf)
            energy = recovery.element_integrals(ngs.grad(gf) * flux).sum()
            group.append(recovery.flux_error(flux) / energy)
    return [group]


def _potential_recovery(model: Model, result: ElectrostaticResult) -> FluxRecovery:
    """The flux recovery of an electrostatic solve's potentials.

    Cut along the interior surfaces the problem holds at a fixed potential
    (its terminals and ground, and the model's PEC and conductor surfaces),
    across which the solved flux jumps.
    """
    sheets = set(result._fixed) | {  # noqa: SLF001
        b.surface for b in model.boundaries if isinstance(b, (PEC, Conductor))
    }
    order = next(iter(result.potential.values())).order
    return FluxRecovery(model, order, sheets=sorted(sheets))


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


# --- the quantity estimator (ADR 0089 and its amendment) ---------------------
#
# Each quantity has an exact quadratic error identity in the error e = u_h - u
# of its field and of its free dual (the potential itself, the other port's
# forward field, the eigenvector). The quantity estimator puts the recovered
# errors (R(flux) - flux, as for the field estimator) in place of e, element
# by element, volume terms only:
#
#   c_K(e_i, e_j) = int_K (R(mu^-1 curl E_i) - mu^-1 curl E_i) . mu (...)_j
#   m_K(e_i, e_j) = int_K (R(eps E_i) - eps E_i) . eps^-1 (...)_j
#
# complex bilinear (not Hermitian). The reported estimate is the magnitude
# |sum c_K| + k^2 |sum m_K| (the two terms are never let cancel over the
# mesh), scaled to the quantity's units. A tetrahedron's weight is the
# magnitude of its term of the identity, |a_K| = |c_K - k^2 m_K| with the same
# scale (ticket 08's indicator): marking is unchanged by the reported sum, and
# the weights add up to neither that sum nor |sum a_K| in general.


def capacitance(
    model: Model,
    result: ElectrostaticResult,
    i: str,
    j: str,
    threads: int | None = None,
) -> tuple[float, np.ndarray]:
    """The relative error estimate of ``capacitance[i, j]`` and each tetrahedron's.

    C_ij,h - C_ij = a(e_i, e_j) for the potentials of terminals i and j at
    1 V, each the other's free dual. A tetrahedron's term is m_K(i, j), the
    recovered flux errors of the two potentials in the eps^-1 product; the
    estimate is sum_K |m_K(i, j)| over |a_h(u_i, u_j)|, never letting the
    terms of an off-diagonal entry (which have no sign) cancel. At i = j it
    is the square of the field estimate of that terminal's potential, the
    weights `electrostatic` gives it.
    """
    names = list(result.potential)
    fields = list(result.potential.values())
    with task_manager(threads):
        gi = _grid_function(fields[names.index(i)])
        gj = gi if i == j else _grid_function(fields[names.index(j)])
        recovery = _potential_recovery(model, result)
        flux_i = recovery.eps * ngs.grad(gi)
        flux_j = flux_i if i == j else recovery.eps * ngs.grad(gj)
        energy = recovery.element_integrals(ngs.grad(gi) * flux_j).sum()
        ri = recovery.flux_residual(flux_i)
        rj = ri if i == j else recovery.flux_residual(flux_j)
        terms = recovery.element_integrals(ri * (recovery.eps_inverse * rj))
    weights = np.abs(terms) / abs(energy)
    return float(weights.sum()), weights


def s_entry(
    model: Model,
    result: DrivenResult,
    to: str,
    source: str,
    threads: int | None = None,
    *,
    adjoints: Mapping[tuple[float, str], Field] | None = None,
) -> list[tuple[float, np.ndarray]]:
    """The absolute error estimate of ``s[to, source]``, per frequency and tetrahedron.

    With port i = ``to``, j = ``source``, e = u_h - u, and z_i the adjoint
    of port i's S functional f_i (A z_i = f_i, A complex symmetric):
    O_ij,h - O_ij = f_i(e_j) = -a(e_j, z_i,h - z_i) for the raw outgoing
    wave O, and S_ij = (O_ij - delta_ij) d_i d_j with the de-embedding d
    (``exp(gamma offset)``, 1 for a lumped port). a = c - k0^2 m with k0 in
    the mesh's length unit; m recovers eps_eff E (`_effective_eps`). The
    estimate is |d_i d_j| (|sum c_K| + k0^2 |sum m_K|) of the recovered
    errors of x_j and z_i, a tetrahedron's |d_i d_j| |c_K - k0^2 m_K|.

    ``adjoints`` holds z_i per (frequency, port) where port i's functional
    is not parallel to its excitation (a hybrid lossy wave port mode).
    Elsewhere z_i = c_i x_i with |c_i| = 1 / (4 k0 eta0) for fields of 1 W
    incident (ADR 0089): the free dual, x_i itself on that scale.
    """
    fields = dict(zip(result.kept, result.e_fields, strict=True))
    adjoints = adjoints or {}
    d = _deembedding(result)
    eta0 = scipy.constants.mu_0 * scipy.constants.c
    out = []
    with task_manager(threads):
        recovery = None
        for n, frequency in enumerate(result.s.frequency):
            f = float(frequency)
            f_to, f_source = fields[f, to], fields[f, source]
            if recovery is None:
                recovery = FluxRecovery(model, f_to.order, cracks=True)
            k = 2 * np.pi * f / scipy.constants.c * recovery.built.scale
            eps, eps_inverse = _effective_eps(recovery, k)
            ri = _residuals(recovery, f_source, eps)
            adjoint = adjoints.get((f, to))
            if adjoint is not None:
                rj, scale = _residuals(recovery, adjoint, eps), 1.0
            else:
                rj = ri if source == to else _residuals(recovery, f_to, eps)
                scale = 1.0 / (4 * k * eta0)
            curl, mass = _pair(recovery, ri, rj, eps_inverse)
            scale *= abs(d[to][n] * d[source][n])
            value = scale * (abs(curl.sum()) + k**2 * abs(mass.sum()))
            out.append((float(value), scale * np.abs(curl - k**2 * mass)))
    return out


def _deembedding(result: DrivenResult) -> dict[str, np.ndarray]:
    """Each port's de-embedding per frequency: ``exp(gamma offset)``, 1 if lumped."""
    offsets = dict(result.offsets)
    ones = np.ones(len(result.s.frequency), complex)
    modes = result.port_modes
    return {
        name: np.exp(np.asarray(modes[name].gamma) * offsets.get(name, 0.0))
        if name in modes
        else ones
        for name in result.s.ports
    }


def omega(
    model: Model,
    result: EigenmodeResult,
    mode: int,
    threads: int | None = None,
) -> tuple[float, np.ndarray]:
    """The relative error estimate of ``omega[mode]`` and each tetrahedron's.

    A mode of the complex-symmetric T(k) is its own dual, and k_h - k ~=
    -e^T T(k) e / (x_h^T T'(k_h) x_h), k = omega scale / c complex. The
    numerator's volume terms are c_K - k^2 m_K: c_K of the recovered error
    of mu^-1 curl E, m_K = int e_D . eps_eff^-1 e_D of the recovered error
    e_D of eps_eff E, eps_eff = eps_r (1 - i tan_delta) - i eta0 sigma scale
    / k (the total current, whose normal component is continuous); the
    products are complex bilinear. T'(k_h) is the eigen operator's own
    (`gfem._eigen.operator.EigenOperator.derivative`), so it holds the
    damping of lumped resistors, impedance, absorbing and conductor surfaces.
    The estimate is (|sum c_K| + |k^2| |sum m_K|) / (|k| |x_h^T T' x_h|), a
    tetrahedron's |c_K - k^2 m_K| on the same scale. On a lossless mode
    T' = -2k M and this is (|sum c_K| + k^2 |sum m_K|) / (2 k^2 |b(E, E)|).
    """
    field = result.e_fields[mode]
    with task_manager(threads):
        recovery = FluxRecovery(model, field.order, cracks=True)
        k = complex(np.asarray(field.omega)) / scipy.constants.c * recovery.built.scale
        lam = k * k
        eps, eps_inverse = _effective_eps(recovery, k)
        parts = _parts(field)
        flux = _times(eps, parts)
        flux_error = [recovery.flux_residual(f) for f in flux]
        curl_error = [recovery.curl_residual(ngs.curl(gf)) for gf in parts]
        curl = _bilinear(recovery, curl_error, _times([recovery.mu], curl_error))
        mass = _bilinear(recovery, flux_error, _times(eps_inverse, flux_error))
        slope = _slope(model, field, k)
    scale = 1.0 / (abs(k) * abs(slope))
    value = scale * (abs(curl.sum()) + abs(lam) * abs(mass.sum()))
    return float(value), scale * np.abs(curl - lam * mass)


def _effective_eps(
    recovery: FluxRecovery, k: complex
) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
    """eps_eff and its inverse at ``k``, each as its real (and imaginary) part.

    eps_eff = eps_r (1 - i tan_delta) - i eta0 sigma scale / k per
    tetrahedron, a scalar or a 3x3 tensor; real (``recovery.eps``) on a
    lossless model.
    """
    from gfem._engine.coefficients import lossy_eps
    from gfem._engine.terms import ETA0

    model, built = recovery.model, recovery.built
    sigma = element_tensors(model, "conductivity").real
    eps = lossy_eps(model)
    if eps.ndim != sigma.ndim:
        eye = np.eye(3)
        eps, sigma = (
            v if v.ndim == 3 else v[:, None, None] * eye for v in (eps, sigma)
        )
    eps = eps - 1j * ETA0 * built.scale * sigma / k
    if not np.any(eps.imag):
        return [recovery.eps], [recovery.eps_inverse]
    inverse = 1.0 / eps if eps.ndim == 1 else np.linalg.inv(eps)

    def parts(values: np.ndarray) -> list[ngs.CoefficientFunction]:
        return [coefficient(built, values.real), coefficient(built, values.imag)]

    return parts(eps), parts(inverse)


def _times(
    w: Sequence[ngs.CoefficientFunction], x: Sequence[ngs.CoefficientFunction]
) -> list[ngs.CoefficientFunction]:
    """``w x`` of a complex coefficient and a complex field, as real parts.

    Each is its real part, then its imaginary part if it has one.
    """
    if len(w) == 1:
        return [w[0] * v for v in x]
    if len(x) == 1:
        return [w[0] * x[0], w[1] * x[0]]
    return [w[0] * x[0] - w[1] * x[1], w[1] * x[0] + w[0] * x[1]]


def _bilinear(
    recovery: FluxRecovery,
    x: list[ngs.CoefficientFunction],
    y: list[ngs.CoefficientFunction],
) -> np.ndarray:
    """``int_K x . y`` per tetrahedron, complex bilinear, of real-part lists."""
    return _complex([[recovery.element_integrals(u * v) for v in y] for u in x])


def _slope(model: Model, field: Field, k: complex) -> complex:
    """``x^T T'(k) x``, T the eigen operator of the field's model (``x``: its DOFs)."""
    from gfem._engine.driven import driven_operator
    from gfem._engine.eigen import eigen_operator
    from gfem._linalg import ResolvedLinearSolver
    from gfem.options import SolverOptions

    op = driven_operator(model, field.order)
    # T'(k) x only multiplies assembled matrices: nothing is factorized, so
    # the linear solver and its tolerance are placeholders the constructor
    # requires. The shift k.real is only where the polynomial pieces' theta
    # is fitted, exact for any shift.
    linear = ResolvedLinearSolver.resolve(
        SolverOptions(linear_solver="direct"), default="direct", problem="eigenmode"
    )
    t = eigen_operator(op, k.real, linear=linear, tolerance=1e-10)
    x = np.asarray(field.coefficients, complex)
    return complex(x @ t.derivative(k, x))


def _parts(field: Field) -> list[ngs.GridFunction]:
    """The real part of a field, and its imaginary part if it has one."""
    if not field.is_complex:
        return [_grid_function(field)]
    return [_grid_function(field, True), _grid_function(field, False)]


def _residuals(
    recovery: FluxRecovery,
    field: Field,
    eps: Sequence[ngs.CoefficientFunction],
) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
    """The recovered errors of mu^-1 curl E and of eps E, as real parts.

    ``eps`` is real, or complex as its real and imaginary parts
    (`_effective_eps`).
    """
    parts = _parts(field)
    curl = [recovery.curl_residual(ngs.curl(gf)) for gf in parts]
    flux = [recovery.flux_residual(v) for v in _times(eps, parts)]
    return curl, flux


def _pair(
    recovery: FluxRecovery,
    ri: tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]],
    rj: tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]],
    eps_inverse: Sequence[ngs.CoefficientFunction],
) -> tuple[np.ndarray, np.ndarray]:
    """c_K and m_K of two fields' recovered errors (`_residuals`), bilinear."""
    curl = _bilinear(recovery, ri[0], _times([recovery.mu], rj[0]))
    mass = _bilinear(recovery, ri[1], _times(eps_inverse, rj[1]))
    return curl, mass


def _complex(table: list[list[Any]]) -> Any:
    """``(a + i b)(c + i d)`` from the parts' products: rows a, b; columns c, d."""
    out = table[0][0] + 0j
    if len(table) > 1:
        out = out + 1j * table[1][0]
    if len(table[0]) > 1:
        out = out + 1j * table[0][1]
    if len(table) > 1 and len(table[0]) > 1:
        out = out - table[1][1]
    return out
