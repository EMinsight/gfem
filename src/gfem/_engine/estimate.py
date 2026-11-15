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

Every kernel takes an optional `RecoveryCache`: the recoveries and recovered
errors of one mesh's solves, shared by every estimate on that mesh (one
`FluxRecovery` per mesh, one projection per field). The quantity kernels
return their identity's complex `Terms`, from which an estimate kind builds
its value and its weights.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import coefficient, element_tensors
from gfem._engine.dispersion import (
    eps_at,
    high_frequency_eps,
    law_regions,
    static_eps,
)
from gfem._engine.domain import em_region, pec_surfaces
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._field import Field
from gfem.model import PEC, PMC, Conductor, Model
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult

__all__ = [
    "FluxRecovery",
    "RecoveryCache",
    "Terms",
    "capacitance",
    "combine",
    "driven",
    "eigenmode",
    "electrostatic",
    "objective_of_s",
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
        static: bool = False,
    ) -> None:
        """Build the recovery of an order-``order`` solution on ``model``.

        ``cracks`` cuts the spaces along the model's interior PEC sheets,
        ``sheets`` along these surfaces too (where they are interior).
        ``static`` marks an electrostatic solve, whose ``eps`` is the static
        permittivity; a wave solve's real ``eps`` takes a law region's
        eps_inf, read only where no law is present (`_lossy_eps_eff`).
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
        self.eps, self.eps_inverse = self._eps_weights(static)
        self.mu, self.mu_inverse, self.mu_parts, self.mu_inverse_parts = (
            self._mu_weights()
        )
        self._p0 = ngs.L2(self.built.mesh, order=0)

    def _projection(
        self, kind: Callable[..., ngs.FESpace], **flags: Any
    ) -> _Projection:
        mesh, order, region = self.built.mesh, max(self.order - 1, 0), self.region

        def make(**more: Any) -> ngs.FESpace:
            return kind(mesh, order=order, definedon=region, **flags, **more)

        return _Projection(make, self.dx, self._cracks, region, self._constrained)

    def _eps_weights(
        self, static: bool
    ) -> tuple[ngs.CoefficientFunction, ngs.CoefficientFunction]:
        """The real eps_r and its inverse, per element.

        On a law region: the static permittivity on an electrostatic solve
        (`static_eps`, the solve's own coefficient), eps_inf on a wave solve.
        """
        regions = law_regions(self.model)
        if not regions:
            return self._weights("eps_r")
        values = high_frequency_eps(self.model).real
        if static:
            eps0 = static_eps(self.model)
            for lr in regions:
                if values.ndim == 1:
                    values[lr.mask] = eps0[lr.mask]
                else:
                    values[lr.mask] = eps0[lr.mask, None, None] * np.eye(3)
        return self._real_weights(values)

    def _weights(
        self, name: str
    ) -> tuple[ngs.CoefficientFunction, ngs.CoefficientFunction]:
        """A real material property and its inverse, scalar or tensor, per element."""
        return self._real_weights(element_tensors(self.model, name).real)

    def _mu_weights(
        self,
    ) -> tuple[
        ngs.CoefficientFunction,
        ngs.CoefficientFunction,
        list[ngs.CoefficientFunction],
        list[ngs.CoefficientFunction],
    ]:
        """mu_r's norm weights and its complex parts, and those of its inverse.

        Returns (the weight of the recovered H error, the weight of the curl
        energy, mu as parts, mu^-1 as parts). For a real mu the weights are mu
        and mu^-1 and each is its own one part. For a complex mu (ADR 0138)
        the weights are the moduli (mu^H mu)^{1/2} and (mu mu^H)^{-1/2}
        (`_modulus_weights`), and mu^-1 curl E and the curl term c_K use the
        complex mu, as real and imaginary parts.
        """
        mu = np.asarray(element_tensors(self.model, "mu_r"))
        if not np.any(mu.imag):
            weight, inverse = self._real_weights(mu.real)
            return weight, inverse, [weight], [inverse]
        mu = mu.astype(complex)
        inverse = 1.0 / mu if mu.ndim == 1 else np.linalg.inv(mu)
        curl_weight, h_weight = _modulus_weights(mu)
        return (
            coefficient(self.built, h_weight),
            coefficient(self.built, curl_weight),
            _split(self.built, mu),
            _split(self.built, inverse),
        )

    def _real_weights(
        self, values: np.ndarray
    ) -> tuple[ngs.CoefficientFunction, ngs.CoefficientFunction]:
        """Real per-element values and their inverse as coefficients."""
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

    def curl_residuals(
        self, curls: Sequence[ngs.CoefficientFunction]
    ) -> list[ngs.CoefficientFunction]:
        """The recovered error of mu^-1 curl E, ``R(mu^-1 curl) - mu^-1 curl``.

        ``curls`` is curl E as its real (and imaginary) part; so is the
        result, which has two parts when mu or the field is complex.
        """
        if self._hcurl is None:
            self._hcurl = self._projection(ngs.HCurl, type1=False)
        hcurl = self._hcurl
        return [
            hcurl.project(field) - field
            for field in _times(self.mu_inverse_parts, curls)
        ]

    def flux_error(self, flux: ngs.CoefficientFunction) -> np.ndarray:
        """``|| R(flux) - flux ||^2_{eps^-1, K}`` per tetrahedron."""
        diff = self.flux_residual(flux)
        return self.element_integrals(diff * (self.eps_inverse * diff))

    def curl_error(self, curl: ngs.CoefficientFunction) -> np.ndarray:
        """``|| R(mu^-1 curl) - mu^-1 curl ||^2_{|mu|, K}`` per tetrahedron."""
        return sum(
            (
                self.element_integrals(diff * (self.mu * diff))
                for diff in self.curl_residuals([curl])
            ),
            np.zeros(len(self.model.mesh.tetrahedra)),
        )


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


class RecoveryCache:
    """The recoveries and recovered errors of one mesh's solves.

    Every estimate made on one mesh (one `refine_mesh` cycle, or one
    `error_estimate` call) shares one cache, so that several quantities
    reading the same solved fields pay for each recovery once:

    - one `FluxRecovery` per (model, order, recovery flavour: ``cracks``,
      ``sheets``, ``static``);
    - one NGSolve view of each field (`_parts`);
    - one eps_eff per (recovery, k) (`_effective_eps`);
    - one projection of each field's mu^-1 curl E and eps_eff E per
      (recovery, field, k) (`wave_residuals`), or of its eps grad V
      (`potential_residual`).

    Keys hold object identities, and the cache holds the keyed objects, so
    an id is never reused while it lives. Make one per mesh and drop it with
    the mesh: it keeps every recovered error alive.
    """

    def __init__(self) -> None:
        self._recoveries: dict[tuple, FluxRecovery] = {}
        self._entries: dict[tuple, Any] = {}
        self._held: list[Any] = []

    def recovery(
        self,
        model: Model,
        order: int,
        *,
        cracks: bool = False,
        sheets: Sequence[str] = (),
        static: bool = False,
    ) -> FluxRecovery:
        """The `FluxRecovery` of ``model`` at ``order``, built on first use."""
        key = (id(model), order, cracks, tuple(sorted(sheets)), static)
        if key not in self._recoveries:
            self._held.append(model)
            self._recoveries[key] = FluxRecovery(
                model, order, cracks=cracks, sheets=sheets, static=static
            )
        return self._recoveries[key]

    def _memo(self, key: tuple, held: Sequence[Any], make: Callable[[], Any]) -> Any:
        if key not in self._entries:
            self._held.extend(held)
            self._entries[key] = make()
        return self._entries[key]

    def parts(self, field: Field) -> list[ngs.GridFunction]:
        """`_parts` of ``field``: its real part, and its imaginary part if any."""
        return self._memo(("parts", id(field)), (field,), lambda: _parts(field))

    def effective_eps(
        self, recovery: FluxRecovery, k: complex
    ) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
        """`_effective_eps` of ``recovery`` at ``k``."""
        return self._memo(
            ("eps", id(recovery), complex(k)),
            (),
            lambda: _effective_eps(recovery, k),
        )

    def wave_residuals(
        self, recovery: FluxRecovery, field: Field, k: complex
    ) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
        """The recovered errors of a wave field at ``k`` (`_residuals`).

        mu^-1 curl E and eps_eff E, each as its real (and imaginary) part;
        eps_eff at ``k`` in the mesh's length unit.
        """

        def make() -> Any:
            eps, _ = self.effective_eps(recovery, k)
            return _residuals(recovery, self.parts(field), eps)

        return self._memo(("wave", id(recovery), id(field), complex(k)), (field,), make)

    def potential_residual(
        self, recovery: FluxRecovery, field: Field
    ) -> tuple[ngs.GridFunction, ngs.CoefficientFunction, ngs.CoefficientFunction]:
        """A potential's view, its flux eps grad V, and ``R(flux) - flux``."""

        def make() -> Any:
            (gf,) = self.parts(field)
            flux = recovery.eps * ngs.grad(gf)
            return gf, flux, recovery.flux_residual(flux)

        return self._memo(("potential", id(recovery), id(field)), (field,), make)


@dataclasses.dataclass(frozen=True)
class Terms:
    """One quantity's error identity on one solve, tetrahedron by tetrahedron.

    The identity's volume part (ADR 0089) is sum_K a_K, with
    a_K = c_K - lam m_K of the recovered errors of the quantity's field and
    of its dual, complex bilinear (not Hermitian):

        c_K = int_K (R(mu^-1 curl E_i) - mu^-1 curl E_i) . mu (...)_j
        m_K = int_K (R(eps E_i) - eps E_i) . eps^-1 (...)_j

    An estimate kind builds its value from these: an entry keeps the curl
    and mass sums apart (`value`); a later kind may combine several entries'
    complex sums (``curl_sum``, ``mass_sum``) or per-tetrahedron terms before
    taking magnitudes.

    Attributes:
        curl: c_K per tetrahedron, complex, gfem tetrahedron order; zero for
            a potential.
        mass: m_K per tetrahedron, complex.
        lam: The factor of m_K in a_K: k0^2 for an S entry, k^2 (complex on
            a lossy mode) for omega, -1 for a capacitance (a_K = m_K).
        scale: The positive factor carrying a_K into the quantity's units.
        elementwise: Whether `value` sums |a_K| over the tetrahedra (a
            capacitance entry, whose off-diagonal terms have no sign)
            instead of keeping the curl and mass sums whole.
    """

    curl: np.ndarray
    mass: np.ndarray
    lam: complex
    scale: float
    elementwise: bool = False

    @property
    def curl_sum(self) -> complex:
        """sum_K c_K."""
        return complex(self.curl.sum())

    @property
    def mass_sum(self) -> complex:
        """sum_K m_K."""
        return complex(self.mass.sum())

    @property
    def weights(self) -> np.ndarray:
        """Each tetrahedron's |a_K| in the quantity's units (ticket 08's indicator)."""
        return self.scale * np.abs(self.curl - self.lam * self.mass)

    @property
    def value(self) -> float:
        """The entry's estimate: ``scale (|sum c_K| + |lam| |sum m_K|)``.

        The two sums are never let cancel over the mesh. With
        ``elementwise``, ``sum_K |a_K| scale`` (the sum of `weights`).
        """
        if self.elementwise:
            return float(self.weights.sum())
        return float(
            self.scale * (abs(self.curl_sum) + abs(self.lam) * abs(self.mass_sum))
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
    model: Model,
    result: ElectrostaticResult,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each terminal's potential.

    One solve (the whole capacitance matrix), holding one field per
    terminal: eta_K^2 / || grad V ||^2_eps per tetrahedron. ``cache`` shares
    the recovery and the recovered errors with the mesh's other estimates.
    """
    cache = cache or RecoveryCache()
    with task_manager(threads):
        recovery = _potential_recovery(model, result, cache)
        group = []
        for field in result.potential.values():
            gf, flux, diff = cache.potential_residual(recovery, field)
            energy = recovery.element_integrals(ngs.grad(gf) * flux).sum()
            error = recovery.element_integrals(diff * (recovery.eps_inverse * diff))
            group.append(error / energy)
    return [group]


def _potential_recovery(
    model: Model, result: ElectrostaticResult, cache: RecoveryCache
) -> FluxRecovery:
    """The flux recovery of an electrostatic solve's potentials.

    Cut along the interior surfaces the problem holds at a fixed potential
    (its terminals and ground, and the model's PEC and conductor surfaces),
    across which the solved flux jumps.
    """
    sheets = set(result._fixed) | {  # noqa: SLF001
        b.surface for b in model.boundaries if isinstance(b, (PEC, Conductor))
    }
    order = next(iter(result.potential.values())).order
    return cache.recovery(model, order, sheets=sorted(sheets), static=True)


def driven(
    model: Model,
    result: DrivenResult,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each kept field, grouped by frequency.

    One group per frequency, holding one field per excited port, estimated
    one at a time.
    """
    cache = cache or RecoveryCache()
    groups: dict[float, list[np.ndarray]] = {}
    with task_manager(threads):
        for (frequency, _), field in zip(result.kept, result.e_fields, strict=True):
            recovery = cache.recovery(model, field.order, cracks=True)
            groups.setdefault(frequency, []).append(_wave(recovery, field, cache))
    return list(groups.values())


def eigenmode(
    model: Model,
    result: EigenmodeResult,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> list[list[np.ndarray]]:
    """Normalized squared indicators of each mode, one group per mode."""
    cache = cache or RecoveryCache()
    fields = list(result.e_fields)
    with task_manager(threads):
        recovery = cache.recovery(model, fields[0].order, cracks=True)
        return [[_wave(recovery, field, cache)] for field in fields]


def _wave(recovery: FluxRecovery, field: Field, cache: RecoveryCache) -> np.ndarray:
    """eta_K^2 / energy of one time-harmonic electric field.

    eta_K^2 = || R(mu^-1 curl E) - mu^-1 curl E ||^2_{|mu|,K}
    + |k|^2 || R(eps_eff E) - eps_eff E ||^2_{|eps_eff|^-1,K}, with k =
    omega / c in the mesh's length unit (complex for a lossy mode) and
    eps_eff = eps_r (1 - i tan_delta) - i eta0 sigma scale / k, the total
    current whose normal component is continuous; |.|^2 of a complex
    residual sums its real and imaginary parts. The energy is
    || curl E ||^2_{|mu|^-1} + |k|^2 int |eps_eff| |E|^2 over the solve
    domain. A complex mu_r enters mu^-1 curl E whole and the norms through
    its modulus (`FluxRecovery`, ADR 0138). Without loss eps_eff = eps_r,
    |mu| = mu, and this is the lossless estimate.
    """
    built = recovery.built
    k = complex(np.asarray(field.omega)) / scipy.constants.c * built.scale
    k2 = abs(k) ** 2
    values = _lossy_eps_eff(recovery, k)
    if values is None:
        flux_weight, mass_weight = recovery.eps_inverse, recovery.eps
    else:
        flux_weight, mass_weight = (
            coefficient(built, w) for w in _modulus_weights(values)
        )
    parts = cache.parts(field)
    curls, fluxes = cache.wave_residuals(recovery, field, k)
    eta2 = np.zeros(len(recovery.model.mesh.tetrahedra))
    energy = 0.0
    for gf, rc, rf in zip(parts, curls, fluxes, strict=True):
        curl = ngs.curl(gf)
        eta2 += recovery.element_integrals(
            rc * (recovery.mu * rc)
        ) + k2 * recovery.element_integrals(rf * (flux_weight * rf))
        energy += recovery.element_integrals(
            curl * (recovery.mu_inverse * curl) + k2 * (gf * (mass_weight * gf))
        ).sum()
    return eta2 / energy


def _modulus_weights(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The flux and energy weights of a complex eps_eff (or mu_r) per tetrahedron.

    |eps|^-1 and |eps| for a scalar; for a tensor A, (A A^H)^{-1/2} and
    (A^H A)^{1/2}, A^-1 and A for a real symmetric A. Real for every tensor
    gfem represents: A = R D R^T with real orthonormal axes R shared by
    eps_r, tan_delta and conductivity (and by mu_r), so A A^H = R |D|^2 R^T
    is real and ``.real`` drops only rounding (ADR 0138).
    """
    if eps.ndim == 1:
        return 1.0 / np.abs(eps), np.abs(eps)

    def power(h: np.ndarray, p: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(h)
        return np.einsum("tij,tj,tkj->tik", vectors, values**p, vectors.conj()).real

    adjoint = eps.conj().transpose(0, 2, 1)
    return power(eps @ adjoint, -0.5), power(adjoint @ eps, 0.5)


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
# complex bilinear (not Hermitian). Each kernel returns these terms whole
# (`Terms`): the complex c_K and m_K, their sums, lam and the scale. An entry's
# reported estimate is the magnitude |sum c_K| + |lam| |sum m_K| (the two terms
# are never let cancel over the mesh), scaled to the quantity's units. A
# tetrahedron's weight is the magnitude of its term of the identity,
# |a_K| = |c_K - lam m_K| with the same scale (ticket 08's indicator): marking
# is unchanged by the reported sum, and the weights add up to neither that sum
# nor |sum a_K| in general.


def capacitance(
    model: Model,
    result: ElectrostaticResult,
    i: str,
    j: str,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> Terms:
    """The terms of ``capacitance[i, j]``'s relative error identity.

    C_ij,h - C_ij = a(e_i, e_j) for the potentials of terminals i and j at
    1 V, each the other's free dual. A tetrahedron's term is m_K(i, j), the
    recovered flux errors of the two potentials in the eps^-1 product (real;
    `Terms.curl` is zero and ``lam`` -1). The scale is 1 / |a_h(u_i, u_j)|
    and the terms are ``elementwise``: the estimate is sum_K |m_K(i, j)|
    over |a_h(u_i, u_j)|, never letting the terms of an off-diagonal entry
    (which have no sign) cancel. At i = j it is the square of the field
    estimate of that terminal's potential, the weights `electrostatic` gives
    it.
    """
    cache = cache or RecoveryCache()
    with task_manager(threads):
        recovery = _potential_recovery(model, result, cache)
        gi, _, ri = cache.potential_residual(recovery, result.potential[i])
        _, flux_j, rj = cache.potential_residual(recovery, result.potential[j])
        energy = recovery.element_integrals(ngs.grad(gi) * flux_j).sum()
        mass = recovery.element_integrals(ri * (recovery.eps_inverse * rj))
    return Terms(
        curl=np.zeros(len(mass), complex),
        mass=mass.astype(complex),
        lam=-1.0,
        scale=1.0 / abs(energy),
        elementwise=True,
    )


def s_entry(
    model: Model,
    result: DrivenResult,
    to: str,
    source: str,
    threads: int | None = None,
    *,
    adjoints: Mapping[tuple[float, Hashable], Field] | None = None,
    cache: RecoveryCache | None = None,
) -> list[Terms]:
    """The terms of ``s[to, source]``'s absolute error identity, per frequency.

    With port i = ``to``, j = ``source``, e = u_h - u, and z_i the adjoint
    of port i's S functional f_i (A z_i = f_i, A complex symmetric):
    O_ij,h - O_ij = f_i(e_j) = -a(e_j, z_i,h - z_i) for the raw outgoing
    wave O, and S_ij = (O_ij - delta_ij) d_i d_j with the de-embedding d
    (``exp(gamma offset)``, 1 for a lumped port). a = c - k0^2 m with k0 in
    the mesh's length unit (``lam``); m recovers eps_eff E
    (`_effective_eps`). The terms are those of the recovered errors of x_j
    and z_i, scaled by |d_i d_j| (and the free dual's |c_i|): the estimate
    is |d_i d_j| (|sum c_K| + k0^2 |sum m_K|), a tetrahedron's
    |d_i d_j| |c_K - k0^2 m_K|.

    ``adjoints`` holds z_i per (frequency, port) where port i is not
    excited or its functional is not parallel to its excitation (a hybrid
    lossy wave port mode).
    Elsewhere z_i = c_i x_i with |c_i| = 1 / (4 k0 eta0) for fields of 1 W
    incident (ADR 0089): the free dual, x_i itself on that scale.
    """
    cache = cache or RecoveryCache()
    adjoints = adjoints or {}
    out = []
    with task_manager(threads):
        for frequency in result.s.frequency:
            pair = _s_pair(
                model,
                result,
                float(frequency),
                (to, source),
                adjoints=adjoints,
                cache=cache,
            )
            out.append(
                Terms(
                    curl=pair.curl,
                    mass=pair.mass,
                    lam=pair.lam,
                    scale=float(abs(pair.factor)),
                )
            )
    return out


@dataclasses.dataclass(frozen=True)
class _SPair:
    """One S entry's identity terms at one frequency, with its complex factor.

    S_ij,h - S_ij ~= -factor sum_K (curl_K - lam mass_K), factor =
    d_i d_j times the free dual's scale ``-i / (4 k0 eta0)`` (the S
    functional of every excited port, lumped or wave, parallel to its
    excitation), or 1 for a solved adjoint.
    """

    curl: np.ndarray
    mass: np.ndarray
    lam: complex
    factor: complex

    @property
    def value(self) -> float:
        """The entry's estimate: |factor| (|sum c_K| + |lam| |sum m_K|)."""
        return float(
            abs(self.factor)
            * (abs(self.curl.sum()) + abs(self.lam) * abs(self.mass.sum()))
        )

    @property
    def weights(self) -> np.ndarray:
        """The complex terms ``factor (c_K - lam m_K)`` per tetrahedron."""
        return self.factor * (self.curl - self.lam * self.mass)


def _s_pair(
    model: Model,
    result: DrivenResult,
    f: float,
    entry: tuple[str, str],
    *,
    adjoints: Mapping[tuple[float, Hashable], Field],
    cache: RecoveryCache,
) -> _SPair:
    """`_SPair` of ``s[to, source]`` at frequency ``f`` (see `s_entry`)."""
    to, source = entry
    fields = dict(zip(result.kept, result.e_fields, strict=True))
    n = list(result.s.frequency).index(f)
    d = _deembedding(result)
    eta0 = scipy.constants.mu_0 * scipy.constants.c
    f_source = fields[f, source]
    recovery = cache.recovery(model, f_source.order, cracks=True)
    k = 2 * np.pi * f / scipy.constants.c * recovery.built.scale
    _, eps_inverse = cache.effective_eps(recovery, k)
    ri = cache.wave_residuals(recovery, f_source, k)
    adjoint = adjoints.get((f, to))
    if adjoint is not None:
        rj, scale = cache.wave_residuals(recovery, adjoint, k), 1.0 + 0j
    else:
        rj = cache.wave_residuals(recovery, fields[f, to], k)
        scale = -1j / (4 * k * eta0)
    curl, mass = _pair(recovery, ri, rj, eps_inverse)
    return _SPair(
        curl=curl,
        mass=mass,
        lam=k**2,
        factor=complex(scale * d[to][n] * d[source][n]),
    )


@dataclasses.dataclass(frozen=True)
class ObjectiveTerms:
    """An objective's estimate on one result: its value and tetrahedron weights.

    Attributes:
        linear: The combined-dual term, summed over frequencies as magnitudes.
        remainder: The Hessian remainder 1/2 sum_ab ||H_ab|| eta_a eta_b.
        weights: Each tetrahedron's term in the objective's units (its
            ``indicators**2``).
    """

    linear: float
    remainder: float
    weights: np.ndarray

    @property
    def value(self) -> float:
        """The estimate: ``linear + remainder``."""
        return self.linear + self.remainder


def objective_of_s(
    model: Model,
    result: DrivenResult,
    reads: Sequence[tuple[float, str, str, complex]],
    curvature: np.ndarray,
    threads: int | None = None,
    *,
    adjoints: Mapping[tuple[float, Hashable], Field] | None = None,
    cache: RecoveryCache | None = None,
) -> ObjectiveTerms:
    """An objective of S's estimate, the C form of ADR 0129.

    eta_J = sum_f (|sum_K X_c,K| + k_f^2 |sum_K X_m,K|) + 1/2 sum_ab
    ||H_ab|| eta_a eta_b, with X_.,K = sum_e ct_e factor_e (.)_K,e over the
    entries e = (f, to, from) read at f: the combined dual of the entries'
    identities (`_SPair`), whose pair sums are combined before taking
    magnitudes. eta_a is entry a's own S estimate.

    A tetrahedron's weight is sum_f |sum_e ct_e factor_e (c_K - k^2 m_K)|,
    plus its share of the remainder, 1/2 sum_ab ||H_ab|| eta_b |a_K,a|.

    Args:
        reads: Per read entry, (frequency, to, from, cotangent), in the
            order of ``curvature``'s rows.
        curvature: ``||H_ab||_2`` over the read entries.
        adjoints: The solved S-functional adjoints per (frequency, port),
            for a read row whose port is not excited or not parallel.
    """
    cache = cache or RecoveryCache()
    rows = [(f, to, source) for f, to, source, _ in reads]
    pairs = _s_pairs(
        model, result, rows, adjoints=adjoints or {}, cache=cache, threads=threads
    )
    cotangents = np.array([ct for *_, ct in reads], complex)
    return _combined(pairs, cotangents, curvature, len(model.mesh.tetrahedra))


def _s_pairs(
    model: Model,
    result: DrivenResult,
    rows: Sequence[tuple[float, str, str]],
    *,
    adjoints: Mapping[tuple[float, Hashable], Field],
    cache: RecoveryCache,
    threads: int | None,
) -> list[tuple[float, _SPair]]:
    """Each row (frequency, to, from)'s `_SPair`, with its frequency."""
    with task_manager(threads):
        return [
            (f, _s_pair(model, result, f, (to, source), adjoints=adjoints, cache=cache))
            for f, to, source in rows
        ]


def _combined(
    pairs: Sequence[tuple[float, _SPair]],
    cotangents: np.ndarray,
    curvature: np.ndarray,
    n_tet: int,
) -> ObjectiveTerms:
    """The C form's terms of ``cotangents`` over the entries ``pairs``."""
    curl: dict[float, np.ndarray] = {}
    mass: dict[float, np.ndarray] = {}
    lam: dict[float, complex] = {}
    etas, entry_weights = [], []
    for (f, pair), ct in zip(pairs, cotangents, strict=True):
        curl[f] = curl.get(f, np.zeros(n_tet, complex)) + ct * pair.factor * pair.curl
        mass[f] = mass.get(f, np.zeros(n_tet, complex)) + ct * pair.factor * pair.mass
        lam[f] = pair.lam
        etas.append(pair.value)
        entry_weights.append(np.abs(pair.weights))
    linear = sum(abs(curl[f].sum()) + abs(lam[f]) * abs(mass[f].sum()) for f in curl)
    weights = sum((np.abs(curl[f] - lam[f] * mass[f]) for f in curl), np.zeros(n_tet))
    eta = np.asarray(etas, float)
    remainder = float(0.5 * eta @ curvature @ eta) if len(eta) else 0.0
    share = 0.5 * curvature @ eta
    for a, w in enumerate(entry_weights):
        weights = weights + share[a] * w
    return ObjectiveTerms(linear=float(linear), remainder=remainder, weights=weights)


@dataclasses.dataclass(frozen=True)
class RewriteTerms(ObjectiveTerms):
    """An objective's lossless rewrite on one result (ADR 0135).

    ``linear``, ``remainder`` and ``weights`` are the C form's of J_lam
    (`objective_of_s`); the global defect term is not in ``weights``.

    Attributes:
        lam: The multiplier per read column, in the order of ``columns``.
        defect: D_c = 1 - sum_i |S_ic,h|^2 per read column.
        columns: The read columns, (frequency, source port).
    """

    lam: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    defect: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    columns: tuple[tuple[float, str], ...] = ()

    @property
    def inner(self) -> float:
        """C(J_lam): the estimate of the rewritten objective's own error."""
        return self.linear + self.remainder

    @property
    def value(self) -> float:
        """The estimate: ``C(J_lam) + |lam . D|``."""
        return self.inner + abs(float(self.lam @ self.defect))


def lossless_rewrite(
    model: Model,
    result: DrivenResult,
    reads: Sequence[tuple[float, str, str, complex]],
    curvature: np.ndarray,
    diagonal: Sequence[np.ndarray],
    *,
    threads: int | None = None,
    adjoints: Mapping[tuple[float, Hashable], Field] | None = None,
    cache: RecoveryCache | None = None,
    lam: np.ndarray | None = None,
) -> RewriteTerms:
    """An objective of S's estimate through its lossless rewrite (ADR 0135).

    On a network that conserves power, sum_i |S_ic|^2 = 1 for every column
    c = (frequency, source), so J_lam = J + sum_c lam_c (sum_i |S_ic|^2 - 1)
    equals J at the true S and err(J) = err(J_lam) + lam . D exactly, D_c =
    1 - sum_i |S_ic,h|^2 computed from S_h. The estimate is C(J_lam) +
    |lam . D|, C the form of `objective_of_s` over every row of the read
    columns: J_lam's cotangent is ct + 2 lam_c conj(S_h), and its Hessian
    adds 2 lam_c to each (Re, Im) diagonal block of the column. lam minimizes
    the estimate (convex in lam, one real variable per column), started at
    lam = 0, which gives back C; never above it.

    Args:
        reads: Per read entry, (frequency, to, from, cotangent), in the
            order of ``curvature``'s rows.
        curvature: ``||H_ab||_2`` over the read entries.
        diagonal: Each read entry's (Re, Im) Hessian block ``H_aa``, 2 x 2.
        adjoints: The solved S-functional adjoints per (frequency, port),
            for a row whose port is not excited.
        lam: Evaluate at this multiplier instead of minimizing.
    """
    import scipy.optimize

    cache = cache or RecoveryCache()
    s = result.s
    values = np.asarray(s.values)
    frequencies = [float(f) for f in s.frequency]
    columns = tuple(dict.fromkeys((f, source) for f, _, source, _ in reads))
    rows = [(f, to, source) for f, to, source, _ in reads]
    rows += [
        (f, to, source)
        for f, source in columns
        for to in s.ports
        if (f, to, source) not in rows
    ]
    n_read, n = len(reads), len(rows)
    column = np.array([columns.index((f, source)) for f, _, source in rows])
    s_h = np.array(
        [
            values[frequencies.index(f), s.ports.index(to), s.excited.index(source)]
            for f, to, source in rows
        ]
    )
    defect = np.array(
        [
            1.0
            - float(
                np.sum(np.abs(values[frequencies.index(f), :, s.excited.index(c)]) ** 2)
            )
            for f, c in columns
        ]
    )
    pairs = _s_pairs(
        model, result, rows, adjoints=adjoints or {}, cache=cache, threads=threads
    )
    cotangents = np.zeros(n, complex)
    cotangents[:n_read] = [ct for *_, ct in reads]
    base = np.zeros((n, n))
    base[:n_read, :n_read] = curvature
    blocks = np.zeros((n, 2, 2))
    blocks[:n_read] = np.asarray(diagonal, float).reshape(n_read, 2, 2)
    # the scalar sums the minimization moves: per entry, per frequency
    curl_sums = np.array([p.factor * p.curl.sum() for _, p in pairs])
    mass_sums = np.array([p.factor * p.mass.sum() for _, p in pairs])
    eta = np.array([p.value for _, p in pairs])
    groups = list(dict.fromkeys(f for f, _ in pairs))
    group = np.zeros((len(groups), n))
    for e, (f, _) in enumerate(pairs):
        group[groups.index(f), e] = 1.0
    k2 = np.array([abs(next(p.lam for g, p in pairs if g == f)) for f in groups])

    def shifted(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lam_e = x[column]
        cts = cotangents + 2 * lam_e * np.conj(s_h)
        norms = base.copy()
        norms[np.diag_indices(n)] = np.linalg.norm(
            blocks + 2 * lam_e[:, None, None] * np.eye(2), 2, axis=(1, 2)
        )
        return cts, norms

    def estimate(x: np.ndarray) -> float:
        cts, norms = shifted(x)
        linear = np.abs(group @ (cts * curl_sums)) + k2 * np.abs(
            group @ (cts * mass_sums)
        )
        return float(linear.sum() + 0.5 * eta @ norms @ eta + abs(x @ defect))

    if lam is None:
        zero = np.zeros(len(columns))
        found = scipy.optimize.minimize(
            estimate,
            zero,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 4000},
        )
        lam = found.x if found.fun < estimate(zero) else zero
    lam = np.asarray(lam, float)
    cts, norms = shifted(lam)
    terms = _combined(pairs, cts, norms, len(model.mesh.tetrahedra))
    return RewriteTerms(
        linear=terms.linear,
        remainder=terms.remainder,
        weights=terms.weights,
        lam=lam,
        defect=defect,
        columns=columns,
    )


def _deembedding(result: DrivenResult) -> dict[str, np.ndarray]:
    """Each port's de-embedding per frequency: ``exp(gamma offset)``, 1 if lumped."""
    from gfem._engine.wave_ports import deembedding

    offsets = dict(result.offsets)
    ones = np.ones(len(result.s.frequency), complex)
    modes = result.port_modes
    return {
        name: deembedding(np.asarray(modes[name].gamma), offsets.get(name, 0.0))
        if name in modes
        else ones
        for name in result.s.ports
    }


def omega(
    model: Model,
    result: EigenmodeResult,
    mode: int,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> Terms:
    """The terms of ``omega[mode]``'s relative error identity.

    A mode of the complex-symmetric T(k) is its own dual, and k_h - k ~=
    -e^T T(k) e / (x_h^T T'(k_h) x_h), k = omega scale / c complex. The
    numerator's volume terms are c_K - k^2 m_K: c_K of the recovered error
    of mu^-1 curl E, m_K = int e_D . eps_eff^-1 e_D of the recovered error
    e_D of eps_eff E, eps_eff = eps_r (1 - i tan_delta) - i eta0 sigma scale
    / k (the total current, whose normal component is continuous); the
    products are complex bilinear. T'(k_h) is the eigen operator's own
    (`gfem._eigen.operator.EigenOperator.derivative`), so it holds the
    damping of lumped resistors, impedance, absorbing and conductor surfaces.
    ``lam`` is k^2 (complex on a lossy mode) and the scale
    1 / (|k| |x_h^T T' x_h|): the estimate is (|sum c_K| + |k^2| |sum m_K|)
    / (|k| |x_h^T T' x_h|), a tetrahedron's |c_K - k^2 m_K| on the same
    scale. On a lossless mode T' = -2k M and this is (|sum c_K| + k^2
    |sum m_K|) / (2 k^2 |b(E, E)|).
    """
    curl, mass, k, slope = _mode_terms(model, result, mode, threads, cache)
    scale = 1.0 / (abs(k) * abs(slope))
    return Terms(curl=curl, mass=mass, lam=k * k, scale=scale)


def omega_cluster(
    model: Model,
    result: EigenmodeResult,
    modes: Sequence[int],
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> Terms:
    """The terms of the relative error of the mean ``omega`` of ``modes`` (ADR 0136).

    The cluster's members X = [x_1 .. x_m] span what the single-mode identity
    (`omega`) reads off one mode: the mean k's error is ~= -tr(G^-1 E^T T E)
    / m, G = X^T T'(k) X, at the members' mean k. Its volume terms are the
    trace tr(G^-1 C_K) and tr(G^-1 M_K) of the members' pair integrals C_K =
    [c_K(e_i, e_j)] and M_K = [m_K(e_i, e_j)] of their recovered errors at
    the mean k (`_pair`), so ``curl`` and ``mass`` hold these traces, ``lam``
    is the mean k squared and the scale 1 / (m |k|): the estimate is
    (|tr(G^-1 C)| + |k^2| |tr(G^-1 M)|) / (m |k|), a tetrahedron's |tr(G^-1
    (C_K - k^2 M_K))| / (m |k|). Value and weights are unchanged under any
    basis of the members' span, so a degenerate cluster's arbitrary basis
    does not enter. One member is `omega`'s identity.
    """
    cache = cache or RecoveryCache()
    fields = [result.e_fields[mode] for mode in modes]
    count = len(fields)
    with task_manager(threads):
        recovery = cache.recovery(model, fields[0].order, cracks=True)
        omegas = [complex(np.asarray(field.omega)) for field in fields]
        k = sum(omegas) / count / scipy.constants.c * recovery.built.scale
        _, eps_inverse = cache.effective_eps(recovery, k)
        errors = [cache.wave_residuals(recovery, field, k) for field in fields]
        inverse = np.linalg.inv(_slopes(model, fields, k))
        curl: Any = 0j
        mass: Any = 0j
        for i in range(count):
            for j in range(i, count):
                c_ij, m_ij = _pair(recovery, errors[i], errors[j], eps_inverse)
                # C and M are symmetric: tr(G^-1 C) = sum_ij (G^-1)_ji C_ij
                weight = inverse[j, i] + (inverse[i, j] if i != j else 0)
                curl = curl + weight * c_ij
                mass = mass + weight * m_ij
    return Terms(curl=curl, mass=mass, lam=k * k, scale=1.0 / (count * abs(k)))


def _mode_terms(
    model: Model,
    result: EigenmodeResult,
    mode: int,
    threads: int | None,
    cache: RecoveryCache | None,
) -> tuple[np.ndarray, np.ndarray, complex, complex]:
    """c_K, m_K, k and x_h^T T'(k_h) x_h of ``omega[mode]``'s identity (`omega`)."""
    cache = cache or RecoveryCache()
    field = result.e_fields[mode]
    with task_manager(threads):
        recovery = cache.recovery(model, field.order, cracks=True)
        k = complex(np.asarray(field.omega)) / scipy.constants.c * recovery.built.scale
        _, eps_inverse = cache.effective_eps(recovery, k)
        errors = cache.wave_residuals(recovery, field, k)
        curl, mass = _pair(recovery, errors, errors, eps_inverse)
        slope = _slope(model, field, k)
    return curl, mass, k, slope


@dataclasses.dataclass(frozen=True)
class QTerms(Terms):
    """The terms of a mode's Q relative error (ADR 0131), from omega's identity.

    Q = |k| / (2 Im k) = 1 / (2 sin phi), phi = arg k_h, so dQ/Q = -cot phi
    Im(delta k / k): the real part of omega's relative error drops out. With
    omega's identity delta k / k ~= -(P1 - P2), P1 = sum c_K / (s k), P2 =
    lam sum m_K / (s k), the value is cot phi (|Im P1| + |Im P2|), the two
    terms never let cancel, and a tetrahedron's weight cot phi |Im((c_K - lam
    m_K) / (s k))|.

    Attributes:
        phase: 1 / (s k), s = x_h^T T'(k_h) x_h.
        scale: cot phi (positive: a decaying mode has 0 < phi < pi / 2).
    """

    phase: complex = 0j

    @property
    def weights(self) -> np.ndarray:
        """Each tetrahedron's weight, cot phi |Im((c_K - lam m_K) / (s k))|."""
        return self.scale * np.abs(
            (self.phase * (self.curl - self.lam * self.mass)).imag
        )

    @property
    def value(self) -> float:
        """Q's estimate, cot phi (|Im P1| + |Im P2|), relative to Q."""
        p1 = self.phase * self.curl_sum
        p2 = self.phase * self.lam * self.mass_sum
        return float(self.scale * (abs(p1.imag) + abs(p2.imag)))


def q(
    model: Model,
    result: EigenmodeResult,
    mode: int,
    threads: int | None = None,
    *,
    cache: RecoveryCache | None = None,
) -> QTerms:
    """The terms of ``q[mode]``'s relative error (ADR 0131); see `QTerms`.

    The mode must be lossy (Im omega > 0): Q is infinite otherwise.
    """
    curl, mass, k, slope = _mode_terms(model, result, mode, threads, cache)
    cot = k.real / k.imag
    return QTerms(
        curl=curl,
        mass=mass,
        lam=k * k,
        scale=abs(cot),
        phase=1.0 / (slope * k),
    )


def _effective_eps(
    recovery: FluxRecovery, k: complex
) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
    """eps_eff and its inverse at ``k``, each as its real (and imaginary) part.

    eps_eff = eps_r (1 - i tan_delta) - i eta0 sigma scale / k per
    tetrahedron, a scalar or a 3x3 tensor; real (``recovery.eps``) on a
    lossless model.
    """
    eps = _lossy_eps_eff(recovery, k)
    if eps is None:
        return [recovery.eps], [recovery.eps_inverse]
    inverse = 1.0 / eps if eps.ndim == 1 else np.linalg.inv(eps)
    return _split(recovery.built, eps), _split(recovery.built, inverse)


def _lossy_eps_eff(recovery: FluxRecovery, k: complex) -> np.ndarray | None:
    """`_eps_eff_values` at ``k``, or ``None`` where it is real (no loss).

    Without loss the recovery's own real ``eps`` coefficients stand. With a
    law region the values stand even when real: a lossless law depends on
    frequency, and ``recovery.eps`` holds its eps_inf (ADR 0111).
    """
    eps = _eps_eff_values(recovery.model, recovery.built, k)
    return eps if np.any(eps.imag) or law_regions(recovery.model) else None


def _split(built: EngineMesh, values: np.ndarray) -> list[ngs.CoefficientFunction]:
    """Complex per-tetrahedron values as their real and imaginary coefficients."""
    return [coefficient(built, values.real), coefficient(built, values.imag)]


def _eps_eff_values(model: Model, built: EngineMesh, k: complex) -> np.ndarray:
    """eps_r (1 - i tan_delta) - i eta0 sigma scale / k per tetrahedron.

    ``(n,)``, or ``(n, 3, 3)`` when any material is a tensor. A law region
    gives its eps(omega) at omega = k c / scale (`dispersion.eps_at`,
    analytic, so a mode's complex k is fine).
    """
    from gfem._engine.coefficients import lossy_eps
    from gfem._engine.terms import ETA0

    sigma = element_tensors(model, "conductivity").real
    if law_regions(model):
        eps = eps_at(model, k * scipy.constants.c / built.scale)
    else:
        eps = lossy_eps(model)
    if eps.ndim != sigma.ndim:
        eye = np.eye(3)
        eps, sigma = (
            v if v.ndim == 3 else v[:, None, None] * eye for v in (eps, sigma)
        )
    return eps - 1j * ETA0 * built.scale * sigma / k


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
    return complex(_slopes(model, [field], k)[0, 0])


def _slopes(model: Model, fields: Sequence[Field], k: complex) -> np.ndarray:
    """``G = X^T T'(k) X``, X the fields' DOFs as columns, T the eigen operator."""
    from gfem._engine.driven import driven_operator
    from gfem._engine.eigen import eigen_operator
    from gfem._linalg import ResolvedLinearSolver
    from gfem.options import SolverOptions

    op = driven_operator(model, fields[0].order)
    # T'(k) x only multiplies assembled matrices: nothing is factorized, so
    # the linear solver and its tolerance are placeholders the constructor
    # requires. The shift k.real is only where the polynomial pieces' theta
    # is fitted, exact for any shift.
    linear = ResolvedLinearSolver.resolve(
        SolverOptions(linear_solver="direct"), default="direct", problem="eigenmode"
    )
    t = eigen_operator(op, k.real, linear=linear, tolerance=1e-10)
    xs = [np.asarray(field.coefficients, complex) for field in fields]
    slopes = [t.derivative(k, x) for x in xs]
    return np.array([[x @ slope for slope in slopes] for x in xs], complex)


def _parts(field: Field) -> list[ngs.GridFunction]:
    """The real part of a field, and its imaginary part if it has one."""
    if not field.is_complex:
        return [_grid_function(field)]
    return [_grid_function(field, True), _grid_function(field, False)]


def _residuals(
    recovery: FluxRecovery,
    parts: Sequence[ngs.GridFunction],
    eps: Sequence[ngs.CoefficientFunction],
) -> tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]]:
    """The recovered errors of mu^-1 curl E and of eps E, as real parts.

    ``parts`` is the field's view (`_parts`). ``eps`` is real, or complex as
    its real and imaginary parts (`_effective_eps`).
    """
    curl = recovery.curl_residuals([ngs.curl(gf) for gf in parts])
    flux = [recovery.flux_residual(v) for v in _times(eps, parts)]
    return curl, flux


def _pair(
    recovery: FluxRecovery,
    ri: tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]],
    rj: tuple[list[ngs.CoefficientFunction], list[ngs.CoefficientFunction]],
    eps_inverse: Sequence[ngs.CoefficientFunction],
) -> tuple[np.ndarray, np.ndarray]:
    """c_K and m_K of two fields' recovered errors (`_residuals`), bilinear."""
    curl = _bilinear(recovery, ri[0], _times(recovery.mu_parts, rj[0]))
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
