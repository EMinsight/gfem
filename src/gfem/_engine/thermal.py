r"""Thermal rules: heat conduction on the thermal domain.

The unknown is the absolute temperature :math:`T` in kelvin, in real H1 of
the solve order on the thermal domain (every tetrahedron minus the volumes
excluded from thermal problems). The steady operator is

.. math::

    a(u, v) = \int k \nabla u \cdot \nabla v
        + \sum_\mathrm{Convection} \int_S h u v,

the load is :math:`\int q v` for the heat sources' power density :math:`q`
(one P0 value per tetrahedron) plus :math:`\int_S q_s v` for their face
density :math:`q_s` (one P0 value per gfem triangle: sheet heat, which the
face's own condition then takes, splits or sends in),
:math:`\int_S \mathrm{flux}\, v` per `gfem.HeatFlux` and
:math:`\int_S h T_\infty v` per `gfem.Convection`, and the
`gfem.FixedTemperature` surfaces are Dirichlet surfaces, lifted with their
values (a Dirichlet value wins on nodes shared with another condition).
Insulated surfaces need no term.

The temperature splits as :math:`T = T_0 + \mathrm{rise}`: T0 is the source-free
temperature (boundary data only) and the rise is linear in the problem's
heat sources, with homogeneous boundary data. Both are right-hand sides of
one prepared operator; T0 needs no solve when every fixed temperature and
ambient share one value and no heat flux is given (then T0 is that value).

Forms are assembled in the mesh's file unit (``scale`` metres): the
assembled system is the SI one divided by ``scale``, so the stiffness
takes k as is, a boundary mass ``h scale``, a boundary load ``scale``
times its SI density and a volume load ``scale ** 2``; a transient's mass
takes ``rho c_p scale ** 2`` (`ThermalRule._prepare`'s ``mass``).

The base class `ThermalRule` holds what every thermal problem shares (the
domain, the boundary data, the space, the operator and loads, the
source-free temperature); `SteadyThermalRule` adds the sink check and the
steady solve. A transient rule reuses the same operator with a mass term
on one handle.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from gfem import _objective
from gfem._adjoint.solve_vjp import adjoint_tolerance, input_kind
from gfem._boundaries import (
    Convection,
    FixedTemperature,
    HeatFlux,
    check_thermal_values,
)
from gfem._checks import is_traced
from gfem._engine import heat_flow, quadrature
from gfem._engine.cache import Entry, _alone, retain_budget
from gfem._engine.coefficients import element_values, p0, surface_p0
from gfem._engine.domain import boundary_region
from gfem._engine.fields import (
    dirichlet_h1,
    face_keys,
    fe_space,
    h1_key,
    support_region,
    surface_triangles,
    tetrahedron_face_keys,
    vertex_dofs,
)
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.scalar_plans import (
    PooledKeptHandle,
    digest,
    plan_site,
    prepare_pooled,
    release,
)
from gfem._engine.sensitivity import (
    FACE_LOAD_KIND,
    LOAD_KIND,
    Derivative,
    FieldPair,
    Integrated,
    LoadField,
    Term,
    element_integrals,
    is_thermal_value,
)
from gfem._engine.solve_request import SolveRequest
from gfem._linalg import Handle, Kind, Operator, ResolvedLinearSolver, report
from gfem._linalg.base import free_mask
from gfem._pytree import register_fields
from gfem.errors import (
    MeshRefinementUnsupportedError,
    MissingMaterialPropertyError,
    RegionNotFoundError,
)
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import SteadyThermal, TransientThermal
from gfem.results import Field, SteadyThermalResult, TransientThermalResult
from gfem.thermal import (
    Trace,
    _FieldTrace,
    _HeatTrace,
    _tet_volumes,
    _triangle_areas,
)

__all__ = ["Loads", "SteadyThermalRule", "ThermalRule", "TransientThermalRule"]

#: The linear solver for ``linear_solver="auto"``.
DEFAULT_SOLVER: Kind = "direct"
#: The physics key of the model's solve domains and exclusions.
_PHYSICS = "thermal"


def source_density(problem: SteadyThermal | TransientThermal, mesh: Mesh) -> jax.Array:
    """The summed power density of the problem's heat sources per tetrahedron.

    W/m^3 in `gfem.Mesh.tetrahedra` order, in JAX: linear in each source's
    amount. Zero without a source.
    """
    density = jnp.zeros(len(mesh.tetrahedra), jnp.float64)
    for source in problem.heat:
        density = density + source._density_on(mesh)  # noqa: SLF001
    return density


def source_face_density(
    problem: SteadyThermal | TransientThermal, mesh: Mesh
) -> jax.Array:
    """The summed face power density of the problem's heat sources per triangle.

    W/m² in `gfem.Mesh.triangles` order, in JAX: linear in each source's
    amount. Zero without sheet heat.
    """
    density = jnp.zeros(len(mesh.triangles), jnp.float64)
    for source in problem.heat:
        density = density + source._face_density_on(mesh)  # noqa: SLF001
    return density


@dataclasses.dataclass(frozen=True, eq=False)
class Loads:
    """A thermal problem as `gfem._adjoint.solve_vjp` sees it.

    Its differentiable leaves only: every heat source resolved in JAX to one
    summed power density per tetrahedron and one summed face density per
    triangle, and for a transient the waveform scales and the traces (whose
    leaves are a section's thermo-optic weights). The rule reads everything
    else from the problem it was built with (`ThermalRule.spec`). The input
    kinds are ``("problem", ".density")``
    (`gfem._engine.sensitivity.LOAD_KIND`), ``("problem", ".face_density")``
    (`gfem._engine.sensitivity.FACE_LOAD_KIND`), ``("problem", ".scales")``
    and ``("problem", ".traces['<name>']...")``.

    Attributes:
        density: W/m^3 per tetrahedron (`gfem.Mesh.tetrahedra` order).
        face_density: W/m² per triangle (`gfem.Mesh.triangles` order), or
            ``None`` (no face heat).
        scales: The waveform's scales, or ``None`` (steady).
        traces: The traces by name, or ``None`` (steady).
    """

    density: Any
    face_density: Any = None
    scales: Any = None
    traces: Any = None


register_fields(Loads, ("density", "face_density", "scales", "traces"))

#: The input kinds of the sources' densities: one adjoint load field
#: serves both (`LoadField`).
LOAD_KINDS = frozenset({LOAD_KIND, FACE_LOAD_KIND})
#: The input kinds of the sources alone: their gradient needs the step
#: adjoints only, so no trajectory is kept for them.
SOURCE_KINDS = LOAD_KINDS | {("problem", ".scales")}


def _stride(steps: int) -> int:
    """The checkpoint stride of a run of ``steps`` steps: ceil(sqrt(steps))."""
    return max(1, math.ceil(math.sqrt(steps)))


def check_properties(model: Model, active: np.ndarray, reads: tuple[str, ...]) -> None:
    """Raise when a region of the thermal domain lacks a property read.

    A region with no material lacks them all. Every missing (region,
    property) pair is listed at once, regions in mesh order.

    Raises:
        MissingMaterialPropertyError: Some region of the domain has no
            value (or a NaN value) for a property in ``reads``.
    """
    mesh = model.mesh
    values = {name: element_values(model, name) for name in reads}
    missing = []
    for region in mesh.volumes:
        tets = mesh.elements(region)
        tets = tets[active[tets]]
        if not tets.size:
            continue
        missing += [
            (region, name) for name in reads if np.isnan(values[name][tets]).any()
        ]
    if missing:
        raise MissingMaterialPropertyError(missing)


def pieces(mesh: Mesh, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The connected pieces of the active tetrahedra, through shared faces.

    Returns:
        The active tetrahedra (indices) and the piece label of each.
    """
    index = np.flatnonzero(active)
    keys = tetrahedron_face_keys(mesh, index)
    _, face = np.unique(keys, return_inverse=True)
    rows = np.repeat(np.arange(len(index)), 4)
    graph = scipy.sparse.coo_matrix(
        (np.ones(rows.size), (rows, len(index) + face.ravel())),
        shape=(len(index) + face.max() + 1,) * 2,
    )
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return index, label[: len(index)]


def _loose(model: Model, active: np.ndarray, sinks: tuple[str, ...]) -> np.ndarray:
    """Per tetrahedron: in a connected piece of the domain that touches no sink."""
    mesh = model.mesh
    index, label = pieces(mesh, active)
    sink_keys = face_keys(surface_triangles(mesh, sinks), len(mesh.imported_nodes))
    faces = tetrahedron_face_keys(mesh, index)
    touching = np.isin(faces, sink_keys).reshape(-1, 4).any(axis=1)
    cooled = np.unique(label[touching])
    loose = np.zeros(len(mesh.tetrahedra), bool)
    loose[index[~np.isin(label, cooled)]] = True
    return loose


def _region_of(mesh: Mesh, tets: np.ndarray) -> str:
    """The first volume region (mesh order) holding one of ``tets``."""
    return next(r for r in mesh.volumes if np.isin(mesh.elements(r), tets).any())


def _check_sinks(model: Model, active: np.ndarray, sinks: tuple[str, ...]) -> None:
    """Every connected piece of the thermal domain needs a heat sink.

    Raises:
        ValueError: A piece touches no `FixedTemperature` or `Convection`
            surface; the message names a region of it.
    """
    loose = _loose(model, active, sinks)
    if not loose.any():
        return
    where = _region_of(model.mesh, np.flatnonzero(loose))
    raise ValueError(
        f"SteadyThermal: the part of the thermal domain that holds region "
        f"{where!r} ({int(loose.sum())} elements) touches no heat sink, so its "
        "steady temperature is undetermined. Give it a gfem.FixedTemperature or a "
        "gfem.Convection on one of its surfaces, or exclude it from thermal "
        f'problems (excluded={{"thermal": ({where!r},)}}).'
    )


def _watts(mesh: Mesh, power: float) -> str:
    """``power`` (W, a slab total on a cross-section) as the user reads it."""
    if mesh.section_thickness is None:
        return f"{power:.3g} W"
    return f"{power / mesh.section_thickness:.3g} W/m"


def _check_heat_inside(
    mesh: Mesh,
    active: np.ndarray,
    density: np.ndarray,
    face: np.ndarray | None = None,
) -> None:
    """Raise for heat outside the thermal domain, naming where and how much.

    Volume heat on a tetrahedron outside the domain raises with its region;
    face heat on a triangle with no tetrahedron of the domain on either side
    raises with its surface. A face with one side in the domain is fine: its
    heat enters there.

    Raises:
        ValueError: A heat source heats an element excluded from thermal
            problems, or a face with no thermal side.
    """
    outside = (np.asarray(density) != 0) & ~active
    if outside.any():
        where = next(r for r in mesh.volumes if outside[mesh.elements(r)].any())
        on = np.zeros_like(outside)
        on[mesh.elements(where)] = outside[mesh.elements(where)]
        power = float(np.sum(np.asarray(density)[on] * _tet_volumes(mesh)[on]))
        raise ValueError(
            f"A heat source puts {_watts(mesh, power)} on region {where!r}, which "
            "is excluded from thermal problems, so the heat has nowhere to go. "
            f"Heat a region of the thermal domain, or keep {where!r} in it (drop "
            'it from excluded={"thermal": ...}).'
        )
    if face is None:
        return
    neighbours = mesh._topology.surface_neighbours  # noqa: SLF001
    sides = np.where(neighbours >= 0, active[np.maximum(neighbours, 0)], False)
    loose = (np.asarray(face) != 0) & ~sides.any(axis=1)
    if not loose.any():
        return
    where = next(s for s in mesh.surfaces if loose[mesh.elements(s)].any())
    on = np.zeros_like(loose)
    on[mesh.elements(where)] = loose[mesh.elements(where)]
    power = float(np.sum(np.asarray(face)[on] * _triangle_areas(mesh)[on]))
    raise ValueError(
        f"A heat source puts {_watts(mesh, power)} on surface {where!r}, which "
        "touches no region of the thermal domain, so the heat has nowhere to go. "
        f"Keep a region next to {where!r} in the thermal domain (drop it from "
        'excluded={"thermal": ...}).'
    )


@dataclasses.dataclass(frozen=True)
class _Boundaries:
    """The model's thermal conditions, as static data (values in SI)."""

    fixed: tuple[tuple[str, float], ...]
    convection: tuple[tuple[str, float, float], ...]
    flux: tuple[tuple[str, float], ...]
    #: Every `HeatFlux` surface, a zero flux included (it is still a boundary).
    flux_surfaces: tuple[str, ...] = ()
    #: Whether a value is traced: the values are then placeholders and the
    #: structure that depends on them (a zero flux, a uniform temperature)
    #: is decided on the host model of each forward.
    traced: bool = False

    @classmethod
    def of(cls, model: Model) -> _Boundaries:
        """The conditions of ``model``, a `HeatFlux` of value 0 dropped.

        A traced value keeps its surface and is never dropped or compared.
        """
        items = model.boundaries
        traced = is_traced(items)

        def value(v: Any) -> Any:
            return v if traced else float(np.asarray(v))

        return cls(
            traced=traced,
            flux_surfaces=tuple(b.surface for b in items if isinstance(b, HeatFlux)),
            fixed=tuple(
                (b.surface, value(b.temperature))
                for b in items
                if isinstance(b, FixedTemperature)
            ),
            convection=tuple(
                (b.surface, value(b.coefficient), value(b.ambient))
                for b in items
                if isinstance(b, Convection)
            ),
            flux=tuple(
                (b.surface, value(b.flux))
                for b in items
                if isinstance(b, HeatFlux) and (traced or value(b.flux) != 0)
            ),
        )

    @property
    def dirichlet(self) -> tuple[str, ...]:
        """The fixed-temperature surfaces, sorted."""
        return tuple(sorted({s for s, _ in self.fixed}))

    @property
    def sinks(self) -> tuple[str, ...]:
        """The surfaces that take heat out: fixed temperatures and convection."""
        return tuple(sorted({*self.dirichlet, *(s for s, _, _ in self.convection)}))

    @property
    def uniform(self) -> float | None:
        """The one temperature of every fixed value and ambient, if any.

        ``None`` when they differ, when a heat flux is given (then the
        source-free temperature is not uniform), or with no value at all.
        """
        if self.traced:
            return None
        values = {t for _, t in self.fixed} | {a for _, _, a in self.convection}
        if len(values) != 1 or self.flux:
            return None
        return next(iter(values))


class ThermalRule:
    """What every thermal rule shares: domain, boundaries, space, operator.

    Construction runs the static checks (no array values needed).
    Subclasses name their `reads`, `form` and problem.
    """

    physics = _PHYSICS
    #: The material properties the operator reads (subclasses extend).
    reads: ClassVar[tuple[str, ...]] = ("thermal_conductivity",)
    #: The problem name in solver logs and messages.
    problem_name: ClassVar[str] = "thermal"

    def __init__(
        self,
        model: Model,
        problem: SteadyThermal | TransientThermal,
        options: SolverOptions,
    ) -> None:
        """Check the model and problem (see the problem's docstring).

        Raises:
            NotImplementedError: The model has a periodic pair, or the
                device is a GPU.
            MissingMaterialPropertyError: A region of the thermal domain
                lacks a property the problem reads.
            ValueError: A heat source heats an excluded region.
        """
        if options.device != "cpu":
            # no GPU smoke test has run for thermal problems
            raise NotImplementedError(
                f"{type(problem).__name__} has no GPU path in this version of "
                f"gfem (device={options.device!r}): solve with device='cpu'."
            )
        self.problem = problem
        #: The problem as built: the static values (a host copy's `problem`
        #: is its `Loads`).
        self.spec = problem
        self.options = options
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem=self.problem_name
        )
        periodic = model._periodic  # noqa: SLF001
        if periodic:
            raise NotImplementedError(
                f"{type(problem).__name__} does not support periodic pairs in this "
                f"version of gfem, but the model has {periodic[0].label}. Model the "
                "device without periodic pairs for thermal problems."
            )
        mesh = model.mesh
        domain = model._domain_of(_PHYSICS)  # noqa: SLF001
        #: Per tetrahedron, whether it is in the thermal domain.
        self.active = domain.active
        #: Volume entities excluded from thermal problems.
        self._excluded = domain.excluded
        #: Volume regions of the space (with an element in the domain), mesh order.
        self.regions = tuple(
            r for r in mesh.volumes if self.active[mesh.elements(r)].any()
        )
        self.boundaries = _Boundaries.of(model)
        self._mesh = mesh
        if not is_traced(tuple(model.materials.values())):
            check_properties(model, self.active, self.reads)
        if not is_traced(problem):
            # concrete sources: evaluated now, even inside a traced function
            with jax.ensure_compile_time_eval():
                density = np.asarray(source_density(problem, mesh))
                face = np.asarray(source_face_density(problem, mesh))
            _check_heat_inside(mesh, self.active, density, face)

    def _bind(self, model: Model) -> None:
        """Read the boundary values of the forward's host ``model``.

        A model built under `jax.grad`, `jax.jit` or `jax.vmap` reaches the
        rule with traced values, so each forward and backward checks the values
        and reads the structure they decide (a zero flux, a uniform
        temperature) from the host model here.

        Raises:
            ValueError: A value is not finite, a temperature is not > 0 K, a
                heat transfer coefficient is not > 0, or two fixed
                temperatures with different values share a mesh node.
        """
        for item in model.boundaries:
            if isinstance(item, FixedTemperature | Convection | HeatFlux):
                check_thermal_values(item)
        model._check_fixed_temperatures()  # noqa: SLF001
        self.boundaries = _Boundaries.of(model)

    def vjp_problem(self, problem: Any) -> Loads:
        """The problem's differentiable leaves, resolved in JAX (`Loads`)."""
        return Loads(
            density=source_density(problem, self._mesh),
            face_density=source_face_density(problem, self._mesh),
        )

    def _checked_loads(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray | None]:
        """This call's densities on the host, after `_check_heat_inside`."""
        loads = self._loads()
        density = np.asarray(loads.density)
        face = None if loads.face_density is None else np.asarray(loads.face_density)
        _check_heat_inside(mesh, self.active, density, face)
        return density, face

    def _loads(self) -> Loads:
        """This call's `Loads`: the host copy's problem, or resolved here."""
        problem = self.problem
        return problem if isinstance(problem, Loads) else self.vjp_problem(problem)

    # --- the discretization ---------------------------------------------------

    def _space(self, model: Model) -> tuple[ngs.FESpace, EngineMesh]:
        """The compressed H1 space on the thermal domain, fixed on the sinks."""
        return dirichlet_h1(
            model.mesh,
            _PHYSICS,
            self.regions,
            self._excluded,
            order=self.options.order,
            fixed=self.boundaries.dirichlet,
        )

    def _coefficient(
        self,
        model: Model,
        built: EngineMesh,
        name: str,
        where: np.ndarray | None = None,
    ) -> ngs.GridFunction:
        """Property ``name`` per element, zero off the domain (and off ``where``)."""
        values = element_values(model, name)
        inside = self.active if where is None else self.active & where
        return p0(built, np.where(inside, values, 0.0))

    def _prepare(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        *,
        mass: float = 0.0,
        where: np.ndarray | None = None,
        width: int = 1,
        tolerance: float | None = None,
    ) -> Handle:
        """The prepared operator ``K + mass * C`` (file units, see the module).

        ``K`` is the conduction plus convection operator; ``C`` the
        consistent heat-capacity mass ``int rho c_p u v`` (read only with
        ``mass`` != 0, as ``rho c_p scale ** 2 * mass``), on the elements
        ``where`` only when given.
        """
        form = ngs.BilinearForm(space, symmetric=True)
        self._add_conduction(form, model, built, space)
        if mass:
            form += self._capacity_term(model, built, space, where, factor=mass)
        site = plan_site(
            self.linear,
            self.options,
            topology=model.mesh._topology,  # noqa: SLF001
            comm=built.comm,
            space=space,
            space_key=h1_key(
                _PHYSICS,
                self.regions,
                self._excluded,
                self.options.order,
                self.boundaries.dirichlet,
            ),
            freedofs=space.FreeDofs(),
            tag=("thermal", float(mass), digest(where if mass else None)),
        )
        return prepare_pooled(
            self.linear,
            site,
            Operator(form, space.FreeDofs()),
            width=width,
            tolerance=tolerance,
        )

    def _add_conduction(
        self,
        form: ngs.BilinearForm,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
    ) -> None:
        """Add ``K``, conduction plus convection (file units), to ``form``."""
        order = self.options.order
        u, v = space.TnT()
        k = self._coefficient(model, built, "thermal_conductivity")
        form += k * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
        for surface, h, _ in self.boundaries.convection:
            region = boundary_region(built, (surface,))
            form += h * built.scale * u * v * quadrature.ds(order, definedon=region)

    def _capacity_term(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        where: np.ndarray | None = None,
        *,
        factor: float = 1.0,
    ) -> Any:
        """The heat-capacity mass ``factor * C`` as a form term (file units)."""
        u, v = space.TnT()
        rho = self._coefficient(model, built, "mass_density", where)
        c_p = self._coefficient(model, built, "specific_heat", where)
        dx = quadrature.dx(self.options.order)
        return factor * built.scale**2 * rho * c_p * u * v * dx

    def _boundary_load(self, built: EngineMesh, space: ngs.FESpace) -> np.ndarray:
        """The convection-ambient and heat-flux load (file units)."""
        order = self.options.order
        scale = built.scale
        v = space.TestFunction()
        form = ngs.LinearForm(space)
        terms = [(s, h * a) for s, h, a in self.boundaries.convection]
        for surface, value in (*terms, *self.boundaries.flux):
            region = boundary_region(built, (surface,))
            form += value * scale * v * quadrature.ds(order, definedon=region)
        form.Assemble()
        return form.vec.FV().NumPy().copy()

    def _source_load(
        self,
        built: EngineMesh,
        space: ngs.FESpace,
        density: np.ndarray,
        face: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""The heat sources' load :math:`\int q v + \int_S q_s v` (file units).

        A face density loads every face it is nonzero on, interior or
        exterior: the conditions of an exterior face act on its heat as on
        any boundary load.
        """
        order = self.options.order
        v = space.TestFunction()
        form = ngs.LinearForm(space)
        q = p0(built, np.asarray(density, float) * built.scale**2)
        form += q * v * quadrature.dx(order)
        if face is not None and np.any(face):
            q_s = surface_p0(built, np.asarray(face, float) * built.scale)
            form += q_s * v * quadrature.ds(order)
        form.Assemble()
        return form.vec.FV().NumPy().copy()

    def _lift(self, built: EngineMesh, space: ngs.FESpace) -> np.ndarray:
        """The fixed temperatures on their DOFs, zero elsewhere."""
        gf = ngs.GridFunction(space)
        fixed = self.boundaries.fixed
        if fixed:
            values = built.mesh.BoundaryCF(dict(fixed), default=ngs.CF(0.0))
            region = boundary_region(built, self.boundaries.dirichlet)
            gf.Set(values, ngs.BND, definedon=region)
        return gf.vec.FV().NumPy().copy()

    def _boundary_data(
        self, built: EngineMesh, space: ngs.FESpace
    ) -> tuple[np.ndarray, np.ndarray]:
        """The source-free temperature's lift and load (`_lift`, `_boundary_load`).

        T0 = lift + A^-1 (load - A lift) on the free DOFs, for the steady
        operator A.
        """
        return self._lift(built, space), self._boundary_load(built, space)

    def _constant(self, space: ngs.FESpace, value: float) -> np.ndarray:
        """The coefficients of a uniform temperature ``value``.

        Exactly: ``value`` on the vertex DOFs, zero on the others (an
        interpolation leaves 1e-11 K of round-off on them, which a heat
        flow's residual reads).
        """
        out = np.zeros(space.ndof)
        out[vertex_dofs(space)] = value
        return out

    def _solve(self, handle: Handle, rhs: np.ndarray) -> np.ndarray:
        x, its = handle.solve_counted(rhs)
        report.solve_step(
            handle, problem=self.problem_name, direction="forward", iterations=its
        )
        return x

    # --- the static layout ----------------------------------------------------

    def still_mesh(self) -> Mesh:
        """The mesh at its imported nodes: the topology without traced values."""
        mesh = self._mesh
        return Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001

    def _ndof(self) -> int:
        with task_manager(self.options.threads):
            _, space, _ = fe_space(
                self.still_mesh(),
                self.regions,
                self._excluded,
                "h1",
                self.options.order,
            )
        return space.ndof

    def heat_order(self) -> np.ndarray:
        """The tetrahedron of each DOF of the heat field's L2 order-0 space."""
        still = self.still_mesh()
        with task_manager(self.options.threads):
            built, space, _ = fe_space(still, self.regions, self._excluded, "l2", 0)
            gf = ngs.GridFunction(space)
            gf.Set(p0(built, np.arange(len(still.tetrahedra), dtype=float)))
        return np.rint(gf.vec.FV().NumPy()).astype(np.int64)

    def _field(self, coefficients: Any, quantity: str) -> Field:
        return Field(
            coefficients=jnp.asarray(coefficients),
            mesh=self._mesh,
            quantity=quantity,
            unit="K",
            space="h1",
            order=self.options.order,
            support=self.regions,
            _excluded=self._excluded,
        )

    def _heat(self) -> tuple[Field, jax.Array]:
        """The volume heat field and the sources' total power (W), in JAX.

        The power holds the face heat too.
        """
        density = source_density(self.spec, self._mesh)
        face = source_face_density(self.spec, self._mesh)
        heat = Field(
            coefficients=density[self.heat_order()],
            mesh=self._mesh,
            quantity="q",
            unit="W/m^3",
            space="l2",
            order=0,
            support=self.regions,
            _excluded=self._excluded,
        )
        power = jnp.sum(density * _tet_volumes(self._mesh)) + jnp.sum(
            face * _triangle_areas(self._mesh)
        )
        return heat, power

    # --- the differentiation interface ----------------------------------------

    def role(self, kind: tuple) -> str:
        """``"read"`` if the thermal solve depends on input ``kind``.

        The thermal properties in `reads`, the thermal boundary values, the
        heat sources (problem leaves) and the node coordinates are read;
        every other material property (the electromagnetic ones,
        ``conductivity``) and every electromagnetic boundary or port value
        is not: an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        if is_thermal_value(kind):
            return "read"
        if kind[0] in ("law", "boundary", "port"):
            return "unread"
        return "read"

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the {self.problem_name} output {name!r} in "
            "this version. Wrap it in jax.lax.stop_gradient to use its value "
            "without its derivative."
        )

    def _domain(self, model: Model, built: EngineMesh) -> ngs.Region:
        return support_region(built, model.mesh, self.regions, self._excluded)

    def _values(self, model: Model, name: str) -> np.ndarray:
        """Property ``name`` per tetrahedron, zero off the domain."""
        return np.where(self.active, element_values(model, name), 0.0)

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:  # noqa: ARG002
        """The conduction form ``int k grad u . grad v`` (file units) as a `Term`.

        The convection terms read no material: they have none. A transient
        adds the heat capacity (`TransientThermalRule.terms`).
        """
        _, built = self._space(model)
        return [
            Term(
                "conduction",
                "gradient",
                coefficient=self._coefficient(model, built, "thermal_conductivity"),
                derivatives={
                    "thermal_conductivity": Derivative(1.0, holomorphic=False)
                },
            )
        ]

    def _adjoint(self, handle: Handle, rhs: np.ndarray, *, reused: bool) -> np.ndarray:
        """``A^-1 rhs`` on the free DOFs, at the adjoint tolerance."""
        x, its = handle.solve_counted(rhs, tolerance=adjoint_tolerance(self.options))
        report.solve_step(
            handle,
            problem=self.problem_name,
            direction="backward",
            iterations=its,
            reused=reused,
        )
        return x

    def _source_free_pair(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        load: np.ndarray,
        t0: np.ndarray,
        *,
        loose: np.ndarray | None = None,
        handle: Handle | None = None,
    ) -> FieldPair:
        """The T0 pair ``(-lambda0, T0)`` of a cotangent ``load`` on T0.

        ``A0 T0 = boundary load`` on the free DOFs, for the steady operator
        A0 (with the heat capacity of a sinkless part, ``loose``, where the
        source-free temperature is its uniform start and the capacity's own
        pair vanishes), so ``load . dT0 = -lambda0^T dA0 T0`` with ``A0
        lambda0 = load``. One more solve, on ``handle`` if given (the steady
        operator, which a steady problem's forward already prepared), else on
        its own prepared operator.
        """
        if handle is not None:
            lam = self._adjoint(handle, load[:, None], reused=True)
        else:
            where = None if loose is None or not loose.any() else loose
            own = self._prepare(
                model,
                built,
                space,
                mass=0.0 if where is None else 1.0,
                where=where,
                tolerance=adjoint_tolerance(self.options),
            )
            try:
                lam = self._adjoint(own, load[:, None], reused=False)
            finally:
                release(own)
        return FieldPair(
            space=space,
            left=-lam,
            right=t0[:, None],
            weights={"conduction": 1.0},
            order=self.options.order,
            domain=self._domain(model, built),
        )


class SteadyThermalRule(ThermalRule):
    """Host-side forward of a `SteadyThermal` problem on a `Model`."""

    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "steady_thermal"
    #: The problem as built.
    spec: SteadyThermal
    problem_name = "steady thermal"
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs: frozenset[str] = frozenset({"temperature", "rise"})

    def __init__(
        self, model: Model, problem: SteadyThermal, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.SteadyThermal`).

        Raises:
            NotImplementedError: The model has a periodic pair, or the
                device is a GPU.
            MissingMaterialPropertyError: A region of the thermal domain has
                no ``thermal_conductivity``.
            ValueError: A piece of the thermal domain has no heat sink; a
                heat source heats an excluded region.
        """
        if not isinstance(problem, SteadyThermal):
            raise TypeError(f"Expected gfem.SteadyThermal, got {problem!r}.")
        super().__init__(model, problem, options)
        _check_sinks(model, self.active, self.boundaries.sinks)

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve the source-free temperature and the rise on one operator.

        Returns:
            ``{"temperature": (ndof,), "rise": (ndof,)}`` in K, and, if
            ``keep``, the cache entry (the prepared operator in a
            `KeptHandle`, ``fields`` the two solutions) or ``None``.
        """
        del request  # a steady solve keeps the same whatever is wanted
        self._bind(model)
        mesh = model.mesh
        check_properties(model, self.active, self.reads)
        density, face = self._checked_loads(mesh)
        uniform = self.boundaries.uniform
        with task_manager(self.options.threads):
            space, built = self._space(model)
            source = self._source_load(built, space, density, face)
            # T0 needs a solve unless it is one uniform value
            boundary = (
                None if uniform is not None else self._boundary_data(built, space)
            )
            handle = self._prepare(
                model,
                built,
                space,
                width=1 if boundary is None else 2,
                tolerance=self.options.tolerance,
            )
            if boundary is None:
                rise = self._solve(handle, source[:, None])[:, 0]
                t0 = self._constant(space, uniform or 0.0)
            else:
                lift, load = boundary
                columns = np.stack([source, load - handle.apply(lift)], axis=1)
                solved = self._solve(handle, columns)
                rise, t0 = solved[:, 0], lift + solved[:, 1]
        temperature = t0 + rise
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=PooledKeptHandle(handle),
                fields={"temperature": temperature, "rise": rise},
            )
        else:
            release(handle)
        return {"temperature": temperature, "rise": rise}, entry

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        del model
        n = self._ndof()
        return {
            "temperature": jax.ShapeDtypeStruct((n,), jnp.float64),
            "rise": jax.ShapeDtypeStruct((n,), jnp.float64),
        }

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
    ) -> list[FieldPair | LoadField]:
        """Fold the temperature and rise cotangents into pairs and a load.

        ``rise = A^-1 F`` on the free DOFs, so with ``A lambda = ct_T +
        ct_rise`` the conductivity pair is ``(-lambda, rise)`` and the
        source density's load is ``lambda`` (`LoadField`, ``F`` reads ``q
        scale ** 2`` in file units). A temperature cotangent adds the T0
        pair (`_source_free_pair`) unless the boundaries share one
        temperature (T0 is then that constant). Solves on the forward's
        kept operator; an entry that lost it prepares it again. ``wanted``
        holds the wanted input kinds.
        """
        self._bind(model)
        wanted = wanted or frozenset()
        conduction = "thermal_conductivity" in {
            k[2] for k in wanted if k[0] == "material"
        }
        order = self.options.order
        with task_manager(self.options.threads):
            space, built = self._space(model)
            free = free_mask(space.FreeDofs(), space.ndof)
            ct_t = cotangents.get("temperature")
            load = np.zeros(space.ndof)
            for name in ("temperature", "rise"):
                if name in cotangents:
                    load = load + np.asarray(cotangents[name], dtype=float)
            load[~free] = 0.0
            handle = None if entry.handle is None else entry.handle.handle
            kept = handle is not None
            if handle is None:
                tolerance = adjoint_tolerance(self.options)
                handle = self._prepare(model, built, space, tolerance=tolerance)
                entry.handle = PooledKeptHandle(handle)
            lam = self._adjoint(handle, load[:, None], reused=kept)[:, 0]
            domain = self._domain(model, built)
            rise = entry.fields["rise"]
            pairs: list[FieldPair | LoadField] = []
            if conduction:
                pairs.append(
                    FieldPair(
                        space=space,
                        left=-lam[:, None],
                        right=rise[:, None],
                        weights={"conduction": 1.0},
                        order=order,
                        domain=domain,
                    )
                )
                if ct_t is not None and self.boundaries.uniform is None:
                    t0 = entry.fields["temperature"] - rise
                    ct0 = np.where(free, np.asarray(ct_t, dtype=float), 0.0)
                    pairs.append(
                        self._source_free_pair(
                            model, built, space, ct0, t0, handle=handle
                        )
                    )
            if wanted & LOAD_KINDS:
                pairs.append(LoadField(space, lam, order, domain, length_power=2))
            self._explicit = self._boundary_terms(
                model, built, space, handle, entry, lam, ct_t, wanted
            )
        return pairs

    def explicit_problem(self) -> dict[tuple, np.ndarray]:
        """The thermal boundary values' cotangents of the last backward."""
        return getattr(self, "_explicit", {})

    def _surface_integral(
        self, built: EngineMesh, space: ngs.FESpace, surface: str, *fields: Any
    ) -> float:
        r""":math:`\int_S \prod f\,\mathrm{d}s` of coefficient vectors, file units."""
        region = boundary_region(built, (surface,))
        integrand: Any = built.scale
        for vector in fields:
            gf = ngs.GridFunction(space)
            gf.vec.FV().NumPy()[:] = vector
            integrand = integrand * gf
        ds = quadrature.ds(self.options.order, definedon=region)
        return float(ngs.Integrate(integrand * ds, built.mesh))

    def _boundary_terms(  # noqa: PLR0917
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        handle: Handle,
        entry: Entry,
        lam: np.ndarray,
        ct_t: np.ndarray | None,
        wanted: frozenset,
    ) -> dict[tuple, np.ndarray]:
        r"""The wanted thermal boundary values' cotangents.

        With :math:`A = K + \sum_S h_S M_S`, :math:`T = T_0 + \Delta T` and
        :math:`A\,\Delta T = F` on the free DOFs, :math:`\lambda` the adjoint
        of the temperature and rise cotangents and :math:`\lambda_0 =
        A^{-1} c_T` that of the temperature alone (zero on the fixed DOFs):

        .. math::

            \frac{dJ}{dh_S} = -\lambda^T M_S \Delta T
                - \lambda_0^T M_S (T_0 - T_{\mathrm{amb},S} \mathbf 1),\quad
            \frac{dJ}{dT_{\mathrm{amb},S}} = h_S \lambda_0^T M_S \mathbf 1,

        .. math::

            \frac{dJ}{dq_S} = \lambda_0^T M_S \mathbf 1,\quad
            \frac{dJ}{dT_s} = c_T \cdot \chi_s - \lambda_0^T A \chi_s,

        with :math:`\chi_s` the lift of a unit temperature on fixed surface
        :math:`s`. The rise reads only :math:`h`, so it adds nothing to the
        other three.
        """
        kinds = {k for k in wanted if is_thermal_value(k)}
        if not kinds:
            return {}
        free = free_mask(space.FreeDofs(), space.ndof)
        rise = entry.fields["rise"]
        t0 = entry.fields["temperature"] - rise
        lam0 = np.zeros(space.ndof)
        direct = None
        if ct_t is not None and np.any(ct_t):
            direct = np.asarray(ct_t, dtype=float)
            load = np.where(free, direct, 0.0)
            lam0 = self._adjoint(handle, load[:, None], reused=True)[:, 0]
        ones = self._constant(space, 1.0)
        out: dict[tuple, np.ndarray] = {}
        fixed_names = [
            b.surface for b in model.boundaries if isinstance(b, FixedTemperature)
        ]
        for kind in kinds:
            _, index, name = kind
            item = model.boundaries[index]
            if name == "coefficient":
                ambient = float(np.asarray(item.ambient))
                value = -self._surface_integral(
                    built, space, item.surface, lam, rise
                ) - self._surface_integral(
                    built, space, item.surface, lam0, t0 - ambient * ones
                )
            elif name == "ambient":
                h = float(np.asarray(item.coefficient))
                value = h * self._surface_integral(
                    built, space, item.surface, lam0, ones
                )
            elif name == "flux":
                value = self._surface_integral(built, space, item.surface, lam0, ones)
            else:
                value = self._fixed_term(
                    built, space, handle, item.surface, fixed_names, lam0, direct
                )
            out[kind] = np.asarray(value)
        return out

    def _fixed_term(  # noqa: PLR0917
        self,
        built: EngineMesh,
        space: ngs.FESpace,
        handle: Handle,
        surface: str,
        fixed: list[str],
        lam0: np.ndarray,
        direct: np.ndarray | None,
    ) -> float:
        r""":math:`c_T \cdot \chi_s - \lambda_0^T A \chi_s` of fixed ``surface``."""
        values = {name: (1.0 if name == surface else 0.0) for name in fixed}
        gf = ngs.GridFunction(space)
        field = built.mesh.BoundaryCF(values, default=ngs.CF(0.0))
        region = boundary_region(built, self.boundaries.dirichlet)
        gf.Set(field, ngs.BND, definedon=region)
        chi = gf.vec.FV().NumPy().copy()
        read = 0.0 if direct is None else float(direct @ chi)
        return read - float(lam0 @ handle.apply(chi[:, None])[:, 0])

    def result(self, outputs: dict) -> SteadyThermalResult:
        """The public result from the forward outputs."""
        heat, power = self._heat()
        # face heat, kept for the heat flow's residual when a source has any
        sheets = any(s._face_density is not None for s in self.spec.heat)  # noqa: SLF001
        return SteadyThermalResult(
            temperature=self._field(outputs["temperature"], "T"),
            rise=self._field(outputs["rise"], "dT"),
            heat=heat,
            power=power,
            _thickness=self._mesh.section_thickness,
            _face_heat=(source_face_density(self.spec, self._mesh) if sheets else None),
        )


# --- transient ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _HeatRow:
    r"""A heat-out trace's weights on one model (`gfem.Trace.heat_out`).

    The trace at :math:`t_n` is

    .. math::

        Q_n = \left[ a^T S_n - e^T (u_n - u_0) + t_n Q_0
            + \sigma_n P \right] / d

    for the running quadrature :math:`S_n` of the rise (`Stepper.quadrature`),
    :math:`a` the balance's temperature weights, :math:`e` the side's heat
    capacity (`gfem._engine.heat_flow.stored`), :math:`Q_0` the source-free
    temperature's flow, :math:`\sigma_n = \sum_{m<n} s_m\, dt` the
    waveform's integral, :math:`P` the side's source power at scale 1 and
    :math:`d` the slab thickness (per length) or 1.

    Attributes:
        index: The trace's column among the problem's traces.
        balance: The side's steady `gfem._engine.heat_flow.Balance`.
        stored: :math:`e` per DOF, J/K.
        divisor: :math:`d`.
    """

    index: int
    balance: heat_flow.Balance
    stored: np.ndarray
    divisor: float

    def power(self, density: np.ndarray, face: np.ndarray | None) -> float:
        """:math:`P`: the sources' power the side reads, W."""
        out = float(self.balance.source @ density)
        if face is not None:
            out += float(self.balance.face_source @ face)
        return out

    def background(self, t0: np.ndarray) -> float:
        """:math:`Q_0`: the source-free temperature's flow, W."""
        return self.balance.flow(t0, np.zeros(len(self.balance.source)))


class _HeatRecorder:
    """The heat-out traces of one forward: the running sums of `_HeatRow`."""

    def __init__(  # noqa: PLR0917 - one forward's state, built once
        self,
        rows: list[_HeatRow],
        stepper: Stepper,
        dt: float,
        t0: np.ndarray,
        start: np.ndarray,
        density: np.ndarray,
        face: np.ndarray | None,
    ) -> None:
        self.rows = rows
        self.quadrature = stepper.quadrature
        self.dt = dt
        self.start = start
        #: S_n and sigma_n
        self.integral = np.zeros_like(start)
        self.sigma = 0.0
        self.temperature = [r.balance.temperature for r in rows]
        self.background = [r.background(t0) for r in rows]
        self.power = [r.power(density, face) for r in rows]

    def step(
        self, u: np.ndarray, stage: np.ndarray | None, nxt: np.ndarray, load: float
    ) -> None:
        r"""Add the step from ``u`` to ``nxt`` to the sums.

        ``load`` is the step's scale times :math:`\Delta t`.
        """
        if not self.rows:
            return
        b0, b1, b2 = self.quadrature
        dt = self.dt
        increment = b0 * u + b2 * nxt
        if stage is not None:
            increment = increment + b1 * stage
        self.integral = self.integral + dt * increment
        self.sigma += load

    def record(self, out: np.ndarray, u: np.ndarray, t: float) -> None:
        """Write each trace's ``Q`` at time ``t`` (state ``u``) into ``out``."""
        for row, a, q0, p in zip(
            self.rows, self.temperature, self.background, self.power, strict=True
        ):
            value = a @ self.integral - row.stored @ (u - self.start)
            out[row.index] = (value + t * q0 + self.sigma * p) / row.divisor


class _HeatAdjoint:
    r"""The heat-out traces' cotangents in a reverse sweep (`_HeatRow`).

    With :math:`c_n` a trace's cotangent over its divisor and
    :math:`G_m = \sum_{n>m} c_n` (the cotangent of step :math:`m`'s
    increment of :math:`S`), and :math:`b_0, b_1, b_2` the step's
    quadrature weights (`Stepper.quadrature`):

    - :math:`u_n` reads :math:`-e c_n` (:math:`+e G_0` for :math:`u_0`),
      :math:`\Delta t\, b_2 a G_{n-1}` as the end of step :math:`n - 1` and
      :math:`\Delta t\, b_0 a G_n` as the start of step :math:`n`;
    - step :math:`m`'s stage reads :math:`\Delta t\, b_1 a G_m`;
    - :math:`T_0` reads :math:`a \sum_n c_n t_n`, step :math:`m`'s scale
      :math:`\Delta t\, P G_m` and the densities
      :math:`\sum_n c_n \sigma_n` times the balance's source weights;
    - the explicit conductivity pair is the balance's on
      :math:`\sum_n c_n (S_n + t_n T_0)` and the capacity pair :math:`-e` on
      :math:`\sum_n c_n (u_n - u_0)`, both summed as the sweep goes
      (`accumulate`).
    """

    def __init__(
        self,
        rows: list[_HeatRow],
        cotangents: np.ndarray,
        stepper: Stepper,
        dt: float,
        scales: np.ndarray,
    ) -> None:
        self.rows = rows
        self.quadrature = stepper.quadrature
        self.dt = dt
        steps = len(scales)
        self.c = [cotangents[:, r.index] / r.divisor for r in rows]
        # G_m = sum_{n > m} c_n, m = 0 .. steps - 1
        self.after = [np.cumsum(c[::-1])[::-1][1:] for c in self.c]
        self.a = [r.balance.temperature for r in rows]
        self.sigma = np.concatenate([[0.0], np.cumsum(scales * dt)])
        self.time = np.arange(steps + 1) * dt
        ndof = len(rows[0].stored) if rows else 0
        self.integral = [np.zeros(ndof) for _ in rows]
        self.change = [np.zeros(ndof) for _ in rows]

    def direct(self, n: int, ndof: int) -> np.ndarray:
        r""":math:`\partial Q / \partial u_n`, contracted with the cotangents."""
        out = np.zeros(ndof)
        b0, _, b2 = self.quadrature
        steps = len(self.time) - 1
        for row, c, after, a in zip(self.rows, self.c, self.after, self.a, strict=True):
            out -= (c[n] if n else -after[0]) * row.stored
            weight = 0.0
            if n >= 1:
                weight += b2 * after[n - 1]
            if n < steps:
                weight += b0 * after[n]
            out += self.dt * weight * a
        return out

    def stage(self, n: int, ndof: int) -> np.ndarray:
        r"""The derivative of :math:`Q` in the stage of step :math:`n`.

        Contracted with the cotangents.
        """
        out = np.zeros(ndof)
        b1 = self.quadrature[1]
        for after, a in zip(self.after, self.a, strict=True):
            out += self.dt * b1 * after[n] * a
        return out

    def source_free(self, ndof: int) -> np.ndarray:
        r""":math:`\partial Q / \partial T_0`, contracted with the cotangents."""
        out = np.zeros(ndof)
        for c, a in zip(self.c, self.a, strict=True):
            out += (c @ self.time) * a
        return out

    def scale(self, n: int, density: np.ndarray, face: np.ndarray | None) -> float:
        r""":math:`\partial Q / \partial s_n` for step :math:`n`'s scale :math:`s_n`.

        Contracted with the cotangents.
        """
        return sum(
            self.dt * after[n] * row.power(density, face)
            for row, after in zip(self.rows, self.after, strict=True)
        )

    def densities(self) -> tuple[np.ndarray, np.ndarray]:
        r"""The cotangents of the volume and the face density.

        :math:`\partial Q / \partial q` and :math:`\partial Q / \partial q_s`,
        contracted with the cotangents.
        """
        volume = sum(
            (c @ self.sigma) * row.balance.source
            for row, c in zip(self.rows, self.c, strict=True)
        )
        face = sum(
            (c @ self.sigma) * row.balance.face_source
            for row, c in zip(self.rows, self.c, strict=True)
        )
        return np.asarray(volume), np.asarray(face)

    def accumulate(
        self, n: int, u: np.ndarray, stage: np.ndarray | None, nxt: np.ndarray
    ) -> None:
        """Add step :math:`n` (``u`` to ``nxt``) to the explicit pairs' fields."""
        b0, b1, b2 = self.quadrature
        increment = b0 * u + b2 * nxt
        if stage is not None:
            increment = increment + b1 * stage
        for k, (c, after) in enumerate(zip(self.c, self.after, strict=True)):
            self.integral[k] += self.dt * after[n] * increment
            self.change[k] += c[n + 1] * nxt
            if n == 0:
                self.change[k] -= after[0] * u

    def pairs(
        self, rule: TransientThermalRule, model: Model, t0: np.ndarray, properties: set
    ) -> list[Integrated]:
        """The explicit conductivity and capacity pairs."""
        out = []
        for row, c, s, du in zip(
            self.rows, self.c, self.integral, self.change, strict=True
        ):
            if "thermal_conductivity" in properties:
                right = s + (c @ self.time) * t0
                out.append(row.balance.conductivity(rule, model, right))
            if properties & {"mass_density", "specific_heat"}:
                side = row.balance.side
                out.append(heat_flow.capacity_pair(rule, model, side, du))
        return out


#: TR-BDF2's stage fraction (Bank et al. 1985): L-stable, both stages on one matrix.
GAMMA = 2.0 - np.sqrt(2.0)


@dataclasses.dataclass
class Stepper:
    """One time step of ``C u' + K u = s(t) F`` on one prepared operator.

    The handle holds ``A = K + mass C`` (file units), with ``mass`` chosen
    so that every stage of the scheme solves with ``A``:

    - TR-BDF2 (``mass = 2 / (gamma dt)``): the trapezoid stage to
      ``t_n + gamma dt``, ``A u_g = (2 mass C - A) u_n + 2 F_n``, then the
      BDF2 stage, ``A u_n+1 = mass C (c1 u_g - c0 u_n) + F_n``, with c1 =
      1 / (gamma (2 - gamma)) and c0 = (1 - gamma) ** 2 / (gamma (2 - gamma)).
    - theta (``mass = 1 / (theta dt)``): ``A u_n+1 = (mass / theta) C u_n -
      ((1 - theta) / theta) A u_n + F_n / theta``.

    ``F_n`` is the load on [t_n, t_n+1): the waveform's value there times
    the sources' load. The stage value of TR-BDF2 is returned by `step`, so
    a reverse sweep can recompute it from ``u_n`` (`stage`; the adjoint runs
    the transposed stages on the same handle, ``A`` and ``C`` being
    symmetric).

    Attributes:
        handle: The prepared ``A``.
        capacity: ``C`` as a sparse matrix (file units).
        free: Per DOF, whether it is free (not on a fixed temperature).
        scheme: ``"tr_bdf2"`` or ``"theta"``.
        theta: The theta scheme's implicitness (read by ``"theta"``).
        mass: The ``mass`` of ``A``.
    """

    handle: Handle
    capacity: Any
    free: np.ndarray
    scheme: str
    theta: float
    mass: float
    solve: Any

    def _solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.where(self.free, rhs, 0.0)
        return self.solve(self.handle, rhs)

    @property
    def quadrature(self) -> tuple[float, float, float]:
        r"""The weights :math:`(b_0, b_1, b_2)` of a step's time integral.

        Over one step,

        .. math::

            \int u\, dt = \Delta t \left( b_0 u_n + b_1 u_\mathrm{stage}
                + b_2 u_{n+1} \right)

        is what the scheme's own stage equations integrate: TR-BDF2
        :math:`(\sqrt{2}/4, \sqrt{2}/4, 1 - \sqrt{2}/2)`, as its two stages
        give with :math:`c_1 - c_0 = 1`; theta :math:`(1 - \theta, 0, \theta)`.
        """
        if self.scheme == "tr_bdf2":
            half = 1.0 / (2.0 * (2.0 - GAMMA))
            return half, half, GAMMA / 2.0
        return 1.0 - self.theta, 0.0, self.theta

    def stage(self, u: np.ndarray, load: np.ndarray) -> np.ndarray:
        """TR-BDF2's trapezoid stage from ``u_n`` (recomputed by a reverse sweep)."""
        cu = self.capacity @ u
        return self._solve(2.0 * self.mass * cu - self.handle.apply(u) + 2.0 * load)

    def step(
        self, u: np.ndarray, load: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``u_n+1`` from ``u_n`` and the interval load (and TR-BDF2's stage)."""
        m = self.mass
        cu = self.capacity @ u
        if self.scheme == "tr_bdf2":
            g = GAMMA
            c1 = 1.0 / (g * (2.0 - g))
            c0 = (1.0 - g) ** 2 / (g * (2.0 - g))
            stage = self._solve(2.0 * m * cu - self.handle.apply(u) + 2.0 * load)
            nxt = self._solve(m * (c1 * (self.capacity @ stage) - c0 * cu) + load)
            return nxt, stage
        th = self.theta
        rhs = (m / th) * cu - ((1.0 - th) / th) * self.handle.apply(u) + load / th
        return self._solve(rhs), None


class TransientThermalRule(ThermalRule):
    """Host-side forward of a `TransientThermal` problem on a `Model`."""

    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "transient_thermal"
    #: The problem as built.
    spec: TransientThermal
    problem_name = "transient thermal"
    reads = ("thermal_conductivity", "mass_density", "specific_heat")
    #: Outputs whose cotangents the backward pass handles: every one (set per
    #: problem, the traces' names are its own).
    differentiable_outputs: frozenset[str] = frozenset()

    def __init__(
        self, model: Model, problem: TransientThermal, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.TransientThermal`).

        Raises:
            NotImplementedError: The model has a periodic pair, or the
                device is a GPU.
            MissingMaterialPropertyError: A region of the thermal domain
                lacks ``thermal_conductivity``, ``mass_density`` or
                ``specific_heat``.
            ValueError: A heat source heats an excluded region; a part of
                the thermal domain touches no heat sink and the problem has
                no ``initial_temperature``, or a `gfem.HeatFlux` heats it.
            RegionNotFoundError: A trace names an unknown region.
        """
        if not isinstance(problem, TransientThermal):
            raise TypeError(f"Expected gfem.TransientThermal, got {problem!r}.")
        super().__init__(model, problem, options)
        self.differentiable_outputs = frozenset(
            {*(f"trace:{name}" for name in problem.traces), "kept_rise", "source_free"}
        )
        #: Per tetrahedron: in a part of the domain that touches no heat sink.
        self.loose = _loose(model, self.active, self.boundaries.sinks)
        if self.loose.any():
            self._check_loose(model)
        for name, trace in problem.traces.items():
            _check_trace(model.mesh, self.regions, name, trace)
        if any(isinstance(t, _HeatTrace) for t in problem.traces.values()):
            # the sides are topology: built here on the imported nodes to
            # raise before the solve
            still = model.replace(mesh=self.still_mesh())
            self._sides(still)

    def _check_loose(self, model: Model) -> None:
        """A sinkless part needs a uniform start and no heat flux.

        Raises:
            ValueError: No ``initial_temperature``, or a `gfem.HeatFlux` on
                the sinkless part.
        """
        mesh = model.mesh
        where = _region_of(mesh, np.flatnonzero(self.loose))
        if self.spec.initial_temperature is None:
            raise ValueError(
                f"TransientThermal: the part of the thermal domain that holds region "
                f"{where!r} touches no heat sink, so it has no source-free (steady) "
                "temperature to start from. Give initial_temperature= (K, for "
                "example 293.15), or a gfem.FixedTemperature or gfem.Convection on "
                "one of its surfaces."
            )
        n_nodes = len(mesh.imported_nodes)
        faces = tetrahedron_face_keys(mesh, self.loose)
        for surface, _ in self.boundaries.flux:
            tri = surface_triangles(mesh, (surface,))
            if np.isin(face_keys(tri, n_nodes), faces).any():
                raise ValueError(
                    f"TransientThermal: gfem.HeatFlux on {surface!r} heats the part of "
                    f"the thermal domain that holds region {where!r}, which touches "
                    "no heat sink, so its source-free temperature drifts and the "
                    "rise has no meaning. Give that part a gfem.FixedTemperature or "
                    "gfem.Convection, or give the heat as a gfem.HeatSource."
                )

    def vjp_problem(self, problem: Any) -> Loads:
        """The density, the waveform scales and the traces (`Loads`)."""
        return Loads(
            density=source_density(problem, self._mesh),
            face_density=source_face_density(problem, self._mesh),
            scales=problem._scales,  # noqa: SLF001
            traces=dict(problem.traces),
        )

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:
        """The conduction form and the heat capacity ``int rho c_p u v``.

        Both in file units (the capacity's coefficient ``rho c_p scale **
        2``); ``d/d rho = c_p scale ** 2`` and ``d/d c_p = rho scale ** 2``.
        """
        out = super().terms(model, nodes=nodes)
        _, built = self._space(model)
        square = built.scale**2
        rho = self._values(model, "mass_density")
        c_p = self._values(model, "specific_heat")
        out.append(
            Term(
                "capacity",
                "value",
                coefficient=square * p0(built, rho * c_p),
                derivatives={
                    "mass_density": Derivative(lambda: square * c_p, holomorphic=False),
                    "specific_heat": Derivative(
                        lambda: square * rho, holomorphic=False
                    ),
                },
            )
        )
        return out

    def _start(
        self, space: ngs.FESpace, free: np.ndarray, t0: np.ndarray
    ) -> np.ndarray:
        """The initial rise: zero, or the initial temperature less T0."""
        start = self.spec.initial_temperature
        if start is None:
            return np.zeros_like(t0)
        return np.where(free, self._constant(space, start) - t0, 0.0)

    def _trajectory(self, handle: Handle, ndof: int, request: SolveRequest) -> str:
        """``"none"``, ``"full"`` or ``"checkpoints"``: what a kept forward stores.

        Decided once, before stepping: no trajectory when only the
        sources are wanted (their gradient reads the step adjoints alone);
        else every state when the handle and the trajectory fit
        ``retain_memory`` alone (the test the cache stores with); else a
        checkpoint every ceil(sqrt(steps)) steps, as a rerun forward does.
        """
        wanted = request.wanted
        if wanted is not None and wanted <= SOURCE_KINDS:
            return "none"
        if request.rerun:
            return "checkpoints"
        trajectory = 8 * ndof * (self.spec.steps + 1)
        held = _alone(Entry(physics=self.physics, handle=PooledKeptHandle(handle)))
        budget = retain_budget(self.options.retain_memory)
        return "full" if held + trajectory <= budget else "checkpoints"

    # --- the static layout ----------------------------------------------------

    @property
    def _grid(self) -> tuple[float, np.ndarray, np.ndarray]:
        """dt, the waveform's start indices and the kept grid indices."""
        problem = self.spec
        dt = problem.duration / problem.steps
        starts = np.array([round(t / dt) for t in problem._switches], np.int64)  # noqa: SLF001
        keep = problem.keep_fields
        if keep is True:
            kept = np.arange(problem.steps + 1)
        else:
            chosen = () if keep is False else keep
            kept = np.unique([*(round(t / dt) for t in chosen), problem.steps]).astype(
                np.int64
            )
        return dt, starts, kept

    def scales(self) -> np.ndarray:
        """The waveform's value on each step's interval, ``(steps,)``."""
        _, starts, _ = self._grid
        values = np.asarray(self._loads().scales, float)
        which = np.searchsorted(starts, np.arange(self.spec.steps), side="right")
        return np.where(which > 0, values[np.maximum(which - 1, 0)], 0.0)

    def _layout(self) -> tuple:
        return (self.regions, self._excluded, "h1", self.options.order)

    def _trace_rows(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        """``(n_traces, ndof)`` twice: each trace is ``on_t @ T + on_rise @ rise``."""
        # by the problem's names: a pytree round trip sorts the mapping's keys
        held = self._loads().traces
        traces = [held[name] for name in self.spec.traces]
        n = self._ndof()
        on_t, on_rise = np.zeros((len(traces), n)), np.zeros((len(traces), n))
        for k, trace in enumerate(traces):
            if isinstance(trace, _HeatTrace):
                continue  # recorded from the balance (`_heat_rows`)
            on_t[k], on_rise[k] = trace_rows(
                trace, mesh, self._layout(), self._field, n
            )
        return on_t, on_rise

    def _sides(self, model: Model) -> dict[int, heat_flow.Side]:
        """The heat-out traces' sides by column (`gfem._engine.heat_flow.side`).

        Raises:
            RegionNotFoundError: A named region does not exist.
            ValueError: A side does not fit (see `gfem.heat_flow`).
        """
        out = {}
        for k, trace in enumerate(self.spec.traces.values()):
            if isinstance(trace, _HeatTrace):
                who = f"Trace.{trace._name}"  # noqa: SLF001
                out[k] = heat_flow.side(self, model, trace._side, who=who)  # noqa: SLF001
        return out

    def _heat_rows(self, model: Model) -> list[_HeatRow]:
        """The heat-out traces' weights on ``model`` (`_HeatRow`)."""
        thickness = self._mesh.section_thickness
        return [
            _HeatRow(
                index=k,
                balance=heat_flow.balance(self, model, side),
                stored=heat_flow.stored(self, model, side),
                divisor=1.0 if thickness is None else float(thickness),
            )
            for k, side in self._sides(model).items()
        ]

    # --- the forward ----------------------------------------------------------

    def _source_free(
        self, model: Model, built: EngineMesh, space: ngs.FESpace
    ) -> np.ndarray:
        """The source-free temperature T0 on its own steady handle (freed after).

        No solve when the boundaries share one temperature and no part is
        sinkless. A sinkless part takes the initial temperature: its heat
        capacity is added to the steady operator there, with the uniform
        initial temperature as the matching load, which K does not see.
        """
        uniform = self.boundaries.uniform
        start = self.spec.initial_temperature
        if not self.loose.any() and uniform is not None:
            return self._constant(space, uniform)
        if self.loose.all() or not self.boundaries.sinks:
            assert start is not None  # noqa: S101 - `_check_loose` checked
            return self._constant(space, start)
        handle = self._prepare(
            model,
            built,
            space,
            mass=1.0 if self.loose.any() else 0.0,
            where=self.loose,
            tolerance=self.options.tolerance,
        )
        try:
            lift, load = self._boundary_data(built, space)
            rhs = load - handle.apply(lift)
            if self.loose.any():
                assert start is not None  # noqa: S101 - `_check_loose` checked
                capacity = _capacity(self, model, built, space, self.loose)
                rhs = rhs + capacity @ self._constant(space, start)
            return lift + self._solve(handle, rhs)
        finally:
            release(handle)

    def _stepper(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        dt: float,
        handle: Handle | None = None,
    ) -> Stepper:
        """The stepping operator of the options' time scheme.

        Prepared here, unless ``handle`` holds it.
        """
        scheme = self.options.time_scheme
        theta = self.options.theta
        mass = 2.0 / (GAMMA * dt) if scheme == "tr_bdf2" else 1.0 / (theta * dt)
        capacity = _capacity(self, model, built, space)
        if handle is None:
            handle = self._prepare(
                model, built, space, mass=mass, tolerance=self.options.tolerance
            )
        return Stepper(
            handle=handle,
            capacity=capacity,
            free=free_mask(space.FreeDofs(), space.ndof),
            scheme=scheme,
            theta=theta,
            mass=mass,
            solve=self._solve,
        )

    def matrices(
        self, model: Model
    ) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray]:
        """``K``, ``C`` (file units, SciPy CSR) and the free DOFs of the scheme.

        ``C u' + K u = F`` is the semi-discrete system the schemes step;
        verification reads its first eigenvalue.
        """
        with task_manager(self.options.threads):
            space, built = self._space(model)
            form = ngs.BilinearForm(space)
            self._add_conduction(form, model, built, space)
            form.Assemble()
            stiffness = _sparse(form.mat)
            capacity = _capacity(self, model, built, space)
            free = free_mask(space.FreeDofs(), space.ndof)
        return stiffness, capacity, free

    def forward(  # noqa: PLR0915
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Step the rise over the time grid on one prepared operator.

        Returns:
            ``{"trace:<name>": (steps + 1,), "kept_rise": (n_kept, ndof),
            "source_free": (ndof,)}``, and, if ``keep``, the cache entry
            (the stepping operator in a `KeptHandle`; ``fields`` the states
            and T0, unless only the sources are wanted, see `_trajectory`)
            or ``None``. ``request`` holds the wanted input kinds and whether
            this forward reruns for a backward that lost its entry.
        """
        request = SolveRequest() if request is None else request
        self._bind(model)
        mesh = model.mesh
        check_properties(model, self.active, self.reads)
        density, face = self._checked_loads(mesh)
        dt, _, kept = self._grid
        scales = self.scales()
        on_t, on_rise = self._trace_rows(mesh)
        # T = T0 + rise: each trace is on_t @ T0 + rows @ rise
        rows = on_t + on_rise
        with task_manager(self.options.threads):
            space, built = self._space(model)
            source = self._source_load(built, space, density, face)
            t0 = self._source_free(model, built, space)
            stepper = self._stepper(model, built, space, dt)
            u = self._start(space, stepper.free, t0)
            steps = self.spec.steps
            layout = (
                self._trajectory(stepper.handle, len(t0), request) if keep else "none"
            )
            stride = 1 if layout == "full" else _stride(steps)
            states = [] if layout == "none" else [u]
            offset = on_t @ t0
            traces = np.empty((steps + 1, len(rows)))
            traces[0] = rows @ u + offset
            heat = _HeatRecorder(
                self._heat_rows(model), stepper, dt, t0, u, density, face
            )
            heat.record(traces[0], u, 0.0)
            rises = np.empty((len(kept), len(t0)))
            slot = {int(n): k for k, n in enumerate(kept)}
            if 0 in slot:
                rises[slot[0]] = u
            for n in range(steps):
                start = u
                u, stage = stepper.step(u, scales[n] * source)
                traces[n + 1] = rows @ u + offset
                heat.step(start, stage, u, dt * scales[n])
                heat.record(traces[n + 1], u, (n + 1) * dt)
                if n + 1 in slot:
                    rises[slot[n + 1]] = u
                if layout != "none" and (n + 1) % stride == 0 and n + 1 < steps:
                    states.append(u)
            if layout == "full":
                states.append(u)
        outputs = {
            f"trace:{name}": traces[:, k] for k, name in enumerate(self.spec.traces)
        }
        outputs["kept_rise"] = rises
        outputs["source_free"] = t0
        entry = None
        if keep:
            fields = {}
            if layout != "none":
                # the trajectory (every state, or a checkpoint every stride
                # steps) and T0: the material and trace-weight pairs read them
                fields = {
                    "states": np.asarray(states),
                    "stride": np.asarray([stride]),
                    "source_free": t0,
                }
            entry = Entry(
                physics=self.physics,
                handle=PooledKeptHandle(stepper.handle),
                fields=fields,
            )
        else:
            release(stepper.handle)
        return outputs, entry

    def _reverse(
        self, entry: Entry, stepper: Stepper, scales: np.ndarray, source: np.ndarray
    ) -> Any:
        """``(n, u_n, stage_n, u_n+1)`` for n = steps - 1 down to 0.

        From the kept trajectory, or segment by segment from its checkpoints
        (one forward step per state). TR-BDF2's stage is
        recomputed; a theta step has none.
        """
        steps = self.spec.steps
        states = entry.fields["states"]
        stride = int(entry.fields["stride"][0])
        tr_bdf2 = stepper.scheme == "tr_bdf2"
        for begin in reversed(range(0, steps, stride)):
            end = min(begin + stride, steps)
            if stride == 1:
                segment = [states[begin], states[end]]
                stages = [
                    stepper.stage(states[begin], scales[begin] * source)
                    if tr_bdf2
                    else None
                ]
            else:
                segment, stages = [states[begin // stride]], []
                for n in range(begin, end):
                    nxt, stage = stepper.step(segment[-1], scales[n] * source)
                    segment.append(nxt)
                    stages.append(stage)
            for n in reversed(range(begin, end)):
                k = n - begin
                yield n, segment[k], stages[k], segment[k + 1]

    def backward(  # noqa: PLR0912, PLR0915
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
    ) -> list[FieldPair | LoadField | Integrated]:
        r"""The discrete adjoint of the exact scheme, run backwards.

        One reverse step per forward step on the forward's stepping operator
        ``A = K + m C`` (symmetric): TR-BDF2 solves ``A mu2 = ubar`` and ``A
        mu1 = m c1 C mu2`` and passes ``(2 m C - A) mu1 - m c0 C mu2`` back;
        theta solves ``A mu = ubar`` and passes ``(m C - (1 - theta) A) mu /
        theta`` back. The step's load adjoint ``d_n`` (``2 mu1 + mu2``, or
        ``mu / theta``) gives the source density's load ``sum_n s_n d_n``
        and each waveform scale's ``F . d_n`` over its steps; with the
        states, the conductivity and heat-capacity pairs of every step are
        contracted as the sweep goes (`Integrated`). The cotangents on T0
        (the source-free output, the traces' temperature part and a
        uniform start's ``-ubar_0``) add the T0 pair. A heat-out trace reads
        the states, the stages, T0, the densities and the scales directly,
        and :math:`k` and :math:`\rho c_p` through its explicit pairs
        (`_HeatAdjoint`).
        ``wanted`` holds the wanted input kinds.
        """
        self._bind(model)
        spec = self.spec
        loads = self._loads()
        wanted = wanted or frozenset()
        self._explicit = {}
        properties = {k[2] for k in wanted if k[0] == "material"}
        trace_kinds = {k for k in wanted if k[0] == "problem"} - SOURCE_KINDS
        values = {k for k in wanted if is_thermal_value(k)}
        steps = spec.steps
        dt, starts, kept = self._grid
        scales = self.scales()
        names = list(spec.traces)
        on_t, on_rise = self._trace_rows(model.mesh)
        rows = on_t + on_rise
        ct_tr = np.zeros((steps + 1, len(names)))
        for k, name in enumerate(names):
            if f"trace:{name}" in cotangents:
                ct_tr[:, k] = np.asarray(cotangents[f"trace:{name}"], dtype=float)
        ct_kept = cotangents.get("kept_rise")
        slot = {int(n): k for k, n in enumerate(kept)}

        def direct(n: int) -> np.ndarray:
            """dJ/du_n read directly (traces and kept fields)."""
            out = rows.T @ ct_tr[n] + heat.direct(n, rows.shape[1])
            if ct_kept is not None and n in slot:
                out = out + np.asarray(ct_kept[slot[n]], dtype=float)
            return out

        order = self.options.order
        with task_manager(self.options.threads):
            space, built = self._space(model)
            domain = self._domain(model, built)
            free = free_mask(space.FreeDofs(), space.ndof)
            face = loads.face_density
            source = self._source_load(
                built,
                space,
                np.asarray(loads.density),
                None if face is None else np.asarray(face),
            )
            handle = None if entry.handle is None else entry.handle.handle
            reused = handle is not None
            stepper = self._stepper(model, built, space, dt, handle=handle)
            if handle is None:
                entry.handle = PooledKeptHandle(stepper.handle)
            heat = _HeatAdjoint(self._heat_rows(model), ct_tr, stepper, dt, scales)
            if values and heat.rows:
                raise NotImplementedError(
                    "gradients of heat-out traces with respect to thermal boundary "
                    "values are not available in this version of gfem"
                )
            masses = {
                k: self._boundary_mass(built, space, model.boundaries[k[1]].surface)
                for k in sorted(values)
                if k[2] == "coefficient"
            }
            by_h = dict.fromkeys(masses, 0.0)
            density = np.asarray(loads.density)
            face_density = None if face is None else np.asarray(face)
            terms = self.terms(model, nodes=False) if properties else []
            pair_terms = [
                t for t in terms if any(p in t.derivatives for p in properties)
            ]
            m, c = stepper.mass, stepper.capacity
            tr_bdf2 = stepper.scheme == "tr_bdf2"
            theta = stepper.theta

            def adjoint(rhs: np.ndarray) -> np.ndarray:
                rhs = np.where(free, rhs, 0.0)
                return self._adjoint(stepper.handle, rhs[:, None], reused=reused)[:, 0]

            states = bool(properties or trace_kinds or masses)
            sweep: Any = (
                self._reverse(entry, stepper, scales, source)
                if states
                else ((n, None, None, None) for n in reversed(range(steps)))
            )
            ubar = direct(steps)
            load = np.zeros(space.ndof)
            per_step = np.zeros(steps)
            summed = np.zeros((space.ndof, len(names)))
            integrals: dict[str, np.ndarray] = {}
            batch: list[FieldPair] = []

            def flush() -> None:
                for name, values in element_integrals(model, batch, pair_terms).items():
                    integrals[name] = integrals.get(name, 0.0) + values
                batch.clear()

            for n, u, stage, nxt in sweep:
                if tr_bdf2:
                    g = (2.0 - GAMMA) * GAMMA
                    c1, c0 = 1.0 / g, (1.0 - GAMMA) ** 2 / g
                    mu2 = adjoint(ubar)
                    mu1 = adjoint(m * c1 * (c @ mu2) + heat.stage(n, len(ubar)))
                    back = (
                        2.0 * m * (c @ mu1)
                        - stepper.handle.apply(mu1)
                        - m * c0 * (c @ mu2)
                    )
                    d = 2.0 * mu1 + mu2
                    if masses:
                        self._add_convection(
                            by_h,
                            masses,
                            np.stack([-mu1, -mu2], axis=1),
                            np.stack([stage + u, nxt], axis=1),
                        )
                    if properties:
                        left = np.stack([-mu1, -mu2], axis=1)
                        batch += [
                            FieldPair(
                                space,
                                left,
                                np.stack([stage + u, nxt], axis=1),
                                {"conduction": 1.0},
                                order,
                                domain,
                            ),
                            FieldPair(
                                space,
                                left,
                                m
                                * np.stack(
                                    [stage - u, nxt - c1 * stage + c0 * u], axis=1
                                ),
                                {"capacity": 1.0},
                                order,
                                domain,
                            ),
                        ]
                else:
                    mu = adjoint(ubar)
                    back = (m / theta) * (c @ mu) - (
                        (1.0 - theta) / theta
                    ) * stepper.handle.apply(mu)
                    d = mu / theta
                    if masses:
                        self._add_convection(
                            by_h,
                            masses,
                            -mu[:, None],
                            (nxt + ((1.0 - theta) / theta) * u)[:, None],
                        )
                    if properties:
                        left = -mu[:, None]
                        batch += [
                            FieldPair(
                                space,
                                left,
                                (nxt + ((1.0 - theta) / theta) * u)[:, None],
                                {"conduction": 1.0},
                                order,
                                domain,
                            ),
                            FieldPair(
                                space,
                                left,
                                (m * (nxt - u))[:, None],
                                {"capacity": 1.0},
                                order,
                                domain,
                            ),
                        ]
                if properties and len(batch) >= 8:
                    flush()
                if trace_kinds:
                    if n == steps - 1:
                        summed += np.outer(nxt, ct_tr[steps])
                    summed += np.outer(u, ct_tr[n])
                if properties and heat.rows:
                    heat.accumulate(n, u, stage, nxt)
                load += scales[n] * d
                per_step[n] = source @ d + heat.scale(n, density, face_density)
                ubar = np.where(free, back, 0.0) + direct(n)
            if batch:
                flush()
            # the cotangent of T0 (dJ/dT0): read directly, and through a
            # uniform start u_0 = P (T_start - T0)
            needs_t0 = (
                (properties and "thermal_conductivity" in properties)
                or trace_kinds
                or values
            )
            pairs: list[FieldPair | LoadField | Integrated] = []
            if integrals:
                pairs.append(Integrated(integrals))
            if needs_t0:
                t0 = entry.fields["source_free"]
                ct0 = on_t.T @ ct_tr.sum(axis=0) + heat.source_free(space.ndof)
                if "source_free" in cotangents:
                    ct0 = ct0 + np.asarray(cotangents["source_free"], dtype=float)
                if spec.initial_temperature is not None:
                    ct0 = ct0 - np.where(free, ubar, 0.0)
                on_fixed = ct0
                ct0 = np.where(free, ct0, 0.0)
                if "thermal_conductivity" in properties and self._t0_solved():
                    pairs.append(
                        self._source_free_pair(
                            model, built, space, ct0, t0, loose=self.loose
                        )
                    )
                if trace_kinds:
                    self._explicit = self._trace_gradients(
                        loads, summed, ct_tr.sum(axis=0), t0
                    )
                if values:
                    self._explicit |= self._boundary_gradients(
                        model, built, space, (ct0, on_fixed), t0, by_h, values
                    )
            if properties and heat.rows:
                t0 = entry.fields["source_free"]
                pairs += heat.pairs(self, model, t0, properties)
            if wanted & LOAD_KINDS:
                pairs.append(LoadField(space, load, order, domain, length_power=2))
                if heat.rows:
                    # the heat traces read the densities directly: added to
                    # the load pass's values (`gfem._adjoint.solve_vjp`)
                    volume, on_faces = heat.densities()
                    self._explicit[LOAD_KIND] = volume
                    if FACE_LOAD_KIND in wanted:
                        self._explicit[FACE_LOAD_KIND] = on_faces
        explicit = getattr(self, "_explicit", {})
        if ("problem", ".scales") in wanted:
            which = np.searchsorted(starts, np.arange(steps), side="right") - 1
            values = np.zeros(len(starts))
            np.add.at(values, which[which >= 0], per_step[which >= 0])
            explicit = {**explicit, ("problem", ".scales"): values}
        self._explicit = explicit
        return pairs

    def _boundary_mass(
        self, built: EngineMesh, space: ngs.FESpace, surface: str
    ) -> scipy.sparse.csr_matrix:
        """The boundary mass ``h``'s operator reads: ``int_S u v`` (file units)."""
        u, v = space.TnT()
        region = boundary_region(built, (surface,))
        form = ngs.BilinearForm(space)
        form += (
            built.scale * u * v * quadrature.ds(self.options.order, definedon=region)
        )
        form.Assemble()
        return _sparse(form.mat)

    @staticmethod
    def _add_convection(
        totals: dict[tuple, float],
        masses: dict[tuple, Any],
        left: np.ndarray,
        right: np.ndarray,
    ) -> None:
        """Add a step's ``left^T M_S right`` to each coefficient's total.

        The coefficient enters the stepping operator as ``h M_S`` beside the
        conduction, so its pair is the conduction pair's fields on ``M_S``.
        """
        for kind, mass in masses.items():
            totals[kind] += float(np.sum(left * (mass @ right)))

    def _surface_load(
        self, built: EngineMesh, space: ngs.FESpace, surface: str
    ) -> np.ndarray:
        """``int_S v`` (file units): a unit flux or ``h T_amb`` of one surface."""
        v = space.TestFunction()
        region = boundary_region(built, (surface,))
        form = ngs.LinearForm(space)
        form += built.scale * v * quadrature.ds(self.options.order, definedon=region)
        form.Assemble()
        return form.vec.FV().NumPy().copy()

    def _unit_lift(
        self, built: EngineMesh, space: ngs.FESpace, surface: str
    ) -> np.ndarray:
        """`_lift` of a unit temperature on ``surface`` and zero on the others."""
        gf = ngs.GridFunction(space)
        unit = {s: float(s == surface) for s, _ in self.boundaries.fixed}
        values = built.mesh.BoundaryCF(unit, default=ngs.CF(0.0))
        region = boundary_region(built, self.boundaries.dirichlet)
        gf.Set(values, ngs.BND, definedon=region)
        return gf.vec.FV().NumPy().copy()

    def _boundary_gradients(  # noqa: PLR0917 - one backward's state
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        ct0: tuple[np.ndarray, np.ndarray],
        t0: np.ndarray,
        stepping: dict[tuple, float],
        values: set[tuple],
    ) -> dict[tuple, np.ndarray]:
        r"""The cotangents of the thermal boundary values.

        ``ct0`` is the cotangent on the source-free temperature :math:`T_0`
        (zero on the fixed DOFs, then as it is), ``stepping`` the coefficients'
        share through the stepping operator. With :math:`A_0 \lambda_0 =
        \bar T_0` on the free DOFs, :math:`M_S` the surface's boundary mass and
        :math:`\ell_S = \int_S v`:

        - :math:`h`: stepping :math:`+ \lambda_0^T (T_\infty \ell_S - M_S T_0)`;
        - :math:`T_\infty`: :math:`h\, \lambda_0^T \ell_S`;
        - flux: :math:`\lambda_0^T \ell_S`;
        - fixed temperature: :math:`\bar T_0 \cdot \chi - \lambda_0^T A_0 \chi`
          for the lift :math:`\chi` of a unit value.

        The rise reads the coefficient alone (through the stepping operator)
        and the start through :math:`T_0`, which ``ct0`` holds. No solve when
        :math:`T_0` is the start everywhere.
        """
        free_cotangent, full_cotangent = ct0
        items = model.boundaries
        coupled = not (self.loose.all() or not self.boundaries.sinks)
        lam = np.zeros(space.ndof)
        reaction = None
        if coupled and np.any(free_cotangent):
            where = self.loose if self.loose.any() else None
            handle = self._prepare(
                model,
                built,
                space,
                mass=0.0 if where is None else 1.0,
                where=where,
                tolerance=adjoint_tolerance(self.options),
            )
            try:
                lam = self._adjoint(handle, free_cotangent[:, None], reused=False)[:, 0]
                reaction = handle.apply
                out = self._boundary_values(
                    items,
                    built,
                    space,
                    lam,
                    full_cotangent,
                    t0,
                    stepping,
                    values,
                    reaction,
                )
            finally:
                handle.free()
            return out
        return self._boundary_values(
            items, built, space, lam, full_cotangent, t0, stepping, values, None
        )

    def _boundary_values(  # noqa: PLR0917 - one backward's state
        self,
        items: Any,
        built: EngineMesh,
        space: ngs.FESpace,
        lam: np.ndarray,
        full_cotangent: np.ndarray,
        t0: np.ndarray,
        stepping: dict[tuple, float],
        values: set[tuple],
        apply: Any,
    ) -> dict[tuple, np.ndarray]:
        """Each wanted value's cotangent from ``lam`` (see `_boundary_gradients`)."""
        last = {
            b.surface: i for i, b in enumerate(items) if isinstance(b, FixedTemperature)
        }
        out: dict[tuple, np.ndarray] = {}
        for kind in values:
            item = items[kind[1]]
            name = kind[2]
            if name == "temperature":
                grad = 0.0
                if last[item.surface] == kind[1]:
                    chi = self._unit_lift(built, space, item.surface)
                    grad = float(full_cotangent @ chi)
                    if apply is not None:
                        grad -= float(lam @ apply(chi))
            else:
                load = self._surface_load(built, space, item.surface)
                grad = float(lam @ load)
                if name == "ambient":
                    grad *= float(item.coefficient)
                elif name == "coefficient":
                    mass = self._boundary_mass(built, space, item.surface)
                    grad = stepping[kind] + float(item.ambient) * grad
                    grad -= float(lam @ (mass @ t0))
            out[kind] = np.asarray(grad)
        return out

    def _t0_solved(self) -> bool:
        """Whether T0 depends on the conductivity (else it is one constant)."""
        if not self.loose.any() and self.boundaries.uniform is not None:
            return False
        return not (self.loose.all() or not self.boundaries.sinks)

    def _trace_gradients(
        self,
        loads: Loads,
        summed: np.ndarray,
        weights: np.ndarray,
        t0: np.ndarray,
    ) -> dict[tuple, np.ndarray]:
        """The cotangents of the traces' own leaves (a section's weights).

        A trace is linear in its leaves and in the fields, so ``sum_n ct_n
        trace(T_n, rise_n)`` is the trace of the cotangent-weighted sums of
        the fields: one reverse pass of its ``_sample`` per trace.
        """
        out: dict[tuple, np.ndarray] = {}
        for k, name in enumerate(self.spec.traces):
            trace = loads.traces[name]
            if not jax.tree_util.tree_leaves(trace):
                continue
            rise = self._field(summed[:, k], "dT")
            temperature = self._field(summed[:, k] + weights[k] * t0, "T")

            def sample(
                tr: Any, temperature: Field = temperature, rise: Field = rise
            ) -> Any:
                return jnp.reshape(tr._sample(temperature, rise), ())  # noqa: SLF001

            _, pull = jax.vjp(sample, trace)
            (grad,) = pull(jnp.ones(()))
            for path, leaf in jax.tree_util.tree_flatten_with_path(
                Loads(density=None, traces={name: grad})
            )[0]:
                out[input_kind(path, "problem")] = np.asarray(leaf)
        return out

    def explicit_problem(self) -> dict[tuple, np.ndarray]:
        """The waveform scales' and the traces' cotangents of the last backward."""
        return getattr(self, "_explicit", {})

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        del model
        n = self._ndof()
        steps = self.spec.steps
        _, _, kept = self._grid
        specs = {
            f"trace:{name}": jax.ShapeDtypeStruct((steps + 1,), jnp.float64)
            for name in self.spec.traces
        }
        specs["kept_rise"] = jax.ShapeDtypeStruct((len(kept), n), jnp.float64)
        specs["source_free"] = jax.ShapeDtypeStruct((n,), jnp.float64)
        return specs

    def result(self, outputs: dict) -> TransientThermalResult:
        """The public result from the forward outputs."""
        problem = self.spec
        dt, _, kept = self._grid
        _, power = self._heat()
        return TransientThermalResult(
            time=jnp.arange(problem.steps + 1) * dt,
            traces={name: outputs[f"trace:{name}"] for name in problem.traces},
            _rises=outputs["kept_rise"],
            _source_free=self._field(outputs["source_free"], "T"),
            power=power,
            kept_times=tuple(float(n * dt) for n in kept),
            switch_times=problem._switches,  # noqa: SLF001
            _thickness=self._mesh.section_thickness,
        )


def _sparse(matrix: Any) -> scipy.sparse.csr_matrix:
    """An assembled NGSolve matrix (of a non-symmetric form) as SciPy CSR."""
    rows, cols, values = matrix.COO()
    shape = (matrix.height, matrix.width)
    return scipy.sparse.csr_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(cols))), shape=shape
    )


def _capacity(
    rule: ThermalRule,
    model: Model,
    built: EngineMesh,
    space: ngs.FESpace,
    where: np.ndarray | None = None,
) -> scipy.sparse.csr_matrix:
    """The heat-capacity mass ``C`` (on ``where``) as SciPy CSR (file units)."""
    form = ngs.BilinearForm(space)
    form += rule._capacity_term(model, built, space, where)  # noqa: SLF001
    form.Assemble()
    return _sparse(form.mat)


def _check_trace(mesh: Mesh, regions: tuple[str, ...], name: str, trace: Trace) -> None:
    """Raise for a mean trace over an unknown region or one outside the domain.

    And for a heat-out trace on the other mesh kind (the twin names it) or
    inside an objective classified for `gfem.refine_mesh`.

    Raises:
        RegionNotFoundError: The region is not in the mesh.
        ValueError: It is a surface, or holds no element of the domain; a
            heat-out trace on the wrong mesh kind.
        MeshRefinementUnsupportedError: A heat-out trace in a refined
            objective: thermal problems have no error estimate.
    """
    if isinstance(trace, _HeatTrace):
        if _objective.recording() is not None:
            who = f"Trace.{trace._name}"  # noqa: SLF001
            raise MeshRefinementUnsupportedError(
                who,
                f"The objective reads gfem.{who}, and thermal problems have no "
                "error estimate: refine the mesh by hand and compare.",
            )
        section = mesh.section_thickness is not None
        if trace._per_length and not section:  # noqa: SLF001
            raise ValueError(
                f"TransientThermal trace {name!r}: Trace.heat_out_per_length records "
                "a cross-section's heat in J/m, but this mesh is 3D: use "
                "gfem.Trace.heat_out (J)."
            )
        if section and not trace._per_length:  # noqa: SLF001
            raise ValueError(
                f"TransientThermal trace {name!r}: Trace.heat_out records a 3D "
                "result's heat in J, but this mesh is a cross-section, whose slab "
                "thickness is a numerical choice: use "
                "gfem.Trace.heat_out_per_length (J/m)."
            )
        return
    if not isinstance(trace, _FieldTrace):
        return
    for region in trace._regions:  # noqa: SLF001
        if region not in mesh.volumes:
            if region in mesh.surfaces:
                raise ValueError(
                    f"TransientThermal trace {name!r}: {region!r} is a surface; "
                    "Trace.mean averages over volume regions."
                )
            raise RegionNotFoundError(region, mesh.volumes, kind="volume")
    if trace._regions and not set(trace._regions) & set(regions):  # noqa: SLF001
        raise ValueError(
            f"TransientThermal trace {name!r}: {list(trace._regions)} hold no "  # noqa: SLF001
            "element of the thermal domain; average over regions it solves on."
        )


def trace_rows(
    trace: Trace, mesh: Mesh, layout: tuple, field: Any, ndof: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(ndof,)`` twice: the trace is ``on_t @ T + on_rise @ rise`` on ``layout``.

    A point or mean trace (`gfem.Trace.point`, `gfem.Trace.mean`) reads the
    temperature: the basis functions at the point, or their exact mean
    over the regions (`gfem.region_mean`'s rows). Any other trace gives its
    rows by differentiating its linear ``_sample`` once (a reverse pass at
    zero fields), ``field(coefficients, quantity)`` building its fields.

    Raises:
        ValueError: A point outside the thermal domain.
    """
    from gfem._engine import probe
    from gfem._engine.fields import mean_rows

    support, excluded, space, order = layout
    if isinstance(trace, _FieldTrace):
        if trace._point is not None:  # noqa: SLF001
            where = probe.Layout(
                support=support, excluded=excluded, space=space, order=order
            )
            point = np.asarray([trace._point], float)  # noqa: SLF001
            located = probe.locate(mesh, where, point)
            found = probe.probe(mesh, where, located)
            row = found.transpose(np.ones((1, 1, 1)))[0, : found.ndof].real
        else:
            inside = tuple(r for r in trace._regions if r in support)  # noqa: SLF001
            row = mean_rows(mesh, layout, inside)[0]
        return row, np.zeros_like(row)
    zero = jnp.zeros(ndof)

    def sample(on_t: jax.Array, on_rise: jax.Array) -> jax.Array:
        value = trace._sample(field(on_t, "T"), field(on_rise, "dT"))  # noqa: SLF001
        return jnp.reshape(value, ())

    _, pull = jax.vjp(sample, zero, zero)
    on_t, on_rise = pull(jnp.ones(()))
    return np.asarray(on_t, float), np.asarray(on_rise, float)
