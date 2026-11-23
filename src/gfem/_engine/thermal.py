"""Thermal rules: heat conduction on the thermal domain.

The unknown is the absolute temperature T in kelvin, in real H1 of the
solve order on the thermal domain (every tetrahedron minus the volumes
excluded from thermal problems). The steady operator is

    a(u, v) = int k grad u . grad v  +  sum_Convection int_S h u v,

the load is ``int q v`` for the heat sources' power density q (one P0
value per tetrahedron), ``int_S flux v`` per `gfem.HeatFlux` and
``int_S h T_ambient v`` per `gfem.Convection`, and the
`gfem.FixedTemperature` surfaces are Dirichlet surfaces, lifted with their
values (a Dirichlet value wins on nodes shared with another condition).
Insulated surfaces need no term.

The temperature splits as T = T0 + rise: T0 is the source-free
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

from gfem._adjoint.solve_vjp import adjoint_tolerance, input_kind
from gfem._boundaries import Convection, FixedTemperature, HeatFlux
from gfem._checks import is_traced
from gfem._engine import quadrature
from gfem._engine.cache import Entry, _alone, retain_budget
from gfem._engine.coefficients import element_values, p0
from gfem._engine.domain import boundary_region
from gfem._engine.electrostatic import KeptHandle
from gfem._engine.fields import (
    dirichlet_h1,
    face_keys,
    fe_space,
    support_region,
    surface_triangles,
    tetrahedron_face_keys,
    vertex_dofs,
)
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import (
    LOAD_KIND,
    Derivative,
    FieldPair,
    Integrated,
    LoadField,
    Term,
    element_integrals,
)
from gfem._engine.solve_request import SolveRequest
from gfem._linalg import Handle, Kind, Operator, ResolvedLinearSolver, report
from gfem._linalg.base import free_mask
from gfem._pytree import register_fields
from gfem.errors import MissingMaterialPropertyError, RegionNotFoundError
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import SteadyThermal, TransientThermal
from gfem.results import Field, SteadyThermalResult, TransientThermalResult
from gfem.thermal import Trace, _FieldTrace, _tet_volumes

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


@dataclasses.dataclass(frozen=True, eq=False)
class Loads:
    """A thermal problem as `gfem._adjoint.solve_vjp` sees it.

    Its differentiable leaves only: every heat source resolved in JAX to one
    summed power density per tetrahedron, and for a transient the waveform
    scales and the traces (whose leaves are a section's thermo-optic
    weights). The rule reads everything else from the problem it was built
    with (`ThermalRule.spec`). The input kinds are ``("problem",
    ".density")`` (`gfem._engine.sensitivity.LOAD_KIND`), ``("problem",
    ".scales")`` and ``("problem", ".traces['<name>']...")``.

    Attributes:
        density: W/m^3 per tetrahedron (`gfem.Mesh.tetrahedra` order).
        scales: The waveform's scales, or ``None`` (steady).
        traces: The traces by name, or ``None`` (steady).
    """

    density: Any
    scales: Any = None
    traces: Any = None


register_fields(Loads, ("density", "scales", "traces"))

#: The input kinds of the sources alone: their gradient needs the step
#: adjoints only, so no trajectory is kept for them.
SOURCE_KINDS = frozenset({LOAD_KIND, ("problem", ".scales")})


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


def _check_heat_inside(mesh: Mesh, active: np.ndarray, density: np.ndarray) -> None:
    """Raise for heat on a tetrahedron outside the thermal domain.

    Raises:
        ValueError: A heat source heats an element excluded from thermal
            problems.
    """
    outside = (np.asarray(density) != 0) & ~active
    if not outside.any():
        return
    where = next(r for r in mesh.volumes if outside[mesh.elements(r)].any())
    raise ValueError(
        f"A heat source heats region {where!r}, which is excluded from thermal "
        "problems, so the heat has nowhere to go. Heat a region of the thermal "
        f"domain, or keep {where!r} in it (drop it from "
        'excluded={"thermal": ...}).'
    )


@dataclasses.dataclass(frozen=True)
class _Boundaries:
    """The model's thermal conditions, as static data (values in SI)."""

    fixed: tuple[tuple[str, float], ...]
    convection: tuple[tuple[str, float, float], ...]
    flux: tuple[tuple[str, float], ...]

    @classmethod
    def of(cls, model: Model) -> _Boundaries:
        items = model.boundaries
        return cls(
            fixed=tuple(
                (b.surface, b.temperature)
                for b in items
                if isinstance(b, FixedTemperature)
            ),
            convection=tuple(
                (b.surface, b.coefficient, b.ambient)
                for b in items
                if isinstance(b, Convection)
            ),
            flux=tuple(
                (b.surface, b.flux)
                for b in items
                if isinstance(b, HeatFlux) and b.flux != 0
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
            _check_heat_inside(mesh, self.active, density)

    def vjp_problem(self, problem: Any) -> Loads:
        """The problem's differentiable leaves, resolved in JAX (`Loads`)."""
        return Loads(density=source_density(problem, self._mesh))

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
        return self.linear.prepare(
            Operator(form, space.FreeDofs()),
            symmetry="hermitian",
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
        self, built: EngineMesh, space: ngs.FESpace, density: np.ndarray
    ) -> np.ndarray:
        """The heat sources' load ``int q v`` (file units)."""
        v = space.TestFunction()
        form = ngs.LinearForm(space)
        q = p0(built, np.asarray(density, float) * built.scale**2)
        form += q * v * quadrature.dx(self.options.order)
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

    def _still(self) -> Mesh:
        """The mesh at its imported nodes: the topology without traced values."""
        mesh = self._mesh
        return Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001

    def _ndof(self) -> int:
        with task_manager(self.options.threads):
            _, space, _ = fe_space(
                self._still(), self.regions, self._excluded, "h1", self.options.order
            )
        return space.ndof

    def _heat_order(self) -> np.ndarray:
        """The tetrahedron of each DOF of the heat field's L2 order-0 space."""
        still = self._still()
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
        """The heat field and the sources' total power (W), in JAX."""
        density = source_density(self.spec, self._mesh)
        heat = Field(
            coefficients=density[self._heat_order()],
            mesh=self._mesh,
            quantity="q",
            unit="W/m^3",
            space="l2",
            order=0,
            support=self.regions,
            _excluded=self._excluded,
        )
        power = jnp.sum(density * _tet_volumes(self._mesh))
        return heat, power

    # --- the differentiation interface ----------------------------------------

    def role(self, kind: tuple) -> str:
        """``"read"`` if the thermal solve depends on input ``kind``.

        The thermal properties in `reads`, the heat sources (problem
        leaves) and the node coordinates are read; every other material
        property (the electromagnetic ones, ``conductivity``) and every
        electromagnetic boundary or port value is not: an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
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
    ) -> FieldPair:
        """The T0 pair ``(-lambda0, T0)`` of a cotangent ``load`` on T0.

        ``A0 T0 = boundary load`` on the free DOFs, for the steady operator
        A0 (with the heat capacity of a sinkless part, ``loose``, where the
        source-free temperature is its uniform start and the capacity's own
        pair vanishes), so ``load . dT0 = -lambda0^T dA0 T0`` with ``A0
        lambda0 = load``. One more solve, on its own prepared operator.
        """
        where = None if loose is None or not loose.any() else loose
        handle = self._prepare(
            model,
            built,
            space,
            mass=0.0 if where is None else 1.0,
            where=where,
            tolerance=adjoint_tolerance(self.options),
        )
        try:
            lam = self._adjoint(handle, load[:, None], reused=False)
        finally:
            handle.free()
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
        mesh = model.mesh
        check_properties(model, self.active, self.reads)
        density = np.asarray(self._loads().density)
        _check_heat_inside(mesh, self.active, density)
        uniform = self.boundaries.uniform
        with task_manager(self.options.threads):
            space, built = self._space(model)
            source = self._source_load(built, space, density)
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
                handle=KeptHandle(handle),
                fields={"temperature": temperature, "rise": rise},
            )
        else:
            handle.free()
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
                entry.handle = KeptHandle(handle)
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
                    pairs.append(self._source_free_pair(model, built, space, ct0, t0))
            if LOAD_KIND in wanted:
                pairs.append(LoadField(space, lam, order, domain, length_power=2))
        return pairs

    def result(self, outputs: dict) -> SteadyThermalResult:
        """The public result from the forward outputs."""
        heat, power = self._heat()
        return SteadyThermalResult(
            temperature=self._field(outputs["temperature"], "T"),
            rise=self._field(outputs["rise"], "dT"),
            heat=heat,
            power=power,
            _thickness=self._mesh.section_thickness,
        )


# --- transient ------------------------------------------------------------------

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
        held = _alone(Entry(physics=self.physics, handle=KeptHandle(handle)))
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
            on_t[k], on_rise[k] = trace_rows(
                trace, mesh, self._layout(), self._field, n
            )
        return on_t, on_rise

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
            handle.free()

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

    def forward(
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
        mesh = model.mesh
        check_properties(model, self.active, self.reads)
        density = np.asarray(self._loads().density)
        _check_heat_inside(mesh, self.active, density)
        dt, _, kept = self._grid
        scales = self.scales()
        on_t, on_rise = self._trace_rows(mesh)
        # T = T0 + rise: each trace is on_t @ T0 + rows @ rise
        rows = on_t + on_rise
        with task_manager(self.options.threads):
            space, built = self._space(model)
            source = self._source_load(built, space, density)
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
            rises = np.empty((len(kept), len(t0)))
            slot = {int(n): k for k, n in enumerate(kept)}
            if 0 in slot:
                rises[slot[0]] = u
            for n in range(steps):
                u, _ = stepper.step(u, scales[n] * source)
                traces[n + 1] = rows @ u + offset
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
                physics=self.physics, handle=KeptHandle(stepper.handle), fields=fields
            )
        else:
            stepper.handle.free()
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
        """The discrete adjoint of the exact scheme, run backwards.

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
        uniform start's ``-ubar_0``) add the T0 pair. ``wanted`` holds the
        wanted input kinds.
        """
        spec = self.spec
        loads = self._loads()
        wanted = wanted or frozenset()
        properties = {k[2] for k in wanted if k[0] == "material"}
        trace_kinds = {k for k in wanted if k[0] == "problem"} - SOURCE_KINDS
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
            out = rows.T @ ct_tr[n]
            if ct_kept is not None and n in slot:
                out = out + np.asarray(ct_kept[slot[n]], dtype=float)
            return out

        order = self.options.order
        with task_manager(self.options.threads):
            space, built = self._space(model)
            domain = self._domain(model, built)
            free = free_mask(space.FreeDofs(), space.ndof)
            source = self._source_load(built, space, np.asarray(loads.density))
            handle = None if entry.handle is None else entry.handle.handle
            reused = handle is not None
            stepper = self._stepper(model, built, space, dt, handle=handle)
            if handle is None:
                entry.handle = KeptHandle(stepper.handle)
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

            states = bool(properties or trace_kinds)
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
                    mu1 = adjoint(m * c1 * (c @ mu2))
                    back = (
                        2.0 * m * (c @ mu1)
                        - stepper.handle.apply(mu1)
                        - m * c0 * (c @ mu2)
                    )
                    d = 2.0 * mu1 + mu2
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
                load += scales[n] * d
                per_step[n] = source @ d
                ubar = np.where(free, back, 0.0) + direct(n)
            if batch:
                flush()
            # the cotangent of T0 (dJ/dT0): read directly, and through a
            # uniform start u_0 = P (T_start - T0)
            needs_t0 = (properties and "thermal_conductivity" in properties) or (
                trace_kinds
            )
            pairs: list[FieldPair | LoadField | Integrated] = []
            if integrals:
                pairs.append(Integrated(integrals))
            if needs_t0:
                t0 = entry.fields["source_free"]
                ct0 = on_t.T @ ct_tr.sum(axis=0)
                if "source_free" in cotangents:
                    ct0 = ct0 + np.asarray(cotangents["source_free"], dtype=float)
                if spec.initial_temperature is not None:
                    ct0 = ct0 - np.where(free, ubar, 0.0)
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
            if LOAD_KIND in wanted:
                pairs.append(LoadField(space, load, order, domain, length_power=2))
        explicit = getattr(self, "_explicit", {})
        if ("problem", ".scales") in wanted:
            which = np.searchsorted(starts, np.arange(steps), side="right") - 1
            values = np.zeros(len(starts))
            np.add.at(values, which[which >= 0], per_step[which >= 0])
            explicit = {**explicit, ("problem", ".scales"): values}
        self._explicit = explicit
        return pairs

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

    Raises:
        RegionNotFoundError: The region is not in the mesh.
        ValueError: It is a surface, or holds no element of the domain.
    """
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
