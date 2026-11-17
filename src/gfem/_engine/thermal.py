"""Thermal rules: heat conduction on the thermal domain (ADRs 0167, 0168, 0172).

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

The temperature splits as T = T0 + rise (ADR 0172): T0 is the source-free
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
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from gfem._boundaries import Convection, FixedTemperature, HeatFlux
from gfem._checks import is_traced
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import element_values, p0
from gfem._engine.domain import boundary_region
from gfem._engine.electrostatic import KeptHandle
from gfem._engine.fields import fe_space, support_region
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._linalg import Handle, Kind, Operator, ResolvedLinearSolver, report
from gfem.errors import MissingMaterialPropertyError
from gfem.mesh import _TET_FACES, Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import SteadyThermal
from gfem.results import Field, SteadyThermalResult
from gfem.thermal import _tet_volumes

__all__ = ["SteadyThermalRule", "ThermalRule"]

#: The linear solver for ``linear_solver="auto"`` (ADR 0184).
DEFAULT_SOLVER: Kind = "direct"
#: The physics key of the model's solve domains and exclusions.
_PHYSICS = "thermal"


def source_density(problem: Any, mesh: Mesh) -> jax.Array:
    """The summed power density of the problem's heat sources per tetrahedron.

    W/m^3 in `gfem.Mesh.tetrahedra` order, in JAX: linear in each source's
    amount (ADR 0178). Zero without a source.
    """
    density = jnp.zeros(len(mesh.tetrahedra), jnp.float64)
    for source in problem.heat:
        density = density + source._density_on(mesh)  # noqa: SLF001
    return density


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


def _face_keys(faces: np.ndarray, n: int) -> np.ndarray:
    f = np.sort(faces, axis=1).astype(np.int64)
    return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]


def pieces(mesh: Mesh, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The connected pieces of the active tetrahedra, through shared faces.

    Returns:
        The active tetrahedra (indices) and the piece label of each.
    """
    index = np.flatnonzero(active)
    n_nodes = len(mesh.imported_nodes)
    keys = _face_keys(mesh.tetrahedra[index][:, _TET_FACES].reshape(-1, 3), n_nodes)
    _, face = np.unique(keys, return_inverse=True)
    rows = np.repeat(np.arange(len(index)), 4)
    graph = scipy.sparse.coo_matrix(
        (np.ones(rows.size), (rows, len(index) + face.ravel())),
        shape=(len(index) + face.max() + 1,) * 2,
    )
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return index, label[: len(index)]


def _check_sinks(model: Model, active: np.ndarray, sinks: tuple[str, ...]) -> None:
    """Every connected piece of the thermal domain needs a heat sink (ADR 0168).

    Raises:
        ValueError: A piece touches no `FixedTemperature` or `Convection`
            surface; the message names a region of it.
    """
    mesh = model.mesh
    index, label = pieces(mesh, active)
    n_nodes = len(mesh.imported_nodes)
    tags = [mesh.region(s).tag for s in sinks]
    sink_keys = _face_keys(mesh.triangles[np.isin(mesh.triangle_tags, tags)], n_nodes)
    faces = _face_keys(mesh.tetrahedra[index][:, _TET_FACES].reshape(-1, 3), n_nodes)
    touching = np.isin(faces, sink_keys).reshape(-1, 4).any(axis=1)
    cooled = np.unique(label[touching])
    loose = ~np.isin(label, cooled)
    if not loose.any():
        return
    tets = index[loose]
    where = next(r for r in mesh.volumes if np.isin(mesh.elements(r), tets).any())
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

    def __init__(self, model: Model, problem: Any, options: SolverOptions) -> None:
        """Check the model and problem (see the problem's docstring).

        Raises:
            NotImplementedError: The model has a periodic pair.
            MissingMaterialPropertyError: A region of the thermal domain
                lacks a property the problem reads.
            ValueError: A heat source heats an excluded region.
        """
        self.problem = problem
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
            _check_heat_inside(
                mesh, self.active, np.asarray(source_density(problem, mesh))
            )

    # --- the discretization ---------------------------------------------------

    def _space(self, model: Model) -> tuple[ngs.FESpace, EngineMesh]:
        """The compressed H1 space on the thermal domain, fixed on the sinks."""
        order = self.options.order
        disc = discretization(model.mesh)
        built = disc.built
        fixed = self.boundaries.dirichlet
        regions, excluded = self.regions, self._excluded

        def build() -> ngs.FESpace:
            names = built.mesh.GetBoundaries()
            space = ngs.H1(
                built.mesh,
                order=order,
                definedon=support_region(built, model.mesh, regions, excluded),
                dirichlet=[j + 1 for j, n in enumerate(names) if n in fixed],
            )
            return ngs.Compress(space)

        key = ((_PHYSICS, regions, excluded), "h1", order, fixed)
        return disc.space(key, build), built

    def _coefficient(
        self, model: Model, built: EngineMesh, name: str
    ) -> ngs.GridFunction:
        """Property ``name`` per element, zero off the thermal domain."""
        values = element_values(model, name)
        return p0(built, np.where(self.active, values, 0.0))

    def _prepare(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        *,
        mass: float = 0.0,
        width: int = 1,
        tolerance: float | None = None,
    ) -> Handle:
        """The prepared operator ``K + mass * C`` (file units, see the module).

        ``K`` is the conduction plus convection operator; ``C`` the
        consistent heat-capacity mass ``int rho c_p u v`` (read only with
        ``mass`` != 0, as ``rho c_p scale ** 2 * mass``).
        """
        order = self.options.order
        scale = built.scale
        u, v = space.TnT()
        form = ngs.BilinearForm(space, symmetric=True)
        k = self._coefficient(model, built, "thermal_conductivity")
        form += k * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
        for surface, h, _ in self.boundaries.convection:
            region = boundary_region(built, (surface,))
            form += h * scale * u * v * quadrature.ds(order, definedon=region)
        if mass:
            rho = self._coefficient(model, built, "mass_density")
            c_p = self._coefficient(model, built, "specific_heat")
            form += mass * scale**2 * rho * c_p * u * v * quadrature.dx(order)
        return self.linear.prepare(
            Operator(form, space.FreeDofs()),
            symmetry="hermitian",
            width=width,
            tolerance=tolerance,
        )

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
        """The coefficients of a uniform temperature ``value``."""
        gf = ngs.GridFunction(space)
        gf.Set(ngs.CF(value))
        return gf.vec.FV().NumPy().copy()

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
        density = source_density(self.problem, self._mesh)
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


class SteadyThermalRule(ThermalRule):
    """Host-side forward of a `SteadyThermal` problem on a `Model`."""

    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "steady_thermal"
    problem_name = "steady thermal"
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs: frozenset[str] = frozenset()

    def __init__(
        self, model: Model, problem: SteadyThermal, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.SteadyThermal`).

        Raises:
            NotImplementedError: The model has a periodic pair.
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
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve the source-free temperature and the rise on one operator.

        Returns:
            ``{"temperature": (ndof,), "rise": (ndof,)}`` in K, and, if
            ``keep``, the cache entry (the prepared operator in a
            `KeptHandle`, ``fields`` the two solutions) or ``None``.
        """
        mesh = model.mesh
        check_properties(model, self.active, self.reads)
        density = np.asarray(source_density(self.problem, mesh))
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
