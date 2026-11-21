"""Electrostatic rule: terminal potentials in H1 and the Maxwell capacitance.

For terminal i, the potential phi_i is 1 on terminal i, 0 on the other
terminals and the ground, and solves the charge-free Laplace problem
``a(phi_i, v) = 0`` with ``a(u, v) = int eps0 eps_r grad u . grad v`` for
every v vanishing on the fixed surfaces. Then ``C_ij = a(phi_i, phi_j)``,
which equals the energy formula ``E(phi_i + phi_j) - (C_ii + C_jj) / 2``
and is symmetric. All terminals share one prepared operator: the right-hand
sides are the liftings, solved together: by CG + BDDC by default, or by
one sparse Cholesky factorization (``direct``). ``phi_i^T K phi_j`` is
quadratic in the solver error (the error vanishes on the fixed DOFs, the
residual on the free ones), so the capacitance is far more accurate than
the solver tolerance.

The terminal potentials are their own adjoints (``dC_ij = phi_i^T dK phi_j``),
so a capacitance gradient needs no solve. A kept entry holds the potentials
and the forward's prepared operator (`KeptHandle`): a potential
cotangent solves its adjoint on it, at the adjoint tolerance, and a backward
with no such cotangent frees it. An entry over ``retain_memory`` loses the
operator first and keeps its potentials; its backward then prepares the
operator again (once per entry).

The energy of excitation i in region r is ``E_ri = phi_i^T K_r phi_i / 2``
with ``K_r`` the operator restricted to the region. It is not stationary in
the potential, so ``dE_ri = phi_i^T dK_r phi_i / 2 + (K_r phi_i)^T dphi_i``:
a field pair on the region's own domain, and a potential cotangent
``K_r phi_i`` that joins the adjoint solve above.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem import _mpi
from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import PEC, Conductor, sheet_of, surfaces_of
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import p0, permittivity
from gfem._engine.dispersion import check_static, law_regions
from gfem._engine.domain import (
    domain_key,
    em_region,
    h1_space,
    region_domain,
)
from gfem._engine.fields import dof_shape, rank_piece
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import Derivative, FieldPair, Term, law_key
from gfem._engine.solve_request import SolveRequest
from gfem._field import FieldMap
from gfem._linalg import Handle, Kind, Operator, ResolvedLinearSolver, report
from gfem._linalg.base import free_mask
from gfem._linalg.layout import ONE_PROCESS, DofLayout, RankLayout
from gfem.model import Model, check_surface
from gfem.options import SolverOptions
from gfem.problems import Electrostatic
from gfem.results import ElectrostaticResult, Field

__all__ = [
    "DualContext",
    "ElectrostaticLoads",
    "ElectrostaticRule",
    "KeptHandle",
    "objective_loads",
    "solve_duals",
    "terminal_lifting",
]

#: The linear solver for ``linear_solver="auto"``: CG + BDDC.
DEFAULT_SOLVER: Kind = "bddc_gmres"


@dataclasses.dataclass(eq=False)
class KeptHandle:
    """An electrostatic forward's prepared operator, kept for its backward pass.

    The cache entry's ``handle`` (`gfem._engine.cache.Entry`), modelled on
    `gfem._eigen.kept.KeptOperator`: the cache counts its bytes
    and, through `free_base`, drops the operator of an entry over
    ``SolverOptions.retain_memory`` before it drops any entry, which keeps
    its potentials. Nothing of it goes to the plan pool.

    Attributes:
        handle: The prepared operator, or ``None`` once freed.
    """

    handle: Handle | None

    @property
    def nbytes(self) -> int:
        """The operator's bytes (0 once freed), but `shared`."""
        return 0 if self.handle is None else int(self.handle.nbytes)

    def shared(self) -> dict[int, int]:
        """The bytes the operator may share with others, by identity."""
        shared = getattr(self.handle, "shared", None)
        return {} if shared is None else shared()

    def free_base(self, comm: Any, *, walk: bool = True) -> None:  # noqa: ARG002
        """Free the operator; the entry keeps its potentials.

        Nothing goes to the plan pool, so there is no walk to run.
        """
        self.free()

    def free(self, *, return_plans: bool = False) -> None:  # noqa: ARG002
        """Free the operator and drop it; nothing returns to the pool."""
        if self.handle is not None:
            self.handle.free()
            self.handle = None


@dataclasses.dataclass(frozen=True, eq=False)
class DualContext:
    """What an `ElectrostaticLoads` callback sees: one solve's potentials.

    Attributes:
        model: The model solved.
        potential: ``(ndof, n_terminals)``, the terminal potentials.
        terminals: The terminal names, in column order.
    """

    model: Model
    potential: np.ndarray
    terminals: tuple[str, ...]
    _space: ngs.FESpace
    _built: EngineMesh
    _eps: ngs.CoefficientFunction
    _order: int
    _support: tuple[str, ...]
    _excluded: tuple[tuple[int, int], ...]

    def region_load(self, region: str, terminal: str) -> np.ndarray:
        """``K_r phi_i``: the load of the dual of region ``r``'s energy (file units)."""
        phi = self.potential[:, [self.terminals.index(terminal)]]
        domain = region_domain(self.model, self._built, region)
        gf = ngs.GridFunction(self._space)
        v = self._space.TestFunction()
        form = ngs.LinearForm(self._space)
        form += (
            (self._eps * ngs.grad(gf))
            * ngs.grad(v)
            * quadrature.dx(self._order, definedon=domain)
        )
        gf.vec.FV().NumPy()[:] = phi[:, 0]
        form.Assemble()
        return np.array(form.vec.FV().NumPy())

    def mean_load(self, regions: tuple[str, ...]) -> np.ndarray:
        """``|R|^-1 int_R v``: the load of a potential mean's dual."""
        from gfem._engine.fields import mean_rows

        layout = (self._support, self._excluded, "h1", self._order)
        return mean_rows(self.model.mesh, layout, regions)[0]


#: An electrostatic dual load callback: given a solve's `DualContext`, the
#: loads ``(ndof,)`` (file units, ``int eps_r grad u . grad v``'s) to solve
#: ``K z = load`` for, by key; each dual comes back in the request's
#: `SolveRequest.adjoints` by key, zero on the fixed DOFs.
ElectrostaticLoads = Callable[[DualContext], Mapping[Hashable, np.ndarray]]


def objective_loads(keys: Sequence[tuple[Any, ...]]) -> ElectrostaticLoads:
    """The dual loads of an electrostatic objective's reads.

    A key ``("energy", region, terminal)`` is the load ``K_r phi_i`` of the
    energy of excitation ``terminal`` in ``region``; ``("mean", regions)``
    the averaging load of a potential mean over the regions (the same for
    every terminal).
    """
    keys = tuple(keys)

    def loads(context: DualContext) -> dict[Hashable, np.ndarray]:
        out: dict[Hashable, np.ndarray] = {}
        for key in keys:
            if key[0] == "energy":
                out[key] = context.region_load(key[1], key[2])
            else:
                out[key] = context.mean_load(key[1])
        return out

    return loads


def solve_duals(
    model: Model,
    result: ElectrostaticResult,
    loads: ElectrostaticLoads,
    threads: int | None,
) -> dict[Hashable, np.ndarray]:
    """The duals of ``loads`` on a finished result: one more factorization.

    For `gfem.error_estimate`, and the first mesh of `gfem.refine_mesh`.
    The result does not record its options: the direct solver at the
    result's order, which the estimate's few digits need no iterative
    tolerance for.
    """
    fields = list(result.potential.values())
    order = fields[0].order
    options = SolverOptions(order=order, linear_solver="direct", threads=threads)
    linear = ResolvedLinearSolver.resolve(
        options, default=DEFAULT_SOLVER, problem="electrostatic"
    )
    fixed = tuple(result._fixed)  # noqa: SLF001
    phi = np.stack([np.asarray(f.coefficients) for f in fields], axis=1)
    with task_manager(threads):
        disc = discretization(model.mesh)
        built = disc.built
        space = disc.space(
            (domain_key(model), "h1", order, fixed),
            lambda: h1_space(model, built, order, fixed),
        )
        eps = permittivity(model, built)
        context = DualContext(
            model=model,
            potential=phi,
            terminals=tuple(result.terminals),
            _space=space,
            _built=built,
            _eps=eps,
            _order=order,
            _support=fields[0].support,
            _excluded=fields[0]._excluded,  # noqa: SLF001
        )
        keyed = dict(loads(context))
        if not keyed:
            return {}
        handle = _prepare_form(
            linear, space, eps, order=order, width=len(keyed), tolerance=1e-12
        )
        try:
            return _solve_keyed(handle, space, keyed, reused=False)
        finally:
            handle.free()


def _prepare_form(
    linear: ResolvedLinearSolver,
    space: ngs.FESpace,
    eps: ngs.CoefficientFunction,
    *,
    order: int,
    width: int,
    tolerance: float,
    layout: DofLayout = ONE_PROCESS,
) -> Handle:
    """The prepared operator ``int eps_r grad u . grad v`` (file units).

    Under ranks ``layout`` is the space's `RankLayout`, on which `prepare`
    dispatches.
    """
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += eps * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
    return linear.prepare(
        Operator(
            form, space.FreeDofs(), layout=None if layout is ONE_PROCESS else layout
        ),
        symmetry="hermitian",
        width=width,
        tolerance=tolerance,
    )


def terminal_lifting(
    model: Model,
    built: EngineMesh,
    space: ngs.FESpace,
    surfaces: tuple[str, ...],
    layout: DofLayout = ONE_PROCESS,
) -> np.ndarray:
    """A terminal's lifting: 1 on the vertex DOFs of its surfaces, 0 elsewhere.

    The constant 1 lies in the vertex block of NGSolve's H1 basis, so the
    indicator of the vertex DOFs of the terminal's triangles is its exact
    lifting. Under ranks a rank may hold a terminal vertex without any of its
    triangles: the mask goes through `DofLayout.held_by_any`, so every holder
    of a DOF agrees and the column is a state.

    Returns:
        ``(ndof,)``, a state column.
    """
    mesh = model.mesh
    tags = [mesh.region(s).tag for s in surfaces]
    local = built.triangle_order[
        np.isin(np.asarray(mesh.triangle_tags)[built.triangle_order], tags)
    ]
    nodes = np.unique(np.asarray(mesh.triangles)[local])
    vertex = np.full(len(mesh.imported_nodes), -1, dtype=np.int64)
    vertex[built.node_order] = np.arange(len(built.node_order))
    mask = np.zeros(space.ndof, dtype=bool)
    for v in vertex[nodes]:
        dofs = [d for d in space.GetDofNrs(ngs.NodeId(ngs.VERTEX, int(v))) if d >= 0]
        mask[dofs] = True
    return layout.held_by_any(mask).astype(float)


def _solve_keyed(
    handle: Handle,
    space: ngs.FESpace,
    keyed: Mapping[Hashable, np.ndarray],
    *,
    reused: bool,
) -> dict[Hashable, np.ndarray]:
    """Solve each keyed load on ``handle``: the duals, zero on the fixed DOFs."""
    keys = list(keyed)
    loads = np.stack([np.asarray(keyed[k], float) for k in keys], axis=1)
    loads[~free_mask(space.FreeDofs(), space.ndof)] = 0.0
    z, its = handle.solve_counted(loads)
    report.solve_step(
        handle,
        problem="electrostatic",
        direction="forward",
        iterations=its,
        reused=reused,
        columns="adjoint load",
    )
    return {k: np.array(z[:, i]) for i, k in enumerate(keys)}


class ElectrostaticRule:
    """Host-side forward of an `Electrostatic` problem on a `Model`.

    Construction runs every static check (it needs no array values);
    `forward` reads the model's leaves (node coordinates, material values)
    and solves.
    """

    physics = "em"
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "electrostatic"
    #: Material properties the electrostatic forms read.
    reads = frozenset({"eps_r"})
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs = frozenset({"capacitance", "energy", "potential"})
    #: Outputs that are a rank's own piece under ranks: the ranks
    #: compare their cotangents' shapes, not their values.
    pieces = frozenset({"potential"})

    def __init__(
        self,
        model: Model,
        problem: Electrostatic,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Electrostatic`).

        ``comm`` is the resolved communicator of a distributed solve.

        Raises:
            NotImplementedError: The model has a periodic pair, or a sheet
                field.
            ValueError: A permittivity law has a Drude term (no static
                permittivity).
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used, or ``"direct"`` or ``"petsc"`` under ``comm``
                without the MPI PETSc stack.
        """
        if not isinstance(problem, Electrostatic):
            raise TypeError(f"Expected gfem.Electrostatic, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        # before any assembly
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem="electrostatic", comm=comm
        )
        # a law reads its static permittivity; a Drude term has none
        check_static(model)
        periodic = model._periodic  # noqa: SLF001
        if periodic:
            raise NotImplementedError(
                "Electrostatic does not support periodic pairs in this version "
                f"of gfem, but the model has {periodic[0].label}: a capacitance "
                "matrix is between the terminals of one finite structure. Model "
                "one cell with the terminals as drawn, or use a gfem.Driven "
                "surrogate."
            )
        for b in model.boundaries:
            if sheet_of(b) is not None:
                (surface,) = surfaces_of(b)
                raise NotImplementedError(
                    f"{type(b).__name__} on {surface!r} carries a sheet field "
                    "(fill=), but a partly filled conducting sheet is not an "
                    "electrostatic object: electrostatic conductors are terminals "
                    "at a fixed potential. Fix the metal pattern (a PEC or "
                    "terminal on the metal's own surface), or optimize the "
                    "capacitance through a gfem.Driven or gfem.Eigenmode "
                    "surrogate, where the sheet field is differentiable."
                )
        in_terminal = {s for _, surfaces in problem.terminals for s in surfaces}
        conductors = tuple(
            b.surface
            for b in model.boundaries
            if isinstance(b, (PEC, Conductor)) and b.surface not in in_terminal
        )
        self.ground = tuple(dict.fromkeys(problem.ground + conductors))
        for _, surfaces in problem.terminals:
            for s in surfaces:
                _check_surface(model, s, "terminal")
        for s in problem.ground:
            _check_surface(model, s, "ground")
        _check_apart(model, problem.terminals, self.ground)
        #: Surfaces at a fixed potential: the terminals and the ground.
        self.fixed = tuple(sorted(in_terminal.union(self.ground)))
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain, mesh order.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        # The potentials live on the mesh as given (traced or not).
        self._mesh = model.mesh
        self._excluded = model._domain.excluded  # noqa: SLF001

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve every terminal; return the outputs and, if ``keep``, an entry.

        Args:
            model: The model, with concrete leaves.
            keep: Keep the potentials and the prepared operator for a
                backward pass.
            request: Its ``adjoint_loads`` (`ElectrostaticLoads`, set by
                `gfem.refine_mesh` for an objective's duals,
                `objective_loads`) are solved with the forward on its handle;
                their adjoints land in its ``adjoints`` by key.

        Returns:
            ``{"capacitance": (n, n) array in F, "energy": (len(regions), n)
            array in J, "potential": (ndof, n) array in V, one column per
            terminal}``, and the cache entry (``fields["potential"]``, the
            prepared operator in a `KeptHandle`) or ``None``.

        Raises:
            ValueError: Part of the solve domain touches no terminal or
                ground, so its potential is undetermined.
        """
        request = SolveRequest() if request is None else request
        with task_manager(self.options.threads, self.comm):
            space, built, layout = self._space(model)
            eps = permittivity(model, built)
            handle = self._prepare(space, eps, self.options.tolerance, layout)
            lift = np.stack(
                [
                    terminal_lifting(model, built, space, surfaces, layout)
                    for _, surfaces in self.problem.terminals
                ],
                axis=1,
            )
            phi, its = handle.solve_counted(-handle.apply(lift))
            report.solve_step(
                handle, problem="electrostatic", direction="forward", iterations=its
            )
            phi += lift
            scale = scipy.constants.epsilon_0 * built.scale
            # a load paired with a state: a rank sum, agreed
            c = handle.layout.pair(handle.apply(phi), phi) * scale
            energy = (
                0.5
                * scale
                * self._region_energies(model, built, space, eps, phi, layout=layout)
            )
            if request.adjoint_loads is not None and np.isfinite(c).all():
                self._solve_duals(
                    model,
                    handle,
                    space,
                    built=built,
                    eps=eps,
                    phi=phi,
                    loads=cast("ElectrostaticLoads", request.adjoint_loads),
                    into=request.adjoints,
                )
        c = 0.5 * (c + c.T)
        if not keep or not np.isfinite(c).all():
            handle.free()
        if not np.isfinite(c).all():
            # read from the agreed capacitance: every rank raises it
            raise _mpi.agreed(
                ValueError(
                    "The electrostatic potential is undetermined: some part of the "
                    "solve domain touches no terminal and no ground. Add a ground "
                    "surface there, or exclude that volume."
                )
            )
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=KeptHandle(handle),
                fields={"potential": phi},
            )
        potential = rank_piece(phi, self._field_width, self.comm, dtype=float)
        outputs = {"capacitance": c, "energy": energy, "potential": potential}
        return outputs, entry

    def _field_width(self) -> int:
        """The rows of a potential at the JAX seam: `fields.dof_shape`'s width."""
        return dof_shape(
            self._mesh,
            self.regions,
            self._excluded,
            "h1",
            self.options.order,
            comm=self.comm,
        )[0]

    def _prepare(
        self,
        space: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        tolerance: float,
        layout: DofLayout,
    ) -> Handle:
        """The prepared operator ``int eps_r grad u . grad v`` (file units)."""
        return _prepare_form(
            self.linear,
            space,
            eps,
            order=self.options.order,
            width=len(self.problem.terminals),
            tolerance=tolerance,
            layout=layout,
        )

    def _solve_duals(
        self,
        model: Model,
        handle: Handle,
        space: ngs.FESpace,
        *,
        built: EngineMesh,
        eps: ngs.CoefficientFunction,
        phi: np.ndarray,
        loads: ElectrostaticLoads,
        into: dict[Hashable, np.ndarray],
    ) -> None:
        """Solve the dual ``loads`` on the forward's handle: one back-substitution.

        The duals land in ``into`` by key.
        """
        context = DualContext(
            model=model,
            potential=phi,
            terminals=tuple(self.problem.terminal_names),
            _space=space,
            _built=built,
            _eps=eps,
            _order=self.options.order,
            _support=self.regions,
            _excluded=self._excluded,
        )
        keyed = dict(loads(context))
        if keyed:
            into.update(_solve_keyed(handle, space, keyed, reused=True))

    def _space(self, model: Model) -> tuple[ngs.FESpace, EngineMesh, DofLayout]:
        """The solve space, its mesh, and how it lies across ranks.

        The layout is cached on the discretization next to the space, as
        driven's is; `ONE_PROCESS` without a communicator.
        """
        order = self.options.order
        disc = discretization(model.mesh, (), self.comm)
        built = disc.built
        fixed = self.fixed
        key = (domain_key(model), "h1", order, fixed)
        space = disc.space(key, lambda: h1_space(model, built, order, fixed))
        layout: DofLayout = ONE_PROCESS
        if self.comm is not None:
            comm = self.comm
            layout = disc.space((*key, "layout"), lambda: RankLayout(space, comm))
        return space, built, layout

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:  # noqa: ARG002
        """Shape and dtype of each forward output (from the mesh topology)."""
        n = len(self.problem.terminals)
        with task_manager(self.options.threads, self.comm):
            width = self._field_width()
        return {
            "capacitance": jax.ShapeDtypeStruct((n, n), jnp.float64),
            "energy": jax.ShapeDtypeStruct((len(self.regions), n), jnp.float64),
            "potential": jax.ShapeDtypeStruct((width, n), jnp.float64),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the electrostatic solve depends on input ``kind``.

        The permittivity and the node coordinates are read; every other
        material property (losses, ``mu_r``, thermal values), boundary and
        port value is not, so its gradient is an exact zero. Of a
        permittivity law, the static permittivity reads eps_inf and the
        strengths only; its frequency parameters are exact zeros.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        if kind[0] == "law":
            return "read" if kind[-1] in _STATIC_PARAMETERS else "unread"
        if kind[0] in ("boundary", "port"):
            return "unread"
        return "read"

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the electrostatic output {name!r}. Wrap "
            "it in jax.lax.stop_gradient to use its value without its derivative."
        )

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
    ) -> list[FieldPair]:
        """Fold the capacitance, energy and potential cotangents into field pairs.

        The terminal potentials are their own adjoints for the capacitance:
        ``sum_ij ct_ij dC_ij = sum_i phi_i^T dK (sum_j G_ij phi_j)`` with
        ``G`` the symmetric part of ``ct`` (C is symmetrized), so no solve.
        A potential cotangent ``ct_i`` takes one adjoint solve per terminal,
        all columns at once: ``K phi_i = 0`` on the free DOFs gives
        ``ct_i . dphi_i = -lambda_i^T dK phi_i`` with ``K lambda_i = ct_i``
        (zero on the fixed DOFs), K the SI operator. It solves on the
        forward's kept operator; an entry that lost it prepares it again at
        the adjoint tolerance and keeps it for later pullbacks. A backward
        with no potential or energy cotangent frees the kept operator.

        An energy cotangent ``ct_ri`` is both: the pair
        ``(ct_ri / 2) phi_i^T dK_r phi_i`` on region r's own domain, and the
        potential cotangent ``sum_r ct_ri K_r phi_i`` (see the module
        docstring).
        """
        del wanted  # every wanted input reads the same pairs
        phi = entry.fields["potential"]
        eps0 = scipy.constants.epsilon_0
        order = self.options.order
        with task_manager(self.options.threads, self.comm):
            space, built, layout = self._space(model)
            pairs = []
            left, right = [], []
            if "capacitance" in cotangents:
                ct = np.asarray(cotangents["capacitance"], dtype=float)
                left.append(phi)
                right.append(phi @ (0.5 * (ct + ct.T)).T)
            load = None
            if "potential" in cotangents:
                # a rank's rows, then its padding
                load = np.array(cotangents["potential"], dtype=float)[: phi.shape[0]]
            if "energy" in cotangents:
                ct = np.asarray(cotangents["energy"], dtype=float)
                eps = permittivity(model, built)
                through = np.zeros_like(phi)
                for r, name in enumerate(self.regions):
                    if not ct[r].any():
                        continue
                    region = region_domain(model, built, name)
                    through += (
                        self._region_apply(space, eps, region, phi, layout) * ct[r]
                    )
                    pairs.append(
                        FieldPair(
                            space=space,
                            left=phi * (0.5 * ct[r]),
                            right=phi,
                            weights=self._pair_weights(model),
                            order=order,
                            domain=region,
                        )
                    )
                if pairs:
                    # the adjoint solve reads the free DOFs only
                    through[~free_mask(space.FreeDofs(), space.ndof)] = 0.0
                    through *= eps0 * built.scale
                    load = through if load is None else load + through
            if load is None:
                if entry.handle is not None:
                    entry.handle.free()
                    entry.handle = None
            else:
                tolerance = adjoint_tolerance(self.options)
                handle = None if entry.handle is None else entry.handle.handle
                kept = handle is not None
                if handle is None:
                    eps = permittivity(model, built)
                    handle = self._prepare(space, eps, tolerance, layout)
                    entry.handle = KeptHandle(handle)
                # the handle solves the form without eps0 * scale
                adjoint, its = handle.solve_counted(load, tolerance=tolerance)
                report.solve_step(
                    handle,
                    problem="electrostatic",
                    direction="backward",
                    iterations=its,
                    reused=kept,
                )
                adjoint /= eps0 * built.scale
                left.append(-adjoint)
                right.append(phi)
            if left:
                pairs.append(
                    FieldPair(
                        space=space,
                        left=np.concatenate(left, axis=1),
                        right=np.concatenate(right, axis=1),
                        weights=self._pair_weights(model),
                        order=order,
                        domain=em_region(model, built),
                    )
                )
        return pairs

    def _region_apply(
        self,
        space: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        region: ngs.Region,
        phi: np.ndarray,
        layout: DofLayout,
    ) -> np.ndarray:
        """``K_r phi_i`` per column: the operator on ``region`` only (file units).

        Each column is a load: under ranks a shared DOF's value is split over its
        holders.
        """
        gf = ngs.GridFunction(space)
        v = space.TestFunction()
        form = ngs.LinearForm(space)
        form += (
            (eps * ngs.grad(gf))
            * ngs.grad(v)
            * quadrature.dx(self.options.order, definedon=region)
        )
        out = np.zeros_like(phi)
        for i in range(phi.shape[1]):
            gf.vec.FV().NumPy()[:] = phi[:, i]
            layout.mark_state(gf.vec)
            form.Assemble()
            out[:, i] = form.vec.FV().NumPy()
        return out

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:  # noqa: ARG002
        """The electrostatic form ``int Re(eps_r) grad u . grad v``, as a `Term`.

        The pairs' weight ``eps0`` and the length power 1 make it SI. Its
        coefficient reads ``r = Re(eps_r)``, so ``dc/dr = 1``, and along
        each principal axis of a field with axes ``dC/dr_k = a_k a_k^T``
        (per-axis factor 1). The terminal
        liftings are DOF values (1 on a terminal's DOFs, 0 elsewhere), so
        they do not move with the nodes.

        Each law region adds a term of its own (`_law_term`), read only by
        the law pass: the static permittivity eps_inf + sum of the strengths
        is linear in them, so each one's form is the region's
        ``grad u . grad v`` (its coefficient 1 there, 0 elsewhere).
        """
        _, built, _ = self._space(model)
        laws = [
            Term(
                _law_term(lr.region),
                "gradient",
                coefficient=p0(built, lr.mask.astype(float)),
                length_power=1,
                law=lr.region,
            )
            for lr in law_regions(model)
        ]
        return [
            *laws,
            Term(
                "grad_grad",
                "gradient",
                coefficient=permittivity(model, built),
                length_power=1,
                derivatives={
                    "eps_r": Derivative(
                        1.0,
                        holomorphic=False,
                        per_axis=Derivative(1.0, holomorphic=False),
                    )
                },
            ),
        ]

    @staticmethod
    def _pair_weights(model: Model) -> dict[str, float]:
        """The pairs' weights by term name and `law_key`.

        ``eps0`` on the form, and on each law parameter the static
        permittivity reads: d eps(0) / dp = 1 on its region's `_law_term`.
        """
        eps0 = scipy.constants.epsilon_0
        weights = {"grad_grad": eps0}
        for lr in law_regions(model):
            name = _law_term(lr.region)
            weights[law_key(name, ("eps_inf",))] = eps0
            for index in range(len(lr.law.terms)):
                weights[law_key(name, (index, "delta_eps"))] = eps0
        return weights

    def result(self, outputs: dict) -> ElectrostaticResult:
        """The public result from the forward outputs."""
        names = self.problem.terminal_names
        potential = FieldMap(
            {
                name: Field(
                    coefficients=jnp.asarray(outputs["potential"][:, i]),
                    mesh=self._mesh,
                    quantity="V",
                    unit="V",
                    space="h1",
                    order=self.options.order,
                    support=self.regions,
                    _excluded=self._excluded,
                    _comm=_mpi.static_comm(self.comm),
                )
                for i, name in enumerate(names)
            },
            label="terminal",
        )
        return ElectrostaticResult(
            capacitance=jnp.asarray(outputs["capacitance"]),
            terminals=names,
            energy={
                r: jnp.asarray(e)
                for r, e in zip(self.regions, outputs["energy"], strict=True)
            },
            potential=potential,
            _fixed=self.fixed,
        )

    def _region_energies(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        phi: np.ndarray,
        *,
        layout: DofLayout,
    ) -> np.ndarray:
        """``phi_i^T K_r phi_i``, shape ``(len(self.regions), n_terminals)``.

        One element pass per terminal (a P0 linear form on the pinned rule),
        summed per region over its active tetrahedra. Under ranks each rank
        sums its own tetrahedra, zero elsewhere, and the sums are rank-summed.
        """
        order = self.options.order
        mesh = model.mesh
        p0 = ngs.L2(built.mesh, order=0)
        q = p0.TestFunction()
        gf = ngs.GridFunction(space)
        grad = ngs.grad(gf)
        form = ngs.LinearForm(p0)
        form += (eps * grad) * grad * q * quadrature.dx(order)
        per_tet = np.zeros((len(mesh.tetrahedra), phi.shape[1]))
        for i in range(phi.shape[1]):
            gf.vec.FV().NumPy()[:] = phi[:, i]
            layout.mark_state(gf.vec)
            form.Assemble()
            per_tet[built.tetrahedron_order, i] = form.vec.FV().NumPy()
        active = model._domain.active  # noqa: SLF001
        out = np.zeros((len(self.regions), phi.shape[1]))
        for k, region in enumerate(self.regions):
            tets = mesh.elements(region)
            out[k] = per_tet[tets[active[tets]]].sum(axis=0)
        return _mpi.reduce_bcast(self.comm, out)


#: The law parameters the static permittivity reads: eps(0) = eps_inf + the
#: strengths (a Drude term, which has no strength, is refused).
_STATIC_PARAMETERS = frozenset({"eps_inf", "delta_eps"})


def _law_term(region: str) -> str:
    """The name of the law pass's term of law region ``region``."""
    return f"grad_grad static {region}"


def _check_surface(model: Model, name: str, role: str) -> None:
    check_surface(
        model.mesh,
        name,
        f"Electrostatic {role}",
        " Give the surfaces of the conductor's shell instead.",
    )
    if not model._domain.touches(name):  # noqa: SLF001
        raise ValueError(
            f"Electrostatic {role} surface {name!r} does not border the solve "
            "domain (it lies inside excluded volumes). Name a surface of the "
            "domain, or exclude fewer volumes."
        )


def _nodes(model: Model, surfaces: tuple[str, ...]) -> np.ndarray:
    mesh = model.mesh
    tags = [mesh.region(s).tag for s in surfaces]
    return np.unique(mesh.triangles[np.isin(mesh.triangle_tags, tags)])


def _check_apart(
    model: Model,
    terminals: tuple[tuple[str, tuple[str, ...]], ...],
    ground: tuple[str, ...],
) -> None:
    """Terminals and the ground must not touch: touching conductors are one."""
    groups = [(f"terminal {n!r}", _nodes(model, s)) for n, s in terminals]
    if ground:
        groups.append(("the ground", _nodes(model, ground)))
    for i, (a, na) in enumerate(groups):
        for b, nb in groups[i + 1 :]:
            shared = np.intersect1d(na, nb).size
            if shared:
                raise ValueError(
                    f"Electrostatic: {a} and {b} touch ({shared} shared mesh "
                    "nodes), so they are one conductor at one potential. Merge "
                    "them into one terminal, or separate them in the geometry."
                )
