"""DC conduction rule: terminal potentials in the conductors and the conductance.

The electrostatic rule (`gfem._engine.electrostatic`) with the permittivity
replaced by the electrical conductivity, on the conductors only. For
terminal i, the potential phi_i is 1 on terminal i, 0 on the other
terminals and the ground, and solves ``a(phi_i, v) = 0`` with
``a(u, v) = int sigma grad u . grad v`` for every v vanishing on the fixed
surfaces; every other surface of a conductor is insulated. Then
``G_ij = a(phi_i, phi_j)`` with no extra solve, and the dissipation per
tetrahedron at unit voltages is ``D_e,ij = (1 / |e|) int_e sigma grad phi_i
. grad phi_j`` (the exact element mean, on the pinned quadrature rule): a
drive with terminal voltages V heats element e with ``V^T D_e V`` W/m^3.

The conduction domain is derived from the materials: every volume whose
conductivity is not zero, minus the volumes excluded from DC conduction.
A conducting piece (connected through shared mesh nodes, the H1 coupling)
that touches no terminal and no ground carries no current: its volume
entities are left out of the space when the piece covers them whole, and
its remaining DOFs are held at 0 V. Either way its potential, heat and
conductivity gradient are exact zeros.

The default linear solver is ``direct``: the conductor-only space is small
next to the thermal one.

The terminal potentials are their own adjoints for the conductance
(``dG_ij = phi_i^T dK phi_j``); a potential cotangent takes one adjoint
solve on the forward's kept operator, as the electrostatic rule does.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from gfem._adjoint import adjoint_tolerance
from gfem._checks import is_traced
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import element_tensors, p0
from gfem._engine.domain import boundary_region
from gfem._engine.electrostatic import KeptHandle
from gfem._engine.fields import (
    dirichlet_h1,
    face_keys,
    fe_space,
    support_region,
    surface_triangles,
    tetrahedron_face_keys,
)
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import Derivative, FieldPair, Term
from gfem._field import FieldMap
from gfem._linalg import Handle, Kind, Operator, ResolvedLinearSolver, report
from gfem._linalg.base import free_mask
from gfem._materials import Material, MaterialField
from gfem.mesh import Mesh
from gfem.model import Model, check_surface
from gfem.options import SolverOptions
from gfem.problems import DCConduction
from gfem.results import DCConductionResult, Field

__all__ = ["DCConductionRule"]

#: The linear solver for ``linear_solver="auto"``.
DEFAULT_SOLVER: Kind = "direct"
#: The problem name in solver logs and messages.
_PROBLEM = "DC conduction"
#: The physics key of the model's solve domains and exclusions.
_PHYSICS = "dc_conduction"


@dataclasses.dataclass(frozen=True, eq=False)
class _Conductors:
    """The conduction domain of a model and problem, decided before any solve.

    Attributes:
        regions: The conducting volume regions that carry current, mesh
            order: the space's support.
        excluded: Volume entities left out of those regions: excluded from
            DC conduction, or covered whole by a floating piece.
        pinned: Tetrahedra of floating pieces inside kept entities, whose
            DOFs are held at 0 V.
        conducting: Per tetrahedron, whether it is in a conducting region
            and the DC conduction domain (floating pieces included).
    """

    regions: tuple[str, ...]
    excluded: tuple[tuple[int, int], ...]
    pinned: np.ndarray
    conducting: np.ndarray


def _conducts(material: Any) -> bool:
    """Whether a region's material conducts, from structure and fixed values.

    A `Material` conducts when its conductivity is not zero; a
    `MaterialField` when its conductivity is given and, for concrete
    values, not all zero. A traced field's values are unknown before the
    solve: it conducts, and a zero value then raises in the forward.
    """
    value = getattr(material, "conductivity", None)
    if value is None:
        return False
    if isinstance(material, Material):
        return bool(np.any(np.asarray(value, float) != 0))
    if isinstance(material, MaterialField) and is_traced(value):
        return True
    return bool(np.any(np.asarray(value) != 0))


def conductors(model: Model, problem: DCConduction) -> _Conductors:
    """The conduction domain: conducting regions minus floating pieces.

    Raises:
        ValueError: No volume of the DC conduction domain conducts, or a
            terminal or the ground touches no conducting volume.
    """
    mesh = model.mesh
    active = model._domain_of(_PHYSICS).active  # noqa: SLF001
    names = [r for r, m in model.materials.items() if _conducts(m)]
    conducting = np.zeros(len(mesh.tetrahedra), bool)
    for name in names:
        conducting[mesh.elements(name)] = True
    conducting &= active
    if not conducting.any():
        raise ValueError(
            "DCConduction: no volume conducts. The conductors are the volumes "
            "whose material has conductivity > 0 (and that are not excluded "
            'from DC conduction, excluded={"dc_conduction": ...}): give the '
            "conductor's material a conductivity, e.g. "
            "gfem.Material(conductivity=2.3e6)."
        )
    tets = mesh.tetrahedra
    n_nodes = len(mesh.imported_nodes)
    faces = tetrahedron_face_keys(mesh, conducting)
    fixed_nodes = []
    for label, surfaces in (
        *((f"terminal {n!r}", s) for n, s in problem.terminals),
        *((("the ground", problem.ground),) if problem.ground else ()),
    ):
        tris = surface_triangles(mesh, surfaces)
        if not np.isin(face_keys(tris, n_nodes), faces).any():
            raise ValueError(
                f"DCConduction: {label} ({', '.join(repr(s) for s in surfaces)}) "
                "touches no conducting volume. The conductors are the volumes "
                "whose material has conductivity > 0, minus those excluded from "
                'DC conduction (excluded={"dc_conduction": ...}). Put the '
                "terminal on a conductor's surface, or give that conductor's "
                "material a conductivity."
            )
        fixed_nodes.append(np.unique(tris))
    # pieces connected through shared nodes: the tetrahedra and their nodes
    # as one bipartite graph
    index = np.flatnonzero(conducting)
    rows = np.repeat(np.arange(len(index)), 4)
    graph = scipy.sparse.coo_matrix(
        (np.ones(rows.size), (rows, len(index) + tets[index].ravel())),
        shape=(len(index) + n_nodes,) * 2,
    )
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    touched = np.unique(label[len(index) + np.concatenate(fixed_nodes)])
    floating = np.zeros(len(tets), bool)
    floating[index[~np.isin(label[: len(index)], touched)]] = True
    # an entity covered whole by floating pieces leaves the space
    keys = np.stack([mesh.tetrahedron_tags, mesh.tetrahedron_entities], axis=1)
    excluded = set(model._domain_of(_PHYSICS).excluded)  # noqa: SLF001
    pinned = floating.copy()
    for tag, entity in {tuple(k) for k in keys[floating].tolist()}:
        mine = (keys[:, 0] == tag) & (keys[:, 1] == entity)
        if not (mine & conducting & ~floating).any():
            excluded.add((int(tag), int(entity)))
            pinned &= ~mine
    kept = conducting & ~floating | pinned
    regions = tuple(
        r for r in mesh.volumes if r in names and kept[mesh.elements(r)].any()
    )
    return _Conductors(
        regions=regions,
        excluded=tuple(sorted(excluded)),
        pinned=np.flatnonzero(pinned),
        conducting=conducting,
    )


def conductivity(model: Model, built: EngineMesh) -> ngs.CoefficientFunction:
    """The conductivity per element, a scalar or a 3 x 3 tensor (S/m)."""
    values = element_tensors(model, "conductivity").real
    if values.ndim == 1:
        return p0(built, values)
    parts = [p0(built, values[:, i, j]) for i in range(3) for j in range(3)]
    return ngs.CF(tuple(parts), dims=(3, 3))


def _check_positive(model: Model, domain: _Conductors) -> None:
    """Raise for an element of a conducting region with no conductivity.

    Raises:
        ValueError: Some element of a conducting volume has zero (or not
            positive definite) conductivity.
    """
    values = element_tensors(model, "conductivity").real
    if values.ndim == 1:
        good = values > 0
    else:
        good = np.linalg.eigvalsh(values).min(axis=1) > 0
    bad = domain.conducting & ~good
    if not bad.any():
        return
    mesh = model.mesh
    where = next(r for r in mesh.volumes if bad[mesh.elements(r)].any())
    raise ValueError(
        f"DCConduction: {int(bad.sum())} element(s) of the conducting volume "
        f"{where!r} have conductivity <= 0, which leaves their potential "
        "undetermined. Give every element of a conductor a positive "
        "conductivity (a gfem.MaterialField void, e.g. "
        "MaterialField.interpolate(density, void=..., solid=...) with a small "
        f"positive void conductivity), or exclude {where!r} from DC conduction "
        f'(excluded={{"dc_conduction": ({where!r},)}}) if it carries no current.'
    )


class DCConductionRule:
    """Host-side forward of a `DCConduction` problem on a `Model`.

    Construction runs every static check and decides the conduction domain
    (it needs no array values); `forward` reads the conductivity and solves.
    """

    physics = _PHYSICS
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "dc_conduction"
    #: Material properties the conduction form reads.
    reads = frozenset({"conductivity"})
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs = frozenset({"conductance", "dissipation", "potential"})

    def __init__(
        self, model: Model, problem: DCConduction, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.DCConduction`).

        Raises:
            NotImplementedError: The model has a periodic pair, or the
                device is a GPU.
            ValueError: No volume conducts; a terminal or the ground touches
                no conductor; terminals touch.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used.
        """
        if not isinstance(problem, DCConduction):
            raise TypeError(f"Expected gfem.DCConduction, got {problem!r}.")
        if options.device != "cpu":
            # no GPU smoke test has run for DC conduction
            raise NotImplementedError(
                "DCConduction has no GPU path in this version of gfem "
                f"(device={options.device!r}): solve with device='cpu'."
            )
        self.problem = problem
        self.options = options
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem=_PROBLEM
        )
        periodic = model._periodic  # noqa: SLF001
        if periodic:
            raise NotImplementedError(
                "DCConduction does not support periodic pairs in this version of "
                f"gfem, but the model has {periodic[0].label}: a conductance is "
                "between the terminals of one finite conductor. Model the "
                "conductor without periodic pairs."
            )
        for _, surfaces in problem.terminals:
            for s in surfaces:
                _check_surface(model, s, "terminal")
        for s in problem.ground:
            _check_surface(model, s, "ground")
        _check_apart(model, problem.terminals, problem.ground)
        #: Surfaces at a fixed potential: the terminals and the ground.
        self.fixed = tuple(
            sorted({s for _, ss in problem.terminals for s in ss}.union(problem.ground))
        )
        self.domain = conductors(model, problem)
        if not is_traced(tuple(model.materials.values())):
            _check_positive(model, self.domain)
        #: Volume regions of the space, mesh order.
        self.regions = self.domain.regions
        self._excluded = self.domain.excluded
        self._mesh = model.mesh

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve every terminal; return the outputs and, if ``keep``, an entry.

        Returns:
            ``{"conductance": (n, n) array in S, "dissipation": (n_tets, n,
            n) array in W/m^3 per V^2, "potential": (ndof, n) array in V,
            one column per terminal}``, and the cache entry
            (``fields["potential"]``, the prepared operator in a
            `KeptHandle`) or ``None``.

        Raises:
            ValueError: An element of a conductor has zero conductivity.
        """
        _check_positive(model, self.domain)
        with task_manager(self.options.threads):
            space, built = self._space(model)
            sigma = conductivity(model, built)
            handle = self._prepare(space, sigma, self.options.tolerance)
            lift = np.zeros((space.ndof, len(self.problem.terminals)))
            for i, (_, surfaces) in enumerate(self.problem.terminals):
                gf = ngs.GridFunction(space)
                gf.Set(ngs.CF(1.0), definedon=boundary_region(built, surfaces))
                lift[:, i] = gf.vec.FV().NumPy()
            phi, its = handle.solve_counted(-handle.apply(lift))
            report.solve_step(
                handle, problem=_PROBLEM, direction="forward", iterations=its
            )
            phi += lift
            g = phi.T @ handle.apply(phi) * built.scale
            dissipation = self._dissipation(model, built, space, sigma, phi)
        g = 0.5 * (g + g.T)
        if not keep:
            handle.free()
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=KeptHandle(handle),
                fields={"potential": phi},
            )
        return {"conductance": g, "dissipation": dissipation, "potential": phi}, entry

    def _free_dofs(self, space: ngs.FESpace) -> ngs.BitArray:
        """The space's free DOFs, less those of floating pieces (held at 0 V)."""
        free = ngs.BitArray(space.FreeDofs())
        if self.domain.pinned.size:
            built = discretization(self._mesh).built
            order = np.argsort(built.cell_order)
            for tet in self.domain.pinned:
                element = ngs.ElementId(ngs.VOL, int(order[tet]))
                for dof in space.GetDofNrs(element):
                    if dof >= 0:
                        free.Clear(dof)
        return free

    def _prepare(
        self, space: ngs.FESpace, sigma: ngs.CoefficientFunction, tolerance: float
    ) -> Handle:
        """The prepared operator ``int sigma grad u . grad v`` (file units)."""
        u, v = space.TnT()
        form = ngs.BilinearForm(space, symmetric=True)
        form += sigma * ngs.grad(u) * ngs.grad(v) * quadrature.dx(self.options.order)
        return self.linear.prepare(
            Operator(form, self._free_dofs(space)),
            symmetry="hermitian",
            width=len(self.problem.terminals),
            tolerance=tolerance,
        )

    def _space(self, model: Model) -> tuple[ngs.FESpace, EngineMesh]:
        return dirichlet_h1(
            model.mesh,
            _PHYSICS,
            self.regions,
            self._excluded,
            order=self.options.order,
            fixed=self.fixed,
        )

    def _dissipation(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        sigma: ngs.CoefficientFunction,
        phi: np.ndarray,
    ) -> np.ndarray:
        """``D_e,ij``, shape ``(n_tets, n, n)`` in W/m^3 per V^2, gfem order.

        One P0 linear form per terminal pair, on the pinned rule, divided
        by the element volume: the exact element mean.
        """
        order = self.options.order
        n = phi.shape[1]
        l2 = ngs.L2(built.mesh, order=0)
        q = l2.TestFunction()
        left, right = ngs.GridFunction(space), ngs.GridFunction(space)
        form = ngs.LinearForm(l2)
        form += (sigma * ngs.grad(left)) * ngs.grad(right) * q * quadrature.dx(order)
        volume = ngs.LinearForm(l2)
        volume += q * quadrature.dx(order)
        volume.Assemble()
        # file units: int_e sigma grad . grad / |e| is in S/m per file unit^2
        per_volume = 1.0 / (volume.vec.FV().NumPy() * built.scale**2)
        out = np.zeros((len(model.mesh.tetrahedra), n, n))
        for i in range(n):
            left.vec.FV().NumPy()[:] = phi[:, i]
            for j in range(i, n):
                right.vec.FV().NumPy()[:] = phi[:, j]
                form.Assemble()
                out[built.cell_order, i, j] = form.vec.FV().NumPy() * per_volume
                out[:, j, i] = out[:, i, j]
        out[~self.domain.conducting] = 0.0
        return out

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        n = len(self.problem.terminals)
        mesh = model.mesh
        still = Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001
        with task_manager(self.options.threads):
            _, space, _ = fe_space(
                still, self.regions, self._excluded, "h1", self.options.order
            )
        return {
            "conductance": jax.ShapeDtypeStruct((n, n), jnp.float64),
            "dissipation": jax.ShapeDtypeStruct(
                (len(mesh.tetrahedra), n, n), jnp.float64
            ),
            "potential": jax.ShapeDtypeStruct((space.ndof, n), jnp.float64),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the conduction solve depends on input ``kind``.

        The conductivity and the node coordinates are read; every other
        material property (the permittivity and its laws, losses,
        ``mu_r``, thermal values), boundary and port value is not, so its
        gradient is an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        if kind[0] in ("law", "boundary", "port"):
            return "unread"
        return "read"

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the DC conduction output {name!r} in this "
            "version. Wrap it in jax.lax.stop_gradient to use its value without "
            "its derivative."
        )

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Fold the conductance, dissipation and potential cotangents into pairs.

        As `gfem._engine.electrostatic.ElectrostaticRule.backward` without
        the energy: the potentials are their own adjoints for the
        conductance (no solve); a potential cotangent takes one adjoint
        solve for every terminal at once on the kept operator.

        A dissipation cotangent ``ct_e,ij`` (on ``D_e,ij = int_e sigma grad
        phi_i . grad phi_j / |e|``) is both: an explicit pair ``(phi_i,
        phi_j)`` per terminal pair, whose term weighs each element by its
        symmetric cotangent over its volume (`terms`), and the potential
        cotangent ``sum_j 2 int sigma (ct_ij / |e|) grad phi_j . grad v``.
        """
        phi = entry.fields["potential"]
        order = self.options.order
        self._dissipation_terms: list[Term] = []
        extra: list[FieldPair] = []
        with task_manager(self.options.threads):
            space, built = self._space(model)
            left, right = [], []
            if "dissipation" in cotangents:
                through, extra = self._dissipation_pairs(
                    model, built, space, phi, cotangents["dissipation"]
                )
                cotangents = dict(cotangents)
                if "potential" in cotangents:
                    through = through + np.asarray(cotangents["potential"], float)
                cotangents["potential"] = through
            if "conductance" in cotangents:
                ct = np.asarray(cotangents["conductance"], dtype=float)
                left.append(phi)
                right.append(phi @ (0.5 * (ct + ct.T)).T)
            if "potential" not in cotangents:
                if entry.handle is not None:
                    entry.handle.free()
                    entry.handle = None
            else:
                load = np.array(cotangents["potential"], dtype=float)
                load[~free_mask(self._free_dofs(space), space.ndof)] = 0.0
                tolerance = adjoint_tolerance(self.options)
                handle = None if entry.handle is None else entry.handle.handle
                kept = handle is not None
                if handle is None:
                    handle = self._prepare(space, conductivity(model, built), tolerance)
                    entry.handle = KeptHandle(handle)
                adjoint, its = handle.solve_counted(load, tolerance=tolerance)
                report.solve_step(
                    handle,
                    problem=_PROBLEM,
                    direction="backward",
                    iterations=its,
                    reused=kept,
                )
                adjoint /= built.scale
                left.append(-adjoint)
                right.append(phi)
            pairs = []
            if left:
                pairs.append(
                    FieldPair(
                        space=space,
                        left=np.concatenate(left, axis=1),
                        right=np.concatenate(right, axis=1),
                        weights={"grad_grad": 1.0},
                        order=order,
                        domain=support_region(
                            built, model.mesh, self.regions, self._excluded
                        ),
                    )
                )
        return pairs + extra

    def _dissipation_pairs(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        phi: np.ndarray,
        cotangent: np.ndarray,
    ) -> tuple[np.ndarray, list[FieldPair]]:
        """A dissipation cotangent's potential cotangent and explicit pairs.

        Sets the explicit terms `terms` returns: one per terminal pair
        ``(i, j)``, ``int grad u . grad v`` per element (file units) with
        ``dc / d sigma_e = ct_e,ij scale / |e|`` (the forward's per-volume
        factor, ``|e|`` in m^3).
        """
        from gfem.thermal import _tet_volumes

        ct = np.asarray(cotangent, dtype=float)
        ct = 0.5 * (ct + ct.transpose(0, 2, 1))
        ct[~self.domain.conducting] = 0.0
        weights = ct * (built.scale / _tet_volumes(model.mesh))[:, None, None]
        n = phi.shape[1]
        order = self.options.order
        sigma = conductivity(model, built)
        domain = support_region(built, model.mesh, self.regions, self._excluded)
        v = space.TestFunction()
        fields = []
        for j in range(n):
            gf = ngs.GridFunction(space)
            gf.vec.FV().NumPy()[:] = phi[:, j]
            fields.append(gf)
        through = np.zeros_like(phi)
        pairs = []
        for i in range(n):
            form = ngs.LinearForm(space)
            for j in range(n):
                w = p0(built, 2.0 * weights[:, i, j])
                form += (
                    w
                    * (sigma * ngs.grad(fields[j]))
                    * ngs.grad(v)
                    * quadrature.dx(order)
                )
            form.Assemble()
            through[:, i] = form.vec.FV().NumPy()
            for j in range(n):
                name = f"dissipation {i} {j}"
                w_ij = weights[:, i, j]
                self._dissipation_terms.append(
                    Term(
                        name,
                        "gradient",
                        coefficient=sigma,
                        derivatives={
                            "conductivity": Derivative(
                                lambda w=w_ij: w,
                                holomorphic=False,
                                per_axis=Derivative(
                                    lambda w=w_ij: np.repeat(w[:, None], 3, axis=1),
                                    holomorphic=False,
                                ),
                            )
                        },
                    )
                )
                pairs.append(
                    FieldPair(
                        space=space,
                        left=phi[:, i : i + 1],
                        right=phi[:, j : j + 1],
                        weights={name: 1.0},
                        order=order,
                        domain=domain,
                    )
                )
        return through, pairs

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:  # noqa: ARG002
        """The conduction form ``int sigma grad u . grad v``, as a `Term`.

        The pairs' weight 1 and the length power 1 make it SI; along each
        principal axis of a field with axes ``dC/dv_k = a_k a_k^T``.
        """
        _, built = self._space(model)
        return [
            *getattr(self, "_dissipation_terms", []),
            Term(
                "grad_grad",
                "gradient",
                coefficient=conductivity(model, built),
                length_power=1,
                derivatives={
                    "conductivity": Derivative(
                        1.0,
                        holomorphic=False,
                        per_axis=Derivative(1.0, holomorphic=False),
                    )
                },
            ),
        ]

    def result(self, outputs: dict) -> DCConductionResult:
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
                )
                for i, name in enumerate(names)
            },
            label="terminal",
        )
        return DCConductionResult(
            conductance=jnp.asarray(outputs["conductance"]),
            terminals=names,
            potential=potential,
            dissipation=jnp.asarray(outputs["dissipation"]),
            _fixed=self.fixed,
            _regions=self.regions,
            _thickness=self._mesh.section_thickness,
            _grounded=bool(self.problem.ground) or len(names) > 1,
        )


def _check_surface(model: Model, name: str, role: str) -> None:
    check_surface(
        model.mesh,
        name,
        f"DCConduction {role}",
        " Give the conductor's contact surfaces instead.",
    )
    if not model._domain_of(_PHYSICS).touches(name):  # noqa: SLF001
        raise ValueError(
            f"DCConduction {role} surface {name!r} does not border the DC "
            "conduction solve domain (it lies inside volumes excluded from DC "
            "conduction). Name a surface of a conductor, or exclude fewer volumes "
            '(excluded={"dc_conduction": ...}).'
        )


def _nodes(model: Model, surfaces: tuple[str, ...]) -> np.ndarray:
    return np.unique(surface_triangles(model.mesh, surfaces))


def _check_apart(
    model: Model,
    terminals: tuple[tuple[str, tuple[str, ...]], ...],
    ground: tuple[str, ...],
) -> None:
    """Terminals and the ground must not touch: touching contacts are one."""
    groups = [(f"terminal {n!r}", _nodes(model, s)) for n, s in terminals]
    if ground:
        groups.append(("the ground", _nodes(model, ground)))
    for i, (a, na) in enumerate(groups):
        for b, nb in groups[i + 1 :]:
            shared = np.intersect1d(na, nb).size
            if shared:
                raise ValueError(
                    f"DCConduction: {a} and {b} touch ({shared} shared mesh "
                    "nodes), so they are one contact at one potential. Merge "
                    "them into one terminal, or separate them in the geometry."
                )
