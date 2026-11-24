r"""The heat a solved temperature carries out of a side (`gfem.heat_flow`'s host).

The flow is read from the residual :math:`r = K T - F` of the discrete heat
equation, never from the temperature gradient. :math:`K` is the full
operator (conduction plus convection) and :math:`F` the full load (sources,
convection ambient, heat flux), both assembled on every DOF, before the
restriction to the free ones, so a load landing on a fixed DOF reaches the
reaction. The test weight is a sum of vertex hat functions: NGSolve's H1
basis is hierarchical, so the vertex hats are the partition of unity and
the edge, face and cell bubbles get weight 0.

A side is a `Side`, five arrays of weights read on a set of elements and
faces, and every flow is

.. math::

    Q = s \left[ \psi^T (F_E - K^{conv}_E T) + w^T K^{cond}_E T
        + 1^T (K^{conv}_O T - F^{bnd}_O) \right]

with :math:`s` the mesh scale (the forms are in file units, see
`gfem._engine.thermal`), :math:`E` the side's elements and faces,
:math:`\psi` the `Side.inner` weight, :math:`w` the `Side.fixed` weight
and :math:`O` the named convection and flux faces measured by their own
integral (`Side.outer`, weighted by every vertex hat):

- a set of elements :math:`V`: :math:`\psi` is 1 on the free vertex DOFs of
  :math:`V`'s closure and :math:`w` 1 on its fixed vertex DOFs, on
  :math:`V`'s elements and the boundary faces all of whose thermal
  neighbours are in :math:`V`. This is minus the residual of :math:`V`'s own
  forms on its free closure DOFs, written with the conduction on the fixed
  DOFs alone (the vertex hats sum to 1 on :math:`V`, so the two agree): the
  source power plus the flux in, minus the convection out, plus the
  reactions of the fixed DOFs computed from :math:`V`'s elements only. A
  side whose closure holds no fixed DOF and no convection face measures
  its source power, with no conductivity in it at all.
- surface names: :math:`\psi = 1` and :math:`w = -1` on the fixed vertex
  DOFs of the named fixed surfaces (minus the residual there: the
  reaction), on every element and face; the named convection faces give
  :math:`\int h (T - T_\infty)` and the named flux faces minus their power.

`Balance` is that flow as an affine function of the temperature and the
source densities, with its explicit conductivity pair. A transient
heat-out trace reuses both: the residual gains the capacity
:math:`C \dot u`, which enters as :math:`-\psi^T C_E \Delta u` on the same
elements.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import ngsolve as ngs
import numpy as np

from gfem._boundaries import Convection, HeatFlux
from gfem._engine import quadrature
from gfem._engine.coefficients import surface_p0
from gfem._engine.domain import boundary_region
from gfem._engine.fields import vertex_dofs
from gfem._engine.sensitivity import FieldPair, Integrated, element_integrals
from gfem._engine.sides import SideSpec, selection
from gfem._linalg.base import free_mask

if TYPE_CHECKING:
    from gfem._engine.thermal import ThermalRule
    from gfem.mesh import Mesh
    from gfem.model import Model

__all__ = [
    "Balance",
    "BoundaryTerms",
    "Side",
    "balance",
    "boundary_terms",
    "capacity_pair",
    "side",
    "stored",
]


@dataclasses.dataclass(frozen=True)
class Side:
    """The test weights of a side (see the module).

    Attributes:
        elements: Per tetrahedron, whether the flow reads its forms
            (conduction, sources, and in a transient its capacity).
        faces: Per gfem triangle, whether the flow reads its boundary terms
            (convection, ambient and flux loads, face heat) with `inner`.
        inner: Per DOF, the weight on the load minus the convection.
        fixed: Per DOF, the weight on the conduction (on fixed DOFs only).
        outer: Per gfem triangle, a named convection or flux face measured
            by its own integral.
    """

    elements: np.ndarray
    faces: np.ndarray
    inner: np.ndarray
    fixed: np.ndarray
    outer: np.ndarray


@dataclasses.dataclass(frozen=True)
class Balance:
    r"""A side's heat flow as an affine function, in W (a slab's on a section).

    .. math::

        Q = c^T (T - T_r 1) + \sum_s b_s^T (T - T_{\infty,s} 1)
            + \mathrm{source}^T q + \mathrm{face\_source}^T q_s + Q_0

    for the temperature's coefficients :math:`T`, the power density
    :math:`q` per tetrahedron (W/m³) and :math:`q_s` per gfem triangle
    (W/m²). :math:`1` is the constant 1 (1 on every vertex DOF), which the
    conduction maps to zero and each convection to its ambient load, so the
    shifts change nothing in exact arithmetic; they keep the absolute
    temperature's few hundred kelvin out of the round-off. :math:`T_r` is
    any reference temperature (`flow` takes the mean on the conduction's
    DOFs).

    Attributes:
        side: The weights it was assembled from.
        conduction: :math:`c = s K^{cond}_E w` per DOF, W/K.
        convection: Per `gfem.Convection`, :math:`b_s` per DOF (W/K) and the
            ambient :math:`T_{\infty,s}` (K).
        ones: The constant 1 per DOF.
        source: :math:`\partial Q / \partial q` per tetrahedron, m³.
        face_source: :math:`\partial Q / \partial q_s` per gfem triangle, m².
        constant: :math:`Q_0`, the heat fluxes' part, W.
    """

    side: Side
    conduction: np.ndarray
    convection: tuple[tuple[np.ndarray, float], ...]
    ones: np.ndarray
    source: np.ndarray
    face_source: np.ndarray
    constant: float

    @property
    def temperature(self) -> np.ndarray:
        r""":math:`\partial Q / \partial T` per DOF, W/K."""
        out = self.conduction.copy()
        for b, _ in self.convection:
            out = out + b
        return out

    def flow(
        self,
        t: np.ndarray,
        density: np.ndarray,
        face_density: np.ndarray | None = None,
    ) -> float:
        """The heat leaving the side, W."""
        t = np.asarray(t, dtype=float)
        on = self.conduction != 0
        reference = float(t[on].mean()) if on.any() else 0.0
        out = self.conduction @ (t - reference * self.ones)
        for b, ambient in self.convection:
            out = out + b @ (t - ambient * self.ones)
        out = out + self.source @ density + self.constant
        if face_density is not None:
            out = out + self.face_source @ face_density
        return float(out)

    def conductivity(
        self, rule: ThermalRule, model: Model, t: np.ndarray
    ) -> Integrated:
        r"""The explicit pair :math:`\partial Q / \partial k_e` per tetrahedron.

        As the material pass reads it. :math:`Q` reads :math:`k` in
        :math:`s w^T K_E T` (`Side.fixed`):
        :math:`\partial Q / \partial k_e = s \int_e \nabla w \cdot \nabla T`
        on the side's elements. The solve's own pairs carry the rest through
        the temperature's cotangent.
        """
        space, built = rule._space(model)  # noqa: SLF001
        pair = FieldPair(
            space=space,
            left=self.side.fixed[:, None],
            right=np.asarray(t, dtype=float)[:, None],
            weights={"conduction": 1.0},
            order=rule.options.order,
            domain=rule._domain(model, built),  # noqa: SLF001
        )
        values = element_integrals(model, [pair], rule.terms(model, nodes=False))
        per = np.asarray(values.get("conduction", np.zeros(len(self.source))))
        per = np.where(self.side.elements, per, 0.0)
        return Integrated({"conduction": built.scale * per})


@dataclasses.dataclass(frozen=True)
class _Layout:
    """What every side reads: the mesh, the thermal domain and its fixed DOFs."""

    mesh: Mesh
    #: Per tetrahedron, in the thermal domain.
    active: np.ndarray
    #: Per DOF, free (not on a fixed temperature).
    free: np.ndarray
    #: Per gfem node, its vertex DOF (-1 off the thermal domain).
    nodes: np.ndarray
    #: Per gfem triangle, its two tetrahedra (-1: none).
    neighbours: np.ndarray
    #: Per gfem triangle and side, a tetrahedron of the thermal domain.
    beside: np.ndarray
    #: The measurement's name in messages.
    who: str

    def dofs(self, nodes: np.ndarray, *, fixed: bool) -> np.ndarray:
        """The fixed (or free) vertex DOFs of the gfem ``nodes``."""
        on = self.nodes[nodes]
        on = on[on >= 0]
        return on[self.free[on] != fixed]


def side(
    rule: ThermalRule, model: Model, which: SideSpec, *, who: str = "heat_flow"
) -> Side:
    """The weights of ``which``: volume regions, a predicate or surface names.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side does not fit (see `gfem.heat_flow`).
    """
    mesh = model.mesh
    space, built = rule._space(model)  # noqa: SLF001
    nodes = np.full(len(mesh.imported_nodes), -1, dtype=np.int64)
    numbers = vertex_dofs(space, built)
    nodes[: len(numbers)] = numbers
    neighbours = np.asarray(mesh.triangle_neighbours)
    layout = _Layout(
        mesh=mesh,
        active=rule.active,
        free=free_mask(space.FreeDofs(), space.ndof),
        nodes=nodes,
        neighbours=neighbours,
        beside=(neighbours >= 0) & rule.active[np.maximum(neighbours, 0)],
        who=who,
    )
    if not callable(which):
        volumes = [n for n in which if mesh.region(n).dim == mesh.dim]
        surfaces = [n for n in which if n not in volumes]
        if volumes and surfaces:
            raise ValueError(
                f"{who}'s side mixes volume regions {volumes} and surfaces "
                f"{surfaces}: give either the volume regions of one side of a cut, "
                "or the surfaces the heat leaves through."
            )
        if surfaces:
            return _surface_side(layout, which, rule.boundaries)
    return _volume_side(layout, which)


def _volume_side(layout: _Layout, which: SideSpec) -> Side:
    who, active, mesh = layout.who, layout.active, layout.mesh
    selected = selection(mesh, which, subject=f"{who}'s side") & active
    if not selected.any():
        raise ValueError(
            f"{who}'s side selects no element of the thermal domain, so it has no "
            "surface: select elements of the thermal domain, or name the sink "
            "surfaces the heat leaves through."
        )
    if not (active & ~selected).any():
        raise ValueError(
            f"{who}'s side selects every element of the thermal domain, so it has "
            "no surface: name the sink surfaces the heat leaves through instead."
        )
    closure = np.unique(mesh.tetrahedra[selected])
    inner = np.zeros(len(layout.free))
    fixed = np.zeros(len(layout.free))
    inner[layout.dofs(closure, fixed=False)] = 1.0
    fixed[layout.dofs(closure, fixed=True)] = 1.0
    # a face belongs to the side when every thermal element beside it does
    beside = layout.beside
    inside = beside & selected[np.maximum(layout.neighbours, 0)]
    faces = beside.any(axis=1) & (inside == beside).all(axis=1)
    return Side(
        elements=selected,
        faces=faces,
        inner=inner,
        fixed=fixed,
        outer=np.zeros(len(faces), bool),
    )


def _surface_side(layout: _Layout, names: tuple[str, ...], boundaries: Any) -> Side:
    who, mesh = layout.who, layout.mesh
    dirichlet = boundaries.dirichlet
    measured = {s for s, _, _ in boundaries.convection} | set(boundaries.flux_surfaces)
    triangles = np.asarray(mesh.triangles)
    outer = np.zeros(len(triangles), bool)
    weight = np.zeros(len(layout.free))
    for name in names:
        faces = mesh.elements(name)
        count = layout.beside[faces].sum(axis=1)
        if not count.any():
            raise ValueError(
                f"{who}: surface {name!r} touches no element of the thermal domain, "
                "so no heat crosses it: name the sink surfaces of the thermal domain."
            )
        if name in dirichlet:
            weight[layout.dofs(np.unique(triangles[faces]), fixed=True)] = 1.0
        elif name in measured:
            outer[faces] = True
        elif (count == 2).any():
            raise ValueError(
                f"{who}: surface {name!r} is inside the thermal domain and carries "
                "no thermal boundary, so no heat leaves the domain through it: give "
                "the volume regions on one side of it instead."
            )
    _check_shared(layout, names, dirichlet)
    return Side(
        elements=layout.active.copy(),
        faces=layout.beside.any(axis=1),
        inner=weight,
        fixed=-weight,
        outer=outer,
    )


def _check_shared(
    layout: _Layout, names: tuple[str, ...], dirichlet: tuple[str, ...]
) -> None:
    """Raise when a named fixed surface shares fixed vertex DOFs with an unnamed one.

    Those DOFs' reaction has no element-wise split between the two surfaces.
    """
    mesh = layout.mesh
    triangles = np.asarray(mesh.triangles)

    def fixed_dofs(name: str) -> np.ndarray:
        return layout.dofs(np.unique(triangles[mesh.elements(name)]), fixed=True)

    for name in (n for n in names if n in dirichlet):
        mine = fixed_dofs(name)
        for other in (d for d in dirichlet if d not in names):
            if np.isin(mine, fixed_dofs(other)).any():
                raise ValueError(
                    f"{layout.who}: fixed-temperature surfaces {name!r} and "
                    f"{other!r} share mesh nodes, whose reaction cannot be split "
                    f"between them: measure their union, side=({name!r}, {other!r})."
                )


def balance(rule: ThermalRule, model: Model, weights: Side) -> Balance:
    """The flow of ``weights`` on ``model``'s conductivity and boundary data."""
    space, built = rule._space(model)  # noqa: SLF001
    order = rule.options.order
    scale = built.scale
    u, v = space.TnT()
    inner, fixed = weights.inner, weights.fixed
    ones = np.zeros(space.ndof)
    ones[vertex_dofs(space)] = 1.0

    k = rule._coefficient(  # noqa: SLF001
        model, built, "thermal_conductivity", weights.elements
    )
    conduction = ngs.BilinearForm(space, symmetric=True)
    conduction += k * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
    conduction.Assemble()
    masks = _Masks(
        faces=surface_p0(built, weights.faces.astype(float)),
        outer=surface_p0(built, weights.outer.astype(float)),
    )
    convection = [
        (_convection_weight(space, built, order, masks, s, h, ones, inner), ambient)
        for s, h, ambient in rule.boundaries.convection
    ]
    flux = sum(
        (
            _flux_weight(space, built, order, masks, s, value, ones, inner)
            for s, value in rule.boundaries.flux
        ),
        0.0,
    )
    weight = ngs.GridFunction(space)
    weight.vec.FV().NumPy()[:] = inner
    source = np.zeros(len(weights.elements))
    p0 = ngs.L2(built.mesh, order=0)
    form = ngs.LinearForm(p0)
    form += weight * p0.TestFunction() * quadrature.dx(order)
    form.Assemble()
    source[built.cell_order] = form.vec.FV().NumPy()
    source = np.where(weights.elements, source * scale**3, 0.0)
    face_source = np.zeros(len(weights.faces))
    s0 = ngs.SurfaceL2(built.mesh, order=0)
    form = ngs.LinearForm(s0)
    form += weight * s0.TestFunction() * quadrature.ds(order)
    form.Assemble()
    face_source[built.facet_order] = form.vec.FV().NumPy()
    face_source = np.where(weights.faces, face_source * scale**2, 0.0)
    return Balance(
        side=weights,
        conduction=scale * _apply(conduction, fixed),
        convection=tuple((scale * b, a) for b, a in convection),
        ones=ones,
        source=source,
        face_source=face_source,
        constant=scale * float(flux),
    )


@dataclasses.dataclass(frozen=True)
class BoundaryTerms:
    r"""A side's flow differentiated in the thermal boundary values, explicitly.

    :math:`Q` is affine in the convection coefficients, the ambients and the
    flux values, at a fixed temperature :math:`T`:

    .. math::

        \frac{\partial Q}{\partial h_s} = w_s^T (T - T_{\infty,s} 1),\quad
        \frac{\partial Q}{\partial T_{\infty,s}} = -h_s w_s^T 1,\quad
        \frac{\partial Q}{\partial q_s} = f_s,

    with :math:`w_s = b_s / h_s` the surface's convection weight at unit
    coefficient and :math:`f_s` its flux part of `Balance.constant` at unit
    flux. A fixed temperature enters through :math:`T` alone. They are
    assembled per unit value, so a flux or a coefficient of 0 has its
    derivative.

    Attributes:
        weights: Per `gfem.Convection` (by index in the model's boundaries),
            :math:`w_s` per DOF, W/K².
        fluxes: Per `gfem.HeatFlux` (by index), :math:`f_s`, W per (W/m²).
        ones: The constant 1 per DOF.
    """

    weights: dict[int, np.ndarray]
    fluxes: dict[int, float]
    ones: np.ndarray

    def derivative(self, model: Model, index: int, name: str, t: np.ndarray) -> float:
        r""":math:`\partial Q / \partial` the ``name`` value of boundary ``index``.

        At the temperature coefficients ``t``; ``"temperature"`` has none.
        """
        item = model.boundaries[index]
        if isinstance(item, HeatFlux):
            return self.fluxes[index]
        if not isinstance(item, Convection):
            return 0.0
        if name == "coefficient":
            ambient = float(np.asarray(item.ambient))
            return float(self.weights[index] @ (np.asarray(t) - ambient * self.ones))
        if name == "ambient":
            h = float(np.asarray(item.coefficient))
            return -h * float(self.weights[index] @ self.ones)
        return 0.0


def boundary_terms(rule: ThermalRule, model: Model, weights: Side) -> BoundaryTerms:
    """The explicit boundary-value derivatives of ``weights``' flow on ``model``."""
    space, built = rule._space(model)  # noqa: SLF001
    order = rule.options.order
    ones = np.zeros(space.ndof)
    ones[vertex_dofs(space)] = 1.0
    masks = _Masks(
        faces=surface_p0(built, weights.faces.astype(float)),
        outer=surface_p0(built, weights.outer.astype(float)),
    )
    inner = weights.inner
    scale = built.scale
    conv: dict[int, np.ndarray] = {}
    flux: dict[int, float] = {}
    for i, item in enumerate(model.boundaries):
        if isinstance(item, Convection):
            args = (space, built, order, masks, item.surface)
            conv[i] = scale * _convection_weight(*args, 1.0, ones, inner)
        elif isinstance(item, HeatFlux):
            args = (space, built, order, masks, item.surface)
            flux[i] = scale * _flux_weight(*args, 1.0, ones, inner)
    return BoundaryTerms(weights=conv, fluxes=flux, ones=ones)


@dataclasses.dataclass(frozen=True)
class _Masks:
    """The side's faces and named faces as surface-constant coefficients."""

    faces: Any
    outer: Any


def _convection_weight(  # noqa: PLR0917 - one assembly's inputs
    space: ngs.FESpace,
    built: Any,
    order: int,
    masks: _Masks,
    surface: str,
    h: float,
    ones: np.ndarray,
    inner: np.ndarray,
) -> np.ndarray:
    """:math:`b_s / s` per DOF of one convection surface at coefficient ``h``."""
    u, v = space.TnT()
    ds = quadrature.ds(order, definedon=boundary_region(built, (surface,)))
    parts = []
    for mask in (masks.faces, masks.outer):
        form = ngs.BilinearForm(space, symmetric=True)
        form += mask * h * built.scale * u * v * ds
        form.Assemble()
        parts.append(form)
    return _apply(parts[1], ones) - _apply(parts[0], inner)


def _flux_weight(  # noqa: PLR0917 - one assembly's inputs
    space: ngs.FESpace,
    built: Any,
    order: int,
    masks: _Masks,
    surface: str,
    value: float,
    ones: np.ndarray,
    inner: np.ndarray,
) -> float:
    """The flow's part from the flux ``value`` on ``surface``, without the scale."""
    v = space.TestFunction()
    ds = quadrature.ds(order, definedon=boundary_region(built, (surface,)))
    out = 0.0
    for mask, weight, sign in ((masks.faces, inner, 1.0), (masks.outer, ones, -1.0)):
        form = ngs.LinearForm(space)
        form += mask * value * built.scale * v * ds
        form.Assemble()
        out += sign * float(weight @ form.vec.FV().NumPy())
    return out


def stored(rule: ThermalRule, model: Model, weights: Side) -> np.ndarray:
    r"""The side's heat capacity per DOF, J/K: :math:`s\, C_E \psi`.

    The heat stored in the side since :math:`t = 0` is ``stored`` times
    :math:`u_n - u_0`, so a transient heat out is the balance's flow less it
    (the capacity term of the residual, on the same elements and with the
    same weight).
    """
    space, built = rule._space(model)  # noqa: SLF001
    form = ngs.BilinearForm(space, symmetric=True)
    form += rule._capacity_term(model, built, space, where=weights.elements)  # noqa: SLF001
    form.Assemble()
    return built.scale * _apply(form, weights.inner)


def capacity_pair(
    rule: ThermalRule, model: Model, weights: Side, du: np.ndarray
) -> Integrated:
    r"""The explicit pair of ``-stored`` times ``du`` per tetrahedron.

    For the material pass: with :math:`\Delta u` = ``du``, its derivative in
    the capacity coefficient :math:`c = \rho c_p s^2`
    (`gfem._engine.thermal.TransientThermalRule.terms`) is
    :math:`-s \int_e \psi \, \Delta u` on each of the side's elements
    :math:`e`.
    """
    space, built = rule._space(model)  # noqa: SLF001
    pair = FieldPair(
        space=space,
        left=weights.inner[:, None],
        right=np.asarray(du, dtype=float)[:, None],
        weights={"capacity": 1.0},
        order=rule.options.order,
        domain=rule._domain(model, built),  # noqa: SLF001
    )
    values = element_integrals(model, [pair], rule.terms(model, nodes=False))
    per = np.asarray(values.get("capacity", np.zeros(len(weights.elements))))
    per = np.where(weights.elements, per, 0.0)
    return Integrated({"capacity": -built.scale * per})


def _apply(form: ngs.BilinearForm, x: np.ndarray) -> np.ndarray:
    """``A x`` for the assembled ``form``, on every DOF."""
    vec = form.mat.CreateColVector()
    out = form.mat.CreateRowVector()
    vec.FV().NumPy()[:] = x
    out.data = form.mat * vec
    return out.FV().NumPy().copy()
