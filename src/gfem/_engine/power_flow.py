"""The power a solved driven field carries out of a set of elements.

The host kernel of `gfem.power_flow`. With A the selected elements, the
driven equations ``a(E, v) = l(v)`` tested with ``v = conj(χ_A E)`` (E on
the DOFs of A's edges, faces and cells, zero elsewhere) split into the
terms on A, whose imaginary part is minus the power A's ports feed and
its losses absorb, and the terms on the outside elements and faces that
share an edge with A. So the outside layer's terms give the power crossing
A's boundary exactly for the discrete field, with no point evaluation:

    P = Im[conj(χ_A e) · Σ_k θ_k(ω) A_k|layer e] / (2 ω mu0 scale)

(the forms are scaled by ``mu0 * scale``, see `terms`). A face term belongs
to A only when every solve-domain element beside it is in A. A port's
term needs its excitation, which a field does not record, so an outside
port face that shares an edge with A raises. A Floquet surface is no
exception: inside A its terms (the absorbing term and the rank-one
corrections, which couple that surface's DOFs only) belong to A, and outside
A it raises when it shares an edge with A, as other ports do.

On a model with `gfem.Periodic` pairs the field holds e = P x on the
unpaired space, Bloch phases included, and the identity holds there with
the unpaired operator, once "shares an edge" and "the DOFs of A" are read
across the pairs: edges are keyed by the tie classes of their nodes, so an
edge and its images are one, and χ_A also keeps the DOFs of the images of
A's edges and faces (the test function ``conj(χ_A x)`` of the periodic
space, expanded). No wave vector is needed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem import _pml
from gfem._boundaries import FloquetPort, surfaces_of
from gfem._engine import probe
from gfem._engine.driven import DrivenOperator, driven_operator
from gfem._engine.mesh import inverse_order
from gfem._periodic import tie_classes
from gfem._pml import Stretch
from gfem.mesh import _TET_EDGES, Mesh, _same_topology

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "Flux",
    "Surface",
    "build",
    "check_field",
    "check_nodes",
    "check_size",
    "power_flow",
    "surface",
]

#: The three edges of a triangle, by vertex.
_TRI_EDGES = np.array([[0, 1], [0, 2], [1, 2]])


@dataclasses.dataclass(frozen=True)
class Surface:
    """The element sets of a side's surface: topology only.

    Attributes:
        tie: The tie class of every node (`tie_classes`).
        inside: The selected tetrahedra of the field's support.
        layer: The outside tetrahedra that share an edge with ``inside``:
            the elements whose terms the power reads.
        rim: The inside tetrahedra that share an edge with the outside.
        faces: The triangles whose terms the power reads (`_outside_faces`).
    """

    tie: np.ndarray
    inside: np.ndarray
    layer: np.ndarray
    rim: np.ndarray
    faces: np.ndarray

    def read_elements(self, mesh: Mesh) -> np.ndarray:
        """The tetrahedra whose materials the power reads.

        The layer, and the solve-domain tetrahedra beside a face it reads.
        """
        out = self.layer.copy()
        nb = mesh.triangle_neighbours[self.faces]
        out[nb[nb >= 0]] = True
        return out


@dataclasses.dataclass(frozen=True)
class Flux:
    """The operator and test mask of a surface, ready to contract a field.

    ``P = Im[conj(D e) . M e] / c0`` with ``D = diag(mask)``, ``M`` the sum
    of the non-port pieces at ω and ``c0 = 2 ω mu0 scale``.
    """

    op: DrivenOperator
    mask: np.ndarray

    def _apply(self, x: np.ndarray, omega: float) -> np.ndarray:
        """``M x`` at ``omega``, ``x`` of shape ``(ndof,)`` or ``(ndof, m)``."""
        out = np.zeros(x.shape, complex)
        for k, piece in enumerate(self.op.pieces):
            if piece.kind != "port":
                out += complex(piece.theta(omega)) * self.op.apply(k, x)
        return out

    def _scale(self, omega: float) -> float:
        return 2 * omega * const.mu_0 * self.op.built.scale

    def power(self, e: np.ndarray, omega: float) -> float:
        """The power leaving the side, W."""
        w = np.where(self.mask, e, 0.0)
        return float(np.imag(np.vdot(w, self._apply(e, omega)))) / self._scale(omega)

    def pullback(self, e: np.ndarray, omega: float) -> np.ndarray:
        """JAX's gradient of `power` in ``e``: ``dP/dx - i dP/dy = 2 conj(H e)``.

        ``P = e^H H e`` with ``H = (D M - M^H D) / (2i c0)`` Hermitian, and
        ``M^H D e = conj(M conj(D e))``: the unpaired pieces are complex
        symmetric.
        """
        w = np.where(self.mask, e, 0.0)
        both = self._apply(np.stack([e, np.conj(w)], axis=1), omega)
        h = (np.where(self.mask, both[:, 0], 0.0) - np.conj(both[:, 1])) / (
            2j * self._scale(omega)
        )
        return 2 * np.conj(h)


def power_flow(
    field: Field,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> float:
    """Net time-averaged power leaving ``side``, W (see `gfem.power_flow`).

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side, the field or a port near the surface does not
            fit (see `gfem.power_flow`).
    """
    check_nodes(field, model)
    where = surface(field, side, model)
    flux = build(field.order, model, where, stretch=_pml.field_stretch(model, field))
    e = np.asarray(field.coefficients)
    check_size(e, flux.op, field.order)
    return flux.power(e, float(np.asarray(field.omega).real))


def surface(
    field: Field,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> Surface:
    """The surface of ``side``, from the mesh topology.

    A predicate side reads the node coordinates too.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side or the field does not fit, or the surface is
            within one element of a port (see `gfem.power_flow`).
    """
    mesh = model.mesh
    support = _check_support(field, model)
    inside = _selection(mesh, side) & support
    outside = support & ~inside
    if not inside.any():
        raise ValueError(
            "The side selects no element of the field's support, so it has no "
            "surface: select elements of the solve domain."
        )
    if not outside.any():
        raise ValueError(
            "The side selects every element of the field's support, so it has "
            "no surface: the side holds the whole solve domain."
        )
    tie = tie_classes(len(mesh.nodes), model._periodic)  # noqa: SLF001
    tet_edges = _edge_keys(tie, mesh.tetrahedra, np.array(_TET_EDGES))
    closure = np.unique(tet_edges[inside])
    layer = outside & np.isin(tet_edges, closure).any(axis=1)
    rim = inside & np.isin(tet_edges, np.unique(tet_edges[outside])).any(axis=1)
    faces = _outside_faces(mesh, tie, inside, support, closure)
    for port in model.ports:
        kind = "FloquetPort" if isinstance(port, FloquetPort) else "port"
        for name in surfaces_of(port):
            if faces[mesh.elements(name)].any():
                raise ValueError(
                    f"The surface of this side is within one element of {kind} "
                    f"{port.name!r}'s surface {name!r}: power_flow cannot "
                    "count a port's excitation. Move the surface at least one "
                    "element away from the port, or put the port's elements "
                    "inside the side."
                )
    return Surface(tie=tie, inside=inside, layer=layer, rim=rim, faces=faces)


def build(
    order: int, model: Model, where: Surface, *, stretch: Stretch | None = None
) -> Flux:
    """The `Flux` of a surface on ``model`` at its node positions.

    ``stretch`` is the field's perfectly matched layer stretch
    (`gfem._pml.field_stretch`).
    """
    op = driven_operator(
        model,
        order,
        elements=where.layer,
        faces=where.faces,
        paired=False,
        stretch=stretch,
    )
    return Flux(op=op, mask=_dofs(op, where.tie, where.rim, where.rim | where.layer))


def check_size(e: np.ndarray, op: DrivenOperator, order: int) -> None:
    """Raise unless the coefficients fit the driven operator's space.

    Raises:
        ValueError: Another number of DOFs.
    """
    if e.shape[0] != op.space.ndof:
        raise ValueError(
            f"The field has {e.shape[0]} coefficients but the model's order-"
            f"{order} driven space has {op.space.ndof} DOFs: the field "
            "was not solved on this model."
        )


def check_nodes(field: Field, model: Model) -> None:
    """Raise unless the field's node positions are the model's.

    Raises:
        ValueError: They differ.
    """
    if not np.array_equal(np.asarray(field.mesh.nodes), np.asarray(model.mesh.nodes)):
        raise _other_mesh()


def _other_mesh() -> ValueError:
    return ValueError(
        "The field lives on another mesh (or other node positions) than the "
        "model: pass the model the field was solved with."
    )


def check_field(field: Field, model: Model) -> np.ndarray:
    """The field's support mask, after checking it was solved on ``model``.

    Raises:
        ValueError: The mesh, its nodes or the support differ.
    """
    check_nodes(field, model)
    return _check_support(field, model)


def _check_support(field: Field, model: Model) -> np.ndarray:
    """The field's support mask, after checking its topology is the model's.

    Raises:
        ValueError: The mesh topology or the support differ.
    """
    mesh = model.mesh
    if not _same_topology(field.mesh, mesh):
        raise _other_mesh()
    excluded = tuple(field._excluded)  # noqa: SLF001
    layout = probe.Layout(tuple(field.support), excluded, field.space, field.order)
    support = probe.in_support(mesh, layout)
    if not np.array_equal(support, model._domain.active):  # noqa: SLF001
        raise ValueError(
            "The field's support is not the model's solve domain (other regions "
            "or exclusions): pass the model the field was solved with."
        )
    return support


def _selection(
    mesh: Mesh, side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """The selected tetrahedra: named volume regions or a centroid predicate.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface, or the predicate does not return
            one boolean per tetrahedron.
    """
    n = len(mesh.tetrahedra)
    if callable(side):
        centroids = np.asarray(mesh.nodes)[mesh.tetrahedra].mean(axis=1)
        out = np.asarray(side(centroids))
        if out.dtype != bool or out.shape != (n,):
            raise ValueError(
                f"The side predicate must return one boolean per tetrahedron "
                f"({n},) from the (n_tetrahedra, 3) element centroids, got an "
                f"array of {out.dtype} of shape {out.shape}."
            )
        return out
    out = np.zeros(n, bool)
    for name in side:
        if mesh.region(name).dim != 3:
            raise ValueError(
                f"{name!r} is a surface region; power_flow's side is a set of "
                "elements: volume regions, or a predicate on element centroids."
            )
        out[mesh.elements(name)] = True
    return out


def _edge_keys(tie: np.ndarray, cells: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """One integer per edge of each cell, the same for the same edge.

    Nodes are read as their tie class representatives (`tie_classes`), so
    an edge and its images across the periodic pairs share a key.
    """
    ends = np.sort(tie[cells[:, pairs]], axis=-1)
    return ends[..., 0].astype(np.int64) * len(tie) + ends[..., 1]


def _outside_faces(
    mesh: Mesh,
    tie: np.ndarray,
    inside: np.ndarray,
    support: np.ndarray,
    closure: np.ndarray,
) -> np.ndarray:
    """Triangles whose terms belong to the outside and reach A's DOFs.

    A face term belongs to A only when every solve-domain element beside
    the face is in A; the outside's reach A when the face shares an edge
    with A.
    """
    nb = mesh.triangle_neighbours
    tet = np.maximum(nb, 0)
    beside = (nb >= 0) & support[tet]
    owned = beside.any(axis=1) & ~(~beside | inside[tet]).all(axis=1)
    touching = np.isin(_edge_keys(tie, mesh.triangles, _TRI_EDGES), closure)
    return owned & touching.any(axis=1)


def _dofs(
    op: DrivenOperator, tie: np.ndarray, elements: np.ndarray, near: np.ndarray
) -> np.ndarray:
    """The DOFs of the cells, edges and faces of the given gfem tetrahedra.

    With the images of those edges and faces across the periodic pairs
    (same `tie_classes` key), looked up on the ``near`` tetrahedra, which
    hold every image the operator reaches. Without pairs the images are
    the edges and faces themselves.
    """
    space, ngmesh = op.space, op.built.mesh
    ngs_index = inverse_order(op.built.tetrahedron_order, len(elements))
    out = np.zeros(space.ndof, bool)

    def mark(node: ngs.NodeId) -> None:
        dofs = np.asarray(space.GetDofNrs(node), dtype=np.int64)
        out[dofs[dofs >= 0]] = True

    def key(node: ngs.NodeId) -> tuple[int, ...]:
        return tuple(sorted(int(tie[v.nr]) for v in ngmesh[node].vertices))

    keys = set()
    for t in ngs_index[np.flatnonzero(elements)]:
        element = ngmesh[ngs.ElementId(ngs.VOL, int(t))]
        keys.update(key(n) for n in (*element.edges, *element.faces))
        mark(ngs.NodeId(ngs.CELL, int(t)))
    for t in ngs_index[np.flatnonzero(near)]:
        element = ngmesh[ngs.ElementId(ngs.VOL, int(t))]
        for n in (*element.edges, *element.faces):
            if key(n) in keys:
                mark(n)
    return out
