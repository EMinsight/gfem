r"""The p-multigrid's direct set, and its count before any space exists.

The p-multigrid solves its direct set exactly and smooths every other DOF
with vertex-star blocks. The direct set is every Whitney (lowest-order
Nédélec) DOF and every DOF on the closure (edges, faces, interior) of a
marked cell: a cell of a perfectly matched layer, inside which no cheap
smoother converges, or a poor cell, whose connected layers stall the
smoother. A cell is poor when its cell aspect, the longest edge over the
shortest altitude,

$$a = \frac{L_{\max} S_{\max}}{3 V},$$

exceeds `POOR_ASPECT` ($a = \sqrt{6}/2 \approx 1.22$ for the regular
tetrahedron, 2-3.5 on sound meshes). Scattered poor cells cost nothing, so
marking every one of them only spends memory on the cells that need it.

`direct_set_count` counts the direct set from the mesh alone, for the memory
estimate made before assembly; `direct_set` builds it on a first-kind
Nédélec space. Both mark the cells through `direct_cells`, so the estimate
and the solver cannot disagree.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import ngsolve as ngs
import numpy as np

#: Cells of a larger aspect are poor: every DOF of theirs joins the direct set.
#: Above about 10 a connected layer adds iterations (+3-4 at 20, +9-13 at 50-65);
#: 5 would double a gmsh microstrip's direct set for one iteration less.
POOR_ASPECT = 10.0

# Vertex pairs of a tetrahedron's six edges and triples of its four faces.
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def cell_aspect(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    r"""The cell aspect $L_{\max} S_{\max}/(3V)$ of each tetrahedron.

    Args:
        nodes: Vertex coordinates, ``(n, 3)``, in any length unit.
        tets: Vertex numbers of each tetrahedron, ``(m, 4)``.

    Returns:
        ``(m,)``: the longest edge over the shortest altitude; ``inf`` for a
        flat cell.
    """
    p = np.asarray(nodes, dtype=float)[np.asarray(tets)]
    edges = p[:, _EDGES[:, 1]] - p[:, _EDGES[:, 0]]
    longest = np.linalg.norm(edges, axis=2).max(axis=1)
    a, b, c = (p[:, _FACES[:, k]] for k in range(3))
    largest = np.linalg.norm(np.cross(b - a, c - a), axis=2).max(axis=1) / 2
    volume = (
        np.abs(np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2]))
        / 6
    )
    with np.errstate(divide="ignore"):
        return longest * largest / (3 * volume)


class DirectSetCount(NamedTuple):
    """The direct set of a mesh, counted from its topology.

    Attributes:
        direct: DOFs on the direct set, the Whitney ones included, fixed DOFs
            counted as `gfem._domain.nedelec_unknowns` counts them.
        whitney: Whitney DOFs, one per edge.
        poor_cells: Cells whose aspect exceeds `POOR_ASPECT`.
    """

    direct: int
    whitney: int
    poor_cells: int


def direct_cells(
    nodes: np.ndarray, tets: np.ndarray, pml: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """The cells whose closure joins the direct set, and how many are poor.

    Args:
        nodes: Vertex coordinates, ``(n, 3)``.
        tets: The solve domain's tetrahedra, ``(m, 4)``.
        pml: Per tetrahedron, whether it is a cell of a perfectly matched
            layer; ``None`` for none.

    Returns:
        ``(m,)`` marks (PML or poor), and the number of poor cells, PML ones
        included.
    """
    poor = cell_aspect(nodes, tets) > POOR_ASPECT
    marked = poor if pml is None else poor | np.asarray(pml, dtype=bool)
    return marked, int(poor.sum())


def direct_set_count(
    nodes: np.ndarray, tets: np.ndarray, order: int, pml: np.ndarray | None = None
) -> DirectSetCount:
    r"""Count the direct set of first-kind Nédélec elements of ``order``.

    $D = W + (p-1)E_X + p(p-1)F_X + \tfrac{p(p-1)(p-2)}{2}T_X$ over the
    edges, faces and tetrahedra of the marked cells $X$ (`direct_cells`),
    with $W$ the edges of ``tets``: the size of `direct_set` on a space
    without fixed DOFs, known before the space exists.

    Args:
        nodes: Vertex coordinates, ``(n, 3)``.
        tets: The solve domain's tetrahedra, ``(m, 4)``.
        order: The element order $p$.
        pml: As `direct_cells`.
    """
    tets = np.asarray(tets)
    marked, poor = direct_cells(nodes, tets, pml)
    whitney = _distinct(tets[:, _EDGES])
    x = tets[marked]
    p = order
    higher = (
        (p - 1) * _distinct(x[:, _EDGES])
        + p * (p - 1) * _distinct(x[:, _FACES])
        + p * (p - 1) * (p - 2) // 2 * len(x)
    )
    return DirectSetCount(whitney + higher, whitney, poor)


@dataclasses.dataclass(frozen=True, eq=False)
class DirectSet:
    """The direct set of a first-kind Nédélec space.

    Attributes:
        dofs: The space's free DOFs on the direct set.
        pml_dofs: The space's free DOFs on the closure of PML cells.
        poor_cells: As `DirectSetCount.poor_cells`.
    """

    dofs: ngs.BitArray
    pml_dofs: int
    poor_cells: int


def direct_set(
    space: ngs.FESpace,
    nodes: np.ndarray,
    tets: np.ndarray,
    elements: np.ndarray,
    pml: np.ndarray | None = None,
) -> DirectSet:
    """Build the direct set of ``space``.

    Without fixed DOFs it holds exactly `direct_set_count`'s ``direct``;
    fixed DOFs are left out of it.

    Args:
        space: First-kind Nédélec space on the solve domain, compressed, so
            that its first DOFs are the Whitney ones of its edges, in order.
        nodes: Vertex coordinates, ``(n, 3)``.
        tets: The space's tetrahedra, ``(m, 4)``: every cell it is defined
            on, once.
        elements: The NGSolve volume element of each row of ``tets``.
        pml: As `direct_cells`.

    Raises:
        ValueError: ``tets`` are not the cells of ``space`` (its size is not
            their first-kind Nédélec count).
    """
    tets = np.asarray(tets)
    elements = np.asarray(elements)
    marked, poor = direct_cells(nodes, tets, pml)
    whitney = _distinct(tets[:, _EDGES])
    order = space.globalorder
    expected = (
        order * whitney
        + order * (order - 1) * _distinct(tets[:, _FACES])
        + order * (order - 1) * (order - 2) // 2 * len(tets)
    )
    if space.ndof != expected:
        raise ValueError(
            f"The direct set needs the space's own cells: {len(tets)} cells give "
            f"{expected} order-{order} DOFs, the space has {space.ndof}."
        )
    free = space.FreeDofs()
    layer = np.zeros(len(tets), dtype=bool) if pml is None else np.asarray(pml, bool)
    on_layer = _closure(space, elements[layer])
    dofs = on_layer | _closure(space, elements[marked & ~layer])
    # NGSolve numbers each edge's lowest-order DOF first; Compress keeps the order
    dofs[:whitney] = True
    on_layer_free = ngs.BitArray(on_layer.tolist()) & free
    return DirectSet(ngs.BitArray(dofs.tolist()) & free, on_layer_free.NumSet(), poor)


def _closure(space: ngs.FESpace, elements: np.ndarray) -> np.ndarray:
    """Whether each DOF of ``space`` lies on the closure of these volume elements.

    Read from the diagonal of a mass form on those elements alone, positive on
    exactly their DOFs, so no Python loop runs over the elements.
    """
    out = np.zeros(space.ndof, dtype=bool)
    if len(elements) == 0:
        return out
    chosen = np.zeros(space.mesh.ne, dtype=bool)
    chosen[elements] = True
    cells = ngs.BitArray(chosen.tolist())
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, diagonal=True)
    form += u * v * ngs.dx(definedonelements=cells)
    form.Assemble()
    out[:] = np.abs(np.asarray(form.mat.AsVector().FV().NumPy())) > 0
    return out


def _distinct(rows: np.ndarray) -> int:
    """How many distinct vertex sets the rows of ``(m, k, j)`` vertex numbers hold."""
    rows = np.sort(rows.reshape(-1, rows.shape[-1]), axis=1)
    if len(rows) == 0:
        return 0
    rows = rows[np.lexsort(rows.T[::-1])]
    return 1 + int(np.any(rows[1:] != rows[:-1], axis=1).sum())
