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

`PmgHandle` is the ``pmg_gmres`` linear solver: GMRES (`KrylovHandle`, with
its preconditioner reuse across a sweep) preconditioned by one symmetric
V-cycle (`VCycle`) of the two levels. The direct level is the operator
restricted to the direct set, factored by gfem's direct inverse: MKL PARDISO
with gfem's flags (`gfem._linalg.pardiso`), sparse Cholesky without it. The
smoother is block Gauss-Seidel over vertex stars (every free DOF on an edge,
face or cell around a vertex), on a copy of the operator in symmetric
storage, for which NGSolve factors each block as an unpivoted band
$L D L^\mathsf{T}$: half the memory of dense inverses, and a faster setup.
GMRES itself runs on the operator in general storage, whose product is
faster. Each setup logs one INFO record on ``gfem.solver``
(`gfem._linalg.report.pmg_build`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import ngsolve as ngs
import numpy as np

from gfem import _timing
from gfem._linalg import Operator, Symmetry, report
from gfem._linalg.base import free_mask, host_bytes
from gfem._linalg.direct import INVERSE, Factors, UpdateFactors
from gfem._linalg.krylov_handle import KrylovHandle
from gfem._linalg.pardiso import PardisoFactors, pardiso_inverse

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


class Cells(NamedTuple):
    """The cells of a solve domain, as `direct_set` reads them.

    Attributes:
        nodes: Vertex coordinates, ``(n, 3)``.
        tets: The tetrahedra the space is defined on, ``(m, 4)``.
        elements: The NGSolve volume element of each row of ``tets``.
        pml: Per row of ``tets``, whether it is a cell of a perfectly
            matched layer; ``None`` for none.
    """

    nodes: np.ndarray
    tets: np.ndarray
    elements: np.ndarray
    pml: np.ndarray | None = None


def mesh_cells(mesh: ngs.Mesh) -> Cells:
    """Every volume element of an NGSolve mesh, none in a perfectly matched layer."""
    ngmesh = mesh.ngmesh
    tets = np.asarray(ngmesh.Elements3D().NumPy()["nodes"][:, :4], dtype=np.int64) - 1
    return Cells(np.asarray(ngmesh.Coordinates()), tets, np.arange(len(tets)))


def vertex_stars(space: ngs.FESpace, free: np.ndarray) -> list[list[int]]:
    """The free DOFs on each vertex star of ``space``, one block per vertex.

    A vertex star holds every DOF on an edge, face or cell around the vertex.
    Vertices without a free DOF give no block.
    """
    blocks = []
    for star in space.CreateSmoothingBlocks(blocktype="vertexpatch"):
        dofs = np.fromiter(star, dtype=np.int64)
        dofs = np.sort(dofs[free[dofs]])
        if len(dofs):
            blocks.append(dofs.tolist())
    return blocks


def _band_bytes(blocks: list[list[int]], *, is_complex: bool) -> int:
    """At most the bytes of the blocks' band factors: one triangle each.

    Counted in 64-bit integers: the square of a large block count overflows 32.
    """
    sizes = np.fromiter((len(b) for b in blocks), dtype=np.int64, count=len(blocks))
    entry = 16 if is_complex else 8
    return int(entry * (sizes * (sizes + 1) // 2).sum())


def _factor(
    matrix: ngs.BaseMatrix, dofs: ngs.BitArray, symmetry: Symmetry, *, is_complex: bool
) -> Factors:
    """The direct level: ``matrix`` on ``dofs``, factored as the direct solver does.

    gfem's PARDISO (`pardiso_inverse`) with the ``mkl`` extra, sparse
    Cholesky without it. The inverse is zero off ``dofs``.
    """
    pardiso = (
        pardiso_inverse(symmetry, is_complex=is_complex) if dofs.NumSet() else None
    )
    inverse = matrix.Inverse(dofs, inverse=pardiso or INVERSE)
    if pardiso is None:
        return UpdateFactors(inverse, dofs.NumSet())
    mask = free_mask(dofs, int(matrix.height))
    return PardisoFactors(inverse, mask, is_complex=is_complex)


class VCycle(ngs.BaseMatrix):
    r"""One symmetric V-cycle of the two-level p-multigrid, as an NGSolve matrix.

    $y = M x$: from $y = 0$, one forward block Gauss-Seidel sweep on
    $A y = x$, the direct correction $y \mathrel{+}= A_{DD}^{-1} (x - A y)_D$
    on the direct set $D$, one backward sweep. The cycle is linear in $b$, and complex
    symmetric when $A$ is.

    Attributes:
        form: The form the cycle was built on: its matrix is the residual's.
        direct: DOFs on the direct set.
        pml_dofs: DOFs on the closure of a PML cell.
        poor_cells: Poor cells (`POOR_ASPECT`).
    """

    def __init__(
        self,
        form: ngs.BilinearForm,
        symmetric: ngs.BilinearForm,
        blocks: list[list[int]],
        direct: DirectSet,
        symmetry: Symmetry,
        *,
        is_complex: bool,
    ) -> None:
        """Factor the direct level and the smoother's blocks of an assembled form.

        ``symmetric`` is ``form``'s operator assembled in symmetric storage,
        the smoother's; ``form``'s matrix (general storage) is the residual's
        and the direct level's.
        """
        super().__init__()
        self.form = form
        matrix = self._matrix = form.mat
        self._symmetric = symmetric
        self._complex = is_complex
        self._smoother = symmetric.mat.CreateBlockSmoother(blocks, parallel=True)
        self._band = _band_bytes(blocks, is_complex=is_complex)
        self._factors = _factor(matrix, direct.dofs, symmetry, is_complex=is_complex)
        self.direct = direct.dofs.NumSet()
        self.pml_dofs = direct.pml_dofs
        self.poor_cells = direct.poor_cells
        self._r = matrix.CreateColVector()
        self._z = matrix.CreateColVector()

    @property
    def mat(self) -> ngs.BaseMatrix:
        """The cycle: what a Krylov solve applies."""
        return self

    def IsComplex(self) -> bool:  # noqa: N802 - NGSolve's name
        return self._complex

    def Height(self) -> int:  # noqa: N802
        return int(self._matrix.height)

    def Width(self) -> int:  # noqa: N802
        return int(self._matrix.width)

    def CreateRowVector(self) -> ngs.BaseVector:  # noqa: N802  # ty: ignore[invalid-method-override]
        return self._matrix.CreateRowVector()

    def CreateColVector(self) -> ngs.BaseVector:  # noqa: N802  # ty: ignore[invalid-method-override]
        return self._matrix.CreateColVector()

    def Mult(self, x: ngs.BaseVector, y: ngs.BaseVector) -> None:  # noqa: N802
        """``y = M x`` (see the class docstring)."""
        r, z = self._r, self._z
        y[:] = 0
        self._smoother.Smooth(y, x)
        r.data = x - self._matrix * y
        z.data = self._factors.inverse * r
        y.data += z
        self._smoother.SmoothBack(y, x)

    @property
    def nbytes(self) -> int:
        """Host memory of the symmetric copy, the band factors and the direct level.

        The general-storage matrix is the handle's, not counted here.
        """
        return host_bytes(self._symmetric.mat) + self._band + int(self._factors.nbytes)


class PmgHandle(KrylovHandle):
    r"""GMRES on one V-cycle of the p-multigrid, reused by a solve per right-hand side.

    The operator must equal its transpose: the smoother factors its blocks
    as $L D L^\mathsf{T}$. The Krylov solve, its warm start and the
    preconditioner reuse across a sweep are `KrylovHandle`'s; this handle
    assembles the operator and builds the V-cycle on it (`VCycle`), and logs
    each build (`gfem._linalg.report.pmg_build`).
    """

    kind = "pmg_gmres"
    preconditioner_name = "p-multigrid"
    # layers of thin cells are the one measured cause of extra iterations
    convergence_remedy = (
        "Use linear_solver='direct' if it fits, or a looser tolerance, and check "
        "the mesh for layers of thin cells."
    )

    def __init__(
        self,
        operator: Operator,
        symmetry: Symmetry,
        *,
        tolerance: float,
        maxiter: int,
        reuse: bool = True,
        width: int = 1,
        solves: int = 1,
    ) -> None:
        """Assemble the form, build the V-cycle, and set up GMRES on it.

        ``reuse=False`` builds the V-cycle again at every `refactorize`.
        ``width`` and ``solves`` are read by the reuse rule only.
        """
        self._cells: Callable[[], Cells] | None = operator.cells
        self._freedofs = operator.freedofs
        super().__init__(
            operator,
            symmetry,
            tolerance=tolerance,
            maxiter=maxiter,
            reuse=reuse,
            width=width,
            solves=solves,
        )

    def _set_up(self, form: ngs.BilinearForm) -> VCycle:
        """Assemble ``form``, then build the V-cycle on its matrix."""
        with _timing.phase("assemble"):
            form.Assemble()
        space = form.space
        is_complex = bool(space.is_complex)
        cells = mesh_cells(space.mesh) if self._cells is None else self._cells()
        direct = direct_set(space, *cells)
        symmetric = ngs.BilinearForm(space, symmetric=True, symmetric_storage=True)
        for integrator in form.integrators:
            symmetric += integrator
        symmetric.Assemble()
        free = free_mask(self._freedofs, int(space.ndof))
        cycle = VCycle(
            form,
            symmetric,
            vertex_stars(space, free),
            direct,
            self.symmetry,
            is_complex=is_complex,
        )
        report.pmg_build(
            direct=cycle.direct,
            pml_dofs=cycle.pml_dofs,
            poor_cells=cycle.poor_cells,
            unknowns=int(free.sum()),
            frequency=self._frequency,
        )
        return cycle

    @property
    def nbytes(self) -> int:
        """Host memory of the matrix and the V-cycle, in bytes.

        A V-cycle reused at another matrix also holds its own.
        """
        # read once, without the lock: a concurrent free gives 0 or the old count
        held, pre, form = self._matrix, self._preconditioner, self._form
        if self._solver is None or held is None or not isinstance(pre, VCycle):
            return 0
        own = 0 if pre.form is form else host_bytes(pre.form.mat)
        return host_bytes(held) + pre.nbytes + own

    def _drop_setup(self) -> None:
        """Nothing beyond the V-cycle, which goes with the preconditioner."""
