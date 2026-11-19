r"""The tree-cotree gauge basis of a driven operator, and its gauge template.

The driven field :math:`x` on the free DOFs of the H(curl) space :math:`V`
is written :math:`x = T y` with the gauge basis

.. math::

    T = \begin{bmatrix} E_c & G_f & X \end{bmatrix},

square and nonsingular from the free DOFs of the gauge space
:math:`Y = V \times H^1 \times \mathbb{C}^k` to those of :math:`V` (ADR
0217):

- :math:`E_c`: a unit column per free cotree H(curl) DOF;
- :math:`G_f`: the discrete gradient of each free H1 DOF, read off NGSolve's
  ``CreateGradient``; a vertex potential is pivoted on the tree edge to its
  parent in a breadth-first spanning forest, a high-order gradient on its
  single H(curl) entry;
- :math:`X`: one floating-potential column :math:`G\,\mathbf{1}_s` per PEC
  conductor :math:`s` other than its region's reference conductor.

Conductors are the connected components of the fixed (PEC) vertices through
fixed edges, each contracted to one node of the forest. Per connected
electromagnetic region one potential is dropped: its reference conductor
(the one with the most vertices, ties to the lowest index), or one vertex
constant in a region without PEC. T depends on the mesh topology and the
PEC set only, so it is cached per topology and survives node moves.

A piece :math:`A_k` becomes :math:`\hat A_k = T^\mathsf{T} A_k T`; a curl
piece is formed on the cotree block alone, :math:`E_c^\mathsf{T} A_k E_c`,
so its gradient rows and columns hold exact zeros, which the low-frequency
accuracy rests on (ADR 0215). Every :math:`\hat A_k` lies in the symbolic
gauged pattern :math:`|T|^\mathsf{T} |P| \, |T|`, :math:`P` the element graph
of :math:`V`; the gauge template is a complex NGSolve matrix on :math:`Y`
whose free CSR is that pattern, in its order.

Not for periodic pairs, Bloch wave vectors or ranks: `resolve_gauge` decides
whether a driven solve runs through the gauge (``SolverOptions.gauge``, ADR
0220).
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Literal

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from gfem._linalg import report
from gfem._linalg.base import free_mask
from gfem._linalg.pieces import _positions
from gfem._pytree import readonly
from gfem.errors import DistributedUnsupportedError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gfem._engine.driven import DrivenOperator

#: Piece kinds whose form reads the field's curl only: zero on gradients.
CURL_KINDS = frozenset({"curl", "curl trace"})

Reference = Literal["largest", "smallest"]

#: Entries of the gauge space's superset graph masked per chunk (`_template`).
_MASK_CHUNK = 1 << 22


@dataclasses.dataclass(frozen=True, eq=False)
class GaugeBasis:
    r"""The gauge basis T of a driven space and its gauge template.

    Rows of :math:`T` are the free DOFs of the driven space :math:`V`
    (`v_dofs`), columns the free DOFs of the gauge space :math:`Y` (`y_dofs`,
    increasing), so the cotree columns come first. "Y-free numbering" below
    is the column numbering of :math:`T`.

    Attributes:
        T: The gauge basis, real, ``(n, n)`` CSR.
        TT: :math:`T^\mathsf{T}`, CSR.
        v_dofs: The free DOFs of :math:`V`, row ``i`` of :math:`T` is DOF
            ``v_dofs[i]``.
        y_dofs: The free DOFs of :math:`Y`, column ``j`` of :math:`T` is DOF
            ``y_dofs[j]``; increasing.
        cotree: The number of cotree columns, the first ones of :math:`T`.
        cotree_rows: The row of :math:`T` of each cotree column's unit entry.
        ndof_y: The DOFs of :math:`Y`: the driven space's base space, the H1
            space of its gradients, one per floating potential.
        free: The free DOFs of :math:`Y` (`y_dofs`) as a BitArray.
        pattern: The symbolic gauged pattern :math:`|T|^\mathsf{T} |P| \, |T|`
            in Y-free numbering, CSR with sorted indices and unit values.
        ungauged_nnz: The entries of :math:`P` on the free DOFs of
            :math:`V`, the ungauged pattern `pattern` is compared with.
        template: The gauge template: a complex NGSolve sparse matrix on
            :math:`Y`, empty off the free rows, whose free CSR is `pattern`
            in its order. Pattern only (its values are not the operator's);
            never pickled.
        key: A digest of :math:`T` and `y_dofs`: two bases with one key
            gauge alike (the free set of :math:`Y`, the conductors and their
            references).
        conductors: The PEC conductors.
        floating: The floating potentials (conductors that are not a
            reference conductor).
        regions: The connected electromagnetic regions, one dropped
            potential each.
        timings: Build time in seconds: ``"basis"`` (gradient and forest)
            and ``"template"`` (pattern and template).
    """

    T: sp.csr_matrix
    TT: sp.csr_matrix
    v_dofs: np.ndarray
    y_dofs: np.ndarray
    cotree: int
    cotree_rows: np.ndarray
    ndof_y: int
    free: ngs.BitArray
    pattern: sp.csr_matrix
    ungauged_nnz: int
    template: Any
    key: tuple
    conductors: int
    floating: int
    regions: int
    timings: Mapping[str, float]

    def transform(self, piece: Any, kind: str) -> sp.csr_matrix:
        r"""The gauged piece :math:`\hat A_k = T^\mathsf{T} A_k T`, Y-free numbering.

        A piece of a `CURL_KINDS` kind is formed on the cotree block only,
        :math:`E_c^\mathsf{T} A_k E_c`: its gradient rows and columns are
        exact zeros, not round-off.

        Args:
            piece: The assembled piece on the whole driven space, an NGSolve
                sparse matrix or a SciPy sparse matrix.
            kind: The piece's kind (`gfem._engine.terms.Piece.kind`).

        Returns:
            CSR with sorted indices, inside `pattern`.
        """
        a = _to_scipy(piece)
        if kind in CURL_KINDS:
            dofs = self.v_dofs[self.cotree_rows]
            hat = sp.csr_matrix(a[dofs][:, dofs])
            n = self.T.shape[1]
            hat.resize((n, n))
        else:
            a = a[self.v_dofs][:, self.v_dofs]
            hat = sp.csr_matrix(self.TT @ (a @ self.T))
        hat.sum_duplicates()
        hat.sort_indices()
        return hat

    def positions(self, hat: sp.csr_matrix) -> np.ndarray:
        """The positions of ``hat``'s entries in `pattern`'s (and `template`'s) values.

        Raises:
            ValueError: ``hat`` has an entry outside `pattern`.
        """
        at = _positions(
            self.pattern.indptr, self.pattern.indices, hat.indptr, hat.indices
        )
        if at is None:
            raise ValueError(
                "The gauged piece has an entry outside the gauged pattern."
            )
        return at


# Per mesh topology, its bases by space key and reference rule (`gauge_basis`).
_BASES: weakref.WeakKeyDictionary[Any, dict[tuple, GaugeBasis]] = (
    weakref.WeakKeyDictionary()
)
_LOCK = threading.Lock()


def gauge_basis(
    operator: DrivenOperator, *, reference: Reference = "largest"
) -> GaugeBasis:
    """The gauge basis of ``operator``'s space, cached per mesh topology.

    The cache is keyed by the operator's topology and space key (solve
    domain, order, PEC surfaces): a node move keeps the basis.

    Args:
        operator: A driven operator on one process, without periodic pairs
            or a Bloch wave vector.
        reference: Which conductor of each region is its reference: the one
            with the most vertices (``"largest"``, the rule), or the fewest
            (``"smallest"``, to check that the field does not depend on it).
            Ties go to the lowest conductor index.

    Raises:
        ValueError: ``operator`` is periodic, Bloch-phased or distributed.
    """
    if operator.expansion is not None or operator.symmetry != "complex_symmetric":
        raise ValueError("The tree-cotree gauge does not take periodic pairs.")
    if operator.layout is not None:
        raise ValueError("The tree-cotree gauge runs on one process.")
    key = (operator.space_key, reference)
    topology = operator.topology
    with _LOCK:
        held = None if topology is None else _BASES.setdefault(topology, {})
        basis = None if held is None else held.get(key)
        built = basis is None
        if basis is None:
            basis = _build(operator.space, reference)
            if held is not None:
                held[key] = basis
    if built:
        report.gauge_basis(
            v_unknowns=len(basis.v_dofs),
            y_unknowns=len(basis.y_dofs),
            conductors=basis.conductors,
            floating=basis.floating,
            nnz=basis.pattern.nnz,
            ungauged_nnz=basis.ungauged_nnz,
            basis_seconds=basis.timings["basis"],
            template_seconds=basis.timings["template"],
        )
    return basis


def _build(space: Any, reference: Reference) -> GaugeBasis:
    """The gauge basis and template of the compressed H(curl) ``space``."""
    start = time.perf_counter()
    base = space.GetBaseSpace()
    active = _active_dofs(space, base)
    gradient, h1 = base.CreateGradient()
    g = sp.csc_matrix(_to_scipy(gradient)[active])
    g.eliminate_zeros()
    free = free_mask(space.FreeDofs(), space.ndof)
    vertex = np.arange(h1.ndof) < base.mesh.nv
    forest = _forest(g, vertex, free, reference)
    # Y index of each column: cotree DOFs in the base space, then H1, then
    # one number per floating potential
    n_base, n_h1 = base.ndof, h1.ndof
    y = np.concatenate(
        [
            active[forest.cotree],
            n_base + forest.h1_columns,
            n_base + n_h1 + np.arange(forest.floating),
        ]
    )
    order = np.argsort(y, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    t = sp.csr_matrix(
        (forest.values, (forest.rows, rank[forest.columns])),
        shape=(len(y), len(y)),
    )
    t.sum_duplicates()
    t.sort_indices()
    y_dofs = y[order]
    n_cotree = len(forest.cotree)
    # the cotree columns keep their order and come first (base DOFs < n_base)
    cotree_rows = forest.rows[:n_cotree]
    basis_time = time.perf_counter() - start
    start = time.perf_counter()
    n_y = n_base + n_h1 + forest.floating
    pattern, ungauged_nnz, template = _template(
        base,
        h1=h1,
        floating=forest.floating,
        t=t,
        y_dofs=y_dofs,
        v_base=active[forest.free_rows],
    )
    template_time = time.perf_counter() - start
    free_y = np.zeros(n_y, bool)
    free_y[y_dofs] = True
    digest = hashlib.blake2b(digest_size=16)
    for array in (y_dofs, t.indptr, t.indices, t.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return GaugeBasis(
        T=t,
        TT=sp.csr_matrix(t.T),
        v_dofs=readonly(forest.free_rows),
        y_dofs=readonly(y_dofs),
        cotree=n_cotree,
        cotree_rows=readonly(cotree_rows),
        ndof_y=n_y,
        free=ngs.BitArray(free_y.tolist()),
        pattern=pattern,
        ungauged_nnz=ungauged_nnz,
        template=template,
        key=("tree_cotree", digest.hexdigest()),
        conductors=forest.conductors,
        floating=forest.floating,
        regions=forest.regions,
        timings={"basis": basis_time, "template": template_time},
    )


@dataclasses.dataclass(frozen=True)
class _Forest:
    """The columns of T as COO triplets over the free rows, before ordering.

    Columns: ``cotree`` unit columns first, then the gradients of
    ``h1_columns``, then ``floating`` floating potentials.
    """

    rows: np.ndarray  # row of T: position among the free DOFs
    columns: np.ndarray
    values: np.ndarray
    free_rows: np.ndarray  # the free DOFs, row i of T is free_rows[i]
    cotree: np.ndarray  # the cotree DOFs
    h1_columns: np.ndarray
    floating: int
    conductors: int
    regions: int


@dataclasses.dataclass(frozen=True)
class _Graph:
    """The vertex graph read off a discrete gradient, conductors contracted.

    Nodes ``0 .. nv - 1`` are the vertex columns (free ones only carry
    edges), ``nv + c`` conductor ``c``. ``edges`` holds, per node pair, the
    row of the gradient joining them plus one (symmetric CSR).
    """

    edges: sp.csr_matrix
    nv: int
    vcols: np.ndarray  # the gradient's column of each vertex
    vfree: np.ndarray  # free vertices
    pec: np.ndarray  # fixed vertices with a gradient entry
    conductor: np.ndarray  # the conductor of each of ``pec``
    sizes: np.ndarray  # the vertices of each conductor
    column_free: np.ndarray  # gradient columns that touch free DOFs only


def _graph(g: sp.csc_matrix, vertex: np.ndarray, free: np.ndarray) -> _Graph:
    """The vertex graph of the discrete gradient ``g``, conductors contracted.

    A row of ``g`` with two vertex entries is an edge. Conductors are the
    connected components of the fixed vertices through fixed edges.
    """
    counts = np.diff(g.indptr)
    unit = sp.csc_matrix((np.ones(g.nnz), g.indices, g.indptr), shape=g.shape)
    column_free = (counts > 0) & ~((unit.T @ (~free).astype(float)) > 0)
    vcols = np.flatnonzero(vertex)
    nv = len(vcols)
    gv = sp.csr_matrix(g[:, vcols])
    vfree = column_free[vcols]
    edge = np.flatnonzero(np.diff(gv.indptr) == 2)
    ea = gv.indices[gv.indptr[edge]]
    eb = gv.indices[gv.indptr[edge] + 1]
    fixed = ~free[edge] & ~vfree[ea] & ~vfree[eb]
    pec = np.flatnonzero(~vfree & (counts[vcols] > 0))
    joined = sp.coo_matrix(
        (np.ones(int(fixed.sum())), (ea[fixed], eb[fixed])), shape=(nv, nv)
    )
    _, label = csgraph.connected_components(joined, directed=False)
    _, conductor = np.unique(label[pec], return_inverse=True)
    n_cond = int(conductor.max()) + 1 if len(pec) else 0
    node = np.where(vfree, np.arange(nv), -1)
    node[pec] = nv + conductor
    nn = nv + n_cond
    # free edges between two distinct nodes, one row per node pair
    ok = free[edge] & (node[ea] >= 0) & (node[eb] >= 0) & (node[ea] != node[eb])
    na, nb, rows = node[ea[ok]], node[eb[ok]], edge[ok]
    lo, hi = np.minimum(na, nb), np.maximum(na, nb)
    _, first = np.unique(lo.astype(np.int64) * nn + hi, return_index=True)
    edges = sp.coo_matrix((rows[first] + 1, (lo[first], hi[first])), shape=(nn, nn))
    return _Graph(
        edges=sp.csr_matrix(edges + edges.T),
        nv=nv,
        vcols=vcols,
        vfree=vfree,
        pec=pec,
        conductor=conductor,
        sizes=np.bincount(conductor, minlength=n_cond),
        column_free=column_free,
    )


def _spanning_forest(
    graph: _Graph, reference: Reference
) -> tuple[np.ndarray, np.ndarray, int]:
    """The breadth-first spanning forest of ``graph``, one tree per region.

    Each connected region is rooted at its reference conductor (most
    vertices for ``"largest"``, fewest for ``"smallest"``, ties to the lowest
    index) or, without one, at its first free vertex; the root's potential
    is dropped.

    Returns:
        The non-root nodes, the gradient row of the tree edge to each one's
        parent (its pivot), and the number of regions.
    """
    edges, nv = graph.edges, graph.nv
    nn = edges.shape[0]
    _, region = csgraph.connected_components(edges, directed=False)
    nodes = np.concatenate([np.flatnonzero(graph.vfree), np.arange(nv, nn)])
    is_vertex = nodes < nv
    weight = np.zeros(len(nodes), np.int64)
    sizes = graph.sizes[nodes[~is_vertex] - nv]
    weight[~is_vertex] = -sizes if reference == "largest" else sizes
    ranked = np.lexsort((nodes, weight, is_vertex, region[nodes]))
    _, first = np.unique(region[nodes][ranked], return_index=True)
    roots = nodes[ranked][first]
    # one search from a virtual node joined to every root
    virtual = sp.coo_matrix(
        (np.ones(len(roots), np.int64), (np.full(len(roots), nn), roots)),
        shape=(nn + 1, nn + 1),
    )
    padded = edges.copy()
    padded.resize((nn + 1, nn + 1))
    reached, parent = csgraph.breadth_first_order(
        sp.csr_matrix(padded + virtual + virtual.T),
        nn,
        directed=False,
        return_predecessors=True,
    )
    child = reached[1:][parent[reached[1:]] != nn]
    pivot = np.asarray(edges[child, parent[child]]).ravel().astype(np.int64) - 1
    return child, pivot, len(roots)


def _forest(
    g: sp.csc_matrix, vertex: np.ndarray, free: np.ndarray, reference: Reference
) -> _Forest:
    """T's columns from the discrete gradient ``g`` (DOFs by H1 DOFs).

    ``vertex`` marks the vertex columns of ``g``, ``free`` the free DOFs.
    """
    n, m = g.shape
    graph = _graph(g, vertex, free)
    child, pivot, regions = _spanning_forest(graph, reference)
    is_v = child < graph.nv
    floating = child[~is_v] - graph.nv
    # columns: free vertex potentials, high-order gradients, floating potentials
    hcols = np.flatnonzero(~vertex & graph.column_free)
    if np.any(np.diff(g.indptr)[hcols] != 1):
        raise RuntimeError("A high-order gradient column has more than one entry.")
    h1_columns = np.concatenate([graph.vcols[child[is_v]], hcols])
    pivots = np.concatenate([pivot[is_v], g.indices[g.indptr[hcols]], pivot[~is_v]])
    taken = np.bincount(pivots, minlength=n)
    if np.any(taken > 1) or not free[pivots].all():
        raise RuntimeError("The gauge pivots clash or hit a fixed DOF.")
    free_rows = np.flatnonzero(free)
    position = np.full(n, -1, np.int64)
    position[free_rows] = np.arange(len(free_rows))
    cotree = free_rows[taken[free_rows] == 0]
    gf = sp.coo_matrix(g[:, h1_columns])
    # floating potentials: the gradient of each conductor's vertex indicator
    column_of = np.full(len(graph.sizes), -1, np.int64)
    column_of[floating] = np.arange(len(floating))
    on = column_of[graph.conductor] >= 0
    indicator = sp.csc_matrix(
        (
            np.ones(int(on.sum())),
            (graph.vcols[graph.pec[on]], column_of[graph.conductor[on]]),
        ),
        shape=(m, len(floating)),
    )
    x = sp.coo_matrix(g @ indicator)
    x.eliminate_zeros()  # the conductor's own edges cancel exactly
    if np.any(position[gf.row] < 0) or np.any(position[x.row] < 0):
        raise RuntimeError("A gradient column of the gauge reaches a fixed DOF.")
    n_cotree = len(cotree)
    return _Forest(
        rows=np.concatenate([position[cotree], position[gf.row], position[x.row]]),
        columns=np.concatenate(
            [
                np.arange(n_cotree),
                n_cotree + gf.col,
                n_cotree + len(h1_columns) + x.col,
            ]
        ),
        values=np.concatenate([np.ones(n_cotree), gf.data, x.data]),
        free_rows=free_rows,
        cotree=cotree,
        h1_columns=h1_columns,
        floating=len(floating),
        conductors=len(graph.sizes),
        regions=regions,
    )


def _template(
    base: Any,
    *,
    h1: Any,
    floating: int,
    t: sp.csr_matrix,
    y_dofs: np.ndarray,
    v_base: np.ndarray,
) -> tuple[sp.csr_matrix, int, Any]:
    r"""The symbolic gauged pattern, :math:`P`'s entries and the gauge template.

    A complex form on :math:`Y` has the full element blocks for its graph
    (each number DOF's row dense), a superset of the gauged pattern, whose
    :math:`V` block is the element graph :math:`P`. The pattern's entries
    are set to 1, every other to 0, and ``DeleteZeroElements`` keeps the
    pattern. ``v_base`` holds the base-space DOF of each row of ``t``.
    Returns the pattern, the entries of :math:`P` on those rows, and the
    template.
    """
    mesh = base.mesh
    spaces = [base, h1, *(ngs.NumberSpace(mesh, complex=True) for _ in range(floating))]
    gauge_space = ngs.FESpace(spaces)
    trial, test = gauge_space.TnT()
    form = ngs.BilinearForm(gauge_space, symmetric=False)
    form += trial[0] * test[0] * ngs.dx
    form.Assemble()
    superset = form.mat
    _, cols, rowptr = superset.CSR()
    cols = np.asarray(cols)
    rowptr = np.asarray(rowptr)
    n_y = gauge_space.ndof
    graph = sp.csr_matrix(
        (np.ones(len(cols), np.float32), cols, rowptr), shape=(n_y, n_y)
    )
    element = graph[v_base][:, v_base]
    ungauged_nnz = int(element.nnz)
    del graph
    a = abs(t).astype(np.float32)
    pattern = sp.csr_matrix(a.T @ (element @ a))
    pattern.sum_duplicates()
    pattern.sort_indices()
    pattern.data = np.ones(pattern.nnz, np.int8)
    _mask(superset, rowptr, cols, pattern, y_dofs)
    template = superset.DeleteZeroElements(0.5)
    if template.nze != pattern.nnz:  # pragma: no cover - NGSolve keeps the mask
        raise RuntimeError("The gauge template lost the gauged pattern.")
    return pattern, ungauged_nnz, template


def _mask(
    matrix: Any,
    rowptr: np.ndarray,
    cols: np.ndarray,
    pattern: sp.csr_matrix,
    y_dofs: np.ndarray,
) -> None:
    """Set ``matrix``'s values to 1 on ``pattern`` (rows/cols ``y_dofs``), 0 elsewhere.

    Searched in chunks of rows so that the int64 keys stay small.

    Raises:
        RuntimeError: an entry of ``pattern`` is not in ``matrix``'s graph.
    """
    values = matrix.AsVector().FV().NumPy()
    values[:] = 0
    rowptr = rowptr.astype(np.int64)
    width = int(matrix.width) + 1
    lengths = np.diff(pattern.indptr)
    per_row = max(1, len(cols) // max(1, len(y_dofs)))
    step = max(1, _MASK_CHUNK // per_row)
    for lo in range(0, len(y_dofs), step):
        hi = min(lo + step, len(y_dofs))
        rows = y_dofs[lo:hi]
        starts, ends = rowptr[rows], rowptr[rows + 1]
        span = ends - starts
        offset = np.cumsum(span) - span
        at = np.repeat(starts - offset, span) + np.arange(span.sum())
        keys = np.repeat(np.arange(hi - lo, dtype=np.int64), span) * width + cols[at]
        a, b = pattern.indptr[lo], pattern.indptr[hi]
        wanted = (
            np.repeat(np.arange(hi - lo, dtype=np.int64), lengths[lo:hi]) * width
            + y_dofs[pattern.indices[a:b]]
        )
        found = np.searchsorted(keys, wanted)
        if len(wanted) and (
            found.max() >= len(keys) or not np.array_equal(keys[found], wanted)
        ):
            raise RuntimeError("The gauged pattern leaves the gauge space's graph.")
        values[at[found]] = 1


def _active_dofs(space: Any, base: Any) -> np.ndarray:
    """The base-space DOF of each DOF of the compressed ``space`` (increasing)."""
    if space.ndof == base.ndof:
        return np.arange(space.ndof)
    tags = ngs.BaseVector(space.ndof, False)
    tags.FV().NumPy()[:] = np.arange(1, space.ndof + 1)
    out = ngs.BaseVector(base.ndof, False)
    out.data = space.Embedding() * tags
    tag = np.rint(out.FV().NumPy()).astype(np.int64)
    active = np.empty(space.ndof, np.int64)
    active[tag[tag > 0] - 1] = np.flatnonzero(tag > 0)
    return active


def _to_scipy(matrix: Any) -> sp.csr_matrix:
    """``matrix`` (an NGSolve or SciPy sparse matrix) as SciPy CSR."""
    if sp.issparse(matrix):
        return sp.csr_matrix(matrix)
    data, cols, rowptr = matrix.CSR()
    return sp.csr_matrix(
        (np.array(data), np.array(cols), np.array(rowptr)),
        shape=(matrix.height, matrix.width),
    )


@dataclasses.dataclass(frozen=True)
class GaugeChoice:
    """Whether a driven solve runs through the tree-cotree gauge (`resolve_gauge`).

    Resolved once by the driven rule, which hands it to the low-frequency
    check, the system operator, its frequency workers and mesh refinement's
    deferred check. Plain data, so it pickles to a worker.

    Attributes:
        on: Whether the solve is gauged.
        reason: Why it is not, in a few words (``None`` when gauged), for
            the ``gfem.solver`` log.
        clause: The sentence `gfem.LowFrequencyWarning` adds on why the
            solve is ungauged and, where one exists, what would gauge it;
            ``None`` when it adds none.
    """

    on: bool
    reason: str | None = None
    clause: str | None = None


#: What a ``"tree_cotree"`` refusal suggests besides the fix.
_OR_AUTO = "or gauge='auto' to solve ungauged"


def resolve_gauge(
    gauge: str,
    *,
    linear_solver: str,
    kind: str,
    device: str,
    auto_reason: str | None = None,
    distributed: bool = False,
    periodic: bool = False,
    bloch: bool = False,
    floquet: bool = False,
) -> GaugeChoice:
    """Resolve ``SolverOptions.gauge`` for one driven solve (ADR 0220).

    The gauge applies to a solve whose linear solver resolved to
    ``"direct"`` on the CPU in one process, for a model with no periodic
    pair, Bloch wave vector or Floquet port. ``"auto"`` gauges such a solve
    and runs every other one ungauged; ``"tree_cotree"`` raises for every
    other one; ``"none"`` runs ungauged.

    Args:
        gauge: ``SolverOptions.gauge``.
        linear_solver: ``SolverOptions.linear_solver``, as written.
        kind: The linear solver it resolved to (never ``"auto"``).
        device: ``SolverOptions.device``.
        auto_reason: Why ``linear_solver="auto"`` chose ``kind``, with its
            numbers (`gfem._linalg.auto_linear_solver`), or ``None``.
        distributed: Whether the solve runs across MPI ranks.
        periodic: Whether the model has a `gfem.Periodic` pair.
        bloch: Whether the problem has a Bloch wave vector.
        floquet: Whether the model has a `gfem.FloquetPort`.

    Raises:
        gfem.DistributedUnsupportedError: ``"tree_cotree"`` across ranks.
        ValueError: ``"tree_cotree"`` for any other solve out of scope.
    """
    out = _out_of_scope(
        linear_solver,
        kind,
        device,
        distributed=distributed,
        periodic=periodic,
        bloch=bloch,
        floquet=floquet,
    )
    if gauge == "tree_cotree" and out is not None:
        _refuse(out[0], linear_solver, kind, auto_reason)
    if gauge == "none":
        if out is None:
            return GaugeChoice(
                on=False,
                reason="gauge='none'",
                clause="The solve is ungauged (gauge='none'): gauge='auto' keeps "
                "these digits.",
            )
        return GaugeChoice(on=False, reason="gauge='none'")
    if out is None:
        return GaugeChoice(on=True)
    reason, remedy = out
    if remedy is None:
        clause = f"This solve cannot be gauged ({reason})."
    else:
        clause = f"The solve is ungauged ({reason}): {remedy} keeps these digits."
    return GaugeChoice(on=False, reason=reason, clause=clause)


def _out_of_scope(
    linear_solver: str,
    kind: str,
    device: str,
    *,
    distributed: bool,
    periodic: bool,
    bloch: bool,
    floquet: bool,
) -> tuple[str, str | None] | None:
    """Why the gauge does not apply and what would apply it, or ``None``.

    The first reason that holds, the most specific first among the model's:
    Floquet ports and wave vectors come with a periodic pair.
    """
    solver: tuple[str, str | None] | None = None
    if kind == "bddc_gmres":
        chose = (
            "'auto' chose 'bddc_gmres'" if linear_solver == "auto" else "'bddc_gmres'"
        )
        solver = f"linear_solver={chose}", "linear_solver='direct'"
    elif kind != "direct":
        solver = f"linear_solver={kind!r}", None
    reasons: list[tuple[bool, tuple[str, str | None] | None]] = [
        (distributed, ("MPI ranks", None)),
        (device != "cpu", (f"device={device!r}", None)),
        (solver is not None, solver),
        (floquet, ("Floquet ports", None)),
        (bloch, ("a Bloch wave vector", None)),
        (periodic, ("a periodic pair", None)),
    ]
    return next((why for holds, why in reasons if holds), None)


def _refuse(
    reason: str, linear_solver: str, kind: str, auto_reason: str | None
) -> None:
    """Raise for ``gauge="tree_cotree"`` on a solve out of the gauge's scope."""
    if reason == "MPI ranks":
        raise DistributedUnsupportedError(
            "tree-cotree gauge",
            "gauge='tree_cotree' runs on one process, and this solve runs across "
            "MPI ranks. Solve on one process, "
            f"{_OR_AUTO} across ranks.",
        )
    if reason.startswith("device="):
        raise ValueError(
            f"gauge='tree_cotree' runs on the CPU, not {reason} (cuDSS solves "
            f"ungauged). Use device='cpu', {_OR_AUTO} on the GPU."
        )
    if linear_solver == "auto" and kind == "bddc_gmres":
        numbers = "" if auto_reason is None else f" ({auto_reason})"
        raise ValueError(
            "gauge='tree_cotree' needs the direct solver, but linear_solver='auto' "
            f"chose 'bddc_gmres'{numbers}. Set linear_solver='direct' to lift that "
            f"memory guard, {_OR_AUTO}."
        )
    if kind != "direct":
        raise ValueError(
            f"gauge='tree_cotree' needs linear_solver='direct', not {kind!r}. Use "
            f"linear_solver='direct', {_OR_AUTO}."
        )
    raise ValueError(
        f"gauge='tree_cotree' does not take {reason} in this version of gfem. "
        "Use gauge='auto' to solve ungauged."
    )
