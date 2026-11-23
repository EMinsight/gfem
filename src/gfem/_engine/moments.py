r"""The moment interpolant of an incident field from its values at the moment points.

An incident field reaches the H(curl) space through its canonical first-kind
Nédélec moments: on each edge :math:`e`, :math:`\int_e (E\cdot t)\,q` for
:math:`q \in P_{p-1}(e)`; on each face :math:`f`, the tangential moments
against :math:`P_{p-2}(f)^2`; in each cell :math:`T`, the moments against
:math:`P_{p-3}(T)^3`. Edge coefficients come from the edge moments alone, face
coefficients from the face moments less what the edge part already carries,
cell coefficients last: a block-triangular local solve. The interpolant keeps
tangential traces (a face where :math:`n\times E = 0` gets a zero tangential
trace) and commutes with the curl.

The moments are integrated with Gauss rules of order :math:`2p` on each
entity, the **moment points**: the values there are all the field enters by,
so a field given as NumPy values (a user function, or a closed form evaluated
on the host) needs no NGSolve coefficient function. NGSolve's dual ``Set``
reads a coefficient function at the same points and takes the same moments,
so both give the same coefficients at orders 1 to 3.

The map from values to coefficients is linear and depends only on the mesh,
the order and the element set, so it is built once per discretization as one
sparse matrix :math:`P`, :math:`w = P\,E(\text{points})`
(`MomentInterpolant`). Reference shapes are evaluated once per orientation
class (the relative order of an element's vertex numbers) and mapped
covariantly, batched over the class's elements.

Each edge and face takes its points from its own vertices, in the order of
their gfem node numbers, so every element (and every rank) holding a shared
entity reads the same points and computes the same coefficients: the operator
of a distributed mesh is rank-local and needs no exchange. An element set
smaller than the domain is the one exception: an entity on the set's edge
that is also a partition face is computed by the rank holding the chosen
element only, so its holders agree through one neighbour exchange.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._engine.cache import discretization
from gfem._engine.domain import hcurl_space, pec_surfaces, space_key
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem.mesh import Mesh
from gfem.model import Model

__all__ = [
    "MomentInterpolant",
    "field_interpolant",
    "incident_values",
    "moment_interpolant",
]

#: NGSolve's reference tetrahedron: vertex ``i`` at row ``i``.
_REFERENCE = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]])
#: The local vertices of an NGSolve tetrahedron's edges, in local edge order.
_EDGES = ((0, 3), (1, 3), (2, 3), (0, 1), (0, 2), (1, 2))
#: The local vertices of its faces, in local face order (face ``k`` is
#: opposite vertex ``k``).
_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


@dataclasses.dataclass(frozen=True, eq=False)
class MomentInterpolant:
    """The moment interpolant of one discretization, order and element set.

    Attributes:
        points: The moment points, ``(n, 3)`` in metres: on every edge, face
            and cell of the element set that carries degrees of freedom.
        matrix: :math:`P`, ``(ndof, 3 n)``: the coefficients of the unpaired
            H(curl) space of the order from the flat values
            :math:`(E_x, E_y, E_z)` point by point, in V/m. The coefficients
            are the engine's (the field's SI value times the mesh scale).
        setup_time: Seconds spent building it.
    """

    points: np.ndarray
    matrix: sp.csr_array
    setup_time: float
    shared: _Shared | None = None

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """The coefficients of the field of ``values`` (``(n, 3)``, V/m) at `points`."""
        out = self.matrix @ np.asarray(values).reshape(-1)
        return out if self.shared is None else self.shared(out)


@dataclasses.dataclass(frozen=True, eq=False)
class _Shared:
    """Agreement of a masked interpolant's shared DOFs across ranks.

    An element set's edge of a shared entity may lie on one rank only: a
    face between a chosen element on one rank and an element left out on
    another. The second rank holds the DOF but computes nothing there, so
    each shared DOF takes the value of the ranks that computed it (they
    agree to round-off): one neighbour exchange per call.

    Attributes:
        layout: The space's layout across ranks.
        counts: Per local DOF, how many of its holders computed it.
    """

    layout: Any
    counts: np.ndarray

    def __call__(self, coefficients: np.ndarray) -> np.ndarray:
        summed = self.layout.exchange(coefficients)
        return np.where(self.counts > 0, summed / np.maximum(self.counts, 1), 0)

    @classmethod
    def of(cls, space: ngs.FESpace, comm: Any, matrix: sp.csr_array) -> _Shared:
        from gfem._linalg.layout import RankLayout

        layout = RankLayout(space, comm)
        computed = (np.diff(matrix.indptr) > 0).astype(float)
        counts = np.real(layout.exchange(computed))
        return cls(layout, counts)


def moment_interpolant(
    model: Model,
    order: int,
    *,
    elements: np.ndarray | None = None,
    comm: Any = None,
) -> MomentInterpolant:
    """The `MomentInterpolant` of ``model``'s mesh at ``order``.

    Its coefficients are those of the unpaired H(curl) space a driven solve of
    ``order`` builds (same key, so the same numbering). Cached with the
    discretization by that space and the element set.

    Args:
        model: The model; its electromagnetic domain bounds the element set.
        order: The Nédélec order :math:`p`, 1 or more.
        elements: One boolean per gfem tetrahedron: the elements whose
            entities get moment points (``None``: every element of the
            domain). The DOFs of the set's edges, faces and cells get
            coefficients; the other rows of `MomentInterpolant.matrix` are
            empty.
        comm: The resolved communicator of a distributed solve: the operator
            covers this rank's elements.
    """
    disc = discretization(model.mesh, model._periodic, comm)  # noqa: SLF001
    built = disc.built
    key = space_key(model, "hcurl", order)
    space = disc.space(
        key,
        lambda: hcurl_space(model, built, order, dirichlet=pec_surfaces(model)),
    )
    if elements is None:
        mask_key = None
    else:
        mask = np.asarray(elements, bool)
        mask_key = (len(mask), hashlib.sha256(np.packbits(mask).tobytes()).hexdigest())
    domains = _active_domains(model, built)
    return disc.space(
        (*key, "moment interpolant", mask_key),
        lambda: _shared(
            _build(space, built, order, domains, elements), space, comm, elements
        ),
    )


def _shared(
    interpolant: MomentInterpolant,
    space: ngs.FESpace,
    comm: Any,
    elements: np.ndarray | None,
) -> MomentInterpolant:
    """``interpolant`` agreeing on shared DOFs across ranks (`_Shared`).

    Under ranks only: an element set's shared DOFs may be computed on some of
    their holders; one process, or every element of the domain (each holder
    computes its entities), needs no exchange and returns ``interpolant``.
    """
    if comm is None or elements is None:
        return interpolant
    return dataclasses.replace(
        interpolant, shared=_Shared.of(space, comm, interpolant.matrix)
    )


def field_interpolant(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    order: int,
    *,
    elements: np.ndarray | None = None,
    comm: Any = None,
) -> MomentInterpolant:
    """The `MomentInterpolant` of a field's H(curl) space, from its layout alone.

    The space of a `gfem.Field` on ``support`` minus ``excluded``
    (`gfem._engine.fields.fe_space`): the numbering of a driven solve's
    space on the same domain, so a kept field and its incident part share
    their rows. ``elements`` and ``comm`` as for `moment_interpolant`: under
    ranks, this rank's rows of the field's piece.
    """
    built, space, _ = fe_space(mesh, support, excluded, "hcurl", order, comm=comm)
    names = mesh._region_names()  # noqa: SLF001
    wanted, out = set(support), set(excluded)
    domains = np.array(
        [names[tag] in wanted and (tag, e) not in out for tag, e in built.domains],
        bool,
    )
    if elements is None:
        mask_key = None
    else:
        mask = np.asarray(elements, bool)
        mask_key = (len(mask), hashlib.sha256(np.packbits(mask).tobytes()).hexdigest())
    disc = discretization(mesh, (), comm)
    key = (("field", tuple(support), tuple(excluded)), "hcurl", order)
    return disc.space(
        (*key, "moment interpolant", mask_key),
        lambda: _shared(
            _build(space, built, order, domains, elements), space, comm, elements
        ),
    )


def incident_values(
    function: Callable[[np.ndarray, float], Any],
    points: np.ndarray,
    frequency: float,
    name: str,
) -> np.ndarray:
    """The values of an incident-field function at ``points``, checked.

    ``function(points, frequency)`` must return an ``(n, 3)`` array of finite
    numbers (the field in V/m at the ``(n, 3)`` points in metres).

    Args:
        function: The incident-field function.
        points: The points, ``(n, 3)`` in metres.
        frequency: Hz.
        name: The source, as error messages name it.

    Returns:
        The values, complex, ``(n, 3)``.

    Raises:
        ValueError: The function returned the wrong shape, a non-numeric
            array or a non-finite value.
    """
    raw = np.asarray(function(points, frequency))
    expected = (len(points), 3)
    if raw.shape != expected:
        msg = (
            f"{name}: the incident-field function returned shape {raw.shape} "
            f"for {len(points)} points; expected {expected} (E in V/m per point)."
        )
        raise ValueError(msg)
    if raw.dtype.kind not in "biufc":
        msg = (
            f"{name}: the incident-field function returned dtype {raw.dtype}; "
            "expected real or complex numbers (E in V/m)."
        )
        raise ValueError(msg)
    values = raw.astype(complex)
    bad = ~np.isfinite(values).all(axis=1)
    if bad.any():
        where = points[np.flatnonzero(bad)[0]]
        msg = (
            f"{name}: the incident-field function returned a non-finite value "
            f"at {int(bad.sum())} point(s), the first at {tuple(where)} m."
        )
        raise ValueError(msg)
    return values


def _active_domains(model: Model, built: EngineMesh) -> np.ndarray:
    """One boolean per NGSolve domain: inside the electromagnetic domain."""
    out = set(model._domain_of("em").excluded)  # noqa: SLF001
    return np.array([key not in out for key in built.domains], bool)


def _rule(kind: Any, order: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """NGSolve's Gauss rule of ``order`` on the reference ``kind``: points, weights."""
    rule = ngs.IntegrationRule(kind, order)
    points = np.array([tuple(p) for p in rule.points], float)[:, :dim]
    return points, np.array(rule.weights, float)


def _monomials(coords: np.ndarray, degree: int) -> np.ndarray:
    """The monomials of total degree at most ``degree`` at ``coords``: ``(nq, m)``."""
    dim = coords.shape[1]
    if degree < 0:
        return np.zeros((len(coords), 0))
    powers = [
        a for a in itertools.product(range(degree + 1), repeat=dim) if sum(a) <= degree
    ]
    return np.stack([np.prod(coords**a, axis=1) for a in powers], axis=1)


def _class_codes(order: np.ndarray) -> np.ndarray:
    """One integer per row of a permutation of 4: its code."""
    return order @ np.array([64, 16, 4, 1])


#: Elements of one orientation class mapped at once: bounds the batch's
#: shape arrays (elements x points x local DOFs x 3).
_BATCH = 2048


@dataclasses.dataclass(frozen=True)
class _Layout:
    """Where an element's DOFs of each entity sit in its local DOF order.

    NGSolve's first-kind Nédélec element lists the six lowest-order edge DOFs,
    then the higher edge DOFs edge by edge, the face DOFs face by face, and the
    cell DOFs last.
    """

    order: int

    @property
    def n_face(self) -> int:
        """DOFs per face: the moments against :math:`P_{p-2}^2`."""
        p = self.order
        return (p - 1) * p

    @property
    def n_cell(self) -> int:
        """DOFs per cell: the moments against :math:`P_{p-3}^3`."""
        p = self.order
        return (p - 2) * (p - 1) * p // 2

    @property
    def n_local(self) -> int:
        return 6 * self.order + 4 * self.n_face + self.n_cell

    def edge(self, k: int) -> list[int]:
        p = self.order
        return [k, *range(6 + k * (p - 1), 6 + (k + 1) * (p - 1))]

    def face(self, k: int) -> list[int]:
        start = 6 * self.order + k * self.n_face
        return list(range(start, start + self.n_face))

    def face_lower(self, k: int) -> list[int]:
        """The DOFs of face ``k``'s own edges.

        The other edges' shapes have no tangential trace on the face.
        """
        return [
            d for j, ends in enumerate(_EDGES) if k not in ends for d in self.edge(j)
        ]

    def cell(self) -> list[int]:
        return list(range(6 * self.order + 4 * self.n_face, self.n_local))

    def cell_lower(self) -> list[int]:
        return list(range(6 * self.order + 4 * self.n_face))


@dataclasses.dataclass(frozen=True)
class _Elements:
    """The local tetrahedra of the element set.

    Attributes:
        numbers: Their NGSolve element numbers.
        vertices: Their NGSolve vertex numbers, ``(n, 4)``.
        nodes: Their gfem node numbers, ``(n, 4)``: the same on every rank.
        dofs: Their DOFs in local order, ``(n, n_local)``.
        coordinates: The NGSolve mesh's vertex coordinates (file units).
    """

    numbers: np.ndarray
    vertices: np.ndarray
    nodes: np.ndarray
    dofs: np.ndarray
    coordinates: np.ndarray


def _elements(
    space: ngs.FESpace,
    built: EngineMesh,
    layout: _Layout,
    domains: np.ndarray,
    elements: np.ndarray | None,
) -> _Elements:
    """The `_Elements` of ``elements`` (gfem mask) within the active ``domains``."""
    ngmesh = built.mesh.ngmesh
    table = ngmesh.Elements3D().NumPy()
    vertices = np.asarray(table["nodes"], np.int64)[:, :4] - 1
    chosen = domains[np.asarray(table["index"], np.int64) - 1]
    if elements is not None:
        chosen &= np.asarray(elements, bool)[built.cell_order]
    numbers = np.flatnonzero(chosen)
    dofs = np.empty((len(numbers), layout.n_local), np.int64)
    for row, number in enumerate(numbers):
        dofs[row] = space.GetDofNrs(ngs.ElementId(ngs.VOL, int(number)))
    local = vertices[numbers]
    return _Elements(
        numbers,
        local,
        np.asarray(built.node_order)[local],
        dofs,
        np.array(ngmesh.Coordinates(), float),
    )


def _first_holders(ids: np.ndarray) -> np.ndarray:
    """Per (element, local entity): whether the element is the entity's first."""
    _, first = np.unique(ids.ravel(), return_index=True)
    mine = np.zeros(ids.size, bool)
    mine[first] = True
    return mine.reshape(ids.shape)


class _Entity(NamedTuple):
    """One local entity of a batch of elements of one orientation class.

    Attributes:
        stage: ``"edge"``, ``"face"`` or ``"cell"``.
        held: Per element of the batch: whether it computes the entity.
        own: The entity's local DOFs.
        lower: The local DOFs of lower stages its coefficients carry.
        reference: Its moment points on the reference element, ``(nq, 3)``.
        physical: Its moment points, ``(n, nq, 3)`` (file units).
        weights: The rule's weights, ``(nq,)``.
        tests: The moment test functions at the points, ``(n, nq, m, 3)``.
    """

    stage: str
    held: np.ndarray
    own: list[int]
    lower: list[int]
    reference: np.ndarray
    physical: np.ndarray
    weights: np.ndarray
    tests: np.ndarray


def _entities(
    x: np.ndarray,
    rank: np.ndarray,
    layout: _Layout,
    held: dict[str, np.ndarray],
) -> list[_Entity]:
    """The entities of a batch of elements with vertices ``x`` (``(n, 4, 3)``).

    ``rank`` is the position of each local vertex in the gfem node order (the
    batch's class); ``held`` the per-element, per-entity first-holder flags.
    The points and moment bases of an edge or face follow its vertices in gfem
    node order, so every holder reads the same ones.
    """
    p, n = layout.order, len(x)
    out = []
    s, w = _rule(ngs.ET.SEGM, 2 * p, 1)
    for k, ends in enumerate(_EDGES):
        a, b = sorted(ends, key=lambda v: rank[v])
        tangent = x[:, b] - x[:, a]
        out.append(
            _Entity(
                "edge",
                held["edge"][:, k],
                layout.edge(k),
                [],
                np.outer(1 - s[:, 0], _REFERENCE[a]) + np.outer(s[:, 0], _REFERENCE[b]),
                x[:, a, None] + s[None, :, 0, None] * tangent[:, None],
                w,
                _monomials(s, p - 1)[None, :, :, None] * tangent[:, None, None, :],
            )
        )
    if layout.n_face:
        s, w = _rule(ngs.ET.TRIG, 2 * p, 2)
        lam = np.column_stack([s[:, 0], s[:, 1], 1 - s[:, 0] - s[:, 1]])
        mono = _monomials(s, p - 2)[None, :, :, None, None]
        for k, corners in enumerate(_FACES):
            ordered = sorted(corners, key=lambda v: rank[v])
            tangents = np.stack(
                [
                    x[:, ordered[1]] - x[:, ordered[0]],
                    x[:, ordered[2]] - x[:, ordered[0]],
                ],
                axis=1,
            )  # (n, 2, 3)
            out.append(
                _Entity(
                    "face",
                    held["face"][:, k],
                    layout.face(k),
                    layout.face_lower(k),
                    lam @ _REFERENCE[ordered],
                    np.einsum("qi,nic->nqc", lam, x[:, ordered]),
                    w,
                    (mono * tangents[:, None, None]).reshape(n, len(s), -1, 3),
                )
            )
    if layout.n_cell:
        s, w = _rule(ngs.ET.TET, 2 * p, 3)
        edges = np.stack([x[:, i] - x[:, 3] for i in range(3)], axis=2)
        mono = _monomials(s, p - 3)
        tests = (mono[:, :, None, None] * np.eye(3)).reshape(len(s), -1, 3)
        out.append(
            _Entity(
                "cell",
                np.ones(n, bool),
                layout.cell(),
                layout.cell_lower(),
                s,
                x[:, 3, None] + np.einsum("qi,nci->nqc", s, edges),
                w,
                np.broadcast_to(tests, (n, *tests.shape)),
            )
        )
    return out


@dataclasses.dataclass
class _Parts:
    """The triplets of one sparse matrix as it is built."""

    rows: list[np.ndarray] = dataclasses.field(default_factory=list)
    cols: list[np.ndarray] = dataclasses.field(default_factory=list)
    vals: list[np.ndarray] = dataclasses.field(default_factory=list)

    def add(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> None:
        shape = vals.shape
        self.rows.append(np.broadcast_to(rows, shape).ravel())
        self.cols.append(np.broadcast_to(cols, shape).ravel())
        self.vals.append(vals.ravel())

    def matrix(self, shape: tuple[int, int]) -> sp.csr_array:
        if not self.rows:
            return sp.csr_array(shape, dtype=float)
        return sp.csr_array(
            (
                np.concatenate(self.vals),
                (np.concatenate(self.rows), np.concatenate(self.cols)),
            ),
            shape=shape,
        )


@dataclasses.dataclass
class _Assembly:
    r"""The stage matrices and moment points as entities are added.

    Per stage, ``direct`` maps the entity's values to its coefficients through
    its moments, and ``lower`` maps the lower stages' coefficients to what they
    already carry of those moments; the operator is
    :math:`P = P_E`, then :math:`P \leftarrow P + P_s - Q_s P` for the face
    and cell stages.
    """

    direct: dict[str, _Parts] = dataclasses.field(
        default_factory=lambda: {s: _Parts() for s in ("edge", "face", "cell")}
    )
    lower: dict[str, _Parts] = dataclasses.field(
        default_factory=lambda: {s: _Parts() for s in ("face", "cell")}
    )
    points: list[np.ndarray] = dataclasses.field(default_factory=list)
    n_points: int = 0

    def add(
        self, entity: _Entity, dofs: np.ndarray, inverse: np.ndarray, fe: Any
    ) -> None:
        """Add ``entity`` of the elements with ``dofs`` and inverse Jacobians."""
        pick = np.flatnonzero(entity.held)
        if not len(pick):
            return
        w, own, low = entity.weights, entity.own, entity.lower
        reference = np.array([np.array(fe.CalcShape(*r)) for r in entity.reference])
        # covariant map of the shapes the stage reads: row vectors times J^{-1};
        # then, flattened over (point, component), (n, nq * 3, local DOFs)
        shapes = np.matmul(reference[None, :, own + low], inverse[pick, None])
        n, nq, m = len(pick), len(w), len(own)
        shapes = shapes.transpose(0, 1, 3, 2).reshape(n, nq * 3, -1)
        weighted = (w[None, :, None, None] * entity.tests[pick]).transpose(0, 2, 1, 3)
        weighted = weighted.reshape(n, m, nq * 3)
        local = np.linalg.solve(weighted @ shapes[:, :, :m], weighted)
        index = self.n_points + np.arange(n * nq).reshape(n, nq)
        rows = dofs[pick][:, own]
        cols = 3 * index[:, None, :, None] + np.arange(3)
        self.direct[entity.stage].add(
            rows[:, :, None], cols.reshape(n, 1, nq * 3), local
        )
        if low:
            carried = local @ shapes[:, :, m:]
            self.lower[entity.stage].add(
                rows[:, :, None], dofs[pick][:, None, low], carried
            )
        self.points.append(entity.physical[pick].reshape(-1, 3))
        self.n_points += n * nq

    def finish(self, ndof: int, scale: float, start: float) -> MomentInterpolant:
        """The `MomentInterpolant`: engine coefficients from SI values."""
        shape = (ndof, 3 * self.n_points)
        out = self.direct["edge"].matrix(shape)
        for stage in ("face", "cell"):
            out = (
                out
                + self.direct[stage].matrix(shape)
                - self.lower[stage].matrix((ndof, ndof)) @ out
            )
        # the engine value of an H(curl) field is its SI value times the scale
        matrix = sp.csr_array(out * scale)
        points = np.concatenate(self.points) if self.points else np.zeros((0, 3))
        return MomentInterpolant(points * scale, matrix, time.perf_counter() - start)


def _build(
    space: ngs.FESpace,
    built: EngineMesh,
    order: int,
    domains: np.ndarray,
    elements: np.ndarray | None,
) -> MomentInterpolant:
    """Build the `MomentInterpolant` of ``space`` (first-kind Nédélec, ``order``)."""
    start = time.perf_counter()
    if built.mesh.ngmesh.ne == 0:
        # a rank of a distributed mesh holding no element (NGSolve's rank 0)
        return _Assembly().finish(space.ndof, built.scale, start)
    layout = _Layout(order)
    els = _elements(space, built, layout, domains, elements)
    held = {"edge": _first_holders(els.dofs[:, [layout.edge(k)[0] for k in range(6)]])}
    if layout.n_face:
        held["face"] = _first_holders(
            els.dofs[:, [layout.face(k)[0] for k in range(4)]]
        )
    # an orientation class: the relative order of the NGSolve vertex numbers
    # (which fixes the reference shapes) and of the gfem node numbers (which
    # fixes the points); the two agree on one process
    by_node = np.argsort(els.nodes, axis=1)
    classes = _class_codes(np.argsort(els.vertices, axis=1)) * 256 + _class_codes(
        by_node
    )
    assembly = _Assembly()
    for code in np.unique(classes):
        members = np.flatnonzero(classes == code)
        fe = space.GetFE(ngs.ElementId(ngs.VOL, int(els.numbers[members[0]])))
        rank = np.empty(4, int)
        rank[by_node[members[0]]] = np.arange(4)
        for begin in range(0, len(members), _BATCH):
            batch = members[begin : begin + _BATCH]
            x = els.coordinates[els.vertices[batch]]  # (n, 4, 3)
            jacobian = np.stack([x[:, i] - x[:, 3] for i in range(3)], axis=2)
            inverse = np.linalg.inv(jacobian)
            flags = {stage: h[batch] for stage, h in held.items()}
            for entity in _entities(x, rank, layout, flags):
                assembly.add(entity, els.dofs[batch], inverse, fe)
    return assembly.finish(space.ndof, built.scale, start)
