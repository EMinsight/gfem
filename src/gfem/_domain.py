"""The solve domain of a mesh for one physics: which elements are excluded.

Each physics (electromagnetic, thermal, DC conduction) has its own domain,
built by `em_domain` from that physics' excluded entities. Pure NumPy on
the `gfem.Mesh` arrays; the engine maps the result to NGSolve.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import Any

import numpy as np

from gfem.mesh import _TET_EDGES, _TET_FACES, Mesh


def physics_of(item: Any) -> str:
    """The physics key of a boundary, port or problem (``"em"`` by default).

    Each of their classes names its physics as ``_physics``: ``"em"``,
    ``"thermal"`` or ``"dc_conduction"``, the keys of ``excluded=``.
    """
    return getattr(type(item), "_physics", "em")


@dataclasses.dataclass(frozen=True, eq=False)
class EMDomain:
    """Excluded gmsh volume entities and how each surface meets the domain.

    Compares by identity: it is static pytree data of a `Model`, derived once
    from the mesh topology and the ``excluded=`` entries.

    Attributes:
        excluded: ``(physical tag, entity)`` pairs removed from the domain.
        active: Per tetrahedron, whether it is in the domain.
        sides: Per surface name, how many of its triangles have the domain
            on 0, 1 and 2 sides (``(n0, n1, n2)``).
    """

    excluded: tuple[tuple[int, int], ...]
    active: np.ndarray
    sides: dict[str, tuple[int, int, int]]

    def exterior(self, surface: str) -> bool:
        """Some triangle of ``surface`` has the domain on one side only."""
        return self.sides[surface][1] > 0

    def touches(self, surface: str) -> bool:
        """Some triangle of ``surface`` borders the domain."""
        return self.sides[surface][1] + self.sides[surface][2] > 0

    def interior(self, surface: str) -> bool:
        """Some triangle of ``surface`` has the domain on both sides."""
        return self.sides[surface][2] > 0


#: Edges, faces and tetrahedra of each domain, once counted
#: (`nedelec_unknowns`), for as long as the domain lives.
_ENTITIES: weakref.WeakKeyDictionary[EMDomain, tuple[int, int, int]] = (
    weakref.WeakKeyDictionary()
)


def nedelec_unknowns(mesh: Mesh, domain: EMDomain, order: int) -> int:
    """First-kind Nédélec unknowns of ``order`` on ``domain`` of ``mesh``.

    ``p E + p(p-1) F + p(p-1)(p-2)/2 T`` over the domain's edges, faces and
    tetrahedra: NGSolve's ``HCurl(order=p, type1=True)`` size. Dirichlet
    unknowns count and periodic pairs are not compressed, so it bounds the
    system size from above. Counted once per domain.
    """
    counts = _ENTITIES.get(domain)
    if counts is None:
        tets = mesh.tetrahedra[domain.active]
        edges = tets[:, np.array(_TET_EDGES)].reshape(-1, 2)
        faces = tets[:, _TET_FACES].reshape(-1, 3)
        counts = (_distinct_rows(edges), _distinct_rows(faces), len(tets))
        _ENTITIES[domain] = counts
    e, f, t = counts
    p = order
    return p * e + p * (p - 1) * f + p * (p - 1) * (p - 2) // 2 * t


def h1_unknowns(mesh: Mesh, domain: EMDomain, order: int) -> int:
    """H1 unknowns of ``order`` on ``domain`` of a 2D ``mesh``.

    ``V + (p-1) E + (p-1)(p-2)/2 T`` over the domain's vertices, edges and
    triangles: NGSolve's ``H1(order=p)`` size on triangles. Dirichlet
    unknowns count, so it bounds the system size from above.
    """
    cells = mesh.volume_elements[domain.active]
    edges = cells[:, np.array([[0, 1], [1, 2], [2, 0]])].reshape(-1, 2)
    v, e, t = len(np.unique(cells)), _distinct_rows(edges), len(cells)
    p = order
    return v + (p - 1) * e + (p - 1) * (p - 2) // 2 * t


def _distinct_rows(rows: np.ndarray) -> int:
    """How many distinct vertex sets the rows hold."""
    rows = np.ascontiguousarray(np.sort(rows, axis=1), dtype=np.int64)
    keys = rows.view(np.dtype((np.void, rows.itemsize * rows.shape[1])))
    return len(np.unique(keys))


def entity_keys(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Distinct ``(tag, entity)`` rows and the row of each volume element."""
    topology = mesh._topology  # noqa: SLF001 - the role arrays of either dimension
    keys = np.stack([topology.volume_tags, topology.volume_entities], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    return uniq, inverse.ravel()


def enclosed_entities(mesh: Mesh, surfaces: tuple[str, ...]) -> set[tuple[int, int]]:
    """Volume entities whose whole boundary lies on the named surfaces.

    An entity's boundary is every tetrahedron face whose other side is
    another entity or outside the mesh.
    """
    keys, dom = entity_keys(mesh)
    tags = [mesh.region(s).tag for s in surfaces]
    shell = np.sort(mesh.triangles[np.isin(mesh.triangle_tags, tags)], axis=1)
    faces = np.sort(mesh.tetrahedra[:, _TET_FACES].reshape(-1, 3), axis=1)
    face_dom = np.repeat(dom, 4)
    _, ids = np.unique(np.concatenate([faces, shell]), axis=0, return_inverse=True)
    ids = ids.ravel()
    face_ids, shell_ids = ids[: len(faces)], ids[len(faces) :]
    # A face seen twice from the same entity is inside it; once, on its boundary.
    pair = face_ids * len(keys) + face_dom
    _, pair_inv, pair_count = np.unique(pair, return_inverse=True, return_counts=True)
    on_boundary = pair_count[pair_inv.ravel()] == 1
    on_shell = np.isin(face_ids, shell_ids)
    bounded = set(np.unique(face_dom[on_boundary]).tolist())
    leaky = set(np.unique(face_dom[on_boundary & ~on_shell]).tolist())
    return {(int(keys[d][0]), int(keys[d][1])) for d in bounded - leaky}


#: The domain with nothing excluded, per mesh topology: every physics with no
#: exclusion shares it.
_WHOLE: weakref.WeakKeyDictionary[object, EMDomain] = weakref.WeakKeyDictionary()


def em_domain(mesh: Mesh, excluded: set[tuple[int, int]]) -> EMDomain:
    """The domain left once the ``excluded`` entities are removed.

    The name is historical: it builds the domain of any physics. The domain
    with nothing excluded is built once per mesh topology.
    """
    if not excluded:
        whole = _WHOLE.get(mesh._topology)  # noqa: SLF001
        if whole is None:
            whole = _WHOLE[mesh._topology] = _domain(mesh, excluded)  # noqa: SLF001
        return whole
    return _domain(mesh, excluded)


def _domain(mesh: Mesh, excluded: set[tuple[int, int]]) -> EMDomain:
    topology = mesh._topology  # noqa: SLF001 - the role arrays of either dimension
    keys = np.stack([topology.volume_tags, topology.volume_entities], axis=1)
    out = np.zeros(len(keys), dtype=bool)
    for tag, ent in excluded:
        out |= (keys[:, 0] == tag) & (keys[:, 1] == ent)
    active = ~out
    active.flags.writeable = False
    nb = topology.surface_neighbours
    n_active = ((nb >= 0) & active[np.maximum(nb, 0)]).sum(axis=1)
    sides = {}
    for r in mesh.regions:
        if r.dim == mesh.dim - 1:
            counts = np.bincount(n_active[topology.surface_tags == r.tag], minlength=3)
            sides[r.name] = (int(counts[0]), int(counts[1]), int(counts[2]))
    return EMDomain(excluded=tuple(sorted(excluded)), active=active, sides=sides)
