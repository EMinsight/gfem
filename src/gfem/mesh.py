"""Mesh import (gmsh ``.msh`` 2.2 ASCII and 4.1) and `Mesh`."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import weakref
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from gfem import _mpi
from gfem._pytree import readonly
from gfem.errors import RegionNotFoundError

#: Supported ``length_unit`` values and their size in metres.
LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}

#: What a failed read raises on one process, so on every rank above one.
_READ_ERRORS = (ValueError, OSError, meshio.ReadError)

_TOPOLOGY_STATICS = ("length_unit", "regions", "source", "section_thickness")
_TOPOLOGY_ARRAYS = (
    "imported_nodes",
    "tetrahedra",
    "tet_tags",
    "tet_entities",
    "triangles",
    "tri_tags",
    "tri_entities",
    "tri_neighbours",
)

# Local face k of a tetrahedron is the one opposite its vertex k.
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
# Local vertex pairs of a tetrahedron's six edges.
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: The end faces a cross-section's slab gains, at the section's plane and one
#: thickness along +z.
END_FACES = ("front", "back")


@dataclasses.dataclass(frozen=True)
class Region:
    """A named volume or surface of the mesh: one gmsh physical group.

    Attributes:
        name: The gmsh physical-group name, an opaque string.
        tag: The gmsh physical-group number.
        dim: 3 for a volume, 2 for a surface.
    """

    name: str
    tag: int
    dim: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Topology:
    """Everything about a mesh except its node coordinates.

    This is the static part of the `Mesh` pytree. It compares by identity, so
    a jitted function retraces only when it sees a different imported mesh.
    """

    length_unit: str
    imported_nodes: np.ndarray
    regions: tuple[Region, ...]
    tetrahedra: np.ndarray
    tet_tags: np.ndarray
    tet_entities: np.ndarray
    triangles: np.ndarray
    tri_tags: np.ndarray
    tri_entities: np.ndarray
    tri_neighbours: np.ndarray
    source: Path | None = None
    section_thickness: float | None = None


_TOPOLOGY_DIGESTS: weakref.WeakKeyDictionary[_Topology, str] = (
    weakref.WeakKeyDictionary()
)


def _topology_sha256(topology: _Topology) -> str:
    """The sha256 (64 hex) of a topology's arrays, computed once per topology.

    Each of `_TOPOLOGY_ARRAYS` enters as `_hash_array` feeds it, in that
    order. The topology is immutable, so the digest is memoized on it.
    """
    digest = _TOPOLOGY_DIGESTS.get(topology)
    if digest is None:
        h = hashlib.sha256()
        for name in _TOPOLOGY_ARRAYS:
            _hash_array(h, name, getattr(topology, name))
        digest = h.hexdigest()
        _TOPOLOGY_DIGESTS[topology] = digest
    return digest


def _hash_array(h, path: str, array) -> None:
    """Feed one array to the hash ``h``, independent of its memory layout.

    A length-prefixed JSON header ``[path, dtype.str, shape]`` comes first,
    then the little-endian C-order bytes. Float and complex values are
    normalized first: $-0$ becomes $+0$ (``a + 0``) and every NaN one quiet NaN.
    """
    a = np.asarray(array)
    if a.dtype.kind in "fc":
        a = a + a.dtype.type(0)
        if a.dtype.kind == "f":
            a = np.where(np.isnan(a), a.dtype.type(np.nan), a)
        else:
            nan = a.real.dtype.type(np.nan)
            re = np.where(np.isnan(a.real), nan, a.real)
            im = np.where(np.isnan(a.imag), nan, a.imag)
            a = np.empty(a.shape, a.dtype)
            a.real, a.imag = re, im
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))
    head = json.dumps([path, a.dtype.str, list(a.shape)]).encode()
    h.update(len(head).to_bytes(8, "little"))
    h.update(head)
    h.update(memoryview(a.reshape(-1)).cast("B"))


def _read_gmsh(path: Path, length_unit: str) -> _Topology:
    """Parse the file: what `Mesh.from_gmsh` does on one process."""
    _check_unit(length_unit)
    raw = meshio.read(path, file_format="gmsh")
    regions = _regions(raw)
    cells = _cells(raw, path)
    tets, tet_tags, tet_entities = cells["tetra"]
    tris, tri_tags, tri_entities = cells["triangle"]
    points = np.asarray(raw.points, dtype=float)
    neighbours = _triangle_neighbours(points, tets, tris, path)
    topology = _Topology(
        length_unit=length_unit,
        imported_nodes=readonly(points * LENGTH_UNITS[length_unit]),
        regions=regions,
        tetrahedra=readonly(tets),
        tet_tags=readonly(tet_tags),
        tet_entities=readonly(tet_entities),
        triangles=readonly(tris),
        tri_tags=readonly(tri_tags),
        tri_entities=readonly(tri_entities),
        tri_neighbours=readonly(neighbours),
        source=path.resolve(),
    )
    _check_tags(topology, path)
    return topology


def _check_unit(length_unit: str) -> None:
    if length_unit not in LENGTH_UNITS:
        raise ValueError(
            f"Unknown length_unit {length_unit!r}; use one of "
            f"{', '.join(repr(u) for u in LENGTH_UNITS)}."
        )


def _check_thickness(thickness) -> float:
    try:
        value = float(thickness)
    except (TypeError, ValueError):
        value = float("nan")
    if not (np.isfinite(value) and value > 0):
        raise ValueError(
            f"thickness must be a positive length in metres, got {thickness!r}; "
            "leave it None to let gfem pick one."
        )
    return value


def _check_section_plane(points: np.ndarray, path: Path) -> None:
    """The section must lie in a plane z = constant."""
    extent = np.ptp(points[:, :2], axis=0).max()
    if np.ptp(points[:, 2]) > 1e-9 * extent:
        raise ValueError(
            f"{path.name}: the cross-section's nodes do not lie in one plane "
            "z = constant. Mesh the section in the xy plane in gmsh."
        )


def _check_lines_on_edges(
    lines: np.ndarray,
    tags: np.ndarray,
    edges: np.ndarray,
    regions: tuple[Region, ...],
    path: Path,
) -> None:
    """Every grouped line must be an edge of the section's triangles."""
    off = (lines < 0).any(axis=1)
    known = {tuple(e) for e in edges.tolist()}
    off |= np.array([tuple(sorted(e)) not in known for e in lines.tolist()], bool)
    if off.any():
        name = {r.tag: r.name for r in regions if r.dim == 2}
        group = name.get(int(tags[np.flatnonzero(off)[0]]))
        raise ValueError(
            f"{path.name}: a line of the 1D physical group {group!r} is not an edge "
            "of the section's triangles, so the lines and the triangles do not "
            "conform. Mesh the lines as part of the 2D mesh in gmsh."
        )


def _default_thickness(points: np.ndarray, edges: np.ndarray) -> float:
    """The mean in-plane edge length: well-shaped tetrahedra (ADR 0170)."""
    return float(
        np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).mean()
    )


def _split_prisms(tris: np.ndarray, n: int, nodes: np.ndarray) -> np.ndarray:
    """Three positive tetrahedra per prism over each triangle, rows per triangle.

    Each quadrilateral side is cut along the diagonal from its lowest-index node
    (the lowest-index rule), so two prisms that share a side cut it alike and the
    tetrahedra conform.
    """
    a, b, c = np.sort(tris, axis=1).T
    tets = np.stack(
        [
            np.stack([a, b, c, c + n], axis=1),
            np.stack([a, b, b + n, c + n], axis=1),
            np.stack([a, a + n, b + n, c + n], axis=1),
        ],
        axis=1,
    ).reshape(-1, 4)
    x = nodes[tets]
    e = x[:, 1:] - x[:, :1]
    negative = np.linalg.det(e) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return tets


def _split_quads(lines: np.ndarray, n: int) -> np.ndarray:
    """Two triangles per extruded line, cut as `_split_prisms` cuts the sides.

    Each triangle's normal is the line's direction crossed with +z, so a
    consistently oriented 1D group gives a consistently oriented surface.
    """
    p, q = lines.T
    low = p < q
    first = np.where(
        low[:, None], np.stack([p, q, q + n], 1), np.stack([p, q, p + n], 1)
    )
    second = np.where(
        low[:, None], np.stack([p, q + n, p + n], 1), np.stack([q, q + n, p + n], 1)
    )
    return np.stack([first, second], axis=1).reshape(-1, 3)


def _read_cross_section(
    path: Path, length_unit: str, thickness: float | None
) -> _Topology:
    """Parse a triangle section and extrude it (`Mesh.from_gmsh_cross_section`)."""
    _check_unit(length_unit)
    if thickness is not None:
        thickness = _check_thickness(thickness)
    raw = meshio.read(path, file_format="gmsh")
    regions = _regions(raw, section=True)
    taken = [r.name for r in regions if r.name in END_FACES]
    if taken:
        raise ValueError(
            f"{path.name} names a physical group {taken[0]!r}, but a cross-section's "
            f"end faces are named {END_FACES[0]!r} and {END_FACES[1]!r}. Rename the "
            "group in gmsh and export the mesh again."
        )
    cells = _cells(raw, path, section=True)
    tris, area_tags, area_entities = cells["triangle"]
    lines, line_tags, line_entities = cells["line"]
    if len(tris) == 0:
        raise ValueError(
            f"{path.name} has no triangle in a 2D physical group: a cross-section "
            "needs its areas meshed and named as gmsh physical groups."
        )
    # keep the section's nodes, in file order
    used = np.unique(tris)
    index = np.full(len(raw.points), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    points = np.asarray(raw.points, dtype=float)[used] * LENGTH_UNITS[length_unit]
    tris, lines = index[tris], index[lines]
    edges = np.unique(np.sort(tris[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)), axis=0)
    _check_section_plane(points, path)
    _check_lines_on_edges(lines, line_tags, edges, regions, path)
    if thickness is None:
        thickness = _default_thickness(points, edges)
    n = len(points)
    nodes = np.concatenate([points, points + np.array([0.0, 0.0, thickness])])
    tets = _split_prisms(tris, n, nodes)
    sides = _split_quads(lines, n)
    tag = max((r.tag for r in regions if r.dim == 2), default=0)
    front, back = Region(END_FACES[0], tag + 1, 2), Region(END_FACES[1], tag + 2, 2)
    ends = np.full(len(tris), 1, dtype=np.int64)
    triangles = np.concatenate([sides, tris, tris + n])
    tri_tags = np.concatenate(
        [np.repeat(line_tags, 2), ends * front.tag, ends * back.tag]
    )
    tri_entities = np.concatenate(
        [np.repeat(line_entities, 2), ends * front.tag, ends * back.tag]
    )
    neighbours = _triangle_neighbours(nodes, tets, triangles, path)
    topology = _Topology(
        length_unit=length_unit,
        imported_nodes=readonly(nodes),
        regions=(*regions, front, back),
        tetrahedra=readonly(tets),
        tet_tags=readonly(np.repeat(area_tags, 3)),
        tet_entities=readonly(np.repeat(area_entities, 3)),
        triangles=readonly(triangles),
        tri_tags=readonly(tri_tags),
        tri_entities=readonly(tri_entities),
        tri_neighbours=readonly(neighbours),
        source=path.resolve(),
        section_thickness=float(thickness),
    )
    _check_tags(topology, path, section=True)
    return topology


def _read_on_rank_0(
    path: Path, length_unit: str, comm, *, section: bool = False, thickness=None
) -> _Topology:
    """Parse on rank 0 and broadcast (ADR 0060); a failed read raises on every rank.

    With ``section``, read a cross-section (`Mesh.from_gmsh_cross_section`,
    whose ``thickness`` argument this is).

    Raises:
        _mpi.AgreedError: Ranks asked for different paths or units.
    """
    name = "Mesh.from_gmsh_cross_section" if section else "Mesh.from_gmsh"
    _mpi.meet(comm, name)
    asked = comm.allgather((str(path.resolve()), length_unit, section, thickness))
    if len(set(asked)) > 1:
        raise _mpi.AgreedError(
            f"{name} is collective, but ranks asked for {sorted(set(asked), key=str)}. "
            "Ranks reading files of their own pass comm=MPI.COMM_SELF."
        )
    topology, error = None, None
    # a read error is sent to every rank; anything else on rank 0 ends the job
    with _mpi.section(comm, agreed=()):
        if comm.rank == 0:
            try:
                topology = (
                    _read_cross_section(path, length_unit, thickness)
                    if section
                    else _read_gmsh(path, length_unit)
                )
            except _READ_ERRORS as err:
                error = err
        error = comm.bcast(error, root=0)
        if error is None:
            statics, arrays = None, None
            if topology is not None:
                statics = {f: getattr(topology, f) for f in _TOPOLOGY_STATICS}
                arrays = {f: getattr(topology, f) for f in _TOPOLOGY_ARRAYS}
            statics = comm.bcast(statics, root=0)
            arrays = _mpi.bcast_arrays(comm, arrays)
    if error is not None:
        raise error
    if topology is not None:
        return topology
    for array in arrays.values():
        array.flags.writeable = False
    return _Topology(**statics, **arrays)


@jax.tree_util.register_pytree_with_keys_class
class Mesh:
    """A straight-sided tetrahedral mesh with named regions.

    Build one with `Mesh.from_gmsh` from a gmsh ``.msh`` file (format 2.2
    ASCII or 4.1), or with `Mesh.from_gmsh_cross_section` from a 2D triangle
    section, read as a one-element-thick slab; ``length_unit`` names the unit
    of the file's coordinates, and `nodes` are in metres. Regions are the
    gmsh physical groups, by name and number.

    A `Mesh` is a JAX
    `pytree <https://docs.jax.dev/en/latest/pytrees.html>`__:
    `jax.grad` reaches `nodes`; the elements, regions and imported
    coordinates are fixed. Element arrays are 0-based NumPy integer arrays,
    read-only, in file order.

    Example:
        A mesh written in micrometres::

            mesh = gfem.Mesh.from_gmsh("filter.msh", length_unit="um")
    """

    def __init__(self, nodes, topology: _Topology) -> None:
        """Internal: use `Mesh.from_gmsh` or `Mesh.from_gmsh_cross_section`."""
        self.nodes = nodes
        self._topology = topology

    def tree_flatten_with_keys(self):
        """JAX pytree protocol: the one leaf is ``nodes``."""
        return ((jax.tree_util.GetAttrKey("nodes"), self.nodes),), self._topology

    def tree_flatten(self):
        """JAX pytree protocol."""
        return (self.nodes,), self._topology

    @classmethod
    def tree_unflatten(cls, topology: _Topology, children) -> Mesh:
        """JAX pytree protocol."""
        return cls(children[0], topology)

    @classmethod
    def from_gmsh(cls, path: str | Path, length_unit: str = "um", *, comm=None) -> Mesh:
        """Read a gmsh ``.msh`` file (format 2.2 ASCII or 4.1).

        Every physical group becomes a region with its gmsh name and number:
        3D groups are volumes, 2D groups are surfaces. Elements outside any
        physical group, points and lines are ignored.

        Above one rank this is collective: rank 0 reads the file once and every
        rank ends up with the same whole mesh, so a call on one rank alone hangs.

        Args:
            path: The ``.msh`` file.
            length_unit: The length unit of the file's coordinates, one of
                `LENGTH_UNITS`. gsim writes micrometres.
            comm: The mpi4py communicator, resolved as `gfem.solve` resolves its
                own. Ranks reading files of their own pass ``MPI.COMM_SELF``.

        Returns:
            The mesh, with node coordinates in metres.

        Raises:
            ValueError: The unit is unknown, the mesh is second order (curved),
                has elements other than tetrahedra and triangles, or its
                physical groups cannot be addressed by name.
            RuntimeError: Ranks asked for different paths or units.
        """
        comm = _mpi.resolve_comm(comm)
        path = Path(path)
        if comm is None:
            topology = _read_gmsh(path, length_unit)
        else:
            topology = _read_on_rank_0(path, length_unit, comm)
        return cls(jnp.asarray(topology.imported_nodes), topology)

    @classmethod
    def from_gmsh_cross_section(
        cls,
        path: str | Path,
        length_unit: str = "um",
        *,
        thickness: float | None = None,
        comm=None,
    ) -> Mesh:
        """Read a 2D gmsh cross-section as a one-element-thick slab.

        The file holds triangles only (format 2.2 ASCII or 4.1), in the xy
        plane. gfem extrudes the section one element thick along +z and
        splits each prism into three tetrahedra, so neighbouring prisms share
        faces. Every 2D physical group becomes a volume and every 1D physical
        group a surface, with its gmsh name and number; the end faces are the
        surfaces ``front`` (the section's plane) and ``back`` (one thickness
        along +z). Elements outside any physical group, points and other
        lines are ignored.

        The thickness is a numerical choice: a z-invariant problem's solution
        does not depend on it, and quantities per unit length divide the slab's
        totals by `section_thickness`.

        Above one rank this is collective, as `from_gmsh` is.

        Args:
            path: The ``.msh`` file of the section.
            length_unit: The length unit of the file's coordinates, one of
                `LENGTH_UNITS`.
            thickness: The slab thickness in metres. ``None`` picks the mean
                in-plane edge length, which keeps the tetrahedra well shaped.
            comm: The mpi4py communicator, as for `from_gmsh`.

        Returns:
            The slab mesh, with node coordinates in metres.

        Raises:
            ValueError: The unit is unknown, the thickness is not a positive
                length, the file holds 3D elements (read it with
                `from_gmsh`), the section is curved or not in one plane z =
                constant, a 1D group is not on the triangles' edges, a group is
                named ``front`` or ``back``, or the groups cannot be addressed
                by name.
            RuntimeError: Ranks asked for different paths, units or thicknesses.

        Example:
            A heater cross-section written in micrometres::

                mesh = gfem.Mesh.from_gmsh_cross_section("heater.msh")
                mesh.section_thickness  # m
        """
        comm = _mpi.resolve_comm(comm)
        path = Path(path)
        if comm is None:
            topology = _read_cross_section(path, length_unit, thickness)
        else:
            topology = _read_on_rank_0(
                path, length_unit, comm, section=True, thickness=thickness
            )
        return cls(jnp.asarray(topology.imported_nodes), topology)

    def __repr__(self) -> str:
        section = self.section_thickness
        return (
            f"Mesh({len(self.imported_nodes)} nodes, {len(self.tetrahedra)} "
            f"tetrahedra, volumes={list(self.volumes)}, "
            f"surfaces={list(self.surfaces)}, length_unit={self.length_unit!r}"
            + ("" if section is None else f", section_thickness={section:.4g} m")
            + ")"
        )

    # --- regions -------------------------------------------------------------

    @property
    def regions(self) -> tuple[Region, ...]:
        """All regions, volumes then surfaces, each by tag."""
        return self._topology.regions

    @property
    def volumes(self) -> tuple[str, ...]:
        """Names of the volume regions, by tag."""
        return tuple(r.name for r in self.regions if r.dim == 3)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Names of the surface regions, by tag."""
        return tuple(r.name for r in self.regions if r.dim == 2)

    def region(self, name: str) -> Region:
        """Look up a region by its gmsh physical-group name.

        Raises:
            RegionNotFoundError: No region has this name.
        """
        for r in self.regions:
            if r.name == name:
                return r
        raise RegionNotFoundError(name, (r.name for r in self.regions))

    def elements(self, region: str) -> np.ndarray:
        """Indices of a region's elements, in mesh order.

        For a volume, rows of `tetrahedra` (the order of a `MaterialField`'s
        per-element values); for a surface, rows of `triangles`.

        Raises:
            RegionNotFoundError: No region has this name.
        """
        r = self.region(region)
        tags = self.tetrahedron_tags if r.dim == 3 else self.triangle_tags
        return readonly(np.flatnonzero(tags == r.tag))

    def adjacency(self, surface: str) -> tuple[tuple[str | None, str | None], ...]:
        """The ``(domin, domout)`` volume pairs on the two sides of a surface.

        For each triangle of the surface, ``domin`` is the volume behind it
        and ``domout`` the volume in front of it, along the triangle's normal
        (right-hand rule on its node order). ``None`` stands for outside the
        mesh. Each distinct pair appears once, sorted by tag.

        Raises:
            RegionNotFoundError: The mesh has no surface of this name.
        """
        tag = self._surface(surface).tag
        tet_tags = self.tetrahedron_tags
        nb = self.triangle_neighbours[self.triangle_tags == tag]
        sides = np.where(nb >= 0, tet_tags[np.maximum(nb, 0)], -1)
        name = {r.tag: r.name for r in self.regions if r.dim == 3}
        pairs = sorted({(int(a), int(b)) for a, b in sides})
        return tuple((name.get(a), name.get(b)) for a, b in pairs)

    def adjacent_volumes(self, surface: str) -> tuple[str, ...]:
        """The volume regions that border a surface, by tag.

        Raises:
            RegionNotFoundError: The mesh has no surface of this name.
        """
        touching = {v for pair in self.adjacency(surface) for v in pair}
        return tuple(v for v in self.volumes if v in touching)

    def _surface(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name and r.dim == 2:
                return r
        raise RegionNotFoundError(name, self.surfaces, kind="surface")

    # --- export --------------------------------------------------------------

    def to_meshio(self) -> meshio.Mesh:
        """The mesh as a `meshio.Mesh`, coordinates in metres.

        Tetrahedra and triangles carry two cell-data arrays: ``region`` (the
        gmsh physical-group number) and ``entity`` (the gmsh entity number).
        """
        return meshio.Mesh(
            np.asarray(self.nodes, dtype=float),
            [("tetra", self.tetrahedra), ("triangle", self.triangles)],
            cell_data={
                "region": [self.tetrahedron_tags, self.triangle_tags],
                "entity": [self.tetrahedron_entities, self.triangle_entities],
            },
        )

    def to_vtu(self, path: str | Path) -> None:
        """Write the mesh and its region ids to a ``.vtu`` file (see `to_meshio`).

        Args:
            path: The output file.
        """
        meshio.write(path, self.to_meshio(), file_format="vtu")

    # --- elements ------------------------------------------------------------

    @property
    def length_unit(self) -> str:
        """The length unit the mesh file was written in."""
        return self._topology.length_unit

    @property
    def section_thickness(self) -> float | None:
        """The slab thickness (m) of a cross-section, ``None`` for any other mesh.

        Set by `from_gmsh_cross_section`; a quantity per unit length is the
        slab's total divided by it.
        """
        return self._topology.section_thickness

    @property
    def source(self) -> Path | None:
        """The absolute path of the imported file."""
        return self._topology.source

    @property
    def imported_nodes(self) -> np.ndarray:
        """Node coordinates (m) as imported or refined, kept when the nodes move."""
        return self._topology.imported_nodes

    @property
    def tetrahedra(self) -> np.ndarray:
        """Node indices of each tetrahedron, shape ``(n_tet, 4)``."""
        return self._topology.tetrahedra

    @property
    def tetrahedron_tags(self) -> np.ndarray:
        """Physical-group number of each tetrahedron's volume region."""
        return self._topology.tet_tags

    @property
    def tetrahedron_entities(self) -> np.ndarray:
        """Gmsh entity (elementary volume) number of each tetrahedron.

        One region may hold several entities: gsim, for example, tags meshed
        conductor volumes with the name of the surrounding dielectric.
        """
        return self._topology.tet_entities

    @property
    def triangles(self) -> np.ndarray:
        """Node indices of each surface-region triangle, shape ``(n_tri, 3)``."""
        return self._topology.triangles

    @property
    def triangle_tags(self) -> np.ndarray:
        """Physical-group number of each triangle's surface region."""
        return self._topology.tri_tags

    @property
    def triangle_entities(self) -> np.ndarray:
        """Gmsh entity (elementary surface) number of each triangle."""
        return self._topology.tri_entities

    @property
    def triangle_neighbours(self) -> np.ndarray:
        """The tetrahedra on each side of each triangle, shape ``(n_tri, 2)``.

        Column 0 is the tetrahedron behind the triangle (``domin``), column 1
        the one in front of it (``domout``), along the triangle's normal;
        -1 means outside the mesh.
        """
        return self._topology.tri_neighbours


#: The `Mesh` arrays that, with its regions, define its elements.
_ELEMENT_ARRAYS = (
    "tetrahedra",
    "tetrahedron_tags",
    "tetrahedron_entities",
    "triangles",
    "triangle_tags",
    "triangle_entities",
)


def _same_topology(a: Mesh, b: Mesh) -> bool:
    """Whether two meshes have the same elements, tags and regions.

    The same object, or equal arrays: the same file read twice (a loaded
    result's mesh and the model's) gives two objects.
    """
    if a._topology is b._topology:  # noqa: SLF001
        return True
    return a.regions == b.regions and all(
        np.array_equal(getattr(a, name), getattr(b, name)) for name in _ELEMENT_ARRAYS
    )


# Per tetrahedron, its shortest edge, by mesh topology (`_shortest_edges`).
_SHORTEST_EDGES: weakref.WeakKeyDictionary[_Topology, np.ndarray] = (
    weakref.WeakKeyDictionary()
)


def _shortest_edges(mesh: Mesh) -> np.ndarray:
    """The shortest edge (m) of each tetrahedron on the imported nodes.

    Cached per mesh topology: moved nodes do not change it.
    """
    topology = mesh._topology  # noqa: SLF001
    h = _SHORTEST_EDGES.get(topology)
    if h is None:
        x = mesh.imported_nodes[mesh.tetrahedra]
        sq = np.full(len(x), np.inf)
        for i, j in _TET_EDGES:
            sq = np.minimum(sq, np.sum((x[:, i] - x[:, j]) ** 2, axis=1))
        h = _SHORTEST_EDGES[topology] = readonly(np.sqrt(sq))
    return h


def _regions(raw: meshio.Mesh, *, section: bool = False) -> tuple[Region, ...]:
    """The physical groups; on a cross-section 2D groups are volumes, 1D surfaces."""
    lift = 1 if section else 0
    regions = (
        Region(name=name, tag=int(tag), dim=int(dim) + lift)
        for name, (tag, dim) in raw.field_data.items()
        if int(dim) + lift in (2, 3)
    )
    return tuple(sorted(regions, key=lambda r: (-r.dim, r.tag)))


# meshio cell types with a 3D (volume) element
_VOLUME_TYPES = ("tetra", "hexahedron", "wedge", "pyramid")
# the dimension of each cell type a reader keeps
_CELL_DIM = {"tetra": 3, "triangle": 2, "line": 1}


def _cells(
    raw: meshio.Mesh, path: Path, *, section: bool = False
) -> dict[str, tuple[np.ndarray, ...]]:
    """Volume and surface cells with their physical and entity tags.

    For `Mesh.from_gmsh` tetrahedra and triangles; for a cross-section
    (``section``) triangles and lines.
    """
    if "gmsh:physical" not in raw.cell_data:
        dims = "areas and lines" if section else "volumes and surfaces"
        raise ValueError(
            f"{path.name} defines no physical groups. Name the {dims} as gmsh "
            "physical groups and export the mesh again."
        )
    width = {"triangle": 3, "line": 2} if section else {"tetra": 4, "triangle": 3}
    volume = "triangle" if section else "tetra"
    types = sorted({block.type for block in raw.cells})
    # meshio names higher-order cells by their node count (tetra10, line3, ...).
    curved = [t for t in types if t[-1].isdigit()]
    if curved:
        raise ValueError(
            f"{path.name} is a second-order (curved) mesh: it has "
            f"{' / '.join(curved)} elements. gfem supports straight-sided meshes "
            "only. Regenerate it with gmsh `ElementOrder 1` (Python: "
            'gmsh.option.setNumber("Mesh.ElementOrder", 1); command line: '
            "gmsh -order 1)."
        )
    solid = [t for t in types if t in _VOLUME_TYPES]
    if section and solid:
        raise ValueError(
            f"{path.name} has {' / '.join(solid)} elements: it is a 3D mesh. Read "
            "it with gfem.Mesh.from_gmsh; Mesh.from_gmsh_cross_section reads a 2D "
            "section of triangles only."
        )
    if not section and not solid and "triangle" in types:
        raise ValueError(
            f"{path.name} has triangles and no tetrahedra: it is a 2D mesh. Read a "
            "2D cross-section with gfem.Mesh.from_gmsh_cross_section, which "
            "extrudes it into a one-element-thick slab."
        )
    other = [t for t in types if t not in (*width, "line", "vertex")]
    if other:
        supported = (
            "cross-sections of triangles only (lines on boundaries)"
            if section
            else "tetrahedral meshes only (triangles on surfaces)"
        )
        raise ValueError(
            f"{path.name} has {' / '.join(other)} elements. gfem supports {supported}."
        )
    parts: dict[str, list[tuple]] = {k: [] for k in width}
    for block, phys, geom in zip(
        raw.cells,
        raw.cell_data["gmsh:physical"],
        raw.cell_data["gmsh:geometrical"],
        strict=True,
    ):
        if block.type in width:
            parts[block.type].append((block.data, phys, geom))
    _check_overlaps_in_cell_sets(raw, path, section=section)
    out = {}
    for kind, items in parts.items():
        if items:
            data, phys, geom = (np.concatenate(c) for c in zip(*items, strict=True))
        else:
            data, phys, geom = np.empty((0, width[kind])), np.empty(0), np.empty(0)
        grouped = phys != 0  # gmsh writes group 0 for elements outside any group
        if kind == volume and not grouped.all():
            what = (
                ("triangles", "2D", "area", "surfaces")
                if section
                else ("tetrahedra", "volume", "volume", "volumes")
            )
            raise ValueError(
                f"{path.name}: {int((~grouped).sum())} {what[0]} lie outside any "
                f"{what[1]} physical group. Every {what[2]} needs a named region: "
                f"add the missing {what[3]} to a physical group in gmsh."
            )
        out[kind] = (
            data[grouped].astype(np.int64),
            phys[grouped].astype(np.int64),
            geom[grouped].astype(np.int64),
        )
        _check_duplicates(out[kind][0], out[kind][1], raw, kind, path)
    return out


def _overlap_error(path: Path, names: set[str]) -> ValueError:
    return ValueError(
        f"{path.name}: elements belong to more than one physical group "
        f"({', '.join(repr(n) for n in sorted(names))}). gfem needs each "
        "tetrahedron in exactly one volume region and each triangle in at most "
        "one surface region: split the overlapping groups in gmsh so that no "
        "entity is in two of them, and export the mesh again."
    )


def _check_overlaps_in_cell_sets(
    raw: meshio.Mesh, path: Path, *, section: bool = False
) -> None:
    """Msh 4.1: meshio keeps the first physical tag; its cell sets keep all."""
    dims = (1, 2) if section else (2, 3)
    names = [n for n, (_, dim) in raw.field_data.items() if int(dim) in dims]
    sets = {n: raw.cell_sets[n] for n in names if n in (raw.cell_sets or {})}
    for i, block in enumerate(raw.cells):
        members = np.zeros(len(block.data), dtype=np.int64)
        for cells in sets.values():
            members[np.asarray(cells[i], dtype=np.int64)] += 1
        if (members > 1).any():
            shared = np.flatnonzero(members > 1)
            raise _overlap_error(
                path, {n for n, c in sets.items() if np.isin(c[i], shared).any()}
            )


def _check_duplicates(
    data: np.ndarray, tags: np.ndarray, raw: meshio.Mesh, kind: str, path: Path
) -> None:
    """Msh 2.2: an element in two physical groups is written twice."""
    _, inverse, counts = np.unique(
        np.sort(data, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    hit = counts[inverse.ravel()] > 1
    if not hit.any():
        return
    dim = _CELL_DIM[kind]
    name = {int(t): n for n, (t, d) in raw.field_data.items() if int(d) == dim}
    raise _overlap_error(path, {name.get(int(t), str(t)) for t in tags[hit]})


def _triangle_neighbours(
    points: np.ndarray, tets: np.ndarray, tris: np.ndarray, path: Path
) -> np.ndarray:
    """For each triangle, the tetrahedron behind it and the one in front."""
    neighbours = np.full((len(tris), 2), -1, dtype=np.int64)
    if len(tris) == 0:
        return neighbours
    faces = np.sort(tets[:, _TET_FACES].reshape(-1, 3), axis=1)
    _, ids = np.unique(
        np.concatenate([faces, np.sort(tris, axis=1)]), axis=0, return_inverse=True
    )
    ids = ids.ravel()
    face_ids, tri_ids = ids[: len(faces)], ids[len(faces) :]
    counts = np.bincount(face_ids, minlength=ids.max() + 1)
    if (counts > 2).any():
        raise ValueError(f"{path.name}: a face is shared by more than two tetrahedra.")
    n_tets = counts[tri_ids]
    if (n_tets == 0).any():
        bad = int(np.flatnonzero(n_tets == 0)[0])
        raise ValueError(
            f"{path.name}: surface triangle {bad} is not a face of any "
            "tetrahedron, so the surface and volume meshes do not conform."
        )
    order = np.argsort(face_ids, kind="stable")
    first = np.searchsorted(face_ids[order], tri_ids)
    a, b, c = (points[tris[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    for slot in (0, 1):
        has = n_tets > slot
        face = order[first[has] + slot]
        tet, opposite = face // 4, tets.reshape(-1)[face]
        side = np.einsum("ij,ij->i", normal[has], points[opposite] - a[has])
        rows = np.flatnonzero(has)
        neighbours[rows[side < 0], 0] = tet[side < 0]
        neighbours[rows[side > 0], 1] = tet[side > 0]
    two = n_tets == 2
    if (neighbours[two] < 0).any():
        raise ValueError(
            f"{path.name}: two tetrahedra overlap on the same side of a face."
        )
    return neighbours


def _check_tags(topology: _Topology, path: Path, *, section: bool = False) -> None:
    """Every element's physical group must exist and be named."""
    for dim, tags, kind in (
        (3, topology.tet_tags, "tetrahedra"),
        (2, topology.tri_tags, "triangles"),
    ):
        named = {r.tag for r in topology.regions if r.dim == dim}
        missing = sorted(set(np.unique(tags).tolist()) - named)
        if missing:
            # a cross-section's message names the file's elements and groups
            in_file = (
                (dim - 1, "triangles" if dim == 3 else "lines")
                if section
                else (dim, kind)
            )
            raise ValueError(
                f"{path.name}: {in_file[1]} belong to {in_file[0]}D physical group(s) "
                f"{missing} that have no name of their own in $PhysicalNames. "
                "gfem addresses regions by name: give every physical group a "
                "unique name in gmsh and export the mesh again."
            )
