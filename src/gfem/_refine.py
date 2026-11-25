"""Bisection refinement of a gfem mesh.

Longest-edge (Rivara) bisection with conforming closure, on gfem's arrays in
NumPy:

- every marked tetrahedron puts its longest edge in the set of split edges;
- each pass, every tetrahedron holding a split edge bisects on its own longest
  edge (which joins the set), until no tetrahedron holds one;
- triangles then bisect the same way: a face's first bisected edge is always
  its longest, so faces follow their tetrahedra.

Edges are ordered globally by length, then by their sorted node ids, so the
result does not depend on anything but the mesh and the mark. A child keeps
its parent's node order with one vertex replaced by the midpoint, so it keeps
the parent's orientation; unrefined elements keep their relative order, old
nodes keep their indices and midpoints are appended.

`refine_model` refines a whole `Model`: it refuses traced leaves,
checks the quality of a moved mesh, bisects it at its current coordinates and
copies each parent's per-element design values to its children with
`transfer`, the region-local gather.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gfem._boundaries import Conductor, Impedance
from gfem._checks import is_traced
from gfem._materials import (
    PROPERTY_NAMES,
    Material,
    MaterialField,
    PermittivityLaw,
)
from gfem._pytree import readonly
from gfem.errors import NonDifferentiableError
from gfem.mesh import _TET_EDGES, Mesh, _Topology, _triangle_neighbours
from gfem.model import Model
from gfem.shape import _check_quality

_TET_EDGE_PAIRS = np.array(_TET_EDGES)
_TRI_EDGE_PAIRS = np.array([(0, 1), (0, 2), (1, 2)])
# Edge key: the smaller node id times this, plus the larger.
_KEY_BASE = np.int64(1) << 31


@dataclasses.dataclass(frozen=True)
class Refinement:
    """Where each element of a bisected mesh comes from, one level up.

    Attributes:
        tetrahedron_parents: For each tetrahedron of the refined mesh, the row
            of the input mesh's `tetrahedra` it lies in.
        triangle_parents: For each triangle of the refined mesh, the row of the
            input mesh's `triangles` it lies in.
    """

    tetrahedron_parents: np.ndarray
    triangle_parents: np.ndarray


def bisect(mesh: Mesh, marked) -> tuple[Mesh, Refinement]:
    """Bisect the marked tetrahedra, and their neighbours until the mesh conforms.

    The mesh is bisected at its current node coordinates, which become the
    refined mesh's imported coordinates. Every marked tetrahedron is bisected
    at least once; children keep their parent's tags, entities and
    orientation, and surface triangles split with the faces they lie on.

    Args:
        mesh: The mesh to refine (concrete node coordinates).
        marked: Boolean mask over the mesh's tetrahedra, shape ``(n_tet,)``.

    Returns:
        The refined mesh (a new topology with no source file) and the one-level
        parent indices of its tetrahedra and triangles.

    Raises:
        ValueError: ``marked`` is not a boolean mask over the tetrahedra.
    """
    tets = np.asarray(mesh.tetrahedra, np.int64)
    marked = np.asarray(marked)
    if marked.dtype != bool or marked.shape != (len(tets),):
        raise ValueError(
            f"marked must be a boolean array of shape ({len(tets)},), one entry "
            f"per tetrahedron; got {marked.dtype} of shape {marked.shape}."
        )
    splits = _Splits(np.asarray(mesh.nodes, dtype=float))
    first = tets[marked]
    splits.add(np.unique(_longest_edge_keys(splits.x, first, _TET_EDGE_PAIRS)))
    tets, tet_parents = _bisect_cells(
        splits, tets, np.arange(len(tets)), _TET_EDGE_PAIRS, grow=True
    )
    tris = np.asarray(mesh.triangles, np.int64)
    tris, tri_parents = _bisect_cells(
        splits, tris, np.arange(len(tris)), _TRI_EDGE_PAIRS, grow=False
    )
    x = splits.x
    old = mesh._topology  # noqa: SLF001
    topology = _Topology(
        length_unit=old.length_unit,
        imported_nodes=readonly(x),
        regions=old.regions,
        volume_elements=readonly(tets),
        volume_tags=readonly(old.volume_tags[tet_parents]),
        volume_entities=readonly(old.volume_entities[tet_parents]),
        surface_elements=readonly(tris),
        surface_tags=readonly(old.surface_tags[tri_parents]),
        surface_entities=readonly(old.surface_entities[tri_parents]),
        surface_neighbours=readonly(
            _triangle_neighbours(x, tets, tris, Path("refined mesh"))
        ),
        source=None,
        # bisection keeps the extrusion length of a cross-section
        section_thickness=old.section_thickness,
    )
    refinement = Refinement(
        tetrahedron_parents=readonly(tet_parents),
        triangle_parents=readonly(tri_parents),
    )
    return Mesh(jnp.asarray(x), topology), refinement


def refine_model(model: Model, marked) -> tuple[Model, Refinement]:
    """Bisect a model's mesh and carry its design to the children.

    A mesh whose nodes have moved is quality-checked first, then bisected
    (`bisect`) at its current coordinates (they become the refined mesh's
    imported nodes); `transfer_design` then builds the refined model. Each
    per-element `MaterialField` property and each `SheetField` fill is copied
    from parent to children (`transfer`); everything else refers to the mesh
    by name and is kept. The refined model is built by `Model`, so it is
    validated like any other.

    Args:
        model: The model to refine (concrete values only).
        marked: Boolean mask over the mesh's tetrahedra, shape ``(n_tet,)``.

    Returns:
        The refined model and the one-level parents of its mesh's elements.

    Raises:
        NonDifferentiableError: The mesh nodes, a `MaterialField` value, a
            permittivity law or a sheet fill is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`.
        MeshQualityError: The moved mesh has an inverted element.
        ValueError: ``marked`` is not a boolean mask over the tetrahedra.
    """
    _check_concrete(model)
    _check_quality(model.mesh, model._domain.active, stacklevel=2)  # noqa: SLF001
    fine, refinement = bisect(model.mesh, marked)
    return transfer_design(model, fine, refinement), refinement


def transfer_design(model: Model, fine: Mesh, refinement: Refinement) -> Model:
    """``model`` on ``fine``, its mesh bisected by `bisect`; see `refine_model`.

    `refine_mesh` bisects first and calls this only for a mesh under its
    ``max_tetrahedra`` cap.
    """
    coarse = model.mesh

    def gather(values: Any, region: str) -> Any:
        return transfer(values, region, coarse=coarse, fine=fine, refinement=refinement)

    materials = {
        region: _gather_field(value, region, gather)
        if isinstance(value, MaterialField)
        else value
        for region, value in model.materials.items()
    }
    boundaries = tuple(
        dataclasses.replace(
            b, fill=dataclasses.replace(b.fill, fill=gather(b.fill.fill, b.surface))
        )
        if isinstance(b, (Conductor, Impedance)) and b.fill is not None
        else b
        for b in model.boundaries
    )
    return Model(
        fine,
        materials=materials,
        boundaries=boundaries,
        ports=model.ports,
        excluded=model.excluded,
        pml=model.pml,
        background=model.background,
    )


def transfer(
    values: Any, region: str, *, coarse: Mesh, fine: Mesh, refinement: Refinement
) -> Any:
    """Copy a region's per-element values from parents to children.

    ``values`` holds one value (or one row) per element of ``region`` on the
    coarse mesh, in ``coarse.elements(region)`` order; the result holds one per
    element of the region on the fine mesh, in ``fine.elements(region)``
    order. Children never leave their parent's region, so the gather is
    defined. It is plain indexing: a NumPy array gives a NumPy array, a JAX
    array a JAX array, and `jax.grad` passes through it.

    Args:
        values: Shape ``(n,)`` or ``(n, k)``, ``n`` the region's coarse elements.
        region: A volume (tetrahedra) or surface (triangles) region name.
        coarse: The mesh the values live on.
        fine: The refined mesh.
        refinement: The parents of ``fine``'s elements in ``coarse``.

    Raises:
        RegionNotFoundError: ``region`` is not a region of the mesh.
        ValueError: ``values`` does not hold one value per coarse element.
    """
    r = coarse.region(region)
    if r.dim == coarse.dim:
        parents, n_all = refinement.tetrahedron_parents, len(coarse.tetrahedra)
    else:
        parents, n_all = refinement.triangle_parents, len(coarse.triangles)
    old = coarse.elements(region)
    shape = np.shape(values)
    if len(shape) not in (1, 2) or shape[0] != len(old):
        raise ValueError(
            f"Cannot transfer values of shape {shape} on {region!r}: give one "
            f"value per element, shape ({len(old)},) or ({len(old)}, k), in the "
            f"order of mesh.elements({region!r})."
        )
    position = np.full(n_all, -1, dtype=np.int64)
    position[old] = np.arange(len(old))
    index = position[parents[fine.elements(region)]]
    if (index < 0).any():
        raise RuntimeError(f"A child of region {region!r} has a parent outside it.")
    return values[index]


def _gather_field(field: MaterialField, region: str, gather) -> MaterialField:
    """The field with every per-element property gathered to the children."""
    changes = {}
    for name in PROPERTY_NAMES:
        value = getattr(field, name)
        if value is not None and value.ndim == 1 + field.has_principal_values(name):
            changes[name] = gather(value, region)
    # a crystal tensor per element, (n, 6, 3), (n, 6, 6) or (n, 3, 6),
    # gathered as (n, k)
    for name in (
        "electro_optic_tensor",
        "photoelastic_tensor",
        "second_order_nonlinear_tensor",
        "stiffness",
        "piezoelectric_tensor",
    ):
        tensor = getattr(field, name)
        if tensor is not None and tensor.ndim == 3:
            flat = gather(tensor.reshape(tensor.shape[0], -1), region)
            changes[name] = flat.reshape(-1, *tensor.shape[1:])
    # a law blend's density
    if field.blend is not None and field.blend.ndim == 1:
        changes["blend"] = gather(field.blend, region)
    return dataclasses.replace(field, **changes)


def _check_concrete(model: Model) -> None:
    """Raise if a leaf refinement reads or copies is traced."""
    leaves = [("the mesh nodes", model.mesh.nodes)]
    leaves += [
        (f"materials[{region!r}]", value)
        for region, value in model.materials.items()
        if isinstance(value, MaterialField)
    ]
    leaves += [
        (f"materials[{region!r}] permittivity law", value.eps_r)
        for region, value in model.materials.items()
        if isinstance(value, Material) and isinstance(value.eps_r, PermittivityLaw)
    ]
    leaves += [
        (f"the fill of the {type(b).__name__} on {b.surface!r}", b.fill)
        for b in model.boundaries
        if isinstance(b, (Conductor, Impedance)) and b.fill is not None
    ]
    for what, leaf in leaves:
        if is_traced(leaf):
            raise NonDifferentiableError(
                f"Cannot refine a model whose {what} is computed inside "
                "jax.grad, jax.jit or jax.vmap: a refined mesh is a new mesh, "
                "which gfem cannot build inside a trace, and refinement has no "
                "derivative. Refine concrete values, outside the transformed "
                "function; the transfer of values to the refined mesh is "
                "differentiable."
            )


class _Splits:
    """The split edges, by key, with their midpoint node ids; and all nodes."""

    def __init__(self, nodes: np.ndarray) -> None:
        self.x = nodes
        self.keys = np.empty(0, np.int64)
        self.mids = np.empty(0, np.int64)

    def has(self, keys: np.ndarray) -> np.ndarray:
        if not len(self.keys):
            return np.zeros(keys.shape, bool)
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return self.keys[i] == keys

    def midpoint(self, keys: np.ndarray) -> np.ndarray:
        return self.mids[np.searchsorted(self.keys, keys)]

    def add(self, keys: np.ndarray) -> None:
        """Split these new edges (unique, sorted keys): append their midpoints."""
        a, b = keys // _KEY_BASE, keys % _KEY_BASE
        mids = len(self.x) + np.arange(len(keys))
        self.x = np.concatenate([self.x, 0.5 * (self.x[a] + self.x[b])])
        all_keys = np.concatenate([self.keys, keys])
        order = np.argsort(all_keys, kind="stable")
        self.keys = all_keys[order]
        self.mids = np.concatenate([self.mids, mids])[order]


def _keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b) * _KEY_BASE + np.maximum(a, b)


def _longest_local_edge(x: np.ndarray, cells: np.ndarray, pairs: np.ndarray):
    """Per cell, its longest local edge: by length, then the larger edge key."""
    a, b = cells[:, pairs[:, 0]], cells[:, pairs[:, 1]]
    length = np.linalg.norm(x[a] - x[b], axis=-1)
    tied = length == length.max(axis=1, keepdims=True)
    return np.argmax(np.where(tied, _keys(a, b), -1), axis=1)


def _longest_edge_keys(x, cells, pairs) -> np.ndarray:
    k = _longest_local_edge(x, cells, pairs)
    rows = np.arange(len(cells))
    return _keys(cells[rows, pairs[k, 0]], cells[rows, pairs[k, 1]])


def _bisect_cells(splits: _Splits, cells, parents, pairs, *, grow: bool):
    """Bisect every cell holding a split edge, on its longest edge, until none does.

    With ``grow``, a longest edge not yet split joins the split set (the
    tetrahedra's closure); without it, it must already be split (triangles
    follow their tetrahedra).
    """
    while True:
        edge_keys = _keys(cells[:, pairs[:, 0]], cells[:, pairs[:, 1]])
        active = splits.has(edge_keys).any(axis=1)
        if not active.any():
            return cells, parents
        hit = cells[active]
        k = _longest_local_edge(splits.x, hit, pairs)
        rows = np.arange(len(hit))
        i, j = pairs[k, 0], pairs[k, 1]
        keys = _keys(hit[rows, i], hit[rows, j])
        new = ~splits.has(keys)
        if new.any():
            if not grow:
                raise RuntimeError(
                    "Bisection left a surface triangle split off its longest edge."
                )
            splits.add(np.unique(keys[new]))
        mid = splits.midpoint(keys)
        first, second = hit.copy(), hit.copy()
        first[rows, j] = mid
        second[rows, i] = mid
        cells = np.concatenate([cells[~active], first, second])
        parents = np.concatenate([parents[~active], parents[active], parents[active]])
