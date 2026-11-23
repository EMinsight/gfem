r"""Incident-field columns of a driven solve: load, scatterer and truncation rules.

An incident field :math:`w` of the background stack enters as the exact
total/scattered change of variables of the scattered-field problem. The
unknown is the total field on every DOF of an element outside the perfectly
matched layer and the scattered field in the layer, and the right-hand side
is

.. math::

    b = A_\mathrm{bg}(\chi\,\Pi w),

with :math:`\Pi` the moment interpolant (`gfem._engine.moments`),
:math:`\chi` the mask of the DOFs of the elements outside the layer, and
:math:`A_\mathrm{bg}` the **background operator**: the operator of the
model with every scatterer element's material replaced by the background
stack's layer at its height, its interior sheets (`gfem.Conductor`,
`gfem.Impedance`) and lumped ports left out, and the same truncation
(layer, `gfem.Absorbing` faces) and wave port terms (`background_model`).
The scatterer difference :math:`D = A - A_\mathrm{bg}` is then the
scatterer: the elements whose region's material is not the stack's layer
at their height (`stack_layers`: no element crosses a stack plane, each
takes the layer at its centroid), compared by region and layer, never by a
value that could move with a design (a `gfem.MaterialField`, a permittivity
law or an anisotropic material is always one, whatever its values), and
the interior sheet and lumped terms.
:math:`b` reads no scatterer material, so it does not depend on a material
design. A `gfem.PEC` surface fixes the total field: a perfect conductor in
the scatterer needs no lift.

The solution holds the total field, the kept field; the scattered part is
:math:`u - \chi\Pi w` and the incident part :math:`\chi\Pi w`, interpolated
again on the field's own space when read (`incident_coefficients`).

Truncation and walls: the incident field is dropped in the layer, so it must
be negligible (`CONTACT`, relative to its peak over the physical region's
vertices) wherever a scatterer meets the layer or an `gfem.Absorbing` face,
and on the physical part of every wave port face; an exterior `gfem.PEC`
wall must hold :math:`|n\times E_\mathrm{inc}|`, an exterior PMC wall
(named, or by default) :math:`\eta|n\times H_\mathrm{inc}|`, below the same
level. The values are read at the vertices and edge midpoints of the faces,
before any factorization. An exterior impedance, conductor or lumped port
face, and a model without a layer or an absorbing face, are refused.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _beam, _incident
from gfem._background import Background
from gfem._boundaries import (
    PEC,
    Absorbing,
    Conductor,
    Impedance,
    LumpedPort,
    surfaces_of,
)
from gfem._checks import is_traced
from gfem._engine.fields import face_keys
from gfem._engine.moments import moment_interpolant
from gfem._incident import Incident, IncidentPart, layered
from gfem._linalg.base import free_mask
from gfem._materials import Material, MaterialField
from gfem._sources import GaussianBeam
from gfem._transfer_matrix import layer_index, number
from gfem.mesh import _TET_FACES, Mesh
from gfem.model import Model

if TYPE_CHECKING:
    from gfem._engine.system_operator import SystemOperator
    from gfem._field import Field
    from gfem._pml import Stretch

__all__ = [
    "CONTACT",
    "IncidentColumns",
    "background_model",
    "check_incident_model",
    "check_stack",
    "incident_coefficients",
    "layer_regions",
    "scatterer_elements",
    "scatterer_regions",
    "stack_layers",
    "truncation_designs",
]

#: The largest incident field, relative to its peak in the physical region,
#: where the field is dropped or a wall must hold it: below it the error of
#: dropping the field sits under a layer's own reflection floor.
CONTACT = 1e-3


def layer_regions(model: Model) -> tuple[str, ...]:
    """The perfectly matched layer's regions; empty without a layer."""
    return () if model.pml is None else tuple(model.pml.regions)


def _elements_of(mesh: Mesh, regions: Sequence[str]) -> np.ndarray:
    """One boolean per tetrahedron: in one of ``regions``."""
    out = np.zeros(len(mesh.tetrahedra), bool)
    for region in regions:
        out[mesh.elements(region)] = True
    return out


def _same_material(material: Any, layer: Material) -> bool:
    """Whether ``material`` is the background ``layer`` for the electric field.

    A constant isotropic `Material` with the layer's electromagnetic values;
    a `gfem.MaterialField`, a law, an anisotropic or a traced material never
    is, whatever its values: the scatterer is chosen per region, not by
    value, so that it does not move with a design.
    """
    if type(material) is not Material or is_traced(material):
        return False
    if not material.isotropic:
        return False
    names = ("eps_r", "mu_r", "tan_delta", "conductivity")
    try:
        return all(getattr(material, n) == getattr(layer, n) for n in names)
    except (TypeError, ValueError):  # a law against a number
        return False


def _region_of(mesh: Mesh, regions: Sequence[str]) -> np.ndarray:
    """Per tetrahedron, the name of its region among ``regions``, or ``""``."""
    out = np.full(len(mesh.tetrahedra), "", object)
    for region in regions:
        if region in mesh.volumes:
            out[mesh.elements(region)] = region
    return out


def _heights(mesh: Mesh) -> np.ndarray:
    """The nodes' :math:`z`, m; the imported ones while the nodes are traced.

    A solve with an incident field refuses traced nodes by name later on.
    """
    nodes = mesh.imported_nodes if is_traced(mesh.nodes) else mesh.nodes
    return np.asarray(nodes, float)[:, 2]


def stack_layers(model: Model, background: Background) -> np.ndarray:
    """The background stack's layer of each tetrahedron, from its centroid.

    Raises:
        ValueError: A tetrahedron of the solve domain has vertices on both
            sides of a stack plane, beyond :math:`10^{-9}` of the mesh's
            :math:`z` extent.
    """
    mesh = model.mesh
    tets = mesh.tetrahedra
    if not background.interfaces:
        return np.zeros(len(tets), int)
    z = _heights(mesh)
    zt = z[tets]
    tol = 1e-9 * float(np.ptp(z))
    active = model._domain.active  # noqa: SLF001
    region_of = _region_of(mesh, mesh.volumes)
    for plane in background.interfaces:
        cut = active & (zt.min(axis=1) < plane - tol) & (zt.max(axis=1) > plane + tol)
        if cut.any():
            regions = ", ".join(repr(r) for r in sorted(set(region_of[cut])))
            raise ValueError(
                f"The background stack plane z = {plane:g} m cuts {int(cut.sum())} "
                f"tetrahedra of regions {regions}: every element must lie on one "
                "side of each plane. Mesh the stack's planes (split the regions "
                "there), or move the interfaces onto the mesh's planes."
            )
    return layer_index(background, zt.mean(axis=1))


def scatterer_elements(model: Model, background: Background) -> np.ndarray:
    """One boolean per tetrahedron of the solve domain: in the scatterer.

    An element is background when its region's material is the stack's
    layer at its height (`stack_layers`, `_same_material`).
    """
    mesh = model.mesh
    layers = stack_layers(model, background)
    out = np.zeros(len(mesh.tetrahedra), bool)
    for region, material in model.materials.items():
        if region not in mesh.volumes:
            continue
        rows = mesh.elements(region)
        same = np.array(
            [_same_material(material, layer) for layer in background.layers]
        )
        out[rows] = ~same[layers[rows]]
    return out & model._domain.active  # noqa: SLF001


def scatterer_regions(model: Model, background: Background) -> tuple[str, ...]:
    """The volume regions with an element in the scatterer.

    Regions without an element in the electromagnetic solve domain are left
    out.
    """
    mesh = model.mesh
    scatterer = scatterer_elements(model, background)
    return tuple(
        region
        for region in model.materials
        if region in mesh.volumes and scatterer[mesh.elements(region)].any()
    )


def _stack_material(
    background: Background, layers: np.ndarray
) -> Material | MaterialField:
    """The stack's material on elements of the given layers, in their order."""
    unique = np.unique(layers)
    if len(unique) == 1:
        return background.layers[int(unique[0])]
    stack = background.layers

    def per(name: str) -> jax.Array:
        return jnp.asarray([number(getattr(stack[j], name)) for j in layers])

    return MaterialField(
        eps_r=per("eps_r"),
        mu_r=per("mu_r"),
        tan_delta=per("tan_delta"),
        conductivity=per("conductivity"),
    )


def background_model(model: Model, background: Background) -> Model:
    """The model whose operator is the background operator (module docstring)."""
    mesh = model.mesh
    scatterer = scatterer_elements(model, background)
    layers = stack_layers(model, background)
    materials = {}
    for region, material in model.materials.items():
        if region in mesh.volumes and scatterer[mesh.elements(region)].any():
            materials[region] = _stack_material(
                background, layers[mesh.elements(region)]
            )
        else:
            materials[region] = material
    return model.replace(
        materials=materials,
        boundaries=[
            b for b in model.boundaries if not isinstance(b, (Conductor, Impedance))
        ],
        ports=[p for p in model.ports if not isinstance(p, LumpedPort)],
    )


def check_stack(model: Model, background: Background) -> None:
    """The stack's match with the mesh, and its ground plane's PEC face.

    Raises:
        ValueError: A tetrahedron across a stack plane (`stack_layers`); with
            a ground plane, an element of the solve domain below it, or a
            face of the physical region's boundary on it that is not a
            `gfem.PEC` wall, or none.
    """
    stack_layers(model, background)
    z_g = background.ground
    if z_g is None:
        return
    mesh = model.mesh
    z = _heights(mesh)
    tol = 1e-9 * float(np.ptp(z))
    active = model._domain.active  # noqa: SLF001
    below = active & (z[mesh.tetrahedra].mean(axis=1) < z_g - tol)
    if below.any():
        raise ValueError(
            f"The model has {int(below.sum())} elements of the solve domain "
            f"below the background stack's ground plane at z = {z_g} m, where "
            "the stack has no field: end the mesh on the ground plane, a PEC "
            "wall there, or exclude what lies below."
        )
    faces = _Faces.of(model)
    layer = _elements_of(mesh, layer_regions(model))
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    on_plane = (np.abs(z[faces.nodes] - z_g) <= tol).all(axis=1)
    wall = on_plane & (b < 0) & ~layer[a]
    pec = {
        s
        for item in model.boundaries
        if isinstance(item, PEC)
        for s in surfaces_of(item)
    }
    covered = np.array([bool(pec & set(n)) for n in faces.surfaces], bool)
    if not wall.any() or not covered[wall].all():
        raise ValueError(
            f"The background stack's ground plane at z = {z_g} m needs an "
            "exterior gfem.PEC wall on that plane under the whole physical "
            "region (outside the PML layer): end the mesh there and put "
            "gfem.PEC on its bottom face."
        )


def _name(source: Incident) -> str:
    return f"incident field {source.name!r}"


def check_incident_model(model: Model, sources: Sequence[Incident]) -> None:
    """The refusals the model's topology alone shows, before any assembly.

    Raises:
        ValueError: The model has no perfectly matched layer and no
            `gfem.Absorbing` face, or an exterior `gfem.Impedance`,
            `gfem.Conductor` or `gfem.LumpedPort` face.
        NotImplementedError: A source or background this version does not
            solve (`gfem._incident.check_solvable`).
    """
    if not sources:
        return
    background = model.background
    if background is None:  # pragma: no cover - refused at solve
        raise ValueError("An incident field needs Model(background=...).")
    for source in sources:
        _incident.check_solvable(source, background)
    check_stack(model, background)
    first = _name(sources[0])
    absorbing = any(isinstance(b, Absorbing) for b in model.boundaries)
    if model.pml is None and not absorbing:
        raise ValueError(
            f"The {first} needs a radiating truncation: the model has no PML layer "
            "and no Absorbing face. A closed model takes dipole sources only."
        )
    domain = model._domain  # noqa: SLF001
    for item in (*model.boundaries, *model.ports):
        if not isinstance(item, (Impedance, Conductor, LumpedPort)):
            continue
        kind = type(item).__name__
        for surface in surfaces_of(item):
            if not domain.exterior(surface):
                continue
            what = (
                f"the lumped port {item.name!r} on the exterior face {surface!r}"
                if isinstance(item, LumpedPort)
                else f"the exterior {kind} face {surface!r}"
            )
            raise ValueError(
                f"The {first} cannot meet {what}; use a PML layer or an Absorbing "
                "face, or put the lossy ground in the background stack."
            )


# --- the faces of the solve domain -----------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Faces:
    """Every face of the solve domain's tetrahedra, once.

    Attributes:
        nodes: ``(n, 3)`` node numbers.
        tets: ``(n, 2)`` the active tetrahedra on either side, -1 for none.
        hull: Whether the face is on the mesh's outer boundary (one
            tetrahedron of the whole mesh).
        surfaces: Per face, the names of the mesh surfaces holding it.
    """

    nodes: np.ndarray
    tets: np.ndarray
    hull: np.ndarray
    surfaces: list[tuple[str, ...]]

    @classmethod
    def of(cls, model: Model) -> _Faces:
        mesh = model.mesh
        active = model._domain.active  # noqa: SLF001
        n_nodes = len(mesh.imported_nodes)
        tets = mesh.tetrahedra
        local = tets[:, _TET_FACES].reshape(-1, 3)
        keys = face_keys(local, n_nodes)
        owner = np.repeat(np.arange(len(tets)), 4)
        on = active[owner]
        # by face, the active tetrahedra first
        order = np.lexsort((~on, keys))
        keys, local, owner, on = keys[order], local[order], owner[order], on[order]
        unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
        side = np.full((len(unique), 2), -1)
        side[on[first], 0] = owner[first][on[first]]
        second = np.minimum(first + 1, len(keys) - 1)
        both = (counts > 1) & on[second]
        side[both, 1] = owner[second][both]
        filled = (side >= 0).sum(axis=1)
        keep = filled > 0
        names = mesh._region_names(surfaces=True)  # noqa: SLF001
        by_key: dict[int, list[str]] = {}
        tri_keys = face_keys(mesh.triangles, n_nodes)
        for key, tag in zip(tri_keys, mesh.triangle_tags, strict=True):
            by_key.setdefault(int(key), []).append(names[tag])
        return cls(
            nodes=local[first][keep],
            tets=side[keep],
            hull=(counts == 1)[keep],
            surfaces=[tuple(by_key.get(int(k), ())) for k in unique[keep]],
        )


def truncation_designs(model: Model) -> dict[str, str]:
    """Why each region touching the truncation cannot be differentiated.

    The regions with an element in the perfectly matched layer, sharing a
    face with one, or on an `gfem.Absorbing` face: with an incident column
    the load drops the incident field there, or reads the material behind
    the face, so the load would depend on a design there.
    """
    mesh = model.mesh
    faces = _Faces.of(model)
    layer = _elements_of(mesh, layer_regions(model))
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    two = b >= 0
    near = layer.copy()
    crossing = two & (layer[a] != layer[np.maximum(b, 0)])
    near[a[crossing]] = True
    near[b[crossing]] = True
    absorbing = {
        s
        for item in model.boundaries
        if isinstance(item, Absorbing)
        for s in surfaces_of(item)
    }
    on_absorbing = np.array([bool(absorbing & set(n)) for n in faces.surfaces], bool)
    near[a[on_absorbing]] = True
    out = {}
    for region in model.materials:
        if region in mesh.volumes and near[mesh.elements(region)].any():
            out[region] = (
                f"region {region!r} meets the PML layer or an Absorbing face, "
                "where an incident column drops its field or reads the material "
                "behind the face, so its load would depend on a design there; "
                "keep the design region off the truncation"
            )
    return out


def _face_points(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vertices and edge midpoints of each face: ``(n, 6, 3)``, m."""
    x = nodes[faces]  # (n, 3, 3)
    mid = (x[:, [0, 0, 1]] + x[:, [1, 2, 2]]) / 2
    return np.concatenate([x, mid], axis=1)


def _normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    x = nodes[faces]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class _Check:
    r"""Points where the incident field must be negligible, and how to say so.

    Attributes:
        what: The message's head: where the field must vanish.
        remedy: The message's tail.
        points: ``(n, 6, 3)`` face points, m.
        normals: ``(n, 3)`` unit normals for a wall (the tangential part is
            checked), ``None`` for the whole field.
        magnetic: The wall is PMC: check :math:`\eta\, n\times H`.
    """

    what: str
    remedy: str
    points: np.ndarray
    normals: np.ndarray | None = None
    magnetic: bool = False


_LAYER_REMEDY = (
    "Move the layer or the scatterer's end out of the incident field, narrow "
    "or move the beam, or make the structure part of the background stack."
)
_ABSORBING_REMEDY = (
    "Keep scatterers off absorbing faces under an incident field, or replace "
    "the face by a PML layer clear of the field."
)
_PORT_REMEDY = (
    "Ports read the scattered field and must lie outside the incident field. "
    "Move the port or the beam; for a receiving antenna, drive the port and "
    "use the far field (reciprocity)."
)
_WALL_REMEDY = (
    "With an incident field an exterior wall must be a symmetry plane of the "
    "incident field or part of the background stack; put a PML layer or an "
    "Absorbing face there."
)


def _near(points: np.ndarray) -> str:
    centre = points.reshape(-1, 3).mean(axis=0)
    return "near ({:.3g}, {:.3g}, {:.3g}) m".format(*centre)


def _checks(model: Model, background: Background) -> list[_Check]:  # noqa: PLR0912, PLR0915 - one pass over the faces
    """Every place the incident field must be negligible (see the module docstring)."""
    mesh = model.mesh
    nodes = np.asarray(mesh.nodes, float)
    faces = _Faces.of(model)
    layer = _elements_of(mesh, layer_regions(model))
    scatterers = scatterer_regions(model, background)
    region_of = _region_of(mesh, scatterers)
    region_of[~scatterer_elements(model, background)] = ""
    stack = stack_layers(model, background) if layered(background) else None
    pml = set(layer_regions(model))
    kinds: dict[str, str] = {}
    for b in model.boundaries:
        for s in surfaces_of(b):
            kinds.setdefault(s, type(b).__name__)
    for p in model.ports:
        for s in surfaces_of(p):
            kinds.setdefault(s, type(p).__name__)
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    two = b >= 0
    in_layer_a = layer[a]
    in_layer_b = np.where(two, layer[np.maximum(b, 0)], False)
    scat_a = region_of[a] != ""
    scat_b = np.where(two, region_of[np.maximum(b, 0)] != "", False)
    out: list[_Check] = []

    def add(what: str, remedy: str, rows: np.ndarray, **kw: Any) -> None:
        if rows.any():
            out.append(
                _Check(what, remedy, _face_points(nodes, faces.nodes[rows]), **kw)
            )

    # a scatterer meets the layer
    crossing = two & (in_layer_a != in_layer_b) & (scat_a | scat_b)
    for region in scatterers:
        rows = crossing & (
            (region_of[a] == region) | (region_of[np.maximum(b, 0)] == region)
        )
        if rows.any():
            pts = _face_points(nodes, faces.nodes[rows])
            what = f"region {region!r}, layer face {_near(pts)}"
            if stack is not None and region in pml:
                # a layer region whose material is not the stack's there
                tets = np.where(region_of[a[rows]] == region, a[rows], b[rows])
                j = int(stack[tets[0]])
                what += (
                    f"; its material {model.materials[region]!r} differs from "
                    f"the background stack's layer {j} there, "
                    f"{background.layers[j]!r}"
                )
            out.append(
                _Check(
                    f"reaches the scatterer where it enters the PML layer "
                    f"({what}): |E_inc| there",
                    _LAYER_REMEDY,
                    pts,
                )
            )
    touches_layer = in_layer_a | in_layer_b
    physical_side = ~in_layer_a | (two & ~in_layer_b)
    surface_rows: dict[str, list[int]] = {}
    for row, names in enumerate(faces.surfaces):
        for name in names:
            surface_rows.setdefault(name, []).append(row)
    for surface, listed in surface_rows.items():
        rows = np.zeros(len(faces.nodes), bool)
        rows[listed] = True
        kind = kinds.get(surface)
        sheet = kind in ("Conductor", "Impedance", "LumpedPort", "PEC")
        if sheet:
            # a sheet, a lumped feed or a conductor's shell in the scatterer
            add(
                f"reaches the {kind} surface {surface!r} where it meets the PML "
                "layer: |E_inc| there",
                _LAYER_REMEDY,
                rows & touches_layer & (two | ~faces.hull),
            )
        if kind == "Absorbing":
            add(
                f"reaches a scatterer on the absorbing face {surface!r}: |E_inc| there",
                _ABSORBING_REMEDY,
                rows & scat_a,
            )
        if kind == "WavePort":
            add(
                f"reaches wave port surface {surface!r}: |E_inc| on the port face",
                _PORT_REMEDY,
                rows & physical_side,
            )
    # the walls: faces of the domain's boundary beside the physical region
    boundary = ~two & ~in_layer_a
    named = np.array([bool(n) for n in faces.surfaces], bool)
    for surface, listed in [*surface_rows.items(), ("", None)]:
        if listed is None:
            rows = boundary & ~named
            kind = None
        else:
            rows = np.zeros(len(faces.nodes), bool)
            rows[listed] = True
            rows &= boundary
            kind = kinds.get(surface)
        if not rows.any():
            continue
        if kind == "PEC":
            rows &= faces.hull  # a conductor's shell is in the scatterer
            add(
                f"does not satisfy the PEC wall {surface!r}: |n x E_inc|",
                _WALL_REMEDY,
                rows,
                normals=_normals(nodes, faces.nodes[rows]) if rows.any() else None,
            )
        elif kind in (None, "PMC"):
            if kind == "PMC":
                where = f"the PMC wall {surface!r}"
            elif surface:
                where = (
                    f"the exterior face {surface!r} (PMC by default, no boundary "
                    "condition given)"
                )
            else:
                where = "an unnamed exterior face (PMC by default)"
            add(
                f"does not satisfy {where}: eta |n x H_inc|",
                _WALL_REMEDY,
                rows,
                normals=_normals(nodes, faces.nodes[rows]),
                magnetic=True,
            )
    return out


def check_contact(
    model: Model, sources: Sequence[Incident], frequencies: Sequence[float]
) -> None:
    """Raise where an incident field is not negligible (see the module docstring).

    Raises:
        ValueError: Naming the face or region, the measured ratio and the
            remedy; a `gfem.GaussianBeam` too tight to normalize at one of
            ``frequencies``.
    """
    background = model.background
    if not sources or background is None:
        return
    for source in sources:
        if isinstance(source, GaussianBeam):
            for f in frequencies:
                _beam.check_waist(source, background, f)
    checks = _checks(model, background)
    if not checks:
        return
    mesh = model.mesh
    layer = _elements_of(mesh, layer_regions(model))
    physical = model._domain.active & ~layer  # noqa: SLF001
    peak_points = np.asarray(mesh.nodes, float)[np.unique(mesh.tetrahedra[physical])]
    for source in sources:
        for f in frequencies:
            peak = np.abs(_incident.values(source, background, peak_points, f)).max()
            if peak == 0:
                continue
            for check in checks:
                ratio = _ratio(check, source, background, f) / peak
                if ratio > CONTACT:
                    raise ValueError(
                        f"The {_name(source)} {check.what} is {ratio:.1e} of its peak "
                        f"in the physical region, above {CONTACT:.0e}. "
                        f"{check.remedy}"
                    )


def _ratio(check: _Check, source: Incident, background: Background, f: float) -> float:
    pts = check.points.reshape(-1, 3)
    if check.magnetic:
        eta = _incident.impedance_at(background, pts)[:, None]
        field = eta * _incident.magnetic(source, background, pts, f)
    else:
        field = _incident.values(source, background, pts, f)
    field = field.reshape(check.points.shape)
    if check.normals is not None:
        field = np.cross(check.normals[:, None, :], field)
    return float(np.abs(field).max()) if field.size else 0.0


# --- the load -------------------------------------------------------------------


def _physical(model: Model) -> np.ndarray:
    """One boolean per tetrahedron: in the solve domain, outside the layer."""
    layer = _elements_of(model.mesh, layer_regions(model))
    return model._domain.active & ~layer  # noqa: SLF001


def _robin(coupling: Any) -> tuple[Any, ...]:
    pieces = getattr(coupling, "pieces", None)
    return (coupling.piece,) if pieces is None else tuple(pieces)


class IncidentColumns:
    r"""The right-hand sides :math:`A_\mathrm{bg}(\chi\Pi w)` of incident sources.

    Built per forward on the concrete model, after the contact and wall
    checks (`check_contact`) at every frequency of the sweep.
    """

    def __init__(
        self,
        model: Model,
        system: SystemOperator,
        sources: Sequence[Incident],
        *,
        order: int,
        stretch: Stretch | None,
    ) -> None:
        """Prepare the loads (`check_contact` has run on the model)."""
        from gfem._engine.driven import driven_operator

        background = model.background
        assert background is not None  # noqa: S101 - checked at the rule
        self.sources = tuple(sources)
        self.background = background
        self.interpolant = moment_interpolant(model, order, elements=_physical(model))
        op = system.operator
        bg = driven_operator(
            background_model(model, background), order, stretch=stretch
        )
        if bg.space is not op.space:  # pragma: no cover - same key by construction
            raise RuntimeError("The background operator is not on the solve space.")
        robin = [p for c in system.surfaces for p in _robin(c)]
        self.operator = dataclasses.replace(bg, pieces=[*bg.pieces, *robin], fixed=None)
        self.system = system
        self.free = free_mask(op.space.FreeDofs(), op.space.ndof)

    def coefficients(self, omega: float) -> np.ndarray:
        r""":math:`\chi\Pi w` of each source at ω, ``(N, n_sources)``."""
        f = omega / (2 * math.pi)
        points = self.interpolant.points
        return np.stack(
            [
                self.interpolant(_incident.values(s, self.background, points, f))
                for s in self.sources
            ],
            axis=1,
        )

    def __call__(self, omega: float) -> np.ndarray:
        """The loads at ω, ``(N, n_sources)``."""
        w = self.coefficients(omega)
        op = self.operator
        b = np.zeros_like(w)
        for k, piece in enumerate(op.pieces):
            b += complex(piece.theta(omega)) * op.apply(k, w)
        if self.system.surfaces:
            term = self.system.corrections(omega)
            right = term.left if term.right is None else term.right
            b += term.left @ (term.coefficients @ (right.T @ w))
        b[~self.free] = 0
        return b


def incident_columns(
    model: Model,
    system: SystemOperator,
    sources: Sequence[Incident],
    *,
    order: int,
    stretch: Stretch | None,
) -> Callable[[float], np.ndarray]:
    """The loads of ``sources`` as a function of ω (see `IncidentColumns`)."""
    return IncidentColumns(model, system, sources, order=order, stretch=stretch)


def incident_coefficients(field: Field, part: IncidentPart) -> np.ndarray:
    r""":math:`\chi\Pi w` on ``field``'s space and rows, engine coefficients.

    The moment interpolant of ``part``'s source at the field's frequency,
    on the elements of the field's support outside the layer.
    """
    from gfem._engine.moments import field_interpolant

    mesh = field.mesh
    elements = ~_elements_of(mesh, part.layers)
    interpolant = field_interpolant(
        mesh,
        field.support,
        field._excluded,  # noqa: SLF001
        field.order,
        elements=elements,
    )
    f = float(np.real(np.asarray(field.omega))) / (2 * math.pi)
    return interpolant(
        _incident.values(part.source, part.background, interpolant.points, f)
    )
