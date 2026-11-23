r"""The far field of a solved field: a reciprocity integral over an element layer.

The host kernel of `gfem.far_field`. With :math:`\chi` the P1 nodal
indicator of a side (1 on the nodes of its elements, 0 elsewhere) and
:math:`w = \hat q\,e^{ik\hat r\cdot r}` the plane wave arriving from
:math:`\hat r` (:math:`\hat q \perp \hat r`), every field that solves the
source-free equations of a homogeneous medium where :math:`\nabla\chi \ne 0`
has the far-field amplitude

.. math::

    \hat q\cdot F(\hat r) = \frac{1}{4\pi}\int_L \Big[
        E\cdot(\nabla\chi\times\nabla\times w)
        + (\nabla\times E)\cdot(\nabla\chi\times w)\Big]\,dV,

the Stratton-Chu surface integral averaged over the level sets of
:math:`\chi`. :math:`L` is the element layer where :math:`\chi` varies:
the elements just outside the side. It needs no surface, normals or face
traces, and it reads the field where it is smooth, inside elements.

By default the side is every physical element (outside the perfectly
matched layer) with no vertex on the layer or on an `gfem.Absorbing` face,
so that :math:`L` is the truncation layer, the physical elements touching
the truncation.

A ground plane, a flat exterior PEC or PMC face that :math:`L` meets, adds
the mirrored plane wave to :math:`w`, so that :math:`w` meets the wall's
condition (image theory) and the wall adds no surface term.

On straight-sided elements a field of fixed coefficients moves with the
nodes covariantly (:math:`E = J^{-T}\hat E`, :math:`\nabla\times E =
J\hat\nabla\times\hat E/\det J`), so the amplitude is a closed-form
function of the layer's node coordinates at fixed reference values:
`amplitudes` is written in `jax.numpy` and differentiated by `jax.vjp`
for both the field and the node coordinates.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Conductor,
    FloquetPort,
    Impedance,
    LumpedPort,
    WavePort,
    surfaces_of,
)
from gfem._engine import power_flow as pf
from gfem._engine import probe
from gfem._engine.incident import _Faces, _same_material, layer_regions
from gfem._materials import Material, MaterialField
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._background import Background
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "Ground",
    "Layer",
    "amplitudes",
    "directions",
    "exterior_layer",
    "rule",
]

#: The sheets a far-field layer may not hold.
_SHEETS = (Conductor, Impedance)
#: Directions per chunk of the dense phase product (bounds its memory).
_CHUNK = 64
#: A ground plane's vertices lie on one plane to this share of the mesh extent.
_FLAT = 1e-9


@dataclasses.dataclass(frozen=True, eq=False)
class Ground:
    """A flat PEC or PMC exterior face the far-field layer meets.

    Attributes:
        point: A point of the plane, m.
        normal: Its unit normal, pointing into the solve domain.
        sign: -1 for a perfect conductor (the image of a tangential test
            field is opposite), +1 for a perfect magnetic wall.
        kind: ``"PEC"`` or ``"PMC"``.
    """

    point: np.ndarray
    normal: np.ndarray
    sign: int
    kind: str


@dataclasses.dataclass(frozen=True, eq=False)
class Layer:
    """The far-field layer of a side: topology, medium and ground plane.

    Attributes:
        chi: The side's P1 indicator, one value per node.
        tetrahedra: The layer's tetrahedra (where ``chi`` varies).
        inside: The tetrahedra with ``chi = 1`` on every vertex, one
            boolean per tetrahedron: the side whose power flow is the
            radiated power.
        eps_r: The exterior medium's relative permittivity.
        mu_r: The exterior medium's relative permeability.
        ground: The ground plane, or ``None``.
    """

    chi: np.ndarray
    tetrahedra: np.ndarray
    inside: np.ndarray
    eps_r: float
    mu_r: float
    ground: Ground | None

    @property
    def index(self) -> float:
        """The exterior medium's refractive index."""
        return math.sqrt(self.eps_r * self.mu_r)


def rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """The layer's quadrature: barycentric points and weights summing to 1.

    Degree :math:`2p + 4`: the field times a plane wave is no polynomial, and
    this degree put the transform's error under the field's at orders 1-3.
    """
    return probe.tetrahedron_rule(2 * order + 4)


def located(layer: Layer, bary: np.ndarray) -> probe.Located:
    """The quadrature points of the layer, tetrahedron-major."""
    n = len(layer.tetrahedra)
    return probe.Located(
        tetrahedra=np.repeat(layer.tetrahedra, len(bary)),
        barycentric=np.tile(bary, (n, 1)),
    )


def directions(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, ...]:
    r""":math:`\hat r`, :math:`\hat\theta`, :math:`\hat\phi`, each ``(n, 3)``."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], axis=-1)
    that = np.stack([ct * cp, ct * sp, -st], axis=-1)
    phat = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return rhat, that, phat


# --- the amplitude ----------------------------------------------------------------


def _jacobians(corners: Any) -> Any:
    """``J[t][:, i] = X_i - X_3``: the map from the first three barycentrics."""
    return jnp.stack([corners[:, i] - corners[:, 3] for i in range(3)], axis=-1)


def reference_values(corners: Any, e: Any, curl: Any) -> tuple[Any, Any]:
    r"""The covariant reference values of ``e`` and ``curl`` at each point.

    :math:`\hat E = J^T E` and :math:`\hat\nabla\times\hat E = \det J\,J^{-1}
    \nabla\times E`, ``(n_tets, n_points, 3)`` from ``(n_tets * n_points,
    3)``: what stays fixed when the nodes move at fixed coefficients.
    """
    jac = _jacobians(corners)
    det = jnp.linalg.det(jac)
    e = e.reshape(len(corners), -1, 3)
    curl = curl.reshape(len(corners), -1, 3)
    e_ref = jnp.einsum("tji,tpj->tpi", jac, e)
    c_ref = det[:, None, None] * jnp.einsum("tij,tpj->tpi", jnp.linalg.inv(jac), curl)
    return e_ref, c_ref


def amplitudes(
    corners: Any,
    e_ref: Any,
    c_ref: Any,
    *,
    chi: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    rhat: np.ndarray,
    k: float,
    ground: Ground | None,
) -> Any:
    r"""The vector :math:`4\pi` times the far-field amplitude, ``(n_dir, 3)``.

    Its components along :math:`\hat\theta` and :math:`\hat\phi` are
    :math:`4\pi F_\theta` and :math:`4\pi F_\phi`. A `jax.numpy` function of
    the layer's corner coordinates ``(n_tets, 4, 3)`` (m) and the field's
    reference values (`reference_values`).
    """
    jac = _jacobians(corners)
    det = jnp.linalg.det(jac)
    inv_t = jnp.linalg.inv(jac).transpose(0, 2, 1)  # J^{-T}
    e = jnp.einsum("tij,tpj->tpi", inv_t, e_ref)
    curl = jnp.einsum("tij,tpj->tpi", jac, c_ref) / det[:, None, None]
    d_chi = jnp.asarray(chi[:, :3] - chi[:, 3:])  # d chi / d barycentric
    grad = jnp.einsum("tij,tj->ti", inv_t, d_chi)
    points = corners[:, 3][:, None] + jnp.einsum("tij,pj->tpi", jac, bary[:, :3])
    w = jnp.abs(det)[:, None] / 6 * weights[None]
    g = grad[:, None, :]
    a = (jnp.cross(curl, g) * w[..., None]).reshape(-1, 3)
    b = (jnp.cross(e, g) * w[..., None]).reshape(-1, 3)
    x = points.reshape(-1, 3)
    out = []
    for start in range(0, len(rhat), _CHUNK):
        r = rhat[start : start + _CHUNK]
        v = _sum(x, a, b, r, k=k, shift=None)
        if ground is not None:
            mirror = np.eye(3) - 2 * np.outer(ground.normal, ground.normal)
            image = _sum(x, a, b, r @ mirror, k=k, shift=(ground.point, r))
            v = v + ground.sign * image @ mirror
        out.append(v)
    return jnp.concatenate(out)


def _sum(
    x: Any,
    a: Any,
    b: Any,
    r: np.ndarray,
    *,
    k: float,
    shift: tuple[np.ndarray, np.ndarray] | None,
) -> Any:
    r""":math:`\sum_p e^{ik r\cdot x_p}(a_p + ik\, b_p\times r)`, ``(n_dir, 3)``.

    ``shift = (c, r0)`` refers an image's phase to the plane point ``c``:
    :math:`e^{ik r\cdot(x - c) + ik r_0\cdot c}`.
    """
    phase = 1j * k * (x @ r.T)  # (n_points, n_dir)
    if shift is not None:
        c, r0 = shift
        phase = phase + 1j * k * (r0 @ c - r @ c)[None, :]
    ph = jnp.exp(phase)
    sa = ph.T @ a
    sb = ph.T @ b
    return sa + 1j * k * jnp.cross(sb, r)


# --- the layer --------------------------------------------------------------------


def exterior_layer(
    field: Field,
    model: Model,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray] | None,
    background: Background,
    *,
    nodes: np.ndarray,
    incident: bool = False,
) -> Layer:
    """The far-field layer of ``side`` (``None``: the truncation layer), checked.

    Reads the node coordinates only for a predicate side, a dipole's
    element and a ground plane's geometry; the caller passes concrete ones.
    ``incident``: the field is an incident-field column's scattered field,
    radiated by the scatterer, which the side must then hold.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: A closed model; a side that selects nothing or every
            element; a layer that is not the background medium, or holds a
            perfectly matched layer, a design, a source, a port, a sheet or
            a lumped element; a source or a scatterer outside the side; a
            boundary of the solve domain that the layer meets and that is not
            one flat PEC or PMC ground plane.
    """
    mesh = model.mesh
    tets = np.asarray(mesh.tetrahedra)
    n_nodes = len(mesh.imported_nodes)
    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if model.pml is None and not absorbing:
        raise ValueError(
            "far_field needs a radiating truncation: the model has no PML layer "
            "and no Absorbing face, so its field is a closed cavity's and "
            "radiates nothing to infinity."
        )
    if model._periodic:  # noqa: SLF001
        raise ValueError(
            "far_field reads a field radiating into free space, but the model "
            "has Periodic pairs: a periodic structure radiates diffraction "
            "orders, read them at a FloquetPort."
        )
    support = pf._check_support(field, model)  # noqa: SLF001
    pml = np.zeros(len(tets), bool)
    for region in layer_regions(model):
        pml[mesh.elements(region)] = True
    if side is None:
        cut = np.zeros(n_nodes, bool)
        cut[np.unique(tets[pml & support])] = True
        for item in absorbing:
            for name in surfaces_of(item):
                cut[np.unique(np.asarray(mesh.triangles)[mesh.elements(name)])] = True
        chi = (~cut).astype(float)
        used = np.zeros(n_nodes, bool)
        used[np.unique(tets[support])] = True
        chi[~used] = 0.0
    else:
        selected = pf._selection(mesh, side) & support  # noqa: SLF001
        if not selected.any():
            raise ValueError(
                "far_field's side selects no element of the field's support: "
                "select elements of the solve domain."
            )
        if selected.all() or (support & ~selected).sum() == 0:
            raise ValueError(
                "far_field's side selects every element of the field's support, "
                "so it has no element layer outside it."
            )
        chi = np.zeros(n_nodes)
        chi[np.unique(tets[selected])] = 1.0
    at = chi[tets]
    varies = support & (at.min(axis=1) < 1) & (at.max(axis=1) > 0)
    inside = support & (at.min(axis=1) == 1)
    if not varies.any() or not inside.any():
        raise ValueError(
            "far_field's side has no element layer just outside it: every "
            "element of the solve domain is inside it, or none is."
        )
    region_of = _regions(model)
    if (varies & pml).any():
        region = region_of[np.flatnonzero(varies & pml)[0]]
        raise ValueError(
            f"The far-field layer just outside this side holds the PML region "
            f"{region!r}, where the field is the stretched one: put the side's "
            "surface at least one element inside the PML's inner face, or leave "
            "side=None for the layer touching the truncation."
        )
    eps_r, mu_r = _medium(model, background, region_of, varies)
    _check_sources(field, model, background, chi, varies, incident=incident)
    ground = _ground(model, chi, varies, pml, nodes)
    return Layer(
        chi=chi,
        tetrahedra=np.flatnonzero(varies),
        inside=inside,
        eps_r=eps_r,
        mu_r=mu_r,
        ground=ground,
    )


def _regions(model: Model) -> np.ndarray:
    """The volume region name of each tetrahedron."""
    mesh = model.mesh
    names = mesh._region_names()  # noqa: SLF001
    tags = np.asarray(mesh._topology.volume_tags)  # noqa: SLF001
    out = np.empty(len(tags), object)
    for tag, name in names.items():
        out[tags == tag] = name
    return out


def _medium(
    model: Model,
    background: Background,
    region_of: np.ndarray,
    varies: np.ndarray,
) -> tuple[float, float]:
    """The exterior medium's ``(eps_r, mu_r)``, after checking the layer is it."""
    if len(background.layers) > 1:
        raise NotImplementedError(
            "far_field over a layered background (a dielectric or layered "
            "half-space) is not computed in this gfem version: give a "
            "homogeneous background, with a PEC or PMC ground plane if needed."
        )
    medium = background.layers[0]
    if np.any(medium.tan_delta) or np.any(medium.conductivity):
        raise NotImplementedError(
            f"far_field in a lossy exterior medium ({medium!r}) is not computed "
            "in this gfem version: a lossy medium has no far field. Give a "
            "lossless background."
        )
    for region in sorted(set(region_of[varies])):
        material = model.materials.get(region)
        if isinstance(material, MaterialField):
            # a value error: the region's material is valid, the side is not
            raise ValueError(  # noqa: TRY004
                f"The far-field layer just outside this side holds the design "
                f"region {region!r} (a MaterialField): the layer must be the "
                "exterior medium. Move the side's surface one element away from "
                "the design region."
            )
        if not (isinstance(material, Material) and _same_material(material, medium)):
            raise ValueError(
                f"The far-field layer just outside this side holds region "
                f"{region!r}, whose material {material!r} is not the exterior "
                f"medium (the background, {medium!r}): the layer must be one "
                "constant, isotropic, lossless medium. Put the side's surface in "
                "the background medium, around every scatterer."
            )
    eps_r = complex(np.asarray(medium.eps_r)).real
    return eps_r, complex(np.asarray(medium.mu_r)).real


def _check_sources(
    field: Field,
    model: Model,
    background: Background,
    chi: np.ndarray,
    varies: np.ndarray,
    *,
    incident: bool,
) -> None:
    """Raise for a source, port, sheet or scatterer not strictly inside the side."""
    from gfem._engine.incident import scatterer_regions
    from gfem._probe import layout as probe_layout

    mesh = model.mesh
    tets = np.asarray(mesh.tetrahedra)
    full = chi[tets].min(axis=1) == 1
    sources = getattr(field, "_point_sources", ())
    if sources:
        points = np.array([position for _, position in sources], float)
        found = probe.locate(field.mesh, probe_layout(field), points)
        for (name, _), tet in zip(sources, found.tetrahedra, strict=True):
            if not full[tet]:
                where = "in" if varies[tet] else "outside"
                raise ValueError(
                    f"The dipole {name!r} that radiates this field is {where} the "
                    "far-field layer just outside the side: the side must hold "
                    "every source, one element clear of the layer. Choose a side "
                    "holding the dipole."
                )
    if incident:
        active = model._domain.active  # noqa: SLF001
        for region in scatterer_regions(model, background):
            elements = np.zeros(len(tets), bool)
            elements[mesh.elements(region)] = True
            if (elements & active & ~full).any():
                raise ValueError(
                    f"The scatterer {region!r} reaches the far-field layer just "
                    "outside this side, or beyond it: the side must hold every "
                    "scatterer, one element clear of the layer."
                )
    active = model._domain.active  # noqa: SLF001
    layer_nodes = np.zeros(len(chi), bool)
    layer_nodes[np.unique(tets[varies])] = True
    for item in (*model.boundaries, *model.ports):
        wall = isinstance(item, (PEC, PMC))
        if not isinstance(
            item, (*_SHEETS, PEC, PMC, LumpedPort, WavePort, FloquetPort)
        ):
            continue  # Absorbing faces: the truncation itself
        port = isinstance(item, (LumpedPort, WavePort, FloquetPort))
        what = (
            f"{type(item).__name__} {item.name!r}"
            if port
            else f"{type(item).__name__} surface"
        )
        for name in surfaces_of(item):
            rows = mesh.elements(name)
            if wall:
                # an exterior wall is a ground plane or beyond the truncation
                # (`_ground`); an interior one is a conductor sheet
                nb = np.asarray(mesh.triangle_neighbours)[rows]
                both = (nb >= 0).all(axis=1)
                both[both] = active[nb[both]].all(axis=1)
                rows = rows[both]
            triangles = np.asarray(mesh.triangles)[rows]
            if not len(triangles):
                continue
            if (chi[triangles].min(axis=1) < 1).any() or (
                layer_nodes[triangles].all(axis=1)
            ).any():
                raise ValueError(
                    f"The {what} ({name!r}) is in the far-field layer just "
                    "outside this side, or beyond it: the side must hold every "
                    "port, sheet and lumped element, one element clear of the "
                    "layer."
                )


def _ground(
    model: Model,
    chi: np.ndarray,
    varies: np.ndarray,
    pml: np.ndarray,
    nodes: np.ndarray,
) -> Ground | None:
    """The PEC or PMC ground plane the layer meets, or ``None``.

    The faces that matter are the solve domain's boundary faces of a layer
    element with a vertex where ``chi = 1``: the reciprocity identity gets a
    surface term there unless the test field meets the wall's condition.
    """
    mesh = model.mesh
    faces = _Faces.of(model)
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    boundary = (b < 0) & (a >= 0)
    rows = boundary & varies[np.maximum(a, 0)] & ~pml[np.maximum(a, 0)]
    rows &= chi[faces.nodes].max(axis=1) > 0
    if not rows.any():
        return None
    kinds: dict[str, str] = {}
    for item in model.boundaries:
        for name in surfaces_of(item):
            kinds.setdefault(name, type(item).__name__)
    for item in model.ports:
        for name in surfaces_of(item):
            kinds.setdefault(name, type(item).__name__)
    found: set[str] = set()
    for row in np.flatnonzero(rows):
        named = [kinds[s] for s in faces.surfaces[row] if s in kinds]
        kind = named[0] if named else "PMC"
        if kind not in ("PEC", "PMC"):
            raise ValueError(
                f"The far-field layer meets the {kind} face {faces.surfaces[row]!r} "
                "on the boundary of the solve domain: the layer may end only on a "
                "flat PEC or PMC ground plane. Truncate with a PML layer or an "
                "Absorbing face there."
            )
        found.add(kind)
    if len(found) > 1:
        raise ValueError(
            "The far-field layer meets both PEC and PMC faces of the solve "
            "domain's boundary: a ground plane is one flat PEC or PMC wall."
        )
    nodes = np.asarray(nodes, float)
    tri = faces.nodes[rows]
    x = nodes[tri]
    normal = np.cross(x[0, 1] - x[0, 0], x[0, 2] - x[0, 0])
    normal /= np.linalg.norm(normal)
    point = x[0, 0]
    extent = float(np.ptp(nodes, axis=0).max())
    if np.abs((x - point) @ normal).max() > _FLAT * extent:
        raise ValueError(
            "The far-field layer meets PEC or PMC faces of the solve domain's "
            "boundary that are not on one plane: a ground plane is one flat wall."
        )
    tets = np.asarray(mesh.tetrahedra)
    inward = nodes[tets[a[rows][0]]].mean(axis=0) - point
    if inward @ normal < 0:
        normal = -normal
    (kind,) = found
    return Ground(
        point=point, normal=normal, sign=-1 if kind == "PEC" else 1, kind=kind
    )


def corners(mesh: Mesh, layer: Layer, nodes: np.ndarray) -> np.ndarray:
    """The layer's tetrahedra's vertex coordinates, ``(n, 4, 3)``, m."""
    return np.asarray(nodes, float)[np.asarray(mesh.tetrahedra)[layer.tetrahedra]]


def chi_corners(mesh: Mesh, layer: Layer) -> np.ndarray:
    """``chi`` at the layer's tetrahedra's vertices, ``(n, 4)``."""
    return layer.chi[np.asarray(mesh.tetrahedra)[layer.tetrahedra]]
