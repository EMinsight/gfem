r"""The coupling of a solved field into an incoming beam: a reaction over a layer.

The host kernel of `gfem.beam_coupling`. With :math:`\chi` the P1 nodal
indicator of a side and :math:`w` the beam travelling toward the side (a
`gfem.GaussianBeam` or a `gfem.IncidentField`), the reaction

.. math::

    R = \int_L \Big[E\cdot(\nabla\chi\times\nabla\times w)
        + (\nabla\times E)\cdot(\nabla\chi\times w)\Big]\,dV

is the far-field kernel of `gfem._engine.far_field` with :math:`w` the
beam: the Lorentz reaction :math:`i\omega\mu\oint(w\times H - E\times
H_w)\cdot n\,dS` averaged over the level sets of :math:`\chi`. The beam's
power into the side is the same layer's form

.. math::

    P = \frac{1}{2\omega\mu}\operatorname{Im}\int_L
        \nabla\chi\cdot(w\times\overline{\nabla\times w})\,dV,

and the coupling :math:`c = R/(4i\omega\mu\sqrt P)` is, by reciprocity,
the wave a port receives when the beam, scaled to 1 W through the side,
lights the device: the S-parameter between the beam and the column's port.

:math:`L` is the layer just outside the side, without its perfectly
matched layer elements; by default the physical elements touching the
truncation face the beam enters through. :math:`L` must be one constant,
isotropic, lossless medium: the beam's.

A `gfem.GaussianBeam` is read in closed form at the quadrature points,
with its curl, so that both follow moving nodes. A `gfem.IncidentField`
gives values only: it enters by its moment interpolant on the layer,
built on the imported mesh and carried with the nodes covariantly (a
frozen test field), which equals the beam only while the layer's nodes
stay where they were imported.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _beam, _incident
from gfem._background import Background
from gfem._boundaries import Absorbing, surfaces_of
from gfem._engine import far_field as ff
from gfem._engine import power_flow as pf
from gfem._engine import probe
from gfem._engine.incident import CONTACT, _Faces, _same_material, layer_regions
from gfem._materials import Material, MaterialField
from gfem._sources import GaussianBeam, IncidentField
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "CouplingLayer",
    "coupling_layer",
    "frozen_reference",
    "reaction",
]

#: Planar groups of truncation faces: unit normals and offsets agree to
#: this share of the mesh extent.
_PLANE = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class CouplingLayer:
    """The element layer of a side, in the beam's medium.

    Attributes:
        chi: The side's P1 indicator, one value per node.
        tetrahedra: The layer's tetrahedra (where ``chi`` varies, outside
            the perfectly matched layer).
        eps_r: The medium's relative permittivity.
        mu_r: The medium's relative permeability.
        material: The medium.
        crossing: The beam's power (W, absolute, summed face by face)
            through the truncation faces of the layer's elements that touch
            the side (``chi > 0``): where the side's surface meets the
            truncation. Large against the beam's power through the layer,
            the surface runs along the truncation under the beam.
    """

    chi: np.ndarray
    tetrahedra: np.ndarray
    eps_r: float
    mu_r: float
    material: Material
    crossing: float = 0.0

    @property
    def background(self) -> Background:
        """The medium as a homogeneous background, where the beam is evaluated."""
        return Background(self.material)


def host_field(
    beam: GaussianBeam | IncidentField,
    material: Material,
    points: np.ndarray,
    frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The beam's :math:`E` and :math:`\nabla\times E` at ``points``, NumPy.

    A `gfem.GaussianBeam` in closed form; a `gfem.IncidentField`'s curl by
    central differences of its values.
    """
    background = Background(material)
    if isinstance(beam, GaussianBeam):
        return _beam.field_and_curl(beam, background, points, frequency)
    e = _incident.values(beam, background, points, frequency)
    h = _incident.magnetic(beam, background, points, frequency)
    mu = const.mu_0 * float(np.real(complex(np.asarray(material.mu_r))))
    return e, -1j * 2 * math.pi * frequency * mu * h


# --- the layer ----------------------------------------------------------------------


def coupling_layer(
    field: Field,
    model: Model,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray] | None,
    beam: GaussianBeam | IncidentField,
    frequency: float,
    *,
    nodes: np.ndarray,
    incident: bool = False,
) -> CouplingLayer:
    """The coupling layer of ``side`` (``None``: the beam's entrance), checked.

    ``nodes`` are concrete node coordinates (the imported ones under a
    trace). ``incident``: the field is an incident column's scattered
    field, whose scatterer the side must hold.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: A closed or periodic model; a side that selects nothing
            or everything; a layer that is not one constant, isotropic,
            lossless medium, or holds a source, port, sheet or lumped
            element; a beam that enters through no truncation face; a
            boundary face of the layer that is neither an `gfem.Absorbing`
            face nor a PEC or PMC wall the beam satisfies.
    """
    mesh = model.mesh
    tets = np.asarray(mesh.tetrahedra)
    n_nodes = len(mesh.imported_nodes)
    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if model.pml is None and not absorbing:
        raise ValueError(
            "beam_coupling needs a radiating truncation: the model has no PML "
            "layer and no Absorbing face, so its field is a closed cavity's and "
            "couples to no beam from outside."
        )
    if model._periodic:  # noqa: SLF001
        raise ValueError(
            "beam_coupling reads a field radiating into free space, but the model "
            "has Periodic pairs: a periodic structure radiates diffraction "
            "orders, read them at a FloquetPort."
        )
    support = pf._check_support(field, model)  # noqa: SLF001
    pml = np.zeros(len(tets), bool)
    for region in layer_regions(model):
        pml[mesh.elements(region)] = True
    region_of = ff._regions(model)  # noqa: SLF001
    if side is None:
        cut = _entrance(
            model,
            support=support,
            pml=pml,
            beam=beam,
            frequency=frequency,
            nodes=nodes,
            region_of=region_of,
        )
        chi = np.zeros(n_nodes)
        chi[np.unique(tets[support])] = 1.0
        chi[cut] = 0.0
    else:
        selected = pf._selection(mesh, side) & support  # noqa: SLF001
        if not selected.any():
            raise ValueError(
                "beam_coupling's side selects no element of the field's support: "
                "select elements of the solve domain."
            )
        if (support & ~selected).sum() == 0:
            raise ValueError(
                "beam_coupling's side selects every element of the field's "
                "support, so it has no element layer outside it."
            )
        chi = np.zeros(n_nodes)
        chi[np.unique(tets[selected])] = 1.0
    at = chi[tets]
    varies = support & ~pml & (at.min(axis=1) < 1) & (at.max(axis=1) > 0)
    if not varies.any():
        raise ValueError(
            "beam_coupling's side has no element layer just outside it outside "
            "the perfectly matched layer: put its surface in the physical region."
        )
    material = _medium(model, beam, region_of, varies)
    ff._check_sources(  # noqa: SLF001
        field,
        model,
        model.background or Background(material),
        chi,
        varies,
        incident=incident,
    )
    tri, owner = _truncation_faces(model, support, pml)
    rows = varies[owner] & (chi[tri].max(axis=1) > 0)
    _, _, flux = _face_flux(
        model,
        tri[rows],
        owner[rows],
        beam=beam,
        frequency=frequency,
        nodes=np.asarray(nodes, float),
        region_of=region_of,
    )
    layer = CouplingLayer(
        chi=chi,
        tetrahedra=np.flatnonzero(varies),
        eps_r=float(np.real(complex(np.asarray(material.eps_r)))),
        mu_r=float(np.real(complex(np.asarray(material.mu_r)))),
        material=material,
        crossing=float(np.abs(flux).sum()),
    )
    _check_walls(
        model,
        layer=layer,
        varies=varies,
        pml=pml,
        beam=beam,
        frequency=frequency,
        nodes=nodes,
    )
    return layer


def _truncation_faces(
    model: Model, support: np.ndarray, pml: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The truncation's faces: ``(nodes (n, 3), physical tetrahedron (n,))``.

    `gfem.Absorbing` faces and the faces between the physical region and
    the perfectly matched layer.
    """
    faces = _Faces.of(model)
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    names = {
        name
        for item in model.boundaries
        if isinstance(item, Absorbing)
        for name in surfaces_of(item)
    }
    physical = support & ~pml
    rows, owners = [], []
    for row, (s, t) in enumerate(zip(a, b, strict=True)):
        if s >= 0 and t >= 0 and physical[s] != physical[t] and (pml[s] or pml[t]):
            rows.append(row)
            owners.append(s if physical[s] else t)
        elif (
            s >= 0 and t < 0 and physical[s] and names.intersection(faces.surfaces[row])
        ):
            rows.append(row)
            owners.append(s)
    return faces.nodes[np.array(rows, int)].reshape(-1, 3), np.array(owners, int)


def _face_flux(
    model: Model,
    tri: np.ndarray,
    owner: np.ndarray,
    *,
    beam: GaussianBeam | IncidentField,
    frequency: float,
    nodes: np.ndarray,
    region_of: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(inward normals, centroids, power in)`` of truncation faces, W each.

    The beam's Poynting flux into the domain at each face's centroid times
    its area, in the medium of the physical element behind it (0 behind a
    material that is not one constant number pair).
    """
    tets = np.asarray(model.mesh.tetrahedra)
    x = nodes[tri]
    normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    area = np.linalg.norm(normal, axis=1) / 2
    normal = normal / (2 * area[:, None])
    centroid = x.mean(axis=1)
    inward = nodes[tets[owner]].mean(axis=1) - centroid
    normal = np.where((np.sum(inward * normal, axis=1) < 0)[:, None], -normal, normal)
    flux = np.zeros(len(tri))
    omega = 2 * math.pi * frequency
    for region in np.unique(region_of[owner]):
        material = model.materials.get(region)
        if not isinstance(material, Material) or _is_varying(material):
            continue
        rows = region_of[owner] == region
        e, curl = host_field(beam, material, centroid[rows], frequency)
        mu = const.mu_0 * float(np.real(complex(np.asarray(material.mu_r))))
        h = curl / (-1j * omega * mu)
        poynting = 0.5 * np.real(np.cross(e, np.conj(h)))
        flux[rows] = area[rows] * np.sum(poynting * normal[rows], axis=1)
    return normal, centroid, flux


def _entrance(
    model: Model,
    *,
    support: np.ndarray,
    pml: np.ndarray,
    beam: GaussianBeam | IncidentField,
    frequency: float,
    nodes: np.ndarray,
    region_of: np.ndarray,
) -> np.ndarray:
    """The nodes of the truncation face the beam enters through (most power in).

    The truncation's faces are grouped into planes; the beam's Poynting flux
    into the domain is summed over each plane's faces (at their centroids,
    in the medium of the physical element behind each), and the plane
    taking the most power is the entrance.
    """
    tri, owner = _truncation_faces(model, support, pml)
    nodes = np.asarray(nodes, float)
    normal, centroid, flux = _face_flux(
        model,
        tri,
        owner,
        beam=beam,
        frequency=frequency,
        nodes=nodes,
        region_of=region_of,
    )
    extent = float(np.ptp(nodes, axis=0).max())
    keys = np.round(
        np.column_stack([normal, np.sum(normal * centroid, axis=1) / extent]) / _PLANE
    ).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.reshape(-1)
    power = np.bincount(group, weights=flux)
    best = int(np.argmax(power))
    if not power[best] > 0:
        raise ValueError(
            f"The beam {beam.name!r} carries no power into the model through its "
            "truncation (PML layer or Absorbing faces): describe it travelling "
            "toward the device, as the fiber launches it, or name the side."
        )
    cut = np.zeros(len(nodes), bool)
    cut[np.unique(tri[group == best])] = True
    return cut


def _is_varying(material: Any) -> bool:
    """A material that is not one constant isotropic number pair."""
    from gfem._materials import _is_law

    if isinstance(material, MaterialField) or _is_law(material):
        return True
    return any(np.ndim(getattr(material, k)) for k in ("eps_r", "mu_r"))


def _medium(
    model: Model,
    beam: GaussianBeam | IncidentField,
    region_of: np.ndarray,
    varies: np.ndarray,
) -> Material:
    """The layer's one constant, isotropic, lossless material: the beam's medium."""
    medium: Material | None = None
    for region in sorted(set(region_of[varies])):
        material = model.materials.get(region)
        lossless = isinstance(material, Material) and not (
            np.any(material.tan_delta) or np.any(material.conductivity)
        )
        if (
            not isinstance(material, Material)
            or _is_varying(material)
            or not lossless
            or (medium is not None and not _same_material(material, medium))
        ):
            first = "" if medium is None else f" (the layer also holds {medium!r})"
            raise ValueError(
                f"beam_coupling's layer just outside the side holds region "
                f"{region!r}, of material {material!r}{first}: the layer must be "
                f"one constant, isotropic, lossless medium, the medium the beam "
                f"{beam.name!r} travels in. Put the side's surface in that "
                "medium, clear of the device."
            )
        medium = material
    if medium is None:  # pragma: no cover - the layer is never empty here
        raise ValueError("beam_coupling's layer is empty.")
    return medium


def _check_walls(
    model: Model,
    *,
    layer: CouplingLayer,
    varies: np.ndarray,
    pml: np.ndarray,
    beam: GaussianBeam | IncidentField,
    frequency: float,
    nodes: np.ndarray,
) -> None:
    r"""Raise for a boundary face of the layer the reaction cannot end on.

    An `gfem.Absorbing` face is the truncation: the layer is open there. A
    PEC (PMC) wall adds no term when the beam's tangential :math:`E`
    (:math:`\nabla\times E`) vanishes on it, to `CONTACT` of its peak in
    the layer.
    """
    faces = _Faces.of(model)
    a, b = faces.tets[:, 0], faces.tets[:, 1]
    owner = np.maximum(a, 0)
    rows = (b < 0) & (a >= 0) & varies[owner] & ~pml[owner]
    rows &= layer.chi[faces.nodes].max(axis=1) > 0
    if not rows.any():
        return
    kinds: dict[str, str] = {}
    for item in (*model.boundaries, *model.ports):
        for name in surfaces_of(item):
            kinds.setdefault(name, type(item).__name__)
    nodes = np.asarray(nodes, float)
    tets = np.asarray(model.mesh.tetrahedra)[layer.tetrahedra]
    peak_e, peak_curl = host_field(
        beam, layer.material, nodes[tets].mean(axis=1), frequency
    )
    scale = {"PEC": np.abs(peak_e).max(), "PMC": np.abs(peak_curl).max()}
    for kind in ("PEC", "PMC"):
        picked = []
        for row in np.flatnonzero(rows):
            named = [kinds[s] for s in faces.surfaces[row] if s in kinds]
            this = named[0] if named else "PMC"
            if this == "Absorbing":
                continue
            if this not in ("PEC", "PMC"):
                raise ValueError(
                    f"beam_coupling's layer meets the {this} face "
                    f"{faces.surfaces[row]!r} on the boundary of the solve domain: "
                    "the layer may end only on the truncation (an Absorbing face "
                    "or the PML) or on a PEC or PMC wall the beam satisfies. Move "
                    "the side's surface away from it."
                )
            if this == kind:
                picked.append(row)
        if not picked:
            continue
        x = nodes[faces.nodes[picked]]
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        points = np.concatenate([x.mean(axis=1), *(x[:, j] for j in range(3))])
        e, curl = host_field(beam, layer.material, points, frequency)
        value = e if kind == "PEC" else curl
        tangential = np.cross(np.tile(n, (4, 1)), value)
        ratio = float(np.abs(tangential).max() / max(scale[kind], 1e-300))
        if ratio > CONTACT:
            what = "electric field" if kind == "PEC" else "magnetic field"
            raise ValueError(
                f"beam_coupling's layer meets a {kind} wall of the solve domain "
                f"where the beam {beam.name!r} has a tangential {what} "
                f"{ratio:.3g} of its peak in the layer, above {CONTACT:.0e}: the "
                "wall would reflect it. Move the side's surface off the wall, or "
                "give the beam the polarization the wall admits."
            )


# --- the reaction -------------------------------------------------------------------


def frozen_reference(
    field: Field,
    model: Model,
    *,
    layer: CouplingLayer,
    beam: IncidentField,
    frequency: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The reference values of an `gfem.IncidentField`'s interpolant on the layer.

    Its moment interpolant on the layer's elements of the field's space, at
    the imported node coordinates, read at the layer's quadrature points
    and pulled back to the reference element (`gfem._engine.far_field.
    reference_values`): the frozen test field.
    """
    from gfem._engine.moments import field_interpolant
    from gfem._probe import layout

    mesh = model.mesh
    elements = np.zeros(len(np.asarray(mesh.tetrahedra)), bool)
    elements[layer.tetrahedra] = True
    comm = field._communicator  # noqa: SLF001
    interpolant = field_interpolant(
        field.mesh,
        field.support,
        field._excluded,  # noqa: SLF001
        field.order,
        elements=elements,
        comm=comm,
    )
    values = _incident.values(beam, layer.background, interpolant.points, frequency)
    coefficients = interpolant(values)
    if comm is not None:  # a rank's piece, padded like the field's
        piece = np.zeros(np.shape(field.coefficients)[0], complex)
        piece[: len(coefficients)] = coefficients
        coefficients = piece
    imported = np.asarray(mesh.imported_nodes, float)
    at_import = Mesh(imported, mesh._topology)  # noqa: SLF001
    bary, _ = ff.rule(order)
    located = ff.located(layer, bary)  # ty: ignore[invalid-argument-type]
    prb = probe.probe(at_import, layout(field), located, comm)
    w = prb.values(coefficients)[0]
    curl = prb.values(coefficients, curl=True)[0]
    corners = imported[np.asarray(mesh.tetrahedra)[layer.tetrahedra]]
    w_ref, c_ref = ff.reference_values(corners, w, curl)
    return np.asarray(w_ref), np.asarray(c_ref)


def moved_layer(
    layer: CouplingLayer, model: Model, nodes: np.ndarray
) -> tuple[int, float]:
    """How many of the layer's nodes moved from their imported position, and how far."""
    mesh = model.mesh
    ids = np.unique(np.asarray(mesh.tetrahedra)[layer.tetrahedra])
    imported = np.asarray(mesh.imported_nodes, float)[ids]
    shift = np.linalg.norm(np.asarray(nodes, float)[ids] - imported, axis=1)
    extent = float(np.ptp(np.asarray(mesh.imported_nodes, float), axis=0).max())
    moved = shift > 1e-12 * extent
    return int(moved.sum()), float(shift.max(initial=0.0))


def reaction(
    corners: Any,
    e_ref: Any,
    c_ref: Any,
    *,
    chi: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    test: Callable[[Any, Any], tuple[Any, Any]],
) -> tuple[Any, Any]:
    r"""``(R, F)``: the reaction, and the beam's flux form.

    :math:`F = \int_L\nabla\chi\cdot(w\times\overline{\nabla\times w})\,dV`.

    A `jax.numpy` function of the layer's corners ``(n_tets, 4, 3)`` (m) and
    the field's reference values; ``test(points, jacobians)`` gives the test
    field and its curl at the ``(n_tets, n_points, 3)`` quadrature points.
    """
    jac = ff._jacobians(corners)  # noqa: SLF001
    det = jnp.linalg.det(jac)
    inv_t = jnp.linalg.inv(jac).transpose(0, 2, 1)
    e = jnp.einsum("tij,tpj->tpi", inv_t, e_ref)
    curl = jnp.einsum("tij,tpj->tpi", jac, c_ref) / det[:, None, None]
    d_chi = jnp.asarray(chi[:, :3] - chi[:, 3:])
    grad = jnp.einsum("tij,tj->ti", inv_t, d_chi)[:, None, :]
    points = corners[:, 3][:, None] + jnp.einsum("tij,pj->tpi", jac, bary[:, :3])
    dv = (jnp.abs(det)[:, None] / 6 * weights[None])[..., None]
    w, w_curl = test(points, jac)
    r = jnp.sum(dv * (e * jnp.cross(grad, w_curl) + curl * jnp.cross(grad, w)))
    flux = jnp.sum(dv * grad * jnp.cross(w, jnp.conj(w_curl)))
    return r, flux


def gaussian_test(
    beam: GaussianBeam, layer: CouplingLayer, frequency: float
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """The closed-form beam at the (moving) quadrature points."""
    background = layer.background

    def test(points: Any, jac: Any) -> tuple[Any, Any]:
        del jac
        shape = points.shape
        e, curl = _beam.field_and_curl(
            beam, background, points.reshape(-1, 3), frequency, jnp
        )
        return e.reshape(shape), curl.reshape(shape)

    return test


def frozen_test(
    w_ref: np.ndarray, c_ref: np.ndarray
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """The frozen interpolant, carried with the nodes covariantly."""

    def test(points: Any, jac: Any) -> tuple[Any, Any]:
        del points
        det = jnp.linalg.det(jac)
        inv_t = jnp.linalg.inv(jac).transpose(0, 2, 1)
        w = jnp.einsum("tij,tpj->tpi", inv_t, w_ref)
        curl = jnp.einsum("tij,tpj->tpi", jac, c_ref) / det[:, None, None]
        return w, curl

    return test


def coupling(r: Any, flux: Any, omega: float, mu_r: float) -> tuple[Any, Any]:
    r"""``(c, P)``: the coupling :math:`R/(4i\omega\mu\sqrt P)` and the beam's power."""
    mu = const.mu_0 * mu_r
    power = jnp.imag(flux) / (2 * omega * mu)
    return r / (4j * omega * mu * jnp.sqrt(power)), power
