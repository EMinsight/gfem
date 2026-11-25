r"""The force and torque a solved field exerts on a body: the host kernel.

The host kernel of `gfem.optical_force`. The time-averaged Lagrangian
:math:`L = \frac{1}{4}\int (\varepsilon |E|^2 - |B|^2/\mu)\,dV` of the
discrete field, at fixed coefficients, moves with the nodes of the mesh. Its
derivative under a rigid motion of the body's nodes is the force and torque,
in the volume form of the Maxwell stress tensor: the body's own elements
drop out (the gradient of the motion vanishes there), so only the stress
shell contributes, the solve-domain elements outside the body that share a
vertex with it. The shell holds one lossless, homogeneous, isotropic medium;
the sensitivity kernel's node pass on it gives the nodal derivative
:math:`g_n` of :math:`\mathrm{Re}\, \bar e^\mathsf{T} A_\omega e`, and

.. math::

    G_n = -\frac{g_n}{4 \mu_0 \omega^2 s},\quad
    F = \sum_{n \in B} G_n,\quad
    \tau_c = \sum_{n \in B} (x_n - c) \times G_n,

with :math:`B` the body's nodes, :math:`c` its volume centroid and :math:`s`
the file's length unit in m (the field's coefficients are in volts per file
unit, and the factor is :math:`\varepsilon_0 s / 4` over :math:`k_0^2`).

The result is the exact derivative of the discrete Lagrangian, not a
residual: it converges with the mesh to the physical force.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem import _pml
from gfem._boundaries import PEC, PMC, Absorbing, Periodic, surfaces_of
from gfem._engine import probe
from gfem._engine.domain import em_region
from gfem._engine.driven import DrivenOperator, driven_operator, operator_terms
from gfem._engine.power_flow import check_nodes, check_size, check_support
from gfem._engine.sensitivity import Term, _node_pass
from gfem._engine.sides import SideSpec, selection
from gfem._engine.terms import _bits
from gfem._materials import Material, _is_law
from gfem._periodic import tie_classes

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "WALL_TOLERANCE",
    "Shell",
    "Stress",
    "build",
    "evaluate",
    "force_torque",
    "shell",
]

#: A body node sits on a wall as the mesh writes it, to round-off: a rigid
#: motion is tangent to the wall when the normal component of a unit
#: motion is below this.
WALL_TOLERANCE = 1e-12

#: The vertex triples of the four faces of a tetrahedron.
_TET_FACES = np.array(list(itertools.combinations(range(4), 3)))


@dataclasses.dataclass(frozen=True)
class Shell:
    """The topology of a body and its stress shell.

    Attributes:
        body: The selected tetrahedra, one boolean each (excluded ones too).
        nodes: The body's nodes, sorted.
        shell: The solve-domain tetrahedra outside the body that share a
            vertex with it.
        centroid: The body's volume centroid, m, ``(3,)``.
        valid: Which of the six rigid motions (the translations along x, y,
            z, then the rotations about the axes through the centroid) keep
            every body node on the walls it touches without owning; the
            others slide along a wall, and their force is NaN by definition.
    """

    body: np.ndarray
    nodes: np.ndarray
    shell: np.ndarray
    centroid: np.ndarray
    valid: np.ndarray


@dataclasses.dataclass(frozen=True)
class Stress:
    """The restricted operator and terms of a shell, ready for the node pass.

    Attributes:
        op: The driven operator on the shell's elements, without faces.
        terms: Its volume terms, restricted to the shell.
        scale: The length unit of the file in m.
    """

    op: DrivenOperator
    terms: list[Term]
    scale: float


def shell(field: Field, body: SideSpec, model: Model) -> Shell:
    """The body, its stress shell and its walls, from the mesh topology.

    A predicate body reads the node coordinates too.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The body or the field does not fit, or the shell is not
            one homogeneous lossless isotropic medium away from ports,
            absorbing surfaces, layers and periodic pairs (see
            `gfem.optical_force`).
    """
    mesh = model.mesh
    check_nodes(field, model)
    support = check_support(field, model)
    selected = selection(
        mesh, body, subject="The body", refuse_surfaces="optical_force"
    )
    if not selected.any():
        raise ValueError("The body selects no element: name a volume region.")
    tets = np.asarray(mesh.tetrahedra)
    nodes = np.unique(tets[selected])
    touching = np.isin(tets, nodes).any(axis=1)
    around = support & ~selected & touching
    if not around.any():
        raise ValueError(
            "The body has no stress shell: no element of the solve domain "
            "outside it shares a vertex with it. Select a body that sits "
            "inside the solve domain."
        )
    _check_incident(field)
    _check_medium(model, around)
    _check_periodic(model, nodes)
    _check_surfaces(model, nodes, around, tets)
    _check_point_sources(field, around)
    x = np.asarray(mesh.nodes)
    centroid = _centroid(x, tets[selected])
    valid = _valid_motions(
        model,
        support=support,
        selected=selected,
        nodes=nodes,
        tets=tets,
        touching=touching,
        x=x,
        centroid=centroid,
    )
    return Shell(selected, nodes, around, centroid, valid)


def _check_incident(field: Field) -> None:
    """Raise for the incident or scattered part of an incident-field column."""
    part = field._incident  # noqa: SLF001
    if (part is not None and part.is_incident) or field._scattered:  # noqa: SLF001
        which = "incident" if part is not None else "scattered"
        raise ValueError(
            f"optical_force reads the total field, got the {which} part of an "
            "incident-field column: the force is the stress of the total "
            "field. Pass the field of result.e_field(f, source), which is the "
            "total field."
        )


def _centroid(x: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """The volume centroid of tetrahedra, m."""
    corners = x[cells]
    volume = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))  # 6 x volume
    return (volume[:, None] * corners.mean(axis=1)).sum(axis=0) / volume.sum()


def _check_medium(model: Model, around: np.ndarray) -> None:
    """Raise unless the shell is one lossless isotropic homogeneous medium."""
    mesh = model.mesh
    seen: dict[tuple[float, float], str] = {}
    for name in mesh.volumes:
        if name not in model.materials or not around[mesh.elements(name)].any():
            continue
        material = model.materials[name]
        why = _not_plain(material)
        if why is not None:
            raise ValueError(
                f"The stress shell around the body holds region {name!r}, "
                f"{why}: the force is read in a homogeneous, isotropic, lossless "
                "medium (a plain gfem.Material with real eps_r and mu_r, no "
                "conductivity or loss tangent). Surround the body with one."
            )
        assert isinstance(material, Material)  # noqa: S101 - _not_plain
        seen.setdefault((float(material.eps_r), float(material.mu_r)), name)  # ty: ignore[invalid-argument-type]
    if len(seen) > 1:
        (a, first), (b, second) = list(seen.items())[:2]
        raise ValueError(
            "The stress shell around the body holds two media: region "
            f"{first!r} (eps_r, mu_r = {a}) and region {second!r} ({b}). The "
            "force is read in one homogeneous medium: keep the body's "
            "neighbourhood in one region."
        )


def _not_plain(material: object) -> str | None:
    """Why ``material`` is not a plain lossless isotropic one, ``None`` if it is."""
    if not isinstance(material, Material):
        return "a gfem.MaterialField (a design)"
    if _is_law(material.eps_r):
        return "a dispersive material"
    if material.axes is not None or not material.isotropic:
        return "an anisotropic material"
    if material.conductivity != 0 or material.tan_delta != 0:
        return "a lossy material"
    return None


def _surface_triangles(model: Model, name: str) -> np.ndarray:
    return np.asarray(model.mesh.triangles)[model.mesh.elements(name)]


def _check_surfaces(
    model: Model, nodes: np.ndarray, around: np.ndarray, tets: np.ndarray
) -> None:
    """Raise for a sheet, port, absorbing wall or layer the force would misread."""
    n_nodes = len(model.mesh.nodes)
    on_body = np.zeros(n_nodes, bool)
    on_body[nodes] = True
    on_shell = on_body.copy()
    on_shell[tets[around]] = True
    for boundary in model.boundaries:
        if getattr(boundary, "_physics", "em") != "em" or isinstance(
            boundary, PEC | PMC | Periodic
        ):
            continue
        for name in surfaces_of(boundary):
            faces = _surface_triangles(model, name)
            if isinstance(boundary, Absorbing) and on_shell[faces].any():
                raise ValueError(
                    f"The body's stress shell reaches the absorbing surface "
                    f"{name!r}: its radiation condition is not a medium. Move "
                    "the body at least one element away from it."
                )
            partial = on_body[faces].any(axis=1) & ~on_body[faces].all(axis=1)
            if partial.any():
                raise ValueError(
                    f"Surface {name!r} ({type(boundary).__name__}) touches the "
                    "body without lying on it: its terms would move with some "
                    "nodes and not others. Keep the sheet off the body, or "
                    "wholly on it."
                )
    for port in model.ports:
        for name in surfaces_of(port):
            if on_shell[_surface_triangles(model, name)].any():
                raise ValueError(
                    f"The body's stress shell reaches port {port.name!r}'s "
                    f"surface {name!r}: the port's load would sit in the "
                    "shell. Move the body at least one element away from it."
                )
    if model.pml is not None:
        layer = np.zeros(len(tets), bool)
        for name in model.pml.regions:
            layer[model.mesh.elements(name)] = True
        on_layer = np.zeros(n_nodes, bool)
        on_layer[tets[layer]] = True
        if (on_layer & on_shell).any():
            raise ValueError(
                "The body's stress shell reaches the perfectly matched layer: "
                "the stretch there is not a medium. Move the body at least one "
                "element away from the layer."
            )


def _check_periodic(model: Model, nodes: np.ndarray) -> None:
    """Raise for a body node on a periodic pair, where its images move too."""
    tie = tie_classes(len(model.mesh.nodes), model._periodic)  # noqa: SLF001
    if (np.bincount(tie)[tie[nodes]] > 1).any():
        raise ValueError(
            "The body has a node on a gfem.Periodic pair's surface, where its "
            "images across the pair would have to move with it: keep the body "
            "off the periodic walls."
        )


def _check_point_sources(field: Field, around: np.ndarray) -> None:
    """Raise if a point dipole whose load solved the field sits in the shell."""
    sources = getattr(field, "_point_sources", ())
    if not sources:
        return
    from gfem._probe import layout as probe_layout

    points = np.array([position for _, position in sources], float)
    located = probe.locate(field.mesh, probe_layout(field), points)
    for (name, _), tet in zip(sources, located.tetrahedra, strict=True):
        if around[tet]:
            raise ValueError(
                f"The body's stress shell holds the dipole {name!r} whose load "
                "solved the field: the force would read its source. Move the "
                "dipole at least one element away from the body."
            )


def _valid_motions(
    model: Model,
    *,
    support: np.ndarray,
    selected: np.ndarray,
    nodes: np.ndarray,
    tets: np.ndarray,
    touching: np.ndarray,
    x: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Which rigid motions keep the body on the walls it touches, ``(6,)`` bool.

    A wall is a face on the boundary of the solve domain (its natural
    perfect magnetic conductor, or a `gfem.PEC` surface of an excluded
    region) or on a named `gfem.PEC` or `gfem.PMC` surface, with a vertex on
    the body, that does not move with it. A face moves with the body when
    every vertex is on it and either solve-domain elements lie on both sides
    (a sheet on the body) or an excluded element of the body lies behind it.
    """
    near = np.flatnonzero(touching)
    keys = np.sort(tets[near][:, _TET_FACES].reshape(-1, 3), axis=1)
    owner = np.repeat(near, 4)
    faces, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    count = len(faces)
    in_domain = np.bincount(inverse, weights=support[owner], minlength=count)
    behind = np.bincount(
        inverse, weights=selected[owner] & ~support[owner], minlength=count
    )
    wall = in_domain == 1
    for boundary in model.boundaries:
        if isinstance(boundary, PEC | PMC):
            named = np.sort(_surface_triangles(model, boundary.surface), axis=1)
            wall |= _isin_rows(faces, named)
    on_body = np.zeros(len(x), bool)
    on_body[nodes] = True
    held = on_body[faces]
    wall &= held.any(axis=1)
    wall &= ~(held.all(axis=1) & ((in_domain == 2) | (behind > 0)))
    valid = np.ones(6, bool)
    touched = faces[wall]
    if not len(touched):
        return valid
    normal = np.cross(
        x[touched[:, 1]] - x[touched[:, 0]], x[touched[:, 2]] - x[touched[:, 0]]
    )
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    length = max(float(np.linalg.norm(x[nodes] - centroid, axis=1).max()), 1e-300)
    for face, n in zip(touched, normal, strict=True):
        arm = (x[face] - centroid)[on_body[face]]
        for axis in range(3):
            if abs(n[axis]) > WALL_TOLERANCE:
                valid[axis] = False
            spin = np.cross(np.eye(3)[axis], arm) @ n
            if np.abs(spin).max() > WALL_TOLERANCE * length:
                valid[3 + axis] = False
    return valid


def _isin_rows(rows: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Which rows of ``rows`` are rows of ``wanted`` (both sorted per row)."""
    if not len(wanted):
        return np.zeros(len(rows), bool)
    _, label = np.unique(np.concatenate([rows, wanted]), axis=0, return_inverse=True)
    label = label.ravel()
    return np.isin(label[: len(rows)], label[len(rows) :])


def build(
    field: Field, model: Model, where: Shell, *, stretch: _pml.Stretch | None = None
) -> Stress:
    """The operator and terms of ``where``'s shell, at ``model``'s node positions.

    ``stretch`` is the field's perfectly matched layer stretch
    (`gfem._pml.field_stretch`).
    """
    op = driven_operator(
        model,
        field.order,
        elements=where.shell,
        faces=np.zeros(len(model.mesh.triangles), bool),
        paired=False,
        stretch=stretch,
    )
    bits = _bits(where.shell, op.built.cell_order, "tetrahedron", part=False)
    terms = [
        dataclasses.replace(t, elements=bits)
        for t in operator_terms(model, op, nodes=False)
        if t.kind in ("value", "curl")
    ]
    return Stress(op=op, terms=terms, scale=float(op.built.scale))


def evaluate(
    stress: Stress,
    model: Model,
    where: Shell,
    e: np.ndarray,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The force and the torque about the centroid at the coefficients ``e``.

    Args:
        stress: The shell's operator (`build`).
        model: The model, at the node positions ``stress`` was built on.
        where: The body and its shell.
        e: The field's coefficients on the operator's space.
        omega: The angular frequency, rad/s (the real part of the field's).

    Returns:
        The force ``(3,)``, N, and the torque about ``where.centroid``
        ``(3,)``, N m, before the wall rule (`Shell.valid`).
    """
    op = stress.op
    weights = {p.name: complex(p.theta(omega)) for p in op.pieces if p.kind != "port"}
    pair = op.pair(np.conj(e), e, weights=weights, domain=em_region(model, op.built))
    g = _node_pass(model, [pair], stress.terms)
    nodal = -g[where.nodes] / (4 * const.mu_0 * omega**2 * stress.scale)
    arm = np.asarray(model.mesh.nodes)[where.nodes] - where.centroid
    return nodal.sum(axis=0), np.cross(arm, nodal).sum(axis=0)


def force_torque(
    field: Field, body: SideSpec, model: Model
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The force, the torque about the centroid, and the centroid of a body.

    The components that slide along a wall are NaN (`Shell.valid`).

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The body or the field does not fit (see
            `gfem.optical_force`).
    """
    where = shell(field, body, model)
    stress = build(field, model, where, stretch=_pml.field_stretch(model, field))
    e = np.asarray(field.coefficients)
    check_size(e, stress.op, field.order)
    omega = float(np.asarray(field.omega).real)
    force, torque = evaluate(stress, model, where, e, omega)
    return (
        np.where(where.valid[:3], force, np.nan),
        np.where(where.valid[3:], torque, np.nan),
        where.centroid,
    )
