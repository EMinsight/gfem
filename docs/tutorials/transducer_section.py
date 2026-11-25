"""The cross-section of the transducer tutorial, built with netgen.

An interdigital transducer on a thin-film lithium niobate stack, in the plane
perpendicular to its fingers: a 360 nm x-cut film etched 180 nm deep into two
2 um ribs, on 4.7 um of buried oxide over a silicon handle, with 50 nm
aluminium fingers at a 2.2 um pitch. `transducer_section` meshes it at a given
size, writes the mesh as a gmsh 2.2 file with its region and line names, and
reads it back through `gfem.Mesh.from_gmsh`, as a mesh of your own would be.
Netgen meshes on one thread here, so a section meshes the same way on every
run and the page's numbers reproduce.

The section in the (x, y) plane, y up, y = 0 at the top of the film's slab, in um:

- ``"ln"``: the film, -0.18 < y < 0, and the two ribs on it, ``RIB_WIDTH`` wide
  at the top, with sidewalls leaning at ``RIB_ANGLE``, 0 < y < 0.18;
- ``"box"``: buried oxide below the film, ``"si"``: the top 2 um of the handle;
- ``"signal"`` and ``"ground"``: the fingers, ``FINGER`` wide (a metallization
  ratio of one half), ``T_AL`` thick, alternating along x (areas; with
  ``electrodes="lines"``, lines on the film's surface instead);
- ``"air"``: 2.5 um above the film, in the electrostatic domain only.

The fingers sit at 2.2 um pitch, signal first. ``outer`` fingers stand beyond
each rib and ``between`` between the two ribs; two fingers are left out
beside each rib, so each rib lies in a 5.5 um gap. The first finger is the
signal, so the section has 6 signal and 5 ground fingers at the default
``outer=4, between=5``.

Around that interior, a frame region of every layer on each side absorbs the
waves that leave it: ``"ln_frame"``, ``"box_frame"`` and ``"si_frame"`` (the
handle's frame also holds its bottom 8 um). The lateral frames are 17.6 um
wide, four wavelengths of the shear plate wave. The handle's bottom is the line
``"bottom"``.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen import occ

import gfem

UM = 1e-6
#: The stack and the fingers (um, degrees).
SLAB, RIB_HEIGHT, RIB_WIDTH, RIB_ANGLE = 0.18, 0.18, 2.0, 70.0
BOX, SI, SI_FRAME, AIR = 4.7, 10.0, 8.0, 2.5
PITCH, FINGER, T_AL = 2.2, 1.1, 0.05
FRAME = 17.6  # the lateral frames' width: four wavelengths of the plate wave
GAP = 2.2  # um between the first finger and the lateral frame
#: Edge lengths at ``edge_scale = 1`` (um): oxide and silicon, film, air, aluminium.
H_BOX, H_FILM, H_AIR, H_AL = 0.5, 0.06, 0.6, 0.03


@dataclasses.dataclass(frozen=True)
class TransducerSection:
    """The section's mesh and the points and numbers the tutorial reads from it.

    Attributes:
        mesh: The 2D mesh (coordinates in m, regions and lines named as in the
            module docstring).
        arms: The ribs' centres at the film's mid-depth, shape ``(n_ribs, 2)``,
            m; empty without ribs.
        interior: ``(x_a, x_b)``: the span between the lateral frames, m.
        fingers: ``(n_signal, n_ground)``, the numbers of signal and ground
            fingers.
        pitch: The finger pitch, m.
    """

    mesh: gfem.Mesh
    arms: np.ndarray
    interior: tuple[float, float]
    fingers: tuple[int, int]
    pitch: float


def transducer_section(
    *,
    outer: int = 4,
    between: int = 5,
    ribs: bool = True,
    electrodes: str = "meshed",
    edge_scale: float = 1.5,
    directory: str | Path | None = None,
) -> TransducerSection:
    """Mesh the transducer's section.

    Writes ``transducer.msh`` in ``directory`` (overwritten on each call;
    ``None``: a temporary directory, removed once the mesh is read) and reads
    it back.

    Args:
        outer: Fingers beyond each rib.
        between: Fingers between the two ribs.
        ribs: Whether the film carries the two ribs.
        electrodes: ``"meshed"`` for 50 nm aluminium fingers (the areas
            ``"signal"`` and ``"ground"``), ``"lines"`` for zero-thickness
            electrodes (the lines ``"signal"`` and ``"ground"`` on the film).
        edge_scale: Multiplies every edge length: 1 is the converged mesh, 2 a
            quick, coarser one.
        directory: Where to write the file.
    """
    if electrodes not in ("meshed", "lines"):
        raise ValueError(f"electrodes is 'meshed' or 'lines', got {electrodes!r}.")
    if outer < 1 or between < 1:
        raise ValueError("outer and between must be at least 1.")
    if directory is None:
        with tempfile.TemporaryDirectory() as temporary:
            return transducer_section(
                outer=outer,
                between=between,
                ribs=ribs,
                electrodes=electrodes,
                edge_scale=edge_scale,
                directory=temporary,
            )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # finger k of 0..n-1 at x0 + k p, two left out beside each rib
    a1, a2 = outer, outer + between
    n = a2 + 2 + outer
    left_out = {a1, a1 + 1, a2, a2 + 1}
    x0 = FRAME + GAP
    fingers = [
        (x0 + k * PITCH, "signal" if k % 2 == 0 else "ground")
        for k in range(n)
        if k not in left_out
    ]
    rib_centres = [x0 + (a1 + 0.5) * PITCH, x0 + (a2 + 0.5) * PITCH] if ribs else []
    length = x0 + (n - 1) * PITCH + PITCH / 2 + FRAME
    s = edge_scale
    mesh = _mesh(
        directory / "transducer.msh",
        length,
        fingers,
        rib_centres,
        electrodes,
        h_box=H_BOX * s,
        h_film=H_FILM * s,
        h_air=H_AIR * s,
        h_al=H_AL * s,
    )
    arms = np.array([(x, -SLAB / 2) for x in rib_centres]).reshape(-1, 2) * UM
    n_signal = sum(1 for _, k in fingers if k == "signal")
    return TransducerSection(
        mesh=mesh,
        arms=arms,
        interior=(FRAME * UM, (length - FRAME) * UM),
        fingers=(n_signal, len(fingers) - n_signal),
        pitch=PITCH * UM,
    )


def _rect(x0, y0, x1, y1, name, *, maxh=None):
    face = occ.WorkPlane().MoveTo(x0, y0).Rectangle(x1 - x0, y1 - y0).Face()
    face.name = name
    if maxh:
        face.maxh = maxh
    return face


def _mesh(
    path, length, fingers, rib_centres, electrodes, *, h_box, h_film, h_air, h_al
):
    """Glue the section's faces, mesh them with netgen and read the file back."""
    y_box = -SLAB - BOX
    y_si = y_box - SI
    y_cut = y_box - (SI - SI_FRAME)  # the handle's interior ends here
    xa, xb = FRAME, length - FRAME
    faces = []
    # the handle: interior, lateral frames, bottom frame
    faces += [
        _rect(xa, y_cut, xb, y_box, "si", maxh=h_box),
        _rect(0, y_cut, xa, y_box, "si_frame", maxh=h_box),
        _rect(xb, y_cut, length, y_box, "si_frame", maxh=h_box),
        _rect(0, y_si, length, y_cut, "si_frame", maxh=h_box),
    ]
    # the oxide and the film: interior and lateral frames
    for lo, hi, name, size in (
        (y_box, -SLAB, "box", h_box),
        (-SLAB, 0.0, "ln", h_film),
    ):
        faces += [
            _rect(xa, lo, xb, hi, name, maxh=size),
            _rect(0, lo, xa, hi, name + "_frame", maxh=size),
            _rect(xb, lo, length, hi, name + "_frame", maxh=size),
        ]
    foot = RIB_HEIGHT / np.tan(np.deg2rad(RIB_ANGLE))
    rib_faces = []
    for xc in rib_centres:
        half, base = RIB_WIDTH / 2, RIB_WIDTH / 2 + foot
        trapezoid = (
            occ.WorkPlane()
            .MoveTo(xc - base, 0)
            .LineTo(xc + base, 0)
            .LineTo(xc + half, RIB_HEIGHT)
            .LineTo(xc - half, RIB_HEIGHT)
            .Close()
            .Face()
        )
        trapezoid.name, trapezoid.maxh = "ln", h_film
        rib_faces.append(trapezoid)
    air = _rect(0, 0, length, AIR, "air", maxh=h_air)
    extra = []
    if electrodes == "meshed":
        metal = [
            _rect(x - FINGER / 2, 0, x + FINGER / 2, T_AL, name, maxh=h_al)
            for x, name in fingers
        ]
        for face in metal + rib_faces:
            air = air - face
        extra = metal
    else:
        for face in rib_faces:
            air = air - face
        for x, name in fingers:
            line = occ.Segment(
                occ.Pnt(x - FINGER / 2, 0, 0), occ.Pnt(x + FINGER / 2, 0, 0)
            )
            line.name, line.maxh = name, h_al
            extra.append(line)
    air.name = "air"
    shape = occ.Glue(faces + rib_faces + [air] + extra)
    for edge in shape.edges:
        if abs(edge.center.y - y_si) < 1e-9:
            edge.name = "bottom"
    ngmesh = ngs.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_box))
    nodes = np.array([v.point for v in ngmesh.vertices])
    areas = [(el.mat, [v.nr for v in el.vertices]) for el in ngmesh.Elements(ngs.VOL)]
    named = {"bottom"} | ({"signal", "ground"} if electrodes == "lines" else set())
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in ngmesh.Elements(ngs.BND)
        if el.mat in named
    ]
    _write_gmsh(path, nodes, areas, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _write_gmsh(path: Path, nodes: np.ndarray, tris: list, lines: list) -> None:
    """Triangles and named lines as gmsh 2.2, with ``$PhysicalNames``."""
    tags: dict[tuple[int, str], int] = {}
    for dim, items in ((1, lines), (2, tris)):
        for name, _ in items:
            tags.setdefault((dim, name), len(tags) + 1)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    out.append(str(len(tags)))
    out += [f'{dim} {t} "{name}"' for (dim, name), t in tags.items()]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(tris))]
    k = 0
    for dim, kind, items in ((1, 1, lines), (2, 2, tris)):
        for name, vertices in items:
            k += 1
            t = tags[dim, name]
            out.append(
                f"{k} {kind} 2 {t} {t} " + " ".join(str(v + 1) for v in vertices)
            )
    out.append("$EndElements")
    path.write_text("\n".join(out) + "\n")
