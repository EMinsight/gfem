"""The cross-sections of the modulator tutorial, built with netgen.

An x-cut thin-film lithium niobate push-pull modulator in a ground-signal-ground
(GSG) section: a 700 nm crystal film etched 400 nm deep into two ridges, one in
each electrode gap, on 2 um of buried oxide, under 1 um of oxide cladding and
air. `tfln_section` meshes it at a given electrode gap and ridge width, writes
each mesh as a gmsh 2.2 file with its region and edge names, and reads it back
through `gfem.Mesh.from_gmsh_cross_section`, as a mesh of your own would be.
Netgen meshes on one thread here, so a section meshes the same way on every
run and the page's numbers reproduce.

The section in the (x, y) plane, y up, y = 0 at the bottom of the film, in um:

- ``"box"``: buried oxide, -2 < y < 0 (the substrate below is left out);
- ``"slab"``: the film left after the etch, 0 < y < 0.3;
- ``"ridge_a"``, ``"ridge_b"``: the ridges, ``width`` wide, 0.3 < y < 0.7,
  centred in the right (a) and the left (b) gap;
- ``"cladding"``: oxide up to y = 1.7, then ``"air"`` up to y = 3;
- ``"signal"``: the signal electrode, |x| < 2, 0.3 < y < 0.5;
- ``"ground_r"``, ``"ground_l"``: the grounds, 4 um wide, ``gap`` away from the
  signal on either side; the section ends 3 um beyond each.

Named edges of the electrostatic mesh: ``"signal_s"`` and ``"ground_s"`` (the
electrodes' outlines), ``"ridge_a_s"`` and ``"ridge_b_s"`` (the ridges'
outlines), ``"ln_if"`` (the film's other top and bottom faces), ``"clad_if"``
(the cladding's top) and ``"bottom"`` (the box's bottom). Naming the crystal's
interfaces lets a gap move slide along them, and holding the ridges keeps their
shape.

Each optical window holds one ridge and the film, oxide and cladding around it,
out to the electrodes' inner edges (its regions: ``"ridge"``, ``"slab"``,
``"box"``, ``"cladding"``; the box is cut at y = -1.2). All three meshes share
one slab thickness, so the windows' elements lie inside the electrostatic
mesh's.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Callable
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6
SLAB, RIB = 0.3, 0.4  # film left after the etch, and the etch depth (um)
BOX, CLADDING_TOP, AIR_TOP = -2.0, 1.7, 3.0  # um
ELECTRODE_THICKNESS = 0.2  # um
SIGNAL_WIDTH, GROUND_WIDTH, MARGIN = 4.0, 4.0, 3.0  # um
WINDOW_BOTTOM = -1.2  # um
#: Edge lengths (um): everywhere, then in the film, the ridges and the electrodes.
H_MAX, H_SLAB, H_RIDGE, H_ELECTRODE = 0.4, 0.08, 0.06, 0.1
H_WINDOW = 0.3  # um, the windows' largest edge
#: The slab thickness of every mesh, m.
THICKNESS = 0.1e-6

#: A rectangle: (x0, y0, x1, y1, region, edge length or None), um.
Rect = tuple[float, float, float, float, str, float | None]


@dataclasses.dataclass(frozen=True)
class TflnSection:
    """The section's meshes at one electrode gap and ridge width.

    Attributes:
        static: The whole section, for the electrostatic model.
        window_a: The optical window of ridge a (in the right gap).
        window_b: The optical window of ridge b (in the left gap).
        gap: The electrode gap, m.
        width: The ridges' top width, m.
    """

    static: gfem.Mesh
    window_a: gfem.Mesh
    window_b: gfem.Mesh
    gap: float
    width: float

    @property
    def signal_edge(self) -> float:
        """The x of the signal electrode's right inner wall, m (mirrored at -x)."""
        return SIGNAL_WIDTH / 2 * UM

    @property
    def ground_edge(self) -> float:
        """The x of the right ground's inner wall, m (mirrored at -x)."""
        return self.signal_edge + self.gap

    @property
    def ridge_centre(self) -> float:
        """The x of ridge a's centre, m (ridge b's is at -x)."""
        return self.signal_edge + self.gap / 2


def tfln_section(
    gap: float,
    width: float,
    *,
    directory: str | Path | None = None,
    coarsen: float = 1.0,
) -> TflnSection:
    """Mesh the section at electrode gap ``gap`` and ridge width ``width`` (m).

    Writes ``static.msh``, ``window_a.msh`` and ``window_b.msh`` in
    ``directory`` (overwritten on each call; ``None``: a temporary directory,
    removed once the meshes are read) and reads them back. ``coarsen``
    multiplies every edge length: 2 gives a quick, coarser section.
    """
    if directory is None:
        with tempfile.TemporaryDirectory() as temporary:
            return tfln_section(gap, width, directory=temporary, coarsen=coarsen)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    g, w = gap / UM, width / UM
    xs = SIGNAL_WIDTH / 2
    xg = xs + g
    xa = xs + g / 2
    half = xg + GROUND_WIDTH + MARGIN
    top, electrode_top = SLAB + RIB, SLAB + ELECTRODE_THICKNESS
    h_slab, h_ridge, h_metal = (coarsen * h for h in (H_SLAB, H_RIDGE, H_ELECTRODE))
    rects: list[Rect] = [
        (-half, SLAB, half, CLADDING_TOP, "cladding", None),
        (-half, CLADDING_TOP, half, AIR_TOP, "air", None),
        (-half, BOX, half, 0.0, "box", None),
        (-half, 0.0, half, SLAB, "slab", h_slab),
        (xa - w / 2, SLAB, xa + w / 2, top, "ridge_a", h_ridge),
        (-xa - w / 2, SLAB, -xa + w / 2, top, "ridge_b", h_ridge),
        (-xs, SLAB, xs, electrode_top, "signal", h_metal),
        (xg, SLAB, xg + GROUND_WIDTH, electrode_top, "ground_r", h_metal),
        (-xg - GROUND_WIDTH, SLAB, -xg, electrode_top, "ground_l", h_metal),
    ]
    right, left = (
        _outline(xg, SLAB, xg + GROUND_WIDTH, electrode_top),
        _outline(-xg - GROUND_WIDTH, SLAB, -xg, electrode_top),
    )
    # first match names an edge: the outlines before the film's faces
    edges = [
        (_outline(-xs, SLAB, xs, electrode_top), "signal_s"),
        (lambda p: right(p) or left(p), "ground_s"),
        (_outline(xa - w / 2, SLAB, xa + w / 2, top), "ridge_a_s"),
        (_outline(-xa - w / 2, SLAB, -xa + w / 2, top), "ridge_b_s"),
        (lambda p: _at(p[1], SLAB) or _at(p[1], 0.0), "ln_if"),
        (lambda p: _at(p[1], CLADDING_TOP), "clad_if"),
        (lambda p: _at(p[1], BOX), "bottom"),
    ]
    static = _mesh(directory / "static.msh", rects, coarsen * H_MAX, edges)
    windows = [
        _mesh(
            directory / f"window_{name}.msh",
            _window(c, g / 2, w, h_slab, h_ridge),
            coarsen * H_WINDOW,
            [],
        )
        for name, c in (("a", xa), ("b", -xa))
    ]
    return TflnSection(static, *windows, gap=gap, width=width)


def _window(centre, half, w, h_slab, h_ridge) -> list[Rect]:
    """One ridge's optical window, out to the electrodes' inner edges."""
    lo, hi = centre - half, centre + half
    return [
        (lo, SLAB, hi, CLADDING_TOP, "cladding", None),
        (lo, WINDOW_BOTTOM, hi, 0.0, "box", None),
        (lo, 0.0, hi, SLAB, "slab", h_slab),
        (centre - w / 2, SLAB, centre + w / 2, SLAB + RIB, "ridge", h_ridge),
    ]


def _at(value: float, target: float) -> bool:
    return abs(value - target) < 1e-9


def _outline(x0: float, y0: float, x1: float, y1: float) -> Callable:
    """Whether a point lies on the rectangle's outline."""

    def on(p) -> bool:
        x, y = p[0], p[1]
        on_side = (_at(x, x0) or _at(x, x1)) and y0 - 1e-9 <= y <= y1 + 1e-9
        on_end = (_at(y, y0) or _at(y, y1)) and x0 - 1e-9 <= x <= x1 + 1e-9
        return on_side or on_end

    return on


def _mesh(path: Path, rects: list[Rect], maxh: float, edges: list) -> gfem.Mesh:
    """Glue the rectangles, mesh them with netgen and read the file back."""
    faces = []
    for x0, y0, x1, y1, name, h in rects:
        face = Rectangle(x1 - x0, y1 - y0).Face().Move((x0, y0, 0))
        face.faces.name = name
        if h is not None:
            face.faces.maxh = h
        faces.append(face)
    shape = Glue(faces)
    for edge in shape.edges:
        c = edge.center
        name = next((n for test, n in edges if test((c.x, c.y))), None)
        if name is not None:
            edge.name = name
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    tris = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    named = {name for _, name in edges}
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat in named
    ]
    _write_gmsh(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh_cross_section(path, thickness=THICKNESS)


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
