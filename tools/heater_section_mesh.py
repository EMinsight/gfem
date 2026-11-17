"""Build the heater tutorial's meshes: a TiN heater over a silicon guide, in section.

Usage (gmsh is not in the lock; ``just heater-section-mesh`` runs this with
``uv run --with gmsh==4.15.2 --no-project``):
    python tools/heater_section_mesh.py docs/tutorials/data

Writes ``heater_w2.msh`` and ``heater_w7p5.msh`` (gmsh 2.2, um, triangles
only, read by `gfem.Mesh.from_gmsh_cross_section`), and ``heater_straight.msh``
(the 2 um section, coarser, extruded along z over STRAIGHT_LENGTH in
STRAIGHT_LAYERS layers of tetrahedra, read by `gfem.Mesh.from_gmsh`; the
thermo-optic phase recipe's short straight heater). The committed files are
this script's output with gmsh 4.15.2 on one thread; another gmsh version can
move nodes, so the script refuses it, and a regenerated mesh means
re-measuring the tutorial's and the recipe's values.

Geometry (um), the section in the (x, y) plane, y up, y = 0 at the top of the
buried oxide; the domain is 40 x 18 um:

- "substrate" (silicon): |x| < 20, -15.14 < y < -2 (13.14 um, its bottom is the
  heat sink);
- "box" (buried oxide): |x| < 20, -2 < y < 0;
- "cladding" (oxide): |x| < 20, 0 < y < 2.86;
- "core" (silicon): |x| < 0.25, 0 < y < 0.22, the guide under the heater;
- "neighbour" (silicon): the same strip centred at x = NEIGHBOUR = 10;
- "heater" (TiN): |x| < width / 2, 2.22 < y < 2.36 (0.14 thick, 2 um above the
  core, 0.5 um of oxide over it);
- "window" and "neighbour_window" (oxide): each core's optical window, its
  rectangle grown by 1.5 um sideways and 1 um up and down; the window takes
  the box and cladding inside that rectangle, so the tutorial can solve each
  core's mode on its window alone.

Surfaces (lines): "bottom" (y = -15.14) and "top" (y = 2.86); the sides are
left unnamed (insulated).

Mesh sizes: H_CORE on the cores' edges, H_HEATER on the heater's, H_WINDOW
inside the windows, each growing by GROW per unit distance, at most H_MAX
(STRAIGHT_SIZES in the same order for the extruded mesh). The extruded mesh
names the faces "bottom" and "top" as the section's lines, and its end faces
"front" (z = 0) and "back" (z = STRAIGHT_LENGTH), as a cross-section's.

=============== ===== ===== =========
mesh            width nodes triangles
=============== ===== ===== =========
heater_w2.msh   2.0   4634  9173
heater_w7p5.msh 7.5   6422  12724
=============== ===== ===== =========
"""

import sys
from pathlib import Path

GMSH_VERSION = "4.15.2"

HALF_WIDTH = 20.0  # um, the domain is |x| < HALF_WIDTH
SUBSTRATE, BOX, CLADDING = 13.14, 2.0, 2.86  # um, layer thicknesses
CORE_WIDTH, CORE_HEIGHT = 0.5, 0.22  # um
HEATER_BOTTOM, HEATER_THICKNESS = 2.22, 0.14  # um
NEIGHBOUR = 10.0  # um, the neighbouring core's centre
WINDOW_SIDE, WINDOW_UP_DOWN = 1.5, 1.0  # um, each optical window around its core
H_CORE, H_HEATER, H_WINDOW, H_MAX, GROW = 0.03, 0.035, 0.1, 2.0, 0.3  # um

#: The tutorial's meshes: file name -> heater width in um.
MESHES = {"heater_w2.msh": 2.0, "heater_w7p5.msh": 7.5}
#: The recipe's extruded straight heater: file name -> heater width in um.
STRAIGHT = {"heater_straight.msh": 2.0}
STRAIGHT_LENGTH, STRAIGHT_LAYERS = 20.0, 4  # um, layers of elements along z
STRAIGHT_SIZES = (0.08, 0.12, 0.25, 2.5, 0.4)  # H_CORE ... GROW, coarser


def _core(x):
    """A core's rectangle (x0, y0, x1, y1) centred at ``x``."""
    return (x - CORE_WIDTH / 2, 0.0, x + CORE_WIDTH / 2, CORE_HEIGHT)


def _window(x):
    """The optical window around the core centred at ``x``."""
    x0, y0, x1, y1 = _core(x)
    return (
        x0 - WINDOW_SIDE,
        y0 - WINDOW_UP_DOWN,
        x1 + WINDOW_SIDE,
        y1 + WINDOW_UP_DOWN,
    )


def _regions(gmsh, width):
    """Draw and fragment the section; name its regions and the bottom and top lines.

    Returns each gmsh surface's region name.
    """
    occ = gmsh.model.occ

    def rect(x0, y0, x1, y1):
        return occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)

    bottom, top = -BOX - SUBSTRATE, CLADDING
    heater = (-width / 2, HEATER_BOTTOM, width / 2, HEATER_BOTTOM + HEATER_THICKNESS)
    # priority: a piece belongs to the first shape that holds it
    shapes = {
        "core": rect(*_core(0.0)),
        "neighbour": rect(*_core(NEIGHBOUR)),
        "heater": rect(*heater),
        "window": rect(*_window(0.0)),
        "neighbour_window": rect(*_window(NEIGHBOUR)),
        "box": rect(-HALF_WIDTH, -BOX, HALF_WIDTH, 0.0),
        "cladding": rect(-HALF_WIDTH, 0.0, HALF_WIDTH, top),
        "substrate": rect(-HALF_WIDTH, bottom, HALF_WIDTH, -BOX),
    }
    tags = list(shapes.values())
    _, pieces = occ.fragment([(2, tags[0])], [(2, t) for t in tags[1:]])
    occ.synchronize()
    owner = {}
    for name, found in zip(shapes, pieces, strict=True):
        for _, piece in found:
            owner.setdefault(piece, name)
    for number, name in enumerate(shapes, start=1):
        surfaces = sorted(p for p, n in owner.items() if n == name)
        gmsh.model.addPhysicalGroup(2, surfaces, number, name=name)
    eps = 1e-6
    for number, (name, y) in enumerate(
        (("bottom", bottom), ("top", top)), start=len(shapes) + 1
    ):
        lines = gmsh.model.getEntitiesInBoundingBox(
            -HALF_WIDTH - eps, y - eps, -eps, HALF_WIDTH + eps, y + eps, eps, dim=1
        )
        gmsh.model.addPhysicalGroup(1, sorted(t for _, t in lines), number, name=name)
    return owner


def _sizes(gmsh, owner, sizes=(H_CORE, H_HEATER, H_WINDOW, H_MAX, GROW)):
    """Set the background mesh size: fine at the cores, the heater and the windows."""
    h_core, h_heater, h_window, h_max, grow = sizes

    def edges(names):
        return sorted(
            {
                c
                for p, n in owner.items()
                if n in names
                for _, c in gmsh.model.getBoundary([(2, p)], oriented=False)
            }
        )

    field = gmsh.model.mesh.field
    sizes = []
    for names, h in ((("core", "neighbour"), h_core), (("heater",), h_heater)):
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", edges(names))
        field.setNumber(distance, "Sampling", 200)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", h)
        field.setNumber(threshold, "SizeMax", h_max)
        field.setNumber(threshold, "DistMin", 0.0)
        field.setNumber(threshold, "DistMax", (h_max - h) / grow)
        sizes.append(threshold)
    for x in (0.0, NEIGHBOUR):
        x0, y0, x1, y1 = _window(x)
        box = field.add("Box")
        for key, value in (
            ("VIn", h_window),
            ("VOut", h_max),
            ("XMin", x0),
            ("XMax", x1),
            ("YMin", y0),
            ("YMax", y1),
            ("Thickness", (h_max - h_window) / grow),
        ):
            field.setNumber(box, key, value)
        sizes.append(box)
    smallest = field.add("Min")
    field.setNumbers(smallest, "FieldsList", sizes)
    field.setAsBackgroundMesh(smallest)
    for k in ("ExtendFromBoundary", "FromPoints", "FromCurvature"):
        gmsh.option.setNumber("Mesh.MeshSize" + k, 0)


def build(out, width):
    """Mesh the section with a heater ``width`` um wide and write it to ``out``.

    Returns the node and triangle counts.
    """
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    _sizes(gmsh, _regions(gmsh, width))
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal-Delaunay
    gmsh.model.mesh.generate(2)
    nodes = len(gmsh.model.mesh.getNodes()[0])
    triangles = len(gmsh.model.mesh.getElementsByType(2)[0])  # 3-node triangles
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(out))
    gmsh.finalize()
    return nodes, triangles


def build_straight(out, width):
    """Extrude the coarse section with a heater ``width`` um wide; write ``out``.

    Every surface of the section is extruded along z over STRAIGHT_LENGTH in
    STRAIGHT_LAYERS layers, each prism split into tetrahedra, so the layers
    conform. Returns the node and tetrahedron counts.
    """
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    owner = _regions(gmsh, width)
    _sizes(gmsh, owner, STRAIGHT_SIZES)
    occ = gmsh.model.occ
    centre = {s: occ.getCenterOfMass(2, s)[:2] for s in owner}
    occ.extrude(
        [(2, s) for s in sorted(owner)],
        0,
        0,
        STRAIGHT_LENGTH,
        numElements=[STRAIGHT_LAYERS],
    )
    occ.removeAllDuplicates()  # one face between two extruded pieces
    occ.synchronize()
    region = {}
    for _, volume in gmsh.model.getEntities(3):
        x, y = occ.getCenterOfMass(3, volume)[:2]
        (surface,) = [
            s for s, c in centre.items() if abs(c[0] - x) + abs(c[1] - y) < 1e-6
        ]
        region[volume] = owner[surface]
    # the section's groups and names: the volumes take its names
    for dim, tag in gmsh.model.getPhysicalGroups():
        gmsh.model.removePhysicalName(gmsh.model.getPhysicalName(dim, tag))
    gmsh.model.removePhysicalGroups()
    for number, name in enumerate(dict.fromkeys(owner.values()), start=1):
        volumes = sorted(v for v, n in region.items() if n == name)
        gmsh.model.addPhysicalGroup(3, volumes, number, name=name)
    eps = 1e-6
    bottom, top = -BOX - SUBSTRATE, CLADDING
    faces = {
        "bottom": (-HALF_WIDTH, bottom, 0.0, HALF_WIDTH, bottom, STRAIGHT_LENGTH),
        "top": (-HALF_WIDTH, top, 0.0, HALF_WIDTH, top, STRAIGHT_LENGTH),
        "front": (-HALF_WIDTH, bottom, 0.0, HALF_WIDTH, top, 0.0),
        "back": (
            -HALF_WIDTH,
            bottom,
            STRAIGHT_LENGTH,
            HALF_WIDTH,
            top,
            STRAIGHT_LENGTH,
        ),
    }
    for number, (name, (x0, y0, z0, x1, y1, z1)) in enumerate(
        faces.items(), start=len(set(owner.values())) + 1
    ):
        found = gmsh.model.getEntitiesInBoundingBox(
            x0 - eps, y0 - eps, z0 - eps, x1 + eps, y1 + eps, z1 + eps, dim=2
        )
        gmsh.model.addPhysicalGroup(2, sorted(t for _, t in found), number, name=name)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal-Delaunay
    gmsh.model.mesh.generate(3)
    nodes = len(gmsh.model.mesh.getNodes()[0])
    tetrahedra = len(gmsh.model.mesh.getElementsByType(4)[0])
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(str(out))
    gmsh.finalize()
    return nodes, tetrahedra


def main(outdir):
    """Write every mesh of ``MESHES`` and ``STRAIGHT`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, width in MESHES.items():
        nodes, triangles = build(outdir / name, width)
        print(f"{outdir / name}: {nodes} nodes, {triangles} triangles")
    for name, width in STRAIGHT.items():
        nodes, tetrahedra = build_straight(outdir / name, width)
        print(f"{outdir / name}: {nodes} nodes, {tetrahedra} tetrahedra")


if __name__ == "__main__":
    main(sys.argv[1])
