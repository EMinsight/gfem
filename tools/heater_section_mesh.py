"""Build the heater tutorial's meshes: a TiN heater over a silicon guide, in section.

Usage (gmsh is not in the lock; ``just heater-section-mesh`` runs this with
``uv run --with gmsh==4.15.2 --no-project``):
    python tools/heater_section_mesh.py docs/tutorials/data

Writes ``heater_w2.msh`` and ``heater_w7p5.msh`` (gmsh 2.2, um, triangles
only, read by `gfem.Mesh.from_gmsh_cross_section`). The committed files are
this script's output with gmsh 4.15.2 on one thread; another gmsh version can
move nodes, so the script refuses it, and a regenerated mesh means
re-measuring the tutorial's values.

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
inside the windows, each growing by GROW per unit distance, at most H_MAX.

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


def _sizes(gmsh, owner):
    """Set the background mesh size: fine at the cores, the heater and the windows."""

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
    for names, h in ((("core", "neighbour"), H_CORE), (("heater",), H_HEATER)):
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", edges(names))
        field.setNumber(distance, "Sampling", 200)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", h)
        field.setNumber(threshold, "SizeMax", H_MAX)
        field.setNumber(threshold, "DistMin", 0.0)
        field.setNumber(threshold, "DistMax", (H_MAX - h) / GROW)
        sizes.append(threshold)
    for x in (0.0, NEIGHBOUR):
        x0, y0, x1, y1 = _window(x)
        box = field.add("Box")
        for key, value in (
            ("VIn", H_WINDOW),
            ("VOut", H_MAX),
            ("XMin", x0),
            ("XMax", x1),
            ("YMin", y0),
            ("YMax", y1),
            ("Thickness", (H_MAX - H_WINDOW) / GROW),
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


def main(outdir):
    """Write every mesh of ``MESHES`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, width in MESHES.items():
        nodes, triangles = build(outdir / name, width)
        print(f"{outdir / name}: {nodes} nodes, {triangles} triangles")


if __name__ == "__main__":
    main(sys.argv[1])
