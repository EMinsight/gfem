"""Build the grating-coupler tutorial's mesh: a silicon grating on oxide, in section.

Usage (gmsh is not in the lock; ``just grating-mesh`` runs this with
``uv run --with gmsh==4.15.2 --no-project``):
    python tools/grating_mesh.py docs/tutorials/data

Writes ``grating.msh`` (gmsh 2.2, um, triangles only, read by
`gfem.Mesh.from_gmsh_cross_section`). The committed file is this script's
output with gmsh 4.15.2 on one thread; another gmsh version can move nodes,
so the script refuses it, and a regenerated mesh means re-measuring the
tutorial's values.

Geometry (um), the section in the (x, y) plane, x along the guide, y up,
y = 0 at the top of the buried oxide; the slab is invariant along z (the
grating's lines):

- "substrate" (silicon): X0 < x < X1, -2.5 < y < -2;
- "box" (buried oxide): -2 < y < 0;
- "si": the 220 nm guide, 0 < y < 0.22, with TEETH = 25 trenches 70 nm deep
  in its top, the first starting at x = FILL * PERIOD (PERIOD = 0.63,
  FILL = 0.5: silicon teeth of 315 nm, then trenches of 315 nm);
- "trench": the oxide in the trenches;
- "clad" (oxide): 0.22 < y < 1.6, and the trenches' top.

Lines: "input" (x = X0, the window WINDOW across the guide: the input wave
port) and "output" (x = X1, the same window: the output wave port), "left"
and "right" (the rest of each end), "top" (y = 1.6) and "bottom" (y = -2.5),
the absorbing faces, and "walls", the trenches' 50 sidewalls (the shape
design moves them).

Mesh sizes: H_SI along the guide's and the trenches' edges, growing by GROW
per unit distance to H_OX in the oxide; H_SUB in the substrate.

=========== ===== =========
mesh        nodes triangles
=========== ===== =========
grating.msh 4457  8591
=========== ===== =========
"""

import sys
from pathlib import Path

GMSH_VERSION = "4.15.2"

X0, X1 = -6.0, 19.0  # um, the ends of the section
SUBSTRATE, BOX, CLADDING = -2.3, -2.0, 1.8  # um: substrate bottom, box bottom, top
T_SI, ETCH = 0.22, 0.07  # um, the guide and the trench depth
PERIOD, FILL, TEETH = 0.63, 0.5, 25  # um, the silicon share of a period, trenches
WINDOW = (-1.0, 1.0)  # um, the wave ports' span in y
H_SI, H_OX, H_SUB, GROW = 0.07, 0.28, 0.14, 1.0  # um

#: The tutorial's mesh.
MESHES = {"grating.msh": None}
#: The named lines.
LINES = ("input", "output", "left", "right", "top", "bottom", "walls")


def trenches():
    """Each trench's (x_left, x_right), um."""
    return [(i * PERIOD + FILL * PERIOD, (i + 1) * PERIOD) for i in range(TEETH)]


def _regions(gmsh):
    """Draw and fragment the section; return each gmsh surface's region name."""
    occ = gmsh.model.occ

    def rect(x0, y0, x1, y1):
        return occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)

    lo, hi = WINDOW
    base = T_SI - ETCH
    shapes = [(rect(a, base, b, T_SI), "trench") for a, b in trenches()]
    shapes += [
        (rect(X0, 0.0, X1, T_SI), "si"),
        (rect(X0, SUBSTRATE, X1, BOX), "substrate"),
        (rect(X0, BOX, X1, lo), "box"),
        (rect(X0, lo, X1, 0.0), "box"),
        (rect(X0, T_SI, X1, hi), "clad"),
        (rect(X0, hi, X1, CLADDING), "clad"),
    ]
    tags = [t for t, _ in shapes]
    # priority: a piece belongs to the first shape that holds it
    _, pieces = occ.fragment([(2, tags[0])], [(2, t) for t in tags[1:]])
    occ.synchronize()
    owner = {}
    for (_, name), found in zip(shapes, pieces, strict=True):
        for _, piece in found:
            owner.setdefault(piece, name)
    for number, name in enumerate(dict.fromkeys(owner.values()), start=1):
        surfaces = sorted(p for p, n in owner.items() if n == name)
        gmsh.model.addPhysicalGroup(2, surfaces, number, name=name)
    walls = {a for pair in trenches() for a in pair}
    found = {name: [] for name in LINES}
    eps = 1e-6
    for _, curve in gmsh.model.getEntities(1):
        x, y, _ = occ.getCenterOfMass(1, curve)
        if abs(x - X0) < eps or abs(x - X1) < eps:
            port = lo < y < hi
            if abs(x - X0) < eps:
                found["input" if port else "left"].append(curve)
            else:
                found["output" if port else "right"].append(curve)
        elif abs(y - CLADDING) < eps:
            found["top"].append(curve)
        elif abs(y - SUBSTRATE) < eps:
            found["bottom"].append(curve)
        elif base < y < T_SI and min(abs(x - w) for w in walls) < eps:
            found["walls"].append(curve)
    for number, name in enumerate(LINES, start=len(set(owner.values())) + 1):
        gmsh.model.addPhysicalGroup(1, sorted(found[name]), number, name=name)
    return owner


def _sizes(gmsh, owner):
    """Fine along the guide and the trenches, coarse in the oxide and substrate."""
    field = gmsh.model.mesh.field
    guide = sorted(
        {
            c
            for p, n in owner.items()
            if n in ("si", "trench")
            for _, c in gmsh.model.getBoundary([(2, p)], oriented=False)
        }
    )
    distance = field.add("Distance")
    field.setNumbers(distance, "CurvesList", guide)
    field.setNumber(distance, "Sampling", 400)
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    field.setNumber(threshold, "SizeMin", H_SI)
    field.setNumber(threshold, "SizeMax", H_OX)
    field.setNumber(threshold, "DistMin", 0.0)
    field.setNumber(threshold, "DistMax", (H_OX - H_SI) / GROW)
    substrate = field.add("Constant")
    field.setNumbers(
        substrate,
        "SurfacesList",
        sorted(p for p, n in owner.items() if n == "substrate"),
    )
    field.setNumber(substrate, "VIn", H_SUB)
    field.setNumber(substrate, "VOut", H_OX)
    smallest = field.add("Min")
    field.setNumbers(smallest, "FieldsList", [threshold, substrate])
    field.setAsBackgroundMesh(smallest)
    for k in ("ExtendFromBoundary", "FromPoints", "FromCurvature"):
        gmsh.option.setNumber("Mesh.MeshSize" + k, 0)


def build(out):
    """Mesh the section and write it to ``out``; return the node and triangle counts."""
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    _sizes(gmsh, _regions(gmsh))
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
    for name in MESHES:
        nodes, triangles = build(outdir / name)
        print(f"{outdir / name}: {nodes} nodes, {triangles} triangles")


if __name__ == "__main__":
    main(sys.argv[1])
