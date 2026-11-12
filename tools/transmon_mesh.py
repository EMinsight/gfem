"""Build the transmon tutorial's meshes: a small 3D transmon in a box cavity.

Usage (gmsh is not in the lock; ``just transmon-mesh`` runs this with
``uv run --with gmsh==4.15.2 --no-project``):
    python tools/transmon_mesh.py docs/tutorials/data

Writes ``transmon.msh`` and ``transmon_coarse.msh`` (gmsh 2.2, mm). The
committed files are this script's output with gmsh 4.15.2 on one thread; another
gmsh version can move nodes, so the script refuses it, and a regenerated mesh
means re-measuring the tutorial's values.

Geometry (mm):
- PEC box |x| < 4, |y| < 17, |z| < 14 (TE101, E along x: the readout mode).
- Chip (volume "chip", silicon) |x| < 2.5, |y| < 3, -0.4 < z < 0, floating at
  the box centre; volume "vacuum" is the rest.
- Two pads on z = 0, a void (no volume, PEC faces): each 0.4 (x) by 0.7 (y) by
  0.05 thick, facing edges at x = -+gap/2 (gap 0.3). Each carries a lead stub
  0.1 wide (y) from its facing edge to x = -+0.05.
- Junction: the rectangle |x| < 0.05, |y| < 0.05 on z = 0 between the stub tips.

Surfaces: "junction" (the port), "pad_edge" (the facing pad edges, x = -+gap/2,
beside the stubs), "stub_wall" (y = -+0.05), "stub_tip" (x = -+0.05), "pad"
(every other metal face), "chip_top" (z = 0 off the metal), "chip_side" (the
chip's other faces), "walls" (the box).

Mesh sizes: h_j at the junction and stubs, h_pad on the pads, each growing by
``grow`` per unit distance, at most h_max.

==================== ====== ===== ==== ===== ====== ===========
mesh                 h_j    h_pad grow h_max nodes  tetrahedra
==================== ====== ===== ==== ===== ====== ===========
transmon.msh         0.04   0.12  0.4  6     1636   7880
transmon_coarse.msh  0.05   0.2   0.5  8     913    4308
==================== ====== ===== ==== ===== ====== ===========

Named groups (gmsh number, name, entities): volumes 1 "vacuum", 2 "chip";
surfaces 11 "junction" (1), 12 "pad_edge" (4), 13 "stub_wall" (4),
14 "stub_tip" (2), 15 "pad", 16 "chip_top", 17 "chip_side", 18 "walls".
"""

import sys
from pathlib import Path

GMSH_VERSION = "4.15.2"

GEOMETRY = dict(
    bx=4.0, by=17.0, bz=14.0,  # box half sizes
    cx=2.5, cy=3.0, ct=0.4,  # chip half sizes and thickness
    gap=0.3, pad_x=0.4, pad_y=0.7, t=0.05,  # pad gap, depth, width, thickness
    lead=0.1, lj=0.1,  # stub width (= junction width), junction length
)  # fmt: skip

MESHES = {
    "transmon.msh": dict(h_j=0.04, h_pad=0.12, grow=0.4, h_max=6.0),
    "transmon_coarse.msh": dict(h_j=0.05, h_pad=0.2, grow=0.5, h_max=8.0),
}

SURFACES = (
    "junction",
    "pad_edge",
    "stub_wall",
    "stub_tip",
    "pad",
    "chip_top",
    "chip_side",
    "walls",
)


def _geometry(gmsh):
    """Box, chip, pads (removed, keeping their faces) and junction.

    Returns the metal faces, the chip's faces and the junction's faces.
    """
    P = GEOMETRY
    g = P["gap"] / 2
    occ = gmsh.model.occ
    bx, by, bz = P["bx"], P["by"], P["bz"]
    air = occ.addBox(-bx, -by, -bz, 2 * bx, 2 * by, 2 * bz)
    chip = occ.addBox(-P["cx"], -P["cy"], -P["ct"], 2 * P["cx"], 2 * P["cy"], P["ct"])
    pads = []
    for s in (-1, 1):
        x0 = -g - P["pad_x"] if s < 0 else g
        pad = occ.addBox(x0, -P["pad_y"] / 2, 0, P["pad_x"], P["pad_y"], P["t"])
        xs = -g if s < 0 else P["lj"] / 2
        stub = occ.addBox(xs, -P["lead"] / 2, 0, g - P["lj"] / 2, P["lead"], P["t"])
        fused, _ = occ.fuse([(3, pad)], [(3, stub)])
        pads.append(fused[0])
    jj = occ.addRectangle(-P["lj"] / 2, -P["lead"] / 2, 0, P["lj"], P["lead"])
    _, parts = occ.fragment([(3, air)], [(3, chip), *pads, (2, jj)])
    occ.synchronize()
    _, chip_parts, padl, padr, jj_parts = parts
    pad_vols = [v for p in (padl, padr) for v in p]
    boundary = gmsh.model.getBoundary(pad_vols, combined=False, oriented=False)
    metal = {abs(t) for _, t in boundary}
    occ.remove(pad_vols)
    occ.synchronize()
    chip_tags = [t for _, t in chip_parts]
    vac_tags = [t for _, t in gmsh.model.getEntities(3) if t not in chip_tags]
    gmsh.model.addPhysicalGroup(3, vac_tags, 1, "vacuum")
    gmsh.model.addPhysicalGroup(3, chip_tags, 2, "chip")
    chip_boundary = gmsh.model.getBoundary(
        [(3, t) for t in chip_tags], combined=False, oriented=False
    )
    chip_faces = {abs(t) for _, t in chip_boundary}
    jj_tags = {t for d, t in jj_parts if d == 2}
    return metal, chip_faces, jj_tags


def _metal_group(box):
    """The surface group of a metal face from its bounding box."""
    P = GEOMETRY
    x0, y0, _, x1, y1, _ = box
    flat_x, flat_y = abs(x0 - x1) < 1e-5, abs(y0 - y1) < 1e-5
    lead = P["lead"] / 2
    if (
        flat_x
        and abs(abs(x0) - P["gap"] / 2) < 1e-5
        and min(abs(y0), abs(y1)) > lead - 1e-5
    ):
        return "pad_edge"
    if flat_y and abs(abs(y0) - lead) < 1e-5:
        return "stub_wall"
    if flat_x and abs(abs(x0) - P["lj"] / 2) < 1e-5:
        return "stub_tip"
    return "pad"


def _surface_groups(gmsh, metal, chip_faces, jj_tags):
    """Name every surface (gmsh numbers 11 to 18, in ``SURFACES`` order)."""
    groups = {k: [] for k in SURFACES}
    for _, s in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, s)
        if s in jj_tags:
            name = "junction"
        elif s in metal:
            name = _metal_group(box)
        elif s in chip_faces:
            top = abs(box[2] - box[5]) < 1e-5 and abs(box[2]) < 1e-5
            name = "chip_top" if top else "chip_side"
        else:
            name = "walls"
        groups[name].append(s)
    for i, (name, tags) in enumerate(groups.items(), start=11):
        if not tags:
            raise RuntimeError(f"no {name} surface")
        gmsh.model.addPhysicalGroup(2, tags, i, name)
    return groups


def _size_fields(gmsh, groups, *, h_j, h_pad, grow, h_max):
    """h_j near the junction and stubs, h_pad near the pads, growing by ``grow``."""
    f = gmsh.model.mesh.field
    near_j = groups["junction"] + groups["stub_wall"] + groups["stub_tip"]
    near_p = groups["pad"] + groups["pad_edge"]
    f.add("Distance", 1)
    f.setNumbers(1, "SurfacesList", near_j)
    f.setNumber(1, "Sampling", 20)
    f.add("Distance", 2)
    f.setNumbers(2, "SurfacesList", near_p)
    f.setNumber(2, "Sampling", 40)
    f.add("MathEval", 3)
    f.setString(3, "F", f"{h_j} + {grow}*F1")
    f.add("MathEval", 4)
    f.setString(4, "F", f"{h_pad} + {grow}*F2")
    f.add("Min", 5)
    f.setNumbers(5, "FieldsList", [3, 4])
    f.setAsBackgroundMesh(5)
    for k in ("ExtendFromBoundary", "FromPoints", "FromCurvature"):
        gmsh.option.setNumber("Mesh.MeshSize" + k, 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)


def build(out, **sizes):
    """Mesh the transmon with the given sizes and write it to ``out``.

    Returns the node and tetrahedron counts.
    """
    import gmsh

    if gmsh.__version__ != GMSH_VERSION:
        raise SystemExit(f"gmsh {GMSH_VERSION} wanted, got {gmsh.__version__}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    groups = _surface_groups(gmsh, *_geometry(gmsh))
    _size_fields(gmsh, groups, **sizes)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(out))
    nodes = len(gmsh.model.mesh.getNodes()[0])
    tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    gmsh.finalize()
    return nodes, tets


def main(outdir):
    """Write every mesh of ``MESHES`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, sizes in MESHES.items():
        nodes, tets = build(outdir / name, **sizes)
        print(f"{outdir / name}: {nodes} nodes, {tets} tetrahedra")


if __name__ == "__main__":
    main(sys.argv[1])
