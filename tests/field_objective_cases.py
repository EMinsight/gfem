"""Driven cases for objectives of fields, with references.

Two pairs of cases, each a lossy region and regions to average over:

- Block guide, `block_guide_model`: a PEC guide 3 x 2 x 1 mm (x, y, z) with
  TE10 wave ports "p1" and "p2" at its ends, at `BLOCK_GUIDE_F`. A lossy
  eps_r 4 (tan_delta `TAN`) block x 1..2, y 0.5..1.5, z 0.25..0.75 touches
  no wall (partial-height edges, eight 3D dielectric corners). Volumes:
  "bsub" (the block's corner octant x 1..1.5, y 0.5..1, z 0.25..0.5), "blk"
  (the rest of the block), "rfar" (air, x 2.25..2.75, y 0.5..1, z
  0.25..0.75), "redge" (air, x 2..2.5, y 1.5..2, z 0.25..0.75, sharing the
  block's vertical edge) and "fill".
- Stepped line, `stepped_line_model`: x 3..7, y 0..1, z 0..1 mm, eps_r 4,
  PMC sides, 50 ohm lumped ports at x = 3 and 7, at `STEPPED_LINE_F`; the
  top plate steps down to z = 0.5 at x = 5 (a 270 degree re-entrant PEC
  edge along y). A lossy layer x 5..6.5, z < 0.25: its middle "lsub" (y
  0.25..0.75) and the rest "lossy"; a box x 4..5, z 0.5..0.75 touching the
  edge, its middle "rbox" and the rest "rbw"; "diel" elsewhere.

References (`BLOCK_GUIDE_J`, `STEPPED_LINE_J`): per objective, its value and
uncertainty in SI units (W for absorbed power, V/m for a mean of E_z,
(V/m)^2 for |mean E|^2). The guide's is order 4 on the n = 4 mesh graded 2
towards the block's planes, its uncertainty the larger difference to
uniform order 4 at n = 4 and uniform order 3 at n = 6. The stepped line's
materials are y-invariant, so its primal is: its reference is order 4 on a
one-y-cell mesh graded 2.5 towards the edge (n = 12), its uncertainty the
larger difference to order 4 at n = 8 and order 3 at n = 16 on that
family. Objective keys: ``abs_<r>`` the power absorbed in region set r,
``mz_<r>`` Re mean E_z over it, ``m2_<r>`` |mean E|^2 over it.

Two earlier cases, whose regions are full height:

- Half-width guide, `half_width_guide_model`: a PEC guide 4 x 2 x 0.8 mm,
  TE10 wave ports, 120 GHz, a lossy eps_r 4 block x 1.5..2.5, y < 1, full
  height; air regions "rfar" (x 3..3.5, y 1..1.5) and "redge" (x 2.5..3, y
  1..1.5, touching the block's edge). References `HALF_WIDTH_GUIDE_J`, per
  n: same-mesh order 4, uncertainty the difference to order 3.
- Edge-box line, `edge_box_line_model`: the step line of
  `refinement_cases` (x 0..10, y 0..2, z 0..1 mm, eps_r 4, top stepping to
  z = 0.5 at x = 5, PEC plates, 50 ohm lumped ports, PMC sides) at 8 GHz,
  with a lossy region x 5..7.5, z < 0.5 and a full-height box "rbox" x
  2.5..5, z 0.5..1 at the edge. Reference `EDGE_BOX_LINE_J`: order 4 on a
  mesh graded 2.5 towards the edge (n = 8), uncertainty the difference to
  n = 6.

Tests never recompute the references.
"""

import itertools
from pathlib import Path

import numpy as np
import structured_meshes as sm

import gfem

#: The lossy materials' loss tangent.
TAN = 0.05
BLOCK_GUIDE_F = 120e9
STEPPED_LINE_F = 8e9
HALF_WIDTH_GUIDE_F = 120e9
EDGE_BOX_LINE_F = 8e9

#: Per objective: (reference, uncertainty), SI.
BLOCK_GUIDE_J = {
    "abs_blk": (0.08052307046449345, 7.31e-05),
    "abs_sub": (0.010299060938065582, 1.35e-05),
    "mz_far": (23392.394155888345, 3.85),
    "m2_far": (658477093.8769604, 5.98e04),
    "mz_edge": (6561.508740168914, 3.15),
    "m2_edge": (101486373.44937222, 2.31e04),
    "mz_sub": (-9428.574525811975, 8.47),
    "m2_sub": (178805110.70774993, 2.22e05),
    "mz_blk": (-6739.931141572785, 2.22),
}

#: The regions of each block-guide objective.
BLOCK_GUIDE_REGIONS = {
    "blk": ("blk", "bsub"),
    "sub": ("bsub",),
    "far": ("rfar",),
    "edge": ("redge",),
}

STEPPED_LINE_J = {
    "abs_layer": (0.003866804839967807, 1.36e-07),
    "abs_lsub": (0.0019334024199839034, 6.79e-08),
    "mz_box": (9401.60328304315, 1.14),
    "m2_box": (103853884.1959225, 2.4e04),
    "mz_boxw": (9401.60328304315, 1.14),
    "m2_boxw": (103853884.1959225, 2.4e04),
}

STEPPED_LINE_REGIONS = {
    "layer": ("lsub", "lossy"),
    "lsub": ("lsub",),
    "box": ("rbox",),
    "boxw": ("rbox", "rbw"),
}

#: Per n.
HALF_WIDTH_GUIDE_J = {
    2: {
        "absorbed": (0.5771087037548603, 7.94e-04),
        "mz_far": (-8982.792782921688, 69.6),
        "m2_far": (120425131.71469285, 1.88e06),
        "mz_edge": (-17033.55184725354, 138.0),
        "m2_edge": (343355785.20281047, 4.37e06),
    },
    4: {
        "absorbed": (0.5770495264147439, 2.57e-05),
        "mz_far": (-8979.372066553919, 1.45),
        "m2_far": (120320420.07843147, 4.05e04),
        "mz_edge": (-17025.982175776255, 2.89),
        "m2_edge": (343118245.4868088, 9.07e04),
    },
}

EDGE_BOX_LINE_J = {
    "absorbed": (0.024696523543655875, 2.44e-07),
    "mz": (5848.204189081531, 0.0311),
    "m2": (75209996.60502425, 2.32e03),
}


def _lines(marks, n):
    out = [float(marks[0])]
    for a, b in itertools.pairwise(marks):
        m = max(1, round((b - a) * n))
        out.extend((a + (b - a) * np.arange(1, m + 1) / m).tolist())
    return np.unique(np.round(np.array(out), 12))


def _inside(c, box):
    return all(lo < v < hi for v, (lo, hi) in zip(c, box, strict=True))


_BLOCK = ((1.0, 2.0), (0.5, 1.5), (0.25, 0.75))
_SUB = ((1.0, 1.5), (0.5, 1.0), (0.25, 0.5))
_FAR = ((2.25, 2.75), (0.5, 1.0), (0.25, 0.75))
_EDGE = ((2.0, 2.5), (1.5, 2.0), (0.25, 0.75))


def _guide_surface(length, width, height):
    def surface(c):
        if abs(c[0]) < 1e-9:
            return "port1"
        if abs(c[0] - length) < 1e-9:
            return "port2"
        if (
            abs(c[2]) < 1e-9
            or abs(c[2] - height) < 1e-9
            or abs(c[1]) < 1e-9
            or abs(c[1] - width) < 1e-9
        ):
            return "wall"
        return ""

    return surface


def _guide_model(mesh, lossy_regions):
    air = gfem.Material(eps_r=1.0)
    lossy = gfem.Material(eps_r=4.0, tan_delta=TAN)
    materials = {r: lossy if r in lossy_regions else air for r in mesh.volumes}
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("wall")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def block_guide_model(directory: Path, n: int) -> gfem.Model:
    """The block guide on a uniform mesh of ``n`` cells per mm."""
    edges = (
        _lines([0, 1, 1.5, 2, 2.25, 2.5, 2.75, 3], n),
        _lines([0, 0.5, 1, 1.5, 2], n),
        _lines([0, 0.25, 0.5, 0.75, 1], n),
    )

    def volume(c):
        if _inside(c, _SUB):
            return "bsub"
        if _inside(c, _BLOCK):
            return "blk"
        if _inside(c, _FAR):
            return "rfar"
        if _inside(c, _EDGE):
            return "redge"
        return "fill"

    mesh = sm.structured(edges, volume, _guide_surface(3.0, 2.0, 1.0))
    path = sm.write_msh22(Path(directory) / f"block_guide_{n}.msh", mesh)
    return _guide_model(gfem.Mesh.from_gmsh(path, length_unit="mm"), ("blk", "bsub"))


def stepped_line_model(directory: Path, n: int) -> gfem.Model:
    """The stepped line on a uniform mesh of ``n`` cells per mm."""
    xs = _lines([3, 4, 5, 6.5, 7], n)
    ys = _lines([0, 0.25, 0.75, 1], n)
    zs = _lines([0, 0.25, 0.5, 0.75, 1], n)

    def volume(c):
        x, y, z = c
        if x > 5 and z > 0.5:
            return None
        middle = 0.25 < y < 0.75
        if 5 < x < 6.5 and z < 0.25:
            return "lsub" if middle else "lossy"
        if 4 < x < 5 and 0.5 < z < 0.75:
            return "rbox" if middle else "rbw"
        return "diel"

    def surface(c):
        x, _, z = c
        tol = 1e-9
        if abs(x - 3) < tol:
            return "port1"
        if abs(x - 7) < tol:
            return "port2"
        if abs(z) < tol:
            return "bottom"
        if (
            abs(z - 1) < tol
            or (abs(x - 5) < tol and z > 0.5 - tol)
            or (abs(z - 0.5) < tol and x > 5 - tol)
        ):
            return "top"
        return "side"

    mesh = sm.carve((xs, ys, zs), volume, surface)
    path = sm.write_msh22(Path(directory) / f"stepped_line_{n}.msh", mesh)
    return _line_model(gfem.Mesh.from_gmsh(path, length_unit="mm"), ("lsub", "lossy"))


def _line_model(mesh, lossy_regions):
    plain = gfem.Material(eps_r=4.0)
    lossy = gfem.Material(eps_r=4.0, tan_delta=TAN)
    materials = {r: lossy if r in lossy_regions else plain for r in mesh.volumes}
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def half_width_guide_model(directory: Path, n: int) -> gfem.Model:
    """The half-width guide on a uniform mesh of ``n`` cells per mm."""
    edges = (
        np.linspace(0, 4.0, 4 * n + 1),
        np.linspace(0, 2.0, 2 * n + 1),
        np.linspace(0, 0.8, max(2, round(0.8 * n)) + 1),
    )

    def volume(c):
        if 1.5 < c[0] < 2.5 and c[1] < 1.0:
            return "block"
        if 1.0 < c[1] < 1.5 and 3.0 < c[0] < 3.5:
            return "rfar"
        if 1.0 < c[1] < 1.5 and 2.5 < c[0] < 3.0:
            return "redge"
        return "fill"

    mesh = sm.structured(edges, volume, _guide_surface(4.0, 2.0, 0.8))
    path = sm.write_msh22(Path(directory) / f"half_width_guide_{n}.msh", mesh)
    return _guide_model(gfem.Mesh.from_gmsh(path, length_unit="mm"), ("block",))


def edge_box_line_model(directory: Path, n: int) -> gfem.Model:
    """The edge-box line on a uniform mesh of ``n`` cells per mm."""
    xs = np.union1d(sm._graded(0, 10, 5, n, 1.0), [2.5, 7.5])
    zs = sm._graded(0, 1, 0.5, n, 1.0)

    def volume(c):
        x, _, z = c
        if x > 5 and z > 0.5:
            return None
        if 5 < x < 7.5 and z < 0.5:
            return "lossy"
        if 2.5 < x < 5 and z > 0.5:
            return "rbox"
        return "dielectric"

    def surface(c):
        x, _, z = c
        tol = 1e-9
        if abs(x) < tol:
            return "port1"
        if abs(x - 10) < tol:
            return "port2"
        if abs(z) < tol:
            return "bottom"
        if (
            abs(z - 1) < tol
            or (abs(x - 5) < tol and z > 0.5 - tol)
            or (abs(z - 0.5) < tol and x > 5 - tol)
        ):
            return "top"
        return "side"

    mesh = sm.carve((xs, np.array([0.0, 2.0]), zs), volume, surface)
    path = sm.write_msh22(Path(directory) / f"edge_box_line_{n}.msh", mesh)
    return _line_model(gfem.Mesh.from_gmsh(path, length_unit="mm"), ("lossy",))
