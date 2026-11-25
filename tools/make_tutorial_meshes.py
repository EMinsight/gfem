"""Build the meshes of the docs tutorials (no gmsh needed).

Usage: python tools/make_tutorial_meshes.py docs/tutorials/data

``line.msh`` (mm): a parallel-plate line along x, 40 mm long, 6 mm wide and
1 mm high. Volumes "feed" (x < 10 and x > 30) and "design" (10 < x < 30);
surfaces "bottom" / "top" (the plates, z = 0 / 1), "port1" (x = 0), "port2"
(x = 40) and "side" (y walls).

``sheet_line.msh`` (mm): a parallel-plate line along x, 40 mm long, plates
"plate1" / "plate2" at y = 0 / 6 (the TEM field is along y), 2 mm high, glued
from the volumes "lower" and "upper" at z = 1; the interior surface "sheet"
between them spans the design length 10 < x < 30. Surfaces "port1" (x = 0),
"port2" (x = 40) and "side" (z walls).

``cpw.msh`` (um): a conductor-backed coplanar line along x, 1500 um long.
Substrate 0 < z < 100 (volume "substrate"), metal 100 < z < 105 (volumes
"signal_metal", |y| < 25, and "ground_metal", |y| > 55, to be excluded),
air above and in the 30 um gaps (volume "air", up to z = 300), |y| < 300.
Surfaces: "signal_wall" (the signal's two sidewalls, y = +-25), "signal"
(its other faces), "ground" (the ground planes' faces), "substrate_top"
(substrate under the gaps), "backside" (z = 0), "top" (z = 300),
"side" (y = +-300), "ends" (x = 0 and 1500 outside the gaps), and the
port gaps "p1a" / "p1b" (x = 0, y > 0 / y < 0) and "p2a" / "p2b"
(x = 1500).

``bend.msh`` (um): a slab model of a 90 degree waveguide bend, x and y in
[-2.5, 2.5] and 0 < z < 0.1 (one cell thick). Volumes "design" (|x|, |y| <
1.5), "core" (the input arm x < -1.5, |y| <= 0.275, and the output arm
y < -1.5, |x| <= 0.275) and "cladding" (the rest). Surfaces "plates"
(z = 0 and 0.1), "port1" (x = -2.5, |y| <= 1.1), "port2" (y = -2.5,
|x| <= 1.1) and "outer" (every other side face). ``bend_7um.msh`` (um): the
same in a 7 x 7 um box, x and y in [-3.5, 3.5]; inside |x|, |y| < 2.5 its
nodes and tetrahedra are those of ``bend.msh``.

``line2d.msh``, ``bend2d.msh`` and ``bend2d_7um.msh``: triangle-only meshes,
the z = 0 face of ``line.msh``, ``bend.msh`` and ``bend_7um.msh`` node for
node. Areas carry the volume names; lines on the boundary carry the side
names: "port1", "port2" and "side" for the line, "port1", "port2" and "outer"
for the bend.

``pml_line.msh`` (um): a TEM line along z, 0.2 x 0.2 um, 1.5 um of air
(volume "air", z < 1.5) closed by 0.5 um of perfectly matched layer (volume
"pml", 1.5 < z < 2), one cell across and ten per um along z, so the layer's
inner face is a grid plane. Surfaces "port" (z = 0) and "plates" (the y
walls); the x walls and the layer's back face (z = 2) stay unnamed.

``antenna_box.msh`` (um): the unit cube of air, volumes "left" (x < 0.5)
and "right", wrapped in 1/3 um of perfectly matched layer (volume "pml") on
every side, six cells per um. A strip in the plane x = 0.5 between them,
0.5 < y < 2/3: its feed "feed" (1/2 < z < 2/3) and its arms "wire" (the rest
of 1/6 < z < 5/6), a strip dipole two thirds of a wavelength long at 1 um.

``metalens_cell.msh`` (um): one period of a row of silicon nitride ridges on
glass, lines along y: x in [0, 0.3], y in [0, 0.16], z in [0, 1.55]. Volumes
"glass" (z < 0.2), "ridge" (0.075 < x < 0.225, 0.2 < z < 1.3) and "air".
Surfaces "bottom" (z = 0) and "top" (z = 1.55), the periodic faces "x_min" /
"x_max" and "y_min" / "y_max", the ridge's walls "sidewall", the plane
"base" (z = 0.2, under the ridge and the air beside it) and the ridge's top
"ridge_top"; the last two are named so that a boundary morph of "sidewall"
keeps those planes where they are. Cells of 0.0375 um across and 0.05 um
high, two along y.

``metalens_supercell.msh`` (um): ten of those cells side by side, x in
[0, 3], with the same names and ridges of the same width.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import structured_meshes as sm

LENGTH, WIDTH, HEIGHT = 40.0, 6.0, 1.0
DESIGN = (10.0, 30.0)


def line():
    """The line mesh as `structured_meshes.structured` returns it."""
    edges = (
        np.linspace(0, LENGTH, 81),
        np.linspace(0, WIDTH, 7),
        np.linspace(0, HEIGHT, 2),
    )
    eps = 1e-9 * LENGTH

    def volume(c):
        return "design" if DESIGN[0] < c[0] < DESIGN[1] else "feed"

    def surface(c):
        if abs(c[0]) < eps:
            return "port1"
        if abs(c[0] - LENGTH) < eps:
            return "port2"
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - HEIGHT) < eps:
            return "top"
        if abs(c[1]) < eps or abs(c[1] - WIDTH) < eps:
            return "side"
        return ""  # the design / feed interfaces carry no boundary

    return sm.structured(edges, volume, surface)


def sheet_line():
    """The sheet line mesh as `structured_meshes.structured` returns it."""
    return sm.sheet_line(LENGTH, WIDTH, 2.0, (40, 6, 1), span=DESIGN)


CPW_LENGTH = 1500.0
CPW_SIGNAL, CPW_GAP = 25.0, 30.0  # signal half-width, gap (um)
CPW_METAL = (100.0, 105.0)


def cpw():
    """The coplanar line mesh as `structured_meshes.structured` returns it."""
    s, g = CPW_SIGNAL, CPW_SIGNAL + CPW_GAP
    half = (0.0, s, 40.0, g, 120.0, 300.0)
    edges = (
        np.linspace(0, CPW_LENGTH, 9),
        np.array(sorted({-y for y in half} | set(half))),
        np.array([0.0, 100.0, 105.0, 160.0, 300.0]),
    )
    eps = 1e-6
    z0, z1 = CPW_METAL

    def volume(c):
        if c[2] < z0:
            return "substrate"
        if c[2] < z1:
            if abs(c[1]) < s:
                return "signal_metal"
            if abs(c[1]) > g:
                return "ground_metal"
        return "air"

    def surface(c):  # noqa: PLR0911
        x, y, z = c
        in_metal = z0 - eps < z < z1 + eps
        if in_metal and abs(abs(y) - s) < eps and z0 + eps < z < z1 - eps:
            return "signal_wall"
        if in_metal and abs(y) < s + eps:
            return "signal"
        if in_metal and abs(y) > g - eps:
            return "ground"
        if abs(z - z0) < eps:
            return "substrate_top"
        if abs(x) < eps or abs(x - CPW_LENGTH) < eps:
            if z0 < z < z1 and s < abs(y) < g:
                end = "p1" if x < eps else "p2"
                return end + ("a" if y > 0 else "b")
            return "ends"
        if abs(z) < eps:
            return "backside"
        if abs(z - 300.0) < eps:
            return "top"
        return "side"

    return sm.structured(edges, volume, surface)


BEND_CORE, BEND_DESIGN, BEND_PORT = 0.275, 1.5, 1.1  # half-widths (um)
BEND_H = 0.1  # cell size and slab thickness (um)


def bend(half=2.5, h=BEND_H):
    """The bend mesh in |x|, |y| <= ``half``, as `structured` returns it.

    ``h`` is the cell size and the slab thickness (um); the reference tier's
    fine 2D mesh is the z = 0 face of a smaller one.
    """
    # +-2.5 stays a grid line so a larger box keeps the 5 x 5 um mesh inside
    lines = {BEND_CORE, BEND_PORT, BEND_DESIGN, 2.5, half}
    pts = sorted({-x for x in lines} | lines)
    xy = np.concatenate(
        [[pts[0]]]
        + [
            np.linspace(a, b, math.ceil((b - a) / h - 1e-9) + 1)[1:]
            for a, b in itertools.pairwise(pts)
        ]
    )
    edges = (xy, xy, np.array([0.0, h]))
    eps = 1e-9

    def volume(c):
        x, y, _ = c
        if abs(x) < BEND_DESIGN and abs(y) < BEND_DESIGN:
            return "design"
        if (x < -BEND_DESIGN and abs(y) <= BEND_CORE) or (
            y < -BEND_DESIGN and abs(x) <= BEND_CORE
        ):
            return "core"
        return "cladding"

    def surface(c):
        x, y, z = c
        if abs(z) < eps or abs(z - h) < eps:
            return "plates"
        if abs(x + half) < eps:
            return "port1" if abs(y) < BEND_PORT else "outer"
        if abs(y + half) < eps:
            return "port2" if abs(x) < BEND_PORT else "outer"
        if abs(x - half) < eps or abs(y - half) < eps:
            return "outer"
        if abs(x) < half - eps and abs(y) < half - eps and eps < z < h - eps:
            return ""  # the region interfaces carry no boundary
        raise ValueError(f"unclassified exterior face at {c}")

    return sm.structured(edges, volume, surface)


PML_SECTION, PML_AIR, PML_THICKNESS = 0.2, 1.5, 0.5


def pml_line():
    """The perfectly matched layer recipe's line, as `structured` returns it."""
    total = PML_AIR + PML_THICKNESS
    edges = (
        np.linspace(0, PML_SECTION, 2),
        np.linspace(0, PML_SECTION, 2),
        np.linspace(0, total, round(total * 10) + 1),
    )
    eps = 1e-9

    def volume(c):
        return "air" if c[2] < PML_AIR else "pml"

    def surface(c):
        if abs(c[2]) < eps:
            return "port"
        if abs(c[1]) < eps or abs(c[1] - PML_SECTION) < eps:
            return "plates"
        return ""

    return sm.structured(edges, volume, surface)


ANTENNA_CELLS, ANTENNA_THICKNESS = 6, 2 / 6


def antenna_box():
    """The far-field recipe's strip dipole in a box, as `structured` returns it."""
    edges = tuple(
        np.linspace(
            -ANTENNA_THICKNESS,
            1 + ANTENNA_THICKNESS,
            round((1 + 2 * ANTENNA_THICKNESS) * ANTENNA_CELLS) + 1,
        )
        for _ in range(3)
    )
    eps = 1e-9

    def volume(c):
        if not ((c > 0) & (c < 1)).all():
            return "pml"
        return "left" if c[0] < 0.5 else "right"

    def surface(c):
        if abs(c[0] - 0.5) > eps or not 0.5 < c[1] < 4 / 6:
            return ""
        if 3 / 6 < c[2] < 4 / 6:
            return "feed"
        return "wire" if 1 / 6 < c[2] < 5 / 6 else ""

    return sm.structured(edges, volume, surface)


METALENS_PITCH, METALENS_WIDTH, METALENS_Y = 0.3, 0.15, 0.16
METALENS_GLASS, METALENS_HEIGHT, METALENS_AIR = 0.2, 1.1, 0.25


def _metalens(cells):
    """``cells`` ridges of the metalens cell side by side, as `structured` returns."""
    a, w = METALENS_PITCH, METALENS_WIDTH
    glass, height = METALENS_GLASS, METALENS_HEIGHT
    one = np.concatenate(
        [
            np.linspace(0, a / 2 - w / 2, 3),
            np.linspace(a / 2 - w / 2, a / 2 + w / 2, 5)[1:],
            np.linspace(a / 2 + w / 2, a, 3)[1:],
        ]
    )
    ex = np.concatenate([one] + [j * a + one[1:] for j in range(1, cells)])
    ez = np.concatenate(
        [
            np.linspace(0, glass, 5),
            np.linspace(glass, glass + height, 23)[1:],
            np.linspace(glass + height, glass + height + METALENS_AIR, 6)[1:],
        ]
    )
    edges = (ex, np.linspace(0, METALENS_Y, 3), ez)
    lengths = (cells * a, METALENS_Y, ez[-1])
    eps = 1e-9

    def volume(c):
        j = min(int(c[0] // a), cells - 1)
        in_ridge = abs(c[0] - (j + 0.5) * a) < w / 2 and c[2] < glass + height
        return "glass" if c[2] < glass else ("ridge" if in_ridge else "air")

    def surface(c):
        for axis, length in enumerate(lengths):
            if abs(c[axis]) < eps:
                return ("x_min", "y_min", "bottom")[axis]
            if abs(c[axis] - length) < eps:
                return ("x_max", "y_max", "top")[axis]
        if abs(c[2] - glass) < eps:
            return "base"
        return "ridge_top" if abs(c[2] - glass - height) < eps else "sidewall"

    return sm.structured(edges, volume, surface)


def metalens_cell():
    """One ridge of the metalens, as `structured_meshes.structured` returns it."""
    return _metalens(1)


def metalens_supercell():
    """Ten ridges side by side, as `structured_meshes.structured` returns it."""
    return _metalens(10)


def write(out: Path) -> list[Path]:
    """Write every tutorial mesh into the directory ``out``; return the files."""
    out.mkdir(parents=True, exist_ok=True)
    return [
        sm.write_msh22(out / "line.msh", line()),
        sm.write_msh22(out / "cpw.msh", cpw()),
        sm.write_msh22(out / "sheet_line.msh", sheet_line()),
        sm.write_msh22(out / "bend.msh", bend()),
        sm.write_msh22(out / "bend_7um.msh", bend(half=3.5)),
        sm.write_msh22(out / "line2d.msh", sm.z0_face(line())),
        sm.write_msh22(out / "bend2d.msh", sm.z0_face(bend())),
        sm.write_msh22(out / "bend2d_7um.msh", sm.z0_face(bend(half=3.5))),
        sm.write_msh22(out / "pml_line.msh", pml_line()),
        sm.write_msh22(out / "antenna_box.msh", antenna_box()),
        sm.write_msh22(out / "metalens_cell.msh", metalens_cell()),
        sm.write_msh22(out / "metalens_supercell.msh", metalens_supercell()),
    ]


if __name__ == "__main__":
    for written in write(Path(sys.argv[1])):
        print(written)
