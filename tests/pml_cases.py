"""Meshes with perfectly matched layers shared by the layer tests.

`box` writes the unit cube [0, 1]^3 ("air") wrapped in layer cells on the
named sides, on the structured Kuhn grid of `structured_meshes`;
`tem_line` and `te10_guide` end a straight guide with a PEC-backed layer,
layered along z, for the closed-form reflection cases, whose measured
floors and errors are recorded next to their gates. Every builder takes
the file to write; nothing here solves.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structured_meshes as sm

#: The six sides of the interior box, as they are written in messages.
SIDES = ("-x", "+x", "-y", "+y", "-z", "+z")


def _edges(sides: tuple[str, ...], thickness: float, cells: int) -> tuple:
    out = []
    for axis in "xyz":
        lo = -thickness if f"-{axis}" in sides else 0.0
        hi = 1.0 + thickness if f"+{axis}" in sides else 1.0
        out.append(np.linspace(lo, hi, round((hi - lo) * cells) + 1))
    return tuple(out)


def box(
    path: str | Path,
    sides: tuple[str, ...] = ("+x",),
    *,
    thickness: float = 0.5,
    cells: int = 2,
    volume: Callable[[np.ndarray], str | None] | None = None,
    surface: Callable[[np.ndarray], str] | None = None,
    mapping: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Path:
    """The unit cube "air" with layer cells "pml" beyond the given sides.

    Args:
        path: The gmsh 2.2 file to write.
        sides: Sides of the cube that carry a layer (`SIDES`); the edge and
            corner cells between two layer sides are layer cells too.
        thickness: The layer thickness, a multiple of ``1 / cells``.
        cells: Cells per unit length.
        volume: Region name of a cell from its centroid, ``None`` to remove
            the cell; by default "air" inside the cube and "pml" outside.
        surface: Name of a boundary face from its centroid, ``""`` for the
            default: "outer" on the grid's bounding box, nothing elsewhere.
        mapping: Maps grid points ``(n, 3)`` to coordinates; the
            classifiers see grid points.

    Returns:
        ``path``.
    """
    edges = _edges(tuple(sides), thickness, cells)
    lo = np.array([e[0] for e in edges])
    hi = np.array([e[-1] for e in edges])
    eps = 1e-9

    def default_volume(c):
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    def named(c):
        name = surface(c) if surface is not None else ""
        if name:
            return name
        if (np.abs(c - lo) < eps).any() or (np.abs(c - hi) < eps).any():
            return "outer"
        return ""

    mesh = sm.carve(edges, volume or default_volume, named, mapping)
    return sm.write_msh22(path, mesh)


# --- closed-form cases ----------------------------------------------------------

#: Speed of light, m/s.
C0 = 299792458.0
#: The cases' vacuum wavelength, um (the mesh unit), and its frequency, Hz.
WAVELENGTH = 1.0
F0 = C0 / (WAVELENGTH * 1e-6)
#: Air before the layer and the layer's thickness, along z (um).
AIR_LENGTH, THICKNESS = 1.5, 0.5
#: The TEM line's square section and the TE10 guide's (um).
LINE_SECTION = (0.2, 0.2)
GUIDE_SECTION = (0.75, 0.3)

#: Reflection floor of a PEC-backed layer on these layered meshes at order
#: 3, h = lambda / 10: measured 9.3e-7 for R0 = 1e-6 (the TEM line,
#: normal incidence; 4.3e-6 for the steeper R0 = 1e-8). A closed form is
#: asserted only where it sits 100 times above it.
FLOOR = 1e-6
#: TE10 floor at h = 1/14 with the reference frequency where k0 = beta:
#: measured 7.2e-5 (h = 0.1 in the raw probe), so V3c asserts 2e-4.
GUIDE_FLOOR = 2e-4
#: Relative gate on a closed-form reflection above its floor. Measured
#: through gfem (order 3, one thread), relative to the closed form: TEM
#: R0 = 1e-2: 4.8e-5; R0 = 1e-3 at f_ref: 6.0e-4 (and 1.5 f_ref: 2.9e-5
#: against 3.2e-5, 2.7e-6 off, under the 3-floor bound); pinned f_ref = 2f:
#: 6.0e-4; eps_r = 2.25: 3.1e-3; TE10 at lambda = 1: 2.2e-5; near cutoff:
#: 8.1e-5; f_ref where k0 = beta: |S11| = 3.2e-6.
REL = 1e-2


def _section_axis(n: float, length: float) -> np.ndarray:
    return np.linspace(0.0, length, max(1, round(length * n)) + 1)


def waveguide(
    path: str | Path,
    section: tuple[float, float],
    *,
    cells: float = 10,
    lateral: float | None = None,
    pec_x: bool = True,
) -> Path:
    """A straight guide along z ended by a PEC-backed layer: "air" then "pml".

    The guide spans ``section`` in x and y, ``AIR_LENGTH`` of "air" from
    z = 0 and ``THICKNESS`` of "pml" beyond. Surfaces: "port" at z = 0,
    "pec" on the y walls, on the x walls when ``pec_x`` (else they stay
    unnamed: PMC), and on the back face of the layer.

    Args:
        path: The gmsh 2.2 file to write.
        section: ``(a, b)``, um.
        cells: Cells per um along z (every z grid plane at a multiple of
            ``1 / cells``, so the layer's face is one of them).
        lateral: Cells per um across the section; ``cells / 2`` by default,
            as the floors were measured (the fields vary slowly across).
        pec_x: Whether the x walls are PEC.

    Returns:
        ``path``.
    """
    a, b = section
    lateral = cells / 2 if lateral is None else lateral
    total = AIR_LENGTH + THICKNESS
    edges = (
        _section_axis(lateral, a),
        _section_axis(lateral, b),
        np.linspace(0.0, total, round(total * cells) + 1),
    )
    eps = 1e-9

    def volume(c):
        return "air" if c[2] < AIR_LENGTH else "pml"

    def surface(c):
        if abs(c[2]) < eps:
            return "port"
        if abs(c[2] - total) < eps or abs(c[1]) < eps or abs(c[1] - b) < eps:
            return "pec"
        if pec_x and (abs(c[0]) < eps or abs(c[0] - a) < eps):
            return "pec"
        return ""

    return sm.write_msh22(path, sm.structured(edges, volume, surface))


def tem_line(path: str | Path, *, cells: float = 10) -> Path:
    """The TEM line: PEC y walls, PMC x walls (a plane wave E_y), `waveguide`."""
    return waveguide(path, LINE_SECTION, cells=cells, pec_x=False)


def te10_guide(path: str | Path, *, cells: float = 14) -> Path:
    """The TE10 guide (a = 0.75, b = 0.3 um, PEC walls), `waveguide`."""
    return waveguide(path, GUIDE_SECTION, cells=cells)


#: The dielectric slab of `slab_line`, along z (um).
SLAB = (0.5, 1.0)


def slab_line(path: str | Path, *, cells: float = 10) -> Path:
    """The TEM line with a "slab" region in its air, for gradients.

    As `tem_line`, two cells across the section, with the tetrahedra
    between z = ``SLAB[0]`` and ``SLAB[1]`` in "slab" and its face at
    z = ``SLAB[0]`` named "front" (an interface for `boundary_morph`).

    Args:
        path: The gmsh 2.2 file to write.
        cells: Cells per um along z (multiples of 1 / ``cells`` must hit the
            slab's faces and the layer's).

    Returns:
        ``path``.
    """
    a, b = LINE_SECTION
    total = AIR_LENGTH + THICKNESS
    edges = (
        np.linspace(0.0, a, 3),
        np.linspace(0.0, b, 3),
        np.linspace(0.0, total, round(total * cells) + 1),
    )
    eps = 1e-9

    def volume(c):
        if c[2] > AIR_LENGTH:
            return "pml"
        return "slab" if SLAB[0] < c[2] < SLAB[1] else "air"

    def surface(c):
        if abs(c[2]) < eps:
            return "port"
        if abs(c[2] - SLAB[0]) < eps:
            return "front"
        if abs(c[2] - total) < eps or abs(c[1]) < eps or abs(c[1] - b) < eps:
            return "pec"
        return ""

    return sm.write_msh22(path, sm.structured(edges, volume, surface))


# --- the Bloch cell ---------------------------------------------------------------

#: The Bloch cell's lateral period along x and y (um, 0.4 lambda) and its
#: depth of air (or of the filling) between the layer and the Floquet face.
CELL_PERIOD = 0.4
CELL_DEPTH = 0.5
#: Cells per um along z: lambda / 10 in vacuum, lambda / 9.3 in the
#: eps_r = 2.25 filling (the layer's face must stay a grid plane).
CELL_CELLS = {1.0: 10, 2.25: 14}
#: Reflection floor of the Bloch cell at order 3, 4 x 4 lateral cells,
#: `CELL_CELLS`: measured |S| = 4.3e-7 for R0 = 1e-6 at normal incidence in
#: vacuum (6.9e-7 in eps_r = 2.25, where the continuum value is 1e-9), and
#: |S - R0^cos| up to 2.6e-6 for R0 = 1e-6 at 60 degrees (5.4e-6 for the
#: steeper R0 = 1e-8). The lateral Bloch phase leaves the TEM line's floor
#: in place. With R0 = 1e-2 the measured errors are at most 6.9e-7
#: (2.6e-4 relative, eps_r = 2.25 at normal incidence) and the
#: cross-polarized entries at most 2.5e-7; every closed form there
#: (1e-3 to 0.1) sits more than 100 times above this floor.
CELL_FLOOR = 3e-6


def bloch_cell(path: str | Path, *, cells: float = 10, lateral: int = 4) -> Path:
    """A periodic cell over a PEC-backed layer: "pml" below z = 0, "air" above.

    The cell is ``CELL_PERIOD`` wide along x and y (paired faces "x_min" /
    "x_max", "y_min" / "y_max"), with ``THICKNESS`` of "pml" from z = 0 down
    to the PEC face "z_min" and ``CELL_DEPTH`` of "air" up to the open face
    "z_max". The layer spans the whole period, so it stretches along z only.

    Args:
        path: The gmsh 2.2 file to write.
        cells: Cells per um along z (the layer's face is a grid plane).
        lateral: Cells across the period, along x and y.

    Returns:
        ``path``.
    """
    side = np.linspace(0.0, CELL_PERIOD, lateral + 1)
    z = np.linspace(-THICKNESS, CELL_DEPTH, round((THICKNESS + CELL_DEPTH) * cells) + 1)

    def volume(c):
        return "air" if c[2] > 0 else "pml"

    return sm.write_msh22(path, sm.periodic_cell((side, side, z), volume, paired="xy"))


# --- the leaky guide (eigenmodes) ---------------------------------------------------

#: The leaky guide, mm: a TE10 guide (PEC walls, ``GUIDE_WIDTH`` wide,
#: ``GUIDE_HEIGHT`` high) shorted at x = 0, filled with eps_r = ``GUIDE_EPS``
#: up to ``GUIDE_CORE``, then ``GUIDE_TAIL`` of vacuum, then a PEC-backed
#: layer ``GUIDE_LAYER`` thick.
GUIDE_CORE, GUIDE_TAIL, GUIDE_LAYER = 3.0, 3.0, 4.0
GUIDE_WIDTH, GUIDE_HEIGHT, GUIDE_EPS = 10.0, 1.0, 10.0
#: Cells per mm along x and across y (h = 1 mm, 5k DOFs at order 2).
GUIDE_CELLS = (1.0, 1.0)
#: Relative gate on the leaky guide's k at order 2 (measured 4.3e-4, plain
#: and Bloch) and on its Q (measured 0.52 % high).
GUIDE_K_REL, GUIDE_Q_REL = 2e-3, 1e-2
#: How far (relative) a returned mode may move when the target reflection
#: drops from 1e-6 to 1e-9: measured 4.3e-4 at most over the four modes
#: nearest the leaky one; a mode of the layer moves 3e-2 and more.
GUIDE_MOVE = 5e-3
#: The guide's leaky resonance under a perfect match, rad/mm: the root of
#: :math:`b_1\cos(b_1 a) + i b_2\sin(b_1 a) = 0`, with
#: :math:`b_1^2 = \varepsilon k^2 - k_t^2`, :math:`b_2^2 = k^2 - k_t^2`,
#: :math:`k_t = \pi / W` (24.128 GHz, Q = 9.453).
GUIDE_ROOT = 0.505676 + 0.026784j


def guide_root(k_guess: complex = GUIDE_ROOT) -> complex:
    """`GUIDE_ROOT` to round-off, by Newton from ``k_guess`` (rad/mm)."""
    import cmath

    kt = np.pi / GUIDE_WIDTH

    def residual(k):
        b1 = cmath.sqrt(k * k * GUIDE_EPS - kt * kt)
        b2 = cmath.sqrt(k * k - kt * kt)
        return b1 * cmath.cos(b1 * GUIDE_CORE) + 1j * b2 * cmath.sin(b1 * GUIDE_CORE)

    k = complex(k_guess)
    for _ in range(60):
        h = 1e-7 * abs(k)
        step = residual(k) / ((residual(k + h) - residual(k - h)) / (2 * h))
        k -= step
        if abs(step) < 1e-14 * abs(k):
            break
    return k


def leaky_guide_model(
    path: str | Path,
    *,
    bloch: bool = False,
    target_reflection: float = 1e-6,
    write: bool = True,
):
    r"""The model of `leaky_guide`'s mesh at ``path`` (written there first).

    ``write=False`` reads a mesh already written, as ranks do: one process
    writes it before they start.

    Returns:
        The model and the Bloch wave vector (rad/m) its solves take, ``None``
        without ``bloch``: :math:`k_y = \pi / W`, so the TE10 field across
        the width is the plain guide's.
    """
    import gfem

    if write:
        leaky_guide(path, bloch=bloch)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    boundaries = [gfem.PEC("short"), gfem.PEC("pec")]
    wave_vector = None
    if bloch:
        boundaries.append(gfem.Periodic(("y_min", "y_max")))
        wave_vector = (0.0, np.pi / (GUIDE_WIDTH * 1e-3), 0.0)
    air = gfem.Material()
    model = gfem.Model(
        mesh,
        materials={"core": gfem.Material(eps_r=GUIDE_EPS), "tail": air, "pml": air},
        boundaries=boundaries,
        pml=gfem.PML("pml", target_reflection=target_reflection),
    )
    return model, wave_vector


def leaky_guide(path: str | Path, *, bloch: bool = False) -> Path:
    """The leaky guide along x: "core", "tail", then "pml" (mm).

    Surfaces: "short" at x = 0, "pec" on the layer's back face and the z
    walls, and on the y walls unless ``bloch``: then the y walls are the
    paired faces "y_min" / "y_max", for a Bloch phase across the width.

    Returns:
        ``path``.
    """
    length = GUIDE_CORE + GUIDE_TAIL + GUIDE_LAYER
    along, across = GUIDE_CELLS
    edges = (
        np.linspace(0.0, length, round(length * along) + 1),
        np.linspace(0.0, GUIDE_WIDTH, round(GUIDE_WIDTH * across) + 1),
        np.linspace(0.0, GUIDE_HEIGHT, 2),
    )
    eps = 1e-9

    def volume(c):
        if c[0] < GUIDE_CORE:
            return "core"
        return "tail" if c[0] < GUIDE_CORE + GUIDE_TAIL else "pml"

    def surface(c):
        if abs(c[0]) < eps:
            return "short"
        on_z = abs(c[2]) < eps or abs(c[2] - GUIDE_HEIGHT) < eps
        if abs(c[0] - length) < eps or on_z:
            return "pec"
        if not bloch and (abs(c[1]) < eps or abs(c[1] - GUIDE_WIDTH) < eps):
            return "pec"
        return ""

    if bloch:
        mesh = sm.periodic_cell(edges, volume, surface=surface, paired="y")
    else:
        mesh = sm.structured(edges, volume, surface)
    return sm.write_msh22(path, mesh)


# --- a strip guide through side layers ----------------------------------------------

#: The strip guide's vacuum wavelength (um), core and cladding indices, and
#: the core's width (x) and height (y), um.
STRIP_WAVELENGTH = 1.55
STRIP_F = C0 / (STRIP_WAVELENGTH * 1e-6)
STRIP_CORE, STRIP_CLADDING = 2.0, 1.45
STRIP_SIZE = (1.0, 0.4)
#: Its side layers' thickness, um.
STRIP_THICKNESS = 0.6
#: The fundamental's effective index on a wide closed section: 1.6429163,
#: measured on an unstructured order 2 mesh with PEC walls 2.5 um from the
#: core edge.
STRIP_NEFF = 1.6429


def _segments(*parts: tuple[float, float]) -> np.ndarray:
    """Grid lines over consecutive ``(length, cell size)`` segments from 0."""
    out, start = [np.zeros(1)], 0.0
    for length, h in parts:
        n = max(1, math.ceil(length / h - 1e-9))
        out.append(start + np.linspace(0.0, length, n + 1)[1:])
        start += length
    return np.concatenate(out)


def strip_guide(
    path: str | Path,
    clearance: float,
    *,
    thickness: float = STRIP_THICKNESS,
    length: float = 1.0,
    quarter: bool = True,
    core_h: float = 0.1,
    clad_h: float = 0.15,
) -> Path:
    """A straight strip guide along z with side layers on +-x, +-y and their edges.

    The core ("core", `STRIP_SIZE`) is centred on the z axis, the cladding
    ("clad") reaches ``clearance`` beyond each core edge, and the layer
    ("pml", ``thickness``; none when 0) wraps it. Surfaces: "p1" at z = 0,
    "p2" at z = ``length``, "pec" on the outer faces. A ``quarter`` model
    keeps x, y >= 0 with "pec" on x = 0 and y = 0 unnamed (PMC): the
    E_x-even fundamental.

    Args:
        path: The gmsh 2.2 file to write.
        clearance: Cladding between the core edge and the layer, um.
        thickness: The layer thickness, um.
        length: The guide's length, um.
        quarter: Whether to keep the quarter x, y >= 0.
        core_h: Cell size in the core, um.
        clad_h: Cell size in the cladding and the layer, um.

    Returns:
        ``path``.
    """
    w, t = STRIP_SIZE
    axes = []
    for half in (w / 2, t / 2):
        parts = [(half, core_h), (clearance, clad_h)]
        if thickness:
            parts.append((thickness, clad_h))
        upper = _segments(*parts)
        axes.append(upper if quarter else np.concatenate([-upper[::-1], upper[1:]]))
    z = _segments((length, clad_h))
    edges = (axes[0], axes[1], z)
    outer = (axes[0][-1], axes[1][-1])
    eps = 1e-9

    def volume(c):
        x, y = abs(c[0]), abs(c[1])
        if x > w / 2 + clearance or y > t / 2 + clearance:
            return "pml"
        return "core" if x < w / 2 and y < t / 2 else "clad"

    def surface(c):
        if abs(c[2]) < eps:
            return "p1"
        if abs(c[2] - length) < eps:
            return "p2"
        if abs(abs(c[0]) - outer[0]) < eps or abs(abs(c[1]) - outer[1]) < eps:
            return "pec"
        if quarter and abs(c[0]) < eps:
            return "pec"
        return ""

    return sm.write_msh22(path, sm.structured(edges, volume, surface))


def strip_model(path: str | Path, pml: object = "default", **kw: object):
    """The `gfem.Model` of a `strip_guide` mesh, wave ports "p1" and "p2".

    ``pml`` is the layer (by default ``gfem.PML("pml")``); a mesh without
    a layer takes none.
    """
    import gfem

    mesh = gfem.Mesh.from_gmsh(path)
    materials = {
        "core": gfem.Material(eps_r=STRIP_CORE**2),
        "clad": gfem.Material(eps_r=STRIP_CLADDING**2),
    }
    if "pml" in mesh.volumes:
        materials["pml"] = materials["clad"]
        pml = gfem.PML("pml") if pml == "default" else pml
    else:
        pml = None
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "p1"), gfem.WavePort("p2", "p2")],
        pml=pml,
        **kw,
    )
