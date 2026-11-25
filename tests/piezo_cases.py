"""A small X-cut lithium niobate transducer section, shared by the piezoelectric tests.

Two interdigital pairs on a lithium niobate half-space with a damping frame on
three sides (lengths in micrometres, read by `gfem.Mesh.from_gmsh`). Two
electrode variants of the same layout:

- ``"lines"``: zero-thickness electrodes, the named lines ``"hot"`` and
  ``"gnd"`` on the surface;
- ``"meshed"``: aluminium fingers of thickness `T_AL`, the areas ``"hot"`` and
  ``"gnd"``, which carry displacement and mass with the potential fixed on them.

Warner's lithium niobate (Pa, C/m^2) in the crystal frame; the lab axes are the
X-cut ones, ``AXES``: SAW along crystal Y, film normal along crystal X.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import ngsolve as ngs
import numpy as np
import thermal_cases as tc
from scipy import constants

import gfem

GPa = 1e9
_C11, _C12, _C13, _C14, _C33, _C44 = 203, 53, 75, 9, 245, 60
C_LN = GPa * np.array(
    [
        [_C11, _C12, _C13, _C14, 0, 0],
        [_C12, _C11, _C13, -_C14, 0, 0],
        [_C13, _C13, _C33, 0, 0, 0],
        [_C14, -_C14, 0, _C44, 0, 0],
        [0, 0, 0, 0, _C44, _C14],
        [0, 0, 0, 0, _C14, (_C11 - _C12) / 2],
    ]
)
_E15, _E22, _E31, _E33 = 3.7, 2.5, 0.2, 1.3
E_LN = np.array(
    [
        [0, 0, 0, 0, _E15, -_E22],
        [-_E22, _E22, 0, _E15, 0, 0],
        [_E31, _E31, _E33, 0, 0, 0],
    ]
)
EPS_LN = (44.0, 44.0, 29.0)
RHO_LN = 4700.0
AXES = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
EPS0 = constants.epsilon_0
QM = 200.0
E_AL, NU_AL, RHO_AL = 70e9, 0.35, 2700.0

#: The section's layout (micrometres): period, pairs, finger width, side margin,
#: frame thickness, depth below the surface, air above, finger thickness.
PERIOD, PAIRS, FINGER, MARGIN = 4.0, 2, 1.0, 4.0
FRAME, DEPTH, AIR, T_AL = 6.0, 8.0, 2.0, 0.2
FREQUENCY = 0.95e9


def fingers() -> list[tuple[float, float, str]]:
    """``(x0, x1, name)`` of every finger, alternating hot and gnd."""
    x0 = -PAIRS * PERIOD / 2
    out = []
    for k in range(2 * PAIRS):
        a = x0 + k * PERIOD / 2 + PERIOD / 8
        out.append((a, a + FINGER, "hot" if k % 2 == 0 else "gnd"))
    return out


def mesh(path: Path, variant: str = "lines", maxh: float = 1.2) -> gfem.Mesh:
    """The section's mesh, written to ``path`` and read back.

    Regions: ``ln`` (interior), ``ln_frame``, ``air``, and for ``"meshed"``
    the areas ``hot`` and ``gnd``; lines: ``bottom`` (the frame's base) and,
    for ``"lines"``, ``hot`` and ``gnd``.
    """
    meshed = variant == "meshed"
    x0 = -PAIRS * PERIOD / 2
    xl, xr = x0 - MARGIN, -x0 + MARGIN
    fing = fingers()
    cuts = sorted({xl, xr, *[a for a, _, _ in fing], *[b for _, b, _ in fing]})
    rects: list[tc.Rect] = []
    for a, b in itertools.pairwise(cuts):
        rects.append((a, -DEPTH, b, 0.0, "ln", None))
        name = next((n for fa, _, n in fing if abs(fa - a) < 1e-9), None)
        if meshed:
            rects.append((a, 0.0, b, T_AL, name or "air", 0.1 if name else None))
            rects.append((a, T_AL, b, AIR, "air", None))
        else:
            rects.append((a, 0.0, b, AIR, "air", None))
    rects += [
        (xl - FRAME, -DEPTH, xl, 0.0, "ln_frame", None),
        (xr, -DEPTH, xr + FRAME, 0.0, "ln_frame", None),
        (xl - FRAME, -DEPTH - FRAME, xr + FRAME, -DEPTH, "ln_frame", None),
    ]
    bottom = -DEPTH - FRAME
    edges: list[tc.Namer] = [(lambda p: abs(p[1] - bottom) < 1e-9, "bottom")]
    if not meshed:
        edges += [
            (lambda p, a=a, b=b: abs(p[1]) < 1e-9 and a - 1e-9 < p[0] < b + 1e-9, n)
            for a, b, n in fing
        ]
    nodes, tris, lines = tc._mesh2d(rects, maxh, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def lithium_niobate(axes=AXES, *, quality=QM, piezoelectric=True) -> gfem.Material:
    """Lithium niobate with the lab frame ``axes`` (rows)."""
    return gfem.Material(
        eps_r=EPS_LN,
        axes=axes,
        stiffness=C_LN,
        piezoelectric_tensor=E_LN if piezoelectric else None,
        mass_density=RHO_LN,
        mechanical_quality_factor=quality,
    )


def aluminium(*, quality=QM) -> gfem.Material:
    """Aluminium fingers (isotropic solid)."""
    return gfem.Material(
        stiffness=gfem.isotropic_stiffness(E_AL, NU_AL),
        mass_density=RHO_AL,
        mechanical_quality_factor=quality,
    )


def model(
    mesh: gfem.Mesh,
    variant: str = "lines",
    *,
    axes=AXES,
    quality=QM,
    piezoelectric=True,
    metal_quality=None,
) -> gfem.Model:
    """The device: lithium niobate and its frame, air out of the mechanical domain."""
    ln = lithium_niobate(axes, quality=quality, piezoelectric=piezoelectric)
    materials = {"ln": ln, "ln_frame": ln, "air": gfem.Material()}
    if variant == "meshed":
        metal = aluminium(quality=quality if metal_quality is None else metal_quality)
        materials |= {"hot": metal, "gnd": metal}
    return gfem.Model(mesh, materials=materials, excluded={"mechanical": ["air"]})


def problem(
    frequency=FREQUENCY, *, terminals=("hot",), ground=("gnd",), frame=True, **kwargs
) -> gfem.Piezoelectric:
    """A problem with the damping frame on the lithium niobate frame."""
    return gfem.Piezoelectric(
        frequency,
        terminals=list(terminals),
        ground=list(ground),
        damping_frame=gfem.DampingFrame(["ln_frame"], loss_factor=1.2)
        if frame
        else None,
        **kwargs,
    )


# --- plates and arrays with closed forms (verification tier) -------------------

#: A 6mm (hexagonal) test crystal with lithium-niobate-like magnitudes: stiffness
#: ``c11 c12 c13 c33 c44`` (Pa), stress constants ``e15 e31 e33`` (C/m^2), relative
#: permittivities ``eps11`` and ``eps33``, density.
HEX = {
    "c11": 203e9,
    "c12": 53e9,
    "c13": 75e9,
    "c33": 245e9,
    "c44": 60e9,
    "e15": 3.7,
    "e31": 0.2,
    "e33": 1.3,
    "eps11": 44.0,
    "eps33": 29.0,
    "rho": 4700.0,
}
#: Plate width and thickness, micrometres.
PLATE_WIDTH, PLATE_THICKNESS = 2.0, 1.0
#: The crystal's x, y, z in the lab frame (rows): the c axis along lab y (``TE``;
#: lab x, y, z are crystal y, z, x) or along lab z (``TS``).
PLATE_AXES = {
    "TE": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "TS": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


def hexagonal(kind: str):
    """``(c, e, eps_r)`` of the plate's crystal in its own frame.

    The thickness-extensional plate has :math:`c_{13} = e_{31} = 0`, so that the
    one-dimensional thickness solution leaves its free side edges traction-free.
    """
    h = HEX
    c13, e31 = (0.0, 0.0) if kind == "TE" else (h["c13"], h["e31"])
    c = np.array(
        [
            [h["c11"], h["c12"], c13, 0, 0, 0],
            [h["c12"], h["c11"], c13, 0, 0, 0],
            [c13, c13, h["c33"], 0, 0, 0],
            [0, 0, 0, h["c44"], 0, 0],
            [0, 0, 0, 0, h["c44"], 0],
            [0, 0, 0, 0, 0, (h["c11"] - h["c12"]) / 2],
        ]
    )
    e = np.array(
        [
            [0, 0, 0, 0, h["e15"], 0],
            [0, 0, 0, h["e15"], 0, 0],
            [e31, e31, h["e33"], 0, 0, 0],
        ]
    )
    return c, e, (h["eps11"], h["eps11"], h["eps33"])


def plate_mesh(path: Path, maxh: float) -> gfem.Mesh:
    """The plate's rectangle: region ``pz``, lines ``bot`` (y = 0) and ``top``."""
    w, t = PLATE_WIDTH, PLATE_THICKNESS
    edges: list[tc.Namer] = [
        (lambda p: abs(p[1]) < 1e-9, "bot"),
        (lambda p: abs(p[1] - t) < 1e-9, "top"),
    ]
    nodes, tris, lines = tc._mesh2d([(0.0, 0.0, w, t, "pz", None)], maxh, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def plate_model(mesh: gfem.Mesh, kind: str, *, quality=None) -> gfem.Model:
    """The plate of crystal ``kind`` (``"TE"`` or ``"TS"``) with free side edges."""
    c, e, eps = hexagonal(kind)
    material = gfem.Material(
        eps_r=eps,
        axes=PLATE_AXES[kind],
        stiffness=c,
        piezoelectric_tensor=e,
        mass_density=HEX["rho"],
        mechanical_quality_factor=quality,
    )
    return gfem.Model(mesh, materials={"pz": material})


def plate_problem(frequency, **kwargs) -> gfem.Piezoelectric:
    """The plate driven between its ``top`` (1 V) and ``bot`` (ground) electrodes."""
    return gfem.Piezoelectric(frequency, terminals=["top"], ground=["bot"], **kwargs)


def plate_form(kind: str, quality=None) -> dict:
    r"""The closed form's constants of the plate of crystal ``kind``.

    With :math:`c` (:math:`c_{33}` for ``"TE"``, :math:`c_{44}` for ``"TS"``)
    carrying the loss :math:`1 + i/Q_m`, :math:`e` and :math:`\varepsilon` (F/m)
    the coupling and permittivity of that direction: ``cbar`` is
    :math:`\bar c = c + e^2/\varepsilon`, ``v`` :math:`\sqrt{\bar c/\rho}`,
    ``k2`` :math:`k^2 = e^2/(\varepsilon\bar c)` (complex with the loss) and
    ``c0`` :math:`C_0 = \varepsilon w/h` per metre along :math:`z`; ``e`` and
    ``eps`` are returned as they are.
    """
    h = HEX
    c, e, eps = (
        (h["c33"], h["e33"], h["eps33"])
        if kind == "TE"
        else (h["c44"], h["e15"], h["eps11"])
    )
    eps = eps * EPS0
    cbar = c * (1 + (0.0 if quality is None else 1j / quality)) + e**2 / eps
    return {
        "cbar": cbar,
        "e": e,
        "eps": eps,
        "c0": eps * PLATE_WIDTH / PLATE_THICKNESS,
        "v": np.sqrt(cbar / h["rho"]),
        "k2": e**2 / (eps * cbar),
    }


def finger_array_mesh(
    path: Path, n: int, eta: float = 0.5, *, hedge: float = 0.02, maxh: float = 0.5
) -> gfem.Mesh:
    """``n`` zero-thickness fingers of pitch 1 micrometre on a dielectric under air.

    The air ``a`` (above ``y = 0``) and the dielectric ``b`` are 6 micrometres
    deep with 6 micrometre margins, graded toward the finger edges (size
    ``hedge``). The lines ``hot`` (even fingers) and ``gnd`` (odd) alternate.
    """
    from netgen import occ

    depth, margin, pitch = 6.0, 6.0, 1.0
    width = (n - 1) * pitch + 2 * margin
    air = occ.WorkPlane().MoveTo(0, 0).Rectangle(width, depth).Face()
    air.name = "a"
    solid = occ.WorkPlane().MoveTo(0, -depth).Rectangle(width, depth).Face()
    solid.name = "b"
    segments = []
    for k in range(n):
        xc = margin + k * pitch
        segment = occ.Segment(
            occ.Pnt(xc - eta * pitch / 2, 0, 0), occ.Pnt(xc + eta * pitch / 2, 0, 0)
        )
        segment.name = "hot" if k % 2 == 0 else "gnd"
        segments.append(segment)
    shape = occ.Glue([air, solid, *segments])
    for vertex in shape.vertices:
        if abs(vertex.p.y) < 1e-12 and margin - 0.6 < vertex.p.x < width - margin + 0.6:
            vertex.maxh = hedge
    netgen_mesh = ngs.Mesh(
        occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh, grading=0.25)
    )
    nodes = np.array([v.point for v in netgen_mesh.vertices])
    tris = [
        (el.mat, [v.nr for v in el.vertices]) for el in netgen_mesh.Elements(ngs.VOL)
    ]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in netgen_mesh.Elements(ngs.BND)
        if el.mat in ("hot", "gnd")
    ]
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def finger_array_model(mesh: gfem.Mesh, eps_a: float, eps_b: float) -> gfem.Model:
    """Air ``a`` and an elastic dielectric ``b`` (a valid solid with no coupling)."""
    solid = gfem.Material(
        eps_r=eps_b,
        stiffness=gfem.isotropic_stiffness(E_AL, NU_AL),
        mass_density=RHO_AL,
    )
    return gfem.Model(
        mesh,
        materials={"a": gfem.Material(eps_r=eps_a), "b": solid},
        excluded={"mechanical": ["a"]},
    )


#: A y-uniform shear-wave strip: a piezoelectric source block, a passive
#: elastic strip and a damping frame, driven at ``STRIP_FREQUENCY``.
STRIP_FREQUENCY = 0.855e9
STRIP_SHEAR_SPEED, STRIP_DENSITY, STRIP_POISSON = 3763.0, 2200.0, 0.17
STRIP_SOURCE = 1.0


def strip_wavelength() -> float:
    """The strip's shear wavelength, micrometres."""
    return STRIP_SHEAR_SPEED / STRIP_FREQUENCY * 1e6


def strip_mesh(path: Path, frame: float, *, per_wavelength: float = 20.0) -> gfem.Mesh:
    """The strip with a frame of ``frame`` wavelengths, free on every outer edge.

    A quarter wavelength high. Regions ``src`` (the first micrometre, with the
    lines ``top`` and ``bot`` its electrodes), ``pass`` (three wavelengths
    of passive solid) and ``frame``.
    """
    lam = strip_wavelength()
    height, x1 = lam / 4, STRIP_SOURCE + 3 * lam
    rects = [
        (0.0, 0.0, STRIP_SOURCE, height, "src", None),
        (STRIP_SOURCE, 0.0, x1, height, "pass", None),
        (x1, 0.0, x1 + frame * lam, height, "frame", None),
    ]
    edges: list[tc.Namer] = [
        (lambda p: abs(p[1] - height) < 1e-9 and p[0] < STRIP_SOURCE + 1e-9, "top"),
        (lambda p: abs(p[1]) < 1e-9 and p[0] < STRIP_SOURCE + 1e-9, "bot"),
    ]
    nodes, tris, lines = tc._mesh2d(rects, lam / per_wavelength, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def strip_model(mesh: gfem.Mesh) -> gfem.Model:
    """A thickness-shear source block, then isotropic solid (shear speed 3763 m/s)."""
    c, e, eps = hexagonal("TS")
    source = gfem.Material(
        eps_r=eps, stiffness=c, piezoelectric_tensor=e, mass_density=HEX["rho"]
    )
    shear = STRIP_DENSITY * STRIP_SHEAR_SPEED**2
    passive = gfem.Material(
        stiffness=gfem.isotropic_stiffness(
            2 * shear * (1 + STRIP_POISSON), STRIP_POISSON
        ),
        mass_density=STRIP_DENSITY,
    )
    return gfem.Model(
        mesh, materials={"src": source, "pass": passive, "frame": passive}
    )
