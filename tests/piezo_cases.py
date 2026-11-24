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

import numpy as np
import thermal_cases as tc

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
EPS0 = 8.8541878128e-12
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
