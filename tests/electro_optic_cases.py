"""Lithium niobate constants and cross-sections shared by the electro-optic tests.

The crystal is congruent lithium niobate at 1550 nm (ordinary and
extraordinary indices, the clamped 3m tensor, the static permittivity),
x-cut: propagation along global z. The sections are built at test time by
`thermal_cases.section` (netgen); lengths are in micrometres.

- `layered_slab`: oxide / 0.6 um crystal film / oxide between electrode
  edges ``left`` and ``right``, a lateral field.
- `l_core`: an L-shaped crystal core (a channel and a foot to one side) in
  oxide between plates ``top`` and ``bottom``, a vertical field.
- `tfln_ridge`: a ridge on a crystal film, oxide below and air above, with a
  signal and a ground electrode on the film (``signal``, ``ground``
  regions; their feet and sides ``signal_b``, ``signal_s``, ``ground_b``,
  ``ground_s``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.constants
import thermal_cases as tc

import gfem

WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
N_O, N_E, N_OX = 2.2111, 2.1376, 1.444
#: Optical permittivity of the crystal, principal values along crystal x, y, z.
LN_OPTICAL = (N_O**2, N_O**2, N_E**2)
#: Static permittivity of the crystal, principal values along crystal x, y, z.
LN_STATIC = (44.0, 44.0, 27.9)
#: The 3m tensor entries of lithium niobate, m/V.
R13, R22, R33, R51 = 9.6e-12, 6.8e-12, 30.9e-12, 32.6e-12
#: x-cut, propagation along global z: crystal x up (global y), crystal y along
#: the guide (global z), crystal z lateral (global x).
X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def tensor_3m(r13=R13, r22=R22, r33=R33, r51=R51) -> np.ndarray:
    """The contracted tensor of class 3m, crystal frame, rows xx yy zz yz xz xy."""
    return np.array(
        [
            (0.0, -r22, r13),
            (0.0, r22, r13),
            (0.0, 0.0, r33),
            (0.0, r51, 0.0),
            (r51, 0.0, 0.0),
            (-r22, 0.0, 0.0),
        ]
    )


R_LN = tuple(map(tuple, tensor_3m()))

# --- layered slab ---------------------------------------------------------------

#: Half the slab's width: the electrode edges are 2 SLAB_HALF apart.
SLAB_HALF = 4.0
#: The crystal film's thickness, um.
SLAB_FILM = 0.6
SLAB_EDGES = [
    (lambda p: abs(p[0] - SLAB_HALF) < 1e-9, "right"),
    (lambda p: abs(p[0] + SLAB_HALF) < 1e-9, "left"),
]


def layered_slab(path: Path, factor: float = 1.0) -> gfem.Mesh:
    """The slab (regions ``below``, ``film``, ``above``); edge lengths x ``factor``."""
    rects = [
        (-SLAB_HALF, -2.0, SLAB_HALF, 0.0, "below", None),
        (-SLAB_HALF, 0.0, SLAB_HALF, SLAB_FILM, "film", 0.08 * factor),
        (-SLAB_HALF, SLAB_FILM, SLAB_HALF, 2.6, "above", None),
    ]
    return tc.section(path, rects, 0.3 * factor, SLAB_EDGES)


# --- L-shaped core ---------------------------------------------------------------

#: The core, the foot and the cladding (x0, y0, x1, y1, region).
L_CORE = [
    (-0.5, 0.0, 0.5, 0.6, "core"),
    (0.5, 0.0, 1.6, 0.25, "foot"),
    (-3.0, -2.0, 3.0, 2.6, "clad"),
]
PLATES = [
    (lambda p: abs(p[1] - 2.6) < 1e-9, "top"),
    (lambda p: abs(p[1] + 2.0) < 1e-9, "bottom"),
]
#: The plates' distance, m: a homogeneous static model has E = -1/GAP along y.
GAP = 4.6e-6


def l_core(path: Path, maxh: float, crystal: float) -> gfem.Mesh:
    """The L-shaped core in its cladding; edge length ``crystal`` in the crystal."""
    rects = [(*r, None if r[4] == "clad" else crystal) for r in L_CORE]
    return tc.section(path, rects, maxh, PLATES)


# --- ridge between electrodes ------------------------------------------------------

#: The ridge's width, the electrodes' gap, width, and the section's half width, um.
RIDGE_WIDTH, ELECTRODE_GAP, ELECTRODE_WIDTH, RIDGE_HALF = 0.9, 3.5, 2.25, 8.0


def _outline(x0, y0, x1, y1):
    def on(p):
        x, y = p[0], p[1]
        along_x = abs(x - x0) < 1e-9 or abs(x - x1) < 1e-9
        along_y = abs(y - y0) < 1e-9 or abs(y - y1) < 1e-9
        inside_x = x0 - 1e-9 <= x <= x1 + 1e-9
        inside_y = y0 - 1e-9 <= y <= y1 + 1e-9
        return (along_x and inside_y) or (along_y and inside_x)

    return on


def _electrode(name, x0, x1):
    return [
        (lambda p: abs(p[1] - 0.3) < 1e-9 and x0 < p[0] < x1, f"{name}_b"),
        (_outline(x0, 0.3, x1, 0.5), f"{name}_s"),
    ]


def tfln_ridge(path: Path, factor: float = 1.0) -> gfem.Mesh:
    """The ridge section; edge lengths x ``factor``."""
    a, w, half = ELECTRODE_GAP / 2, ELECTRODE_WIDTH, RIDGE_HALF
    rects = [
        (-half, 0.3, half, 3.0, "air", None),
        (-half, -2.0, half, 0.0, "box", None),
        (-half, 0.0, half, 0.3, "film", 0.08 * factor),
        (-RIDGE_WIDTH / 2, 0.3, RIDGE_WIDTH / 2, 0.6, "ridge", 0.06 * factor),
        (a, 0.3, a + w, 0.5, "signal", 0.1 * factor),
        (-a - w, 0.3, -a, 0.5, "ground", 0.1 * factor),
    ]
    edges = [*_electrode("signal", a, a + w), *_electrode("ground", -a - w, -a)]
    return tc.section(path, rects, 0.4 * factor, edges)


#: The ridge's potential: the signal electrode against the ground electrode.
RIDGE_DRIVE = gfem.Electrostatic(
    terminals={"signal": ["signal_b", "signal_s"]}, ground=["ground_b", "ground_s"]
)
