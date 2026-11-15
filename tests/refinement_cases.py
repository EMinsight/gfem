"""Singular benchmark cases shared by the mesh refinement tests, with references.

One case per problem type, each a 2D 270 degree re-entrant corner extruded by
one structured cell (`structured_meshes`, no gmsh). Uniform splits converge
at h^(4/3) on all three (`RATE`, order 2), so mesh refinement has room to beat
them. Every builder that writes a mesh takes the directory to write it to and
``n``, the cells per unit length of the section; ``grade`` > 1
packs the cells towards the corner.

- Square coax, `coax_model`: a quarter of a square coax (inner half-side
  1 um, outer 2 um, extruded 0.5 um), `COAX_PROBLEM`; quantity
  `coax_capacitance`, the full coax's C / (eps0 len), reference `COAX_C`.
- Step line, `step_model`: a parallel-plate line in mm (L = 10, W = 2,
  H = 1, eps_r = 4) whose top plate steps down to H / 2 at L / 2, two 50 ohm
  lumped ports, `step_problem`; quantity `step_s`, reference `STEP_S`.
- L cavity, `cavity_model`: a PEC cavity of L-shaped section (side 2 mm,
  height 0.5 mm), `CAVITY_PROBLEM`; quantity `cavity_frequency`, reference
  `CAVITY_F1`.

Two more cases, not singular at a re-entrant corner:

- Sheet, `sheet_model`: two bars "a" and "b" in a grounded box with a
  zero-thickness grounded sheet "fin" between them
  (`structured_meshes.sheet_screen`), `SHEET_PROBLEM`; the only case whose
  estimate is cut along an interior sheet. References `SHEET_C` (F),
  Richardson extrapolations of uniform order-2 n = 16, 32, 64 with the rate
  fitted per entry (C_aa 1.32, C_ab 0.98: the sheet's tip dominates the
  off-diagonal entry), within `SHEET_C_BOUND` (relative) of the
  extrapolation from n = 8, 16, 32.
- Two bars, `bars_model`: bars "a" and "b" in a grounded box
  (`structured_meshes.two_bars`), `BARS_PROBLEM`, the off-diagonal entry
  C_ab: "near" (strongly coupled, |C_ab / C_aa| = 0.09) and "screen" (a
  grounded fin between unequal offset bars, 0.0014). Every bar corner is
  re-entrant (C converges at h^(4/3)). References `BARS_C_AB` (F),
  Richardson extrapolations (alpha = 4/3) of uniform order-2 n = 32 and 64
  (fitted alpha 1.334 and 1.318), within `BARS_C_AB_BOUND` (relative) of
  the n = 16, 32 pair and of a mesh graded 3 towards the corners.
- Anisotropic coax, `anisotropic_coax_model`: the circular coax quarter
  (`structured_meshes.coax_quarter`, radii 1 and 2 um, extruded 0.5 um)
  filled with eps_r (2, 5, 3) along rotated axes, `ANISOTROPIC_COAX`; its
  reference is an order-5 solve on the same mesh, which the tests run.
- Anisotropic box, `anisotropic_box_model`: the PEC box 1 x 0.7 x 0.4 mm
  with a magnetic or anisotropic fill, `ANISOTROPIC_BOX`. The "iso_mu" and
  "diag" fills have a closed form; for "rot3" and "layer_mu" the references
  are `ANISOTROPIC_BOX_F`, order-4 solves on the m = 2 mesh.
- Lossy guide, `lossy_guide_model`: a PEC guide 4 mm long (x), 2 x 0.8 mm
  (y, z), wave ports "p1" and "p2" at its ends, at `LOSSY_GUIDE_FREQUENCY`.
  "uniform" fills it with a lossy vacuum (TE10, its S functional parallel
  to its excitation); "slab" puts a lossy eps_r 4 slab under y = 1 mm along
  its length (a hybrid lossy port mode, not parallel); "step" widens that
  slab to y < 1.5 mm from x = 2 mm on. References `LOSSY_GUIDE_S`, order 4
  on the n = 4 mesh with the same offsets.

Tests never recompute the references. Provenance: the coax and step values
are Richardson extrapolations (alpha = 4/3) of uniform order-2 runs, coax
n = 64 and 128, step n = 16 and 32, each checked against the fitted alpha
(1.335 and 1.345), the previous pair and an independent graded mesh (grade 3:
coax n = 64, step n = 32). The cavity's first mode is TM_z, so k^2 is the
first Dirichlet eigenvalue of the L (Trefethen and Betcke, "Computed
eigenmodes of planar regions", 2006); Richardson on uniform n = 32 and 64
lands 1.6e-6 from it, which validates the procedure used for the other two.

Errors of a uniform order-2 solve (relative), for orientation:

    n    coax      step S21 (3 GHz)   cavity f1
    2    8.54e-3   1.22e-3            5.66e-3
    4    3.32e-3   4.68e-4            2.22e-3
    8    1.30e-3   1.83e-4            9.55e-4
"""

from pathlib import Path

import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem

#: Convergence rate of every case under uniform splits, per halving of h.
RATE = 4 / 3

#: Full square coax, C / (eps0 len); the quarter model (extruded 0.5 um) has
#: C = COAX_C * eps0 * 0.5e-6 / 4 = 1.1326822e-17 F.
COAX_C = 10.2340924
#: Absolute bound of `COAX_C` (8e-8 relative).
COAX_C_BOUND = 8.4e-7

#: The step line's S entries per frequency (Hz), with absolute bounds.
STEP_S = {
    3e9: {
        "S21": 0.1831794 - 0.9146944j,
        "S21_bound": 5.5e-7,
        "S11": 0.2103088 + 0.2924889j,
        "S11_bound": 3.9e-7,
    },
    8e9: {
        "S21": -0.7821258 + 0.1740548j,
        "S21_bound": 1.4e-6,
        "S11": 0.5964552 - 0.0471740j,
        "S11_bound": 7.1e-7,
    },
}

#: First Dirichlet eigenvalue of the L of side 2, in mm^-2 (closed to ~1e-13).
CAVITY_LAMBDA1 = 9.6397238440219
#: The cavity's first resonance, Hz: c sqrt(CAVITY_LAMBDA1) / (2 pi), side in mm.
CAVITY_F1 = 148.14026965e9

#: Uniform 3D split (each tetrahedron into 8) from the n = 2 mesh: (DOF, error)
#: per level, order 2, direct. Step: |S21(3 GHz) - STEP_S|; cavity: relative
#: error of f1. Level 0 is the n = 2 mesh (pre-asymptotic; `uniform_split_error`
#: fits the later levels).
STEP_SPLIT = ((1706, 1.1377e-3), (11292, 4.5280e-4), (81464, 1.8045e-4))
CAVITY_SPLIT = (
    (674, 5.6633e-3),
    (4484, 2.4081e-3),
    (32456, 9.7531e-4),
    (246416, 3.8907e-4),
)

#: The air box 1 x 0.7 x 0.4 mm (`structured_meshes.plate`), PEC but for a lossy
#: top wall: its 110 mode's omega, rad/s, per wall. Richardson extrapolations
#: of uniform order-2 direct solves on h = 0.2 / m mm, m = 4 and 5, with the
#: rate fitted on m = 3, 4, 5 (alpha 3.98 impedance and absorbing, 4.02
#: conductor); the alpha = 4 extrapolation lies within 4.2e-9 (relative), and
#: `LOSSY_WALL_BOUND` bounds both, 1/100 of the m = 3 error (2.4e-6 or more).
#: Provenance: .scratch/adaptive-refinement-followups/probes/04 (wall.jsonl).
LOSSY_WALL_OMEGA = {
    "impedance": 1888057987875.6094 + 236196247959.60132j,  # 377 ohm/sq
    "absorbing": 1887838945992.7559 + 236275835911.12192j,  # first order
    "conductor": 1639154393785.1357 + 3222701363.0419827j,  # 1e5 S/m
    # 50 ohm/sq (Q 16.6): Richardson with alpha = 4 on m = 4, 5; the (3, 5)
    # extrapolation lies within 1.2e-10 (relative), its Q within 2.4e-8.
    # Provenance: .scratch/quantity-estimates/probes/06 (wall.jsonl).
    "impedance 50": 1646446067072.405 + 49650544763.42546j,
}
#: Relative bound of each `LOSSY_WALL_OMEGA`.
LOSSY_WALL_BOUND = 2e-8

#: The L cavity (`cavity_model`) filled with eps_r 4 inside `gfem.Conductor`
#: walls of 1e5 S/m: its first mode's omega, rad/s (Q 45.35). Richardson of
#: uniform order-2 direct solves on n = 16, 32, 64 with the complex rate
#: fitted on them (alpha 1.33, the corner's 4/3). Against the alpha = 4/3
#: extrapolation its Q moves by `CONDUCTOR_CAVITY_Q_BOUND` (relative), 16 %
#: of the n = 16 Q error and 25 % of the n = 32 one, which is not asserted.
#: Provenance: .scratch/quantity-estimates/probes/06 (lwall.jsonl).
CONDUCTOR_CAVITY_OMEGA = 460170192679.7072 + 5074129465.918461j
#: Relative bound of the Q of `CONDUCTOR_CAVITY_OMEGA`.
CONDUCTOR_CAVITY_Q_BOUND = 8.3e-4

#: The sheet case's capacitance matrix, F, terminals in order a, b.
SHEET_C = ((4.08621e-17, -1.39274e-19), (-1.39274e-19, 4.19939e-17))
#: Relative bound of each `SHEET_C` entry.
SHEET_C_BOUND = ((6e-6, 1.5e-4), (1.5e-4, 6e-6))

#: The two-bars cases' C_ab, F (`structured_meshes.two_bars`, `BARS_PROBLEM`).
BARS_C_AB = {"near": -3.760064826e-18, "screen": -5.618272068e-20}
#: Relative bound of each `BARS_C_AB` entry.
BARS_C_AB_BOUND = {"near": 2e-6, "screen": 2e-5}


def rotation(axis: str, degrees: float) -> np.ndarray:
    """The rotation matrix about ``axis`` ("x", "y" or "z") by ``degrees``."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    out = np.eye(3)
    out[i, i] = out[j, j] = c
    out[i, j], out[j, i] = -s, s
    return out


#: A generic rotation, about z, y and x by 30, 25 and 40 degrees: no
#: principal axis along a coordinate axis.
ROT3 = rotation("z", 30) @ rotation("y", 25) @ rotation("x", 40)

#: The anisotropic coax's fills: eps_r (2, 5, 3) along the columns of a
#: rotation about z by 30 degrees ("rotz"), or about z, y and x by 30, 25 and
#: 40 degrees ("rot3": no principal axis along the coax).
ANISOTROPIC_COAX = {
    name: gfem.Material(eps_r=(2.0, 5.0, 3.0), axes=tuple(map(tuple, r.T)))
    for name, r in (("rotz", rotation("z", 30)), ("rot3", ROT3))
}

#: The anisotropic box's fills, per region: one region "air" filling the box
#: with eps_r 2 and mu_r 3 ("iso_mu"), with eps_r (2, 3, 4) and mu_r
#: (1.5, 2, 2.5) along the coordinate axes ("diag") or along `ROT3`'s columns
#: ("rot3"); or two layers, "a" (eps_r 3) under "b" (mu_r 2.5) ("layer_mu").
ANISOTROPIC_BOX = {
    "iso_mu": {"air": gfem.Material(eps_r=2.0, mu_r=3.0)},
    "diag": {"air": gfem.Material(eps_r=(2.0, 3.0, 4.0), mu_r=(1.5, 2.0, 2.5))},
    "rot3": {
        "air": gfem.Material(
            eps_r=(2.0, 3.0, 4.0),
            mu_r=(1.5, 2.0, 2.5),
            axes=tuple(map(tuple, ROT3.T)),
        )
    },
    "layer_mu": {"a": gfem.Material(eps_r=3.0), "b": gfem.Material(mu_r=2.5)},
}

#: The four lowest modes of the "rot3" and "layer_mu" boxes, Hz: order 4 on
#: the m = 2 mesh, direct (ostau, 2026-11-12). On the "diag" box, same mesh,
#: order 4 lies within 1.7e-8 (relative) of the closed form; the order-2
#: errors these check are 1.2e-5 to 9.6e-5.
ANISOTROPIC_BOX_F = {
    "rot3": (106.357683863e9, 146.424985394e9, 173.585662901e9, 186.214213145e9),
    "layer_mu": (161.103601495e9, 226.726156671e9, 238.567948030e9, 255.087323796e9),
}

#: The lossy fills of the PEC box (ADR 0138), per region: eps_r 2 with a
#: complex mu_r 2 (1 - 0.1i) ("lossy_mu"); "a" (eps_r 3) under "b" (mu_r
#: 2.5 (1 - 0.2i)) ("lossy_layer_mu"); `ANISOTROPIC_BOX`'s "diag" fill with
#: tan_delta 0.1 ("lossy_diag_tan"), conductivity (5, 10, 20) S/m
#: ("lossy_diag_sigma") or per-axis complex eps_r, tan_delta (0.05, 0.1, 0.3)
#: ("lossy_diag_complex"); that per-axis loss along `ROT3`'s columns
#: ("lossy_rot3").
LOSSY_EPS_DIAG = np.array([2.0, 3.0, 4.0]) * (1 - 1j * np.array([0.05, 0.1, 0.3]))
LOSSY_BOX = {
    "lossy_mu": {"air": gfem.MaterialField(eps_r=2.0, mu_r=2.0 * (1 - 0.1j))},
    "lossy_layer_mu": {
        "a": gfem.Material(eps_r=3.0),
        "b": gfem.MaterialField(mu_r=2.5 * (1 - 0.2j)),
    },
    "lossy_diag_tan": {
        "air": gfem.Material(eps_r=(2.0, 3.0, 4.0), mu_r=(1.5, 2.0, 2.5), tan_delta=0.1)
    },
    "lossy_diag_sigma": {
        "air": gfem.Material(
            eps_r=(2.0, 3.0, 4.0), mu_r=(1.5, 2.0, 2.5), conductivity=(5.0, 10.0, 20.0)
        )
    },
    "lossy_diag_complex": {
        "air": gfem.MaterialField(
            eps_r=LOSSY_EPS_DIAG,
            mu_r=(1.5, 2.0, 2.5),
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
    },
    "lossy_rot3": {
        "air": gfem.MaterialField(
            eps_r=LOSSY_EPS_DIAG,
            mu_r=(1.5, 2.0, 2.5),
            axes=tuple(map(tuple, ROT3.T)),
        )
    },
}

#: Modes of the "lossy_layer_mu" and "lossy_rot3" boxes, omega in rad/s:
#: order 4 on the m = 2 mesh, direct, each the mode nearest the lossless
#: box's `ANISOTROPIC_BOX_F` frequency (its key here, Hz) (ostau,
#: 2026-11-14). `LOSSY_BOX_BOUND` (relative) bounds each by |order 3 -
#: order 4|, under 2 % of the order-2 error it checks.
#: Provenance: .scratch/quantity-estimates/probes/10 (cmu.jsonl, aniso.jsonl).
LOSSY_BOX_OMEGA = {
    "lossy_layer_mu": {
        161.103601495e9: 1007119253424.7919 + 68907669884.89853j,
        226.726156671e9: 1421434413657.8762 + 93317793057.87018j,
        238.567948030e9: 1493088982074.622 + 38904586117.109695j,
    },
    "lossy_rot3": {
        106.357683863e9: 657737744736.5281 + 70972525364.01263j,
        146.424985394e9: 904878485590.797 + 102641036684.74458j,
        173.585662901e9: 1080416508990.5245 + 82256927116.25586j,
        186.214213145e9: 1215771515468.8926 + 84519811757.66254j,
    },
}
#: Relative bound of each `LOSSY_BOX_OMEGA`, same keys.
LOSSY_BOX_BOUND = {
    "lossy_layer_mu": {
        161.103601495e9: 4.1e-8,
        226.726156671e9: 3.9e-7,
        238.567948030e9: 7.8e-7,
    },
    "lossy_rot3": {
        106.357683863e9: 3.1e-7,
        146.424985394e9: 1.1e-6,
        173.585662901e9: 9.0e-7,
        186.214213145e9: 3.1e-6,
    },
}

#: The layered boxes' modes, omega in rad/s, keyed by the target (Hz) each was
#: the mode nearest to: the PEC box 1 x 0.7 x 0.4 mm on the m = 2 mesh with a
#: 0.2 mm "lower" layer under 0.2 mm of air, the lower one a Drude law
#: (eps_inf 4, plasma 300 GHz, damping 30 GHz; "law") or that law frozen at
#: 217.4 GHz ("constant"). Order 4 on the same mesh, direct (ostau,
#: 2026-11-14); `LAYERED_BOX_BOUND` (relative) bounds each by |order 3 -
#: order 4|, at most 3 % of the order-2 error it checks.
#: Provenance: .scratch/quantity-estimates/probes/10 (law.jsonl).
LAYERED_BOX_OMEGA = {
    "law": {
        200e9: 1365886721224.771 + 28583488045.186317j,
        300e9: 1855493638560.5923 + 21210889092.263752j,
        380e9: 2396599931171.4844 + 13611651521.137688j,
    },
    "constant": {
        200e9: 1362352081374.5503 + 40387269805.226944j,
        300e9: 1866221045627.128 + 72030402229.23325j,
        380e9: 2344744872972.081 + 111638577282.47076j,
    },
}
#: Relative bound of each `LAYERED_BOX_OMEGA`, same keys.
LAYERED_BOX_BOUND = {
    "law": {200e9: 4.8e-8, 300e9: 1.5e-6, 380e9: 2.0e-5},
    "constant": {200e9: 6.9e-8, 300e9: 8.0e-7, 380e9: 4.7e-6},
}

#: The air box 1 x 0.7 x 0.4 mm with second-order `Absorbing` side walls, PEC
#: top and bottom: the modes continuing its PEC 110, 210 and 120 modes, omega
#: in rad/s, keyed (m, mode). Order 4 on the same h = 0.2 / m mm mesh,
#: direct, the mode of six nearest the PEC frequency (ostau, 2026-11-14);
#: `ABSORBING_BOX_BOUND` (relative) bounds each by |order 3 - order 4|, at
#: most 9.4 % of the order-2 error it checks.
#: Provenance: .scratch/quantity-estimates/probes/10 (abc2.jsonl).
ABSORBING_BOX_OMEGA = {
    (1, (1, 1, 0)): 1427348796225.951 + 451233361025.6063j,
    (2, (1, 1, 0)): 1427390703452.3496 + 451169484224.0431j,
    (1, (2, 1, 0)): 2578944193656.4575 + 335038120087.43616j,
    (2, (2, 1, 0)): 2578933663570.1753 + 335101659007.0283j,
    (1, (1, 2, 0)): 2700555782230.8823 + 341180615784.23846j,
    (2, (1, 2, 0)): 2700605224545.1787 + 341260372951.9436j,
}
#: Relative bound of each `ABSORBING_BOX_OMEGA`, same keys.
ABSORBING_BOX_BOUND = {
    (1, (1, 1, 0)): 3.4e-4,
    (2, (1, 1, 0)): 2.5e-5,
    (1, (2, 1, 0)): 1.5e-4,
    (2, (2, 1, 0)): 1.1e-5,
    (1, (1, 2, 0)): 2.3e-4,
    (2, (1, 2, 0)): 1.6e-5,
}

#: The plasmon box's LSM (2, 1) mode, rad/s: the 1 x 0.7 x 0.4 mm PEC box with
#: a 0.2 mm Drude floor (plasma 350 GHz, damping 5 GHz; Re eps -1.73 at the
#: mode) under 0.2 mm of air, the root of the LSM^z dispersion relation
#: (kz1 / e1) tan(kz1 d1) + (kz2 / e2) tan(kz2 d2) = 0, solved by Newton's
#: method to `PLASMON_BOUND` (relative); order-4 solves agree with it to
#: 5e-9 to 1e-5. Provenance: .scratch/quantity-estimates/probes/10 (lsm.py).
PLASMON_OMEGA = 1331954822019.4377 + 10841584018.340836j
#: Relative bound of `PLASMON_OMEGA`.
PLASMON_BOUND = 1e-12

#: The block guide's frequency, Hz: TE10 and TE20 propagate in the air guide.
BLOCK_GUIDE_FREQUENCY = 170e9
#: The block guide's S entries (to, from) at `BLOCK_GUIDE_FREQUENCY`: order 4
#: on the n = 4 mesh, which an order-3 solve on n = 6 meets to 6.4e-5 (3-7 %
#: of the order-2 error at n = 8).
BLOCK_GUIDE_S = {
    ("b1", "a1"): -0.7079432 + 0.3847732j,
    ("b2", "a2"): 0.5864315 + 0.1803482j,
    ("b2", "a1"): -0.1076877 + 0.2060751j,
    ("a2", "a1"): -0.2027340 + 0.1480379j,
    ("a1", "a1"): 0.2856943 + 0.3897492j,
    ("a2", "a2"): 0.1262016 - 0.7003084j,
}

#: The lossy guide's frequency, Hz: TE10 propagates, TE20 does not.
LOSSY_GUIDE_FREQUENCY = 170e9
#: The lossy guide's S entries (to, from), keyed (kind, tan_delta, offset of
#: both ports in mm): order 4 on the n = 4 mesh (adaptive-refinement
#: follow-ups probe 12). An order-3 solve on n = 6 meets them to 1e-6
#: (uniform) and 2e-4 (slab, step), under 20 % of the order-2 error of every
#: entry the tests use; the tan_delta 0.3 thru entries of slab and step at
#: n = 8 are above that and not used.
LOSSY_GUIDE_S = {
    ("uniform", 0.05, 0): {
        ("p2", "p1"): 0.6549341658 - 0.1545045333j,
        ("p1", "p2"): 0.6549341658 - 0.1545045333j,
        ("p1", "p1"): -0.0027973552 + 0.0091826709j,
        ("p2", "p2"): -0.0027973552 + 0.0091826709j,
    },
    ("uniform", 0.05, 1): {
        ("p2", "p1"): 0.8150705911 - 0.0948174733j,
        ("p1", "p2"): 0.8150705911 - 0.0948174733j,
        ("p1", "p1"): -0.0046827863 + 0.0107281798j,
        ("p2", "p2"): -0.0046827863 + 0.0107281798j,
    },
    ("uniform", 0.3, 0): {
        ("p2", "p1"): 0.0891372791 - 0.0416226583j,
        ("p1", "p2"): 0.0891372791 - 0.0416226583j,
        ("p1", "p1"): 0.0073199509 + 0.0901872005j,
        ("p2", "p2"): 0.0073199509 + 0.0901872005j,
    },
    ("uniform", 0.3, 1): {
        ("p2", "p1"): 0.3099134668 - 0.0685522195j,
        ("p1", "p2"): 0.3099134668 - 0.0685522195j,
        ("p1", "p1"): -0.0402122290 + 0.2891572358j,
        ("p2", "p2"): -0.0402122290 + 0.2891572358j,
    },
    ("slab", 0.05, 0): {
        ("p2", "p1"): 0.1391315192 - 0.4504759364j,
        ("p1", "p2"): 0.1392483630 - 0.4504398195j,
        ("p1", "p1"): -0.0016203356 + 0.0168414211j,
        ("p2", "p2"): -0.0016203313 + 0.0168417900j,
    },
    ("slab", 0.05, 1): {
        ("p2", "p1"): 0.5527070920 - 0.4077918410j,
        ("p1", "p2"): 0.5528128391 - 0.4076484465j,
        ("p1", "p1"): -0.0164649478 + 0.0183428585j,
        ("p2", "p2"): -0.0164652618 + 0.0183432947j,
    },
    ("slab", 0.3, 0): {
        ("p2", "p1"): -0.0005324469 - 0.0117754010j,
        ("p1", "p2"): -0.0005141147 - 0.0117762162j,
        ("p1", "p1"): 0.0068112483 + 0.0839707675j,
        ("p2", "p2"): 0.0068114282 + 0.0839710285j,
    },
    ("slab", 0.3, 1): {
        ("p2", "p1"): 0.0758057075 - 0.0792812897j,
        ("p1", "p2"): 0.0759290422 - 0.0791631858j,
        ("p1", "p1"): -0.5211950332 + 0.5856389582j,
        ("p2", "p2"): -0.5211956334 + 0.5856418469j,
    },
    ("step", 0.05, 0): {
        ("p2", "p1"): -0.1168939451 - 0.3990112088j,
        ("p1", "p2"): -0.0838297827 - 0.4084525667j,
        ("p1", "p1"): -0.0017797940 + 0.0191893159j,
        ("p2", "p2"): -0.0178326019 - 0.0264509073j,
    },
    ("step", 0.05, 1): {
        ("p2", "p1"): 0.3518543715 - 0.4895479334j,
        ("p1", "p2"): 0.3921407127 - 0.4601728133j,
        ("p1", "p1"): -0.0186824711 + 0.0209575285j,
        ("p2", "p2"): 0.0255207393 - 0.0383163218j,
    },
    ("step", 0.3, 0): {
        ("p2", "p1"): -0.0029550764 + 0.0032810485j,
        ("p1", "p2"): -0.0043567268 + 0.0019937294j,
        ("p1", "p1"): 0.0068027220 + 0.0840269645j,
        ("p2", "p2"): 0.0103244163 + 0.0993725874j,
    },
    ("step", 0.3, 1): {
        ("p2", "p1"): -0.0380443766 + 0.0234107314j,
        ("p1", "p2"): -0.0479742177 + 0.0069164433j,
        ("p1", "p1"): -0.5216279567 + 0.5859428563j,
        ("p2", "p2"): 0.3661330586 + 1.0359589494j,
    },
}

_EPS_R = 4.0
_QUARTER = 0.5e-6 / 4  # the coax model's extrusion over the full coax's 4 quarters

COAX_PROBLEM = gfem.Electrostatic(terminals={"inner": "inner"}, ground="outer")
SHEET_PROBLEM = gfem.Electrostatic(
    terminals={"a": "a", "b": "b"}, ground=["box", "fin"]
)
BARS_PROBLEM = gfem.Electrostatic(terminals={"a": "a", "b": "b"}, ground="box")
CAVITY_PROBLEM = gfem.Eigenmode(target=CAVITY_F1, n_modes=1)


def uniform_split_error(split: tuple, dofs: int) -> float:
    """The uniform split's error at ``dofs``: a power law through levels 1 on."""
    n, e = np.log(np.asarray(split[1:])).T
    slope, offset = np.polyfit(n, e, 1)
    return float(np.exp(offset + slope * np.log(dofs)))


def step_problem(frequencies=tuple(STEP_S)) -> gfem.Driven:
    """The step line's uniform sweep, at the reference frequencies by default."""
    return gfem.Driven(frequency=list(frequencies))


def _mesh(directory: Path, name: str, mesh: tuple, unit: str) -> gfem.Mesh:
    path = sm.write_msh22(Path(directory) / f"{name}.msh", mesh)
    return gfem.Mesh.from_gmsh(path, length_unit=unit)


def coax_model(directory: Path, n: int, *, grade: float = 1.0) -> gfem.Model:
    """The square coax quarter, in um."""
    mesh = _mesh(directory, "coax", sm.square_coax_quarter(n, grade=grade), "um")
    return gfem.Model(mesh, materials={"dielectric": gfem.Material()})


def sheet_model(directory: Path, n: int) -> gfem.Model:
    """The sheet case, in um, vacuum."""
    mesh = _mesh(directory, "sheet", sm.sheet_screen(n), "um")
    vacuum = gfem.Material()
    return gfem.Model(mesh, materials={"left": vacuum, "right": vacuum})


def bars_model(directory: Path, case: str, n: int) -> gfem.Model:
    """A two-bars case (``"near"`` or ``"screen"``), in um, vacuum."""
    mesh = _mesh(directory, "bars", sm.two_bars(case, n), "um")
    return gfem.Model(mesh, materials={"dielectric": gfem.Material()})


def anisotropic_coax_model(directory: Path, k: int, fill: str) -> gfem.Model:
    """The circular coax quarter, 2k cells across, 4k around, filled with ``fill``."""
    cells = sm.coax_quarter(1.0, 2.0, 0.5, n_r=2 * k, n_theta=4 * k)
    mesh = _mesh(directory, "coax", cells, "um")
    return gfem.Model(mesh, materials={"dielectric": ANISOTROPIC_COAX[fill]})


def anisotropic_box_model(directory: Path, m: int, fill: str) -> gfem.Model:
    """The PEC box 1 x 0.7 x 0.4 mm, h = 0.2 / m mm, filled with ``fill``.

    ``fill`` is a key of `ANISOTROPIC_BOX` or `LOSSY_BOX`.
    """
    regions = {**ANISOTROPIC_BOX, **LOSSY_BOX}[fill]
    layers = tuple((name, 0.4 / len(regions)) for name in regions)
    cells = sm.plate(1.0, 0.7, layers, (5 * m, 4 * m, 2 * m // len(regions)))
    mesh = _mesh(directory, "box", cells, "mm")
    return gfem.Model(
        mesh,
        materials=regions,
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )


def block_guide_model(directory: Path, n: int) -> gfem.Model:
    """The block guide, in mm, ``n`` cells per mm, two modes per port surface.

    An air guide 4 mm long (x), 2 x 0.8 mm (y, z), PEC walls, with an eps_r 4
    block at x 1.5-2.5 and y 0-1 mm through its height; wave ports "a1",
    "a2" (TE10, TE20) on "port1" at x = 0 and "b1", "b2" on "port2".
    """
    length, width, height = 4.0, 2.0, 0.8
    edges = (
        np.linspace(0, length, int(length * n) + 1),
        np.linspace(0, width, int(width * n) + 1),
        np.linspace(0, height, max(2, int(height * n)) + 1),
    )

    def volume(c: np.ndarray) -> str:
        return "block" if 1.5 < c[0] < 2.5 and c[1] < 1.0 else "air"

    def surface(c: np.ndarray) -> str:
        if abs(c[0]) < 1e-9:
            return "port1"
        if abs(c[0] - length) < 1e-9:
            return "port2"
        walls = (abs(c[1]) < 1e-9, abs(c[1] - width) < 1e-9)
        walls += (abs(c[2]) < 1e-9, abs(c[2] - height) < 1e-9)
        return "wall" if any(walls) else ""

    mesh = _mesh(directory, "guide", sm.structured(edges, volume, surface), "mm")
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "block": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("wall")],
        ports=[
            gfem.WavePort(name, surface, mode=mode)
            for name, surface, mode in (
                ("a1", "port1", 1),
                ("a2", "port1", 2),
                ("b1", "port2", 1),
                ("b2", "port2", 2),
            )
        ],
    )


def lossy_guide_model(
    directory: Path, n: int, kind: str, tan_delta: float, offset: float = 0.0
) -> gfem.Model:
    """The lossy guide, in mm, ``n`` cells per mm (see the module docstring).

    ``kind`` is "uniform", "slab" or "step"; every material has
    ``tan_delta``; both wave ports are de-embedded by ``offset`` (m).
    """
    length, width, height = 4.0, 2.0, 0.8
    edges = (
        np.linspace(0, length, int(length * n) + 1),
        np.linspace(0, width, int(width * n) + 1),
        np.linspace(0, height, max(2, int(height * n)) + 1),
    )

    def volume(c: np.ndarray) -> str:
        top = 1.5 if kind == "step" and c[0] >= 2.0 else 1.0
        return "slab" if kind != "uniform" and c[1] < top else "fill"

    def surface(c: np.ndarray) -> str:
        if abs(c[0]) < 1e-9:
            return "port1"
        if abs(c[0] - length) < 1e-9:
            return "port2"
        walls = (abs(c[1]) < 1e-9, abs(c[1] - width) < 1e-9)
        walls += (abs(c[2]) < 1e-9, abs(c[2] - height) < 1e-9)
        return "wall" if any(walls) else ""

    cells = sm.structured(edges, volume, surface)
    mesh = _mesh(directory, f"lossy_{kind}{n}", cells, "mm")
    materials = {"fill": gfem.Material(tan_delta=tan_delta)}
    if kind != "uniform":
        materials["slab"] = gfem.Material(eps_r=4.0, tan_delta=tan_delta)
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("wall")],
        ports=[
            gfem.WavePort("p1", "port1", offset=offset),
            gfem.WavePort("p2", "port2", offset=offset),
        ],
    )


def step_model(
    directory: Path, n: int, *, grade: float = 1.0, unit: str = "mm"
) -> gfem.Model:
    """The step line, in mm, PEC plates, 50 ohm lumped ports "p1" and "p2".

    ``unit`` reads the same mesh in another length unit (a chip-scale line
    in "um").
    """
    mesh = _mesh(directory, "step", sm.step_line(n, grade=grade), unit)
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=_EPS_R)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def cavity_model(directory: Path, n: int, *, grade: float = 1.0) -> gfem.Model:
    """The L cavity, in mm, PEC walls."""
    mesh = _mesh(directory, "cavity", sm.l_cavity(n, grade=grade), "mm")
    return gfem.Model(
        mesh, materials={"air": gfem.Material()}, boundaries=[gfem.PEC("wall")]
    )


def solve(model: gfem.Model, problem, order: int = 2):
    """An order-``order`` direct solve, the setting the references were measured at."""
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(model, problem, options=options)


def coax_capacitance(result: gfem.ElectrostaticResult) -> float:
    """The full coax's C / (eps0 len) from the quarter model's result."""
    return float(result.capacitance[0, 0]) / (const.epsilon_0 * _QUARTER)


def step_s(result: gfem.DrivenResult) -> np.ndarray:
    """The S matrices, shape (frequencies, 2, 2), ports in order p1, p2."""
    return np.asarray(result.s.values)


def cavity_frequency(result: gfem.EigenmodeResult) -> float:
    """The first mode's frequency, Hz (real part)."""
    return float(np.real(np.asarray(result.frequency)[0]))
