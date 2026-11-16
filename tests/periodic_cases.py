"""Periodic cells shared by the periodic tests, with their closed forms.

The cells C1-C4 of the periodic verification cases, built on
`structured_meshes.periodic_cell` (opposite faces named "x_min" / "x_max",
... and matching under translation). Every builder that writes a mesh takes
the directory to write it to; nothing here solves, so the tolerances can be
read without NGSolve. Meshes are in mm, wave vectors in rad/m.

- C1, `c1`: a parallel-plate TEM line periodic along x (PEC plates at z = 0
  and H, PMC y walls) with two 50 ohm lumped ports on interior sheets; its
  open-circuit impedance matrix is `bloch_line_z`.
- C2, `c2`: an empty parallel-plate cell periodic along x and y; its bands
  are the empty lattice, `empty_lattice`.
- C3, `c3`: a layered (Bragg) TEM line periodic along x; its bands are the
  transfer-matrix roots, `tmm_frequency`.
- C4, `c4`: an empty vacuum cube periodic along x, y and z (no other
  boundary); its lowest band is a doublet at `empty_lattice`.
- G, `g`: the gradient cell, periodic along x and y, with two lumped
  ports, a PEC ground, an absorbing top, a resistive `gfem.Impedance` sheet
  touching both pairs and a sidewall "wall" crossing the x seam (the
  periodic cells of the gradient verification matrix).
- F1, `f1`: a Fresnel slab between Floquet ports (meshes in um), its S the
  tangential-E Airy form `f1_airy`; with an air substrate, the Brewster
  cell. F2, `f2`: a lamellar grating between Floquet ports.

C1 loaded by its two 50 ohm ports is also a damped Bloch eigen case: its
complex resonances are the roots of det(Z(f, k) + R) = 0,
`loaded_line_frequency`. The eigen gradient cell, `eigen_cell`, is a
damped, lossy parallel-plate cell periodic along x and y, with a resistive
sheet crossing the y seam.

The tolerances are about 10-50x the error measured on each cell at its
order and mesh.
"""

import math
from pathlib import Path

import numpy as np
import scipy.constants as const
import scipy.optimize
import structured_meshes as sm

import gfem

MM = 1e-3
C0 = const.c
ETA0 = const.mu_0 * const.c

#: |S - S_exact| on C1.
C1_S_TOL = 1e-5
#: Relative error of the 4 lowest C2 bands (real part).
C2_BAND_RTOL = 3e-4
#: |E_z(p2) / E_z(p1) - exp(-i k·Δr)| on the C2 band with G = 0.
C2_PHASE_TOL = 1e-2
#: Relative error of a complex C3 frequency.
C3_FREQ_RTOL = 2e-6
#: Relative error of the lowest C4 doublet.
C4_RTOL = 1e-4
#: Relative error of the complex resonance of C1 loaded by its ports
#: (6.6e-6 measured).
C1_EIGEN_RTOL = 1e-4


def _mesh(directory: Path, name: str, cell: tuple) -> gfem.Mesh:
    path = sm.write_msh22(Path(directory) / f"{name}.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _edges(size, cells):
    return tuple(np.linspace(0.0, s, n + 1) for s, n in zip(size, cells, strict=True))


# --- C1: Bloch line two-port ------------------------------------------------------

C1_A, C1_W, C1_H = 10.0, 2.0, 1.0  # mm
C1_X1, C1_X2 = 2.5, 6.25  # port sheets, mm
C1_EPS = 4.0
C1_TAN_DELTA = 0.02
C1_FREQUENCY = 5e9
C1_ORDER = 2
#: The Bloch wave vector of C1, rad/m (also solved at -k and 0).
C1_K = (0.3 * math.pi / (C1_A * MM), 0.0, 0.0)


def c1_mesh(directory: Path, cells: tuple[int, int, int] = (16, 1, 1)) -> gfem.Mesh:
    """C1's mesh; the sheets "sheet1" (x = X1) and "sheet2" (x = X2) are grid planes."""
    cell = sm.periodic_cell(
        _edges((C1_A, C1_W, C1_H), cells),
        sheets=(("sheet1", C1_X1), ("sheet2", C1_X2)),
    )
    return _mesh(directory, "c1", cell)


def c1(mesh: gfem.Mesh) -> gfem.Model:
    """C1: periodic in x, PEC at z = 0 and H, default PMC y walls, two ports."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material(eps_r=C1_EPS, tan_delta=C1_TAN_DELTA)},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[
            gfem.LumpedPort("p1", "sheet1", "+z", impedance=50.0),
            gfem.LumpedPort("p2", "sheet2", "+z", impedance=50.0),
        ],
    )


def bloch_line_z(frequency: complex, k: float, eps: complex) -> np.ndarray:
    """C1's open-circuit impedance matrix (ohm) at Bloch wave number k (rad/m).

    Two shunt current sheets at X1 and X2 on a TEM line of period a, with
    E(x + a) = exp(-i k a) E(x):

        Z11 = Z22 = (Z0 / 2) [1 / (1 - q2) + q1 / (1 - q1)]
        Z21(k) = (Z0 / 2) [exp(-i b d) / (1 - q2) + exp(+i b d) q1 / (1 - q1)]
        Z12(k) = Z21(-k)

    with q1 = exp(-i (b + k) a), q2 = exp(-i (b - k) a), d = X2 - X1 and
    b = sqrt(eps) k0. ``eps`` is complex for a lossy line, ``frequency``
    for a free oscillation.
    """
    a, d = C1_A * MM, (C1_X2 - C1_X1) * MM
    k0 = 2 * np.pi * frequency / C0
    b = np.sqrt(complex(eps)) * k0
    z0 = ETA0 * C1_H / (C1_W * np.sqrt(complex(eps)))

    def q(kk):
        return np.exp(-1j * (b + kk) * a), np.exp(-1j * (b - kk) * a)

    def z21(kk):
        q1, q2 = q(kk)
        return (
            z0
            / 2
            * (np.exp(-1j * b * d) / (1 - q2) + np.exp(1j * b * d) * q1 / (1 - q1))
        )

    q1, q2 = q(k)
    z11 = z0 / 2 * (1 / (1 - q2) + q1 / (1 - q1))
    return np.array([[z11, z21(-k)], [z21(k), z11]])


#: The C1 eigen target, Hz: the loaded line's resonance near 14.18 + 2.51i GHz
#: (Q 2.8) at `C1_K`.
C1_EIGEN_TARGET = 14e9


def loaded_line_frequency(k: float, guess: complex) -> complex:
    """C1's complex resonance (Hz) nearest ``guess`` with both ports loaded.

    The root of det(Z(f, k) + R) = 0 (`bloch_line_z`, R = 50 ohm on each
    port): the port currents of a free oscillation see their loads.
    """
    eps = C1_EPS * (1 - 1j * C1_TAN_DELTA)

    def residual(fz):
        z = bloch_line_z(complex(fz[0], fz[1]), k, eps) + 50.0 * np.eye(2)
        d = np.linalg.det(z)
        return [d.real, d.imag]

    re, im = scipy.optimize.fsolve(residual, [guess.real, guess.imag], xtol=1e-14)
    return complex(re, im)


# --- C2: empty parallel-plate cell, two pairs --------------------------------------

C2_AX, C2_AY, C2_H = 10.0, 7.0, 1.0  # mm
C2_EPS = 2.0
C2_ORDER = 2
#: The Bloch wave vector of C2, rad/m.
C2_K = (0.4 * math.pi / (C2_AX * MM), -0.25 * math.pi / (C2_AY * MM), 0.0)
#: The two points (m) of C2's field sign pin.
C2_PROBES = (
    (0.31 * C2_AX * MM, 0.42 * C2_AY * MM, 0.5 * C2_H * MM),
    (0.77 * C2_AX * MM, 0.13 * C2_AY * MM, 0.5 * C2_H * MM),
)


def c2_mesh(directory: Path, cells: tuple[int, int, int] = (8, 8, 1)) -> gfem.Mesh:
    """C2's mesh."""
    return _mesh(directory, "c2", sm.periodic_cell(_edges((C2_AX, C2_AY, C2_H), cells)))


def c2_gmsh_mesh() -> gfem.Mesh:
    """C2's cell meshed unstructured by gmsh, the pairs' faces tied by setPeriodic.

    194 nodes, 500 tetrahedra; written by tools/make_mesh_fixtures.py.
    """
    path = Path(__file__).parent / "unit" / "data" / "periodic_cell_gmsh.msh"
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def c2(mesh: gfem.Mesh) -> gfem.Model:
    """C2: periodic in x and y (the pairs share the z edges), PEC at z = 0 and h."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material(eps_r=C2_EPS)},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
    )


def empty_lattice(
    k: tuple[float, float, float],
    periods: tuple[float | None, float | None, float | None],
    eps: float = 1.0,
    n: int = 4,
) -> np.ndarray:
    """The ``n`` lowest empty-lattice frequencies c|k + G| / (2 pi sqrt(eps)), Hz.

    ``periods`` (m) per axis, ``None`` for an axis without a pair (G = 0
    there). One value per G: no polarization multiplicity (C4's lowest
    value is a doublet).
    """
    axes = [range(-3, 4) if p else range(1) for p in periods]
    kv = np.asarray(k, dtype=float)
    out = []
    for idx in np.ndindex(*(len(r) for r in axes)):
        g = [
            2 * math.pi * r[i] / p if p else 0.0
            for r, i, p in zip(axes, idx, periods, strict=True)
        ]
        out.append(C0 * np.linalg.norm(kv + g) / (2 * math.pi * math.sqrt(eps)))
    return np.sort(out)[:n]


# --- C3: Bragg line -----------------------------------------------------------------

C3_A, C3_W, C3_H = 10.0, 2.0, 1.0  # mm
C3_D1 = 4.0  # thickness of the low-index layer, mm
C3_EPS_LOW, C3_EPS_HIGH = 2.0, 9.0
C3_TAN_DELTA = 0.01  # the lossy variant
C3_ORDER = 3
#: The plain (Gamma) target, away from the zero-frequency TEM field, Hz.
C3_PLAIN_TARGET = 11e9


def c3_mesh(directory: Path, cells: tuple[int, int, int] = (10, 1, 1)) -> gfem.Mesh:
    """C3's mesh: "low" for x < D1, "high" beyond (D1 must be a grid plane)."""
    cell = sm.periodic_cell(
        _edges((C3_A, C3_W, C3_H), cells),
        lambda c: "low" if c[0] < C3_D1 else "high",
    )
    return _mesh(directory, "c3", cell)


def c3(mesh: gfem.Mesh, tan_delta: float = 0.0) -> gfem.Model:
    """C3: periodic in x, PEC at z = 0 and H, default PMC y walls."""
    return gfem.Model(
        mesh,
        materials={
            "low": gfem.Material(eps_r=C3_EPS_LOW, tan_delta=tan_delta),
            "high": gfem.Material(eps_r=C3_EPS_HIGH, tan_delta=tan_delta),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
    )


def tmm_frequency(k: float, guess: complex, tan_delta: float = 0.0) -> complex:
    """The C3 band frequency (Hz, complex when lossy) nearest ``guess`` at k (rad/m).

    The root of the transfer-matrix (Rytov) dispersion relation of the two
    layers, cos(n1 k0 d1) cos(n2 k0 d2) - (n1/n2 + n2/n1)/2 sin(n1 k0 d1)
    sin(n2 k0 d2) = cos(k a), with n = sqrt(eps (1 - i tan_delta)).
    """
    a, d1 = C3_A * MM, C3_D1 * MM
    d2 = a - d1
    n1 = np.sqrt(C3_EPS_LOW * (1 - 1j * tan_delta))
    n2 = np.sqrt(C3_EPS_HIGH * (1 - 1j * tan_delta))

    def residual(fz):
        k0 = 2 * math.pi * (fz[0] + 1j * fz[1]) / C0
        r = (
            np.cos(n1 * k0 * d1) * np.cos(n2 * k0 * d2)
            - 0.5 * (n1 / n2 + n2 / n1) * np.sin(n1 * k0 * d1) * np.sin(n2 * k0 * d2)
            - np.cos(k * a)
        )
        return [r.real, r.imag]

    re, im = scipy.optimize.fsolve(residual, [guess.real, guess.imag], xtol=1e-14)
    return complex(re, im)


# --- C4: empty cube, three pairs ------------------------------------------------------

C4_L = 10.0  # mm
C4_ORDER = 2
#: The Bloch wave vector of C4, rad/m.
C4_K = tuple(v * math.pi / (C4_L * MM) for v in (0.3, -0.2, 0.15))


def c4_mesh(directory: Path, cells: tuple[int, int, int] = (3, 3, 3)) -> gfem.Mesh:
    """C4's mesh."""
    return _mesh(directory, "c4", sm.periodic_cell(_edges((C4_L,) * 3, cells)))


def c4(mesh: gfem.Mesh) -> gfem.Model:
    """C4: vacuum, periodic in x, y and z, no other boundary."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material()},
        boundaries=[gfem.Periodic((f"{a}_min", f"{a}_max")) for a in "xyz"],
    )


# --- G: the gradient cell ---------------------------------------------------------

G_A, G_B, G_H = 4.0, 5.0, 3.0  # mm
#: The grid planes, mm: ports at x = 1 and 3 and 3 < y < 4, the wall at
#: y = 2, the floor below z = 1, the sheet at z = 2.
G_EDGES = ((0.0, 1.0, 3.0, 4.0), (0.0, 2.0, 3.0, 4.0, 5.0), (0.0, 1.0, 2.0, 3.0))
G_SHEET_Z = 2.0  # the Impedance sheet, mm
G_WALL_Y = 2.0  # the sidewall between "core" and "sub", mm
G_FREQUENCIES = (20e9, 28e9)
#: The Bloch wave vector of G, rad/m.
G_K = (0.3 * math.pi / (G_A * MM), -0.2 * math.pi / (G_B * MM), 0.0)


def g_mesh(directory: Path) -> gfem.Mesh:
    """G's mesh: a 4 x 5 x 3 mm cell periodic along x and y.

    Volumes "core" (y < 2, z < 1), "sub" (y > 2, z < 1) and "cover" (z >
    1); surfaces the box faces, "sheet" (the plane z = 2, which touches
    both pairs along edges), "wall" (y = 2 between "core" and "sub", which
    crosses the x seam) and the lumped-port faces "port1" (x = 1) and
    "port2" (x = 3), both on 3 < y < 4, z < 1, away from the pairs.
    """
    edges = tuple(np.asarray(e) for e in G_EDGES)
    tol = 1e-9 * G_B

    def volume(c):
        name = "cover" if c[2] > 1 else "core" if c[1] < G_WALL_Y else "sub"
        # cut at the interior planes, so their faces are offered to `named`
        return f"{name}\0{int(c[0] > 1)}{int(c[0] > 3)}{int(c[2] > G_SHEET_Z)}"

    def named(c):
        for axis, a in enumerate("xyz"):
            hi = (G_A, G_B, G_H)[axis]
            if abs(c[axis]) < tol:
                return f"{a}_min"
            if abs(c[axis] - hi) < tol:
                return f"{a}_max"
        if abs(c[2] - G_SHEET_Z) < tol:
            return "sheet"
        for x, port in ((1.0, "port1"), (3.0, "port2")):
            if abs(c[0] - x) < tol:
                return port if 3 < c[1] < 4 and c[2] < 1 else ""
        if abs(c[1] - G_WALL_Y) < tol and c[2] < 1:
            return "wall"
        return ""

    nodes, tets, names, tris, tri_names = sm.structured(edges, volume, named)
    cell = (nodes, tets, [n.split("\0")[0] for n in names], tris, tri_names)
    return _mesh(directory, "g", cell)


def g(
    mesh: gfem.Mesh,
    *,
    eps_r=None,
    conductivity=None,
    fill=None,
) -> gfem.Model:
    """G: lumped ports, PEC ground, absorbing top, a resistive sheet across the seams.

    ``eps_r`` and ``conductivity``: per-element values over "core" then
    "sub" (mesh order of each), lossy constants by default; ``fill``: the
    sheet's per-face `gfem.SheetField` fill (solid by default).
    """
    n_core = len(mesh.elements("core"))
    n = n_core + len(mesh.elements("sub"))
    eps = np.full(n, 3.0 * (1 - 0.02j)) if eps_r is None else eps_r
    sigma = np.full(n, 0.2) if conductivity is None else conductivity
    split = {"core": slice(0, n_core), "sub": slice(n_core, n)}
    sheet = None if fill is None else gfem.SheetField(fill=fill)
    return gfem.Model(
        mesh,
        materials={
            **{
                r: gfem.MaterialField(eps_r=eps[s], conductivity=sigma[s])
                for r, s in split.items()
            },
            "cover": gfem.Material(),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
            gfem.Absorbing("z_max"),
            gfem.Impedance("sheet", resistance=300.0, fill=sheet),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=50.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=50.0),
        ],
    )


# --- the eigen gradient cell ---------------------------------------------------------

EG_A, EG_B, EG_H = 10.0, 3.0, 1.0  # mm
EG_SHEET = 5.0  # the resistive sheet's plane x, mm
EG_EPS = (3.5, 4.5)  # the permittivity ramp over the tetrahedra
EG_SIGMA = (0.02, 0.06)  # the conductivity ramp, S/m
EG_SHEET_R = 2000.0  # ohm per square
EG_TAN_DELTA = 0.02  # the linear lossy variant
#: The Bloch wave vector of the eigen gradient cell, rad/m.
EG_K = (0.4 * math.pi / (EG_A * MM), 0.3 * math.pi / (EG_B * MM), 0.0)
#: Targets: the Bloch cell's lowest band (about 8 GHz) and the plain
#: cell's split G = ±2π/a pair (about 15 GHz).
EG_BLOCH_TARGET = 7e9
EG_PLAIN_TARGET = 14e9


def eigen_cell_mesh(directory: Path, cells: tuple[int, int, int] = (8, 2, 2)):
    """The eigen gradient cell's mesh: the sheet "sheet" at x = EG_SHEET."""
    cell = sm.periodic_cell(
        _edges((EG_A, EG_B, EG_H), cells), sheets=(("sheet", EG_SHEET),)
    )
    return _mesh(directory, "eigen_cell", cell)


def eigen_cell(
    mesh: gfem.Mesh, eps, sigma=None, fill=None, *, tan_delta: float | None = None
) -> gfem.Model:
    """The eigen gradient cell: periodic in x and y, PEC at z = 0 and H.

    Per-element ``eps`` (and ``sigma``, S/m, a damping), and with ``fill`` a
    resistive `gfem.Impedance` sheet (`EG_SHEET_R` per square, per-face
    fill) on "sheet", which touches the y pair along its edges. Without
    ``sigma`` and ``fill``, a ``tan_delta`` makes the cell lossy but
    undamped (a linear problem).
    """
    boundaries = [
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ]
    if fill is not None:
        boundaries.append(
            gfem.Impedance(
                "sheet", resistance=EG_SHEET_R, fill=gfem.SheetField(fill=fill)
            )
        )
    material = gfem.MaterialField(eps_r=eps, conductivity=sigma, tan_delta=tan_delta)
    return gfem.Model(mesh, materials={"cell": material}, boundaries=boundaries)


def eigen_cell_design(mesh: gfem.Mesh, seed: int = 0) -> dict:
    """``eps``, ``sigma`` and ``fill`` of the eigen gradient cell."""
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("cell"))
    faces = len(mesh.elements("sheet"))
    return {
        "eps": np.linspace(*EG_EPS, n),
        "sigma": np.linspace(*EG_SIGMA, n),
        "fill": rng.uniform(0.5, 1.0, faces),
    }


def coordinate_classes(
    mesh: gfem.Mesh, periods: tuple[float | None, ...]
) -> np.ndarray:
    """A class label per node: nodes equal modulo the ``periods`` (m) share one.

    Twins of every pair (and the corners of several) move together in a
    periodic node direction.
    """
    nodes = np.asarray(mesh.imported_nodes)
    keys = nodes.copy()
    for axis, period in enumerate(periods):
        if period:
            keys[:, axis] = np.mod(keys[:, axis], period)
            keys[np.isclose(keys[:, axis], period), axis] = 0.0
    scale = 1e-6 * float(np.ptp(nodes, axis=0).max())
    _, labels = np.unique(np.round(keys / scale), axis=0, return_inverse=True)
    return labels.reshape(-1)


# --- F1: Fresnel slab between Floquet ports -----------------------------------------

UM = 1e-6
#: lambda0 = 1 um; F1's mesh is in um.
F1_FREQUENCY = C0 / UM
F1_A = 0.4  # lateral period along x and y, um
F1_EPS_SLAB = 4.0
F1_EPS_SUB = 2.25
F1_H = 0.3  # slab thickness, um
F1_D = 0.25  # distance from each Floquet face to the slab, um
#: |S - Airy| on F1 at p = 2 on the coarse mesh `f1_mesh` (sign pin; 2.6e-3
#: measured, a wrong sign is off by about 0.6).
F1_COARSE_TOL = 2e-2
#: F1's cells at h about lambda/10 in each medium (`f1_mesh`), solved at p = 3
#: for the gates below.
F1_FINE_CELLS = (4, 4, 6, 3)
#: |S - Airy| on F1 at p = 3, `F1_FINE_CELLS` (7.1e-7 measured at normal
#: incidence, 1.3e-6 at `F1_THETA`, 1.8e-6 on the Brewster cell).
F1_S_TOL = 3e-4
#: |S| of the cross-polarized entries there (9.7e-7 measured at normal
#: incidence, 6.4e-7 at `F1_THETA`, 1.8e-6 on the Brewster cell).
F1_CROSS_TOL = 1e-5
#: The power defect 1 - sum |S|^2 of a column there (2.1e-9 measured at
#: normal incidence, 1.1e-7 at `F1_THETA`, 5.6e-7 on the Brewster cell on
#: `F1_FINE_CELLS`).
F1_POWER_TOL = 1e-4
#: F1's oblique (conical) incidence from the air: theta from the top face's
#: inward normal, phi from ê1 = +x toward ê2 = +y (`gfem.Model.bloch_wave_vector`).
F1_THETA = math.radians(40.0)
F1_PHI = math.radians(30.0)
#: The Brewster cell (`f1` with ``eps_sub = 1``: air / eps 4 slab / air) at
#: Brewster's angle atan 2 from the air: TM reflects nothing at either
#: interface.
BREWSTER_EPS_SUB = 1.0
BREWSTER_THETA = math.atan(2.0)
#: The Brewster cell's cells: one lateral cell more than `F1_FINE_CELLS`
#: and four in the air, for the steeper in-plane phase (cross-polarization
#: 4.2e-6 on `F1_FINE_CELLS`, 1.8e-6 here).
BREWSTER_CELLS = (5, 4, 6, 4)
#: |S11_TM| of the Brewster cell at p = 3, `BREWSTER_CELLS` (1.5e-6 measured).
BREWSTER_TOL = 3e-5


def f1_mesh(
    directory: Path,
    cells: tuple[int, int, int, int] = (3, 3, 3, 3),
    d: float = F1_D,
) -> gfem.Mesh:
    """F1's mesh, um: ``cells`` = (lateral, substrate, slab, air) cells per side.

    Substrate z in [0, d], slab [d, d + H], air [d + H, 2 d + H]; the open
    faces are "z_min" (substrate) and "z_max" (air), ``d`` um from the slab.
    """
    n, ns, nh, na = cells
    lateral = np.linspace(0.0, F1_A, n + 1)
    z = np.concatenate(
        [
            np.linspace(0.0, d, ns + 1),
            np.linspace(d, d + F1_H, nh + 1)[1:],
            np.linspace(d + F1_H, 2 * d + F1_H, na + 1)[1:],
        ]
    )

    def volume(c):
        if c[2] < d:
            return "sub"
        return "slab" if c[2] < d + F1_H else "air"

    cell = sm.periodic_cell((lateral, lateral, z), volume, paired="xy")
    path = sm.write_msh22(Path(directory) / "f1.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def f1_ports(top: str = "z_max", bottom: str = "z_min") -> list[gfem.FloquetPort]:
    """The specular TE and TM ports of both faces: R_TE, R_TM (top), T_TE, T_TM."""
    return [
        gfem.FloquetPort(f"{side}_{pol}", surface, polarization=pol)
        for side, surface in (("R", top), ("T", bottom))
        for pol in ("TE", "TM")
    ]


def f1(
    mesh: gfem.Mesh, ports: list | None = None, eps_sub: float = F1_EPS_SUB
) -> gfem.Model:
    """F1: an eps 4 slab on an ``eps_sub`` substrate under air, Floquet ports on both.

    ``eps_sub = BREWSTER_EPS_SUB`` gives the Brewster cell.
    """
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.Material(eps_r=eps_sub),
            "slab": gfem.Material(eps_r=F1_EPS_SLAB),
            "air": gfem.Material(),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=f1_ports() if ports is None else ports,
    )


def airy(
    frequency: float,
    kt: float,
    layers: list[tuple[float, float]],
    polarization: str,
) -> tuple[complex, complex]:
    """S11 and S21 of a planar stack between two port planes (tangential-E Airy form).

    ``layers`` lists ``(eps_r, thickness in m)`` from the top port plane down
    to the bottom port plane (the first and last are the port media, their
    thicknesses the port distances; mu_r = 1). The wave comes from the top
    with tangential wave number ``kt`` (rad/m). S11 is the tangential-E
    reflection coefficient at the top plane; S21 the tangential-E
    transmission at the bottom plane times sqrt(Z_top / Z_bottom), the
    power-normalized entry of `gfem.conventions` (e^{+iωt}).
    """
    k0 = 2 * math.pi * frequency / C0

    def admittance(eps):
        kz = np.sqrt(complex(k0**2 * eps - kt**2))
        return kz, (kz if polarization == "TE" else k0**2 * eps / kz)

    # (E, H) at the bottom plane for a unit transmitted wave, carried up
    e, h = 1.0 + 0j, admittance(layers[-1][0])[1]
    for eps, d in reversed(layers):
        kz, y = admittance(eps)
        c, s = np.cos(kz * d), np.sin(kz * d)
        e, h = c * e + 1j * s * h / y, 1j * y * s * e + c * h
    y_top, y_bottom = admittance(layers[0][0])[1], admittance(layers[-1][0])[1]
    t = 2 * y_top / (h + y_top * e)
    return complex(t * e - 1), complex(t * np.sqrt(y_bottom / y_top))


def f1_airy(
    polarization: str,
    frequency: float = F1_FREQUENCY,
    kt: float = 0.0,
    eps_sub: float = F1_EPS_SUB,
):
    """F1's (S11, S21) from the top, `airy`; ``kt`` the in-plane wave number, rad/m."""
    return airy(
        frequency,
        kt,
        [
            (1.0, F1_D * UM),
            (F1_EPS_SLAB, F1_H * UM),
            (eps_sub, F1_D * UM),
        ],
        polarization,
    )


# --- F2: lamellar grating between Floquet ports ------------------------------------

#: F2's periods, um: 1.2 lambda0 along x (the grating), 0.4 lambda0 along y.
F2_A = (1.2, 0.4)
F2_RIDGE = (0.3, 0.9)  # the ridge's x extent, um
F2_EPS_RIDGE = 4.0
F2_H = 0.3  # ridge height, um
F2_D = 0.3  # distance from each Floquet face to the ridge layer, um


def f2_mesh(
    directory: Path, cells: tuple[int, int, int, int, int] = (4, 2, 2, 2, 2)
) -> gfem.Mesh:
    """F2's mesh, um: ``cells`` = (x, y, substrate, ridge, air) cells.

    ``x`` cells per ridge or groove width (0.6 um each, grid lines on the
    ridge edges). Substrate z in [0, D], ridge layer [D, D + H] (regions
    "ridge" and "groove"), air [D + H, 2 D + H]; open faces "z_min"
    (substrate) and "z_max" (air).
    """
    nx, ny, ns, nh, na = cells
    x0, x1 = F2_RIDGE
    x = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, x0, max(nx // 2, 1) + 1),
                np.linspace(x0, x1, nx + 1),
                np.linspace(x1, F2_A[0], max(nx // 2, 1) + 1),
            ]
        )
    )
    y = np.linspace(0.0, F2_A[1], ny + 1)
    z = np.concatenate(
        [
            np.linspace(0.0, F2_D, ns + 1),
            np.linspace(F2_D, F2_D + F2_H, nh + 1)[1:],
            np.linspace(F2_D + F2_H, 2 * F2_D + F2_H, na + 1)[1:],
        ]
    )

    def volume(c):
        if c[2] < F2_D:
            return "sub"
        if c[2] < F2_D + F2_H:
            return "ridge" if x0 < c[0] < x1 else "groove"
        return "air"

    cell = sm.periodic_cell((x, y, z), volume, paired="xy")
    path = sm.write_msh22(Path(directory) / "f2.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def f2_ports(
    orders: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0)),
) -> list[gfem.FloquetPort]:
    """TE and TM ports of ``orders`` on both faces: "R{m},{n}_TE" (top), "T..."."""
    return [
        gfem.FloquetPort(
            f"{side}{m},{n}_{pol}", surface, order=(m, n), polarization=pol
        )
        for side, surface in (("R", "z_max"), ("T", "z_min"))
        for m, n in orders
        for pol in ("TE", "TM")
    ]


def f2(
    mesh: gfem.Mesh, ports: list | None = None, eps_ridge: complex = F2_EPS_RIDGE
) -> gfem.Model:
    """F2: an eps 4 lamellar grating on an eps 2.25 substrate under air.

    ``eps_ridge`` lossy (``4 * (1 - 0.05j)``) for G20L. The default ports
    are every order propagating at normal incidence (`f2_ports`).
    """
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.Material(eps_r=F1_EPS_SUB),
            "ridge": gfem.Material(eps_r=eps_ridge),
            "groove": gfem.Material(),
            "air": gfem.Material(),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=f2_ports() if ports is None else ports,
    )


# --- F2's frozen RCWA references (ticket 09 Q3) ------------------------------------

#: F2's cells at h about lambda0 / 10 with grid lines on the ridge edges
#: (`f2_mesh`), solved at p = 3 for the grating gates below.
F2_FINE_CELLS = (6, 4, 3, 3, 3)
#: F2's cases: (incidence theta, phi from the air in degrees, ridge eps).
F2_CASES = {
    "G0": (0.0, 0.0, F2_EPS_RIDGE),
    "G20": (20.0, 0.0, F2_EPS_RIDGE),
    "G20c": (20.0, 30.0, F2_EPS_RIDGE),
    "G20L": (20.0, 0.0, F2_EPS_RIDGE * (1 - 0.05j)),
}
#: Diffraction efficiencies of F2 by case, incident polarization (from the
#: air, the specular order) and (face, m): "R" the air side, "T" the
#: substrate; the sum over both output polarizations of |S|^2. Every
#: propagating order of each case is listed; order labels follow ADR 0147
#: (b1 along +x).
#:
#: Method: the planar cases (G0, G20, G20L) from a Li-rule (inverse rule)
#: 1D RCWA at N = 640 harmonics, the conical G20c from nannos 2.6.5
#: ("tangent" formulation) at nh = 321. The two codes agree to 1e-8 at
#: equal truncation; the values move by at most 4e-7 (planar) and 3e-6
#: (conical) from half the truncation, power balance closes to 1e-11 in the
#: lossless cases: the values are good to about 1e-6. The generation
#: scripts stay with the effort's notes; no RCWA package is a test
#: dependency.
F2_RCWA = {
    "G0": {
        "TE": {
            ("R", -1): 0.00347658,
            ("R", 0): 0.04656300,
            ("R", 1): 0.00347658,
            ("T", -1): 0.25787622,
            ("T", 0): 0.43073139,
            ("T", 1): 0.25787622,
        },
        "TM": {
            ("R", -1): 0.02413918,
            ("R", 0): 0.01169126,
            ("R", 1): 0.02413918,
            ("T", -1): 0.24120150,
            ("T", 0): 0.45762738,
            ("T", 1): 0.24120150,
        },
    },
    "G20": {
        "TE": {
            ("R", -1): 0.06556192,
            ("R", 0): 0.00532756,
            ("T", -2): 0.07640159,
            ("T", -1): 0.18441132,
            ("T", 0): 0.27416941,
            ("T", 1): 0.39412820,
        },
        "TM": {
            ("R", -1): 0.01563704,
            ("R", 0): 0.00842941,
            ("T", -2): 0.01767815,
            ("T", -1): 0.21207643,
            ("T", 0): 0.49783872,
            ("T", 1): 0.24834025,
        },
    },
    "G20c": {
        "TE": {
            ("R", -1): 0.05060379,
            ("R", 0): 0.00911272,
            ("T", -2): 0.06267673,
            ("T", -1): 0.17843395,
            ("T", 0): 0.33630617,
            ("T", 1): 0.36286663,
        },
        "TM": {
            ("R", -1): 0.03154545,
            ("R", 0): 0.00727369,
            ("T", -2): 0.04835677,
            ("T", -1): 0.20106154,
            ("T", 0): 0.46709015,
            ("T", 1): 0.24467240,
        },
    },
    "G20L": {
        "TE": {
            ("R", -1): 0.05873477,
            ("R", 0): 0.00533568,
            ("T", -2): 0.05637469,
            ("T", -1): 0.17072437,
            ("T", 0): 0.22627813,
            ("T", 1): 0.34492168,
        },
        "TM": {
            ("R", -1): 0.01774115,
            ("R", 0): 0.00870128,
            ("T", -2): 0.01501870,
            ("T", -1): 0.19253965,
            ("T", 0): 0.46512389,
            ("T", 1): 0.21423294,
        },
    },
}
#: G20L's absorption 1 - sum of the efficiencies, by incident polarization.
F2_ABSORPTION = {"TE": 0.13763068, "TM": 0.08664219}
