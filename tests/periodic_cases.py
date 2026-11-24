"""Periodic cells shared by the periodic tests, with their closed forms.

The cells C1-C4 of the periodic verification cases, built on
`structured_meshes.periodic_cell` (opposite faces named "x_min" / "x_max",
... and matching under translation). Every builder that writes a mesh takes
the directory to write it to; nothing here solves, so the tolerances can be
read without NGSolve. Meshes are in mm, wave vectors in rad/m.

- C1, `c1`: a parallel-plate TEM line periodic along x (PEC plates at z = 0
  and H, PMC y walls) with two 50 ohm lumped ports on interior sheets; its
  open-circuit impedance matrix is `bloch_line_z`. `c1edge` moves p1 onto
  the last x cell's y = 0 strip, touching the seam: its Bloch S is far from
  reciprocal.
- C2, `c2`: an empty parallel-plate cell periodic along x and y; its bands
  are the empty lattice, `empty_lattice`. `c2_square_mesh` makes it square.
- C3, `c3`: a layered (Bragg) TEM line periodic along x; its bands are the
  transfer-matrix roots, `tmm_frequency`.
- C4, `c4`: an empty vacuum cube periodic along x, y and z (no other
  boundary); its lowest band is a doublet at `empty_lattice`.
- `sheared_cell`: a vacuum cell whose two pairs make any angle, the
  hexagonal (60 or 120 degrees) and oblique lattices, for geometry only.
- G, `g`: the gradient cell, periodic along x and y, with two lumped
  ports, a PEC ground, an absorbing top, a resistive `gfem.Impedance` sheet
  touching both pairs and a sidewall "wall" crossing the x seam (the
  periodic cells of the gradient verification matrix).
  `cellx` is G with its x pair only; `periodic_box` is `mpi_cases.box`
  periodic along x with ports touching the seam: the one-pair cases of the
  distributed periodic tests.
- F1, `f1`: a Fresnel slab between Floquet ports (meshes in um), its S the
  tangential-E Airy form `f1_airy`; with an air substrate, the Brewster
  cell. F2, `f2`: a lamellar grating between Floquet ports.
- Rods, `rods_tutorial_mesh`: the photonic-crystal tutorial's square lattice
  of alumina rods, a committed 2D section read as a slab; its converged
  bands at Γ, X and M are `ROD_BANDS`, its band group velocities at the
  path's segment midpoints `ROD_V_G`. `rods_mesh` staircases the same
  cell on the structured grid, one element thick, and `rods` makes it a
  model, PEC (TM) or PMC (TE) plates, with loss in the rod only: the loss
  varies in space.

C1 loaded by its two 50 ohm ports is also a damped Bloch eigen case: its
complex resonances are the roots of det(Z(f, k) + R) = 0,
`loaded_line_frequency`. The eigen gradient cell, `eigen_cell`, is a
damped, lossy parallel-plate cell periodic along x and y, with a resistive
sheet crossing the y seam.

The tolerances are about 10-50x the error measured on each cell at its
order and mesh.
"""

import itertools
import math
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import rods_mesh as rods_gmsh
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
#: :math:`|E_z(p_2) / E_z(p_1) - e^{-i\mathbf k\cdot\Delta\mathbf r}|` on the
#: C2 band with G = 0.
C2_PHASE_TOL = 1e-2
#: Relative error of a complex C3 frequency.
C3_FREQ_RTOL = 2e-6
#: :math:`|v_g - v_\mathrm{exact}|` on the C2 bands, as a fraction of c (6e-4
#: measured, band 4).
C2_VG_TOL = 3e-3
#: :math:`|v_g - v_\mathrm{exact}|` on the C3 bands, as a fraction of c (1.8e-7
#: measured).
C3_VG_TOL = 5e-6
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


def c1edge_mesh(directory: Path) -> gfem.Mesh:
    """C1's mesh with p1's surface "edge_port" on the last x cell's y = 0 strip.

    Its edges on x = A lie on the pair's second surface; "sheet2" as `c1_mesh`.
    """
    dx = C1_A / 16

    def surface(c):
        return "edge_port" if c[1] < 1e-9 and c[0] > C1_A - dx else ""

    edges = _edges((C1_A, C1_W, C1_H), (16, 1, 1))
    cell = sm.periodic_cell(edges, sheets=(("sheet2", C1_X2),), surface=surface)
    return _mesh(directory, "c1edge", cell)


def c1edge(mesh: gfem.Mesh, *, eps_r=None, conductivity=None) -> gfem.Model:
    """C1 with port p1 touching the seam: a Bloch S far from reciprocal.

    ``eps_r`` and ``conductivity`` per element make "cell" a
    `gfem.MaterialField`; by default it is C1's lossy dielectric.
    """
    material = gfem.Material(eps_r=C1_EPS, tan_delta=C1_TAN_DELTA)
    if eps_r is not None:
        material = gfem.MaterialField(eps_r=eps_r, conductivity=conductivity)
    return gfem.Model(
        mesh,
        materials={"cell": material},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[
            gfem.LumpedPort("p1", "edge_port", "+z", impedance=50.0),
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


def c2_square_mesh(
    directory: Path, cells: tuple[int, int, int] = (8, 8, 1)
) -> gfem.Mesh:
    """C2's cell made square, ``C2_AX`` along x and y: a square lattice for `c2`."""
    cell = sm.periodic_cell(_edges((C2_AX, C2_AX, C2_H), cells))
    return _mesh(directory, "c2_square", cell)


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


def c3(mesh: gfem.Mesh, tan_delta: float | tuple[float, float] = 0.0) -> gfem.Model:
    """C3: periodic in x, PEC at z = 0 and H, default PMC y walls.

    ``tan_delta``: the loss tangent of both layers, or of (low, high).
    """
    low, high = (tan_delta, tan_delta) if np.isscalar(tan_delta) else tan_delta
    return gfem.Model(
        mesh,
        materials={
            "low": gfem.Material(eps_r=C3_EPS_LOW, tan_delta=low),
            "high": gfem.Material(eps_r=C3_EPS_HIGH, tan_delta=high),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
    )


def _tmm_dispersion(
    tan_delta: float | tuple[float, float],
) -> tuple[Callable, Callable]:
    r"""The C3 Rytov function :math:`F(k_0)` of the two layers and its derivative.

    ``tan_delta``: the loss tangent of both layers, or of (low, high). The
    band is :math:`F(k_0) = \cos(k a)`, with
    :math:`n = \sqrt{\varepsilon (1 - i\tan\delta)}`.
    """
    a, d1 = C3_A * MM, C3_D1 * MM
    d2 = a - d1
    low, high = (tan_delta, tan_delta) if np.isscalar(tan_delta) else tan_delta
    n1 = np.sqrt(C3_EPS_LOW * (1 - 1j * low))
    n2 = np.sqrt(C3_EPS_HIGH * (1 - 1j * high))
    r = 0.5 * (n1 / n2 + n2 / n1)

    def terms(k0):
        return (
            np.cos(n1 * k0 * d1),
            np.sin(n1 * k0 * d1),
            np.cos(n2 * k0 * d2),
            np.sin(n2 * k0 * d2),
        )

    def value(k0):
        c1, s1, c2, s2 = terms(k0)
        return c1 * c2 - r * s1 * s2

    def slope(k0):
        c1, s1, c2, s2 = terms(k0)
        return (
            -n1 * d1 * s1 * c2
            - n2 * d2 * c1 * s2
            - r * (n1 * d1 * c1 * s2 + n2 * d2 * s1 * c2)
        )

    return value, slope


def tmm_frequency(
    k: float, guess: complex, tan_delta: float | tuple[float, float] = 0.0
) -> complex:
    r"""The C3 band frequency (Hz, complex when lossy) nearest ``guess`` at k (rad/m).

    The root of the transfer-matrix (Rytov) dispersion relation of the two
    layers, :math:`\cos(n_1 k_0 d_1)\cos(n_2 k_0 d_2) - \tfrac12 (n_1/n_2 +
    n_2/n_1)\sin(n_1 k_0 d_1)\sin(n_2 k_0 d_2) = \cos(k a)`, with
    :math:`n = \sqrt{\varepsilon (1 - i\tan\delta)}`; a pair
    ``tan_delta`` is the loss of (low, high).
    """
    value, _ = _tmm_dispersion(tan_delta)
    a = C3_A * MM

    def residual(fz):
        r = value(2 * math.pi * (fz[0] + 1j * fz[1]) / C0) - np.cos(k * a)
        return [r.real, r.imag]

    re, im = scipy.optimize.fsolve(residual, [guess.real, guess.imag], xtol=1e-14)
    return complex(re, im)


def tmm_group_velocity(
    k: float, frequency: complex, tan_delta: float | tuple[float, float] = 0.0
) -> complex:
    r"""The C3 band's :math:`\mathrm d\omega/\mathrm dk` (m/s, complex when lossy).

    At the root ``frequency``, by implicit differentiation of the Rytov relation
    of `tmm_frequency`: :math:`F'(k_0)\,\mathrm dk_0 = -a\sin(k a)\,\mathrm dk`,
    with :math:`\omega = c k_0` and ``frequency`` (Hz) the root at :math:`k`
    (rad/m) it returns.
    """
    _, slope = _tmm_dispersion(tan_delta)
    k0 = 2 * math.pi * frequency / C0
    return C0 * (-C3_A * MM * np.sin(k * C3_A * MM)) / slope(k0)


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


# --- the sheared cell: two pairs at any angle ----------------------------------------

SHEAR_A, SHEAR_H = 10.0, 1.0  # mm


def sheared_cell_mesh(
    directory: Path, angle: float, cells: tuple[int, int, int] = (3, 3, 1)
) -> gfem.Mesh:
    """A Kuhn box sheared so that the y pair's translation makes ``angle`` (deg) with x.

    Both pairs' translations are ``SHEAR_A`` long: a1 = (a, 0, 0) and
    a2 = (a cos(angle), a sin(angle), 0). The faces stay translates of each
    other, so `sheared_cell` pairs them; 60 and 120 give a hexagonal lattice,
    other angles an oblique one. For geometry only.
    """
    theta = math.radians(angle)
    nodes, *rest = sm.periodic_cell(
        _edges((SHEAR_A, SHEAR_A * math.sin(theta), SHEAR_H), cells)
    )
    nodes = np.array(nodes, dtype=float)
    nodes[:, 0] += nodes[:, 1] / math.tan(theta)
    return _mesh(directory, f"sheared_{angle:g}", (nodes, *rest))


def sheared_cell(mesh: gfem.Mesh) -> gfem.Model:
    """The sheared cell: vacuum, periodic in x and along the sheared y faces."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
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
#: The warning filter of G's solves under `G_K`: its Bloch wave meets the
#: first-order `Absorbing` top at 39.6° at 20 GHz (r = 0.13), so a solve with
#: concrete "cover" values warns `gfem.ObliqueAbsorberWarning`; the
#: checks on G compare gfem with itself (gradients, solvers), not with a
#: perfect absorber.
G_OBLIQUE = "ignore::gfem.ObliqueAbsorberWarning"


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


# --- one pair across ranks: cellx and the periodic box -------------------------------

#: The frequencies of the periodic box, Hz.
BOX_FREQUENCIES = (60e12, 90e12)


def cellx(mesh: gfem.Mesh, *, eps_r=None, conductivity=None, cover=None) -> gfem.Model:
    """G with its x pair only: one pair, as a distributed solve takes.

    ``eps_r`` and ``conductivity`` as `g`; ``cover``: the "cover" material
    (vacuum by default).
    """
    model = g(mesh, eps_r=eps_r, conductivity=conductivity)
    keep = [b for b in model.boundaries if b != gfem.Periodic(("y_min", "y_max"))]
    materials = {**model.materials, "cover": cover or gfem.Material()}
    return model.replace(boundaries=keep, materials=materials)


def periodic_box(mesh: gfem.Mesh, *, eps_r=None, conductivity=None) -> gfem.Model:
    """`mpi_cases.box` periodic along x, with two lumped ports touching the seam.

    Ports on "port" (z = 0) and on the interior "sheet", an absorbing top
    "abc"; "diel" is a `gfem.MaterialField` of ``eps_r`` and ``conductivity``
    per element (lossy constants by default).
    """
    n = len(mesh.elements("diel"))
    eps = np.full(n, 3.0) if eps_r is None else eps_r
    sigma = np.full(n, 1e3) if conductivity is None else conductivity
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "metal": gfem.Material(eps_r=2.0, conductivity=2e5),
            "diel": gfem.MaterialField(eps_r=eps, conductivity=sigma),
        },
        boundaries=[gfem.Periodic(("xmin", "xmax")), gfem.Absorbing("abc")],
        ports=[
            gfem.LumpedPort("p1", "port", "+y", impedance=50.0),
            gfem.LumpedPort("p2", "sheet", "+y", impedance=50.0),
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


# --- Rods: the photonic-crystal tutorial's square lattice of alumina rods -----------

ROD_A = rods_gmsh.CELL  # lattice constant, mm
ROD_R = rods_gmsh.RADIUS  # rod radius (0.2 ROD_A), mm
ROD_EPS = 8.9  # alumina
#: Converged bands of the rod crystal as :math:`\nu = \omega a / 2\pi c`: polarization
#: ("Ez", E along the rods, PEC front and back faces; "Hz", H along the rods,
#: natural faces) -> point -> the four lowest bands, a degenerate pair written
#: twice. Band 1 at Γ is the static field, 0 here and dropped by gfem's static
#: filter. From a curved order-6 2D FEM on the round rod, cross-checked by a
#: plane-wave expansion to 1e-5; the Ez gap, from band 1 at M to band 2 at X,
#: is 31.4 % of its midgap, as published.
ROD_BANDS = {
    "Ez": {
        "Γ": (0.0, 0.582310, 0.627802, 0.627802),
        "X": (0.274706, 0.442519, 0.635953, 0.772234),
        "M": (0.322395, 0.548831, 0.548831, 0.693589),
    },
    "Hz": {
        "Γ": (0.0, 0.627812, 0.823528, 0.823528),
        "X": (0.417567, 0.461676, 0.701194, 0.854966),
        "M": (0.548843, 0.601899, 0.601899, 0.681160),
    },
}
#: Band group velocities of the rod crystal at the segment midpoints of
#: Γ-X-M-Γ, :math:`v_g \cdot \hat t / c`` along each segment's direction :math:`\hat t`
#: (Γ-X at :math:`(1/4, 0)` along +x, X-M at :math:`(1/2, 1/4)` along +y, M-Γ at
#: :math:`(1/4, 1/4)` along :math:`-(1, 1)/\sqrt 2`, reduced coordinates), bands 1-4 as
#: in `ROD_BANDS`; central differences of the same 2D FEM with
#: :math:`\delta\kappa = 10^{-4}`.
ROD_V_G = {
    "Ez": {
        "Γ-X": (0.63165, -0.37635, 0.02559, 0.31346),
        "X-M": (0.14764, 0.31101, -0.25977, -0.22334),
        "M-Γ": (-0.53545, 0.08654, 0.17520, -0.21752),
    },
    "Hz": {
        "Γ-X": (0.88428, -0.34209, -0.35509, 0.09746),
    },
}


def rods_tutorial_mesh(name: str = "rods.msh") -> gfem.Mesh:
    """The committed rod cell ``name`` of ``rods_gmsh.MESHES``, read as a slab (mm).

    Areas "rod" and "air"; the pairs "x_min" / "x_max" and "y_min" /
    "y_max"; the slab's end faces "front" and "back".
    """
    path = Path(__file__).parents[1] / "docs" / "tutorials" / "data" / name
    return gfem.Mesh.from_gmsh_cross_section(path, length_unit="mm")


def rods_slab(mesh: gfem.Mesh, *, ez: bool = True) -> gfem.Model:
    """The tutorial's rod cell (`rods_tutorial_mesh`) as a slab of the 2D crystal.

    Periodic in x and y. PEC "front" and "back" faces leave the Ez bands (E
    along the rods); natural faces (``ez=False``) the Hz bands.
    """
    boundaries = [
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
    ]
    if ez:
        boundaries += [gfem.PEC("front"), gfem.PEC("back")]
    return gfem.Model(
        mesh,
        materials={
            "rod": gfem.Material(eps_r=ROD_EPS),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=boundaries,
    )


def rods_mesh(directory: Path, cells: int = 8) -> gfem.Mesh:
    """The rod cell staircased on a ``cells`` x ``cells`` x 1 grid, one element thick.

    A tetrahedron is "rod" when its centroid is within ``ROD_R`` of the
    cell's centre, "air" otherwise. The thickness is ``ROD_A / cells``.
    """
    edges = _edges((ROD_A, ROD_A, ROD_A / cells), (cells, cells, 1))

    def volume(c):
        inside = math.hypot(c[0] - ROD_A / 2, c[1] - ROD_A / 2) < ROD_R
        return "rod" if inside else "air"

    cell = sm.periodic_cell(edges, volume, paired="xy")
    return _mesh(directory, f"rods{cells}", cell)


def rods(mesh: gfem.Mesh, *, tan_delta: float = 0.0, tm: bool = True) -> gfem.Model:
    """`rods_mesh`'s cell: periodic in x and y, loss tangent ``tan_delta`` in the rod.

    PEC z faces leave the z-uniform modes with :math:`E` along z, the 2D TM
    problem; PMC z faces (``tm=False``) leave the TE one.
    """
    wall = gfem.PEC if tm else gfem.PMC
    return gfem.Model(
        mesh,
        materials={
            "rod": gfem.Material(eps_r=ROD_EPS, tan_delta=tan_delta),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            wall("z_min"),
            wall("z_max"),
        ],
    )


def rods_frequency(nu: float) -> float:
    r"""The frequency, Hz, of the normalized :math:`\nu = \omega a / 2\pi c`."""
    return nu * C0 / (ROD_A * MM)


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
    mesh: gfem.Mesh, ports: list | None = None, tan_delta: float = 0.0
) -> gfem.Model:
    """F2: an eps 4 lamellar grating on an eps 2.25 substrate under air.

    ``tan_delta`` the ridge's loss tangent: 0.05 for G20L, eps 4 (1 - 0.05 i)
    under e^{+iωt}. The default ports
    are every order propagating at normal incidence (`f2_ports`).
    """
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.Material(eps_r=F1_EPS_SUB),
            "ridge": gfem.Material(eps_r=F2_EPS_RIDGE, tan_delta=tan_delta),
            "groove": gfem.Material(),
            "air": gfem.Material(),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=f2_ports() if ports is None else ports,
    )


# --- F2's frozen RCWA references ------------------------------------------------

#: F2's cells at h about lambda0 / 10 with grid lines on the ridge edges
#: (`f2_mesh`), solved at p = 3 for the grating gates below.
F2_FINE_CELLS = (6, 4, 3, 3, 3)
#: F2's cases: (incidence theta, phi from the air in degrees, the ridge's
#: loss tangent).
F2_CASES = {
    "G0": (0.0, 0.0, 0.0),
    "G20": (20.0, 0.0, 0.0),
    "G20c": (20.0, 30.0, 0.0),
    "G20L": (20.0, 0.0, 0.05),
}
#: Diffraction efficiencies of F2 by case, incident polarization (from the
#: air, the specular order) and (face, m): "R" the air side, "T" the
#: substrate; the sum over both output polarizations of |S|^2. Every
#: propagating order of each case is listed; order labels follow gfem's lattice
#: frame (b1 along +x).
#:
#: Method: the planar cases (G0, G20, G20L) from a Li-rule (inverse rule)
#: 1D RCWA at N = 640 harmonics, the conical G20c from nannos 2.6.5
#: ("tangent" formulation) at nh = 321. The two codes agree to 1e-8 at
#: equal truncation; the values move by at most 4e-7 (planar) and 3e-6
#: (conical) from half the truncation, power balance closes to 1e-11 in the
#: lossless cases: the values are good to about 1e-6. The generation
#: scripts are not in the repository; no RCWA package is a test
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


# --- the bare interface: glass below, air above --------------------------------

N_GLASS = 1.5
#: The glass and air layers' thickness, um: each Floquet face that far from
#: the interface.
D = 0.25


def interface_mesh(directory):
    """Glass below z = D, air above, 0.4 x 0.4 um, 3 cells per layer."""
    lateral = np.linspace(0.0, F1_A, 3)
    z = np.linspace(0.0, 2 * D, 7)
    cell = sm.periodic_cell(
        (lateral, lateral, z),
        lambda c: "glass" if c[2] < D else "air",
        paired="xy",
    )
    path = sm.write_msh22(directory / "interface.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def interface_model(mesh, ports, boundaries=()):
    return gfem.Model(
        mesh,
        materials={"glass": gfem.Material(eps_r=N_GLASS**2), "air": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            *boundaries,
        ],
        ports=ports,
    )


# --- F2's gradient cell (driven with Floquet ports) ---------------------------

#: F2's orders with ports on both faces: every order that propagates at
#: either frequency, in either problem, on either side (the -2 order in the
#: substrate under Bloch at f0), so no order is unrequested.
FLOQUET_ORDERS = ((0, 0), (-1, 0), (1, 0), (-2, 0))
#: f0 and 0.8 f0: at 0.8 f0 the ±1 orders are evanescent in the air (rows
#: that read 0), and propagate in the substrate.
FLOQUET_FREQUENCIES = (F1_FREQUENCY, 0.8 * F1_FREQUENCY)
#: The conical Bloch incidence from the air, rad.
FLOQUET_THETA, FLOQUET_PHI = np.radians(20.0), np.radians(30.0)
#: The smallest |gamma| / k of a propagating order (nearer
#: grazing the differences carry the forward's noise divided by the step).
FLOQUET_GRAZING = 1e-2
#: The excited ports: the specular order from the air in both polarizations.
FLOQUET_EXCITE = ("R0,0_TE", "R0,0_TM")


def floquet_gradient_problem(
    model: gfem.Model, *, bloch: bool, fixed_angle: bool
) -> gfem.Driven:
    """The cell's problem, checked away from grazing.

    Every order that propagates on either face at either frequency has a
    port, at ``|gamma| / k >= FLOQUET_GRAZING`` (cos theta, ``k`` in the
    face's medium); the zero row ``R1,0`` does not propagate at 0.8 f0
    (plain) or at either frequency (Bloch). With ``bloch`` the incidence is the conical
    ``FLOQUET_THETA``, ``FLOQUET_PHI`` from the air; ``fixed_angle`` scales
    the Bloch wave vector with frequency from f0.
    """
    k = None
    if bloch:
        k = model.bloch_wave_vector(
            "z_max", FLOQUET_FREQUENCIES[0], theta=FLOQUET_THETA, phi=FLOQUET_PHI
        )

    def k_at(f):
        if k is None or not fixed_angle:
            return k
        return tuple(c * f / FLOQUET_FREQUENCIES[0] for c in k)

    declared = {(p.surface, p.order) for p in model.ports}
    eps = {"z_max": 1.0, "z_min": F1_EPS_SUB}
    a1, a2 = (a * UM for a in F2_A)
    for surface, n2 in eps.items():
        for f in FLOQUET_FREQUENCIES:
            orders = model.propagating_orders(surface, [f], wave_vector=k_at(f))
            assert {(surface, o) for o in orders} <= declared, (surface, orders)
            kt = np.zeros(2) if k is None else np.asarray(k_at(f))[:2]
            k2 = (2 * np.pi * f / C0) ** 2 * n2
            for m, n in orders:
                kappa = kt + 2 * np.pi * np.array([m / a1, n / a2])
                ratio = 1 - kappa @ kappa / k2
                assert ratio < 0 or ratio >= FLOQUET_GRAZING**2, (surface, f, m, n)
    low = FLOQUET_FREQUENCIES[1]
    zero = model.propagating_orders("z_max", [low], wave_vector=k_at(low))
    assert (1, 0) not in zero
    return gfem.Driven(
        frequency=FLOQUET_FREQUENCIES,
        wave_vector=k,
        wave_vector_frequency=FLOQUET_FREQUENCIES[0] if fixed_angle else None,
        excite=FLOQUET_EXCITE,
    )


def floquet_gradient_model(mesh: gfem.Mesh, eps: jax.Array) -> gfem.Model:
    """F2 with ``FLOQUET_ORDERS``, ``eps`` per element over ridge then groove."""
    n = len(mesh.elements("ridge"))
    return (
        f2(mesh, ports=f2_ports(FLOQUET_ORDERS))
        .with_material("ridge", gfem.MaterialField(eps_r=eps[:n]))
        .with_material("groove", gfem.MaterialField(eps_r=eps[n:]))
    )


def floquet_gradient_design(mesh: gfem.Mesh, rng: np.random.Generator) -> jax.Array:
    """A lossy per-element permittivity of the ridge layer, symmetry broken."""
    ridge = rng.uniform(3.5, 4.5, len(mesh.elements("ridge")))
    groove = rng.uniform(1.0, 1.5, len(mesh.elements("groove")))
    return jnp.asarray(np.concatenate([ridge, groove]) * (1 - 0.02j))


# --- the ridge cell: a dielectric meta-atom, infinite along y ----------------------

#: A dielectric ridge (eps 5, 0.6 um high, width ``RIDGE_W0``) on glass
#: (eps 2.1), pitch 0.5 um, lit at 1 um wavelength; Floquet ports TE/TM of the
#: specular order on both faces, offset onto the substrate surface and the
#: ridge's top plane. The grid keeps nodes on the sidewall planes with a fixed
#: number of cells inside and outside the ridge, so every width has the same
#: topology (a smooth family under a boundary morph).
RIDGE_PITCH = 0.5  # um
RIDGE_HEIGHT = 0.6
RIDGE_SUB = 0.3  # substrate thickness
RIDGE_AIR = 0.35  # air thickness above the ridge
RIDGE_EPS = 5.0
RIDGE_EPS_SUB = 2.1
RIDGE_W0 = 0.25  # the nominal width, 1092 tetrahedra
RIDGE_FREQUENCY = C0 / 1.0e-6


def ridge_mesh(directory: Path, w: float = RIDGE_W0) -> gfem.Mesh:
    """The ridge cell of width ``w`` (um), cells ``(3 + 2 x 2, 2, ...)``.

    Regions "substrate", "atom", "air"; faces "x_min", ..., "z_max" plus the
    interfaces "base", "atom_top" and "sidewall".
    """
    lo, hi = RIDGE_PITCH / 2 - w / 2, RIDGE_PITCH / 2 + w / 2
    ex = np.concatenate(
        [
            np.linspace(0, lo, 3),
            np.linspace(lo, hi, 4)[1:],
            np.linspace(hi, RIDGE_PITCH, 3)[1:],
        ]
    )
    ey = np.array([0.0, 0.08, 0.16])
    zs = [
        0.0,
        RIDGE_SUB,
        RIDGE_SUB + RIDGE_HEIGHT,
        RIDGE_SUB + RIDGE_HEIGHT + RIDGE_AIR,
    ]
    ez = [0.0]
    for z0, z1 in itertools.pairwise(zs):
        ez += list(np.linspace(z0, z1, max(1, round((z1 - z0) / 0.1)) + 1)[1:])
    ez = np.array(ez)
    eps = 1e-9
    top = RIDGE_SUB + RIDGE_HEIGHT

    def volume(c):
        if c[2] < RIDGE_SUB:
            return "substrate"
        return "atom" if lo < c[0] < hi and c[2] < top else "air"

    def surface(c):
        for axis, (a, b) in enumerate(((0, RIDGE_PITCH), (0, ey[-1]), (0, ez[-1]))):
            if abs(c[axis] - a) < eps:
                return "xyz"[axis] + "_min"
            if abs(c[axis] - b) < eps:
                return "xyz"[axis] + "_max"
        if abs(c[2] - RIDGE_SUB) < eps:
            return "base"
        if abs(c[2] - top) < eps:
            return "atom_top"
        return "sidewall"

    cell = sm.structured((ex, ey, ez), volume, surface)
    path = sm.write_msh22(directory / f"ridge_{w:.6f}.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def ridge(mesh: gfem.Mesh) -> gfem.Model:
    """The ridge cell between Floquet ports TE and TM on both faces."""
    ports = [
        gfem.FloquetPort(f"{side} {pol}", face, polarization=pol, offset=off * UM)
        for side, face, off in (
            ("sub", "z_min", RIDGE_SUB),
            ("air", "z_max", RIDGE_AIR),
        )
        for pol in ("TE", "TM")
    ]
    return gfem.Model(
        mesh,
        materials={
            "substrate": gfem.Material(eps_r=RIDGE_EPS_SUB),
            "atom": gfem.Material(eps_r=RIDGE_EPS),
            "air": gfem.Material(),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=ports,
    )


def ridge_width_map(morph) -> np.ndarray:
    r"""Per morph parameter, the normal displacement of a unit widening of the ridge.

    The widening moves each sidewall plane outward by half of it, so a node
    on a sidewall has :math:`t_i = u(x_i) \cdot n_i` in um.
    """
    x = np.asarray(morph.points) * 1e6
    n = np.asarray(morph.normals)
    half = np.max(np.abs(x[:, 0] - RIDGE_PITCH / 2))
    on_wall = np.abs(np.abs(x[:, 0] - RIDGE_PITCH / 2) - half) < 1e-6
    u = np.zeros_like(x)
    u[:, 0] = 0.5 * np.sign(np.round(x[:, 0] - RIDGE_PITCH / 2, 9)) * on_wall
    return np.einsum("ij,ij->i", u, n)
