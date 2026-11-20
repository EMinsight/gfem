"""DC conduction against closed forms with current crowding.

A quarter annulus between two radial contacts carries an azimuthal
current: the potential is the angle over pi/2, so R = (pi/2) / (sigma t
ln(r2/r1)) and the Joule density sigma (2 / (pi r))^2 crowds to the inner
edge (4x the outer one here). The mesh has its nodes on the circles and
straight sides. A square-corner L bend has no closed form; its corner's
excess resistance, in squares (about 0.56 in the thin-film literature),
is frozen at the finest mesh measured at implementation.
"""

import math

import numpy as np
import pytest
import structured_meshes as sm
from numpy.polynomial import legendre

import gfem

UM = 1e-6
SIGMA = 1e6
R1, R2, T = 2.0, 4.0, 0.2  # um
#: The corner of the L bend in squares, at 32 cells per um and order 3
#: (79,872 tetrahedra; 0.557524 / 0.558255 at order 2 and 16 / 32 cells,
#: 0.558244 at order 3 and 16 cells).
L_CORNER = 0.558538


def _arc(tmp_path, h, order):
    n_r = round((R2 - R1) / h)
    n_theta = math.ceil(R2 * math.pi / 2 / h)
    edges = (
        np.linspace(R1, R2, n_r + 1),
        np.linspace(0, math.pi / 2, n_theta + 1),
        np.linspace(0, T, max(1, round(T / h)) + 1),
    )

    def surface(c):
        if abs(c[1]) < 1e-12:
            return "a"
        return "b" if abs(c[1] - math.pi / 2) < 1e-12 else ""

    def mapping(p):
        r, theta, z = p.T
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    path = sm.write_msh22(
        tmp_path / "arc.msh", sm.structured(edges, lambda _: "m", surface, mapping)
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(mesh, materials={"m": gfem.Material(conductivity=SIGMA)})
    dc = gfem.solve(
        model,
        gfem.DCConduction({"a": "a"}, ground="b"),
        options=gfem.SolverOptions(order=order, linear_solver="direct"),
    )
    return mesh, dc


def test_an_annulus_bend_has_the_closed_form_resistance(tmp_path):
    _, dc = _arc(tmp_path, 0.4, 2)
    exact = (math.pi / 2) / (SIGMA * T * UM * math.log(R2 / R1))
    # 5.7e-6 measured (straight-sided mesh, nodes on the circles)
    assert float(dc.resistance[0, 0]) == pytest.approx(exact, rel=5e-5)


def _tet_rule(n):
    """A conical-product rule on the unit tetrahedron, exact to degree 2n - 3."""
    x, w = legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    a, b, c = (g.ravel() for g in np.meshgrid(x, x, x, indexing="ij"))
    wa, wb, wc = (g.ravel() for g in np.meshgrid(w, w, w, indexing="ij"))
    points = np.stack([a, b * (1 - a), c * (1 - a) * (1 - b)], axis=1)
    return points, wa * wb * wc * (1 - a) ** 2 * (1 - b)


def test_the_joule_source_is_the_exact_element_mean(tmp_path):
    mesh, dc = _arc(tmp_path, 0.2, 2)
    corners = np.asarray(mesh.nodes)[mesh.tetrahedra]
    jac = np.stack([corners[:, k] - corners[:, 0] for k in (1, 2, 3)], axis=2)
    points, weights = _tet_rule(10)
    x = corners[:, 0][:, None, :] + np.einsum("eij,qj->eqi", jac, points)
    density = SIGMA * (2 / (math.pi * np.hypot(x[..., 0], x[..., 1]))) ** 2  # 1 V
    mean = (density * weights).sum(axis=1) / weights.sum()
    got = np.asarray(dc.dissipation[:, 0, 0])
    volume = np.abs(np.linalg.det(jac)) / 6
    error = math.sqrt(np.sum(volume * (got - mean) ** 2) / np.sum(volume * mean**2))
    # 2.9e-4 measured at h = 0.2 (1.2e-3 at h = 0.4)
    assert error < 1e-3
    # the inner edge carries about 4x the outer density
    assert got.max() / got.min() > 3.0


def test_the_l_bend_corner_is_frozen(tmp_path):
    arm, width, n = 6.0, 1.0, 16
    xs = np.linspace(0, arm + width, round((arm + width) * n) + 1)
    edges = (xs, xs, np.array([0.0, T]))

    def volume(c):
        x, y, _ = c
        if (y < width and x < arm + width) or (x > arm and y < arm + width):
            return "m"
        return None

    def surface(c):
        if abs(c[0]) < 1e-9:
            return "a"
        return "b" if abs(c[1] - (arm + width)) < 1e-9 else ""

    path = sm.write_msh22(tmp_path / "l.msh", sm.carve(edges, volume, surface))
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(mesh, materials={"m": gfem.Material(conductivity=SIGMA)})
    dc = gfem.solve(
        model,
        gfem.DCConduction({"a": "a"}, ground="b"),
        options=gfem.SolverOptions(order=3, linear_solver="direct"),
    )
    squares = float(dc.resistance[0, 0]) * SIGMA * T * UM
    corner = squares - 2 * arm / width
    # 5.3e-4 measured (0.558244 at 16 cells per um, order 3)
    assert corner == pytest.approx(L_CORNER, rel=1e-3)
