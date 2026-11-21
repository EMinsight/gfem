r"""2D models against closed forms and against the slab.

The loaded line (V1-V4) is the topology tutorial's line in top view
(``docs/tutorials/data/line2d.msh``, mm): 40 mm long and :math:`w = 6` mm wide,
a section of :math:`\varepsilon_r = 10` over :math:`10 < x < 30` between air
feeds, line ports on :math:`x = 0` and :math:`x = 40`, the sides the line
``side``. The section is uniform across the width, so each port's mode is the
same function of :math:`y` in every part of the line and no other mode is
excited: S is that of a three-part transmission line, in closed form.
"""

import math
from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import structured_meshes as sm
from netgen.read_gmsh import ReadGmsh
from scipy.constants import c, epsilon_0, mu_0

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
LENGTH, WIDTH, SECTION, EPS = 40.0, 6.0, (10.0, 30.0), 10.0


def loaded_line(frequency, polarization, *, sine):
    r"""``(S11, S21)`` of the loaded line, closed form.

    The mode is uniform (TEM) or :math:`\sin(\pi y / w)`, so each part carries
    :math:`\beta^2 = k_0^2\varepsilon_r - q^2` with :math:`q = 0` or
    :math:`\pi / w`. At each interface :math:`u` and :math:`w\,\partial_x u`
    are continuous, :math:`w = 1` for Ez and :math:`1/\varepsilon_r` for Hz.
    S is read from the electric field across the line (along :math:`y`). In
    Hz that field is :math:`\partial_x H_z / (i\omega\varepsilon)`: a
    reflected wave's has the opposite sign of its :math:`H_z`, a transmitted
    wave's the same, so S11 is minus the :math:`H_z` reflection and S21 the
    :math:`H_z` transmission.
    """
    k0 = 2 * math.pi * frequency / c * 1e-3  # per mm
    q2 = (math.pi / WIDTH) ** 2 if sine else 0.0
    eps = (1.0, EPS, 1.0)
    beta = [np.sqrt(k0**2 * e - q2 + 0j) for e in eps]
    weight = [1 / e if polarization == "Hz" else 1.0 for e in eps]
    starts = (0.0, *SECTION)

    def value(j, x):  # u of (A_j, B_j) at x, A_j forward, B_j backward
        d = x - starts[j]
        return np.array([np.exp(-1j * beta[j] * d), np.exp(1j * beta[j] * d)])

    def flux(j, x):
        return weight[j] * 1j * beta[j] * value(j, x) * np.array([-1, 1])

    # unknowns B_0, A_1, B_1, A_2; A_0 = 1 and B_2 = 0
    m = np.zeros((4, 4), complex)
    r = np.zeros(4, complex)
    for row, part in ((0, value), (1, flux)):
        left, right = part(0, SECTION[0]), part(1, SECTION[0])
        m[row, :3] = left[1], -right[0], -right[1]
        r[row] = -left[0]
    for row, part in ((2, value), (3, flux)):
        left, right = part(1, SECTION[1]), part(2, SECTION[1])
        m[row, 1:] = left[0], left[1], -right[0]
    b0, _, _, a2 = np.linalg.solve(m, r)
    s11, s21 = b0, a2 * np.exp(-1j * beta[2] * (LENGTH - SECTION[1]))
    return (-s11 if polarization == "Hz" else s11), s21


@pytest.fixture(scope="module")
def line_mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


@pytest.fixture(scope="module")
def fine_line_mesh(tmp_path_factory):
    """The loaded line on a 0.5 mm grid: the sine mode's cases need it."""
    edges = (np.linspace(0, LENGTH, 81), np.linspace(0, WIDTH, 13), np.array([0, 1]))

    def area(centroid):
        return "design" if SECTION[0] < centroid[0] < SECTION[1] else "feed"

    def line(centroid):
        x, y, z = centroid
        if abs(x) < 1e-9 or abs(x - LENGTH) < 1e-9:
            return "port1" if abs(x) < 1e-9 else "port2"
        if abs(y) < 1e-9 or abs(y - WIDTH) < 1e-9:
            return "side"
        return "" if 1e-9 < z < 1 - 1e-9 else "plates"

    path = tmp_path_factory.mktemp("line") / "line_h0.5.msh"
    sm.write_msh22(path, sm.z0_face(sm.structured(edges, area, line)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def line_model(mesh, boundaries=()):
    return gfem.Model(
        mesh,
        materials={
            "feed": gfem.Material(eps_r=1.0),
            "design": gfem.Material(eps_r=EPS),
        },
        boundaries=boundaries,
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )


def assert_loaded_line(model, polarization, frequencies, *, sine, order):
    problem = gfem.Driven(frequencies, excite="p1", polarization=polarization)
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    s = gfem.solve(model, problem, options=options).s
    expected = np.array([loaded_line(f, polarization, sine=sine) for f in frequencies])
    np.testing.assert_allclose(np.asarray(s["p1", "p1"]), expected[:, 0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(s["p2", "p1"]), expected[:, 1], atol=1e-3)


# --- V1-V4: the loaded line, each polarization with each side condition -----------


def test_v1_ez_tem_line_with_pmc_sides_matches_the_closed_form(line_mesh):
    assert_loaded_line(line_model(line_mesh), "Ez", (5e9, 8e9), sine=False, order=2)


def test_v2_hz_tem_line_with_pec_sides_matches_the_closed_form(line_mesh):
    model = line_model(line_mesh, (gfem.PEC("side"),))
    assert_loaded_line(model, "Hz", (5e9, 8e9), sine=False, order=2)


def test_v3_ez_sine_line_with_pec_sides_matches_the_closed_form(fine_line_mesh):
    model = line_model(fine_line_mesh, (gfem.PEC("side"),))
    assert_loaded_line(model, "Ez", (30e9, 40e9), sine=True, order=3)


def test_v4_hz_sine_line_with_pmc_sides_matches_the_closed_form(fine_line_mesh):
    # the sides are the default PMC: no boundary named
    assert_loaded_line(
        line_model(fine_line_mesh), "Hz", (30e9, 40e9), sine=True, order=3
    )


# --- a straight dielectric guide passes its mode -----------------------------------

um = 1e-6
N_CORE, N_CLADDING, CORE = 2.25, 1.444, 0.55  # the bend tutorial's guide (um)
FREQUENCIES = [c / (w * um) for w in (1.50, 1.55, 1.60)]


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    """A straight guide along x, 4 um long, 2.2 um wide between PEC walls, h 0.05 um.

    The core's edges y = +-0.275 um are grid lines.
    """
    half = 1.1
    ys = np.concatenate(
        [
            np.linspace(-half, -CORE / 2, 17),
            np.linspace(-CORE / 2, CORE / 2, 12)[1:],
            np.linspace(CORE / 2, half, 17)[1:],
        ]
    )
    edges = (np.linspace(0, 4, 81), ys, np.array([0.0, 0.05]))

    def area(centroid):
        return "core" if abs(centroid[1]) < CORE / 2 else "cladding"

    def line(centroid):
        x, y, z = centroid
        if abs(x) < 1e-9 or abs(x - 4) < 1e-9:
            return "port1" if abs(x) < 1e-9 else "port2"
        if abs(abs(y) - half) < 1e-9:
            return "outer"
        return "" if 1e-9 < z < 0.05 - 1e-9 else "plates"

    path = tmp_path_factory.mktemp("guide") / "guide.msh"
    sm.write_msh22(path, sm.z0_face(sm.structured(edges, area, line)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=N_CORE**2),
            "cladding": gfem.Material(eps_r=N_CLADDING**2),
        },
        # the walls and the port's ends take one condition: the port's mode is
        # the guide's
        boundaries=(gfem.PEC("outer"),),
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_straight_guide_passes_its_mode(guide, polarization):
    problem = gfem.Driven(FREQUENCIES, excite="p1", polarization=polarization)
    s = gfem.solve(guide, problem, options=gfem.SolverOptions(linear_solver="direct")).s
    s21 = np.abs(np.asarray(s["p2", "p1"]))
    assert s21.min() > 0.9999, s21


# --- V8: the topology line, 2D line ports against the slab's lumped ports ----------


@pytest.mark.parametrize("design", ["air", "ceramic"])
def test_v8_line_ports_match_the_slabs_lumped_ports_and_the_closed_form(
    line_mesh, design
):
    """The parallel-plate line of the topology tutorial, 1 mm high, as a slab.

    Its lumped ports, matched to the line's TEM impedance, launch and absorb
    the same TEM wave as the 2D model's line ports.
    """
    frequencies = (5e9, 8e9)
    eps = EPS if design == "ceramic" else 1.0
    z_line = math.sqrt(mu_0 / epsilon_0) * 1.0 / WIDTH  # ohm, height 1 mm
    materials = {"feed": gfem.Material(), "design": gfem.Material(eps_r=eps)}
    slab = gfem.Model(
        gfem.Mesh.from_gmsh(DATA / "line.msh", length_unit="mm"),
        materials=materials,
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
            gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
        ],
    )
    plane = gfem.Model(
        line_mesh,
        materials=materials,
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )
    options = gfem.SolverOptions(linear_solver="direct")
    s_slab = gfem.solve(slab, gfem.Driven(frequencies, excite="p1"), options=options).s
    problem = gfem.Driven(frequencies, excite="p1", polarization="Ez")
    s_plane = gfem.solve(plane, problem, options=options).s
    closed = np.array([_closed_line(f, eps) for f in frequencies])
    for pair, k in ((("p1", "p1"), 0), (("p2", "p1"), 1)):
        np.testing.assert_allclose(np.asarray(s_plane[pair]), closed[:, k], atol=1e-3)
        np.testing.assert_allclose(np.asarray(s_slab[pair]), closed[:, k], atol=1e-3)
        np.testing.assert_allclose(
            np.asarray(s_plane[pair]), np.asarray(s_slab[pair]), atol=1e-3
        )


def _closed_line(frequency, eps):
    """``(S11, S21)`` of the line with its section of permittivity ``eps``."""
    if eps == EPS:
        return loaded_line(frequency, "Ez", sine=False)
    k0 = 2 * math.pi * frequency / c * 1e-3  # per mm
    return 0.0, np.exp(-1j * k0 * LENGTH)


# --- a metal in Hz is a hole whose edge is named PEC -------------------------------


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    """A 10 mm stub of the line, ended by its line ``wall`` at x = 10 mm.

    Past the wall there is no mesh: in Hz a metal there is a hole, its edge
    the wall.
    """
    edges = (np.linspace(0, 10, 21), np.linspace(0, WIDTH, 7), np.array([0, 1]))

    def line(centroid):
        x, y, z = centroid
        if abs(x) < 1e-9 or abs(x - 10) < 1e-9:
            return "port1" if abs(x) < 1e-9 else "wall"
        if abs(y) < 1e-9 or abs(y - WIDTH) < 1e-9:
            return "side"
        return "" if 1e-9 < z < 1 - 1e-9 else "plates"

    path = tmp_path_factory.mktemp("stub") / "stub.msh"
    sm.write_msh22(path, sm.z0_face(sm.structured(edges, lambda _: "feed", line)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_pec_wall_reflects_the_electric_field_with_minus_one(stub, polarization):
    r"""Both polarizations see :math:`S_{11} = -e^{-2ik_0L}` off a wall at L.

    S reads the electric field, which a perfect conductor reflects with
    :math:`R_E = -1`; in Hz the wall is the natural edge of a hole, where
    :math:`H_z` reflects with :math:`R_H = -R_E = +1`, so it doubles there.
    """
    # the sides must leave the TEM field free: natural in each polarization
    sides = (gfem.PEC("side"),) if polarization == "Hz" else ()
    model = gfem.Model(
        stub,
        materials={"feed": gfem.Material()},
        boundaries=(*sides, gfem.PEC("wall")),
        ports=(gfem.WavePort("p1", "port1"),),
    )
    f = 5e9
    problem = gfem.Driven(f, keep_fields=True, polarization=polarization)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))
    k0 = 2 * math.pi * f / c
    s11 = complex(np.asarray(result.s["p1", "p1"])[0])
    assert abs(s11 - (-np.exp(-2j * k0 * 10e-3))) < 1e-4
    field = (result.h_field if polarization == "Hz" else result.e_field)(f, "p1")
    port, wall = np.asarray(field.at(np.array([[0.0, 3e-3], [10e-3, 3e-3]])))
    # the field's own reflection: R_E = S11, and R_H = -S11 in Hz
    reflection = -s11 if polarization == "Hz" else s11
    incident = abs(port / (1 + reflection))  # the incident wave's amplitude
    expected = 2.0 if polarization == "Hz" else 0.0
    assert abs(abs(wall) / incident - expected) < 1e-4


# --- V6, V7, V9: the bend against its slab -----------------------------------------

THICKNESS = 0.1 * um  # the slab's one cell along z
BEND_SLAB = DATA / "bend.msh"
BEND_2D = DATA / "bend2d.msh"
CLAD = gfem.Material(eps_r=N_CLADDING**2)
CORE_MATERIAL = gfem.Material(eps_r=N_CORE**2)


def bend(mesh, design, boundaries):
    return gfem.Model(
        mesh,
        materials={"cladding": CLAD, "core": CORE_MATERIAL, "design": design},
        boundaries=boundaries,
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )


def slab_and_plane(design, polarization, absorbing, *, keep=False):
    """The slab bend (order 2) and the 2D bend (order 3), solved from p1.

    The slab's plates are PEC for Ez and left natural (PMC) for Hz, so its
    z-uniform field is the 2D field of that polarization.
    """
    slab_mesh = gfem.Mesh.from_gmsh(BEND_SLAB, length_unit="um")
    plane_mesh = gfem.Mesh.from_gmsh(BEND_2D, length_unit="um")
    plates = (gfem.PEC("plates"),) if polarization == "Ez" else ()
    slab = bend(slab_mesh, design, (*plates, absorbing))
    plane = bend(plane_mesh, design, (absorbing,))
    keep_fields = [FREQUENCIES[1]] if keep else False
    slab_result = gfem.solve(
        slab,
        gfem.Driven(FREQUENCIES, excite="p1", keep_fields=keep_fields),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    plane_result = gfem.solve(
        plane,
        gfem.Driven(
            FREQUENCIES,
            excite="p1",
            keep_fields=keep_fields,
            polarization=polarization,
        ),
        options=gfem.SolverOptions(order=3, linear_solver="direct"),
    )
    return slab_result, plane_result


@pytest.fixture(scope="module")
def ez_cladding():
    return slab_and_plane(CLAD, "Ez", gfem.Absorbing("outer", order=2), keep=True)


def assert_same_s(slab_result, plane_result, atol):
    for pair in (("p1", "p1"), ("p2", "p1")):
        np.testing.assert_allclose(
            np.asarray(plane_result.s[pair]), np.asarray(slab_result.s[pair]), atol=atol
        )


def test_v6_ez_bend_matches_its_pec_plate_slab_with_cladding(ez_cladding):
    assert_same_s(*ez_cladding, atol=2e-3)


def test_v6_ez_bend_matches_its_pec_plate_slab_with_core():
    results = slab_and_plane(CORE_MATERIAL, "Ez", gfem.Absorbing("outer", order=2))
    assert_same_s(*results, atol=2e-3)


def _line_mode(k0, dofs, mw, stiff, mass):
    r"""A port line's fundamental mode: :math:`\beta_0` and :math:`\phi`,
    :math:`\phi^T M_w\phi = 1`, its largest entry positive."""
    sub = np.ix_(dofs, dofs)
    lam, vec = sla.eigh((k0**2 * mass - stiff)[sub].toarray(), mw[sub].toarray())
    phi = np.zeros(mw.shape[0])
    phi[dofs] = vec[:, -1] / np.sqrt(vec[:, -1] @ mw[sub] @ vec[:, -1])
    return np.sqrt(lam[-1]), phi * np.sign(phi[np.argmax(np.abs(phi))])


def _solve_with_rank_update(matrix, columns, weights, b):
    r"""Solve :math:`(A + UCU^T)x = b`, :math:`C = \mathrm{diag}(1/w)`, by Woodbury."""
    inverse = matrix.Inverse(inverse="sparsecholesky")
    rhs, sol = matrix.CreateColVector(), matrix.CreateColVector()

    def solve(vector):
        rhs.FV().NumPy()[:] = vector
        sol.data = inverse * rhs
        return sol.FV().NumPy().copy()

    u_cols = np.array(columns).T
    a_inv_u = np.column_stack([solve(col) for col in columns])
    x0 = solve(b)
    small = np.diag(weights) + u_cols.T @ a_inv_u
    return x0 - a_inv_u @ np.linalg.solve(small, u_cols.T @ x0)


def hz_bend_reference(port_term, order=3):
    r"""S11, S21 of the 2D bend, design all cladding, from a plain :math:`H_z` solve.

    :math:`\int\varepsilon_r^{-1}\nabla u\cdot\nabla v - k_0^2\int uv`, the
    order-1 absorbing term :math:`\int\varepsilon_r^{-1}ik\,uv` on ``outer``,
    and on each port a term :math:`\int c\,uv` plus the rank-two correction
    :math:`W = hh^T/(h^T\phi) - gg^T/(g^T\phi)`, :math:`h = i\beta_0 M_w\phi`,
    :math:`g = B_c\phi`, which makes the port exact on its mode :math:`\phi`
    (:math:`M_w` the :math:`\varepsilon_r^{-1}`-weighted line mass, :math:`B_c`
    the port term's matrix, the port's ends natural). ``port_term`` picks
    :math:`c`: ``"line"``, :math:`i\beta_0\varepsilon_r^{-1}` (then
    :math:`W = 0`), or ``"slab"``, :math:`ik_0^2/\beta_0`, a 3D wave port's
    term written in :math:`H_z`. S is E-based: :math:`S_{11} = -S_{11}^H`.
    """
    mesh = ngs.Mesh(ReadGmsh(str(BEND_2D.with_suffix(""))))  # um
    fes = ngs.H1(mesh, order=order, complex=True)
    u, v = fes.TnT()
    real = ngs.H1(mesh, order=order)
    ur, vr = real.TnT()
    n = fes.ndof
    eps = mesh.MaterialCF({"core": N_CORE**2}, default=N_CLADDING**2)

    def scipy_matrix(integrand):
        form = ngs.BilinearForm(real)
        form += integrand
        form.Assemble()
        rows, cols, values = form.mat.COO()
        return sp.csr_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(cols))), (n, n)
        )

    def line_forms(name):
        r"""The port line's :math:`\varepsilon_r^{-1}`, its dofs and line forms."""
        across = ngs.y if name == "port1" else ngs.x
        eps_line = ngs.IfPos((CORE / 2) ** 2 - across**2, N_CORE**2, N_CLADDING**2)
        dofs = sorted(
            {
                d
                for el in mesh.Elements(ngs.BND)
                if el.mat == name
                for d in real.GetDofNrs(el)
            }
        )
        ds = ngs.ds(name)
        dt = ur.Trace().Deriv() * vr.Trace().Deriv()
        mats = [
            scipy_matrix(w * ds) for w in (ur * vr / eps_line, dt / eps_line, ur * vr)
        ]
        return 1 / eps_line, dofs, mats

    lines = {name: line_forms(name) for name in ("port1", "port2")}
    s11, s21 = [], []
    for f in FREQUENCIES:
        k0 = 2 * np.pi * f / c * um
        system = ngs.BilinearForm(fes, symmetric=True)
        system += (ngs.grad(u) * ngs.grad(v) / eps - k0**2 * u * v) * ngs.dx
        system += 1j * k0 / N_CLADDING * u * v * ngs.ds("outer")
        columns, weights, fluxes, modes = [], [], [], []
        for name, (inv_eps, dofs, (mw, stiff, mass)) in lines.items():
            beta, phi = _line_mode(k0, dofs, mw, stiff, mass)
            if port_term == "line":
                system += 1j * beta * inv_eps * u * v * ngs.ds(name)
                g = 1j * beta * (mw @ phi)
            else:
                system += 1j * k0**2 / beta * u * v * ngs.ds(name)
                g = 1j * k0**2 / beta * (mass @ phi)
            h = 1j * beta * (mw @ phi)
            columns += [h, g]
            weights += [h @ phi, -(g @ phi)]  # W = U diag(1/weights) U^T
            fluxes.append(h)
            modes.append(phi)
        system.Assemble()
        x = _solve_with_rank_update(system.mat, columns, weights, 2 * fluxes[0])
        s11.append(1 - fluxes[0] @ x / (fluxes[0] @ modes[0]))
        s21.append(fluxes[1] @ x / (fluxes[1] @ modes[1]))
    return np.array(s11), np.array(s21)


@pytest.fixture(scope="module")
def hz_cladding():
    # order 1 on both: the slab's order 2 is its order 1 for this field
    return slab_and_plane(CLAD, "Hz", gfem.Absorbing("outer", order=1))


def test_v7a_hz_bend_is_the_plain_hz_solve_with_the_line_port_term(hz_cladding):
    r"""The 2D Hz bend against :func:`hz_bend_reference` with
    :math:`\partial_n H_z + i\beta_0 H_z = 0` on its ports: the same forms on
    the same triangles, so equal to round-off.
    """
    _, plane_result = hz_cladding
    s11, s21 = hz_bend_reference(port_term="line")
    np.testing.assert_allclose(np.asarray(plane_result.s["p1", "p1"]), s11, atol=1e-10)
    np.testing.assert_allclose(np.asarray(plane_result.s["p2", "p1"]), s21, atol=1e-10)


def test_v7b_hz_slab_is_the_plain_hz_solve_with_the_slab_port_term(hz_cladding):
    r"""The PMC-plate slab against :func:`hz_bend_reference` with
    :math:`\partial_n H_z + i(k^2/\beta_0)H_z = 0` on its ports.

    A 3D wave port is impedance-matched to TE waves of its mode's
    :math:`\beta_0`, which for the slab's TM field is an impedance
    :math:`\omega\mu/\beta_0`; the 2D Hz line port is matched to TM waves.
    Both are exact on the port's own mode but reflect the bend's radiation
    with opposite signs (:math:`\pm 0.15` in the cladding at normal
    incidence), so the slab and the 2D bend differ by about 3e-3 in S. The
    reference with the slab's port term shows they solve the same Hz physics
    otherwise; what is left is the slab's edge field against H1.
    """
    slab_result, _ = hz_cladding
    s11, s21 = hz_bend_reference(port_term="slab")
    np.testing.assert_allclose(np.asarray(slab_result.s["p1", "p1"]), s11, atol=2e-3)
    np.testing.assert_allclose(np.asarray(slab_result.s["p2", "p1"]), s21, atol=2e-3)


def test_v9_the_2d_field_is_the_slabs_times_the_square_root_of_its_thickness(
    ez_cladding,
):
    r"""1 W/m in 2D against 1 W through a slab h thick: :math:`E^{2D} = \sqrt{h}E`.

    At 50 points of the cladding beside the input arm, in its guided tail.
    """
    slab_result, plane_result = ez_cladding
    f = FREQUENCIES[1]
    xs = np.linspace(-2.4, -1.6, 5)
    ys = np.array([0.35, 0.5, 0.7, 0.85, 1.0])
    xy = np.array([(x, s * y) for x in xs for y in ys for s in (-1, 1)]) * um
    mid = np.column_stack([xy, np.full(len(xy), THICKNESS / 2)])
    slab = np.asarray(slab_result.e_field(f, "p1").at(mid))[:, 2]
    plane = np.asarray(plane_result.e_field(f, "p1").at(xy))
    expected = math.sqrt(THICKNESS) * slab
    error = np.linalg.norm(plane - expected) / np.linalg.norm(expected)
    assert error < 1e-2
