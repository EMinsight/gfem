"""Driven solves with Floquet ports: S on coarse cells against closed forms.

Cell F1 of `periodic_cases` (an eps 4 slab on an eps 2.25 substrate, air
above, Floquet ports on both open faces) at normal incidence, coarse (p = 2),
against the tangential-E Airy form in ADR 0147's lattice frame; a bare
glass/air interface; a cell closed by PEC behind one Floquet face; the
`DrivenResult.diffraction_orders` it reports; and the `bddc_gmres` solver
against `direct`.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem

P2 = gfem.SolverOptions(order=2, threads=2)


@pytest.fixture(scope="module")
def f1_mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("f1"))


@pytest.fixture(scope="module")
def f1_result(f1_mesh):
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"])
    return gfem.solve(pc.f1(f1_mesh), problem, options=P2)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_f1_at_normal_incidence_matches_the_airy_form(f1_result, pol):
    r, t = pc.f1_airy(pol)
    s = f1_result.s
    assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_COARSE_TOL
    assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_COARSE_TOL


def test_f1_reports_the_diffraction_order_of_each_floquet_port(f1_result):
    k0 = 2 * math.pi * pc.F1_FREQUENCY / pc.C0
    for name, surface, index in (("R_TM", "z_max", 1.0), ("T_TE", "z_min", 1.5)):
        order = f1_result.diffraction_orders[name]
        assert order.port == name
        assert order.surface == surface
        assert order.order == (0, 0)
        assert order.polarization == name[-2:]
        assert order.index == pytest.approx(index)
        np.testing.assert_array_equal(order.frequency, [pc.F1_FREQUENCY])
        np.testing.assert_array_equal(order.wave_vector, np.zeros((1, 3)))
        np.testing.assert_allclose(order.gamma, [1j * k0 * index], rtol=1e-12)
        np.testing.assert_array_equal(order.propagating, [True])
        np.testing.assert_array_equal(order.theta, [0.0])
        np.testing.assert_array_equal(order.phi, [0.0])
    assert set(f1_result.diffraction_orders) == {"R_TE", "R_TM", "T_TE", "T_TM"}


def test_the_diffraction_orders_round_trip_through_a_result_bundle(f1_mesh, tmp_path):
    job = gfem.Job(pc.f1(f1_mesh), gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE"]), P2)
    result = job.run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert set(loaded.diffraction_orders) == set(result.diffraction_orders)
    for name, order in result.diffraction_orders.items():
        back = loaded.diffraction_orders[name]
        assert isinstance(back, gfem.DiffractionOrderResult)
        for field in ("port", "order", "polarization", "surface", "index"):
            assert getattr(back, field) == getattr(order, field)
        for field in ("frequency", "wave_vector", "gamma", "propagating", "theta"):
            np.testing.assert_array_equal(getattr(back, field), getattr(order, field))


def test_bddc_gmres_agrees_with_direct(f1_mesh, f1_result):
    options = gfem.SolverOptions(order=2, threads=2, linear_solver="bddc_gmres")
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"])
    iterative = gfem.solve(pc.f1(f1_mesh), problem, options=options)
    error = np.abs(np.asarray(iterative.s.values) - np.asarray(f1_result.s.values))
    assert error.max() <= 10 * options.tolerance


# --- a bare interface and a closed cell -------------------------------------------

N_GLASS = 1.5
#: The glass and air layers' thickness, um: each Floquet face that far from
#: the interface.
D = 0.25


def _interface_mesh(directory):
    """Glass below z = D, air above, 0.4 x 0.4 um, 3 cells per layer."""
    lateral = np.linspace(0.0, pc.F1_A, 3)
    z = np.linspace(0.0, 2 * D, 7)
    cell = sm.periodic_cell(
        (lateral, lateral, z),
        lambda c: "glass" if c[2] < D else "air",
        paired="xy",
    )
    path = sm.write_msh22(directory / "interface.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _interface_model(mesh, ports, boundaries=()):
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


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_a_bare_glass_air_interface_reflects_n_minus_1_over_n_plus_1(tmp_path, pol):
    ports = [
        gfem.FloquetPort("glass", "z_min", polarization=pol, offset=D * 1e-6),
        gfem.FloquetPort("air", "z_max", polarization=pol, offset=D * 1e-6),
    ]
    model = _interface_model(_interface_mesh(tmp_path), ports)
    options = gfem.SolverOptions(order=3, threads=2)
    s = gfem.solve(
        model, gfem.Driven([pc.F1_FREQUENCY], excite=["glass"]), options=options
    ).s
    # tangential E at the interface: r = (n - 1)/(n + 1) seen from the glass,
    # t = 1 + r, times sqrt(Z_glass / Z_air) = sqrt(1 / n) in power waves
    r = (N_GLASS - 1) / (N_GLASS + 1)
    assert abs(complex(s["glass", "glass"][0]) - r) <= 1e-3
    assert abs(complex(s["air", "glass"][0]) - (1 + r) / math.sqrt(N_GLASS)) <= 1e-3


def test_a_cell_closed_by_pec_reflects_all_power(tmp_path):
    ports = [
        gfem.FloquetPort(f"R_{pol}", "z_max", polarization=pol) for pol in ("TE", "TM")
    ]
    model = _interface_model(
        _interface_mesh(tmp_path), ports, boundaries=[gfem.PEC("z_min")]
    )
    options = gfem.SolverOptions(order=3, threads=2)
    s = gfem.solve(model, gfem.Driven([pc.F1_FREQUENCY]), options=options).s
    values = np.asarray(s.values)[0]
    np.testing.assert_allclose(np.abs(np.diag(values)), 1.0, atol=1e-6)


def test_exciting_an_evanescent_order_raises_before_any_solve(f1_mesh):
    ports = [
        *pc.f1_ports(),
        gfem.FloquetPort("R1", "z_max", order=(1, 0), polarization="TE"),
    ]
    with pytest.raises(
        ValueError,
        match=r"FloquetPort 'R1' \(order \(1, 0\), TE\) cannot be excited at "
        r".*excite=\['R_TE', 'R_TM', 'T_TE', 'T_TM'\]",
    ):
        gfem.solve(pc.f1(f1_mesh, ports), gfem.Driven([pc.F1_FREQUENCY]), options=P2)
