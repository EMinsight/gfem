r"""Electromagnetic heat against transmission-line closed forms.

The heat a driven field leaves in the model (`gfem.dissipated_power`, and
through it `gfem.electromagnetic_heating`) is checked against problems whose
loss is known in closed form, each carrying the discretization error of its
field only:

- V-H1, a lossy section on the topology tutorial's slab line
  (``docs/tutorials/data/line.msh``, lumped ports at the line's TEM
  impedance): the section's loss is that of a three-part line with a complex
  permittivity (tests/loaded_line.py).
- V-H2, the same line as a cross-section slab of ``line2d.msh``, in Ez (PEC
  front and back) and Hz (PEC sides): the slab's heat per length against the
  native 2D model's :math:`1 - \sum|S|^2` and against the closed form.
- V-H3, a parallel-plate stub ended by a thick `gfem.Conductor` wall: the wall
  takes :math:`1 - |\Gamma|^2`, :math:`\Gamma = (Z_s - \eta_0)/(Z_s + \eta_0)`.
- V-H4, the stub ended by a volume metal block, its skin resolved, shorted
  behind: the block takes :math:`1 - |\Gamma|^2` of a shorted lossy line,
  :math:`Z_{in} = Z_m \tanh(\gamma_m t)`.
- V-H5, the conductor stub as a heater: the feed conducts heat to a fixed
  temperature at the port face, the wall's sheet heat is uniform across the
  wall (TEM), so the rise is linear along the stub, exact at order 2.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import structured_meshes as sm
from loaded_line import WIDTH, absorbed
from scipy.constants import c, epsilon_0, mu_0

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
ETA = math.sqrt(mu_0 / epsilon_0)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
FREQUENCIES = (5e9, 8e9)


def _lost(result, frequency, model, port="p1"):
    lost = gfem.dissipated_power(result.e_field(frequency, port), model=model)
    return {name: float(value) for name, value in lost.items()}


# --- V-H1: a lossy section on the slab line ------------------------------------


@pytest.fixture(scope="module")
def slab_line():
    return gfem.Mesh.from_gmsh(DATA / "line.msh", length_unit="mm")


@pytest.mark.parametrize("tan_delta", [0.02, 0.1])
def test_v_h1_a_lossy_section_dissipates_the_closed_form_loss(slab_line, tan_delta):
    z_line = ETA * 1.0 / WIDTH  # ohm: the 1 mm high, 6 mm wide line
    model = gfem.Model(
        slab_line,
        materials={
            "feed": gfem.Material(),
            "design": gfem.Material(eps_r=10.0, tan_delta=tan_delta),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
            gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
        ],
    )
    problem = gfem.Driven(FREQUENCIES, excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    for f in FREQUENCIES:
        lost = _lost(result, f, model)
        closed = absorbed(f, "Ez", 10.0 * (1 - 1j * tan_delta))
        assert lost["feed"] == 0.0
        # gate 1e-5 (7.3e-6 measured, tan delta 0.02 at 8 GHz)
        assert sum(lost.values()) == pytest.approx(closed, rel=1e-5)


# --- V-H2: the cross-section slab against native 2D ---------------------------

CROSS_TAN_DELTA = 0.05
PER_LENGTH = 2.0  # W/m incident


def _cross_models(polarization):
    plane_mesh = gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")
    slab_mesh = gfem.Mesh.from_gmsh_cross_section(DATA / "line2d.msh", length_unit="mm")
    materials = {
        "feed": gfem.Material(thermal_conductivity=0.03),
        "design": gfem.Material(
            eps_r=10.0, tan_delta=CROSS_TAN_DELTA, thermal_conductivity=1.0
        ),
    }
    side = (gfem.PEC("side"),) if polarization == "Hz" else ()
    plates = (gfem.PEC("front"), gfem.PEC("back")) if polarization == "Ez" else ()
    ports = (gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2"))
    plane = gfem.Model(plane_mesh, materials=materials, boundaries=side, ports=ports)
    sink = gfem.FixedTemperature("side", temperature=300.0)
    slab = gfem.Model(
        slab_mesh,
        materials=materials,
        boundaries=(*side, *plates, sink),
        ports=ports,
    )
    return plane, slab


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_v_h2_the_slab_heats_as_the_2d_model_and_the_closed_form_lose(polarization):
    plane, slab = _cross_models(polarization)
    plane_problem = gfem.Driven(FREQUENCIES, excite="p1", polarization=polarization)
    s = gfem.solve(plane, plane_problem, options=OPTIONS).s
    driven = gfem.solve(
        slab, gfem.Driven(FREQUENCIES, excite="p1", keep_fields=True), options=OPTIONS
    )
    for i, f in enumerate(FREQUENCIES):
        heat = gfem.electromagnetic_heating(
            driven,
            model=slab,
            frequency=f,
            incident_power_per_length={"p1": PER_LENGTH},
        )
        steady = gfem.solve(slab, gfem.SteadyThermal(heat), options=OPTIONS)
        per_length = float(steady.power_per_length)
        plane_lost = 1 - sum(
            abs(complex(np.asarray(s[k, "p1"])[i])) ** 2 for k in ("p1", "p2")
        )
        closed = absorbed(f, polarization, 10.0 * (1 - 1j * CROSS_TAN_DELTA))
        # gate 1e-5 (Hz at 8 GHz: 6.6e-6 against 2D, 4.2e-6 against the
        # closed form measured)
        assert per_length == pytest.approx(PER_LENGTH * plane_lost, rel=1e-5)
        assert per_length == pytest.approx(PER_LENGTH * closed, rel=1e-5)


# --- V-H3 to V-H5: a parallel-plate stub ended by a wall -----------------------

STUB_F = 5e9
STUB_LENGTH, STUB_HEIGHT = 10.0, 1.0  # mm, the stub's length and plate gap


def _stub(path, x_edges, metal_from=None):
    """The 6 mm wide stub, plates at z = 0 and 1 mm, port at x = 0, wall at the end.

    With ``metal_from``, the elements past that x are the region ``metal``.
    """
    edges = (x_edges, np.linspace(0, WIDTH, 7), np.array([0.0, STUB_HEIGHT]))
    end = x_edges[-1]

    def region(centroid):
        return (
            "metal" if metal_from is not None and centroid[0] > metal_from else "feed"
        )

    def face(centroid):
        x, y, z = centroid
        if abs(x) < 1e-9:
            return "port1"
        if abs(x - end) < 1e-9:
            return "wall"
        if abs(y) < 1e-9 or abs(y - WIDTH) < 1e-9:
            return "side"
        return "plates" if abs(z) < 1e-9 or abs(z - STUB_HEIGHT) < 1e-9 else ""

    sm.write_msh22(path, sm.structured(edges, region, face))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _wall_absorbed(sigma):
    """``1 - |Gamma|^2`` of a thick conductor ending an air line."""
    omega = 2 * math.pi * STUB_F
    delta = math.sqrt(2 / (mu_0 * sigma * omega))
    zs = (1 + 1j) / (sigma * delta)
    gamma = (zs - ETA) / (zs + ETA)
    return 1 - abs(gamma) ** 2


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    path = tmp_path_factory.mktemp("stub") / "stub.msh"
    return _stub(path, np.linspace(0, STUB_LENGTH, 21))  # h = 0.5 mm


def _wall_model(mesh, sigma, thermal=()):
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(thermal_conductivity=WALL_K)},
        boundaries=(gfem.PEC("plates"), gfem.Conductor("wall", sigma), *thermal),
        ports=(gfem.WavePort("p1", "port1"),),
    )


WALL_K = 0.5  # W/(m K), the feed's


@pytest.mark.parametrize("sigma", [8.0, 80.0])
def test_v_h3_a_conductor_wall_takes_one_minus_its_reflection(stub, sigma):
    model = _wall_model(stub, sigma)
    problem = gfem.Driven(STUB_F, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    lost = _lost(result, STUB_F, model)
    assert lost["feed"] == 0.0
    # gate 1e-7 (1.7e-8 measured; copper's 2e-4 absorption, not gated, 1.8e-8)
    assert lost["wall"] == pytest.approx(_wall_absorbed(sigma), rel=1e-7)


def test_v_h4_a_resolved_skin_takes_the_shorted_lossy_line_s_loss(tmp_path):
    sigma, thickness = 8.0, 10.0  # S/m, mm of metal
    omega = 2 * math.pi * STUB_F
    delta = math.sqrt(2 / (mu_0 * sigma * omega)) * 1e3  # mm, 2.5
    cells = math.ceil(thickness / delta * 4)  # 4 cells per skin depth
    x = np.concatenate(
        [
            np.linspace(0, STUB_LENGTH, 21),
            np.linspace(STUB_LENGTH, STUB_LENGTH + thickness, cells + 1)[1:],
        ]
    )
    mesh = _stub(tmp_path / "block.msh", x, metal_from=STUB_LENGTH)
    model = gfem.Model(
        mesh,
        materials={
            "feed": gfem.Material(),
            "metal": gfem.Material(conductivity=sigma),
        },
        boundaries=(gfem.PEC("plates"), gfem.PEC("wall")),
        ports=(gfem.WavePort("p1", "port1"),),
    )
    result = gfem.solve(model, gfem.Driven(STUB_F, keep_fields=True), options=OPTIONS)
    lost = _lost(result, STUB_F, model)
    eps_metal = 1 - 1j * sigma / (epsilon_0 * omega)
    k0 = omega / c * 1e-3  # per mm
    z_metal = ETA / np.sqrt(eps_metal)
    z_in = z_metal * np.tanh(1j * k0 * np.sqrt(eps_metal) * thickness)
    closed = 1 - abs((z_in - ETA) / (z_in + ETA)) ** 2
    assert lost["feed"] == 0.0
    # gate 5e-5 (1.25e-5 measured)
    assert lost["metal"] == pytest.approx(closed, rel=5e-5)


# --- V-H5: the wall as a heater ----------------------------------------------

WALL_SIGMA = 80.0
AREA = WIDTH * STUB_HEIGHT * 1e-6  # m^2, the stub's cross-section
HEAT_LENGTH = STUB_LENGTH * 1e-3  # m, port face to wall


def _wall_points():
    """Three points (m) across the wall, mid-gap."""
    y = np.array([0.5, 3.0, 5.5]) * 1e-3
    return np.stack([np.full(3, HEAT_LENGTH), y, np.full(3, 0.5e-3)], axis=1)


@pytest.fixture(scope="module")
def wall_heater(stub):
    """The stub's driven result and the wall's absorbed fraction of 1 W."""
    sink = gfem.FixedTemperature("port1", temperature=300.0)
    model = _wall_model(stub, WALL_SIGMA, (sink,))
    result = gfem.solve(model, gfem.Driven(STUB_F, keep_fields=True), options=OPTIONS)
    return model, result, _wall_absorbed(WALL_SIGMA)


POWER = 0.2  # W incident


def test_v_h5_the_wall_heater_rises_as_its_absorbed_power_conducts(wall_heater):
    model, driven, fraction = wall_heater
    heat = gfem.electromagnetic_heating(
        driven, model=model, frequency=STUB_F, incident_power={"p1": POWER}
    )
    steady = gfem.solve(model, gfem.SteadyThermal(heat), options=OPTIONS)
    rise = np.asarray(steady.rise.at(_wall_points()))
    # gate 1e-6 (2.4e-7 measured)
    np.testing.assert_allclose(
        rise, POWER * fraction * HEAT_LENGTH / (WALL_K * AREA), rtol=1e-6
    )
    flow = float(gfem.heat_flow(steady, "port1", model=model))
    # the balance is exact: the port face takes all of the wall's heat
    assert flow == pytest.approx(float(steady.power), rel=1e-12)
    # the wall's closed form: 1.7e-8 measured
    assert flow == pytest.approx(POWER * fraction, rel=1e-7)


def test_v_h5_a_convection_wall_keeps_its_share(wall_heater):
    model, driven, fraction = wall_heater
    h = 100.0  # W/(m^2 K), against k / L = 50
    cooled = model.replace(
        boundaries=[
            *model.boundaries,
            gfem.Convection("wall", coefficient=h, ambient=300.0),
        ]
    )
    heat = gfem.electromagnetic_heating(
        driven, model=cooled, frequency=STUB_F, incident_power={"p1": POWER}
    )
    steady = gfem.solve(cooled, gfem.SteadyThermal(heat), options=OPTIONS)
    share = h / (h + WALL_K / HEAT_LENGTH)
    wall = float(gfem.heat_flow(steady, "wall", model=cooled))
    port = float(gfem.heat_flow(steady, "port1", model=cooled))
    power = float(steady.power)
    # the split is exact for the linear profile (1e-14 measured)
    assert wall == pytest.approx(share * power, rel=1e-12)
    assert port == pytest.approx((1 - share) * power, rel=1e-12)
    assert power == pytest.approx(POWER * fraction, rel=1e-7)
