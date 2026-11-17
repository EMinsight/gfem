"""The route phase against the 2D shift and against station re-solves (ADR 0183).

A heater cross-section shaped like femwell's TiN heater (a silicon wafer
under a 2 um oxide box, a 500 x 220 nm silicon core, a 2 um x 140 nm heater
2 um above it, oxide cladding), extruded along z in prism layers; the
wafer's bottom sinks the heat and the ends are insulated. One optical
window (the core in oxide, a cross-section slab) is swept along the core.

- With the heater along the whole extrusion the rise is z-invariant, so the
  route phase over the length L is L times the window's 2D shift on the
  section's slab (-2.7e-5 on femwell's section, thermal probes/09).
- With the heater in the middle 4 um of a 16 um extrusion only, the
  first-order phase per length at the middle station is within 1e-3 of a
  mode re-solve of the window placed there with eps(rise), at the drive
  that would give a pi shift over 320 um (7e-5 on the spiral, thermal probes/09).
"""

import jax
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem

UM = 1e-6
WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
N_SI, N_OX = 3.4777, 1.444
HALF = 6.0  # um, half width of the section
THICKNESS = 0.2e-6  # m, the slabs' thickness
#: The extrusion: L along z in LAYERS prism layers.
LENGTH, LAYERS = 4.0, 8
#: A heater length for the pi drive: P_pi over it is about 70 W/m.
PI_LENGTH = 320e-6
CORE = (-0.25, 0.0, 0.25, 0.22, "core", 0.04)
SECTION = [
    (-HALF, -2.5, HALF, -2.0, "wafer", None),
    (-HALF, -2.0, HALF, 0.0, "box", None),
    (-HALF, 0.0, HALF, 2.8, "clad", None),
    CORE,
    (-1.0, 2.22, 1.0, 2.36, "heater", 0.1),
]
WINDOW = [(-1.75, -1.0, 1.75, 1.22, "clad", None), CORE]
BOTTOM = [(lambda p: abs(p[1] + 2.5) < 1e-9, "bottom")]
OXIDE = {
    "eps_r": N_OX**2,
    "thermal_conductivity": 1.38,
    "thermo_optic_coefficient": 1e-5,
}
MATERIALS = {
    "wafer": gfem.Material(thermal_conductivity=148.0),
    "box": gfem.Material(**OXIDE),
    "clad": gfem.Material(**OXIDE),
    "core": gfem.Material(
        eps_r=N_SI**2, thermal_conductivity=90.0, thermo_optic_coefficient=1.86e-4
    ),
    "heater": gfem.Material(thermal_conductivity=28.0),
}
#: The route: along the core's bottom centre line, up the section's y.
ROUTE = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, LENGTH]]) * UM
UP = (0.0, 1.0, 0.0)


def _thermal(mesh):
    return gfem.Model(
        mesh,
        materials={k: v for k, v in MATERIALS.items() if k in mesh.volumes},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )


def _window_model(mesh):
    return gfem.Model(
        mesh,
        materials={k: MATERIALS[k] for k in ("clad", "core")},
        ports=[gfem.WavePort("p", "front")],
    )


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    path = tmp_path_factory.mktemp("window") / "window.msh"
    return _window_model(tc.section(path, WINDOW, 0.6, [], thickness=THICKNESS))


def _route(window, rise):
    return gfem.route_phase(
        window, rise, routes={"core": ROUTE}, port="p", frequency=FREQUENCY, up=UP
    )


def test_a_z_invariant_route_is_the_2d_shift_times_its_length(tmp_path_factory, window):
    folder = tmp_path_factory.mktemp("extrusion")
    slab = tc.section(folder / "slab.msh", SECTION, 0.6, BOTTOM, thickness=THICKNESS)
    per_length = gfem.HeatSource.uniform("heater", power_per_length=1.0)
    flat = gfem.solve(_thermal(slab), gfem.SteadyThermal(per_length))
    shift = gfem.thermo_optic_shift(window, flat.rise, port="p", frequency=FREQUENCY)

    mesh = tc.extruded(
        folder / "3d.msh", SECTION, 0.6, BOTTOM, length=LENGTH, layers=LAYERS
    )
    heat = gfem.HeatSource.uniform("heater", power=LENGTH * UM)  # 1 W/m
    steady = gfem.solve(_thermal(mesh), gfem.SteadyThermal(heat))
    route = _route(window, steady.rise)
    assert route.length["core"] == pytest.approx(LENGTH * UM, rel=1e-14)
    # (-2.3e-5 measured)
    per_metre = float(route.phase["core"]) / route.length["core"]
    assert per_metre == pytest.approx(float(shift.phase_per_length), rel=1e-3)
    # the window reads its own dn/dT, not the 3D model's
    marked = gfem.Model(
        mesh,
        materials={
            **_thermal(mesh).materials,
            "core": gfem.Material(
                eps_r=N_SI**2, thermal_conductivity=90.0, thermo_optic_coefficient=1.0
            ),
        },
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    again = _route(window, gfem.solve(marked, gfem.SteadyThermal(heat)).rise)
    assert float(again.phase["core"]) == pytest.approx(
        float(route.phase["core"]), rel=1e-12
    )


def _placed(window, origin):
    """The window moved to a station along z, its slab centred on the station."""
    nodes = np.asarray(window.mesh.nodes) + origin - [0.0, 0.0, THICKNESS / 2]
    mesh = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(window.mesh), [nodes]
    )
    return _window_model(mesh)


def test_the_first_order_matches_a_station_re_solve(tmp_path_factory, window):
    # 16 um in 2 um layers, the heater in the two middle ones (z 6..10 um)
    length = 16.0
    route = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]]) * UM

    def region(layer, name):
        return name if name != "heater" or layer in (3, 4) else "clad"

    path = tmp_path_factory.mktemp("station") / "3d.msh"
    mesh = tc.extruded(
        path, SECTION, 0.6, BOTTOM, length=length, layers=8, region=region
    )

    def solve(power):
        heat = gfem.HeatSource.uniform("heater", power=power)
        return gfem.solve(_thermal(mesh), gfem.SteadyThermal(heat))

    def phase(rise):
        return gfem.route_phase(
            window, rise, routes={"core": route}, port="p", frequency=FREQUENCY, up=UP
        )

    s = phase(None).arc_length["core"]
    j = int(np.argmin(np.abs(s - length * UM / 2)))  # z = 8 um, mid-heater
    per_length = phase(solve(1e-3).rise).phase_per_length["core"]
    # (0.34 measured)
    assert float(per_length[0]) < 0.8 * float(per_length[j])  # not z-invariant
    # the drive whose middle-station phase per length turns pi over PI_LENGTH
    driven = solve(1e-3 * np.pi / (float(per_length[j]) * PI_LENGTH))
    first = phase(driven.rise).phase_per_length["core"][j]
    k0 = 2 * np.pi / WAVELENGTH
    np.testing.assert_allclose(float(first) * PI_LENGTH, np.pi, rtol=1e-10)

    placed = _placed(window, route[0] + [0.0, 0.0, s[j]])
    cold = gfem.solve(placed, gfem.PortMode("p", FREQUENCY))
    hot = gfem.solve(
        placed.with_temperature_rise(driven.rise), gfem.PortMode("p", FREQUENCY)
    )
    resolved = float(np.real(hot.n_eff[0] - cold.n_eff[0]))
    # the second-order term: 1.0e-4 (1.03e-4 measured)
    assert float(first) / k0 == pytest.approx(resolved, rel=1e-3)
