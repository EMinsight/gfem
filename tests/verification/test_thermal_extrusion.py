"""A z-invariant 3D extrusion against the one-element slab.

A heater cross-section shaped like femwell's TiN heater (a silicon wafer
under a 2 um oxide box, a 500 x 220 nm silicon core, a 2 um x 140 nm TiN
heater 2 um above it, oxide cladding) is solved twice: as a cross-section
slab, and extruded along z over 4 um in 4 prism layers (each prism split
into tetrahedra) with insulated ends. Both carry the heater's DC terminals
on their end faces and heat by Joule heating, so the 3D path from DC
conduction to temperature runs on the slab's physics.

- The resistance per length to round-off (the 1e-3 thermal gate, tightened: the
  potential is linear along z, so both spaces hold it; 6e-14 measured).
- The core's mean rise per W/m within 1e-3 (the split prisms do not hold a
  z-invariant field exactly; 2.4e-5 measured, 4e-5 spread
  measured).
"""

import pytest
import thermal_cases as tc

import gfem

UM = 1e-6
HALF = 6.0  # um, half width of the section
LENGTH, LAYERS = 4.0, 4  # um, the extrusion
SECTION = [
    (-HALF, -2.5, HALF, -2.0, "wafer", None),
    (-HALF, -2.0, HALF, 0.0, "box", None),
    (-HALF, 0.0, HALF, 2.8, "clad", None),
    (-0.25, 0.0, 0.25, 0.22, "core", 0.04),
    (-1.0, 2.22, 1.0, 2.36, "heater", 0.1),
]
BOTTOM = [(lambda p: abs(p[1] + 2.5) < 1e-9, "bottom")]
OXIDE = gfem.Material(thermal_conductivity=1.38)
MATERIALS = {
    "wafer": gfem.Material(thermal_conductivity=148.0),
    "box": OXIDE,
    "clad": OXIDE,
    "core": gfem.Material(thermal_conductivity=90.0),
    "heater": gfem.Material(conductivity=2.3e6, thermal_conductivity=28.0),
}
TERMINALS = gfem.DCConduction({"heater": "front"}, ground="back")


def _solve(mesh, **drive):
    model = gfem.Model(
        mesh,
        materials=MATERIALS,
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    dc = gfem.solve(model, TERMINALS)
    steady = gfem.solve(model, gfem.SteadyThermal(gfem.joule_heating(dc, **drive)))
    return dc, float(gfem.region_mean(steady.rise, "core"))


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    mesh = tc.section(path, SECTION, 0.6, BOTTOM, thickness=0.5e-6)
    return _solve(mesh, power_per_length=1.0)


@pytest.fixture(scope="module")
def extruded(tmp_path_factory):
    path = tmp_path_factory.mktemp("extruded") / "3d.msh"
    mesh = tc.extruded(
        path, SECTION, 0.6, BOTTOM, length=LENGTH, layers=LAYERS, ends=True
    )
    return _solve(mesh, power=LENGTH * UM)  # 1 W/m


def test_the_resistance_per_length_is_the_slabs(slab, extruded):
    per_length = float(extruded[0].resistance[0, 0]) / (LENGTH * UM)
    assert per_length == pytest.approx(
        float(slab[0].resistance_per_length[0, 0]), rel=1e-12
    )


def test_the_core_rise_per_watt_per_metre_is_the_slabs(slab, extruded):
    # (-2.4e-5 measured)
    assert extruded[1] == pytest.approx(slab[1], rel=1e-3)
