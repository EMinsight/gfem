"""`gfem.route_phase`: the thermo-optic phase along polylines through a 3D rise.

One optical window (a silicon core in oxide on a cross-section slab) is
swept along routes through a small 3D block heated evenly and sunk on its
face x = 0, so the rise grows with x alone: a route along y sees the same
rise at every station, and which side of the route the window's core
samples shows in the phase.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem
from gfem.thermo_optic import _stations

UM = 1e-6
WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
N_SI, N_OX = 3.4777, 1.444
DN_SI, DN_OX = 1.86e-4, 1.0e-5
#: The block: x 0..8 um (sink at x = 0, insulated elsewhere), y 0..4, z 0..3.
SIZE = (8.0, 4.0)
HEIGHT = 3.0
POWER_DENSITY = 1e12  # W/m^3
K_BLOCK = 1.38
#: The route height and the route's x: the window spans z 0.9..2.32 um.
Z0, X0 = 1.5, 4.0


def _window_mesh(path, core_x):
    rects = [
        (-1.25, -0.6, 1.25, 0.82, "clad", None),
        (core_x - 0.25, 0.0, core_x + 0.25, 0.22, "core", 0.05),
    ]
    return tc.section(path, rects, 0.3, [])


def _window(mesh, dn_core=DN_SI):
    materials = {
        "clad": gfem.Material(eps_r=N_OX**2, thermo_optic_coefficient=DN_OX),
        "core": gfem.MaterialField(eps_r=N_SI**2, thermo_optic_coefficient=dn_core),
    }
    return gfem.Model(mesh, materials=materials, ports=[gfem.WavePort("p", "front")])


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    path = tmp_path_factory.mktemp("window") / "centred.msh"
    return _window(_window_mesh(path, 0.0))


@pytest.fixture(scope="module")
def offset_window(tmp_path_factory):
    path = tmp_path_factory.mktemp("window") / "offset.msh"
    return _window(_window_mesh(path, 0.5))


@pytest.fixture(scope="module")
def steady(tmp_path_factory):
    path = tmp_path_factory.mktemp("block") / "block.msh"
    sink = [(lambda p: abs(p[0]) < 1e-9, "sink")]
    mesh = tc.brick(path, SIZE, [("block", HEIGHT)], 0.8, sink)
    model = gfem.Model(
        mesh,
        materials={"block": gfem.Material(thermal_conductivity=K_BLOCK)},
        boundaries=[gfem.FixedTemperature("sink", temperature=300.0)],
    )
    heat = gfem.HeatSource.uniform("block", power_density=POWER_DENSITY)
    return gfem.solve(model, gfem.SteadyThermal(heat))


def _rise(x_um):
    """The exact rise of the block at x (um): q (L x - x^2 / 2) / k."""
    x, length = x_um * UM, SIZE[0] * UM
    return POWER_DENSITY * (length * x - x**2 / 2) / K_BLOCK


def _line(*points):
    return np.asarray(points, float) * UM


ALONG_Y = _line((X0, 0.5, Z0), (X0, 3.5, Z0))


def _phase(window, rise, routes, **kwargs):
    return gfem.route_phase(
        window, rise, routes=routes, port="p", frequency=FREQUENCY, **kwargs
    )


def test_stations_are_even_in_arc_length_with_the_ends(window):
    # two segments, 1.0 and 0.6 um: 0.5 um spacing gives 5 stations, 0.4 um apart
    route = _line((2.0, 0.5, Z0), (2.0, 1.5, Z0), (2.6, 1.5, Z0))
    result = _phase(window, None, {"bend": route})
    np.testing.assert_allclose(
        result.arc_length["bend"], np.linspace(0, 1.6, 5) * UM, rtol=0, atol=1e-18
    )
    assert result.length["bend"] == pytest.approx(1.6 * UM, rel=1e-14)
    _, points, tangent, left, up = _stations("bend", route, 0.5 * UM, (0, 0, 1))
    expected = _line(
        (2.0, 0.5, Z0), (2.0, 0.9, Z0), (2.0, 1.3, Z0), (2.2, 1.5, Z0), (2.6, 1.5, Z0)
    )
    np.testing.assert_allclose(points, expected, rtol=0, atol=1e-18)
    np.testing.assert_allclose(tangent[0], (0, 1, 0), atol=1e-15)
    np.testing.assert_allclose(tangent[-1], (1, 0, 0), atol=1e-15)
    # left of travel: up x tangent; up stays the wafer normal
    np.testing.assert_allclose(left[0], (-1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(left[-1], (0, 1, 0), atol=1e-15)
    np.testing.assert_allclose(up, np.tile((0, 0, 1.0), (5, 1)), atol=1e-15)
    # a vertex hit by a station takes the mean of its two directions
    _, points, tangent, _, _ = _stations("v", route, 0.2 * UM, (0, 0, 1))
    np.testing.assert_allclose(points[5], _line((2.0, 1.5, Z0))[0], atol=1e-18)
    np.testing.assert_allclose(tangent[5], np.array([1, 1, 0]) / np.sqrt(2))


def test_a_segment_along_up_raises_naming_route_and_segment(window):
    route = _line((X0, 0.5, Z0), (X0, 1.5, Z0), (X0, 1.5, Z0 + 0.5))
    with pytest.raises(ValueError, match=r"route 'riser'.*segment 1 .*parallel to up"):
        _phase(window, None, {"riser": route})


def test_a_traced_polyline_raises_naming_routes(window):
    def phase(points):
        return _phase(window, None, {"r": points}).n_eff

    with pytest.raises(TypeError, match="routes"):
        jax.make_jaxpr(phase)(jnp.asarray(ALONG_Y))
    with pytest.raises(TypeError, match="up"):
        jax.make_jaxpr(lambda u: _phase(window, None, {"r": ALONG_Y}, up=u).n_eff)(
            jnp.asarray([0.0, 0.0, 1.0])
        )


def test_a_sample_outside_the_rise_raises_naming_route_and_arc_length(window, steady):
    # stations every 0.5 um: the one at 3.5 um sits on the face x = 8 um,
    # the last (4 um) is past it
    route = _line((6.0, 0.5, Z0), (6.0, 2.0, Z0), (8.5, 2.0, Z0))
    with pytest.raises(ValueError, match=r"route 'edge'.*arc length 4e-06 m"):
        _phase(window, steady.rise, {"edge": route, "inside": ALONG_Y})


def test_the_window_pins_left_of_travel_on_an_asymmetric_section(offset_window, steady):
    # the core sits at section x = +0.5 um, left of travel: along +y that is
    # global x = 3.5 um (cooler), along -y global x = 4.5 um (hotter)
    north, south = ALONG_Y, ALONG_Y[::-1]
    result = _phase(offset_window, steady.rise, {"north": north, "south": south})
    ratio = float(result.phase["north"] / result.phase["south"])
    # the core holds most of the phase: the ratio follows the rise at the
    # core's centre, 0.845 (-5.9e-5 measured)
    assert ratio == pytest.approx(_rise(X0 - 0.5) / _rise(X0 + 0.5), rel=1e-3)
    assert ratio < 0.9
    # a z-invariant rise along the route: constant phase per length
    per_length = np.asarray(result.phase_per_length["north"])
    np.testing.assert_allclose(per_length, per_length.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        result.phase["north"], per_length.mean() * result.length["north"], rtol=1e-6
    )


def test_the_trace_equals_the_forward_on_a_steady_rise(window, steady):
    routes = {"line": ALONG_Y, "bend": _line((2, 0.5, Z0), (2, 2, Z0), (5, 2.5, Z0))}
    result = _phase(window, steady.rise, routes)
    for name in routes:
        traced = result.trace(name)
        assert isinstance(traced, gfem.thermal.Trace)
        assert traced.unit == "rad"
        value = traced._sample(None, steady.rise)
        np.testing.assert_allclose(value, result.phase[name], rtol=1e-12)
    # without a rise: the same trace, located on first use
    bare = _phase(window, None, routes)
    assert bare.phase is None
    assert bare.phase_per_length is None
    np.testing.assert_allclose(bare.n_eff, result.n_eff, rtol=1e-12)
    value = bare.trace("bend")._sample(None, steady.rise)
    np.testing.assert_allclose(value, result.phase["bend"], rtol=1e-12)
    with pytest.raises(KeyError, match="routes"):
        result.trace("nope")


def test_the_phase_is_positive_and_linear_in_the_rise_and_coefficient(window, steady):
    result = _phase(window, steady.rise, {"line": ALONG_Y})
    phase = float(result.phase["line"])
    assert phase > 0
    assert N_OX < float(result.n_eff) < N_SI
    # rad per K of uniform rise: k0 sum w dn/dT over the route length
    rise = steady.rise

    def of_coefficients(c):
        field = dataclasses.replace(rise, coefficients=c)
        return _phase(window, field, {"line": ALONG_Y}).phase["line"]

    grad = jax.grad(of_coefficients)(rise.coefficients)
    # linear: the gradient dotted with the coefficients is the phase
    assert float(grad @ rise.coefficients) == pytest.approx(phase, rel=1e-12)

    mesh = window.mesh

    def of_dn(dn):
        return _phase(_window(mesh, dn), steady.rise, {"line": ALONG_Y}).phase["line"]

    core_only = of_dn(DN_SI) - float(of_dn(0.0))
    assert float(jax.grad(of_dn)(DN_SI)) * DN_SI == pytest.approx(
        float(core_only), rel=1e-12
    )


def test_a_route_trace_is_the_route_phase_of_the_transient_rise(window, steady):
    model = gfem.Model(
        steady.rise.mesh,
        materials={
            "block": gfem.Material(
                thermal_conductivity=K_BLOCK, mass_density=2200.0, specific_heat=700.0
            )
        },
        boundaries=[gfem.FixedTemperature("sink", temperature=300.0)],
    )
    routes = {"line": ALONG_Y}
    bare = _phase(window, None, routes)
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("block", power_density=POWER_DENSITY),
        duration=5e-6,
        steps=4,
        traces={"phase": bare.trace("line")},
    )
    result = gfem.solve(model, problem)
    end = result.kept_times[-1]
    expected = _phase(window, result.rise(end), routes).phase["line"]
    assert float(result.traces["phase"][-1]) == pytest.approx(
        float(expected), rel=1e-12
    )
    assert float(result.traces["phase"][0]) == 0.0
