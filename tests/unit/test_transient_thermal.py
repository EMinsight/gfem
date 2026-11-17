"""`gfem.TransientThermal`: transient heat conduction on a fixed time grid.

The problem is linear with time-invariant operators, so the stepping is
linear in the waveform: an on/off run equals the difference of two
shifted step responses, and doubling the power doubles the rise, to
round-off (ADRs 0164, 0172). The trace helpers are gsim's, checked on
gsim's own test signal (ADR 0162), and warn on an unsettled trace
(ADR 0181).
"""

import dataclasses
import math
import warnings

import jax
import numpy as np
import pytest
import thermal_cases as tc

import gfem

UM = 1e-6

# --- the trace helpers -------------------------------------------------------


def _gsim_signal():
    """gsim's helper test: 3 (1 - exp(-t / 5)) on 0..50 (tau 5, settled)."""
    t = np.linspace(0.0, 50.0, 5001)
    return t, 3.0 * (1.0 - np.exp(-t / 5.0))


def test_the_trace_helpers_match_gsims_test():
    t, y = _gsim_signal()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # gsim's gate 1e-3 (4.99961 and 10.98410 measured)
        assert gfem.time_constant(t, y) == pytest.approx(5.0, rel=1e-3)
        assert gfem.rise_time(t, y) == pytest.approx(5.0 * math.log(9.0), rel=1e-3)


def test_a_falling_trace_gives_the_same_times():
    t, y = _gsim_signal()
    assert gfem.time_constant(t, 10.0 - y) == pytest.approx(
        gfem.time_constant(t, y), rel=1e-12
    )
    assert gfem.rise_time(t, 10.0 - y) == pytest.approx(gfem.rise_time(t, y), rel=1e-12)


def test_an_unsettled_trace_warns():
    t = np.linspace(0.0, 15.0, 301)  # 3 time constants
    y = 1.0 - np.exp(-t / 5.0)
    with pytest.warns(gfem.UnsettledTraceWarning, match="1 %"):
        gfem.rise_time(t, y)
    with pytest.warns(gfem.UnsettledTraceWarning):
        gfem.time_constant(t, y)


def test_fewer_than_three_samples_after_the_last_switch_warn():
    t, y = _gsim_signal()
    with pytest.warns(gfem.UnsettledTraceWarning, match="three samples"):
        gfem.time_constant(t, y, switch_times=(t[-3],))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gfem.time_constant(t, y, switch_times=(0.0,))


def test_the_trace_helpers_refuse_a_gradient():
    t, y = _gsim_signal()
    with pytest.raises(gfem.NonDifferentiableError, match="rise_time"):
        jax.grad(lambda a: gfem.rise_time(t, a * y))(1.0)
    with pytest.raises(gfem.NonDifferentiableError, match="time_constant"):
        jax.grad(lambda a: gfem.time_constant(t, a * y))(1.0)


# --- the solver on the gsim slab ---------------------------------------------

#: gsim's 1D slab (T/test_solver.py:230-327): depth L, sink at the bottom,
#: insulated top, uniform heat q; tau1 = 4 rho c_p L^2 / (pi^2 k).
L, K, RHO, CP, Q = 10.0, 1.4, 1600.0, 1000.0, 1e11
TAU1 = 4 * RHO * CP * (L * UM) ** 2 / (math.pi**2 * K)  # 46.318 us
T_SINK = 300.0
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    edges = [
        (lambda p: abs(p[1]) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - L) < 1e-9, "top"),
    ]
    return tc.section(path, [(0.0, 0.0, 1.0, L, "slab", None)], 2.5, edges)


def _model(mesh, boundaries=None):
    material = gfem.Material(thermal_conductivity=K, mass_density=RHO, specific_heat=CP)
    if boundaries is None:
        boundaries = [gfem.FixedTemperature("bottom", temperature=T_SINK)]
    return gfem.Model(mesh, materials={"slab": material}, boundaries=boundaries)


def _top(mesh):
    return (0.5 * UM, L * UM, mesh.section_thickness / 2)


def _heat(scale=1.0):
    return gfem.HeatSource.uniform("slab", power_density=scale * Q)


def _run(mesh, waveform=((0.0, 1.0),), heat=None, options=OPTIONS, **kwargs):
    kwargs.setdefault("duration", 2 * TAU1)
    kwargs.setdefault("steps", 40)
    traces = {"top": gfem.Trace.point(_top(mesh)), "mean": gfem.Trace.mean("slab")}
    problem = gfem.TransientThermal(
        _heat() if heat is None else heat, waveform=waveform, traces=traces, **kwargs
    )
    return gfem.solve(_model(mesh), problem, options=options)


@pytest.mark.parametrize(
    "options",
    [OPTIONS, gfem.SolverOptions(order=2, time_scheme="theta", theta=0.5)],
    ids=["tr_bdf2", "theta"],
)
def test_on_then_off_is_the_difference_of_two_steps(slab, options):
    dt = 2 * TAU1 / 40
    off = 15 * dt
    on_off = _run(slab, [(0.0, 1.0), (off, 0.0)], options=options)
    step = _run(slab, options=options)
    late = _run(slab, [(off, 1.0)], options=options)
    for name in ("top", "mean"):
        expected = (step.traces[name] - T_SINK) - (late.traces[name] - T_SINK)
        swing = float(np.max(np.abs(expected)))
        # linear stepping on one operator: round-off of the absolute 300 K
        # traces (4.7e-13 of the swing measured, both schemes)
        np.testing.assert_allclose(
            on_off.traces[name] - T_SINK, expected, rtol=0, atol=1e-12 * swing
        )
    end = 2 * TAU1
    expected = np.asarray(step.rise(end).coefficients) - np.asarray(
        late.rise(end).coefficients
    )
    np.testing.assert_allclose(
        on_off.rise(end).coefficients,
        expected,
        rtol=0,
        atol=1e-12 * np.abs(expected).max(),
    )


def test_doubling_the_power_doubles_the_rise(slab):
    one = _run(slab)
    two = _run(slab, heat=[_heat(), _heat()])
    # round-off (2.8e-13 measured)
    for name in ("top", "mean"):
        rise = one.traces[name] - T_SINK
        np.testing.assert_allclose(
            two.traces[name] - T_SINK, 2 * rise, rtol=0, atol=1e-12 * rise.max()
        )
    assert float(two.power) == pytest.approx(2 * float(one.power), rel=1e-12)


def test_traces_have_steps_plus_one_samples(slab):
    result = _run(slab, steps=40)
    assert result.time.shape == (41,)
    assert float(result.time[-1]) == pytest.approx(2 * TAU1, rel=1e-12)
    assert set(result.traces) == {"top", "mean"}
    for trace in result.traces.values():
        assert trace.shape == (41,)
        # the run starts from the source-free temperature
        assert float(trace[0]) == pytest.approx(T_SINK, rel=1e-12)
    assert result.kept_times == (pytest.approx(2 * TAU1, rel=1e-12),)
    assert result.switch_times == (0.0,)


def test_the_kept_field_agrees_with_the_traces(slab):
    dt = 2 * TAU1 / 40
    result = _run(slab, keep_fields=[10 * dt])
    assert len(result.kept_times) == 2
    for k, t in ((10, 10 * dt), (40, 2 * TAU1)):
        field = result.temperature(t)
        assert field.quantity == "T"
        assert float(field.at(_top(slab))) == pytest.approx(
            float(result.traces["top"][k]), rel=1e-12
        )
        assert float(gfem.region_mean(field, "slab")) == pytest.approx(
            float(result.traces["mean"][k]), rel=1e-12
        )
        rise = result.rise(t)
        assert rise.quantity == "dT"
        assert float(rise.at(_top(slab))) + T_SINK == pytest.approx(
            float(field.at(_top(slab))), rel=1e-12
        )
    with pytest.raises(ValueError, match="keeps the temperature at"):
        result.temperature(5 * dt)


def test_keep_fields_true_keeps_every_step(slab):
    result = _run(slab, steps=4, keep_fields=True)
    assert len(result.kept_times) == 5
    assert float(result.rise(0.0).at(_top(slab))) == 0.0


def test_an_off_grid_switch_raises_naming_the_grid_times():
    dt = 1e-6
    with pytest.raises(ValueError, match=r"nearest grid times are 3e-06 s and 4e-06 s"):
        gfem.TransientThermal(
            _heat(), duration=10e-6, steps=10, waveform=[(0.0, 1.0), (3.5e-6, 0.0)]
        )
    with pytest.raises(ValueError, match=r"nearest grid times are 7e-06 s and 8e-06 s"):
        gfem.TransientThermal(_heat(), duration=10 * dt, steps=10, keep_fields=[7.2e-6])
    with pytest.raises(ValueError, match="outside the run"):
        gfem.TransientThermal(_heat(), duration=10 * dt, steps=10, keep_fields=[11e-6])


def test_theta_outside_its_range_raises():
    with pytest.raises(ValueError, match=r"theta must be a number from 0\.5"):
        gfem.SolverOptions(time_scheme="theta", theta=0.4)
    with pytest.raises(ValueError, match="time_scheme must be one of"):
        gfem.SolverOptions(time_scheme="bdf2")
    # read by the theta scheme only (ADR 0098), but checked always
    assert gfem.SolverOptions(theta=0.5).time_scheme == "tr_bdf2"


def test_fixed_problem_values_refuse_a_gradient():
    with pytest.raises(gfem.NonDifferentiableError, match="duration"):
        jax.grad(
            lambda d: gfem.TransientThermal(_heat(), duration=d, steps=4).duration
        )(1e-6)


def test_an_unsettled_slab_warns_and_a_settled_one_does_not(slab):
    # tail estimate 5.4 % at 3 tau1 and 4.7e-5 at 10 tau1 (measured)
    short = _run(slab, duration=3 * TAU1, steps=60)
    with pytest.warns(gfem.UnsettledTraceWarning):
        gfem.time_constant(
            short.time, short.traces["top"], switch_times=short.switch_times
        )
    long = _run(slab, duration=10 * TAU1, steps=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gfem.rise_time(long.time, long.traces["top"], switch_times=long.switch_times)
        gfem.time_constant(
            long.time, long.traces["top"], switch_times=long.switch_times
        )


def test_an_insulated_body_heats_at_its_power_over_its_capacity(slab):
    # no heat sink: the energy balance of the scheme is exact
    model = _model(slab, boundaries=[])
    steps, duration = 8, 0.3 * TAU1
    problem = gfem.TransientThermal(
        _heat(),
        duration=duration,
        steps=steps,
        initial_temperature=T_SINK,
        traces={"mean": gfem.Trace.mean("slab")},
    )
    result = gfem.solve(model, problem, options=OPTIONS)
    expected = T_SINK + Q * np.asarray(result.time) / (RHO * CP)
    np.testing.assert_allclose(result.traces["mean"], expected, rtol=1e-12)
    with pytest.raises(ValueError, match="initial_temperature="):
        gfem.solve(
            model,
            gfem.TransientThermal(_heat(), duration=duration, steps=steps),
            options=OPTIONS,
        )


def test_a_uniform_start_starts_there(slab):
    result = _run(slab, initial_temperature=320.0, steps=10)
    # the top point is away from the sink, so every DOF there starts at 320 K
    assert float(result.traces["top"][0]) == pytest.approx(320.0, rel=1e-12)


def test_a_missing_heat_capacity_is_listed(slab):
    model = gfem.Model(
        slab,
        materials={"slab": gfem.Material(thermal_conductivity=K)},
        boundaries=[gfem.FixedTemperature("bottom", temperature=T_SINK)],
    )
    problem = gfem.TransientThermal(_heat(), duration=1e-6, steps=2)
    with pytest.raises(gfem.MissingMaterialPropertyError) as err:
        gfem.solve(model, problem, options=OPTIONS)
    assert set(err.value.missing) == {
        ("slab", "mass_density"),
        ("slab", "specific_heat"),
    }


def test_a_sinkless_part_keeps_its_uniform_start(tmp_path):
    # piece "a" between a fixed bottom and a convective top (unequal values);
    # piece "b", apart, touches no sink and starts at 310 K
    edges = [
        (lambda p: abs(p[1]) < 1e-9 and p[0] < 1.5, "bottom"),
        (lambda p: abs(p[1] - L) < 1e-9 and p[0] < 1.5, "top"),
    ]
    rects = [(0.0, 0.0, 1.0, L, "a", None), (2.0, 0.0, 3.0, L, "b", None)]
    mesh = tc.section(tmp_path / "two.msh", rects, 2.5, edges)
    material = gfem.Material(thermal_conductivity=K, mass_density=RHO, specific_heat=CP)
    boundaries = [
        gfem.FixedTemperature("bottom", temperature=T_SINK),
        gfem.Convection("top", coefficient=1e5, ambient=290.0),
    ]
    model = gfem.Model(
        mesh, materials={"a": material, "b": material}, boundaries=boundaries
    )
    top = _top(mesh)
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("a", power_density=Q),
        duration=TAU1,
        steps=10,
        initial_temperature=310.0,
        traces={"b": gfem.Trace.mean("b"), "top": gfem.Trace.point(top)},
    )
    result = gfem.solve(model, problem, options=OPTIONS)
    np.testing.assert_allclose(result.traces["b"], 310.0, rtol=1e-12)
    assert float(result.traces["top"][0]) == pytest.approx(310.0, rel=1e-12)
    # the rise is relative to the source-free temperature of piece "a"
    sinked = gfem.Model(
        mesh,
        materials={"a": material, "b": material},
        boundaries=boundaries,
        excluded={"thermal": ("b",)},
    )
    free = gfem.solve(sinked, gfem.SteadyThermal(), options=OPTIONS)
    end = result.kept_times[-1]
    assert float(result.temperature(end).at(top)) - float(
        result.rise(end).at(top)
    ) == pytest.approx(float(free.temperature.at(top)), rel=1e-12)


# --- traces beyond points and means ------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Mixed(gfem.Trace):
    """A trace of both fields: twice the mean rise plus half the top temperature."""

    point: tuple
    unit: str = "K"

    def _sample(self, temperature, rise):
        return 2.0 * gfem.region_mean(rise, "slab") + 0.5 * temperature.at(self.point)


def test_a_trace_of_both_fields_is_recorded_through_its_sample(slab):
    traces = {"mixed": _Mixed(_top(slab))}
    problem = gfem.TransientThermal(
        _heat(), duration=2 * TAU1, steps=8, traces=traces, keep_fields=True
    )
    result = gfem.solve(_model(slab), problem, options=OPTIONS)
    for k, t in enumerate(result.kept_times):
        expected = traces["mixed"]._sample(result.temperature(t), result.rise(t))
        assert float(result.traces["mixed"][k]) == pytest.approx(
            float(expected), rel=1e-12
        )


def test_a_thermo_optic_trace_is_the_shift_of_the_rise(tmp_path):
    # the section of tests/unit/test_thermo_optic.py, with heat capacities
    rects = [
        (-1.75, -1.0, 1.75, 1.8, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
        (-0.5, 1.5, 0.5, 1.65, "heater", 0.1),
    ]
    mesh = tc.section(
        tmp_path / "section.msh",
        rects,
        0.3,
        [(lambda p: abs(p[1] + 1.0) < 1e-9, "bottom")],
    )
    heat = {"mass_density": 2200.0, "specific_heat": 700.0}
    materials = {
        "clad": gfem.Material(
            eps_r=1.444**2,
            thermal_conductivity=1.38,
            thermo_optic_coefficient=1e-5,
            **heat,
        ),
        "core": gfem.Material(
            eps_r=3.4777**2,
            thermal_conductivity=90.0,
            thermo_optic_coefficient=1.86e-4,
            **heat,
        ),
        "heater": gfem.Material(thermal_conductivity=28.0, **heat),
    }
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        ports=[gfem.WavePort("p", "front")],
        excluded={"em": ["heater"]},
    )
    frequency = 299792458.0 / 1.55e-6
    shift = gfem.thermo_optic_shift(model, None, port="p", frequency=frequency)
    problem = gfem.TransientThermal(
        gfem.HeatSource.uniform("heater", power_per_length=1.0),
        duration=5e-6,
        steps=5,
        traces={"phase": shift.trace()},
    )
    result = gfem.solve(model, problem, options=OPTIONS)
    end = result.kept_times[-1]
    expected = gfem.thermo_optic_shift(
        model, result.rise(end), port="p", frequency=frequency
    ).phase_per_length
    assert float(result.traces["phase"][-1]) == pytest.approx(
        float(expected), rel=1e-12
    )
    assert float(result.traces["phase"][0]) == 0.0
