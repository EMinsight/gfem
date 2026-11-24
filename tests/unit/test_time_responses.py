r"""`NetworkParameters.impulse_response` / `step_response` and `gfem.TimeResponse`.

Expected values come from closed forms. The main case is a lossless 75 ohm
line of one-way delay tau behind a 50 ohm port, ended by a 30 ohm load:

.. math::

    S_{11}(f) = \frac{\Gamma_1 + \Gamma_L e^{-4\pi i f \tau}}
        {1 + \Gamma_1 \Gamma_L e^{-4\pi i f \tau}},

with :math:`\Gamma_1 = (75 - 50)/(75 + 50) = 0.2` and
:math:`\Gamma_L = (30 - 75)/(30 + 75)`. The reflected step is
:math:`\Gamma_1` on :math:`(0, 2\tau)`, so the impedance profile reads 75 ohm
there; on :math:`(2\tau, 4\tau)` it reads 32.55 ohm (the multiple reflection
is not peeled) and late it settles on the load's 30 ohm. The impulse peaks at
:math:`\Gamma_1` at 0 and at :math:`(1 - \Gamma_1^2)\Gamma_L = -0.4114` at
:math:`2\tau`.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
import scipy.special

import gfem

Z0, Z1, RL, TAU = 50.0, 75.0, 30.0, 0.25e-9
G1 = (Z1 - Z0) / (Z1 + Z0)
GL = (RL - Z1) / (RL + Z1)
DF, F_MAX = 50e6, 40e9
N = round(F_MAX / DF)


def _harmonic(n=N, df=DF):
    return df * np.arange(1, n + 1)


def _s11(f):
    e = np.exp(-2j * math.pi * f * 2 * TAU)
    return (G1 + GL * e) / (1 + G1 * GL * e)


def _net(values, frequency, z_ref=Z0, ports=("p1",), excited=None):
    """A network from S ``(n_freq, n_ports, n_excited)`` (or ``(n_freq,)``)."""
    values = np.asarray(values, complex)
    if values.ndim == 1:
        values = values[:, None, None]
    return gfem.NetworkParameters(
        values=jnp.asarray(values),
        z_ref=jnp.asarray(np.broadcast_to(z_ref, (len(frequency), len(ports)))),
        frequency=tuple(float(f) for f in frequency),
        ports=ports,
        excited=ports if excited is None else excited,
    )


def _line(frequency=None, **kwargs):
    f = _harmonic() if frequency is None else np.asarray(frequency)
    return _net(_s11(f), f, **kwargs)


def _at(response, t, trace):
    return float(np.interp(t, response.time, np.asarray(trace)))


def test_the_impedance_profile_of_a_mismatched_line_reads_its_segments():
    step = _line().step_response()
    z = step.impedance("p1")
    assert abs(_at(step, TAU, z) - 75.0) <= 0.05
    assert abs(_at(step, 3 * TAU, z) - 32.55) <= 0.05
    assert abs(_at(step, 9e-9, z) - 30.0) <= 0.02


def test_the_impulse_peaks_at_each_reflection_at_its_round_trip_time():
    impulse = _line().impulse_response()
    h = np.asarray(impulse["p1", "p1"])
    t = impulse.time
    dt = t[1] - t[0]
    first = np.argmax(h)
    second = np.argmin(h)
    assert abs(h[first] - G1) <= 1e-3
    assert abs(t[first]) <= dt
    assert abs(h[second] - (1 - G1**2) * GL) <= 1e-3
    assert abs(t[second] - 2 * TAU) <= dt


def test_conjugated_data_put_the_reflection_at_negative_time():
    """Data from an e^{-iωt} code, not conjugated back, land before the edge."""
    f = _harmonic()
    impulse = _net(np.conj(_s11(f)), f).impulse_response()
    h = np.asarray(impulse["p1", "p1"])
    dt = impulse.time[1] - impulse.time[0]
    assert abs(impulse.time[np.argmin(h)] + 2 * TAU) <= dt


def _delay_line(frequency=None, delay=1e-9):
    """A matched lossless 2-port line: S21 = S12 = exp(-iωτ), S11 = S22 = 0."""
    f = _harmonic() if frequency is None else np.asarray(frequency)
    e = np.exp(-2j * math.pi * f * delay)
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 0, 1] = s[:, 1, 0] = e
    return _net(s, f, ports=("p1", "p2"))


def _ideal_low_pass_rise():
    r"""10-90 % rise of the unwindowed step :math:`1/2 + \mathrm{Si}(x)/\pi`, x span."""
    x = scipy.optimize.brentq(
        lambda x: scipy.special.sici(x)[0] / math.pi - 0.4, 0.1, 3.0
    )
    return x / math.pi


@pytest.mark.parametrize(
    ("beta", "expected"),
    [(6.0, 0.999), (0.0, _ideal_low_pass_rise()), (13.0, 1.437)],
)
# the tail ripples (5e-4 of the step at beta = 0), far too little to move the edge
@pytest.mark.filterwarnings("ignore::gfem.UnsettledTraceWarning")
def test_the_rise_time_of_a_step_is_set_by_the_span_and_the_window(beta, expected):
    """The network-analyzer low-pass table: 0.99, 0.45 and 1.47 over the span.

    Without a window the step is the ideal low-pass one, whose rise is 0.4458
    over the span.
    """
    step = _delay_line().step_response(kaiser_beta=beta)
    rise = gfem.rise_time(step.time, np.asarray(step["p2", "p1"]))
    assert rise * F_MAX == pytest.approx(expected, rel=0.02)


def test_padding_only_interpolates_the_impulse():
    net = _line()
    default = net.impulse_response()
    padded = net.impulse_response(n_time=4 * len(default.time))
    peak = np.max(np.asarray(default["p1", "p1"]))
    assert np.max(np.asarray(padded["p1", "p1"])) == pytest.approx(peak, abs=1e-3)


def test_dc_extrapolated_from_three_points_matches_the_exact_dc():
    exact = _line(DF * np.arange(0, N + 1)).step_response()
    extrapolated = _line().step_response()
    difference = np.asarray(extrapolated.values) - np.asarray(exact.values)
    assert np.max(np.abs(difference)) <= 1e-4


def test_a_zero_frequency_sample_is_the_dc_value_and_the_step_ends_there():
    f = np.r_[_harmonic(), 0.0]  # in any order
    s = _s11(f)
    s[-1] = 0.1
    step = _net(s, f).step_response()
    assert float(step["p1", "p1"][-1]) == pytest.approx(0.1, abs=1e-12)


@pytest.mark.parametrize(
    "frequency",
    [DF * np.arange(1, 101), np.linspace(DF, 100 * DF, 100)],
    ids=["arange", "linspace"],
)
def test_a_harmonic_grid_is_accepted(frequency):
    assert _line(frequency).step_response().values.shape[0] == 2048


@pytest.mark.parametrize(
    "frequency",
    [
        DF * np.arange(2, 102),
        np.delete(DF * np.arange(1, 101), 50),
        np.geomspace(DF, 100 * DF, 100),
        DF * np.arange(1, 3),
    ],
    ids=["starts-at-2df", "missing-point", "log", "two-points"],
)
def test_a_grid_that_is_not_harmonic_is_refused_with_the_remedies(frequency):
    net = _line(frequency)
    for call in (net.step_response, net.impulse_response):
        with pytest.raises(ValueError, match="harmonic") as caught:
            call()
        assert "np.arange(1, n + 1)" in str(caught.value)
        assert "fit_rational().network(" in str(caught.value)


def test_the_time_axis_is_the_two_sided_record():
    step = _line().step_response()
    assert isinstance(step.time, tuple)
    t = np.asarray(step.time)
    assert t.size == 16384  # the smallest power of two >= 20 N = 16000
    assert np.allclose(np.diff(t), 1 / (t.size * DF), rtol=1e-12, atol=0)
    assert t[0] == pytest.approx(-1 / (2 * DF), rel=1e-12)
    assert np.all(np.diff(t) > 0)


@pytest.mark.parametrize("n_time", [2 * N + 1, 2 * N - 2, 2.0 * N, True])
def test_an_odd_or_short_record_is_refused(n_time):
    with pytest.raises(ValueError, match="n_time"):
        _line().step_response(n_time=n_time)


def test_the_shortest_record_is_twice_the_frequencies():
    assert len(_line().step_response(n_time=2 * N).time) == 2 * N


@pytest.mark.parametrize("beta", [-1.0, math.nan, math.inf])
def test_a_negative_or_non_finite_kaiser_beta_is_refused(beta):
    with pytest.raises(ValueError, match="kaiser_beta"):
        _line().impulse_response(kaiser_beta=beta)


C0 = 299_792_458.0
FC = 10e9  # TE10 cutoff of the guide
GUIDE = 0.02  # length, m


def _guide(frequency):
    """A matched empty guide above cutoff: S21 = exp(-iβL), β = 2π√(f² - fc²)/c."""
    f = np.asarray(frequency)
    e = np.exp(-2j * math.pi * np.sqrt(f**2 - FC**2) / C0 * GUIDE)
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 0, 1] = s[:, 1, 0] = e
    return _net(s, f, ports=("p1", "p2"))


def test_a_band_pass_impulse_peaks_at_the_group_delay():
    f = np.linspace(1.25 * FC, 1.9 * FC, 300)
    impulse = _guide(f).impulse_response(mode="band_pass")
    assert impulse.mode == "band_pass"
    assert impulse.kind == "impulse"
    t = impulse.time
    dt = t[1] - t[0]
    centre = (f[0] + f[-1]) / 2
    group_delay = GUIDE / (C0 * math.sqrt(1 - (FC / centre) ** 2))
    h21 = np.asarray(impulse["p2", "p1"])
    assert abs(t[np.argmax(h21)] - group_delay) <= 2 * dt
    assert np.max(np.abs(np.asarray(impulse["p1", "p1"]))) <= 1e-3


def test_a_band_pass_impulse_of_a_constant_reflection_peaks_at_its_magnitude():
    f = np.linspace(1.25 * FC, 1.9 * FC, 300)
    impulse = _net(np.full(f.size, -0.3 + 0.4j), f).impulse_response(mode="band_pass")
    assert np.max(np.asarray(impulse["p1", "p1"])) == pytest.approx(0.5, rel=1e-12)


def test_a_band_pass_impulse_refuses_an_uneven_grid():
    f = np.geomspace(1.25 * FC, 1.9 * FC, 300)
    with pytest.raises(ValueError, match="evenly spaced"):
        _guide(f).impulse_response(mode="band_pass")


def test_a_step_has_no_band_pass_mode():
    f = np.linspace(1.25 * FC, 1.9 * FC, 300)
    with pytest.raises(TypeError):
        _guide(f).step_response(mode="band_pass")  # ty: ignore[unknown-argument]


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="band_pass"):
        _line().impulse_response(mode="high_pass")  # ty: ignore[invalid-argument-type]


def test_z0_renormalizes_and_the_impedance_profile_does_not_change():
    on_50 = _line().step_response()
    on_75 = _line().step_response(z0=75.0)
    assert np.asarray(on_75.z_ref).tolist() == [75.0]
    # on its own impedance the line's first segment reflects nothing
    assert abs(_at(on_75, TAU, on_75["p1", "p1"])) <= 1e-3
    # the line and the load read the same; the second segment does not, since
    # on 75 ohm no multiple reflection mixes in and it reads the load's 30 ohm
    assert _at(on_75, 3 * TAU, on_75.impedance("p1")) == pytest.approx(30.0, abs=0.05)
    for t in (TAU, 9e-9):
        z50 = _at(on_50, t, on_50.impedance("p1"))
        assert _at(on_75, t, on_75.impedance("p1")) == pytest.approx(z50, abs=0.1)


def test_ports_may_have_different_real_references():
    f = _harmonic(100)
    net = _net(np.zeros((f.size, 2, 2)), f, z_ref=[50.0, 75.0], ports=("a", "b"))
    assert np.asarray(net.step_response().z_ref).tolist() == [50.0, 75.0]


def test_a_modal_network_responds_on_unit_modal_power_without_an_impedance():
    f = _harmonic(100)
    step = _net(_s11(f), f, z_ref=np.nan).step_response()
    assert np.isnan(np.asarray(step.z_ref)).all()
    assert np.isfinite(np.asarray(step.values)).all()
    with pytest.raises(ValueError, match="no reference impedance"):
        step.impedance("p1")
    with pytest.raises(ValueError, match="voltage_path"):
        _net(_s11(f), f, z_ref=np.nan).step_response(z0=50.0)


@pytest.mark.parametrize(
    "z_ref",
    [
        np.full(100, 48.0 + 3.0j),
        np.linspace(45.0, 55.0, 100),
    ],
    ids=["complex", "frequency-dependent"],
)
def test_a_reference_that_is_not_one_real_value_is_refused(z_ref):
    f = _harmonic(100)
    net = _net(_s11(f), f, z_ref=z_ref[:, None])
    with pytest.raises(ValueError, match=r"z0=50\.0"):
        net.step_response()
    assert np.asarray(net.step_response(z0=50.0).z_ref).tolist() == [50.0]


def _near(z, n, spread=0.0, imag=0.0):
    """``z`` ohm over ``n`` frequencies, off by ``spread`` and ``imag`` relative."""
    ramp = np.linspace(-0.5, 0.5, n)
    return z * (1 + spread * ramp) + 1j * z * imag * ramp


@pytest.mark.parametrize(
    ("spread", "imag"), [(1e-13, 0.0), (0.0, 1e-13)], ids=["spread", "imaginary"]
)
def test_a_reference_constant_to_round_off_is_its_mean_real_value(spread, imag):
    # a TEM wave port's Z_PV varies across frequency by about 1e-13 relative
    f = _harmonic(100)
    z_ref = np.stack([_near(50.0, f.size, spread, imag), _near(75.0, f.size)], -1)
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 0, 0] = _s11(f)
    net = _net(s, f, z_ref=z_ref, ports=("a", "b"))
    step = net.step_response()
    np.testing.assert_allclose(np.asarray(step.z_ref), [50.0, 75.0], rtol=1e-15)
    exact = _line(f).step_response()
    np.testing.assert_allclose(
        np.asarray(step["a", "a"]), np.asarray(exact["p1", "p1"]), atol=1e-12
    )
    impulse = net.impulse_response()
    np.testing.assert_allclose(np.asarray(impulse.z_ref), [50.0, 75.0], rtol=1e-15)


@pytest.mark.parametrize(
    ("spread", "imag"), [(1e-6, 0.0), (0.0, 1e-6)], ids=["spread", "imaginary"]
)
def test_a_reference_off_by_more_than_round_off_is_refused(spread, imag):
    f = _harmonic(100)
    net = _net(_s11(f), f, z_ref=_near(50.0, f.size, spread, imag)[:, None])
    with pytest.raises(ValueError, match=r"z0=50\.0"):
        net.step_response()
    with pytest.raises(ValueError, match=r"z0=50\.0"):
        net.impulse_response()


def test_a_reference_missing_on_some_ports_is_refused():
    f = _harmonic(100)
    s = np.zeros((f.size, 2, 2), complex)
    s[:, 1, :] = np.nan
    net = _net(s, f, z_ref=[50.0, np.nan], ports=("a", "b"))
    with pytest.raises(ValueError, match="'b' has no reference impedance"):
        net.impulse_response()


@pytest.mark.parametrize("z0", [0.0, -50.0, 50.0 + 1j, math.nan, [50.0, 50.0]])
def test_a_bad_z0_is_refused(z0):
    with pytest.raises(ValueError, match="z0"):
        _line().step_response(z0=z0)


def test_z0_needs_every_port_excited():
    net = _delay_line(_harmonic(100))
    partial = dataclasses.replace(net, values=net.values[:, :, :1], excited=("p1",))
    with pytest.raises(ValueError, match="excite"):
        partial.step_response(z0=75.0)


def test_a_response_is_read_by_port_names():
    step = _delay_line(_harmonic(100)).step_response()
    assert isinstance(step, gfem.TimeResponse)
    assert step.kind == "step"
    assert step.mode == "low_pass"
    assert step.values.shape == (2048, 2, 2)
    assert np.array_equal(step["p2", "p1"], step.values[:, 1, 0])
    with pytest.raises(KeyError, match="'p3'"):
        step["p3", "p1"]
    with pytest.raises(KeyError, match="'p3'"):
        step["p1", "p3"]
    assert repr(step).startswith("TimeResponse(step, low_pass, ports=['p1', 'p2']")


def test_a_partially_excited_network_has_its_excited_columns():
    net = _delay_line(_harmonic(100))
    partial = dataclasses.replace(net, values=net.values[:, :, :1], excited=("p1",))
    step = partial.step_response()
    assert step.excited == ("p1",)
    assert step.values.shape == (2048, 2, 1)
    assert np.array_equal(step["p2", "p1"], net.step_response()["p2", "p1"])
    with pytest.raises(KeyError, match="not excited"):
        step["p1", "p2"]
    with pytest.raises(ValueError, match="not excited"):
        step.impedance("p2")


def test_the_impedance_profile_needs_a_step():
    with pytest.raises(ValueError, match="step response"):
        _line(_harmonic(100)).impulse_response().impedance("p1")
    with pytest.raises(KeyError, match="'p9'"):
        _line(_harmonic(100)).step_response().impedance("p9")


def test_a_response_exports_to_xarray():
    pytest.importorskip("xarray")
    step = _delay_line(_harmonic(100)).step_response()
    ds = step.to_xarray()
    assert ds["response"].dims == ("time", "port", "excited")
    assert ds["z_ref"].dims == ("port",)
    assert ds["z_ref"].attrs["units"] == "ohm"
    assert ds["time"].attrs["units"] == "s"
    assert ds.attrs == {"kind": "step", "mode": "low_pass"}
    assert list(ds["excited"].values) == ["p1", "p2"]
    assert np.array_equal(ds["response"].values, np.asarray(step.values))


def test_a_traced_response_does_not_export_to_xarray():
    pytest.importorskip("xarray")
    net = _line(_harmonic(100))
    with pytest.raises(TypeError, match="to_xarray"):
        jax.jit(lambda n: n.step_response().to_xarray())(net)


def test_the_step_differentiates_as_the_linear_map_it_is():
    """S -> step is linear, so d/dp of S0 + p S1 is the response to S1."""
    net = _line(_harmonic(100))
    other = _delay_line(_harmonic(100))
    s1 = other.values[:, :1, :1]
    weights = np.cos(np.arange(2048) / 37.0)[:, None, None]

    def objective(p):
        moved = dataclasses.replace(net, values=net.values + p * s1)
        return jnp.sum(weights * moved.step_response().values)

    expected = objective(1.0) - objective(0.0)
    assert jax.grad(objective)(0.3) == pytest.approx(expected, abs=1e-10)


def test_the_impulse_differentiates_with_respect_to_z0():
    net = _line(_harmonic(100))

    def objective(z0):
        return net.impulse_response(z0=z0)["p1", "p1"][1100]

    h = 1e-3
    central = (objective(60.0 + h) - objective(60.0 - h)) / (2 * h)
    assert jax.grad(objective)(60.0) == pytest.approx(float(central), rel=1e-6)


def test_jit_gives_the_same_values():
    net = _line(_harmonic(100))
    eager = net.step_response(z0=60.0)
    traced = jax.jit(lambda n: n.step_response(z0=60.0))(net)
    assert np.allclose(traced.values, eager.values, rtol=0, atol=1e-15)
    assert np.array_equal(traced.time, eager.time)
    assert np.allclose(
        jax.jit(lambda n: n.impulse_response(mode="band_pass").values)(net),
        net.impulse_response(mode="band_pass").values,
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("argument", ["kaiser_beta", "n_time"])
def test_a_traced_window_or_record_is_not_differentiable(argument):
    net = _line(_harmonic(100))
    value = {"kaiser_beta": 6.0, "n_time": 2048}[argument]
    with pytest.raises(gfem.NonDifferentiableError, match=argument):
        jax.jit(lambda v: net.step_response(**{argument: v}).values)(value)
    with pytest.raises(gfem.NonDifferentiableError, match="kaiser_beta"):
        jax.grad(lambda b: net.impulse_response(kaiser_beta=b).values.sum())(6.0)


def test_a_jitted_function_of_a_response_takes_responses_on_other_grids():
    # the time axis is a static label, as a network's frequencies are
    profile = jax.jit(lambda step: step.impedance("p1"))
    for frequency in (_harmonic(100), _harmonic(100, df=2 * DF), _harmonic(50)):
        step = _line(frequency).step_response()
        np.testing.assert_allclose(
            profile(step), step.impedance("p1"), rtol=1e-12, atol=0
        )
