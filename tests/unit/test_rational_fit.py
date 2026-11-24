r"""`NetworkParameters.fit_rational` and `gfem.RationalModel`.

The data are exact rational S written out here as
:math:`S(f) = D + \sum_n R_n / (2\pi i f - p_n)`, so the fit must recover
their poles, and analytic lossless lines, whose pole count grows with the
number of phase wraps in the band.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

import gfem

#: Highest data frequency of the synthetic models, Hz.
F_MAX = 10e9
W_MAX = 2 * math.pi * F_MAX


def _rational_s(f, poles, residues, d):
    """Exact S ``(n_freq, P, P)`` of poles (rad/s) and residues on ``f`` (Hz)."""
    s = 2j * math.pi * np.asarray(f, float)
    terms = residues[None] / (s[:, None] - poles[None, :])[:, :, None, None]
    return d[None] + terms.sum(axis=1)


def _net(f, s, z_ref=50.0, ports=None, excited=None):
    s = np.asarray(s, complex)
    n_ports = s.shape[1]
    ports = tuple(f"p{i + 1}" for i in range(n_ports)) if ports is None else ports
    return gfem.NetworkParameters(
        values=jnp.asarray(s),
        z_ref=jnp.asarray(np.broadcast_to(z_ref, (len(f), n_ports))),
        frequency=tuple(float(x) for x in f),
        ports=ports,
        excited=ports if excited is None else excited,
    )


def _pairs(real, cplx):
    """Pole array (normalized): real poles, then each pair with Im > 0 first."""
    out = list(real)
    for p in cplx:
        out += [p, np.conj(p)]
    return np.asarray(out, complex)


def _sigma_max(f, poles, residues, d):
    s = _rational_s(f, poles, residues, d)
    return np.linalg.svd(s, compute_uv=False)[:, 0].max()


def _three_port(sigma=0.95, symmetric=True):
    """2 real poles and 6 lightly damped pairs, largest singular value ``sigma``.

    The residues and ``d`` are symmetric, or the same draws unsymmetrized.
    """

    def sym(m):
        return m + m.T if symmetric else 2 * m

    rng = np.random.default_rng(0)
    real = np.array([-0.05, -0.7])
    wr = np.array([0.15, 0.3, 0.45, 0.6, 0.8, 0.95])
    cplx = -wr * np.array([0.02, 0.05, 0.01, 0.03, 0.02, 0.04]) + 1j * wr
    poles = _pairs(real, cplx) * W_MAX
    residues = np.zeros((len(poles), 3, 3), complex)
    for k in range(len(real)):
        residues[k] = sym(rng.standard_normal((3, 3)))
    for k in range(len(cplx)):
        m = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        i = len(real) + 2 * k
        residues[i] = sym(m)
        residues[i + 1] = np.conj(sym(m))
    residues *= W_MAX / 10
    d = rng.standard_normal((3, 3))
    d = 0.05 * sym(d)
    scale = sigma / _sigma_max(np.linspace(0, 3, 20001) * F_MAX, poles, residues, d)
    return poles, residues * scale, d * scale


def _fit(net, **kwargs):
    return net.fit_rational(enforce_passivity=False, **kwargs)


def _relative_pole_error(found, true):
    found = np.sort_complex(np.asarray(found))
    true = np.sort_complex(true)
    assert found.shape == true.shape
    return np.max(np.abs(found - true) / np.abs(true))


def test_an_exact_three_port_rational_s_is_recovered_to_round_off():
    poles, residues, d = _three_port()
    f = np.linspace(0.1, 1.0, 200) * F_MAX
    net = _net(f, _rational_s(f, poles, residues, d))

    model = _fit(net, tolerance=1e-10)

    assert model.error <= 1e-12
    assert model.fit_error == model.error
    assert model.n_poles == 14
    assert _relative_pole_error(model.poles, poles) <= 1e-10
    assert model.ports == ("p1", "p2", "p3")
    assert np.array_equal(model.z_ref, [50.0, 50.0, 50.0])
    assert model.frequency_range == (f[0], f[-1])
    assert model.passivity_violations == ()
    assert model.nominal_reference is False


def _two_port():
    """1 real pole and 2 pairs, 2 ports."""
    rng = np.random.default_rng(1)
    poles = _pairs([-0.2], [-0.01 + 0.4j, -0.02 + 0.7j]) * W_MAX
    residues = np.zeros((5, 2, 2), complex)
    residues[0] = rng.standard_normal((2, 2))
    for k in (1, 3):
        m = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        residues[k] = m
        residues[k + 1] = np.conj(m)
    d = 0.1 * rng.standard_normal((2, 2))
    return poles, residues * W_MAX / 10, d


def test_twenty_points_give_the_exact_order_and_no_far_pole():
    poles, residues, d = _two_port()
    f = np.linspace(0.1, 1.0, 20) * F_MAX
    net = _net(f, _rational_s(f, poles, residues, d))

    model = _fit(net, tolerance=1e-10)

    # 4 poles fall short, 8 fit; the 3 spare ones are pruned, none left far out
    assert model.error <= 1e-11
    assert model.n_poles == 5
    assert np.abs(model.poles).max() <= 10 * W_MAX
    assert _relative_pole_error(model.poles, poles) <= 1e-8


def _sigma(f, poles, residues, d):
    """Largest singular value of the exact S at each of ``f`` (Hz)."""
    s = _rational_s(np.atleast_1d(f), poles, residues, d)
    return np.linalg.svd(s, compute_uv=False)[:, 0]


def _swept_violations(poles, residues, d, f_top=3 * F_MAX, n=60001):
    """Intervals (Hz) where the largest singular value exceeds 1.

    A dense sweep brackets each edge, which is then refined by root finding
    on the largest singular value; the sweep must end below 1.
    """
    f = np.linspace(0.0, f_top, n)
    over = _sigma(f, poles, residues, d) > 1
    assert not over[-1]

    def edge(i):
        return scipy.optimize.brentq(
            lambda x: _sigma(x, poles, residues, d)[0] - 1, f[i], f[i + 1], xtol=1e-3
        )

    changes = np.flatnonzero(over[1:] != over[:-1])
    edges = ([0.0] if over[0] else []) + [edge(i) for i in changes]
    return list(zip(edges[0::2], edges[1::2], strict=True))


#: 200 data points on 0.1-1 of the highest frequency, no DC.
F200 = np.linspace(0.1, 1.0, 200) * F_MAX


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_violations_of_a_fit_are_where_a_dense_sweep_finds_them(symmetric):
    poles, residues, d = _three_port(sigma=1.02, symmetric=symmetric)
    net = _net(F200, _rational_s(F200, poles, residues, d))

    model = _fit(net, tolerance=1e-10)

    expected = _swept_violations(model.poles, model.residues, model.d)
    assert expected
    assert len(model.passivity_violations) == len(expected)
    for got, want in zip(model.passivity_violations, expected, strict=True):
        assert np.allclose(got, want, rtol=1e-6, atol=0)
    assert model.passive is False


def _noisy(s, level, seed=2):
    """``s`` plus symmetric complex noise of RMS ``level`` per element."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)
    noise = (noise + noise.transpose(0, 2, 1)) / 2
    return s + level * noise / np.sqrt(np.mean(np.abs(noise) ** 2))


@pytest.mark.filterwarnings("ignore::gfem.NonPassiveDataWarning")
def test_enforcement_makes_a_slightly_active_noisy_fit_passive_and_keeps_it_close():
    poles, residues, d = _three_port(sigma=1.02)
    truth = _rational_s(F200, poles, residues, d)
    net = _net(F200, _noisy(truth, 1e-4))

    model = net.fit_rational()

    assert model.passivity_violations
    assert model.passive
    dense = np.linspace(0.0, 3.0, 60001) * F_MAX
    s_dense = _rational_s(dense, model.poles, model.residues, model.d)
    assert np.linalg.svd(s_dense, compute_uv=False)[:, 0].max() <= 1
    # RMS over every element and frequency: the 2 % excess sits on a sharp
    # resonance, so the element it is largest on moves more
    fitted = np.asarray(model.network(F200).values)
    assert np.sqrt(np.mean(np.abs(fitted - truth) ** 2)) <= 1e-3
    assert np.abs(s_dense - s_dense.transpose(0, 2, 1)).max() <= 1e-12
    assert np.linalg.svd(model.d, compute_uv=False).max() <= 1 - 1e-4 + 1e-12


def test_enforcement_that_moves_s_beyond_the_tolerance_warns_non_passive_data():
    poles, residues, d = _three_port(sigma=1.15)
    net = _net(F200, _rational_s(F200, poles, residues, d))

    with pytest.warns(gfem.NonPassiveDataWarning) as record:
        model = net.fit_rational()

    message = str(record[0].message)
    assert "largest singular value of 1.15 at 4.5" in message
    assert record[0].filename == __file__
    assert "with_offsets" in message
    assert model.fit_error <= 1e-3 < model.error
    assert model.passive


@pytest.mark.filterwarnings("ignore::gfem.NonPassiveDataWarning")
def test_a_constant_term_above_one_is_clipped_below_one_minus_the_margin():
    poles, residues, d = _two_port()
    rotation = np.array([[0.6, 0.8], [-0.8, 0.6]])
    d = 1.01 * rotation
    net = _net(F20, _rational_s(F20, poles, residues / 100, d))

    model = net.fit_rational(tolerance=1e-10)

    assert model.passivity_violations[-1][1] == math.inf
    assert np.linalg.svd(model.d, compute_uv=False).max() <= 1 - 1e-4 + 1e-12
    assert model.passive


def test_a_lossless_through_is_a_model_without_poles_clipped_passive():
    s = np.zeros((len(F20), 2, 2), complex)
    s[:, 0, 1] = s[:, 1, 0] = 1.0

    model = _net(F20, s).fit_rational()

    assert model.n_poles == 0
    assert model.passive
    np.testing.assert_allclose(model.error, 1e-4, rtol=1e-6)


def _line(f, delay, z_line, z_port=50.0):
    """S of a lossless line of impedance ``z_line`` and delay between ports."""
    theta = 2 * math.pi * np.asarray(f) * delay
    a = np.cos(theta)
    b = 1j * z_line * np.sin(theta)
    c = 1j * np.sin(theta) / z_line
    den = a + b / z_port + c * z_port + a
    s = np.empty((len(f), 2, 2), complex)
    s[:, 0, 0] = s[:, 1, 1] = (b / z_port - c * z_port) / den
    s[:, 0, 1] = s[:, 1, 0] = 2 / den
    return s


#: Harmonic grid of 400 points to 40 GHz.
LINE_F = 40e9 / 400 * np.arange(1, 401)


def test_a_matched_line_of_ten_phase_wraps_takes_about_two_poles_per_wrap(recwarn):
    net = _net(LINE_F, _line(LINE_F, 10 / LINE_F[-1], 50.0))

    model = _fit(net)

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    assert model.error <= 1e-3
    assert model.n_poles <= 32


# A lossless line is unitary over the band: its fit hovers at sigma = 1 and
# its D near a through, where the Hamiltonian's eigenvalues lose crossings.
@pytest.mark.parametrize("wraps", [8, 10])
def test_the_passive_fit_of_a_lossless_line_stays_below_one_between_samples(wraps):
    f = 3e12 / 150 * np.arange(1, 151)
    net = _net(f, _line(f, wraps / f[-1], 50.0))

    model = net.fit_rational()

    assert model.passive
    dense = np.linspace(0.0, f[-1], 20001)
    sigma = np.linalg.svd(np.asarray(model.network(dense).values), compute_uv=False)
    assert sigma.max() <= 1.0


def test_a_mismatched_line_of_thirty_wraps_warns_that_the_fit_did_not_converge():
    net = _net(LINE_F, _line(LINE_F, 30 / LINE_F[-1], 75.0))

    with pytest.warns(gfem.RationalFitNotConvergedWarning) as record:
        model = _fit(net)

    message = str(record[0].message)
    assert "max_poles=100" in message
    assert "0.1-40 GHz" in message
    assert "with_offsets" in message
    assert model.fit_error > 1e-3
    assert model.n_poles <= 100


#: A small exact 2-port network on 20 frequencies, for the refusals.
F20 = np.linspace(0.1, 1.0, 20) * F_MAX


def _small(**kwargs):
    poles, residues, d = _two_port()
    return _net(F20, _rational_s(F20, poles, residues, d), **kwargs)


@pytest.mark.parametrize("tolerance", [0.0, -1e-3, math.nan, math.inf])
def test_a_tolerance_that_is_not_positive_and_finite_is_refused(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        _fit(_small(), tolerance=tolerance)


@pytest.mark.parametrize("max_poles", [3, 20, 2.5, True])
def test_max_poles_outside_four_to_n_freq_minus_one_is_refused(max_poles):
    with pytest.raises(ValueError, match=r"max_poles.*\(19\)"):
        _fit(_small(), max_poles=max_poles)


def test_max_poles_at_its_bounds_is_accepted():
    assert _fit(_small(), max_poles=19).n_poles == 5
    with pytest.warns(gfem.RationalFitNotConvergedWarning):
        assert _fit(_small(), max_poles=4).n_poles <= 4


@pytest.mark.parametrize(
    "weights", [np.ones(19), np.ones((20, 1)), -np.ones(20), np.zeros(20)]
)
def test_weights_of_the_wrong_shape_or_sign_are_refused(weights):
    with pytest.raises(ValueError, match="weights"):
        _fit(_small(), weights=weights)


def test_weights_leave_an_exact_fit_exact_and_the_error_unweighted():
    model = _fit(_small(), tolerance=1e-10, weights=np.linspace(1, 100, 20))
    assert model.error <= 1e-11
    assert model.n_poles == 5


def test_fewer_than_six_frequencies_are_refused():
    s = np.zeros((5, 1, 1))
    with pytest.raises(ValueError, match="at least 6 frequencies"):
        _fit(_net(F20[:5], s))


@pytest.mark.parametrize(
    ("f", "match"),
    [
        (np.r_[F20[:6], F20[5]], "distinct"),
        (np.r_[-1e9, F20[:6]], ">= 0 Hz"),
    ],
)
def test_duplicate_or_negative_frequencies_are_refused(f, match):
    with pytest.raises(ValueError, match=match):
        _fit(_net(f, np.full((f.size, 1, 1), 0.1)))


def test_frequencies_in_any_order_and_a_dc_sample_are_fitted():
    poles, residues, d = _two_port()
    f = np.r_[0.0, F20][::-1]
    model = _fit(_net(f, _rational_s(f, poles, residues, d)), tolerance=1e-10)
    assert model.frequency_range == (0.0, F20[-1])
    assert model.error <= 1e-11
    assert model.n_poles == 5


def test_s_that_is_not_finite_is_refused():
    poles, residues, d = _two_port()
    s = _rational_s(F20, poles, residues, d)
    s[3, 0, 1] = np.inf
    with pytest.raises(ValueError, match="finite S"):
        _fit(_net(F20, s))


def test_s_that_is_not_finite_is_named_before_a_missing_reference():
    poles, residues, d = _two_port()
    s = _rational_s(F20, poles, residues, d)
    s[3, 0, 1] = np.nan
    with pytest.raises(ValueError, match="finite S"):
        _fit(_net(F20, s, z_ref=np.nan), z0=50.0)


def test_a_partially_excited_network_is_refused():
    poles, residues, d = _two_port()
    s = _rational_s(F20, poles, residues, d)[:, :, :1]
    with pytest.raises(ValueError, match="full scattering matrix"):
        _fit(_net(F20, s, excited=("p1",)))


@pytest.mark.parametrize(
    "z_ref",
    [
        np.array([50.0, 50.0 + 1j]),
        np.linspace(49, 51, 20)[:, None] * np.ones(2),
    ],
)
def test_a_complex_or_frequency_dependent_reference_needs_z0(z_ref):
    with pytest.raises(ValueError, match=r"z0=50\.0"):
        _fit(_small(z_ref=z_ref))


def _near_50(spread=0.0, imag=0.0):
    """A 50 ohm ``z_ref`` ``(20, 2)`` off by ``spread`` and ``imag`` relative."""
    ramp = np.linspace(-0.5, 0.5, len(F20))[:, None] * np.array([1.0, -1.0])
    return 50.0 * (1 + spread * ramp) + 50j * imag * ramp


@pytest.mark.parametrize(
    ("spread", "imag"), [(1e-13, 0.0), (0.0, 1e-13)], ids=["spread", "imaginary"]
)
def test_a_reference_constant_to_round_off_is_its_mean_real_value(spread, imag):
    # a TEM wave port's Z_PV varies across frequency by about 1e-13 relative
    model = _fit(_small(z_ref=_near_50(spread, imag)))
    np.testing.assert_allclose(model.z_ref, [50.0, 50.0], rtol=1e-15)


@pytest.mark.parametrize(
    ("spread", "imag"), [(1e-6, 0.0), (0.0, 1e-6)], ids=["spread", "imaginary"]
)
def test_a_reference_off_by_more_than_round_off_needs_z0(spread, imag):
    with pytest.raises(ValueError, match=r"z0=50\.0"):
        _fit(_small(z_ref=_near_50(spread, imag)))


def test_z0_renormalizes_s_before_the_fit():
    # a 75 ohm line, one phase wrap, on a complex reference at port 2
    on_50 = _net(LINE_F, _line(LINE_F, 1 / LINE_F[-1], 75.0))
    net = on_50.renormalize([50.0, 60.0 + 5j])

    model = _fit(net, z0=[50.0, 75.0], tolerance=1e-6)

    expected = np.asarray(on_50.renormalize([50.0, 75.0]).values)
    got = np.asarray(model.network(LINE_F).values)
    assert np.array_equal(model.z_ref, [50.0, 75.0])
    assert model.error <= 1e-6
    assert np.abs(got - expected).max() <= 1e-5
    assert model.nominal_reference is False


def test_ports_with_different_real_references_are_fitted_as_they_are():
    model = _fit(_small(z_ref=np.array([50.0, 30.0])))
    assert np.array_equal(model.z_ref, [50.0, 30.0])


@pytest.mark.parametrize("z0", [0.0, -50.0, 50.0 + 1j, [50.0], math.nan, True])
def test_a_z0_that_is_not_one_positive_real_value_or_one_per_port_is_refused(z0):
    with pytest.raises(ValueError, match="z0"):
        _fit(_small(), z0=z0)


def test_a_network_without_reference_impedances_needs_z0_as_a_label():
    net = _small(z_ref=np.nan)
    with pytest.raises(ValueError, match="z0="):
        _fit(net)

    model = _fit(net, z0=50.0, tolerance=1e-10)

    assert model.nominal_reference is True
    assert np.array_equal(model.z_ref, [50.0, 50.0])
    # S is not renormalized: the model reproduces the data as given
    assert np.abs(np.asarray(model.network(F20).values - net.values)).max() <= 1e-11
    assert np.isnan(np.asarray(model.network(F20).z_ref)).all()


def test_a_network_with_some_ports_without_reference_is_refused():
    with pytest.raises(ValueError, match="'p2' has no reference impedance"):
        _fit(_small(z_ref=np.array([50.0, np.nan])), z0=50.0)


def test_enforcement_that_cannot_finish_names_the_intervals_left(monkeypatch):
    import gfem._broadband._passivity as passivity

    monkeypatch.setattr(passivity, "_ITERATIONS", 0)
    poles, residues, d = _three_port(sigma=1.02)
    net = _net(F200, _rational_s(F200, poles, residues, d))

    with pytest.raises(gfem.PassivityEnforcementError) as info:
        net.fit_rational()

    message = str(info.value)
    assert "4.4911-4.50921 GHz" in message
    assert "enforce_passivity=False" in message
    assert issubclass(gfem.PassivityEnforcementError, RuntimeError)


def _traced_fit(which, x):
    """Fit with ``x`` (traced) as S, z_ref, z0 or weights; a scalar out."""
    net = _small()
    if which in ("values", "z_ref"):
        net = gfem.NetworkParameters(
            values=net.values * x if which == "values" else net.values,
            z_ref=net.z_ref * x if which == "z_ref" else net.z_ref,
            frequency=net.frequency,
            ports=net.ports,
            excited=net.excited,
        )
    kwargs = {"z0": 50.0 * x} if which == "z0" else {}
    if which == "weights":
        kwargs = {"weights": jnp.ones(20) * x}
    model = _fit(net, **kwargs)
    return jnp.sum(jnp.abs(jnp.asarray(model.poles)))


@pytest.mark.parametrize("which", ["values", "z_ref", "z0", "weights"])
@pytest.mark.parametrize("transform", [jax.grad, jax.jit])
def test_a_traced_input_to_the_fit_raises_non_differentiable(which, transform):
    with pytest.raises(gfem.NonDifferentiableError, match="stop_gradient"):
        transform(lambda x: _traced_fit(which, x))(1.0)


@pytest.mark.parametrize(
    ("option", "value"),
    [("tolerance", 1e-3), ("max_poles", 8), ("enforce_passivity", True)],
)
def test_a_traced_option_of_the_fit_raises_non_differentiable(option, value):
    net = _small()
    with pytest.raises(gfem.NonDifferentiableError, match=f"fit_rational {option}"):
        jax.jit(lambda v: jnp.asarray(net.fit_rational(**{option: v}).error))(value)


def test_a_fit_inside_jit_raises_non_differentiable_even_on_a_concrete_network():
    net = _small()
    with pytest.raises(gfem.NonDifferentiableError, match="stop_gradient"):
        jax.jit(lambda t: jnp.asarray(_fit(net).error) * t)(1.0)


def _model(**changes):
    """A hand-built 2-port model: 1 real pole and 1 pair, on 50 and 30 ohm."""
    p = 2 * math.pi * 1e9
    fields = {
        "poles": np.array([-0.3 * p, -0.05 * p + 2j * p, -0.05 * p - 2j * p]),
        "residues": np.array(
            [
                0.1 * p * np.array([[1.0, 0.5], [0.5, -1.0]]),
                p * np.array([[0.1 + 0.2j, 0.05j], [0.05j, -0.1 + 0.1j]]),
                p * np.array([[0.1 - 0.2j, -0.05j], [-0.05j, -0.1 - 0.1j]]),
            ]
        ),
        "d": np.array([[0.2, 0.0], [0.0, -0.3]]),
        "z_ref": np.array([50.0, 30.0]),
        "ports": ("a", "b"),
        "frequency_range": (1e8, 5e9),
    }
    fields.update(changes)
    return gfem.RationalModel(**fields)


def test_a_model_built_by_hand_keeps_the_default_diagnostics():
    model = _model()
    assert model.n_poles == 3
    assert math.isnan(model.error)
    assert math.isnan(model.fit_error)
    assert model.passivity_violations == ()
    assert model.nominal_reference is False
    assert model.d.dtype == float
    assert model.poles.dtype == complex
    assert not model.poles.flags.writeable


def test_the_dc_limit_is_d_minus_the_residues_over_the_poles():
    model = _model()
    expected = model.d - np.sum(model.residues / model.poles[:, None, None], axis=0)

    net = model.network([0.0])

    assert np.abs(np.asarray(net.values[0]) - expected).max() <= 1e-14
    assert net.ports == net.excited == ("a", "b")
    assert np.array_equal(np.asarray(net.z_ref), [[50.0, 30.0]])
    assert net.frequency == (0.0,)


def test_the_network_is_the_pole_residue_sum():
    model = _model()
    f = np.array([3e9, 1e8, 2.5e9])
    expected = _rational_s(f, model.poles, model.residues, model.d)
    assert np.abs(np.asarray(model.network(f).values) - expected).max() <= 1e-14


def test_the_network_refuses_frequencies_above_the_fitted_range():
    with pytest.raises(ValueError, match=r"above the fitted range 0\.1-5 GHz"):
        _model().network([1e9, 5.1e9])
    with pytest.raises(ValueError, match=">= 0 Hz"):
        _model().network([-1.0])
    # no range: no check
    assert _model(frequency_range=None).network([1e12]).values.shape == (1, 2, 2)


def test_a_traced_frequency_raises_non_differentiable():
    model = _model()
    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(lambda f: jnp.abs(model.network(f).values).sum())(1e9)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"residues": np.zeros((2, 2, 2))}, "residues must have shape"),
        ({"d": np.zeros((3, 3))}, "d must have shape"),
        ({"z_ref": np.array([50.0])}, "z_ref"),
        ({"poles": np.array([-1.0, 1j - 0.1, -1j - 0.1]) * 0}, "negative real"),
        ({"poles": np.array([1.0, -1 + 1j, -1 - 1j])}, "negative real"),
        ({"poles": np.array([-1.0, -1 - 1j, -1 + 1j])}, "conjugate"),
        ({"poles": np.array([-1.0, -1 + 1j, -2 - 1j])}, "conjugate"),
        ({"z_ref": np.array([50.0, 0.0])}, "z_ref"),
        ({"z_ref": np.array([50.0, -30.0])}, "z_ref"),
        ({"d": np.array([[0.2j, 0.0], [0.0, 0.0]])}, "real"),
        ({"ports": ("a", "a")}, "distinct"),
    ],
)
def test_a_model_that_is_not_stable_real_or_consistent_is_refused(changes, match):
    with pytest.raises(ValueError, match=match):
        _model(**changes)


def test_unpaired_residues_are_refused():
    residues = _model().residues.copy()
    residues[2] = residues[1]
    with pytest.raises(ValueError, match="conjugate"):
        _model(residues=residues)
    residues = _model().residues.copy()
    residues[0, 0, 0] = 1j
    with pytest.raises(ValueError, match="real pole"):
        _model(residues=residues)


def test_the_repr_names_ports_poles_range_error_and_passivity():
    model = _fit(_small(), tolerance=1e-10)
    text = repr(model)
    assert text.startswith("RationalModel(ports=['p1', 'p2'], n_poles=5, ")
    assert "1-10 GHz" in text
    assert f"error={model.error:.3g}" in text
    # the 2-port's singular values reach 11.8: fitted as it is, it is active
    assert text.endswith(", passive=False)")


def test_a_hand_built_model_reports_whether_it_is_passive():
    # sigma_max peaks at 4.98 near 2 GHz
    assert not _model().passive
    quiet = _model().residues / 20
    assert _model(residues=quiet).passive
    assert not _model(residues=quiet, d=np.array([[1.2, 0.0], [0.0, -0.3]])).passive
    # a short circuit, S = -I: singular values exactly 1 are passive
    short = _model(residues=np.zeros((3, 2, 2)), d=-np.eye(2))
    assert short.passive
    assert _model(poles=[], residues=np.zeros((0, 2, 2)), d=-np.eye(2)).passive


def test_the_fit_docstring_states_the_numbers_the_fit_uses():
    from gfem._broadband import _passivity, _vector_fit

    doc = " ".join(gfem.NetworkParameters.fit_rational.__doc__.split())
    margin = round(math.log10(_passivity.MARGIN))
    assert 10.0**margin == _passivity.MARGIN
    assert f"steps of {_vector_fit._POLE_STEP} poles" in doc
    assert f"``tolerance``/{round(1 / _vector_fit._PRUNE)}" in doc
    assert f":math:`1 - 10^{{{margin}}}`" in doc
    assert f"{_passivity._ITERATIONS} iterations" in doc
    assert f"min({_vector_fit._MAX_POLES}, n_freq - 1)" in doc
    assert f"fewer than {_vector_fit._MIN_FREQUENCIES} distinct" in doc
