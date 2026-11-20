"""`gfem.quantum.dressed_modes`, mesh free.

Inputs follow pyEPR's tutorial convention where a test reproduces one of its
stored outputs: ``E_J / h`` in GHz and the zero-point flux ``phi`` of each
mode, turned into what gfem takes (``omega`` in rad/s, the signed junction
participation ``p = 2 E_J phi^2 / (hbar omega)`` and ``L_J = phi_0^2 / E_J``).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem import quantum

TWO_PI = 2 * math.pi
PHI_0 = const.hbar / (2 * const.e)  # reduced flux quantum


def _inputs(f_ghz, phi, ej_ghz):
    """``(omega, participation, inductance)`` from pyEPR's ``f``, ``phi``, ``E_J``."""
    f = np.asarray(f_ghz, float) * 1e9
    phi = np.asarray(phi, float)
    ej = const.h * ej_ghz * 1e9
    p = np.sign(phi) * 2 * ej * phi**2 / (const.h * f)
    return TWO_PI * f, p, PHI_0**2 / ej


def _mhz(x):
    return np.asarray(x) / TWO_PI / 1e6


def _ghz(x):
    return np.asarray(x) / TWO_PI / 1e9


# --- pyEPR Tutorial 6 --------------------------------------------------------


def _one_transmon():
    ej, ec = 20.0, 0.3
    return _inputs([math.sqrt(8 * ej * ec)], [(2 * ec / ej) ** 0.25], ej)


def _two_modes():
    return _inputs([5.5, 7.0], [0.2, 0.02], 20.0)


def test_one_transmon_reproduces_pyepr_tutorial_6():
    omega, p, lj = _one_transmon()
    dressed = quantum.dressed_modes(omega, p, inductance=lj, levels=15)
    assert _ghz(dressed.omega[0]) == pytest.approx(6.6135, rel=1e-3)
    assert -_mhz(dressed.anharmonicity[0]) == pytest.approx(336.9, rel=1e-3)


def test_two_modes_reproduce_pyepr_tutorial_6():
    omega, p, lj = _two_modes()
    dressed = quantum.dressed_modes(omega, p, inductance=lj)
    assert -_mhz(dressed.anharmonicity[0]) == pytest.approx(15.89, rel=3e-3)
    assert -_mhz(dressed.dispersive_shift[0, 1]) == pytest.approx(0.308, rel=3e-3)
    assert _ghz(dressed.omega[0]) == pytest.approx(5.4839, rel=3e-3)


def test_regression_pin_of_the_probe_digits():
    # A pin of a direct diagonalization (exact cosine), not evidence.
    omega, p, lj = _one_transmon()
    one = quantum.dressed_modes(omega, p, inductance=lj, levels=15)
    assert _ghz(one.omega[0]) == pytest.approx(6.61345, abs=5e-6)
    assert _mhz(one.anharmonicity[0]) == pytest.approx(-336.876, abs=5e-4)
    omega, p, lj = _two_modes()
    two = quantum.dressed_modes(omega, p, inductance=lj)
    assert _mhz(two.anharmonicity[0]) == pytest.approx(-15.885, abs=5e-4)
    assert _mhz(two.dispersive_shift[0, 1]) == pytest.approx(-0.3075, abs=5e-5)
    assert _ghz(two.omega[0]) == pytest.approx(5.48391, abs=5e-6)


# --- first order -------------------------------------------------------------


def test_first_order_is_the_closed_form():
    omega = np.array([2 * np.pi * 5.9e9, 2 * np.pi * 7.1e9, 2 * np.pi * 9.3e9])
    p = np.array([0.9, -0.02, 0.003])
    lj = 9e-9
    ej = PHI_0**2 / lj / const.hbar
    out = quantum.dressed_modes(omega, p, inductance=lj, method="first_order")
    for m in range(3):
        k_mm = omega[m] ** 2 * p[m] ** 2 / (4 * ej)
        assert out.anharmonicity[m] == pytest.approx(-k_mm / 2, rel=1e-12)
        assert out.dispersive_shift[m, m] == pytest.approx(-k_mm / 2, rel=1e-12)
        lamb = -sum(omega[m] * omega[n] * abs(p[m] * p[n]) for n in range(3)) / (8 * ej)
        assert out.lamb_shift[m] == pytest.approx(lamb, rel=1e-12)
        assert out.omega[m] == pytest.approx(omega[m] + lamb, rel=1e-12)
        for n in range(3):
            if n != m:
                k = omega[m] * omega[n] * abs(p[m] * p[n]) / (4 * ej)
                assert out.dispersive_shift[m, n] == pytest.approx(-k, rel=1e-12)
    assert out.overlap is None


def test_first_order_anharmonicity_of_one_mode_is_minus_e_c():
    lj, c = 10e-9, 80e-15
    omega = 1 / math.sqrt(lj * c)
    out = quantum.dressed_modes([omega], [1.0], inductance=lj, method="first_order")
    e_c = const.e**2 / (2 * c)
    assert out.anharmonicity[0] == pytest.approx(-e_c / const.hbar, rel=1e-12)


def test_diagonalization_tends_to_first_order_as_the_flux_vanishes():
    omega, p, lj = _two_modes()

    def gap(s, read):
        full = quantum.dressed_modes(omega, p * s**2, inductance=lj)
        first = quantum.dressed_modes(
            omega, p * s**2, inductance=lj, method="first_order"
        )
        return abs(float(read(full) / read(first)) - 1)

    # The relative gap is O(phi^2): it falls by 4 per halving of the flux scale s,
    # down to where round-off takes over (s of about 1/32 here).
    chi = lambda d: d.dispersive_shift[0, 1]  # noqa: E731
    alpha = lambda d: d.anharmonicity[0]  # noqa: E731
    for read, scales in ((chi, (0.25, 0.125)), (alpha, (0.125, 0.0625))):
        for s in scales:
            assert gap(s, read) / gap(s / 2, read) == pytest.approx(4, rel=0.1)


# --- a transmon and a resonator, exactly solved ------------------------------

EJ_GHZ = 20.0
L_J = PHI_0**2 / (const.h * EJ_GHZ * 1e9)
C_Q = const.e**2 / (2 * const.h * 0.25e9)  # E_C / h = 250 MHz
C_G = 5e-15


def _circuit(lj, f_r, c_q=C_Q):
    """Linear modes of a junction LC capacitively coupled to a 50 ohm LC resonator.

    Returns ``omega`` (rad/s, ascending) and the signed junction participation:
    the junction's share of each mode's inductive energy, signed by its flux.
    """
    c_r = 1 / (TWO_PI * f_r * 50.0)
    l_r = 50.0 / (TWO_PI * f_r)
    c = jnp.array([[c_q + C_G, -C_G], [-C_G, c_r + C_G]])
    k = jnp.diag(jnp.stack([1 / lj, jnp.asarray(1 / l_r)]))
    lc = jnp.linalg.cholesky(c)
    a = jnp.linalg.solve(lc, jnp.linalg.solve(lc, k).T)
    w2, y = jnp.linalg.eigh((a + a.T) / 2)
    v = jnp.linalg.solve(lc.T, y)
    p = jnp.sign(v[0]) * (v[0] ** 2 / lj) / jnp.einsum("im,ij,jm->m", v, k, v)
    return jnp.sqrt(w2), p


def test_the_circuit_participation_is_the_inductance_sensitivity():
    _, p = _circuit(L_J, 7.0e9)
    sens = jax.jit(jax.jacrev(lambda x: jnp.log(_circuit(jnp.exp(x), 7.0e9)[0])))(
        jnp.log(L_J)
    )
    np.testing.assert_allclose(np.abs(p), -2 * np.asarray(sens), rtol=1e-10)
    assert float(jnp.sum(jnp.abs(p))) == pytest.approx(1.0, abs=1e-10)


def test_twelve_levels_are_converged():
    omega, p = _circuit(L_J, 7.0e9)
    a = quantum.dressed_modes(omega, p, inductance=L_J, levels=12)
    b = quantum.dressed_modes(omega, p, inductance=L_J, levels=16)
    np.testing.assert_allclose(a.anharmonicity, b.anharmonicity, rtol=1e-4)
    assert float(a.dispersive_shift[0, 1]) == pytest.approx(
        float(b.dispersive_shift[0, 1]), rel=1e-5
    )


def test_a_transmon_below_its_resonator_has_negative_shifts():
    omega, p = _circuit(L_J, 7.0e9)
    d = quantum.dressed_modes(omega, p, inductance=L_J)
    assert np.all(np.asarray(d.anharmonicity) < 0)
    assert np.all(np.asarray(d.lamb_shift) < 0)
    assert np.all(np.asarray(d.dispersive_shift) < 0)
    np.testing.assert_array_equal(d.dispersive_shift, d.dispersive_shift.T)
    np.testing.assert_array_equal(np.diagonal(d.dispersive_shift), d.anharmonicity)
    np.testing.assert_allclose(d.lamb_shift, d.omega - omega, rtol=1e-12)
    assert 0.5 < float(d.overlap) <= 1.0
    # One junction: the signs of the participations cancel.
    flipped = quantum.dressed_modes(omega, -p, inductance=L_J)
    np.testing.assert_allclose(flipped.dispersive_shift, d.dispersive_shift, rtol=1e-12)


def test_complex_omega_uses_its_real_part():
    omega, p = _circuit(L_J, 7.0e9)
    d = quantum.dressed_modes(omega, p, inductance=L_J)
    lossy = quantum.dressed_modes(omega + 1j * omega / 1e5, p, inductance=L_J)
    np.testing.assert_array_equal(lossy.dispersive_shift, d.dispersive_shift)


# --- first derivatives -------------------------------------------------------

#: Scales that make every input of order one, at the 7 GHz circuit.
_OMEGA, _P = _circuit(L_J, 7.0e9)
_X0 = np.ones(5)


def _from_x(x):
    """``(omega, participation, inductance)`` at ``x * (omega, 0.9 p, L_J)``.

    The participations are scaled down so that a difference step keeps them in
    [-1, 1].
    """
    return x[:2] * _OMEGA, x[2:4] * 0.9 * _P, x[4] * L_J


_OUTPUTS = {
    "dispersive_shift": lambda d: d.dispersive_shift[0, 1],
    "anharmonicity": lambda d: d.anharmonicity[0],
    "lamb_shift": lambda d: d.lamb_shift[1],
    "omega": lambda d: d.omega[0],
}


def _objective(read, levels=8, method="diagonalization"):
    def f(x):
        omega, p, lj = _from_x(x)
        out = quantum.dressed_modes(
            omega, p, inductance=lj, levels=levels, method=method
        )
        return read(out) / TWO_PI / 1e6  # MHz

    return f


@pytest.mark.parametrize("output", list(_OUTPUTS))
def test_gradient_matches_fourth_order_differences(output):
    # Steps: the frequencies are 6 detunings away from zero, so they take a
    # smaller step than the participations and the inductance.
    f = _objective(_OUTPUTS[output])
    grad = jax.grad(f)(jnp.asarray(_X0))
    random = unit_directions(np.random.default_rng(10), (5,), 3)
    # 3e-8: macOS's arm64 build lands at 1.4e-8 on the dispersive shift
    assert_gradient_matches(f, jnp.asarray(_X0), random, step=3e-4, rel=3e-8, grad=grad)
    axes = [np.eye(5)[i] for i in (2, 3, 4)]
    assert_gradient_matches(f, jnp.asarray(_X0), axes, step=1e-3, rel=1e-8, grad=grad)


def _chi_by_eigh(x, levels=8):
    """The dispersive shift with `jnp.linalg.eigh` differentiated directly (MHz)."""
    omega, p, lj = _from_x(x)
    ej = PHI_0**2 / lj / const.hbar
    phi = jnp.sign(p) * jnp.sqrt(jnp.abs(p) * omega / (2 * ej))
    a = np.diag(np.sqrt(np.arange(1.0, levels)), 1)
    eye = np.eye(levels)
    ops = [np.kron(a, eye), np.kron(eye, a)]
    x_op = sum(phi[m] * (ops[m] + ops[m].T) for m in range(2))
    lam, u = jnp.linalg.eigh(x_op)
    cos = (u * (jnp.cos(lam) - 1 + lam**2 / 2)) @ u.T
    h = sum(omega[m] * ops[m].T @ ops[m] for m in range(2)) - ej * cos
    w, v = jnp.linalg.eigh(h)
    level = [w[jnp.argmax(v[i] ** 2)] for i in (0, 1, levels, levels + 1)]
    return (level[3] - level[1] - level[2] + level[0]) / TWO_PI / 1e6


def test_gradient_equals_differentiating_eigh():
    got = jax.grad(_objective(_OUTPUTS["dispersive_shift"]))(jnp.asarray(_X0))
    want = jax.jit(jax.grad(_chi_by_eigh))(jnp.asarray(_X0))
    assert float(jax.jit(_chi_by_eigh)(jnp.asarray(_X0))) == pytest.approx(
        float(_objective(_OUTPUTS["dispersive_shift"])(_X0)), rel=1e-10
    )
    np.testing.assert_allclose(
        got, want, rtol=1e-7, atol=1e-7 * float(jnp.max(jnp.abs(want)))
    )


def test_forward_mode_equals_reverse_mode():
    f = _objective(_OUTPUTS["dispersive_shift"])
    np.testing.assert_allclose(
        jax.jacfwd(f)(jnp.asarray(_X0)), jax.jacrev(f)(jnp.asarray(_X0)), rtol=1e-10
    )


def test_gradient_is_finite_and_right_at_equal_zero_point_fluxes():
    # phi_m^2 is proportional to p_m omega_m: equal products give equal phi_m.
    omega = TWO_PI * np.array([6.0e9, 7.0e9])
    p = np.array([0.1 * 7 / 6, 0.1])
    lj = 10e-9

    def f(x):
        out = quantum.dressed_modes(
            x[:2] * omega, x[2:4] * p, inductance=x[4] * lj, levels=8
        )
        return out.dispersive_shift[0, 1] / TWO_PI / 1e6

    grad = jax.grad(f)(jnp.asarray(_X0))
    assert np.all(np.isfinite(grad))
    assert_gradient_matches(
        f,
        jnp.asarray(_X0),
        [np.eye(5)[i] for i in range(5)],
        step=1e-3,
        rel=1e-7,
        grad=grad,
    )


def test_jit_and_vmap_return_dressed_modes():
    run = jax.jit(quantum.dressed_modes, static_argnames=("levels", "method"))
    out = run(_OMEGA, _P, inductance=L_J, levels=8)
    assert isinstance(out, quantum.DressedModes)
    eager = quantum.dressed_modes(_OMEGA, _P, inductance=L_J, levels=8)
    np.testing.assert_allclose(out.dispersive_shift, eager.dispersive_shift, rtol=1e-9)
    batch = jax.vmap(
        lambda lj: (
            quantum.dressed_modes(_OMEGA, _P, inductance=lj, levels=8).dispersive_shift
        )
    )(jnp.array([L_J, 1.01 * L_J]))
    np.testing.assert_allclose(batch[0], eager.dispersive_shift, rtol=1e-9)


# --- second derivatives ------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        jax.hessian,
        lambda f: jax.hessian(jax.jit(f)),
        lambda f: lambda x: jax.jvp(jax.grad(f), (x,), (jnp.ones_like(x),))[1],
    ],
    ids=["hessian", "hessian(jit)", "jvp(grad)"],
)
def test_a_second_derivative_raises(transform):
    f = _objective(_OUTPUTS["dispersive_shift"])
    with pytest.raises(gfem.NonDifferentiableError, match="first order only"):
        transform(f)(jnp.asarray(_X0))


def test_first_order_has_a_finite_hessian():
    hess = jax.jit(
        jax.hessian(_objective(_OUTPUTS["dispersive_shift"], method="first_order"))
    )(jnp.asarray(_X0))
    assert np.all(np.isfinite(hess))
    assert np.any(hess != 0)


def test_a_gradient_through_overlap_raises():
    def f(x):
        omega, p, lj = _from_x(x)
        return quantum.dressed_modes(omega, p, inductance=lj, levels=8).overlap

    with pytest.raises(gfem.NonDifferentiableError, match="overlap"):
        jax.grad(f)(jnp.asarray(_X0))
    with pytest.raises(gfem.NonDifferentiableError, match="overlap"):
        jax.jit(jax.grad(f))(jnp.asarray(_X0))
    # Forward mode cannot tell a read tangent from a discarded one: it is NaN.
    assert np.all(np.isnan(jax.jacfwd(f)(jnp.asarray(_X0))))


# --- refusals ----------------------------------------------------------------


def test_straddling_modes_raise_a_mode_label_error():
    omega, p = _circuit(L_J, 5.8e9)
    with pytest.raises(gfem.ModeLabelError, match=r"by 0\.3[56]\d") as info:
        quantum.dressed_modes(omega, p, inductance=L_J)
    assert "first_order' is not a remedy" in str(info.value)
    assert "leave the hybridized mode out" in str(info.value)


def test_the_label_check_survives_jit_when_only_the_dispersive_shift_is_read():
    omega, p = _circuit(L_J, 5.8e9)
    chi = jax.jit(
        lambda o, q: quantum.dressed_modes(o, q, inductance=L_J).dispersive_shift
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="ModeLabelError"):
        np.asarray(chi(omega, p))


def test_a_zero_participation_is_refused():
    with pytest.raises(ValueError, match="exactly zero"):
        quantum.dressed_modes(_OMEGA, _P.at[1].set(0.0), inductance=L_J)
    with pytest.raises(ValueError, match="exactly zero"):
        quantum.dressed_modes(
            _OMEGA, _P.at[1].set(0.0), inductance=L_J, method="first_order"
        )
    run = jax.jit(
        lambda q: quantum.dressed_modes(_OMEGA, q, inductance=L_J).dispersive_shift
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="exactly zero"):
        np.asarray(run(_P.at[1].set(0.0)))


def test_the_dimension_cap_is_refused():
    omega = TWO_PI * np.array([5e9, 6e9, 7e9, 8e9])
    p = np.array([0.9, 0.01, 0.01, 0.01])
    with pytest.raises(
        ValueError, match=r"20736.*MAX_DIMENSION = 2000.*fewer modes or lower"
    ):
        quantum.dressed_modes(omega, p, inductance=L_J)
    # The closed forms build no Hilbert space.
    quantum.dressed_modes(omega, p, inductance=L_J, method="first_order")


@pytest.mark.parametrize(
    ("omega", "p", "kwargs", "match"),
    [
        (_OMEGA, np.stack([_P, _P]), {}, "one junction"),
        (_OMEGA, _P[:1], {}, "same nonzero length"),
        (_OMEGA, _P, {"levels": 2}, "at least 3"),
        (_OMEGA, _P, {"levels": (8, 8, 8)}, "one int per mode"),
        (_OMEGA, _P, {"method": "second_order"}, "method must be one of"),
        (_OMEGA, _P, {"inductance": -L_J}, "inductance must be"),
        (-_OMEGA, _P, {}, "positive real part"),
        (_OMEGA, 2 * _P / _P[0], {}, r"\[-1, 1\]"),
        (_OMEGA.at[0].set(np.nan), _P, {}, "finite"),
    ],
)
def test_bad_arguments_raise_value_error(omega, p, kwargs, match):
    kwargs = {"inductance": L_J, **kwargs}
    with pytest.raises(ValueError, match=match):
        quantum.dressed_modes(omega, p, **kwargs)


def test_mode_label_error_is_public():
    assert gfem.ModeLabelError is gfem.errors.ModeLabelError
    assert issubclass(gfem.ModeLabelError, RuntimeError)
    assert "ModeLabelError" in gfem.__all__
    assert "quantum" in gfem.__all__
    assert quantum.MAX_DIMENSION == 2000
