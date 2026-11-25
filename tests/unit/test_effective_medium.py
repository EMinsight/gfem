"""Effective-medium retrieval: closed-form slabs, round trip, jit, gradient.

The pure retrieval `retrieve` is checked on the closed-form S of a homogeneous
slab (no FEM). Tolerances come from the conditioning of the inverse problem:
the error of the recovered parameter is bounded by the sum over the four real
inputs x of S of |d value / d x| |x| eps_mach, with the Jacobian taken by
`jax.jacobian` at the exact S, times `SAFETY`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem  # noqa: F401  (turns on double precision)
from gfem._diffraction.effective import retrieve

EPS = float(np.finfo(float).eps)
# The test's own S costs a dozen rounded operations (square root, exponential of
# an argument up to about 20, a quotient) and the retrieval about as many more;
# each contributes at most eps_mach relative error amplified by the condition
# number already in the Jacobian bound. 64 covers about forty such steps with
# room for the argument of the exponential, and a wrong sign, branch or axis is
# off by order one, many orders above this.
SAFETY = 64


def slab_s(eps, mu, k0d, eps_b=1.0, mu_b=1.0):
    """Closed-form (S11, S21) of a slab in a background, exp(+i w t), passive root."""
    n = jnp.sqrt(eps * mu + 0j)
    n = jnp.where(jnp.imag(n) > 0, -n, n)
    n_b = math.sqrt(eps_b * mu_b)
    z = (n / eps) / (n_b / eps_b)  # Z / Z_b with Z = n / eps (relative to Z0)
    gamma = (z - 1) / (z + 1)
    t = jnp.exp(-1j * n * k0d)
    den = 1 - gamma**2 * t**2
    return gamma * (1 - t**2) / den, t * (1 - gamma**2) / den


def _inputs(x):
    return x[..., 0] + 1j * x[..., 1], x[..., 2] + 1j * x[..., 3]


def _x_of(s11, s21):
    return jnp.stack([s11.real, s11.imag, s21.real, s21.imag], axis=-1)


def _retrieved(x, k0d, branch, eps_b, mu_b):
    out = retrieve(*_inputs(x), k0d, eps_b, mu_b, branch)
    return out[0], out[1]


def _bound(s11, s21, k0d, branch=0, eps_b=1.0, mu_b=1.0):
    """Error bound of (eps, mu), per frequency: SAFETY eps_mach sum |J| |x|."""
    x = _x_of(s11, s21)

    def parts(x):
        e, m = _retrieved(x, k0d, branch, eps_b, mu_b)
        return jnp.stack([e.real, e.imag, m.real, m.imag], axis=-1)  # (n, 4)

    jac = jax.jit(jax.jacobian(parts))(x)  # (n, 4, n, 4); frequencies couple only via unwrap
    n = x.shape[0]
    diag = jnp.stack([jac[i, :, i, :] for i in range(n)])  # (n, 4 out, 4 in)
    err = SAFETY * EPS * jnp.einsum("noi,ni->no", jnp.abs(diag), jnp.abs(x))
    return np.asarray(jnp.hypot(err[:, 0], err[:, 1])), np.asarray(
        jnp.hypot(err[:, 2], err[:, 3])
    )


def _round_trip(eps, mu, k0d, branch=0, eps_b=1.0, mu_b=1.0):
    k0d = jnp.asarray(k0d, float)
    s11, s21 = slab_s(eps, mu, k0d, eps_b, mu_b)
    got = retrieve(s11, s21, k0d, eps_b, mu_b, branch)
    tol_e, tol_m = _bound(s11, s21, k0d, branch, eps_b, mu_b)
    np.testing.assert_array_less(np.abs(got[0] - eps), np.maximum(tol_e, 1e-300) * 1.0)
    np.testing.assert_array_less(np.abs(got[1] - mu), np.maximum(tol_m, 1e-300) * 1.0)
    return got, (s11, s21, k0d)


K_THIN = np.linspace(0.3, 1.2, 5)
# n = 2: n k0 d runs from 0.4 to 4.4 rad in steps below pi, crossing pi and 2 pi
K_SWEEP = np.linspace(0.2, 2.2, 41)


CASES = {
    "lossless": (4.0 + 0j, 1.0 + 0j, K_THIN),
    "lossy": (4.0 - 0.4j, 1.0 + 0j, K_THIN),
    "magnetic": (3.0 + 0j, 2.0 + 0j, K_THIN),
    "negative_index": (-1.0 - 0.05j, -1.0 - 0.05j, np.linspace(0.3, 2.0, 7)),
    "reactive": (-2.0 + 0j, 1.0 + 0j, np.array([1.0])),
}


@pytest.mark.parametrize("name", CASES)
def test_round_trip_recovers_the_slab_to_the_conditioning_bound(name):
    eps, mu, k0d = CASES[name]
    _round_trip(eps, mu, k0d)


def test_round_trip_in_a_dielectric_background():
    # the background itself would return (eps_b, mu_b); a slab in it is relative to it
    _round_trip(6.0 - 0.3j, 1.5 + 0j, K_THIN, eps_b=2.25, mu_b=1.0)


def test_thick_slab_swept_across_two_pi_unwraps_the_phase():
    (e, m, *_), _ = _round_trip(4.0 + 0j, 1.0 + 0j, K_SWEEP)
    # a lost 2 pi would shift Re n by 2 pi / k0 d, order one at these k0 d
    assert float(jnp.max(jnp.abs(e - 4.0))) < 1e-9
    assert float(jnp.max(jnp.abs(m - 1.0))) < 1e-9


def test_branch_one_selects_the_next_phase_sheet():
    # n k0 d from 5.0 up: its principal phase is n k0 d - 2 pi, so branch 0 is wrong
    k0d = np.linspace(2.5, 3.5, 11)
    _round_trip(4.0 - 0.1j, 1.0 + 0j, k0d, branch=1)
    s11, s21 = slab_s(4.0 - 0.1j, 1.0 + 0j, jnp.asarray(k0d))
    wrong = retrieve(s11, s21, jnp.asarray(k0d), 1.0, 1.0, 0)[0]
    assert float(jnp.max(jnp.abs(wrong - 4.0 + 0.1j))) > 0.5


def test_the_background_retrieves_itself():
    k0d = jnp.asarray(K_THIN)
    n_b = math.sqrt(2.25 * 1.2)
    s21 = jnp.exp(-1j * n_b * k0d)
    e, m, n, z = retrieve(jnp.zeros_like(s21), s21, k0d, 2.25, 1.2, 0)
    # S11 = 0: z = 1 exactly, so eps and mu are the background's up to the phase
    np.testing.assert_allclose(e, 2.25, rtol=0, atol=SAFETY * EPS * 4)
    np.testing.assert_allclose(m, 1.2, rtol=0, atol=SAFETY * EPS * 4)
    np.testing.assert_allclose(n, n_b, rtol=0, atol=SAFETY * EPS * 4)


def test_two_columns_are_retrieved_independently():
    k0d = jnp.asarray(K_THIN)
    s11a, s21a = slab_s(4.0 - 0.4j, 1.0 + 0j, k0d)
    s11b, s21b = slab_s(2.0 + 0j, 1.5 + 0j, k0d)
    e, m, n, z = retrieve(
        jnp.stack([s11a, s11b], axis=1),
        jnp.stack([s21a, s21b], axis=1),
        k0d,
        1.0,
        1.0,
        0,
    )
    assert e.shape == m.shape == n.shape == z.shape == (len(K_THIN), 2)
    ta, ma = _bound(s11a, s21a, k0d)
    tb, mb = _bound(s11b, s21b, k0d)
    assert np.all(np.abs(e[:, 0] - (4.0 - 0.4j)) < ta)
    assert np.all(np.abs(e[:, 1] - 2.0) < tb)
    assert np.all(np.abs(m[:, 1] - 1.5) < mb)


def test_jit_agrees_with_eager_to_the_same_bound():
    eps, mu = 4.0 - 0.4j, 2.0 + 0j
    k0d = jnp.asarray(K_SWEEP)
    s11, s21 = slab_s(eps, mu, k0d)
    eager = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
    jitted = jax.jit(retrieve, static_argnums=(5,))(s11, s21, k0d, 1.0, 1.0, 0)
    tol_e, tol_m = _bound(s11, s21, k0d)
    # eager and fused evaluations differ by a few units in the last place per
    # operation, amplified by the same conditioning; 2 SAFETY covers both
    assert np.all(np.abs(eager[0] - jitted[0]) < 2 * tol_e)
    assert np.all(np.abs(eager[1] - jitted[1]) < 2 * tol_m)


@pytest.mark.parametrize("eps0", [4.0, 9.0])
def test_gradient_of_the_retrieved_permittivity_is_one(eps0):
    k0d = jnp.asarray(K_THIN)
    mu = 1.0 + 0j

    def parts(t):
        s11, s21 = slab_s(t - 0.4j, mu, k0d)
        e, m, *_ = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
        return jnp.stack([e.real, e.imag, m.real, m.imag])

    d = np.asarray(jax.jacfwd(parts)(jnp.asarray(eps0)))  # (4, n)

    # error of the chain rule: SAFETY eps sum_k |d out/d x_k| |d x_k/d t|
    def x_of_t(t):
        return _x_of(*slab_s(t - 0.4j, mu, k0d))

    s11, s21 = slab_s(eps0 - 0.4j, mu, k0d)
    x = _x_of(s11, s21)
    dx = jax.jacfwd(x_of_t)(jnp.asarray(eps0))  # (n, 4)

    def out_of_x(x):
        e, m = _retrieved(x, k0d, 0, 1.0, 1.0)
        return jnp.stack([e.real, e.imag, m.real, m.imag])

    jac = jax.jit(jax.jacobian(out_of_x))(x)  # (4, n, n, 4)
    tol = np.zeros((4, len(K_THIN)))
    for i in range(len(K_THIN)):
        tol[:, i] = SAFETY * EPS * np.abs(jac[:, i, i, :]) @ np.abs(dx[i])
    expected = np.array([1.0, 0.0, 0.0, 0.0])[:, None] * np.ones_like(d)
    assert np.all(np.abs(d - expected) < np.maximum(tol, 1e-300))


def test_a_reactive_impedance_takes_the_passive_root_when_the_principal_one_is_wrong():
    # Z = -1e-4 - 0.7i has Re z < 0 (round-off in a reactive slab), so the
    # principal square root returns -z; |T| <= 1 picks the right sign back.
    z = -1e-4 - 0.7j
    n = -1.2j
    k0d = jnp.asarray([0.8, 1.0])
    gamma = (z - 1) / (z + 1)
    t = jnp.exp(-1j * n * k0d)
    den = 1 - gamma**2 * t**2
    s11, s21 = gamma * (1 - t**2) / den, t * (1 - gamma**2) / den
    assert float(jnp.abs(t[0])) < 1
    e, m, n_got, z_got = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
    tol = 1e-10  # |z| = 0.7 and T = 0.4: conditioning of order 1e2 on 1e-16 inputs
    np.testing.assert_allclose(z_got, z, rtol=0, atol=tol)
    np.testing.assert_allclose(n_got, n, rtol=0, atol=tol)
