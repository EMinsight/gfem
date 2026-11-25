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

import gfem
from gfem._diffraction.effective import retrieve

EPS = float(np.finfo(float).eps)
# The test's own S costs a dozen rounded operations (square root, exponential of
# an argument up to about 20, a quotient) and the retrieval about as many more;
# each contributes at most eps_mach relative error amplified by the condition
# number already in the Jacobian bound. 64 covers about forty such steps with
# room for the argument of the exponential, and a wrong sign, branch or axis is
# off by order one, many orders above this.
SAFETY = 64
# Floor of a bound that is exactly zero (a retrieved value with no round-off):
# the smallest normal double, so the strict comparison still holds at equality.
TINY = float(np.finfo(float).tiny)


def slab_s(eps, mu, k0d, *, eps_b=1.0, mu_b=1.0):
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


def _bound(s11, s21, k0d, *, branch=0, eps_b=1.0, mu_b=1.0, which=(0, 1)):
    """Error bound of two outputs of `retrieve`, per frequency.

    SAFETY eps_mach sum |J| |x| of the outputs ``which`` (0 eps, 1 mu, 2 n,
    3 z), each as a complex modulus.
    """
    x = _x_of(s11, s21)

    def parts(x):
        out = retrieve(*_inputs(x), k0d, eps_b, mu_b, branch)
        a, b = (out[i] for i in which)
        return jnp.stack([a.real, a.imag, b.real, b.imag], axis=-1)  # (n, 4)

    jac = jax.jit(jax.jacobian(parts))(
        x
    )  # (n, 4, n, 4); frequencies couple only via unwrap
    n = x.shape[0]
    diag = jnp.stack([jac[i, :, i, :] for i in range(n)])  # (n, 4 out, 4 in)
    err = SAFETY * EPS * jnp.einsum("noi,ni->no", jnp.abs(diag), jnp.abs(x))
    return np.asarray(jnp.hypot(err[:, 0], err[:, 1])), np.asarray(
        jnp.hypot(err[:, 2], err[:, 3])
    )


def _round_trip(eps, mu, k0d, *, branch=0, eps_b=1.0, mu_b=1.0):
    k0d = jnp.asarray(k0d, float)
    s11, s21 = slab_s(eps, mu, k0d, eps_b=eps_b, mu_b=mu_b)
    got = retrieve(s11, s21, k0d, eps_b, mu_b, branch)
    tol_e, tol_m = _bound(s11, s21, k0d, branch=branch, eps_b=eps_b, mu_b=mu_b)
    np.testing.assert_array_less(np.abs(got[0] - eps), np.maximum(tol_e, TINY))
    np.testing.assert_array_less(np.abs(got[1] - mu), np.maximum(tol_m, TINY))
    return got, (s11, s21, k0d)


K_THIN = np.linspace(0.3, 1.2, 5)
# n = 2: n k0 d runs from 0.4 to 10 rad in steps of 0.1 rad (well below pi, so
# the unwrap is unambiguous), which crosses the principal phase's wraps at pi
# and 3 pi. The grid hits no multiple of pi (0.4 + 0.1 j is never 3.14159...),
# where a lossless slab is singular; the bound below follows the conditioning
# at the nearest points.
K_SWEEP = np.linspace(0.2, 5.0, 97)


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
    _, (s11, s21, k0d) = _round_trip(4.0 + 0j, 1.0 + 0j, K_SWEEP)
    # a lost 2 pi shifts n by 2 pi / k0 d and eps = n / z by that over |z| = 1/2
    # (z = sqrt(mu / eps)); the round trip above holds the error to the
    # conditioning bound, which must lie far below that shift for the check to
    # see a lost wrap: 1/2 of the shift is the midpoint between right and wrong
    shift = 2 * np.pi / (float(k0d.max()) * 0.5)
    tol_e, tol_m = _bound(s11, s21, k0d)
    assert np.max(tol_e) < shift / 2
    assert np.max(tol_m) < shift / 2


def test_branch_one_selects_the_next_phase_sheet():
    # n k0 d from 5.0 up: its principal phase is n k0 d - 2 pi, so branch 0 is wrong
    k0d = np.linspace(2.5, 3.5, 11)
    _round_trip(4.0 - 0.1j, 1.0 + 0j, k0d, branch=1)
    s11, s21 = slab_s(4.0 - 0.1j, 1.0 + 0j, jnp.asarray(k0d))
    wrong = retrieve(s11, s21, jnp.asarray(k0d), 1.0, 1.0, 0)[0]
    # the wrong sheet moves n by 2 pi / k0 d, so eps = n / z by 2 pi / (k0 d |z|),
    # |z| = |n / eps| of the slab; the error is that to the bound
    n = jnp.sqrt(4.0 - 0.1j)
    shift = 2 * np.pi / (k0d * float(abs(n / (4.0 - 0.1j))))
    tol_e, _ = _bound(s11, s21, jnp.asarray(k0d), branch=0)
    assert np.all(np.abs(np.abs(wrong - (4.0 - 0.1j)) - shift) < tol_e + TINY)


def test_a_pair_of_branches_sets_each_column_sheet():
    # column 0 on sheet 0 (n k0 d from 1.5 to 2.3 rad, below pi), column 1 on
    # sheet 1 (n = 3, n k0 d from 3.6 to 5.7 rad, above pi and below 3 pi)
    k0d = jnp.asarray(np.linspace(1.2, 1.9, 5))
    ea, eb = 1.5 + 0j, 9.0 - 0.05j
    s11a, s21a = slab_s(ea, 1.0 + 0j, k0d)
    s11b, s21b = slab_s(eb, 1.0 + 0j, k0d)
    s11, s21 = jnp.stack([s11a, s11b], axis=1), jnp.stack([s21a, s21b], axis=1)
    e, *_ = retrieve(s11, s21, k0d, 1.0, 1.0, (0, 1))
    ta, _ = _bound(s11a, s21a, k0d, branch=0)
    tb, _ = _bound(s11b, s21b, k0d, branch=1)
    assert np.all(np.abs(e[:, 0] - ea) < ta)
    assert np.all(np.abs(e[:, 1] - eb) < tb)
    # one int serves both columns, so column 1 is then on the wrong sheet: off
    # by 2 pi / (k0 d |z|), |z| = |n / eps|, to the bound
    same, *_ = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
    shift = 2 * np.pi / (k0d * abs(jnp.sqrt(eb) / eb))
    assert np.all(np.abs(np.abs(same[:, 1] - eb) - shift) < tb + TINY)
    # a broadcast pair equals the int
    np.testing.assert_array_equal(
        retrieve(s11, s21, k0d, 1.0, 1.0, (1, 1))[0],
        retrieve(s11, s21, k0d, 1.0, 1.0, 1)[0],
    )


def test_the_background_retrieves_itself():
    k0d = jnp.asarray(K_THIN)
    n_b = math.sqrt(2.25 * 1.2)
    scale = 2.25  # the largest of eps_b, mu_b and n_b: the round-off is relative
    s21 = jnp.exp(-1j * n_b * k0d)
    e, m, n, _ = retrieve(jnp.zeros_like(s21), s21, k0d, 2.25, 1.2, 0)
    # S11 = 0: z = 1 exactly, so eps and mu are the background's up to the phase
    np.testing.assert_allclose(e, 2.25, rtol=0, atol=SAFETY * EPS * scale)
    np.testing.assert_allclose(m, 1.2, rtol=0, atol=SAFETY * EPS * scale)
    np.testing.assert_allclose(n, n_b, rtol=0, atol=SAFETY * EPS * scale)


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
    ta, _ = _bound(s11a, s21a, k0d)
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
        e, m, _, _ = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
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
    assert np.all(np.abs(d - expected) < np.maximum(tol, TINY))


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
    _, _, n_got, z_got = retrieve(s11, s21, k0d, 1.0, 1.0, 0)
    # round-off of S propagated through the retrieval at this S (the same
    # SAFETY eps_mach sum |J| |x| as the eps, mu bound), for (n, z)
    tol_n, tol_z = _bound(s11, s21, k0d, which=(2, 3))
    assert np.all(np.abs(z_got - z) < tol_z)
    assert np.all(np.abs(n_got - n) < tol_n)


# --- the reader on a tiny real cell ----------------------------------------------

import dataclasses  # noqa: E402
import warnings  # noqa: E402

import periodic_cases as pc  # noqa: E402

from gfem import diffraction as dif  # noqa: E402
from gfem._diffraction import effective as eff  # noqa: E402

OPTS = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
TOP, BOT = "z_max", "z_min"
F0 = pc.F1_FREQUENCY
D = pc.F1_D * pc.UM


def _ports(offset=D, drop=()):
    return [
        dataclasses.replace(p, offset=offset)
        for p in pc.f1_ports()
        if p.name not in drop
    ]


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("em"))


@pytest.fixture(scope="module")
def model(mesh):
    # air / eps 4 slab / air, port planes on the slab faces
    return pc.f1(mesh, ports=_ports(), eps_sub=1.0)


@pytest.fixture(scope="module")
def solved(model):
    return gfem.solve(model, gfem.Driven([F0]), options=OPTS)


def test_a_solved_slab_reads_eps_four_and_mu_one_in_both_columns(solved, model):
    # n k0 d = 2 * 2 pi * 0.3 = 3.77 rad exceeds pi: the principal phase has
    # wrapped, so the slab is on the second sheet (branch 0 returns eps = -2.67)
    em = dif.effective_medium(solved, TOP, BOT, model=model, branch=1)
    assert em.branch == (1, 1)
    assert em.permittivity.shape == em.permeability.shape == (1, 2)
    # the thickness is a difference of two plane coordinates of the face
    # geometry (up to 2 D + H in um), each good to SAFETY eps_mach relative
    # to that extent
    extent = 2 * pc.F1_D + pc.F1_H
    assert em.thickness == pytest.approx(
        pc.F1_H * pc.UM, rel=SAFETY * EPS * extent / pc.F1_H
    )
    # |S - Airy| <= F1_COARSE_TOL per entry, so each real and imaginary part of
    # S11 and S21 is off by at most that. The retrieved value moves by the
    # Jacobian of retrieve integrated along the segment from the Airy S to the
    # solved S; the l1 row sum of |J| times F1_COARSE_TOL bounds it with the
    # largest |J| on the segment, and J is smooth along it, so its endpoints
    # bracket it up to the second-order remainder, relative size
    # F1_COARSE_TOL times the curvature of retrieve, far below one. A swap
    # eps <-> mu (the TM sign question) or of the columns moves the value by 3
    # or more.
    # the Airy slab between planes on its faces: the ports' offsets put the
    # reference planes there, so the air layers have zero thickness
    layers = [(1.0, 0.0), (pc.F1_EPS_SLAB, pc.F1_H * pc.UM), (1.0, 0.0)]
    airy = pc.airy(F0, 0.0, layers, "TM")
    solved_s = (
        complex(solved.s["R_TM", "R_TM"][0]),
        complex(solved.s["T_TM", "R_TM"][0]),
    )
    k0d = jnp.asarray([2 * math.pi * F0 / pc.C0 * pc.F1_H * pc.UM])

    def l1_rows(s11, s21):  # |d (Re eps, Im eps, Re mu, Im mu) / d (Re, Im of S)|
        def out(x):
            e, m = _retrieved(x, k0d, 1, 1.0, 1.0)
            return jnp.stack([e.real, e.imag, m.real, m.imag], axis=-1)

        x = _x_of(jnp.asarray([s11]), jnp.asarray([s21]))
        return np.abs(np.asarray(jax.jacobian(out)(x))[0, :, 0, :])  # (4, 4)

    rows = np.maximum(l1_rows(*airy), l1_rows(*solved_s)).sum(axis=1)
    err = pc.F1_COARSE_TOL * rows
    tol_e, tol_m = np.hypot(err[0], err[1]), np.hypot(err[2], err[3])
    assert np.all(np.abs(np.asarray(em.permittivity[0]) - 4.0) < tol_e)
    assert np.all(np.abs(np.asarray(em.permeability[0]) - 1.0) < tol_m)


def test_an_angle_sweep_is_refused(solved, model):
    with pytest.raises(ValueError, match="angle sweep"):
        dif.effective_medium([solved, solved], TOP, BOT, model=model)


def test_oblique_incidence_is_refused(model):
    k = model.bloch_wave_vector(TOP, F0, theta=pc.F1_THETA, phi=pc.F1_PHI)
    result = gfem.solve(
        model,
        gfem.Driven([F0], wave_vector=k, wave_vector_frequency=F0),
        options=OPTS,
    )
    with pytest.raises(ValueError, match="oblique incidence"):
        dif.effective_medium(result, TOP, BOT, model=model)


def test_the_same_face_twice_is_refused(solved, model):
    with pytest.raises(ValueError, match="both 'z_max'"):
        dif.effective_medium(solved, TOP, TOP, model=model)


def test_an_unknown_face_is_refused(solved, model):
    with pytest.raises(ValueError, match="carries no Floquet port"):
        dif.effective_medium(solved, TOP, "x_min", model=model)


def test_different_media_behind_the_faces_are_not_implemented(mesh):
    model = pc.f1(mesh, ports=_ports())  # substrate eps 2.25 under air
    result = gfem.solve(model, gfem.Driven([F0]), options=OPTS)
    with pytest.raises(NotImplementedError, match="differ"):
        dif.effective_medium(result, TOP, BOT, model=model)


def test_crossed_reference_planes_are_refused(solved, model):
    # the faces are 2 D + H = 0.8 um apart: 0.5 um inward from each crosses them
    crossed = solved.with_offsets(dict.fromkeys(solved.s.ports, 0.5e-6))
    with pytest.raises(ValueError, match="meeting or crossed"):
        dif.effective_medium(crossed, TOP, BOT, model=model)


def test_frequencies_must_increase(model):
    result = gfem.solve(model, gfem.Driven([1.05 * F0, F0]), options=OPTS)
    with pytest.raises(ValueError, match="increase strictly"):
        dif.effective_medium(result, TOP, BOT, model=model)


def test_a_missing_polarization_port_is_refused(mesh):
    model = pc.f1(mesh, ports=_ports(drop=("T_TM",)), eps_sub=1.0)
    with pytest.warns(gfem.UnrequestedOrderWarning):
        result = gfem.solve(model, gfem.Driven([F0]), options=OPTS)
    with pytest.raises(ValueError, match="no Floquet port reads order"):
        dif.effective_medium(result, TOP, BOT, model=model)


def test_an_unexcited_incident_column_is_refused(model):
    result = gfem.solve(
        model, gfem.Driven([F0], excite=["R_TE", "T_TE", "T_TM"]), options=OPTS
    )
    with pytest.raises(ValueError, match="was not excited"):
        dif.effective_medium(result, TOP, BOT, model=model)


def test_ports_of_one_order_at_different_offsets_are_refused(solved, model):
    skew = solved.with_offsets({"R_TE": 0.1e-6})
    with pytest.raises(ValueError, match="different offsets"):
        dif.effective_medium(skew, TOP, BOT, model=model)


def test_a_pair_of_branches_is_kept_per_column(solved, model):
    em = dif.effective_medium(solved, TOP, BOT, model=model, branch=(1, 1))
    assert em.branch == (1, 1)
    both = dif.effective_medium(solved, TOP, BOT, model=model, branch=np.int64(1))
    assert both.branch == (1, 1)
    # column 1 on sheet 0 differs from the int; column 0 is unchanged
    pair = dif.effective_medium(solved, TOP, BOT, model=model, branch=(1, 0))
    assert pair.branch == (1, 0)
    np.testing.assert_array_equal(pair.index[:, 0], em.index[:, 0])
    assert not np.allclose(pair.index[:, 1], em.index[:, 1])


@pytest.mark.parametrize(
    "branch", [(1, 2, 3), (1,), (0, 1.0), (0, True), ("0", 0), [0, 1.5], ()]
)
def test_a_bad_branch_pair_is_refused(solved, model, branch):
    with pytest.raises(TypeError, match="branch must be an int or a pair of ints"):
        dif.effective_medium(solved, TOP, BOT, model=model, branch=branch)


def test_a_traced_element_of_a_branch_pair_is_not_differentiable(solved, model):
    with pytest.raises(gfem.NonDifferentiableError, match="branch"):
        jax.jit(
            lambda b: dif.effective_medium(solved, TOP, BOT, model=model, branch=(1, b))
        )(1)


@pytest.mark.parametrize("branch", [1.0, "0", True])
def test_branch_must_be_an_int(solved, model, branch):
    with pytest.raises(TypeError, match="branch must be an int"):
        dif.effective_medium(solved, TOP, BOT, model=model, branch=branch)


def test_a_traced_branch_is_not_differentiable(solved, model):
    with pytest.raises(gfem.NonDifferentiableError, match="branch"):
        jax.jit(
            lambda b: dif.effective_medium(solved, TOP, BOT, model=model, branch=b)
        )(1)


def test_the_slab_is_symmetric_and_isotropic_so_nothing_warns(solved, model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dif.effective_medium(solved, TOP, BOT, model=model)


# Injected faults, in multiples of the diagnostic threshold they must cross. The
# test slab is passive and nearly lossless, |r|^2 + |t|^2 close to 1 and |t| the
# larger share, so a cross entry of LEAK times the co-polarized transmission
# exceeds the threshold relative to sqrt(|r|^2 + |t|^2).
LEAK = 10 * eff._DIAGNOSTIC
SKEW = 5 * eff._DIAGNOSTIC


def test_cross_polarized_transmission_warns(solved, model, monkeypatch):
    real = eff.jones_block

    def leaky(s, out, inc):
        block = real(s, out, inc)
        if out[0] != inc[0]:  # the transmission block: add LEAK of TM into TE
            block = block.at[:, 1, 0].set(LEAK * block[:, 0, 0])
        return block

    monkeypatch.setattr(eff, "jones_block", leaky)
    with pytest.warns(UserWarning, match="cross-polarized transmission"):
        dif.effective_medium(solved, TOP, BOT, model=model)


def test_cross_polarized_noise_beside_a_vanishing_reflection_does_not_warn(
    solved, model, monkeypatch
):
    real = eff.jones_block
    noise = 64 * EPS  # round-off level of a reflection that cancels

    def matched(s, out, inc):
        block = real(s, out, inc)
        if out[0] == inc[0]:  # both reflections: co-polarized 0, cross at noise
            block = jnp.zeros_like(block)
            return block.at[:, 0, 1].set(noise).at[:, 1, 0].set(noise)
        return block

    monkeypatch.setattr(eff, "jones_block", matched)
    with warnings.catch_warnings():
        # relative to the reflection's own co-polarized entry (0) the cross
        # entry would be infinite; relative to the column's co-polarized total
        # (|t| about 1) it is `noise`, far below the threshold
        warnings.simplefilter("error")
        dif.effective_medium(solved, TOP, BOT, model=model)


def test_faces_that_are_not_antiparallel_are_refused(solved, model, monkeypatch):
    real = type(eff.cell_of(solved, model, "t")).face

    def tilted(self, surface):
        face = real(self, surface)
        if surface == BOT:  # the same inward normal as the top face
            return dataclasses.replace(face, normal=-face.normal)
        return face

    monkeypatch.setattr(type(eff.cell_of(solved, model, "t")), "face", tilted)
    with pytest.raises(ValueError, match="not antiparallel"):
        dif.effective_medium(solved, TOP, BOT, model=model)


def test_a_reflection_that_depends_on_the_side_warns(solved, model, monkeypatch):
    real = eff.jones_block

    def lopsided(s, out, inc):
        block = real(s, out, inc)
        if out == inc and out[0].startswith("T"):  # the back-side reflection
            block = block + SKEW
        return block

    monkeypatch.setattr(eff, "jones_block", lopsided)
    with pytest.warns(UserWarning, match="differ by"):
        dif.effective_medium(solved, TOP, BOT, model=model)


def test_effective_medium_checks_its_shapes():
    ok = np.ones((3, 2), complex)
    with pytest.raises(ValueError, match="permittivity must have shape"):
        dif.EffectiveMedium(ok[:, :1], ok, ok, ok, 1e-7, np.arange(3.0), 0)
    with pytest.raises(ValueError, match="frequency must hold 3"):
        dif.EffectiveMedium(ok, ok, ok, ok, 1e-7, np.arange(2.0), 0)
    assert "3 frequencies" in repr(
        dif.EffectiveMedium(ok, ok, ok, ok, 1e-7, np.arange(3.0), 0)
    )
