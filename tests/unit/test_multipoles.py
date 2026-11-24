r"""Multipoles: the projection, the Cartesian maps, the helpers and the result types.

Everything here is synthetic: closed-form far fields of point dipoles and
random coefficients, no solved field.
"""

import dataclasses
import math
import pickle

import free_space_cases as fsc
import jax
import jax.numpy as jnp
import multipole_cases as mc
import numpy as np
import pytest
import scipy.constants as const

from gfem import _multipoles as mpm

C0, MU0, EPS0 = const.c, const.mu_0, const.epsilon_0
F0 = C0 / 1e-6
BACKGROUNDS = [(1.0, 1.0), (2.0, 3.0)]


def _background(eps_b, mu_b, frequency=F0):
    omega = 2 * math.pi * frequency
    eps, mu = EPS0 * eps_b, MU0 * mu_b
    return omega, eps, mu, omega * math.sqrt(eps * mu), math.sqrt(mu / eps)


def _directions(grid):
    th, ph = grid.theta, grid.phi
    n = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    that = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    phat = np.stack([-np.sin(ph), np.cos(ph), 0 * ph], -1)
    return n, that, phat


def _dipole_far_field(grid, k, eps, eta, *, p=None, m=None, at=(0.0, 0.0, 0.0)):
    r"""The far-field amplitude :math:`F` (V) of point moments at ``at``."""
    n, that, phat = _directions(grid)
    f = np.zeros(n.shape, complex)
    if p is not None:
        f += k**2 / (4 * math.pi * eps) * np.cross(n, np.cross(p, n))
    if m is not None:
        f += k**2 * eta / (4 * math.pi) * np.cross(m, n)
    f = f * np.exp(1j * k * (n @ np.asarray(at)))[..., None]
    return np.sum(f * that, -1), np.sum(f * phat, -1)


def _multipoles(alpha, beta, *, lmax, frequency=(F0,), bg=(1.0, 1.0), e_inc=1.0):
    eta = _background(bg[0], bg[1], frequency[0])[4]
    omega = np.asarray(frequency) * 2 * math.pi
    ks = tuple(float(o * math.sqrt(EPS0 * bg[0] * MU0 * bg[1])) for o in omega)
    return mpm.Multipoles(
        frequency=tuple(frequency),
        origin=(0.0, 0.0, 0.0),
        lmax=lmax,
        wavenumber=ks,
        impedance=eta,
        incident_amplitude=jnp.full(len(frequency), e_inc, complex),
        alpha=alpha,
        beta=beta,
    )


_INDEX_MESSAGE = (
    r"^Multipoles\.index: \(l, m\) = \(3, 4\) is not a multipole of this "
    r"result: 1 <= l <= lmax = 7 and -l <= m <= l\.$"
)
RNG = np.random.default_rng(7)
P_VEC = RNG.normal(size=3) + 1j * RNG.normal(size=3)
M_VEC = RNG.normal(size=3) + 1j * RNG.normal(size=3)


def test_gauss_grid_weights():
    g = mpm._gauss_grid(9)
    assert g.theta.shape == (9, 18) == g.weight.shape
    assert g.weight.sum() == pytest.approx(4 * math.pi, rel=1e-14)
    assert np.all((g.theta > 0) & (g.theta < math.pi))


@pytest.mark.parametrize("bg", BACKGROUNDS)
def test_u1_dipole_projection_identity(bg):
    """Dipoles at the origin give only l = 1 and p, m back, in a medium too."""
    _, eps, _, k, eta = _background(*bg)
    lmax = 4
    grid = mpm._gauss_grid(16)
    pm = {}
    for name, kw in (("p", {"p": P_VEC * 1e-30}), ("m", {"m": M_VEC * 1e-22})):
        ft, fp = _dipole_far_field(grid, k, eps, eta, **kw)
        c_a, c_b = mpm._project(ft[None], fp[None], grid, k=[k], lmax=lmax)
        res = _multipoles(c_a, c_b, lmax=lmax, bg=bg)
        norm = np.linalg.norm(np.concatenate([c_a, c_b], axis=1))
        higher = np.concatenate([c_a[:, 3:], c_b[:, 3:]], axis=1)
        assert np.linalg.norm(higher) / norm < 1e-12
        pm[name] = res
    p, m = (
        np.asarray(pm["p"].electric_dipole)[0],
        np.asarray(pm["m"].magnetic_dipole)[0],
    )
    assert np.linalg.norm(p - P_VEC * 1e-30) / np.linalg.norm(P_VEC * 1e-30) < 1e-12
    assert np.linalg.norm(m - M_VEC * 1e-22) / np.linalg.norm(M_VEC * 1e-22) < 1e-12
    # The electric dipole has no magnetic dipole and the reverse.
    assert (
        np.abs(np.asarray(pm["p"].beta)).max()
        < 1e-12 * np.abs(np.asarray(pm["p"].alpha)).max()
    )
    assert (
        np.abs(np.asarray(pm["m"].alpha)).max()
        < 1e-12 * np.abs(np.asarray(pm["m"].beta)).max()
    )


@pytest.mark.parametrize("bg", BACKGROUNDS)
def test_u2_off_origin_dipole_keeps_the_power(bg):
    """Moved by d, a dipole spreads over the degrees; the total power holds."""
    _, eps, _, k, eta = _background(*bg)
    d = np.array([0.31, -0.22, 0.45]) * 1e-6
    x = k * np.linalg.norm(d)
    lmax = mpm._wiscombe(x)
    n_theta = math.ceil(x) + lmax + 4
    grid = mpm._gauss_grid(n_theta)
    p = P_VEC * 1e-30
    ft, fp = _dipole_far_field(grid, k, eps, eta, p=p, at=d)
    c_a, c_b = mpm._project(ft[None], fp[None], grid, k=[k], lmax=lmax)
    power = float(
        (np.sum(np.abs(c_a) ** 2) + np.sum(np.abs(c_b) ** 2)) / (2 * eta * k**2)
    )
    expected = k**4 * np.linalg.norm(p) ** 2 / (12 * math.pi * eps**2 * eta)
    assert abs(power - expected) / expected < 1e-8
    # The partial cross sections moved: the expansion is not l = 1 only.
    spread = np.linalg.norm(np.asarray(c_a)[0, 3:]) / np.linalg.norm(np.asarray(c_a))
    assert spread > 1e-2


def test_u3_cartesian_maps_match_partial_cross_sections():
    lmax, n_freq = 3, 2
    n_lm = lmax * (lmax + 2)
    a = RNG.normal(size=(n_freq, n_lm)) + 1j * RNG.normal(size=(n_freq, n_lm))
    b = RNG.normal(size=(n_freq, n_lm)) + 1j * RNG.normal(size=(n_freq, n_lm))
    freqs = (F0, 1.3 * F0)
    for bg in BACKGROUNDS:
        e_inc = 1.7 - 0.4j
        res = _multipoles(a, b, lmax=lmax, frequency=freqs, bg=bg, e_inc=e_inc)
        k = np.asarray(res.wavenumber)
        eta = res.impedance
        omega = 2 * math.pi * np.asarray(freqs)
        eps = k / (omega * eta)
        e2 = abs(e_inc) ** 2
        p, m = np.asarray(res.electric_dipole), np.asarray(res.magnetic_dipole)
        qe, qm = (
            np.asarray(res.electric_quadrupole),
            np.asarray(res.magnetic_quadrupole),
        )
        sp = k**4 * np.sum(np.abs(p) ** 2, 1) / (6 * math.pi * eps**2 * e2)
        sm = k**4 * eta**2 * np.sum(np.abs(m) ** 2, 1) / (6 * math.pi * e2)
        sqe = k**6 * np.sum(np.abs(qe) ** 2, (1, 2)) / (720 * math.pi * eps**2 * e2)
        sqm = k**6 * eta**2 * np.sum(np.abs(qm) ** 2, (1, 2)) / (720 * math.pi * e2)
        se, sh = (
            np.asarray(res.electric_scattering),
            np.asarray(res.magnetic_scattering),
        )
        for got, want in (
            (sp, se[:, 0]),
            (sm, sh[:, 0]),
            (sqe, se[:, 1]),
            (sqm, sh[:, 1]),
        ):
            assert np.max(np.abs(got - want) / want) < 1e-12
        for q in (qe, qm):
            assert np.abs(q - q.transpose(0, 2, 1)).max() < 1e-12 * np.abs(q).max()
            assert np.abs(np.trace(q, axis1=1, axis2=2)).max() < 1e-12 * np.abs(q).max()
        assert np.allclose(res.scattering, se.sum(1) + sh.sum(1), rtol=1e-14)
        assert np.allclose(res.intensity, e2 / (2 * eta), rtol=1e-14)


def test_u3_degree_one_blocks_are_the_first_three():
    """The degree-1 entries come first whatever lmax."""
    a = RNG.normal(size=(1, 8)) + 0j
    b = RNG.normal(size=(1, 8)) + 0j
    short = _multipoles(a, b, lmax=2)
    longer = _multipoles(
        np.pad(a, ((0, 0), (0, 7))), np.pad(b, ((0, 0), (0, 7))), lmax=3
    )
    assert np.allclose(short.electric_dipole, longer.electric_dipole, rtol=1e-14)
    assert np.allclose(
        short.electric_quadrupole, longer.electric_quadrupole, rtol=1e-14
    )


def test_u4_mie_helpers():
    lmax = 6
    for case in mc.CASES.values():
        alpha, beta = mc.mie_multipoles(case.eps, case.x, lmax, mu=case.mu)
        a, b = mc.mie_coefficients(case.eps, case.x, lmax, mu=case.mu)
        res = _multipoles(alpha[None], beta[None], lmax=lmax)
        # The map gives the textbook partial cross sections, via k^2 = 1.
        k2 = np.asarray(res.wavenumber) ** 2
        n = np.arange(1, lmax + 1)
        want_e = 2 * math.pi * (2 * n + 1) * np.abs(a) ** 2
        want_m = 2 * math.pi * (2 * n + 1) * np.abs(b) ** 2
        got_e = np.asarray(res.electric_scattering)[0] * k2[0]
        got_m = np.asarray(res.magnetic_scattering)[0] * k2[0]
        assert np.max(np.abs(got_e - want_e)) < 1e-13 * want_e.max() + 1e-300
        assert np.max(np.abs(got_m - want_m)) < 1e-13 * want_m.max() + 1e-300
        # Only m = +-1 is occupied.
        assert np.all(alpha[np.abs(res.m) != 1] == 0)
        assert np.all(beta[np.abs(res.m) != 1] == 0)


@pytest.mark.parametrize("case", ["A", "L", "PEC"])
def test_u4_mie_coefficients_reproduce_the_efficiencies(case):
    c = mc.CASES[case]
    lmax = mpm._wiscombe(c.x) + 6
    a, b = mc.mie_coefficients(c.eps, c.x, lmax)
    n = np.arange(1, lmax + 1)
    ext = 2 / c.x**2 * np.sum((2 * n + 1) * np.real(a + b))
    sca = 2 / c.x**2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    e0, ab0, s0 = fsc.mie(c.eps, c.x)
    assert abs(ext - e0) < 1e-12 * e0
    assert abs(sca - s0) < 1e-12 * s0
    assert abs((ext - sca) - ab0) < 1e-12 * max(abs(ab0), 1e-3)


def test_u4_reference_values():
    """Kerker C has a_1 = b_1 and the anapole D' has a_1 = 0 (conjugate of BH's)."""
    a, b = mc.mie_coefficients(**_case_args("C"), lmax=2)
    assert abs(a[0] - (0.1049243606116 + 0.3064559334747j)) < 1e-11
    assert abs(a[0] - b[0]) < 1e-9
    assert abs(a[1] - (6.653478236088e-5 + 8.156614216917e-3j)) < 1e-12
    a, b = mc.mie_coefficients(**_case_args("D'"), lmax=2)
    assert abs(a[0]) < 1e-9
    assert abs(b[0] - (7.622191710054e-2 - 2.653528527340e-1j)) < 1e-11


def _case_args(name):
    c = mc.CASES[name]
    return {"eps": c.eps, "x": c.x, "mu": c.mu}


def test_u4_error_metric():
    alpha, beta = mc.mie_multipoles(4 - 1j, 1.5, 3)
    assert np.all(mc.degree_errors(alpha, beta, alpha, beta) == 0)
    assert mc.leak(alpha, beta, alpha, beta) == 0
    bumped = alpha.copy()
    j = 5  # (l, m) = (2, 0), empty in the Mie map
    bumped[j] = 0.01 * np.linalg.norm(np.concatenate([alpha, beta]))
    e = mc.degree_errors(bumped, beta, alpha, beta)
    assert e[0] == 0
    assert e[2] == 0
    assert e[1] == pytest.approx(0.01)
    assert mc.leak(bumped, beta, alpha, beta) == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("x", "lmax"), [(0.0, 2), (0.7839, 7), (1.0, 7), (1.1654, 8), (2.0, 10)]
)
def test_u5_default_degree(x, lmax):
    assert mpm._wiscombe(x) == lmax


def _result(lmax=3):
    n_lm = lmax * (lmax + 2)
    a = RNG.normal(size=(2, n_lm)) + 1j * RNG.normal(size=(2, n_lm))
    b = RNG.normal(size=(2, n_lm)) + 1j * RNG.normal(size=(2, n_lm))
    return _multipoles(a, b, lmax=lmax, frequency=(F0, 1.1 * F0), e_inc=1.5 + 0.5j)


def test_u6_index():
    res = _result(lmax=7)
    seen = [res.index(l, m) for l in range(1, 8) for m in range(-l, l + 1)]
    assert seen == list(range(7 * 9))
    assert res.index(1, -1) == 0
    assert res.index(1, 1) == 2
    assert res.index(2, -2) == 3
    assert np.all(res.l[res.index(3, -3) : res.index(3, 3) + 1] == 3)
    assert res.alpha.shape == (2, 63)
    with pytest.raises(
        ValueError,
        match=_INDEX_MESSAGE,
    ):
        res.index(3, 4)
    for bad in ((0, 0), (8, 0), (2, -3)):
        with pytest.raises(ValueError, match="not a multipole"):
            res.index(*bad)
    for bad in ((True, 0), (1, False), (1.0, 0), (1, 0.0), ("1", 0)):
        with pytest.raises(TypeError, match="must be an int"):
            res.index(*bad)
    assert res.index(np.int64(2), np.int64(1)) == 6


def test_u6_l_and_m_are_read_only():
    res = _result()
    assert res.l.tolist() == [1] * 3 + [2] * 5 + [3] * 7
    assert res.m.tolist() == [-1, 0, 1, *range(-2, 3), *range(-3, 4)]
    for a in (res.l, res.m):
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.lmax = 4


def test_u6_pytree_and_pickle():
    res = _result()
    cur = mpm.CurrentMoments(
        frequency=(F0,),
        origin=(0.0, 0.0, 0.0),
        wavenumber=(2 * math.pi / 1e-6,),
        long_wavelength_dipole=jnp.ones((1, 3), complex),
        toroidal_dipole=2 * jnp.ones((1, 3), complex),
    )
    for obj, kids in (
        (res, ("incident_amplitude", "alpha", "beta")),
        (cur, ("long_wavelength_dipole", "toroidal_dipole")),
    ):
        assert len(jax.tree.leaves(obj)) == len(kids)
        same = jax.tree.map(lambda x: x, obj)
        assert type(same) is type(obj)
        for f in dataclasses.fields(obj):
            a, b = getattr(obj, f.name), getattr(same, f.name)
            assert np.array_equal(a, b)
            assert type(a) is type(b)
        back = pickle.loads(pickle.dumps(obj))  # noqa: S301
        for f in dataclasses.fields(obj):
            assert np.array_equal(getattr(obj, f.name), getattr(back, f.name))
        scaled = jax.tree.map(lambda x: 2 * x, obj)
        assert np.allclose(getattr(scaled, kids[-1]), 2 * getattr(obj, kids[-1]))
    # A scalar objective of a property differentiates through the pytree.
    g = jax.grad(lambda r: r.scattering.sum())(
        jax.tree.map(lambda x: x.real if np.iscomplexobj(x) else x, res)
    )
    assert np.all(np.isfinite(np.asarray(g.alpha)))


def test_u6_repr():
    res = _result()
    text = repr(res)
    assert text.startswith("Multipoles(lmax=3, origin=(0, 0, 0) m; ")
    assert text.count(" Hz: ED ") == 2
    assert "MQ" in text
    assert text.endswith("m²)")
    cur = mpm.CurrentMoments(
        frequency=(F0,),
        origin=(0.0, 1e-7, 0.0),
        wavenumber=(2 * math.pi / 1e-6,),
        long_wavelength_dipole=jnp.ones((1, 3), complex),
        toroidal_dipole=jnp.ones((1, 3), complex),
    )
    for part in ("|p0|", "|kT|", "1e-07"):
        assert part in repr(cur)
    seen = []

    def f(a):
        seen.append(repr(dataclasses.replace(res, alpha=a)))
        seen.append(repr(dataclasses.replace(cur, toroidal_dipole=a[:, :3])))
        return a.real.sum()

    jax.jit(f)(res.alpha)
    assert seen == ["Multipoles(traced)", "CurrentMoments(traced)"]


def test_construction_checks():
    res = _result()
    with pytest.raises(ValueError, match="lmax"):
        dataclasses.replace(res, lmax=1)
    with pytest.raises(ValueError, match="alpha must have shape"):
        dataclasses.replace(res, alpha=res.alpha[:, :-1])
