r"""Multipoles: the projection, the Cartesian maps, the helpers and the result types.

Everything here is synthetic: closed-form far fields of point dipoles and
random coefficients, no solved field.
"""

import dataclasses
import math
import pickle
import re

import free_space_cases as fsc
import jax
import jax.numpy as jnp
import multipole_cases as mc
import numpy as np
import pml_cases as pc
import pytest
import scipy.constants as const
from strip2d import write_strip

import gfem
from gfem import _incident
from gfem import _multipoles as mpm
from gfem._sources import polarization_vector, propagation_basis

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


def test_u6_top_level_identity():
    from gfem import measurements

    for name in ("Multipoles", "CurrentMoments", "multipoles"):
        assert getattr(gfem, name) is getattr(measurements, name)
        assert name in gfem.__all__
        assert name in measurements.__all__
    assert gfem.multipoles is mpm.multipoles
    assert gfem.Multipoles is mpm.Multipoles


# --- the entry on the coarse cube -------------------------------------------------
#
# A one-cell dielectric cube centred on the origin, order 2, one module solve with a
# plane wave, the same wave as an `IncidentField` and a dipole column.

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
AIR = gfem.Material()
WAVE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")
DIPOLE_AT = (-0.245e-6, 0.035e-6, 0.495e-6)
UM = 1e-6
SHIFT = (0.5, 0.5, 0.5)


def _te_function(points, frequency):
    """``WAVE`` as an incident-field function."""
    khat, _, _ = propagation_basis(WAVE.theta, WAVE.phi)
    k = 2 * math.pi * frequency / C0
    e = polarization_vector(WAVE.theta, WAVE.phi, "TE")
    return np.exp(-1j * k * points @ khat)[:, None] * e[None, :]


def _solve(model, *sources, **kw):
    problem = gfem.Driven(F0, sources=sources, keep_fields=True, **kw)
    return gfem.solve(model, problem, options=OPTIONS)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mp")
    path = mc.cube(tmp / "c.msh", centred=True)
    model = mc.portless_scatterer_model(path)
    sources = (
        WAVE,
        gfem.IncidentField("f", _te_function),
        gfem.ElectricDipole("d", position=DIPOLE_AT, moment=(0, 0, 1e-30)),
    )
    return model, _solve(model, *sources)


@pytest.fixture(scope="module")
def reference(run):
    model, result = run
    return gfem.multipoles(result, "te", model=model)


def test_u7_checks_are_in_order_and_refusals_are_named(run, tmp_path_factory):
    model, result = run
    call = lambda **kw: gfem.multipoles(result, "te", model=model, **kw)  # noqa: E731
    # 1 / the 2D refusal comes first, even before the types
    flat = gfem.Model(
        gfem.Mesh.from_gmsh(
            write_strip(tmp_path_factory.mktemp("strip") / "s.msh"), length_unit="um"
        ),
        materials={k: gfem.Material(eps_r=2.0) for k in "abc"},
        boundaries=(gfem.PEC("outer"),),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"gfem\.multipoles is not avail.*2D mesh.*3D mesh",
    ):
        gfem.multipoles("not a result", "te", model=flat)
    # 2 / types
    with pytest.raises(
        TypeError, match=r"multipoles needs a gfem\.DrivenResult, got str"
    ):
        gfem.multipoles("r", "te", model=model)
    with pytest.raises(
        TypeError, match=r"multipoles needs model= a gfem\.Model, got int"
    ):
        gfem.multipoles(result, "te", model=1)
    # 3 / traced origin
    with pytest.raises(
        gfem.NonDifferentiableError, match=r"multipoles' origin must be fixed"
    ):
        jax.grad(lambda o: call(origin=o).scattering.sum())(jnp.zeros(3))
    # 4 / origin form
    for bad in ((0.0, 0.0), (0.0, np.nan, 0.0), (0.0, np.inf, 0.0), "abc", None, 1.0):
        with pytest.raises(ValueError, match=r"origin must be three finite numbers"):
            call(origin=bad)
    # 5-6 / lmax
    for bad in (2.0, True, "3", np.float64(3)):
        with pytest.raises(TypeError, match=r"lmax must be an int"):
            call(lmax=bad)
    with pytest.raises(ValueError, match=r"lmax=1 is below 2"):
        call(lmax=1)
    # 7 / unknown column
    with pytest.raises(KeyError, match=r"'nope' is not a column of the result"):
        gfem.multipoles(result, "nope", model=model)
    # 8 / a dipole column
    with pytest.raises(
        ValueError, match=r"'d' is a ElectricDipole column.*gfem\.far_field"
    ):
        gfem.multipoles(result, "d", model=model)
    # 10-12 / the background
    with pytest.raises(ValueError, match=r"needs the model's background"):
        gfem.multipoles(result, "te", model=model.replace(background=None))
    lossy = gfem.Material(tan_delta=0.01)
    with pytest.raises(
        NotImplementedError,
        match=r"multipoles in a lossy background \(.*\) is not computed",
    ):
        gfem.multipoles(result, "te", model=model.replace(background=lossy))
    stack = gfem.Background([AIR, gfem.Material(eps_r=2.0)], interfaces=[1e-6])
    with pytest.raises(NotImplementedError, match=r"over a layered background"):
        gfem.multipoles(result, "te", model=model.replace(background=stack))
    ground = gfem.Background(AIR, ground=-1e-6)
    with pytest.raises(NotImplementedError, match=r"over a layered background"):
        gfem.multipoles(result, "te", model=model.replace(background=ground))
    # 14 / intensity
    with pytest.raises(
        ValueError, match=r"multipoles of the IncidentField 'f' needs intensity="
    ):
        gfem.multipoles(result, "f", model=model)
    with pytest.raises(
        ValueError, match=r"multipoles' intensity must be a positive finite"
    ):
        gfem.multipoles(result, "f", model=model, intensity=-1.0)
    with pytest.raises(
        ValueError, match=r"multipoles takes intensity= for an IncidentField only"
    ):
        call(intensity=1.0)
    with pytest.raises(
        gfem.NonDifferentiableError, match=r"multipoles' intensity is static"
    ):
        jax.grad(
            lambda i: gfem.multipoles(
                result, "f", model=model, intensity=i
            ).scattering.sum()
        )(1.0)


def test_u7_a_column_without_kept_fields_or_incident_part(run):
    model, result = run
    problem = gfem.Driven(F0, sources=WAVE)
    unkept = gfem.solve(model, problem, options=OPTIONS)
    with pytest.raises(ValueError, match=r"multipoles needs the kept fields of 'te'"):
        gfem.multipoles(unkept, "te", model=model)
    fields = tuple(dataclasses.replace(f, _incident=None) for f in result.e_fields)
    stale = dataclasses.replace(result, e_fields=fields)
    with pytest.raises(ValueError, match=r"'te' carries no incident part"):
        gfem.multipoles(stale, "te", model=model)


def _walled(tmp_path_factory, sides, surface=None, boundaries=()):
    path = pc.box(
        tmp_path_factory.mktemp("wall") / "w.msh",
        sides,
        thickness=0.25,
        cells=4,
        surface=surface,
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    return gfem.Model(
        mesh,
        materials={"air": AIR, "pml": AIR},
        boundaries=boundaries,
        pml=gfem.PML("pml"),
        background=AIR,
    )


def test_u7_walls_are_refused_before_the_far_field_layer(run, tmp_path_factory):
    _, result = run
    plane = lambda c: "ground" if abs(c[2]) < 1e-9 else ""  # noqa: E731
    cases = {
        "slab": _walled(tmp_path_factory, ("-z", "+z")),
        "ground plane": _walled(
            tmp_path_factory,
            ("-x", "+x", "-y", "+y", "+z"),
            plane,
            (gfem.PEC("ground"),),
        ),
    }
    for wall_model in cases.values():
        with pytest.raises(
            NotImplementedError,
            match=r"ends on the (PEC|PMC) wall '.*' beside its physical region",
        ):
            gfem.multipoles(result, "te", model=wall_model)
    # the far field itself would say something else about the slab
    with pytest.raises(NotImplementedError, match=r"mirror half model or slab model"):
        gfem.multipoles(result, "te", model=cases["slab"])


def test_u7_a_scatterer_in_the_truncation_layer_or_at_a_port(tmp_path_factory):
    guide = _port_model(tmp_path_factory)
    solved = _solve(guide, gfem.IncidentField("w", _away_from_the_port), excite=())
    with pytest.raises(ValueError, match=r"port 'o1''s surface 'feed'"):
        gfem.multipoles(solved, "w", model=guide, intensity=1.0)
    grown = _touching_model(tmp_path_factory)
    solved = _solve(grown, gfem.IncidentField("w", _windowed))
    with pytest.raises(
        ValueError,
        match=r"multipoles: the scatterer region 'scat' reaches the truncation "
        r"layer.*no finite multipole expansion",
    ):
        gfem.multipoles(solved, "w", model=grown, intensity=1.0)


def _port_surface(c):
    """A wave port patch on the x = 0 wall; every other side wall absorbs."""
    if abs(c[0]) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75:
        return "feed"
    if 0 < c[2] < 1 and min(abs(c[0]), abs(c[0] - 1), abs(c[1]), abs(c[1] - 1)) < 1e-9:
        return "side"
    return ""


def _port_model(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("port") / "p.msh",
        ("-z", "+z"),
        cells=8,
        surface=_port_surface,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"air": AIR, "pml": AIR},
        boundaries=[gfem.Absorbing("side")],
        ports=[gfem.WavePort("o1", "feed")],
        pml=gfem.PML("pml"),
        background=AIR,
    )


def _away_from_the_port(points, frequency):
    """Along +z, E along x, zero up to x = 0.3 um: none on the port face."""
    k = 2 * math.pi * frequency / C0
    x = points[:, 0] * 1e6
    window = np.where(x > 0.3, np.sin(np.clip(x - 0.3, 0, 0.1) * 5 * math.pi) ** 2, 0)
    return (np.exp(-1j * k * points[:, 2]) * window)[:, None] * np.array([1, 0, 0])


def _touching_volume(c):
    inside = bool(((c > 0) & (c < 1)).all())
    if c[0] > 0.75 and inside:
        return "scat"
    return "air" if inside else "pml"


def _touching_model(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("touch") / "t.msh",
        pc.SIDES,
        thickness=0.25,
        cells=4,
        volume=_touching_volume,
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    return gfem.Model(
        mesh,
        materials={"air": AIR, "pml": AIR, "scat": gfem.Material(eps_r=4.0)},
        pml=gfem.PML("pml"),
        background=AIR,
    )


def _windowed(points, frequency):
    """A plane wave cut off before x = 0.7 um: zero where the cube meets the layer."""
    k = 2 * math.pi * frequency / C0
    x = points[:, 0] * 1e6
    window = np.where(x < 0.7, np.cos(np.clip(x - 0.6, 0, 0.1) * 5 * math.pi) ** 2, 0)
    return (np.exp(-1j * k * points[:, 2]) * window)[:, None] * np.array([0, 1, 0])


def test_u8_an_empty_scatterer_scatters_round_off(tmp_path_factory, run):
    path = mc.cube(tmp_path_factory.mktemp("empty") / "e.msh", centred=True)
    model = mc.portless_scatterer_model(path, eps=1.0)
    mp = gfem.multipoles(_solve(model, WAVE), "te", model=model)
    assert mp.lmax == 2
    size = np.abs(np.asarray(run[1].e_field(F0, "te").coefficients)).max()
    assert size > 0
    for c in (mp.alpha, mp.beta):
        assert np.abs(np.asarray(c)).max() <= 1e-10, np.abs(np.asarray(c)).max()


def test_u9_a_design_equal_to_the_background_scatters_nothing(tmp_path_factory):
    path = mc.cube(tmp_path_factory.mktemp("field") / "d.msh", centred=True)
    model = mc.portless_scatterer_model(path, eps=1.0, field=True)
    mp = gfem.multipoles(_solve(model, WAVE), "te", model=model)
    # a region chosen per region, not by value: it still counts for the default
    assert mp.lmax == 8
    for c in (mp.alpha, mp.beta):
        assert np.abs(np.asarray(c)).max() <= 1e-10


def test_u10_an_incident_field_gives_the_plane_waves_coefficients(run, reference):
    model, result = run
    eta = _incident.impedance(model.background)
    mp = gfem.multipoles(result, "f", model=model, intensity=1 / (2 * eta))
    assert np.allclose(mp.alpha, reference.alpha, rtol=0, atol=1e-12)
    assert np.allclose(mp.beta, reference.beta, rtol=0, atol=1e-12)
    assert np.allclose(mp.incident_amplitude, 1.0, rtol=1e-12)
    assert np.allclose(reference.incident_amplitude, 1.0, rtol=1e-12)
    assert np.allclose(mp.intensity, 1 / (2 * eta), rtol=1e-12)


def test_u11_a_translated_scatterer_has_the_translated_multipoles(
    reference, tmp_path_factory
):
    path = mc.cube(tmp_path_factory.mktemp("shift") / "s.msh", centre=SHIFT)
    model = mc.portless_scatterer_model(path)
    result = _solve(model, WAVE)
    centre = np.array(SHIFT) * UM
    khat, _, _ = propagation_basis(WAVE.theta, WAVE.phi)
    k = reference.wavenumber[0]
    moved = gfem.multipoles(result, "te", model=model, origin=tuple(centre))
    phase = np.exp(-1j * k * khat @ centre)
    scale = np.abs(np.asarray(reference.alpha)).max()
    assert moved.lmax == reference.lmax
    for got, want in ((moved.alpha, reference.alpha), (moved.beta, reference.beta)):
        assert np.abs(np.asarray(got) - phase * np.asarray(want)).max() <= 1e-8 * scale
    # the partial cross sections depend on the origin; the sum does not
    other = gfem.multipoles(result, "te", model=model)
    assert other.lmax > moved.lmax
    sigma = lambda m: np.asarray(m.electric_scattering)[0, 0]  # noqa: E731
    assert abs(sigma(other) - sigma(moved)) > 1e-3 * sigma(moved)
    total = lambda m: float(np.asarray(m.scattering)[0])  # noqa: E731
    assert abs(total(other) - total(moved)) <= 1e-2 * total(moved)


def test_u12_lmax_and_its_bounds(run, reference):
    model, result = run
    assert reference.lmax == 8
    short = gfem.multipoles(result, "te", model=model, lmax=3)
    assert short.lmax == 3
    scale = np.abs(np.asarray(reference.alpha)).max()
    for a, b in ((short.alpha, reference.alpha), (short.beta, reference.beta)):
        assert np.abs(np.asarray(a) - np.asarray(b)[:, :15]).max() <= 1e-8 * scale
    with pytest.raises(ValueError, match=r"lmax=1 is below 2"):
        gfem.multipoles(result, "te", model=model, lmax=1)
    with pytest.raises(
        ValueError, match=r"lmax=1000 is above (\d+), the highest degree"
    ) as info:
        gfem.multipoles(result, "te", model=model, lmax=1000)
    cap = int(re.search(r"is above (\d+)", str(info.value)).group(1))
    assert gfem.multipoles(result, "te", model=model, lmax=cap).lmax == cap
    with pytest.raises(ValueError, match=rf"lmax={cap + 1} is above {cap}"):
        gfem.multipoles(result, "te", model=model, lmax=cap + 1)
    assert "kR = " in str(info.value)


def test_sphere_case_puts_the_volume_equivalent_size_parameter_on_the_case(tmp_path):
    case = mc.CASES["C"]
    mesh, a_vol, frequency = mc.sphere_case(
        tmp_path / "s.msh", case, maxh=0.4, sphere_maxh=0.2
    )
    k = 2 * math.pi * frequency / C0
    assert k * a_vol * UM == pytest.approx(case.x, rel=1e-12)
    # the mesh keywords moved the sphere: radius x / 2 pi = 0.1248 um, faceted
    assert 0.8 * case.x / (2 * math.pi) < a_vol < case.x / (2 * math.pi)
    nodes = np.asarray(mesh.nodes)
    assert nodes.max() == pytest.approx((case.x / (2 * math.pi) + 0.4 + 0.35) * UM)
