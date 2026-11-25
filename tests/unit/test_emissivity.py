"""Emissivity by Kirchhoff's law: the hemisphere rule, its messages, a PEC-backed cell.

The rule and the messages need no solve. The PEC-backed lossy F1 (coarse mesh,
p = 2, direct solver) is swept over a normal and a conical incidence against the
Airy form of the PEC-terminated stack.
"""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
from scipy.special import roots_legendre

import gfem
from gfem import diffraction as dif

OPTS = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
TOP = "z_max"
F0 = pc.F1_FREQUENCY
DEG = math.pi / 180.0
EPS_SLAB = pc.F1_EPS_SLAB * (1 - 1j * pc.F1_TAN_DELTA)


def _fake_sweep(theta, phi):
    """A sweep with labels only: enough for the checks that precede any read."""
    n = len(theta)
    return dif.AngleSweepResult((None,) * n, tuple(theta), tuple(phi), TOP, (F0,))


# --- the rule -------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_hemisphere_is_exact_on_mu_powers(n):
    rule = dif.hemisphere(n, 1)
    mu = np.cos(rule.theta)
    for k in range(2 * n):
        exact = 2.0 / (k + 2)
        assert abs(np.sum(rule.weight * mu**k) - exact) <= 1e-14
    assert abs(rule.weight.sum() - 1) <= 1e-14
    assert np.all(np.diff(rule.theta) > 0)
    assert rule.theta[-1] < math.pi / 2
    assert not rule.mirror
    assert not rule.theta.flags.writeable
    assert not rule.weight.flags.writeable


def test_folding_keeps_the_symmetric_azimuths():
    folded = dif.hemisphere(5, 16, rotation=4, mirror=True)
    full = dif.hemisphere(5, 16)
    assert len(folded.theta) == 5 * 3
    assert len(np.unique(np.round(folded.phi, 12))) == 3
    assert folded.mirror

    def c4v(t, p):
        return (
            0.3
            + 0.2 * np.cos(t) ** 3
            + 0.1 * np.cos(4 * p) * np.sin(t) ** 2
            + 0.05 * np.cos(8 * p) * np.sin(t) ** 4
        )

    a = np.sum(folded.weight * c4v(folded.theta, folded.phi))
    b = np.sum(full.weight * c4v(full.theta, full.phi))
    assert abs(a - b) <= 1e-14
    assert abs(folded.weight.sum() - 1) <= 1e-14


def test_the_mirror_fold_assumes_the_plane_containing_e1():
    # e = 1 + 0.5 cos(phi) is mirrored about the plane containing e1: the fold
    # is exact. e = 1 + 0.5 sin(phi) is mirrored only about the plane
    # containing e2 (true mean 1): the fold is wrong, so the documented
    # remedy is mirror=False.
    full = dif.hemisphere(3, 16)
    folded = dif.hemisphere(3, 16, mirror=True)
    for rule, tol in ((full, 1e-14), (folded, 1e-14)):
        mean = np.sum(rule.weight * (1 + 0.5 * np.cos(rule.phi)))
        assert abs(mean - 1.0) <= tol
    assert abs(np.sum(full.weight * (1 + 0.5 * np.sin(full.phi))) - 1.0) <= 1e-14
    wrong = np.sum(folded.weight * (1 + 0.5 * np.sin(folded.phi)))
    assert abs(wrong - 1.0) > 0.1
    # both mirror planes: rotation=2 with mirror=True folds the e2 mirror too
    both = dif.hemisphere(3, 16, rotation=2, mirror=True)
    mean = np.sum(both.weight * (1 + 0.5 * np.sin(both.phi) ** 2))
    ref = np.sum(full.weight * (1 + 0.5 * np.sin(full.phi) ** 2))
    assert abs(mean - ref) <= 1e-14


# --- the messages -------------------------------------------------------------


def test_hemisphere_texts():
    with pytest.raises(
        TypeError, match=r"^hemisphere: n_theta must be an integer, got 2.5\.$"
    ):
        dif.hemisphere(2.5, 1)
    with pytest.raises(TypeError, match=r"n_phi must be an integer, got True"):
        dif.hemisphere(2, True)
    with pytest.raises(
        TypeError, match=r"^hemisphere: mirror must be True or False, got 1\.$"
    ):
        dif.hemisphere(2, 1, mirror=1)
    with pytest.raises(
        ValueError, match=r"^hemisphere: rotation must be >= 1, got 0\.$"
    ):
        dif.hemisphere(2, 4, rotation=0)
    with pytest.raises(
        ValueError, match=r"^hemisphere: n_theta must be >= 1, got 0\.$"
    ):
        dif.hemisphere(0, 4)
    with pytest.raises(
        ValueError,
        match=r"n_phi=7 is not a multiple of rotation=4, so the azimuths are not "
        r"invariant under a rotation by 2\*pi/4\. Use n_phi = 8\.$",
    ):
        dif.hemisphere(2, 7, rotation=4)


def test_texts_of_emissivity(f1_model):
    with pytest.raises(
        TypeError,
        match=r"^emissivity needs a gfem.diffraction.AngleSweepResult .* got list: "
        r"the emission directions are the sweep's incidences reversed\.",
    ):
        dif.emissivity([], model=f1_model)
    sweep = _fake_sweep([0.0], [0.0])
    with pytest.raises(
        ValueError,
        match=r"^emissivity: polarization must be 'TE', 'TM', 'RCP', 'LCP' or "
        r"'unpolarized', got 'X'\.$",
    ):
        dif.emissivity(sweep, model=f1_model, polarization="X")
    with pytest.raises(
        ValueError, match=r"is a Jones pair; emissivity is defined here"
    ):
        dif.emissivity(sweep, model=f1_model, polarization=(1, 0))
    with pytest.raises(
        TypeError,
        match=r"^hemispherical_emissivity needs rule= a gfem.diffraction."
        r"HemisphereRule .* got str\.$",
    ):
        dif.hemispherical_emissivity(sweep, "rule", model=f1_model)
    with pytest.raises(TypeError, match=r"^hemispherical_emissivity needs a gfem"):
        dif.hemispherical_emissivity(None, dif.hemisphere(1, 1), model=f1_model)


def test_texts_of_the_hemispherical_emissivity(f1_model):
    rule = dif.hemisphere(2, 1)
    with pytest.raises(
        ValueError,
        match=r"incidences are not the rule's nodes: the sweep has 1 incidences, "
        r"the rule 2 nodes\. Sweep the rule's nodes: "
        r"angle_sweep\(\.\.\., theta=rule.theta, phi=rule.phi\)\.$",
    ):
        dif.hemispherical_emissivity(_fake_sweep([0.0], [0.0]), rule, model=f1_model)
    bad = _fake_sweep([rule.theta[0], rule.theta[1] + 1e-9], [0.0, 0.0])
    with pytest.raises(
        ValueError,
        match=r"incidence 1 is \(theta, phi\) = \(.*\) rad, node 1 is \(.*\) rad",
    ):
        dif.hemispherical_emissivity(bad, rule, model=f1_model)
    folded = dif.hemisphere(2, 4, rotation=4, mirror=True)
    with pytest.raises(
        ValueError,
        match=r"polarization='RCP' with a rule folded by mirror=True: a mirror turns "
        r"RCP into LCP, so the folded nodes do not carry RCP\. Build the rule with "
        r"mirror=False, or use 'TE', 'TM' or 'unpolarized'\.$",
    ):
        dif.hemispherical_emissivity(
            _fake_sweep(folded.theta, folded.phi),
            folded,
            model=f1_model,
            polarization="RCP",
        )


# --- the rule on an analytic emitter --------------------------------------


def _airy_absorptance(mu, pol):
    k0 = 2 * math.pi * F0 / pc.C0
    out = []
    for m in mu:
        kt = k0 * math.sqrt(1 - m * m)
        layers = [
            (1.0, pc.F1_D * pc.UM),
            (EPS_SLAB, pc.F1_H * pc.UM),
            (pc.F1_EPS_SUB, pc.F1_D * pc.UM),
        ]
        out.append(_a(kt, layers, pol))
    return np.asarray(out)


def _a(kt, layers, pol):
    r, t = pc.airy(F0, kt, layers, pol)
    return 1 - abs(r) ** 2 - abs(t) ** 2


@pytest.mark.parametrize("pol", ["TE", "TM", "unpolarized"])
def test_six_nodes_integrate_the_airy_emissivity(pol):
    def e(mu):
        if pol == "unpolarized":
            return 0.5 * (_airy_absorptance(mu, "TE") + _airy_absorptance(mu, "TM"))
        return _airy_absorptance(mu, pol)

    x, w = roots_legendre(4000)
    mu = (x + 1) / 2
    reference = np.sum(w / 2 * 2 * mu * e(mu))
    rule = dif.hemisphere(6, 1)
    got = np.sum(rule.weight * e(np.cos(rule.theta)))
    assert abs(got - reference) <= 2e-5


# --- PEC-backed lossy F1 ---------------------------------------------------


@pytest.fixture(scope="module")
def f1_model(tmp_path_factory):
    return pc.f1(pc.f1_mesh(tmp_path_factory.mktemp("emis_f1")))


@pytest.fixture(scope="module")
def pec_cell(tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("emis_pec"))
    return pc.pec_backed_lossy(mesh)


@pytest.fixture(scope="module")
def pec_sweep(pec_cell):
    return dif.angle_sweep(
        pec_cell,
        [F0],
        surface=TOP,
        theta=[0.0, 40 * DEG],
        phi=[0.0, 30 * DEG],
        options=OPTS,
    )


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_pec_backed_emissivity_is_one_minus_r_squared(pec_cell, pec_sweep, pol):
    emis = dif.emissivity(pec_sweep, model=pec_cell, polarization=pol)
    bal = dif.power_balance(pec_sweep, TOP, polarization=pol, model=pec_cell)
    assert np.asarray(bal.transmittance).max() == 0
    assert np.asarray(bal.coupled).max() == 0
    k0 = 2 * math.pi * F0 / pc.C0
    for i, theta in enumerate(pec_sweep.theta):
        r = pc.airy_pec(k0 * math.sin(theta), pol, EPS_SLAB, pc.F1_EPS_SUB)
        assert abs(float(emis.values[i, 0]) - (1 - abs(r) ** 2)) <= 5e-3
    assert emis.theta == pec_sweep.theta
    assert emis.phi[0] == pytest.approx(math.pi)
    assert emis.phi[1] == pytest.approx(210 * DEG - 2 * math.pi)
    assert emis.polarization == pol
    assert emis.surface == TOP
    assert emis.values.shape == (2, 1)


def test_an_undeclared_polarization_names_emissivity(pec_cell):
    model = gfem.Model(
        pec_cell.mesh,
        materials=pec_cell.materials,
        boundaries=pec_cell.boundaries,
        ports=[gfem.FloquetPort("R_TE", TOP, polarization="TE")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnrequestedOrderWarning)
        sweep = dif.angle_sweep(model, [F0], surface=TOP, theta=[0.0], options=OPTS)
    with pytest.raises(ValueError, match=r"^emissivity: Diffraction order \(0, 0\) TM"):
        dif.emissivity(sweep, model=model, polarization="TE")


def test_the_hemispherical_sum_is_the_weighted_rows(pec_cell):
    rule = dif.hemisphere(1, 1)
    sweep = dif.angle_sweep(
        pec_cell, [F0], surface=TOP, theta=rule.theta, phi=rule.phi, options=OPTS
    )
    eh = dif.hemispherical_emissivity(sweep, rule, model=pec_cell)
    e = dif.emissivity(sweep, model=pec_cell)
    assert float(jnp.abs(eh - jnp.asarray(rule.weight) @ e.values).max()) == 0
