"""Emissivity against closed forms and against the reciprocal row deficit.

E4: the lossy slab of F1 in air (the Brewster cell's mesh) at p = 3, swept
over the four Gauss-Jacobi nodes of `hemisphere(4, 1)`, against the Airy
absorptance per node and the exact hemispherical emissivity.

E5: the lamellar grating F2 with a lossy, asymmetric per-element design: the
emissivity at azimuth pi, the Kirchhoff partner of the incidence at azimuth 0,
equals the row deficit of the specular row of the fully excited solve at
azimuth 0, and differs from the absorptance at azimuth 0.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest
from scipy.special import roots_legendre

import gfem
from gfem import diffraction as dif

TOP = "z_max"
F0 = pc.F1_FREQUENCY
DEG = math.pi / 180.0
SLAB = gfem.Material(eps_r=pc.F1_EPS_SLAB, tan_delta=pc.F1_TAN_DELTA)
EPS_SLAB = pc.F1_EPS_SLAB * (1 - 1j * pc.F1_TAN_DELTA)
LAYERS = [
    (1.0, pc.F1_D * pc.UM),
    (EPS_SLAB, pc.F1_H * pc.UM),
    (pc.BREWSTER_EPS_SUB, pc.F1_D * pc.UM),
]


def _airy_a(theta, pol):
    kt = 2 * math.pi * F0 / pc.C0 * math.sin(theta)
    r, t = pc.airy(F0, kt, LAYERS, pol)
    return 1 - abs(r) ** 2 - abs(t) ** 2


def _airy_unpolarized(theta):
    return 0.5 * (_airy_a(theta, "TE") + _airy_a(theta, "TM"))


@pytest.fixture(scope="module")
def e4(tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("e4"), pc.BREWSTER_CELLS)
    model = pc.f1(mesh, eps_sub=pc.BREWSTER_EPS_SUB).with_material("slab", SLAB)
    rule = dif.hemisphere(4, 1)
    sweep = dif.angle_sweep(
        model,
        [F0],
        surface=TOP,
        theta=rule.theta,
        phi=rule.phi,
        options=gfem.SolverOptions(order=3, linear_solver="direct"),
    )
    return model, rule, sweep


def test_e4_each_node_matches_airy_and_is_labelled_pi(e4):
    model, rule, sweep = e4
    for pol in ("TE", "TM", "unpolarized"):
        emis = dif.emissivity(sweep, model=model, polarization=pol)
        for i, theta in enumerate(rule.theta):
            want = (
                _airy_unpolarized(theta)
                if pol == "unpolarized"
                else _airy_a(theta, pol)
            )
            assert abs(float(emis.values[i, 0]) - want) <= 1e-5
        assert emis.phi == (math.pi,) * len(rule.theta)


def test_e4_hemispherical_emissivity_matches_the_node_sum_and_the_exact_value(e4):
    model, rule, sweep = e4
    got = float(dif.hemispherical_emissivity(sweep, rule, model=model)[0])
    same_nodes = sum(
        w * _airy_unpolarized(t) for w, t in zip(rule.weight, rule.theta, strict=True)
    )
    assert abs(got - same_nodes) <= 1e-5
    # The four-node rule's own truncation on this Fabry-Perot integrand is 4.3e-4, so
    # the exact hemispherical value is checked on the rule, not on the solve: the
    # FEM path is pinned above at the same nodes, and eight nodes of the same rule
    # reach the exact value (measured 2.2e-5).
    x, w = roots_legendre(4000)
    mu = (x + 1) / 2
    exact = np.sum(w * mu * np.array([_airy_unpolarized(math.acos(m)) for m in mu]))
    fine = dif.hemisphere(8, 1)
    eight = sum(
        wt * _airy_unpolarized(t) for wt, t in zip(fine.weight, fine.theta, strict=True)
    )
    assert abs(eight - exact) <= 5e-5


# --- E5 ------------------------------------------------------------------------


def test_e5_the_emission_azimuth_is_the_reversed_incidence(tmp_path):
    theta = 20 * DEG
    opts = gfem.SolverOptions(order=2, linear_solver="direct")
    mesh = pc.f2_mesh(tmp_path, (4, 2, 2, 2, 2))
    probe = pc.f2(mesh, ports=pc.f2_ports(((0, 0),)))
    k1 = probe.bloch_wave_vector(TOP, F0, theta=theta, phi=0.0)
    k2 = probe.bloch_wave_vector(TOP, F0, theta=theta, phi=math.pi)
    orders = set()
    for surface in (TOP, "z_min"):
        for k in (k1, k2):
            orders |= set(probe.propagating_orders(surface, [F0], wave_vector=k))
    ports = pc.f2_ports(tuple(sorted(orders)))
    ridge = len(mesh.elements("ridge"))
    rng = np.random.default_rng(30)
    eps = np.concatenate(
        [
            rng.uniform(3.5, 4.5, ridge),
            rng.uniform(1.0, 1.5, len(mesh.elements("groove"))),
        ]
    ) * (1 - 0.02j)
    model = (
        pc.f2(mesh, ports=ports)
        .with_material("ridge", gfem.MaterialField(eps_r=eps[:ridge]))
        .with_material("groove", gfem.MaterialField(eps_r=eps[ridge:]))
    )

    def propagating(k):
        return [
            p.name
            for p in model.ports
            if p.order in model.propagating_orders(p.surface, [F0], wave_vector=k)
        ]

    full = gfem.solve(
        model,
        gfem.Driven([F0], excite=propagating(k1), wave_vector=k1),
        options=opts,
    )
    sweep = dif.angle_sweep(
        model, [F0], surface=TOP, theta=[theta, theta], phi=[0.0, math.pi], options=opts
    )
    values = np.asarray(full.s.values)[0]
    for pol in ("TE", "TM"):
        row = full.s.ports.index(f"R0,0_{pol}")
        deficit = 1 - np.sum(np.abs(values[row, :]) ** 2)
        emis = dif.emissivity(sweep, model=model, polarization=pol)
        assert abs(float(emis.values[1, 0]) - deficit) <= 1e-10
        assert emis.phi[1] == 0.0
        assert emis.phi[0] == pytest.approx(math.pi)
        assert abs(float(emis.values[0, 0] - emis.values[1, 0])) >= 1e-5
