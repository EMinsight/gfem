r"""Resonant modes of a guide closed by a wave port.

A rectangular WR-90 guide with layers that fill its cross-section, closed
by a short at one end and a wave port at the other: each guide mode is an
exact transmission line along the guide, so its complex resonances solve a
scalar equation (the reference values below, residual 1e-13). The port's
TE10 mode leaves without reflection, so the resonance's Q is its radiation
through the port.

The port enters the eigen operator as a Robin term with
:math:`\theta = \gamma_0(k)` and a modal correction
:math:`U C(k) U^\mathsf{T}` (zero for a lone TE mode, not for the hybrid
guide's), both from the port mode sweep's reduced model at the complex
wavenumber.
"""

import dataclasses
import math
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._eigen.operator import split
from gfem._engine.driven import driven_operator, operator_terms
from gfem._engine.eigen import eigen_ports
from gfem.errors import AntiBoundRootWarning

DATA = Path(__file__).parent / "data"
C0 = 299792458.0
A_MM, B_MM = 22.86, 10.16
ORDER_3 = gfem.SolverOptions(order=3, linear_solver="direct")

# (name, eps_r, thickness mm) from the short to the port
CASE_A = (("dielectric", 4.0, 10.0), ("air", 1.0, 5.0))
CASE_A3 = (("dielectric", 4.0, 30.0), ("air", 1.0, 5.0))
# exact roots, GHz (outgoing branch of the port everywhere)
ROOT_A = 11.665906948910 + 1.062461078093j
ROOT_A3 = 7.033666935873 + 0.147969608374j
ANTI_BOUND_A3 = 4.619035589541


def shorted_guide(path, layers, *, nx=6, nz_per_mm=0.5, ny=2):
    """WR-90 with ``layers`` from a short at z = 0 to the port at the far end."""
    tops = np.cumsum([t for _, _, t in layers])
    zs = [np.zeros(1)]
    for (_, _, t), z1 in zip(layers, tops, strict=True):
        n = max(1, round(t * nz_per_mm))
        zs.append(z1 - t + np.linspace(0, t, n + 1)[1:])
    edges = (
        np.linspace(0, A_MM, nx + 1),
        np.linspace(0, B_MM, ny + 1),
        np.concatenate(zs),
    )
    length = tops[-1]

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        return "walls"

    sm.write_msh22(path, sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def guide_model(mesh, layers):
    return gfem.Model(
        mesh,
        materials={name: gfem.Material(eps_r=eps) for name, eps, _ in layers},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )


def ghz(omega):
    return np.asarray(omega) / (2 * math.pi) / 1e9


@pytest.fixture(scope="module")
def case_a(tmp_path_factory):
    mesh = shorted_guide(tmp_path_factory.mktemp("a") / "a.msh", CASE_A)
    return guide_model(mesh, CASE_A)


@pytest.fixture(scope="module")
def case_a3(tmp_path_factory):
    mesh = shorted_guide(tmp_path_factory.mktemp("a3") / "a3.msh", CASE_A3)
    return guide_model(mesh, CASE_A3)


# --- the solve ------------------------------------------------------------------


def test_lone_wave_port_resonance_matches_the_exact_root(case_a):
    # the two modes nearer the target than the port's resonance are of other
    # guide families (TE30/TE21-like), which the TE10 port only partly absorbs
    result = gfem.solve(
        case_a, gfem.Eigenmode(target=11.67e9, n_modes=3), options=ORDER_3
    )
    f = ghz(result.omega)
    j = int(np.argmin(np.abs(f - ROOT_A)))
    assert abs(f[j] - ROOT_A) / abs(ROOT_A) <= 1e-4
    assert result.error[j] <= 1e-12


def test_anti_bound_root_below_cutoff_is_dropped_with_a_warning(case_a3):
    # on the outgoing branch the guide has a real root at 4.619 GHz, below
    # the air guide's cutoff (6.557 GHz): its field grows into the guide, so
    # it is not a resonance; the search drops it and carries on
    with pytest.warns(AntiBoundRootWarning, match=r"4\.619"):
        result = gfem.solve(
            case_a3, gfem.Eigenmode(target=6.7e9, n_modes=8), options=ORDER_3
        )
    f = ghz(result.omega)
    assert len(f) == 8
    assert np.min(np.abs(f - ANTI_BOUND_A3)) > 1e-2
    assert np.min(np.abs(f - ROOT_A3)) / abs(ROOT_A3) <= 1e-4


# --- the operator ---------------------------------------------------------------


@pytest.fixture(scope="module")
def hybrid():
    # WR-90 with an eps_r = 6 block: a hybrid port mode (E_n != 0), whose
    # modal correction is not zero; the far end is a short
    mesh = gfem.Mesh.from_gmsh(DATA / "hybrid_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"block": gfem.Material(eps_r=6.0), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls"), gfem.PEC("port2")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    target = 12e9
    rule = eigen_rule.EigenmodeRule(
        model, gfem.Eigenmode(target=target), gfem.SolverOptions(order=2)
    )
    op = driven_operator(model, 2)
    ports = eigen_ports(model, op, target=target, tolerance=1e-6)
    return model, rule, op, ports


def free_vector(eop, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eop.free)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return np.where(eop.free, x, 0)


def test_port_piece_is_nonpolynomial_and_never_an_operator_term(hybrid):
    model, _, op, (port,) = hybrid
    piece = port.piece
    assert piece.kind == "port"
    with_port = dataclasses.replace(op, pieces=[*op.pieces, piece])
    k = 2 * math.pi * 12e9 / C0 * op.built.scale
    assert piece in split(with_port.pieces, op.built.scale, k).nonpolynomial
    names = {t.name for t in operator_terms(model, with_port, nodes=False)}
    assert piece.name not in names
    assert all(not n.startswith("waveport") for n in names)


def test_operator_derivative_matches_a_central_difference(hybrid):
    _, rule, op, ports = hybrid
    sigma = 2 * math.pi * 12e9 / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        (low_rank,) = eop.low_rank
        k = sigma * (1.02 + 0.05j)
        _, c = low_rank.at(k)
        assert np.linalg.norm(c) > 1e-6  # the hybrid mode's correction is live
        x = free_vector(eop)
        h = 1e-5 * abs(k)
        difference = (eop.apply(k + h, x) - eop.apply(k - h, x)) / (2 * h)
        slope = eop.derivative(k, x)
        assert np.linalg.norm(slope - difference) <= 1e-8 * np.linalg.norm(slope)
    finally:
        eop.free_handle()


def test_shift_holds_the_low_rank_term_across_a_reshift(hybrid):
    _, rule, op, ports = hybrid
    sigma = 2 * math.pi * 12e9 / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        x = free_vector(eop, 1)
        # the prepared T(sigma) solves the operator apply gives, port terms in
        back = eop.solve(eop.apply(sigma, x))
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
        kx = eop.parts(x)[0]
        k = sigma * (1.01 + 0.03j)
        eop.reshift(k)
        back = eop.solve(eop.apply(k, x))
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
        # K is the same whatever the shift
        assert np.linalg.norm(eop.parts(x)[0] - kx) <= 1e-10 * np.linalg.norm(kx)
    finally:
        eop.free_handle()


# --- several ports and shared surfaces ------------------------------------------

CASE_B3 = (
    ("air_l", 1.0, 3.0),
    ("window_1", 10.0, 2.4),
    ("air_1", 1.0, 20.0),
    ("window_2", 10.0, 2.4),
    ("air_2", 1.0, 9.9),
    ("window_3", 10.0, 2.4),
    ("air", 1.0, 3.0),
)
CASE_N = (("dielectric", 4.0, 6.0), ("air", 1.0, 5.0))
# exact roots, GHz: B3 between two TE10 ports; C's TE20 family, on a guide
# 5 mm high whose one port surface carries TE10 and TE20; N on the outgoing
# branch, Re f at 0.977 of the TE10 cutoff (6.557 GHz)
# (relative error at order 3 on these meshes: B3 6.9e-6, C 3.4e-5, N 3.9e-7)
ROOT_B3 = 9.974259423230 + 0.096704835502j
ROOT_C20 = 15.216633733515 + 0.207573093830j
ROOT_N = 6.403885446223 + 0.564967925392j


def ported_guide(
    path, layers, *, left_port, b_mm=B_MM, nx=6, nz_per_mm=0.5, ny=2, mirror=False
):
    """``layers`` from z = 0 (a short, or the port ``portl``) to ``port``.

    ``mirror``: the cells of the upper half in z mirror the lower half's, so
    a symmetric stack gives a mesh symmetric under the reflection that swaps
    the ports.
    """
    tops = np.cumsum([t for _, _, t in layers])
    zs = [np.zeros(1)]
    for (_, _, t), z1 in zip(layers, tops, strict=True):
        n = max(1, round(t * nz_per_mm))
        zs.append(z1 - t + np.linspace(0, t, n + 1)[1:])
    edges = (
        np.linspace(0, A_MM, nx + 1),
        np.linspace(0, b_mm, ny + 1),
        np.concatenate(zs),
    )
    length = tops[-1]

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if left_port and abs(c[2]) < 1e-9:
            return "portl"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        return "walls"

    mesh = sm.structured(edges, volume, surface, mirror=(False, False, mirror))
    sm.write_msh22(path, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def ported_model(mesh, layers, ports):
    return gfem.Model(
        mesh,
        materials={name: gfem.Material(eps_r=eps) for name, eps, _ in layers},
        boundaries=[gfem.PEC("walls")],
        ports=ports,
    )


def nearest_error(result, root):
    f = ghz(result.omega)
    j = int(np.argmin(np.abs(f - root)))
    return abs(f[j] - root) / abs(root), j


def test_two_port_surfaces_resonance_matches_the_exact_root(tmp_path):
    mesh = ported_guide(tmp_path / "b3.msh", CASE_B3, left_port=True)
    model = ported_model(
        mesh, CASE_B3, [gfem.WavePort("left", "portl"), gfem.WavePort("right", "port")]
    )
    result = gfem.solve(
        model, gfem.Eigenmode(target=9.97e9, n_modes=3), options=ORDER_3
    )
    error, j = nearest_error(result, ROOT_B3)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12


def test_shared_port_surface_resonance_matches_the_exact_root(tmp_path):
    # the TE20 family needs the surface's modal correction: its propagation
    # constant is not the lead (TE10) mode's of the Robin term
    mesh = ported_guide(tmp_path / "c.msh", CASE_A3, left_port=False, b_mm=5.0)
    model = ported_model(
        mesh,
        CASE_A3,
        [gfem.WavePort("te10", "port"), gfem.WavePort("te20", "port", mode=2)],
    )
    result = gfem.solve(
        model, gfem.Eigenmode(target=15.2e9, n_modes=3), options=ORDER_3
    )
    error, j = nearest_error(result, ROOT_C20)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12


def test_decaying_root_below_cutoff_is_kept_and_its_mirror_dropped(tmp_path):
    # a low-Q root with Re f below the port's cutoff decays in time: it is a
    # resonance; its mirror (Im f < 0) grows in time and never comes back
    mesh = ported_guide(tmp_path / "n.msh", CASE_N, left_port=False)
    model = ported_model(mesh, CASE_N, [gfem.WavePort("p1", "port")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntiBoundRootWarning)
        result = gfem.solve(
            model, gfem.Eigenmode(target=6.7e9, n_modes=4), options=ORDER_3
        )
    error, j = nearest_error(result, ROOT_N)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12
    f = ghz(result.omega)
    assert np.all(f.imag > 0)


# --- the coupling Q -------------------------------------------------------------

# two eps_r = 10 windows: a partial reflector, Q 869
CASE_A4 = (
    ("air", 1.0, 20.0),
    ("window", 10.0, 2.4),
    ("air", 1.0, 9.9),
    ("window", 10.0, 2.4),
    ("air", 1.0, 3.0),
)
ROOT_A4 = 9.968822933448 + 0.005732947875j


def coupling_sum_error(result, j):
    """``|sum 1/q_p - 1/Q| Q`` of mode ``j``, ``Q = result.q``."""
    q = float(result.q[j])
    total = sum(1 / float(v[j]) for v in result.q_port.values())
    return abs(total - 1 / q) * q


def test_a_wave_port_s_coupling_q_sums_to_the_mode_s_q_at_high_q(tmp_path):
    # the outgoing power is the mode's loss, so 1/q_port is 1/Q up to the
    # O(1/Q^2) of the electric energy against the stored energy
    mesh = shorted_guide(tmp_path / "a4.msh", CASE_A4)
    result = gfem.solve(
        guide_model(mesh, CASE_A4),
        gfem.Eigenmode(target=9.97e9, n_modes=1),
        options=ORDER_3,
    )
    assert abs(ghz(result.omega[0]) - ROOT_A4) / abs(ROOT_A4) <= 1e-4
    assert list(result.q_port) == ["p1"]
    assert coupling_sum_error(result, 0) <= 1e-6  # 2.8e-7


def test_a_wave_port_s_coupling_q_is_within_a_percent_of_q_at_q_5(case_a):
    result = gfem.solve(
        case_a, gfem.Eigenmode(target=11.67e9, n_modes=3), options=ORDER_3
    )
    j = int(np.argmin(np.abs(ghz(result.omega) - ROOT_A)))
    assert coupling_sum_error(result, j) <= 1e-2  # 7.9e-3


# port, air 5, eps_r = 10 4, air 5, port (mm): each port carries half the power
CASE_B = (("air", 1.0, 5.0), ("window", 10.0, 4.0), ("air", 1.0, 5.0))
ROOT_B = 11.884342073581 + 2.083440824738j
TWO_PORTS = [gfem.WavePort("left", "portl"), gfem.WavePort("right", "port")]


def test_the_two_ports_of_a_mirrored_guide_share_its_coupling_q_equally(tmp_path):
    # many high-Q modes of other guide families lie nearer the real target
    # than this Q 2.9 one
    mesh = ported_guide(tmp_path / "b.msh", CASE_B, left_port=True, mirror=True)
    result = gfem.solve(
        ported_model(mesh, CASE_B, TWO_PORTS),
        gfem.Eigenmode(target=11.88e9, n_modes=16),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    error, j = nearest_error(result, ROOT_B)
    assert error <= 2e-2
    q1, q2 = (float(result.q_port[name][j]) for name in ("left", "right"))
    assert abs(q1 - q2) <= 1e-8 * q1  # 8e-15


# the ports' shares of B3's outgoing power at the port planes, exact: in the
# high-Q limit q_p = Q / share
SHARE_B3 = {"left": 0.943972, "right": 0.056028}


def test_two_ports_split_the_coupling_q_as_their_shares_of_the_power(tmp_path):
    mesh = ported_guide(tmp_path / "b3.msh", CASE_B3, left_port=True)
    result = gfem.solve(
        ported_model(mesh, CASE_B3, TWO_PORTS),
        gfem.Eigenmode(target=9.97e9, n_modes=3),
        options=ORDER_3,
    )
    error, j = nearest_error(result, ROOT_B3)
    assert error <= 1e-4
    q = float(result.q[j])
    for name, share in SHARE_B3.items():
        # 1.0e-4 (left) and 3.5e-4 (right) at order 3
        assert float(result.q_port[name][j]) == pytest.approx(q / share, rel=2e-3)


# C's TE10 family, on the guide 5 mm high whose port surface requests TE10
# and TE20
ROOT_C10 = 14.123326299100 + 0.380179415083j


def test_each_port_of_a_shared_surface_reports_its_own_mode_s_coupling_q(tmp_path):
    mesh = ported_guide(tmp_path / "c.msh", CASE_A3, left_port=False, b_mm=5.0)
    model = ported_model(
        mesh,
        CASE_A3,
        [gfem.WavePort("te10", "port"), gfem.WavePort("te20", "port", mode=2)],
    )
    result = gfem.solve(
        model, gfem.Eigenmode(target=14.6e9, n_modes=4), options=ORDER_3
    )
    for root, own, other in ((ROOT_C10, "te10", "te20"), (ROOT_C20, "te20", "te10")):
        error, j = nearest_error(result, root)
        assert error <= 1e-4
        # the mode leaves through its own port mode alone (6e-4 at Q 19 and 37)
        assert coupling_sum_error(result, j) <= 2e-3
        assert float(result.q_port[other][j]) >= 1e6 * float(result.q_port[own][j])


def _case_a_field(mesh, eps):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.MaterialField(eps_r=eps),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )


GRADIENT_OPTIONS = gfem.SolverOptions(
    order=2, linear_solver="direct", eigen_tolerance=1e-12
)


def test_a_wave_port_s_coupling_q_gradient_matches_a_central_difference(tmp_path):
    # the load reads the port mode's amplitude and power at the mode's
    # complex k through the reduced model's derivatives
    mesh = shorted_guide(tmp_path / "a.msh", CASE_A, nx=4)

    problem = gfem.Eigenmode(target=11.67e9, n_modes=3)
    eps = jax.numpy.full(len(mesh.elements("dielectric")), 4.0)
    first = gfem.solve(_case_a_field(mesh, eps), problem, options=GRADIENT_OPTIONS)
    j = nearest_error(first, ROOT_A)[1]

    def objective(eps):
        result = gfem.solve(_case_a_field(mesh, eps), problem, options=GRADIENT_OPTIONS)
        return 1 / result.q_port["p1"][j]

    rng = np.random.default_rng(3)
    assert_gradient_matches(
        objective, eps, unit_directions(rng, eps.shape, 2), step=1e-2
    )
