r"""Time responses and rational models of S from real solves.

V1, the line of the get-started page (``docs/tutorials/data/line.msh``: a
6 mm wide, 1 mm high parallel-plate line, 10 mm of air, 20 mm of
:math:`\varepsilon_r = 10` ceramic, 10 mm of air, lumped ports on the air
line's impedance :math:`Z_0 = 376.73/6` Ω), seen by a time-domain
reflectometer: S on the harmonic grid :math:`f_k = k\,\Delta f`,
:math:`\Delta f` = 50 MHz, 400 points to 20 GHz, from an adaptive sweep. The
ceramic reads :math:`Z_0/\sqrt{10}`, the far air line :math:`Z_0` again, the
first edge reflects :math:`\Gamma = (1/\sqrt{10} - 1)/(1/\sqrt{10} + 1)` at
the round trip of 10 mm of air, and the edge rises in about
:math:`0.99/f_\mathrm{max}` (Kaiser :math:`\beta = 6`).

V2, the rational model of V1's S: its resampled and closed-form steps
against the windowed step of the data.

V3, an electrically long wave-port line (10 phase wraps at the top of the
band): S against the exact TEM line, and the poles the fit needs.

V4, the band-pass impulse response of a waveguide above cutoff, the
dielectric-slab insert of tests/verification/test_shared_port_surface.py
with single-mode ports, against the same transform of the mode-matching S.
"""

import dataclasses
import math
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from mode_matching import s_matrix

import gfem

C0 = 299792458.0
DOCS_MESH = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"
ONE_THREAD = gfem.SolverOptions(threads=1)

# --- V1, V2: the docs line ----------------------------------------------------

Z_LINE = 376.73 / 6  # ohm
DF, N = 50e6, 400  # Hz: the record T = 1/DF = 20 ns
F_MAX = DF * N
T1 = 2 * 10e-3 / C0  # round trip to the ceramic, s
T2 = T1 + 2 * 20e-3 * math.sqrt(10) / C0  # ... and to its far end


@pytest.fixture(scope="module")
def docs_line():
    mesh = gfem.Mesh.from_gmsh(DOCS_MESH, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=Z_LINE),
            gfem.LumpedPort("p2", "port2", "+z", impedance=Z_LINE),
        ],
    )
    f = DF * np.arange(1, N + 1)
    result = gfem.solve(model, gfem.Driven(f, sweep="adaptive"), options=ONE_THREAD)
    assert result.sweep.converged
    return result.s


@pytest.fixture(scope="module")
def tdr(docs_line):
    return docs_line.step_response()


def _at(response, t, trace):
    return float(np.interp(t, response.time, np.asarray(trace)))


def test_v1_the_impedance_profile_reads_the_ceramic_and_the_far_line(tdr):
    z = tdr.impedance("p1")
    plateau = _at(tdr, (T1 + T2) / 2, z)
    late = _at(tdr, tdr.time[-1] - 0.2e-9, z)
    print(f"\nV1: ceramic {plateau:.4f} ohm, far line {late:.4f} ohm")  # noqa: T201
    assert plateau == pytest.approx(Z_LINE / math.sqrt(10), abs=0.1)  # 19.860
    assert late == pytest.approx(Z_LINE, abs=0.1)  # 62.783


def test_v1_the_impulse_reflects_the_first_edge_at_its_round_trip(docs_line):
    gamma = (1 / math.sqrt(10) - 1) / (1 / math.sqrt(10) + 1)
    h = docs_line.impulse_response()
    trace = np.asarray(h["p1", "p1"])
    i = int(np.argmin(trace))
    conj = dataclasses.replace(docs_line, values=jnp.conj(docs_line.values))
    h_conj = conj.impulse_response()
    j = int(np.argmin(np.asarray(h_conj["p1", "p1"])))
    print(  # noqa: T201
        f"\nV1: impulse minimum {trace[i]:.5f} at {h.time[i] * 1e12:.2f} ps, "
        f"conjugated at {h_conj.time[j] * 1e12:.2f} ps"
    )
    assert trace[i] == pytest.approx(gamma, abs=5e-3)  # -0.5190
    assert h.time[i] == pytest.approx(T1, abs=3e-12)  # 65.9 ps
    # under the wrong time convention the reflection lands before the step
    assert h_conj.time[j] == pytest.approx(-T1, abs=3e-12)


def test_v1_the_transmitted_step_settles_at_one(tdr):
    end = float(np.asarray(tdr["p2", "p1"])[-1])
    print(f"\nV1: TDT end {end:.6f}")  # noqa: T201
    assert end == pytest.approx(1.0, abs=1e-3)  # 0.99996


def test_v1_the_first_edge_rises_in_about_one_over_the_span(tdr):
    t = np.asarray(tdr.time)
    window = (t > 0) & (t < 0.3e-9)
    rise = gfem.rise_time(t[window], np.asarray(tdr["p1", "p1"])[window])
    print(f"\nV1: rise {rise * 1e12:.2f} ps vs {0.99 / F_MAX * 1e12:.2f}")  # noqa: T201
    assert rise == pytest.approx(0.99 / F_MAX, rel=0.05)  # 49.2 vs 49.5 ps


@pytest.fixture(scope="module")
def docs_fit(docs_line):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return docs_line.fit_rational()


def test_v2_the_docs_line_fits_passive_within_tolerance(docs_fit):
    print(  # noqa: T201
        f"\nV2: {docs_fit.n_poles} poles, error {docs_fit.error:.2e}, "
        f"fit error {docs_fit.fit_error:.2e}"
    )
    assert docs_fit.error <= 1e-3
    assert docs_fit.passive
    assert docs_fit.n_poles <= 40  # 24


def test_v2_the_resampled_step_is_the_direct_step(docs_fit, tdr):
    resampled = docs_fit.network(DF * np.arange(0, N + 1)).step_response()
    np.testing.assert_allclose(resampled.time, tdr.time, rtol=1e-12)
    late = np.asarray(tdr.time) > 0
    gap = np.abs(np.asarray(resampled.values) - np.asarray(tdr.values))[late].max()
    print(f"\nV2: resampled vs direct step {gap:.2e}")  # noqa: T201
    assert gap <= 2e-3  # 3.7e-4


def test_v2_the_closed_form_step_is_the_windowed_step(docs_fit, tdr):
    # the window rounds the response in band, about 2.7 y''/(2 pi f_max)^2:
    # the ceramic's echoes (round trip 0.42 ns, its resonances are pole pairs
    # of Q 14 to 21) keep sharp edges up to about 2.5 ns (gap 1.3e-2 from
    # 1 ns), so the gap is read once they have decayed (3.3e-4 from 3 ns)
    t = np.asarray(tdr.time)
    late = t >= 3e-9
    exact = docs_fit.step_response(t[late])
    gap = np.abs(np.asarray(exact.values) - np.asarray(tdr.values)[late]).max()
    print(f"\nV2: closed form vs windowed step {gap:.2e}")  # noqa: T201
    assert gap <= 2e-3  # 3.3e-4


# --- V3: an electrically long wave-port line ----------------------------------

LW, LD, LONG = 40.0, 10.0, 500.0  # um: plate width, gap, line length
LONG_EPS = 4.0
LONG_F = 3e12 / 150 * np.arange(1, 151)  # 10 phase wraps at 3 THz


@pytest.fixture(scope="module")
def long_line(tmp_path_factory):
    edges = (np.array([0, LW]), np.linspace(0, LD, 3), np.linspace(0, LONG, 101))
    tol = 1e-9 * LONG

    def surface(c):
        if abs(c[2]) < tol:
            return "port1"
        if abs(c[2] - LONG) < tol:
            return "port2"
        if abs(c[1]) < tol or abs(c[1] - LD) < tol:
            return "plates"
        return "sides"

    path = sm.write_msh22(
        tmp_path_factory.mktemp("long") / "long.msh",
        sm.structured(edges, lambda _: "fill", surface),
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")

    def path_at(z):
        return ((LW / 2 * 1e-6, 0.0, z * 1e-6), (LW / 2 * 1e-6, LD * 1e-6, z * 1e-6))

    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=LONG_EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=path_at(0.0)),
            gfem.WavePort("p2", "port2", voltage_path=path_at(LONG)),
        ],
    )
    options = gfem.SolverOptions(order=3, threads=1, adaptive_max_samples=30)
    result = gfem.solve(model, gfem.Driven(LONG_F, sweep="adaptive"), options=options)
    assert result.sweep.converged
    return result.s


def test_v3_a_long_wave_port_line_is_the_exact_tem_line(long_line):
    s = np.asarray(long_line.values)
    exact = np.exp(-2j * np.pi * LONG_F * LONG * 1e-6 * math.sqrt(LONG_EPS) / C0)
    through = np.abs(s[:, 1, 0] - exact).max()
    reflection = np.abs(s[:, 0, 0]).max()
    print(f"\nV3: |S21 - exact| {through:.2e}, |S11| {reflection:.2e}")  # noqa: T201
    assert through <= 1e-5  # 9.5e-7
    assert reflection <= 1e-5


def test_v3_the_fit_takes_about_two_poles_per_wrap(long_line):
    # the TEM mode's Z_PV is one real value, up to the mode solve's round-off,
    # and the fit keeps it as it is, at each port's mean
    z_ref = np.asarray(long_line.z_ref)
    spread = np.ptp(z_ref.real, axis=0).max() / np.abs(z_ref).max()
    model = long_line.fit_rational()
    np.testing.assert_allclose(model.z_ref, z_ref.real.mean(axis=0), rtol=1e-15)
    on_50 = long_line.fit_rational(z0=50.0)
    print(  # noqa: T201
        f"\nV3: Z_PV {model.z_ref.tolist()} ohm, relative spread {spread:.1e}, "
        f"imaginary {np.abs(z_ref.imag).max():.1e} ohm; {model.n_poles} poles, "
        f"error {model.error:.2e}; on 50 ohm {on_50.n_poles} poles, error "
        f"{on_50.error:.2e}"
    )
    assert model.n_poles <= 32  # 24
    assert model.error <= 1e-3
    assert model.passive
    # the mismatch to 50 ohm bounces on the line: more poles
    assert on_50.n_poles <= 56  # 44
    assert on_50.error <= 1e-3


# --- V4: band pass on a waveguide above cutoff --------------------------------

A, B, SLAB, INSERT, LEAD = 22.86, 5.0, 8.0, 6.0, 40.0  # mm
END = 2 * LEAD + INSERT
GUIDE_EPS = 4.0
# TE10 alone propagates from 6.56 to 13.1 GHz; at 11 GHz the TE20 the slab
# excites is down to 2.5e-3 at the ports. A record of 1/(150 MHz) = 6.7 ns,
# over ten times the transit.
BAND = np.linspace(8e9, 11e9, 21)


def _guide(directory):
    def lines(*pieces):
        return np.unique(np.concatenate([np.linspace(*p) for p in pieces]))

    edges = (
        lines((0, SLAB, 9), (SLAB, A, 16)),
        np.linspace(0, B, 3),
        lines((0, LEAD, 21), (LEAD, LEAD + INSERT, 7), (LEAD + INSERT, END, 21)),
    )
    eps = 1e-9

    def volume(c):
        return "slab" if c[0] < SLAB and LEAD < c[2] < LEAD + INSERT else "air"

    def surface(c):
        x, y, z = c
        if abs(z) < eps:
            return "port1"
        if abs(z - END) < eps:
            return "port2"
        if min(x, A - x, y, B - y) < eps:
            return "walls"
        return ""

    path = sm.write_msh22(
        directory / "insert.msh", sm.structured(edges, volume, surface)
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _lead_gamma(f):
    k0 = 2 * np.pi * np.asarray(f) / C0
    return 1j * np.sqrt(k0**2 - (np.pi / (A * 1e-3)) ** 2)


def _reference_s(f):
    """Mode-matching S of the insert, moved out to the ports along the leads."""
    s = s_matrix(
        f, a=A * 1e-3, slab=SLAB * 1e-3, eps=GUIDE_EPS, length=INSERT * 1e-3, modes=(1,)
    )
    return s * np.exp(-2 * _lead_gamma(f) * LEAD * 1e-3)


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    # no voltage path: S on unit modal power, as mode matching gives it
    model = gfem.Model(
        _guide(tmp_path_factory.mktemp("guide")),
        materials={"slab": gfem.Material(eps_r=GUIDE_EPS), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    # order 2: S within 2.2e-4 of mode matching (order 3: 1.6e-5, eight
    # times the cost)
    options = gfem.SolverOptions(order=2, threads=1, linear_solver="direct")
    result = gfem.solve(model, gfem.Driven(BAND, sweep="adaptive"), options=options)
    assert result.sweep.converged
    fem = result.s
    reference = dataclasses.replace(
        fem, values=jnp.asarray(np.array([_reference_s(f) for f in BAND]))
    )
    return fem, reference


def test_v4_the_band_pass_impulse_is_the_mode_matching_one(guide):
    fem, reference = guide
    s_gap = np.abs(np.asarray(fem.values) - np.asarray(reference.values)).max()
    h_fem = fem.impulse_response(mode="band_pass")
    h_ref = reference.impulse_response(mode="band_pass")
    gap = np.abs(np.asarray(h_fem.values) - np.asarray(h_ref.values)).max()
    print(  # noqa: T201
        f"\nV4: max |S - S_ref| {s_gap:.2e}, max_t ||h| - |h_ref|| {gap:.2e}"
    )
    assert gap <= 1e-2  # 1.9e-5


def test_v4_the_transmitted_impulse_peaks_at_the_group_delay(guide):
    fem, _ = guide
    h = fem.impulse_response(mode="band_pass")
    peak = h.time[int(np.argmax(np.asarray(h["p2", "p1"])))]
    # the leads' and the insert's group delay at the band centre, -d phase /
    # d omega of the mode-matching S21
    centre, step = BAND.mean(), 1e6
    phase = np.unwrap(
        [np.angle(_reference_s(f)[1, 0]) for f in (centre - step, centre + step)]
    )
    delay = -(phase[1] - phase[0]) / (2 * np.pi * 2 * step)
    print(  # noqa: T201
        f"\nV4: |h21| peaks at {peak * 1e12:.1f} ps, group delay {delay * 1e12:.1f} ps"
    )
    assert peak == pytest.approx(delay, rel=0.05)  # 442.7 vs 441.6 ps
