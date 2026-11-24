"""The port mode normalization as a pure function, and its hand-written pullback.

Each cotangent is checked against central differences of `normalize_rows` on
the rows of a lossy slab at three frequencies. The overlap chain is included:
holding the previous row fixed is detectably wrong.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_normalization import normalize_rows, normalize_vjp
from gfem._port_section import port_section
from gfem._sweep.port_mode import _Greedy

DATA = Path(__file__).parent / "data"
FREQS = np.array([11e9, 12.5e9, 14e9])


@dataclasses.dataclass
class Case:
    fields: object
    mass: np.ndarray
    y: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    volts: np.ndarray
    greedy: _Greedy


@pytest.fixture(scope="module")
def case():
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=4.0, tan_delta=0.05),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    section = port_section(model, model.ports[0])
    space = PortModeSpace(section, 2)
    g = _Greedy(space, FREQS, [1], 1e-6, [section.port])
    g.run()
    fields = space.reduce(g.reduced.v)
    mass = fields.w.T @ (space.mass @ fields.w)
    y = np.array([g.coords_at(i, 0) for i in range(len(FREQS))])
    gamma = 1j * np.sqrt(-g.lam[0] + 0j)
    rng = np.random.default_rng(3)
    volts = rng.standard_normal(fields.w.shape[1]) + 1j * rng.standard_normal(
        fields.w.shape[1]
    )
    return Case(fields, mass, y, gamma, 2 * math.pi * FREQS, volts, g)


def test_the_sweep_normalizes_through_the_pure_function(case):
    _, coords, gamma, voltage, phase = case.greedy.normalized([None], [None])
    rows = normalize_rows(case.fields, case.mass, case.y, case.gamma, case.omega)
    np.testing.assert_array_equal(rows.z, coords[0])
    np.testing.assert_array_equal(rows.phase, phase[0])
    assert np.isnan(voltage[0]).all()
    np.testing.assert_allclose(gamma[0] * case.greedy.space.section.scale, case.gamma)


def _objective(norm, probes):
    a, b, c = probes
    return float(
        np.real(np.sum(a * norm.z))
        + np.real(np.sum(b * norm.voltage))
        + np.real(np.sum(c * norm.phase))
    )


def _probes(case, seed=0):
    rng = np.random.default_rng(seed)
    shape = case.y.shape[0], case.fields.w.shape[1]
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    return cplx(*shape), cplx(shape[0]), cplx(shape[0])


def _check(case, *, volts=None, polarity=None, h=1e-6):
    a, b, c = probes = _probes(case)
    kw = {"volts": volts, "polarity": polarity}
    base = normalize_rows(case.fields, case.mass, case.y, case.gamma, case.omega, **kw)
    cot = normalize_vjp(
        case.fields,
        case.mass,
        base,
        case.y,
        case.gamma,
        case.omega,
        z_bar=a,
        voltage_bar=b if volts is not None else None,
        phase_bar=c,
        **kw,
    )
    rng = np.random.default_rng(11)
    dy = rng.standard_normal(case.y.shape) + 1j * rng.standard_normal(case.y.shape)
    dg = rng.standard_normal(3) + 1j * rng.standard_normal(3)

    def value(t):
        norm = normalize_rows(
            case.fields,
            case.mass,
            case.y + t * dy,
            case.gamma + t * dg,
            case.omega,
            **kw,
        )
        if volts is None:
            norm.voltage[:] = 0
        return _objective(norm, probes)

    fd = (value(h) - value(-h)) / (2 * h)
    got = float(np.real(np.sum(cot.y * dy) + np.sum(cot.gamma * dg)))
    # the normalization does not see the eigenvector's scale
    scale = np.abs(cot.y).max() * np.abs(case.y).max()
    assert abs(np.sum(cot.y * case.y, axis=1)).max() <= 1e-9 * scale * 10
    return got, fd, cot, base


def test_the_pullback_matches_differences_with_the_overlap_chain(case):
    got, fd, _, _ = _check(case)
    assert got == pytest.approx(fd, rel=1e-9)


def test_a_voltage_path_pulls_back_the_sign_and_the_voltage(case):
    got, fd, _, base = _check(case, volts=case.volts)
    assert got == pytest.approx(fd, rel=1e-9)
    # both signs occur over random functionals; the case is not trivial
    assert np.isfinite(base.voltage).all()


def test_a_polarity_functional_fixes_every_row_without_a_chain(case):
    rng = np.random.default_rng(5)
    n = case.fields.w.shape[1]
    polarity = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    got, fd, _, _ = _check(case, polarity=polarity)
    assert got == pytest.approx(fd, rel=1e-9)


def test_cutting_the_overlap_chain_is_detectably_wrong(case):
    probes = _probes(case)
    rng = np.random.default_rng(11)
    dy = rng.standard_normal(case.y.shape) + 1j * rng.standard_normal(case.y.shape)
    h = 1e-6

    def full(t):
        norm = normalize_rows(
            case.fields, case.mass, case.y + t * dy, case.gamma, case.omega
        )
        return float(np.real(np.sum(probes[0] * norm.z)))

    base = normalize_rows(case.fields, case.mass, case.y, case.gamma, case.omega)

    def cut(t):
        # every row after the first is fixed against the base previous row
        first = normalize_rows(
            case.fields,
            case.mass,
            (case.y + t * dy)[:1],
            case.gamma[:1],
            case.omega[:1],
        )
        z = [first.z[0]]
        for i in (1, 2):
            row = normalize_rows(
                case.fields,
                case.mass,
                (case.y + t * dy)[i : i + 1],
                case.gamma[i : i + 1],
                case.omega[i : i + 1],
                reference=base.z[i - 1],
            )
            z.append(row.z[0])
        a = probes[0]
        return float(np.real(np.sum(a * np.array(z))))

    fd_full = (full(h) - full(-h)) / (2 * h)
    fd_cut = (cut(h) - cut(-h)) / (2 * h)
    assert abs(fd_cut - fd_full) > 1e-4 * abs(fd_full)
    cot = normalize_vjp(
        case.fields,
        case.mass,
        base,
        case.y,
        case.gamma,
        case.omega,
        z_bar=probes[0],
    )
    assert float(np.real(np.sum(cot.y * dy))) == pytest.approx(fd_full, rel=1e-9)


def test_the_form_pairs_carry_the_explicit_form_dependence(case):
    probes = _probes(case)
    base = normalize_rows(case.fields, case.mass, case.y, case.gamma, case.omega)
    cot = normalize_vjp(
        case.fields,
        case.mass,
        base,
        case.y,
        case.gamma,
        case.omega,
        z_bar=probes[0],
        phase_bar=probes[2],
    )
    fields = case.fields
    rng = np.random.default_rng(2)
    n = fields.w.shape[0]
    names = {p.form for p in cot.pairs}
    assert names == {"power_g", "power_t", "gauge_n", "gauge_t"}
    g_full, t_full = fields.space.power_forms
    ln, lt = fields.space.gauge_forms
    dg = g_full.copy()
    dg.data = rng.standard_normal(dg.data.shape)
    dt = t_full.copy()
    dt.data = rng.standard_normal(dt.data.shape)
    dln, dlt = rng.standard_normal(n), rng.standard_normal(n)
    direction = {"power_g": dg, "power_t": dt, "gauge_n": dln, "gauge_t": dlt}
    got = 0.0
    for p in cot.pairs:
        d = direction[p.form]
        value = (p.left @ (d @ p.right)) if p.left is not None else d @ p.right
        got += float(np.real(p.weight * value))
    h = 1e-6

    def value(t):
        f = fields
        saved = f._g, f._t, f._ln, f._lt
        w = f.w
        f._g = saved[0] + t * (w.T @ (dg @ w))
        f._t = saved[1] + t * (w.T @ (dt @ w))
        f._ln = saved[2] + t * (dln @ w)
        f._lt = saved[3] + t * (dlt @ w)
        try:
            norm = normalize_rows(f, case.mass, case.y, case.gamma, case.omega)
        finally:
            f._g, f._t, f._ln, f._lt = saved
        norm.voltage[:] = 0
        return _objective(norm, (probes[0], np.zeros(3), probes[2]))

    fd = (value(h) - value(-h)) / (2 * h)
    assert got == pytest.approx(fd, rel=1e-8)
    del ln, lt
