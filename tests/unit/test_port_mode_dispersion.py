r"""The mode derivative of an exact port mode solve: group index and dispersion.

On a homogeneously filled PEC guide the discrete modes obey
:math:`\beta_h^2 = \varepsilon k_0^2 - k_{c,h}^2` with a cutoff
:math:`k_{c,h}` fixed by the mesh, so the computed :math:`\gamma` gives
:math:`k_{c,h}` and the closed form isolates the method error from the mesh
error. There the mode's shape does not change with :math:`k_0`
(:math:`x' = 0`), so the slab-loaded guide carries the cases with a mode
derivative to solve for.
"""

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pml_cases
import pytest
import scipy.constants
import scipy.sparse.linalg as spla

import gfem
from gfem._engine.mode_derivative import ModeDerivative, gmres_on, mode_derivative
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.taylor import Taylor
from gfem._linalg.superlu import ShiftInvert
from gfem._port_section import port_section
from gfem._sweep import port_mode as sweep
from gfem._sweep.port_mode import _Greedy, _wavenumbers

DATA = Path(__file__).parent / "data"
LINE_MESH = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line2d.msh"
C0 = 299792458.0
EPS0 = scipy.constants.epsilon_0
F = 15e9

# A strongly dispersive Lorentz law, resonance above the X band
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40e9, damping=2e9)],
)


def _space(mesh, region, fill, *others):
    materials = {region: gfem.Material(eps_r=fill)}
    materials.update({name: gfem.Material() for name in others})
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    return PortModeSpace(port_section(model, model.ports[0]), 2)


def _fundamental(space, frequency):
    """``k0``, the factor, and the fundamental mode's eigenvalue and vector."""
    k0, _, shift = _wavenumbers(space.section, frequency)
    a, b = space.pencil.at(k0)
    factor = ShiftInvert.factor(a, b, shift)
    lam, vecs = factor.eigs(6)
    q = int(np.argmax(np.sqrt(-lam + 0j).real))
    return k0, factor, complex(lam[q]), vecs[:, q]


def _derive(space, k0, lam, x, solve):
    pencil = space.pencil
    return mode_derivative(
        pencil.a, pencil.b, pencil.taylor(k0), lam, x, solve=solve, k0=k0
    )


def _relative(a, b):
    return abs(a - b) / abs(b)


def _eps_series(fill, k0, scale):
    r""":math:`\varepsilon` along real :math:`k_0`, written out from the law."""
    if not isinstance(fill, gfem.PermittivityLaw):
        return Taylor.constant(fill)
    f = Taylor.variable(k0) * (C0 / (2 * math.pi * scale))
    eps = Taylor.constant(fill.eps_inf)
    for t in fill.terms:
        f0, g = t.resonance_frequency, t.damping
        eps = eps + t.delta_eps * f0**2 / (f0**2 - f * f + 1j * g * f)
    return eps


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def slab_mesh():
    return gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")


@pytest.mark.parametrize("fill", [1.0, 4.0, LORENTZ], ids=["air", "eps4", "lorentz"])
def test_a_filled_guide_s_dispersion_is_its_closed_form(guide_mesh, fill):
    space = _space(guide_mesh, "air", fill)
    scale = space.section.scale
    k0, factor, lam, x = _fundamental(space, F)
    derivative = _derive(space, k0, lam, x, gmres_on(factor, 1e-10))
    assert isinstance(derivative, ModeDerivative)
    n_g, beta_2, beta_3 = derivative.dispersion(scale)
    # the discrete cutoff from the computed gamma, then beta_h along k0
    eps = _eps_series(fill, k0, scale)
    k = Taylor.variable(k0)
    cutoff = eps.c[0] * k0**2 + lam
    beta = (eps * k * k - cutoff).sqrt()
    assert _relative(n_g, beta.c[1]) <= 1e-12
    assert _relative(beta_2, 2 * beta.c[2] * scale / C0**2) <= 1e-10
    assert _relative(beta_3, 6 * beta.c[3] * scale**2 / C0**3) <= 1e-9


def test_a_filled_guide_s_mode_derivative_needs_no_direct_solve(
    guide_mesh, monkeypatch
):
    # x' = 0 exactly: the right-hand side is rounding of its uncancelled terms
    space = _space(guide_mesh, "air", 1.0)
    k0, factor, lam, x = _fundamental(space, F)
    direct = []
    monkeypatch.setattr(spla, "splu", lambda *args: direct.append(args))
    derivative = _derive(space, k0, lam, x, gmres_on(factor, 1e-10))
    assert derivative.residual <= 1e-12
    assert direct == []


def test_gmres_that_does_not_converge_falls_back_to_a_direct_solve(slab_mesh):
    space = _space(slab_mesh, "slab", 4.0, "air")
    k0, factor, lam, x = _fundamental(space, F)
    preconditioned = _derive(space, k0, lam, x, gmres_on(factor, 1e-10))
    # no preconditioner: 4 restarts of 50 iterations cannot reach 1e-10
    identity = SimpleNamespace(solve=lambda r: r)
    fallen_back = _derive(space, k0, lam, x, gmres_on(identity, 1e-10))
    assert fallen_back.residual <= 1e-12
    scale = space.section.scale
    got = fallen_back.dispersion(scale)
    ref = preconditioned.dispersion(scale)
    for value, gate, want in zip(got, (1e-12, 1e-10, 1e-9), ref, strict=True):
        assert _relative(value, want) <= gate


# --- the sweep's exact solves ----------------------------------------------------


def _greedy(slab_mesh, dispersion=()):
    space = _space(slab_mesh, "slab", 4.0, "air")
    grid = np.linspace(10e9, 20e9, 20)
    return _Greedy(
        space, grid, [1], 1e-6, [gfem.WavePort("p1", "port1")], dispersion=dispersion
    )


def _spy_rtol(monkeypatch):
    seen = []

    def spy(factor, rtol):
        seen.append(rtol)
        return gmres_on(factor, rtol)

    monkeypatch.setattr(sweep, "gmres_on", spy)
    return seen


def test_a_sample_s_mode_derivative_solves_to_its_own_backward_error(
    slab_mesh, monkeypatch
):
    seen = _spy_rtol(monkeypatch)
    greedy = _greedy(slab_mesh, dispersion=(0,))
    first = greedy._solve(0)
    assert first.residual[0] < 1e-10
    greedy._keep(0, first, [0])
    # a sample at the low-frequency floor solves no tighter than its mode
    greedy._keep(1, dataclasses.replace(first, residual=np.full(1, 1e-6)), [0])
    assert seen == [1e-10, 1e-6]


def test_a_kept_sample_holds_its_derivative_and_drops_its_factor(slab_mesh):
    greedy = _greedy(slab_mesh, dispersion=(0,))
    first = greedy._solve(0)
    assert first.factor is not None
    greedy._keep(0, first, [0])
    assert greedy.candidates[0].factor is None
    (derivative,) = greedy.derivatives[0]
    assert derivative.lam == first.lam[0]
    assert derivative.k0 == greedy.k0[0]


def test_a_re_picked_sample_refactors_for_its_new_mode(slab_mesh, monkeypatch):
    greedy = _greedy(slab_mesh, dispersion=(0,))
    greedy._keep(0, greedy._solve(0), [0])
    factors = []
    real = ShiftInvert.factor

    def counted(*args):
        factors.append(args)
        return real(*args)

    monkeypatch.setattr(ShiftInvert, "factor", counted)
    greedy._keep(0, greedy.candidates[0], [1])
    assert len(factors) == 1
    assert greedy.derivatives[0][0].lam == greedy.candidates[0].lam[1]


def test_without_dispersion_a_sweep_keeps_no_factor_and_derives_nothing(
    slab_mesh, monkeypatch
):
    seen = _spy_rtol(monkeypatch)
    greedy = _greedy(slab_mesh)
    first = greedy._solve(0)
    assert first.factor is None
    greedy._keep(0, first, [0])
    assert greedy.derivatives == {}
    assert seen == []


def test_derivatives_leave_the_sweep_bit_identical(slab_mesh):
    plain, derived = _greedy(slab_mesh), _greedy(slab_mesh, dispersion=(0,))
    plain.run()
    derived.run()
    np.testing.assert_array_equal(derived.lam, plain.lam)
    np.testing.assert_array_equal(derived.residual, plain.residual)
    np.testing.assert_array_equal(derived.solved, plain.solved)
    assert sorted(derived.derivatives) == np.flatnonzero(derived.solved).tolist()


# --- the public result: homogeneous fills -----------------------------------------
#
# A homogeneous fill's discrete mode obeys beta^2 = eps(omega) k0^2 - k_c^2 with
# k_c fixed by the mesh and read from the reported gamma. The permittivity's
# omega derivatives are written out here, apart from the laws' own code.


def _law_eps(eps_inf, term):
    r""":math:`\varepsilon` and its first three :math:`\omega` derivatives."""

    def eps(w):
        g = 2 * math.pi * term.damping
        if isinstance(term, gfem.Lorentz):
            w0 = 2 * math.pi * term.resonance_frequency
            d, d1, d2 = w0**2 - w * w + 1j * g * w, -2 * w + 1j * g, -2.0
            k = term.delta_eps * w0**2
        else:  # Drude
            d, d1, d2 = w * w - 1j * g * w, 2 * w - 1j * g, 2.0
            k = -((2 * math.pi * term.plasma_frequency) ** 2)
        return np.array(
            [
                eps_inf + k / d,
                -k * d1 / d**2,
                k * (2 * d1**2 / d**3 - d2 / d**2),
                k * (-6 * d1**3 / d**4 + 6 * d1 * d2 / d**3),
            ]
        )

    return eps


def _constant(eps):
    return lambda _w: np.array([eps, 0, 0, 0], complex)


def _conductive(eps, sigma):
    s = sigma / EPS0
    return lambda w: np.array(
        [
            eps - 1j * s / w,
            1j * s / w**2,
            -2j * s / w**3,
            6j * s / w**4,
        ]
    )


def _closed_form(gamma, f, eps):
    r""":math:`n_g`, :math:`\beta_2`, :math:`\beta_3` of
    :math:`\beta^2 = \varepsilon(\omega)k_0^2 - k_c^2`, :math:`k_c` from ``gamma``.
    """
    beta = gamma / 1j
    w = 2 * math.pi * f
    k0 = w / C0
    e0, e1, e2, e3 = eps(w) * C0 ** np.arange(4)  # derivatives in k0
    q1 = 2 * k0 * e0 + k0**2 * e1
    q2 = 2 * e0 + 4 * k0 * e1 + k0**2 * e2
    q3 = 6 * e1 + 6 * k0 * e2 + k0**2 * e3
    b1 = q1 / (2 * beta)
    b2 = (q2 / 2 - b1**2) / beta
    b3 = (q3 / 2 - 3 * b1 * b2) / beta
    return b1, b2 / C0**2, b3 / C0**3


def _assert_closed_form(result, eps):
    want = _closed_form(complex(result.gamma[0]), float(result.frequency[0]), eps)
    got = (result.n_g[0], result.beta_2[0], result.beta_3[0])
    for value, gate, ref in zip(got, (1e-12, 1e-10, 1e-9), want, strict=True):
        assert _relative(complex(value), ref) <= gate


def _law(eps_inf, term):
    """A material of one law term, and its permittivity series (`_law_eps`)."""
    law = gfem.PermittivityLaw(eps_inf=eps_inf, terms=[term])
    return gfem.Material(eps_r=law), _law_eps(eps_inf, term)


GUIDE_FILLS = {
    "air": (gfem.Material(), _constant(1.0)),
    "eps4": (gfem.Material(eps_r=4.0), _constant(4.0)),
    "lorentz": _law(
        2.0, gfem.Lorentz(delta_eps=1.5, resonance_frequency=30e9, damping=0.0)
    ),
    "lossy-lorentz": _law(
        2.0, gfem.Lorentz(delta_eps=1.5, resonance_frequency=30e9, damping=3e9)
    ),
    "drude": _law(1.0, gfem.Drude(plasma_frequency=5e9, damping=0.0)),
    "lossy-drude": _law(1.0, gfem.Drude(plasma_frequency=5e9, damping=1e9)),
}


@pytest.mark.parametrize("fill", GUIDE_FILLS)
def test_a_filled_guide_s_reported_dispersion_is_its_closed_form(guide_mesh, fill):
    material, eps = GUIDE_FILLS[fill]
    model = gfem.Model(
        guide_mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    _assert_closed_form(gfem.solve(model, gfem.PortMode("p1", 10e9)), eps)


LINE_LAW = (
    2.0,
    gfem.Lorentz(delta_eps=2.0, resonance_frequency=200e9, damping=5e9),
)
# a loss tangent of 0.4 at the line cases' frequency
LINE_SIGMA = 0.4 * 2 * math.pi * 50e9 * EPS0
LINE_CASES = {
    "Ez-law": ("Ez", _law(*LINE_LAW), {}),
    "Hz-law": ("Hz", _law(*LINE_LAW), {}),
    "Hz-law-pec-ends": ("Hz", _law(*LINE_LAW), {"pec_edges": ("side",)}),
    "Hz-conductive-pec-ends": (
        "Hz",
        (
            gfem.Material(eps_r=4.0, conductivity=LINE_SIGMA),
            _conductive(4.0, LINE_SIGMA),
        ),
        {"pec_edges": ("side",)},
    ),
    "Ez-conductive-pec-ends": (
        "Ez",
        (
            gfem.Material(eps_r=4.0, conductivity=LINE_SIGMA),
            _conductive(4.0, LINE_SIGMA),
        ),
        {"pec_edges": ("side",)},
    ),
}


@pytest.mark.parametrize("case", LINE_CASES)
def test_a_filled_line_port_s_dispersion_is_its_closed_form(case):
    polarization, (material, eps), port = LINE_CASES[case]
    mesh = gfem.Mesh.from_gmsh(LINE_MESH, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"feed": material, "design": material},
        ports=[gfem.WavePort("p1", "port1", **port), gfem.WavePort("p2", "port2")],
    )
    problem = gfem.PortMode("p1", 50e9, polarization=polarization)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=3))
    _assert_closed_form(result, eps)


# --- the public result: finite differences of exact solves -------------------------
#
# Up to three frequencies are all exact solves, and a perfectly matched layer
# keeps the lowest one's stretch, so the outer rows of one three-row PortMode
# difference the middle row's derivatives. Each order is checked against the
# difference of the order below it (n_g of beta, beta_2 of n_g, beta_3 of
# beta_2), never a second difference of beta, which rounding swamps.

DELTA = 1e-4


def _guide_walls(walls):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[walls],
        ports=[gfem.WavePort("p1", "port1")],
    )
    return model, 10e9, {}


def _lossy_plates():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.Conductor("plates", conductivity=1e6)],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return model, 50e9, {}


def _absorbing_end_beside_a_law(polarization, damping):
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=200e9, damping=damping)],
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(LINE_MESH, length_unit="mm"),
        materials={
            "feed": gfem.Material(eps_r=law),
            "design": gfem.Material(eps_r=4.0),
        },
        boundaries=[gfem.Absorbing("side", order=2)],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return model, 50e9, {"polarization": polarization, "order": 3}


def _pml_strip(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.6, length=0.15, clad_h=0.2, core_h=0.1
    )
    return pml_cases.strip_model(path), pml_cases.STRIP_F, {}


FD_CASES = {
    "thick-conductor": lambda _: _guide_walls(
        gfem.Conductor("walls", conductivity=5.8e7)
    ),
    "film-conductor": lambda _: _guide_walls(
        gfem.Conductor("walls", conductivity=5.8e7, thickness=1e-7)
    ),
    "lossy-plates": lambda _: _lossy_plates(),
    "absorbing-end-Ez": lambda _: _absorbing_end_beside_a_law("Ez", 0.0),
    "absorbing-end-Hz-lossy": lambda _: _absorbing_end_beside_a_law("Hz", 5e9),
    "pml-strip": _pml_strip,
}


@pytest.mark.parametrize("case", FD_CASES)
def test_dispersion_is_the_difference_of_the_order_below(case, tmp_path):
    model, f, options = FD_CASES[case](tmp_path)
    polarization = options.pop("polarization", None)
    grid = [f * (1 - DELTA), f, f * (1 + DELTA)]
    result = gfem.solve(
        model,
        gfem.PortMode("p1", grid, polarization=polarization),
        options=gfem.SolverOptions(**options),
    )
    assert np.asarray(result.solved).all()
    step = 2 * (2 * math.pi * f * DELTA)
    beta = np.asarray(result.gamma) / 1j
    n_g, beta_2, beta_3 = (
        np.asarray(v) for v in (result.n_g, result.beta_2, result.beta_3)
    )
    assert _relative(n_g[1], C0 * (beta[2] - beta[0]) / step) <= 1e-7
    assert _relative(beta_2[1], (n_g[2] - n_g[0]) / (C0 * step)) <= 1e-6
    assert _relative(beta_3[1], (beta_2[2] - beta_2[0]) / step) <= 2e-6
