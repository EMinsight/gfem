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
import pytest
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
C0 = 299792458.0
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
