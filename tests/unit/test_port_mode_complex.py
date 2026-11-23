"""A wave port surface's term at a complex frequency, from its port mode sweep.

An eigenmode problem with wave ports needs each surface's propagation
constant and modal correction at complex wavenumbers. They come from the
Rayleigh-Ritz model of a real port mode sweep over a band around the target,
checked against exact 2D solves at the complex wavenumber, against the
driven coupling on the real grid, and against difference quotients.
"""

import cmath
import math
from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem._engine.driven import driven_operator
from gfem._engine.port_term import outgoing, port_terms
from gfem._engine.wave_ports import wave_port_couplings
from gfem._linalg.superlu import shift_invert_eigs
from gfem.errors import PortModeError

DATA = Path(__file__).parent / "data"
C0 = 299792458.0
A = 22.86e-3  # WR-90 width, m
QS = (10.0, 1e2, 1e3, 1e4)


def wavenumber(f, scale=1e-3):
    """The wavenumber per file unit of a frequency in Hz."""
    return 2 * math.pi * f / C0 * scale


def terms_of(model, order, target, tolerance=1e-6):
    op = driven_operator(model, order)
    return port_terms(
        model, op.built, op.space, order=order, target=target, tolerance=tolerance
    )


def relative(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# --- cross-sections -----------------------------------------------------------


@pytest.fixture(scope="module")
def slab_mesh():
    return gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")


def slab(mesh, eps_r=4.0, tan_delta=0.0):
    # WR-90 with an eps_r = 4 slab: gamma has no closed form
    return gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=eps_r, tan_delta=tan_delta),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


@pytest.fixture(scope="module", params=[0.0, 1e-2], ids=["lossless", "lossy"])
def slab_term(request, slab_mesh):
    (term,) = terms_of(slab(slab_mesh, tan_delta=request.param), 2, 10e9)
    return term


@pytest.fixture(scope="module")
def hybrid_mesh():
    return gfem.Mesh.from_gmsh(DATA / "hybrid_waveguide.msh", length_unit="mm")


def shared_hybrid(mesh, tan_delta=0.0):
    # WR-90 with an eps_r = 6 block: two hybrid modes (E_n != 0) on one
    # surface, so the modal correction is far from zero
    return gfem.Model(
        mesh,
        materials={
            "block": gfem.Material(eps_r=6.0, tan_delta=tan_delta),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[
            gfem.WavePort("m1", "port1", mode=1),
            gfem.WavePort("m2", "port1", mode=2),
        ],
    )


HYBRID_TARGET = 18e9


def hybrid_term(mesh, tan_delta=0.0, tolerance=1e-6):
    model = shared_hybrid(mesh, tan_delta)
    (term,) = terms_of(model, 3, HYBRID_TARGET, tolerance)
    return term


# --- exact 2D solves at a complex wavenumber ------------------------------------


def exact_modes(term, k, value):
    """Each port's exact 2D pair at ``k``, nearest the evaluator's eigenvalue."""
    a, b = term.modes.pencil.at(k)
    out = []
    for gamma in value.gamma:
        lam = gamma**2
        lams, vecs = shift_invert_eigs(a, b, lam, 4)
        i = int(np.argmin(np.abs(lams - lam)))
        out.append((complex(lams[i]), vecs[:, i]))
    return out


def exact_correction(term, k, value):
    r"""The modal correction on the port face's trace DOFs from exact modes.

    :math:`H R_\mathrm{sym}^{-1} H^\mathsf{T} - (\gamma_0 S)(\gamma_0
    S^\mathsf{T} E)^{-1}(\gamma_0 S)^\mathsf{T}`, the driven definition with
    each mode's n x H :math:`H`, trace :math:`E` and Robin trace :math:`S`.
    """
    modes = term.modes
    g_form, t_form = modes.coupling_forms
    tangential = modes.free < modes.n_t
    h, s, e, gammas = [], [], [], []
    for lam, x in exact_modes(term, k, value):
        gamma = outgoing(lam)
        xt, xn = np.where(tangential, x, 0), np.where(tangential, 0, x)
        h.append(t_form @ xt + g_form @ xn)
        s.append(t_form @ xt / gamma)
        full = modes.full(xt / gamma)
        e.append(full[: modes.n_t])
        gammas.append(gamma)
    h, s, e = (np.stack(v, axis=1) for v in (h, s, e))
    g0s = gammas[value.lead] * s
    r = h.T @ e
    return h @ np.linalg.solve((r + r.T) / 2, h.T) - g0s @ np.linalg.solve(
        g0s.T @ e, g0s.T
    )


def reduced_correction(term, value):
    """:math:`W` on the port face's trace DOFs from the evaluator's ``C``."""
    modes = term.modes
    w = modes.reduce(term._reduced.v).w
    basis = np.zeros((modes.space.ndof, w.shape[1]))
    basis[modes.free] = w
    r = modes.port_columns(basis)["r"]
    return r @ value.c @ r.T


@pytest.mark.parametrize("f", [8e9, 10e9, 11.9e9])
def test_gamma_0_matches_an_exact_2d_solve_at_a_complex_wavenumber(slab_term, f):
    # measured <= 1.3e-13 (probe, lossy at Q 10)
    for q in QS:
        k = wavenumber(f) * (1 + 0.5j / q)
        value = slab_term.at(k)
        ((lam, _),) = exact_modes(slab_term, k, value)
        assert abs(value.gamma0**2 - lam) <= 1e-12 * abs(lam)


@pytest.fixture(scope="module", params=[0.0, 1e-2], ids=["lossless", "lossy"])
def hybrid(request, hybrid_mesh):
    return hybrid_term(hybrid_mesh, request.param)


@pytest.mark.parametrize("f", [14e9, 18e9, 23e9])
def test_the_correction_matches_exact_modes_to_the_guards_error(hybrid, f):
    # The modal correction reads the modes' shapes, whose error is first order
    # in the reduced backward error (gamma's is second order): measured
    # |dW| / |W| <= 4.3 x the error (1e-8 to 3e-7 at the default tolerance).
    for q in QS:
        k = wavenumber(f) * (1 + 0.5j / q)
        value = hybrid.at(k)
        assert value.error <= hybrid.tolerance
        for gamma, (lam, _) in zip(
            value.gamma, exact_modes(hybrid, k, value), strict=True
        ):
            assert abs(gamma**2 - lam) <= 1e-12 * abs(lam)  # measured 3.5e-15
        exact = exact_correction(hybrid, k, value)
        error = relative(reduced_correction(hybrid, value), exact)
        assert error <= 10 * max(value.error, 1e-14)


def test_a_forced_guard_failure_enriches_the_basis_and_keeps_the_values(hybrid_mesh):
    # at tolerance 1e-12 no reduced pair off the real grid passes
    term = hybrid_term(hybrid_mesh, tolerance=1e-12)
    size = term.vectors.shape[1]
    for f in (14e9, 23e9):
        k = wavenumber(f) * (1 + 0.5j / 10)
        value = term.at(k)
        assert value.version == term.version > 0
        assert value.c.shape == (term.vectors.shape[1],) * 2
        assert value.error <= 1e-12
        for gamma, (lam, _) in zip(
            value.gamma, exact_modes(term, k, value), strict=True
        ):
            assert abs(gamma**2 - lam) <= 1e-12 * abs(lam)
        exact = exact_correction(term, k, value)
        # measured 4.3e-14 and 9.0e-13
        assert relative(reduced_correction(term, value), exact) <= 1e-11
    assert term.vectors.shape[1] > size
    assert len(term.enrichments) == term.version


# --- the real grid ----------------------------------------------------------------


@pytest.mark.parametrize("case", ["slab", "shared hybrid"])
def test_on_the_real_grid_the_term_is_the_driven_couplings(
    case, slab_mesh, hybrid_mesh
):
    if case == "slab":
        model, order, target = slab(slab_mesh, tan_delta=1e-2), 2, 10e9
    else:
        model, order, target = shared_hybrid(hybrid_mesh), 3, HYBRID_TARGET
    op = driven_operator(model, order)
    (term,) = port_terms(
        model, op.built, op.space, order=order, target=target, tolerance=1e-6
    )
    (surface,) = wave_port_couplings(
        model,
        op.built,
        op.space,
        order=order,
        frequency=tuple(term.band),
        tolerance=1e-6,
    )
    np.testing.assert_array_equal(term.vectors, surface.vectors)
    scale = term.modes.section.scale
    lead = surface.lead
    for j, f in enumerate(term.band):
        value = term.at(wavenumber(f, scale))
        c = surface.correction_coefficients(2 * math.pi * f)
        # the same tabulated modes: rounding only
        assert np.linalg.norm(value.c - c) <= 1e-14 * max(np.linalg.norm(c), 1.0)
        gamma0 = lead.sweep.gamma[j] * scale
        assert abs(value.gamma0 - gamma0) <= 1e-14 * abs(gamma0)
    assert term.version == 0


# --- derivatives ------------------------------------------------------------------


def law_slab(mesh):
    law = gfem.PermittivityLaw(
        eps_inf=3.5,
        terms=[gfem.Lorentz(delta_eps=0.5, resonance_frequency=30e9, damping=2e9)],
    )
    return slab(mesh).with_material("slab", gfem.Material(eps_r=law))


@pytest.fixture(scope="module")
def plate_mesh():
    return gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")


def conductor_plate(mesh):
    # a TEM line whose plates' surface impedance enters the port section
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.Conductor("plates", conductivity=5.8e7)],
        ports=[gfem.WavePort("p1", "port1")],
    )


def central(fn, k, h):
    return (fn(k + h) - fn(k - h)) / (2 * h)


@pytest.mark.parametrize(
    "case", ["shared hybrid", "lossy shared hybrid", "law", "conductor"]
)
def test_derivatives_match_central_differences(
    case, hybrid_mesh, slab_mesh, plate_mesh
):
    if case.endswith("shared hybrid"):
        term = hybrid_term(hybrid_mesh, 1e-2 if case.startswith("lossy") else 0.0)
        f, scale = HYBRID_TARGET, 1e-3
    elif case == "law":
        (term,) = terms_of(law_slab(slab_mesh), 2, 10e9)
        f, scale = 10e9, 1e-3
    else:
        (term,) = terms_of(conductor_plate(plate_mesh), 2, 2e12)
        f, scale = 2e12, 1e-6
    for q in (5.0, 50.0):
        k = wavenumber(f, scale) * (1 + 0.5j / q)
        value = term.at(k)
        h = 1e-5 * abs(k)
        plus, minus = term.at(k + h), term.at(k - h)
        assert plus.version == minus.version == value.version
        d_gamma = (plus.gamma - minus.gamma) / (2 * h)
        d_c = (plus.c - minus.c) / (2 * h)
        # measured 3e-11 to 1.2e-10 (probe): the difference quotient's floor
        assert relative(value.d_gamma, d_gamma) <= 1e-8
        assert np.linalg.norm(value.d_c - d_c) <= 1e-8 * max(
            np.linalg.norm(d_c), np.linalg.norm(value.c)
        )
        for c in (value.c, value.d_c):
            assert np.linalg.norm(c - c.T) <= 1e-12 * np.linalg.norm(value.c)


def test_the_robin_piece_is_gamma_0_as_a_nonpolynomial_port_term(slab_mesh):
    model = slab(slab_mesh)
    op = driven_operator(model, 2)
    (term,) = port_terms(
        model, op.built, op.space, order=2, target=10e9, tolerance=1e-6
    )
    piece = term.robin_piece(model, op.built, 2)
    assert piece.kind == "port"
    assert piece.frequency_class == "nonpolynomial"
    omega = 2 * math.pi * 10e9 * (1 + 0.05j)
    value = term.at(omega * 1e-3 / C0)  # k per mm
    assert piece.theta(omega) == pytest.approx(value.gamma0, rel=1e-13)
    assert piece.d_theta(omega) == pytest.approx(value.d_gamma0 * 1e-3 / C0, rel=1e-13)


# --- the branch -------------------------------------------------------------------


def test_the_outgoing_branch_is_the_principal_root_at_a_decaying_mode(slab_term):
    for f in (8e9, 11.9e9):
        real = slab_term.at(wavenumber(f)).gamma0
        for q in QS:
            gamma = slab_term.at(wavenumber(f) * (1 + 0.5j / q)).gamma0
            principal = 1j * cmath.sqrt(-(gamma**2))
            assert abs(gamma - principal) <= 1e-14 * abs(gamma)
            # the decay makes the field grow away from the model
            assert gamma.real < real.real


def test_the_outgoing_branch_is_continuous_across_the_real_axis_below_cutoff(
    slab_term,
):
    # below the cutoff (about 4.5 GHz here) the principal root's cut lies on
    # the real axis; the rotated branch crosses it smoothly
    k = wavenumber(3e9)
    delta = 1e-8 * k
    above, below = slab_term.at(k + 1j * delta), slab_term.at(k - 1j * delta)
    assert (above.gamma0**2).real > 0  # below cutoff
    jump = abs(above.gamma0 - below.gamma0)
    assert jump <= 4 * abs(above.d_gamma0) * delta
    # the anti-bound side: the field grows away from the model
    assert above.gamma0.real < -100 * abs(above.gamma0.imag)
    lam = above.gamma0**2
    if abs(lam.imag) <= 1e-6 * abs(lam):  # lossless: lam on the principal cut
        principal = 1j * cmath.sqrt(-lam)
        mirror = 1j * cmath.sqrt(-(below.gamma0**2))
        assert abs(principal - mirror) > abs(lam) ** 0.5


def test_the_rotated_cut_lies_on_the_negative_imaginary_axis():
    # in beta^2 = -lam (eps mu k0^2 - kc^2): the cut is beta^2 in -i R+
    beta = 2.0
    assert outgoing(-(beta**2)) == pytest.approx(2j, abs=1e-15)
    assert outgoing(4.0) == pytest.approx(-2.0, abs=1e-15)
    t = 3.0
    across = outgoing(1j * t * (1 + 1e-12j)) - outgoing(1j * t * (1 - 1e-12j))
    assert abs(across) > 1.0
    for beta2 in (-t, t, 1j * t):  # continuous everywhere else
        lam = -beta2
        assert abs(outgoing(lam * (1 + 1e-12j)) - outgoing(lam * (1 - 1e-12j))) < 1e-10


# --- the band and the refusals -----------------------------------------------------


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


def air_guide(mesh, walls=None, material=None):
    return gfem.Model(
        mesh,
        materials={"air": material or gfem.Material()},
        boundaries=[walls or gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def test_the_band_starts_above_the_cutoff_near_the_target(guide_mesh):
    (term,) = terms_of(air_guide(guide_mesh), 2, 8e9)
    cutoff = C0 / (2 * A)  # TE10, 6.557 GHz
    assert len(term.band) == 9
    assert term.band[-1] == pytest.approx(1.4 * 8e9)
    assert term.band[0] == pytest.approx(1.05 * cutoff, rel=3e-3)


def test_the_band_starts_at_0_7_target_far_above_the_cutoff(slab_term):
    assert slab_term.band[0] == pytest.approx(7e9)
    assert slab_term.band[-1] == pytest.approx(14e9)


def test_a_target_below_the_cutoff_raises(guide_mesh):
    with pytest.raises(PortModeError, match=r"'p1'.*does not propagate at the eigen"):
        terms_of(air_guide(guide_mesh), 2, 6e9)


def test_an_absorbing_edge_beside_a_law_raises_at_a_complex_wavenumber(guide_mesh):
    law = gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[gfem.Lorentz(delta_eps=0.2, resonance_frequency=40e9, damping=1e9)],
    )
    model = air_guide(guide_mesh, gfem.Absorbing("walls"), gfem.Material(eps_r=law))
    (term,) = terms_of(model, 2, 10e9)
    with pytest.raises(NotImplementedError, match=r"'p1'.*absorbing edge beside"):
        term.at(wavenumber(10e9) * (1 + 0.05j))
