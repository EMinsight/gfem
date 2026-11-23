r"""Resonances of guides closed by wave ports, against exact roots and the driven sweep.

The guides are those of `wave_port_cases` (full-cross-section layers, so
each mode family is an exact transmission line): a resonance's complex
frequency converges with the order to the line's root. Gates are pinned
at 3x to 30x the measured errors, written beside each.

- A (Q 5.5) and A4 (Q 869), one port: the Robin term at a complex
  wavenumber, orders 2 to 4.
- C, a port surface carrying TE10 and TE20: the TE20 family needs the
  surface's modal correction; each family is told by the field's parity
  about :math:`x = a/2`.
- B3, two port surfaces: the root, and the ports' coupling Q in the ratio of
  their shares of the outgoing power.
- L, a lossy anisotropic port section; D, a coax (TEM); A3, a root just
  above the port's cutoff; E, a circular guide on TM01, a port mode with
  :math:`E_n \ne 0`.
- The driven cross-check: eigen and driven solves share the mesh, the port
  modes and the operator, so the pole of a one-pole fit of the driven S
  (degree-6 numerator, 41 points over one half-width each side) matches the
  eigen solve's root to the port mode tolerance (1e-10 on both sides), not
  to the discretization error. On A4, C and H, whose hybrid, dispersive
  port mode has no exact value.
"""

import numpy as np
import pytest
import wave_port_cases as w
from resonance import fit_resonance

import gfem
from gfem.errors import PortModeError


def direct(order, **options):
    return gfem.SolverOptions(order=order, linear_solver="direct", **options)


def eigen(model, target, n_modes, order, **options):
    problem = gfem.Eigenmode(target=target, n_modes=n_modes)
    return gfem.solve(model, problem, options=direct(order, **options))


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    directory = tmp_path_factory.mktemp("guides")
    return lambda name, build, *args, **kw: build(
        directory / f"{name}.msh", *args, **kw
    )


def lone_port(meshes, name, layers):
    mesh = meshes(name, w.ported_guide, layers, left_port=False)
    return w.ported_model(mesh, layers, [gfem.WavePort("p1", "port")])


# --- a lone port's Robin term -------------------------------------------------

# name: layers, root, target, (e_3, e_4, Im e_3) gates
LONE = {
    # measured e 1.2e-3 / 7.5e-6 / 3.3e-8, Im e_3 4.7e-5
    "A": (w.CASE_A, w.ROOT_A, 11.67e9, (1e-4, 3e-7, 1e-3)),
    # measured e 3.7e-4 / 3.6e-6 / 3.5e-8, Im e_3 4.5e-4
    "A4": (w.CASE_A4, w.ROOT_A4, 9.97e9, (1e-4, 3e-7, 1e-3)),
}


@pytest.mark.parametrize("name", list(LONE))
def test_a_lone_port_s_resonance_converges_with_order(meshes, name):
    layers, root, target, (gate_3, gate_4, gate_im) = LONE[name]
    model = lone_port(meshes, name, layers)
    errors, im = [], None
    for order in (2, 3, 4):
        result = eigen(model, target, 3 if name == "A" else 1, order)
        error, j = w.nearest_error(result, root)
        errors.append(error)
        if order == 3:
            im = w.im_error(result, j, root)
    e = errors
    assert e[0] > 10 * e[1] > 100 * e[2], e
    assert e[1] <= gate_3, e
    assert e[2] <= gate_4, e
    assert im <= gate_im, im


# --- a shared port surface (the modal correction at a complex wavenumber) --------


@pytest.fixture(scope="module")
def case_c(meshes):
    mesh = meshes("c", w.ported_guide, w.CASE_A3, left_port=False, b_mm=w.C_B_MM)
    return w.ported_model(mesh, w.CASE_A3, w.SHARED_PORTS)


def parity(result, j):
    """``E_y(3a/4) / E_y(a/4)`` of mode ``j`` in C's dielectric: +1 TE10, -1 TE20."""
    a, b = w.A_MM * 1e-3, w.C_B_MM * 1e-3
    points = [[a / 4, b / 2, 15e-3], [3 * a / 4, b / 2, 15e-3]]
    ey = np.asarray(result.e_field(j).at(points))[:, 1]
    return ey[1] / ey[0]


# family: root, target, parity, gates on e and Im e
FAMILIES = {
    # measured e 3.4e-5, Im 2.3e-4
    "te20": (w.ROOT_C20, 15.2e9, -1, (1e-4, 3e-3)),
    # from its own target: measured e 4.6e-6, Im 2.9e-5
    "te10": (w.ROOT_C10, 14.12e9, 1, (1e-4, 1e-3)),
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_each_family_of_a_shared_port_surface_meets_its_root(case_c, family):
    root, target, sign, (gate, gate_im) = FAMILIES[family]
    result = eigen(case_c, target, 3, 3)
    error, j = w.nearest_error(result, root)
    assert error <= gate
    assert w.im_error(result, j, root) <= gate_im
    assert abs(parity(result, j) - sign) <= 1e-2


def test_a_shared_port_surface_s_te20_root_at_order_4(case_c):
    result = eigen(case_c, 15.2e9, 3, 4)
    error, j = w.nearest_error(result, w.ROOT_C20)
    assert error <= 3e-6  # 5.6e-7
    assert w.im_error(result, j, w.ROOT_C20) <= 1e-4  # 4.7e-6


# --- two port surfaces, a lossy port section, TEM, near cutoff ------------------


def test_two_port_surfaces_meet_the_root_and_split_the_coupling_q(meshes):
    mesh = meshes("b3", w.ported_guide, w.CASE_B3, left_port=True)
    result = eigen(w.ported_model(mesh, w.CASE_B3, w.TWO_PORTS), 9.97e9, 3, 3)
    error, j = w.nearest_error(result, w.ROOT_B3)
    assert error <= 1e-4  # 6.9e-6
    q = {name: float(v[j]) for name, v in result.q_port.items()}
    # the shares are exact at the port planes; the coupling Q follows them to
    # O(1/Q) (Q 52): measured 2.4e-4 off
    share = w.SHARE_B3["left"] / w.SHARE_B3["right"]
    assert q["right"] / q["left"] == pytest.approx(share, rel=2e-2)


def test_a_lossy_anisotropic_port_section_meets_its_root(meshes):
    # Q 2.8: twelve modes, since higher-Q modes of other families lie nearer
    # the real target
    result = eigen(lone_port(meshes, "l", w.CASE_L), 11.67e9, 12, 3)
    error, j = w.nearest_error(result, w.ROOT_L)
    assert error <= 1e-4  # 7.9e-6
    assert w.im_error(result, j, w.ROOT_L) <= 1e-3  # 4.4e-5


def test_a_coax_s_tem_roots_meet_the_closed_form(meshes):
    mesh = meshes("d", w.coax_guide, w.CASE_A, n_r=4, n_theta=8)
    model = w.ported_model(
        mesh,
        w.CASE_A,
        [gfem.WavePort("p1", "port")],
        boundaries=(gfem.PEC("walls"), gfem.PMC("symmetry")),
    )
    result = eigen(model, 7.5e9, 2, 3)
    roots = w.coax_roots(4.0, 10.0, (0, 1))
    found = []
    for root in roots:
        error, j = w.nearest_error(result, root)
        assert error <= 1e-4  # 4.7e-6 and 1.7e-6
        found.append(w.ghz(result.omega[j]))
    # every root of the line has the same Im f: 8.7e-6 apart
    assert abs(found[0].imag - found[1].imag) <= 1e-4 * roots[0].imag


def test_a_root_just_above_the_port_s_cutoff_meets_the_exact_root(meshes):
    result = eigen(lone_port(meshes, "a3", w.CASE_A3), 7.03e9, 1, 3)
    error, _ = w.nearest_error(result, w.ROOT_A3)
    assert error <= 3e-6  # 3.5e-7


# The port absorbs TM01 only, so the guide's other families ring at high Q
# around the target, nearer it than the low-Q TM01 root (Q 3.7): 8 modes
# reach it (7 do not).
def test_a_port_mode_with_a_normal_field_meets_the_line_root(meshes):
    # straight-sided mesh of the disk (16 x 16): the reference takes the
    # port's discrete cutoff, so only the 3D discretization error remains
    mesh = meshes("e", w.circular_guide, w.CASE_A, n=16)
    model = w.ported_model(mesh, w.CASE_A, [gfem.WavePort("p1", "port", mode=3)])
    f0 = 13.4e9
    modes = gfem.solve(model, gfem.PortMode("p1", f0), options=direct(3))
    k0 = 2 * np.pi * f0 / w.C0
    gamma = complex(np.asarray(modes.gamma).ravel()[0])
    kc = float(np.sqrt((gamma**2 + k0**2).real))
    root = w.line_root("TM", kc, [(4.0, 10.0), (1.0, 5.0)], 13.4 + 1.8j)
    result = eigen(model, f0, 8, 3)
    error, _ = w.nearest_error(result, root)
    assert error <= 3e-4  # 2.3e-6


# --- the driven cross-check -----------------------------------------------------

TIGHT = 1e-10  # port mode tolerance of both solves


def driven_pole(model, pole, entry, order):
    """The pole (Hz) of a fit of S[entry, entry] over ``pole`` +- one half-width."""
    half = pole.imag
    f = np.linspace(pole.real - half, pole.real + half, 41)
    driven = gfem.Driven(tuple(f))
    s = np.asarray(
        gfem.solve(
            model, driven, options=direct(order, port_mode_tolerance=TIGHT)
        ).s.values
    )
    fit = fit_resonance(f, s[:, entry, entry], degree=6)
    return fit.pole if fit.pole.imag > 0 else fit.pole.conjugate()


def cross_check(model, target, n_modes, order, *, root=None, entry=0):
    result = eigen(model, target, n_modes, order, port_mode_tolerance=TIGHT)
    j = 0 if root is None else w.nearest_error(result, root)[1]
    pole = complex(result.omega[j]) / (2 * np.pi)
    fitted = driven_pole(model, pole, entry, order)
    return abs(fitted - pole) / abs(pole), abs(fitted.imag - pole.imag) / pole.imag


def test_a_lone_port_s_root_is_the_driven_pole(meshes):
    # A4 at order 3: measured 1.1e-13, Im 1.9e-10 (fit bias 2e-14)
    error, im = cross_check(lone_port(meshes, "a4", w.CASE_A4), 9.97e9, 1, 3)
    assert error <= 1e-12
    assert im <= 2e-9


def test_a_shared_port_surface_s_root_is_the_driven_pole(meshes):
    # mirrored in x and y, so families of other parities do not couple to
    # TE20: on the plain mesh two of them (15.173 and 15.237 GHz, Q 5e4 and
    # 200) lie within the window and bias the one-pole fit to 1.5e-5.
    # Measured 4.8e-9, Im 3.5e-7: the fit's own bias on the exact S
    mesh = meshes(
        "c_mirrored",
        w.ported_guide,
        w.CASE_A3,
        left_port=False,
        b_mm=w.C_B_MM,
        mirror=(True, True, False),
    )
    model = w.ported_model(mesh, w.CASE_A3, w.SHARED_PORTS)
    error, im = cross_check(model, 15.2e9, 3, 3, root=w.ROOT_C20, entry=1)
    assert error <= 5e-8
    assert im <= 5e-6


@pytest.fixture(scope="module")
def case_h(meshes):
    return w.hybrid_model(meshes("h", w.hybrid_guide))


def test_the_hybrid_guide_carries_one_propagating_mode_at_its_pole(case_h):
    # its second mode's cutoff is near 9.65 GHz, so up to there S11 of the
    # lossless guide has modulus one and the fit sees a single port mode
    with pytest.raises(PortModeError, match=r"carries 1 propagating mode"):
        gfem.solve(case_h, gfem.PortMode("p", 7.3e9, mode=2), options=direct(2))


def test_a_hybrid_dispersive_port_s_root_is_the_driven_pole(case_h):
    # order 2, pole 7.2964 + 0.2146i GHz: measured 5.1e-8, Im 1.5e-7
    error, im = cross_check(case_h, 7.3e9, 1, 2)
    assert error <= 3e-7
    assert im <= 1.5e-6
