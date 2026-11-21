r"""Perfectly matched layers against their closed-form reflection.

A PEC-backed layer of target reflection :math:`R_0` reflects
:math:`|R| = R_0^{n f \cos\theta / f_\text{ref}}` at the frequency
:math:`f`, the angle :math:`\theta` from its normal and in a background of
index :math:`n` (`gfem._pml.reflection`). The cases end a straight guide
with a layer and read :math:`|S_{11}|` at a wave port on the far end, which
is exact for the guide's mode, so :math:`|S_{11}| = |R|`:

- a TEM line (PEC y walls, PMC x walls): normal incidence;
- a TE10 guide: two plane waves at :math:`\cos\theta = \beta_{10}/k_0`;
- a periodic cell over a layer, read at a Floquet port on its top face:
  oblique incidence under the Bloch wave vector, TE and TM, in vacuum and in
  a filling of index :math:`n = 1.5`, where the vacuum angle refracts to
  :math:`\theta_n` and :math:`|R| = R_0^{n\cos\theta_n}`.

Order 3 on z-layered structured meshes, :math:`h = \lambda/10`
(:math:`\lambda/14` for TE10), where the reflection floor is about
:math:`10^{-6}`; each closed form sits at least 100 times above its floor
and is asserted to 1 % (`pml_cases`).
"""

import math
import warnings

import numpy as np
import pml_cases as pc
import pytest

import gfem
from gfem import _pml

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


def _s11(path, frequencies, pml, eps_r=1.0):
    mesh = gfem.Mesh.from_gmsh(path)
    material = gfem.Material(eps_r=eps_r)
    model = gfem.Model(
        mesh,
        materials={"air": material, "pml": material},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=pml,
    )
    result = gfem.solve(model, gfem.Driven(tuple(frequencies)), options=OPTIONS)
    return np.abs(np.asarray(result.s.values)[:, 0, 0])


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return pc.tem_line(tmp_path_factory.mktemp("pml") / "line.msh")


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    return pc.te10_guide(tmp_path_factory.mktemp("pml") / "guide.msh")


def _close(measured, expected):
    assert abs(measured - expected) <= pc.REL * expected, (measured, expected)


def test_a_layer_behind_a_tem_line_reflects_its_target(line):
    (s11,) = _s11(line, [pc.F0], gfem.PML("pml", target_reflection=1e-2))
    _close(s11, 1e-2)


def test_the_lowest_solve_frequency_is_the_reference(line):
    s11 = _s11(line, [pc.F0, 1.5 * pc.F0], gfem.PML("pml", target_reflection=1e-3))
    _close(s11[0], 1e-3)
    # 10^-4.5 sits only 30 times above the floor: an absolute bound of 3 floors
    assert abs(s11[1] - _pml.reflection(1e-3, 1.5, 1.0, 1.0)) <= 3 * pc.FLOOR


def test_a_pinned_reference_frequency_sets_the_reflection(line):
    pml = gfem.PML("pml", reference_frequency=2 * pc.F0)
    (s11,) = _s11(line, [pc.F0], pml)
    _close(s11, 1e-3)


def test_a_denser_background_reflects_less(line):
    # the stretch is normalized by the vacuum wavenumber: R0^n in index n
    (s11,) = _s11(line, [pc.F0], gfem.PML("pml", target_reflection=1e-2), eps_r=2.25)
    _close(s11, 1e-3)


def _cos_te10(wavelength):
    return math.sqrt(1 - (wavelength / (2 * pc.GUIDE_SECTION[0])) ** 2)


def test_a_te10_guide_ended_by_a_layer(guide):
    (s11,) = _s11(guide, [pc.F0], gfem.PML("pml", target_reflection=1e-3))
    expected = _pml.reflection(1e-3, 1.0, 1.0, _cos_te10(1.0))
    assert expected == pytest.approx(5.807e-3, rel=1e-3)
    _close(s11, expected)


def test_a_guide_near_cutoff_reflects_more(guide):
    (s11,) = _s11(guide, [pc.F0 / 1.4], gfem.PML("pml"))
    expected = _pml.reflection(1e-6, 1.0, 1.0, _cos_te10(1.4))
    assert expected == pytest.approx(7.013e-3, rel=1e-3)
    _close(s11, expected)


def test_a_reference_frequency_at_beta_restores_the_target(guide):
    # f_ref where k0 = beta: the layer meets R0 = 1e-6 again, below the floor
    f = pc.F0 / 1.4
    (s11,) = _s11(guide, [f], gfem.PML("pml", reference_frequency=f * _cos_te10(1.4)))
    assert s11 <= pc.GUIDE_FLOOR


# --- a layer under a periodic cell ------------------------------------------------

#: Incidence angles (degrees, in vacuum) and azimuths, as the Fresnel slab's
#: conical case holds them at 40 degrees.
ANGLES = ((0.0, 0.0), (40.0, 30.0), (60.0, 0.0))
#: Cross-polarized |S| allowed: the Floquet slab's gate.
CROSS_TOL = 1e-5


@pytest.fixture(scope="module", params=[1.0, 2.25], ids=["vacuum", "eps2.25"])
def cell(request, tmp_path_factory):
    """S of the Bloch cell over a layer, at each of `ANGLES`, both polarizations."""
    eps_r = request.param
    path = pc.bloch_cell(
        tmp_path_factory.mktemp("pml") / "cell.msh", cells=pc.CELL_CELLS[eps_r]
    )
    material = gfem.Material(eps_r=eps_r)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"air": material, "pml": material},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
        ],
        ports=[
            gfem.FloquetPort(f"R_{pol}", "z_max", polarization=pol)
            for pol in ("TE", "TM")
        ],
        pml=gfem.PML("pml", target_reflection=1e-2),
    )
    n = math.sqrt(eps_r)
    out = []
    for theta, phi in ANGLES:
        # the vacuum angle theta refracted into the filling
        theta_n = math.asin(math.sin(math.radians(theta)) / n)
        k = model.bloch_wave_vector(
            "z_max", pc.F0, theta=theta_n, phi=math.radians(phi)
        )
        problem = gfem.Driven([pc.F0], excite=["R_TE", "R_TM"], wave_vector=k)
        with warnings.catch_warnings():
            warnings.simplefilter("error", gfem.ObliqueAbsorberWarning)
            result = gfem.solve(model, problem, options=OPTIONS)
        expected = _pml.reflection(1e-2, n * pc.F0, pc.F0, math.cos(theta_n))
        out.append((theta, expected, result.s))
    return out


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_a_layer_under_a_periodic_cell_absorbs_oblique_waves(cell, pol):
    other = "TM" if pol == "TE" else "TE"
    for theta, expected, s in cell:
        # the 1 % gate needs the closed form 100 times above the cell's floor
        assert expected >= 100 * pc.CELL_FLOOR, theta
        _close(abs(complex(s[f"R_{pol}", f"R_{pol}"][0])), expected)
        assert abs(complex(s[f"R_{other}", f"R_{pol}"][0])) <= CROSS_TOL, theta


# --- a leaky guide's resonance ------------------------------------------------------


def _ks(result):
    """The modes' k, rad/mm."""
    return np.asarray(result.omega) / (pc.C0 * 1e3)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "bloch"])
def leaky(request, tmp_path_factory):
    """The leaky guide's modes nearest its resonance: four at R0 = 1e-6, six at 1e-9.

    The second solve looks wider, so that the first's farthest mode is not
    cut from it by a neighbour's small move.
    """
    root = pc.guide_root()
    target = root.real * pc.C0 * 1e3 / (2 * math.pi)
    out = []
    for r0, n_modes in ((1e-6, 4), (1e-9, 6)):
        model, k = pc.leaky_guide_model(
            tmp_path_factory.mktemp("pml") / "guide.msh",
            bloch=request.param,
            target_reflection=r0,
        )
        problem = gfem.Eigenmode(target=target, n_modes=n_modes, wave_vector=k)
        options = gfem.SolverOptions(order=2, linear_solver="direct")
        with warnings.catch_warnings():
            # the far modes trapped below the air's cutoff may take the gain
            # sign (a unit test covers the warning)
            warnings.simplefilter("ignore", gfem.LayerClearanceWarning)
            out.append(gfem.solve(model, problem, options=options))
    return root, out


def test_a_leaky_guide_resonance_matches_its_root(leaky):
    root, (result, stricter) = leaky
    ks = _ks(result)
    nearest = int(np.argmin(np.abs(ks - root)))
    k = ks[nearest]
    assert abs(k - root) <= pc.GUIDE_K_REL * abs(root)
    q_root = abs(root) / (2 * root.imag)
    assert float(result.q[nearest]) == pytest.approx(q_root, rel=pc.GUIDE_Q_REL)
    # every mode returned is the device's: a mode of the layer would move
    # with the stretch
    others = _ks(stricter)
    for z in ks:
        assert np.min(np.abs(others - z)) <= pc.GUIDE_MOVE * abs(z), z


# --- a strip guide through side layers -----------------------------------------------

#: The strip's clearance (um) and the bound on its effective index against a
#: wide closed section, 1e-2 exp(-2 gamma_t c) with the fundamental's
#: transverse decay gamma_t = k0 sqrt(n_eff^2 - n_cl^2) = 3.13 / um.
STRIP_CLEARANCE = 1.0
STRIP_NEFF_TOL = 1.9e-5
STRIP_LENGTH = 1.0
#: Cell size (um) in the cladding and the layer, 0.1 in the core. Measured
#: at order 2, one thread: |S11| = 4.5e-5, |S21 - exp(-gamma L)| = 4.3e-4,
#: n_eff 2.7e-7 off the closed section (at 0.15: 1.3e-4, 1.7e-3, 1.0e-7).
STRIP_H = 0.1
#: A clearance where the layer perturbs the mode itself (a clearance
#: warning) and an unstretched port section mismatches it: with 0.15 um
#: cells |S11| = 2.3e-3 to 6.3e-3 without the section stretch (as more of
#: it is dropped), 1.0e-4 with it, n_eff 5.6e-6 off the closed section.
#: Its effective index bound is 1e-2 exp(-2 gamma_t c).
NEAR_CLEARANCE = 0.4
NEAR_H = 0.15
NEAR_NEFF_TOL = 8e-4


def _strip_n_eff(gamma):
    k0 = 2 * math.pi * pc.STRIP_F / pc.C0
    return complex(gamma).imag / k0


@pytest.fixture(scope="module")
def strip_n_eff(tmp_path_factory):
    """The fundamental's effective index on a wide closed section."""
    path = pc.strip_guide(
        tmp_path_factory.mktemp("strip") / "closed.msh",
        2.0,
        thickness=0.0,
        length=0.15,
        clad_h=STRIP_H,
    )
    reference = gfem.solve(
        pc.strip_model(path),
        gfem.PortMode("p1", (pc.STRIP_F,)),
        options=gfem.SolverOptions(order=2),
    )
    return _strip_n_eff(np.asarray(reference.gamma)[0])


def test_a_strip_guide_through_side_layers(tmp_path, strip_n_eff):
    # both port faces cut the side layers: their sections take the stretch,
    # so each port stays exact for its own mode
    path = pc.strip_guide(
        tmp_path / "strip.msh", STRIP_CLEARANCE, length=STRIP_LENGTH, clad_h=STRIP_H
    )
    model = pc.strip_model(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.LayerClearanceWarning)
        result = gfem.solve(
            model,
            gfem.Driven((pc.STRIP_F,)),
            options=gfem.SolverOptions(order=2, linear_solver="direct"),
        )
    s = np.asarray(result.s.values)[0]
    gamma = complex(np.asarray(result.port_modes["p1"].gamma)[0])
    assert abs(s[0, 0]) <= 2e-4
    assert abs(s[1, 0] - np.exp(-gamma * STRIP_LENGTH * 1e-6)) <= 3e-3
    assert abs(s[0, 1] - s[1, 0]) <= 1e-8
    # a kappa = 1 layer gives the guided mode a gain-sign Re gamma < 0 of
    # order 1e-2 exp(-2 gamma_t c) k0 (measured -3.9e-6 k0): the modal
    # waves then carry |exp(-gamma L)|^2 = 1 + 3.2e-5, not 1 (measured sum
    # 1 + 2.7e-5)
    budget = abs(np.exp(-gamma * STRIP_LENGTH * 1e-6)) ** 2
    assert abs(s[0, 0]) ** 2 + abs(s[1, 0]) ** 2 <= budget + 1e-8
    assert abs(_strip_n_eff(gamma) - strip_n_eff) <= STRIP_NEFF_TOL


def test_a_strip_guide_close_to_its_side_layers(tmp_path, strip_n_eff):
    # the layer perturbs the mode at this clearance: an unstretched port
    # section would solve a different mode than the guide carries and
    # reflect it, the stretched one stays matched
    path = pc.strip_guide(
        tmp_path / "strip.msh", NEAR_CLEARANCE, length=STRIP_LENGTH, clad_h=NEAR_H
    )
    with pytest.warns(gfem.LayerClearanceWarning, match="clearance"):
        result = gfem.solve(
            pc.strip_model(path),
            gfem.Driven((pc.STRIP_F,)),
            options=gfem.SolverOptions(order=2, linear_solver="direct"),
        )
    s = np.asarray(result.s.values)[0]
    gamma = complex(np.asarray(result.port_modes["p1"].gamma)[0])
    assert abs(s[0, 0]) <= 5e-4
    assert abs(_strip_n_eff(gamma) - strip_n_eff) <= NEAR_NEFF_TOL
