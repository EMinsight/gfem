"""Eigenmode solves of a model with a permittivity law (ADR 0115).

The frozen seed and residual inverse iteration find the modes of a law
model exactly outside the law's unresolved band. The references:

- a closed PEC box (10 x 8 x 5 mm) filled with the law: each cavity cutoff
  omega_c gives the roots of eps(omega) omega^2 = omega_c^2 (both polariton
  branches of a Lorentz medium, the plasma branch of a Drude medium). With
  the box's analytic cutoffs this is the closed form; with the discrete
  cutoffs of the same space (a vacuum solve of the same box) it is the exact
  spectrum of the discrete nonlinear problem;
- a constant `Material` holding the law's value at a lossless mode.
"""

import math

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._engine import terms

GHZ = 1e9
A, B, D = 10.0, 8.0, 5.0  # the box, mm
C_MM = 299792458.0 * 1e3  # mm/s
ORDER_3 = gfem.SolverOptions(order=3, eigen_tolerance=1e-12)
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=25 * GHZ, damping=0.0)],
)
DRUDE = gfem.PermittivityLaw(
    eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=0.0)]
)
LOSSY_DRUDE = gfem.PermittivityLaw(
    eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=1 * GHZ)]
)


def _box(directory, cells):
    path = directory / "box.msh"
    sm.write_msh22(path, sm.plate(A, B, (("box", D),), cells))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    """The box at (5, 4, 3) cells (h_max 3.28 mm).

    At order 3 its cutoffs below the band (kh < pi, 45 GHz in vacuum) are
    good to 5e-5, and eight of them lie there.
    """
    return _box(tmp_path_factory.mktemp("box"), (5, 4, 3))


@pytest.fixture(scope="module")
def discrete(box):
    """The box's discrete cutoffs (Hz) near 30 GHz: 24 to 42 GHz."""
    vacuum = gfem.solve(
        filled(box, 1.0), gfem.Eigenmode(target=30 * GHZ, n_modes=8), options=ORDER_3
    )
    return np.sort(np.real(np.asarray(vacuum.omega))) / (2 * math.pi)


def filled(mesh, eps_r):
    return gfem.Model(
        mesh,
        materials={"box": gfem.Material(eps_r=eps_r)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )


def cutoffs(limit=80 * GHZ):
    """The box's analytic cavity cutoffs (Hz) below ``limit``, with multiplicity."""
    out = []
    n = 12
    for i in range(n):
        for j in range(n):
            for k in range(n):
                zeros = (i == 0) + (j == 0) + (k == 0)
                if zeros >= 2:
                    continue
                fc = 0.5 * C_MM * math.sqrt((i / A) ** 2 + (j / B) ** 2 + (k / D) ** 2)
                if fc < limit:
                    out += [fc] * (1 if zeros else 2)
    return np.sort(out)


def roots(law, fc):
    """The frequencies (Hz) with eps(f) f^2 = fc^2, as polynomial roots.

    Lorentz (lossless): 2 f^4 ... in x = f^2; Drude: eps_inf f^3 - i g
    eps_inf f^2 - (fp^2 + fc^2) f + i g fc^2 = 0 (e^{+i omega t},
    chi = -fp^2 / (f (f - i g))).
    """
    (term,) = law.terms
    e = float(law.eps_inf)
    if isinstance(term, gfem.Lorentz):
        d, f0 = float(term.delta_eps), float(term.resonance_frequency)
        # (e + d f0^2 / (f0^2 - x)) x = fc^2
        x = np.roots([e, -(e * f0**2 + d * f0**2 + fc**2), fc**2 * f0**2])
        return np.sqrt(x.astype(complex))
    fp, g = float(term.plasma_frequency), float(term.damping)
    found = np.roots([e, -1j * g * e, -(fp**2 + fc**2), 1j * g * fc**2])
    return found[found.real > 0]


def closed_form_error(law, f):
    """The relative distance from ``f`` to the nearest closed-form root."""
    candidates = np.concatenate([roots(law, fc) for fc in np.unique(cutoffs())])
    return float(np.min(np.abs(candidates - f)) / abs(f))


def discrete_error(law, f, discrete):
    """``f``'s distance to an exact root on the discrete cutoffs of the space.

    ``eps(f) f^2`` must be the square of a discrete cutoff (real): one of
    the vacuum box's on the same space.
    """
    fc2 = complex(law(f)) * f * f
    return float(np.min(np.abs(fc2 - discrete**2)) / abs(fc2))


@pytest.mark.parametrize(
    ("law", "target"),
    [
        (LORENTZ, 12 * GHZ),  # lower polariton branch, below the 25 GHz pole
        (LORENTZ, 32 * GHZ),  # upper branch, above eps = 0 at 30.6 GHz
        (DRUDE, 22 * GHZ),
        (LOSSY_DRUDE, 22 * GHZ),
    ],
    ids=["lorentz-lower", "lorentz-upper", "drude", "drude-lossy"],
)
def test_a_filled_box_resonates_at_the_closed_form_branches(
    box, discrete, law, target
):
    result = gfem.solve(
        filled(box, law), gfem.Eigenmode(target=target, n_modes=4), options=ORDER_3
    )
    f = np.asarray(result.omega) / (2 * math.pi)
    for mode in f:
        assert closed_form_error(law, mode) < 1e-4, mode
        assert discrete_error(law, mode, discrete) < 1e-10, mode
    if law is LOSSY_DRUDE:
        assert np.all(f.imag != 0)


def test_a_drude_box_returns_no_mode_at_its_plasma_frequency(box):
    # eps(f_B) = 0 at f_B = fp / sqrt(eps_inf) = 14.14 GHz: every gradient
    # field is an eigenvector there, and none is returned; the transverse
    # modes start at the plasma branch of the lowest cutoff (22.09 GHz)
    f_b = 20 * GHZ / math.sqrt(2.0)
    result = gfem.solve(
        filled(box, DRUDE),
        gfem.Eigenmode(target=1.01 * f_b, n_modes=2),
        options=ORDER_3,
    )
    f = np.real(np.asarray(result.omega)) / (2 * math.pi)
    assert np.all(f > 1.5 * f_b)
    for mode in f:
        assert closed_form_error(DRUDE, mode) < 1e-4


@pytest.fixture
def operators(monkeypatch):
    """The eigen operators the solves build."""
    seen = []
    build = eigen_rule.EigenmodeRule.operator

    def spy(self, op, sigma):
        eop = build(self, op, sigma)
        seen.append(eop)
        return eop

    monkeypatch.setattr(eigen_rule.EigenmodeRule, "operator", spy)
    return seen


def test_a_lossless_drude_law_is_solved_as_a_polynomial_problem(
    box, operators, monkeypatch
):
    # theta = -k^2 chi = k_p^2, a constant in K: the real snap applies
    problem = gfem.Eigenmode(target=22 * GHZ, n_modes=2)
    model = filled(box, DRUDE)
    polynomial = np.asarray(gfem.solve(model, problem, options=ORDER_3).omega)
    assert not operators[-1].nonpolynomial
    assert np.all(polynomial.imag == 0)
    # the general route: the same law as a nonpolynomial piece
    monkeypatch.setattr(terms, "polynomial_law", lambda law: False)
    general = np.asarray(gfem.solve(model, problem, options=ORDER_3).omega)
    assert operators[-1].nonpolynomial
    np.testing.assert_allclose(polynomial, general, rtol=1e-12)
