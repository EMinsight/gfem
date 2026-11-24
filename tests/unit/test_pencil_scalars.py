r"""The real-axis Taylor coefficients of every scalar of the port mode pencil.

Each frequency-dependent scalar :math:`\theta(k_0)` of the port mode pencil
gives :math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real
:math:`k_0`. The oracle is a central difference of the scalar's own value
along real :math:`k_0` (steps :math:`h` and :math:`h/2`,
:math:`h = 5\times10^{-3}k_0`, one Richardson step), compared on the
natural scale :math:`|\theta|/k_0^n`: a thin film's :math:`\theta` is
nearly linear in :math:`k_0`, so its higher coefficients cancel to almost
nothing and a per-coefficient relative error would read noise.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

import gfem
from gfem._engine.line_mode import InverseEpsScalar, LineModeSpace
from gfem._engine.port_mode import (
    ConductorScalar,
    LawEdgeScalar,
    LawScalar,
    PortModeSpace,
)
from gfem._port_section import port_section

DATA = Path(__file__).parent / "data"
LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line2d.msh"

SCALE = 1e-3  # mm
K0 = 2 * math.pi * 15e9 / c * SCALE  # 15 GHz per mm
# measured 4e-9 at worst (Richardson's truncation): 20x margin
TOLERANCE = 1e-7

LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40e9, damping=2e9)],
)
LOSSLESS = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40e9, damping=0.0)],
)
DRUDE = gfem.PermittivityLaw(
    eps_inf=10.0, terms=[gfem.Drude(plasma_frequency=20e9, damping=1e9)]
)
MIXED = gfem.PermittivityLaw(
    eps_inf=3.0,
    terms=[
        gfem.Debye(delta_eps=0.5, relaxation_frequency=8e9),
        gfem.DjordjevicSarkar(delta_eps=0.3, low_frequency=1e3, high_frequency=1e12),
    ],
)
LAWS = {"Lorentz": LORENTZ, "lossless": LOSSLESS, "Drude": DRUDE, "mixed": MIXED}


def richardson(fun, k0, rel=5e-3):
    """:math:`c_1, c_2, c_3` of ``fun`` at ``k0`` by central differences."""

    def stencil(h):
        f = {m: complex(fun(k0 + m * h)) for m in (-2, -1, 0, 1, 2)}
        d1 = (f[1] - f[-1]) / (2 * h)
        d2 = (f[1] - 2 * f[0] + f[-1]) / h**2
        d3 = (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h**3)
        return np.array([d1, d2 / 2, d3 / 6])

    h = rel * k0
    return (4 * stencil(h / 2) - stencil(h)) / 3


def natural_error(taylor, reference, k0):
    """Errors of :math:`c_1..c_3` on the natural scale :math:`|c_0|/k_0^n`."""
    nat = abs(taylor[0])
    return [abs(taylor[n] - reference[n - 1]) / (nat / k0**n) for n in (1, 2, 3)]


def assert_series(scalar, k0=K0):
    taylor = scalar.taylor(k0)
    assert taylor.shape == (4,)
    assert abs(taylor[0] - scalar(k0)) <= 1e-15 * abs(scalar(k0))
    assert max(natural_error(taylor, richardson(scalar, k0), k0)) <= TOLERANCE


@pytest.mark.parametrize("law", LAWS.values(), ids=LAWS.keys())
def test_a_law_piece_gives_its_taylor_coefficients(law):
    assert_series(LawScalar(law, SCALE))


@pytest.mark.parametrize("inverse", [False, True], ids=["Y", "1/Y"])
@pytest.mark.parametrize("law", LAWS.values(), ids=LAWS.keys())
def test_an_absorbing_edge_beside_a_law_gives_its_real_axis_coefficients(law, inverse):
    assert_series(LawEdgeScalar(law, 1.5, SCALE, inverse=inverse))


CONDUCTORS = {"thick": None, "film 1 um": 1e-6, "film 1 nm": 1e-9}


@pytest.mark.parametrize("thickness", CONDUCTORS.values(), ids=CONDUCTORS.keys())
def test_a_conductor_gives_its_taylor_coefficients(thickness):
    assert_series(ConductorScalar(5.8e7, thickness, SCALE))


SIGMA = 0.4 * 2 * math.pi * 15e9 * epsilon_0  # sigma / (omega eps0) = 0.4
HZ = {
    "conductive": (4.0, SIGMA, None),
    "Lorentz": (2.0, 0.0, LORENTZ),
    "Lorentz, conductive": (4.0, SIGMA, LORENTZ),
}


@pytest.mark.parametrize("case", HZ.values(), ids=HZ.keys())
def test_an_hz_inverse_permittivity_gives_its_taylor_coefficients(case):
    assert_series(InverseEpsScalar(*case, SCALE))


# --- the trap: a contour on a scalar that reads Re eps -------------------------------


def contour(fun, k0, radius=0.05, n=64):
    """:math:`c_0..c_3` by a Cauchy integral on a circle around ``k0``."""
    r = radius * abs(k0)
    values = [complex(fun(k)) for k in k0 + r * np.exp(2j * np.pi * np.arange(n) / n)]
    coefficients = np.fft.fft(values) / n
    return coefficients[:4] / r ** np.arange(4)


def test_a_contour_misses_an_absorbing_edge_beside_a_law():
    # Re eps(omega) is not analytic: a complex step reads another function
    # (measured 7.8e-2 here), so the series must stay on the real axis
    edge = LawEdgeScalar(LORENTZ, 1.5, SCALE)
    taylor = edge.taylor(K0)
    assert max(natural_error(taylor, contour(edge, K0)[1:], K0)) >= 1e4 * TOLERANCE
    law = LawScalar(LORENTZ, SCALE)  # analytic: the contour agrees
    assert max(natural_error(law.taylor(K0), contour(law, K0)[1:], K0)) <= TOLERANCE


# --- the pencil ---------------------------------------------------------------------


def space(mesh, materials, boundaries, unit="mm"):
    mesh = gfem.Mesh.from_gmsh(DATA / mesh, length_unit=unit)
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries,
        ports=[gfem.WavePort("p1", "port1")],
    )
    return PortModeSpace(port_section(model, model.ports[0]), 1)


@pytest.mark.parametrize("thickness", [None, 1e-6], ids=["thick", "film"])
def test_a_planar_pencils_first_coefficients_are_its_slopes(thickness):
    pencil = space(
        "parallel_plate.msh",
        {"fill": gfem.Material(eps_r=DRUDE)},
        [gfem.Conductor("plates", conductivity=5.8e7, thickness=thickness)],
        unit="um",
    ).pencil
    k0 = 2 * math.pi * 2e12 / c * 1e-6
    taylor = pencil.taylor(k0)
    assert taylor.shape == (len(pencil.a), 4) == (5, 4)
    np.testing.assert_allclose(taylor[:, 0], pencil.theta(k0), rtol=1e-15, atol=0)
    np.testing.assert_allclose(taylor[:, 1], pencil.d_theta(k0), rtol=1e-14, atol=0)
    assert_series(pencil.extra[0], k0)


def test_a_pencil_with_an_absorbing_edge_beside_a_law_has_a_series_not_a_slope():
    pencil = space(
        "waveguide.msh",
        {"air": gfem.Material(eps_r=LORENTZ)},
        [gfem.Absorbing("walls")],
    ).pencil
    taylor = pencil.taylor(K0)
    assert taylor.shape == (5, 4)
    np.testing.assert_allclose(taylor[:, 0], pencil.theta(K0), rtol=1e-15, atol=0)
    with pytest.raises(NotImplementedError, match="absorbing edge beside"):
        pencil.d_theta(K0)


def test_a_line_pencil_has_a_series_of_each_scalar():
    mesh = gfem.Mesh.from_gmsh(LINE, length_unit="mm")
    law = gfem.Material(eps_r=LORENTZ, conductivity=SIGMA)
    model = gfem.Model(
        mesh,
        materials={"feed": law, "design": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.Absorbing("side")],
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )
    pencil = LineModeSpace(port_section(model, model.ports[0]), 2, "Hz").pencil
    kinds = [type(f) for f in pencil.extra]
    assert kinds == [InverseEpsScalar, LawEdgeScalar]
    taylor = pencil.taylor(K0)
    assert taylor.shape == (5, 4)
    np.testing.assert_allclose(taylor[:, 0], pencil.theta(K0), rtol=1e-15, atol=0)
    np.testing.assert_array_equal(
        taylor[:3], [[1, 0, 0, 0], [-(K0**2), -2 * K0, -1, 0], [1j * K0, 1j, 0, 0]]
    )
