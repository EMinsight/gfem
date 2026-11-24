r"""Each library material against its source, as the paper prints it.

The source formulas are typed here from the papers, in their own units
(wavelength in um, photon energy in eV), independently of the library's
conversion to Hz:

- Si: Li 1980 eq. (22) at 293 K; the library's lossless refit holds within
  its stated :math:`8.3\times10^{-4}` in :math:`n`, inside Li's
  :math:`\pm 2\times10^{-3}`.
- SiO2: Malitson 1965's Sellmeier, converted exactly.
- SiN: Luke 2015's printed formula; the refit holds within
  :math:`7.2\times10^{-5}`.
- LN: Zelmon 1997's congruent ordinary and extraordinary Sellmeier,
  converted exactly; Turner 1966's clamped tensor; Smith and Welsh 1971's
  clamped static permittivity, printed in nF/m.
- Au, Cu, Al: Rakic 1998's Lorentz-Drude model (Table 2, eV), converted
  exactly, and ten-row CC0 excerpts of refractiveindex.info's tables of
  that model (``data/*_rakic1998_ld.csv``), which are written under
  :math:`e^{-i\omega t}` with :math:`n + ik`: the extinction coefficient
  :math:`k \ge 0` does not depend on the convention, so they check the sign
  of the law's loss.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, h

from gfem import materials

DATA = Path(__file__).parent / "data"
X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def _hz(wavelength_um):
    return c / (np.asarray(wavelength_um) * 1e-6)


def _n(law, wavelength_um):
    return np.sqrt(np.asarray(law(_hz(wavelength_um)), complex))


def _sellmeier(wavelength_um, b, lam2):
    """:math:`n` of :math:`n^2 = 1 + \\sum B_i \\lambda^2 / (\\lambda^2 - C_i)`."""
    l2 = np.asarray(wavelength_um) ** 2
    return np.sqrt(1 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, lam2, strict=True)))


def _li_1980(wavelength_um):
    """Li's eq. (22) at 293 K.

    :math:`n^2 = \\varepsilon(T) + L(T) A(T) / \\lambda^2`; :math:`L(293) =
    1.0005` reproduces Li's own table (3.4757 at 1.55 um, 3.5167 at 1.2 um).
    """
    t = 293.0
    eps = 11.4445 + 2.7739e-4 * t + 1.7050e-6 * t**2 - 8.1347e-10 * t**3
    a = 0.8948 + 4.3977e-4 * t + 7.3835e-8 * t**2
    return np.sqrt(eps + 1.0005 * a / np.asarray(wavelength_um) ** 2), eps


def test_silicon_against_li_1980():
    law = materials.silicon().eps_r
    lam = np.geomspace(1.2, 14.0, 500)
    li, static = _li_1980(lam)
    assert _li_1980(1.55)[0] == pytest.approx(3.4757, abs=5e-5)
    assert np.max(np.abs(_n(law, lam) - li)) <= 8.3e-4
    assert float(np.real(law(0.0))) == pytest.approx(static, rel=1e-3)
    assert static == pytest.approx(11.65, abs=5e-3)


def test_silica_against_malitson_1965():
    lam = np.geomspace(0.21, 3.71, 500)
    malitson = _sellmeier(
        lam,
        (0.6961663, 0.4079426, 0.8974794),
        (0.0684043**2, 0.1162414**2, 9.896161**2),
    )
    assert np.max(np.abs(_n(materials.silica().eps_r, lam) - malitson)) <= 1e-12


def test_silicon_nitride_against_luke_2015():
    lam = np.geomspace(0.31, 5.504, 500)
    luke = _sellmeier(lam, (3.0249, 40314.0), (0.1353406**2, 1239.842**2))
    law = materials.silicon_nitride().eps_r
    assert np.max(np.abs(_n(law, lam) - luke)) <= 7.2e-5


def test_lithium_niobate_against_zelmon_1997():
    lam = np.geomspace(0.4, 5.0, 20)
    ordinary = _sellmeier(lam, (2.6734, 1.2290, 12.614), (0.01764, 0.05914, 474.60))
    extraordinary = _sellmeier(lam, (2.9804, 0.5981, 8.9543), (0.02047, 0.0666, 416.08))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the range's own edges are inside it
        eps = np.array(
            [materials.lithium_niobate(frequency=f, axes=X_CUT).eps_r for f in _hz(lam)]
        )
    np.testing.assert_allclose(np.sqrt(eps[:, 0]), ordinary, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(eps[:, 1]), ordinary, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(eps[:, 2]), extraordinary, rtol=0, atol=1e-12)


#: Rakic 1998, Table 1 (hbar omega_p, eV) and Table 2 (f_0, Gamma_0, then
#: f_j, Gamma_j, omega_j in eV).
RAKIC = {
    "gold": (
        9.03,
        0.760,
        0.053,
        [
            (0.024, 0.241, 0.415),
            (0.010, 0.345, 0.830),
            (0.071, 0.870, 2.969),
            (0.601, 2.494, 4.304),
            (4.384, 2.214, 13.32),
        ],
    ),
    "copper": (
        10.83,
        0.575,
        0.030,
        [
            (0.061, 0.378, 0.291),
            (0.104, 1.056, 2.957),
            (0.723, 3.213, 5.300),
            (0.638, 4.305, 11.18),
        ],
    ),
    "aluminium": (
        14.98,
        0.523,
        0.047,
        [
            (0.227, 0.333, 0.162),
            (0.050, 0.312, 1.544),
            (0.166, 1.351, 1.808),
            (0.030, 3.382, 3.473),
        ],
    ),
}


def _rakic(metal, energy_ev):
    """Rakic's eqs. (2)-(3) in eV, under :math:`e^{+i\\omega t}` as printed."""
    wp, f0, g0, oscillators = RAKIC[metal]
    w = np.asarray(energy_ev, complex)
    eps = 1 - f0 * wp**2 / (w * (w - 1j * g0))
    for fj, gj, wj in oscillators:
        eps = eps + fj * wp**2 / ((wj**2 - w**2) + 1j * w * gj)
    return eps


@pytest.mark.parametrize("metal", RAKIC)
def test_metals_against_rakic_1998(metal):
    energy = np.geomspace(0.1, 6.0, 500)
    law = getattr(materials, metal)().eps_r
    np.testing.assert_allclose(
        np.asarray(law(energy * e / h)), _rakic(metal, energy), rtol=1e-12
    )


@pytest.mark.parametrize("metal", RAKIC)
def test_metals_against_the_tabulated_rakic_model(metal):
    lam, n, k = np.loadtxt(
        DATA / f"{metal}_rakic1998_ld.csv", delimiter=",", skiprows=7, unpack=True
    )
    index = _n(getattr(materials, metal)().eps_r, lam)  # n - ik under e^{+i omega t}
    np.testing.assert_allclose(index.real, n, rtol=5e-4)
    np.testing.assert_allclose(-index.imag, k, rtol=5e-4)


def test_the_clamped_tensor_and_static_permittivity():
    r = np.asarray(
        materials.lithium_niobate(frequency=_hz(1.55), axes=X_CUT).electro_optic_tensor
    )
    # Turner's clamped r13, r22, r33, r51, pm/V
    entries = (r[0, 2], r[1, 1], r[2, 2], r[3, 1])
    np.testing.assert_allclose(
        np.array(entries) * 1e12, (8.6, 3.4, 30.8, 28.0), rtol=1e-12
    )
    # Smith and Welsh print eps_11^S = 0.392 nF/m and eps_33^S = 0.247 nF/m
    static = materials.lithium_niobate_static(axes=X_CUT).eps_r
    printed = np.array([0.392, 0.392, 0.247]) * 1e-9 / epsilon_0
    np.testing.assert_allclose(static, printed, atol=0.05)
