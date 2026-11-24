r"""Library materials: materials built from one cited published source each.

Each function returns a plain `gfem.Material` whose ``name`` carries a short
source tag (``"silicon (Li 1980)"``), which a saved job keeps. The numbers are
the parameters of the source's formula, or of gfem's lossless refit of it,
never a data table; each entry's docstring gives the full citation, the
specimen, the accuracy and what goes wrong outside its fitted range.

- The optical dielectrics (`silicon`, `silica`, `silicon_nitride`) are
  lossless permittivity laws. With ``frequency=`` they give the constant
  :math:`\mathrm{Re}\,\varepsilon_r` at that frequency instead, which the
  electro-optic and thermo-optic shifts need, as they refuse laws.
- The metals (`gold`, `copper`, `aluminium`) are Drude plus Lorentz laws;
  gfem has no constant metal, so they take no frequency.
- Lithium niobate is anisotropic and laws are isotropic, so
  `lithium_niobate` is built at one frequency on the principal axes given,
  with its clamped electro-optic tensor, and `lithium_niobate_static` is
  its static permittivity for the electrostatic model.

Every law carries its fitted range (`gfem.PermittivityLaw.frequency_range`):
a solve that reads it outside warns `gfem.FittedRangeWarning`. The library
gives no static or RF permittivity for the optical dielectrics (oxide and
nitride vary by process): an electrostatic or RF model takes a constant from
the user's own process.

Change a returned material with `dataclasses.replace` (a region name, a
conductivity, thermal properties, another electro-optic tensor); every check
of `gfem.Material` still runs. For a material or a specimen not here, fit
its measured table with `gfem.fit_permittivity`. Cite the source paper of
each material you publish results with.

Example:
    A nitride guide in oxide with a gold heater, and an x-cut lithium
    niobate film at 1.55 µm::

        from scipy.constants import c

        from gfem import materials

        core, cladding = materials.silicon_nitride(), materials.silica()
        heater = materials.gold()
        f0 = c / 1.55e-6
        axes = ((0, 1, 0), (0, 0, 1), (1, 0, 0))
        film = materials.lithium_niobate(frequency=f0, axes=axes)
        film_static = materials.lithium_niobate_static(axes=axes)
        oxide = materials.silica(frequency=f0)  # a constant, for the shifts

Sources:

.. list-table::
   :header-rows: 1

   * - Function
     - Source
     - Fitted range
     - DOI
   * - `silicon`
     - H. H. Li 1980, 293 K; lossless refit by gfem
     - 1.2-14 µm (21.41-249.8 THz)
     - 10.1063/1.555624
   * - `silica`
     - I. H. Malitson 1965, fused silica
     - 0.21-3.71 µm (80.81-1428 THz)
     - 10.1364/JOSA.55.001205
   * - `silicon_nitride`
     - K. Luke et al. 2015, annealed LPCVD film; lossless refit by gfem
     - 0.31-5.504 µm (54.47-967.1 THz)
     - 10.1364/OL.40.004823
   * - `gold`
     - A. D. Rakić et al. 1998, Lorentz-Drude model, evaporated films
     - 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz)
     - 10.1364/AO.37.005271
   * - `copper`
     - A. D. Rakić et al. 1998, Lorentz-Drude model, evaporated films
     - 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz)
     - 10.1364/AO.37.005271
   * - `aluminium`
     - A. D. Rakić et al. 1998, Lorentz-Drude model, evaporated films
     - 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz)
     - 10.1364/AO.37.005271
   * - `lithium_niobate`
     - D. E. Zelmon et al. 1997, congruent (indices); E. H. Turner 1966,
       clamped (electro-optic tensor)
     - 0.4-5 µm (59.96-749.5 THz)
     - 10.1364/JOSAB.14.003319, 10.1063/1.1754449
   * - `lithium_niobate_static`
     - R. T. Smith and F. S. Welsh 1971, clamped
     - none: a static value
     - 10.1063/1.1660528
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import numpy as np
from scipy.constants import c, e, h

from gfem._checks import is_traced
from gfem._engine.dispersion import FITTED_EDGE
from gfem._materials import Drude, Lorentz, Material, PermittivityLaw
from gfem._text import hertz, hertz_band
from gfem.errors import FittedRangeWarning, NonDifferentiableError

__all__ = [
    "aluminium",
    "copper",
    "gold",
    "lithium_niobate",
    "lithium_niobate_static",
    "silica",
    "silicon",
    "silicon_nitride",
]

_UM = 1e-6
#: Hz per eV of photon energy.
_EV = e / h


def _wavelengths(short: float, long: float) -> tuple[float, float]:
    """The fitted range in Hz of a band given in micrometres."""
    return (c / (long * _UM), c / (short * _UM))


def _energies(low: float, high: float) -> tuple[float, float]:
    """The fitted range in Hz of a band given in eV."""
    return (low * _EV, high * _EV)


#: ``(hbar omega_p, f_0, Gamma_0, ((f_j, Gamma_j, omega_j), ...))``, in eV.
type _RakicRow = tuple[float, float, float, tuple[tuple[float, float, float], ...]]


@dataclasses.dataclass(frozen=True)
class _Source:
    r"""One cited source's numbers, as the paper (or gfem's refit) prints them.

    ``sellmeier`` holds lossless terms :math:`B \lambda^2 / (\lambda^2 -
    \lambda_0^2)` as ``(B, lambda_0 in um)``; ``rakic`` a Lorentz-Drude row
    ``(hbar omega_p, f_0, Gamma_0, ((f_j, Gamma_j, omega_j), ...))`` in eV.
    """

    tag: str
    fitted_range: tuple[float, float]
    sellmeier: tuple[tuple[float, float], ...] = ()
    rakic: _RakicRow | None = None

    def law(self) -> PermittivityLaw:
        """The source's permittivity law, with its fitted range."""
        terms: tuple[Drude | Lorentz, ...]
        if self.rakic is None:
            terms = tuple(
                Lorentz(delta_eps=b, resonance_frequency=c / (lam * _UM), damping=0.0)
                for b, lam in self.sellmeier
            )
        else:
            # Rakic et al. print the model under exp(+i omega t) with gfem's own
            # term forms, so the parameters map one to one, without conjugation.
            wp, f0, g0, oscillators = self.rakic
            drude = Drude(plasma_frequency=math.sqrt(f0) * wp * _EV, damping=g0 * _EV)
            terms = (
                drude,
                *(
                    Lorentz(
                        delta_eps=fj * wp**2 / wj**2,
                        resonance_frequency=wj * _EV,
                        damping=gj * _EV,
                    )
                    for fj, gj, wj in oscillators
                ),
            )
        return PermittivityLaw(
            eps_inf=1.0, terms=terms, frequency_range=self.fitted_range
        )


# Si: gfem's lossless minimax refit of Li's eq. (22) at 293 K, whose
# A / lambda^2 term is no passive pole.
_LI = _Source(
    tag="silicon (Li 1980)",
    fitted_range=_wavelengths(1.2, 14.0),
    sellmeier=((10.6576594, 0.3009794),),
)
# SiO2: Malitson's three-term Sellmeier, verbatim.
_MALITSON = _Source(
    tag="silica (Malitson 1965)",
    fitted_range=_wavelengths(0.21, 3.71),
    sellmeier=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
)
# SiN: gfem's lossless minimax refit of Luke's formula, whose printed second
# term is a 242 GHz resonance of strength 4e4; the refit's IR pole sits at
# 100 um.
_LUKE = _Source(
    tag="silicon nitride (Luke 2015)",
    fitted_range=_wavelengths(0.31, 5.504),
    sellmeier=((3.0244902, 0.1354027), (261.41515, 100.0)),
)
# LN, congruent: Zelmon's ordinary and extraordinary Sellmeier, verbatim
# (B, C in um^2, so lambda_0 = sqrt(C)).
_ZELMON_O = _Source(
    tag="lithium niobate, ordinary (Zelmon 1997)",
    fitted_range=_wavelengths(0.4, 5.0),
    sellmeier=(
        (2.6734, math.sqrt(0.01764)),
        (1.2290, math.sqrt(0.05914)),
        (12.614, math.sqrt(474.60)),
    ),
)
_ZELMON_E = _Source(
    tag="lithium niobate, extraordinary (Zelmon 1997)",
    fitted_range=_wavelengths(0.4, 5.0),
    sellmeier=(
        (2.9804, math.sqrt(0.02047)),
        (0.5981, math.sqrt(0.0666)),
        (8.9543, math.sqrt(416.08)),
    ),
)
# Au, Cu, Al: Rakic et al. Table 1 (hbar omega_p) and Table 2, verbatim.
_RAKIC_GOLD = _Source(
    tag="gold (Rakic 1998)",
    fitted_range=_energies(0.1, 6.0),
    rakic=(
        9.03,
        0.760,
        0.053,
        (
            (0.024, 0.241, 0.415),
            (0.010, 0.345, 0.830),
            (0.071, 0.870, 2.969),
            (0.601, 2.494, 4.304),
            (4.384, 2.214, 13.32),
        ),
    ),
)
_RAKIC_COPPER = _Source(
    tag="copper (Rakic 1998)",
    fitted_range=_energies(0.1, 6.0),
    rakic=(
        10.83,
        0.575,
        0.030,
        (
            (0.061, 0.378, 0.291),
            (0.104, 1.056, 2.957),
            (0.723, 3.213, 5.300),
            (0.638, 4.305, 11.18),
        ),
    ),
)
_RAKIC_ALUMINIUM = _Source(
    tag="aluminium (Rakic 1998)",
    fitted_range=_energies(0.1, 6.0),
    rakic=(
        14.98,
        0.523,
        0.047,
        (
            (0.227, 0.333, 0.162),
            (0.050, 0.312, 1.544),
            (0.166, 1.351, 1.808),
            (0.030, 3.382, 3.473),
        ),
    ),
)
_LITHIUM_NIOBATE_TAG = "lithium niobate (Zelmon 1997, Turner 1966)"
#: Turner's clamped (constant-strain) coefficients at 633 nm, m/V.
_TURNER = {"r13": 8.6e-12, "r22": 3.4e-12, "r33": 30.8e-12, "r51": 28e-12}
#: Smith and Welsh's clamped static permittivity along crystal x, y, z.
_SMITH_WELSH = (44.3, 44.3, 27.9)
_LITHIUM_NIOBATE_STATIC_TAG = "lithium niobate static (Smith & Welsh 1971)"


def _frequency(frequency: Any, owner: str) -> float:
    """One fixed frequency in Hz: a finite real number > 0."""
    if is_traced(frequency):
        raise NonDifferentiableError(
            f"{owner}: frequency is computed inside jax.grad, jax.jit or jax.vmap, "
            "but it is a fixed number: the material is a constant at that "
            "frequency. Pass a plain float."
        )
    if np.ndim(frequency) != 0:
        raise ValueError(
            f"{owner} takes one frequency, got {np.size(frequency)} values: "
            "call it once per frequency in a loop."
        )
    if isinstance(frequency, (bool, np.bool_)):
        raise ValueError(  # noqa: TRY004 (every unusable frequency is one class)
            f"{owner}: frequency must be one real number in Hz, got {frequency!r}."
        )
    try:
        f = float(frequency)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}: frequency must be one real number in Hz, got {frequency!r}."
        ) from None
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{owner}: frequency must be a finite number > 0 Hz, got {f}.")
    return f


def _warn_outside(
    source: _Source, tag: str, frequency: float, owner: str, stacklevel: int
) -> None:
    """Warn `FittedRangeWarning` when ``frequency`` lies outside the source's range.

    ``stacklevel`` points the warning at the user's call of the entry.
    """
    lo, hi = source.fitted_range
    if lo * (1 - FITTED_EDGE) <= frequency <= hi * (1 + FITTED_EDGE):
        return
    warnings.warn(
        f"{owner}: the permittivity law of {tag} was fitted over "
        f"{hertz_band(lo, hi, 4)}, and frequency={hertz(frequency, 4)} lies "
        "outside it. The constant is the law's value there, passive but of "
        "unknown accuracy. Choose a frequency inside the range, or give the "
        "region a constant gfem.Material(eps_r=...) from data that cover it.",
        FittedRangeWarning,
        stacklevel=stacklevel,
    )


def _real_value(source: _Source, frequency: float) -> float:
    """``Re law(frequency)``: the whole value, since the law is lossless."""
    return float(np.real(complex(source.law()(frequency))))


def _dielectric(source: _Source, frequency: Any, owner: str) -> Material:
    if frequency is None:
        return Material(eps_r=source.law(), name=source.tag)
    f = _frequency(frequency, owner)
    _warn_outside(source, source.tag, f, owner, stacklevel=4)
    return Material(eps_r=_real_value(source, f), name=source.tag)


def silicon(*, frequency: float | None = None) -> Material:
    r"""Crystalline silicon at 293 K, a lossless permittivity law (Li 1980).

    H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), doi:10.1063/1.555624:
    critically evaluated reference data for silicon at 293 K, stated within
    :math:`\pm 2\times10^{-3}` in :math:`n`. Li's formula
    :math:`n^2 = \varepsilon + A/\lambda^2` is not a sum of passive terms, so
    the law is gfem's lossless minimax refit of it in :math:`n`: one `Lorentz`
    term with :math:`\varepsilon_\infty = 1`, ``delta_eps`` 10.6576594 and its
    resonance at 0.3009794 µm (996.06 THz), no damping. It is within
    :math:`8.3\times10^{-4}` of Li's :math:`n` over the fitted range, inside
    Li's uncertainty; :math:`n = 3.474948` at 1.55 µm, against Li's 3.475718.

    Fitted range: 1.2-14 µm (21.41-249.8 THz), the law's ``frequency_range``;
    a solve that reads the law outside it warns `gfem.FittedRangeWarning`.
    Outside it the law stays lossless, also below about 1.1 µm where silicon
    absorbs. Its static value :math:`\varepsilon_r(0) = 11.66` is close to
    silicon's 11.7.

    Args:
        frequency: One frequency in Hz. Given, the material is the constant
            :math:`\mathrm{Re}\,\varepsilon_r` of the law there (the law is
            lossless), as the electro-optic and thermo-optic shifts need: it
            holds at that frequency only, not as a static value, and carries
            no fitted range, so later solves read it silently. ``None``: the
            law itself.

    Returns:
        A `gfem.Material` named ``"silicon (Li 1980)"``.

    Raises:
        ValueError: ``frequency`` is not one finite real number > 0 (an
            array: call once per frequency).
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`: it is a fixed number.

    Warns:
        FittedRangeWarning: ``frequency`` lies outside the fitted range; the
            constant is still returned.
    """
    return _dielectric(_LI, frequency, "silicon")


def silica(*, frequency: float | None = None) -> Material:
    r"""Fused silica, a lossless permittivity law (Malitson 1965).

    I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965),
    doi:10.1364/JOSA.55.001205: fused silica at 20 °C, a three-term Sellmeier
    formula fitted to about :math:`10^{-5}` in :math:`n`. Its terms are
    lossless poles, so the law is the formula verbatim: three `Lorentz` terms
    with :math:`\varepsilon_\infty = 1`, ``delta_eps`` 0.6961663, 0.4079426
    and 0.8974794 and resonances at 0.0684043, 0.1162414 and 9.896161 µm, no
    damping; :math:`n = 1.444024` at 1.55 µm. Thermal, deposited and
    sputtered oxides differ from fused silica by :math:`10^{-3}` to
    :math:`10^{-2}` in :math:`n`.

    Fitted range: 0.21-3.71 µm (80.81-1428 THz), the law's
    ``frequency_range``; a solve that reads the law outside it warns
    `gfem.FittedRangeWarning`. Outside it the law stays lossless through the
    infrared absorption of silica near 9 µm, and its static value
    :math:`\varepsilon_r(0) = 3.00` is well below the 3.8-4.2 of real oxide:
    an electrostatic or RF model takes a constant for the oxide of its
    process. Likewise ``silica(frequency=c / 1.55e-6)`` is 2.085, an optical
    value, not a static one.

    Args:
        frequency: One frequency in Hz. Given, the material is the constant
            :math:`\mathrm{Re}\,\varepsilon_r` of the law there (the law is
            lossless), as the electro-optic and thermo-optic shifts need: it
            holds at that frequency only, not as a static value, and carries
            no fitted range, so later solves read it silently. ``None``: the
            law itself.

    Returns:
        A `gfem.Material` named ``"silica (Malitson 1965)"``.

    Raises:
        ValueError: ``frequency`` is not one finite real number > 0 (an
            array: call once per frequency).
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`: it is a fixed number.

    Warns:
        FittedRangeWarning: ``frequency`` lies outside the fitted range; the
            constant is still returned.
    """
    return _dielectric(_MALITSON, frequency, "silica")


def silicon_nitride(*, frequency: float | None = None) -> Material:
    r"""Stoichiometric silicon nitride, a lossless permittivity law (Luke 2015).

    K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson, Opt. Lett.
    40, 4823 (2015), doi:10.1364/OL.40.004823: an LPCVD film annealed at
    1200 °C, measured by ellipsometry and reduced to a two-term Sellmeier
    formula, with no stated accuracy. The formula's second term is a
    resonance at 242 GHz of strength :math:`4\times10^{4}`, so the law is
    gfem's lossless minimax refit of it in :math:`n`: two `Lorentz` terms with
    :math:`\varepsilon_\infty = 1`, ``delta_eps`` 3.0244902 at 0.1354027 µm
    and 261.41515 at 100 µm (3 THz), no damping. It is within
    :math:`7.1\times10^{-5}` of Luke's :math:`n` over the fitted range;
    :math:`n = 1.996228` at 1.55 µm, against Luke's 1.996280. Deposited
    (PECVD) nitride differs: fit its own data with `gfem.fit_permittivity`.

    Fitted range: 0.31-5.504 µm (54.47-967.1 THz), the law's
    ``frequency_range``; a solve that reads the law outside it warns
    `gfem.FittedRangeWarning`. Outside it the law is far off: its infrared
    term is a lossless pole at 100 µm (3 THz), so a driven sweep through
    3 THz raises on it, :math:`\varepsilon_r` is -21.8 at 30 µm, and its
    static value :math:`\varepsilon_r(0) = 265` is about 35 times that of
    real silicon nitride: an electrostatic or RF model takes a constant for
    the nitride of its process.

    Args:
        frequency: One frequency in Hz. Given, the material is the constant
            :math:`\mathrm{Re}\,\varepsilon_r` of the law there (the law is
            lossless), as the electro-optic and thermo-optic shifts need: it
            holds at that frequency only, not as a static value, and carries
            no fitted range, so later solves read it silently. ``None``: the
            law itself.

    Returns:
        A `gfem.Material` named ``"silicon nitride (Luke 2015)"``.

    Raises:
        ValueError: ``frequency`` is not one finite real number > 0 (an
            array: call once per frequency).
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`: it is a fixed number.

    Warns:
        FittedRangeWarning: ``frequency`` lies outside the fitted range; the
            constant is still returned.
    """
    return _dielectric(_LUKE, frequency, "silicon_nitride")


def gold() -> Material:
    r"""Evaporated gold films, a Drude plus Lorentz permittivity law (Rakić 1998).

    A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski, Appl. Opt. 37,
    5271 (1998), doi:10.1364/AO.37.005271: the Lorentz-Drude model of Table 2
    (with :math:`\hbar\omega_p` from Table 1), fitted to measured film data
    over 0.1-6 eV. The paper writes the model under :math:`e^{+i\omega t}`
    with gfem's own term forms, so the law is its parameters verbatim, with
    no conjugation: :math:`\varepsilon_\infty = 1`, one `Drude` term and
    five `Lorentz` terms. Its error against measured films, 10-16 %, is
    about the spread between films (5-18 %); :math:`n = 0.575 - 9.66i` at 1.55 µm.

    Fitted range: 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz), the law's
    ``frequency_range``; a solve that reads the law outside it warns
    `gfem.FittedRangeWarning`. Four of its damped Lorentz resonances lie
    inside the fitted range (two in 0.4-2 µm, one in 1.26-1.68
    µm); each costs an adaptive sweep two solves. Below the range the
    law's loss gives :math:`1.6\times10^{7}` S/m at 1 GHz-1 THz,
    0.38 times bulk gold: for gold at RF use `gfem.Conductor`
    with its bulk conductivity.

    There is no ``frequency=`` constant: a metal's permittivity is negative,
    which a constant `gfem.Material` refuses.

    Returns:
        A `gfem.Material` named ``"gold (Rakic 1998)"``.
    """
    return Material(eps_r=_RAKIC_GOLD.law(), name=_RAKIC_GOLD.tag)


def copper() -> Material:
    r"""Evaporated copper films, a Drude plus Lorentz permittivity law (Rakić 1998).

    A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski, Appl. Opt. 37,
    5271 (1998), doi:10.1364/AO.37.005271: the Lorentz-Drude model of Table 2
    (with :math:`\hbar\omega_p` from Table 1), fitted to measured film data
    over 0.1-6 eV. The paper writes the model under :math:`e^{+i\omega t}`
    with gfem's own term forms, so the law is its parameters verbatim, with
    no conjugation: :math:`\varepsilon_\infty = 1`, one `Drude` term and
    four `Lorentz` terms. Its error against measured films, 10-16 %, is
    about the spread between films (5-18 %); :math:`n = 0.470 - 10.45i` at 1.55 µm.

    Fitted range: 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz), the law's
    ``frequency_range``; a solve that reads the law outside it warns
    `gfem.FittedRangeWarning`. Three of its damped Lorentz resonances lie
    inside the fitted range (one in 0.4-2 µm, none in 1.26-1.68
    µm); each costs an adaptive sweep two solves. Below the range the
    law's loss gives :math:`3.0\times10^{7}` S/m at 1 GHz-1 THz,
    0.51 times bulk copper: for copper at RF use `gfem.Conductor`
    with its bulk conductivity.

    There is no ``frequency=`` constant: a metal's permittivity is negative,
    which a constant `gfem.Material` refuses.

    Returns:
        A `gfem.Material` named ``"copper (Rakic 1998)"``.
    """
    return Material(eps_r=_RAKIC_COPPER.law(), name=_RAKIC_COPPER.tag)


def aluminium() -> Material:
    r"""Evaporated aluminium films, a Drude plus Lorentz permittivity law (Rakić 1998).

    A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski, Appl. Opt. 37,
    5271 (1998), doi:10.1364/AO.37.005271: the Lorentz-Drude model of Table 2
    (with :math:`\hbar\omega_p` from Table 1), fitted to measured film data
    over 0.1-6 eV. The paper writes the model under :math:`e^{+i\omega t}`
    with gfem's own term forms, so the law is its parameters verbatim, with
    no conjugation: :math:`\varepsilon_\infty = 1`, one `Drude` term and
    four `Lorentz` terms. Its error against measured films, 10-16 %, is
    about the spread between films (5-18 %); :math:`n = 1.536 - 15.33i` at 1.55 µm.

    Fitted range: 0.1-6 eV, 0.21-12.4 µm (24.18-1451 THz), the law's
    ``frequency_range``; a solve that reads the law outside it warns
    `gfem.FittedRangeWarning`. Four of its damped Lorentz resonances lie
    inside the fitted range (two in 0.4-2 µm, none in 1.26-1.68
    µm); each costs an adaptive sweep two solves. Below the range the
    law's loss gives :math:`3.4\times10^{7}` S/m at 1 GHz-1 THz,
    0.89 times bulk aluminium: for aluminium at RF use `gfem.Conductor`
    with its bulk conductivity.

    There is no ``frequency=`` constant: a metal's permittivity is negative,
    which a constant `gfem.Material` refuses.

    Returns:
        A `gfem.Material` named ``"aluminium (Rakic 1998)"``.
    """
    return Material(eps_r=_RAKIC_ALUMINIUM.law(), name=_RAKIC_ALUMINIUM.tag)


def _tensor_3m(r13: float, r22: float, r33: float, r51: float) -> tuple:
    """The contracted tensor of class 3m: rows xx yy zz yz xz xy, columns x y z."""
    return (
        (0.0, -r22, r13),
        (0.0, r22, r13),
        (0.0, 0.0, r33),
        (0.0, r51, 0.0),
        (r51, 0.0, 0.0),
        (-r22, 0.0, 0.0),
    )


def lithium_niobate(*, frequency: float, axes: Any) -> Material:
    r"""Congruent lithium niobate at one frequency, with its electro-optic tensor.

    The principal values are :math:`(n_o^2, n_o^2, n_e^2)` along the
    crystal's x, y and z (z the optic axis), from the congruent ordinary and
    extraordinary Sellmeier formulas of D. E. Zelmon, D. L. Small, D. Jundt,
    J. Opt. Soc. Am. B 14, 3319 (1997), doi:10.1364/JOSAB.14.003319, at
    21 °C. Each formula is three lossless poles, read here exactly;
    :math:`n_o = 2.211111` and :math:`n_e = 2.137560` at 1.55 µm. The
    electro-optic tensor is the clamped (constant-strain) class 3m set of
    E. H. Turner, Appl. Phys. Lett. 8, 303 (1966), doi:10.1063/1.1754449,
    as compiled by R. S. Weis and T. K. Gaylord, Appl. Phys. A 37, 191
    (1985), doi:10.1007/BF00614817: :math:`r_{13} = 8.6`,
    :math:`r_{22} = 3.4`, :math:`r_{33} = 30.8` and :math:`r_{51} = 28`
    pm/V at 633 nm, used unchanged at other wavelengths. Clamped values hold
    for thin films and for drives above the acoustic resonances; give
    another tensor with `dataclasses.replace`.

    Laws are isotropic, so the material is a constant at ``frequency``.
    Fitted range of the index formulas: 0.4-5 µm (59.96-749.5 THz); a
    ``frequency`` outside it warns `gfem.FittedRangeWarning` at the call.
    The formulas' static values, 17.5 and 13.5, are not the crystal's static
    permittivity: the electrostatic model takes `lithium_niobate_static`.

    Args:
        frequency: The frequency in Hz the indices are taken at.
        axes: The crystal's x, y and z as three rows in the model frame, the
            material's principal axes and the frame of its electro-optic
            tensor; right-handed. For a section in :math:`(x, y)` with the
            guide along :math:`z`, x-cut is ``((0, 1, 0), (0, 0, 1),
            (1, 0, 0))`` and z-cut ``((1, 0, 0), (0, 0, -1), (0, 1, 0))``.
            Pass the same ``axes`` to `lithium_niobate_static`.

    Returns:
        A `gfem.Material` named ``"lithium niobate (Zelmon 1997, Turner
        1966)"``.

    Raises:
        ValueError: ``frequency`` is not one finite real number > 0 (an
            array: call once per frequency), or ``axes`` are not orthonormal
            and right-handed.
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`: it is a fixed number.

    Warns:
        FittedRangeWarning: ``frequency`` lies outside 0.4-5 µm; the material
            is still returned.
    """
    owner = "lithium_niobate"
    f = _frequency(frequency, owner)
    _warn_outside(_ZELMON_O, _LITHIUM_NIOBATE_TAG, f, owner, stacklevel=3)
    n_o2, n_e2 = _real_value(_ZELMON_O, f), _real_value(_ZELMON_E, f)
    return Material(
        eps_r=(n_o2, n_o2, n_e2),
        axes=axes,
        electro_optic_tensor=_tensor_3m(**_TURNER),
        name=_LITHIUM_NIOBATE_TAG,
    )


def lithium_niobate_static(*, axes: Any) -> Material:
    r"""Congruent lithium niobate's clamped static permittivity.

    The principal values :math:`(44.3, 44.3, 27.9)` along the crystal's x, y
    and z: the clamped (constant-strain) values of R. T. Smith and
    F. S. Welsh, J. Appl. Phys. 42, 2219 (1971), doi:10.1063/1.1660528, at
    25 °C, for the electrostatic model of an electro-optic device. It holds
    no electro-optic tensor: the electrostatic solve reads only ``eps_r``.

    Args:
        axes: The crystal's x, y and z as three rows in the model frame, the
            same ``axes`` as the optical `lithium_niobate`, which this
            function cannot check.

    Returns:
        A `gfem.Material` named ``"lithium niobate static (Smith & Welsh
        1971)"``.

    Raises:
        ValueError: ``axes`` are not orthonormal.
    """
    return Material(eps_r=_SMITH_WELSH, axes=axes, name=_LITHIUM_NIOBATE_STATIC_TAG)
