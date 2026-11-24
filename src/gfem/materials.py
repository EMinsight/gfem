r"""Library materials: ready-made materials from cited published parameters.

TODO sources table.
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
    rakic: tuple[float, float, float, tuple[tuple[float, float, float], ...]] | None = (
        None
    )

    def law(self) -> PermittivityLaw:
        if self.rakic is None:
            terms = [
                Lorentz(delta_eps=b, resonance_frequency=c / (lam * _UM), damping=0.0)
                for b, lam in self.sellmeier
            ]
        else:
            # Rakic et al. print the model under exp(+i omega t) with gfem's own
            # term forms, so the parameters map one to one, without conjugation.
            wp, f0, g0, oscillators = self.rakic
            terms = [Drude(plasma_frequency=math.sqrt(f0) * wp * _EV, damping=g0 * _EV)]
            terms += [
                Lorentz(
                    delta_eps=fj * wp**2 / wj**2,
                    resonance_frequency=wj * _EV,
                    damping=gj * _EV,
                )
                for fj, gj, wj in oscillators
            ]
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
    try:
        f = float(frequency)
    except (TypeError, ValueError):
        raise ValueError(
            f"{owner}: frequency must be one real number in Hz, got {frequency!r}."
        ) from None
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"{owner}: frequency must be a finite number > 0 Hz, got {f}.")
    return f


def _warn_outside(source: _Source, tag: str, frequency: float, owner: str) -> None:
    """Warn `FittedRangeWarning` when ``frequency`` lies outside the source's range."""
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
        stacklevel=3,
    )


def _real_value(source: _Source, frequency: float) -> float:
    """``Re law(frequency)``: the whole value, since the law is lossless."""
    return float(np.real(complex(source.law()(frequency))))


def _dielectric(source: _Source, frequency: Any, owner: str) -> Material:
    if frequency is None:
        return Material(eps_r=source.law(), name=source.tag)
    f = _frequency(frequency, owner)
    _warn_outside(source, source.tag, f, owner)
    return Material(eps_r=_real_value(source, f), name=source.tag)


def silicon(*, frequency: float | None = None) -> Material:
    """Crystalline silicon at 293 K (Li 1980)."""
    return _dielectric(_LI, frequency, "silicon")


def silica(*, frequency: float | None = None) -> Material:
    """Fused silica (Malitson 1965)."""
    return _dielectric(_MALITSON, frequency, "silica")


def silicon_nitride(*, frequency: float | None = None) -> Material:
    """Stoichiometric LPCVD silicon nitride (Luke 2015)."""
    return _dielectric(_LUKE, frequency, "silicon_nitride")


def gold() -> Material:
    """Evaporated gold film (Rakic 1998)."""
    return Material(eps_r=_RAKIC_GOLD.law(), name=_RAKIC_GOLD.tag)


def copper() -> Material:
    """Evaporated copper film (Rakic 1998)."""
    return Material(eps_r=_RAKIC_COPPER.law(), name=_RAKIC_COPPER.tag)


def aluminium() -> Material:
    """Evaporated aluminium film (Rakic 1998)."""
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
    """Congruent lithium niobate at one optical frequency."""
    owner = "lithium_niobate"
    f = _frequency(frequency, owner)
    _warn_outside(_ZELMON_O, _LITHIUM_NIOBATE_TAG, f, owner)
    n_o2, n_e2 = _real_value(_ZELMON_O, f), _real_value(_ZELMON_E, f)
    return Material(
        eps_r=(n_o2, n_o2, n_e2),
        axes=axes,
        electro_optic_tensor=_tensor_3m(**_TURNER),
        name=_LITHIUM_NIOBATE_TAG,
    )


def lithium_niobate_static(*, axes: Any) -> Material:
    """Congruent lithium niobate's clamped static permittivity."""
    return Material(eps_r=_SMITH_WELSH, axes=axes, name=_LITHIUM_NIOBATE_STATIC_TAG)
