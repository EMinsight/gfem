"""Every per-element read of a permittivity law.

A `Material` whose ``eps_r`` is a `gfem.PermittivityLaw` has no one
permittivity: `gfem._engine.coefficients.element_values` refuses it, and the
readers that understand a law read it here, each at a named frequency:

- `high_frequency_eps`: the law's eps_inf, the shared mass piece's value;
- `eps_at`: the complex permittivity at an angular frequency;
- `static_eps`: the static permittivity eps(0), the electrostatic
  coefficient (`check_static` refuses a Drude term);
- `law_regions`: the laws and their tetrahedra, for the susceptibility
  pieces (`susceptibility`, `d_susceptibility`) and the checks;
- `blend_regions`: the density blends with a law end, whose
  eps_inf limit `high_frequency_eps` holds and whose susceptibility parts
  (`blend_parts`) the driven operator reads; `gfem.solve` refuses them on
  every other problem;
- `check_frequencies`: what a driven sweep checks before any assembly;
- `check_band`: the eigenmode problem's unresolved band, and
  `nearest_term`, the term a message names;
- `resonance_seeds`: the grid frequencies the adaptive sweep solves first;
- `absorbing_surfaces`: the absorbing surfaces each law region lies behind;
- `has_loss`, `check_estimator_frequencies`: what the refinement estimators
  ask of a law.

The law is evaluated here on the host in complex scalars, by hand, term by
term (``chi`` and ``d chi / d omega``), never through JAX: assembly does not
trace. Frequencies are angular (rad/s); the laws take hertz.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import numpy as np

from gfem import _mpi
from gfem._boundaries import Absorbing
from gfem._checks import is_traced
from gfem._engine.coefficients import element_values, lossy_eps
from gfem._materials import (
    Blend,
    Debye,
    DjordjevicSarkar,
    Drude,
    Lorentz,
    Material,
    MaterialField,
    PermittivityLaw,
    refractive_index,
)
from gfem._text import hertz, hertz_band
from gfem.errors import FittedRangeWarning
from gfem.model import Model

__all__ = [
    "BlendPart",
    "BlendRegion",
    "LawRegion",
    "absorbing_surfaces",
    "blend_mass",
    "blend_parts",
    "blend_regions",
    "brillouin_eps",
    "check_band",
    "check_energy",
    "check_estimator_frequencies",
    "check_frequencies",
    "check_static",
    "d_energy_eps",
    "d_eps",
    "d_susceptibility",
    "energy_eps",
    "eps_at",
    "has_loss",
    "high_frequency_eps",
    "law_regions",
    "nearest_term",
    "refuse_blends",
    "resonance_seeds",
    "static_eps",
    "susceptibility",
    "warn_fitted_range",
]

_TWO_PI = 2 * math.pi


@dataclasses.dataclass(frozen=True)
class LawRegion:
    """A region whose material has a permittivity law.

    Attributes:
        region: The volume region.
        law: Its law.
        mask: Its tetrahedra on the solve domain, a boolean per gfem
            tetrahedron.
    """

    region: str
    law: PermittivityLaw
    mask: np.ndarray = dataclasses.field(compare=False)


def law_regions(model: Model) -> tuple[LawRegion, ...]:
    """The model's law regions, in ``model.materials`` order."""
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    out = []
    for region, material in model.materials.items():
        if isinstance(material, Material) and isinstance(
            material.eps_r, PermittivityLaw
        ):
            mask = np.zeros(len(mesh.volume_elements), bool)
            mask[mesh.elements(region)] = True
            out.append(LawRegion(region, material.eps_r, mask & active))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class BlendRegion:
    """A region whose `MaterialField` blends a law end by a density.

    Attributes:
        region: The volume region.
        ends: The fixed ends and scheme.
        mask: Its tetrahedra on the solve domain, a boolean per gfem
            tetrahedron.
        rho: The density per gfem tetrahedron (0 off the region).
    """

    region: str
    ends: Blend
    mask: np.ndarray = dataclasses.field(compare=False)
    rho: np.ndarray = dataclasses.field(compare=False)


def blend_regions(model: Model) -> tuple[BlendRegion, ...]:
    """The model's law blends, in ``model.materials`` order (concrete densities).

    Per element: a layer grid's pixels, sampled at the quadrature points
    (`gfem._engine.points`), are left out.
    """
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    out = []
    for region, material in model.materials.items():
        if (
            isinstance(material, MaterialField)
            and material.blend_ends is not None
            and material.blend_ends.pixels is None
        ):
            elements = mesh.elements(region)
            mask = np.zeros(len(mesh.volume_elements), bool)
            mask[elements] = True
            rho = np.zeros(len(mesh.volume_elements))
            rho[elements] = np.asarray(material.blend, dtype=float)
            out.append(BlendRegion(region, material.blend_ends, mask & active, rho))
    return tuple(out)


def _end_eps_inf(end: PermittivityLaw | complex) -> float:
    """The eps_inf of a blend end; a constant end's is Re eps."""
    if isinstance(end, PermittivityLaw):
        return float(end.eps_inf)
    return complex(end).real


def _end_chi(end: PermittivityLaw | complex, omega: complex) -> complex:
    """eps(omega) - eps_inf of a blend end (a constant end's: i Im eps)."""
    if isinstance(end, PermittivityLaw):
        return susceptibility(end, omega)
    return 1j * complex(end).imag


def _end_d_chi(end: PermittivityLaw | complex, omega: complex) -> complex:
    """D chi / d omega of a blend end."""
    if isinstance(end, PermittivityLaw):
        return d_susceptibility(end, omega)
    return 0.0


@dataclasses.dataclass(frozen=True)
class BlendPart:
    """One susceptibility piece of a law blend: eps = mass + sum_parts c X(omega).

    Attributes:
        name: The part, ``"void"``, ``"solid"`` or ``"cross"``.
        chi: X(omega), with an analytic dX/domega in `d_chi`.
        d_chi: dX / domega.
        c: The coefficient per gfem tetrahedron (0 off the region).
        dc: dc / d rho per gfem tetrahedron.
    """

    name: str
    chi: Callable[[complex], complex]
    d_chi: Callable[[complex], complex]
    c: np.ndarray = dataclasses.field(compare=False)
    dc: np.ndarray = dataclasses.field(compare=False)


def blend_mass(br: BlendRegion) -> tuple[np.ndarray, np.ndarray]:
    """The blend's eps_inf limit on its region and d/d rho, per gfem tetrahedron.

    ``"index"``: ((1 - rho) sqrt(eps_inf,a) + rho sqrt(eps_inf,b))^2; a weight
    scheme: (1 - w) eps_inf,a + w eps_inf,b. Zero off the region.
    """
    ea, eb = _end_eps_inf(br.ends.void), _end_eps_inf(br.ends.solid)
    rho = br.rho
    if br.ends.scheme == "index":
        sa, sb = math.sqrt(ea), math.sqrt(eb)
        n = (1 - rho) * sa + rho * sb
        value, slope = n**2, 2 * n * (sb - sa)
    else:
        w = br.ends.weight(rho)
        value, slope = (1 - w) * ea + w * eb, br.ends.slope(rho) * (eb - ea)
    return np.where(br.mask, value, 0.0), np.where(br.mask, slope, 0.0)


def blend_parts(br: BlendRegion) -> list[BlendPart]:
    """The susceptibility parts of a law blend.

    ``"index"``: chi_a with c = (1 - rho)^2, chi_b with c = rho^2, and the
    cross part N_a N_b - sqrt(eps_inf,a eps_inf,b) with c = 2 rho (1 - rho),
    N = sqrt(eps) with Im N <= 0. A weight scheme: chi_a with c = 1 - w,
    chi_b with c = w. A lossless constant end has no part of its own.
    """
    ends, rho, mask = br.ends, br.rho, br.mask
    a, b = ends.void, ends.solid
    ea, eb = _end_eps_inf(a), _end_eps_inf(b)
    if ends.scheme == "index":
        coefficients = {
            "void": ((1 - rho) ** 2, -2 * (1 - rho)),
            "solid": (rho**2, 2 * rho),
        }
    else:
        w, dw = ends.weight(rho), ends.slope(rho)
        coefficients = {"void": (1 - w, -dw), "solid": (w, dw)}
    out = []
    for name, end in (("void", a), ("solid", b)):
        if not isinstance(end, PermittivityLaw) and end.imag == 0:
            continue
        c, dc = coefficients[name]
        out.append(
            BlendPart(
                name,
                lambda w, end=end: _end_chi(end, w),
                lambda w, end=end: _end_d_chi(end, w),
                np.where(mask, c, 0.0),
                np.where(mask, dc, 0.0),
            )
        )
    if ends.scheme == "index":
        root = math.sqrt(ea * eb)

        def n(end: PermittivityLaw | complex, e: float, w: complex) -> complex:
            return refractive_index(e + _end_chi(end, w))

        def cross(w: complex) -> complex:
            return n(a, ea, w) * n(b, eb, w) - root

        def d_cross(w: complex) -> complex:
            na, nb = n(a, ea, w), n(b, eb, w)
            return _end_d_chi(a, w) / (2 * na) * nb + na * _end_d_chi(b, w) / (2 * nb)

        out.append(
            BlendPart(
                "cross",
                cross,
                d_cross,
                np.where(mask, 2 * rho * (1 - rho), 0.0),
                np.where(mask, 2 - 4 * rho, 0.0),
            )
        )
    return out


def high_frequency_eps(model: Model) -> np.ndarray:
    """`gfem._engine.coefficients.lossy_eps` with eps_inf on the law regions.

    The value of the shared mass piece: a law's frequency
    dependence is its susceptibility piece. A law blend's region holds its
    eps_inf limit (`blend_mass`).
    """
    eps = lossy_eps(model, law=0.0)
    values = [(lr.mask, float(lr.law.eps_inf)) for lr in law_regions(model)]
    values += [(br.mask, blend_mass(br)[0][br.mask]) for br in blend_regions(model)]
    for mask, value in values:
        if eps.ndim == 1:
            eps[mask] = value
        else:
            eps[mask] = np.multiply.outer(value, np.eye(3))
    return eps


def eps_at(model: Model, omega: complex) -> np.ndarray:
    """The complex relative permittivity per tetrahedron at ``omega`` (rad/s).

    ``eps_r (1 - i tan_delta)`` off the law regions (`lossy_eps`'s shape),
    eps_inf + chi(omega) on them, and a law blend's permittivity at that
    frequency on its region. The conductivity is not included. ``omega``
    may be complex (the analytic continuation).
    """
    eps = high_frequency_eps(model).astype(complex)
    chis = [(lr.mask, susceptibility(lr.law, omega)) for lr in law_regions(model)]
    for br in blend_regions(model):
        chi = sum((p.c * p.chi(omega) for p in blend_parts(br)), np.zeros(len(br.rho)))
        chis.append((br.mask, chi[br.mask]))
    for mask, chi in chis:
        if eps.ndim == 1:
            eps[mask] += chi
        else:
            eps[mask] += np.multiply.outer(chi, np.eye(3))
    return eps


def check_static(model: Model) -> None:
    """Raise for a law region with no static permittivity (a Drude term).

    Static: it reads the term kinds only, so it runs before any value is
    known. A lossless Drude term diverges at 0 Hz and a lossy
    one gives Re chi(0) = -ωp²/γ² < 0, which would make the electrostatic
    operator indefinite; dropping the term would give a metal eps_inf.

    A density blend with a law end is refused too
    (`refuse_blends`).

    Raises:
        ValueError: A law region's law, or an end of a law blend, has a
            `gfem.Drude` term.
        NotImplementedError: A law blend without a Drude end.
    """
    refuse_blends(model, "the Electrostatic problem", static=True)
    for region, material in model.materials.items():
        if not (
            isinstance(material, Material)
            and isinstance(material.eps_r, PermittivityLaw)
        ):
            continue
        if any(isinstance(t, Drude) for t in material.eps_r.terms):
            name = material.name or region
            raise ValueError(
                f"Material {name!r} (region {region!r}) has a Drude term, and a "
                "Drude metal has no static permittivity: it conducts at zero "
                "frequency. In an electrostatic model, make it a terminal or a "
                "PEC surface."
            )


#: The remedy of every refusal of a law blend.
_UNIFORM_ONLY = (
    "a metal-dielectric blend (MaterialField.interpolate with a permittivity "
    "law) is supported on the driven uniform sweep only (gfem.Driven with "
    'sweep="uniform")'
)


def _blends(model: Model) -> list[tuple[str, Blend]]:
    """``(region, ends)`` of each law blend, in ``model.materials`` order.

    Reads the ends only (static data), never the density, so it runs on a
    traced model too.
    """
    return [
        (region, material.blend_ends)
        for region, material in model.materials.items()
        if isinstance(material, MaterialField) and material.blend_ends is not None
    ]


def refuse_blends(model: Model, where: str, *, static: bool = False) -> None:
    """Raise if the model has a law blend: ``where`` does not support one.

    A density blend with a permittivity-law end is solved by
    `gfem.Driven` on the uniform sweep only: its cross piece's square root
    has branch cuts in complex omega (eigenmode), the adaptive sweep, the
    port mode problem, the estimators and the energies do not read it.
    ``where`` names the caller in the message (``"the Eigenmode problem"``,
    ``"error_estimate"``); with ``static`` a `Drude` end is named first,
    since it has no static permittivity at all.

    Raises:
        ValueError: With ``static``, an end of the first blend has a
            `gfem.Drude` term.
        NotImplementedError: The model has a law blend.
    """
    blends = _blends(model)
    if not blends:
        return
    region, ends = blends[0]
    if static and ends.drude:
        raise ValueError(
            f"Region {region!r} blends a permittivity law with a Drude term, "
            "which has no static permittivity (eps(0) is infinite), so the "
            "electrostatic problem cannot read it. Use a constant material for "
            "the electrostatic model."
        )
    if ends.pixels is not None:
        raise NotImplementedError(
            f"Region {region!r} holds a layer-grid design of a 2D model, which "
            f"{where} does not support: it is solved by gfem.Driven only."
        )
    raise NotImplementedError(
        f"Region {region!r} holds a density blend with a permittivity-law end, "
        f"which {where} does not support: {_UNIFORM_ONLY}."
    )


def static_eps(model: Model) -> np.ndarray:
    """The real static permittivity per tetrahedron, ``(n,)``.

    `gfem._engine.coefficients.element_values`'s real ``eps_r`` off the law
    regions; ``law(0)`` on them, eps_inf + the strengths of the Lorentz,
    Debye and Djordjevic-Sarkar terms, evaluated by the law itself so a
    constant `Material` set to ``law(0)`` reads the same value to the bit.

    Raises:
        ValueError: A law has a Drude term (`check_static`).
    """
    check_static(model)
    eps = element_values(model, "eps_r", per_axis=0.0, law=0.0).real
    mesh = model.mesh
    for region, material in model.materials.items():
        if isinstance(material, Material) and isinstance(
            material.eps_r, PermittivityLaw
        ):
            eps[mesh.elements(region)] = float(np.real(complex(material.eps_r(0.0))))
    return eps


# --- the terms, in hertz: chi(f) and d chi / d f ---------------------------------


def _term(term: object, f: complex) -> tuple[complex, complex]:
    """``(chi, d chi / d f)`` of one term at ``f`` (Hz, complex accepted)."""
    if isinstance(term, Drude):
        fp, g = float(term.plasma_frequency), float(term.damping)
        den = f * (f - 1j * g)
        return -(fp**2) / den, fp**2 * (2 * f - 1j * g) / den**2
    if isinstance(term, Lorentz):
        d, f0, g = (
            float(term.delta_eps),
            float(term.resonance_frequency),
            float(term.damping),
        )
        den = f0**2 - f * f + 1j * g * f
        return d * f0**2 / den, d * f0**2 * (2 * f - 1j * g) / den**2
    if isinstance(term, Debye):
        d, fr = float(term.delta_eps), float(term.relaxation_frequency)
        den = 1 + 1j * f / fr
        return d / den, -d * (1j / fr) / den**2
    if isinstance(term, DjordjevicSarkar):
        d, f1, f2 = (
            float(term.delta_eps),
            float(term.low_frequency),
            float(term.high_frequency),
        )
        scale = d / math.log(f2 / f1)
        value = scale * cmath.log((f2 + 1j * f) / (f1 + 1j * f))
        return value, scale * (1j / (f2 + 1j * f) - 1j / (f1 + 1j * f))
    raise TypeError(f"Unknown dispersion term {term!r}.")  # pragma: no cover


def _by_parameter(term: object, f: complex) -> dict[str, complex]:
    """D chi / dp of one term at ``f`` (Hz) by each parameter, in its own units."""
    if isinstance(term, Drude):
        fp, g = float(term.plasma_frequency), float(term.damping)
        chi = -(fp**2) / (f * (f - 1j * g))
        return {"plasma_frequency": 2 * chi / fp, "damping": chi * 1j / (f - 1j * g)}
    if isinstance(term, Lorentz):
        d, f0, g = (
            float(term.delta_eps),
            float(term.resonance_frequency),
            float(term.damping),
        )
        den = f0**2 - f * f + 1j * g * f
        return {
            "delta_eps": f0**2 / den,
            "resonance_frequency": 2 * d * f0 * (1j * g * f - f * f) / den**2,
            "damping": -d * f0**2 * 1j * f / den**2,
        }
    if isinstance(term, Debye):
        d, fr = float(term.delta_eps), float(term.relaxation_frequency)
        den = 1 + 1j * f / fr
        return {
            "delta_eps": 1 / den,
            "relaxation_frequency": d * 1j * f / (fr * den) ** 2,
        }
    if isinstance(term, DjordjevicSarkar):
        d, f1, f2 = (
            float(term.delta_eps),
            float(term.low_frequency),
            float(term.high_frequency),
        )
        span = math.log(f2 / f1)
        log = cmath.log((f2 + 1j * f) / (f1 + 1j * f))
        scale = d / span
        return {
            "delta_eps": log / span,
            "low_frequency": scale * log / (f1 * span) - scale / (f1 + 1j * f),
            "high_frequency": -scale * log / (f2 * span) + scale / (f2 + 1j * f),
        }
    raise TypeError(f"Unknown dispersion term {term!r}.")  # pragma: no cover


def d_eps(law: PermittivityLaw, omega: complex) -> dict[tuple, complex]:
    """D eps / dp at ``omega`` (rad/s) by each parameter of ``law``.

    Keyed in the order of the law's leaves: ``("eps_inf",)`` (d eps / d
    eps_inf = 1), then ``(i, name)`` for parameter ``name`` of
    ``law.terms[i]``, per Hz for a frequency. Written by hand beside
    `d_susceptibility`: the law pass weighs each law piece's
    form with it. Non-finite (never raising) on a lossless pole.
    """
    f = omega / _TWO_PI
    out: dict[tuple, complex] = {("eps_inf",): 1.0}
    for index, term in enumerate(law.terms):
        try:
            with np.errstate(all="ignore"):
                values = _by_parameter(term, f)
        except ZeroDivisionError:
            names = (fd.name for fd in dataclasses.fields(term))
            values = {name: complex(math.inf, math.nan) for name in names}
        out |= {(index, name): complex(v) for name, v in values.items()}
    return out


def _terms(law: PermittivityLaw, omega: complex) -> list[tuple[complex, complex]]:
    """``(chi, d chi / d f)`` of each term of ``law`` at ``omega`` (rad/s)."""
    f = omega / _TWO_PI
    with np.errstate(all="ignore"):
        return [_term(t, f) for t in law.terms]


def susceptibility(law: PermittivityLaw, omega: complex) -> complex:
    """chi(omega) = eps(omega) - eps_inf, summed over the law's terms.

    Non-finite (never raising) on a lossless pole: `check_frequencies` keeps
    such a frequency out of a sweep.
    """
    try:
        return complex(sum(chi for chi, _ in _terms(law, omega)))
    except ZeroDivisionError:
        return complex(math.inf, math.nan)


def d_susceptibility(law: PermittivityLaw, omega: complex) -> complex:
    """The slope d chi / d omega, analytic in omega."""
    try:
        return complex(sum(d for _, d in _terms(law, omega))) / _TWO_PI
    except ZeroDivisionError:
        return complex(math.inf, math.nan)


def _second(term: object, f: complex) -> complex:
    """``d^2 chi / d f^2`` of one term at ``f`` (Hz)."""
    if isinstance(term, Drude):
        fp, g = float(term.plasma_frequency), float(term.damping)
        den, slope = f * (f - 1j * g), 2 * f - 1j * g
        return fp**2 * (2 * den - 2 * slope**2) / den**3
    if isinstance(term, Lorentz):
        d, f0, g = (
            float(term.delta_eps),
            float(term.resonance_frequency),
            float(term.damping),
        )
        den, slope = f0**2 - f * f + 1j * g * f, -2 * f + 1j * g
        return d * f0**2 * (2 * den + 2 * slope**2) / den**3
    if isinstance(term, Debye):
        d, fr = float(term.delta_eps), float(term.relaxation_frequency)
        den, slope = 1 + 1j * f / fr, 1j / fr
        return 2 * d * slope**2 / den**3
    if isinstance(term, DjordjevicSarkar):
        d, f1, f2 = (
            float(term.delta_eps),
            float(term.low_frequency),
            float(term.high_frequency),
        )
        scale = d / math.log(f2 / f1)
        return scale * (1 / (f2 + 1j * f) ** 2 - 1 / (f1 + 1j * f) ** 2)
    raise TypeError(f"Unknown dispersion term {term!r}.")  # pragma: no cover


def _derivatives(
    law: PermittivityLaw, omega: float
) -> tuple[complex, complex, complex]:
    """``(eps, omega eps', omega^2 eps'')`` of ``law`` at real ``omega`` (rad/s)."""
    f = omega / _TWO_PI
    with np.errstate(all="ignore"):
        try:
            pairs = [_term(t, f) for t in law.terms]
            second = sum(_second(t, f) for t in law.terms)
        except ZeroDivisionError:
            nan = complex(math.nan, math.nan)
            return nan, nan, nan
    eps = complex(law.eps_inf) + sum(chi for chi, _ in pairs)
    # omega d/d omega = f d/d f
    return eps, f * sum(d for _, d in pairs), f * f * second


def energy_eps(law: PermittivityLaw, omega: float) -> float:
    """The energy permittivity Re[eps + (omega / 2) eps'] at ``omega`` (rad/s).

    The weight of ``|E|^2`` in a mode's total energy: half the
    Brillouin electric weight plus Re eps. eps_inf for a lossless Drude law.
    Non-finite on a lossless pole.
    """
    eps, slope, _ = _derivatives(law, omega)
    return float((eps + 0.5 * slope).real)


def d_energy_eps(law: PermittivityLaw, omega: float) -> float:
    """``d eps_W / d omega = Re[(3/2) eps' + (omega / 2) eps'']`` per rad/s."""
    _, slope, curve = _derivatives(law, omega)
    return float((1.5 * slope + 0.5 * curve).real) / omega


def brillouin_eps(law: PermittivityLaw, omega: float) -> float:
    """The Brillouin electric weight Re d(omega eps)/d omega at ``omega`` (rad/s).

    ``gfem.electric_energy`` reads ``1/4 eps0 Re d(omega eps)/d omega |E|^2``;
    Re eps_r for a constant material.
    """
    eps, slope, _ = _derivatives(law, omega)
    return float((eps + slope).real)


# --- checks before assembly --------------------------------------------------------


def resonance_seeds(model: Model, frequencies: Sequence[float]) -> list[int]:
    """Indices into ``frequencies`` the adaptive greedy solves first.

    For each `Lorentz` term of each law region, in ``model.materials`` and
    term order, whose resonance frequency lies strictly inside the grid's
    range: the grid frequency just below it (or on it) and the one just
    above. The field shapes next to a resonance are absent from a basis
    built away from it, so a greedy that has not touched them can stop with
    S wrong. Drude (bounded theta at 0), Debye and Djordjevic-Sarkar terms
    are smooth and add none. Repeats are kept once, first place.
    """
    grid = np.asarray(frequencies, dtype=float)
    seeds: dict[int, None] = {}
    for lr in law_regions(model):
        for term in lr.law.terms:
            if not isinstance(term, Lorentz):
                continue
            f0 = float(term.resonance_frequency)
            if not grid.min() < f0 < grid.max():
                continue
            below = np.flatnonzero(grid <= f0)
            above = np.flatnonzero(grid > f0)
            seeds[int(below[np.argmax(grid[below])])] = None
            seeds[int(above[np.argmin(grid[above])])] = None
    return list(seeds)


def check_frequencies(
    model: Model, frequencies: Iterable[float], *, absorbing: bool = True
) -> None:
    """Raise for an output frequency a law region cannot be solved at.

    With ``absorbing``, a law region behind a face of an `Absorbing` surface
    must hold a propagating medium (Re eps > 0) at every frequency: its face
    pieces read sqrt(Re eps(omega)), at either order.

    Static: it reads the laws and the frequencies only, so every rank of a
    distributed solve raises the same error before any collective. A law
    computed under a JAX transformation is skipped: the driven forward runs
    this again on the host, on its concrete values.

    Raises:
        ValueError: A frequency lies exactly on the resonance of a lossless
            term (its permittivity is infinite there); with ``absorbing``,
            Re eps <= 0 at a frequency on a law region behind an absorbing
            face.
    """
    frequencies = list(frequencies)
    _check_blend_poles(model, frequencies)
    # a traced law is checked on the host, on its concrete values
    regions = [lr for lr in law_regions(model) if not is_traced(lr.law)]
    if not regions:
        return
    behind = absorbing_surfaces(model, regions) if absorbing else {}
    for f in frequencies:
        for lr in regions:
            for index, term in enumerate(lr.law.terms):
                try:
                    chi, _ = _term(term, complex(f))
                    finite = cmath.isfinite(chi)
                except ZeroDivisionError:
                    finite = False
                if not finite:
                    raise ValueError(
                        f"Driven frequency {f:g} Hz lies on the resonance of "
                        f"law.terms[{index}] ({type(term).__name__}, damping = "
                        f"{float(getattr(term, 'damping', 0.0)):g} Hz) of the "
                        f"permittivity law of region {lr.region!r}: its "
                        "permittivity is infinite there. Give the term a damping, "
                        "or leave this frequency out."
                    )
            surfaces = behind.get(lr.region, ())
            if not surfaces:
                continue
            value = float(lr.law.eps_inf) + susceptibility(lr.law, 2 * math.pi * f).real
            if not value > 0:
                raise ValueError(
                    f"The Absorbing surface {surfaces[0]!r} lies on region "
                    f"{lr.region!r}, whose permittivity law gives Re eps = "
                    f"{value:.3g} at {f:g} Hz. An absorbing boundary needs a "
                    "propagating medium behind it (Re eps > 0): surround the "
                    "metal with a dielectric or air layer and put the boundary "
                    "there."
                )


#: How close to an edge of a fitted range, relative, a frequency still counts
#: as inside: it absorbs the rounding of a frequency computed as c / wavelength.
#: There is no other margin, because each edge is the source's own.
FITTED_EDGE = 1e-9


def _fitted_laws(
    model: Model, regions: Collection[str] | None
) -> list[tuple[str, str, PermittivityLaw]]:
    """``(key, label, law)`` of each law of ``model`` that has a fitted range.

    The law regions in ``model.materials`` order, each with the ends of its
    blend after it; ``regions`` keeps those regions only.
    """
    out = []
    for region, material in model.materials.items():
        if regions is not None and region not in regions:
            continue
        if isinstance(material, Material):
            law = material.eps_r
            if isinstance(law, PermittivityLaw) and law.frequency_range is not None:
                name = f" ({material.name})" if material.name else ""
                out.append((region, f"region {region!r}{name}", law))
        elif isinstance(material, MaterialField) and material.blend_ends is not None:
            ends = material.blend_ends
            for which, end in (("void", ends.void), ("solid", ends.solid)):
                if isinstance(end, PermittivityLaw) and end.frequency_range is not None:
                    out.append(
                        (
                            f"{region}:{which}",
                            f"the {which} end of the blend in region {region!r}",
                            end,
                        )
                    )
    return out


def warn_fitted_range(
    model: Model,
    frequencies: Iterable[float],
    what: str,
    *,
    comm: Any = None,
    noun: str = "frequencies",
    static: bool = False,
    regions: Collection[str] | None = None,
    skip: Collection[str] = (),
) -> set[str]:
    """Warn once if a law is read outside its fitted range.

    Every law of a law region, and every law end of a blend, whose
    ``frequency_range`` does not hold one of ``frequencies`` (Hz) gives one
    line of a single `gfem.FittedRangeWarning`; within `FITTED_EDGE` of an
    edge counts as inside. The law stays passive outside its range, so the
    solve runs on. Static: it reads the ranges and the frequencies only, so
    it runs on a traced model too, and every rank of a distributed solve
    sees the same warning, which rank 0 shows (`gfem._mpi.warn`).

    Args:
        model: The model.
        frequencies: The frequencies the solve reads the laws at, Hz.
        what: The reader in the message: ``"driven solve"``,
            ``"eigenmode target"``...
        comm: The solve's communicator.
        noun: What ``frequencies`` count in the message (``"modes"``).
        static: The solve reads each law's static permittivity (0 Hz): the
            message gives that value and the electrostatic remedy.
        regions: Check these regions only (a port's cross-section).
        skip: Keys already warned for in this solve (see Returns).

    Returns:
        The keys of the laws warned for (a region, or ``region:end`` for a
        blend end), so a later read in the same solve does not repeat them.
    """
    freqs = [float(f) for f in frequencies]
    lines = []
    keys = set()
    for key, label, law in _fitted_laws(model, regions):
        if key in skip:
            continue
        lo, hi = law.frequency_range  # ty: ignore[not-iterable]
        outside = [
            f for f in freqs if f < lo * (1 - FITTED_EDGE) or f > hi * (1 + FITTED_EDGE)
        ]
        if not outside:
            continue
        keys.add(key)
        if static:
            where = "0 Hz"
            if not is_traced(law):
                where += f", static permittivity {complex(law(0.0)).real:.4g}"
            remedy = (
                "Give the region a constant gfem.Material(eps_r=<its static "
                "permittivity>) in the electrostatic model"
            )
        else:
            if len(freqs) == 1:
                where = hertz(outside[0], 4)
            else:
                span = hertz_band(min(outside), max(outside), 4)
                where = f"{len(outside)} of {len(freqs)} {noun}, {span}"
            if min(outside) < lo and any(isinstance(t, Drude) for t in law.terms):
                remedy = "For a metal at RF use gfem.Conductor with its bulk conductivity"
            else:
                remedy = "Narrow the frequencies or fit data that cover them"
        lines.append(
            f"{label}: its permittivity law was fitted over {hertz_band(lo, hi, 4)}, "
            f"and this {what} reads it at {where}. The law stays passive there, "
            f"but its accuracy is unknown. {remedy}, or clear the range with "
            "dataclasses.replace(law, frequency_range=None) once checked."
        )
    if lines:
        _mpi.warn(comm, "\n".join(lines), FittedRangeWarning, stacklevel=2)
    return keys


def _check_blend_poles(model: Model, frequencies: Iterable[float]) -> None:
    """Raise for a frequency on a lossless pole of a law blend's end.

    Reads the ends only (static data), never the density, so it runs on a
    traced model too.

    Raises:
        ValueError: A frequency lies on the resonance of a lossless term.
    """
    frequencies = list(frequencies)
    for region, ends in _blends(model):
        for which, end in (("void", ends.void), ("solid", ends.solid)):
            if not isinstance(end, PermittivityLaw):
                continue
            for f in frequencies:
                if not cmath.isfinite(susceptibility(end, 2 * math.pi * f)):
                    raise ValueError(
                        f"Driven frequency {f:g} Hz lies on a lossless resonance "
                        f"of the permittivity law at the {which} end of the blend "
                        f"in region {region!r}: its permittivity is infinite "
                        "there. Give the term a damping, or leave this frequency "
                        "out."
                    )


def absorbing_surfaces(
    model: Model, regions: Iterable[LawRegion] | None = None
) -> dict[str, tuple[str, ...]]:
    """The `Absorbing` surfaces with a face on each law region, by region.

    A face belongs to the tetrahedron its terms read
    (`gfem._engine.terms.face_owner`). Regions without such a face are left
    out.
    """
    # imported here: terms imports this module
    from gfem._engine.terms import face_owner

    regions = law_regions(model) if regions is None else tuple(regions)
    absorbing = [b.surface for b in model.boundaries if isinstance(b, Absorbing)]
    if not regions or not absorbing:
        return {}
    mesh = model.mesh
    owner = face_owner(model)
    out = {}
    for lr in regions:
        found = []
        for surface in dict.fromkeys(absorbing):
            tags = mesh._topology.surface_tags  # noqa: SLF001 - either dimension
            tets = owner[tags == mesh.region(surface).tag]
            if lr.mask[tets[tets >= 0]].any():
                found.append(surface)
        if found:
            out[lr.region] = tuple(found)
    return out


# --- the eigenmode problem's unresolved band ----------------------------

#: k0 Re sqrt(eps) h_max above which a law region is in its unresolved band:
#: its wavelength is shorter than two of its longest edges.
BAND = math.pi
#: Each term's frequencies a message measures nearness by.
_CORNERS: dict[type, tuple[str, ...]] = {
    Drude: ("plasma_frequency",),
    Lorentz: ("resonance_frequency",),
    Debye: ("relaxation_frequency",),
    DjordjevicSarkar: ("low_frequency", "high_frequency"),
}
_NAMES = {
    "plasma_frequency": "plasma frequency",
    "resonance_frequency": "resonance",
    "relaxation_frequency": "relaxation frequency",
    "low_frequency": "low corner",
    "high_frequency": "high corner",
}


def nearest_term(law: PermittivityLaw, frequency: float) -> str:
    """The term of ``law`` nearest ``frequency`` (Hz), for messages.

    Nearness is measured to each term's own frequencies (a Lorentz term's
    resonance, a Drude term's plasma frequency, a Debye term's relaxation
    frequency, a Djordjevic-Sarkar term's corners): ``"law.terms[0]
    (Lorentz, resonance 25 GHz)"``.
    """
    best = None
    for index, term in enumerate(law.terms):
        for name in _CORNERS[type(term)]:
            value = float(getattr(term, name))
            distance = abs(value - frequency)
            if best is None or distance < best[0]:
                best = (distance, index, term, name, value)
    if best is None:  # pragma: no cover - a law has terms
        return "its high-frequency permittivity"
    _, index, term, name, value = best
    return f"law.terms[{index}] ({type(term).__name__}, {_NAMES[name]} {hertz(value)})"


def _longest_edges(model: Model, regions: Iterable[LawRegion]) -> dict[str, float]:
    """The longest element edge (m) of each law region on the solve domain."""
    nodes = np.asarray(model.mesh.nodes, dtype=float)
    cells = np.asarray(model.mesh.volume_elements)
    out = {}
    for lr in regions:
        corners = nodes[cells[lr.mask]]
        if not len(corners):
            continue
        n = corners.shape[1]
        edges = [
            np.linalg.norm(corners[:, i] - corners[:, j], axis=1).max()
            for i in range(n)
            for j in range(i + 1, n)
        ]
        out[lr.region] = float(max(edges))
    return out


def check_band(model: Model, frequency: float, what: str) -> None:
    """Raise if ``frequency`` (Hz) lies in a law region's unresolved band.

    A law region with longest edge ``h_max`` is in its band where ``k0 Re
    sqrt(eps(f)) h_max > pi`` (`BAND`, principal root): its propagating
    wavelength is shorter than two longest edges. There, next to a Lorentz
    resonance, the modes crowd at the resonance, which of them the mesh
    holds depends on it, and the seed solve cannot tell the nearest apart.
    An infinite permittivity (a lossless
    pole) is in the band. ``what`` names the frequency in the message (the
    target, or a mode).

    Raises:
        ValueError: ``frequency`` lies in a law region's band.
    """
    regions = law_regions(model)
    if not regions:
        return
    omega = _TWO_PI * frequency
    k0 = omega / 299792458.0
    for lr in regions:
        h = _longest_edges(model, [lr]).get(lr.region)
        if h is None:
            continue
        eps = complex(lr.law.eps_inf) + susceptibility(lr.law, omega)
        value = k0 * cmath.sqrt(eps).real * h if cmath.isfinite(eps) else math.inf
        if value > BAND:
            shown = f"{eps:.3g}" if cmath.isfinite(eps) else "infinite"
            raise ValueError(
                f"Eigenmode's {what} {hertz(frequency)} lies in the unresolved band "
                f"of region {lr.region!r}: its permittivity law gives eps = {shown} "
                f"there, near {nearest_term(lr.law, frequency)}, and k0 Re "
                f"sqrt(eps) h_max = {value:.3g} > pi (h_max = {h * 1e3:.3g} mm, the "
                "region's longest edge): its wavelength is shorter than two "
                "elements. The modes there crowd at the resonance, and which of "
                "them the mesh holds depends on the mesh. Move Eigenmode.target "
                "away from the resonance, or refine the mesh in the region."
            )


#: The remedy of a weight that is not positive.
_LOSS_DOMINATES = (
    "the energy is undefined where loss dominates the law (a metal below "
    "its damping frequency, a Lorentz pole inside its linewidth): model a "
    "metal as gfem.Conductor or gfem.PEC"
)


def check_energy(
    model: Model,
    frequency: float,
    what: str,
    *,
    brillouin: bool = False,
    regions: Iterable[str] | None = None,
) -> None:
    """Raise where a law region's energy weight at ``frequency`` (Hz) is <= 0.

    The mode energy reads the energy permittivity `energy_eps`; with
    ``brillouin``, `gfem.electric_energy` reads `brillouin_eps`.
    ``what`` names the frequency in the message (a mode, or a field);
    ``regions`` limits the check to those regions.

    Raises:
        ValueError: The weight is not positive (or not finite) in a law
            region.
    """
    omega = _TWO_PI * frequency
    weight, name = (
        (brillouin_eps, "Re d(omega eps)/d omega")
        if brillouin
        else (energy_eps, "the energy permittivity Re[eps + (omega/2) eps']")
    )
    keep = None if regions is None else set(regions)
    for lr in law_regions(model):
        if keep is not None and lr.region not in keep:
            continue
        value = weight(lr.law, omega)
        if not value > 0:
            remedy = (
                "or move Eigenmode.target."
                if not brillouin
                else "or solve at another frequency."
            )
            raise ValueError(
                f"The {what} {hertz(frequency)} has {name} = {value:.3g} <= 0 in "
                f"region {lr.region!r}, near {nearest_term(lr.law, frequency)}: "
                f"{_LOSS_DOMINATES}, {remedy}"
            )


# --- refinement estimators ---------------------------------------------------------


def has_loss(law: PermittivityLaw) -> bool:
    """Whether a term of ``law`` carries loss (Im eps < 0 at real frequencies).

    A damped Drude or Lorentz term, and every Debye and Djordjevic-Sarkar
    term; a lossless law is real but not eps_inf. A traced damping counts
    as loss (the refining entry points refuse a traced law).
    """
    return any(
        not isinstance(term, (Drude, Lorentz))
        or is_traced(term.damping)
        or float(term.damping) > 0
        for term in law.terms
    )


#: |eps| / max(1, |eps_inf|) at or below which a law's permittivity is zero.
_ZERO = 1e-9


def check_estimator_frequencies(model: Model, frequencies: Iterable[float]) -> None:
    """Raise for a frequency where a law region's eps_eff is zero.

    The flux recovery of every wave estimator weights a tetrahedron's flux
    error with |eps_eff|^-1: at eps(omega) = 0 (a lossless Drude term at its
    plasma frequency; zero to `_ZERO` relative to eps_inf, since rounding
    leaves a residual) and no conductivity it has no weight. The solve
    itself is defined there, so only the estimators refuse.

    Raises:
        ValueError: A law region without conductivity has eps = 0 at one of
            ``frequencies`` (Hz).
    """
    regions = [
        lr
        for lr in law_regions(model)
        # a traced law is refused by the refining entry points themselves
        if not is_traced(lr.law)
        and not np.any(np.asarray(model.materials[lr.region].conductivity))
    ]
    for f in frequencies:
        for lr in regions:
            eps = complex(lr.law.eps_inf) + susceptibility(lr.law, _TWO_PI * f)
            # relative to eps_inf: a computed zero is a rounding residual
            if abs(eps) <= _ZERO * max(1.0, abs(float(lr.law.eps_inf))):
                raise ValueError(
                    f"The permittivity law of region {lr.region!r} gives |eps| = 0 "
                    f"at {f / 1e9:g} GHz ({f:g} Hz): the flux recovery of the "
                    "error estimate has no weight there. Leave this frequency out "
                    "of the estimated sweep, or estimate at a nearby one."
                )
