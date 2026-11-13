"""Every per-element read of a permittivity law (ADR 0111).

A `Material` whose ``eps_r`` is a `gfem.PermittivityLaw` has no one
permittivity: `gfem._engine.coefficients.element_values` refuses it, and the
readers that understand a law read it here, each at a named frequency:

- `high_frequency_eps`: the law's eps_inf, the shared mass piece's value
  (ADR 0110);
- `eps_at`: the complex permittivity at an angular frequency;
- `static_eps`: the static permittivity eps(0), the electrostatic
  coefficient (`check_static` refuses a Drude term);
- `law_regions`: the laws and their tetrahedra, for the susceptibility
  pieces (`susceptibility`, `d_susceptibility`) and the checks;
- `blend_regions`: the density blends with a law end (ADR 0113), whose
  eps_inf limit `high_frequency_eps` holds and whose susceptibility parts
  (`blend_parts`) the driven operator reads; `gfem.solve` refuses them on
  every other problem;
- `check_frequencies`: what a driven sweep checks before any assembly;
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
from collections.abc import Callable, Iterable, Sequence

import numpy as np

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
from gfem.model import Model

__all__ = [
    "BlendPart",
    "BlendRegion",
    "LawRegion",
    "absorbing_surfaces",
    "blend_mass",
    "blend_parts",
    "blend_regions",
    "check_estimator_frequencies",
    "check_frequencies",
    "check_static",
    "d_eps",
    "d_susceptibility",
    "eps_at",
    "has_loss",
    "high_frequency_eps",
    "law_regions",
    "resonance_seeds",
    "static_eps",
    "susceptibility",
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
            mask = np.zeros(len(mesh.tetrahedra), bool)
            mask[mesh.elements(region)] = True
            out.append(LawRegion(region, material.eps_r, mask & active))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class BlendRegion:
    """A region whose `MaterialField` blends a law end by a density (ADR 0113).

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
    """The model's law blends, in ``model.materials`` order (concrete densities)."""
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    out = []
    for region, material in model.materials.items():
        if isinstance(material, MaterialField) and material.blend_ends is not None:
            elements = mesh.elements(region)
            mask = np.zeros(len(mesh.tetrahedra), bool)
            mask[elements] = True
            rho = np.zeros(len(mesh.tetrahedra))
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
    """The susceptibility parts of a law blend (ADR 0113).

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

    The value of the shared mass piece (ADR 0110): a law's frequency
    dependence is its susceptibility piece. A law blend's region holds its
    eps_inf limit (`blend_mass`, ADR 0113).
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
    known (ticket 09 Q4). A lossless Drude term diverges at 0 Hz and a lossy
    one gives Re chi(0) = -ωp²/γ² < 0, which would make the electrostatic
    operator indefinite; dropping the term would give a metal eps_inf.

    A density blend with a law end (ADR 0113) is refused too: with a Drude
    end for the same reason, otherwise because the electrostatic problem
    does not read a blend.

    Raises:
        ValueError: A law region's law, or an end of a law blend, has a
            `gfem.Drude` term.
        NotImplementedError: A law blend without a Drude end.
    """
    for region, material in model.materials.items():
        ends = material.blend_ends if isinstance(material, MaterialField) else None
        if ends is not None and ends.drude:
            raise ValueError(
                f"Region {region!r} blends a permittivity law with a Drude term, "
                "which has no static permittivity (eps(0) is infinite), so the "
                "electrostatic problem cannot read it. Use a constant material "
                "for the electrostatic model."
            )
        if ends is not None:
            raise NotImplementedError(
                f"Region {region!r} is a density blend with a permittivity-law "
                "end (MaterialField.interpolate), which the Electrostatic "
                "problem does not support: a blend is solved by gfem.Driven on "
                "the uniform sweep only."
            )
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
    `d_susceptibility` (ADR 0112): the law pass weighs each law piece's
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


# --- checks before assembly --------------------------------------------------------


def resonance_seeds(model: Model, frequencies: Sequence[float]) -> list[int]:
    """Indices into ``frequencies`` the adaptive greedy solves first (ADR 0116).

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
    pieces read sqrt(Re eps(omega)) (ADR 0110, ticket 09 Q3), at either order.

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


def _check_blend_poles(model: Model, frequencies: Iterable[float]) -> None:
    """Raise for a frequency on a lossless pole of a law blend's end (ADR 0113).

    Reads the ends only (static data), never the density, so it runs on a
    traced model too.

    Raises:
        ValueError: A frequency lies on the resonance of a lossless term.
    """
    for region, material in model.materials.items():
        ends = material.blend_ends if isinstance(material, MaterialField) else None
        if ends is None:
            continue
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
            tets = owner[mesh.triangle_tags == mesh.region(surface).tag]
            if lr.mask[tets[tets >= 0]].any():
                found.append(surface)
        if found:
            out[lr.region] = tuple(found)
    return out


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


def check_estimator_frequencies(model: Model, frequencies: Iterable[float]) -> None:
    """Raise for a frequency where a law region's eps_eff is zero (ADR 0111).

    The flux recovery of every wave estimator weights a tetrahedron's flux
    error with |eps_eff|^-1: at eps(omega) = 0 exactly (a lossless Drude term
    at its plasma frequency) and no conductivity it has no weight. The solve
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
            if eps == 0:
                raise ValueError(
                    f"The permittivity law of region {lr.region!r} gives |eps| = 0 "
                    f"at {f / 1e9:g} GHz ({f:g} Hz): the flux recovery of the "
                    "error estimate has no weight there. Leave this frequency out "
                    "of the estimated sweep, or estimate at a nearby one."
                )
