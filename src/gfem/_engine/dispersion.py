"""Every per-element read of a permittivity law (ADR 0111).

A `Material` whose ``eps_r`` is a `gfem.PermittivityLaw` has no one
permittivity: `gfem._engine.coefficients.element_values` refuses it, and the
readers that understand a law read it here, each at a named frequency:

- `high_frequency_eps`: the law's eps_inf, the shared mass piece's value
  (ADR 0110);
- `eps_at`: the complex permittivity at an angular frequency;
- `law_regions`: the laws and their tetrahedra, for the susceptibility
  pieces (`susceptibility`, `d_susceptibility`) and the checks;
- `check_frequencies`: what a driven sweep checks before any assembly;
- `absorbing_surfaces`: the absorbing surfaces each law region lies behind.

The law is evaluated here on the host in complex scalars, by hand, term by
term (``chi`` and ``d chi / d omega``), never through JAX: assembly does not
trace. Frequencies are angular (rad/s); the laws take hertz.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from collections.abc import Iterable

import numpy as np

from gfem._boundaries import Absorbing
from gfem._checks import is_traced
from gfem._engine.coefficients import lossy_eps
from gfem._materials import (
    Debye,
    DjordjevicSarkar,
    Drude,
    Lorentz,
    Material,
    PermittivityLaw,
)
from gfem.model import Model

__all__ = [
    "LawRegion",
    "absorbing_surfaces",
    "check_frequencies",
    "d_susceptibility",
    "eps_at",
    "high_frequency_eps",
    "law_regions",
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


def high_frequency_eps(model: Model) -> np.ndarray:
    """`gfem._engine.coefficients.lossy_eps` with eps_inf on the law regions.

    The value of the shared mass piece (ADR 0110): a law's frequency
    dependence is its susceptibility piece.
    """
    eps = lossy_eps(model, law=0.0)
    for lr in law_regions(model):
        value = float(lr.law.eps_inf)
        if eps.ndim == 1:
            eps[lr.mask] = value
        else:
            eps[lr.mask] = value * np.eye(3)
    return eps


def eps_at(model: Model, omega: complex) -> np.ndarray:
    """The complex relative permittivity per tetrahedron at ``omega`` (rad/s).

    ``eps_r (1 - i tan_delta)`` off the law regions (`lossy_eps`'s shape),
    eps_inf + chi(omega) on them. The conductivity is not included. ``omega``
    may be complex (the analytic continuation).
    """
    eps = high_frequency_eps(model).astype(complex)
    for lr in law_regions(model):
        chi = susceptibility(lr.law, omega)
        if eps.ndim == 1:
            eps[lr.mask] += chi
        else:
            eps[lr.mask] += chi * np.eye(3)
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
