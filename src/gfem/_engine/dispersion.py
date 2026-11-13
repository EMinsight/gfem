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
- `check_band`: the eigenmode problem's unresolved band (ADR 0115), and
  `nearest_term`, the term a message names;
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
from gfem._text import hertz
from gfem.model import Model

__all__ = [
    "LawRegion",
    "absorbing_surfaces",
    "check_band",
    "check_estimator_frequencies",
    "check_frequencies",
    "d_eps",
    "d_susceptibility",
    "eps_at",
    "has_loss",
    "high_frequency_eps",
    "law_regions",
    "nearest_term",
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


# --- the eigenmode problem's unresolved band (ADR 0115) ----------------------------

#: k0 Re sqrt(eps) h_max above which a law region is in its unresolved band:
#: its wavelength is shorter than two of its longest edges.
BAND = math.pi
#: Each term's frequencies a message measures nearness by.
_CORNERS = {
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
    """The longest tetrahedron edge (m) of each law region on the solve domain."""
    nodes = np.asarray(model.mesh.nodes, dtype=float)
    tets = np.asarray(model.mesh.tetrahedra)
    out = {}
    for lr in regions:
        corners = nodes[tets[lr.mask]]
        if not len(corners):
            continue
        edges = [
            np.linalg.norm(corners[:, i] - corners[:, j], axis=1).max()
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        out[lr.region] = float(max(edges))
    return out


def check_band(model: Model, frequency: float, what: str) -> None:
    """Raise if ``frequency`` (Hz) lies in a law region's unresolved band.

    A law region with longest edge ``h_max`` is in its band where ``k0 Re
    sqrt(eps(f)) h_max > pi`` (`BAND`, principal root): its propagating
    wavelength is shorter than two longest edges. There, next to a Lorentz
    resonance, the modes crowd at the resonance, which of them the mesh
    holds depends on it, and the seed solve cannot tell the nearest apart
    (ADR 0115 and its amendment). An infinite permittivity (a lossless
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
