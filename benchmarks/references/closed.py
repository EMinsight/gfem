r"""Closed-form references, evaluated at 40 significant digits with ``decimal``.

- ``lcavity``: :math:`f_1 = c \sqrt{\lambda_1} / (2\pi)` with the first Dirichlet
  eigenvalue :math:`\lambda_1 = 9.6397238440219\,\mathrm{mm^{-2}}` of the L of
  side 2 (closed to about 1e-13), so the bound is 1e-12 relative. The rounded
  148.14026965 GHz constant of the tests is 1.4e-11 low and is not used.
- ``bragg``: the two lowest roots of the transfer-matrix (Rytov) dispersion relation
  of the two-layer cell at :math:`k = \pi / a`,
  :math:`\cos(n_1 k_0 d_1) \cos(n_2 k_0 d_2) - \tfrac12 (n_1/n_2 + n_2/n_1)
  \sin(n_1 k_0 d_1) \sin(n_2 k_0 d_2) = \cos(k a)`. With PEC z walls and PMC y
  walls those bands are exact TEM layer modes; the roots are simple. Bound 1e-12
  relative, ceiling 1e-5 relative (a physics case).
- ``r10M_void``: no converged reference (``method: "none"``); the file pins the DOF.

The numbers are checked against mpmath values in the unit tests. DOF
pins, same-mesh errors and ceilings are the observed figures named beside them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from decimal import Decimal, localcontext
from typing import Any

from benchmarks.harness.case import Case
from benchmarks.references import document as refdoc

DIGITS = 40
_WORK = DIGITS + 10
C0 = Decimal(299792458)
"""Speed of light in vacuum, m/s (exact)."""

LCAVITY_LAMBDA1 = Decimal("9.6397238440219")
"""First Dirichlet eigenvalue of the L of side 2, mm^-2 (tests/refinement_cases.py
``CAVITY_LAMBDA1`` at 858bc66)."""
BRAGG_A_MM, BRAGG_D1_MM = Decimal(10), Decimal(4)
"""Cell period and low-index layer thickness, mm (tests/periodic_cases.py ``C3_A``,
``C3_D1`` at 858bc66; the pinned mesh is that cell)."""
BRAGG_EPS_LOW, BRAGG_EPS_HIGH = Decimal(2), Decimal(9)
BRAGG_GUESSES_HZ = (Decimal("5.1855e9"), Decimal("7.2068e9"))
CLOSED_BOUND = 1e-12
BRAGG_CEILING = 1e-5

SOURCES = {
    "lcavity": "c sqrt(lambda1) / (2 pi), lambda1 = 9.6397238440219 mm^-2 "
    "(tests/refinement_cases.py CAVITY_LAMBDA1 at 858bc66)",
    "bragg": "TMM (Rytov) roots at k = pi/a, a = 10 mm, d1 = 4 mm, eps 2 and 9 "
    "(tests/periodic_cases.py tmm_frequency and C3_* at 858bc66)",
}
DOF = {
    "lcavity": {"2": 246_416},
    "bragg": {"3": 10_620},
    "r10M_void": {"2": 10_248_856},
}
"""DOF per order on each case's pinned mesh (gfem adapter runs)."""
SAME_MESH_ERROR = {"lcavity": {"2": 3.8907e-4}, "bragg": {"3": 2.9203e-11}}
"""gfem direct's relative error on the pinned mesh. lcavity: f1
(tests/refinement_cases.py ``CAVITY_SPLIT`` level 3, the same 246 416-DOF mesh).
bragg: max over the two modes of the gfem direct runs at 1x13 and 1x16 on the
once-split mesh (2.92e-11 each; the R1 mesh gave 1.86e-9, a factor 64 = 2^6 per
split, the p3 eigenvalue rate); its ceiling stays the explicit physics-case 1e-5."""
NO_REFERENCE = (
    "no_reference: no converged reference fits the budget (one split is about "
    "80 M DOF); records carry the same-code error only"
)


# --- decimal arithmetic ------------------------------------------------------------


def pi() -> Decimal:
    r""":math:`\pi` at the context's precision (the ``decimal`` docs recipe)."""
    with localcontext() as ctx:
        ctx.prec += 2
        three = Decimal(3)
        lasts, t, s, n, na, d, da = Decimal(0), three, Decimal(3), 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
    return +s


def _taylor(x: Decimal, first: Decimal, start: int) -> Decimal:
    """Sum of the alternating series of cos (``start`` 0) or sin (``start`` 1)."""
    with localcontext() as ctx:
        ctx.prec += 2
        i, lasts, s, fact, num, sign = start, Decimal(0), first, Decimal(1), first, 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= x * x
            sign *= -1
            s += num / fact * sign
    return +s


def cos(x: Decimal) -> Decimal:
    """Cosine by its Taylor series."""
    return _taylor(x, Decimal(1), 0)


def sin(x: Decimal) -> Decimal:
    """Sine by its Taylor series."""
    return _taylor(x, x, 1)


def _secant(f: Callable[[Decimal], Decimal], x0: Decimal) -> Decimal:
    x1 = x0 * (1 + Decimal("1e-6"))
    f0, f1 = f(x0), f(x1)
    for _ in range(200):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) <= abs(x2) * Decimal(10) ** -(_WORK - 2):
            return x2
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return x1


# --- the closed forms --------------------------------------------------------------


def lcavity_f1() -> Decimal:
    """The L-cavity's first resonance, Hz, at 40 significant digits."""
    with localcontext() as ctx:
        ctx.prec = _WORK
        f1 = C0 * LCAVITY_LAMBDA1.sqrt() * 1000 / (2 * pi())
    with localcontext() as ctx:
        ctx.prec = DIGITS
        return +f1


def bragg_modes() -> tuple[Decimal, Decimal]:
    """The two lowest band frequencies of the bragg cell at the zone edge, Hz."""
    with localcontext() as ctx:
        ctx.prec = _WORK
        a = BRAGG_A_MM / 1000
        d1 = BRAGG_D1_MM / 1000
        d2 = a - d1
        n1, n2 = BRAGG_EPS_LOW.sqrt(), BRAGG_EPS_HIGH.sqrt()
        p = pi()
        mix = (n1 / n2 + n2 / n1) / 2
        cos_ka = cos(p)  # k a = pi

        def dispersion(f: Decimal) -> Decimal:
            k0 = 2 * p * f / C0
            u, v = n1 * k0 * d1, n2 * k0 * d2
            return cos(u) * cos(v) - mix * sin(u) * sin(v) - cos_ka

        roots = [_secant(dispersion, g) for g in BRAGG_GUESSES_HZ]
    with localcontext() as ctx:
        ctx.prec = DIGITS
        return +roots[0], +roots[1]


# --- the files ---------------------------------------------------------------------

CASES = ("lcavity", "bragg", "r10M_void")


def _frequencies(values: tuple[Decimal, ...]) -> dict[str, Any]:
    return {
        "values": [float(v) for v in values],
        "values_decimal": [str(v) for v in values],
        "bound": CLOSED_BOUND,
        "bound_kind": "relative",
        "metric": "max_rel_dRe_f",
    }


def document(
    case: Case, *, commit: str = "0000000", now: dt.datetime | None = None
) -> dict[str, Any]:
    """The ``reference.json`` content of a closed-form (or no-reference) case."""
    if case.name not in CASES:
        raise ValueError(f"{case.name}: no closed form (known: {', '.join(CASES)})")
    doc = refdoc.skeleton(case, commit=commit, now=now or dt.datetime.now(dt.UTC))
    doc["meshes"][case.mesh.sha256]["dof"] = dict(DOF[case.name])
    script = "benchmarks/references/closed.py"
    if case.name == "r10M_void":
        doc["reason"] = NO_REFERENCE
        doc["provenance"]["script"] = script
        return _ordered(doc)
    values = (lcavity_f1(),) if case.name == "lcavity" else bragg_modes()
    doc["method"] = "closed_form"
    doc["quantities"] = {"f_hz": _frequencies(values)}
    errors = SAME_MESH_ERROR.get(case.name, {})
    doc["same_mesh_error"] = dict(errors)
    if case.name == "bragg":
        doc["ceiling"] = {str(case.order): BRAGG_CEILING}
    else:
        doc["ceiling"] = {k: float(f"{10 * v:.6g}") for k, v in errors.items()}
    doc["provenance"].update(script=script, formula=SOURCES[case.name])
    return _ordered(doc)


def _ordered(doc: dict[str, Any]) -> dict[str, Any]:
    order = [*refdoc.REQUIRED[:8], "reason", *refdoc.REQUIRED[8:]]
    return {k: doc[k] for k in order if k in doc}
