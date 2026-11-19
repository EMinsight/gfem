r"""The numbers of a campaign report, as plain JSON values (``report.json``).

:func:`build` turns the checked inputs (:mod:`benchmarks.report.load`) into one
``suite-report/1`` mapping: every median, spread, ratio and interpolated cost the
Markdown prints, so a later reader builds on data, never on scraped Markdown.
Performance is observed, never judged: no value here is a pass or a fail.

- A **cell** is the passing records of one (case, core budget, node class, solver,
  stack, layout, tolerance), the default layout apart from the grid. Every per-case
  view is computed per (core budget, node class); a case with several holds them
  under ``budgets`` (ADR 0452), a case with one keeps them at its top level. A
  record passes when its status is ``ok`` and no check failed; every other record
  goes to the failures table and to no view.
- Medians over passing repeats, each bucket taken independently; a ``null`` bucket
  stays ``null``. Spread is :math:`(\max - \min) / \mathrm{median}` of work time,
  **noisy** above 10 %.
- **Best** layout: the lowest median work time among the cells at the anchor
  tolerance with at least three passing repeats; cells whose min-max range overlaps
  the best's are **ties**; cells with fewer repeats are no candidates.
- **Equal achieved error**: each iterative solver's work time and memory read at
  the ends, the geometric midpoint and every decade of the overlap of the solvers'
  same-code error ranges (:func:`overlap_points`), by log-log
  linear interpolation between neighbouring rungs, never extrapolated.
- A cross-code ratio uses cells of one Slurm job; across jobs it carries the label
  **different nodes**.
- **Scaling** (``scaling``, a case with passing cells at two or more budgets): per
  solver its best-layout cell at every (core budget, node class), with speedup
  :math:`S(c) = T(c_0)/T(c)` and parallel efficiency
  :math:`E(c) = c_0 T(c_0) / (c\,T(c))` against its smallest budget :math:`c_0` of
  the same node class, and the cross-code ratios per budget.

Standard library only (ADR 0400).
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

from benchmarks.adapters import ADAPTERS
from benchmarks.harness import case as case_mod
from benchmarks.harness import config, records, score
from benchmarks.report.load import (
    Inputs,
    LoadedCampaign,
    ReportError,
    reference_families,
)

SCHEMA = "suite-report/1"
NOISY = 0.10
AT_FLOOR = 0.01  # a run within 1 % of its order's discretization floor (ADR 0457)
SLOW_NODE = 0.10
REPEATS = 3
DEFAULT_ANCHOR = 1e-8
DIFFERENT_NODES = "different nodes"
SAME_MESH = "same-mesh comparison"
DIFFERENT_DISCRETIZATIONS = "different discretizations"
TREND_NEEDS = "trend needs two campaigns"
UNRESOLVED = "unresolved"
TCP = "TCP between nodes"
FIRST_NODE = "first node only"
NOMINAL = (
    "nominal tolerances differ; iterative kinds compare at equal achieved error "
    "(the equal-error table of each budget)"
)
PHASES = (
    "mesh_read_s",
    "discretize_s",
    "assemble_s",
    "setup_s",
    "solve_s",
    "estimate_s",
    "postpro_s",
    "rom_s",
    "other_s",
)
TIMING_FIELDS = (
    "work_s",
    "startup_s",
    "setup_total_s",
    "solve_total_s",
    "per_sample_s",
    *PHASES,
)
MEMORY_FIELDS = (
    "peak_rss_sum_gib",
    "peak_rss_rank_max_gib",
    "peak_rss_largest_node_gib",
    "pss_peak_sum_gib",
    "sacct_max_rss_gib",
)
CAVEATS = (
    (
        "Times and memory are observations of one machine, software stack and "
        "campaign; nothing here is a pass/fail, and no ratio is a claim beyond this "
        "campaign."
    ),
    (
        "Headline time is work time (wall minus startup); startup is shown apart. "
        "Estimator and postprocessing are inside work time because Palace users cannot "
        "switch them off."
    ),
    (
        "Memory is the sum of per-rank peak RSS (shared pages counted per rank); "
        "PSS and sacct are cross-checks."
    ),
    (
        "Palace phases are max-of-rows (an upper bound on the slowest rank); rank 0 "
        "shown beside."
    ),
    (
        "Nominal Krylov tolerances mean different things in the two codes; iterative "
        "codes are compared only at equal achieved error."
    ),
    (
        "Accuracy is against the named converged reference; unresolved errors are "
        "shown as value ± reference bound; Palace errors carry the definition offset "
        "δ; same-code error is solver error, not accuracy."
    ),
    (
        "Each code runs at its own defaults except the listed pins; structural "
        "differences are listed per case, never removed."
    ),
    (
        "Excluded records: everything in the failures table. Medians over passing "
        "repeats; n < 3 shown."
    ),
    (
        'Ratios across Slurm jobs are labelled "different nodes"; node class is '
        "observed, not controlled."
    ),
)


# --- small helpers ------------------------------------------------------------------


def num(x: float | None) -> float | None:
    """Six significant digits: stable, readable JSON."""
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


def median(values: Iterable[float | None]) -> float | None:
    """Median of the values present, None without any."""
    present = [v for v in values if v is not None]
    return num(statistics.median(present)) if present else None


def code_kind(solver: str) -> tuple[str, str]:
    """``"gfem:bddc"`` -> ``("gfem", "bddc")``."""
    code, _, kind = solver.partition(":")
    return code, kind


def klass(solver: str) -> str:
    """``direct`` or ``iterative``."""
    code, kind = code_kind(solver)
    adapter = ADAPTERS.get(code)
    if adapter is not None and kind in adapter.kinds:
        return adapter.kinds[kind].klass
    return "direct" if kind in ("direct", "mumps", "petsc-mumps") else "iterative"


VERIFIED = ("placement", "layout")
"""Checks a passing run must have passed, not skipped: its layout was seen."""
UNVERIFIED = "unverified"


def unverified(r: records.SuiteRecord) -> bool:
    """An ``ok`` run without a failed check whose placement or layout went unseen."""
    return (
        r.status == "ok"
        and "fail" not in r.checks.values()
        and any(r.checks.get(c) != "pass" for c in VERIFIED)
    )


def passing(r: records.SuiteRecord) -> bool:
    """Status ``ok``, no failed check, placement and layout verified."""
    return (
        r.status == "ok"
        and "fail" not in r.checks.values()
        and all(r.checks.get(c) == "pass" for c in VERIFIED)
    )


def layout_of(r: records.SuiteRecord) -> str:
    """The resolved ``RxT``."""
    lay = r.run.layout
    return f"{lay.ranks}x{lay.threads}" if lay.ranks and lay.threads else lay.asked


def interpolate(points: Sequence[tuple[float, float]], e: float) -> float | None:
    """``y`` at error ``e`` on log-log axes between neighbours; None outside."""
    pts = sorted((x, y) for x, y in points if x > 0 and y > 0)
    if not pts or e < pts[0][0] * (1 - 1e-9) or e > pts[-1][0] * (1 + 1e-9):
        return None
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if x0 * (1 - 1e-9) <= e <= x1 * (1 + 1e-9):
            if x1 == x0:
                return num(y0)
            f = (math.log(e) - math.log(x0)) / (math.log(x1) - math.log(x0))
            f = min(max(f, 0.0), 1.0)
            return num(math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0))))
    return num(pts[0][1])


def overlap_points(lo: float, hi: float) -> list[float]:
    """Errors to compare at inside ``[lo, hi]``, never outside it.

    Both ends, the geometric midpoint and the powers of ten in between, ascending;
    empty when the range is empty.
    """
    if lo <= 0 or hi < lo:
        return []
    out: list[float] = []
    for e in sorted([lo, math.sqrt(lo * hi), hi, *decades(lo, hi)]):
        if not out or e > out[-1] * (1 + 1e-9):
            out.append(e)
    return out


def decades(lo: float, hi: float) -> list[float]:
    """Powers of ten in ``[lo, hi]``."""
    if lo <= 0 or hi < lo:
        return []
    first = math.ceil(math.log10(lo) - 1e-9)
    last = math.floor(math.log10(hi) + 1e-9)
    return [float(f"1e{k}") for k in range(first, last + 1)]


# --- cells --------------------------------------------------------------------------


def _items(rs: Sequence[records.SuiteRecord], get: Callable) -> list[float] | None:
    lists = [v for r in rs if (v := get(r))]
    if not lists:
        return None
    n = min(len(v) for v in lists)
    return [median(v[i] for v in lists) or 0.0 for i in range(n)]


def _iterations(r: records.SuiteRecord) -> float | None:
    its = r.outputs.get("iterations")
    if not isinstance(its, list) or not its:
        return None
    return float(sum(i for i in its if isinstance(i, (int, float))))


def cell(rs: Sequence[records.SuiteRecord], *, default: bool) -> dict[str, Any]:
    """The medians of one cell's passing records."""
    first = rs[0]
    code, kind = code_kind(first.run.solver)
    work = [r.timings.work_s for r in rs if r.timings.work_s is not None]
    mid = median(work)
    spread = num((max(work) - min(work)) / mid) if work and mid else None
    timings = {f: median(getattr(r.timings, f) for r in rs) for f in TIMING_FIELDS}
    rank0 = None
    if any(r.timings.rank0 for r in rs):
        rank0 = {p: median((r.timings.rank0 or {}).get(p) for r in rs) for p in PHASES}
    ranks = first.run.layout.ranks or 1
    imbalance = None
    if rank0 is not None and ranks > 1:
        top = sum(v for p in PHASES if (v := timings[p]) is not None)
        low = sum(v for v in rank0.values() if v is not None)
        imbalance = num(top / low) if low else None
    acc = [r.accuracy for r in rs if r.accuracy is not None]
    error = median(a.value for a in acc)
    bound = acc[0].bound if acc else None
    same = [r.same_code_error for r in rs if r.same_code_error]
    self_keys = sorted({k for r in rs for k in r.memory.self_reports})
    reasons = sorted(
        {str(r.extra.get("accuracy_reason")) for r in rs if not r.accuracy}
    )
    return {
        "case": first.run.case,
        "solver": first.run.solver,
        "code": code,
        "kind": kind,
        "class": klass(first.run.solver),
        "stack": first.run.stack,
        "layout": layout_of(first),
        "ranks": ranks,
        "default": default,
        "tolerance": first.run.tolerance,
        "order": first.case.order,
        "ndof": first.case.ndof,
        "n": len(rs),
        "names": [r.name for r in rs],
        "jobs": sorted({r.run.job for r in rs if r.run.job}),
        "version": first.provenance.solver.get("version"),
        "work_s": mid,
        "work_min_s": num(min(work)) if work else None,
        "work_max_s": num(max(work)) if work else None,
        "spread": spread,
        "noisy": spread is not None and spread > NOISY,
        "timings": timings,
        "rank_reduction": first.timings.rank_reduction,
        "rank0": rank0 if ranks > 1 else None,
        "imbalance": imbalance,
        "iterations": median(_iterations(r) for r in rs),
        "setups": median(r.structural.get("setups") for r in rs),
        "solves": median(r.structural.get("solves") for r in rs),
        "memory": {f: median(getattr(r.memory, f) for r in rs) for f in MEMORY_FIELDS},
        "self_reports": {
            k: median(r.memory.self_reports.get(k) for r in rs) for k in self_keys
        },
        "metric": acc[0].metric if acc else None,
        "error": error,
        "bound": num(bound),
        "resolved": None if error is None or bound is None else error >= 10 * bound,
        "error_items": _items(rs, lambda r: r.accuracy and r.accuracy.per_item),
        "accuracy_reason": "; ".join(reasons) if reasons and error is None else None,
        "plausible": _common(r.checks.get("accuracy_plausible") for r in rs),
        "same_code_error": median(s.get("value") for s in same),
        "same_code_label": same[0].get("label") if same else None,
        "same_code_against": same[0].get("against") if same else None,
        "reference": first.case.reference.id if first.case.reference else None,
    }


def _common(values: Iterable[str | None]) -> str | None:
    seen = sorted({v for v in values if v is not None})
    return "/".join(seen) if seen else None


def _cells(rows: Sequence[records.SuiteRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[records.SuiteRecord]] = {}
    for r in rows:
        if not passing(r):
            continue
        key = (
            r.run.case,
            r.run.core_budget,
            r.provenance.machine.node_class or "",
            r.run.solver,
            r.run.stack,
            r.run.layout.asked == "default",
            layout_of(r),
            r.run.tolerance,
        )
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups, key=_cell_order):
        rs = sorted(groups[key], key=lambda r: r.run.repeat)
        out.append(cell(rs, default=key[5]))
    return out


def _cell_order(key: tuple) -> tuple:
    case, budget, node_class, solver, stack, default, layout, tol = key
    ranks = int(layout.split("x")[0]) if "x" in layout else 0
    return (
        case,
        budget,
        node_class,
        solver,
        stack,
        not default,
        ranks,
        layout,
        -(tol or 0.0),
    )


# --- per case -----------------------------------------------------------------------


def _anchor(cells: Sequence[dict[str, Any]], anchor: float) -> float | None:
    tols = {c["tolerance"] for c in cells}
    if anchor in tols:
        return anchor
    present = [t for t in tols if t is not None]
    return min(present) if present else None


def _layouts(solver: str, cells: list[dict], unrun: list[dict]) -> dict[str, Any]:
    candidates = [c for c in cells if c["n"] >= REPEATS and c["work_s"] is not None]
    best = min(candidates, key=lambda c: (c["work_s"], c["layout"]), default=None)
    default = next((c for c in cells if c["default"]), None)
    entries = []
    for c in cells:
        labels = []
        if c is default:
            labels.append("default")
        if c is best:
            labels.append("best")
        elif (
            best is not None
            and c in candidates
            and c["work_min_s"] <= best["work_max_s"]
            and c["work_max_s"] >= best["work_min_s"]
        ):
            labels.append("tie")
        if c["n"] < REPEATS:
            labels.append(f"n = {c['n']}")
        entries.append(
            {
                "layout": c["layout"],
                "default": c["default"],
                "n": c["n"],
                "work_s": c["work_s"],
                "work_min_s": c["work_min_s"],
                "work_max_s": c["work_max_s"],
                "labels": labels,
            }
        )
    ratio = None
    if best is not None and default is not None and best["work_s"]:
        ratio = num(default["work_s"] / best["work_s"])
    return {
        "solver": solver,
        "tolerance": cells[0]["tolerance"] if cells else None,
        "cells": entries,
        "best": best["layout"] if best else None,
        "default": default["layout"] if default else None,
        "ties": [e["layout"] for e in entries if "tie" in e["labels"]],
        "default_over_best": ratio,
        "listed_only": all(c["default"] for c in cells),
        "unrun": unrun,
    }


def _unverified_reason(r: records.SuiteRecord) -> str:
    unseen = [c for c in VERIFIED if r.checks.get(c) != "pass"]
    return f"{' and '.join(unseen)} not verified"


def _unrun(rows: Sequence[records.SuiteRecord], solver: str, tol: float | None):
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.run.solver != solver or r.run.tolerance != tol or passing(r):
            continue
        lay = layout_of(r) + (" (default)" if r.run.layout.asked == "default" else "")
        entry = seen.setdefault(lay, {"layout": lay, "statuses": [], "reasons": []})
        entry["statuses"].append(UNVERIFIED if unverified(r) else r.status)
        reason = _unverified_reason(r) if unverified(r) else r.reason
        if reason and reason not in entry["reasons"]:
            entry["reasons"].append(reason)
    return [
        {
            "layout": e["layout"],
            "status": ", ".join(sorted(set(e["statuses"]))),
            "reason": "; ".join(e["reasons"]) or None,
        }
        for _, e in sorted(seen.items())
    ]


def _headline(cells: list[dict], best: str | None) -> dict[str, Any] | None:
    """The cell at the best layout, else the default one, else the first."""
    for c in cells:
        if c["layout"] == best and not c["default"]:
            return c
    for c in cells:
        if c["layout"] == best:
            return c
    return next((c for c in cells if c["default"]), cells[0] if cells else None)


def _ladder(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One cell per tolerance, all at one layout, tightest tolerance last.

    The best layout when its cells cover every tolerance of the solver, else the
    default layout (where the ladder runs), else the layout with most rungs.
    """
    groups: dict[tuple[str, bool], dict[Any, dict[str, Any]]] = {}
    for c in info["cells"]:
        groups.setdefault((c["layout"], c["default"]), {})[c["tolerance"]] = c
    tols = {c["tolerance"] for c in info["cells"]}
    best = info["layouts"]["best"]
    ranked = sorted(
        groups.items(),
        key=lambda kv: (
            set(kv[1]) != tols or kv[0][0] != best,
            not kv[0][1],
            -len(kv[1]),
            kv[0],
        ),
    )
    if not ranked:
        return []
    chosen = ranked[0][1]
    return [chosen[t] for t in sorted(chosen, key=lambda t: -(t or 0.0))]


def _point(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver": c["solver"],
        "code": c["code"],
        "class": c["class"],
        "layout": c["layout"],
        "tolerance": c["tolerance"],
        "work_s": c["timings"]["work_s"],
        "peak_rss_sum_gib": c["memory"]["peak_rss_sum_gib"],
    }


def _difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> float | None:
    """Max |a - b| over the S entries both outputs hold (not accuracy)."""
    fa, fb = a.get("frequencies_hz") or [], b.get("frequencies_hz") or []
    common = [f for f in fa if f in fb]
    worst = None
    for key, va in (a.get("s") or {}).items():
        vb = (b.get("s") or {}).get(key)
        if vb is None:
            continue
        for f in common:
            za, zb = complex(*va[fa.index(f)]), complex(*vb[fb.index(f)])
            worst = max(worst or 0.0, abs(za - zb))
    if worst is None and a.get("f_hz") and b.get("f_hz"):
        ra = sorted(m[0] for m in a["f_hz"])
        rb = sorted(m[0] for m in b["f_hz"])
        worst = max(abs(x - y) / abs(y) for x, y in zip(ra, rb, strict=False) if y)
    return num(worst)


def _differences(
    heads: Sequence[dict[str, Any]],
    rows: Sequence[records.SuiteRecord],
    anchor: records.SuiteRecord,
) -> list[dict[str, Any]]:
    out = []
    for h in heads:
        mine = [r for r in rows if r.name in h["names"]]
        d = median(_difference(r.outputs, anchor.outputs) for r in mine)
        if d is not None:
            out.append({**_point(h), "difference": d})
    return out


def _floors(
    reference: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], float | None]:
    r"""Same-mesh error per order with the reference bound, and the unresolved band.

    A floor is known only as well as the reference: it carries the reference's
    (largest) bound :math:`b` and is resolved at :math:`f \ge 10b`, like a run's
    error. The band is :math:`10b`.
    """
    if reference is None:
        return {}, None
    bounds = [q.get("bound") for q in (reference.get("quantities") or {}).values()]
    bounds = [b for b in bounds if isinstance(b, (int, float))]
    bound = max(bounds) if bounds else None
    floors = {}
    for k, v in sorted((reference.get("same_mesh_error") or {}).items()):
        f = num(v)
        floors[k] = {
            "error": f,
            "bound": num(bound),
            "resolved": None if f is None or bound is None else f >= 10 * bound,
        }
    return floors, (num(10 * bound) if bound is not None else None)


def _at_floor(
    runs: Sequence[Mapping[str, Any]], floors: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    r"""Per order: how many scored runs sit within 1 % of the order's floor.

    :math:`|e - f_p| \le 0.01 f_p`, the smallest deviation three printed digits
    show. It does not involve the reference bound (ADR 0457).
    """
    out: dict[str, dict[str, Any]] = {}
    for order, floor in floors.items():
        f = floor["error"]
        if not f:
            continue
        mine = [r for r in runs if r["error"] is not None and str(r["order"]) == order]
        if not mine:
            continue
        devs = [abs(r["error"] - f) / f for r in mine]
        out[order] = {
            "n": len(mine),
            "at_floor": sum(abs(r["error"] - f) <= AT_FLOOR * f for r in mine),
            "largest_deviation": num(max(devs)),
            "off": [
                {
                    "solver": r["solver"],
                    "layout": r["layout"],
                    "tolerance": r["tolerance"],
                    "error": r["error"],
                    "deviation": num(d),
                }
                for r, d in zip(mine, devs, strict=True)
                if abs(r["error"] - f) > AT_FLOOR * f
            ],
        }
    return out


def _error_cost(
    per_solver: Mapping[str, dict[str, Any]],
    rows: Sequence[records.SuiteRecord],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    points_a, ladders, directs, differences, scored = [], [], [], [], []
    has_reference = reference is not None and reference.get("method") != "none"
    gfem_direct = [
        r for r in rows if passing(r) and r.run.solver == "gfem:direct" and r.outputs
    ]
    for solver, info in per_solver.items():
        heads = _ladder(info)
        for h in heads:
            if h["error"] is None:
                continue
            p = _point(h)
            p.update(
                error=h["error"],
                bound=h["bound"],
                unresolved=h["resolved"] is False,
                plotted=h["error"],
            )
            points_a.append(p)
            scored.append(h)
        if info["class"] == "direct":
            if heads:
                directs.append(_point(heads[0]))
        else:
            rungs = []
            for h in heads:
                if h["same_code_error"] is None:
                    continue
                rungs.append({**_point(h), "same_code_error": h["same_code_error"]})
            if rungs:
                ladders.append({"solver": solver, "code": info["code"], "rungs": rungs})
        if not has_reference and gfem_direct and solver != "gfem:direct":
            differences += _differences(heads, rows, gfem_direct[0])
    floors, band = _floors(reference) if has_reference else ({}, None)
    return {
        "reference": has_reference,
        "points": points_a,
        "floors": floors,
        "at_floor": _at_floor(scored, floors),
        "unresolved_band": band,
        "ladders": ladders,
        "directs": directs,
        "differences": differences,
        "equal_error": _equal_error(ladders),
    }


def _equal_error(ladders: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(ladders) < 2:
        return None
    spans = []
    for lad in ladders:
        errs = [r["same_code_error"] for r in lad["rungs"] if r["same_code_error"] > 0]
        if not errs:
            return None
        spans.append((min(errs), max(errs)))
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    solvers = [lad["solver"] for lad in ladders]
    base = next((s for s in solvers if s.startswith("gfem:")), solvers[0])
    rows = []
    for e in overlap_points(lo, hi):
        values = {}
        for lad in ladders:
            pts = [(r["same_code_error"], r) for r in lad["rungs"]]
            values[lad["solver"]] = {
                "work_s": interpolate(
                    [(x, r["work_s"]) for x, r in pts if r["work_s"]], e
                ),
                "peak_rss_sum_gib": interpolate(
                    [
                        (x, r["peak_rss_sum_gib"])
                        for x, r in pts
                        if r["peak_rss_sum_gib"]
                    ],
                    e,
                ),
            }
        ratios = {}
        for s in solvers:
            if s == base:
                continue
            a, b = values[s]["work_s"], values[base]["work_s"]
            ratios[s] = num(a / b) if a is not None and b else None
        rows.append({"error": e, "values": values, "ratios": ratios})
    return {"solvers": solvers, "base": base, "rows": rows}


def _phase_rows(per_solver: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for info in per_solver.values():
        anchor = info["anchor_cells"]
        lay = info["layouts"]
        picked = [c for c in anchor if c["default"]]
        picked += [c for c in anchor if c["layout"] == lay["best"] and c not in picked]
        for c in picked:
            role = "default" if c["default"] else "best"
            if c["default"] and c["layout"] == lay["best"]:
                role = "default, best"
            out.append({"role": role, **c})
    return out


def _delta(reference: Mapping[str, Any] | None, version: str | None, code: str):
    if reference is None:
        return None
    for x in reference.get("cross_checks") or []:
        if x.get("code") == code and x.get("version") == version:
            return num(x.get("delta"))
    return None


def _accuracy(
    cells: Sequence[dict[str, Any]], reference: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    out = []
    ceiling = (reference or {}).get("ceiling") or {}
    for c in cells:
        delta = (
            _delta(reference, c["version"], c["code"]) if c["code"] != "gfem" else None
        )
        bound = c["bound"]
        out.append(
            {
                "solver": c["solver"],
                "code": c["code"],
                "kind": c["kind"],
                "order": c["order"],
                "layout": c["layout"],
                "default": c["default"],
                "tolerance": c["tolerance"],
                "metric": c["metric"],
                "error": c["error"],
                "bound": bound,
                "resolved": c["resolved"],
                "accuracy_reason": c["accuracy_reason"],
                "ceiling": num(ceiling.get(str(c["order"]))),
                "plausible": c["plausible"],
                "same_code_error": c["same_code_error"],
                "same_code_label": c["same_code_label"],
                "delta": delta,
                "delta_floor": delta is not None
                and bound is not None
                and delta > bound,
                "rescored": c.get("rescored"),
            }
        )
    return out


def _item_bounds(
    reference: Mapping[str, Any] | None, xs: Sequence[float], kind: str
) -> list[float | None]:
    """The reference bound at each item: per frequency where the reference has it."""
    quantities = (reference or {}).get("quantities") or {}
    scalars = [q.get("bound") for q in quantities.values()]
    scalars = [b for b in scalars if isinstance(b, (int, float))]
    fallback = num(max(scalars)) if scalars else None
    per: list[tuple[float, float]] = []
    if kind != "eigen":
        for q in quantities.values():
            per += zip(
                q.get("frequencies_hz") or [],
                q.get("bound_per_frequency") or [],
                strict=False,
            )
    out = []
    for x in xs:
        hits = [b for f, b in per if math.isclose(f, x, rel_tol=1e-9)]
        out.append(num(max(hits)) if hits else fallback)
    return out


def _error_items(
    per_solver: Mapping[str, dict[str, Any]],
    rows,
    kind: str,
    reference: Mapping[str, Any] | None = None,
):
    lines = []
    for solver, info in per_solver.items():
        h = _headline(info["anchor_cells"], info["layouts"]["best"])
        if h is None or not h["error_items"]:
            continue
        if kind == "eigen":
            x = list(range(1, len(h["error_items"]) + 1))
        else:
            first = next(r for r in rows if r.name == h["names"][0])
            x = list(first.outputs.get("frequencies_hz") or [])[: len(h["error_items"])]
        lines.append(
            {
                "solver": solver,
                "code": info["code"],
                "class": info["class"],
                "layout": h["layout"],
                "x": x,
                "error": h["error_items"],
            }
        )
    out: dict[str, Any] = {
        "x": "mode" if kind == "eigen" else "frequency_hz",
        "lines": lines,
    }
    unresolved = any(
        c["resolved"] is False for info in per_solver.values() for c in info["cells"]
    )
    if lines and unresolved:
        out["bound"] = _item_bounds(reference, cast("list[float]", lines[0]["x"]), kind)
    return out


def _snapshot(per_solver: Mapping[str, dict[str, Any]], machines: Mapping[str, Any]):
    out = []
    for k in ("direct", "iterative"):
        entries: list[dict[str, Any]] = []
        for solver, info in per_solver.items():
            if info["class"] != k:
                continue
            lay = info["layouts"]
            for role, layout in (("default", lay["default"]), ("best", lay["best"])):
                c = next(
                    (
                        c
                        for c in info["anchor_cells"]
                        if c["layout"] == layout and c["default"] == (role == "default")
                    ),
                    None,
                )
                if c is None and role == "best":
                    c = next(
                        (c for c in info["anchor_cells"] if c["layout"] == layout), None
                    )
                if c is None:
                    continue
                entries.append(
                    {
                        "solver": solver,
                        "code": info["code"],
                        "role": role,
                        "layout": c["layout"],
                        "n": c["n"],
                        "work_s": c["timings"]["work_s"],
                        "peak_rss_sum_gib": c["memory"]["peak_rss_sum_gib"],
                        "error": c["error"],
                        "jobs": c["jobs"],
                        "slow_node": any(
                            machines.get(j, {}).get("slow_node") for j in c["jobs"]
                        ),
                    }
                )
        ratios = []
        if k == "direct":
            gfem = [e for e in entries if e["code"] == "gfem"]
            others = [e for e in entries if e["code"] != "gfem"]
            for role in ("default", "best"):
                g = next((e for e in gfem if e["role"] == role), None)
                for o in (e for e in others if e["role"] == role):
                    if g is None:
                        continue
                    same_job = len(g["jobs"]) == 1 and g["jobs"] == o["jobs"]
                    ratios.append(
                        {
                            "role": role,
                            "solver": o["solver"],
                            "against": g["solver"],
                            "work": _ratio(o["work_s"], g["work_s"]),
                            "memory": _ratio(
                                o["peak_rss_sum_gib"], g["peak_rss_sum_gib"]
                            ),
                            "pairing": "same job" if same_job else DIFFERENT_NODES,
                        }
                    )
        out.append({"class": k, "entries": entries, "ratios": ratios})
    return out


def _ratio(a: float | None, b: float | None) -> float | None:
    return num(a / b) if a is not None and b else None


def _structural(rows: Sequence[records.SuiteRecord]) -> dict[str, list[str]]:
    facts: dict[str, set[str]] = {}
    for r in rows:
        if not passing(r):
            continue
        for k, v in r.structural.items():
            if k in ("setups", "solves", "excitations"):
                continue
            facts.setdefault(r.run.solver, set()).add(f"{k}: {v}")
    return {s: sorted(v) for s, v in sorted(facts.items())}


def _comparison(case: str, campaigns, rows) -> dict[str, Any]:
    for c in campaigns:
        detail = (c.manifest.checks.get("dof_equal_detail") or {}).get(case)
        if detail:
            same = detail.get("result") == "pass"
            return {
                "label": SAME_MESH if same else DIFFERENT_DISCRETIZATIONS,
                "ndof": detail.get("ndof"),
            }
    ndof: dict[str, set[int]] = {}
    for r in rows:
        if passing(r) and r.case.ndof is not None:
            ndof.setdefault(code_kind(r.run.solver)[0], set()).add(r.case.ndof)
    counts = {k: sorted(v) for k, v in sorted(ndof.items())}
    if len(counts) < 2:
        return {"label": None, "ndof": counts}
    same = len({n for v in counts.values() for n in v}) == 1
    return {"label": SAME_MESH if same else DIFFERENT_DISCRETIZATIONS, "ndof": counts}


def _reference_block(case: str, rows, inputs: Inputs) -> dict[str, Any]:
    named = sorted({r.case.reference.id for r in rows if r.case.reference is not None})
    rid = named[-1] if named else None
    ref = inputs.references.get(rid) if rid else None
    block: dict[str, Any] = {"id": rid, "ids": named, "in_store": ref is not None}
    if ref is not None:
        doc = ref.doc
        bounds = {k: q.get("bound") for k, q in (doc.get("quantities") or {}).items()}
        block.update(
            computed_utc=doc.get("computed_utc"),
            method=doc.get("method"),
            bounds=bounds,
            sha256=ref.sha256,
            reason=doc.get("reason"),
        )
        named_sha = {
            r.case.reference.sha256 for r in rows if r.case.reference is not None
        }
        block["sha256_matches"] = named_sha == {ref.sha256}
    families = reference_families(case)
    newer = [
        (x.doc.get("computed_utc") or "", x_id)
        for x_id, x in inputs.references.items()
        if x.doc.get("family") in families
        and x_id not in named
        and (
            ref is None
            or (x.doc.get("computed_utc") or "") > (ref.doc.get("computed_utc") or "")
        )
    ]
    if newer:
        when, nid = max(newer)
        block["newer"] = {"id": nid, "computed_utc": when}
    return block


def _rescore(
    case_name: str, rid: str, rows: Sequence[records.SuiteRecord], inputs: Inputs
) -> dict[str, dict[str, Any]]:
    """Name -> rescored accuracy of the case's passing records (records untouched)."""
    try:
        case = case_mod.load_case(config.CASES_DIR / case_name)
    except case_mod.CaseError as error:
        raise ReportError(f"--rescore {case_name}: {error}") from error
    ref = inputs.references[rid]
    out = {}
    for r in rows:
        if not passing(r):
            continue
        copy = records.load_record(r.to_json())
        copy.checks.pop("reference_matches", None)
        score.score(copy, case, ref.doc, None)
        a = copy.accuracy
        out[r.name] = {
            "error": None if a is None else a.value,
            "bound": None if a is None else a.bound,
            "reason": copy.extra.get("accuracy_reason") if a is None else None,
        }
    return out


def _attach_rescore(cells: list[dict], scored: Mapping[str, dict], rid: str) -> None:
    for c in cells:
        mine = [scored[n] for n in c["names"] if n in scored]
        if not mine:
            continue
        err = median(m["error"] for m in mine)
        bound = next((m["bound"] for m in mine if m["bound"] is not None), None)
        reasons = sorted({m["reason"] for m in mine if m["reason"]})
        c["rescored"] = {
            "against": rid,
            "error": err,
            "bound": num(bound),
            "resolved": None if err is None or bound is None else err >= 10 * bound,
            "reason": "; ".join(reasons) or None,
        }


def _newest(campaigns: Iterable[LoadedCampaign]) -> list[LoadedCampaign]:
    """The campaigns at the latest gfem commit (per-case views; the trend has all)."""
    found = list(campaigns)
    key = max(
        (
            c.manifest.pins.gfem.get("commit_utc") or "",
            c.manifest.pins.gfem.get("commit"),
        )
        for c in found
    )
    return [
        c
        for c in found
        if (
            c.manifest.pins.gfem.get("commit_utc") or "",
            c.manifest.pins.gfem.get("commit"),
        )
        == key
    ]


def _case_order(campaigns: Sequence[LoadedCampaign], rows) -> list[str]:
    order: list[str] = []
    for c in campaigns:
        for job in c.manifest.jobs:
            for name in job.extra.get("cases") or []:
                if name not in order:
                    order.append(name)
    for name in sorted({r.run.case for r in rows}):
        if name not in order:
            order.append(name)
    return [n for n in order if any(r.run.case == n for r in rows)]


def _partition(rows: Sequence[records.SuiteRecord]) -> list[dict[str, Any]]:
    """The case's rows per (core budget, node class), smallest budget first.

    A record without a node class (never started, refused) takes its job's class,
    else its budget's only class.
    """
    by_job: dict[str, set[str]] = {}
    by_budget: dict[int, set[str]] = {}
    for r in rows:
        if nc := r.provenance.machine.node_class:
            by_job.setdefault(r.run.job or "", set()).add(nc)
            by_budget.setdefault(r.run.core_budget, set()).add(nc)

    def node_class(r: records.SuiteRecord) -> str | None:
        if nc := r.provenance.machine.node_class:
            return nc
        for known in (by_job.get(r.run.job or ""), by_budget.get(r.run.core_budget)):
            if known and len(known) == 1:
                return next(iter(known))
        return None

    groups: dict[tuple[int, str | None], list[records.SuiteRecord]] = {}
    for r in rows:
        groups.setdefault((r.run.core_budget, node_class(r)), []).append(r)
    out = []
    for (budget, nc), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        nodes = max((len(r.provenance.machine.nodes) for r in rs), default=0)
        out.append(
            {
                "core_budget": budget,
                "node_class": nc,
                "nodes": nodes or None,
                "rows": rs,
            }
        )
    return out


def _views(
    rows: Sequence[records.SuiteRecord],
    *,
    ref_doc: Mapping[str, Any] | None,
    anchor: float,
    machines: Mapping[str, Any],
    rescored: tuple[Mapping[str, dict[str, Any]], str] | None,
) -> dict[str, Any]:
    """Every per-case view of one (core budget, node class) of a case."""
    cells = _cells(rows)
    if rescored is not None:
        _attach_rescore(cells, *rescored)
    per_solver: dict[str, dict[str, Any]] = {}
    for solver in sorted({c["solver"] for c in cells}):
        mine = [c for c in cells if c["solver"] == solver]
        tol = _anchor(mine, anchor)
        at = [c for c in mine if c["tolerance"] == tol]
        per_solver[solver] = {
            "code": code_kind(solver)[0],
            "class": klass(solver),
            "cells": mine,
            "anchor_cells": at,
            "layouts": _layouts(solver, at, _unrun(rows, solver, tol)),
        }
    kind = "eigen" if any("f_hz" in r.outputs for r in rows if passing(r)) else "driven"
    error_cost = _error_cost(per_solver, rows, ref_doc)
    accuracy = _accuracy(cells, ref_doc)
    return {
        "cells": cells,
        "error_cost": error_cost,
        "phases": _phase_rows(per_solver),
        "accuracy": accuracy,
        "accuracy_floor": _at_floor(accuracy, error_cost["floors"]),
        "error_items": _error_items(per_solver, rows, kind, ref_doc),
        "layouts": [info["layouts"] for info in per_solver.values()],
        "snapshot": _snapshot(per_solver, machines),
    }


def _case(
    name: str,
    rows: Sequence[records.SuiteRecord],
    *,
    inputs: Inputs,
    anchor: float,
    machines: Mapping[str, Any],
    rescore: Mapping[str, str],
) -> dict[str, Any]:
    """One case; its views per (core budget, node class) when it has several."""
    reference = _reference_block(name, rows, inputs)
    ref_doc = inputs.references[reference["id"]].doc if reference["in_store"] else None
    rescored = None
    if name in rescore:
        rescored = (_rescore(name, rescore[name], rows, inputs), rescore[name])
    kind = "eigen" if any("f_hz" in r.outputs for r in rows if passing(r)) else "driven"
    parts = _partition(rows)
    blocks = [
        _views(
            p["rows"],
            ref_doc=ref_doc,
            anchor=anchor,
            machines=machines,
            rescored=rescored,
        )
        for p in parts
    ]
    codes = sorted({code_kind(c["solver"])[0] for b in blocks for c in b["cells"]})
    notes = []
    if ref_doc is not None and ref_doc.get("method") == "none":
        notes.append(
            f"{name}: no converged reference ({ref_doc.get('reason') or 'none'})"
        )
    comparison = _comparison(name, inputs.campaigns, rows)
    if comparison["label"] == DIFFERENT_DISCRETIZATIONS:
        notes.append(f"{name}: {DIFFERENT_DISCRETIZATIONS} (DOF counts differ)")
    rescore_cmd = None
    if "newer" in reference and name not in rescore:
        rescore_cmd = f"--rescore {name}={reference['newer']['id']}"
    out: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "reference": reference,
        "rescore_flag": rescore_cmd,
        "rescored_against": rescore.get(name),
        "comparison": comparison,
        "structural": _structural(rows),
        "pins": {
            code: list(ADAPTERS[code].pins if code in ADAPTERS else ())
            for code in codes
        },
        "notes": notes,
        "figures": {},
    }
    if len(blocks) == 1:
        out.update(blocks[0])
    else:
        out["budgets"] = [
            {
                "core_budget": p["core_budget"],
                "node_class": p["node_class"],
                "nodes": p["nodes"],
                **b,
                "figures": {},
            }
            for p, b in zip(parts, blocks, strict=True)
        ]
        scaling = _scaling(parts, blocks)
        if scaling is not None:
            out["scaling"] = scaling
    return out


def _scaling_cell(block: Mapping[str, Any], lay: Mapping[str, Any]):
    """The best-layout cell of one solver in one budget block, and its default."""
    at = [
        c
        for c in block["cells"]
        if c["solver"] == lay["solver"] and c["tolerance"] == lay["tolerance"]
    ]
    default = next((c for c in at if c["default"]), None)
    best = None
    if lay["best"] is not None:
        best = next(
            (c for c in at if c["layout"] == lay["best"] and not c["default"]), None
        ) or next((c for c in at if c["layout"] == lay["best"]), None)
    return best, default


def _scaling_point(part: Mapping[str, Any], block: Mapping[str, Any], lay) -> Any:
    best, default = _scaling_cell(block, lay)
    c = best or default
    if c is None or c["work_s"] is None:
        return None
    nodes = part["nodes"] or 1
    labels = [] if best is not None else ["default (no best)"]
    if c["n"] < REPEATS:
        labels.append(f"n = {c['n']}")
    if nodes > 1 and c["stack"] == "wheels" and c["code"] == "palace":
        labels.append(TCP)
    if nodes > 1:
        labels.append(f"largest node GiB: {FIRST_NODE}")
    return {
        "core_budget": part["core_budget"],
        "node_class": part["node_class"],
        "nodes": part["nodes"],
        "layout": c["layout"],
        "tolerance": c["tolerance"],
        "n": c["n"],
        "work_s": c["work_s"],
        "work_min_s": c["work_min_s"],
        "work_max_s": c["work_max_s"],
        "default_layout": default["layout"] if default else None,
        "default_work_s": default["work_s"] if default else None,
        "peak_rss_sum_gib": c["memory"]["peak_rss_sum_gib"],
        "peak_rss_largest_node_gib": (
            c["memory"]["peak_rss_largest_node_gib"] if nodes > 1 else None
        ),
        "error": c["error"],
        "bound": c["bound"],
        "resolved": c["resolved"],
        "accuracy_reason": c["accuracy_reason"],
        "jobs": c["jobs"],
        "labels": labels,
    }


def _efficiency(points: list[dict[str, Any]]) -> None:
    r"""Speedup :math:`S(c) = T(c_0)/T(c)` and efficiency :math:`c_0 T(c_0)/(c T(c))`.

    :math:`c_0` is the smallest budget of the point's node class; a point whose class
    differs from the previous point's restarts the line.
    """
    base: dict[str | None, dict[str, Any]] = {}
    prev: str | None = points[0]["node_class"] if points else None
    for p in points:
        nc = p["node_class"]
        first = base.setdefault(nc, p)
        p["baseline"] = first["core_budget"]
        p["speedup"] = num(first["work_s"] / p["work_s"]) if p["work_s"] else None
        p["efficiency"] = None
        if p["work_s"] and p["core_budget"]:
            c0_t0 = first["core_budget"] * first["work_s"]
            p["efficiency"] = num(c0_t0 / (p["core_budget"] * p["work_s"]))
        if nc != prev:
            p["restart"] = f"line restarts: node class {prev} -> {nc}"
        prev = nc


def _scaling(
    parts: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Per solver across (core budget, node class) and cross-code ratios per budget.

    None unless passing cells sit at two or more core budgets.
    """
    pairs = zip(parts, blocks, strict=True)
    budgets = {p["core_budget"] for p, b in pairs if b["cells"]}
    if len(budgets) < 2:
        return None
    lines: dict[str, list[dict[str, Any]]] = {}
    for part, block in zip(parts, blocks, strict=True):
        for lay in block["layouts"]:
            point = _scaling_point(part, block, lay)
            if point is not None:
                lines.setdefault(lay["solver"], []).append(point)
    solvers: list[dict[str, Any]] = []
    for solver, points in sorted(lines.items()):
        _efficiency(points)
        code = code_kind(solver)[0]
        solvers.append(
            {"solver": solver, "code": code, "class": klass(solver), "points": points}
        )
    ratios = []
    for part in parts:
        here = {
            s["solver"]: p
            for s in solvers
            for p in s["points"]
            if (p["core_budget"], p["node_class"])
            == (part["core_budget"], part["node_class"])
        }
        for other in sorted(here):
            o = here[other]
            if code_kind(other)[0] == "gfem":
                continue
            for against in sorted(here):
                g = here[against]
                if code_kind(against)[0] != "gfem" or klass(against) != klass(other):
                    continue
                same_job = len(g["jobs"]) == 1 and g["jobs"] == o["jobs"]
                ratios.append(
                    {
                        "core_budget": part["core_budget"],
                        "node_class": part["node_class"],
                        "class": klass(other),
                        "solver": other,
                        "against": against,
                        "work": _ratio(o["work_s"], g["work_s"]),
                        "memory": _ratio(o["peak_rss_sum_gib"], g["peak_rss_sum_gib"]),
                        "pairing": "same job" if same_job else DIFFERENT_NODES,
                        "caveat": NOMINAL if klass(other) == "iterative" else None,
                    }
                )
    return {"solvers": solvers, "ratios": ratios}


def budget_views(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The per-budget view blocks of a case: itself when it has a single budget."""
    return case.get("budgets") or [cast("dict[str, Any]", case)]


# --- campaign level -----------------------------------------------------------------


def _failing(r: records.SuiteRecord) -> list[str]:
    return sorted(k for k, v in r.checks.items() if v == "fail")


def _gates(campaigns: Sequence[LoadedCampaign]) -> dict[str, Any]:
    out = []
    failures = []
    for c in campaigns:
        checks = c.manifest.checks
        out.append(
            {
                "campaign": c.manifest.id,
                "complete": checks.get("complete"),
                "provenance": checks.get("provenance"),
                "dof_equal": checks.get("dof_equal"),
                "dof_equal_detail": checks.get("dof_equal_detail") or {},
                "harness_failures": checks.get("harness_failures") or [],
                "mixed": c.mixed,
            }
        )
        for r in sorted(c.rows, key=lambda r: r.name):
            if passing(r):
                continue
            failures.append(
                {
                    "campaign": c.manifest.id,
                    "name": r.name,
                    "status": r.status,
                    "failure_class": r.failure_class,
                    "reason": r.reason,
                    "failing": _failing(r),
                }
            )
    return {"campaigns": out, "failures": failures}


def _machines(campaigns: Sequence[LoadedCampaign]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in campaigns:
        for job, m in sorted(c.machines.items()):
            rows = m.get("rows") or []
            top = max(rows, key=lambda r: r.get("threads") or 0, default={})
            out.append(
                {
                    "campaign": c.manifest.id,
                    "job": job,
                    "host": m.get("host"),
                    "threads": top.get("threads"),
                    "dgemm_gflops": num((top.get("dgemm") or {}).get("gflops_median")),
                    "zgemm_gflops": num((top.get("zgemm") or {}).get("gflops_median")),
                    "triad_gbs": num((top.get("triad") or {}).get("gbs_median")),
                }
            )
    triads = [m["triad_gbs"] for m in out if m["triad_gbs"]]
    mid = statistics.median(triads) if triads else None
    for m in out:
        m["slow_node"] = bool(
            mid and m["triad_gbs"] and m["triad_gbs"] < (1 - SLOW_NODE) * mid
        )
    return out


def _series_key(r: records.SuiteRecord) -> dict[str, Any]:
    return {
        "definition": r.case.definition_sha256,
        "mesh": r.case.mesh_sha256,
        "stack": r.run.stack,
        "solver version": r.provenance.solver.get("version"),
        "layout": layout_of(r),
        "core budget": r.run.core_budget,
        "node class": r.provenance.machine.node_class,
        "machine": r.provenance.machine.name,
    }


def _trend(campaigns: Sequence[LoadedCampaign], anchor: float) -> dict[str, Any]:
    commits = {c.manifest.pins.gfem.get("commit") for c in campaigns}
    if len(commits) < 2:
        return {"note": TREND_NEEDS, "series": []}
    ordered = sorted(
        campaigns,
        key=lambda c: (c.manifest.pins.gfem.get("commit_utc") or "", c.manifest.id),
    )
    lines: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in ordered:
        groups: dict[tuple[str, str], list[records.SuiteRecord]] = {}
        for r in c.rows:
            if passing(r) and r.run.layout.asked == "default":
                groups.setdefault((r.run.case, r.run.solver), []).append(r)
        for key, group in sorted(groups.items()):
            tols = {r.run.tolerance for r in group}
            tol = (
                anchor
                if anchor in tols
                else min((t for t in tols if t is not None), default=None)
            )
            rs = [r for r in group if r.run.tolerance == tol]
            work = [r.timings.work_s for r in rs if r.timings.work_s is not None]
            commit = c.manifest.pins.gfem.get("commit") or ""
            lines.setdefault(key, []).append(
                {
                    "campaign": c.manifest.id,
                    "commit": commit[:7],
                    "date": c.manifest.id[:8],
                    "key": _series_key(rs[0]),
                    "n": len(rs),
                    "work_s": median(work),
                    "work_min_s": num(min(work)) if work else None,
                    "work_max_s": num(max(work)) if work else None,
                    "peak_rss_sum_gib": median(r.memory.peak_rss_sum_gib for r in rs),
                    "error": median(r.accuracy.value for r in rs if r.accuracy),
                }
            )
    series = []
    for (case, solver), points in sorted(lines.items()):
        for prev, p in itertools.pairwise(points):
            changed = [k for k in p["key"] if p["key"][k] != prev["key"][k]]
            if changed:
                p["restart"] = "series restarts: " + ", ".join(changed) + " changed"
        series.append(
            {
                "case": case,
                "solver": solver,
                "code": code_kind(solver)[0],
                "points": points,
            }
        )
    return {"note": None, "series": series}


def _stack_fingerprints(campaigns: Sequence[LoadedCampaign]) -> list[str]:
    notes = []
    for c in campaigns:
        prints: dict[str, set[str]] = {}
        for r in c.rows:
            fp = r.stack.get("fingerprint")
            if fp:
                prints.setdefault(r.run.stack, set()).add(str(fp))
        for stack, fps in sorted(prints.items()):
            if len(fps) > 1:
                notes.append(
                    f"{c.manifest.id}: stack {stack} has {len(fps)} fingerprints "
                    f"({', '.join(sorted(f[:8] for f in fps))})"
                )
    return notes


def _header(inputs: Inputs) -> dict[str, Any]:
    out = []
    for c in inputs.campaigns:
        m = c.manifest
        rows = [r for r in c.rows if r.status != "not_started"]
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "gfem_commit": m.pins.gfem.get("commit"),
                "gfem_commit_utc": m.pins.gfem.get("commit_utc"),
                "harness_commit": m.pins.harness.get("commit"),
                "solvers": dict(sorted(m.pins.solvers.items())),
                "stacks": {
                    k: dict(sorted(v.items())) for k, v in sorted(m.pins.stacks.items())
                },
                "machine": m.machine,
                "node_classes": sorted(
                    {n for r in rows if (n := r.provenance.machine.node_class)}
                ),
                "core_budget": m.core_budget,
                "billed_core_h": num(sum(j.billed_core_h or 0.0 for j in m.jobs)),
                "jobs": [
                    {
                        "job": j.job,
                        "slurm_ids": j.extra.get("slurm_ids") or [],
                        "nodes": j.nodes,
                        "billed_core_h": j.billed_core_h,
                        "state": j.state,
                    }
                    for j in m.jobs
                ],
            }
        )
    return {"campaigns": out}


def command(ids: Sequence[str], flags: Mapping[str, Any]) -> str:
    """The regenerate command, every flag that shaped the report."""
    parts = ["python -m benchmarks report", *ids]
    if flags.get("allow_mixed"):
        parts.append("--allow-mixed")
    for case, rid in sorted((flags.get("rescore") or {}).items()):
        parts.append(f"--rescore {case}={rid}")
    if flags.get("no_figures"):
        parts.append("--no-figures")
    parts.append("--out DIR")
    return " ".join(parts)


def build(
    inputs: Inputs,
    *,
    flags: Mapping[str, Any],
    inputs_root: str,
    matplotlib: str | None,
    harness_revision: str | None = None,
) -> dict[str, Any]:
    """The ``suite-report/1`` mapping of ``inputs``.

    ``harness_revision`` is the commit of the harness tree that renders it.
    """
    rescore = dict(flags.get("rescore") or {})
    campaigns = inputs.campaigns
    rows = [r for c in campaigns for r in c.rows]
    digest = inputs.digest()
    commits = {c.manifest.pins.gfem.get("commit") for c in campaigns}
    newest = max(c.manifest.id[:8] for c in campaigns)
    rid = f"{'trend-' if len(commits) > 1 else ''}{newest}-{digest[:7]}"
    anchors = {
        float(a)
        for c in campaigns
        if (a := c.manifest.extra.get("anchor_tolerance")) is not None
    }
    anchor = min(anchors) if anchors else DEFAULT_ANCHOR
    machines = _machines(campaigns)
    by_job = {m["job"]: m for m in machines}
    stamps = []
    if any(c.mixed for c in campaigns):
        stamps.append("MIXED")
    if inputs.local_only:
        stamps.append("local, not comparable")
    cases = []
    for n in _case_order(campaigns, rows):
        newest = _newest(c for c in campaigns if any(r.run.case == n for r in c.rows))
        mine = [r for c in newest for r in c.rows if r.run.case == n]
        entry = _case(
            n, mine, inputs=inputs, anchor=anchor, machines=by_job, rescore=rescore
        )
        entry["campaigns"] = [c.manifest.id for c in newest]
        cases.append(entry)
    notes = _stack_fingerprints(campaigns)
    notes += [n for c in cases for n in c["notes"]]
    if stamps:
        notes.append("Campaign stamped: " + ", ".join(stamps))
    report = {
        "schema": SCHEMA,
        "id": rid,
        "inputs_sha256": digest,
        "inputs_root": inputs_root,
        "inputs": [c.rel for c in campaigns],
        "campaign_ids": [c.manifest.id for c in campaigns],
        "flags": {
            "allow_mixed": bool(flags.get("allow_mixed")),
            "rescore": dict(sorted(rescore.items())),
            "no_figures": bool(flags.get("no_figures")),
        },
        "command": command([c.manifest.id for c in campaigns], flags),
        "matplotlib": matplotlib,
        "harness_revision": harness_revision,
        "stamps": stamps,
        "anchor_tolerance": anchor,
        "header": _header(inputs),
        "gates": _gates(campaigns),
        "machines": machines,
        "cases": cases,
        "trend": _trend(campaigns, anchor),
        "caveats": list(CAVEATS),
        "notes": notes,
    }
    if not flags.get("no_figures"):
        assign_figures(report)
    return report


def _suffix(case: Mapping[str, Any], block: Mapping[str, Any]) -> str:
    """``-c<budget>`` on a multi-budget case; ``-<class>`` too if a budget has two."""
    if "budgets" not in case:
        return ""
    budget = block["core_budget"]
    twin = sum(b["core_budget"] == budget for b in case["budgets"]) > 1
    return f"-c{budget}" + (f"-{block['node_class']}" if twin else "")


def budget_title(case: Mapping[str, Any], block: Mapping[str, Any]) -> str:
    """``<case> at c<budget>`` (and the class when a budget has two); else the name."""
    if "budgets" not in case:
        return str(case["name"])
    title = f"{case['name']} at c{block['core_budget']}"
    if sum(b["core_budget"] == block["core_budget"] for b in case["budgets"]) > 1:
        title += f" {block['node_class']}"
    return title


def assign_figures(report: dict[str, Any]) -> None:
    """Name each figure the report draws (``fig-*.svg``)."""
    for c in report["cases"]:
        name = c["name"]
        for b in budget_views(c):
            tail = _suffix(c, b)
            figs = {"error_cost": f"fig-{name}-error-cost{tail}.svg"}
            if b["phases"]:
                figs["phases"] = f"fig-{name}-phases{tail}.svg"
                figs["memory"] = f"fig-{name}-memory{tail}.svg"
            if b["error_items"]["lines"]:
                figs["error_items"] = f"fig-{name}-error-items{tail}.svg"
            if any(len(lay["cells"]) > 1 for lay in b["layouts"]):
                figs["layouts"] = f"fig-{name}-layouts{tail}.svg"
            b["figures"] = figs
        if "scaling" in c:
            c["figures"]["scaling"] = f"fig-{name}-scaling.svg"
        if any(s["case"] == name for s in report["trend"]["series"]):
            c["figures"]["trend"] = f"fig-{name}-trend.svg"
