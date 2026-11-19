"""``index.md`` of a campaign report, from the ``report.json`` mapping (ADR 0413).

Every figure is followed by the table of its numbers. The first line is an HTML
comment naming the inputs digest. Standard library only (ADR 0400).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DASH = "—"
BUCKET_LABELS = (
    ("mesh_read_s", "mesh read"),
    ("discretize_s", "discretize"),
    ("assemble_s", "assemble"),
    ("setup_s", "setup"),
    ("solve_s", "solve"),
    ("estimate_s", "estimate"),
    ("postpro_s", "postpro"),
    ("rom_s", "rom"),
    ("other_s", "other"),
)


def g(x: float | None, digits: int = 3) -> str:
    """A number with ``digits`` significant digits, ``—`` for None."""
    return DASH if x is None else f"{x:.{digits}g}"


def e(x: float | None) -> str:
    """An error value."""
    return DASH if x is None else f"{x:.2e}"


def tol(x: float | None) -> str:
    """A tolerance, ``—`` for a direct run."""
    return DASH if x is None else f"{x:g}"


def pct(x: float | None) -> str:
    """A fraction as a percentage."""
    return DASH if x is None else f"{100 * x:.1f} %"


def table(head: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """A Markdown table."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*out, ""]


def _figure(case: Mapping[str, Any], view: str, alt: str) -> list[str]:
    name = case["figures"].get(view)
    return [f"![{alt}]({name})", ""] if name else []


def _header(r: Mapping[str, Any]) -> list[str]:
    title = f"# Campaign report {r['id']}"
    for stamp in r["stamps"]:
        title += f" ({stamp})"
    out = [
        f"<!-- {r['schema']} inputs sha256 {r['inputs_sha256']} -->",
        title,
        "",
        f"- Report id: `{r['id']}`",
        f"- Inputs: sha256 `{r['inputs_sha256']}` (`inputs.sha256`)",
    ]
    for c in r["header"]["campaigns"]:
        out.append(f"- Campaign `{c['id']}` ({c['name']})")
        out.append(
            f"  - gfem `{(c['gfem_commit'] or '')[:12]}` ({c['gfem_commit_utc']}), "
            f"harness `{(c['harness_commit'] or '')[:12]}`"
        )
        solvers = ", ".join(f"{k} {v}" for k, v in c["solvers"].items()) or DASH
        out.append(f"  - Solvers: {solvers}")
        for name, pins in c["stacks"].items():
            hashes = ", ".join(f"{k} `{str(v)[:12]}`" for k, v in pins.items())
            out.append(f"  - Stack {name}: {hashes}")
        classes = ", ".join(c["node_classes"]) or DASH
        out.append(
            f"  - Machine {c['machine']}, node classes {classes}, core budget "
            f"{c['core_budget']}, billed {g(c['billed_core_h'], 4)} core-h"
        )
    for case in r["cases"]:
        ref = case["reference"]
        if ref["id"] is None:
            out.append(f"- Reference {case['name']}: none named")
        elif not ref["in_store"]:
            out.append(f"- Reference {case['name']}: `{ref['id']}` (not in the store)")
        else:
            out.append(
                f"- Reference {case['name']}: `{ref['id']}` computed "
                f"{ref['computed_utc']}, {ref['method']}"
            )
    mpl = r["matplotlib"]
    out.append(f"- Figures: matplotlib {mpl}" if mpl else "- Figures: none")
    out.append(f"- Anchor tolerance: {tol(r['anchor_tolerance'])}")
    out.append(f"- Regenerate: `{r['command']}`")
    return [*out, ""]


def _nodes(r: Mapping[str, Any]) -> list[str]:
    if not r["machines"]:
        return []
    out = ["## Nodes", "", "Machine benchmark per job (`jobs/<job>/machine.json`)."]
    out.append("")
    out += table(
        [
            "campaign",
            "job",
            "host",
            "threads",
            "dgemm GF/s",
            "zgemm GF/s",
            "triad GB/s",
            "",
        ],
        (
            [
                m["campaign"],
                m["job"],
                m["host"] or DASH,
                m["threads"] or DASH,
                g(m["dgemm_gflops"]),
                g(m["zgemm_gflops"]),
                g(m["triad_gbs"]),
                "slow node" if m["slow_node"] else "",
            ]
            for m in r["machines"]
        ),
    )
    out.append("A job whose triad is more than 10 % below the median is a slow node.")
    return [*out, ""]


def _gates(r: Mapping[str, Any]) -> list[str]:
    out = ["## Gates", ""]
    out += table(
        ["campaign", "complete", "provenance", "dof_equal", "harness failures"],
        (
            [
                c["campaign"],
                c["complete"] or DASH,
                c["provenance"] or DASH,
                c["dof_equal"] or DASH,
                len(c["harness_failures"]),
            ]
            for c in r["gates"]["campaigns"]
        ),
    )
    for c in r["gates"]["campaigns"]:
        for case, d in sorted(c["dof_equal_detail"].items()):
            ndof = ", ".join(
                f"{k} {v}" for k, v in sorted((d.get("ndof") or {}).items())
            )
            out.append(
                f"- `{c['campaign']}` dof_equal {case}: {d.get('result')} ({ndof})"
            )
        for m in c["mixed"]:
            out.append(f"- MIXED `{c['campaign']}`: {m}")
    failures = r["gates"]["failures"]
    out += ["", f"### Failures ({len(failures)} records, excluded from every view)", ""]
    if failures:
        out += table(
            ["record", "status", "class", "reason", "failing checks"],
            (
                [
                    f["name"],
                    f["status"],
                    f["failure_class"] or DASH,
                    f["reason"] or DASH,
                    ", ".join(f["failing"]) or DASH,
                ]
                for f in failures
            ),
        )
    else:
        out += ["None.", ""]
    return out


def _error_display(a: Mapping[str, Any]) -> str:
    if a["error"] is None:
        return a["accuracy_reason"] or DASH
    if a["resolved"] is False:
        return f"≤ 10·bound = {e(10 * a['bound'])} (unresolved)"
    return e(a["error"])


def _error_cost(case: Mapping[str, Any]) -> list[str]:
    ec = case["error_cost"]
    out = ["### Error against cost", ""]
    out += _figure(case, "error_cost", f"{case['name']}: error against work time")
    if ec["reference"]:
        out.append(
            "(a) Error against the converged reference, at each solver's best layout."
        )
        out.append("")
        out += table(
            ["solver", "layout", "tol", "work s", "peak RSS sum GiB", "error"],
            (
                [
                    p["solver"],
                    p["layout"],
                    tol(p["tolerance"]),
                    g(p["work_s"]),
                    g(p["peak_rss_sum_gib"]),
                    f"≤ {e(p['plotted'])} (unresolved)"
                    if p["unresolved"]
                    else e(p["error"]),
                ]
                for p in ec["points"]
            ),
        )
        if ec["floors"]:
            floors = ", ".join(f"p{k} {e(v)}" for k, v in ec["floors"].items())
            out += [f"Discretization floor (same-mesh error): {floors}.", ""]
        if ec["unresolved_band"] is not None:
            out += [
                f"Below reference resolution: errors under {e(ec['unresolved_band'])}.",
                "",
            ]
    else:
        out += ["(a) No converged reference: no accuracy panel.", ""]
    out += [
        "(b) Same-code error (solver error, not accuracy) along the tolerance ladder.",
        "",
    ]
    rungs = [(lad["solver"], r) for lad in ec["ladders"] for r in lad["rungs"]]
    if not rungs:
        out += ["No tolerance ladder in this case.", ""]
    out += (
        []
        if not rungs
        else table(
            [
                "solver",
                "layout",
                "tol",
                "work s",
                "peak RSS sum GiB",
                "same-code error",
            ],
            (
                [
                    s,
                    r["layout"],
                    tol(r["tolerance"]),
                    g(r["work_s"]),
                    g(r["peak_rss_sum_gib"]),
                    e(r["same_code_error"]),
                ]
                for s, r in rungs
            ),
        )
    )
    if ec["directs"]:
        directs = ", ".join(
            f"{d['solver']} {g(d['work_s'])} s ({d['layout']})" for d in ec["directs"]
        )
        out += [f"Direct runs (same-code error 0 by definition): {directs}.", ""]
    if ec["differences"]:
        out += ["Difference from gfem direct (not accuracy):", ""]
        out += table(
            ["solver", "layout", "tol", "work s", "difference"],
            (
                [
                    d["solver"],
                    d["layout"],
                    tol(d["tolerance"]),
                    g(d["work_s"]),
                    e(d["difference"]),
                ]
                for d in ec["differences"]
            ),
        )
    eq = ec["equal_error"]
    if eq is not None:
        out += [
            (
                "Cost at equal achieved same-code error (interpolated log-log between "
                "neighbouring rungs, never extrapolated):"
            ),
            "",
        ]
        head = ["error"]
        for s in eq["solvers"]:
            head += [f"{s} work s", f"{s} GiB"]
        head += [f"{s} / {eq['base']}" for s in eq["solvers"] if s != eq["base"]]
        rows = []
        for row in eq["rows"]:
            cells = [e(row["error"])]
            for s in eq["solvers"]:
                v = row["values"][s]
                cells += [g(v["work_s"]), g(v["peak_rss_sum_gib"])]
            cells += [g(row["ratios"][s]) for s in eq["solvers"] if s != eq["base"]]
            rows.append(cells)
        out += table(head, rows) if rows else ["No overlap of the error ranges.", ""]
    return out


def _time(case: Mapping[str, Any]) -> list[str]:
    out = ["### Time", ""]
    out += _figure(case, "phases", f"{case['name']}: phases")
    head = [
        "solver",
        "layout",
        "role",
        "tol",
        "n",
        "work s",
        "spread",
        "startup s",
        "setup total s",
        "solve total s",
        "per sample s",
        *[label + " s" for _, label in BUCKET_LABELS],
        "iterations",
        "setups/solves",
        "imbalance",
    ]
    rows = []
    for p in case["phases"]:
        t = p["timings"]
        spread = pct(p["spread"]) + (" noisy" if p["noisy"] else "")
        rows.append(
            [
                p["solver"],
                p["layout"],
                p["role"],
                tol(p["tolerance"]),
                p["n"],
                g(t["work_s"]),
                spread,
                g(t["startup_s"]),
                g(t["setup_total_s"]),
                g(t["solve_total_s"]),
                g(t["per_sample_s"]),
                *[g(t[k]) for k, _ in BUCKET_LABELS],
                g(p["iterations"]),
                f"{g(p['setups'])}/{g(p['solves'])}",
                g(p["imbalance"]),
            ]
        )
        if p["rank0"] is not None:
            r0 = p["rank0"]
            rows.append(
                [
                    "",
                    "",
                    "rank 0",
                    "",
                    "",
                    DASH,
                    "",
                    "",
                    "",
                    "",
                    "",
                    *[g(r0[k]) for k, _ in BUCKET_LABELS],
                    "",
                    "",
                    "",
                ]
            )
    out += table(head, rows)
    reductions = sorted(
        {
            f"{p['solver']}: {p['rank_reduction']}"
            for p in case["phases"]
            if p["rank_reduction"]
        }
    )
    out.append(
        "Work time is wall minus startup; startup is not in work time. Rank "
        "reduction of the buckets: " + ("; ".join(reductions) or DASH) + ". "
        "Max-of-rows is an upper bound on the slowest rank (rows may peak on "
        "different ranks); imbalance = headline bucket sum / rank-0 sum."
    )
    if any("bddc_build" in f for facts in case["structural"].values() for f in facts):
        out.append("")
        out.append(
            "gfem BDDC: the assemble bucket includes the preconditioner build "
            "(NGSolve builds BDDC while it assembles the first matrix of a setup)."
        )
    return [*out, ""]


def _memory(case: Mapping[str, Any]) -> list[str]:
    out = ["### Memory", ""]
    out += _figure(case, "memory", f"{case['name']}: peak RSS sum")
    rows = []
    for p in case["phases"]:
        m = p["memory"]
        own = ", ".join(f"{k} {g(v)}" for k, v in p["self_reports"].items()) or DASH
        rows.append(
            [
                p["solver"],
                p["layout"],
                p["role"],
                g(m["peak_rss_sum_gib"]),
                g(m["peak_rss_rank_max_gib"]),
                g(m["peak_rss_largest_node_gib"]),
                g(m["pss_peak_sum_gib"]),
                g(m["sacct_max_rss_gib"]),
                own,
            ]
        )
    out += table(
        [
            "solver",
            "layout",
            "role",
            "peak RSS sum GiB",
            "rank max GiB",
            "largest node GiB",
            "PSS sum GiB",
            "sacct MaxRSS GiB",
            "self-reports",
        ],
        rows,
    )
    out.append(
        "The RSS sum counts shared pages once per rank in both codes alike; it is an "
        "upper bound on the simultaneous footprint, and PSS corrects it. sacct MaxRSS "
        "only for runs over 60 s."
    )
    return [*out, ""]


def _accuracy(case: Mapping[str, Any]) -> list[str]:
    out = ["### Accuracy", ""]
    ref = case["reference"]
    if ref["id"] is not None and ref["in_store"]:
        bounds = ", ".join(
            f"{k} {e(v)}" for k, v in sorted((ref.get("bounds") or {}).items())
        )
        out.append(
            f"Reference `{ref['id']}` ({ref['computed_utc']}, {ref['method']}, bound "
            f"{bounds or DASH}, sha256 `{ref['sha256'][:12]}`)."
        )
        if not ref.get("sha256_matches", True):
            out.append(
                "The archived file's sha256 differs from the one the records name."
            )
    if "newer" in ref:
        new = ref["newer"]
        out.append(
            f"Scored against `{ref['id']}`; a newer reference `{new['id']}` "
            f"({new['computed_utc']}) exists. Rescore with: "
            f"`python -m benchmarks report <ids> {case['rescore_flag']} --out DIR`."
            if case["rescore_flag"]
            else f"A newer reference `{new['id']}` exists."
        )
    if case["rescored_against"]:
        out.append(
            f"Rescored cells: vs `{case['rescored_against']}`; stored values beside."
        )
    out.append("")
    head = [
        "solver",
        "order",
        "layout",
        "tol",
        "metric",
        "error",
        "bound",
        "resolved",
        "ceiling",
        "plausible",
        "same-code error",
        "δ",
    ]
    if case["rescored_against"]:
        head.append("rescored")
    rows = []
    for a in case["accuracy"]:
        row = [
            a["solver"],
            a["order"],
            a["layout"] + (" (default)" if a["default"] else ""),
            tol(a["tolerance"]),
            a["metric"] or DASH,
            _error_display(a),
            e(a["bound"]),
            DASH if a["resolved"] is None else ("yes" if a["resolved"] else "no"),
            e(a["ceiling"]),
            a["plausible"] or DASH,
            e(a["same_code_error"]),
            e(a["delta"]) if a["delta"] is not None else DASH,
        ]
        if case["rescored_against"]:
            rs = a["rescored"]
            if rs is None:
                row.append(DASH)
            elif rs["error"] is None:
                row.append(f"{rs['reason']} (rescored vs {rs['against']})")
            else:
                shown = (
                    e(rs["error"])
                    if rs["resolved"] is not False
                    else (f"≤ {e(10 * rs['bound'])} (unresolved)")
                )
                row.append(f"{shown} (rescored vs {rs['against']})")
        rows.append(row)
    out += table(head, rows)
    if any(a["delta_floor"] for a in case["accuracy"]):
        out += [
            "Palace's error has a floor of δ, a definition difference, not accuracy.",
            "",
        ]
    items = case["error_items"]
    if items["lines"]:
        what = "mode" if items["x"] == "mode" else "frequency Hz"
        out += _figure(case, "error_items", f"{case['name']}: error per {what}")
        xs = items["lines"][0]["x"]
        out += table(
            [what, *[f"{ln['solver']} ({ln['layout']})" for ln in items["lines"]]],
            (
                [
                    g(x, 6),
                    *[
                        e(ln["error"][i]) if i < len(ln["error"]) else DASH
                        for ln in items["lines"]
                    ],
                ]
                for i, x in enumerate(xs)
            ),
        )
    return out


def _layouts(case: Mapping[str, Any]) -> list[str]:
    out = ["### Layouts", ""]
    out += _figure(case, "layouts", f"{case['name']}: work time per layout")
    rows = [
        [
            lay["solver"],
            tol(lay["tolerance"]),
            c["layout"],
            c["n"],
            g(c["work_s"]),
            f"{g(c['work_min_s'])}-{g(c['work_max_s'])}",
            ", ".join(c["labels"]),
        ]
        for lay in case["layouts"]
        for c in lay["cells"]
    ]
    out += table(["solver", "tol", "layout", "n", "median work s", "min-max", ""], rows)
    out += table(
        ["solver", "default", "best", "ties", "default / best"],
        (
            [
                lay["solver"],
                lay["default"] or DASH,
                (lay["best"] or DASH)
                + (" (best of listed)" if lay["listed_only"] else ""),
                ", ".join(lay["ties"]) or DASH,
                g(lay["default_over_best"]),
            ]
            for lay in case["layouts"]
        ),
    )
    unrun = [(lay["solver"], u) for lay in case["layouts"] for u in lay["unrun"]]
    if unrun:
        out += ["Not plotted (not passing):", ""]
        out += [
            f"- {s} {u['layout']}: {u['status']} ({u['reason'] or DASH})"
            for s, u in unrun
        ]
        out.append("")
    return out


def _snapshot(case: Mapping[str, Any]) -> list[str]:
    comp = case["comparison"]
    title = "### Cross-solver snapshot"
    if comp["label"]:
        title += f" ({comp['label']})"
    out = [title, ""]
    ndof = ", ".join(f"{k} {v}" for k, v in sorted((comp["ndof"] or {}).items()))
    if ndof:
        out += [f"DOF: {ndof}.", ""]
    for block in case["snapshot"]:
        if not block["entries"]:
            continue
        out.append(f"{block['class'].capitalize()} kinds:")
        out.append("")
        out += table(
            [
                "solver",
                "role",
                "layout",
                "n",
                "work s",
                "peak RSS sum GiB",
                "error",
                "jobs",
            ],
            (
                [
                    x["solver"],
                    x["role"],
                    x["layout"],
                    x["n"],
                    g(x["work_s"]),
                    g(x["peak_rss_sum_gib"]),
                    e(x["error"]),
                    ", ".join(x["jobs"]) + (" (slow node)" if x["slow_node"] else ""),
                ]
                for x in block["entries"]
            ),
        )
        if block["ratios"]:
            out += table(
                ["role", "ratio", "work", "memory", "pairing"],
                (
                    [
                        x["role"],
                        f"{x['solver']} / {x['against']}",
                        g(x["work"]),
                        g(x["memory"]),
                        x["pairing"],
                    ]
                    for x in block["ratios"]
                ),
            )
        if block["class"] == "iterative":
            out += [
                "Iterative kinds compare only at equal achieved error (table above).",
                "",
            ]
    out += ["Structural differences (reported, never removed):", ""]
    for solver, facts in case["structural"].items():
        for f in facts:
            out.append(f"- {solver}: {f}")
    for code, pins in case["pins"].items():
        out.append(f"- pinned for {code}: " + "; ".join(pins))
    return [*out, ""]


def _trend(r: Mapping[str, Any]) -> list[str]:
    t = r["trend"]
    out = ["## Trend", ""]
    if t["note"]:
        return [*out, t["note"] + ".", ""]
    figs = {c["name"]: c["figures"].get("trend") for c in r["cases"]}
    for name in sorted({s["case"] for s in t["series"]}):
        out += [f"### {name}", ""]
        if figs.get(name):
            out += [f"![{name}: trend]({figs[name]})", ""]
        rows = [
            [
                s["solver"],
                p["commit"],
                p["date"],
                p["n"],
                g(p["work_s"]),
                f"{g(p['work_min_s'])}-{g(p['work_max_s'])}",
                g(p["peak_rss_sum_gib"]),
                e(p["error"]),
                p.get("restart", ""),
            ]
            for s in t["series"]
            if s["case"] == name
            for p in s["points"]
        ]
        out += table(
            [
                "solver",
                "gfem",
                "date",
                "n",
                "work s",
                "min-max",
                "peak RSS sum GiB",
                "error",
                "",
            ],
            rows,
        )
    return out


def render(r: Mapping[str, Any]) -> str:
    """``index.md``."""
    out = _header(r) + _nodes(r) + _gates(r)
    for case in r["cases"]:
        out += [f"## {case['name']}", ""]
        out += ["Campaigns: " + ", ".join(f"`{x}`" for x in case["campaigns"]), ""]
        out += _error_cost(case) + _time(case) + _memory(case) + _accuracy(case)
        out += _layouts(case) + _snapshot(case)
    out += _trend(r)
    out += ["## Caveats", ""]
    out += [f"{i}. {c}" for i, c in enumerate(r["caveats"], 1)]
    n = len(r["caveats"]) + 1
    if r["notes"]:
        out.append(f"{n}. Notes of this report:")
        out += [f"   - {note}" for note in r["notes"]]
    else:
        out.append(f"{n}. No case-specific notes.")
    return "\n".join(out) + "\n"
