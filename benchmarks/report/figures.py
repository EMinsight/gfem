"""Deterministic SVG figures of a campaign report.

Drawn from the ``report.json`` mapping only. matplotlib is imported here and nowhere
else in the suite, and only when a report is drawn (``uv run --group docs``). Output
is byte-stable for one matplotlib version: fixed ``svg.hashsalt``, no date or creator
metadata, text kept as text, explicit light surface.

Encoding: colour = code, marker = solver class (direct square, iterative circle),
open markers for unresolved errors, an ordinal blue ramp for phases with startup in
grey, log axes for time, memory and error, one y scale per panel. With at most four
series the series are labelled directly; with more, a legend sits below the axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

COLORS = {"gfem": "#2a78d6", "palace": "#eb6834"}
THIRD = "#1baf7a"
MARKERS = {"direct": "s", "iterative": "o"}
RAMP = ("#86b6ef", "#6da7ec", "#2a78d6", "#184f95")
STARTUP = "#9a9a96"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
INK = "#0b0b0b"
MUTED = "#5f5e5a"
RC: dict[str, Any] = {
    "svg.hashsalt": "suite-report",
    "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "path.simplify": False,
}
SEGMENTS = (
    ("setup total", ("mesh_read_s", "discretize_s", "assemble_s", "setup_s")),
    ("solve", ("solve_s",)),
    ("estimate", ("estimate_s",)),
    ("postpro + other", ("postpro_s", "rom_s", "other_s")),
)


def color(code: str) -> str:
    """The code's colour slot (a third code gets slot 3)."""
    return COLORS.get(code, THIRD)


def _axes_style(ax: Any) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label_or_legend(ax: Any, series: Sequence[tuple[str, float, float]]) -> None:
    """Direct labels for at most four series, else a legend below the axes."""
    if len(series) <= 4:
        for k, (text, x, y) in enumerate(series):
            ax.annotate(
                text,
                (x, y),
                xytext=(6, 4 - 10 * k),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            fontsize=8,
        )


def _plain(value: float, _pos: object) -> str:
    return f"{value:.3g}"


def _short(value: float, _pos: object) -> str:
    return f"{value:.3g}"


def _blank(_value: float, _pos: object) -> str:
    return ""


def _log(ax: Any, *axes: str) -> None:
    """Log scale with plain tick labels on the named axes (``"x"``, ``"y"``)."""
    for name in axes:
        axis = ax.xaxis if name == "x" else ax.yaxis
        (ax.set_xscale if name == "x" else ax.set_yscale)("log")
        axis.set_major_formatter(FuncFormatter(_plain))
        low, high = ax.get_xlim() if name == "x" else ax.get_ylim()
        narrow = low > 0 and high / low < 30
        axis.set_minor_formatter(FuncFormatter(_short if narrow else _blank))


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})


def _positive(values: Sequence[float | None]) -> bool:
    return any(v is not None and v > 0 for v in values)


def _accuracy_panel(
    ax: Any,
    ec: Mapping[str, Any],
    xkey: str,
    xlabel: str,
    *,
    row: int,
    band: float | None,
) -> None:
    _axes_style(ax)
    if not ec["reference"]:
        ax.text(
            0.5,
            0.5,
            "no converged reference",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        labels = []
        by_solver: dict[str, list[dict]] = {}
        for p in ec["points"]:
            by_solver.setdefault(p["solver"], []).append(p)
        for solver, found in by_solver.items():
            pts = [p for p in found if p[xkey]]
            if not pts:
                continue
            c = color(pts[0]["code"])
            ax.plot(
                [p[xkey] for p in pts],
                [p["plotted"] for p in pts],
                color=c,
                linewidth=1,
                alpha=0.6,
            )
            for p in pts:
                ax.plot(
                    [p[xkey]],
                    [p["plotted"]],
                    linestyle="none",
                    marker=MARKERS[p["class"]],
                    markersize=8,
                    color=c,
                    markerfacecolor=SURFACE if p["unresolved"] else c,
                    label=solver,
                )
            labels.append((solver, pts[-1][xkey], pts[-1]["plotted"]))
        for order, floor in ec["floors"].items():
            if floor:
                ax.axhline(floor, color=MUTED, linewidth=1, linestyle="--")
                ax.annotate(
                    f"p{order} floor",
                    (0.5, floor),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=MUTED,
                )
        if band:
            ax.axhspan(band / 100, band, color=GRID, alpha=0.6, linewidth=0)
            ax.annotate(
                "below reference resolution",
                (0.99, band),
                xycoords=("axes fraction", "data"),
                xytext=(0, -10),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=MUTED,
            )
        if labels:
            _log(ax, "x", "y")
        _label_or_legend(ax, labels)
        ax.set_ylabel("error against the reference")
    ax.set_xlabel(xlabel)
    ax.set_title("(a) accuracy" if row == 0 else "(a) accuracy, memory", loc="left")


def _solver_panel(
    ax: Any, ec: Mapping[str, Any], xkey: str, xlabel: str, row: int
) -> None:
    _axes_style(ax)
    labels = []
    for lad in ec["ladders"]:
        pts = [r for r in lad["rungs"] if r[xkey] and r["same_code_error"] > 0]
        if not pts:
            continue
        c = color(lad["code"])
        ax.plot(
            [r[xkey] for r in pts],
            [r["same_code_error"] for r in pts],
            color=c,
            marker="o",
            markersize=8,
            label=lad["solver"],
        )
        labels.append((lad["solver"], pts[-1][xkey], pts[-1]["same_code_error"]))
    for d in ec["directs"]:
        if d[xkey]:
            ax.axvline(d[xkey], color=color(d["code"]), linewidth=1, linestyle=":")
            ax.annotate(
                d["solver"],
                (d[xkey], 0.98),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color=INK,
            )
    for d in ec["differences"]:
        if d[xkey] and d["difference"]:
            ax.plot(
                [d[xkey]],
                [d["difference"]],
                linestyle="none",
                marker="D",
                markersize=7,
                color=color(d["code"]),
                markerfacecolor=SURFACE,
                label="difference from gfem direct (not accuracy)",
            )
    if labels or ec["differences"]:
        _log(ax, "x", "y")
    elif any(d[xkey] for d in ec["directs"]):
        _log(ax, "x")
    _label_or_legend(ax, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("same-code error (solver error)")
    ax.set_title(
        "(b) solver error" if row == 0 else "(b) solver error, memory", loc="left"
    )


def _error_cost(case: Mapping[str, Any], path: Path) -> None:
    ec = case["error_cost"]
    solver_panel = bool(ec["ladders"] or ec["differences"])
    fig = Figure(figsize=(10 if solver_panel else 5.4, 7.4), layout="constrained")
    axes = fig.subplots(2, 2 if solver_panel else 1, squeeze=False)
    low = min((p["plotted"] for p in ec["points"] if p["plotted"]), default=None)
    band = ec["unresolved_band"]
    if band and low and band < low / 100:
        band = None  # a closed form's band sits far below every error: left out
    for row, (xkey, xlabel) in enumerate(
        (
            ("work_s", "work time, median [s]"),
            ("peak_rss_sum_gib", "peak RSS sum, median [GiB]"),
        )
    ):
        _accuracy_panel(axes[row][0], ec, xkey, xlabel, row=row, band=band)
        if solver_panel:
            _solver_panel(axes[row][1], ec, xkey, xlabel, row)
    fig.suptitle(f"{case['name']}: error against cost", x=0.01, ha="left")
    _save(fig, path)


def _row_label(p: Mapping[str, Any]) -> str:
    return f"{p['solver']} {p['layout']} ({p['role']})"


def _phases(case: Mapping[str, Any], path: Path) -> None:
    rows = case["phases"]
    fig = Figure(figsize=(9, 0.5 * len(rows) + 1.6), layout="constrained")
    ax = fig.subplots()
    _axes_style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, p in zip(ys, rows, strict=True):
        t = p["timings"]
        start = t["startup_s"] or 0.0
        ax.barh(y, -start, color=STARTUP, height=0.6)
        left = 0.0
        for k, (_, keys) in enumerate(SEGMENTS):
            width = sum(t[x] or 0.0 for x in keys)
            if width > 0:
                ax.barh(
                    y,
                    width,
                    left=left,
                    color=RAMP[k],
                    height=0.6,
                    edgecolor=SURFACE,
                    linewidth=2,
                )
                left += width
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(ys, [_row_label(p) for p in rows])
    ax.set_xlabel("seconds, median per bucket (startup left of zero, not in work time)")
    handles = [Patch(color=STARTUP, label="startup (not in work time)")]
    handles += [Patch(color=RAMP[k], label=s[0]) for k, s in enumerate(SEGMENTS)]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    ax.set_title(f"{case['name']}: phases", loc="left")
    _save(fig, path)


def _memory(case: Mapping[str, Any], path: Path) -> None:
    rows = case["phases"]
    fig = Figure(figsize=(9, 0.45 * len(rows) + 1.2), layout="constrained")
    ax = fig.subplots()
    _axes_style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, p in zip(ys, rows, strict=True):
        v = p["memory"]["peak_rss_sum_gib"]
        if v:
            ax.barh(y, v, color=color(p["code"]), height=0.6)
            ax.annotate(
                f"{v:.3g}",
                (v, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.set_yticks(ys, [_row_label(p) for p in rows])
    ax.set_xlabel("peak RSS summed over ranks, median [GiB]")
    ax.set_title(f"{case['name']}: memory", loc="left")
    _save(fig, path)


def _error_items(case: Mapping[str, Any], path: Path) -> None:
    items = case["error_items"]
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    _axes_style(ax)
    labels = []
    scale = 1e-9 if items["x"] != "mode" else 1.0
    for ln in items["lines"]:
        xs = [x * scale for x in ln["x"]]
        ys = ln["error"]
        ax.plot(
            xs,
            ys,
            color=color(ln["code"]),
            marker=MARKERS[ln["class"]],
            markersize=8,
            label=ln["solver"],
        )
        labels.append((ln["solver"], xs[-1], ys[-1]))
    band = case["error_cost"]["unresolved_band"]
    low = min((y for ln in items["lines"] for y in ln["error"] if y > 0), default=None)
    if band and (low is None or band >= low / 100):
        ax.axhspan(band / 100, band, color=GRID, alpha=0.6, linewidth=0)
        ax.annotate(
            "below reference resolution",
            (0.99, band),
            xycoords=("axes fraction", "data"),
            xytext=(0, -10),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=MUTED,
        )
    if _positive([y for ln in items["lines"] for y in ln["error"]]):
        _log(ax, "y")
    _label_or_legend(ax, labels)
    ax.set_xlabel("mode" if items["x"] == "mode" else "frequency [GHz]")
    ax.set_ylabel("error against the reference")
    ax.set_title(
        f"{case['name']}: error per {'mode' if items['x'] == 'mode' else 'frequency'}",
        loc="left",
    )
    _save(fig, path)


def _ranks(layout: str) -> tuple[int, str]:
    head = layout.split("x", maxsplit=1)[0]
    return (int(head) if head.isdigit() else 0, layout)


def _layouts(case: Mapping[str, Any], path: Path) -> None:
    sets = [lay for lay in case["layouts"] if len(lay["cells"]) > 1]
    fig = Figure(figsize=(3.4 * len(sets) + 0.6, 3.6), layout="constrained")
    axes = fig.subplots(1, len(sets), squeeze=False)[0]
    for ax, lay in zip(axes, sets, strict=True):
        _axes_style(ax)
        code = lay["solver"].partition(":")[0]
        names = sorted({c["layout"] for c in lay["cells"]}, key=_ranks)
        for c in lay["cells"]:
            if c["work_s"] is None:
                continue
            x = names.index(c["layout"])
            ax.errorbar(
                [x],
                [c["work_s"]],
                yerr=[[c["work_s"] - c["work_min_s"]], [c["work_max_s"] - c["work_s"]]],
                color=color(code),
                marker="o",
                markersize=7,
                capsize=3,
                linewidth=1.5,
            )
            text = ", ".join(c["labels"])
            if text:
                ax.annotate(
                    text,
                    (x, c["work_s"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK,
                )
        ax.set_xticks(range(len(names)), names)
        _log(ax, "y")
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_xlabel("layout (ranks x threads)")
        tol = "direct" if lay["tolerance"] is None else f"tol {lay['tolerance']:g}"
        ax.set_title(f"{lay['solver']} ({tol})", loc="left", fontsize=9)
    axes[0].set_ylabel("work time, median, min-max [s]")
    _save(fig, path)


def _trend(report: Mapping[str, Any], name: str, path: Path) -> None:
    series = [s for s in report["trend"]["series"] if s["case"] == name]
    metrics = (
        ("work_s", "work time [s]"),
        ("peak_rss_sum_gib", "peak RSS sum [GiB]"),
        ("error", "error"),
    )
    fig = Figure(figsize=(10, 2.6 * len(series) + 0.4), layout="constrained")
    axes = fig.subplots(len(series), len(metrics), squeeze=False)
    for row, s in zip(axes, series, strict=True):
        ticks = [f"{p['commit']} {p['date']}" for p in s["points"]]
        for ax, (key, label) in zip(row, metrics, strict=True):
            _axes_style(ax)
            segment: list[tuple[int, float]] = []
            for i, p in enumerate(s["points"]):
                if p.get("restart") and segment:
                    ax.plot(
                        *zip(*segment, strict=True), color=color(s["code"]), marker="o"
                    )
                    segment = []
                    ax.axvline(i - 0.5, color=MUTED, linewidth=1, linestyle=":")
                    ax.annotate(
                        p["restart"],
                        (i - 0.5, 0.98),
                        xycoords=("data", "axes fraction"),
                        rotation=90,
                        va="top",
                        ha="right",
                        fontsize=7,
                        color=MUTED,
                    )
                if p[key] is not None:
                    segment.append((i, p[key]))
                if key == "work_s" and p["work_s"] is not None:
                    ax.plot(
                        [i, i],
                        [p["work_min_s"], p["work_max_s"]],
                        color=color(s["code"]),
                        linewidth=1,
                    )
            if segment:
                ax.plot(*zip(*segment, strict=True), color=color(s["code"]), marker="o")
            if _positive([p[key] for p in s["points"]]):
                _log(ax, "y")
            ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
            ax.set_xlim(-0.5, len(ticks) - 0.5)
            ax.set_ylabel(label)
            ax.set_title(s["solver"], loc="left", fontsize=9)
    _save(fig, path)


def render(report: Mapping[str, Any], out: Path) -> list[Path]:
    """Draw every figure the report names into ``out``; return the paths."""
    written = []
    with mpl.rc_context(cast("Any", RC)):
        for case in report["cases"]:
            figs = case["figures"]
            draw = {
                "error_cost": lambda p, c=case: _error_cost(c, p),
                "phases": lambda p, c=case: _phases(c, p),
                "memory": lambda p, c=case: _memory(c, p),
                "error_items": lambda p, c=case: _error_items(c, p),
                "layouts": lambda p, c=case: _layouts(c, p),
                "trend": lambda p, c=case: _trend(report, c["name"], p),
            }
            for view, name in sorted(figs.items()):
                path = out / name
                draw[view](path)
                written.append(path)
    return written
