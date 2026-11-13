"""Write the whitepaper's data files from the docs' own sources.

``just whitepaper`` runs it before compiling the paper. It writes, under
``whitepaper/data``:

- ``front.json``: the 3D bend's history, parameters and results
  (from ``docs/_data/front-page.json``, built by ``tools/front_page_data.py``);
- ``snapshot-<iteration>.png``: the four design snapshots of the front page,
  the guides plus the design square's density in ink at alpha 0.88 v, as
  ``docs/_static/front/front.js`` draws them;
- ``caps.json``: the gradient matrix and the raises list, derived from the
  verification tests by ``docs/_ext/gfem_capabilities.py``;
- ``scaling.json``: the weak-scaling table and the facts the paper states about
  the cluster runs, read from the tables ``tools/scale_tables.py`` generates in
  ``docs/project/_scaling``.

Nothing here is measured: every number comes from a docs data file or a test.
Each scaling sentence of the paper is checked against those tables here, and the
script stops with `ScalingClaimError` when a table no longer supports it, so a
new set of runs cannot leave a stale sentence in the paper.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "whitepaper" / "data"
INK = (38, 38, 38)  # --ink-field in docs/_static/gfem.css
SCALE = 8  # nearest-neighbour upscale: the snapshots stay pixelated


def snapshots(front: dict) -> list[dict]:
    """Render each snapshot as an RGBA PNG of the whole box, y up."""
    x0, x1, _, y1 = front["extent_um"]
    d, w = front["design_um"], front["guide_um"]
    snap = front["snapshots"]
    ny, nx = snap["shape"]
    pitch = 2 * d / nx
    n = round((x1 - x0) / pitch)
    xs = x0 + (np.arange(n) + 0.5) * pitch
    ys = y1 - (np.arange(n) + 0.5) * pitch  # rows from max y down
    X, Y = np.meshgrid(xs, ys)
    guides = ((-d > X) & (np.abs(Y) < w)) | ((-d > Y) & (np.abs(X) < w))
    i0 = round((y1 - d) / pitch)
    j0 = round((-d - x0) / pitch)
    out = []
    for it, t, b64 in zip(snap["iterations"], snap["t"], snap["density"], strict=True):
        rho = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(ny, nx) / 255
        alpha = np.where(guides, 0.88, 0.0)
        alpha[i0 : i0 + ny, j0 : j0 + nx] = 0.88 * rho
        rgba = np.zeros((n, n, 4), dtype=np.uint8)
        rgba[..., :3] = INK
        rgba[..., 3] = np.round(255 * alpha).astype(np.uint8)
        size = (n * SCALE, n * SCALE)
        img = Image.fromarray(rgba, "RGBA").resize(size, Image.NEAREST)
        name = f"snapshot-{it}.png"
        img.save(OUT / name)
        out.append({"iteration": it, "t": t, "file": name})
    return out


def capabilities() -> dict:
    """The gradient matrix and the raises list, as the docs build derives them."""
    sys.path.insert(0, str(REPO / "docs" / "_ext"))
    import gfem_capabilities

    caps = gfem_capabilities.derive(REPO)
    keep = ("columns", "rows", "counts", "provenance", "raises")
    return json.loads(json.dumps({k: caps[k] for k in keep}, default=str))


SCALING = REPO / "docs" / "project" / "_scaling"


class ScalingClaimError(RuntimeError):
    """A scaling table no longer supports a sentence of the paper."""


def _check(condition: bool, claim: str) -> None:
    if not condition:
        msg = f"the scaling tables no longer support: {claim}; edit whitepaper.typ"
        raise ScalingClaimError(msg)


def _cell(text: str) -> Any:
    """A table cell as a number when it is one: 3,061,188, 36, 53%, 1e-08; else text."""
    text = text.replace("`", "").strip()
    if text in {"-", ""}:
        return None
    plain = text.replace(",", "")
    if re.fullmatch(r"-?\d+", plain):
        return int(plain)
    if re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", plain):
        return float(plain)
    if re.fullmatch(r"\d+(\.\d+)?%", plain):
        return float(plain[:-1]) / 100
    return text


def read_table(name: str) -> tuple[list[dict], str]:
    """The first Markdown table of a generated snippet, and the commit it names."""
    lines = (SCALING / name).read_text().splitlines()
    commit = re.search(r"run records of commit (\w+)", lines[0])
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    head = [h.replace("`", "").strip() for h in rows[0].strip("|").split("|")]
    table = [
        dict(zip(head, map(_cell, row.strip("|").split("|")), strict=True))
        for row in rows[2:]
    ]
    return table, commit.group(1) if commit else ""


def _finished(row: dict) -> bool:
    return isinstance(row["gfem.solve median, s"], (int, float))


def _iterations(row: dict) -> list[int]:
    return [int(i) for i in str(row["Iterations, ports 1/2"]).split("/")]


def scaling() -> dict:
    """The weak-scaling table and the facts of the paper's Scale section."""
    weak, commit = read_table("weak.md")
    strong108, _ = read_table("strong-108m.md")
    strong23, _ = read_table("strong-23m.md")
    phases, _ = read_table("phases.md")
    agreement, _ = read_table("agreement.md")
    bench, _ = read_table("machine.md")
    solve = "gfem.solve median, s"

    def by(rows: list[dict], solver: str) -> list[dict]:
        return [r for r in rows if r["Solver"] == solver]

    # The weak-scaling table, one row per node count.
    table = []
    for b in by(weak, "bddc_gmres"):
        d = next(r for r in by(weak, "direct") if r["Nodes"] == b["Nodes"])
        table.append(
            {
                "nodes": b["Nodes"],
                "dof": b["DOF"],
                "bddc_s": b[solve],
                "iterations": _iterations(b),
                "direct_s": d[solve] if _finished(d) else None,
                "direct_gib": d["Largest node, GiB"],
                "direct_end": None if _finished(d) else d[solve],
            }
        )

    # Largest model: BDDC + GMRES on the largest mesh, and its fastest node count.
    big = [r for r in by(strong108, "bddc_gmres") if _finished(r)]
    fastest = min(big, key=lambda r: r[solve])
    nodes_big = sorted(r["Nodes"] for r in big)
    _check(
        all(r[solve] > fastest[solve] for r in big if r["Nodes"] > fastest["Nodes"]),
        "past the fastest node count, more nodes made BDDC + GMRES slower",
    )
    iters = [i for r in by(weak, "bddc_gmres") for i in _iterations(r)]
    dofs = [r["DOF"] for r in by(weak, "bddc_gmres")]

    # Which solver: the weak rungs split at one size; above it BDDC wins everywhere.
    pairs = [(r["dof"], r["direct_s"], r["bddc_s"]) for r in table]
    direct_wins = [dof for dof, d, b in pairs if d is not None and d < b]
    bddc_wins = [dof for dof, d, b in pairs if d is None or b < d]
    _check(
        bool(direct_wins) and bool(bddc_wins) and max(direct_wins) < min(bddc_wins),
        "the direct solver is faster up to one size and BDDC + GMRES above it",
    )
    direct23 = {r["Nodes"]: r for r in by(strong23, "direct") if _finished(r)}
    both = [
        (b, direct23[b["Nodes"]])
        for b in by(strong23, "bddc_gmres")
        if _finished(b) and b["Nodes"] in direct23
    ]
    _check(
        all(b[solve] < d[solve] for b, d in both),
        "at the crossover size, BDDC + GMRES is faster at every node count tried",
    )
    one_b, one_d = min(both, key=lambda pair: pair[0]["Nodes"])
    _check(
        one_b["Largest node, GiB"] < one_d["Largest node, GiB"],
        "BDDC + GMRES needs less memory than the direct solver",
    )
    finished_direct = [r for r in by(weak + strong23, "direct") if _finished(r)]
    direct_big = max(finished_direct, key=lambda r: r["DOF"])
    direct_fail = [r for r in by(weak + strong108, "direct") if not _finished(r)]
    _check(
        all(r["DOF"] > direct_big["DOF"] for r in direct_fail),
        "the direct solver failed only above its largest finished model",
    )

    # Where the time goes at the largest model.
    mesh_big = big[0]["Mesh"]
    ph = sorted((r for r in phases if r["Mesh"] == mesh_big), key=lambda r: r["Nodes"])
    share = [r["Coarse apply in gfem.solve"] for r in ph]
    setup = {r["Nodes"]: r["Coarse setup, s"] for r in ph}
    _check(
        setup[max(setup)] > setup[fastest["Nodes"]],
        "the coarse setup grows from the fastest node count to the largest",
    )
    before_solve = [r["Mesh read, s"] + r["Partition, s"] for r in ph]

    # Agreement: every rank count against one process, and the two solvers.
    ranks = [
        r
        for r in agreement
        if str(r["Reference: solver, mesh, ranks x threads, tol"]).split(" on ")[0]
        == r["Solver"]
        and r["Gate"] != "observed, no gate"
    ]
    _check(
        all(r["S bitwise on all ranks"] == "yes" for r in agreement),
        "every rank held the same S, bit for bit",
    )
    _check(
        all(r["Difference in S"] <= r["Gate"] for r in ranks),
        "each rank count agreed with one process within its gate",
    )
    _check(
        all(r["Iterations"] == "same" for r in ranks if r["Solver"] == "bddc_gmres"),
        "BDDC + GMRES took the same iterations on every rank count",
    )
    cross = [
        r
        for r in agreement
        if r["Solver"] == "bddc_gmres"
        and str(r["Reference: solver, mesh, ranks x threads, tol"]).startswith("direct")
    ]
    _check(
        len(cross) == 1 and cross[0]["Difference in S"] <= cross[0]["Gate"],
        "the two solvers agreed within the gate",
    )

    cores = max(r["Threads"] for r in bench if isinstance(r.get("Threads"), int))
    date = subprocess.run(
        ["git", "show", "-s", "--format=%cs", commit],  # noqa: S607
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "commit": commit,
        "commit_date": date,
        "cores_per_node": cores,
        "layout": weak[0]["Ranks x threads"],
        "tol": weak[0]["tol"],
        "weak": table,
        "largest": {
            "dof": fastest["DOF"],
            "nodes": [nodes_big[0], nodes_big[-1]],
            "fastest_s": fastest[solve],
            "fastest_nodes": fastest["Nodes"],
            "iterations": [min(iters), max(iters)],
            "size_ratio": max(dofs) / min(dofs),
        },
        "solvers": {
            "direct_faster_upto": max(direct_wins),
            "bddc_faster_from": min(bddc_wins),
            "one_node": {
                "dof": one_b["DOF"],
                "bddc_s": one_b[solve],
                "direct_s": one_d[solve],
                "bddc_gib": one_b["Largest node, GiB"],
                "direct_gib": one_d["Largest node, GiB"],
            },
            "direct_largest": {
                "dof": direct_big["DOF"],
                "nodes": direct_big["Nodes"],
                "s": direct_big[solve],
            },
        },
        "time": {
            "coarse_share": [min(share), max(share)],
            "coarse_setup": [setup[fastest["Nodes"]], setup[max(setup)]],
            "setup_nodes": [fastest["Nodes"], max(setup)],
            "before_solve_s": sorted(before_solve)[len(before_solve) // 2],
        },
        "agreement": {
            "dof": ranks[0]["DOF"],
            "nodes": [min(r["Nodes"] for r in ranks), max(r["Nodes"] for r in ranks)],
            "max_diff": max(r["Difference in S"] for r in ranks),
            "cross_dof": cross[0]["DOF"],
            "cross_nodes": cross[0]["Nodes"],
            "cross_diff": cross[0]["Difference in S"],
            "cross_gate": cross[0]["Gate"],
            "cross_tol": cross[0]["tol"],
        },
    }


def main() -> None:
    """Write every data file the whitepaper reads."""
    OUT.mkdir(parents=True, exist_ok=True)
    front = json.loads((REPO / "docs/_data/front-page.json").read_text())
    data = {
        "source": front["source"],
        "history": front["history"],
        "params": front["params"],
        "results": front["results"],
        "extent_um": front["extent_um"],
        "design_um": front["design_um"],
        "snapshots": snapshots(front),
    }
    (OUT / "front.json").write_text(json.dumps(data, indent=1))
    (OUT / "caps.json").write_text(json.dumps(capabilities(), indent=1))
    (OUT / "scaling.json").write_text(json.dumps(scaling(), indent=1))
    poster = REPO / "docs/_static/front/bend3d-optimization-light.png"
    shutil.copy(poster, OUT / "bend3d-final.png")
    for logo in ("gfem-lockup-light.svg", "gfem-mark-light.svg"):
        shutil.copy(REPO / "docs/_static/logo" / logo, OUT / logo)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
