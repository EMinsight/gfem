"""`python -m benchmarks report`: a deterministic campaign report (ADR 0413).

The fixture campaign under ``tests/unit/data/bench/campaign`` holds synthetic records
(numbers invented, no real run's timing) and the store's ``references/`` beside it.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.report import views

BENCH = Path(__file__).parent / "data" / "bench"
FIXTURE = BENCH / "campaign"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


def _report(tmp_path: Path, *args: str, name: str = "r") -> Path:
    out = tmp_path / name
    assert main(["report", *args, "--out", str(out)]) == 0
    return out


def test_first_line_names_the_inputs_digest(tmp_path):
    out = _report(tmp_path, str(FIXTURE), "--no-figures")
    manifest = (out / "inputs.sha256").read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    first = (out / "index.md").read_text().splitlines()[0]
    assert first.startswith("<!--")
    assert digest in first
    assert json.loads((out / "report.json").read_text())["inputs_sha256"] == digest
    paths = [line.split("  ", 1)[1] for line in manifest.decode().splitlines()]
    assert paths == sorted(paths)
    assert "campaign/campaign.json" in paths
    assert "campaign/jobs/41000001/machine.json" in paths
    assert "references/microstrip-20261118-858bc66/reference.json" in paths
    assert sum(p.startswith("campaign/records/") for p in paths) == 45


def test_two_runs_are_byte_identical(tmp_path):
    a = _report(tmp_path, str(FIXTURE), "--no-figures", name="a")
    b = _report(tmp_path, str(FIXTURE), "--no-figures", name="b")
    names = sorted(p.name for p in a.iterdir())
    assert names == ["index.md", "inputs.sha256", "report.json"]
    assert names == sorted(p.name for p in b.iterdir())
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


# --- helpers ----------------------------------------------------------------------


def _store(tmp_path: Path, edit=None, *, rehash: bool = True, name: str = "campaign"):
    """A copy of the fixture store; ``edit(record dict)`` rewrites records."""
    root = tmp_path / "store"
    if not (root / "references").exists():
        shutil.copytree(BENCH / "references", root / "references")
    camp = root / name
    shutil.copytree(FIXTURE, camp)
    manifest = json.loads((camp / "campaign.json").read_text())
    for rel in manifest["records"]:
        p = camp / rel
        data = json.loads(p.read_text())
        if edit is not None:
            edit(data)
            p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        if rehash:
            manifest["records"][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (camp / "campaign.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return camp


def _json(tmp_path: Path, *campaigns: Path, extra=()) -> dict:
    out = _report(tmp_path, *map(str, campaigns), "--no-figures", *extra, name="j")
    return json.loads((out / "report.json").read_text())


def _case(report: dict, name: str) -> dict:
    return next(c for c in report["cases"] if c["name"] == name)


def _phase(case: dict, solver: str, layout: str) -> dict:
    return next(
        p for p in case["phases"] if p["solver"] == solver and p["layout"] == layout
    )


def _layouts(case: dict, solver: str) -> dict:
    return next(lay for lay in case["layouts"] if lay["solver"] == solver)


# --- views ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    assert main(["report", str(FIXTURE), "--no-figures", "--out", str(out)]) == 0
    return json.loads((out / "report.json").read_text()), (out / "index.md").read_text()


def test_medians_and_a_noisy_spread(fixture_report):
    ms = _case(fixture_report[0], "microstrip")
    direct = _phase(ms, "gfem:direct", "1x16")
    assert direct["n"] == 3
    assert direct["timings"]["work_s"] == 30.5
    assert direct["noisy"] is False
    mumps = _phase(ms, "palace:mumps", "16x1")
    assert mumps["timings"]["work_s"] == 41.0
    assert mumps["spread"] == pytest.approx(8 / 41, rel=1e-5)
    assert mumps["noisy"] is True
    assert "19.5 % noisy" in fixture_report[1]


def test_best_layout_ties_and_short_cells(fixture_report):
    bddc = _layouts(_case(fixture_report[0], "microstrip"), "gfem:bddc")
    assert bddc["default"] == "1x16"
    assert bddc["best"] == "4x4"
    assert bddc["ties"] == ["2x8"]
    short = next(c for c in bddc["cells"] if c["layout"] == "16x1")
    assert short["n"] == 2
    assert "n = 2" in short["labels"]
    assert "best" not in short["labels"]
    assert bddc["default_over_best"] == pytest.approx(12.2 / 11.0, rel=1e-5)
    pmg = _layouts(_case(fixture_report[0], "microstrip"), "palace:pmg")
    assert pmg["best"] == "4x4"
    assert pmg["unrun"] == [
        {
            "layout": "1x16",
            "status": "not_started, timeout",
            "reason": "wall cap 600 s; earlier_repeat_timeout",
        }
    ]


@pytest.mark.parametrize("check", ["placement", "layout"])
def test_an_unverified_run_is_left_out_of_the_best_layout(tmp_path, check):
    def edit(data):
        run = data["run"]
        if run["solver"] == "gfem:bddc" and run["layout"]["ranks"] == 4:
            data["checks"][check] = "skipped"

    report = _json(tmp_path, _store(tmp_path, edit))
    bddc = _layouts(_case(report, "microstrip"), "gfem:bddc")
    assert bddc["best"] == "2x8"
    assert "4x4" not in [c["layout"] for c in bddc["cells"]]
    assert {"layout": "4x4", "status": "unverified"}.items() <= next(
        u for u in bddc["unrun"] if u["layout"] == "4x4"
    ).items()


def test_equal_error_interpolates_and_never_extrapolates(fixture_report):
    eq = _case(fixture_report[0], "microstrip")["error_cost"]["equal_error"]
    assert eq["base"] == "gfem:bddc"
    # overlap [5e-9, 1e-5]: both ends, the geometric midpoint and the decades inside
    assert [r["error"] for r in eq["rows"]] == pytest.approx(
        [5e-9, 1e-8, 1e-7, (5e-9 * 1e-5) ** 0.5, 1e-6, 1e-5], rel=1e-5
    )
    at = {r["error"]: r for r in eq["rows"]}
    # gfem sits on a rung at 1e-7 (12.2 s); Palace interpolates between 4e-7 (22.2 s)
    # and 5e-9 (28.2 s): 22.2 (28.2 / 22.2) ** (ln 4 / ln 80) = 23.94 s.
    assert at[1e-7]["values"]["gfem:bddc"]["work_s"] == pytest.approx(12.2)
    assert at[1e-7]["values"]["palace:pmg"]["work_s"] == pytest.approx(23.94, rel=1e-3)
    assert at[1e-7]["ratios"]["palace:pmg"] == pytest.approx(23.94 / 12.2, rel=1e-3)
    assert 1e-9 not in at
    assert 1e-4 not in at


@pytest.mark.parametrize(
    ("lo", "hi", "points"),
    [
        (2e-7, 5e-7, [2e-7, 1e-6 * 0.1**0.5, 5e-7]),
        (3e-8, 3e-8, [3e-8]),
        (4e-7, 2e-7, []),
    ],
)
def test_equal_error_points_always_sit_inside_the_overlap(lo, hi, points):
    assert views.overlap_points(lo, hi) == pytest.approx(points, rel=1e-5)


def test_cross_job_ratios_are_labelled_different_nodes(fixture_report):
    ms = _case(fixture_report[0], "microstrip")
    direct = next(b for b in ms["snapshot"] if b["class"] == "direct")
    ratio = next(r for r in direct["ratios"] if r["role"] == "default")
    assert ratio["pairing"] == "different nodes"
    assert ratio["work"] == pytest.approx(41.0 / 30.5, rel=1e-5)
    palace = next(e for e in direct["entries"] if e["solver"] == "palace:mumps")
    assert palace["slow_node"] is True  # job 41000002's triad is 50 against 57


def test_unresolved_errors_and_delta_beside_palace(fixture_report):
    report, md = fixture_report
    acc = _case(report, "microstrip")["accuracy"]
    direct = next(a for a in acc if a["solver"] == "gfem:direct")
    assert direct["resolved"] is False
    assert "≤ 10·bound = 2.00e-06 (unresolved)" in md
    assert "1.50e-06" not in [
        line.split("|")[6].strip()
        for line in md.splitlines()
        if line.startswith("| gfem:direct | 2 |")
    ]
    mumps = next(a for a in acc if a["solver"] == "palace:mumps")
    assert mumps["delta"] == 5e-4
    assert mumps["delta_floor"] is True
    assert direct["delta"] is None
    assert "Palace's error has a floor of δ" in md


def test_imbalance_column_for_both_codes(fixture_report):
    ms = _case(fixture_report[0], "microstrip")
    gfem = _phase(ms, "gfem:bddc", "4x4")
    assert gfem["imbalance"] == pytest.approx(1 / 0.8)
    assert gfem["rank0"]["solve_s"] == pytest.approx(0.8 * 0.4 * 11.0)
    palace = _phase(ms, "palace:pmg", "16x1")
    assert palace["imbalance"] == pytest.approx(1 / 0.85, rel=1e-5)
    assert _phase(ms, "gfem:direct", "1x16")["imbalance"] is None


def test_bddc_assemble_includes_the_preconditioner_build(fixture_report):
    assert "assemble bucket includes the preconditioner build" in fixture_report[1]


def test_gates_list_every_excluded_record(fixture_report):
    failures = fixture_report[0]["gates"]["failures"]
    names = [f["name"] for f in failures]
    assert "microstrip/gfem-bddc__wheels__16x1__rep3" in names
    bad = failures[names.index("microstrip/gfem-bddc__wheels__16x1__rep3")]
    assert bad["failing"] == ["converged"]
    assert bad["failure_class"] == "solver"
    assert len(failures) == 4


def test_same_mesh_comparison_and_trend_needs_two(fixture_report):
    report, md = fixture_report
    assert _case(report, "lcavity")["comparison"]["label"] == "same-mesh comparison"
    assert report["trend"]["note"] == "trend needs two campaigns"
    assert report["id"].startswith("20261120-")
    assert len(report["caveats"]) == 9
    assert "## Caveats" in md


def test_eigen_case_errors_per_mode(fixture_report):
    lc = _case(fixture_report[0], "lcavity")
    items = lc["error_items"]
    assert items["x"] == "mode"
    gfem = next(ln for ln in items["lines"] if ln["solver"] == "gfem:direct")
    assert gfem["x"] == [1]
    assert gfem["error"] == [3.89e-4]
    assert lc["error_cost"]["ladders"] == []


# --- refusals ---------------------------------------------------------------------


def _refused(tmp_path, capsys, *args) -> str:
    assert (
        main(["report", *map(str, args), "--no-figures", "--out", str(tmp_path / "x")])
        == 1
    )
    return capsys.readouterr().err


def test_refuses_a_record_whose_sha256_differs(tmp_path, capsys):
    def edit(d):
        if d["name"] == "microstrip/gfem-direct__wheels__default__rep1":
            d["timings"]["work_s"] = 1.0

    camp = _store(tmp_path, edit, rehash=False)
    err = _refused(tmp_path, capsys, camp)
    assert "gfem-direct__wheels__default__rep1.json: sha256" in err


def test_refuses_an_unknown_major(tmp_path, capsys):
    def edit(d):
        if d["name"].startswith("lcavity/gfem-direct") and d["run"]["repeat"] == 1:
            d["schema"] = "suite-record/9"

    err = _refused(tmp_path, capsys, _store(tmp_path, edit))
    assert "suite-record/9 unknown" in err


def test_refuses_a_mixed_campaign_unless_allowed(tmp_path, capsys):
    def edit(d):
        if d["name"] == "lcavity/gfem-direct__wheels__default__rep2":
            d["provenance"]["gfem"]["commit"] = "deadbeef"

    camp = _store(tmp_path, edit)
    assert "is mixed (pass --allow-mixed" in _refused(tmp_path, capsys, camp)
    report = _json(tmp_path, camp, extra=("--allow-mixed",))
    assert report["stamps"] == ["MIXED"]
    assert "--allow-mixed" in report["command"]
    assert report["gates"]["campaigns"][0]["mixed"]


def test_refuses_local_records_beside_cluster_ones(tmp_path, capsys):
    def edit(d):
        if d["name"].startswith("lcavity/") and "provenance" in d:
            d["provenance"]["machine"]["name"] = "local"

    err = _refused(tmp_path, capsys, _store(tmp_path, edit))
    assert "run --local mixed with cluster records" in err


def test_a_local_only_report_says_not_comparable(tmp_path):
    def edit(d):
        if "provenance" in d:
            d["provenance"]["machine"]["name"] = "local"

    report = _json(tmp_path, _store(tmp_path, edit))
    assert report["stamps"] == ["local, not comparable"]


# --- check, rescore, trend ---------------------------------------------------------


def test_check_passes_on_its_output_and_fails_on_a_tampered_record(tmp_path, capsys):
    camp = _store(tmp_path)
    out = tmp_path / "store" / "reports" / "r"
    assert main(["report", str(camp), "--no-figures", "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["inputs_root"] == "../.."
    assert main(["report", "--check", str(out)]) == 0
    assert "unchanged" in capsys.readouterr().out
    record = camp / "records" / "lcavity" / "gfem-direct__wheels__default__rep1.json"
    record.write_text(record.read_text().replace("20.0", "2.0", 1))
    assert main(["report", "--check", str(out)]) == 1
    assert "sha256" in capsys.readouterr().err


def test_check_reports_drift_of_an_edited_report(tmp_path, capsys):
    out = _report(tmp_path, str(_store(tmp_path)), "--no-figures")
    (out / "index.md").write_text((out / "index.md").read_text() + "edited\n")
    assert main(["report", "--check", str(out)]) == 1
    assert "drift: index.md" in capsys.readouterr().out


def test_newer_reference_prints_the_rescore_command(fixture_report):
    report, md = fixture_report
    ms = _case(report, "microstrip")
    assert ms["reference"]["newer"]["id"] == "microstrip-20261125-aaaaaaa"
    assert "--rescore microstrip=microstrip-20261125-aaaaaaa" in md
    assert "a newer reference `microstrip-20261125-aaaaaaa`" in md


def test_newer_reference_follows_the_case_files_reference_key(tmp_path):
    # microstrip_p3 is scored by the microstrip reference ([reference] case key)
    def edit(d):
        if d["run"]["case"] == "microstrip":
            d["run"]["case"] = "microstrip_p3"

    report = _json(tmp_path, _store(tmp_path, edit))
    p3 = _case(report, "microstrip_p3")
    assert p3["reference"]["newer"]["id"] == "microstrip-20261125-aaaaaaa"
    assert p3["rescore_flag"] == "--rescore microstrip_p3=microstrip-20261125-aaaaaaa"


def test_rescore_marks_cells_and_rewrites_no_record(tmp_path):
    camp = _store(tmp_path)
    before = {p: p.read_bytes() for p in camp.rglob("*.json")}
    report = _json(
        tmp_path, camp, extra=("--rescore", "microstrip=microstrip-20261125-aaaaaaa")
    )
    assert {p: p.read_bytes() for p in camp.rglob("*.json")} == before
    ms = _case(report, "microstrip")
    assert ms["rescored_against"] == "microstrip-20261125-aaaaaaa"
    direct = next(a for a in ms["accuracy"] if a["solver"] == "gfem:direct")
    assert direct["error"] == 1.5e-6  # the stored value stays beside
    assert direct["rescored"]["against"] == "microstrip-20261125-aaaaaaa"
    # the synthetic records do not match the committed case: refused, with its reason
    assert "reference_mismatch" in direct["rescored"]["reason"]
    assert "microstrip-20261125-aaaaaaa" in report["inputs_sha256"] or any(
        "microstrip-20261125-aaaaaaa" in line
        for line in (tmp_path / "j" / "inputs.sha256").read_text().splitlines()
    )


def test_rescore_refuses_a_reference_not_in_the_store(tmp_path, capsys):
    camp = _store(tmp_path)
    err = _refused(tmp_path, capsys, camp, "--rescore", "microstrip=nope")
    assert "no references/nope/reference.json" in err


def test_accuracy_reason_and_no_reference(tmp_path):
    def edit(d):
        if d["name"].startswith("lcavity/") and d.get("accuracy"):
            d["accuracy"] = None
            d["accuracy_reason"] = "no_reference"
            d["case"]["reference"] = None

    report = _json(tmp_path, _store(tmp_path, edit))
    lc = _case(report, "lcavity")
    assert {a["accuracy_reason"] for a in lc["accuracy"]} == {"no_reference"}
    assert lc["error_cost"]["points"] == []


def test_different_discretizations(tmp_path):
    def edit(d):
        if d["name"].startswith("lcavity/palace") and "case" in d:
            d["case"]["ndof"] = 246000

    camp = _store(tmp_path, edit)
    manifest = json.loads((camp / "campaign.json").read_text())
    del manifest["checks"]["dof_equal_detail"]
    (camp / "campaign.json").write_text(json.dumps(manifest))
    report = _json(tmp_path, camp)
    lc = _case(report, "lcavity")
    assert lc["comparison"] == {
        "label": "different discretizations",
        "ndof": {"gfem": [246416], "palace": [246000]},
    }
    assert any("different discretizations" in n for n in report["notes"])


def test_trend_over_two_gfem_commits_restarts_on_a_key_change(tmp_path):
    first = _store(tmp_path)

    def later(d):
        d["campaign"] = "20261201-aaaaaaa-smoke"
        if "provenance" in d:
            d["provenance"]["gfem"]["commit"] = "a" * 40
            d["provenance"]["harness"]["commit"] = "a" * 40
            d["timings"]["work_s"] = (d["timings"]["work_s"] or 0) * 0.9
        if d["name"].startswith("lcavity/") and "case" in d:
            d["case"]["mesh_sha256"] = "mesh-lcavity-2"

    second = _store(tmp_path, later, name="campaign2")
    manifest = json.loads((second / "campaign.json").read_text())
    manifest["id"] = "20261201-aaaaaaa-smoke"
    manifest["pins"]["gfem"] = {
        "commit": "a" * 40,
        "commit_utc": "2026-11-30T00:00:00Z",
    }
    manifest["pins"]["harness"] = {"commit": "a" * 40}
    (second / "campaign.json").write_text(json.dumps(manifest))
    report = _json(tmp_path, second, first)
    assert report["id"].startswith("trend-20261201-")
    series = {(s["case"], s["solver"]): s["points"] for s in report["trend"]["series"]}
    direct = series[("microstrip", "gfem:direct")]
    assert [p["commit"] for p in direct] == ["858bc66", "aaaaaaa"]
    assert direct[1]["work_s"] == pytest.approx(0.9 * 30.5)
    assert "restart" not in direct[1]
    lc = series[("lcavity", "gfem:direct")]
    assert lc[1]["restart"] == "series restarts: mesh changed"
    ms = _case(report, "microstrip")
    assert ms["campaigns"] == ["20261201-aaaaaaa-smoke"]
    assert _phase(ms, "gfem:direct", "1x16")["n"] == 3


# --- core budgets (ADR 0452) ------------------------------------------------------


def _budget_campaign(tmp_path: Path, budget: int, edit=None) -> Path:
    """The fixture campaign rerun at ``budget`` cores, same gfem commit."""
    cid = f"20261120-858bc66-smoke-c{budget}"

    def at_budget(d):
        d["campaign"] = cid
        d["run"]["core_budget"] = budget
        if edit is not None:
            edit(d)

    camp = _store(tmp_path, at_budget, name=f"campaign-c{budget}")
    manifest = json.loads((camp / "campaign.json").read_text())
    manifest["id"] = cid
    manifest["core_budget"] = budget
    (camp / "campaign.json").write_text(json.dumps(manifest))
    return camp


def _two_budgets(tmp_path: Path) -> Path:
    """c8 and c16 of the fixture; at c8 gfem:bddc is fastest at 2x8 (halved)."""

    def faster_2x8(d):
        run = d["run"]
        if run["solver"] == "gfem:bddc" and run["layout"]["ranks"] == 2:
            d["timings"]["work_s"] = (d["timings"]["work_s"] or 0) * 0.5

    c8 = _budget_campaign(tmp_path, 8, faster_2x8)
    c16 = _budget_campaign(tmp_path, 16)
    return _report(tmp_path, str(c8), str(c16), "--no-figures", name="budgets")


def _budget(case: dict, budget: int) -> dict:
    return next(b for b in case["budgets"] if b["core_budget"] == budget)


def test_each_core_budget_has_its_own_best_layout(tmp_path):
    out = _two_budgets(tmp_path)
    ms = _case(json.loads((out / "report.json").read_text()), "microstrip")
    assert [(b["core_budget"], b["node_class"]) for b in ms["budgets"]] == [
        (8, "standard"),
        (16, "standard"),
    ]
    # c16 is the fixture: 4x4 best (11.0 s); c8 halves 2x8 below it
    assert _layouts(_budget(ms, 16), "gfem:bddc")["best"] == "4x4"
    assert _layouts(_budget(ms, 8), "gfem:bddc")["best"] == "2x8"
    assert _phase(_budget(ms, 8), "gfem:direct", "1x16")["n"] == 3
    assert "layouts" not in ms
    assert ms["reference"]["id"] is not None


def test_each_core_budget_gets_a_subsection(tmp_path):
    index = (_two_budgets(tmp_path) / "index.md").read_text()
    c8 = index.index("### microstrip at c8 (standard, 1 node)")
    c16 = index.index("### microstrip at c16 (standard, 1 node)")
    assert index.index("## microstrip") < c8 < c16
    section = index[c8:c16]
    assert "#### Error against cost" in section
    assert "#### Layouts" in section
    assert index.count("Structural differences (reported, never removed)") == 2


def test_figures_of_several_budgets_carry_the_budget(tmp_path):
    pytest.importorskip("matplotlib")
    c8 = _budget_campaign(tmp_path, 8)
    c16 = _budget_campaign(tmp_path, 16)
    out = _report(tmp_path, str(c8), str(c16), name="figs")
    svgs = sorted(p.name for p in out.glob("fig-microstrip-*.svg"))
    assert "fig-microstrip-error-cost-c8.svg" in svgs
    assert "fig-microstrip-layouts-c16.svg" in svgs
    assert "fig-microstrip-error-cost.svg" not in svgs
    index = (out / "index.md").read_text()
    for name in svgs:
        assert f"]({name})" in index


def _slower(factor: float):
    """Every work time of a campaign times ``factor``."""

    def edit(d):
        if d.get("timings", {}).get("work_s") is not None:
            d["timings"]["work_s"] *= factor

    return edit


def _on(node_class: str, *, factor: float = 1.0, nodes: int = 1):
    """Every record on ``node_class`` nodes (``nodes`` of them), work times scaled."""

    def edit(d):
        _slower(factor)(d)
        machine = d.get("provenance", {}).get("machine")
        if machine is not None:
            machine["node_class"] = node_class
            machine["nodes"] = [f"node{k:03d}" for k in range(1, nodes + 1)]

    return edit


def _point(case: dict, solver: str, budget: int) -> dict:
    line = next(s for s in case["scaling"]["solvers"] if s["solver"] == solver)
    return next(p for p in line["points"] if p["core_budget"] == budget)


def test_scaling_gives_speedup_and_efficiency_against_the_smallest_budget(tmp_path):
    c8 = _budget_campaign(tmp_path, 8, _slower(1.6))
    c16 = _budget_campaign(tmp_path, 16)
    ms = _case(_json(tmp_path, c8, c16), "microstrip")
    first, second = (_point(ms, "gfem:direct", b) for b in (8, 16))
    assert (first["speedup"], first["efficiency"]) == (1.0, 1.0)
    # T(c8) = 1.6 T(c16): S = 1.6, E = 8 * 1.6 / 16 = 0.8
    assert second["speedup"] == pytest.approx(1.6, rel=1e-5)
    assert second["efficiency"] == pytest.approx(0.8, rel=1e-5)
    assert second["work_s"] == pytest.approx(30.5)
    assert first["work_s"] == pytest.approx(1.6 * 30.5, rel=1e-5)
    # best layout of each budget, the default beside it
    bddc = _point(ms, "gfem:bddc", 16)
    assert (bddc["layout"], bddc["default_layout"]) == ("4x4", "1x16")
    assert bddc["work_s"] == pytest.approx(11.0)  # median of 10.8, 11.0, 11.2
    assert bddc["default_work_s"] == pytest.approx(12.2)  # of 12.0, 12.2, 12.4
    assert bddc["error"] is not None
    assert bddc["peak_rss_sum_gib"] is not None


def test_scaling_cross_code_ratio_is_labelled_different_nodes(tmp_path):
    c8 = _budget_campaign(tmp_path, 8, _slower(1.6))
    c16 = _budget_campaign(tmp_path, 16)
    ms = _case(_json(tmp_path, c8, c16), "microstrip")
    ratios = ms["scaling"]["ratios"]
    direct = next(
        r
        for r in ratios
        if r["core_budget"] == 16
        and (r["solver"], r["against"]) == ("palace:mumps", "gfem:direct")
    )
    assert direct["work"] == pytest.approx(41.0 / 30.5, rel=1e-5)
    assert direct["pairing"] == "different nodes"
    iterative = next(
        r
        for r in ratios
        if r["core_budget"] == 8
        and (r["solver"], r["against"]) == ("palace:pmg", "gfem:bddc")
    )
    assert iterative["pairing"] == "different nodes"
    assert "equal achieved error" in iterative["caveat"]
    assert direct["caveat"] is None


def test_a_node_class_change_restarts_the_scaling_line(tmp_path):
    c8 = _budget_campaign(tmp_path, 8, _on("standard", factor=3.0))
    c16 = _budget_campaign(tmp_path, 16, _on("highmem"))
    c32 = _budget_campaign(tmp_path, 32, _on("highmem", factor=0.5, nodes=2))
    ms = _case(_json(tmp_path, c8, c16, c32), "microstrip")
    p8, p16, p32 = (_point(ms, "gfem:direct", b) for b in (8, 16, 32))
    assert "restart" not in p8
    assert p16["restart"] == "line restarts: node class standard -> highmem"
    assert (p16["speedup"], p16["efficiency"]) == (1.0, 1.0)
    assert p16["baseline"] == 16
    # against c16 on highmem, not c8 on standard: S = 2, E = 16 * 2 / 32 = 1
    assert p32["speedup"] == pytest.approx(2.0)
    assert p32["efficiency"] == pytest.approx(1.0)
    assert "restart" not in p32
    assert "TCP between nodes" in _point(ms, "palace:mumps", 32)["labels"]
    assert "TCP between nodes" not in p32["labels"]


def test_scaling_section_and_figure(tmp_path):
    c8 = _budget_campaign(tmp_path, 8, _slower(1.6))
    c16 = _budget_campaign(tmp_path, 16)
    out = _report(tmp_path, str(c8), str(c16), "--no-figures", name="scal")
    index = (out / "index.md").read_text()
    section = index.index("### microstrip scaling across core budgets")
    assert index.index("### microstrip at c16 (standard, 1 node)") < section
    end = index.index("## lcavity")
    assert section < end
    text = index[section:end]
    assert "$S(c) = T(c_0)/T(c)$" in text
    assert "| palace:mumps / gfem:direct |" in text
    assert "different nodes" in text
    pytest.importorskip("matplotlib")
    figs = _report(tmp_path, str(c8), str(c16), name="scalfig")
    assert (figs / "fig-microstrip-scaling.svg").is_file()
    assert "](fig-microstrip-scaling.svg)" in (figs / "index.md").read_text()


def test_a_single_budget_keeps_the_case_keys(fixture_report):
    assert sorted(_case(fixture_report[0], "microstrip")) == [
        "accuracy",
        "campaigns",
        "cells",
        "comparison",
        "error_cost",
        "error_items",
        "figures",
        "kind",
        "layouts",
        "name",
        "notes",
        "phases",
        "pins",
        "reference",
        "rescore_flag",
        "rescored_against",
        "snapshot",
        "structural",
    ]
    assert "### microstrip at" not in fixture_report[1]


# --- figures ----------------------------------------------------------------------


def test_figures_are_deterministic_svg_and_check_passes(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    camp = _store(tmp_path)
    a = tmp_path / "store" / "reports" / "a"
    assert main(["report", str(camp), "--out", str(a)]) == 0
    svgs = sorted(p.name for p in a.glob("fig-*.svg"))
    assert "fig-microstrip-error-cost.svg" in svgs
    assert "fig-microstrip-layouts.svg" in svgs
    assert "fig-lcavity-error-items.svg" in svgs
    index = (a / "index.md").read_text()
    for name in svgs:
        assert f"]({name})" in index
        text = (a / name).read_text()
        assert "<dc:date>" not in text
        assert "#fcfcfb" in text
    assert main(["report", "--check", str(a)]) == 0
    assert "unchanged" in capsys.readouterr().out


def test_no_figures_needs_no_matplotlib(tmp_path):
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "from benchmarks.__main__ import main\n"
        f"assert main(['report', {str(FIXTURE)!r}, "
        f"'--out', {str(tmp_path / 'a')!r}]) == 1\n"
        f"sys.exit(main(['report', {str(FIXTURE)!r}, '--no-figures', "
        f"'--out', {str(tmp_path / 'b')!r}]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert done.returncode == 0, done.stderr
    assert "matplotlib is not importable" in done.stderr
    assert (tmp_path / "b" / "index.md").is_file()
