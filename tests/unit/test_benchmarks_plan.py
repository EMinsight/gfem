"""`benchmarks.harness.matrix`: a campaign expanded into jobs, runs and an estimate."""

import itertools
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from benchmarks import stacks
from benchmarks.__main__ import main
from benchmarks.adapters import (
    ADAPTERS,
    Kind,
    Launch,
    Layout,
    Outcome,
    RunRefusedError,
)
from benchmarks.harness import config, matrix, records
from benchmarks.harness.matrix import PlanError

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "unit" / "data" / "bench"
SMOKE = DATA / "campaigns" / "smoke.toml"
R10M = DATA / "campaigns" / "smoke-r10m.toml"
EXAMPLE_SITE = REPO / "benchmarks" / "site.example.toml"
COMMIT = "858bc66" + "0" * 33

#: probes/12/plan.out, core-hours per job (ticket 12 Q3); bragg less its six refused
#: gfem runs (25 x 33 / 39; ticket 12 amendment from ticket 14), then re-sized for the
#: once-split mesh (est_s palace:mumps 15 s, palace:pmg 60 s: 35 runs, 49 core-h).
PROBE_CORE_H = {"bragg": 49, "lcavity": 168, "microstrip": 253, "r10M_void": 50}


@pytest.fixture(scope="module")
def site():
    return config.load_site(EXAMPLE_SITE, environ={})


@pytest.fixture(scope="module")
def smoke(site):
    return matrix.expand(config.load_campaign(SMOKE), site, commit=COMMIT)


@pytest.fixture(scope="module")
def r10m(site):
    return matrix.expand(config.load_campaign(R10M), site, commit=COMMIT)


def _write(tmp_path: Path, text: str, name: str = "c.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _campaign(tmp_path, old: str = "", new: str = "", name: str = "smoke.toml"):
    text = SMOKE.read_text().replace(old, new, 1) if old else SMOKE.read_text()
    return config.load_campaign(_write(tmp_path, text, name))


# --- layout grid -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "layouts"),
    [
        (16, ["1x16", "2x8", "4x4", "8x2", "16x1"]),
        (64, ["1x64", "8x8", "16x4", "32x2", "64x1"]),
        (
            112,
            [
                "1x112",
                "2x56",
                "4x28",
                "8x14",
                "14x8",
                "16x7",
                "28x4",
                "56x2",
                "112x1",
            ],
        ),
    ],
)
def test_grid_is_every_layout_with_no_rank_across_a_socket(budget, layouts):
    assert [str(lay) for lay in matrix.grid(budget)] == layouts


# --- the smoke matrix ------------------------------------------------------------


def test_smoke_expands_to_149_runs_in_three_jobs(smoke):
    assert [job.name for job in smoke.jobs] == [
        "bragg-wheels",
        "lcavity-wheels",
        "microstrip-wheels",
    ]
    # bragg: 11 cells x 3 repeats + gfem petsc-mumps and bddc refused, once each
    assert [len(job.steps) for job in smoke.jobs] == [35, 39, 75]
    assert smoke.runs == 149


def test_smoke_cells_per_case(smoke):
    anchor = [c for c in smoke.cells if not c.rung]
    per_case = {
        case: sum(c.case == case for c in anchor)
        for case in ("bragg", "lcavity", "microstrip")
    }
    assert per_case == {"bragg": 13, "lcavity": 13, "microstrip": 21}


def test_grid_cells_leave_out_refused_layouts(smoke):
    def layouts(case, solver):
        return [c.layout for c in smoke.cells if (c.case, c.solver) == (case, solver)]

    assert layouts("lcavity", "gfem:bddc") == ["1x16"]
    assert layouts("microstrip", "gfem:direct") == ["1x16"]
    assert layouts("microstrip", "palace:mumps") == [
        "1x16",
        "2x8",
        "4x4",
        "8x2",
        "16x1",
    ]
    assert {c.asked for c in smoke.cells if not c.rung} == {"grid"}


def test_a_kind_refused_at_every_grid_layout_is_planned_once_refused(smoke):
    cells = [c for c in smoke.cells if c.case == "bragg" and c.refused]
    assert sorted((c.solver, c.layout, c.asked) for c in cells) == [
        ("gfem:bddc", "1x16", "grid"),
        ("gfem:petsc-mumps", "1x16", "grid"),
    ]
    for cell in cells:
        assert "direct" in cell.refused
        assert cell.repeats == 1
        assert cell.default


def test_default_layout_is_marked_in_the_grid(smoke):
    defaults = {(c.case, c.solver): c.layout for c in smoke.cells if c.default}
    assert defaults[("microstrip", "gfem:bddc")] == "1x16"
    assert defaults[("microstrip", "palace:pmg")] == "16x1"


def test_ladder_rungs_run_off_anchor_at_the_default_layout(smoke):
    rungs = [c for c in smoke.cells if c.rung]
    assert sorted((c.solver, c.layout, c.tolerance) for c in rungs) == [
        ("gfem:bddc", "1x16", 1e-10),
        ("gfem:bddc", "1x16", 1e-06),
        ("palace:pmg", "16x1", 1e-10),
        ("palace:pmg", "16x1", 1e-06),
    ]
    assert all(c.case == "microstrip" and c.repeats == 3 for c in rungs)
    assert all(c.asked == "default" for c in rungs)


def test_tolerance_is_the_anchor_on_iterative_kinds_and_none_on_direct(smoke):
    tolerances = {c.solver: c.tolerance for c in smoke.cells if not c.rung}
    assert tolerances == {
        "gfem:direct": None,
        "gfem:petsc-mumps": None,
        "gfem:bddc": 1e-8,
        "palace:mumps": None,
        "palace:pmg": 1e-8,
    }


def test_run_names_are_unique_and_rungs_name_their_tolerance(smoke):
    names = [step.name for job in smoke.jobs for step in job.steps]
    assert len(set(names)) == len(names)
    assert "microstrip/gfem-bddc__wheels__default__tol1e-06__rep2" in names
    assert "microstrip/gfem-bddc__wheels__1x16__rep2" in names


def test_rotation_never_puts_two_repeats_of_a_row_back_to_back(smoke):
    for job in smoke.jobs:
        cells = [step.cell for step in job.steps]
        assert all(a != b for a, b in itertools.pairwise(cells))
        assert [s.line for s in job.steps] == list(range(1, len(job.steps) + 1))


@pytest.mark.parametrize("n", range(2, 12))
@pytest.mark.parametrize("repeats", [2, 3, 4])
def test_rotation_property_for_any_row_count(n, repeats):
    order = matrix.rotate(list(range(n)), repeats)
    assert sorted(order) == sorted(list(range(n)) * repeats)
    assert all(a != b for a, b in itertools.pairwise(order))
    assert order[:n] == list(range(n))  # every row's repeat 1 runs in round 1


def test_estimate_within_five_percent_of_the_sizing_probe(smoke, r10m):
    for job in (*smoke.jobs, *r10m.jobs):
        expected = PROBE_CORE_H[job.cases[0]]
        assert job.est_core_h == pytest.approx(expected, rel=0.05), job.name
    assert smoke.est_core_h == pytest.approx(469, rel=0.05)


def test_qos_by_time_limit_and_limit_is_one_and_a_half_estimates(smoke):
    qos = {job.name: job.qos for job in smoke.jobs}
    assert qos == {
        "bragg-wheels": "debug",
        "lcavity-wheels": "long",
        "microstrip-wheels": "long",
    }
    for job in smoke.jobs:
        assert job.limit_s == pytest.approx(1.5 * job.est_s)


def test_timeout_cell_counts_once_in_the_estimate(smoke):
    (cell,) = [
        c
        for c in smoke.cells
        if (c.case, c.solver, c.layout) == ("microstrip", "palace:mumps", "1x16")
    ]
    assert cell.est_s == 600
    assert cell.est_total_s == 600


def test_r10m_is_one_highmem_job_of_four_default_rows(r10m):
    (job,) = r10m.jobs
    assert job.constraint == "highmem"
    assert len(job.steps) == 4
    assert {(s.cell.solver, s.cell.layout) for s in job.steps} == {
        ("gfem:direct", "1x112"),
        ("gfem:bddc", "1x112"),
        ("palace:mumps", "112x1"),
        ("palace:pmg", "112x1"),
    }
    assert {s.cell.asked for s in job.steps} == {"default"}


def test_group_stack_packs_every_case_into_one_job(tmp_path, site):
    campaign = _campaign(tmp_path, 'group = "case"', 'group = "stack"')
    plan = matrix.expand(campaign, site, commit=COMMIT)
    (job,) = plan.jobs
    assert job.name == "wheels"
    assert job.cases == ("bragg", "lcavity", "microstrip")
    assert len(job.steps) == 149


# --- same-code direct runs ---------------------------------------------------------


def test_same_code_direct_added_when_missing(tmp_path, site):
    campaign = _campaign(
        tmp_path,
        'solvers = ["gfem:direct", "gfem:petsc-mumps", "gfem:bddc", '
        '"palace:mumps", "palace:pmg"]',
        'solvers = ["gfem:bddc", "palace:pmg"]',
    )
    plan = matrix.expand(campaign, site, commit=COMMIT)
    added = [c for c in plan.cells if c.added_for]
    # bragg's gfem:bddc is refused (Bloch phase), so it needs no gfem direct run
    assert sorted((c.case, c.solver, c.layout) for c in added) == [
        ("bragg", "palace:mumps", "16x1"),
        ("lcavity", "gfem:direct", "1x16"),
        ("lcavity", "palace:mumps", "16x1"),
        ("microstrip", "gfem:direct", "1x16"),
        ("microstrip", "palace:mumps", "16x1"),
    ]
    assert {(c.added_for, c.repeats, c.asked) for c in added} == {
        ("same_code_error", 1, "default")
    }


def test_same_code_direct_not_added_when_present(smoke):
    assert not [c for c in smoke.cells if c.added_for]


def test_any_direct_kind_of_the_code_serves(tmp_path, site):
    campaign = _campaign(
        tmp_path,
        'solvers = ["gfem:direct", "gfem:petsc-mumps", "gfem:bddc", '
        '"palace:mumps", "palace:pmg"]',
        'solvers = ["gfem:petsc-mumps", "gfem:bddc", "palace:pmg"]',
    )
    plan = matrix.expand(campaign, site, commit=COMMIT)
    assert [(c.case, c.solver) for c in plan.cells if c.added_for] == [
        ("bragg", "palace:mumps"),
        ("lcavity", "palace:mumps"),
        ("microstrip", "palace:mumps"),
    ]


# --- refusals ----------------------------------------------------------------------


def test_refused_cells_are_planned_with_their_reason(tmp_path, site):
    campaign = _campaign(tmp_path, 'layouts = ["best"]', 'layouts = ["default", "4x4"]')
    plan = matrix.expand(campaign, site, commit=COMMIT)
    refused = {(c.case, c.solver, c.layout): c.refused for c in plan.cells if c.refused}
    assert "one process only" in refused[("microstrip", "gfem:direct", "4x4")]
    assert "eigen" in refused[("lcavity", "gfem:bddc", "4x4")]
    for cell in plan.cells:
        if cell.refused:
            assert cell.repeats == 1
            assert cell.est_total_s == 0
    steps = [s for job in plan.jobs for s in job.steps if s.cell.refused]
    assert steps
    assert all(s.to_json()["refused"] for s in steps)


def test_explicit_layout_over_budget_is_refused():
    campaign = config.load_campaign(SMOKE)
    with pytest.raises(PlanError, match="core budget"):
        matrix.override(campaign, layouts=("8x4",))


def test_best_on_a_large_case_is_refused(tmp_path, site):
    text = R10M.read_text().replace('layouts = ["default"]', 'layouts = ["best"]')
    campaign = config.load_campaign(_write(tmp_path, text, "smoke-r10m.toml"))
    with pytest.raises(PlanError, match="large"):
        matrix.expand(campaign, site, commit=COMMIT)


def test_native_stack_without_provenance_is_refused(site):
    campaign = matrix.override(config.load_campaign(SMOKE), stacks=("native-oss",))
    with pytest.raises(PlanError, match=r"provenance\.json"):
        matrix.expand(campaign, site, commit=COMMIT)


def test_overrides_narrow_the_axes(site):
    campaign = matrix.override(
        config.load_campaign(SMOKE),
        cases=("bragg",),
        solvers=("gfem:direct", "palace:mumps"),
        layouts=("default",),
        repeats=2,
    )
    plan = matrix.expand(campaign, site, commit=COMMIT)
    assert plan.runs == 4
    assert campaign.ladder == ()


# --- the run directory ---------------------------------------------------------------


def _rundir(tmp_path, plan, site, **kw):
    out = tmp_path / "run"
    manifest = matrix.write(
        plan, out, site=site, commit_utc="2026-11-17T00:00:00Z", **kw
    )
    return out, manifest


def test_write_lays_out_the_run_directory(tmp_path, smoke, site):
    out, manifest = _rundir(tmp_path, smoke, site)
    assert (out / "COMMIT").read_text().strip() == COMMIT
    resolved = tomllib.loads((out / "campaign.toml").read_text())
    assert resolved["cases"] == ["bragg", "lcavity", "microstrip"]
    assert resolved["stacks"]["wheels"]["versions"]["palace"] == "0.19.0"
    tsv = (out / "plan.tsv").read_text().splitlines()
    assert tsv[0].split("\t")[0] == "job"
    assert len(tsv) == 4
    lines = (out / "jobs" / "microstrip-wheels.steps").read_text().splitlines()
    assert len(lines) == 75
    first = json.loads(lines[0])
    assert first["line"] == 1
    assert first["job"] == "microstrip-wheels"
    assert first["wall_cap_s"] == 600
    loaded = records.load_campaign(out / "campaign.json")
    assert loaded.id == manifest.id
    assert manifest.id.endswith(f"-{COMMIT[:7]}-smoke")
    assert loaded.pins.gfem["commit"] == COMMIT
    assert loaded.pins.solvers == {"palace": "0.19.0"}
    assert [j.job for j in loaded.jobs] == [j.name for j in smoke.jobs]
    assert loaded.machine == "example-cluster"


def test_written_files_name_no_site_path(tmp_path, smoke, site):
    out, _ = _rundir(tmp_path, smoke, site)
    for path in out.rglob("*"):
        if path.is_file():
            assert site.cluster_root not in path.read_text(), path


def test_replanning_an_existing_run_directory_at_another_commit_is_refused(
    tmp_path, smoke, site
):
    out, manifest = _rundir(tmp_path, smoke, site)
    again = matrix.expand(config.load_campaign(SMOKE), site, commit=COMMIT)
    assert matrix.write(again, out, site=site, commit_utc="x").id == manifest.id
    other = matrix.expand(config.load_campaign(SMOKE), site, commit="f" * 40)
    with pytest.raises(PlanError, match="commit"):
        matrix.write(other, out, site=site, commit_utc="x")


def test_replanning_another_campaign_into_a_run_directory_is_refused(
    tmp_path, smoke, r10m, site
):
    out, _ = _rundir(tmp_path, smoke, site)
    with pytest.raises(PlanError, match="smoke-r10m"):
        matrix.write(r10m, out, site=site, commit_utc="x")


# --- command line ------------------------------------------------------------------


def test_plan_command_prints_jobs_and_total(tmp_path, capsys):
    argv = [
        "plan",
        str(SMOKE),
        "--site",
        str(EXAMPLE_SITE),
        "--out",
        str(tmp_path / "r"),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "bragg-wheels" in out
    assert "microstrip-wheels" in out
    assert "149 runs" in out
    assert (tmp_path / "r" / "campaign.json").is_file()


def test_plan_command_refuses_native_stack_with_provenance_reason(tmp_path, capsys):
    argv = [
        "plan",
        str(SMOKE),
        "--stack",
        "native-oss",
        "--site",
        str(EXAMPLE_SITE),
        "--out",
        str(tmp_path / "r"),
    ]
    assert main(argv) == 2
    assert "provenance.json" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


# --- a third code ------------------------------------------------------------------


@dataclass(frozen=True)
class _ThirdAdapter:
    """A stand-in solver adapter: a code the suite has never heard of."""

    code: str = "third"
    kinds: Mapping[str, Kind] = field(
        default_factory=lambda: {"lu": Kind("direct"), "cg": Kind("iterative")}
    )
    direct_kind: str = "lu"
    pins: tuple[str, ...] = ("no field output",)
    timing_source: str = "third.json"
    memory_metric: str = "VmHWM_sum"
    stack_keys: tuple[str, ...] = ("entry",)

    def supports(self, _case, _kind, _layout):
        return None

    def default_layout(self, _case, _kind, budget):
        return Layout(1, budget)

    def version(self, _stack):
        return Launch(commands=())

    def prepare(self, _run):
        raise RunRefusedError("a stand-in")

    def collect(self, _run, _measured):
        return Outcome(status="refused")


THIRD_SECTION = """
[third]
modules = []
mpi = "pmix"
entry = "third-bin"

[third.threads]
OMP_NUM_THREADS = "threads"
"""


def test_a_third_code_is_one_adapter_one_registry_line_and_a_stack_section(
    site, tmp_path, monkeypatch
):
    monkeypatch.setitem(ADAPTERS, "third", _ThirdAdapter())
    stacks_dir = tmp_path / "stacks"
    stacks_dir.mkdir()
    wheels = (REPO / "benchmarks" / "stacks" / "wheels.toml").read_text()
    wheels = wheels.replace("[versions]", '[versions]\nthird = "1.2"', 1)
    (stacks_dir / "wheels.toml").write_text(wheels + THIRD_SECTION)
    monkeypatch.setattr(stacks, "STACKS_DIR", stacks_dir)
    text = (DATA / "tiny" / "campaign.toml").read_text()
    text = text.replace('cases_dir = ".."', f'cases_dir = "{DATA}"')
    text = text.replace(
        'solvers = ["gfem:direct", "gfem:bddc"]', 'solvers = ["gfem:bddc", "third:cg"]'
    )
    plan = matrix.expand(
        config.load_campaign(_write(tmp_path, text)), site, commit=COMMIT
    )
    added = {c.solver for c in plan.cells if c.added_for == matrix.SAME_CODE_ERROR}
    assert {c.solver for c in plan.cells} == {
        "gfem:bddc",
        "gfem:direct",
        "third:cg",
        "third:lu",
    }
    assert added == {"gfem:direct", "third:lu"}
    assert plan.solver_versions() == {"third": "1.2"}
    script = stacks.render(plan.resolved["wheels"], "third", threads=4)
    assert "export OMP_NUM_THREADS=4" in script


def test_a_stack_without_the_codes_section_is_refused_at_plan(
    site, tmp_path, monkeypatch
):
    monkeypatch.setitem(ADAPTERS, "third", _ThirdAdapter())
    text = (DATA / "tiny" / "campaign.toml").read_text()
    text = text.replace('cases_dir = ".."', f'cases_dir = "{DATA}"')
    text = text.replace(
        'solvers = ["gfem:direct", "gfem:bddc"]', 'solvers = ["third:cg"]'
    )
    with pytest.raises(PlanError, match=r"\[third\]"):
        matrix.expand(config.load_campaign(_write(tmp_path, text)), site, commit=COMMIT)
