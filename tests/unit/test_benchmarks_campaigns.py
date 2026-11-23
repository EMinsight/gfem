"""Committed campaign files (`benchmarks/campaigns/*.toml`) plan cleanly."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.harness import config, matrix

REPO = Path(__file__).resolve().parents[2]
CAMPAIGNS = REPO / "benchmarks" / "campaigns"
FIXTURES = REPO / "tests" / "unit" / "data" / "bench" / "campaigns"
EXAMPLE_SITE = REPO / "benchmarks" / "site.example.toml"
RELEASE = CAMPAIGNS / "release-gfem.toml"
COMMIT = "858bc66" + "0" * 33

COMMITTED = sorted(CAMPAIGNS.glob("*.toml"))

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


@pytest.fixture(scope="module")
def site():
    return config.load_site(EXAMPLE_SITE, environ={})


def test_release_and_smoke_campaigns_are_committed():
    names = {p.stem for p in COMMITTED}
    assert {"release-gfem", "smoke", "smoke-r10m"} <= names
    assert {
        "reference-microstrip-pilot",
        "reference-microstrip-l2",
        "reference-microstrip-palace-0.18.1.post5",
        "reference-microstrip-palace-0.19.0",
    } <= names


#: The campaign series: small cases, bragg and scale cases at
#: every single-node core budget, plus bragg's Palace pmg cell at 16 cores.
BUDGETS = (8, 16, 32, 56, 112)
SERIES = sorted(
    [f"full-{s}c{b}" for s in ("", "bragg-", "scale-") for b in BUDGETS]
    + ["full-bragg-pmg-c16"]
)


def test_the_series_and_its_rehearsal_are_committed():
    names = {p.stem for p in COMMITTED}
    assert {*SERIES, "rehearsal-scale-c16"} <= names
    assert len(SERIES) == 16


def test_the_series_plans_26_jobs_1221_runs_and_about_6985_core_hours(site):
    campaigns = [config.load_campaign(CAMPAIGNS / f"{n}.toml") for n in SERIES]
    plans = [matrix.expand(c, site, commit=COMMIT) for c in campaigns]
    assert sum(len(p.jobs) for p in plans) == 26
    assert sum(p.runs for p in plans) == 1221
    # scale hints from the dress rehearsal (was 8476 on the guessed hints)
    assert sum(p.est_core_h for p in plans) == pytest.approx(6985, rel=0.03)


def test_the_rehearsal_plans_the_two_scale_cases_at_16_cores(site):
    path = CAMPAIGNS / "rehearsal-scale-c16.toml"
    plan = matrix.expand(config.load_campaign(path), site, commit=COMMIT)
    assert [job.cases for job in plan.jobs] == [("r10M_void",), ("microstrip_p3",)]
    assert {s.core_budget for j in plan.jobs for s in j.steps} == {16}
    # 385 when it was submitted; 314 on the rehearsal's own hints
    assert plan.est_core_h == pytest.approx(314, rel=0.03)


def test_series_hints_cover_every_planned_layout(site):
    # a cell without a "<solver>@<RxT>" hint would be estimated at the wall cap
    for name in SERIES:
        campaign = config.load_campaign(CAMPAIGNS / f"{name}.toml")
        cases = matrix.load_cases(campaign.cases)
        plan = matrix.expand(campaign, site, commit=COMMIT, cases=cases)
        for cell in plan.cells:
            if not cell.refused:
                key = f"{cell.solver}@{cell.layout}"
                assert key in cases[cell.case].est_s, (name, cell.case, key)


#: The only refusals a committed campaign plans: gfem's non-direct kinds on bragg's
#: Bloch-periodic eigenproblem.
BLOCH_REFUSED = {("bragg", "gfem:petsc-mumps"), ("bragg", "gfem:bddc")}


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.stem)
def test_every_committed_campaign_plans_only_the_expected_refusals(path, site):
    # besides the Bloch refusals, only gfem:direct (one process) on the scale files'
    # explicit multi-rank layout
    plan = matrix.expand(config.load_campaign(path), site, commit=COMMIT)
    assert plan.jobs
    for c in plan.cells:
        if c.refused and (c.case, c.solver) not in BLOCH_REFUSED:
            assert c.solver == "gfem:direct", (c.case, c.solver, c.layout)
            assert int(c.layout.split("x")[0]) > 1, (c.case, c.layout)


@pytest.mark.parametrize("name", ["smoke", "smoke-r10m"])
def test_smoke_campaigns_match_their_fixtures(name):
    committed = config.load_campaign(CAMPAIGNS / f"{name}.toml")
    fixture = config.load_campaign(FIXTURES / f"{name}.toml")
    assert replace(committed, path=fixture.path) == fixture


def test_release_gfem_is_one_stack_job_of_gfem_default_rows(site):
    plan = matrix.expand(config.load_campaign(RELEASE), site, commit=COMMIT)
    (job,) = plan.jobs
    assert job.group == "stack"
    assert job.name == "wheels"
    assert job.cases == ("bragg", "lcavity", "microstrip")
    assert job.constraint is None
    cells = {(s.cell.case, s.cell.solver) for s in job.steps}
    assert {solver for _, solver in cells} == {
        "gfem:direct",
        "gfem:petsc-mumps",
        "gfem:bddc",
    }
    assert len(cells) == 9
    refused = {(s.cell.case, s.cell.solver) for s in job.steps if s.cell.refused}
    assert refused == BLOCH_REFUSED
    assert {s.cell.asked for s in job.steps} == {"default"}
    assert {s.core_budget for s in job.steps} == {16}
    assert not any(s.cell.rung for s in job.steps)
    assert len(job.steps) == 23  # 7 rows x 3 repeats + 2 refused rows once
    assert {s.cell.tolerance for s in job.steps if s.cell.solver == "gfem:bddc"} == {
        1e-8
    }


def test_release_gfem_plan_command_prints_one_job(tmp_path, capsys):
    argv = ["plan", str(RELEASE), "--site", str(EXAMPLE_SITE)]
    assert main([*argv, "--out", str(tmp_path / "rel")]) == 0
    out = capsys.readouterr().out
    assert "23 runs" in out
    assert "wheels" in out
    assert "palace" not in out
