"""Committed campaign files (`benchmarks/campaigns/*.toml`) plan cleanly."""

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


#: The only refusals a committed campaign plans: gfem's non-direct kinds on bragg's
#: Bloch-periodic eigenproblem (ticket 12 amendment from ticket 14).
BLOCH_REFUSED = {("bragg", "gfem:petsc-mumps"), ("bragg", "gfem:bddc")}


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.stem)
def test_every_committed_campaign_plans_only_the_bloch_refusals(path, site):
    plan = matrix.expand(config.load_campaign(path), site, commit=COMMIT)
    assert plan.jobs
    refused = {(c.case, c.solver) for c in plan.cells if c.refused}
    assert refused <= BLOCH_REFUSED


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
