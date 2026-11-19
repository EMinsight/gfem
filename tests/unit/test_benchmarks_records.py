"""`benchmarks.harness.records`: suite records, campaign manifests, the store set."""

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.harness import records

DATA = Path(__file__).resolve().parent / "data" / "bench"
RECORD = DATA / "record.example.json"
CAMPAIGN = DATA / "campaign.example.json"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_record_round_trips_the_example():
    raw = _json(RECORD)
    record = records.SuiteRecord.from_json(raw)
    assert record.run.layout.threads == 16
    assert record.case.reference is not None
    assert record.case.reference.id == "microstrip-20261121-858bc66"
    assert record.to_json() == raw


def test_campaign_round_trips_the_example():
    raw = _json(CAMPAIGN)
    campaign = records.Campaign.from_json(raw)
    assert campaign.pins.solvers == {"gfem": "0.1.0", "palace": "0.19.0"}
    assert campaign.jobs[0].billed_core_h == pytest.approx(165.5)
    assert campaign.to_json() == raw


def test_record_json_starts_with_its_schema():
    record = records.SuiteRecord.from_json(_json(RECORD))
    assert next(iter(record.to_json())) == "schema"


def test_load_reads_a_stored_record_and_manifest():
    assert records.load_record(RECORD).name == (
        "microstrip/gfem-direct__wheels__default__rep1"
    )
    assert records.load_campaign(CAMPAIGN).id == "20261120-858bc66-smoke"


def test_unknown_major_is_refused_with_its_path(tmp_path):
    raw = _json(RECORD) | {"schema": "suite-record/2"}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(records.RecordError, match=r"future\.json.*suite-record/2"):
        records.load_record(path)


def test_a_manifest_is_not_read_as_a_record():
    with pytest.raises(records.RecordError, match="suite-campaign/1"):
        records.load_record(CAMPAIGN)


def test_unknown_status_and_check_value_are_refused():
    with pytest.raises(records.RecordError, match="status 'done'"):
        records.SuiteRecord.from_json(_json(RECORD) | {"status": "done"})
    raw = _json(RECORD)
    raw["checks"]["dof"] = "ok"
    with pytest.raises(records.RecordError, match=r"checks\.dof"):
        records.SuiteRecord.from_json(raw)


def test_null_phases_stay_null_through_a_write(tmp_path):
    record = records.SuiteRecord.from_json(_json(RECORD))
    path = records.write_record(record, tmp_path / "records", roots={})
    stored = _json(path)
    assert stored["timings"]["estimate_s"] is None
    assert stored["timings"]["rom_s"] is None
    assert stored["timings"]["rank0"]["postpro_s"] is None
    assert (
        path == tmp_path / "records/microstrip/gfem-direct__wheels__default__rep1.json"
    )


def test_a_started_stub_needs_only_the_run():
    run = records.Run(
        case="bragg",
        solver="palace:pmg",
        stack="wheels",
        layout=records.Layout(asked="4x4", ranks=4, threads=4),
        core_budget=16,
        repeat=2,
    )
    stub = records.SuiteRecord(campaign="c", name="n", run=run, status="started")
    again = records.SuiteRecord.from_json(json.loads(json.dumps(stub.to_json())))
    assert again == stub
    assert again.timings.solve_s is None


ROOTS = {
    "{cluster.root}": "/gpfs/projects/acct42/someone/gfem",
    "{stacks.wheels.palace_wheelhouse}": "/gpfs/projects/acct42/someone/gfem/palace-wh",
}


def test_scrub_replaces_site_roots_longest_first():
    text = {
        "env_script": "export P=/gpfs/projects/acct42/someone/gfem/envs/x/bin\n",
        "wheel": "/gpfs/projects/acct42/someone/gfem/palace-wh/palace_solver.whl",
        "sibling": "/gpfs/projects/acct42/someone/gfem2",
    }
    roots = ROOTS | {"{other}": "/opt/sw"}
    with pytest.raises(records.ScrubError, match=r"\$\.sibling: /gpfs/"):
        records.scrub(text, roots)
    del text["sibling"]
    assert records.scrub(text, roots) == {
        "env_script": "export P={cluster.root}/envs/x/bin\n",
        "wheel": "{stacks.wheels.palace_wheelhouse}/palace_solver.whl",
    }


def test_scrub_catches_a_gpfs_path_and_a_home_path():
    with pytest.raises(records.ScrubError, match=r"/gpfs/scratch/acct42/x\.log"):
        records.scrub({"log": "see /gpfs/scratch/acct42/x.log"}, ROOTS)
    with pytest.raises(records.ScrubError, match=r"/home/someone/\.venv"):
        records.scrub(["ok", {"ldd": "libblas => /home/someone/.venv/lib"}], ROOTS)


def test_scrub_catches_the_account_and_user():
    with pytest.raises(records.ScrubError, match="acct42"):
        records.scrub({"qos": "acct42_debug x acct42"}, ROOTS, secrets=["acct42"])
    assert records.scrub({"a": "acct420"}, ROOTS, secrets=["acct42"]) == {
        "a": "acct420"
    }


def test_writing_a_record_scrubs_it(tmp_path):
    raw = _json(RECORD)
    raw["stack"]["env"] = "source /home/someone/env.sh\n"
    record = records.SuiteRecord.from_json(raw)
    with pytest.raises(records.ScrubError, match=r"stack\.env"):
        records.write_record(record, tmp_path, roots=ROOTS)
    assert not list(tmp_path.rglob("*.json"))


def test_site_roots_cover_every_site_path(tmp_path):
    site = tmp_path / "site.toml"
    site.write_text(
        'name = "c"\n[slurm]\naccount = "acct42"\n[cluster]\nroot = "/r"\n'
        '[stacks]\nroot = "/s"\n[stacks.wheels]\nbase_venv = "{root}/venv"\n'
        'wheel_dirs = ["{root}/w1", "{root}/w2"]\n'
        '[data]\ndir = "{root}/data"\nstores = ["{root}/store"]\n'
        '[ship]\nmesh_store = "{root}/store"\n[output]\ndir = "{root}/runs"\n'
    )
    from benchmarks.harness import config

    loaded = config.load_site(site)
    assert records.site_roots(loaded) == {
        "{cluster.root}": "/r",
        "{stacks.root}": "/s",
        "{data.dir}": "/r/data",
        "{output.dir}": "/r/runs",
        "{data.stores[0]}": "/r/store",
        "{ship.mesh_store}": "/r/store",
        "{stacks.wheels.base_venv}": "/r/venv",
        "{stacks.wheels.wheel_dirs[0]}": "/r/w1",
        "{stacks.wheels.wheel_dirs[1]}": "/r/w2",
    }
    assert records.site_secrets(loaded, {"USER": "someone"}) == ("acct42", "someone")


def test_record_name_joins_the_axes():
    assert (
        records.record_name("microstrip", "gfem:bddc", "wheels", "4x4", 3)
        == "microstrip/gfem-bddc__wheels__4x4__rep3"
    )


def test_record_name_of_an_off_anchor_rung_names_its_tolerance():
    assert (
        records.record_name(
            "microstrip", "gfem:bddc", "wheels", "default", 1, rung=1e-6
        )
        == "microstrip/gfem-bddc__wheels__default__tol1e-06__rep1"
    )


def test_supersede_moves_a_record_to_the_first_free_number(tmp_path):
    path = tmp_path / "gfem-direct__wheels__default__rep1.json"
    assert records.supersede(path) is None
    path.write_text("first")
    first = records.supersede(path)
    path.write_text("second")
    second = records.supersede(path)
    assert first is not None
    assert first.name.endswith(".superseded-1.json")
    assert second is not None
    assert second.name.endswith(".superseded-2.json")
    assert second.read_text() == "second"
    assert not path.exists()
    assert records.is_superseded(second)
    assert not records.is_superseded(path)


def test_campaign_id_takes_minus_two_on_a_same_day_rerun():
    day = dt.date(2026, 11, 20)
    sha = "858bc66c0ffee"
    assert records.campaign_id(day, sha, "smoke", ()) == "20261120-858bc66-smoke"
    taken = ["20261120-858bc66-smoke", "20261120-858bc66-smoke-2", "x"]
    assert records.campaign_id(day, sha, "smoke", taken) == "20261120-858bc66-smoke-3"
    assert records.campaign_id(day, sha, "smoke-r10m", taken[:1]) == (
        "20261120-858bc66-smoke-r10m"
    )


def _palace_record() -> records.SuiteRecord:
    raw = _json(RECORD)
    raw["name"] = "microstrip/palace-mumps__wheels__default__rep1"
    raw["run"]["solver"] = "palace:mumps"
    raw["provenance"]["solver"] = {"code": "palace", "version": "0.19.0"}
    return records.SuiteRecord.from_json(raw)


def test_a_campaign_agreeing_with_its_pins_is_not_mixed():
    campaign = records.load_campaign(CAMPAIGN)
    rows = [records.load_record(RECORD), _palace_record()]
    assert records.mixed(campaign, rows) == []
    assert records.refuse_mixed(campaign, rows) == []


def test_mixed_campaigns_are_refused_unless_allowed():
    campaign = records.load_campaign(CAMPAIGN)
    other = _palace_record()
    other.provenance.solver["version"] = "0.18.1.post5"
    stray = records.load_record(RECORD)
    stray.provenance.gfem["commit"] = "1234567aaaa"
    rows = [other, stray]
    with pytest.raises(records.RecordError, match=r"palace 0\.18\.1\.post5 != pinned"):
        records.refuse_mixed(campaign, rows)
    problems = records.refuse_mixed(campaign, rows, allow_mixed=True)
    assert len(problems) == 2
    assert any("gfem commit 1234567aaaa" in p for p in problems)


def test_a_record_of_another_campaign_or_stack_hash_is_mixed():
    campaign = records.load_campaign(CAMPAIGN)
    row = records.load_record(RECORD)
    row.campaign = "20261121-858bc66-smoke"
    row.provenance.stack["env_manifest_sha256"] = "other"
    problems = records.mixed(campaign, [row])
    assert len(problems) == 2


def test_a_refused_record_without_provenance_is_not_mixed():
    campaign = records.load_campaign(CAMPAIGN)
    row = records.load_record(RECORD)
    row.status = "not_started"
    row.provenance = records.Provenance()
    assert records.mixed(campaign, [row]) == []


def test_a_campaign_refuses_runs_at_another_commit_or_solver_version():
    campaign = records.load_campaign(CAMPAIGN)
    same = records.Pins.from_json(campaign.pins.to_json())
    records.check_extends(campaign, same)
    moved = records.Pins.from_json(campaign.pins.to_json())
    moved.solvers["palace"] = "0.18.1.post5"
    with pytest.raises(records.RecordError, match="new commit or solver version"):
        records.check_extends(campaign, moved)
    moved = records.Pins.from_json(campaign.pins.to_json())
    moved.gfem["commit"] = "abcdef0"
    with pytest.raises(records.RecordError, match="gfem commit abcdef0"):
        records.check_extends(campaign, moved)


STORE_SET = {
    "campaign.json",
    "records/microstrip/gfem-direct__wheels__default__rep1.json",
    "records/bragg/palace-pmg__wheels__4x4__rep2.json",
    "jobs/41234567/node.txt",
    "jobs/41234567/pip-freeze.txt",
    "jobs/41234567/machine.json",
}


def _run_dir(root: Path) -> Path:
    rundir = root / "20261120-858bc66-smoke"
    noise = {
        "records/microstrip/gfem-direct__wheels__default__rep1.superseded-1.json",
        "records/microstrip/gfem-direct__wheels__default__rep1.log",
        "records/microstrip/gfem-direct__wheels__default__rep1.json.tmp",
        "work/microstrip/gfem-direct__wheels__default__rep1/outcome.json",
        "logs/41234567.out",
        "jobs/41234567/job.log",
        "jobs/41234567/steps.tsv",
        "submitted.tsv",
        "job-1.sbatch",
    }
    for rel in STORE_SET | noise:
        path = rundir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return rundir


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


needs_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="fetch runs rsync"
)


@needs_rsync
def test_fetch_records_only_writes_exactly_the_store_set(tmp_path):
    rundir = _run_dir(tmp_path / "cluster")
    dest = tmp_path / "bench-data" / "campaigns" / rundir.name
    status = main(["fetch", "--from", str(rundir), "--to", str(dest), "--records-only"])
    assert status == 0
    assert _files(dest) == STORE_SET


@needs_rsync
def test_fetch_copies_the_whole_run_dir(tmp_path):
    rundir = _run_dir(tmp_path / "cluster")
    dest = tmp_path / "copy"
    assert main(["fetch", "--from", f"{rundir}/", "--to", str(dest)]) == 0
    assert _files(dest) == _files(rundir)


@needs_rsync
def test_fetch_records_only_flags_a_leaked_path(tmp_path, capsys):
    rundir = _run_dir(tmp_path / "cluster")
    (rundir / "jobs/41234567/pip-freeze.txt").write_text(
        "gfem @ file:///gpfs/projects/acct42/someone/gfem.whl\n"
    )
    dest = tmp_path / "store"
    assert main(["fetch", "--from", str(rundir), "--to", str(dest), "--records-only"])
    assert "jobs/41234567/pip-freeze.txt" in capsys.readouterr().err


def test_fetch_reports_a_failing_rsync(tmp_path, capsys):
    calls = []

    def fake(argv, *, check):
        assert check is False
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 23)

    status = records.fetch("host:/nowhere/run", tmp_path / "d", run=fake)
    assert status == 23
    assert calls[0][0] == "rsync"
    assert calls[0][-2:] == ["host:/nowhere/run/", f"{tmp_path / 'd'}/"]
    assert "rsync exited 23" in capsys.readouterr().err


def test_provenance_env_keeps_only_the_allow_list():
    environ = {
        "OMP_NUM_THREADS": "4",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_ACCOUNT": "acct42",
        "HOME": "/home/someone",
        "UCX_TLS": "tcp",
        "PATH": "/usr/bin",
    }
    assert records.allowed_env(environ) == {
        "OMP_NUM_THREADS": "4",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_NTASKS": "2",
        "UCX_TLS": "tcp",
    }


def test_reading_does_not_share_state_with_the_source():
    raw = _json(RECORD)
    record = records.SuiteRecord.from_json(raw)
    record.outputs["iterations"].append(9)
    assert raw["outputs"]["iterations"] == [0, 0]


def test_a_stack_block_from_the_stacks_module_is_stored_as_is(tmp_path):
    from benchmarks import stacks
    from benchmarks.harness import config

    site_file = tmp_path / "site.toml"
    site_file.write_text(
        'name = "c"\n[slurm]\naccount = "acct42"\n'
        '[cluster]\nroot = "/home/someone/gfem"\n'
        '[stacks.wheels]\nbase_venv = "{root}/venv"\nwheel_dirs = ["{root}/w"]\n'
        'petsc_wheelhouse = "{root}/petsc"\npalace_wheelhouse = "{root}/palace"\n'
        'python = "{root}/py/bin/python3"\n'
        '[data]\ndir = "{root}/data"\n[output]\ndir = "{root}/runs"\n'
    )
    site = config.load_site(site_file)
    resolved = stacks.resolve("wheels", site, commit="858bc66" + "0" * 33)
    block = stacks.record(
        resolved,
        env_script=stacks.render(resolved, "gfem", threads=4),
        ldd="libopenblas.so => /home/someone/gfem/venv/lib/libopenblas.so",
    )
    record = records.SuiteRecord.from_json(_json(RECORD) | {"stack": block})
    path = records.write_record(
        record,
        tmp_path / "records",
        roots=records.site_roots(site),
        secrets=records.site_secrets(site, {"USER": "someone"}),
    )
    assert _json(path)["stack"] == block
