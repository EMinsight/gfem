"""The command-line runner: python -m gfem run job.json --out DIR."""

import json
import logging
import subprocess
import sys

import numpy as np
import pytest
import structured_meshes as sm
from solver_misses import cannot_converge_at

import gfem
from gfem import _cuda, _petsc
from gfem.__main__ import main

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
PROBLEM = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})


@pytest.fixture
def model(tmp_path):
    path = sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


def _job(model, tmp_path, **options):
    job = gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1, **options))
    return job.to_json(tmp_path / "job.json")


def test_cli_round_trip_equals_the_in_process_result(model, tmp_path):
    expected = gfem.solve(model, PROBLEM, options=gfem.SolverOptions(order=1))
    job = _job(model, tmp_path)
    out = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, "-m", "gfem", "run", str(job), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = gfem.load_result(out)
    assert result.terminals == expected.terminals
    # The same to rounding: threaded assembly does not fix the summation order.
    np.testing.assert_allclose(result.capacitance, expected.capacitance, rtol=1e-12)


def test_a_second_run_with_the_same_cache_reads_the_result_back(model, tmp_path):
    job = _job(model, tmp_path)
    cache = tmp_path / "cache"
    statuses = []
    for out in (tmp_path / "first", tmp_path / "second"):
        argv = ["run", str(job), "--out", str(out), "--cache", str(cache)]
        assert main(argv) == 0
        statuses.append(json.loads((out / "result.json").read_text())["run"]["cache"])
    assert [s["status"] for s in statuses] == ["miss", "hit"]
    assert statuses[1]["key"] == statuses[0]["key"]
    np.testing.assert_array_equal(
        gfem.load_result(tmp_path / "second").capacitance,
        gfem.load_result(tmp_path / "first").capacitance,
    )


def test_device_override_is_recorded(model, tmp_path):
    job = _job(model, tmp_path, linear_solver="direct", device="cuda")
    out = tmp_path / "results"
    assert main(["run", str(job), "--out", str(out), "--device", "cpu"]) == 0
    run = json.loads((out / "result.json").read_text())["run"]
    assert run["device"] == "cpu"
    assert run["device_override"] == "cpu"
    assert run["ranks"] == 1
    assert run["options"]["device"] == "cpu"


def test_an_auto_job_written_for_a_gpu_runs_the_cpu_default(model, tmp_path):
    job = _job(model, tmp_path, device="cuda")
    assert gfem.Job.from_json(job).options == gfem.SolverOptions(order=1, device="cuda")
    out = tmp_path / "results"
    assert main(["run", str(job), "--out", str(out), "--device", "cpu"]) == 0
    assert (
        json.loads((out / "result.json").read_text())["run"]["options"]["linear_solver"]
        == "auto"
    )


def test_no_override_records_the_job_device(model, tmp_path):
    out = tmp_path / "results"
    assert main(["run", str(_job(model, tmp_path)), "--out", str(out)]) == 0
    run = json.loads((out / "result.json").read_text())["run"]
    assert run["device"] == "cpu"
    assert run["device_override"] is None


@pytest.mark.parametrize("device", ["cuda", "tpu"])
def test_a_device_that_cannot_run_exits_1(model, tmp_path, capsys, monkeypatch, device):
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(0, "CUDA_ERROR_NO_DEVICE"))
    _cuda.probe.cache_clear()
    job = _job(model, tmp_path, linear_solver="direct")
    out = tmp_path / "results"
    assert main(["run", str(job), "--out", str(out), "--device", device]) == 1
    err = capsys.readouterr().err
    assert err.startswith("gfem: error:")
    assert "device" in err
    assert not (out / "result.json").exists()
    _cuda.probe.cache_clear()


@pytest.mark.skipif(
    sys.platform == "darwin", reason="macOS refuses CUDA before probing"
)
def test_a_gpu_job_without_a_gpu_names_the_cuda_error(
    model, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(0, "CUDA_ERROR_NO_DEVICE"))
    _cuda.probe.cache_clear()
    job = _job(model, tmp_path, linear_solver="direct", device="cuda")
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    err = capsys.readouterr().err
    assert "DeviceUnavailableError" in err
    assert "CUDA_ERROR_NO_DEVICE" in err
    assert "--device cpu" in err
    _cuda.probe.cache_clear()


@pytest.mark.skipif(
    sys.platform == "darwin", reason="macOS refuses CUDA before probing"
)
def test_a_default_job_sent_to_a_missing_gpu_names_the_cuda_error(
    model, tmp_path, capsys, monkeypatch
):
    # "auto" is the direct solver on a GPU, so the run reaches the GPU check
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(0, "CUDA_ERROR_NO_DEVICE"))
    _cuda.probe.cache_clear()
    job = _job(model, tmp_path)
    out = tmp_path / "r"
    assert main(["run", str(job), "--out", str(out), "--device", "cuda"]) == 1
    err = capsys.readouterr().err
    assert "DeviceUnavailableError" in err
    assert "CUDA_ERROR_NO_DEVICE" in err
    assert "ValueError" not in err
    _cuda.probe.cache_clear()


def test_a_petsc_job_without_petsc4py_exits_1(model, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "petsc4py", None)
    _petsc.require_petsc.cache_clear()
    job = _job(model, tmp_path, linear_solver="petsc")
    out = tmp_path / "r"
    assert main(["run", str(job), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "SolverUnavailableError" in err
    assert "petsc4py" in err
    assert not (out / "result.json").exists()
    _petsc.require_petsc.cache_clear()


def test_a_sweep_that_stops_exits_1_and_says_where_the_partial_result_is(
    tmp_path, capsys, monkeypatch
):
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1))
    )
    line = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    f = (2e9, 3e9, 4e9, 5e9, 6e9)
    job = gfem.Job(
        line, gfem.Driven(frequency=f), gfem.SolverOptions(linear_solver="bddc_gmres")
    ).to_json(tmp_path / "job.json")
    out = tmp_path / "results"
    cannot_converge_at(monkeypatch, (f[2],))
    assert main(["run", str(job), "--out", str(out)]) == 1
    error, note = capsys.readouterr().err.splitlines()
    assert error.startswith("gfem: error: LinearSolverConvergenceError: At 4e+09 Hz")
    assert note.startswith(
        f"gfem: note: The partial result (2 of 5 frequencies) is in {out}"
    )
    assert gfem.load_result(out / "partial.json").s.frequency == f[:2]
    assert not (out / "result.json").exists()


def test_log_shows_gfem_s_log_lines_on_stderr_for_the_run(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    argv = [
        "run",
        str(job),
        "--out",
        str(tmp_path / "r"),
        "--cache",
        str(tmp_path / "c"),
    ]
    assert main([*argv, "--log", "info"]) == 0
    lines = capsys.readouterr().err.splitlines()
    assert any(" INFO  cache  miss " in line for line in lines), lines
    assert logging.getHandlerByName("gfem.log") is None
    assert logging.getLogger("gfem").propagate


def test_without_log_the_run_writes_nothing_to_stderr(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    argv = [
        "run",
        str(job),
        "--out",
        str(tmp_path / "r"),
        "--cache",
        str(tmp_path / "c"),
    ]
    assert main(argv) == 0
    assert capsys.readouterr().err == ""
    assert logging.getHandlerByName("gfem.log") is None


def test_log_leaves_no_handler_when_the_run_fails(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    (tmp_path / "stack.msh").write_text("changed")
    assert main(["run", str(job), "--out", str(tmp_path / "r"), "--log", "DEBUG"]) == 1
    assert "sha256" in capsys.readouterr().err
    assert logging.getHandlerByName("gfem.log") is None
    assert logging.getLogger("gfem").propagate


def test_an_unknown_log_level_is_a_usage_error(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", str(job), "--out", str(tmp_path / "r"), "--log", "bogus"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage" in err
    assert "--log" in err
    assert "BOGUS" in err
    assert logging.getHandlerByName("gfem.log") is None
    assert not (tmp_path / "r").exists()


def test_a_broken_job_exits_1(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    (tmp_path / "stack.msh").write_text("changed")
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    assert "sha256" in capsys.readouterr().err


def test_a_newer_job_exits_1_asking_for_an_upgrade(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    doc = json.loads(job.read_text())
    doc["schema_version"] = 99
    job.write_text(json.dumps(doc))
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    assert "upgrade gfem" in capsys.readouterr().err


def test_a_job_with_an_unknown_entry_exits_1_asking_for_an_upgrade(
    model, tmp_path, capsys
):
    job = _job(model, tmp_path)
    doc = json.loads(job.read_text())
    doc["gfem_version"] = "99.0"
    doc["job"]["options"]["field_of_a_newer_gfem"] = 1
    job.write_text(json.dumps(doc))
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("gfem: error: ValueError:")
    assert "SolverOptions has no 'field_of_a_newer_gfem'" in err
    assert "upgrade gfem" in err


@pytest.mark.parametrize(
    "argv",
    [[], ["solve", "job.json"], ["run"], ["run", "job.json"], ["run", "--out", "r"]],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_a_missing_job_file_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r")])
    assert exc.value.code == 2
    assert "nope.json" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert gfem.__version__ in capsys.readouterr().out
