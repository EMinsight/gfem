"""`benchmarks.adapters.palace`: render, launch and parse per Palace dialect (ADR 0404).

Outputs are committed fixtures. 0.18: wheel 0.18.1.post5 runs on the cluster (r8_void
driven at 4x1 and 1x4, the bragg eigen run, a killed r10M_void run) and a 2x1 run of
microstrip at order 1 rendered by this adapter. 0.19: a native 0.19.0 build's
``--version`` and 2x1 runs of bragg and microstrip at order 1 rendered by this
adapter (wave-port timer rows included). No test runs Palace.
"""

import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import adapters, stacks
from benchmarks.adapters import (
    CommandResult,
    Layout,
    Measured,
    Run,
    RunRefusedError,
    palace,
)
from benchmarks.harness import config
from benchmarks.harness.case import CaseError, Problem, load_case

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "benchmarks" / "cases"
DATA = Path(__file__).parent / "data" / "bench"
P18 = DATA / "palace-0.18"
P19 = DATA / "palace-0.19"
COMMIT = "858bc66" + "0" * 33
SITE = """
name = "test"
[slurm]
account = "A"
[slurm.qos]
debug = "d"
[cluster]
root = "{root}"
[stacks.wheels]
base_venv = "{root}/venv"
wheel_dirs = ["{root}/wheels"]
petsc_wheelhouse = "{root}/petsc"
palace_wheelhouse = "{root}/palace"
python = "{root}/python"
[data]
dir = "{root}/data"
[output]
dir = "{root}/out"
"""

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


def _stack(root: Path, version: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "site.toml").write_text(SITE.replace("{root}", str(root)))
    site = config.load_site(root / "site.toml")
    resolved = stacks.resolve("wheels", site, {"palace": version}, commit=COMMIT)
    binary = Path(resolved.code_env["palace"]) / "lib/python3.12/site-packages"
    binary = binary / "palace_solver/bin/palace-real"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.touch()
    return resolved


@pytest.fixture
def stack18(tmp_path):
    return _stack(tmp_path / "s18", "0.18.1.post5")


@pytest.fixture
def stack19(tmp_path):
    return _stack(tmp_path / "s19", "0.19.0")


def _run(case_name, solver, stack, workdir, *, layout="4x1", tolerance=None, **kw):
    case = load_case(CASES / case_name)
    return Run(
        case=case,
        solver=solver,
        tolerance=tolerance,
        layout=Layout.parse(layout),
        core_budget=Layout.parse(layout).cores,
        repeat=1,
        stack=stack,
        workdir=workdir,
        mesh=Path("/staged/mesh.msh"),
        **kw,
    )


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


def _rendered(run):
    launch = palace.ADAPTER.prepare(run)
    return launch, json.loads((run.workdir / "config.json").read_text())


# --- case frequency ranges follow Palace's GetNumSteps --------------------------


def _range(start, stop, step):
    return Problem(
        kind="driven", frequencies_hz={"start": start, "stop": stop, "step": step}
    )


def test_range_keeps_a_sample_round_off_puts_just_past_stop():
    # 0.1 Hz short of 2 GHz is 1e-7 steps of 1 MHz: past a step-relative 1e-9 slack,
    # inside Palace's 1e-9 * stop (2 Hz), so Palace keeps the 1001st sample.
    freqs = _range(1e9, 2e9 - 0.1, 1e6).frequencies()
    assert len(freqs) == 1001
    assert freqs[-1] == pytest.approx(2e9)


def test_range_counts_plain_ranges_as_palace():
    assert _range(1e9, 2e9, 0.25e9).frequencies() == (1e9, 1.25e9, 1.5e9, 1.75e9, 2e9)
    assert len(_range(1e9, 2.1e9, 0.25e9).frequencies()) == 5


def test_empty_range_is_refused():
    with pytest.raises(CaseError, match="empty"):
        _range(2e9, 1e9, 1e6).frequencies()


# --- dialects and render ---------------------------------------------------------


def test_both_dialects_exist_and_versions_map_to_them():
    assert set(palace.DIALECTS) == {"0.18", "0.19"}
    assert palace.dialect_of("0.18.1.post5").name == "0.18"
    assert palace.dialect_of("v0.19.0").name == "0.19"
    with pytest.raises(RunRefusedError, match="no Palace dialect"):
        palace.dialect_of("0.17.0")


@pytest.mark.parametrize("stack_name", ["stack18", "stack19"])
def test_rendered_direct_driven_config_carries_the_pins(stack_name, request, tmp_path):
    stack = request.getfixturevalue(stack_name)
    _, cfg = _rendered(_run("microstrip", "palace:mumps", stack, tmp_path))
    linear = cfg["Solver"]["Linear"]
    assert linear["Type"] == "MUMPS"
    assert linear["MGMaxLevels"] == 1
    assert linear["ComplexCoarseSolve"] is True
    assert linear["Tol"] == 1e-10
    assert linear["MaxIts"] == 2000
    assert linear["EstimatorTol"] == 1e-2
    assert linear["EstimatorMG"] is True
    assert cfg["Model"]["Refinement"] == {"UniformLevels": 0, "MaxIts": 0}
    assert cfg["Model"]["L0"] == 1e-6
    assert cfg["Model"]["Mesh"] == "/staged/mesh.msh"
    assert cfg["Boundaries"]["Absorbing"] == {"Attributes": [13], "Order": 1}
    assert cfg["Problem"]["OutputFormats"] == {"Paraview": False, "GridFunction": False}
    sample = cfg["Solver"]["Driven"]["Samples"]
    assert sample == [
        {"Type": "Point", "Freq": [1.0, 25.0, 50.0, 75.0, 100.0], "SaveStep": 0}
    ]
    ports = cfg["Boundaries"]["LumpedPort"]
    assert [p["Excitation"] for p in ports] == [1, 2]
    assert ports[0]["Elements"] == [{"Attributes": [8], "Direction": "+Z"}]
    assert ports[0]["R"] == 50.0
    keys = set(_keys(cfg))
    assert "AbsTol" not in keys
    assert "NSamples" not in keys
    crack = "RefineCrackElements" in cfg["Model"]
    assert crack == (stack_name == "stack18")
    if crack:
        assert cfg["Model"]["RefineCrackElements"] is False
    assert not any("Crack" in k for k in keys - {"RefineCrackElements"})


def test_rendered_pmg_uses_the_run_tolerance_and_pins_type(stack18, tmp_path):
    run = _run("microstrip", "palace:pmg", stack18, tmp_path, tolerance=1e-8)
    _, cfg = _rendered(run)
    linear = cfg["Solver"]["Linear"]
    assert linear["Type"] == "SuperLU"
    assert linear["Tol"] == 1e-8
    assert "MGMaxLevels" not in linear


def test_pmg_driven_without_a_tolerance_is_refused(stack18, tmp_path):
    with pytest.raises(RunRefusedError, match="tolerance"):
        palace.ADAPTER.prepare(_run("microstrip", "palace:pmg", stack18, tmp_path))


@pytest.mark.parametrize("kind", ["mumps", "pmg"])
def test_rendered_eigen_pins_both_tolerances(kind, stack19, tmp_path):
    run = _run("lcavity", f"palace:{kind}", stack19, tmp_path, tolerance=1e-6)
    _, cfg = _rendered(run)
    eigen = cfg["Solver"]["Eigenmode"]
    assert eigen["Target"] == pytest.approx(100.0)
    assert eigen["N"] == 1
    assert eigen["Tol"] == 1e-10
    assert cfg["Solver"]["Linear"]["Tol"] == 1e-12
    assert cfg["Boundaries"] == {"PEC": {"Attributes": [2]}}


def test_bragg_wave_vector_is_rendered_per_mesh_unit(stack18, tmp_path):
    _, cfg = _rendered(_run("bragg", "palace:pmg", stack18, tmp_path))
    periodic = cfg["Boundaries"]["Periodic"]
    assert periodic["FloquetWaveVector"] == pytest.approx([math.pi / 10, 0.0, 0.0])
    assert periodic["BoundaryPairs"] == [
        {
            "DonorAttributes": [3],
            "ReceiverAttributes": [8],
            "Translation": [10.0, 0.0, 0.0],
        }
    ]


# --- launch -------------------------------------------------------------------------


def test_argv_is_the_srun_line_around_palace_real(stack18, tmp_path):
    launch, _ = _rendered(_run("microstrip", "palace:mumps", stack18, tmp_path))
    (command,) = launch.commands
    real = str(next(Path(stack18.code_env["palace"]).rglob("palace-real")))
    assert command.argv == (
        "srun", "--exact", "-n", "4", "-c", "1", "--cpu-bind=cores",
        "--distribution=block:block", "--hint=nomultithread", "--ear=off",
        "--mpi=pmi2", real, str(tmp_path / "config.json"),
    )  # fmt: skip
    assert command.program == (real, str(tmp_path / "config.json"))
    assert command.launch == "srun"
    assert command.env == {}
    assert command.watch is not None
    assert command.watch(" Level 0 (p = 2): 315870 unknowns") == 315870


def test_one_rank_taskset_fallback_runs_the_bare_program():
    argv, launch = adapters.launched(
        ("palace-real", "c.json"), Layout(1, 16), "pmi2", "taskset"
    )
    assert (argv, launch) == (("palace-real", "c.json"), "taskset")
    argv, launch = adapters.launched(
        ("palace-real", "c.json"), Layout(2, 8), "pmi2", "taskset"
    )
    assert argv[:2] == ("srun", "--exact")
    assert launch == "srun"


def test_a_stack_without_the_binary_is_refused(stack18, tmp_path):
    shutil.rmtree(stack18.code_env["palace"])
    with pytest.raises(RunRefusedError, match="no palace-real"):
        palace.ADAPTER.prepare(_run("microstrip", "palace:mumps", stack18, tmp_path))


def test_an_unknown_palace_version_is_refused(tmp_path):
    stack = _stack(tmp_path / "s17", "0.17.0")
    with pytest.raises(RunRefusedError, match="no Palace dialect"):
        palace.ADAPTER.prepare(
            _run("microstrip", "palace:mumps", stack, tmp_path / "w")
        )


# --- supports and default layout ----------------------------------------------------


def test_supports_refuses_exclusions_and_unmapped_problems():
    case = load_case(CASES / "microstrip")
    layout = Layout(4, 4)
    assert palace.ADAPTER.supports(case, "mumps", layout) is None
    assert "excluded" in palace.ADAPTER.supports(
        replace(case, excluded=(("metal1_xy", "metal1_z"),)), "mumps", layout
    )
    abc2 = tuple(
        replace(b, settings={**b.settings, "order": 2}) if b.kind == "absorbing" else b
        for b in case.boundaries
    )
    assert "order 2" in palace.ADAPTER.supports(
        replace(case, boundaries=abc2), "mumps", layout
    )
    adaptive = replace(case, problem=replace(case.problem, sweep="adaptive"))
    assert "adaptive" in palace.ADAPTER.supports(adaptive, "mumps", layout)
    assert "no kind" in palace.ADAPTER.supports(case, "ams", layout)


def test_default_layout_is_one_rank_per_core():
    case = load_case(CASES / "lcavity")
    assert palace.ADAPTER.default_layout(case, "pmg", 16) == Layout(16, 1)


# --- parse --------------------------------------------------------------------------


def _workdir(tmp_path, fixture):
    work = tmp_path / "work"
    (work / "output").mkdir(parents=True)
    for path in fixture.iterdir():
        target = (
            work / "palace.stdout"
            if path.name == "palace.stdout"
            else work / "output" / path.name
        )
        shutil.copyfile(path, target)
    return work


def _collect(run, returncode=0, version=None):
    measured = Measured(
        results=(CommandResult(returncode, run.workdir / "palace.stdout"),),
        version=version,
    )
    return palace.ADAPTER.collect(run, measured)


def test_rank0_buckets_partition_total(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-1x4")
    outcome = _collect(_run("r10M_void", "palace:mumps", stack18, work, layout="1x4"))
    assert outcome.rank_reduction == "rank0"
    phases = outcome.phases
    assert phases["discretize_s"] is None
    assert phases["setup_s"] == pytest.approx(43.745, abs=1e-3)
    assert phases["mesh_read_s"] == pytest.approx(0.073 + 0.274, abs=2e-3)
    total = sum(v for v in phases.values() if v is not None)
    assert total == pytest.approx(outcome.work_span_s, rel=1e-12)
    assert outcome.work_span_s == pytest.approx(59.793, abs=1e-3)
    assert outcome.ranks_seen == 1
    assert outcome.threads_seen == 4
    assert outcome.ndof == 162640
    assert outcome.setups == 2
    assert outcome.solves == 2


def test_more_ranks_take_the_max_of_each_row(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    outcome = _collect(_run("r10M_void", "palace:pmg", stack18, work, tolerance=1e-8))
    assert outcome.rank_reduction == "max-of-rows"
    assert outcome.phases["setup_s"] == pytest.approx(37.239, abs=1e-3)
    assert outcome.phases["solve_s"] == pytest.approx(
        14.421 + 53.235 + 235.664, abs=3e-3
    )
    assert outcome.rank0["solve_s"] == pytest.approx(
        14.312 + 53.235 + 234.016, abs=3e-3
    )
    assert outcome.work_span_s == pytest.approx(425.047, abs=1e-3)
    assert outcome.self_peak_mem_gib == pytest.approx(1693.9375 / 1024)
    assert outcome.self_peak_rank_max_gib == pytest.approx(443.015625 / 1024)
    assert outcome.self_peak_largest_node_gib == pytest.approx(1693.9375 / 1024)
    assert outcome.checks["converged"] == "pass"
    assert outcome.iterations == [17, 17]


def test_no_table_on_several_ranks_falls_back_to_rank0(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    text = (work / "palace.stdout").read_text()
    (work / "palace.stdout").write_text(text.split("Elapsed Time Report")[0])
    outcome = _collect(_run("r10M_void", "palace:pmg", stack18, work, tolerance=1e-8))
    assert outcome.rank_reduction == "rank0"
    assert outcome.phases == outcome.rank0


@pytest.mark.parametrize("where", ["palace.json", "stdout"])
def test_an_unknown_timer_key_fails_the_parse(where, stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    if where == "palace.json":
        meta = json.loads((work / "output/palace.json").read_text())
        meta["ElapsedTime"]["Durations"]["Warp Drive"] = 1.0
        (work / "output/palace.json").write_text(json.dumps(meta))
    else:
        text = (work / "palace.stdout").read_text()
        row = "Warp Drive                       1.000       1.000       1.000\n"
        (work / "palace.stdout").write_text(text.replace("Disk IO", row + "Disk IO", 1))
    outcome = _collect(_run("r10M_void", "palace:pmg", stack18, work, tolerance=1e-8))
    assert outcome.status == "failed"
    assert outcome.failure_class == "harness"
    assert "WarpDrive" in outcome.reason or "Warp Drive" in outcome.reason


def test_wave_port_sub_timers_are_known_to_0_19_only():
    durations = {"Eigenproblem": 1.0, "WavePorts": 2.0, "Total": 4.0}
    with pytest.raises(palace.PalaceParseError, match="Eigenproblem"):
        palace.buckets(durations, palace.DIALECTS["0.18"])
    assert palace.buckets(durations, palace.DIALECTS["0.19"])["assemble_s"] == 3.0


def test_killed_run_has_null_phases_and_is_incomplete(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "killed")
    (work / "palace.stdout").write_text("")
    run = _run(
        "r10M_void", "palace:pmg", stack18, work, tolerance=1e-6, pinned_ndof=10248856
    )
    outcome = _collect(run, returncode=None)
    assert outcome.status == "failed"
    assert outcome.reason.startswith("incomplete")
    assert outcome.checks["complete"] == "fail"
    assert outcome.checks["dof"] == "pass"
    assert set(outcome.phases.values()) == {None}
    assert outcome.ndof == 10248856
    assert outcome.ranks_seen == 4


def test_port_s_of_a_run_ended_mid_sweep_keeps_its_null_cells_empty(tmp_path):
    # Palace writes NULL for excitations it has not reached yet
    path = tmp_path / "port-S.csv"
    path.write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.), |S[1][2]| (dB), "
        "arg(S[1][2]) (deg.)\n"
        " 1.0e+00, -2.0e+01, +1.0e+01, -6.0e+00, +0.0e+00\n"
        " 2.5e+01, -2.0e+01, +1.0e+01, NULL, NULL\n"
    )
    out = palace.port_s(path, ["p1", "p2"])
    assert out["s"]["p1,p2"][1] is None
    assert out["s"]["p1,p2"][0] == pytest.approx([10 ** (-6 / 20), 0.0])


def test_port_s_db_and_degrees_become_complex(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    outcome = _collect(_run("r10M_void", "palace:pmg", stack18, work, tolerance=1e-8))
    q = outcome.quantities
    assert q["frequencies_hz"] == [50e9]
    # stdout: S[2][1] = +4.337e-02-9.312e-01i, S[1][1] = +2.714e-01-5.490e-03i
    assert q["s"]["p2,p1"][0] == pytest.approx([4.337e-02, -9.312e-01], abs=1e-4)
    assert q["s"]["p1,p1"][0] == pytest.approx([2.714e-01, -5.490e-03], abs=1e-4)
    assert set(q["s"]) == {"p1,p1", "p2,p1"}  # the case's pairs only
    assert outcome.checks["complete"] == "pass"
    assert outcome.status == "ok"


def test_dof_level_line_is_the_guard():
    watch = palace.dof_watch(2)
    assert watch(" Level 1 (p = 2): 162640 unknowns") == 162640
    assert watch(" Level 1 (auxiliary) (p = 2): 37660 unknowns") is None
    assert watch(" Level 0 (p = 1): 32057 unknowns") is None
    assert watch(" H1 (p = 2): 37660, ND (p = 2): 162640") is None
    text = (P18 / "eigen-bragg/palace.stdout").read_text()
    assert palace.stdout_ndof(text, 3) == 1560


def test_dof_mismatch_fails_the_run(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    run = _run(
        "r10M_void", "palace:pmg", stack18, work, tolerance=1e-8, pinned_ndof=10248856
    )
    outcome = _collect(run)
    assert outcome.status == "failed"
    assert outcome.checks["dof"] == "fail"
    assert (
        outcome.reason == "dof_mismatch: palace 0.18.1.post5 162640 != pinned 10248856"
    )


def test_a_gmres_that_did_not_converge_fails(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "driven-4x1")
    text = (work / "palace.stdout").read_text()
    text = text.replace(
        "GMRES solver converged in 17", "GMRES solver did NOT converge in 2000", 1
    )
    (work / "palace.stdout").write_text(text)
    outcome = _collect(_run("r10M_void", "palace:pmg", stack18, work, tolerance=1e-8))
    assert outcome.checks["converged"] == "fail"
    assert outcome.status == "failed"
    assert outcome.failure_class == "solver"
    assert outcome.iterations == [2000, 17]


def test_eigen_outputs_and_inputs_match_on_bragg(stack18, tmp_path):
    work = _workdir(tmp_path, P18 / "eigen-bragg")
    outcome = _collect(_run("bragg", "palace:pmg", stack18, work, layout="1x4"))
    assert outcome.status == "ok"
    assert outcome.quantities["f_hz"][0] == pytest.approx(
        [5.185530238881e9, 0.1644672329944], rel=1e-12
    )
    assert outcome.quantities["q"][1] == pytest.approx(2.586479945140e10)
    assert outcome.quantities["error_bkwd"][0] == pytest.approx(5.505125444833e-12)
    assert outcome.checks == {
        "inputs_match": "pass",
        "converged": "pass",
        "complete": "pass",
    }
    assert outcome.ndof == 1560


def test_inputs_match_fails_on_a_mutated_case():
    case = load_case(CASES / "bragg")
    resolved = json.loads((P18 / "eigen-bragg/config_resolved.json").read_text())
    assert palace.inputs_match(case, resolved) == []
    hotter = replace(case, problem=replace(case.problem, target_hz=4.5e9))
    assert any("target" in d for d in palace.inputs_match(hotter, resolved))
    denser = replace(
        case, materials=(replace(case.materials[0], eps_r=2.1), case.materials[1])
    )
    assert any("materials" in d for d in palace.inputs_match(denser, resolved))


def test_inputs_match_sees_abc_order_and_excitations_of_another_run():
    # the killed r10M run had ABC order 2 and both ports excited; the case has neither
    case = load_case(CASES / "r10M_void")
    resolved = json.loads((P18 / "killed/config_resolved.json").read_text())
    diffs = palace.inputs_match(case, resolved)
    assert any("order" in d for d in diffs)
    assert any("excitations" in d for d in diffs)


def test_rendered_config_imports_back_equal_to_its_case(stack18, tmp_path):
    for name in ("microstrip", "r10M_void", "bragg", "lcavity"):
        run = _run(name, "palace:mumps", stack18, tmp_path / name)
        _, cfg = _rendered(run)
        assert palace.inputs_match(run.case, cfg) == [], name


# --- version --------------------------------------------------------------------------


def test_version_query_runs_the_binary_bare(stack18):
    (command,) = palace.ADAPTER.version(stack18).commands
    assert command.argv[1:] == ("--version",)
    assert command.argv[0].endswith("palace_solver/bin/palace-real")


def test_version_mismatch_is_refused(stack19, tmp_path):
    text = (P18 / "palace-version.txt").read_text()
    work = _workdir(tmp_path, P18 / "driven-4x1")
    run = _run("r10M_void", "palace:pmg", stack19, work, tolerance=1e-8)
    outcome = _collect(run, version=text)
    assert outcome.status == "refused"
    assert outcome.reason == "Palace v0.18.1 installed, stack wheels declares 0.19.0"


def test_a_matching_install_reports_the_stack_version_it_was_checked_against(
    stack18, tmp_path
):
    # the 0.18.1.post5 wheel's binary says v0.18.1: the record's version is the
    # wheel's (what the campaign pins), the binary's tag kept beside it
    text = (P18 / "palace-version.txt").read_text()
    work = _workdir(tmp_path, P18 / "driven-4x1")
    run = _run("r10M_void", "palace:mumps", stack18, work)
    outcome = _collect(run, version=text)
    assert outcome.provenance["version"] == "0.18.1.post5"
    assert outcome.provenance["binary_version"] == "v0.18.1"


def test_version_text_of_both_dialects():
    v18 = palace.parse_version((P18 / "palace-version.txt").read_text())
    assert (v18["version"], v18["schema"]) == ("v0.18.1", "1-7-0")
    assert "dependencies" not in v18
    v19 = palace.parse_version((P19 / "palace-version.txt").read_text())
    assert v19["version"] == "v0.19.0"
    assert v19["dependencies"]["MUMPS"] == "5.9.1"


def test_fixtures_stay_small():
    size = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    assert size <= 300_000


# --- 0.19.0 outputs (native build) and real resolved driven configs --------------


def _order1(run):
    return replace(run, case=replace(run.case, order=1))


def test_0_19_eigen_run_parses_with_the_wave_port_rows(stack19, tmp_path):
    work = _workdir(tmp_path, P19 / "eigen-bragg")
    outcome = _collect(_run("bragg", "palace:pmg", stack19, work, layout="2x1"))
    assert outcome.status == "ok"
    assert {"Eigenproblem", "Fieldreconstruction", "Modepostprocessing"} <= set(
        outcome.native["durations"]
    )
    assert outcome.rank_reduction == "max-of-rows"
    assert outcome.work_span_s == pytest.approx(93.310, abs=1e-3)
    assert outcome.ndof == 1560
    assert outcome.checks == {
        "inputs_match": "pass",
        "converged": "pass",
        "complete": "pass",
    }
    assert outcome.quantities["f_hz"][0][0] == pytest.approx(
        5.185530224675e9, rel=1e-12
    )


def test_0_19_eigen_run_is_refused_by_the_0_18_parse(stack18, tmp_path):
    work = _workdir(tmp_path, P19 / "eigen-bragg")
    outcome = _collect(_run("bragg", "palace:pmg", stack18, work, layout="2x1"))
    assert outcome.status == "failed"
    assert "Eigenproblem" in outcome.reason


@pytest.mark.parametrize(
    ("stack_name", "fixture"), [("stack18", P18), ("stack19", P19)]
)
def test_driven_run_of_each_version_matches_its_case(
    stack_name, fixture, request, tmp_path
):
    # microstrip at order 1, rendered by this adapter and run at 2x1
    stack = request.getfixturevalue(stack_name)
    work = _workdir(tmp_path, fixture / "driven-2x1")
    run = _order1(_run("microstrip", "palace:mumps", stack, work, layout="2x1"))
    outcome = _collect(run)
    assert outcome.status == "ok", outcome.reason
    assert outcome.checks == {
        "inputs_match": "pass",
        "converged": "pass",
        "complete": "pass",
    }
    assert outcome.ndof == 60269
    assert outcome.setups == 10  # one factorization per frequency and excitation
    assert outcome.quantities["frequencies_hz"] == pytest.approx(
        [1e9, 25e9, 50e9, 75e9, 100e9]
    )
    # stdout at 1 GHz: S[1][1] = +3.253e-02+2.596e-03i, S[2][1] = +9.663e-01-5.006e-02i
    assert outcome.quantities["s"]["p1,p1"][0] == pytest.approx(
        [3.253e-02, 2.596e-03], abs=1e-5
    )
    assert outcome.quantities["s"]["p2,p1"][0] == pytest.approx(
        [9.663e-01, -5.006e-02], abs=1e-4
    )
    text = (work / "palace.stdout").read_text()
    assert palace.stdout_ndof(text, 1) == 60269
    resolved = json.loads((work / "output/config_resolved.json").read_text())
    assert palace.inputs_match(run.case, resolved) == []
    assert palace.inputs_match(load_case(CASES / "microstrip"), resolved) == [
        "order 1 != case 2"
    ]
