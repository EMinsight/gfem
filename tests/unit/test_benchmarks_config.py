"""`benchmarks.harness.config`: site and campaign files; the suite's command line."""

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.harness import config
from benchmarks.harness.config import ConfigError

REPO = Path(__file__).resolve().parents[2]

SITE = """\
name = "cluster"

[slurm]
account = "acct"
constraint = ""

[slurm.qos]
debug = "debug_q"
long = "long_q"

[cluster]
root = "/scratch/me/gfem"

[stacks]
root = "/scratch/me/stacks"

[stacks.wheels]
base_venv = "{root}/venv"
wheel_dirs = ["{root}/wheels", "{root}/more"]
petsc_wheelhouse = "{root}/petsc"
palace_wheelhouse = "{root}/palace"
python = "{root}/py/bin/python3"

[data]
dir = "{root}/data"
stores = ["{root}/mesh-store", "/elsewhere/store"]

[ship]
host = "transfer"
mesh_store = "{root}/mesh-store"

[output]
dir = "{root}/runs"

[budget]
ceiling_core_h = 2000
prior_core_h = 38.3
"""

CAMPAIGN = """\
schema = 1
cases = ["bragg", "microstrip"]
solvers = ["gfem:direct", "gfem:bddc", "palace:pmg"]
layouts = ["best"]
core_budget = 16
repeats = 3
node_class = "standard"
anchor_tolerance = 1e-8

[jobs]
group = "case"

[wall_cap_s]
bragg = 120
microstrip = 600

[[ladder]]
cases = ["microstrip"]
solvers = ["gfem:bddc", "palace:pmg"]
tolerances = [1e-6, 1e-10]
layout = "default"

[stacks.wheels.versions]
palace = "0.19.0"

[stacks.wheels.env]
JAX_PLATFORMS = "cpu"
"""

CASES = ("bragg", "lcavity", "microstrip")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _campaign(tmp_path: Path, text: str = CAMPAIGN):
    return config.load_campaign(_write(tmp_path, "smoke.toml", text), cases=CASES)


# --- site file -------------------------------------------------------------------


def test_site_fills_root_placeholder(tmp_path):
    site = config.load_site(_write(tmp_path, "site.toml", SITE), environ={})
    assert site.name == "cluster"
    assert site.cluster_root == "/scratch/me/gfem"
    assert site.stacks_root == "/scratch/me/stacks"
    assert site.stacks["wheels"]["base_venv"] == "/scratch/me/gfem/venv"
    assert site.stacks["wheels"]["wheel_dirs"] == [
        "/scratch/me/gfem/wheels",
        "/scratch/me/gfem/more",
    ]
    assert site.data_dir == "/scratch/me/gfem/data"
    assert site.data_stores == ("/scratch/me/gfem/mesh-store", "/elsewhere/store")
    assert site.ship_mesh_store == "/scratch/me/gfem/mesh-store"
    assert site.ship_host == "transfer"
    assert site.output_dir == "/scratch/me/gfem/runs"
    assert site.account == "acct"
    assert site.qos == {"debug": "debug_q", "long": "long_q"}
    assert site.constraint == ""
    assert site.ceiling_core_h == 2000
    assert site.prior_core_h == 38.3


def test_site_from_environment_variable(tmp_path):
    path = _write(tmp_path, "site.toml", SITE)
    site = config.load_site(None, environ={"GFEM_BENCH_SITE": str(path)})
    assert site.path == path


def test_site_flag_wins_over_environment_variable(tmp_path):
    path = _write(tmp_path, "site.toml", SITE)
    site = config.load_site(path, environ={"GFEM_BENCH_SITE": "/nowhere.toml"})
    assert site.path == path


def test_site_missing_is_refused():
    with pytest.raises(ConfigError, match="GFEM_BENCH_SITE"):
        config.load_site(None, environ={})


def test_site_budget_defaults_to_the_effort_ceiling(tmp_path):
    text = SITE.split("[budget]", maxsplit=1)[0]
    site = config.load_site(_write(tmp_path, "site.toml", text), environ={})
    assert site.ceiling_core_h == 2000
    assert site.prior_core_h == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('host = "transfer"', 'host = "transfer"\nport = 22', r"\[ship\].*port"),
        ('name = "cluster"', 'name = "cluster"\nuser = "me"', "user"),
        ('[output]\ndir = "{root}/runs"', "", "output"),
        ('dir = "{root}/data"', 'dir = "{home}/data"', r"\{home\}"),
        ('root = "/scratch/me/gfem"', "root = 3", "cluster.root"),
    ],
)
def test_site_refuses_unknown_missing_or_wrong_keys(tmp_path, old, new, message):
    text = SITE.replace(old, new, 1)
    with pytest.raises(ConfigError, match=message):
        config.load_site(_write(tmp_path, "site.toml", text), environ={})


@pytest.mark.parametrize(
    "name", ["OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"]
)
def test_site_env_table_refuses_affinity_variables(tmp_path, name):
    text = SITE + f'\n[env]\n{name} = "cores"\n'
    with pytest.raises(ConfigError, match=name):
        config.load_site(_write(tmp_path, "site.toml", text), environ={})


def test_example_site_file_loads():
    site = config.load_site(REPO / "benchmarks" / "site.example.toml", environ={})
    assert site.stacks["wheels"]["base_venv"].startswith(site.cluster_root)
    assert site.ceiling_core_h == 2000


# --- campaign file ---------------------------------------------------------------


def test_campaign_parses(tmp_path):
    campaign = _campaign(tmp_path)
    assert campaign.name == "smoke"
    assert campaign.cases == ("bragg", "microstrip")
    assert campaign.solvers == ("gfem:direct", "gfem:bddc", "palace:pmg")
    assert campaign.stacks == ("wheels",)
    assert campaign.layouts == ("best",)
    assert campaign.core_budget == 16
    assert campaign.repeats == 3
    assert campaign.node_class == "standard"
    assert campaign.anchor_tolerance == 1e-8
    assert campaign.group == "case"
    assert campaign.wall_cap_s == {"bragg": 120, "microstrip": 600}
    (ladder,) = campaign.ladder
    assert ladder.cases == ("microstrip",)
    assert ladder.solvers == ("gfem:bddc", "palace:pmg")
    assert ladder.tolerances == (1e-6, 1e-10)
    assert ladder.layout == "default"
    assert campaign.stack_versions == {"wheels": {"palace": "0.19.0"}}
    assert campaign.stack_env == {"wheels": {"JAX_PLATFORMS": "cpu"}}


def test_campaign_defaults(tmp_path):
    text = CAMPAIGN.split("[jobs]", maxsplit=1)[0] + (
        "[wall_cap_s]\nbragg = 1\nmicrostrip = 1\n"
        "[stacks.wheels]\n[stacks.native-oss]\n"
    )
    campaign = _campaign(tmp_path, text)
    assert campaign.stacks == ("wheels", "native-oss")
    assert campaign.group == "case"
    assert campaign.ladder == ()
    assert campaign.stack_versions == {}
    assert campaign.stack_env == {}


def test_campaign_without_stacks_is_refused(tmp_path):
    text = CAMPAIGN.split("[[ladder]]", maxsplit=1)[0]
    with pytest.raises(ConfigError, match="stacks"):
        _campaign(tmp_path, text)


def test_campaign_explicit_layout_within_budget(tmp_path):
    campaign = _campaign(tmp_path, CAMPAIGN.replace('["best"]', '["default", "4x4"]'))
    assert campaign.layouts == ("default", "4x4")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema = 1", "schema = 2", "schema"),
        ("repeats = 3", "repeats = 3\nthreads = 4", "threads"),
        ('group = "case"', 'group = "case"\nsize = 2', r"\[jobs\].*size"),
        ('["bragg", "microstrip"]', '["bragg", "nope"]', "nope"),
        ('"palace:pmg"]', '"palace:amg"]', "palace:amg"),
        ("[stacks.wheels.versions]", "[stacks.conda.versions]", "conda"),
        ('["best"]', '["5x4"]', "5x4"),
        ('["best"]', '["fast"]', "fast"),
        ('"standard"', '"gpu"', "gpu"),
        ('group = "case"', 'group = "node"', "node"),
        ("microstrip = 600", "", "microstrip"),
        ("core_budget = 16", "core_budget = 0", "core_budget"),
        ("repeats = 3", 'repeats = "3"', "repeats"),
        ('palace = "0.19.0"', 'mumps = "5"', "mumps"),
        ("[stacks.wheels.env]", "[stacks.wheels.pins]", "pins"),
        ('layout = "default"', 'layout = "best"', "best"),
        ("tolerances = [1e-6, 1e-10]", "tolerances = []", "tolerances"),
    ],
)
def test_campaign_refuses_unknown_missing_or_wrong_values(tmp_path, old, new, message):
    text = CAMPAIGN.replace(old, new, 1)
    with pytest.raises(ConfigError, match=message):
        _campaign(tmp_path, text)


@pytest.mark.parametrize(
    "name", ["OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"]
)
def test_campaign_env_table_refuses_affinity_variables(tmp_path, name):
    text = CAMPAIGN + f'{name} = "close"\n'
    with pytest.raises(ConfigError, match=name):
        _campaign(tmp_path, text)


def test_affinity_refusal_is_reusable_for_any_env_table():
    config.check_env({"OMP_NUM_THREADS": "4"}, where="stack wheels [gfem]")
    with pytest.raises(ConfigError, match=r"stack wheels \[gfem\].*OMP_PROC_BIND"):
        config.check_env({"OMP_PROC_BIND": "close"}, where="stack wheels [gfem]")


def test_cases_registry_reads_case_directories(tmp_path):
    (tmp_path / "bragg").mkdir()
    (tmp_path / "bragg" / "case.toml").write_text("format = 1\n")
    (tmp_path / "notes").mkdir()
    assert config.known_cases(tmp_path) == ("bragg",)


# --- command line ----------------------------------------------------------------


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for name in (
        "plan",
        "submit",
        "run",
        "step",
        "status",
        "close",
        "reference",
        "ship",
        "fetch",
        "report",
        "case",
    ):
        assert name in out


def test_plan_list_prints_axis_values(capsys):
    assert main(["plan", "--list"]) == 0
    out = capsys.readouterr().out
    for value in config.SOLVERS + config.STACKS + ("default", "best", "RxT"):
        assert value in out


@pytest.mark.parametrize(
    "argv",
    [
        ["report", "a", "b", "--out", "dir"],
    ],
)
def test_unimplemented_subcommands_exit_2(argv, capsys):
    assert main(argv) == 2
    assert "not yet" in capsys.readouterr().err


def test_plan_refuses_a_bad_campaign_with_status_2(tmp_path, capsys):
    campaign = _write(tmp_path, "smoke.toml", CAMPAIGN.replace("schema = 1", "x = 1"))
    site = _write(tmp_path, "site.toml", SITE)
    argv = ["plan", str(campaign), "--site", str(site), "--out", str(tmp_path / "r")]
    assert main(argv) == 2
    assert "schema" in capsys.readouterr().err


def test_plan_needs_campaign_and_out_without_list():
    with pytest.raises(SystemExit) as stop:
        main(["plan"])
    assert stop.value.code == 2


def test_module_runs_from_repo_root():
    out = subprocess.run(
        [sys.executable, "-m", "benchmarks", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "plan" in out.stdout
