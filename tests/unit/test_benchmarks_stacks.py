"""`benchmarks.stacks`: software stacks resolved against a site file."""

import json
import sys
from pathlib import Path

import pytest

from benchmarks import stacks
from benchmarks.harness import config
from benchmarks.harness.config import ConfigError

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_SITE = REPO / "benchmarks" / "site.example.toml"
COMMIT = "858bc66" + "0" * 33

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


@pytest.fixture
def site():
    return config.load_site(EXAMPLE_SITE)


def test_committed_descriptions_load():
    assert stacks.names() == ("native-intel", "native-oss", "wheels")
    wheels = stacks.load("wheels")
    assert wheels.kind == "wheels"
    assert wheels.versions == {
        "palace": "0.19.0",
        "ngsolve": "6.2.2608",
        "petsc": "3.26.0",
    }
    palace = wheels.codes["palace"]
    assert (palace.launcher, palace.mpi, palace.entry) == (
        "srun",
        "pmi2",
        "palace-real",
    )
    assert stacks.load("native-oss").codes["gfem"].packages == ("ngsolve", "petsc")


def test_registry_matches_config():
    assert stacks.names() == tuple(sorted(config.STACKS))


def test_resolve_wheels_with_example_site_and_override(site):
    resolved = stacks.resolve(
        "wheels", site, versions={"palace": "0.18.1.post5"}, commit=COMMIT
    )
    assert resolved.versions["palace"] == "0.18.1.post5"
    assert resolved.versions["ngsolve"] == "6.2.2608"
    assert (
        resolved.inputs["palace_wheelhouse"]
        == "/path/to/cluster/root/wheelhouse-palace"
    )
    fp8 = resolved.fingerprint[:8]
    assert resolved.env_dir == f"/path/to/cluster/root/envs/bench-858bc66-wheels-{fp8}"
    assert (
        resolved.code_env["palace"] == "/path/to/cluster/root/envs/palace-0.18.1.post5"
    )
    assert resolved.code_env["gfem"] == resolved.env_dir


def test_resolve_native_prefixes(site):
    resolved = stacks.resolve("native-oss", site, commit=COMMIT)
    assert resolved.prefixes == {
        "ngsolve": "/path/to/stacks/native-oss/ngsolve-6.2.2608",
        "petsc": "/path/to/stacks/native-oss/petsc-3.26.0",
        "palace": "/path/to/stacks/native-oss/palace-0.19.0",
    }


def test_fingerprint_changes_with_an_override(site):
    plain = stacks.resolve("wheels", site, commit=COMMIT)
    again = stacks.resolve("wheels", site, commit=COMMIT)
    other = stacks.resolve(
        "wheels", site, versions={"palace": "0.18.1.post5"}, commit=COMMIT
    )
    assert len(plain.fingerprint) == 64
    assert plain.fingerprint == again.fingerprint
    assert plain.fingerprint != other.fingerprint
    assert plain.env_dir != other.env_dir


def test_fingerprint_ignores_the_commit(site):
    a = stacks.resolve("wheels", site, commit=COMMIT)
    b = stacks.resolve("wheels", site, commit="1" * 40)
    assert a.fingerprint == b.fingerprint
    assert a.env_dir != b.env_dir


def test_resolve_refuses_unknown_stack_and_version_key(site):
    with pytest.raises(ConfigError, match="unknown stack"):
        stacks.resolve("spack", site, commit=COMMIT)
    with pytest.raises(ConfigError, match="mumps"):
        stacks.resolve("wheels", site, versions={"mumps": "5"}, commit=COMMIT)


def test_wheels_without_its_site_table_resolves_but_never_prepares(tmp_path):
    text = EXAMPLE_SITE.read_text()
    cut = text.split("[stacks.wheels]")[0] + "[data]" + text.split("[data]")[1]
    path = tmp_path / "site.toml"
    path.write_text(cut)
    resolved = stacks.resolve("wheels", config.load_site(path), commit=COMMIT)
    assert resolved.inputs == {}
    with pytest.raises(ConfigError, match=r"\[stacks.wheels\].*run --local"):
        stacks.prepare(resolved, wheel=tmp_path / "gfem.whl")


UNSET = (
    "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NGS_NUM_THREADS "
    "OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY KMP_AFFINITY"
)


def test_wheels_gfem_script_order(site):
    resolved = stacks.resolve("wheels", site, env={"UCX_TLS": "self,sm"}, commit=COMMIT)
    script = stacks.render(resolved, "gfem", threads=16)
    assert script.splitlines() == [
        "module purge >/dev/null 2>&1",
        "module load openmpi/4.1.6-gcc",
        f'export PATH={resolved.env_dir}/bin:"$PATH"',
        UNSET,
        "export JAX_PLATFORMS=cpu",
        "export JAX_ENABLE_X64=1",
        "export OMPI_MCA_opal_warn_on_missing_libcuda=0",
        "export UCX_TLS=self,sm",
        "export OMP_NUM_THREADS=16",
        "export MKL_NUM_THREADS=16",
        "export NGS_NUM_THREADS=16",
        "export OPENBLAS_NUM_THREADS=16",
        "export OMP_PLACES=cores OMP_PROC_BIND=close",
    ]
    assert "LD_LIBRARY_PATH" not in script


def _exports(script):
    return dict(
        line.removeprefix("export ").split("=", 1)
        for line in script.splitlines()
        if line.startswith("export ") and "_THREADS=" in line
    )


@pytest.mark.parametrize(
    ("threads", "gfem", "palace"),
    [
        (
            16,
            {"OMP": "16", "MKL": "16", "NGS": "16", "OPENBLAS": "16"},
            {"OMP": "16", "OPENBLAS": "1"},
        ),
        (
            1,
            {"OMP": "1", "MKL": "1", "NGS": "1", "OPENBLAS": "1"},
            {"OMP": "1", "OPENBLAS": "1"},
        ),
        (
            112,
            {"OMP": "112", "MKL": "112", "NGS": "112", "OPENBLAS": "64"},
            {"OMP": "112", "OPENBLAS": "1"},
        ),
    ],
    ids=["1x16", "16x1", "1x112"],
)
def test_wheels_thread_exports(site, threads, gfem, palace):
    resolved = stacks.resolve("wheels", site, commit=COMMIT)
    for code, want in (("gfem", gfem), ("palace", palace)):
        got = _exports(stacks.render(resolved, code, threads=threads))
        assert got == {f"{k}_NUM_THREADS": v for k, v in want.items()}


def test_wheels_palace_script_uses_its_version_env(site):
    resolved = stacks.resolve("wheels", site, commit=COMMIT)
    lines = stacks.render(resolved, "palace", threads=4).splitlines()
    assert lines[:3] == [
        "module purge >/dev/null 2>&1",
        'export PATH=/path/to/cluster/root/envs/palace-0.19.0/bin:"$PATH"',
        UNSET,
    ]


def test_native_intel_script_sources_activate_and_sets_mkl(site):
    resolved = stacks.resolve("native-intel", site, commit=COMMIT)
    lines = stacks.render(resolved, "palace", threads=8).splitlines()
    assert lines == [
        "module purge >/dev/null 2>&1",
        ". /path/to/stacks/native-intel/palace-0.19.0/activate.sh",
        UNSET,
        "export OMP_NUM_THREADS=8",
        "export OPENBLAS_NUM_THREADS=8",
        "export MKL_NUM_THREADS=8",
        "export OMP_PLACES=cores OMP_PROC_BIND=close",
    ]
    gfem = stacks.render(resolved, "gfem").splitlines()
    assert gfem[1:3] == [
        ". /path/to/stacks/native-intel/ngsolve-6.2.2608/activate.sh",
        ". /path/to/stacks/native-intel/petsc-3.26.0/activate.sh",
    ]
    assert gfem[-1] == "export JAX_ENABLE_X64=1"  # static part: no thread exports


def test_campaign_env_refuses_affinity(site):
    with pytest.raises(ConfigError, match="OMP_PROC_BIND"):
        stacks.resolve("wheels", site, env={"OMP_PROC_BIND": "spread"}, commit=COMMIT)


def _description(tmp_path, text):
    (tmp_path / "odd.toml").write_text(text)
    return tmp_path


ODD = """\
schema = 1
name = "odd"
kind = "wheels"
[versions]
palace = "1"
ngsolve = "2"
petsc = "3"
[gfem]
mpi = "pmix"
{gfem}
[gfem.threads]
OMP_NUM_THREADS = "threads"
{threads}
[palace]
mpi = "pmi2"
launcher = "srun"
entry = "palace-real"
[palace.threads]
OMP_NUM_THREADS = "threads"
"""


@pytest.mark.parametrize(
    ("gfem", "threads", "match"),
    [
        ('env = { OMP_PLACES = "cores" }', "", "OMP_PLACES"),
        ('env = { OMP_NUM_THREADS = "4" }', "", "thread table"),
        ("", 'OMP_PROC_BIND = "threads"', "OMP_PROC_BIND"),
        ("", 'MKL_NUM_THREADS = "many"', "positive count"),
        ('modules = ["a"]\nprefix = "/data/x"', "", "prefix"),
    ],
)
def test_description_refusals(tmp_path, gfem, threads, match):
    stacks_dir = _description(tmp_path, ODD.format(gfem=gfem, threads=threads))
    with pytest.raises(ConfigError, match=match):
        stacks.load("odd", stacks_dir)


# --- native refusal on a temporary prefix tree ------------------------------------


def _native_site(tmp_path):
    text = (
        EXAMPLE_SITE.read_text()
        .replace('root = "/path/to/stacks"', f'root = "{tmp_path / "stacks"}"')
        .replace('root = "/path/to/cluster/root"', f'root = "{tmp_path / "cluster"}"')
    )
    path = tmp_path / "site.toml"
    path.write_text(text)
    return config.load_site(path)


def _finish(resolved, toolchain=None):
    for package, prefix in resolved.prefixes.items():
        Path(prefix).mkdir(parents=True, exist_ok=True)
        (Path(prefix) / "activate.sh").write_text("")
        sha = (toolchain or {}).get(package, "a" * 64)
        (Path(prefix) / "provenance.json").write_text(
            json.dumps({"toolchain": {"sha256": sha}, "package": package})
        )


def test_native_refuses_without_provenance(tmp_path):
    resolved = stacks.resolve("native-oss", _native_site(tmp_path), commit=COMMIT)
    reason = stacks.refusal(resolved)
    assert (
        reason == "native-oss/ngsolve-6.2.2608: no provenance.json (unfinished prefix)"
    )
    with pytest.raises(stacks.StackRefusedError, match=r"no provenance\.json"):
        stacks.prepare(resolved)
    _finish(resolved)
    (Path(resolved.prefixes["palace"]) / "provenance.json").unlink()
    assert stacks.refusal(resolved) == (
        "native-oss/palace-0.19.0: no provenance.json (unfinished prefix)"
    )


def test_native_refuses_differing_toolchains(tmp_path):
    resolved = stacks.resolve("native-oss", _native_site(tmp_path), commit=COMMIT)
    _finish(resolved, {"palace": "b" * 64})
    assert stacks.refusal(resolved) == (
        "native-oss: toolchain hashes differ "
        f"(ngsolve {'a' * 12}, petsc {'a' * 12}, palace {'b' * 12})"
    )


def test_native_prepares_when_finished(tmp_path):
    resolved = stacks.resolve("native-oss", _native_site(tmp_path), commit=COMMIT)
    _finish(resolved)
    assert stacks.refusal(resolved) is None
    env_dir = stacks.prepare(resolved)
    assert env_dir == Path(resolved.env_dir)
    assert (env_dir / "env-palace.sh").read_text() == stacks.render(resolved, "palace")
    prov = stacks.provenance(resolved)
    assert prov["packages"]["palace"]["content"]["package"] == "palace"
    assert len(prov["packages"]["palace"]["sha256"]) == 64


# --- records ------------------------------------------------------------------------


def test_record_block_is_scrubbed(site):
    resolved = stacks.resolve("wheels", site, commit=COMMIT)
    script = stacks.render(resolved, "gfem", threads=4)
    block = stacks.record(
        resolved,
        env_script=script,
        modules="openmpi/4.1.6-gcc\n",
        ldd="libopenblas.so => /path/to/cluster/root/venv/lib/libopenblas.so",
    )
    text = json.dumps(block)
    assert "/path/to" not in text
    env_dir = f"{{cluster.root}}/envs/bench-858bc66-wheels-{resolved.fingerprint[:8]}"
    assert block["env"].splitlines()[2] == f'export PATH={env_dir}/bin:"$PATH"'
    assert block["ldd"].endswith("{stacks.wheels.base_venv}/lib/libopenblas.so")
    assert block["description"]["inputs"]["wheel_dirs"] == [
        "{stacks.wheels.wheel_dirs[0]}"
    ]
    assert (block["name"], block["kind"], block["fingerprint"]) == (
        "wheels",
        "wheels",
        resolved.fingerprint,
    )
    assert block["versions"]["palace"] == "0.19.0"


# --- wheels prepare against a fake site ----------------------------------------------

FAKE_PYTHON = """\
#!/bin/bash
# Stands in for python: records pip calls, makes venvs, prints a freeze.
here=$(cd "$(dirname "$0")/.." && pwd)
if [ "$1 $2" = "-m venv" ]; then
  mkdir -p "$3/bin" && cp "$0" "$3/bin/python"
  printf '#!/bin/bash\\necho "Palace version: 0.19.0"\\n' > "$3/bin/palace"
  chmod +x "$3/bin/palace"
  echo "palace-solver==0.19.0" > "$3/freeze.txt"
elif [ "$1 $2 $3" = "-m pip freeze" ]; then
  cat "$here/freeze.txt"
else
  echo "$*" >> "$here/pip.log"
fi
"""


def _wheels_site(tmp_path, ngsolve="6.2.2608"):
    root = tmp_path / "cluster"
    base = root / "venv" / "bin"
    base.mkdir(parents=True)
    (base / "python").write_text(FAKE_PYTHON)
    (base / "python").chmod(0o755)
    (root / "venv" / "freeze.txt").write_text(
        f"gfem==0.1.0\nngsolve=={ngsolve}\npetsc==3.26.0\nmpi4py==4.1.2\n"
    )
    for d in ("wheels", "wheelhouse-petsc", "wheelhouse-palace"):
        (root / d).mkdir()
    (root / "wheelhouse-petsc" / "petsc-3.26.0-py3-none-linux_x86_64.whl").write_text(
        "p"
    )
    (root / "wheelhouse-palace" / "palace_solver-0.19.0-py3-none-x.whl").write_text("w")
    (root / "python" / "bin").mkdir(parents=True)
    (root / "python" / "bin" / "python3").write_text(FAKE_PYTHON)
    (root / "python" / "bin" / "python3").chmod(0o755)
    wheel = tmp_path / "gfem-0.1.0-py3-none-any.whl"
    wheel.write_text("gfem")
    text = EXAMPLE_SITE.read_text().replace("/path/to/cluster/root", str(root))
    path = tmp_path / "site.toml"
    path.write_text(text)
    return config.load_site(path), wheel


def test_wheels_prepare_builds_once(tmp_path):
    site, wheel = _wheels_site(tmp_path)
    resolved = stacks.resolve("wheels", site, commit=COMMIT)
    env = stacks.prepare(resolved, wheel=wheel)
    assert env == Path(resolved.env_dir)
    log = (env / "pip.log").read_text().splitlines()
    assert log[0] == "-m pip uninstall -y -q gfem"
    assert log[-1].endswith("mpi4py==4.1.2")
    manifest = (env / "manifest.txt").read_text()
    assert manifest.startswith(f"commit {COMMIT}\n")
    assert "ngsolve==6.2.2608" in manifest
    assert (env / "env-gfem.sh").read_text() == stacks.render(resolved, "gfem")
    palace = Path(resolved.code_env["palace"])
    assert "palace_solver-0.19.0-py3-none-x.whl" in (palace / "pip.log").read_text()

    stacks.prepare(resolved, wheel=wheel)
    assert (env / "pip.log").read_text().splitlines() == log

    prov = stacks.provenance(resolved)
    assert prov["palace_version"] == "Palace version: 0.19.0"
    assert len(prov["palace_wheel_sha256"]) == 64
    assert prov["gfem_manifest"]["text"] == manifest


def test_wheels_prepare_refuses_a_version_mismatch(tmp_path):
    site, wheel = _wheels_site(tmp_path, ngsolve="6.2.2601")
    resolved = stacks.resolve("wheels", site, commit=COMMIT)
    with pytest.raises(
        stacks.StackRefusedError, match=r"ngsolve 6\.2\.2601 .* pins 6\.2\.2608"
    ):
        stacks.prepare(resolved, wheel=wheel)
    assert not Path(resolved.env_dir).exists()


def test_wheels_prepare_refuses_a_missing_palace_wheel(tmp_path):
    site, wheel = _wheels_site(tmp_path)
    resolved = stacks.resolve(
        "wheels", site, versions={"palace": "0.18.1.post5"}, commit=COMMIT
    )
    with pytest.raises(
        stacks.StackRefusedError, match=r"no palace_solver-0.18.1.post5-\*.whl"
    ):
        stacks.prepare(resolved, wheel=wheel)


def test_wheels_prepare_of_a_gfem_only_campaign_needs_no_palace_wheel(tmp_path):
    site, wheel = _wheels_site(tmp_path)
    resolved = stacks.resolve(
        "wheels", site, versions={"palace": "0.18.1.post5"}, commit=COMMIT
    )
    env = stacks.prepare(resolved, wheel=wheel, codes=("gfem",))
    assert (env / "manifest.txt").is_file()
    assert not Path(resolved.code_env["palace"]).exists()
    prov = stacks.provenance(resolved)
    assert "gfem_manifest" in prov
    assert "palace_manifest" not in prov
    assert "palace_version" not in prov


def test_frozen_versions_read_wheels_installed_from_files():
    from benchmarks.stacks import wheels

    freeze = (
        "ngsolve==6.2.2608\n"
        "petsc @ file:///w/wheelhouse-mpi/petsc-3.26.0-py3-none-linux_x86_64.whl"
        "#sha256=830a\n"
        "petsc4py @ file:///w/petsc4py-3.26.0-cp312-cp312-linux_x86_64.whl\n"
    )
    assert wheels.frozen_versions(freeze) == {
        "ngsolve": "6.2.2608",
        "petsc": "3.26.0",
        "petsc4py": "3.26.0",
    }
