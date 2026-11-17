"""`tools/stacks/`: provenance files of native software stacks, and the scripts."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: The stack scripts run on Linux clusters: POSIX paths, bash.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stack scripts are Linux only"
)

STACKS = Path(__file__).parents[2] / "tools" / "stacks"
STACK_SH = STACKS / "stack.sh"

_spec = importlib.util.spec_from_file_location("provenance", STACKS / "provenance.py")
assert _spec is not None
assert _spec.loader is not None
provenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance)


def _record(package: str = "ngsolve", toolchain: str = "a" * 64) -> dict:
    """The fragments a build hands to the writer: everything but the stamp."""
    return {
        "package": package,
        "version": "1.0",
        "stack": "native-oss",
        "source": {"url": "https://example.org/x.git", "tag": "v1.0", "commit": "c"},
        "toolchain": {"sha256": toolchain, "modules": ["gcc/12.3.0"]},
        "configure": {"options": ["-DX=ON"]},
        "recipe": {"commit": "r", "sha256": "b" * 64, "dirty": False},
        "build": {"job_id": "1", "host": "gs01r1b01"},
        "depends": {},
        "smoke": {"command": "x --version", "output": "x 1.0"},
    }


def _prefix(root: Path, package: str, toolchain: str, **extra: object) -> Path:
    prefix = root / "native-oss" / f"{package}-1.0"
    prefix.mkdir(parents=True)
    provenance.write(prefix, {**_record(package, toolchain), **extra})
    return prefix


def _stack_sh(*args: str, root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "STACKS_ROOT": str(root)}
    return subprocess.run(
        ["bash", str(STACK_SH), *args],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_prefix_is_complete_once_its_provenance_file_is_written(tmp_path):
    prefix = tmp_path / "native-oss" / "ngsolve-1.0"
    prefix.mkdir(parents=True)
    assert provenance.read(prefix) is None

    provenance.write(prefix, _record())

    record = provenance.read(prefix)
    assert record is not None
    assert record["schema"] == 1
    assert record["package"] == "ngsolve"
    assert record["prefix"] == str(prefix)
    assert record["complete_utc"].endswith("Z")


def test_a_record_missing_a_schema_field_leaves_the_prefix_incomplete(tmp_path):
    record = _record()
    del record["toolchain"]

    with pytest.raises(ValueError, match="toolchain"):
        provenance.write(tmp_path, record)

    assert provenance.read(tmp_path) is None


def test_a_record_without_a_toolchain_hash_is_refused(tmp_path):
    record = _record()
    del record["toolchain"]["sha256"]

    with pytest.raises(ValueError, match=r"toolchain\.sha256"):
        provenance.write(tmp_path, record)


def test_packages_of_one_stack_built_on_one_toolchain_agree(tmp_path):
    _prefix(tmp_path, "ngsolve", "a" * 64)
    _prefix(tmp_path, "petsc", "a" * 64)

    report = provenance.check_stack(tmp_path / "native-oss")

    assert report.errors == []


def test_packages_of_one_stack_built_on_two_toolchains_fail_the_check(tmp_path):
    _prefix(tmp_path, "ngsolve", "a" * 64)
    _prefix(tmp_path, "petsc", "c" * 64)

    report = provenance.check_stack(tmp_path / "native-oss")

    assert len(report.errors) == 1
    assert "ngsolve-1.0" in report.errors[0]
    assert "petsc-1.0" in report.errors[0]


def test_an_unfinished_prefix_is_reported_but_not_compared(tmp_path):
    _prefix(tmp_path, "ngsolve", "a" * 64)
    (tmp_path / "native-oss" / "petsc-1.0" / "lib").mkdir(parents=True)

    report = provenance.check_stack(tmp_path / "native-oss")

    assert report.errors == []
    assert any("petsc-1.0" in note and "incomplete" in note for note in report.notes)


def _mumps_palace() -> dict:
    """Palace's MUMPS block as ticket 05's table describes it."""
    return {
        "version": "5.9.1",
        "tarball_sha256": "659c9b57" + "0" * 56,
        "intsize": 32,
        "gemmt": False,
        "blas": {"vendor": "openblas", "version": "0.3.34", "threading": "openmp"},
        "scalapack": "2.2.3",
        "compiler": "GNU Fortran (GCC) 12.3.0",
        "fflags": "-O3 -march=sapphirerapids",
        "arithmetics": ["d"],
        "openmp": True,
        "orderings": ["amd", "metis", "parmetis", "pord"],
        "metis": "p13",
    }


def _mumps_petsc() -> dict:
    """gfem's PETSc MUMPS block as ticket 05's table describes it."""
    return {
        **_mumps_palace(),
        "blas": {"vendor": "openblas", "version": "0.3.34", "threading": "pthreads"},
        "arithmetics": ["s", "d", "c", "z"],
        "openmp": False,
        "orderings": ["amd", "metis", "pord", "ptscotch"],
        "metis": "p12",
    }


def _mumps_stack(root: Path, palace: dict, petsc: dict) -> Path:
    _prefix(root, "palace", "a" * 64, mumps=palace)
    _prefix(root, "petsc", "a" * 64, mumps=petsc)
    return root / "native-oss"


def test_the_two_mumps_builds_differ_only_where_ticket_05_allows(tmp_path):
    report = provenance.check_stack(
        _mumps_stack(tmp_path, _mumps_palace(), _mumps_petsc())
    )

    assert report.errors == []
    allowed = "\n".join(note for note in report.notes if "allowed" in note)
    for field in ("arithmetics", "openmp", "orderings", "metis", "blas.threading"):
        assert f"mumps {field}" in allowed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "5.8.1"),
        ("tarball_sha256", "f" * 64),
        ("scalapack", "2.2.2"),
        ("compiler", "GNU Fortran (GCC) 14.2.0"),
        ("fflags", "-O2"),
        ("blas", {"vendor": "mkl", "version": "2025.2", "threading": "gnu"}),
    ],
)
def test_mumps_builds_that_differ_in_a_must_match_field_fail(tmp_path, field, value):
    petsc = {**_mumps_petsc(), field: value}

    report = provenance.check_stack(_mumps_stack(tmp_path, _mumps_palace(), petsc))

    assert any(f"mumps {field}" in error for error in report.errors)


def test_mumps_with_64_bit_integers_fails_even_in_both_builds(tmp_path):
    report = provenance.check_stack(
        _mumps_stack(
            tmp_path,
            {**_mumps_palace(), "intsize": 64},
            {**_mumps_petsc(), "intsize": 64},
        )
    )

    assert any("mumps intsize" in error for error in report.errors)


def test_a_mumps_build_with_gemmt_fails(tmp_path):
    report = provenance.check_stack(
        _mumps_stack(tmp_path, {**_mumps_palace(), "gemmt": True}, _mumps_petsc())
    )

    assert any("gemmt" in error and "palace-1.0" in error for error in report.errors)


def test_a_mumps_build_without_metis_fails(tmp_path):
    palace = {**_mumps_palace(), "orderings": ["amd", "parmetis", "pord"]}

    report = provenance.check_stack(_mumps_stack(tmp_path, palace, _mumps_petsc()))

    assert any("metis" in error and "palace-1.0" in error for error in report.errors)


def test_parmetis_in_gfems_petsc_fails(tmp_path):
    petsc = {**_mumps_petsc(), "orderings": ["metis", "parmetis", "ptscotch"]}

    report = provenance.check_stack(_mumps_stack(tmp_path, _mumps_palace(), petsc))

    assert any("parmetis" in error and "petsc-1.0" in error for error in report.errors)


def test_a_mumps_block_missing_a_field_fails(tmp_path):
    petsc = _mumps_petsc()
    del petsc["intsize"]

    report = provenance.check_stack(_mumps_stack(tmp_path, _mumps_palace(), petsc))

    assert any("intsize" in error and "petsc-1.0" in error for error in report.errors)


def test_an_unpinned_tarball_is_trusted_on_first_fetch_and_checked_after(tmp_path):
    tarball = tmp_path / "parmetis-p9.tar.gz"
    tarball.write_bytes(b"release")
    sums = tmp_path / "SHA256SUMS"

    assert provenance.tofu(sums, tarball) == "recorded"
    assert provenance.tofu(sums, tarball) == "verified"
    # The record is sha256sum's format: coreutils can check it too.
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],  # noqa: S607
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr


def test_a_tarball_that_changed_since_its_first_fetch_is_refused(tmp_path):
    tarball = tmp_path / "parmetis-p9.tar.gz"
    tarball.write_bytes(b"release")
    sums = tmp_path / "SHA256SUMS"
    provenance.tofu(sums, tarball)
    tarball.write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"parmetis-p9\.tar\.gz"):
        provenance.tofu(sums, tarball)


def test_a_build_finds_its_prefix_absent_unfinished_matching_or_different(tmp_path):
    prefix = tmp_path / "native-oss" / "ngsolve-1.0"
    recipe, toolchain = "b" * 64, "a" * 64
    assert provenance.state(prefix, recipe, toolchain) == "absent"

    prefix.mkdir(parents=True)
    assert provenance.state(prefix, recipe, toolchain) == "incomplete"

    provenance.write(prefix, _record())
    assert provenance.state(prefix, recipe, toolchain) == "complete"
    assert provenance.state(prefix, "d" * 64, toolchain) == "differs"
    assert provenance.state(prefix, recipe, "d" * 64) == "differs"


def test_the_recipe_hash_follows_the_content_of_the_shipped_tree(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    for tree in (one, two):
        (tree / "packages").mkdir(parents=True)
        (tree / "stack.sh").write_text("echo stack\n")
        (tree / "packages" / "petsc.sh").write_text("DEPENDS=()\n")
    assert provenance.tree_sha256(one) == provenance.tree_sha256(two)

    (two / "packages" / "petsc.sh").write_text("DEPENDS=(openblas)\n")
    assert provenance.tree_sha256(one) != provenance.tree_sha256(two)


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STACKS / "provenance.py"), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_a_build_writes_its_provenance_from_one_json_file_per_field(tmp_path):
    meta, prefix = tmp_path / "meta", tmp_path / "native-oss" / "ngsolve-1.0"
    meta.mkdir()
    prefix.mkdir(parents=True)
    for field, value in _record().items():
        (meta / f"{field}.json").write_text(json.dumps(value))

    result = _cli("write", str(prefix), str(meta))

    assert result.returncode == 0, result.stderr
    assert json.loads((prefix / "provenance.json").read_text())["smoke"] == {
        "command": "x --version",
        "output": "x 1.0",
    }


def test_the_toolchain_block_names_the_frozen_toolchain_files_hash(tmp_path):
    toolchain = tmp_path / "toolchain.sh"
    toolchain.write_text("module load gcc/12.3.0\n")
    modules = tmp_path / "modules.txt"
    modules.write_text("gcc/12.3.0\nopenmpi/4.1.6-gcc\n")
    env = {**os.environ, "CC": "cc", "CFLAGS": "-O3 -march=sapphirerapids"}
    expected = subprocess.run(
        ["sha256sum", str(toolchain)],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]

    result = _cli("toolchain", str(toolchain), str(modules), env=env)

    assert result.returncode == 0, result.stderr
    block = json.loads(result.stdout)
    assert block["sha256"] == expected
    assert block["modules"] == ["gcc/12.3.0", "openmpi/4.1.6-gcc"]
    assert block["flags"]["CFLAGS"] == "-O3 -march=sapphirerapids"
    assert block["compilers"]["CC"]["path"]


def test_check_passes_on_an_empty_stacks_root(tmp_path):
    result = _stack_sh("check", root=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_check_fails_on_a_stack_built_on_two_toolchains(tmp_path):
    _prefix(tmp_path, "ngsolve", "a" * 64)
    _prefix(tmp_path, "petsc", "c" * 64)

    result = _stack_sh("check", root=tmp_path)

    assert result.returncode != 0
    assert "toolchain hashes differ" in result.stdout + result.stderr


def test_check_of_one_stack_prints_the_allowed_mumps_differences(tmp_path):
    _mumps_stack(tmp_path, _mumps_palace(), _mumps_petsc())

    result = _stack_sh("check", "native-oss", root=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mumps openmp allowed difference" in result.stdout


def test_the_stack_script_needs_a_stacks_root():
    env = {k: v for k, v in os.environ.items() if k != "STACKS_ROOT"}

    result = subprocess.run(
        ["bash", str(STACK_SH), "check"],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "STACKS_ROOT" in result.stderr


def test_a_build_outside_a_slurm_job_is_refused(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    env["STACKS_ROOT"] = str(tmp_path)

    result = subprocess.run(
        ["bash", str(STACK_SH), "build", "native-oss", "openblas"],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SLURM_JOB_ID" in result.stderr
    assert not (tmp_path / "native-oss").exists()


def test_every_package_recipe_declares_the_recipe_interface(tmp_path):
    for package in ("openblas", "palace", "ngsolve", "petsc", "pyenv"):
        result = _stack_sh("check", "native-oss", package, root=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        fields = dict(
            line.strip().split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("  ") and "=" in line
        )
        assert f"native-oss/{package}-{fields['version']}: absent" in result.stdout
        assert "depends" in fields
        assert fields["cpus"].isdigit()
        assert fields["time"].count(":") == 2


@pytest.mark.parametrize(
    "script", [*sorted(STACKS.rglob("*.sh")), STACKS / "build.sbatch"], ids=str
)
def test_every_stack_script_parses(script):
    result = subprocess.run(
        ["bash", "-n", str(script)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_the_commit_file_of_a_shipped_tree_leaves_the_recipe_hash_alone(tmp_path):
    (tmp_path / "stack.sh").write_text("echo stack\n")
    before = provenance.tree_sha256(tmp_path)

    (tmp_path / "COMMIT").write_text("0" * 40 + "\n")

    assert provenance.tree_sha256(tmp_path) == before
