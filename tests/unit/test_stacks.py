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


def _recipes(root: Path) -> Path:
    """A shipped recipe tree with two packages, ``a`` and ``b``."""
    (root / "packages").mkdir(parents=True)
    for name in ("stack.sh", "build.sbatch", "provenance.py", "README.md"):
        (root / name).write_text(f"# {name}\n")
    for package in ("a", "b"):
        (root / "packages" / f"{package}.sh").write_text("VERSION=1.0\n")
    return root


def test_filling_another_packages_recipe_leaves_a_built_prefix_complete(tmp_path):
    recipes = _recipes(tmp_path / "recipes")
    prefix = tmp_path / "native-oss" / "a-1.0"
    prefix.mkdir(parents=True)
    record = _record("a")
    record["recipe"]["sha256"] = provenance.recipe_sha256(recipes, "a")
    provenance.write(prefix, record)

    (recipes / "packages" / "b.sh").write_text("VERSION=2.0\nbuild_b() { :; }\n")
    (recipes / "README.md").write_text("# rewritten\n")
    (recipes / "COMMIT").write_text("0" * 40 + "\n")

    now = provenance.recipe_sha256(recipes, "a")
    assert provenance.state(prefix, now, "a" * 64) == "complete"


@pytest.mark.parametrize(
    "changed", ["stack.sh", "build.sbatch", "provenance.py", "packages/a.sh"]
)
def test_a_change_to_the_build_machinery_or_own_recipe_changes_the_hash(
    tmp_path, changed
):
    recipes = _recipes(tmp_path)
    before = provenance.recipe_sha256(recipes, "a")

    (recipes / changed).write_text("# changed\n")

    assert provenance.recipe_sha256(recipes, "a") != before


def test_the_recipe_hash_follows_content_not_location(tmp_path):
    one, two = _recipes(tmp_path / "one"), _recipes(tmp_path / "two")

    assert provenance.recipe_sha256(one, "a") == provenance.recipe_sha256(two, "a")


def test_palaces_hash_covers_sources_py_which_its_build_runs(tmp_path):
    recipes = _recipes(tmp_path)
    (recipes / "packages" / "palace.sh").write_text("VERSION=0.19.0\n")
    (recipes / "sources.py").write_text("# sources.py\n")
    palace, other = (provenance.recipe_sha256(recipes, p) for p in ("palace", "a"))

    (recipes / "sources.py").write_text("# changed\n")

    assert provenance.recipe_sha256(recipes, "palace") != palace
    assert provenance.recipe_sha256(recipes, "a") == other


# --- source mirror: what `stack.sh fetch` reads from a fetched tag ------------

_sources_spec = importlib.util.spec_from_file_location("sources", STACKS / "sources.py")
assert _sources_spec is not None
assert _sources_spec.loader is not None
sources = importlib.util.module_from_spec(_sources_spec)
_sources_spec.loader.exec_module(sources)

#: An excerpt of Palace v0.19.0's ``cmake/ExternalGitTags.cmake``.
PALACE_GIT_TAGS = """
# ARPACK-NG
set(EXTERN_ARPACK_URL
  "https://github.com/opencollab/arpack-ng.git" CACHE STRING
  "URL for external ARPACK-NG build"
)
set(EXTERN_ARPACK_GIT_BRANCH
  "master" CACHE STRING
  "Git branch for external ARPACK-NG build"
)
set(EXTERN_ARPACK_GIT_TAG
  "3.9.1" CACHE STRING
  "Git tag for external ARPACK-NG build"
)

# METIS
set(EXTERN_METIS_URL
  "https://bitbucket.org/petsc/pkg-metis.git" CACHE STRING
  "URL for external METIS build"
)
set(EXTERN_METIS_GIT_TAG
  "v5.1.0-p13" CACHE STRING
  "Git tag for external METIS build"
)

# nlohmann/json
set(EXTERN_JSON_URL
  "https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz" CACHE STRING
  "URL for external nlohmann/json build"
)

# nlohmann/json-schema-validator
set(EXTERN_JSON_SCHEMA_VALIDATOR_URL
  "https://github.com/pboettch/json-schema-validator/archive/refs/tags/2.4.1.tar.gz"
  CACHE STRING
  "URL for external nlohmann/json-schema-validator build"
)

# fmt
set(EXTERN_FMT_URL
  "https://github.com/fmtlib/fmt/releases/download/12.2.0/fmt-12.2.0.zip" CACHE STRING
  "URL for external fmt build"
)

# scn
set(EXTERN_SCN_URL
  "https://github.com/eliaskosunen/scnlib/archive/refs/tags/v4.0.1.zip" CACHE STRING
  "URL for external scn build"
)

# Eigen
set(EXTERN_EIGEN_URL
  "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz" CACHE STRING
  "URL for external Eigen build"
)

# PETSc (for SLEPc)
set(EXTERN_PETSC_URL
  "https://gitlab.com/petsc/petsc.git" CACHE STRING
  "URL for external PETSc build"
)
set(EXTERN_PETSC_GIT_TAG
  "v3.26.0" CACHE STRING
  "Git tag for external PETSc build"
)
"""


def test_palaces_git_dependencies_are_read_from_its_tag():
    deps = sources.palace_git(PALACE_GIT_TAGS)

    assert deps == [
        ("ARPACK", "https://github.com/opencollab/arpack-ng.git", "3.9.1"),
        ("METIS", "https://bitbucket.org/petsc/pkg-metis.git", "v5.1.0-p13"),
        ("PETSC", "https://gitlab.com/petsc/petsc.git", "v3.26.0"),
    ]


def test_palaces_url_tarballs_keep_the_names_of_the_source_mirror():
    tarballs = sources.palace_tarballs(PALACE_GIT_TAGS)

    assert [(name, file) for name, file, _ in tarballs] == [
        ("JSON", "json-3.12.0.tar.xz"),
        ("JSON_SCHEMA_VALIDATOR", "json-schema-validator-2.4.1.tar.gz"),
        ("FMT", "fmt-12.2.0.zip"),
        ("SCN", "scnlib-v4.0.1.zip"),
        ("EIGEN", "eigen-5.0.1.tar.gz"),
    ]
    assert tarballs[0][2] == (
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz"
    )


@pytest.mark.parametrize(
    ("url", "name"),
    [
        (
            "https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz",
            "petsc-pkg-parmetis-v4.0.3-p9.tar.gz",
        ),
        (
            "https://github.com/Reference-ScaLAPACK/scalapack/archive/v2.2.3.tar.gz",
            "scalapack-v2.2.3.tar.gz",
        ),
        ("https://mumps-solver.org/MUMPS_5.9.1.tar.gz", "MUMPS_5.9.1.tar.gz"),
    ],
)
def test_a_tarball_is_named_after_its_project_and_version(url, name):
    assert sources.tarball_name(url) == name


#: Excerpts of Palace v0.19.0's ``ExternalMFEM.cmake`` and ``ExternalSTRUMPACK.cmake``.
PALACE_PATCH_CALLS = """
set(MFEM_PATCH_DIR "${CMAKE_BINARY_DIR}/extern/mfem-patches")
set(MFEM_PATCH_FILES)
download_patch(MFEM_PATCH_FILES "${MFEM_PATCH_DIR}"
  mfem_pr3847.diff
  "https://github.com/mfem/mfem/compare/2d574015756711029556c14d096ca52c15d5b663...50ead1a9a785e3273b2a72ff59ac8ed8a496b498.diff"
  e9be1a0d4b2642ed1b72f31c36065cf0aacbe342b6594d5d943782c73a6177f4
)
if(PALACE_WITH_STRUMPACK)
  download_patch(ZFP_PATCH_FILES "${CMAKE_BINARY_DIR}/extern/zfp-patches"
    stream-maximum-size-overflow.patch
    "https://raw.githubusercontent.com/spack/spack-packages/94ba1d9eb8e89a4e01919a9f41b8c857b5fbe5cc/repos/spack_repo/builtin/packages/zfp/stream-maximum-size-overflow.patch"
    7ed755650fdcd79208ab165b42c902d64a916187be6dade4643814c09ba6d894
  )
endif()
"""


def test_palaces_patches_land_where_its_build_tree_looks_for_them():
    patches = sources.patches(PALACE_PATCH_CALLS)

    assert patches == [
        (
            "extern/mfem-patches",
            "mfem_pr3847.diff",
            "https://github.com/mfem/mfem/compare/2d574015756711029556c14d096ca52c15d5b663...50ead1a9a785e3273b2a72ff59ac8ed8a496b498.diff",
            "e9be1a0d4b2642ed1b72f31c36065cf0aacbe342b6594d5d943782c73a6177f4",
        ),
        (
            "extern/zfp-patches",
            "stream-maximum-size-overflow.patch",
            "https://raw.githubusercontent.com/spack/spack-packages/94ba1d9eb8e89a4e01919a9f41b8c857b5fbe5cc/repos/spack_repo/builtin/packages/zfp/stream-maximum-size-overflow.patch",
            "7ed755650fdcd79208ab165b42c902d64a916187be6dade4643814c09ba6d894",
        ),
    ]


def test_git_dependencies_hard_coded_in_cmake_are_read_from_the_tag():
    catch2 = """
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
  )
"""
    suitesparse = """
    ExternalProject_Add(
      suitesparse
      DEPENDS ${LAPACK_PROJECTS}
      GIT_REPOSITORY https://github.com/jlblancoc/suitesparse-metis-for-windows.git
      GIT_TAG 1618fd16ea34be287c0fbc32789ca280012a9280
      LIST_SEPARATOR |
"""

    assert sources.git_repositories(catch2) == [
        ("https://github.com/catchorg/Catch2.git", "v3.8.1")
    ]
    assert sources.git_repositories(suitesparse) == [
        (
            "https://github.com/jlblancoc/suitesparse-metis-for-windows.git",
            "1618fd16ea34be287c0fbc32789ca280012a9280",
        )
    ]


def test_submodules_are_read_as_spelled_in_gitmodules():
    text = """[submodule "external_dependencies/pybind11"]
\tpath = external_dependencies/pybind11
\turl = https://github.com/ngsolve/pybind11.git
"""

    assert sources.gitmodules(text) == [
        ("external_dependencies/pybind11", "https://github.com/ngsolve/pybind11.git")
    ]


def test_palaces_mumps_tarball_is_the_release_its_scivision_tag_names():
    source_json = """{
  "mumps_sha256": {
    "5.9.0": "02c6efdb91749ec0f82351d40f3f860547272a1eb1d899126a4265b4d6bcc4ca",
    "5.9.1": "659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46"
  },
  "mumps_sources": {}
}"""

    assert sources.mumps_tarball(source_json, "v5.9.1.2") == [
        (
            "MUMPS_5.9.1.tar.gz",
            "https://mumps-solver.org/MUMPS_5.9.1.tar.gz",
            "659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46",
        )
    ]


# --- `stack.sh fetch`: the source mirror helpers, through a fake recipe -------

FAKE_RECIPE = """VERSION=1.0
CPUS=1
TIME=00:10:00
STACKS=(native-oss)
fetch_fake() {
  local up=file://$FAKE_UPSTREAM
  stack_tarball tarballs pinned-1.0.tar.gz "$up/pinned.tar.gz" "$FAKE_PIN"
  stack_tarball tarballs unpinned-1.0.tar.gz "$up/unpinned.tar.gz"
  stack_mirror https://example.invalid/owner/repo.git v1.0
}
build_fake() { :; }
smoke_fake() { :; }
"""


def _fake_fetch(tmp_path: Path, pin: str | None = None):
    """A recipe tree with package ``fake`` and its upstream, ready to fetch."""
    import hashlib
    import shutil

    recipes = tmp_path / "recipes"
    shutil.copytree(STACKS, recipes, ignore=shutil.ignore_patterns("__pycache__"))
    (recipes / "packages" / "fake.sh").write_text(FAKE_RECIPE)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "pinned.tar.gz").write_bytes(b"pinned tarball")
    (upstream / "unpinned.tar.gz").write_bytes(b"unpinned tarball")
    repo = upstream / "owner" / "repo.git"
    for command in (
        ["git", "init", "-q", "-b", "main", str(repo)],
        ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "one"],
        ["git", "-C", str(repo), "tag", "v1.0"],
    ):
        subprocess.run(command, check=True, capture_output=True)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        f'[url "file://{upstream}/"]\n\tinsteadOf = https://example.invalid/\n'
        "[user]\n\tname = t\n\temail = t@t\n"
    )
    env = {
        **os.environ,
        "STACKS_ROOT": str(tmp_path / "root"),
        "FAKE_UPSTREAM": str(upstream),
        "FAKE_PIN": pin or hashlib.sha256(b"pinned tarball").hexdigest(),
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    env.pop("SLURM_JOB_ID", None)

    def fetch() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(recipes / "stack.sh"), "fetch", "fake"],  # noqa: S607
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    return fetch, tmp_path / "root" / "src"


def test_a_fetch_fills_the_mirror_and_a_second_one_finds_nothing_new(tmp_path):
    fetch, src = _fake_fetch(tmp_path)

    first = fetch()
    second = fetch()

    assert first.returncode == 0, first.stdout + first.stderr
    assert (src / "tarballs" / "pinned-1.0.tar.gz").read_bytes() == b"pinned tarball"
    assert (src / "mirrors" / "example.invalid" / "owner" / "repo.git").is_dir()
    assert "3 new" in first.stdout
    assert second.returncode == 0, second.stdout + second.stderr
    assert "nothing new" in second.stdout


def test_only_unpinned_tarballs_are_recorded_on_first_fetch(tmp_path):
    fetch, src = _fake_fetch(tmp_path)

    fetch()

    names = [
        line.split()[1]
        for line in (src / "tarballs" / "SHA256SUMS").read_text().splitlines()
    ]
    assert names == ["unpinned-1.0.tar.gz"]


def test_an_unpinned_tarball_changed_since_its_first_fetch_fails_the_fetch(tmp_path):
    fetch, src = _fake_fetch(tmp_path)
    fetch()

    (src / "tarballs" / "unpinned-1.0.tar.gz").write_bytes(b"tampered")
    result = fetch()

    assert result.returncode != 0
    assert "unpinned-1.0.tar.gz" in result.stderr


def test_a_tarball_that_does_not_match_its_pin_is_not_kept(tmp_path):
    fetch, src = _fake_fetch(tmp_path, pin="0" * 64)

    result = fetch()

    assert result.returncode != 0
    assert "pinned-1.0.tar.gz" in result.stderr
    assert not (src / "tarballs" / "pinned-1.0.tar.gz").exists()


def _recipe_array(package: str, name: str) -> list[str]:
    result = subprocess.run(
        [  # noqa: S607
            "bash",
            "-c",
            f'source "$1"; printf "%s\\n" "${{{name}[@]}}"',
            "-",
            str(STACKS / "packages" / f"{package}.sh"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


def test_the_stack_venvs_runtime_pins_are_gfems_lock():
    import tomllib

    lock = tomllib.loads((STACKS.parents[1] / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    # Ticket 07: never the PyPI NGSolve, netgen or their OpenBLAS in a stack venv.
    native = {"ngsolve", "netgen-mesher", "ngsolve-openblas"}
    closure, todo = set(), [d["name"] for d in packages["gfem"]["dependencies"]]
    while todo:
        name = todo.pop()
        if name not in closure:
            closure.add(name)
            todo += [d["name"] for d in packages[name].get("dependencies", [])]
    expected = sorted(
        f"{name}=={packages[name]['version']}" for name in closure - native
    )

    assert sorted(_recipe_array("pyenv", "PIP_RUNTIME")) == expected
    # mpi4py is built from its sdist against the stack MPI, at the lock's pin.
    assert _recipe_array("pyenv", "PIP_SDISTS") == [
        f"mpi4py=={packages['mpi4py']['version']}"
    ]


# --- package openblas: a build with make and module faked ---------------------

#: Stand-in for OpenBLAS's make: records each call, and on ``install`` puts a
#: tiny library (naive dgemm/zdotc, OpenBLAS's config and threading queries)
#: under the name OpenBLAS would give it.
FAKE_OPENBLAS_MAKE = r"""#!/usr/bin/env bash
set -euo pipefail
echo "$* COMMON_OPT=${COMMON_OPT:-} FCOMMON_OPT=${FCOMMON_OPT:-}" >>"$FAKE_MAKE_LOG"
prefix= suffix= openmp=0
for arg in "$@"; do
  case $arg in
  PREFIX=*) prefix=${arg#*=} ;;
  LIBNAMESUFFIX=*) suffix=${arg#*=} ;;
  USE_OPENMP=1) openmp=1 ;;
  esac
done
[ "${FAKE_IGNORE_OPENMP:-}" = 1 ] && openmp=0
[ -n "$prefix" ] || exit 0
work=$(mktemp -d)
cat >"$work/blas.f90" <<'F90'
subroutine dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  character :: ta, tb
  integer :: m, n, k, lda, ldb, ldc
  double precision :: alpha, beta, a(lda, *), b(ldb, *), c(ldc, *)
  c(1:m, 1:n) = alpha * matmul(a(1:m, 1:k), b(1:k, 1:n)) + beta * c(1:m, 1:n)
end subroutine
complex(kind(0d0)) function zdotc(n, x, incx, y, incy)
  integer :: n, incx, incy
  complex(kind(0d0)) :: x(*), y(*)
  zdotc = sum(conjg(x(1:n)) * y(1:n))
end function
F90
if [ "$openmp" = 1 ]; then flag="USE_OPENMP " parallel=2; else flag= parallel=1; fi
cat >"$work/query.c" <<C
char *openblas_get_config(void) {
  return "OpenBLAS 0.3.34 NO_AFFINITY ${flag}SAPPHIRERAPIDS MAX_THREADS=256";
}
int openblas_get_parallel(void) { return $parallel; }
int openblas_get_num_threads(void) { return 1; }
C
mkdir -p "$prefix/lib/pkgconfig" "$prefix/include"
[ "$openmp" = 1 ] && omp="-fopenmp -Wl,--no-as-needed -lgomp" || omp=
# shellcheck disable=SC2086
gfortran -shared -fPIC $omp $LDFLAGS -o "$prefix/lib/libopenblas$suffix.so" \
  "$work/blas.f90" "$work/query.c"
echo "Libs: -lopenblas$suffix" >"$prefix/lib/pkgconfig/openblas.pc"
echo "/* cblas */" >"$prefix/include/cblas.h"
rm -rf "$work"
"""


def _openblas_build(
    tmp_path: Path, **env: str
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run ``stack.sh build native-oss openblas`` with make and module faked."""
    import shutil
    import tarfile

    recipes = tmp_path / "recipes"
    shutil.copytree(STACKS, recipes, ignore=shutil.ignore_patterns("__pycache__"))
    fakes = tmp_path / "bin"
    fakes.mkdir()
    (fakes / "make").write_text(FAKE_OPENBLAS_MAKE)
    (fakes / "module").write_text("#!/bin/sh\nexit 0\n")
    for fake in fakes.iterdir():
        fake.chmod(0o755)
    root = tmp_path / "root"
    source = tmp_path / "OpenBLAS-0.3.34"
    source.mkdir()
    (source / "Makefile").write_text("# fake\n")
    (root / "src" / "tarballs").mkdir(parents=True)
    with tarfile.open(
        root / "src" / "tarballs" / "OpenBLAS-0.3.34.tar.gz", "w:gz"
    ) as tar:
        tar.add(source, arcname="OpenBLAS-0.3.34")
    (tmp_path / "job").mkdir()
    log = tmp_path / "make.log"
    result = subprocess.run(
        [  # noqa: S607
            "bash",
            str(recipes / "stack.sh"),
            "build",
            "--allow-dirty",
            "native-oss",
            "openblas",
        ],
        env={
            **os.environ,
            "PATH": f"{fakes}:{os.environ['PATH']}",
            "STACKS_ROOT": str(root),
            "SLURM_JOB_ID": "1",
            "SLURM_CPUS_PER_TASK": "2",
            "TMPDIR": str(tmp_path / "job"),
            "FAKE_MAKE_LOG": str(log),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, root / "native-oss" / "openblas-0.3.34", log


needs_gfortran = pytest.mark.skipif(
    __import__("shutil").which("gfortran") is None, reason="needs gfortran"
)


@needs_gfortran
def test_openblas_builds_a_pthreads_and_an_openmp_library_into_one_prefix(tmp_path):
    result, prefix, log = _openblas_build(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (prefix / "lib" / "libopenblas.so").is_file()
    assert (prefix / "lib" / "libopenblas_omp.so").is_file()
    calls = log.read_text().splitlines()
    common = {
        "TARGET=SAPPHIRERAPIDS",
        "DYNAMIC_ARCH=0",
        "NUM_THREADS=256",
        "NO_AFFINITY=1",
    }
    builds = [set(call.split()) for call in calls if "install" not in call.split()]
    assert len(builds) == 2
    pthreads, openmp = sorted(builds, key=lambda args: "USE_OPENMP=1" in args)
    assert common | {"USE_THREAD=1", "USE_OPENMP=0"} <= pthreads
    assert common | {"USE_OPENMP=1", "LIBNAMESUFFIX=_omp"} <= openmp
    assert all("COMMON_OPT=-O3 -march=sapphirerapids" in call for call in calls)
    record = json.loads((prefix / "provenance.json").read_text())
    assert set(record["smoke"]["libraries"]) == {"openblas", "openblas_omp"}
    assert "zdotc ok" in record["smoke"]["libraries"]["openblas_omp"]


@needs_gfortran
def test_an_openblas_build_whose_openmp_library_is_not_openmp_fails_its_smoke(tmp_path):
    result, prefix, _ = _openblas_build(tmp_path, FAKE_IGNORE_OPENMP="1")

    assert result.returncode != 0
    assert "openblas_omp" in result.stderr
    assert not (prefix / "provenance.json").exists()


# --- Palace: the build's edits and checks -------------------------------------

#: An excerpt of Palace v0.19.0's ``cmake/ExternalMUMPS.cmake``.
PALACE_EXTERNAL_MUMPS = """\
set(MUMPS_OPTIONS ${PALACE_SUPERBUILD_DEFAULT_ARGS})
list(APPEND MUMPS_OPTIONS
  "-DMUMPS_scalapack=ON"
  "-DMUMPS_USE_MKL=OFF"
  "-DMUMPS_gemmt=ON"
  "-DPARMETIS_LIBRARY=${PARMETIS_LIBRARIES}"
)
"""


def _palace_function(*call: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run one function of ``packages/palace.sh`` (sourced as stack.sh does)."""
    script = (
        'set -euo pipefail; stack_die() { echo "$*" >&2; exit 1; }; '
        f'source "{STACKS}/packages/palace.sh"; "$@"'
    )
    return subprocess.run(
        ["bash", "-c", script, "palace", *call],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_palaces_mumps_is_built_from_the_mirrors_tarball_without_gemmt(tmp_path):
    cmake = tmp_path / "ExternalMUMPS.cmake"
    cmake.write_text(PALACE_EXTERNAL_MUMPS)
    tarball = "/stacks/src/tarballs/MUMPS_5.9.1.tar.gz"

    result = _palace_function("palace_edit_mumps", str(cmake), tarball, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    edited = cmake.read_text().splitlines()
    original = PALACE_EXTERNAL_MUMPS.splitlines()
    changed = [new for old, new in zip(original, edited, strict=True) if old != new]
    assert changed == [f'  "-DMUMPS_gemmt=OFF" "-DMUMPS_url={tarball}"']


def test_an_external_mumps_file_without_the_gemmt_line_fails_the_edit(tmp_path):
    cmake = tmp_path / "ExternalMUMPS.cmake"
    cmake.write_text(PALACE_EXTERNAL_MUMPS.replace("gemmt=ON", "gemmt=${X}"))

    result = _palace_function(
        "palace_edit_mumps", str(cmake), "/t.tar.gz", cwd=tmp_path
    )

    assert result.returncode != 0
    assert "MUMPS_gemmt" in result.stderr


needs_gcc = pytest.mark.skipif(
    __import__("shutil").which("gcc") is None, reason="needs gcc"
)


@needs_gcc
def test_a_library_calling_a_gemmt_is_named_by_the_gemmt_check(tmp_path):
    for name, routine in (("with", "dgemmt_"), ("without", "dgemm_")):
        (tmp_path / f"{name}.c").write_text(
            f"void {routine}(void);\nvoid call(void) {{ {routine}(); }}\n"
        )
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", f"lib{name}.so", f"{name}.c"],  # noqa: S607
            cwd=tmp_path,
            check=True,
        )

    result = _palace_function(
        "palace_gemmt_imports", "libwith.so", "libwithout.so", cwd=tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["libwith.so:dgemmt_"]
