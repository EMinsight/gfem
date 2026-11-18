"""Provenance files of native software stacks: write, read and compare them.

A package prefix ``$STACKS_ROOT/<stack>/<pkg>-<version>/`` is complete once its
``provenance.json`` exists: ``stack.sh build`` writes it last, after install and
smoke step (ADR 0191). This module is the only writer and reader; ``stack.sh``
calls its command line. It runs on the cluster's system Python, so it keeps to
Python 3.9 and the standard library.

Command line (``python3 provenance.py COMMAND ...``)::

    write PREFIX META_DIR      write PREFIX/provenance.json from META_DIR/<field>.json
    check STACK_DIR...         compare each stack's complete prefixes; exit 1 on errors
    state PREFIX RECIPE_HASH TOOLCHAIN_HASH
                               print absent, incomplete, complete or differs
    tofu SUMS FILE             record FILE's sha256 in SUMS on first fetch, check after
    toolchain FILE MODULES     print the toolchain block (FILE: frozen toolchain file,
                               MODULES: output of ``module -t list``)
    recipe-sha256 DIR PKG      print the recipe hash of package PKG in tree DIR
    sha256 FILE                print a file's sha256
    petsc-contract WHEELHOUSE RECIPE OPENBLAS
                               compare packages/petsc.sh and packages/openblas.sh's
                               tarball pin with tools/petsc_wheelhouse.sh; exit 1 on
                               drift (ADR 0209)
    petsc-mumps PREFIX BUILD_DIR
                               print the ``mumps`` block of a PETSc prefix (headers in
                               PREFIX/include, the other PETSC_BUILD_FILES in
                               BUILD_DIR); exit 1 if it breaks ADR 0067's gates
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: The provenance schema this module writes and reads.
SCHEMA = 1
#: The file whose presence marks a prefix complete.
FILENAME = "provenance.json"
#: The fields every schema-1 record holds (issue 06 Q7, ADR 0191).
FIELDS = (
    "package",
    "version",
    "stack",
    "source",
    "toolchain",
    "configure",
    "recipe",
    "build",
    "depends",
    "smoke",
)


def validate(record: dict) -> None:
    """Refuse a record that lacks a schema-1 field.

    Raises:
        ValueError: naming the first missing field.
    """
    for field in FIELDS:
        if field not in record:
            msg = f"provenance record has no {field!r} field"
            raise ValueError(msg)
    if not record["toolchain"].get("sha256"):
        msg = "provenance record has no 'toolchain.sha256' field"
        raise ValueError(msg)


def write(prefix: Path, record: dict) -> Path:
    """Write the provenance file of ``prefix``, marking it complete.

    Args:
        prefix: The package prefix.
        record: The build's fields; ``schema``, ``prefix`` and ``complete_utc``
            are stamped here.

    Returns:
        The path of the provenance file.
    """
    validate(record)
    prefix = Path(prefix)
    stamped = dict(record)
    stamped["schema"] = SCHEMA
    stamped["prefix"] = str(prefix)
    stamped["complete_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = prefix / FILENAME
    partial = prefix / (FILENAME + ".partial")
    partial.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    partial.replace(path)
    return path


def read(prefix: Path) -> dict | None:
    """The provenance record of ``prefix``, or ``None`` if it is not complete."""
    path = Path(prefix) / FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


#: The ``mumps`` block of a provenance file (ticket 05's table, ADR 0209):
#: fields both MUMPS builds of a stack must share ...
MUMPS_MUST_MATCH = (
    "version",
    "tarball_sha256",
    "intsize",
    "scalapack",
    "compiler",
    "fflags",
    "blas.vendor",
    "blas.version",
)
#: ... and fields where they may differ, printed by every check.
MUMPS_ALLOWED = ("arithmetics", "openmp", "orderings", "metis", "blas.threading")
#: The packages that build a MUMPS and so must record its block.
MUMPS_PACKAGES = ("palace", "petsc")
#: MUMPS's default selective 64-bit build: 32-bit row and column indices.
MUMPS_INTSIZE = 32


class Report:
    """What ``check`` found: errors fail it, notes are printed."""

    def __init__(self) -> None:
        """An empty report."""
        self.errors: list[str] = []
        self.notes: list[str] = []


def prefixes(stack_dir: Path) -> list[Path]:
    """The package prefixes of one software stack, finished or not."""
    stack_dir = Path(stack_dir)
    if not stack_dir.is_dir():
        return []
    return sorted(
        path
        for path in stack_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and "-" in path.name
    )


def check_stack(stack_dir: Path) -> Report:
    """Check the complete prefixes of one software stack against each other.

    Every complete prefix must name the same toolchain hash: one toolchain
    for every code of a stack (ADR 0191). Incomplete prefixes are noted.
    """
    report = Report()
    records = {}
    for prefix in prefixes(stack_dir):
        record = read(prefix)
        if record is None:
            report.notes.append(f"{prefix.name}: incomplete (no {FILENAME})")
        else:
            records[prefix.name] = record
            report.notes.append(f"{prefix.name}: complete {record['complete_utc']}")
    _check_mumps(records, report)
    hashes = {name: r["toolchain"]["sha256"] for name, r in records.items()}
    if len(set(hashes.values())) > 1:
        listing = ", ".join(f"{name} {h[:12]}" for name, h in sorted(hashes.items()))
        report.errors.append(f"toolchain hashes differ within the stack: {listing}")
    return report


def _field(block: dict, dotted: str) -> object:
    """The value at ``dotted`` (``"blas.vendor"``) in ``block``; KeyError if none."""
    value = block
    for key in dotted.split("."):
        value = value[key]
    return value


def _canonical(dotted: str, value: object) -> object:
    """``value`` as compared: compiler strings reduce to ``gfortran <version>``.

    Recipes report the Fortran compiler differently (``gfortran --version``'s
    first line against ``gfortran -dumpfullversion``); the same GCC must match.
    """
    if dotted == "compiler" and isinstance(value, str):
        version = re.search(r"(\d+\.\d+\.\d+)\s*$", value.strip())
        gnu = "gfortran" in value or "GNU Fortran" in value
        if gnu and version:
            return f"gfortran {version.group(1)}"
    return value


def _check_mumps(records: dict, report: Report) -> None:
    """Compare the ``mumps`` blocks of a stack's complete prefixes (ADR 0209)."""
    blocks = {}
    for name, record in sorted(records.items()):
        block = record.get("mumps")
        if block is None:
            if record["package"] in MUMPS_PACKAGES:
                report.errors.append(f"{name}: no mumps block (ADR 0209)")
            continue
        missing = [
            dotted
            for dotted in (*MUMPS_MUST_MATCH, *MUMPS_ALLOWED, "gemmt")
            if not _has(block, dotted)
        ]
        if missing:
            report.errors.append(
                f"{name}: mumps block has no {', '.join(repr(m) for m in missing)}"
            )
            continue
        blocks[name] = block
        _check_mumps_block(name, record["package"], block, report)
    names = sorted(blocks)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = blocks[first], blocks[second]
            for dotted in MUMPS_MUST_MATCH:
                if _canonical(dotted, _field(a, dotted)) != _canonical(
                    dotted, _field(b, dotted)
                ):
                    report.errors.append(
                        f"mumps {dotted} must match: {first} {_field(a, dotted)!r}, "
                        f"{second} {_field(b, dotted)!r}"
                    )
            for dotted in MUMPS_ALLOWED:
                if _field(a, dotted) != _field(b, dotted):
                    report.notes.append(
                        f"mumps {dotted} allowed difference: "
                        f"{first} {_field(a, dotted)!r}, "
                        f"{second} {_field(b, dotted)!r}"
                    )


def _check_mumps_block(name: str, package: str, block: dict, report: Report) -> None:
    """The rules one MUMPS build meets on its own (ADR 0205, ADR 0209)."""
    if block["intsize"] != MUMPS_INTSIZE:
        report.errors.append(
            f"mumps intsize: {name} has {block['intsize']}, must be {MUMPS_INTSIZE}"
        )
    if block["gemmt"]:
        report.errors.append(f"mumps gemmt: {name} uses ?gemmt_ (ADR 0205)")
    if "metis" not in block["orderings"]:
        report.errors.append(f"mumps orderings: {name} has no metis")
    if package == "petsc" and "parmetis" in block["orderings"]:
        report.errors.append(
            f"mumps orderings: {name} has parmetis (ADR 0012, ADR 0067)"
        )
    raw = block.get("fflags_raw")
    if raw is not None:
        missing = [flag for flag in block["fflags"].split() if flag not in raw.split()]
        if missing:
            report.errors.append(
                f"mumps fflags_raw: {name} compiled MUMPS with {raw!r}, "
                f"without the toolchain's {missing}"
            )


def _has(block: dict, dotted: str) -> bool:
    try:
        _field(block, dotted)
    except (KeyError, TypeError):
        return False
    return True


def sha256(path: Path) -> str:
    """The sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tofu(sums: Path, path: Path) -> str:
    """Trust a tarball no upstream file pins on its first fetch, check it after.

    The record ``sums`` is in ``sha256sum`` format, one ``<hex>  <name>`` line
    per tarball, keyed by file name (ADR 0212).

    Returns:
        ``"recorded"`` on the first fetch, ``"verified"`` after.

    Raises:
        ValueError: if the tarball's hash differs from the recorded one.
    """
    sums, path = Path(sums), Path(path)
    recorded = {}
    if sums.is_file():
        for line in sums.read_text().splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                recorded[name.lstrip("*")] = digest
    actual = sha256(path)
    if path.name not in recorded:
        with sums.open("a") as handle:
            handle.write(f"{actual}  {path.name}\n")
        return "recorded"
    if recorded[path.name] != actual:
        msg = (
            f"{path.name}: sha256 {actual} differs from the one recorded on "
            f"first fetch in {sums} ({recorded[path.name]})"
        )
        raise ValueError(msg)
    return "verified"


_SCALAR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=([^\s(#]+)", re.MULTILINE)
_ARRAY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=\(([^)]*)\)", re.MULTILINE)
_PIN = re.compile(
    r"^\s*([a-z0-9]+)-([0-9][0-9.]*)\)\s*echo\s+([0-9a-f]{64})", re.MULTILINE
)
_OPENBLAS_PIN = re.compile(r"^\s*([0-9][0-9.]*)\)\s*pin=([0-9a-f]{64})", re.MULTILINE)


def _bash_scalars(text: str) -> dict[str, str]:
    """The plain ``NAME=value`` assignments of a bash file (first one wins)."""
    found: dict[str, str] = {}
    for name, value in _SCALAR.findall(text):
        found.setdefault(name, value.strip("\"'"))
    return found


def _bash_tarballs(text: str, name: str) -> dict[str, list[str]]:
    """``NAME URL [SHA256]`` entries of the bash array ``name``, keyed by file name."""
    for array, body in _ARRAY.findall(text):
        if array == name:
            entries = [line.split() for line in re.findall(r'"([^"]*)"', body)]
            return {entry[0]: entry[1:] for entry in entries if entry}
    return {}


def petsc_contract_drift(wheelhouse: str, recipe: str, openblas: str) -> list[str]:
    """Where the native recipes differ from ``tools/petsc_wheelhouse.sh``.

    The native PETSc recipe copies the wheelhouse's ADR 0067 contract (ADR
    0209): the same PETSc/SLEPc version, the same sdists and the same
    ``--download`` tarballs (by URL and sha256). The wheelhouse's OpenBLAS
    tarball is built by package openblas instead, which must pin the same
    sha256 for that release.

    Args:
        wheelhouse: The text of ``tools/petsc_wheelhouse.sh``.
        recipe: The text of ``tools/stacks/packages/petsc.sh``.
        openblas: The text of ``tools/stacks/packages/openblas.sh``.

    Returns:
        One line per difference; empty when they agree.
    """
    house, ours = _bash_scalars(wheelhouse), _bash_scalars(recipe)
    version = ours.get("VERSION")
    drift = [
        f"version: petsc_wheelhouse.sh {name}={house.get(name)}, "
        f"packages/petsc.sh VERSION={version}"
        for name in ("PETSC_VERSION", "SLEPC_VERSION")
        if house.get(name) != version
    ]
    pins = {
        package: digest
        for package, pinned, digest in _PIN.findall(recipe)
        if pinned == version
    }
    drift += [
        f"{package} sdist sha256: petsc_wheelhouse.sh "
        f"{house.get('SHA256_' + package)}, packages/petsc.sh {pins.get(package)}"
        for package in ("petsc", "petsc4py", "slepc", "slepc4py")
        if house.get(f"SHA256_{package}") != pins.get(package)
    ]
    native = _bash_tarballs(recipe, "PETSC_TARBALLS")
    drift += [
        f"{name}: petsc_wheelhouse.sh {' '.join(entry)}, "
        f"packages/petsc.sh {' '.join(native.get(name, ['(none)']))}"
        for name, entry in _bash_tarballs(wheelhouse, "TARBALLS").items()
        if not name.startswith("OpenBLAS-") and native.get(name) != entry
    ]
    pinned = dict(_OPENBLAS_PIN.findall(openblas))
    for name, entry in _bash_tarballs(wheelhouse, "TARBALLS").items():
        release = re.fullmatch(r"OpenBLAS-([0-9.]+)\.tar\.gz", name)
        if release and pinned.get(release.group(1)) != entry[-1]:
            drift.append(
                f"{name} sha256: petsc_wheelhouse.sh {entry[-1]}, packages/openblas.sh "
                f"{pinned.get(release.group(1), '(no pin)')}"
            )
    return drift


def _defined(header: str, name: str) -> bool:
    return re.search(rf"^\s*#\s*define\s+{name}\b", header, re.MULTILINE) is not None


def _define_int(header: str, name: str) -> int | None:
    match = re.search(rf"^\s*#\s*define\s+{name}\s+(\d+)", header, re.MULTILINE)
    return int(match.group(1)) if match else None


#: The files ``petsc_mumps`` reads, as the PETSc recipe saves them.
PETSC_BUILD_FILES = (
    "petscconf.h",
    "mumps_int_def.h",
    "zmumps_c.h",
    "scotch.h",
    "Makefile.inc",
    "dynsym.txt",
    "needed.txt",
    "compiler.txt",
    "tarballs.txt",
)
#: PT-Scotch before 7.0.18 crashes MUMPS's parallel analysis (ADR 0067).
PTSCOTCH_MIN = (7, 0, 18)


def _petsc_tarballs(text: str, errors: list[str]) -> dict[str, tuple[str, str]]:
    """Version and sha256 of each ``--download`` tarball, from ``sha256sum`` lines."""
    tarballs = {}
    for line in text.splitlines():
        if line.strip():
            digest, path = line.split(None, 1)
            tarballs[Path(path.strip()).name] = digest
    found = {}
    for key, pattern in (
        ("mumps", r"MUMPS_([0-9.]+)\.tar\.gz"),
        ("scalapack", r"scalapack-v([0-9.]+)\.tar\.gz"),
        ("ptscotch", r"scotch-v([0-9.]+)\.tar\.gz"),
        ("metis", r"petsc-pkg-metis-v([0-9.]+-p[0-9]+)\.tar\.gz"),
    ):
        matches = [
            (match.group(1), digest)
            for name, digest in tarballs.items()
            if (match := re.fullmatch(pattern, name))
        ]
        if not matches:
            errors.append(f"no --download tarball matching {pattern}")
        found[key] = matches[0] if matches else ("", "")
    return found


#: ``petscconf.h`` defines gfem's PETSc must have (ADR 0067), with their gate.
PETSC_REQUIRED = (
    ("PETSC_USE_COMPLEX", "complex scalars"),
    ("PETSC_USE_64BIT_INDICES", "64-bit indices"),
    ("PETSC_HAVE_MUMPS", "MUMPS"),
    ("PETSC_HAVE_PTSCOTCH", "PT-Scotch"),
    ("PETSC_HAVE_METIS", "METIS"),
)
#: The MUMPS orderings a ``petscconf.h`` define brings (AMD and PORD are built in).
PETSC_ORDERINGS = (
    ("PETSC_HAVE_METIS", "metis"),
    ("PETSC_HAVE_PTSCOTCH", "ptscotch"),
    ("PETSC_HAVE_PARMETIS", "parmetis"),
)


def _petsc_conf(conf: str, errors: list[str]) -> list[str]:
    """ADR 0067's configure gates; returns the MUMPS orderings."""
    errors += [
        f"PETSc has no {gate} ({name}, ADR 0067)"
        for name, gate in PETSC_REQUIRED
        if not _defined(conf, name)
    ]
    if _defined(conf, "PETSC_HAVE_MPIUNI"):
        errors.append("PETSc is built without MPI (ADR 0209)")
    orderings = ["amd", "pord"] + [
        ordering for name, ordering in PETSC_ORDERINGS if _defined(conf, name)
    ]
    if "parmetis" in orderings:
        errors.append("PETSc has parmetis (ADR 0012, ADR 0067)")
    return sorted(orderings)


def _petsc_blas(needed: str, env: dict[str, str], errors: list[str]) -> dict:
    """The BLAS libpetsc links: the stack's, pthreads OpenBLAS or mkl_rt."""
    vendor = env.get("STACK_BLAS_VENDOR", "")
    version = env.get("STACK_BLAS_VERSION", "")
    threading = "unknown"
    if vendor == "openblas":
        if re.search(r"\blibopenblas_omp\.so", needed):
            threading = "openmp"
            errors.append("libpetsc links libopenblas_omp: gfem's PETSc takes pthreads")
        elif re.search(r"\blibopenblas\.so", needed):
            threading = "pthreads"
        else:
            errors.append("libpetsc links no libopenblas")
        lib = re.search(r"libopenblas\S*\.so\S*\s*=>\s*(\S+)", needed)
        if lib is None or f"openblas-{version}/" not in lib.group(1):
            errors.append(
                f"libpetsc's OpenBLAS is not the stack's openblas-{version}: "
                f"{lib.group(1) if lib else 'not found'}"
            )
    elif vendor == "mkl":
        threading = "mkl_rt"
        if "libmkl_rt.so" not in needed:
            errors.append("libpetsc links no libmkl_rt")
    return {"vendor": vendor, "version": version, "threading": threading}


def petsc_mumps(build: dict[str, str], env: dict[str, str]) -> tuple[dict, list[str]]:
    """The ``mumps`` block of a PETSc prefix, and the gates it fails.

    Besides the compared fields, the block records MUMPS's ``Makefile.inc``
    ``OPTF`` and ``FC`` as written (``fflags_raw``, ``compiler_raw``): for
    information, and ``check`` finds the toolchain's ``fflags`` in
    ``fflags_raw``.

    The gates are ADR 0067's (complex, 64-bit ``PetscInt``, MPI, no ParMETIS,
    PT-Scotch 7.0.18 or newer, no ``?gemmt_`` import) and ADR 0209's (MUMPS
    from the pinned tarball, the toolchain's Fortran flags, the stack's
    pthreads BLAS).

    Args:
        build: The text of each of ``PETSC_BUILD_FILES``: installed headers,
            MUMPS's ``Makefile.inc``, ``nm -D`` and ``ldd`` of the libraries
            MUMPS is folded into, ``<basename of FC> <version>``, and the
            ``sha256sum`` lines of the ``--download`` tarballs.
        env: The toolchain's ``FFLAGS``, ``STACK_BLAS_VENDOR`` and
            ``STACK_BLAS_VERSION``.

    Returns:
        The block, and one line per failed gate (empty when it passes).
    """
    errors: list[str] = []
    conf = build["petscconf.h"]
    tarballs = _petsc_tarballs(build["tarballs.txt"], errors)
    version, digest = tarballs["mumps"]
    header = re.search(r'MUMPS_VERSION\s+"([^"]+)"', build["zmumps_c.h"])
    installed = header.group(1) if header else None
    if installed != version:
        errors.append(f"MUMPS {installed} installed, the tarball is MUMPS {version}")
    gemmt = re.search(r"\b[sdcz]gemmt_\b", build["dynsym.txt"]) is not None
    if gemmt:
        errors.append("MUMPS imports ?gemmt_ (ADR 0205, ADR 0067)")
    arithmetics = sorted(set(re.findall(r"\bT ([sdcz])mumps_c\b", build["dynsym.txt"])))
    if "z" not in arithmetics:
        errors.append(f"no complex double MUMPS (zmumps_c) in {arithmetics}")
    orderings = _petsc_conf(conf, errors)
    scotch = ".".join(
        str(_define_int(build["scotch.h"], f"SCOTCH_{part}") or 0)
        for part in ("VERSION", "RELEASE", "PATCHLEVEL")
    )
    if (
        tuple(map(int, scotch.split("."))) < PTSCOTCH_MIN
        or scotch != tarballs["ptscotch"][0]
    ):
        errors.append(
            f"PT-Scotch {scotch} installed (tarball {tarballs['ptscotch'][0]}), "
            "ADR 0067 needs 7.0.18 or newer"
        )
    optf = re.search(r"^OPTF\s*=(.*)$", build["Makefile.inc"], re.MULTILINE)
    optf_flags = optf.group(1).split() if optf else []
    fc = re.search(r"^FC\s*=(.*)$", build["Makefile.inc"], re.MULTILINE)
    fflags = env.get("FFLAGS", "")
    missing = [flag for flag in fflags.split() if flag not in optf_flags]
    if missing:
        errors.append(f"MUMPS's OPTF {optf_flags} lacks the toolchain's {missing}")
    block = {
        "version": version,
        "tarball_sha256": digest,
        "intsize": 64 if _defined(build["mumps_int_def.h"], "MUMPS_INTSIZE64") else 32,
        "gemmt": gemmt,
        "blas": _petsc_blas(build["needed.txt"], env, errors),
        "scalapack": tarballs["scalapack"][0],
        "compiler": build["compiler.txt"].strip(),
        "fflags": fflags,
        "arithmetics": arithmetics,
        "openmp": _defined(conf, "PETSC_HAVE_OPENMP")
        or "libgomp" in build["needed.txt"],
        "orderings": orderings,
        "metis": tarballs["metis"][0],
        "ptscotch": tarballs["ptscotch"][0],
        "index_bits": 64 if _defined(conf, "PETSC_USE_64BIT_INDICES") else 32,
        "fflags_raw": " ".join(optf_flags),
        "compiler_raw": fc.group(1).strip() if fc else "",
    }
    return block, errors


#: The files of the recipe tree that every package build runs.
RECIPE_COMMON = ("stack.sh", "build.sbatch", "provenance.py")
#: Further files of the recipe tree that one package's build runs: Palace's
#: build reads its pins from the fetched tag with ``sources.py``.
RECIPE_EXTRA = {"palace": ("sources.py",)}


def recipe_sha256(root: Path, package: str) -> str:
    """The recipe hash of one package: what its build runs, by name and content.

    It covers ``stack.sh``, ``build.sbatch``, ``provenance.py``,
    ``packages/<package>.sh`` and the package's :data:`RECIPE_EXTRA`, and
    nothing else: the README, other packages' recipes and a shipped tree's
    ``COMMIT`` file leave it alone, so filling in another recipe does not make
    a built prefix differ. The toolchain has its own hash.
    """
    root = Path(root)
    digest = hashlib.sha256()
    extra = RECIPE_EXTRA.get(package, ())
    for relative in (*RECIPE_COMMON, f"packages/{package}.sh", *extra):
        digest.update(f"{sha256(root / relative)}  {relative}\n".encode())
    return digest.hexdigest()


def state(prefix: Path, recipe_hash: str, toolchain_hash: str) -> str:
    """What a build of ``prefix`` finds there (ADR 0191 idempotency).

    Returns:
        ``"absent"``: no prefix; ``"incomplete"``: a prefix without provenance
        file (a killed job's leftovers, to wipe); ``"complete"``: built from
        this recipe and toolchain (a no-op); ``"differs"``: built from another
        recipe or toolchain (refused without ``--rebuild``).
    """
    prefix = Path(prefix)
    if not prefix.is_dir():
        return "absent"
    record = read(prefix)
    if record is None:
        return "incomplete"
    if (
        record["recipe"].get("sha256") == recipe_hash
        and record["toolchain"]["sha256"] == toolchain_hash
    ):
        return "complete"
    return "differs"


def _first_line(*command: str) -> str | None:
    """The first output line of ``command``, or ``None`` if it cannot run."""
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0] if lines else None


def toolchain(toolchain_file: Path, modules_file: Path) -> dict:
    """The ``toolchain`` block of a build, read from the sourced environment.

    The toolchain file sets the compilers, flags and ``STACK_*`` variables
    (see the README); the hash of the frozen file is what ``check`` compares.
    """
    env = os.environ
    modules = [
        line.strip()
        for line in Path(modules_file).read_text().splitlines()
        if line.strip() and not line.strip().endswith(":")
    ]
    compilers = {}
    for var in ("CC", "CXX", "FC"):
        name = env.get(var)
        path = shutil.which(name) if name else None
        compilers[var] = {
            "command": name,
            "path": path,
            "version": _first_line(path, "--version") if path else None,
        }
    mpicc = env.get("MPICC", "mpicc")
    return {
        "sha256": sha256(Path(toolchain_file)),
        "stack": env.get("STACK_NAME"),
        "modules": modules,
        "compilers": compilers,
        "flags": {
            var: env.get(var)
            for var in ("CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS", "LDFLAGS")
        },
        "blas": {
            "vendor": env.get("STACK_BLAS_VENDOR"),
            "version": env.get("STACK_BLAS_VERSION"),
            "link": env.get("STACK_BLAS_LINK"),
            "threading": env.get("STACK_BLAS_THREADING"),
        },
        "scalapack": env.get("STACK_SCALAPACK"),
        "mpi": {
            "vendor": env.get("STACK_MPI_VENDOR"),
            "version": env.get("STACK_MPI_VERSION"),
            "show": _first_line(mpicc, "-show"),
        },
        "cmake": _first_line("cmake", "--version"),
        "python": env.get("STACK_PYTHON_VERSION"),
    }


def _write(prefix: str, meta: str) -> int:
    record = {
        path.stem: json.loads(path.read_text())
        for path in sorted(Path(meta).glob("*.json"))
    }
    print(write(Path(prefix), record))
    return 0


def _check(*stack_dirs: str) -> int:
    failed = False
    for stack_dir in stack_dirs:
        report = check_stack(Path(stack_dir))
        print(f"{Path(stack_dir).name}:")
        for note in report.notes:
            print(f"  {note}")
        for error in report.errors:
            print(f"  ERROR {error}")
        failed = failed or bool(report.errors)
    return 1 if failed else 0


def _petsc_contract(wheelhouse: str, recipe: str, openblas: str) -> int:
    drift = petsc_contract_drift(
        Path(wheelhouse).read_text(),
        Path(recipe).read_text(),
        Path(openblas).read_text(),
    )
    if not drift:
        print(f"petsc contract: matches {wheelhouse}")
        return 0
    for line in drift:
        print(f"  ERROR petsc contract drift (ADR 0209): {line}")
    return 1


def _petsc_mumps(prefix: str, build_dir: str) -> int:
    """``petsc-mumps``: headers from PREFIX/include, the rest from BUILD_DIR."""
    build = {}
    for name in PETSC_BUILD_FILES:
        path = Path(prefix) / "include" / name
        if not path.is_file():
            path = Path(build_dir) / name
        build[name] = path.read_text() if path.is_file() else ""
    block, errors = petsc_mumps(build, dict(os.environ))
    print(json.dumps(block, indent=2, sort_keys=True))
    for error in errors:
        print(f"provenance.py: petsc mumps: {error}", file=sys.stderr)
    return 1 if errors else 0


def _print(value: object) -> int:
    print(value)
    return 0


def _tofu(sums: str, path: str) -> int:
    try:
        outcome = tofu(Path(sums), Path(path))
    except ValueError as error:
        print(f"provenance.py: {error}", file=sys.stderr)
        return 1
    print(f"{Path(path).name}: sha256 {outcome}")
    return 0


#: The command line: name, number of arguments (None: one or more), handler.
COMMANDS: dict[str, tuple[int | None, Callable[..., int]]] = {
    "write": (2, _write),
    "check": (None, _check),
    "state": (3, lambda p, r, t: _print(state(Path(p), r, t))),
    "tofu": (2, _tofu),
    "toolchain": (
        2,
        lambda f, m: _print(json.dumps(toolchain(Path(f), Path(m)), indent=2)),
    ),
    "recipe-sha256": (2, lambda d, p: _print(recipe_sha256(Path(d), p))),
    "sha256": (1, lambda f: _print(sha256(Path(f)))),
    "petsc-contract": (3, _petsc_contract),
    "petsc-mumps": (2, _petsc_mumps),
}


def main(argv: list[str]) -> int:
    """Run one command of the command line."""
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    count, handler = COMMANDS[argv[0]]
    args = argv[1:]
    if (count is None and not args) or (count is not None and len(args) != count):
        print(__doc__, file=sys.stderr)
        return 2
    return handler(*args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
