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
"""

from __future__ import annotations

import hashlib
import json
import os
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


def _check_mumps(records: dict, report: Report) -> None:
    """Compare the ``mumps`` blocks of a stack's complete prefixes (ADR 0209)."""
    blocks = {}
    for name, record in sorted(records.items()):
        block = record.get("mumps")
        if block is None:
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
                if _field(a, dotted) != _field(b, dotted):
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


#: The files of the recipe tree that every package build runs.
RECIPE_COMMON = ("stack.sh", "build.sbatch", "provenance.py")


def recipe_sha256(root: Path, package: str) -> str:
    """The recipe hash of one package: what its build runs, by name and content.

    It covers ``stack.sh``, ``build.sbatch``, ``provenance.py`` and
    ``packages/<package>.sh``, and nothing else: the README, other packages'
    recipes and a shipped tree's ``COMMIT`` file leave it alone, so filling in
    another recipe does not make a built prefix differ. The toolchain has its
    own hash.
    """
    root = Path(root)
    digest = hashlib.sha256()
    for relative in (*RECIPE_COMMON, f"packages/{package}.sh"):
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
