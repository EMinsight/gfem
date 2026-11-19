"""The solve cache: a finished result stored under its solve key, read back.

The **solve key** of a solve is the sha256 (64 hex characters) of everything
the result depends on: the model, the problem, the solver options that change
the numerical method or its stopping rule, and the environment that computes
it (gfem's version and code, NGSolve's version, and PETSc's or SLEPc's when
they solve). Two calls with the same key give the same result to gfem's
documented precision.

The key hashes, in this order:

1. A compact JSON document ``{"key_version", "schema_version", "env",
   "job"}`` (``separators=(",", ":")``, ASCII, no NaN, mappings in insertion
   order). ``job`` is the `gfem._codec` encoding of the job under
   ``Context(key=True)``, which writes a mesh by its in-memory topology
   (`gfem.mesh._topology_sha256`) instead of its file; the encoder's
   ``numpy`` flags are dropped and a scalar $-0$ is written as $0$.
2. Every array of the encoder's table, sorted by leaf path, as
   `gfem.mesh._hash_array` feeds it.

The options that say only where and how fast a solve runs (``threads``,
``frequency_workers``, ``retain_memory``, ``device``) are reset to their
defaults first, so a result computed on one machine serves another.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gfem
from gfem import _codec
from gfem.job import Job
from gfem.mesh import _hash_array
from gfem.options import SolverOptions

#: The version of the key's rules. Bump it when anything the key hashes, or
#: how, changes: entries of an older version then miss.
KEY_VERSION = 1

#: The options reset before keying: they change where and how fast a solve
#: runs, not its result beyond documented precision.
_PLACEMENT = {
    "threads": None,
    "frequency_workers": 1,
    "retain_memory": None,
    "device": "cpu",
}


def solve_key(
    model: Any, problem: Any, options: SolverOptions
) -> tuple[str, dict[str, str | None]]:
    """The solve key of a solve, and the environment it includes.

    Args:
        model: The model.
        problem: The problem.
        options: The solver options.

    Returns:
        ``(key, env)``: the sha256 as 64 hex characters, and the environment
        record (package versions and gfem's code digest) that entered it.

    Raises:
        gfem._codec.UnencodableError: An input of a type gfem cannot write.
    """
    env = _env(options)
    job = Job(model, problem, dataclasses.replace(options, **_PLACEMENT))
    encoder = _codec.Encoder(_codec.Context(Path("/"), key=True))
    doc = {
        "key_version": KEY_VERSION,
        "schema_version": _codec.SCHEMA_VERSION,
        "env": env,
        "job": _canonical(encoder.encode(job)),
    }
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    h = hashlib.sha256(text.encode("ascii"))
    for path in sorted(encoder.arrays):
        _hash_array(h, path, encoder.arrays[path])
    return h.hexdigest(), env


def _canonical(node: Any) -> Any:
    """The encoded JSON without ``numpy`` flags, with $-0$ written as $0$."""
    if isinstance(node, list):
        return [_canonical(v) for v in node]
    if isinstance(node, float):
        return node + 0.0
    if isinstance(node, dict):
        array = "$array" in node
        return {
            k: _canonical(v) for k, v in node.items() if not (array and k == "numpy")
        }
    return node


def _env(options: SolverOptions) -> dict[str, str | None]:
    env: dict[str, str | None] = {
        "gfem": gfem.__version__,
        "gfem_code": _gfem_code(),
        "ngsolve": _version("ngsolve"),
    }
    if options.linear_solver == "petsc":
        env["petsc4py"] = _version("petsc4py")
    if options.eigen_solver == "slepc":
        env["slepc4py"] = _version("slepc4py")
    return env


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@functools.cache
def _gfem_code() -> str:
    """The sha256 of the installed gfem's ``.py`` files: paths, then bytes."""
    root = Path(gfem.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        h.update(len(name).to_bytes(8, "little"))
        h.update(name)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()
