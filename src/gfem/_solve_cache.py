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

A `Store` is the directory the user names. It holds a marker
(``solve-cache.json``: format, layout, ``max_bytes``), ``CACHEDIR.TAG``, a
``.gitignore``, a ``.tmp/`` area, and one **cache entry** per key: a
directory named by the key holding ``entry.json`` and, when the result has
arrays, ``arrays.npz``. An entry's fields refer to the model's mesh by its
topology digest and are read back onto the caller's ``model.mesh``. Entries
are built in ``.tmp/`` and published by one directory rename, so a reader
sees a whole entry or none; the least recently used leave once the entries
pass ``max_bytes``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import errno
import functools
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
import warnings
import zipfile
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

import gfem
from gfem import _codec, errors
from gfem.errors import SolveCacheWarning
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


# --- the store ------------------------------------------------------------------

#: The marker's format name, and the layout this gfem reads and writes.
CACHE_FORMAT = "gfem solve cache"
LAYOUT = 1
#: An entry's format name: not a result bundle's, so `gfem.load_result`
#: refuses it.
ENTRY_FORMAT = "gfem solve cache entry"
#: The size cap a new cache gets: 10 GiB.
DEFAULT_MAX_BYTES = 10 * 1024**3

MARKER = "solve-cache.json"
ENTRY_JSON = "entry.json"
ENTRY_NPZ = "arrays.npz"
TMP = ".tmp"
_CACHEDIR_TAG = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by gfem's solve cache.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)
#: An entry's name: a solve key.
_KEY = re.compile(r"[0-9a-f]{64}\Z")
#: ``.tmp/`` children older than this (crashed writers) are removed.
_TMP_AGE_S = 24 * 3600
#: What reading a damaged entry raises; ``JSONDecodeError`` is a ``ValueError``.
_READ_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError)

_log = logging.getLogger("gfem.solve_cache")


class Store:
    """A solve cache directory: entries read, written and evicted.

    A path that does not exist, or an empty directory, becomes a cache. A
    cache of another layout warns once and is left alone: `read` misses and
    `write` stores nothing (`usable` is False).

    Every method that warns takes ``stacklevel``, as `warnings.warn` does
    for a warning raised in the method's caller (1: the calling line).

    Attributes:
        root: The directory.
        usable: False for a cache of another layout.
        max_bytes: The cap on the entries' total size, from the marker.

    Raises:
        ValueError: A non-empty directory without a readable marker, or a
            marker whose ``max_bytes`` is not a whole number of bytes.
        OSError: The directory cannot be created or initialized.
    """

    def __init__(self, directory: str | os.PathLike, *, stacklevel: int = 1) -> None:
        """Open, or create, the cache at ``directory``."""
        self.root = Path(directory)
        self.usable = True
        self.max_bytes = DEFAULT_MAX_BYTES
        if not self.root.is_dir() or not _occupied(self.root):
            self.root.mkdir(parents=True, exist_ok=True)
            self._initialize()
        marker = self._marker()
        if marker["layout"] != LAYOUT:
            self.usable = False
            warnings.warn(
                f"{self.root} is a solve cache of another layout "
                f"({marker['layout']}, this gfem writes {LAYOUT}), so this call "
                "neither reads nor stores; the result is correct. Name another "
                "directory, or delete this one.",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return
        self.max_bytes = marker["max_bytes"]

    def _initialize(self) -> None:
        """Write the marker (atomically), then the tag, ``.gitignore`` and ``.tmp/``."""
        text = json.dumps(
            {"format": CACHE_FORMAT, "layout": LAYOUT, "max_bytes": DEFAULT_MAX_BYTES}
        )
        fd, tmp = tempfile.mkstemp(prefix=MARKER + ".", dir=self.root)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text + "\n")
            Path(tmp).replace(self.root / MARKER)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        (self.root / "CACHEDIR.TAG").write_text(_CACHEDIR_TAG)
        (self.root / ".gitignore").write_text("*\n")
        (self.root / TMP).mkdir(exist_ok=True)

    def _marker(self) -> dict[str, Any]:
        path = self.root / MARKER
        try:
            marker = json.loads(path.read_text())
        except (OSError, ValueError):
            marker = None
        if (
            not isinstance(marker, dict)
            or marker.get("format") != CACHE_FORMAT
            or not _whole(marker.get("layout"))
        ):
            raise ValueError(
                f"{self.root} is not empty and is not a gfem solve cache (no "
                f"readable {MARKER}); gfem writes a cache only into a new or "
                "empty directory. Name another directory for cache=."
            )
        if marker["layout"] == LAYOUT and not (
            _whole(marker.get("max_bytes")) and marker["max_bytes"] >= 0
        ):
            raise ValueError(
                f"{path} has max_bytes {marker.get('max_bytes')!r}: give the "
                "solve cache's size cap as a whole number of bytes."
            )
        return marker

    def read(
        self, key: str, model: Any, *, stacklevel: int = 1
    ) -> tuple[Any, dict[str, Any], tuple[tuple[type[Warning], str], ...]] | None:
        """The entry of ``key``, its fields on ``model.mesh``; None on a miss.

        A hit marks the entry as used now. An entry that cannot be read (a
        damaged file, a key or mesh that does not match) is removed, with a
        `gfem.SolveCacheWarning`, and is a miss.

        Args:
            key: The solve key.
            model: The model of the call: the entry's fields come back on
                its mesh object.
            stacklevel: Where the warning points, as for `warnings.warn`.

        Returns:
            ``(result, run, recorded)``: the result, the solving run's record
            (``device``, ``options``, ``ranks``, ``wall_time_s``) and the
            warnings recorded while solving, as ``(category, message)``.
        """
        if not self.usable:
            return None
        entry = self.root / key
        if not entry.is_dir():
            return None
        try:
            found = _load(entry, key, model)
        except _READ_ERRORS as err:
            if not entry.exists():
                return None  # evicted while being read
            _discard(self.root, entry)
            warnings.warn(
                f"The solve cache entry {entry} could not be read "
                f"({type(err).__name__}: {err}), so it was removed and this call "
                "solved again; the new result is stored. Entries are written "
                "whole, so something outside gfem changed it (a full disk, a "
                "sync tool, a hand edit).",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return None
        with contextlib.suppress(OSError):  # a read-only cache still serves hits
            os.utime(entry / ENTRY_JSON)
        return found

    def write(  # noqa: PLR0917
        self,
        key: str,
        model: Any,
        env: dict[str, str | None],
        result: Any,
        run: dict[str, Any],
        recorded: Sequence[tuple[type[Warning], str]],
        *,
        stacklevel: int = 1,
    ) -> Path | str:
        """Store ``result`` under ``key``, then evict down to ``max_bytes``.

        A result that cannot be stored (a failed write, or larger than
        ``max_bytes``) warns with `gfem.SolveCacheWarning` and is not stored;
        nothing is raised. When another process publishes the same key first,
        its entry stays.

        Args:
            key: The solve key.
            model: The model solved: fields on its mesh refer to it.
            env: The environment that entered the key.
            result: The result.
            run: The solving run's record: ``device``, ``options``,
                ``ranks``, ``wall_time_s``.
            recorded: The warnings to replay on a hit, as
                ``(category, message)``; categories are `gfem.errors` classes.
            stacklevel: Where the warning points, as for `warnings.warn`.

        Returns:
            The entry's directory, or why nothing was stored.
        """
        if not self.usable:
            return "a solve cache of another layout"
        encoder = _codec.Encoder(_codec.Context(self.root / key, model_mesh=model.mesh))
        doc: dict[str, Any] = {
            "format": ENTRY_FORMAT,
            "layout": LAYOUT,
            "key": key,
            "env": env,
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "bytes": 0,
            "arrays": None,
            "run": encoder.encode(run, ".run"),
            "warnings": [{"category": c.__name__, "message": m} for c, m in recorded],
            "result": encoder.encode(result),
        }
        size = sum(a.nbytes for a in encoder.arrays.values())
        target = self.root / key
        if size <= self.max_bytes:
            try:
                size = _publish(self.root, target, doc, encoder.arrays, self.max_bytes)
            except OSError as err:
                warnings.warn(
                    f"This solve's result could not be stored in the solve cache "
                    f"{self.root} ({type(err).__name__}: {err}); the result is "
                    "correct. Check that the directory is writable and has space.",
                    SolveCacheWarning,
                    stacklevel=stacklevel + 1,
                )
                return f"the write failed: {err}"
        if size > self.max_bytes:
            warnings.warn(
                f"This solve's result ({_size(size)}) is larger than the solve "
                f"cache's limit ({_size(self.max_bytes)}) and was not stored; the "
                f"result is correct. Raise max_bytes in {self.root / MARKER} to "
                "keep results this large.",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return "larger than the cache's max_bytes"
        self._evict()
        _sweep(self.root / TMP)
        return target

    def _evict(self) -> None:
        """Remove the least recently used entries until within ``max_bytes``."""
        entries = _entries(self.root)
        total = sum(size for _, size, _ in entries)
        evicted = freed = 0
        for _, size, path in sorted(entries, key=lambda e: e[0]):
            if total <= self.max_bytes:
                break
            if _discard(self.root, path):
                total -= size
                evicted += 1
                freed += size
        if evicted:
            _log.info("evicted %d entries, %s", evicted, _size(freed))


def _publish(
    root: Path,
    target: Path,
    doc: dict[str, Any],
    arrays: dict[str, np.ndarray],
    max_bytes: int,
) -> int:
    """Build the entry in ``.tmp/`` and rename it to ``target``; its size.

    An entry larger than ``max_bytes`` is built, measured and dropped. A
    rename that finds ``target`` taken leaves the other writer's entry.
    """
    (root / TMP).mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=root / TMP))
    try:
        size = _write_entry(tmp, doc, arrays)
        if size <= max_bytes:
            try:
                tmp.rename(target)
            except OSError as err:
                if err.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return size


def _entries(root: Path) -> list[tuple[float, int, Path]]:
    """Every readable entry as ``(last used, bytes, directory)``."""
    entries = []
    try:
        scan = list(os.scandir(root))
    except OSError:
        return []
    for item in scan:
        if not _KEY.match(item.name) or not item.is_dir(follow_symlinks=False):
            continue
        path = Path(item.path) / ENTRY_JSON
        try:
            used = path.stat().st_mtime
            size = json.loads(path.read_text())["bytes"]
        except (OSError, ValueError, KeyError, TypeError):
            continue  # being replaced, or damaged: its reader removes it
        if _whole(size):
            entries.append((used, size, Path(item.path)))
    return entries


def _sweep(tmp: Path) -> None:
    """Remove what crashed writers left in ``.tmp/`` more than a day ago."""
    try:
        children = list(tmp.iterdir())
    except OSError:
        return
    now = time.time()
    for child in children:
        try:
            if now - child.lstat().st_mtime < _TMP_AGE_S:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except OSError:
            pass


def _occupied(root: Path) -> bool:
    """Whether ``root`` holds anything besides a marker being written."""
    return any(not p.name.startswith(MARKER + ".") for p in root.iterdir())


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load(
    entry: Path, key: str, model: Any
) -> tuple[Any, dict[str, Any], tuple[tuple[type[Warning], str], ...]]:
    """Read and decode one entry, raising one of `_READ_ERRORS` if it is damaged."""
    doc = json.loads((entry / ENTRY_JSON).read_text())
    if not isinstance(doc, dict) or doc.get("format") != ENTRY_FORMAT:
        raise ValueError(f"{entry / ENTRY_JSON} is not a solve cache entry.")
    if doc.get("layout") != LAYOUT:
        raise ValueError(f"the entry has layout {doc.get('layout')!r}, not {LAYOUT}")
    if doc.get("key") != key:
        raise ValueError(f"the entry holds key {doc.get('key')!r}, not its name")
    arrays: dict[str, np.ndarray] = {}
    if doc["arrays"]:
        with np.load(entry / ENTRY_NPZ, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    context = _codec.Context(entry, model_mesh=model.mesh)
    decoder = _codec.Decoder(context, arrays, path=entry / ENTRY_JSON)
    run = decoder.decode(doc["run"])
    if not isinstance(run, dict):
        raise ValueError("the entry's run is not a record")  # noqa: TRY004
    recorded = tuple(
        (_category(w["category"]), str(w["message"])) for w in doc["warnings"]
    )
    return decoder.decode(doc["result"]), run, recorded


def _category(name: Any) -> type[Warning]:
    """A gfem warning class by its name."""
    cls = getattr(errors, str(name), None)
    if not (isinstance(cls, type) and issubclass(cls, Warning)):
        raise ValueError(f"the entry records an unknown warning class {name!r}")  # noqa: TRY004
    return cls


def _write_entry(tmp: Path, doc: dict[str, Any], arrays: dict[str, np.ndarray]) -> int:
    """Write ``entry.json`` (and ``arrays.npz``) into ``tmp``; their total size.

    ``bytes`` in ``entry.json`` is that total, the document's own size included.
    """
    size = 0
    if arrays:
        with (tmp / ENTRY_NPZ).open("wb") as f:
            np.savez(f, allow_pickle=False, **arrays)
        doc["arrays"] = ENTRY_NPZ
        size = (tmp / ENTRY_NPZ).stat().st_size
    total = size
    while True:  # the size's digits change the size: at most a few rounds
        doc["bytes"] = total
        data = (json.dumps(doc, indent=1, allow_nan=False) + "\n").encode()
        if size + len(data) == total:
            break
        total = size + len(data)
    (tmp / ENTRY_JSON).write_bytes(data)
    return total


def _discard(root: Path, entry: Path) -> bool:
    """Remove an entry: renamed into ``.tmp/`` first, so no reader sees half of it."""
    dead = root / TMP / f"{entry.name}.{secrets.token_hex(8)}.old"
    try:
        (root / TMP).mkdir(exist_ok=True)
        entry.rename(dead)
    except OSError:
        return False
    shutil.rmtree(dead, ignore_errors=True)
    return True


def _size(n: int) -> str:
    """A byte count for people: ``"4.5 MB"``."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"
