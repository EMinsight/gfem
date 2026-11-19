"""Case data: staging pinned meshes into the site's data directory.

A suite case's mesh is pinned by the sha256 of its uncompressed ``.msh`` (ADR 0405).
Its bytes are committed beside the case (``[mesh] file``, an ``.xz``) or, when
external, held as ``<sha256>.msh.xz`` in the mesh stores the site file lists
(``[data] stores``, searched in order; ADR 0407).

The prepare step stages each planned case's mesh into the data directory, a cache,
as ``<data dir>/meshes/<sha256>.msh``: decompressed into a temp file, hashed while
written, renamed into place only if it matches, then made read-only. ``step``
re-verifies the staged file before every run. A mismatch is never repaired: the
bytes are kept aside as ``<sha256>.bad-<got12>`` and the run is refused.
Standard library only (ADR 0400).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from benchmarks.harness.case import Case, open_mesh
from benchmarks.harness.records import file_sha256

_CHUNK = 1 << 20


class Refused(Exception):  # noqa: N818 (the record status it leads to)
    """A mesh that cannot be staged or verified; its runs are ``refused``."""


def staged_path(data_dir: Path, sha256: str) -> Path:
    """Where the mesh pinned to ``sha256`` is staged."""
    return data_dir / "meshes" / f"{sha256}.msh"


def _quarantine(path: Path, pinned: str, got: str) -> Path:
    bad = path.with_name(f"{pinned}.bad-{got[:12]}")
    path.replace(bad)
    return bad


def _mismatch(source: Path, got: str, pinned: str, bad: Path) -> Refused:
    return Refused(
        f"mesh hash: {source} gives {got[:12]}, pinned {pinned[:12]} "
        f"(kept as {bad.name})"
    )


def sources(case: Case, stores: Sequence[Path]) -> list[Path]:
    """Candidate files of a case's mesh in search order: committed, then stores."""
    sha = case.mesh.sha256
    found = []
    if case.mesh.file is not None and case.directory is not None:
        found.append(case.directory / case.mesh.file)
    for store in stores:
        found += [store / f"{sha}.msh.xz", store / f"{sha}.msh"]
    return found


def _install(source: Path, target: Path, pinned: str) -> None:
    tmp = target.with_name(f".{target.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    try:
        with open_mesh(source) as src, tmp.open("wb") as dst:
            for block in iter(lambda: src.read(_CHUNK), b""):
                digest.update(block)
                dst.write(block)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    got = digest.hexdigest()
    if got != pinned:
        raise _mismatch(source, got, pinned, _quarantine(tmp, pinned, got))
    tmp.chmod(0o444)
    tmp.replace(target)


def stage(case: Case, data_dir: Path, stores: Sequence[Path]) -> Path:
    """Stage ``case``'s mesh into ``data_dir`` (if not already) and return its path.

    A staged file that no longer verifies is moved aside and restaged. Raises
    :class:`Refused` when the source gives another hash (quarantined) or when the
    mesh is neither committed nor in any of ``stores`` (named in the reason).
    """
    pinned = case.mesh.sha256
    target = staged_path(data_dir, pinned)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        got = file_sha256(target)
        if got == pinned:
            return target
        _quarantine(target, pinned, got)
    for source in sources(case, stores):
        if source.exists():
            _install(source, target, pinned)
            return target
    searched = ", ".join(str(s) for s in stores) or "none"
    msg = (
        f"mesh {pinned[:12]} of {case.name}: not committed and in no mesh store "
        f"(searched: {searched})"
    )
    raise Refused(msg)


def verify_staged(path: Path, pinned: str) -> None:
    """Re-verify a staged mesh before a run; a mismatch is moved aside and refused."""
    if not path.exists():
        msg = f"mesh {pinned[:12]}: not staged at {path}"
        raise Refused(msg)
    got = file_sha256(path)
    if got != pinned:
        raise _mismatch(path, got, pinned, _quarantine(path, pinned, got))
