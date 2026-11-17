"""``ship REV --to HOST:ROOT``: a committed revision to the cluster (ADRs 0400, 0407).

``git archive REV`` into a temp tree, ``COMMIT`` (full sha) and ``COMMIT_UTC`` (the
commit date) beside it, ``uv build --wheel`` into its ``dist/``, then ``rsync -a`` to
``ROOT/refs/<sha7>/``: mn5run's layout, so a commit shipped by either tool is reused
by the other (skipped when ``ROOT/refs/<sha7>/dist/gfem-*.whl`` already exists).
Only committed content ships.

External meshes (cases without a committed ``[mesh] file``) go to the site's
``[ship] mesh_store`` with ``rsync --ignore-existing``, each verified against its pin
first from the local mesh stores; a store file with another hash fails ``ship``.

``fetch`` is :func:`benchmarks.harness.records.fetch`. Standard library only.
"""

from __future__ import annotations

import datetime as dt
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness import config
from benchmarks.harness.case import Case, load_case, mesh_digest
from benchmarks.harness.config import ConfigError, Runner

ROOT = config.PACKAGE.parent
SSH = ("ssh", "-o", "LogLevel=ERROR")


class ShipError(ConfigError):
    """A revision or mesh that ``ship`` refuses to send."""


@dataclass(frozen=True)
class Target:
    """``HOST:ROOT``."""

    host: str
    root: str

    @classmethod
    def parse(cls, text: str) -> Target:
        """``host:/some/root`` -> ``Target("host", "/some/root")``."""
        host, sep, root = text.partition(":")
        if not sep or not host or not root:
            raise ShipError(f"--to {text!r} is not HOST:ROOT")
        return cls(host, root.rstrip("/"))

    def refs(self, sha7: str) -> str:
        """``ROOT/refs/<sha7>``."""
        return f"{self.root}/refs/{sha7}"


def _git(repo: Path, *args: str, run: Runner = subprocess.run) -> str:
    done = run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise ShipError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def commit_of(repo: Path, rev: str) -> tuple[str, str]:
    """``(full sha, commit date UTC)`` of ``rev``."""
    out = _git(repo, "show", "-s", "--format=%H %cI", f"{rev}^{{commit}}").split()
    when = dt.datetime.fromisoformat(out[1]).astimezone(dt.UTC)
    return out[0], when.strftime("%Y-%m-%dT%H:%M:%SZ")


def files(repo: Path, sha: str) -> list[str]:
    """The files ``git archive`` ships, plus ``COMMIT`` and ``COMMIT_UTC``."""
    listed = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
    return sorted([*listed, "COMMIT", "COMMIT_UTC"])


def external_cases(
    cases_dir: Path = config.CASES_DIR, references_dir: Path | None = None
) -> list[Case]:
    """Cases whose mesh is not committed beside them (``[mesh] file`` absent).

    With ``references_dir``, the reference-only rungs under it too
    (``<family>/<rung>/case.toml``); one case per mesh.
    """
    paths = sorted(cases_dir.glob("*/case.toml"))
    if references_dir is not None:
        paths += sorted(references_dir.glob("*/*/case.toml"))
    found: dict[str, Case] = {}
    for path in paths:
        case = load_case(path)
        if case.mesh.file is None:
            found.setdefault(case.mesh.sha256, case)
    return list(found.values())


def mesh_sources(cases: Sequence[Case], stores: Sequence[Path]) -> list[Path]:
    """Each external case's ``<sha256>.msh.xz``, verified, from the local stores."""
    out = []
    for case in cases:
        sha = case.mesh.sha256
        candidates = [s / f"{sha}.msh.xz" for s in stores]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ShipError(
                f"mesh {sha[:12]} of {case.name} is in no local store "
                f"({', '.join(map(str, stores)) or 'none given'})"
            )
        got, _ = mesh_digest(source)
        if got != sha:
            raise ShipError(
                f"mesh hash: {source} gives {got[:12]}, pinned {sha[:12]}; "
                "fix the store, never the pin"
            )
        out.append(source)
    return out


def ship(
    rev: str,
    target: Target,
    *,
    repo: Path = ROOT,
    mesh_store: str | None = None,
    stores: Sequence[Path] = (),
    cases_dir: Path = config.CASES_DIR,
    references_dir: Path | None = config.REFERENCES_DIR,
    dry_run: bool = False,
    run: Runner = subprocess.run,
) -> int:
    """Ship ``rev`` to ``target``; on a dry run print what would go, run nothing."""
    sha, when = commit_of(repo, rev)
    sha7 = sha[:7]
    dest = target.refs(sha7)
    rsync_e = ("-e", shlex.join(SSH))
    if dry_run:
        print(f"ship {sha} ({when}) -> {target.host}:{dest}/")
        for name in files(repo, sha):
            print(f"  {name}")
        print("  dist/gfem-*.whl (uv build --wheel)")
    else:
        found = run(
            [*SSH, target.host, f"ls {dest}/dist/gfem-*.whl"],
            capture_output=True,
            text=True,
            check=False,
        )
        if found.returncode == 0 and found.stdout.strip():
            print(f"{target.host}:{dest} already holds {sha7}'s wheel: not shipped")
            # mn5run ships the same layout without the commit date: add it.
            utc = shlex.quote(f"{dest}/COMMIT_UTC")
            run(
                [*SSH, target.host, f"test -s {utc} || echo {when} > {utc}"],
                check=False,
            )
        else:
            _ship_tree(
                repo,
                sha=sha,
                when=when,
                dest=f"{target.host}:{dest}/",
                rsync_e=rsync_e,
                run=run,
            )
            print(f"shipped {sha7} to {target.host}:{dest}")
    externals = external_cases(cases_dir, references_dir)
    if not externals:
        return 0
    if mesh_store is None:
        print("no [ship] mesh_store (pass --site): external meshes not shipped")
        return 0
    sources = mesh_sources(externals, stores)
    argv = [
        "rsync",
        "-a",
        "--ignore-existing",
        *rsync_e,
        *map(str, sources),
        f"{target.host}:{mesh_store}/",
    ]
    if dry_run:
        print(shlex.join(argv))
        return 0
    done = run(argv, check=False)
    if done.returncode != 0:
        print(f"ship: mesh rsync exited {done.returncode}", file=sys.stderr)
        return done.returncode
    return 0


def _ship_tree(
    repo: Path,
    *,
    sha: str,
    when: str,
    dest: str,
    rsync_e: tuple[str, ...],
    run: Runner,
) -> None:
    with tempfile.TemporaryDirectory(prefix="bench-ship-") as tmp:
        tree = Path(tmp) / "tree"
        tree.mkdir()
        archive = Path(tmp) / "tree.tar"
        _git(repo, "archive", "--format=tar", "-o", str(archive), sha, run=run)
        done = run(["tar", "-x", "-f", str(archive), "-C", str(tree)], check=False)
        if done.returncode != 0:
            raise ShipError("tar -x of the archive failed")
        (tree / "COMMIT").write_text(sha + "\n")
        (tree / "COMMIT_UTC").write_text(when + "\n")
        done = run(
            ["uv", "build", "--wheel", "-q", "-o", "dist"], cwd=tree, check=False
        )
        if done.returncode != 0:
            raise ShipError("uv build --wheel failed")
        done = run(["rsync", "-a", "--delete", *rsync_e, f"{tree}/", dest], check=False)
        if done.returncode != 0:
            raise ShipError(f"rsync to {dest} exited {done.returncode}")
