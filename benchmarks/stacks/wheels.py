"""The ``wheels`` stack kind: pip wheels installed offline into per-commit envs.

Site inputs come from the site file's ``[stacks.<name>]`` table: ``base_venv`` (a
venv with gfem's dependencies), ``wheel_dirs``, ``petsc_wheelhouse`` (the MPI PETSc
stack's wheels), ``palace_wheelhouse`` and ``python`` (which makes the Palace
venvs). :func:`prepare` ports ``tools/scale_env.sh``: gfem's env is a hard-link
copy of the base venv with the commit's wheel, the PETSc wheels and mpi4py
installed offline, at ``{cluster.root}/envs/bench-<sha7>-<stack>-<fp8>/``; Palace's
env is a fresh venv per version at ``{cluster.root}/envs/palace-<version>/``,
shared across commits. ``manifest.txt``, written last, marks an env finished.
The ``palace`` version selects the wheel; ``ngsolve`` and ``petsc`` are checked
against ``pip freeze`` and a mismatch is refused. Standard library only.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.harness.config import ConfigError

if TYPE_CHECKING:
    from benchmarks.harness.config import Site
    from benchmarks.stacks import Description, Resolved

KEYS = ("base_venv", "wheel_dirs", "petsc_wheelhouse", "palace_wheelhouse", "python")
MPI4PY = "mpi4py==4.1.2"
CHECKED = ("ngsolve", "petsc")
"""Versions the site's base venv and wheelhouse fix: checked, never chosen."""
MANIFEST = "manifest.txt"
_PIP_ENV = {"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}


def inputs(
    description: Description, _versions: dict[str, str], site: Site
) -> tuple[dict[str, Any], dict[str, str]]:
    """The site's ``[stacks.<name>]`` table and a placeholder per path in it."""
    name = description.name
    table = site.stacks.get(name)
    if table is None:
        raise ConfigError(f"{site.path}: no [stacks.{name}] table for stack {name}")
    missing = [k for k in KEYS if k not in table]
    if missing:
        raise ConfigError(f"{site.path} [stacks.{name}]: missing {', '.join(missing)}")
    roots: dict[str, str] = {}
    for key, value in table.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                roots[f"{{stacks.{name}.{key}[{i}]}}"] = item
        else:
            roots[f"{{stacks.{name}.{key}}}"] = value
    return {k: table[k] for k in KEYS}, roots


def paths(
    _description: Description, versions: dict[str, str], site: Site, env_dir: str
) -> dict[str, Any]:
    """The env of each code: the stack's env dir, Palace's shared per version."""
    palace = f"{site.cluster_root}/envs/palace-{versions['palace']}"
    return {"code_env": {"gfem": env_dir, "palace": palace}}


def activation(resolved: Resolved, code: str) -> list[str]:
    """One ``module load`` per module, then the code's venv ``bin`` on ``PATH``."""
    section = resolved.description.codes[code]
    lines = [f"module load {module}" for module in section.modules]
    lines.append(f'export PATH={resolved.code_env[code]}/bin:"$PATH"')
    return lines


def refusal(_resolved: Resolved) -> str | None:
    """Nothing to refuse before prepare: the envs are built from the site's inputs."""
    return None


def _refused(message: str) -> Exception:
    from benchmarks.stacks import StackRefusedError

    return StackRefusedError(message)


def _run(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env={**os.environ, **_PIP_ENV},
        check=False,
    )
    if result.returncode != 0:
        raise _refused(
            f"{' '.join(argv)} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_versions(freeze: str) -> dict[str, str]:
    """``name==version`` lines of ``pip freeze``, names normalised."""
    versions = {}
    for line in freeze.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)$", line.strip())
        if match:
            versions[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return versions


def _check_versions(resolved: Resolved, freeze: str, env: Path) -> None:
    found = frozen_versions(freeze)
    for package in CHECKED:
        want = resolved.versions[package]
        got = found.get(package)
        if got != want:
            raise _refused(
                f"stack {resolved.name}: {package} {got or 'absent'} in {env}, "
                f"the stack pins {want} "
                "(a new base venv or wheelhouse is a site action)"
            )


def _links(resolved: Resolved) -> list[str]:
    links = ["--find-links", resolved.inputs["petsc_wheelhouse"]]
    for directory in resolved.inputs["wheel_dirs"]:
        links += ["--find-links", directory]
    return links


def _gfem_env(resolved: Resolved, wheel: Path) -> Path:
    env = Path(resolved.env_dir)
    if (env / MANIFEST).is_file():
        _check_versions(resolved, (env / MANIFEST).read_text(), env)
        return env
    if env.exists():
        shutil.rmtree(env)
    env.parent.mkdir(parents=True, exist_ok=True)
    base = resolved.inputs["base_venv"]
    _run(["cp", "-al", base, str(env)])
    python = str(env / "bin" / "python")
    links = _links(resolved)
    petsc = sorted(Path(resolved.inputs["petsc_wheelhouse"]).glob("*.whl"))
    _run([python, "-m", "pip", "uninstall", "-y", "-q", "gfem"])
    _run([python, "-m", "pip", "install", "-q", "--no-index", *links, str(wheel)])
    _run([python, "-m", "pip", "install", "-q", "--no-index", *links, *map(str, petsc)])
    _run([python, "-m", "pip", "install", "-q", "--no-index", *links, MPI4PY])
    freeze = _run([python, "-m", "pip", "freeze"])
    try:
        _check_versions(resolved, freeze, env)
    except Exception:
        shutil.rmtree(env)
        raise
    lines = [f"commit {resolved.commit}", f"{_sha256(wheel)}  {wheel}"]
    lines += [f"{_sha256(p)}  {p}" for p in petsc]
    (env / MANIFEST).write_text("\n".join(lines) + "\n" + freeze)
    return env


def palace_wheel(resolved: Resolved) -> Path:
    """The one ``palace_solver-<version>-*.whl`` of the Palace wheelhouse."""
    version = resolved.versions["palace"]
    house = Path(resolved.inputs["palace_wheelhouse"])
    found = sorted(house.glob(f"palace_solver-{version}-*.whl"))
    if len(found) != 1:
        what = "no" if not found else f"{len(found)}"
        raise _refused(
            f"stack {resolved.name}: {what} palace_solver-{version}-*.whl in {house}"
        )
    return found[0]


def _palace_env(resolved: Resolved) -> Path:
    env = Path(resolved.code_env["palace"])
    if (env / MANIFEST).is_file():
        return env
    wheel = palace_wheel(resolved)
    if env.exists():
        shutil.rmtree(env)
    env.parent.mkdir(parents=True, exist_ok=True)
    _run([resolved.inputs["python"], "-m", "venv", str(env)])
    python = str(env / "bin" / "python")
    house = resolved.inputs["palace_wheelhouse"]
    links = ["--find-links", house, *_links(resolved)]
    _run([python, "-m", "pip", "install", "-q", "--no-index", *links, str(wheel)])
    version = _run([str(env / "bin" / "palace"), "--version"]).strip().splitlines()
    freeze = _run([python, "-m", "pip", "freeze"])
    lines = [
        f"{_sha256(wheel)}  {wheel}",
        f"palace --version: {version[0] if version else ''}",
    ]
    (env / MANIFEST).write_text("\n".join(lines) + "\n" + freeze)
    return env


def prepare(resolved: Resolved, *, wheel: Path) -> Path:
    """Build (or reuse) gfem's env from ``wheel`` and Palace's env; return gfem's."""
    if not Path(wheel).is_file():
        raise _refused(f"stack {resolved.name}: no gfem wheel at {wheel}")
    palace_wheel(resolved)  # refuse before building anything
    env = _gfem_env(resolved, Path(wheel))
    _palace_env(resolved)
    return env


def provenance(resolved: Resolved) -> dict[str, Any]:
    """sha256 and text of both manifests, the Palace wheel's sha256 and version."""
    out: dict[str, Any] = {}
    for code, env in resolved.code_env.items():
        text = (Path(env) / MANIFEST).read_text()
        out[f"{code}_manifest"] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
        }
    palace = out["palace_manifest"]["text"].splitlines()
    out["palace_wheel_sha256"] = palace[0].split()[0]
    out["palace_version"] = palace[1].removeprefix("palace --version: ")
    return out
