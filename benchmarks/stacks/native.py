"""The ``native`` stack kind: prefixes built by ``tools/stacks/``.

Each package of a native stack lives at ``{stacks.root}/<stack>/<pkg>-<version>/``
with an ``activate.sh`` (which sources the stack's frozen toolchain) and, once
finished, a ``provenance.json``. Nothing is built here: a stack refuses while any
of its prefixes lacks ``provenance.json`` or its packages name different
toolchain hashes (the check ``stack.sh check`` makes). Standard library only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.harness.config import ConfigError

if TYPE_CHECKING:
    from benchmarks.harness.config import Site
    from benchmarks.stacks import Description, Resolved


def _packages(description: Description) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for section in description.codes.values():
        for package in section.packages:
            seen[package] = None
    return tuple(seen)


def inputs(
    description: Description, _versions: dict[str, str], site: Site
) -> tuple[dict[str, Any], dict[str, str]]:
    """The site's stacks root, the only site input of a native stack."""
    if site.stacks_root is None:
        raise ConfigError(
            f"{site.path}: native stack {description.name} needs [stacks] root"
        )
    return {"root": site.stacks_root}, {"{stacks.root}": site.stacks_root}


def paths(
    description: Description, versions: dict[str, str], site: Site, _env_dir: str
) -> dict[str, Any]:
    """Package -> prefix, at the versions the campaign resolved."""
    base = f"{site.stacks_root}/{description.name}"
    return {
        "prefixes": {p: f"{base}/{p}-{versions[p]}" for p in _packages(description)}
    }


def activation(resolved: Resolved, code: str) -> list[str]:
    """Source each package's ``activate.sh`` in the code's order."""
    section = resolved.description.codes[code]
    return [f". {resolved.prefixes[p]}/activate.sh" for p in section.packages]


def _label(resolved: Resolved, package: str) -> str:
    return f"{resolved.name}/{package}-{resolved.versions[package]}"


def refusal(resolved: Resolved) -> str | None:
    """Why the stack cannot launch: an unfinished prefix or differing toolchains."""
    toolchains: dict[str, str] = {}
    for package, prefix in resolved.prefixes.items():
        path = Path(prefix) / "provenance.json"
        if not path.is_file():
            return (
                f"{_label(resolved, package)}: no provenance.json (unfinished prefix)"
            )
        try:
            toolchains[package] = json.loads(path.read_text())["toolchain"]["sha256"]
        except (ValueError, KeyError, TypeError):
            return (
                f"{_label(resolved, package)}: provenance.json has no toolchain.sha256"
            )
    if len(set(toolchains.values())) > 1:
        listing = ", ".join(f"{p} {h[:12]}" for p, h in toolchains.items())
        return f"{resolved.name}: toolchain hashes differ ({listing})"
    return None


def prepare(resolved: Resolved, **options: Any) -> Path:
    """Validate the prefixes and make the env directory for the env scripts."""
    from benchmarks.stacks import StackRefusedError

    if options:
        raise TypeError(f"native prepare takes no options, got {sorted(options)}")
    reason = refusal(resolved)
    if reason is not None:
        raise StackRefusedError(reason)
    env_dir = Path(resolved.env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    return env_dir


def provenance(resolved: Resolved) -> dict[str, Any]:
    """Per package: prefix, sha256 and content of its ``provenance.json``."""
    packages = {}
    for package, prefix in resolved.prefixes.items():
        data = (Path(prefix) / "provenance.json").read_bytes()
        packages[package] = {
            "prefix": prefix,
            "sha256": hashlib.sha256(data).hexdigest(),
            "content": json.loads(data),
        }
    return {"packages": packages}
