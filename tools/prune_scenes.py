"""Delete the 3D scenes no tutorial shows any more.

Usage: python tools/prune_scenes.py [--dry-run]

``just notebooks`` exports each PyVista view of the tutorials as a pair of
``.vtksz`` files under ``docs/_static/scenes/``, named after a hash of the
source of the cell that shows it (docs/_ext/gfem_scenes.py). Running a cell
again overwrites its scenes in place; editing a cell gives its scenes new
names and leaves the old files behind. This deletes every scene file that no
output of ``docs/tutorials/*.ipynb`` names, so one notebook can be re-executed
without losing the scenes of the others.
"""

import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
SCENES = DOCS / "_static" / "scenes"
NOTEBOOKS = DOCS / "tutorials"

# A scene as the viewer's iframe names it: ...?fileURL=scenes/<name>.vtksz
_SCENE = re.compile(r"scenes/([\w.-]+\.vtksz)")


def referenced(notebooks: Path = NOTEBOOKS) -> dict[str, set[str]]:
    """The scene file names each notebook's outputs show, by notebook name."""
    shown = {}
    for path in sorted(notebooks.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        names = set()
        for cell in notebook["cells"]:
            for output in cell.get("outputs", ()):
                html = output.get("data", {}).get("text/html", "")
                names.update(_SCENE.findall("".join(html)))
        shown[path.name] = names
    return shown


def unreferenced(scenes: Path = SCENES, notebooks: Path = NOTEBOOKS) -> list[Path]:
    """The scene files under ``scenes`` that no notebook output names."""
    keep = set().union(*referenced(notebooks).values())
    return sorted(p for p in scenes.glob("*.vtksz") if p.name not in keep)


def prune(scenes: Path = SCENES, notebooks: Path = NOTEBOOKS) -> list[Path]:
    """Delete the unreferenced scene files; return them."""
    stale = unreferenced(scenes, notebooks)
    for path in stale:
        path.unlink()
    return stale


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv[1:]
    for scene in unreferenced() if dry else prune():
        print(("would delete " if dry else "deleted ") + scene.name)
