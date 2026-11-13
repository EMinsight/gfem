"""The tutorials' 3D scenes on disk are the ones their outputs show.

A tutorial's interactive views are iframes on ``docs/_static/scenes/*.vtksz``
files, committed next to the notebooks. A view whose file is missing renders
empty, and a file no view names is dead weight in the repository. This reads
files only: it executes no notebook. ``tools/prune_scenes.py`` is what
``just notebooks`` runs to drop the scenes of edited cells, in place of
clearing the directory, so that re-executing one notebook keeps the others'
scenes.
"""

import json
import re
from pathlib import Path

import prune_scenes

ROOT = Path(__file__).parents[2]


def _notebook(path: Path, *scenes: str) -> None:
    html = [f'<iframe src="v.html?fileURL=scenes/{name}"></iframe>' for name in scenes]
    cell = {
        "cell_type": "code",
        "source": ["plotter.show()"],
        "outputs": [{"output_type": "display_data", "data": {"text/html": html}}],
    }
    text = {"cell_type": "markdown", "source": ["scenes/prose-0-light.vtksz"]}
    path.write_text(json.dumps({"cells": [text, cell]}), encoding="utf-8")


def test_every_scene_a_tutorial_shows_is_on_disk():
    shown = prune_scenes.referenced()
    assert shown, "no tutorial notebook found"
    assert any(shown.values()), "no tutorial shows a 3D scene"
    missing = {
        notebook: sorted(n for n in names if not (prune_scenes.SCENES / n).is_file())
        for notebook, names in shown.items()
    }
    assert {k: v for k, v in missing.items() if v} == {}


def test_no_scene_on_disk_is_left_unshown():
    stale = [path.name for path in prune_scenes.unreferenced()]
    assert stale == [], "run `python tools/prune_scenes.py`"


def test_pruning_keeps_the_scenes_of_every_notebook(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    names = ["aaa-0-light.vtksz", "aaa-0-dark.vtksz", "bbb-0-light.vtksz"]
    stale = ["old-0-light.vtksz", "prose-0-light.vtksz"]
    for name in (*names, *stale):
        (scenes / name).write_bytes(b"scene")
    (scenes / "notes.txt").write_text("not a scene")
    _notebook(tmp_path / "one.ipynb", *names[:2])
    _notebook(tmp_path / "two.ipynb", names[2])

    deleted = prune_scenes.prune(scenes, tmp_path)

    assert sorted(p.name for p in deleted) == sorted(stale)
    assert sorted(p.name for p in scenes.iterdir()) == sorted([*names, "notes.txt"])
    assert prune_scenes.prune(scenes, tmp_path) == []


def test_the_notebooks_recipe_takes_names_and_prunes():
    """`just notebooks NAME` must not clear the scenes of the other notebooks."""
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    match = re.search(r"^notebooks(.*):\n((?:    .*\n)+)", justfile, re.MULTILINE)
    assert match, "no notebooks recipe in the justfile"
    parameters, body = match[1], match[2]
    assert "*" in parameters, "the recipe takes no notebook names"
    assert "rm -rf" not in body
    assert body.rstrip().endswith("tools/prune_scenes.py")
