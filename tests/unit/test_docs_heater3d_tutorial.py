"""The 3D heater tutorial's run record and committed outputs.

The page (docs/tutorials/heater3d.ipynb) reads data/heater3d.{json,npz},
written by tools/heater3d_run.py, and solves nothing: it refuses a record
whose script hashes differ from the scripts'. This checks the record's
schema and size, that the committed record is the current scripts', and the
committed notebook's outputs; no solve.
"""

import hashlib
import json
import re
from pathlib import Path

import heater3d_run as run
import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "docs" / "tutorials" / "heater3d.ipynb"
DATA = NOTEBOOK.parent / "data"

# The record's solver outputs, typed here, not read back from the script,
# so a drift in either is a failure.
LEVEL = ("resistance", "power", "route_phase", "pass_phase", "heater_rise", "core_rise")
NPZ_KEYS = [
    *(f"{level}_{key}" for level in ("main", "medium", "fine") for key in LEVEL),
    *("pass_names", "pass_y", "spiral_arc_length", "spiral_phase_per_length"),
    *("centre_pass_arc_length", "centre_pass_phase_per_length"),
    *("core_polygon", "heater_polygon", "pad_polygons"),
    *("n_eff", "route_length", "sink_power"),
    *("map_x", "map_y", "map_core", "map_heater", "cut_y", "cut_z", "map_cut"),
    *("section_power_per_length", "section_phase_per_length"),
    *("time", "p_pi", "trace_spiral", "trace_centre_pass", "trace_outer_pass"),
    *("trace_heater", "rise_time_on", "time_constant_on"),
    *("rise_time_off", "time_constant_off"),
]
STAGES = ("main", "medium", "fine")


def _record():
    """A record with every key the schema asks for, and small arrays."""
    stage = dict.fromkeys(("start_utc", "end_utc", "command", "environment"), "")
    record = {
        "profile": "smoke",
        "provenance": {
            "commit": "",
            "sha256": {},
            "stages": {s: dict(stage) for s in STAGES},
        },
        "settings": run.settings(run.PROFILES["smoke"], {}),
        "observations": {
            s: {"wall_s": 0.0, "peak_rss_bytes": 0, "steps_s": {}} for s in STAGES
        },
    }
    return record, {key: np.zeros(3) for key in NPZ_KEYS}


def test_the_record_holds_exactly_the_listed_keys(tmp_path):
    record, arrays = _record()
    run.write_record(record, arrays, tmp_path)
    loaded = json.loads((tmp_path / "heater3d.json").read_text())
    assert set(loaded) == {"profile", "provenance", "settings", "observations"}
    with np.load(tmp_path / "heater3d.npz") as npz:
        assert sorted(npz.files) == sorted(NPZ_KEYS)
    del arrays["p_pi"]
    with pytest.raises(ValueError, match="p_pi"):
        run.write_record(record, arrays, tmp_path / "again")
    assert not (tmp_path / "again").exists()


def test_a_record_over_3_mb_is_refused(tmp_path):
    record, arrays = _record()
    arrays["map_core"] = np.random.default_rng(0).random(
        500_000
    )  # 4 MB, incompressible
    with pytest.raises(ValueError, match="over 3000000"):
        run.write_record(record, arrays, tmp_path)
    assert not list(tmp_path.iterdir())


def test_the_settings_give_the_device_as_built():
    settings = run.settings(run.PROFILES["full"], {})
    geometry = settings["geometry"]
    assert geometry["inner_radius"] == 10.0
    assert geometry["pitch"] == 2.0
    assert geometry["straights_per_side"] == 8
    assert geometry["heater_width"] == 5.0
    assert geometry["route_length_um"] == pytest.approx(510 + 233 * np.pi)
    assert geometry["heater_straights_y_um"] == [12.0, -10.0]
    assert settings["order"] == 2
    assert settings["linear_solver"] == "direct"
    json.dumps(settings)


@pytest.fixture(scope="module")
def record():
    return json.loads((DATA / "heater3d.json").read_text())


def test_the_committed_record_is_a_full_run_of_the_current_scripts(record):
    assert record["profile"] == "full"
    for script, recorded in record["provenance"]["sha256"].items():
        current = hashlib.sha256((ROOT / script).read_bytes()).hexdigest()
        assert current == recorded, f"{script} changed: rerun `just heater3d-data`"
    assert sorted(record["provenance"]["sha256"]) == sorted(run.SCRIPTS)
    with np.load(DATA / "heater3d.npz") as npz:
        assert sorted(npz.files) == sorted(NPZ_KEYS)


def test_the_committed_record_is_under_3_mb():
    total = sum((DATA / name).stat().st_size for name in run.RECORD)
    assert total < 3_000_000


@pytest.fixture(scope="module")
def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def _code(cells):
    return [c for c in cells if c["cell_type"] == "code"]


def test_the_page_checks_the_record_before_reading_it(cells):
    first = _source(_code(cells)[0])
    assert "sha256" in first
    assert "heater3d-data" in first
    assert '"full"' in first


def test_the_committed_run_reached_the_closing_cell(cells):
    code = _code(cells)
    assert all(c["execution_count"] is not None for c in code), "not executed"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
    streams = [o for c in code for o in c["outputs"] if o.get("name") == "stderr"]
    assert streams == [], "stderr in the committed outputs fails the docs build"


def test_the_closing_cell_asserts_the_checks(cells):
    closing = _source(_code(cells)[-1])
    for name in ("route_length", "sink_power", "squares", "section", "ladder"):
        assert re.search(rf"\b{name}", closing), name


def test_the_page_names_no_other_solver_or_layout_tool(cells):
    text = " ".join(_source(c) for c in cells).lower()
    assert "palace" not in text
    assert "gdsfactory" not in text
