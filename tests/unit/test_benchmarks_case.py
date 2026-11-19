"""`benchmarks.harness.case`: case files, the Palace-config importer and mesh pins."""

import json
import lzma
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.harness import case
from benchmarks.harness.case import CaseError

DATA = Path(__file__).resolve().parent / "data"
MESH41 = DATA / "two_box_msh41.msh"

#: ``$PhysicalNames`` of ``two_box_msh41.msh``, read by eye from the file.
TWO_BOX_REGIONS = {
    "metal_xy": (2, 4),
    "metal_z": (2, 5),
    "air__sub": (2, 7),
    "sub__None": (2, 8),
    "air__None": (2, 9),
    "sub": (3, 1),
    "air": (3, 2),
}

CASE_TOML = """\
format = 1
name = "tiny"
size = "small"
family = "tiny"
order = 2

[mesh]
sha256 = "{sha}"
bytes = 1234
length_unit = "um"
file = "mesh.msh.xz"

[mesh.source]
recipe = "by hand"
note = "a test case"

[regions]
sub = [3, 1]
air = [3, 2]
metal_xy = [2, 4]
metal_z = [2, 5]
air__sub = [2, 7]
sub__None = [2, 8]
air__None = [2, 9]

[problem]
kind = "driven"
frequencies_hz = [1e9, 5e10]
sweep = "uniform"
excite = ["p1"]

[quantities]
s = [["p1", "p1"], ["p2", "p1"]]

[[materials]]
region = "sub"
eps_r = 4.1
tan_delta = [0.0, 0.0, 0.001]

[[materials]]
region = "air"
eps_r = 1.0

[[boundaries]]
kind = "conductor"
region = "metal_xy"
conductivity = 3.77e7
thickness = 0.42

[[boundaries]]
kind = "absorbing"
region = "air__None"
order = 1

[[ports]]
kind = "lumped"
name = "p1"
impedance_ohm = 50.0
elements = [{{region = "air__sub", direction = "+y"}}]

[[ports]]
kind = "lumped"
name = "p2"
impedance_ohm = 50.0
elements = [{{region = "sub__None", direction = "-z"}}]
""".format(sha="ab" * 32)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


# --- $PhysicalNames ----------------------------------------------------------------


def test_region_table_comes_from_physical_names_of_msh41():
    assert case.physical_names(MESH41) == TWO_BOX_REGIONS


def test_region_table_of_an_msh22_file_and_its_xz(tmp_path):
    msh = tmp_path / "m.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n"
        '3 1 "air"\n2 2 "wall"\n$EndPhysicalNames\n$Nodes\n0\n$EndNodes\n'
    )
    xz = tmp_path / "m.msh.xz"
    xz.write_bytes(lzma.compress(msh.read_bytes()))
    want = {"air": (3, 1), "wall": (2, 2)}
    assert case.physical_names(msh) == want
    assert case.physical_names(xz) == want


def test_a_mesh_without_physical_names_is_refused(tmp_path):
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    with pytest.raises(CaseError, match="PhysicalNames"):
        case.physical_names(msh)


# --- case.toml ---------------------------------------------------------------------


def _write_case(directory: Path, text: str = CASE_TOML) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "case.toml").write_text(text)
    return directory


def test_case_file_loads_into_typed_fields(tmp_path):
    c = case.load_case(_write_case(tmp_path / "tiny"))
    assert (c.name, c.size, c.family, c.order) == ("tiny", "small", "tiny", 2)
    assert c.mesh.sha256 == "ab" * 32
    assert c.mesh.file == "mesh.msh.xz"
    assert c.mesh.source is not None
    assert c.mesh.source.recipe == "by hand"
    assert c.regions["metal_xy"] == (2, 4)
    assert c.materials[0].tan_delta == (0.0, 0.0, 0.001)
    assert c.materials[1].tan_delta is None
    assert [b.kind for b in c.boundaries] == ["conductor", "absorbing"]
    assert c.boundaries[0].settings == {"conductivity": 3.77e7, "thickness": 0.42}
    assert c.ports[0].elements[0].direction == "+y"
    assert c.problem.frequencies() == (1e9, 5e10)
    assert c.quantities.s == (("p1", "p1"), ("p2", "p1"))
    assert c.directory == tmp_path / "tiny"


def test_case_file_round_trips_through_the_writer(tmp_path):
    raw = tomllib.loads(CASE_TOML)
    text = case.dumps(raw)
    assert tomllib.loads(text) == raw
    c = case.load_case(_write_case(tmp_path / "tiny", text))
    assert tomllib.loads(case.dumps(c.to_dict())) == raw


def test_range_frequencies_follow_the_floor_rule(tmp_path):
    text = CASE_TOML.replace(
        "frequencies_hz = [1e9, 5e10]",
        "frequencies_hz = {start = 1e9, stop = 2e9, step = 0.3e9}",
    )
    c = case.load_case(_write_case(tmp_path / "tiny", text))
    assert c.problem.frequencies() == pytest.approx((1e9, 1.3e9, 1.6e9, 1.9e9))


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("format = 1", "format = 2", "format"),
        ('size = "small"', 'size = "huge"', "size"),
        ('region = "sub"', 'region = "nowhere"', "nowhere"),
        ('direction = "+y"', 'direction = "y"', "direction"),
        ('excite = ["p1"]', 'excite = ["p9"]', "p9"),
        ('length_unit = "um"', 'length_unit = "in"', "length_unit"),
        ("order = 2", "order = 2\nthreads = 4", "threads"),
        ('["p2", "p1"]', '["p3", "p1"]', "p3"),
    ],
)
def test_a_bad_case_file_is_refused(tmp_path, old, new, match):
    assert old in CASE_TOML
    path = _write_case(tmp_path / "tiny", CASE_TOML.replace(old, new, 1))
    with pytest.raises(CaseError, match=match):
        case.load_case(path)


REFERENCE_KEY = 'order = 2\n\n[reference]\ncase = "big"'


def test_a_reference_key_names_the_family_case_without_changing_the_physics(tmp_path):
    plain = case.load_case(_write_case(tmp_path / "plain" / "tiny"))
    text = CASE_TOML.replace("order = 2", REFERENCE_KEY, 1)
    c = case.load_case(_write_case(tmp_path / "keyed" / "tiny", text))
    assert plain.reference_case is None
    assert c.reference_case == "big"
    assert case.reference_directory(plain) == tmp_path / "plain" / "tiny"
    assert case.reference_directory(c) == tmp_path / "keyed" / "big"
    assert case.physics_sha256(c) == case.physics_sha256(plain)
    assert case.definition_sha256(c) != case.definition_sha256(plain)
    assert tomllib.loads(case.dumps(c.to_dict())) == tomllib.loads(text)


@pytest.mark.parametrize(
    ("reference", "match"),
    [
        ('case = "../big"', "reference.case"),
        ('case = ""', "reference.case"),
        ("case = 3", "reference.case"),
        ('case = "big"\nid = "x"', "id"),
    ],
)
def test_a_bad_reference_key_is_refused(tmp_path, reference, match):
    text = CASE_TOML.replace("order = 2", f"order = 2\n\n[reference]\n{reference}", 1)
    with pytest.raises(CaseError, match=match):
        case.load_case(_write_case(tmp_path / "tiny", text))


def test_case_name_must_match_its_directory(tmp_path):
    with pytest.raises(CaseError, match="other"):
        case.load_case(_write_case(tmp_path / "other"))


# --- importer ----------------------------------------------------------------------


def _render_palace(c: case.Case) -> dict:
    """The case's boundary blocks as Palace attributes (a test-side renderer)."""
    tag = {name: t for name, (_, t) in c.regions.items()}
    conductors = [b for b in c.boundaries if b.kind == "conductor"]
    absorbing = [b for b in c.boundaries if b.kind == "absorbing"]
    return {
        "Materials": [([tag[m.region]], m.eps_r, m.tan_delta) for m in c.materials],
        "Conductivity": [
            ([tag[b.region]], b.settings["conductivity"], b.settings["thickness"])
            for b in conductors
        ],
        "Absorbing": (
            [tag[b.region] for b in absorbing],
            absorbing[0].settings["order"],
        ),
        "LumpedPort": [
            [([tag[e.region]], e.direction) for e in p.elements] for p in c.ports
        ],
    }


@pytest.mark.parametrize(
    ("fixture", "elements"),
    [
        # single-face port: Attributes and Direction on the port, "Z" unsigned
        ("gsim_microstrip.json", [[([7], "+z")]]),
        # two-face port under Elements, signed directions kept
        ("gsim_cpw.json", None),
    ],
)
def test_import_renders_back_to_the_original_attributes(tmp_path, fixture, elements):
    config = json.loads((DATA / fixture).read_text())
    shutil.copy(MESH41, tmp_path / "device.msh")
    (tmp_path / "config.json").write_text(json.dumps(config))
    pinned = case.import_case(
        tmp_path / "config.json",
        tmp_path / "device.msh",
        tmp_path / "cases" / "dev",
        frequencies_hz=[1e9, 2e9],
    )
    c = case.load_case(tmp_path / "cases" / "dev")
    assert pinned.committed
    assert (tmp_path / "cases" / "dev" / "source" / "config.json").exists()
    assert c.mesh.length_unit == "um"
    assert c.regions == TWO_BOX_REGIONS
    back = _render_palace(c)
    bnd = config["Boundaries"]
    assert [m[:2] for m in back["Materials"]] == [
        ([t], m["Permittivity"])
        for m in config["Domains"]["Materials"]
        for t in m["Attributes"]
    ]
    # thickness stays in mesh units: no 4.1999999999999995e-07 round-off
    assert back["Conductivity"] == [
        ([t], c_["Conductivity"], c_["Thickness"])
        for c_ in bnd["Conductivity"]
        for t in c_["Attributes"]
    ]
    assert back["Absorbing"] == (
        bnd["Absorbing"]["Attributes"],
        bnd["Absorbing"]["Order"],
    )
    want = elements or [
        [(e["Attributes"], e["Direction"].lower()) for e in p["Elements"]]
        for p in bnd["LumpedPort"]
    ]
    assert back["LumpedPort"] == want
    # not voided (no build.json): the metal shell encloses an excluded volume
    assert c.excluded == (("metal_xy", "metal_z"),)
    assert c.problem.excite == ("p1",)
    assert c.problem.frequencies() == (1e9, 2e9)


def test_import_without_frequencies_is_refused():
    config = json.loads((DATA / "gsim_microstrip.json").read_text())
    with pytest.raises(CaseError, match="frequencies"):
        case.import_palace(config, case.physical_names(MESH41), name="m", voided=False)


def test_import_reads_a_palace_linear_frequency_range():
    config = json.loads((DATA / "gsim_microstrip.json").read_text())
    config["Solver"] = {
        "Driven": {
            "Samples": [
                {"Type": "Linear", "MinFreq": 1, "MaxFreq": 2, "FreqStep": 0.5}
            ],
            "AdaptiveTol": 0.02,
        }
    }
    raw = case.import_palace(
        config, case.physical_names(MESH41), name="m", voided=False
    )
    assert raw["problem"]["frequencies_hz"] == {"start": 1e9, "stop": 2e9, "step": 5e8}
    assert raw["problem"]["sweep"] == "adaptive"


def test_import_of_a_voided_case_has_no_exclusion(tmp_path):
    shutil.copy(DATA / "gsim_microstrip.json", tmp_path / "config.json")
    shutil.copy(MESH41, tmp_path / "device.msh")
    (tmp_path / "build.json").write_text('{"excluded_entities": [["sub", 5]]}')
    case.import_case(
        tmp_path / "config.json",
        tmp_path / "device.msh",
        tmp_path / "cases" / "v",
        frequencies_hz=[1e9],
    )
    c = case.load_case(tmp_path / "cases" / "v")
    assert c.excluded == ()
    assert (tmp_path / "cases" / "v" / "source" / "build.json").exists()


def test_import_reads_materials_axes_and_eigen_problems():
    config = {
        "Problem": {"Type": "Eigenmode"},
        "Model": {"L0": 1e-3},
        "Solver": {"Order": 3, "Eigenmode": {"N": 3, "Target": 4.5}},
        "Domains": {
            "Materials": [
                {
                    "Attributes": [1],
                    "Permittivity": [2.0, 3.0, 4.0],
                    "Permeability": 1.5,
                    "LossTan": [0.0, 0.1, 0.2],
                    "MaterialAxes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "Conductivity": 0.5,
                }
            ]
        },
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "Periodic": {
                "BoundaryPairs": [
                    {
                        "DonorAttributes": [3],
                        "ReceiverAttributes": [4],
                        "Translation": [10.0, 0.0, 0.0],
                    }
                ],
                "FloquetWaveVector": [0.25, 0.0, 0.0],
            },
        },
    }
    regions = {"cell": (3, 1), "z": (2, 2), "x_min": (2, 3), "x_max": (2, 4)}
    raw = case.import_palace(config, regions, name="e", voided=True)
    assert raw["materials"] == [
        {
            "region": "cell",
            "eps_r": [2.0, 3.0, 4.0],
            "mu_r": 1.5,
            "tan_delta": [0.0, 0.1, 0.2],
            "conductivity": 0.5,
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    ]
    assert raw["boundaries"] == [
        {"kind": "pec", "region": "z"},
        {
            "kind": "periodic",
            "donor": "x_min",
            "receiver": "x_max",
            "translation": [10.0, 0.0, 0.0],
        },
    ]
    assert raw["problem"] == {
        "kind": "eigen",
        "target_hz": 4.5e9,
        "n_modes": 3,
        "wave_vector": [250.0, 0.0, 0.0],  # rad/mm -> rad/m
    }
    assert raw["quantities"] == {"modes": 3}
    assert raw["order"] == 3


def test_import_refuses_a_boundary_it_does_not_translate():
    config = {
        "Problem": {"Type": "Driven"},
        "Model": {"L0": 1e-6},
        "Boundaries": {"Floquet": {}},
    }
    with pytest.raises(CaseError, match="Floquet"):
        case.import_palace(config, {}, name="f", voided=True)


# --- pin ---------------------------------------------------------------------------


def test_pin_writes_mesh_and_regions_and_commits_a_small_mesh(tmp_path):
    case_dir = tmp_path / "tiny"
    _write_case(case_dir, '[mesh.source]\nrecipe = "by hand"\n')
    pinned = case.pin(case_dir, MESH41, "um")
    raw = tomllib.loads((case_dir / "case.toml").read_text())
    data = MESH41.read_bytes()
    assert pinned.sha256 == raw["mesh"]["sha256"]
    assert raw["mesh"]["bytes"] == len(data)
    assert raw["mesh"]["file"] == "mesh.msh.xz"
    assert raw["mesh"]["length_unit"] == "um"
    assert raw["mesh"]["source"] == {"recipe": "by hand"}
    assert lzma.decompress((case_dir / "mesh.msh.xz").read_bytes()) == data
    assert {k: tuple(v) for k, v in raw["regions"].items()} == TWO_BOX_REGIONS


def test_pin_refuses_to_overwrite_an_existing_pin(tmp_path):
    case_dir = tmp_path / "tiny"
    case.pin(case_dir, MESH41, "um")
    before = (case_dir / "case.toml").read_text()
    with pytest.raises(CaseError, match="already pinned"):
        case.pin(case_dir, MESH41, "um")
    assert (case_dir / "case.toml").read_text() == before


def test_pin_puts_a_large_mesh_in_the_first_store(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "MAX_COMMITTED_BYTES", 100)
    stores = [tmp_path / "store-a", tmp_path / "store-b"]
    pinned = case.pin(tmp_path / "big", MESH41, "um", stores)
    raw = tomllib.loads((tmp_path / "big" / "case.toml").read_text())
    assert not pinned.committed
    assert "file" not in raw["mesh"]
    assert sorted(p.name for p in (tmp_path / "big").iterdir()) == ["case.toml"]
    stored = stores[0] / f"{pinned.sha256}.msh.xz"
    assert pinned.path == stored
    assert lzma.decompress(stored.read_bytes()) == MESH41.read_bytes()


def test_pin_of_an_external_mesh_needs_a_store(tmp_path, monkeypatch):
    monkeypatch.setattr(case, "MAX_COMMITTED_BYTES", 100)
    with pytest.raises(CaseError, match="no mesh store"):
        case.pin(tmp_path / "big", MESH41, "um")


# --- command line ------------------------------------------------------------------


def test_case_pin_command_prints_the_pin_then_refuses_a_second(tmp_path, capsys):
    argv = [
        "case",
        "pin",
        "tiny",
        str(MESH41),
        "--unit",
        "um",
        "--cases",
        str(tmp_path),
    ]
    assert main(argv) == 0
    sha = tomllib.loads((tmp_path / "tiny" / "case.toml").read_text())["mesh"]["sha256"]
    assert sha in capsys.readouterr().out
    assert main(argv) == 2
    assert "already pinned" in capsys.readouterr().err


def test_case_import_command_writes_a_loadable_case(tmp_path):
    shutil.copy(DATA / "gsim_cpw.json", tmp_path / "config.json")
    argv = [
        "case",
        "import",
        str(tmp_path / "config.json"),
        str(MESH41),
        "--name",
        "cpw",
        "--cases",
        str(tmp_path / "cases"),
        "--frequencies-ghz",
        "1,50",
    ]
    assert main(argv) == 0
    assert case.load_case(tmp_path / "cases" / "cpw").ports[0].name == "p1"


def test_case_commands_refuse_missing_inputs_without_writing(tmp_path, capsys):
    pin = ["case", "pin", "x", str(tmp_path / "no.msh"), "--unit", "um"]
    imp = ["case", "import", str(tmp_path / "no.json"), str(MESH41), "--name", "y"]
    for argv in (pin, imp):
        assert main([*argv, "--cases", str(tmp_path / "cases")]) == 2
        assert "no such" in capsys.readouterr().err
    assert not (tmp_path / "cases").exists()


# --- the microstrip_p3 suite case --------------------------------------------------


def test_a_plan_with_microstrip_p3_pins_it_at_907_791_dof(tmp_path):
    from benchmarks.harness import config, matrix, slurm

    text = """\
schema = 1
cases = ["microstrip_p3"]
solvers = ["gfem:direct", "gfem:petsc-mumps", "gfem:bddc", "palace:mumps", "palace:pmg"]
layouts = ["default", "4x4"]
core_budget = 16
repeats = 1
node_class = "standard"
anchor_tolerance = 1e-8

[wall_cap_s]
microstrip_p3 = 1800

[stacks.wheels.versions]
palace = "0.19.0"
"""
    path = tmp_path / "p3.toml"
    path.write_text(text)
    site = config.load_site(config.CASES_DIR.parent / "site.example.toml", environ={})
    plan = matrix.expand(config.load_campaign(path), site, commit="858bc66" + "0" * 33)
    assert {c.case for c in plan.cells} == {"microstrip_p3"}
    refused = {(c.solver, c.layout) for c in plan.cells if c.refused}
    assert refused == {("gfem:direct", "4x4")}  # one rank only
    direct = next(c for c in plan.cells if c.solver == "gfem:direct" and not c.refused)
    assert direct.est_s == 120
    p3 = case.load_case(config.CASES_DIR / "microstrip_p3")
    assert slurm.pinned_ndof(p3) == 907_791
    assert case.reference_directory(p3) == config.CASES_DIR / "microstrip"
