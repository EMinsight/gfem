"""The gsim recipe's mapping runs on the two shapes of port gsim writes.

docs/recipes/gsim.md needs gsim to make its mesh, so its first block never
runs here. Its "Read the mesh" and "Build the model" blocks do, on a small
mesh (tests/unit/data/two_box_msh41.msh) and two config.json files cut from
gsim's output down to that mesh's tags: a microstrip whose single-element
port carries `Attributes` and an unsigned `Direction` at its top level, and a
coplanar line whose port lists two `Elements`. Nothing is solved.
"""

import shutil
from pathlib import Path

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "gsim.md"
DATA = Path(__file__).parent / "data"
SECTIONS = ("2. Read the mesh", "3. Build the model from `config.json`")


def _run(config: str, tmp_path: Path) -> dict:
    """The page's mapping blocks run on ``config``, and the names they define."""
    sim = tmp_path / "sim"
    sim.mkdir()
    shutil.copy(DATA / "two_box_msh41.msh", sim / "device.msh")
    shutil.copy(DATA / config, sim / "config.json")
    page = PAGE.read_text(encoding="utf-8")
    code = [e.code for e in dp.examples(page) if e.section in SECTIONS]
    assert len(code) == 3
    namespace: dict = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        for block in code:
            exec(compile(block, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace


def test_a_single_element_port_with_an_unsigned_axis(tmp_path):
    done = _run("gsim_microstrip.json", tmp_path)
    (port,) = done["ports"]
    assert port.name == "p1"
    assert port.surface == ("air__sub",)
    assert port.direction == ((0.0, 0.0, 1.0),)
    assert port.impedance == 50.0
    assert done["model"].ports == (port,)
    assert len(done["model"].excluded) == 1  # the trace inside the substrate


def test_a_port_of_several_elements(tmp_path):
    done = _run("gsim_cpw.json", tmp_path)
    (port,) = done["ports"]
    assert port.surface == ("air__sub", "air__None")
    assert port.direction == ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    assert done["model"].ports == (port,)


def test_the_thermal_section_reads_a_cross_section_and_links_the_recipes():
    page = PAGE.read_text(encoding="utf-8")
    section = dp.section(page, "5. Thermal cross-sections")
    (block,) = dp.blocks(section, "python")
    assert "gfem.Mesh.from_gmsh_cross_section(" in block
    for link in ("{doc}`heater-temperature`", "{doc}`thermo-optic-phase`"):
        assert link in section, link
    # facts only: no claim that gsim's thermal workflow runs gfem
    assert "backend" not in section.lower()


def test_the_thermal_section_maps_physical_groups_as_the_reader_does(tmp_path):
    import gfem

    path = tmp_path / "thermal.msh"
    path.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n"
        '1 1 "bottom"\n2 2 "oxide"\n$EndPhysicalNames\n$Nodes\n3\n'
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n$Elements\n2\n"
        "1 1 2 1 1 1 2\n2 2 2 2 2 1 2 3\n$EndElements\n"
    )
    section = gfem.Mesh.from_gmsh_cross_section(path, length_unit="um")
    assert section.volumes == ("oxide",)
    assert {"bottom", "front", "back"} <= set(section.surfaces)
