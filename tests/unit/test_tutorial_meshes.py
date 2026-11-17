"""The tutorials' committed meshes are what their generator writes.

docs/tutorials/data/*.msh are read by the tutorials, the get-started page and
several drift tests, and tools/make_tutorial_meshes.py is the only description
of how they were built. This regenerates them (no gmsh, about two seconds) and
compares bytes, so neither can change without the other. The transmon meshes
are gmsh-built by tools/transmon_mesh.py and pinned by test_transmon_mesh.py;
the metagrating meshes are gmsh-built by tools/metagrating_mesh.py
(`just metagrating-mesh`); the heater meshes (and the thermo-optic recipe's extruded
heater_straight.msh) are gmsh-built by tools/heater_section_mesh.py
(`just heater-section-mesh`).
"""

from pathlib import Path

import heater_section_mesh
import make_tutorial_meshes
import metagrating_mesh
import pytest
import transmon_mesh

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
GMSH_BUILT = (
    transmon_mesh.MESHES
    | metagrating_mesh.MESHES
    | heater_section_mesh.MESHES
    | heater_section_mesh.STRAIGHT
)
COMMITTED = sorted(
    path.name for path in DATA.glob("*.msh") if path.name not in GMSH_BUILT
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("tutorial-meshes")
    return {path.name: path for path in make_tutorial_meshes.write(out)}


def test_the_generator_writes_every_committed_mesh(generated):
    assert sorted(generated) == COMMITTED


@pytest.mark.parametrize("name", COMMITTED)
def test_a_committed_mesh_is_the_generators_output(generated, name):
    assert generated[name].read_bytes() == (DATA / name).read_bytes(), (
        f"docs/tutorials/data/{name} differs from what tools/make_tutorial_meshes.py "
        "writes: rerun `python tools/make_tutorial_meshes.py docs/tutorials/data`"
    )
