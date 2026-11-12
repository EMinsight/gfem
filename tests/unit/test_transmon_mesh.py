"""The transmon tutorial's committed meshes (tools/transmon_mesh.py).

The script needs gmsh, which is not in the lock, so this cannot regenerate the
files as test_tutorial_meshes.py does. It pins what the tutorial and the
flagship test rely on instead: the named groups and the node and tetrahedron
counts, and that the script's docstring table says the same.
"""

from pathlib import Path

import pytest
import transmon_mesh

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"

COUNTS = {"transmon.msh": (1636, 7880), "transmon_coarse.msh": (913, 4308)}
SURFACES = {
    "junction",
    "pad_edge",
    "stub_wall",
    "stub_tip",
    "pad",
    "chip_top",
    "chip_side",
    "walls",
}


@pytest.mark.parametrize("name", sorted(COUNTS))
def test_named_groups(name):
    mesh = gfem.Mesh.from_gmsh(DATA / name, length_unit="mm")
    assert set(mesh.volumes) == {"vacuum", "chip"}
    assert set(mesh.surfaces) == SURFACES


@pytest.mark.parametrize("name", sorted(COUNTS))
def test_node_and_tetrahedron_counts(name):
    mesh = gfem.Mesh.from_gmsh(DATA / name, length_unit="mm")
    assert (len(mesh.imported_nodes), len(mesh.tetrahedra)) == COUNTS[name]


def test_the_scripts_docstring_table_matches():
    rows = {
        line.split()[0]: tuple(int(n) for n in line.split()[-2:])
        for line in transmon_mesh.__doc__.splitlines()
        if line.startswith("transmon")
    }
    assert rows == COUNTS
    assert set(transmon_mesh.MESHES) == set(COUNTS)
    assert set(transmon_mesh.SURFACES) == SURFACES
