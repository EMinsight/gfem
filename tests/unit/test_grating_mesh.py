"""The grating-coupler tutorial's committed mesh (tools/grating_mesh.py).

The script needs gmsh, which is not in the lock, so this cannot regenerate the
file as test_tutorial_meshes.py does. It pins what the tutorial relies on
instead: the named groups, the node and triangle counts, that the script's
docstring table says the same, and that the walls are the trenches' sidewalls.
"""

from pathlib import Path

import grating_mesh
import meshio
import numpy as np

import gfem

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
MESH = DATA / "grating.msh"
COUNTS = (4457, 8591)
VOLUMES = {"substrate", "box", "si", "trench", "clad"}


def test_named_groups():
    mesh = gfem.Mesh.from_gmsh_cross_section(MESH, length_unit="um")
    assert set(mesh.volumes) == VOLUMES
    assert set(mesh.surfaces) == {*grating_mesh.LINES, "front", "back"}


def test_node_and_triangle_counts():
    section = meshio.read(MESH)
    triangles = sum(len(b.data) for b in section.cells if b.type == "triangle")
    assert (len(section.points), triangles) == COUNTS


def test_the_scripts_docstring_table_matches():
    rows = {
        line.split()[0]: tuple(int(n) for n in line.split()[-2:])
        for line in grating_mesh.__doc__.splitlines()
        if line.startswith("grating.msh")
    }
    assert rows == {"grating.msh": COUNTS}
    assert set(grating_mesh.MESHES) == {"grating.msh"}


def test_the_walls_are_the_trenches_sidewalls():
    mesh = gfem.Mesh.from_gmsh_cross_section(MESH, length_unit="um")
    faces = np.asarray(mesh.triangles)[mesh.elements("walls")]
    x = np.asarray(mesh.nodes)[faces][..., 0] / 1e-6
    walls = np.array([w for pair in grating_mesh.trenches() for w in pair])
    nearest = np.abs(x[..., None] - walls).min(axis=-1)
    assert nearest.max() < 1e-9
    found = np.unique(np.round(x, 9))
    assert len(found) == 2 * grating_mesh.TEETH
