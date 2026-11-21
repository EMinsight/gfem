"""`gfem.Mesh.from_gmsh` on a triangle-only file: a mesh of dimension 2.

Its areas are the volumes and its lines the surfaces; the shape-named arrays of
a tetrahedral mesh raise and name the role array to read instead.
"""

from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest
from strip2d import write_strip

import gfem
from gfem._engine.mesh import to_ngsolve

HEATER = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "heater_w2.msh"
AREAS = (
    "core",
    "neighbour",
    "heater",
    "window",
    "neighbour_window",
    "box",
    "cladding",
    "substrate",
)


@pytest.fixture
def strip(tmp_path):
    return write_strip(tmp_path / "strip.msh")


@pytest.fixture(scope="module")
def heater():
    return gfem.Mesh.from_gmsh(HEATER, length_unit="um")


def test_a_triangle_only_file_reads_as_a_mesh_of_dimension_2(heater):
    assert heater.dim == 2
    assert heater.nodes.shape == (4634, 2)
    assert heater.volumes == AREAS
    assert heater.surfaces == ("bottom", "top")
    assert {r.dim for r in heater.regions} == {1, 2}


def test_its_volume_elements_are_triangles_and_its_surface_elements_segments(heater):
    assert heater.volume_elements.shape == (9173, 3)
    assert heater.surface_elements.shape == (73, 2)
    rows = heater.elements("core")
    assert heater.volume_elements[rows].shape[1] == 3
    assert heater.surface_elements[heater.elements("top")].shape[1] == 2


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("tetrahedra", "volume_elements"),
        ("tetrahedron_tags", "volume_elements"),
        ("tetrahedron_entities", "volume_elements"),
        ("triangles", "surface_elements"),
        ("triangle_tags", "surface_elements"),
        ("triangle_entities", "surface_elements"),
        ("triangle_neighbours", "surface_elements"),
    ],
)
def test_a_tetrahedral_array_raises_naming_the_role_array(heater, name, role):
    with pytest.raises(ValueError, match=rf"Mesh\.{name} .*Mesh\.{role}"):
        getattr(heater, name)


def test_a_tetrahedral_mesh_has_dimension_3_and_its_arrays_under_both_names():
    mesh = gfem.Mesh.from_gmsh(Path(__file__).parent / "data" / "two_box_msh22.msh")
    assert mesh.dim == 3
    assert mesh.volume_elements is mesh.tetrahedra
    assert mesh.surface_elements is mesh.triangles


def test_a_line_has_on_its_left_the_area_behind_its_normal(strip):
    mesh = gfem.Mesh.from_gmsh(strip)
    assert mesh.adjacency("ab") == (("a", "b"),)
    assert mesh.adjacency("cb") == (("c", "b"),)
    assert mesh.adjacency("outer") == (
        (None, "a"),
        ("a", None),
        ("b", None),
        ("c", None),
    )


def test_a_line_borders_the_areas_its_slab_face_borders(strip):
    # the slab's faces take domin/domout from the tetrahedra: the 3D rule
    plane = gfem.Mesh.from_gmsh(strip)
    slab = gfem.Mesh.from_gmsh_cross_section(strip)
    for line in plane.surfaces:
        assert plane.adjacency(line) == slab.adjacency(line)


def test_the_same_file_read_as_a_cross_section_is_still_a_slab(strip):
    slab = gfem.Mesh.from_gmsh_cross_section(strip)
    assert slab.dim == 3
    assert slab.nodes.shape[1] == 3
    assert slab.surfaces == ("ab", "cb", "outer", "front", "back")


def test_a_2d_mesh_names_its_elements_and_exports_them(strip):
    mesh = gfem.Mesh.from_gmsh(strip)
    assert repr(mesh) == (
        "Mesh(8 nodes, 6 triangles, volumes=['a', 'b', 'c'], "
        "surfaces=['ab', 'cb', 'outer'], length_unit='um')"
    )
    out = mesh.to_meshio()
    assert [c.type for c in out.cells] == ["triangle", "line"]
    assert out.points.shape == (8, 2)
    np.testing.assert_array_equal(out.cell_data["region"][1][:2], [11, 12])


def test_a_2d_mesh_out_of_one_plane_raises(strip):
    text = strip.read_text().replace("\n8 3 1 0\n", "\n8 3 1 0.5\n")
    strip.write_text(text)
    with pytest.raises(ValueError, match="do not lie in one plane"):
        gfem.Mesh.from_gmsh(strip)


def test_a_2d_mesh_with_a_line_off_the_triangles_edges_raises(strip):
    text = strip.read_text().replace(" 11 11 2 6\n", " 11 11 1 7\n")
    strip.write_text(text)
    with pytest.raises(ValueError, match="'ab' is not an edge"):
        gfem.Mesh.from_gmsh(strip)


def test_an_area_of_a_2d_mesh_is_a_volume_to_a_layer_grid(heater):
    grid = gfem.design.LayerGrid(heater, "core", pitch=0.1e-6)
    assert grid.n_elements == len(heater.elements("core"))
    assert grid.n_faces is None


# --- the engine mesh -------------------------------------------------------------


def _areas(mesh):
    x = np.asarray(mesh.nodes)[mesh.volume_elements]
    e1, e2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def test_the_engine_mesh_is_2d_and_covers_the_areas(heater):
    built = to_ngsolve(heater)
    assert built.mesh.dim == 2
    assert set(built.mesh.GetMaterials()) == set(AREAS)
    assert set(built.mesh.GetBoundaries()) == {"bottom", "top"}
    area = ngs.Integrate(1, built.mesh) * built.scale**2
    assert area == pytest.approx(_areas(heater).sum(), rel=1e-12)
    for name in AREAS:
        region = built.mesh.Materials(name)
        part = ngs.Integrate(1, built.mesh, definedon=region) * built.scale**2
        expected = _areas(heater)[heater.elements(name)].sum()
        assert part == pytest.approx(expected, rel=1e-12)


def test_a_p0_value_in_mesh_order_reads_back_at_each_centroid(heater):
    built = to_ngsolve(heater)
    values = np.arange(len(heater.volume_elements), dtype=float)
    p0 = ngs.GridFunction(ngs.L2(built.mesh, order=0))
    p0.vec.FV().NumPy()[:] = values[built.cell_order]
    rows = np.random.default_rng(0).choice(len(values), 50, replace=False)
    centroids = np.asarray(heater.nodes)[heater.volume_elements[rows]].mean(axis=1)
    for row, (x, y) in zip(rows, centroids / built.scale, strict=True):
        assert p0(built.mesh(x, y)) == row


def test_engine_segments_are_the_mesh_segments_in_facet_order(heater):
    built = to_ngsolve(heater)
    segments = heater.surface_elements[built.facet_order]
    for i, el in enumerate(built.mesh.Elements(ngs.BND)):
        assert sorted(v.nr for v in el.vertices) == sorted(segments[i])


def test_a_p0_value_reads_domin_behind_a_segments_normal(strip):
    # the engine normal points out of domin, as a face's does in 3D
    mesh = gfem.Mesh.from_gmsh(strip)
    built = to_ngsolve(mesh)
    value = {"a": 1.0, "b": 2.0, "c": 3.0}
    p0 = ngs.GridFunction(ngs.L2(built.mesh, order=0))
    tags = {mesh.region(n).tag: v for n, v in value.items()}
    per_element = np.array([tags[int(t)] for t in mesh._topology.volume_tags])
    p0.vec.FV().NumPy()[:] = per_element[built.cell_order]
    normal = ngs.specialcf.normal(2)
    nodes = np.asarray(mesh.nodes) / built.scale
    for line, (behind, front) in (("ab", ("a", "b")), ("cb", ("c", "b"))):
        (segment,) = mesh.surface_elements[mesh.elements(line)]
        mid = nodes[segment].mean(axis=0)
        n = np.array(normal(built.mesh(*mid, VOL_or_BND=ngs.BND)))
        assert p0(built.mesh(*(mid - 0.1 * n))) == value[behind]
        assert p0(built.mesh(*(mid + 0.1 * n))) == value[front]
    for segment in mesh.surface_elements[mesh.elements("outer")]:
        mid = nodes[segment].mean(axis=0)
        n = np.array(normal(built.mesh(*mid, VOL_or_BND=ngs.BND)))
        assert n @ (mid - [1.5, 0.5]) > 0  # out of the mesh


def test_a_2d_engine_mesh_under_ranks_raises_naming_the_slab(strip):
    mesh = gfem.Mesh.from_gmsh(strip)
    with pytest.raises(NotImplementedError, match="from_gmsh_cross_section"):
        to_ngsolve(mesh, comm=object())


def _message(path):
    with pytest.raises(ValueError) as info:
        gfem.Mesh.from_gmsh(path)
    return str(info.value)


def test_a_2d_mesh_out_of_one_plane_is_named_a_2d_mesh(strip):
    strip.write_text(strip.read_text().replace("\n8 3 1 0\n", "\n8 3 1 0.5\n"))
    message = _message(strip)
    assert "the 2D mesh's nodes" in message
    assert "cross-section" not in message


def test_a_2d_mesh_with_quadrilaterals_is_named_a_2d_mesh(strip):
    text = strip.read_text().replace("$EndElements\n", "")
    n = int(text.split("$Elements\n")[1].split("\n")[0])
    text = text.replace(f"$Elements\n{n}\n", f"$Elements\n{n + 1}\n")
    text += f"{n + 1} 3 2 1 1 1 2 6 5\n$EndElements\n"
    strip.write_text(text)
    message = _message(strip)
    assert "2D meshes of triangles only" in message
    assert "cross-section" not in message
