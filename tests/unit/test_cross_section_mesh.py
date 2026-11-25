"""`gfem.Mesh.from_gmsh_cross_section`: a triangle section read as a one-element slab.

The fixtures are written here with meshio, as gmsh writes a 2D mesh: a 3 x 2 um
section of two layers, "lower" (y < 1, tag 1) and "upper" (tag 2), on an
irregular triangulation (perturbed nodes, alternating diagonals), with the lines
"bottom" (y = 0, tag 11), "top" (y = 2, tag 12), "interface" (y = 1, tag 13)
and "left" (the top 0.5 um of x = 0, tag 14).
"""

import json
from pathlib import Path

import meshio
import numpy as np
import pytest

import gfem
from gfem.mesh import _TET_FACES, _triangle_neighbours
from gfem.shape import boundary_morph

NX, NY = 6, 4  # cells across the 3 um width and the 2 um height
WIDTH, HEIGHT = 3.0, 2.0
VOLUMES = {"lower": 1, "upper": 2}
LINES = {"bottom": 11, "top": 12, "interface": 13, "left": 14}
TWO_BOX = Path(__file__).parent / "data" / "two_box_msh22.msh"


def _section():
    """Nodes, triangles with their group, and grouped boundary lines."""
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.linspace(0, WIDTH, NX + 1), np.linspace(0, HEIGHT, NY + 1))
    x, y = x.ravel(), y.ravel()
    inner = (x > 0) & (x < WIDTH) & (y > 0) & (y < HEIGHT) & ~np.isclose(y, 1.0)
    h = WIDTH / NX
    x[inner] += rng.uniform(-0.15, 0.15, inner.sum()) * h
    y[inner] += rng.uniform(-0.15, 0.15, inner.sum()) * h
    # shuffle the node numbers so the split meets every pattern
    perm = rng.permutation(len(x))
    nodes = np.zeros((len(x), 3))
    nodes[perm, 0], nodes[perm, 1] = x, y

    def n(i, j):
        return perm[j * (NX + 1) + i]

    tris, groups = [], []
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)
            pair = ((a, b, c), (a, c, d)) if (i + j) % 2 else ((a, b, d), (b, c, d))
            tris += pair
            groups += [VOLUMES["lower" if j < NY // 2 else "upper"]] * 2
    lines, line_groups = [], []
    for name, j in (("bottom", 0), ("top", NY), ("interface", NY // 2)):
        for i in range(NX):
            lines.append((n(i, j), n(i + 1, j)))
            line_groups.append(LINES[name])
    for j in range(NY - 1, NY):
        lines.append((n(0, j + 1), n(0, j)))
        line_groups.append(LINES["left"])
    return (
        nodes,
        np.array(tris),
        np.array(groups),
        np.array(lines),
        np.array(line_groups),
    )


def write_section(path: Path, fmt: str = "msh22", *, extra=()) -> Path:
    """Write the section as gmsh would, in msh 2.2 or 4.1 ASCII.

    ``extra`` adds groups, each ``(meshio cell type, cells, tag, name, dim)``.
    """
    nodes, tris, groups, lines, line_groups = _section()
    blocks = [("triangle", tris, groups, 2), ("line", lines, line_groups, 1)]
    names = {k: (v, 2) for k, v in VOLUMES.items()} | {
        k: (v, 1) for k, v in LINES.items()
    }
    for kind, data, tag, name, dim in extra:
        blocks.append((kind, np.asarray(data), np.full(len(data), tag), dim))
        names[name] = (tag, dim)
    if fmt == "msh41":
        path.write_text(_msh41(nodes, blocks, names))
        return path
    raw = meshio.Mesh(
        nodes,
        [(kind, data) for kind, data, _, _ in blocks],
        cell_data={
            "gmsh:physical": [tags for _, _, tags, _ in blocks],
            "gmsh:geometrical": [tags for _, _, tags, _ in blocks],
        },
        field_data={k: np.array(v) for k, v in names.items()},
    )
    meshio.write(path, raw, file_format="gmsh22", binary=False)
    return path


_GMSH_TYPES = {"line": 1, "triangle": 2, "tetra": 4}


def _msh41(nodes, blocks, names) -> str:
    """Msh 4.1 ASCII: one entity per physical group, its number the group's."""
    out = [
        "$MeshFormat",
        "4.1 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(names)),
    ]
    out += [f'{dim} {tag} "{name}"' for name, (tag, dim) in names.items()]
    out += ["$EndPhysicalNames", "$Entities"]
    entities = {
        d: sorted({(t, d) for _, _, tags, dd in blocks if dd == d for t in tags})
        for d in range(4)
    }
    out.append(" ".join(str(len(entities[d])) for d in range(4)))
    for d in range(1, 4):
        for tag, _ in entities[d]:
            out.append(f"{tag} 0 0 0 1 1 1 1 {tag} 0")
    out += ["$EndEntities", "$Nodes", f"1 {len(nodes)} 1 {len(nodes)}"]
    first = entities[2][0][0]
    out.append(f"2 {first} 0 {len(nodes)}")
    out += [str(i + 1) for i in range(len(nodes))]
    out += [" ".join(repr(float(c)) for c in x) for x in nodes]
    out.append("$EndNodes")
    parts = [
        (dim, int(tag), _GMSH_TYPES[kind], data[tags == tag])
        for kind, data, tags, dim in blocks
        for tag in np.unique(tags)
    ]
    total = sum(len(c) for *_, c in parts)
    out += ["$Elements", f"{len(parts)} {total} 1 {total}"]
    k = 1
    for dim, tag, kind, cells in parts:
        out.append(f"{dim} {tag} {kind} {len(cells)}")
        for cell in cells:
            out.append(" ".join(map(str, [k, *(np.asarray(cell) + 1)])))
            k += 1
    out.append("$EndElements")
    return "\n".join(out) + "\n"


@pytest.fixture(params=["msh22", "msh41"])
def section_path(request, tmp_path):
    return write_section(tmp_path / f"section_{request.param}.msh", request.param)


@pytest.fixture
def slab(section_path):
    return gfem.Mesh.from_gmsh_cross_section(section_path, length_unit="um")


def _section_edges():
    _, tris, _, _, _ = _section()
    edges = np.sort(tris[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0)


def _mean_edge_m():
    nodes, *_ = _section()
    e = _section_edges()
    return float(np.linalg.norm(nodes[e[:, 0]] - nodes[e[:, 1]], axis=1).mean()) * 1e-6


# --- regions -----------------------------------------------------------------------


def test_2d_groups_become_volumes_and_1d_groups_surfaces(slab):
    assert {r.name: r.tag for r in slab.regions if r.dim == 3} == VOLUMES
    surfaces = {r.name: r.tag for r in slab.regions if r.dim == 2}
    assert {k: surfaces[k] for k in LINES} == LINES
    assert slab.volumes == ("lower", "upper")
    assert set(slab.surfaces) == {*LINES, "front", "back"}
    # each triangle is one prism of three tetrahedra, in its region
    _, tris, groups, _, line_groups = _section()
    assert len(slab.tetrahedra) == 3 * len(tris)
    np.testing.assert_array_equal(slab.tetrahedron_tags, np.repeat(groups, 3))
    for name, tag in LINES.items():
        assert len(slab.elements(name)) == 2 * (line_groups == tag).sum()


def test_end_faces_front_and_back_cover_the_section(slab):
    t = slab.section_thickness
    nodes = np.asarray(slab.nodes)
    _, tris, *_ = _section()
    for name, z in (("front", 0.0), ("back", t)):
        faces = slab.triangles[slab.elements(name)]
        assert len(faces) == len(tris)
        np.testing.assert_allclose(nodes[faces][..., 2], z, atol=1e-18)
        # each end triangle lies on one tetrahedron, outside on the other side
        nb = slab.triangle_neighbours[slab.elements(name)]
        assert ((nb >= 0).sum(axis=1) == 1).all()
    assert set(slab.adjacent_volumes("front")) == set(VOLUMES)
    assert slab.adjacency("interface") in (
        (("lower", "upper"),),
        (("upper", "lower"),),
    )


def test_every_prism_split_conforms(slab):
    """Each face is shared by two tetrahedra or lies on the slab's boundary."""
    nodes = np.asarray(slab.imported_nodes)
    tets = slab.tetrahedra
    faces = np.sort(tets[:, _TET_FACES].reshape(-1, 3), axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    assert counts.max() == 2
    boundary = unique[counts == 1]
    # a split that disagreed on a side would leave its two cuts as extra faces
    _, tris, _, _, _ = _section()
    edges = _section_edges()
    outer = sum(
        1
        for e in edges
        if (np.sort(tris[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1) == e)
        .all(axis=1)
        .sum()
        == 1
    )
    assert len(boundary) == 2 * len(tris) + 2 * outer
    # the reader's own face search places every boundary face: no orphan
    nb = _triangle_neighbours(nodes, tets, boundary, Path("slab"))
    assert ((nb >= 0).sum(axis=1) == 1).all()
    # positive tetrahedra filling width x height x thickness
    x = nodes[tets]
    det = np.linalg.det(x[:, 1:] - x[:, :1])
    assert (det > 0).all()
    np.testing.assert_allclose(
        det.sum() / 6, WIDTH * HEIGHT * 1e-12 * slab.section_thickness, rtol=1e-12
    )


def test_default_thickness_is_the_mean_in_plane_edge(slab):
    assert slab.section_thickness == pytest.approx(_mean_edge_m(), rel=1e-12)
    assert "section_thickness=" in repr(slab)


def test_thickness_is_in_metres(section_path):
    mesh = gfem.Mesh.from_gmsh_cross_section(section_path, thickness=2e-7)
    assert mesh.section_thickness == 2e-7
    assert np.asarray(mesh.nodes)[:, 2].max() == 2e-7


@pytest.mark.parametrize("bad", [0.0, -1e-6, float("nan"), "thick"])
def test_a_thickness_that_is_not_a_positive_length_raises(section_path, bad):
    with pytest.raises(ValueError, match="positive length in metres"):
        gfem.Mesh.from_gmsh_cross_section(section_path, thickness=bad)


def test_an_ordinary_mesh_has_no_section_thickness():
    mesh = gfem.Mesh.from_gmsh(TWO_BOX, length_unit="um")
    assert mesh.section_thickness is None
    assert "section_thickness" not in repr(mesh)


# --- the split depends on coordinates, not node numbers ------------------------------

CELL = 1.0  # um, the periodic cell's side
NC = 4  # cells along each side
SIDES = {"x_min": 11, "x_max": 12, "y_min": 13, "y_max": 14}


def _renumbered(path: Path, out: Path, seed: int) -> Path:
    """The same section with its node numbers shuffled."""
    raw = meshio.read(path, file_format="gmsh")
    perm = np.random.default_rng(seed).permutation(len(raw.points))
    new = np.empty_like(perm)
    new[perm] = np.arange(len(perm))
    shuffled = meshio.Mesh(
        raw.points[perm],
        [(c.type, new[c.data]) for c in raw.cells],
        cell_data=raw.cell_data,
        field_data=raw.field_data,
    )
    meshio.write(out, shuffled, file_format="gmsh22", binary=False)
    return out


def write_periodic_cell(path: Path, *, shuffled=False, sheared=False) -> Path:
    r"""A periodic cell of side `CELL`, numbered as gmsh numbers a periodic mesh.

    Opposite sides hold the same nodes up to the lattice translation, as
    ``setPeriodic`` meshes them, but gmsh numbers each side along its own
    direction around the cell: the corners first, then the bottom and right
    sides forward, the top and left sides backward, then the inner nodes
    (perturbed, on alternating diagonals). ``shuffled`` shuffles the node
    numbers; ``sheared`` shears the cell onto a hexagonal lattice, with the
    lattice vectors :math:`a_1 = (L, 0)` and :math:`a_2 = (L/2, \sqrt{3} L/2)`.
    """
    rng = np.random.default_rng(1)
    order = [(0, 0), (NC, 0), (NC, NC), (0, NC)]
    order += [(i, 0) for i in range(1, NC)] + [(NC, j) for j in range(1, NC)]
    order += [(i, NC) for i in range(NC - 1, 0, -1)]
    order += [(0, j) for j in range(NC - 1, 0, -1)]
    order += [(i, j) for j in range(1, NC) for i in range(1, NC)]
    number = {ij: k for k, ij in enumerate(order)}
    h = CELL / NC
    nodes = np.array([[i * h, j * h, 0.0] for i, j in order])
    inner = len(order) - (NC - 1) ** 2
    nodes[inner:, :2] += rng.uniform(-0.15, 0.15, ((NC - 1) ** 2, 2)) * h
    if sheared:
        x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
        nodes[:, 0], nodes[:, 1] = x + 0.5 * y, y * np.sqrt(3) / 2
    tris = []
    for j in range(NC):
        for i in range(NC):
            a, b = number[i, j], number[i + 1, j]
            c, d = number[i + 1, j + 1], number[i, j + 1]
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 else [(a, b, d), (b, c, d)]
    lines = {
        "x_min": [(number[0, j], number[0, j + 1]) for j in range(NC)],
        "x_max": [(number[NC, j], number[NC, j + 1]) for j in range(NC)],
        "y_min": [(number[i, 0], number[i + 1, 0]) for i in range(NC)],
        "y_max": [(number[i, NC], number[i + 1, NC]) for i in range(NC)],
    }
    line_tags = np.repeat([SIDES[k] for k in lines], NC)
    raw = meshio.Mesh(
        nodes,
        [("triangle", np.array(tris)), ("line", np.concatenate(list(lines.values())))],
        cell_data={
            "gmsh:physical": [np.ones(len(tris), int), line_tags],
            "gmsh:geometrical": [np.ones(len(tris), int), line_tags],
        },
        field_data={"cell": np.array([1, 2])}
        | {k: np.array([v, 1]) for k, v in SIDES.items()},
    )
    meshio.write(path, raw, file_format="gmsh22", binary=False)
    return _renumbered(path, path, 2) if shuffled else path


def _by_coordinates(mesh, cells):
    """Each cell as the sorted set of its node coordinates, the cells sorted."""
    x = np.asarray(mesh.imported_nodes)[cells]
    return sorted(tuple(sorted(map(tuple, c))) for c in x.tolist())


def test_the_split_does_not_depend_on_the_node_numbering(tmp_path):
    path = write_section(tmp_path / "s.msh")
    slab = gfem.Mesh.from_gmsh_cross_section(path, length_unit="um")
    other = gfem.Mesh.from_gmsh_cross_section(
        _renumbered(path, tmp_path / "r.msh", 3), length_unit="um"
    )
    assert _by_coordinates(other, other.tetrahedra) == _by_coordinates(
        slab, slab.tetrahedra
    )
    assert _by_coordinates(other, other.triangles) == _by_coordinates(
        slab, slab.triangles
    )


@pytest.mark.parametrize("shuffled", [False, True], ids=["gmsh_order", "shuffled"])
@pytest.mark.parametrize("sheared", [False, True], ids=["square", "hexagonal"])
def test_a_periodic_section_pairs_whatever_its_node_numbering(
    tmp_path, shuffled, sheared
):
    path = write_periodic_cell(tmp_path / "c.msh", shuffled=shuffled, sheared=sheared)
    model = gfem.Model(
        gfem.Mesh.from_gmsh_cross_section(path, length_unit="um"),
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
    )
    a2 = (0.5, np.sqrt(3) / 2) if sheared else (0.0, 1.0)
    np.testing.assert_allclose(
        model.periodic_translations,
        [(CELL * 1e-6, 0.0, 0.0), (a2[0] * CELL * 1e-6, a2[1] * CELL * 1e-6, 0.0)],
        atol=1e-18,
    )


# --- refusals ------------------------------------------------------------------------


def test_the_cross_section_reader_refuses_a_tetrahedral_file_naming_from_gmsh():
    with pytest.raises(
        ValueError, match=r"3D mesh\. Read it with gfem\.Mesh\.from_gmsh"
    ):
        gfem.Mesh.from_gmsh_cross_section(TWO_BOX, length_unit="um")


def test_from_gmsh_reads_a_triangle_only_file_as_a_2d_mesh_not_a_slab(section_path):
    mesh = gfem.Mesh.from_gmsh(section_path, length_unit="um")
    assert mesh.dim == 2
    assert mesh.section_thickness is None


def test_a_group_named_like_an_end_face_raises(tmp_path):
    path = write_section(tmp_path / "s.msh", extra=[("line", [[0, 0]], 20, "front", 1)])
    with pytest.raises(ValueError, match=r"'front'.*end faces"):
        gfem.Mesh.from_gmsh_cross_section(path)


def test_a_line_off_the_triangle_edges_raises(tmp_path):
    _, tris, *_ = _section()
    far = [[tris[0][0], tris[-1][0]]]  # opposite corners of the section
    path = write_section(tmp_path / "s.msh", extra=[("line", far, 20, "stray", 1)])
    with pytest.raises(ValueError, match="'stray' is not an edge"):
        gfem.Mesh.from_gmsh_cross_section(path)


def test_ungrouped_triangles_raise(tmp_path):
    _, tris, *_ = _section()
    path = write_section(tmp_path / "s.msh", extra=[("triangle", tris[:1], 0, "x", 2)])
    with pytest.raises(
        ValueError, match="1 triangles lie outside any 2D physical group"
    ):
        gfem.Mesh.from_gmsh_cross_section(path)


# --- the thickness survives ----------------------------------------------------------

PLATE = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
ORDER_1 = gfem.SolverOptions(order=1, linear_solver="direct")


@pytest.fixture
def model(section_path):
    mesh = gfem.Mesh.from_gmsh_cross_section(section_path, thickness=3e-7)
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
    )


def test_model_replace_keeps_the_thickness(model):
    other = model.replace(
        materials={"lower": gfem.Material(), "upper": gfem.Material()}
    )
    assert other.mesh.section_thickness == 3e-7


def test_a_morphed_mesh_keeps_the_thickness(model):
    morph = boundary_morph(model, "interface", grows="lower")
    moved = morph(np.full(morph.n_params, 1e-8))
    assert moved.section_thickness == 3e-7
    assert not np.array_equal(np.asarray(moved.nodes), moved.imported_nodes)


def test_refine_mesh_of_an_em_problem_keeps_the_thickness(model, tmp_path):
    corner = gfem.Electrostatic(terminals={"bottom": "bottom"}, ground="left")
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            model,
            corner,
            tolerance=1e-4,
            max_cycles=1,
            options=gfem.SolverOptions(order=2, linear_solver="direct"),
        )
    assert len(ref.model.mesh.tetrahedra) > len(model.mesh.tetrahedra)
    assert ref.model.mesh.section_thickness == 3e-7
    # a refined mesh has no file: a job stores its arrays, and the thickness
    path = gfem.Job(ref.model, PLATE, ORDER_1).to_json(tmp_path / "j.json")
    loaded = gfem.Job.from_json(path).model.mesh
    assert loaded.source is None
    assert loaded.section_thickness == 3e-7


def test_a_job_and_its_result_keep_the_thickness(model, tmp_path):
    path = gfem.Job(model, PLATE, ORDER_1).to_json(tmp_path / "job.json")
    assert (
        json.loads(path.read_text())["job"]["model"]["mesh"]["section_thickness"]
        == 3e-7
    )
    job = gfem.Job.from_json(path)
    assert job.model.mesh.section_thickness == 3e-7
    np.testing.assert_array_equal(
        job.model.mesh.imported_nodes, model.mesh.imported_nodes
    )
    job.run(out=tmp_path / "results")
    loaded = gfem.load_result(tmp_path / "results")
    assert loaded.potential["top"].mesh.section_thickness == 3e-7


def test_an_ordinary_mesh_job_writes_no_thickness_key(tmp_path):
    model = gfem.Model(
        gfem.Mesh.from_gmsh(TWO_BOX),
        materials={"sub": gfem.Material(), "air": gfem.Material()},
    )
    path = gfem.Job(model, gfem.Electrostatic(terminals=["metal_xy"]), ORDER_1).to_json(
        tmp_path / "j.json"
    )
    assert "section_thickness" not in path.read_text()
