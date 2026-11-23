"""The p-multigrid: its direct set and the topological count of it."""

import math

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import pmg


def test_direct_set_regular_tetrahedron_aspect():
    # longest edge 1 over shortest altitude sqrt(2/3): sqrt(6)/2
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0.5, math.sqrt(3) / 6, 0]]
    )
    nodes[3, 2] = math.sqrt(2 / 3)
    aspect = pmg.cell_aspect(nodes, np.array([[0, 1, 2, 3]]))
    assert aspect == pytest.approx([math.sqrt(6) / 2])


def test_direct_set_sound_structured_mesh_has_no_poor_cells():
    # n^3 Kuhn cubes: 3n(n+1)^2 axis edges, 3n^2(n+1) face and n^3 body diagonals
    n = 4
    nodes, tets, *_ = _grid(cells=n, shell=0.0, film=())
    whitney = 3 * n * (n + 1) ** 2 + 3 * n**2 * (n + 1) + n**3
    assert pmg.cell_aspect(nodes, tets).max() < 3.5
    for order in (2, 3):
        count = pmg.direct_set_count(nodes, tets, order)
        assert count == (whitney, whitney, 0)


#: A 1 nm film across the unit box (um): its cells have aspect 50 and more.
FILM = (0.5, 0.501)


def _grid(*, cells=4, shell=0.25, film=FILM):
    """[0, 1]^3 "air" in a "pml" shell, cut through by a film at z in ``film``.

    The shell covers every side but +z; the outer faces are "outer". Returns
    `structured_meshes.structured`'s arrays.
    """

    def line(lo, hi, extra=()):
        return np.unique(
            np.r_[np.linspace(lo, hi, round((hi - lo) * cells) + 1), extra]
        )

    edges = (line(-shell, 1 + shell), line(-shell, 1 + shell), line(-shell, 1, film))

    def volume(c):
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    lo, hi = np.array([e[0] for e in edges]), np.array([e[-1] for e in edges])

    def surface(c):
        return "outer" if np.isclose(c, lo).any() or np.isclose(c, hi).any() else ""

    return sm.structured(edges, volume, surface)


def _box(path, **kw):
    """`_grid` as a `gfem.Mesh` in micrometres."""
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, _grid(**kw)), length_unit="um")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return _box(tmp_path_factory.mktemp("pmg") / "box.msh")


def _built(mesh, order):
    """The mesh's NGSolve space, its cells in NGSolve order and their PML marks."""
    built = to_ngsolve(mesh)
    space = ngs.Compress(ngs.HCurl(built.mesh, order=order, type1=True, complex=True))
    tets = mesh.tetrahedra[built.cell_order]
    pml = mesh.tetrahedron_tags[built.cell_order] == mesh.region("pml").tag
    return space, tets, pml


@pytest.mark.parametrize("order", [2, 3])
def test_direct_set_count_is_the_built_set(box, order):
    mesh = box
    space, tets, pml = _built(mesh, order)
    nodes = np.asarray(mesh.nodes)

    count = pmg.direct_set_count(nodes, tets, order, pml)
    built = pmg.direct_set(space, nodes, tets, np.arange(len(tets)), pml)

    poor = pmg.cell_aspect(nodes, tets) > pmg.POOR_ASPECT
    assert (poor & pml).any()
    assert (poor & ~pml).any()
    assert 0 < count.whitney < count.direct < space.ndof
    assert built.dofs.NumSet() == count.direct
    assert built.poor_cells == count.poor_cells


def test_direct_set_counts_a_cell_both_pml_and_poor_once():
    # one sliver PML cell: its order-2 element, 6 edges x 2 + 4 faces x 2 DOFs
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, 0.01]])
    tets = np.array([[0, 1, 2, 3]])
    assert pmg.cell_aspect(nodes, tets)[0] > pmg.POOR_ASPECT
    count = pmg.direct_set_count(nodes, tets, 2, np.array([True]))
    assert count == (20, 6, 1)


def test_direct_set_poor_cell_share_halves_with_h():
    # a spanning film is a surface of cells: its share of the volume goes as h
    share = []
    for cells in (4, 8):
        nodes, tets, *_ = _grid(cells=cells, shell=0.0)
        share.append(pmg.direct_set_count(nodes, tets, 2).poor_cells / len(tets))
    assert share[0] > 0
    assert share[1] < 0.6 * share[0]


def test_direct_set_on_part_of_the_mesh_holds_its_whitney_dofs(box):
    # a space on "air" alone: its cells only, the shell's left out
    mesh = box
    built = to_ngsolve(mesh)
    space = ngs.Compress(
        ngs.HCurl(
            built.mesh,
            order=2,
            type1=True,
            complex=True,
            definedon=built.mesh.Materials("air"),
        )
    )
    air = mesh.tetrahedron_tags[built.cell_order] == mesh.region("air").tag
    tets, nodes = mesh.tetrahedra[built.cell_order][air], np.asarray(mesh.nodes)

    direct = pmg.direct_set(space, nodes, tets, np.flatnonzero(air))

    assert direct.dofs.NumSet() == pmg.direct_set_count(nodes, tets, 2).direct
    assert direct.pml_dofs == 0
    edges = {
        e.nr for el in built.mesh.Elements(ngs.VOL) if air[el.nr] for e in el.edges
    }
    assert all(direct.dofs[space.GetDofNrs(ngs.NodeId(ngs.EDGE, e))[0]] for e in edges)
    with pytest.raises(ValueError, match="own cells"):
        pmg.direct_set(space, nodes, mesh.tetrahedra, np.arange(len(mesh.tetrahedra)))


def test_direct_set_leaves_fixed_dofs_out(box):
    mesh = box
    built = to_ngsolve(mesh)
    space = ngs.Compress(
        ngs.HCurl(built.mesh, order=2, type1=True, complex=True, dirichlet="outer")
    )
    tets, nodes = mesh.tetrahedra[built.cell_order], np.asarray(mesh.nodes)
    fixed = space.ndof - space.FreeDofs().NumSet()

    direct = pmg.direct_set(space, nodes, tets, np.arange(len(tets)))

    assert fixed > 0
    assert (direct.dofs & ~space.FreeDofs()).NumSet() == 0
    assert direct.dofs.NumSet() < pmg.direct_set_count(nodes, tets, 2).direct
