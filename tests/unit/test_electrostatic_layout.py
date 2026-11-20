"""What an electrostatic solve across ranks rests on, checked on one process.

The terminal lifting is the indicator of its vertex DOFs (ADR 0501), which must be
the lifting ``gf.Set`` gave; the map of an H1 piece to one-process numbering must
name each DOF as `GetDofNrs` does, entity by entity.
"""

import mpi_cases
import ngsolve as ngs
import numpy as np
import pytest

from gfem._engine import fields
from gfem._engine.cache import discretization
from gfem._engine.domain import boundary_region, h1_space
from gfem._engine.electrostatic import terminal_lifting


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("surface", ["top", "oxide__nitride"])
def test_the_lifting_is_the_one_gf_set_gives(tmp_path, order, surface):
    model = mpi_cases.stack(tmp_path)
    built = discretization(model.mesh).built
    space = h1_space(model, built, order, ())
    gf = ngs.GridFunction(space)
    gf.Set(ngs.CF(1.0), definedon=boundary_region(built, (surface,)))
    lift = terminal_lifting(model, built, space, (surface,))
    assert set(np.unique(lift)) == {0.0, 1.0}
    np.testing.assert_allclose(lift, gf.vec.FV().NumPy(), rtol=0, atol=1e-12)


# --- the H1 map to one-process numbering -------------------------------------------


def _rule(mesh, support, order):
    """Each entity's one-process DOFs by the rule, keyed as `GetDofNrs` walks them.

    Used vertices by gfem node, then each support edge's ``p-1`` DOFs (edges
    lexicographic in gfem nodes), each face's ``(p-1)(p-2)/2``, each cell's
    ``(p-1)(p-2)(p-3)/6`` in the H(curl) map's cell order.
    """
    tables = fields._one_process_tables(mesh, support, ())
    cells = tables["cells"]
    vertices = np.unique(mesh.tetrahedra[cells >= 0])
    n_nodes = len(mesh.imported_nodes)
    on_edge = order - 1
    on_face = (order - 1) * (order - 2) // 2
    in_cell = (order - 1) * (order - 2) * (order - 3) // 6
    first_edge = len(vertices)
    first_face = first_edge + len(tables["edges"]) * on_edge
    first_cell = first_face + len(tables["faces"]) * on_face

    def vertex(a):
        return [int(np.searchsorted(vertices, a))]

    def edge(a, b):
        k = np.searchsorted(tables["edges"], a * n_nodes + b)
        return list(first_edge + k * on_edge + np.arange(on_edge))

    def face(a, b, c):
        low = np.searchsorted(tables["edges"], a * n_nodes + b)
        k = np.searchsorted(tables["faces"], low * n_nodes + c)
        return list(first_face + k * on_face + np.arange(on_face))

    def cell(tet):
        return list(first_cell + cells[tet] * in_cell + np.arange(in_cell))

    return vertex, edge, face, cell


def _walk(fes, built):
    """``(rule key, DOFs)`` of every entity holding DOFs, by `GetDofNrs`."""
    ngm, node_of = built.mesh, built.node_order
    for nodes, node_type, kind in (
        (range(ngm.nv), ngs.VERTEX, "vertex"),
        ((e.nr for e in ngm.edges), ngs.EDGE, "edge"),
        ((f.nr for f in ngm.faces), ngs.FACE, "face"),
    ):
        for nr in nodes:
            node = ngs.NodeId(node_type, nr)
            dofs = [d for d in fes.GetDofNrs(node) if d >= 0]
            if dofs:
                vertices = ngm[node].vertices if kind != "vertex" else [nr]
                ids = sorted(int(node_of[getattr(v, "nr", v)]) for v in vertices)
                yield (kind, *ids), dofs
    for el in range(ngm.ne):
        dofs = [d for d in fes.GetDofNrs(ngs.NodeId(ngs.CELL, el)) if d >= 0]
        if dofs:
            yield ("cell", int(built.tetrahedron_order[el])), dofs


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "support", [("air", "diel", "metal"), ("air", "diel"), ("diel",)]
)
def test_the_h1_map_names_each_dof_as_get_dof_nrs_does(order, support):
    mesh = mpi_cases.permuted(mpi_cases.box(3))
    built, fes, _ = fields.fe_space(mesh, support, (), "h1", order)
    numbers = fields.one_process_dofs(mesh, support, (), order, space="h1")
    kinds = ("vertex", "edge", "face", "cell")
    rules = dict(zip(kinds, _rule(mesh, support, order), strict=True))
    walked = 0
    for (kind, *ids), dofs in _walk(fes, built):
        assert list(numbers[dofs]) == rules[kind](*ids), (kind, ids)
        walked += len(dofs)
    assert walked == fes.ndof
    # one process's own numbering is the map's
    np.testing.assert_array_equal(numbers, np.arange(fes.ndof))
