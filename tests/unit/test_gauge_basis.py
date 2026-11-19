"""The tree-cotree gauge basis T and its gauge template (ADR 0215, 0217).

On the parallel-plate line of test_driven.py and the tutorial CPW: T is
square and nonsingular from the free gauge space Y to the free driven space
V, the curl pieces hold exact zeros on its gradient columns, the reference
conductor does not change the field, and the template's pattern is the
symbolic gauged pattern.
"""

import dataclasses
from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import structured_meshes as sm

import gfem
from gfem._engine import driven, gauge

L, W, H = 10.0, 2.0, 1.0  # mm
CPW = Path(__file__).parents[2] / "docs/tutorials/data/cpw.msh"


def _line_mesh(tmp, **kwargs):
    path = tmp / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (10, 1, 1), **kwargs))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, *, pec=True, boundaries=(), excluded=()):
    walls = [gfem.PEC("bottom"), gfem.PEC("top")] if pec else []
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        excluded=excluded,
        boundaries=[*walls, *boundaries],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return _line(_line_mesh(tmp_path_factory.mktemp("line")))


@pytest.fixture(scope="module")
def cpw():
    mesh = gfem.Mesh.from_gmsh(CPW, length_unit="um")
    return gfem.Model(
        mesh,
        materials={"substrate": gfem.Material(eps_r=4.0), "air": gfem.Material()},
        excluded=("signal_metal", "ground_metal"),
        boundaries=[
            gfem.PEC("signal"),
            gfem.PEC("signal_wall"),
            gfem.PEC("ground"),
            gfem.PEC("backside"),
            gfem.Absorbing("top"),
            gfem.Absorbing("side"),
        ],
        ports=[
            gfem.LumpedPort("p1", ("p1a", "p1b"), ("+y", "-y"), impedance=50.0),
            gfem.LumpedPort("p2", ("p2a", "p2b"), ("+y", "-y"), impedance=50.0),
        ],
    )


def _assert_square_nonsingular(basis):
    t = basis.T
    assert t.shape == (len(basis.v_dofs), len(basis.y_dofs))
    assert t.shape[0] == t.shape[1]
    # SuperLU raises on an exactly singular matrix; ticket 03 saw pivots >= 0.83
    lu = spla.splu(t.tocsc())
    assert np.abs(lu.U.diagonal()).min() > 1e-3


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("name", ["line", "cpw"])
def test_basis_is_square_and_nonsingular(request, name, order):
    model = request.getfixturevalue(name)
    basis = gauge.gauge_basis(driven.driven_operator(model, order))
    _assert_square_nonsingular(basis)
    # the line's two plates: one reference, one floating potential
    if name == "line":
        assert (basis.conductors, basis.floating, basis.regions) == (2, 1, 1)


def _free_block(basis, matrix):
    data, cols, rowptr = matrix.CSR()
    a = sp.csr_matrix(
        (np.array(data), np.array(cols), np.array(rowptr)),
        shape=(matrix.height, matrix.width),
    )
    return a[basis.v_dofs][:, basis.v_dofs]


def _assembled(operator, piece):
    u, v = operator.space.TnT()
    form = ngs.BilinearForm(operator.space, symmetric=False)
    form += piece.integrand(u, v)
    form.Assemble()
    return form.mat


@pytest.mark.parametrize("order", [1, 2])
def test_curl_pieces_are_exactly_zero_on_gradients(tmp_path, order):
    model = _line(_line_mesh(tmp_path), boundaries=[gfem.Absorbing("side", order=2)])
    operator = driven.driven_operator(model, order)
    basis = gauge.gauge_basis(operator)
    kinds = {piece.kind for piece in operator.pieces}
    assert {"curl", "curl trace"} <= kinds
    for piece in operator.pieces:
        matrix = _assembled(operator, piece)
        hat = basis.transform(matrix, piece.kind)
        if piece.kind in ("curl", "curl trace"):
            gradients = hat[:, basis.cotree :].tocsc()
            assert gradients.nnz == 0
            assert hat[basis.cotree :].nnz == 0
            assert hat.nnz > 0
            # the cotree block is the floating-point product's
            full = basis.TT @ (_free_block(basis, matrix) @ basis.T)
            n = basis.cotree
            assert np.allclose(hat[:n, :n].toarray(), full[:n, :n].toarray())
        else:  # a mass or trace piece acts on gradients
            assert hat[basis.cotree :, basis.cotree :].count_nonzero() > 0


def _gauged_solve(operator, basis, frequency, load):
    """The field ``T d (D Â D)^{-1} d T^T b`` of the gauged system, by SuperLU."""
    omega = 2 * np.pi * frequency
    a = sum(
        complex(piece.theta(omega))
        * basis.transform(_assembled(operator, piece), piece.kind)
        for piece in operator.pieces
    )
    diagonal = np.abs(a.diagonal())
    d = 1 / np.sqrt(np.where(diagonal > 0, diagonal, 1.0))
    scaled = sp.diags(d) @ a @ sp.diags(d)
    y = spla.splu(sp.csc_matrix(scaled)).solve(d * (basis.TT @ load))
    return basis.T @ (d * y)


@pytest.mark.parametrize("order", [1, 2])
def test_reference_conductor_does_not_change_the_field(cpw, order):
    operator = driven.driven_operator(cpw, order)
    largest = gauge.gauge_basis(operator)
    smallest = gauge.gauge_basis(operator, reference="smallest")
    # signal, two ground strips, backside: one reference, three floating
    assert (largest.conductors, largest.floating) == (4, 3)
    assert largest.key != smallest.key
    assert np.array_equal(largest.v_dofs, smallest.v_dofs)
    rng = np.random.default_rng(0)
    n = len(largest.v_dofs)
    load = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a = _gauged_solve(operator, largest, 1e4, load)
    b = _gauged_solve(operator, smallest, 1e4, load)
    assert np.abs(a - b).max() <= 1e-13 * np.abs(a).max()


@pytest.mark.parametrize("order", [1, 2])
def test_pec_free_and_two_region_models_build(tmp_path, order):
    # no PEC: one region, its vertex constant dropped
    free = gauge.gauge_basis(
        driven.driven_operator(_line(_line_mesh(tmp_path), pec=False), order)
    )
    _assert_square_nonsingular(free)
    assert (free.conductors, free.floating, free.regions) == (0, 0, 1)
    # an excluded core cuts the line in two regions, each with two plates
    split = _line(_line_mesh(tmp_path, core=(4.0, 6.0)), excluded=("core",))
    two = gauge.gauge_basis(driven.driven_operator(split, order))
    _assert_square_nonsingular(two)
    assert (two.conductors, two.floating, two.regions) == (4, 2, 2)


def _moved(model, topology=None):
    """``model`` with its nodes moved along x (sin profile), same elements."""
    mesh = model.mesh
    nodes = np.array(mesh.nodes)
    length = nodes[:, 0].max()
    nodes[:, 0] += 0.02 * length * np.sin(np.pi * nodes[:, 0] / length)
    moved = gfem.Mesh(nodes, mesh._topology if topology is None else topology)
    return model.replace(mesh=moved)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_basis_survives_a_node_move(cpw, order):
    basis = gauge.gauge_basis(driven.driven_operator(cpw, order))
    moved = driven.driven_operator(_moved(cpw), order)
    assert moved.space is not driven.driven_operator(cpw, order).space
    assert gauge.gauge_basis(moved) is basis  # a cache hit
    # built again on the moved nodes (a copy of the topology misses the cache)
    copy = dataclasses.replace(cpw.mesh._topology)
    again = gauge.gauge_basis(driven.driven_operator(_moved(cpw, copy), order))
    assert again is not basis
    for a, b in ((again.T, basis.T), (again.pattern, basis.pattern)):
        assert np.array_equal(a.indptr, b.indptr)
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.data, b.data)
    assert np.array_equal(again.y_dofs, basis.y_dofs)
    assert again.key == basis.key


@pytest.mark.parametrize("order", [1, 2])
def test_template_free_csr_is_the_gauged_pattern(cpw, order):
    operator = driven.driven_operator(cpw, order)
    basis = gauge.gauge_basis(operator)
    template, pattern = basis.template, basis.pattern
    assert template.is_complex
    assert (template.height, template.width) == (basis.ndof_y,) * 2
    assert len(basis.free) == basis.ndof_y
    assert basis.free.NumSet() == len(basis.y_dofs) == pattern.shape[0]
    _, cols, rowptr = template.CSR()
    lengths = np.diff(np.asarray(rowptr))
    fixed = np.ones(basis.ndof_y, bool)
    fixed[basis.y_dofs] = False
    assert not lengths[fixed].any()
    assert np.array_equal(lengths[basis.y_dofs], np.diff(pattern.indptr))
    assert np.array_equal(np.asarray(cols), basis.y_dofs[pattern.indices])
    # every gauged piece lies in it, at positions of the template's values
    for piece in operator.pieces:
        hat = basis.transform(_assembled(operator, piece), piece.kind)
        at = basis.positions(hat)
        assert len(at) == hat.nnz
        assert np.array_equal(pattern.indices[at], hat.indices)
