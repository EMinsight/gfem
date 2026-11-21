"""The pinned quadrature rule reaches the elements of both mesh dimensions.

NGSolve applies a rule only to the element type it is keyed by and silently
uses its own order elsewhere, so each dimension's volume and surface elements
must be keyed: tetrahedra and triangles in 3D, triangles and segments in 2D.
"""

import netgen.meshing as ngm
import ngsolve as ngs
import pytest

from gfem._engine import quadrature


def _unit_square():
    """The unit square as two triangles, its bottom edge a segment "bottom"."""
    m = ngm.Mesh(dim=2)
    p = [
        m.Add(ngm.MeshPoint(ngm.Pnt(x, y, 0)))
        for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]
    m.SetMaterial(1, "square")
    m.Add(ngm.Element2D(1, [p[0], p[1], p[2]]))
    m.Add(ngm.Element2D(1, [p[0], p[2], p[3]]))
    m.Add(ngm.Element1D([p[0], p[1]], index=1))
    m.SetBCName(0, "bottom")
    return ngs.Mesh(m)


def _unit_cube():
    from netgen.csg import CSGeometry, OrthoBrick, Pnt

    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1)).bc("wall"))
    return ngs.Mesh(geo.GenerateMesh(maxh=10.0))


def _integral(mesh, k, measure):
    return ngs.Integrate(ngs.x**k * measure, mesh)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_x_to_the_k_is_exact_to_degree_2p_on_triangles(order):
    mesh = _unit_square()
    for k in range(2 * order + 1):
        assert _integral(mesh, k, quadrature.dx(order)) == pytest.approx(
            1 / (k + 1), rel=1e-13
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_x_to_the_k_is_exact_to_degree_2p_on_segments(order):
    mesh = _unit_square()
    ds = quadrature.ds(order, definedon=mesh.Boundaries("bottom"))
    for k in range(2 * order + 1):
        assert _integral(mesh, k, ds) == pytest.approx(1 / (k + 1), rel=1e-13)


@pytest.mark.parametrize("order", [1, 2])
def test_the_pinned_rule_is_the_one_used_on_triangles_and_segments(order):
    # one degree past the rule, a pinned rule errs where NGSolve's own would not
    mesh = _unit_square()
    k = 2 * order + 2
    exact = 1 / (k + 1)
    on_area = _integral(mesh, k, quadrature.dx(order))
    on_line = _integral(
        mesh, k, quadrature.ds(order, definedon=mesh.Boundaries("bottom"))
    )
    assert abs(on_area - exact) > 1e-6
    assert abs(on_line - exact) > 1e-6


@pytest.mark.parametrize("order", [1, 2])
def test_tetrahedra_and_their_faces_keep_the_rule(order):
    # on the cube's six faces, x**k (k > 0) integrates to 4 / (k + 1) + 1
    mesh = _unit_cube()
    for k in range(1, 2 * order + 1):
        assert _integral(mesh, k, quadrature.dx(order)) == pytest.approx(
            1 / (k + 1), rel=1e-12
        )
        assert _integral(mesh, k, quadrature.ds(order)) == pytest.approx(
            4 / (k + 1) + 1, rel=1e-12
        )
    k = 2 * order + 2
    assert abs(_integral(mesh, k, quadrature.dx(order)) - 1 / (k + 1)) > 1e-6
    assert abs(_integral(mesh, k, quadrature.ds(order)) - 4 / (k + 1) - 1) > 1e-6
