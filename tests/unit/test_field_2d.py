"""A scalar `gfem.Field` on a 2D mesh: the kept Ez or Hz of a 2D model's solve.

It is evaluated at (m, 2) points in the (x, y) plane, refuses (m, 3) points,
exports triangle cells and has no flux density. The fields here are set
directly on the engine's H1 space, independently of any solve.
"""

import meshio
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine.fields import fe_space

UM = 1e-6
OMEGA = 2 * np.pi * 1e9


def _area(c):
    return "left" if c[0] < 2 else "right"


def _outer(c):
    eps = 1e-9
    on = c[0] < eps or c[0] > 4 - eps or c[1] < eps or c[1] > 3 - eps
    return "outer" if on else ""


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    """A 4 um x 3 um rectangle of two areas, "left" (x < 2 um) and "right"."""
    edges = (np.linspace(0, 4, 5), np.linspace(0, 3, 4), np.linspace(0, 1, 2))
    face = sm.z0_face(sm.structured(edges, _area, _outer))
    path = tmp_path_factory.mktemp("face") / "face.msh"
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, face), length_unit="um")


def _quadratic(x, y):
    """A quadratic in the plane, x and y in m."""
    return (x / UM) ** 2 - 2 * (x / UM) * (y / UM) + 0.5 * (y / UM) ** 2 + x / UM - 3


def _field(mesh, support=("left", "right"), quantity="Ez", unit="V/m"):
    """The quadratic interpolated in P2 (exactly), as a complex field."""
    built, fes, _ = fe_space(mesh, support, (), "h1", 2)
    gf = ngs.GridFunction(fes)
    x, y = ngs.x * built.scale, ngs.y * built.scale
    gf.Set(_quadratic(x, y))
    real = gf.vec.FV().NumPy().copy()
    return gfem.Field(
        coefficients=(1 - 2j) * real,
        mesh=mesh,
        quantity=quantity,
        unit=unit,
        space="h1",
        order=2,
        support=support,
        omega=OMEGA,
    )


def test_at_is_exact_for_a_quadratic_at_points_in_the_plane(mesh):
    rng = np.random.default_rng(0)
    points = rng.uniform((0, 0), (4 * UM, 3 * UM), size=(50, 2))
    values = np.asarray(_field(mesh).at(points))
    assert values.shape == (50,)
    expected = (1 - 2j) * _quadratic(points[:, 0], points[:, 1])
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


def test_at_one_point_in_the_plane_gives_one_value(mesh):
    value = _field(mesh).at(np.array([1.5 * UM, 2.5 * UM]))
    assert value.shape == ()
    np.testing.assert_allclose(value, (1 - 2j) * _quadratic(1.5 * UM, 2.5 * UM))


@pytest.mark.parametrize("points", [np.zeros((4, 3)), np.zeros(3)])
def test_at_refuses_points_with_a_z_coordinate(mesh, points):
    with pytest.raises(ValueError, match=r"\(x, y\) plane: pass \(m, 2\) points"):
        _field(mesh).at(points)


def test_at_outside_the_support_names_the_area(mesh):
    with pytest.raises(ValueError, match="'right'"):
        _field(mesh, support=("left",)).at(np.array([[3 * UM, 1 * UM]]))


def test_at_outside_the_mesh_raises(mesh):
    with pytest.raises(ValueError, match="outside the mesh"):
        _field(mesh).at(np.array([[5 * UM, 1 * UM]]))


def test_to_vtu_writes_the_field_on_triangles(mesh, tmp_path):
    path = tmp_path / "ez.vtu"
    _field(mesh).to_vtu(path)
    back = meshio.read(path)
    assert [block.type for block in back.cells] == ["triangle"]
    # order 2: each of the 24 triangles cut into 4
    assert len(back.cells[0].data) == 4 * 24
    x, y = back.points[:, 0], back.points[:, 1]
    np.testing.assert_allclose(back.points[:, 2], 0.0)
    np.testing.assert_allclose(
        back.point_data["Ez_real"], _quadratic(x, y), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        back.point_data["Ez_imag"], -2 * _quadratic(x, y), rtol=0, atol=1e-12
    )
    assert set(back.cell_data["region"][0]) == {1, 2}


def test_b_of_a_scalar_field_raises(mesh):
    with pytest.raises(ValueError, match="quantity is 'Ez'"):
        _field(mesh).b()
