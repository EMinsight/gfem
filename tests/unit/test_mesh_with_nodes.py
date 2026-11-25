"""`gfem.Mesh.with_nodes`: a moved mesh from node coordinates.

The strip is three unit squares of right isosceles triangles (tests/strip2d.py);
the plate a one-layer parallel-plate capacitor of tetrahedra.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm
from strip2d import write_strip

import gfem

EPS0 = scipy.constants.epsilon_0
UM = 1e-6
PROBLEM = gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom")


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    return gfem.Mesh.from_gmsh(write_strip(tmp_path_factory.mktemp("strip") / "s.msh"))


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, (("oxide", 0.5),), (2, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"oxide": gfem.Material(eps_r=3.9)}
    )


@pytest.fixture(params=["strip", "plate"])
def mesh(request, strip, plate):
    return strip if request.param == "strip" else plate.mesh


def _mirrored(mesh):
    """The nodes reflected through x = 0: every element turns inside out."""
    flip = np.ones(mesh.dim)
    flip[0] = -1.0
    return np.asarray(mesh.nodes) * flip


def test_the_moved_mesh_keeps_topology_regions_and_imported_nodes(mesh):
    nodes = np.asarray(mesh.nodes) * 1.01
    moved = mesh.with_nodes(nodes)
    np.testing.assert_array_equal(moved.nodes, nodes)
    assert moved.nodes.dtype == np.float64
    assert moved.dim == mesh.dim
    assert moved.regions == mesh.regions
    assert moved.imported_nodes is mesh.imported_nodes
    assert moved.volume_elements is mesh.volume_elements
    assert moved.surface_elements is mesh.surface_elements
    assert moved.length_unit == mesh.length_unit


def test_a_wrong_shape_raises_naming_both_shapes(mesh):
    n = len(mesh.imported_nodes)
    wrong = np.zeros((n, 5 - mesh.dim))
    match = rf"\({n}, {5 - mesh.dim}\).*\({n}, {mesh.dim}\)"
    with pytest.raises(ValueError, match=match):
        mesh.with_nodes(wrong)
    with pytest.raises(ValueError, match=match):
        jax.jit(lambda x: mesh.with_nodes(x).nodes)(wrong)


def test_non_finite_coordinates_raise(mesh):
    nodes = np.asarray(mesh.nodes).copy()
    nodes[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mesh.with_nodes(nodes)


def test_complex_coordinates_raise(mesh):
    with pytest.raises(TypeError, match="real"):
        mesh.with_nodes(np.asarray(mesh.nodes) + 0j)


def test_a_mirrored_map_raises_on_its_inverted_elements(mesh):
    with pytest.raises(gfem.MeshQualityError, match="inverted"):
        mesh.with_nodes(_mirrored(mesh))


def test_an_element_below_a_quarter_of_its_quality_warns(strip):
    # squashed along y by 10, each right isosceles triangle's smallest angle
    # falls from 45 deg to atan(0.1), its quality to 0.14 x the imported one
    squashed = np.asarray(strip.nodes) * np.array([1.0, 0.1])
    with pytest.warns(gfem.MeshQualityWarning, match="region 'a'") as caught:
        strip.with_nodes(squashed)
    assert caught[0].filename == __file__
    # by 2 it falls to 0.63 x: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strip.with_nodes(np.asarray(strip.nodes) * np.array([1.0, 0.5]))


def test_traced_nodes_stay_traced_and_unchecked(mesh):
    # an inverted traced mesh is the solve's forward's to refuse
    mirrored = jax.jit(lambda x: mesh.with_nodes(x).nodes)(_mirrored(mesh))
    np.testing.assert_array_equal(mirrored, _mirrored(mesh))

    def total(s):
        return mesh.with_nodes(s * mesh.nodes).nodes.sum()

    g = jax.jit(jax.grad(total))(1.0)
    assert float(g) == pytest.approx(float(np.sum(mesh.imported_nodes)), rel=1e-15)


def _capacitance(plate, build):
    """C of the plate stretched along z by ``s``, its mesh built by ``build``.

    One thread: a threaded solve sums in an order that changes from call to
    call, so two solves of the same model can differ in the last bit, and the
    comparison below is bitwise.
    """
    options = gfem.SolverOptions(threads=1)

    def c(s):
        nodes = plate.mesh.nodes * jnp.array([1.0, 1.0, s])
        model = plate.replace(mesh=build(nodes))
        return gfem.solve(model, PROBLEM, options=options).capacitance[0, 0]

    return c


def test_a_node_map_gradient_through_a_solve_is_the_private_rebuilds(plate):
    mesh = plate.mesh
    public = jax.jit(jax.grad(_capacitance(plate, mesh.with_nodes)))(1.0)
    private = jax.jit(
        jax.grad(_capacitance(plate, lambda x: gfem.Mesh(x, mesh._topology)))
    )(1.0)
    # the same arithmetic on both paths, so the same bits: a tolerance would let
    # the public path add a step (a scale and its inverse) unseen
    assert float(public) == float(private)
    # C = eps0 eps_r A / (s d): dC / ds = -C at s = 1
    c = EPS0 * 3.9 * (4 * UM) * (3 * UM) / (0.5 * UM)
    assert float(public) == pytest.approx(-c, rel=1e-10)
