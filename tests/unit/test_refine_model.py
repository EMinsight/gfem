"""`gfem._refine.refine_model` and `transfer`: designs follow a refined mesh."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._refine import refine_model, transfer
from gfem.errors import MeshQualityError, NonDifferentiableError

LAYERS = (("a", 0.5), ("b", 0.5), ("c", 0.5))
AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(3.0, 3.0, LAYERS, (3, 3, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _model(mesh, nodes=None):
    if nodes is not None:
        mesh = gfem.Mesh(nodes, mesh._topology)
    n = {r: len(mesh.elements(r)) for r in ("a", "b", "c", "a__b", "b__c")}
    a = gfem.MaterialField(
        eps_r=2.0 + jnp.arange(n["a"]) / n["a"], conductivity=jnp.asarray(0.5)
    )
    b = gfem.MaterialField(
        eps_r=3.0 + jnp.arange(3 * n["b"]).reshape(-1, 3) / n["b"],
        region="b",
        axes=AXES,
    )
    c = gfem.MaterialField(eps_r=5.0 + jnp.arange(n["c"]) / n["c"])
    return gfem.Model(
        mesh,
        materials={"a": a, "b": b, "c": c},
        boundaries=[
            gfem.Impedance(
                "a__b",
                resistance=10.0,
                fill=gfem.SheetField(fill=jnp.arange(n["a__b"]) / n["a__b"]),
            ),
            gfem.Conductor(
                "b__c",
                conductivity=1e7,
                fill=gfem.SheetField(
                    fill=jnp.arange(n["b__c"]) / n["b__c"], surface="b__c"
                ),
            ),
        ],
    )


def _mark_bottom_corner(mesh):
    """Mark the tetrahedra of layer a nearest the origin."""
    x = np.asarray(mesh.nodes)
    c = x[mesh.tetrahedra].mean(axis=1)
    marked = np.zeros(len(c), bool)
    in_a = mesh.tetrahedron_tags == mesh.region("a").tag
    near = np.linalg.norm(c[:, :2], axis=1) < 0.6e-6
    marked[in_a & near] = True
    assert marked.any()
    return marked


def _locate(x, cells, points, dim):
    """Index of the cell holding each point (brute force on barycentric coords)."""
    p = x[cells]
    if dim == 3:
        t = np.transpose(p[:, 1:] - p[:, :1], (0, 2, 1))
        lam = np.linalg.solve(t[None], (points[:, None] - p[None, :, 0])[..., None])
        lam = lam[..., 0]
    else:  # a z = const triangle: barycentric in the xy plane
        t = np.transpose(p[:, 1:, :2] - p[:, :1, :2], (0, 2, 1))
        rhs = points[:, None, :2] - p[None, :, 0, :2]
        lam = np.linalg.solve(t[None], rhs[..., None])[..., 0]
    lam = np.concatenate([1 - lam.sum(axis=-1, keepdims=True), lam], axis=-1)
    inside = (lam > -1e-9).all(axis=-1)
    assert (inside.sum(axis=1) == 1).all()
    return inside.argmax(axis=1)


def _by_region(model):
    out = {r: np.asarray(model.materials[r].eps_r) for r in ("a", "b", "c")}
    for boundary in model.boundaries:
        out[boundary.surface] = np.asarray(boundary.fill.fill)
    return out


def test_the_refined_model_keeps_its_design_at_every_child_centroid(mesh):
    coarse = _model(mesh)
    fine, _ = refine_model(coarse, _mark_bottom_corner(mesh))
    assert len(fine.mesh.tetrahedra) > len(mesh.tetrahedra)
    old, new = _by_region(coarse), _by_region(fine)
    x0, x1 = np.asarray(mesh.nodes), np.asarray(fine.mesh.nodes)
    for region, values in old.items():
        dim = fine.mesh.region(region).dim
        cells0 = (mesh.tetrahedra if dim == 3 else mesh.triangles)[
            mesh.elements(region)
        ]
        cells1 = (fine.mesh.tetrahedra if dim == 3 else fine.mesh.triangles)[
            fine.mesh.elements(region)
        ]
        where = _locate(x0, cells0, x1[cells1].mean(axis=1), dim)
        np.testing.assert_array_equal(new[region], values[where])
    a = fine.materials["a"]
    assert float(a.conductivity) == 0.5
    assert fine.materials["b"].axes == AXES
    assert fine.materials["b"].region == "b"
    assert fine.boundaries[1].fill.surface == "b__c"
    assert fine.boundaries[1].conductivity == 1e7


def test_an_untouched_region_keeps_its_values_in_order(mesh):
    coarse = _model(mesh)
    fine, _ = refine_model(coarse, _mark_bottom_corner(mesh))
    assert len(fine.mesh.elements("c")) == len(mesh.elements("c"))
    np.testing.assert_array_equal(
        fine.materials["c"].eps_r, coarse.materials["c"].eps_r
    )


def test_the_refined_mesh_has_no_source(mesh):
    fine, _ = refine_model(_model(mesh), _mark_bottom_corner(mesh))
    assert fine.mesh.source is None
    assert fine.mesh.length_unit == "um"


def test_a_moved_mesh_is_refined_at_its_current_coordinates(mesh):
    nodes = mesh.nodes.at[:, 2].multiply(1.1)  # stretch along z: no inversion
    fine, _ = refine_model(_model(mesh, nodes), _mark_bottom_corner(mesh))
    n = len(mesh.imported_nodes)
    np.testing.assert_array_equal(fine.mesh.imported_nodes[:n], np.asarray(nodes))
    np.testing.assert_array_equal(np.asarray(fine.mesh.nodes), fine.mesh.imported_nodes)


def test_a_moved_mesh_with_an_inverted_element_raises(mesh):
    tet = mesh.tetrahedra[0]
    x = np.asarray(mesh.nodes)
    a, b, c = x[tet[1]], x[tet[2]], x[tet[3]]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    mirrored = x[tet[0]] - 2 * np.dot(x[tet[0]] - a, normal) * normal
    nodes = mesh.nodes.at[tet[0]].set(mirrored)
    with pytest.raises(MeshQualityError, match="inverted"):
        refine_model(_model(mesh, nodes), _mark_bottom_corner(mesh))


def _traced_nodes(model, t):
    return model.replace(mesh=gfem.Mesh(model.mesh.nodes * t, model.mesh._topology))


def _traced_material(model, t):
    field = gfem.MaterialField(eps_r=model.materials["c"].eps_r * t)
    return model.with_material("c", field)


def _traced_fill(model, t):
    sheet = model.boundaries[0]
    traced = gfem.Impedance(
        "a__b", resistance=10.0, fill=gfem.SheetField(fill=sheet.fill.fill * t)
    )
    return model.replace(boundaries=(traced, model.boundaries[1]))


@pytest.mark.parametrize(
    ("trace", "leaf"),
    [
        (_traced_nodes, "mesh nodes"),
        (_traced_material, r"materials\['c'\]"),
        (_traced_fill, "a__b"),
    ],
)
def test_a_traced_leaf_raises(mesh, trace, leaf):
    model = _model(mesh)
    marked = _mark_bottom_corner(mesh)

    def objective(t):
        refine_model(trace(model, t), marked)
        return t

    with pytest.raises(NonDifferentiableError, match=leaf):
        jax.grad(objective)(1.0)


@pytest.mark.parametrize("region", ["a", "a__b"])
def test_transfer_gathers_region_local_values_of_either_shape(mesh, region):
    coarse = _model(mesh)
    fine, refinement = refine_model(coarse, _mark_bottom_corner(mesh))
    n = len(mesh.elements(region))
    flat = np.arange(n, dtype=float)
    tensor = np.stack([flat, -flat, 2 * flat], axis=1)
    out = transfer(flat, region, coarse=mesh, fine=fine.mesh, refinement=refinement)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(fine.mesh.elements(region)),)
    out3 = transfer(
        jnp.asarray(tensor), region, coarse=mesh, fine=fine.mesh, refinement=refinement
    )
    assert isinstance(out3, jax.Array)
    np.testing.assert_array_equal(out3, tensor[out.astype(int)])
    by_model = _by_region(fine)[region]
    np.testing.assert_array_equal(
        by_model,
        transfer(
            _by_region(coarse)[region],
            region,
            coarse=mesh,
            fine=fine.mesh,
            refinement=refinement,
        ),
    )


def test_transfer_is_differentiable(mesh):
    refined, refinement = refine_model(_model(mesh), _mark_bottom_corner(mesh))
    fine = refined.mesh
    n = len(mesh.elements("a"))

    def total(v):
        return transfer(v, "a", coarse=mesh, fine=fine, refinement=refinement).sum()

    g = jax.grad(total)(jnp.ones(n))
    # each coarse element counts once per child; children only multiply, so
    # the counts sum to the refined region's size
    assert float(g.sum()) == len(fine.elements("a"))
    assert (g >= 1).all()


def test_transfer_refuses_values_of_the_wrong_length(mesh):
    fine, refinement = refine_model(_model(mesh), _mark_bottom_corner(mesh))
    with pytest.raises(ValueError, match=r"mesh\.elements\('a'\)"):
        transfer(np.zeros(3), "a", coarse=mesh, fine=fine.mesh, refinement=refinement)
