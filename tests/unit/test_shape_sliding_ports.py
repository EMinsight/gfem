"""`boundary_morph(sliding_ports=...)`: wave port faces that slide in their plane.

A WR-90 guide sheared so that its x = A sidewall meets port2's plane at 26.6
degrees from the normal (a wall perpendicular to the port would not tell a
projected displacement from a plain normal one).
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem._port_section import port_section
from gfem.mesh import Mesh
from gfem.shape import boundary_morph

DATA = Path(__file__).parent / "data"
A = 22.86e-3
ALPHA = 0.5


def sheared_model(alpha=ALPHA, ports=None):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    topology = mesh._topology
    x = np.array(topology.imported_nodes)
    x[:, 0] = x[:, 0] + alpha * (x[:, 0] / A) * x[:, 2]
    x.flags.writeable = False
    sheared = Mesh(jnp.asarray(x), dataclasses.replace(topology, imported_nodes=x))
    return gfem.Model(
        sheared,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=ports or [gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def side(alpha=ALPHA):
    return lambda x: x[:, 0] > A * (1 + alpha * x[:, 2] / A) - 1e-9


def off_plane(mesh, nodes, surface):
    """The face's largest distance to its best-fit plane, over its diameter."""
    used = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region(surface).tag])
    xyz = nodes[used]
    centered = xyz - xyz.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][2]
    return np.abs(centered @ normal).max() / np.ptp(xyz, axis=0).max(), used


@pytest.fixture(scope="module")
def model():
    return sheared_model()


def test_a_held_port_face_keeps_its_nodes_and_parameters_as_before(model):
    morph = boundary_morph(model, "walls", where=side())
    assert morph.n_params == 6
    moved = np.asarray(morph(jnp.full(6, -1e-3)).nodes)
    for surface in ("port1", "port2"):
        assert off_plane(model.mesh, moved, surface)[0] < 1e-12


@pytest.mark.parametrize("t", [-1e-4, -1e-3])
def test_a_sliding_face_stays_planar_and_its_edge_nodes_stay_on_the_moved_wall(
    model, t
):
    morph = boundary_morph(model, "walls", where=side(), sliding_ports=("p2",))
    assert morph.n_params > 6  # the face's edge nodes move with the wall
    mesh = model.mesh
    moved = np.asarray(morph(jnp.full(morph.n_params, t)).nodes)
    flat, used = off_plane(mesh, moved, "port2")
    assert flat < 1e-12
    assert off_plane(mesh, moved, "port1")[0] < 1e-12
    edge = [n for n in np.asarray(morph._nodes) if n in used]
    assert edge
    wall = np.array([1.0, 0.0, -ALPHA]) / np.hypot(1.0, ALPHA)
    d = (moved[edge] - mesh.imported_nodes[edge]) @ wall
    np.testing.assert_allclose(np.abs(d), abs(t), rtol=1e-12)
    # the forward accepts the moved face
    port_section(model.replace(mesh=morph(jnp.full(morph.n_params, t))), model.ports[1])


def test_a_moving_node_on_a_sliding_face_moves_by_the_projected_normal(model):
    morph = boundary_morph(model, "walls", where=side(), sliding_ports=("p2",))
    mesh = model.mesh
    used = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region("port2").tag])
    on_face = np.isin(np.asarray(morph._nodes), used)
    wall = np.array([1.0, 0.0, -ALPHA]) / np.hypot(1.0, ALPHA)
    wall *= np.sign(morph.normals[~on_face][0] @ wall)  # into the domain
    normal = np.array([0.0, 0.0, 1.0])
    n_t = wall - (wall @ normal) * normal
    expected = n_t / (n_t @ n_t)
    np.testing.assert_allclose(
        morph.normals[on_face], np.tile(expected, (on_face.sum(), 1)), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(morph.normals[~on_face], axis=1), 1.0, atol=1e-12
    )


def test_a_wall_perpendicular_to_the_port_changes_nothing():
    flat = sheared_model(alpha=0.0)
    held = boundary_morph(flat, "walls", where=side(0.0))
    sliding = boundary_morph(flat, "walls", where=side(0.0), sliding_ports=("p2",))
    t = jnp.linspace(-1e-4, 1e-4, sliding.n_params)
    on_wall = np.isin(
        np.asarray(sliding._nodes),
        np.asarray(held._nodes),
    )
    np.testing.assert_allclose(sliding.normals[on_wall], held.normals, atol=1e-12)
    assert np.asarray(sliding(t).nodes).shape == flat.mesh.nodes.shape


def test_naming_one_port_of_a_shared_surface_slides_the_whole_surface():
    ports = [gfem.WavePort("a", "port2", mode=1), gfem.WavePort("b", "port2", mode=2)]
    shared = sheared_model(ports=[gfem.WavePort("p1", "port1"), *ports])
    morph = boundary_morph(shared, "walls", where=side(), sliding_ports=("a",))
    assert morph.n_params > 6


def test_unknown_lumped_and_own_names_raise_with_their_ports():
    lumped = sheared_model(
        ports=[
            gfem.LumpedPort("lp", "port1", direction="+y"),
            gfem.WavePort("p2", "port2"),
        ]
    )
    with pytest.raises(ValueError, match=r"'lp', a LumpedPort.*Its wave ports: 'p2'"):
        boundary_morph(lumped, "walls", where=side(), sliding_ports=("lp",))
    with pytest.raises(ValueError, match=r"'nope'.*not a port of the model"):
        boundary_morph(lumped, "walls", where=side(), sliding_ports=("nope",))


def test_the_moving_surface_cannot_be_a_sliding_ports_own_face(model):
    with pytest.raises(ValueError, match=r"face of the sliding wave port 'p2'"):
        boundary_morph(model, "port2", sliding_ports=("p2",))


def test_a_sliding_ports_face_cannot_be_held_by_fixed(model):
    with pytest.raises(ValueError, match=r"'port2'.*fixed"):
        boundary_morph(
            model, "walls", where=side(), fixed=("port2",), sliding_ports=("p2",)
        )


def test_a_wall_nearly_parallel_to_the_port_is_refused():
    steep = sheared_model(alpha=80.0)
    with pytest.raises(ValueError, match="within 1 degree"):
        boundary_morph(steep, "walls", where=side(80.0), sliding_ports=("p2",))


def test_sliding_ports_is_a_sequence_of_names(model):
    with pytest.raises(TypeError, match="sliding_ports"):
        boundary_morph(model, "walls", where=side(), sliding_ports="p2")
