"""Exact gradients of a piezoelectric solve with respect to the node coordinates.

A user's JAX node map moves the nodes of a transducer section (`Mesh.with_nodes`)
and the gradient of the admittance, the clamped capacitance and readings at
fixed points follows through it. Each map is checked against fourth-order central
differences at the gate of `finite_differences`:

- ``pitch``: the interdigitated area stretched along x, the damping frames
  translated rigidly;
- ``stretch``: every node scaled along x, so the frames stretch with the device;
- ``bump``: a smooth displacement in x and y at a finger edge and in a frame
  corner;
- ``thickness`` (meshed fingers): the metal layer scaled along y.
"""

import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import piezo_cases as pc
import pytest

import gfem

jax.config.update("jax_enable_x64", True)

UM = 1e-6
#: The interdigitated area (with its margin) along x, metres.
X_LEFT = (-pc.PAIRS * pc.PERIOD / 2 - pc.MARGIN) * UM
X_RIGHT = -X_LEFT


def _pitch(nodes, p):
    """The area between the frames stretched by ``1 + p``; the frames shifted."""
    s = 1.0 + p[0]
    x = nodes[:, 0]
    inside = jnp.clip(x, X_LEFT, X_RIGHT)
    shifted = x - inside + s * inside
    return nodes.at[:, 0].set(shifted)


def _stretch(nodes, p):
    return nodes.at[:, 0].set((1.0 + p[0]) * nodes[:, 0])


def _bump(nodes, p):
    """Two Gaussian bumps in x and y: a finger's left edge and a frame corner."""
    out = nodes
    for (cx, cy), width, (ax, ay) in (
        ((-4.0 * UM + 0.5 * UM, 0.0), 0.8 * UM, (p[0], p[1])),
        ((X_LEFT - 4.0 * UM, -8.0 * UM), 2.0 * UM, (p[2], p[3])),
    ):
        r2 = ((nodes[:, 0] - cx) ** 2 + (nodes[:, 1] - cy) ** 2) / width**2
        g = jnp.exp(-r2)
        out = out.at[:, 0].add(UM * ax * g).at[:, 1].add(UM * ay * g)
    return out


def _thickness(nodes, p):
    """The layer between the substrate and the air above the fingers rescaled."""
    t = pc.T_AL * UM
    y = nodes[:, 1]
    above = jnp.clip(y, 0.0, t)
    return nodes.at[:, 1].set(y + p[0] * above)


MAPS = {
    "pitch": (_pitch, (1,), 3e-4),
    "stretch": (_stretch, (1,), 3e-4),
    "bump": (_bump, (4,), 1e-2),
    "thickness": (_thickness, (1,), 1e-2),
}


@pytest.fixture(scope="module", params=("lines", "meshed"))
def device(request, tmp_path_factory):
    variant = request.param
    path = tmp_path_factory.mktemp(variant) / "idt.msh"
    mesh = pc.mesh(path, variant)
    return variant, mesh, pc.model(mesh, variant)


def _admittance(model, mesh, node_map, p):
    moved = mesh.with_nodes(node_map(jnp.asarray(mesh.imported_nodes), p))
    result = gfem.solve(model.replace(mesh=moved), pc.problem())
    y = result.admittance_per_length[0, 0, 0]
    return jnp.stack([y.real, y.imag, result.clamped_capacitance_per_length[0, 0]])


@pytest.mark.parametrize("name", list(MAPS))
def test_the_admittance_gradient_through_a_node_map_matches_finite_differences(
    device, name
):
    variant, mesh, model = device
    if name == "thickness" and variant == "lines":
        pytest.skip("the thickness map moves the meshed fingers only")
    node_map, shape, step = MAPS[name]
    rng = np.random.default_rng(5)
    p0 = jnp.zeros(shape)
    directions = fd.unit_directions(rng, shape, 2)
    for k in range(3):  # Re Y, Im Y, C0

        def objective(p, k=k):
            return _admittance(model, mesh, node_map, p)[k]

        fd.assert_gradient_matches(objective, p0, directions, step=step)


# --- readings at fixed points ---------------------------------------------------------


def _inside_point(mesh, target):
    """The centroid of the lithium niobate triangle nearest ``target`` (metres).

    A centroid keeps the point away from the element's faces, so a small node
    step does not move it into a neighbour.
    """
    nodes = np.asarray(mesh.imported_nodes)
    cells = np.asarray(mesh.volume_elements)[mesh.elements("ln")]
    centres = nodes[cells].mean(axis=1)
    return centres[np.argmin(np.linalg.norm(centres - np.asarray(target), axis=1))]


def _readings(model, mesh, node_map, p, point):
    """``[Re u_z, Im u_z, Re S_yz, Im S_yz, Re S_xz]`` at ``point``."""
    moved = mesh.with_nodes(node_map(jnp.asarray(mesh.imported_nodes), p))
    result = gfem.solve(model.replace(mesh=moved), pc.problem(keep_fields=True))
    displacement = result.displacement_fields[0]
    uz = displacement["z"].at(point)
    strain = gfem.strain(displacement, point)
    return jnp.stack([uz.real, uz.imag, strain[3].real, strain[3].imag, strain[4].real])


@pytest.mark.parametrize("name", ["pitch", "bump"])
def test_a_point_reading_follows_the_nodes_at_a_fixed_point(device, name):
    _variant, mesh, model = device
    node_map, shape, step = MAPS[name]
    point = _inside_point(mesh, (0.3 * UM, -0.9 * UM))
    rng = np.random.default_rng(9)
    p0 = jnp.zeros(shape)
    directions = fd.unit_directions(rng, shape, 2)
    for k in range(5):

        def objective(p, k=k):
            return _readings(model, mesh, node_map, p, point)[k]

        fd.assert_gradient_matches(objective, p0, directions, step=step)


# --- the cotangent of the nodes -----------------------------------------------------


def test_the_node_cotangent_of_the_admittance_sums_to_zero_under_translation(device):
    _variant, mesh, model = device

    def objective(nodes):
        moved = mesh.with_nodes(nodes)
        result = gfem.solve(model.replace(mesh=moved), pc.problem())
        y = result.admittance_per_length[0, 0, 0]
        return y.imag + 1e3 * result.clamped_capacitance_per_length[0, 0]

    grad = np.asarray(jax.grad(objective)(jnp.asarray(mesh.imported_nodes)))
    assert grad.shape == mesh.imported_nodes.shape
    assert np.abs(grad).max() > 0
    assert np.abs(grad.sum(axis=0)).max() < 1e-12 * np.abs(grad).sum()
