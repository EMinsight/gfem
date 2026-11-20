r"""Gradients next to a perfectly matched layer, against finite differences.

The layer's stretch is a coefficient of position, fixed at the imported
mesh (its interior box and thicknesses). A node move then changes a form
on the layer by the transport of its coefficient and by
:math:`\int (W \cdot \nabla) C\,(\dots)`, the coefficient read at the
moved quadrature points; without that term the rows of the nodes of layer
tetrahedra are off by 10-30 %. The cases end a TEM line (PEC y walls, PMC
x walls) with a PEC-backed layer along z, with a dielectric slab in the
air, at order 2, and check each gradient to :math:`10^{-6}` relative:

- :math:`|S_{11}|^2` by the slab's permittivity;
- :math:`|S_{11}|^2` by single node coordinates: an air node, a node of the
  layer's inner face and a node inside the layer;
- :math:`|S_{11}|^2` by a `boundary_morph` of the slab's face, which holds
  the layer;
- a field value read inside the layer (`Field.at`) by the slab's
  permittivity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem.shape import boundary_morph

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
PROBLEM = gfem.Driven(pc.F0)
#: The slab's permittivity.
EPS_SLAB = 2.25
#: Node steps, m (edges are 0.1 um).
NODE_STEP = 1e-9


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    mesh = gfem.Mesh.from_gmsh(
        pc.slab_line(tmp_path_factory.mktemp("pml") / "line.msh")
    )
    air = gfem.Material()
    n = len(mesh.elements("slab"))
    return gfem.Model(
        mesh,
        materials={
            "air": air,
            "pml": air,
            "slab": gfem.MaterialField(eps_r=np.full(n, EPS_SLAB)),
        },
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


def _s11(model):
    s = gfem.solve(model, PROBLEM, options=OPTIONS).s.values
    return jnp.abs(s[0, 0, 0]) ** 2


def _with_slab(model, eps):
    return model.with_material("slab", gfem.MaterialField(eps_r=eps))


def test_s11_by_the_slab_permittivity(line):
    eps = jnp.asarray(line.materials["slab"].eps_r)
    directions = unit_directions(np.random.default_rng(0), eps.shape, 2)
    assert_gradient_matches(
        lambda e: _s11(_with_slab(line, e)), eps, directions, step=1e-2
    )


def _node(mesh, z):
    """The node at the section's centre and height ``z`` (um)."""
    x = np.asarray(mesh.imported_nodes) * 1e6
    a, b = pc.LINE_SECTION
    return int(np.argmin(np.linalg.norm(x - [a / 2, b / 2, z], axis=1)))


@pytest.mark.parametrize(
    ("z", "axes", "inward"),
    [
        (1.2, (0, 1, 2), 0.0),  # an air node one element from the inner face
        # the inner face: a node moved past it would put air in the stretch,
        # so the differences sit inside it, far enough (0.3 of an element)
        # that no quadrature point crosses the face, where the profile's
        # second derivative jumps (measured: 9e-8 relative there, 6e-4 at
        # 0.1 of an element)
        (pc.AIR_LENGTH, (2,), 3e-8),
        (1.8, (0, 1, 2), 0.0),  # a node inside the layer
    ],
    ids=["air node", "inner-face node", "layer node"],
)
def test_s11_by_one_node(line, z, axes, inward):
    mesh = line.mesh
    node = _node(mesh, z)
    structure = jax.tree_util.tree_structure(mesh)
    x0 = jnp.asarray(mesh.nodes).at[node, 2].add(-inward)

    def objective(x):
        moved = jax.tree_util.tree_unflatten(structure, [x])
        return _s11(line.replace(mesh=moved))

    directions = []
    for axis in axes:
        d = np.zeros(x0.shape)
        d[node, axis] = 1.0
        directions.append(d)
    assert_gradient_matches(objective, x0, directions, step=NODE_STEP)


def test_s11_by_a_boundary_morph_of_the_slab(line):
    morph = boundary_morph(line, "front", grows="slab")
    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(1), t.shape, 2)
    assert_gradient_matches(
        lambda t: _s11(line.replace(mesh=morph(t))), t, directions, step=NODE_STEP
    )


def test_a_field_value_inside_the_layer_by_the_slab_permittivity(line):
    a, b = pc.LINE_SECTION
    point = np.array([a / 2, b / 2, 1.73]) * 1e-6
    problem = gfem.Driven(pc.F0, keep_fields=True)

    def objective(e):
        result = gfem.solve(_with_slab(line, e), problem, options=OPTIONS)
        value = result.e_field(pc.F0).at(point)
        return jnp.sum(jnp.abs(value) ** 2)

    eps = jnp.asarray(line.materials["slab"].eps_r)
    directions = unit_directions(np.random.default_rng(2), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)
