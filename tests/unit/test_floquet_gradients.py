"""What a Floquet model refuses to differentiate, and how it says so.

Design gradients inside the cell go through Floquet S on the existing adjoint
path (tests/verification/test_gradient_matrix.py, `_driven_floquet_cells`).
The incidence and the port's half-space are constants of the solve: a traced
electromagnetic property of the region behind a Floquet surface raises
`gfem.NonDifferentiableError` while tracing, a moved node of a Floquet surface
raises `gfem.PortGeometryError` on the host in the backward pass, and the
offset is static. A traced angle of `gfem.Model.bloch_wave_vector` is pinned in
tests/unit/test_floquet_orders.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest

import gfem

P2 = gfem.SolverOptions(order=2, threads=2)
PROBLEM = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE"])


@pytest.fixture(scope="module")
def f1_model(tmp_path_factory):
    return pc.f1(pc.f1_mesh(tmp_path_factory.mktemp("f1")))


@pytest.mark.parametrize(
    ("region", "surface", "port"), [("air", "z_max", "R_TE"), ("sub", "z_min", "T_TE")]
)
@pytest.mark.parametrize("name", ["eps_r", "mu_r", "tan_delta", "conductivity"])
def test_a_traced_property_of_the_half_space_raises_while_tracing(
    f1_model, region, surface, port, name
):
    n = len(f1_model.mesh.elements(region))
    value = {"eps_r": 1.5, "mu_r": 1.2, "tan_delta": 0.01, "conductivity": 0.1}[name]

    def objective(x):
        field = gfem.MaterialField(**{name: jnp.full(n, x)})
        model = f1_model.with_material(region, field)
        return jnp.sum(jnp.abs(gfem.solve(model, PROBLEM, options=P2).s.values) ** 2)

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=(
            rf"FloquetPort '{port}' on surface '{surface}': region '{region}' is the "
            rf"half-space behind it, and its {name} fixes the diffraction orders "
            r"\(their wave numbers, impedances and the face's absorbing term\); it "
            r"is not differentiable\. Hold it fixed with jax\.lax\.stop_gradient, "
            r"or design a layer of another region inside the cell\."
        ),
    ):
        jax.make_jaxpr(jax.grad(objective))(value)


def test_a_concrete_field_on_the_half_space_is_still_refused_as_a_medium(f1_model):
    # no tracer: the model-time check names the field
    n = len(f1_model.mesh.elements("air"))
    with pytest.raises(NotImplementedError, match="has a MaterialField"):
        f1_model.with_material("air", gfem.MaterialField(eps_r=np.ones(n)))


def test_a_moved_node_of_a_floquet_surface_raises_on_the_host(f1_model):
    mesh = f1_model.mesh
    face = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region("z_max").tag])
    x = np.asarray(mesh.imported_nodes)
    inside = (x[face, :2] > 1e-9 * pc.UM).all(axis=1) & (
        x[face, :2] < (pc.F1_A - 1e-3) * pc.UM
    ).all(axis=1)
    node = int(face[inside][0])

    def objective(dx):
        nodes = jnp.asarray(mesh.nodes).at[node, 0].add(dx)
        moved = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(mesh), [nodes]
        )
        model = f1_model.replace(mesh=moved)
        return jnp.sum(jnp.abs(gfem.solve(model, PROBLEM, options=P2).s.values) ** 2)

    # raised on the host in the backward pass, inside JAX's runtime error
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"PortGeometryError: FloquetPort 'R_TE': 1 node\(s\) of its surface "
        r"'z_max' moved",
    ):
        jax.grad(objective)(1e-9)


def test_the_offset_of_a_floquet_port_is_static():
    def objective(offset):
        gfem.FloquetPort("R", "z_max", polarization="TE", offset=offset)
        return offset

    with pytest.raises(gfem.NonDifferentiableError, match=r"FloquetPort\.offset"):
        jax.grad(objective)(1e-7)
