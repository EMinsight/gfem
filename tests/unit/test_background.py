"""The background stack: `gfem.Background` and ``Model(background=...)``."""

import math

import jax
import pytest
import structured_meshes as sm

import gfem

AIR = gfem.Material()
SILICA = gfem.Material(eps_r=2.1)
SILICON = gfem.Material(eps_r=12.1)


def test_one_material_is_a_one_layer_background():
    background = gfem.Background(AIR)
    assert background.layers == (AIR,)
    assert background.interfaces == ()
    assert background.ground is None


def test_layers_run_bottom_to_top_between_their_interfaces():
    stack = gfem.Background([SILICON, SILICA, AIR], interfaces=[-2e-6, 0.0])
    assert stack.layers == (SILICON, SILICA, AIR)
    assert stack.interfaces == (-2e-6, 0.0)
    assert "interfaces=(-2e-06, 0.0)" in repr(stack)


def test_a_stack_may_end_on_a_ground_plane_below_its_layers():
    stack = gfem.Background([SILICA, AIR], interfaces=[1e-6], ground=0.0)
    assert stack.ground == 0.0
    with pytest.raises(ValueError, match=r"ground .* below the lowest interface"):
        gfem.Background([SILICA, AIR], interfaces=[1e-6], ground=2e-6)


@pytest.mark.parametrize(
    ("layers", "interfaces", "match"),
    [
        ([SILICA, AIR], (), "2 layers need 1 interface"),
        ([SILICA], (0.0,), "1 layer needs 0 interfaces"),
        ([SILICON, SILICA, AIR], (1.0, 0.0), "interfaces must ascend"),
        ([SILICON, SILICA, AIR], (0.0, 0.0), "interfaces must ascend"),
        ([SILICA, AIR], (math.nan,), "interfaces must be finite"),
        ([], (), "at least one layer"),
    ],
)
def test_the_planes_must_match_the_layers_and_ascend(layers, interfaces, match):
    with pytest.raises(ValueError, match=match):
        gfem.Background(layers, interfaces=interfaces)


@pytest.mark.parametrize(
    ("layer", "match"),
    [
        (gfem.Material(eps_r=(2.0, 2.0, 3.0)), "anisotropic"),
        (
            gfem.Material(
                eps_r=gfem.PermittivityLaw(
                    eps_inf=2.0,
                    terms=[gfem.Debye(delta_eps=1.0, relaxation_frequency=1e9)],
                )
            ),
            "permittivity law",
        ),
    ],
)
def test_a_layer_must_be_constant_and_isotropic(layer, match):
    with pytest.raises(NotImplementedError, match=match):
        gfem.Background([layer, AIR], interfaces=[0.0])


def test_a_material_field_layer_is_refused():
    field = gfem.MaterialField.uniform("a", SILICA)
    with pytest.raises(NotImplementedError, match="MaterialField"):
        gfem.Background(field)


def test_a_layer_must_be_a_material():
    with pytest.raises(TypeError, match=r"Background layers are gfem.Material"):
        gfem.Background("air")


def test_a_traced_interface_is_not_differentiable():
    def build(z):
        gfem.Background([SILICA, AIR], interfaces=[z])
        return z

    with pytest.raises(gfem.NonDifferentiableError, match=r"Background.interfaces"):
        jax.grad(build)(0.0)


# --- on the model -----------------------------------------------------------------


@pytest.fixture
def mesh(tmp_path):
    path = sm.write_msh22(
        tmp_path / "plate.msh", sm.plate(1.0, 1.0, (("air", 1.0),), (1, 1, 1))
    )
    return gfem.Mesh.from_gmsh(path)


def test_a_model_has_no_background_by_default(mesh):
    assert gfem.Model(mesh, materials={"air": AIR}).background is None


def test_a_material_background_is_a_one_layer_stack(mesh):
    model = gfem.Model(mesh, materials={"air": AIR}, background=AIR)
    assert model.background == gfem.Background(AIR)
    assert "background: Background(" in repr(model)


def test_a_model_keeps_its_background_through_replace_and_the_pytree(mesh):
    stack = gfem.Background([SILICA, AIR], interfaces=[0.5])
    model = gfem.Model(mesh, materials={"air": AIR}, background=stack)
    assert model.with_material("air", SILICA).background is stack
    assert model.replace(background=None).background is None
    leaves, tree = jax.tree.flatten(model)
    assert jax.tree.unflatten(tree, leaves).background is stack


def test_a_model_background_must_be_a_stack_or_a_material(mesh):
    with pytest.raises(TypeError, match=r"Background layers are gfem.Material"):
        gfem.Model(mesh, materials={"air": AIR}, background="air")
