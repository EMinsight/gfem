"""`gfem.region_mean`: the mean of a field over named volume regions.

A parallel-plate capacitor in two equal layers ("low" under "high", plates
at z = 0 and z = 1 um, the top plate at 1 V) has a potential that is
linear in each layer, so every order holds it exactly: the interface sits
at V_i = eps_high / (eps_low + eps_high), the low layer's mean is V_i / 2
and the high layer's (1 + V_i) / 2, on any mesh. A driven line's mean
field is checked against an exact quadrature of `Field.at` over the
region's tetrahedra, an independent engine path.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._refine import refine_model

LAYERS = (("low", 0.5), ("high", 0.5))
EPS = {"low": 2.0, "high": 5.0}
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
PROBLEM = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


@pytest.fixture(scope="module")
def plate_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.strip_line(2.0, 1.0, 1.0, (2, 1, 1), layers=LAYERS))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _capacitor(mesh, eps_high=None):
    high = gfem.Material(eps_r=EPS["high"])
    if eps_high is not None:
        uniform = gfem.MaterialField.uniform("high", high)
        high = dataclasses.replace(uniform, eps_r=eps_high)
    return gfem.Model(
        mesh, materials={"low": gfem.Material(eps_r=EPS["low"]), "high": high}
    )


def _interface(eps_low, eps_high):
    return eps_high / (eps_low + eps_high)


@pytest.fixture(scope="module")
def potential(plate_mesh):
    result = gfem.solve(_capacitor(plate_mesh), PROBLEM, options=OPTIONS)
    return result.potential["top"]


def test_region_mean_of_a_potential_is_its_exact_layer_mean(potential):
    v = _interface(EPS["low"], EPS["high"])
    low = gfem.region_mean(potential, "low")
    assert low.shape == ()
    assert float(low) == pytest.approx(v / 2, abs=1e-10)
    assert float(gfem.region_mean(potential, "high")) == pytest.approx(
        (1 + v) / 2, abs=1e-10
    )


def test_region_mean_over_several_regions_is_the_union_mean(potential):
    both = gfem.region_mean(potential, ["low", "high"])
    v = _interface(EPS["low"], EPS["high"])
    assert float(both) == pytest.approx((v / 2 + (1 + v) / 2) / 2, abs=1e-10)


def test_region_mean_is_the_same_quantity_on_a_refined_mesh(plate_mesh):
    model = _capacitor(plate_mesh)
    rng = np.random.default_rng(0)
    marked = rng.random(len(plate_mesh.tetrahedra)) < 0.5
    fine, _ = refine_model(model, marked)
    assert len(fine.mesh.tetrahedra) > len(plate_mesh.tetrahedra)
    coarse = gfem.solve(model, PROBLEM, options=OPTIONS).potential["top"]
    refined = gfem.solve(fine, PROBLEM, options=OPTIONS).potential["top"]
    for region in ("low", "high"):
        assert float(gfem.region_mean(refined, region)) == pytest.approx(
            float(gfem.region_mean(coarse, region)), abs=1e-10
        )


def test_region_mean_gradient_matches_the_closed_form_and_differences(plate_mesh):
    def mean_low(eps_high):
        model = _capacitor(plate_mesh, jnp.asarray(eps_high))
        result = gfem.solve(model, PROBLEM, options=OPTIONS)
        return gfem.region_mean(result.potential["top"], "low")

    e = EPS["high"]
    grad = float(jax.grad(mean_low)(e))
    exact = EPS["low"] / (2 * (EPS["low"] + e) ** 2)
    h = 1e-4
    differences = (float(mean_low(e + h)) - float(mean_low(e - h))) / (2 * h)
    assert grad == pytest.approx(exact, rel=1e-7)
    assert grad == pytest.approx(differences, rel=1e-6)


def test_region_mean_works_under_jit_and_vmap(potential):
    jitted = jax.jit(lambda f: gfem.region_mean(f, "low"))(potential)
    assert float(jitted) == pytest.approx(float(gfem.region_mean(potential, "low")))

    c = potential.coefficients

    def mean(scale):
        field = dataclasses.replace(potential, coefficients=c * scale)
        return gfem.region_mean(field, "high")

    scales = jnp.array([1.0, 2.0, -0.5])
    batched = jax.vmap(mean)(scales)
    one = float(gfem.region_mean(potential, "high"))
    np.testing.assert_allclose(np.asarray(batched), one * np.asarray(scales))


# --- a driven field -----------------------------------------------------------------

#: Degree-2 tetrahedron rule (four points, equal weights), exact on the
#: order-2 fields of the driven line.
_A, _B = 0.5854101966249685, 0.1381966011250105
_RULE = np.array(
    [[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]]
)


def _quadrature_mean(field, mesh, regions):
    tets = np.concatenate([mesh.elements(r) for r in regions])
    corners = np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[tets]]
    edges = corners[:, 1:] - corners[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / 6
    points = np.einsum("qc,tcd->tqd", _RULE, corners).reshape(-1, 3)
    values = np.asarray(field.at(points)).reshape(len(tets), 4, -1).mean(axis=1)
    return (volumes[:, None] * values).sum(axis=0) / volumes.sum()


@pytest.fixture(scope="module")
def driven(plate_mesh):
    model = gfem.Model(
        plate_mesh,
        materials={"low": gfem.Material(eps_r=2.0), "high": gfem.Material(eps_r=5.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    problem = gfem.Driven(frequency=20e12, excite="p1", keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS).e_field(20e12)


@pytest.mark.parametrize("regions", ["low", ("low", "high")])
def test_region_mean_of_a_driven_field_matches_a_quadrature(driven, regions):
    model, field = driven
    names = (regions,) if isinstance(regions, str) else regions
    mean = gfem.region_mean(field, regions)
    assert mean.shape == (3,)
    assert jnp.iscomplexobj(mean)
    expected = _quadrature_mean(field, model.mesh, names)
    np.testing.assert_allclose(
        np.asarray(mean), expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max()
    )


def test_region_mean_of_a_driven_field_has_its_transpose_as_gradient(driven):
    _, field = driven
    w = jnp.array([0.3, -1.0, 2.0])

    def weighted(c):
        f = dataclasses.replace(field, coefficients=c)
        return jnp.real(jnp.dot(w, gfem.region_mean(f, "high")))

    c = field.coefficients
    grad = jax.grad(weighted)(c)
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))
    h = 1e-3
    differences = (weighted(c + h * v) - weighted(c - h * v)) / (2 * h)
    # gfem's contract: the cotangent of a complex input is dJ/dx - i dJ/dy
    assert float(jnp.real(jnp.vdot(jnp.conj(grad), v))) == pytest.approx(
        float(differences), rel=1e-8
    )


# --- errors -------------------------------------------------------------------------


def test_region_mean_refuses_what_is_not_a_field():
    with pytest.raises(TypeError, match=r"gfem\.Field"):
        gfem.region_mean(np.zeros(3), "low")


def test_region_mean_refuses_an_unknown_region(potential):
    with pytest.raises(gfem.RegionNotFoundError):
        gfem.region_mean(potential, "nowhere")


def test_region_mean_refuses_a_surface_region(potential):
    with pytest.raises(ValueError, match="surface"):
        gfem.region_mean(potential, "top")


def test_region_mean_refuses_a_region_outside_the_support(potential):
    field = dataclasses.replace(potential, support=("high",))
    with pytest.raises(ValueError, match="support"):
        gfem.region_mean(field, "low")


def test_region_mean_refuses_a_region_name_list_that_is_empty(potential):
    with pytest.raises(TypeError, match="region"):
        gfem.region_mean(potential, [])
