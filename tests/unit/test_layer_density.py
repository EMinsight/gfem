"""A layer-grid design on a 2D mesh: its spelling, storage, coverage and codec."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem import design

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
MM = 1e-3


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


@pytest.fixture(scope="module")
def grid(mesh):
    return design.LayerGrid(mesh, "design", pitch=1 * MM)


def pixels(grid, seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(0.1, 0.9, grid.shape))


def test_to_elements_on_a_2d_mesh_holds_the_pixels(grid):
    rho = pixels(grid)
    density = grid.to_elements(rho)
    assert isinstance(density, design.LayerDensity)
    assert density.grid is grid
    np.testing.assert_array_equal(density.pixels, rho)


def test_a_layer_density_is_no_array(grid):
    density = grid.to_elements(pixels(grid))
    for convert in (np.asarray, jnp.asarray):
        with pytest.raises(TypeError, match=r"quadrature points.*density\.pixels"):
            convert(density)
    with pytest.raises(TypeError):
        density * 2


def test_to_elements_checks_the_pixel_shape(grid):
    with pytest.raises(ValueError, match="shape"):
        grid.to_elements(jnp.zeros((2, 3)))


def test_jit_and_grad_go_through_to_elements(grid):
    rho = pixels(grid)
    density = jax.jit(grid.to_elements)(rho)
    assert isinstance(density, design.LayerDensity)
    np.testing.assert_array_equal(density.pixels, rho)

    def total(r):
        return jnp.sum(grid.to_elements(r).pixels ** 2)

    np.testing.assert_allclose(jax.grad(total)(rho), 2 * rho)


#: A Drude metal at 5 GHz: Re eps about -15.
DRUDE = gfem.PermittivityLaw(
    eps_inf=1.0, terms=[gfem.Drude(plasma_frequency=20e9, damping=2e9)]
)
ENDS = {
    "constant": (gfem.Material(eps_r=2.0), gfem.Material(eps_r=12.0, mu_r=1.5)),
    "law": (gfem.Material(eps_r=2.0), gfem.Material(eps_r=DRUDE)),
}


@pytest.mark.parametrize("scheme", ["linear", "simp", "ramp", "index"])
@pytest.mark.parametrize("ends", ENDS)
def test_interpolate_keeps_the_pixels_in_the_blend(grid, scheme, ends):
    rho = pixels(grid)
    void, solid = ENDS[ends]
    field = gfem.MaterialField.interpolate(
        grid.to_elements(rho), void, solid, scheme=scheme
    )
    np.testing.assert_array_equal(field.blend, rho)
    assert field.blend_ends.pixels == grid.geometry
    assert field.blend_ends.scheme == scheme
    assert field.region == "design"
    assert field.n_elements is None
    assert field.properties == ()
    others = {name: (a, b) for name, a, b in field.blend_ends.others}
    assert others["mu_r"] == (1.0, 1.5 if ends == "constant" else 1.0)
    assert others["conductivity"] == (0.0, 0.0)
    assert "layer grid (6, 20)" in repr(field)


def test_the_geometry_locates_the_pixels(grid):
    g = grid.geometry
    assert g.region == "design"
    assert g.shape == grid.shape == (6, 20)
    assert g.pitch == grid.pitch
    assert g.origin == tuple(grid.origin)
    assert g.periodic == (False, False)


def test_anisotropic_ends_raise_for_a_layer_density(grid):
    tensor = gfem.Material(eps_r=(2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="anisotropic"):
        gfem.MaterialField.interpolate(
            grid.to_elements(pixels(grid)), gfem.Material(), tensor
        )


def test_a_layer_density_field_is_assigned_to_its_region_of_a_2d_model(mesh, grid):
    field = gfem.MaterialField.interpolate(
        grid.to_elements(pixels(grid)), gfem.Material(), gfem.Material(eps_r=4.0)
    )
    gfem.Model(mesh, materials={"feed": gfem.Material(), "design": field})
    with pytest.raises(ValueError, match="built for region 'design'"):
        gfem.Model(mesh, materials={"feed": field, "design": gfem.Material()})


def test_a_layer_density_field_on_a_3d_model_raises(grid, tmp_path):
    import structured_meshes as sm

    path = tmp_path / "stack.msh"
    layers = (("substrate", 0.2), ("design", 0.1), ("air", 0.3))
    sm.write_msh22(path, sm.plate(1.0, 0.5, layers, (8, 4, 2)))
    stack = gfem.Mesh.from_gmsh(path, length_unit="mm")
    field = gfem.MaterialField.interpolate(
        grid.to_elements(pixels(grid)), gfem.Material(), gfem.Material(eps_r=4.0)
    )
    materials = {"substrate": gfem.Material(), "design": field, "air": gfem.Material()}
    with pytest.raises(ValueError, match="2D mesh"):
        gfem.Model(stack, materials=materials)


def line(mesh, field):
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": field},
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )


def test_a_2d_job_with_a_layer_density_round_trips(mesh, grid, tmp_path):
    from gfem._solve_cache import solve_key

    field = gfem.MaterialField.interpolate(
        grid.to_elements(pixels(grid)),
        gfem.Material(eps_r=2.0),
        gfem.Material(eps_r=DRUDE),
        scheme="index",
    )
    job = gfem.Job(
        line(mesh, field),
        gfem.Driven(5e9, polarization="Ez"),
        gfem.SolverOptions(order=2),
    )
    back = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    read = back.model.materials["design"]
    assert read.blend_ends == field.blend_ends
    np.testing.assert_array_equal(read.blend, field.blend)
    assert (
        solve_key(back.model, back.problem, back.options)[0]
        == (solve_key(job.model, job.problem, job.options)[0])
    )


def test_the_solve_key_carries_the_pixels_the_geometry_and_the_order(mesh, grid):
    from gfem._solve_cache import solve_key

    def key(rho, *, pitch=1 * MM, order=2):
        g = grid if pitch == 1 * MM else design.LayerGrid(mesh, "design", pitch=pitch)
        if rho.shape != g.shape:
            rho = jnp.full(g.shape, 0.5)
        field = gfem.MaterialField.interpolate(
            g.to_elements(rho), gfem.Material(), gfem.Material(eps_r=4.0)
        )
        problem = gfem.Driven(5e9, polarization="Ez")
        options = gfem.SolverOptions(order=order)
        return solve_key(line(mesh, field), problem, options)[0]

    rho = jnp.full(grid.shape, 0.5)
    base = key(rho)
    assert key(rho.at[0, 0].set(0.6)) != base
    assert key(rho, pitch=2 * MM) != base
    assert key(rho, order=3) != base


#: A 3D layer-grid law blend as gfem wrote it before 2D designs: its materials'
#: JSON (keys sorted, zeros unsigned), and its solve key in a fixed environment
#: (the key's environment holds a digest of gfem's code).
FROZEN_3D_MATERIALS = (
    '{"air": {"$type": "Material", "axes": null, "conductivity": 0.0, "eps_r": 1.0, "'
    'mass_density": null, "mu_r": 1.0, "name": null, "specific_heat": null, "tan_delt'
    'a": 0.0, "thermal_conductivity": null, "thermo_optic_coefficient": null}, "desig'
    'n": {"$type": "MaterialField", "axes": null, "blend": {"$array": ".model.materia'
    'ls[\'design\'].blend"}, "blend_ends": {"$type": "Blend", "penalty": 3.0, "scheme":'
    ' "index", "solid": {"$type": "PermittivityLaw", "eps_inf": 1.0, "terms": [{"$typ'
    'e": "Drude", "damping": 2000000000.0, "plasma_frequency": 20000000000.0}]}, "voi'
    'd": {"$complex": [2.0, 0.0]}}, "conductivity": {"$array": ".model.materials[\'de'
    'sign\'].conductivity"}, "eps_r": null, "mass_density": null, "mu_r": {"$array": "'
    '.model.materials[\'design\'].mu_r"}, "region": null, "specific_heat": null, "tan_d'
    'elta": null, "thermal_conductivity": null, "thermo_optic_coefficient": null}, "s'
    'ubstrate": {"$type": "Material", "axes": null, "conductivity": 0.0, "eps_r": 4.0'
    ', "mass_density": null, "mu_r": 1.0, "name": null, "specific_heat": null, "tan_d'
    'elta": 0.0, "thermal_conductivity": null, "thermo_optic_coefficient": null}}'
)
FROZEN_3D_KEY = "e7384e61d177d0d8070dae468896bb3d246d65f34217864a6964e1b4d58a07fb"


def test_a_3d_layer_grid_job_and_its_key_are_unchanged(tmp_path, monkeypatch):
    import json

    import structured_meshes as sm

    from gfem import _solve_cache

    monkeypatch.setattr(_solve_cache, "_env", lambda _: {"gfem": "frozen"})
    layers = (("substrate", 0.2), ("design", 0.1), ("air", 0.3))
    path = tmp_path / "stack.msh"
    sm.write_msh22(path, sm.plate(1.0, 0.5, layers, (8, 4, 2)))
    stack = gfem.Mesh.from_gmsh(path, length_unit="mm")
    grid3 = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    ny, nx = grid3.shape
    rho = np.linspace(0.1, 0.9, ny * nx).reshape(ny, nx)
    field = gfem.MaterialField.interpolate(
        grid3.to_elements(rho),
        gfem.Material(eps_r=2.0),
        gfem.Material(eps_r=DRUDE),
        scheme="index",
    )
    materials = {
        "substrate": gfem.Material(eps_r=4.0),
        "design": field,
        "air": gfem.Material(),
    }
    model = gfem.Model(stack, materials=materials)
    problem, options = gfem.Driven(5e9), gfem.SolverOptions(order=2)
    written = gfem.Job(model, problem, options).to_json(tmp_path / "job.json")
    doc = json.loads(written.read_text())
    # -0.0 as 0.0, as the solve key writes it: the sign of a zero loss's
    # imaginary part differs between platforms
    materials = _solve_cache._canonical(doc["job"]["model"]["materials"])
    text = json.dumps(materials, sort_keys=True)
    assert text == FROZEN_3D_MATERIALS
    assert _solve_cache.solve_key(model, problem, options)[0] == FROZEN_3D_KEY


def test_the_lowest_order_points_are_the_edge_midpoints():
    import ngsolve as ngs

    rule = ngs.IntegrationRule(ngs.ET.TRIG, 2)
    points = sorted((round(p[0], 12), round(p[1], 12)) for p in rule.points)
    # the reference triangle (1, 0), (0, 1), (0, 0): its edge midpoints
    assert points == [(0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]


def test_sampled_on_a_2d_mesh_counts_the_edge_midpoints(mesh):
    nodes = np.asarray(mesh.nodes)
    triangles = mesh.volume_elements[mesh.elements("design")]
    centroids = nodes[triangles].mean(axis=1)
    # a pitch at which some pixels hold an edge midpoint but no centroid
    pitch = 0.2 * MM
    with pytest.warns(gfem.UnsampledPixelsWarning, match="lowest-order rule"):
        grid = design.LayerGrid(mesh, "design", pitch=pitch)
    midpoints = (nodes[triangles] + nodes[triangles[:, [1, 2, 0]]]) / 2

    def touched(points):
        idx, w = design._bilinear(
            points.reshape(-1, 2), grid.origin, grid.pitch, grid.shape
        )
        hit = np.bincount(idx.ravel(), w.ravel(), minlength=grid.sampled.size)
        return hit.reshape(grid.shape) > 0

    np.testing.assert_array_equal(grid.sampled, touched(midpoints))
    assert (touched(midpoints) & ~touched(centroids)).any()
