"""What thermal and DC conduction problems refuse, by name (ADRs 0161, 0184).

MPI, mesh refinement, error estimates, a GPU device and periodic models
have no path for `gfem.SteadyThermal`, `gfem.TransientThermal` and
`gfem.DCConduction`; each raises its named error before any solve.
"""

import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem
from gfem import _mpi

L, W, T = 4.0, 1.0, 0.2  # bar, um
METAL = gfem.Material(
    conductivity=1e6,
    thermal_conductivity=28.0,
    mass_density=5240.0,
    specific_heat=598.0,
)
OXIDE = gfem.Material(
    eps_r=3.9, thermal_conductivity=1.4, mass_density=2200.0, specific_heat=700.0
)
HEAT = gfem.HeatSource.uniform("bar", power=1e-3)
DIRECT = gfem.SolverOptions(order=1, linear_solver="direct")


def _problems():
    return {
        "steady": gfem.SteadyThermal(HEAT),
        "transient": gfem.TransientThermal(HEAT, duration=1e-6, steps=2),
        "dc": gfem.DCConduction(terminals={"a": "end_a"}, ground="end_b"),
    }


#: The `DistributedUnsupportedError` and `MeshRefinementUnsupportedError` feature.
FEATURE = {"steady": "thermal", "transient": "thermal", "dc": "dc_conduction"}
#: The error estimate's unadmitted-result feature (the result type's name).
ESTIMATE_FEATURE = {
    "steady": "steadythermal",
    "transient": "transientthermal",
    "dc": "dcconduction",
}
KINDS = tuple(FEATURE)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """A metal bar in an oxide box: contacts "end_a", "end_b", a "floor" sink."""
    edges = (
        np.linspace(0.0, L, 5),
        np.array([-1.0, 0.0, W, 2.0]),
        np.array([0.0, T, 0.6]),
    )
    tol = 1e-9

    def volume(c):
        return "bar" if c[2] < T and 0 < c[1] < W else "oxide"

    def surface(c):
        in_bar = c[2] < T and 0 < c[1] < W
        if abs(c[0]) < tol and in_bar:
            return "end_a"
        if abs(c[0] - L) < tol and in_bar:
            return "end_b"
        return "floor" if abs(c[2]) < tol and not in_bar else ""

    path = tmp_path_factory.mktemp("bar") / "bar.msh"
    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(path, sm.structured(edges, volume, surface))
    )
    return gfem.Model(
        mesh,
        materials={"bar": METAL, "oxide": OXIDE},
        boundaries=[gfem.FixedTemperature("floor", temperature=300.0)],
    )


@pytest.mark.parametrize("kind", KINDS)
def test_mpi_refuses_each_problem_by_name(kind, model):
    with pytest.raises(gfem.DistributedUnsupportedError, match="COMM_SELF") as caught:
        _mpi.refuse(model, _problems()[kind], gfem.SolverOptions())
    assert caught.value.feature == FEATURE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_mesh_refinement_refuses_each_problem_by_name(kind, model):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(model, _problems()[kind], tolerance=1e-3)
    assert caught.value.feature == FEATURE[kind]
    assert "no path" in str(caught.value)


@pytest.mark.parametrize("kind", KINDS)
def test_an_error_estimate_refuses_each_result_by_name(kind, model):
    result = gfem.solve(model, _problems()[kind], options=DIRECT)
    quantity = gfem.QuantityEstimator(objective=lambda _result, _model: jnp.zeros(()))
    for estimator in (gfem.FieldEstimator(), quantity):
        with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
            gfem.error_estimate(result, estimator, model=model)
        assert caught.value.feature == ESTIMATE_FEATURE[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_a_gpu_device_raises_naming_the_cpu(kind, model):
    options = gfem.SolverOptions(device="cuda")
    with pytest.raises(NotImplementedError, match="device='cpu'"):
        gfem.solve(model, _problems()[kind], options=options)


@pytest.mark.parametrize("kind", KINDS)
def test_a_periodic_model_raises_for_each_problem(kind, tmp_path):
    cell = pc.c1_mesh(tmp_path, cells=(4, 1, 1))
    material = gfem.Material(
        eps_r=2.0,
        conductivity=1e6,
        thermal_conductivity=1.0,
        mass_density=1000.0,
        specific_heat=1000.0,
    )
    model = gfem.Model(
        cell,
        materials={"cell": material},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.FixedTemperature("z_min", temperature=300.0),
        ],
    )
    problem = {
        "steady": gfem.SteadyThermal(gfem.HeatSource.uniform("cell", power=1e-3)),
        "transient": gfem.TransientThermal(duration=1e-6, steps=2),
        "dc": gfem.DCConduction(terminals={"a": "z_min"}, ground="z_max"),
    }[kind]
    with pytest.raises(NotImplementedError, match="periodic"):
        gfem.solve(model, problem, options=DIRECT)
