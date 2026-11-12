"""`gfem.electric_energy`: the electric energy a field stores in each region.

A lossy slab with a uniform loss tangent absorbs ``2 omega tan_delta W_e``
of a driven field, ``W_e`` its time-averaged electric energy, and
`gfem.power_flow` gives the absorbed power exactly for the solved field, so
the two agree to solver precision whatever the mesh. An eigenmode's
`bulk_participation` is the same integral under the mode normalization.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from anisotropy import turned

import gfem
from gfem.conventions import MODE_ENERGY

LENGTH, WIDTH, HEIGHT = 20.0, 4.0, 2.0  # um, the line along x
SLAB = (7.5, 12.5)  # um, the lossy slab "core", two cells from each port
F0 = 2.0e12
OMEGA = 2 * math.pi * F0
TAN_DELTA = 0.02
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")

SLABS = {
    "isotropic": gfem.Material(eps_r=9.0, tan_delta=TAN_DELTA),
    # a biaxial crystal on axes turned about the line, one loss tangent
    "anisotropic": gfem.Material(
        eps_r=(3.0, 5.0, 9.0), axes=turned(0, 35), tan_delta=TAN_DELTA
    ),
}


@pytest.fixture(scope="module")
def slab_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    sm.write_msh22(path, sm.strip_line(LENGTH, WIDTH, HEIGHT, (8, 1, 1), core=SLAB))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _line(mesh, slab="isotropic"):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0), "core": SLABS[slab]},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def _field(model):
    problem = gfem.Driven(frequency=F0, excite="p1", keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS).e_field(F0)


@pytest.fixture(scope="module")
def line(slab_mesh):
    model = _line(slab_mesh)
    return model, _field(model)


def test_a_lossy_slab_absorbs_twice_omega_tan_delta_times_its_energy(line):
    model, e = line
    energy = gfem.electric_energy(e, model=model)
    assert set(energy) == {"dielectric", "core"}
    assert all(w.shape == () and w.dtype == jnp.float64 for w in energy.values())
    absorbed = -float(gfem.power_flow(e, "core", model=model))
    assert absorbed > 5e-3
    assert 2 * OMEGA * TAN_DELTA * float(energy["core"]) == pytest.approx(
        absorbed, rel=1e-9
    )


def test_an_anisotropic_slab_stores_its_tensor_energy(slab_mesh):
    model = _line(slab_mesh, "anisotropic")
    e = _field(model)
    absorbed = -float(gfem.power_flow(e, "core", model=model))
    assert absorbed > 5e-3
    energy = gfem.electric_energy(e, model=model)
    assert 2 * OMEGA * TAN_DELTA * float(energy["core"]) == pytest.approx(
        absorbed, rel=1e-9
    )


# --- a periodic cell ---------------------------------------------------------------

MM = 1e-3
CELL = (8.0, 2.0, 2.0)  # mm
CELL_FREQUENCY = 20e9


def _lossless(c):
    # a slab around the port sheet and a bottom layer under the whole cell
    return (2.0 < c[0] < 4.0) or c[2] < 1.0


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    edges = tuple(
        np.linspace(0.0, s, n + 1) for s, n in zip(CELL, (8, 2, 4), strict=True)
    )
    raw = sm.periodic_cell(
        edges,
        lambda c: "core" if _lossless(c) else "lossy",
        sheets=(("sheet", 3.0),),
    )
    path = sm.write_msh22(tmp_path_factory.mktemp("cell") / "cell.msh", raw)
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "core": gfem.Material(eps_r=2.0),
            "lossy": gfem.Material(eps_r=2.0, tan_delta=0.05),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[gfem.LumpedPort("p1", "sheet", "+z", impedance=50.0)],
    )


@pytest.mark.parametrize(
    "wave_vector",
    [
        pytest.param(None, id="plain"),
        pytest.param(
            (0.3 * math.pi / (CELL[0] * MM), 0.2 * math.pi / (CELL[1] * MM), 0.0),
            id="bloch",
        ),
    ],
)
def test_a_periodic_cell_absorbs_its_lossy_region_energy(cell, wave_vector):
    problem = gfem.Driven(
        frequency=CELL_FREQUENCY,
        excite="p1",
        keep_fields=True,
        wave_vector=wave_vector,
    )
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    e = gfem.solve(cell, problem, options=options).e_field(CELL_FREQUENCY)
    absorbed = -float(gfem.power_flow(e, "lossy", model=cell))
    assert absorbed > 0.01
    energy = gfem.electric_energy(e, model=cell)
    loss = 2 * (2 * math.pi * CELL_FREQUENCY) * 0.05 * float(energy["lossy"])
    assert loss == pytest.approx(absorbed, rel=1e-9)


# --- an eigenmode ------------------------------------------------------------------


def test_a_mode_field_holds_half_its_bulk_participation(tmp_path):
    # the mode normalization counts 1/2 Re ∫ D* . E, twice the time average
    path = tmp_path / "layered.msh"
    layers = (("low", 0.3), ("high", 0.7))
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), layers=layers))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"low": gfem.Material(eps_r=4.0), "high": gfem.Material(eps_r=2.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )
    modes = gfem.solve(model, gfem.Eigenmode(target=8e9))
    energy = gfem.electric_energy(modes.e_field(0), model=model)
    assert set(energy) == {"low", "high"}
    for region, share in modes.bulk_participation.items():
        assert 2 * float(energy[region]) == pytest.approx(
            float(share[0]) * MODE_ENERGY, rel=1e-10
        )


# --- what it refuses ---------------------------------------------------------------


def test_electric_energy_names_what_it_needs(line):
    model, e = line
    with pytest.raises(TypeError, match=r"gfem\.Field"):
        gfem.electric_energy(np.asarray(e.coefficients), model=model)
    with pytest.raises(TypeError, match=r"model= a gfem\.Model"):
        gfem.electric_energy(e, model=model.mesh)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.b(), "quantity 'B' in 'hdiv'"),
        (lambda e: dataclasses.replace(e, omega=None), "static"),
        (lambda e: dataclasses.replace(e, order=2), "not solved on this model"),
    ],
)
def test_electric_energy_needs_a_time_harmonic_e_field(line, change, message):
    model, e = line
    with pytest.raises(ValueError, match=message):
        gfem.electric_energy(change(e), model=model)


def test_electric_energy_needs_the_model_the_field_was_solved_on(line, cell):
    _, e = line
    with pytest.raises(ValueError, match="another mesh"):
        gfem.electric_energy(e, model=cell)


def test_electric_energy_is_not_differentiable(line):
    model, e = line

    def of_the_field(scale):
        field = dataclasses.replace(e, coefficients=scale * e.coefficients)
        return gfem.electric_energy(field, model=model)["core"]

    def of_the_model(eps):
        n = len(model.mesh.elements("core"))
        fill = gfem.MaterialField(eps_r=jnp.full(n, eps))
        changed = model.with_material("core", fill)
        return gfem.electric_energy(e, model=changed)["core"]

    for objective, at in ((of_the_field, 1.0), (of_the_model, 9.0)):
        with pytest.raises(gfem.NonDifferentiableError, match="electric_energy"):
            jax.grad(objective)(at)
