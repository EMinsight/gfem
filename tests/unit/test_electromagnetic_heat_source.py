"""`gfem.electromagnetic_heating`: a driven result's loss as a thermal heat source.

The 10 x 2 x 1 mm parallel-plate line of `test_electromagnetic_heating.py`
(lumped ports at both ends, a lossy 2 mm "core" one element from each port)
carries the volume checks: the heat of a drive is the incident power times
the dissipated power of the 1 W field (`gfem.dissipated_power`), and a
coherent drive of both ports heats as the superposed field does. The same
model is a thermal one (thermal conductivities, a fixed-temperature bottom
plate), so a driven solve, its heat and a steady solve chain on one mesh.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem.thermal import _tet_volumes, _triangle_areas

F = 7e9
#: Order 3: the lumped ports' S is ~1e-6 off the loss at order 2 on these
#: one-cell port faces (see test_electromagnetic_heating.py).
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
K_DIELECTRIC, K_CORE = 1.4, 30.0


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, *, tan_delta=0.05, k=K_CORE):
    n = len(mesh.elements("core"))
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=4.0, thermal_conductivity=K_DIELECTRIC),
            "core": gfem.MaterialField(
                eps_r=jnp.full(n, 3.0),
                tan_delta=tan_delta * jnp.ones(n),
                thermal_conductivity=k * jnp.ones(n),
            ),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.FixedTemperature("bottom", temperature=300.0),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _solve(model, excite=None):
    problem = gfem.Driven(frequency=F, excite=excite, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


@pytest.fixture(scope="module")
def line(mesh):
    model = _line(mesh)
    return model, _solve(model)


@pytest.fixture(scope="module")
def single(line):
    """The line's result with port p1 alone excited."""
    return _solve(line[0], excite="p1")


def _total(heat, mesh):
    """W: the volume and face heat of a source on ``mesh``."""
    volume = jnp.sum(heat._density_on(mesh) * _tet_volumes(mesh))
    return volume + jnp.sum(heat._face_density_on(mesh) * _triangle_areas(mesh))


def _lost(result, model, port):
    lost = gfem.dissipated_power(result.e_field(F, port), model=model)
    return sum(float(v) for v in lost.values())


def test_the_heat_is_the_incident_power_times_the_dissipated_power(line):
    model, result = line
    heat = gfem.electromagnetic_heating(
        result, model=model, frequency=F, incident_power={"p1": 2e-3}
    )
    want = 2e-3 * _lost(result, model, "p1")
    assert want > 1e-5
    assert float(_total(heat, model.mesh)) == pytest.approx(want, rel=1e-12)
    density = np.asarray(heat._density_on(model.mesh))
    assert not density[model.mesh.elements("dielectric")].any()
    assert (density[model.mesh.elements("core")] > 0).all()
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    assert float(steady.power) == pytest.approx(want, rel=1e-12)
    assert repr(heat) == (
        f"HeatSource(electromagnetic, regions=['dielectric', 'core'], "
        f"total {want:.6g} W)"
    )


# --- a coherent drive ---------------------------------------------------------


def _drive(result, model, phase=None, **powers):
    return gfem.electromagnetic_heating(
        result, model=model, frequency=F, incident_power=powers, incident_phase=phase
    )


def test_two_ports_driven_together_heat_as_the_superposed_field(line):
    model, result = line
    e1, e2 = result.e_field(F, "p1"), result.e_field(F, "p2")
    powers, phase = {"p1": 3e-3, "p2": 1e-3}, 0.7
    a1, a2 = np.sqrt(powers["p1"]), np.sqrt(powers["p2"]) * np.exp(1j * phase)
    both = dataclasses.replace(
        e1, coefficients=a1 * e1.coefficients + a2 * e2.coefficients
    )
    want = -float(gfem.power_flow(both, "core", model=model))
    heat = _drive(result, model, {"p2": phase}, **powers)
    density = heat._density_on(model.mesh)
    core = model.mesh.elements("core")
    got = float(jnp.sum((density * _tet_volumes(model.mesh))[core]))
    assert got == pytest.approx(want, rel=1e-12)
    assert want > 1e-5


def test_a_phase_of_pi_cancels_the_cross_terms(line):
    # the cross terms flip sign with the relative phase, the ports' own heat
    # does not: the in-phase and anti-phase heats average to the incoherent one
    model, result = line
    mesh = model.mesh
    even = float(_total(_drive(result, model, p1=1e-3, p2=1e-3), mesh))
    odd = float(_total(_drive(result, model, {"p2": np.pi}, p1=1e-3, p2=1e-3), mesh))
    alone = sum(
        float(_total(_drive(result, model, **{f"p{i}": 1e-3}), mesh)) for i in (1, 2)
    )
    assert abs(even - odd) > 1e-2 * alone
    assert (even + odd) / 2 == pytest.approx(alone, rel=1e-12)


def test_a_port_at_0_w_with_no_driven_partner_has_a_finite_gradient(line, single):
    # the diagonal term is linear in the power: d heat / d P1 at 0 W is the
    # 1 W field's dissipated power, and the cross term with a 0 W partner
    # gives nothing (no 0 * inf)
    model, result = line

    def total(p):
        return _total(_drive(result, model, p1=p, p2=0.0), model.mesh)

    grad = float(jax.grad(total)(0.0))
    assert grad == pytest.approx(_lost(result, model, "p1"), rel=1e-12)
    one = jax.grad(
        lambda p: _total(
            gfem.electromagnetic_heating(
                single, model=model, frequency=F, incident_power=p
            ),
            model.mesh,
        )
    )(0.0)
    assert float(one) == pytest.approx(_lost(single, model, "p1"), rel=1e-12)


def test_a_cross_term_at_0_w_next_to_a_driven_partner_is_unbounded(line):
    model, result = line

    def total(p):
        return _total(_drive(result, model, p1=1e-3, p2=p), model.mesh)

    assert np.isinf(float(jax.grad(total)(0.0)))


def test_heat_compiles_and_maps_as_joule_heating_does(line):
    model, result = line
    want = _lost(result, model, "p1")

    def total(p):
        return _total(_drive(result, model, p1=p), model.mesh)

    powers = jnp.asarray([1e-3, 2e-3, 0.0])
    np.testing.assert_allclose(jax.vmap(total)(powers), want * powers, rtol=1e-12)
    assert float(jax.jit(total)(2e-3)) == pytest.approx(2e-3 * want, rel=1e-12)
    traced = jax.jit(lambda p: _drive(result, model, p1=p))(1e-3)
    assert repr(traced) == (
        "HeatSource(electromagnetic, regions=['dielectric', 'core'])"
    )


# --- a cross-section slab -----------------------------------------------------

#: The tutorial's 40 x 6 mm line (docs/tutorials/data/line2d.msh): a lossy
#: "design" section between two lossless "feed" lengths, wave ports at both
#: ends, posed on the slab with PEC front and back (the Ez line).
LINE2D = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line2d.msh"
SLAB_F = 5e9
SLAB_OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture(scope="module")
def slab():
    mesh = gfem.Mesh.from_gmsh_cross_section(LINE2D, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "feed": gfem.Material(thermal_conductivity=0.03),
            "design": gfem.Material(
                eps_r=10.0, tan_delta=0.05, thermal_conductivity=1.0
            ),
        },
        boundaries=(
            gfem.PEC("front"),
            gfem.PEC("back"),
            gfem.FixedTemperature("side", temperature=300.0),
        ),
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )
    problem = gfem.Driven(SLAB_F, keep_fields=True)
    return model, gfem.solve(model, problem, options=SLAB_OPTIONS)


def test_a_slab_is_heated_per_length(slab):
    model, result = slab
    per_length = 2.0  # W/m
    heat = gfem.electromagnetic_heating(
        result,
        model=model,
        frequency=SLAB_F,
        incident_power_per_length={"p1": per_length},
    )
    lost = sum(
        float(v)
        for v in gfem.dissipated_power(
            result.e_field(SLAB_F, "p1"), model=model
        ).values()
    )
    assert lost > 0.1
    h = model.mesh.section_thickness
    want = per_length * lost
    assert float(_total(heat, model.mesh)) / h == pytest.approx(want, rel=1e-12)
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    assert float(steady.power_per_length) == pytest.approx(want, rel=1e-12)
    assert repr(heat) == (
        f"HeatSource(electromagnetic, regions=['feed', 'design'], total {want:.6g} W/m)"
    )


def test_the_phase_is_given_at_the_reference_plane(slab):
    # moving port p2's plane half a guided wavelength into the line turns its
    # incident wave by pi at the port: the drive at the new plane with phase 0
    # is the drive at the old one with phase pi
    model, result = slab
    gamma = complex(np.asarray(result.port_modes["p2"].gamma)[0])
    moved = result.with_offsets({"p2": np.pi / abs(gamma.imag)})
    drive = {"p1": 1.0, "p2": 0.5}

    def heat(r, phase):
        source = gfem.electromagnetic_heating(
            r,
            model=model,
            frequency=SLAB_F,
            incident_power_per_length=drive,
            incident_phase=phase,
        )
        return float(_total(source, model.mesh))

    flipped = heat(result, {"p2": np.pi})
    assert abs(flipped - heat(result, None)) > 1e-2 * flipped
    assert heat(moved, None) == pytest.approx(flipped, rel=1e-9)


# --- sheet heat -----------------------------------------------------------------

#: Case A of the closed forms: a 10 x 6 x 1 mm parallel-plate feed (wave
#: port at x = 0) ended by a `Conductor` wall at x = 10 mm. The lossless feed
#: dissipates nothing, so every watt the port does not get back heats the wall.
WALL_F = 5e9


def _stub(path):
    edges = (np.linspace(0, 10, 21), np.linspace(0, 6, 7), np.array([0.0, 1.0]))

    def face(c):
        x, y, z = c
        if abs(x) < 1e-9:
            return "port1"
        if abs(x - 10) < 1e-9:
            return "wall"
        if abs(y) < 1e-9 or abs(y - 6) < 1e-9:
            return "side"
        return "plates" if abs(z) < 1e-9 or abs(z - 1) < 1e-9 else ""

    sm.write_msh22(path, sm.structured(edges, lambda _: "feed", face))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def test_a_conductor_wall_is_heated_by_what_the_port_does_not_get_back(tmp_path):
    model = gfem.Model(
        _stub(tmp_path / "stub.msh"),
        materials={"feed": gfem.Material(thermal_conductivity=0.03)},
        boundaries=(
            gfem.PEC("plates"),
            gfem.Conductor("wall", conductivity=80.0),
            gfem.FixedTemperature("port1", temperature=300.0),
        ),
        ports=(gfem.WavePort("p1", "port1"),),
    )
    problem = gfem.Driven(WALL_F, keep_fields=True)
    result = gfem.solve(model, problem, options=SLAB_OPTIONS)
    absorbed = 1 - abs(complex(np.asarray(result.s["p1", "p1"])[0])) ** 2
    assert absorbed > 0.1
    heat = gfem.electromagnetic_heating(
        result, model=model, frequency=WALL_F, incident_power=1.0
    )
    mesh = model.mesh
    face = np.asarray(heat._face_density_on(mesh)) * _triangle_areas(mesh)
    wall = mesh.elements("wall")
    assert face[wall].sum() == pytest.approx(absorbed, abs=1e-9)
    assert not np.delete(face, wall).any()
    assert not np.asarray(heat._density_on(mesh)).any()
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    assert float(steady.power) == pytest.approx(face.sum(), rel=1e-12)
    sheet = heat._sheet_total
    assert repr(heat) == (
        f"HeatSource(electromagnetic, regions=['feed'], total {sheet:.6g} W: "
        f"volume 0 W, sheet {sheet:.6g} W)"
    )


def sheet_line_model(mesh, sheet):
    """A lossy line loaded by an interior sheet ``sheet``, both ports fed.

    The ports are near the line's 530 Ohm and the sheet is light, so each
    port's field crosses the sheet and the two overlap there.
    """
    half = gfem.Material(eps_r=2.0, tan_delta=0.01, thermal_conductivity=1.0)
    return gfem.Model(
        mesh,
        materials={"lower": half, "upper": half},
        boundaries=[
            gfem.PEC("plate1"),
            gfem.PEC("plate2"),
            sheet,
            gfem.FixedTemperature("plate1", temperature=300.0),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=530.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=530.0),
        ],
    )


@pytest.fixture(scope="module")
def sheet_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(12.0, 2.0, 1.0, (12, 2, 1), span=(4.0, 8.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def sheet_line(sheet_mesh):
    """`sheet_line_model` with an `Impedance` sheet, and its result."""
    model = sheet_line_model(sheet_mesh, gfem.Impedance("sheet", resistance=2000.0))
    problem = gfem.Driven(frequency=SHEET_F, keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


SHEET_F = 6e9


def test_two_ports_heat_a_sheet_as_the_superposed_field(sheet_line):
    # the face cross terms: the sheet's heat under a coherent drive is the
    # sheet's dissipated power of the superposed field
    model, result = sheet_line
    e1, e2 = result.e_field(SHEET_F, "p1"), result.e_field(SHEET_F, "p2")
    powers, phase = {"p1": 2e-3, "p2": 5e-4}, 1.8
    a1, a2 = np.sqrt(powers["p1"]), np.sqrt(powers["p2"]) * np.exp(1j * phase)
    both = dataclasses.replace(
        e1, coefficients=a1 * e1.coefficients + a2 * e2.coefficients
    )
    want = float(gfem.dissipated_power(both, model=model)["sheet"])
    alone = sum(
        p * float(gfem.dissipated_power(e, model=model)["sheet"])
        for p, e in zip(powers.values(), (e1, e2), strict=True)
    )
    assert abs(want - alone) > 1e-2 * want  # the cross term matters
    heat = gfem.electromagnetic_heating(
        result,
        model=model,
        frequency=SHEET_F,
        incident_power=powers,
        incident_phase={"p2": phase},
    )
    mesh = model.mesh
    face = heat._face_density_on(mesh) * _triangle_areas(mesh)
    assert float(jnp.sum(face)) == pytest.approx(want, rel=1e-12)
    assert heat._sheet_total == pytest.approx(want, rel=1e-12)


# --- refusals --------------------------------------------------------------------


def _heat(result, model, **kwargs):
    kwargs = {"frequency": F, "incident_power": {"p1": 1e-3}} | kwargs
    return gfem.electromagnetic_heating(result, model=model, **kwargs)


def test_other_results_are_refused_by_name(line):
    model, _ = line
    with pytest.raises(TypeError, match=r"gfem\.joule_heating"):
        _heat(object.__new__(gfem.DCConductionResult), model)
    with pytest.raises(
        NotImplementedError, match=r"HeatSource\.uniform\(\.\.\., power_per_length=\)"
    ):
        _heat(object.__new__(gfem.PortModeResult), model)
    with pytest.raises(TypeError, match="an eigenmode has no drive"):
        _heat(object.__new__(gfem.EigenmodeResult), model)
    with pytest.raises(TypeError, match=r"gfem\.Driven.*got dict"):
        _heat({}, model)
    with pytest.raises(TypeError, match=r"model= a gfem\.Model"):
        _heat(line[1], model.mesh)


def test_the_drive_must_fit_the_result(line, single):
    model, result = line
    with pytest.raises(KeyError, match=r"'p3'.*not excited.*Driven\(excite="):
        _heat(result, model, incident_power={"p3": 1e-3})
    with pytest.raises(KeyError, match=r"'p2'.*Driven\(excite="):
        _heat(single, model, incident_power={"p2": 1e-3})
    with pytest.raises(ValueError, match=r"2 excited ports.*incident_power=\{'p1'"):
        _heat(result, model, incident_power=1e-3)
    with pytest.raises(ValueError, match="No field kept at 8e"):
        _heat(result, model, frequency=8e9)
    with pytest.raises(ValueError, match="exactly one drive"):
        _heat(result, model, incident_power=None)
    with pytest.raises(ValueError, match="exactly one drive"):
        _heat(result, model, incident_power_per_length={"p1": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        _heat(result, model, incident_power={"p1": -1e-3})


def test_the_per_length_drive_is_the_cross_section_s(line, slab):
    model, result = line
    with pytest.raises(ValueError, match="give the total incident_power= in W"):
        _heat(result, model, incident_power=None, incident_power_per_length=1.0)
    model, result = slab
    with pytest.raises(
        ValueError, match=r"incident_power= is not accepted.*incident_power_per_length="
    ):
        _heat(result, model, frequency=SLAB_F)


def test_the_phase_is_a_mapping_over_the_driven_ports(line):
    model, result = line
    with pytest.raises(TypeError, match="relative, so one number cannot carry it"):
        _heat(result, model, incident_phase=1.0)
    with pytest.raises(KeyError, match=r"'p2'.*does not feed.*incident power"):
        _heat(result, model, incident_phase={"p2": 1.0})


def test_a_traced_frequency_is_refused(line):
    model, result = line

    def total(f):
        return _total(_heat(result, model, frequency=f), model.mesh)

    with pytest.raises(gfem.NonDifferentiableError, match="fixed input"):
        jax.grad(total)(F)


def test_a_result_across_ranks_is_refused(line):
    from gfem import _mpi

    model, result = line
    pieces = [
        dataclasses.replace(e, _comm=_mpi.StaticComm(4, object()))
        for e in result.e_fields
    ]
    spread = dataclasses.replace(result, e_fields=pieces)
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        _heat(spread, model)
    assert caught.value.feature == "electromagnetic_heating"


# --- gradients through the chain -------------------------------------------------
# driven solve -> heat -> steady thermal solve -> mean temperature


def _check(objective, p, *, step, seed=0):
    p = jnp.asarray(p, jnp.float64)
    directions = unit_directions(np.random.default_rng(seed), p.shape, 1)
    assert_gradient_matches(objective, p, directions, step=step)


def _mean_rise(model, heat, region):
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    return gfem.region_mean(steady.temperature, region)


def _line_chain(mesh, *, tan_delta=0.05, k=K_CORE, powers=None, phase=None):
    model = _line(mesh, tan_delta=tan_delta, k=k)
    heat = gfem.electromagnetic_heating(
        _solve(model),
        model=model,
        frequency=F,
        incident_power=powers or {"p1": 0.2, "p2": 0.1},
        incident_phase=phase,
    )
    return _mean_rise(model, heat, "core")


def test_chain_gradient_by_a_loss_tangent(mesh):
    _check(lambda p: _line_chain(mesh, tan_delta=p), 0.05, step=1e-3)


def test_chain_gradient_by_a_thermal_conductivity(mesh):
    _check(lambda p: _line_chain(mesh, k=p), K_CORE, step=0.5)


def test_chain_gradient_by_the_incident_powers_and_phase(mesh):
    def objective(p):
        return _line_chain(mesh, powers={"p1": p[0], "p2": p[1]}, phase={"p2": p[2]})

    _check(objective, [0.2, 0.1, 0.7], step=1e-3)


def _sheet_chain(mesh, sheet):
    def objective(p):
        model = sheet_line_model(mesh, sheet(p))
        problem = gfem.Driven(frequency=SHEET_F, keep_fields=True)
        heat = gfem.electromagnetic_heating(
            gfem.solve(model, problem, options=OPTIONS),
            model=model,
            frequency=SHEET_F,
            incident_power={"p1": 0.2, "p2": 0.05},
            incident_phase={"p2": 0.4},
        )
        return _mean_rise(model, heat, "upper")

    return objective


def test_chain_gradient_by_an_impedance_resistance(sheet_mesh):
    _check(
        _sheet_chain(sheet_mesh, lambda r: gfem.Impedance("sheet", resistance=r)),
        2000.0,
        step=10.0,
    )


def test_chain_gradient_by_a_sheet_fill(sheet_mesh):
    n = len(sheet_mesh.elements("sheet"))
    fill = np.random.default_rng(7).uniform(0.3, 1.0, n)
    _check(
        _sheet_chain(
            sheet_mesh,
            lambda p: gfem.Impedance(
                "sheet", resistance=2000.0, fill=gfem.SheetField(fill=p)
            ),
        ),
        fill,
        step=1e-2,
    )
