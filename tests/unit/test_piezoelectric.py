"""A piezoelectric solve of a 2D transducer section: admittance, capacitance, fields."""

import numpy as np
import piezo_cases as pc
import pytest

import gfem

VARIANTS = ("lines", "meshed")


@pytest.fixture(scope="module", params=VARIANTS)
def device(request, tmp_path_factory):
    variant = request.param
    path = tmp_path_factory.mktemp(variant) / "idt.msh"
    mesh = pc.mesh(path, variant)
    return variant, mesh, pc.model(mesh, variant)


def test_a_solve_gives_the_admittance_and_the_clamped_capacitance(device):
    _, _, model = device
    result = gfem.solve(model, pc.problem([0.9e9, pc.FREQUENCY]))
    assert isinstance(result, gfem.PiezoelectricResult)
    assert result.terminals == ("hot",)
    assert result.admittance_per_length.shape == (2, 1, 1)
    assert result.clamped_capacitance_per_length.shape == (1, 1)
    y = np.asarray(result.admittance_per_length)
    assert np.isfinite(y).all()
    assert (y.imag > 0).all()  # capacitive
    assert float(result.clamped_capacitance_per_length[0, 0]) > 0


def _system(model, problem):
    """The assembled system of a problem: its states and operator pieces."""
    from gfem._engine.piezoelectric import PiezoelectricRule, PiezoelectricSystem
    from gfem._engine.runtime import task_manager

    options = gfem.SolverOptions()
    rule = PiezoelectricRule(model, problem, options)
    with task_manager(options.threads):
        return PiezoelectricSystem(model, problem, options, rule)


def _raw(model, problem):
    """Y' before the symmetrization, and C0."""
    from gfem._engine.piezoelectric import PiezoelectricRule

    options = gfem.SolverOptions()
    rule = PiezoelectricRule(model, problem, options)
    out, _ = rule.forward(model, keep=False)
    return out


def _rel(a, b):
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


def test_the_loss_balances_the_conductance(device):
    _, _, model = device
    system = _system(model, pc.problem())
    x, ax = system.state(pc.FREQUENCY)
    omega = 2 * np.pi * pc.FREQUENCY
    y = -1j * omega * (x.T @ ax)  # the quadratic form
    probe = system.matrix.CreateColVector()
    out = system.matrix.CreateColVector()
    probe.FV().NumPy()[:] = x[:, 0]
    out.data = system.loss_matrix * probe
    dissipated = np.vdot(x[:, 0], out.FV().NumPy())
    assert abs(dissipated.imag) / abs(dissipated) < 1e-12
    assert abs(y[0, 0].real - omega * dissipated.real) / abs(y[0, 0]) < 1e-12


def test_without_the_piezoelectric_tensor_the_admittance_is_the_clamped_capacitance(
    device,
):
    variant, mesh, _ = device
    model = pc.model(mesh, variant, piezoelectric=False)
    result = gfem.solve(model, pc.problem())
    y = np.asarray(result.admittance_per_length)[0, 0, 0]
    c0 = float(result.clamped_capacitance_per_length[0, 0])
    assert abs(y - 2j * np.pi * pc.FREQUENCY * c0) / abs(y) < 1e-12


def test_a_lossless_section_without_a_frame_has_no_conductance(device):
    variant, mesh, _ = device
    model = pc.model(mesh, variant, quality=None)
    result = gfem.solve(model, pc.problem(frame=False))
    y = np.asarray(result.admittance_per_length)[0, 0, 0]
    assert abs(y.real) / abs(y) < 1e-12


def test_two_terminals_are_reciprocal_neutral_and_dissipative(device):
    _, _, model = device
    problem = pc.problem(terminals=("hot", "gnd"), ground=())
    out = _raw(model, problem)
    raw = out["admittance_raw"][0]
    scale = np.abs(raw).max()
    assert abs(raw[0, 1] - raw[1, 0]) / scale < 1e-12
    assert np.abs(raw.sum(axis=0)).max() / scale < 1e-12  # charge neutrality
    real = out["admittance"][0].real
    assert np.linalg.eigvalsh(real).min() > -1e-12 * np.abs(real).max()


@pytest.mark.parametrize("angle", [np.pi / 2, np.deg2rad(37.0)])
def test_a_rigid_rotation_of_the_nodes_and_axes_changes_nothing(device, angle):
    variant, mesh, model = device
    reference = gfem.solve(model, pc.problem())
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    nodes = np.asarray(mesh.nodes)
    moved = mesh.with_nodes(nodes @ rotation[:2, :2].T)
    axes = np.asarray(pc.AXES) @ rotation.T
    turned = pc.model(moved, variant, axes=tuple(map(tuple, axes)))
    result = gfem.solve(turned, pc.problem())
    assert _rel(result.admittance_per_length, reference.admittance_per_length) < 1e-10
    assert (
        _rel(
            result.clamped_capacitance_per_length,
            reference.clamped_capacitance_per_length,
        )
        < 1e-10
    )


# --- the kept fields -------------------------------------------------------------


def test_kept_fields_read_the_displacement_and_the_potential(device):
    variant, _, model = device
    frequencies = [0.9e9, pc.FREQUENCY]
    result = gfem.solve(
        model,
        pc.problem(
            frequencies, terminals=("hot", "gnd"), ground=(), keep_fields=[pc.FREQUENCY]
        ),
    )
    assert result.kept == ((pc.FREQUENCY, "hot"), (pc.FREQUENCY, "gnd"))
    with pytest.raises(ValueError, match="pass terminal="):
        result.potential(pc.FREQUENCY)
    with pytest.raises(ValueError, match="No field kept at 9e\\+08 Hz"):
        result.potential(0.9e9, "hot")
    with pytest.raises(KeyError, match="No terminal 'x'"):
        result.displacement(pc.FREQUENCY, "x")
    # 1 V on the hot electrode, 0 V on the other one
    x, y = next((a + b) / 2 for a, b, n in pc.fingers() if n == "hot"), 0.1e0
    y = 0.0 if variant == "lines" else pc.T_AL / 2
    hot = result.potential(pc.FREQUENCY, "hot")
    gnd = result.potential(pc.FREQUENCY, "gnd")
    point = np.array([[x * 1e-6, y * 1e-6]])
    assert abs(complex(hot.at(point)[0]) - 1.0) < 1e-9
    assert abs(complex(gnd.at(point)[0])) < 1e-9
    u = result.displacement(pc.FREQUENCY, "hot")
    assert list(u) == ["x", "y", "z"]
    assert u["z"].quantity == "uz"
    below = np.array([[x * 1e-6, -0.5e-6]])
    assert abs(complex(u["z"].at(below)[0])) > 0
    # the kept state is the one the admittance came from
    assert hot.omega == pytest.approx(2 * np.pi * pc.FREQUENCY)


# --- refusals --------------------------------------------------------------------


def _raises(error, match, call):
    with pytest.raises(error, match=match):
        call()


def test_a_solve_on_a_tetrahedral_or_slab_mesh_refuses_by_name(tmp_path):
    import thermal_cases as tc

    problem = pc.problem()
    tets = tc.brick(
        tmp_path / "brick.msh",
        (1.0, 1.0),
        [("a", 1.0)],
        0.5,
        [(lambda p: p[2] < 1e-9, "hot")],
    )
    slab = tc.section(
        tmp_path / "slab.msh",
        [(0, 0, 1, 1, "a", None)],
        0.5,
        [(lambda p: p[1] < 1e-9, "hot")],
    )
    for mesh in (tets, slab):
        model = gfem.Model(mesh, materials={"a": gfem.Material()})
        _raises(
            NotImplementedError,
            r"Piezoelectric problem solves a 2D mesh.*triangle-only",
            lambda model=model: gfem.solve(model, problem),
        )


def test_the_numerics_the_rule_does_not_have_refuse_by_name(device):
    _, _, model = device
    petsc = gfem.SolverOptions(linear_solver="petsc")
    _raises(
        NotImplementedError,
        r"no PETSc path.*linear_solver=\"direct\"",
        lambda: gfem.solve(model, pc.problem(), options=petsc),
    )
    workers = gfem.SolverOptions(frequency_workers=2)
    _raises(
        NotImplementedError,
        r"one process.*frequency_workers",
        lambda: gfem.solve(model, pc.problem([0.9e9, 1e9]), options=workers),
    )


def test_an_absorbing_line_on_the_mechanical_domain_raises(device):
    _, mesh, base = device
    model = gfem.Model(
        mesh,
        materials=base.materials,
        boundaries=(gfem.Absorbing("bottom"),),
        excluded={"mechanical": ["air"]},
    )
    _raises(
        ValueError,
        r"Absorbing on 'bottom'.*damping_frame=gfem\.DampingFrame",
        lambda: gfem.solve(model, pc.problem()),
    )


def test_an_area_terminal_outside_the_electromagnetic_domain_raises(tmp_path):
    mesh = pc.mesh(tmp_path / "idt.msh", "meshed")
    model = pc.model(mesh, "meshed").replace(
        excluded={"mechanical": ["air"], "em": ["hot"]}
    )
    _raises(
        ValueError,
        r"terminal 'hot' is an area outside the electromagnetic domain.*shell lines",
        lambda: gfem.solve(model, pc.problem()),
    )


def test_a_mechanical_region_without_stiffness_or_density_lists_every_gap(device):
    _, mesh, model = device
    materials = {k: v for k, v in model.materials.items() if k != "air"}
    materials["air"] = gfem.Material()
    bare = gfem.Model(mesh, materials=materials)
    with pytest.raises(gfem.MissingMaterialPropertyError) as info:
        gfem.solve(bare, pc.problem())
    assert ("air", "stiffness") in info.value.missing
    assert ("air", "mass_density") in info.value.missing
    assert "mechanical domain" in str(info.value)


def test_a_model_with_every_volume_out_of_the_mechanical_domain_raises(device):
    _, mesh, model = device
    names = list(mesh.volumes)
    bare = model.replace(excluded={"mechanical": names})
    _raises(
        ValueError,
        r"excluded from mechanical problems",
        lambda: gfem.solve(bare, pc.problem()),
    )


def test_a_piezoelectric_region_outside_the_electromagnetic_domain_raises(device):
    _, _, model = device
    bare = model.replace(excluded={"mechanical": ["air"], "em": ["ln"]})
    _raises(
        ValueError,
        r"Region 'ln' is piezoelectric but excluded from \"em\"",
        lambda: gfem.solve(bare, pc.problem()),
    )


def test_names_the_mesh_lacks_or_misuses_raise(device):
    _, _, model = device
    _raises(
        gfem.RegionNotFoundError,
        "nowhere",
        lambda: gfem.solve(model, pc.problem(ground=("nowhere",))),
    )
    _raises(
        ValueError,
        r"clamped 'air' must be a line",
        lambda: gfem.solve(model, pc.problem(clamped="air")),
    )
    _raises(
        ValueError,
        r"DampingFrame region 'air' is outside the mechanical domain",
        lambda: gfem.solve(
            model,
            gfem.Piezoelectric(
                pc.FREQUENCY,
                terminals=["hot"],
                damping_frame=gfem.DampingFrame(["air"]),
            ),
        ),
    )
    _raises(
        TypeError,
        r"DampingFrame region 'bottom' is a line",
        lambda: gfem.solve(
            model,
            gfem.Piezoelectric(
                pc.FREQUENCY,
                terminals=["hot"],
                damping_frame=gfem.DampingFrame(["bottom"]),
            ),
        ),
    )


def test_a_model_traced_by_jax_raises_until_gradients_exist(device):
    import jax

    _, mesh, model = device

    def objective(scale):
        moved = mesh.with_nodes(mesh.nodes * scale)
        result = gfem.solve(model.replace(mesh=moved), pc.problem())
        return result.clamped_capacitance_per_length[0, 0]

    _raises(
        gfem.NonDifferentiableError,
        "not available yet",
        lambda: jax.grad(objective)(1.0),
    )


# --- the problem and its exports ---------------------------------------------------


def test_the_problem_validates_and_reads_back():
    frame = gfem.DampingFrame(["a", "b"], loss_factor=0.6)
    assert repr(frame) == "DampingFrame(['a', 'b'], loss_factor=0.6)"
    problem = gfem.Piezoelectric(
        [1e9, 2e9],
        terminals={"s": "sig"},
        ground="g",
        clamped="bottom",
        damping_frame=frame,
    )
    assert problem.terminal_names == ("s",)
    assert problem.ground == ("g",)
    assert problem.clamped == ("bottom",)
    assert repr(problem) == (
        "Piezoelectric(2 frequencies 1-2 GHz, terminals={'s': ['sig']}, "
        "ground=['g'], clamped=['bottom'], "
        "damping_frame=DampingFrame(['a', 'b'], loss_factor=0.6))"
    )
    _raises(ValueError, "must be > 0", lambda: gfem.DampingFrame("a", loss_factor=0))
    _raises(ValueError, "listed twice", lambda: gfem.DampingFrame(["a", "a"]))
    _raises(TypeError, "area region names", lambda: gfem.DampingFrame([]))
    _raises(
        TypeError,
        "gfem.DampingFrame or None",
        lambda: gfem.Piezoelectric(1e9, terminals=["s"], damping_frame="a"),
    )
    _raises(
        ValueError, "twice", lambda: gfem.Piezoelectric([1e9, 1e9], terminals=["s"])
    )
    _raises(
        ValueError,
        r"Piezoelectric.keep_fields frequency",
        lambda: gfem.Piezoelectric(1e9, terminals=["s"], keep_fields=[2e9]),
    )


def test_a_traced_frequency_or_loss_factor_raises():
    import jax

    _raises(
        gfem.NonDifferentiableError,
        "fixed",
        lambda: jax.grad(lambda f: gfem.Piezoelectric(f, terminals=["s"]) and 0.0)(1e9),
    )
    _raises(
        gfem.NonDifferentiableError,
        "fixed",
        lambda: jax.grad(lambda eta: gfem.DampingFrame("a", loss_factor=eta) and 0.0)(
            1.0
        ),
    )


def test_the_new_names_are_exported_from_their_modules():
    from gfem import problems, results

    assert gfem.Piezoelectric is problems.Piezoelectric
    assert gfem.DampingFrame is problems.DampingFrame
    assert gfem.PiezoelectricResult is results.PiezoelectricResult
    for name in ("Piezoelectric", "DampingFrame", "PiezoelectricResult"):
        assert name in gfem.__all__
