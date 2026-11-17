"""`gfem.SteadyThermal`: steady heat conduction on the thermal domain.

Exact checks (ADR 0180): across layered slabs heated uniformly, the
temperature is piecewise quadratic in depth, so at order 2 it lies in the
space of any mesh that conforms to the layer interfaces, and the solve
reproduces gsim's closed forms to round-off on an unstructured 2D
cross-section slab and on an unstructured 3D brick.
"""

import numpy as np
import pytest
import thermal_cases as tc

import gfem

UM = 1e-6
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
WIDTH = 2.0  # um, x extent of the slab section and the brick
DEPTH = 1.0  # um, y extent of the brick


def _side(p):
    return abs(p[0]) < 1e-9 or abs(p[0] - WIDTH) < 1e-9


def _at_height(z):
    return lambda p: abs(p[1] - z) < 1e-9 and not _side(p)


def _section(path, layers, maxh=0.7):
    """A WIDTH-wide cross-section of ``layers`` (name, thickness) up y."""
    rects, y = [], 0.0
    edges = [(_at_height(0.0), "bottom")]
    for name, t in layers:
        rects.append((0.0, y, WIDTH, y + t, name, None))
        y += t
        edges.append((_at_height(y), "top" if name == layers[-1][0] else f"{name}_top"))
    return tc.section(path, rects, maxh, edges)


def _brick(path, layers, maxh=0.7):
    def at(z):
        return lambda p: abs(p[2] - z) < 1e-9

    faces, z = [(at(0.0), "bottom")], 0.0
    for name, t in layers:
        z += t
        faces.append((at(z), "top" if name == layers[-1][0] else f"{name}_top"))
    return tc.brick(path, (WIDTH, DEPTH), layers, maxh, faces)


def _top_points(mesh, height):
    """Points on the top face (m), spread across it, inside the slab or brick."""
    x = np.array([0.3, 1.0, 1.7]) * UM
    if mesh.section_thickness is None:
        return np.stack([x, np.full(3, 0.5 * UM), np.full(3, height * UM)], axis=1)
    mid = mesh.section_thickness / 2
    return np.stack([x, np.full(3, height * UM), np.full(3, mid)], axis=1)


SERIES = (("lower", 6.0), ("upper", 2.0))
#: gsim's two-layer slab: k 1.4 below, 0.2 above, q = 1e12 W/m^3 above.
SERIES_RISE = 2e6 * 6e-6 / 1.4 + 1e12 * (2e-6) ** 2 / (2 * 0.2)  # 18.571429 K
ROBIN = (("slab", 10.0),)


@pytest.fixture(scope="module", params=["slab", "brick"])
def series_mesh(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("series") / f"{request.param}.msh"
    build = _section if request.param == "slab" else _brick
    return build(path, SERIES)


@pytest.fixture(scope="module", params=["slab", "brick"])
def robin_mesh(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("robin") / f"{request.param}.msh"
    build = _section if request.param == "slab" else _brick
    return build(path, ROBIN, maxh=2.0)


def _series_model(mesh, boundaries=None):
    materials = {
        "lower": gfem.Material(thermal_conductivity=1.4),
        "upper": gfem.Material(thermal_conductivity=0.2),
    }
    if boundaries is None:
        boundaries = [gfem.FixedTemperature("bottom", temperature=300.0)]
    return gfem.Model(mesh, materials=materials, boundaries=boundaries)


def _upper_heat():
    return gfem.HeatSource.uniform("upper", power_density=1e12)


def test_a_series_slab_matches_the_closed_form(series_mesh):
    result = gfem.solve(
        _series_model(series_mesh), gfem.SteadyThermal(_upper_heat()), options=DIRECT
    )
    top = _top_points(series_mesh, 8.0)
    assert pytest.approx(18.571429, abs=1e-6) == SERIES_RISE
    # round-off of the solve (1.1e-15 measured)
    np.testing.assert_allclose(result.rise.at(top), SERIES_RISE, rtol=1e-9)
    np.testing.assert_allclose(
        result.temperature.at(top), 300 + SERIES_RISE, rtol=1e-12
    )
    assert result.temperature.quantity == "T"
    assert result.temperature.unit == "K"
    assert result.rise.quantity == "dT"


def _robin_model(mesh, boundaries):
    materials = {"slab": gfem.Material(thermal_conductivity=1.0)}
    return gfem.Model(mesh, materials=materials, boundaries=boundaries)


ROBIN_HEAT = gfem.HeatSource.uniform("slab", power_density=1e10)


@pytest.mark.parametrize("ambient", [300.0, 290.0])
def test_a_robin_top_matches_the_closed_form(robin_mesh, ambient):
    # gsim's Robin top: L 10 um, k 1, q 1e10, h 1e5, sink below: 0.25 K
    model = _robin_model(
        robin_mesh,
        [
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=ambient),
        ],
    )
    result = gfem.solve(model, gfem.SteadyThermal(ROBIN_HEAT), options=DIRECT)
    top = _top_points(robin_mesh, 10.0)
    # 1.5e-14 measured
    np.testing.assert_allclose(result.rise.at(top), 0.25, rtol=1e-9)
    # the source-free temperature: linear, h (T0 - ambient) = k (300 - T0) / L
    t0 = 300 - (300 - ambient) / 2
    np.testing.assert_allclose(result.temperature.at(top), t0 + 0.25, rtol=1e-12)


def test_a_robin_only_bottom_matches_the_closed_form(robin_mesh):
    # gsim's Robin-only bottom: no fixed temperature, all heat out below
    model = _robin_model(
        robin_mesh, [gfem.Convection("bottom", coefficient=1e5, ambient=300.0)]
    )
    result = gfem.solve(model, gfem.SteadyThermal(ROBIN_HEAT), options=DIRECT)
    np.testing.assert_allclose(
        result.rise.at(_top_points(robin_mesh, 0.0)), 1.0, rtol=1e-9
    )
    np.testing.assert_allclose(
        result.rise.at(_top_points(robin_mesh, 10.0)), 1.5, rtol=1e-9
    )


def test_order_one_is_within_gsims_tolerance(series_mesh):
    options = gfem.SolverOptions(order=1, linear_solver="direct")
    result = gfem.solve(
        _series_model(series_mesh), gfem.SteadyThermal(_upper_heat()), options=options
    )
    top = _top_points(series_mesh, 8.0)
    # gsim's 5e-3 (1.1e-3 on the slab, 3.7e-3 on the brick measured)
    np.testing.assert_allclose(result.rise.at(top), SERIES_RISE, rtol=5e-3)


def test_absolute_minus_rise_is_the_source_free_temperature(robin_mesh):
    model = _robin_model(
        robin_mesh,
        [
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=290.0),
        ],
    )
    hot = gfem.solve(model, gfem.SteadyThermal(ROBIN_HEAT), options=DIRECT)
    cold = gfem.solve(model, gfem.SteadyThermal(), options=DIRECT)
    np.testing.assert_array_equal(np.asarray(cold.rise.coefficients), 0.0)
    np.testing.assert_allclose(
        hot.temperature.coefficients - hot.rise.coefficients,
        cold.temperature.coefficients,
        rtol=1e-13,
        atol=1e-10,  # higher-order coefficients of 0 K, round-off 3e-18
    )
    doubled = gfem.HeatSource.uniform("slab", power_density=2e10)
    twice = gfem.solve(model, gfem.SteadyThermal(doubled), options=DIRECT)
    np.testing.assert_allclose(
        twice.rise.coefficients, 2 * hot.rise.coefficients, rtol=1e-12, atol=1e-14
    )


# --- interior surfaces (thermal probes/06) --------------------------------------

STACK = (("a", 1.0), ("b", 1.0))


@pytest.fixture(scope="module")
def stack_mesh(tmp_path_factory):
    return _section(tmp_path_factory.mktemp("stack") / "stack.msh", STACK, maxh=0.25)


def _stack_model(mesh, boundaries):
    materials = {
        "a": gfem.Material(thermal_conductivity=1.0),
        "b": gfem.Material(thermal_conductivity=2.0),
    }
    return gfem.Model(mesh, materials=materials, boundaries=boundaries)


def _column(mesh, ys):
    mid = mesh.section_thickness / 2
    return np.array([[0.5 * UM, y * UM, mid] for y in ys])


def test_a_heat_flux_on_an_interior_surface_is_a_surface_source(stack_mesh):
    model = _stack_model(
        stack_mesh,
        [
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.HeatFlux("a_top", flux=5e7),
        ],
    )
    result = gfem.solve(model, gfem.SteadyThermal(), options=DIRECT)
    # all of q crosses a: T(mid) = 300 + q L_a / k_a; b above is isothermal
    t = result.temperature.at(_column(stack_mesh, [0.5, 1.0, 1.5, 2.0]))
    np.testing.assert_allclose(t, [325.0, 350.0, 350.0, 350.0], rtol=1e-12)
    # a heat flux is boundary data: no rise, no source power
    assert not np.asarray(result.rise.coefficients).any()
    assert float(result.power) == 0.0


def test_a_fixed_temperature_on_an_interior_surface_pins_both_sides(stack_mesh):
    model = _stack_model(
        stack_mesh,
        [
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.FixedTemperature("a_top", temperature=310.0),
        ],
    )
    result = gfem.solve(model, gfem.SteadyThermal(), options=DIRECT)
    t = result.temperature.at(_column(stack_mesh, [0.5, 1.5, 2.0]))
    np.testing.assert_allclose(t, [305.0, 310.0, 310.0], rtol=1e-12)


# --- sources and totals -----------------------------------------------------------


def test_per_length_totals_on_a_cross_section(stack_mesh):
    model = _stack_model(
        stack_mesh, [gfem.FixedTemperature("bottom", temperature=300.0)]
    )
    heat = gfem.HeatSource.uniform("b", power_per_length=2.0)
    result = gfem.solve(model, gfem.SteadyThermal(heat), options=DIRECT)
    thickness = stack_mesh.section_thickness
    assert float(result.power_per_length) == pytest.approx(2.0, rel=1e-12)
    assert float(result.power) == pytest.approx(2.0 * thickness, rel=1e-12)
    assert "power per length 2 W/m" in repr(result)


def test_a_uniform_source_spreads_its_power_by_volume(series_mesh):
    if series_mesh.section_thickness is not None:
        pytest.skip("power= is a 3D amount")
    heat = gfem.HeatSource.uniform(["lower", "upper"], power=1e-3)
    result = gfem.solve(
        _series_model(series_mesh), gfem.SteadyThermal(heat), options=DIRECT
    )
    volume = WIDTH * DEPTH * 8.0 * UM**3
    assert float(result.power) == pytest.approx(1e-3, rel=1e-12)
    np.testing.assert_allclose(
        result.heat.at(_top_points(series_mesh, 4.0)), 1e-3 / volume, rtol=1e-12
    )
    assert result.heat.quantity == "q"
    assert result.heat.space == "l2"
    with pytest.raises(ValueError, match="cross-section mesh only"):
        _ = result.power_per_length
    assert (
        repr(result) == "SteadyThermalResult(regions=['lower', 'upper'], power 0.001 W)"
    )


def test_sources_superpose(series_mesh):
    model = _series_model(series_mesh)
    one = gfem.HeatSource.uniform("upper", power_density=1e12)
    two = gfem.HeatSource.uniform("lower", power_density=3e11)
    both = gfem.solve(model, gfem.SteadyThermal([one, two]), options=DIRECT)
    apart = [
        gfem.solve(model, gfem.SteadyThermal(s), options=DIRECT) for s in (one, two)
    ]
    np.testing.assert_allclose(
        both.rise.coefficients,
        apart[0].rise.coefficients + apart[1].rise.coefficients,
        rtol=1e-12,
        atol=1e-12 * float(np.abs(both.rise.coefficients).max()),
    )


def test_the_problem_is_a_pytree_of_its_amounts():
    import jax

    problem = gfem.SteadyThermal(heat=[_upper_heat(), _upper_heat()])
    assert len(jax.tree.leaves(problem)) == 2
    assert gfem.SteadyThermal().heat == ()
    with pytest.raises(TypeError, match="HeatSource"):
        gfem.SteadyThermal(heat=1.0)


# --- checks -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gap_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("gap") / "gap.msh"
    return _brick(path, (("lower", 2.0), ("gap", 1.0), ("upper", 2.0)), maxh=1.0)


def test_a_piece_with_no_heat_sink_raises_naming_a_region(gap_mesh):
    k = gfem.Material(eps_r=3.9, thermal_conductivity=1.4)
    model = gfem.Model(
        gap_mesh,
        materials={"lower": k, "gap": gfem.Material(eps_r=1.0), "upper": k},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        excluded={"thermal": ("gap",)},
    )
    with pytest.raises(ValueError, match=r"holds region 'upper'.*no heat sink"):
        gfem.solve(model, gfem.SteadyThermal(), options=DIRECT)


def test_missing_properties_are_listed_together(series_mesh):
    model = gfem.Model(
        series_mesh,
        materials={"lower": gfem.Material(eps_r=3.9)},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        excluded={"em": ("upper",)},
    )
    with pytest.raises(gfem.errors.MissingMaterialPropertyError) as info:
        gfem.solve(model, gfem.SteadyThermal(), options=DIRECT)
    assert info.value.missing == (
        ("lower", "thermal_conductivity"),
        ("upper", "thermal_conductivity"),
    )


def test_heat_on_an_excluded_region_raises(gap_mesh):
    k = gfem.Material(eps_r=3.9, thermal_conductivity=1.4)
    model = gfem.Model(
        gap_mesh,
        materials={"lower": k, "gap": gfem.Material(eps_r=1.0), "upper": k},
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.FixedTemperature("top", temperature=300.0),
        ],
        excluded={"thermal": ("gap",)},
    )
    heat = gfem.HeatSource.uniform("gap", power=1e-3)
    with pytest.raises(ValueError, match="'gap', which is excluded from thermal"):
        gfem.solve(model, gfem.SteadyThermal(heat), options=DIRECT)


# --- linear solvers -----------------------------------------------------------------


def _assert_close(field, reference, rel):
    """``field`` within ``rel`` of ``reference`` relative to its largest coefficient."""
    a = np.asarray(field.coefficients)
    b = np.asarray(reference.coefficients)
    assert np.abs(a - b).max() <= rel * np.abs(b).max()


def test_bddc_matches_direct(series_mesh):
    model = _series_model(series_mesh)
    problem = gfem.SteadyThermal(_upper_heat())
    direct = gfem.solve(model, problem, options=DIRECT)
    bddc = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(
            order=2, linear_solver="bddc_gmres", tolerance=1e-14
        ),
    )
    # 1.8e-16 measured, in the max norm
    _assert_close(bddc.temperature, direct.temperature, 1e-10)


@pytest.mark.petsc
def test_petsc_matches_direct_thermal(series_mesh):
    model = _series_model(series_mesh)
    problem = gfem.SteadyThermal(_upper_heat())
    direct = gfem.solve(model, problem, options=DIRECT)
    petsc = gfem.solve(
        model, problem, options=gfem.SolverOptions(order=2, linear_solver="petsc")
    )
    # in the max norm: the bubbles of the exact quadratic are round-off
    for name in ("temperature", "rise"):
        _assert_close(getattr(petsc, name), getattr(direct, name), 1e-12)


# --- measurements -------------------------------------------------------------------


def test_element_means_sum_to_the_region_mean(series_mesh):
    result = gfem.solve(
        _series_model(series_mesh), gfem.SteadyThermal(_upper_heat()), options=DIRECT
    )
    means = gfem.element_mean(result.rise, "upper")
    tets = series_mesh.elements("upper")
    assert means.shape == (len(tets),)
    p = np.asarray(series_mesh.nodes)[series_mesh.tetrahedra[tets]]
    volume = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6
    weighted = float(np.sum(np.asarray(means) * volume) / volume.sum())
    assert weighted == pytest.approx(
        float(gfem.region_mean(result.rise, "upper")), rel=1e-12
    )
    # the top layer heats most at its top, and every element mean lies between
    assert float(np.min(means)) > 2e6 * 6e-6 / 1.4 - 1e-9
    assert float(np.max(means)) < SERIES_RISE


# --- temperature into an electromagnetic solve ------------------------------------


def test_a_driven_solve_runs_on_a_permittivity_from_the_temperature(tmp_path):
    # a full 3D eps(T) driven solve needs no thermal code: the rise per
    # element sets a MaterialField permittivity (ADR 0160)
    import structured_meshes as sm

    path = tmp_path / "plate.msh"
    layers = (("low", 0.5), ("high", 0.5))
    sm.write_msh22(path, sm.strip_line(2.0, 1.0, 1.0, (2, 1, 1), layers=layers))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    low = gfem.Material(eps_r=2.0, thermal_conductivity=1.4)
    high = gfem.Material(eps_r=5.0, thermal_conductivity=1.4)
    boundaries = [gfem.PEC("bottom"), gfem.PEC("top")]
    ports = [
        gfem.LumpedPort("p1", "port1", "+z"),
        gfem.LumpedPort("p2", "port2", "+z"),
    ]
    thermal = gfem.Model(
        mesh,
        materials={"low": low, "high": high},
        boundaries=[*boundaries, gfem.FixedTemperature("bottom", temperature=300.0)],
        ports=ports,
    )
    heat = gfem.HeatSource.uniform("high", power=1e-4)
    rise = gfem.solve(thermal, gfem.SteadyThermal(heat), options=DIRECT).rise
    n0, dn_dt = np.sqrt(5.0), 1.8e-4
    eps = (n0 + dn_dt * gfem.element_mean(rise, "high")) ** 2
    assert float(eps.min()) > 5.0
    optical = gfem.Model(
        mesh,
        materials={"low": low, "high": gfem.MaterialField(eps_r=eps, region="high")},
        boundaries=boundaries,
        ports=ports,
    )
    driven = gfem.solve(optical, gfem.Driven(frequency=20e12), options=DIRECT)
    s21 = np.asarray(driven.s["p2", "p1"])
    assert np.isfinite(s21).all()
    assert 0 < np.abs(s21).max() <= 1 + 1e-9
