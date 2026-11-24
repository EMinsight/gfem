"""`gfem.heat_flow` and `gfem.heat_flow_per_length`: heat through a surface.

The flow is read from the residual of the discrete heat equation, tested
with the vertex hat functions, so the balance identities hold to round-off
on any mesh: the flows out of every boundary surface sum to the source
power, and a cut's flow is minus its complement's.

The case is a 4 x 2 um rectangle (a substrate below y = 1.5, a cladding
above holding a heater) with a fixed-temperature bottom, convection on the
top and the right (each touching the fixed bottom or the other), and a heat
flux into the left, as a 3D extrusion and as a cross-section slab.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import thermal_cases as tc
from finite_differences import assert_gradient_matches

import gfem

DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
LENGTH = 1.0  # um, the extrusion length
#: the substrate, the heater and the cladding around it (um)
RECTS = [
    (0.0, 0.0, 4.0, 1.5, "sub", None),
    (1.0, 1.5, 2.0, 1.8, "heater", None),
    (0.0, 1.5, 1.0, 2.0, "clad", None),
    (2.0, 1.5, 4.0, 2.0, "clad", None),
    (1.0, 1.8, 2.0, 2.0, "clad", None),
]
EDGES = [
    (lambda p: abs(p[1]) < 1e-9, "bottom"),
    (lambda p: abs(p[1] - 2.0) < 1e-9, "top"),
    (lambda p: abs(p[0] - 4.0) < 1e-9, "right"),
    (lambda p: abs(p[0]) < 1e-9, "left"),
    (lambda p: abs(p[1] - 1.5) < 1e-9, "interface"),
]
SURFACES = ("bottom", "top", "right", "left")


@pytest.fixture(scope="module")
def brick(tmp_path_factory):
    path = tmp_path_factory.mktemp("heat_flow") / "brick.msh"
    return tc.extruded(path, RECTS, 0.35, EDGES, length=LENGTH, layers=2)


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    path = tmp_path_factory.mktemp("heat_flow") / "section.msh"
    return tc.section(path, RECTS, 0.35, EDGES)


CAPACITY = {"mass_density": 2200.0, "specific_heat": 700.0}


def _model(mesh, k_sub=1.4, boundaries=None):
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.MaterialField(
                thermal_conductivity=jnp.full(len(mesh.elements("sub")), k_sub),
                mass_density=jnp.full(len(mesh.elements("sub")), 2200.0),
                specific_heat=jnp.full(len(mesh.elements("sub")), 700.0),
            ),
            "clad": gfem.Material(thermal_conductivity=0.3, **CAPACITY),
            "heater": gfem.Material(thermal_conductivity=20.0, **CAPACITY),
        },
        boundaries=boundaries
        or [
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=293.0),
            gfem.Convection("right", coefficient=1e5, ambient=293.0),
            gfem.HeatFlux("left", flux=5e6),
        ],
    )


def _heat(mesh):
    if mesh.section_thickness is None:
        return gfem.HeatSource.uniform("heater", power=2e-4)
    return gfem.HeatSource.uniform("heater", power_per_length=200.0)


@pytest.fixture(scope="module")
def solved(brick):
    model = _model(brick)
    return model, gfem.solve(model, gfem.SteadyThermal(_heat(brick)), options=DIRECT)


def test_the_flows_out_of_every_surface_sum_to_the_source_power(solved):
    model, result = solved
    flows = {s: float(gfem.heat_flow(result, s, model=model)) for s in SURFACES}
    power = float(result.power)
    assert power == pytest.approx(2e-4, rel=1e-12)
    # 1e-14 relative measured; the flux puts heat in, so its flow is negative
    assert sum(flows.values()) == pytest.approx(power, rel=1e-12)
    assert flows["left"] == pytest.approx(-5e6 * 2e-6 * LENGTH * 1e-6, rel=1e-12)
    union = float(gfem.heat_flow(result, SURFACES, model=model))
    assert union == pytest.approx(power, rel=1e-12)


@pytest.fixture(scope="module")
def solved_section(section):
    model = _model(section)
    return model, gfem.solve(model, gfem.SteadyThermal(_heat(section)), options=DIRECT)


def test_a_section_balances_per_length(solved_section):
    model, result = solved_section
    flows = [float(gfem.heat_flow_per_length(result, s, model=model)) for s in SURFACES]
    assert float(result.power_per_length) == pytest.approx(200.0, rel=1e-12)
    assert sum(flows) == pytest.approx(200.0, rel=1e-12)
    # the end faces are insulated
    assert float(gfem.heat_flow_per_length(result, "front", model=model)) == 0.0
    assert flows[3] == pytest.approx(-5e6 * 2e-6, rel=1e-12)


CUTS = {
    # ends on the fixed bottom: the reaction of the rim nodes is split
    "left half": lambda c: c[:, 0] < 2e-6,
    # straddles the convection right and the flux left by staircase faces
    "upper part": lambda c: c[:, 1] > 1.03e-6,
    "named": ("clad", "heater"),
}


@pytest.mark.parametrize("cut", CUTS)
def test_a_cut_is_minus_its_complement(solved, cut):
    model, result = solved
    side = CUTS[cut]
    complement = (lambda c: ~side(c)) if callable(side) else ("sub",)
    out = float(gfem.heat_flow(result, side, model=model))
    back = float(gfem.heat_flow(result, complement, model=model))
    # 1e-14 relative to the power measured
    assert out + back == pytest.approx(0.0, abs=1e-12 * float(result.power))
    assert abs(out) > 1e-3 * float(result.power)


# --- the slabs of tests/unit/test_steady_thermal.py: hand fluxes -----------------

WIDTH, DEPTH = 2.0, 1.0  # um


def _layers(path, kind, layers, maxh):
    """The slab section or the brick of ``layers`` (name, thickness) up the stack."""
    axis = 1 if kind == "slab" else 2

    def at(z):
        return lambda p: (
            abs(p[axis] - z) < 1e-9 and (kind == "brick" or 1e-9 < p[0] < WIDTH - 1e-9)
        )

    named, z = [(at(0.0), "bottom")], 0.0
    for name, t in layers:
        z += t
        named.append((at(z), "top" if name == layers[-1][0] else f"{name}_top"))
    if kind == "brick":
        return tc.brick(path, (WIDTH, DEPTH), layers, maxh, named)
    rects, y = [], 0.0
    for name, t in layers:
        rects.append((0.0, y, WIDTH, y + t, name, None))
        y += t
    return tc.section(path, rects, maxh, named)


def _flow(result, side, model):
    """W/m^2 of the slab's or brick's cross-section."""
    if result.temperature.mesh.section_thickness is None:
        return float(gfem.heat_flow(result, side, model=model)) / (
            WIDTH * DEPTH * 1e-12
        )
    return float(gfem.heat_flow_per_length(result, side, model=model)) / (WIDTH * 1e-6)


@pytest.mark.parametrize("kind", ["slab", "brick"])
def test_a_series_slab_matches_the_hand_flux(tmp_path, kind):
    mesh = _layers(tmp_path / "series.msh", kind, (("lower", 6.0), ("upper", 2.0)), 0.7)
    model = gfem.Model(
        mesh,
        materials={
            "lower": gfem.Material(thermal_conductivity=1.4),
            "upper": gfem.Material(thermal_conductivity=0.2),
        },
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    heat = gfem.HeatSource.uniform("upper", power_density=1e12)
    result = gfem.solve(model, gfem.SteadyThermal(heat), options=DIRECT)
    flux = 1e12 * 2e-6  # W/m^2: all of the upper layer's heat, down
    assert _flow(result, "bottom", model) == pytest.approx(flux, rel=1e-12)
    assert _flow(result, "upper", model) == pytest.approx(flux, rel=1e-12)
    assert _flow(result, "lower", model) == pytest.approx(-flux, rel=1e-12)
    assert _flow(result, "top", model) == 0.0


@pytest.mark.parametrize("kind", ["slab", "brick"])
@pytest.mark.parametrize("ambient", [300.0, 290.0])
def test_a_robin_slab_matches_the_hand_flux(tmp_path, kind, ambient):
    # L 10 um, k 1, q 1e10 W/m^3, h 1e5 W/(m^2 K) on top, 300 K below:
    # T = 300 + a x - q x^2 / (2 k), a (k + h L) = q L - h (300 - Ta) + h q L^2 / (2 k)
    mesh = _layers(tmp_path / "robin.msh", kind, (("slab", 10.0),), 2.0)
    model = gfem.Model(
        mesh,
        materials={"slab": gfem.Material(thermal_conductivity=1.0)},
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=ambient),
        ],
    )
    heat = gfem.HeatSource.uniform("slab", power_density=1e10)
    result = gfem.solve(model, gfem.SteadyThermal(heat), options=DIRECT)
    bottom = {300.0: 7.5e4, 290.0: -4.25e5}[ambient]  # k a
    # the threaded direct solve's round-off varies between runs by ~1e-12
    assert _flow(result, "bottom", model) == pytest.approx(bottom, rel=1e-11)
    assert _flow(result, "top", model) == pytest.approx(1e5 - bottom, rel=1e-11)


# --- gradients ----------------------------------------------------------------------


def _flow_of_k(mesh, side, k_sub):
    model = _model(mesh, k_sub)
    result = gfem.solve(model, gfem.SteadyThermal(_heat(mesh)), options=DIRECT)
    return gfem.heat_flow(result, side, model=model)


def test_a_side_with_no_sink_measures_its_source_power_and_no_conductivity(brick):
    # the heater's closure holds no fixed node and no convection face
    value, grad = jax.value_and_grad(lambda k: _flow_of_k(brick, "heater", k))(1.4)
    assert float(value) == pytest.approx(2e-4, rel=1e-12)
    assert float(grad) == 0.0


@pytest.mark.parametrize("side", ["bottom", "top", ("clad", "heater")])
def test_the_conductivity_gradient_matches_finite_differences(brick, side):
    grad = float(jax.grad(lambda k: _flow_of_k(brick, side, k))(1.4))
    h = 1e-4
    fd = (
        float(_flow_of_k(brick, side, 1.4 + h))
        - float(_flow_of_k(brick, side, 1.4 - h))
    ) / (2 * h)
    # central differences of a smooth function: 1e-9 measured
    assert grad == pytest.approx(fd, rel=1e-6)
    assert abs(grad) > 1e-3 * abs(float(_flow_of_k(brick, side, 1.4)))


def _flow_of_values(mesh, side, x):
    t, h, amb, q = x * jnp.asarray([300.0, 1e5, 293.0, 5e6])
    model = _model(
        mesh,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=t),
            gfem.Convection("top", coefficient=h, ambient=amb),
            gfem.Convection("right", coefficient=0.5 * h, ambient=amb + 3.0),
            gfem.HeatFlux("left", flux=q),
        ],
    )
    result = gfem.solve(model, gfem.SteadyThermal(_heat(mesh)), options=DIRECT)
    return gfem.heat_flow(result, side, model=model)


@pytest.mark.parametrize("side", ["bottom", "top", "right", "left", ("clad", "heater")])
def test_the_four_boundary_values_match_finite_differences(brick, side):
    x = jnp.ones(4)

    def objective(v):
        return _flow_of_values(brick, side, v) / 1e-4

    grad = np.asarray(jax.grad(objective)(x))
    for i in range(4):
        d = np.eye(4)[i]
        assert_gradient_matches(objective, x, [d], step=1e-2, grad=grad)


def test_a_flux_traced_at_zero_still_has_a_flow_gradient(brick):
    def flow(q):
        model = _model(
            brick,
            boundaries=[
                gfem.FixedTemperature("bottom", temperature=300.0),
                gfem.Convection("top", coefficient=1e5, ambient=293.0),
                gfem.HeatFlux("left", flux=q),
            ],
        )
        result = gfem.solve(model, gfem.SteadyThermal(_heat(brick)), options=DIRECT)
        return gfem.heat_flow(result, "left", model=model)

    # the flux is the left's flow: dQ/dq is minus the face's area
    assert float(jax.grad(flow)(0.0)) < 0.0


# --- refusals -----------------------------------------------------------------------


def _shared_fixed(mesh):
    # the left wall touches the bottom at the corner nodes
    return _model(
        mesh,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.FixedTemperature("left", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=293.0),
        ],
    )


def test_one_of_two_fixed_surfaces_sharing_nodes_raises_and_their_union_measures(
    brick,
):
    model = _shared_fixed(brick)
    result = gfem.solve(model, gfem.SteadyThermal(_heat(brick)), options=DIRECT)
    with pytest.raises(ValueError, match=r"'bottom' and 'left' share.*union"):
        gfem.heat_flow(result, "bottom", model=model)
    union = gfem.heat_flow(result, ("bottom", "left"), model=model)
    top = gfem.heat_flow(result, "top", model=model)
    assert float(union + top) == pytest.approx(float(result.power), rel=1e-12)


SIDE_REFUSALS = {
    "mixed": (("sub", "top"), ValueError, r"mixes volume regions.*either"),
    "interior surface": ("interface", ValueError, r"inside the thermal domain.*volume"),
    "empty": (lambda c: c[:, 0] < -1.0, ValueError, r"selects no element.*sink"),
    "everything": (lambda c: c[:, 0] > -1.0, ValueError, r"every element.*sink"),
    "predicate shape": (lambda c: c[:, 0], ValueError, r"one boolean"),
    "unknown": ("nowhere", gfem.RegionNotFoundError, "nowhere"),
    "not a side": (3, TypeError, r"side must be"),
}


@pytest.mark.parametrize("case", SIDE_REFUSALS)
def test_a_side_that_does_not_fit_raises_with_its_remedy(solved, case):
    model, result = solved
    side, error, match = SIDE_REFUSALS[case]
    with pytest.raises(error, match=match):
        gfem.heat_flow(result, side, model=model)


def test_each_mesh_kind_names_its_twin(solved, solved_section):
    model, result = solved
    with pytest.raises(ValueError, match=r"3D mesh: use gfem\.heat_flow \(W\)"):
        gfem.heat_flow_per_length(result, "bottom", model=model)
    model, result = solved_section
    with pytest.raises(ValueError, match=r"cross-section.*gfem\.heat_flow_per_length"):
        gfem.heat_flow(result, "bottom", model=model)


def test_a_result_of_another_model_raises(solved, solved_section):
    _, result = solved
    other, _ = solved_section
    with pytest.raises(ValueError, match="not solved on this model"):
        gfem.heat_flow(result, "bottom", model=other)
    model, _ = solved
    excluded = model.replace(excluded={"thermal": ("heater",)})
    with pytest.raises(ValueError, match="not solved on this model"):
        gfem.heat_flow(result, "bottom", model=excluded)


def test_a_transient_result_raises_naming_the_heat_out_trace(section):
    model = _model(section)
    problem = gfem.TransientThermal(_heat(section), duration=1e-9, steps=1)
    result = gfem.solve(model, problem, options=DIRECT)
    with pytest.raises(TypeError, match=r"Trace\.heat_out"):
        gfem.heat_flow_per_length(result, "bottom", model=model)


def test_fields_across_ranks_raise(solved, monkeypatch):
    model, result = solved
    monkeypatch.setattr(gfem.Field, "_communicator", property(lambda _: object()))
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.heat_flow(result, "bottom", model=model)
    assert caught.value.feature == "thermal"


def test_a_refined_objective_raises(solved, monkeypatch):
    from gfem import measurements

    model, result = solved
    monkeypatch.setattr(measurements._objective, "recording", object)
    with pytest.raises(gfem.MeshRefinementUnsupportedError, match="no error estimate"):
        gfem.heat_flow(result, "bottom", model=model)


def test_the_node_coordinates_raise_under_grad(solved):
    model, result = solved

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        return gfem.heat_flow(result, "bottom", model=model.replace(mesh=mesh))

    with pytest.raises(gfem.NonDifferentiableError, match="mesh node coordinates"):
        jax.grad(objective)(jnp.asarray(model.mesh.nodes))


# --- the residual's face load: what face heat will read ---------------------------


@pytest.mark.parametrize("side", [("clad", "heater"), ("bottom",)])
def test_a_face_density_weighs_like_the_heat_flux_it_equals(solved, side):
    # a face heat of 5e6 W/m^2 on the left is the left's heat flux, so the
    # balance with the flux as a face density is the balance with the flux
    from gfem._engine import heat_flow as engine
    from gfem._heat_flow import steady_rule

    model, result = solved
    bare = model.replace(boundaries=model.boundaries[:3])
    t = np.asarray(result.temperature.coefficients)
    density = np.zeros(len(model.mesh.tetrahedra))
    density[model.mesh.elements("heater")] = np.asarray(result.heat.coefficients).max()
    face = np.zeros(len(model.mesh.triangles))
    face[model.mesh.elements("left")] = 5e6
    flows = []
    for m, extra in ((model, None), (bare, face)):
        rule = steady_rule(m, 2)
        weights = engine.side(rule, m, side)
        flows.append(engine.balance(rule, m, weights).flow(t, density, extra))
    assert flows[1] == pytest.approx(flows[0], rel=1e-13)


# --- sheet heat --------------------------------------------------------------------


def _sheet(mesh, scale=1.0):
    """Face heat on the fixed bottom and the convection top, beside the heater's."""
    face = np.zeros(len(mesh.triangles))
    face[mesh.elements("bottom")] = 3e7
    face[mesh.elements("top")] = 2e7
    return gfem.HeatSource(
        regions=(),
        _kind="electromagnetic",
        _density=jnp.zeros(len(mesh.tetrahedra)),
        _face_density=jnp.asarray(face) * scale,
    )


@pytest.fixture(scope="module")
def sheet_solved(brick):
    model = _model(brick)
    problem = gfem.SteadyThermal((_heat(brick), _sheet(brick)))
    return model, gfem.solve(model, problem, options=DIRECT)


def test_sheet_heat_on_a_fixed_face_is_in_the_balance(sheet_solved):
    model, result = sheet_solved
    # 4 um x 1 um faces: 3e7 and 2e7 W/m^2 on 4e-12 m^2 each
    power = float(result.power)
    assert power == pytest.approx(2e-4 + 2e-4, rel=1e-12)
    flows = sum(float(gfem.heat_flow(result, s, model=model)) for s in SURFACES)
    assert flows == pytest.approx(power, rel=1e-12)
    out = float(gfem.heat_flow(result, CUTS["left half"], model=model))
    back = float(gfem.heat_flow(result, lambda c: ~CUTS["left half"](c), model=model))
    assert out + back == pytest.approx(0.0, abs=1e-12 * power)


def test_the_sheet_heat_gradient_matches_finite_differences(brick):
    model = _model(brick)

    def flow(scale):
        problem = gfem.SteadyThermal((_heat(brick), _sheet(brick, scale)))
        result = gfem.solve(model, problem, options=DIRECT)
        return gfem.heat_flow(result, "bottom", model=model)

    grad = float(jax.grad(flow)(1.0))
    fd = (float(flow(1.0 + 1e-3)) - float(flow(1.0 - 1e-3))) / 2e-3
    assert grad == pytest.approx(fd, rel=1e-6)


INTERFACE_HEAT = 4e7  # W/m^2 on the 4 um x 1 um interface between sub and the rest


@pytest.mark.parametrize("how", ["sheet", "flux"])
def test_heat_on_a_cut_belongs_to_neither_side(brick, how):
    # a face belongs to a side when every element beside it is in the side,
    # so heat put on the interface itself leaves through both sides at once:
    # flow(V) + flow(complement) = -P on the cut
    model = _model(brick)
    heat = [_heat(brick)]
    if how == "sheet":
        face = np.zeros(len(brick.triangles))
        face[brick.elements("interface")] = INTERFACE_HEAT
        heat.append(
            gfem.HeatSource(
                regions=(),
                _kind="electromagnetic",
                _density=jnp.zeros(len(brick.tetrahedra)),
                _face_density=jnp.asarray(face),
            )
        )
    else:
        flux = gfem.HeatFlux("interface", flux=INTERFACE_HEAT)
        model = model.replace(boundaries=[*model.boundaries, flux])
    result = gfem.solve(model, gfem.SteadyThermal(tuple(heat)), options=DIRECT)
    on_cut = INTERFACE_HEAT * 4e-6 * LENGTH * 1e-6
    out = float(gfem.heat_flow(result, ("clad", "heater"), model=model))
    back = float(gfem.heat_flow(result, ("sub",), model=model))
    # 1.4e-14 relative to the cut heat measured
    assert out + back == pytest.approx(-on_cut, rel=1e-12)
    assert abs(out) > 1e-2 * on_cut
