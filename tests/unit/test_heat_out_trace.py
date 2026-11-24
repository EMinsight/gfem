"""`gfem.Trace.heat_out` and `heat_out_per_length`: the heat a transient sends out.

The trace records the heat that has left a side since t = 0 from the
stepping scheme's own balance (the residual with the capacity term, the
time integral of the temperature taken with the scheme's stage
quadrature), so the energy balance closes to round-off at every grid time,
whatever the scheme.

The case is test_heat_flow's rectangle: a substrate below y = 1.5 um, a
cladding above holding a heater, a fixed-temperature bottom, convection on
the top and the right, and a heat flux into the left.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import thermal_cases as tc

import gfem
from gfem._engine import thermal

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
]
SINKS = ("bottom", "top", "right")
#: rho c_p per region, J/(m^3 K)
RHO = {"sub": 2330.0, "clad": 2200.0, "heater": 7900.0}
CP = {"sub": 700.0, "clad": 740.0, "heater": 500.0}
K = {"sub": 1.4, "clad": 0.3, "heater": 20.0}
FLUX = 5e6  # W/m^2 into the left, 2 um tall
POWER = 200.0  # W/m (3D: times the extrusion length)
LENGTH = 1.0  # um
STEPS = 12
DURATION = 2e-5


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    path = tmp_path_factory.mktemp("heat_out") / "section.msh"
    return tc.section(path, RECTS, 0.4, EDGES)


@pytest.fixture(scope="module")
def brick(tmp_path_factory):
    path = tmp_path_factory.mktemp("heat_out") / "brick.msh"
    return tc.extruded(path, RECTS, 0.5, EDGES, length=LENGTH, layers=1)


def _model(mesh, k_sub=K["sub"], rho_sub=RHO["sub"], cp_sub=CP["sub"]):
    n = len(mesh.elements("sub"))
    materials = {
        r: gfem.Material(
            thermal_conductivity=K[r], mass_density=RHO[r], specific_heat=CP[r]
        )
        for r in ("clad", "heater")
    }
    materials["sub"] = gfem.MaterialField(
        thermal_conductivity=jnp.full(n, k_sub),
        mass_density=jnp.full(n, rho_sub),
        specific_heat=jnp.full(n, cp_sub),
    )
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e5, ambient=293.0),
            gfem.Convection("right", coefficient=1e5, ambient=293.0),
            gfem.HeatFlux("left", flux=FLUX),
        ],
    )


def _per_length(mesh):
    return mesh.section_thickness is not None


def _source(mesh, amount=1.0):
    if _per_length(mesh):
        return gfem.HeatSource.uniform("heater", power_per_length=amount * POWER)
    return gfem.HeatSource.uniform("heater", power=amount * POWER * LENGTH * 1e-6)


def _out(mesh, side):
    if _per_length(mesh):
        return gfem.Trace.heat_out_per_length(side)
    return gfem.Trace.heat_out(side)


#: on, a lower plateau, then off
def _waveform(scale=1.0):
    return [(0.0, scale), (DURATION * 4 / STEPS, 0.3), (DURATION * 8 / STEPS, 0.0)]


def _volumes(mesh, region):
    v = np.asarray(mesh.nodes)[mesh.tetrahedra[mesh.elements(region)]]
    det = np.einsum(
        "ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0]
    )
    return float(np.abs(det).sum() / 6.0)


SCHEMES = {
    "tr_bdf2": gfem.SolverOptions(order=2, linear_solver="direct"),
    "crank_nicolson": gfem.SolverOptions(
        order=2, linear_solver="direct", time_scheme="theta", theta=0.5
    ),
    "backward_euler": gfem.SolverOptions(
        order=2, linear_solver="direct", time_scheme="theta", theta=1.0
    ),
}


@pytest.mark.parametrize(
    ("kind", "scheme"),
    [("section", s) for s in SCHEMES] + [("brick", "tr_bdf2")],
)
def test_the_sinks_heat_out_and_the_stored_heat_add_to_the_source_energy(
    kind, scheme, request
):
    mesh = request.getfixturevalue(kind)
    traces = {"sinks": _out(mesh, SINKS)}
    traces |= {r: gfem.Trace.mean(r) for r in RHO}
    problem = gfem.TransientThermal(
        _source(mesh),
        duration=DURATION,
        steps=STEPS,
        waveform=_waveform(),
        traces=traces,
    )
    result = gfem.solve(_model(mesh), problem, options=SCHEMES[scheme])
    out = np.asarray(result.traces["sinks"])
    assert out.shape == (STEPS + 1,)
    assert out[0] == 0.0
    # the stored heat from the exact region means of the temperature
    per = mesh.section_thickness if _per_length(mesh) else 1.0
    heat = {r: RHO[r] * CP[r] * _volumes(mesh, r) / per for r in RHO}
    stored = sum(
        heat[r] * (np.asarray(result.traces[r]) - float(result.traces[r][0]))
        for r in RHO
    )
    t = np.asarray(result.time)
    dt = DURATION / STEPS
    on = np.array([1.0] * 4 + [0.3] * 4 + [0.0] * 4)
    source = np.concatenate([[0.0], np.cumsum(on * dt)])
    scale = 1.0 if _per_length(mesh) else LENGTH * 1e-6
    flux = FLUX * 2e-6 * scale  # W (per length on the section)
    energy = POWER * scale * source + flux * t
    np.testing.assert_allclose(out + stored, energy, rtol=0, atol=1e-12 * energy[-1])
    assert abs(out[-1]) > 0.1 * energy[-1]


def test_the_late_step_flow_tends_to_the_steady_heat_flow(section):
    model = _model(section)
    options = SCHEMES["tr_bdf2"]
    side = lambda c: c[:, 1] > 1.03e-6  # noqa: E731 - straddles convection and flux
    steady = gfem.solve(model, gfem.SteadyThermal(_source(section)), options=options)
    expected = float(gfem.heat_flow_per_length(steady, side, model=model))
    problem = gfem.TransientThermal(
        _source(section),
        duration=2e-4,
        steps=60,
        traces={"cut": gfem.Trace.heat_out_per_length(side)},
    )
    result = gfem.solve(model, problem, options=options)
    flow = np.diff(np.asarray(result.traces["cut"])) / (2e-4 / 60)
    assert flow[-1] == pytest.approx(expected, rel=1e-6)
    assert abs(flow[0] - expected) > 1e-2 * abs(expected)


def test_each_mesh_kind_names_its_twin(section, brick):
    for mesh, trace, twin in (
        (section, gfem.Trace.heat_out("bottom"), r"Trace\.heat_out_per_length"),
        (brick, gfem.Trace.heat_out_per_length("bottom"), r"Trace\.heat_out\b"),
    ):
        problem = gfem.TransientThermal(
            _source(mesh), duration=1e-9, steps=1, traces={"out": trace}
        )
        with pytest.raises(ValueError, match=twin):
            gfem.solve(_model(mesh), problem, options=SCHEMES["tr_bdf2"])


def test_a_side_that_does_not_fit_raises_at_solve_time(section):
    problem = gfem.TransientThermal(
        _source(section),
        duration=1e-9,
        steps=1,
        traces={"out": gfem.Trace.heat_out_per_length(("sub", "bottom"))},
    )
    with pytest.raises(ValueError, match="mixes volume regions"):
        gfem.solve(_model(section), problem, options=SCHEMES["tr_bdf2"])


def test_a_heat_out_trace_in_a_refined_objective_raises(section, monkeypatch):
    from gfem import _objective

    monkeypatch.setattr(_objective, "recording", object)
    problem = gfem.TransientThermal(
        _source(section),
        duration=1e-9,
        steps=1,
        traces={"out": gfem.Trace.heat_out_per_length("bottom")},
    )
    with pytest.raises(
        gfem.MeshRefinementUnsupportedError, match="no error estimate"
    ) as caught:
        gfem.solve(_model(section), problem, options=SCHEMES["tr_bdf2"])
    assert caught.value.feature == "Trace.heat_out_per_length"


def test_a_side_of_the_wrong_type_raises_when_built():
    with pytest.raises(TypeError, match=r"Trace\.heat_out"):
        gfem.Trace.heat_out(3)
    assert gfem.Trace.heat_out("bottom").unit == "J"
    assert gfem.Trace.heat_out_per_length("bottom").unit == "J/m"


# --- gradients ----------------------------------------------------------------------

#: the inputs, at their values: k, rho and c_p of the substrate, the source
#: amount and the first waveform scale
X0 = np.array([K["sub"], RHO["sub"], CP["sub"], 1.0, 1.0])
#: relative finite-difference steps
H = 1e-3


def _objective(mesh, options):
    weights = np.linspace(1.0, 2.0, STEPS + 1)

    def objective(x):
        problem = gfem.TransientThermal(
            _source(mesh, x[3]),
            duration=DURATION,
            steps=STEPS,
            waveform=_waveform(x[4]),
            initial_temperature=296.0,
            traces={
                "sink": gfem.Trace.heat_out_per_length("bottom"),
                "cut": gfem.Trace.heat_out_per_length(lambda c: c[:, 1] > 1.03e-6),
            },
        )
        result = gfem.solve(_model(mesh, x[0], x[1], x[2]), problem, options=options)
        return jnp.sum(weights * (result.traces["sink"] + 2.0 * result.traces["cut"]))

    return objective


def _difference(f, i):
    """Fourth-order central difference along input ``i``."""
    h = H * X0[i]

    def at(step):
        y = np.array(X0)
        y[i] += step
        return float(f(jnp.asarray(y)))

    return (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)


@pytest.mark.parametrize("scheme", ["tr_bdf2", "theta"])
def test_the_gradient_matches_finite_differences_in_both_layouts(
    section, scheme, monkeypatch
):
    options = gfem.SolverOptions(
        order=2,
        linear_solver="direct",
        time_scheme=scheme,
        theta=0.5 if scheme == "theta" else 1.0,
    )
    f = _objective(section, options)
    fd = np.array([_difference(f, i) for i in range(len(X0))])
    full = np.asarray(jax.grad(f)(jnp.asarray(X0)))
    monkeypatch.setattr(
        thermal.TransientThermalRule, "_trajectory", lambda *_: "checkpoints"
    )
    checkpointed = np.asarray(jax.grad(f)(jnp.asarray(X0)))
    assert np.all(np.abs(fd) > 0)
    np.testing.assert_allclose(full, fd, rtol=1e-6)
    np.testing.assert_allclose(checkpointed, full, rtol=1e-12)
