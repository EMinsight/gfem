"""Gradients of thermal solves: the transient adjoint's layouts and leaves.

The finite-difference cells are in tests/verification/test_gradient_matrix.py;
these tests pin what the cells cannot see: the trajectory layout a
differentiated transient forward keeps (ADR 0179), the checkpointed reverse
sweep against the full one, and the traces' own leaves (a section's
thermo-optic weights, read by a JAX reverse pass inside the host forward)
under `jax.grad`.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem
from gfem._engine import thermal
from gfem._engine.sensitivity import LOAD_KIND

L, K, RHO, CP = 4.0, 1.4, 2200.0, 700.0
STEPS = 9
DURATION = 2e-6
EDGES = [
    (lambda p: abs(p[1]) < 1e-9, "bottom"),
    (lambda p: abs(p[1] - L) < 1e-9, "top"),
]


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("gradients") / "slab.msh"
    rects = [(0.0, 0.0, 3.0, 3.0, "clad", None), (1.0, 3.0, 2.0, L, "heater", None)]
    return tc.section(path, rects, 0.6, EDGES)


def _model(mesh, k=None):
    def material(region):
        n = len(mesh.elements(region))
        values = jnp.full(n, K) if k is None else k[mesh.elements(region)]
        return gfem.MaterialField(
            thermal_conductivity=values,
            mass_density=jnp.full(n, RHO),
            specific_heat=jnp.full(n, CP),
        )

    return gfem.Model(
        mesh,
        materials={"clad": material("clad"), "heater": material("heater")},
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=1e4, ambient=290.0),
        ],
    )


def _problem(mesh, scale=1.0):
    return gfem.TransientThermal(
        gfem.HeatSource.uniform("heater", power_per_length=1.0),
        duration=DURATION,
        steps=STEPS,
        waveform=[(0.0, scale), (DURATION * 5 / STEPS, 0.3)],
        initial_temperature=295.0,
        traces={"heater": gfem.Trace.mean("heater")},
        keep_fields=True,
    )


def _objective(mesh, options):
    weights = np.linspace(1.0, 2.0, STEPS + 1)

    def objective(k):
        result = gfem.solve(_model(mesh, k), _problem(mesh), options=options)
        field = result.temperature(DURATION * 3 / STEPS).coefficients
        return jnp.sum(result.traces["heater"] * weights) + jnp.sum(field[:40])

    return objective


def _k(mesh):
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(1.0, 2.0, len(mesh.tetrahedra)))


@pytest.mark.parametrize("scheme", ["tr_bdf2", "theta"])
def test_the_checkpointed_sweep_equals_the_full_one(mesh, scheme, monkeypatch):
    options = gfem.SolverOptions(
        order=2, time_scheme=scheme, theta=0.5 if scheme == "theta" else 1.0
    )
    k = _k(mesh)
    full = jax.grad(_objective(mesh, options))(k)
    # a forward that decides on checkpoints, and a backward that reruns its
    # forward (retain_memory=0 keeps no entry)
    rerun = gfem.SolverOptions(
        order=2,
        time_scheme=scheme,
        theta=0.5 if scheme == "theta" else 1.0,
        retain_memory=0,
    )
    from_rerun = jax.grad(_objective(mesh, rerun))(k)
    monkeypatch.setattr(
        thermal.TransientThermalRule, "_trajectory", lambda *_: "checkpoints"
    )
    checkpointed = jax.grad(_objective(mesh, options))(k)
    assert np.abs(np.asarray(full)).max() > 0
    for other in (checkpointed, from_rerun):
        error = np.linalg.norm(other - full) / np.linalg.norm(full)
        assert error <= 1e-13, error


def _kept_entry(mesh, wanted, options=None):
    options = gfem.SolverOptions(order=2) if options is None else options
    problem = _problem(mesh)
    model = _model(mesh)
    rule = thermal.TransientThermalRule(model, problem, options)
    rule.wanted_kinds = frozenset(wanted)
    _, entry = rule.forward(model, keep=True)
    return rule, entry


def test_only_source_leaves_keep_no_checkpoint_or_trajectory(mesh):
    _, entry = _kept_entry(mesh, {LOAD_KIND, ("problem", ".scales")})
    try:
        assert entry.nbytes == entry.handle.nbytes
    finally:
        entry.free()
    k = ("material", "clad", "thermal_conductivity")
    rule, entry = _kept_entry(mesh, {LOAD_KIND, k})
    try:
        ndof = rule._ndof()
        assert entry.fields["states"].shape == (STEPS + 1, ndof)
        assert entry.nbytes >= entry.handle.nbytes + 8 * ndof * (STEPS + 1)
    finally:
        entry.free()


def test_a_trajectory_over_the_budget_keeps_checkpoints(mesh):
    k = ("material", "clad", "thermal_conductivity")
    rule, entry = _kept_entry(mesh, {k})
    held = entry.handle.nbytes
    entry.free()
    tight = gfem.SolverOptions(order=2, retain_memory=held + 8 * rule._ndof())
    rule, entry = _kept_entry(mesh, {k}, tight)
    try:
        stride = math.ceil(math.sqrt(STEPS))
        assert entry.fields["stride"][0] == stride
        assert entry.fields["states"].shape == (math.ceil(STEPS / stride), rule._ndof())
    finally:
        entry.free()


def _difference(f, x, h):
    """Fourth-order central difference."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def test_a_waveform_scale_differentiates(mesh):
    options = gfem.SolverOptions(order=2)
    model = _model(mesh)

    def objective(scale):
        result = gfem.solve(model, _problem(mesh, scale), options=options)
        return jnp.sum(result.traces["heater"])

    grad = jax.grad(objective)(1.0)
    assert grad == pytest.approx(_difference(objective, 1.0, 1e-3), rel=1e-8)


# --- a section's thermo-optic trace: its weights are problem leaves -------------

FREQUENCY = scipy.constants.c / 1.55e-6
SECTION = [
    (-1.75, -1.0, 1.75, 1.8, "clad", None),
    (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    (-0.5, 1.5, 0.5, 1.65, "heater", 0.1),
]


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    path = tmp_path_factory.mktemp("shift_trace") / "section.msh"
    return tc.section(path, SECTION, 0.3, [(lambda p: abs(p[1] + 1.0) < 1e-9, "bottom")])


def _section_objective(mesh):
    n_core = len(mesh.elements("core"))
    heat = gfem.HeatSource.uniform("heater", power_per_length=1.0)

    def objective(dn, k_clad):
        materials = {
            "clad": gfem.MaterialField(
                eps_r=jnp.full(len(mesh.elements("clad")), 1.444**2),
                thermal_conductivity=k_clad,
                mass_density=jnp.full(len(k_clad), RHO),
                specific_heat=jnp.full(len(k_clad), CP),
                thermo_optic_coefficient=jnp.full(len(k_clad), 1e-5),
            ),
            "core": gfem.MaterialField(
                eps_r=jnp.full(n_core, 3.4777**2),
                thermal_conductivity=jnp.full(n_core, 90.0),
                mass_density=jnp.full(n_core, 2330.0),
                specific_heat=jnp.full(n_core, 700.0),
                thermo_optic_coefficient=dn,
            ),
            "heater": gfem.Material(
                thermal_conductivity=28.0, mass_density=4500.0, specific_heat=500.0
            ),
        }
        model = gfem.Model(
            mesh,
            materials=materials,
            boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
            ports=[gfem.WavePort("p", "front")],
            excluded={"em": ["heater"]},
        )
        shift = gfem.thermo_optic_shift(model, None, port="p", frequency=FREQUENCY)
        problem = gfem.TransientThermal(
            heat,
            duration=1e-6,
            steps=6,
            traces={"phase": shift.trace(), "core": gfem.Trace.mean("core")},
        )
        result = gfem.solve(model, problem)
        return jnp.sum(result.traces["phase"]) + 0.1 * jnp.sum(result.traces["core"])

    return objective


def test_a_thermo_optic_trace_differentiates_by_its_weights_and_the_rise(section):
    objective = _section_objective(section)
    rng = np.random.default_rng(3)
    dn = jnp.asarray(rng.uniform(1.5e-4, 2.0e-4, len(section.elements("core"))))
    k = jnp.asarray(rng.uniform(1.2, 1.6, len(section.elements("clad"))))
    g_dn, g_k = jax.grad(objective, argnums=(0, 1))(dn, k)
    # the objective is linear in dn/dT: a plain difference is exact
    d_dn = dn
    d_k = rng.normal(size=k.shape) * 0.1
    want_dn = objective(dn + d_dn, k) - objective(dn, k)
    want_k = _difference(lambda t: objective(dn, k + t * d_k), 0.0, 1e-2)
    assert float(g_dn @ d_dn) == pytest.approx(float(want_dn), rel=1e-7)
    assert float(g_k @ d_k) == pytest.approx(float(want_k), rel=1e-7)
