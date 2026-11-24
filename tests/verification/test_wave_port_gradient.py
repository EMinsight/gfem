"""S-parameter gradients through wave ports against finite differences.

A two-layer parallel-plate line fed by a wave port at each end, with a
"core" section in the middle that carries a random per-element lossy
permittivity and conductivity. That design stays away from the port faces,
so the port modes do not move with it; the ``face_*`` variants design the
lower layer of the leads instead, which touches both faces, so the modes
follow it too (the port pass of the driven backward). The layers differ in
permittivity, so the port mode is hybrid (``E_n != 0``) and the modal
correction is active:

- **lossless** ports: the mode is real up to one phase, so a port's S
  functional is its excitation scaled and the S adjoint is the port's own
  solution (no solve);
- **lossy** ports (a lossy lower layer in the leads, offset reference
  planes): complex propagation constants, a complex ``Z_PV`` and
  de-embedding offsets enter the port terms. Their biorthogonal S functional
  is still parallel to the excitation, so the adjoint is again the port's
  own solution; a solve is needed only for a port that is not excited.

The cells of the verification matrix (port x input x solver x sweep) are in
test_gradient_matrix.py (the line is `gradient_cases.wave_line`); here a
wave port beside a lumped one, the complex permittivity and the adjoint
paths. Adaptive checks read S at the frozen sample frequencies, where the
reduced model reproduces the full solve of the perturbed design too, and
its basis holds the S functionals' adjoints, so the frozen-basis gradient
is the full solve's there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import (
    core_design,
    face_line,
    ramp,
    s_functional,
    tight,
    wave_line,
    wave_mesh,
)

import gfem

SOLVERS = ["direct", "bddc_gmres", "pmg_gmres"]
FREQUENCIES = (12e9, 24e9)
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
ON = [GRID.index(f) for f in SAMPLES]
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return wave_mesh(tmp_path_factory.mktemp("line"))


def _s_objective(model, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        core = gfem.MaterialField(**{**values, name: p})
        m = model.with_material("core", core)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        s = gfem.solve(m, problem, options=options).s.values[jnp.asarray(rows)]
        return s_functional(s)

    return objective


# Fourth-order differences (see tests/finite_differences.py).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}


def _line_and_design(mesh, port, seed):
    """A port variant's model and its design ``(region, values, rng)``.

    ``"lossless"``, ``"lossy"`` and ``"mixed"``: the core's random field, away
    from the port faces. ``"face_lossless"`` and ``"face_lossy"``: the lower
    layer's ramp, which touches both faces, so the port modes follow it.
    """
    if port.startswith("face_"):
        n = len(mesh.elements("lo"))
        values = {"eps_r": ramp(mesh, "lo", 3.5, 4.5)}
        if port == "face_lossy":
            values["tan_delta"] = jnp.full(n, 0.05)
        return face_line(mesh, port), "lo", values, np.random.default_rng(seed)
    values, rng = core_design(mesh, seed=seed)
    return wave_line(mesh, port), "core", values, rng


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_wave_port_and_a_lumped_port_mix(mesh, sweep):
    values, rng = core_design(mesh, seed=4)
    problem, rows = (
        (gfem.Driven(frequency=FREQUENCIES), [0, 1])
        if sweep == "uniform"
        else (ADAPTIVE, ON)
    )
    objective = _s_objective(
        wave_line(mesh, "mixed"), values, "eps_r", tight("direct"), problem, rows
    )
    p = values["eps_r"]
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=STEPS["eps_r"])


@pytest.mark.parametrize("port", ["lossless", "lossy", "mixed", "face_lossy"])
def test_a_kept_field_objective_with_wave_ports(mesh, port):
    # the kept-field cotangent is one more load on the adjoint solve, whose
    # operator holds the wave ports' Robin terms and modal corrections; with a
    # design at the port faces the modes move too (the port pass)
    model, region, values, rng = _line_and_design(mesh, port, 4)
    kept = FREQUENCIES[1]
    problem = gfem.Driven(FREQUENCIES, excite="p1", keep_fields=kept)
    options = tight("direct", port_mode_tolerance=1e-12)

    def solve(p):
        field = gfem.MaterialField(**{**values, "eps_r": p})
        m = model.with_material(region, field)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return gfem.solve(m, problem, options=options)

    p = values["eps_r"]
    n = solve(p).e_field(kept, "p1").coefficients.shape[0]
    w = np.array([1, 1j]) @ np.random.default_rng(1).normal(size=(2, n))
    w = jnp.asarray(w / np.linalg.norm(w))

    def objective(p):
        result = solve(p)
        e = result.e_field(kept, "p1").coefficients
        return jnp.abs(jnp.sum(w * e)) ** 2 + jnp.sum(jnp.abs(result.s.values) ** 2)

    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=STEPS["eps_r"])


def test_complex_permittivity_gradient_has_both_parts(mesh):
    values, rng = core_design(mesh, seed=2)
    problem = gfem.Driven(frequency=FREQUENCIES)
    objective = _s_objective(
        wave_line(mesh, "lossy"), values, "eps_r", tight("direct"), problem, [0, 1]
    )
    eps = values["eps_r"]
    (d,) = unit_directions(rng, eps.shape, 1)
    assert_gradient_matches(objective, eps, [d, 1j * d], step=STEPS["eps_r"])


def test_a_face_gradient_on_reduced_port_rows_matches_exact_differences(mesh):
    # between its exact samples a port mode sweep gives reduced rows, off by up
    # to the sweep's tolerance: a difference of solves at that tolerance would
    # see it, and the gradient (a Newton step brings each row to the exact pair
    # first) does not. The differences solve the modes to 1e-12.
    model, region, values, rng = _line_and_design(mesh, "face_lossy", 6)
    problem = gfem.Driven(frequency=GRID)
    rows = jnp.arange(len(GRID))

    def objective_with(options):
        def objective(p):
            field = gfem.MaterialField(**{**values, "eps_r": p})
            m = model.with_material(region, field)
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            return s_functional(gfem.solve(m, problem, options=options).s.values[rows])

        return objective

    production = gfem.SolverOptions(order=2, linear_solver="direct")
    exact = tight("direct", port_mode_tolerance=1e-12)
    modes = gfem.solve(
        model.with_material(region, gfem.MaterialField(**values)),
        problem,
        options=production,
    ).port_modes
    assert (~np.asarray(modes["p1"].solved)).sum() >= len(GRID) // 2
    p = values["eps_r"]
    grad = jax.grad(objective_with(production))(p)
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(
        objective_with(exact), p, directions, step=STEPS["eps_r"], grad=grad
    )


# --- consistency: the port's own solution vs an explicit adjoint solve -----------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("port", ["lossless", "lossy", "face_lossless", "face_lossy"])
def test_the_s21_gradient_does_not_depend_on_which_ports_are_excited(
    mesh, port, solver
):
    # Exciting p1 only leaves S21's adjoint to a solve with the port
    # operator; with both ports excited either kind reuses p2's own
    # solution. With a design at the port faces the port pass reads the free
    # (reciprocity) and the solved adjoint columns alike.
    model, region, values, _ = _line_and_design(mesh, port, 3)
    options = tight(solver, port_mode_tolerance=1e-12)

    def s21(problem):
        def objective(eps):
            m = model.with_material(
                region, gfem.MaterialField(**{**values, "eps_r": eps})
            )
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            s = gfem.solve(m, problem, options=options).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21(gfem.Driven(frequency=FREQUENCIES))
    adjoint = s21(gfem.Driven(frequency=FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


# a lone wave port's S functional is parallel to its load, lossy or not:
# its adjoint is its own solution scaled
@pytest.mark.parametrize(("port", "solves"), [("lossless", 0), ("lossy", 0)])
def test_lone_wave_ports_take_no_adjoint_solve(mesh, port, solves, monkeypatch):
    from gfem import _linalg
    from gfem._engine.system_operator import SystemOperator

    values, _ = core_design(mesh, seed=5)
    objective = _s_objective(
        wave_line(mesh, port),
        values,
        "eps_r",
        tight("direct"),
        gfem.Driven(frequency=FREQUENCIES),
        [0, 1],
    )
    value, pullback = jax.vjp(objective, values["eps_r"])
    solved, prepared = [], []
    solve, prepare = SystemOperator.solve_adjoint, _linalg.prepare

    def counting_solve(self, *args, **kwargs):
        solved.append(1)
        return solve(self, *args, **kwargs)

    def counting_prepare(*args, **kwargs):
        prepared.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(SystemOperator, "solve_adjoint", counting_solve)
    monkeypatch.setattr(_linalg, "prepare", counting_prepare)
    pullback(jnp.ones_like(value))
    assert len(solved) == solves  # one per frequency
    # the backward refactorizes the forward's base: it prepares nothing
    assert prepared == []
