"""Thermal boundary values are pytree leaves.

A fixed temperature, a heat transfer coefficient, an ambient and a flux can be
traced: a model builds with them, every forward checks their values on the
host, and the solves that do not read them give an exact zero gradient. The
thermal solves refuse a traced value by name until their backward reads it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem.errors import NonDifferentiableError

L, W, T = 4.0, 1.0, 0.2  # bar, um
METAL = gfem.Material(
    conductivity=1e6,
    eps_r=3.9,
    thermal_conductivity=28.0,
    mass_density=5240.0,
    specific_heat=598.0,
)
OXIDE = gfem.Material(
    eps_r=3.9, thermal_conductivity=1.4, mass_density=2200.0, specific_heat=700.0
)
DIRECT = gfem.SolverOptions(order=1, linear_solver="direct")


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
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

    path = tmp_path_factory.mktemp("leaves") / "bar.msh"
    return gfem.Mesh.from_gmsh(
        sm.write_msh22(path, sm.structured(edges, volume, surface))
    )


def _model(mesh, boundaries):
    return gfem.Model(
        mesh, materials={"bar": METAL, "oxide": OXIDE}, boundaries=boundaries
    )


def _traced(mesh, t, h, amb, q):
    return _model(
        mesh,
        [
            gfem.FixedTemperature("floor", temperature=t),
            gfem.Convection("end_a", coefficient=h, ambient=amb),
            gfem.HeatFlux("end_b", flux=q),
        ],
    )


VALUES = (300.0, 25.0, 290.0, 5.0)
LEAF_NAMES = ("temperature", "coefficient", "ambient", "flux")


def test_the_four_values_are_leaves_in_order(mesh):
    model = _traced(mesh, *VALUES)
    leaves = jax.tree_util.tree_leaves(model.boundaries)
    assert leaves == list(VALUES)
    paths = [
        jax.tree_util.keystr(p)
        for p, _ in jax.tree_util.tree_flatten_with_path(model.boundaries)[0]
    ]
    assert [p.rsplit(".", 1)[1] for p in paths] == list(LEAF_NAMES)


@pytest.mark.parametrize("which", range(4))
def test_a_traced_value_builds(mesh, which):
    def f(v):
        args = list(VALUES)
        args[which] = v
        _traced(mesh, *args)
        return v

    assert float(jax.grad(f)(VALUES[which])) == 1.0


def test_dc_conduction_gives_an_exact_zero_gradient(mesh):
    problem = gfem.DCConduction(terminals={"a": "end_a"}, ground="end_b")

    def r(values):
        model = _traced(mesh, *values)
        return gfem.solve(model, problem, options=DIRECT).resistance[0, 0]

    values = jnp.asarray(VALUES)
    assert float(r(values)) > 0
    np.testing.assert_array_equal(np.asarray(jax.grad(r)(values)), 0.0)


def test_electrostatics_gives_an_exact_zero_gradient(mesh):
    problem = gfem.Electrostatic(terminals={"a": "end_a"}, ground="end_b")

    def c(values):
        model = _traced(mesh, *values)
        return gfem.solve(model, problem, options=DIRECT).capacitance[0, 0]

    values = jnp.asarray(VALUES)
    assert float(c(values)) != 0
    np.testing.assert_array_equal(np.asarray(jax.grad(c)(values)), 0.0)


def _roles(rule):
    return {name: rule.role(("boundary", 0, name)) for name in LEAF_NAMES}


@pytest.mark.parametrize("rule", ["driven", "eigenmode", "port_mode"])
def test_the_electromagnetic_rules_read_no_thermal_value(rule):
    from gfem._eigen.rule import EigenmodeRule
    from gfem._engine.driven import DrivenRule
    from gfem._sweep.port_mode import PortModeRule

    cls = {"driven": DrivenRule, "eigenmode": EigenmodeRule, "port_mode": PortModeRule}
    bare = object.__new__(cls[rule])
    assert set(_roles(bare).values()) == {"unread"}
    assert bare.role(("boundary", 0, "resistance")) == "read"


def test_the_thermal_rules_read_their_own_values():
    from gfem._engine.thermal import SteadyThermalRule, TransientThermalRule

    for cls in (SteadyThermalRule, TransientThermalRule):
        bare = object.__new__(cls)
        assert set(_roles(bare).values()) == {"read"}
        assert bare.role(("boundary", 0, "resistance")) == "unread"


def test_touching_fixed_surfaces_with_different_values_raise_also_traced(mesh):
    problem = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-3))

    def build(a, b):
        return _model(
            mesh,
            [
                gfem.FixedTemperature("floor", temperature=a),
                gfem.FixedTemperature("end_a", temperature=b),
            ],
        )

    with pytest.raises(ValueError, match="share"):
        gfem.solve(build(300.0, 310.0), problem, options=DIRECT)

    def traced(a):
        result = gfem.solve(build(a, 310.0), problem, options=DIRECT)
        return result.temperature.coefficients.sum()

    # under jit the values reach the host forward, which checks them
    with pytest.raises(Exception, match="share"):
        jax.jit(traced)(300.0)


def test_a_model_built_in_jit_solves_like_the_concrete_one(mesh):
    problem = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-3))
    # a flux of 0 traced is still a boundary, and the heat flow through it is 0
    values = (300.0, 25.0, 290.0, 0.0)

    def solve(*args):
        result = gfem.solve(_traced(mesh, *args), problem, options=DIRECT)
        return result.temperature.coefficients.sum()

    def rise(*args):
        model = _traced(mesh, *args)
        result = gfem.solve(model, problem, options=DIRECT)
        return gfem.heat_flow(result, ("end_a", "end_b"), model=model)

    assert float(jax.jit(rise)(*values)) == pytest.approx(float(rise(*values)))
    assert float(jax.jit(solve)(*values)) == pytest.approx(float(solve(*values)))


@pytest.mark.parametrize(
    ("which", "bad", "message"),
    [
        (0, -1.0, "must be > 0 K"),
        (1, -1.0, "must be > 0 W"),
        (2, 0.0, "must be > 0 K"),
        (3, float("inf"), "finite"),
    ],
)
def test_a_traced_invalid_value_raises_at_the_forward(mesh, which, bad, message):
    problem = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-3))

    def f(v):
        args = list(VALUES)
        args[which] = v
        result = gfem.solve(_traced(mesh, *args), problem, options=DIRECT)
        return result.temperature.coefficients.sum()

    with pytest.raises(Exception, match=message) as raised:
        jax.jit(f)(bad)
    assert LEAF_NAMES[which] in str(raised.value)


@pytest.mark.parametrize("which", range(4))
def test_a_thermal_solve_refuses_a_traced_value_by_name(mesh, which):
    problem = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-3))

    def f(v):
        args = list(VALUES)
        args[which] = v
        model = _traced(mesh, *args)
        return gfem.solve(model, problem, options=DIRECT).power

    with pytest.raises(NonDifferentiableError, match=LEAF_NAMES[which]):
        jax.grad(f)(VALUES[which])


def test_a_zero_flux_surface_stays_a_boundary_and_its_flow_measures(mesh):
    model = _traced(mesh, 300.0, 25.0, 290.0, 0.0)
    problem = gfem.SteadyThermal(gfem.HeatSource.uniform("bar", power=1e-3))
    result = gfem.solve(model, problem, options=DIRECT)
    flow = float(gfem.heat_flow(result, "end_b", model=model))
    assert flow == pytest.approx(0.0, abs=1e-12 * float(result.power))
