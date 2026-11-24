"""`solve_vjp` hands every host call the design state it was traced with."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._adjoint import adjoint_tolerance, solve_vjp
from gfem._adjoint.solve_vjp import _add_explicit_inputs
from gfem._pytree import register_fields


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A problem with one array leaf."""

    scale: object


register_fields(Scaled, ("scale",))


class ScaledRule:
    """Returns ``scale * sum(eps_r)``, reading the problem from ``self.problem``."""

    physics = "em"
    form = "electrostatic"
    differentiable_outputs = frozenset()

    def __init__(self, problem):
        self.problem = problem
        self.options = gfem.SolverOptions()

    def output_specs(self, model):  # noqa: ARG002
        return {"y": jax.ShapeDtypeStruct((), jnp.float64)}

    def role(self, kind):  # noqa: ARG002
        return "unread"

    def forward(self, model, *, keep, request=None):  # noqa: ARG002
        eps = np.asarray(model.materials["box"].eps_r)
        return {"y": float(np.asarray(self.problem.scale)) * eps.sum()}, None


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("box"))
    return gfem.Model(mesh, materials={"box": gfem.MaterialField(eps_r=np.ones(n))})


def _solve(model, scale):
    problem = Scaled(scale)
    return solve_vjp(ScaledRule(problem), model, problem)["y"]


def test_host_forward_reads_problem_leaves_from_the_traced_values(model):
    n = len(model.mesh.elements("box"))
    assert float(jax.jit(lambda s: _solve(model, s))(2.0)) == 2.0 * n
    scales = jnp.asarray([1.0, 3.0, 5.0])
    np.testing.assert_allclose(jax.vmap(lambda s: _solve(model, s))(scales), scales * n)


class FixedScaleRule(ScaledRule):
    """Reads the problem's scale but refuses to differentiate with respect to it."""

    def role(self, kind):
        return "read" if kind[0] == "problem" else "unread"

    def unsupported_input(self, kind):  # noqa: ARG002
        return "it is fixed"


def test_a_refused_problem_input_is_named_by_its_attribute(model):
    def objective(s):
        problem = Scaled(s)
        return solve_vjp(FixedScaleRule(problem), model, problem)["y"]

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"problem\.scale: it is fixed"
    ):
        jax.grad(objective)(2.0)


def test_adjoint_tolerance_defaults_to_the_forward_tolerance():
    assert adjoint_tolerance(gfem.SolverOptions()) == 1e-6
    assert adjoint_tolerance(gfem.SolverOptions(tolerance=1e-9)) == 1e-9
    options = gfem.SolverOptions(tolerance=1e-9, adjoint_tolerance=1e-11)
    assert adjoint_tolerance(options) == 1e-11


def test_explicit_terms_add_to_the_wanted_lumped_values():
    port_l = ("port", 0, "inductance")
    port_c = ("port", 0, "capacitance")
    out = {port_l: np.asarray(1.0)}
    _add_explicit_inputs(
        out, {port_l: 2.0, port_c: 5.0}, wants={port_l, ("material", "box", "eps_r")}
    )
    # an unwanted kind's explicit term has nothing to add to
    assert out == {port_l: 3.0}


def test_an_explicit_lumped_term_of_a_wanted_kind_without_its_pass_is_a_bug():
    kind = ("port", 0, "inductance")
    with pytest.raises(RuntimeError, match=r"port.*0.*inductance"):
        _add_explicit_inputs({}, {kind: 2.0}, wants={kind})


def test_an_explicit_term_of_any_other_kind_adds_or_stands_alone_as_an_array():
    law = ("law", "air", 0, "delta_eps")
    fill = ("boundary", 1, "fill", "fill")
    unwanted = ("law", "air", "eps_inf")
    out = {law: np.asarray([1.0, 2.0])}
    _add_explicit_inputs(
        out,
        {law: np.asarray([0.5, 0.5]), fill: np.asarray([3.0]), unwanted: 7.0},
        wants={law, fill},
    )
    np.testing.assert_array_equal(out[law], [1.5, 2.5])
    np.testing.assert_array_equal(out[fill], [3.0])
    assert unwanted not in out
