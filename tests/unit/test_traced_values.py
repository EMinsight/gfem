"""Fixed physics values raise `gfem.NonDifferentiableError` when traced.

Frequencies, kept or frozen sample frequencies, eigen targets, port
directions and voltage paths are fixed numbers of a problem or a port:
under `jax.jit` or `jax.grad` they raise `gfem.NonDifferentiableError`
(which an existing ``except ValueError`` still catches) naming the value.
"""

import re

import jax
import jax.numpy as jnp
import pytest
import refinement_cases as rc

import gfem

CASES = {
    "PortMode.frequency": lambda x: gfem.PortMode("p1", x),
    "Driven.frequency": lambda x: gfem.Driven(frequency=x),
    "Driven.keep_fields": lambda x: gfem.Driven(frequency=1e9, keep_fields=x),
    "Driven.samples": lambda x: gfem.Driven(
        frequency=(1e9, 2e9), sweep="adaptive", samples=x
    ),
    "Eigenmode.target": lambda x: gfem.Eigenmode(target=x),
    "direction": lambda x: gfem.LumpedPort("p1", "port1", x * jnp.ones(3)),
    "voltage_path": lambda x: gfem.WavePort(
        "p1", "port1", voltage_path=x * jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("transform", ["jit", "grad"])
def test_a_traced_fixed_value_raises_nondifferentiable(name, transform):
    build = CASES[name]

    def objective(x):
        build(x)
        return x

    wrapped = jax.jit(objective) if transform == "jit" else jax.grad(objective)
    with pytest.raises(gfem.NonDifferentiableError, match=name.split(".")[-1]):
        wrapped(1e9)


def test_the_message_names_the_jax_transformations():
    with pytest.raises(gfem.NonDifferentiableError) as err:
        jax.grad(lambda f: gfem.Driven(frequency=f) and f)(1e9)
    message = str(err.value)
    assert "computed inside jax.grad, jax.jit or jax.vmap" in message
    assert not re.search(r"\b(traced|tracer|leaf|leaves|pytree)\b", message)


def test_the_error_is_still_a_value_error():
    with pytest.raises(ValueError, match="frequency"):
        jax.jit(lambda f: gfem.Driven(frequency=f))(1e9)


# --- mesh refinement ---------------------------------------------------------
# `refine_mesh` and `error_estimate` are never differentiated: a traced model
# leaf (under jax.grad or jax.vmap) and any call inside jax.jit, even on a
# concrete closed-over model, raise naming the function.

REFINE_OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture(scope="module")
def coax(tmp_path_factory):
    return rc.coax_model(tmp_path_factory.mktemp("coax"), 2)


@pytest.fixture(scope="module")
def refinement_calls(coax):
    """name -> call(model): each refinement function on ``model``."""
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=REFINE_OPTIONS)
    return {
        "refine_mesh": lambda model: gfem.refine_mesh(
            model, rc.COAX_PROBLEM, max_cycles=0, options=REFINE_OPTIONS
        ),
        "error_estimate": lambda model: gfem.error_estimate(
            result, gfem.FieldEstimator(), model=model
        ),
    }


@pytest.mark.parametrize("name", ["refine_mesh", "error_estimate"])
@pytest.mark.parametrize("transform", ["grad", "vmap", "jit"])
def test_mesh_refinement_raises_nondifferentiable(
    coax, refinement_calls, name, transform
):
    call = refinement_calls[name]
    n = len(coax.mesh.elements("dielectric"))

    def objective(x):
        # a traced leaf: the design is the transformed variable
        field = gfem.MaterialField(eps_r=x * jnp.ones(n))
        call(coax.replace(materials={"dielectric": field}))
        return x

    if transform == "jit":
        # concrete inputs, but the call is staged: the closure raises
        wrapped, x = jax.jit(lambda x: (call(coax), x)[1]), 1.0
    elif transform == "grad":
        wrapped, x = jax.grad(objective), 1.0
    else:
        wrapped, x = jax.vmap(objective), jnp.ones(2)
    with pytest.raises(gfem.NonDifferentiableError, match=name):
        wrapped(x)
