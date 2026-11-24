"""`Driven(source_fields=False)` on the terminated line: port waves by reciprocity.

Each dipole's port waves are read from the ports' own solutions, one point
overlap per dipole, and equal the dipole's own column to round-off.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reciprocity_cases as rc

import gfem
from gfem import _codec


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return rc.line_model(tmp_path_factory.mktemp("line"))


def _solve(model, **kwargs):
    problem = gfem.Driven(rc.grid(), sources=rc.dipoles(), **kwargs)
    return gfem.solve(model, problem, options=rc.ORDER2)


@pytest.mark.parametrize("excite", [None, ("p1",), ()])
def test_reciprocal_port_waves_are_the_dipoles_own_columns(model, excite):
    columns = _solve(model, excite=excite)
    reciprocal = _solve(model, excite=excite, source_fields=False)
    waves = np.asarray(columns.port_waves)
    # round-off apart: PARDISO and the transposed overlap order sums differently
    assert np.abs(np.asarray(reciprocal.port_waves) - waves).max() <= (
        1e-12 * np.abs(waves).max()
    )
    assert reciprocal.port_waves.shape == columns.port_waves.shape
    assert reciprocal.s.excited == columns.s.excited
    np.testing.assert_allclose(
        reciprocal.s.values, columns.s.values, rtol=0, atol=1e-12
    )
    assert reciprocal.emitted_power == {}
    assert list(reciprocal.sources) == list(columns.sources)


def test_only_the_excited_ports_keep_fields(model):
    f = rc.grid(3)[1]
    problem = gfem.Driven(
        rc.grid(3),
        excite=("p1",),
        sources=rc.dipoles(2),
        keep_fields=[f],
        source_fields=False,
    )
    result = gfem.solve(model, problem, options=rc.ORDER2)
    assert result.kept == ((f, "p1"),)
    assert result.e_field(f, "p1").coefficients.shape[0] > 0
    with pytest.raises(ValueError, match="source_fields=False"):
        result.e_field(f, "d0")


def test_iterations_count_each_right_hand_side(model):
    problem = gfem.Driven(
        rc.grid(2), excite=("p1",), sources=rc.dipoles(2), source_fields=False
    )
    options = gfem.SolverOptions(order=2, linear_solver="bddc_gmres")
    result = gfem.solve(model, problem, options=options)
    # the excited port p1, then the load of p2
    assert [len(row) for row in result.sweep.iterations] == [2, 2]


def test_a_saved_reciprocal_result_still_refuses_source_fields(model):
    result = _solve(model, excite=(), source_fields=False)
    encoder = _codec.Encoder(_codec.Context(Path()))
    doc = json.loads(json.dumps(encoder.encode(result), allow_nan=False))
    back = _codec.Decoder(_codec.Context(Path()), encoder.arrays).decode(doc)
    np.testing.assert_array_equal(back.port_waves, result.port_waves)
    assert back.emitted_power == {}
    with pytest.raises(ValueError, match="source_fields=False"):
        gfem.purcell_factor(back, "d0", model=model)


def test_an_adaptive_sweep_is_not_available_yet(model):
    problem = gfem.Driven(
        rc.grid(), sources=rc.dipoles(2), sweep="adaptive", source_fields=False
    )
    with pytest.raises(NotImplementedError, match="adaptive"):
        gfem.solve(model, problem, options=rc.ORDER2)


def test_gradients_are_not_available_yet(model):
    structure = jax.tree_util.tree_structure(model.mesh)
    problem = gfem.Driven(
        rc.grid(3),
        excite=(),
        sources=rc.dipoles(2, magnetic=False),
        source_fields=False,
    )

    def objective(nodes):
        moved = model.replace(mesh=jax.tree_util.tree_unflatten(structure, [nodes]))
        result = gfem.solve(moved, problem, options=rc.ORDER2)
        return jnp.abs(result.port_wave("p1", "d0")[1]) ** 2

    # raised on the host in the backward pass, inside JAX's runtime error
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"NotImplementedError: Gradients of a Driven\(source_fields=False",
    ):
        jax.grad(objective)(model.mesh.nodes)
