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


def test_the_one_solved_field_needs_no_name(model):
    f = rc.grid(3)[1]
    problem = gfem.Driven(
        rc.grid(3),
        excite=("p1",),
        sources=rc.dipoles(2),
        keep_fields=[f],
        source_fields=False,
    )
    result = gfem.solve(model, problem, options=rc.ORDER2)
    # the dipoles were not solved: p1 is the only field, so no name is asked
    assert result.e_field(f) is result.e_field(f, "p1")


def test_iterations_count_each_right_hand_side(model):
    problem = gfem.Driven(
        rc.grid(2), excite=("p1",), sources=rc.dipoles(2), source_fields=False
    )
    options = gfem.SolverOptions(order=2, linear_solver="bddc_gmres")
    result = gfem.solve(model, problem, options=options)
    # the excited port p1, then the load of p2
    assert [len(row) for row in result.sweep.iterations] == [2, 2]


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_saved_reciprocal_result_still_refuses_source_fields(model, sweep):
    if sweep == "adaptive":
        result = _adaptive(model, 2)
    else:
        result = _solve(model, excite=(), source_fields=False)
    encoder = _codec.Encoder(_codec.Context(Path()))
    doc = json.loads(json.dumps(encoder.encode(result), allow_nan=False))
    back = _codec.Decoder(_codec.Context(Path()), encoder.arrays).decode(doc)
    np.testing.assert_array_equal(back.port_waves, result.port_waves)
    assert back.emitted_power == {}
    with pytest.raises(ValueError, match="source_fields=False"):
        gfem.purcell_factor(back, "d0", model=model)


ADAPTIVE = gfem.SolverOptions(order=2, linear_solver="direct", adaptive_tolerance=1e-4)


def _adaptive(model, n=5, *, excite=(), options=ADAPTIVE, **kwargs):
    problem = gfem.Driven(
        rc.grid(),
        excite=excite,
        sources=rc.dipoles(n),
        sweep="adaptive",
        source_fields=False,
        **kwargs,
    )
    return gfem.solve(model, problem, options=options)


@pytest.mark.parametrize("excite", [(), ("p1",)])
def test_an_adaptive_sweep_reads_reciprocal_port_waves(model, excite):
    result = _adaptive(model, excite=excite)
    reference = _solve(model, excite=excite)
    waves = np.asarray(reference.port_waves)
    # measured 2e-11 relative on this line
    assert (
        np.abs(np.asarray(result.port_waves) - waves).max()
        <= 1e-6 * np.abs(waves).max()
    )
    assert result.sweep.converged
    # with no excited column the error would read 0 without the port loads
    errors = np.asarray(result.sweep.sample_errors)[2:]
    assert errors.size
    assert np.isfinite(errors).all()
    assert (errors > 0).all()
    assert result.emitted_power == {}


def test_the_adaptive_basis_does_not_grow_with_the_dipoles(model):
    one = _adaptive(model, 1).sweep
    many = _adaptive(model, 20).sweep
    assert many.sampled_frequencies == one.sampled_frequencies
    assert many.basis_size == one.basis_size


def test_frozen_samples_trace_reciprocal_port_waves(model):
    first = _adaptive(model, excite=("p1",))
    problem = gfem.Driven(
        rc.grid(),
        excite=("p1",),
        sources=rc.dipoles(5),
        sweep="adaptive",
        samples=first.sweep.sampled_frequencies,
        keep_fields=[rc.grid()[3]],
        source_fields=False,
    )

    def objective(m):
        result = gfem.solve(m, problem, options=ADAPTIVE)
        return jnp.sum(jnp.abs(result.port_waves) ** 2)

    eager = objective(model)
    np.testing.assert_allclose(jax.jit(objective)(model), eager, rtol=1e-12)
    np.testing.assert_allclose(
        eager, jnp.sum(jnp.abs(first.port_waves) ** 2), rtol=1e-9
    )
    # a padded basis of two vectors per sample, excited port and port load
    n = len(first.sweep.sampled_frequencies)
    result = gfem.solve(model, problem, options=ADAPTIVE)
    assert result.e_fields.basis.shape[1] == 2 * n * (1 + 1)


def test_adaptive_iterations_count_each_right_hand_side(model):
    options = gfem.SolverOptions(
        order=2, linear_solver="bddc_gmres", adaptive_tolerance=1e-3
    )
    result = _adaptive(model, 2, excite=("p1",), options=options)
    assert all(len(row) == 2 for row in result.sweep.iterations)
