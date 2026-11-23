"""Sources on the solve paths: frequency workers, the solve cache, saving.

A dielectric cube in an air box wrapped in a perfectly matched layer, with a
lumped port on the cube's +x face (`free_space_cases.scatterer_box`), driven
by a plane wave and an electric dipole. Frequency workers solve the same
columns as one process; a user incident function must load in a spawned
worker, checked before any worker starts; the solve cache keys the built-in
sources and passes a user function by with a warning that names it.
"""

import logging
import sys

import free_space_cases as fs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem._engine.driven_sweep import BlockOutcome, SweepSplit
from gfem._engine.solve_request import SolveRequest

F = fs.F0 / 2
GRID = (F, 1.1 * F)
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
DIPOLE = gfem.ElectricDipole(
    "d", position=fs.DIPOLE_POSITION, moment=(0.0, 1e-28, 0.3e-28)
)
#: Direct solves of the same matrices in two processes: round-off apart.
WORKER_GATE = 1e-12


def _options(workers=1):
    return gfem.SolverOptions(
        order=2, linear_solver="direct", frequency_workers=workers
    )


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return fs.scatterer_model(
        fs.scatterer_box(tmp_path_factory.mktemp("box") / "b.msh")
    )


def _relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / np.abs(b).max()


def test_two_workers_solve_the_sources_as_one_process(model):
    problem = gfem.Driven(GRID, sources=(WAVE, DIPOLE), keep_fields=True)
    one = gfem.solve(model, problem, options=_options(1))
    two = gfem.solve(model, problem, options=_options(2))
    assert _relative(two.s.values, one.s.values) < WORKER_GATE
    assert _relative(two.port_waves, one.port_waves) < WORKER_GATE
    assert _relative(two.emitted_power["d"], one.emitted_power["d"]) < WORKER_GATE
    for f in GRID:
        for part in ("total", "scattered"):
            got = two.e_field(f, "w", part=part).coefficients
            want = one.e_field(f, "w", part=part).coefficients
            assert _relative(got, want) < WORKER_GATE, (f, part)
        got = two.e_field(f, "d").coefficients
        assert _relative(got, one.e_field(f, "d").coefficients) < WORKER_GATE


def _wave(points, frequency):  # noqa: ARG001 - a uniform field
    out = np.zeros((len(points), 3), complex)
    out[:, 1] = 1.0
    return out


@pytest.fixture
def no_worker(monkeypatch):
    """Fail the test if a frequency worker starts."""
    from gfem._engine import workers

    def refuse(*_args, **_kwargs):
        raise AssertionError("a frequency worker started")

    monkeypatch.setattr(workers, "run", refuse)


def test_an_incident_function_that_does_not_pickle_is_refused_before_spawning(
    model,
    no_worker,  # noqa: ARG001 - a guard
):
    source = gfem.IncidentField("f", lambda p, f: _wave(p, f) * 2)
    problem = gfem.Driven(GRID, excite=(), sources=source)
    with pytest.raises(
        ValueError,
        match=r"IncidentField 'f'.*cannot be sent to a frequency worker.*"
        r"module level.*frequency_workers=1",
    ):
        gfem.solve(model, problem, options=_options(2))


def test_an_incident_function_of_an_interactive_session_is_refused(
    model,
    no_worker,  # noqa: ARG001 - a guard
    monkeypatch,
):
    # a function of a notebook, a REPL or `python -c` pickles by reference to
    # a __main__ that a spawned worker cannot import
    main = sys.modules["__main__"]
    monkeypatch.delattr(main, "__file__", raising=False)

    def interactive(points, frequency):
        return _wave(points, frequency)

    interactive.__module__ = "__main__"
    interactive.__qualname__ = "_interactive_wave"
    monkeypatch.setattr(main, "_interactive_wave", interactive, raising=False)
    problem = gfem.Driven(GRID, excite=(), sources=gfem.IncidentField("f", interactive))
    with pytest.raises(ValueError, match=r"IncidentField 'f'.*__main__.*module level"):
        gfem.solve(model, problem, options=_options(2))
    # one process needs no pickling
    gfem.solve(model, problem, options=_options(1))


# --- the solve cache ------------------------------------------------------------------


def test_a_second_solve_of_built_in_sources_is_a_cache_hit(model, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solve_cache")
    problem = gfem.Driven(F, sources=(WAVE, DIPOLE), keep_fields=True)
    first = gfem.solve(model, problem, options=_options(), cache=tmp_path / "c")
    second = gfem.solve(model, problem, options=_options(), cache=tmp_path / "c")
    lines = [r.getMessage() for r in caplog.records if r.name == "gfem.solve_cache"]
    assert [line.split()[0] for line in lines] == ["miss", "hit"]
    np.testing.assert_array_equal(second.port_waves, first.port_waves)
    np.testing.assert_array_equal(second.emitted_power["d"], first.emitted_power["d"])
    np.testing.assert_array_equal(
        second.e_field(F, "w", part="scattered").coefficients,
        first.e_field(F, "w", part="scattered").coefficients,
    )


def test_another_plane_wave_is_another_cache_entry(model, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solve_cache")
    for theta in (0.4, 0.5):
        source = gfem.PlaneWave("w", theta=theta, phi=0.3, polarization="TM")
        problem = gfem.Driven(F, excite=(), sources=source)
        gfem.solve(model, problem, options=_options(), cache=tmp_path / "c")
    lines = [r.getMessage() for r in caplog.records if r.name == "gfem.solve_cache"]
    assert [line.split()[0] for line in lines] == ["miss", "miss"]


def test_an_incident_function_bypasses_the_cache_with_a_warning_naming_it(
    model, tmp_path
):
    problem = gfem.Driven(F, excite=(), sources=gfem.IncidentField("f", _wave))
    with pytest.warns(gfem.SolveCacheWarning) as caught:
        result = gfem.solve(model, problem, options=_options(), cache=tmp_path / "c")
    (message,) = [str(w.message) for w in caught]
    assert "IncidentField 'f'" in message
    assert "_wave" in message
    assert "PlaneWave" in message
    assert "subclass" not in message
    assert result.port_waves is not None
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").rglob("*.json"))


# --- result widths --------------------------------------------------------------------


def test_a_dipole_on_an_adaptive_sweep_without_ports_differentiates(model):
    # the traced outputs hold the source columns' port waves, empty without ports
    closed = model.replace(ports=[], background=None)
    n = len(closed.mesh.elements("scat"))
    band = tuple(np.linspace(0.9, 1.1, 5) * F)
    problem = gfem.Driven(
        band, sources=DIPOLE, sweep="adaptive", samples=(band[0], band[2], band[4])
    )
    options = gfem.SolverOptions(order=1, linear_solver="direct")

    def power(eps):
        moved = closed.with_material("scat", gfem.MaterialField(eps_r=eps))
        return gfem.solve(moved, problem, options=options).emitted_power["d"][2]

    eps = jnp.full(n, fs.SCATTERER_EPS)
    value, grad = jax.value_and_grad(power)(eps)
    assert value > 0
    assert np.isfinite(np.asarray(grad)).all()


class _Missed:
    """A block runner whose only block misses at its first frequency."""

    def run(self, model, blocks, *, keep, request):  # noqa: ARG002
        (block,) = blocks
        return [BlockOutcome(block.indices, None, None, (0, "no luck."), None)]


def test_a_split_sweep_that_solved_nothing_keeps_the_source_widths():
    split = SweepSplit(
        frequency=GRID,
        wave_ports=(),
        port_names=("p1",),
        excite=("p1",),
        physics="em",
        options=_options(),
        blocks=([0, 1],),
        runner=_Missed(),
        n_sources=2,
        n_dipoles=1,
    )
    request = SolveRequest(keeps_partial=True)
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        split.forward(None, keep=False, request=request)
    outputs = caught.value._partial.outputs
    assert outputs.s.shape == (0, 1, 1)
    assert outputs.port_waves.shape == (0, 1, 2)
    assert outputs.emitted_power.shape == (0, 1)
