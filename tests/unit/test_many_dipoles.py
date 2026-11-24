"""Many dipole sources: adaptive sweeps' thin SVD and block extend, sparse loads.

Many point sources add snapshot columns to every sample of the adaptive sweep,
which makes the rational-interpolation step's matrix tall; its full left
factor would be a dense square of that height. The sweep keeps the thin factor
and appends each sample's snapshot to the basis in one block, with the same
samples and outputs as the one-column-at-a-time build.
"""

import tempfile
import tracemalloc
from pathlib import Path

import eigen_cases as ec
import numpy as np
import pytest
import scipy.sparse as sp
from adaptive_reference import per_column_extend, reference_next_sample
from eigen_cases import line_mesh, terminated_line

import gfem
from gfem._engine import probe as probe_module
from gfem._engine.dipoles import point_loads
from gfem._engine.driven import driven_operator
from gfem._engine.probe import Layout, probe
from gfem._linalg.base import free_mask
from gfem._sweep import adaptive

M = 10
GRID = tuple(np.linspace(4e9, 11e9, 201))


def _positions(m):
    rng = np.random.default_rng(0)
    return list(
        zip(
            rng.uniform(0.3e-3, 9.7e-3, m),
            rng.uniform(0.2e-3, 1.8e-3, m),
            rng.uniform(0.2e-3, 0.8e-3, m),
            strict=True,
        )
    )


def _solve():
    mesh = line_mesh(Path(tempfile.mkdtemp()), (20, 1, 1))
    model = terminated_line(mesh, 25.0)
    sources = [
        gfem.ElectricDipole(f"d{i}", position=p, moment=(0, 0, 1e-15))
        for i, p in enumerate(_positions(M))
    ]
    problem = gfem.Driven(GRID, sweep="adaptive", sources=sources)
    options = gfem.SolverOptions(order=2, adaptive_tolerance=1e-4)
    return gfem.solve(model, problem, options=options)


def test_many_dipoles_sweep_matches_the_full_svd_per_column_reference(monkeypatch):
    new = _solve()
    monkeypatch.setattr(adaptive.ReducedModel, "extend", per_column_extend)
    monkeypatch.setattr(
        adaptive,
        "_next_sample",
        lambda *args: reference_next_sample(*args)[0],
    )
    old = _solve()
    a, b = new.sweep, old.sweep
    assert a.sampled_frequencies == b.sampled_frequencies
    assert a.basis_size == b.basis_size
    assert a.converged == b.converged
    for name in ("s", "port_waves", "emitted_power"):
        x, y = getattr(new, name), getattr(old, name)
        if name == "s":
            x, y = x.values, y.values
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            x, y = np.array(list(x.values())), np.array(list(y.values()))
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=0)
    np.testing.assert_allclose(a.sample_errors, b.sample_errors, rtol=1e-6, atol=1e-14)


def test_a_many_dipole_adaptive_sweep_never_holds_the_full_svd_factor():
    tracemalloc.start()
    try:
        result = _solve()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    n, p = result.sweep.basis_size, 2 + M
    assert peak <= 0.5 * (2 * n * p) ** 2 * 16


# --- point loads built in one pass, stored sparse ---


@pytest.fixture(scope="module")
def loads_line(tmp_path_factory):
    mesh = ec.line_mesh(tmp_path_factory.mktemp("many_dipoles"), (20, 1, 1))
    model = ec.terminated_line(mesh, 25.0)
    op = driven_operator(model, 2)
    layout = Layout(support=("dielectric",), excluded=(), space="hcurl", order=2)
    return model, op.space, layout


def _seeded_dipoles(m, *, magnetic=0):
    """``m`` seeded dipoles inside the line, the last ``magnetic`` magnetic."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3e-3, 9.7e-3, m)
    y = rng.uniform(0.2e-3, 1.8e-3, m)
    z = rng.uniform(0.2e-3, 0.8e-3, m)
    out = []
    for i in range(m):
        cls = gfem.MagneticDipole if i >= m - magnetic else gfem.ElectricDipole
        moment = (1e-15 * (1 + i % 3), 2e-15j, 1e-15 * (i % 2))
        out.append(cls(f"d{i}", position=(x[i], y[i], z[i]), moment=moment))
    return out


def _per_dipole(model, space, layout, sources, loads):
    """The one-dipole-at-a-time build the sparse loads replace."""
    probed = probe(model.mesh, layout, loads.located)
    n = len(sources)
    out = np.zeros((probed.width, n), complex)
    for k in range(n):
        g = np.zeros((1, n, 3), complex)
        g[0, k] = loads.moments[k]
        out[:, k] = probed.transpose(g, curl=bool(loads.magnetic[k]))[0]
    out = out[: space.ndof]
    out[~free_mask(space.FreeDofs(), space.ndof)] = 0.0
    return out


def test_sparse_point_loads_equal_the_per_dipole_transpose(loads_line):
    model, space, layout = loads_line
    sources = _seeded_dipoles(20, magnetic=5)
    loads = point_loads(model, sources, layout, space, (5e9,))
    assert sp.issparse(loads.vectors)
    assert loads.vectors.shape == (space.ndof, 20)
    reference = _per_dipole(model, space, layout, sources, loads)
    dense = loads.vectors.toarray()
    scale = np.abs(reference).max(axis=0)
    assert (np.abs(dense - reference).max(axis=0) <= 1e-13 * scale).all()
    fixed = ~free_mask(space.FreeDofs(), space.ndof)
    assert not dense[fixed].any()


def test_point_loads_evaluate_the_shape_functions_once_for_all_seeded_dipoles(
    loads_line, monkeypatch
):
    model, space, layout = loads_line
    calls = []
    evaluate = probe_module.Probe._evaluate

    def counted(self, *args, **kwargs):
        calls.append(1)
        return evaluate(self, *args, **kwargs)

    monkeypatch.setattr(probe_module.Probe, "_evaluate", counted)
    counts = []
    for m in (5, 50):
        calls.clear()
        point_loads(model, _seeded_dipoles(m, magnetic=1), layout, space, (5e9,))
        counts.append(len(calls))
    assert counts[0] == counts[1] > 0


def test_the_loads_of_many_dipoles_take_no_dense_array(loads_line):
    model, space, layout = loads_line
    sources = _seeded_dipoles(1000)
    tracemalloc.start()
    try:
        point_loads(model, sources, layout, space, (5e9,))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= 0.25 * space.ndof * len(sources) * 16
