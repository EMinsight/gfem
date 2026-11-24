"""The adaptive frequency sweep on a small dense affine system.

A(ω) = K - ω² M + iω C with real symmetric K, M, C (so A is complex
symmetric, like the driven operator), two right-hand sides i ω b_p and two
outputs l_p · x. K has four eigenvalues whose resonances fall in the band
and the rest far above it, so the response is a rational function with a
handful of poles in band plus a smooth background: the greedy needs a few
samples, never the whole grid.
"""

import dataclasses
import math

import numpy as np
import pytest

import gfem
from gfem._sweep.adaptive import (
    AffineSystem,
    AffineTerm,
    ReducedModel,
    _next_sample,
    adaptive_sweep,
)

N = 60
GRID = np.linspace(1.0, 3.0, 201)


def _system(seed=0, damping=0.02, correction=False):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    eigenvalues = np.concatenate([[1.4, 2.9, 4.8, 7.3], rng.uniform(40, 90, N - 4)])
    k = q @ np.diag(eigenvalues) @ q.T
    m = np.eye(N)
    c = damping * np.diag(rng.uniform(0.5, 1.5, N))
    b = rng.standard_normal((N, 2))
    pieces = (k, m, c)
    thetas = (lambda _: 1.0, lambda w: -(w**2), lambda w: 1j * w)

    # a low-rank term U C(ω) Uᵀ with a tabulated-looking coefficient, like a
    # wave port's modal correction (rank two, complex symmetric)
    u = rng.standard_normal((N, 3))
    p_vec = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))

    def coefficient(w):
        d = np.diag([1.0 / (1 + 1j * w), -0.5j * w])
        return p_vec @ d @ p_vec.T

    corrections = (AffineTerm(u, coefficient),) if correction else ()

    def matrix(w):
        a = sum(t(w) * p for t, p in zip(thetas, pieces, strict=True))
        for term in corrections:
            a = a + term.vectors @ term.coefficients(w) @ term.vectors.T
        return a

    calls = []

    def solve(w, rhs, x0):
        calls.append((w, x0))
        return np.linalg.solve(matrix(w), rhs)

    system = AffineSystem(
        thetas=thetas,
        apply=lambda k_, x: pieces[k_] @ x,
        excitation=(AffineTerm(b, lambda w: 1j * w * np.eye(2)),),
        outputs=(AffineTerm(b, lambda _: np.eye(2)),),
        solve=solve,
        corrections=corrections,
    )

    def exact(w):
        x = np.linalg.solve(matrix(w), 1j * w * b)
        return x, b.T @ x

    return system, exact, calls


def _sweep(system, **kwargs):
    kwargs.setdefault("tolerance", 1e-8)
    kwargs.setdefault("max_samples", 40)
    kwargs.setdefault("memory", 2)
    return adaptive_sweep(system, GRID, **kwargs)


def test_greedy_samples_grid_points_from_the_endpoints_until_converged():
    system, _, _ = _system()
    sweep = _sweep(system)
    assert sweep.converged
    assert sweep.samples[:2] == (0, len(GRID) - 1)
    assert len(set(sweep.samples)) == len(sweep.samples)
    # a rational response with four poles in band: a handful of samples
    assert 4 <= len(sweep.samples) <= 16
    assert math.isinf(sweep.errors[0])
    assert math.isinf(sweep.errors[1])
    # memory: the last two samples confirm convergence
    assert all(e < 1e-8 for e in sweep.errors[-2:])
    assert sweep.errors[-3] >= 1e-8
    # two excitations, Re and Im: up to four basis columns per sample
    assert sweep.basis.shape == (N, sweep.model.size)
    assert sweep.model.size <= 4 * len(sweep.samples)
    np.testing.assert_allclose(
        sweep.basis.T @ sweep.basis, np.eye(sweep.model.size), atol=1e-12
    )


def test_reduced_outputs_match_full_solves_over_the_grid():
    system, exact, _ = _system()
    sweep = _sweep(system)
    worst = max(
        np.abs(sweep.outputs[i] - exact(w)[1]).max() / np.abs(exact(w)[1]).max()
        for i, w in enumerate(GRID)
    )
    assert worst < 1e-8
    # at a sample the reduced field is the full solution
    for i in sweep.samples:
        x, _ = exact(GRID[i])
        field = sweep.basis @ sweep.coordinates[i]
        np.testing.assert_allclose(field, x, atol=1e-11 * np.abs(x).max())


def test_a_low_rank_operator_term_is_projected_with_its_coefficient():
    system, exact, _ = _system(correction=True)
    sweep = _sweep(system)
    assert sweep.converged
    worst = max(
        np.abs(sweep.outputs[i] - exact(w)[1]).max() / np.abs(exact(w)[1]).max()
        for i, w in enumerate(GRID)
    )
    assert worst < 1e-8
    for i in sweep.samples:
        x, _ = exact(GRID[i])
        field = sweep.basis @ sweep.coordinates[i]
        np.testing.assert_allclose(field, x, atol=1e-11 * np.abs(x).max())
    # the reduced model's operator is V^T (A + U C U^T) V
    w = GRID[17]
    v = sweep.basis
    full = v.T @ (
        sum(t(w) * system.apply(k, v) for k, t in enumerate(system.thetas))
        + system.corrections[0].vectors
        @ system.corrections[0].coefficients(w)
        @ (system.corrections[0].vectors.T @ v)
    )
    np.testing.assert_allclose(
        sweep.model.matrix(w), full, atol=1e-12 * np.abs(full).max()
    )


def test_a_conjugated_output_term_projects_like_an_explicit_conjugate():
    system, *_ = _system()
    rng = np.random.default_rng(3)
    v = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
    rows = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    flagged = AffineTerm(v, lambda _: rows, conjugate=True)
    explicit = AffineTerm(np.conj(v), lambda _: rows)
    models = [
        ReducedModel(dataclasses.replace(system, outputs=(t,)))
        for t in (flagged, explicit)
    ]
    snapshots = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    for model in models:
        model.extend(snapshots[:, :1])
        model.extend(snapshots[:, 1:])
    np.testing.assert_allclose(
        models[0].outputs[0], models[1].outputs[0], rtol=1e-14, atol=0
    )
    x = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    np.testing.assert_allclose(
        flagged.project(x), np.conj(v).T @ x, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        flagged.combine(rows.T), np.conj(v) @ rows.T, rtol=1e-14, atol=1e-14
    )


def test_adjoint_loads_join_the_basis_at_every_sample():
    # An output functional g not parallel to the excitation: at a sample the
    # basis holds the solution, but the Galerkin adjoint of g is exact only
    # once A^-1 g is in the basis too.
    system, exact, calls = _system()
    rng = np.random.default_rng(7)
    g = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    picks = (0, 200, 80)
    plain = _sweep(system, samples=picks)
    calls.clear()
    enriched = _sweep(
        dataclasses.replace(system, adjoint_loads=lambda _: g[:, None]),
        samples=picks,
    )
    assert enriched.model.size == plain.model.size + 2 * len(picks)
    # one full solve per sample, the loads as extra right-hand-side columns
    assert len(calls) == len(picks)
    for i in picks:
        w = GRID[i]
        adjoint = np.linalg.solve(system_matrix(system, w).T, g)
        v = enriched.basis
        reduced = v @ np.linalg.solve(enriched.model.matrix(w).T, v.T @ g)
        np.testing.assert_allclose(reduced, adjoint, atol=1e-10 * abs(adjoint).max())
        x, _ = exact(w)
        np.testing.assert_allclose(
            enriched.basis @ enriched.coordinates[i], x, atol=1e-11 * abs(x).max()
        )
    # without the loads the reduced adjoint misses
    v = plain.basis
    w = GRID[80]
    adjoint = np.linalg.solve(system_matrix(system, w).T, g)
    reduced = v @ np.linalg.solve(plain.model.matrix(w).T, v.T @ g)
    assert np.linalg.norm(reduced - adjoint) > 1e-3 * np.linalg.norm(adjoint)


def system_matrix(system, w):
    eye = np.eye(N)
    return sum(t(w) * system.apply(k, eye) for k, t in enumerate(system.thetas))


def test_each_new_sample_starts_from_the_reduced_prediction():
    system, exact, calls = _system()
    sweep = _sweep(system)
    assert [w for w, _ in calls] == [GRID[i] for i in sweep.samples]
    assert calls[0][1] is None
    x0 = calls[-1][1]
    x, _ = exact(calls[-1][0])
    # the last sample confirmed convergence: its guess is already accurate
    assert np.linalg.norm(x0 - x) <= 1e-8 * np.linalg.norm(x)


def test_seeds_are_solved_right_after_the_endpoints_then_the_greedy_resumes():
    system, exact, calls = _system()
    sweep = _sweep(system, seeds=(120, 121, 0))
    # an endpoint among the seeds is not solved twice
    assert sweep.samples[:4] == (0, len(GRID) - 1, 120, 121)
    assert [w for w, _ in calls[:4]] == [GRID[i] for i in sweep.samples[:4]]
    assert len(set(sweep.samples)) == len(sweep.samples)
    # the seeds are measured like any greedy sample
    assert all(np.isfinite(sweep.errors[2:]))
    assert sweep.converged
    np.testing.assert_allclose(sweep.outputs[60], exact(GRID[60])[1], atol=1e-6, rtol=0)


def test_frozen_samples_ignore_the_seeds():
    system, _, _ = _system()
    picks = (100, 0, 200)
    assert _sweep(system, samples=picks, seeds=(120, 121)).samples == picks


def test_frozen_samples_are_solved_as_given_without_greedy():
    system, exact, calls = _system()
    picks = (100, 0, 200, 37)
    sweep = _sweep(system, samples=picks)
    assert sweep.samples == picks
    assert [w for w, _ in calls] == [GRID[i] for i in picks]
    assert math.isinf(sweep.errors[0])
    assert all(np.isfinite(sweep.errors[1:]))
    for i in picks:
        np.testing.assert_allclose(
            sweep.outputs[i], exact(GRID[i])[1], atol=1e-10, rtol=0
        )


def test_the_sample_cap_stops_the_greedy_with_a_warning():
    system, _, _ = _system()
    with pytest.warns(gfem.SweepNotConvergedWarning, match="adaptive_max_samples"):
        sweep = _sweep(system, max_samples=3)
    assert not sweep.converged
    assert len(sweep.samples) == 3


def test_an_exhausted_grid_is_exact_and_converged():
    system, exact, _ = _system()
    grid = GRID[::50]  # five frequencies
    sweep = adaptive_sweep(system, grid, tolerance=1e-12, max_samples=20, memory=5)
    assert sweep.converged
    assert sorted(sweep.samples) == list(range(5))
    for i, w in enumerate(grid):
        np.testing.assert_allclose(sweep.outputs[i], exact(w)[1], rtol=1e-10)


def test_one_frequency_is_one_sample():
    system, exact, _ = _system()
    sweep = adaptive_sweep(system, GRID[:1], tolerance=1e-6, max_samples=20, memory=2)
    assert sweep.samples == (0,)
    assert sweep.converged
    np.testing.assert_allclose(sweep.outputs[0], exact(GRID[0])[1], rtol=1e-10)


def test_a_rank_deficient_interpolation_warns_with_its_own_class():
    # three samples of one unknown span two columns: the interpolation has lost rank
    column = np.ones((1, 1), complex)
    with pytest.warns(gfem.SweepRankDeficientWarning, match="rank-deficient"):
        chosen = _next_sample(1j * GRID[:5], [0, 2, 4], [column] * 3, 1)
    assert chosen in (1, 3)
    assert issubclass(gfem.SweepRankDeficientWarning, UserWarning)
    assert not issubclass(gfem.SweepRankDeficientWarning, gfem.SweepNotConvergedWarning)
    assert issubclass(gfem.UnsampledPixelsWarning, UserWarning)


def _old_next_sample(z, chosen, snapshots, size):
    """The full-matrices SVD choice, kept as the reference for the thin one."""
    columns = []
    for i, c in zip(chosen, snapshots, strict=True):
        padded = np.zeros((size, c.shape[1]), complex)
        padded[: c.shape[0]] = c
        flat = padded.reshape(-1)
        columns.append(np.concatenate([flat, 1j * z[i] * flat]))
    u = np.column_stack(columns)
    _, sigma, vh = np.linalg.svd(u)
    sigma = np.concatenate([sigma, np.zeros(len(chosen) - len(sigma))])
    m = len(chosen) - 1
    while m > 0 and sigma[m] < 1e-12 * sigma[0]:
        m -= 1
    deficient = m < len(chosen) - 1
    q = vh[m].conj()
    zs = z[chosen]
    candidates = np.setdiff1d(np.arange(len(z)), chosen)
    denominator = np.abs((q[None, :] / (z[candidates, None] - zs[None, :])).sum(axis=1))
    return int(candidates[np.argmin(denominator)]), deficient


@pytest.mark.parametrize(
    ("rows", "chosen"),
    [(40, [0, 50, 100, 150, 200]), (2, [0, 40, 80, 120, 160, 200])],
    ids=["wide", "narrow"],
)
def test_the_next_sample_is_the_full_svd_s_choice(rows, chosen):
    # wide: rows 2np far above the samples; narrow: fewer rows than samples, so
    # the thin V^H has fewer rows and the padded sigma path runs
    import warnings

    rng = np.random.default_rng(3)
    z = 1j * GRID
    snapshots = [
        rng.standard_normal((rows, 2)) + 1j * rng.standard_normal((rows, 2))
        for _ in chosen
    ]
    expected, deficient = _old_next_sample(z, chosen, snapshots, rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _next_sample(z, chosen, snapshots, rows)
    warned = any(issubclass(w.category, gfem.SweepRankDeficientWarning) for w in caught)
    assert got == expected
    assert warned == deficient


def _old_extend(model, snapshot):
    """The one-column-at-a-time ``ReducedModel.extend``, as it was."""
    from gfem._sweep.adaptive import DROP

    columns = np.stack([snapshot.real, snapshot.imag], axis=2).reshape(
        snapshot.shape[0], -1
    )
    old = model.basis
    basis = old
    for column in columns.T:
        norm = np.linalg.norm(column)
        if norm == 0:
            continue
        w = column.copy()
        for _ in range(2):
            w -= basis @ (basis.T @ w)
        remaining = np.linalg.norm(w)
        if remaining <= DROP * norm:
            continue
        basis = np.column_stack([basis, w / remaining])
    new = basis[:, old.shape[1] :]
    if not new.shape[1]:
        return 0
    for k, reduced in enumerate(model.operators):
        product = np.asarray(model.system.apply(k, new))
        top = old.T @ product
        corner = new.T @ product
        corner = 0.5 * (corner + corner.T)
        model.operators[k] = np.block([[reduced, top], [top.T, corner]])
    model.excitation = [
        np.vstack([r, new.T @ t.vectors])
        for r, t in zip(model.excitation, model.system.excitation, strict=True)
    ]
    model.outputs = [
        np.hstack([r, t.project(new)])
        for r, t in zip(model.outputs, model.system.outputs, strict=True)
    ]
    model.corrections = [
        np.vstack([r, new.T @ t.vectors])
        for r, t in zip(model.corrections, model.system.corrections, strict=True)
    ]
    model.basis = basis
    return new.shape[1]


def test_a_block_extend_builds_the_per_column_basis():
    system, _, _ = _system(correction=True)
    rng = np.random.default_rng(5)
    first = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    # an exact duplicate column, a column inside DROP of the span, a zero
    # column and independent columns
    second = np.zeros((N, 5), complex)
    second[:, 0] = first[:, 0]
    second[:, 1] = first[:, 1] + 1e-13 * rng.standard_normal(N)
    second[:, 3] = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    second[:, 4] = rng.standard_normal(N)
    new, ref = ReducedModel(system), ReducedModel(system)
    for snapshot in (first, second):
        assert new.extend(snapshot) == _old_extend(ref, snapshot)
    assert new.size == ref.size
    scale = lambda a: np.abs(a).max()  # noqa: E731
    pairs = [(new.basis, ref.basis)]
    for name in ("operators", "excitation", "outputs", "corrections"):
        pairs += list(zip(getattr(new, name), getattr(ref, name), strict=True))
    for a, b in pairs:
        assert a.shape == b.shape
        assert scale(a - b) <= 1e-13 * scale(b)
