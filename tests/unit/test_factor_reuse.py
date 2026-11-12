"""Direct factors reused across designs under the amortized rule (ADR 0094).

The rule on synthetic and recorded costs, with no timing; then a serial PARDISO
handle with its gate forced (`DirectHandle._reuse_pays`), and left alone.
"""

import importlib.metadata

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache, quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._engine.system_operator import PlanSlot, SystemSweep
from gfem._linalg import Operator, prepare
from gfem._linalg.direct import DirectHandle
from gfem._linalg.factor_reuse import (
    FIRST_ITERATIONS,
    REPROBE_STEPS,
    REUSE_MIN_REFACTOR_SECONDS,
    REUSED_TOLERANCE,
    AmortizedRule,
)
from gfem._linalg.pieces import AssembledPieces, PieceSum

F = 1.0

# --- the rule ------------------------------------------------------------------------


def _refactorized_step(rule, *, t22=10.0, t33=0.1, columns=6, frequency=F):
    rule.refactorized(t22, frequency)
    rule.solved_direct(columns, columns * t33)


def _reused_step(rule, *, seconds, iterations=5, columns=6):
    rule.solved_reused(columns, iterations * columns, seconds)


def _next(rule, *, frequency=F, held=True, pays=None):
    return rule.next_step(
        frequency=frequency, held=held, pays=rule.pays if pays is None else pays
    )


def test_after_a_refactorization_the_next_step_reuses():
    rule = AmortizedRule()
    _refactorized_step(rule)
    assert _next(rule)


def test_reuse_continues_while_the_last_reused_step_costs_less_than_the_mean():
    rule = AmortizedRule()
    _refactorized_step(rule)  # 10.6 s
    assert _next(rule)
    _reused_step(rule, seconds=3.0)  # mean (10.6 + 3) / 2 = 6.8
    assert _next(rule)
    _reused_step(rule, seconds=6.0)  # mean 6.53: 6 below it
    assert _next(rule)


def test_a_reused_step_at_the_mean_refactorizes():
    rule = AmortizedRule()
    _refactorized_step(rule, t22=10.0, t33=0.1)  # 10.6 s
    assert _next(rule)
    _reused_step(rule, seconds=10.6)  # mean (10.6 + 10.6) / 2: at it
    assert not _next(rule)


def test_the_recorded_16_thread_run_gives_its_refactorization_pattern():
    # ticket 02's amortized run on the 2.49M-DOF 3D bend, 16 threads
    # (probes/02/all/t16-amort): t22, a step's direct seconds; then each
    # reused step's GMRES seconds and iterations per column, 6 columns
    chains = [
        (75.79, 5.1, [(71.5, 14), (96.7, 19)]),
        (76.37, 5.0, [(56.0, 11), (76.1, 15)]),
        (76.38, 5.1, [(50.7, 10), (61.1, 12), (68.0, 13)]),
        (76.45, 5.1, [(35.5, 7), (45.6, 9), (56.0, 11)]),
        (76.33, 5.1, [(40.8, 8), (50.8, 10), (60.8, 12)]),
        (76.33, 5.0, [(46.0, 9), (61.1, 12), (86.7, 17)]),
    ]
    rule = AmortizedRule()
    steps = []
    for t22, direct, reused in chains:
        steps.append("F" if not steps else ("u" if _next(rule) else "R"))
        rule.refactorized(t22, F)
        rule.solved_direct(6, direct)
        for seconds, iterations in reused:
            steps.append("u" if _next(rule) else "R")
            rule.solved_reused(6, 6 * iterations, seconds)
    assert "".join(steps) == "FuuRuuRuuuRuuuRuuuRuuu"


def test_the_gate_is_off_while_r_over_c_is_below_the_first_reused_iterations():
    rule = AmortizedRule()
    # R = 10 / 0.25 = 40, C = 6: R/C = 6.7, below the starting I1 = 10
    _refactorized_step(rule, t22=10.0, t33=0.25)
    assert not _next(rule)


def test_the_gate_is_on_at_r_over_c_at_least_the_first_reused_iterations():
    rule = AmortizedRule()
    # R/C = 10 / 0.1 / 6 = 16.7 >= 10
    _refactorized_step(rule, t22=10.0, t33=0.1)
    assert _next(rule)


def test_the_first_reused_step_of_a_chain_sets_the_iterations_the_gate_compares():
    rule = AmortizedRule()
    _refactorized_step(rule, t22=10.0, t33=0.1)  # R/C 16.7
    assert _next(rule)
    _reused_step(rule, seconds=2.0, iterations=20)  # I1 = 20 > 16.7
    assert not _next(rule)


def test_a_closed_gate_reprobes_after_five_refactorized_steps_and_refreshes_i1():
    # the closing run at 16 threads: one reused step at I1 above R/C closed
    # the gate, and nothing measured I1 again
    rule = AmortizedRule()
    _refactorized_step(rule, t22=10.0, t33=0.1)  # R/C 16.7
    assert _next(rule)
    _reused_step(rule, seconds=2.0, iterations=20)  # I1 = 20: closed
    for _ in range(REPROBE_STEPS):
        assert not _next(rule)
        _refactorized_step(rule, t22=10.0, t33=0.1)
    assert _next(rule)  # the probe
    _reused_step(rule, seconds=2.0, iterations=8)
    assert _next(rule)  # I1 = 8 <= R/C 16.7: open again
    assert rule.first_iterations == 8


def test_the_reprobe_waits_five_refactorized_steps():
    assert REPROBE_STEPS == 5
    rule = AmortizedRule()
    _refactorized_step(rule, t22=10.0, t33=0.25)  # R/C 6.7 < 10
    for _ in range(REPROBE_STEPS - 1):
        assert not _next(rule)
        _refactorized_step(rule, t22=10.0, t33=0.25)
    assert not _next(rule)  # the fifth closed step
    _refactorized_step(rule, t22=10.0, t33=0.25)
    assert _next(rule)


def test_a_gate_closed_by_the_floor_never_reprobes():
    rule = AmortizedRule()
    for _ in range(3 * REPROBE_STEPS):
        _refactorized_step(rule, t22=0.5 * REUSE_MIN_REFACTOR_SECONDS, t33=1e-5)
        assert not _next(rule)


def test_the_gate_is_off_when_the_last_refactorization_took_under_the_floor():
    rule = AmortizedRule()
    # R/C = 1e4, but the refactorization took half a second
    _refactorized_step(rule, t22=0.5 * REUSE_MIN_REFACTOR_SECONDS, t33=1e-5)
    assert not _next(rule)


def test_another_frequency_refactorizes():
    rule = AmortizedRule()
    _refactorized_step(rule, frequency=1.0)
    assert not _next(rule, frequency=2.0)


def test_a_frequency_not_told_refactorizes():
    rule = AmortizedRule()
    _refactorized_step(rule)
    assert not _next(rule, frequency=None)


def test_released_factors_refactorize():
    rule = AmortizedRule()
    _refactorized_step(rule)
    assert not _next(rule, held=False)


def test_a_forced_gate_still_needs_a_refactorization_to_start_a_chain():
    # the first factorization (the handle's creation) is not timed
    rule = AmortizedRule()
    assert not _next(rule, pays=lambda: True)


def test_ending_the_chain_refactorizes_the_next_step():
    rule = AmortizedRule()
    _refactorized_step(rule)
    rule.end_chain()
    assert not _next(rule, pays=lambda: True)


def test_the_cap_is_r_over_c_rounded_up_plus_one():
    rule = AmortizedRule()
    _refactorized_step(rule, t22=10.0, t33=0.1, columns=6)  # R/C = 16.67
    assert _next(rule)
    assert rule.cap == 18


def test_a_capped_reused_step_starts_a_new_chain():
    rule = AmortizedRule()
    _refactorized_step(rule)
    assert _next(rule)
    _reused_step(rule, seconds=1.0, columns=1)
    # the handle refactorizes at the new matrix and finishes direct
    rule.refactorized(12.0, F)
    rule.solved_direct(5, 0.5)
    assert _next(rule)  # a new chain: its first reused step
    _reused_step(rule, seconds=11.0)  # mean (12 + 6 * 0.1 + 11) / 2 = 11.8
    assert _next(rule)


def test_the_reused_stop_is_a_twentieth_of_the_tolerance():
    assert REUSED_TOLERANCE == 0.05


# --- a PARDISO handle ----------------------------------------------------------------


def _mkl_installed() -> bool:
    try:
        importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_mkl = pytest.mark.skipif(not _mkl_installed(), reason="needs the mkl extra")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (3, 3, 3)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _helmholtz(built, k2, *, complex_valued=True, dirichlet="bottom|top"):
    """A complex symmetric (or real symmetric) operator ``-Δ - k2``."""
    space = ngs.H1(built.mesh, order=2, dirichlet=dirichlet, complex=complex_valued)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) - k2 * u * v) * quadrature.dx(2)
    return Operator(form, space.FreeDofs()), space


def _loads(space, columns=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (space.ndof, columns)
    B = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return B * np.asarray([space.FreeDofs()[i] for i in range(space.ndof)])[:, None]


def _fresh(built, k2, B):
    op, _ = _helmholtz(built, k2)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct")
    try:
        return handle.solve(B)
    finally:
        handle.free()


def _relative_residual(handle, X, B):
    free = handle._free[:, None]
    R = np.where(free, handle.apply(X) - B, 0)
    return np.linalg.norm(R, axis=0) / np.linalg.norm(B, axis=0)


@pytest.fixture
def forced(monkeypatch):
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: True)


def _chain(built, tolerance=1e-6):
    """A handle refactorized once at ``k2 = 1 - 1j`` (a chain starts), and loads.

    Prepared at another design: a refactorization on unchanged values is none.
    """
    op, space = _helmholtz(built, 2.0 - 1j)
    handle = DirectHandle.from_operator(op, "complex_symmetric", tolerance=tolerance)
    handle.refactorize(_helmholtz(built, 1.0 - 1j)[0], frequency=F)
    B = _loads(space)
    handle.solve(B)
    return handle, B


@pytest.fixture
def uncapped(monkeypatch):
    # a test-sized R/C gives a cap of a few iterations, which the timing sets
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 100))


@needs_mkl
@pytest.mark.usefixtures("forced", "uncapped")
def test_a_nearby_design_is_solved_on_the_kept_factors_to_a_twentieth_of_the_tolerance(
    built,
):
    tolerance = 1e-6
    handle, B = _chain(built, tolerance)
    near = 1.0 - 1j + 1e-3
    handle.refactorize(_helmholtz(built, near)[0], frequency=F)
    X, counts = handle.solve_counted(B)
    assert len(counts) == B.shape[1]
    assert all(c > 0 for c in counts)
    assert np.all(_relative_residual(handle, X, B) <= REUSED_TOLERANCE * tolerance)
    exact = _fresh(built, near, B)
    assert np.linalg.norm(X - exact) <= tolerance * np.linalg.norm(exact)
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_a_column_past_the_cap_refactorizes_and_finishes_direct(built, monkeypatch):
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 3))
    handle, B = _chain(built)
    far = 30.0 + 5j  # far from 1 - 1j: no Krylov solve converges in the cap
    handle.refactorize(_helmholtz(built, far)[0], frequency=F)
    X, counts = handle.solve_counted(B)
    exact = _fresh(built, far, B)
    np.testing.assert_allclose(X, exact, rtol=0, atol=1e-12 * np.abs(exact).max())
    # the first column reached the cap; the rest were solved direct
    assert counts[0] > 0
    assert counts[1:] == (0,) * (B.shape[1] - 1)
    # later solves on the step are direct
    assert handle.solve_counted(B)[1] == ()
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_a_capped_column_counts_once_in_its_step(built, monkeypatch):
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 3))
    handle, B = _chain(built)
    far = 30.0 + 5j
    handle.refactorize(_helmholtz(built, far)[0], frequency=F)
    handle.solve_counted(B)
    # the next step closes this one: C is its columns
    handle.refactorize(_helmholtz(built, far + 1e-3)[0], frequency=F)
    assert handle._rule.columns == B.shape[1]
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced", "uncapped")
def test_the_same_design_again_is_no_reused_step(built):
    handle, B = _chain(built)
    handle.refactorize(_helmholtz(built, 1.0 - 1j)[0], frequency=F)
    # the factors are the design's: solved direct, in the step in progress
    assert handle.solve_counted(B)[1] == ()
    near = 1.0 - 1j + 1e-3
    handle.refactorize(_helmholtz(built, near)[0], frequency=F)
    assert handle._rule.first_iterations == FIRST_ITERATIONS
    X, counts = handle.solve_counted(B)
    assert all(c > 0 for c in counts)
    exact = _fresh(built, near, B)
    assert np.linalg.norm(X - exact) <= 1e-6 * np.linalg.norm(exact)
    handle.free()


def _piece_sums(built):
    """``k2 -> -Δ - k2`` as a piece sum on one set of pieces, and its loads."""
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) - u * v) * quadrature.dx(2)
    pieces = AssembledPieces(
        space,
        (
            lambda u, v: ngs.grad(u) * ngs.grad(v) * quadrature.dx(2),
            lambda u, v: u * v * quadrature.dx(2),
        ),
    )

    def at(k2):
        return Operator(form, space.FreeDofs(), PieceSum(pieces, (1.0, -k2)))

    return at, _loads(space)


@pytest.fixture
def sums(monkeypatch):
    """The number of piece sums summed so far."""
    count = [0]
    sum_into = AssembledPieces.sum_into

    def counted(self, weights, matrix):
        count[0] += 1
        sum_into(self, weights, matrix)

    monkeypatch.setattr(AssembledPieces, "sum_into", counted)
    return count


@needs_mkl
@pytest.mark.usefixtures("forced", "uncapped")
def test_a_new_piece_sum_is_summed_once_whether_reused_or_refactorized(built, sums):
    at, B = _piece_sums(built)
    handle = DirectHandle.from_operator(at(2.0 - 1j), "complex_symmetric")
    handle.refactorize(at(1.0 - 1j), frequency=F)
    handle.solve(B)
    sums[0] = 0
    handle.refactorize(at(1.0 - 1j + 1e-3), frequency=F)
    assert all(c > 0 for c in handle.solve_counted(B)[1])  # reused
    assert sums[0] == 1
    handle.refactorize(at(1.0 - 1j + 2e-3), frequency=2 * F)
    assert handle.solve_counted(B)[1] == ()  # refactorized
    assert sums[0] == 2
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced", "uncapped")
def test_the_same_piece_sum_again_is_not_summed_and_is_no_step(built, sums):
    at, B = _piece_sums(built)
    handle = DirectHandle.from_operator(at(2.0 - 1j), "complex_symmetric")
    handle.refactorize(at(1.0 - 1j), frequency=F)
    handle.solve(B)
    near = 1.0 - 1j + 1e-3
    handle.refactorize(at(near), frequency=F)
    first = handle.solve_counted(B)[1]
    sums[0] = 0
    handle.refactorize(at(near), frequency=F)
    assert sums[0] == 0
    # the reused step in progress goes on: its iterations, not ~1 per column
    again = handle.solve_counted(B)[1]
    assert sum(again) >= sum(first) // 2
    handle.refactorize(at(near + 1e-3), frequency=F)
    assert handle._rule.first_iterations == sum(first + again) / (2 * B.shape[1])
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_a_release_ends_the_chain(built):
    handle, B = _chain(built)
    handle.release_factors()
    near = 1.0 - 1j + 1e-3
    handle.refactorize(_helmholtz(built, near)[0], frequency=F)
    X, counts = handle.solve_counted(B)
    assert counts == ()
    np.testing.assert_allclose(X, _fresh(built, near, B), rtol=1e-12, atol=0)
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_a_refactorization_after_reused_steps_solves_exactly(built):
    # the gather index was built from the factored values, not a reused write
    handle, B = _chain(built)
    for shift in (1e-3, 2e-3):
        handle.refactorize(_helmholtz(built, 1.0 - 1j + shift)[0], frequency=F)
        handle.solve(B)
    handle.release_factors()  # ends the chain: the next step refactorizes
    far = 3.0 - 1j
    handle.refactorize(_helmholtz(built, far)[0], frequency=F)
    X, counts = handle.solve_counted(B)
    assert counts == ()
    np.testing.assert_allclose(X, _fresh(built, far, B), rtol=1e-12, atol=0)
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_another_frequency_refactorizes_even_with_the_gate_forced(built):
    handle, B = _chain(built)
    near = 1.0 - 1j + 1e-3
    handle.refactorize(_helmholtz(built, near)[0], frequency=2 * F)
    assert handle.solve_counted(B)[1] == ()
    handle.free()


@needs_mkl
@pytest.mark.usefixtures("forced")
def test_a_general_operator_never_reuses(built):
    op, space = _helmholtz(built, 1.0 - 1j)
    handle = DirectHandle.from_operator(op, "general")
    handle.refactorize(_helmholtz(built, 1.0 - 1j)[0], frequency=F)
    B = _loads(space)
    handle.solve(B)
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 1e-3)[0], frequency=F)
    assert handle.solve_counted(B)[1] == ()
    handle.free()


@needs_mkl
def test_a_small_model_never_reuses_without_the_gate_forced(built):
    # the floor: its refactorization takes milliseconds
    handle, B = _chain(built)
    for shift in (1e-3, 2e-3, 3e-3):
        handle.refactorize(_helmholtz(built, 1.0 - 1j + shift)[0], frequency=F)
        assert handle.solve_counted(B)[1] == ()
    handle.free()


@needs_mkl
def test_a_handle_taken_from_the_pool_solves_at_the_taking_sweeps_tolerance(built):
    # a backward prepared it at the adjoint tolerance; a forward takes it next
    op, _ = _helmholtz(built, 1.0 - 1j)
    adjoint = DirectHandle.from_operator(op, "complex_symmetric", tolerance=1e-3)
    topology = object()
    slot = PlanSlot(("site",), topology, budget=2**40)
    cache.ENTRIES.clear()
    try:
        cache.ENTRIES.give_plan(slot.key(), topology, adjoint, slot.budget)
        forward = SystemSweep(
            lambda _: _helmholtz(built, 1.0 - 1j + 1e-3)[0],
            lambda base, _: base,
            lambda _: pytest.fail("the pooled handle serves"),
            plans=slot,
            tolerance=1e-8,
        )
        assert forward.at(F) is adjoint
        assert adjoint.tolerance == 1e-8
    finally:
        cache.ENTRIES.clear()
