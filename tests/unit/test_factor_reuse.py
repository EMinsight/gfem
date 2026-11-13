"""Direct factors reused across designs under the amortized rule (ADR 0094).

The rule on synthetic and recorded counts, with no clock; the counts PARDISO
and sparse Cholesky report; then a direct handle (PARDISO or sparse Cholesky)
with its gate forced (`DirectHandle._reuse_pays`), and left alone.
"""

import importlib.metadata
import itertools
from types import SimpleNamespace

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache, quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._engine.system_operator import PlanSlot, SystemSweep
from gfem._linalg import Operator, direct, prepare
from gfem._linalg.direct import DirectHandle, UpdateFactors
from gfem._linalg.factor_reuse import (
    FIRST_ITERATIONS,
    REPROBE_STEPS,
    REUSED_TOLERANCE,
    AmortizedRule,
    kappa,
)
from gfem._linalg.pardiso import PardisoFactors, factor_nonzeros
from gfem._linalg.pieces import AssembledPieces, PieceSum

F = 1.0

# --- the rule ------------------------------------------------------------------------


def _rule(ratio=100.0, *, above_floor=True):
    """A rule on a plan whose refactorization costs ``ratio`` direct columns."""
    return AmortizedRule(ratio, above_floor=above_floor)


def _refactorized_step(rule, *, columns=6, frequency=F):
    rule.refactorized(frequency)
    rule.solved_direct(columns)


def _reused_step(rule, *, iterations=5, columns=6):
    rule.solved_reused(columns, iterations * columns)


def _next(rule, *, frequency=F, held=True, pays=None):
    return rule.next_step(
        frequency=frequency, held=held, pays=rule.pays if pays is None else pays
    )


def test_after_a_refactorization_the_next_step_reuses():
    rule = _rule()
    _refactorized_step(rule)
    assert _next(rule)


def test_reuse_continues_while_the_last_reused_step_costs_less_than_the_mean():
    rule = _rule()
    _refactorized_step(rule)  # 100 + 6 columns
    assert _next(rule)
    _reused_step(rule, iterations=5)  # 30: mean (106 + 30) / 2 = 68
    assert _next(rule)
    _reused_step(rule, iterations=10)  # 60: mean 65.3, 60 below it
    assert _next(rule)


def test_a_reused_step_at_the_mean_refactorizes():
    rule = _rule(94.0)
    _refactorized_step(rule)  # 94 + 6 = 100 columns
    assert _next(rule)
    rule.solved_reused(6, 20)  # I1 3.3 keeps the gate open; mean 60
    assert _next(rule)
    rule.solved_reused(6, 60)  # mean (100 + 20 + 60) / 3 = 60: at it
    assert not _next(rule)
    assert rule.reason == "chain"


# ticket 02's amortized run on the 2.49M-DOF 3D bend, 16 threads
# (probes/02/all/t16-amort): each chain's reused steps, in GMRES iterations
# per column of 6 columns; its measured R = t22 / t33 was 89-92 columns
RECORDED_CHAINS = (
    (14, 19),
    (11, 15),
    (10, 12, 13),
    (7, 9, 11),
    (8, 10, 12),
    (9, 12, 17),
)


def _recorded_decisions(ratio):
    rule = _rule(ratio)
    steps = []
    for reused in RECORDED_CHAINS:
        steps.append("F" if not steps else ("u" if _next(rule) else "R"))
        _refactorized_step(rule)
        for iterations in reused:
            steps.append("u" if _next(rule) else "R")
            _reused_step(rule, iterations=iterations)
    return "".join(steps)


def test_the_recorded_16_thread_run_gives_its_refactorization_pattern():
    assert _recorded_decisions(90.0) == "FuuRuuRuuuRuuuRuuuRuuu"


def test_the_same_counts_give_the_same_decisions():
    # a pure function of the counts: no clock, so a script repeats them
    assert _recorded_decisions(90.0) == _recorded_decisions(90.0)


@pytest.mark.parametrize("s", [0.95, 1.05])
def test_a_model_ratio_near_the_measured_one_gives_the_same_decisions(s):
    # R-hat = s R: a calibration a few percent off chooses as the measured R
    assert _recorded_decisions(s * 90.0) == _recorded_decisions(90.0)


def test_the_gate_is_off_while_r_over_c_is_below_the_first_reused_iterations():
    rule = _rule(25.0)  # R/C = 25 / 6 = 4.2, below the starting I1 = 5
    _refactorized_step(rule)
    assert not _next(rule)


def test_the_gate_starts_at_five_iterations():
    # ticket 15: the fewest iterations per column measured at lag 1 (CPW, 5)
    assert FIRST_ITERATIONS == 5.0
    rule = _rule(36.0)  # R/C = 6: the 4-port CPWs reuse from the second step
    _refactorized_step(rule)
    assert _next(rule)


def test_the_gate_is_on_at_r_over_c_at_least_the_first_reused_iterations():
    rule = _rule(100.0)  # R/C = 16.7 >= 5
    _refactorized_step(rule)
    assert _next(rule)


def test_the_first_reused_step_of_a_chain_sets_the_iterations_the_gate_compares():
    rule = _rule()  # R/C 16.7
    _refactorized_step(rule)
    assert _next(rule)
    _reused_step(rule, iterations=20)  # I1 = 20 > 16.7
    assert not _next(rule)


def test_a_closed_gate_reprobes_after_five_refactorized_steps_and_refreshes_i1():
    # the closing run at 16 threads: one reused step at I1 above R/C closed
    # the gate, and nothing measured I1 again
    rule = _rule()  # R/C 16.7
    _refactorized_step(rule)
    assert _next(rule)
    _reused_step(rule, iterations=20)  # I1 = 20: closed
    for _ in range(REPROBE_STEPS):
        assert not _next(rule)
        _refactorized_step(rule)
    assert _next(rule)  # the probe
    _reused_step(rule, iterations=8)
    assert _next(rule)  # I1 = 8 <= R/C 16.7: open again
    assert rule.first_iterations == 8


def test_the_reprobe_waits_five_refactorized_steps():
    assert REPROBE_STEPS == 5
    rule = _rule(25.0)  # R/C 4.2 < 5
    _refactorized_step(rule)
    for _ in range(REPROBE_STEPS - 1):
        assert not _next(rule)
        _refactorized_step(rule)
    assert not _next(rule)  # the fifth closed step
    _refactorized_step(rule)
    assert _next(rule)


def test_a_probe_that_leaves_the_gate_closed_doubles_the_wait():
    # a model whose R/C never reaches the lag-1 iterations: each probe saves
    # nothing; the probes thin out instead of coming every sixth step
    rule = _rule(25.0)  # R/C 4.2 < 5
    _refactorized_step(rule)
    opened = []
    for _ in range(60):
        reused = _next(rule)
        opened.append(reused)
        if reused:
            _reused_step(rule, iterations=12)  # I1 12 > 4.2
        else:
            _refactorized_step(rule)
    probes = [k for k, reused in enumerate(opened) if reused]
    assert probes[0] == REPROBE_STEPS
    gaps = [b - a for a, b in itertools.pairwise(probes)]
    assert gaps == sorted(gaps)
    assert gaps[0] < gaps[-1]
    assert len(probes) <= 4


def test_a_gate_closed_by_the_floor_never_reprobes():
    rule = _rule(1e4, above_floor=False)
    for _ in range(3 * REPROBE_STEPS):
        _refactorized_step(rule)
        assert not _next(rule)


def test_the_gate_is_off_on_a_plan_below_the_floor():
    rule = _rule(1e4, above_floor=False)  # R/C 1667, but a small model
    _refactorized_step(rule)
    assert not _next(rule)


def test_another_frequency_refactorizes():
    rule = _rule()
    _refactorized_step(rule, frequency=1.0)
    assert not _next(rule, frequency=2.0)


def test_a_frequency_not_told_refactorizes():
    rule = _rule()
    _refactorized_step(rule)
    assert not _next(rule, frequency=None)


def test_released_factors_refactorize():
    rule = _rule()
    _refactorized_step(rule)
    assert not _next(rule, held=False)


def test_a_forced_gate_still_needs_a_refactorization_to_start_a_chain():
    # the first factorization (the handle's creation) is no step
    rule = _rule()
    assert not _next(rule, pays=lambda: True)


def test_ending_the_chain_refactorizes_the_next_step():
    rule = _rule()
    _refactorized_step(rule)
    rule.end_chain()
    assert not _next(rule, pays=lambda: True)


def test_the_cap_is_r_over_c_rounded_up_plus_one():
    rule = _rule(100.0)
    _refactorized_step(rule, columns=6)  # R/C = 16.67
    assert _next(rule)
    assert rule.cap == 18


def test_a_capped_reused_step_starts_a_new_chain():
    rule = _rule()
    _refactorized_step(rule)
    assert _next(rule)
    _reused_step(rule, iterations=18, columns=1)
    # the handle refactorizes at the new matrix and finishes direct
    rule.refactorized(F)
    rule.solved_direct(5)
    assert _next(rule)  # a new chain (100 + 6 columns): its first reused step
    _reused_step(rule, iterations=11)  # 66: mean (106 + 66) / 2 = 86
    assert _next(rule)


def test_the_reused_stop_is_a_twentieth_of_the_tolerance():
    assert REUSED_TOLERANCE == 0.05


def _reason(rule, **kwargs):
    reused = _next(rule, **kwargs)
    return None if reused else rule.reason


def test_each_refactorized_step_reports_the_first_failing_condition():
    # the order of next_step: chain started, held, frequency, gate, chain stop
    rule = _rule()
    assert _reason(rule, pays=lambda: True) == "first"
    _refactorized_step(rule)
    assert _reason(rule, held=False) == "factors released (retain_memory)"
    rule = _rule()
    _refactorized_step(rule, frequency=1.0)
    assert _reason(rule, frequency=2.0, held=False) == (
        "factors released (retain_memory)"
    )
    _refactorized_step(rule, frequency=1.0)
    assert _reason(rule, frequency=2.0, pays=lambda: False) == "other frequency"
    _refactorized_step(rule)
    assert _reason(rule, frequency=None) == "other frequency"


def test_a_closed_gate_reports_the_floor_before_the_ratio():
    rule = _rule(1e4, above_floor=False)
    _refactorized_step(rule)
    assert _reason(rule) == "below floor"
    rule = _rule(25.0)
    _refactorized_step(rule)  # R/C 4.2 < I1 = 5
    assert _reason(rule) == "gate"
    assert rule.ratio == pytest.approx(25.0 / 6)
    assert rule.first_iterations == FIRST_ITERATIONS


def test_a_chain_past_its_mean_reports_chain():
    rule = _rule(94.0)
    _refactorized_step(rule)  # 100 columns
    assert _reason(rule) is None
    rule.solved_reused(6, 20)
    assert _reason(rule) is None
    rule.solved_reused(6, 70)  # mean 63.3
    assert _reason(rule) == "chain"


def test_a_release_or_a_new_plan_ending_the_chain_names_its_cause():
    rule = _rule()
    _refactorized_step(rule)
    rule.end_chain(released=True)
    assert _reason(rule, held=False) == "factors released (retain_memory)"
    _refactorized_step(rule)
    rule.end_chain()
    assert _reason(rule) == "first"


def test_a_capped_column_reports_capped():
    rule = _rule()
    _refactorized_step(rule)
    assert _next(rule)
    assert rule.reason is None
    rule.refactorized(F)
    assert rule.reason == "capped"


# --- the counts ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("threads", "expected"),
    [
        (1, 0.0113),  # a table entry
        (6, 0.0075 + (0.0060 - 0.0075) * np.log(6 / 4) / np.log(2)),  # 4 to 8
        (12, 0.0060 + (0.0037 - 0.0060) * np.log(12 / 8) / np.log(2)),  # 8 to 16
        (112, 0.0018),  # at or past 56
    ],
)
def test_kappa_is_interpolated_log_linearly_in_the_thread_count(threads, expected):
    assert kappa(threads) == pytest.approx(expected, rel=1e-12)


def test_a_factor_nonzero_count_past_int32_reads_unsigned():
    # the 2.49M-DOF bend at 8 threads (MN5 job 49006702): iparm[17] and [16]
    assert factor_nonzeros(-2065854147, 36411405) == 2_229_113_149


def test_a_factor_nonzero_count_wrapped_past_2_32_is_corrected_by_the_factor_size():
    true = 3 * 2**32 + 12345
    wrapped = (true + 2**31) % 2**32 - 2**31  # as int32
    kib = true * 18 // 1024  # about 18 B per complex entry, as measured
    assert factor_nonzeros(wrapped, kib) == true
    # real entries take 8 B
    assert factor_nonzeros(wrapped, true * 9 // 1024, entry_bytes=8) == true


def _pardiso(mflop, nnz, kib, threads):
    params = [0] * 64
    params[16], params[17], params[18] = kib, nnz, mflop
    inverse = SimpleNamespace(_params=params, _gfem_threads=threads)
    return PardisoFactors(inverse, np.ones(10, dtype=bool))


def test_pardiso_counts_give_the_model_ratio_and_the_floor():
    # the 2.49M-DOF bend: F / nnz(L) = 20900, 8 threads: R-hat about 125
    factors = _pardiso(46587563, -2065854147, 36411405, 8)
    assert factors.model_ratio() == pytest.approx(0.0060 * 46587563e6 / 2229113149)
    assert factors.above_floor()
    assert _pardiso(100_000, 10**6, 10**4, 8).above_floor()
    assert not _pardiso(99_999, 10**6, 10**4, 8).above_floor()


def test_sparse_cholesky_counts_give_the_model_ratio_and_the_floor():
    factors = UpdateFactors(SimpleNamespace(nze=5.3e8), 381_134)
    assert factors.model_ratio() == pytest.approx(0.14 * 5.3e8 / 381_134)
    assert factors.above_floor()
    assert UpdateFactors(SimpleNamespace(nze=2e7), 10**5).above_floor()
    assert not UpdateFactors(SimpleNamespace(nze=2e7 - 1), 10**5).above_floor()


# --- a direct handle -----------------------------------------------------------------


def _mkl_installed() -> bool:
    try:
        importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_mkl = pytest.mark.skipif(not _mkl_installed(), reason="needs the mkl extra")


@pytest.fixture(
    params=[pytest.param("pardiso", marks=needs_mkl), "sparsecholesky"],
)
def backend(request, monkeypatch):
    """The direct handle's factorization: PARDISO, or the fallback without MKL."""
    if request.param == "sparsecholesky":
        monkeypatch.setattr(direct, "pardiso_inverse", lambda *_, **__: None)
    return request.param


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
    # a test-sized R/C gives a cap of a few iterations, which the counts set
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 100))


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
@pytest.mark.usefixtures("forced", "uncapped")
def test_a_handle_names_each_steps_action_reason_and_numbers(built):
    def step(handle):
        return handle.last_action, handle.last_reason

    op, space = _helmholtz(built, 2.0 - 1j)
    handle = DirectHandle.from_operator(op, "complex_symmetric")
    assert step(handle) == ("factorized", "first")
    handle.refactorize(_helmholtz(built, 1.0 - 1j)[0], frequency=F)
    assert step(handle) == ("refactorized", "first")
    B = _loads(space)
    handle.solve(B)
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 1e-3)[0], frequency=F)
    assert step(handle) == ("reused", None)
    handle.solve(B)
    # the same values again: no step, nothing changes
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 1e-3)[0], frequency=F)
    assert step(handle) == ("reused", None)
    handle.note_release(base_bytes=123, budget_bytes=45)
    handle.release_factors()
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 2e-3)[0], frequency=F)
    assert step(handle) == ("refactorized", "factors released (retain_memory)")
    assert handle.last_details == {"base_bytes": 123, "budget_bytes": 45}
    handle.solve(B)
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 3e-3)[0], frequency=2 * F)
    assert step(handle) == ("refactorized", "other frequency")
    assert handle.last_details == {}
    handle.free()


@pytest.mark.usefixtures("backend")
def test_a_handle_counts_its_plan_and_a_small_model_is_below_the_floor(built):
    handle, _ = _chain(built)
    assert handle._rule.model_ratio > 0
    assert not handle._rule.above_floor
    handle.free()


@pytest.mark.usefixtures("backend")
def test_a_closed_gate_gives_its_ratio_and_first_iterations(built, monkeypatch):
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: False)
    handle, _ = _chain(built)
    handle._rule.above_floor = True  # as a large model
    handle.refactorize(_helmholtz(built, 1.0 - 1j + 1e-3)[0], frequency=F)
    assert handle.last_reason == "gate"
    assert set(handle.last_details) == {"ratio", "first_iterations"}
    assert handle.last_details["first_iterations"] == FIRST_ITERATIONS
    handle.free()


@pytest.mark.usefixtures("backend")
@pytest.mark.usefixtures("forced")
def test_a_capped_column_names_its_step(built, monkeypatch):
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 3))
    handle, B = _chain(built)
    handle.refactorize(_helmholtz(built, 30.0 + 5j)[0], frequency=F)
    assert handle.last_action == "reused"
    handle.solve(B)
    assert (handle.last_action, handle.last_reason) == (
        "reused, capped, refactorized",
        "capped",
    )
    handle.free()


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
@pytest.mark.usefixtures("forced")
def test_another_frequency_refactorizes_even_with_the_gate_forced(built):
    handle, B = _chain(built)
    near = 1.0 - 1j + 1e-3
    handle.refactorize(_helmholtz(built, near)[0], frequency=2 * F)
    assert handle.solve_counted(B)[1] == ()
    handle.free()


@pytest.mark.usefixtures("backend")
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


@pytest.mark.usefixtures("backend")
def test_a_small_model_never_reuses_without_the_gate_forced(built):
    # the floor: its refactorization takes milliseconds
    handle, B = _chain(built)
    for shift in (1e-3, 2e-3, 3e-3):
        handle.refactorize(_helmholtz(built, 1.0 - 1j + shift)[0], frequency=F)
        assert handle.solve_counted(B)[1] == ()
    handle.free()


@pytest.mark.usefixtures("backend")
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
