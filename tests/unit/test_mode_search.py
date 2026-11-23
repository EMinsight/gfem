"""The mode search on stub operators: selection, twins and the gap set.

`ModeSearch.find` runs on an operator with no mesh: its eigen solver returns
fixed Ritz pairs and its refinement maps a seed to a chosen mode, so the
selection's twin, guard and gap-set decisions show directly.
"""

import dataclasses
import logging
import math

import numpy as np
import pytest

import gfem
from gfem._eigen.reach import GapSet
from gfem._eigen.refine import Refined
from gfem._eigen.search import ModeSearch, _in_span
from gfem._eigen.seed import SeedBoundReached
from gfem._linalg.layout import ONE_PROCESS


@dataclasses.dataclass
class _Stub:
    """An operator that only holds its shift; damped, so no real snap."""

    shift: complex
    nonpolynomial: tuple = ()
    layout = ONE_PROCESS
    seed_damped = True
    general = False
    scale = 1e-3

    def reshift(self, k, **_):
        self.shift = k


def _search(n_modes, refine, ritz, vectors) -> ModeSearch:
    """A search with default options whose seed solve returns ``(ritz, vectors)``."""
    options = gfem.SolverOptions()
    return ModeSearch(
        n_modes=n_modes,
        target=8e9,
        eigen_tolerance=options.eigen_tolerance,
        max_refine_iterations=options.max_refine_iterations,
        cluster_rtol=options.cluster_rtol,
        laws=False,
        eigen_solver=lambda *_, **__: (ritz, vectors),
        crowding=lambda: None,
        refine=refine,
    )


def test_a_vector_in_the_span_of_a_degenerate_cluster_is_a_duplicate():
    # any combination of a cluster's accepted members is that cluster again,
    # also at 45 degrees from each (overlap 0.71 with either)
    a, b, c = np.eye(3, dtype=complex)

    def mode(x, k=1.0):
        return Refined(k, x, 1e-12, 0, True)

    accepted = [mode(a), mode(b)]
    assert _in_span(mode((a + 1j * b) / math.sqrt(2)), accepted, 1e-6)
    assert _in_span(mode(3 * a), accepted, 1e-6)
    assert not _in_span(mode(c), accepted, 1e-6)
    assert not _in_span(mode((a + c) / math.sqrt(2)), accepted, 1e-6)
    # the same field at another frequency is another mode
    assert not _in_span(mode(a, k=1.1), accepted, 1e-6)


def _twin_find(n_modes):
    """`ModeSearch.find` with seeds at 12.2 (e0) and 13.8 (e2), the shift at 12.5.

    The operator has a degenerate mode at 12 (span of e0, e1) and one at 14
    (e2). The frozen seed solve gave one Ritz value for the degenerate pair.
    The seed at 13.8 refined away from its own shift lands on the pair's
    eigenspace with 36 % of its norm outside the accepted member's span: in
    the span by the 0.9 test, yet that part is the twin (lossy-wall mirrored
    cavity on MN5).
    """
    e0, e1, e2, _ = np.eye(4, dtype=complex)
    landed = math.sqrt(1 - 0.36**2) * e0 + 0.36 * e1

    def refine(op, seed, x, **_):
        if seed == 13.8 and op.shift != 13.8:
            k, v = 12.0, landed
        else:
            pair = x[:2].copy()
            if np.linalg.norm(pair) > 0.5 * np.linalg.norm(x):
                k, v = 12.0, np.concatenate([pair / np.linalg.norm(pair), [0, 0]])
            else:
                k, v = 14.0, e2
        return Refined(complex(k), v, 0.0, 1, converged=True)

    ritz = np.array([12.2, 13.8], dtype=complex)
    search = _search(n_modes, refine, ritz, np.stack([e0, e2], axis=1))
    return search.find(_Stub(12.5), None, 12.5, guard=True, polish_isolated=False)


def test_a_wandered_pair_reaching_out_of_the_span_gives_the_twin():
    # before, the pair was dropped and the set skipped the twin for 14
    found = _twin_find(n_modes=2)
    assert found.ks == [12.0, 12.0]
    assert abs(np.vdot(found.modes[0].x, found.modes[1].x)) < 1e-12
    assert found.guard.k == 14.0


def test_a_guard_pair_reaching_out_of_the_span_gives_the_twin():
    # after the count is met the same landing gives the guard mode: the twin
    found = _twin_find(n_modes=1)
    assert found.ks == [12.0]
    assert found.guard.k == 12.0
    assert abs(np.vdot(found.modes[0].x, found.guard.x)) < 1e-12


def _stub_find():
    """`ModeSearch.find` on a stub operator: seeds at 10 and 13, the target at 11.4.

    The seed at 10 is far from the shift; its first refinement wanders to
    12.6 (nearer the seed at 13: lost) and is kept, and its retry at its own
    Ritz value gives 10. The count of one is then met by 12.6 and 10 is cut;
    the seed at 13 gives 13 after the count.
    """
    basis = np.eye(4, dtype=complex)

    def refine(op, seed, x, **_):
        if seed == 10 and op.shift != 10:
            k, v = 12.6, basis[:, 2]
        else:
            k, v = seed, x
        return Refined(complex(k), v, 0.0, 1, converged=True)

    ritz = np.array([10.0, 13.0], dtype=complex)
    search = _search(1, refine, ritz, basis[:, :2])
    return search.find(_Stub(11.4), None, 11.4, guard=True, polish_isolated=False)


def test_gap_set_holds_cut_modes_and_the_guard():
    found = _stub_find()
    assert found.ks == [12.6]
    assert found.guard.k == 10  # the nearest mode cut off
    # every converged distinct pair: the mode, the cut one, the one found after
    np.testing.assert_array_equal(found.gaps.pool, [12.6, 10, 13])
    np.testing.assert_array_equal(found.gaps.ritz, [10, 13])
    assert found.gaps.guard == found.guard.k
    assert found.gaps.gap(12.6) == pytest.approx(0.4 / 13)
    # the guard alone is a neighbour
    alone = GapSet(np.array([12.6 + 0j]), found.guard.k, found.gaps.ritz)
    assert alone.gap(12.6) == pytest.approx(2.6 / 12.6)


# --- the seed solve records ------------------------------------------------------


def _seed_records(caplog):
    return [
        r
        for r in caplog.records
        if r.name == "gfem.solver" and getattr(r, "attempt", None) is not None
    ]


def _identity(op, seed, x, **_):
    return Refined(complex(seed), x, 0.0, 1, converged=True)


def test_each_seed_solve_logs_one_record_so_a_retry_shows(caplog):
    # the first attempt returns one pair for two modes: the search retries
    # with twice the pairs and gets both
    basis = np.eye(4, dtype=complex)
    asked = []

    def solver(_op, _project, nev, **_):
        asked.append(nev)
        if len(asked) == 1:
            return np.array([10.0 + 0j]), basis[:, :1]
        return np.array([10.0, 13.0], dtype=complex), basis[:, :2]

    search = dataclasses.replace(
        _search(2, _identity, None, None), eigen_solver=solver
    )
    caplog.set_level(logging.INFO, logger="gfem.solver")
    found = search.find(_Stub(11.4), None, 11.4, guard=False, polish_isolated=False)
    assert found.ks == [10.0, 13.0]
    records = _seed_records(caplog)
    assert [
        (r.problem, r.attempt, r.pairs, r.returned, r.modes, r.wanted, r.bound)
        for r in records
    ] == [
        ("eigenmode", 1, asked[0], 1, 1, 2, False),
        ("eigenmode", 2, asked[1], 2, 2, 2, False),
    ]
    assert asked[1] == 2 * asked[0]
    assert records[0].levelno == logging.INFO
    assert records[1].getMessage() == (
        f"eigenmode seed solve 2: {asked[1]} pairs asked, 2 returned, "
        "2 of 2 modes selected"
    )


def test_a_seed_solve_at_its_restart_bound_logs_the_bound(caplog):
    basis = np.eye(4, dtype=complex)

    def solver(*_, **__):
        raise SeedBoundReached(
            np.array([10.0 + 0j]), basis[:, :1], restarts=5, applies=9
        )

    search = dataclasses.replace(
        _search(1, _identity, None, None), eigen_solver=solver
    )
    caplog.set_level(logging.INFO, logger="gfem.solver")
    search.find(_Stub(11.4), None, 11.4, guard=False, polish_isolated=False)
    (record,) = _seed_records(caplog)
    assert record.bound is True
    assert record.getMessage().endswith(
        "1 returned, 1 of 1 modes selected, restart bound reached"
    )


def test_no_seed_solve_record_is_built_when_the_logger_is_off(monkeypatch):
    from gfem._linalg import report

    monkeypatch.setattr(report.LOGGER, "isEnabledFor", lambda _level: False)
    monkeypatch.setattr(
        report.LOGGER,
        "info",
        lambda *_, **__: pytest.fail("a record was built while off"),
    )
    _stub_find()
