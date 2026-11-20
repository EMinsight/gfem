"""The solve cache across ranks.

Rank 0 alone reads and writes the store and broadcasts what it found; every rank
returns the same bits, checked by `on_ranks`. Only a fieldless sweep is cached: a
solve keeping fields on several ranks bypasses with one agreed warning.
"""

import warnings
from pathlib import Path

import gradient_cases as cases
import numpy as np
import pytest
from ranks import on_ranks

import gfem
from gfem._mpi import AgreedError
from gfem._solve_cache import SolveCache
from gfem.solve import _solve

FREQUENCIES = [4e9, 6e9]
RANKS = 2


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _options():
    # bddc_gmres needs no MPI PETSc stack
    return cases.tight("bddc_gmres", threads=1)


def _cached(directory, cache, **problem):
    """``(status, S)`` of one solve through ``cache``."""
    model = cases.layered_line(Path(directory))
    use = SolveCache(cache)
    result, _ = _solve(
        model, gfem.Driven(FREQUENCIES, **problem), options=_options(), cache=use
    )
    return use.report()["status"], np.asarray(result.s.values)


def _twice(directory, cache):
    first = _cached(directory, cache)
    second = _cached(directory, cache)
    return first, second


def test_a_second_sweep_on_two_ranks_is_a_hit(tmp_path):
    cache = tmp_path / "cache"
    ((miss, s), (hit, s_hit)), *_ = on_ranks(
        _twice, tmp_path, cache, ranks=RANKS, limit=300
    )
    assert (miss, hit) == ("miss", "hit")
    assert np.array_equal(s_hit, s)
    entries = [p for p in cache.iterdir() if len(p.name) == 64]
    assert len(entries) == 1


def test_an_entry_written_by_one_process_serves_two_ranks(tmp_path):
    cache = tmp_path / "cache"
    status, s_one = _cached(tmp_path, cache)
    assert status == "miss"
    (status, s), *_ = on_ranks(_cached, tmp_path, cache, ranks=RANKS, limit=300)
    assert status == "hit"
    assert np.array_equal(s, s_one)


def _kept(directory, cache):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        status, _ = _cached(directory, cache, keep_fields=True)
    return status, [
        str(w.message) for w in seen if issubclass(w.category, gfem.SolveCacheWarning)
    ]


def test_kept_fields_on_two_ranks_warn_once_and_store_nothing(tmp_path):
    cache = tmp_path / "cache"
    values = on_ranks(_kept, tmp_path, cache, ranks=RANKS, limit=300, replicated=False)
    assert [status for status, _ in values] == ["bypass"] * RANKS
    (_, shown), *others = values
    assert [o for _, o in others] == [[]] * (RANKS - 1)
    assert len(shown) == 1
    assert "Job.run(out=" in shown[0]
    assert "COMM_SELF" in shown[0]
    assert not cache.exists() or not [p for p in cache.iterdir() if len(p.name) == 64]


def _kept_under_an_error_filter(directory, cache):
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.SolveCacheWarning)
        try:
            _cached(directory, cache, keep_fields=True)
        except gfem.SolveCacheWarning:
            return _world().allreduce(1)
    return None


def test_the_kept_fields_warning_raises_on_every_rank_under_a_filter(tmp_path):
    raised, *_ = on_ranks(
        _kept_under_an_error_filter, tmp_path, tmp_path / "cache", ranks=RANKS
    )
    assert raised == RANKS


def _per_rank_cache(directory, kind):
    rank = _world().rank
    cache = Path(directory) / f"cache-{rank}"
    if kind == "none on rank 1" and rank == 1:
        cache = None
    model = cases.layered_line(Path(directory))
    try:
        gfem.solve(model, gfem.Driven(FREQUENCIES), options=_options(), cache=cache)
    except AgreedError as err:
        return "the solve cache" in str(err), _world().allreduce(1)
    return None


@pytest.mark.parametrize("kind", ["another directory", "none on rank 1"])
def test_ranks_naming_different_caches_raise_an_agreed_error(tmp_path, kind):
    (named, ranks), *_ = on_ranks(
        _per_rank_cache, tmp_path, kind, ranks=RANKS, limit=300
    )
    assert named
    assert ranks == RANKS
