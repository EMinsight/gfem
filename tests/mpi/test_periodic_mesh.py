"""A mesh with one periodic pair distributes through the strict search.

Every rank checks the mesh it received before wrapping it. A partition is kept
unswapped only with no missing and no reverse twins; one with missing twins and no
reverse ones is distributed again with the pair swapped; any other moves to the
next element order. Which seed or swap a rank count takes depends on METIS in the
wheel, so the tests assert the rule's outcome, not a path.
"""

import mpi_cases
import numpy as np
import pytest
from ranks import on_ranks

from gfem import _mpi
from gfem._engine.mesh import forced_search, partition_memo, to_ngsolve
from gfem._engine.seam_check import seam_counts
from gfem._periodic import pair_names, pairing

N = 8


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _pair(mesh):
    return (pairing(mesh, "xmin", "xmax"),)


def _distribute(start=0, swap=False):
    world = _world()
    mesh = mpi_cases.box(N)
    pairs = _pair(mesh)
    with forced_search(start=start, swap=swap):
        paired = to_ngsolve(mesh, pairs, world)
    again = to_ngsolve(mesh, pairs, world)
    plain = to_ngsolve(mesh, (), world)
    memo = partition_memo(mesh._topology)[_mpi.comm_key(world), pair_names(pairs)]
    counts = _mpi.reduce_bcast(world, seam_counts(paired.mesh.ngmesh))
    return {
        "seed": memo.seed,
        "swapped": memo.swapped,
        "pairs": [(p.first, p.second, p.translation) for p in paired.pairs],
        "counts": counts.tolist(),
        "tets": paired.cell_order,
        "same tets as plain": bool(np.array_equal(paired.cell_order, plain.cell_order)),
        "same maps again": all(
            np.array_equal(getattr(paired, f), getattr(again, f))
            for f in ("node_order", "cell_order", "facet_order")
        ),
        "plain seed": partition_memo(mesh._topology)[_mpi.comm_key(world), ()].seed,
    }


def _assert_strict(parts, start=0):
    mesh = mpi_cases.box(N)
    first = parts[0]
    for p in parts:
        assert (p["seed"], p["swapped"], p["pairs"]) == (
            first["seed"],
            first["swapped"],
            first["pairs"],
        )
        assert p["same tets as plain"]
        assert p["same maps again"]
        assert p["plain seed"] == p["seed"]
    assert first["seed"] >= start
    missing_edges, missing_faces, reverse_edges, reverse_faces = first["counts"]
    # the kept orientation misses nothing; an unswapped one has no reverse either
    assert missing_edges == missing_faces == 0
    if not first["swapped"][0]:
        assert reverse_edges == reverse_faces == 0
    names = ("xmax", "xmin") if first["swapped"][0] else ("xmin", "xmax")
    translation = -1e-6 if first["swapped"][0] else 1e-6
    assert first["pairs"][0][:2] == names
    assert first["pairs"][0][2][0] == pytest.approx(translation)
    tets = np.concatenate([p["tets"] for p in parts])
    assert np.array_equal(np.sort(tets), np.arange(len(mesh.tetrahedra)))


@pytest.mark.parametrize("ranks", [5, 8, 16])
def test_one_pair_distributes_through_the_strict_search(ranks):
    parts = on_ranks(_distribute, ranks=ranks, replicated=False, limit=240)
    _assert_strict(parts)


def test_a_forced_start_and_swap_are_recorded():
    parts = on_ranks(_distribute, 1, True, ranks=4, replicated=False, limit=240)
    _assert_strict(parts, start=1)
    assert parts[0]["swapped"] == (True,)


def _exhaust():
    world = _world()
    mesh = mpi_cases.box(N)
    try:
        with forced_search(tries=1):
            to_ngsolve(mesh, _pair(mesh), world)
    except _mpi.AgreedError as err:
        return str(err), world.allreduce(1)
    raise AssertionError("the search passed in one partition")


def test_running_out_of_tries_raises_on_every_rank():
    # box8 on 16 ranks needs 3 partitions on this wheel
    values = on_ranks(_exhaust, ranks=16, limit=240)
    message, total = values[0]
    assert total == 16
    assert "in 1 tries" in message
    assert "16 ranks" in message


def _conflict():
    world = _world()
    mesh = mpi_cases.box(N)
    to_ngsolve(mesh, (), world)
    try:
        with forced_search(start=1):
            to_ngsolve(mesh, _pair(mesh), world)
    except _mpi.AgreedError as err:
        return str(err), world.allreduce(1)
    raise AssertionError("the search moved off the plain mesh's seed")


def test_a_search_off_the_plain_meshs_seed_raises_on_every_rank():
    message, total = on_ranks(_conflict, ranks=4, limit=240)[0]
    assert total == 4
    assert "without its pairs first" in message
