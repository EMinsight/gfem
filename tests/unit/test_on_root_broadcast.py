"""`gfem._mpi.on_root` sends the arrays only where the caller reads them."""

import numpy as np

from gfem import _mpi


class _Comm:
    """A two-rank communicator seen from one rank: it records what is sent."""

    def __init__(self, rank):
        self.rank = rank
        self.sent = []

    def bcast(self, value, root=0):
        self.sent.append(("bcast", value))
        return value if self.rank == root else (None, [])

    def Bcast(self, array, root=0):  # noqa: ARG002, N802
        self.sent.append(("Bcast", array))


def _arrays():
    return {"a": np.arange(3.0)}


def test_the_arrays_are_not_sent_when_only_rank_0_reads_them():
    root = _Comm(0)
    got = _mpi.on_root(root, _arrays, broadcast=False)
    assert list(got) == ["a"]
    assert [kind for kind, _ in root.sent] == ["bcast"]  # the outcome only
    other = _Comm(1)
    assert _mpi.on_root(other, _arrays, broadcast=False) == {}
    assert [kind for kind, _ in other.sent] == ["bcast"]


def test_the_arrays_are_sent_by_default():
    root = _Comm(0)
    _mpi.on_root(root, _arrays)
    assert [kind for kind, _ in root.sent] == ["bcast", "bcast", "Bcast"]
