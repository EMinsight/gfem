"""The ``gfem.timing`` records across ranks (ADR 0190).

Each rank logs its own records: its ``rank``, the job's ``ranks`` and, on
``discretize``, the DOFs it owns. Summed over the ranks those are the
problem's DOFs, the count one process logs.
"""

import logging

import gradient_cases as cases
import pytest
from ranks import on_ranks

import gfem

FREQUENCY = 5e9


class _Keep(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _timing_records(directory):
    """This rank's ``gfem.timing`` records of a BDDC solve, as tuples.

    ``(phase, rank, ranks, ndof)``, ``ndof`` ``None`` but on ``discretize``.
    """
    logger = logging.getLogger("gfem.timing")
    keep = _Keep()
    logger.addHandler(keep)
    logger.setLevel(logging.INFO)
    try:
        gfem.solve(
            cases.layered_line(directory),
            gfem.Driven([FREQUENCY]),
            options=cases.tight("bddc_gmres", threads=1),
        )
    finally:
        logger.removeHandler(keep)
        logger.setLevel(logging.NOTSET)
    return [
        (r.phase, r.rank, r.ranks, getattr(r, "ndof", None))  # ty: ignore[unresolved-attribute]
        for r in keep.records
    ]


def _owned(records):
    (ndof,) = [ndof for phase, *_, ndof in records if phase == "discretize"]
    return ndof


@pytest.mark.parametrize("ranks", [3, 4])
def test_the_owned_dofs_sum_to_the_one_process_count(tmp_path, ranks):
    one = _timing_records(tmp_path)
    per_rank = on_ranks(
        _timing_records, tmp_path, ranks=ranks, limit=300, replicated=False
    )
    assert sum(_owned(records) for records in per_rank) == _owned(one)
    for rank, records in enumerate(per_rank):
        assert {phase for phase, *_ in records} == {
            "discretize",
            "assemble",
            "setup",
            "solve",
        }
        assert {(r, n) for _, r, n, _ in records} == {(rank, ranks)}
        # NGSolve's rank 0 holds no elements, so it owns no DOFs
        assert 0 <= _owned(records) < _owned(one)
