"""``gfem.log`` under a real two-rank launch.

The world rank comes from the launcher's environment and an imported mpi4py,
not from a patched one. Rank 0 tags its lines and draws the ``uniform sweep``
row; another rank shows only warnings unless told otherwise, and never draws
progress, which is rank 0's alone.
"""

import io
import os
import re
import sys

import gradient_cases as cases
import pytest
from ranks import on_ranks

import gfem
import gfem.log

FREQUENCY = 5e9
ROW = "uniform sweep"
#: rich's colours and the live row's cursor moves, which split no word
ESCAPES = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _logged_solve(directory, rank0_only):
    """What this rank's stderr shows of a BDDC solve with ``gfem.log`` enabled."""
    # rich treats the stand-in stderr as a terminal, so rank 0 draws its row
    os.environ["TTY_COMPATIBLE"] = "1"
    stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        gfem.log.enable(rank0_only=rank0_only)
        try:
            gfem.solve(
                cases.layered_line(directory),
                gfem.Driven([FREQUENCY]),
                options=cases.tight("bddc_gmres", threads=1),
            )
        finally:
            # stops the live row, whose last frame stays in the text
            gfem.log.disable()
        return ESCAPES.sub("", sys.stderr.getvalue())
    finally:
        sys.stderr = stderr


@pytest.mark.parametrize("rank0_only", [True, False])
def test_rank_zero_tags_its_lines_and_alone_draws_the_row(tmp_path, rank0_only):
    zero, one = on_ranks(
        _logged_solve, tmp_path, rank0_only, ranks=2, limit=300, replicated=False
    )
    assert " r0/2 " in zero
    assert ROW in zero
    if rank0_only:
        assert one == ""
    else:
        assert " r1/2 " in one
        assert ROW not in one
