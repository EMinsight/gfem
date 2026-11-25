"""NGSolve threading for one engine call."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Generator
from typing import Any

import ngsolve as ngs

from gfem import _mpi
from gfem._linalg import blas

__all__ = ["available_cpus", "serial_spaces", "solve_threads", "task_manager"]

#: The TaskManagers `task_manager` started (not entered inside another one),
#: innermost last: the one `serial_spaces` stops.
_STARTED: list[ngs.TaskManager] = []


def available_cpus() -> int:
    """Physical cores in this process's affinity mask, >= 1.

    Python's own override (``PYTHON_CPU_COUNT``, ``-X cpu_count``, 3.13) counts as
    given. Without an affinity mask (macOS, Windows) the standard library cannot tell
    a core from a hardware thread, so this is the logical CPUs there.
    """
    if not hasattr(os, "sched_getaffinity"):
        if hasattr(os, "process_cpu_count"):  # Python 3.13
            return os.process_cpu_count() or 1
        return os.cpu_count() or 1
    mask = frozenset(os.sched_getaffinity(0))
    if hasattr(os, "process_cpu_count") and _cpu_count_overridden():
        return os.process_cpu_count() or 1
    return _mpi.node_share(mask, [mask])


def _cpu_count_overridden() -> bool:
    """Whether Python's own CPU count is set: ``PYTHON_CPU_COUNT``, ``-X cpu_count``.

    Both arrived with ``os.process_cpu_count`` in 3.13; ``"default"`` is no override.
    """
    given = sys._xoptions.get("cpu_count") or os.environ.get("PYTHON_CPU_COUNT")  # noqa: SLF001
    return isinstance(given, str) and given not in ("", "default")


def solve_threads(threads: int | None) -> int:
    """One process's count: ``threads``, else ``NGS_NUM_THREADS``, else its cores.

    An ``NGS_NUM_THREADS`` that is not a positive integer is treated as unset
    (`gfem._mpi.ngs_num_threads`). Under ranks `gfem._mpi.default_threads` decides
    instead.
    """
    if threads is not None:
        return threads
    return _mpi.ngs_num_threads() or available_cpus()


@contextlib.contextmanager
def task_manager(threads: int | None, comm: Any = None) -> Generator[None]:
    """Run NGSolve assembly and solves threaded.

    Inside a TaskManager the caller already entered, the call runs on that
    TaskManager's count whatever ``threads`` says: NGSolve cannot change the count
    while one is active, and prints a line for every attempt.

    Args:
        threads: CPU threads (``SolverOptions.threads``); ``None`` is `solve_threads`,
            and under ``comm`` is `gfem._mpi.default_threads`.
        comm: The resolved communicator. The first distributed solve also sets
            each loaded BLAS to the threads the rank runs on.
    """
    starts = ngs.GetNumThreads() == 1  # 1 outside any TaskManager
    if not starts:
        threads = ngs.GetNumThreads()
    elif comm is None:
        threads = solve_threads(threads)
        ngs.SetNumThreads(threads)
    else:
        threads = _mpi.default_threads(comm) if threads is None else threads
        ngs.SetNumThreads(threads)
    if comm is not None:
        blas.settle(threads)
    manager = ngs.TaskManager()
    with manager:
        if starts:
            _STARTED.append(manager)
        try:
            yield
        finally:
            if starts:
                _STARTED.remove(manager)


@contextlib.contextmanager
def serial_spaces() -> Generator[None]:
    """Build NGSolve FE spaces here with the TaskManager stopped.

    An FE space colors its elements when it is built (``FinalizeUpdate``), in a
    parallel loop whose threads raise a shared color count without
    synchronization (NGSolve 6.2.2608). A count left below the largest color
    makes the loop write past the end of an array: heap corruption, which
    aborts or segfaults the process later, at some unrelated allocation. With
    no TaskManager running the loop runs on one thread. So the TaskManager
    `task_manager` started is stopped for the block and started again after,
    on the same thread count; outside it, or inside a TaskManager the caller
    entered (which this cannot stop), the block runs as it is.

    A space whose elements share no DOFs (an ``L2`` or ``SurfaceL2`` space of
    order 0, and compounds of them) colors every element 0 and needs no stop.
    """
    if not _STARTED or ngs.GetNumThreads() == 1:
        yield
        return
    manager, threads = _STARTED[-1], ngs.GetNumThreads()
    manager.__exit__(None, None, None)
    try:
        yield
    finally:
        # a solve inside the block may have set another count
        ngs.SetNumThreads(threads)
        manager.__enter__()
