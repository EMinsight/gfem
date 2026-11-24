"""Frequency workers: blocks of a uniform sweep solved in spawned processes.

A frequency worker is a separate process that solves one contiguous block
of a uniform sweep's frequencies (`SolverOptions.frequency_workers`). Each
worker is a process pool of its own, started with ``spawn``, never
``fork``: JAX's and NGSolve's threads are already running in the parent,
and NGSolve's TaskManager is process-global, so threads of one process
cannot solve concurrently. A pool per worker ties a killed worker to its
own block, so the error names the frequencies it was solving. A worker
starts with the OpenMP, MKL and OpenBLAS thread variables set to its share
of the threads (its environment at spawn: the libraries read them when
imported); NGSolve's own threads follow the worker's
``SolverOptions.threads``. It starts without the launcher's variables too: a worker is
a plain one-process run, and a script that imports mpi4py at its top would otherwise
die in ``MPI_Init`` when the worker imports it again.

`gfem.Model` does not pickle (its materials are a mapping proxy); its
pytree parts do (`model_parts`, `model_from_parts`).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import queue as queues
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import jax
import numpy as np

from gfem import _progress
from gfem._engine.runtime import solve_threads
from gfem.model import Model

__all__ = ["model_from_parts", "model_parts", "run", "worker_threads"]

#: Thread-count variables of the libraries a solve may run on.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
#: What a launcher tells its tasks; a worker never resolves a communicator.
_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "SLURM_STEP_")
#: Held while this process's environment carries a worker's thread variables.
_START_LOCK = threading.Lock()


def worker_threads(threads: int | None, workers: int) -> int:
    """Each worker's share of ``threads`` (``None``: the solve threads), >= 1."""
    return max(1, solve_threads(threads) // workers)


def model_parts(model: Model) -> tuple[Any, Any]:
    """``model`` as picklable parts, its leaves as NumPy arrays."""
    children, aux = jax.tree.map(np.asarray, model).tree_flatten()
    return aux, children


def model_from_parts(aux: Any, children: Any) -> Model:
    """The model of `model_parts`."""
    return Model.tree_unflatten(aux, children)


@contextlib.contextmanager
def _worker_environment(threads: int) -> Generator[None]:
    """This process's environment as a worker's: its threads, and no launcher.

    A process spawned meanwhile starts with it. The environment is restored on
    exit; `_START_LOCK` keeps two `run` calls from interleaving.
    """
    with _START_LOCK:
        launcher = [name for name in os.environ if name.startswith(_LAUNCHER_PREFIXES)]
        names = (*_THREAD_VARIABLES, *launcher)
        saved = {name: os.environ.get(name) for name in names}
        os.environ.update(dict.fromkeys(_THREAD_VARIABLES, str(threads)))
        for name in launcher:
            del os.environ[name]
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def run(
    function: Callable[..., Any],
    arguments: Sequence[tuple],
    labels: Sequence[str],
    *,
    threads: int,
    keywords: Sequence[dict[str, Any]] | None = None,
    row: _progress.ProgressRow | None = None,
) -> list[Any]:
    """``function(*a, **k)`` for each ``a`` of ``arguments``, in its own worker.

    Args:
        function: A module-level function (it is pickled by name).
        arguments: One argument tuple per worker.
        labels: What each worker solves, for errors (e.g. its frequencies).
        threads: Each worker's threads (`worker_threads`).
        keywords: One keyword-argument dict per worker; none by default.
        row: The parent's progress row (`gfem._progress.task`) the workers'
            advances count on, or ``None``. Only a live row gets a queue.

    Returns:
        The workers' results, in ``arguments`` order.

    Raises:
        RuntimeError: A worker stopped without a result (killed, e.g. for
            lack of memory, or dead while loading its inputs). A worker's
            own exception is re-raised as is.
    """
    context = multiprocessing.get_context("spawn")
    # a queue and its feeder threads only while a row is drawn
    feed = (context.Queue(), row) if isinstance(row, _progress.Row) else None
    pool_options: dict[str, Any] = {}
    if feed is not None:
        pool_options = {"initializer": _progress.forward_to, "initargs": feed[:1]}
    try:
        with contextlib.ExitStack() as stack:
            futures = []
            # a pool starts its process on the first submit
            with _worker_environment(threads):
                for i, args in enumerate(arguments):
                    pool = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=1, mp_context=context, **pool_options
                        )
                    )
                    kwargs = keywords[i] if keywords is not None else {}
                    futures.append(pool.submit(function, *args, **kwargs))
            if feed is not None:
                _advance_until_done(futures, *feed)
            out = []
            for future, label in zip(futures, labels, strict=True):
                try:
                    out.append(future.result())
                except BrokenProcessPool as err:
                    raise RuntimeError(
                        f"The frequency worker solving {label} stopped without a "
                        "result: it was killed, most likely for lack of memory (each "
                        "worker holds one solve), or it died loading its inputs (a "
                        "function it could not import). Use fewer "
                        "SolverOptions.frequency_workers, or frequency_workers=1."
                    ) from err
        if feed is not None:
            # a worker's feeder thread flushes its last advances as it exits,
            # which the pools' shutdown above waited for
            _drain(*feed)
    finally:
        # on every path: a block's error leaves no queue feeder behind
        if feed is not None:
            feed[0].close()
    return out


def _advance_until_done(
    futures: Sequence[Future], forwarded: Any, row: _progress.ProgressRow
) -> None:
    """Advance ``row`` by the workers' advances until every future is done."""
    pending = set(futures)
    while pending:
        _, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        _drain(forwarded, row)


def _drain(forwarded: Any, row: _progress.ProgressRow) -> None:
    """Advance ``row`` by each advance waiting on ``forwarded``."""
    while True:
        try:
            n, fields = forwarded.get_nowait()
        except queues.Empty:
            return
        row.advance(n, **fields)
