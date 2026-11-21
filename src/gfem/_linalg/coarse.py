"""BDDC's coarse problem across ranks on PETSc MUMPS.

On rank 0 the coarse factorization holds about a third of a mesh's DOFs in one
process and every rank waits for its apply. NGSolve lets a registered ``coarsetype``
take its place, so with the MPI PETSc stack `CoarseMumps` factors the wirebasket
across ranks, with no change to BDDC: iterations and S stay those of the rank 0 route.

``PCTELESCOPE`` in front of MUMPS runs the factorization on every factor-th rank,
without building a communicator by hand. `rule` picks that factor and the ordering
from counts known at the set-up: the coarse DOFs, the ranks, the columns the set-up
is expected to serve, and whether this process has set up the same pattern before.
It reads counts only, so every rank and every rerun of a script picks the same; none
of it is an option, since S does not change with it beyond round-off.

- PT-Scotch by default. Its fill grows with the ranks it orders on (r10M: 556 M
  entries at 104 k wirebasket DOF per rank, 4520 M at 52 k), so it takes at most
  one rank per `DOFS_PER_RANK`. With an even count per node MUMPS orders on the
  largest power of two of them: 56 ranks order on 32, but 64 order on 64, where
  the fill jumps. Hence `MAX_RANKS`.
- Sequential METIS on every `COARSE_EVERY`-th rank, at least
  `COARSE_DOFS_PER_RANK` DOFs each, when that gives more ranks than PT-Scotch may
  take. Its fill does not depend on the ranks, so its apply keeps getting faster
  with nodes, but its analysis runs on one process and costs several PT-Scotch
  set-ups. It is taken only where that is paid back: a set-up expected to serve
  `METIS_COLUMNS` columns or more, or a pattern this process has set up before (a
  design loop, a sweep's rebuilds, an eigen reshift), whose analysis the pool keeps.

`stack_kind` tells which PETSc stack every rank runs: one with ParMETIS, one whose
MUMPS caps PT-Scotch's parallel ordering, as gfem's build script makes it, or a
stock one. Ranks on different stacks take the kind they
share, which changes the coarse problem's speed, never S.

METIS's ``NITER=1`` takes about 15 % off that serial analysis for 2 % more fill.
PETSc has no option for it and MUMPS reads it only from the file named by
``MUMPS_CONFIG_FILE``, so a METIS set-up names the packaged `MUMPS_CONFIG` there,
under `_petsc.LOCK` and for that set-up only (`mumps_config`): nothing is written at
run time, and MUMPS factors set up elsewhere in the process never read it. A value
the user set is left as it is.

MUMPS factors the coarse problem by :math:`L D L^T` where the operator equals its
transpose and by LU otherwise (a Bloch phase). NGSolve hands the creator only the
matrix and the free DOFs, so each symmetry has its own registered ``coarsetype``
(`coarse_type`).

MUMPS's analysis of the coarse problem costs up to several of its numeric
factorizations, and BDDC is set up again on the same wirebasket at every design, at a
sweep's rebuilds and at an eigen reshift. So a freed factorization goes to a pool of
this process, keyed by communicator, symmetry and wirebasket pattern, and a later
set-up on that key writes its values into the pooled matrix and refactorizes
numerically only. A pooled factorization belongs to at most one live set-up, and one
whose factorization failed is never pooled: PETSc refuses a numeric factorization
after a failed one. A pooled factorization holds its factors, so the solve cache
counts the pool against ``retain_memory``; ``retain_memory=0`` empties it at the end
of each solve, and ``gfem.release_memory()`` and the process's exit free it.

petsc4py cannot reach the KSP behind ``PCTELESCOPE``, so `_petsc.telescope_ksp`
reads it through ``libpetsc``. Its MUMPS factor gives the bytes the pool and the
solve cache count, and the failure every rank agrees on: only a workspace error is
retried, with a larger ICNTL(14), while a singular operator (at a resonance) raises
at once. Where that symbol cannot be loaded the factor counts 0 bytes, and every
failure is retried, since it may be the workspace.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import dataclasses
import hashlib
import logging
import math
import os
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import ngsolve as ngs
import numpy as np

from gfem import _mpi, _petsc
from gfem._linalg.petsc import (
    MUMPS_RETRIES,
    MUMPS_WORKSPACE_ERROR,
    RANK_MUMPS_OPTIONS,
    FreeNumbering,
    distributed_aij,
    set_values,
)
from gfem._linalg.petsc import defaults as factor_defaults

#: Wirebasket DOFs per coarse rank: PT-Scotch's fill is fine at 66 k and 104 k and
#: jumps at 52 k and 33 k on r0.5 and r10M; "about 100 k" errs on the fine side.
DOFS_PER_RANK = 100_000
#: Coarse ranks at most under PT-Scotch: with an even count per node it orders on the
#: largest power of two of them, so 56 orders on 32 and 64 on 64, where the fill jumps
#: (B2_void_split: 8.87 G entries on 56 ranks, 10.8-11.3 G on 64).
MAX_RANKS = 56

#: The coarse orderings `rule` picks from; `PARMETIS` only on a stack that has it.
PT_SCOTCH: Final = "pt-scotch"
METIS: Final = "metis"
PARMETIS: Final = "parmetis"
type Ordering = Literal["pt-scotch", "metis", "parmetis"]
#: Coarse ranks are every this-many-th rank, or the even layout nearest it: 7 per
#: node at 28 ranks per node, under METIS, ParMETIS and capped PT-Scotch alike.
COARSE_EVERY = 4
#: Wirebasket DOFs per coarse rank at least, but under stock PT-Scotch
#: (`DOFS_PER_RANK`): r0.5 on 224 ranks, 31.5 k each, still applied faster than on
#: 56.
COARSE_DOFS_PER_RANK = 30_000
#: Columns from which one METIS set-up repays its serial analysis: about 300-370
#: applies at B2's 64 iterations per column.
METIS_COLUMNS = 6
#: MUMPS reads it through ``MUMPS_CONFIG_FILE``: METIS's ``NITER=1`` takes about
#: 15 % off the serial analysis for 2 % more fill, and PETSc has no option for it.
MUMPS_CONFIG = Path(__file__).with_name("mumps_metis.cfg")
#: The variable MUMPS reads its configuration file from.
CONFIG_VARIABLE = "MUMPS_CONFIG_FILE"
#: MUMPS's ordering options under METIS: a sequential analysis (ICNTL(28) 1) with
#: METIS (ICNTL(7) 5), so ICNTL(29), PT-Scotch's choice, is left out.
METIS_MUMPS_OPTIONS: tuple[tuple[str, Any], ...] = (
    ("mat_mumps_icntl_28", 1),
    ("mat_mumps_icntl_7", 5),
)
#: MUMPS's ordering option under ParMETIS: a parallel analysis (ICNTL(28) 2, as
#: under ranks) with ParMETIS (ICNTL(29) 2).
PARMETIS_MUMPS_OPTIONS: tuple[tuple[str, Any], ...] = (("mat_mumps_icntl_29", 2),)
#: MUMPS's first ICNTL(14) under PT-Scotch: 5 of 31 PT-Scotch set-ups of B2 ran out
#: of workspace at 60, against none of about 20 under METIS or ParMETIS, which
#: start at `RANK_MUMPS_OPTIONS`'s.
SCOTCH_WORKSPACE = 120

#: Says what the stack kind leaves to the user, and each set-up's choice.
_LOGGER = logging.getLogger("gfem.solver")
#: The kinds of PETSc stack `stack_kind` tells apart: with ParMETIS (`PARMETIS`),
#: with the cap gfem's build script puts on PT-Scotch's parallel ordering, or
#: neither.
CAPPED: Final = "capped"
STOCK: Final = "stock"
type StackKind = Literal["parmetis", "capped", "stock"]

#: Failures a larger workspace may cure.
RETRIED = (MUMPS_WORKSPACE_ERROR,)
#: The failure read where the factor behind ``PCTELESCOPE`` cannot be: retried too,
#: since it may be the workspace.
UNREAD = 0

#: Per communicator, the factorizations NGSolve asked for since the last `finish`,
#: of either symmetry.
_made: dict[int | None, list[CoarseMumps]] = {}
#: The ``coarsetype`` names registered with NGSolve in this process.
_registered: set[str] = set()

#: A pool key: communicator key, :math:`A^T = A`, and the wirebasket pattern's
#: digest, the same on every rank.
type PoolKey = tuple[int | None, bool, bytes]
#: The pooled factorizations no set-up holds, least recently returned first.
_pool: list[Kept] = []
#: The keys set up in this process: they outlive eviction, so a pattern met again
#: after its factorization went is still known as met.
_seen: set[PoolKey] = set()
#: Per communicator key, its `stack_kind`.
_kinds: dict[int | None, StackKind] = {}
#: Whether `say_stock` has spoken in this process.
_said_stock: list[bool] = []
#: Whether `release` runs at exit, before PETSc finalizes.
_at_exit: list[bool] = []


@dataclasses.dataclass(eq=False)
class Kept:
    """One coarse factorization and what a refactorization on its pattern reuses.

    Attributes:
        key: Its pool key.
        aij: The matrix, refilled in place on the same pattern.
        index: The positions of its entries in the wirebasket's value vector.
        x: A solution vector of ``aij``.
        b: A load vector of ``aij``.
        ordering: The ordering its KSP was set up with.
        factor: ``PCTELESCOPE``'s factor its KSP was set up with.
        workspace: MUMPS's ICNTL(14) of its KSP.
        ksp: The KSP, ``None`` until set up and after a failed factorization.
        prefix: The KSP's options prefix.
        pairs: The options read under ``prefix``.
    """

    key: PoolKey
    aij: Any
    index: np.ndarray
    x: Any
    b: Any
    ordering: Ordering
    factor: int
    workspace: int
    ksp: Any = None
    prefix: str = ""
    pairs: tuple[tuple[str, Any], ...] = ()

    def inner_factor(self) -> Any:
        """MUMPS's factor matrix on this rank, ``None`` on a rank telescoping drops."""
        pc = self.ksp.getPC()
        if self.factor == 1:
            return pc.getFactorMatrix()
        inner = _petsc.telescope_ksp(pc)
        return None if inner is None else inner.getPC().getFactorMatrix()

    @property
    def nbytes(self) -> int:
        """This rank's factors, by MUMPS's own count (INFO(22)).

        0 on a rank telescoping drops, and where the factor behind ``PCTELESCOPE``
        cannot be read (`_petsc.telescope_symbols`).
        """
        if self.ksp is None:
            return 0
        factor = self.inner_factor()
        return 0 if factor is None else 10**6 * int(factor.getMumpsInfo(22))

    def drop_ksp(self) -> None:
        """Destroy the KSP: a failed factorization is not set up again."""
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None

    def destroy(self) -> None:
        """Destroy every PETSc object; every rank calls this together."""
        self.drop_ksp()
        for obj in (self.aij, self.x, self.b):
            obj.destroy()


def pattern_key(
    comm: Any, local: Any, numbers: np.ndarray, *, symmetric: bool
) -> PoolKey:
    """The pool key of a wirebasket: its rows on this rank and its free numbering.

    Each rank hashes its own rows and the digests are summed over the ranks, so every
    rank reads one key and the pool's history is the same on each.
    """
    _, cols, rowptr = local.CSR()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.int64(_mpi.rank(comm)).tobytes())
    for array in (cols, rowptr, numbers):
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    words = np.frombuffer(digest.digest(), dtype=np.uint64).copy()
    agreed = np.asarray(_mpi.reduce_bcast(comm, words), dtype=np.uint64)
    return _mpi.comm_key(comm), symmetric, agreed.tobytes()


def seen(key: PoolKey) -> bool:
    """Whether this process has set a coarse problem up on ``key`` before."""
    return key in _seen


def pooled() -> int:
    """The factorizations in the pool."""
    return len(_pool)


def held(owner: int | None) -> list[Kept]:
    """The pooled factorizations of the communicator ``owner``, oldest first."""
    return [k for k in _pool if k.key[0] == owner]


def held_all() -> list[Kept]:
    """Every pooled factorization, oldest first."""
    return list(_pool)


def evict(kept: Kept) -> None:
    """Free one pooled factorization; every rank of its communicator together."""
    with _petsc.LOCK:
        _pool.remove(kept)
        kept.destroy()


def empty(comm: Any) -> None:
    """Free the pooled factorizations of ``comm``; its ranks together."""
    for kept in held(_mpi.comm_key(comm)):
        evict(kept)


def release() -> None:
    """Free every pooled factorization and forget the keys seen.

    Every rank calls it at the same point.
    """
    with _petsc.LOCK:
        while _pool:
            _pool.pop(0).destroy()
        _seen.clear()


def _take(key: PoolKey, ordering: Ordering, factor: int) -> Kept | None:
    """The most recently returned factorization on ``key``, out of the pool.

    One set up otherwise (another ordering once the rule changed it) is freed: it
    would never be taken again.
    """
    for kept in [k for k in _pool if k.key == key]:
        if (kept.ordering, kept.factor) != (ordering, factor):
            evict(kept)
    for kept in reversed(_pool):
        if kept.key == key:
            _pool.remove(kept)
            return kept
    return None


def _give(kept: Kept) -> None:
    """Pool ``kept``; the first time, `release` is registered to run at exit.

    Registered after PETSc was imported, so it runs before PETSc finalizes.
    """
    _pool.append(kept)
    if not _at_exit:
        atexit.register(release)
        _at_exit.append(True)


def stack_kind(comm: Any) -> StackKind:
    """The kind of PETSc stack every rank of ``comm`` runs, once per communicator.

    `PARMETIS` where every rank's stack has ParMETIS, else `CAPPED` where every
    rank's ``libpetsc`` carries the ordering cap, else `STOCK`. Every rank calls it
    together, at most two reductions the first time. Ranks that run different stacks
    fall through to the next kind, and a rank whose own stack has more says so once:
    the kind changes the coarse problem's speed, never S.
    """
    key = _mpi.comm_key(comm)
    if key in _kinds:
        return _kinds[key]
    lost = []
    kind: StackKind = STOCK
    for name, here, candidate in (
        ("ParMETIS", _petsc.has_parmetis, PARMETIS),
        ("the ordering cap", _petsc.has_paraord_cap, CAPPED),
    ):
        mine = here()
        if _mpi.agree_all(comm, flag=mine):
            kind = candidate
            break
        if mine:
            lost.append(name)
    _kinds[key] = kind
    if lost:
        _LOGGER.warning(
            "This rank's PETSc stack has %s and another rank's has not, so the ranks "
            "run different stacks: BDDC's coarse problem follows the %s stack's rule",
            " and ".join(lost),
            kind,
        )
    return kind


def say_stock(ranks: int, capped: int) -> None:
    """Say once per process that a rebuilt stack would give the coarse more ranks.

    Args:
        ranks: The coarse ranks this `STOCK` stack's rule takes.
        capped: Those it would take on a `CAPPED` stack. Nothing is said, and the
            once is kept, unless that is more.
    """
    if capped <= ranks or _said_stock:
        return
    _said_stock.append(True)
    _LOGGER.info(
        "BDDC's coarse problem runs on %d ranks: this PETSc stack does not cap "
        "PT-Scotch's parallel ordering, whose fill grows with the ranks it orders "
        "on. A stack rebuilt with tools/petsc_wheelhouse.sh --mpi caps it, and "
        "the coarse problem would then run on %d ranks.",
        ranks,
        capped,
    )


def reduction(dofs: int, size: int) -> int:
    """``PCTELESCOPE``'s factor, under `DOFS_PER_RANK` and `MAX_RANKS`.

    Args:
        dofs: The coarse problem's free DOFs.
        size: The ranks of the communicator.

    Returns:
        1 for no reduction. PETSc keeps every factor-th rank, ``ceil(size / factor)``
        of them.
    """
    return max(1, math.ceil(size / _scotch_ranks(dofs, size)))


def _scotch_ranks(dofs: int, size: int) -> int:
    """The ranks PT-Scotch's factorization takes: `reduction`'s and `rule`'s."""
    return max(1, min(size, dofs // DOFS_PER_RANK, MAX_RANKS))


def is_even(factor: int, nodes: Sequence[int]) -> bool:
    """Whether every node holding a rank ``PCTELESCOPE`` keeps holds as many.

    ``nodes``: the node of each rank (`gfem._mpi.node_ids`). PETSc keeps every
    ``factor``-th rank. A node that holds none does not count: MUMPS counts the
    processes per node of its own communicator only.
    """
    counts = collections.Counter(nodes[::factor])
    return len(set(counts.values())) == 1


def even_factor(dofs: int, nodes: Sequence[int]) -> int:
    r"""``PCTELESCOPE``'s factor of the even coarse layout nearest every fourth rank.

    With :math:`n` the coarse DOFs and :math:`P` the ranks, the target is
    :math:`c^* = \min(\lceil P / 4 \rceil, \max(1, \lfloor n / 30\,000 \rfloor))`
    coarse ranks. Of the factors :math:`f` whose :math:`\lceil P / f \rceil` kept
    ranks are even per node (`is_even`) and at most the floor's
    :math:`\max(1, \lfloor n / 30\,000 \rfloor)`, it takes the count nearest
    :math:`c^*`, ties to more ranks, then the smallest factor. Every rank holds the
    same ``nodes``, so every rank picks the same.

    Args:
        dofs: The coarse problem's free DOFs.
        nodes: The node of each rank (`gfem._mpi.node_ids`).
    """
    size = len(nodes)
    floor = max(1, dofs // COARSE_DOFS_PER_RANK)
    target = min(math.ceil(size / COARSE_EVERY), floor)
    even = [
        f
        for f in range(1, size + 1)
        if math.ceil(size / f) <= floor and is_even(f, nodes)
    ]
    # f = size keeps one rank, always even, so ``even`` is never empty
    return min(even, key=lambda f: (abs(math.ceil(size / f) - target), f))


def capped_ranks(dofs: int, nodes: Sequence[int]) -> int:
    """The coarse ranks `rule` takes on a `CAPPED` stack, for `say_stock`."""
    size = len(nodes)
    return max(math.ceil(size / even_factor(dofs, nodes)), _scotch_ranks(dofs, size))


def rule(
    dofs: int,
    size: int,
    columns: int,
    *,
    seen: bool,
    kind: StackKind = STOCK,
    nodes: Sequence[int] | None = None,
) -> tuple[Ordering, int]:
    r"""The coarse ordering and ``PCTELESCOPE``'s factor of one set-up.

    On a `PARMETIS` or `CAPPED` stack the fill no longer grows with the ranks
    ordering it, so the coarse problem takes ParMETIS or PT-Scotch on the even
    layout nearest every fourth rank (`even_factor`), for every set-up alike. A
    layout so placed that it holds fewer ranks than stock PT-Scotch's takes stock
    PT-Scotch's factor instead.

    On a `STOCK` stack, with :math:`n` the coarse DOFs and :math:`P` the ranks,
    PT-Scotch takes :math:`c_P = \max(1, \min(P, \lfloor n / 100\,000 \rfloor,
    56))` ranks and sequential METIS :math:`c_M = \max(1, \min(\lceil P / 4
    \rceil, \lfloor n / 30\,000 \rfloor))`. METIS orders when :math:`c_M > c_P`
    and the set-up serves at least `METIS_COLUMNS` columns or repeats a pattern: its
    serial analysis costs several PT-Scotch set-ups, which only a faster apply over
    many columns, or an analysis kept across set-ups, pays back.

    The rule reads counts and the agreed node of each rank only, so every rank and
    every rerun picks the same.

    Args:
        dofs: The coarse problem's free DOFs.
        size: The ranks of the communicator.
        columns: The columns the set-up is expected to serve; read on `STOCK` only.
        seen: This process has set up the same pattern before; read on `STOCK`
            only.
        kind: The stack every rank runs (`stack_kind`).
        nodes: The node of each rank (`gfem._mpi.node_ids`); ``None`` for one node.

    Returns:
        The ordering and the factor: PETSc keeps every factor-th rank.
    """
    scotch = _scotch_ranks(dofs, size)
    if kind != STOCK:
        layout = tuple(nodes) if nodes is not None else (0,) * size
        factor = even_factor(dofs, layout)
        if math.ceil(size / factor) < scotch:
            factor = reduction(dofs, size)
        return (PARMETIS if kind == PARMETIS else PT_SCOTCH), factor
    metis = max(1, min(math.ceil(size / COARSE_EVERY), dofs // COARSE_DOFS_PER_RANK))
    if metis > scotch and (columns >= METIS_COLUMNS or seen):
        return METIS, math.ceil(size / metis)
    return PT_SCOTCH, reduction(dofs, size)


def workspace(ordering: Ordering) -> int:
    """MUMPS's first ICNTL(14) for the coarse factorization under ``ordering``."""
    if ordering == PT_SCOTCH:
        return SCOTCH_WORKSPACE
    return int(dict(RANK_MUMPS_OPTIONS)["mat_mumps_icntl_14"])


@contextlib.contextmanager
def mumps_config(ordering: Ordering) -> Generator[None]:
    """``MUMPS_CONFIG_FILE`` on `MUMPS_CONFIG` for a METIS set-up, under `_petsc.LOCK`.

    MUMPS reads the variable at each job, so rank-local factors set up outside the
    scope never see it, while the coarse factor keeps ``NITER=1`` for its later
    jobs. A value the user set is left as it is: it applies to every MUMPS in the
    process.
    """
    with _petsc.LOCK:
        if ordering != METIS or CONFIG_VARIABLE in os.environ:
            yield
            return
        os.environ[CONFIG_VARIABLE] = str(MUMPS_CONFIG)
        try:
            yield
        finally:
            os.environ.pop(CONFIG_VARIABLE, None)


def options(
    symmetric: bool,
    *,
    workspace: int,
    factor: int,
    ordering: Ordering = PT_SCOTCH,
) -> tuple[tuple[str, Any], ...]:
    """PETSc's options: an exact MUMPS factorization, behind telescoping if it reduces.

    Args:
        symmetric: The operator has :math:`A^T = A`: :math:`L D L^T`, else LU.
        workspace: MUMPS's ICNTL(14).
        factor: ``PCTELESCOPE``'s reduction (`rule`); 1 for none.
        ordering: `PT_SCOTCH`, gfem's options under ranks; `METIS`
            (`METIS_MUMPS_OPTIONS` in their place); or `PARMETIS`
            (`PARMETIS_MUMPS_OPTIONS` over them).
    """
    mumps = dict(RANK_MUMPS_OPTIONS)
    if ordering == METIS:
        del mumps["mat_mumps_icntl_29"]
        mumps.update(METIS_MUMPS_OPTIONS)
    elif ordering == PARMETIS:
        mumps.update(PARMETIS_MUMPS_OPTIONS)
    mumps["mat_mumps_icntl_14"] = workspace
    factorization = (*factor_defaults(symmetric)[1:], *mumps.items())
    if factor == 1:
        return (("ksp_type", "preonly"), *factorization)
    return (
        ("ksp_type", "preonly"),
        ("pc_type", "telescope"),
        ("pc_telescope_reduction_factor", factor),
        ("telescope_ksp_type", "preonly"),
        *((f"telescope_{name}", value) for name, value in factorization),
    )


class CoarseMumps(ngs.BaseMatrix):
    """The coarse factorization: a ``ParallelMatrix`` factored by MUMPS across ranks.

    NGSolve calls the creator twice per set-up, first with the form's whole matrix,
    which is not assembled yet, and last with the wirebasket, so nothing is factored
    until `set_up`. Its PETSc objects come from the pool where one on the same
    pattern is free, and go back to it at `free`, which every rank calls together,
    never a collector that runs at another time on each rank.
    """

    def __init__(
        self,
        matrix: Any,
        freedofs: Any,
        comm: Any,
        *,
        symmetric: bool,
    ) -> None:
        """Hold what NGSolve gives; nothing is read from it yet.

        ``symmetric``: the operator has :math:`A^T = A`, so MUMPS factors it by
        :math:`L D L^T` on a matrix flagged symmetric; LU otherwise.
        """
        super().__init__()
        self.symmetric = symmetric
        self._source = matrix
        self._freedofs = freedofs
        self._comm = comm
        self._held: Kept | None = None
        self._freed = False
        self._readable = True
        #: The columns the set-up is expected to serve, for `rule`.
        self.columns = 1
        #: The pool key of the wirebasket, once set up.
        self.key: PoolKey | None = None
        #: The set-up's failed attempts that a new KSP followed.
        self._retries = 0
        #: The stack kind its last set-up read (`stack_kind`).
        self._kind: StackKind = STOCK

    @property
    def _ksp(self) -> Any:
        return None if self._held is None else self._held.ksp

    @property
    def _x(self) -> Any:
        return None if self._held is None else self._held.x

    @property
    def _b(self) -> Any:
        return None if self._held is None else self._held.b

    def set_up(self) -> None:
        """Factor the wirebasket, on a new KSP while MUMPS's workspace is too small.

        A pooled factorization on the same key is refilled and refactorized
        numerically; a failed one is dropped and the next attempt has a new KSP.

        Raises:
            gfem._mpi.AgreedError: The factorization failed, on every rank.
        """
        comm = self._comm
        with _petsc.LOCK:
            self._work = self._source.CreateColVector()
            self._numbering = FreeNumbering(
                self._source.row_pardofs, self._freedofs, self._work, comm
            )
            local = self._source.local_mat
            self.key = pattern_key(
                comm, local, self._numbering.numbers, symmetric=self.symmetric
            )
            total = self._numbering.total
            kind = self._kind = stack_kind(comm)
            nodes = _mpi.node_ids(comm)
            if kind == STOCK:
                say_stock(_scotch_ranks(total, comm.size), capped_ranks(total, nodes))
            self._ordering, self._factor = rule(
                total,
                comm.size,
                self.columns,
                seen=seen(self.key),
                kind=kind,
                nodes=nodes,
            )
            _LOGGER.info(
                "BDDC's coarse problem: %s on %d of %d ranks, %d DOFs, on a %s "
                "PETSc stack",
                self._ordering,
                math.ceil(comm.size / self._factor),
                comm.size,
                total,
                kind,
            )
            # every rank must take the same retries
            self._readable = self._factor == 1 or _mpi.agree_all(
                comm, flag=_petsc.telescope_symbols() is not None
            )
            kept = _take(self.key, self._ordering, self._factor)
            pooled = kept is not None
            if kept is None:
                aij, index = distributed_aij(
                    local, self._numbering, comm, symmetric=self.symmetric
                )
                x, b = aij.createVecs()
                kept = Kept(
                    self.key,
                    aij,
                    index,
                    x,
                    b,
                    self._ordering,
                    self._factor,
                    workspace(self._ordering),
                )
            self._held = kept
            try:
                if pooled:
                    set_values(kept.aij, local.CSR()[0], kept.index)
                self._factorize(kept)
            except BaseException:
                self._held = None
                kept.destroy()
                raise
            _seen.add(self.key)

    def _factorize(self, kept: Kept) -> None:
        """Set ``kept``'s KSP up, a new one at each attempt the workspace may cure."""
        PETSc = _petsc.require_petsc()
        comm = self._comm
        retried = RETRIED if self._readable else (*RETRIED, UNREAD)
        self._retries = 0
        for attempt in range(MUMPS_RETRIES + 1):
            new = kept.ksp is None
            if new:
                kept.ksp = PETSc.KSP().create(comm)
                kept.prefix = _petsc.next_prefix()
                kept.ksp.setOptionsPrefix(kept.prefix)
                kept.pairs = options(
                    self.symmetric,
                    workspace=kept.workspace,
                    factor=kept.factor,
                    ordering=kept.ordering,
                )
            # the same matrix with new values: MUMPS keeps its analysis
            kept.ksp.setOperators(kept.aij)
            with (
                _petsc.scoped_options(kept.prefix, kept.pairs),
                mumps_config(kept.ordering),
            ):
                if new:
                    kept.ksp.setFromOptions()
                kept.ksp.setUp()
                if kept.factor > 1:
                    # the inner factorization happens at the first apply
                    kept.b.set(1.0)
                    kept.ksp.solve(kept.b, kept.x)
            failure = self._agreed_failure()
            if failure is None:
                return
            kept.drop_ksp()
            if failure not in retried or attempt == MUMPS_RETRIES:
                infog = failure if self._readable else "unread behind PCTELESCOPE"
                raise _mpi.AgreedError(
                    "The factorization of BDDC's coarse problem failed (MUMPS "
                    f"INFOG(1)={infog}): the "
                    "operator is singular at this frequency, or MUMPS needs more "
                    f"workspace than ICNTL(14) {kept.workspace} gives."
                )
            kept.workspace *= 2
            self._retries = attempt + 1

    def _agreed_failure(self) -> int | None:
        """``None`` if the set-up succeeded, else INFOG(1) on every rank.

        Behind ``PCTELESCOPE`` the ranks it keeps read the inner factor's and the
        others give 0, so the minimum is the failure. Where the inner factor cannot
        be read (`_petsc.telescope_symbols`), the failure is `UNREAD`, seen as the
        first apply's failed preconditioner or a solution that is not finite.
        """
        PETSc = _petsc.require_petsc()
        pc = self._ksp.getPC()
        failed = pc.getFailedReason() != PETSc.PC.FailedReason.NOERROR
        if not self._readable:
            failed = failed or not np.isfinite(self._x.array).all()
            return UNREAD if _mpi.agree_any(self._comm, flag=failed) else None
        factor = self._inner_factor()
        info = 0 if factor is None else int(factor.getMumpsInfog(1))
        if not _mpi.agree_any(self._comm, flag=failed or info < 0):
            return None
        return _mpi.agree_min(self._comm, info)

    def _inner_factor(self) -> Any:
        """MUMPS's factor matrix on this rank, ``None`` on a rank telescoping drops."""
        return None if self._held is None else self._held.inner_factor()

    @property
    def nbytes(self) -> int:
        """This rank's factors, by MUMPS's own count (INFO(22)).

        0 on a rank telescoping drops, and where the factor behind ``PCTELESCOPE``
        cannot be read (`_petsc.telescope_symbols`).
        """
        return 0 if self._held is None else self._held.nbytes

    def IsComplex(self) -> bool:  # noqa: N802 - NGSolve's name
        """Only complex operators take this route (`gfem._linalg.bddc`)."""
        return True

    def Height(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.height

    def Width(self) -> int:  # noqa: N802 - NGSolve's name
        """The wirebasket's local DOFs."""
        return self._source.width

    def CreateRowVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateRowVector()

    def CreateColVector(self) -> Any:  # ty: ignore[invalid-method-override]  # noqa: N802
        """A vector of the wirebasket."""
        return self._source.CreateColVector()

    def Mult(self, x: Any, y: Any) -> None:  # noqa: N802 - NGSolve's name
        """``y`` = the inverse on ``x``: exchange, owned, MUMPS, owned, exchange."""
        if self._freed:
            raise RuntimeError("This coarse factorization was freed.")
        with _petsc.LOCK:
            if self._ksp is None:
                self.set_up()
            work = self._work
            work.data = x
            work.Cumulate()
            self._b.array[:] = self._numbering.state_rows(work.FV().NumPy())
            self._ksp.solve(self._b, self._x)
            y.FV().NumPy()[:] = self._numbering.state(self._x.array)
            y.SetParallelStatus(ngs.la.PARALLEL_STATUS.CUMULATED)

    def free(self) -> None:
        """Give the factorization back to the pool; every rank calls this together."""
        with _petsc.LOCK:
            self._freed = True
            kept, self._held = self._held, None
            if kept is not None:
                _give(kept)


def coarse_type(comm: Any, symmetric: bool) -> str:
    """The ``coarsetype`` flag of BDDC on ``comm``, registered on first use.

    One per communicator and symmetry, ``gfem_mumps_{key}_{lu|cholesky}``: the
    creator closes over its factorization, since NGSolve gives it no symmetry. Both
    keep their factorizations in one list, for `finish`.
    """
    key = _mpi.comm_key(comm)
    name = f"gfem_mumps_{key}_{'cholesky' if symmetric else 'lu'}"
    _made.setdefault(key, [])
    if name not in _registered:

        def create(matrix: Any, freedofs: Any, *_: Any) -> CoarseMumps:
            made = CoarseMumps(matrix, freedofs, comm, symmetric=symmetric)
            _made.setdefault(key, []).append(made)
            return made

        ngs.comp.RegisterPreconditioner(name, create)
        _registered.add(name)
    return name


def finish(comm: Any, *, columns: int = 1) -> CoarseMumps | None:
    """After a set-up: its factorization, factored; ``None`` on the rank 0 route.

    BDDC keeps the last one NGSolve asked for, and an earlier one was never factored.
    ``columns``: those the set-up is expected to serve, for `rule`.
    """
    made = _made.get(_mpi.comm_key(comm)) if comm is not None else None
    if not made:
        return None
    last = made[-1]
    made.clear()
    last.columns = columns
    last.set_up()
    return last
