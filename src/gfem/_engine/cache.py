"""The engine's caches.

**Discretizations** (`discretization`): the NGSolve mesh of a `gfem.Mesh` at
given node coordinates, and its finite-element spaces keyed by
``(physics domain, space, order)``. Optimization iterations and batch
members on one mesh share them. Bounded by count.

**Solve entries** (`ENTRIES`): what a differentiated forward solve leaves for
its backward pass, a solver `Handle` (factors or preconditioner) and the
forward fields, under an integer key that travels with the JAX residuals.
An entry a backward has read is **consumed**, and is freed when the next
differentiated forward on its communicator starts, before that forward
assembles (ADR 0081): an optimization loop keeps only its current step. Reading
never frees an entry, so ``jax.jacrev`` and repeated pullbacks reuse it. The
rest is bounded in bytes by ``SolverOptions.retain_memory``, least recently
used out first, and only an entry of the storing solve's communicator goes out,
since freeing may be collective.
A missing entry is not an error: the backward pass recomputes it from the
design state in its residuals (time, never correctness).

**The plan pool** (in `ENTRIES`, ADR 0082): serial CPU direct base handles no
solve holds, kept for a later design to refactorize on their solver plan
(`EntryCache.take_plan`). A handle returns (`EntryCache.give_plan`) at the end
of a non-differentiated solve, or when its consumed entry is freed, with its
factors and summed pattern: the next forward refactorizes in place, on memory
already touched. The pool holds several handles per key
(one per ``jax.vmap`` variant). Pooled handles whose mesh topology has no
discretization left are freed.

One byte budget, ``retain_memory``, bounds pooled handles, entries' bases and
entries together. Over it, the walk of a stored entry first releases pooled
handles' PARDISO factors (keeping their plans), then frees pooled handles (both
least recently returned first), then live entries' bases (oldest entry first),
then whole entries (least recently used first); ``retain_memory=0`` keeps none
of them. A handle returning to the pool, or a differentiated forward freeing
consumed entries, walks the pool only: a solve evicts other solves' entries
only where it stores one.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import itertools
import os
import pathlib
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

from gfem import _mpi
from gfem._engine.mesh import EngineMesh, to_ngsolve
from gfem._periodic import Pairing, pair_names
from gfem.mesh import Mesh

__all__ = [
    "ENTRIES",
    "Discretization",
    "Entry",
    "EntryCache",
    "default_retain_memory",
    "discretization",
    "retain_budget",
]

_MAX_DISCRETIZATIONS = 4


@dataclasses.dataclass(eq=False)
class Discretization:
    """The NGSolve mesh of one mesh topology at one set of node coordinates.

    Attributes:
        built: The NGSolve mesh and its element maps.
        spaces: Finite-element spaces by ``(physics domain, space, order)``.
        ports: The wave port couplings and port networks the last driven
            solve on it built, behind their key (`gfem._engine.driven`), or
            ``None``.
    """

    built: EngineMesh
    topology: Any
    spaces: dict[Hashable, Any] = dataclasses.field(default_factory=dict)
    ports: Any = None

    def space(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """The cached space ``key``, built by ``build()`` on first use."""
        if key not in self.spaces:
            self.spaces[key] = build()
        return self.spaces[key]


_DISCRETIZATIONS: collections.OrderedDict[tuple, Discretization] = (
    collections.OrderedDict()
)


def _digest(nodes: np.ndarray) -> str:
    return hashlib.blake2b(
        np.ascontiguousarray(nodes).tobytes(), digest_size=16
    ).hexdigest()


def discretization(
    mesh: Mesh, periodic: tuple[Pairing, ...] = (), comm: Any = None
) -> Discretization:
    """The cached discretization of ``mesh`` at its current node coordinates.

    With ``periodic`` (a model's periodic pairs) the NGSolve mesh carries
    their point identifications (`to_ngsolve`); the pairs are part of the
    key, so a model with pairs and one without do not share a mesh. So is the
    resolved ``comm``. Above one rank a miss is collective, and all ranks miss
    together: their key's nodes and history are the same.
    """
    topology = mesh._topology  # noqa: SLF001
    nodes = np.asarray(mesh.nodes, dtype=float)
    key = (id(topology), _digest(nodes), pair_names(periodic), _mpi.comm_key(comm))
    found = _DISCRETIZATIONS.get(key)
    if found is not None and found.topology is topology:
        _DISCRETIZATIONS.move_to_end(key)
        return found
    entry = Discretization(built=to_ngsolve(mesh, periodic, comm), topology=topology)
    _DISCRETIZATIONS[key] = entry
    if len(_DISCRETIZATIONS) > _MAX_DISCRETIZATIONS:
        while len(_DISCRETIZATIONS) > _MAX_DISCRETIZATIONS:
            _DISCRETIZATIONS.popitem(last=False)
        ENTRIES.free_plans_off({id(d.topology) for d in _DISCRETIZATIONS.values()})
    return entry


def cached(built: EngineMesh) -> Discretization | None:
    """The cached discretization whose NGSolve mesh is ``built``, if still cached."""
    return next((d for d in _DISCRETIZATIONS.values() if d.built is built), None)


# --- solve entries ---------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class Entry:
    """What a forward solve keeps for its backward pass.

    Attributes:
        physics: The physics domain the solve ran on (``"em"``, ...).
        handle: The prepared solver, or ``None`` if the backward needs none.
        fields: Forward fields as NumPy arrays (DOF vectors), by name.
        consumed: Whether a backward has read the entry (ADR 0081).
    """

    physics: str
    handle: Any = None
    fields: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    consumed: bool = False

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes."""
        held = sum(int(f.nbytes) for f in self.fields.values())
        return held + (int(self.handle.nbytes) if self.handle is not None else 0)

    def free(self, *, return_plans: bool = False) -> None:
        """Release the solver handle and the fields.

        With ``return_plans`` (a consumed entry, ADR 0081) a base handle the
        solver handle holds goes to the plan pool instead, where it can
        (handles with a ``free_base``).
        """
        if self.handle is not None:
            if return_plans and hasattr(self.handle, "free_base"):
                self.handle.free(return_plans=True)
            else:
                self.handle.free()
            self.handle = None
        self.fields = {}


_PROC_CGROUP = pathlib.Path("/proc/self/cgroup")
_CGROUP_ROOT = pathlib.Path("/sys/fs/cgroup")


def _physical_memory() -> int | None:
    """The machine's RAM in bytes, or ``None`` if it cannot be read."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _read_limit(path: pathlib.Path) -> int | None:
    """A cgroup limit file's bytes; ``None`` for "max" or an unreadable file."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_memory_limit() -> int | None:
    """The smallest memory limit on this process's cgroup and its ancestors.

    cgroup v2 (``0::<path>``) reads ``memory.max`` under the root; v1 (a line
    whose controllers include ``memory``) reads ``memory.limit_in_bytes`` under
    the root's ``memory`` directory. ``None`` without a readable limit.
    """
    try:
        lines = _PROC_CGROUP.read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and not controllers:
            base, name = _CGROUP_ROOT, "memory.max"
        elif "memory" in controllers.split(","):
            base, name = _CGROUP_ROOT / "memory", "memory.limit_in_bytes"
        else:
            continue
        directory = base / path.strip().lstrip("/")
        while True:
            limit = _read_limit(directory / name)
            if limit is not None:
                limits.append(limit)
            if directory == base:
                break
            directory = directory.parent
    return min(limits, default=None)


@functools.cache
def default_retain_memory() -> int:
    """Half of the memory this process may use, in bytes.

    The smaller of the physical memory and the process's cgroup memory limit
    (searched up through parent cgroups, so a job's limit counts); "max" or an
    unreadable limit means none. Without readable physical memory or a limit,
    8 GiB. Read once per process.
    """
    sizes = [s for s in (_physical_memory(), _cgroup_memory_limit()) if s is not None]
    return min(sizes, default=16 * 2**30) // 2


def retain_budget(retain_memory: int | None) -> int:
    """The byte budget of ``SolverOptions.retain_memory`` (``None``: the default)."""
    return default_retain_memory() if retain_memory is None else int(retain_memory)


@dataclasses.dataclass(eq=False)
class _Pooled:
    """A handle in the plan pool, its key, and the mesh topology of its plan."""

    key: Hashable
    topology: Any
    handle: Any


class EntryCache:
    """Solve entries by integer key, and the plan pool, under one byte budget."""

    def __init__(self) -> None:
        """An empty cache."""
        self._entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()
        self._comms: dict[int, int | None] = {}
        self._keys = itertools.count()
        # least recently returned first
        self._plans: collections.OrderedDict[int, _Pooled] = collections.OrderedDict()

    def put(self, entry: Entry, budget: int, comm: Any = None) -> int:
        """Store ``entry`` and return its key; evict down to ``budget`` bytes.

        An entry larger than the budget alone is kept without its base handle
        when that fits, and is otherwise freed at once and evicts nothing (with
        ``budget=0`` nothing is kept); its key then reads as evicted, so the
        backward pass recomputes it. Eviction is the walk of the module
        docstring: pooled factors, pooled handles, entries' bases, entries.

        Under ``comm`` the bytes are the largest node's and the budget the smallest
        node's, so all ranks evict alike: a rank's own bytes would leave some ranks
        to re-run a forward without the others, and hang (ADR 0058). All entries count
        against the budget, but only ``comm``'s are evicted: freeing a base may be
        collective on a communicator whose ranks are not here, so the budget does not
        bound what a solve on another communicator keeps. The plan pool is serial:
        only a one-process walk counts it and frees from it.
        """
        key = next(self._keys)
        budget = _mpi.node_budget(comm, budget)
        if _mpi.node_bytes(comm, entry.nbytes) > budget:
            free_base = getattr(entry.handle, "free_base", None)
            if free_base is not None:
                free_base(comm)
            if _mpi.node_bytes(comm, entry.nbytes) > budget:
                entry.free()
                return key
        self._entries[key] = entry
        self._comms[key] = _mpi.comm_key(comm)
        self._evict(budget, comm)
        return key

    def _evict(self, budget: int, comm: Any) -> None:
        """Free down to ``budget`` (the node's): the pool, bases, entries."""
        owner = _mpi.comm_key(comm)

        def over() -> bool:
            # the serial pool counts only where the walk can free it
            held = self.nbytes if comm is None else self._entry_nbytes
            return _mpi.node_bytes(comm, held) > budget

        if comm is None:
            self._free_pool(budget)
        for key, entry in list(self._entries.items()):
            free_base = getattr(entry.handle, "free_base", None)
            if self._comms[key] != owner or free_base is None:
                continue
            if not over():
                return
            free_base(comm)
        while over():
            oldest = next((k for k in self._entries if self._comms[k] == owner), None)
            if oldest is None:
                break
            del self._comms[oldest]
            self._entries.pop(oldest).free()

    def get(self, key: int) -> Entry | None:
        """The entry under ``key``, or ``None`` if it was evicted."""
        entry = self._entries.get(int(key))
        if entry is not None:
            self._entries.move_to_end(int(key))
        return entry

    def discard(self, key: int) -> None:
        """Free the entry under ``key``, if this process still holds it."""
        entry = self._entries.pop(int(key), None)
        if entry is not None:
            del self._comms[int(key)]
            entry.free()

    def free_consumed(self, comm: Any = None, budget: int | None = None) -> None:
        """Free the consumed entries of ``comm`` (ADR 0081).

        Called when a differentiated forward on ``comm`` starts. Under ranks the
        backward is replicated, so every rank frees the same entries together.
        With a ``budget`` their base handles return to the plan pool, where
        their handles can (`Entry.free`), and on one process pooled handles go
        down to it. No entry that is not consumed goes: only `put` evicts
        entries.
        """
        owner = _mpi.comm_key(comm)
        consumed = [
            k
            for k, e in self._entries.items()
            if e.consumed and self._comms[k] == owner
        ]
        # out of the cache before any is freed: a base returning to the pool
        # walks the cache
        freed = []
        for key in consumed:
            del self._comms[key]
            freed.append(self._entries.pop(key))
        for entry in freed:
            entry.free(return_plans=budget is not None)
        if budget is not None and comm is None:
            self._free_pool(budget)

    # --- the plan pool -----------------------------------------------------------

    def give_plan(self, key: Hashable, topology: Any, handle: Any, budget: int) -> None:
        """Pool ``handle`` under ``key``, with its factors, within ``budget``.

        While the cache is over ``budget``, pooled handles' factors are
        released, then pooled handles freed, least recently returned first;
        entries and their bases stay (only `put` evicts them),
        so another solve's pending entries outlive a ``retain_memory=0`` solve.
        ``topology`` is the mesh topology of its plan: the handle is freed once
        no discretization of it is left (`free_plans_off`). One process only.
        """
        self._plans[id(handle)] = _Pooled(key, topology, handle)
        self._free_pool(budget)

    def _free_pool(self, budget: int) -> None:
        """Walk the pool down to ``budget``, least recently returned first.

        Releasing factors keeps a handle's plan, so it goes before freeing any.
        """
        for pooled in self._plans.values():
            if self.nbytes <= budget:
                return
            release = getattr(pooled.handle, "release_factors", None)
            if release is not None:
                release()
        while self._plans and self.nbytes > budget:
            self._plans.popitem(last=False)[1].handle.free()

    def take_plan(self, key: Hashable) -> Any | None:
        """The most recently returned pooled handle under ``key``, or ``None``.

        It leaves the pool: no other solve takes it until it returns.
        """
        found = next(
            (k for k, p in reversed(self._plans.items()) if p.key == key), None
        )
        return None if found is None else self._plans.pop(found).handle

    def free_plans_off(self, topologies: set[int]) -> None:
        """Free the pooled handles whose topology's id is not in ``topologies``."""
        for k, pooled in list(self._plans.items()):
            if id(pooled.topology) not in topologies:
                del self._plans[k]
                pooled.handle.free()

    @property
    def pooled(self) -> int:
        """The handles in the plan pool."""
        return len(self._plans)

    def free_bases(self, comm: Any, keep: Entry | None = None) -> None:
        """Free the base handles kept on ``comm``, all but ``keep``'s (ADR 0065).

        A design loop under ranks holds one factorization at a time. An entry keeps
        its fields, and a later backward from it prepares again. Nothing happens on
        the one-process path, where each entry keeps its handles.
        """
        if comm is None:
            return
        for entry in self._entries.values():
            free_base = getattr(entry.handle, "free_base", None)
            if entry is not keep and free_base is not None:
                free_base(comm)

    @property
    def nbytes(self) -> int:
        """Bytes held by every entry and every pooled handle."""
        return self._entry_nbytes + sum(
            int(p.handle.nbytes) for p in self._plans.values()
        )

    @property
    def _entry_nbytes(self) -> int:
        """Bytes held by every entry."""
        return sum(e.nbytes for e in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Free every entry and every pooled handle."""
        self._comms.clear()
        while self._entries:
            _, old = self._entries.popitem(last=False)
            old.free()
        while self._plans:
            self._plans.popitem(last=False)[1].handle.free()


ENTRIES = EntryCache()
