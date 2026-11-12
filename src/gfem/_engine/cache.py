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

**The plan pool** (in `ENTRIES`, ADRs 0082, 0093): serial CPU direct base
handles no solve holds, kept for a later design to refactorize on their solver
plan (`EntryCache.take_plan`). A handle returns (`EntryCache.give_plan`) once
no solve entry holds it: at the end of a non-differentiated solve, when its
consumed entry is freed, or when an entry gives it up under ``retain_memory``.
It returns with its factors and summed pattern: the next forward refactorizes
in place, on memory already touched. An entry that gave up its base serves its
backward by taking a plan back from the pool, or by preparing one. The pool
holds several handles per key (one per ``jax.vmap`` variant). Pooled handles
whose mesh topology has no discretization left are freed.

**Kept discretization state** (ADRs 0082, 0091): a discretization's kept fixed
parts (`KeptParts` stores among its spaces) and its kept wave port couplings and
port networks (``Discretization.ports``).

One byte budget, ``retain_memory``, bounds pooled handles, entries, their bases
and the kept discretization state together. Objects several holders reference
(couplings, networks, kept parts, a pieces' pattern) are counted once, by
identity, at the cache: an entry's own bytes never include them. Over budget,
the walk of a stored entry (`EntryCache.put`):

1. releases pooled handles' PARDISO factors, keeping their plans;
2. drops kept fixed parts, whole stores, least recently used discretization
   first;
3. frees pooled handles, least recently returned first;
4. moves live entries' bases to the pool, oldest entry first, where tiers 1
   and 3 run on them again (freed where they cannot pool);
5. frees whole entries, least recently used first;
6. drops kept port state, least recently used discretization first.

A handle returning to the pool, or a differentiated forward freeing consumed
entries, walks tiers 1, 2, 3 and 6 only: a solve evicts other solves' entries
only where it stores one. Eviction drops references and the keys that find
them; nothing kept is changed, so a live entry keeps reading its own objects and
a later solve rebuilds what was dropped. ``retain_memory=0`` keeps none of it.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import itertools
import os
import pathlib
from collections.abc import Callable, Collection, Hashable, Iterable
from typing import Any

import numpy as np

from gfem import _mpi
from gfem._engine.mesh import EngineMesh, to_ngsolve
from gfem._linalg.pieces import KeptParts
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
            ``None``. Anything with a ``shared()`` of its bytes by identity.
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

    @property
    def comm_key(self) -> int | None:
        """The key of the NGSolve mesh's communicator (`gfem._mpi.comm_key`)."""
        return _mpi.comm_key(self.built.comm)

    def _stores(self) -> list[KeptParts]:
        return [s for s in self.spaces.values() if isinstance(s, KeptParts)]

    def shared(self) -> dict[int, int]:
        """The kept fixed parts' and port state's bytes, by object identity."""
        held: dict[int, int] = {}
        for store in self._stores():
            held.update(store.shared())
        if self.ports is not None:
            held.update(self.ports.shared())
        return held

    @property
    def has_fixed_parts(self) -> bool:
        """Whether a kept fixed part store holds a part or a digest seen."""
        return any(s.kept or s.seen for s in self._stores())

    def drop_fixed_parts(self) -> None:
        """Clear every kept fixed part store (`KeptParts.clear`)."""
        for store in self._stores():
            store.clear()

    def drop_ports(self) -> None:
        """Drop the kept port state: the next driven solve builds it again."""
        self.ports = None


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
        """Host memory held, in bytes, but the shared objects (`shared`)."""
        held = sum(int(f.nbytes) for f in self.fields.values())
        return held + (int(self.handle.nbytes) if self.handle is not None else 0)

    def shared(self) -> dict[int, int]:
        """Bytes of the objects the handle may share with others, by identity."""
        return _shared(self.handle)

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


def _shared(holder: Any) -> dict[int, int]:
    """``holder.shared()``, or nothing for a holder without one."""
    shared = getattr(holder, "shared", None)
    return {} if shared is None else shared()


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


_Held = tuple[int, dict[int, int]]
_NOTHING: _Held = (0, {})


class _Tally:
    """Bytes of several holders: their own, and shared objects once each.

    A holder reads as ``(own bytes, shared bytes by identity)`` (`of`); a
    shared object counts while one holder references it.
    """

    def __init__(self) -> None:
        self._own = 0
        self._refs: dict[int, int] = {}
        self._bytes: dict[int, int] = {}

    @staticmethod
    def of(holder: Any) -> _Held:
        """``holder``'s own bytes and its shared objects' bytes.

        A discretization's kept state is all shared.
        """
        own = 0 if isinstance(holder, Discretization) else int(holder.nbytes)
        return own, _shared(holder)

    def replace(self, before: _Held, after: _Held) -> None:
        """A holder that read ``before`` now reads ``after``."""
        self._own += after[0] - before[0]
        for key, size in after[1].items():
            self._refs[key] = self._refs.get(key, 0) + 1
            self._bytes[key] = size
        for key in before[1]:
            self._refs[key] -= 1
            if not self._refs[key]:
                del self._refs[key], self._bytes[key]

    @property
    def total(self) -> int:
        """The bytes held."""
        return self._own + sum(self._bytes.values())


class _Walk:
    """A walk's tally against its budget (the node's under ``comm``)."""

    def __init__(self, tally: _Tally, budget: int, comm: Any) -> None:
        self._tally, self._budget, self._comm = tally, budget, comm

    def over(self) -> bool:
        """Whether the node's bytes exceed the budget (collective under ranks)."""
        return _mpi.node_bytes(self._comm, self._tally.total) > self._budget

    def step(self, holder: Any, act: Callable[[], Any], *, gone: bool = False) -> bool:
        """Run ``act`` on ``holder`` if over budget; whether it ran.

        ``gone``: ``holder`` has left the cache, and its bytes leave the tally;
        the caller checked the budget.
        """
        if not gone and not self.over():
            return False
        before = _Tally.of(holder)
        act()
        self._tally.replace(before, _NOTHING if gone else _Tally.of(holder))
        return True

    def move(self, holder: Any, act: Callable[[], Any]) -> Any:
        """Run ``act`` on ``holder`` and re-read it; what ``act`` returned.

        The caller checked the budget, and `enter`s where the bytes went.
        """
        before = _Tally.of(holder)
        out = act()
        self._tally.replace(before, _Tally.of(holder))
        return out

    def enter(self, holder: Any) -> None:
        """Count a holder that joined the cache."""
        self._tally.replace(_NOTHING, _Tally.of(holder))


@dataclasses.dataclass(eq=False)
class _Pooled:
    """A handle in the plan pool, its key, and the mesh topology of its plan."""

    key: Hashable
    topology: Any
    handle: Any


class EntryCache:
    """Solve entries by integer key, the plan pool and kept discretization state.

    One byte budget bounds them together (see the module docstring).
    """

    def __init__(
        self, discretizations: Callable[[], Iterable[Discretization]] = tuple
    ) -> None:
        """An empty cache counting the kept state of ``discretizations()``.

        They come least recently used first; ``ENTRIES`` reads the cached ones.
        """
        self._entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()
        self._comms: dict[int, int | None] = {}
        self._keys = itertools.count()
        # least recently returned first
        self._plans: collections.OrderedDict[int, _Pooled] = collections.OrderedDict()
        self._discretizations = discretizations
        # an entry is giving up its base: its walker walks, `give_plan` does not
        self._giving_up = False

    def put(self, entry: Entry, budget: int, comm: Any = None) -> int:
        """Store ``entry`` and return its key; evict down to ``budget`` bytes.

        An entry whose own bytes exceed the budget gives up its base handle
        (to the plan pool, where it can: the walk then releases its factors,
        and frees it only if it still does not fit). It is kept without it
        when that fits, and is otherwise freed at once and evicts no other
        entry (with ``budget=0`` nothing is kept); its key then reads as evicted,
        so the backward pass recomputes it. Eviction is the walk of the module
        docstring, all six tiers; a freed entry's walk skips tiers 4 and 5.

        Under ``comm`` the bytes are the largest node's and the budget the smallest
        node's, so all ranks evict alike: a rank's own bytes would leave some ranks
        to re-run a forward without the others, and hang (ADR 0058). All entries count
        against the budget, but only ``comm``'s are evicted: freeing a base may be
        collective on a communicator whose ranks are not here, so the budget does not
        bound what a solve on another communicator keeps. The plan pool and kept
        fixed parts are serial: only a one-process walk counts them and frees them.
        """
        key = next(self._keys)
        budget = _mpi.node_budget(comm, budget)
        if _mpi.node_bytes(comm, entry.nbytes) > budget:
            self._give_up_base(entry, comm)
            if _mpi.node_bytes(comm, entry.nbytes) > budget:
                entry.free()
                self._walk(budget, comm, entries=False)
                return key
        self._entries[key] = entry
        self._comms[key] = _mpi.comm_key(comm)
        self._walk(budget, comm, entries=True)
        return key

    def trim(self, budget: int, comm: Any = None) -> None:
        """Walk tiers 1, 2, 3 and 6 down to ``budget``; no entry goes.

        For a backward that reran its forward and freed that entry: the kept
        state the forward rebuilt is bounded as at `put`.
        """
        self._walk(_mpi.node_budget(comm, budget), comm, entries=False)

    def _walk(self, budget: int, comm: Any, *, entries: bool) -> None:
        """Free down to ``budget`` (the node's) by the six tiers; 4-5 if ``entries``.

        The bytes are tallied once, then follow each step: a step re-reads only
        the holder it changed, and a shared object leaves the tally with the
        last holder that references it.
        """
        owner = _mpi.comm_key(comm)
        discs = [d for d in self._discretizations() if d.comm_key == owner]
        walk = _Walk(self._tally(discs, pool=comm is None), budget, comm)
        if comm is None and self._walk_pool(walk, discs):
            return
        if entries and self._walk_entries(walk, owner, comm, pool=comm is None):
            return
        for disc in discs:
            if disc.ports is not None and not walk.step(disc, disc.drop_ports):
                return

    def _walk_pool(self, walk: _Walk, discs: list[Discretization]) -> bool:
        """Tiers 1, 2 and 3 (one process); whether the cache fits after."""
        for pooled in self._plans.values():
            release = getattr(pooled.handle, "release_factors", None)
            if release is not None and not walk.step(pooled.handle, release):
                return True
        for disc in discs:
            if disc.has_fixed_parts and not walk.step(disc, disc.drop_fixed_parts):
                return True
        return self._free_plans(walk)

    def _free_plans(self, walk: _Walk) -> bool:
        """Tier 3: free pooled handles, oldest returned first; whether it fits."""
        while self._plans and walk.over():
            handle = self._plans.popitem(last=False)[1].handle
            walk.step(handle, handle.free, gone=True)
        return not walk.over()

    def _walk_entries(
        self, walk: _Walk, owner: int | None, comm: Any, *, pool: bool
    ) -> bool:
        """Tiers 4 and 5 (`put`): ``owner``'s bases, then entries; whether it fits.

        With ``pool`` (one process) a base given up to the pool joins the tally
        there, and tiers 1 and 3 run on it before the next entry's base goes.
        """
        for key, entry in list(self._entries.items()):
            if self._comms[key] != owner or not hasattr(entry.handle, "free_base"):
                continue
            if not walk.over():
                return True
            pooled = walk.move(
                entry, functools.partial(self._give_up_base, entry, comm)
            )
            if not pool:
                continue
            for handle in pooled:
                walk.enter(handle)
                release = getattr(handle, "release_factors", None)
                if release is not None:
                    walk.step(handle, release)
            if self._free_plans(walk):
                return True
        while walk.over():
            oldest = next((k for k in self._entries if self._comms[k] == owner), None)
            if oldest is None:
                return False
            del self._comms[oldest]
            entry = self._entries.pop(oldest)
            walk.step(entry, entry.free, gone=True)
        return True

    def _give_up_base(self, entry: Entry, comm: Any) -> list[Any]:
        """``entry``'s base handle to the plan pool, without a walk (ADR 0093).

        Where its handle can (`free_base`): other bases are freed. The handles
        it pooled, for the caller's walk to count.
        """
        free_base = getattr(entry.handle, "free_base", None)
        if free_base is None:
            return []
        before = set(self._plans)
        self._giving_up = True
        try:
            free_base(comm)
        finally:
            self._giving_up = False
        return [p.handle for k, p in self._plans.items() if k not in before]

    def _tally(self, discs: Iterable[Discretization], *, pool: bool) -> _Tally:
        """Every entry, with ``pool`` the pool's handles, and ``discs``' kept state."""
        tally = _Tally()
        holders: list[Any] = [*self._entries.values()]
        if pool:
            holders += [p.handle for p in self._plans.values()]
        holders += list(discs)
        for holder in holders:
            tally.replace(_NOTHING, _Tally.of(holder))
        return tally

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
        their handles can (`Entry.free`), and the walk's tiers 1, 2, 3 and 6
        run down to it. No entry that is not consumed goes: only `put` evicts
        entries.
        """
        owner = _mpi.comm_key(comm)
        consumed = [
            k
            for k, e in self._entries.items()
            if e.consumed and self._comms[k] == owner
        ]
        # out of the cache before any is freed: a base returning to the pool
        # walks the pool against the cache's bytes, which must not count the
        # consumed entries still to be freed
        freed = []
        for key in consumed:
            del self._comms[key]
            freed.append(self._entries.pop(key))
        for entry in freed:
            entry.free(return_plans=budget is not None)
        if budget is not None:
            self._walk(_mpi.node_budget(comm, budget), comm, entries=False)

    # --- the plan pool -----------------------------------------------------------

    def give_plan(
        self,
        key: Hashable,
        topology: Any,
        handle: Any,
        budget: int,
        kept: Collection[Any] = (),
    ) -> None:
        """Pool ``handle`` under ``key``, with its factors, within ``budget``.

        While the cache is over ``budget``, the walk's tiers 1, 2, 3 and 6 run,
        unless an entry is giving the handle up (that walk counts it);
        entries and their bases stay (only `put` evicts them),
        so another solve's pending entries outlive a ``retain_memory=0`` solve.
        ``topology`` is the mesh topology of its plan: the handle is freed once
        no discretization of it is left (`free_plans_off`). One process only.

        ``kept`` holds the patterns of the kept fixed parts (ADR 0082) of the
        handle's space: a handle's pieces' pattern that is none of them is
        dropped, since no later piece sum can share it.
        """
        pattern = getattr(handle, "pattern", None)
        if pattern is not None and not any(pattern is k for k in kept):
            handle.drop_pattern()
        self._plans[id(handle)] = _Pooled(key, topology, handle)
        if not self._giving_up:
            self._walk(budget, None, entries=False)

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
        """Bytes held by every entry, every pooled handle and the kept state.

        Each object several of them reference is counted once.
        """
        return self._tally(self._discretizations(), pool=True).total

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


ENTRIES = EntryCache(_DISCRETIZATIONS.values)
