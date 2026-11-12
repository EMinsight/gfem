"""Kept state counts against ``retain_memory`` (ADRs 0091, 0093).

Kept fixed parts (`KeptParts` stores) and kept port state
(``Discretization.ports``) count in `EntryCache.nbytes`, each object once, by
identity. Over budget the walk releases pooled factors, drops kept fixed parts,
frees pooled plans, then (in `put` only) entries' bases and entries, and last
drops kept port state. Eviction drops references and keys only.
"""

import types

import numpy as np

from gfem._engine.cache import Discretization, Entry, EntryCache
from gfem._linalg.pieces import KeptParts

TOPOLOGY = object()


class FakeHandle:
    def __init__(self, nbytes, *, released_bytes=None, log=None, name=""):
        self._nbytes = nbytes
        self._released_bytes = nbytes if released_bytes is None else released_bytes
        self.released = False
        self.freed = False
        self._log, self._name = log, name

    @property
    def nbytes(self):
        if self.freed:
            return 0
        return self._released_bytes if self.released else self._nbytes

    def release_factors(self):
        if self._log is not None and not self.released:
            self._log.append(f"release {self._name}")
        self.released = True

    def free(self):
        self.freed = True
        if self._log is not None:
            self._log.append(f"free {self._name}")


class FakeKept:
    """An entry's handle that holds a base and references shared arrays."""

    def __init__(self, nbytes, base=None, shared=(), log=None, name=""):
        self._nbytes = nbytes
        self.base = base
        self._shared = list(shared)
        self._log, self._name = log, name

    @property
    def nbytes(self):
        return self._nbytes + (0 if self.base is None else self.base.nbytes)

    def shared(self):
        return {id(a): a.nbytes for a in self._shared}

    def free_base(self, comm, *, walk=True):  # noqa: ARG002
        if self.base is not None:
            self.base.free()
            if self._log is not None:
                self._log.append(f"base {self._name}")
        self.base = None

    def free(self, *, return_plans=False):  # noqa: ARG002
        self.free_base(None)
        self._shared = []
        if self._log is not None:
            self._log.append(f"entry {self._name}")


class FakeParts:
    """A kept fixed part: its arrays."""

    def __init__(self, *arrays):
        self.arrays = arrays

    def shared(self):
        return {id(a): a.nbytes for a in self.arrays}


class FakePorts:
    def __init__(self, *arrays):
        self.arrays = arrays

    def shared(self):
        return {id(a): a.nbytes for a in self.arrays}


def _disc(fixed=0, ports=0):
    """A discretization keeping ``fixed`` bytes of fixed parts, ``ports`` of ports."""
    disc = Discretization(built=types.SimpleNamespace(comm=None), topology=TOPOLOGY)
    if fixed:
        store = disc.space(("em", "hcurl", 1, "fixed pieces"), KeptParts)
        store.kept["stiffness"] = FakeParts(np.zeros(fixed, np.uint8))
        store.seen["stiffness"] = "d"
    if ports:
        disc.ports = FakePorts(np.zeros(ports, np.uint8))
    return disc


def _store(disc):
    return disc.spaces[("em", "hcurl", 1, "fixed pieces")]


def _logged(entries, discs, log):
    """Log each fixed-part and port drop of ``discs``, in walk order."""
    for i, disc in enumerate(discs):
        store = _store(disc) if disc.spaces else None
        if store is not None:
            clear = store.clear

            def logged_clear(clear=clear, i=i):
                log.append(f"fixed {i}")
                clear()

            store.clear = logged_clear
    return entries


def test_kept_state_counts_once_by_identity():
    shared = np.zeros(100, np.uint8)
    disc = _disc(fixed=30, ports=0)
    disc.ports = FakePorts(shared)
    entries = EntryCache(discretizations=lambda: [disc])
    entries.put(Entry(physics="em", handle=FakeKept(5, shared=[shared])), budget=1000)
    assert entries.nbytes == 5 + 100 + 30


def test_walks_outside_put_drop_fixed_parts_between_pooled_factors_and_plans():
    log = []
    old, new = _disc(fixed=10), _disc(fixed=10)
    discs = [old, new]  # least recently used first
    entries = _logged(EntryCache(discretizations=lambda: discs), discs, log)
    a = FakeHandle(100, released_bytes=40, log=log, name="a")
    b = FakeHandle(100, released_bytes=40, log=log, name="b")
    entries.give_plan(("a",), TOPOLOGY, a, budget=10**6)
    entries.give_plan(("b",), TOPOLOGY, b, budget=10**6)
    # 220 over 0: every factor, every fixed part, every plan
    entries.free_consumed(comm=None, budget=0)
    assert log == ["release a", "release b", "fixed 0", "fixed 1", "free a", "free b"]
    assert entries.nbytes == 0
    assert not _store(old).kept
    assert not _store(old).seen


def test_a_walk_stops_at_the_first_tier_that_fits():
    log = []
    old, new = _disc(fixed=50), _disc(fixed=50)
    discs = [old, new]
    entries = _logged(EntryCache(discretizations=lambda: discs), discs, log)
    pooled = FakeHandle(100, released_bytes=40, log=log, name="a")
    # 200 over 150: the release (to 140) is enough
    entries.give_plan(("a",), TOPOLOGY, pooled, budget=150)
    assert log == ["release a"]
    # 140 over 100: the least recently used discretization's fixed parts go
    entries.free_consumed(comm=None, budget=100)
    assert log == ["release a", "fixed 0"]
    assert _store(new).kept


def test_walks_outside_put_drop_port_state_last_and_leave_entries():
    discs = [_disc(ports=20), _disc(ports=20)]
    entries = EntryCache(discretizations=lambda: discs)
    base = FakeHandle(50)
    key = entries.put(Entry(physics="em", handle=FakeKept(10, base)), budget=10**6)
    # 100 over 90
    entries.free_consumed(comm=None, budget=90)
    # the older discretization's ports go first; entries and bases stay
    assert discs[0].ports is None
    assert discs[1].ports is not None
    assert not base.freed
    entries.free_consumed(comm=None, budget=0)
    assert discs[1].ports is None
    assert entries.get(key) is not None


def test_put_walks_all_six_tiers_in_order():
    log = []
    discs = [_disc(fixed=10, ports=10)]
    entries = _logged(EntryCache(discretizations=lambda: discs), discs, log)
    pooled = FakeHandle(100, released_bytes=40, log=log, name="pool")
    entries.give_plan(("a",), TOPOLOGY, pooled, budget=10**6)
    base = FakeHandle(50)
    entries.put(
        Entry(physics="em", handle=FakeKept(30, base, log=log, name="old")),
        budget=10**6,
    )
    ports = discs[0].ports
    entries.put(Entry(physics="em", handle=FakeKept(0, log=log, name="new")), budget=0)
    assert log == [
        "release pool",
        "fixed 0",
        "free pool",
        "base old",
        "entry old",
        "entry new",
    ]
    assert ports is not None
    assert discs[0].ports is None
    assert entries.nbytes == 0


def test_bases_and_entries_go_before_port_state_in_put():
    discs = [_disc(ports=10)]
    entries = EntryCache(discretizations=lambda: discs)
    base = FakeHandle(50)
    key = entries.put(Entry(physics="em", handle=FakeKept(30, base)), budget=10**6)
    # 90 over 40: the base goes (to 40), the entry and the ports stay
    entries.put(Entry(physics="em", handle=FakeKept(0)), budget=40)
    assert base.freed
    assert entries.get(key) is not None
    assert discs[0].ports is not None


def test_an_entry_freed_at_put_still_walks_the_kept_state():
    discs = [_disc(fixed=10, ports=10)]
    entries = EntryCache(discretizations=lambda: discs)
    entries.put(Entry(physics="em", handle=FakeKept(5)), budget=0)
    assert entries.nbytes == 0
    assert discs[0].ports is None
    assert not _store(discs[0]).kept


def test_a_store_a_live_entry_holds_too_stays_kept_with_its_digests():
    values = np.zeros(40, np.uint8)
    disc = _disc()
    store = disc.space(("em", "hcurl", 1, "fixed pieces"), KeptParts)
    store.kept["mass"] = FakeParts(values)
    store.seen["mass"] = "d"
    entries = EntryCache(discretizations=lambda: [disc])
    kept = FakeKept(1, shared=[values])
    key = entries.put(Entry(physics="em", handle=kept), budget=10**6)
    # dropping the store would free nothing: the live entry holds its values
    entries.free_consumed(comm=None, budget=0)
    assert store.kept
    assert store.seen == {"mass": "d"}
    assert entries.get(key) is not None
    assert entries.nbytes == 1 + 40


def test_a_store_no_other_holder_references_is_dropped():
    held, alone = np.zeros(40, np.uint8), np.zeros(30, np.uint8)
    shared_disc, alone_disc = _disc(), _disc()
    for disc, values in ((shared_disc, held), (alone_disc, alone)):
        store = disc.space(("em", "hcurl", 1, "fixed pieces"), KeptParts)
        store.kept["mass"] = FakeParts(values)
        store.seen["mass"] = "d"
    entries = EntryCache(discretizations=lambda: [shared_disc, alone_disc])
    entries.put(Entry(physics="em", handle=FakeKept(1, shared=[held])), budget=10**6)
    entries.free_consumed(comm=None, budget=50)
    # the older store is held by the entry: the newer one goes, and fits
    assert _store(shared_disc).kept
    assert not _store(alone_disc).kept
    assert not _store(alone_disc).seen
    assert entries.nbytes == 1 + 40


class PoolingKept(FakeKept):
    """An entry's handle whose base goes back to the plan pool (ADR 0093)."""

    def __init__(self, entries, budget, nbytes, base=None, **kwargs):
        super().__init__(nbytes, base, **kwargs)
        self._entries, self._budget = entries, budget

    def free_base(self, comm, *, walk=True):  # noqa: ARG002
        if self.base is not None:
            self._entries.give_plan(
                (self._name,), TOPOLOGY, self.base, self._budget, walk=walk
            )
        self.base = None


def test_an_entry_over_budget_gives_its_base_to_the_pool_released():
    entries = EntryCache()
    base = FakeHandle(100, released_bytes=30)
    entry = Entry(physics="em", handle=PoolingKept(entries, 60, 20, base))
    # 120 over 60: the base goes to the pool, where its factors go (to 50)
    key = entries.put(entry, budget=60)
    assert entries.get(key) is entry
    assert entries.pooled == 1
    assert base.released
    assert not base.freed
    assert entries.nbytes == 50


def test_an_entry_giving_up_its_base_walks_in_put_order():
    discs = [_disc(ports=10)]
    entries = EntryCache(discretizations=lambda: discs)
    old_base = FakeHandle(50, released_bytes=40)
    entries.put(
        Entry(physics="em", handle=PoolingKept(entries, 60, 30, old_base, name="old")),
        budget=10**6,
    )
    new_base = FakeHandle(100)
    new = PoolingKept(entries, 60, 5, new_base, name="new")
    # 105 over 60: the new base goes to the pool and is freed there; then the
    # old base goes to the pool released, and is freed (5 + 30 + 10 = 45)
    entries.put(Entry(physics="em", handle=new), budget=60)
    assert new_base.freed
    assert old_base.released
    assert old_base.freed
    # port state goes last: it never went
    assert discs[0].ports is not None
    assert entries.nbytes == 45


def test_a_base_given_up_in_the_walk_frees_entries_before_port_state():
    discs = [_disc(ports=10)]
    entries = EntryCache(discretizations=lambda: discs)
    base = FakeHandle(50, released_bytes=40)
    old = PoolingKept(entries, 35, 30, base, name="old")
    old_key = entries.put(Entry(physics="em", handle=old), budget=10**6)
    # 90 over 35: the old base goes to the pool released, then is freed (40);
    # then the old entry goes (10), and the port state stays
    new_key = entries.put(Entry(physics="em", handle=FakeKept(0)), budget=35)
    assert base.freed
    assert entries.get(old_key) is None
    assert entries.get(new_key) is not None
    assert discs[0].ports is not None


def test_an_entry_whose_fields_alone_are_over_budget_walks_the_pool():
    entries = EntryCache()
    pooled = FakeHandle(40, released_bytes=10)
    entries.give_plan(("other",), TOPOLOGY, pooled, budget=10**6)
    base = FakeHandle(100, released_bytes=30)
    entry = Entry(physics="em", handle=PoolingKept(entries, 30, 50, base))
    # the fields alone (50) are over 30: the entry goes, its base went to the
    # pool, and the walk frees down to the budget: every factor (to 40), then
    # the least recently returned plan
    key = entries.put(entry, budget=30)
    assert entries.get(key) is None
    assert base.released
    assert pooled.freed
    assert not base.freed
    assert entries.nbytes == 30
