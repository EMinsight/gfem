"""The plan pool: serial CPU direct solves refactorize a held handle (ADR 0082).

A driven solve on one process with the CPU ``direct`` solver takes its base
handle from the plan pool when a handle of the same key is free there, and
refactorizes it instead of planning again. A handle returns to the pool at the
end of a non-differentiated solve, or when its consumed entry is freed (ADR
0081), its PARDISO factors released. ``retain_memory`` bounds pooled handles,
entries' bases and entries with one walk, in that order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache
from gfem._engine.cache import Entry, EntryCache
from gfem._linalg.direct import DirectHandle

F = (21e9, 24e9)
WEIGHTS = jnp.asarray([[1.0, 0.5], [2.0, -1.0]])


# --- the pool in the entry cache ---------------------------------------------------


class FakeHandle:
    def __init__(self, nbytes, *, released_bytes=None):
        self._nbytes = nbytes
        self._released_bytes = nbytes if released_bytes is None else released_bytes
        self.released = False
        self.freed = False

    @property
    def nbytes(self):
        return self._released_bytes if self.released else self._nbytes

    def release_factors(self):
        self.released = True

    def free(self):
        self.freed = True


class FakeKept:
    """An entry's handle that holds a base: `free_base` drops it."""

    def __init__(self, nbytes, base):
        self._nbytes = nbytes
        self.base = base

    @property
    def nbytes(self):
        return self._nbytes + (0 if self.base is None else self.base.nbytes)

    def free_base(self, comm):  # noqa: ARG002
        if self.base is not None:
            self.base.free()
        self.base = None

    def free(self, *, return_plans=False):  # noqa: ARG002
        self.free_base(None)


TOPOLOGY = object()


def test_a_pooled_handle_is_released_and_taken_back_under_its_key():
    entries = EntryCache()
    handle = FakeHandle(100, released_bytes=30)
    entries.give_plan(("k",), TOPOLOGY, handle, budget=1000)
    assert handle.released
    assert entries.nbytes == 30
    assert entries.take_plan(("other",)) is None
    assert entries.take_plan(("k",)) is handle
    assert entries.take_plan(("k",)) is None
    assert entries.nbytes == 0


def test_the_pool_holds_several_handles_per_key():
    entries = EntryCache()
    handles = [FakeHandle(10) for _ in range(3)]
    for handle in handles:
        entries.give_plan(("k",), TOPOLOGY, handle, budget=1000)
    assert entries.pooled == 3
    taken = {id(entries.take_plan(("k",))) for _ in range(3)}
    assert taken == {id(h) for h in handles}


def test_a_tight_budget_frees_pooled_handles_before_bases_and_entries():
    entries = EntryCache()
    old, new = FakeHandle(40), FakeHandle(40)
    entries.give_plan(("a",), TOPOLOGY, old, budget=1000)
    entries.give_plan(("b",), TOPOLOGY, new, budget=1000)
    base = FakeHandle(50)
    key = entries.put(Entry(physics="em", handle=FakeKept(20, base)), budget=1000)
    # 40 + 40 + 70: the least recently returned handle goes first
    entries.put(Entry(physics="em", handle=FakeKept(10, None)), budget=120)
    assert old.freed
    assert not new.freed
    assert entries.pooled == 1
    # then the rest of the pool, then the oldest entry's base, the entry kept
    entries.put(Entry(physics="em", handle=FakeKept(10, None)), budget=40)
    assert new.freed
    assert base.freed
    assert entries.get(key) is not None
    # then whole entries
    entries.put(Entry(physics="em", handle=FakeKept(10, None)), budget=20)
    assert entries.get(key) is None


def test_an_entry_too_big_with_its_base_is_kept_without_it():
    entries = EntryCache()
    base = FakeHandle(100)
    entry = Entry(physics="em", handle=FakeKept(20, base))
    key = entries.put(entry, budget=50)
    assert base.freed
    assert entries.get(key) is entry


def test_a_zero_budget_pools_nothing():
    entries = EntryCache()
    handle = FakeHandle(10)
    entries.give_plan(("k",), TOPOLOGY, handle, budget=0)
    assert handle.freed
    assert entries.pooled == 0


def test_a_consumed_entry_returns_its_base_to_the_pool():
    entries = EntryCache()
    returned = []

    class Returning(FakeKept):
        def free(self, *, return_plans=False):
            returned.append(return_plans)

    entry = Entry(physics="em", handle=Returning(10, None))
    entries.put(entry, budget=100)
    entry.consumed = True
    entries.free_consumed(comm=None, budget=100)
    assert returned == [True]


def test_freeing_consumed_entries_survives_their_bases_returning_to_the_pool():
    # each consumed entry returns a base on the way out, over the budget
    entries = EntryCache()

    class Returning(FakeKept):
        def free(self, *, return_plans=False):
            if return_plans and self.base is not None:
                entries.give_plan(("k",), TOPOLOGY, self.base, budget=0)
            self.base = None

    first = Entry(physics="em", handle=Returning(10, FakeHandle(50)))
    second = Entry(physics="em", handle=Returning(10, FakeHandle(50)))
    entries.put(first, budget=1000)
    entries.put(second, budget=1000)
    first.consumed = second.consumed = True
    entries.free_consumed(comm=None, budget=0)
    assert len(entries) == 0
    assert first.handle is None
    assert second.handle is None


def test_a_walk_after_a_solve_leaves_other_solves_unconsumed_entries():
    # put kept the entry; nothing consumed, a returned plan: only pooled
    # handles go
    entries = EntryCache()
    base = FakeHandle(50)
    pending = Entry(physics="em", handle=FakeKept(20, base))
    key = entries.put(pending, budget=1000)
    pooled = FakeHandle(10)
    entries.give_plan(("k",), TOPOLOGY, pooled, budget=1000)
    entries.free_consumed(comm=None, budget=0)
    assert pooled.freed
    entries.give_plan(("k",), TOPOLOGY, FakeHandle(10), budget=0)
    assert entries.get(key) is pending
    assert not base.freed
    assert entries.pooled == 0


class FakeComm:
    """A communicator of one process for the cache's reductions."""


@pytest.fixture
def fake_comm(monkeypatch):
    monkeypatch.setattr(
        cache._mpi, "comm_key", lambda comm: None if comm is None else id(comm)
    )
    monkeypatch.setattr(cache._mpi, "node_bytes", lambda _comm, local: int(local))
    monkeypatch.setattr(cache._mpi, "node_budget", lambda _comm, local: int(local))
    return FakeComm()


def test_a_walk_under_a_communicator_does_not_count_pooled_handles(fake_comm):
    entries = EntryCache()
    pooled = FakeHandle(500)
    entries.give_plan(("k",), TOPOLOGY, pooled, budget=1000)
    first = Entry(physics="em", handle=FakeKept(20, FakeHandle(30)))
    k_first = entries.put(first, budget=200, comm=fake_comm)
    k_second = entries.put(
        Entry(physics="em", handle=FakeKept(20, FakeHandle(30))),
        budget=200,
        comm=fake_comm,
    )
    # the pool is serial: this walk cannot free it, and does not count it
    assert not pooled.freed
    assert entries.get(k_first) is first
    assert first.handle.base is not None
    assert entries.get(k_second) is not None


def test_clear_frees_the_pool_too():
    entries = EntryCache()
    handle = FakeHandle(10)
    entries.give_plan(("k",), TOPOLOGY, handle, budget=100)
    entries.clear()
    assert handle.freed
    assert entries.pooled == 0


# --- driven loops --------------------------------------------------------------


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    out = tmp_path_factory.mktemp("line") / "line.msh"
    return sm.write_msh22(out, sm.strip_line(3.0, 2.0, 1.0, (6, 2, 1), core=(1.0, 2.0)))


def _model(mesh, eps):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=2.0),
            "core": gfem.MaterialField(eps_r=eps),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture
def plans(monkeypatch):
    """Counts the direct handles' analyses (a new plan, PARDISO phase 11)."""
    calls = []
    factor = DirectHandle._factor

    def counted(self, *args, **kwargs):
        calls.append(1)
        return factor(self, *args, **kwargs)

    monkeypatch.setattr(DirectHandle, "_factor", counted)
    cache.ENTRIES.clear()
    yield calls
    cache.ENTRIES.clear()


def _objective(mesh, retain_memory=None):
    options = gfem.SolverOptions(linear_solver="direct", retain_memory=retain_memory)

    def objective(eps):
        s = gfem.solve(_model(mesh, eps), gfem.Driven(frequency=F), options=options).s
        # every port excited and no Bloch vector: no adjoint solve
        return jnp.sum(jnp.abs(s.values) ** 2 * WEIGHTS)

    return objective


def _loop(objective, eps, steps=4):
    step = jax.value_and_grad(objective)
    out = []
    for _ in range(steps):
        value, grad = step(eps)
        out.append((float(value), np.asarray(grad)))
        eps = eps + 0.5 * grad / jnp.max(jnp.abs(grad))
    return out


def _assert_same(got, want):
    for (v, g), (v0, g0) in zip(got, want, strict=True):
        assert v == pytest.approx(v0, rel=1e-12, abs=0)
        np.testing.assert_allclose(g, g0, rtol=0, atol=1e-12 * np.abs(g0).max())


def _eps(mesh):
    return jnp.full(len(mesh.elements("core")), 3.0)


def test_a_density_loop_plans_once(path, plans):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    got = _loop(_objective(mesh), _eps(mesh))
    assert len(plans) == 1
    plans.clear()
    want = _loop(_objective(mesh, retain_memory=0), _eps(mesh))
    # retain_memory=0 keeps no plan and no entry: each step plans for its
    # forward and again for the forward its backward runs, as before the pool
    assert len(plans) == 8
    _assert_same(got, want)


def test_a_node_move_loop_plans_once(path, plans):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    eps = _eps(mesh)
    nodes = np.asarray(mesh.nodes)

    def moved(k):
        shift = np.zeros_like(nodes)
        shift[:, 0] = 0.01 * k * np.sin(nodes[:, 1])
        return type(mesh)(jnp.asarray(nodes + shift), mesh._topology)

    def run(retain_memory=None):
        out = []
        for k in range(4):
            value, grad = jax.value_and_grad(_objective(moved(k), retain_memory))(eps)
            out.append((float(value), np.asarray(grad)))
        return out

    got = run()
    assert len(plans) == 1
    plans.clear()
    want = run(retain_memory=0)
    assert len(plans) == 8
    _assert_same(got, want)


def test_non_differentiated_solves_share_one_plan(path, plans):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    for eps in (3.0, 3.5, 4.0):
        gfem.solve(
            _model(mesh, jnp.full(len(mesh.elements("core")), eps)),
            gfem.Driven(frequency=F),
            options=gfem.SolverOptions(linear_solver="direct"),
        )
    assert len(plans) == 1
    assert cache.ENTRIES.pooled == 1


def test_vmapped_variants_hold_one_pooled_handle_each(path, plans):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    objective = _objective(mesh)
    robust = jax.value_and_grad(
        lambda eps: jnp.max(
            jax.vmap(objective)(eps[None] * jnp.array([[0.9], [1.0], [1.1]]))
        )
    )
    eps = _eps(mesh)
    for _ in range(3):
        _, grad = robust(eps)
        eps = eps + 0.5 * grad / jnp.max(jnp.abs(grad))
    assert len(plans) == 3
    cache.ENTRIES.free_consumed(comm=None, budget=2**40)
    assert cache.ENTRIES.pooled == 3


@pytest.mark.usefixtures("plans")  # an empty cache before and after
def test_pooled_handles_of_a_topology_left_out_of_the_discretizations_are_freed(path):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    gfem.solve(
        _model(mesh, _eps(mesh)),
        gfem.Driven(frequency=F),
        options=gfem.SolverOptions(linear_solver="direct"),
    )
    assert cache.ENTRIES.pooled == 1
    for _ in range(cache._MAX_DISCRETIZATIONS):
        cache.discretization(gfem.Mesh.from_gmsh(path, length_unit="mm"))
    assert cache.ENTRIES.pooled == 0


@pytest.mark.usefixtures("plans")  # an empty cache before and after
def test_a_bloch_forward_returns_its_base_to_the_pool(tmp_path):
    # the adjoint solves are on the transposed system: they never read the base
    mesh = pc.g_mesh(tmp_path)
    base = pc.g(mesh)
    n = len(mesh.elements("cover"))
    problem = gfem.Driven(pc.G_FREQUENCIES, wave_vector=pc.G_K)
    options = gfem.SolverOptions(linear_solver="direct")

    def objective(eps):
        model = base.with_material("cover", gfem.MaterialField(eps_r=eps))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    eps = jnp.full(n, 2.0)
    _, vjp = jax.vjp(objective, eps)
    (entry,) = cache.ENTRIES._entries.values()
    assert entry.handle.system._adjoint_sweep is None
    assert cache.ENTRIES.pooled == 1
    (got,) = vjp(1.0)
    cache.ENTRIES.clear()
    want = jax.grad(objective)(eps)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)
