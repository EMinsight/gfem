"""Engine caches, the pinned quadrature rule, and kept forward solves."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache, quadrature
from gfem._engine.cache import Entry, EntryCache, discretization
from gfem._engine.electrostatic import ElectrostaticRule
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, report
from gfem._linalg.direct import DirectHandle
from gfem._linalg.pieces import KeptParts


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=2.0)}
    )


# --- quadrature --------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pinned_rule_integrates_degree_2p_exactly(model, order):
    mesh = discretization(model.mesh).built.mesh  # unit cube in file units
    f = ngs.x ** (2 * order)
    vol = ngs.Integrate(f * quadrature.dx(order), mesh)
    top = ngs.Integrate(
        f * quadrature.ds(order, definedon=mesh.Boundaries("top")), mesh
    )
    assert vol == pytest.approx(1 / (2 * order + 1), rel=1e-13)
    assert top == pytest.approx(1 / (2 * order + 1), rel=1e-13)


def test_pinned_rule_is_the_one_used_not_a_default(model):
    # degree 2 cannot integrate x^4 exactly on two cells: the rule is pinned
    mesh = discretization(model.mesh).built.mesh
    value = ngs.Integrate(ngs.x**4 * quadrature.dx(1), mesh)
    assert abs(value - 1 / 5) > 1e-6


# --- discretization cache ----------------------------------------------------------


def test_discretization_is_shared_by_one_mesh_and_rebuilt_for_moved_nodes(model):
    first = discretization(model.mesh)
    assert discretization(model.mesh) is first
    moved = model.mesh.nodes.at[:, 2].multiply(2.0)
    other = discretization(type(model.mesh)(moved, model.mesh._topology))
    assert other is not first
    assert other.built.mesh.nv == first.built.mesh.nv


def test_spaces_are_cached_by_key(model):
    disc = discretization(model.mesh)
    calls = []

    def build():
        calls.append(1)
        return object()

    a = disc.space(("em", (), "h1", 7), build)
    assert disc.space(("em", (), "h1", 7), build) is a
    assert len(calls) == 1


# --- solve entries -----------------------------------------------------------------


class FakeHandle:
    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.freed = False

    def free(self):
        self.freed = True


def _entry(nbytes):
    return Entry(physics="em", handle=FakeHandle(nbytes))


def test_entries_are_evicted_least_recently_used_first_over_budget():
    entries = EntryCache()
    a, b, c = _entry(40), _entry(40), _entry(40)
    ka = entries.put(a, budget=100)
    kb = entries.put(b, budget=100)
    assert entries.get(ka) is a  # a is now the most recent
    kc = entries.put(c, budget=100)
    assert entries.get(kb) is None
    assert b.handle is None  # freed on eviction
    assert entries.get(ka) is a
    assert entries.get(kc) is c
    assert entries.nbytes == 80


def test_reading_an_entry_never_frees_it():
    entries = EntryCache()
    e = _entry(10)
    key = entries.put(e, budget=100)
    for _ in range(3):
        assert entries.get(key) is e
    assert not e.handle.freed


def test_zero_budget_keeps_nothing():
    entries = EntryCache()
    handle = FakeHandle(1)
    key = entries.put(Entry(physics="em", handle=handle), budget=0)
    assert entries.get(key) is None
    assert handle.freed
    assert len(entries) == 0


def test_an_entry_over_its_budget_is_not_kept_and_evicts_nothing():
    entries = EntryCache()
    kept = _entry(40)
    key = entries.put(kept, budget=100)
    big = _entry(60)
    assert entries.get(entries.put(big, budget=50)) is None
    assert big.handle is None
    entries.put(_entry(1), budget=0)
    assert entries.get(key) is kept  # other solves' entries stay
    assert not kept.handle.freed


def test_entry_bytes_count_fields_and_handle():
    e = Entry(physics="em", handle=FakeHandle(100), fields={"u": np.zeros(10)})
    assert e.nbytes == 180
    e.free()
    assert e.nbytes == 0


def test_default_budget_is_half_the_machine():
    assert cache.default_retain_memory() > 2**28


GIB = 2**30


@pytest.fixture(autouse=True)
def fresh_default_budget():
    """The default budget read anew: it is read once per process."""
    cache.memory_limit.cache_clear()
    cache.default_retain_memory.cache_clear()
    yield
    cache.memory_limit.cache_clear()
    cache.default_retain_memory.cache_clear()


def test_the_default_budget_is_read_once_per_process(monkeypatch):
    reads = []

    def physical_memory():
        reads.append(1)
        return 64 * GIB

    monkeypatch.setattr(cache, "_physical_memory", physical_memory)
    assert cache.default_retain_memory() == cache.default_retain_memory()
    assert cache.retain_budget(None) == cache.default_retain_memory()
    assert len(reads) == 1


def _fake_cgroups(tmp_path, monkeypatch, membership, limits):
    """A ``/proc/self/cgroup`` reading ``membership`` and limit files under a root."""
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(membership)
    root = tmp_path / "sys_fs_cgroup"
    root.mkdir()
    for relative, (name, value) in limits.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(value)
    monkeypatch.setattr(cache, "_PROC_CGROUP", proc)
    monkeypatch.setattr(cache, "_CGROUP_ROOT", root)
    monkeypatch.setattr(cache, "_physical_memory", lambda: 64 * GIB)


def test_default_budget_follows_a_v2_limit_set_on_a_parent_cgroup(
    tmp_path, monkeypatch
):
    _fake_cgroups(
        tmp_path,
        monkeypatch,
        "0::/slurm/job_1/step_0/task_0\n",
        {
            "slurm/job_1/step_0/task_0": ("memory.max", "max\n"),
            "slurm/job_1/step_0": ("memory.max", "max\n"),
            "slurm/job_1": ("memory.max", f"{16 * GIB}\n"),
            "slurm": ("memory.max", f"{32 * GIB}\n"),
        },
    )
    assert cache.default_retain_memory() == 8 * GIB


def test_default_budget_reads_a_v1_memory_controller(tmp_path, monkeypatch):
    _fake_cgroups(
        tmp_path,
        monkeypatch,
        "4:cpu,cpuacct:/job\n3:memory:/job/step\n0::/\n",
        {
            "memory/job/step": ("memory.limit_in_bytes", "9223372036854771712\n"),
            "memory/job": ("memory.limit_in_bytes", f"{12 * GIB}\n"),
        },
    )
    assert cache.default_retain_memory() == 6 * GIB


def test_default_budget_is_half_the_ram_without_a_limit(tmp_path, monkeypatch):
    _fake_cgroups(
        tmp_path,
        monkeypatch,
        "0::/user.slice\n",
        {
            "user.slice": ("memory.max", "max\n"),
            ".": ("memory.max", "garbage\n"),
        },
    )
    assert cache.default_retain_memory() == 32 * GIB
    monkeypatch.setattr(cache, "_PROC_CGROUP", tmp_path / "missing")
    cache.default_retain_memory.cache_clear()
    assert cache.default_retain_memory() == 32 * GIB


def test_default_budget_without_readable_ram_or_limit_is_8_gib(tmp_path, monkeypatch):
    _fake_cgroups(tmp_path, monkeypatch, "0::/\n", {})
    monkeypatch.setattr(cache, "_physical_memory", lambda: None)
    assert cache.default_retain_memory() == 8 * GIB


def test_a_limit_below_the_ram_wins_without_readable_ram(tmp_path, monkeypatch):
    _fake_cgroups(tmp_path, monkeypatch, "0::/\n", {".": ("memory.max", f"{GIB}")})
    monkeypatch.setattr(cache, "_physical_memory", lambda: None)
    assert cache.default_retain_memory() == GIB // 2


# --- the default budget's exemption -----------------------------------------------


class Base:
    """A pooled direct base: releasing its factors leaves ``released_bytes``.

    It may share ``pattern`` with a kept fixed part.
    """

    def __init__(self, nbytes, *, released_bytes=10, pattern=None):
        self._nbytes, self._released_bytes = nbytes, released_bytes
        self.pattern = pattern
        self.released = self.freed = False

    @property
    def nbytes(self):
        return self._released_bytes if self.released else self._nbytes

    @property
    def factor_bytes(self):
        return 0 if self.released else self._nbytes - self._released_bytes

    def shared(self):
        pattern = self.pattern
        return {} if pattern is None else {id(pattern): pattern.nbytes}

    def release_factors(self):
        self.released = True

    def free(self):
        self.freed = True


TOPOLOGY = object()


@pytest.fixture
def machine(monkeypatch):
    """A process limited to 1000 bytes (default budget 500); set its peak RSS."""
    peak = [0]
    monkeypatch.setattr(cache, "_physical_memory", lambda: 1000)
    monkeypatch.setattr(cache, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(cache, "peak_rss", lambda: peak[0])
    return peak


@pytest.mark.usefixtures("machine")
def test_the_default_budget_is_half_the_memory_limit():
    assert cache.memory_limit() == 1000
    assert cache.retain_budget(None) == 500


@pytest.mark.parametrize(("peak", "kept"), [(900, True), (901, False)])
def test_the_latest_base_keeps_its_factors_over_the_default_while_there_is_room(
    machine, peak, kept
):
    machine[0] = peak  # kept while peak + 1000/10 <= 1000
    entries = EntryCache()
    base = Base(600)
    entries.give_plan(("k",), TOPOLOGY, base, budget=cache.retain_budget(None))
    assert base.released is not kept
    # later walks under the default leave it as well
    entries.trim(cache.retain_budget(None))
    entries.free_consumed(None, cache.retain_budget(None))
    assert base.released is not kept
    assert not base.freed


@pytest.mark.usefixtures("machine")
def test_an_explicit_retain_memory_always_releases():
    entries = EntryCache()
    base = Base(600)
    entries.give_plan(("k",), TOPOLOGY, base, budget=cache.retain_budget(500))
    assert base.released


@pytest.mark.usefixtures("machine")
def test_only_the_latest_returned_base_is_exempt():
    entries = EntryCache()
    old, new = Base(600), Base(600)
    budget = cache.retain_budget(None)
    entries.give_plan(("a",), TOPOLOGY, old, budget=budget)
    entries.give_plan(("b",), TOPOLOGY, new, budget=budget)
    assert old.released
    assert not new.released
    # a handle that frees nothing on release is freed; the latest stays whole
    stuck = Base(600, released_bytes=600)
    entries.give_plan(("c",), TOPOLOGY, stuck, budget=budget, walk=False)
    entries.give_plan(("b",), TOPOLOGY, entries.take_plan(("b",)), budget=budget)
    assert stuck.freed
    assert not new.released
    assert not new.freed


@pytest.mark.usefixtures("machine")
def test_an_exempt_base_still_counts_its_plan_and_matrix():
    entries = EntryCache()
    budget = cache.retain_budget(None)
    # nothing to release: only freeing it trims it
    old = Base(350, released_bytes=350)
    entries.give_plan(("a",), TOPOLOGY, old, budget=budget)
    assert not old.freed
    # the latest keeps its 400 bytes of factors; its other 200 count, and
    # 350 + 200 exceeds the 500 of the default
    latest = Base(600, released_bytes=200)
    entries.give_plan(("b",), TOPOLOGY, latest, budget=budget)
    assert not latest.released
    assert old.freed


@pytest.mark.usefixtures("machine")
def test_an_exempt_base_keeps_the_kept_part_it_shares_counted():
    pattern = np.zeros(300, np.uint8)
    store = KeptParts()
    store.kept["piece"] = SimpleNamespace(shared=lambda: {id(pattern): pattern.nbytes})
    disc = _held(ports_bytes=400)
    disc.spaces["em"] = store
    entries = EntryCache(lambda: [disc])
    base = Base(600, released_bytes=10, pattern=pattern)
    # 10 + 300 (shared once) + 400 > 500: dropping the kept part frees
    # nothing while the exempt base holds its pattern, so the port state goes
    budget = cache.retain_budget(None)
    entries.give_plan(("k",), TOPOLOGY, base, budget, kept=[pattern])
    assert not base.released
    assert store.kept
    assert disc.ports is None


# --- design initial guesses ---------------------------------------------------------


class Ports:
    """Kept port state of ``nbytes``."""

    def __init__(self, nbytes):
        self.array = np.zeros(nbytes, np.uint8)

    def shared(self):
        return {id(self.array): self.array.nbytes}


def _held(ports_bytes=0):
    built = SimpleNamespace(comm=None)
    disc = cache.Discretization(built=built, topology=TOPOLOGY)
    if ports_bytes:
        disc.ports = Ports(ports_bytes)
    return disc


def test_a_discretization_keeps_a_design_guess_only_for_its_match():
    disc = _held()
    block = np.ones((4, 2), complex)
    disc.keep_guess(("cols", 1.0), "free digest", block)
    assert disc.guess(("cols", 1.0), "free digest") is block
    assert disc.guess(("cols", 1.0), "other free DOFs") is None
    assert disc.guess(("cols", 2.0), "free digest") is None
    assert disc.shared() == {id(block): block.nbytes}


def test_the_walk_drops_design_guesses_before_releasing_pooled_factors():
    disc = _held(ports_bytes=100)
    disc.keep_guess("k", "free", np.zeros(400, np.uint8))
    entries = EntryCache(lambda: [disc])
    base = Base(600)
    # 1100 bytes over 800: the guess alone goes
    entries.give_plan(("k",), TOPOLOGY, base, budget=800)
    assert disc.guess("k", "free") is None
    assert not base.released
    assert disc.ports is not None


def test_a_zero_budget_keeps_no_design_guess():
    disc = _held()
    disc.keep_guess("k", "free", np.zeros(10, np.uint8))
    EntryCache(lambda: [disc]).trim(0)
    assert disc.guess("k", "free") is None


@pytest.mark.usefixtures("machine")
def test_an_entry_over_the_default_gives_its_base_to_the_pool_with_factors():
    entries = EntryCache()
    budget = cache.retain_budget(None)

    class Kept:
        def __init__(self, base):
            self.base = base

        @property
        def nbytes(self):
            return 20 + (0 if self.base is None else self.base.nbytes)

        def free_base(self, comm, *, walk=True):  # noqa: ARG002
            entries.give_plan(("k",), TOPOLOGY, self.base, budget, walk=walk)
            self.base = None

        def free(self, *, return_plans=False):  # noqa: ARG002
            self.base = None

    base = Base(600)
    entry = Entry(physics="em", handle=Kept(base))
    key = entries.put(entry, budget)
    assert entries.get(key) is entry
    assert entries.take_plan(("k",)) is base
    assert not base.released


@pytest.fixture(scope="module")
def helmholtz(tmp_path_factory):
    """A complex symmetric ``-Δ - k2`` on a small box, and loads."""
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (3, 3, 3)))
    mesh = to_ngsolve(gfem.Mesh.from_gmsh(path))
    space = ngs.H1(mesh.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()

    def at(k2):
        form = ngs.BilinearForm(space, symmetric=True)
        form += (ngs.grad(u) * ngs.grad(v) - k2 * u * v) * quadrature.dx(2)
        return Operator(form, space.FreeDofs())

    rng = np.random.default_rng(0)
    free = np.asarray([space.FreeDofs()[i] for i in range(space.ndof)])
    B = (rng.standard_normal((space.ndof, 2)) + 0j) * free[:, None]
    return at, B


@pytest.mark.parametrize("room", [True, False])
def test_the_next_step_names_a_release_for_retain_memory(
    machine, monkeypatch, caplog, helmholtz, room
):
    peak, action = (0, "reused") if room else (2**62, "refactorized")
    at, B = helmholtz
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
    handle = DirectHandle.from_operator(at(2.0 - 1j), "complex_symmetric")
    handle.refactorize(at(1.0 - 1j), frequency=1.0)
    handle.solve(B)
    base_bytes = handle.nbytes
    # the factors alone are over the default budget
    machine[0] = peak
    monkeypatch.setattr(cache, "_physical_memory", lambda: 3 * base_bytes // 2)
    entries = EntryCache()
    budget = cache.retain_budget(None)
    entries.give_plan(("k",), TOPOLOGY, handle, budget=budget)
    taken = entries.take_plan(("k",))
    taken.refactorize(at(1.0 - 1j + 1e-3), frequency=1.0)
    with caplog.at_level("INFO", logger="gfem.solver"):
        report.solve_step(taken, problem="driven", direction="forward", iterations=())
    (record,) = [r for r in caplog.records if r.name == "gfem.solver"]
    assert record.action == action
    if action == "refactorized":
        assert record.reason == "factors released (retain_memory)"
        assert (record.base_bytes, record.budget_bytes) == (base_bytes, budget)
    handle.free()


# --- consumed entries --------------------------------------------------------------


def test_only_consumed_entries_are_freed_when_a_differentiated_forward_starts():
    entries = EntryCache()
    read, unread = _entry(10), _entry(10)
    k_read = entries.put(read, budget=100)
    k_unread = entries.put(unread, budget=100)
    read.consumed = True
    entries.free_consumed(comm=None)
    assert entries.get(k_read) is None
    assert read.handle is None
    assert entries.get(k_unread) is unread
    assert not unread.handle.freed


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (2, 2, 2)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"oxide": gfem.Material(), "nitride": gfem.Material()},
    )


def _capacitance(stack):
    problem = gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    )
    n_oxide = len(stack.mesh.elements("oxide"))

    def capacitance(eps):
        model = stack.replace(
            materials={
                "oxide": gfem.MaterialField(eps_r=eps[:n_oxide]),
                "nitride": gfem.MaterialField(eps_r=eps[n_oxide:]),
            }
        )
        return gfem.solve(model, problem).capacitance

    return capacitance


@pytest.fixture
def forwards(monkeypatch):
    """The ``keep`` flag of every electrostatic forward, in order."""
    seen = []
    forward = ElectrostaticRule.forward

    def counted(self, model, *, keep):
        seen.append(keep)
        return forward(self, model, keep=keep)

    monkeypatch.setattr(ElectrostaticRule, "forward", counted)
    cache.ENTRIES.clear()
    yield seen
    cache.ENTRIES.clear()


WEIGHTS = jnp.asarray([[1.0, 0.3], [-0.7, 2.0]])


def _design(stack, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(2.0, 9.0, len(stack.mesh.tetrahedra)))


def test_an_optimization_loop_keeps_only_the_current_steps_entry(stack, forwards):
    capacitance = _capacitance(stack)
    step = jax.jit(jax.value_and_grad(lambda e: jnp.sum(WEIGHTS * capacitance(e))))
    eps = _design(stack)
    for _ in range(5):
        _, grad = step(eps)
        eps = eps - 1e-3 * grad
    assert forwards == [True] * 5
    assert len(cache.ENTRIES) == 1


def test_jacrev_and_repeated_pullbacks_run_no_extra_forward(stack, forwards):
    capacitance = _capacitance(stack)
    eps = _design(stack)
    jacobian = jax.jacrev(capacitance)(eps)
    assert forwards == [True]
    _, pullback = jax.vjp(capacitance, eps)
    grads = [pullback(k * WEIGHTS)[0] for k in (1.0, 2.0, 3.0)]
    assert forwards == [True, True]
    np.testing.assert_allclose(
        grads[0], jnp.einsum("ij,ijk->k", WEIGHTS, jacobian), rtol=1e-12
    )
    np.testing.assert_allclose(grads[2], 3 * grads[0], rtol=1e-12)


def test_a_pullback_after_a_later_forward_runs_its_own_again(stack, forwards):
    capacitance = _capacitance(stack)
    eps = _design(stack)
    _, pullback = jax.vjp(capacitance, eps)
    first = pullback(WEIGHTS)[0]
    jax.grad(lambda e: jnp.sum(WEIGHTS * capacitance(e)))(eps + 1.0)
    assert forwards == [True, True]
    again = pullback(WEIGHTS)[0]
    assert forwards == [True, True, True]
    np.testing.assert_allclose(again, first, rtol=1e-12)


# --- a kept forward solve --------------------------------------------------------


def test_kept_electrostatic_solve_holds_the_potentials_and_its_operator(model):
    # the operator serves an energy or potential gradient; the
    # cache drops it before the potentials
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions(order=2))
    outputs, entry = rule.forward(model, keep=True)
    assert entry is not None
    assert entry.physics == "em"
    assert entry.handle.nbytes > 0
    phi = entry.fields["potential"]
    assert phi.shape[1] == 1
    assert entry.nbytes == phi.nbytes + entry.handle.nbytes
    entry.handle.free_base(None)
    assert entry.nbytes == phi.nbytes
    # 1 V on the top plate, 0 V on the bottom one
    assert phi[:, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert phi[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert outputs["capacitance"].shape == (1, 1)


def test_unkept_solve_returns_no_entry(model):
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions())
    _, entry = rule.forward(model, keep=False)
    assert entry is None
    assert rule.reads == {"eps_r"}
