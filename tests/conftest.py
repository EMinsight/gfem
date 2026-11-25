"""Mark every test with its tier, taken from its directory under tests/.

tests/unit -> unit, tests/verification -> verification, tests/reference ->
reference, tests/benchmark -> benchmark, tests/cuda -> cuda, tests/mpi -> mpi. A
test file outside these directories is a collection error, so each test has one tier.

The reference data (Palace runs, imported meshes) is found through the
``GFEM_REFERENCE_DIR`` environment variable: the `reference_dir` fixture
gives its path and skips the test when it is unset, and a test taking a
``palace_case`` argument runs once per case directory holding a
``palace.msh``.

Tests marked ``petsc`` or ``slepc`` need gfem's PETSc or SLEPc solver: they
are skipped, with the user-facing reason, where `gfem._petsc` cannot use it.
Tests in tests/mpi are skipped where mpi4py or the launcher is missing, and one
fails if it starts PETSc in pytest's process.

A test that sets NGSolve's thread count, or enters its own `ngs.TaskManager()`, uses
the `ngsolve_threads` fixture, which then sets the count `threads=None` would use.

A test marked ``bitwise`` compares direct solves to the last bit. PARDISO repeats
its bits on any thread count; NGSolve's sparse Cholesky, the fallback without the
mkl extra, only on one thread, so without PARDISO the test runs on one.

``--shard i/n`` keeps the i-th (0-based) of n shards of the selected tests, so n CI
jobs share a tier. Tests are dealt longest first to the least-loaded shard, by the
seconds recorded in tests/durations.json (see tests/shards.py); a test the file
does not know costs the median of the positive recorded times.

A test whose recorded duration is `SLOW_SECONDS` or more, or that carries a hand
``@pytest.mark.slow``, is marked ``slow``, so ``-m "not slow"`` leaves it out.
"""

import os
import sys
from pathlib import Path

import pytest
import shards

TIERS = ("unit", "verification", "reference", "benchmark", "cuda", "mpi")
TESTS_DIR = Path(__file__).parent
REFERENCE_ENV = "GFEM_REFERENCE_DIR"
DURATIONS_FILE = TESTS_DIR / "durations.json"
SLOW_SECONDS = 30.0


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        default=None,
        metavar="I/N",
        help="run shard I (0-based) of N, balanced by tests/durations.json",
    )


class _Shard:
    """Keeps shard ``index`` of ``count`` once `-m` and `-k` have deselected."""

    def __init__(self, index: int, count: int) -> None:
        self.index, self.count = index, count

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        costs = shards.costs_of(
            [item.nodeid for item in items], shards.load(DURATIONS_FILE)
        )
        owner = shards.assign(costs, self.count)
        kept = [item for item in items if owner[item.nodeid] == self.index]
        dropped = [item for item in items if owner[item.nodeid] != self.index]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def pytest_configure(config: pytest.Config) -> None:
    shard = config.getoption("--shard")
    if shard is None:
        return
    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard takes I/N, got {shard!r}") from None
    if not 0 <= index < count:
        raise pytest.UsageError(f"--shard {shard}: I must be in 0..N-1")
    config.pluginmanager.register(_Shard(index, count), "gfem-shard")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the tier and `slow` markers before pytest's `-m` deselection runs."""
    recorded = shards.load(DURATIONS_FILE)
    for item in items:
        if recorded.get(shards.junit_key(item.nodeid), 0.0) >= SLOW_SECONDS:
            item.add_marker("slow")
        tier = item.path.relative_to(TESTS_DIR).parts[0]
        if tier not in TIERS:
            raise pytest.UsageError(
                f"{item.nodeid}: tests must live in tests/<tier>/ with tier in {TIERS}"
            )
        item.add_marker(tier)
        if tier == "mpi" and _uses_petsc_marker(item):
            raise pytest.UsageError(
                f"{item.nodeid}: a launched test skips on _petsc.has_mpi(): a petsc or "
                "slepc marker would start PETSc, and on the MPI stack MPI, in this "
                "process, which launched jobs then inherit"
            )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Skip what cannot run here, among the tests `-m` left.

    Probing a deselected test would start PETSc in this process for nothing, and on
    the MPI PETSc stack load MPI here, which launched jobs inherit.
    """
    _skip_without_petsc(session.items)
    _skip_without_mpi(session.items)


def _uses_petsc_marker(item: pytest.Item) -> bool:
    return any(item.get_closest_marker(name) for name in ("petsc", "slepc"))


def _skip_without_petsc(items: list[pytest.Item]) -> None:
    """Skip `petsc` / `slepc` tests with the `gfem.SolverUnavailableError` message.

    The probe is gfem's own, which initialises PETSc without pytest's command
    line, and it runs only when some test carries either marker.
    """
    marked = [
        (item, name)
        for item in items
        for name in ("petsc", "slepc")
        if item.get_closest_marker(name) is not None
    ]
    if not marked:
        return
    from gfem import _petsc
    from gfem.errors import SolverUnavailableError

    probes = {"petsc": _petsc.require_petsc, "slepc": _petsc.require_slepc}
    for item, name in marked:
        try:
            probes[name]()
        except SolverUnavailableError as err:
            item.add_marker(pytest.mark.skip(reason=str(err)))


def _skip_without_mpi(items: list[pytest.Item]) -> None:
    """Skip the launched tests where they cannot launch.

    The probe runs only when some test carries the marker, and in `ranks`, so
    this one-process pytest never loads MPI.
    """
    marked = [item for item in items if item.get_closest_marker("mpi") is not None]
    if not marked:
        return
    import ranks

    reason = ranks.unavailable()
    if reason is not None:
        for item in marked:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _log_display_off():
    """Turn the log display off after each test, so no test inherits one."""
    yield
    log = sys.modules.get("gfem.log")
    if log is not None:
        log.disable()


@pytest.fixture(autouse=True)
def _bitwise_threads(request, monkeypatch):
    """One solve thread for a ``bitwise`` test where PARDISO is missing."""
    if request.node.get_closest_marker("bitwise") is None:
        return
    from gfem._linalg.pardiso import pardiso_available

    if not pardiso_available():
        monkeypatch.setenv("NGS_NUM_THREADS", "1")


@pytest.fixture(autouse=True)
def _launched_tests_leave_petsc_unstarted(request):
    """Fail a tests/mpi test that starts PETSc in pytest's process.

    On the MPI stack, starting PETSc makes this process an MPI singleton: later
    launches inherit its MPI state and abort in ``MPI_Init``, and its finalize can
    hang at exit. Importing ``petsc4py.PETSc`` initialises PETSc.
    """
    if request.node.get_closest_marker("mpi") is None:
        yield
        return
    started = "petsc4py.PETSc" in sys.modules
    yield
    if not started and "petsc4py.PETSc" in sys.modules:
        pytest.fail(
            "started PETSc in pytest's process: run what needs it on the ranks",
            pytrace=False,
        )


@pytest.fixture
def ngsolve_threads():
    """``SetNumThreads`` is process-wide: put the default count back after."""
    import ngsolve as ngs

    from gfem._engine.runtime import solve_threads

    threads = solve_threads(None)
    yield
    ngs.SetNumThreads(threads)


def _reference_root() -> Path | None:
    ref = os.environ.get(REFERENCE_ENV)
    return Path(ref) if ref else None


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """The reference data directory; skips the test when it is not set."""
    root = _reference_root()
    if root is None:
        pytest.skip(f"{REFERENCE_ENV} not set (point it at the reference data)")
    return root


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize ``palace_case`` over the reference cases with a mesh."""
    if "palace_case" not in metafunc.fixturenames:
        return
    root = _reference_root()
    if root is None:
        metafunc.parametrize(
            "palace_case",
            [
                pytest.param(
                    None,
                    id="no-reference-dir",
                    marks=pytest.mark.skip(reason=f"{REFERENCE_ENV} not set"),
                )
            ],
        )
        return
    cases = sorted(p.parent for p in root.glob("*/*/palace.msh"))
    metafunc.parametrize(
        "palace_case", cases, ids=[str(c.relative_to(root)) for c in cases]
    )
