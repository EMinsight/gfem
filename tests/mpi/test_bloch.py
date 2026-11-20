"""A driven Bloch solve on a model with one periodic pair across ranks (ADR 0040, 0071).

Each row compares ranks with the same function in pytest's one process, relative to
the largest one-process entry, at ADR 0070's gates; `on_ranks` checks that ranks
agree bit for bit. The wave vector lies along the pair, ``k = 0.3 pi / a``. The cases:

- cellx (`periodic_cases.cellx`): G with its x pair, a resistive sheet across the
  seam, ports away from it; S far from reciprocal.
- c1edge (`periodic_cases.c1edge`): C1 with p1 touching the pair's second surface,
  the seam-touching port of a swapped partition; S far from reciprocal.
- box8 and box12 (`periodic_cases.periodic_box`): both ports touch both seams, and
  the search takes clean, swapped and reordered partitions on this wheel; S is
  nearly reciprocal there, so these rows check the search paths.

A forced swap (`forced_search`) runs the swapped pairing, whose phase is the
inverse, on a mesh that needs none. The tests assert results, not paths.
"""

import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import mpi_cases
import numpy as np
import periodic_cases as pc
import pytest
from ranks import on_ranks

import gfem
from gfem import _petsc
from gfem._engine.cache import ENTRIES
from gfem._engine.mesh import forced_search
from gfem._linalg import coarse

#: Relative to the largest one-process entry: ADR 0070's gates.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-11}
#: A kept field at points, and a field objective, relative (7.1e-13 measured).
FIELD_GATE = 1e-10
#: The Bloch condition across the seam, relative to the largest field (3.4e-16).
SEAM_GATE = 1e-12
#: :math:`\max|S - S^T| / \max|S|` above this: a transposition error cannot pass.
NONRECIPROCAL = 1e-2
needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")
#: Points in units of the cell: two seam twins (x = 0 and x = a) and inside points.
POINTS = np.array(
    [
        [0.0, 0.31, 0.27],
        [1.0, 0.31, 0.27],
        [0.0, 0.73, 0.61],
        [1.0, 0.73, 0.61],
        [0.437, 0.523, 0.191],
        [0.771, 0.113, 0.836],
    ]
)
WEIGHTS = np.array([[1.0, 0.5], [2.0, -1.0]])

pytestmark = pytest.mark.filterwarnings(pc.G_OBLIQUE)


@functools.cache
def _mesh(case, directory):
    """One mesh per case and directory in a process: its partition memo is shared."""
    directory = Path(directory)
    if case.startswith("cellx"):
        path = directory / "g.msh"
        # pytest's one process writes it; ranks only read it
        if path.exists():
            return gfem.Mesh.from_gmsh(path, length_unit="mm")
        return pc.g_mesh(directory)
    if case == "c1edge":
        path = directory / "c1edge.msh"
        if path.exists():
            return gfem.Mesh.from_gmsh(path, length_unit="mm")
        return pc.c1edge_mesh(directory)
    return mpi_cases.box(int(case.removeprefix("box")))


def _setup(case, directory):
    """``(build(eps, sigma), eps, sigma, problem(**kw), size)``.

    ``problem(**kw)`` is the case's Bloch `gfem.Driven` over its frequencies (one
    with ``first``), at a fixed angle for cellxangle.
    """
    mesh = _mesh(case, str(directory))
    rng = np.random.default_rng(5)
    if case.startswith("cellx"):
        n = len(mesh.elements("core")) + len(mesh.elements("sub"))
        scale, frequencies = 1.0, list(pc.G_FREQUENCIES)
        size = np.array([pc.G_A, pc.G_B, pc.G_H]) * pc.MM

        def build(eps, sigma):
            return pc.cellx(mesh, eps_r=eps, conductivity=sigma)

    elif case == "c1edge":
        n = len(mesh.elements("cell"))
        scale, frequencies = 0.05, [pc.C1_FREQUENCY, 1.1 * pc.C1_FREQUENCY]
        size = np.array([pc.C1_A, pc.C1_W, pc.C1_H]) * pc.MM

        def build(eps, sigma):
            return pc.c1edge(mesh, eps_r=eps, conductivity=sigma)

    else:
        n = len(mesh.elements("diel"))
        scale, frequencies = 1e4, list(pc.BOX_FREQUENCIES)
        size = np.full(3, 1e-6)

        def build(eps, sigma):
            return pc.periodic_box(mesh, eps_r=eps, conductivity=sigma)

    eps = jnp.asarray(rng.uniform(2.0, 5.0, n))
    sigma = jnp.asarray(rng.uniform(0.05, 0.3, n) * scale)
    a = float(np.linalg.norm(build(eps, sigma).periodic_translations[0]))
    k = (0.3 * math.pi / a, 0.0, 0.0)
    angle = frequencies[0] if case == "cellxangle" else None

    def problem(*, first=False, **kw):
        sweep = frequencies[:1] if first else frequencies
        return gfem.Driven(sweep, wave_vector=k, wave_vector_frequency=angle, **kw)

    return build, eps, sigma, problem, size


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12, threads=1)


def _objective(s):
    return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
        jnp.real((0.3 - 0.7j) * s[:, 1, 0])
    )


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.abs(got - want).max() / np.abs(want).max())


def _swapped(function):
    """``function``, on a forced swapped partition when called with ``swap``."""

    @functools.wraps(function)
    def run(*args, swap=False):
        if not swap:
            return function(*args)
        with forced_search(swap=True):
            return function(*args)

    return run


# --- the functions each rank runs ------------------------------------------------


@_swapped
def _s(case, solver, directory):
    build, eps, sigma, problem, _ = _setup(case, directory)
    result = gfem.solve(build(eps, sigma), problem(), options=_options(solver))
    return np.asarray(result.s.values)


@_swapped
def _gradient(case, solver, directory):
    """The weighted |S|^2 objective and its gradient wrt eps_r and conductivity."""
    build, eps, sigma, problem, _ = _setup(case, directory)

    def objective(e, s):
        result = gfem.solve(build(e, s), problem(), options=_options(solver))
        return _objective(result.s.values)

    value, (g_eps, g_sigma) = jax.value_and_grad(objective, argnums=(0, 1))(eps, sigma)
    return float(value), np.asarray(g_eps), np.asarray(g_sigma)


@_swapped
def _fields(case, solver, directory):
    """p1's kept field at `POINTS`, the Bloch phase over a, and a field objective.

    The objective is |E|^2 at two seam twins and an inside point: its gradient
    wrt eps_r is a field cotangent's.
    """
    build, eps, sigma, problem, size = _setup(case, directory)
    kept = problem(first=True, excite=["p1"], keep_fields=True)
    result = gfem.solve(build(eps, sigma), kept, options=_options(solver))
    field = np.asarray(result.e_fields[0].at(POINTS * size))
    k, a = kept.wave_vector[0], size[0]
    points = jnp.asarray(POINTS[[0, 1, 4]] * size)

    def objective(e):
        result = gfem.solve(build(e, sigma), kept, options=_options(solver))
        return jnp.sum(jnp.abs(result.e_fields[0].at(points)) ** 2)

    value, grad = jax.value_and_grad(objective)(eps)
    return field, complex(np.exp(-1j * k * a)), float(value), np.asarray(grad)


def _bases_held(directory):
    """Factorizations a Bloch gradient holds: after its forward, after its backward."""
    build, eps, sigma, problem, _ = _setup("cellx", directory)
    ENTRIES.clear()

    def objective(e):
        result = gfem.solve(build(e, sigma), problem(), options=_options("direct"))
        return _objective(result.s.values)

    def held():
        forward = transposed = 0
        for entry in ENTRIES._entries.values():
            system = entry.handle.system
            sweep = system._adjoint_sweep
            forward += sweep is not None and sweep._base is not None
            other = system._transposed_system
            sweep = None if other is None else other._adjoint_sweep
            transposed += sweep is not None and sweep._base is not None
        return forward, transposed

    _, pullback = jax.vjp(objective, eps)
    after_forward = held()
    pullback(jnp.asarray(1.0))
    after_backward = held()
    entries = len(ENTRIES)
    ENTRIES.clear()
    return after_forward, after_backward, entries


def _plain_and_bloch(directory):
    """cellx's S plain (k = 0) and Bloch on bddc_gmres, and the coarse types made."""
    build, eps, sigma, problem, _ = _setup("cellx", directory)
    model, options = build(eps, sigma), _options("bddc_gmres")
    plain = gfem.solve(model, gfem.Driven(list(pc.G_FREQUENCIES)), options=options)
    bloch = gfem.solve(model, problem(), options=options)
    names = sorted({n.rsplit("_", 1)[-1] for n in coarse._registered})
    return np.asarray(plain.s.values), np.asarray(bloch.s.values), names


def _petsc_refusal(directory):
    """The one-process ``ValueError`` of ``petsc`` on a Bloch solve, then a sum."""
    from mpi4py import MPI

    build, eps, sigma, problem, _ = _setup("cellx", directory)
    try:
        gfem.solve(build(eps, sigma), problem(), options=_options("petsc"))
    except ValueError as err:
        return str(err), MPI.COMM_WORLD.allreduce(1)
    raise AssertionError("petsc ran a Bloch solve")


# --- the rows ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "solver", "ranks", "swap"),
    [
        pytest.param("cellx", "direct", 3, False, marks=needs_stack),
        pytest.param("cellx", "direct", 4, True, marks=needs_stack),
        ("cellx", "bddc_gmres", 3, False),
        pytest.param("c1edge", "direct", 4, True, marks=needs_stack),
        ("c1edge", "bddc_gmres", 4, True),
    ],
)
def test_a_nonreciprocal_s_matches_one_process(case, solver, ranks, swap, tmp_path):
    want = _s(case, solver, tmp_path)
    nonreciprocity = np.abs(want - np.swapaxes(want, 1, 2)).max() / np.abs(want).max()
    assert nonreciprocity > NONRECIPROCAL
    got = on_ranks(_s, case, solver, tmp_path, swap=swap, ranks=ranks, limit=600)[0]
    assert _rel(got, want) < GATES[solver]


@pytest.mark.parametrize(
    ("case", "solver", "ranks"),
    [
        pytest.param("box8", "direct", 5, marks=needs_stack),
        pytest.param("box8", "direct", 16, marks=needs_stack),
        ("box8", "bddc_gmres", 8),
        pytest.param("box12", "direct", 32, marks=needs_stack),
    ],
)
def test_s_on_the_search_paths_matches_one_process(case, solver, ranks, tmp_path):
    want = _s(case, solver, tmp_path)
    got = on_ranks(_s, case, solver, tmp_path, ranks=ranks, limit=900)[0]
    assert _rel(got, want) < GATES[solver]


@pytest.mark.parametrize(
    ("case", "solver", "ranks", "swap"),
    [
        pytest.param("cellx", "direct", 3, False, marks=needs_stack),
        pytest.param("c1edge", "direct", 4, True, marks=needs_stack),
        ("c1edge", "bddc_gmres", 4, True),
        pytest.param("box8", "direct", 8, False, marks=needs_stack),
        pytest.param("cellxangle", "direct", 3, False, marks=needs_stack),
    ],
)
def test_an_s_gradient_matches_one_process(case, solver, ranks, swap, tmp_path):
    value, g_eps, g_sigma = _gradient(case, solver, tmp_path)
    got = on_ranks(
        _gradient, case, solver, tmp_path, swap=swap, ranks=ranks, limit=900
    )[0]
    assert abs(got[0] - value) / abs(value) < GATES[solver]
    assert _rel(got[1], g_eps) < GATES[solver]
    assert _rel(got[2], g_sigma) < GATES[solver]


@needs_stack
def test_a_fixed_angle_s_matches_one_process(tmp_path):
    want = _s("cellxangle", "direct", tmp_path)
    got = on_ranks(_s, "cellxangle", "direct", tmp_path, ranks=3, limit=600)[0]
    assert _rel(got, want) < GATES["direct"]


@needs_stack
@pytest.mark.parametrize(
    ("case", "ranks", "swap"), [("cellx", 3, False), ("c1edge", 4, True)]
)
def test_fields_keep_the_bloch_condition_and_match_one_process(
    case, ranks, swap, tmp_path
):
    field, _, value, grad = _fields(case, "direct", tmp_path)
    got, phase, v, g = on_ranks(
        _fields, case, "direct", tmp_path, swap=swap, ranks=ranks, limit=600
    )[0]
    assert _rel(got, field) < FIELD_GATE
    # E_t(x + a) = exp(-i k a) E_t(x) at the seam twins, tangential (y, z)
    scale = np.abs(got).max()
    for first, second in ((0, 1), (2, 3)):
        seam = np.abs(got[second, 1:] - phase * got[first, 1:]).max()
        assert seam / scale < SEAM_GATE
    assert abs(v - value) / abs(value) < FIELD_GATE
    assert _rel(g, grad) < FIELD_GATE


@needs_stack
def test_a_bloch_backward_holds_one_factorization(tmp_path):
    # the adjoint solves run on the transposed system, A(-k): the forward's base is
    # freed after its sweep, so the backward's prepare finds none alive
    after_forward, after_backward, entries = on_ranks(
        _bases_held, tmp_path, ranks=3, limit=600
    )[0]
    assert entries == 1
    assert after_forward == (0, 0)
    assert after_backward == (0, 1)


def test_a_plain_and_a_bloch_bddc_solve_share_a_communicator(tmp_path):
    plain, bloch, _ = _plain_and_bloch(tmp_path)
    got_plain, got_bloch, names = on_ranks(
        _plain_and_bloch, tmp_path, ranks=3, limit=600
    )[0]
    assert _rel(got_plain, plain) < GATES["bddc_gmres"]
    assert _rel(got_bloch, bloch) < GATES["bddc_gmres"]
    # one coarse creator per symmetry on the MPI stack: LDL^T for the plain one, LU
    # for Bloch; without it the coarse problem stays on rank 0 and none is made
    assert names == (["cholesky", "lu"] if _petsc.has_mpi() else [])


@needs_stack
def test_petsc_refuses_a_bloch_solve_on_every_rank(tmp_path):
    message, ranks = on_ranks(_petsc_refusal, tmp_path, ranks=3)[0]
    assert "linear_solver='petsc' factors symmetric operators only" in message
    assert ranks == 3
