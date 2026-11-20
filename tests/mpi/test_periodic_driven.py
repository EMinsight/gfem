"""A driven solve on a model with one plain periodic pair, across ranks (ADR 0071).

Each row compares ranks with the same function in pytest's one process, relative to
the largest one-process entry, at ADR 0070's gates; `on_ranks` checks that ranks
agree bit for bit. The cases are `periodic_cases.cellx` (G with its x pair) and
`periodic_cases.periodic_box` on `mpi_cases.box`, whose ports touch the seam. On this
wheel box8 takes a clean partition at 3 and 4 ranks, a swapped one at 5 and 8 and a
reordered one at 16 (ADR 0071), but the tests assert results, not paths: which
seed or swap a partition takes depends on METIS in the wheel.
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import mpi_cases
import numpy as np
import periodic_cases as pc
import pytest
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc
from gfem._engine import cache
from gfem._engine.mesh import forced_search, partition_memo
from gfem._periodic import pair_names

#: Relative to the largest one-process entry: ADR 0070's gates.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-11}
#: A kept field at points, relative to its largest one-process value (1.9e-13
#: measured).
FIELD_GATE = 1e-10
DIRECT = pytest.param(
    "direct",
    marks=pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack"),
)
#: Points in units of the cell: on the seam (x = 0 and x = a, twins) and inside,
#: off every face of both meshes' grids.
POINTS = np.array(
    [
        [0.0, 0.31, 0.27],
        [1.0, 0.31, 0.27],
        [0.0, 0.73, 0.61],
        [1.0, 0.73, 0.61],
        [0.437, 0.523, 0.191],
        [0.771, 0.113, 0.836],
        [0.553, 0.447, 0.662],
        [0.129, 0.883, 0.714],
    ]
)
#: The axes of cellx's per-axis cover.
AXES = ((0.8, 0.6, 0.0), (-0.6, 0.8, 0.0), (0.0, 0.0, 1.0))
WEIGHTS = np.array([[1.0, 0.5], [2.0, -1.0]])

pytestmark = pytest.mark.filterwarnings(pc.G_OBLIQUE)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


@functools.cache
def _mesh(case, directory):
    """One mesh per case and directory in a process: its partition memo is shared."""
    if case == "cellx":
        path = Path(directory) / "g.msh"
        # pytest's one process writes it; ranks only read it
        if path.exists():
            return gfem.Mesh.from_gmsh(path, length_unit="mm")
        return pc.g_mesh(Path(directory))
    return mpi_cases.box(int(case.removeprefix("box")))


def _setup(case, directory):
    """``(mesh, build(eps, sigma, cover=None), eps, sigma, frequencies, size)``."""
    mesh = _mesh(case, str(directory))
    rng = np.random.default_rng(5)
    if case == "cellx":
        n = len(mesh.elements("core")) + len(mesh.elements("sub"))
        scale = 1.0

        def build(eps, sigma, cover=None):
            return pc.cellx(mesh, eps_r=eps, conductivity=sigma, cover=cover)

        frequencies = pc.G_FREQUENCIES
        size = np.array([pc.G_A, pc.G_B, pc.G_H]) * pc.MM
    else:
        n = len(mesh.elements("diel"))
        scale = 1e4

        def build(eps, sigma, cover=None):  # noqa: ARG001
            return pc.periodic_box(mesh, eps_r=eps, conductivity=sigma)

        frequencies = pc.BOX_FREQUENCIES
        size = np.full(3, 1e-6)
    eps = jnp.asarray(rng.uniform(2.0, 5.0, n))
    sigma = jnp.asarray(rng.uniform(0.05, 0.3, n) * scale)
    return mesh, build, eps, sigma, list(frequencies), size


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12, threads=1)


def _objective(s):
    return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
        jnp.real((0.3 - 0.7j) * s[:, 1, 0])
    )


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.abs(got - want).max() / np.abs(want).max())


# --- the functions each rank runs ------------------------------------------------


def _s(case, solver, directory):
    _, build, eps, sigma, frequencies, _ = _setup(case, directory)
    result = gfem.solve(
        build(eps, sigma), gfem.Driven(frequencies), options=_options(solver)
    )
    return np.asarray(result.s.values)


def _field(case, solver, directory):
    """p1's kept field at `POINTS`, the first solve of the job."""
    _, build, eps, sigma, frequencies, size = _setup(case, directory)
    problem = gfem.Driven(frequencies[0], excite=["p1"], keep_fields=True)
    result = gfem.solve(build(eps, sigma), problem, options=_options(solver))
    return np.asarray(result.e_fields[0].at(POINTS * size))


def _gradient(case, solver, directory):
    """The weighted |S|^2 objective and its gradient wrt eps_r and conductivity."""
    _, build, eps, sigma, frequencies, _ = _setup(case, directory)
    problem = gfem.Driven(frequencies)

    def objective(e, s):
        result = gfem.solve(build(e, s), problem, options=_options(solver))
        return _objective(result.s.values)

    value, (g_eps, g_sigma) = jax.value_and_grad(objective, argnums=(0, 1))(eps, sigma)
    return float(value), np.asarray(g_eps), np.asarray(g_sigma)


def _per_axis(solver, directory):
    """The gradient wrt cellx's per-axis cover, through the moment pass."""
    mesh, build, eps, sigma, frequencies, _ = _setup("cellx", directory)
    m = len(mesh.elements("cover"))
    rng = np.random.default_rng(6)
    ve = jnp.asarray(rng.uniform(2.0, 5.0, (m, 3)))
    vm = jnp.asarray(rng.uniform(1.0, 1.5, (m, 3)))
    problem = gfem.Driven(frequencies)

    def objective(e, u):
        cover = gfem.MaterialField(eps_r=e, mu_r=u, axes=AXES)
        result = gfem.solve(build(eps, sigma, cover), problem, options=_options(solver))
        return _objective(result.s.values)

    value, (g_eps, g_mu) = jax.value_and_grad(objective, argnums=(0, 1))(ve, vm)
    return float(value), np.asarray(g_eps), np.asarray(g_mu)


def _field_objective(case, solver, directory):
    """The gradient of |E|^2 at seam and inside points: a field cotangent."""
    _, build, eps, sigma, frequencies, size = _setup(case, directory)
    points = jnp.asarray(POINTS[[0, 1, 4]] * size)
    problem = gfem.Driven(frequencies[0], excite=["p1"], keep_fields=True)

    def objective(e):
        result = gfem.solve(build(e, sigma), problem, options=_options(solver))
        return jnp.sum(jnp.abs(result.e_fields[0].at(points)) ** 2)

    value, grad = jax.value_and_grad(objective)(eps)
    return float(value), np.asarray(grad)


def _forced_paths(start, swap, solver, directory):
    """S and the gradient on a forced partition, and what the memo recorded."""
    world = _world()
    mesh, build, eps, sigma, _, _ = _setup("cellx", directory)
    with forced_search(start=start, swap=swap):
        s = _s("cellx", solver, directory)
        gradient = _gradient("cellx", solver, directory)
    periodic = build(eps, sigma)._periodic
    memo = partition_memo(mesh._topology)[_mpi.comm_key(world), pair_names(periodic)]
    paired = cache.discretization(mesh, periodic, world).built
    plain = cache.discretization(mesh, (), world).built
    same = np.array_equal(paired.tetrahedron_order, plain.tetrahedron_order)
    return s, gradient, memo.seed, memo.swapped, world.allreduce(int(same))


# --- the rows ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "solver", "ranks"),
    [
        *(
            pytest.param("box8", "direct", r, marks=DIRECT.marks)
            for r in (3, 4, 5, 8, 16)
        ),
        ("box8", "bddc_gmres", 4),
        ("box8", "bddc_gmres", 8),
        pytest.param("box12", "direct", 32, marks=DIRECT.marks),
    ],
)
def test_s_matches_one_process(case, solver, ranks, tmp_path):
    want = _s(case, solver, tmp_path)
    got = on_ranks(_s, case, solver, tmp_path, ranks=ranks, limit=600)[0]
    assert _rel(got, want) < GATES[solver]


@pytest.mark.parametrize(("case", "ranks"), [("cellx", 4), ("box8", 8), ("box8", 16)])
def test_a_kept_field_matches_one_process_at_the_seam(case, ranks, tmp_path):
    want = _field(case, "direct", tmp_path)
    got = on_ranks(_field, case, "direct", tmp_path, ranks=ranks, limit=600)[0]
    assert _rel(got, want) < FIELD_GATE
    # the twins across the seam: one tangential field
    scale = np.abs(want).max()
    np.testing.assert_allclose(got[1, 1:], got[0, 1:], rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(got[3, 1:], got[2, 1:], rtol=0, atol=1e-10 * scale)


@pytest.mark.parametrize(
    ("case", "solver", "ranks"),
    [
        pytest.param("box8", "direct", 4, marks=DIRECT.marks),
        pytest.param("box8", "direct", 16, marks=DIRECT.marks),
        pytest.param("cellx", "direct", 3, marks=DIRECT.marks),
        ("cellx", "bddc_gmres", 3),
    ],
)
def test_an_s_gradient_matches_one_process(case, solver, ranks, tmp_path):
    value, g_eps, g_sigma = _gradient(case, solver, tmp_path)
    got = on_ranks(_gradient, case, solver, tmp_path, ranks=ranks, limit=600)[0]
    assert abs(got[0] - value) / abs(value) < GATES[solver]
    assert _rel(got[1], g_eps) < GATES[solver]
    assert _rel(got[2], g_sigma) < GATES[solver]


@pytest.mark.parametrize("solver", [DIRECT, "bddc_gmres"])
def test_a_per_axis_cover_gradient_matches_one_process(solver, tmp_path):
    value, g_eps, g_mu = _per_axis(solver, tmp_path)
    got = on_ranks(_per_axis, solver, tmp_path, ranks=3, limit=600)[0]
    assert abs(got[0] - value) / abs(value) < GATES[solver]
    assert _rel(got[1], g_eps) < GATES[solver]
    assert _rel(got[2], g_mu) < GATES[solver]


@pytest.mark.parametrize(
    ("case", "ranks"),
    [
        pytest.param("cellx", 3, marks=DIRECT.marks),
        pytest.param("box8", 16, marks=DIRECT.marks),
    ],
)
def test_a_field_objective_gradient_matches_one_process(case, ranks, tmp_path):
    value, grad = _field_objective(case, "direct", tmp_path)
    got = on_ranks(_field_objective, case, "direct", tmp_path, ranks=ranks, limit=600)
    assert abs(got[0][0] - value) / abs(value) < FIELD_GATE
    assert _rel(got[0][1], grad) < FIELD_GATE


@pytest.mark.parametrize(
    ("start", "swap"), [(1, False), (None, True)], ids=["seed 1", "swap"]
)
@pytest.mark.parametrize("solver", [DIRECT])
def test_forced_paths_match_one_process_and_are_recorded(start, swap, solver, tmp_path):
    s = _s("cellx", solver, tmp_path)
    value, g_eps, g_sigma = _gradient("cellx", solver, tmp_path)
    got = on_ranks(_forced_paths, start, swap, solver, tmp_path, ranks=4, limit=600)[0]
    s_ranks, (v, ge, gs), seed, swapped, same = got
    assert _rel(s_ranks, s) < GATES[solver]
    assert abs(v - value) / abs(value) < GATES[solver]
    assert _rel(ge, g_eps) < GATES[solver]
    assert _rel(gs, g_sigma) < GATES[solver]
    assert seed >= (start or 0)
    if swap:
        assert swapped == (True,)
    assert same == 4  # the plain mesh has the paired one's local tets, in order
