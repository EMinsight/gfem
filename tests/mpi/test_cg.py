"""A real symmetric BDDC solve across ranks: gfem's CG on agreed products (ADR 0500).

Every value is compared bit for bit across ranks (`on_ranks`), so each rank stops at
the same CG iteration; the solution's functionals match pytest's one process, where
NGSolve's ``CGSolver`` runs, to `GATE`. The layout's state products and its "held by
any rank" mask are checked on the same space.
"""

import gradient_cases as cases
import ngsolve as ngs
import numpy as np
import pytest
from ranks import on_ranks

from gfem import _mpi
from gfem._engine.cache import discretization
from gfem._engine.domain import h1_space
from gfem._linalg import Operator, prepare
from gfem._linalg.layout import ONE_PROCESS, RankLayout

#: Relative to the one-process value: ADR 0070's BDDC gate.
GATE = 1e-9
ORDER = 3


def _space(directory):
    comm = _mpi.resolve_comm()
    model = cases.capacitor(directory)
    built = discretization(model.mesh, (), comm).built
    space = h1_space(model, built, ORDER, ("bottom",))
    layout = ONE_PROCESS if comm is None else RankLayout(space, comm)
    return comm, space, layout


def _cg_solve(directory):
    """Two loads on the capacitor's Laplacian, grounded at its bottom."""
    comm, space, layout = _space(directory)
    u, v = space.TnT()
    eps = space.mesh.MaterialCF({"oxide": 3.9, "nitride": 7.5}, default=1.0)
    form = ngs.BilinearForm(space, symmetric=True)
    form += eps * ngs.grad(u) * ngs.grad(v) * ngs.dx
    loads = np.stack(
        [
            np.array(ngs.LinearForm(f * v * ngs.dx).Assemble().vec.FV().NumPy())
            for f in (ngs.CF(1.0), ngs.x + 2 * ngs.z)
        ],
        axis=1,
    )
    operator = Operator(form, space.FreeDofs(), layout=None if comm is None else layout)
    handle = prepare(operator, symmetry="hermitian", kind="bddc_gmres", tolerance=1e-11)
    try:
        assert handle.layout is layout
        states, counts = handle.solve_counted(loads)
    finally:
        handle.free()
    Q = layout.state_qr(states)
    ones = layout.held_by_any(np.ones(space.ndof, bool)).astype(float)
    return {
        "counts": counts,
        "pair": layout.pair(loads, states),
        "norms": [layout.state_norm(states[:, j]) for j in range(2)],
        "inner": layout.state_inner(states[:, 0], states[:, 1]),
        "gram": layout.state_gram(states, states),
        "orthonormal": np.abs(layout.state_gram(Q, Q) - np.eye(2)).max(),
        # the projection of the constant onto the solutions' span: phase-free
        "projection": np.linalg.norm(layout.state_gram(Q, ones)),
    }


@pytest.fixture(scope="module")
def one_process(tmp_path_factory):
    return _cg_solve(tmp_path_factory.mktemp("one"))


@pytest.mark.parametrize("ranks", [3, 4])
def test_cg_matches_one_process_with_one_iteration_count(tmp_path, one_process, ranks):
    rows = on_ranks(_cg_solve, tmp_path, ranks=ranks)
    got = rows[0]  # every rank's value is the same, bit for bit
    # against one process an observation, not a gate (ticket 02 Q12)
    print(f"CG iterations: {ranks} ranks {got['counts']}, one {one_process['counts']}")  # noqa: T201
    for key in ("pair", "norms", "inner", "gram", "projection"):
        want = np.asarray(one_process[key])
        scale = np.abs(want).max()
        assert np.abs(np.asarray(got[key]) - want).max() <= GATE * scale, key
    assert got["orthonormal"] < 1e-13


def _layout_counts(directory):
    """Each state product counts each DOF once; a mask set on one rank reaches all."""
    comm, space, layout = _space(directory)
    ones = layout.exchange(layout.owners.astype(float))
    # the last rank marks the DOFs it holds and does not own
    mark = ~layout.owners if comm.rank == comm.size - 1 else np.zeros(space.ndof, bool)
    held = layout.held_by_any(mark)
    columns = np.stack([ones, 2 * ones], axis=1)
    return {
        "global": space.ndofglobal,
        "inner": layout.state_inner(ones, ones),
        "norm": layout.state_norm(ones),
        "gram": layout.state_gram(columns, columns),
        "marked": int(_mpi.reduce_bcast(comm, np.array([int(mark.sum())]))[0]),
        "held and owned": int(
            _mpi.reduce_bcast(comm, np.array([int((held & layout.owners).sum())]))[0]
        ),
        # a holder that did not mark it still sees the mark
        "held, unmarked": int(
            _mpi.reduce_bcast(comm, np.array([int((held & ~mark).sum())]))[0]
        ),
    }


def test_state_products_count_each_dof_once(tmp_path):
    rows, *_ = on_ranks(_layout_counts, tmp_path, ranks=4)
    n = rows["global"]
    assert rows["inner"] == n
    assert rows["norm"] == pytest.approx(np.sqrt(n), rel=1e-14)
    np.testing.assert_array_equal(rows["gram"], [[n, 2 * n], [2 * n, 4 * n]])
    assert rows["marked"] > 0  # some DOF is shared, or this proves nothing
    assert rows["held and owned"] == rows["marked"]
    assert rows["held, unmarked"] >= rows["marked"]
