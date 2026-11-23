"""A lumped-port driven solve through the real `gfem.solve` on the NGSolve wheel.

Every test compares ranks with the same function in pytest's one process: S bit
for bit across ranks (`on_ranks`), to `S_GATE` against one process, and the Krylov
iterations exactly. Only the count shows a Krylov method that lost its global
products, since such a method still converges to the right S.
"""

import inspect
import os
import re
import sys
from pathlib import Path

import gradient_cases as cases
import numpy as np
import pytest
from ranks import RanksFailed, on_ranks, on_ranks_argv

import gfem
from gfem import _mpi, _petsc
from gfem._engine.cache import ENTRIES, Entry
from gfem._engine.driven import DrivenRule, driven_operator
from gfem._engine.port_network import parallel_scale
from gfem._engine.runtime import task_manager
from gfem._engine.system_operator import SystemOperator
from gfem._linalg import ResolvedLinearSolver, bddc, blas, coarse, prepare
from gfem._linalg.base import local_matrix
from gfem._linalg.direct import INVERSE
from gfem._linalg.gmres import Gmres
from gfem._linalg.pardiso import pardiso_inverse
from gfem._linalg.pieces import AssembledPieces

#: Relative to the largest entry of the one-process S: the BDDC rank gate.
S_GATE = 1e-9
FREQUENCY = 5e9
#: Close enough for one BDDC setup to serve all three.
SWEEP = (4.9e9, 5.0e9, 5.1e9)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _largest(comm, values: np.ndarray) -> float:
    from mpi4py import MPI

    return comm.allreduce(float(np.abs(values).max(initial=0.0)), op=MPI.MAX)


def _s_and_iterations(directory, frequencies):
    """S, the Krylov iterations and BDDC's setups of `layered_line`, one thread each."""
    seen = {"setups": 0}
    forward = DrivenRule.forward
    set_up = bddc._set_up

    def count_setups(*args, **kwargs):
        seen["setups"] += 1
        return set_up(*args, **kwargs)

    def spy(self, model, *, keep, request=None):
        outputs, entry = forward(self, model, keep=keep, request=request)
        seen["iterations"] = outputs["iterations"]
        return outputs, entry

    DrivenRule.forward = spy
    bddc._set_up = count_setups
    try:
        result = gfem.solve(
            cases.layered_line(directory),
            gfem.Driven(list(frequencies)),
            options=cases.tight("bddc_gmres", threads=1),
        )
    finally:
        DrivenRule.forward = forward
        bddc._set_up = set_up
    return np.asarray(result.s.values), np.asarray(seen["iterations"]), seen["setups"]


def _assert_agrees(directory, frequencies, ranks):
    s_one, iterations_one, setups_one = _s_and_iterations(directory, frequencies)
    (s, iterations, setups), *_ = on_ranks(
        _s_and_iterations, directory, frequencies, ranks=ranks, limit=300
    )
    error = np.abs(s - s_one).max() / np.abs(s_one).max()
    print(  # noqa: T201
        f"ranks={ranks} S error={error:.3g} iterations={iterations.tolist()} "
        f"one process={iterations_one.tolist()} setups={setups}"
    )
    assert error < S_GATE, f"S off by {error:.3g} relative"
    np.testing.assert_array_equal(iterations, iterations_one)
    assert setups == setups_one
    return setups


@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_lumped_s_and_iterations_match_one_process(tmp_path, ranks):
    _assert_agrees(tmp_path, (FREQUENCY,), ranks)


@pytest.mark.parametrize("ranks", [3, 4])
def test_a_sweep_reusing_bddc_matches_one_process(tmp_path, ranks):
    assert _assert_agrees(tmp_path, SWEEP, ranks) < len(SWEEP)


def test_a_capped_warm_solve_sets_bddc_up_on_every_rank(tmp_path):
    # BDDC set up at 1 GHz and reused at 60 GHz stops at its cap, set up again, and
    # goes on from the partial iterate, by iteration counts all ranks hold
    assert _assert_agrees(tmp_path, (1e9, 60e9), 4) == 2


def _a_short_stop(directory):
    """S with one GMRES stopped early, above the tolerance: it restarts from there."""
    original = Gmres._SolveImpl
    iterations = []

    def stops_short(self, rhs, sol):
        maxiter = self.maxiter
        if not iterations:
            self.maxiter = min(2, maxiter)
        try:
            original(self, rhs=rhs, sol=sol)
        finally:
            self.maxiter = maxiter
        iterations.append(self.iterations)

    Gmres._SolveImpl = stops_short
    try:
        s, *_ = _s_and_iterations(directory, (FREQUENCY,))
    finally:
        Gmres._SolveImpl = original
    return s, iterations[:2]


def test_a_short_stop_restarts_on_every_rank(tmp_path):
    s_one, *_ = _s_and_iterations(tmp_path, (FREQUENCY,))
    (s, iterations), *_ = on_ranks(_a_short_stop, tmp_path, ranks=4, limit=300)
    assert iterations[0] == 2
    assert iterations[1] > 0
    assert np.abs(s - s_one).max() / np.abs(s_one).max() < S_GATE


# --- the coarse problem's two routes ------------------------------------------------

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


#: `coarse.reduction` per route: a tiny mesh's own is to one rank.
EVERY_RANK, HALF, ONE_RANK = "every rank", "half", "one rank"
REDUCTIONS = {
    EVERY_RANK: lambda _, __: 1,
    HALF: lambda _, __: 2,
    ONE_RANK: coarse.reduction,
}
#: Sequential METIS on every second rank: the rule's thresholds patched, since a
#: tiny wirebasket takes PT-Scotch on one rank. ``"learnt"`` keeps the columns'
#: threshold, so a set-up takes METIS from enough columns or a pattern seen before.
#: ``"capped"``: the rule of a stack with the ordering cap, PT-Scotch on every
#: second rank, whatever the stack.
METIS, LEARNT, CAPPED = "metis", "learnt", "capped"
THRESHOLDS = {
    METIS: {"COARSE_EVERY": 2, "COARSE_DOFS_PER_RANK": 1, "METIS_COLUMNS": 1},
    LEARNT: {"COARSE_EVERY": 2, "COARSE_DOFS_PER_RANK": 1},
    CAPPED: {"COARSE_EVERY": 2, "COARSE_DOFS_PER_RANK": 1},
}


def _take_route(route):
    """Put the coarse problem on ``route`` in this process: `REDUCTIONS`, `THRESHOLDS`.

    ``"rank 0"`` is what a wheel-only install runs. Every other route but
    ``"capped"`` takes the stock stack's rule, which `REDUCTIONS` patches, on any
    stack.
    """
    if route == "rank 0":
        bddc._coarse_on_mumps = lambda *_, **__: False
        return
    kind = coarse.CAPPED if route == CAPPED else coarse.STOCK
    coarse.stack_kind = lambda _comm: kind
    if route in THRESHOLDS:
        for name, value in THRESHOLDS[route].items():
            setattr(coarse, name, value)
    else:
        coarse.reduction = REDUCTIONS[route]


def _factored():
    """Each coarse set-up's preconditioner type, telescoping factor and ordering."""
    factored = []
    set_up = coarse.CoarseMumps.set_up

    def spy(self):
        set_up(self)
        pc = self._ksp.getPC()
        factored.append((pc.getType(), self._factor, self._ordering))

    coarse.CoarseMumps.set_up = spy
    return factored


def _on_a_route(directory, frequencies, route):
    """`_s_and_iterations` with the coarse problem on ``route``; what factored it."""
    _take_route(route)
    factored = _factored()
    return (*_s_and_iterations(directory, frequencies), factored)


def _assert_routes_agree(directory, frequencies, ranks, *, routes):
    reference, *_ = on_ranks(
        _on_a_route, directory, frequencies, "rank 0", ranks=ranks, limit=300
    )
    s_zero, iterations_zero, setups_zero, _ = reference
    seen = {}
    for route in routes:
        (s, iterations, setups, factored), *_ = on_ranks(
            _on_a_route, directory, frequencies, route, ranks=ranks, limit=300
        )
        error = np.abs(s - s_zero).max() / np.abs(s_zero).max()
        print(  # noqa: T201
            f"ranks={ranks} {route}: S error={error:.3g} "
            f"iterations={iterations.tolist()} rank 0={iterations_zero.tolist()} "
            f"factored={factored}"
        )
        assert error < S_GATE, f"S off by {error:.3g} relative"
        np.testing.assert_array_equal(iterations, iterations_zero)
        assert setups == setups_zero
        seen[route] = factored
    return seen


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_coarse_on_mumps_matches_the_rank_0_route(tmp_path, ranks):
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[EVERY_RANK])
    assert seen[EVERY_RANK] == [("cholesky", 1, coarse.PT_SCOTCH)]


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_telescoping_changes_neither_iterations_nor_s(tmp_path, ranks):
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[HALF, ONE_RANK])
    assert seen[HALF] == [("telescope", 2, coarse.PT_SCOTCH)]
    assert seen[ONE_RANK] == [("telescope", ranks, coarse.PT_SCOTCH)]


@needs_stack
def test_a_sweep_reusing_bddc_on_either_coarse_route(tmp_path):
    _assert_routes_agree(tmp_path, SWEEP, 4, routes=[EVERY_RANK, ONE_RANK])


@needs_stack
@pytest.mark.parametrize("ranks", [4, 8])
def test_a_metis_coarse_changes_neither_iterations_nor_s(tmp_path, ranks):
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[METIS])
    assert seen[METIS] == [("telescope", 2, coarse.METIS)]


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_a_capped_stacks_coarse_changes_neither_iterations_nor_s(tmp_path, ranks):
    # three ranks on one node: every second rank is 2, an even layout
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[CAPPED])
    assert seen[CAPPED] == [("telescope", 2, coarse.PT_SCOTCH)]


@needs_stack
def test_a_capped_stack_takes_one_rule_for_many_columns_or_a_pattern_seen(tmp_path):
    # METIS would order from 6 columns, or from the second set-up, on a stock stack
    seen = _assert_routes_agree(tmp_path, SWEEP, 4, routes=[CAPPED])
    assert {f[1:] for f in seen[CAPPED]} == {(2, coarse.PT_SCOTCH)}


def _kind_and_nodes():
    """The stack kind, each rank's node and whether the stack has ParMETIS.

    `_petsc.has_parmetis` initialises PETSc, so it is read on the ranks: on the MPI
    stack it would make pytest's process an MPI singleton, whose state later
    launches inherit and whose finalize can hang at exit.
    """
    world = _world()
    return coarse.stack_kind(world), _mpi.node_ids(world), _petsc.has_parmetis()


@needs_stack
def test_every_rank_reads_the_stack_kind_and_one_node_here():
    (kind, nodes, parmetis), *_ = on_ranks(_kind_and_nodes, ranks=4)
    expected = coarse.CAPPED if _petsc.has_paraord_cap() else coarse.STOCK
    assert kind == (coarse.PARMETIS if parmetis else expected)
    assert nodes == (0, 0, 0, 0)  # a launch of this tier is on one node


@needs_stack
def test_a_sweep_serving_enough_columns_takes_metis_from_its_first_set_up(tmp_path):
    # 3 frequencies of 2 ports on one set-up: 6 columns; one frequency: 2
    seen = _assert_routes_agree(tmp_path, SWEEP, 4, routes=[LEARNT])
    assert seen[LEARNT][0] == ("telescope", 2, coarse.METIS)
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), 4, routes=[LEARNT])
    assert seen[LEARNT] == [("telescope", 4, coarse.PT_SCOTCH)]


def _analyses():
    """MUMPS's symbolic factorizations so far, the most any rank ran, read on call.

    A coarse analysis runs on each rank telescoping keeps, so the most over the
    ranks is the analyses.
    """
    from mpi4py import MPI

    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    stage = PETSc.Log.Stage("gfem coarse analyses")
    events = [PETSc.Log.Event(name) for name in ("MatCholFctrSym", "MatLUFactorSym")]
    stage.push()

    def count():
        here = sum(event.getPerfInfo(stage.id)["count"] for event in events)
        return _world().allreduce(here, op=MPI.MAX)

    return count


def _designs(directory, designs, frequencies):
    """S of each design in turn, ``designs`` substrate permittivities, on one mesh."""
    model = cases.layered_line(directory)
    options = cases.tight("bddc_gmres", threads=1)
    s = []
    for eps in designs:
        design = model.replace(
            materials={**model.materials, "sub": gfem.Material(eps_r=eps)}
        )
        result = gfem.solve(design, gfem.Driven(list(frequencies)), options=options)
        s.append(np.asarray(result.s.values))
    return np.stack(s)


def _designs_on_a_route(directory, designs, frequencies, route):
    """`_designs` on ``route``, with its coarse set-ups and analyses on every rank."""
    _take_route(route)
    factored = _factored()
    analyses = _analyses()
    s = _designs(directory, designs, frequencies)
    return s, factored, analyses()


#: A sweep BDDC set up at 1 GHz caps at 60 GHz and sets up again: one wirebasket.
REBUILT = (1e9, 60e9)


@needs_stack
@pytest.mark.parametrize("route", [ONE_RANK, METIS, CAPPED])
@pytest.mark.parametrize(
    ("designs", "frequencies", "set_ups"),
    [((2.0,), REBUILT, 2), ((2.0, 3.0), (FREQUENCY,), 2)],
    ids=["sweep with a rebuild", "two designs"],
)
def test_set_ups_on_one_wirebasket_analyse_it_once_on_every_rank(
    tmp_path, route, designs, frequencies, set_ups
):
    s_one = _designs(tmp_path, designs, frequencies)
    # bitwise across the ranks by `on_ranks`
    (s, factored, analyses), *_ = on_ranks(
        _designs_on_a_route, tmp_path, designs, frequencies, route, ranks=4, limit=300
    )
    print(f"{route}: factored={factored} analyses={analyses}")  # noqa: T201
    assert len(factored) == set_ups
    assert analyses == 1
    if route == CAPPED:  # no METIS on the pattern seen
        assert {f[1:] for f in factored} == {(2, coarse.PT_SCOTCH)}
    assert np.abs(s - s_one).max() / np.abs(s_one).max() < S_GATE


@needs_stack
def test_a_pattern_seen_before_takes_metis_and_keeps_its_analysis(tmp_path):
    designs = (2.0, 3.0, 4.0)
    s_one = _designs(tmp_path, designs, (FREQUENCY,))
    (s, factored, analyses), *_ = on_ranks(
        _designs_on_a_route,
        tmp_path,
        designs,
        (FREQUENCY,),
        LEARNT,
        ranks=4,
        limit=300,
    )
    print(f"factored={factored} analyses={analyses}")  # noqa: T201
    # 2 columns: PT-Scotch first, then METIS on the pattern seen, analysed once
    assert [ordering for *_, ordering in factored] == [
        coarse.PT_SCOTCH,
        coarse.METIS,
        coarse.METIS,
    ]
    assert analyses == 2
    assert np.abs(s - s_one).max() / np.abs(s_one).max() < S_GATE


def _with_the_default_inverse_forbidden(directory):
    """NGSolve's default inverse raises: the coarse route must not fall back on it."""
    import ngsolve as ngs

    class Forbidden(ngs.la.SparseFactorizationInterface):
        def __init__(self, *_, **__):
            raise AssertionError("gfem used NGSolve's default inverse type")

    ngs.la.RegisterInverseType("gfem_forbidden", Forbidden)
    ngs.BaseMatrix.SetDefaultInverseType("gfem_forbidden")
    return _s_and_iterations(directory, (FREQUENCY,))[1]


@needs_stack
def test_the_coarse_route_never_reads_ngsolves_default_inverse(tmp_path):
    iterations, *_ = on_ranks(_with_the_default_inverse_forbidden, tmp_path, ranks=3)
    assert iterations.size


def _rank_0_flags(directory):
    """The flags BDDC is given when the coarse problem stays on rank 0."""
    import ngsolve as ngs

    seen = []
    made = ngs.Preconditioner

    def spy(form, kind, **flags):
        seen.append(flags)
        return made(form, kind, **flags)

    bddc._coarse_on_mumps = lambda *_, **__: False
    bddc.ngs.Preconditioner = spy
    _s_and_iterations(directory, (FREQUENCY,))
    return seen


def test_the_rank_0_route_names_its_inverse(tmp_path):
    # the coarse rule of one process: PARDISO with the mkl extra, else sparse Cholesky
    (flags,), *_ = on_ranks(_rank_0_flags, tmp_path, ranks=3)
    print(f"rank 0 inverse: {flags}")  # noqa: T201
    assert flags.keys() == {"inverse"}
    assert flags["inverse"] in (
        pardiso_inverse("complex_symmetric", is_complex=True),
        INVERSE,
    )


def _forced_coarse_failures(directory, route, failures):
    """INFOG(1)=-9 ``failures`` times on the real route: a retry's count."""
    _take_route(route)
    calls = []
    real = coarse.CoarseMumps._agreed_failure

    def forced(self):
        calls.append(1)
        return coarse.MUMPS_WORKSPACE_ERROR if len(calls) <= failures else real(self)

    coarse.CoarseMumps._agreed_failure = forced
    options = cases.tight("bddc_gmres", threads=1)
    problem = gfem.Driven(FREQUENCY)
    try:
        s = gfem.solve(cases.layered_line(directory), problem, options=options).s.values
    except _mpi.AgreedError as err:
        return "coarse problem failed" in str(err), len(calls), _world().allreduce(1)
    return np.asarray(s), len(calls)


@needs_stack
@pytest.mark.parametrize("route", [EVERY_RANK, ONE_RANK])
def test_a_coarse_workspace_too_small_retries_on_a_new_ksp(tmp_path, route):
    s_one, *_ = _s_and_iterations(tmp_path, (FREQUENCY,))
    (s, calls), *_ = on_ranks(
        _forced_coarse_failures, tmp_path, route, 1, ranks=4, limit=300
    )
    assert calls == 2
    assert np.abs(s - s_one).max() < S_GATE * np.abs(s_one).max()


def _a_singular_coarse_problem(directory, route):
    """A real failure in MUMPS, not a forced value: the real paths must see it."""
    _take_route(route)
    built = coarse.distributed_aij

    def singular(*args, **kwargs):
        # one zero pivot: a zero matrix also overruns MUMPS's workspace
        aij, index = built(*args, **kwargs)
        lo, hi = aij.getOwnershipRange()
        first = np.arange(lo, min(lo + 1, hi), dtype=_petsc.require_petsc().IntType)
        aij.zeroRowsColumns(first, diag=0.0)
        return aij, index

    coarse.distributed_aij = singular
    set_ups = _count_set_ups()
    options = cases.tight("bddc_gmres", threads=1)
    problem = gfem.Driven(FREQUENCY)
    try:
        gfem.solve(cases.layered_line(directory), problem, options=options)
    except _mpi.AgreedError as err:
        return str(err), len(set_ups), _world().allreduce(1)
    raise AssertionError("a zero matrix factored")


def _count_set_ups():
    """One entry per coarse KSP set up: each takes a new options prefix."""
    set_ups = []
    prefix = coarse._petsc.next_prefix

    def counted():
        set_ups.append(1)
        return prefix()

    coarse._petsc.next_prefix = counted
    return set_ups


@needs_stack
@pytest.mark.parametrize("route", [EVERY_RANK, ONE_RANK])
def test_a_singular_coarse_problem_raises_after_one_set_up_naming_infog1(
    tmp_path, route
):
    (text, set_ups, ranks), *_ = on_ranks(
        _a_singular_coarse_problem, tmp_path, route, ranks=4, limit=300
    )
    print(text)  # noqa: T201
    assert "coarse problem failed" in text
    assert re.search(r"INFOG\(1\)=-\d+", text)
    assert set_ups == 1
    assert ranks == 4


def _telescoped_nbytes(directory):
    """Each rank's count of its coarse factor, behind telescoping to one rank."""
    _take_route(ONE_RANK)
    counted = []
    set_up = coarse.CoarseMumps.set_up

    def spy(self):
        set_up(self)
        counted.append(self.nbytes)

    coarse.CoarseMumps.set_up = spy
    _s_and_iterations(directory, (FREQUENCY,))
    return counted


@needs_stack
def test_a_telescoped_coarse_factor_counts_its_bytes_on_the_kept_rank(tmp_path):
    counted = on_ranks(_telescoped_nbytes, tmp_path, ranks=4, replicated=False)
    print(f"nbytes per rank: {counted}")  # noqa: T201
    assert counted[0]
    assert all(n > 0 for n in counted[0])
    assert all(n == 0 for rank in counted[1:] for n in rank)


class _Infog1:
    """An inner KSP whose factor reads ``value`` for INFOG(1), the rest as it is."""

    def __init__(self, ksp, value):
        self._factor = ksp.getPC().getFactorMatrix()
        self._value = value

    def getPC(self):  # noqa: N802 - petsc4py's name
        return self

    def getFactorMatrix(self):  # noqa: N802 - petsc4py's name
        return self

    def getMumpsInfog(self, i):  # noqa: N802 - petsc4py's name
        return self._value if i == 1 else self._factor.getMumpsInfog(i)

    def getMumpsInfo(self, i):  # noqa: N802 - petsc4py's name
        return self._factor.getMumpsInfo(i)


def _workspace_error_behind_telescoping(directory):
    """The first telescoped set-up's inner factor reads INFOG(1)=-9."""
    _take_route(ONE_RANK)
    set_ups = _count_set_ups()
    first = []
    real = coarse._petsc.telescope_ksp

    def forced(pc):
        inner = real(pc)
        first[:] = first or [pc.getOptionsPrefix()]
        if inner is None or pc.getOptionsPrefix() != first[0]:
            return inner
        return _Infog1(inner, coarse.MUMPS_WORKSPACE_ERROR)

    coarse._petsc.telescope_ksp = forced
    s, *_ = _s_and_iterations(directory, (FREQUENCY,))
    return s, len(set_ups)


@needs_stack
def test_a_workspace_error_behind_telescoping_retries_once(tmp_path):
    s_one, *_ = _s_and_iterations(tmp_path, (FREQUENCY,))
    (s, set_ups), *_ = on_ranks(
        _workspace_error_behind_telescoping, tmp_path, ranks=4, limit=300
    )
    assert set_ups == 2
    assert np.abs(s - s_one).max() < S_GATE * np.abs(s_one).max()


# --- what lies under the solve ------------------------------------------------------


def _system(directory):
    model = cases.layered_line(directory)
    comm = _mpi.resolve_comm()
    operator = driven_operator(model, 2, comm=comm)
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver="bddc_gmres"),
        default="bddc_gmres",
        problem="driven",
    )
    return comm, SystemOperator(operator, [], linear, width=1, adjoint_tolerance=1e-8)


def _piece_sum_against_a_fresh_assembly(directory):
    comm, system = _system(directory)
    worst = 0.0
    for frequency in SWEEP:
        operator = system.operator_at(2 * np.pi * frequency)
        operator.form.Assemble()
        fresh = local_matrix(operator.form.mat, comm).AsVector().FV().NumPy()
        summed = operator.piece_sum.matrix()
        values = local_matrix(summed, comm).AsVector().FV().NumPy()
        worst = max(worst, _largest(comm, values - fresh) / _largest(comm, fresh))
    return worst


def test_the_piece_sum_equals_a_fresh_assembly(tmp_path):
    worst, *_ = on_ranks(_piece_sum_against_a_fresh_assembly, tmp_path, ranks=4)
    assert worst < 1e-14


def _disjoint_pieces(directory):
    """Two pieces on two surfaces: no rank's pattern holds the other's entries."""
    import ngsolve as ngs

    comm, system = _system(directory)

    def on(surface):
        def integrand(u, v):
            return u.Trace() * v.Trace() * ngs.ds(surface)

        return integrand

    pieces = AssembledPieces(
        system.operator.space, [on("port1"), on("port2")], comm=comm
    )
    return pieces.pattern is None


def test_no_common_pattern_is_the_same_answer_on_every_rank(tmp_path):
    outside, *_ = on_ranks(_disjoint_pieces, tmp_path, ranks=4)
    assert outside


def _layout_counts(directory):
    """Each product of the layout on one load: 1 at each DOF, split to its owner."""
    comm, system = _system(directory)
    space = system.operator.space
    layout = system.operator.layout
    load = layout.owners.astype(complex)
    state = layout.exchange(load)
    return {
        "pair": layout.pair(load, state),
        "pair of columns": layout.pair(np.stack([load, load], axis=1), state),
        "inner": layout.inner(load, load),
        "norm": layout.norm(load),
        "parallel": parallel_scale(2j * load, load, layout),
        # one entry off, on the last rank alone: each rank must see it
        "not parallel": parallel_scale(_nudged(comm, 2j * load), load, layout),
        "global": space.ndofglobal,
        "local": comm.allreduce(space.ndof),
    }


def _nudged(comm, load):
    if comm.rank == comm.size - 1:
        load[0] += 1.0
    return load


def test_a_layout_counts_each_dof_once(tmp_path):
    rows, *_ = on_ranks(_layout_counts, tmp_path, ranks=4)
    n = rows["global"]
    assert rows["local"] > n  # some DOF is shared, or this proves nothing
    assert rows["pair"] == n
    np.testing.assert_array_equal(rows["pair of columns"], [n, n])
    assert rows["inner"] == n
    assert rows["norm"] == pytest.approx(np.sqrt(n), rel=1e-14)
    assert rows["parallel"] == 2j
    assert rows["not parallel"] is None


def _apply_then_solve(directory):
    """``apply`` takes a state and gives a load; ``solve`` brings the state back."""
    _, system = _system(directory)
    operator = system.operator_at(2 * np.pi * FREQUENCY)
    handle = prepare(
        operator, symmetry="complex_symmetric", kind="bddc_gmres", tolerance=1e-12
    )
    layout = handle.layout
    free = np.array(operator.freedofs, dtype=bool)
    state = layout.exchange(np.where(layout.owners & free, 1.0 + 0.5j, 0.0))
    load = handle.apply(state)
    back, counts = handle.solve_counted(load)
    return (
        layout.pair(load, state),
        layout.pair(handle.apply(back), state),
        counts,
    )


def test_apply_takes_a_state_and_gives_a_load(tmp_path):
    (energy, again, counts), *_ = on_ranks(_apply_then_solve, tmp_path, ranks=4)
    assert abs(again - energy) < 1e-8 * abs(energy)
    assert counts[0] > 0


# --- what a job sees ----------------------------------------------------------------


def _entries_kept(sizes_by_rank, budget):
    """Rank-dependent sizes: every rank must evict the same entries."""
    world = _world()

    class Sized:
        def __init__(self, nbytes):
            self.nbytes = nbytes

        def free(self):
            pass

    keys = [
        ENTRIES.put(Entry("em", Sized(size)), budget, world)
        for size in sizes_by_rank[world.rank]
    ]
    kept = [ENTRIES.get(k) is not None for k in keys]
    ENTRIES.clear()
    return kept


def test_every_rank_evicts_the_same_entries():
    # one node here: 102 bytes per put, so a budget of 150 keeps the newest alone,
    # and a rank's own bytes would keep all three on two ranks of three
    sizes = [[100, 100, 100], [1, 1, 1], [1, 1, 1]]
    kept, *_ = on_ranks(_entries_kept, sizes, 150, ranks=3)
    assert kept == [False, False, True]


def _mismatched_input(directory):
    """One rank changes one material: every rank hears which leaf, then goes on."""
    world = _world()
    model = cases.layered_line(directory)
    eps = 2.0 if world.rank == world.size - 1 else 1.0
    material = gfem.Material(eps_r=eps)
    model = model.replace(materials={**model.materials, "sub": material})
    try:
        gfem.solve(model, gfem.Driven(FREQUENCY))
    except _mpi.AgreedError as err:
        return "sub" in str(err), world.allreduce(1)
    raise AssertionError("different inputs ran")


def test_a_mismatched_input_raises_on_every_rank_naming_the_leaf(tmp_path):
    named, *_ = on_ranks(_mismatched_input, tmp_path, ranks=3, limit=60)
    assert named == (True, 3)


def _one_rank_fails_in_the_solve(directory):
    world = _world()
    if world.rank == world.size - 1:
        import gfem._linalg.gmres as module

        def broken(*_args, **_kwargs):
            raise OSError("this rank's own failure")

        module.Gmres._gmres = broken
    gfem.solve(cases.layered_line(directory), gfem.Driven(FREQUENCY))


def test_a_failure_on_one_rank_in_the_solve_ends_the_job(tmp_path):
    with pytest.raises(RanksFailed, match="this rank's own failure"):
        on_ranks(_one_rank_fails_in_the_solve, tmp_path, ranks=3, limit=60)


def _a_miss_on_every_rank(directory):
    """Two iterations cannot converge: the error is agreed, and a job goes on."""
    world = _world()
    # `prepare` is timed by a wrapper, which carries no keyword defaults
    inspect.unwrap(prepare).__kwdefaults__["maxiter"] = 2
    try:
        gfem.solve(cases.layered_line(directory), gfem.Driven(list(SWEEP)))
    except gfem.LinearSolverConvergenceError as err:
        return err.frequency, list(err.unsolved), world.allreduce(1)
    raise AssertionError("two iterations converged")


def test_a_convergence_error_reaches_every_rank(tmp_path):
    (frequency, unsolved, ranks), *_ = on_ranks(
        _a_miss_on_every_rank, tmp_path, ranks=3, limit=120
    )
    assert (frequency, unsolved, ranks) == (SWEEP[0], list(SWEEP), 3)


def _threads_inside_a_task_manager(*, read_the_mask):
    """What ``threads=None`` gives NGSolve and each BLAS, variable or no variable."""
    import ngsolve as ngs

    if read_the_mask:
        del os.environ["NGS_NUM_THREADS"]
    comm = _mpi.resolve_comm()
    with task_manager(None, comm):
        return ngs.GetNumThreads(), sorted({lib.get() for lib in blas.libraries()})


def test_threads_none_reads_the_variable_first():
    (count, counts), *_ = on_ranks(
        _threads_inside_a_task_manager, read_the_mask=False, ranks=3, threads=2
    )
    assert count == 2
    assert counts == [2]


def test_threads_none_without_the_variable_is_one_count_on_every_rank():
    # bitwise across ranks by `on_ranks`; ranks share one mask here, unbound
    (count, counts), *_ = on_ranks(
        _threads_inside_a_task_manager, read_the_mask=True, ranks=3
    )
    assert count == max(1, _physical_cores() // 3)
    assert counts == [count]


def _physical_cores() -> int:
    mask = frozenset(os.sched_getaffinity(0))
    return _mpi.node_share(mask, [mask])


def _blas_in_each_phase(directory):
    """Each loaded BLAS's count in a setup, in a solve, and after.

    These are the phases where gfem sets the BLAS threads.
    """
    import gfem._linalg.gmres as krylov

    seen = {"setup": set(), "solve": set()}
    set_up, solve = bddc._set_up, krylov.Gmres._gmres

    def spy_set_up(*args, **kwargs):
        seen["setup"] |= {lib.get() for lib in blas.libraries()}
        return set_up(*args, **kwargs)

    def spy_solve(*args, **kwargs):
        seen["solve"] |= {lib.get() for lib in blas.libraries()}
        return solve(*args, **kwargs)

    bddc._set_up, krylov.Gmres._gmres = spy_set_up, spy_solve
    gfem.solve(
        cases.layered_line(directory),
        gfem.Driven(list(SWEEP)),
        options=gfem.SolverOptions(linear_solver="bddc_gmres", threads=2),
    )
    return seen, sorted({lib.get() for lib in blas.libraries()})


def test_blas_runs_at_the_ranks_threads_in_a_setup_and_at_one_in_a_solve(tmp_path):
    (seen, after), *_ = on_ranks(_blas_in_each_phase, tmp_path, ranks=3, threads=1)
    assert seen == {"setup": {2}, "solve": {1}}
    assert after == [2]


#: What a launched script imports the test helpers from.
SCRIPT_PATH = (Path(__file__).parents[1], Path(__file__).parents[2] / "tools")
WAITING = """
import sys
import time

from mpi4py import MPI

import gfem
from gfem import _mpi
import gradient_cases as cases

_mpi.ENTRY_PATIENCE = 0.2
model = cases.layered_line(sys.argv[1])
if MPI.COMM_WORLD.rank == 1:
    time.sleep(2)
_mpi.entry_check(
    MPI.COMM_WORLD, "solve", model, gfem.Driven(5e9), gfem.SolverOptions()
)
"""


def test_a_long_wait_at_the_door_says_so_once_per_rank_and_goes_on(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(WAITING)
    code, log = on_ranks_argv(
        [sys.executable, str(script), str(tmp_path)],
        ranks=3,
        limit=60,
        env={"PYTHONPATH": os.pathsep.join(map(str, SCRIPT_PATH))},
    )
    assert code == 0, log
    # Only the solve door is waited on by design; reading the mesh may wait too
    # on a slow runner.
    assert log.count("waiting over 0.2 s for the other ranks at solve") == 2, log
