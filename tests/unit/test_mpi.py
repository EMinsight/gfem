"""Launcher resolution and refusals, with no MPI: stub communicators, fake mpi4py."""

import dataclasses
import importlib
import math
import os
import pickle
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import gradient_cases as cases
import jax
import mpi_cases
import numpy as np
import periodic_cases
import pytest
import ranks
import refinement_cases as rc

import gfem
from gfem import _mpi
from gfem._linalg import coarse

TASK_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "SLURM_STEP_NUM_TASKS", "PMI_SIZE")
TESTS = Path(__file__).resolve().parents[1]


class Stub:
    """As much of a communicator as resolution reads."""

    rank = 0

    def __init__(self, size: int) -> None:
        self.size = size


def fake_mpi(vendor: str = "Open MPI", world: int = 4) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        COMM_WORLD=Stub(world),
        COMM_SELF=Stub(1),
        get_vendor=lambda: (vendor, (0, 0, 0)),
        Is_initialized=lambda: True,
        Is_finalized=lambda: False,
    )


@pytest.fixture(autouse=True)
def no_launcher(monkeypatch):
    for name in TASK_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py", raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    # the hook a distributed solve chains must not outlive the test
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)


def use_mpi(monkeypatch, mpi) -> None:
    module = types.ModuleType("mpi4py")
    module.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", module)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)


# --- launched tasks ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "tasks"),
    [
        pytest.param({}, 1, id="no launcher"),
        pytest.param({"SLURM_NTASKS": "16"}, 1, id="batch script, no launcher"),
        pytest.param({"SLURM_PROCID": "3"}, 1, id="only a task id"),
        pytest.param({"SLURM_STEP_NUM_TASKS": "1"}, 1, id="srun -n 1"),
        pytest.param({"SLURM_STEP_NUM_TASKS": "4"}, 4, id="srun -n 4"),
        pytest.param(
            {"OMPI_COMM_WORLD_SIZE": "1", "SLURM_NTASKS": "16"},
            1,
            id="mpirun -n 1 in a batch job",
        ),
        pytest.param(
            {"OMPI_COMM_WORLD_SIZE": "4", "SLURM_NTASKS": "16"},
            4,
            id="mpirun -n 4 in a batch job",
        ),
        pytest.param(
            {"OMPI_COMM_WORLD_SIZE": "4", "SLURM_STEP_NUM_TASKS": "1"},
            4,
            id="mpirun inside an srun step",
        ),
        pytest.param({"PMI_SIZE": "3"}, 3, id="Hydra"),
        pytest.param({"OMPI_COMM_WORLD_SIZE": ""}, 1, id="empty"),
    ],
)
def test_launched_tasks_reads_the_launcher_variables(environment, tasks):
    assert _mpi.launched_tasks(environment) == tasks


def test_launched_tasks_reads_the_world_size_of_a_loaded_mpi4py(monkeypatch):
    use_mpi(monkeypatch, fake_mpi(world=8))
    assert _mpi.launched_tasks({}) == 8


# --- the one-process path ---------------------------------------------------------


def test_every_call_is_trivial_on_the_one_process_path():
    rows = np.arange(6.0).reshape(3, 2)
    _mpi.check_call(None, "a call", {"the rows": rows})
    _mpi.entry_check(None, "a call", None, None, None)
    assert _mpi.gather_rows(None, rows) is rows
    arrays = {"rows": rows}
    assert _mpi.bcast_arrays(None, arrays) is arrays
    assert "mpi4py" not in sys.modules


# --- resolution -------------------------------------------------------------------


def test_one_task_resolves_to_the_one_process_path_without_mpi4py():
    assert _mpi.resolve_comm(None) is None
    assert "mpi4py" not in sys.modules


def test_a_communicator_of_size_one_needs_no_mpi4py():
    assert _mpi.resolve_comm(Stub(1)) is None
    assert "mpi4py" not in sys.modules


def test_a_non_communicator_is_a_type_error():
    with pytest.raises(TypeError, match="communicator"):
        _mpi.resolve_comm("world")


@pytest.mark.parametrize("variable", TASK_VARIABLES)
def test_a_given_world_of_one_under_more_tasks_raises(variable, monkeypatch):
    mpi = fake_mpi(world=1)
    use_mpi(monkeypatch, mpi)
    monkeypatch.setenv(variable, "4")
    with pytest.raises(gfem.SolverUnavailableError, match=r"--mpi=pmix.*COMM_SELF"):
        _mpi.resolve_comm(mpi.COMM_WORLD)


def test_comm_self_under_more_tasks_stays_a_copy(monkeypatch):
    mpi = fake_mpi(world=1)
    use_mpi(monkeypatch, mpi)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert _mpi.resolve_comm(mpi.COMM_SELF) is None
    assert _mpi.resolve_comm(Stub(1)) is None


def test_a_given_world_of_one_alone_is_the_one_process_path(monkeypatch):
    mpi = fake_mpi(world=1)
    use_mpi(monkeypatch, mpi)
    assert _mpi.resolve_comm(mpi.COMM_WORLD) is None


def test_launched_resolves_to_the_world(monkeypatch):
    mpi = fake_mpi()
    use_mpi(monkeypatch, mpi)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert _mpi.resolve_comm(None) is mpi.COMM_WORLD


def test_a_launched_world_of_one_names_pmix_and_comm_self(monkeypatch):
    use_mpi(monkeypatch, fake_mpi(world=1))
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "4")
    with pytest.raises(gfem.SolverUnavailableError, match=r"--mpi=pmix.*COMM_SELF"):
        _mpi.resolve_comm(None)


@pytest.mark.parametrize("vendor", ["MPICH", "Intel MPI"])
def test_any_mpi_but_open_mpi_is_refused_above_one_rank(vendor, monkeypatch):
    use_mpi(monkeypatch, fake_mpi(vendor))
    with pytest.raises(gfem.SolverUnavailableError, match=vendor):
        _mpi.resolve_comm(Stub(2))


def test_no_vendor_refusal_at_size_one(monkeypatch):
    use_mpi(monkeypatch, fake_mpi("MPICH"))
    assert _mpi.resolve_comm(Stub(1)) is None


def _child(code: str, *paths, **environment) -> subprocess.CompletedProcess:
    python_path = os.pathsep.join(str(p) for p in paths)
    # Windows needs these to start Python and its DLLs; neither says launched.
    system = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
    env = {**system, "JAX_PLATFORMS": "cpu", "PYTHONPATH": python_path, **environment}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


RESOLVE = """
    import sys
    {block}
    from gfem import _mpi
    from gfem.errors import SolverUnavailableError
    try:
        _mpi.resolve_comm(None)
    except SolverUnavailableError as err:
        print(err)
    else:
        print("resolved")
"""


def test_launched_without_mpi4py_is_unavailable():
    block = "sys.modules['mpi4py'] = None"
    proc = _child(RESOLVE.format(block=block), OMPI_COMM_WORLD_SIZE="2")
    assert proc.returncode == 0, proc.stderr
    assert "gfem[mpi]" in proc.stdout
    assert "Open MPI" in proc.stdout


def test_launched_with_a_library_less_mpi4py_is_unavailable(tmp_path):
    package = tmp_path / "mpi4py"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "MPI.py").write_text("raise RuntimeError('cannot load MPI library')\n")
    proc = _child(RESOLVE.format(block=""), tmp_path, OMPI_COMM_WORLD_SIZE="2")
    assert proc.returncode == 0, proc.stderr
    assert "cannot load MPI library" in proc.stdout
    assert "Open MPI" in proc.stdout


@pytest.mark.parametrize("environment", [{}, {"OMPI_COMM_WORLD_SIZE": "1"}])
def test_a_one_process_solve_never_imports_mpi4py(environment, tmp_path):
    code = f"""
    import sys
    import gfem, mpi_cases
    gfem.solve(
        mpi_cases.stack({str(tmp_path)!r}),
        gfem.Electrostatic(terminals=mpi_cases.TERMINALS),
        options=gfem.SolverOptions(order=1),
    )
    print([m for m in sys.modules if m.startswith("mpi4py")])
    """
    proc = _child(code, TESTS, TESTS.parent / "tools", **environment)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("dofs", "size", "factor"),
    [
        (3_300_000, 28, 1),  # r10M at 28x4: 119 k per rank, nothing to reduce
        (3_300_000, 64, 2),  # 32 ranks, as ticket 18 ran
        (22_356_889, 896, 14),  # r10M_void_split on 32 nodes: 64 ranks, not 180
        (22_356_889, 64, 1),
        (33_000_000, 896, 14),  # ADR 0078's arithmetic at 100 M DOF: still 64
        (250_000, 8, 4),
        (1_000, 8, 8),  # a tiny mesh: one rank
        (1_000, 1, 1),
    ],
)
def test_the_coarse_problem_runs_on_at_most_one_rank_per_100k_dofs_and_64_ranks(
    dofs, size, factor
):
    assert coarse.reduction(dofs, size) == factor
    assert math.ceil(size / factor) <= max(1, dofs // coarse.DOFS_PER_RANK)
    assert math.ceil(size / factor) <= coarse.MAX_RANKS


# --- refusals ---------------------------------------------------------------------


def test_the_error_is_a_not_implemented_error_with_its_key():
    err = gfem.DistributedUnsupportedError("adaptive_sweep", "no path; use COMM_SELF")
    assert isinstance(err, NotImplementedError)
    assert err.feature == "adaptive_sweep"
    assert str(err) == "no path; use COMM_SELF"
    copy = pickle.loads(pickle.dumps(err))  # noqa: S301
    assert (copy.feature, str(copy)) == (err.feature, str(err))


@pytest.mark.parametrize("key", mpi_cases.KEYS)
def test_a_solve_above_one_rank_raises_its_key_and_a_remedy(key, tmp_path, monkeypatch):
    model, problem, options = mpi_cases.refused(key, tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    with pytest.raises(gfem.DistributedUnsupportedError, match="COMM_SELF") as caught:
        gfem.solve(model, problem, options=options, comm=Stub(4))
    assert caught.value.feature == mpi_cases.feature(key)


def test_a_launched_job_raises_without_being_given_a_communicator(
    tmp_path, monkeypatch
):
    model, problem, options = mpi_cases.refused("arpack", tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.solve(model, problem, options=options)
    assert caught.value.feature == "arpack"


def test_a_job_passes_its_communicator_on(tmp_path, monkeypatch):
    model, problem, options = mpi_cases.refused("arpack", tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    with pytest.raises(gfem.DistributedUnsupportedError):
        gfem.Job(model, problem, options).run(comm=Stub(4))


def test_a_refusal_under_jax_keeps_its_class(tmp_path, monkeypatch):
    line = cases.layered_line(tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    n = len(line.mesh.elements("sub")) + len(line.mesh.elements("cover"))
    adaptive = gfem.Driven([1e9, 5e9], sweep="adaptive")

    def objective(eps):
        model = cases.with_fields(line, {"eps_r": eps})
        return gfem.solve(model, adaptive, comm=Stub(4)).s.values.real.sum()

    with pytest.raises(gfem.DistributedUnsupportedError):
        jax.grad(objective)(np.ones(n))


def test_a_port_solve_is_all_that_passes_so_far(tmp_path):
    line = cases.layered_line(tmp_path)
    wave = line.replace(ports=[gfem.WavePort("w", "port1")])
    frequency = gfem.Driven(5e9)
    for solver in ("auto", "bddc_gmres", "direct", "petsc"):
        options = gfem.SolverOptions(linear_solver=solver)
        assert _mpi.refuse(line, frequency, options) is None
        assert _mpi.refuse(wave, frequency, options) is None
    for options in (gfem.SolverOptions(), *SHARED_REFUSALS.values()):
        # a port mode ignores the device and the linear solver: rank 0 runs it
        assert _mpi.refuse(wave, gfem.PortMode("w", 5e9), options) is None
    kept = gfem.Driven(5e9, keep_fields=[5e9])
    assert _mpi.refuse(line, kept, gfem.SolverOptions()) is None


def test_an_eigenmode_solve_passes_with_auto_and_slepc(tmp_path):
    line = cases.layered_line(tmp_path)
    problem = gfem.Eigenmode(target=5e9)
    for solver in ("auto", "slepc"):
        for linear in ("auto", "direct", "bddc_gmres"):
            options = gfem.SolverOptions(eigen_solver=solver, linear_solver=linear)
            assert _mpi.refuse(line, problem, options) is None


MODE_FIELD_OUTPUTS = ("q_port", "participation", "bulk", "principal", "sheet")


@pytest.mark.parametrize("name", [*MODE_FIELD_OUTPUTS, "e_field"])
def test_a_mode_field_cotangent_above_one_rank_raises_its_key(name, tmp_path):
    from gfem._eigen.rule import EigenmodeRule

    line = cases.layered_line(tmp_path)
    rule = EigenmodeRule(line, gfem.Eigenmode(target=5e9), gfem.SolverOptions())
    # the check reads the communicator's size, the output and the wanted kinds:
    # replicated values, so every rank raises it, and nothing communicates
    rule.comm = Stub(4)
    eps = frozenset({("material", "sub", "eps_r")})
    with pytest.raises(gfem.DistributedUnsupportedError, match="COMM_SELF") as caught:
        rule.nondifferentiable_output(name, eps)
    assert caught.value.feature == "mode_field_gradient"
    assert "stop_gradient" in str(caught.value)


def test_the_eigenfrequencies_still_differentiate_above_one_rank(tmp_path):
    from gfem._eigen.rule import EigenmodeRule

    line = cases.layered_line(tmp_path)
    rule = EigenmodeRule(line, gfem.Eigenmode(target=5e9), gfem.SolverOptions())
    rule.comm = Stub(4)
    eps = frozenset({("material", "sub", "eps_r")})
    assert rule.nondifferentiable_output("omega", eps) is None
    assert rule.nondifferentiable_output("cluster", eps) is None
    # the backward error has no derivative anywhere: one process's reason
    reason = rule.nondifferentiable_output("error", eps)
    assert reason is not None
    assert "backward error" in reason


def test_a_bloch_eigenmode_on_a_periodic_cell_keeps_its_refusal(tmp_path):
    cell = periodic_cases.c1(periodic_cases.c1_mesh(tmp_path))
    problem = gfem.Eigenmode(target=5e9, wave_vector=periodic_cases.C1_K)
    with pytest.raises(gfem.DistributedUnsupportedError, match="COMM_SELF") as caught:
        _mpi.refuse(cell, problem, gfem.SolverOptions())
    assert caught.value.feature == "bloch"


def test_one_periodic_pair_passes_for_a_driven_solve_plain_or_bloch(tmp_path):
    cell = mpi_cases._cell(tmp_path)  # one pair
    options = gfem.SolverOptions(linear_solver="direct")
    assert _mpi.refuse(cell, gfem.Driven(5e9), options) is None
    bloch = gfem.Driven(5e9, wave_vector=periodic_cases.C1_K)
    assert _mpi.refuse(cell, bloch, options) is None
    angle = gfem.Driven(
        [5e9, 6e9], wave_vector=periodic_cases.C1_K, wave_vector_frequency=5e9
    )
    assert _mpi.refuse(cell, angle, options) is None


def test_one_periodic_pair_passes_for_a_plain_eigenmode_solve(tmp_path):
    cell = mpi_cases._cell(tmp_path)  # one pair
    options = gfem.SolverOptions(linear_solver="direct")
    eigen = gfem.Eigenmode(target=5e9, n_modes=1)
    assert _mpi.refuse(cell, eigen, options) is None


def test_a_bloch_eigenmode_keeps_its_refusal(tmp_path):
    cell = mpi_cases._cell(tmp_path)  # one pair
    problem = gfem.Eigenmode(target=5e9, n_modes=1, wave_vector=periodic_cases.C1_K)
    with pytest.raises(gfem.DistributedUnsupportedError, match="eigenmode") as caught:
        _mpi.refuse(cell, problem, gfem.SolverOptions())
    assert caught.value.feature == "bloch"


def test_two_periodic_pairs_refuse_a_driven_and_an_eigenmode_solve(tmp_path):
    cell = periodic_cases.c2(periodic_cases.c2_mesh(tmp_path))  # two pairs
    options = gfem.SolverOptions(linear_solver="direct")
    bloch = gfem.Driven(5e9, wave_vector=periodic_cases.C2_K)
    for problem in (gfem.Driven(5e9), bloch, gfem.Eigenmode(target=5e9, n_modes=1)):
        with pytest.raises(gfem.DistributedUnsupportedError) as caught:
            _mpi.refuse(cell, problem, options)
        assert caught.value.feature == "periodic"


#: Options every problem refuses above one rank, by the key each raises.
SHARED_REFUSALS = {
    "device": gfem.SolverOptions(linear_solver="direct", device="cuda"),
    "bddc_order_1": gfem.SolverOptions(order=1, linear_solver="bddc_gmres"),
    "bddc_order_1_auto": gfem.SolverOptions(order=1),
    "petsc_preconditioner": gfem.SolverOptions(
        linear_solver="petsc", petsc_options={"pc_type": "ilu"}
    ),
}


def _problem(key, directory):
    """``(model, problem)`` of a refused case, or of an electrostatic solve."""
    if key == "electrostatic":
        return mpi_cases.stack(directory), mpi_cases.ELECTROSTATIC
    return mpi_cases.refused(key, directory)[:2]


@pytest.mark.parametrize("problem_key", ["electrostatic", "arpack"])
@pytest.mark.parametrize("key", SHARED_REFUSALS)
def test_the_shared_refusals_hold_for_every_problem(
    key, problem_key, tmp_path, monkeypatch
):
    model, problem = _problem(problem_key, tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.solve(model, problem, options=SHARED_REFUSALS[key], comm=Stub(4))
    assert caught.value.feature == mpi_cases.feature(key)


@pytest.mark.parametrize("key", ["thermal", "thermal_transient", "dc_conduction"])
def test_a_refused_physics_raises_its_own_key_before_the_shared_ones(key, tmp_path):
    model, problem, _ = mpi_cases.refused(key, tmp_path)
    for options in SHARED_REFUSALS.values():
        with pytest.raises(gfem.DistributedUnsupportedError) as caught:
            _mpi.refuse(model, problem, options)
        assert caught.value.feature == mpi_cases.feature(key)


@pytest.mark.parametrize("solver", ["auto", "direct", "bddc_gmres", "petsc"])
def test_an_electrostatic_solve_has_a_distributed_path(solver, tmp_path):
    options = gfem.SolverOptions(order=2, linear_solver=solver)
    _mpi.refuse(mpi_cases.stack(tmp_path), mpi_cases.ELECTROSTATIC, options)
    assert "electrostatic" not in _mpi._REFUSALS


def test_an_electrostatic_solve_on_a_periodic_model_is_refused(tmp_path):
    model, _, options = mpi_cases.refused("periodic", tmp_path)
    problem = gfem.Electrostatic(terminals={"a": "z_max"}, ground="z_min")
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        _mpi.refuse(model, problem, options)
    assert caught.value.feature == "periodic"


def test_element_mean_has_region_means_text():
    assert _mpi._REFUSALS["element_mean"] == _mpi._REFUSALS["region_mean"].replace(
        "region_mean", "element_mean"
    )


def test_every_refusal_key_is_snake_case():
    assert all(re.fullmatch(r"[a-z0-9_]+", key) for key in _mpi._REFUSALS)


@pytest.mark.parametrize("key", ["power_flow", "region_mean", "electric_energy"])
def test_a_refused_measurement_names_the_route_through_a_saved_result(key):
    text = _mpi._REFUSALS[key]
    assert "Job.run(out=)" in text
    assert "load_result" in text
    assert "Field.at" not in text  # it gives no power or energy


PC_REFUSED = [
    {"pc_type": "ilu"},
    {"pc_type": "icc"},
    {"pc_type": "lu", "pc_factor_mat_solver_type": "superlu"},
    {"pc_type": "cholesky", "pc_factor_mat_solver_type": "petsc"},
]
PC_RUNS = [
    {},
    {"pc_type": "bjacobi", "sub_pc_type": "ilu"},
    {"pc_type": "asm", "sub_pc_type": "icc"},
    {"pc_type": "lu"},
    {"pc_type": "cholesky", "pc_factor_mat_solver_type": "mumps"},
    {"pc_type": "gamg"},
]


@pytest.mark.parametrize("petsc_options", PC_REFUSED)
def test_a_petsc_preconditioner_with_no_distributed_form_is_refused(
    petsc_options, tmp_path
):
    line = cases.layered_line(tmp_path)
    options = gfem.SolverOptions(linear_solver="petsc", petsc_options=petsc_options)
    with pytest.raises(gfem.DistributedUnsupportedError, match="bjacobi") as caught:
        _mpi.refuse(line, gfem.Driven(5e9), options)
    assert caught.value.feature == "petsc_preconditioner"


@pytest.mark.parametrize("petsc_options", PC_RUNS)
def test_a_petsc_preconditioner_with_a_distributed_form_passes(petsc_options, tmp_path):
    line = cases.layered_line(tmp_path)
    options = gfem.SolverOptions(linear_solver="petsc", petsc_options=petsc_options)
    assert _mpi.refuse(line, gfem.Driven(5e9), options) is None


def test_load_result_above_one_rank_refuses_a_bundle_with_fields(tmp_path, monkeypatch):
    options = gfem.SolverOptions(order=1, linear_solver="direct")
    kept = gfem.Driven(5e9, keep_fields=True)
    line = cases.layered_line(tmp_path)
    gfem.Job(line, kept, options).run(tmp_path / "fields")
    gfem.Job(line, gfem.Driven(5e9), options).run(tmp_path / "plain")
    use_mpi(monkeypatch, fake_mpi())
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.load_result(tmp_path / "fields", comm=Stub(4))
    assert caught.value.feature == "load_result_fields"
    assert "COMM_SELF" in str(caught.value)
    assert gfem.load_result(tmp_path / "plain", comm=Stub(4)).e_fields == ()
    assert gfem.load_result(tmp_path / "fields", comm=Stub(1)).e_fields


#: One MN5 node's hyperthreads (ticket 18): CPU c and c + 112 share a core.
def _mn5_core(cpu: int) -> int:
    return cpu % 112


def _cpus(text: str) -> frozenset[int]:
    out = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        out.update(range(int(first), int(last or first) + 1))
    return frozenset(out)


#: Masks read under each launcher (ticket 18, job 48435924), and each rank's threads.
RECORDED_MASKS = [
    ("srun -c7", ["0-6", "56-62"], 7),
    ("srun -c7 --cpu-bind=cores", ["0-6,112-118", "56-62,168-174"], 7),
    ("srun --cpu-bind=none, 16 ranks", ["0-223"] * 16, 7),
    ("mpirun -n 16", ["0-55,112-167", "56-111,168-223"] * 8, 7),
    ("mpirun -n 16 --bind-to none", ["0-223"] * 16, 7),
    ("mpirun -n 8 --map-by ppr:4:socket:PE=14", ["0-13,112-125", "14-27,126-139"], 14),
    ("mpirun -n 2", ["0,112", "1,113"], 1),
    ("one core, 4 ranks on it", ["0,112"] * 4, 1),
]


@pytest.mark.parametrize(
    ("masks", "threads"),
    [row[1:] for row in RECORDED_MASKS],
    ids=[row[0] for row in RECORDED_MASKS],
)
def test_a_ranks_threads_are_its_physical_cores_over_the_ranks_sharing_them(
    masks, threads
):
    node = [_cpus(mask) for mask in masks]
    assert {_mpi.node_share(mask, node, _mn5_core) for mask in node} == {threads}


def test_the_threads_rule_reads_the_kernels_siblings_or_each_cpu_alone():
    assert _mpi.node_share(frozenset({123456}), [frozenset({123456})]) == 1


def test_an_unknown_problem_stays_a_type_error(tmp_path, monkeypatch):
    line = cases.layered_line(tmp_path)
    use_mpi(monkeypatch, fake_mpi())
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(line, object(), comm=Stub(4))


def test_a_communicator_of_size_one_runs_what_ranks_refuse(tmp_path):
    model, problem = mpi_cases.series(mpi_cases.series_mesh(tmp_path))
    options = gfem.SolverOptions(order=1, linear_solver="direct")
    with pytest.raises(gfem.DistributedUnsupportedError):
        _mpi.refuse(model, problem, options)
    top = mpi_cases.SERIES_TOP
    expected = gfem.solve(model, problem, options=options).temperature.at(top)
    result = gfem.solve(model, problem, options=options, comm=Stub(1))
    np.testing.assert_allclose(result.temperature.at(top), expected, rtol=1e-12)
    job = gfem.Job(model, problem, options).run(comm=Stub(1))
    np.testing.assert_allclose(job.temperature.at(top), expected, rtol=1e-12)


# --- agreement, trivial at one process --------------------------------------------


def test_agreement_is_trivial_at_one_process():
    assert _mpi.agree_any(None, flag=True) is True
    assert _mpi.agree_any(None, flag=False) is False
    assert _mpi.agree_all(None, flag=True) is True
    assert _mpi.agree_all(None, flag=False) is False
    values = np.array([1.0 + 2.0j, 3.0])
    assert _mpi.reduce_bcast(None, values) is values
    assert _mpi.node_bytes(None, 7) == 7
    assert _mpi.node_budget(None, 9) == 9
    assert _mpi.rank(None) == 0


def test_a_section_at_one_process_lets_any_error_through():
    with pytest.raises(ValueError, match="local"), _mpi.section(None):
        raise ValueError("local")


def test_an_error_is_agreed_by_its_class_or_its_mark():
    assert _mpi.is_agreed(gfem.SolverUnavailableError("all ranks"))
    assert not _mpi.is_agreed(ValueError("one rank"))
    assert _mpi.is_agreed(_mpi.agreed(ValueError("all ranks")))
    assert not _mpi.is_agreed(gfem.SolverUnavailableError("all ranks"), ())
    assert _mpi.is_agreed(ValueError("all ranks"), (ValueError,))


def test_raised_together_marks_only_the_classes_named():
    with pytest.raises(ValueError) as caught, _mpi.raised_together(ValueError):
        raise ValueError("replicated")
    assert _mpi.is_agreed(caught.value, ())
    with pytest.raises(KeyError) as other, _mpi.raised_together(ValueError):
        raise KeyError("local")
    assert not _mpi.is_agreed(other.value, ())


def test_a_warning_at_one_process_warns_at_the_callers_line():
    with pytest.warns(UserWarning, match="once") as caught:
        _mpi.warn(None, "once", UserWarning)
    assert caught[0].filename == __file__


# --- a kept field's communicator -------------------------------------------------


def _piece(tmp_path) -> gfem.Field:
    return gfem.Field(
        coefficients=np.zeros(3, complex),
        mesh=cases.layered_line(tmp_path).mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=2,
        support=("sub", "cover"),
        omega=1.0,
        _comm=_mpi.StaticComm(5, Stub(4)),
    )


def test_a_field_from_ranks_refuses_export_on_a_stub(tmp_path):
    piece = _piece(tmp_path)
    for method in (piece.to_meshio, piece.to_pyvista):
        with pytest.raises(gfem.DistributedUnsupportedError) as caught:
            method()
        assert caught.value.feature == "field_export"
        assert "Field.to_vtu" in str(caught.value)
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.power_flow(piece, "sub", model=cases.layered_line(tmp_path))
    assert caught.value.feature == "power_flow"
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.region_mean(piece, "sub")
    assert caught.value.feature == "region_mean"
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.electric_energy(piece, model=cases.layered_line(tmp_path))
    assert caught.value.feature == "electric_energy"


def test_a_scalar_field_from_ranks_refuses_element_mean_on_a_stub(tmp_path):
    piece = dataclasses.replace(
        _piece(tmp_path), quantity="T", unit="K", space="h1", omega=None
    )
    with pytest.raises(gfem.DistributedUnsupportedError, match="COMM_SELF") as caught:
        gfem.element_mean(piece, "sub")
    assert caught.value.feature == "element_mean"


def test_a_result_from_ranks_refuses_error_estimate_on_a_stub(tmp_path):
    model = rc.coax_model(tmp_path, 2)
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    result = gfem.solve(model, rc.COAX_PROBLEM, options=options)
    across = dataclasses.replace(
        result,
        potential={
            name: dataclasses.replace(field, _comm=_mpi.StaticComm(5, Stub(4)))
            for name, field in result.potential.items()
        },
    )
    for estimator in (
        gfem.FieldEstimator(),
        gfem.QuantityEstimator(capacitance=("inner", "inner")),
    ):
        with pytest.raises(gfem.DistributedUnsupportedError) as caught:
            gfem.error_estimate(across, estimator, model=model)
        assert caught.value.feature == "error_estimate"
        assert "COMM_SELF" in str(caught.value)
    gfem.error_estimate(result, gfem.FieldEstimator(), model=model)


def test_a_communicator_is_a_static_value_of_a_field(tmp_path):
    piece = _piece(tmp_path)
    leaves, structure = jax.tree_util.tree_flatten(piece)
    again = jax.tree_util.tree_unflatten(structure, leaves)
    assert again._communicator is piece._communicator
    other = _mpi.StaticComm(5, Stub(4))
    assert other == piece._comm
    assert hash(other) == hash(piece._comm)
    assert other != _mpi.StaticComm(6, Stub(4))


def test_a_communicator_is_not_written_with_a_field(tmp_path):
    from gfem.job import _field_args

    assert "_comm" not in _field_args(_piece(tmp_path), None)


def test_a_launch_above_the_cap_skips(monkeypatch):
    monkeypatch.setenv(ranks.MAX_RANKS, "4")
    with pytest.raises(pytest.skip.Exception, match="8 ranks"):
        ranks.on_ranks_argv(["never-run"], ranks=8)


def test_a_general_operator_is_no_distributed_refusal():
    # a general operator under ranks reaches its backend (ADR 0078 amendment)
    assert "general_operator" not in _mpi._REFUSALS
    assert "general_operator" not in mpi_cases.GUARD_KEYS


def test_each_symmetry_registers_its_own_coarse_type(monkeypatch):
    # a plain and a Bloch BDDC solve on one communicator never share a creator
    registered = {}
    monkeypatch.setattr(coarse, "_made", {})
    monkeypatch.setattr(coarse, "_registered", set())
    monkeypatch.setattr(_mpi, "comm_key", lambda _: 7)
    monkeypatch.setattr(
        coarse.ngs.comp, "RegisterPreconditioner", registered.setdefault
    )
    comm = Stub(4)
    lu, cholesky = coarse.coarse_type(comm, False), coarse.coarse_type(comm, True)
    assert (lu, cholesky) == ("gfem_mumps_7_lu", "gfem_mumps_7_cholesky")
    assert coarse.coarse_type(comm, False) == lu
    assert set(registered) == {lu, cholesky}
    made = [registered[name](None, None) for name in (lu, cholesky)]
    assert [m.symmetric for m in made] == [False, True]
    assert coarse._made[7] == made  # `finish` takes the last one NGSolve asked for


# --- field pieces -----------------------------------------------------------------


def test_each_rule_names_its_field_pieces():
    from gfem._eigen.rule import EigenmodeRule
    from gfem._engine.driven import DrivenRule
    from gfem._engine.electrostatic import ElectrostaticRule

    assert DrivenRule.pieces == {"e_field"}
    assert ElectrostaticRule.pieces == {"potential"}
    assert EigenmodeRule.pieces == {"e_field"}


def test_ranks_compare_a_pieces_cotangent_by_shape_and_any_other_by_value():
    solve_vjp = importlib.import_module("gfem._adjoint.solve_vjp")
    cotangents = {"potential": np.ones((5, 2)), "capacitance": np.eye(2)}
    replicated = solve_vjp._replicated_cotangents(cotangents, {"potential"})
    np.testing.assert_array_equal(replicated["the cotangent of potential"], [5, 2])
    np.testing.assert_array_equal(replicated["the cotangent of capacitance"], np.eye(2))
