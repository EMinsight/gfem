"""`SolverOptions.gauge`: its resolution, refusals, warnings and logs.

``"auto"`` gauges a one-process CPU direct driven solve without periodic
pairs, a Bloch wave vector or Floquet ports and runs every other one
ungauged; ``"tree_cotree"`` refuses where the solve would run ungauged;
``"none"`` is the ungauged path. A gauged solve does not warn below the
low-frequency floor unless the model has wave ports, whose mode problem is
not gauged; an ungauged one says why it is.
"""

import logging
import math
import warnings
from pathlib import Path

import numpy as np
import periodic_cases as pc
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._engine import driven, driven_sweep, driven_system, gauge, workers
from gfem._engine.driven import DrivenRule

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
DATA = Path(__file__).parent / "data"
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")


def _strip(directory, *, periodic=False):
    edges = tuple(
        np.linspace(0, s, n + 1) for s, n in zip((L, W, H), (8, 2, 1), strict=True)
    )
    path = sm.write_msh22(directory / "line.msh", sm.periodic_cell(edges))
    boundaries = [gfem.PEC("z_min"), gfem.PEC("z_max")]
    if periodic:
        boundaries.append(gfem.Periodic(("y_min", "y_max")))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"cell": gfem.Material(eps_r=EPS)},
        boundaries=boundaries,
        ports=[
            gfem.LumpedPort("p1", "x_min", "+z"),
            gfem.LumpedPort("p2", "x_max", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return _strip(tmp_path_factory.mktemp("line"))


@pytest.fixture(scope="module")
def plates():
    """A parallel-plate line between two wave ports, in µm."""
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def _below(model, k0h=5e-7):
    """A frequency (Hz) at which ``model``'s k0 h_min is ``k0h``."""
    floor = driven.LowFrequencyFloor.of(model)
    return k0h * const.c / (2 * math.pi * floor.h_min)


def _gauge_records(caplog):
    return [r for r in caplog.records if hasattr(r, "gauge")]


def _low_frequency(record):
    return [w for w in record if w.category is gfem.LowFrequencyWarning]


def _rule(model, problem, options):
    """The rule, and the low-frequency warnings its construction gave."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rule = DrivenRule(model, problem, options)
    return rule, _low_frequency(record)


# --- the option -----------------------------------------------------------------


def test_the_gauge_defaults_to_auto():
    assert gfem.SolverOptions().gauge == "auto"


def test_an_unknown_gauge_names_the_choices():
    with pytest.raises(ValueError, match="'auto', 'tree_cotree', 'none'"):
        gfem.SolverOptions(gauge="coulomb")


# --- resolution and its log -----------------------------------------------------


def test_the_default_gauges_a_direct_line_solve_and_logs_it(line, caplog):
    with caplog.at_level(logging.INFO, logger="gfem.solver"):
        rule, _ = _rule(line, gfem.Driven(1e9), DIRECT)
    assert rule.gauge.on
    (record,) = _gauge_records(caplog)
    assert record.gauge == "tree_cotree"
    assert record.reason is None
    assert "tree-cotree gauge on" in record.getMessage()


def test_auto_runs_bddc_gmres_ungauged_and_logs_why(line, caplog):
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    with caplog.at_level(logging.INFO, logger="gfem.solver"):
        rule, _ = _rule(line, gfem.Driven(1e9), options)
    assert not rule.gauge.on
    (record,) = _gauge_records(caplog)
    assert record.gauge is None
    assert record.reason == "linear_solver='bddc_gmres'"
    assert "tree-cotree gauge off (linear_solver='bddc_gmres')" in record.getMessage()


def test_auto_runs_a_periodic_model_ungauged(tmp_path):
    rule, _ = _rule(_strip(tmp_path, periodic=True), gfem.Driven(1e9), DIRECT)
    assert not rule.gauge.on
    assert rule.gauge.reason == "a periodic pair"


def test_a_gauge_basis_build_is_logged_once(tmp_path, caplog):
    model = _strip(tmp_path)  # a topology of its own: the basis is built here
    with caplog.at_level(logging.INFO, logger="gfem.solver"):
        gfem.solve(model, gfem.Driven([1e9, 2e9]), options=DIRECT)
        gfem.solve(model, gfem.Driven(3e9), options=DIRECT)
    (record,) = [r for r in caplog.records if hasattr(r, "floating")]
    assert record.y_unknowns >= record.v_unknowns
    assert record.conductors == 2
    assert record.floating == 1
    assert record.nnz > 0
    assert record.ungauged_nnz > 0
    assert record.basis_seconds >= 0
    assert record.template_seconds >= 0


def test_none_solves_on_the_ungauged_path(line, monkeypatch):
    def refused(*_args, **_kwargs):
        raise AssertionError("gauge='none' built a gauge basis")

    monkeypatch.setattr(driven_system, "gauge_basis", refused)
    options = gfem.SolverOptions(order=2, linear_solver="direct", gauge="none")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.LowFrequencyWarning)
        s = gfem.solve(line, gfem.Driven(1e4), options=options).s.values
    assert np.all(np.isfinite(np.asarray(s)))


def test_auto_and_none_differ_only_in_round_off_above_the_floor(line):
    problem = gfem.Driven([1e9, 5e9])
    gauged = gfem.solve(line, problem, options=DIRECT).s.values
    options = gfem.SolverOptions(order=2, linear_solver="direct", gauge="none")
    ungauged = gfem.solve(line, problem, options=options).s.values
    assert np.abs(np.asarray(gauged) - np.asarray(ungauged)).max() <= 1e-9


def test_a_model_whose_vertices_are_all_on_one_conductor_solves_gauged(tmp_path):
    # one cell across z and PEC side walls: every vertex is on one conductor,
    # so the spanning forest has no edge
    model = _strip(tmp_path)
    walls = [gfem.PEC(s) for s in ("z_min", "z_max", "y_min", "y_max")]
    model = model.replace(boundaries=walls)
    problem = gfem.Driven(1e9)
    gauged = gfem.solve(model, problem, options=DIRECT).s.values
    options = gfem.SolverOptions(order=2, linear_solver="direct", gauge="none")
    ungauged = gfem.solve(model, problem, options=options).s.values
    assert np.abs(np.asarray(gauged) - np.asarray(ungauged)).max() <= 1e-9


# --- refusals of an explicit gauge ----------------------------------------------


@pytest.mark.parametrize(
    ("linear_solver", "text"),
    [
        (
            "bddc_gmres",
            "gauge='tree_cotree' needs linear_solver='direct', not 'bddc_gmres'",
        ),
        pytest.param(
            "petsc",
            "gauge='tree_cotree' needs linear_solver='direct', not 'petsc'",
            marks=pytest.mark.petsc,
        ),
    ],
)
def test_tree_cotree_refuses_another_linear_solver(line, linear_solver, text):
    options = gfem.SolverOptions(linear_solver=linear_solver, gauge="tree_cotree")
    with pytest.raises(ValueError, match=text) as err:
        DrivenRule(line, gfem.Driven(1e9), options)
    assert "gauge='auto'" in str(err.value)


def test_tree_cotree_refuses_auto_s_iterative_choice_with_its_numbers(
    line, monkeypatch
):
    monkeypatch.setattr(driven, "default_retain_memory", lambda: 1)
    options = gfem.SolverOptions(gauge="tree_cotree")
    with pytest.raises(ValueError, match="linear_solver='auto' chose 'bddc_gmres'"):
        DrivenRule(line, gfem.Driven(1e9), options)


def test_tree_cotree_names_the_estimate_and_the_remedy(line, monkeypatch):
    monkeypatch.setattr(driven, "default_retain_memory", lambda: 1)
    options = gfem.SolverOptions(gauge="tree_cotree")
    with pytest.raises(ValueError, match="GiB budget") as err:
        DrivenRule(line, gfem.Driven(1e9), options)
    assert "linear_solver='direct'" in str(err.value)


def test_tree_cotree_refuses_a_gpu(line):
    options = gfem.SolverOptions(device="cuda", gauge="tree_cotree")
    with pytest.raises(ValueError, match="device='cuda'"):
        DrivenRule(line, gfem.Driven(1e9), options)


def test_tree_cotree_refuses_a_periodic_model(tmp_path):
    options = gfem.SolverOptions(linear_solver="direct", gauge="tree_cotree")
    with pytest.raises(ValueError, match="periodic pair"):
        DrivenRule(_strip(tmp_path, periodic=True), gfem.Driven(1e9), options)


def test_tree_cotree_refuses_a_bloch_wave_vector(tmp_path):
    options = gfem.SolverOptions(linear_solver="direct", gauge="tree_cotree")
    problem = gfem.Driven(1e9, wave_vector=(0.0, 10.0, 0.0))
    with pytest.raises(ValueError, match="Bloch wave vector"):
        DrivenRule(_strip(tmp_path, periodic=True), problem, options)


def test_tree_cotree_refuses_floquet_ports(tmp_path):
    model = pc.f1(pc.f1_mesh(tmp_path))
    options = gfem.SolverOptions(linear_solver="direct", gauge="tree_cotree")
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE"])
    with pytest.raises(ValueError, match="Floquet ports"):
        DrivenRule(model, problem, options)


def test_tree_cotree_refuses_ranks():
    with pytest.raises(gfem.DistributedUnsupportedError, match="one process"):
        gauge.resolve_gauge(
            "tree_cotree",
            linear_solver="direct",
            kind="direct",
            device="cpu",
            distributed=True,
        )


def test_tree_cotree_refuses_ranks_before_the_solver_s_stack(line, monkeypatch):
    from gfem import _petsc

    def missing(*_):
        raise gfem.SolverUnavailableError("no MPI PETSc stack")

    monkeypatch.setattr(_petsc, "require_mpi_petsc", missing)
    options = gfem.SolverOptions(linear_solver="direct", gauge="tree_cotree")
    with pytest.raises(gfem.DistributedUnsupportedError, match="one process"):
        DrivenRule(line, gfem.Driven(1e9), options, comm=object())


def test_a_refusal_lists_every_reason(tmp_path):
    options = gfem.SolverOptions(device="cuda", gauge="tree_cotree")
    with pytest.raises(ValueError, match="device='cuda'") as err:
        DrivenRule(_strip(tmp_path, periodic=True), gfem.Driven(1e9), options)
    assert "periodic pair" in str(err.value)


def test_a_refusal_across_ranks_lists_every_reason():
    with pytest.raises(gfem.DistributedUnsupportedError, match="one process") as err:
        gauge.resolve_gauge(
            "tree_cotree",
            linear_solver="bddc_gmres",
            kind="bddc_gmres",
            device="cpu",
            distributed=True,
            periodic=True,
        )
    assert "linear_solver='direct'" in str(err.value)
    assert "periodic pair" in str(err.value)


def test_the_model_s_own_refusal_comes_before_the_gauge_s(tmp_path):
    crystal = gfem.Material(eps_r=(2.0, 3.6, 2.5))
    model = _strip(tmp_path)
    model = gfem.Model(
        model.mesh,
        materials={"cell": crystal},
        boundaries=[*model.boundaries, gfem.Absorbing("y_min", order=2)],
        ports=model.ports,
    )
    options = gfem.SolverOptions(device="cuda", gauge="tree_cotree")
    with pytest.raises(NotImplementedError, match="order 2 needs an isotropic"):
        DrivenRule(model, gfem.Driven(1e9), options)


def test_tree_cotree_is_honoured_in_scope(line):
    options = gfem.SolverOptions(linear_solver="direct", gauge="tree_cotree")
    rule, _ = _rule(line, gfem.Driven(1e9), options)
    assert rule.gauge.on


def test_other_problem_types_ignore_the_gauge(line):
    options = gfem.SolverOptions(linear_solver="bddc_gmres", gauge="tree_cotree")
    problem = gfem.Electrostatic(terminals={"top": "z_max"}, ground="z_min")
    gfem.solve(line.replace(ports=[]), problem, options=options)


# --- the low-frequency warning --------------------------------------------------


def test_a_gauged_lumped_port_solve_below_the_floor_does_not_warn(line):
    rule, low = _rule(line, gfem.Driven(_below(line)), DIRECT)
    assert rule.gauge.on
    assert low == []


def test_a_gauged_wave_port_solve_below_the_floor_warns_about_the_port_modes(plates):
    rule, (w,) = _rule(plates, gfem.Driven(_below(plates)), DIRECT)
    assert rule.gauge.on
    message = str(w.message)
    assert "1 of 1 frequencies" in message
    assert "The volume solve is gauged and keeps its digits" in message
    assert "wave ports' mode solve is not" in message
    assert "gfem.PortModeError" in message
    assert "Use lumped ports at these frequencies" in message
    assert "gfem.Electrostatic" not in message


@pytest.mark.parametrize(
    ("options", "clause"),
    [
        (
            gfem.SolverOptions(linear_solver="bddc_gmres"),
            (
                "The solve is ungauged (linear_solver='bddc_gmres'): "
                "linear_solver='direct' keeps these digits."
            ),
        ),
        (
            gfem.SolverOptions(linear_solver="direct", gauge="none"),
            "The solve is ungauged (gauge='none'): gauge='auto' keeps these digits.",
        ),
    ],
)
def test_an_ungauged_solve_below_the_floor_says_why(line, options, clause):
    _, (w,) = _rule(line, gfem.Driven(_below(line)), options)
    message = str(w.message)
    assert message.endswith(clause)
    assert "gfem.Electrostatic" in message


def test_a_solve_that_cannot_be_gauged_says_so(tmp_path):
    model = _strip(tmp_path, periodic=True)
    _, (w,) = _rule(model, gfem.Driven(_below(model)), DIRECT)
    assert str(w.message).endswith("This solve cannot be gauged (a periodic pair).")


def test_none_out_of_scope_keeps_the_ungauged_message(tmp_path):
    model = _strip(tmp_path, periodic=True)
    options = gfem.SolverOptions(linear_solver="direct", gauge="none")
    _, (w,) = _rule(model, gfem.Driven(_below(model)), options)
    assert str(w.message).endswith("coarsen the smallest elements.")


# --- frequency workers and mesh refinement --------------------------------------


def test_a_frequency_worker_takes_its_parent_s_gauge(line, caplog):
    off = gauge.GaugeChoice(on=False, reason="gauge='none'")
    problem = gfem.Driven(1e9)
    parts = workers.model_parts(line)
    with caplog.at_level(logging.INFO, logger="gfem.solver"):
        driven_sweep.solve_block(*parts, problem, DIRECT, False, gauge=off)
    # taken as it is: not resolved (and logged) again, no basis built
    assert _gauge_records(caplog) == []
    assert [r for r in caplog.records if hasattr(r, "floating")] == []


def test_a_gauged_refinement_does_not_warn_below_the_floor(line):
    f = _below(line)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gfem.refine_mesh(line, gfem.Driven(f), max_cycles=1, options=DIRECT)
    assert _low_frequency(record) == []


def test_an_ungauged_refinement_warns_once_with_its_reason(line):
    f = _below(line)
    options = gfem.SolverOptions(order=2, linear_solver="direct", gauge="none")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gfem.refine_mesh(line, gfem.Driven(f), max_cycles=1, options=options)
    (w,) = _low_frequency(record)
    assert str(w.message).endswith("gauge='auto' keeps these digits.")
