"""Driven ``linear_solver="auto"``: direct while it fits, then the p-multigrid."""

import dataclasses
import logging
from pathlib import Path

import ngsolve as ngs
import numpy as np
import periodic_cases as pc
import pml_cases
import pytest
import structured_meshes as sm

import gfem
from gfem import _domain
from gfem._domain import nedelec_unknowns
from gfem._engine import driven
from gfem._engine.domain import hcurl_space
from gfem._engine.driven import DrivenRule
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import PmgInputs, ResolvedLinearSolver, auto_linear_solver, pmg

DATA = Path(__file__).parent / "data"
METAL = gfem.enclosed_by("metal_xy", "metal_z")
KIB = 2**10
GIB = 2**30


def test_direct_while_the_estimate_fits_the_budget():
    # 48 KiB per unknown: 1000 unknowns need 49,152,000 bytes
    kind, _, estimate, *_ = auto_linear_solver(
        "bddc_gmres", unknowns=1000, in_flight=1, general=False, budget=49_152_000
    )
    assert kind == "direct"
    assert estimate == 49_152_000
    kind, *_ = auto_linear_solver(
        "bddc_gmres", unknowns=1000, in_flight=1, general=False, budget=49_151_999
    )
    assert kind == "bddc_gmres"


def test_each_factorization_in_flight_counts():
    fits = {"unknowns": 1000, "general": False, "budget": 3 * 48 * KIB * 1000}
    assert auto_linear_solver("bddc_gmres", in_flight=3, **fits)[0] == "direct"
    assert auto_linear_solver("bddc_gmres", in_flight=4, **fits)[0] == "bddc_gmres"


def test_a_bloch_operator_counts_double():
    fits = {"unknowns": 1000, "in_flight": 1, "budget": 2 * 48 * KIB * 1000}
    assert auto_linear_solver("bddc_gmres", general=True, **fits)[0] == "direct"
    fits["budget"] -= 1
    assert auto_linear_solver("bddc_gmres", general=True, **fits)[0] == "bddc_gmres"


def test_the_reason_states_unknowns_estimate_and_budget():
    _, reason, *_ = auto_linear_solver(
        "bddc_gmres", unknowns=340_000, in_flight=1, general=False, budget=16 * GIB
    )
    assert reason == (
        "340,000 unknowns, est. 15.6 GiB factorization <= 16.0 GiB budget "
        "(half of 32 GiB)"
    )
    _, reason, *_ = auto_linear_solver(
        "bddc_gmres", unknowns=349_526, in_flight=1, general=False, budget=16 * GIB
    )
    assert "> 16.0 GiB budget" in reason


def _resolve(options=None, **keywords):
    return ResolvedLinearSolver.resolve(
        options or gfem.SolverOptions(),
        default="bddc_gmres",
        problem="driven",
        **keywords,
    )


def test_resolve_applies_the_rule_with_unknowns_on_one_cpu_process():
    small = {"unknowns": 1000, "in_flight": 1, "general": False, "budget": GIB}
    big = dict(small, budget=1000)
    assert _resolve(**small).kind == "direct"
    assert _resolve(**big).kind == "bddc_gmres"
    # without unknowns (electrostatic, eigenmode) the problem's default stands
    assert _resolve().kind == "bddc_gmres"


def test_a_distributed_solve_or_a_gpu_bypasses_the_rule():
    small = {"unknowns": 1000, "in_flight": 1, "general": False, "budget": GIB}
    assert _resolve(comm=object(), **small).kind == "bddc_gmres"
    gpu = gfem.SolverOptions(device="cuda")
    assert _resolve(gpu, **{**small, "budget": 1000}).kind == "direct"


def test_an_explicit_choice_is_kept_and_logs_nothing(caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    linear = _resolve(options, unknowns=10, in_flight=1, general=False, budget=GIB)
    assert linear.kind == "bddc_gmres"
    assert not [r for r in caplog.records if r.name == "gfem.solver"]


def test_the_choice_is_logged_once_with_its_numbers(caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    _resolve(unknowns=1000, in_flight=2, general=False, budget=GIB)
    (record,) = [r for r in caplog.records if r.name == "gfem.solver"]
    assert record.levelno == logging.INFO
    assert record.linear_solver == "direct"
    assert record.unknowns == 1000
    assert record.estimate == 2 * 48 * KIB * 1000
    assert record.budget == GIB
    assert record.getMessage().startswith(
        "linear_solver='auto' chose 'direct' for this driven problem: 1,000 unknowns"
    )


def test_a_fixed_auto_choice_is_logged_without_an_estimate(caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    _resolve()
    (record,) = [r for r in caplog.records if r.name == "gfem.solver"]
    assert record.linear_solver == "bddc_gmres"
    assert (record.unknowns, record.estimate, record.budget) == (None, None, None)
    assert record.getMessage() == (
        "linear_solver='auto' chose 'bddc_gmres' for this driven problem"
    )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_unknowns_count_the_first_kind_nedelec_space(order):
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    built = to_ngsolve(mesh)
    whole = gfem.Model(
        mesh, materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()}
    )
    plain = ngs.HCurl(built.mesh, order=order, type1=True)
    assert nedelec_unknowns(mesh, whole._domain, order) == plain.ndof
    cut = gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=(METAL,),
    )
    space = hcurl_space(cut, built, order=order)
    assert nedelec_unknowns(mesh, cut._domain, order) == space.ndof
    assert space.ndof < plain.ndof


def test_a_domain_counts_its_entities_once_and_holds_no_count(monkeypatch):
    # the domain is static pytree data compared by identity: it holds its
    # arrays only, and the count lives beside it
    assert [f.name for f in dataclasses.fields(_domain.EMDomain)] == [
        "excluded",
        "active",
        "sides",
    ]
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    model = gfem.Model(
        mesh, materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()}
    )
    counted = []
    distinct = _domain._distinct_rows

    def spy(rows):
        counted.append(1)
        return distinct(rows)

    monkeypatch.setattr(_domain, "_distinct_rows", spy)
    for order in (1, 2, 3):
        nedelec_unknowns(mesh, model._domain, order)
    assert len(counted) == 2  # edges and faces, once


# --- the p-multigrid where direct does not fit --------------------------------------


def _counted(direct, whitney, calls=None):
    """A direct-set count (`gfem._linalg.pmg.direct_set_count`) of fixed numbers."""

    def count():
        if calls is not None:
            calls.append(1)
        return pmg.DirectSetCount(direct, whitney, 0)

    return count


def _pmg(order=2, limit=GIB, direct=300, whitney=200, calls=None):
    return PmgInputs(order=order, limit=limit, count=_counted(direct, whitney, calls))


#: 1000 unknowns over a budget of 1000 bytes: direct needs 49,152,000.
OVER = {"unknowns": 1000, "in_flight": 1, "general": False, "budget": 1000}


@pytest.mark.parametrize(
    ("order", "estimate"),
    [
        # 8 KiB x 1000 + 48 KiB x (300 - 200)
        (2, 13_107_200),
        # 16 KiB x 1000 + 48 KiB x (300 - 200)
        (3, 21_299_200),
    ],
)
def test_pmg_gmres_where_direct_does_not_fit_and_its_estimate_fits_the_memory(
    order, estimate
):
    choice = auto_linear_solver("bddc_gmres", **OVER, pmg=_pmg(order, limit=estimate))
    assert choice.kind == "pmg_gmres"
    assert (choice.pmg_estimate, choice.direct_set) == (estimate, 300)
    choice = auto_linear_solver(
        "bddc_gmres", **OVER, pmg=_pmg(order, limit=estimate - 1)
    )
    assert choice.kind == "bddc_gmres"
    assert choice.pmg_estimate == estimate


def test_direct_while_it_fits_never_counts_the_direct_set():
    calls = []
    fits = dict(OVER, budget=GIB)
    choice = auto_linear_solver("bddc_gmres", **fits, pmg=_pmg(calls=calls))
    assert choice.kind == "direct"
    assert (choice.pmg_estimate, choice.direct_set) == (None, None)
    assert calls == []
    auto_linear_solver("bddc_gmres", **OVER, pmg=_pmg(calls=calls))
    assert calls == [1]


def test_frequency_workers_multiply_both_estimates():
    # two in flight: direct 98,304,000 bytes, the p-multigrid 26,214,400
    two = dict(OVER, in_flight=2, budget=98_304_000)
    assert auto_linear_solver("bddc_gmres", **two, pmg=_pmg())[0] == "direct"
    two["budget"] -= 1
    choice = auto_linear_solver("bddc_gmres", **two, pmg=_pmg(limit=26_214_400))
    assert (choice.kind, choice.pmg_estimate) == ("pmg_gmres", 26_214_400)
    choice = auto_linear_solver("bddc_gmres", **two, pmg=_pmg(limit=26_214_399))
    assert choice.kind == "bddc_gmres"


def test_a_layer_keeps_direct_where_neither_estimate_fits():
    layer = "the layer excludes 'bddc_gmres'"
    choice = auto_linear_solver("direct", **OVER, pmg=_pmg(limit=1), reason=layer)
    assert choice.kind == "direct"
    assert choice.reason.endswith(f"GiB memory; {layer}")
    assert "(direct set 30.0 %) > " in choice.reason
    choice = auto_linear_solver("direct", **OVER, pmg=_pmg(), reason=layer)
    assert choice.kind == "pmg_gmres"
    assert layer not in choice.reason


def test_the_reason_states_the_p_multigrid_estimate_and_the_memory():
    big = {"unknowns": 1_000_000, "in_flight": 1, "general": False, "budget": 16 * GIB}
    pmg_inputs = _pmg(limit=32 * GIB, direct=200_000, whitney=100_000)
    choice = auto_linear_solver("bddc_gmres", **big, pmg=pmg_inputs)
    assert choice.kind == "pmg_gmres"
    assert choice.reason == (
        "1,000,000 unknowns, est. 45.8 GiB factorization > 16.0 GiB budget "
        "(half of 32 GiB); p-multigrid est. 12.2 GiB (direct set 20.0 %) "
        "<= 32 GiB memory"
    )


def test_the_record_carries_the_p_multigrid_estimate(caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    linear = _resolve(**OVER, pmg=_pmg())
    assert linear.kind == "pmg_gmres"
    (record,) = [r for r in caplog.records if r.name == "gfem.solver"]
    assert (record.pmg_estimate, record.direct_set) == (13_107_200, 300)
    assert record.estimate == 49_152_000
    assert "p-multigrid est." in record.getMessage()
    caplog.clear()
    _resolve(**dict(OVER, budget=GIB), pmg=_pmg())
    (record,) = [r for r in caplog.records if r.name == "gfem.solver"]
    assert (record.pmg_estimate, record.direct_set) == (None, None)


# --- the driven rule ------------------------------------------------------------------

#: The strip line of the p-multigrid tests, at 50 MHz: about 0.003 wavelengths.
LINE_F = 50e6


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _over_budget(monkeypatch, memory=None):
    """Direct's budget below every estimate; ``memory`` bytes, if given."""
    monkeypatch.setattr(driven, "default_retain_memory", lambda: 1)
    if memory is not None:
        monkeypatch.setattr(driven, "memory_limit", lambda: memory)


def test_auto_picks_pmg_gmres_for_a_small_metal_line_past_the_budget(line, monkeypatch):
    problem = gfem.Driven(frequency=LINE_F)
    direct = gfem.solve(
        line, problem, options=gfem.SolverOptions(linear_solver="direct")
    )
    _over_budget(monkeypatch)
    assert DrivenRule(line, problem, gfem.SolverOptions()).linear.kind == "pmg_gmres"
    auto = gfem.solve(line, problem, options=gfem.SolverOptions())
    # the p-multigrid's GMRES ran: a few iterations per excited port
    (its,) = auto.sweep.iterations
    assert len(its) == 2
    assert all(0 < n <= 10 for n in its)
    np.testing.assert_allclose(
        np.asarray(auto.s.values), np.asarray(direct.s.values), rtol=0, atol=1e-6
    )


def test_auto_takes_bddc_gmres_where_neither_estimate_fits(line, monkeypatch):
    _over_budget(monkeypatch, memory=1)
    rule = DrivenRule(line, gfem.Driven(LINE_F), gfem.SolverOptions())
    assert rule.linear.kind == "bddc_gmres"


def _layer(directory):
    mesh = gfem.Mesh.from_gmsh(pml_cases.tem_line(directory / "pml.msh", cells=6))
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "pml": gfem.Material()},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


@pytest.mark.parametrize(("memory", "kind"), [(None, "pmg_gmres"), (1, "direct")])
def test_a_layer_past_the_budget_takes_pmg_gmres_where_it_fits(
    tmp_path, monkeypatch, caplog, memory, kind
):
    _over_budget(monkeypatch, memory)
    caplog.set_level(logging.INFO, logger="gfem.solver")
    problem = gfem.Driven(frequency=pml_cases.F0)
    rule = DrivenRule(_layer(tmp_path), problem, gfem.SolverOptions())
    assert rule.linear.kind == kind
    (record,) = [r for r in caplog.records if hasattr(r, "pmg_estimate")]
    assert record.linear_solver == kind
    assert record.pmg_estimate > 0
    # the layer's cells join the direct set: more than its Whitney DOFs
    assert record.direct_set > 0
    assert ("perfectly matched layer" in record.getMessage()) == (kind == "direct")


def _order_1(_):
    return None, gfem.Driven(LINE_F), {"order": 1}


def _periodic(directory):
    return pc.c1(pc.c1_mesh(directory)), gfem.Driven(pc.C1_FREQUENCY), {}


def _bloch(directory):
    problem = gfem.Driven(pc.C1_FREQUENCY, wave_vector=pc.C1_K)
    return pc.c1(pc.c1_mesh(directory)), problem, {}


@pytest.mark.parametrize("case", [_order_1, _periodic, _bloch])
def test_a_model_pmg_gmres_does_not_solve_keeps_bddc_gmres(
    line, tmp_path, monkeypatch, caplog, case
):
    model, problem, options = case(tmp_path)
    _over_budget(monkeypatch)
    caplog.set_level(logging.INFO, logger="gfem.solver")
    rule = DrivenRule(model or line, problem, gfem.SolverOptions(**options))
    assert rule.linear.kind == "bddc_gmres"
    (record,) = [r for r in caplog.records if hasattr(r, "pmg_estimate")]
    assert (record.pmg_estimate, record.direct_set) == (None, None)


def test_tree_cotree_keeps_the_refusal_of_auto_s_iterative_choice(line, monkeypatch):
    _over_budget(monkeypatch)
    options = gfem.SolverOptions(gauge="tree_cotree")
    with pytest.raises(ValueError, match="linear_solver='auto' chose 'bddc_gmres'"):
        DrivenRule(line, gfem.Driven(LINE_F), options)
