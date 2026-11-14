"""``linear_solver="auto"`` for a driven problem: direct while it fits (ADR 0124)."""

import dataclasses
import logging
from pathlib import Path

import ngsolve as ngs
import pytest

import gfem
from gfem import _domain
from gfem._domain import nedelec_unknowns
from gfem._engine.domain import hcurl_space
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import ResolvedLinearSolver, auto_linear_solver

DATA = Path(__file__).parent / "data"
METAL = gfem.enclosed_by("metal_xy", "metal_z")
KIB = 2**10
GIB = 2**30


def test_direct_while_the_estimate_fits_the_budget():
    # 48 KiB per unknown: 1000 unknowns need 49,152,000 bytes
    kind, _, estimate = auto_linear_solver(
        "bddc_gmres", unknowns=1000, in_flight=1, general=False, budget=49_152_000
    )
    assert kind == "direct"
    assert estimate == 49_152_000
    kind, _, _ = auto_linear_solver(
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
    _, reason, _ = auto_linear_solver(
        "bddc_gmres", unknowns=340_000, in_flight=1, general=False, budget=16 * GIB
    )
    assert reason == (
        "340,000 unknowns, est. 15.6 GiB factorization <= 16.0 GiB budget "
        "(half of 32 GiB)"
    )
    _, reason, _ = auto_linear_solver(
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
