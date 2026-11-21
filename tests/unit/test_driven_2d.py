r"""The driven solve of a 2D model: its checks, its fields and its solver.

Most cases use the line tutorial's triangle mesh
(``docs/tutorials/data/line2d.msh``, mm): a line 40 mm long and :math:`w = 6`
mm wide, areas ``feed`` and ``design``, line ports ``port1`` (x = 0) and
``port2`` (x = 40), sides ``side``.
"""

import logging
import math
from pathlib import Path

import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0
from strip2d import write_strip

import gfem
from gfem._domain import h1_unknowns
from gfem._engine import cache
from gfem._engine.mesh import to_ngsolve
from gfem._linalg.base import PlannedHandle

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
ETA0 = math.sqrt(mu_0 / epsilon_0)
WIDTH = 6e-3  # m
F = 5e9


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


def line(mesh, *, design=None, boundaries=(), excluded=()):
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": design or gfem.Material()},
        boundaries=boundaries,
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
        excluded=excluded,
    )


# --- natural conditions on interior lines ------------------------------------------


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    path = write_strip(tmp_path_factory.mktemp("strip") / "strip.msh")
    return gfem.Mesh.from_gmsh(path, length_unit="um")


INTERIOR = {
    "PEC in Hz": (gfem.PEC, "Hz", r"PEC\('ab'\).*inside.*Hz.*transparent.*excluded"),
    "PMC in Ez": (gfem.PMC, "Ez", r"PMC\('ab'\).*inside.*Ez.*transparent.*Cut"),
}


@pytest.mark.parametrize("case", INTERIOR)
def test_a_natural_condition_on_an_interior_line_raises(strip, case):
    condition, polarization, message = INTERIOR[case]
    model = gfem.Model(
        strip,
        materials={name: gfem.Material() for name in "abc"},
        boundaries=(condition("ab"),),
        ports=(gfem.WavePort("p", "outer"),),
    )
    problem = gfem.Driven(1e12, polarization=polarization)
    with pytest.raises(ValueError, match=message):
        gfem.solve(model, problem)


def test_an_hz_material_field_with_a_conductivity_raises(mesh):
    n = len(mesh.elements("design"))
    model = line(
        mesh, design=gfem.MaterialField(eps_r=jnp.full(n, 2.0), conductivity=1.0)
    )
    with pytest.raises(NotImplementedError, match=r"'design'.*conductivity.*Hz"):
        gfem.solve(model, gfem.Driven(F, polarization="Hz"))


# --- the fields of a port fed 1 W/m --------------------------------------------------


@pytest.mark.parametrize(
    ("polarization", "boundaries", "amplitude"),
    [
        # E_z of a TEM line w wide carrying 1 W/m: (1/2) |E|^2 / eta0 w = 1
        ("Ez", (), math.sqrt(2 * ETA0 / WIDTH)),
        # H_z: (1/2) eta0 |H|^2 w = 1
        ("Hz", (gfem.PEC("side"),), math.sqrt(2 / (ETA0 * WIDTH))),
    ],
)
def test_a_matched_line_keeps_the_incident_field_of_1_w_per_metre(
    mesh, polarization, boundaries, amplitude
):
    problem = gfem.Driven(F, excite="p1", keep_fields=True, polarization=polarization)
    result = gfem.solve(line(mesh, boundaries=boundaries), problem)
    field = (result.h_field if polarization == "Hz" else result.e_field)(F, "p1")
    assert field.quantity == polarization
    assert field.unit == ("A/m" if polarization == "Hz" else "V/m")
    points = np.array([[x, y] for x in (5e-3, 20e-3, 35e-3) for y in (1e-3, 4e-3)])
    values = np.asarray(field.at(points))
    np.testing.assert_allclose(np.abs(values), amplitude, rtol=1e-3)


# --- the linear solver ---------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_h1_unknowns_count_the_h1_space(mesh, order):
    built = to_ngsolve(mesh)
    whole = ngs.H1(built.mesh, order=order)
    assert h1_unknowns(mesh, line(mesh)._domain, order) == whole.ndof
    feed = ngs.Compress(ngs.H1(built.mesh, order=order, definedon="feed"))
    cut = line(mesh, excluded=("design",))._domain
    assert h1_unknowns(mesh, cut, order) == feed.ndof < whole.ndof


def test_auto_sizes_a_2d_solve_by_its_h1_unknowns(mesh, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    options = gfem.SolverOptions(linear_solver="auto", order=2)
    gfem.solve(line(mesh), gfem.Driven(F, polarization="Ez"), options=options)
    (record,) = [r for r in caplog.records if hasattr(r, "unknowns")]
    assert record.linear_solver == "direct"
    assert record.unknowns == ngs.H1(to_ngsolve(mesh).mesh, order=2).ndof


# --- factorization reuse across designs ----------------------------------------------


def test_the_plan_pool_refactorizes_a_second_design(mesh, monkeypatch):
    """A second design on the same mesh keeps the first's analysis."""
    calls = []
    refactorize = PlannedHandle.refactorize

    def counted(self, *args, **kwargs):
        calls.append(1)
        return refactorize(self, *args, **kwargs)

    monkeypatch.setattr(PlannedHandle, "refactorize", counted)
    cache.ENTRIES.clear()
    cache._DISCRETIZATIONS.clear()
    n = len(mesh.elements("design"))
    options = gfem.SolverOptions(linear_solver="direct")
    problem = gfem.Driven(F, excite="p1", polarization="Ez")
    for eps in (2.0, 3.0):
        design = gfem.MaterialField(eps_r=jnp.full(n, eps))
        gfem.solve(line(mesh, design=design), problem, options=options)
        if eps == 2.0:
            first = len(calls)
    assert len(calls) > first


def test_frequency_workers_give_the_one_process_sweep(mesh):
    # PEC sides are natural in Hz: the TEM mode
    model = line(mesh, design=gfem.Material(eps_r=4.0), boundaries=(gfem.PEC("side"),))
    problem = gfem.Driven((4e9, 5e9, 6e9), excite="p1", polarization="Hz")
    one = gfem.solve(model, problem, options=gfem.SolverOptions(linear_solver="direct"))
    split = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(linear_solver="direct", frequency_workers=2),
    )
    np.testing.assert_allclose(
        np.asarray(split.s.values), np.asarray(one.s.values), rtol=0, atol=1e-12
    )
