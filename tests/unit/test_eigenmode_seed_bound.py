"""Every eigenmode seed solve stops after a bounded number of restarts (ADR 0122).

A lossy region whose modes lie closer together than their linewidth (a
modal overlap M of about 100 or more) keeps the shift-invert Arnoldi from
telling the nearest ones apart: without a bound, ARPACK's default of 10 n
restarts ran for days. The seed solve now stops at ``_SEED_RESTARTS``
restarts, ends the attempt ladder, and the solve raises
`EigenmodeConvergenceError` naming the seed solve and the crowded region.

The crowded case is ticket 14's constant twin: a 10 x 8 x 5 mm PEC plate
filled with eps = 22.3 (1 - 0.448 i) at 24.5 GHz (M about 260 over the
plate), with the bound patched to 5 so that it raises in seconds. The law
case of ticket 14 (a Lorentz fill one damping below its resonance) joins
these tests with the permittivity laws.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import structured_meshes as sm
from eigen_cases import line_mesh, terminated_line

import gfem
from gfem._eigen import arpack, seed
from gfem._eigen import rule as eigen_rule
from gfem._eigen.operator import EigenOperator

BOUND = 5  # the patched restart bound
CROWDED = gfem.Eigenmode(target=24.5e9, n_modes=4)
ORDER_1 = gfem.SolverOptions(order=1)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(
        path, sm.plate(10.0, 8.0, (("host", 3.0), ("core", 2.0)), (13, 11, 5))
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


def crowded(mesh):
    """The plate filled with eps 22.3 - 10i: about 260 modes within a linewidth."""
    fill = gfem.Material(eps_r=22.3, tan_delta=0.448)
    return gfem.Model(
        mesh,
        materials={"host": fill, "core": fill},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )


@pytest.fixture
def seed_solves(monkeypatch):
    """``(nev, ncv, applies)`` per seed solve, recorded even when it raises."""
    solves = [0]
    solve = EigenOperator.solve

    def counted(self, b):
        solves[0] += 1
        return solve(self, b)

    rule = eigen_rule.EigenmodeRule
    shift_invert = rule._shift_invert
    seen = []

    def spy(self, op, project, nev, **kwargs):
        before = solves[0]
        try:
            return shift_invert(self, op, project, nev, **kwargs)
        finally:
            dims = seed.seed_operator(op, project).dimensions(nev)
            seen.append((*dims, solves[0] - before))

    monkeypatch.setattr(EigenOperator, "solve", counted)
    monkeypatch.setattr(rule, "_shift_invert", spy)
    return seen


def test_the_seed_restart_bound_is_100():
    assert seed._SEED_RESTARTS == 100


def test_arpack_seed_solves_run_at_most_the_restart_bound(line, monkeypatch):
    eigs, maxiter = spla.eigs, []

    def spy(*args, **kwargs):
        maxiter.append(kwargs.get("maxiter"))
        return eigs(*args, **kwargs)

    monkeypatch.setattr(arpack.spla, "eigs", spy)
    monkeypatch.setattr(seed, "_SEED_RESTARTS", 37)
    gfem.solve(terminated_line(line, 1000.0), gfem.Eigenmode(8e9))
    assert maxiter == [37]


def test_a_crowded_seed_solve_stops_at_its_bound_and_raises(
    plate, monkeypatch, seed_solves
):
    monkeypatch.setattr(seed, "_SEED_RESTARTS", BOUND)
    with pytest.raises(gfem.EigenmodeConvergenceError) as raised:
        gfem.solve(crowded(plate), CROWDED, options=ORDER_1)
    message = str(raised.value)
    assert f"did not converge in {BOUND} restarts" in message
    # the worst region: the 3 mm host holds about 155 of the plate's 260
    assert "Region 'host' holds about 155 modes" in message
    assert "eps_max_it" in message
    # one seed solve: the ladder does not retry a solve that hit its bound;
    # ncv applies build the first subspace, ncv - nev each restart (81
    # measured: one more than that)
    ((nev, ncv, applies),) = seed_solves
    assert applies <= ncv + 1 + BOUND * (ncv - nev)
    assert f"({applies} shift solves)" in message


@pytest.mark.slepc
def test_a_crowded_slepc_seed_solve_stops_at_its_bound_and_raises(
    plate, monkeypatch, seed_solves
):
    monkeypatch.setattr(seed, "_SEED_RESTARTS", BOUND)
    options = gfem.SolverOptions(order=1, eigen_solver="slepc")
    with pytest.raises(gfem.EigenmodeConvergenceError, match="did not converge in 5"):
        gfem.solve(crowded(plate), CROWDED, options=options)
    ((nev, ncv, applies),) = seed_solves
    assert applies <= ncv + 1 + BOUND * (ncv - nev)


@pytest.fixture
def max_its(monkeypatch):
    """SLEPc's ``max_it`` at each EPS solve."""
    from gfem import _petsc

    SLEPc = _petsc.require_slepc()
    seen = []

    class EPS(SLEPc.EPS):
        def solve(self):
            seen.append(self.getTolerances()[1])
            super().solve()

    monkeypatch.setattr(SLEPc, "EPS", EPS)
    return seen


@pytest.mark.slepc
@pytest.mark.parametrize(
    ("petsc_options", "expected"), [({}, 100), ({"eps_max_it": 7}, 7)]
)
def test_slepc_seed_solves_get_the_bound_unless_eps_max_it(
    line, max_its, petsc_options, expected
):
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options=petsc_options)
    gfem.solve(terminated_line(line, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert max_its == [expected]


def test_a_seed_solve_well_inside_its_bound_is_unchanged(line, seed_solves):
    # the bound is never reached by a good case: one seed solve, few applies
    result = gfem.solve(terminated_line(line, 1000.0), gfem.Eigenmode(8e9, n_modes=2))
    assert np.all(np.isfinite(result.omega))
    assert len(seed_solves) == 1


#: Ticket 14's law case: a Lorentz fill (eps_inf 2, delta 1, f0 25 GHz,
#: gamma 0.5 GHz) one damping below its resonance, eps 22.3 - 10.0i there:
#: below the unresolved band (kh 3.02), in the same crowd as the twin.
CROWDED_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=25e9, damping=0.5e9)],
)


def test_a_crowded_law_seed_solve_stops_at_its_bound_and_names_the_term(
    plate, monkeypatch, seed_solves
):
    monkeypatch.setattr(seed, "_SEED_RESTARTS", BOUND)
    fill = gfem.Material(eps_r=CROWDED_LAW)
    model = crowded(plate).replace(materials={"host": fill, "core": fill})
    with pytest.raises(gfem.EigenmodeConvergenceError) as raised:
        gfem.solve(model, CROWDED, options=ORDER_1)
    message = str(raised.value)
    assert f"did not converge in {BOUND} restarts" in message
    assert "Region 'host' holds about 15" in message
    assert "law.terms[0] (Lorentz, resonance 25 GHz)" in message
    ((nev, ncv, applies),) = seed_solves
    assert applies <= ncv + 1 + BOUND * (ncv - nev)


def test_a_constant_regions_overlap_names_no_law_term(plate, monkeypatch):
    monkeypatch.setattr(seed, "_SEED_RESTARTS", BOUND)
    with pytest.raises(gfem.EigenmodeConvergenceError) as raised:
        gfem.solve(crowded(plate), CROWDED, options=ORDER_1)
    assert "law.terms" not in str(raised.value)


def test_a_scalar_permittivity_with_principal_loss_tangents_names_its_overlap(
    plate, monkeypatch
):
    # Material admits a scalar eps_r beside three principal loss tangents
    monkeypatch.setattr(seed, "_SEED_RESTARTS", BOUND)
    fill = gfem.Material(eps_r=22.3, tan_delta=(0.448, 0.448, 0.46))
    model = crowded(plate).replace(materials={"host": fill, "core": fill})
    with pytest.raises(gfem.EigenmodeConvergenceError) as raised:
        gfem.solve(model, CROWDED, options=ORDER_1)
    assert "Region 'host' holds about" in str(raised.value)
