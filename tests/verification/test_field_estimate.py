"""The field estimator and mesh refinement on the square coax (a singular case).

For a capacitance with one terminal at 1 V, C_h - C = a(e, e) exactly, so the
true relative energy error of the potential is sqrt((C_h - C) / C), with C
the hard-coded reference `refinement_cases.COAX_C`. The field estimator's
effectivity (estimate over that error) measured 1.17, 1.26, 1.31 and 1.32
under uniform refinement, n = 2 to 16. Mesh refinement driven by it, run
until the next mesh would pass a quarter of the uniform n = 16 tetrahedra,
ends on 936 tetrahedra and 1574 DOF with a capacitance error of 4.5e-4,
below the uniform n = 16 error of 5.1e-4 on 9603 DOF.
"""

import math

import pytest
import refinement_cases as rc

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


def _energy_error(result: gfem.ElectrostaticResult) -> float:
    return math.sqrt(abs(rc.coax_capacitance(result) - rc.COAX_C) / rc.COAX_C)


def _dofs(result: gfem.ElectrostaticResult) -> int:
    return len(result.potential["inner"].coefficients)


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    """Uniform order-2 solves and their field estimates, n = 2, 4, 8, 16."""
    out = {}
    for n in (2, 4, 8, 16):
        model = rc.coax_model(tmp_path_factory.mktemp(f"coax{n}"), n)
        result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
        estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
        out[n] = (model, result, estimate)
    return out


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_field_estimator_effectivity_is_near_one_on_uniform_meshes(uniform, n):
    _, result, estimate = uniform[n]
    effectivity = estimate.value / _energy_error(result)
    assert 0.9 < effectivity < 1.6, effectivity


def test_mesh_refinement_reaches_the_uniform_error_with_a_fifth_of_its_dofs(uniform):
    fine_model, fine, _ = uniform[16]
    target = abs(rc.coax_capacitance(fine) - rc.COAX_C) / rc.COAX_C
    coarse, _, _ = uniform[2]
    # run until the next mesh would pass a quarter of the uniform tetrahedra
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            coarse,
            rc.COAX_PROBLEM,
            tolerance=1e-3,
            max_cycles=30,
            max_tetrahedra=len(fine_model.mesh.tetrahedra) // 4,
            options=OPTIONS,
        )
    assert ref.stop == "max_tetrahedra"
    error = abs(rc.coax_capacitance(ref.result) - rc.COAX_C) / rc.COAX_C
    assert error <= target
    assert ref.cycles[-1].dofs == _dofs(ref.result)
    assert ref.cycles[-1].dofs < _dofs(fine) / 5
