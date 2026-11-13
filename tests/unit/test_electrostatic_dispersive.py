"""The electrostatic problem on a dispersive material: its static permittivity.

A law region solves with eps(0) = eps_inf + the strengths of its Lorentz,
Debye and Djordjevic-Sarkar terms (ADR 0111): exactly as a constant
material set to ``law(0)``. A Drude term has no static permittivity and
raises by name. The finite-difference cell of the law's parameters is S1 of
the gradient matrix (tests/verification/test_gradient_matrix.py).

The stack is a 4 x 3 mm plate of two layers, oxide under nitride; the
nitride holds the law.
"""

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem

GHZ = 1e9
LAYERS = (("oxide", 0.5), ("nitride", 0.25))
PROBLEM = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
OPTIONS = gfem.SolverOptions(linear_solver="direct")

#: A dielectric law with no Drude term: law(0) is not eps_inf + the strengths
#: to the last bit (the Lorentz and Djordjevic-Sarkar forms round at f = 0).
LAW = gfem.PermittivityLaw(
    eps_inf=2.3,
    terms=[
        gfem.Lorentz(delta_eps=1.7, resonance_frequency=37 * GHZ, damping=2 * GHZ),
        gfem.Debye(delta_eps=0.9, relaxation_frequency=3 * GHZ),
        gfem.DjordjevicSarkar(delta_eps=1.3, low_frequency=2e3, high_frequency=3e11),
    ],
)
DRUDE = gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (2, 2, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=LAW),
        },
    )


def _static(law):
    return float(np.real(complex(law(0))))


def test_a_law_solves_as_a_material_at_its_static_permittivity(stack):
    reference = stack.with_material("nitride", gfem.Material(eps_r=_static(LAW)))
    got = gfem.solve(stack, PROBLEM, options=OPTIONS)
    want = gfem.solve(reference, PROBLEM, options=OPTIONS)
    # bit for bit: the same coefficient reaches the same assembly
    assert np.array_equal(got.capacitance, want.capacitance)
    for region in ("oxide", "nitride"):
        assert np.array_equal(got.energy[region], want.energy[region])
    assert np.array_equal(
        got.potential["top"].coefficients, want.potential["top"].coefficients
    )


def test_a_drude_term_raises_by_name_before_any_solve(stack, monkeypatch):
    law = gfem.PermittivityLaw(eps_inf=1.0, terms=[DRUDE, *LAW.terms])
    gold = stack.with_material("nitride", gfem.Material(eps_r=law, name="gold"))
    from gfem._engine import electrostatic

    def no_solve(*_args, **_kwargs):
        pytest.fail("the refusal comes before any assembly")

    monkeypatch.setattr(electrostatic.ElectrostaticRule, "forward", no_solve)
    with pytest.raises(ValueError, match=r"'gold'.*has a Drude term.*no static"):
        gfem.solve(gold, PROBLEM, options=OPTIONS)


def test_the_error_estimate_weights_a_law_with_its_static_permittivity(stack):
    # a random oxide makes the field three-dimensional, so the estimate is not 0
    n = len(stack.mesh.elements("oxide"))
    eps = np.random.default_rng(0).uniform(2.0, 9.0, n)
    stack = stack.with_material("oxide", gfem.MaterialField(eps_r=eps))
    reference = stack.with_material("nitride", gfem.Material(eps_r=_static(LAW)))
    estimates = [
        gfem.error_estimate(
            gfem.solve(model, PROBLEM, options=OPTIONS),
            gfem.FieldEstimator(),
            model=model,
        )
        for model in (stack, reference)
    ]
    got, want = estimates
    assert got.value > 1e-3
    # the same weights; the recovery itself repeats only to rounding
    np.testing.assert_allclose(got.value, want.value, rtol=1e-12)
    np.testing.assert_allclose(got.indicators, want.indicators, rtol=1e-12)


def _capacitance(model):
    return gfem.solve(model, PROBLEM, options=OPTIONS).capacitance[0, 0] * 1e12


def _by_law(stack):
    def capacitance(law):
        return _capacitance(stack.with_material("nitride", gfem.Material(eps_r=law)))

    return capacitance


def test_eps_inf_and_each_strength_differentiate_as_the_static_permittivity(stack):
    # eps(0) is eps_inf + the strengths: each one's gradient is the gradient by
    # a uniform permittivity of the region, summed over its elements
    n = len(stack.mesh.elements("nitride"))

    def by_field(eps):
        field = gfem.MaterialField(eps_r=eps)
        return _capacitance(stack.with_material("nitride", field))

    want = float(jax.grad(by_field)(np.full(n, _static(LAW))).sum())
    grad = jax.grad(_by_law(stack))(LAW)
    assert want > 0
    np.testing.assert_allclose(grad.eps_inf, want, rtol=1e-10)
    for term in grad.terms:
        np.testing.assert_allclose(term.delta_eps, want, rtol=1e-10)


def test_the_frequency_parameters_are_an_exact_zero(stack):
    # the static permittivity never reads them (ADR 0011's amendment)
    grad = jax.grad(_by_law(stack))(LAW)
    lorentz, debye, sarkar = grad.terms
    assert lorentz.resonance_frequency == 0
    assert lorentz.damping == 0
    assert debye.relaxation_frequency == 0
    assert sarkar.low_frequency == 0
    assert sarkar.high_frequency == 0
