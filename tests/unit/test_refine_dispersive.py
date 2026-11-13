"""Error estimates and mesh refinement of a model with a permittivity law.

The estimators read a law region at the solve frequency (ADR 0111): the
field and S estimates of a law region equal those of the constant material
holding the law's value there. The eigen omega estimate refuses a law, and
so does every estimator where eps(omega) = 0 (the flux recovery has no
weight there). The metal case's effectivity is verified in
`tests/verification/test_field_estimate.py`.

The line is a 10 x 2 x 1 mm parallel-plate line with lumped ports at both
ends; its 2 mm "core" (x in [4, 6]) holds the law.
"""

import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

import gfem
from gfem.refinement import _lossy_elements

GHZ = 1e9
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
EPS_HOST = 4.0

#: Lossless: real, but not eps_inf (3 + 2 x 64 / (64 - 9) at 3 GHz).
LOSSLESS = gfem.PermittivityLaw(
    eps_inf=3.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=8 * GHZ, damping=0.0)],
)
#: Lossy: a Debye relaxation.
LOSSY = gfem.PermittivityLaw(
    eps_inf=2.0, terms=[gfem.Debye(delta_eps=3.0, relaxation_frequency=5 * GHZ)]
)
#: A lossless Drude metal, eps = 1 - (5 GHz / f)^2: zero at 5 GHz exactly.
PLASMA = gfem.PermittivityLaw(terms=[gfem.Drude(plasma_frequency=5 * GHZ, damping=0.0)])


@pytest.fixture
def no_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement._solve", solve)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, core):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS_HOST), "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=30.0),
        ],
    )


def _constant(law, f):
    """The constant `Material` holding ``law``'s value at ``f`` (Re eps > 0)."""
    eps = complex(law(f))
    return gfem.Material(
        eps_r=eps.real, tan_delta=-eps.imag / eps.real, name=f"at {f:g}"
    )


def _estimates(model, f, estimator):
    problem = gfem.Driven([f], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    return gfem.error_estimate(result, estimator, model=model)


@pytest.mark.parametrize("law", [LOSSLESS, LOSSY], ids=["lossless", "lossy"])
@pytest.mark.parametrize(
    "estimator",
    [gfem.FieldEstimator(), gfem.QuantityEstimator(s=("p2", "p1"))],
    ids=["field", "s"],
)
def test_a_law_region_is_estimated_at_its_value_at_the_solve_frequency(
    mesh, law, estimator
):
    f = 3 * GHZ
    with_law = _estimates(_line(mesh, gfem.Material(eps_r=law)), f, estimator)
    constant = _estimates(_line(mesh, _constant(law, f)), f, estimator)
    assert with_law.value == pytest.approx(constant.value, rel=1e-9)
    np.testing.assert_allclose(
        with_law.indicators, constant.indicators, rtol=1e-9, atol=1e-14
    )


@pytest.mark.usefixtures("no_solve")
def test_the_omega_estimator_refuses_a_law(tmp_path):
    cavity = rc.cavity_model(tmp_path, 2)
    model = gfem.Model(
        cavity.mesh,
        materials={"air": gfem.Material(eps_r=LOSSY)},
        boundaries=cavity.boundaries,
    )
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model, rc.CAVITY_PROBLEM, estimator=gfem.QuantityEstimator(omega=0)
        )
    assert caught.value.feature == "dispersive eigenmode"
    assert "FieldEstimator()" in str(caught.value)


@pytest.mark.parametrize(
    "estimator",
    [gfem.FieldEstimator(), gfem.QuantityEstimator(s=("p2", "p1"))],
    ids=["field", "s"],
)
@pytest.mark.usefixtures("no_solve")
def test_refinement_refuses_a_frequency_where_a_law_is_zero(mesh, estimator):
    model = _line(mesh, gfem.Material(eps_r=PLASMA))
    with pytest.raises(ValueError, match=r"'core'.*5 GHz.*no weight") as caught:
        gfem.refine_mesh(model, gfem.Driven([3 * GHZ, 5 * GHZ]), estimator=estimator)
    assert "|eps| = 0" in str(caught.value)


def test_an_estimate_refuses_a_frequency_where_a_law_is_zero(mesh):
    model = _line(mesh, gfem.Material(eps_r=PLASMA))
    # the solve itself is defined there: only the recovery has no weight
    result = gfem.solve(
        model, gfem.Driven([5 * GHZ], keep_fields=True), options=OPTIONS
    )
    with pytest.raises(ValueError, match=r"'core'.*5 GHz.*no weight"):
        gfem.error_estimate(result, gfem.FieldEstimator(), model=model)


@pytest.mark.parametrize(
    ("law", "lossy"),
    [
        (LOSSLESS, False),
        (PLASMA, False),
        (LOSSY, True),
        (
            gfem.PermittivityLaw(
                terms=[gfem.Drude(plasma_frequency=5 * GHZ, damping=1 * GHZ)]
            ),
            True,
        ),
    ],
    ids=["lorentz", "drude", "debye", "damped-drude"],
)
def test_a_law_region_is_lossy_when_a_term_carries_loss(mesh, law, lossy):
    model = _line(mesh, gfem.Material(eps_r=law))
    core = np.zeros(len(mesh.tetrahedra), bool)
    core[mesh.elements("core")] = True
    out = _lossy_elements(model)
    assert np.all(out[core] == lossy)
    assert not out[~core].any()
