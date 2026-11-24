"""Error estimates and mesh refinement refuse incident fields and point dipoles.

The estimators' residuals know nothing of a volume or a point source, so
they would mark the scatterer and the dipole's position spuriously: every
estimator and `gfem.refine_mesh` refuse a problem with sources by name. A
unit cube of air closed by an absorbing face: no layer, whose own refusal
would come first.
"""

import pml_cases as pc
import pytest

import gfem

AIR = gfem.Material()
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
DIPOLE = gfem.ElectricDipole(
    "d", position=(0.47e-6, 0.53e-6, 0.56e-6), moment=(0.0, 0.0, 1e-30)
)
SOURCES = {"incident field": WAVE, "point dipole": DIPOLE}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = pc.box(tmp_path_factory.mktemp("refusals") / "c.msh", (), cells=2)
    return gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": AIR},
        boundaries=[gfem.Absorbing("outer")],
        background=AIR,
    )


@pytest.fixture(scope="module")
def results(model):
    return {
        feature: gfem.solve(
            model, gfem.Driven(F, sources=source, keep_fields=True), options=OPTIONS
        )
        for feature, source in SOURCES.items()
    }


ESTIMATORS = [
    gfem.FieldEstimator(),
    gfem.QuantityEstimator(objective=lambda r, _: abs(r.emitted_power["d"][0])),
]


@pytest.mark.parametrize("feature", SOURCES)
@pytest.mark.parametrize("estimator", ESTIMATORS, ids=["field", "objective"])
def test_an_error_estimate_of_a_source_column_is_refused(
    model, results, feature, estimator
):
    name = SOURCES[feature].name
    subject = type(estimator).__name__
    with pytest.raises(
        gfem.MeshRefinementUnsupportedError,
        match=rf"^gfem.{subject} has no path for the {feature} {name!r}",
    ) as caught:
        gfem.error_estimate(results[feature], estimator, model=model)
    assert caught.value.feature == feature


@pytest.mark.parametrize("feature", SOURCES)
def test_refine_mesh_refuses_a_source_before_any_solve(model, monkeypatch, feature):
    from gfem import refinement

    def no_solve(*_, **__):
        raise AssertionError("solved")

    monkeypatch.setattr(refinement, "solve", no_solve, raising=False)
    source = SOURCES[feature]
    with pytest.raises(
        gfem.MeshRefinementUnsupportedError,
        match=rf"^gfem.refine_mesh has no path for the {feature} {source.name!r}",
    ) as caught:
        gfem.refine_mesh(model, gfem.Driven(F, sources=source))
    assert caught.value.feature == feature


def test_the_source_is_named_before_the_layer(tmp_path):
    # refining the model closed by an absorbing face, as the layer's refusal
    # advises, would refuse the source next: name it first
    path = pc.box(tmp_path / "l.msh", pc.SIDES, cells=2)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": AIR, "pml": AIR},
        pml=gfem.PML("pml"),
        background=AIR,
    )
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(model, gfem.Driven(F, sources=DIPOLE))
    assert caught.value.feature == "point dipole"
