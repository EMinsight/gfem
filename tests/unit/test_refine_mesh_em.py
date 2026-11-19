"""`gfem.refine_mesh` and `gfem.error_estimate` on driven and eigenmode problems.

The step line and the L cavity of `refinement_cases` at n = 2 carry a
re-entrant edge, where the field error concentrates; a rectangular PEC box
(`structured_meshes.plate`) carries smooth modes for the material cases.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F_BOX = 2.6e11  # near the box's first mode with eps_r = 1 (TM110, 261 GHz)


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    return rc.step_model(tmp_path_factory.mktemp("step"), 2)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    return rc.cavity_model(tmp_path_factory.mktemp("cavity"), 2)


def _box(directory, materials, *, sheet=False, excluded=()):
    """A 1 x 0.7 x 0.4 mm PEC box of two layers "a" (below) and "b".

    With ``sheet`` the layers' interface is PEC too; with ``excluded=["b"]``
    the box is layer "a" alone, closed by PEC on the interface.
    """
    layers = (("a", 0.2), ("b", 0.2))
    mesh = sm.plate(1.0, 0.7, layers, (5, 4, 1))
    path = sm.write_msh22(directory / "box.msh", mesh)
    walls = ["bottom", "side"]
    walls += ["a__b"] if sheet or excluded else []
    walls += [] if excluded else ["top"]
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials=materials,
        boundaries=[gfem.PEC(s) for s in walls],
        excluded=list(excluded),
    )


def _touches(model, point, columns):
    """Whether each tetrahedron has a node on the line through ``point``."""
    nodes = np.asarray(model.mesh.nodes)[:, columns]
    on = np.linalg.norm(nodes - np.asarray(point), axis=1) < 1e-12
    return on[np.asarray(model.mesh.tetrahedra)].any(axis=1)


def _estimate(model, problem):
    result = gfem.solve(model, problem, options=OPTIONS)
    return gfem.error_estimate(result, gfem.FieldEstimator(), model=model)


# --- the estimate ------------------------------------------------------------


def test_a_driven_estimate_has_one_value_per_frequency(step):
    estimate = _estimate(step, gfem.Driven([3e9, 8e9], keep_fields=True))
    assert len(estimate.values) == 2
    assert estimate.value == max(estimate.values)
    assert estimate.indicators.shape == (len(step.mesh.tetrahedra),)
    assert 1e-3 < estimate.value < 1
    # the step's re-entrant edge is the line x = 5 mm, z = 0.5 mm
    edge = _touches(step, (5e-3, 0.5e-3), [0, 2])
    assert edge[np.argmax(estimate.indicators)]


def test_frequencies_are_combined_by_root_mean_square(step):
    both = _estimate(step, gfem.Driven([3e9, 8e9], keep_fields=True))
    each = [_estimate(step, gfem.Driven(f, keep_fields=True)) for f in (3e9, 8e9)]
    assert both.values == pytest.approx([e.value for e in each], rel=1e-9)
    mean = (each[0].indicators ** 2 + each[1].indicators ** 2) / 2
    np.testing.assert_allclose(both.indicators**2, mean, rtol=1e-8, atol=0)


def test_a_driven_result_without_kept_fields_names_the_remedy(step):
    result = gfem.solve(step, gfem.Driven(3e9), options=OPTIONS)
    with pytest.raises(ValueError, match=r"keep_fields=True"):
        gfem.error_estimate(result, gfem.FieldEstimator(), model=step)


def test_an_eigenmode_estimate_has_one_value_per_mode(cavity):
    estimate = _estimate(cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2))
    assert len(estimate.values) == 2
    assert 1e-3 < estimate.values[0] < 1
    corner = _touches(cavity, (0.0, 0.0), [0, 1])
    single = _estimate(cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=1))
    assert corner[np.argmax(single.indicators)]
    assert single.values[0] == pytest.approx(estimate.values[0], rel=1e-6)


@pytest.mark.parametrize(
    "materials",
    [
        pytest.param(
            {"a": gfem.Material(eps_r=(2.0, 2.0, 2.0)), "b": gfem.Material()},
            id="tensor",
        ),
        pytest.param(
            lambda mesh: {
                "a": gfem.MaterialField(eps_r=np.full(len(mesh.elements("a")), 2.0)),
                "b": gfem.Material(),
            },
            id="field",
        ),
    ],
)
def test_tensor_and_field_permittivities_weigh_like_a_scalar(tmp_path, materials):
    scalar = _box(tmp_path, {"a": gfem.Material(eps_r=2.0), "b": gfem.Material()})
    if callable(materials):
        materials = materials(scalar.mesh)
    problem = gfem.Eigenmode(target=F_BOX / math.sqrt(1.5), n_modes=1)
    reference = _estimate(scalar, problem)
    estimate = _estimate(scalar.replace(materials=materials), problem)
    assert estimate.value == pytest.approx(reference.value, rel=1e-6)


def test_an_anisotropic_permittivity_runs(tmp_path):
    model = _box(
        tmp_path,
        {"a": gfem.Material(eps_r=(7.0, 3.0, 2.0)), "b": gfem.Material(eps_r=1.5)},
    )
    estimate = _estimate(model, gfem.Eigenmode(target=F_BOX / 1.3, n_modes=1))
    assert 1e-3 < estimate.value < 1


def test_a_lossy_layer_beside_a_tensor_weighs_like_a_scalar(tmp_path):
    # A tensor anywhere makes eps_eff a tensor everywhere: the lossy layer's
    # weights |eps_eff|^-1 and |eps_eff| then come from the matrix modulus.
    lossy = gfem.Material(eps_r=1.5, tan_delta=0.2, conductivity=20.0)
    scalar = _box(tmp_path, {"a": gfem.Material(eps_r=2.0), "b": lossy})
    tensor = scalar.replace(
        materials={"a": gfem.Material(eps_r=(2.0,) * 3), "b": lossy}
    )
    problem = gfem.Eigenmode(target=F_BOX / 1.3, n_modes=1)
    reference = _estimate(scalar, problem)
    assert 1e-3 < reference.value < 1
    assert _estimate(tensor, problem).value == pytest.approx(reference.value, rel=1e-6)


LOSSY_ANISOTROPIC = {
    "a": gfem.Material(eps_r=(7.0, 3.0, 2.0), tan_delta=0.01),
    "b": gfem.Material(),
}


def test_the_field_estimator_estimates_loss_in_an_anisotropic_material(tmp_path):
    # ADR 0138: the modulus weights of a complex tensor eps_eff are exact
    # (real axes), so the field estimate admits anisotropic loss.
    model = _box(tmp_path, LOSSY_ANISOTROPIC)
    problem = gfem.Eigenmode(target=F_BOX / 1.3, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert 1e-3 < estimate.value < 1
    assert np.all(np.isfinite(estimate.indicators))


def test_recovery_is_broken_across_an_interior_pec_sheet(tmp_path):
    # The sheet between the layers decouples them: the first mode lives in
    # the denser layer "a" alone, as in a model without layer "b". Recovered
    # across the sheet, its flux would be smeared into "b" and peak there.
    materials = {"a": gfem.Material(eps_r=2.0), "b": gfem.Material()}
    sheet = _box(tmp_path, materials, sheet=True)
    alone = _box(tmp_path, {"a": materials["a"]}, excluded=["b"])
    problem = gfem.Eigenmode(target=F_BOX / math.sqrt(2.0), n_modes=1)
    with_sheet = _estimate(sheet, problem)
    without_b = _estimate(alone, problem)
    a = sheet.mesh.elements("a")
    b = sheet.mesh.elements("b")
    assert with_sheet.value == pytest.approx(without_b.value, rel=1e-6)
    np.testing.assert_allclose(
        with_sheet.indicators[a], without_b.indicators[a], rtol=1e-5, atol=0
    )
    assert with_sheet.indicators[b].max() < 1e-6 * with_sheet.indicators.max()


# --- the loop ----------------------------------------------------------------


def test_a_driven_loop_keeps_the_fields_it_estimates(step):
    problem = rc.step_problem((3e9,))
    assert not problem.keep_fields
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(step, problem, max_cycles=1, options=OPTIONS)
    assert len(ref.cycles) == 2
    assert ref.cycles[0].estimate > ref.cycles[1].estimate == ref.estimate.value
    assert len(ref.result.e_fields) == 2  # one per excited port
    assert ref.cycles[-1].dofs == len(ref.result.e_fields[0].coefficients)
    again = gfem.error_estimate(ref.result, gfem.FieldEstimator(), model=ref.model)
    np.testing.assert_allclose(again.values, ref.estimate.values, rtol=1e-10)


def test_an_eigenmode_loop_refines_at_the_corner(cavity):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            cavity, rc.CAVITY_PROBLEM, max_cycles=1, tolerance=1e-3, options=OPTIONS
        )
    assert ref.cycles[0].estimate > ref.cycles[1].estimate
    corner = _touches(cavity, (0.0, 0.0), [0, 1])
    marked = np.bincount(ref.tetrahedron_parents, minlength=len(corner)) > 1
    assert marked[corner].all()


def test_a_wave_port_line_refines(tmp_path):
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            model, gfem.Driven(10e9), max_cycles=1, tolerance=1e-4, options=OPTIONS
        )
    assert len(ref.cycles) == 2
    assert ref.cycles[1].tetrahedra > ref.cycles[0].tetrahedra
    assert np.isfinite(ref.estimate.value)


def _h_min(model):
    nodes = np.asarray(model.mesh.nodes)
    tets = np.asarray(model.mesh.tetrahedra)
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return min(
        np.linalg.norm(nodes[tets[:, i]] - nodes[tets[:, j]], axis=1).min()
        for i, j in pairs
    )


def test_a_chip_scale_loop_warns_of_the_low_frequency_floor_once(tmp_path):
    # in um, k0 h_min = 5e-7 on the input mesh: every cycle is below the floor;
    # ungauged, since a gauged solve with lumped ports does not warn
    model = rc.step_model(tmp_path, 2, unit="um")
    f = 5e-7 * const.c / (2 * math.pi * _h_min(model))
    options = dataclasses.replace(OPTIONS, gauge="none")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ref = gfem.refine_mesh(model, gfem.Driven(f), max_cycles=2, options=options)
    (low,) = [w for w in record if w.category is gfem.LowFrequencyWarning]
    # checked once, on the returned model's mesh
    assert f"h_min = {_h_min(ref.model):.3g} m" in str(low.message)
    assert _h_min(ref.model) < _h_min(model)
    assert low.filename == __file__
