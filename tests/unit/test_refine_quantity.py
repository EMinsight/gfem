"""`gfem.QuantityEstimator`: key checks, refusals, mode tracking, sheets.

The effectivity of the estimate is verified in
`tests/verification/test_quantity_estimate.py`; these cases check what is
asked of it before any solve, and how it reads a result.
"""

import dataclasses
import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

import gfem
from gfem._engine import estimate as kernels
from gfem.refinement import _dorfler

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture
def no_solve(monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement.solve", solve)


@pytest.fixture(scope="module")
def coax(tmp_path_factory):
    return rc.coax_model(tmp_path_factory.mktemp("coax"), 2)


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    return rc.step_model(tmp_path_factory.mktemp("step"), 2)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    return rc.cavity_model(tmp_path_factory.mktemp("cavity"), 2)


def _with(model, **changes):
    """``model`` rebuilt with other materials, boundaries or ports."""
    kwargs = {
        "materials": model.materials,
        "boundaries": model.boundaries,
        "ports": model.ports,
    }
    return gfem.Model(model.mesh, **{**kwargs, **changes})


def _waves(*, offset=0.0):
    return [
        gfem.WavePort("p1", "port1", offset=offset),
        gfem.WavePort("p2", "port2"),
    ]


# --- key checks, before any solve --------------------------------------------


@pytest.mark.usefixtures("no_solve")
def test_an_unknown_terminal_or_port_raises_key_error(coax, step):
    with pytest.raises(KeyError, match="'outer'"):
        gfem.refine_mesh(
            coax,
            rc.COAX_PROBLEM,
            estimator=gfem.QuantityEstimator(capacitance=("inner", "outer")),
        )
    with pytest.raises(KeyError, match="'p3'"):
        gfem.refine_mesh(
            step,
            rc.step_problem(),
            estimator=gfem.QuantityEstimator(s=("p3", "p1")),
        )


@pytest.mark.usefixtures("no_solve")
def test_an_unexcited_port_raises_with_the_remedy(step):
    problem = gfem.Driven([3e9], excite="p1")
    with pytest.raises(ValueError, match=r"Driven\.excite"):
        gfem.refine_mesh(
            step, problem, estimator=gfem.QuantityEstimator(s=("p2", "p1"))
        )


@pytest.mark.usefixtures("no_solve")
def test_a_mode_index_beyond_n_modes_raises(cavity):
    with pytest.raises(ValueError, match="n_modes=1"):
        gfem.refine_mesh(
            cavity, rc.CAVITY_PROBLEM, estimator=gfem.QuantityEstimator(omega=1)
        )


@pytest.mark.parametrize(
    "estimator",
    [
        gfem.QuantityEstimator(s=("inner", "inner")),
        gfem.QuantityEstimator(omega=0),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_a_keyword_of_another_problem_raises_type_error(coax, estimator):
    with pytest.raises(TypeError, match="Electrostatic"):
        gfem.refine_mesh(coax, rc.COAX_PROBLEM, estimator=estimator)


def test_error_estimate_checks_the_key_against_the_result(coax):
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    with pytest.raises(KeyError):
        gfem.error_estimate(
            result, gfem.QuantityEstimator(capacitance=("x", "x")), model=coax
        )
    with pytest.raises(TypeError):
        gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=coax)


# --- entries with no verification case, before any solve ---------------------


@pytest.mark.parametrize(
    ("case", "estimator", "feature"),
    [
        ("step-lossy-wave", ("s", ("p2", "p1")), "lossy wave port"),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_untested_entries_raise_before_any_solve(step, *, case, estimator, feature):
    model, problem = {
        "step-lossy-wave": (
            _with(
                step,
                materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=1e-3)},
                ports=_waves(),
            ),
            rc.step_problem(),
        ),
    }[case]
    keyword, key = estimator
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model, problem, estimator=gfem.QuantityEstimator(**{keyword: key})
        )
    assert caught.value.feature == feature


@pytest.mark.parametrize(
    ("case", "estimator"),
    [
        ("coax-anisotropic", ("capacitance", ("inner", "inner"))),
        ("step-mu", ("s", ("p2", "p1"))),
        ("step-anisotropic-mu", ("s", ("p2", "p1"))),
    ],
)
def test_anisotropic_permittivity_and_permeability_are_estimated(
    coax, step, *, case, estimator
):
    # Verified in tests/verification/test_quantity_estimate.py (anisotropic
    # coax, permeability and anisotropic lines and boxes).
    rotated = gfem.Material(
        eps_r=(3.0, 5.0, 2.0), mu_r=(1.5, 2.5, 2.0), axes=tuple(map(tuple, rc.ROT3.T))
    )
    model, problem = {
        "coax-anisotropic": (
            _with(coax, materials={"dielectric": gfem.Material(eps_r=(1, 2, 3))}),
            rc.COAX_PROBLEM,
        ),
        "step-mu": (
            _with(step, materials={"dielectric": gfem.Material(eps_r=4.0, mu_r=2.0)}),
            gfem.Driven([3e9], keep_fields=True),
        ),
        "step-anisotropic-mu": (
            _with(step, materials={"dielectric": rotated}),
            gfem.Driven([3e9], keep_fields=True),
        ),
    }[case]
    keyword, key = estimator
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(**{keyword: key}), model=model
    )
    assert np.all(np.isfinite(estimate.values))
    assert np.all(np.asarray(estimate.values) > 0)


def test_a_lossless_wave_port_s_entry_is_estimated(step):
    model = _with(step, ports=_waves())
    result = gfem.solve(model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS)
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=model
    )
    assert 0 < estimate.value < 1e-2


def test_a_de_embedding_offset_leaves_the_s_estimate_unchanged(step):
    # The offset multiplies S by exp(i beta offset) after the solve, of
    # modulus 1 on a lossless port: the fields and the estimate stay.
    estimator = gfem.QuantityEstimator(s=("p2", "p1"))
    values = []
    for offset in (0.0, 1e-3):
        model = _with(step, ports=_waves(offset=offset))
        result = gfem.solve(
            model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS
        )
        values.append(gfem.error_estimate(result, estimator, model=model).value)
    assert values[1] == pytest.approx(values[0], rel=1e-12)


# --- reading a result --------------------------------------------------------


def test_the_capacitance_estimate_is_the_squared_field_estimate(coax):
    # C_h - C = a(e, e) for a terminal at 1 V: the relative capacitance
    # estimate is the relative energy estimate squared.
    result = gfem.solve(coax, rc.COAX_PROBLEM, options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=coax)
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    quantity = gfem.error_estimate(result, estimator, model=coax)
    assert quantity.values == pytest.approx((field.value**2,), rel=1e-10)
    assert quantity.estimator == estimator
    # and it marks the same tetrahedra: its indicators are the field estimator's
    np.testing.assert_allclose(quantity.indicators, field.indicators, rtol=1e-10)


def test_the_diagonal_capacitance_kernel_is_the_field_kernel_bit_for_bit(coax):
    # The (i, j) kernel at i = j is the diagonal estimate it replaced: the
    # field estimator's squared indicators of that terminal's potential.
    problem = gfem.Electrostatic(terminals=["inner", "outer"])
    result = gfem.solve(coax, problem, options=OPTIONS)
    (group,) = kernels.electrostatic(coax, result)
    for index, terminal in enumerate(("inner", "outer")):
        value, weights = kernels.capacitance(coax, result, terminal, terminal)
        assert value == float(group[index].sum())
        np.testing.assert_array_equal(weights, group[index])


def test_an_off_diagonal_capacitance_is_estimated(tmp_path):
    model = rc.sheet_model(tmp_path, 2)
    result = gfem.solve(model, rc.SHEET_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    assert 0 < estimate.value < 1
    assert float(np.sum(estimate.indicators**2)) == pytest.approx(
        estimate.value, rel=1e-12
    )
    # the entry is symmetric, and so is its estimate
    swapped = gfem.QuantityEstimator(capacitance=("b", "a"))
    assert gfem.error_estimate(result, swapped, model=model).value == pytest.approx(
        estimate.value, rel=1e-12
    )


def test_an_s_estimate_has_one_value_per_frequency(step):
    result = gfem.solve(
        step, gfem.Driven([3e9, 8e9], keep_fields=True), options=OPTIONS
    )
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    assert len(estimate.values) == 2
    assert estimate.value == max(estimate.values)
    assert all(0 < v < 1e-2 for v in estimate.values)


@pytest.mark.parametrize("frequency", [3e9, 8e9])
def test_an_s_estimate_marks_as_the_field_estimator(step, frequency):
    # The indicator is |a_K(e~_1, e~_2)|, the curl and mass terms of one
    # tetrahedron together: on the step line it follows the field estimator's
    # (correlation 0.9999997 at n = 4 in the DWR prototype) and marks the same
    # tetrahedra.
    result = gfem.solve(step, gfem.Driven(frequency, keep_fields=True), options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=step)
    quantity = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=step
    )
    assert np.corrcoef(quantity.indicators**2, field.indicators**2)[0, 1] > 0.99999
    np.testing.assert_array_equal(
        _dorfler(quantity.indicators, 0.5), _dorfler(field.indicators, 0.5)
    )


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_cycles_record_the_quantity(coax, step, cavity):
    ref = gfem.refine_mesh(
        coax,
        rc.COAX_PROBLEM,
        estimator=gfem.QuantityEstimator(capacitance=("inner", "inner")),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == float(ref.result.capacitance[0, 0])
    assert ref.cycles[0].estimate == ref.estimate.value
    ref = gfem.refine_mesh(
        step,
        rc.step_problem(),
        estimator=gfem.QuantityEstimator(s=("p2", "p1")),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == tuple(complex(v) for v in ref.result.s["p2", "p1"])
    ref = gfem.refine_mesh(
        cavity,
        rc.CAVITY_PROBLEM,
        estimator=gfem.QuantityEstimator(omega=0),
        max_cycles=0,
        options=OPTIONS,
    )
    assert ref.cycles[0].value == complex(ref.result.omega[0])


# --- eigenmode tracking ------------------------------------------------------


def test_a_mode_in_a_degenerate_cluster_raises(cavity):
    result = gfem.solve(
        cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2), options=OPTIONS
    )
    degenerate = dataclasses.replace(result, cluster=jnp.zeros(2, int))
    estimator = gfem.QuantityEstimator(omega=0)
    gfem.error_estimate(result, estimator, model=cavity)
    with pytest.raises(gfem.DegenerateModeError, match="cluster_rtol"):
        gfem.error_estimate(degenerate, estimator, model=cavity)


def test_a_mode_within_cluster_rtol_of_another_raises(cavity):
    # refine_mesh's rule, with the default SolverOptions().cluster_rtol = 1e-6
    result = gfem.solve(
        cavity, gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=2), options=OPTIONS
    )
    omega = np.asarray(result.omega)
    estimator = gfem.QuantityEstimator(omega=0)
    apart = dataclasses.replace(
        result, omega=jnp.asarray([omega[0], omega[0] * 1.00001])
    )
    gfem.error_estimate(apart, estimator, model=cavity)
    close = dataclasses.replace(
        result, omega=jnp.asarray([omega[0], omega[0] * 1.0000001])
    )
    with pytest.raises(gfem.DegenerateModeError, match="cluster_rtol=1e-06"):
        gfem.error_estimate(close, estimator, model=cavity)


@pytest.mark.filterwarnings("ignore::gfem.MeshRefinementNotConvergedWarning")
def test_the_mode_is_followed_by_nearest_omega(cavity):
    # Mode 1 of three around the first resonance: on every refined mesh the
    # estimate stays with the mode nearest the previous omega.
    problem = gfem.Eigenmode(target=rc.CAVITY_F1, n_modes=3)
    ref = gfem.refine_mesh(
        cavity,
        problem,
        estimator=gfem.QuantityEstimator(omega=1),
        tolerance=1e-9,
        max_cycles=2,
        options=OPTIONS,
    )
    omegas = [cycle.value for cycle in ref.cycles]
    first = np.asarray(gfem.solve(cavity, problem, options=OPTIONS).omega)
    assert omegas[0] == complex(first[1])
    for before, after in itertools.pairwise(omegas):
        assert abs(after - before) < abs(first[1] - first[0]) / 2


# --- lossy eigenmodes --------------------------------------------------------


def _box(directory, material=None, top=None):
    """The 1 x 0.7 x 0.4 mm box, h = 0.2 mm, PEC but for ``top``."""
    path = sm.write_msh22(
        directory / "box.msh", sm.plate(1.0, 0.7, (("air", 0.4),), (5, 4, 2))
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": material or gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("side"), top or gfem.PEC("top")],
    )


#: Near the box's 110 mode, Hz.
BOX_110 = 3e8 * math.sqrt(1 / 1e-6 + 1 / 0.49e-6) / 2


def test_a_lossless_mode_keeps_its_estimate(tmp_path):
    # Measured with the lossless kernel (before the lossy form): on a
    # lossless mode the lossy form is the same number.
    model = _box(tmp_path)
    problem = gfem.Eigenmode(target=BOX_110, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    assert estimate.value == pytest.approx(0.0016711308802387496, rel=1e-12)
    assert np.sum(estimate.indicators**2) == pytest.approx(
        0.0008213829986877815, rel=1e-12
    )


@pytest.mark.parametrize(
    ("material", "top"),
    [
        (gfem.Material(tan_delta=0.1, conductivity=5.0), None),
        (gfem.MaterialField(eps_r=1 - 0.1j), None),
        (None, gfem.Impedance("top", resistance=377.0)),
        (None, gfem.Absorbing("top")),
        (None, gfem.Conductor("top", conductivity=1e5)),
    ],
    ids=["volume", "complex-eps", "impedance", "absorbing", "conductor"],
)
def test_a_lossy_mode_is_estimated(tmp_path, material, top):
    model = _box(tmp_path, material, top)
    problem = gfem.Eigenmode(target=BOX_110, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    assert abs(complex(result.omega[0]).imag) > 0
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    assert 0 < estimate.value < 1e-2
    assert np.all(np.isfinite(estimate.indicators))


@pytest.mark.parametrize(
    ("material", "boundary"),
    [
        (None, gfem.Absorbing("wall", order=2)),
        (gfem.MaterialField(mu_r=1 - 0.01j), None),
        (gfem.Material(eps_r=(1, 2, 3), tan_delta=0.01), None),
    ],
    ids=["second-order-absorbing", "lossy-mu", "anisotropic-loss"],
)
@pytest.mark.usefixtures("no_solve")
def test_the_lossy_eigenmode_refusal_names_what_stays_refused(
    cavity, material, boundary
):
    changes = {}
    if material is not None:
        changes["materials"] = {"air": material}
    if boundary is not None:
        changes["boundaries"] = [boundary]
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            _with(cavity, **changes),
            rc.CAVITY_PROBLEM,
            estimator=gfem.QuantityEstimator(omega=0),
        )
    message = str(caught.value)
    assert caught.value.feature == "lossy eigenmode"
    assert "second-order" in message
    assert "wave port" not in message


# --- electrostatic sheets ----------------------------------------------------


def test_an_interior_sheet_terminal_does_not_smear_the_estimate(tmp_path):
    # A sheet terminal between two layers, ground above and below: V is
    # linear in each layer, so the solve is exact and its flux jumps across
    # the sheet. Recovered across the sheet, the flux would be smeared.
    layers = (("a", 0.5), ("b", 1.0))
    path = sm.write_msh22(tmp_path / "plate.msh", sm.plate(3.0, 2.0, layers, (2, 2, 2)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"a": gfem.Material(eps_r=2.0), "b": gfem.Material(eps_r=5.0)},
    )
    problem = gfem.Electrostatic(terminals={"sheet": "a__b"}, ground=["bottom", "top"])
    result = gfem.solve(model, problem, options=OPTIONS)
    field = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    estimator = gfem.QuantityEstimator(capacitance=("sheet", "sheet"))
    quantity = gfem.error_estimate(result, estimator, model=model)
    assert field.value < 1e-6
    assert quantity.value < 1e-12
