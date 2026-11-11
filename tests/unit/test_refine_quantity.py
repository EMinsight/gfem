"""`gfem.QuantityEstimator`: key checks, refusals, mode tracking, sheets.

The effectivity of the estimate is verified in
`tests/verification/test_quantity_estimate.py`; these cases check what is
asked of it before any solve, and how it reads a result.
"""

import dataclasses
import itertools

import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

import gfem

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
        (
            "coax-anisotropic",
            ("capacitance", ("inner", "inner")),
            "anisotropic permittivity",
        ),
        ("step-mu", ("s", ("p2", "p1")), "permeability"),
        ("step-deembedded", ("s", ("p2", "p1")), "de-embedding"),
        ("step-shared", ("s", ("p2", "p1")), "shared port surface"),
        ("step-lossy-wave", ("s", ("p2", "p1")), "lossy wave port"),
        ("cavity-lossy", ("omega", 0), "lossy eigenmode"),
        ("cavity-absorbing", ("omega", 0), "lossy eigenmode"),
    ],
)
@pytest.mark.usefixtures("no_solve")
def test_untested_entries_raise_before_any_solve(
    coax, step, cavity, *, case, estimator, feature
):
    model, problem = {
        "coax-anisotropic": (
            _with(coax, materials={"dielectric": gfem.Material(eps_r=(1, 2, 3))}),
            rc.COAX_PROBLEM,
        ),
        "step-mu": (
            _with(step, materials={"dielectric": gfem.Material(eps_r=4.0, mu_r=2.0)}),
            rc.step_problem(),
        ),
        "step-deembedded": (
            _with(step, ports=_waves(offset=1e-3)),
            rc.step_problem(),
        ),
        "step-shared": (
            _with(
                step,
                ports=[
                    gfem.WavePort("p1", "port1"),
                    gfem.WavePort("p2", "port2"),
                    gfem.WavePort("p3", "port2", mode=2),
                ],
            ),
            rc.step_problem(),
        ),
        "step-lossy-wave": (
            _with(
                step,
                materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=1e-3)},
                ports=_waves(),
            ),
            rc.step_problem(),
        ),
        "cavity-lossy": (
            _with(cavity, materials={"air": gfem.Material(tan_delta=1e-3)}),
            rc.CAVITY_PROBLEM,
        ),
        "cavity-absorbing": (
            _with(cavity, boundaries=[gfem.Absorbing("wall")]),
            rc.CAVITY_PROBLEM,
        ),
    }[case]
    keyword, key = estimator
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            model, problem, estimator=gfem.QuantityEstimator(**{keyword: key})
        )
    assert caught.value.feature == feature


@pytest.mark.usefixtures("no_solve")
def test_an_off_diagonal_capacitance_raises_before_any_solve(coax):
    problem = gfem.Electrostatic(terminals=["inner", "outer"])
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(
            coax,
            problem,
            estimator=gfem.QuantityEstimator(capacitance=("inner", "outer")),
        )
    assert caught.value.feature == "off-diagonal capacitance"


def test_a_lossless_wave_port_s_entry_is_estimated(step):
    model = _with(step, ports=_waves())
    result = gfem.solve(model, gfem.Driven([3e9], keep_fields=True), options=OPTIONS)
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(s=("p2", "p1")), model=model
    )
    assert 0 < estimate.value < 1e-2


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
