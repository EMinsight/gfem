"""Floquet models off the uniform driven sweep, readers, the oblique-absorber warning.

Every path but the uniform driven sweep refuses a model with Floquet ports
by name (ADR 0149): the adaptive sweep, the GPU solver and PETSc before the
linear solver is resolved, eigenmode, mesh refinement, error estimates and
`gfem.PortMode`; the MPI refusal is a row of `mpi_cases`. An objective that
reads `DrivenResult.diffraction_orders` is a refused leaf of the estimates,
and `gfem.power_flow` admits a Floquet face inside its side and names one
outside it. A first-order `gfem.Absorbing` face of a periodic cell warns when
a propagating wave (a diffraction order or the Bloch wave) meets it
obliquely (ADR 0150).
"""

import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem import _objective
from gfem._engine import driven

P2 = gfem.SolverOptions(order=2, threads=2)
#: |power_flow + |S_TR|^2| through the substrate of coarse F1 (2.0e-5
#: measured: the Robin term absorbs the unmodelled evanescent remainder).
POWER_TOL = 1e-4
PORTS = "'R_TE', 'R_TM', 'T_TE', 'T_TM'"


@pytest.fixture(scope="module")
def f1_mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("f1"))


@pytest.fixture(scope="module")
def f1_model(f1_mesh):
    return pc.f1(f1_mesh)


@pytest.fixture(scope="module")
def f1_kept(f1_model):
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE"], keep_fields=True)
    return gfem.solve(f1_model, problem, options=P2)


@pytest.fixture
def unresolved(monkeypatch):
    def resolve(*_args, **_kwargs):
        raise AssertionError("the linear solver was resolved before the refusal")

    monkeypatch.setattr(driven.ResolvedLinearSolver, "resolve", resolve)


# --- the driven path: adaptive sweep, GPU, PETSc -----------------------------------


@pytest.mark.usefixtures("unresolved")
@pytest.mark.parametrize(
    ("sweep", "options", "text"),
    [
        (
            "adaptive",
            gfem.SolverOptions(),
            (
                "the adaptive sweep cannot solve them in this version of gfem: use "
                "sweep='uniform' (SolverOptions.frequency_workers spreads its "
                "frequencies over processes)."
            ),
        ),
        (
            "uniform",
            gfem.SolverOptions(device="cuda"),
            (
                "the GPU solver (device='cuda', cuDSS) has no Floquet port path in "
                "this version of gfem: use device='cpu'."
            ),
        ),
        (
            "uniform",
            gfem.SolverOptions(linear_solver="petsc"),
            (
                "linear_solver='petsc' has no Floquet port path in this version of "
                "gfem: use linear_solver='direct' or 'bddc_gmres'."
            ),
        ),
    ],
    ids=["adaptive", "cuda", "petsc"],
)
def test_a_driven_path_with_no_floquet_path_refuses_before_resolving_the_solver(
    f1_model, sweep, options, text
):
    frequency = [pc.F1_FREQUENCY, 1.1 * pc.F1_FREQUENCY]
    problem = gfem.Driven(frequency, sweep=sweep)
    expected = f"Driven: the model has Floquet ports ({PORTS}), and {text}"
    with pytest.raises(NotImplementedError, match=re.escape(expected)):
        gfem.solve(f1_model, problem, options=options)


# --- the other paths -------------------------------------------------------------


def test_eigenmode_refuses_a_floquet_port(f1_model):
    with pytest.raises(NotImplementedError) as caught:
        gfem.solve(f1_model, gfem.Eigenmode(target=pc.F1_FREQUENCY, n_modes=1))
    assert str(caught.value) == (
        "Eigenmode does not support Floquet port 'R_TE' in this version of gfem: "
        "its orders' propagation constants must be evaluated at the complex "
        "eigenfrequency. Find the resonances and Q from a gfem.Driven sweep, or "
        "end the cell with gfem.Absorbing on 'z_max' (first order: it reflects "
        "oblique waves)."
    )


def test_mesh_refinement_refuses_a_floquet_port(f1_model, monkeypatch):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement._solve", solve)
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.refine_mesh(f1_model, gfem.Driven(pc.F1_FREQUENCY), options=P2)
    assert caught.value.feature == "floquet port"
    assert str(caught.value) == (
        "Mesh refinement has no path for a Floquet port: refining one side of a "
        "periodic pair would break its node pairing. Refine the mesh in your "
        "mesher."
    )


@pytest.mark.parametrize(
    "estimator",
    [gfem.FieldEstimator(), gfem.QuantityEstimator(s=("R_TE", "R_TE"))],
    ids=["field", "quantity"],
)
def test_an_error_estimate_refuses_a_floquet_port(f1_model, f1_kept, estimator):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        gfem.error_estimate(f1_kept, estimator, model=f1_model)
    assert caught.value.feature == "floquet port"
    assert str(caught.value) == (
        f"gfem.{type(estimator).__name__} has no path for a Floquet port: its "
        "flux recovery does not pair the periodic surfaces."
    )


def test_a_port_mode_problem_refuses_a_floquet_port(f1_model):
    with pytest.raises(ValueError) as caught:
        gfem.solve(f1_model, gfem.PortMode("T_TM", pc.F1_FREQUENCY))
    assert str(caught.value) == (
        "PortMode(port='T_TM'): 'T_TM' is a FloquetPort; the port mode problem "
        "needs a WavePort. A Floquet port's orders are plane waves with no mode "
        "solve: Model.propagating_orders lists them, and "
        "DrivenResult.diffraction_orders gives each order's gamma and angles."
    )


# --- readers ---------------------------------------------------------------------


def test_an_objective_reading_a_diffraction_order_is_a_refused_leaf(f1_model, f1_kept):
    # error_estimate refuses the model first; the classifier refuses the leaf
    # whichever path reaches it
    def objective(result, _model):
        return jnp.abs(result.diffraction_orders["R_TE"].gamma[0])

    with pytest.raises(gfem.MeshRefinementUnsupportedError) as caught:
        _objective.classify(objective, f1_kept, f1_model)
    assert caught.value.feature == "diffraction order output"


def test_power_flow_admits_a_floquet_face_inside_its_side(f1_model, f1_kept):
    # the substrate holds the T face: the power leaving it is minus the
    # power transmitted into it, |S(T_TE, R_TE)|^2 of the 1 W incident wave
    field = f1_kept.e_field(pc.F1_FREQUENCY)
    power = float(gfem.power_flow(field, ("sub",), model=f1_model))
    transmitted = abs(complex(f1_kept.s["T_TE", "R_TE"][0])) ** 2
    assert power == pytest.approx(-transmitted, abs=POWER_TOL)


def test_power_flow_names_a_floquet_face_outside_its_side(f1_model, f1_kept):
    # half the cell along x: each Floquet face's outside half shares edges
    # with it; the first port in model order is named
    def half(x):
        return x[:, 0] > pc.F1_A / 2 * 1e-6

    with pytest.raises(ValueError, match=r"of FloquetPort 'R_TE''s surface 'z_max'"):
        gfem.power_flow(f1_kept.e_field(pc.F1_FREQUENCY), half, model=f1_model)


# --- the oblique-absorber warning ---------------------------------------------------

MM = 1e-3
#: A cell along x between two `Absorbing` faces, periodic along y and z.
CELL_L, CELL_SHEET = 6.0, 2.5  # mm


def _absorbing_cell(directory, a_y, a_z, eps_r):
    edges = (
        np.linspace(0.0, CELL_L, 25),
        np.linspace(0.0, a_y, 3),
        np.linspace(0.0, a_z, 3),
    )
    raw = sm.periodic_cell(edges, sheets=(("sheet", CELL_SHEET),), paired="yz")
    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(directory / "cell.msh", raw), length_unit="mm"
    )
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material(eps_r=eps_r)},
        boundaries=[
            gfem.Periodic(("y_min", "y_max")),
            gfem.Periodic(("z_min", "z_max")),
            gfem.Absorbing("x_min"),
            gfem.Absorbing("x_max"),
        ],
        ports=[gfem.LumpedPort("p1", "sheet", "+z", impedance=50.0)],
    )


def _oblique(model, problem):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gfem.solve(
            model, problem, options=gfem.SolverOptions(order=1, linear_solver="direct")
        )
    return [str(w.message) for w in caught if w.category is gfem.ObliqueAbsorberWarning]


def _text(surface, theta, order, frequency):
    r = (1 - math.cos(theta)) / (1 + math.cos(theta))
    return (
        f"Absorbing {surface!r} meets a plane wave at {math.degrees(theta):.1f}° "
        f"(order {order} at {frequency}), and its first-order condition reflects "
        f"about {r:.1%} of the amplitude of such a wave; a gfem.FloquetPort on "
        f"{surface!r} absorbs every order at any angle."
    )


def test_a_grating_wider_than_the_wavelength_warns_at_normal_incidence(tmp_path):
    # period 1 mm along y at 360 GHz in air: orders (±1, 0) propagate at
    # sin(theta) = c / (f a); the 0.5 mm z period keeps (0, ±1) evanescent
    model = _absorbing_cell(tmp_path, 1.0, 0.5, 1.0)
    theta = math.asin(const.c / (360e9 * MM))
    assert theta > math.radians(50)
    messages = _oblique(model, gfem.Driven(360e9))
    assert messages == [
        _text("x_min", theta, (-1, 0), "360 GHz"),
        _text("x_max", theta, (-1, 0), "360 GHz"),
    ]


def test_an_oblique_bloch_slab_warns_and_a_normal_one_does_not(tmp_path):
    model = _absorbing_cell(tmp_path, 1.0, 1.0, 3.0)
    f = 10e9
    k = 2 * math.pi * f / const.c * math.sqrt(3.0)
    theta = math.radians(30.0)
    oblique = gfem.Driven(f, wave_vector=(0.0, 0.0, k * math.sin(theta)))
    assert _oblique(model, oblique) == [
        _text("x_min", theta, (0, 0), "10 GHz"),
        _text("x_max", theta, (0, 0), "10 GHz"),
    ]
    assert _oblique(model, gfem.Driven(f)) == []


def test_a_wave_below_one_percent_reflection_does_not_warn(tmp_path):
    # r = 0.01 at cos(theta) = 0.99/1.01, about 11.4 degrees: 10 degrees passes
    model = _absorbing_cell(tmp_path, 1.0, 1.0, 3.0)
    f = 10e9
    k = 2 * math.pi * f / const.c * math.sqrt(3.0)
    problem = gfem.Driven(f, wave_vector=(0.0, k * math.sin(math.radians(10.0)), 0.0))
    assert _oblique(model, problem) == []
