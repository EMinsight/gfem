"""What an eigenmode solve with a wave port refuses, and the refinement it runs.

Each refusal names the wave port and gives a way out: more than one MPI
rank, a Bloch wave vector, a quantity estimate, a port face cutting a
perfectly matched layer, a target below the cutoff of a requested port
mode, and an absorbing edge beside a permittivity law in the port's
section. Field-estimator refinement runs.
"""

import math
import re

import numpy as np
import pml_cases
import pytest
import structured_meshes as sm

import gfem
from gfem import _mpi

C0 = 299792458.0
A_MM, B_MM = 22.86, 10.16
# a dielectric layer at the short, air at the port (TE10 cutoff 6.557 GHz)
CASE_A = (("dielectric", 4.0, 10.0), ("air", 1.0, 5.0))
ROOT_A = 11.665906948910 + 1.062461078093j  # GHz, the exact root
ORDER_2 = gfem.SolverOptions(order=2, linear_solver="direct")


def shorted_guide(path, layers, *, nx=6, nz_per_mm=0.5, ny=2):
    """WR-90 with ``layers`` from a short at z = 0 to the port at the far end."""
    tops = np.cumsum([t for _, _, t in layers])
    zs = [np.zeros(1)]
    for (_, _, t), z1 in zip(layers, tops, strict=True):
        n = max(1, round(t * nz_per_mm))
        zs.append(z1 - t + np.linspace(0, t, n + 1)[1:])
    edges = (
        np.linspace(0, A_MM, nx + 1),
        np.linspace(0, B_MM, ny + 1),
        np.concatenate(zs),
    )
    length = tops[-1]

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2] - length) < 1e-9:
            return "port"
        if any(abs(c[2] - z) < 1e-9 for z in tops[:-1]):
            return ""
        return "walls"

    sm.write_msh22(path, sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def case_a(tmp_path_factory):
    mesh = shorted_guide(tmp_path_factory.mktemp("a") / "a.msh", CASE_A)
    return gfem.Model(
        mesh,
        materials={name: gfem.Material(eps_r=eps) for name, eps, _ in CASE_A},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )


# --- refusals -------------------------------------------------------------------


def test_more_than_one_rank_raises_naming_the_port_before_any_mesh(case_a):
    # a static check on replicated values: every rank raises it alone
    with pytest.raises(gfem.DistributedUnsupportedError) as raised:
        _mpi.refuse(case_a, gfem.Eigenmode(target=11.67e9), ORDER_2)
    assert raised.value.feature == "eigenmode_wave_port"
    message = str(raised.value)
    assert "'p1'" in message
    assert "COMM_SELF" in message
    assert "gfem.Driven" in message


def test_a_driven_solve_with_the_wave_port_still_passes_above_one_rank(case_a):
    assert _mpi.refuse(case_a, gfem.Driven(11.67e9), ORDER_2) is None


def _periodic_cell_with_a_port(tmp_path):
    # the port's nodes run from x = 2.5 to 7.5 mm: none on the x pair
    def surface(c):
        return "port" if abs(c[2]) < 1e-9 and 2.5 < c[0] < 7.5 else ""

    edges = tuple(np.linspace(0, s, n + 1) for s, n in [(10, 4), (7, 3), (1, 1)])
    path = sm.write_msh22(
        tmp_path / "cell.msh", sm.periodic_cell(edges, surface=surface)
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material()},
        boundaries=[gfem.Periodic(("x_min", "x_max"))],
        ports=[gfem.WavePort("w", "port")],
    )


def test_a_bloch_wave_vector_raises_naming_the_port(tmp_path):
    model = _periodic_cell_with_a_port(tmp_path)
    problem = gfem.Eigenmode(target=20e9, wave_vector=(0.3 * math.pi / 10e-3, 0, 0))
    with pytest.raises(NotImplementedError) as raised:
        gfem.solve(model, problem, options=ORDER_2)
    message = str(raised.value)
    assert "WavePort 'w'" in message
    assert "wave_vector" in message
    assert "gfem.Driven" in message


@pytest.mark.parametrize(
    "estimator",
    [
        gfem.QuantityEstimator(omega=0),
        gfem.QuantityEstimator(q=0),
        gfem.QuantityEstimator(objective=lambda result, _model: result.omega[0].real),
    ],
    ids=["omega", "q", "objective"],
)
def test_refine_mesh_refuses_a_quantity_estimate_naming_the_port(case_a, estimator):
    with pytest.raises(gfem.MeshRefinementUnsupportedError) as raised:
        gfem.refine_mesh(
            case_a,
            gfem.Eigenmode(target=11.67e9),
            estimator=estimator,
            tolerance=1e-2,
            options=ORDER_2,
        )
    assert raised.value.feature == "lossy eigenmode"
    message = str(raised.value)
    assert "WavePort 'p1'" in message
    assert "gfem.FieldEstimator()" in message


def test_error_estimate_refuses_a_quantity_estimate_naming_the_port(case_a):
    result = gfem.solve(case_a, gfem.Eigenmode(target=11.67e9), options=ORDER_2)
    with pytest.raises(gfem.MeshRefinementUnsupportedError, match="WavePort 'p1'"):
        gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=case_a)


def test_a_port_face_cutting_a_perfectly_matched_layer_raises(tmp_path):
    # the port at x = 0 crosses the side layers beyond y = 0 and y = 1
    def surface(c):
        return "end" if abs(c[0]) < 1e-9 else ""

    mesh = gfem.Mesh.from_gmsh(
        pml_cases.box(tmp_path / "box.msh", ("-y", "+y"), surface=surface)
    )
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "pml": gfem.Material()},
        ports=[gfem.WavePort("w", "end")],
        pml=gfem.PML("pml"),
    )
    with pytest.raises(NotImplementedError) as raised:
        gfem.solve(model, gfem.Eigenmode(target=300e6), options=ORDER_2)
    message = str(raised.value)
    assert "WavePort 'w'" in message
    assert "perfectly matched layer" in message
    assert "gfem.Driven" in message


def test_a_target_below_the_port_cutoff_raises_naming_the_cutoff(case_a):
    # the air section's TE10 cutoff is c / (2 a) = 6.557 GHz
    with pytest.raises(gfem.PortModeError) as raised:
        gfem.solve(case_a, gfem.Eigenmode(target=5e9), options=ORDER_2)
    message = str(raised.value)
    assert "'p1'" in message
    assert "mode 1" in message
    assert re.search(r"cutoff is near 6\.[56]\d* GHz", message)  # to 1 %
    assert "gfem.Driven" in message


def test_an_absorbing_edge_beside_a_law_is_refused_before_the_port(tmp_path):
    # the port section's absorbing edge beside a law is refused at a complex
    # wavenumber; the 3D absorbing face behind the same law region raises first
    law = gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[gfem.Lorentz(delta_eps=0.2, resonance_frequency=40e9, damping=1e9)],
    )
    layers = (("guide", 1.0, 15.0),)
    mesh = shorted_guide(tmp_path / "law.msh", layers)
    model = gfem.Model(
        mesh,
        materials={"guide": gfem.Material(eps_r=law)},
        boundaries=[gfem.Absorbing("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )
    with pytest.raises(NotImplementedError) as raised:
        gfem.solve(model, gfem.Eigenmode(target=12e9), options=ORDER_2)
    message = str(raised.value)
    assert "Absorbing 'walls' lies on region 'guide'" in message
    assert "permittivity law" in message
    assert "gfem.Driven" in message


# --- refinement -----------------------------------------------------------------


def test_one_field_estimator_refinement_cycle_completes(case_a):
    problem = gfem.Eigenmode(target=11.67e9, n_modes=3)
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        refined = gfem.refine_mesh(
            case_a, problem, tolerance=1e-6, max_cycles=1, options=ORDER_2
        )
    assert len(refined.cycles) == 2
    first, last = refined.cycles
    assert last.tetrahedra > first.tetrahedra
    f = np.asarray(refined.result.omega) / (2 * math.pi) / 1e9
    assert np.min(np.abs(f - ROOT_A)) / abs(ROOT_A) < 1e-2
