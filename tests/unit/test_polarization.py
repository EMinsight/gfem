"""`gfem.Driven(polarization=...)`: the field a 2D model's driven solve computes.

The value is checked when the problem is built, written to job files and keyed
by the solve cache; a 3D problem leaves it out of both, so its files and keys
are those of a gfem that predates it.
"""

import json

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._solve_cache import solve_key
from gfem.job import _write_result
from gfem.problems import _check_polarization


@pytest.mark.parametrize("polarization", ["Ez", "Hz", None])
def test_a_driven_problem_holds_its_polarization(polarization):
    problem = gfem.Driven(frequency=1e9, polarization=polarization)
    assert problem.polarization == polarization


def test_a_driven_problem_has_no_polarization_by_default():
    assert gfem.Driven(frequency=1e9).polarization is None


@pytest.mark.parametrize("value", ["TE", "TM", "ez", "Ex", 1])
def test_an_unknown_polarization_raises_naming_both(value):
    with pytest.raises(ValueError, match=r"Driven\.polarization.*'Ez'.*'Hz'"):
        gfem.Driven(frequency=1e9, polarization=value)


# --- job files and the solve cache ---------------------------------------------

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}

# A Driven problem as gfem 0.1.0 wrote it in a job file, before polarization.
DRIVEN_0_1_0 = {
    "$type": "Driven",
    "frequency": [1000000000.0],
    "excite": None,
    "sweep": "uniform",
    "keep_fields": False,
    "samples": None,
    "wave_vector": None,
    "wave_vector_frequency": None,
}


@pytest.fixture
def model(tmp_path):
    path = sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


def _written_problem(model, problem, tmp_path):
    path = gfem.Job(model, problem).to_json(tmp_path / "job" / "job.json")
    return json.loads(path.read_text())["job"]["problem"], path


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_job_file_keeps_the_polarization(model, tmp_path, polarization):
    problem = gfem.Driven(frequency=1e9, polarization=polarization)
    _, path = _written_problem(model, problem, tmp_path)
    assert gfem.Job.from_json(path).problem == problem


def test_a_3d_problem_is_written_as_gfem_0_1_0_wrote_it(model, tmp_path):
    written, _ = _written_problem(model, gfem.Driven(frequency=1e9), tmp_path)
    assert written == DRIVEN_0_1_0


def test_a_job_file_written_before_polarization_reads_without_one(model, tmp_path):
    _, path = _written_problem(model, gfem.Driven(frequency=2e9), tmp_path)
    text = json.loads(path.read_text())
    text["job"]["problem"] = DRIVEN_0_1_0
    path.write_text(json.dumps(text))
    assert gfem.Job.from_json(path).problem == gfem.Driven(frequency=1e9)


def test_the_solve_key_tells_ez_from_hz(model):
    options = gfem.SolverOptions(order=1)

    def key(polarization):
        problem = gfem.Driven(frequency=1e9, polarization=polarization)
        return solve_key(model, problem, options)[0]

    assert len({key("Ez"), key("Hz"), key(None)}) == 3


# --- the kept field of a 2D solve --------------------------------------------------

F0 = 1e9


@pytest.fixture
def strip(tmp_path):
    face = sm.z0_face(sm.plate(4.0, 3.0, LAYERS, (2, 1, 1)))
    return gfem.Mesh.from_gmsh(sm.write_msh22(tmp_path / "face.msh", face))


def _result(mesh, polarization, *, keep=True):
    quantity, unit = {"Ez": ("Ez", "V/m"), "Hz": ("Hz", "A/m")}[polarization]
    field = gfem.Field(
        coefficients=np.zeros(6, complex),
        mesh=mesh,
        quantity=quantity,
        unit=unit,
        space="h1",
        order=1,
        support=("oxide",),
        omega=2 * np.pi * F0,
    )
    s = gfem.NetworkParameters(
        values=np.zeros((1, 1, 1), complex),
        z_ref=np.full((1, 1), np.nan),
        frequency=(F0,),
        ports=("p1",),
        excited=("p1",),
    )
    kept = ((F0, "p1"),) if keep else ()
    fields = (field,) if keep else ()
    if polarization == "Ez":
        return gfem.DrivenResult(s, e_fields=fields, kept=kept, polarization="Ez")
    return gfem.DrivenResult(s, h_fields=fields, kept=kept, polarization="Hz")


def test_an_ez_result_reads_its_field_with_e_field(strip):
    result = _result(strip, "Ez")
    assert result.polarization == "Ez"
    assert result.e_field(F0).quantity == "Ez"
    assert result.h_fields == ()


def test_an_hz_result_reads_its_field_with_h_field(strip):
    result = _result(strip, "Hz")
    assert result.polarization == "Hz"
    assert result.h_field(F0, excite="p1").quantity == "Hz"
    assert result.e_fields == ()


def test_e_field_on_an_hz_result_raises_naming_h_field(strip):
    with pytest.raises(ValueError, match=r"Hz polarization.*result\.h_field"):
        _result(strip, "Hz").e_field(F0)


def test_h_field_on_an_ez_result_raises_naming_e_field(strip):
    with pytest.raises(ValueError, match=r"Ez polarization.*result\.e_field"):
        _result(strip, "Ez").h_field(F0)


def test_h_field_on_a_3d_result_raises_naming_e_field(strip):
    s = _result(strip, "Ez", keep=False).s
    with pytest.raises(ValueError, match=r"Hz polarization of a 2D.*e_field"):
        gfem.DrivenResult(s).h_field(F0)


def test_h_field_without_kept_fields_says_how_to_keep_them(strip):
    with pytest.raises(ValueError, match="keep_fields"):
        _result(strip, "Hz", keep=False).h_field(F0)


def test_a_result_with_both_kinds_of_fields_is_refused(strip):
    ez = _result(strip, "Ez")
    with pytest.raises(ValueError, match="h_fields"):
        gfem.DrivenResult(
            ez.s, e_fields=ez.e_fields, h_fields=ez.e_fields, kept=ez.kept
        )


# --- result bundles ------------------------------------------------------------------


def _bundle(result, tmp_path):
    # the writer behind Job.run's result bundle: a 2D solve has no rule yet
    out = tmp_path / "out"
    _write_result(out, result, {}, None)
    return out


def test_an_hz_result_bundle_reads_back_its_field_and_polarization(strip, tmp_path):
    result = _result(strip, "Hz")
    back = gfem.load_result(_bundle(result, tmp_path))
    assert back.polarization == "Hz"
    assert back.h_field(F0).quantity == "Hz"
    assert back.h_field(F0).mesh.dim == 2


def test_a_3d_result_bundle_has_neither_h_fields_nor_polarization(strip, tmp_path):
    result = _result(strip, "Ez", keep=False)
    plain = gfem.DrivenResult(result.s)
    out = _bundle(plain, tmp_path)
    written = json.loads((out / "result.json").read_text())["result"]
    assert "h_fields" not in written
    assert "polarization" not in written


# --- checked against the mesh at solve time --------------------------------------


def test_a_polarization_on_a_3d_mesh_raises_at_solve(model):
    problem = gfem.Driven(frequency=1e9, polarization="Ez")
    with pytest.raises(ValueError, match="polarization applies to a 2D mesh"):
        gfem.solve(model, problem)


def test_a_2d_mesh_without_a_polarization_raises_at_solve():
    # gfem.solve's check, called directly: a Model does not build on a 2D mesh yet
    with pytest.raises(ValueError, match=r"pass Driven\(polarization=\"Ez\"\)"):
        _check_polarization(gfem.Driven(frequency=1e9), dim=2)


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_2d_mesh_with_a_polarization_passes_the_check(polarization):
    _check_polarization(gfem.Driven(frequency=1e9, polarization=polarization), dim=2)


# --- the port mode problem ---------------------------------------------------------


@pytest.mark.parametrize("value", ["TE", "ez", 1])
def test_an_unknown_port_mode_polarization_raises_naming_both(value):
    with pytest.raises(ValueError, match=r"PortMode\.polarization.*'Ez'.*'Hz'"):
        gfem.PortMode("p1", 1e9, polarization=value)


def test_a_port_mode_polarization_on_a_3d_mesh_raises_at_solve():
    problem = gfem.PortMode("p1", 1e9, polarization="Hz")
    with pytest.raises(ValueError, match=r"PortMode\.polarization applies to a 2D"):
        _check_polarization(problem, dim=3)


@pytest.mark.parametrize("polarization", ["Ez", "Hz", None])
def test_a_job_file_keeps_a_port_modes_polarization(model, tmp_path, polarization):
    problem = gfem.PortMode("p1", 1e9, polarization=polarization)
    written, path = _written_problem(model, problem, tmp_path)
    assert gfem.Job.from_json(path).problem == problem
    # a 3D problem is written as before polarization existed
    assert ("polarization" in written) == (polarization is not None)
