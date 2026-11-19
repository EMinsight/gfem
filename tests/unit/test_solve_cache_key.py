"""The solve key: one sha256 per (model, problem, options) that a solve depends on."""

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _codec
from gfem._refine import refine_model
from gfem._solve_cache import solve_key

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
PROBLEM = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
OPTIONS = gfem.SolverOptions(order=1)


@pytest.fixture
def mesh_path(tmp_path):
    return sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))


@pytest.fixture
def model(mesh_path):
    return gfem.Model(gfem.Mesh.from_gmsh(mesh_path), materials=MATERIALS)


def _key(model, problem=PROBLEM, options=OPTIONS):
    return solve_key(model, problem, options)[0]


def test_the_key_is_a_sha256_with_the_environment(model):
    key, env = solve_key(model, PROBLEM, OPTIONS)
    assert len(key) == 64
    assert int(key, 16) >= 0
    assert env["gfem"] == gfem.__version__
    assert len(env["gfem_code"]) == 64
    assert "ngsolve" in env
    assert "petsc4py" not in env


def test_a_job_read_back_in_another_directory_has_the_same_key(tmp_path):
    bundle = tmp_path / "a"
    bundle.mkdir()
    mesh = sm.write_msh22(bundle / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    model = gfem.Model(gfem.Mesh.from_gmsh(mesh), materials=MATERIALS)
    path = gfem.Job(model, PROBLEM, OPTIONS).to_json(bundle / "job.json")
    moved = tmp_path / "b"
    shutil.copytree(path.parent, moved)
    shutil.rmtree(path.parent)
    back = gfem.Job.from_json(moved / "job.json")
    assert back.model.mesh.source != model.mesh.source
    assert _key(back.model, back.problem, back.options) == _key(model)


def test_a_subprocess_computes_the_same_key(model):
    script = (
        "import sys, gfem\n"
        "from gfem._solve_cache import solve_key\n"
        "model = gfem.Model(gfem.Mesh.from_gmsh(sys.argv[1]), materials={\n"
        "    'oxide': gfem.Material(eps_r=3.9), 'nitride': gfem.Material(eps_r=7.5)})\n"
        "terminals = {'top': 'top', 'mid': 'oxide__nitride'}\n"
        "problem = gfem.Electrostatic(terminals=terminals)\n"
        "print(solve_key(model, problem, gfem.SolverOptions(order=1))[0])\n"
    )
    src = Path(gfem.__file__).parent.parent
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(src), *sys.path])}
    out = subprocess.run(
        [sys.executable, "-c", script, str(model.mesh.source)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == _key(model)


def _with_field(model, values):
    return model.with_material("oxide", gfem.MaterialField(eps_r=values))


def test_numpy_and_jax_arrays_have_the_same_key(model):
    n = len(model.mesh.elements("oxide"))
    values = np.linspace(2.0, 4.0, n)
    assert _key(_with_field(model, values)) == _key(
        _with_field(model, jnp.asarray(values))
    )
    nodes = np.asarray(model.mesh.nodes) * 1.01
    on_numpy = model.replace(mesh=gfem.Mesh(nodes, model.mesh._topology))
    on_jax = model.replace(mesh=gfem.Mesh(jnp.asarray(nodes), model.mesh._topology))
    assert isinstance(on_numpy.mesh.nodes, np.ndarray)
    assert _key(on_numpy) == _key(on_jax)


def test_a_float32_field_has_the_same_key_from_numpy_and_jax(model):
    n = len(model.mesh.elements("oxide"))
    values = np.linspace(2.0, 4.0, n).astype(np.float32)
    first = _key(_with_field(model, values))
    assert first == _key(_with_field(model, jnp.asarray(values)))
    assert first == _key(_with_field(model, values.copy()))


def test_negative_zero_keys_as_zero(model):
    n = len(model.mesh.elements("oxide"))
    zeros = np.zeros(n)
    field = gfem.MaterialField(eps_r=np.full(n, 3.9), conductivity=zeros)
    negative = gfem.MaterialField(eps_r=np.full(n, 3.9), conductivity=-zeros)
    assert _key(model.with_material("oxide", field)) == _key(
        model.with_material("oxide", negative)
    )
    scalar = gfem.Material(eps_r=3.9, conductivity=0.0)
    negative_scalar = gfem.Material(eps_r=3.9, conductivity=-0.0)
    assert _key(model.with_material("oxide", scalar)) == _key(
        model.with_material("oxide", negative_scalar)
    )


@pytest.mark.parametrize(
    "placement",
    [
        {"threads": 3},
        {"frequency_workers": 2},
        {"retain_memory": 1},
        {"device": "cuda"},
    ],
)
def test_where_and_how_fast_a_solve_runs_is_not_in_the_key(model, placement):
    options = dataclasses.replace(OPTIONS, **placement)
    assert _key(model, options=options) == _key(model)


def test_the_tolerance_is_in_the_key(model):
    options = dataclasses.replace(OPTIONS, tolerance=1e-8)
    assert _key(model, options=options) != _key(model)


def test_the_frequency_is_in_the_key(model):
    assert _key(model, gfem.Driven(frequency=(1e9,))) != _key(
        model, gfem.Driven(frequency=(2e9,))
    )


def test_a_material_value_is_in_the_key(model):
    changed = model.with_material("oxide", gfem.Material(eps_r=4.0))
    assert _key(changed) != _key(model)


def test_a_moved_node_is_in_the_key(model):
    nodes = model.mesh.nodes.at[0, 2].add(1e-9)
    moved = model.replace(mesh=gfem.Mesh(nodes, model.mesh._topology))
    assert _key(moved) != _key(model)


def test_the_terminal_order_is_in_the_key(model):
    # the capacitance matrix's rows follow the terminals: reversed, it permutes
    reversed_ = gfem.Electrostatic(terminals={"mid": "oxide__nitride", "top": "top"})
    assert _key(model, reversed_) != _key(model)


def _refined(model):
    marked = np.zeros(len(model.mesh.tetrahedra), bool)
    marked[::3] = True
    return refine_model(model, marked)[0]


def test_a_refined_mesh_keys_by_its_arrays(model, tmp_path):
    refined = _refined(model)
    assert refined.mesh.source is None
    key = _key(refined)
    assert key != _key(model)
    path = gfem.Job(refined, PROBLEM, OPTIONS).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path)
    assert back.model.mesh._topology is not refined.mesh._topology
    assert _key(back.model, back.problem, back.options) == key


def test_a_mesh_from_a_file_keys_as_the_same_mesh_in_memory(model, mesh_path):
    topology = dataclasses.replace(model.mesh._topology, source=None)
    in_memory = model.replace(mesh=gfem.Mesh(model.mesh.nodes, topology))
    assert in_memory.mesh.source is None
    mesh_path.write_text("edited after loading")
    assert _key(in_memory) == _key(model)


def test_a_subclassed_material_cannot_be_keyed(model):
    @dataclasses.dataclass(frozen=True)
    class Glass(gfem.Material):
        pass

    with pytest.raises(_codec.UnencodableError, match="Glass"):
        solve_key(model.with_material("oxide", Glass(eps_r=3.9)), PROBLEM, OPTIONS)
    assert issubclass(_codec.UnencodableError, TypeError)
