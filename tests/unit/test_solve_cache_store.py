"""The solve cache's store: entries written, read back, removed and evicted."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._solve_cache import Store, solve_key

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
OPTIONS = gfem.SolverOptions(order=1)
ENV = {"gfem": gfem.__version__, "gfem_code": "0" * 64, "ngsolve": "x"}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """An electrostatic model, its result and solve key."""
    path = tmp_path_factory.mktemp("stack") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    model = gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)
    result = gfem.solve(model, ELECTROSTATIC, options=OPTIONS)
    return model, result, solve_key(model, ELECTROSTATIC, OPTIONS)[0]


def _run(options=OPTIONS):
    return {"device": "cpu", "options": options, "ranks": 1, "wall_time_s": 0.5}


def _assert_bit_equal(a, b):
    assert type(a) is type(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        assert type(x) is type(y)
        assert np.asarray(x).dtype == np.asarray(y).dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_an_electrostatic_result_reads_back_bit_equal_on_the_model_mesh(
    stack, tmp_path
):
    model, result, key = stack
    store = Store(tmp_path / "cache")
    store.write(key, model, ENV, result, _run(), ())
    back, run, recorded = store.read(key, model)
    _assert_bit_equal(back, result)
    assert all(f.mesh is model.mesh for f in back.potential.values())
    assert run["options"] == OPTIONS
    assert run["ranks"] == 1
    assert recorded == ()
    entry = json.loads((tmp_path / "cache" / key / "entry.json").read_text())
    assert entry["format"] == "gfem solve cache entry"
    assert entry["key"] == key
    assert entry["env"] == ENV


def test_a_driven_result_with_kept_fields_reads_back_bit_equal(tmp_path):
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (4, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    problem = gfem.Driven(frequency=(2e9, 4e9), keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    key = solve_key(model, problem, OPTIONS)[0]
    recorded = ((gfem.MeshQualityWarning, "a thin element"),)
    store = Store(tmp_path / "cache")
    assert store.write(key, model, ENV, result, _run(), recorded) == store.root / key
    back, _, replay = store.read(key, model)
    _assert_bit_equal(back, result)
    assert back.e_fields
    assert all(f.mesh is model.mesh for f in back.e_fields)
    assert replay == recorded
    assert (store.root / key / "arrays.npz").is_file()


def test_a_new_directory_becomes_a_solve_cache(tmp_path):
    root = tmp_path / "a" / "cache"
    Store(root)
    marker = json.loads((root / "solve-cache.json").read_text())
    assert marker == {
        "format": "gfem solve cache",
        "layout": 1,
        "max_bytes": 10 * 2**30,
    }
    assert (
        (root / "CACHEDIR.TAG")
        .read_text()
        .startswith("Signature: 8a477f597d28d172789f06886806bc55")
    )
    assert (root / ".gitignore").read_text() == "*\n"
    Store(root)  # opens again


def test_an_empty_directory_becomes_a_solve_cache(tmp_path):
    Store(tmp_path)
    assert (tmp_path / "solve-cache.json").is_file()


def test_a_directory_that_is_not_a_cache_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="is not a gfem solve cache"):
        Store(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_a_cache_of_another_layout_warns_and_stores_nothing(stack, tmp_path):
    model, result, key = stack
    marker = {"format": "gfem solve cache", "layout": 2, "max_bytes": 10}
    (tmp_path / "solve-cache.json").write_text(json.dumps(marker))
    with pytest.warns(gfem.SolveCacheWarning, match="another layout") as caught:
        store = Store(tmp_path)
    assert caught[0].filename == __file__
    assert not store.usable
    assert store.write(key, model, ENV, result, _run(), ()) == (
        "a solve cache of another layout"
    )
    assert store.read(key, model) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["solve-cache.json"]


def _filled(stack, root):
    model, result, key = stack
    store = Store(root)
    store.write(key, model, ENV, result, _run(), ())
    return store


def test_a_truncated_arrays_file_is_removed_and_reported(stack, tmp_path):
    model, _, key = stack
    store = _filled(stack, tmp_path)
    npz = tmp_path / key / "arrays.npz"
    npz.write_bytes(npz.read_bytes()[: npz.stat().st_size // 2])
    with pytest.warns(gfem.SolveCacheWarning, match="could not be read") as caught:
        assert store.read(key, model) is None
    assert caught[0].filename == __file__
    assert not (tmp_path / key).exists()
    assert list((tmp_path / ".tmp").iterdir()) == []


def test_an_entry_with_an_edited_key_is_removed_and_reported(stack, tmp_path):
    model, _, key = stack
    store = _filled(stack, tmp_path)
    path = tmp_path / key / "entry.json"
    path.write_text(path.read_text().replace(key, "f" * 64))
    with pytest.warns(gfem.SolveCacheWarning, match="removed"):
        assert store.read(key, model) is None
    assert not (tmp_path / key).exists()


def test_an_entry_on_another_mesh_is_removed_and_reported(stack, tmp_path):
    _, _, key = stack
    store = _filled(stack, tmp_path)
    path = sm.write_msh22(tmp_path / "other.msh", sm.plate(4.0, 3.0, LAYERS, (2, 1, 1)))
    other = gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)
    with pytest.warns(gfem.SolveCacheWarning, match="topology"):
        assert store.read(key, other) is None
    assert not (tmp_path / key).exists()


def test_a_missing_entry_is_a_silent_miss(stack, tmp_path):
    model, _, _ = stack
    assert Store(tmp_path).read("0" * 64, model) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_read_only_cache_still_reads_and_warns_on_write(stack, tmp_path):
    model, result, key = stack
    store = _filled(stack, tmp_path)
    for path in (tmp_path / ".tmp", tmp_path):
        path.chmod(0o555)
    try:
        back, _, _ = store.read(key, model)
        _assert_bit_equal(back, result)
        with pytest.warns(gfem.SolveCacheWarning, match="could not be stored"):
            reason = store.write("1" * 64, model, ENV, result, _run(), ())
        assert reason.startswith("the write failed")
    finally:
        for path in (tmp_path, tmp_path / ".tmp"):
            path.chmod(0o755)
    assert not (tmp_path / ("1" * 64)).exists()


def _cap(root, max_bytes):
    marker = json.loads((root / "solve-cache.json").read_text())
    (root / "solve-cache.json").write_text(
        json.dumps(marker | {"max_bytes": max_bytes})
    )


def test_a_small_cap_evicts_the_least_recently_used_first(stack, tmp_path):
    model, result, _ = stack
    Store(tmp_path)
    keys = [c * 64 for c in "abc"]
    store = Store(tmp_path)
    for i, key in enumerate(keys):
        store.write(key, model, ENV, result, _run(), ())
        os.utime(tmp_path / key / "entry.json", (1e9 + i, 1e9 + i))
    size = json.loads((tmp_path / keys[0] / "entry.json").read_text())["bytes"]
    assert store.read(keys[0], model) is not None  # "a" is now the most recent
    _cap(tmp_path, 3 * size)
    Store(tmp_path).write("d" * 64, model, ENV, result, _run(), ())
    left = sorted(p.name[0] for p in tmp_path.iterdir() if len(p.name) == 64)
    assert left == ["a", "c", "d"]
    assert list((tmp_path / ".tmp").iterdir()) == []


def test_a_result_over_the_cap_is_not_stored(stack, tmp_path):
    model, result, key = stack
    Store(tmp_path)
    _cap(tmp_path, 1000)
    with pytest.warns(gfem.SolveCacheWarning, match="larger than") as caught:
        reason = Store(tmp_path).write(key, model, ENV, result, _run(), ())
    assert caught[0].filename == __file__
    assert "max_bytes" in reason
    assert not (tmp_path / key).exists()
    assert list((tmp_path / ".tmp").iterdir()) == []


def test_old_temporary_entries_are_swept_after_a_write(stack, tmp_path):
    model, result, key = stack
    store = Store(tmp_path)
    crashed = tmp_path / ".tmp" / f"{key}.x"
    crashed.mkdir()
    fresh = tmp_path / ".tmp" / f"{key}.y"
    fresh.mkdir()
    os.utime(crashed, (1e9, 1e9))
    store.write(key, model, ENV, result, _run(), ())
    assert sorted(p.name for p in (tmp_path / ".tmp").iterdir()) == [fresh.name]


def test_eviction_leaves_files_that_are_not_entries_alone(stack, tmp_path):
    model, result, _ = stack
    store = Store(tmp_path)
    store.write("e" * 64, model, ENV, result, _run(), ())
    (tmp_path / "README").write_text("mine")
    (tmp_path / ("A" * 64)).mkdir()
    size = json.loads((tmp_path / ("e" * 64) / "entry.json").read_text())["bytes"]
    os.utime(tmp_path / ("e" * 64) / "entry.json", (1e9, 1e9))
    _cap(tmp_path, size)
    Store(tmp_path).write("f" * 64, model, ENV, result, _run(), ())
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith(".")) == [
        "A" * 64,
        "CACHEDIR.TAG",
        "README",
        "f" * 64,
        "solve-cache.json",
    ]


_PUBLISH = """
import os, sys, time, pickle
from pathlib import Path
import gfem
from gfem import _solve_cache

root, barrier, me, mesh = sys.argv[1:5]
model = gfem.Model(gfem.Mesh.from_gmsh(mesh), materials={
    "oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)})
problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
options = gfem.SolverOptions(order=1)
result = gfem.solve(model, problem, options=options)
key = _solve_cache.solve_key(model, problem, options)[0]
rename = os.rename

def meet(src, dst):  # both writers reach the rename before either renames
    if Path(src).parent.name == ".tmp":
        Path(barrier, me).touch()
        while len(os.listdir(barrier)) < 2:
            time.sleep(0.01)
    rename(src, dst)

_solve_cache.os.rename = meet
run = {"device": "cpu", "options": options, "ranks": 1, "wall_time_s": 0.0}
print(_solve_cache.Store(root).write(key, model, {}, result, run, ()))
"""


def test_two_processes_publishing_one_key_leave_one_complete_entry(stack, tmp_path):
    model, result, key = stack
    root, barrier = tmp_path / "cache", tmp_path / "barrier"
    barrier.mkdir()
    Store(root)
    src = Path(gfem.__file__).parent.parent
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(src), *sys.path])}
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PUBLISH,
                str(root),
                str(barrier),
                me,
                str(model.mesh.source),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for me in "ab"
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err for _, err in outs]
    assert [out.strip() for out, _ in outs] == [str(root / key)] * 2
    assert [p.name for p in root.iterdir() if len(p.name) == 64] == [key]
    assert list((root / ".tmp").iterdir()) == []
    back, _, _ = Store(root).read(key, model)
    # the entry was solved in another process, whose thread count may differ:
    # equal to the precision the solve key promises, not bitwise
    assert type(back) is type(result)
    for x, y in zip(
        jax.tree_util.tree_leaves(back), jax.tree_util.tree_leaves(result), strict=True
    ):
        want = np.asarray(y)
        scale = float(np.abs(want).max()) if want.size else 0.0
        np.testing.assert_allclose(np.asarray(x), want, rtol=1e-12, atol=1e-12 * scale)


def test_a_result_the_store_cannot_write_warns_and_stores_nothing(stack, tmp_path):
    model, _, key = stack

    class Unwritable:
        pass

    store = Store(tmp_path)
    with pytest.warns(gfem.SolveCacheWarning, match="could not be stored") as caught:
        reason = store.write(key, model, ENV, Unwritable(), _run(), ())
    assert caught[0].filename == __file__
    assert "Unwritable" in reason
    assert not (tmp_path / key).exists()
    assert list((tmp_path / ".tmp").iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_damaged_entry_that_cannot_be_removed_is_reported_as_left(stack, tmp_path):
    model, _, key = stack
    store = _filled(stack, tmp_path)
    (tmp_path / key / "entry.json").write_text("{")
    tmp_path.chmod(0o555)
    try:
        with pytest.warns(gfem.SolveCacheWarning, match="could not be read") as caught:
            assert store.read(key, model) is None
    finally:
        tmp_path.chmod(0o755)
    message = str(caught[0].message)
    assert "was removed" not in message
    assert "could not be removed" in message
    assert (tmp_path / key).exists()


def test_damaged_entries_count_toward_the_cap_and_are_evicted(stack, tmp_path):
    model, result, _ = stack
    store = Store(tmp_path)
    store.write("a" * 64, model, ENV, result, _run(), ())
    size = json.loads((tmp_path / ("a" * 64) / "entry.json").read_text())["bytes"]
    damaged = tmp_path / ("b" * 64)
    damaged.mkdir()
    (damaged / "entry.json").write_text("{")
    (damaged / "arrays.npz").write_bytes(b"\0" * size)
    os.utime(tmp_path / ("a" * 64) / "entry.json", (1e9, 1e9))
    for path in (damaged / "entry.json", damaged):  # the least recently used
        os.utime(path, (9e8, 9e8))
    _cap(tmp_path, 2 * size)
    Store(tmp_path).write("c" * 64, model, ENV, result, _run(), ())
    left = sorted(p.name[0] for p in tmp_path.iterdir() if len(p.name) == 64)
    assert left == ["a", "c"]


def test_a_field_on_a_mesh_in_other_units_is_not_read_back_on_the_model_mesh(
    stack, tmp_path
):
    model, result, key = stack
    in_mm = gfem.Mesh.from_gmsh(model.mesh.source, length_unit="mm")
    # moved onto the model's node positions: same elements, same nodes
    in_mm = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(in_mm), [model.mesh.nodes]
    )
    moved = dataclasses.replace(
        result,
        potential={
            name: dataclasses.replace(field, mesh=in_mm)
            for name, field in result.potential.items()
        },
    )
    store = Store(tmp_path)
    store.write(key, model, ENV, moved, _run(), ())
    back, _, _ = store.read(key, model)
    assert all(f.mesh is not model.mesh for f in back.potential.values())
    assert all(f.mesh.length_unit == "mm" for f in back.potential.values())
