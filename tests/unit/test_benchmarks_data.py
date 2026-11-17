"""`benchmarks.harness.data`: committed pins, size caps and mesh staging."""

import hashlib
import lzma
import os
from pathlib import Path

import pytest

from benchmarks.harness import case, config, data
from benchmarks.harness.data import Refused

DATA = Path(__file__).resolve().parent / "data"
MESH41 = DATA / "two_box_msh41.msh"
MiB = 1 << 20

CASES = config.known_cases()


# --- the committed cases -----------------------------------------------------------


def test_the_suite_cases_are_committed():
    assert set(CASES) >= {"microstrip", "lcavity", "bragg", "r10M_void"}


@pytest.mark.parametrize("name", CASES)
def test_every_case_file_loads(name):
    c = case.load_case(config.CASES_DIR / name)
    assert c.mesh.source is not None


@pytest.mark.parametrize("name", CASES)
def test_committed_mesh_decompresses_to_its_pin(name):
    c = case.load_case(config.CASES_DIR / name)
    if c.mesh.file is None:
        pytest.skip(f"{name}: external mesh")
    path = config.CASES_DIR / name / c.mesh.file
    raw = lzma.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == c.mesh.sha256
    assert len(raw) == c.mesh.bytes
    assert path.stat().st_size <= MiB
    # the region table is the mesh's $PhysicalNames
    assert c.regions == case.physical_names(path)


def test_committed_cases_hold_at_most_5_mib():
    total = sum(p.stat().st_size for p in config.CASES_DIR.rglob("*") if p.is_file())
    assert total <= 5 * MiB


def test_external_cases_have_no_file_and_nothing_beside_them():
    external = [
        n for n in CASES if case.load_case(config.CASES_DIR / n).mesh.file is None
    ]
    assert "r10M_void" in external
    for name in external:
        assert not list((config.CASES_DIR / name).glob("*.msh*"))


# --- staging -----------------------------------------------------------------------


def _pinned_case(tmp_path: Path, *, max_committed: int | None = None) -> case.Case:
    if max_committed is not None:
        mp = pytest.MonkeyPatch()
        mp.setattr(case, "MAX_COMMITTED_BYTES", max_committed)
    try:
        case.pin(tmp_path / "cases" / "tiny", MESH41, "um", [tmp_path / "store"])
    finally:
        if max_committed is not None:
            mp.undo()
    raw = (tmp_path / "cases" / "tiny" / "case.toml").read_text()
    (tmp_path / "cases" / "tiny" / "case.toml").write_text(
        raw.replace(
            'name = "tiny"', 'name = "tiny"\nsize = "small"\nfamily = "tiny"\norder = 1'
        )
        + '[problem]\nkind = "eigen"\ntarget_hz = 1e9\nn_modes = 1\n'
        "[quantities]\nmodes = 1\n"
    )
    return case.load_case(tmp_path / "cases" / "tiny")


def test_staging_a_committed_mesh_gives_a_read_only_verified_copy(tmp_path):
    c = _pinned_case(tmp_path)
    staged = data.stage(c, tmp_path / "data", [])
    assert staged == tmp_path / "data" / "meshes" / f"{c.mesh.sha256}.msh"
    assert staged.read_bytes() == MESH41.read_bytes()
    assert not os.access(staged, os.W_OK)
    assert data.stage(c, tmp_path / "data", []) == staged  # cached
    data.verify_staged(staged, c.mesh.sha256)


def test_staging_an_external_mesh_searches_the_stores_in_order(tmp_path):
    c = _pinned_case(tmp_path, max_committed=100)
    assert c.mesh.file is None
    stores = [tmp_path / "empty", tmp_path / "store"]
    staged = data.stage(c, tmp_path / "data", stores)
    assert staged.read_bytes() == MESH41.read_bytes()


def test_a_missing_mesh_is_refused_naming_the_stores(tmp_path):
    c = _pinned_case(tmp_path, max_committed=100)
    stores = [tmp_path / "a", tmp_path / "b"]
    with pytest.raises(Refused) as error:
        data.stage(c, tmp_path / "data", stores)
    assert str(tmp_path / "a") in str(error.value)
    assert str(tmp_path / "b") in str(error.value)


def test_a_wrong_pin_is_refused_and_quarantined(tmp_path):
    c = _pinned_case(tmp_path)
    xz = tmp_path / "cases" / "tiny" / "mesh.msh.xz"
    xz.write_bytes(lzma.compress(b"$MeshFormat\nnot the pinned mesh\n"))
    got = hashlib.sha256(b"$MeshFormat\nnot the pinned mesh\n").hexdigest()
    with pytest.raises(Refused, match=f"gives {got[:12]}, pinned {c.mesh.sha256[:12]}"):
        data.stage(c, tmp_path / "data", [])
    meshes = tmp_path / "data" / "meshes"
    assert not (meshes / f"{c.mesh.sha256}.msh").exists()
    assert (meshes / f"{c.mesh.sha256}.bad-{got[:12]}").exists()


def test_a_staged_file_that_no_longer_verifies_is_moved_aside(tmp_path):
    c = _pinned_case(tmp_path)
    staged = data.stage(c, tmp_path / "data", [])
    staged.chmod(0o644)
    staged.write_bytes(b"changed")
    got = hashlib.sha256(b"changed").hexdigest()
    with pytest.raises(Refused, match="gives"):
        data.verify_staged(staged, c.mesh.sha256)
    assert not staged.exists()
    assert (staged.parent / f"{c.mesh.sha256}.bad-{got[:12]}").exists()
    # prepare restages it from the source
    assert data.stage(c, tmp_path / "data", []).read_bytes() == MESH41.read_bytes()
