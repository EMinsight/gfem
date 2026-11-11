"""Piece values outside material-field regions, kept across designs (ADR 0082).

A CPU direct driven solve assembles each volume piece as its part on the
`MaterialField` regions plus the part on the other elements; the latter is
kept per discretization behind a digest of its coefficient, so a density
loop assembles it once. S stays within summation round-off of the whole
assembly (the reference here: the same values as a plain `Material`, which
has no material-field region and so no split).
"""

import types

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine.system_operator import FixedPieces
from gfem._linalg import pieces

F = (21e9, 24e9)
DIRECT = gfem.SolverOptions(linear_solver="direct")
#: The volume pieces of the line below: curl_curl and mass.
VOLUME_PIECES = 2


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    out = tmp_path_factory.mktemp("line") / "line.msh"
    return sm.write_msh22(out, sm.strip_line(3.0, 2.0, 1.0, (6, 2, 1), core=(1.0, 2.0)))


def _model(path, core, dielectric=None):
    # a new mesh per model: a new topology, so a discretization of its own
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    if not isinstance(core, gfem.Material):
        core = gfem.MaterialField(eps_r=np.full(len(mesh.elements("core")), core))
    return gfem.Model(
        mesh,
        materials={"dielectric": dielectric or gfem.Material(eps_r=2.0), "core": core},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _s(model, options=DIRECT):
    return np.asarray(
        gfem.solve(model, gfem.Driven(frequency=F), options=options).s.values
    )


@pytest.fixture
def keeps(monkeypatch):
    """Counts the fixed-part assemblies (one per kept piece part)."""
    calls = []
    original = pieces.FixedPart.keep

    def keep(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pieces.FixedPart, "keep", keep)
    return calls


def test_density_loop_assembles_the_fixed_part_once(path, keeps):
    base = _model(path, 3.0)
    for eps in (3.0, 3.5, 4.0):
        field = gfem.MaterialField(eps_r=np.full(len(base.mesh.elements("core")), eps))
        got = _s(base.with_material("core", field))
        want = _s(_model(path, gfem.Material(eps_r=eps)))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-11 * np.abs(want).max())
    assert sorted(keeps) == ["curl_curl", "mass"]


def test_a_changed_fixed_material_rebuilds_the_fixed_part(path, keeps):
    model = _model(path, 3.0)
    for _ in range(3):
        _s(model)
    assert len(keeps) == VOLUME_PIECES
    changed = model.with_material("dielectric", gfem.Material(eps_r=2.5))
    want = _s(
        _model(path, gfem.Material(eps_r=3.0), dielectric=gfem.Material(eps_r=2.5))
    )
    for _ in range(3):
        got = _s(changed)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-11 * np.abs(want).max())
    # only the mass reads eps_r: the curl_curl part is still the kept one
    assert sorted(keeps) == ["curl_curl", "mass", "mass"]


def test_a_model_without_material_field_keeps_nothing(path, keeps):
    model = _model(path, gfem.Material(eps_r=3.0))
    for _ in range(3):
        _s(model)
    assert keeps == []


def test_iterative_solves_keep_nothing(path, keeps):
    model = _model(path, 3.0)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    for _ in range(3):
        _s(model, options)
    assert keeps == []


def test_equal_values_on_another_region_layout_key_another_fixed_part():
    # four elements of value 1: the same values outside either layout's regions
    store = pieces.KeptParts()
    values = np.ones(4)

    def split(outside):
        piece = types.SimpleNamespace(name="mass", values=values, integrand="whole")
        varying = types.SimpleNamespace(name="mass", values=values, integrand="part")
        fixed = FixedPieces(store, np.asarray(outside), lambda: [varying])
        (part,) = fixed.split([piece])[1]
        return part.digest

    assert split([True, True, False, False]) != split([False, False, True, True])


def test_split_pieces_count_their_shared_positions_once():
    import ngsolve as ngs
    from netgen.csg import unit_cube

    mesh = ngs.Mesh(unit_cube.GenerateMesh(maxh=0.5))
    space = ngs.H1(mesh, order=1, complex=True)
    region = ngs.IfPos(ngs.x - 0.5, 1, 0)
    store = pieces.KeptParts()

    def build():
        fixed = [
            pieces.FixedPart(
                store, "stiffness", "d", lambda u, v: ngs.grad(u) * ngs.grad(v) * ngs.dx
            ),
            pieces.FixedPart(store, "mass", "d", lambda u, v: u * v * ngs.dx),
        ]
        varying = [
            lambda u, v: region * ngs.grad(u) * ngs.grad(v) * ngs.dx,
            lambda u, v: region * u * v * ngs.dx,
        ]
        assembled = pieces.AssembledPieces(space, varying, fixed=fixed)
        return assembled, assembled.pattern

    build()  # the digest's first sight: nothing kept
    assembled, pattern = build()
    nze = pattern.nze
    # the kept parts are the store's; the two varying parts' complex values,
    # and their one int64 positions array
    assert assembled.nbytes == 2 * 16 * nze + 8 * nze
