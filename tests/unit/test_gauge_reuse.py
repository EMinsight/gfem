r"""Reused direct factors under the tree-cotree gauge stop in the field's norm.

ADR 0094 and 0216. On the parallel-plate line (12 280 free DOFs), a gauged
direct handle refactorized at one permittivity solves a nearby one on the
factors it holds. Its GMRES stops on the field :math:`X = T (d \circ \hat Y)`,
so the field is within a twentieth of the tolerance of the exact gauged solve;
the scaled gauge coordinates :math:`\hat Y` alone stop up to 50 times early.
"""

import importlib.metadata
import math

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import driven, gauge
from gfem._engine.port_network import PortNetwork
from gfem._linalg import Operator, direct
from gfem._linalg.direct import DirectHandle
from gfem._linalg.factor_reuse import REUSED_TOLERANCE, AmortizedRule
from gfem._linalg.gauge import GaugedHandle, GaugedPieces
from gfem._linalg.pieces import AssembledPieces, PieceSum

L, W, H = 10.0, 2.0, 1.0  # mm
TOLERANCE = 1e-6
FREQUENCY = 1e9


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    """The driven operator of the line, its gauge basis and its gauged pieces."""
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 4, 4)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    op = driven.driven_operator(model, 2)
    network = PortNetwork.build(model, op, [], columns=[0, 1], n_freq=1)
    loads = np.asarray(network.rhs(2 * math.pi * FREQUENCY))
    basis = gauge.gauge_basis(op)
    pieces = GaugedPieces(
        AssembledPieces(
            op.space,
            [piece.integrand for piece in op.pieces],
            symmetric=op.symmetric,
        ),
        basis,
        [piece.kind for piece in op.pieces],
    )
    return op, basis, pieces, loads


def _mkl_installed() -> bool:
    try:
        importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_mkl = pytest.mark.skipif(not _mkl_installed(), reason="needs the mkl extra")


@pytest.fixture(params=[pytest.param("pardiso", marks=needs_mkl), "sparsecholesky"])
def backend(request, monkeypatch):
    """The direct handle's factorization: PARDISO, or the fallback without MKL."""
    if request.param == "sparsecholesky":
        monkeypatch.setattr(direct, "pardiso_inverse", lambda *_, **__: None)
    return request.param


@pytest.fixture
def forced(monkeypatch):
    """The reuse gate open and the cap lifted, whatever the plan's ratio."""
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
    monkeypatch.setattr(AmortizedRule, "cap", property(lambda _: 100))


def _operator(line, scale):
    """The gauged operator at ``FREQUENCY`` with the permittivity scaled."""
    op, basis, pieces, _ = line
    omega = 2 * math.pi * FREQUENCY
    weights = [complex(piece.theta(omega)) for piece in op.pieces]
    kinds = [piece.kind for piece in op.pieces]
    weights[kinds.index("value")] *= scale
    return Operator(op.form(omega), basis.free, PieceSum(pieces, tuple(weights)))


def _field(line, handle, operator, B):
    """The driven field of ``B`` through the gauged ``handle`` at ``operator``."""
    op, basis, pieces, _ = line
    scaling = pieces.scaling(operator.piece_sum.weights)
    wrapped = GaugedHandle(handle, basis, scaling, ndof=op.space.ndof, borrow=True)
    return wrapped.solve_counted(B)


@pytest.mark.usefixtures("backend", "forced")
@pytest.mark.parametrize("delta", [1e-3, 1e-1])
def test_a_reused_gauged_step_gives_the_field_to_a_twentieth_of_the_tolerance(
    line, delta
):
    B = line[3]
    start, near = _operator(line, 1.0), _operator(line, 1.0 + delta)
    handle = DirectHandle.from_operator(
        _operator(line, 2.0), "complex_symmetric", tolerance=TOLERANCE
    )
    handle.refactorize(start, frequency=FREQUENCY)
    _field(line, handle, start, B)
    handle.refactorize(near, frequency=FREQUENCY)
    assert handle.last_action == "reused"
    X, counts = _field(line, handle, near, B)
    assert all(c > 0 for c in counts)
    exact_handle = DirectHandle.from_operator(near, "complex_symmetric")
    exact, _ = _field(line, exact_handle, near, B)
    errors = np.linalg.norm(X - exact, axis=0) / np.linalg.norm(exact, axis=0)
    assert errors.max() <= REUSED_TOLERANCE * TOLERANCE
    handle.free()
    exact_handle.free()
