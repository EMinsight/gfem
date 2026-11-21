"""Deflated GMRES on the live factors and the forward's mode polish.

A stub eigen operator ``T(k) = K - k^2 I`` with ``K`` real symmetric (so
complex symmetric) stands in for the eigen operator: its eigenpairs are
``K``'s, so the deflated solve's answer is known in closed form. Its
"factors" are either the exact inverse of ``T(shift)`` or, to make GMRES
run out of applies, the identity.
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eigen_cases import EPS, line_mesh, shunt_line, shunt_mesh
from finite_differences import derivative

import gfem
from gfem._eigen.adjoint import deflated_solve, polish
from gfem._eigen.kept import KeptOperator
from gfem._eigen.reach import isolated
from gfem._eigen.refine import Refined
from gfem._eigen.rule import EigenmodeRule
from gfem._eigen.search import ModeSearch
from gfem._engine import cache
from gfem._linalg import direct
from gfem._linalg.layout import ONE_PROCESS
from gfem.errors import EigenmodeConvergenceError, LinearSolverConvergenceError


class Stub:
    """``T(k) = K - k^2 I`` on dense matrices, with a choice of factors.

    ``exact``: whether `solve` applies ``T(shift)^-1`` (else the identity),
    a callable of the number of reshifts so far.
    """

    general = False
    layout = ONE_PROCESS

    def __init__(self, lam, *, shift, exact=lambda _: True, seed=0):
        n = len(lam)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        self.lam = np.asarray(lam, dtype=float)
        self.vectors = q
        self.K = q @ np.diag(self.lam) @ q.T
        self.free = np.ones(n, dtype=bool)
        self.shift = complex(shift)
        self.reshifts: list[complex] = []
        self.exact = exact

    def apply(self, k, x):
        return self.K @ x - k * k * x

    def derivative(self, k, x):
        return -2 * k * x

    def solve(self, b):
        if not self.exact(len(self.reshifts)):
            return b
        return np.linalg.solve(self.K - self.shift**2 * np.eye(len(b)), b)

    def reshift(self, k, **_):
        self.shift = complex(k)
        self.reshifts.append(complex(k))

    def rayleigh(self, x, k, *, left=None):  # noqa: ARG002
        k2 = (x @ self.K @ x) / (x @ x)
        root = np.sqrt(complex(k2))
        return root if abs(root - k) <= abs(root + k) else -root

    def backward_error(self, k, x):
        den = (np.linalg.norm(self.K, 2) + abs(k) ** 2) * np.linalg.norm(x)
        return float(np.linalg.norm(self.apply(k, x)) / den)


def _deflated_reference(stub, j, b):
    """``T(k_j) z = P b``, ``z`` free of mode j: the eigenbasis expansion."""
    v = stub.vectors
    k2 = stub.lam[j]
    c = v.T @ b
    c[j] = 0
    others = np.arange(len(c)) != j
    c[others] /= stub.lam[others] - k2
    return v @ c


def test_deflated_solve_accepts_on_rho():
    lam = np.linspace(1.0, 4.0, 40) ** 2
    j = 10
    k = math.sqrt(lam[j])
    stub = Stub(lam, shift=k * (1 + 1e-3))
    x = stub.vectors[:, j] * (0.3 - 0.4j)  # any scale and phase
    b = np.random.default_rng(1).standard_normal(40) + 0j
    solved = deflated_solve(stub, x, None, b, k=k, gap=0.05, mode=0)
    assert solved.rho <= 1e-8
    assert not solved.reshifted
    assert stub.reshifts == []
    assert solved.applies < 60
    np.testing.assert_allclose(solved.z, _deflated_reference(stub, j, b), rtol=1e-8)


def _spread(n=200):
    """Eigenvalues over four decades: GMRES on the identity needs ~n applies."""
    return np.geomspace(1.0, 1e4, n)


@pytest.mark.parametrize(("gap", "offset"), [(math.inf, 1e-3), (0.02, 2e-4)])
def test_reshift_offset_is_a_thousandth_without_a_neighbour(gap, offset):
    lam = _spread()
    j = 100
    k = math.sqrt(lam[j])
    # identity factors until the reshift, exact ones after
    stub = Stub(lam, shift=k * 1.01, exact=lambda shifts: shifts > 0)
    b = np.random.default_rng(2).standard_normal(len(lam)) + 0j
    solved = deflated_solve(stub, stub.vectors[:, j], None, b, k=k, gap=gap, mode=0)
    assert solved.reshifted
    assert stub.reshifts == [pytest.approx(k * (1 + offset), rel=1e-15)]
    # never exactly at k
    assert stub.reshifts[0] != k
    assert solved.rho <= 1e-8
    np.testing.assert_allclose(solved.z, _deflated_reference(stub, j, b), rtol=1e-7)


def test_cap_after_reshift_raises_naming_the_mode():
    lam = _spread()
    j = 100
    k = math.sqrt(lam[j])
    stub = Stub(lam, shift=k * 1.01, exact=lambda _: False)
    b = np.random.default_rng(3).standard_normal(len(lam)) + 0j
    with pytest.raises(LinearSolverConvergenceError, match=r"mode 3.*rho = ") as err:
        deflated_solve(stub, stub.vectors[:, j], None, b, k=k, gap=math.inf, mode=3)
    assert len(stub.reshifts) == 1  # one reshift, then the error
    assert "60" in str(err.value)


def test_polish_corrects_a_mixed_mode_in_one_step():
    lam = np.linspace(1.0, 4.0, 40) ** 2
    j = 10
    stub = Stub(lam, shift=math.sqrt(lam[j]) * 1.2)
    v = stub.vectors
    x = v[:, j] + 1e-6 * v[:, j + 1]  # the other mode's admixture
    x /= np.linalg.norm(x)
    mode = Refined(stub.rayleigh(x, 0), x, stub.backward_error(1, x), 2, True)
    polished = polish(stub, mode, 0.05, index=0)
    assert polished.k == pytest.approx(math.sqrt(lam[j]), rel=1e-14)
    assert abs(abs(polished.x @ v[:, j]) - 1) < 1e-12
    assert np.linalg.norm(polished.x) == pytest.approx(1)
    assert polished.error < 1e-14
    assert polished.iterations == 2  # the refinement's steps, unchanged
    assert stub.reshifts == []


def test_polish_raises_after_three_large_corrections():
    # two nearly equal eigenvalues and an even mix of their modes: each
    # Newton step swings the vector between them by order one
    lam = np.linspace(1.0, 4.0, 40) ** 2
    lam[11] = lam[10] * (1 + 1e-9)
    j = 10
    stub = Stub(lam, shift=math.sqrt(lam[j]) * 1.2)
    v = stub.vectors
    x = (v[:, j] + 0.7 * v[:, j + 1]) / np.linalg.norm([1, 0.7])
    mode = Refined(stub.rayleigh(x, 0), x, 0.0, 1, True)
    with pytest.raises(EigenmodeConvergenceError, match="mode 1"):
        polish(stub, mode, 1e-9, index=1)


def test_a_polish_that_fails_keeps_the_unpolished_mode_and_marks_it():
    # the forward's fallback: the mixed
    # mode of the test above is kept as it came, its neighbour is polished,
    # and the failure is recorded for the reach test (gap 0)
    lam = np.linspace(1.0, 4.0, 40) ** 2
    lam[11] = lam[10] * (1 + 1e-9)
    stub = Stub(lam, shift=math.sqrt(lam[10]) * 1.2)
    v = stub.vectors
    mixed = (v[:, 10] + 0.7 * v[:, 11]) / np.linalg.norm([1, 0.7])
    clean = v[:, 20] + 1e-6 * v[:, 21]
    clean /= np.linalg.norm(clean)
    modes = [
        Refined(stub.rayleigh(mixed, 0), mixed, 1e-12, 1, True),
        Refined(stub.rayleigh(clean, 0), clean, 1e-12, 1, True),
    ]
    # the seed solve gives the two modes and refinement keeps them as they
    # are: the gap set misses the mixed mode's twin, as the cavity's does
    stub.nonpolynomial, stub.seed_damped = (), True  # no real snap
    options = gfem.SolverOptions()
    search = ModeSearch(
        n_modes=2,
        target=1e9,
        eigen_tolerance=options.eigen_tolerance,
        max_refine_iterations=options.max_refine_iterations,
        cluster_rtol=options.cluster_rtol,
        laws=False,
        eigen_solver=lambda *_, **__: (
            np.array([m.k for m in modes]),
            np.stack([m.x for m in modes], axis=1),
        ),
        crowding=lambda: None,
        refine=lambda _op, seed, _x, **_: next(m for m in modes if m.k == seed),
    )
    found = search.find(stub, None, stub.shift, guard=False, polish_isolated=True)
    np.testing.assert_array_equal(found.polish_failed, [True, False])
    assert found.modes[0] is modes[0]
    assert found.modes[1].k == pytest.approx(math.sqrt(lam[20]), rel=1e-14)
    # a failed polish's gap of 0 is out of reach
    reach = isolated(np.array([0, 1]), np.zeros(2, bool), np.array([0.0, 0.1]), 1e-6)
    np.testing.assert_array_equal(reach, [False, True])


# --- the mode-field backward -----------------------------------------------------


class _Defective(Stub):
    """A stub whose ``T'(k)`` is the identity: ``x^T T' x = x^T x``."""

    def derivative(self, k, x):  # noqa: ARG002
        return x


def test_defective_mode_raises_on_a_stub_entry():
    # x = (1, i, 0, ...) has x^T x = 0: its left and right eigenvectors are
    # orthogonal, the adjoint system has no solution (a lossy exceptional
    # point), whatever the mode's gap
    n = 6
    stub = _Defective(np.arange(1.0, n + 1), shift=1.5)
    x = np.zeros((n, 1), dtype=complex)
    x[:2, 0] = (1.0, 1j)
    entry = cache.Entry(
        "em",
        handle=KeptOperator(stub),
        fields={
            "x": x,
            "k": np.array([1.0 + 0.1j]),
            "cluster": np.array([0]),
            "gram": np.zeros((1, 1), complex),
            "incomplete": np.zeros(1, bool),
            "gap": np.array([math.inf]),
            "polish_failed": np.zeros(1, bool),
        },
    )
    rule = EigenmodeRule.__new__(EigenmodeRule)
    rule.options = gfem.SolverOptions()
    rule.problem = gfem.Eigenmode(target=1e9)
    op = SimpleNamespace(built=SimpleNamespace(scale=1e-3))
    with pytest.raises(gfem.DegenerateModeError, match=r"mode.s. 0 .*defective"):
        rule._field_pairs(None, entry, {"bulk": np.ones((1, 1))}, op, None)


@pytest.fixture(scope="module")
def line_mesh_dir(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


def _terminated_field(mesh, eps):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.MaterialField(eps_r=eps)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=1000.0),
            gfem.LumpedPort("p2", "port2", "+z", resistance=1000.0, inductance=1e-9),
        ],
    )


def test_sparsecholesky_backward_matches_pardiso(line_mesh_dir, monkeypatch):
    # NGSolve's sparse Cholesky (LDL^T) as the shift's factorization without
    # MKL: the forward polish and the adjoint iterate on its factors, and the
    # gradient of the ratios agrees with differences as on PARDISO's
    monkeypatch.setattr(direct, "pardiso_inverse", lambda *_, **__: None)
    mesh = line_mesh_dir
    n = len(mesh.elements("dielectric"))
    eps = jnp.asarray(np.linspace(3.5, 4.5, n))
    options = gfem.SolverOptions(eigen_tolerance=1e-13)

    def objective(e):
        result = gfem.solve(
            _terminated_field(mesh, e), gfem.Eigenmode(target=8e9), options=options
        )
        return (
            2.0 / result.q_port["p1"][0]
            + result.participation["p2"][0]
            + result.bulk_participation["dielectric"][0]
        )

    d = np.random.default_rng(4).standard_normal(n)
    d /= np.linalg.norm(d)
    along = float(np.asarray(jax.grad(objective)(eps)) @ d)
    assert along != 0
    np.testing.assert_allclose(along, derivative(objective, eps, d, 1e-2), rtol=1e-6)


@pytest.mark.parametrize("read", ["ratios", "field"])
def test_capacitive_sheet_mode_field_gradient_matches_differences(tmp_path, read):
    # the sheet's share of the mode energy differentiates with the
    # rest of it: a region's share and the sheet participation, and the field
    # normalized to that energy, in eps_r; the mode at 5.6 GHz puts its
    # voltage on the sheet, which stores 6 % of its electric energy
    mesh = shunt_mesh(tmp_path)
    options = gfem.SolverOptions(eigen_tolerance=1e-12)
    probe = np.random.default_rng(6).standard_normal(
        gfem.solve(
            shunt_line(mesh, gfem.Impedance("sheet", capacitance=1e-13)),
            gfem.Eigenmode(target=5.6e9),
        )
        .e_fields[0]
        .coefficients.shape
    )

    def objective(e):
        model = gfem.Model(
            mesh,
            materials={
                "left": gfem.MaterialField(eps_r=e),
                "right": gfem.Material(eps_r=EPS),
            },
            boundaries=[
                gfem.PEC("bottom"),
                gfem.PEC("top"),
                gfem.Impedance("sheet", inductance=1e-9, capacitance=1e-13),
            ],
        )
        result = gfem.solve(model, gfem.Eigenmode(target=5.6e9), options=options)
        if read == "field":
            # squared: the mode's sign rule picks among tied entries of the
            # uniform line field, which a perturbation reorders
            projection = result.e_fields[0].coefficients @ probe
            return jnp.real(jnp.exp(0.7j) * projection**2) * 1e-12
        return (
            result.bulk_participation["left"][0]
            + 0.5 * result.sheet_participation["sheet"][0]
        )

    eps = jnp.full(len(mesh.elements("left")), EPS)
    d = np.random.default_rng(5).standard_normal(len(eps))
    d /= np.linalg.norm(d)
    along = float(np.asarray(jax.grad(objective)(eps)) @ d)
    assert along != 0
    np.testing.assert_allclose(along, derivative(objective, eps, d, 1e-2), rtol=1e-6)
