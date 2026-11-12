"""Deflated GMRES on the live factors and the forward's mode polish (ADR 0104).

A stub eigen operator ``T(k) = K - k^2 I`` with ``K`` real symmetric (so
complex symmetric) stands in for the eigen operator: its eigenpairs are
``K``'s, so the deflated solve's answer is known in closed form. Its
"factors" are either the exact inverse of ``T(shift)`` or, to make GMRES
run out of applies, the identity.
"""

import math

import numpy as np
import pytest

from gfem._eigen.adjoint import deflated_solve, polish
from gfem._eigen.refine import Refined
from gfem.errors import EigenmodeConvergenceError, LinearSolverConvergenceError


class Stub:
    """``T(k) = K - k^2 I`` on dense matrices, with a choice of factors.

    ``exact``: whether `solve` applies ``T(shift)^-1`` (else the identity),
    a callable of the number of reshifts so far.
    """

    general = False

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
