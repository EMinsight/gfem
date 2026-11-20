"""The eigen pipeline's products on a layout whose DOFs have several holders.

`Ghosts` is a one-process stand-in for a rank layout: some DOFs appear in the
column more than once, as a shared DOF does on each rank that holds it. A state
column repeats the DOF's value on every copy; a load column splits it between
them, so only the layout's products (`gfem._linalg.layout`) give the
one-process value. The eigen operator, its refinement, the polish and the
selection's span tests on ghosted columns must give the plain problem's
results, as they will on ranks (ticket 08 Q3).
"""

import numpy as np
import pytest

from gfem._eigen.adjoint import deflated_solve, polish
from gfem._eigen.operator import EigenOperator, NonPolynomial
from gfem._eigen.refine import Refined, refine
from gfem._eigen.rule import _in_span, _orthonormal_multiples, _outside, _same
from gfem._linalg.layout import ONE_PROCESS, RankLayout

N = 14
#: The extra copies: DOF 4 has three holders, DOFs 1 and 9 two.
GHOSTS = np.array([1, 4, 4, 9])
#: DOF 0 is fixed (a PEC DOF).
FIXED = 0


class Ghosts(RankLayout):
    """``N`` DOFs, each copy in `GHOSTS` held a second time; no communicator."""

    def __init__(self) -> None:
        self.comm = None
        self.of = np.concatenate([np.arange(N), GHOSTS])
        self.owners = np.arange(len(self.of)) < N
        self._dtype = np.complex128

    def state(self, x: np.ndarray) -> np.ndarray:
        """The ghosted state of a plain column (each copy holds the value)."""
        return x[self.of]

    def load(self, b: np.ndarray) -> np.ndarray:
        """The ghosted load of a plain column: split evenly over the copies."""
        holders = np.bincount(self.of, minlength=N)
        return b[self.of] / holders[self.of]

    def gather(self, load: np.ndarray) -> np.ndarray:
        """The plain column of a ghosted load (the copies summed)."""
        out = np.zeros(N, dtype=complex)
        np.add.at(out, self.of, load)
        return out

    def plain(self, state: np.ndarray) -> np.ndarray:
        """The plain column of a ghosted state."""
        return state[:N]

    def exchange(self, load: np.ndarray) -> np.ndarray:
        return self.gather(load)[self.of]


GHOSTED = Ghosts()


def _matrices():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((N, N))
    stiffness = a @ a.T + N * np.eye(N)
    # the mass and damping pieces' signs of a wave problem: T = K - k^2 |M| + ...
    mass = -np.diag(1 + rng.random(N))
    damping = 0.05j * np.diag(rng.random(N))
    surface = np.diag(rng.random(N))
    constant = np.diag(0.1 * rng.random(N))
    return stiffness, mass, damping, surface, constant


class _Handle:
    """``T(k)`` applied and solved densely, its columns in a layout's roles."""

    def __init__(self, t: np.ndarray, roles) -> None:
        self.t, self.roles = t, roles

    def apply(self, x, *, transpose=False):  # noqa: ARG002 - complex symmetric
        return self.roles.load(self.t @ self.roles.plain(x))

    def solve(self, b, *, transpose=False):  # noqa: ARG002 - complex symmetric
        return self.roles.state(np.linalg.solve(self.t, self.roles.gather(b)))

    def free(self):
        pass


class _Plain:
    """The plain layout's column roles: every map is the identity."""

    @staticmethod
    def state(x):
        return x

    load = gather = plain = state


def _operator(layout, *, numbers=None, shift=3.0) -> EigenOperator:
    """``T(k) = K + k D + k^2 M + sqrt(k) A`` on the free DOFs of ``layout``.

    Its modes lie near ``k = 3`` (``K`` of order ``N``, ``|M|`` of 1 to 2).
    """
    stiffness, mass, damping, surface, constant = _matrices()
    roles = _Plain() if layout is ONE_PROCESS else layout
    free_plain = np.arange(N) != FIXED
    keep = np.outer(free_plain, free_plain)
    stiffness = np.where(keep, stiffness, np.eye(N))

    def linear(matrix):
        def apply(x, *, transpose=False):  # noqa: ARG001 - symmetric
            return roles.load(matrix @ roles.plain(x))

        return apply

    def t_of(k):
        return (
            stiffness
            + k * np.where(keep, damping, 0)
            + k * k * np.where(keep, mass, 0)
            + np.sqrt(k) * np.where(keep, surface, 0)
        )

    theta = NonPolynomial(
        "surface", linear(surface), np.sqrt, lambda k: 0.5 / np.sqrt(k)
    )
    return EigenOperator(
        shift=complex(shift),
        operator=t_of,
        # a dense stand-in for a handle: apply, solve and free only
        prepare=lambda t, solves=0: _Handle(t, roles),  # noqa: ARG005  # ty: ignore[invalid-argument-type]
        damping=linear(damping),
        mass=linear(mass),
        free=roles.state(free_plain),
        scale=1.0,
        nonpolynomial=(theta,),
        law_constant=linear(constant),
        layout=layout,
        numbers=numbers,
    )


def _pair():
    """The plain operator and its ghosted twin, their norm starts alike."""
    plain = _operator(ONE_PROCESS, numbers=np.arange(N))
    ghosted = _operator(GHOSTED, numbers=GHOSTED.of)
    return plain, ghosted


def _vector(seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x[FIXED] = 0
    return x


def test_the_operators_functionals_are_the_plain_ones_on_ghosted_columns():
    plain, ghosted = _pair()
    x, k = _vector(), 2.9 + 0.01j
    gx = GHOSTED.state(x)
    assert ghosted.rayleigh(gx, k) == pytest.approx(plain.rayleigh(x, k), rel=1e-13)
    assert ghosted.curl_ratio(k, gx) == pytest.approx(plain.curl_ratio(k, x), rel=1e-13)
    np.testing.assert_allclose(ghosted.norms, plain.norms, rtol=1e-13)
    assert ghosted.backward_error(k, gx) == pytest.approx(
        plain.backward_error(k, x), rel=1e-12
    )


def test_the_norm_start_is_a_state_of_the_one_process_numbers():
    # the ghosted start repeats each DOF's value on its copies, so a split of
    # the same DOFs gives the plain estimates
    plain, ghosted = _pair()
    np.testing.assert_allclose(ghosted.norms, plain.norms, rtol=1e-13)


def _same_mode(a: Refined, b: Refined, *, plain_x: np.ndarray) -> None:
    assert a.k == pytest.approx(b.k, rel=1e-12)
    assert a.iterations == b.iterations
    overlap = abs(np.vdot(plain_x, b.x[:N]))
    assert overlap == pytest.approx(1.0, abs=1e-10)


def test_refinement_on_ghosted_columns_finds_the_plain_mode():
    plain, ghosted = _pair()
    x0 = _vector(5)
    first = refine(plain, 3.0, x0, tolerance=1e-13, max_iterations=30)
    second = refine(ghosted, 3.0, GHOSTED.state(x0), tolerance=1e-13, max_iterations=30)
    assert first.converged
    _same_mode(first, second, plain_x=first.x)
    # the refined mode is a state: every copy holds its DOF's value
    np.testing.assert_array_equal(second.x, GHOSTED.state(second.x[:N]))


def test_polish_on_ghosted_columns_gives_the_plain_mode():
    plain, ghosted = _pair()
    mode = refine(plain, 3.0, _vector(5), tolerance=1e-13, max_iterations=30)
    rough = mode.x + 1e-3 * _vector(8)
    unpolished = Refined(mode.k, rough, 1e-3, mode.iterations, True)
    first = polish(plain, unpolished, 0.5, index=0)
    second = polish(
        ghosted,
        Refined(mode.k, GHOSTED.state(rough), 1e-3, mode.iterations, True),
        0.5,
        index=0,
    )
    assert first.k == pytest.approx(second.k, rel=1e-12)
    np.testing.assert_allclose(second.x[:N], first.x, atol=1e-10)


def test_a_deflated_solve_on_ghosted_columns_gives_the_plain_solution():
    plain, ghosted = _pair()
    mode = refine(plain, 3.0, _vector(5), tolerance=1e-13, max_iterations=30)
    b = _vector(9)
    first = deflated_solve(plain, mode.x, None, b, k=mode.k, gap=0.5, mode=0)
    second = deflated_solve(
        ghosted,
        GHOSTED.state(mode.x),
        None,
        GHOSTED.load(b),
        k=mode.k,
        gap=0.5,
        mode=0,
    )
    assert first.applies == second.applies
    np.testing.assert_allclose(second.z[:N], first.z, atol=1e-10 * abs(first.z).max())


def _modes(k, columns):
    return [
        Refined(complex(kk), c, 0.0, 0, True)
        for kk, c in zip(k, columns.T, strict=True)
    ]


def test_the_selections_span_tests_read_ghosted_columns_as_plain_ones():
    rng = np.random.default_rng(4)
    basis = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))
    k = np.array([5.0, 5.0 * (1 + 1e-12)])
    inside = basis @ np.array([0.3, -0.7j])
    twin = inside + 0.5 * _vector(6)
    accepted = _modes(k, basis)
    ghosted = _modes(k, GHOSTED.state(basis))

    def mode(x):
        return Refined(5.0 + 0j, x, 0.0, 0, True)

    for x, expected in ((inside, True), (twin, False)):
        assert _in_span(mode(x), accepted, 1e-6) is expected
        assert (
            _in_span(mode(GHOSTED.state(x)), ghosted, 1e-6, layout=GHOSTED) is expected
        )
    rest = _outside(mode(twin), accepted, 1e-6)
    ghost_rest = _outside(mode(GHOSTED.state(twin)), ghosted, 1e-6, layout=GHOSTED)
    assert rest is not None
    assert ghost_rest is not None
    np.testing.assert_allclose(ghost_rest, GHOSTED.state(rest), atol=1e-12)
    assert _same(accepted[0], accepted[0], 1e-6)
    assert _same(ghosted[0], ghosted[0], 1e-6, layout=GHOSTED)
    assert not _same(ghosted[0], ghosted[1], 1e-6, layout=GHOSTED)


def test_multiple_ritz_vectors_are_orthonormal_over_owned_dofs():
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
    k = np.array([4.0, 4.0, 7.0])
    out = _orthonormal_multiples(k, GHOSTED.state(vectors), layout=GHOSTED)
    gram = out[:N, :2].conj().T @ out[:N, :2]
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-13)
    # a state still, and the lone vector untouched
    np.testing.assert_allclose(out, GHOSTED.state(out[:N]), rtol=1e-14)
    np.testing.assert_array_equal(out[:, 2], GHOSTED.state(vectors[:, 2]))
