"""The system operator of a driven solve, at its interface.

Each test builds a `SystemOperator` of a parallel-plate line's driven
operator, without wave ports and without a rule: its adjoint handles, its
pieces and its sweeps' base handles.
"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import structured_meshes as sm
from solver_misses import cannot_converge_at

import gfem
from gfem._engine import driven
from gfem._engine.system_operator import LowRankTerm, SystemOperator, SystemSweep
from gfem._linalg import ResolvedLinearSolver, prepare
from gfem._linalg.bddc import BddcHandle

OMEGAS = (2 * math.pi * 4e9, 2 * math.pi * 5e9)
ADJOINT_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _system(line, kind, width=2):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver=kind), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    system = SystemOperator(
        op, (), linear, width=width, adjoint_tolerance=ADJOINT_TOLERANCE
    )
    return system, linear


def _loads(system, width=2):
    rng = np.random.default_rng(0)
    shape = (system.operator.space.ndof, width)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _piece_values(system):
    form = system.operator.form(1.0)
    form.Assemble()
    return form.mat.AsVector().FV().NumPy().nbytes


def test_cpu_adjoint_solves_keep_one_handle_per_frequency(line, monkeypatch):
    system, linear = _system(line, "direct")
    loads = _loads(system)
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr("gfem._linalg.prepare", spy)
    assert system.nbytes == 0
    x0 = system.solve_adjoint(0, OMEGAS[0], loads)
    held = system.nbytes
    assert held > 0
    again = system.solve_adjoint(0, OMEGAS[0], loads)
    assert system.nbytes == held
    x1 = system.solve_adjoint(1, OMEGAS[1], loads)
    assert system.nbytes > held
    assert len(calls) == 2
    monkeypatch.undo()
    for x, omega in ((x0, OMEGAS[0]), (again, OMEGAS[0]), (x1, OMEGAS[1])):
        fresh = linear.prepare(
            system.operator_at(omega),
            symmetry="complex_symmetric",
            width=2,
            tolerance=ADJOINT_TOLERANCE,
        )
        expected = fresh.solve(loads)
        fresh.free()
        np.testing.assert_allclose(x, expected, rtol=1e-12, atol=0)
    system.free()
    assert system.nbytes == 0  # no wave port: no W vectors


@pytest.mark.parametrize(
    "kind", ["direct", "bddc_gmres", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_pieces_are_held_only_for_the_piece_sum_linear_solvers(line, kind):
    system, _ = _system(line, kind)
    loads = _loads(system)
    with system.sweep() as sweep:
        x, _ = sweep.solve(OMEGAS[0], loads)
    assert np.abs(x).max() > 0
    if kind == "bddc_gmres":  # BDDC assembles the form
        assert system.nbytes == 0
    else:
        # at least the values of the curl-curl and mass pieces, one of them
        # in the pattern the base handle shares
        held = system.nbytes + sum(system.shared().values())
        assert held >= 2 * _piece_values(system)
    system.free()
    assert system.nbytes == 0


def test_a_sweep_frees_its_base_when_it_raises(line, monkeypatch):
    system, _ = _system(line, "bddc_gmres")
    loads = _loads(system)
    cannot_converge_at(monkeypatch, (OMEGAS[1] / (2 * math.pi),))
    bases = []
    at = SystemSweep.at

    def spy(self, omega):
        handle = at(self, omega)
        bases.append(self._base)
        return handle

    monkeypatch.setattr(SystemSweep, "at", spy)
    freed = []
    free = BddcHandle.free

    def spy_free(self):
        freed.append(self)
        return free(self)

    monkeypatch.setattr(BddcHandle, "free", spy_free)

    def sweep():
        with system.sweep() as swept:
            x, _ = swept.solve(OMEGAS[0], loads)
            assert not freed
            swept.solve(OMEGAS[1], loads, x0=x)  # raises

    with pytest.raises(gfem.LinearSolverConvergenceError):
        sweep()
    assert any(h is bases[-1] for h in freed)


def test_corrections_are_block_diagonal_over_wave_port_surfaces(line):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    rng = np.random.default_rng(1)
    ndof = op.space.ndof
    pairs = (
        (rng.standard_normal((ndof, 6)), rng.standard_normal((6, 4))),  # n = 2
        (rng.standard_normal((ndof, 4)), rng.standard_normal((4, 2))),  # n = 1
    )
    blocks = [rng.standard_normal((4, 4)), rng.standard_normal((2, 2))]
    surfaces = [
        SimpleNamespace(
            piece=dataclasses.replace(op.pieces[0], name=f"waveport s{k}"),
            vectors=v,
            correction=lambda _, p=p, d=d: (p, d),
        )
        for k, ((v, p), d) in enumerate(zip(pairs, blocks, strict=True))
    ]
    system = SystemOperator(
        op, surfaces, linear, width=1, adjoint_tolerance=ADJOINT_TOLERANCE
    )
    term = system.corrections(OMEGAS[0])
    np.testing.assert_array_equal(
        term.left, np.concatenate([v @ p for v, p in pairs], 1)
    )
    np.testing.assert_array_equal(term.coefficients, scipy.linalg.block_diag(*blocks))
    # wave ports alone: U C Uᵀ, no flags, the plain capacitance
    assert term.right is None
    assert term.infinite is None
    assert not term.row_scaled


def _wave_port(op, rng, name):
    """A wave-port-like coupling: ``U C Uᵀ`` with ``C`` complex symmetric."""
    ndof = op.space.ndof
    free = np.array(list(op.space.FreeDofs()))
    vectors = rng.standard_normal((ndof, 2)) * free[:, None] / np.sqrt(ndof)
    d = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return SimpleNamespace(
        piece=dataclasses.replace(op.pieces[0], name=name),
        vectors=vectors,
        correction=lambda _: (np.eye(2), d + d.T),
    )


def _floquet_like(op, rng, name, shifts, infinite=None):
    """A coupling of ``U C Vᵀ``, ``V = conj(U)``, ``C`` diagonal, row-scaled.

    Its transposed form is ``conj(U) Cᵀ Uᵀ``.
    """
    ndof = op.space.ndof
    free = np.array(list(op.space.FreeDofs()))
    p = len(shifts)
    u = rng.standard_normal((ndof, p)) + 1j * rng.standard_normal((ndof, p))
    u = u * free[:, None] / np.sqrt(ndof)
    flags = None if infinite is None else np.asarray(infinite, bool)

    def term(_):
        return LowRankTerm(
            u, np.diag(shifts), right=np.conj(u), infinite=flags, row_scaled=True
        )

    piece = dataclasses.replace(op.pieces[0], name=name)
    coupling = SimpleNamespace(piece=piece, vectors=u, term=term)
    coupling.transposed = lambda: SimpleNamespace(
        piece=piece, vectors=u, term=lambda omega: term(omega).transposed()
    )
    return coupling, u


def _dense_free(system, omega):
    """``A(ω)`` of ``system`` on its free DOFs, densely."""
    op = system.operator
    free = np.array(list(op.space.FreeDofs()))
    handle = prepare(system.operator_at(omega), symmetry="general", kind="direct")
    dense = handle.apply(np.eye(op.space.ndof))[np.ix_(free, free)]
    handle.free()
    return dense, free


def test_corrections_concatenate_right_factors_and_flags(line):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    rng = np.random.default_rng(2)
    wave = _wave_port(op, rng, "waveport s0")
    floquet, u = _floquet_like(op, rng, "floquet s1", [0.5j, 0.0], [False, True])
    system = SystemOperator(
        op, [wave, floquet], linear, width=1, adjoint_tolerance=ADJOINT_TOLERANCE
    )
    term = system.corrections(OMEGAS[0])
    np.testing.assert_array_equal(term.left, np.concatenate([wave.vectors, u], 1))
    # a wave port's right factor is its left one
    np.testing.assert_array_equal(
        term.right, np.concatenate([wave.vectors, np.conj(u)], 1)
    )
    np.testing.assert_array_equal(term.infinite, [False, False, False, True])
    assert term.row_scaled


def test_the_transposed_system_takes_each_couplings_transposed_form(line):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver="direct"), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    rng = np.random.default_rng(3)
    wave = _wave_port(op, rng, "waveport s0")
    shifts = np.array([0.3 + 0.2j, -0.4 + 1.1j, 2.0 - 0.5j])
    floquet, u = _floquet_like(op, rng, "floquet s1", shifts)
    # the line is symmetric: its operator stands for the one at -k
    system = SystemOperator(
        op,
        [wave, floquet],
        linear,
        width=1,
        adjoint_tolerance=ADJOINT_TOLERANCE,
        transposed=lambda: op,
    )
    omega = OMEGAS[0]
    A, free = _dense_free(system, omega)
    vw = wave.vectors[free]
    _, d = wave.correction(omega)
    wave_term = vw @ d @ vw.T
    forward = A + wave_term + u[free] @ np.diag(shifts) @ np.conj(u[free]).T
    adjoint = A + wave_term + np.conj(u[free]) @ np.diag(shifts) @ u[free].T
    loads = _loads(system, width=1)
    try:
        with system.sweep() as sweep:
            x, _ = sweep.solve(omega, loads)
        want = np.linalg.solve(forward, loads[free])
        np.testing.assert_allclose(
            x[free], want, rtol=0, atol=1e-10 * np.abs(want).max()
        )
        x = system.solve_adjoint(0, omega, loads)
        want = np.linalg.solve(adjoint, loads[free])
        np.testing.assert_allclose(
            x[free], want, rtol=0, atol=1e-10 * np.abs(want).max()
        )
        # the forward's term is not the adjoint's: the transposed form matters
        assert (
            np.abs(np.linalg.solve(forward, loads[free]) - want).max()
            > 1e-3 * np.abs(want).max()
        )
        assert system._adjoint_system().surfaces[0] is wave
    finally:
        system.free()


def test_a_backward_starts_at_the_kept_base_frequency_up_to_round_off(line):
    system, _ = _system(line, "direct")
    # the base's frequency, computed another way: one ulp off the backward's
    first = np.nextafter(OMEGAS[0], np.inf)
    with system.sweep() as sweep:
        sweep.at(first)
        system.keep_base(sweep)
    try:
        assert system.adjoint_order(OMEGAS) == [0, 1]
        system.free_base()
        with system.sweep() as sweep:
            sweep.at(OMEGAS[1])
            system.keep_base(sweep)
        assert system.adjoint_order(OMEGAS) == [1, 0]
    finally:
        system.free()
