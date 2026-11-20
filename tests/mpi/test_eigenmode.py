"""Eigenmode solves across ranks against one process (ADR 0502).

Each case is solved on one process with ``eigen_solver="slepc"`` and ``direct`` (the
seed method a distributed solve runs) and on 2, 3 and 4 ranks with ``eigen_solver``
left at ``"auto"``, which names SLEPc above one rank. The selected modes must be
the same, in the same order: `omega` to `OMEGA_GATE`, the coupling Q to `Q_GATE`,
the clusters equal, the participations to `PARTICIPATION_GATE` (a distributed
polish, ADR 0104), the mode fields through `gfem.Field.at` to `FIELD_GATE` with
their sign, and `gfem.interface_participation` on their pieces to
`INTERFACE_GATE`. The eigenfrequency's gradient matches one process to
`GRADIENT_GATE`; a mode-field gradient raises on every rank when the backward is
traced. `on_ranks` checks that every rank holds the same bits. Every solve needs
the MPI PETSc stack (SLEPc), so the tests skip without it.
"""

from pathlib import Path

import eigen_cases
import gradient_cases as cases
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ranks import on_ranks

import gfem
from gfem import _petsc

#: Decision 08 Q6: relative to each mode's |omega|.
OMEGA_GATE = 1e-10
#: ``bddc_gmres``'s shift solves are iterative.
BDDC_OMEGA_GATE = 1e-8
#: Relative.
Q_GATE = 1e-7
#: Absolute: an unpolished mode is off by 1e-6 to 3e-4 (ADR 0104 amendment 1).
PARTICIPATION_GATE = 1e-9
#: Relative to the largest one-process value, sign included.
FIELD_GATE = 1e-8
#: Relative.
INTERFACE_GATE = 1e-9
#: Relative to the largest one-process entry (decision 08 Q6 case 6).
GRADIENT_GATE = 1e-9
RANKS = (2, 3, 4)
LIMIT = 600

pytestmark = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _model(directory, case):
    """``(model, problem)`` of a case of decision 08 Q6."""
    directory = Path(directory)
    if case == "terminated":  # damped, companion seed, lumped R
        mesh = eigen_cases.line_mesh(directory)
        model = eigen_cases.terminated_line(mesh, 1000.0)
        return model, gfem.Eigenmode(8e9, n_modes=3)
    if case == "kinetic":  # polished mode fields, bulk and sheet participation
        mesh = cases.sheet_cavity(directory)
        fill, _ = cases.cavity_fill(mesh, seed=1)
        return cases.kinetic_cavity(mesh, fill), gfem.Eigenmode(cases.KINETIC_TARGET)
    if case == "conductor":  # nonlinear: a conductor's surface impedance
        mesh = eigen_cases.line_mesh(directory)
        model = eigen_cases.conductor_line(mesh)
        return model, gfem.Eigenmode(0.75 * eigen_cases.F1)
    if case == "dispersive":  # a far-pole Lorentz law on the cover
        resonator = cases.layered_resonator(directory)
        model = cases.dispersive_resonator(resonator, cases.EIGEN_LAWS["lorentz"])
        return model, gfem.Eigenmode(cases.RESONATOR_TARGET)
    raise ValueError(case)


def _points(mesh, n=12, seed=5):
    """Points well inside ``n`` tets, off every face, where the field is smooth."""
    tets = np.asarray(mesh.tetrahedra)
    chosen = np.random.default_rng(seed).choice(len(tets), n, replace=False)
    return np.asarray(mesh.nodes)[tets[chosen]].mean(axis=1)


def _modes(directory, case, linear_solver, eigen_solver):
    """The case's outputs as plain arrays, one thread a rank."""
    model, problem = _model(directory, case)
    options = gfem.SolverOptions(
        order=2,
        linear_solver=linear_solver,
        eigen_solver=eigen_solver,
        tolerance=1e-12,
        threads=1,
    )
    result = gfem.solve(model, problem, options=options)
    out = {
        "omega": np.asarray(result.omega),
        "cluster": np.asarray(result.cluster),
        "q_port": np.array([np.asarray(v) for v in result.q_port.values()]),
        "participation": np.array(
            [np.asarray(v) for v in result.participation.values()]
        ),
        "bulk": np.array([np.asarray(v) for v in result.bulk_participation.values()]),
        "sheet": np.array([np.asarray(v) for v in result.sheet_participation.values()]),
    }
    if case == "kinetic":
        points = _points(model.mesh)
        out["field"] = np.array([np.asarray(f.at(points)) for f in result.e_fields])
        layer = gfem.interface_participation(
            result, "sheet", 3e-9, 10.0, kind="default"
        )
        out["interface"] = np.asarray(layer.participation)
    return out


def _one(directory, case):
    """One process's outputs: SLEPc on ``direct``, in a launch of one rank."""
    (out,) = on_ranks(_modes, directory, case, "direct", "slepc", ranks=1, limit=LIMIT)
    return out


def _relative(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - b) / np.abs(b)))


def _assert_same_modes(got, one, *, omega_gate=OMEGA_GATE):
    np.testing.assert_array_equal(got["cluster"], one["cluster"])
    error = _relative(got["omega"], one["omega"])
    print(f"omega error={error:.3g}")  # noqa: T201
    assert error < omega_gate, f"omega off by {error:.3g} relative"


@pytest.mark.parametrize("ranks", RANKS)
def test_a_damped_line_has_one_process_modes_and_coupling_q(tmp_path, ranks):
    one = _one(tmp_path, "terminated")
    got, *_ = on_ranks(
        _modes, tmp_path, "terminated", "direct", "auto", ranks=ranks, limit=LIMIT
    )
    _assert_same_modes(got, one)
    error = _relative(got["q_port"], one["q_port"])
    print(f"ranks={ranks} q_port error={error:.3g}")  # noqa: T201
    assert error < Q_GATE, f"q_port off by {error:.3g} relative"


@pytest.mark.parametrize("ranks", RANKS)
def test_a_polished_mode_has_one_process_participations_and_field(tmp_path, ranks):
    one = _one(tmp_path, "kinetic")
    got, *_ = on_ranks(
        _modes, tmp_path, "kinetic", "direct", "auto", ranks=ranks, limit=LIMIT
    )
    _assert_same_modes(got, one)
    for name in ("participation", "bulk", "sheet"):
        error = float(np.max(np.abs(got[name] - one[name]), initial=0.0))
        print(f"ranks={ranks} {name} error={error:.3g}")  # noqa: T201
        assert error < PARTICIPATION_GATE, f"{name} off by {error:.3g}"
    field = np.abs(got["field"] - one["field"]).max() / np.abs(one["field"]).max()
    print(f"ranks={ranks} field error={field:.3g}")  # noqa: T201
    assert field < FIELD_GATE, f"the mode field off by {field:.3g} relative"
    error = _relative(got["interface"], one["interface"])
    print(f"ranks={ranks} interface error={error:.3g}")  # noqa: T201
    assert error < INTERFACE_GATE, f"interface participation off by {error:.3g}"


@pytest.mark.parametrize("ranks", RANKS)
def test_a_nonlinear_line_has_one_process_modes(tmp_path, ranks):
    one = _one(tmp_path, "conductor")
    got, *_ = on_ranks(
        _modes, tmp_path, "conductor", "direct", "auto", ranks=ranks, limit=LIMIT
    )
    _assert_same_modes(got, one)


def test_a_dispersive_cover_has_one_process_modes(tmp_path):
    one = _one(tmp_path, "dispersive")
    got, *_ = on_ranks(
        _modes, tmp_path, "dispersive", "direct", "auto", ranks=2, limit=LIMIT
    )
    _assert_same_modes(got, one)


@pytest.mark.parametrize("ranks", RANKS)
def test_bddc_shift_solves_give_one_process_modes(tmp_path, ranks):
    one = _one(tmp_path, "terminated")
    got, *_ = on_ranks(
        _modes, tmp_path, "terminated", "bddc_gmres", "auto", ranks=ranks, limit=LIMIT
    )
    _assert_same_modes(got, one, omega_gate=BDDC_OMEGA_GATE)


def _cavity(directory):
    """``(mesh, eps, fill)``: the sheet cavity, a P0 permittivity below its sheet."""
    mesh = cases.sheet_cavity(Path(directory))
    fill, rng = cases.cavity_fill(mesh, seed=1)
    eps = jnp.asarray(rng.uniform(1.5, 2.5, len(mesh.elements("lower"))))
    return mesh, eps, fill


def _frequency_gradient(directory, eigen_solver):
    """The eigenfrequency's gradient by the permittivity field and the sheet's fill."""
    mesh, eps, fill = _cavity(directory)
    options = gfem.SolverOptions(
        order=2,
        linear_solver="direct",
        eigen_solver=eigen_solver,
        tolerance=1e-12,
        threads=1,
    )

    def objective(eps, fill):
        model = cases.cavity_model(mesh, fill).with_material(
            "lower", gfem.MaterialField(eps_r=eps)
        )
        result = gfem.solve(model, gfem.Eigenmode(cases.CAVITY_TARGET), options=options)
        # frequency and loss, weighted
        return jnp.real(result.omega[0] * cases.OMEGA_WEIGHT)

    grads = jax.grad(objective, argnums=(0, 1))(eps, fill)
    return {"eps": np.asarray(grads[0]), "fill": np.asarray(grads[1])}


@pytest.mark.parametrize("ranks", RANKS)
def test_an_eigenfrequency_gradient_matches_one_process(tmp_path, ranks):
    (one,) = on_ranks(_frequency_gradient, tmp_path, "slepc", ranks=1, limit=LIMIT)
    got, *_ = on_ranks(_frequency_gradient, tmp_path, "auto", ranks=ranks, limit=LIMIT)
    for name in ("eps", "fill"):
        scale = np.abs(one[name]).max()
        assert scale > 0, f"no {name} gradient on one process"
        error = np.abs(got[name] - one[name]).max() / scale
        print(f"ranks={ranks} {name} gradient error={error:.3g}")  # noqa: T201
        assert error < GRADIENT_GATE, f"the {name} gradient off by {error:.3g}"


def _mode_field_gradient_then_a_collective(directory):
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    mesh, eps, fill = _cavity(directory)
    options = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)

    def objective(eps):
        model = cases.cavity_model(mesh, fill).with_material(
            "lower", gfem.MaterialField(eps_r=eps)
        )
        result = gfem.solve(model, gfem.Eigenmode(cases.CAVITY_TARGET), options=options)
        return result.bulk_participation["lower"][0]

    try:
        # jit traces the backward before any host call, so before any mesh
        jax.jit(jax.grad(objective))(eps)
    except gfem.DistributedUnsupportedError as err:
        return err.feature, world.allreduce(world.rank)
    raise AssertionError("a mode-field gradient ran")


def test_every_rank_refuses_a_mode_field_gradient_and_a_later_collective_runs(
    tmp_path,
):
    values = on_ranks(
        _mode_field_gradient_then_a_collective, tmp_path, ranks=3, limit=LIMIT
    )
    assert values[0] == ("mode_field_gradient", 0 + 1 + 2)
