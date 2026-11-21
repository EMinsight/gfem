"""A perfectly matched layer across ranks: the stretch needs no exchange.

Each rank evaluates the stretch on its own elements, from the layer geometry of
the whole mesh, so the distributed S equals one process's to the direct solver's
rounding. The TEM line of `pml_cases` ends in a PEC-backed layer; its S is
compared to `S_GATE` against one process's `direct`. ``"auto"`` with a layer
takes `direct` (MUMPS) under ranks too, where it would otherwise take
`bddc_gmres`, whose iterations stall on the stretch.

An eigenmode solve with a layer projects on the stretched weight, a complex
H1 form factored by MUMPS across ranks. The leaky guide of `pml_cases` is
solved on one process with ``eigen_solver="slepc"`` and on `RANKS` with
``"auto"`` (SLEPc): the modes, their Q and the leaky mode's eigenfrequency
gradient by the core's permittivity must be one process's, and one
process's SLEPc modes ARPACK's.
"""

import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc
from gfem._engine.driven import DrivenRule

#: Relative to the largest entry of the one-process S.
S_GATE = 1e-12
#: Relative to each mode's |omega|, and the Q's, across ranks.
OMEGA_GATE = 1e-10
#: Relative to each mode's |omega|, SLEPc's seeds against ARPACK's: both are
#: refined on the same operator.
SEED_GATE = 1e-9
#: Relative to the largest one-process entry.
GRADIENT_GATE = 1e-9
#: The leaky guide's modes nearest its resonance.
N_MODES = 2
RANKS = 3
#: A reflection well above the layer's floor, so S11 is not rounding noise.
PML = gfem.PML("pml", target_reflection=1e-2)

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, threads=1)


def _model(path):
    mesh = gfem.Mesh.from_gmsh(path)
    air = gfem.Material(eps_r=1.0)
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=PML,
    )


def _problem():
    return gfem.Driven((pc.F0,))


def _s(path, solver):
    result = gfem.solve(_model(path), _problem(), options=_options(solver))
    return np.asarray(result.s.values)


def _auto(path):
    """The linear solver ``"auto"`` resolves to on this rank, and the S it gives."""
    rule = DrivenRule(
        _model(path), _problem(), _options("auto"), comm=_mpi.resolve_comm()
    )
    return rule.linear.kind, _s(path, "auto")


@pytest.fixture
def line(tmp_path):
    """The TEM line at h = 1/6 um, written once, before the ranks read it."""
    return pc.tem_line(tmp_path / "line.msh", cells=6)


def _error(s, s_one):
    return np.abs(s - s_one).max() / np.abs(s_one).max()


@needs_stack
def test_direct_s_matches_one_process(line):
    s_one = _s(line, "direct")
    assert abs(s_one[0, 0, 0]) > 1e-3  # the layer's reflection, not its floor
    s, *_ = on_ranks(_s, line, "direct", ranks=RANKS, limit=300)
    error = _error(s, s_one)
    print(f"direct ranks={RANKS} S error={error:.3g}")  # noqa: T201
    assert error < S_GATE, f"S off by {error:.3g} relative"


@needs_stack
def test_auto_takes_direct_under_ranks(line):
    s_one = _s(line, "direct")
    (kind, s), *_ = on_ranks(_auto, line, ranks=RANKS, limit=300)
    assert kind == "direct"
    error = _error(s, s_one)
    print(f"auto ranks={RANKS} S error={error:.3g}")  # noqa: T201
    assert error < S_GATE, f"S off by {error:.3g} relative"


def _guide(path):
    """The leaky guide at ``path``, its core a permittivity field."""
    model, _ = pc.leaky_guide_model(Path(path), write=False)
    eps = np.full(len(model.mesh.elements("core")), pc.GUIDE_EPS)
    model = model.with_material("core", gfem.MaterialField(eps_r=eps))
    target = pc.guide_root().real * pc.C0 * 1e3 / (2 * math.pi)
    return model, gfem.Eigenmode(target=target, n_modes=N_MODES)


def _eigen_options(eigen_solver):
    return gfem.SolverOptions(
        order=2,
        linear_solver="direct",
        eigen_solver=eigen_solver,
        tolerance=1e-12,
        threads=1,
    )


def _modes(path, eigen_solver):
    """The guide's omega and Q, and the leaky mode's d omega / d eps_core."""
    model, problem = _guide(path)
    options = _eigen_options(eigen_solver)
    with warnings.catch_warnings():
        # a far mode trapped below the air's cutoff may take the gain sign
        warnings.simplefilter("ignore", gfem.LayerClearanceWarning)
        result = gfem.solve(model, problem, options=options)
        one = gfem.Eigenmode(target=problem.target)

        def omega(eps):
            moved = model.with_material("core", gfem.MaterialField(eps_r=eps))
            return jnp.real(gfem.solve(moved, one, options=options).omega[0])

        grad = jax.grad(omega)(jnp.asarray(model.materials["core"].eps_r))
    return {
        "omega": np.asarray(result.omega),
        "q": np.asarray(result.q),
        "grad": np.asarray(grad),
    }


@pytest.fixture
def guide(tmp_path):
    """The leaky guide's mesh, written once, before the ranks read it."""
    path = tmp_path / "guide.msh"
    pc.leaky_guide(path)
    return str(path)


def _relative(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - b) / np.abs(b)))


@needs_stack
def test_slepc_finds_arpack_modes_through_a_layer(guide):
    (one,) = on_ranks(_modes, guide, "slepc", ranks=1, limit=600)
    arpack = _modes(guide, "arpack")
    error = _relative(one["omega"], arpack["omega"])
    print(f"slepc vs arpack omega error={error:.3g}")  # noqa: T201
    assert error < SEED_GATE, f"omega off by {error:.3g} relative"


@needs_stack
def test_eigen_with_a_layer_matches_one_process(guide):
    (one,) = on_ranks(_modes, guide, "slepc", ranks=1, limit=600)
    got, *_ = on_ranks(_modes, guide, "auto", ranks=RANKS, limit=600)
    for name in ("omega", "q"):
        error = _relative(got[name], one[name])
        print(f"ranks={RANKS} {name} error={error:.3g}")  # noqa: T201
        assert error < OMEGA_GATE, f"{name} off by {error:.3g} relative"
    scale = np.abs(one["grad"]).max()
    assert scale > 0
    error = np.abs(got["grad"] - one["grad"]).max() / scale
    print(f"ranks={RANKS} gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"the gradient off by {error:.3g} relative"
