"""A perfectly matched layer across ranks: the stretch needs no exchange.

Each rank evaluates the stretch on its own elements, from the layer geometry of
the whole mesh, so the distributed S equals one process's to the direct solver's
rounding. The TEM line of `pml_cases` ends in a PEC-backed layer; its S is
compared to `S_GATE` against one process's `direct`. ``"auto"`` with a layer
takes `direct` (MUMPS) under ranks too, where it would otherwise take
`bddc_gmres`, whose iterations stall on the stretch.
"""

import numpy as np
import pml_cases as pc
import pytest
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc
from gfem._engine.driven import DrivenRule

#: Relative to the largest entry of the one-process S.
S_GATE = 1e-12
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
