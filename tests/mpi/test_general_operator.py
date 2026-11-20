"""A system unequal to its transpose across ranks: MUMPS LU, BDDC with an LU coarse.

A Bloch-phased space makes ``A^T != A`` (ADR 0040); here a gyrotropic permittivity
does, on the layered line's distributed space, so no periodic partition is needed.
Each test compares the rank sums ``S_ij = f_i^T A^{-1} f_j`` bit for bit across ranks
(`on_ranks`) and against one process's ``direct`` (PARDISO or UMFPACK LU) at ADR
0070's gates. ``S_12`` far from ``S_21`` keeps a symmetric factorization from passing.
"""

import gradient_cases as cases
import ngsolve as ngs
import numpy as np
import pytest
from ranks import on_ranks

from gfem import _mpi, _petsc
from gfem._engine import quadrature
from gfem._engine.driven import driven_operator
from gfem._linalg import Operator, prepare
from gfem._linalg.layout import ONE_PROCESS

#: Relative to the largest one-process entry: ADR 0070's gates.
GATES = {"direct": 1e-11, "bddc_gmres": 1e-9}
#: The wavenumber, in the line's length unit: about one wavelength along it.
KAPPA = 1.5
#: Lossy, so the system is regular at any partition.
EPSILON = 3.0 - 1.0j
#: The gyrotropic part of the permittivity, ``i g`` off the diagonal.
GYRATION = 1.0

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _general_operator(directory):
    """``curl curl - kappa^2 epsilon`` with a gyrotropic ``epsilon``, and two loads."""
    comm = _mpi.resolve_comm()
    space = driven_operator(cases.layered_line(directory), 2, comm=comm)
    fes, layout = space.space, space.layout
    u, v = fes.TnT()
    epsilon = ngs.CF(
        (EPSILON, 1j * GYRATION, 0, -1j * GYRATION, EPSILON, 0, 0, 0, EPSILON),
        dims=(3, 3),
    )
    form = ngs.BilinearForm(fes, symmetric=False)
    form += (
        ngs.curl(u) * ngs.curl(v) - KAPPA**2 * ngs.InnerProduct(epsilon * u, v)
    ) * quadrature.dx(4)
    loads = []
    for source in (ngs.CF((1 + ngs.x / 4, 0, 0)), ngs.CF((0, 1, ngs.y / 2))):
        linear = ngs.LinearForm(fes)
        linear += source * v * quadrature.dx(4)
        linear.Assemble()
        loads.append(np.array(linear.vec.FV().NumPy()))
    return Operator(form, fes.FreeDofs(), layout=layout), np.stack(loads, axis=1)


def _s(directory, kind):
    """``S_ij = f_i^T A^{-1} f_j``, summed over ranks, one thread."""
    ngs.SetNumThreads(1)
    operator, loads = _general_operator(directory)
    handle = prepare(
        operator, symmetry="general", kind=kind, tolerance=1e-12, maxiter=2000
    )
    states = handle.solve(loads)
    s = (operator.layout or ONE_PROCESS).pair(loads, states)
    handle.free()
    return s


@needs_stack
@pytest.mark.parametrize("kind", ["direct", "bddc_gmres"])
@pytest.mark.parametrize("ranks", [2, 4, 8, 16])
def test_a_general_system_matches_one_process(tmp_path, kind, ranks):
    one = _s(tmp_path, "direct")
    scale = np.abs(one).max()
    assert abs(one[0, 1] - one[1, 0]) > 1e-2 * scale  # A^T != A shows in S
    s, *_ = on_ranks(_s, tmp_path, kind, ranks=ranks, limit=300)
    error = np.abs(s - one).max() / scale
    print(f"{kind} ranks={ranks} S error={error:.3g}")  # noqa: T201
    assert error < GATES[kind], f"S off by {error:.3g} relative"


def _petsc_refusal(directory):
    """The one-process ``ValueError`` of ``petsc`` on a general operator, then a sum."""
    comm = _mpi.resolve_comm()
    operator, _ = _general_operator(directory)
    try:
        with _mpi.section(comm):
            prepare(operator, symmetry="general", kind="petsc")
    except ValueError as err:
        return str(err), comm.allreduce(1)
    raise AssertionError("petsc prepared a general operator")


@needs_stack
def test_petsc_refuses_a_general_operator_on_every_rank(tmp_path):
    message, ranks = on_ranks(_petsc_refusal, tmp_path, ranks=3)[0]
    assert "linear_solver='petsc' needs A^T = A" in message
    assert ranks == 3
