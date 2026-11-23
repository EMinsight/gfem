"""The PETSc linear solver: PETSc KSP / PC behind the solver-handle seam.

By default an exact MUMPS LDL^T factorization (``ksp_type preonly``,
``pc_type cholesky``); ``petsc_options`` configures any other KSP / PC.
Every test needs complex petsc4py: skipped without it (the ``petsc`` marker).
"""

import dataclasses
import logging
import os
import re
import types
from pathlib import Path

import gradient_cases
import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm
from petsc_recipe import KSP_VIEW, MUMPS_LU

import gfem
from gfem import _petsc
from gfem._engine import cache, quadrature
from gfem._engine.driven import DrivenRule
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, coarse, petsc, prepare
from gfem._linalg.petsc import PetscHandle
from gfem._linalg.pieces import AssembledPieces, PieceSum

pytestmark = pytest.mark.petsc


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _laplace(built, complex_valued=False, shift=0.0, order=2):
    space = ngs.H1(
        built.mesh, order=order, dirichlet="bottom|top", complex=complex_valued
    )
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) + shift * u * v) * quadrature.dx(order)
    return Operator(form, space.FreeDofs()), space


def _free(space):
    return np.array(list(space.FreeDofs()))


def _scotch_version() -> tuple[int, int, int]:
    """The version of the stack's Scotch, from its header."""
    import petsc4py

    header = Path(petsc4py.get_config()["PETSC_DIR"], "include", "scotch.h").read_text()
    fields = ("SCOTCH_VERSION", "SCOTCH_RELEASE", "SCOTCH_PATCHLEVEL")
    major, minor, patch = (
        int(re.search(rf"#define\s+{name}\s+(\d+)", header)[1]) for name in fields
    )
    return major, minor, patch


# ParMETIS, not free for commercial use, is on no stack gfem's CI builds; a user's
# own --with-parmetis stack declares itself with GFEM_PARMETIS_STACK=1.
PARMETIS_STACK = os.environ.get("GFEM_PARMETIS_STACK") == "1"


def test_the_petsc_stack_is_complex_with_mumps_and_no_non_free_package():
    # The licence control (no non-free or non-commercial dependency) for a stack the
    # licence gate cannot see (what tools/petsc_wheelhouse.sh compiles into the petsc
    # wheel).
    PETSc = _petsc.require_petsc()
    assert np.dtype(PETSc.ScalarType) == np.complex128
    assert PETSc.Sys.hasExternalPackage("mumps")
    assert PETSc.Sys.hasExternalPackage("metis")
    # OpenBLAS's ZGEMMT overflows on complex fronts of 32,768 or more: MUMPS without it
    assert not _petsc.has_zgemmt()
    has_mpi = _petsc.has_mpi()
    for package in ("ptscotch", "scalapack"):
        assert PETSc.Sys.hasExternalPackage(package) is has_mpi, package
    if has_mpi:
        # 2^31 graph entries pass near 54 M DOF; PETSc's own Scotch 7.0.15 crashes
        # MUMPS's parallel analysis
        assert np.dtype(PETSc.IntType) == np.int64
        assert _scotch_version() >= (7, 0, 18)


def _configure_options() -> str:
    """The options PETSc was configured with, from its installed ``petscvariables``."""
    import petsc4py

    config = petsc4py.get_config()
    variables = Path(
        config["PETSC_DIR"], config.get("PETSC_ARCH") or "", "lib", "petsc", "conf"
    )
    text = (variables / "petscvariables").read_text()
    return re.search(r"^CONFIGURE_OPTIONS\s*=(.*)$", text, re.MULTILINE)[1]


def test_a_stack_the_script_builds_carries_the_ordering_cap():
    # the script hands PETSc its own MUMPS tarball, the one with the cap
    if not re.search(r"MUMPS_[\d.]+-gfem\.tar\.gz", _configure_options()):
        pytest.skip("a stock stack: its MUMPS is not the build script's")
    assert _petsc.has_paraord_cap()
    kind = coarse.PARMETIS if _petsc.has_parmetis() else coarse.CAPPED
    assert coarse.stack_kind(None) == kind


@pytest.mark.skipif(
    PARMETIS_STACK, reason="GFEM_PARMETIS_STACK=1: a user's own ParMETIS stack"
)
def test_parmetis_is_on_no_stack_gfems_ci_builds():
    assert not _petsc.require_petsc().Sys.hasExternalPackage("parmetis")


@pytest.mark.skipif(not PARMETIS_STACK, reason="GFEM_PARMETIS_STACK is not 1")
def test_a_users_parmetis_stack_is_an_mpi_stack():
    if _petsc.require_petsc().Sys.hasExternalPackage("parmetis"):
        assert _petsc.has_mpi()


def test_petsc_solves_many_right_hand_sides_on_the_free_dofs(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="petsc")
    assert handle.kind == "petsc"
    free = _free(space)
    rng = np.random.default_rng(0)
    B = rng.standard_normal((space.ndof, 3))
    X = handle.solve(B)
    assert X.shape == (space.ndof, 3)
    assert X.dtype == np.float64  # a real operator gives real solutions
    np.testing.assert_array_equal(X[~free], 0)
    A = handle.apply(np.eye(space.ndof))
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10 * np.abs(B).max())
    # one column at a time (a single solve, not a block): the same factors
    np.testing.assert_allclose(handle.solve(B[:, 1]), X[:, 1], rtol=1e-12)
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)


def test_petsc_solves_a_complex_symmetric_operator(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    free = _free(space)
    rng = np.random.default_rng(1)
    B = rng.standard_normal((space.ndof, 2)) + 1j * rng.standard_normal((space.ndof, 2))
    X = handle.solve(B)
    A = handle.apply(np.eye(space.ndof))
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10 * np.abs(B).max())
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)
    direct = prepare(
        _laplace(built, True, 1 - 2j)[0], symmetry="complex_symmetric", kind="direct"
    )
    np.testing.assert_allclose(X, direct.solve(B), rtol=1e-10)


def test_a_real_operator_solves_a_complex_rhs_by_parts(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="petsc")
    rng = np.random.default_rng(2)
    real, imag = rng.standard_normal((2, space.ndof))
    X = handle.solve(real + 1j * imag)
    np.testing.assert_allclose(X, handle.solve(real) + 1j * handle.solve(imag))


def test_without_options_petsc_is_an_exact_mumps_cholesky(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    ksp = handle._ksp
    pc = ksp.getPC()
    assert (ksp.getType(), pc.getType()) == ("preonly", "cholesky")
    assert pc.getFactorSolverType() == "mumps"
    A, _ = ksp.getOperators()
    assert A.isSymmetricKnown() == (True, True)
    assert not A.isHermitianKnown()[1]
    assert handle.solve_counted(np.ones(handle.ndof, complex))[1] == ()


def _gfem_keys():
    PETSc = _petsc.require_petsc()
    return [k for k in PETSc.Options().getAll() if k.startswith("gfem")]


def test_options_live_under_the_handles_prefix_only_while_it_reads_them(built):
    PETSc = _petsc.require_petsc()
    options = PETSc.Options()
    # a user's own, unprefixed options never reach gfem's KSP
    options.setValue("ksp_type", "gmres")
    options.setValue("pc_type", "jacobi")
    try:
        op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
        handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
        assert re.fullmatch(r"gfem\d+_", handle._ksp.getOptionsPrefix())
        assert handle._ksp.getType() == "preonly"
        assert _gfem_keys() == []
        handle.solve(np.ones(space.ndof, complex))
        handle.refactorize(_laplace(built, complex_valued=True, shift=2 + 1j)[0])
        assert _gfem_keys() == []
        handle.free()
        assert _gfem_keys() == []
    finally:
        options.delValue("ksp_type")
        options.delValue("pc_type")


def _pieces(built, integrands, weights):
    """Complex H1 pieces as a form and as a piece sum (`Operator.piece_sum`)."""
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    for w, integrand in zip(weights, integrands, strict=True):
        form += w * integrand(u, v)
    pieces = AssembledPieces(space, integrands)
    return Operator(form, space.FreeDofs(), PieceSum(pieces, weights)), pieces


PIECES = (
    lambda u, v: ngs.grad(u) * ngs.grad(v) * quadrature.dx(2),
    lambda u, v: u * v * quadrature.dx(2),
    lambda u, v: u * v * ngs.ds("side"),
)


def _agree(handle, fresh, seed):
    n = handle.ndof
    A = fresh.apply(np.eye(n))
    assert np.abs(handle.apply(np.eye(n)) - A).max() <= 1e-14 * np.abs(A).max()
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12


def test_a_refactorization_solves_like_a_fresh_handle(built):
    # a sweep's second frequency: new values on the first one's solver plan
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    ksp = handle._ksp
    handle.refactorize(_laplace(built, complex_valued=True, shift=3 + 1j)[0])
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    assert handle._ksp is ksp
    _agree(handle, fresh, 12)


def test_a_refactorization_sums_the_handles_own_pieces_in_place(built):
    op, pieces = _pieces(built, PIECES, (1.0, 2 - 1j, 3j))
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    with pytest.raises(Exception, match="assemble"):  # the form is not assembled
        _ = op.form.mat
    matrix = handle._matrix
    weights = (1.0, -1 + 1j, 0.5)
    op, _ = _pieces(built, PIECES, weights)
    handle.refactorize(dataclasses.replace(op, piece_sum=PieceSum(pieces, weights)))
    assert handle._matrix is matrix  # summed into the held matrix
    fresh = prepare(
        dataclasses.replace(op, piece_sum=None),
        symmetry="complex_symmetric",
        kind="petsc",
    )
    _agree(handle, fresh, 3)


def _on_the_side(built, volume):
    """An operator on the side's DOFs, with or without a volume term.

    On those DOFs a side mass term alone is definite; without the volume
    term its matrix couples fewer DOFs: another sparsity pattern.
    """
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    free = space.FreeDofs() & space.GetDofs(built.mesh.Boundaries("side"))
    form = ngs.BilinearForm(space, symmetric=True)
    form += (2 - 1j) * u * v * ngs.ds("side")
    if volume:
        form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(2)
    return Operator(form, free)


def test_a_refactorization_on_another_pattern_plans_again(built):
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    nonzeros = handle._values.size
    handle.refactorize(_on_the_side(built, False))
    assert handle._values.size < nonzeros
    fresh = prepare(
        _on_the_side(built, False), symmetry="complex_symmetric", kind="petsc"
    )
    _agree(handle, fresh, 4)


def _factorizations():
    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    return lambda: {
        name: PETSc.Log.Event(name).getPerfInfo()["count"]
        for name in ("MatCholFctrSym", "MatCholFctrNum")
    }


def test_a_same_pattern_refactorization_is_numeric_only(built):
    counts = _factorizations()
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    before = counts()
    handle.refactorize(_laplace(built, complex_valued=True, shift=3 + 1j)[0])
    after = counts()
    assert after["MatCholFctrSym"] == before["MatCholFctrSym"]
    assert after["MatCholFctrNum"] == before["MatCholFctrNum"] + 1


def test_a_freed_handle_refuses_to_solve_and_holds_nothing(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="petsc")
    n = handle.ndof
    assert handle.nbytes > 0
    handle.free()
    handle.free()  # idempotent
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(np.ones(n))
    with pytest.raises(RuntimeError, match="freed"):
        handle.apply(np.ones(n))
    with pytest.raises(RuntimeError, match="This solver handle was freed"):
        handle.refactorize(_laplace(built)[0])


def test_nbytes_counts_the_mumps_factorization(built):
    # MUMPS reports its memory in millions of bytes: a larger operator
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j, order=5)
    exact = prepare(op, symmetry="complex_symmetric", kind="petsc")
    factor = exact._ksp.getPC().getFactorMatrix()
    assert factor.getMumpsInfog(22) > 0
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j, order=5)
    options = (("ksp_type", "gmres"), ("pc_type", "jacobi"))
    krylov = prepare(
        op, symmetry="complex_symmetric", kind="petsc", petsc_options=options
    )
    assert exact.nbytes - krylov.nbytes == 10**6 * factor.getMumpsInfog(22)


# --- Krylov configurations --------------------------------------------------------

GMRES = (("ksp_type", "gmres"), ("pc_type", "jacobi"))


def _krylov(built, shift=1 - 2j, tolerance=1e-8, options=GMRES, **kwargs):
    op, _ = _laplace(built, complex_valued=True, shift=shift, order=4)
    return prepare(
        op,
        symmetry="complex_symmetric",
        kind="petsc",
        tolerance=tolerance,
        petsc_options=options,
        **kwargs,
    )


def _relative_residual(handle, B, X):
    free = handle._free
    return np.linalg.norm((handle.apply(X) - B)[free]) / np.linalg.norm(B[free])


def test_a_krylov_solve_stops_at_the_true_relative_residual(built):
    handle = _krylov(built)
    ksp = handle._ksp
    PETSc = _petsc.require_petsc()
    # the unpreconditioned norm, so the preconditioner goes on the right
    assert ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED
    assert ksp.getPCSide() == PETSc.PC.Side.RIGHT
    rng = np.random.default_rng(13)
    B = rng.standard_normal((handle.ndof, 2)) + 0j
    X, its = handle.solve_counted(B)
    assert len(its) == 2
    assert all(n > 1 for n in its)
    for j in range(2):
        assert _relative_residual(handle, B[:, j], X[:, j]) <= 1e-8
    exact = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j, order=4)[0],
        symmetry="complex_symmetric",
        kind="direct",
    ).solve(B)
    np.testing.assert_allclose(X, exact, atol=1e-6 * np.abs(exact).max())
    np.testing.assert_array_equal(X[~handle._free], 0)


def test_a_krylov_solve_accepts_numpy_numbers_for_its_defaults(built):
    handle = _krylov(built, tolerance=np.float64(1e-8), maxiter=np.int64(400))
    rtol, _, _, max_it = handle._ksp.getTolerances()
    assert (rtol, max_it) == (1e-8, 400)
    b = np.random.default_rng(16).standard_normal(handle.ndof) + 0j
    x = handle.solve(b)
    assert _relative_residual(handle, b, x) <= 1e-8


def test_a_user_norm_type_is_kept(built):
    PETSc = _petsc.require_petsc()
    handle = _krylov(built, options=(*GMRES, ("ksp_norm_type", "preconditioned")))
    assert handle._ksp.getNormType() == PETSc.KSP.NormType.PRECONDITIONED


def test_a_krylov_solve_at_another_tolerance_restores_the_prepared_one(built):
    handle = _krylov(built, tolerance=1e-3)
    b = np.random.default_rng(14).standard_normal(handle.ndof) + 0j
    (loose,) = handle.solve_counted(b)[1]
    x, (tight,) = handle.solve_counted(b, tolerance=1e-10)
    assert tight > loose
    assert _relative_residual(handle, b, x) <= 1e-10
    assert handle.solve_counted(b)[1] == (loose,)


def test_a_krylov_initial_guess_saves_iterations(built):
    handle = _krylov(built)
    b = np.random.default_rng(15).standard_normal(handle.ndof) + 0j
    cold, cold_its = handle.solve_counted(b)
    warm, warm_its = handle.solve_counted(b, x0=cold * (1 + 1e-6))
    assert warm_its[0] < cold_its[0]
    np.testing.assert_allclose(warm, cold, atol=1e-6 * np.abs(cold).max())
    assert handle.solve_counted(b)[1] == cold_its  # no guess: from zero again


def test_a_krylov_solve_that_does_not_converge_raises(built):
    handle = _krylov(built, tolerance=1e-14, maxiter=3)
    with pytest.raises(
        gfem.LinearSolverConvergenceError,
        match=r"PETSc gmres \+ jacobi did not converge",
    ) as err:
        handle.solve(np.ones(handle.ndof, complex))
    assert isinstance(err.value, RuntimeError)
    # outside a sweep the error carries no frequency and no partial result
    assert err.value.frequency is None
    assert err.value.unsolved is None
    assert err.value.result is None
    assert "DIVERGED_MAX_IT" in str(err.value)
    assert "1e-14 in 3 iterations" in str(err.value)


def test_a_krylov_solve_that_does_not_converge_reports_four_digits(built):
    # at two digits a stall just above the tolerance can print as the
    # tolerance itself, hiding how far off it is
    handle = _krylov(built, tolerance=1e-14, maxiter=10)
    b = np.ones(handle.ndof, complex)
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match="DIVERGED_MAX_IT"
    ) as err:
        handle.solve(b)
    stalled = handle._ksp.getResidualNorm() / np.linalg.norm(b[handle._free])
    reached = re.search(r"reached (\S+)\)", str(err.value))
    assert reached is not None
    assert float(reached.group(1)) == pytest.approx(stalled, rel=1e-3)


def test_cg_on_a_complex_symmetric_operator_points_to_gmres(built):
    handle = _krylov(
        built, shift=-40 + 1j, options=(("ksp_type", "cg"), ("pc_type", "none"))
    )
    b = np.random.default_rng(16).standard_normal(handle.ndof) + 0j
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match="DIVERGED_INDEFINITE_MAT"
    ) as err:
        handle.solve(b)
    assert "use ksp_type gmres" in str(err.value)


def _pc_setups():
    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    return lambda: PETSc.Log.Event("PCSetUp").getPerfInfo()["count"]


@pytest.mark.parametrize("reuse", [True, False])
def test_reuse_preconditioner_false_overrides_petscs_reuse(built, reuse):
    # the adaptive sweep prepares with reuse_preconditioner=False
    options = (*GMRES, ("ksp_reuse_preconditioner", True))
    handle = _krylov(built, options=options, reuse_preconditioner=reuse)
    b = np.ones(handle.ndof, complex)
    handle.solve(b)
    setups = _pc_setups()
    before = setups()
    op, _ = _laplace(built, complex_valued=True, shift=1.1 - 2j, order=4)
    handle.refactorize(op)
    handle.solve(b)
    assert setups() - before == (0 if reuse else 1)


#: GMRES on gfem's MUMPS Cholesky, its factors kept across refactorizations
MUMPS_REUSE = (("ksp_type", "gmres"), ("ksp_reuse_preconditioner", True))


def _far(built):
    """An operator far from `_krylov`'s: GMRES on its factors needs many iterations."""
    return _laplace(built, complex_valued=True, shift=-40 + 1j, order=4)[0]


def _no_iterations(handle):
    """Cap the handle's Krylov solves at 0 iterations: every solve misses."""
    rtol, atol, divtol, _ = handle._ksp.getTolerances()
    handle._ksp.setTolerances(rtol=rtol, atol=atol, divtol=divtol, max_it=0)


def test_a_reused_preconditioner_is_set_up_again_where_gmres_misses(built):
    handle = _krylov(built, options=(*MUMPS_REUSE, ("ksp_max_it", 5)))
    b = np.random.default_rng(17).standard_normal(handle.ndof) + 0j
    handle.solve(b)
    setups = _pc_setups()
    handle.refactorize(_far(built))
    before = setups()
    # 5 iterations on the old factors, then on new ones
    x, (spent,) = handle.solve_counted(b)
    assert setups() - before == 1
    assert spent > 5  # both solves of the column
    assert _relative_residual(handle, b, x) <= 1e-8
    exact = prepare(_far(built), symmetry="complex_symmetric", kind="petsc").solve(b)
    np.testing.assert_allclose(x, exact, atol=1e-6 * np.abs(exact).max())
    # the new factors are the ones kept: a nearby operator converges on them
    op, _ = _laplace(built, complex_valued=True, shift=-40.001 + 1j, order=4)
    handle.refactorize(op)
    after = setups()
    x, (spent,) = handle.solve_counted(b)
    assert setups() == after
    assert spent <= 5
    assert _relative_residual(handle, b, x) <= 1e-8


def test_a_reused_preconditioner_that_misses_again_once_set_up_raises(built):
    handle = _krylov(built, options=MUMPS_REUSE)
    b = np.ones(handle.ndof, complex)
    handle.solve(b)
    setups = _pc_setups()
    handle.refactorize(_far(built))
    _no_iterations(handle)
    before = setups()
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match="DIVERGED_MAX_IT"
    ) as err:
        handle.solve(b)
    assert isinstance(err.value, RuntimeError)
    assert setups() - before == 1  # set up again once, then raised


def test_a_miss_after_a_set_up_anew_reports_the_columns_iterations(built):
    # an unreachable tolerance: GMRES misses on the stale factors and again
    # on the new ones, 3 iterations each
    options = (*MUMPS_REUSE, ("ksp_max_it", 3))
    handle = _krylov(built, tolerance=1e-300, options=options)
    b = np.ones(handle.ndof, complex)
    with pytest.raises(gfem.LinearSolverConvergenceError):
        handle.solve(b)  # current factors: one solve, no set-up anew
    handle.refactorize(_far(built))
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match="DIVERGED_MAX_IT"
    ) as err:
        handle.solve(b)
    assert "in 6 iterations" in str(err.value)  # both solves of the column


@pytest.mark.parametrize(
    ("options", "reuse"),
    [
        ((("ksp_type", "gmres"),), True),  # PETSc's reuse not asked for
        (MUMPS_REUSE, False),  # asked for, overridden as the adaptive sweep does
    ],
)
def test_a_handle_without_reuse_never_sets_its_preconditioner_up_twice(
    built, options, reuse
):
    handle = _krylov(built, options=options, reuse_preconditioner=reuse)
    b = np.ones(handle.ndof, complex)
    handle.solve(b)
    setups = _pc_setups()
    before = setups()
    handle.refactorize(_far(built))
    assert setups() - before == 1  # the refactorization's own set-up
    _no_iterations(handle)
    with pytest.raises(gfem.LinearSolverConvergenceError, match="DIVERGED_MAX_IT"):
        handle.solve(b)
    assert setups() - before == 1


def test_preonly_refuses_petscs_preconditioner_reuse(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(
        ValueError, match=r"ksp_reuse_preconditioner.*previous operator"
    ):
        prepare(
            op,
            symmetry="complex_symmetric",
            kind="petsc",
            petsc_options=(("ksp_reuse_preconditioner", None),),
        )
    assert _gfem_keys() == []


@pytest.mark.parametrize(
    "options",
    [
        (("pc_type", "jacobi"),),
        (("pc_type", "sor"),),
        (("pc_type", "none"),),
        (("pc_type", "bjacobi"),),
        (("pc_type", "ilu"), ("pc_factor_mat_solver_type", "petsc")),
    ],
)
def test_preonly_refuses_a_preconditioner_that_is_not_an_exact_factorization(
    built, options
):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    pc = options[0][1]
    with pytest.raises(
        ValueError, match=rf"pc_type '{pc}'.*preonly needs pc_type lu or cholesky"
    ):
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert _gfem_keys() == []


def test_preonly_refuses_a_pivot_shift(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    options = (
        ("pc_factor_mat_solver_type", "petsc"),
        ("pc_factor_shift_type", "nonzero"),
        ("pc_type", "lu"),
    )
    with pytest.raises(
        ValueError, match=r"pc_factor_shift_type.*factors of another matrix"
    ):
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert _gfem_keys() == []


def test_a_krylov_solve_petsc_ends_without_a_residual_raises(built):
    options = (*GMRES, ("ksp_convergence_test", "skip"), ("ksp_max_it", 5))
    handle = _krylov(built, options=options)
    with pytest.raises(ValueError, match="without measuring a residual"):
        handle.solve(np.ones(handle.ndof, complex))


# --- what PETSc and gfem refuse ---------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        (("ksp_tpye", "gmres"),),  # a typo
        # not MUMPS's PC
        (("mat_mumps_icntl_14", 50), ("ksp_type", "gmres"), ("pc_type", "jacobi")),
    ],
)
def test_an_option_petsc_did_not_read_raises(built, options):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    unread = options[0][0]
    with pytest.raises(ValueError, match=rf"\['{unread}'\] was not read") as err:
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert "check the name" in str(err.value)
    assert _gfem_keys() == []


def test_an_option_petsc_reads_is_honoured(built):
    handle = prepare(
        _on_the_side(built, True),
        symmetry="complex_symmetric",
        kind="petsc",
        petsc_options=(("mat_mumps_icntl_14", 55), ("pc_type", "lu")),
    )
    pc = handle._ksp.getPC()
    assert (pc.getType(), pc.getFactorSolverType()) == ("lu", "mumps")
    assert pc.getFactorMatrix().getMumpsIcntl(14) == 55
    # read again by a new pattern's MUMPS instance
    handle.refactorize(_on_the_side(built, False))
    assert handle._ksp.getPC().getFactorMatrix().getMumpsIcntl(14) == 55


def _recipe(built, options):
    """A complex symmetric operator, solved with one of the recipe's dicts."""
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    pairs = tuple(options.items())
    handle = prepare(
        op, symmetry="complex_symmetric", kind="petsc", petsc_options=pairs
    )
    direct = prepare(
        _laplace(built, True, 1 - 2j)[0], symmetry="complex_symmetric", kind="direct"
    )
    rng = np.random.default_rng(3)
    b = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    return handle, handle.solve(b), direct.solve(b)


def test_the_recipes_mumps_lu_options_give_mumps_more_workspace(built):
    handle, x, direct = _recipe(built, MUMPS_LU)
    pc = handle._ksp.getPC()
    assert (pc.getType(), pc.getFactorSolverType()) == ("lu", "mumps")
    assert pc.getFactorMatrix().getMumpsIcntl(14) == 50
    np.testing.assert_allclose(x, direct, rtol=1e-10)


def test_the_recipes_ksp_view_prints_the_solver(built, capfd):
    capfd.readouterr()
    _, x, direct = _recipe(built, KSP_VIEW)
    printed = capfd.readouterr().out
    assert "type: preonly" in printed
    assert "MUMPS" in printed
    np.testing.assert_allclose(x, direct, rtol=1e-10)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ((("ksp_type", "gmress"),), "Unable to find requested KSP type gmress"),
        ((*GMRES, ("ksp_max_it", "many")), "PETSc error 63"),
    ],
)
def test_options_petsc_refuses_raise_a_value_error(built, options, reason):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(ValueError, match="refused by PETSc") as err:
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert reason in str(err.value)
    assert _gfem_keys() == []


def test_ilu_needs_petscs_own_factorization_not_mumps(built):
    # the recipe: MUMPS has no ILU, so pc_type ilu alone is refused
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    ilu = (("ksp_type", "gmres"), ("pc_type", "ilu"))
    with pytest.raises(ValueError, match="refused by PETSc") as err:
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=ilu)
    assert "mumps does not support factorization type ILU" in str(err.value)
    assert _gfem_keys() == []
    handle = _krylov(built, options=(*ilu, ("pc_factor_mat_solver_type", "petsc")))
    pc = handle._ksp.getPC()
    assert (pc.getType(), pc.getFactorSolverType()) == ("ilu", "petsc")
    b = np.random.default_rng(17).standard_normal(handle.ndof) + 0j
    assert _relative_residual(handle, b, handle.solve(b)) <= 1e-8


def test_an_error_in_petscs_set_up_raises_a_runtime_error(built):
    # the recipe's MUMPS has no OpenMP
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(RuntimeError, match=r"PETSc failed in KSPSetUp \(KSP preonly"):
        prepare(
            op,
            symmetry="complex_symmetric",
            kind="petsc",
            petsc_options=(("mat_mumps_use_omp_threads", 2),),
        )
    assert _gfem_keys() == []


def test_a_singular_operator_raises_a_factorization_error(built):
    # a side mass term alone: the rows of the inner DOFs are zero
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += u * v * ngs.ds("side")
    with pytest.raises(RuntimeError, match="factorization failed") as err:
        prepare(
            Operator(form, space.FreeDofs()), symmetry="complex_symmetric", kind="petsc"
        )
    assert "MUMPS INFOG(1)=" in str(err.value)
    assert "mat_mumps_icntl_14" in str(err.value)
    # a factorization failure is not a Krylov convergence question
    assert not isinstance(err.value, gfem.LinearSolverConvergenceError)


def _aijs(monkeypatch, error=None) -> list:
    """The AIJs PETSc creates from now on; or every creation raises ``error``."""
    PETSc = _petsc.require_petsc()
    made = []

    class Mat(PETSc.Mat):
        def createAIJWithArrays(self, *args, **kwargs):  # noqa: N802 - PETSc name
            if error is not None:
                raise error
            made.append(self)
            return super().createAIJWithArrays(*args, **kwargs)

    monkeypatch.setattr(PETSc, "Mat", Mat)
    return made


class CreationError(Exception):
    """Raised by a PETSc object creation."""


def test_a_failed_ksp_creation_destroys_the_aij(built, monkeypatch):
    PETSc = _petsc.require_petsc()
    made = _aijs(monkeypatch)

    class KSP(PETSc.KSP):
        def create(self, comm=None):  # noqa: ARG002 - PETSc signature
            raise CreationError("no KSP")

    monkeypatch.setattr(PETSc, "KSP", KSP)
    with pytest.raises(CreationError, match="no KSP"):
        prepare(_laplace(built)[0], symmetry="hermitian", kind="petsc")
    assert len(made) == 1
    assert not made[0]  # destroyed: its handle is null


def test_a_failed_replan_keeps_the_handle_on_its_operator(built, monkeypatch):
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    with monkeypatch.context() as patch:
        _aijs(patch, CreationError("no AIJ"))
        with pytest.raises(CreationError, match="no AIJ"):
            handle.refactorize(_on_the_side(built, False))  # another pattern
    fresh = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    _agree(handle, fresh, 5)
    # and it refactorizes on the pattern it holds
    handle.refactorize(_on_the_side(built, True))
    _agree(handle, fresh, 6)


def test_a_replan_sets_the_new_aij_up_before_destroying_the_old(built, monkeypatch):
    PETSc = _petsc.require_petsc()
    events = []

    class Mat(PETSc.Mat):
        def destroy(self):
            events.append(("destroy", id(self)))
            return super().destroy()

    class KSP(PETSc.KSP):
        def setOperators(self, A, P=None):  # noqa: N802 - PETSc name
            events.append(("setOperators", id(A)))
            return super().setOperators(A, P)

        def setUp(self):
            events.append(("setUp", None))
            return super().setUp()

    monkeypatch.setattr(PETSc, "Mat", Mat)
    monkeypatch.setattr(PETSc, "KSP", KSP)
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    old = id(handle._mat)
    events.clear()
    handle.refactorize(_on_the_side(built, False))
    new = id(handle._mat)
    assert events == [("setOperators", new), ("setUp", None), ("destroy", old)]
    handle.free()


@pytest.mark.parametrize(
    ("complex_valued", "symmetry"), [(False, "general"), (True, "hermitian")]
)
def test_petsc_refuses_operators_ldlt_cannot_factor(built, complex_valued, symmetry):
    op, _ = _laplace(built, complex_valued=complex_valued)
    with pytest.raises(ValueError, match=r"needs A\^T = A"):
        prepare(op, symmetry=symmetry, kind="petsc")
    with pytest.raises(TypeError, match="assemble"):
        _ = op.form.mat  # refused before the assembly


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"symmetric_storage": True}, "symmetric \\(triangular\\) storage"),
        ({"nonassemble": True}, "sparse matrix"),
    ],
)
def test_petsc_needs_the_full_sparse_matrix(built, flags, message):
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, **flags)
    form += (ngs.grad(u) * ngs.grad(v) + u * v) * quadrature.dx(2)
    with pytest.raises(ValueError, match=message):
        prepare(
            Operator(form, space.FreeDofs()), symmetry="complex_symmetric", kind="petsc"
        )


# --- petsc_options reach every solve ----------------------------------------------


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=1.0)}
    )


def _record(monkeypatch) -> list:
    """Spy on the PETSc handles' creation: the options each one reads."""
    calls = []
    from_operator = PetscHandle.from_operator.__func__

    def spy(cls, *args, **kwargs):
        calls.append(kwargs["options"])
        return from_operator(cls, *args, **kwargs)

    monkeypatch.setattr(PetscHandle, "from_operator", classmethod(spy))
    return calls


LU_GMRES = {"ksp_type": "gmres", "pc_type": "lu"}


def test_petsc_options_reach_the_forward_and_adjoint_driven_solves(line, monkeypatch):
    calls = _record(monkeypatch)
    # p2 is not excited: its S row needs an adjoint solve
    problem = gfem.Driven(frequency=(3e9, 4e9), excite="p1")
    options = gfem.SolverOptions(linear_solver="petsc", petsc_options=LU_GMRES)
    n = len(line.mesh.tetrahedra)

    def objective(eps, options):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    eps = jnp.full(n, 4.0)
    grad = jax.grad(objective)(eps, options)
    assert len(calls) >= 2  # the sweep's handle, and the adjoints'
    assert all(dict(c) == LU_GMRES for c in calls)
    direct = jax.grad(objective)(eps, gfem.SolverOptions(linear_solver="direct"))
    np.testing.assert_allclose(grad, direct, rtol=0, atol=1e-8 * np.abs(direct).max())


@pytest.mark.parametrize("problem", ["electrostatic", "driven", "eigenmode"])
def test_an_unread_petsc_option_raises_from_every_problem(line, plate, problem):
    model, p = {
        "electrostatic": (
            plate,
            gfem.Electrostatic(terminals={"t": "top"}, ground="bottom"),
        ),
        "driven": (line, gfem.Driven((3e9, 4e9))),
        "eigenmode": (line, gfem.Eigenmode(target=12e9, n_modes=1)),
    }[problem]
    options = gfem.SolverOptions(
        order=1, linear_solver="petsc", petsc_options={"ksp_tpye": "gmres"}
    )
    with pytest.raises(ValueError, match=r"\['ksp_tpye'\] was not read"):
        gfem.solve(model, p, options=options)


# --- the options across ranks, by symmetry, on COMM_SELF ----------------------------


def _advection(built, shift=1 - 2j):
    r"""A complex operator with :math:`A^T \neq A`: a Laplacian with a drift term."""
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    drift = ngs.InnerProduct(ngs.CF((3.0, 1.0, 0.0)), ngs.grad(u)) * v
    form = ngs.BilinearForm(space, symmetric=False)
    form += (ngs.grad(u) * ngs.grad(v) + shift * u * v + drift) * quadrature.dx(2)
    return Operator(form, space.FreeDofs()), space


def _free_numbering(space):
    """Every free DOF numbered in order, as one rank owning them all numbers them."""
    free = _free(space)
    numbers = np.where(free, np.cumsum(free) - 1, -1)
    count = int(free.sum())
    return types.SimpleNamespace(numbers=numbers, count=count, total=count)


#: MUMPS's parallel analysis on PT-Scotch, which only the MPI stack has.
PARALLEL_ANALYSIS = ("mat_mumps_icntl_28", "mat_mumps_icntl_29")


def _rank_options(symmetric):
    return (*petsc.defaults(symmetric), *petsc.RANK_MUMPS_OPTIONS)


def _coarse_options(symmetric):
    return coarse.options(symmetric, workspace=60, factor=1)


@pytest.mark.parametrize("options", [_rank_options, _coarse_options])
@pytest.mark.parametrize(
    ("symmetry", "factorization", "flagged"),
    [
        ("general", "lu", (False, False)),
        ("complex_symmetric", "cholesky", (True, True)),
    ],
)
def test_across_ranks_mumps_factors_by_lu_unless_a_equals_its_transpose(
    built, options, symmetry, factorization, flagged
):
    # what `direct` and BDDC's coarse problem set up under ranks, on COMM_SELF
    PETSc = _petsc.require_petsc()
    general = symmetry == "general"
    op, space = _advection(built) if general else _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    A = op.form.mat
    symmetric = not general
    aij, _ = petsc.distributed_aij(
        A, _free_numbering(space), PETSc.COMM_SELF, symmetric=symmetric
    )
    assert aij.isSymmetricKnown() == flagged
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    prefix = _petsc.next_prefix()
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(aij)
    pairs = options(symmetric)
    if not _petsc.has_mpi():  # no PT-Scotch: the parallel analysis is the ranks'
        pairs = tuple((k, v) for k, v in pairs if k not in PARALLEL_ANALYSIS)
    with _petsc.scoped_options(prefix, pairs):
        ksp.setFromOptions()
        ksp.setUp()
    pc = ksp.getPC()
    assert pc.getFailedReason() == PETSc.PC.FailedReason.NOERROR
    assert pc.getType() == factorization
    copy = aij.copy().convert("dense")  # convert alone would replace the factored AIJ
    dense = copy.getDenseArray().copy()
    copy.destroy()
    if general:
        assert np.abs(dense - dense.T).max() > 0.1 * np.abs(dense).max()
    rng = np.random.default_rng(4)
    b, x = aij.createVecs()
    b.array[:] = rng.standard_normal(b.size) + 1j * rng.standard_normal(b.size)
    ksp.solve(b, x)
    np.testing.assert_allclose(
        dense @ x.array, b.array, atol=1e-10 * np.abs(b.array).max()
    )
    for obj in (ksp, aij, b, x):
        obj.destroy()


@pytest.mark.parametrize(
    ("symmetric", "factorization"), [(False, "lu"), (True, "cholesky")]
)
def test_a_telescoped_coarse_problem_factors_by_symmetry_behind_it(
    symmetric, factorization
):
    options = dict(coarse.options(symmetric, workspace=60, factor=4))
    assert options["pc_type"] == "telescope"
    assert options["telescope_pc_type"] == factorization


def _one_process_coarse(monkeypatch, op, space, ordering=coarse.PT_SCOTCH, factor=1):
    """A real `coarse.CoarseMumps` on ``op``'s assembled matrix, on one process.

    One rank owns every free DOF and agreements are its own values. A ``factor``
    above 1 puts the factorization behind ``PCTELESCOPE``, which keeps the one rank.
    Without the MPI stack there is no parallel analysis to ask for.
    """
    PETSc = _petsc.require_petsc()
    monkeypatch.setattr(coarse, "FreeNumbering", lambda *_: _free_numbering(space))
    monkeypatch.setattr(coarse, "rule", lambda *_, **__: (ordering, factor))
    options = coarse.options

    def one_rank(*args, **kwargs):
        pairs = dict(options(*args, **kwargs))
        if "pc_telescope_reduction_factor" in pairs:
            pairs["pc_telescope_reduction_factor"] = 1
        if _petsc.has_mpi():
            return tuple(pairs.items())
        return tuple(
            (k, v)
            for k, v in pairs.items()
            if not (k.endswith(PARALLEL_ANALYSIS) and (k.endswith("29") or v == 2))
        )

    monkeypatch.setattr(coarse, "options", one_rank)
    for name in ("agree_any", "agree_all"):
        monkeypatch.setattr(coarse._mpi, name, lambda _, *, flag: flag)
    for name in ("agree_min", "agree_max"):
        monkeypatch.setattr(coarse._mpi, name, lambda _, value: value)
    # the pool keys it as the one-process walk of the cache reads it
    monkeypatch.setattr(coarse._mpi, "comm_key", lambda _: None)
    monkeypatch.setattr(coarse._mpi, "reduce_bcast", lambda _, values: values)
    monkeypatch.setattr(coarse._mpi, "node_ids", lambda comm: (0,) * comm.size)
    source = types.SimpleNamespace(
        local_mat=op.form.mat,
        row_pardofs=None,
        CreateColVector=op.form.mat.CreateColVector,
    )
    return coarse.CoarseMumps(source, space.FreeDofs(), PETSc.COMM_SELF, symmetric=True)


@pytest.fixture(autouse=True)
def empty_coarse_pool():
    """No pooled coarse factorization outlives a test."""
    yield
    coarse.release()


@pytest.fixture
def no_telescope_symbol(monkeypatch):
    """A PETSc build without ``PCTelescopeGetKSP``."""
    monkeypatch.setattr(_petsc.ctypes, "CDLL", lambda *_, **__: object())
    _petsc.telescope_symbols.cache_clear()
    yield
    _petsc.telescope_symbols.cache_clear()


def _zeroed(monkeypatch):
    """Every coarse matrix singular by one zero row and column; set-ups counted."""
    built = coarse.distributed_aij
    set_ups = []

    def singular(*args, **kwargs):
        aij, index = built(*args, **kwargs)
        aij.zeroRowsColumns(np.array([0], dtype=_petsc.require_petsc().IntType), 0.0)
        return aij, index

    monkeypatch.setattr(coarse, "distributed_aij", singular)
    monkeypatch.setattr(
        coarse._petsc,
        "next_prefix",
        lambda real=_petsc.next_prefix: set_ups.append(1) or real(),
    )
    return set_ups


def test_the_factor_behind_telescoping_is_read(built, monkeypatch):
    op, space = _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    made = _one_process_coarse(monkeypatch, op, space, factor=2)
    made.set_up()
    try:
        pc = made._ksp.getPC()
        assert pc.getType() == "telescope"
        inner = _petsc.telescope_ksp(pc)
        assert inner.getPC().getType() == "cholesky"
        assert made.nbytes > 0
    finally:
        made.free()


def test_a_singular_coarse_behind_telescoping_raises_after_one_set_up(
    built, monkeypatch
):
    op, space = _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    set_ups = _zeroed(monkeypatch)
    made = _one_process_coarse(monkeypatch, op, space, factor=2)
    with pytest.raises(gfem._mpi.AgreedError, match=r"INFOG\(1\)=-\d+"):
        made.set_up()
    assert len(set_ups) == 1


@pytest.mark.usefixtures("no_telescope_symbol")
def test_without_the_telescope_symbol_the_inner_ksp_is_none_and_said_once(
    built, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    op, space = _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    made = _one_process_coarse(monkeypatch, op, space, factor=2)
    made.set_up()
    try:
        pc = made._ksp.getPC()
        assert _petsc.telescope_ksp(pc) is None
        assert _petsc.telescope_ksp(pc) is None
        assert made.nbytes == 0
    finally:
        made.free()
    said = [r for r in caplog.records if "PCTelescopeGetKSP" in r.getMessage()]
    assert len(said) == 1


@pytest.mark.usefixtures("no_telescope_symbol")
def test_without_the_telescope_symbol_every_coarse_failure_is_retried(
    built, monkeypatch
):
    op, space = _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    set_ups = _zeroed(monkeypatch)
    made = _one_process_coarse(monkeypatch, op, space, factor=2)
    with pytest.raises(gfem._mpi.AgreedError, match="unread behind PCTELESCOPE"):
        made.set_up()
    assert len(set_ups) == petsc.MUMPS_RETRIES + 1


@pytest.mark.parametrize("theirs", [False, True])
def test_a_metis_coarse_set_up_reads_gfems_mumps_config_only_while_it_factors(
    built, monkeypatch, tmp_path, theirs
):
    # one rank, METIS forced: the variable during the set-up, and only then
    users = None
    if theirs:  # MUMPS reads it at every job, so it must exist
        users = str(tmp_path / "mumps.cfg")
        Path(users).write_text("METIS_OPTIONS(7)=1\n")
    PETSc = _petsc.require_petsc()
    op, space = _laplace(built, True, 1 - 2j)
    op.form.Assemble()
    seen = []

    class Spy(PETSc.KSP):
        def setUp(self):
            seen.append(coarse.os.environ.get(coarse.CONFIG_VARIABLE))
            return super().setUp()

    spied = types.SimpleNamespace(**{**vars(PETSc), "KSP": Spy})
    monkeypatch.setattr(_petsc, "require_petsc", lambda *_: spied)
    if users is None:
        monkeypatch.delenv(coarse.CONFIG_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(coarse.CONFIG_VARIABLE, users)
    made = _one_process_coarse(monkeypatch, op, space, coarse.METIS)
    made.set_up()
    try:
        assert seen == [str(coarse.MUMPS_CONFIG) if users is None else users]
        assert coarse.os.environ.get(coarse.CONFIG_VARIABLE) == users
        factor = made._ksp.getPC().getFactorMatrix()
        assert (factor.getMumpsIcntl(28), factor.getMumpsIcntl(7)) == (1, 5)
    finally:
        made.free()


# --- the coarse pool ---------------------------------------------------------------


def _coarse_on(monkeypatch, built, shift, order=2, ordering=coarse.PT_SCOTCH):
    """A one-process coarse factorization of a shifted Laplacian, not set up."""
    op, space = _laplace(built, True, shift, order=order)
    op.form.Assemble()
    return _one_process_coarse(monkeypatch, op, space, ordering)


@pytest.fixture
def analyses():
    """MUMPS's analyses (symbolic Cholesky factorizations) so far, read on call.

    In a log stage of its own: a PETSc error inside an event leaves that event open
    in its stage, which then counts it no more.
    """
    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    stage = PETSc.Log.Stage("gfem coarse pool")
    event = PETSc.Log.Event("MatCholFctrSym")
    stage.push()
    yield lambda: event.getPerfInfo(stage.id)["count"]
    stage.pop()


def _coarse_solution(made):
    """The coarse KSP's solution of one fixed right-hand side."""
    rhs = np.random.default_rng(7).standard_normal(made._b.getLocalSize())
    made._b.array[:] = rhs * (1 + 0.5j)
    made._ksp.solve(made._b, made._x)
    return made._x.array.copy()


def test_a_second_coarse_set_up_on_one_pattern_refactors_numerically_only(
    built, monkeypatch, analyses
):
    before = analyses()
    first = _coarse_on(monkeypatch, built, 1 - 2j)
    first.set_up()
    first.free()
    kept = _coarse_on(monkeypatch, built, 3 + 1j)  # new values, one pattern
    kept.set_up()
    assert analyses() - before == 1
    assert kept.key == first.key
    solution = _coarse_solution(kept)
    kept.free()
    coarse.release()
    fresh = _coarse_on(monkeypatch, built, 3 + 1j)
    fresh.set_up()
    try:
        assert analyses() - before == 2
        assert np.array_equal(solution, _coarse_solution(fresh))
    finally:
        fresh.free()


def test_another_wirebasket_pattern_analyses_anew(built, monkeypatch, analyses):
    before = analyses()
    first = _coarse_on(monkeypatch, built, 1 - 2j)
    first.set_up()
    first.free()
    other = _coarse_on(monkeypatch, built, 1 - 2j, order=3)
    other.set_up()
    other.free()
    assert analyses() - before == 2
    assert other.key != first.key
    assert coarse.pooled() == 2


def test_two_live_coarse_factorizations_on_one_pattern_hold_two_ksps(
    built, monkeypatch
):
    one = _coarse_on(monkeypatch, built, 1 - 2j)
    two = _coarse_on(monkeypatch, built, 1 - 2j)
    one.set_up()
    two.set_up()
    assert one.key == two.key
    assert one._ksp.handle != two._ksp.handle
    one.free()
    two.free()
    assert coarse.pooled() == 2
    # each frees once: no live owner left to take a pooled KSP back
    assert one._ksp is None
    assert two._ksp is None


def _failing_once(monkeypatch, failure):
    """The next coarse set-up's first factorization reads ``failure``."""
    real = coarse.CoarseMumps._agreed_failure
    calls = []

    def forced(self):
        calls.append(1)
        return failure if len(calls) == 1 else real(self)

    monkeypatch.setattr(coarse.CoarseMumps, "_agreed_failure", forced)
    return calls


def test_a_failed_coarse_refactorization_is_not_pooled(built, monkeypatch):
    first = _coarse_on(monkeypatch, built, 1 - 2j)
    first.set_up()
    first.free()
    assert coarse.pooled() == 1
    _failing_once(monkeypatch, -10)  # singular: not retried
    kept = _coarse_on(monkeypatch, built, 3 + 1j)
    with pytest.raises(gfem._mpi.AgreedError, match=r"INFOG\(1\)=-10"):
        kept.set_up()
    kept.free()
    assert coarse.pooled() == 0


def test_a_workspace_retry_on_refactorization_pools_its_doubled_icntl_14(
    built, monkeypatch, analyses
):
    before = analyses()
    first = _coarse_on(monkeypatch, built, 1 - 2j)
    first.set_up()
    first.free()
    calls = _failing_once(monkeypatch, coarse.MUMPS_WORKSPACE_ERROR)
    retried = _coarse_on(monkeypatch, built, 3 + 1j)
    retried.set_up()
    assert len(calls) == 2
    # PT-Scotch's first 120, doubled
    assert retried._ksp.getPC().getFactorMatrix().getMumpsIcntl(14) == 240
    retried.free()
    assert coarse.pooled() == 1
    # a new KSP after the failure: one more analysis, then the pooled one serves
    third = _coarse_on(monkeypatch, built, 2 - 1j)
    third.set_up()
    try:
        assert analyses() - before == 2
        assert third._ksp.getPC().getFactorMatrix().getMumpsIcntl(14) == 240
    finally:
        third.free()


def test_switching_a_pattern_to_metis_frees_its_pt_scotch_factorizations(
    built, monkeypatch, analyses
):
    before = analyses()
    scotch = _coarse_on(monkeypatch, built, 1 - 2j)
    scotch.set_up()
    scotch.free()
    metis = _coarse_on(monkeypatch, built, 1 - 2j, ordering=coarse.METIS)
    metis.set_up()
    metis.free()
    assert analyses() - before == 2
    assert coarse.pooled() == 1


def test_a_seen_pattern_stays_seen_when_its_factorization_is_evicted(
    built, monkeypatch
):
    made = _coarse_on(monkeypatch, built, 1 - 2j)
    made.set_up()
    made.free()
    assert coarse.seen(made.key)
    coarse.empty(None)
    assert coarse.pooled() == 0
    assert coarse.seen(made.key)
    gfem.release_memory()
    assert not coarse.seen(made.key)


def test_release_memory_empties_the_coarse_pool(built, monkeypatch):
    made = _coarse_on(monkeypatch, built, 1 - 2j)
    made.set_up()
    made.free()
    assert coarse.pooled() == 1
    gfem.release_memory()
    assert coarse.pooled() == 0


@pytest.mark.parametrize("retain_memory", [0, None])
def test_retain_memory_0_leaves_the_coarse_pool_empty_after_a_solve(
    built, monkeypatch, tmp_path, retain_memory
):
    made = _coarse_on(monkeypatch, built, 1 - 2j)
    made.set_up()
    made.free()
    model = gradient_cases.layered_line(tmp_path)
    options = gfem.SolverOptions(
        order=1, linear_solver="direct", retain_memory=retain_memory
    )
    gfem.solve(model, gfem.Driven(frequency=24e9), options=options)
    assert coarse.pooled() == (0 if retain_memory == 0 else 1)


def test_retain_memory_0_empties_the_coarse_pool_when_the_forward_raises(
    built, monkeypatch, tmp_path
):
    made = _coarse_on(monkeypatch, built, 1 - 2j)
    made.set_up()
    made.free()
    model = gradient_cases.layered_line(tmp_path)

    def singular(*_, **__):
        # as a singular coarse set-up raises, on every rank, before any walk
        raise gfem._mpi.AgreedError("the coarse factorization failed")

    monkeypatch.setattr(DrivenRule, "forward", singular)
    options = gfem.SolverOptions(order=1, linear_solver="direct", retain_memory=0)
    with pytest.raises(gfem._mpi.AgreedError):
        gfem.solve(model, gfem.Driven(frequency=24e9), options=options)
    assert coarse.pooled() == 0


def _traced_s(tmp_path, retain_memory):
    """S of the layered line as a JAX function of its permittivity, and that value."""
    line = gradient_cases.layered_line(tmp_path)
    values, _ = gradient_cases.line_design(line, seed=3)
    options = gfem.SolverOptions(
        order=1, linear_solver="direct", retain_memory=retain_memory
    )

    def s_of(eps):
        model = gradient_cases.with_fields(line, {**values, "eps_r": eps})
        solved = gfem.solve(model, gfem.Driven(frequency=24e9), options=options)
        return gradient_cases.s_functional(solved.s.values)

    return s_of, values["eps_r"]


def test_retain_memory_0_empties_the_coarse_pool_when_a_jitted_forward_raises(
    built, monkeypatch, tmp_path
):
    # a primal forward walks no cache before its set-ups, and under jit it runs after
    # gfem.solve has returned
    s_of, eps = _traced_s(tmp_path, 0)

    def singular(*_, **__):
        made = _coarse_on(monkeypatch, built, 1 - 2j)
        made.set_up()
        made.free()
        # as a later singular coarse set-up raises, on every rank
        raise gfem._mpi.AgreedError("the coarse factorization failed")

    monkeypatch.setattr(DrivenRule, "forward", singular)
    # the host error reaches the caller through JAX's callback
    with pytest.raises(Exception, match="the coarse factorization failed"):
        jax.block_until_ready(jax.jit(s_of)(eps))
    assert coarse.pooled() == 0


@pytest.mark.parametrize("retain_memory", [0, None])
def test_retain_memory_0_empties_the_coarse_pool_after_a_jitted_forward(
    built, monkeypatch, tmp_path, retain_memory
):
    # under jit the forward runs when the compiled function does, after gfem.solve
    # has returned
    s_of, eps = _traced_s(tmp_path, retain_memory)
    real = DrivenRule.forward

    def pooling(self, *args, **kwargs):
        solved = real(self, *args, **kwargs)
        # returned after the forward's last cache walk
        made = _coarse_on(monkeypatch, built, 1 - 2j)
        made.set_up()
        made.free()
        return solved

    monkeypatch.setattr(DrivenRule, "forward", pooling)
    jax.block_until_ready(jax.jit(s_of)(eps))
    assert coarse.pooled() == (0 if retain_memory == 0 else 1)


def test_pooled_coarse_factorizations_count_and_go_at_the_caches_tier_3(
    built, monkeypatch
):
    gfem.release_memory()
    entries = cache.ENTRIES
    older = _coarse_on(monkeypatch, built, 1 - 2j)
    latest = _coarse_on(monkeypatch, built, 1 - 2j)
    older.set_up()
    latest.set_up()
    one = older.nbytes
    assert one > 0
    older.free()
    latest.free()
    assert entries.nbytes >= 2 * one
    # the default budget leaves the most recently returned one its factors
    entries.trim(cache.DefaultBudget(0))
    assert coarse.pooled() == 1
    entries.trim(0)
    assert coarse.pooled() == 0
    assert entries.nbytes == 0


def test_a_coarse_set_up_hands_the_rule_its_columns_seen_stack_kind_and_nodes(
    built, monkeypatch
):
    calls = []

    def rule(_dofs, _size, columns, *, seen, kind, nodes):
        calls.append((columns, seen, kind, nodes))
        return coarse.PT_SCOTCH, 1

    monkeypatch.setattr(coarse, "_kinds", {})
    monkeypatch.setattr(coarse._petsc, "has_parmetis", lambda: False)
    monkeypatch.setattr(coarse._petsc, "has_paraord_cap", lambda: True)
    for shift in (1 - 2j, 3 + 1j):
        made = _coarse_on(monkeypatch, built, shift)
        monkeypatch.setattr(coarse, "rule", rule)
        coarse._made[None] = [made]
        assert coarse.finish(made._comm, columns=12) is made
        made.free()
    assert calls == [(12, False, "capped", (0,)), (12, True, "capped", (0,))]


@pytest.mark.parametrize(
    ("ordering", "workspace"), [(coarse.PT_SCOTCH, 120), (coarse.METIS, 60)]
)
def test_a_coarse_factorization_starts_at_its_orderings_workspace(
    built, monkeypatch, ordering, workspace
):
    made = _coarse_on(monkeypatch, built, 1 - 2j, ordering=ordering)
    made.set_up()
    assert int(made._inner_factor().getMumpsIcntl(14)) == workspace
    made.free()


def test_a_coarse_set_up_logs_its_ordering_ranks_and_stack_kind(
    built, monkeypatch, caplog
):
    monkeypatch.setattr(coarse, "_kinds", {None: coarse.CAPPED})
    made = _coarse_on(monkeypatch, built, 1 - 2j)
    with caplog.at_level("INFO", logger="gfem.solver"):
        made.set_up()
    made.free()
    (record,) = [r for r in caplog.records if "coarse problem" in r.message]
    assert "pt-scotch on 1 of 1 ranks" in record.message
    assert "capped PETSc stack" in record.message


def test_a_stock_stack_says_the_ranks_its_rule_took_not_pt_scotchs(
    built, monkeypatch, caplog
):
    # PT-Scotch would take 5 ranks; the rule took METIS on the one rank there is
    monkeypatch.setattr(coarse, "_kinds", {None: coarse.STOCK})
    monkeypatch.setattr(coarse, "_said_stock", [])
    monkeypatch.setattr(coarse, "_scotch_ranks", lambda *_: 5)
    monkeypatch.setattr(coarse, "capped_ranks", lambda *_: 4)
    made = _coarse_on(monkeypatch, built, 1 - 2j, ordering=coarse.METIS)
    with caplog.at_level("INFO", logger="gfem.solver"):
        made.set_up()
    made.free()
    (record,) = [r for r in caplog.records if "rebuilt" in r.message]
    assert "runs on 1 ranks" in record.message
    assert "would then run on 4 ranks" in record.message
