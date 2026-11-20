r"""An eigenmode solve on a model with one plain periodic pair, across ranks (ADR 0071).

The case is `periodic_cases.c3`, the two-layer cell periodic in x, solved on one
process with ``eigen_solver="slepc"`` and ``direct`` and on 2, 3 and 4 ranks with
``eigen_solver`` left at ``"auto"``. The modes must be one process's: `omega` to
`OMEGA_GATE`, the mode fields through `gfem.Field.at` to `FIELD_GATE` with their
sign, and the eigenfrequency gradient with respect to a `gfem.MaterialField` on the
low layer to `GRADIENT_GATE`. The divergence-free projection between the periodic
spaces leaves no gradient part: $\|G^T M_0 P x\| / \|G^T M_0 x\|$ below
`RESIDUAL_GATE`, the load $G^T M_0 y$ assembled as $\int \varepsilon\, y \cdot
\nabla \psi$ on the periodic H1 space, independently of the projection's own $G$.
`on_ranks` checks that every rank holds the same bits. Every solve needs the MPI
PETSc stack (SLEPc), so the tests skip without it.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
from ranks import on_ranks

import gfem
from gfem import _petsc

#: Decision 06, periodic-eigenmode slice: relative to each mode's |omega|.
OMEGA_GATE = 1e-10
#: Relative to the largest one-process value, sign included.
FIELD_GATE = 1e-8
#: Relative to the largest one-process entry.
GRADIENT_GATE = 1e-9
#: The projection's residual, relative.
RESIDUAL_GATE = 1e-10
RANKS = (2, 3, 4)
LIMIT = 600
N_MODES = 2

pytestmark = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


@pytest.fixture(scope="module")
def mesh_path(tmp_path_factory):
    """C3's mesh file: pytest's one process writes it, ranks only read it."""
    directory = tmp_path_factory.mktemp("c3")
    pc.c3_mesh(directory)
    return str(directory / "c3.msh")


def _mesh(path):
    return gfem.Mesh.from_gmsh(Path(path), length_unit="mm")


def _model(mesh, eps=None):
    """C3, with the low layer a `gfem.MaterialField` of ``eps`` when given."""
    model = pc.c3(mesh)
    if eps is None:
        return model
    materials = dict(model.materials)
    materials["low"] = gfem.MaterialField(eps_r=eps)
    return model.replace(materials=materials)


def _options(eigen_solver):
    return gfem.SolverOptions(
        order=pc.C3_ORDER,
        linear_solver="direct",
        eigen_solver=eigen_solver,
        tolerance=1e-12,
        threads=1,
    )


def _points(mesh, n=12, seed=5):
    """Points well inside ``n`` tets, off every face, where the field is smooth."""
    tets = np.asarray(mesh.tetrahedra)
    chosen = np.random.default_rng(seed).choice(len(tets), n, replace=False)
    return np.asarray(mesh.nodes)[tets[chosen]].mean(axis=1)


def _eps(mesh):
    n = len(mesh.elements("low"))
    return np.random.default_rng(3).uniform(1.8, 2.2, n)


# --- the functions each rank runs ------------------------------------------------


def _modes(path, eigen_solver):
    """``omega`` and the mode fields at points, one thread a rank."""
    mesh = _mesh(path)
    problem = gfem.Eigenmode(target=pc.C3_PLAIN_TARGET, n_modes=N_MODES)
    result = gfem.solve(_model(mesh), problem, options=_options(eigen_solver))
    points = _points(mesh)
    return {
        "omega": np.asarray(result.omega),
        "field": np.array([np.asarray(f.at(points)) for f in result.e_fields]),
    }


def _gradient(path, eigen_solver):
    """The eigenfrequencies and their gradient by the low layer's ``eps_r``."""
    mesh = _mesh(path)
    problem = gfem.Eigenmode(target=pc.C3_PLAIN_TARGET, n_modes=N_MODES)
    options = _options(eigen_solver)

    def objective(eps):
        omega = gfem.solve(_model(mesh, eps), problem, options=options).omega
        return jnp.real(omega[0]) + 0.5 * jnp.real(omega[1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(_eps(mesh)))
    return {"value": float(value), "grad": np.asarray(grad)}


def _residual(path):
    r"""$\|G^T M_0 P x\| / \|G^T M_0 x\|$ of a smooth field $x$, free H1 DOFs."""
    import ngsolve as ngs

    from gfem import _mpi
    from gfem._engine import quadrature
    from gfem._engine.cache import discretization
    from gfem._engine.coefficients import coefficient
    from gfem._engine.domain import potential_surfaces, space_key
    from gfem._engine.driven import driven_operator
    from gfem._engine.eigen import projector, real_eps
    from gfem._engine.runtime import task_manager
    from gfem._linalg.base import free_mask
    from gfem._linalg.layout import ONE_PROCESS, RankLayout

    comm = _mpi.resolve_comm(None)
    model = _model(_mesh(path))
    order = pc.C3_ORDER
    with task_manager(1, comm):
        op = driven_operator(model, order, comm=comm)
        divfree = projector(model, op)
        built = op.built
        disc = discretization(model.mesh, model._periodic, built.comm)
        fixed = potential_surfaces(model)
        h1_key = space_key(model, "h1", order, periodic=True, phases=None, fixed=fixed)
        # both built by the projector: these look them up
        h1 = disc.space(h1_key, lambda: None)
        hcurl = disc.space(
            space_key(model, "hcurl", order, periodic=True, real=True), lambda: None
        )
        h1_layout = ONE_PROCESS
        if comm is not None:
            h1_layout = disc.space((*h1_key, "layout"), lambda: None)
            assert isinstance(h1_layout, RankLayout)
        hfree = free_mask(h1.FreeDofs(), h1.ndof)
        vfree = free_mask(op.space.FreeDofs(), op.space.ndof)
        field = ngs.GridFunction(op.space)
        field.Set(
            ngs.CF(
                (
                    ngs.sin(300 * ngs.y) + 1e3 * ngs.x * ngs.z,
                    1e4 * ngs.x * ngs.z - ngs.cos(500 * ngs.x),
                    ngs.cos(700 * ngs.x) * ngs.sin(400 * ngs.y),
                )
            )
        )
        if op.layout is not None:
            field.vec.Cumulate()
        x = np.where(vfree, field.vec.FV().NumPy(), 0)
        eps = coefficient(built, real_eps(model))
        psi = h1.TestFunction()

        def load(y):
            r"""$G^T M_0 y$ as $\int \varepsilon\, y \cdot \nabla \psi$, a load."""
            gf = ngs.GridFunction(hcurl)
            gf.vec.FV().NumPy()[:] = y
            if op.layout is not None:
                op.layout.mark_state(gf.vec)
            form = ngs.LinearForm(h1)
            form += (eps * gf) * ngs.grad(psi) * quadrature.dx(order)
            form.Assemble()
            return np.where(hfree, form.vec.FV().NumPy(), 0)

        px = np.where(vfree, divfree(x), 0)
        before = h1_layout.norm(load(x.real))
        after = h1_layout.norm(load(px.real))
        divfree.free()
    return {"before": float(before), "residual": float(after / before)}


# --- the tests ---------------------------------------------------------------------


def _relative(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("ranks", [1, *RANKS])
def test_the_periodic_projection_leaves_no_gradient_part(mesh_path, ranks):
    got, *_ = on_ranks(_residual, mesh_path, ranks=ranks, limit=LIMIT)
    print(f"ranks={ranks} residual={got['residual']:.3g}")  # noqa: T201
    assert got["before"] > 0
    assert got["residual"] < RESIDUAL_GATE


@pytest.mark.parametrize("ranks", RANKS)
def test_a_periodic_cell_has_one_process_modes_and_fields(mesh_path, ranks):
    (one,) = on_ranks(_modes, mesh_path, "slepc", ranks=1, limit=LIMIT)
    got, *_ = on_ranks(_modes, mesh_path, "auto", ranks=ranks, limit=LIMIT)
    error = float(np.max(np.abs(got["omega"] - one["omega"]) / np.abs(one["omega"])))
    print(f"ranks={ranks} omega error={error:.3g}")  # noqa: T201
    assert error < OMEGA_GATE, f"omega off by {error:.3g} relative"
    field = _relative(got["field"], one["field"])
    print(f"ranks={ranks} field error={field:.3g}")  # noqa: T201
    assert field < FIELD_GATE, f"the mode field off by {field:.3g} relative"


@pytest.mark.parametrize("ranks", RANKS)
def test_a_periodic_cell_has_one_process_eigenfrequency_gradients(mesh_path, ranks):
    (one,) = on_ranks(_gradient, mesh_path, "slepc", ranks=1, limit=LIMIT)
    got, *_ = on_ranks(_gradient, mesh_path, "auto", ranks=ranks, limit=LIMIT)
    value = abs(got["value"] - one["value"]) / abs(one["value"])
    print(f"ranks={ranks} value error={value:.3g}")  # noqa: T201
    assert value < OMEGA_GATE
    error = _relative(got["grad"], one["grad"])
    print(f"ranks={ranks} gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"the gradient off by {error:.3g} relative"
