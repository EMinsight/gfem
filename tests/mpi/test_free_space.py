"""Sources and their measurements across ranks: one process's values.

The dielectric cube of `free_space_cases.scatterer_box` (an air box in a
perfectly matched layer, a lumped port on the cube's +x face), driven by a
plane wave and an electric dipole, on `RANKS` ranks against pytest's one
process:

- each rank assembles its own part of a plane wave's load, and the dipole's
  load sits on the one rank holding its element, here not rank 0;
- the collective measurements of the source columns (S, the port waves, the
  emitted power, the Purcell factor, the kept fields and their scattered and
  incident parts at points) equal one process's to `GATE`;
- the moment interpolant of an incident field is rank-local: a field's
  coefficients on each rank, read back at points, are one process's to
  `EXACT_GATE`, with and without an element set whose edge crosses the
  partition;
- the Purcell factor's gradient by the cube's permittivity is one process's;
- ranks given different backgrounds stop at the entry check, naming it.
"""

import free_space_cases as fs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc

#: NGSolve's rank 0 holds no element: two more ranks share a partition face.
RANKS = 3
#: Relative to the largest one-process value: direct solves of one system
#: split over ranks (the measurements' gate).
GATE = 1e-10
#: The interpolant involves no solve: round-off of the same arithmetic.
EXACT_GATE = 1e-12
#: Relative to the largest one-process entry of the gradient.
GRADIENT_GATE = 1e-9
F = fs.F0 / 2
GRID = (F, 1.1 * F)
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
MOMENT = (0.0, 1e-28, 0.3e-28)
#: Dipole positions in the air, off every face of the Kuhn grid; the first
#: one whose element lies on one rank alone, not rank 0, is used.
CANDIDATES = (
    fs.DIPOLE_POSITION,
    (0.87e-6, 0.59e-6, 0.13e-6),
    (0.13e-6, 0.59e-6, 0.13e-6),
    (0.87e-6, 0.41e-6, 0.87e-6),
    (0.59e-6, 0.13e-6, 0.87e-6),
    (0.41e-6, 0.87e-6, 0.13e-6),
    (0.13e-6, 0.13e-6, 0.41e-6),
    (0.87e-6, 0.87e-6, 0.59e-6),
)

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _options():
    return gfem.SolverOptions(order=2, linear_solver="direct", threads=1)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _points(mesh, n=12, seed=3):
    """Centroids of tets outside the layer, off every face."""
    tets = np.concatenate([mesh.elements("air"), mesh.elements("scat")])
    chosen = np.random.default_rng(seed).choice(tets, n, replace=False)
    return np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)


def _holders(model, position, comm):
    """The ranks whose local mesh holds the dipole's element."""
    from gfem._engine.cache import discretization
    from gfem._engine.probe import Layout, locate

    layout = Layout(("air", "pml", "scat"), (), "hcurl", 2)
    tet = int(locate(model.mesh, layout, np.array([position]), None).tetrahedra[0])
    built = discretization(model.mesh, (), comm).built
    held = bool(np.isin(tet, np.asarray(built.cell_order)))
    return [r for r, h in enumerate(comm.allgather(held)) if h]


def _off_root(model, comm):
    """The first candidate whose element one rank alone holds, not rank 0."""
    for position in CANDIDATES:
        holders = _holders(model, position, comm)
        if len(holders) == 1 and holders[0] != 0:
            return position, holders
    raise AssertionError("no candidate position lies on one rank alone")


def _measurements(model, position):
    """Every collective value of the two source columns, on this process."""
    dipole = gfem.ElectricDipole("d", position=position, moment=MOMENT)
    problem = gfem.Driven(GRID, sources=(WAVE, dipole), keep_fields=True)
    result = gfem.solve(model, problem, options=_options())
    points = _points(model.mesh)
    fields = {
        f"{label} {part} {k}": np.asarray(
            result.e_field(f, label, part=part).at(points)
        )
        for k, f in enumerate(GRID)
        for label, parts in (
            ("w", ("total", "scattered", "incident")),
            ("d", ("total",)),
        )
        for part in parts
    }
    return {
        "s": np.asarray(result.s.values),
        "port waves": np.asarray(result.port_waves),
        "emitted power": np.asarray(result.emitted_power["d"]),
        "purcell factor": np.asarray(gfem.purcell_factor(result, "d", model=model)),
        **fields,
    }


def _on_ranks_measurements(path):
    comm = _world()
    model = fs.scatterer_model(path)
    position, holders = _off_root(model, comm)
    return {"position": position, "holders": holders} | _measurements(model, position)


@pytest.fixture
def box(tmp_path):
    """The scatterer box, written once, before the ranks read it."""
    return str(fs.scatterer_box(tmp_path / "box.msh"))


def _relative(a, b):
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


@needs_stack
def test_source_measurements_across_ranks_are_one_process_s(box):
    got, *_ = on_ranks(_on_ranks_measurements, box, ranks=RANKS, limit=300)
    # the dipole's load is assembled on one rank alone, not rank 0
    assert len(got["holders"]) == 1
    assert got["holders"][0] != 0
    one = _measurements(fs.scatterer_model(box), got["position"])
    for name, value in one.items():
        error = _relative(got[name], value)
        print(f"ranks={RANKS} {name} error={error:.3g}")  # noqa: T201
        assert error < GATE, f"{name} off by {error:.3g} relative"


# --- the moment interpolant -----------------------------------------------------------


def _interpolated(path, masked, comm=None):
    """A plane wave's moment interpolant as a field, read at points."""
    from gfem._engine import fields as engine_fields
    from gfem._engine.moments import field_interpolant
    from gfem._incident import values

    mesh = gfem.Mesh.from_gmsh(path)
    support, order = ("air", "pml", "scat"), 2
    # the elements below a slanted plane: its edge crosses any partition
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)].mean(axis=1)
    elements = centroids @ np.array([1.0, 0.6, 0.3]) < 1.0e-6 if masked else None
    interpolant = field_interpolant(
        mesh, support, (), order, elements=elements, comm=comm
    )
    coefficients = interpolant(
        values(WAVE, gfem.Background([gfem.Material()]), interpolant.points, F)
    )
    width, ndof = engine_fields.dof_shape(mesh, support, (), "hcurl", order, comm=comm)
    piece = np.zeros(width, complex)
    piece[:ndof] = coefficients
    field = gfem.Field(
        coefficients=jnp.asarray(piece),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=order,
        support=support,
        omega=jnp.asarray(2 * np.pi * F),
        _comm=_mpi.static_comm(comm),
    )
    return np.asarray(field.at(_points(mesh)))


def _on_ranks_interpolated(path, masked):
    return _interpolated(path, masked, _world())


@pytest.mark.parametrize("masked", [False, True], ids=["domain", "element set"])
def test_the_moment_interpolant_is_one_process_s_across_ranks(box, masked):
    got, *_ = on_ranks(_on_ranks_interpolated, box, masked, ranks=RANKS, limit=120)
    one = _interpolated(box, masked)
    error = _relative(got, one)
    print(f"ranks={RANKS} masked={masked} interpolant error={error:.3g}")  # noqa: T201
    assert error < EXACT_GATE, f"off by {error:.3g} relative"


# --- the Purcell factor's gradient ----------------------------------------------------


def _purcell_gradient(path, position):
    model = fs.scatterer_model(path, field=True)
    dipole = gfem.ElectricDipole("d", position=position, moment=MOMENT)
    problem = gfem.Driven(F, excite=(), sources=dipole)

    def purcell(eps):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(moved, problem, options=_options())
        return gfem.purcell_factor(result, "d", model=moved)[0]

    eps = jnp.asarray(model.materials["scat"].eps_r)
    value, grad = jax.value_and_grad(purcell)(eps)
    return np.asarray(value), np.asarray(grad)


def _on_ranks_purcell_gradient(path):
    position, holders = _off_root(fs.scatterer_model(path), _world())
    return position, holders, *_purcell_gradient(path, position)


@needs_stack
def test_the_purcell_gradient_across_ranks_is_one_process_s(box):
    (position, holders, value, grad), *_ = on_ranks(
        _on_ranks_purcell_gradient, box, ranks=RANKS, limit=600
    )
    assert len(holders) == 1
    assert holders[0] != 0
    value_one, grad_one = _purcell_gradient(box, position)
    assert np.abs(grad_one).max() > 0
    assert _relative(value, value_one) < GATE
    error = _relative(grad, grad_one)
    print(f"ranks={RANKS} Purcell gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"the gradient off by {error:.3g} relative"


# --- the entry check ------------------------------------------------------------------


def _different_backgrounds(path):
    comm = _world()
    model = fs.scatterer_model(path)
    model = model.replace(background=gfem.Material(eps_r=1.0 + 0.5 * comm.rank))
    try:
        gfem.solve(model, gfem.Driven(F, excite=(), sources=WAVE), options=_options())
    except _mpi.AgreedError as err:
        return str(err)
    raise AssertionError("ranks with different backgrounds solved")


def test_ranks_with_different_backgrounds_are_told_at_the_entry(box):
    messages = on_ranks(_different_backgrounds, box, ranks=RANKS, limit=120)
    assert all("the background" in m for m in messages), messages
