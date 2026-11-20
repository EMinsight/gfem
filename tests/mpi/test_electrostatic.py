"""Electrostatic solves across ranks (ADRs 0500, 0501).

Every row compares ranks with the same function in pytest's one process: replicated
values bit for bit across ranks (`on_ranks`), and each value to its solver's gate in
`GATES`, relative to the largest one-process entry of one process's `direct`. `direct`
is PETSc MUMPS, on the MPI stack only; BDDC runs CG on owned-entry products with
its real coarse problem on rank 0. BDDC rows run on the capacitor and the split
bars, not on the thin stack, where real BDDC with CG stalls near 5e-10.
"""

from pathlib import Path

import gradient_cases as cases
import jax
import jax.numpy as jnp
import meshio
import mpi_cases
import ngsolve as ngs
import numpy as np
import pytest
import refinement_cases as rc
from anisotropy import TILTED
from finite_differences import assert_gradient_matches, unit_directions
from ranks import on_ranks

import gfem
from gfem import _mpi, _petsc
from gfem._engine import fields as engine_fields
from gfem._engine.cache import ENTRIES
from gfem._engine.electrostatic import ElectrostaticRule

#: Relative to the largest one-process entry: ADR 0070's gates.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-11}
#: A bundle's gates, as `tests/mpi/test_job.py`'s.
BUNDLE_GATES = {"bddc_gmres": 1e-9, "direct": 1e-13}
needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")
SOLVERS = ("bddc_gmres", pytest.param("direct", marks=needs_stack))
#: The capacitor's problem: one terminal over two dielectric layers.
CAPACITOR = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
STEPS = {"eps_r": 1e-2}
#: A static permittivity law: eps_inf and one Debye strength.
LAW = gfem.PermittivityLaw(
    eps_inf=4.0, terms=(gfem.Debye(delta_eps=2.0, relaxation_frequency=3e9),)
)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _case(name, directory):
    """``(model, problem, order)`` of a case by name."""
    directory = Path(directory)
    if name == "stack":
        return mpi_cases.stack(directory), cases.STACK_PROBLEM, 2
    if name == "capacitor":
        return cases.capacitor(directory), CAPACITOR, 3
    if name == "bars":
        return rc.bars_split_model(directory, "near", 2), rc.BARS_PROBLEM, 3
    raise ValueError(name)


def _options(solver, order, **kwargs):
    return gfem.SolverOptions(
        order=order, linear_solver=solver, tolerance=1e-12, threads=1, **kwargs
    )


def _relative(a, b) -> float:
    b = np.asarray(b)
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def _assert_below(a, b, gate, what):
    error = _relative(a, b)
    print(f"{what} error={error:.3g}")  # noqa: T201
    assert error < gate, f"{what} off by {error:.3g} relative"


def _inside(mesh, n=12, seed=3):
    """Points well inside ``n`` tets, off every face."""
    chosen = np.random.default_rng(seed).choice(len(mesh.tetrahedra), n, replace=False)
    return np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)


# --- capacitance and energy ----------------------------------------------------------


def _solved(name, directory, solver):
    model, problem, order = _case(name, directory)
    result = gfem.solve(model, problem, options=_options(solver, order))
    return {
        "capacitance": np.asarray(result.capacitance),
        "energy": np.stack([np.asarray(e) for e in result.energy.values()]),
    }


def _rows(name, solver, ranks):
    marks = needs_stack if solver == "direct" else ()
    return [pytest.param(name, solver, r, marks=marks) for r in ranks]


@pytest.mark.parametrize(
    ("name", "solver", "ranks"),
    [
        # 16 ranks on the small stack: ranks with few or no DOFs
        *_rows("stack", "direct", (3, 4, 8, 16)),
        *_rows("bars", "direct", (3, 4, 8)),
        *_rows("capacitor", "bddc_gmres", (3, 4, 8)),
        *_rows("bars", "bddc_gmres", (3, 4, 8)),
    ],
)
def test_capacitance_and_energy_match_one_process(tmp_path, name, solver, ranks):
    one = _solved(name, tmp_path, "direct")
    got, *_ = on_ranks(_solved, name, tmp_path, solver, ranks=ranks, limit=300)
    for key in ("capacitance", "energy"):
        _assert_below(
            got[key], one[key], GATES[solver], f"{name} {solver} ranks={ranks} {key}"
        )


# --- potentials as pieces ------------------------------------------------------------


def _pieces(name, directory, solver, points):
    comm = _world()
    model, problem, order = _case(name, directory)
    result = gfem.solve(model, problem, options=_options(solver, order))
    fields = list(result.potential.values())
    first = fields[0]
    width, ndof = engine_fields.dof_shape(
        model.mesh, first.support, first._excluded, "h1", order, comm=comm
    )
    shaped = all(f.coefficients.shape == (width,) for f in fields)
    padded = all(not np.asarray(f.coefficients)[ndof:].any() for f in fields)
    return {
        "width": width,
        "shaped": _mpi.agree_all(comm, flag=shaped),
        "padded": _mpi.agree_all(comm, flag=padded),
        "at": np.stack([np.asarray(f.at(points)) for f in fields]),
    }


@pytest.mark.parametrize("ranks", [3, 4])
@pytest.mark.parametrize(
    ("name", "solver"),
    [
        pytest.param("stack", "direct", marks=needs_stack),
        ("capacitor", "bddc_gmres"),
    ],
)
def test_potentials_are_pieces_that_probe_as_one_process(tmp_path, name, solver, ranks):
    model, problem, order = _case(name, tmp_path)
    one = gfem.solve(model, problem, options=_options("direct", order))
    points = _inside(model.mesh)
    expected = np.stack([np.asarray(f.at(points)) for f in one.potential.values()])
    got, *_ = on_ranks(_pieces, name, tmp_path, solver, points, ranks=ranks, limit=300)
    assert got["shaped"]
    assert got["padded"]
    # the largest rank's rows: below one process's, above zero
    assert 0 < got["width"] < one.potential[problem.terminal_names[0]].coefficients.size
    _assert_below(got["at"], expected, GATES[solver], f"{name} {solver} Field.at")


# --- the H1 map, the bundle and one .vtu ----------------------------------------------


def _walked(mesh, support, order, comm):
    """``(rule key, DOF) -> DOF`` of every entity by `GetDofNrs`, in gfem's numbers."""
    built, fes, _ = engine_fields.fe_space(mesh, support, (), "h1", order, comm=comm)
    ngm, node_of = built.mesh, built.node_order
    rows = {}
    for nodes, node_type in (
        (range(ngm.nv), ngs.VERTEX),
        ((e.nr for e in ngm.edges), ngs.EDGE),
        ((f.nr for f in ngm.faces), ngs.FACE),
    ):
        for nr in nodes:
            node = ngs.NodeId(node_type, nr)
            dofs = fes.GetDofNrs(node)
            ids = (
                [int(node_of[nr])]
                if node_type == ngs.VERTEX
                else sorted(int(node_of[v.nr]) for v in ngm[node].vertices)
            )
            key = (len(ids), *ids, *[-1] * (3 - len(ids)))
            rows |= {(*key, j): d for j, d in enumerate(dofs) if d >= 0}
    for el in range(ngm.ne):
        dofs = fes.GetDofNrs(ngs.NodeId(ngs.CELL, el))
        key = (4, int(built.tetrahedron_order[el]), -1, -1)
        rows |= {(*key, j): d for j, d in enumerate(dofs) if d >= 0}
    return rows, int(fes.ndof)


#: The H1 map's layouts: the whole box, and one volume of it.
SUPPORTS = (("air", "diel", "metal"), ("diel",))


def _map_against_the_walk(serial):
    comm = _world()
    mesh = mpi_cases.permuted(mpi_cases.box(4))
    same = True
    for (support, order), table in serial.items():
        rows, ndof = _walked(mesh, support, order, comm)
        walked = np.full(ndof, -1, dtype=np.int64)
        for key, dof in rows.items():
            walked[dof] = table[key]
        numbers = engine_fields.one_process_dofs(
            mesh, support, (), order, comm, space="h1"
        )
        same &= bool(np.array_equal(numbers, walked) and (walked >= 0).all())
    return _mpi.agree_all(comm, flag=same)


@pytest.mark.parametrize("ranks", [3, 8])
def test_the_h1_map_is_the_walk(ranks):
    mesh = mpi_cases.permuted(mpi_cases.box(4))
    serial = {
        (support, order): _walked(mesh, support, order, None)[0]
        for support in SUPPORTS
        for order in (1, 2, 3)
    }
    same, *_ = on_ranks(_map_against_the_walk, serial, ranks=ranks, limit=300)
    assert same


def _job(directory, solver):
    model, problem, order = _case("bars", directory)
    return gfem.Job(model, problem, _options(solver, order))


def _run_bundle(directory, out, vtu):
    result = gfem.Job.from_json(Path(directory) / "job.json").run(out)
    result.potential["a"].to_vtu(vtu)
    return np.asarray(result.capacitance)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_bundle_and_a_vtu_read_as_one_process(tmp_path, solver):
    job = _job(tmp_path, solver)
    job.to_json(tmp_path / "job.json")
    out, vtu = tmp_path / "out", tmp_path / "a.vtu"
    on_ranks(_run_bundle, tmp_path, out, vtu, ranks=4, limit=300)
    one = _job(tmp_path, "direct").run()
    loaded = gfem.load_result(out)
    gate = BUNDLE_GATES[solver]
    for name, field in one.potential.items():
        got = loaded.potential[name].coefficients
        assert got.shape == field.coefficients.shape
        _assert_below(got, field.coefficients, gate, f"bundle {solver} {name}")
    _assert_below(loaded.capacitance, one.capacitance, gate, f"bundle {solver} C")
    expected = tmp_path / "one.vtu"
    one.potential["a"].to_vtu(expected)
    read, want = meshio.read(vtu), meshio.read(expected)
    np.testing.assert_array_equal(read.points, want.points)
    np.testing.assert_array_equal(read.cells_dict["tetra"], want.cells_dict["tetra"])
    assert read.point_data.keys() == want.point_data.keys()
    for key, values in want.point_data.items():
        _assert_below(read.point_data[key], values, gate * 100, f"vtu {solver} {key}")


# --- gradients -----------------------------------------------------------------------


def _capacitor(directory):
    return _case("capacitor", directory)[0]


def _c(model, solver, problem=CAPACITOR, order=3, **kwargs):
    return gfem.solve(model, problem, options=_options(solver, order, **kwargs))


def _eps_field(directory, solver):
    model = _capacitor(directory)
    n = len(model.mesh.elements("oxide"))
    rng = np.random.default_rng(11)

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        return _c(model.with_material("oxide", field), solver).capacitance[0, 0] * 1e12

    p = jnp.asarray(rng.uniform(3.0, 5.0, n))
    return objective, p, unit_directions(rng, p.shape, 2)


def _eps_axes(directory, solver):
    model = _capacitor(directory)
    n = len(model.mesh.elements("nitride"))
    rng = np.random.default_rng(12)

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps, axes=TILTED)
        return (
            _c(model.with_material("nitride", field), solver).capacitance[0, 0] * 1e12
        )

    return objective, jnp.asarray(rng.uniform(4.0, 8.0, (n, 3))), []


def _uniform(directory, solver):
    model = _capacitor(directory)

    def objective(field):
        return _c(model.with_material("oxide", field), solver).capacitance[0, 0] * 1e12

    field = gfem.MaterialField.uniform("oxide", gfem.Material(eps_r=3.9))
    return objective, field, []


def _law(directory, solver):
    model = _capacitor(directory)

    def objective(law):
        material = gfem.Material(eps_r=law)
        result = _c(model.with_material("nitride", material), solver)
        return result.capacitance[0, 0] * 1e12

    return objective, LAW, []


def _nodes(directory, solver):
    model = _capacitor(directory)
    mesh = model.mesh
    structure = jax.tree_util.tree_structure(mesh)

    def objective(x):
        moved = model.replace(mesh=jax.tree_util.tree_unflatten(structure, [x]))
        return _c(moved, solver).capacitance[0, 0] * 1e12

    return objective, jnp.asarray(mesh.nodes), []


def _energy(directory, solver):
    model = _capacitor(directory)
    n = len(model.mesh.elements("oxide"))

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        energy = _c(model.with_material("oxide", field), solver).energy
        return (energy["oxide"][0] + 0.3 * energy["nitride"][0]) * 1e12

    return objective, jnp.linspace(3.0, 5.0, n), []


def _potential(directory, solver):
    model = _capacitor(directory)
    n = len(model.mesh.elements("oxide"))
    points = _inside(model.mesh)

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        potential = _c(model.with_material("oxide", field), solver).potential["top"]
        return jnp.sum(potential.at(points) ** 2)

    return objective, jnp.linspace(3.0, 5.0, n), []


def _bars(directory, solver):
    model, problem, order = _case("bars", directory)
    n = len(model.mesh.elements("left"))
    weights = jnp.asarray([[1.0, -0.4], [0.7, 2.0]])

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        result = _c(model.with_material("left", field), solver, problem, order)
        return jnp.sum(weights * result.capacitance) * 1e18

    return objective, jnp.linspace(1.0, 3.0, n), []


#: Each gradient row: objective, design and finite-difference directions.
ROWS = {
    "eps_field": _eps_field,
    "eps_axes": _eps_axes,
    "uniform": _uniform,
    "law": _law,
    "nodes": _nodes,
    "energy": _energy,
    "potential": _potential,
    "bars": _bars,
}


def _flat(tree):
    return np.concatenate(
        [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree)]
    )


def _gradients(directory, solver, *, differences=False):
    """Every row's gradient; with ``differences`` the first row's samples too."""
    out = {}
    for row, build in ROWS.items():
        objective, p, directions = build(directory, solver)
        out[row] = _flat(jax.grad(objective)(p))
        if differences and directions:
            samples = []

            def sampled(x, objective=objective, samples=samples):
                value = objective(x)
                samples.append(float(value))
                return value

            grad = out[row]
            assert_gradient_matches(
                sampled, p, directions, step=STEPS["eps_r"], grad=grad
            )
            out[f"{row} samples"] = np.asarray(samples)
    return out


@pytest.mark.parametrize("ranks", [3, 4, 8])
@pytest.mark.parametrize("solver", SOLVERS)
def test_gradients_match_one_process(tmp_path, solver, ranks):
    one = _gradients(tmp_path, "direct")
    differences = solver == "bddc_gmres" and ranks == 3
    got, *_ = on_ranks(
        _gradients, tmp_path, solver, differences=differences, ranks=ranks, limit=900
    )
    for row in ROWS:
        assert np.isfinite(got[row]).all()
        _assert_below(
            got[row], one[row], GATES[solver], f"{row} {solver} ranks={ranks}"
        )
    if differences:
        assert len(got["eps_field samples"])


# --- the kept handle -----------------------------------------------------------------


def _counted_forwards(function, *args):
    """``function(*args)`` and how many forwards ran under it."""
    seen = []
    forward = ElectrostaticRule.forward

    def spy(self, model, *, keep):
        seen.append(keep)
        return forward(self, model, keep=keep)

    ElectrostaticRule.forward = spy
    try:
        return function(*args), len(seen)
    finally:
        ElectrostaticRule.forward = forward


def _energy_objective(directory, solver, **kwargs):
    model = _capacitor(directory)
    n = len(model.mesh.elements("oxide"))

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps)
        result = _c(model.with_material("oxide", field), solver, **kwargs)
        return result.energy["oxide"][0] * 1e12

    return objective, jnp.linspace(3.0, 5.0, n)


def _kept(directory, solver, lost):
    ENTRIES.clear()
    objective, p = _energy_objective(directory, solver)
    reference = np.asarray(jax.grad(objective)(p))
    # retain_memory=0: nothing is kept, the backward runs the forward again
    nothing, _ = _energy_objective(directory, solver, retain_memory=0)
    retained, retained_forwards = _counted_forwards(jax.grad(nothing), p)
    # an entry gone on one rank alone: every rank runs its forward again
    _, pull = jax.vjp(objective, p)
    get = ENTRIES.get
    if _world().rank == lost:
        ENTRIES.get = lambda _key: None
    try:
        (after,), lost_forwards = _counted_forwards(pull, jnp.asarray(1.0))
    finally:
        ENTRIES.get = get
    # a capacitance-only backward frees the kept operator, on every rank
    ENTRIES.clear()
    capacitance, q, _ = _eps_field(directory, solver)
    _, pull = jax.vjp(capacitance, q)
    (entry,) = ENTRIES._entries.values()
    held = entry.handle.nbytes > 0
    pull(jnp.asarray(1.0))
    return {
        "reference": reference,
        "retained": np.asarray(retained),
        "retained forwards": retained_forwards,
        "after": np.asarray(after),
        "lost forwards": lost_forwards,
        # rank 0 holds no DOFs, so nothing of a BDDC operator
        "held": _mpi.agree_any(_world(), flag=held),
        "freed": _mpi.agree_all(_world(), flag=entry.handle is None),
    }


@pytest.mark.parametrize("lost", [0, 1])
@pytest.mark.parametrize("solver", SOLVERS)
def test_the_kept_handle_on_every_rank(tmp_path, solver, lost):
    got, *_ = on_ranks(_kept, tmp_path, solver, lost, ranks=3, limit=600)
    gate = GATES[solver]
    assert got["retained forwards"] == 2
    _assert_below(got["retained"], got["reference"], gate, "retain_memory=0")
    assert got["lost forwards"] == 1
    _assert_below(got["after"], got["reference"], gate, f"lost on rank {lost}")
    assert got["held"]
    assert got["freed"]


# --- JAX's transformations ------------------------------------------------------------


def _transforms(directory):
    objective, p, _ = _eps_field(directory, "bddc_gmres")
    grad = jax.grad(objective)
    batch = jnp.stack([p, p * 1.01])
    model, problem, order = _case("bars", directory)
    n = len(model.mesh.elements("left"))

    def outputs(eps):
        field = gfem.MaterialField(eps_r=eps)
        result = _c(model.with_material("left", field), "bddc_gmres", problem, order)
        return jnp.ravel(result.capacitance) * 1e18

    q = jnp.linspace(1.0, 3.0, n)
    return {
        "grad": np.asarray(grad(p)),
        "second": np.asarray(grad(batch[1])),
        "jit": np.asarray(jax.block_until_ready(jax.jit(grad)(p))),
        "vmap": np.asarray(jax.block_until_ready(jax.vmap(grad)(batch))),
        "jacrev": np.asarray(jax.block_until_ready(jax.jacrev(outputs)(q))),
        "rows": np.stack(
            [np.asarray(jax.grad(lambda e, i=i: outputs(e)[i])(q)) for i in range(4)]
        ),
    }


def test_jit_vmap_and_jacrev_match_grad(tmp_path):
    got, *_ = on_ranks(_transforms, tmp_path, ranks=3, limit=900)
    _assert_below(got["jit"], got["grad"], 1e-12, "jit against grad")
    _assert_below(got["vmap"][0], got["grad"], 1e-12, "vmap's first")
    _assert_below(got["vmap"][1], got["second"], 1e-12, "vmap's second")
    _assert_below(got["jacrev"], got["rows"], 1e-12, "jacrev against grad")


# --- what raises on every rank ----------------------------------------------------


def _refusals(directory):
    """Each refused case's feature, then proof that the job goes on."""
    model = mpi_cases.stack(Path(directory))
    problem = mpi_cases.ELECTROSTATIC
    options = {
        "bddc_order_1": gfem.SolverOptions(order=1, linear_solver="bddc_gmres"),
        "bddc_order_1_auto": gfem.SolverOptions(order=1),
        "petsc_preconditioner": gfem.SolverOptions(
            linear_solver="petsc", petsc_options={"pc_type": "ilu"}
        ),
        "device": gfem.SolverOptions(linear_solver="direct", device="cuda"),
    }
    seen = {}
    for key, option in options.items():
        try:
            gfem.solve(model, problem, options=option)
            seen[key] = None
        except gfem.DistributedUnsupportedError as err:
            seen[key] = err.feature
    for key in ("thermal", "thermal_transient", "dc_conduction"):
        refused = mpi_cases.refused(key, Path(directory))
        try:
            gfem.solve(*refused[:2], options=refused[2])
            seen[key] = None
        except gfem.DistributedUnsupportedError as err:
            seen[key] = err.feature
    return seen, _world().allreduce(1)


def test_every_rank_refuses_what_has_no_distributed_path(tmp_path):
    (seen, ranks), *_ = on_ranks(_refusals, tmp_path, ranks=3, limit=300)
    assert ranks == 3
    assert seen == {key: mpi_cases.feature(key) for key in seen}


def _undetermined(directory, solver):
    """The undetermined-potential error on every rank, then a solve that runs.

    A NaN vacuum permittivity makes the agreed capacitance non-finite, as an
    undetermined potential does: a singular block's result is not portable.
    """
    import scipy.constants

    model, problem, order = _case("capacitor", directory)
    options = _options(solver, order)
    epsilon_0 = scipy.constants.epsilon_0
    scipy.constants.epsilon_0 = float("nan")
    try:
        gfem.solve(model, problem, options=options)
        raised = None
    except ValueError as err:
        raised = "undetermined" in str(err)
    finally:
        scipy.constants.epsilon_0 = epsilon_0
    after = gfem.solve(model, problem, options=options).capacitance
    return raised, _world().allreduce(1), np.asarray(after)


@pytest.mark.parametrize("solver", SOLVERS)
def test_an_undetermined_potential_raises_on_every_rank(tmp_path, solver):
    (raised, ranks, after), *_ = on_ranks(
        _undetermined, tmp_path, solver, ranks=3, limit=300
    )
    assert raised is True
    assert ranks == 3
    assert np.isfinite(after).all()
