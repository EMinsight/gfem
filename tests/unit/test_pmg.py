"""The p-multigrid: its direct set and the topological count of it."""

import logging
import math
import sys
import types

import jax
import jax.numpy as jnp
import mpi_cases
import ngsolve as ngs
import numpy as np
import periodic_cases as pc
import pml_cases
import pytest
import structured_meshes as sm
from preconditioners import count_builds
from strip2d import write_strip

import gfem
from gfem._engine.driven import DrivenRule
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import krylov_handle, pardiso, pmg
from gfem._linalg.krylov_handle import KrylovHandle


def test_direct_set_regular_tetrahedron_aspect():
    # longest edge 1 over shortest altitude sqrt(2/3): sqrt(6)/2
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0.5, math.sqrt(3) / 6, 0]]
    )
    nodes[3, 2] = math.sqrt(2 / 3)
    aspect = pmg.cell_aspect(nodes, np.array([[0, 1, 2, 3]]))
    assert aspect == pytest.approx([math.sqrt(6) / 2])


def test_direct_set_sound_structured_mesh_has_no_poor_cells():
    # n^3 Kuhn cubes: 3n(n+1)^2 axis edges, 3n^2(n+1) face and n^3 body diagonals
    n = 4
    nodes, tets, *_ = _grid(cells=n, shell=0.0, film=())
    whitney = 3 * n * (n + 1) ** 2 + 3 * n**2 * (n + 1) + n**3
    assert pmg.cell_aspect(nodes, tets).max() < 3.5
    for order in (2, 3):
        count = pmg.direct_set_count(nodes, tets, order)
        assert count == (whitney, whitney, 0)


#: A 1 nm film across the unit box (um): its cells have aspect 50 and more.
FILM = (0.5, 0.501)


def _grid(*, cells=4, shell=0.25, film=FILM):
    """[0, 1]^3 "air" in a "pml" shell, cut through by a film at z in ``film``.

    The shell covers every side but +z; the outer faces are "outer". Returns
    `structured_meshes.structured`'s arrays.
    """

    def line(lo, hi, extra=()):
        return np.unique(
            np.r_[np.linspace(lo, hi, round((hi - lo) * cells) + 1), extra]
        )

    edges = (line(-shell, 1 + shell), line(-shell, 1 + shell), line(-shell, 1, film))

    def volume(c):
        return "air" if ((c > 0) & (c < 1)).all() else "pml"

    lo, hi = np.array([e[0] for e in edges]), np.array([e[-1] for e in edges])

    def surface(c):
        return "outer" if np.isclose(c, lo).any() or np.isclose(c, hi).any() else ""

    return sm.structured(edges, volume, surface)


def _box(path, **kw):
    """`_grid` as a `gfem.Mesh` in micrometres."""
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, _grid(**kw)), length_unit="um")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return _box(tmp_path_factory.mktemp("pmg") / "box.msh")


def _built(mesh, order):
    """The mesh's NGSolve space, its cells in NGSolve order and their PML marks."""
    built = to_ngsolve(mesh)
    space = ngs.Compress(ngs.HCurl(built.mesh, order=order, type1=True, complex=True))
    tets = mesh.tetrahedra[built.cell_order]
    pml = mesh.tetrahedron_tags[built.cell_order] == mesh.region("pml").tag
    return space, tets, pml


@pytest.mark.parametrize("order", [2, 3])
def test_direct_set_count_is_the_built_set(box, order):
    mesh = box
    space, tets, pml = _built(mesh, order)
    nodes = np.asarray(mesh.nodes)

    count = pmg.direct_set_count(nodes, tets, order, pml)
    built = pmg.direct_set(space, nodes, tets, np.arange(len(tets)), pml)

    poor = pmg.cell_aspect(nodes, tets) > pmg.POOR_ASPECT
    assert (poor & pml).any()
    assert (poor & ~pml).any()
    assert 0 < count.whitney < count.direct < space.ndof
    assert built.dofs.NumSet() == count.direct
    assert built.poor_cells == count.poor_cells


def test_direct_set_counts_a_cell_both_pml_and_poor_once():
    # one poor PML cell: its order-2 element, 6 edges x 2 + 4 faces x 2 DOFs
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, 0.01]])
    tets = np.array([[0, 1, 2, 3]])
    assert pmg.cell_aspect(nodes, tets)[0] > pmg.POOR_ASPECT
    count = pmg.direct_set_count(nodes, tets, 2, np.array([True]))
    assert count == (20, 6, 1)


def test_direct_set_poor_cell_share_halves_with_h():
    # a spanning film is a surface of cells: its share of the volume goes as h
    share = []
    for cells in (4, 8):
        nodes, tets, *_ = _grid(cells=cells, shell=0.0)
        share.append(pmg.direct_set_count(nodes, tets, 2).poor_cells / len(tets))
    assert share[0] > 0
    assert share[1] < 0.6 * share[0]


def test_direct_set_on_part_of_the_mesh_holds_its_whitney_dofs(box):
    # a space on "air" alone: its cells only, the shell's left out
    mesh = box
    built = to_ngsolve(mesh)
    space = ngs.Compress(
        ngs.HCurl(
            built.mesh,
            order=2,
            type1=True,
            complex=True,
            definedon=built.mesh.Materials("air"),
        )
    )
    air = mesh.tetrahedron_tags[built.cell_order] == mesh.region("air").tag
    tets, nodes = mesh.tetrahedra[built.cell_order][air], np.asarray(mesh.nodes)

    direct = pmg.direct_set(space, nodes, tets, np.flatnonzero(air))

    assert direct.dofs.NumSet() == pmg.direct_set_count(nodes, tets, 2).direct
    assert direct.pml_dofs == 0
    edges = {
        e.nr for el in built.mesh.Elements(ngs.VOL) if air[el.nr] for e in el.edges
    }
    assert all(direct.dofs[space.GetDofNrs(ngs.NodeId(ngs.EDGE, e))[0]] for e in edges)
    with pytest.raises(ValueError, match="own cells"):
        pmg.direct_set(space, nodes, mesh.tetrahedra, np.arange(len(mesh.tetrahedra)))


def test_direct_set_leaves_fixed_dofs_out(box):
    mesh = box
    built = to_ngsolve(mesh)
    space = ngs.Compress(
        ngs.HCurl(built.mesh, order=2, type1=True, complex=True, dirichlet="outer")
    )
    tets, nodes = mesh.tetrahedra[built.cell_order], np.asarray(mesh.nodes)
    fixed = space.ndof - space.FreeDofs().NumSet()

    direct = pmg.direct_set(space, nodes, tets, np.arange(len(tets)))

    assert fixed > 0
    assert (direct.dofs & ~space.FreeDofs()).NumSet() == 0
    assert direct.dofs.NumSet() < pmg.direct_set_count(nodes, tets, 2).direct


# --- pmg_gmres end to end -----------------------------------------------------------

#: The strip line of test_driven: 10 x 2 x 1 mm, eps_r 4, between PEC plates.
LINE = (10.0, 2.0, 1.0)
EPS = 4.0
F = 5e9


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(*LINE, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _s(model, problem=None, **options):
    problem = problem or gfem.Driven(frequency=F)
    return np.asarray(
        gfem.solve(model, problem, options=gfem.SolverOptions(**options)).s.values
    )


@pytest.mark.parametrize("factor", ["pardiso", "sparse_cholesky"])
@pytest.mark.parametrize("order", [2, 3])
def test_pmg_gmres_matches_direct(line, monkeypatch, order, factor):
    if factor == "pardiso" and not pardiso.pardiso_available():
        pytest.skip("the mkl extra is not installed")
    if factor == "sparse_cholesky":
        # without MKL the direct level is NGSolve's sparse Cholesky
        monkeypatch.setattr(pmg, "pardiso_inverse", lambda *_, **__: None)
    s = _s(line, order=order, linear_solver="pmg_gmres")
    direct = _s(line, order=order, linear_solver="direct")
    np.testing.assert_allclose(s, direct, rtol=0, atol=1e-6)


def test_each_pmg_build_logs_its_direct_set_and_is_timed_as_setup(line, caplog):
    caplog.set_level(logging.INFO, logger="gfem.solver")
    caplog.set_level(logging.INFO, logger="gfem.timing")
    problem = gfem.Driven(frequency=(5e9, 5.5e9))
    _s(line, problem, linear_solver="pmg_gmres")
    (build,) = [r for r in caplog.records if hasattr(r, "direct_set")]
    # the free Whitney DOFs alone: no cell is poor or in a layer
    assert build.direct_set == 63
    assert 0 < build.direct_share < 0.5
    assert (build.pml_share, build.poor_cells) == (0.0, 0)
    assert build.frequency == pytest.approx(5e9)
    assert "on the direct set" in build.getMessage()
    setups = [
        r
        for r in caplog.records
        if getattr(r, "phase", None) == "setup" and r.solver == "pmg_gmres"
    ]
    assert setups
    assert setups[0].nbytes > 0


def test_pmg_gmres_solves_a_perfectly_matched_layer_on_its_direct_level(
    tmp_path, caplog
):
    # the TEM line of pml_cases ended by a layer, fed by a wave port
    mesh = gfem.Mesh.from_gmsh(pml_cases.tem_line(tmp_path / "line.msh", cells=6))
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "pml": gfem.Material()},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )
    problem = gfem.Driven(frequency=pml_cases.F0)
    caplog.set_level(logging.INFO, logger="gfem.solver")
    s = _s(model, problem, linear_solver="pmg_gmres")
    (build,) = [r for r in caplog.records if hasattr(r, "direct_set")]
    assert 0 < build.pml_share < build.direct_share
    np.testing.assert_allclose(
        s, _s(model, problem, linear_solver="direct"), rtol=0, atol=1e-6
    )


# --- reuse across a sweep, and the backward on the forward's handle -------------------

TWO_FREQUENCIES = gfem.Driven(frequency=(5e9, 5.5e9))


def test_a_pmg_sweep_builds_once_and_matches_fresh_builds(line, monkeypatch):
    builds = count_builds(monkeypatch, "pmg_gmres")
    options = gfem.SolverOptions(linear_solver="pmg_gmres")
    result = gfem.solve(line, TWO_FREQUENCIES, options=options)
    assert len(builds) == 1
    assert result.sweep.preconditioner_reused == (False, True)
    # without refactorization, a V-cycle is built at each frequency
    monkeypatch.delattr(KrylovHandle, "refactorize")
    builds.clear()
    fresh = _s(line, TWO_FREQUENCIES, linear_solver="pmg_gmres")
    assert len(builds) == 2
    np.testing.assert_allclose(result.s.values, fresh, rtol=0, atol=1e-6)


def test_a_reused_pmg_solve_past_its_cap_builds_again(line, monkeypatch):
    # no slack: the second frequency's columns cap at n_ref and build there
    monkeypatch.setattr(krylov_handle, "_slack", lambda _: 0)
    builds = count_builds(monkeypatch, "pmg_gmres")
    problem = gfem.Driven(frequency=(3e9, 3.5e9))
    s = _s(line, problem, linear_solver="pmg_gmres")
    assert len(builds) == 2
    np.testing.assert_allclose(
        s, _s(line, problem, linear_solver="direct"), rtol=0, atol=1e-6
    )


def test_an_adaptive_pmg_sweep_builds_at_each_sample(line, monkeypatch):
    builds = count_builds(monkeypatch, "pmg_gmres")
    problem = gfem.Driven(frequency=np.linspace(3e9, 6e9, 31), sweep="adaptive")
    options = gfem.SolverOptions(linear_solver="pmg_gmres")
    result = gfem.solve(line, problem, options=options)
    assert len(builds) == len(result.sweep.sampled_frequencies) > 1


def test_the_pmg_backward_builds_nothing_new(line, monkeypatch):
    builds = count_builds(monkeypatch, "pmg_gmres")
    backward = DrivenRule.backward
    during = []

    def counted_backward(self, *args, **kwargs):
        before = len(builds)
        try:
            return backward(self, *args, **kwargs)
        finally:
            during.append(len(builds) - before)

    monkeypatch.setattr(DrivenRule, "backward", counted_backward)
    options = gfem.SolverOptions(linear_solver="pmg_gmres")
    problem = gfem.Driven(frequency=(5e9, 5.5e9), excite="p1")

    def s21(eps):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        s = gfem.solve(model, problem, options=options).s
        return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

    gfem.release_memory()
    jax.grad(s21)(jnp.full(len(line.mesh.tetrahedra), EPS))
    assert len(builds) == 1
    assert during == [0]


def test_a_pmg_sweep_takes_the_same_iterations_every_run(line):
    # the reuse rule counts iterations, never times (each run from no kept
    # state: the second would start from the first's solution)
    problem = gfem.Driven(frequency=(3e9, 3.5e9, 4e9, 6e9))
    rule = DrivenRule(line, problem, gfem.SolverOptions(linear_solver="pmg_gmres"))
    gfem.release_memory()
    first, _ = rule.forward(line, keep=False)
    gfem.release_memory()
    second, _ = rule.forward(line, keep=False)
    assert first["iterations"] == second["iterations"]


def test_a_pmg_design_starts_from_the_previous_designs_solution(line, caplog):
    gfem.release_memory()
    caplog.set_level(logging.INFO, logger="gfem.solver")

    def guesses():
        logged = [r for r in caplog.records if getattr(r, "direction", None)]
        caplog.clear()
        return [r.guess for r in logged if r.direction == "forward"]  # ty: ignore[unresolved-attribute]

    def design(eps):
        return line.with_material("dielectric", gfem.Material(eps_r=eps))

    first = gfem.solve(
        design(EPS),
        TWO_FREQUENCIES,
        options=gfem.SolverOptions(linear_solver="pmg_gmres"),
    )
    assert guesses() == [None, None]
    second = gfem.solve(
        design(1.02 * EPS),
        TWO_FREQUENCIES,
        options=gfem.SolverOptions(linear_solver="pmg_gmres"),
    )
    assert guesses() == ["previous design", None]
    assert max(second.sweep.iterations[0]) < max(first.sweep.iterations[0])
    direct = _s(design(1.02 * EPS), TWO_FREQUENCIES, linear_solver="direct")
    np.testing.assert_allclose(second.s.values, direct, rtol=0, atol=1e-5)
    gfem.release_memory()


# --- the options -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "message"),
    [
        (1, r"order 2 or 3, got order 1: at order 1 its direct level is the whole"),
        (4, r"order 2 or 3, got order 4: its vertex-star blocks above order 3"),
        (
            12,
            r"order 2 or 3, got order 12: .*Use linear_solver='direct' or 'bddc_gmres'",
        ),
    ],
)
def test_pmg_gmres_takes_order_2_or_3_only(order, message):
    with pytest.raises(ValueError, match=rf"linear_solver='pmg_gmres' needs {message}"):
        gfem.SolverOptions(linear_solver="pmg_gmres", order=order)
    for taken in (2, 3):
        gfem.SolverOptions(linear_solver="pmg_gmres", order=taken)


# --- refusals ------------------------------------------------------------------------


def _eigenmode(directory):
    return mpi_cases.refused("arpack", directory)[0], gfem.Eigenmode(F, n_modes=1)


def _electrostatic(directory):
    return mpi_cases.stack(directory), mpi_cases.ELECTROSTATIC


def _thermal(directory):
    return mpi_cases.stack(directory), gfem.SteadyThermal()


def _dc_conduction(directory):
    return mpi_cases.stack(directory), gfem.DCConduction(terminals=mpi_cases.TERMINALS)


def _two_d(directory):
    mesh = gfem.Mesh.from_gmsh(write_strip(directory / "strip.msh"), length_unit="um")
    materials = {name: gfem.Material(eps_r=2.0) for name in "abc"}
    model = gfem.Model(mesh, materials=materials, boundaries=(gfem.PEC("outer"),))
    return model, gfem.Driven(F)


def _bloch(directory):
    problem = gfem.Driven(pc.C1_FREQUENCY, wave_vector=pc.C1_K)
    return pc.c1(pc.c1_mesh(directory)), problem


def _periodic(directory):
    return pc.c1(pc.c1_mesh(directory)), gfem.Driven(pc.C1_FREQUENCY)


def _floquet(directory):
    return pc.f1(pc.f1_mesh(directory)), gfem.Driven(pc.F1_FREQUENCY)


def _strip_line(directory):
    return mpi_cases.refused("device", directory)[0], gfem.Driven(F)


#: Per refusal: the case, extra options, the error, and its message.
REFUSALS = {
    "eigenmode": (
        _eigenmode,
        {},
        ValueError,
        (
            r"linear_solver='pmg_gmres' solves driven problems only, not eigenmode "
            r"problems\. Use linear_solver='direct' \(or 'auto', which picks it\)\.$"
        ),
    ),
    "electrostatic": (
        _electrostatic,
        {},
        ValueError,
        (
            r"solves driven problems only, not electrostatic problems\. Use "
            r"linear_solver='auto', which picks 'bddc_gmres', or 'direct'\.$"
        ),
    ),
    "thermal": (
        _thermal,
        {},
        ValueError,
        r"not steady thermal problems\. Use linear_solver='direct'",
    ),
    "dc_conduction": (
        _dc_conduction,
        {},
        ValueError,
        r"not DC conduction problems\. Use linear_solver='direct'",
    ),
    "2d": (
        _two_d,
        {},
        NotImplementedError,
        (
            r"linear_solver='pmg_gmres' is not available on a 2D mesh\. Use "
            r"linear_solver=\"direct\""
        ),
    ),
    "bloch": (
        _bloch,
        {},
        ValueError,
        (
            r"Driven\.wave_vector sets a Bloch phase, so the operator is not "
            r"symmetric, and linear_solver='pmg_gmres' factors symmetric operators "
            r"only: use linear_solver='direct' or 'bddc_gmres'\.$"
        ),
    ),
    "periodic": (
        _periodic,
        {},
        ValueError,
        (
            r"linear_solver='pmg_gmres' does not solve models with periodic pairs or "
            r"Floquet ports in this version of gfem \(Periodic\(.*\)\)\. Use "
            r"linear_solver='direct' \(or 'auto', which picks it while it fits\)\.$"
        ),
    ),
    "floquet_port": (
        _floquet,
        {},
        ValueError,
        r"periodic pairs or Floquet ports in this version of gfem \(.*FloquetPort",
    ),
    "mpi": (
        _strip_line,
        {"comm": 4},
        gfem.DistributedUnsupportedError,
        (
            r"linear_solver='pmg_gmres' runs on one process in this version of gfem\. "
            r"Across MPI ranks use linear_solver='bddc_gmres' \(or 'auto', which picks "
            r"it\) or 'direct' with the MPI PETSc stack, or run independent "
            r"one-process copies with comm=MPI\.COMM_SELF\.$"
        ),
    ),
    "tree_cotree": (
        _strip_line,
        {"gauge": "tree_cotree"},
        ValueError,
        r"gauge='tree_cotree' needs linear_solver='direct', not 'pmg_gmres'",
    ),
    "gpu": (
        _strip_line,
        {"device": "cuda"},
        ValueError,
        r"device='cuda' needs linear_solver='direct'",
    ),
}


class _Ranks:
    """As much of a communicator as resolution reads."""

    rank = 0

    def __init__(self, size: int) -> None:
        self.size = size


def _with_ranks(monkeypatch, size):
    """A stub Open MPI of ``size`` ranks; its world communicator."""
    world = _Ranks(size)
    mpi = types.SimpleNamespace(
        COMM_WORLD=world,
        COMM_SELF=_Ranks(1),
        get_vendor=lambda: ("Open MPI", (0, 0, 0)),
        Is_initialized=lambda: True,
        Is_finalized=lambda: False,
    )
    module = types.ModuleType("mpi4py")
    module.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", module)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    return world


@pytest.mark.parametrize("refusal", REFUSALS)
def test_refuses_what_it_does_not_solve_before_assembly(refusal, tmp_path, monkeypatch):
    case, extra, error, message = REFUSALS[refusal]
    model, problem = case(tmp_path)
    extra = dict(extra)
    kwargs = {}
    if "comm" in extra:
        kwargs["comm"] = _with_ranks(monkeypatch, extra.pop("comm"))

    def assembled(*_, **__):
        raise AssertionError("assembled before the refusal")

    monkeypatch.setattr(ngs.BilinearForm, "Assemble", assembled)
    monkeypatch.setattr(ngs.LinearForm, "Assemble", assembled)

    def solve():
        options = gfem.SolverOptions(linear_solver="pmg_gmres", **extra)
        gfem.solve(model, problem, options=options, **kwargs)

    with pytest.raises(error, match=message):
        solve()


def test_a_pmg_solve_that_does_not_converge_names_thin_cells(line, monkeypatch):
    init = pmg.PmgHandle.__init__

    def one_iteration(self, *args, **kwargs):
        init(self, *args, **{**kwargs, "maxiter": 1})

    monkeypatch.setattr(pmg.PmgHandle, "__init__", one_iteration)
    with pytest.raises(
        gfem.LinearSolverConvergenceError,
        match=r"p-multigrid \+ GMRES did not converge .*Use linear_solver='direct' "
        r"if it fits, or a looser tolerance, and check the mesh for layers of thin "
        r"cells",
    ):
        _s(line, linear_solver="pmg_gmres", tolerance=1e-12)
