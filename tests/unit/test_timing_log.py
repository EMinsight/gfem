"""The ``gfem.timing`` records: one per phase occurrence per rank (ADR 0190).

Small driven and eigenmode solves read through ``caplog``: which phases each
solve logs, at which frequency, and the attributes a handler reads (the
owned DOF count, the solver's bytes, the Krylov iterations). Timings are
observations: no test reads ``seconds`` beyond its type.
"""

import logging

import pytest
import structured_meshes as sm
from eigen_cases import line_mesh, terminated_line

import gfem

F = (21e9, 22e9)
LOGGER = "gfem.timing"
#: The H(curl) DOFs of the strip line below at the default order 2: two per
#: edge and two per face of first-kind Nédélec elements, on its 81 edges and
#: 94 faces (36 tetrahedra).
LINE_DOFS = 2 * 81 + 2 * 94
#: The same count on the eigenmode line (20 cells long): 285 edges, 322 faces.
EIGEN_DOFS = 2 * 285 + 2 * 322
PHASES = ("discretize", "assemble", "setup", "solve")


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=2.0)},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture
def records(caplog):
    gfem.release_memory()
    caplog.set_level(logging.INFO, logger=LOGGER)

    def read(phase=None):
        return [
            r
            for r in caplog.records
            if r.name == LOGGER and (phase is None or r.phase == phase)  # ty: ignore[unresolved-attribute]
        ]

    yield read
    gfem.release_memory()


def _driven(line, solver, frequency=F, **options):
    options = gfem.SolverOptions(linear_solver=solver, **options)
    return gfem.solve(line, gfem.Driven(frequency=frequency), options=options)


def _check_every_record(logged, problem):
    for r in logged:
        assert r.levelno == logging.INFO
        assert r.phase in PHASES
        assert r.problem == problem
        assert isinstance(r.seconds, float)
        assert r.seconds >= 0
        assert (r.rank, r.ranks) == (0, 1)


def _check_driven(records, solver):
    logged = records()
    _check_every_record(logged, "driven")
    (discretize,) = records("discretize")
    assert discretize.ndof == LINE_DOFS
    assert discretize.frequency is None
    for f in F:
        at = [r for r in logged if r.frequency == pytest.approx(f)]
        assert {r.phase for r in at} == {"assemble", "setup", "solve"}
        (solve,) = [r for r in at if r.phase == "solve"]
        assert isinstance(solve.iterations, tuple)
        for setup in (r for r in at if r.phase == "setup"):
            assert setup.solver == solver
            assert setup.action is not None
            assert setup.nbytes > 0
    return logged


def test_a_direct_sweep_logs_every_phase_at_every_frequency(line, records):
    _driven(line, "direct")
    logged = _check_driven(records, "direct")
    assert [r.iterations for r in records("solve")] == [(), ()]
    # discretize comes first, each frequency's solve after its setup
    assert logged[0].phase == "discretize"


def test_a_bddc_sweep_logs_every_phase_and_its_iterations(line, records):
    _driven(line, "bddc_gmres")
    _check_driven(records, "bddc_gmres")
    solves = records("solve")
    assert [len(r.iterations) for r in solves] == [2, 2]  # per excited port
    assert all(its > 0 for r in solves for its in r.iterations)
    # set up at the first frequency, reused at the second
    assert [r.action for r in records("setup")] == ["factorized", "reused"]


@pytest.mark.petsc
def test_a_petsc_sweep_logs_every_phase(line, records):
    _driven(line, "petsc")
    _check_driven(records, "petsc")


def test_nothing_is_logged_below_info(line, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    _driven(line, "direct")
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_timing_records_leave_the_solver_logger_silent(line, records, caplog):
    _driven(line, "direct")
    assert records()
    assert not [r for r in caplog.records if r.name == "gfem.solver"]


def test_an_eigenmode_search_logs_its_setups_and_one_solve(tmp_path, records):
    model = terminated_line(line_mesh(tmp_path), 1000.0)
    gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=1))
    logged = records()
    _check_every_record(logged, "eigenmode")
    assert {r.phase for r in logged} == set(PHASES)
    (discretize,) = records("discretize")
    assert discretize.ndof == EIGEN_DOFS
    (solve,) = records("solve")
    assert solve.iterations == ()
    for r in logged:
        assert r.frequency is None
    for setup in records("setup"):
        assert setup.solver == "direct"
        assert setup.nbytes > 0
