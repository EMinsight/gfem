"""The ``gfem.solver`` records: one per solve step, forward and backward (ADR 0120).

Driven and electrostatic solves on small models, read through ``caplog``:
the attributes a handler sees, under ``jax.value_and_grad`` too, the actions
of a direct base across designs and of a BDDC base across frequencies.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import structured_meshes as sm

import gfem
from gfem._engine.driven import DrivenRule

F = (21e9, 22e9)
LOGGER = "gfem.solver"


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material()},
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

    def read(direction=None):
        return [
            r
            for r in caplog.records
            if r.name == LOGGER and (direction is None or r.direction == direction)  # ty: ignore[unresolved-attribute]
        ]

    yield read
    gfem.release_memory()


def _with_eps(line, eps):
    return line.with_material("dielectric", gfem.MaterialField(eps_r=eps))


def _s(line, eps, *, frequency=F, solver="direct", excite=None):
    options = gfem.SolverOptions(linear_solver=solver)
    problem = gfem.Driven(frequency=frequency, excite=excite)
    return gfem.solve(_with_eps(line, eps), problem, options=options).s.values


def test_a_driven_forward_logs_one_record_per_frequency(line, records):
    _s(line, 2.0)
    logged = records()
    assert [r.direction for r in logged] == ["forward", "forward"]
    assert [r.frequency for r in logged] == pytest.approx(F)
    for r in logged:
        assert r.levelno == logging.INFO
        assert r.problem == "driven"
        assert r.solver == "direct"
        assert r.iterations == ()
        assert r.action in {"factorized", "refactorized"}


def test_a_driven_gradient_logs_every_forward_and_adjoint_solve(line, records):
    # S21 of a port that is not excited needs an adjoint solve; S11 needs
    # none (port 1's own solution), and logs nothing backward
    def objective(eps):
        return jnp.sum(jnp.abs(_s(line, eps, excite="p1")) ** 2)

    jax.value_and_grad(objective)(jnp.asarray(2.0))
    forward, backward = records("forward"), records("backward")
    assert [r.frequency for r in forward] == pytest.approx(F)
    assert sorted(r.frequency for r in backward) == pytest.approx(F)
    for r in forward + backward:
        assert r.problem == "driven"
        assert r.action is not None
        assert isinstance(r.iterations, tuple)


def test_nothing_is_logged_at_warning(line, records, caplog):
    jax.grad(lambda eps: jnp.sum(jnp.abs(_s(line, eps)) ** 2))(jnp.asarray(2.0))
    assert records()
    assert not [r for r in records() if r.levelno >= logging.WARNING]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_direct_base_factorizes_then_refactorizes_with_its_reason(line, records):
    # one frequency: the base pools between designs, and refactorizes on
    # its plan; the small line is below the reuse floor
    for eps in (2.0, 2.5, 3.0):
        _s(line, eps, frequency=F[:1])
    logged = records()
    assert [(r.action, r.reason) for r in logged] == [
        ("factorized", "first"),
        # the creation's factorization starts no chain (it is not timed)
        ("refactorized", "first"),
        ("refactorized", "below floor"),
    ]


def test_a_backward_on_a_frequency_base_holding_its_design_refactorizes_nothing(
    line, records, caplog
):
    # from the third design each frequency refactorizes its own pooled base
    # (ADR 0121): the backward takes back the base its forward left at F[0],
    # which holds this design's values
    def objective(eps):
        return jnp.sum(jnp.abs(_s(line, eps, excite="p1")) ** 2)

    step = jax.value_and_grad(objective)
    for eps in (2.0, 2.5):
        step(jnp.asarray(eps))
    caplog.clear()
    step(jnp.asarray(3.0))
    (first,) = [r for r in records("backward") if r.frequency == pytest.approx(F[0])]
    assert (first.action, first.reason) == ("held", "same design")
    assert "direct held (same design)" in first.getMessage()


def test_a_bddc_base_is_set_up_once_and_reused_across_frequencies(line, records):
    _s(line, 2.0, solver="bddc_gmres")
    logged = records()
    assert [(r.action, r.reason) for r in logged] == [
        ("factorized", "first"),
        ("reused", None),
    ]
    for r in logged:
        assert r.solver == "bddc_gmres"
        assert len(r.iterations) == 2
        assert all(its > 0 for its in r.iterations)


def test_the_message_names_the_step(line, records):
    _s(line, 2.0, frequency=F[:1])
    (record,) = records()
    message = record.getMessage()
    assert "driven forward" in message
    assert "direct factorized" in message
    assert "21 GHz" in message


@pytest.mark.filterwarnings("ignore::gfem.PortModeProjectionWarning")
def test_adjoint_port_solves_are_logged(tmp_path, records):
    model = rc.lossy_guide_model(tmp_path, 2, "slab", 0.05)
    frequencies = [rc.LOSSY_GUIDE_FREQUENCY, 1.01 * rc.LOSSY_GUIDE_FREQUENCY]
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    rule = DrivenRule(model, gfem.Driven(frequencies), options)
    rule.adjoint_ports = ("p2",)
    rule.forward(model, keep=False)
    adjoint = [r for r in records("forward") if r.columns == "adjoint port"]
    assert [r.frequency for r in adjoint] == pytest.approx(frequencies)
    # on the forward's handle, as the forward left it
    assert [r.action for r in adjoint] == ["reused", "reused"]
    message = adjoint[0].getMessage()
    assert "driven forward solve of the adjoint port columns" in message
    # without the forward, each frequency's handle is the adjoint solve's own
    rule.solve_adjoints(model)
    later = [r for r in records("forward") if r.columns == "adjoint port"][2:]
    assert [r.frequency for r in later] == pytest.approx(frequencies)
    assert all(r.action in {"factorized", "refactorized"} for r in later)


# --- electrostatic -------------------------------------------------------------------


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "plate.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, (("oxide", 0.5),), (2, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"oxide": gfem.Material(eps_r=3.9)}
    )


@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_an_electrostatic_energy_gradient_logs_its_forward_and_backward(
    plate, records, solver
):
    problem = gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom")
    options = gfem.SolverOptions(linear_solver=solver)

    def energy(eps):
        model = plate.with_material("oxide", gfem.MaterialField(eps_r=eps))
        return gfem.solve(model, problem, options=options).energy["oxide"][0]

    n = len(plate.mesh.elements("oxide"))
    jax.grad(energy)(jnp.asarray(np.linspace(3.0, 4.0, n)))
    forward, backward = records("forward"), records("backward")
    assert len(forward) == 1
    assert len(backward) == 1
    for r in forward + backward:
        assert r.problem == "electrostatic"
        assert r.solver == solver
        assert r.frequency is None
    # the backward solves on the forward's kept operator (ADR 0125)
    assert (forward[0].action, forward[0].reason) == ("factorized", "first")
    assert (backward[0].action, backward[0].reason) == ("reused", None)
    # one column per terminal
    assert len(forward[0].iterations) == (0 if solver == "direct" else 1)
