"""Angle sweeps of a Floquet surface: validation, solve counts, resume, stacking.

Coarse F1 (an eps 4 slab on an eps 2.25 substrate under air, specular ports on
both faces) and coarse F2 (a lamellar grating) at p = 2 with the direct solver.
`gfem.solve` is wrapped to count the solves a sweep makes.
"""

import logging
import math
import warnings

import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

OPTS = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
TOP = "z_max"
F0 = pc.F1_FREQUENCY
DEG = math.pi / 180.0


@pytest.fixture(scope="module")
def f1(tmp_path_factory):
    return pc.f1(pc.f1_mesh(tmp_path_factory.mktemp("sweep_f1")))


@pytest.fixture(scope="module")
def f2(tmp_path_factory):
    mesh = pc.f2_mesh(tmp_path_factory.mktemp("sweep_f2"))
    return pc.f2(mesh, pc.f2_ports(((0, 0),)))


@pytest.fixture
def solves(monkeypatch):
    """The problems `gfem.solve` is called with, in order."""
    calls = []
    real = gfem.solve

    def counting(model, problem, **kwargs):
        calls.append(problem)
        return real(model, problem, **kwargs)

    monkeypatch.setattr(gfem, "solve", counting)
    return calls


def test_s1_a_bad_angle_at_the_last_position_raises_before_any_solve(f1, solves):
    with pytest.raises(ValueError, match=r"theta must be in \[-pi/2, pi/2\]"):
        dif.angle_sweep(f1, [F0], surface=TOP, theta=[0.1, 0.2, 2.0], options=OPTS)
    assert solves == []


def test_s1_sw1_theta_and_phi_must_make_one_flat_list(f1, solves):
    grid = np.zeros((2, 2))
    with pytest.raises(ValueError, match=r"must broadcast to one flat list.*\(2, 2\)"):
        dif.angle_sweep(f1, [F0], surface=TOP, theta=grid, options=OPTS)
    with pytest.raises(ValueError, match=r"got shapes \(2,\) and \(3,\)"):
        dif.angle_sweep(
            f1, [F0], surface=TOP, theta=[0.1, 0.2], phi=[0.0, 0.1, 0.2], options=OPTS
        )
    with pytest.raises(ValueError, match=r"angle_sweep: theta is empty"):
        dif.angle_sweep(f1, [F0], surface=TOP, theta=[], options=OPTS)
    assert solves == []


def test_s1_sw2_excite_none_needs_a_specular_port_on_the_surface(f1, solves):
    model = f1.replace(ports=[p for p in f1.ports if p.surface != TOP])
    with pytest.raises(ValueError) as error:
        dif.angle_sweep(model, [F0], surface=TOP, theta=[0.1], options=OPTS)
    text = str(error.value)
    assert text.startswith("angle_sweep: excite=None excites the specular Floquet")
    assert "'z_max' has none" in text
    assert "gfem.diffraction.floquet_ports(model, 'z_max', frequency" in text
    assert solves == []


def test_s1_sw3_an_excited_port_must_propagate_at_every_incidence(f1, solves):
    with pytest.raises(ValueError) as error:
        dif.angle_sweep(f1, [F0], surface=TOP, theta=[0.1, math.pi / 2], options=OPTS)
    text = str(error.value)
    assert text.startswith(
        "angle_sweep: at incidence 1 (theta = 90°, phi = 0°): FloquetPort 'R_TE'"
    )
    assert "cannot be excited" in text
    assert solves == []


def test_s1_an_unknown_excite_name_raises_before_any_solve(f1, solves):
    with pytest.raises(ValueError, match="names port 'nope'"):
        dif.angle_sweep(f1, [F0], surface=TOP, theta=[0.1], excite="nope", options=OPTS)
    assert solves == []


def test_r2_a_traced_angle_raises_under_the_sweeps_own_name(f1, solves):
    import jax

    def objective(t):
        sweep = dif.angle_sweep(f1, [F0], surface=TOP, theta=t, options=OPTS)
        return sweep[0].s.values.real.sum()

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"angle_sweep: theta and phi must be fixed numbers",
    ):
        jax.grad(objective)(0.3)
    assert solves == []


def test_s2_one_warning_for_the_grid_and_none_per_solve(f2, solves):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dif.angle_sweep(
            f2, [F0], surface=TOP, theta=[0.0, 20 * DEG, 30 * DEG], options=OPTS
        )
    mine = [w for w in caught if issubclass(w.category, gfem.UnrequestedOrderWarning)]
    assert len(mine) == 1
    text = str(mine[0].message)
    assert text.startswith("angle_sweep: propagating order(s) ")
    assert "(-1, 0) TE and TM on 'z_max'" in text
    assert "theta = 20°, phi = 0°" in text
    assert "filter UnrequestedOrderWarning" in text
    assert len(solves) == 3


def test_s3_normal_incidence_and_repeats_are_solved_once(f1, solves):
    # one thread: a threaded factorization is not bitwise repeatable
    opts = gfem.SolverOptions(order=2, threads=1, linear_solver="direct")
    theta = [0.0, 0.0, 20 * DEG, 20 * DEG]
    phi = [0.0, math.pi / 2, 0.0, 0.0]
    sweep = dif.angle_sweep(f1, [F0], surface=TOP, theta=theta, phi=phi, options=opts)
    assert len(solves) == 2
    assert solves[0].wave_vector is None
    assert solves[1].wave_vector is not None
    assert sweep.phi == tuple(phi)
    assert sweep.theta == tuple(theta)
    assert sweep[0] is sweep[1]
    assert sweep[2] is sweep[3]
    plain = gfem.solve(f1, gfem.Driven([F0], excite=["R_TE", "R_TM"]), options=opts)
    assert np.array_equal(np.asarray(sweep[0].s.values), np.asarray(plain.s.values))


def test_s4_an_error_propagates_and_the_rerun_resumes_from_it(
    f1, tmp_path, monkeypatch, caplog
):
    real = gfem.solve
    state = {"calls": 0, "fail_at": 2}

    def flaky(model, problem, **kwargs):
        i = state["calls"]
        state["calls"] += 1
        if i == state["fail_at"]:
            raise RuntimeError("injected")
        return real(model, problem, **kwargs)

    monkeypatch.setattr(gfem, "solve", flaky)
    theta = [0.1, 0.2, 0.3]
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="injected"):
        dif.angle_sweep(f1, [F0], surface=TOP, theta=theta, options=OPTS, cache=cache)
    state.update(calls=0, fail_at=-1)
    with caplog.at_level(logging.INFO, logger="gfem.solve_cache"):
        sweep = dif.angle_sweep(
            f1, [F0], surface=TOP, theta=theta, options=OPTS, cache=cache
        )
    messages = [r.getMessage() for r in caplog.records]
    assert sum(m.startswith("hit ") for m in messages) == 2
    assert sum("solved and stored" in m for m in messages) == 1
    assert len(sweep) == 3


@pytest.fixture(scope="module")
def sweep(f1):
    return dif.angle_sweep(
        f1,
        [F0, 1.1 * F0],
        surface=TOP,
        theta=[0.0, 20 * DEG, 40 * DEG],
        phi=[0.0, 0.0, 30 * DEG],
        options=OPTS,
    )


def test_s5_the_sweep_is_a_sequence_of_results(sweep):
    import collections.abc

    assert isinstance(sweep, collections.abc.Sequence)
    assert len(sweep) == 3
    assert all(isinstance(r, gfem.DrivenResult) for r in sweep)
    assert list(sweep) == list(sweep.results)
    assert sweep.surface == TOP
    assert sweep.frequency == (F0, 1.1 * F0)
    assert isinstance(sweep[1:], dif.AngleSweepResult)
    assert sweep[1:].theta == sweep.theta[1:]
    text = repr(sweep)
    assert "3 incidences" in text
    assert "40°" in text


def test_s5_the_outputs_of_a_sweep_are_the_stacked_member_outputs(sweep, f1):
    members = list(sweep)
    eff = dif.efficiencies(sweep, TOP, polarization="TE", model=f1)
    bal = dif.power_balance(sweep, TOP, polarization="unpolarized", model=f1)
    jon = dif.jones(sweep, TOP, TOP, model=f1)
    for i, member in enumerate(members):
        e = dif.efficiencies(member, TOP, polarization="TE", model=f1)
        b = dif.power_balance(member, TOP, polarization="unpolarized", model=f1)
        j = dif.jones(member, TOP, TOP, model=f1)
        assert np.array_equal(np.asarray(eff.values[i]), np.asarray(e.values))
        assert np.array_equal(np.asarray(bal.absorptance[i]), np.asarray(b.absorptance))
        assert np.array_equal(np.asarray(jon.values[i]), np.asarray(j.values))
    assert eff.values.shape[0] == 3


def test_s6_the_specular_angle_is_the_same_at_every_frequency(sweep):
    for member, theta in zip(sweep, sweep.theta, strict=True):
        order = member.diffraction_orders["R_TE"]
        angle = np.asarray(order.theta)
        assert angle.shape == (2,)
        assert np.abs(angle - angle[0]).max() <= 1e-12
        assert abs(angle[0] - abs(theta)) <= 1e-12
