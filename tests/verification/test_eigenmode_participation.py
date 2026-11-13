"""Mode-field gradients of the eigenmode ratios beside the gradient matrix.

The gradient matrix's ``eigenmode_participation`` cells check the ratios
(participation, port Q, region and principal shares) in the permittivity,
the conductivity and the sheet fill. Here, on the same fixture J
(tests/gradient_cases.py, `junction_line`), the loss tangent and the
permeability (whose explicit energy term is zero: the mode energy reads
``Re(eps_r)`` only, ADR 0103), and a port that reads no voltage, whose
``q_port`` is infinite and has no derivative while ``1 / q_port`` is an
exact zero. And the convention of a complex field's cotangent: a real
``g`` of the normalized mode field's coefficients ``e`` pulls back with
``dg = Re(ct^T de)``, unconjugated (quantum-metrics ticket 03).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import (
    JUNCTION_TARGET,
    junction_line,
    junction_mesh,
    junction_values,
    ramp,
)

import gfem
from gfem._engine import terms

OPTIONS = gfem.SolverOptions(order=2, eigen_tolerance=1e-13)
PROBLEM = gfem.Eigenmode(target=JUNCTION_TARGET)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return junction_mesh(tmp_path_factory.mktemp("junction"))


def _shares(result):
    return (
        result.bulk_participation["sub"][0]
        - 0.6 * result.bulk_participation["cover"][0]
        + result.participation["jj"][0]
    )


def test_bulk_participation_in_a_tan_delta_leaf(mesh):
    # the lossy fixture: the cover's loss tangent moves the mode, not the
    # energy's coefficient
    values = junction_values(mesh, "lossy")
    loss = ramp(mesh, "cover", 0.01, 0.05)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["cover"]["tan_delta"] = p
        model = junction_line(mesh, "lossy", moved)
        return _shares(gfem.solve(model, PROBLEM, options=OPTIONS))

    directions = unit_directions(np.random.default_rng(0), loss.shape, 2)
    assert_gradient_matches(objective, loss, directions, step=1e-3)


def test_bulk_participation_in_a_mu_r_leaf(mesh):
    values = junction_values(mesh)
    mu = ramp(mesh, "sub", 1.0, 2.0)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"]["mu_r"] = p
        model = junction_line(mesh, "inductive", moved)
        return _shares(gfem.solve(model, PROBLEM, options=OPTIONS))

    directions = unit_directions(np.random.default_rng(1), mu.shape, 2)
    assert_gradient_matches(objective, mu, directions, step=1e-2)


def test_one_over_q_port_without_port_voltage_is_exactly_zero(mesh, monkeypatch):
    # No physical port reads exactly zero voltage on a mode (its field there
    # is round-off, not zero), so port r's voltage functional is replaced by
    # zero: the port still loads the mode, but V = 0 exactly.
    original = terms.port_vectors

    def deaf(space, built, order, data):
        b, v = original(space, built, order, data)
        return (b, np.zeros_like(v)) if data.port.name == "r" else (b, v)

    monkeypatch.setattr(terms, "port_vectors", deaf)
    values = junction_values(mesh)
    eps = values["sub"]["eps_r"]

    def solve(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"]["eps_r"] = p
        return gfem.solve(
            junction_line(mesh, "resistive", moved), PROBLEM, options=OPTIONS
        )

    result = solve(eps)
    assert np.isinf(result.q_port["r"][0])  # V = 0 exactly
    assert np.imag(result.omega[0]) > 0  # loaded all the same
    grad = jax.grad(lambda p: 1.0 / solve(p).q_port["r"][0])(eps)
    assert not np.asarray(grad).any()

    # beside another ratio of the mode: no NaN from the infinite value
    def beside(p):
        result = solve(p)
        return 1.0 / result.q_port["r"][0] + result.participation["jj"][0]

    both = jax.grad(beside)(eps)
    assert np.isfinite(np.asarray(both)).all()
    assert np.abs(np.asarray(both)).max() > 0
    with pytest.raises(jax.errors.JaxRuntimeError, match="1 / q_port"):
        jax.grad(lambda p: solve(p).q_port["r"][0])(jnp.asarray(eps))


def test_field_cotangent_convention_is_unconjugated(mesh):
    # a lossy mode (complex k, complex field): jax.grad of g = Re(c^T e)
    # along d equals Re(c^T de) by differences of e itself, and not the
    # conjugated pairing Re(c^H de)
    values = junction_values(mesh)
    eps = values["sub"]["eps_r"]

    def field(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"]["eps_r"] = p
        model = junction_line(mesh, "resistive", moved)
        return gfem.solve(model, PROBLEM, options=OPTIONS).e_fields[0].coefficients

    e0 = np.asarray(field(eps))
    rng = np.random.default_rng(7)
    c = (rng.standard_normal(e0.shape) + 1j * rng.standard_normal(e0.shape)) / (
        np.abs(e0).max() * np.sqrt(e0.size)
    )
    (d,) = unit_directions(rng, eps.shape, 1)
    grad = jax.grad(lambda p: jnp.real(jnp.sum(c * field(p))))(eps)
    got = float(jnp.dot(grad, d))
    h = 1e-2
    samples = [np.asarray(field(eps + s * h * d)) for s in (-2, -1, 1, 2)]
    de = (8 * (samples[2] - samples[1]) - (samples[3] - samples[0])) / (12 * h)
    unconjugated = float(np.real(np.sum(c * de)))
    conjugated = float(np.real(np.sum(np.conj(c) * de)))
    assert np.iscomplexobj(e0)
    assert abs(np.imag(e0)).max() > 1e-3 * abs(e0).max()  # a complex field
    np.testing.assert_allclose(got, unconjugated, rtol=1e-6)
    assert abs(got - conjugated) > 1e-2 * abs(unconjugated)
