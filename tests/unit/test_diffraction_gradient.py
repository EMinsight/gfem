"""Gradients of the aperture field and its propagation against finite differences.

An analytic cell response (9 samples of the meta-atom parameter on
[0.2, 0.4] um, a smooth S with its exact derivative), a 24 x 24 aperture of
pitch 0.5 um lit at 1 um by right-handed circular light under a complex
Gaussian envelope. The objective is the intensity at the centre plus 0.3 times
an off-axis sample after `propagate` over 8 um. Four fourth-order central
difference checks (`tests/finite_differences.py`): the design map through a
sigmoid onto the sampled range, the envelope, and the response's stored
`values` and `derivative` (complex).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

from gfem import diffraction as dif

jax.config.update("jax_enable_x64", True)

UM = 1e-6
N = 24
PITCH = 0.5 * UM
#: The finite-difference step (norm of the perturbation).
STEP = 1e-3
#: The gate: relative error of each directional derivative.
TOL = 1e-6
CENTRE = (N // 2, N // 2)
OFF_AXIS = (N // 2 - 5, N // 2 + 4)


@pytest.fixture(scope="module")
def problem():
    square = ((PITCH, 0.0, 0.0), (0.0, PITCH, 0.0))
    response = pc.analytic_response(lattice={"z_min": square, "z_max": square})
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.standard_normal((N, N)))
    xs = (np.arange(N) - (N - 1) / 2) * PITCH
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    envelope = jnp.asarray(
        np.exp(-(gx**2 + gy**2) / (2 * (4 * PITCH) ** 2) + 0.7j * gx / PITCH)
    )
    return response, x, envelope


def _objective(response, x, envelope):
    lo, hi = float(response.parameter[0]), float(response.parameter[-1])
    field = dif.aperture_field(
        response,
        lo + (hi - lo) * jax.nn.sigmoid(x),
        "z_max",
        "z_min",
        polarization="RCP",
        amplitude=envelope,
    )
    i = dif.propagate(field, 8 * UM).intensity[0]
    return i[CENTRE] + 0.3 * i[OFF_AXIS]


def _directions(shape, complex_input):
    d = unit_directions(np.random.default_rng(11), shape, 3)
    if complex_input:
        d[2] = 1j * d[2]
    return d


def test_the_design_map(problem):
    response, x, envelope = problem
    assert_gradient_matches(
        lambda q: _objective(response, q, envelope),
        x,
        _directions(x.shape, False),
        step=STEP,
        rel=TOL,
    )


def test_the_envelope(problem):
    response, x, envelope = problem
    assert_gradient_matches(
        lambda a: _objective(response, x, a),
        envelope,
        _directions(envelope.shape, True),
        step=STEP,
        rel=TOL,
    )


@pytest.mark.parametrize("name", ["values", "derivative"])
def test_the_stored_response_arrays(problem, name):
    response, x, envelope = problem
    stored = getattr(response, name)
    assert_gradient_matches(
        lambda v: _objective(dataclasses.replace(response, **{name: v}), x, envelope),
        stored,
        _directions(stored.shape, True),
        step=STEP,
        rel=TOL,
    )
