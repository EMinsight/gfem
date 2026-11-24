r"""`gfem.RationalModel.step_response`, the closed-form step of a rational model.

The model is a 2-port with one real pole and two conjugate pairs. Its step
is written out here from the textbook inverse Laplace transforms of
:math:`S(s)/s`, term by term: a real pole :math:`-\gamma` with residue
:math:`r` gives

.. math::

    \mathcal{L}^{-1}\left[\frac{r}{s (s + \gamma)}\right]
        = \frac{r}{\gamma} \left(1 - e^{-\gamma t}\right),

and a pair :math:`-\alpha \pm i\beta` with residues :math:`a \pm ib` sums to
:math:`(c_1 s + c_0) / ((s + \alpha)^2 + \beta^2)`, :math:`c_1 = 2a`,
:math:`c_0 = 2(a\alpha - b\beta)`, whose step is

.. math::

    c_1 \frac{e^{-\alpha t} \sin \beta t}{\beta}
        + \frac{c_0}{\alpha^2 + \beta^2} \left[1 - e^{-\alpha t}
        \left(\cos \beta t + \frac{\alpha}{\beta} \sin \beta t\right)\right].
"""

import math

import jax
import numpy as np
import pytest

import gfem

#: Scale of the poles, rad/s (1 GHz). The pairs ring at 1.5 and 4 GHz with a
#: quality factor near 1, so the step has settled to the low-pass view's
#: accuracy well inside the 10 ns record of the resampled test.
W = 2 * math.pi * 1e9
REAL = [(-0.8 * W, np.array([[0.3, 0.1], [0.1, -0.2]]) * W)]
PAIRS = [
    (
        complex(-1.0, 1.5) * W,
        np.array([[0.2 - 0.1j, 0.05 + 0.2j], [0.05 + 0.2j, -0.1 + 0.3j]]) * W,
    ),
    (
        complex(-2.0, 4.0) * W,
        np.array([[-0.1 + 0.2j, 0.3 - 0.1j], [0.3 - 0.1j, 0.2 + 0.05j]]) * W,
    ),
]
D = np.array([[0.1, 0.2], [0.2, -0.05]])


def _model(**kwargs):
    poles = [p for p, _ in REAL]
    residues = [r for _, r in REAL]
    for p, r in PAIRS:
        poles += [p, np.conj(p)]
        residues += [r, np.conj(r)]
    return gfem.RationalModel(
        poles=np.asarray(poles),
        residues=np.asarray(residues),
        d=D,
        z_ref=np.array([50.0, 75.0]),
        ports=("p1", "p2"),
        **kwargs,
    )


def _inverse_laplace(t):
    """The step ``(n_time, 2, 2)`` from the real-valued table entries above."""
    t = np.asarray(t, float)[:, None, None]
    out = np.broadcast_to(D, (t.shape[0], 2, 2)).copy()
    for p, r in REAL:
        gamma = -p
        out += r / gamma * (1 - np.exp(-gamma * t))
    for p, r in PAIRS:
        alpha, beta = -p.real, p.imag
        a, b = r.real, r.imag
        c1, c0 = 2 * a, 2 * (a * alpha - b * beta)
        decay = np.exp(-alpha * t)
        out += c1 * decay * np.sin(beta * t) / beta
        out += (
            c0
            / (alpha**2 + beta**2)
            * (1 - decay * (np.cos(beta * t) + alpha / beta * np.sin(beta * t)))
        )
    return out


def test_step_is_the_inverse_laplace_transform_of_s_over_s():
    t = np.linspace(0, 5e-9, 1001)
    step = _model().step_response(t)
    assert step.kind == "step"
    assert step.mode == "closed_form"
    assert step.ports == step.excited == ("p1", "p2")
    np.testing.assert_array_equal(step.time, t)
    values = np.asarray(step.values)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, _inverse_laplace(t), rtol=0, atol=1e-12)


def test_step_is_zero_before_the_edge_and_jumps_to_d():
    t = np.array([-1e-9, -1e-12, -1e-300, 0.0, 1e-9])
    values = np.asarray(_model().step_response(t).values)
    np.testing.assert_array_equal(values[:3], 0.0)
    np.testing.assert_allclose(values[3], D, rtol=0, atol=1e-15)
    np.testing.assert_allclose(values[4], _inverse_laplace([1e-9])[0], atol=1e-12)


def test_late_step_is_the_dc_value_of_the_network():
    model = _model()
    late = np.asarray(model.step_response([1e-6]).values)[0]
    dc = np.asarray(model.network([0.0]).values)[0]
    np.testing.assert_allclose(late, dc.real, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dc.imag, 0.0, atol=1e-12)


def test_step_carries_the_model_reference():
    t = np.linspace(0, 1e-9, 3)
    np.testing.assert_array_equal(_model().step_response(t).z_ref, [50.0, 75.0])
    step = _model(nominal_reference=True).step_response(t)
    assert np.isnan(np.asarray(step.z_ref)).all()
    with pytest.raises(ValueError, match="NaN"):
        step.impedance("p1")
    z = _model().step_response(t).impedance("p2")
    np.testing.assert_allclose(z[0], 75.0 * (1 + D[1, 1]) / (1 - D[1, 1]))


def test_traced_time_raises():
    model = _model()

    def late(t):
        return model.step_response(t).values.sum()

    with pytest.raises(gfem.NonDifferentiableError, match="concrete times"):
        jax.grad(late)(1e-9)
    with pytest.raises(gfem.NonDifferentiableError, match="concrete times"):
        jax.jit(late)(np.linspace(0, 1e-9, 3))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_time_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        _model().step_response([0.0, bad])


#: Harmonic grid of the resampled step: 50 MHz steps up to 20 GHz (10 ns
#: half record), with the zero-frequency sample.
DF, F_MAX = 50e6, 20e9
#: The low-pass step's 10-90 % rise at the default Kaiser beta = 6.
RISE = 0.99 / F_MAX


def test_resampled_low_pass_step_agrees_with_the_closed_form_after_the_edge():
    # The low-pass view windows S in frequency, which rounds the step off in
    # proportion to its curvature and to the squared ratio of the poles to
    # f_max: measured 2.1e-4 at 10 rise times, shrinking as the poles decay.
    model = _model()
    n = round(F_MAX / DF)
    low_pass = model.network(DF * np.arange(0, n + 1)).step_response()
    assert low_pass.mode == "low_pass"
    late = low_pass.time >= 10 * RISE
    closed = model.step_response(low_pass.time[late])
    np.testing.assert_allclose(
        np.asarray(low_pass.values)[late], closed.values, rtol=0, atol=2e-3
    )
