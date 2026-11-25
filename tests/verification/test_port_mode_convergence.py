"""Port modes converge to analytic waveguide modes as the element order rises.

One coarse fixed mesh per guide; p-refinement only. Cases:
- empty WR-90: TE10 propagation constant, power-voltage impedance
  ``2 b Z_TE / a`` along the centre line, and the ``sin(pi x / a)`` profile;
- WR-90 with a full-height dielectric slab against one wall: the TE10-like
  mode of the transcendental dispersion relation (a hybrid material case).

The group index and the second and third order dispersion converge with the
order as well, to the closed form (TE10) and to 40-digit literals of the
dispersion relation's derivatives (slab-loaded guide).
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
C0 = 299792458.0
ETA0 = 376.730313412
A, B = 22.86e-3, 10.16e-3
SLAB, EPS_SLAB = 8e-3, 4.0
FREQS = np.array([10e9, 16e9])
ORDERS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def te10():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    path = ((A / 2, 0.0, 0.0), (A / 2, B, 0.0))
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1", voltage_path=path)],
    )
    return {
        p: gfem.solve(
            model, gfem.PortMode("p1", FREQS), options=gfem.SolverOptions(order=p)
        )
        for p in ORDERS
    }


def _te10_beta(f):
    k0 = 2 * np.pi * f / C0
    return np.sqrt(k0**2 - (np.pi / A) ** 2)


def test_te10_propagation_constant_converges_with_order(te10):
    beta = _te10_beta(FREQS)
    errors = [np.abs(np.asarray(te10[p].gamma) / (1j * beta) - 1).max() for p in ORDERS]
    assert all(e2 < e1 / 10 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-10, errors
    for p in ORDERS:
        assert np.abs(np.asarray(te10[p].gamma).real).max() < 1e-12 * beta.min()


def _te10_dispersion(f):
    r"""$n_g$, $\beta_2$, $\beta_3$ of TE10 from $\beta = \sqrt{k_0^2 - k_c^2}$."""
    k0, beta, kc2 = 2 * np.pi * f / C0, _te10_beta(f), (np.pi / A) ** 2
    return {
        "n_g": k0 / beta,
        "beta_2": -kc2 / (C0**2 * beta**3),
        "beta_3": 3 * kc2 * k0 / (C0**3 * beta**5),
    }


def test_te10_dispersion_converges_with_order(te10):
    exact = _te10_dispersion(FREQS)
    for name, gate in (("n_g", 1e-10), ("beta_2", 1e-9), ("beta_3", 1e-9)):
        errors = [
            np.abs(np.asarray(getattr(te10[p], name)) / exact[name] - 1).max()
            for p in ORDERS
        ]
        assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(errors)), errors
        assert errors[-1] < gate, (name, errors)


def test_te10_power_voltage_impedance_converges_with_order(te10):
    k0 = 2 * np.pi * FREQS / C0
    z_te = ETA0 * k0 / _te10_beta(FREQS)
    exact = 2 * B * z_te / A
    errors = [np.abs(np.asarray(te10[p].Z_PV) / exact - 1).max() for p in ORDERS]
    assert all(e2 < e1 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-6, errors


def test_te10_profile_is_a_half_sine_at_one_watt(te10):
    x = np.linspace(0.05, 0.95, 7) * A
    points = np.stack([x, np.full_like(x, B / 3), np.zeros_like(x)], axis=1)
    e = np.asarray(te10[4].e_field.at(points))
    k0 = 2 * np.pi * FREQS / C0
    z_te = ETA0 * k0 / _te10_beta(FREQS)
    e0 = np.sqrt(4 * z_te / (A * B))  # P = a b E0^2 / (4 Z_TE) = 1 W
    expected = e0[:, None] * np.sin(np.pi * x / A)[None, :]
    np.testing.assert_allclose(e[..., 1], expected, rtol=2e-5)
    np.testing.assert_allclose(e[..., [0, 2]], 0.0, atol=2e-5 * e0.max())


def _slab_beta(f, eps=EPS_SLAB):
    k0 = 2 * math.pi * f / C0

    def mismatch(beta):
        k1 = np.sqrt(complex(eps * k0**2 - beta**2))
        k2 = np.sqrt(complex(k0**2 - beta**2))
        # k1 cot(k1 s) + k2 cot(k2 (a - s)) = 0, multiplied through to be pole-free
        # and real for real or imaginary k2.
        s2 = np.sin(k2 * (A - SLAB)) / k2
        return (
            k1 * np.cos(k1 * SLAB) * s2 + np.cos(k2 * (A - SLAB)) * np.sin(k1 * SLAB)
        ).real

    grid = np.linspace(1e-6, math.sqrt(eps) * k0 * (1 - 1e-9), 4000)
    values = [mismatch(b) for b in grid]
    roots = [
        brentq(mismatch, grid[i], grid[i + 1])
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    return max(roots)


# n_g, beta_2 (s^2/m), beta_3 (s^3/m) of the dispersion relation at 8 and 12 GHz
# (40-digit derivatives of `_slab_beta`'s relation, rounded).
SLAB_DISPERSION = {
    "n_g": [2.3131144036306833, 2.2376220080974062],
    "beta_2": [-1.7002258510098628e-20, -1.0306202573954573e-20],
    "beta_3": [3.0937328674974562e-30, 2.8723862689500176e-32],
}


def test_slab_loaded_guide_mode_converges_with_order():
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"slab": gfem.Material(eps_r=EPS_SLAB), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    freqs = np.array([8e9, 12e9])
    beta = np.array([_slab_beta(f) for f in freqs])
    errors = []
    dispersion = {name: [] for name in SLAB_DISPERSION}
    for p in ORDERS:
        result = gfem.solve(
            model, gfem.PortMode("p1", freqs), options=gfem.SolverOptions(order=p)
        )
        errors.append(np.abs(np.asarray(result.gamma) / (1j * beta) - 1).max())
        for name, series in dispersion.items():
            ref = np.array(SLAB_DISPERSION[name])
            # beta_3 passes through a small value inside the band: its error is
            # measured against the band's maximum, never against its own value.
            scale = np.abs(ref).max() if name == "beta_3" else np.abs(ref)
            series.append(
                (np.abs(np.asarray(getattr(result, name)) - ref) / scale).max()
            )
    assert all(e2 < e1 / 10 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-9, errors
    for name, gate in (("n_g", 2e-9), ("beta_2", 2e-7), ("beta_3", 5e-8)):
        series = dispersion[name]
        assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(series)), (name, series)
        assert series[-1] < gate, (name, series)


def test_slab_loaded_guide_root_derivative_converges_with_order():
    """d beta / d eps of the dispersion relation's root, to the discrete derivative.

    The discrete model satisfies the relation only up to its discretization
    error, so the gradient converges to the root's derivative (fourth order,
    like the root itself).
    """
    import jax
    import jax.numpy as jnp

    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    n = len(mesh.elements("slab"))
    freqs = np.array([8e9, 12e9])
    h = 1e-3
    exact = np.array(
        [
            (
                8 * (_slab_beta(f, EPS_SLAB + h) - _slab_beta(f, EPS_SLAB - h))
                - (_slab_beta(f, EPS_SLAB + 2 * h) - _slab_beta(f, EPS_SLAB - 2 * h))
            )
            / (12 * h)
            for f in freqs
        ]
    )
    errors = []
    for p in ORDERS[1:]:
        options = gfem.SolverOptions(order=p, port_mode_tolerance=1e-12)

        def beta(eps, options=options):
            model = gfem.Model(
                mesh,
                materials={
                    "slab": gfem.MaterialField(eps_r=eps * jnp.ones(n)),
                    "air": gfem.Material(),
                },
                boundaries=[gfem.PEC("walls")],
                ports=[gfem.WavePort("p1", "port1")],
            )
            return gfem.solve(model, gfem.PortMode("p1", freqs), options=options)

        got = np.array(
            [
                jax.grad(lambda x, k=k: beta(x).gamma[k].imag)(EPS_SLAB)
                for k in range(len(freqs))
            ]
        )
        errors.append(np.abs(got / exact - 1).max())
    assert all(e2 < e1 / 10 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-8, errors
