r"""The second-harmonic overlap of a port mode converges to closed forms.

A homogeneously filled perfectly conducting rectangle $a \times b$ carries TE10
at $\omega$ and at $2\omega$ with $\beta^2 = n^2 k_0^2 - k_c^2$ exactly. A
tensor with $d_{yyy} = d$ alone couples the two through
$\int \sin^3(\pi x / a)\,\mathrm{d}x = 4a / 3\pi$ and the 1 W amplitudes
$E_0^2 = 4 Z_0 / (a b\, n_\mathrm{eff})$, giving

.. math::

    \eta_\mathrm{norm} = |\kappa|^2 =
    \frac{256\,k_0^2 d^2 Z_0}{9\pi^2\,ab\,n_{\mathrm{eff},\omega}^2\,
    n_{\mathrm{eff},2\omega}},

with $n_{\mathrm{eff},\omega} = \sqrt{n^2 - (\lambda / 2a)^2}$ and
$n_{\mathrm{eff},2\omega} = \sqrt{n^2 - (\lambda / 4a)^2}$. The literals below
are this formula and $\Delta\beta = \beta_{2\omega} - 2\beta_\omega$ evaluated
for $a = 2$ um, $b = 1$ um, $n = 2.2$, $d = 20$ pm/V and $\lambda = 1.55$ um
(``chi2_check.py`` of the research, agreeing with a quadrature of the
integral to $5\times10^{-16}$). The same mode crossed by an Ez line port, its
$b$ taken as 1 m and fields at 1 W/m, gives the same figure in 1/(W m) with
only $d_{zzz}$ acting. TE10 drives TE$m$0 at $2\omega$ for odd $m$ only, so
TE20 does not couple. The element order refines the mesh's solution towards
the closed forms, and the gradient of $\eta_\mathrm{norm}$ in a design scale of
the permittivity is held to fourth-order differences of exact solves and to the
derivative of the closed form.
"""

import itertools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as sc
from finite_differences import assert_gradient_matches
from slab2d import write_slab

import gfem

ORDERS = (2, 3, 4)
LINE_ORDERS = (2, 3, 4)
DATA = Path(__file__).parents[1] / "unit" / "data"

LAMBDA = 1.55e-6
N = 2.2
D = 20e-12
#: the closed-form rectangle: a x b = 2 um x 1 um (the WR-90 mesh scaled in x, y, z)
A_REF, B_REF = 2e-6, 1e-6
#: eta_norm (1/(W m^2)) at b = 1 um, and for the line port (1/(W m)) at b = 1 m
ETA_RECT_B1UM = 3.4718769094e5
ETA_LINE = 0.34718769094
#: beta(2 w) - 2 beta(w) (1/m) and its first-order poling period (m)
DELTA_BETA = 2.0955095853e5
PERIOD = 29.9840446990e-6
#: |kappa| (1/(sqrt(W) m)) of the rectangle at b = 1 um
KAPPA_RECT_B1UM = 589.22634950

#: WR-90 mesh section (m), rescaled to the closed-form rectangle's width
WR90_A, WR90_B = 22.86e-3, 10.16e-3
SCALE = A_REF / WR90_A
B_MESH = WR90_B * SCALE
F = sc.c / LAMBDA


def _converges(errors, floor):
    """Each order 5x below the last (measured 12x or more), the last under ``floor``."""
    assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < floor, errors


def _d_tensor(row):
    d = np.zeros((3, 6))
    d[row, row] = D
    return d


def _rectangle(eps_scale=1.0, d=None):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    nodes = jnp.asarray(np.asarray(mesh.imported_nodes) * SCALE)
    mesh = jax.tree.unflatten(jax.tree.structure(mesh), [nodes])
    material = gfem.Material(
        eps_r=N**2 * eps_scale,
        second_order_nonlinear_tensor=_d_tensor(1) if d is None else d,
    )
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def _mode(model, frequency, order, **kwargs):
    return gfem.solve(
        model,
        gfem.PortMode("p1", [frequency], **kwargs),
        options=gfem.SolverOptions(order=order),
    )


def _closed_eta(eps_scale, b):
    """The closed form of eta_norm of the filled TE10 rectangle, differentiable."""
    z0 = math.sqrt(sc.mu_0 / sc.epsilon_0)
    k0 = 2 * math.pi / LAMBDA
    n2 = N**2 * eps_scale
    n_p = jnp.sqrt(n2 - (LAMBDA / (2 * A_REF)) ** 2)
    n_h = jnp.sqrt(n2 - (LAMBDA / (4 * A_REF)) ** 2)
    return 256 * k0**2 * D**2 * z0 / (9 * math.pi**2 * A_REF * b * n_p**2 * n_h)


@pytest.fixture(scope="module")
def rectangle():
    return _rectangle()


@pytest.fixture(scope="module")
def te10_modes(rectangle):
    """The TE10 pump and harmonic of the rectangle at each order, solved once."""
    return {p: (_mode(rectangle, F, p), _mode(rectangle, 2 * F, p)) for p in ORDERS}


def test_the_filled_te10_converges_to_its_closed_form(rectangle, te10_modes):
    # measured relative errors of eta_norm 5.7e-5, 1.7e-7, 1.5e-9 at orders 2-4
    # (the mesh's field error, squared in the overlap), of the period 7e-7,
    # 5e-8, 1e-11. Order 5 sits on the literals' own 11 digits and on the
    # CODATA constants of scipy against the literals' (6e-10 in eta_norm,
    # mu_0 and k_0 enter), so the floors stop above that: 1e-8, 1e-9 resp.
    eta, dbeta, period = [], [], []
    for p in ORDERS:
        pump, harm = te10_modes[p]
        sh = gfem.second_harmonic(rectangle, pump, harm)
        # b of the mesh is 8/9 um, eta_norm is 1/b
        eta.append(abs(float(sh.eta_norm[0]) * B_MESH / B_REF / ETA_RECT_B1UM - 1))
        dbeta.append(abs(float(sh.delta_beta[0]) / DELTA_BETA - 1))
        period.append(abs(float(sh.poling_period[0]) / PERIOD - 1))
        k = complex(sh.kappa[0])
        # TE10 in the port-mode gauge has E_t real: no E_z term acts through
        # d_yyy; round-off of the sum of the products, 1e-12 of |kappa|
        assert abs(k.imag) <= 1e-12 * abs(k)
        # |kappa| is half of eta_norm's error: 2.9e-5 at order 2, 1e-4 gates it
        assert abs(k) * math.sqrt(B_MESH / B_REF) == pytest.approx(
            KAPPA_RECT_B1UM, rel=1e-4
        )
        assert float(sh.delta_beta[0]) > 0
        assert sh.modes == (1, 1)
    _converges(eta, 1e-8)
    _converges(dbeta, 1e-9)
    # the period is 2 pi / delta_beta of the row, so it converges alike; the
    # literals' 11 digits bound it at 1e-9
    assert period[-1] < 1e-9, period


def test_te20_at_the_harmonic_is_not_driven_by_te10(rectangle, te10_modes):
    p = 3
    pump, te10 = te10_modes[p]
    te20 = _mode(rectangle, 2 * F, p, mode=2)
    # mode 2 is the TE20: n_eff = sqrt(n^2 - (lambda / 2a)^2), the pump's
    n_eff = math.sqrt(N**2 - (LAMBDA / (2 * A_REF)) ** 2)
    assert float(te20.n_eff[0].real) == pytest.approx(n_eff, rel=1e-6)
    k = np.asarray(gfem.three_wave_coupling(rectangle, [pump, te10, te20]).kappa)
    # sin(2 pi x / a) sin^2(pi x / a) integrates to zero by the mirror of the
    # guide, and the structured mesh is mirror symmetric: the discrete overlap
    # is round-off (2e-15 of the TE10 coupling measured), 1e-10 gates it
    assert abs(k[2, 0, 0]) < 1e-10 * abs(k[1, 0, 0])
    assert abs(k[1, 0, 0]) > 0


def _line_model(path, eps_scale=1.0, d=None):
    """A 2 um line across a filled PEC-ended slab: closed form's rectangle, b = 1 m."""
    material = gfem.MaterialField(
        eps_r=N**2 * eps_scale,
        second_order_nonlinear_tensor=_d_tensor(2) if d is None else d,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"core": material, "clad": material},
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )


@pytest.fixture(scope="module")
def line_mesh_path(tmp_path_factory):
    # 2 um across (1 um "core" and 0.5 um "clad" each side, one material),
    # 4 cells per um
    return write_slab(
        tmp_path_factory.mktemp("line") / "slab.msh",
        thick=1.0,
        clear=0.5,
        h_core=0.25,
        h_clad=0.25,
    )


def test_the_ez_line_port_converges_to_its_closed_form(line_mesh_path):
    model = _line_model(line_mesh_path)
    eta, dbeta = [], []
    for p in LINE_ORDERS:
        pump = _mode(model, F, p, polarization="Ez")
        harm = _mode(model, 2 * F, p, polarization="Ez")
        sh = gfem.second_harmonic(model, pump, harm)
        eta.append(abs(float(sh.eta_norm[0]) / ETA_LINE - 1))
        dbeta.append(abs(float(sh.delta_beta[0]) / DELTA_BETA - 1))
        k = complex(sh.kappa[0])
        assert abs(k.imag) <= 1e-12 * abs(k)
    # measured relative errors of eta_norm 1.1e-6, 6.3e-9 at orders 2, 3, then
    # the plateau of the literals' digits and constants, 7e-10 (as in the
    # rectangle): the third order is held under 2e-9
    _converges(eta[:2], 1e-8)
    assert eta[2] < 2e-9, eta


def test_the_gradient_of_eta_norm_in_the_permittivity_matches_differences(
    line_mesh_path,
):
    def eta(s, order):
        model = _line_model(line_mesh_path, eps_scale=s)
        pump = _mode(model, F, order, polarization="Ez")
        harm = _mode(model, 2 * F, order, polarization="Ez")
        return gfem.second_harmonic(model, pump, harm).eta_norm[0] / ETA_LINE

    s0 = jnp.asarray(1.0)
    # the adjoint gradient at order 3 against the derivative of the closed form:
    # the mesh's error of the gradient, 8e-9 measured, 1e-6 gates it
    grad = float(jax.grad(lambda s: eta(s, 3))(s0))
    exact = float(jax.grad(lambda s: _closed_eta(s, 1.0) / ETA_LINE)(s0))
    assert grad == pytest.approx(exact, rel=1e-6)
    # and, at order 2, against the fourth-order difference of exact solves of the
    # same discretization (step 1e-2 as for a permittivity: truncation about
    # 1e-8, round-off 1e-13 of the modes' eigensolves, gate 1e-6)
    assert_gradient_matches(
        lambda s: eta(s, 2), s0, [np.array(1.0)], step=1e-2, rel=1e-6
    )
