r"""The locally periodic approximation against a full supercell solve.

A ridge (eps 5, 0.6 um high) on glass (2.1), pitch 0.5 um, lit at 1 um from
the substrate at normal incidence with E along the ridge (TE). The cell
response is tabulated with a mesh per sample at 0.16, 0.18, 0.24, 0.26, 0.32
and 0.34 um, so the Hermite interpolant is exact at every width a supercell
uses and the comparison isolates the neighbour coupling the approximation
neglects. A supercell holds N ridges, each meshed exactly as its single cell,
with a Floquet port per propagating order and polarization on both faces.

The approximation predicts transmitted TE order :math:`m` of an N-cell
supercell of period :math:`\Lambda = Na` as

.. math::

    S_m = \sigma_m \sqrt{\cos\theta_m}\;\frac1N\sum_j E_{2,j}\,
    e^{+2\pi i m x_j/\Lambda},

with :math:`E_{2,j}` the TE component of `aperture_field` at the cell
centres :math:`x_j = (j + \tfrac12)a` in the supercell's coordinates,
:math:`\sqrt{\cos\theta_m}` the power normalization of an oblique TE order and
:math:`\sigma_m = -1` for :math:`m < 0`, where the TE reference
:math:`\hat e_3\times\hat\kappa` of the port is :math:`-\hat e_2`.
"""

import functools

import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

UM = 1e-6
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=2)
SAMPLES = [0.16, 0.18, 0.24, 0.26, 0.32, 0.34]
SLOW8 = [0.34, 0.32, 0.26, 0.18, 0.16, 0.18, 0.26, 0.32]
UNIFORM4 = [0.24] * 4
#: Largest |S_m - LPA| over the transmitted TE orders of the slowly varying
#: supercell (phase steps up to 40 degrees). Measured 0.014.
SLOW_TOL = 0.03
#: The same for a uniform supercell, where the approximation is exact up to
#: solver round-off. Measured 7e-15.
UNIFORM_TOL = 1e-10


@pytest.fixture(scope="module")
def response(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ridge")

    @functools.cache
    def cell(p):
        base = pc.ridge(pc.ridge_mesh(directory, p))
        morph = gfem.shape.boundary_morph(base, "sidewall", grows="atom")
        return base, morph, jnp.asarray(pc.ridge_width_map(morph))

    def make_cell(p, dp):
        base, morph, wm = cell(float(p))
        return base.replace(mesh=morph(dp * UM * wm))

    return dif.cell_response(
        make_cell,
        SAMPLES,
        frequency=[pc.RIDGE_FREQUENCY],
        excite=["sub TE"],
        options=OPTIONS,
    )


def _supercell(directory, widths):
    """The supercell's S row of every transmitted TE order, and its angle."""
    mesh = pc.ridge_supercell_mesh(directory, widths)
    model = pc.ridge(mesh).replace(ports=[])
    ports = dif.floquet_ports(
        model, "z_min", pc.RIDGE_FREQUENCY, offset=pc.RIDGE_SUB * UM
    ) + dif.floquet_ports(model, "z_max", pc.RIDGE_FREQUENCY, offset=pc.RIDGE_AIR * UM)
    model = model.replace(ports=ports)
    excite = "z_min (0, 0) TE"
    result = gfem.solve(
        model, gfem.Driven([pc.RIDGE_FREQUENCY], excite=[excite]), options=OPTIONS
    )
    out = {}
    for port in ports:
        if port.surface == "z_max" and port.polarization == "TE":
            order = result.diffraction_orders[port.name]
            out[port.order[0]] = (
                complex(np.asarray(result.s[port.name, excite])[0]),
                float(np.asarray(order.theta)[0]),
            )
    return out


def _prediction(response, widths, m, theta):
    n = len(widths)
    a = pc.RIDGE_PITCH * UM
    field = dif.aperture_field(response, widths, "z_max", "z_min", polarization="TE")
    e2 = np.asarray(field.values)[0, 1]
    xj = (np.arange(n) + 0.5) * a
    c = np.mean(e2 * np.exp(2j * np.pi * m * xj / (n * a)))
    sigma = -1.0 if m < 0 else 1.0
    return sigma * np.sqrt(np.cos(theta)) * c


@pytest.mark.parametrize(
    ("widths", "tol"),
    [(SLOW8, SLOW_TOL), (UNIFORM4, UNIFORM_TOL)],
    ids=["slow8", "uniform4"],
)
def test_the_supercell_matches_the_locally_periodic_approximation(
    response, tmp_path, widths, tol
):
    rows = _supercell(tmp_path, widths)
    assert 0 in rows
    assert len(rows) >= 2 * len(widths) - 1
    worst = max(
        abs(s - _prediction(response, widths, m, theta))
        for m, (s, theta) in rows.items()
    )
    assert worst <= tol, worst
