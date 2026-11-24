r"""Rational fits of S from wave ports: the reference impedance and passivity.

A wave port with ``voltage_path`` refers S to its mode's :math:`Z_{PV}`,
complex and frequency-dependent on a lossy line, so the fit needs a real
``z0``. Reference planes moved inward with `gfem.DrivenResult.with_offsets`
until they meet leave a lossless through, a constant S. Moved past each
other by the whole line, they give :math:`S_{21} = e^{+\gamma L}`: more gain
than loss, :math:`|S_{21}| = e^{\alpha L} > 1`, but also a phase advance
:math:`e^{+i\beta L}` that no causal model follows (the fit's constant term
grows to hundreds and enforcement cannot recover). The active-data case
therefore keeps the crossed planes' gain and drops their advance:
:math:`S_{21} = e^{\alpha L - i\beta L}`, active and causal, which the fit
must warn about and still make passive.

The parallel-plate fixture of tests/unit/test_driven_wave_ports.py, with
`gfem.Conductor` plates at :math:`10^5` S/m: :math:`\alpha L` is 0.06 to
0.11 at 1-3 THz, far above the fit's tolerance.
"""

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

import gfem

DATA = Path(__file__).parent / "data"
W, D, L = 40e-6, 10e-6, 20e-6  # plate width, gap and length, m
SIGMA = 1e5  # S/m
GRID = tuple(np.linspace(1.0e12, 3.0e12, 9))


@pytest.fixture(scope="module")
def lossy_line():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.Conductor("plates", conductivity=SIGMA)],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=((W / 2, 0, 0), (W / 2, D, 0))),
            gfem.WavePort("p2", "port2", voltage_path=((W / 2, 0, L), (W / 2, D, L))),
        ],
    )
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    with warnings.catch_warnings():
        # a higher port mode crosses the TEM one's rank near 3 THz
        warnings.simplefilter("ignore", gfem.PortModeTrackingWarning)
        return gfem.solve(model, gfem.Driven(GRID), options=options)


def _gamma(result):
    return np.asarray(result.port_modes["p1"].gamma)


def test_the_lossy_lines_reference_is_complex_and_frequency_dependent(lossy_line):
    z = np.asarray(lossy_line.s.z_ref)
    assert np.abs(z.imag).min() > 1.0  # ohm
    assert np.ptp(z.real) > 1.0
    assert (_gamma(lossy_line).real * L > 0.05).all()


def test_a_complex_z_pv_needs_z0(lossy_line):
    with pytest.raises(ValueError, match="z0"):
        lossy_line.s.fit_rational()


def test_a_fit_on_a_real_z0_is_passive_without_warning(lossy_line):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = lossy_line.s.fit_rational(z0=47.0)
    assert model.passive
    assert model.error <= 1e-3
    np.testing.assert_array_equal(model.z_ref, [47.0, 47.0])


def test_planes_that_meet_leave_a_through_fitted_without_poles(lossy_line):
    met = lossy_line.with_offsets({"p1": L / 2, "p2": L / 2})
    s = np.asarray(met.s.values)
    np.testing.assert_allclose(s[:, 1, 0], 1.0, atol=1e-5)
    assert np.abs(s[:, 0, 0]).max() < 1e-4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = met.s.fit_rational(z0=47.0)
    assert model.n_poles == 0
    assert model.passive
    assert model.error <= 1e-3


def test_active_data_warn_and_give_a_passive_model(lossy_line):
    gamma = _gamma(lossy_line)
    crossed = lossy_line.with_offsets({"p1": L, "p2": L})
    s21 = np.asarray(crossed.s.values)[:, 1, 0]
    # the line's S21 times exp(2 gamma L): exp(+gamma L), up to the mesh
    np.testing.assert_allclose(s21, np.exp(gamma * L), atol=1e-4)
    # keep the crossed planes' gain, drop their phase advance
    advance = np.exp(2j * gamma.imag * L)[:, None, None]
    active = dataclasses.replace(crossed.s, values=crossed.s.values / advance)
    s21 = np.asarray(active.values)[:, 1, 0]
    assert (np.abs(s21) > 1.05).all()
    with pytest.warns(gfem.NonPassiveDataWarning):
        model = active.fit_rational(z0=47.0)
    assert model.passive
    assert model.fit_error <= 1e-3  # the fit itself follows the active data
    assert model.error > 0.05  # enforcement had to remove the gain
    sigma = np.linalg.svd(np.asarray(model.network(GRID).values), compute_uv=False)
    assert sigma.max() <= 1.0
