r"""Far fields of spheres against Mie series, at order 3.

The spheres of `free_space_cases`, lit by a plane wave at 30 degrees in TE
and TM as two columns of one solve:

- T6: the extinction cross section from the forward far-field amplitude
  (the optical theorem, :math:`\sigma_\mathrm{ext} = -\tfrac{4\pi}{k}
  \operatorname{Im}(\hat e^*\cdot F(\hat k))/E_0` under :math:`e^{+i\omega
  t}`) against `gfem.cross_sections`' power balance, on the lossy sphere:
  2 %.
- T5 (reference tier): the bistatic radar cross section of the refined
  lossless sphere in the plane of incidence against the Bohren-Huffman
  amplitudes :math:`\sigma = 4\pi|S|^2/k^2` (:math:`S_2` for TM, :math:`S_1`
  for TE) at the meshed sphere's volume-equivalent radius, at 8 angles
  where :math:`\sigma \ge 0.1 \max\sigma`: 2 %.
"""

import math

import free_space_cases as fsc
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

import gfem

if not hasattr(fsc, "sphere"):  # the sphere cases land with the cross sections
    pytest.skip("free_space_cases has no sphere cases", allow_module_level=True)

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


def _solve(model):
    problem = gfem.Driven(fsc.F0, sources=fsc.SOURCES, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


def test_t6_the_forward_amplitude_gives_the_extinction(tmp_path):
    mesh, _ = fsc.sphere(tmp_path / "sphere.msh", mirrored=True)
    model = fsc.model(mesh, fsc.LOSSY)
    result = _solve(model)
    for source in fsc.SOURCES:
        ff = gfem.far_field(result, source.name, fsc.THETA, 0.0, model=model)
        # TE is along phi-hat, TM along theta-hat, at the incidence direction
        forward = ff.e_phi[0] if source.polarization == "TE" else ff.e_theta[0]
        extinction = -4 * math.pi / fsc.K0 * float(np.imag(forward))
        xs = gfem.cross_sections(result, source.name, model=model)
        assert extinction == pytest.approx(float(xs.extinction[0]), rel=2e-2)


def _amplitudes(eps: float, x: float, angle: np.ndarray) -> tuple[np.ndarray, ...]:
    r"""Bohren-Huffman :math:`S_1, S_2` of a lossless sphere at scattering angles."""
    n = np.arange(1, math.ceil(x + 4 * x ** (1 / 3) + 2) + 6)
    m = math.sqrt(eps)

    def psi(z):
        return z * spherical_jn(n, z)

    def dpsi(z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    dh = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    xi, dxi = x * h, h + x * dh
    a = (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (
        m * psi(m * x) * dxi - xi * dpsi(m * x)
    )
    b = (psi(m * x) * dpsi(x) - m * psi(x) * dpsi(m * x)) / (
        psi(m * x) * dxi - m * xi * dpsi(m * x)
    )
    mu = np.cos(angle)
    tau = _tau(n, mu)
    pi_n = _pi(n, mu)
    weight = (2 * n + 1) / (n * (n + 1))
    s1 = np.sum(weight * (a * pi_n + b * tau), axis=1)
    s2 = np.sum(weight * (a * tau + b * pi_n), axis=1)
    return s1, s2


def _pi(n, mu):
    """Bohren-Huffman's angular functions pi_n by their upward recurrence."""
    out = np.zeros((len(mu), len(n)))
    prev, cur = np.zeros_like(mu), np.ones_like(mu)
    for k in range(1, len(n) + 1):
        out[:, k - 1] = cur
        prev, cur = cur, ((2 * k + 1) * mu * cur - (k + 1) * prev) / k
    return out


def _tau(n, mu):
    """tau_n = n mu pi_n - (n + 1) pi_{n-1}."""
    p = _pi(n, mu)
    before = np.column_stack([np.zeros(len(mu)), p[:, :-1]])
    return n * mu[:, None] * p - (n + 1) * before


@pytest.mark.reference
def test_t5_the_radar_cross_section_of_a_lossless_sphere_matches_mie(tmp_path):
    mesh, radius = fsc.sphere(tmp_path / "fine.msh", sphere_maxh=fsc.REFINED_MAXH)
    model = fsc.model(mesh, fsc.LOSSLESS)
    result = _solve(model)
    # 8 directions in the plane of incidence (phi = 0 or pi)
    theta = np.radians([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 75.0])
    phi = np.where(np.arange(8) % 2 == 0, 0.0, math.pi)
    rhat = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1
    )
    k_inc = np.array([math.sin(fsc.THETA), 0.0, math.cos(fsc.THETA)])
    angle = np.arccos(np.clip(rhat @ k_inc, -1, 1))
    s1, s2 = _amplitudes(4.0, fsc.K0 * radius * 1e-6, angle)
    for source in fsc.SOURCES:
        # TE is normal to the plane of incidence (S1), TM in it (S2)
        s = s1 if source.polarization == "TE" else s2
        expected = 4 * math.pi * np.abs(s) ** 2 / fsc.K0**2
        ff = gfem.far_field(result, source.name, theta, phi, model=model)
        got = np.asarray(ff.rcs[0])
        large = expected >= 0.1 * expected.max()
        np.testing.assert_allclose(got[large], expected[large], rtol=2e-2)
