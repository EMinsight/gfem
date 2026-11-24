r"""Multipoles of a sphere: the induced-current route against the projection.

The lossy sphere of `free_space_cases` at order 3 under an axial plane wave:
the spherical coefficients read from the induced current over the sphere
(`gfem._multipoles._volume_coefficients`, test-only) against the far-field
route of `gfem.multipoles`, and the long-wavelength moments of
`gfem.current_moments` against the exact dipole, :math:`p \approx p_0 - ikT`.
At the anapole of a high-index sphere (reference tier) the long-wavelength
dipole is the one that does not vanish.

The measured errors and the gates are recorded in `multipole_cases`.
"""

import free_space_cases as fsc
import multipole_cases as mc
import numpy as np
import pytest
import scipy.constants as const

import gfem
from gfem import _multipoles as mpm

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
AXIAL = gfem.PlaneWave("z", theta=0.0, phi=0.0, polarization="TM")

#: Degree-1 error of the volume route against the projection (both types, over
#: the projection's degree-1 norm), above 3x the measured 2.4e-4 and under the
#: 1e-3 cap.
VOLUME_GATE = 1e-3
#: The two routes differ by the discretization of the far-field layer, so the
#: mesh is finer than the free-space default (maxh 0.4, sphere 0.2): measured
#: 2.6e-3 there, 6.6e-4 at maxh 0.25, 2.4e-4 here.
MAXH, SPHERE_MAXH = 0.2, 0.1


@pytest.fixture(scope="module")
def lossy_sphere(tmp_path_factory):
    mesh, _ = fsc.sphere(
        tmp_path_factory.mktemp("mp") / "sphere.msh",
        maxh=MAXH,
        sphere_maxh=SPHERE_MAXH,
        mirrored=True,
    )
    model = fsc.model(mesh, fsc.LOSSY)
    problem = gfem.Driven(fsc.F0, sources=AXIAL, keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


def test_v6_the_volume_route_gives_the_projections_degree_one(lossy_sphere):
    model, result = lossy_sphere
    mp = gfem.multipoles(result, "z", model=model)
    volume = mpm._volume_coefficients(result, "z", model=model, lmax=mp.lmax)
    one = slice(0, 3)
    e_inc = np.asarray(mp.incident_amplitude)[:, None]
    want = [np.asarray(c)[:, one] * e_inc for c in (mp.alpha, mp.beta)]
    got = [c[:, one] for c in volume]
    error = np.sqrt(
        sum(np.linalg.norm(g - w) ** 2 for g, w in zip(got, want, strict=True))
        / sum(np.linalg.norm(w) ** 2 for w in want)
    )
    assert error <= VOLUME_GATE


def test_v7_the_toroidal_correction_improves_the_long_wavelength_dipole(lossy_sphere):
    model, result = lossy_sphere
    mp = gfem.multipoles(result, "z", model=model)
    cm = gfem.current_moments(result, "z", model=model)
    p = np.asarray(mp.electric_dipole)[0]
    p0 = np.asarray(cm.long_wavelength_dipole)[0]
    kt = 1j * cm.wavenumber[0] * np.asarray(cm.toroidal_dipole)[0]
    assert np.linalg.norm(p0 - kt - p) <= np.linalg.norm(p0 - p) / 5


#: Reference tier: the anapole sphere D' (index 4, :math:`x = 1.1654`), where the
#: exact dipole vanishes and the long-wavelength one does not.
#: Measured at maxh 0.25, sphere 0.05: the exact dipole's share 2.3e-3; the
#: moments 1.7e-3 and 8e-4 from the quadrature values; the miss 0.070.
ANAPOLE_GATE = 1e-2
MOMENT_GATE = 1e-2


@pytest.mark.reference
def test_rm2_the_anapole_has_no_exact_dipole_but_long_wavelength_moments(tmp_path):
    case = mc.CASES["D'"]
    mesh, _, frequency = mc.sphere_case(tmp_path / "anapole.msh", case)
    model = fsc.model(mesh, gfem.Material(eps_r=case.eps.real))
    problem = gfem.Driven(frequency, sources=AXIAL, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    mp = gfem.multipoles(result, "z", model=model)
    ref_alpha, ref_beta = mc.mie_multipoles(case.eps, case.x, mp.lmax)
    norm = np.sqrt(np.sum(abs(ref_alpha) ** 2) + np.sum(abs(ref_beta) ** 2))
    alpha = np.asarray(mp.alpha)[0]
    dipole = np.linalg.norm(alpha[:3]) / norm
    assert dipole <= ANAPOLE_GATE
    # the long-wavelength moments, in units of 6 pi eps0 E0 / k^3
    cm = gfem.current_moments(result, "z", model=model)
    k = cm.wavenumber[0]
    unit = 6 * np.pi * const.epsilon_0 / k**3
    p0 = np.asarray(cm.long_wavelength_dipole)[0] / unit
    kt = 1j * k * np.asarray(cm.toroidal_dipole)[0] / unit
    want_p0, want_kt = mc.ANAPOLE_LONG_WAVELENGTH["D'"]
    assert p0[0] == pytest.approx(want_p0, rel=MOMENT_GATE)
    assert -kt[0] == pytest.approx(want_kt, rel=MOMENT_GATE)
    miss = np.linalg.norm(p0 - kt) / np.linalg.norm(p0)
    assert 0.03 <= miss <= 0.12
