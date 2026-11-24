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
import numpy as np
import pytest

import gfem
from gfem import _multipoles as mpm

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
AXIAL = gfem.PlaneWave("z", theta=0.0, phi=0.0, polarization="TM")

#: Degree-1 error of the volume route against the projection (over the
#: projection's degree-1 norm), 3x the measured one under the 1e-3 cap.
VOLUME_GATE = 1e-3


@pytest.fixture(scope="module")
def lossy_sphere(tmp_path_factory):
    mesh, _ = fsc.sphere(tmp_path_factory.mktemp("mp") / "sphere.msh", mirrored=True)
    model = fsc.model(mesh, fsc.LOSSY)
    problem = gfem.Driven(fsc.F0, sources=AXIAL, keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


def test_v6_the_volume_route_gives_the_projections_degree_one(lossy_sphere):
    model, result = lossy_sphere
    mp = gfem.multipoles(result, "z", model=model)
    volume = mpm._volume_coefficients(result, "z", model=model, lmax=mp.lmax)
    one = slice(0, 3)
    for got, want in zip(volume, (mp.alpha, mp.beta), strict=True):
        want = np.asarray(want)[:, one] * np.asarray(mp.incident_amplitude)[:, None]
        error = np.linalg.norm(got[:, one] - want) / np.linalg.norm(want)
        print("V6", error)
        assert error <= VOLUME_GATE


def test_v7_the_toroidal_correction_improves_the_long_wavelength_dipole(lossy_sphere):
    model, result = lossy_sphere
    mp = gfem.multipoles(result, "z", model=model)
    cm = gfem.current_moments(result, "z", model=model)
    p = np.asarray(mp.electric_dipole)[0]
    p0 = np.asarray(cm.long_wavelength_dipole)[0]
    kt = 1j * cm.wavenumber[0] * np.asarray(cm.toroidal_dipole)[0]
    print("V7", np.linalg.norm(p0 - p), np.linalg.norm(p0 - kt - p))
    assert np.linalg.norm(p0 - kt - p) <= np.linalg.norm(p0 - p) / 5
