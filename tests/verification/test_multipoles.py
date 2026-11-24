r"""Multipoles against Mie spheres and exact dipoles, at order 3.

- V1-V5: the lossy sphere of `multipole_cases` (:math:`\varepsilon = 4 - i`,
  :math:`x = \pi/2`), one factorization with five columns (an axial wave, the
  two 30 degree waves, a wide beam, the axial wave with a complex amplitude):
  the Mie map and the exact dipoles, the partial cross sections, the sum
  against `gfem.cross_sections`, the incident amplitude, the origin shift.
- V8, V9: a perfectly conducting sphere and an :math:`\varepsilon_r = \mu_r = 2`
  sphere on the same mesh.
- V10, V11: dipoles of complex moment in a box, in vacuum and in a medium
  (:math:`\varepsilon_b = 2`, :math:`\mu_b = 3`), through the private
  projection.
- RM1 (reference tier): the first Kerker point of an index-3.5 sphere.
- V6, V7, RM2 (reference tier): the induced-current route (below).

The gates and the measured floors are in `multipole_cases`.
"""

import math

import free_space_cases as fsc
import multipole_cases as mc
import numpy as np
import pml_cases as pc
import pytest
import scipy.constants as const

import gfem
from gfem import _multipoles as mpm

#: Nothing here is differentiated: keep no factorization between solves, or
#: the module's solves add up past the memory of an eight-core job.
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct", retain_memory=0)
EPS0, MU0, C0 = const.epsilon_0, const.mu_0, const.c
#: The lossy sphere's mesh: the sphere at maxh 0.06 in a layer 0.5 um thick.
#: The partial cross sections need both. At the free-space layer (0.35 um,
#: under two elements of maxh 0.25) the layer's reflection, scattered back by
#: the sphere, leaves about 5e-3 of the norm in degree 1 whatever the sphere's
#: mesh; at sphere maxh 0.1 the faceted sphere leaves about 1.6e-3 in degree
#: 2, eight times that in the degree-2 partial cross section (5.7 % of the
#: total). Either alone puts some incidence direction near 2e-2.
SPHERE_MAXH, LAYER_THICKNESS = 0.06, 0.5

# gates (>= 3x the measured floors recorded in multipole_cases)
E1_GATE, E2_GATE, LEAK_GATE = 1.5e-2, 5e-3, 3.5e-3
CARTESIAN_GATE = 2e-2
SUM_GATE = 4e-3
PARTIAL_GATE = fsc.TOLERANCE
BEAM_L1_GATE, BEAM_SUM_GATE = 1e-4, 5e-3
SHIFT_GATE = 1e-8
PEC_GATES = (2e-2, 1e-2, 7e-3, 4e-3)
MU_GATES = (2e-2, 6e-3, 3e-3, 4e-3)
ROUND_TRIP_GATE, HIGHER_GATE, SHIFT_POWER_GATE, RADIATED_GATE = 2e-3, 5e-3, 1e-6, 1e-3
MEDIUM_GATE = 3e-3
RM1_GATES = {"e1": 2.2e-2, "e2": 2e-3, "leak": 5.5e-3, "sum": 5e-3, "kerker": 2e-2}

AXIAL = gfem.PlaneWave("z", theta=0, phi=0, polarization="TM")
COMPLEX = 0.6 - 0.8j
SOURCES = (
    AXIAL,
    *fsc.SOURCES,
    gfem.GaussianBeam("beam", waist=3e-6, theta=0, phi=0, polarization="TM"),
    gfem.PlaneWave("zc", theta=0, phi=0, polarization="TM", amplitude=COMPLEX),
)
SHIFT = (30e-9, -20e-9, 50e-9)


def _solve(model, frequency, sources):
    problem = gfem.Driven(frequency, sources=sources, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


class _Sphere:
    """A solved sphere with memoized multipoles per column."""

    def __init__(self, model, result, case, frequency):
        self.model, self.result, self.case, self.frequency = (
            model,
            result,
            case,
            frequency,
        )
        self._cache = {}

    def multipoles(self, name, **kwargs):
        key = (name, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            self._cache[key] = gfem.multipoles(
                self.result, name, model=self.model, **kwargs
            )
        return self._cache[key]

    def errors(self, name="z", mu=1.0):
        mp = self.multipoles(name)
        ref = mc.mie_multipoles(self.case.eps, self.case.x, mp.lmax, mu=mu)
        a, b = np.asarray(mp.alpha)[0], np.asarray(mp.beta)[0]
        return mc.degree_errors(a, b, *ref), mc.leak(a, b, *ref)

    def sum_vs_cross_sections(self, name="z"):
        xs = gfem.cross_sections(self.result, name, model=self.model)
        return float(self.multipoles(name).scattering[0] / xs.scattering[0] - 1)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("multipoles") / "sphere.msh"
    return mc.sphere_case(
        path, mc.CASES["L"], sphere_maxh=SPHERE_MAXH, thickness=LAYER_THICKNESS
    )


@pytest.fixture(scope="module")
def lossy(mesh):
    model = fsc.model(mesh[0], fsc.LOSSY)
    return _Sphere(model, _solve(model, mesh[2], SOURCES), mc.CASES["L"], mesh[2])


def test_v1_the_axial_wave_gives_the_mie_coefficients(lossy):
    errors, leak = lossy.errors()
    assert errors[0] < E1_GATE
    assert errors[1] < E2_GATE
    assert leak < LEAK_GATE
    # exact dipoles against the Cartesian closed form
    mp = lossy.multipoles("z")
    k, eta = mp.wavenumber[0], mp.impedance
    a, b = mc.mie_coefficients(lossy.case.eps, lossy.case.x, 1)
    p = 6 * math.pi * EPS0 / (1j * k**3) * a[0]
    m = 6 * math.pi / (1j * k**3 * eta) * b[0]
    got_p, got_m = np.asarray(mp.electric_dipole)[0], np.asarray(mp.magnetic_dipole)[0]
    assert np.linalg.norm(got_p - [p, 0, 0]) < CARTESIAN_GATE * abs(p)
    assert np.linalg.norm(got_m - [0, m, 0]) < CARTESIAN_GATE * abs(m)


@pytest.mark.parametrize("name", ["z", "te", "tm"])
def test_v2_partial_cross_sections_match_mie(lossy, name):
    mp = lossy.multipoles(name)
    k = mp.wavenumber[0]
    a, b = mc.mie_coefficients(lossy.case.eps, lossy.case.x, 2)
    n = np.array([1, 2])
    exact = (
        2 * math.pi / k**2 * (2 * n + 1) * np.abs(a) ** 2,
        2 * math.pi / k**2 * (2 * n + 1) * np.abs(b) ** 2,
    )
    total = float(mp.scattering[0])
    got = (
        np.asarray(mp.electric_scattering)[0, :2],
        np.asarray(mp.magnetic_scattering)[0, :2],
    )
    for g, e in zip(got, exact, strict=True):
        for gi, ei in zip(g, e, strict=True):
            if ei >= 1e-2 * total:
                assert gi == pytest.approx(ei, rel=PARTIAL_GATE)
            else:
                assert abs(gi - ei) <= PARTIAL_GATE * total


@pytest.mark.parametrize("name", ["z", "te", "tm"])
def test_v3_the_sum_matches_the_cross_section(lossy, name):
    assert abs(lossy.sum_vs_cross_sections(name)) < SUM_GATE


def test_v4_complex_amplitude_and_beam(lossy):
    z, zc = lossy.multipoles("z"), lossy.multipoles("zc")
    assert complex(zc.incident_amplitude[0]) == pytest.approx(COMPLEX, abs=1e-15)
    for got, ref in ((zc.alpha, z.alpha), (zc.beta, z.beta)):
        assert (
            np.abs(np.asarray(got) - np.asarray(ref)).max()
            < 1e-12 * np.abs(np.asarray(ref)).max()
        )
    beam = lossy.multipoles("beam")
    e = complex(beam.incident_amplitude[0])
    i0 = SOURCES[3].focal_intensity(lossy.frequency, model=lossy.model)
    assert e.imag == 0
    assert e.real > 0
    assert e.real == pytest.approx(math.sqrt(2 * beam.impedance * i0), rel=1e-12)
    a, b = np.asarray(z.alpha)[0], np.asarray(z.beta)[0]
    errors = mc.degree_errors(np.asarray(beam.alpha)[0], np.asarray(beam.beta)[0], a, b)
    assert errors[0] < BEAM_L1_GATE
    ratio = float(beam.scattering[0] / z.scattering[0] - 1)
    assert abs(ratio) < BEAM_SUM_GATE


def test_v5_the_origin_moves_the_shares_not_the_total(lossy):
    base, moved = lossy.multipoles("z"), lossy.multipoles("z", origin=SHIFT)
    assert abs(float(moved.scattering[0] / base.scattering[0] - 1)) < SHIFT_GATE
    e1 = float(moved.electric_scattering[0, 0] / base.electric_scattering[0, 0] - 1)
    assert abs(e1) > 1e-3


def test_v8_a_conducting_sphere(mesh):
    model = fsc.model(mesh[0], None)
    sphere = _Sphere(model, _solve(model, mesh[2], [AXIAL]), mc.CASES["L"], mesh[2])
    # the reference sphere is the same size: the PEC closed form at its x
    sphere.case = mc.SphereCase("PEC", None, mc.CASES["L"].x)
    errors, leak = sphere.errors()
    assert errors[0] < PEC_GATES[0]
    assert errors[1] < PEC_GATES[1]
    assert leak < PEC_GATES[2]
    assert abs(sphere.sum_vs_cross_sections()) < PEC_GATES[3]


def test_v9_a_magnetic_sphere(mesh):
    material = gfem.Material(eps_r=2.0, mu_r=2.0)
    model = fsc.model(mesh[0], material)
    case = mc.SphereCase("MU", 2.0 + 0j, mc.CASES["L"].x, mu=2.0)
    sphere = _Sphere(model, _solve(model, mesh[2], [AXIAL]), case, mesh[2])
    errors, leak = sphere.errors(mu=2.0)
    assert errors[0] < MU_GATES[0]
    assert errors[1] < MU_GATES[1]
    assert leak < MU_GATES[2]
    assert abs(sphere.sum_vs_cross_sections()) < MU_GATES[3]


# --- dipoles ------------------------------------------------------------------------

R0 = np.array([0.47, 0.53, 0.56]) * fsc.UM
PM = np.array([0.3 + 0.4j, -0.5 + 0.2j, 0.7 - 0.1j]) * 1e-30
MM = np.array([-0.2 + 0.6j, 0.5 + 0.1j, 0.4 - 0.3j]) * 1e-22
CENTRE = np.array([0.5, 0.5, 0.5]) * fsc.UM
#: The dipoles' box: 7 cells per um, a layer of 3 cells. The content of degree
#: 2 and above is the discretization of the air around the dipole (6 cells per
#: um leave 1.9e-3 of the norm, order 4 on them 1.3e-4); the layer's
#: reflections do not reach the projection, whose layer integral keeps only
#: what radiates from inside it (2 to 4 layer cells give the same digits), but
#: they do reach `gfem.far_field`'s radiated power: 4.4e-4 with 2 cells, 1.4e-4
#: with 3.
BOX_CELLS, BOX_LAYER_CELLS = 7, 3


def _dipole_solve(tmp_path_factory, eps_b, mu_b, frequency):
    path = pc.box(
        tmp_path_factory.mktemp("dipoles") / "box.msh",
        sides=pc.SIDES,
        thickness=BOX_LAYER_CELLS / BOX_CELLS,
        cells=BOX_CELLS,
    )
    air = gfem.Material(eps_r=eps_b, mu_r=mu_b)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": air, "pml": air},
        pml=gfem.PML("pml"),
        background=air,
    )
    sources = [
        gfem.ElectricDipole("p", position=R0, moment=tuple(PM)),
        gfem.MagneticDipole("m", position=R0, moment=tuple(MM)),
    ]
    omega = 2 * math.pi * frequency
    eps, mu = EPS0 * eps_b, MU0 * mu_b
    k, eta = omega * math.sqrt(eps * mu), math.sqrt(mu / eps)
    return model, _solve(model, frequency, sources), k, eta


def _projected(  # noqa: PLR0917
    model, result, name, wave, origin, lmax
):
    k, eta, frequency = wave
    alpha, beta = mpm._column_coefficients(
        result, name, model=model, origin=origin, lmax=lmax
    )
    return mpm.Multipoles(
        frequency=(frequency,),
        origin=tuple(float(x) for x in origin),
        lmax=lmax,
        wavenumber=(k,),
        impedance=eta,
        incident_amplitude=np.ones(1, complex),
        alpha=alpha,
        beta=beta,
    )


def _power(mp):
    c = np.concatenate([np.asarray(mp.alpha), np.asarray(mp.beta)], axis=1)
    return float(np.sum(np.abs(c) ** 2) / (2 * mp.impedance * mp.wavenumber[0] ** 2))


@pytest.fixture(scope="module")
def vacuum(tmp_path_factory):
    return (*_dipole_solve(tmp_path_factory, 1.0, 1.0, fsc.F0), fsc.F0)


@pytest.fixture(scope="module")
def medium(tmp_path_factory):
    f = fsc.F0 / 2
    return (*_dipole_solve(tmp_path_factory, 2.0, 3.0, f), f)


def _back(mp, name):
    if name == "p":
        return np.asarray(mp.electric_dipole)[0], PM
    return np.asarray(mp.magnetic_dipole)[0], MM


@pytest.mark.parametrize("name", ["p", "m"])
def test_v10_dipoles_round_trip_in_vacuum(vacuum, name):
    model, result, k, eta, f = vacuum
    mp = _projected(model, result, name, (k, eta, f), R0, 4)
    got, ref = _back(mp, name)
    assert np.linalg.norm(got - ref) < ROUND_TRIP_GATE * np.linalg.norm(ref)
    a, b = np.asarray(mp.alpha)[0], np.asarray(mp.beta)[0]
    norm = math.sqrt(np.sum(np.abs(a) ** 2) + np.sum(np.abs(b) ** 2))
    higher = math.sqrt(np.sum(np.abs(a[3:]) ** 2) + np.sum(np.abs(b[3:]) ** 2))
    assert higher < HIGHER_GATE * norm
    # the total power does not depend on the origin
    d = float(np.linalg.norm(CENTRE - R0))
    lmax = max(4, mpm._wiscombe(k * d))
    shifted = _projected(model, result, name, (k, eta, f), CENTRE, lmax)
    assert _power(shifted) == pytest.approx(_power(mp), rel=SHIFT_POWER_GATE)
    radiated = float(
        gfem.far_field(result, name, 0.3, 0.2, model=model).radiated_power[0]
    )
    assert _power(shifted) == pytest.approx(radiated, rel=RADIATED_GATE)


@pytest.mark.parametrize("name", ["p", "m"])
def test_v11_dipoles_round_trip_in_a_medium(medium, name):
    model, result, k, eta, f = medium
    mp = _projected(model, result, name, (k, eta, f), R0, 4)
    got, ref = _back(mp, name)
    assert np.linalg.norm(got - ref) < MEDIUM_GATE * np.linalg.norm(ref)


@pytest.mark.reference
def test_rm1_the_kerker_sphere(tmp_path):
    case = mc.CASES["C"]
    mesh, _, frequency = mc.sphere_case(tmp_path / "c.msh", case)
    model = fsc.model(mesh, gfem.Material(eps_r=case.eps.real))
    sphere = _Sphere(model, _solve(model, frequency, [AXIAL]), case, frequency)
    errors, leak = sphere.errors()
    assert errors[0] < RM1_GATES["e1"]
    assert errors[1] < RM1_GATES["e2"]
    assert leak < RM1_GATES["leak"]
    assert abs(sphere.sum_vs_cross_sections()) < RM1_GATES["sum"]
    mp = sphere.multipoles("z")
    p = np.asarray(mp.electric_dipole)[0, 0]
    m = np.asarray(mp.magnetic_dipole)[0, 1]
    assert abs(p - m / C0) < RM1_GATES["kerker"] * abs(p)


#: Degree-1 error of the volume route against the projection (both types, over
#: the projection's degree-1 norm), above 3x the measured 2.4e-4 and under the
#: 1e-3 cap.
VOLUME_GATE = 1e-3
#: The two routes differ by the discretization of the far-field layer, so the
#: mesh is finer than the free-space default (maxh 0.4, sphere 0.2): measured
#: 2.6e-3 there, 6.6e-4 at maxh 0.25, 2.4e-4 here. That solve takes minutes, so
#: both tests on it run in the reference tier.
FINE_MAXH, FINE_SPHERE_MAXH = 0.2, 0.1


@pytest.fixture(scope="module")
def fine_lossy_sphere(tmp_path_factory):
    mesh, _ = fsc.sphere(
        tmp_path_factory.mktemp("mp") / "sphere.msh",
        maxh=FINE_MAXH,
        sphere_maxh=FINE_SPHERE_MAXH,
        mirrored=True,
    )
    model = fsc.model(mesh, fsc.LOSSY)
    problem = gfem.Driven(fsc.F0, sources=AXIAL, keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


@pytest.mark.reference
def test_v6_the_volume_route_gives_the_projections_degree_one(fine_lossy_sphere):
    model, result = fine_lossy_sphere
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


@pytest.mark.reference
def test_v7_the_toroidal_correction_improves_the_long_wavelength_dipole(
    fine_lossy_sphere,
):
    model, result = fine_lossy_sphere
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
