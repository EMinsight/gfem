"""Cross sections of spheres against Mie series, at order 3.

Lossy, lossless and perfectly conducting spheres of `free_space_cases`, a
plane wave at 30 degrees in TE and TM as two columns of one solve, compared
with the Bohren-Huffman series at the meshed sphere's volume-equivalent
radius, and on the lossy sphere Gaussian beams of the same directions
against the plane waves; the measured errors and the gates are recorded in
`free_space_cases`. The refined spheres run in the reference tier.
"""

import free_space_cases as fsc
import numpy as np
import pytest

import gfem

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
NAMES = ("extinction", "absorption", "scattering")


def _cross_sections(model, sources=fsc.SOURCES):
    problem = gfem.Driven(fsc.F0, sources=sources, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    return {s.name: gfem.cross_sections(result, s.name, model=model) for s in sources}


def _values(xs):
    return np.array([float(np.asarray(getattr(xs, n))[0]) for n in NAMES])


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    # the half model's mesh and its mirror image: a symmetric field is the
    # half model's
    return fsc.sphere(tmp_path_factory.mktemp("mie") / "sphere.msh", mirrored=True)


#: Gaussian beams along the plane waves, focused on the sphere, wide enough
#: to light it like a plane wave: over the sphere the focal field varies by
#: about :math:`(a/w_0)^2 = 0.7` %.
BEAMS = tuple(
    gfem.GaussianBeam(
        f"beam_{s.name}",
        waist=3e-6,
        theta=s.theta,
        phi=s.phi,
        polarization=s.polarization,
    )
    for s in fsc.SOURCES
)


@pytest.fixture(scope="module")
def lossy(mesh):
    # the beams share the plane waves' factorization
    return _cross_sections(fsc.model(mesh[0], fsc.LOSSY), (*fsc.SOURCES, *BEAMS))


@pytest.mark.parametrize("name", ["te", "tm"])
def test_a_lossy_sphere_matches_mie(mesh, lossy, name):
    reference = fsc.mie_cross_sections(4 - 1j, mesh[1])
    np.testing.assert_allclose(_values(lossy[name]), reference, rtol=fsc.TOLERANCE)


@pytest.mark.parametrize("name", ["te", "tm"])
def test_a_wide_beam_over_its_focal_intensity_is_the_plane_wave(lossy, name):
    np.testing.assert_allclose(
        _values(lossy[f"beam_{name}"]), _values(lossy[name]), rtol=fsc.BEAM_TOLERANCE
    )


def test_a_lossless_sphere_matches_mie_and_absorbs_nothing(mesh):
    xs = _cross_sections(fsc.model(mesh[0], fsc.LOSSLESS))
    reference = fsc.mie_cross_sections(4.0, mesh[1])
    area = np.pi * (fsc.RADIUS * 1e-6) ** 2
    for values in map(_values, xs.values()):
        assert values[0] == pytest.approx(reference[0], rel=fsc.TOLERANCE)
        assert values[2] == pytest.approx(reference[2], rel=fsc.TOLERANCE)
        assert abs(values[1]) < 1e-12 * area


def test_a_conducting_sphere_matches_mie(mesh):
    xs = _cross_sections(fsc.model(mesh[0], None))
    reference = fsc.mie_cross_sections(None, mesh[1])
    area = np.pi * (fsc.RADIUS * 1e-6) ** 2
    for values in map(_values, xs.values()):
        assert values[0] == pytest.approx(reference[0], rel=fsc.PEC_TOLERANCE)
        assert abs(values[1]) < 1e-12 * area


@pytest.mark.parametrize(("name", "wall"), [("te", "PEC"), ("tm", "PMC")])
def test_a_half_model_reports_half_the_cross_sections(tmp_path, lossy, name, wall):
    # the plane y = 0 holds E normal (TE) or H normal (TM): a mirror wall; the
    # whole mesh is this half and its image, so the two agree to round-off
    half, _ = fsc.sphere(tmp_path / "half.msh", half=True)
    source = next(s for s in fsc.SOURCES if s.name == name)
    xs = _cross_sections(fsc.model(half, fsc.LOSSY, mirror=wall), (source,))
    np.testing.assert_allclose(2 * _values(xs[name]), _values(lossy[name]), rtol=1e-10)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("eps", "material"), [(4 - 1j, fsc.LOSSY), (4.0, fsc.LOSSLESS)]
)
def test_a_refined_sphere_matches_mie_within_one_percent(tmp_path, eps, material):
    mesh, radius = fsc.sphere(tmp_path / "fine.msh", sphere_maxh=fsc.REFINED_MAXH)
    xs = _cross_sections(fsc.model(mesh, material), fsc.SOURCES[:1])
    reference = fsc.mie_cross_sections(eps, radius)
    values = _values(xs["te"])
    for k in (0, 2) if eps == 4.0 else (0, 1, 2):
        assert values[k] == pytest.approx(reference[k], rel=fsc.REFINED_TOLERANCE)
