r"""The far field against closed forms, at order 3.

- T11: a Hertzian dipole in vacuum (:math:`h = \lambda/4`, a
  :math:`\lambda/2` layer): its pattern :math:`F = k^2 p_\perp
  e^{ik\hat r\cdot r_0}/(4\pi\varepsilon_0)`, the intensity integrated over
  the sphere against the radiated power, the peak directivity 1.5; and a
  dipole of moment :math:`p(\hat x - i\hat y)` (two crossed dipoles in
  quadrature), right-hand circular along :math:`+z`: axial ratio 1.
- T12: a dipole above a PEC and a PMC ground plane, normal and tangential,
  against image theory: the far field of the dipole and its image.

Pattern errors are the largest :math:`|F_h - F|` over the directions,
relative to the largest :math:`|F|`. Measured (order 3, one thread): T11
pattern 1.6e-3, power 6e-5, directivity +0.02 %, axial ratio 1.002; T12
(height 0.21 um) 1.5e-3 to 2.6e-3. The layer's own reflection floor is
6e-4 to 1e-3 at this thickness.
"""

import math

import free_space_cases as fsc
import numpy as np
import pml_cases as pc
import pytest

import gfem

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
P = 1e-30
EPS0 = 1 / (fsc.MU0 * fsc.C0**2)
R0 = np.array([0.47, 0.53, 0.56]) * fsc.UM


def _box(path, sides=pc.SIDES, surface=None):
    return pc.box(path, sides=sides, thickness=0.5, cells=4, surface=surface)


def _model(mesh, **kwargs):
    air = gfem.Material()
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air},
        pml=gfem.PML("pml"),
        background=air,
        **kwargs,
    )


def _sphere(n=8):
    """Gauss-Legendre in cos(theta) times uniform phi: exact for the patterns."""
    x, w = np.polynomial.legendre.leggauss(n)
    phi = np.arange(2 * n) * math.pi / n
    weights = w[:, None] * np.full(2 * n, math.pi / n)[None, :]
    return np.arccos(x)[:, None], phi[None, :], weights


def _basis(theta, phi):
    theta, phi = np.broadcast_arrays(theta, phi)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], -1)
    that = np.stack([ct * cp, ct * sp, -st], -1)
    phat = np.stack([-sp, cp, 0 * phi], -1)
    return rhat, that, phat


def _dipoles_far_field(rhat, moments_and_positions):
    """The closed-form amplitude of point dipoles in vacuum, ``(..., 3)``."""
    out = 0
    for p, r in moments_and_positions:
        p = np.asarray(p, complex)
        perp = p - rhat * (rhat @ p)[..., None]
        out = out + perp * np.exp(1j * fsc.K0 * rhat @ r)[..., None]
    return fsc.K0**2 / (4 * math.pi * EPS0) * out


def _pattern_error(ff, theta, phi, exact):
    _, that, phat = _basis(theta, phi)
    error = np.hypot(
        np.abs(np.asarray(ff.e_theta[0]) - np.sum(exact * that, -1)),
        np.abs(np.asarray(ff.e_phi[0]) - np.sum(exact * phat, -1)),
    )
    return error.max() / np.linalg.norm(exact, axis=-1).max()


@pytest.fixture(scope="module")
def vacuum(tmp_path_factory):
    mesh = gfem.Mesh.from_gmsh(_box(tmp_path_factory.mktemp("t11") / "box.msh"))
    model = _model(mesh)
    dipoles = [
        gfem.ElectricDipole("z", position=R0, moment=(0, 0, P)),
        gfem.ElectricDipole("c", position=R0, moment=(P, -1j * P, 0)),
    ]
    problem = gfem.Driven(fsc.F0, sources=dipoles, keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


def test_t11_a_hertzian_dipole_radiates_its_closed_form_pattern(vacuum):
    model, result = vacuum
    theta, phi, weights = _sphere()
    ff = gfem.far_field(result, "z", theta, phi, model=model)
    exact = _dipoles_far_field(_basis(theta, phi)[0], [((0, 0, P), R0)])
    assert _pattern_error(ff, theta, phi, exact) < 5e-3
    total = float(np.sum(np.asarray(ff.intensity[0]) * weights))
    assert total == pytest.approx(float(ff.radiated_power[0]), rel=5e-3)
    # the peak at theta = pi/2
    peak = gfem.far_field(result, "z", math.pi / 2, 0.3, model=model)
    assert float(peak.directivity[0]) == pytest.approx(1.5, rel=1e-2)


def test_t11_a_quadrature_pair_radiates_circular_polarization(vacuum):
    model, result = vacuum
    ff = gfem.far_field(result, "c", 0.0, 0.0, model=model)
    assert float(ff.axial_ratio[0]) == pytest.approx(1.0, abs=1e-2)
    assert abs(complex(ff.e_left[0])) < 5e-3 * abs(complex(ff.e_right[0]))


@pytest.fixture(scope="module")
def grounded(tmp_path_factory):
    path = _box(
        tmp_path_factory.mktemp("t12") / "ground.msh",
        sides=("-x", "+x", "-y", "+y", "+z"),
        surface=lambda c: "ground" if abs(c[2]) < 1e-9 else "",
    )
    return gfem.Mesh.from_gmsh(path)


@pytest.mark.parametrize(("wall", "sign"), [(gfem.PEC, 1), (gfem.PMC, -1)])
def test_t12_a_ground_plane_adds_the_image_of_the_dipole(grounded, wall, sign):
    model = _model(grounded, boundaries=[wall("ground")])
    r0 = np.array([0.47, 0.53, 0.21]) * fsc.UM
    image_r = r0 * np.array([1, 1, -1])
    dipoles = [
        gfem.ElectricDipole("v", position=r0, moment=(0, 0, P)),
        gfem.ElectricDipole("h", position=r0, moment=(P, 0, 0)),
    ]
    problem = gfem.Driven(fsc.F0, sources=dipoles, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    theta, phi, _ = _sphere()
    theta = theta[theta[:, 0] < math.pi / 2]  # the half-space above the plane
    rhat = _basis(theta, phi)[0]
    for d in dipoles:
        p = np.real(np.asarray(d.moment))
        # over a perfect conductor the image moment is (-px, -py, pz)
        image_p = sign * np.array([-p[0], -p[1], p[2]])
        exact = _dipoles_far_field(rhat, [(p, r0), (image_p, image_r)])
        ff = gfem.far_field(result, d.name, theta, phi, model=model)
        assert _pattern_error(ff, theta, phi, exact) < 1e-2, d.name
