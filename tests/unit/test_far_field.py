r"""The far field: its amplitude, IEEE 145 outputs, ground planes and refusals.

Coarse boxes (order 2, :math:`h = \lambda/4`) in a perfectly matched
layer: a Hertzian dipole's pattern is within a few percent of its closed
form there; the closed-form gates live in
tests/verification/test_far_field.py.
"""

import math

import free_space_cases as fsc
import numpy as np
import pml_cases
import pytest

import gfem

ORDER2 = gfem.SolverOptions(order=2, linear_solver="direct")
P = 1e-30
#: A generic point inside the cube, off every face of the grid, m.
INSIDE = np.array([0.47, 0.53, 0.56]) * fsc.UM


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return gfem.Mesh.from_gmsh(
        fsc.dipole_box(tmp_path_factory.mktemp("far") / "box.msh")
    )


def _vacuum(mesh, **kwargs):
    kwargs.setdefault("background", gfem.Material())
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "pml": gfem.Material()},
        pml=gfem.PML("pml"),
        **kwargs,
    )


@pytest.fixture(scope="module")
def dipole(box):
    model = _vacuum(box)
    source = gfem.ElectricDipole("d", position=INSIDE, moment=(0, 0, P))
    problem = gfem.Driven(fsc.F0, sources=source, keep_fields=True)
    return model, gfem.solve(model, problem, options=ORDER2)


def test_a_hertzian_dipole_radiates_its_closed_form_pattern(dipole):
    model, result = dipole
    theta = np.linspace(0.1, math.pi - 0.1, 7)
    phi = np.linspace(0.0, 2 * math.pi, 5)[:, None]
    ff = gfem.far_field(result, "d", theta, phi, model=model)
    # F = k^2 / (4 pi eps0) p sin(theta) (-theta_hat) e^{i k r.r0}
    rhat = np.stack(
        np.broadcast_arrays(
            np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        ),
        axis=-1,
    )
    phase = np.exp(1j * fsc.K0 * rhat @ INSIDE)
    eps0 = 1 / (fsc.MU0 * fsc.C0**2)
    exact = -(fsc.K0**2) / (4 * math.pi * eps0) * P * np.sin(theta) * phase
    assert ff.e_theta.shape == (1, 5, 7)
    peak = np.abs(exact).max()
    # order 2 at lambda/4: 5 % of the peak measured
    np.testing.assert_allclose(ff.e_theta[0], exact, atol=8e-2 * peak)
    np.testing.assert_allclose(ff.e_phi[0], 0.0, atol=8e-2 * peak)


def _sphere(n=16):
    """Gauss-Legendre in cos(theta) times a uniform phi grid: weights sum to 4 pi."""
    x, w = np.polynomial.legendre.leggauss(n)
    phi = np.arange(2 * n) * math.pi / n
    theta = np.arccos(x)[:, None]
    weights = w[:, None] * np.full(2 * n, math.pi / n)[None, :]
    return theta, phi[None, :], weights


def test_the_intensity_integrates_to_the_radiated_power(dipole):
    model, result = dipole
    theta, phi, weights = _sphere()
    ff = gfem.far_field(result, "d", theta, phi, model=model)
    total = float(np.sum(np.asarray(ff.intensity[0]) * weights))
    radiated = float(ff.radiated_power[0])
    assert total == pytest.approx(radiated, rel=3e-2)
    # the residual flux through a lossless layer is the emitted power
    assert radiated == pytest.approx(float(result.emitted_power["d"][0]), rel=1e-8)
    directivity = np.asarray(ff.directivity[0])
    assert directivity.max() == pytest.approx(1.5, rel=3e-2)
    assert float(np.sum(directivity * weights)) == pytest.approx(4 * math.pi, rel=3e-2)
    # a dipole has no mismatch: gain = realized gain, efficiency 1 (lossless)
    np.testing.assert_allclose(ff.gain, ff.realized_gain)
    assert float(ff.radiation_efficiency[0]) == pytest.approx(1.0, rel=1e-8)


def _image_pattern(rhat, r0, p, sign):
    """The closed-form amplitude of a dipole and its image in the plane z = 0.

    ``sign`` = +1 over a perfect conductor (image moment (-px, -py, pz)), -1
    over a perfect magnetic wall.
    """
    eps0 = 1 / (fsc.MU0 * fsc.C0**2)
    image_p = sign * np.array([-p[0], -p[1], p[2]])
    image_r = np.array([r0[0], r0[1], -r0[2]])
    out = 0
    for q, r in ((np.asarray(p), r0), (image_p, image_r)):
        perp = q - rhat * (rhat @ q)[..., None]
        out = out + perp * np.exp(1j * fsc.K0 * rhat @ r)[..., None]
    return fsc.K0**2 / (4 * math.pi * eps0) * out


@pytest.fixture(scope="module")
def grounded(tmp_path_factory):
    path = pml_cases.box(
        tmp_path_factory.mktemp("ground") / "ground.msh",
        sides=("-x", "+x", "-y", "+y", "+z"),
        thickness=0.5,
        cells=4,
        surface=lambda c: "ground" if abs(c[2]) < 1e-9 else "",
    )
    return gfem.Mesh.from_gmsh(path)


@pytest.mark.parametrize(("wall", "sign"), [(gfem.PEC, 1), (gfem.PMC, -1)])
def test_a_ground_plane_adds_the_image_of_the_dipole(grounded, wall, sign):
    model = _vacuum(grounded, boundaries=[wall("ground")])
    r0 = np.array([0.47, 0.53, 0.21]) * fsc.UM
    dipoles = [
        gfem.ElectricDipole("v", position=r0, moment=(0, 0, P)),
        gfem.ElectricDipole("h", position=r0, moment=(P, 0, 0)),
    ]
    problem = gfem.Driven(fsc.F0, sources=dipoles, keep_fields=True)
    result = gfem.solve(model, problem, options=ORDER2)
    theta = np.linspace(0.05, math.pi / 2, 6)
    phi = np.linspace(0.0, 2 * math.pi, 5)[:, None]
    for d in dipoles:
        ff = gfem.far_field(result, d.name, theta, phi, model=model)
        _, that, phat = _basis(theta, phi)
        exact = _image_pattern(_basis(theta, phi)[0], r0, np.asarray(d.moment), sign)
        peak = np.linalg.norm(exact, axis=-1).max()
        error = np.hypot(
            np.abs(ff.e_theta[0] - np.sum(exact * that, -1)),
            np.abs(ff.e_phi[0] - np.sum(exact * phat, -1)),
        ).max()
        # order 2 at lambda/4: 4.5-9 % of the peak measured
        assert error < 0.15 * peak


def _basis(theta, phi):
    theta, phi = np.broadcast_arrays(theta, phi)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], -1)
    that = np.stack([ct * cp, ct * sp, -st], -1)
    phat = np.stack([-sp, cp, 0 * phi], -1)
    return rhat, that, phat


def test_a_direction_below_the_ground_plane_is_refused(grounded):
    model = _vacuum(grounded, boundaries=[gfem.PEC("ground")])
    r0 = np.array([0.47, 0.53, 0.21]) * fsc.UM
    dipole = gfem.ElectricDipole("v", position=r0, moment=(0, 0, P))
    problem = gfem.Driven(fsc.F0, sources=dipole, keep_fields=True)
    result = gfem.solve(model, problem, options=ORDER2)
    with pytest.raises(ValueError, match="PEC ground plane.*below it"):
        gfem.far_field(result, "v", 2.0, 0.0, model=model)


# --- Q5: a lumped-port-fed strip dipole --------------------------------------------


def _strip_surface(c):
    """A strip in the plane x = 0.5 um, y in [0.5, 2/3]: its arms and its feed."""
    eps = 1e-9
    if abs(c[0] - 0.5) > eps or not 0.5 < c[1] < 4 / 6:
        return ""
    if 3 / 6 < c[2] < 4 / 6:
        return "feed"
    return "wire" if 1 / 6 < c[2] < 5 / 6 else ""


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    path = pml_cases.box(
        tmp_path_factory.mktemp("strip") / "strip.msh",
        sides=pml_cases.SIDES,
        thickness=2 / 6,
        cells=6,
        # two vacuum regions, so that the strip lies on their interface
        volume=lambda c: (
            ("air" if c[0] < 0.5 else "right") if ((c > 0) & (c < 1)).all() else "pml"
        ),
        surface=_strip_surface,
    )
    vacuum = gfem.Material()
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": vacuum, "right": vacuum, "pml": vacuum},
        pml=gfem.PML("pml"),
        background=vacuum,
        boundaries=[gfem.PEC("wire")],
        ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
    )
    problem = gfem.Driven(fsc.F0, keep_fields=True)
    return model, gfem.solve(model, problem, options=ORDER2)


def test_a_port_fed_antenna_meets_the_ieee_145_identities(strip):
    model, result = strip
    theta, phi, weights = _sphere()
    ff = gfem.far_field(result, "p1", theta, phi, model=model)
    radiated = float(ff.radiated_power[0])
    total = float(np.sum(np.asarray(ff.intensity[0]) * weights))
    assert total == pytest.approx(radiated, rel=1e-2)
    s11 = complex(result.s["p1", "p1"][0])
    accepted = 1 - abs(s11) ** 2
    assert float(ff.accepted_power[0]) == pytest.approx(accepted, rel=1e-12)
    # lossless: every accepted watt radiates. A lumped port's S comes from
    # its voltage and current, which the discrete power balance meets only to
    # the discretization: 3.3 % at order 2 here, 3.1 % at order 3
    assert float(ff.radiation_efficiency[0]) == pytest.approx(1.0, abs=5e-2)
    # the radiated power is the residual flux through every enclosing side
    pml = -float(gfem.power_flow(result.e_field(fsc.F0), "pml", model=model))
    assert radiated == pytest.approx(pml, rel=1e-10)
    # the port is fed 1 W: the realized gain is 4 pi U
    np.testing.assert_allclose(
        ff.realized_gain, 4 * math.pi * np.asarray(ff.intensity), rtol=1e-14
    )
    np.testing.assert_allclose(
        ff.gain, 4 * math.pi * np.asarray(ff.intensity) / accepted, rtol=1e-12
    )


def test_a_circular_dipole_radiates_circular_polarization_along_its_axis(box):
    model = _vacuum(box)
    # p = P (x - i y): the quadrature pair, right-hand circular along +z
    source = gfem.ElectricDipole("c", position=INSIDE, moment=(P, -1j * P, 0))
    problem = gfem.Driven(fsc.F0, sources=source, keep_fields=True)
    result = gfem.solve(model, problem, options=ORDER2)
    ff = gfem.far_field(result, "c", 0.0, 0.0, model=model)
    # order 2 at lambda/4: 1.11 measured; the 1 % gate is at order 3
    # (tests/verification/test_far_field.py)
    assert float(ff.axial_ratio[0]) == pytest.approx(1.0, abs=0.2)
    assert abs(complex(ff.e_left[0])) < 0.1 * abs(complex(ff.e_right[0]))
    # the other hand: p = P (x + i y)
    flipped = gfem.ElectricDipole("l", position=INSIDE, moment=(P, 1j * P, 0))
    problem = gfem.Driven(fsc.F0, sources=flipped, keep_fields=True)
    left = gfem.far_field(
        gfem.solve(model, problem, options=ORDER2), "l", 0.0, 0.0, model=model
    )
    assert abs(complex(left.e_right[0])) < 0.1 * abs(complex(left.e_left[0]))
