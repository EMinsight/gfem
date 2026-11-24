r"""The Gaussian beam's field on the host, its normalization and its refusals.

The beam is evaluated where a solve evaluates it (`gfem._incident.values`)
and checked against what it must be, independently of its closed form:
Maxwell's equations by fourth-order finite differences, the angular
spectrum of real plane waves it is a superposition of, and 1 W through a
plane normal to its axis by a quadrature of the Poynting flux. Lengths in
wavelengths of the background, 1 um.
"""

import math

import numpy as np
import pml_cases as pc
import pytest
import scipy.constants as const

import gfem
import gfem.conventions
from gfem._incident import magnetic, values
from gfem._sources import polarization_vector

C0 = const.c
ETA0 = const.mu_0 * const.c
UM = 1e-6
F = C0 / UM
K = 2 * math.pi / UM
AIR = gfem.Background(gfem.Material())


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    mesh = gfem.Mesh.from_gmsh(
        pc.box(tmp_path_factory.mktemp("beam") / "b.msh", pc.SIDES, cells=2)
    )
    air = gfem.Material()
    return gfem.Model(
        mesh, materials={"air": air, "pml": air}, pml=gfem.PML("pml"), background=air
    )


def _beam(kw0, **kwargs):
    args = {"theta": 0.0, "polarization": "TM"} | kwargs
    return gfem.GaussianBeam("b", waist=kw0 / K, **args)


def _field(beam, points, background=AIR, frequency=F):
    return values(beam, background, np.asarray(points, float), frequency)


def _derivative(fn, points, axis, h):
    e = np.zeros(3)
    e[axis] = h
    return (
        -fn(points + 2 * e)
        + 8 * fn(points + e)
        - 8 * fn(points - e)
        + fn(points - 2 * e)
    ) / (12 * h)


def _residuals(beam, points, h):
    r"""The curl-curl and divergence residuals of the beam at ``points``.

    :math:`|\nabla\times\nabla\times E - k^2 E| / (k^2|E|)` and
    :math:`|\nabla\cdot E| / (k|E|)`.
    """

    def e(p):
        return _field(beam, p)

    def div(p):
        return sum(_derivative(lambda q, i=i: e(q)[:, i], p, i, h) for i in range(3))

    grad_div = np.stack([_derivative(div, points, i, h) for i in range(3)], axis=1)
    lap = 0
    for i in range(3):
        s = np.zeros(3)
        s[i] = h
        lap = lap + (
            -e(points + 2 * s)
            + 16 * e(points + s)
            - 30 * e(points)
            + 16 * e(points - s)
            - e(points - 2 * s)
        ) / (12 * h * h)
    field = e(points)
    size = np.linalg.norm(field, axis=1)
    curl_curl = grad_div - lap
    return (
        np.linalg.norm(curl_curl - K**2 * field, axis=1) / (K**2 * size),
        np.abs(div(points)) / (K * size),
    )


@pytest.mark.parametrize("kw0", [3.0, 12.0])
@pytest.mark.parametrize(
    "direction",
    [
        {"theta": 0.0, "polarization": "TM"},
        {
            "theta": 2.2,
            "phi": 0.7,
            "polarization": "RCP",
            "focus": (0.3e-6, -0.2e-6, 0.5e-6),
        },
    ],
    ids=["axis", "tilted"],
)
def test_the_beam_solves_maxwell_in_its_background(kw0, direction):
    beam = _beam(kw0, **direction)
    w0 = beam.waist
    zr = K * w0**2 / 2
    focus = np.array(beam.focus)
    rng = np.random.default_rng(0)
    points = focus + rng.uniform(-1, 1, (8, 3)) * np.array([1.5 * w0, 1.5 * w0, 2 * zr])
    curl, div = _residuals(beam, points, min(w0, UM) / 100)
    # the finite-difference floor at this step, falling as h^4: a paraxial
    # beam is off by 7e-2 at k w0 = 12
    assert curl.max() < 2e-6
    assert div.max() < 2e-6


def _spectrum_field(kw0, points, n=400, m=64):
    r"""1 W of the spectrum :math:`(1+\cos\theta)e^{a(\cos\theta-1)}/(4\pi)`.

    The beam along +z, along x, focused at the origin: the plane waves
    :math:`(\cos\phi\,\hat\theta - \sin\phi\,\hat\phi)e^{-ik\hat s\cdot r}`,
    Gauss-Legendre in :math:`\cos\theta` on each hemisphere, trapezoidal in
    :math:`\phi`; its power through a plane normal to z is
    :math:`(2\pi/k)^2/(2\eta)` times the difference of
    :math:`\int|A|^2 d\Omega` over the two hemispheres.
    """
    a = kw0**2 / 2
    x, w = np.polynomial.legendre.leggauss(n)
    u = np.concatenate([(x - 1) / 2, (x + 1) / 2])
    wu = np.concatenate([w, w]) / 2
    phi = 2 * np.pi * np.arange(m) / m
    weight = (1 + u) * np.exp(a * (u - 1)) / (4 * np.pi)
    sign = np.sign(u)
    power = (
        (2 * np.pi / K) ** 2 / (2 * ETA0) * 2 * np.pi * np.sum(wu * sign * weight**2)
    )
    uu, pp = (g.ravel() for g in np.meshgrid(u, phi, indexing="ij"))
    ww = np.repeat(wu * weight, m) * (2 * np.pi / m)
    st = np.sqrt(1 - uu**2)
    s = np.stack([st * np.cos(pp), st * np.sin(pp), uu], axis=1)
    theta_hat = np.stack([uu * np.cos(pp), uu * np.sin(pp), -st], axis=1)
    phi_hat = np.stack([-np.sin(pp), np.cos(pp), 0 * pp], axis=1)
    pol = np.cos(pp)[:, None] * theta_hat - np.sin(pp)[:, None] * phi_hat
    phase = np.exp(-1j * K * np.asarray(points) @ s.T)
    field = (phase * ww) @ pol
    return field / np.sqrt(power)


@pytest.mark.parametrize("kw0", [3.0, 6.0])
def test_the_beam_is_its_angular_spectrum_carrying_one_watt(kw0):
    beam = _beam(kw0)
    w0, zr = beam.waist, K * beam.waist**2 / 2
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, (6, 3)) * np.array([2 * w0, 2 * w0, 2 * zr])
    points = np.concatenate([np.zeros((1, 3)), points])
    expected = _spectrum_field(kw0, points)
    np.testing.assert_allclose(
        _field(beam, points), expected, rtol=0, atol=1e-8 * np.abs(expected).max()
    )


def test_one_watt_crosses_a_plane_normal_to_the_axis():
    beam = _beam(6.0, theta=0.0, phi=0.4, polarization=(1.0, 0.5j))
    w0, zr = beam.waist, K * beam.waist**2 / 2
    side = np.linspace(-12 * w0, 12 * w0, 401)
    x, y = np.meshgrid(side, side, indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, zr)], axis=1)
    e = _field(beam, points)
    h = magnetic(beam, AIR, points, F)
    flux = 0.5 * np.real(np.cross(e, np.conj(h))[:, 2]).sum() * (side[1] - side[0]) ** 2
    assert flux == pytest.approx(gfem.conventions.INCIDENT_POWER, rel=1e-6)


def test_a_wide_beam_is_finite_and_paraxial():
    # k w0 = 200: e^a would overflow; the focal field is the paraxial one to
    # 1/(k w0)^2
    beam = _beam(200.0)
    w0, zr = beam.waist, K * beam.waist**2 / 2
    points = np.array([[0, 0, 0], [w0, 0, 3 * zr], [5 * w0, 2 * w0, -zr]])
    field = _field(beam, points)
    assert np.isfinite(field).all()
    paraxial = math.sqrt(4 * ETA0 / (math.pi * w0**2))
    assert abs(field[0, 0]) == pytest.approx(paraxial, rel=1e-4)


def test_the_field_is_continuous_across_the_ring_and_the_series_switch():
    # in the focal plane R^2 = rho^2 - b^2: zero on the ring rho = b, and
    # |k^2 R^2| = 1/4 where the closed forms take over from the series
    beam = _beam(3.0)
    b = K * beam.waist**2 / 2
    for rho2 in (b**2, b**2 + 0.25 / K**2, b**2 - 0.25 / K**2):
        rho = np.sqrt(rho2 * (1 + np.array([-1e-9, 0.0, 1e-9])))
        points = np.stack([rho * np.cos(0.3), rho * np.sin(0.3), 0 * rho], axis=1)
        field = _field(beam, points)
        assert np.abs(field - field[1]).max() < 1e-7 * np.abs(field[1]).max()


@pytest.mark.parametrize("polarization", ["TM", "TE", "RCP", (0.3, 2.0 - 1j)], ids=str)
def test_the_focal_field_is_real_and_positive_along_the_polarization(
    model, polarization
):
    beam = _beam(
        4.0, theta=2.0, phi=-0.6, focus=(1e-7, 2e-7, -3e-7), polarization=polarization
    )
    peak = beam.focal_field(F, model=model)
    at_focus = _field(beam, [beam.focus])[0]
    direction = polarization_vector(beam.theta, beam.phi, beam.polarization)
    np.testing.assert_allclose(at_focus, peak * direction, rtol=0, atol=1e-12 * peak)
    assert peak > 0
    assert beam.focal_intensity(F, model=model) == pytest.approx(
        peak**2 / (2 * ETA0), rel=1e-12
    )


def test_the_te_beam_is_the_tm_beam_turned_about_its_axis():
    tm = _beam(4.0, theta=0.0, phi=0.0, polarization="TM")
    te = _beam(4.0, theta=0.0, phi=np.pi / 2, polarization="TM")
    turned = _beam(4.0, theta=0.0, phi=0.0, polarization="TE")
    rng = np.random.default_rng(2)
    points = rng.uniform(-1, 1, (5, 3)) * tm.waist * 3
    np.testing.assert_allclose(
        _field(turned, points),
        _field(te, points),
        atol=1e-12 * abs(_field(tm, [[0, 0, 0]])).max(),
    )


def test_a_denser_background_brings_the_beam_closer_to_paraxial(model):
    # 1 W in the paraxial limit is I0 = 2 / (pi w0^2) whatever the medium; a
    # tight beam puts less on its axis, and a denser medium (k w0 up by 1.5)
    # loosens it
    beam = _beam(3.0)
    paraxial = 2 / (math.pi * beam.waist**2)
    air = beam.focal_intensity(F, model=model) / paraxial
    glass = model.replace(background=gfem.Material(eps_r=2.25))
    dense = beam.focal_intensity(F, model=glass) / paraxial
    assert 0.6 < air < dense < 1


@pytest.mark.parametrize(
    ("background", "error", "match"),
    [
        (gfem.Material(tan_delta=0.01), NotImplementedError, "lossy background"),
        (
            gfem.Background(
                [gfem.Material(), gfem.Material(eps_r=2.0)], interfaces=[0.0]
            ),
            NotImplementedError,
            "layered background",
        ),
    ],
)
def test_a_beam_needs_a_homogeneous_lossless_background(
    model, background, error, match
):
    beam = _beam(4.0)
    with pytest.raises(error, match=match):
        beam.focal_field(F, model=model.replace(background=background))
    with pytest.raises(error, match=match):
        _field(
            beam,
            [[0, 0, 0]],
            background=gfem.Background(background)
            if isinstance(background, gfem.Material)
            else background,
        )


def test_a_waist_too_small_to_normalize_is_refused(model):
    beam = _beam(0.19)
    with pytest.raises(ValueError, match=r"k w0 = 0\.19 .* below 0\.2"):
        beam.focal_intensity(F, model=model)
    with pytest.raises(ValueError, match=r"k w0 = 0\.19"):
        _field(beam, [[0, 0, 0]])
    assert beam.focal_field(2 * F, model=model) > 0


def test_the_focal_field_needs_a_model_with_a_background(model):
    beam = _beam(4.0)
    with pytest.raises(ValueError, match="background"):
        beam.focal_field(F, model=model.replace(background=None))
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        beam.focal_field(F, model=None)
