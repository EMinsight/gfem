r"""A plane wave's field in a background stack against closed forms.

The host transfer matrix gives the field of a `gfem.PlaneWave` in a
`gfem.Background` of several layers, or over a ground plane. Each check
compares it with an independent closed form, written out here: Fresnel's
coefficients at one interface (propagating and total internal reflection),
Airy's sum for one film (lossy), the standing wave over a perfect conductor,
and the conservation of the normal power flux through a lossless stack.

TE fields are compared through :math:`E`, TM fields through :math:`H`: both
lie along :math:`\hat\phi`, so each is one scalar times
:math:`e^{-i k_t\cdot r}`, with :math:`k_z` on the branch
:math:`\operatorname{Im} k_z \le 0`.
"""

import math

import numpy as np
import pytest

import gfem
from gfem._transfer_matrix import StackWave

C0 = 299792458.0
ETA0 = 376.730313412
UM = 1e-6
F = C0 / UM  # vacuum wavelength 1 um
K0 = 2 * math.pi / UM
PHI = 0.3
RNG = np.random.default_rng(4)


def _kz(eps, kt):
    kz = np.sqrt(complex(eps) * K0**2 - kt**2 + 0j)
    return -kz if kz.imag > 0 else kz


def _points(lo, hi, n=40):
    """Random points with lo < z < hi, um extent, in m."""
    xy = RNG.uniform(-1, 1, (n, 2)) * UM
    z = RNG.uniform(lo, hi, n) * UM
    return np.c_[xy, z]


def _basis():
    kappa = np.array([math.cos(PHI), math.sin(PHI), 0.0])
    phi_hat = np.array([-math.sin(PHI), math.cos(PHI), 0.0])
    return kappa, phi_hat


def _along_phi(field):
    _, phi_hat = _basis()
    return field @ phi_hat


def _lateral(points, kt):
    kappa, _ = _basis()
    return np.exp(-1j * kt * (points @ kappa))


def _scalar(wave, background, points, pol):
    """The TE electric or TM magnetic field of the stack along phi-hat."""
    w = StackWave(background, wave, F)
    if pol == "TE":
        return _along_phi(w.electric(points))
    return _along_phi(w.magnetic(points))


def _check(actual, expected, tol=1e-12):
    scale = np.abs(expected).max()
    assert np.abs(actual - expected).max() <= tol * scale


def _wave(pol, theta, amplitude=1.0):
    return gfem.PlaneWave(
        "w", theta=theta, phi=PHI, polarization=pol, amplitude=amplitude
    )


def _fresnel(pol, eps1, eps2, kz1, kz2):
    """Reflection of the TE electric or the TM magnetic field."""
    if pol == "TE":
        return (kz1 - kz2) / (kz1 + kz2)
    return (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)


def _scale(pol, eps, amplitude):
    """The incident scalar's amplitude: E0 for TE, n E0 / eta0 for TM."""
    return amplitude if pol == "TE" else amplitude * math.sqrt(eps) / ETA0


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize(
    ("eps_in", "eps_out", "degrees"),
    [(2.25, 1.0, 30.0), (2.25, 1.0, 50.0), (1.0, 2.25, 40.0)],
    ids=["below", "tir", "denser"],
)
@pytest.mark.parametrize("upward", [True, False], ids=["up", "down"])
def test_one_interface_is_fresnel(pol, eps_in, eps_out, degrees, upward):
    z0 = 0.2  # the interface, um
    theta = math.radians(degrees)
    amplitude = 0.7 - 0.2j
    kt = K0 * math.sqrt(eps_in) * math.sin(theta)
    kz1, kz2 = _kz(eps_in, kt), _kz(eps_out, kt)
    r = _fresnel(pol, eps_in, eps_out, kz1, kz2)
    a = _scale(pol, eps_in, amplitude)
    medium = (gfem.Material(eps_r=eps_in), gfem.Material(eps_r=eps_out))
    if upward:
        background = gfem.Background(medium, interfaces=[z0 * UM])
        wave = _wave(pol, theta, amplitude)
        s = 1  # the incident wave runs along +z
    else:
        background = gfem.Background(medium[::-1], interfaces=[z0 * UM])
        wave = _wave(pol, math.pi - theta, amplitude)
        s = -1
    # the incident half-space, then the other one; z measured from the plane
    near = _points(z0 - 1.5, z0) if upward else _points(z0, z0 + 1.5)
    far = _points(z0, z0 + 1.5) if upward else _points(z0 - 1.5, z0)
    zn, zf = near[:, 2] - z0 * UM, far[:, 2] - z0 * UM
    # incident amplitude at the plane, phase at the origin
    a0 = a * np.exp(-1j * s * kz1 * z0 * UM)
    expected_near = a0 * (np.exp(-1j * s * kz1 * zn) + r * np.exp(1j * s * kz1 * zn))
    # the scalar is tangential, so continuous: t = 1 + r
    expected_far = a0 * (1 + r) * np.exp(-1j * s * kz2 * zf)
    _check(
        _scalar(wave, background, near, pol), expected_near * _lateral(near, kt)
    )
    _check(_scalar(wave, background, far, pol), expected_far * _lateral(far, kt))


@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("degrees", [0.0, 35.0, 70.0])
def test_a_lossy_film_is_airy(pol, degrees):
    eps = (2.25, 4.0 - 0.4j, 1.0)
    d = 0.12  # film thickness, um
    theta = math.radians(degrees)
    kt = K0 * 1.5 * math.sin(theta)
    kz = [_kz(e, kt) for e in eps]
    r12 = _fresnel(pol, eps[0], eps[1], kz[0], kz[1])
    r23 = _fresnel(pol, eps[1], eps[2], kz[1], kz[2])
    phase = np.exp(-2j * kz[1] * d * UM)
    r = (r12 + r23 * phase) / (1 + r12 * r23 * phase)
    t = (1 + r12) * (1 + r23) * np.exp(-1j * kz[1] * d * UM)
    t /= 1 + r12 * r23 * phase
    background = gfem.Background(
        [
            gfem.Material(eps_r=2.25),
            gfem.Material(eps_r=4.0, tan_delta=0.1),
            gfem.Material(),
        ],
        interfaces=[0.0, d * UM],
    )
    a = _scale(pol, 2.25, 1.0)
    below, above = _points(-1.5, 0.0), _points(d, 1.5)
    zb, za = below[:, 2], above[:, 2] - d * UM
    expected_below = a * (np.exp(-1j * kz[0] * zb) + r * np.exp(1j * kz[0] * zb))
    expected_above = a * t * np.exp(-1j * kz[2] * za)
    wave = _wave(pol, theta)
    _check(
        _scalar(wave, background, below, pol), expected_below * _lateral(below, kt)
    )
    _check(
        _scalar(wave, background, above, pol), expected_above * _lateral(above, kt)
    )


def _flux(wave, background, z):
    points = np.c_[np.zeros((len(z), 2)), np.asarray(z) * UM]
    w = StackWave(background, wave, F)
    e, h = w.electric(points), w.magnetic(points)
    return np.real(np.cross(e, np.conj(h)))[:, 2] / 2


@pytest.mark.parametrize(
    ("theta", "pol"),
    [
        (math.pi - math.radians(25), (1, 1j)),
        (math.radians(10), "TE"),
        (math.radians(20), "TM"),
        (math.radians(70), (0.6, 0.8j)),
    ],
    ids=["down", "up", "tir-tm", "tir-mixed"],
)
def test_the_power_flux_is_constant_through_a_lossless_stack(theta, pol):
    background = gfem.Background(
        [
            gfem.Material(eps_r=12.0),
            gfem.Material(eps_r=2.1),
            gfem.Material(eps_r=4.0),
            gfem.Material(),
        ],
        interfaces=[0.0, 0.3 * UM, 0.45 * UM],
    )
    flux = _flux(_wave(pol, theta), background, [-0.5, 0.1, 0.4, 0.9, 2.0])
    eps_in = 12.0 if theta < math.pi / 2 else 1.0
    incident = math.sqrt(eps_in) / (2 * ETA0) * abs(math.cos(theta))
    assert np.ptp(flux) <= 1e-14 * incident
    assert abs(flux[0]) <= incident * (1 + 1e-12)


def test_a_thick_frustrated_gap_stays_finite():
    glass, air = gfem.Material(eps_r=2.25), gfem.Material()
    background = gfem.Background(
        [glass, air, glass, air], interfaces=[0.0, 30 * UM, 30.5 * UM]
    )
    w = StackWave(background, _wave((0.6, 0.8j), math.radians(60)), F)
    points = _points(-1.0, 32.0, n=200)
    assert np.isfinite(w.electric(points)).all()
    assert np.abs(w.electric(points)).max() <= 3.0  # |1 + r| <= 2 per component


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_over_a_ground_plane_the_wave_is_standing(pol):
    z_g = -0.3  # um
    theta = math.radians(35)
    kt = K0 * math.sin(theta)
    kz = _kz(1.0, kt)
    background = gfem.Background(gfem.Material(), ground=z_g * UM)
    w = StackWave(background, _wave(pol, math.pi - theta, 2.0), F)
    points = _points(z_g, 1.0)
    z = points[:, 2]
    # down wave e^{+i kz z}; E_t(z_g) = 0 reflects TE with -1 and TM's H with +1
    r = -np.exp(2j * kz * z_g * UM) * (1 if pol == "TE" else -1)
    a = _scale(pol, 1.0, 2.0)
    expected = a * (np.exp(1j * kz * z) + r * np.exp(-1j * kz * z))
    got = _along_phi(w.electric(points) if pol == "TE" else w.magnetic(points))
    _check(got, expected * _lateral(points, kt))


def test_a_ground_plane_holds_no_tangential_field_under_a_stack():
    background = gfem.Background(
        [gfem.Material(eps_r=2.25), gfem.Material(eps_r=4.0, tan_delta=0.1)]
        + [gfem.Material()],
        interfaces=[0.0, 0.12 * UM],
        ground=-0.3 * UM,
    )
    w = StackWave(background, _wave((0.6, 0.8j), math.pi - 0.6), F)
    on = np.c_[RNG.uniform(-1, 1, (20, 2)) * UM, np.full(20, -0.3 * UM)]
    above = on + [0, 0, 0.5 * UM]
    tangential = np.abs(w.electric(on)[:, :2]).max()
    assert tangential <= 1e-14 * np.abs(w.electric(above)).max()
    # lossy film, PEC below: what comes down and is not reflected is absorbed
    flux = _flux(_wave((0.6, 0.8j), math.pi - 0.6), background, [0.5])
    assert flux[0] < 0


def test_incidence_layer_amplitude_and_phase_are_the_plane_wave():
    # where the stack ends far below, the incident part above is the plain wave
    background = gfem.Background(
        [gfem.Material(eps_r=2.25), gfem.Material()], interfaces=[-50 * UM]
    )
    wave = gfem.PlaneWave("w", theta=math.pi - 0.4, phi=PHI, polarization="RCP")
    w = StackWave(background, wave, F)
    points = _points(0.0, 1.0)
    from gfem._sources import polarization_vector, propagation_basis

    k_hat, _, _ = propagation_basis(math.pi - 0.4, PHI)
    plain = np.exp(-1j * K0 * points @ k_hat)[:, None] * polarization_vector(
        math.pi - 0.4, PHI, "RCP"
    )
    # minus the reflected wave, which is a fixed up-going plane wave
    kt = K0 * math.sin(0.4)
    reflected = w.electric(points) - plain
    kz = _kz(1.0, kt)
    up = np.exp(-1j * kz * points[:, 2]) * _lateral(points, kt)
    ratio = reflected / up[:, None]
    assert np.abs(ratio - ratio[0]).max() <= 1e-12 * np.abs(plain).max()
