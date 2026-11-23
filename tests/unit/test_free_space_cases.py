"""The closed forms the free-space cases are gated against."""

import math

import free_space_cases as fsc
import pytest


def test_mie_matches_bohren_and_huffman_s_worked_example():
    # Bohren and Huffman, appendix A: m = 1.55, a = 0.525 um at 0.6328 um
    # (x = 5.213) gives Q_ext = Q_sca = 3.10543
    ext, absorption, sca = fsc.mie(1.55**2, 2 * math.pi * 0.525 / 0.6328)
    assert ext == pytest.approx(3.10543, abs=1e-5)
    assert absorption == pytest.approx(0, abs=1e-12)
    assert sca == pytest.approx(ext, rel=1e-12)


def test_a_small_lossy_sphere_absorbs_as_a_rayleigh_dipole():
    eps, x = 4.0 - 1.0j, 1e-3
    alpha = (eps - 1) / (eps + 2)
    _, absorption, sca = fsc.mie(eps, x)
    # under e^{+i w t} a lossy sphere has Im(alpha) < 0
    assert absorption == pytest.approx(-4 * x * alpha.imag, rel=1e-4)
    assert sca == pytest.approx(8 / 3 * x**4 * abs(alpha) ** 2, rel=1e-4)


def test_a_small_conducting_sphere_scatters_ten_thirds_x_to_the_fourth():
    _, absorption, sca = fsc.mie(None, 1e-2)
    assert sca == pytest.approx(10 / 3 * 1e-8, rel=1e-3)
    assert abs(absorption) < 1e-12


def test_cross_sections_scale_the_efficiencies_by_the_area():
    values = fsc.mie_cross_sections(4.0, fsc.RADIUS)
    q = fsc.mie(4.0, 2 * math.pi * fsc.RADIUS)
    assert values[0] == pytest.approx(q[0] * math.pi * (fsc.RADIUS * 1e-6) ** 2)


@pytest.mark.parametrize("normal", [True, False])
@pytest.mark.parametrize("height", [0.05, 0.2, 0.41])
def test_the_half_space_integral_over_a_conductor_is_image_theory(normal, height):
    image = fsc.image_power(height, normal=normal, sign=1)
    assert fsc.half_space_power(height, None, normal=normal) == pytest.approx(
        image, rel=1e-9
    )


def test_the_half_space_integral_over_vacuum_is_one():
    for normal in (True, False):
        assert fsc.half_space_power(0.2, 1.0, normal=normal) == pytest.approx(
            1.0, abs=1e-12
        )


def test_far_above_a_dielectric_the_power_tends_to_the_bulk_power():
    # the reflected field falls as 1/h: within a few percent at 5 wavelengths
    for normal in (True, False):
        ratio = fsc.half_space_power(5.0, 2.25, normal=normal)
        assert ratio == pytest.approx(1.0, abs=0.05)
