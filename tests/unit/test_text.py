"""Frequencies in messages: `gfem._text.hertz` and `gfem._text.hertz_band`."""

import pytest

from gfem._text import hertz, hertz_band


@pytest.mark.parametrize(
    ("f", "text"),
    [
        (193.414e12, "193.414 THz"),
        (50e9, "50 GHz"),
        (16.5714e9, "16.5714 GHz"),
        (1.5e12, "1.5 THz"),
        (2.5e6, "2.5 MHz"),
        (3e3, "3 kHz"),
        (60.0, "60 Hz"),
        (0.5, "0.5 Hz"),
        (0.0, "0 Hz"),
    ],
)
def test_a_frequency_takes_the_unit_that_puts_it_in_1_to_1000(f, text):
    assert hertz(f) == text


def test_digits_round_the_mantissa():
    assert hertz(193.414e12, 4) == "193.4 THz"
    assert hertz(2000e9, 4) == "2 THz"


def test_a_band_takes_the_unit_of_its_upper_end():
    assert hertz_band(55e9, 65e9, 4) == "55-65 GHz"
    assert hertz_band(500e9, 1.5e12) == "0.5-1.5 THz"
