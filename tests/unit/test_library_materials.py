"""`gfem.materials`: library materials built from cited published parameters.

Each entry is a function returning a plain `gfem.Material`. The expected
numbers below are the sources' own (ranges in Hz from the wavelength or
photon-energy bands the papers state, indices at 1.55 um from the source
formulas), not recomputed the way the module computes them.
"""

import dataclasses
import json
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

import gfem
from gfem import _codec, materials

ENTRIES = (
    "silicon",
    "silica",
    "silicon_nitride",
    "gold",
    "copper",
    "aluminium",
    "lithium_niobate",
    "lithium_niobate_static",
)
#: x-cut for a section in (x, y) with the guide along z.
X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
F_1550 = 299792458.0 / 1.55e-6


def _build(entry):
    if entry == "lithium_niobate":
        return materials.lithium_niobate(frequency=F_1550, axes=X_CUT)
    if entry == "lithium_niobate_static":
        return materials.lithium_niobate_static(axes=X_CUT)
    return getattr(materials, entry)()


def test_the_module_holds_the_eight_entries_and_nothing_else():
    assert gfem.materials is materials
    assert "materials" in gfem.__all__
    assert sorted(materials.__all__) == sorted(ENTRIES)
    for entry in ENTRIES:
        assert type(_build(entry)) is gfem.Material
    # a re-export of Material here would move its API page off gfem.model
    assert not set(materials.__all__) & set(gfem.model.__all__)


#: ADR-fixed fitted ranges in Hz: the source bands 1.2-14 um, 0.31-5.504 um,
#: 0.21-3.71 um and 0.1-6 eV.
RANGES = {
    "silicon": (2.141375e13, 2.498270e14),
    "silicon_nitride": (5.446811e13, 9.670724e14),
    "silica": (8.080659e13, 1.427583e15),
    "gold": (2.417989e13, 1.450794e15),
    "copper": (2.417989e13, 1.450794e15),
    "aluminium": (2.417989e13, 1.450794e15),
}
DIELECTRICS = ("silicon", "silica", "silicon_nitride")


@pytest.mark.parametrize("entry", RANGES)
def test_each_law_carries_its_fitted_range(entry):
    law = getattr(materials, entry)().eps_r
    assert isinstance(law, gfem.PermittivityLaw)
    np.testing.assert_allclose(law.frequency_range, RANGES[entry], rtol=1e-6)


@pytest.mark.parametrize("entry", RANGES)
def test_library_laws_are_passive_on_a_dense_grid(entry):
    law = getattr(materials, entry)().eps_r
    lo, hi = law.frequency_range
    f = np.geomspace(1e-3 * lo, 1e3 * hi, 20000)
    poles = [t.resonance_frequency for t in law.terms if t.damping == 0]
    f = f[np.all([np.abs(f - p) > 1e-6 * p for p in poles], axis=0)]
    eps = np.asarray(law(f))
    assert np.all(np.isfinite(eps))
    assert np.all(eps.imag <= 0)
    if entry in DIELECTRICS:
        assert all(t.damping == 0 for t in law.terms)
        assert np.all(eps.imag == 0)


def _n(eps):
    return np.sqrt(complex(eps))


def test_the_values_at_1550_nm_are_pinned():
    for entry, n in (("silicon", 3.474948), ("silica", 1.444024)):
        assert _n(getattr(materials, entry)().eps_r(F_1550)) == pytest.approx(
            n, abs=1e-6
        )
    assert _n(materials.silicon_nitride().eps_r(F_1550)) == pytest.approx(
        1.996228, abs=1e-6
    )
    ln = materials.lithium_niobate(frequency=F_1550, axes=X_CUT)
    n_o, _, n_e = np.sqrt(ln.eps_r)
    assert n_o == pytest.approx(2.211111, abs=1e-6)
    assert n_e == pytest.approx(2.137560, abs=1e-6)
    for entry, n in (
        ("gold", 0.575 - 9.66j),
        ("copper", 0.470 - 10.45j),
        ("aluminium", 1.536 - 15.33j),
    ):
        assert _n(getattr(materials, entry)().eps_r(F_1550)) == pytest.approx(
            n, rel=1e-3
        )


def test_frequency_gives_the_law_s_real_value():
    for entry in DIELECTRICS:
        constant = getattr(materials, entry)(frequency=F_1550)
        law = getattr(materials, entry)().eps_r
        assert type(constant.eps_r) is float
        assert constant.eps_r == float(np.real(law(F_1550)))
        assert constant.name == getattr(materials, entry)().name
    # silica at 1.55 um is 1.444^2, not its static permittivity
    assert materials.silica(frequency=F_1550).eps_r == pytest.approx(2.085, abs=1e-3)


@pytest.mark.parametrize(
    "frequency", [np.array([F_1550, 2 * F_1550]), [F_1550], 0.0, -F_1550, np.nan, 1j]
)
def test_frequency_refusals(frequency):
    with pytest.raises(ValueError, match="frequency"):
        materials.silica(frequency=frequency)
    with pytest.raises(ValueError, match="frequency"):
        materials.lithium_niobate(frequency=frequency, axes=X_CUT)


def test_a_traced_frequency_is_refused():
    for build in (
        lambda f: materials.silicon(frequency=f).eps_r,
        lambda f: materials.lithium_niobate(frequency=f, axes=X_CUT).eps_r[0],
    ):
        with pytest.raises(gfem.NonDifferentiableError, match="frequency"):
            jax.grad(build)(F_1550)


def test_frequency_outside_the_range_warns_at_the_call():
    with pytest.warns(gfem.FittedRangeWarning, match="silicon") as record:
        constant = materials.silicon(frequency=299792458.0 / 1.0e-6)
    assert len(record) == 1
    assert "silicon (Li 1980)" in str(record[0].message)
    assert record[0].filename == __file__
    assert constant.eps_r == float(
        np.real(materials.silicon().eps_r(299792458.0 / 1.0e-6))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for entry in DIELECTRICS:
            getattr(materials, entry)(frequency=F_1550)
        # the edges of the range, as c / wavelength rounds them, count as inside
        materials.silicon(frequency=299792458.0 / 1.2e-6)
        materials.silicon(frequency=299792458.0 / 14e-6)
        materials.lithium_niobate(frequency=F_1550, axes=X_CUT)


def _tensor_3m(r13, r22, r33, r51):
    return (
        (0.0, -r22, r13),
        (0.0, r22, r13),
        (0.0, 0.0, r33),
        (0.0, r51, 0.0),
        (r51, 0.0, 0.0),
        (-r22, 0.0, 0.0),
    )


def test_lithium_niobate_on_the_given_axes():
    ln = materials.lithium_niobate(frequency=F_1550, axes=X_CUT)
    n_o2, _, n_e2 = ln.eps_r
    assert ln.eps_r == (n_o2, n_o2, n_e2)
    assert ln.axes == X_CUT
    assert ln.electro_optic_tensor == _tensor_3m(8.6e-12, 3.4e-12, 30.8e-12, 28e-12)
    assert ln.name == "lithium niobate (Zelmon 1997, Turner 1966)"
    # the fitted range is Zelmon's 0.4-5 um: one warning, at 0.35 and at 6 um
    for wavelength in (0.35e-6, 6e-6):
        with pytest.warns(gfem.FittedRangeWarning, match="lithium niobate") as record:
            materials.lithium_niobate(frequency=299792458.0 / wavelength, axes=X_CUT)
        assert len(record) == 1
        assert record[0].filename == __file__


def test_lithium_niobate_needs_frequency_and_right_handed_axes():
    with pytest.raises(TypeError):
        materials.lithium_niobate(axes=X_CUT)
    with pytest.raises(TypeError):
        materials.lithium_niobate(frequency=F_1550)
    with pytest.raises(TypeError):
        materials.lithium_niobate_static()
    left = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="right-handed"):
        materials.lithium_niobate(frequency=F_1550, axes=left)


def test_lithium_niobate_static_is_clamped():
    static = materials.lithium_niobate_static(axes=X_CUT)
    assert static.eps_r == (44.3, 44.3, 27.9)
    assert static.axes == X_CUT
    assert static.electro_optic_tensor is None
    assert static.name == "lithium niobate static (Smith & Welsh 1971)"


@pytest.mark.parametrize("entry", ["gold", "copper", "aluminium"])
def test_metals_take_no_frequency(entry):
    with pytest.raises(TypeError):
        getattr(materials, entry)(frequency=F_1550)


def test_replace_keeps_the_law_and_its_range():
    core = dataclasses.replace(materials.silicon(), name="core")
    assert core.name == "core"
    assert core.eps_r == materials.silicon().eps_r
    np.testing.assert_allclose(core.eps_r.frequency_range, RANGES["silicon"], rtol=1e-6)


@pytest.mark.parametrize("entry", ENTRIES)
def test_library_materials_round_trip_through_the_codec(entry):
    material = _build(entry)
    context = _codec.Context(Path())
    encoder = _codec.Encoder(context)
    doc = json.loads(json.dumps(encoder.encode(material), allow_nan=False))
    assert _codec.Decoder(context, encoder.arrays).decode(doc) == material
    assert ("frequency_range" in json.dumps(doc)) == (entry in RANGES)
    assert _codec.SCHEMA_VERSION == 2
