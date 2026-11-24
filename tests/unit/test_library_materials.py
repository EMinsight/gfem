"""`gfem.materials`: library materials built from cited published parameters.

Each entry is a function returning a plain `gfem.Material`. The expected
numbers below are the sources' own (ranges in Hz from the wavelength or
photon-energy ranges the papers state, indices at 1.55 um from the source
formulas), not recomputed the way the module computes them.
"""

import dataclasses
import json
import warnings
from pathlib import Path

import electro_optic_cases as eo
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


#: Each entry's name tag, and what its docstring and its row of the module's
#: sources table cite: the DOIs, then the fitted range in um and in Hz.
SOURCES = {
    "silicon": (
        "silicon (Li 1980)",
        ["10.1063/1.555624"],
        ["1.2-14 µm", "21.41-249.8 THz"],
    ),
    "silica": (
        "silica (Malitson 1965)",
        ["10.1364/JOSA.55.001205"],
        ["0.21-3.71 µm", "80.81-1428 THz"],
    ),
    "silicon_nitride": (
        "silicon nitride (Luke 2015)",
        ["10.1364/OL.40.004823"],
        ["0.31-5.504 µm", "54.47-967.1 THz"],
    ),
    **{
        metal: (
            f"{metal} (Rakic 1998)",
            ["10.1364/AO.37.005271"],
            ["0.1-6 eV", "0.21-12.4 µm", "24.18-1451 THz"],
        )
        for metal in ("gold", "copper", "aluminium")
    },
    "lithium_niobate": (
        "lithium niobate (Zelmon 1997, Turner 1966)",
        ["10.1364/JOSAB.14.003319", "10.1063/1.1754449"],
        ["0.4-5 µm", "59.96-749.5 THz"],
    ),
    "lithium_niobate_static": (
        "lithium niobate static (Smith & Welsh 1971)",
        ["10.1063/1.1660528"],
        [],
    ),
}


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_entry_names_its_source(entry):
    tag, dois, ranges = SOURCES[entry]
    assert _build(entry).name == tag
    doc = " ".join(getattr(materials, entry).__doc__.split())
    for text in [*dois, *ranges]:
        assert text in doc
    assert ("fitted range" in doc.lower()) == bool(ranges)


def _table_rows(doc):
    """The sources table's rows: the function name, then the row's text."""
    table = doc[doc.index(".. list-table::") :]
    rows = {}
    for block in table.split("   * - ")[2:]:
        name, _, rest = block.partition("\n")
        rows[name.strip().strip("`")] = " ".join(rest.split())
    return rows


def test_the_sources_table_lists_every_entry():
    rows = _table_rows(materials.__doc__)
    assert sorted(rows) == sorted(ENTRIES)
    for entry, (_, dois, ranges) in SOURCES.items():
        for text in [*dois, *ranges[:2]]:
            assert text in rows[entry]


def test_the_docstrings_state_the_out_of_range_values():
    for entry, stated in (
        ("silicon", "11.66"),
        ("silica", "3.00"),
        ("silicon_nitride", "265"),
    ):
        law = getattr(materials, entry)().eps_r
        assert f"{float(np.real(law(0.0))):.{len(stated.partition('.')[2])}f}" == stated
        assert stated in getattr(materials, entry).__doc__
    # the nitride refit's lossless infrared term sits at 100 um (3 THz)
    nitride = materials.silicon_nitride().eps_r
    assert any(
        t.damping == 0 and t.resonance_frequency == pytest.approx(299792458.0 / 100e-6)
        for t in nitride.terms
    )
    assert "100 µm" in materials.silicon_nitride.__doc__
    # the lithium niobate laws' static values, which are not the crystal's
    doc = materials.lithium_niobate.__doc__
    assert "17.5" in doc
    assert "13.5" in doc


#: The fitted ranges in Hz: each is its source's own printed range, 1.2-14 µm,
#: 0.31-5.504 µm, 0.21-3.71 µm and 0.1-6 eV.
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
    # silica at 1.55 µm is :math:`1.444^2`, not its static permittivity
    assert materials.silica(frequency=F_1550).eps_r == pytest.approx(2.085, abs=1e-3)


@pytest.mark.parametrize(
    "frequency",
    [
        np.array([F_1550, 2 * F_1550]),
        [F_1550],
        0.0,
        -F_1550,
        np.nan,
        1j,
        # a flag, not 1 Hz
        True,
        np.True_,
    ],
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


def _sellmeier(wavelength_um, b, c):
    lam2 = wavelength_um**2
    return 1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c, strict=True))


def test_the_library_feeds_the_electro_optic_shift(tmp_path):
    mesh = eo.layered_slab(tmp_path / "slab.msh", factor=2.0)
    drive = gfem.Electrostatic(terminals={"r": ["right"]}, ground=["left"])
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def shift(ln_static, ln, oxide):
        static = gfem.Model(
            mesh,
            materials={
                "film": ln_static,
                "below": gfem.Material(eps_r=3.9),
                "above": gfem.Material(eps_r=3.9),
            },
        )
        potential = gfem.solve(static, drive, options=options).potential["r"]
        optical = gfem.Model(
            mesh,
            materials={"below": oxide, "film": ln, "above": oxide},
            boundaries=[gfem.PEC("left"), gfem.PEC("right")],
            ports=[gfem.WavePort("p", "front")],
        )
        return gfem.electro_optic_shift(optical, potential, port="p", frequency=F_1550)

    library = shift(
        materials.lithium_niobate_static(axes=X_CUT),
        materials.lithium_niobate(frequency=F_1550, axes=X_CUT),
        materials.silica(frequency=F_1550),
    )
    # Zelmon's congruent Sellmeier, Malitson's, Turner's clamped tensor and
    # Smith and Welsh's static permittivity, typed from the papers
    n_o2 = _sellmeier(1.55, (2.6734, 1.2290, 12.614), (0.01764, 0.05914, 474.60))
    n_e2 = _sellmeier(1.55, (2.9804, 0.5981, 8.9543), (0.02047, 0.0666, 416.08))
    oxide = _sellmeier(
        1.55,
        (0.6961663, 0.4079426, 0.8974794),
        (0.0684043**2, 0.1162414**2, 9.896161**2),
    )
    by_hand = shift(
        gfem.Material(eps_r=(44.3, 44.3, 27.9), axes=X_CUT),
        gfem.Material(
            eps_r=(n_o2, n_o2, n_e2),
            axes=X_CUT,
            electro_optic_tensor=_tensor_3m(8.6e-12, 3.4e-12, 30.8e-12, 28e-12),
        ),
        gfem.Material(eps_r=oxide),
    )
    assert abs(float(library.delta_n_eff_per_volt)) > 0
    np.testing.assert_allclose(library.n_eff, by_hand.n_eff, rtol=1e-12)
    np.testing.assert_allclose(
        library.delta_n_eff_per_volt, by_hand.delta_n_eff_per_volt, rtol=1e-12
    )
