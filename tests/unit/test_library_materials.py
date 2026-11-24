"""`gfem.materials`: library materials built from cited published parameters.

Each entry is a function returning a plain `gfem.Material`. The expected
numbers below are the sources' own (ranges in Hz from the wavelength or
photon-energy bands the papers state, indices at 1.55 um from the source
formulas), not recomputed the way the module computes them.
"""

import gfem
from gfem import materials

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
