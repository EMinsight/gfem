r"""T15: a dipole above a dielectric half-space against Sommerfeld's integral.

An electric dipole in air at three heights above a glass half-space
(:math:`\varepsilon = 2.25`), normal and tangential, at order 3 on a
structured mesh with :math:`h = \lambda_\text{glass}/4`; the glass fills
the lower half of the unit cube (um) and the perfectly matched layer
below and beside it, whose regions are split at the interface. The emitted
power over the vacuum bulk power is compared with
`free_space_cases.half_space_power` (Sommerfeld's reflection integrals,
checked there against image theory over a perfect conductor).

Measured (order 3, one thread, 2026-11-23): see `MEASURED`; gate 3 %.
"""

import free_space_cases as fsc
import numpy as np
import pml_cases as pc
import pytest

import gfem

pytestmark = pytest.mark.reference

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
EPS = 2.25
REL = 3e-2
P = 1e-30
#: Heights above the interface z = 1/2, in wavelengths, off the grid planes.
HEIGHTS = (0.1, 0.2, 0.35)


def _volume(c):
    layer = "glass" if c[2] < 0.5 else "air"
    return layer if ((c > 0) & (c < 1)).all() else f"pml_{layer}"


def test_t15_a_dipole_above_glass_emits_the_sommerfeld_power(tmp_path):
    path = pc.box(
        tmp_path / "half.msh", pc.SIDES, thickness=1 / 3, cells=6, volume=_volume
    )
    mesh = gfem.Mesh.from_gmsh(path)
    glass, air = gfem.Material(eps_r=EPS), gfem.Material()
    model = gfem.Model(
        mesh,
        materials={"glass": glass, "air": air, "pml_glass": glass, "pml_air": air},
        pml=gfem.PML(["pml_glass", "pml_air"]),
    )
    dipoles, expected = [], {}
    for k, h in enumerate(HEIGHTS):
        r = np.array([0.47, 0.53, 0.5 + h]) * fsc.UM
        for axis, normal in (("z", True), ("x", False)):
            moment = (0, 0, P) if normal else (P, 0, 0)
            name = f"p{axis}{k}"
            dipoles.append(gfem.ElectricDipole(name, position=r, moment=moment))
            expected[name] = fsc.half_space_power(h, EPS, normal=normal)
    result = gfem.solve(model, gfem.Driven(fsc.F0, sources=dipoles), options=OPTIONS)
    bulk = fsc.bulk_power(fsc.OMEGA0, np.array([P, 0, 0]))
    errors = {}
    for name, reference in expected.items():
        ratio = float(result.emitted_power[name][0]) / bulk
        errors[name] = ratio / reference - 1
    print({name: f"{e:+.2%}" for name, e in errors.items()})
    assert max(abs(e) for e in errors.values()) <= REL, errors
