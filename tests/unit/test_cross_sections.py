"""Cross sections and the corrected power flow of incident-field columns.

A unit cube of air ("air", um) in a thin perfectly matched layer, with a
one-cell cube scatterer ("scat") off its centre, on a coarse structured
mesh at order 2. The checks are identities of the discrete power balance,
which hold on any mesh, and the measurements' refusals.
"""

import math

import numpy as np
import pml_cases as pc
import pytest

import gfem

ALL = ("-x", "+x", "-y", "+y", "-z", "+z")
AIR = gfem.Material()
LOSSY = gfem.Material(eps_r=4.0, tan_delta=0.25)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0
TE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")
TM = gfem.PlaneWave("tm", theta=0.4, phi=0.3, polarization="TM", amplitude=2.0)
ETA0 = 376.730313412


def _inside(c, lo, hi):
    return bool(((c > lo) & (c < hi)).all())


def _scatterer(c):
    if _inside(c, 0.25, 0.5):
        return "scat"
    if _inside(c, 0.25, 0.75):
        return "box"
    return "air" if _inside(c, 0, 1) else "pml"


def _touching(c):
    """A scatterer of the cube's +x quarter, in the truncation layer."""
    if c[0] > 0.75 and _inside(c, 0, 1):
        return "scat"
    return "air" if _inside(c, 0, 1) else "pml"


def _air(c):
    return "air" if _inside(c, 0, 1) else "pml"


def _box(path, volume):
    return gfem.Mesh.from_gmsh(pc.box(path, ALL, thickness=0.25, cells=4, volume=volume))


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xsec")
    return {
        "scatterer": _box(tmp / "s.msh", _scatterer),
        "touching": _box(tmp / "t.msh", _touching),
        "empty": _box(tmp / "e.msh", _air),
    }


def _model(mesh, scat=None, **kw):
    materials = {"air": AIR, "box": AIR, "pml": AIR}
    if scat is not None:
        materials["scat"] = scat
    materials = {k: v for k, v in materials.items() if k in mesh.volumes}
    return gfem.Model(
        mesh, materials=materials, pml=gfem.PML("pml"), background=AIR, **kw
    )


def _solve(model, *sources, **kw):
    problem = gfem.Driven(F, sources=sources, keep_fields=True, **kw)
    return gfem.solve(model, problem, options=OPTIONS)


@pytest.fixture(scope="module")
def lossy(meshes):
    model = _model(meshes["scatterer"], LOSSY)
    return model, _solve(model, TE, TM)


def _value(x):
    return float(np.asarray(x).reshape(()))


# --- identities -----------------------------------------------------------------


def test_extinction_is_absorption_plus_scattering_exactly(lossy):
    model, result = lossy
    for name in ("te", "tm"):
        xs = gfem.cross_sections(result, name, model=model)
        np.testing.assert_array_equal(xs.extinction, xs.absorption + xs.scattering)
        assert (xs.absorption > 0).all()
        assert (xs.scattering > 0).all()


def test_plane_wave_intensity_is_its_peak_amplitude_over_two_eta(lossy):
    model, result = lossy
    np.testing.assert_allclose(
        gfem.cross_sections(result, "tm", model=model).intensity,
        [4.0 / (2 * ETA0)],
        rtol=1e-9,
    )


def test_corrected_power_flow_is_minus_the_absorbed_power(lossy):
    model, result = lossy
    for name in ("te", "tm"):
        xs = gfem.cross_sections(result, name, model=model)
        total = result.e_field(F, name)
        absorbed = xs.absorption[0] * xs.intensity[0]
        for side in ("scat", "box", ("scat", "box")):
            power = _value(gfem.power_flow(total, side, model=model))
            assert power == pytest.approx(-absorbed, rel=1e-10), side


def test_scattering_is_the_scattered_flux_through_any_side(lossy):
    model, result = lossy
    xs = gfem.cross_sections(result, "te", model=model)
    scattered = result.e_field(F, "te", part="scattered")
    for side in ("scat", "box"):
        power = _value(gfem.power_flow(scattered, side, model=model))
        assert power == pytest.approx(xs.scattering[0] * xs.intensity[0], rel=1e-10)


def test_a_lossless_scatterer_absorbs_nothing(meshes):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    xs = gfem.cross_sections(_solve(model, TE), "te", model=model)
    assert abs(xs.absorption[0]) <= 1e-10 * xs.scattering[0]


def test_without_a_scatterer_every_cross_section_vanishes(meshes, lossy):
    reference = gfem.cross_sections(lossy[1], "te", model=lossy[0])
    model = _model(meshes["empty"])
    xs = gfem.cross_sections(_solve(model, TE), "te", model=model)
    size = reference.extinction[0]
    for value in (xs.extinction, xs.absorption, xs.scattering):
        assert abs(value[0]) <= 1e-12 * size
