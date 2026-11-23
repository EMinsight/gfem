r"""A particle on a layered substrate: the power balance of its cross sections.

A lossy cube particle on a lossless film over a glass substrate, under air,
in a unit cube (um) wrapped in a perfectly matched layer whose regions are
split at the stack's planes, at order 3. The extinction is the power the
particle removes from the stack's own field (reflected and transmitted
waves included); it equals absorption plus scattering, and each part is
the same read on any side holding the particle: the corrected power flow of
the total field is minus the absorbed power, the scattered field's flow the
scattered power. Incidence from the top at 40 degrees and from the
substrate at 50 degrees (total internal reflection: the particle is lit by
the evanescent tail), TE and TM.

Measured (order 3, one thread, 2026-11-23): the balance on every side to
about :math:`10^{-12}`; the values (observations, not gates) are recorded
in `test_the_values_are_those_of_a_particle_of_its_size`.
"""

import math

import numpy as np
import pml_cases as pc
import pytest

import gfem

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
F = pc.F0
UM = 1e-6
SUB = gfem.Material(eps_r=2.25)
FILM = gfem.Material(eps_r=4.0)
AIR = gfem.Material()
PARTICLE = gfem.Material(eps_r=4.0, tan_delta=0.25)
#: Substrate below z = 1/2, film to z = 2/3, air above (um): grid planes.
STACK = gfem.Background([SUB, FILM, AIR], interfaces=[0.5 * UM, 2 / 3 * UM])
SOURCES = (
    gfem.PlaneWave("top-te", theta=math.pi - math.radians(40), polarization="TE"),
    gfem.PlaneWave("top-tm", theta=math.pi - math.radians(40), polarization="TM"),
    gfem.PlaneWave("tir-te", theta=math.radians(50), phi=0.3, polarization="TE"),
    gfem.PlaneWave("tir-tm", theta=math.radians(50), phi=0.3, polarization="TM"),
)
#: The particle's edge, um: a third of a wavelength.
EDGE = 1 / 3
REL = 1e-10


def _layer(z):
    return "sub" if z < 0.5 else "film" if z < 2 / 3 else "air"


def _volume(c):
    inside = bool(((c > 0) & (c < 1)).all())
    if not inside:
        return f"pml_{_layer(c[2])}"
    if (c[:2] > 1 / 3).all() and (c[:2] < 2 / 3).all() and 2 / 3 < c[2] < 5 / 6:
        return "particle"
    if (c > 1 / 6).all() and (c < 5 / 6).all():
        return f"box_{_layer(c[2])}"  # a side around it, one region per layer
    return _layer(c[2])


BOX = ("particle", "box_sub", "box_film", "box_air")


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("layered") / "particle.msh",
        pc.SIDES,
        thickness=1 / 3,
        cells=6,
        volume=_volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    materials = {"particle": PARTICLE}
    for layer, material in (("sub", SUB), ("film", FILM), ("air", AIR)):
        for prefix in ("", "pml_", "box_"):
            materials[prefix + layer] = material
    model = gfem.Model(
        mesh,
        materials=materials,
        pml=gfem.PML(["pml_sub", "pml_film", "pml_air"]),
        background=STACK,
    )
    problem = gfem.Driven(F, sources=SOURCES, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    sections = {
        s.name: gfem.cross_sections(result, s.name, model=model) for s in SOURCES
    }
    return model, result, sections


def _value(x):
    return float(np.asarray(x).reshape(-1)[0])


@pytest.mark.parametrize("name", [s.name for s in SOURCES])
def test_t18_extinction_is_absorption_plus_scattering(solved, name):
    _, _, sections = solved
    xs = sections[name]
    ext, absorption, sca = (
        _value(xs.extinction),
        _value(xs.absorption),
        _value(xs.scattering),
    )
    assert ext == pytest.approx(absorption + sca, rel=REL)
    assert absorption > 0
    assert sca > 0


@pytest.mark.parametrize("name", [s.name for s in SOURCES])
@pytest.mark.parametrize("side", [("particle",), BOX], ids=["particle", "box"])
def test_t18_each_power_is_the_same_on_any_side(solved, name, side):
    model, result, sections = solved
    xs = sections[name]
    i0 = _value(xs.intensity)
    total = result.e_field(F, name)
    scattered = result.e_field(F, name, part="scattered")
    absorbed = _value(gfem.power_flow(total, side, model=model))
    assert absorbed == pytest.approx(-_value(xs.absorption) * i0, rel=REL)
    flow = _value(gfem.power_flow(scattered, side, model=model))
    assert flow == pytest.approx(_value(xs.scattering) * i0, rel=REL)


def test_the_values_are_those_of_a_particle_of_its_size(solved):
    _, _, sections = solved
    area = (EDGE * UM) ** 2
    for name, xs in sections.items():
        ext = _value(xs.extinction)
        # a resonant-size lossy particle: extinction of order its area
        assert 0.05 * area < ext < 10 * area, (name, ext / area)
