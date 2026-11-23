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

Measured (order 3, one thread, 2026-11-23): every power the same on both
sides to :math:`5\cdot10^{-13}`; extinction / absorption / scattering over
the particle's top area :math:`(\lambda/3)^2` (observations, not gates):
top TE 1.311 / 0.599 / 0.712, top TM 1.595 / 0.597 / 0.999, TIR TE 0.914 /
0.462 / 0.452, TIR TM 3.142 / 1.187 / 1.955.
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


def _solve(tmp_path):
    path = pc.box(
        tmp_path / "particle.msh", pc.SIDES, thickness=1 / 3, cells=6, volume=_volume
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
    return model, gfem.solve(model, problem, options=OPTIONS)


def _value(x):
    return float(np.asarray(x).reshape(-1)[0])


def test_t18_a_particle_on_a_substrate_balances_its_power_on_any_side(tmp_path):
    # one test, so that a sharded run builds the order-3 solve once
    model, result = _solve(tmp_path)
    area = (EDGE * UM) ** 2
    for source in SOURCES:
        name = source.name
        xs = gfem.cross_sections(result, name, model=model)
        ext, absorption, sca = (
            _value(xs.extinction),
            _value(xs.absorption),
            _value(xs.scattering),
        )
        assert ext == pytest.approx(absorption + sca, rel=REL), name
        assert absorption > 0, name
        assert sca > 0, name
        # a resonant-size lossy particle: extinction of order its area
        assert 0.05 * area < ext < 10 * area, (name, ext / area)
        i0 = _value(xs.intensity)
        total = result.e_field(F, name)
        scattered = result.e_field(F, name, part="scattered")
        for side in (("particle",), BOX):
            absorbed = _value(gfem.power_flow(total, side, model=model))
            assert absorbed == pytest.approx(-absorption * i0, rel=REL), (name, side)
            flow = _value(gfem.power_flow(scattered, side, model=model))
            assert flow == pytest.approx(sca * i0, rel=REL), (name, side)
