"""Wave ports whose cross-section holds a dispersive dielectric (ADR 0114).

A 10 mm WR-28 guide filled with a permittivity law, between two wave ports,
over 14-18 GHz. In a uniform guide TE10 has the closed form ``gamma =
sqrt((pi / a)^2 - eps(f) k0^2)`` at every frequency (``eps(f)`` complex), the
guide does not reflect, and S21 is ``exp(-gamma L)``. Two fills:

- a Debye law relaxing inside the band (eps 3.002 - 0.001j at 16 GHz);
- a Lorentz law below its pole at 30 GHz (eps 3.92 to 4.34 over the band).

Both are weakly lossy so that the guide's own reflection, of order
alpha/beta for a lossy mode referenced to itself (`gfem.conventions`), stays
below 1e-3.

The port mode sweep projects one susceptibility piece per law: its modes
equal those of a constant section, ``Material(eps_r=Re eps(f), tan_delta=
-Im eps(f) / Re eps(f))``, solved at that one frequency. So do those of a
section whose sides are absorbing and whose triangles beside them hold the
law (the law's own edge piece).
"""

import math
from pathlib import Path

import anisotropy
import numpy as np
import pytest
import structured_meshes as sm

import gfem

C0 = 299792458.0
LENGTH = 10.0  # mm
A = anisotropy.WR28[0] * 1e-3  # m
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
GRID = np.linspace(14e9, 18e9, 9)

DEBYE = gfem.PermittivityLaw(
    eps_inf=3.0, terms=[gfem.Debye(delta_eps=0.002, relaxation_frequency=16e9)]
)
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.5, resonance_frequency=30e9, damping=2e7)],
)
LAWS = {"debye": DEBYE, "lorentz": LORENTZ}


def _eps(law, f):
    return complex(law(f))


def _te10(law, f):
    """TE10's propagation constant in the filled guide, 1/m (Re >= 0)."""
    k0 = 2 * math.pi * f / C0
    gamma = np.sqrt(complex((math.pi / A) ** 2 - _eps(law, f) * k0**2))
    return gamma if gamma.real >= 0 else -gamma


def _constant(law, f):
    """The constant material with the law's permittivity at ``f``."""
    eps = _eps(law, f)
    return gfem.Material(eps_r=eps.real, tan_delta=-eps.imag / eps.real)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return anisotropy.wr28(
        tmp_path_factory.mktemp("wr28"), length=LENGTH, cells=(8, 4, 10)
    )


def _guide(mesh, material):
    return gfem.Model(
        mesh,
        materials={"fill": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


@pytest.fixture(scope="module", params=sorted(LAWS))
def driven(request, mesh):
    law = LAWS[request.param]
    model = _guide(mesh, gfem.Material(eps_r=law))
    return law, gfem.solve(model, gfem.Driven(GRID), options=OPTIONS)


def test_te10_has_the_closed_form_propagation_constant_at_every_frequency(driven):
    law, result = driven
    gamma = np.asarray(result.port_modes["p1"].gamma)
    expected = np.array([_te10(law, f) for f in GRID])
    # measured on ostau (debye, lorentz): 9.2e-9, 4.1e-9
    np.testing.assert_allclose(gamma, expected, rtol=2e-8)


def test_the_filled_guide_transmits_its_mode_without_reflection(driven):
    law, result = driven
    s = np.asarray(result.s.values)
    expected = np.exp(-np.array([_te10(law, f) for f in GRID]) * LENGTH * 1e-3)
    # measured on ostau (debye, lorentz): |S11| = |S22| 4.0e-4, 2.2e-4 (the
    # lossy line's own reflection, alpha/beta about 1e-3); S21's phase
    # against gamma L 1.5e-6, 5.0e-6 rad and its magnitude 3.0e-7, 4.5e-8
    assert np.abs(s[:, 0, 0]).max() < 1e-3
    assert np.abs(s[:, 1, 1]).max() < 1e-3
    assert np.abs(np.angle(s[:, 1, 0] / expected)).max() < 1.5e-5
    assert np.abs(np.abs(s[:, 1, 0]) - np.abs(expected)).max() < 1e-6


#: The grid frequencies where the swept modes meet constant sections.
CHECKED = (0, 4, 8)


def _swept_against_constant(model, region, law):
    """Max relative distance of the swept gamma from constant-section solves."""
    swept = np.asarray(
        gfem.solve(model, gfem.PortMode("p1", GRID), options=OPTIONS).gamma
    )
    errors = []
    for i in CHECKED:
        constant = model.with_material(region, _constant(law, GRID[i]))
        result = gfem.solve(constant, gfem.PortMode("p1", GRID[i]), options=OPTIONS)
        reference = complex(np.asarray(result.gamma)[0])
        errors.append(abs(swept[i] - reference) / abs(reference))
    return max(errors)


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_swept_mode_equals_a_constant_section_at_each_frequency(mesh, name):
    law = LAWS[name]
    model = _guide(mesh, gfem.Material(eps_r=law))
    # measured on ostau (debye, lorentz): 2.7e-15, 9.0e-16
    assert _swept_against_constant(model, "fill", law) < 1e-10


@pytest.fixture(scope="module")
def open_mesh(tmp_path_factory):
    """WR-28's cross-section with absorbing sides (x = 0, a), 2 mm long."""
    a, b = anisotropy.WR28
    length = 2.0
    edges = (np.linspace(0, a, 9), np.linspace(0, b, 5), np.linspace(0, length, 3))
    tol = 1e-9 * a

    def surface(c):
        if abs(c[2]) < tol:
            return "port1"
        if abs(c[2] - length) < tol:
            return "port2"
        if abs(c[0]) < tol or abs(c[0] - a) < tol:
            return "sides"
        return "walls"

    mesh = sm.structured(edges, lambda _: "fill", surface)
    path = sm.write_msh22(Path(tmp_path_factory.mktemp("open")) / "open.msh", mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.mark.parametrize("name", sorted(LAWS))
def test_absorbing_edges_beside_a_law_take_its_permittivity_at_each_frequency(
    open_mesh, name
):
    law = LAWS[name]
    model = gfem.Model(
        open_mesh,
        materials={"fill": gfem.Material(eps_r=law)},
        boundaries=[gfem.PEC("walls"), gfem.Absorbing("sides")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    # measured on ostau (debye, lorentz): 5.3e-14, 9.2e-14; 2.7e-3, 2.2e-3
    # with the law's edge admittance 1 % off
    assert _swept_against_constant(model, "fill", law) < 1e-10
