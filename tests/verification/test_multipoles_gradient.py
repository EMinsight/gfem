r"""Gradients of the multipoles by a material design against finite differences.

The case is the lossy cube of the free-space gradient test (six perfectly
matched sides, order 2) with the multipole origin at its centre. Each
gradient matches a fourth-order central difference to :math:`2.5\times10^{-9}`
relative, above the measured worst 5.8e-10 (`finite_differences`):

- the magnetic dipole's scattering cross section under a TM plane wave
  (the default degree is 8 on the grid of 15 polar nodes, pinned here);
- a phase-sensitive Kerker mix of the dipole and quadrupole coefficients,
  plus the exact moments :math:`|p|^2 - |m|^2/c^2`;
- the electric dipole's share of the scattering;
- the same cross section under a circularly polarized Gaussian beam.

The multipoles read the far-field layer and the scattered field only, so the
design reaches them through the solved field: with the field held fixed the
gradient is exactly zero.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
AIR = gfem.Material()
F = pc.F0 / 2
C0 = 299792458.0
TM = gfem.PlaneWave("tm", theta=0.4, phi=0.3, polarization="TM")
BEAM = gfem.GaussianBeam(
    "beam",
    waist=0.8e-6,
    focus=(0.5e-6, 0.5e-6, 0.5e-6),
    theta=0.4,
    phi=0.3,
    polarization="RCP",
)
#: The cube's permittivity: the design is its real part.
EPS, LOSS = 3.0, 0.6j
ORIGIN = (0.5e-6, 0.5e-6, 0.5e-6)
STEP = 1e-2
#: The directional derivatives' relative error, measured 6e-11 to 5.8e-10.
REL = 2.5e-9


def _volume(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("multipoles") / "box.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        cells=4,
        volume=_volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("scat"))
    return gfem.Model(
        mesh,
        materials={
            "air": AIR,
            "pml": AIR,
            "scat": gfem.MaterialField(eps_r=np.full(n, EPS - LOSS)),
        },
        pml=gfem.PML("pml"),
        background=AIR,
    )


def _design(model, eps):
    return model.with_material("scat", gfem.MaterialField(eps_r=eps - LOSS))


def _start(model):
    return jnp.full(len(model.mesh.elements("scat")), EPS)


def _multipoles(model, eps, source):
    moved = _design(model, eps)
    problem = gfem.Driven(F, sources=source, keep_fields=True)
    result = gfem.solve(moved, problem, options=OPTIONS)
    return gfem.multipoles(result, source.name, model=moved, origin=ORIGIN)


def _check(objective, model, seed):
    eps = _start(model)
    directions = unit_directions(np.random.default_rng(seed), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=STEP, rel=REL)


def test_g1_the_magnetic_dipole_cross_section_by_the_scatterer_permittivity(cube):
    mp = _multipoles(cube, _start(cube), TM)
    # the default degree and grid are part of the gate
    assert mp.lmax == 8

    def objective(eps):
        return 1e12 * _multipoles(cube, eps, TM).magnetic_scattering[0, 0]

    _check(objective, cube, 11)


def test_g2_a_kerker_mix_and_the_exact_moments_by_the_scatterer_permittivity(cube):
    mp = _multipoles(cube, _start(cube), TM)
    scale = float(jnp.sum(jnp.abs(mp.electric_dipole[0]) ** 2))
    a11, a1m = mp.index(1, 1), mp.index(1, -1)
    a20 = mp.index(2, 0)

    def objective(eps):
        m = _multipoles(cube, eps, TM)
        al, be = m.alpha[0], m.beta[0]
        k2 = m.wavenumber[0] ** 2
        mix = (
            jnp.abs(al[a11] - be[a11]) ** 2
            + 0.5 * jnp.real(al[a1m] * jnp.conj(be[a1m]))
            - 0.3 * jnp.abs(al[a20]) ** 2
        ) / k2
        p, mag = m.electric_dipole[0], m.magnetic_dipole[0]
        moments = (jnp.sum(jnp.abs(p) ** 2) - jnp.sum(jnp.abs(mag) ** 2) / C0**2) / (
            scale
        )
        return 1e12 * mix + 0.1 * moments

    _check(objective, cube, 12)


def test_g3_the_electric_dipole_share_by_the_scatterer_permittivity(cube):
    def objective(eps):
        m = _multipoles(cube, eps, TM)
        return m.electric_scattering[0, 0] / m.scattering[0]

    _check(objective, cube, 13)


def test_g4_the_magnetic_dipole_cross_section_under_a_gaussian_beam(cube):
    def objective(eps):
        return 1e12 * _multipoles(cube, eps, BEAM).magnetic_scattering[0, 0]

    _check(objective, cube, 14)


def test_g5_the_multipoles_read_no_scatterer_material(cube):
    moved = _design(cube, _start(cube))
    problem = gfem.Driven(F, sources=TM, keep_fields=True)
    frozen = jax.lax.stop_gradient(gfem.solve(moved, problem, options=OPTIONS))

    def objective(eps):
        model = _design(cube, eps)
        m = gfem.multipoles(frozen, "tm", model=model, origin=ORIGIN)
        return 1e12 * m.magnetic_scattering[0, 0]

    gradient = jax.grad(objective)(_start(cube))
    assert np.all(np.asarray(gradient) == 0)
    assert math.isfinite(float(objective(_start(cube))))
