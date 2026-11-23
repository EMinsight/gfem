r"""Gradients of the far field against finite differences.

Order 2, coarse boxes in a perfectly matched layer, each gradient against a
fourth-order central difference to :math:`10^{-6}` relative
(`finite_differences`):

- G3: the bistatic radar cross section of a plane wave's scatterer, and
  the directivity of a port-fed strip dipole, at one direction, by a
  `gfem.MaterialField` (the field-only kernel: the design is inside the
  side, away from the layer);
- N3: :math:`|F|^2` of the port-fed strip under a `boundary_morph` of its
  arms, on a named side whose element layer moves (the kernel's node pass:
  the plane wave's position-gradient term) and on the default side, whose
  layer, touching the perfectly matched layer, the morph holds; the strip
  then sits in a margin of air, so that the shell around the core is not
  held;
- N4: :math:`|F|^2` of an electric dipole under the same morph, whose
  extension moves the dipole's element (its point-transport term);
- the directivity of the strip under the morph, on the named side: the
  radiated power it divides by goes through `gfem.power_flow`'s node pass.
"""

import math

import free_space_cases as fsc
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem.shape import boundary_morph

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
EPS_STEP = 1e-2
NODE_STEP = 1e-9
#: The direction every objective reads.
THETA, PHI = 1.1, 0.4


#: The side around the strip's core.
CORE = ("left", "right", "design")


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    return fsc.strip(tmp_path_factory.mktemp("strip") / "strip.msh")


@pytest.fixture(scope="module")
def wide_strip(tmp_path_factory):
    return fsc.strip(tmp_path_factory.mktemp("wide") / "strip.msh", margin=True)


def _with_design(model, eps):
    return model.with_material("design", gfem.MaterialField(eps_r=eps))


def _far(model, name, problem, side=None):
    result = gfem.solve(model, problem, options=OPTIONS)
    return gfem.far_field(result, name, THETA, PHI, model=model, side=side)


def test_g3_directivity_of_a_port_fed_strip_by_a_design(strip):
    problem = gfem.Driven(fsc.F0, keep_fields=True)

    def directivity(eps):
        return _far(_with_design(strip, eps), "p1", problem).directivity[0]

    eps = jnp.asarray(strip.materials["design"].eps_r)
    directions = unit_directions(np.random.default_rng(0), eps.shape, 2)
    assert_gradient_matches(directivity, eps, directions, step=EPS_STEP)


def test_g3_radar_cross_section_of_a_scatterer_by_its_design(strip):
    # the strip's feed is no source here: the plane wave's column alone
    wave = gfem.PlaneWave("w", theta=0.3, phi=0.2, polarization="TM")
    problem = gfem.Driven(fsc.F0, excite=(), sources=wave, keep_fields=True)
    model = strip.replace(boundaries=[], ports=[])

    def rcs(eps):
        return _far(_with_design(model, eps), "w", problem).rcs[0]

    eps = jnp.asarray(strip.materials["design"].eps_r)
    directions = unit_directions(np.random.default_rng(1), eps.shape, 2)
    assert_gradient_matches(rcs, eps, directions, step=EPS_STEP)


def _morph(model):
    # the lower arm, below the design block
    morph = boundary_morph(
        model, "wire", grows="right", where=lambda x: x[:, 2] < (2 / 6 + 1e-3) * fsc.UM
    )
    assert morph.n_params > 0
    return morph


@pytest.mark.parametrize("side", [None, CORE], ids=["default", "core"])
def test_n3_far_field_of_a_port_fed_strip_under_a_morph(wide_strip, side):
    strip = wide_strip
    morph = _morph(strip)
    problem = gfem.Driven(fsc.F0, keep_fields=True)

    def power(t):
        ff = _far(strip.replace(mesh=morph(t)), "p1", problem, side)
        return jnp.abs(ff.e_theta[0]) ** 2 + jnp.abs(ff.e_phi[0]) ** 2

    t = jnp.zeros(morph.n_params)
    if side is None:
        fsc.assert_truncation_layer_held(strip, morph)
    else:
        fsc.assert_layer_moves(strip, morph, side)
    directions = unit_directions(np.random.default_rng(2), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)


def test_n4_far_field_of_a_dipole_under_a_morph(strip):
    morph = _morph(strip)
    position = np.array([0.43, 0.55, 0.47]) * fsc.UM
    dipole = gfem.ElectricDipole("d", position=position, moment=(0, 0, 1e-30))
    model = strip.replace(ports=[])
    problem = gfem.Driven(fsc.F0, sources=dipole, keep_fields=True)

    def power(t):
        ff = _far(model.replace(mesh=morph(t)), "d", problem)
        return jnp.abs(ff.e_theta[0]) ** 2 + jnp.abs(ff.e_phi[0]) ** 2

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(3), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)


def test_the_directivity_of_the_strip_under_a_morph(wide_strip):
    strip = wide_strip
    morph = _morph(strip)
    problem = gfem.Driven(fsc.F0, keep_fields=True)

    def directivity(t):
        return _far(strip.replace(mesh=morph(t)), "p1", problem, CORE).directivity[0]

    t = jnp.zeros(morph.n_params)
    fsc.assert_layer_moves(strip, morph, CORE)
    directions = unit_directions(np.random.default_rng(4), t.shape, 2)
    assert_gradient_matches(directivity, t, directions, step=NODE_STEP)


def test_far_field_directions_are_fixed(strip):
    problem = gfem.Driven(fsc.F0, keep_fields=True)
    result = gfem.solve(strip, problem, options=OPTIONS)

    def amplitude(theta):
        ff = gfem.far_field(result, "p1", theta, PHI, model=strip)
        return jnp.abs(ff.e_theta[0]) ** 2

    with pytest.raises(gfem.NonDifferentiableError, match="theta"):
        jax.grad(amplitude)(math.pi / 3)
