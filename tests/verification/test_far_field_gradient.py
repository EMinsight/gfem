r"""Gradients of the far field against finite differences.

Order 2, coarse boxes in a perfectly matched layer, each gradient against a
fourth-order central difference to :math:`10^{-6}` relative
(`finite_differences`):

- G3: the bistatic radar cross section of a plane wave's scatterer, and
  the directivity of a port-fed strip dipole, at one direction, by a
  `gfem.MaterialField` (the field-only kernel: the design is inside the
  side, away from the layer);
- N3: :math:`|F|^2` of the port-fed strip under a `boundary_morph` of its
  arms, on the default side, whose layer (touching the perfectly matched
  layer) the morph holds; and the intensity and the directivity at the
  solved coefficients by a random move of every node, layers included, on
  the default side and on a named side, to :math:`10^{-8}` (the kernel's
  node pass: the plane wave's position-gradient term, and for the
  directivity the radiated power's `gfem.power_flow` node pass), with the
  strip in a margin of air so that the shell around its core is no
  truncation layer;
- N4: :math:`|F|^2` of an electric dipole under the same morph, whose
  extension moves the dipole's element (its point-transport term);
- the directivity of the strip under the morph.
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
#: A measurement at fixed coefficients, relative (no solve in the difference).
KERNEL = 1e-8
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


def test_n3_far_field_of_a_port_fed_strip_under_a_morph(strip):
    # the default side: its layer touches the perfectly matched layer, and
    # the morph holds it
    morph = _morph(strip)
    problem = gfem.Driven(fsc.F0, keep_fields=True)

    def power(t):
        ff = _far(strip.replace(mesh=morph(t)), "p1", problem)
        return jnp.abs(ff.e_theta[0]) ** 2 + jnp.abs(ff.e_phi[0]) ** 2

    t = jnp.zeros(morph.n_params)
    fsc.assert_truncation_layer_held(strip, morph)
    directions = unit_directions(np.random.default_rng(2), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)


def _at(result, model, x):
    """``result`` and ``model`` with node coordinates ``x``, coefficients kept."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(result)
    moved = [
        x if jax.tree_util.keystr(p).endswith("mesh.nodes") else v for p, v in leaves
    ]
    mesh = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(model.mesh), [x])
    return jax.tree_util.tree_unflatten(treedef, moved), model.replace(mesh=mesh)


@pytest.mark.parametrize("output", ["intensity", "directivity"])
@pytest.mark.parametrize("side", [None, CORE], ids=["default", "core"])
def test_n3_the_far_field_at_fixed_coefficients_by_every_node(wide_strip, side, output):
    # every node but the port's moves, the layer's too, at the solved
    # coefficients: the amplitude's node pass and, for the directivity, the
    # radiated power's power_flow node pass
    problem = gfem.Driven(fsc.F0, keep_fields=True)
    result = gfem.solve(wide_strip, problem, options=OPTIONS)
    x0 = jnp.asarray(wide_strip.mesh.nodes)

    def objective(x):
        moved, model = _at(result, wide_strip, x)
        ff = gfem.far_field(moved, "p1", THETA, PHI, model=model, side=side)
        return getattr(ff, output)[0]

    # the port's surface stays put: a lumped port's geometry is checked
    mesh = wide_strip.mesh
    feed = np.unique(np.asarray(mesh.triangles)[mesh.elements("feed")])
    directions = []
    for d in unit_directions(np.random.default_rng(5), x0.shape, 2):
        d[feed] = 0.0
        directions.append(d / np.linalg.norm(d))
    assert_gradient_matches(objective, x0, directions, step=NODE_STEP, rel=KERNEL)


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


def test_the_directivity_of_the_strip_under_a_morph(strip):
    morph = _morph(strip)
    problem = gfem.Driven(fsc.F0, keep_fields=True)

    def directivity(t):
        return _far(strip.replace(mesh=morph(t)), "p1", problem).directivity[0]

    t = jnp.zeros(morph.n_params)
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
