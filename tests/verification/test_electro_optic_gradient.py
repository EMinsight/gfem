r"""The electro-optic shift's derivative with respect to the electrostatic nodes.

An x-cut lithium niobate ridge on a film of the crystal, in oxide below and
air above, between a signal and a ground electrode on the film (both
excluded from the solves). The electrostatic model names the crystal's
interfaces: the ridge outline is held and the film's faces slide, so a
move of the signal electrode's inner wall (the gap) deforms the static
mesh around the crystal without changing its shape.

The static field is read at fixed points of the optical model's
tetrahedra. Its derivative with respect to the static nodes at those
points is the Eulerian one: the change of the coefficients through the
electrostatic solve, plus the change of :math:`-\nabla V` at a fixed point
while the element moves under it.

- With the coefficients held, the sampler alone against a difference at a
  :math:`10^{-12}` m step on every node, at orders 1 to 3.
- The whole gap chain (morph, solve, sample) with the optical model on the
  static model's mesh, against a fourth-order difference at 2 nm. The
  points then stay in their tetrahedra, so the objective is smooth.
- Across meshes, an optical window whose crystal is the ridge alone: the
  held ridge's points cross no static face.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc
from finite_differences import assert_gradient_matches

import gfem
from gfem.shape import boundary_morph

FREQUENCY = scipy.constants.c / 1.55e-6
N_O, N_E, N_OX = 2.2111, 2.1376, 1.444
LN_OPTICAL = (N_O**2, N_O**2, N_E**2)
#: Static permittivity of the crystal, principal values along crystal x, y, z.
LN_STATIC = (44.0, 44.0, 27.9)
#: The 3m tensor of lithium niobate, crystal frame, m/V (rows xx yy zz yz xz xy).
R13, R22, R33, R51 = 9.6e-12, 6.8e-12, 30.9e-12, 32.6e-12
R_LN = (
    (0.0, -R22, R13),
    (0.0, R22, R13),
    (0.0, 0.0, R33),
    (0.0, R51, 0.0),
    (R51, 0.0, 0.0),
    (-R22, 0.0, 0.0),
)
#: x-cut, propagation along global z: crystal x up (global y), crystal y along
#: the guide (global z), crystal z lateral (global x).
X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
#: One slab thickness for both meshes: the window's elements lie in the section's.
THICKNESS = 0.2e-6
HALF_GAP, WIDTH = 1.5, 0.9  # um
RIDGE = (-WIDTH / 2, 0.3, WIDTH / 2, 0.6, "ridge", 0.1)
SECTION = [
    (-4.0, 0.3, 4.0, 2.0, "air", None),
    (-4.0, -1.5, 4.0, 0.0, "box", None),
    (-4.0, 0.0, 4.0, 0.3, "film", 0.15),
    RIDGE,
    (HALF_GAP, 0.3, HALF_GAP + 1.5, 0.5, "signal", 0.15),
    (-HALF_GAP - 1.5, 0.3, -HALF_GAP, 0.5, "ground", 0.15),
]
WINDOW = [
    (-1.4, 0.3, 1.4, 1.4, "air", None),
    (-1.4, -1.0, 1.4, 0.0, "box", None),
    (-1.4, 0.0, 1.4, 0.3, "film", 0.15),
    RIDGE,
]
STEP = 2e-9  # m of wall motion
PROBLEM = gfem.Electrostatic(
    terminals={"s": ["signal_b", "signal_s"]}, ground=["ground_b", "ground_s"]
)


def _outline(x0, y0, x1, y1):
    def on(p):
        x, y = p[0], p[1]
        along_x = abs(x - x0) < 1e-9 or abs(x - x1) < 1e-9
        along_y = abs(y - y0) < 1e-9 or abs(y - y1) < 1e-9
        inside_x = x0 - 1e-9 <= x <= x1 + 1e-9
        inside_y = y0 - 1e-9 <= y <= y1 + 1e-9
        return (along_x and inside_y) or (along_y and inside_x)

    return on


def _electrode(name, x0, x1):
    return [
        (lambda p: abs(p[1] - 0.3) < 1e-9 and x0 < p[0] < x1, f"{name}_b"),
        (_outline(x0, 0.3, x1, 0.5), f"{name}_s"),
    ]


#: First match names an edge: the electrodes' feet, then the ridge outline,
#: then the film's other faces.
EDGES = [
    *_electrode("signal", HALF_GAP, HALF_GAP + 1.5),
    *_electrode("ground", -HALF_GAP - 1.5, -HALF_GAP),
    (_outline(*RIDGE[:4]), "ridge_s"),
    (lambda p: abs(p[1] - 0.3) < 1e-9 or abs(p[1]) < 1e-9, "film_s"),
]


def _ln(eps, tensor=R_LN):
    return gfem.Material(eps_r=eps, axes=X_CUT, electro_optic_tensor=tensor)


def _options(order):
    return gfem.SolverOptions(order=order, linear_solver="direct")


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    path = tmp_path_factory.mktemp("electro_optic_gradient") / "section.msh"
    return tc.section(path, SECTION, 0.5, EDGES, thickness=THICKNESS)


@pytest.fixture(scope="module")
def static(section):
    crystal = gfem.Material(eps_r=LN_STATIC, axes=X_CUT)
    return gfem.Model(
        section,
        materials={
            "air": gfem.Material(),
            "box": gfem.Material(eps_r=3.9),
            "film": crystal,
            "ridge": crystal,
            "signal": gfem.Material(),
            "ground": gfem.Material(),
        },
        excluded={"em": ["signal", "ground"]},
    )


@pytest.fixture(scope="module")
def optical(section):
    """The optical model on the static model's mesh; the crystal: film and ridge."""
    crystal = _ln(LN_OPTICAL)
    return gfem.Model(
        section,
        materials={
            "air": gfem.Material(),
            "box": gfem.Material(eps_r=N_OX**2),
            "film": crystal,
            "ridge": crystal,
            "signal": gfem.Material(),
            "ground": gfem.Material(),
        },
        ports=[gfem.WavePort("p", "front")],
        excluded={"em": ["signal", "ground"]},
    )


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    """An optical window on its own mesh: the crystal is the ridge alone."""
    path = tmp_path_factory.mktemp("electro_optic_window") / "window.msh"
    mesh = tc.section(path, WINDOW, 0.4, [], thickness=THICKNESS)
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "box": gfem.Material(eps_r=N_OX**2),
            "film": gfem.Material(eps_r=LN_OPTICAL, axes=X_CUT),
            "ridge": _ln(LN_OPTICAL),
        },
        ports=[gfem.WavePort("p", "front")],
    )


@pytest.fixture(scope="module")
def morph(static):
    """The signal electrode's inner wall along its normal, the ridge held."""

    def wall(x):
        return np.abs(x[:, 0] - HALF_GAP * 1e-6) < 1e-12

    return boundary_morph(static, "signal_s", where=wall, fixed=["ridge_s"])


def _shift(model, potential):
    shift = gfem.electro_optic_shift(model, potential, port="p", frequency=FREQUENCY)
    return shift.delta_n_eff_per_volt


def _gap_objective(static, morph, model, order=2, *, frozen=False):
    """The shift per volt with the signal wall moved by ``s`` m."""

    def objective(s):
        moved = static.replace(mesh=morph(s * jnp.ones(morph.n_params)))
        potential = gfem.solve(moved, PROBLEM, options=_options(order)).potential["s"]
        if frozen:
            # read the field at the moved coefficients, with the rows of the
            # mesh it was solved on held: the coefficients' path alone
            mesh = jax.lax.stop_gradient(potential.mesh)
            potential = dataclasses.replace(potential, mesh=mesh)
        return _shift(model, potential)

    return objective


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_sampler_alone_matches_a_difference_on_every_node(static, optical, order):
    potential = gfem.solve(static, PROBLEM, options=_options(order)).potential["s"]
    structure = jax.tree_util.tree_structure(potential.mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(structure, [nodes])
        return _shift(optical, dataclasses.replace(potential, mesh=mesh))

    rng = np.random.default_rng(order)
    nodes = jnp.asarray(potential.mesh.nodes)
    # every node moves by about the step (a unit RMS per coordinate)
    directions = [rng.standard_normal(nodes.shape) for _ in range(2)]
    assert_gradient_matches(objective, nodes, directions, step=1e-12)


def test_the_gap_chain_on_one_mesh_matches_a_difference(static, optical, morph):
    objective = _gap_objective(static, morph, optical)
    grad = jax.grad(objective)(0.0)
    assert_gradient_matches(objective, 0.0, [1.0], step=STEP, grad=grad)
    # the element's motion under the fixed points carries a large share: the
    # coefficients' path alone misses it
    rows_held = jax.grad(_gap_objective(static, morph, optical, frozen=True))(0.0)
    assert abs(float(rows_held) - float(grad)) > 0.1 * abs(float(grad))


def test_the_held_ridge_through_a_window_matches_a_difference(static, window, morph):
    objective = _gap_objective(static, morph, window)
    assert_gradient_matches(objective, 0.0, [1.0], step=STEP)


def test_jit_and_vmap_of_the_gap_gradient_match_a_loop(static, optical, morph):
    grad = jax.grad(_gap_objective(static, morph, optical))
    walls = jnp.array([0.0, 1e-9])
    looped = np.array([float(grad(s)) for s in walls])
    np.testing.assert_allclose(np.asarray(jax.vmap(grad)(walls)), looped, rtol=1e-9)
    np.testing.assert_allclose(float(jax.jit(grad)(walls[0])), looped[0], rtol=1e-9)
