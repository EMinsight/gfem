"""`gfem.power_flow` differentiated through its kept field, against differences.

The power through a surface is a real quadratic form of the field, read
on the element layer just outside the side, so its gradient in a design
that stays out of that layer reaches the design through the field alone.
The designs here are a random lossy permittivity and conductivity per
element of a "core" section (`gradient_cases.core_design`) that the side
holds, or that lies away from its surface:

- the two-layer line of `gradient_cases.wave_mesh` with lumped ports and
  absorbing side walls (direct and BDDC solves, and under `jax.jit`), and
  with wave ports;
- the C1 line, periodic along x (`periodic_cases`), with the side crossing
  the periodic seam;
- the network identity: with the core inside a side that holds port 1,
  and nothing lossy outside it, the power leaving the side is the power
  port 2 takes, |S21|^2, and so are their gradients.

And by the node coordinates, on a dipole below a lossy block in an air box
wrapped in a perfectly matched layer (`free_space_cases.lid_box`, order 2):

- N1: the power leaving the block, whose element layer moves with a
  `boundary_morph` of its top face;
- N2: the power the layer absorbs (a side of layer elements, whose element
  layer is the truncation layer), and the power leaving the physical
  region (whose element layer is in the perfectly matched layer, whose
  stretch is read at the moved quadrature points).

Each at two levels: the measurement alone, at fixed field coefficients, by
a random move of every node (fourth-order differences to
:math:`10^{-8}`), and the chain through the solve and the morph
(:math:`10^{-6}`).
"""

import dataclasses

import free_space_cases as fsc
import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import core_design, tight, wave_line, wave_mesh

import gfem
from gfem.shape import boundary_morph

FREQUENCY = 12e9
#: Fourth-order differences on a relative permittivity of a few.
STEP = 1e-2


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return wave_mesh(tmp_path_factory.mktemp("line"))


def lumped_line(mesh, *, absorbing=True):
    """The wave-port line's mesh with lumped ports (and absorbing y walls)."""
    walls = [gfem.Absorbing("side")] if absorbing else []
    return gfem.Model(
        mesh,
        materials={
            "lo": gfem.Material(eps_r=4.0),
            "hi": gfem.Material(eps_r=2.0),
            "core": gfem.Material(eps_r=3.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *walls],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _designed(model, values, region="core"):
    def build(eps):
        return model.with_material(
            region, gfem.MaterialField(**{**values, "eps_r": eps})
        )

    return build


def _power(build, side, options, frequency=FREQUENCY):
    """The power leaving ``side`` for the design ``eps``, W."""
    problem = gfem.Driven(frequency=frequency, excite="p1", keep_fields=True)

    def objective(eps):
        m = build(eps)
        result = gfem.solve(m, problem, options=options)
        return gfem.power_flow(result.e_field(frequency, "p1"), side, model=m)

    return objective


@pytest.mark.parametrize("solver", ["direct", "bddc_gmres"])
def test_power_absorbed_in_the_core_on_a_lumped_line(mesh, solver):
    values, rng = core_design(mesh, seed=3)
    objective = _power(_designed(lumped_line(mesh), values), "core", tight(solver))
    eps = values["eps_r"]
    assert float(objective(eps)) < -1e-3  # the core absorbs: power enters it
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=STEP)


def test_power_flow_gradient_under_jit(mesh):
    values, rng = core_design(mesh, seed=5)
    objective = _power(_designed(lumped_line(mesh), values), "core", tight("direct"))
    eps = values["eps_r"]
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(
        jax.jit(objective),
        eps,
        directions,
        step=STEP,
        grad=jax.jit(jax.grad(objective))(eps),
    )


def test_power_absorbed_in_the_core_between_wave_ports(mesh):
    values, rng = core_design(mesh, seed=4)
    objective = _power(
        _designed(wave_line(mesh, "lossy"), values), "core", tight("direct")
    )
    eps = values["eps_r"]
    assert float(objective(eps)) < -1e-3
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=STEP)


# --- the periodic line ---------------------------------------------------------------

#: C1 on 16 cells along x: "core" between x = 3.75 and 5 mm (between the
#: port sheets), "seam" the two cells at each end of the cell (across the
#: periodic pair).
CORE = (3.75, 5.0)
SEAM = 1.25


@pytest.fixture(scope="module")
def periodic_line(tmp_path_factory):
    def volume(c):
        if CORE[0] < c[0] < CORE[1]:
            return "core"
        if c[0] < SEAM or c[0] > pc.C1_A - SEAM:
            return "seam"
        return "cell"

    edges = (
        np.linspace(0.0, pc.C1_A, 17),
        np.linspace(0.0, pc.C1_W, 2),
        np.linspace(0.0, pc.C1_H, 2),
    )
    cell = sm.periodic_cell(
        edges, volume, sheets=(("sheet1", pc.C1_X1), ("sheet2", pc.C1_X2))
    )
    path = sm.write_msh22(tmp_path_factory.mktemp("c1") / "c1.msh", cell)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    lossy = gfem.Material(eps_r=pc.C1_EPS, tan_delta=pc.C1_TAN_DELTA)
    return gfem.Model(
        mesh,
        materials={"cell": lossy, "seam": lossy, "core": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[
            gfem.LumpedPort("p1", "sheet1", "+z", impedance=50.0),
            gfem.LumpedPort("p2", "sheet2", "+z", impedance=50.0),
        ],
    )


def test_power_absorbed_across_the_periodic_seam(periodic_line):
    values, rng = core_design(periodic_line.mesh, seed=6)
    objective = _power(
        _designed(periodic_line, values),
        "seam",
        tight("direct"),
        frequency=pc.C1_FREQUENCY,
    )
    eps = values["eps_r"]
    assert float(objective(eps)) < -1e-4
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=STEP)


# --- the network identity ------------------------------------------------------------

#: A TEM parallel-plate line, 8 cells along x, the cells between x = 3 and
#: 5 mm the "core"; the design is one lossy permittivity per core cell
#: (x slab), so the line scatters no field that is not the TEM mode, which
#: the wave ports absorb exactly (tests/unit/test_power_flow.py).
#: The plane between the two slabs, mm.
SLAB_PLANE = 4.0


@pytest.fixture(scope="module")
def tem_line(tmp_path_factory):
    path = tmp_path_factory.mktemp("tem") / "line.msh"
    sm.write_msh22(path, sm.strip_line(8.0, 2.0, 1.0, (8, 1, 1), core=(3.0, 5.0)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0), "core": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_the_power_past_the_core_has_the_gradient_of_s21(tem_line):
    # lossless leads and PMC y walls: what leaves a side holding port 1 and
    # the lossy core reaches port 2, which absorbs its mode: |S21|^2 of 1 W
    # incident (order 3, as in the unit test, for an exact TEM mode)
    mesh = tem_line.mesh
    x = mesh.nodes[mesh.tetrahedra[mesh.elements("core")]].mean(axis=1)[:, 0]
    slab = jnp.asarray((x > SLAB_PLANE * 1e-3).astype(int))
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    problem = gfem.Driven(frequency=FREQUENCY, excite="p1", keep_fields=True)

    def both(p):
        m = tem_line.with_material("core", gfem.MaterialField(eps_r=p[slab]))
        result = gfem.solve(m, problem, options=options)
        power = gfem.power_flow(result.e_field(FREQUENCY, "p1"), _below(6.0), model=m)
        return power, jnp.abs(result.s.values[0, 1, 0]) ** 2

    p = jnp.asarray([5.0 - 0.3j, 2.0 - 0.05j])
    power, s21 = both(p)
    assert 0.1 < float(s21) < 0.99
    assert float(power) == pytest.approx(float(s21), rel=1e-9)
    from_power = jax.grad(lambda q: both(q)[0])(p)
    from_s = jax.grad(lambda q: both(q)[1])(p)
    scale = float(jnp.linalg.norm(from_s))
    assert scale > 0
    assert float(jnp.linalg.norm(from_power - from_s)) <= 1e-8 * scale


def _below(x_mm):
    return lambda c: c[:, 0] < x_mm * 1e-3


# --- the node coordinates ------------------------------------------------------------

#: A side of the block, of the layer, and of the physical region.
BLOCK = ("block",)
LAYER = ("pml",)
PHYSICAL = ("below", "above", "block")
#: The measurement alone, relative (fourth-order differences of a function
#: with no solve in it).
KERNEL = 1e-8
#: A dipole in the corner cell below the block, which shares no edge with
#: it or with the truncation layer, off the faces of its tetrahedra, m.
CORNER = np.array([0.24, 0.27, 0.22]) * fsc.UM
#: Node steps, m (edges are 1/6 um).
NODE_STEP = 1e-9


@pytest.fixture(scope="module")
def lid(tmp_path_factory):
    model = fsc.lid_box(tmp_path_factory.mktemp("lid") / "lid.msh")
    model = model.with_material("block", gfem.Material(eps_r=2.0, tan_delta=0.2))
    dipole = gfem.ElectricDipole("d", position=CORNER, moment=(0, 0, 1e-30))
    problem = gfem.Driven(fsc.F0, sources=dipole, keep_fields=True)
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    return model, problem, options


def _at(model, x):
    """``model`` with its node coordinates ``x``."""
    mesh = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(model.mesh), [x])
    return model.replace(mesh=mesh)


@pytest.mark.parametrize(
    "side", [BLOCK, LAYER, PHYSICAL], ids=["n1", "n2-pml", "n2-adjacent"]
)
def test_the_power_at_fixed_coefficients_by_every_node(lid, side):
    model, problem, options = lid
    field = gfem.solve(model, problem, options=options).e_field(fsc.F0, "d")
    x0 = jnp.asarray(model.mesh.nodes)

    def power(x):
        moved = _at(model, x)
        at = dataclasses.replace(field, mesh=moved.mesh)
        return gfem.power_flow(at, side, model=moved)

    assert abs(float(power(x0))) > 0
    rng = np.random.default_rng(7)
    directions = unit_directions(rng, x0.shape, 2)
    assert_gradient_matches(power, x0, directions, step=NODE_STEP, rel=KERNEL)


@pytest.mark.parametrize(
    "side", [BLOCK, LAYER, PHYSICAL], ids=["n1", "n2-pml", "n2-adjacent"]
)
def test_the_power_by_a_morph_of_the_block(lid, side):
    model, problem, options = lid
    morph = boundary_morph(model, "lid", grows="block")

    def power(t):
        moved = model.replace(mesh=morph(t))
        result = gfem.solve(moved, problem, options=options)
        return gfem.power_flow(result.e_field(fsc.F0, "d"), side, model=moved)

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(8), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)
