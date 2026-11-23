r"""Gradients of free-space outputs against finite differences.

The cases are the layer's TEM line of `pml_cases` (PEC y walls, PMC x
walls, a wave port at :math:`z = 0`, a PEC-backed layer at the far end) with
a lossy design slab in its air and a point dipole between the port and the
slab, at order 2. Each gradient matches a fourth-order central difference to
:math:`10^{-6}` relative (`finite_differences`):

- the Purcell factor by the slab's permittivity, for a real moment (the
  dipole's own solution is its adjoint: no adjoint solve) and for a circular
  one (one adjoint column);
- the beta factor :math:`|b|^2 / P` of the wave port, through the dipole's
  port wave;
- the emitted power by a `boundary_morph` of the slab's face, whose
  extension moves the dipole's element (the point load's transport term);
- the emitted power on an adaptive sweep, by the slab's permittivity, at a
  sample (the frozen basis's gradient is the full solve's there), for a
  real and a circular moment;
- the cross sections of a lossy cube under plane waves, and the corrected
  power flow of its total field, by the cube's permittivity. Both read
  background elements, the load and the incident interpolant only, so the
  design reaches them through the solved field alone, which a frozen-field
  check asserts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem._engine.probe import locate
from gfem._engine.system_operator import SystemOperator
from gfem._probe import layout
from gfem.shape import boundary_morph

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
#: The slab's permittivity and loss: its loss makes the emitted power depend
#: on it through the absorbed share as well.
EPS_SLAB = 2.25 * (1 - 0.05j)
#: Steps: the slab's permittivity, and node displacements in m.
EPS_STEP = 1e-2
NODE_STEP = 1e-9
#: Along the line's E (y), between the port and the slab, off the grid's faces.
POSITION = np.array([0.07, 0.13, 0.44]) * 1e-6
REAL = gfem.ElectricDipole("d", position=POSITION, moment=(0.0, 1e-28, 0.3e-28))
CIRCULAR = gfem.ElectricDipole(
    "d", position=POSITION, moment=np.array([0.0, 1.0, 1j]) * 1e-28 / np.sqrt(2)
)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    mesh = gfem.Mesh.from_gmsh(
        pc.slab_line(tmp_path_factory.mktemp("dipole") / "line.msh")
    )
    air = gfem.Material()
    n = len(mesh.elements("slab"))
    return gfem.Model(
        mesh,
        materials={
            "air": air,
            "pml": air,
            "slab": gfem.MaterialField(eps_r=np.full(n, EPS_SLAB)),
        },
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


def _with_slab(model, eps):
    return model.with_material("slab", gfem.MaterialField(eps_r=eps))


def _power(model, dipole, frequency=pc.F0, **kwargs):
    problem = gfem.Driven(frequency, excite=(), sources=dipole, **kwargs)
    return gfem.solve(model, problem, options=OPTIONS).emitted_power["d"]


@pytest.fixture
def adjoint_columns(monkeypatch):
    """The columns of every adjoint solve, as they run."""
    seen = []
    solve = SystemOperator.solve_adjoint

    def counted(self, n, omega, load):
        seen.append(load.shape[1] if np.ndim(load) == 2 else 1)
        return solve(self, n, omega, load)

    monkeypatch.setattr(SystemOperator, "solve_adjoint", counted)
    return seen


def _purcell(model, dipole):
    problem = gfem.Driven(pc.F0, excite=(), sources=dipole)
    result = gfem.solve(model, problem, options=OPTIONS)
    return gfem.purcell_factor(result, "d", model=model)[0]


@pytest.mark.parametrize(
    ("dipole", "solves"), [(REAL, 0), (CIRCULAR, 1)], ids=["real", "circular"]
)
def test_the_purcell_factor_by_the_slab_permittivity(
    line, adjoint_columns, dipole, solves
):
    eps = jnp.asarray(line.materials["slab"].eps_r)
    grad = jax.grad(lambda e: _purcell(_with_slab(line, e), dipole))(eps)
    # the real moment's adjoint is its own solution: no adjoint solve
    assert len(adjoint_columns) == solves
    directions = unit_directions(np.random.default_rng(0), eps.shape, 2)
    assert_gradient_matches(
        lambda e: _purcell(_with_slab(line, e), dipole),
        eps,
        directions,
        step=EPS_STEP,
        grad=grad,
    )


def test_the_beta_factor_by_the_slab_permittivity(line):
    def beta(eps):
        problem = gfem.Driven(pc.F0, excite=(), sources=REAL)
        result = gfem.solve(_with_slab(line, eps), problem, options=OPTIONS)
        wave = result.port_wave("p1", "d")[0]
        return jnp.abs(wave) ** 2 / result.emitted_power["d"][0]

    eps = jnp.asarray(line.materials["slab"].eps_r)
    directions = unit_directions(np.random.default_rng(1), eps.shape, 2)
    assert_gradient_matches(beta, eps, directions, step=EPS_STEP)


def test_emitted_power_by_a_morph_that_moves_the_dipole_element(line):
    morph = boundary_morph(line, "front", grows="slab")
    t = jnp.zeros(morph.n_params)
    moved = np.asarray(morph(t + 1e-8).nodes) - np.asarray(line.mesh.nodes)
    # the dipole's element moves
    located = locate(line.mesh, layout(_power_field(line)), POSITION[None, :])
    tet = line.mesh.tetrahedra[located.tetrahedra[0]]
    assert np.abs(moved[tet]).max() > 0
    directions = unit_directions(np.random.default_rng(2), t.shape, 2)
    assert_gradient_matches(
        lambda t: _power(line.replace(mesh=morph(t)), REAL)[0],
        t,
        directions,
        step=NODE_STEP,
    )


def _power_field(model):
    problem = gfem.Driven(pc.F0, excite=(), sources=REAL, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS).e_field(pc.F0)


#: An adaptive band around F0 and its frozen samples.
BAND = tuple(np.linspace(0.9, 1.1, 5) * pc.F0)
SAMPLES = (BAND[0], BAND[2], BAND[4])


@pytest.mark.parametrize("dipole", [REAL, CIRCULAR], ids=["real", "circular"])
def test_emitted_power_on_an_adaptive_sweep(line, dipole):
    # at a sample the full solution and its adjoint lie in the frozen basis
    # (a circular moment's adjoint load is solved at every sample), so the
    # gradient is the full solve's; between samples it follows the reduced
    # model's error, as S's does
    def power(eps):
        problem = gfem.Driven(
            BAND, excite=(), sources=dipole, sweep="adaptive", samples=SAMPLES
        )
        result = gfem.solve(_with_slab(line, eps), problem, options=OPTIONS)
        return result.emitted_power["d"][2]

    eps = jnp.asarray(line.materials["slab"].eps_r)
    directions = unit_directions(np.random.default_rng(3), eps.shape, 2)
    assert_gradient_matches(power, eps, directions, step=EPS_STEP)


def test_emitted_power_on_an_adaptive_sweep_by_a_morph(line):
    morph = boundary_morph(line, "front", grows="slab")

    def power(t):
        problem = gfem.Driven(
            BAND, excite=(), sources=REAL, sweep="adaptive", samples=SAMPLES
        )
        result = gfem.solve(line.replace(mesh=morph(t)), problem, options=OPTIONS)
        return result.emitted_power["d"][2]

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(4), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)


# --- cross sections and the corrected power flow -------------------------------

AIR = gfem.Material()
F = pc.F0 / 2
TE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")
TM = gfem.PlaneWave("tm", theta=0.4, phi=0.3, polarization="TM")
#: The cube's permittivity: the design is its real part.
EPS, LOSS = 3.0, 0.6j


def _volume(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("xsec") / "box.msh",
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


def _solve(model, eps):
    moved = _design(model, eps)
    problem = gfem.Driven(F, sources=(TE, TM), keep_fields=True)
    return moved, gfem.solve(moved, problem, options=OPTIONS)


def _cross_sections(model, result):
    te = gfem.cross_sections(result, "te", model=model)
    tm = gfem.cross_sections(result, "tm", model=model)
    scale = 1e12  # m² to um²
    return scale * (te.scattering[0] + 0.7 * te.absorption[0] + 0.4 * tm.extinction[0])


def _start(model):
    return jnp.full(len(model.mesh.elements("scat")), EPS)


def test_cross_sections_by_the_scatterer_permittivity(cube):
    model = cube

    def objective(eps):
        return _cross_sections(*_solve(model, eps))

    eps = _start(model)
    directions = unit_directions(np.random.default_rng(5), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_cross_sections_read_no_scatterer_material(cube):
    model = cube
    _, result = _solve(model, _start(model))
    frozen = jax.lax.stop_gradient(result)

    def objective(eps):
        return _cross_sections(_design(model, eps), frozen)

    gradient = jax.grad(objective)(_start(model))
    assert np.all(np.asarray(gradient) == 0)


def test_corrected_power_flow_by_the_absorber_permittivity(cube):
    model = cube
    def objective(eps):
        moved, result = _solve(model, eps)
        total = result.e_field(F, "tm")
        return -1e15 * gfem.power_flow(total, "scat", model=moved)

    eps = _start(model)
    directions = unit_directions(np.random.default_rng(7), eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)
