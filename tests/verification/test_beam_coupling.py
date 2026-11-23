r"""Beam coupling against reciprocity, a closed form and finite differences.

- T24: on a slab grating (`free_space_cases.grating_slab`: a silicon guide
  with five trenches in oxide, a wave port at one end, oxide PML above and
  below, absorbing ends, PEC slab walls a TE field meets), the coupling of
  the port-fed field into a z-invariant TE fiber (`free_space_cases.Fiber`,
  an `gfem.IncidentField`) equals the wave the port receives when that
  fiber lights the grating, ``port_wave("p1", "fiber")``, as complex
  numbers. Order 2: measured 1.4e-3 at oxide maxh 0.15 um (3.9e-2 at 0.25
  um, where the side's layer is one element below the PML, 7.9e-3 with
  absorbing faces instead of the PML, the first-order absorber's
  reflection at 10 degrees); gate `RECIPROCITY`.
- T25: the coupling read on two planes in the cladding equals the default
  side's: measured 9e-6; gate `PLANES`.
- A dipole's coupling into a `gfem.GaussianBeam` is, by reciprocity,
  :math:`-i\omega\,p\cdot w(r_0)/4` with :math:`w` the 1 W beam at the
  dipole: in the beam box (`free_space_cases.beam_box`, order 2,
  :math:`\lambda/3.5` in x) measured 1.7 % (TM) and 0.07 % (TE) of the
  coupled polarization's value; gate `DIPOLE`.
- T20 (second half): the 1 W the beam carries through the default side is
  read there to 0.999 W: the call does not warn.
- G12: :math:`|c|^2` by a `gfem.MaterialField`, for a `gfem.GaussianBeam`
  (in a block below the side, `free_space_cases.lid_box`) and for the
  `gfem.IncidentField` fiber (in the slab's trenches): the field-only
  kernel's chain.
- N5: :math:`|c|^2` of a `gfem.GaussianBeam` under a `boundary_morph` of
  the block's top face, which lies in the side's surface, so the layer's
  nodes move: the kernel's node pass with the beam's position-gradient
  term.

N6 (an `gfem.IncidentField` on the default side of a morphed model) needs
`boundary_morph` to hold the truncation layer; it lands with that hold.
"""

import math
import warnings

import free_space_cases as fsc
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem import _beam
from gfem._background import Background
from gfem.shape import boundary_morph

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
UM = fsc.UM
#: Forward against reciprocal, relative (measured 1.4e-3).
RECIPROCITY = 1e-2
#: One plane against another, relative (measured 9e-6).
PLANES = 2e-3
#: The dipole's coupling against reciprocity, relative to the coupled
#: polarization's value (measured 1.7e-2).
DIPOLE = 6e-2
EPS_STEP = 1e-2
NODE_STEP = 1e-9


def _fiber(mesh, name="fiber"):
    return gfem.IncidentField(
        name,
        fsc.Fiber(
            waist=1.5 * UM,
            focus=(1.6 * UM, 0.22 * UM),
            angle=math.radians(10),
            thickness=mesh.section_thickness,
        ),
    )


@pytest.fixture(scope="module")
def grating(tmp_path_factory):
    mesh = fsc.grating_slab(
        tmp_path_factory.mktemp("grating") / "grating.msh", h_ox=0.15
    )
    model = fsc.grating_model(mesh)
    fiber = _fiber(mesh)
    # both routes on one factorization: the port's column and the fiber's
    problem = gfem.Driven(fsc.GRATING_F0, sources=fiber, keep_fields=True)
    return model, fiber, gfem.solve(model, problem, options=OPTIONS)


def test_t24_the_coupling_is_the_beam_lit_port_wave(grating):
    model, fiber, result = grating
    forward = complex(gfem.beam_coupling(result, "p1", fiber, model=model)[0])
    reciprocal = complex(result.port_wave("p1", "fiber")[0])
    error = abs(forward - reciprocal) / abs(reciprocal)
    assert error < RECIPROCITY, (forward, reciprocal, error)


@pytest.mark.parametrize("height", [1.2, 1.4])
def test_t25_the_coupling_does_not_depend_on_the_plane(grating, height):
    model, fiber, result = grating
    default = complex(gfem.beam_coupling(result, "p1", fiber, model=model)[0])

    def below(c):
        return c[:, 1] < height * UM

    plane = complex(gfem.beam_coupling(result, "p1", fiber, model=model, side=below)[0])
    assert abs(plane - default) < PLANES * abs(default), (plane, default)


def _tilted(polarization, *, z=0.6):
    return gfem.GaussianBeam(
        "beam",
        waist=0.5 * UM,
        focus=(0.1 * UM, 0.0, z * UM),
        theta=math.pi - 0.2,
        phi=0.3,
        polarization=polarization,
    )


@pytest.fixture(scope="module")
def dipole(tmp_path_factory):
    model = fsc.beam_box(tmp_path_factory.mktemp("beam") / "box.msh")
    position = np.array([0.05, -0.03, 0.02]) * UM
    moment = np.array([1.0, 0.3, 0.2]) * 1e-30
    source = gfem.ElectricDipole("d", position=position, moment=moment)
    problem = gfem.Driven(fsc.F0, sources=source, keep_fields=True)
    return model, position, moment, gfem.solve(model, problem, options=OPTIONS)


def test_a_dipole_couples_into_the_beam_at_its_position(dipole):
    model, position, moment, result = dipole
    exact = {}
    got = {}
    for polarization in ("TM", "TE"):
        beam = _tilted(polarization)
        w = _beam.values(beam, Background(gfem.Material()), position[None], fsc.F0)
        exact[polarization] = -1j * fsc.OMEGA0 * (moment @ w[0]) / 4
        got[polarization] = complex(
            gfem.beam_coupling(result, "d", beam, model=model)[0]
        )
    scale = abs(exact["TM"])
    for polarization, value in got.items():
        error = abs(value - exact[polarization]) / scale
        assert error < DIPOLE, (polarization, value, exact[polarization], error)


def test_t20_an_unclipped_beam_carries_its_watt_through_the_default_side(dipole):
    model, _, _, result = dipole
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # less than 0.999 W would warn
        gfem.beam_coupling(result, "d", _tilted("TM"), model=model)
    with pytest.warns(UserWarning, match="clips it"):  # focused 0.6 um lower
        gfem.beam_coupling(result, "d", _tilted("TM", z=0.0), model=model)


# --- gradients --------------------------------------------------------------------

#: The lid box's open side: the plane z = 4/6 um.
LID_SIDE = ("below", "block")
LID_BEAM = gfem.GaussianBeam(
    "beam",
    waist=0.2 * UM,
    focus=(0.5 * UM, 0.5 * UM, fsc.LID * UM),
    theta=math.pi - 0.2,
    phi=0.3,
    polarization="TM",
)
LID_PROBLEM = gfem.Driven(
    fsc.F0,
    sources=gfem.ElectricDipole("d", position=fsc.LID_DIPOLE, moment=(1e-30, 0, 5e-31)),
    keep_fields=True,
)


def _lid_power(model):
    result = gfem.solve(model, LID_PROBLEM, options=OPTIONS)
    c = gfem.beam_coupling(result, "d", LID_BEAM, model=model, side=LID_SIDE)
    return jnp.abs(c[0]) ** 2


@pytest.mark.filterwarnings("ignore:The beam 'beam' carries")
def test_g12_coupling_into_a_gaussian_beam_by_a_design(tmp_path):
    model = fsc.lid_box(tmp_path / "lid.msh", field=True)

    def power(eps):
        return _lid_power(model.with_material("block", gfem.MaterialField(eps_r=eps)))

    eps = jnp.asarray(model.materials["block"].eps_r)
    directions = unit_directions(np.random.default_rng(0), eps.shape, 2)
    assert_gradient_matches(power, eps, directions, step=EPS_STEP)


def test_g12_coupling_into_an_incident_field_fiber_by_a_design(tmp_path):
    mesh = fsc.grating_slab(tmp_path / "coarse.msh", h_si=0.12, h_ox=0.4)
    n = len(mesh.elements("trench"))
    model = fsc.grating_model(mesh, trench=gfem.MaterialField(eps_r=np.full(n, 2.0)))
    fiber = _fiber(mesh)
    problem = gfem.Driven(fsc.GRATING_F0, keep_fields=True)

    def power(eps):
        moved = model.with_material("trench", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(moved, problem, options=OPTIONS)
        return jnp.abs(gfem.beam_coupling(result, "p1", fiber, model=moved)[0]) ** 2

    eps = jnp.asarray(model.materials["trench"].eps_r)
    directions = unit_directions(np.random.default_rng(1), eps.shape, 2)
    assert_gradient_matches(power, eps, directions, step=EPS_STEP)


@pytest.mark.filterwarnings("ignore:The beam 'beam' carries")
def test_n5_coupling_into_a_gaussian_beam_under_a_morph(tmp_path):
    model = fsc.lid_box(tmp_path / "lid.msh")
    morph = boundary_morph(model, "lid", grows="block")
    assert morph.n_params > 0
    # the lid lies in the side's surface: the layer above it moves
    moved = np.asarray(morph(jnp.ones(morph.n_params) * 1e-8).nodes)
    shift = np.abs(moved - np.asarray(model.mesh.nodes)).max(axis=1)
    tets = np.asarray(model.mesh.tetrahedra)[model.mesh.elements("above")]
    assert shift[np.unique(tets)].max() > 0

    def power(t):
        return _lid_power(model.replace(mesh=morph(t)))

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(2), t.shape, 2)
    assert_gradient_matches(power, t, directions, step=NODE_STEP)
