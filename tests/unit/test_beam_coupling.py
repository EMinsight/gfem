r"""Beam coupling: its kernel's node derivative, its warnings and its refusals.

The kernel (N7): the coupling of fixed field reference values into a
beam over a few tetrahedra, as a function of their corners, against
fourth-order central differences to :math:`10^{-8}`, for the two test
fields: a `gfem.GaussianBeam` read in closed form at the moving
quadrature points (its position-gradient term), and a frozen interpolant
carried with the nodes. The 1 W normalization is in the same function, so
its derivative is checked with them.

The solved cases are coarse: a dipole in a box of air in a perfectly
matched layer (order 2, 4 cells per um). The closed-form and reciprocity
gates live in tests/verification/test_beam_coupling.py.
"""

import math
import warnings

import free_space_cases as fsc
import jax.numpy as jnp
import numpy as np
import pml_cases
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem._engine import beam_coupling as bc
from gfem._engine import far_field as ff

ORDER2 = gfem.SolverOptions(order=2, linear_solver="direct")
UM = fsc.UM
INSIDE = np.array([0.47, 0.53, 0.56]) * UM
#: A beam coming down onto the box's top face, 0.2 rad off the normal.
BEAM = gfem.GaussianBeam(
    "fiber",
    waist=0.3 * UM,
    focus=(0.5 * UM, 0.5 * UM, 1.0 * UM),
    theta=math.pi - 0.2,
    phi=0.3,
    polarization="TM",
)


# --- the kernel ---------------------------------------------------------------------


def _corners(rng):
    """Six jittered tetrahedra of a 0.2 um cube near the beam's focus, m."""
    cube = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    # the Kuhn split of the cube along its main diagonal
    tets = [
        (0, 1, 3, 7),
        (0, 1, 5, 7),
        (0, 2, 3, 7),
        (0, 2, 6, 7),
        (0, 4, 5, 7),
        (0, 4, 6, 7),
    ]
    points = 0.2 * (cube + 0.05 * rng.normal(size=cube.shape)) + [0.4, 0.4, 0.8]
    return points[np.array(tets)] * UM


def _layer():
    return bc.CouplingLayer(
        chi=np.zeros(1),
        tetrahedra=np.zeros(0, int),
        eps_r=1.0,
        mu_r=1.0,
        material=gfem.Material(),
    )


@pytest.mark.parametrize("frozen", [False, True], ids=["gaussian", "frozen"])
def test_n7_the_kernel_follows_its_layer_nodes(frozen):
    rng = np.random.default_rng(4)
    corners = _corners(rng)
    bary, weights = ff.rule(2)
    shape = (len(corners), len(bary), 3)

    def field(scale):
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    e_ref, c_ref = field(1.0), field(1e7)
    # the side's indicator: vertex 0 of every tetrahedron inside, the rest out
    chi = np.tile([1.0, 0.0, 0.0, 0.0], (len(corners), 1))
    omega = 2 * math.pi * fsc.F0
    if frozen:
        test = bc.frozen_test(field(1e3), field(1e10))
    else:
        test = bc.gaussian_test(BEAM, _layer(), fsc.F0)

    def coupling(x):
        r, flux = bc.reaction(
            x.reshape(corners.shape),
            jnp.asarray(e_ref),
            jnp.asarray(c_ref),
            chi=chi,
            bary=bary,
            weights=weights,
            test=test,
        )
        c, _ = bc.coupling(r, flux, omega, 1.0)
        return jnp.real(c) + 0.3 * jnp.imag(c)

    x = jnp.asarray(corners.reshape(-1))
    directions = unit_directions(rng, x.shape, 3)
    assert_gradient_matches(coupling, x, directions, step=2e-11, rel=1e-8)


# --- solved cases --------------------------------------------------------------------

#: A dipole in the beam box's air, off the faces of its grid, m.
POSITION = np.array([0.05, -0.03, -0.2]) * UM


@pytest.fixture(scope="module")
def box_model(tmp_path_factory):
    return fsc.beam_box(tmp_path_factory.mktemp("coupling") / "box.msh", cells=4)


def _dipole(model, position=POSITION):
    source = gfem.ElectricDipole("d", position=position, moment=(1e-30, 0, 0))
    problem = gfem.Driven(fsc.F0, sources=source, keep_fields=True)
    return gfem.solve(model, problem, options=ORDER2)


@pytest.fixture(scope="module")
def dipole(box_model):
    return box_model, _dipole(box_model)


def _beam(name="fiber", *, waist=0.5, z=0.6):
    """A beam coming down 0.2 rad off the normal, focused at height ``z`` um."""
    return gfem.GaussianBeam(
        name,
        waist=waist * UM,
        focus=(0.1 * UM, 0.0, z * UM),
        theta=math.pi - 0.2,
        phi=0.3,
        polarization="TM",
    )


def _below(z):
    def side(c):
        return c[:, 2] < z * UM

    return side


def test_a_coupling_per_kept_frequency(dipole):
    model, result = dipole
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # held whole: no clipping warning
        c = gfem.beam_coupling(result, "d", _beam(), model=model)
    assert c.shape == (1,)
    assert np.iscomplexobj(np.asarray(c))
    assert abs(complex(c[0])) > 0


def test_a_beam_wider_than_the_side_is_clipped(dipole):
    model, result = dipole
    # focused 0.6 um below the top face, the beam is wider there
    wide = _beam("wide", z=0.0)
    with pytest.warns(UserWarning, match=r"'wide' carries .* W through.*clips it"):
        gfem.beam_coupling(result, "d", wide, model=model)


def test_the_beam_must_travel_toward_the_side(dipole):
    model, result = dipole

    def above(c):
        return c[:, 2] > -0.35 * UM

    # the beam comes down, so it leaves the region above z = -0.35 um
    with pytest.raises(ValueError, match="carries no power into the side"):
        gfem.beam_coupling(result, "d", _beam(), model=model, side=above)


def test_a_closed_side_brings_no_net_power(dipole):
    model, result = dipole

    def core(c):
        return (np.abs(c) < np.array([1.0, 1.0, 0.45]) * UM).all(axis=1)

    with pytest.raises(ValueError, match="not a closed surface"):
        gfem.beam_coupling(result, "d", _beam(), model=model, side=core)


def test_a_side_running_along_the_truncation_under_the_beam(dipole):
    model, result = dipole

    def patch(c):
        # below the top row of cells, and a patch of it reaching the layer
        top = (np.abs(c[:, :2]) < 0.5 * UM).all(axis=1) & (c[:, 2] < 0.6 * UM)
        return (c[:, 2] < 0.3 * UM) | top

    with pytest.raises(ValueError, match="meets the truncation"):
        gfem.beam_coupling(result, "d", _beam(), model=model, side=patch)


def test_the_layer_must_be_the_beam_medium(tmp_path):
    def volume(c):
        if not ((c > 0) & (c < 1)).all():
            return "pml"
        return "glass" if c[0] < 0.25 else "air"

    path = pml_cases.box(
        tmp_path / "glass.msh", pml_cases.SIDES, thickness=0.5, cells=4, volume=volume
    )
    air = gfem.Material()
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": air, "pml": air, "glass": gfem.Material(eps_r=2.25)},
        pml=gfem.PML("pml"),
    )
    result = _dipole(model, INSIDE)
    beam = _beam(z=1.0)
    with pytest.raises(ValueError, match="one constant, isotropic, lossless medium"):
        gfem.beam_coupling(result, "d", beam, model=model, side=_below(0.8))


def test_a_pec_wall_the_beam_does_not_satisfy(tmp_path):
    sides = ("-x", "-y", "+y", "-z", "+z")  # +x ends on a wall
    path = pml_cases.box(tmp_path / "wall.msh", sides, thickness=0.5, cells=4)
    air = gfem.Material()
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": air, "pml": air},
        boundaries=[gfem.PEC("outer")],
        pml=gfem.PML("pml"),
    )
    result = _dipole(model, INSIDE)
    near = gfem.GaussianBeam(
        "near",
        waist=0.3 * UM,
        focus=(0.8 * UM, 0.5 * UM, 1.0 * UM),
        theta=math.pi,
        polarization="TM",
    )
    with pytest.raises(ValueError, match=r"PEC wall.*tangential electric field"):
        gfem.beam_coupling(result, "d", near, model=model)


def _plane_wave_down(points, frequency):
    k = 2 * math.pi * frequency / fsc.C0
    e = np.zeros((len(points), 3), complex)
    e[:, 0] = np.exp(1j * k * points[:, 2])
    return e


def test_an_incident_field_beam_on_a_moved_layer(box_model):
    mesh = box_model.mesh
    nodes = np.asarray(mesh.nodes)
    # the air's own nodes, off the perfectly matched layer's faces
    half = np.array(fsc.BEAM_BOX) / 2 * UM
    inner = (np.abs(nodes) < half - 1e-9).all(axis=1)
    moved = nodes + np.where(inner[:, None], [0.0, 0.0, 1e-9], 0.0)
    model = box_model.replace(mesh=gfem.Mesh(jnp.asarray(moved), mesh._topology))
    result = _dipole(model)
    beam = gfem.IncidentField("down", _plane_wave_down)
    with pytest.raises(ValueError, match=r"'down' as interpolated.*have moved"):
        gfem.beam_coupling(result, "d", beam, model=model)


def test_wrong_arguments(dipole):
    model, result = dipole
    wave = gfem.PlaneWave("w", theta=math.pi, polarization="TE")
    with pytest.raises(TypeError, match=r"GaussianBeam or a gfem\.IncidentField"):
        gfem.beam_coupling(result, "d", wave, model=model)
    with pytest.raises(TypeError, match="side must be"):
        gfem.beam_coupling(result, "d", _beam(), model=model, side=3)
    with pytest.raises(KeyError, match="neither an excited port nor a source"):
        gfem.beam_coupling(result, "nope", _beam(), model=model)
