"""`gfem.electro_optic_shift`: the first-order Pockels shift of a wave port's mode.

An L-shaped lithium niobate channel (a core and a foot to one side) in
oxide on a small cross-section slab, between a top and a bottom electrode
surface: the static field has a vertical and a lateral part, and the full
3m tensor of the crystal acts through its diagonal and its off-diagonal
entries. The shift is first order in the potential from one mode solve, so
it is exactly linear in the potential and in the tensor; the re-solve
checks live in the verification tier.
"""

import dataclasses
import types

import electro_optic_cases as eo
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc
from finite_differences import assert_gradient_matches, unit_directions
from strip2d import write_strip

import gfem
from gfem.mesh import Mesh

FREQUENCY = eo.FREQUENCY
N_OX = eo.N_OX
LN_OPTICAL, LN_STATIC = eo.LN_OPTICAL, eo.LN_STATIC
R22, R_LN, X_CUT = eo.R22, eo.R_LN, eo.X_CUT
#: A quarter turn about the guide: (x, y) -> (-y, x).
TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MAXH = 0.5
PLATES, GAP = eo.PLATES, eo.GAP


def _crystal(axes, eps, tensor=R_LN):
    return gfem.Material(eps_r=eps, axes=axes, electro_optic_tensor=tensor)


def _electrostatic(mesh, crystal_eps=LN_STATIC, axes=X_CUT, order=2):
    crystal = gfem.Material(eps_r=crystal_eps, axes=axes)
    regions = mesh._region_names().values()
    static = gfem.Model(
        mesh,
        materials={
            r: crystal if r in ("core", "foot") else gfem.Material(eps_r=3.9)
            for r in regions
        },
    )
    problem = gfem.Electrostatic(terminals={"t": ["top"]}, ground=["bottom"])
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(static, problem, options=options)


def _potential(mesh, **kw):
    return _electrostatic(mesh, **kw).potential["t"]


def _uniform(mesh):
    """A potential with the exact uniform field -1/GAP along y: one medium."""
    return _potential(mesh, crystal_eps=3.9, order=1)


def _optical(mesh, crystal, **clad):
    return gfem.Model(
        mesh,
        materials={
            "clad": gfem.Material(eps_r=N_OX**2, **clad),
            "core": crystal,
            "foot": crystal,
        },
        ports=[gfem.WavePort("p", "front")],
    )


def _shift(model, potential, **kw):
    return gfem.electro_optic_shift(
        model, potential, port="p", frequency=FREQUENCY, **kw
    )


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("electro_optic") / "lchannel.msh"
    return eo.l_core(path, MAXH, 0.15)


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    """The same section, coarse: the gradient checks solve the mode many times."""
    path = tmp_path_factory.mktemp("electro_optic") / "coarse.msh"
    return eo.l_core(path, 1.2, 0.4)


@pytest.fixture(scope="module")
def electrostatic(mesh):
    return _electrostatic(mesh)


@pytest.fixture(scope="module")
def potential(electrostatic):
    return electrostatic.potential["t"]


@pytest.fixture(scope="module")
def uniform(mesh):
    return _uniform(mesh)


@pytest.fixture(scope="module")
def model(mesh):
    return _optical(mesh, _crystal(X_CUT, LN_OPTICAL))


@pytest.fixture(scope="module")
def shift(model, potential):
    return _shift(model, potential)


@pytest.fixture(scope="module")
def uniform_shift(model, uniform):
    return _shift(model, uniform)


def test_the_shift_is_exported_and_reports_per_volt(shift):
    assert gfem.electro_optic_shift is gfem.electro_optic.electro_optic_shift
    assert {"ElectroOpticShift", "electro_optic_shift"} <= set(gfem.__all__)
    assert isinstance(shift, gfem.ElectroOpticShift)
    assert (shift.port, shift.mode, shift.frequency) == ("p", 1, FREQUENCY)
    k0 = 2 * np.pi * FREQUENCY / scipy.constants.c
    np.testing.assert_allclose(
        shift.phase_per_length_per_volt, k0 * shift.delta_n_eff_per_volt, rtol=1e-15
    )
    wavelength = scipy.constants.c / FREQUENCY
    np.testing.assert_allclose(
        shift.v_pi_l, wavelength / (2 * abs(shift.delta_n_eff_per_volt)), rtol=1e-15
    )
    assert "/V" in repr(shift)
    assert "V m" in repr(shift)


def test_an_x_cut_and_a_z_cut_turned_a_quarter_agree(mesh, shift):
    # turning the section and the crystal together about the guide changes
    # nothing physical: an x-cut film under a vertical field is a z-cut film
    # under a lateral one
    turned = Mesh(np.asarray(mesh.nodes) @ TURN.T, mesh._topology)
    axes = tuple(map(tuple, np.asarray(X_CUT) @ TURN.T))
    model = _optical(turned, _crystal(axes, LN_OPTICAL))
    rotated = _shift(model, _potential(turned, axes=axes))
    np.testing.assert_allclose(rotated.n_eff, shift.n_eff, rtol=1e-10)
    np.testing.assert_allclose(
        rotated.delta_n_eff_per_volt, shift.delta_n_eff_per_volt, rtol=1e-10
    )


def test_r22_acts_only_where_the_mode_has_no_weight(model, uniform, uniform_shift):
    # x-cut under a vertical field: r22 couples the field (crystal x) into
    # the crystal (x, y) entry, transverse-normal in the section, where the
    # lossless mode's real weight vanishes (E_n in quadrature with E_t)
    r22 = np.zeros((6, 3))
    r22[0, 1], r22[1, 1], r22[5, 0] = -R22, R22, -R22
    alone = _shift(_optical(model.mesh, _crystal(X_CUT, LN_OPTICAL, r22)), uniform)
    share = abs(alone.delta_n_eff_per_volt / uniform_shift.delta_n_eff_per_volt)
    assert share < 1e-12


def test_a_uniform_field_is_read_exactly_on_another_mesh(
    tmp_path, model, uniform_shift
):
    # the plates' exact linear potential, on a mesh of the outline alone
    rects = [(-3.0, -2.0, 3.0, 2.6, "clad", 0.7)]
    other = _uniform(tc.section(tmp_path / "plates.msh", rects, 0.7, PLATES))
    across = _shift(model, other)
    np.testing.assert_allclose(
        across.delta_n_eff_per_volt, uniform_shift.delta_n_eff_per_volt, rtol=1e-10
    )


def test_an_isotropic_change_is_the_thermo_optic_shift(mesh, uniform):
    # r with r_xx,y = r_yy,y = r_zz,y = a: in E = (0, -1/GAP, 0) an isotropic
    # medium of index n sees delta eps = n^4 a / GAP, i.e. delta n = n^3 a / (2 GAP),
    # which the thermo-optic shift gives for a unit rise and dn/dT = delta n
    n, a = 2.2, 20e-12
    r = np.zeros((6, 3))
    r[0, 1] = r[1, 1] = r[2, 1] = a
    crystal = gfem.Material(
        eps_r=n**2,
        electro_optic_tensor=r,
        thermo_optic_coefficient=n**3 * a / (2 * GAP),
    )
    model = _optical(mesh, crystal)
    electro = _shift(model, uniform)
    one = dataclasses.replace(
        uniform,
        coefficients=jnp.ones_like(uniform.coefficients),
        quantity="T",
        unit="K",
    )
    thermo = gfem.thermo_optic_shift(model, one, port="p", frequency=FREQUENCY)
    np.testing.assert_allclose(
        electro.delta_n_eff_per_volt, thermo.delta_n_eff, rtol=1e-10
    )


def test_the_shift_is_linear_in_the_tensor_and_the_drive(model, potential):
    # Euler: a function linear in r and in the potential's coefficients
    # equals the sum of each input times its derivative
    def shift(r, c):
        crystal = {
            region: gfem.MaterialField(
                eps_r=jnp.asarray(LN_OPTICAL),
                axes=X_CUT,
                electro_optic_tensor=r,
                region=region,
            )
            for region in ("core", "foot")
        }
        optical = model.replace(materials={**model.materials, **crystal})
        drive = dataclasses.replace(potential, coefficients=c)
        return _shift(optical, drive).delta_n_eff_per_volt

    r, c = jnp.asarray(R_LN), potential.coefficients
    value, (dr, dc) = jax.value_and_grad(shift, argnums=(0, 1))(r, c)
    np.testing.assert_allclose(jnp.sum(r * dr), value, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(c * dc), value, rtol=1e-12)


def test_the_tensor_gradient_matches_finite_differences(coarse):
    model = _optical(coarse, _crystal(X_CUT, LN_OPTICAL))
    potential = _potential(coarse)

    def shift(r):
        crystal = gfem.MaterialField(
            eps_r=jnp.asarray(LN_OPTICAL), axes=X_CUT, electro_optic_tensor=r
        )
        optical = model.replace(materials={**model.materials, "core": crystal})
        return _shift(optical, potential).delta_n_eff_per_volt

    r = jnp.asarray(R_LN)
    directions = unit_directions(np.random.default_rng(0), r.shape, 2)
    assert_gradient_matches(shift, r, directions, step=1e-11)


def test_the_static_permittivity_gradient_matches_finite_differences(coarse):
    # the crystal's three principal values on its axes and the oxide's value
    # of the electrostatic model reach the shift through the potential only
    model = _optical(coarse, _crystal(X_CUT, LN_OPTICAL))
    problem = gfem.Electrostatic(terminals={"t": ["top"]}, ground=["bottom"])
    options = gfem.SolverOptions(order=2, linear_solver="direct")

    def shift(p):
        crystal = gfem.MaterialField(eps_r=p[:3], axes=X_CUT)
        static = gfem.Model(
            coarse,
            materials={
                "core": crystal,
                "foot": crystal,
                "clad": gfem.MaterialField(eps_r=p[3]),
            },
        )
        es = gfem.solve(static, problem, options=options)
        return _shift(model, es.potential["t"]).delta_n_eff_per_volt

    p = jnp.asarray((*LN_STATIC, 3.9))
    directions = unit_directions(np.random.default_rng(1), p.shape, 2)
    assert_gradient_matches(shift, p, directions, step=1e-1)


# --- refusals --------------------------------------------------------------------


@pytest.mark.parametrize(
    "pick",
    [
        lambda es: es.potential,
        lambda es: es,
        lambda es: dataclasses.replace(es.potential["t"], quantity="T", unit="K"),
    ],
    ids=["mapping", "result", "temperature"],
)
def test_anything_but_one_terminals_potential_raises(model, electrostatic, pick):
    with pytest.raises(TypeError, match=r"pick one\s+terminal"):
        _shift(model, pick(electrostatic))


def test_a_field_that_is_no_potential_names_what_is_taken(model, potential):
    temperature = dataclasses.replace(potential, quantity="T", unit="K")
    with pytest.raises(TypeError, match=r"a potential in volts \(quantity 'V' on h1"):
        _shift(model, temperature)


def test_a_model_that_is_no_model_raises(potential):
    with pytest.raises(TypeError, match=r"electro_optic_shift needs a gfem\.Model"):
        _shift("model", potential)


def test_a_section_without_a_tensor_raises_naming_the_port(mesh, potential):
    model = _optical(mesh, gfem.Material(eps_r=LN_OPTICAL, axes=X_CUT))
    with pytest.raises(ValueError, match=r"wave port 'p'.*electro_optic_tensor"):
        _shift(model, potential)


@pytest.mark.parametrize(
    ("loss", "match"),
    [({"tan_delta": 1e-3}, "tan_delta"), ({"conductivity": 1.0}, "conductivity")],
)
def test_a_lossy_section_raises(mesh, potential, loss, match):
    model = _optical(mesh, _crystal(X_CUT, LN_OPTICAL), **loss)
    with pytest.raises(NotImplementedError, match=match):
        _shift(model, potential)


def test_crystal_axes_tilted_at_the_port_raise(mesh, potential):
    # the extraordinary axis turned from lateral toward the guide
    c, s = np.cos(0.3), np.sin(0.3)
    tilted = ((0.0, 1.0, 0.0), (-s, 0.0, c), (c, 0.0, s))
    model = _optical(mesh, _crystal(tilted, LN_OPTICAL))
    with pytest.raises(ValueError, match="principal axis along the port normal"):
        _shift(model, potential)


def test_a_model_or_potential_on_a_2d_mesh_raises(tmp_path, model, potential):
    flat = gfem.Mesh.from_gmsh(write_strip(tmp_path / "strip.msh"), length_unit="um")
    flat_model = gfem.Model(flat, materials={k: gfem.Material() for k in "abc"})
    flat_potential = gfem.Field(
        coefficients=jnp.zeros(len(flat.nodes)),
        mesh=flat,
        quantity="V",
        unit="V",
        space="h1",
        order=1,
        support=("a", "b", "c"),
    )
    for call in (
        lambda: _shift(flat_model, potential),
        lambda: _shift(model, flat_potential),
    ):
        with pytest.raises(NotImplementedError, match=r"electro_optic_shift.*2D mesh"):
            call()


def test_a_potential_split_across_ranks_raises(model, potential):
    piece = dataclasses.replace(potential)
    object.__setattr__(piece, "_comm", types.SimpleNamespace(comm=object()))
    with pytest.raises(NotImplementedError, match="gather on one rank"):
        _shift(model, piece)


def test_a_crystal_outside_the_potential_raises_naming_the_point(tmp_path, model):
    # a static model of the core's surroundings only: the foot reaches past it
    rects = [(-1.0, -2.0, 1.0, 2.6, "clad", None), (-0.5, 0.0, 0.5, 0.6, "core", 0.15)]
    window = tc.section(tmp_path / "window.msh", rects, MAXH, PLATES)
    with pytest.raises(ValueError, match=r"wave port 'p'.*Point \(.*outside the mesh"):
        _shift(model, _potential(window))


def test_a_crystal_outside_the_potential_s_support_raises_naming_the_region(
    model, potential
):
    # the same mesh, the crystal left out of the support: probe's message
    clad_only = dataclasses.replace(potential, support=("clad",))
    with pytest.raises(
        ValueError, match=r"Point \(.*\) m lies in volume region '(core|foot)'"
    ):
        _shift(model, clad_only)


def test_a_sequence_of_frequencies_or_an_unknown_port_raises(model, potential):
    with pytest.raises(ValueError, match="one frequency"):
        gfem.electro_optic_shift(
            model, potential, port="p", frequency=[FREQUENCY, 2 * FREQUENCY]
        )
    with pytest.raises(ValueError, match="no port named 'q'"):
        gfem.electro_optic_shift(model, potential, port="q", frequency=FREQUENCY)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("eps_r", LN_OPTICAL),
        ("tan_delta", (0.0, 0.0, 0.0)),
        ("mu_r", (1.0, 1.0, 1.0)),
        ("conductivity", (0.0, 0.0, 0.0)),
    ],
)
def test_an_optical_material_input_raises_under_grad(mesh, potential, name, value):
    # the mode is held constant: its derivative would be missing
    def shift(x):
        given = {"eps_r": jnp.asarray(LN_OPTICAL), name: x}
        crystal = gfem.MaterialField(
            **given, axes=X_CUT, electro_optic_tensor=jnp.asarray(R_LN)
        )
        return _shift(_optical(mesh, crystal), potential).delta_n_eff_per_volt

    with pytest.raises(gfem.NonDifferentiableError, match="electro-optic shift"):
        jax.grad(shift)(jnp.asarray(value))


def test_the_optical_node_coordinates_raise_under_grad(model, potential):
    # they move the mode and the points the static field is read at
    def shift(nodes):
        moved = Mesh(nodes, model.mesh._topology)
        return _shift(model.replace(mesh=moved), potential).delta_n_eff_per_volt

    with pytest.raises(gfem.NonDifferentiableError, match="node coordinates"):
        jax.grad(shift)(jnp.asarray(model.mesh.nodes))
