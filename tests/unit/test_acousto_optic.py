"""`gfem.acousto_optic_shift`: the first-order shift of a wave port's mode.

The optical section is the layered slab of the electro-optic tests (oxide, a
0.6 um lithium niobate film, oxide) between electrode walls; the acoustic
result is either manufactured (exact affine or quadratic fields on a 2D mesh
of the same extent: a uniform strain and a uniform field are read exactly at
any point) or a small real piezoelectric solve. The shift is first order from
one mode solve, so it is exactly linear in the fields and in the tensors; the
mode's own dependence is checked by re-solves.
"""

import dataclasses

import acousto_optic_cases as ao
import electro_optic_cases as eo
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import piezo_cases as pc
import pytest
import scipy.constants

import gfem
from gfem._materials import electro_optic_in_frame, stiffness_in_frame

P_LN, R_LN = ao.P_LN, eo.R_LN
X_CUT, N_E, N_O, N_OX = eo.X_CUT, eo.N_E, eo.N_O, eo.N_OX
FREQUENCY = eo.FREQUENCY
#: The slab mesh's edge-length factor: the shift's tests run on a coarse mesh,
#: the closed form on a finer one, the gradients (exact on any mesh) on a rough one.
COARSE, FINE, ROUGH = 6.0, 3.0, 12.0
#: Strain along the lateral axis, V/m of lateral field, per volt.
STRAIN, FIELD = 1e-3, 1e5
#: Acoustic frequencies of the manufactured result, Hz (labels only).
ACOUSTIC = (1.0e9, 2.0e9)


def _optical(mesh, film, **window):
    oxide = gfem.Material(eps_r=N_OX**2, **window)
    return gfem.Model(
        mesh,
        materials={"below": oxide, "film": film, "above": oxide},
        boundaries=[gfem.PEC("left"), gfem.PEC("right")],
        ports=[gfem.WavePort("p", "front")],
    )


def _film(axes=X_CUT, eps=eo.LN_OPTICAL, p=P_LN, r=R_LN):
    return gfem.Material(
        eps_r=eps,
        axes=axes,
        photoelastic_tensor=p,
        electro_optic_tensor=r,
    )


def _crystal(p=P_LN, r=R_LN, axes=X_CUT):
    """The film as a `gfem.MaterialField`, whose tensors may be traced."""
    return gfem.MaterialField(
        eps_r=jnp.asarray(eo.LN_OPTICAL),
        axes=axes,
        photoelastic_tensor=jnp.asarray(p),
        electro_optic_tensor=jnp.asarray(r),
    )


def _shift(model, acoustic, **kw):
    return gfem.acousto_optic_shift(
        model, acoustic, port="p", frequency=FREQUENCY, **kw
    )


def _uniform(mesh, strains=None, fields=None, **kw):
    """Fields of uniform lateral strain and field, one per acoustic frequency.

    ``strains[k]`` is the ``(3, 2)`` complex gradient array of the displacement
    and ``fields[k]`` the ``(2,)`` gradient of the potential.
    """
    n = len(ACOUSTIC)
    strains = np.zeros((n, 3, 2), complex) if strains is None else strains
    fields = np.zeros((n, 2), complex) if fields is None else fields
    return ao.manufactured_result(mesh, ACOUSTIC, strains, fields, **kw)


def _lateral(a, e):
    """A displacement with ``S_xx = a`` and a potential with ``E_x = e`` (complex)."""
    strains = np.zeros((len(ACOUSTIC), 3, 2), complex)
    strains[:, 0, 0] = a
    fields = np.zeros((len(ACOUSTIC), 2), complex)
    fields[:, 0] = -np.asarray(e)
    return strains, fields


@pytest.fixture(scope="module")
def acoustic_mesh(tmp_path_factory):
    return ao.acoustic_mesh(tmp_path_factory.mktemp("acousto_optic") / "acoustic.msh")


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    return eo.layered_slab(tmp_path_factory.mktemp("slab") / "coarse.msh", COARSE)


@pytest.fixture(scope="module")
def fine(tmp_path_factory):
    return eo.layered_slab(tmp_path_factory.mktemp("slab") / "fine.msh", FINE)


@pytest.fixture(scope="module")
def rough(tmp_path_factory):
    """The slab at twice the coarse edge length: a gradient's mesh, exact at any."""
    return eo.layered_slab(tmp_path_factory.mktemp("slab") / "rough.msh", ROUGH)


@pytest.fixture(scope="module")
def model(coarse):
    return _optical(coarse, _film())


@pytest.fixture(scope="module")
def acoustic(acoustic_mesh):
    # a complex lateral strain and field at each frequency, plus shears
    a = np.array([(1.0 + 0.5j) * STRAIN, (-0.5 + 2.0j) * STRAIN])
    e = np.array([(1.0 - 0.7j) * FIELD, (0.3 + 1.0j) * FIELD])
    strains, fields = _lateral(a, e)
    strains[0, 2, 0] = 4e-4  # S_xz = d_x u_z
    strains[1, 1, 1] = -2e-4j  # S_yy
    fields[0, 1] = 3e4  # E_y
    return _uniform(acoustic_mesh, strains, fields)


@pytest.fixture(scope="module")
def shift(model, acoustic):
    return _shift(model, acoustic)


# --- the result --------------------------------------------------------------------


def test_the_shift_is_exported_and_reports_per_acoustic_frequency(shift):
    assert gfem.acousto_optic_shift is gfem.acousto_optic.acousto_optic_shift
    assert {"AcoustoOpticShift", "acousto_optic_shift"} <= set(gfem.__all__)
    assert isinstance(shift, gfem.AcoustoOpticShift)
    assert (shift.port, shift.mode, shift.terminal) == ("p", 1, "hot")
    assert shift.frequency == FREQUENCY
    np.testing.assert_array_equal(shift.acoustic_frequency, ACOUSTIC)
    assert isinstance(shift.acoustic_frequency, np.ndarray)
    assert 1.9 < float(shift.n_eff) < 2.0
    for name in (
        "delta_n_eff_per_volt",
        "photoelastic_per_volt",
        "electro_optic_per_volt",
        "phase_per_length_per_volt",
    ):
        value = getattr(shift, name)
        assert value.shape == (2,)
        assert value.dtype == jnp.complex128
        assert np.isfinite(np.asarray(value)).all()
    assert np.abs(shift.photoelastic_per_volt).min() > 0
    assert np.abs(shift.electro_optic_per_volt).min() > 0


def test_the_parts_sum_and_the_phase_and_v_pi_l_follow(shift):
    # delta = pe + eo, summed once: the identity holds to the rounding of one sum
    total = np.asarray(shift.photoelastic_per_volt + shift.electro_optic_per_volt)
    np.testing.assert_allclose(shift.delta_n_eff_per_volt, total, rtol=1e-14, atol=0)
    k0 = 2 * np.pi * FREQUENCY / scipy.constants.c
    np.testing.assert_allclose(
        shift.phase_per_length_per_volt,
        k0 * np.asarray(shift.delta_n_eff_per_volt),
        rtol=1e-14,
        atol=0,
    )
    wavelength = scipy.constants.c / FREQUENCY
    np.testing.assert_allclose(
        shift.v_pi_l,
        wavelength / (2 * np.abs(np.asarray(shift.delta_n_eff_per_volt))),
        rtol=1e-14,
    )
    assert shift.v_pi_l.shape == (2,)


def test_the_result_is_a_pytree_with_static_frequencies_and_a_readable_repr(shift):
    leaves, tree = jax.tree_util.tree_flatten(shift)
    assert len(leaves) == 5
    back = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(back.acoustic_frequency, shift.acoustic_frequency)
    assert back.terminal == "hot"
    text = repr(shift)
    assert text.startswith("AcoustoOpticShift(port 'p', mode 1, terminal 'hot'")
    assert "2 acoustic frequencies" in text
    assert "v_pi_l" in text
    assert shift.acoustic_frequency.flags.writeable is False


# --- closed form, re-solves ---------------------------------------------------------


@pytest.fixture(scope="module")
def slab_shift(fine, acoustic_mesh):
    """The fine slab's shift under a uniform lateral strain and field, one frequency."""
    strains, fields = _lateral(np.array([STRAIN, 0.0]), np.array([FIELD, 0.0]))
    result = _uniform(acoustic_mesh, strains, fields)
    return _shift(_optical(fine, _film()), result)


def test_a_uniform_strain_and_field_match_the_1d_slab_closed_form(slab_shift):
    # PEC on the walls keeps the TE mode the 1D slab's, E along x = crystal z:
    # Delta eps_zz = -n_e^4 (p33 S_xx + r33 E_x), delta n = Delta eps Gamma / (2 n)
    # with Gamma the film's share of the integral of |E|^2. The shift is first
    # order, the closed form exact, so the gap is the mode's discretization on this
    # mesh: second order in the edge length, measured 4e-3 and 8e-3 at edge factor
    # 6, 8e-4 and 1e-3 at factor 3 (this mesh), 5e-5 at factor 1.
    n, share = ao.te_slab(N_E**2)
    pe = -(N_E**4) * P_LN[2, 2] * STRAIN * share / (2 * n)
    er = -(N_E**4) * eo.R33 * FIELD * share / (2 * n)
    np.testing.assert_allclose(slab_shift.n_eff, n, rtol=3e-4)
    np.testing.assert_allclose(slab_shift.photoelastic_per_volt[0], pe, rtol=2e-3)
    np.testing.assert_allclose(slab_shift.electro_optic_per_volt[0], er, rtol=2e-3)
    # the second frequency holds no field: exactly nothing
    assert slab_shift.delta_n_eff_per_volt[1] == 0


def test_the_shift_matches_central_differences_of_two_port_mode_re_solves(
    coarse, acoustic_mesh
):
    # A diagonal change of the crystal-frame permittivity: a lateral strain acts
    # as Delta eps = -eps (p S) eps with S_zz = a (crystal frame), diagonal for 3m,
    # and a lateral field through r13, r13, r33. Re and Im of the shift are the
    # derivatives of n_eff along h Re(Delta eps) and h Im(Delta eps). The step
    # moves eps by 1e-3 of itself at h = 1: the truncation of a central
    # difference is (h Delta eps / eps)^2 ~ 1e-6 of the shift, the eigen-solve's
    # round-off ~1e-12 / 2h.
    a, e = (1.0 + 0.5j) * STRAIN, (1.0 - 0.7j) * FIELD
    strains, fields = _lateral(np.array([a, 0.0]), np.array([e, 0.0]))
    result = _uniform(acoustic_mesh, strains, fields)
    first = _shift(_optical(coarse, _film()), result)
    eps = np.asarray(eo.LN_OPTICAL)
    p = np.asarray(P_LN)
    r = np.asarray(R_LN)
    eta = np.array(
        [
            p[0, 2] * a + r[0, 2] * e,
            p[1, 2] * a + r[1, 2] * e,
            p[2, 2] * a + r[2, 2] * e,
        ]
    )
    delta_eps = -(eps**2) * eta

    def n_eff(change):
        model = _optical(coarse, _film(eps=tuple(eps + change), p=None, r=None))
        out = gfem.solve(model, gfem.PortMode("p", FREQUENCY))
        return float(np.real(out.n_eff[0]))

    for part, expected in (
        (np.real, first.delta_n_eff_per_volt[0].real),
        (np.imag, first.delta_n_eff_per_volt[0].imag),
    ):
        h = 1.0
        step = h * part(delta_eps)
        fd = (n_eff(step) - n_eff(-step)) / (2 * h)
        np.testing.assert_allclose(fd, expected, rtol=1e-5)


# --- frames ------------------------------------------------------------------------

#: Quarter turns of the crystal about the guide: the principal axes stay on the
#: coordinate axes, so the pre-rotated tensors go with a diagonal global eps_r.
_TURNED = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
_SWAPPED = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "axes", [X_CUT, _TURNED, _SWAPPED], ids=["x-cut", "turned", "swapped"]
)
def test_crystal_frame_tensors_with_axes_equal_pre_rotated_global_tensors(
    coarse, acoustic, axes
):
    # global-frame tensors with identity axes: another operation order, so a
    # relative 1e-12 rather than bit-equality
    crystal = _shift(_optical(coarse, _film(axes=axes)), acoustic)
    a = np.asarray(axes)
    eps_global = tuple(np.diag(a.T @ np.diag(eo.LN_OPTICAL) @ a))
    lab = gfem.Material(
        eps_r=eps_global,
        photoelastic_tensor=stiffness_in_frame(np.asarray(P_LN), axes),
        electro_optic_tensor=electro_optic_in_frame(np.asarray(R_LN), axes),
    )
    pre = _shift(_optical(coarse, lab), acoustic)
    for name in ("photoelastic_per_volt", "electro_optic_per_volt"):
        np.testing.assert_allclose(
            getattr(pre, name), getattr(crystal, name), rtol=1e-12, atol=0
        )
    np.testing.assert_allclose(
        pre.delta_n_eff_per_volt, crystal.delta_n_eff_per_volt, rtol=1e-12
    )


# --- the electro-optic part against the static shift ---------------------------------


def test_the_electro_optic_part_equals_the_static_shift_of_the_matching_potential(
    coarse, acoustic_mesh, model
):
    # a linear complex potential, phi = (g_re + i g_im) . (x, y): its real and
    # imaginary parts are the static shifts of two real linear 3D potentials
    g = np.array([[-1.2e5 + 0.4e5j, 0.5e5 - 0.9e5j], [2.0e5, -1.0e5j]])
    result = _uniform(acoustic_mesh, None, g)
    ao_shift = _shift(model, result)
    support = ("below", "film", "above")
    for k in range(2):
        parts = [
            gfem.electro_optic_shift(
                model,
                ao.affine_field(coarse, support, "V", "V", part(g[k])),
                port="p",
                frequency=FREQUENCY,
            ).delta_n_eff_per_volt
            for part in (np.real, np.imag)
        ]
        np.testing.assert_allclose(
            ao_shift.electro_optic_per_volt[k], parts[0] + 1j * parts[1], rtol=1e-12
        )


# --- identities --------------------------------------------------------------------


# Two shifts on separate mode solves differ by the eigen-solve's round-off, 6e-14
# relative seen on 8 threads, so such comparisons use 1e-12.


def test_the_shift_is_linear_in_the_tensors_and_the_fields(
    coarse, acoustic_mesh, acoustic, shift
):
    doubled = _shift(
        _optical(coarse, _film(p=2 * P_LN, r=2 * np.asarray(R_LN))), acoustic
    )
    for name in ("photoelastic_per_volt", "electro_optic_per_volt"):
        np.testing.assert_allclose(
            getattr(doubled, name), 2 * np.asarray(getattr(shift, name)), rtol=1e-12
        )
    # twice the strain and half the field: the parts scale on their own
    strains, fields = _lateral(
        np.array([2 * STRAIN, STRAIN]), np.array([0.5 * FIELD, FIELD])
    )
    base = _shift(_optical(coarse, _film()), _uniform(acoustic_mesh, strains, fields))
    twice = _shift(
        _optical(coarse, _film()), _uniform(acoustic_mesh, 2 * strains, 0.5 * fields)
    )
    np.testing.assert_allclose(
        twice.photoelastic_per_volt,
        2 * np.asarray(base.photoelastic_per_volt),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        twice.electro_optic_per_volt,
        0.5 * np.asarray(base.electro_optic_per_volt),
        rtol=1e-12,
    )


def test_no_photoelastic_tensor_or_a_zero_one_gives_no_photoelastic_part(
    coarse, acoustic, shift
):
    eo_only = _shift(_optical(coarse, _film(p=None)), acoustic)
    zeroed = _shift(_optical(coarse, _film(p=np.zeros((6, 6)))), acoustic)
    for result in (eo_only, zeroed):
        np.testing.assert_array_equal(result.photoelastic_per_volt, 0)
        np.testing.assert_allclose(
            result.electro_optic_per_volt, shift.electro_optic_per_volt, rtol=1e-12
        )
    # and a zero electro-optic tensor, or none, leaves the photoelastic part
    p_only = _shift(_optical(coarse, _film(r=None)), acoustic)
    np.testing.assert_array_equal(p_only.electro_optic_per_volt, 0)
    np.testing.assert_allclose(
        p_only.photoelastic_per_volt, shift.photoelastic_per_volt, rtol=1e-12
    )


def test_a_tensor_only_needs_the_field_it_reads(coarse, acoustic_mesh):
    # an electro-optic-only crystal needs no mechanical domain, a photoelastic-only
    # one no potential support
    strains, fields = _lateral(np.array([STRAIN, STRAIN]), np.array([FIELD, FIELD]))
    no_u = _uniform(acoustic_mesh, strains, fields, mechanical=("below",))
    no_v = _uniform(acoustic_mesh, strains, fields, electric=("below",))
    full = _uniform(acoustic_mesh, strains, fields)
    eo_only = _optical(coarse, _film(p=None))
    p_only = _optical(coarse, _film(r=None))
    np.testing.assert_allclose(
        _shift(eo_only, no_u).delta_n_eff_per_volt,
        _shift(eo_only, full).delta_n_eff_per_volt,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _shift(p_only, no_v).delta_n_eff_per_volt,
        _shift(p_only, full).delta_n_eff_per_volt,
        rtol=1e-12,
    )


def test_offset_moves_where_the_fields_are_read(
    coarse, acoustic_mesh, model, acoustic, shift
):
    # S and E of the affine fields do not depend on where they are read
    moved = _shift(model, acoustic, offset=(0.4e-6, -0.3e-6))
    np.testing.assert_allclose(
        moved.delta_n_eff_per_volt, shift.delta_n_eff_per_volt, rtol=1e-12
    )
    # u_x = c x^2 has S_xx = 2 c x: reading at x + d adds the uniform strain 2 c d
    c, d = 2e3, 0.5e-6
    zero = np.zeros((2, 3, 2), complex)
    curvature = np.zeros((2, 3), complex)
    curvature[:, 0] = c
    quadratic = ao.manufactured_result(
        acoustic_mesh,
        ACOUSTIC,
        zero,
        np.zeros((2, 2), complex),
        order=2,
        curvature=curvature,
    )
    uniform = _uniform(acoustic_mesh, *_lateral(np.full(2, 2 * c * d), np.zeros(2)))
    in_x = _optical(coarse, _film(r=None))
    at_zero = _shift(in_x, quadratic)
    at_d = _shift(in_x, quadratic, offset=(d, 0.0))
    step = _shift(in_x, uniform)
    np.testing.assert_allclose(
        np.asarray(at_d.photoelastic_per_volt)
        - np.asarray(at_zero.photoelastic_per_volt),
        step.photoelastic_per_volt,
        rtol=1e-10,
    )


# --- a real piezoelectric solve ---------------------------------------------------


@pytest.fixture(scope="module")
def idt(tmp_path_factory):
    return pc.mesh(tmp_path_factory.mktemp("idt") / "idt.msh", "lines")


@pytest.fixture(scope="module")
def piezo(idt):
    mesh = idt
    frequencies = [0.9e9, 0.95e9, 1.0e9]
    problem = pc.problem(frequencies, keep_fields=True)
    return gfem.solve(pc.model(mesh, "lines"), problem)


def test_a_piezoelectric_solve_under_the_window_gives_finite_phasors(piezo, coarse):
    # the crystal under the surface (y < 0): the window sits 0.8 um below it
    film = _film(axes=pc.AXES)
    result = _shift(_optical(coarse, film), piezo, offset=(0.0, -0.8e-6))
    assert result.delta_n_eff_per_volt.shape == (3,)
    np.testing.assert_allclose(result.acoustic_frequency, [0.9e9, 0.95e9, 1.0e9])
    assert np.isfinite(np.asarray(result.delta_n_eff_per_volt)).all()
    assert (np.abs(np.asarray(result.photoelastic_per_volt)) > 0).all()
    assert (np.abs(np.asarray(result.electro_optic_per_volt)) > 0).all()
    assert (np.abs(np.asarray(result.v_pi_l)) < np.inf).all()
    # one terminal: None and its name read the same fields
    named = _shift(_optical(coarse, film), piezo, offset=(0.0, -0.8e-6), terminal="hot")
    # two mode solves of one model: the eigen-solve's round-off with threads
    # (1e-13 seen on 8 threads), far below the shift's 1e-10 gate
    np.testing.assert_allclose(
        named.delta_n_eff_per_volt, result.delta_n_eff_per_volt, rtol=1e-10
    )


# --- refusals ------------------------------------------------------------------------


def test_a_model_that_is_no_model_or_a_result_that_is_none_raises(model, acoustic):
    with pytest.raises(TypeError, match=r"acousto_optic_shift needs a gfem\.Model"):
        gfem.acousto_optic_shift(acoustic, acoustic, port="p", frequency=FREQUENCY)
    for bad in (None, acoustic.displacement(ACOUSTIC[0]), 3.0):
        with pytest.raises(TypeError, match=r"acousto_optic_shift takes a gfem\.Piezo"):
            _shift(model, bad)


def test_a_model_on_a_2d_mesh_raises(acoustic_mesh, acoustic):
    flat = gfem.Model(
        acoustic_mesh, materials={r: gfem.Material() for r in ao.ACOUSTIC_REGIONS}
    )
    with pytest.raises(NotImplementedError, match=r"acousto_optic_shift.*2D mesh"):
        _shift(flat, acoustic)


def test_a_result_without_kept_fields_names_keep_fields(model, piezo):
    # the same result with nothing kept
    import dataclasses

    bare = dataclasses.replace(
        piezo, displacement_fields=(), potential_fields=(), kept=()
    )
    with pytest.raises(ValueError, match=r"kept no fields.*keep_fields=True"):
        _shift(model, bare)


def test_an_unknown_or_ambiguous_terminal_raises_naming_the_terminals(
    model, acoustic_mesh
):
    strains, fields = _lateral(np.full(2, STRAIN), np.full(2, FIELD))
    two = ao.manufactured_result(
        acoustic_mesh, ACOUSTIC, strains, fields, terminals=("hot", "cold")
    )
    with pytest.raises(ValueError, match=r"terminals 'hot', 'cold'; pass terminal="):
        _shift(model, two)
    with pytest.raises(ValueError, match=r"terminal 'warm' is not a terminal.*'hot'"):
        _shift(model, two, terminal="warm")
    cold = _shift(model, two, terminal="cold")
    assert cold.terminal == "cold"


def test_a_port_that_is_not_a_wave_port_or_a_sequence_of_frequencies_raises(
    model, acoustic
):
    with pytest.raises(ValueError, match="no port named 'q'"):
        gfem.acousto_optic_shift(model, acoustic, port="q", frequency=FREQUENCY)
    with pytest.raises(ValueError, match="one frequency"):
        gfem.acousto_optic_shift(
            model, acoustic, port="p", frequency=[FREQUENCY, 2 * FREQUENCY]
        )


def test_a_section_without_a_tensor_raises_naming_the_port(coarse, acoustic):
    plain = _optical(coarse, gfem.Material(eps_r=eo.LN_OPTICAL, axes=X_CUT))
    with pytest.raises(
        ValueError, match=r"wave port 'p'.*photoelastic_tensor or an electro_optic"
    ):
        _shift(plain, acoustic)


@pytest.mark.parametrize(
    ("loss", "match"),
    [({"tan_delta": 1e-3}, "tan_delta"), ({"conductivity": 1.0}, "conductivity")],
)
def test_a_lossy_section_raises(coarse, acoustic, loss, match):
    film = gfem.Material(
        eps_r=eo.LN_OPTICAL,
        axes=X_CUT,
        photoelastic_tensor=P_LN,
        electro_optic_tensor=R_LN,
        **loss,
    )
    with pytest.raises(NotImplementedError, match=rf"acousto_optic_shift.*{match}"):
        _shift(_optical(coarse, film), acoustic)


def test_a_point_outside_the_mechanical_domain_or_the_potential_raises_naming_it(
    acoustic_mesh, model
):
    strains, fields = _lateral(np.full(2, STRAIN), np.full(2, FIELD))
    no_u = _uniform(acoustic_mesh, strains, fields, mechanical=("below",))
    with pytest.raises(
        ValueError, match=r"wave port 'p'.*reading the strain.*Point \(.*\) m lies"
    ):
        _shift(model, no_u)
    no_v = _uniform(acoustic_mesh, strains, fields, electric=("above",))
    with pytest.raises(
        ValueError, match=r"wave port 'p'.*reading the field.*Point \(.*\) m lies"
    ):
        _shift(model, no_v)
    # a point off the acoustic mesh: the offset moves the window out of it
    full = _uniform(acoustic_mesh, strains, fields)
    with pytest.raises(ValueError, match=r"reading the strain.*outside the mesh"):
        _shift(model, full, offset=(0.0, 50e-6))


@pytest.mark.parametrize(
    "offset", [(0.0,), (0.0, 0.0, 0.0), ((0.0, 0.0), (0.0, 0.0)), (np.nan, 0.0)]
)
def test_an_offset_that_is_not_two_finite_numbers_raises(model, acoustic, offset):
    with pytest.raises(ValueError, match=r"acousto_optic_shift: offset"):
        _shift(model, acoustic, offset=offset)


def test_the_optical_node_coordinates_raise_under_grad(model, acoustic):
    # they move the mode and the points the acoustic fields are read at
    def f(nodes):
        moved = gfem.mesh.Mesh(nodes, model.mesh._topology)
        return _shift(model.replace(mesh=moved), acoustic).delta_n_eff_per_volt.real

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"node coordinates.*stop_gradient"
    ):
        jax.jacrev(f)(jnp.asarray(model.mesh.nodes))


def test_a_traced_optical_permittivity_raises_under_grad_with_the_stop_gradient_advice(
    coarse, acoustic
):
    # the mode is held constant: its derivative in eps_r would be missing
    def f(eps):
        film = gfem.MaterialField(
            eps_r=eps,
            axes=X_CUT,
            photoelastic_tensor=jnp.asarray(P_LN),
            electro_optic_tensor=jnp.asarray(R_LN),
        )
        return _shift(_optical(coarse, film), acoustic).delta_n_eff_per_volt.real

    with pytest.raises(gfem.NonDifferentiableError, match="acousto-optic shift"):
        jax.jacrev(f)(jnp.asarray(eo.LN_OPTICAL))


# --- gradients ---------------------------------------------------------------------

# Every check differences a scalar objective of the shift with the fourth-order
# central difference of `finite_differences` at its gate (rtol 1e-6). Its
# truncation is h^4 J^(5)/30 and its round-off 1.5 delta/h, with delta the
# relative noise of one evaluation of the objective; a step is chosen per check
# so both sit well below the gate, and the measured errors are quoted.


def _objective(delta):
    """``sum_f |delta n_f|^2``: real, and it reads both parts of the phasor."""
    return jnp.sum(jnp.abs(delta) ** 2)


@pytest.mark.parametrize(
    ("leaf", "shape", "scale"),
    [("p", (6, 6), 0.03), ("r", (6, 3), 3e-12)],
    ids=["photoelastic", "electro_optic"],
)
def test_a_tensor_gradient_matches_finite_differences(
    rough, acoustic, leaf, shape, scale
):
    # The shift is exactly linear in each tensor, so |delta|^2 is quadratic in the
    # perturbation and the fourth-order difference has no truncation (J^(5) = 0).
    # What is left is the mode solve's round-off, 6e-14 relative (seen on 8
    # threads), so the difference carries 1.5 * 6e-14 / h of it. The perturbation
    # `scale` is ~20% of p13 and ~10% of r33: the objective moves by order 1 over
    # a unit of h, so h = 1 puts the round-off at ~1e-13 against the gate of 1e-6.
    # Measured relative errors: 2e-15 (p) and 1e-13 (r).
    rng = np.random.default_rng(21)
    base = {"p": np.asarray(P_LN), "r": np.asarray(R_LN)}

    def objective(q):
        given = {**base, leaf: base[leaf] + scale * q}
        film = _crystal(p=given["p"], r=given["r"])
        return _objective(_shift(_optical(rough, film), acoustic).delta_n_eff_per_volt)

    fd.assert_gradient_matches(
        objective, jnp.zeros(shape), fd.unit_directions(rng, shape, 1), step=1.0
    )


def test_the_forward_under_grad_equals_the_plain_forward(rough, acoustic):
    # the same mode solve and the same reading, traced or not: only the order of
    # the floating-point sums differs, and the eigen-solve's round-off with
    # threads is 6e-14 relative (seen on 8 threads), so 1e-12
    def objective(q):
        film = _crystal(p=np.asarray(P_LN) + 0.03 * q)
        return _objective(_shift(_optical(rough, film), acoustic).delta_n_eff_per_volt)

    q = jnp.asarray(np.random.default_rng(4).standard_normal((6, 6)))
    value, grad = jax.value_and_grad(objective)(q)
    np.testing.assert_allclose(value, objective(q), rtol=1e-12, atol=0)
    assert np.isfinite(np.asarray(grad)).all()
    assert np.abs(np.asarray(grad)).max() > 0


# a real solve on a coarse transducer section (exact gradients on any mesh), the
# window under the surface as in the finite-phasors test above
WINDOW = (0.0, -0.8e-6)


@pytest.fixture(scope="module")
def idt_rough(tmp_path_factory):
    return pc.mesh(tmp_path_factory.mktemp("idt") / "rough.msh", "lines", maxh=2.4)


@pytest.fixture(scope="module")
def piezo_rough(idt_rough):
    return gfem.solve(pc.model(idt_rough, "lines"), pc.problem(keep_fields=True))


def _piezo_objective(mesh, optical, *, node_map=None, edit=None, q=None):
    """``_objective`` of the shift of a one-frequency piezoelectric solve."""
    if node_map is not None:
        mesh = mesh.with_nodes(node_map(jnp.asarray(mesh.imported_nodes), q))
    model = pc.model(mesh, "lines")
    if edit is not None:
        materials = dict(model.materials)
        for region in ("ln", "ln_frame"):
            field = gfem.MaterialField.uniform(region, materials[region])
            materials[region] = edit(field, q)
        model = gfem.Model(mesh, materials=materials, excluded={"mechanical": ["air"]})
    acoustic = gfem.solve(model, pc.problem(keep_fields=True))
    shift = _shift(optical, acoustic, offset=WINDOW)
    return _objective(shift.delta_n_eff_per_volt)


def test_the_offset_gradient_matches_finite_differences_in_both_components(
    rough, piezo_rough
):
    # The fields are those of a real solve (order-2 elements): the strain is
    # piecewise linear, so the objective is piecewise quadratic in the offset and
    # the fourth-order difference has no truncation while no reading point crosses
    # an element edge (a crossing would add the jump of the strain over the step).
    # Nothing but the points moves, so the noise is the reading's own: 1e-16
    # relative plus the rounding of a 1e-6 m coordinate, 1e-22 m. The unit of q is
    # 1e-6 m and the step 1e-4 of it, 1e-10 m, 1e-4 of an element edge: the ~1e2
    # points of the section cross an edge with probability ~1e-2 in all, and the
    # meshes are fixed. The objective moves by ~1e-4 of itself over a step, so the
    # round-off of the difference is 1.5e-16 / 1e-4 ~ 1e-12. Measured relative
    # errors: 4e-12 (x), 4e-12 (y), 8e-13 (mixed).
    unit = 1e-6
    optical = _optical(rough, _film(axes=pc.AXES))

    def objective(q):
        shift = _shift(optical, piezo_rough, offset=jnp.asarray(WINDOW) + unit * q)
        return _objective(shift.delta_n_eff_per_volt)

    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, -0.8])]
    fd.assert_gradient_matches(objective, jnp.zeros(2), directions, step=1e-4)

    # and the forward under grad is the plain forward, at the same values
    value, _ = jax.value_and_grad(objective)(jnp.zeros(2))
    np.testing.assert_allclose(value, objective(jnp.zeros(2)), rtol=1e-12, atol=0)


def _stretch(nodes, q):
    return nodes.at[:, 0].set((1.0 + q[0]) * nodes[:, 0])


def _c44(field, q):
    """Stiffness with ``c44`` (and its twin ``c55``) raised by ``1e10 q`` Pa."""
    bump = np.zeros((6, 6))
    bump[3, 3] = bump[4, 4] = 1e10
    return dataclasses.replace(field, stiffness=jnp.asarray(pc.C_LN) + bump * q[0])


@pytest.mark.parametrize(
    ("kwargs", "step"),
    [
        ({"node_map": _stretch}, 1e-6),
        ({"edit": _c44}, 1e-2),
    ],
)
def test_an_acoustic_design_gradient_matches_finite_differences(
    idt_rough, rough, kwargs, step
):
    # Through the solve's cotangents of the kept fields and the Eulerian reading at
    # fixed optical points. Each evaluation is a direct solve of its own (see
    # `finite_differences`) with ~1e-15 relative noise, so the difference carries
    # 1.5e-15 / h of it. c44: the step is the piezoelectric gradient tests', 1e-2
    # of a unit of 1e10 Pa (3% of c44), whose fourth-order truncation is ~1e-8.
    # The stretch moves the acoustic nodes, so the elements under the fixed points,
    # by at most 2 step * 4 um = 8e-12 m: the strain jumps across an element edge,
    # so the step is kept to 1e-6 (a ~1e-5 chance for each of ~1e2 points to cross
    # one), the truncation is ~1e-16 and the round-off ~1e-9 of the objective's own
    # change over the step. Measured relative errors: 1e-9 (stretch), 1e-9 (c44).
    optical = _optical(rough, _film(axes=pc.AXES))

    def objective(q):
        return _piezo_objective(idt_rough, optical, q=q, **kwargs)

    fd.assert_gradient_matches(objective, jnp.zeros(1), [np.array([1.0])], step=step)
