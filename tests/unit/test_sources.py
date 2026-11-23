"""Source declarations: plane waves, Gaussian beams, user incident fields, dipoles.

The direction and polarization rules are checked against vectors written out
by hand (the propagation direction in spherical angles from +z, TM along
theta-hat, TE along phi-hat, IEEE handedness under exp(+iwt)); the
constructors' refusals by their messages.
"""

import math

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._sources import polarization_vector, propagation_basis

S2 = math.sqrt(0.5)


# --- direction and polarization ---------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "phi", "k", "tm", "te"),
    [
        # along +z: TM along (cos phi, sin phi, 0), TE along z x that
        (0.0, 0.0, (0, 0, 1), (1, 0, 0), (0, 1, 0)),
        (0.0, math.pi / 2, (0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        # along -z: TM along -(cos phi, sin phi, 0)
        (math.pi, 0.0, (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        # along +x: TM along -z, TE along +y
        (math.pi / 2, 0.0, (1, 0, 0), (0, 0, -1), (0, 1, 0)),
        # along +y: TE along -x
        (math.pi / 2, math.pi / 2, (0, 1, 0), (0, 0, -1), (-1, 0, 0)),
        # 45 deg in the xz plane, going up
        (math.pi / 4, 0.0, (S2, 0, S2), (S2, 0, -S2), (0, 1, 0)),
    ],
)
def test_the_propagation_basis_is_the_spherical_one(theta, phi, k, tm, te):
    k_hat, theta_hat, phi_hat = propagation_basis(theta, phi)
    np.testing.assert_allclose(k_hat, k, atol=1e-15)
    np.testing.assert_allclose(theta_hat, tm, atol=1e-15)
    np.testing.assert_allclose(phi_hat, te, atol=1e-15)


def test_the_basis_is_right_handed_and_orthonormal_in_every_direction():
    rng = np.random.default_rng(0)
    for theta, phi in zip(
        rng.uniform(0, np.pi, 20), rng.uniform(-7, 7, 20), strict=True
    ):
        k_hat, theta_hat, phi_hat = propagation_basis(theta, phi)
        np.testing.assert_allclose(np.cross(theta_hat, phi_hat), k_hat, atol=1e-15)
        frame = np.array([theta_hat, phi_hat, k_hat])
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-15)
        # TE is z x kappa, kappa the in-plane direction (the Floquet TE reference)
        kappa = np.array([np.cos(phi), np.sin(phi), 0.0])
        np.testing.assert_allclose(phi_hat, np.cross([0, 0, 1], kappa), atol=1e-15)


def test_te_and_tm_are_the_basis_vectors():
    _, theta_hat, phi_hat = propagation_basis(0.7, 0.3)
    np.testing.assert_allclose(polarization_vector(0.7, 0.3, "TM"), theta_hat)
    np.testing.assert_allclose(polarization_vector(0.7, 0.3, "TE"), phi_hat)


def test_rcp_is_theta_minus_i_phi_over_root_two():
    k_hat, theta_hat, phi_hat = propagation_basis(0.7, 0.3)
    rcp = polarization_vector(0.7, 0.3, "RCP")
    lcp = polarization_vector(0.7, 0.3, "LCP")
    np.testing.assert_allclose(rcp, (theta_hat - 1j * phi_hat) * S2, atol=1e-15)
    np.testing.assert_allclose(lcp, (theta_hat + 1j * phi_hat) * S2, atol=1e-15)
    # IEEE: with the thumb along k, the real field Re(e exp(+iwt)) turns
    # counterclockwise (right hand) for RCP, clockwise for LCP
    for e, sign in ((rcp, 1), (lcp, -1)):
        now, later = np.real(e), np.real(e * np.exp(0.1j))
        assert sign * np.dot(np.cross(now, later), k_hat) > 0


def test_rcp_going_down_is_x_plus_i_y():
    # the Floquet recipe's check: at k = -z, (x + iy)/sqrt(2) is right-handed
    rcp = polarization_vector(math.pi, 0.0, "RCP")
    np.testing.assert_allclose(rcp, -np.array([1, 1j, 0]) * S2, atol=1e-15)


def test_a_jones_pair_is_normalized():
    e = polarization_vector(0.7, 0.3, (3.0, 4j))
    _, theta_hat, phi_hat = propagation_basis(0.7, 0.3)
    np.testing.assert_allclose(e, 0.6 * theta_hat + 0.8j * phi_hat, atol=1e-15)
    np.testing.assert_allclose(
        polarization_vector(0.7, 0.3, (1, -1j)), polarization_vector(0.7, 0.3, "RCP")
    )


# --- constructors -------------------------------------------------------------------


def test_a_plane_wave_keeps_its_parameters():
    wave = gfem.PlaneWave("x", theta=0.5, phi=0.25, polarization=(1, 1j), amplitude=2j)
    assert wave.name == "x"
    assert (wave.theta, wave.phi) == (0.5, 0.25)
    assert wave.polarization == (1 + 0j, 1j)
    assert wave.amplitude == 2j
    plain = gfem.PlaneWave("y", theta=0.0, polarization="TM")
    assert (plain.phi, plain.amplitude) == (0.0, 1.0)
    assert repr(plain) == "PlaneWave('y', theta=0.0, phi=0.0, polarization='TM')"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"theta": -0.1}, ValueError, r"theta must be in \[0, pi\]"),
        ({"theta": 3.2}, ValueError, r"theta must be in \[0, pi\]"),
        ({"theta": math.nan}, ValueError, "theta must be finite"),
        ({"phi": math.inf}, ValueError, "phi must be finite"),
        ({"polarization": "s"}, ValueError, "'TE', 'TM', 'RCP', 'LCP' or a Jones"),
        ({"polarization": (1, 2, 3)}, ValueError, "Jones pair"),
        ({"polarization": (0, 0)}, ValueError, "Jones pair .* is zero"),
        ({"polarization": (1, math.nan)}, ValueError, "Jones pair .* finite"),
        ({"amplitude": math.inf}, ValueError, "amplitude must be finite"),
        ({"amplitude": "big"}, ValueError, "amplitude must be a complex number"),
    ],
)
def test_a_plane_wave_refuses_bad_parameters(kwargs, error, match):
    args = {"theta": 0.0, "polarization": "TE"} | kwargs
    with pytest.raises(error, match=match):
        gfem.PlaneWave("x", **args)


@pytest.mark.parametrize("name", [None, 3, ""])
def test_a_source_needs_a_name(name):
    with pytest.raises(TypeError, match=r"PlaneWave.name must be a non-empty string"):
        gfem.PlaneWave(name, theta=0.0, polarization="TE")


def test_a_traced_angle_is_not_differentiable():
    def build(theta):
        gfem.PlaneWave("x", theta=theta, polarization="TE")
        return theta

    with pytest.raises(gfem.NonDifferentiableError, match=r"PlaneWave.theta"):
        jax.grad(build)(0.3)


def test_a_gaussian_beam_keeps_its_parameters():
    beam = gfem.GaussianBeam(
        "fiber", waist=5.2e-6, focus=(1e-6, 0, 2e-6), theta=3.0, polarization="TE"
    )
    assert (beam.waist, beam.focus, beam.theta, beam.phi) == (
        5.2e-6,
        (1e-6, 0.0, 2e-6),
        3.0,
        0.0,
    )
    assert beam.polarization == "TE"
    centred = gfem.GaussianBeam("b", waist=1e-6, theta=0.0, polarization="RCP")
    assert centred.focus == (0.0, 0.0, 0.0)
    assert "waist=1e-06" in repr(centred)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"waist": 0.0}, "waist must be > 0 m"),
        ({"waist": -1.0}, "waist must be > 0 m"),
        ({"focus": (0, 0)}, "focus must be 3 coordinates"),
        ({"focus": (0, 0, math.inf)}, "focus must be 3 finite coordinates"),
        ({"theta": 4.0}, r"theta must be in \[0, pi\]"),
    ],
)
def test_a_gaussian_beam_refuses_bad_parameters(kwargs, match):
    args = {"waist": 1e-6, "theta": 0.0, "polarization": "TE"} | kwargs
    with pytest.raises(ValueError, match=match):
        gfem.GaussianBeam("b", **args)


def _wave(points, frequency):
    return np.zeros((len(points), 3), complex) + frequency * 0


def test_an_incident_field_shows_its_function_by_qualified_name():
    field = gfem.IncidentField("mine", _wave)
    assert field.function is _wave
    assert repr(field) == f"IncidentField('mine', function={__name__}._wave)"


def test_an_incident_field_needs_a_function():
    with pytest.raises(TypeError, match=r"IncidentField.function must be callable"):
        gfem.IncidentField("mine", np.zeros(3))


def test_dipoles_keep_position_and_moment():
    p = gfem.ElectricDipole("d", position=(0, 0, 5e-8), moment=(0, 0, 1e-30))
    assert p.position == (0.0, 0.0, 5e-8)
    assert p.moment == (0j, 0j, 1e-30 + 0j)
    m = gfem.MagneticDipole("m", position=(1, 2, 3), moment=(1, 1j, 0))
    assert m.moment == (1 + 0j, 1j, 0j)
    assert repr(m) == (
        "MagneticDipole('m', position=(1.0, 2.0, 3.0), moment=((1+0j), 1j, 0j))"
    )


@pytest.mark.parametrize("cls", [gfem.ElectricDipole, gfem.MagneticDipole])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"position": (0, 0)}, "position must be 3 coordinates"),
        ({"position": (0, 0, math.nan)}, "position must be 3 finite coordinates"),
        ({"moment": (1, 0)}, "moment must be 3 components"),
        ({"moment": (1, 0, math.inf)}, "moment must be 3 finite components"),
    ],
)
def test_a_dipole_refuses_bad_parameters(cls, kwargs, match):
    args = {"position": (0, 0, 0), "moment": (0, 0, 1)} | kwargs
    with pytest.raises(ValueError, match=f"{cls.__name__}.{match}"):
        cls("d", **args)


def test_a_traced_dipole_moment_is_not_differentiable():
    def build(p):
        gfem.ElectricDipole("d", position=(0, 0, 0), moment=(0, 0, p))
        return p

    with pytest.raises(gfem.NonDifferentiableError, match=r"ElectricDipole.moment"):
        jax.grad(build)(1.0)


# --- Driven(sources=...) ------------------------------------------------------------

WAVE = gfem.PlaneWave("x", theta=0.0, polarization="TM")
DIPOLE = gfem.ElectricDipole("d", position=(0, 0, 0), moment=(0, 0, 1e-30))


def test_driven_takes_one_source_or_a_sequence_in_order():
    assert gfem.Driven(5e9, sources=WAVE).sources == (WAVE,)
    assert gfem.Driven(5e9, sources=[DIPOLE, WAVE]).sources == (DIPOLE, WAVE)
    assert gfem.Driven(5e9).sources == ()


def test_excite_none_still_means_every_port_beside_sources():
    assert gfem.Driven(5e9, sources=WAVE).excite is None


def test_no_excited_port_is_accepted_with_sources():
    problem = gfem.Driven(5e9, excite=(), sources=DIPOLE)
    assert problem.excite == ()


def test_no_excited_port_and_no_source_is_refused():
    with pytest.raises(ValueError, match=r"nothing to excite.*sources="):
        gfem.Driven(5e9, excite=())


def test_a_non_source_is_refused():
    with pytest.raises(TypeError, match=r"Driven.sources takes .*PlaneWave.*got str"):
        gfem.Driven(5e9, sources="x")
    with pytest.raises(TypeError, match=r"Driven.sources takes .*got LumpedPort"):
        gfem.Driven(5e9, sources=[WAVE, gfem.LumpedPort("p", "s", "+z")])


def test_source_names_are_unique():
    twin = gfem.ElectricDipole("x", position=(0, 0, 0), moment=(1, 0, 0))
    with pytest.raises(ValueError, match=r"Driven.sources names 'x' twice"):
        gfem.Driven(5e9, sources=[WAVE, twin])


def test_a_source_named_like_an_excited_port_is_refused():
    with pytest.raises(ValueError, match="'x' names both an excited port and a source"):
        gfem.Driven(5e9, excite="x", sources=WAVE)


def test_sources_refuse_a_bloch_wave_vector():
    with pytest.raises(NotImplementedError, match=r"sources.*wave_vector.*periodic"):
        gfem.Driven(5e9, sources=DIPOLE, wave_vector=(0, 0, 0))


@pytest.mark.parametrize(
    "incident",
    [
        WAVE,
        gfem.GaussianBeam("b", waist=1e-6, theta=0.0, polarization="TE"),
        gfem.IncidentField("f", _wave),
    ],
)
def test_the_adaptive_sweep_refuses_an_incident_field(incident):
    with pytest.raises(NotImplementedError, match=r"adaptive.*incident field.*uniform"):
        gfem.Driven([5e9, 6e9], sweep="adaptive", sources=[DIPOLE, incident])


def test_the_adaptive_sweep_takes_dipoles():
    problem = gfem.Driven([5e9, 6e9], sweep="adaptive", sources=DIPOLE)
    assert problem.sources == (DIPOLE,)


def test_sources_refuse_a_2d_polarization():
    with pytest.raises(NotImplementedError, match=r"sources.*3D.*2D"):
        gfem.Driven(5e9, sources=DIPOLE, polarization="Ez")


def test_a_driven_problem_with_sources_shows_them():
    assert "sources=(PlaneWave('x'" in repr(gfem.Driven(5e9, sources=WAVE))


# --- at gfem.solve: what the model shows ------------------------------------------


def _line(tmp_path, *, periodic=False, background=None):
    edges = tuple(np.linspace(0, s, n + 1) for s, n in ((4.0, 2), (2.0, 2), (1.0, 1)))
    path = sm.write_msh22(tmp_path / "line.msh", sm.periodic_cell(edges))
    boundaries = [gfem.PEC("z_min"), gfem.PEC("z_max")]
    if periodic:
        boundaries.append(gfem.Periodic(("y_min", "y_max")))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"cell": gfem.Material()},
        boundaries=boundaries,
        ports=[
            gfem.LumpedPort("p1", "x_min", "+z"),
            gfem.LumpedPort("p2", "x_max", "+z"),
        ],
        background=background,
    )


def test_a_source_named_like_a_port_is_refused_at_solve(tmp_path):
    clash = gfem.ElectricDipole("p2", position=(1e-3, 1e-3, 5e-4), moment=(0, 0, 1))
    with pytest.raises(ValueError, match="'p2' names both a port of the model and a"):
        gfem.solve(_line(tmp_path), gfem.Driven(5e9, excite="p1", sources=clash))


def test_an_incident_field_needs_a_background(tmp_path):
    with pytest.raises(ValueError, match=r"PlaneWave 'x'.*Model\(background="):
        gfem.solve(_line(tmp_path), gfem.Driven(5e9, sources=WAVE))


def test_sources_refuse_a_periodic_pair(tmp_path):
    model = _line(tmp_path, periodic=True)
    with pytest.raises(NotImplementedError, match=r"ElectricDipole 'd'.*periodic"):
        gfem.solve(model, gfem.Driven(5e9, sources=DIPOLE))


def test_solving_a_source_column_is_not_available_yet(tmp_path):
    # the declarations land first; each source kind's solve replaces this
    model = _line(tmp_path, background=gfem.Material())
    with pytest.raises(NotImplementedError, match="cannot solve source columns"):
        gfem.solve(model, gfem.Driven(5e9, sources=WAVE))
