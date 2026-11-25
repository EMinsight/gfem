r"""Three-wave coupling and second-harmonic generation: identities and refusals.

The overlap runs on a homogeneously filled PEC rectangle (WR-90 at 8 GHz and its
second harmonic at 16 GHz, TE10 in both), where the discrete modes obey
:math:`\beta^2 = \varepsilon k_0^2 - k_c^2` and the overlap has the closed form
of `tests/verification/test_three_wave.py`; here the closed form is a sanity
gate at the field's mesh error, and the rest are exact identities of the
implementation. `efficiency` is checked against its closed forms on synthetic
rows, with no solve.
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as sc
from slab2d import write_slab

import gfem
from gfem import _chi2
from gfem.errors import NonDifferentiableError
from gfem.three_wave import _decayed_sinhc_sinc

DATA = Path(__file__).parent / "data"
#: WR-90 (m)
A, B = 22.86e-3, 10.16e-3
N = 1.5
F = np.array([8e9, 8.5e9])
ORDER = gfem.SolverOptions(order=2)
D_YYY = 20e-12
EPS = np.finfo(float).eps
_ROW = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def _d_yyy(scale=1.0):
    d = np.zeros((3, 6))
    d[1, 1] = D_YYY * scale
    return d


def guide_model(mesh, material):
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def crystal(n=N, d=None, **kwargs):
    d = _d_yyy() if d is None else d
    return gfem.Material(eps_r=n**2, second_order_nonlinear_tensor=d, **kwargs)


def mode_of(model, f, order=ORDER, **kwargs):
    return gfem.solve(model, gfem.PortMode("p1", f, **kwargs), options=order)


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def shg(guide_mesh):
    """Two pump rows, their harmonics, and the second-harmonic figures."""
    model = guide_model(guide_mesh, crystal())
    pump = mode_of(model, F)
    harm = mode_of(model, 2 * F)
    return model, pump, harm, gfem.second_harmonic(model, pump, harm)


# --- closed form and the identities of the tensor -----------------------------------


def test_the_filled_te10_follows_its_closed_form_at_the_mesh_error(shg):
    _, pump, harm, sh = shg
    k0 = 2 * np.pi * F / sc.c
    n_p = np.asarray(pump.gamma).imag / k0
    n_h = np.asarray(harm.gamma).imag / (2 * k0)
    z0 = math.sqrt(sc.mu_0 / sc.epsilon_0)
    eta = 256 * k0**2 * D_YYY**2 * z0 / (9 * np.pi**2 * A * B * n_p**2 * n_h)
    # the field's own mesh error: 6e-5 measured at order 2
    np.testing.assert_allclose(np.asarray(sh.eta_norm), eta, rtol=2e-4)
    # TE10 in this gauge has a real overlap (no E_z term acts through d_yyy)
    np.testing.assert_allclose(
        np.asarray(sh.kappa).imag, 0.0, atol=1e-12 * np.abs(sh.kappa).max()
    )
    # normal dispersion in a filled guide: beta(2w) > 2 beta(w)
    assert (np.asarray(sh.delta_beta) > 0).all()
    np.testing.assert_allclose(
        np.asarray(sh.poling_period), 2 * np.pi / np.asarray(sh.delta_beta), rtol=1e-15
    )


def test_the_tensor_is_symmetric_in_its_two_amplitudes_and_names_its_waves(shg):
    model, pump, harm, _ = shg
    c = gfem.three_wave_coupling(model, [pump, harm])
    assert c.port == "p1"
    assert c.waves == ((1, F[0]), (1, F[1]), (1, 2 * F[0]), (1, 2 * F[1]))
    k = np.asarray(c.kappa)
    assert k.shape == (4, 4, 4)
    assert k.dtype == np.complex128
    # two sums of the same 27 terms per point in a different order: round-off
    # of a few eps per point, 1e-13 of the largest entry covers the sum over the
    # points
    np.testing.assert_allclose(
        k, np.swapaxes(k, 1, 2), rtol=0, atol=1e-13 * np.abs(k).max()
    )


def test_second_harmonic_kappa_is_the_coupling_entry_of_each_row(shg):
    model, pump, harm, sh = shg
    c = np.asarray(gfem.three_wave_coupling(model, [pump, harm]).kappa)
    # the same overlap summed in a different order (a row at a time against one
    # einsum): round-off, 1e-12 relative
    for i in range(2):
        assert c[2 + i, i, i] == pytest.approx(complex(sh.kappa[i]), rel=1e-12)


def test_the_result_names_its_port_modes_and_frequencies(shg):
    _, pump, harm, sh = shg
    assert (sh.port, sh.modes) == ("p1", (1, 1))
    np.testing.assert_array_equal(sh.frequency, F)
    np.testing.assert_array_equal(sh.pump_gamma, pump.gamma)
    np.testing.assert_array_equal(sh.harmonic_gamma, harm.gamma)
    assert sh.kappa.shape == (2,)


def test_the_coupling_is_exactly_linear_in_the_tensor_and_zero_off_its_entries(
    guide_mesh, shg
):
    _, pump, harm, sh = shg
    twice = guide_model(guide_mesh, crystal(d=_d_yyy(2.0)))
    again = gfem.second_harmonic(twice, pump, harm)
    np.testing.assert_allclose(again.kappa, 2 * sh.kappa, rtol=1e-13)
    # d_xxx alone: TE10 has no E_x, so nothing couples
    other = np.zeros((3, 6))
    other[0, 0] = D_YYY
    zero = gfem.second_harmonic(guide_model(guide_mesh, crystal(d=other)), pump, harm)
    # the discrete TE10 has a stray E_x of the mesh's size: 1.7e-9 of kappa measured
    np.testing.assert_allclose(zero.kappa, 0.0, atol=1e-7 * np.abs(sh.kappa).max())


# --- frames, fields and blends -----------------------------------------------------


def test_a_tensor_with_axes_equals_the_pre_rotated_tensor_without(guide_mesh, shg):
    _, pump, harm, _ = shg
    q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    axes = q * np.sign(np.linalg.det(q))
    d = np.random.default_rng(6).normal(size=(3, 6)) * 1e-12
    full = d[:, _ROW]
    r = axes.T
    rotated = np.einsum("pl,qm,sn,lmn->pqs", r, r, r, full)[
        :, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]
    ]
    with_axes = gfem.second_harmonic(
        guide_model(guide_mesh, crystal(d=d, axes=tuple(map(tuple, axes)))), pump, harm
    )
    without = gfem.second_harmonic(
        guide_model(guide_mesh, crystal(d=rotated)), pump, harm
    )
    scale = np.abs(without.kappa).max()
    assert scale > 0
    # the rotated entries differ by at most 2 * 27 * 3 eps of the largest entry
    # (the helper and the einsum above sum in another order), and the overlap sums
    # a few thousand such terms of both signs: 1e-12 of the largest kappa
    np.testing.assert_allclose(
        with_axes.kappa, without.kappa, rtol=0, atol=1e-12 * scale
    )


def test_a_uniform_field_equals_its_material(guide_mesh, shg):
    _, pump, harm, sh = shg
    field = gfem.MaterialField.uniform("air", crystal())
    got = gfem.second_harmonic(guide_model(guide_mesh, field), pump, harm)
    np.testing.assert_allclose(got.kappa, sh.kappa, rtol=1e-13)


def test_a_blend_is_linear_in_the_density_and_an_end_without_the_tensor_counts_zero(
    guide_mesh, shg
):
    _, pump, harm, sh = shg
    n_el = len(guide_mesh.elements("air"))
    for rho in (0.25, 1.0, 0.0):
        field = gfem.MaterialField.interpolate(
            jnp.full(n_el, rho),
            gfem.Material(eps_r=N**2),
            crystal(),
            scheme="linear",
        )
        got = gfem.second_harmonic(guide_model(guide_mesh, field), pump, harm)
        # rho = 0 is the end without the tensor: zero
        np.testing.assert_allclose(got.kappa, rho * sh.kappa, rtol=1e-12, atol=1e-30)


def test_the_gradient_in_the_tensor_is_exact_by_linearity(guide_mesh, shg):
    _, pump, harm, sh = shg

    def objective(scale):
        field = gfem.MaterialField(
            eps_r=N**2, second_order_nonlinear_tensor=scale * _d_yyy()
        )
        model = guide_model(guide_mesh, field)
        return jnp.real(gfem.second_harmonic(model, pump, harm).kappa[0])

    value, grad = jax.value_and_grad(objective)(1.0)
    # kappa is linear in the tensor: the derivative is the value, to the
    # round-off of one overlap
    assert float(grad) == pytest.approx(float(value), rel=1e-12)
    assert float(value) == pytest.approx(float(sh.kappa[0].real), rel=1e-13)


# --- refusals ----------------------------------------------------------------------


def test_a_model_without_the_tensor_names_the_regions(guide_mesh, shg):
    _, pump, harm, _ = shg
    plain = guide_model(guide_mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(
        ValueError, match=r"wave port 'p1'.*second_order_nonlinear_tensor.*'air'"
    ):
        gfem.second_harmonic(plain, pump, harm)
    with pytest.raises(ValueError, match="second_order_nonlinear_tensor"):
        gfem.three_wave_coupling(plain, [pump, harm])


def test_a_result_without_a_field_or_a_non_result_is_a_type_error(shg):
    model, pump, harm, _ = shg
    bare = dataclasses.replace(pump, e_field=None)
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.second_harmonic(model, bare, harm)
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.second_harmonic(model, "mode", harm)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.three_wave_coupling(model, "mode")  # ty: ignore[invalid-argument-type]


# --- the Ez line port ---------------------------------------------------------------


SLAB_F = sc.c / 1.55e-6
SLAB_ORDER = gfem.SolverOptions(order=3)


def ez(model, f, order=SLAB_ORDER):
    return mode_of(model, f, order, polarization="Ez")


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=1.0,
        h_core=0.055,
        h_clad=0.2,
    )

    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")

    def build(d, n=3.476):
        return gfem.Model(
            mesh,
            materials={
                "core": gfem.Material(eps_r=n**2, second_order_nonlinear_tensor=d),
                "clad": gfem.Material(eps_r=1.444**2),
            },
            ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
        )

    d = np.zeros((3, 6))
    d[2, 2] = 25e-12
    model = build(d)
    pump = ez(model, [SLAB_F])
    harm = ez(model, [2 * SLAB_F])
    return model, pump, harm, build


def test_an_ez_line_port_couples_through_d_zzz_only(slab):
    model, pump, harm, build = slab
    sh = gfem.second_harmonic(model, pump, harm)
    assert float(sh.eta_norm[0]) > 0
    # d_xxx of the global frame does not act on Ez
    off = np.zeros((3, 6))
    off[0, 0] = 25e-12
    zero = gfem.second_harmonic(build(off), pump, harm)
    assert float(jnp.abs(zero.kappa[0])) == 0.0
    # linear in d_zzz
    twice = np.zeros((3, 6))
    twice[2, 2] = 50e-12
    again = gfem.second_harmonic(build(twice), pump, harm)
    assert complex(again.kappa[0]) == pytest.approx(2 * complex(sh.kappa[0]), rel=1e-13)


def test_an_hz_line_port_is_refused(slab):
    model, _, harm, _ = slab
    hz = mode_of(model, [SLAB_F], SLAB_ORDER, polarization="Hz")
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.second_harmonic(model, hz, harm)


def test_the_pair_form_of_one_row_is_the_row_of_the_second_harmonic(slab):
    model, pump, harm, _ = slab
    row = gfem.second_harmonic(model, pump, harm)
    pair = gfem.three_wave_coupling(model, [pump, harm])
    # one overlap summed in two orders (a row at a time against the wave map):
    # round-off, 1e-12 relative
    assert complex(pair.kappa[1, 0, 0]) == pytest.approx(
        complex(row.kappa[0]), rel=1e-12
    )


def test_the_pairing_of_the_two_results_is_checked(slab):
    model, pump, _, _ = slab
    two = ez(model, [2 * SLAB_F, 2.2 * SLAB_F])
    with pytest.raises(ValueError, match=r"pump has 1 rows but the harmonic 2"):
        gfem.second_harmonic(model, pump, two)
    detuned = ez(model, [2 * SLAB_F * (1 + 1e-7)])
    with pytest.raises(ValueError, match=r"row 0 of the harmonic.*not twice"):
        gfem.second_harmonic(model, pump, detuned)
    # inside the tolerance (a wavelength-to-frequency round trip) it is accepted
    near = ez(model, [2 * SLAB_F * (1 + 1e-10)])
    gfem.second_harmonic(model, pump, near)


def test_waves_from_two_models_of_one_mesh_are_accepted_another_mesh_is_not(slab):
    model, pump, harm, build = slab
    sh = gfem.second_harmonic(model, pump, harm)
    # the harmonic solved on a model with another permittivity (a crystal's
    # eps is one constant per frequency); the tensor is read from the model passed
    other = build(model.materials["core"].second_order_nonlinear_tensor, n=3.5)
    harm2 = ez(other, [2 * SLAB_F])
    mixed = gfem.second_harmonic(model, pump, harm2)
    assert np.isfinite(np.asarray(mixed.kappa)).all()
    assert not np.allclose(mixed.kappa, sh.kappa, rtol=1e-3)
    assert gfem.three_wave_coupling(model, [pump, harm2]).kappa.shape == (2, 2, 2)
    nodes = jnp.asarray(np.asarray(model.mesh.imported_nodes) * 1.01)
    mesh = jax.tree.unflatten(jax.tree.structure(model.mesh), [nodes])
    with pytest.raises(ValueError, match=r"not the one wave port 'p1'"):
        gfem.second_harmonic(model.replace(mesh=mesh), pump, harm2)


def test_waves_of_different_ports_or_orders_are_refused(slab):
    model, pump, _, _ = slab
    ports = gfem.Model(
        model.mesh,
        materials=model.materials,
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
    other = gfem.solve(
        ports,
        gfem.PortMode("p2", [SLAB_F], polarization="Ez"),
        options=SLAB_ORDER,
    )
    with pytest.raises(ValueError, match=r"share one port.*'p1'.*'p2'"):
        gfem.three_wave_coupling(ports, [pump, other])
    third = ez(model, [SLAB_F], order=gfem.SolverOptions(order=2))
    with pytest.raises(ValueError, match=r"share one port, element order"):
        gfem.three_wave_coupling(model, [pump, third])


def test_traced_node_coordinates_raise_by_name(slab):
    model, _, _, _ = slab
    nodes = np.array(model.mesh.imported_nodes)
    low = gfem.SolverOptions(order=2)

    def objective(scale):
        mesh = jax.tree.unflatten(
            jax.tree.structure(model.mesh), [jnp.asarray(nodes) * scale]
        )
        moved = model.replace(mesh=mesh)
        pump, harm = (ez(moved, [k * SLAB_F], low) for k in (1, 2))
        return jnp.real(gfem.second_harmonic(moved, pump, harm).kappa[0])

    with pytest.raises(NonDifferentiableError, match="second_harmonic"):
        jax.grad(objective)(1.0)


def test_the_gradient_through_the_pair_form_matches_the_second_harmonic_row(slab):
    model, pump, harm, build = slab
    d = model.materials["core"].second_order_nonlinear_tensor

    def via_pair(scale):
        m = build(scale * d)
        return jnp.real(gfem.three_wave_coupling(m, [pump, harm]).kappa[1, 0, 0])

    def via_row(scale):
        m = build(scale * d)
        return jnp.real(gfem.second_harmonic(m, pump, harm).kappa[0])

    g_pair, g_row = jax.grad(via_pair)(1.0), jax.grad(via_row)(1.0)
    # both are linear in the tensor: the derivative is the value, to the
    # round-off of one overlap, so each also equals the value
    assert float(g_pair) == pytest.approx(float(via_pair(1.0)), rel=1e-12)
    assert float(g_pair) == pytest.approx(float(g_row), rel=1e-12)


def test_a_layer_grid_blend_carrying_the_tensor_is_refused(slab):
    model, _, _, _ = slab
    grid = gfem.design.LayerGrid(model.mesh, "core", pitch=0.05e-6)
    rho = grid.to_elements(np.ones(grid.shape))
    with pytest.raises(
        ValueError, match=r"second_order_nonlinear_tensor, but a layer density"
    ):
        gfem.MaterialField.interpolate(
            rho, model.materials["clad"], model.materials["core"]
        )


# --- efficiency ---------------------------------------------------------------------


def _rows(kappa=1.3e3, b1=1.0e7, db=2.5e3, a1=0.0, a2=0.0):
    """One synthetic row: the pump's and harmonic's propagation constants."""
    return gfem.SecondHarmonic(
        port="p1",
        modes=(1, 1),
        frequency=np.array([2e14]),
        kappa=jnp.array([kappa * np.exp(0.4j)]),
        pump_gamma=jnp.array([a1 + 1j * b1]),
        harmonic_gamma=jnp.array([a2 + 1j * (2 * b1 + db)]),
    )


def test_a_zero_length_or_a_zero_duty_cycle_gives_zero():
    sh = _rows()
    assert float(sh.efficiency(0.0, 1.0)[0]) == 0.0
    assert float(sh.efficiency(1e-3, 1.0, period=1e-5, duty_cycle=0.0)[0]) == 0.0
    # order 2 at D = 1/2: sin(pi) is 1.2e-16, squared and divided by (pi m)^2/4
    # it is below 1e-33 of the unpoled value
    full = float(sh.efficiency(1e-3, 1.0)[0])
    assert float(sh.efficiency(1e-3, 1.0, period=1e-5, order=2)[0]) < 1e-30 * full


def test_perfect_phase_matching_gives_four_over_pi_squared_of_the_quadratic_law():
    sh = _rows()
    eta, p, length = float(sh.eta_norm[0]), 0.5, 3e-3
    period = float(sh.poling_period[0])
    got = float(sh.efficiency(length, p, period=period)[0])
    # delta is the round-off of |db| - 2 pi / (2 pi / |db|), 1e-16 of db, so
    # the sinc is 1 - (delta L)^2 / 24 = 1 - 1e-26
    assert got == pytest.approx(4 / np.pi**2 * eta * p * length**2, rel=1e-13)
    third = float(
        sh.efficiency(length, p, period=3 * period, order=3, duty_cycle=1 / 6)[0]
    )
    # |G_3|^2 = (2 / (3 pi))^2 sin(pi / 2)^2 at D = 1/6
    assert third == pytest.approx(
        (2 / (3 * np.pi)) ** 2 * eta * p * length**2, rel=1e-13
    )


def test_the_unpoled_law_is_sinc_squared_and_zero_at_a_whole_wave_of_mismatch():
    sh = _rows()
    eta, db = float(sh.eta_norm[0]), float(sh.delta_beta[0])
    for length in (1e-4, 7e-4, 2e-3):  # delta L / 2 = 0.125, 0.9, 2.5: both branches
        want = eta * length**2 * np.sinc(db * length / 2 / np.pi) ** 2
        assert float(sh.efficiency(length, 1.0)[0]) == pytest.approx(want, rel=1e-13)
    length = 2 * np.pi / db
    # sin(pi) = 1.2e-16: the value is below 1e-32 of the quadratic law
    assert float(sh.efficiency(length, 1.0)[0]) < 1e-30 * eta * length**2


def test_the_lossy_formula_matches_its_closed_form_and_reduces_to_sinc_squared():
    a1, a2 = 5.0, 12.0
    sh = _rows(a1=a1, a2=a2)
    eta, db = float(sh.eta_norm[0]), float(sh.delta_beta[0])
    da, length, p = a2 - 2 * a1, 2e-3, 0.3
    want = (
        eta
        * p
        * np.exp(-(a2 + 2 * a1) * length)
        * (np.sinh(da * length / 2) ** 2 + np.sin(db * length / 2) ** 2)
        / ((da / 2) ** 2 + (db / 2) ** 2)
    )
    assert float(sh.efficiency(length, p)[0]) == pytest.approx(want, rel=1e-12)
    # the lossless limit of the same call
    lossless = _rows()
    want0 = eta * p * length**2 * np.sinc(db * length / 2 / np.pi) ** 2
    assert float(lossless.efficiency(length, p)[0]) == pytest.approx(want0, rel=1e-13)


def test_it_is_finite_at_zero_mismatch_and_zero_loss_difference_with_its_gradient():
    # Delta alpha = 2 alpha_p - 2 alpha_p = 0 and Delta beta = 0
    a1 = 3.0

    def eff(length, power, db):
        sh = gfem.SecondHarmonic(
            port="p1",
            modes=(1, 1),
            frequency=np.array([2e14]),
            kappa=jnp.array([1.3e3 + 0j]),
            pump_gamma=jnp.array([a1 + 1e7j]),
            harmonic_gamma=jnp.array([2 * a1 + 1j * (2e7 + db)]),
        )
        return sh.efficiency(length, power)[0]

    length, power = 2e-3, 0.4
    eta = 1.3e3**2
    decay = np.exp(-4 * a1 * length)
    assert float(eff(length, power, 0.0)) == pytest.approx(
        eta * power * length**2 * decay, rel=1e-13
    )
    g_len, g_pow, g_db = jax.grad(eff, argnums=(0, 1, 2))(length, power, 0.0)
    assert float(g_len) == pytest.approx(
        eta * power * decay * (2 * length - 4 * a1 * length**2), rel=1e-12
    )
    assert float(g_pow) == pytest.approx(eta * length**2 * decay, rel=1e-13)
    # even in the mismatch: the derivative at zero vanishes, and is finite
    assert abs(float(g_db)) < 1e-20


@pytest.mark.parametrize(("a_pump", "a_harm"), [(0.0, 1e4), (1e4, 0.0)])
def test_a_large_loss_difference_stays_finite_with_its_gradient(a_pump, a_harm):
    # |Delta alpha| L / 2 = 500 or 1000: sinh^2 alone overflows (above 355) while
    # the decay underflows. There e^{-(a_h + 2 a_p) L} sinh^2(u) is 1/4 to
    # e^{-1000} and sin^2 is damped to zero, so the value is
    # kappa^2 P / 4 / ((Delta alpha / 2)^2 + (Delta beta / 2)^2) exactly to the
    # round-off of the quotient
    sh = gfem.SecondHarmonic(
        port="p1",
        modes=(1, 1),
        frequency=np.array([2e14]),
        kappa=jnp.array([1.0e3 + 0j]),
        pump_gamma=jnp.array([a_pump + 1e7j]),
        harmonic_gamma=jnp.array([a_harm + 1j * (2e7 + 1e3)]),
    )
    da, db = a_harm - 2 * a_pump, 1e3
    want = 1e6 / 4 / ((da / 2) ** 2 + (db / 2) ** 2)
    length = 0.1
    assert float(sh.efficiency(length, 1.0)[0]) == pytest.approx(want, rel=1e-13)
    g_len, g_pow = jax.grad(lambda x, y: sh.efficiency(x, y)[0], argnums=(0, 1))(
        length, 1.0
    )
    # linear in the power; constant in the length up to e^{-1000}
    assert float(g_pow) == pytest.approx(want, rel=1e-13)
    assert abs(float(g_len)) < 1e-100


def test_duty_cycle_and_order_default_with_a_period_and_are_refused_without():
    sh = _rows()
    same = sh.efficiency(1e-3, 1.0, period=1e-5)
    explicit = sh.efficiency(1e-3, 1.0, period=1e-5, duty_cycle=0.5, order=1)
    np.testing.assert_array_equal(same, explicit)
    for kwargs in ({"duty_cycle": 0.3}, {"order": 1}, {"duty_cycle": 0.5, "order": 3}):
        with pytest.raises(ValueError, match="give the period"):
            sh.efficiency(1e-3, 1.0, **kwargs)


def test_a_complex_argument_is_a_type_error_scalar_or_array():
    sh = _rows()
    for bad in (1e-3 + 0j, np.array([1e-3 + 0j, 2e-3])):
        with pytest.raises(TypeError, match="length must be real"):
            sh.efficiency(bad, 1.0)
    with pytest.raises(TypeError, match="power must be real"):
        sh.efficiency(1e-3, 1.0 + 0j)
    with pytest.raises(TypeError, match="order must be real"):
        sh.efficiency(1e-3, 1.0, period=1e-5, order=1 + 0j)


def test_the_series_and_the_quotient_agree_where_they_meet():
    # r2 = u^2 + v^2 = 0.25: the series is truncated after k = 9, whose next
    # term is c_10 * 10 * 0.25^9 = 8e-17; the quotient's round-off is a few eps
    for angle in (0.0, 0.7, 1.4):
        for r2 in (0.25 - 1e-9, 0.25 + 1e-9):
            u, v = np.sqrt(r2) * np.cos(angle), np.sqrt(r2) * np.sin(angle)
            direct = (np.sinh(u) ** 2 + np.sin(v) ** 2) / r2
            assert float(_decayed_sinhc_sinc(u, v, 0.0)) == pytest.approx(
                direct, rel=1e-14
            )
    assert float(_decayed_sinhc_sinc(0.0, 0.0, 0.0)) == 1.0


def test_it_broadcasts_against_the_rows_and_jits_with_the_result_as_an_argument():
    sh = gfem.SecondHarmonic(
        port="p1",
        modes=(1, 1),
        frequency=np.array([2e14, 2.1e14]),
        kappa=jnp.array([1e3 + 0j, 2e3 + 0j]),
        pump_gamma=jnp.array([1e7j, 1.05e7j]),
        harmonic_gamma=jnp.array([2.0001e7j, 2.1002e7j]),
    )
    lengths = jnp.array([1e-3, 2e-3, 3e-3])[:, None]
    out = sh.efficiency(lengths, 0.1)
    assert out.shape == (3, 2)
    jitted = jax.jit(lambda s, length: s.efficiency(length, 0.1))(sh, lengths)
    np.testing.assert_allclose(jitted, out, rtol=1e-13)
    # the same under jit when the result is a closure and the period is traced
    traced = jax.jit(lambda p: sh.efficiency(2e-3, 0.1, period=p))(1e-5)
    np.testing.assert_allclose(
        traced, sh.efficiency(2e-3, 0.1, period=1e-5), rtol=1e-13
    )


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"length": -1e-3, "power": 1.0}, ["length must be >= 0"]),
        ({"length": 1e-3, "power": -1.0}, ["power must be >= 0"]),
        ({"length": 1e-3, "power": 1.0, "period": 0.0}, ["period must be > 0"]),
        ({"length": 1e-3, "power": 1.0, "period": -1e-6}, ["period must be > 0"]),
        (
            {"length": 1e-3, "power": 1.0, "period": 1e-5, "duty_cycle": 1.5},
            ["duty_cycle must be in [0, 1]"],
        ),
        (
            {"length": 1e-3, "power": 1.0, "period": 1e-5, "order": 0},
            ["order must be >= 1"],
        ),
        (
            {"length": 1e-3, "power": 1.0, "period": 1e-5, "order": 1.5},
            ["order must be an integer"],
        ),
    ],
)
def test_a_concrete_argument_out_of_range_names_itself(kwargs, words):
    with pytest.raises(ValueError) as e:
        _rows().efficiency(**kwargs)
    for word in words:
        assert word in str(e.value)


# --- repr and pytree ----------------------------------------------------------------


def test_the_repr_gives_the_ranges_and_a_short_form_under_a_trace(shg):
    model, pump, harm, sh = shg
    text = repr(sh)
    assert text.startswith("SecondHarmonic(port 'p1', modes 1 -> 1, 2 frequencies")
    assert "%/(W cm^2)" in text
    assert "um" in text
    coupling = gfem.three_wave_coupling(model, [pump, harm])
    assert repr(coupling).startswith("ThreeWaveCoupling(port 'p1', 4 waves, |kappa|")
    seen = []

    def objective(scale):
        field = gfem.MaterialField(
            eps_r=N**2, second_order_nonlinear_tensor=scale * _d_yyy()
        )
        m = guide_model(model.mesh, field)
        result = gfem.second_harmonic(m, pump, harm)
        seen.extend([repr(result), repr(gfem.three_wave_coupling(m, [pump, harm]))])
        return jnp.real(result.kappa[0])

    jax.grad(objective)(1.0)
    assert seen == [
        "SecondHarmonic(port 'p1', modes 1 -> 1, 2 frequencies)",
        "ThreeWaveCoupling(port 'p1', 4 waves)",
    ]


def test_the_results_are_pytrees_and_a_function_of_a_leaf_is_differentiable(shg):
    model, pump, harm, sh = shg
    leaves, tree = jax.tree.flatten(sh)
    assert len(leaves) == 3
    again = jax.tree.unflatten(tree, leaves)
    assert (again.port, again.modes) == ("p1", (1, 1))
    np.testing.assert_array_equal(again.frequency, sh.frequency)
    np.testing.assert_array_equal(again.kappa, sh.kappa)
    doubled = jax.tree.map(lambda x: 2 * x, sh)
    np.testing.assert_allclose(doubled.kappa, 2 * sh.kappa, rtol=1e-15)
    c = gfem.three_wave_coupling(model, [pump, harm])
    cl, ct = jax.tree.flatten(c)
    assert len(cl) == 1
    back = jax.tree.unflatten(ct, cl)
    assert back.waves == c.waves
    assert back.port == "p1"
    np.testing.assert_array_equal(back.kappa, c.kappa)


def test_the_helper_contracts_as_the_full_tensor_does():
    d = _d_yyy()
    f = np.array([0.0, 1.0 + 1j, 0.0])
    got = _chi2.contract(d, f, f)
    np.testing.assert_allclose(got, [0.0, D_YYY * (1 + 1j) ** 2, 0.0], rtol=1e-15)
