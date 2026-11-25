r"""The Kerr nonlinear coefficient of a port mode: closed forms and discrete identities.

On a homogeneously filled PEC guide the discrete modes obey
:math:`\beta_h^2 = \varepsilon k_0^2 - k_{c,h}^2` with a cutoff fixed by the
mesh, and the mode's shape does not change with frequency. The coefficient is
then a closed form in the computed index, which removes the method error in
:math:`\beta`; what is left is the field's own mesh error (measured per gate).
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases
import pytest
import scipy.constants as sc
from slab2d import write_slab

import gfem
from gfem import kerr
from gfem.errors import NonDifferentiableError

DATA = Path(__file__).parent / "data"
#: WR-90 (m): the guide's section, not the extent of its points
A, B = 22.86e-3, 10.16e-3
N, N2, TPA = 1.5, 2.6e-20, 5e-12
F = 8e9
ORDER = gfem.SolverOptions(order=2)
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40e9, damping=0.0)],
)


def k0_of(f):
    return 2 * np.pi * np.asarray(f) / sc.c


def guide_model(mesh, material):
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def mode_of(model, f=F, order=ORDER, **kwargs):
    return gfem.solve(model, gfem.PortMode("p1", f, **kwargs), options=order)


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def filled(guide_mesh):
    """The filled guide at 8 GHz with n2 and TPA, its mode and coefficient."""
    material = gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA)
    model = guide_model(guide_mesh, material)
    mode = mode_of(model, [F])
    return model, mode, gfem.kerr_nonlinearity(model, mode)


# --- closed forms --------------------------------------------------------------


def test_the_filled_te10_follows_its_closed_form(filled):
    _, mode, kn = filled
    k0 = k0_of(F)
    n_eff = float(mode.gamma.imag[0] / k0)
    reference = 1.5 * k0 * N2 / (A * B) * (N / n_eff) ** 2
    # the field's own mesh error: 7.0e-6 measured at order 2 (3.4e-8 at order 3)
    assert kn.gamma_nl[0].real == pytest.approx(reference, rel=1e-5)


def test_the_result_names_its_port_mode_and_frequencies(filled):
    _, _, kn = filled
    assert (kn.port, kn.mode) == ("p1", 1)
    np.testing.assert_array_equal(kn.frequency, [F])
    assert kn.gamma_nl.shape == (1,)
    assert kn.gamma_nl.dtype == jnp.complex128


@pytest.mark.parametrize("fill", ["constant", "lorentz"])
def test_every_row_of_a_grid_follows_the_row_scaling(guide_mesh, fill):
    # 41 rows, two exact: the others are reduced rows. The homogeneous fill's
    # discrete shape does not depend on frequency and a reduced beta is exact
    # to 2.5e-14, so the ratio of rows is a closed form of the computed indices
    # (and, for the law, of Re eps_r); round-off over a few operations is 1e-15,
    # the gate 1e-10 leaves the margin for solver noise in beta.
    f = np.linspace(7e9, 8.5e9, 41)
    eps = LORENTZ if fill == "lorentz" else N**2
    model = guide_model(guide_mesh, gfem.Material(eps_r=eps, nonlinear_index=N2))
    mode = mode_of(model, f)
    assert (~np.asarray(mode.solved)).any()
    gamma = np.asarray(gfem.kerr_nonlinearity(model, mode).gamma_nl)
    n_eff = np.asarray(mode.gamma.imag) / k0_of(f)
    re_eps = np.real(np.asarray(LORENTZ(f))) if fill == "lorentz" else np.full(41, N**2)
    expected = gamma[0] * (f / f[0]) * (n_eff[0] / n_eff) ** 2 * re_eps / re_eps[0]
    np.testing.assert_allclose(gamma, expected, rtol=1e-10)


def test_two_photon_absorption_is_the_imaginary_part(filled):
    _, _, kn = filled
    g = complex(kn.gamma_nl[0])
    # the integrand is shared: the ratio is exact up to two roundings
    assert g.imag >= 0
    assert g.imag / g.real == pytest.approx(TPA / (2 * k0_of(F) * N2), rel=1e-12)


def test_a_material_with_two_photon_absorption_only_has_no_kerr_part(guide_mesh):
    model = guide_model(
        guide_mesh, gfem.Material(eps_r=N**2, two_photon_absorption=TPA)
    )
    g = complex(gfem.kerr_nonlinearity(model, mode_of(model)).gamma_nl[0])
    assert g.real == 0.0
    assert g.imag > 0


def test_a_lossy_fill_changes_gamma_through_the_mode_normalization_only(
    guide_mesh, filled
):
    # tan_delta adds an imaginary part to eps_r only, and the integrand stays
    # real for a real chi. The 1 W normalization of a lossy mode is by the
    # modulus of its complex power, so the shape is the lossless one and
    # gamma scales as (n_eff / |n_eff,lossy|)^2; both identities hold to
    # round-off over a few operations, the gate 1e-10 leaves margin.
    _, mode, _ = filled
    lossy = guide_model(
        guide_mesh,
        gfem.Material(eps_r=N**2, tan_delta=0.05, nonlinear_index=N2),
    )
    lossy_mode = mode_of(lossy, [F])
    g = complex(gfem.kerr_nonlinearity(lossy, lossy_mode).gamma_nl[0])
    assert abs(g.imag) <= 1e-14 * abs(g.real)
    lossless = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    g0 = complex(gfem.kerr_nonlinearity(lossless, mode_of(lossless, [F])).gamma_nl[0])
    n0 = complex(mode.n_eff[0]).real
    n_lossy = complex(lossy_mode.n_eff[0])
    assert abs(n_lossy.imag) > 1e-2  # the case is lossy
    assert g.real / g0.real == pytest.approx((n0 / abs(n_lossy)) ** 2, rel=1e-10)


# --- gradients -----------------------------------------------------------------


def _field_model(mesh, eps, n2, beta):
    material = gfem.MaterialField(
        eps_r=eps,
        nonlinear_index=n2,
        two_photon_absorption=beta,
        region="air",
    )
    return guide_model(mesh, material)


def test_gamma_differentiates_in_the_permittivity_and_the_two_coefficients(
    guide_mesh, filled
):
    _, mode, kn = filled
    g = complex(kn.gamma_nl[0])
    n_eff = float(np.real(mode.n_eff[0]))

    def gamma(eps, n2, beta):
        model = _field_model(guide_mesh, eps, n2, beta)
        return gfem.kerr_nonlinearity(model, mode_of(model, [F])).gamma_nl[0]

    args = (jnp.asarray(N**2), jnp.asarray(N2), jnp.asarray(TPA))

    def real(*a):
        return jnp.real(gamma(*a))

    def imag(*a):
        return jnp.imag(gamma(*a))

    value, d_real = jax.value_and_grad(real, argnums=(0, 1, 2))(*args)
    d_imag = jax.grad(imag, argnums=(0, 1, 2))(*args)
    # the traced vector gives the eager value: the same operations, up to the
    # solve's own round-off (1e-14), the gate 1e-12
    assert float(value) == pytest.approx(g.real, rel=1e-12)
    # on a homogeneous fill the discrete cutoff is fixed, so
    # d ln(gamma) / d eps = 1 / eps - 1 / n_eff^2; the derivative of the
    # discrete problem is exact up to the adjoint solve's round-off (1e-10)
    slope = 1 / N**2 - 1 / n_eff**2
    assert float(d_real[0]) == pytest.approx(g.real * slope, rel=1e-8)
    assert float(d_imag[0]) == pytest.approx(g.imag * slope, rel=1e-8)
    # exact: gamma is linear in each coefficient
    assert float(d_real[1]) == pytest.approx(g.real / N2, rel=1e-12)
    assert float(d_imag[2]) == pytest.approx(g.imag / TPA, rel=1e-12)
    assert float(d_real[2]) == pytest.approx(0.0, abs=1e-12 * g.real / TPA)
    assert float(d_imag[1]) == pytest.approx(0.0, abs=1e-12 * g.imag / N2)


# --- a perfectly matched layer -------------------------------------------------


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    path = pml_cases.strip_guide(
        tmp_path_factory.mktemp("strip") / "strip.msh", 0.4, quarter=False, length=0.15
    )
    model = pml_cases.strip_model(path)
    materials = dict(model.materials)
    materials["core"] = dataclasses.replace(materials["core"], nonlinear_index=6e-18)
    model = model.replace(materials=materials)
    mode = gfem.solve(model, gfem.PortMode("p1", pml_cases.STRIP_F), options=ORDER)
    return model, mode


def test_a_property_on_the_layer_alone_changes_nothing(strip):
    model, mode = strip
    base = complex(gfem.kerr_nonlinearity(model, mode).gamma_nl[0])
    materials = dict(model.materials)
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-17)
    on_layer = model.replace(materials=materials)
    # the layer's triangles carry weight 0: every term of the sum is the same
    got = complex(gfem.kerr_nonlinearity(on_layer, mode).gamma_nl[0])
    assert got == pytest.approx(base, rel=1e-14)


def test_a_property_only_on_the_layer_is_refused(strip):
    model, mode = strip
    materials = {
        name: dataclasses.replace(m, nonlinear_index=None)
        for name, m in model.materials.items()
    }
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-17)
    with pytest.raises(ValueError, match=r"wave port 'p1'.*layer"):
        gfem.kerr_nonlinearity(model.replace(materials=materials), mode)


# --- the columns -----------------------------------------------------------------


def _rule_points(space, raised=0):
    """Global points ``(n_el, n_q, 2 or 3)`` of the rule's points, in section order.

    NGSolve itself maps the points: a coordinate function set into an
    integration rule space holds the position of each of its points.
    """
    import ngsolve as ngs

    section = space.section
    planar = len(section.normal) == 3
    irs = ngs.comp.IntegrationRuleSpace(space.mesh, order=2 * space.order + raised)
    local = []
    for coordinate in (ngs.x, ngs.y) if planar else (ngs.x,):
        gf = ngs.GridFunction(irs)
        gf.Set(coordinate, ngs.VOL)
        local.append(gf.vec.FV().NumPy().copy())
    owner = np.array(
        [list(irs.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(space.mesh.ne)]
    )
    local = np.stack(local, axis=-1)[owner]  # (n_el, n_q, 2 or 1)
    if not planar:
        order = np.empty(len(owner), int)
        order[np.asarray(space.element_segment)] = np.arange(len(owner))
        local = local[order]
    shape = local.shape[:2]
    if planar:
        return section.to_global(local.reshape(-1, 2)).reshape(*shape, 3)
    return section.to_global(local.reshape(-1)).reshape(*shape, 2)


@pytest.mark.parametrize("raised", [0, 2])
def test_the_planar_columns_are_the_field_at_the_rule_points(filled, raised):
    _, mode, _ = filled
    field = mode.e_field
    space = field._space
    cols = kerr._columns(space, raised)
    points = _rule_points(space, raised)
    x = np.asarray(kerr._vector(field))
    waves = kerr._Waves(
        "p1", (1,), np.array([F]), jnp.asarray(x), space.section, cols, None
    )
    got = np.asarray(waves.field())[0]
    expected = np.asarray(field.at(points.reshape(-1, 3)))[0].reshape(got.shape)
    # point values of a polynomial field through two paths: round-off only
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    area = cols.measure.sum()
    assert area == pytest.approx(A * B, rel=1e-12)


def test_the_integral_is_exact_at_the_rule_degree(filled, monkeypatch):
    model, mode, kn = filled
    original = kerr._columns
    monkeypatch.setattr(kerr, "_columns", lambda space: original(space, 2))
    higher = gfem.kerr_nonlinearity(model, mode)
    # degree 4p is exact for a degree-p field and a material constant per
    # element: the two rules differ by summation order only (1e-16 relative)
    np.testing.assert_allclose(higher.gamma_nl, kn.gamma_nl, rtol=1e-13)


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=1.0,
        h_core=0.055,
        h_clad=0.2,
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(eps_r=3.476**2, nonlinear_index=6e-18),
            "clad": gfem.Material(eps_r=1.444**2, nonlinear_index=2.6e-20),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )
    mode = mode_of(
        model, sc.c / 1.55e-6, polarization="Ez", order=gfem.SolverOptions(order=3)
    )
    return model, mode


@pytest.mark.parametrize("raised", [0, 2])
def test_the_line_columns_are_the_field_at_the_rule_points(slab, raised):
    _, mode = slab
    field = mode.e_field
    space = field._space
    cols = kerr._columns(space, raised)
    points = _rule_points(space, raised)
    section = space.section
    waves = kerr._Waves(
        "p1", (1,), np.array([1.0]), kerr._vector(field), section, cols, None
    )
    got = np.asarray(waves.field())[0, ..., 0]
    expected = np.asarray(field.at(points.reshape(-1, 2)))[0].reshape(got.shape)
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    length = np.ptp(section.points) * section.scale
    assert cols.measure.sum() == pytest.approx(length, rel=1e-12)


def test_an_ez_line_port_gives_its_slab_value_per_watt(slab):
    model, mode = slab
    kn = gfem.kerr_nonlinearity(model, mode)
    g = complex(kn.gamma_nl[0])
    assert g.imag == 0.0
    # the order of magnitude of a 220 nm silicon slab TE0: 1e-4 1/W, and the
    # exact closed form is the verification tier's
    assert 5e-5 < g.real < 2e-4


# --- refusals ----------------------------------------------------------------------


def test_a_model_without_a_property_names_the_port(guide_mesh):
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(ValueError, match=r"wave port 'p1'.*layer"):
        gfem.kerr_nonlinearity(model, mode_of(model))


def test_an_anisotropic_region_with_a_property_is_refused(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, thickness=0, quarter=False, length=0.15
    )
    base = pml_cases.strip_model(path)
    aniso = gfem.Material(eps_r=(2.1, 2.1, 2.3))
    core = gfem.Material(eps_r=pml_cases.STRIP_CORE**2, nonlinear_index=6e-18)
    f = pml_cases.STRIP_F

    def materials(clad):
        return base.replace(materials={"core": core, "clad": clad})

    works = materials(aniso)  # without the property it contributes nothing
    mode = gfem.solve(works, gfem.PortMode("p1", f), options=ORDER)
    assert gfem.kerr_nonlinearity(works, mode).gamma_nl[0].real > 0
    refused = materials(dataclasses.replace(aniso, nonlinear_index=2.6e-20))
    with pytest.raises(NotImplementedError, match=r"region 'clad'.*anisotropic"):
        gfem.kerr_nonlinearity(refused, mode)


def test_an_hz_line_port_is_refused(slab):
    model, _ = slab
    hz = mode_of(model, sc.c / 1.55e-6, polarization="Hz")
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.kerr_nonlinearity(model, hz)


def test_a_result_without_a_field_or_a_non_result_is_a_type_error(filled):
    model, mode, _ = filled
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.kerr_nonlinearity(model, dataclasses.replace(mode, e_field=None))
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_nonlinearity(model, "mode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.kerr_nonlinearity(mode, mode)  # ty: ignore[invalid-argument-type]


def test_traced_node_coordinates_raise_by_name(guide_mesh, filled):
    model, _, _ = filled
    nodes = np.array(guide_mesh.imported_nodes)

    def objective(scale):
        mesh = jax.tree.unflatten(
            jax.tree.structure(guide_mesh), [jnp.asarray(nodes) * scale]
        )
        moved = model.replace(mesh=mesh)
        mode = mode_of(moved, [F])
        return jnp.real(gfem.kerr_nonlinearity(moved, mode).gamma_nl[0])

    with pytest.raises(NonDifferentiableError, match="kerr_nonlinearity"):
        jax.grad(objective)(1.0)


# --- repr and pytree ---------------------------------------------------------------


def test_the_repr_gives_the_ranges_and_a_short_form_under_a_trace(guide_mesh, filled):
    _, _, kn = filled
    text = repr(kn)
    assert text.startswith("KerrNonlinearity(port 'p1', mode 1, 1 frequencies")
    assert "Re " in text
    assert "Im " in text
    assert "1/(W m)" in text
    seen = []

    def objective(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        result = gfem.kerr_nonlinearity(model, mode_of(model, [F]))
        seen.append(repr(result))
        return jnp.real(result.gamma_nl[0])

    jax.grad(objective)(jnp.asarray(N2))
    assert seen == ["KerrNonlinearity(port 'p1', mode 1, 1 frequencies)"]


def test_the_result_is_a_pytree_with_gamma_as_its_only_leaf(filled):
    _, _, kn = filled
    leaves, tree = jax.tree.flatten(kn)
    assert len(leaves) == 1
    again = jax.tree.unflatten(tree, leaves)
    assert (again.port, again.mode) == ("p1", 1)
    np.testing.assert_array_equal(again.frequency, kn.frequency)
    np.testing.assert_array_equal(again.gamma_nl, kn.gamma_nl)
    doubled = jax.tree.map(lambda x: 2 * x, kn)
    np.testing.assert_allclose(doubled.gamma_nl, 2 * kn.gamma_nl, rtol=1e-15)
    assert math.isfinite(float(jnp.abs(doubled.gamma_nl[0])))


# --- the coupling tensor ---------------------------------------------------------


@pytest.fixture(scope="module")
def waves(guide_mesh):
    """Signal, pump and idler of one mode on the filled guide, TPA 0."""
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    f = np.array([7.5e9, 8e9, 8.5e9])
    mode = mode_of(model, f)
    return model, mode, gfem.kerr_coupling(model, mode)


def test_the_tensor_names_its_waves_and_has_one_entry_per_tuple(waves):
    _, _, tensor = waves
    assert tensor.port == "p1"
    assert tensor.waves == ((1, 7.5e9), (1, 8e9), (1, 8.5e9))
    assert tensor.gamma_nl.shape == (3, 3, 3, 3)
    assert tensor.gamma_nl.dtype == jnp.complex128


def test_four_wave_mixing_follows_the_indices_of_its_waves(waves):
    # the homogeneous fill's discrete shape is the same at every frequency and
    # E_j scales as n_j^(-1/2) at 1 W: gamma_sppi / gamma_ssss =
    # n_s^2 / (n_p sqrt(n_s n_i)); round-off over a few operations is 1e-15,
    # the gate 1e-10 leaves margin for solver noise in the indices
    _, mode, tensor = waves
    n = np.asarray(mode.n_eff).real
    g = np.asarray(tensor.gamma_nl)
    s, p, i = 0, 1, 2
    expected = n[s] ** 2 / (n[p] * np.sqrt(n[s] * n[i]))
    assert g[s, p, p, i] / g[s, s, s, s] == pytest.approx(expected, rel=1e-10)


def test_the_diagonal_is_the_nonlinear_coefficient_of_each_row(waves):
    model, mode, tensor = waves
    kn = gfem.kerr_nonlinearity(model, mode)
    diagonal = np.asarray(tensor.gamma_nl)[range(3), range(3), range(3), range(3)]
    # two sums of the same terms in a different order: round-off, 1e-16
    np.testing.assert_allclose(diagonal, kn.gamma_nl, rtol=1e-13)


def test_the_tensor_is_symmetric_in_its_two_amplitudes_and_conserves_energy(waves):
    _, mode, tensor = waves
    g = np.asarray(tensor.gamma_nl)
    omega = 2 * np.pi * np.asarray(mode.frequency)
    scale = np.abs(g).max()
    # gamma_abcd = gamma_acbd: the product of A_b and A_c is symmetric
    np.testing.assert_allclose(g, g.transpose(0, 2, 1, 3), rtol=0, atol=1e-13 * scale)
    # the energy identity gamma_abcd / omega_a = conj(gamma_badc) / omega_b holds
    # for the real chi of a lossless fill; round-off of a few operations, 1e-16
    left = g / omega[:, None, None, None]
    right = np.conj(g.transpose(1, 0, 3, 2)) / omega[None, :, None, None]
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-13 * scale / omega.min())


def test_a_mode_twice_gives_every_entry_the_nonlinear_coefficient(filled):
    model, mode, kn = filled
    tensor = gfem.kerr_coupling(model, [mode, mode])
    g = np.asarray(tensor.gamma_nl)
    assert g.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(g, complex(kn.gamma_nl[0]), rtol=1e-13)
    # the cross-phase sum of the amplitude equation counts the other wave twice
    assert g[0, 0, 1, 1] + g[0, 1, 0, 1] == pytest.approx(
        2 * complex(kn.gamma_nl[0]), rel=1e-13
    )


def test_a_single_row_gives_one_wave(filled):
    model, mode, kn = filled
    tensor = gfem.kerr_coupling(model, mode)
    assert tensor.gamma_nl.shape == (1, 1, 1, 1)
    assert complex(tensor.gamma_nl[0, 0, 0, 0]) == pytest.approx(
        complex(kn.gamma_nl[0]), rel=1e-13
    )


def test_two_photon_absorption_enters_with_the_frequency_of_the_equation(guide_mesh):
    # chi_a = 4/3 n^2 c eps0 (n2 - i beta c / (2 omega_a)): the ratio of the
    # imaginary to the real part of the entries of wave a is the diagonal's
    model = guide_model(
        guide_mesh,
        gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA),
    )
    f = np.array([7.5e9, 8.5e9])
    tensor = np.asarray(gfem.kerr_coupling(model, mode_of(model, f)).gamma_nl)
    ratio = tensor.imag / tensor.real
    for a in range(2):
        assert ratio[a] == pytest.approx(TPA / (2 * k0_of(f[a]) * N2), rel=1e-12)


@pytest.fixture(scope="module")
def cross_phase(guide_mesh):
    """TE10 and TE01 of the filled guide at 12 GHz, order 3."""
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    order = gfem.SolverOptions(order=3)
    modes = [mode_of(model, [12e9], order=order, mode=k) for k in (1, 3)]
    return model, modes, gfem.kerr_coupling(model, modes)


def test_cross_phase_of_te10_by_te01_is_four_ninths_of_its_self_phase(cross_phase):
    _, modes, tensor = cross_phase
    n_m, n_n = (float(np.real(m.n_eff[0])) for m in modes)
    g = np.asarray(tensor.gamma_nl)
    m, n = 0, 1
    ratio = (g[m, m, n, n] + g[m, n, m, n]) / g[m, m, m, m]
    # the modes' fields are the guide's TE10 and TE01 up to the mesh error of
    # their shapes: 5.8e-4, 1.4e-5 and 4.5e-8 measured at orders 2, 3 and 4;
    # the gate is 3.5 times the order-3 error
    assert ratio.real == pytest.approx(4 / 9 * n_m / n_n, rel=5e-5)
    # real fields in the gauge: the two orderings of the conjugated pair agree
    assert g[m, n, n, m] == pytest.approx(g[m, m, n, n], rel=1e-14)


def test_waves_of_different_ports_or_orders_are_refused(guide_mesh, filled):
    model, mode, _ = filled
    ports = gfem.Model(
        guide_mesh,
        materials=model.materials,
        boundaries=model.boundaries,
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    other = gfem.solve(ports, gfem.PortMode("p2", [F]), options=ORDER)
    with pytest.raises(ValueError, match=r"share one port.*'p1'.*'p2'"):
        gfem.kerr_coupling(ports, [mode, other])
    third = mode_of(model, [F], order=gfem.SolverOptions(order=3))
    with pytest.raises(ValueError, match=r"share one port, element order"):
        gfem.kerr_coupling(model, [mode, third])


def test_the_coupling_refuses_what_the_coefficient_refuses(filled, slab):
    model, mode, _ = filled
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_coupling(model, "mode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_coupling(model, [])
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.kerr_coupling(model, [dataclasses.replace(mode, e_field=None)])
    bare = guide_model(model.mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(ValueError, match=r"wave port 'p1'"):
        gfem.kerr_coupling(bare, mode)
    slab_model, _ = slab
    hz = mode_of(slab_model, sc.c / 1.55e-6, polarization="Hz")
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.kerr_coupling(slab_model, hz)


def test_the_coupling_is_linear_in_the_nonlinear_index(guide_mesh):
    f = np.array([7.5e9, 8.5e9])

    def entry(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        tensor = gfem.kerr_coupling(model, mode_of(model, f))
        return jnp.real(tensor.gamma_nl[0, 1, 1, 0])

    value, slope = jax.value_and_grad(entry)(jnp.asarray(N2))
    # exact: every term of the integral carries chi once
    assert float(slope) == pytest.approx(float(value) / N2, rel=1e-12)


def test_the_repr_gives_the_waves_and_a_short_form_under_a_trace(guide_mesh, waves):
    _, _, tensor = waves
    assert repr(tensor).startswith("KerrCoupling(port 'p1', 3 waves, |gamma_nl| up to ")
    seen = []

    def objective(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        result = gfem.kerr_coupling(model, mode_of(model, [F]))
        seen.append(repr(result))
        return jnp.real(result.gamma_nl[0, 0, 0, 0])

    jax.grad(objective)(jnp.asarray(N2))
    assert seen == ["KerrCoupling(port 'p1', 1 waves)"]


def test_the_tensor_is_a_pytree_with_gamma_as_its_only_leaf(waves):
    _, _, tensor = waves
    leaves, tree = jax.tree.flatten(tensor)
    assert len(leaves) == 1
    again = jax.tree.unflatten(tree, leaves)
    assert (again.port, again.waves) == (tensor.port, tensor.waves)
    np.testing.assert_array_equal(again.gamma_nl, tensor.gamma_nl)
