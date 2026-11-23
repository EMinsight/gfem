"""`Material` and `MaterialField`: keyword-only physics values with checks."""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from anisotropy import TILTED, rodrigues

import gfem


def test_material_is_keyword_only():
    with pytest.raises(TypeError):
        gfem.Material(11.7)  # ty: ignore[too-many-positional-arguments]


def test_material_defaults_are_vacuum_and_thermal_properties_are_reserved():
    m = gfem.Material()
    assert (m.eps_r, m.mu_r, m.tan_delta, m.conductivity) == (1.0, 1.0, 0.0, 0.0)
    assert m.axes is None
    assert m.thermal_conductivity is None
    assert m.mass_density is None
    assert m.specific_heat is None
    assert m.thermo_optic_coefficient is None


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"eps_r": -2.0}, "eps_r"),
        ({"eps_r": 0.0}, "eps_r"),
        ({"eps_r": (2.0, -1.0, 2.0)}, "eps_r"),
        ({"mu_r": 0.0}, "mu_r"),
        ({"tan_delta": -1e-3}, "tan_delta"),
        ({"conductivity": -1.0}, "conductivity"),
    ],
)
def test_material_rejects_active_media(kwargs, word):
    with pytest.raises(ValueError, match="passive") as err:
        gfem.Material(**kwargs)
    assert word in str(err.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_material_rejects_non_finite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        gfem.Material(eps_r=bad)


def test_material_rejects_non_positive_thermal_values():
    with pytest.raises(ValueError, match="thermal_conductivity"):
        gfem.Material(thermal_conductivity=0.0)
    m = gfem.Material(mass_density=2330.0, specific_heat=700.0)
    assert m.mass_density == 2330.0
    with pytest.raises(ValueError, match="mass_density must be > 0 kg/m"):
        gfem.Material(mass_density=0.0)


def test_mass_density_has_no_density_attribute():
    m = gfem.Material(mass_density=2330.0)
    with pytest.raises(AttributeError, match="mass_density"):
        m.density  # noqa: B018
    f = gfem.MaterialField(mass_density=jnp.array([2330.0, 2200.0]))
    with pytest.raises(AttributeError, match="mass_density"):
        f.density  # noqa: B018


@pytest.mark.parametrize(
    "build",
    [
        lambda v: gfem.Material(density=v),  # ty: ignore[unknown-argument]
        lambda v: gfem.MaterialField(density=jnp.asarray(v)),  # ty: ignore[unknown-argument]
    ],
    ids=["Material", "MaterialField"],
)
def test_deprecated_density_keyword_warns_and_maps_to_mass_density(build):
    with pytest.warns(DeprecationWarning, match="mass_density") as record:
        m = build(2330.0)
    assert record[0].filename == __file__  # stacklevel 2: the caller's line
    np.testing.assert_allclose(m.mass_density, 2330.0)


def test_deprecated_density_keyword_beside_mass_density_raises():
    with pytest.raises(TypeError, match="mass_density"):
        gfem.Material(density=1.0, mass_density=2.0)  # ty: ignore[unknown-argument]


@pytest.mark.parametrize("value", [1.8e-4, -1.0e-4, 0.0])
def test_thermo_optic_coefficient_is_a_real_scalar_of_any_sign(value):
    assert gfem.Material(thermo_optic_coefficient=value).thermo_optic_coefficient == (
        value
    )


@pytest.mark.parametrize(
    "bad",
    [1e-4 + 1e-6j, (1e-4, 1e-4, 2e-4), np.eye(3) * 1e-4, float("nan")],
    ids=["complex", "principal", "matrix", "nan"],
)
def test_thermo_optic_coefficient_rejects_complex_tensor_and_non_finite(bad):
    with pytest.raises(ValueError, match="thermo_optic_coefficient"):
        gfem.Material(thermo_optic_coefficient=bad)


def test_thermo_optic_coefficient_per_element_field():
    values = jnp.array([1.8e-4, -2.0e-5, 0.0])
    f = gfem.MaterialField(thermo_optic_coefficient=values)
    np.testing.assert_array_equal(f.thermo_optic_coefficient, values)
    assert f.n_elements == 3
    assert "thermo_optic_coefficient" in f.properties
    # one value per element even beside principal values
    g = gfem.MaterialField(
        eps_r=jnp.ones((3, 3)) * 2.0, thermo_optic_coefficient=values, axes=np.eye(3)
    )
    assert g.thermo_optic_coefficient.shape == (3,)
    with pytest.raises(ValueError, match="thermo_optic_coefficient must be real"):
        gfem.MaterialField(thermo_optic_coefficient=values * (1 + 1j))
    with pytest.raises(ValueError, match="thermo_optic_coefficient has shape"):
        gfem.MaterialField(thermo_optic_coefficient=jnp.ones((3, 3)))
    grad = jax.grad(
        lambda v: jnp.sum(
            gfem.MaterialField(thermo_optic_coefficient=v).thermo_optic_coefficient
        )
    )(values)
    np.testing.assert_array_equal(grad, 1.0)


def test_uniform_field_carries_mass_density_and_thermo_optic_coefficient():
    m = gfem.Material(mass_density=2330.0, thermo_optic_coefficient=-1e-4)
    f = gfem.MaterialField.uniform("core", m)
    assert float(f.mass_density) == 2330.0
    assert float(f.thermo_optic_coefficient) == -1e-4


def test_interpolate_blends_mass_density_and_thermo_optic_coefficient():
    void = gfem.Material(mass_density=1000.0, thermo_optic_coefficient=1e-5)
    solid = gfem.Material(mass_density=3000.0, thermo_optic_coefficient=-1e-5)
    f = gfem.MaterialField.interpolate(jnp.array([0.0, 0.5, 1.0]), void, solid)
    np.testing.assert_allclose(f.mass_density, [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(f.thermo_optic_coefficient, [1e-5, 0.0, -1e-5])


def test_missing_material_property_error_lists_mass_density_and_remedies():
    err = gfem.errors.MissingMaterialPropertyError(
        [
            ("heater", "thermal_conductivity"),
            ("heater", "mass_density"),
            ("box", "specific_heat"),
        ]
    )
    assert isinstance(err, ValueError)
    assert gfem.MissingMaterialPropertyError is gfem.errors.MissingMaterialPropertyError
    assert err.missing == (
        ("heater", "thermal_conductivity"),
        ("heater", "mass_density"),
        ("box", "specific_heat"),
    )
    text = str(err)
    for word in ("'heater'", "'box'", "W/(m K)", "kg/m^3", "J/(kg K)", "excluded="):
        assert word in text
    assert re.search(r"mass_density \(kg/m\^3, e\.g\. [0-9.]+\)", text)


def test_anisotropic_material_takes_three_principal_values_and_orthonormal_axes():
    m = gfem.Material(
        eps_r=np.array([9.3, 9.3, 11.5]),
        axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    assert m.eps_r == (9.3, 9.3, 11.5)
    assert m.axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    hash(m)  # static pytree data must be hashable
    with pytest.raises(ValueError, match="three"):
        gfem.Material(eps_r=(1.0, 2.0))
    with pytest.raises(ValueError, match="orthonormal"):
        gfem.Material(eps_r=(1.0, 2.0, 3.0), axes=((1, 0, 0), (1, 1, 0), (0, 0, 1)))


def test_loss_tangent_takes_three_values_along_the_axes():
    # sapphire: Palace's LossTan is one value per principal axis, like eps_r
    m = gfem.Material(
        eps_r=(9.3, 9.3, 11.5),
        tan_delta=np.array([3e-5, 3e-5, 8.6e-5]),
        axes=((0.8, 0.6, 0), (-0.6, 0.8, 0), (0, 0, 1)),
    )
    assert m.tan_delta == (3e-5, 3e-5, 8.6e-5)
    assert not m.isotropic
    assert not gfem.Material(eps_r=4.0, tan_delta=(0.0, 0.0, 1e-3)).isotropic
    hash(m)
    with pytest.raises(ValueError, match="three"):
        gfem.Material(tan_delta=(1e-3, 2e-3))
    with pytest.raises(ValueError, match="passive") as err:
        gfem.Material(tan_delta=(1e-3, -1e-4, 0.0))
    assert "tan_delta" in str(err.value)


def _tensor(values, axes):
    a = np.eye(3) if axes is None else np.asarray(axes)
    return sum(v * np.outer(a_k, a_k) for v, a_k in zip(values, a, strict=True))


def test_a_symmetric_matrix_becomes_principal_values_along_axes():
    r = rodrigues((1, 2, 3), 0.4)
    matrix = r @ np.diag([2.0, 3.0, 5.0]) @ r.T
    m = gfem.Material(eps_r=matrix)
    # r is close to the identity: column k of r is the axis closest to e_k
    np.testing.assert_allclose(m.eps_r, (2.0, 3.0, 5.0), rtol=1e-12)
    np.testing.assert_allclose(m.axes, r.T, atol=1e-12)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), matrix, atol=1e-12)
    # the same matrix from permuted, negated eigenvectors: the same normal form
    q = r[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    same = gfem.Material(eps_r=q @ np.diag([5.0, 2.0, 3.0]) @ q.T)
    np.testing.assert_allclose(same.eps_r, m.eps_r, rtol=1e-12)
    np.testing.assert_allclose(same.axes, m.axes, atol=1e-12)
    hash(m)
    assert repr(m).startswith("Material(eps_r=(")
    assert "axes=((" in repr(m)
    assert "array" not in repr(m)


@pytest.mark.parametrize(
    ("angle", "values", "axes"),
    [
        # Rz(2.5): both in-plane eigenvectors point away from +x / +y, so the
        # dominant component of each is made positive
        (
            2.5,
            (2.0, 3.0, 5.0),
            ((-np.cos(2.5), -np.sin(2.5), 0), (np.sin(2.5), -np.cos(2.5), 0)),
        ),
        # Rz(pi/2 + 0.2): the first eigenvector lies closest to y, so it is
        # axis 2 and its value moves with it
        (
            np.pi / 2 + 0.2,
            (3.0, 2.0, 5.0),
            (
                (np.sin(np.pi / 2 + 0.2), -np.cos(np.pi / 2 + 0.2), 0),
                (np.cos(np.pi / 2 + 0.2), np.sin(np.pi / 2 + 0.2), 0),
            ),
        ),
    ],
)
def test_derived_axes_are_ordered_by_the_closest_coordinate_axis_and_signed(
    angle, values, axes
):
    r = rodrigues((0, 0, 1), angle)
    m = gfem.Material(eps_r=r @ np.diag([2.0, 3.0, 5.0]) @ r.T)
    np.testing.assert_allclose(m.eps_r, values, rtol=1e-12)
    np.testing.assert_allclose(m.axes, (*axes, (0, 0, 1)), atol=1e-12)


def test_a_diagonal_matrix_equals_its_principal_values_on_the_coordinate_axes():
    m = gfem.Material(eps_r=np.diag([2.0, 2.0, 3.0]))
    assert m == gfem.Material(eps_r=(2.0, 2.0, 3.0))
    assert hash(m) == hash(gfem.Material(eps_r=(2.0, 2.0, 3.0)))
    assert m.axes is None
    assert repr(m) == "Material(eps_r=(2.0, 2.0, 3.0))"
    off = np.diag([2.0, 3.0, 4.0]) + 1e-12 * (np.ones((3, 3)) - np.eye(3))
    assert gfem.Material(mu_r=off).axes is None


def test_a_degenerate_eigenspace_takes_the_projected_coordinate_axes():
    r = rodrigues((1, 2, 3), 0.4)
    u = r[:, 2]  # the eigenvector of 5, closest to z
    p = np.eye(3) - np.outer(u, u)  # the eigenspace of 2
    a0 = p[:, 0] / np.linalg.norm(p[:, 0])
    a1 = p[:, 1] - (a0 @ p[:, 1]) * a0
    a1 /= np.linalg.norm(a1)
    expected = (a0, a1, u)
    for spin in (0.0, 1.0, 2.0):  # any basis of the eigenspace: one normal form
        q = r @ rodrigues((0, 0, 1), spin)
        m = gfem.Material(eps_r=q @ np.diag([2.0, 2.0, 5.0]) @ q.T)
        np.testing.assert_allclose(m.eps_r, (2.0, 2.0, 5.0), rtol=1e-12)
        np.testing.assert_allclose(m.axes, expected, atol=1e-12)


def _message(**kwargs):
    with pytest.raises(ValueError) as err:
        gfem.Material(**kwargs)
    return str(err.value)


def test_matrix_input_rules():
    r = rodrigues((1, 2, 3), 0.4)
    tilted = r @ np.diag([2.0, 3.0, 5.0]) @ r.T
    assert _message(eps_r=tilted, axes=np.eye(3)) == (
        "Material: eps_r is a 3x3 matrix, which carries its own principal axes; "
        "leave `axes` as None."
    )
    assert _message(eps_r=tilted, tan_delta=(1e-3, 1e-3, 2e-3)) == (
        "Material: eps_r is a 3x3 matrix but tan_delta gives three principal "
        "values along `axes`, which the matrix derives itself; give tan_delta "
        "as one value or a 3x3 matrix."
    )
    assert _message(eps_r=tilted * (1 - 1e-3j)) == (
        "Material: eps_r must be real; give dielectric loss with tan_delta or "
        "conductivity, or a complex value through gfem.MaterialField."
    )
    assert _message(mu_r=1 - 0.1j) == (
        "Material: mu_r must be real; for magnetic loss give a complex mu_r "
        "through gfem.MaterialField."
    )
    assert _message(tan_delta=1e-3j) == "Material: tan_delta must be real."
    bad = tilted.copy()
    bad[0, 1] = np.nan
    assert "Material: eps_r must be finite" in _message(eps_r=bad)
    assert _message(eps_r=np.eye(2)) == (
        "Material: eps_r takes one value, three principal values (along `axes`) "
        "or a 3x3 matrix, got shape (2, 2)."
    )
    skew = tilted + 1e-6 * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    assert _message(eps_r=skew, name="ferrite") == (
        "Material 'ferrite': eps_r is not symmetric; non-symmetric (gyrotropic) "
        "tensors are not supported."
    )
    # an asymmetry at round-off is accepted and symmetrized
    nearly = tilted + 1e-11 * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    m = gfem.Material(eps_r=nearly)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), tilted, atol=1e-11)


def test_a_material_s_matrices_share_one_set_of_principal_axes():
    r = rodrigues((1, 2, 3), 0.4)
    assert _message(
        eps_r=r @ np.diag([2.0, 3.0, 5.0]) @ r.T,
        conductivity=np.diag([1.0, 2.0, 3.0]),
        name="x",
    ) == (
        "eps_r and conductivity of Material 'x' have different principal axes; "
        "a material's tensors share one set of principal axes."
    )
    # eps_r is degenerate; conductivity, which commutes with it, splits the
    # eigenspace and so fixes the axes
    eps = r @ np.diag([2.0, 2.0, 5.0]) @ r.T
    sigma = r @ np.diag([1.0, 3.0, 0.5]) @ r.T
    m = gfem.Material(eps_r=eps, conductivity=sigma)
    np.testing.assert_allclose(m.axes, r.T, atol=1e-12)
    np.testing.assert_allclose(m.eps_r, (2.0, 2.0, 5.0), rtol=1e-12)
    np.testing.assert_allclose(m.conductivity, (1.0, 3.0, 0.5), rtol=1e-12)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), eps, atol=1e-12)
    np.testing.assert_allclose(_tensor(m.conductivity, m.axes), sigma, atol=1e-12)


def test_passivity_is_checked_per_principal_value():
    assert _message(conductivity=(1.0, -1.0, 0.0), name="x") == (
        "Material 'x' is not passive: conductivity has principal value -1.0 "
        "(along axis 2) but every principal value of conductivity must be >= 0. "
        "A passive medium has eps_r > 0, mu_r > 0 and losses (tan_delta, "
        "conductivity) >= 0."
    )
    axes = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    # the eigenvector of -1 lies closest to x, so it is axis 1, signed +x
    assert _message(eps_r=axes.T @ np.diag([2.0, -1.0, 3.0]) @ axes) == (
        "Material is not passive: eps_r has principal value -1.0 (along axis "
        "(0.8, -0.6, 0.0)) but every principal value of eps_r must be > 0. A "
        "passive medium has eps_r > 0, mu_r > 0 and losses (tan_delta, "
        "conductivity) >= 0."
    )
    wire = gfem.Material(conductivity=(0.0, 0.0, 1.0))  # conducts along z only
    assert wire.conductivity == (0.0, 0.0, 1.0)
    assert not wire.isotropic
    assert gfem.Material(conductivity=(0.0, 0.0, 0.0)).conductivity == 0.0


def test_equal_principal_values_are_one_value():
    m = gfem.Material(eps_r=(2.0, 2.0, 2.0))
    assert m.eps_r == 2.0
    assert m.isotropic
    assert m == gfem.Material(eps_r=2)
    assert hash(m) == hash(gfem.Material(eps_r=2))
    assert repr(m) == "Material(eps_r=2.0)"
    r = rodrigues((1, 2, 3), 0.4)
    near = r @ (2 * np.diag([1 + 1e-10, 1.0, 1 - 1e-10])) @ r.T
    m = gfem.Material(eps_r=near)
    assert m.isotropic
    assert m.axes is None
    assert m.eps_r == pytest.approx(2.0, rel=1e-12)
    assert not gfem.Material(conductivity=(1.0, 1.0, 2.0)).isotropic
    f = gfem.MaterialField.uniform("r", gfem.Material(eps_r=(2.0, 2.0, 2.0)))
    assert float(f.eps_r) == 2.0


def test_material_conductivity_is_documented_as_electrical():
    assert "electrical" in gfem.Material.__doc__
    assert "S/m" in gfem.Material.__doc__


def test_material_under_a_gradient_raises_with_the_material_field_remedy():
    def objective(eps):
        return gfem.Material(eps_r=eps).eps_r

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"MaterialField\.uniform"
    ) as e:
        jax.grad(objective)(2.0)
    message = str(e.value)
    assert "computed inside jax.grad, jax.jit or jax.vmap" in message
    assert "jax.grad reaches its values" in message
    assert not re.search(r"\b(traced|leaf|leaves|pytree)\b", message)


# --- MaterialField -------------------------------------------------------------


def test_material_field_is_keyword_only_with_absent_properties_none():
    with pytest.raises(TypeError):
        gfem.MaterialField(jnp.ones(3))  # ty: ignore[too-many-positional-arguments]
    f = gfem.MaterialField(eps_r=jnp.ones(3))
    assert f.conductivity is None
    assert f.thermal_conductivity is None
    assert jax.tree.leaves(f)[0].shape == (3,)
    assert len(jax.tree.leaves(f)) == 1


def test_material_field_values_are_float64_one_per_element():
    f = gfem.MaterialField(eps_r=[1, 2, 3], conductivity=np.zeros(3))
    assert f.eps_r.dtype == jnp.float64
    assert f.n_elements == 3
    with pytest.raises(ValueError, match="one value per element"):
        gfem.MaterialField(eps_r=jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="same number"):
        gfem.MaterialField(eps_r=jnp.ones(3), conductivity=jnp.ones(4))
    with pytest.raises(ValueError, match="at least one"):
        gfem.MaterialField()


def test_material_field_rejects_active_concrete_values():
    with pytest.raises(ValueError, match="passive"):
        gfem.MaterialField(eps_r=jnp.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="passive"):
        # e^{+iwt}: a lossy permittivity has a negative imaginary part
        gfem.MaterialField(eps_r=jnp.array([4.0 + 0.1j]))
    assert gfem.MaterialField(eps_r=jnp.array([4.0 - 0.1j])).eps_r.dtype == (
        jnp.complex128
    )


def test_interpolate_blends_every_property_both_materials_define():
    void = gfem.Material()
    solid = gfem.Material(eps_r=10.0, conductivity=2.0, thermal_conductivity=150.0)
    rho = jnp.array([0.0, 0.5, 1.0])
    f = gfem.MaterialField.interpolate(rho, void, solid)
    np.testing.assert_allclose(f.eps_r, [1.0, 5.5, 10.0])
    np.testing.assert_allclose(f.conductivity, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(f.mu_r, [1.0, 1.0, 1.0])
    assert f.thermal_conductivity is None  # the void defines no thermal value


@pytest.mark.parametrize(
    ("scheme", "expected"),
    # rho = 0.5, penalty 3: simp 0.5**3 = 0.125; ramp 0.5 / (1 + 3 * 0.5) = 0.2
    [("linear", 0.5), ("simp", 0.125), ("ramp", 0.2)],
)
def test_interpolation_schemes(scheme, expected):
    f = gfem.MaterialField.interpolate(
        jnp.array([0.5]), gfem.Material(), gfem.Material(eps_r=2.0), scheme=scheme
    )
    np.testing.assert_allclose(f.eps_r, [1.0 + expected])


def test_interpolate_is_differentiable_in_the_density():
    void, solid = gfem.Material(), gfem.Material(eps_r=10.0)

    def total(rho):
        return jnp.sum(gfem.MaterialField.interpolate(rho, void, solid).eps_r)

    np.testing.assert_allclose(jax.grad(total)(jnp.full(4, 0.3)), np.full(4, 9.0))


def test_interpolate_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        gfem.MaterialField.interpolate(
            jnp.ones(2), gfem.Material(), gfem.Material(), scheme="cubic"
        )


def test_uniform_field_holds_one_differentiable_value_per_property():
    sub = gfem.Material(eps_r=11.7, tan_delta=1e-4)
    f = gfem.MaterialField.uniform("sub", sub)
    assert f.region == "sub"
    assert f.eps_r.shape == ()
    assert float(f.eps_r) == 11.7
    assert float(f.tan_delta) == 1e-4
    assert f.thermal_conductivity is None

    def eps(field):
        return field.eps_r * 2.0

    assert float(jax.grad(eps)(f).eps_r) == 2.0


def test_material_field_is_a_pytree_with_named_leaf_paths():
    f = gfem.MaterialField(eps_r=jnp.ones(2), conductivity=jnp.zeros(2))
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(f)]
    assert paths == [".eps_r", ".conductivity"]
    g = jax.tree.map(lambda x: x * 2, f)
    assert isinstance(g, gfem.MaterialField)
    np.testing.assert_allclose(g.eps_r, [2.0, 2.0])


def test_material_repr_shows_only_what_differs_from_vacuum():
    assert repr(gfem.Material()) == "Material()"
    m = gfem.Material(eps_r=11.7, tan_delta=1e-4, name="si")
    assert repr(m) == "Material(eps_r=11.7, tan_delta=0.0001, name='si')"


# --- MaterialField with axes: principal values on fixed axes -------------------


def _crystal(axes=TILTED, order=(0, 1, 2), signs=(1, 1, 1)):
    """A biaxial lossy crystal, its axes (and values) permuted or sign-flipped."""
    eps, tan, mu = (2.0, 3.0, 5.0), (1e-3, 2e-3, 3e-3), (1.5, 1.0, 1.2)
    return gfem.Material(
        eps_r=tuple(eps[k] for k in order),
        tan_delta=tuple(tan[k] for k in order),
        mu_r=tuple(mu[k] for k in order),
        axes=tuple(
            tuple(s * x for x in axes[k]) for k, s in zip(order, signs, strict=True)
        ),
    )


def _field_tensor(field, name, element=None):
    """``sum_k v_k a_k a_k^T`` of a field's principal values (independent sum)."""
    values = np.asarray(getattr(field, name))
    values = values if element is None else values[element]
    return _tensor(values, field.axes)


def test_a_material_field_with_axes_holds_three_principal_values_per_element():
    values = jnp.array([[2.0, 3.0, 5.0], [2.5, 3.5, 4.0]])
    f = gfem.MaterialField(eps_r=values, thermal_conductivity=[1.0, 2.0], axes=TILTED)
    assert f.n_elements == 2
    assert f.axes == TILTED
    assert f.eps_r.shape == (2, 3)
    assert f.thermal_conductivity.shape == (2,)
    uniform = gfem.MaterialField(eps_r=[2.0, 3.0, 5.0], axes=np.eye(3))
    assert uniform.n_elements is None
    assert uniform.axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert "axes=" in repr(f)
    assert "axes" not in repr(gfem.MaterialField(eps_r=jnp.ones(2)))
    # axes are static: a tree map keeps them, and the leaves are the values
    g = jax.tree.map(lambda x: x * 2, f)
    assert g.axes == TILTED
    assert [x.shape for x in jax.tree.leaves(f)] == [(2, 3), (2,)]


def test_material_field_shape_rules_with_and_without_axes():
    with pytest.raises(ValueError, match=r"principal values need `axes`"):
        gfem.MaterialField(eps_r=jnp.ones((4, 3)))
    with pytest.raises(ValueError, match="three principal values"):
        gfem.MaterialField(eps_r=jnp.ones((4, 2)), axes=TILTED)
    with pytest.raises(ValueError, match="three principal values"):
        gfem.MaterialField(eps_r=jnp.ones(4), axes=TILTED)
    with pytest.raises(ValueError, match="same number"):
        gfem.MaterialField(
            eps_r=jnp.ones((4, 3)), conductivity=jnp.ones((3, 3)), axes=TILTED
        )
    with pytest.raises(ValueError, match="orthonormal"):
        gfem.MaterialField(eps_r=jnp.ones(3), axes=((1, 0, 0), (1, 1, 0), (0, 0, 1)))


def test_material_field_passivity_is_checked_per_principal_value():
    with pytest.raises(ValueError, match="passive"):
        gfem.MaterialField(eps_r=jnp.array([[2.0, 3.0, -1.0]]), axes=TILTED)
    with pytest.raises(ValueError, match="passive"):
        gfem.MaterialField(conductivity=jnp.array([0.0, -1.0, 0.0]), axes=TILTED)
    with pytest.raises(ValueError, match="negative imaginary part"):
        gfem.MaterialField(eps_r=jnp.array([2.0, 3.0 + 0.1j, 4.0]), axes=TILTED)
    lossy = gfem.MaterialField(eps_r=jnp.array([2.0, 3.0 - 0.1j, 4.0]), axes=TILTED)
    assert lossy.eps_r.dtype == jnp.complex128


def test_traced_material_field_axes_raise():
    def objective(angle):
        c, s = jnp.cos(angle), jnp.sin(angle)
        axes = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return gfem.MaterialField(eps_r=jnp.ones(3), axes=axes).eps_r.sum()

    with pytest.raises(gfem.NonDifferentiableError, match="axes are fixed"):
        jax.grad(objective)(0.3)


def test_uniform_field_of_an_anisotropic_material_holds_its_principal_values():
    crystal = _crystal()
    f = gfem.MaterialField.uniform("r", crystal)
    assert f.axes == crystal.axes
    for name in ("eps_r", "mu_r", "tan_delta", "conductivity"):
        assert getattr(f, name).shape == (3,)
    np.testing.assert_array_equal(f.eps_r, crystal.eps_r)
    np.testing.assert_array_equal(f.conductivity, [0.0, 0.0, 0.0])
    assert f.thermal_conductivity is None
    # per-axis values without axes: along the coordinate axes
    wire = gfem.MaterialField.uniform("w", gfem.Material(conductivity=(0.0, 0.0, 1.0)))
    assert wire.axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(wire.eps_r, [1.0, 1.0, 1.0])
    # an isotropic material keeps the isotropic field
    assert gfem.MaterialField.uniform("r", gfem.Material(eps_r=2.0)).axes is None


def test_interpolate_blends_an_isotropic_and_an_anisotropic_material_per_axis():
    crystal, air = _crystal(), gfem.Material()
    rho = jnp.array([0.0, 0.5, 1.0])
    for void, solid, ends in ((air, crystal, (0, 2)), (crystal, air, (2, 0))):
        f = gfem.MaterialField.interpolate(rho, void, solid)
        assert f.eps_r.shape == f.mu_r.shape == f.tan_delta.shape == (3, 3)
        eps = _tensor(crystal.eps_r, crystal.axes)
        np.testing.assert_allclose(
            _field_tensor(f, "eps_r", ends[0]), np.eye(3), atol=1e-14
        )
        np.testing.assert_allclose(_field_tensor(f, "eps_r", ends[1]), eps, atol=1e-14)
        np.testing.assert_allclose(
            _field_tensor(f, "eps_r", 1), (eps + np.eye(3)) / 2, atol=1e-14
        )
        mu = _tensor(crystal.mu_r, crystal.axes)
        np.testing.assert_allclose(_field_tensor(f, "mu_r", ends[1]), mu, atol=1e-14)
    one = gfem.MaterialField.interpolate(jnp.asarray(1.0), air, crystal)
    assert one.eps_r.shape == (3,)
    assert sorted(np.asarray(one.eps_r).tolist()) == [2.0, 3.0, 5.0]  # exactly


def test_interpolate_gives_one_field_for_permuted_or_sign_flipped_axes():
    rho = jnp.array([0.2, 0.7])
    air = gfem.Material()
    reference = gfem.MaterialField.interpolate(rho, air, _crystal())
    for order, signs in (((1, 2, 0), (1, 1, 1)), ((0, 1, 2), (-1, 1, -1))):
        other = _crystal(order=order, signs=signs)
        f = gfem.MaterialField.interpolate(rho, air, other)
        assert f.axes == reference.axes
        for name in ("eps_r", "mu_r", "tan_delta", "conductivity"):
            np.testing.assert_array_equal(getattr(f, name), getattr(reference, name))


def test_interpolate_blends_two_coaxial_crystals_and_refuses_others():
    a = _crystal()
    b = gfem.Material(eps_r=(4.0, 6.0, 7.0), axes=TILTED, conductivity=(0, 1, 0))
    f = gfem.MaterialField.interpolate(jnp.array([0.25]), a, b)
    expected = 0.75 * _tensor(a.eps_r, a.axes) + 0.25 * _tensor(b.eps_r, b.axes)
    np.testing.assert_allclose(_field_tensor(f, "eps_r", 0), expected, atol=1e-14)
    sigma = 0.25 * _tensor(b.conductivity, b.axes)
    np.testing.assert_allclose(_field_tensor(f, "conductivity", 0), sigma, atol=1e-14)
    turned = _crystal(axes=tuple(tuple(r) for r in rodrigues((0, 0, 1), 0.3).T))
    with pytest.raises(ValueError, match="different principal axes"):
        gfem.MaterialField.interpolate(jnp.array([0.5]), a, turned)


def test_interpolate_of_two_isotropic_materials_keeps_an_isotropic_field():
    f = gfem.MaterialField.interpolate(
        jnp.array([0.0, 0.5]), gfem.Material(), gfem.Material(eps_r=3.0)
    )
    assert f.axes is None
    assert f.eps_r.shape == (2,)
    np.testing.assert_array_equal(f.eps_r, [1.0, 2.0])


def test_element_values_refuse_a_property_held_as_principal_values(tmp_path):
    import structured_meshes as sm

    from gfem._engine import coefficients

    path = sm.write_msh22(
        tmp_path / "two.msh", sm.plate(4.0, 3.0, (("a", 0.5), ("b", 0.5)), (1, 1, 1))
    )
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("a"))
    values = np.tile([2.0, 3.0, 5.0], (n, 1)) * np.linspace(1, 2, n)[:, None]
    field = gfem.MaterialField(eps_r=values, axes=TILTED)
    model = gfem.Model(mesh, materials={"a": field, "b": gfem.Material(eps_r=4.0)})
    with pytest.raises(ValueError, match=r"'a'.*eps_r.*principal values"):
        coefficients.element_values(model, "eps_r")
    # an anisotropic Material has no one value either: no mean of its
    # principal values, the same placeholder or error as a field
    crystal = gfem.Material(eps_r=(2.0, 3.0, 5.0), tan_delta=1e-3)
    with_crystal = model.replace(materials={"a": gfem.Material(), "b": crystal})
    with pytest.raises(ValueError, match=r"'b'.*eps_r.*principal values"):
        coefficients.element_values(with_crystal, "eps_r")
    placeholder = coefficients.element_values(with_crystal, "eps_r", per_axis=7.0)
    np.testing.assert_array_equal(placeholder[mesh.elements("b")], 7.0)
    np.testing.assert_array_equal(
        coefficients.element_values(with_crystal, "tan_delta")[mesh.elements("b")],
        1e-3,
    )
    np.testing.assert_array_equal(
        coefficients.element_values(model, "mu_r"), np.ones(len(mesh.tetrahedra))
    )
    tensors = coefficients.element_tensors(model, "eps_r")
    for i, e in enumerate(mesh.elements("a")):
        np.testing.assert_allclose(
            tensors[e], _tensor(values[i], TILTED), rtol=0, atol=1e-14
        )
    b = mesh.elements("b")
    np.testing.assert_array_equal(
        tensors[b], np.broadcast_to(4.0 * np.eye(3), (len(b), 3, 3))
    )


# --- traced values are checked on the host at solve time ---------------------------


def _line_model(mesh, order=2):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=order),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def traced_line(tmp_path_factory):
    path = tmp_path_factory.mktemp("traced") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 3, 1)))
    return _line_model(gfem.Mesh.from_gmsh(path, length_unit="mm"))


def _traced_value_and_grad(model, build, p, solver="direct"):
    def objective(p):
        s = gfem.solve(
            model.with_material("dielectric", build(p)),
            gfem.Driven((21e9,)),
            options=gfem.SolverOptions(linear_solver=solver),
        ).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    return jax.value_and_grad(objective)(jnp.asarray(p))


def _one(n, value, rest=3.0):
    p = np.full(n, rest)
    p[0] = value
    return p


_BAD_TRACED = {
    "eps_r nan, direct": ("eps_r", "must be finite", np.nan, "direct"),
    "eps_r nan, auto": ("eps_r", "must be finite", np.nan, "auto"),
    "eps_r negative": ("eps_r", "is not passive", -2.0, "direct"),
    "eps_r zero": ("eps_r", "is not passive", 0.0, "direct"),
    "eps_r gain": ("eps_r", "is not passive", 3.0, "direct"),
    "mu_r negative": ("mu_r", "is not passive", -1.0, "direct"),
    "tan_delta negative": ("tan_delta", "is not passive", -0.1, "direct"),
    "conductivity negative": ("conductivity", "is not passive", -1.0, "direct"),
    "conductivity inf": ("conductivity", "must be finite", np.inf, "direct"),
}


@pytest.mark.parametrize("case", list(_BAD_TRACED))
def test_a_traced_material_value_out_of_range_raises_like_an_eager_one(
    traced_line, case
):
    name, word, value, solver = _BAD_TRACED[case]
    n = len(traced_line.mesh.tetrahedra)
    if case == "eps_r gain":
        build, p = (lambda p: gfem.MaterialField(eps_r=p * (1 + 0.1j))), np.full(n, 3.0)
    elif name == "tan_delta":
        build = lambda p: gfem.MaterialField(eps_r=3.0, tan_delta=p)  # noqa: E731
        p = np.full(n, value)
    elif name == "conductivity":
        build = lambda p: gfem.MaterialField(conductivity=p)  # noqa: E731
        p = np.full(n, value) if value < 0 else _one(n, value)
    else:
        build = lambda p: gfem.MaterialField(**{name: p})  # noqa: E731
        p = _one(n, value)
    # the same numbers are refused when the field is built eagerly
    with pytest.raises(ValueError, match=word):
        build(np.asarray(p))
    with pytest.raises(Exception, match=word) as raised:
        _traced_value_and_grad(traced_line, build, p, solver)
    message = str(raised.value)
    assert "MaterialField" in message
    assert name in message
    assert "region 'dielectric'" in message


@pytest.mark.parametrize("order", [1, 2])
def test_a_traced_negative_permittivity_behind_an_absorbing_wall_raises(
    traced_line, order
):
    mesh = traced_line.mesh
    model = _line_model(mesh, order)
    wall = mesh.triangles[mesh.elements("side")]
    behind = next(
        e for e, tet in enumerate(mesh.tetrahedra) if len(set(tet) & set(wall[0])) == 3
    )
    p = np.full(len(mesh.tetrahedra), 3.0)
    p[behind] = -2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(Exception, match=r"is not passive.*region 'dielectric'"):
            _traced_value_and_grad(model, lambda p: gfem.MaterialField(eps_r=p), p)


def test_valid_traced_material_values_still_solve(traced_line):
    n = len(traced_line.mesh.tetrahedra)
    value, grad = _traced_value_and_grad(
        traced_line, lambda p: gfem.MaterialField(eps_r=p), np.full(n, 3.0)
    )
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad)).all()
    rng = np.random.default_rng(0)
    rho = np.concatenate([[0.0, 1.0], rng.uniform(0, 1, n - 2)])
    value, grad = _traced_value_and_grad(
        traced_line,
        lambda r: gfem.MaterialField.interpolate(
            r, gfem.Material(), gfem.Material(eps_r=4.0, conductivity=10.0)
        ),
        rho,
    )
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grad)).all()


# --- electro-optic tensor --------------------------------------------------------

# Voigt rows of the contracted tensor: xx, yy, zz, yz, xz, xy
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
# a z-cut-like crystal: r13, r33, r22, r51 of lithium niobate (clamped, m/V)
_R_LN = np.zeros((6, 3))
_R_LN[0, 2] = _R_LN[1, 2] = 8.6e-12
_R_LN[2, 2] = 30.8e-12
_R_LN[0, 1], _R_LN[1, 1], _R_LN[5, 0] = -3.4e-12, 3.4e-12, -3.4e-12
_R_LN[3, 1] = _R_LN[4, 0] = 28.0e-12
# x-cut: crystal x up (global y), crystal y along global z, crystal z lateral
_LAW = gfem.PermittivityLaw(
    eps_inf=4.0,
    terms=[gfem.Lorentz(delta_eps=0.6, resonance_frequency=6e15, damping=0.0)],
)
_X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def _r_global(r, axes):
    """The contracted tensor in the global frame, through the full r_ijk."""
    full = np.zeros((3, 3, 3))
    for row, (i, j) in enumerate(_VOIGT):
        full[i, j] = full[j, i] = np.asarray(r)[row]
    a = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    g = np.einsum("li,mj,nk,lmn->ijk", a, a, a, full)
    return np.array([g[i, j] for i, j in _VOIGT])


def test_electro_optic_tensor_is_a_6x3_matrix_in_the_frame_of_the_axes():
    m = gfem.Material(eps_r=(4.6, 4.6, 4.9), axes=_X_CUT, electro_optic_tensor=_R_LN)
    assert gfem.Material().electro_optic_tensor is None
    assert m.electro_optic_tensor == tuple(
        tuple(float(x) for x in row) for row in _R_LN
    )
    assert m.axes == _X_CUT
    # the tensor does not make a material direction-dependent at zero field
    poled = gfem.Material(eps_r=2.5, axes=_X_CUT, electro_optic_tensor=_R_LN)
    assert poled.isotropic
    assert poled.axes == _X_CUT
    zero = gfem.Material(eps_r=2.0, electro_optic_tensor=np.zeros((6, 3)))
    assert zero.electro_optic_tensor == ((0.0, 0.0, 0.0),) * 6
    assert "electro_optic_tensor" in repr(m)
    assert hash(m) == hash(
        gfem.Material(eps_r=(4.6, 4.6, 4.9), axes=_X_CUT, electro_optic_tensor=_R_LN)
    )


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        (
            {"electro_optic_tensor": np.zeros((3, 3, 3))},
            ["6x3", "xx, yy, zz, yz, xz, xy"],
        ),
        ({"electro_optic_tensor": np.zeros((3, 6))}, ["6x3", "xx, yy, zz, yz, xz, xy"]),
        ({"electro_optic_tensor": _R_LN * (1 + 1e-3j)}, ["must be real"]),
        ({"electro_optic_tensor": _R_LN * np.nan}, ["finite"]),
        ({"electro_optic_tensor": _R_LN * 1e12}, ["m/V", "30.9e-12"]),
        (
            {"eps_r": np.diag([4.9, 4.6, 4.9]), "electro_optic_tensor": _R_LN},
            ["eps_r is a 3x3 matrix", "principal values", "axes"],
        ),
        (
            {
                "eps_r": (4.6, 4.6, 4.9),
                "axes": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                "electro_optic_tensor": _R_LN,
            },
            ["right-handed", "reverse"],
        ),
        (
            {
                "eps_r": _LAW,
                "electro_optic_tensor": _R_LN,
            },
            ["PermittivityLaw", "constant eps_r", "axes"],
        ),
        ({"electro_optic_tensor": _R_LN}, ["optical eps_r", "n^4"]),
    ],
    ids=[
        "3x3x3",
        "3x6",
        "complex",
        "nan",
        "pm/V",
        "matrix",
        "left-handed",
        "law",
        "no-eps_r",
    ],
)
def test_electro_optic_tensor_refusals_name_the_remedy(kwargs, words):
    with pytest.raises(ValueError, match="electro_optic_tensor") as e:
        gfem.Material(**kwargs)
    for word in words:
        assert word in str(e.value)


def test_electro_optic_tensor_beside_a_matrix_conductivity_raises():
    sigma = np.diag([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="conductivity is a 3x3 matrix"):
        gfem.Material(conductivity=sigma, electro_optic_tensor=_R_LN)


def test_electro_optic_tensor_under_a_gradient_raises_with_the_field_remedy():
    def objective(r33):
        r = jnp.zeros((6, 3)).at[2, 2].set(r33)
        return gfem.Material(eps_r=4.6, electro_optic_tensor=r).eps_r

    with pytest.raises(gfem.NonDifferentiableError, match=r"MaterialField\.uniform"):
        jax.grad(objective)(30.8e-12)


def test_electro_optic_tensor_per_element_field():
    n = 4
    r = jnp.tile(jnp.asarray(_R_LN), (n, 1, 1)) * jnp.arange(n)[:, None, None]
    f = gfem.MaterialField(eps_r=4.6, electro_optic_tensor=r)
    assert f.electro_optic_tensor.dtype == jnp.float64
    assert f.n_elements == n
    assert "electro_optic_tensor" in f.properties
    g = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.6), axes=_X_CUT, electro_optic_tensor=_R_LN
    )
    assert g.electro_optic_tensor.shape == (6, 3)
    assert g.n_elements == n
    with pytest.raises(ValueError, match="same number of values"):
        gfem.MaterialField(eps_r=jnp.ones(3), electro_optic_tensor=r)


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        ({"electro_optic_tensor": jnp.zeros((3, 3, 3))}, ["(6, 3)", "xx, yy, zz"]),
        ({"electro_optic_tensor": jnp.zeros((2, 3, 6, 3))}, ["(n_elements, 6, 3)"]),
        ({"electro_optic_tensor": _R_LN * 1j}, ["must be real"]),
        ({"electro_optic_tensor": _R_LN * 1e12}, ["m/V", "30.9e-12"]),
        ({"electro_optic_tensor": np.full((6, 3), np.inf)}, ["finite"]),
        (
            {
                "eps_r": jnp.ones(3),
                "axes": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                "electro_optic_tensor": _R_LN,
            },
            ["right-handed", "reverse"],
        ),
        ({"electro_optic_tensor": _R_LN}, ["optical eps_r", "n^4"]),
    ],
    ids=["3x3x3", "4d", "complex", "pm/V", "inf", "left-handed", "no-eps_r"],
)
def test_electro_optic_tensor_field_refusals_name_the_remedy(kwargs, words):
    with pytest.raises(ValueError, match="electro_optic_tensor") as e:
        gfem.MaterialField(**kwargs)
    for word in words:
        assert word in str(e.value)


def test_jax_grad_reaches_a_material_field_s_electro_optic_tensor():
    field = gfem.MaterialField(
        eps_r=jnp.full(3, 4.6), axes=_X_CUT, electro_optic_tensor=_R_LN
    )
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(field)
    ]
    assert ".electro_optic_tensor" in paths
    weight = jnp.arange(18.0).reshape(6, 3)

    def objective(f):
        return jnp.sum(weight * f.electro_optic_tensor) + jnp.sum(f.eps_r)

    grad = jax.grad(objective)(field)
    np.testing.assert_array_equal(grad.electro_optic_tensor, weight)
    # and when the field is built from a traced tensor
    built = jax.grad(
        lambda r: objective(gfem.MaterialField(eps_r=4.6, electro_optic_tensor=r))
    )(jnp.asarray(_R_LN))
    np.testing.assert_array_equal(built, weight)


def test_uniform_field_carries_the_electro_optic_tensor_in_the_material_s_frame():
    crystal = gfem.Material(
        eps_r=(4.6, 4.6, 4.9), axes=_X_CUT, electro_optic_tensor=_R_LN
    )
    f = gfem.MaterialField.uniform("core", crystal)
    assert f.axes == _X_CUT
    np.testing.assert_array_equal(f.electro_optic_tensor, _R_LN)
    # an isotropic poled material keeps its frame
    poled = gfem.Material(eps_r=2.5, axes=_X_CUT, electro_optic_tensor=_R_LN)
    g = gfem.MaterialField.uniform("core", poled)
    assert g.axes == _X_CUT
    np.testing.assert_array_equal(g.eps_r, [2.5, 2.5, 2.5])
    np.testing.assert_array_equal(g.electro_optic_tensor, _R_LN)
    assert gfem.MaterialField.uniform("r", gfem.Material(eps_r=2.0)).axes is None
    assert gfem.MaterialField.uniform("r", gfem.Material()).electro_optic_tensor is None


def test_interpolate_blends_the_electro_optic_tensor_with_a_missing_end_zero():
    crystal = gfem.Material(
        eps_r=(4.6, 4.6, 4.9), axes=_X_CUT, electro_optic_tensor=_R_LN
    )
    air = gfem.Material()
    expected = _r_global(_R_LN, _X_CUT)
    # r33 of an x-cut acts along global x: Delta(1/n^2)_xx = r33 E_x
    assert expected[0, 0] == 30.8e-12
    rho = jnp.array([0.0, 0.5, 1.0])
    f = gfem.MaterialField.interpolate(rho, air, crystal)
    assert f.electro_optic_tensor.shape == (3, 6, 3)
    assert np.linalg.det(np.asarray(f.axes)) > 0
    for k, w in enumerate((0.0, 0.5, 1.0)):
        np.testing.assert_allclose(
            _r_global(f.electro_optic_tensor[k], f.axes), w * expected, atol=1e-25
        )
    # two isotropic ends: an isotropic field, the tensor in the global frame
    poled = gfem.Material(eps_r=2.5, axes=_X_CUT, electro_optic_tensor=_R_LN)
    g = gfem.MaterialField.interpolate(rho, air, poled)
    assert g.axes is None
    np.testing.assert_allclose(g.electro_optic_tensor[2], expected, atol=1e-25)
    # the index scheme blends it in the global frame too
    h = gfem.MaterialField.interpolate(rho, air, poled, scheme="index")
    np.testing.assert_allclose(h.electro_optic_tensor[1], expected / 2, atol=1e-25)
    # ends without a tensor give a field without one
    assert gfem.MaterialField.interpolate(rho, air, air).electro_optic_tensor is None


@pytest.mark.parametrize("scheme", ["simp", "index"])
def test_interpolate_refuses_a_law_end_beside_an_electro_optic_end(scheme):
    # the tensor's frame needs a constant optical permittivity
    poled = gfem.Material(eps_r=2.5, axes=_X_CUT, electro_optic_tensor=_R_LN)
    law = gfem.Material(eps_r=_LAW)
    with pytest.raises(ValueError, match="electro_optic_tensor") as e:
        gfem.MaterialField.interpolate(jnp.array([0.0, 1.0]), law, poled, scheme=scheme)
    for word in ("PermittivityLaw", "constant eps_r", "axes"):
        assert word in str(e.value)


def test_an_electro_optic_material_and_field_round_trip_through_the_job_codec(
    tmp_path,
):
    mesh_path = sm.write_msh22(
        tmp_path / "stack.msh",
        sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
    )
    mesh = gfem.Mesh.from_gmsh(mesh_path)
    n = len(mesh.elements("oxide"))
    crystal = gfem.Material(
        eps_r=(4.6, 4.6, 4.9), axes=_X_CUT, electro_optic_tensor=_R_LN
    )
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.6),
        axes=_X_CUT,
        electro_optic_tensor=jnp.tile(jnp.asarray(_R_LN), (n, 1, 1)),
    )
    model = gfem.Model(mesh, materials={"oxide": field, "nitride": crystal})
    problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model.materials
    assert back["nitride"] == crystal
    assert back["oxide"].axes == _X_CUT
    assert back["oxide"].electro_optic_tensor.dtype == jnp.float64
    np.testing.assert_array_equal(
        back["oxide"].electro_optic_tensor, field.electro_optic_tensor
    )


def test_refinement_copies_a_per_element_electro_optic_tensor_to_the_children(
    tmp_path,
):
    from gfem._refine import refine_model

    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(
            tmp_path / "stack.msh",
            sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
        )
    )
    n = len(mesh.elements("oxide"))
    scale = jnp.linspace(0.5, 1.0, n)[:, None, None]
    field = gfem.MaterialField(
        eps_r=jnp.full(n, 4.6), electro_optic_tensor=scale * jnp.asarray(_R_LN)
    )
    model = gfem.Model(
        mesh, materials={"oxide": field, "nitride": gfem.Material(eps_r=7.5)}
    )
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[mesh.elements("oxide")[0]] = True
    fine, _ = refine_model(model, marked)
    r = fine.materials["oxide"].electro_optic_tensor
    assert r.shape == (len(fine.mesh.elements("oxide")), 6, 3)
    assert r.shape[0] > n
    assert set(np.round(np.asarray(r[:, 2, 2]) / 30.8e-12, 12)) == set(
        np.round(np.asarray(scale[:, 0, 0]), 12)
    )
