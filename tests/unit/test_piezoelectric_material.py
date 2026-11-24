"""A solid's stiffness, piezoelectric tensor and mechanical quality factor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem._materials import piezoelectric_in_frame, stiffness_in_frame

GPa = 1e9
# Warner's lithium niobate, crystal frame (Pa, C/m^2, kg/m^3)
_C11, _C12, _C13, _C14, _C33, _C44 = 203, 53, 75, 9, 245, 60
_C66 = (_C11 - _C12) / 2
C_LN = GPa * np.array(
    [
        [_C11, _C12, _C13, _C14, 0, 0],
        [_C12, _C11, _C13, -_C14, 0, 0],
        [_C13, _C13, _C33, 0, 0, 0],
        [_C14, -_C14, 0, _C44, 0, 0],
        [0, 0, 0, 0, _C44, _C14],
        [0, 0, 0, 0, _C14, _C66],
    ]
)
_E15, _E22, _E31, _E33 = 3.7, 2.5, 0.2, 1.3
E_LN = np.array(
    [
        [0, 0, 0, 0, _E15, -_E22],
        [-_E22, _E22, 0, _E15, 0, 0],
        [_E31, _E31, _E33, 0, 0, 0],
    ]
)
EPS_LN = (44.0, 44.0, 29.0)
RHO_LN = 4700.0
# X-cut, SAW along crystal Y (global x), film normal along crystal X (global y)
X_CUT = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
# Voigt order xx, yy, zz, yz, xz, xy
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _random_axes(seed):
    q = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
    return q * np.sign(np.linalg.det(q))


def _bond(q):
    """Auld's stress transformation matrix M of direction cosines ``q``."""
    m = np.zeros((6, 6))
    for row, (i, j) in enumerate(_VOIGT):
        for col, (k, l) in enumerate(_VOIGT):
            m[row, col] = q[i, k] * q[j, l] + (q[i, l] * q[j, k] if k != l else 0)
    return m


def _relative(a, b):
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


# --- frames ------------------------------------------------------------------------


@pytest.mark.parametrize("axes", [X_CUT, _random_axes(0), _random_axes(7)])
def test_stiffness_and_piezoelectric_tensor_rotate_as_bond_matrices(axes):
    # the global frame from the frame of axes: Q = a^T, global_i = Q_ik crystal_k
    q = np.asarray(axes).T
    m = _bond(q)
    assert _relative(stiffness_in_frame(C_LN, axes), m @ C_LN @ m.T) < 1e-13
    assert _relative(piezoelectric_in_frame(E_LN, axes), q @ E_LN @ m.T) < 1e-13
    # back to the crystal frame, and through JAX
    back = stiffness_in_frame(stiffness_in_frame(C_LN, axes), None, axes)
    assert _relative(back, C_LN) < 1e-13
    lab = piezoelectric_in_frame(jnp.asarray(E_LN), axes)
    assert isinstance(lab, jax.Array)
    assert _relative(lab, q @ E_LN @ m.T) < 1e-13


def test_rotation_keeps_leading_element_axes():
    c = np.stack([C_LN, 2 * C_LN])
    e = np.stack([E_LN, -E_LN, 0 * E_LN])
    axes = _random_axes(3)
    # each element rotates on its own, to round-off of the batched sums
    lab = stiffness_in_frame(c, axes)
    assert lab.shape == (2, 6, 6)
    assert _relative(lab[1], stiffness_in_frame(2 * C_LN, axes)) < 1e-14
    lab = piezoelectric_in_frame(e, axes)
    assert lab.shape == (3, 3, 6)
    assert _relative(lab[1], piezoelectric_in_frame(-E_LN, axes)) < 1e-14


def test_a_reflection_flips_c14_so_crystal_axes_are_right_handed():
    mirrored = stiffness_in_frame(C_LN, np.diag([1.0, 1.0, -1.0]))
    assert mirrored[0, 3] == -C_LN[0, 3]
    assert mirrored[0, 0] == C_LN[0, 0]
    left = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    for given in (
        {"stiffness": C_LN},
        {"stiffness": C_LN, "piezoelectric_tensor": E_LN},
    ):
        with pytest.raises(ValueError, match="left-handed") as e:
            gfem.Material(eps_r=EPS_LN, axes=left, **given)
        assert "stiffness is given in the frame" in str(e.value)
        assert "reverse one axis" in str(e.value)


# --- Material ----------------------------------------------------------------------


def _ln(**overrides):
    kwargs = {
        "eps_r": EPS_LN,
        "axes": X_CUT,
        "stiffness": C_LN,
        "piezoelectric_tensor": E_LN,
        "mass_density": RHO_LN,
        "mechanical_quality_factor": 1000.0,
        "name": "LN",
    }
    return gfem.Material(**(kwargs | overrides))


def test_a_material_keeps_its_tensors_in_the_frame_of_its_axes():
    m = _ln()
    assert m.stiffness == tuple(tuple(float(x) for x in row) for row in C_LN)
    assert m.piezoelectric_tensor == tuple(tuple(float(x) for x in row) for row in E_LN)
    assert m.mechanical_quality_factor == 1000.0
    assert m.axes == X_CUT
    assert hash(m) == hash(_ln())
    assert m == _ln()
    assert "stiffness" in repr(m)
    plain = gfem.Material()
    assert plain.stiffness is None
    assert plain.piezoelectric_tensor is None
    assert plain.mechanical_quality_factor is None
    # an isotropic solid is isotropic, and takes no axes
    al = gfem.Material(stiffness=gfem.isotropic_stiffness(70e9, 0.35))
    assert al.isotropic
    assert al.axes is None
    # a constant material has no leaves
    assert jax.tree_util.tree_leaves(m) == []


_LEFT = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_LAW = gfem.PermittivityLaw(
    eps_inf=40.0,
    terms=[gfem.Lorentz(delta_eps=4.0, resonance_frequency=6e12, damping=0.0)],
)


@pytest.mark.parametrize(
    ("overrides", "words"),
    [
        (
            {"stiffness": C_LN * (1 + 1e-3j)},
            ["stiffness must be real", "mechanical_quality_factor"],
        ),
        ({"stiffness": C_LN * np.nan}, ["stiffness must be finite"]),
        (
            {"stiffness": C_LN[:3, :3]},
            ["stiffness is", "6x6", "shape (6, 6)", "(3, 3)"],
        ),
        (
            {"stiffness": np.zeros((3, 3, 3, 3))},
            ["xx, yy, zz, yz, xz, xy", "contract a full c_ijkl", "no factors of 2"],
        ),
        ({"stiffness": C_LN / GPa}, ["stiffness", "in Pa", "203e9"]),
        (
            {"stiffness": C_LN + np.triu(np.ones((6, 6)), 1) * GPa},
            ["not symmetric", "major symmetry"],
        ),
        (
            {"stiffness": C_LN - 100 * GPa * np.eye(6)},
            ["not positive definite", "strain energy"],
        ),
        ({"piezoelectric_tensor": E_LN * 1j}, ["piezoelectric_tensor must be real"]),
        (
            {"piezoelectric_tensor": np.full((3, 6), np.inf)},
            ["piezoelectric_tensor must be finite"],
        ),
        ({"piezoelectric_tensor": E_LN.T}, ["3x6", "shape (3, 6)", "(6, 3)"]),
        (
            {"piezoelectric_tensor": np.zeros((3, 3, 3))},
            ["contract a full e_kij", "xx, yy, zz, yz, xz, xy"],
        ),
        (
            {"piezoelectric_tensor": E_LN * 1e-11},
            ["C/m^2", "strain constants d", "e = d c^E"],
        ),
        ({"stiffness": None}, ["piezoelectric_tensor needs a stiffness"]),
        (
            {"stiffness": None, "piezoelectric_tensor": None},
            ["mechanical_quality_factor needs a stiffness"],
        ),
        ({"mechanical_quality_factor": 0.0}, ["mechanical_quality_factor must be > 0"]),
        (
            {"mechanical_quality_factor": (1.0, 2.0, 3.0)},
            ["mechanical_quality_factor must be a single number"],
        ),
        ({"eps_r": 1.0}, ["piezoelectric_tensor needs", "clamped", "eps_r"]),
        (
            {"eps_r": np.diag(EPS_LN), "axes": None},
            ["eps_r is a 3x3 matrix", "beside a stiffness", "explicit `axes`"],
        ),
        ({"axes": _LEFT}, ["left-handed", "stiffness is given in the frame"]),
        (
            {"electro_optic_tensor": np.zeros((6, 3))},
            ["electro_optic_tensor", "two Materials", "acoustic", "optical"],
        ),
        (
            {"eps_r": _LAW, "axes": None},
            ["PermittivityLaw", "takes no axes", "constant eps_r"],
        ),
    ],
    ids=[
        "complex",
        "nan",
        "3x3",
        "3x3x3x3",
        "GPa",
        "asymmetric",
        "indefinite",
        "e-complex",
        "e-inf",
        "6x3",
        "e-3x3x3",
        "d-for-e",
        "e-without-c",
        "Qm-without-c",
        "Qm-zero",
        "Qm-tensor",
        "eps-1",
        "matrix",
        "left-handed",
        "electro-optic",
        "law",
    ],
)
def test_every_crystal_tensor_guard_names_the_remedy(overrides, words):
    with pytest.raises(ValueError) as e:
        _ln(**overrides)
    for word in words:
        assert word in str(e.value)
    assert "Material 'LN'" in str(e.value)


def test_crystal_tensors_beside_a_permittivity_law_raise_without_axes_too():
    for given in (
        {"stiffness": C_LN},
        {"piezoelectric_tensor": E_LN, "stiffness": C_LN},
    ):
        with pytest.raises(ValueError, match="PermittivityLaw"):
            gfem.Material(eps_r=_LAW, **given)


def test_unit_guards_pass_real_solids_and_an_all_zero_piezoelectric_tensor():
    # the softest c11 near 1e9 Pa and the weakest e near 0.05 C/m^2 pass
    soft = gfem.isotropic_stiffness(1e9, 0.45)
    quartz = np.zeros((3, 6))
    quartz[0, 0], quartz[0, 1], quartz[0, 3] = 0.171, -0.171, -0.0406
    gfem.Material(eps_r=4.5, stiffness=soft, piezoelectric_tensor=quartz)
    zero = gfem.Material(
        eps_r=4.5, stiffness=soft, piezoelectric_tensor=np.zeros((3, 6))
    )
    assert zero.piezoelectric_tensor == ((0.0,) * 6,) * 3


def test_crystal_tensors_under_a_gradient_raise_with_the_field_remedy():
    def objective(c11, which):
        c = jnp.asarray(C_LN).at[0, 0].set(c11)
        e = jnp.asarray(E_LN)
        given = (
            {"stiffness": c}
            if which == "c"
            else {"stiffness": C_LN, "piezoelectric_tensor": e * c11 / C_LN[0, 0]}
        )
        return gfem.Material(eps_r=EPS_LN, axes=X_CUT, **given).eps_r[0]

    for which in ("c", "e"):
        with pytest.raises(
            gfem.NonDifferentiableError, match=r"MaterialField\.uniform"
        ):
            jax.grad(objective)(C_LN[0, 0], which)


# --- MaterialField -----------------------------------------------------------------


def test_jax_grad_reaches_a_material_field_s_solid_tensors():
    n = 3
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 44.0),
        axes=X_CUT,
        stiffness=jnp.tile(jnp.asarray(C_LN), (n, 1, 1)),
        piezoelectric_tensor=E_LN,
        mechanical_quality_factor=jnp.full(n, 1000.0),
        mass_density=RHO_LN,
    )
    assert field.n_elements == n
    assert field.stiffness.dtype == jnp.float64
    assert {"stiffness", "piezoelectric_tensor", "mechanical_quality_factor"} <= set(
        field.properties
    )
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(field)
    ]
    for name in ("stiffness", "piezoelectric_tensor", "mechanical_quality_factor"):
        assert f".{name}" in paths
    wc = jnp.arange(36.0).reshape(6, 6)
    we = jnp.arange(18.0).reshape(3, 6)

    def objective(f):
        return (
            jnp.sum(wc * f.stiffness) / GPa
            + jnp.sum(we * f.piezoelectric_tensor)
            + jnp.sum(f.mechanical_quality_factor)
        )

    grad = jax.grad(objective)(field)
    np.testing.assert_array_equal(grad.stiffness, jnp.tile(wc / GPa, (n, 1, 1)))
    np.testing.assert_array_equal(grad.piezoelectric_tensor, we)
    np.testing.assert_array_equal(grad.mechanical_quality_factor, jnp.ones(n))
    # through isotropic_stiffness, by the chain rule
    lam_mu = jax.grad(
        lambda e: jnp.sum(
            gfem.MaterialField(stiffness=gfem.isotropic_stiffness(e, 0.35)).stiffness
        )
    )(jnp.asarray(70e9))
    # d/dE of 9 lambda + 6 mu + 3 mu = 9 lambda / E + 9 mu / E
    nu = 0.35
    expected = 9 * nu / ((1 + nu) * (1 - 2 * nu)) + 9 / (2 * (1 + nu))
    assert abs(float(lam_mu) - expected) < 1e-15 * expected


def test_a_field_s_per_element_sizes_must_agree():
    with pytest.raises(ValueError, match="same number of values"):
        gfem.MaterialField(
            eps_r=jnp.ones(2) * 44,
            stiffness=jnp.tile(jnp.asarray(C_LN), (3, 1, 1)),
        )


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        (
            {"stiffness": jnp.zeros((3, 3, 3, 3))},
            ["(6, 6)", "(n_elements, 6, 6)", "contract a full c_ijkl"],
        ),
        (
            {"stiffness": C_LN * 1j},
            ["stiffness must be real", "mechanical_quality_factor"],
        ),
        ({"stiffness": C_LN / GPa}, ["in Pa", "203e9"]),
        ({"stiffness": np.stack([C_LN, -C_LN])}, ["not positive definite"]),
        (
            {"stiffness": C_LN, "piezoelectric_tensor": E_LN * 1e-11},
            ["strain constants d"],
        ),
        (
            {"stiffness": C_LN, "piezoelectric_tensor": jnp.zeros((2, 6, 3))},
            ["(3, 6)", "(n_elements, 3, 6)"],
        ),
        (
            {"piezoelectric_tensor": E_LN, "eps_r": 44.0},
            ["piezoelectric_tensor needs a stiffness"],
        ),
        (
            {"mechanical_quality_factor": 100.0},
            ["mechanical_quality_factor needs a stiffness"],
        ),
        (
            {"stiffness": C_LN, "mechanical_quality_factor": -1.0},
            ["mechanical_quality_factor must be > 0"],
        ),
        ({"stiffness": C_LN, "piezoelectric_tensor": E_LN}, ["clamped", "eps_r"]),
        (
            {"stiffness": C_LN, "eps_r": jnp.ones(3), "axes": _LEFT},
            ["left-handed", "stiffness is given in the frame"],
        ),
        (
            {"stiffness": C_LN, "eps_r": 4.6, "electro_optic_tensor": np.zeros((6, 3))},
            ["two Materials"],
        ),
    ],
    ids=[
        "3x3x3x3",
        "complex",
        "GPa",
        "indefinite",
        "d-for-e",
        "e-shape",
        "e-without-c",
        "Qm-without-c",
        "Qm-negative",
        "no-eps_r",
        "left-handed",
        "electro-optic",
    ],
)
def test_material_field_solid_refusals_name_the_remedy(kwargs, words):
    with pytest.raises(ValueError) as e:
        gfem.MaterialField(**kwargs)
    for word in words:
        assert word in str(e.value)


def test_a_traced_field_stiffness_is_checked_at_solve_time():
    from gfem._materials import check_field_values

    def build(scale):
        return gfem.MaterialField(stiffness=scale * jnp.asarray(C_LN))

    traced = jax.jit(build)(-1.0)  # the constructor saw a tracer
    with pytest.raises(ValueError, match=r"positive definite.*region 'film'"):
        check_field_values({"film": traced})


def test_uniform_field_carries_the_solid_tensors_in_the_material_s_frame():
    f = gfem.MaterialField.uniform("film", _ln())
    assert f.axes == X_CUT
    np.testing.assert_array_equal(f.stiffness, C_LN)
    np.testing.assert_array_equal(f.piezoelectric_tensor, E_LN)
    assert f.mechanical_quality_factor == 1000.0
    np.testing.assert_array_equal(f.eps_r, EPS_LN)
    # an isotropic solid with axes keeps its frame; without, the global one
    c = gfem.isotropic_stiffness(70e9, 0.35)
    turned = gfem.MaterialField.uniform(
        "finger", gfem.Material(stiffness=c, axes=X_CUT, mass_density=2700.0)
    )
    assert turned.axes == X_CUT
    plain = gfem.MaterialField.uniform("finger", gfem.Material(stiffness=c))
    assert plain.axes is None
    assert plain.piezoelectric_tensor is None
    np.testing.assert_array_equal(plain.stiffness, c)


def test_interpolate_blends_solid_tensors_linearly_and_qm_through_its_loss():
    rho = jnp.array([0.0, 0.25, 1.0])
    al = gfem.Material(
        stiffness=gfem.isotropic_stiffness(70e9, 0.35),
        mass_density=2700.0,
        mechanical_quality_factor=100.0,
    )
    ln = _ln()
    f = gfem.MaterialField.interpolate(rho, al, ln)
    assert np.linalg.det(np.asarray(f.axes)) > 0
    c_al = np.asarray(gfem.isotropic_stiffness(70e9, 0.35))
    c_ln = stiffness_in_frame(C_LN, X_CUT)
    e_ln = piezoelectric_in_frame(E_LN, X_CUT)
    for k, w in enumerate((0.0, 0.25, 1.0)):
        c = stiffness_in_frame(np.asarray(f.stiffness[k]), f.axes)
        assert _relative(c, c_al + w * (c_ln - c_al)) < 1e-14
        e = piezoelectric_in_frame(np.asarray(f.piezoelectric_tensor[k]), f.axes)
        np.testing.assert_allclose(e, w * e_ln, rtol=0, atol=1e-14)
        # the loss 1 / Q_m is linear in the density
        loss = 1 / 100.0 + w * (1 / 1000.0 - 1 / 100.0)
        assert abs(float(f.mechanical_quality_factor[k]) * loss - 1) < 1e-14
    np.testing.assert_allclose(f.mass_density, [2700.0, 2700 + 0.25 * 2000, 4700.0])
    # two isotropic ends: the global frame
    sio2 = gfem.Material(
        stiffness=gfem.isotropic_stiffness(72e9, 0.17), mechanical_quality_factor=50.0
    )
    g = gfem.MaterialField.interpolate(
        rho,
        gfem.Material(stiffness=c_al, mechanical_quality_factor=100.0),
        sio2,
        scheme="index",
    )
    assert g.axes is None
    assert g.stiffness.shape == (3, 6, 6)
    assert g.piezoelectric_tensor is None
    # lossless ends stay lossless
    h = gfem.MaterialField.interpolate(
        rho, gfem.Material(stiffness=c_al), gfem.Material(stiffness=2 * c_al)
    )
    assert h.mechanical_quality_factor is None
    assert _relative(h.stiffness[1], 1.25 * c_al) < 1e-15


@pytest.mark.parametrize(
    ("void", "words"),
    [
        (gfem.Material(), ["void end has no stiffness", "singular"]),
        (
            gfem.Material(stiffness=gfem.isotropic_stiffness(70e9, 0.35)),
            ["mechanical_quality_factor", "void end", "both ends"],
        ),
    ],
    ids=["air", "lossless"],
)
def test_interpolate_refuses_an_end_the_blend_cannot_reach(void, words):
    with pytest.raises(ValueError) as e:
        gfem.MaterialField.interpolate(jnp.array([0.0, 1.0]), void, _ln())
    for word in words:
        assert word in str(e.value)


# --- isotropic solids and bulk waves -----------------------------------------------


def test_isotropic_stiffness_is_the_lame_closed_form():
    c = gfem.isotropic_stiffness(70e9, 0.35)
    assert isinstance(c, np.ndarray)
    # aluminium: lambda = 60.4938..., mu = 25.9259... GPa
    lam, mu = 70e9 * 0.35 / (1.35 * 0.3), 70e9 / 2.7
    expected = np.zeros((6, 6))
    expected[:3, :3] = lam
    expected += np.diag([2 * mu] * 3 + [mu] * 3)
    assert _relative(c, expected) < 1e-15
    assert abs(c[0, 0] / GPa - 112.34567901234566) < 1e-12
    # the same in every frame
    assert _relative(stiffness_in_frame(c, _random_axes(1)), c) < 1e-14
    # bulk speeds of aluminium (probe values, m/s)
    assert abs(np.sqrt(c[3, 3] / 2700) - 3098.74) < 0.01
    assert abs(np.sqrt(c[0, 0] / 2700) - 6450.54) < 0.01
    assert isinstance(gfem.isotropic_stiffness(jnp.asarray(70e9), 0.35), jax.Array)


@pytest.mark.parametrize(
    ("e", "nu", "word"),
    [
        (0.0, 0.3, "young_modulus"),
        (70e9, 0.5, "poisson_ratio"),
        (70e9, -1.0, "poisson_ratio"),
    ],
)
def test_isotropic_stiffness_refuses_a_solid_that_is_not_stable(e, nu, word):
    with pytest.raises(ValueError, match=word):
        gfem.isotropic_stiffness(e, nu)


def _speeds(c, e, eps, rho, n):
    """Bulk speeds along ``n`` from the Christoffel matrix, with and without e."""
    voigt = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    full_c = c[voigt][:, :, voigt]
    full_e = e[:, voigt]
    gamma = np.einsum("ijkl,j,l->ik", full_c, n, n)
    out = [np.sort(np.sqrt(np.linalg.eigvalsh(gamma) / rho))]
    pe = np.einsum("pij,p,j->i", full_e, n, n)
    stiffened = gamma + np.outer(pe, pe) / (n @ eps @ n)
    out.append(np.sort(np.sqrt(np.linalg.eigvalsh(stiffened) / rho)))
    return out


@pytest.mark.parametrize(
    ("n", "plain", "stiffened"),
    [
        # global x is crystal Y, the SAW axis of the x-cut frame
        ((1, 0, 0), (3556.1, 3994.7, 6581.1), (3994.7, 4458.1, 6853.7)),
        # global y is crystal X
        ((0, 1, 0), (3445.2, 4105.4, 6572.0), (4079.2, 4793.1, 6572.0)),
        # global z is minus crystal Z
        ((0, 0, 1), (3572.9, 3572.9, 7219.9), (3572.9, 3572.9, 7316.3)),
    ],
    ids=["x", "y", "z"],
)
def test_christoffel_speeds_of_the_rotated_crystal(n, plain, stiffened):
    m = _ln()
    c = stiffness_in_frame(np.asarray(m.stiffness), m.axes)
    e = piezoelectric_in_frame(np.asarray(m.piezoelectric_tensor), m.axes)
    a = np.asarray(m.axes)
    eps = 8.8541878128e-12 * a.T @ np.diag(m.eps_r) @ a
    v, vs = _speeds(c, e, eps, m.mass_density, np.asarray(n, dtype=float))
    np.testing.assert_allclose(v, plain, rtol=1e-3)
    np.testing.assert_allclose(vs, stiffened, rtol=1e-3)


def test_refinement_copies_per_element_solid_tensors_to_the_children(tmp_path):
    import structured_meshes as sm

    from gfem._refine import refine_model

    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(
            tmp_path / "stack.msh",
            sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
        )
    )
    n = len(mesh.elements("oxide"))
    scale = jnp.linspace(1.0, 2.0, n)
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 44.0),
        axes=X_CUT,
        stiffness=scale[:, None, None] * jnp.asarray(C_LN),
        piezoelectric_tensor=scale[:, None, None] * jnp.asarray(E_LN),
        mechanical_quality_factor=1000 * scale,
    )
    model = gfem.Model(
        mesh, materials={"oxide": field, "nitride": gfem.Material(eps_r=7.5)}
    )
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[mesh.elements("oxide")[0]] = True
    fine, _ = refine_model(model, marked)
    refined = fine.materials["oxide"]
    n_fine = len(fine.mesh.elements("oxide"))
    assert n_fine > n
    assert refined.stiffness.shape == (n_fine, 6, 6)
    assert refined.piezoelectric_tensor.shape == (n_fine, 3, 6)
    ratio = np.asarray(refined.stiffness[:, 0, 0]) / C_LN[0, 0]
    np.testing.assert_allclose(
        np.asarray(refined.piezoelectric_tensor[:, 0, 4]) / E_LN[0, 4], ratio
    )
    np.testing.assert_allclose(
        np.asarray(refined.mechanical_quality_factor), 1000 * ratio
    )
    assert set(np.round(ratio, 12)) == set(np.round(np.asarray(scale), 12))


# --- the mechanical domain's missing properties ------------------------------------


def test_missing_mechanical_properties_list_every_pair_and_both_remedies():
    err = gfem.MissingMaterialPropertyError(
        [
            ("clad", "stiffness"),
            ("clad", "mass_density"),
            ("finger", "mass_density"),
            ("clad", "stiffness"),
        ],
        physics="mechanical",
    )
    assert err.missing == (
        ("clad", "stiffness"),
        ("clad", "mass_density"),
        ("finger", "mass_density"),
    )
    text = str(err)
    assert text.startswith("A piezoelectric problem needs material properties")
    assert "region 'clad': stiffness (Pa, 6x6" in text
    assert "mass_density (kg/m^3" in text
    assert "region 'finger': mass_density" in text
    assert "gfem.isotropic_stiffness" in text
    assert "excluded={'mechanical': ('clad', 'finger')}" in text
    # any property name, unit or not
    assert "region 'x': whatever" in str(
        gfem.MissingMaterialPropertyError([("x", "whatever")], physics="mechanical")
    )


def test_missing_thermal_properties_read_as_before():
    text = str(gfem.MissingMaterialPropertyError([("si", "thermal_conductivity")]))
    assert text == (
        "A thermal problem needs material properties that are not given: region "
        "'si': thermal_conductivity (W/(m K), e.g. 1.4). Give each on the region's "
        "material (for example gfem.Material(thermal_conductivity=1.4, "
        "mass_density=2200, specific_heat=700) for silica), or leave the region "
        "out of thermal problems with Model(excluded={'thermal': ('si',)})."
    )


def test_solid_materials_round_trip_through_the_job_codec(tmp_path):
    import json

    import structured_meshes as sm

    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(
            tmp_path / "stack.msh",
            sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
        )
    )
    n = len(mesh.elements("oxide"))
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 44.0),
        axes=X_CUT,
        stiffness=jnp.linspace(1, 2, n)[:, None, None] * jnp.asarray(C_LN),
        piezoelectric_tensor=E_LN,
        mechanical_quality_factor=jnp.full(n, 800.0),
        mass_density=RHO_LN,
    )
    model = gfem.Model(
        mesh,
        materials={"oxide": field, "nitride": _ln()},
        excluded={"mechanical": ("nitride",)},
    )
    problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model
    assert back.materials["nitride"] == _ln()
    assert back.excluded_from("mechanical") == ("nitride",)
    oxide = back.materials["oxide"]
    assert oxide.axes == X_CUT
    for name in ("stiffness", "piezoelectric_tensor", "mechanical_quality_factor"):
        assert getattr(oxide, name).dtype == jnp.float64
        np.testing.assert_array_equal(getattr(oxide, name), getattr(field, name))
    # a material without them is written as before
    plain = gfem.Model(
        mesh,
        materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material()},
    )
    doc = json.loads(gfem.Job(plain, problem).to_json(tmp_path / "p.json").read_text())
    written = doc["job"]["model"]["materials"]["oxide"]
    assert not {"stiffness", "piezoelectric_tensor", "mechanical_quality_factor"} & set(
        written
    )
