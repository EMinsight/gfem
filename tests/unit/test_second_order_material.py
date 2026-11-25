"""A material's second-order nonlinear tensor: checks, frame, blend, codec."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _chi2
from gfem._materials import check_field_values
from gfem._refine import refine_model
from gfem._solve_cache import solve_key

# lithium niobate (3m), crystal frame, pm/V converted to m/V
_D15, _D22, _D31, _D33 = 4.6e-12, 2.2e-12, 4.6e-12, 25.0e-12
D_LN = np.array(
    [
        [0, 0, 0, 0, _D15, -_D22],
        [-_D22, _D22, 0, _D15, 0, 0],
        [_D31, _D31, _D33, 0, 0, 0],
    ]
)
X_CUT = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
_LEFT = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_EPS = (2.21**2, 2.21**2, 2.14**2)
_LAW = gfem.PermittivityLaw(
    eps_inf=40.0,
    terms=[gfem.Lorentz(delta_eps=4.0, resonance_frequency=6e12, damping=0.0)],
)
_VI, _VJ = np.array([0, 1, 2, 1, 0, 0]), np.array([0, 1, 2, 2, 2, 1])
_ROW = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
# the round-off of one rotated entry: 27 products of an entry and three
# direction cosines (each of modulus <= 1) summed, so at most 27 * 3 eps of the
# largest entry; two evaluations (a different order of summation) differ by at
# most twice that
_ROUND = 2 * 27 * 3 * np.finfo(float).eps


def _ln(**overrides):
    kwargs = {
        "eps_r": _EPS,
        "axes": X_CUT,
        "second_order_nonlinear_tensor": D_LN,
        "name": "LN",
    }
    return gfem.Material(**(kwargs | overrides))


def _random_axes(seed):
    q = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
    return q * np.sign(np.linalg.det(q))


def _rotated(d, q):
    """The direct einsum :math:`d^t_{pqs} = Q_{pl} Q_{qm} Q_{sn} d^s_{lmn}`."""
    full = d[..., _ROW]
    out = np.einsum("pl,qm,sn,...lmn->...pqs", q, q, q, full)
    return out[..., _VI, _VJ]


# --- Material ----------------------------------------------------------------------


def test_a_material_keeps_its_tensor_and_has_none_by_default():
    m = _ln()
    assert m.second_order_nonlinear_tensor == tuple(
        tuple(float(x) for x in r) for r in D_LN
    )
    assert m == _ln()
    assert hash(m) == hash(_ln())
    assert "second_order_nonlinear_tensor" in repr(m)
    assert gfem.Material().second_order_nonlinear_tensor is None
    # a constant material has no leaves
    assert jax.tree_util.tree_leaves(m) == []
    # a cubic crystal: isotropic permittivity, no axes
    gaas = gfem.Material(
        eps_r=3.4**2, second_order_nonlinear_tensor=np.eye(3, 6, 3) * 100e-12
    )
    assert gaas.axes is None


@pytest.mark.parametrize(
    ("overrides", "words"),
    [
        (
            {"second_order_nonlinear_tensor": D_LN * (1 + 1e-3j)},
            ["must be real", "no nonlinear loss is modelled"],
        ),
        ({"second_order_nonlinear_tensor": D_LN * np.nan}, ["must be finite"]),
        (
            {"second_order_nonlinear_tensor": D_LN[:, :3]},
            ["the contracted 3x6", "shape (3, 6)", "(3, 3)"],
        ),
        (
            {"second_order_nonlinear_tensor": np.zeros((3, 3, 3))},
            ["contract a full d_ijk", "xx, yy, zz, yz, xz, xy", "no factors of 2"],
        ),
        (
            {"second_order_nonlinear_tensor": D_LN * 1e3},
            ["entry of", "above 1e-08 m/V", "25 pm/V is 25e-12"],
        ),
        (
            {"stiffness": np.eye(6) * 2e11},
            ["stiffness sits beside a second_order_nonlinear_tensor", "two Materials"],
        ),
        (
            {"eps_r": np.diag(_EPS), "axes": None},
            ["eps_r is a 3x3 matrix", "second_order_nonlinear_tensor", "explicit"],
        ),
        (
            {"axes": _LEFT},
            ["left-handed", "second_order_nonlinear_tensor is given in the frame"],
        ),
        (
            {"eps_r": 1.0, "axes": None},
            ["second_order_nonlinear_tensor needs the crystal's optical eps_r"],
        ),
    ],
    ids=[
        "complex",
        "nan",
        "3x3",
        "3x3x3",
        "pm-per-V",
        "beside-stiffness",
        "matrix",
        "left-handed",
        "vacuum",
    ],
)
def test_every_material_guard_names_the_remedy(overrides, words):
    with pytest.raises(ValueError) as e:
        _ln(**overrides)
    for word in words:
        assert word in str(e.value)
    assert "Material 'LN'" in str(e.value)


def test_the_largest_entry_passes_and_the_bound_is_an_entry_of_1e_8():
    gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=np.eye(3, 6) * 1e-8)
    with pytest.raises(ValueError, match="above 1e-08"):
        gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=np.eye(3, 6) * 1.0001e-8)
    # a negative entry counts by magnitude
    with pytest.raises(ValueError, match="above 1e-08"):
        gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=-np.eye(3, 6) * 2e-8)


def test_a_traced_tensor_in_a_material_raises_with_the_field_remedy():
    def build(s):
        return gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=s * D_LN)

    with pytest.raises(gfem.NonDifferentiableError) as e:
        jax.jit(build)(1.0)
    assert "MaterialField.uniform" in str(e.value)
    assert "second_order_nonlinear_tensor" in str(e.value)


def test_a_permittivity_law_is_accepted_with_the_tensor_in_the_global_frame():
    m = gfem.Material(eps_r=_LAW, second_order_nonlinear_tensor=D_LN)
    assert m.axes is None
    assert m.second_order_nonlinear_tensor[2][2] == pytest.approx(_D33)
    # a law takes no axes: the tensor cannot be given in a frame
    with pytest.raises(ValueError, match="axes"):
        gfem.Material(eps_r=_LAW, axes=X_CUT, second_order_nonlinear_tensor=D_LN)


def test_the_stiffness_beside_it_in_a_law_material_is_refused():
    with pytest.raises(ValueError, match="stiffness sits beside"):
        gfem.Material(
            eps_r=_LAW, stiffness=np.eye(6) * 2e11, second_order_nonlinear_tensor=D_LN
        )


# --- frame -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_frame_rotation_is_the_direct_einsum(seed):
    axes = _random_axes(seed)
    want = _rotated(D_LN, axes.T)
    got = _chi2.in_global_frame(D_LN, axes)
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, want, rtol=0, atol=_ROUND * np.abs(D_LN).max())
    # traced values stay traced
    jgot = jax.jit(lambda d: _chi2.in_global_frame(d, axes))(jnp.asarray(D_LN))
    np.testing.assert_allclose(jgot, want, rtol=0, atol=_ROUND * np.abs(D_LN).max())
    # a frame and its inverse give the tensor back
    back = _chi2.in_global_frame(got, axes.T)
    np.testing.assert_allclose(back, D_LN, rtol=0, atol=2 * _ROUND * np.abs(D_LN).max())


def test_the_lithium_niobate_tensor_is_invariant_under_a_three_fold_turn_about_z():
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # axes are the rows of the rotated crystal frame: Q = axes^T = rot
    got = _chi2.in_global_frame(D_LN, rot.T)
    # the 3m form holds to its own digits (a few pm/V entries, 1e-12 m/V);
    # the rotated entries are sums of 27 products of cosines 1/2 and sqrt(3)/2
    np.testing.assert_allclose(got, D_LN, rtol=0, atol=_ROUND * np.abs(D_LN).max())


def test_the_x_mirror_leaves_it_and_a_reversed_domain_flips_every_entry():
    mirror = np.diag([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        _rotated(D_LN, mirror), D_LN, rtol=0, atol=_ROUND * np.abs(D_LN).max()
    )
    # inversion: the domain reversed of a 3m crystal carries -d (odd rank)
    np.testing.assert_allclose(
        _chi2.in_global_frame(D_LN, -np.eye(3)),
        -D_LN,
        rtol=0,
        atol=_ROUND * np.abs(D_LN).max(),
    )


def test_the_contraction_is_the_sum_over_the_two_field_indices():
    rng = np.random.default_rng(0)
    f = rng.normal(size=3) + 1j * rng.normal(size=3)
    g = rng.normal(size=3) + 1j * rng.normal(size=3)
    got = _chi2.contract(D_LN, f, g)
    full = D_LN[:, _ROW]
    want = np.einsum("ijk,j,k->i", full, f, g)
    np.testing.assert_allclose(
        got, want, rtol=0, atol=20 * np.finfo(float).eps * 25e-12 * 10
    )
    # symmetric in the two fields
    np.testing.assert_allclose(
        _chi2.contract(D_LN, g, f),
        got,
        rtol=0,
        atol=20 * np.finfo(float).eps * 25e-12 * 10,
    )
    # d_zzz alone: the Ez field of a line port
    ez = np.array([0, 0, 1.0 + 0j])
    np.testing.assert_allclose(_chi2.contract(D_LN, ez, ez), [0, 0, _D33], rtol=1e-15)


# --- MaterialField -----------------------------------------------------------------


def test_uniform_field_carries_the_tensor_in_the_material_s_frame():
    f = gfem.MaterialField.uniform("film", _ln())
    assert f.axes == X_CUT
    np.testing.assert_array_equal(f.second_order_nonlinear_tensor, D_LN)
    assert "second_order_nonlinear_tensor" in f.properties
    assert f.n_elements is None
    turned = gfem.MaterialField.uniform(
        "film",
        gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=D_LN, axes=X_CUT),
    )
    assert turned.axes == X_CUT


def test_jax_grad_reaches_a_field_s_tensor_per_element_and_uniform():
    n = 3
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        second_order_nonlinear_tensor=jnp.tile(jnp.asarray(D_LN), (n, 1, 1)),
    )
    assert field.n_elements == n
    assert field.second_order_nonlinear_tensor.dtype == jnp.float64
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(field)
    ]
    assert ".second_order_nonlinear_tensor" in paths
    w = jnp.arange(18.0).reshape(3, 6)
    # the field hands its leaves to the objective unchanged: the gradient is the
    # objective's own weight, bit for bit
    grad = jax.grad(lambda f: jnp.sum(w * f.second_order_nonlinear_tensor))(field)
    np.testing.assert_array_equal(
        grad.second_order_nonlinear_tensor, jnp.tile(w, (n, 1, 1))
    )
    one = gfem.MaterialField.uniform("film", _ln())
    grad = jax.grad(lambda f: jnp.sum(w * f.second_order_nonlinear_tensor))(one)
    np.testing.assert_array_equal(grad.second_order_nonlinear_tensor, w)


def test_a_traced_value_is_checked_at_solve_time_and_a_field_checks_its_shape():
    traced = jax.jit(
        lambda s: gfem.MaterialField(eps_r=4.0, second_order_nonlinear_tensor=s * D_LN)
    )(1e3)
    with pytest.raises(ValueError, match=r"above 1e-08.*region 'film'"):
        check_field_values({"film": traced})
    with pytest.raises(ValueError, match="same number of values"):
        gfem.MaterialField(
            eps_r=jnp.ones(2) * 4.0,
            second_order_nonlinear_tensor=jnp.tile(jnp.asarray(D_LN), (3, 1, 1)),
        )


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        (
            {"second_order_nonlinear_tensor": jnp.zeros((3, 3, 3))},
            ["(3, 6)", "(n_elements, 3, 6)", "contract a full d_ijk"],
        ),
        (
            {"second_order_nonlinear_tensor": jnp.asarray(D_LN) * 1j},
            ["second_order_nonlinear_tensor must be real"],
        ),
        (
            {"second_order_nonlinear_tensor": jnp.asarray(D_LN) * 1e3},
            ["above 1e-08"],
        ),
        (
            {
                "second_order_nonlinear_tensor": jnp.asarray(D_LN),
                "stiffness": jnp.eye(6) * 2e11,
            },
            ["stiffness sits beside a second_order_nonlinear_tensor"],
        ),
        (
            {"second_order_nonlinear_tensor": jnp.asarray(D_LN), "axes": _LEFT},
            ["left-handed", "second_order_nonlinear_tensor"],
        ),
        (
            {"second_order_nonlinear_tensor": jnp.asarray(D_LN)},
            ["second_order_nonlinear_tensor needs the crystal's optical eps_r"],
        ),
    ],
    ids=["shape", "complex", "pm-per-V", "beside-stiffness", "left-handed", "no-eps"],
)
def test_material_field_refusals_name_the_remedy(kwargs, words):
    with pytest.raises(ValueError) as e:
        gfem.MaterialField(**kwargs)
    for word in words:
        assert word in str(e.value)


# --- blends ------------------------------------------------------------------------


def test_interpolate_blends_the_tensor_linearly_in_the_field_frame():
    rho = jnp.array([0.0, 0.25, 1.0])
    oxide = gfem.Material(eps_r=1.444**2)
    ln = _ln()
    f = gfem.MaterialField.interpolate(rho, oxide, ln)
    assert np.linalg.det(np.asarray(f.axes)) > 0
    # the blend frame is the crystal's: the end without a tensor counts as zero
    d_ln = _chi2.in_global_frame(D_LN, X_CUT)
    for k, w in enumerate((0.0, 0.25, 1.0)):
        d = _chi2.in_global_frame(
            np.asarray(f.second_order_nonlinear_tensor[k]), f.axes
        )
        np.testing.assert_allclose(
            d, w * d_ln, rtol=0, atol=_ROUND * np.abs(D_LN).max()
        )
    # two ends with tensors: linear in rho, each in the blend frame
    flat = gfem.Material(eps_r=2.0, second_order_nonlinear_tensor=np.eye(3, 6) * 1e-12)
    g = gfem.MaterialField.interpolate(rho, flat, ln)
    d_flat = _chi2.in_global_frame(np.eye(3, 6) * 1e-12, None)
    for k, w in enumerate((0.0, 0.25, 1.0)):
        d = _chi2.in_global_frame(
            np.asarray(g.second_order_nonlinear_tensor[k]), g.axes
        )
        np.testing.assert_allclose(
            d,
            d_flat + w * (d_ln - d_flat),
            rtol=0,
            atol=_ROUND * np.abs(D_LN).max(),
        )
    # the index scheme blends in the global frame
    h = gfem.MaterialField.interpolate(
        rho,
        gfem.Material(eps_r=2.0, second_order_nonlinear_tensor=np.eye(3, 6) * 1e-12),
        gfem.Material(eps_r=4.0, second_order_nonlinear_tensor=np.eye(3, 6) * 3e-12),
        scheme="index",
    )
    assert h.axes is None
    np.testing.assert_allclose(
        h.second_order_nonlinear_tensor[1], np.eye(3, 6) * 1.5e-12, rtol=1e-14
    )
    # no tensor at either end: none in the field
    assert (
        gfem.MaterialField.interpolate(
            rho, gfem.Material(), gfem.Material(eps_r=2.0)
        ).second_order_nonlinear_tensor
        is None
    )


def test_interpolate_accepts_a_law_end_with_the_tensor_in_the_global_frame():
    f = gfem.MaterialField.interpolate(
        jnp.array([0.0, 0.5, 1.0]),
        gfem.Material(eps_r=_LAW),
        gfem.Material(eps_r=_LAW, second_order_nonlinear_tensor=D_LN),
    )
    assert f.blend is not None
    assert f.axes is None
    np.testing.assert_allclose(
        f.second_order_nonlinear_tensor[1], 0.5 * D_LN, rtol=1e-14
    )


def test_interpolate_is_differentiable_in_the_density():
    gain = jax.grad(
        lambda r: jnp.sum(
            gfem.MaterialField.interpolate(
                r, gfem.Material(eps_r=2.0), _ln(), scheme="linear"
            ).second_order_nonlinear_tensor[..., 2, 2]
        )
    )(jnp.array([0.2, 0.7]))
    # the zz entry in the x-cut frame is d_xxx of the crystal: linear in rho
    d_zz = _chi2.in_global_frame(D_LN, X_CUT)[2, 2]
    np.testing.assert_allclose(gain, d_zz, rtol=1e-13)


# --- refinement, codec and the solve key -------------------------------------------


def _stack(tmp_path):
    return gfem.Mesh.from_gmsh(
        sm.write_msh22(
            tmp_path / "stack.msh",
            sm.plate(4.0, 3.0, (("oxide", 0.5), ("nitride", 0.25)), (1, 1, 1)),
        )
    )


def test_refinement_copies_a_per_element_tensor_to_the_children(tmp_path):
    mesh = _stack(tmp_path)
    n = len(mesh.elements("oxide"))
    scale = jnp.linspace(1.0, 2.0, n)
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        second_order_nonlinear_tensor=scale[:, None, None] * jnp.asarray(D_LN),
    )
    model = gfem.Model(
        mesh, materials={"oxide": field, "nitride": gfem.Material(eps_r=7.5)}
    )
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[mesh.elements("oxide")[0]] = True
    fine, _ = refine_model(model, marked)
    refined = fine.materials["oxide"].second_order_nonlinear_tensor
    assert refined.shape == (len(fine.mesh.elements("oxide")), 3, 6)
    ratio = np.asarray(refined[:, 2, 2]) / _D33
    wanted = np.asarray(scale)
    nearest = wanted[np.argmin(np.abs(ratio[:, None] - wanted[None, :]), axis=1)]
    # each child keeps its parent's scale, to the round-off of one division
    np.testing.assert_allclose(ratio, nearest, rtol=1e-14, atol=0)


def test_materials_round_trip_through_the_job_codec_and_the_solve_key(tmp_path):
    mesh = _stack(tmp_path)
    n = len(mesh.elements("oxide"))
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        second_order_nonlinear_tensor=jnp.linspace(1, 2, n)[:, None, None]
        * jnp.asarray(D_LN),
    )
    model = gfem.Model(mesh, materials={"oxide": field, "nitride": _ln()})
    problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model
    assert back.materials["nitride"] == _ln()
    oxide = back.materials["oxide"]
    assert oxide.second_order_nonlinear_tensor.dtype == jnp.float64
    np.testing.assert_array_equal(
        oxide.second_order_nonlinear_tensor, field.second_order_nonlinear_tensor
    )
    # a material without one is written as before: the file and the key of a
    # linear solve carry no trace of the property
    plain = gfem.Model(
        mesh, materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material()}
    )
    doc = json.loads(gfem.Job(plain, problem).to_json(tmp_path / "p.json").read_text())
    assert "second_order_nonlinear_tensor" not in json.dumps(doc)
    explicit = gfem.Model(
        mesh,
        materials={
            "oxide": gfem.Material(eps_r=3.9, second_order_nonlinear_tensor=None),
            "nitride": gfem.Material(),
        },
    )
    options = gfem.SolverOptions(order=1)
    assert (
        solve_key(explicit, problem, options)[0]
        == solve_key(plain, problem, options)[0]
    )
