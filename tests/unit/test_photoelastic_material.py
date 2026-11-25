"""A material's photoelastic tensor: checks, frame, `MaterialField`, codec."""

import acousto_optic_cases as ao
import electro_optic_cases as eo
import jax
import jax.numpy as jnp
import numpy as np
import piezo_cases as pc
import pytest

import gfem
from gfem._materials import stiffness_in_frame

P_LN = ao.P_LN
X_CUT = eo.X_CUT
_LAW = gfem.PermittivityLaw(
    eps_inf=40.0,
    terms=[gfem.Lorentz(delta_eps=4.0, resonance_frequency=6e12, damping=0.0)],
)
_LEFT = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
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


def _ln(**overrides):
    kwargs = {
        "eps_r": eo.LN_OPTICAL,
        "axes": X_CUT,
        "photoelastic_tensor": P_LN,
        "electro_optic_tensor": eo.R_LN,
        "name": "LN",
    }
    return gfem.Material(**(kwargs | overrides))


# --- Material ----------------------------------------------------------------------


def test_a_material_keeps_its_photoelastic_tensor_beside_the_electro_optic_one():
    m = _ln()
    assert m.photoelastic_tensor == tuple(tuple(float(x) for x in r) for r in P_LN)
    assert m.electro_optic_tensor == eo.R_LN
    assert m == _ln()
    assert hash(m) == hash(_ln())
    assert "photoelastic_tensor" in repr(m)
    assert gfem.Material().photoelastic_tensor is None
    # a constant material has no leaves
    assert jax.tree_util.tree_leaves(m) == []
    # an isotropic photoelastic material needs no axes
    iso = gfem.Material(eps_r=2.1**2, photoelastic_tensor=np.eye(6) * 0.2)
    assert iso.axes is None


@pytest.mark.parametrize(
    ("overrides", "words"),
    [
        (
            {"photoelastic_tensor": P_LN * (1 + 1e-3j)},
            ["photoelastic_tensor must be real", "no photoelastic loss is modelled"],
        ),
        (
            {"photoelastic_tensor": P_LN * np.nan},
            ["photoelastic_tensor must be finite"],
        ),
        (
            {"photoelastic_tensor": P_LN[:3, :3]},
            ["photoelastic_tensor is", "6x6", "shape (6, 6)", "(3, 3)"],
        ),
        (
            {"photoelastic_tensor": np.zeros((3, 3, 3, 3))},
            ["contract a full p_ijkl", "xx, yy, zz, yz, xz, xy", "no factors of 2"],
        ),
        ({"photoelastic_tensor": 100 * P_LN}, ["entry of", "above 10", "percent"]),
        (
            {"stiffness": pc.C_LN, "electro_optic_tensor": None},
            ["stiffness sits beside a photoelastic_tensor", "two Materials"],
        ),
        (
            {
                "stiffness": pc.C_LN,
                "piezoelectric_tensor": pc.E_LN,
                "electro_optic_tensor": None,
            },
            ["sits beside a photoelastic_tensor", "acoustic", "optical"],
        ),
        (
            {
                "eps_r": np.diag(eo.LN_OPTICAL),
                "axes": None,
                "electro_optic_tensor": None,
            },
            ["eps_r is a 3x3 matrix", "photoelastic_tensor", "explicit `axes`"],
        ),
        (
            {"axes": _LEFT, "electro_optic_tensor": None},
            ["left-handed", "photoelastic_tensor is given in the frame"],
        ),
        (
            {"eps_r": _LAW, "axes": None, "electro_optic_tensor": None},
            ["photoelastic_tensor needs a constant eps_r", "PermittivityLaw"],
        ),
        (
            {"eps_r": 1.0, "axes": None, "electro_optic_tensor": None},
            ["photoelastic_tensor needs the crystal's optical eps_r", "n^4"],
        ),
    ],
    ids=[
        "complex",
        "nan",
        "3x3",
        "3x3x3x3",
        "percent",
        "beside-stiffness",
        "beside-piezoelectric",
        "matrix",
        "left-handed",
        "law",
        "vacuum",
    ],
)
def test_every_photoelastic_guard_names_the_remedy(overrides, words):
    with pytest.raises(ValueError) as e:
        _ln(**overrides)
    for word in words:
        assert word in str(e.value)
    assert "Material 'LN'" in str(e.value)


def test_the_largest_tabulated_entry_passes_and_the_bound_is_an_entry_of_ten():
    gfem.Material(eps_r=4.0, photoelastic_tensor=np.eye(6) * 10.0)
    with pytest.raises(ValueError, match="above 10"):
        gfem.Material(eps_r=4.0, photoelastic_tensor=np.eye(6) * 10.0001)
    # a negative entry counts by magnitude
    with pytest.raises(ValueError, match="above 10"):
        gfem.Material(eps_r=4.0, photoelastic_tensor=-np.eye(6) * 11.0)


def test_a_photoelastic_tensor_under_a_gradient_raises_with_the_field_remedy():
    def objective(p11):
        p = jnp.asarray(P_LN).at[0, 0].set(p11)
        return gfem.Material(
            eps_r=eo.LN_OPTICAL, axes=X_CUT, photoelastic_tensor=p
        ).eps_r[0]

    with pytest.raises(gfem.NonDifferentiableError, match=r"MaterialField\.uniform"):
        jax.grad(objective)(P_LN[0, 0])


# --- frames ------------------------------------------------------------------------


@pytest.mark.parametrize("axes", [X_CUT, _random_axes(0), _random_axes(7)])
def test_the_global_frame_tensor_rotates_as_a_stiffness_and_round_trips(axes):
    # the global frame from the frame of axes: Q = a^T, global_i = Q_ik crystal_k
    m = _bond(np.asarray(axes).T)
    lab = stiffness_in_frame(P_LN, axes)
    assert _relative(lab, m @ P_LN @ m.T) < 1e-13
    # back to the crystal frame to round-off of the contractions: 1e-14 relative
    back = stiffness_in_frame(lab, None, axes)
    assert _relative(back, P_LN) < 1e-14
    # a non-symmetric p (p14 != p41 in 3m) survives the round trip
    assert abs(back[0, 3] - back[3, 0]) > 0.05
    assert _relative(back[0, 3], P_LN[0, 3]) < 1e-12


def test_a_reflection_flips_p14_so_crystal_axes_are_right_handed():
    mirrored = stiffness_in_frame(P_LN, np.diag([1.0, 1.0, -1.0]))
    assert mirrored[0, 3] == -P_LN[0, 3]
    assert mirrored[0, 0] == P_LN[0, 0]


# --- MaterialField -----------------------------------------------------------------


def test_uniform_field_carries_the_photoelastic_tensor_in_the_material_s_frame():
    f = gfem.MaterialField.uniform("film", _ln())
    assert f.axes == X_CUT
    np.testing.assert_array_equal(f.photoelastic_tensor, P_LN)
    np.testing.assert_array_equal(f.electro_optic_tensor, eo.R_LN)
    assert "photoelastic_tensor" in f.properties
    # an isotropic photoelastic material with axes keeps its frame
    turned = gfem.MaterialField.uniform(
        "film",
        gfem.Material(eps_r=4.0, photoelastic_tensor=np.eye(6) * 0.1, axes=X_CUT),
    )
    assert turned.axes == X_CUT


def test_jax_grad_reaches_a_field_s_photoelastic_tensor_per_element_and_uniform():
    n = 3
    field = gfem.MaterialField(
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        photoelastic_tensor=jnp.tile(jnp.asarray(P_LN), (n, 1, 1)),
    )
    assert field.n_elements == n
    assert field.photoelastic_tensor.dtype == jnp.float64
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(field)
    ]
    assert ".photoelastic_tensor" in paths
    w = jnp.arange(36.0).reshape(6, 6)
    # the field hands its leaves to the objective unchanged: the gradient is the
    # objective's own weight, bit for bit
    grad = jax.grad(lambda f: jnp.sum(w * f.photoelastic_tensor))(field)
    np.testing.assert_array_equal(grad.photoelastic_tensor, jnp.tile(w, (n, 1, 1)))
    one = gfem.MaterialField.uniform("film", _ln())
    grad = jax.grad(lambda f: jnp.sum(w * f.photoelastic_tensor))(one)
    np.testing.assert_array_equal(grad.photoelastic_tensor, w)


def test_a_traced_value_is_checked_at_solve_time_and_a_field_checks_its_shape():
    from gfem._materials import check_field_values

    traced = jax.jit(
        lambda s: gfem.MaterialField(eps_r=4.0, photoelastic_tensor=s * P_LN)
    )(100.0)
    with pytest.raises(ValueError, match=r"above 10.*region 'film'"):
        check_field_values({"film": traced})
    with pytest.raises(ValueError, match="same number of values"):
        gfem.MaterialField(
            eps_r=jnp.ones(2) * 4.0,
            photoelastic_tensor=jnp.tile(jnp.asarray(P_LN), (3, 1, 1)),
        )


@pytest.mark.parametrize(
    ("kwargs", "words"),
    [
        (
            {"photoelastic_tensor": jnp.zeros((3, 3, 3, 3))},
            ["(6, 6)", "(n_elements, 6, 6)", "contract a full p_ijkl"],
        ),
        (
            {"photoelastic_tensor": jnp.asarray(P_LN) * 1j},
            ["photoelastic_tensor must be real"],
        ),
        ({"photoelastic_tensor": jnp.asarray(P_LN) * 100}, ["above 10"]),
        (
            {
                "photoelastic_tensor": jnp.asarray(P_LN),
                "stiffness": jnp.asarray(pc.C_LN),
            },
            ["stiffness sits beside a photoelastic_tensor"],
        ),
        (
            {"photoelastic_tensor": jnp.asarray(P_LN), "axes": _LEFT},
            ["left-handed", "photoelastic_tensor"],
        ),
        (
            {"photoelastic_tensor": jnp.asarray(P_LN)},
            ["photoelastic_tensor needs the crystal's optical eps_r", "n^4"],
        ),
    ],
    ids=["shape", "complex", "percent", "beside-stiffness", "left-handed", "no-eps"],
)
def test_material_field_photoelastic_refusals_name_the_remedy(kwargs, words):
    with pytest.raises(ValueError) as e:
        gfem.MaterialField(**kwargs)
    for word in words:
        assert word in str(e.value)


def test_interpolate_blends_the_photoelastic_tensor_linearly_in_the_field_frame():
    rho = jnp.array([0.0, 0.25, 1.0])
    oxide = gfem.Material(
        eps_r=1.444**2, photoelastic_tensor=np.diag([0.12] * 3 + [0.0] * 3)
    )
    ln = _ln(electro_optic_tensor=None)
    f = gfem.MaterialField.interpolate(rho, oxide, ln)
    assert np.linalg.det(np.asarray(f.axes)) > 0
    p_oxide = np.asarray(stiffness_in_frame(np.diag([0.12] * 3 + [0.0] * 3), None))
    p_ln = stiffness_in_frame(P_LN, X_CUT)
    for k, w in enumerate((0.0, 0.25, 1.0)):
        p = stiffness_in_frame(np.asarray(f.photoelastic_tensor[k]), f.axes)
        assert _relative(p, p_oxide + w * (p_ln - p_oxide)) < 1e-14
    # an end without one counts as zero
    g = gfem.MaterialField.interpolate(rho, gfem.Material(eps_r=1.0), ln)
    p = stiffness_in_frame(np.asarray(g.photoelastic_tensor[2]), g.axes)
    assert _relative(p, p_ln) < 1e-14
    # the index scheme blends in the global frame
    h = gfem.MaterialField.interpolate(
        rho,
        gfem.Material(eps_r=2.0, photoelastic_tensor=np.eye(6) * 0.1),
        gfem.Material(eps_r=4.0, photoelastic_tensor=np.eye(6) * 0.3),
        scheme="index",
    )
    assert h.axes is None
    np.testing.assert_allclose(h.photoelastic_tensor[1], np.eye(6) * 0.15, rtol=1e-14)
    # no tensor at either end: none in the field
    assert (
        gfem.MaterialField.interpolate(
            rho, gfem.Material(), gfem.Material(eps_r=2.0)
        ).photoelastic_tensor
        is None
    )


def test_interpolate_refuses_a_law_end_with_a_photoelastic_tensor():
    with pytest.raises(ValueError, match="photoelastic_tensor needs a constant eps_r"):
        gfem.MaterialField.interpolate(
            jnp.array([0.0, 1.0]),
            gfem.Material(eps_r=_LAW),
            _ln(electro_optic_tensor=None),
        )


# --- refinement and the job codec --------------------------------------------------


def test_refinement_copies_a_per_element_photoelastic_tensor_to_the_children(tmp_path):
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
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        photoelastic_tensor=scale[:, None, None] * jnp.asarray(P_LN),
    )
    model = gfem.Model(
        mesh, materials={"oxide": field, "nitride": gfem.Material(eps_r=7.5)}
    )
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[mesh.elements("oxide")[0]] = True
    fine, _ = refine_model(model, marked)
    refined = fine.materials["oxide"]
    assert refined.photoelastic_tensor.shape == (len(fine.mesh.elements("oxide")), 6, 6)
    ratio = np.asarray(refined.photoelastic_tensor[:, 0, 1]) / P_LN[0, 1]
    # each child keeps its parent's scale, to the round-off of one division
    # (1e-15 relative); every scale is told apart by far more than that
    wanted = np.asarray(scale)
    nearest = wanted[np.argmin(np.abs(ratio[:, None] - wanted[None, :]), axis=1)]
    np.testing.assert_allclose(ratio, nearest, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.unique(nearest), np.unique(wanted), rtol=0, atol=0)


def test_photoelastic_materials_round_trip_through_the_job_codec(tmp_path):
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
        eps_r=jnp.full((n, 3), 4.9),
        axes=X_CUT,
        photoelastic_tensor=jnp.linspace(1, 2, n)[:, None, None] * jnp.asarray(P_LN),
    )
    model = gfem.Model(mesh, materials={"oxide": field, "nitride": _ln()})
    problem = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model
    assert back.materials["nitride"] == _ln()
    oxide = back.materials["oxide"]
    assert oxide.photoelastic_tensor.dtype == jnp.float64
    np.testing.assert_array_equal(oxide.photoelastic_tensor, field.photoelastic_tensor)
    # a material without one is written as before
    plain = gfem.Model(
        mesh, materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material()}
    )
    doc = json.loads(gfem.Job(plain, problem).to_json(tmp_path / "p.json").read_text())
    assert "photoelastic_tensor" not in doc["job"]["model"]["materials"]["oxide"]
