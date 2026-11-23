"""gfem.solve on an Electrostatic problem: inputs, ground rules, result."""

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import structured_meshes as sm
from scipy.constants import epsilon_0

import gfem
from gfem.job import _write_result

DATA = Path(__file__).parent / "data"
LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
STACK = {"top": "top", "mid": "oxide__nitride"}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


def _c(model, problem, **options):
    return gfem.solve(model, problem, options=gfem.SolverOptions(**options))


# --- the problem value -----------------------------------------------------------


def test_terminals_keep_their_order_and_surfaces():
    p = gfem.Electrostatic(terminals={"b": ["s1", "s2"], "a": "s3"}, ground="g")
    assert p.terminal_names == ("b", "a")
    assert p.terminals == (("b", ("s1", "s2")), ("a", ("s3",)))
    assert p.ground == ("g",)


def test_a_list_of_surfaces_makes_one_terminal_per_surface():
    p = gfem.Electrostatic(terminals=["trace", "jj"])
    assert p.terminals == (("trace", ("trace",)), ("jj", ("jj",)))
    assert p.ground == ()


def test_problem_is_frozen_hashable_and_static_under_jax():
    p = gfem.Electrostatic(terminals={"a": "s"}, ground=["g"])
    assert p == gfem.Electrostatic(terminals={"a": ("s",)}, ground="g")
    assert hash(p) == hash(gfem.Electrostatic(terminals={"a": "s"}, ground="g"))
    assert jax.tree.leaves(p) == []
    with pytest.raises(AttributeError):
        p.ground = ()  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"terminals": {}}, "at least one terminal"),
        ({"terminals": {"a": "s", "b": ["s"]}}, "terminals 'a' and 'b'"),
        ({"terminals": {"a": "s"}, "ground": "s"}, "both in terminal 'a'"),
        ({"terminals": {"a": ["s", "s"]}}, "twice"),
    ],
)
def test_inconsistent_terminals_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gfem.Electrostatic(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"terminals": {"a": 3}}, {"terminals": {1: "s"}}, {"terminals": "s", "ground": 2}],
)
def test_non_names_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        gfem.Electrostatic(**kwargs)


# --- solving -----------------------------------------------------------------------


def test_result_is_labelled_symmetric_with_palace_signs(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    c = np.asarray(cap.capacitance)
    assert cap.terminals == ("top", "mid")
    assert cap.index("mid") == 1
    np.testing.assert_array_equal(c, c.T)
    assert (np.diag(c) > 0).all()
    assert c[0, 1] < 0
    assert "fF" in repr(cap)
    assert "mid" in repr(cap)


def test_result_is_a_pytree_of_the_matrix_and_the_region_energies(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    leaves, tree = jax.tree.flatten(cap)
    assert leaves[0] is cap.capacitance
    assert any(leaf is cap.potential["top"].coefficients for leaf in leaves)
    assert isinstance(cap.capacitance, jax.Array)
    assert cap.capacitance.dtype == np.float64
    assert all(isinstance(e, jax.Array) for e in cap.energy.values())
    back = jax.tree.unflatten(tree, [leaf * 2 for leaf in leaves])
    assert back.terminals == cap.terminals
    np.testing.assert_allclose(back.mutual, 2 * cap.mutual)


def test_mutual_and_inverse_follow_the_maxwell_matrix(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    c = np.asarray(cap.capacitance)
    m = np.asarray(cap.mutual)
    # Palace terminal-Cm.csv: -C off the diagonal, the row sums on it
    np.testing.assert_allclose(m - np.diag(np.diag(m)), -(c - np.diag(np.diag(c))))
    np.testing.assert_allclose(np.diag(m), c.sum(axis=1))
    np.testing.assert_allclose(np.asarray(cap.inverse) @ c, np.eye(2), atol=1e-12)


def test_mutual_and_inverse_are_differentiable_functions_of_the_matrix(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))

    def to_ground(c):
        leaves = jax.tree.leaves(cap)
        res = jax.tree.unflatten(jax.tree.structure(cap), [c, *leaves[1:]])
        return res.mutual[1, 1] + res.inverse[0, 0]

    g = jax.grad(to_ground)(cap.capacitance)
    assert g.shape == (2, 2)
    assert np.isfinite(np.asarray(g)).all()


def test_region_energies_add_up_to_half_the_self_capacitance(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    assert list(cap.energy) == ["oxide", "nitride"]
    total = sum(np.asarray(e) for e in cap.energy.values())
    np.testing.assert_allclose(total, np.diag(cap.capacitance) / 2, rtol=1e-12)
    assert all(np.asarray(e).shape == (2,) for e in cap.energy.values())


def test_unknown_terminal_index_names_the_terminals(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    with pytest.raises(KeyError, match="'top', 'mid'"):
        cap.index("bottom")


def test_without_ground_each_row_sums_to_zero(stack):
    cap = _c(stack, gfem.Electrostatic(terminals={**STACK, "low": "bottom"}))
    c = np.asarray(cap.capacitance)
    np.testing.assert_allclose(c.sum(axis=1), 0, atol=1e-12 * np.abs(c).max())


@pytest.mark.parametrize("condition", [gfem.PEC, gfem.Conductor])
def test_model_conductors_that_are_not_terminals_are_grounded(stack, condition):
    boundary = (
        gfem.PEC("bottom")
        if condition is gfem.PEC
        else gfem.Conductor("bottom", conductivity=5.8e7, thickness=1e-6)
    )
    grounded = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    with_pec = _c(
        stack.replace(boundaries=[boundary]), gfem.Electrostatic(terminals=STACK)
    )
    np.testing.assert_allclose(with_pec.capacitance, grounded.capacitance, rtol=1e-12)


def test_a_terminal_overrides_the_model_pec_on_its_surface(stack):
    model = stack.replace(boundaries=[gfem.PEC("top"), gfem.PEC("bottom")])
    plain = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    cap = _c(model, gfem.Electrostatic(terminals=STACK))
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-12)


def test_other_boundaries_carry_no_charge(stack):
    model = stack.replace(boundaries=[gfem.Absorbing("side"), gfem.PMC("top")])
    plain = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    cap = _c(model, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-12)


@pytest.mark.parametrize(
    ("conductivity", "axes"),
    [
        (1.0, None),
        (
            (1.0, 0.0, 30.0),
            ((1 / 3, 2 / 3, 2 / 3), (2 / 3, 1 / 3, -2 / 3), (2 / 3, -2 / 3, 1 / 3)),
        ),
    ],
    ids=["isotropic", "tilted-tensor"],
)
def test_conductivity_and_loss_do_not_change_the_capacitance(stack, conductivity, axes):
    oxide = gfem.Material(
        eps_r=3.9, tan_delta=0.1, conductivity=conductivity, mu_r=4.0, axes=axes
    )
    lossy = {"oxide": oxide, "nitride": gfem.Material(eps_r=7.5, conductivity=1e3)}
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    plain = _c(stack, problem)
    cap = _c(stack.replace(materials=lossy), problem)
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-13)


def test_a_uniform_field_of_a_crystal_has_the_crystal_s_capacitance(stack):
    crystal = gfem.Material(
        eps_r=(3.0, 5.0, 6.0),
        tan_delta=(0.01, 0.02, 0.03),
        conductivity=(0.02, 0.05, 0.03),
        axes=((1 / 3, 2 / 3, 2 / 3), (2 / 3, 1 / 3, -2 / 3), (2 / 3, -2 / 3, 1 / 3)),
    )
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    expected = _c(stack.with_material("oxide", crystal), problem).capacitance
    field = gfem.MaterialField.uniform("oxide", crystal)
    cap = _c(stack.with_material("oxide", field), problem).capacitance
    np.testing.assert_array_equal(cap, expected)  # measured on MN5: bit for bit


def test_solving_twice_gives_the_same_matrix(stack):
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    first = _c(stack, problem, order=2)
    _c(stack, problem, order=1)
    second = _c(stack, problem, order=2)
    np.testing.assert_array_equal(first.capacitance, second.capacitance)


def test_moved_nodes_change_the_capacitance(stack):
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    nodes = stack.mesh.nodes
    z = nodes[:, 2]
    stretched = stack.replace(
        mesh=jax.tree.unflatten(
            jax.tree.structure(stack.mesh), [nodes.at[:, 2].set(2 * z)]
        )
    )
    c = float(_c(stack, problem).capacitance[0, 0])
    c2 = float(_c(stretched, problem).capacitance[0, 0])
    assert c2 == pytest.approx(c / 2, rel=1e-11)


# --- errors ------------------------------------------------------------------------


def test_an_inverted_moved_mesh_raises_before_solving(stack):
    mirrored = jax.tree.unflatten(
        jax.tree.structure(stack.mesh), [stack.mesh.nodes.at[:, 2].multiply(-1.0)]
    )
    with pytest.raises(gfem.MeshQualityError):
        _c(stack.replace(mesh=mirrored), gfem.Electrostatic(terminals=STACK))


def test_unknown_surface_raises_with_a_hint(stack):
    with pytest.raises(gfem.RegionNotFoundError, match="'top'"):
        _c(stack, gfem.Electrostatic(terminals={"t": "tpo"}, ground="bottom"))
    with pytest.raises(gfem.RegionNotFoundError):
        _c(stack, gfem.Electrostatic(terminals={"t": "top"}, ground="bottm"))


def test_a_volume_terminal_asks_for_its_surfaces(stack):
    with pytest.raises(ValueError, match="volume"):
        _c(stack, gfem.Electrostatic(terminals={"t": "oxide"}))


def test_touching_terminals_raise(stack):
    # "side" shares the plate edges with "top" and "bottom"
    with pytest.raises(ValueError, match="touch"):
        _c(stack, gfem.Electrostatic(terminals={"t": "top", "s": "side"}))
    with pytest.raises(ValueError, match="touch"):
        _c(stack, gfem.Electrostatic(terminals={"t": "top"}, ground="side"))


def test_a_terminal_inside_an_excluded_volume_raises():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        excluded=("sub",),
    )
    with pytest.raises(ValueError, match="does not border"):
        _c(model, gfem.Electrostatic(terminals={"m": "metal_xy"}))


def test_terminal_on_an_excluded_conductor_shell():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    materials = {"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()}
    problem = gfem.Electrostatic(
        terminals={"metal": ["metal_xy", "metal_z"]}, ground="sub__None"
    )
    meshed = _c(gfem.Model(mesh, materials=materials), problem)
    excluded = gfem.Model(
        mesh,
        materials=materials,
        excluded=(gfem.enclosed_by("metal_xy", "metal_z"),),
    )
    # the conductor's inside is at one potential: excluding it changes nothing
    cap = _c(excluded, problem)
    np.testing.assert_allclose(cap.capacitance, meshed.capacitance, rtol=1e-10)
    assert float(cap.capacitance[0, 0]) > 0
    # the field-free inside of the conductor stores no energy
    assert set(cap.energy) == set(meshed.energy)
    np.testing.assert_allclose(
        sum(cap.energy.values()), sum(meshed.energy.values()), rtol=1e-10
    )


@pytest.mark.parametrize(
    "solver", ["auto", "bddc_gmres", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_each_linear_solver_agrees_with_the_direct_solver(tmp_path, solver):
    # order 3: face DOFs are off the wirebasket, so BDDC + CG really iterates
    path = tmp_path / "fine.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (3, 3, 2)))
    model = gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    direct = _c(model, problem, order=3, linear_solver="direct")
    other = _c(model, problem, order=3, linear_solver=solver)
    np.testing.assert_allclose(other.capacitance, direct.capacitance, rtol=1e-9)


def test_solve_rejects_wrong_argument_types(stack):
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(stack, "electrostatic")
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.solve(stack.mesh, gfem.Electrostatic(terminals=STACK))
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.solve(stack, gfem.Electrostatic(terminals=STACK), options={"order": 2})


# --- a slab cross-section: capacitance per length ---------------------------------

#: The plates' section in um: width, and the layers bottom up with eps_r.
WIDTH = 4.0
SECTION_LAYERS = (("oxide", 0.5, 3.9), ("nitride", 0.25, 7.5))


def _write_plates(path):
    """Two layers between plates "bottom" (y = 0) and "top", msh 2.2 triangles."""
    xs = np.linspace(0.0, WIDTH, 5)
    ys = [0.0, 0.25, 0.5, 0.625, 0.75]
    nx = len(xs)
    nodes = [(x, y) for y in ys for x in xs]

    def n(i, j):
        return j * nx + i + 1

    tags = {"oxide": 1, "nitride": 2, "bottom": 3, "top": 4}
    tris = []
    for j in range(len(ys) - 1):
        name = "oxide" if ys[j] < 0.5 else "nitride"
        for i in range(nx - 1):
            a, b, c, d = n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)
            pair = ((a, b, c), (a, c, d)) if (i + j) % 2 else ((a, b, d), (b, c, d))
            tris += [(name, t) for t in pair]
    top = len(ys) - 1
    lines = [("bottom", (n(i, 0), n(i + 1, 0))) for i in range(nx - 1)]
    lines += [("top", (n(i, top), n(i + 1, top))) for i in range(nx - 1)]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4"]
    out += ['2 1 "oxide"', '2 2 "nitride"', '1 3 "bottom"', '1 4 "top"']
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{k + 1} {float(x)!r} {float(y)!r} 0" for k, (x, y) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(tris))]
    k = 0
    for name, (a, b) in lines:
        k += 1
        out.append(f"{k} 1 2 {tags[name]} {tags[name]} {a} {b}")
    for name, (a, b, c) in tris:
        k += 1
        out.append(f"{k} 2 2 {tags[name]} {tags[name]} {a} {b} {c}")
    out.append("$EndElements")
    path.write_text("\n".join(out) + "\n")
    return path


@pytest.fixture(scope="module")
def plates_path(tmp_path_factory):
    return _write_plates(tmp_path_factory.mktemp("plates") / "plates.msh")


def _plates(path, thickness):
    mesh = gfem.Mesh.from_gmsh_cross_section(path, thickness=thickness)
    materials = {name: gfem.Material(eps_r=e) for name, _, e in SECTION_LAYERS}
    model = gfem.Model(mesh, materials=materials)
    problem = gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom")
    return gfem.solve(model, problem, options=gfem.SolverOptions(order=1))


@pytest.mark.parametrize("thickness", [0.3e-6, 2e-6])
def test_parallel_plates_on_a_section_have_the_series_capacitance_per_length(
    plates_path, thickness
):
    result = _plates(plates_path, thickness)
    # layers in series between plates of width w: C / L = eps0 w / sum(d / eps_r)
    um = 1e-6
    exact = epsilon_0 * WIDTH * um / sum(d * um / e for _, d, e in SECTION_LAYERS)
    per_length = np.asarray(result.capacitance_per_length)
    assert per_length.shape == (1, 1)
    assert float(per_length[0, 0]) == pytest.approx(exact, rel=1e-11)
    # the slab totals still scale with the slab
    slab = float(result.capacitance[0, 0])
    assert slab == pytest.approx(exact * thickness, rel=1e-11)


def test_the_repr_of_a_section_shows_the_capacitance_per_length(plates_path):
    text = repr(_plates(plates_path, 1e-6))
    assert "slab capacitance in fF" in text
    assert "capacitance per length in pF/m" in text


def test_to_xarray_of_a_section_adds_the_capacitance_per_length(plates_path):
    pytest.importorskip("xarray")
    result = _plates(plates_path, 1e-6)
    ds = result.to_xarray()
    assert ds["capacitance_per_length"].attrs["units"] == "F/m"
    np.testing.assert_array_equal(
        ds["capacitance_per_length"].values, result.capacitance_per_length
    )


def test_capacitance_per_length_on_a_3d_mesh_names_the_remedy(stack):
    result = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    with pytest.raises(
        ValueError,
        match=r"capacitance_per_length is defined on a cross-section mesh only "
        r"\(gfem\.Mesh\.from_gmsh_cross_section\).*read "
        r"ElectrostaticResult\.capacitance instead",
    ):
        _ = result.capacitance_per_length
    assert "per length" not in repr(result)
    if importlib.util.find_spec("xarray"):
        assert "capacitance_per_length" not in result.to_xarray()


def test_a_saved_3d_result_has_no_thickness_and_a_section_keeps_it(
    stack, plates_path, tmp_path
):
    # a 3D result bundle stays as gfem 0.1.0 wrote it and reads there
    flat = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    _write_result(tmp_path / "flat", flat, {}, None)
    written = json.loads((tmp_path / "flat" / "result.json").read_text())["result"]
    assert "_thickness" not in json.dumps(written)
    assert gfem.load_result(tmp_path / "flat")._thickness is None
    section = _plates(plates_path, 1e-6)
    _write_result(tmp_path / "section", section, {}, None)
    back = gfem.load_result(tmp_path / "section")
    np.testing.assert_array_equal(
        back.capacitance_per_length, section.capacitance_per_length
    )
