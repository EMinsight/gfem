"""`gfem.dissipated_power`: the power a driven field dissipates, per region and sheet.

Two lines carry the checks. A 10 x 2 x 1 mm parallel-plate line with lumped
ports at both ends holds the volume losses in its 2 mm "core" (x in [4, 6]),
one element away from both ports, so its value is minus `gfem.power_flow`
out of the core, a residual of the same discrete equations (round-off). The
sheet line of `test_sheet_field.py` loads the line with an interior sheet
tangential to the TEM field. With lumped ports of real reference impedance and
nothing radiating, every watt that does not leave by a port is dissipated, so
the values sum to ``1 - sum_j |S_j1|^2``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from anisotropy import turned
from finite_differences import assert_gradient_matches, unit_directions

import gfem
from gfem import _mpi

GHZ = 1e9
F = 7 * GHZ
#: Order 3: at order 2 a lumped port's voltage read (S) differs from the power its
#: resistance piece absorbs from the field by ~1e-6 relative on these one-cell
#: port faces, a discretization error of S, not of the loss (4e-12 at order 3).
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")

#: One law with lossy terms (Re eps > 0 at 7 GHz).
LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=20 * GHZ, damping=1 * GHZ),
        gfem.Debye(delta_eps=2.0, relaxation_frequency=5 * GHZ),
    ],
)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, core, *, dielectric=None, boundaries=()):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": dielectric or gfem.Material(eps_r=4.0),
            "core": core,
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *boundaries],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _solve(model, **kwargs):
    problem = gfem.Driven(frequency=F, excite="p1", keep_fields=True, **kwargs)
    return gfem.solve(model, problem, options=OPTIONS)


def _absorbed(result):
    s = np.asarray(result.s.values)[0]
    return 1 - abs(s[0, 0]) ** 2 - abs(s[1, 0]) ** 2


def _core_elements(mesh):
    return len(mesh.elements("core"))


def _cores(mesh):
    n = _core_elements(mesh)
    rng = np.random.default_rng(1)
    eps = 3.0 + rng.uniform(0, 1, n)
    axes = turned(1, 30)
    index_rho = rng.uniform(0.2, 0.8, n)
    return {
        "tan_delta": gfem.Material(eps_r=3.0, tan_delta=0.05),
        "conductivity": gfem.Material(eps_r=3.0, conductivity=0.3),
        "per-axis tensor": gfem.MaterialField(
            eps_r=np.repeat(eps[:, None], 3, axis=1),
            tan_delta=np.tile([0.02, 0.06, 0.01], (n, 1)),
            conductivity=np.tile([0.1, 0.0, 0.4], (n, 1)),
            axes=axes,
        ),
        "complex eps_r": gfem.MaterialField(eps_r=eps * (1 - 0.04j)),
        "law": gfem.Material(eps_r=LAW),
        "index blend": gfem.MaterialField.interpolate(
            index_rho,
            gfem.Material(eps_r=2.0),
            gfem.Material(eps_r=6.0, tan_delta=0.1),
            scheme="index",
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "tan_delta",
        "conductivity",
        "per-axis tensor",
        "complex eps_r",
        "law",
        "index blend",
    ],
)
def test_a_region_dissipates_what_power_flow_says_enters_it(mesh, case):
    model = _line(mesh, _cores(mesh)[case])
    result = _solve(model)
    e = result.e_field(F)
    lost = gfem.dissipated_power(e, model=model)
    entering = -float(gfem.power_flow(e, "core", model=model))
    assert entering > 1e-3
    assert float(lost["core"]) == pytest.approx(entering, rel=1e-12, abs=1e-15)
    assert float(lost["dielectric"]) == 0.0
    assert set(lost) == {"dielectric", "core"}
    total = sum(float(v) for v in lost.values())
    assert total == pytest.approx(_absorbed(result), abs=1e-9)


# --- sheets -------------------------------------------------------------------


L, W, H = 12.0, 2.0, 1.0  # mm
SHEET_CELLS = (12, 2, 1)
SHEET_F = 6e9
#: The sheet spans these x (mm): its ends stay away from the ports, whose section
#: is homogeneous, so the lumped ports read the TEM field their S assumes.
SPAN = (4.0, 8.0)
#: Both halves: a homogeneous lossy line.
HALF = gfem.Material(eps_r=2.0, tan_delta=0.01)


@pytest.fixture(scope="module")
def sheet_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(L, W, H, SHEET_CELLS, span=SPAN))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _sheet_model(mesh, *sheet):
    return gfem.Model(
        mesh,
        materials={"lower": HALF, "upper": HALF},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), *sheet],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=50.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=50.0),
        ],
    )


def _sheet_solve(model):
    problem = gfem.Driven(frequency=SHEET_F, excite="p1", keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


def _fill(mesh):
    n = len(mesh.elements("sheet"))
    return gfem.SheetField(fill=np.random.default_rng(2).uniform(0.3, 1.0, n))


@pytest.mark.parametrize(
    "sheet",
    [
        lambda _: gfem.Impedance("sheet", resistance=80.0, inductance=1e-10),
        lambda m: gfem.Impedance("sheet", resistance=80.0, fill=_fill(m)),
        lambda _: gfem.Conductor("sheet", conductivity=2e5, thickness=1e-6),
        lambda m: gfem.Conductor("sheet", conductivity=2e5, fill=_fill(m)),
    ],
    ids=["impedance", "impedance fill", "conductor", "conductor fill"],
)
def test_volume_and_sheet_loss_sum_to_the_power_the_ports_do_not_take(
    sheet_mesh, sheet
):
    model = _sheet_model(sheet_mesh, sheet(sheet_mesh))
    result = _sheet_solve(model)
    lost = gfem.dissipated_power(result.e_field(SHEET_F), model=model)
    assert set(lost) == {"lower", "upper", "sheet"}
    assert float(lost["sheet"]) > 1e-4
    assert float(lost["lower"]) > 1e-4
    assert float(lost["upper"]) > 1e-4
    total = sum(float(v) for v in lost.values())
    assert total == pytest.approx(_absorbed(result), abs=1e-9)


# --- what is not heat ----------------------------------------------------------


def test_ports_and_absorbing_faces_dissipate_nothing(tmp_path):
    # a lossless line with lumped loads and absorbing side walls: power
    # leaves, none is dissipated
    path = tmp_path / "open.msh"
    layers = (("sub", 0.5), ("cover", 0.5))
    sm.write_msh22(
        path, sm.strip_line(4.0, 2.0, 1.0, (8, 2, 1), layers=layers, core=(1.0, 3.0))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "sub": gfem.Material(eps_r=4.0),
            "cover": gfem.Material(eps_r=2.0),
            "core": gfem.Material(eps_r=3.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    problem = gfem.Driven(frequency=25e9, excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    lost = gfem.dissipated_power(result.e_field(25e9), model=model)
    assert _absorbed(result) > 1e-3  # the absorbing walls take power
    assert {k: float(v) for k, v in lost.items()} == {
        "sub": 0.0,
        "cover": 0.0,
        "core": 0.0,
    }


def test_a_perfectly_matched_layer_is_not_heat(tmp_path):
    # the layer's background is lossy, but the field there is the stretched
    # one: the layer gets no key, and the lossless guide dissipates nothing
    import pml_cases

    path = pml_cases.strip_guide(tmp_path / "strip.msh", 0.3, length=0.6)
    model = pml_cases.strip_model(path)
    lossy = gfem.Material(eps_r=pml_cases.STRIP_CLADDING**2, tan_delta=0.1)
    model = model.replace(materials={**model.materials, "pml": lossy})
    problem = gfem.Driven(frequency=pml_cases.STRIP_F, excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    lost = gfem.dissipated_power(result.e_field(pml_cases.STRIP_F), model=model)
    assert {k: float(v) for k, v in lost.items()} == {"core": 0.0, "clad": 0.0}


# --- refusals ------------------------------------------------------------------


@pytest.fixture(scope="module")
def lossy(mesh):
    model = _line(mesh, gfem.Material(eps_r=3.0, tan_delta=0.05))
    return model, _solve(model).e_field(F)


def test_a_complex_mu_r_is_refused_by_region(mesh):
    n = _core_elements(mesh)
    model = _line(
        mesh, gfem.MaterialField(eps_r=np.full(n, 3.0), mu_r=np.full(n, 1.5 - 0.1j))
    )
    e = _solve(model).e_field(F)
    with pytest.raises(NotImplementedError, match=r"'core' has a complex mu_r"):
        gfem.dissipated_power(e, model=model)


def test_a_law_blend_is_refused(mesh):
    n = _core_elements(mesh)
    blend = gfem.MaterialField.interpolate(
        np.full(n, 0.5), gfem.Material(eps_r=2.0), gfem.Material(eps_r=LAW)
    )
    model = _line(mesh, blend)
    e = _solve(model).e_field(F)
    with pytest.raises(NotImplementedError, match="dissipated_power"):
        gfem.dissipated_power(e, model=model)


def test_a_field_across_ranks_is_refused(lossy):
    model, e = lossy
    piece = dataclasses.replace(e, _comm=_mpi.StaticComm(5, object()))
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.dissipated_power(piece, model=model)
    assert caught.value.feature == "dissipated_power"
    assert "load_result" in str(caught.value)


def test_a_name_that_is_a_region_and_a_sheet_is_refused(sheet_mesh, monkeypatch):
    # gmsh meshes gfem reads give each physical group its own name, so the
    # clash is staged: the sheet surface listed among the volume names
    model = _sheet_model(sheet_mesh, gfem.Impedance("sheet", resistance=80.0))
    e = _sheet_solve(model).e_field(SHEET_F)
    volumes = (*sheet_mesh.volumes, "sheet")
    monkeypatch.setattr(type(sheet_mesh), "volumes", property(lambda _: volumes))
    with pytest.raises(ValueError, match="'sheet' names both a volume region"):
        gfem.dissipated_power(e, model=model)


def test_an_eigenmode_field_has_no_drive(lossy):
    model, e = lossy
    mode = dataclasses.replace(e, omega=e.omega * (1 + 1e-3j))
    with pytest.raises(ValueError, match="an eigenmode has no drive"):
        gfem.dissipated_power(mode, model=model)


def test_a_2d_model_is_refused_with_the_slab_remedy():
    mesh = type("Mesh2D", (), {"dim": 2})()
    field = type("F", (), {"mesh": mesh})()
    with pytest.raises(NotImplementedError, match="from_gmsh_cross_section"):
        gfem.dissipated_power(field, model=None)


def test_node_coordinates_and_frequency_are_refused_under_grad(lossy):
    model, e = lossy

    def by_nodes(scale):
        moved = gfem.Mesh(model.mesh.nodes * scale, model.mesh._topology)
        return gfem.dissipated_power(
            dataclasses.replace(e, mesh=moved), model=model.replace(mesh=moved)
        )["core"]

    with pytest.raises(gfem.NonDifferentiableError, match="node coordinates"):
        jax.grad(by_nodes)(1.0)

    def by_omega(scale):
        field = dataclasses.replace(e, omega=e.omega * scale)
        return gfem.dissipated_power(field, model=model)["core"]

    with pytest.raises(gfem.NonDifferentiableError, match="fixed input"):
        jax.grad(by_omega)(1.0)


def test_an_adaptive_sweep_field_is_refused_under_grad(mesh):
    model = _line(mesh, gfem.Material(eps_r=3.0, tan_delta=0.05))
    problem = gfem.Driven(
        frequency=np.linspace(6e9, 8e9, 5).tolist(),
        excite="p1",
        keep_fields=True,
        sweep="adaptive",
    )
    result = gfem.solve(model, problem, options=OPTIONS)
    e = result.e_field(7e9)
    forward = gfem.dissipated_power(e, model=model)["core"]
    assert float(forward) > 0

    def scaled(a):
        field = dataclasses.replace(e, coefficients=a * e.coefficients)
        return gfem.dissipated_power(field, model=model)["core"]

    with pytest.raises(gfem.NonDifferentiableError, match='sweep="uniform"'):
        jax.grad(scaled)(1.0)


# --- gradients -----------------------------------------------------------------


def _check(objective, p, *, step, n=2, seed=0):
    rng = np.random.default_rng(seed)
    p = jnp.asarray(p)
    directions = unit_directions(rng, p.shape, n)
    if jnp.iscomplexobj(p):
        directions = [
            d * np.exp(1j * rng.uniform(0, 2 * np.pi, p.shape)) for d in directions
        ]
        directions = [d / np.linalg.norm(d) for d in directions]
    assert_gradient_matches(objective, p, directions, step=step)


def _core_loss(mesh, materials):
    def objective(p):
        model = _line(mesh, materials(p))
        e = _solve(model).e_field(F)
        lost = gfem.dissipated_power(e, model=model)
        return lost["core"] + 0.3 * lost["dielectric"]

    return objective


def test_gradient_by_a_complex_permittivity(mesh):
    n = _core_elements(mesh)
    eps = (3.0 + np.random.default_rng(3).uniform(0, 1, n)) * (1 - 0.04j)
    _check(_core_loss(mesh, lambda p: gfem.MaterialField(eps_r=p)), eps, step=1e-2)


def test_gradient_by_a_loss_tangent(mesh):
    n = _core_elements(mesh)
    tan = np.random.default_rng(4).uniform(0.02, 0.08, n)
    _check(
        _core_loss(
            mesh, lambda p: gfem.MaterialField(eps_r=np.full(n, 3.0), tan_delta=p)
        ),
        tan,
        step=1e-2,
    )


def test_gradient_by_a_per_axis_conductivity(mesh):
    n = _core_elements(mesh)
    sigma = np.tile([0.1, 0.05, 0.4], (n, 1))
    _check(
        _core_loss(
            mesh,
            lambda p: gfem.MaterialField(
                eps_r=np.full((n, 3), 3.0), conductivity=p, axes=turned(1, 30)
            ),
        ),
        sigma,
        step=5e-2,
    )


def test_gradient_by_a_law_parameter(mesh):
    def law(p):
        return gfem.Material(
            eps_r=gfem.PermittivityLaw(
                eps_inf=2.0,
                terms=[
                    gfem.Debye(delta_eps=p, relaxation_frequency=5 * GHZ),
                ],
            )
        )

    _check(_core_loss(mesh, law), 2.0, step=1e-2, n=1)


def test_gradient_by_a_non_law_index_blend_density(mesh):
    # an "index" blend of two constant ends is a complex eps_r per element,
    # a function of the density in JAX: the driven operator differentiates
    # it through the mass piece's holomorphic eps_r derivative, and so does
    # the heat
    n = _core_elements(mesh)
    rho = np.random.default_rng(5).uniform(0.2, 0.8, n)

    def blend(p):
        return gfem.MaterialField.interpolate(
            p,
            gfem.Material(eps_r=2.0),
            gfem.Material(eps_r=6.0, tan_delta=0.1),
            scheme="index",
        )

    _check(_core_loss(mesh, blend), rho, step=1e-2)


def _sheet_loss(mesh, sheet):
    def objective(p):
        model = _sheet_model(mesh, sheet(p))
        lost = gfem.dissipated_power(_sheet_solve(model).e_field(SHEET_F), model=model)
        return lost["sheet"] + 0.5 * lost["lower"]

    return objective


def test_gradient_by_an_impedance_resistance(sheet_mesh):
    _check(
        _sheet_loss(sheet_mesh, lambda r: gfem.Impedance("sheet", resistance=r)),
        80.0,
        step=1.0,
        n=1,
    )


def test_gradient_by_an_impedance_inductance_is_the_field_s_only(sheet_mesh):
    _check(
        _sheet_loss(
            sheet_mesh,
            lambda l: gfem.Impedance("sheet", resistance=80.0, inductance=l),
        ),
        2e-10,
        step=2e-12,
        n=1,
    )


def test_gradient_by_a_sheet_fill(sheet_mesh):
    fill = np.asarray(_fill(sheet_mesh).fill)
    _check(
        _sheet_loss(
            sheet_mesh,
            lambda p: gfem.Conductor(
                "sheet", conductivity=2e5, fill=gfem.SheetField(fill=p)
            ),
        ),
        fill,
        step=1e-2,
    )


def test_dissipated_power_compiles_and_maps(lossy):
    model, e = lossy
    p = float(gfem.dissipated_power(e, model=model)["core"])

    def scaled(a):
        field = dataclasses.replace(e, coefficients=a * e.coefficients)
        return gfem.dissipated_power(field, model=model)["core"]

    scales = jnp.asarray([1.0, 2.0, 0.5j])
    np.testing.assert_allclose(
        jax.vmap(scaled)(scales), p * np.abs(scales) ** 2, rtol=1e-12
    )
    assert float(jax.jit(scaled)(2.0)) == pytest.approx(4 * p, rel=1e-12)


def test_gradient_by_an_impedance_sheet_fill(sheet_mesh):
    fill = np.asarray(_fill(sheet_mesh).fill)
    _check(
        _sheet_loss(
            sheet_mesh,
            lambda p: gfem.Impedance(
                "sheet", resistance=80.0, fill=gfem.SheetField(fill=p)
            ),
        ),
        fill,
        step=1e-2,
    )


# --- the loss matrices of ports driven together ------------------------------------

A = np.array([0.6 + 0.2j, -0.3 + 0.7j])


def _coherent(matrices, a):
    """``sum_pq conj(a_p) a_q D[p, q]`` per element."""
    d = matrices["tetrahedra_re"] + 1j * matrices["tetrahedra_im"]
    return jnp.real(jnp.einsum("p,epq,q->e", jnp.conj(a), d, a))


def test_two_ports_driven_together_dissipate_the_superposed_field_s_power(mesh):
    from gfem._heat import loss_matrices

    model = _line(mesh, gfem.Material(eps_r=3.0, tan_delta=0.05))
    problem = gfem.Driven(frequency=F, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    e1, e2 = result.e_field(F, "p1"), result.e_field(F, "p2")
    matrices = loss_matrices([e1, e2], model)
    d = np.asarray(matrices["tetrahedra_re"]) + 1j * np.asarray(
        matrices["tetrahedra_im"]
    )
    assert np.array_equal(d, np.conj(np.swapaxes(d, 1, 2)))  # Hermitian
    assert np.abs(d[:, 0, 1]).max() > 1e-3 * np.abs(d[:, 0, 0]).max()
    both = dataclasses.replace(
        e1, coefficients=A[0] * e1.coefficients + A[1] * e2.coefficients
    )
    want = float(gfem.dissipated_power(both, model=model)["core"])
    core = model.mesh.elements("core")
    got = float(jnp.sum(_coherent(matrices, A)[core]))
    assert got == pytest.approx(want, rel=1e-12)


def test_fields_on_different_supports_raise(mesh, monkeypatch):
    # every field's support is checked against the model's solve domain, so
    # two differing supports cannot reach the kernel from the public entries:
    # the kernel still refuses them rather than keep one of them
    from gfem._engine import power_flow
    from gfem._heat import loss_matrices

    model = _line(mesh, gfem.Material(eps_r=3.0, tan_delta=0.05))
    result = gfem.solve(
        model, gfem.Driven(frequency=F, keep_fields=True), options=OPTIONS
    )
    fields = [result.e_field(F, "p1"), result.e_field(F, "p2")]
    supports = iter(
        [np.ones(len(mesh.tetrahedra), bool), np.zeros(len(mesh.tetrahedra), bool)]
    )
    monkeypatch.setattr(power_flow, "check_field", lambda *_: next(supports))
    with pytest.raises(ValueError, match="different supports"):
        loss_matrices(fields, model)


def test_gradient_of_a_coherent_two_port_drive(mesh):
    from gfem._heat import loss_matrices

    n = _core_elements(mesh)
    tan = np.random.default_rng(6).uniform(0.02, 0.08, n)

    def objective(p):
        model = _line(mesh, gfem.MaterialField(eps_r=np.full(n, 3.0), tan_delta=p))
        result = gfem.solve(
            model, gfem.Driven(frequency=F, keep_fields=True), options=OPTIONS
        )
        fields = [result.e_field(F, "p1"), result.e_field(F, "p2")]
        return jnp.sum(_coherent(loss_matrices(fields, model), jnp.asarray(A)))

    _check(objective, tan, step=1e-2)
