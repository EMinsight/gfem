"""Metal-dielectric density blends (ADR 0113): the index scheme and law ends.

`MaterialField.interpolate(..., scheme="index")` blends n - i kappa
linearly, eps = ((1 - rho) N_void + rho N_solid)^2. With a permittivity law at
either end the field holds the density as its leaf and the ends as fixed
data, and the driven operator reads the blend at each frequency. Here: the
field's construction, the driven forward against a constant per-element
permittivity at one frequency, and the paths that refuse a blend. The
finite-difference cells of the density gradient are cells D11 and D12 of the
gradient matrix (tests/verification/test_gradient_matrix.py, input
``"blend"``).

The line is a 10 x 2 x 1 mm parallel-plate line with lumped ports at both
ends; its 2 mm "core" (x in [4, 6]) holds the blend.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem

GHZ = 1e9
OPTIONS = gfem.SolverOptions(linear_solver="direct")

#: A gold-like Drude metal at microwave frequencies: eps = -14.4 - 3.1j at 5 GHz.
METAL = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=1 * GHZ)],
    )
)
DIELECTRIC = gfem.Material(eps_r=2.0)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, core, *, boundaries=(), ports=None):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0), "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *boundaries],
        ports=ports
        or [
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _rho(mesh):
    n = len(mesh.elements("core"))
    return np.random.default_rng(3).uniform(0.2, 0.8, n)


# --- the field -------------------------------------------------------------------


def test_index_blend_of_two_constant_materials_blends_the_refractive_index():
    # n = 2 and n = 3: halfway, n = 2.5 and eps = 6.25
    field = gfem.MaterialField.interpolate(
        jnp.array([0.0, 0.5, 1.0]),
        gfem.Material(eps_r=4.0),
        gfem.Material(eps_r=9.0),
        scheme="index",
    )
    np.testing.assert_allclose(np.asarray(field.eps_r), [4.0, 6.25, 9.0], rtol=1e-14)
    assert field.blend is None


def test_index_blend_of_lossy_constants_is_complex_with_negative_imaginary_part():
    field = gfem.MaterialField.interpolate(
        jnp.array([0.5]),
        gfem.Material(eps_r=1.0),
        gfem.Material(eps_r=4.0, tan_delta=0.5, conductivity=2.0),
        scheme="index",
    )
    n_b = np.sqrt(4.0 * (1 - 0.5j))
    np.testing.assert_allclose(np.asarray(field.eps_r), [((1 + n_b) / 2) ** 2])
    assert field.tan_delta is None
    # the other properties blend linearly in rho
    np.testing.assert_allclose(np.asarray(field.conductivity), [1.0])


def test_a_law_end_gives_a_blend_that_holds_the_density_as_a_leaf():
    rho = jnp.array([0.2, 0.7])
    field = gfem.MaterialField.interpolate(rho, DIELECTRIC, METAL, scheme="index")
    assert field.eps_r is None
    assert field.n_elements == 2
    assert field.blend_ends.solid is METAL.eps_r
    assert field.blend_ends.void == 2.0
    np.testing.assert_array_equal(np.asarray(field.blend), np.asarray(rho))
    assert any(leaf is field.blend for leaf in jax.tree.leaves(field))
    assert "blend 'index'" in repr(field)


@pytest.mark.parametrize("scheme", ["linear", "simp", "ramp"])
def test_a_law_end_is_admitted_under_every_weight_scheme(scheme):
    field = gfem.MaterialField.interpolate(
        jnp.array([0.4]), DIELECTRIC, METAL, scheme=scheme
    )
    assert field.blend_ends.scheme == scheme


def test_a_traced_law_inside_a_blend_raises_with_the_remedy():
    def objective(plasma):
        law = gfem.PermittivityLaw(
            eps_inf=1.0, terms=[gfem.Drude(plasma_frequency=plasma, damping=1e9)]
        )
        field = gfem.MaterialField.interpolate(
            jnp.array([0.5]), DIELECTRIC, gfem.Material(eps_r=law), scheme="index"
        )
        return jnp.sum(field.blend)

    with pytest.raises(
        gfem.NonDifferentiableError,
        match="differentiate the law on a `Material` region",
    ):
        jax.grad(objective)(2e10)


def test_index_refuses_an_anisotropic_end():
    with pytest.raises(ValueError, match="anisotropic"):
        gfem.MaterialField.interpolate(
            jnp.array([0.5]),
            gfem.Material(eps_r=(2.0, 3.0, 4.0)),
            gfem.Material(eps_r=4.0),
            scheme="index",
        )


# --- the driven forward ----------------------------------------------------------


def _eps(end, f):
    return (
        complex(end.eps_r(f))
        if isinstance(end.eps_r, gfem.PermittivityLaw)
        else end.eps_r
    )


@pytest.mark.parametrize("scheme", ["index", "linear", "ramp"])
def test_a_law_blend_solves_as_its_permittivity_at_the_frequency(mesh, scheme):
    # at one frequency the blend is a complex eps per element: its closed form
    f = 5 * GHZ
    rho = _rho(mesh)
    blend = gfem.MaterialField.interpolate(rho, DIELECTRIC, METAL, scheme=scheme)
    ea, eb = _eps(DIELECTRIC, f), _eps(METAL, f)
    if scheme == "index":
        na, nb = np.sqrt(ea), np.sqrt(eb)
        nb = -nb if nb.imag > 0 else nb
        eps = ((1 - rho) * na + rho * nb) ** 2
    else:
        w = rho / (1 + 3.0 * (1 - rho)) if scheme == "ramp" else rho
        eps = (1 - w) * ea + w * eb
    problem = gfem.Driven(frequency=f)
    s = gfem.solve(_line(mesh, blend), problem, options=OPTIONS).s.values
    # Re eps < 0 is no MaterialField value: the reference overrides the mass
    reference = _constant_eps_solve(mesh, eps, problem)
    np.testing.assert_allclose(s, reference, rtol=1e-10, atol=1e-12)


def _constant_eps_solve(mesh, eps, problem):
    """The line with ``eps`` per core element in the mass, through a patched reader."""
    from gfem._engine import dispersion

    core = mesh.elements("core")
    original = dispersion.high_frequency_eps

    def patched(model):
        out = original(model).astype(complex)
        out[core] = eps
        return out

    field = gfem.MaterialField(eps_r=np.full(len(core), 2.0))
    model = _line(mesh, field)
    try:
        dispersion.high_frequency_eps = patched
        return gfem.solve(model, problem, options=OPTIONS).s.values
    finally:
        dispersion.high_frequency_eps = original


# --- the paths that refuse a blend ------------------------------------------------

#: The remedy every refusal of a law blend names (ADR 0113).
UNIFORM_ONLY = (
    r"metal-dielectric blend \(MaterialField.interpolate with a permittivity "
    r"law\) is supported on the driven uniform sweep only"
)


def _blend(mesh, end=METAL, scheme="index"):
    return gfem.MaterialField.interpolate(_rho(mesh), DIELECTRIC, end, scheme=scheme)


@pytest.mark.parametrize(
    ("problem", "match"),
    [
        (gfem.Eigenmode(target=10 * GHZ), "Eigenmode problem"),
        (
            gfem.Driven(frequency=np.linspace(4, 6, 11) * GHZ, sweep="adaptive"),
            "adaptive sweep",
        ),
        (gfem.PortMode("p1", 5 * GHZ), "PortMode problem"),
    ],
    ids=["eigenmode", "adaptive", "port_mode"],
)
def test_a_blend_off_the_driven_uniform_sweep_raises(mesh, problem, match):
    with pytest.raises(NotImplementedError, match=f"'core'.*{match}.*{UNIFORM_ONLY}"):
        gfem.solve(_line(mesh, _blend(mesh)), problem, options=OPTIONS)


def test_a_blend_with_a_drude_end_raises_on_electrostatic(mesh):
    problem = gfem.Electrostatic(terminals={"t": "top"}, ground="bottom")
    with pytest.raises(ValueError, match=r"'core' blends .* Drude"):
        gfem.solve(_line(mesh, _blend(mesh)), problem, options=OPTIONS)


def test_a_blend_behind_an_absorbing_face_raises(mesh):
    model = _line(mesh, _blend(mesh), boundaries=[gfem.Absorbing("side")])
    with pytest.raises(
        NotImplementedError, match="Absorbing 'side' lies on region 'core'"
    ):
        gfem.solve(model, gfem.Driven(frequency=5 * GHZ), options=OPTIONS)


def test_a_reader_that_takes_no_frequency_refuses_a_blend(mesh):
    from gfem._engine.coefficients import element_values

    with pytest.raises(NotImplementedError, match="'core'"):
        element_values(_line(mesh, _blend(mesh)), "eps_r")


def test_a_traced_blend_density_refuses_refinement(mesh):
    from gfem._refine import refine_model

    marked = np.zeros(len(mesh.tetrahedra), bool)

    def objective(rho):
        field = gfem.MaterialField.interpolate(rho, DIELECTRIC, METAL, scheme="index")
        refine_model(_line(mesh, field), marked)
        return jnp.sum(rho)

    with pytest.raises(gfem.NonDifferentiableError, match="materials\\['core'\\]"):
        jax.grad(objective)(jnp.asarray(_rho(mesh)))


def test_refinement_copies_a_blend_density_to_the_children(mesh):
    from gfem._refine import refine_model

    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[mesh.elements("core")[0]] = True
    fine, _ = refine_model(_line(mesh, _blend(mesh)), marked)
    field = fine.materials["core"]
    assert field.n_elements == len(fine.mesh.elements("core"))
    assert set(np.round(np.asarray(field.blend), 12)) == set(np.round(_rho(mesh), 12))


def test_a_frequency_on_a_lossless_pole_of_a_blend_end_raises(mesh):
    lossless = gfem.Material(
        eps_r=gfem.PermittivityLaw(
            eps_inf=2.0,
            terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=8 * GHZ, damping=0)],
        )
    )
    model = _line(mesh, _blend(mesh, end=lossless))
    with pytest.raises(ValueError, match="solid end of the blend in region 'core'"):
        gfem.solve(model, gfem.Driven(frequency=8 * GHZ), options=OPTIONS)


def test_the_static_check_refuses_a_blend_with_a_drude_end(mesh):
    # the electrostatic rule's own check, whatever calls it (ADR 0111, 0113)
    from gfem._engine.dispersion import check_static

    with pytest.raises(ValueError, match=r"'core' blends .* Drude"):
        check_static(_line(mesh, _blend(mesh)))


def test_the_static_check_refuses_a_blend_without_a_drude_end_as_unsupported(mesh):
    lossy = gfem.Material(
        eps_r=gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1e9)
    )
    from gfem._engine.dispersion import check_static

    with pytest.raises(NotImplementedError, match=f"'core'.*{UNIFORM_ONLY}"):
        check_static(_line(mesh, _blend(mesh, end=lossy)))


@pytest.fixture(scope="module")
def solved_blend(mesh):
    model = _line(mesh, _blend(mesh))
    problem = gfem.Driven(frequency=[5 * GHZ], excite="p1", keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


def test_an_error_estimate_of_a_blend_raises_with_the_remedy(solved_blend):
    model, result = solved_blend
    with pytest.raises(
        NotImplementedError, match=f"'core'.*error_estimate.*{UNIFORM_ONLY}"
    ):
        gfem.error_estimate(result, gfem.FieldEstimator(), model=model)


def test_the_electric_energy_of_a_blend_raises_with_the_remedy(solved_blend):
    model, result = solved_blend
    with pytest.raises(
        NotImplementedError, match=f"'core'.*electric_energy.*{UNIFORM_ONLY}"
    ):
        gfem.electric_energy(result.e_field(5 * GHZ), model=model)


@pytest.mark.parametrize(
    "estimator",
    [gfem.FieldEstimator(), gfem.QuantityEstimator(s=("p2", "p1"))],
    ids=["field", "s"],
)
def test_mesh_refinement_of_a_blend_raises_before_any_solve(
    mesh, monkeypatch, estimator
):
    def solve(*_args, **_kwargs):
        raise AssertionError("refine_mesh solved before refusing")

    monkeypatch.setattr("gfem.refinement._solve", solve)
    with pytest.raises(
        NotImplementedError, match=f"'core'.*refine_mesh.*{UNIFORM_ONLY}"
    ):
        gfem.refine_mesh(
            _line(mesh, _blend(mesh)),
            gfem.Driven(frequency=[5 * GHZ]),
            estimator=estimator,
        )


def test_a_reader_that_takes_no_frequency_names_the_remedy(mesh):
    from gfem._engine.coefficients import lossy_eps

    with pytest.raises(NotImplementedError, match=f"'core'.*{UNIFORM_ONLY}") as err:
        lossy_eps(_line(mesh, _blend(mesh)))
    assert "internal" not in str(err.value)


# --- serialization and the density's range -----------------------------------------


@pytest.mark.parametrize(
    ("void", "solid", "scheme"),
    [
        (gfem.Material(eps_r=3.0, tan_delta=0.01), METAL, "ramp"),
        (METAL, gfem.Material(eps_r=3.0, tan_delta=0.01), "index"),
    ],
    ids=["constant-void", "law-void"],
)
def test_a_job_with_a_law_blend_round_trips(mesh, tmp_path, void, solid, scheme):
    blend = gfem.MaterialField.interpolate(_rho(mesh), void, solid, scheme=scheme)
    model = _line(mesh, blend)
    job = gfem.Job(model, gfem.Driven(frequency=[5 * GHZ]))
    back = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    core = back.model.materials["core"]
    assert core.blend_ends == blend.blend_ends
    np.testing.assert_array_equal(np.asarray(core.blend), _rho(mesh))


@pytest.mark.parametrize("value", [1.2, -0.1])
@pytest.mark.parametrize("scheme", ["index", "linear", "simp"])
def test_a_blend_density_outside_0_1_raises(mesh, value, scheme):
    rho = _rho(mesh)
    rho[0] = value
    with pytest.raises(ValueError, match=r"blend.*\[0, 1\]"):
        gfem.MaterialField.interpolate(rho, DIELECTRIC, METAL, scheme=scheme)


@pytest.mark.parametrize("value", [1.2, -0.1])
def test_a_traced_blend_density_outside_0_1_raises_naming_the_region(mesh, value):
    rho = _rho(mesh)
    rho[0] = value

    def objective(r):
        field = gfem.MaterialField.interpolate(r, DIELECTRIC, METAL, scheme="linear")
        problem = gfem.Driven(frequency=[5 * GHZ])
        s = gfem.solve(_line(mesh, field), problem, options=OPTIONS).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    with pytest.raises(Exception, match=r"blend.*\[0, 1\]") as raised:
        jax.grad(objective)(jnp.asarray(rho))
    assert "region 'core'" in str(raised.value)
