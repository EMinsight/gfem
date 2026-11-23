"""Incident-field columns: the total/scattered solve, its parts, its rules.

A unit cube of air ("air", um) wrapped in a perfectly matched layer, with
an optional dielectric cube ("scat") in its middle, on a coarse structured
mesh: the checks here are identities of the formulation and refusals,
which hold on any mesh.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest

import gfem

ALL = ("-x", "+x", "-y", "+y", "-z", "+z")
AIR = gfem.Material()
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")


def _scatterer(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _air(c):
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _feed(c):
    """The scatterer's +x face, a lumped port's surface."""
    on = abs(c[0] - 0.75) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75
    return "feed" if on else ""


def _slab(c):
    """A slab of the cube's +x end, touching the layer."""
    if c[0] > 0.75 and ((c > 0) & (c < 1)).all():
        return "slab"
    return _air(c)


def _walls(c):
    """The cube's x and y faces as walls; the layer runs along z."""
    if abs(c[0]) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75:
        return "xfeed"
    if abs(c[0]) < 1e-9 or abs(c[0] - 1) < 1e-9:
        return "xwall"
    if abs(c[1]) < 1e-9 or abs(c[1] - 1) < 1e-9:
        return "ywall"
    return ""


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("incident")
    return {
        "slab": gfem.Mesh.from_gmsh(pc.box(tmp / "sl.msh", ALL, cells=4, volume=_slab)),
        "closed": gfem.Mesh.from_gmsh(pc.box(tmp / "c.msh", (), cells=4, volume=_slab)),
        "walls": gfem.Mesh.from_gmsh(
            pc.box(tmp / "w.msh", ("-z", "+z"), cells=4, surface=_walls)
        ),
        "fed": gfem.Mesh.from_gmsh(
            pc.box(tmp / "f.msh", ALL, cells=4, volume=_scatterer, surface=_feed)
        ),
        "empty": gfem.Mesh.from_gmsh(pc.box(tmp / "e.msh", ALL, cells=4, volume=_air)),
        "scatterer": gfem.Mesh.from_gmsh(
            pc.box(tmp / "s.msh", ALL, cells=4, volume=_scatterer)
        ),
    }


def _model(mesh, scat=None, *, region="scat", **kw):
    materials = {"air": AIR, "pml": AIR}
    if scat is not None:
        materials[region] = scat
    return gfem.Model(
        mesh, materials=materials, pml=gfem.PML("pml"), background=AIR, **kw
    )


def _solve(model, *sources, **kw):
    problem = gfem.Driven(F, sources=sources, keep_fields=True, **kw)
    return gfem.solve(model, problem, options=OPTIONS)


def _norm(field):
    return float(np.linalg.norm(np.asarray(field.coefficients)))


#: A beam focused in the middle of the cube, tilted, circular.
BEAM = gfem.GaussianBeam(
    "w",
    waist=0.6e-6,
    focus=(0.5e-6, 0.5e-6, 0.5e-6),
    theta=0.5,
    phi=1.0,
    polarization="RCP",
)


@pytest.mark.parametrize("source", [WAVE, BEAM], ids=["plane_wave", "beam"])
def test_without_a_scatterer_the_scattered_field_vanishes(meshes, source):
    result = _solve(_model(meshes["empty"]), source)
    total = result.e_field(F, "w")
    scattered = result.e_field(F, "w", part="scattered")
    assert _norm(scattered) <= 1e-12 * _norm(total)


def _coefficients(field):
    return np.asarray(field.coefficients)


def test_total_is_scattered_plus_incident(meshes):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    result = _solve(model, WAVE)
    total = _coefficients(result.e_field(F, "w"))
    scattered = _coefficients(result.e_field(F, "w", part="scattered"))
    incident = _coefficients(result.e_field(F, "w", part="incident"))
    np.testing.assert_allclose(
        scattered + incident, total, rtol=0, atol=1e-15 * np.abs(total).max()
    )
    # the scatterer scatters
    assert np.linalg.norm(scattered) > 1e-2 * np.linalg.norm(total)


def test_a_material_field_equal_to_the_background_scatters_nothing(meshes):
    n = len(meshes["scatterer"].elements("scat"))
    model = _model(meshes["scatterer"], gfem.MaterialField(eps_r=np.ones(n)))
    result = _solve(model, WAVE)
    total = result.e_field(F, "w")
    assert _norm(result.e_field(F, "w", part="scattered")) <= 1e-12 * _norm(total)


def test_the_incident_part_is_the_wave_outside_the_layer_and_zero_in_it(meshes):
    f = F / 4  # resolved: k h = 0.4 at order 2
    result = gfem.solve(
        _model(meshes["empty"]),
        gfem.Driven(f, sources=WAVE, keep_fields=True),
        options=OPTIONS,
    )
    incident = result.e_field(f, part="incident")
    inside = np.array([[0.3, 0.6, 0.45], [0.7, 0.2, 0.8]]) * 1e-6
    k = 2 * np.pi * f / 299792458.0
    k_hat, e = (
        np.array([np.sin(0.4) * np.cos(0.3), np.sin(0.4) * np.sin(0.3), np.cos(0.4)]),
        np.array([np.cos(0.4) * np.cos(0.3), np.cos(0.4) * np.sin(0.3), -np.sin(0.4)]),
    )
    exact = np.exp(-1j * k * inside @ k_hat)[:, None] * e
    np.testing.assert_allclose(np.asarray(incident.at(inside)), exact, atol=2e-2)
    in_layer = np.array([[1.3, 0.5, 0.5], [0.5, -0.3, 1.2]]) * 1e-6
    assert np.abs(np.asarray(incident.at(in_layer))).max() == 0


def _same_wave(points, frequency):
    """``WAVE`` as an incident-field function."""
    k = 2 * np.pi * frequency / 299792458.0
    k_hat = np.array(
        [np.sin(0.4) * np.cos(0.3), np.sin(0.4) * np.sin(0.3), np.cos(0.4)]
    )
    e = np.array([np.cos(0.4) * np.cos(0.3), np.cos(0.4) * np.sin(0.3), -np.sin(0.4)])
    return np.exp(-1j * k * points @ k_hat)[:, None] * e


def test_a_plane_wave_and_the_same_incident_field_solve_alike(meshes):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    result = _solve(model, WAVE, gfem.IncidentField("f", _same_wave))
    for part in ("total", "scattered", "incident"):
        a = _coefficients(result.e_field(F, "w", part=part))
        b = _coefficients(result.e_field(F, "f", part=part))
        assert np.linalg.norm(a - b) <= 1e-12 * np.linalg.norm(a), part


WAVE_2 = gfem.PlaneWave("v", theta=2.0, phi=-1.0, polarization="RCP", amplitude=2j)


def _fed_model(mesh):
    port = gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)
    return _model(mesh, gfem.Material(eps_r=4.0), ports=[port])


def test_ports_and_sources_in_one_solve_are_separate_solves(meshes, caplog):
    model = _fed_model(meshes["fed"])
    gfem.release_memory()
    caplog.set_level(logging.INFO, logger="gfem.timing")
    together = _solve(model, WAVE, WAVE_2, excite="p1")
    setups = [r for r in caplog.records if getattr(r, "phase", None) == "setup"]
    assert len(setups) == 1  # one factorization for the three columns
    port = gfem.solve(model, gfem.Driven(F, keep_fields=True), options=OPTIONS)
    np.testing.assert_allclose(together.s.values, port.s.values, rtol=1e-12)
    for name, wave in (("w", WAVE), ("v", WAVE_2)):
        alone = _solve(model, wave, excite=())
        assert alone.s.values.shape == (1, 1, 0)
        np.testing.assert_allclose(
            together.port_wave("p1", name), alone.port_wave("p1", name), rtol=1e-12
        )
        a = _coefficients(together.e_field(F, name))
        b = _coefficients(alone.e_field(F, name))
        assert np.linalg.norm(a - b) <= 1e-12 * np.linalg.norm(a)
    a = _coefficients(together.e_field(F, "p1"))
    b = _coefficients(port.e_field(F, "p1"))
    assert np.linalg.norm(a - b) <= 1e-12 * np.linalg.norm(a)
    assert np.abs(together.port_wave("p1", "v")) > 0


# --- where the incident field meets the truncation, ports and walls -------------


def _envelope(width):
    """A wave along +x under a Gaussian envelope centred near the -x end."""

    def field(points, frequency):
        k = 2 * np.pi * frequency / 299792458.0
        r2 = np.sum((points / 1e-6 - [0.1, 0.5, 0.5]) ** 2, axis=1)
        e = np.exp(-1j * k * points[:, 0] - r2 / width**2)
        return np.stack([0 * e, 0 * e, e], axis=1)

    return field


def test_the_incident_contact_below_its_limit_is_accepted(meshes):
    n = len(meshes["slab"].elements("slab"))
    model = _model(
        meshes["slab"], gfem.MaterialField(eps_r=np.full(n, 2.0)), region="slab"
    )
    result = _solve(model, gfem.IncidentField("g", _envelope(0.2)))
    assert _norm(result.e_field(F, part="scattered")) > 0


def test_the_incident_contact_above_its_limit_is_refused_with_its_ratio(meshes):
    model = _model(meshes["slab"], gfem.Material(eps_r=2.0), region="slab")
    with pytest.raises(
        ValueError,
        match=r"incident field 'g' reaches the scatterer where it enters the PML "
        r"layer \(region 'slab'.*\|E_inc\| there is 1\.[0-9]e-01 of its peak in the "
        r"physical region, above 1e-03",
    ):
        _solve(model, gfem.IncidentField("g", _envelope(0.6)))


def test_a_material_field_is_a_scatterer_whatever_its_values(meshes):
    n = len(meshes["slab"].elements("slab"))
    # the background's material: no scatterer, the wave may cross the layer
    _solve(_model(meshes["slab"], AIR, region="slab"), WAVE)
    same = gfem.MaterialField(eps_r=np.ones(n))
    with pytest.raises(ValueError, match="region 'slab'"):
        _solve(_model(meshes["slab"], same, region="slab"), WAVE)


def test_a_scatterer_on_an_absorbing_face_is_refused(meshes):
    model = gfem.Model(
        meshes["closed"],
        materials={"air": AIR, "slab": gfem.Material(eps_r=2.0)},
        boundaries=[gfem.Absorbing("outer")],
        background=AIR,
    )
    with pytest.raises(
        ValueError, match=r"reaches a scatterer on the absorbing face 'outer'"
    ):
        _solve(model, WAVE)


def test_a_closed_model_is_refused(meshes):
    model = gfem.Model(
        meshes["closed"],
        materials={"air": AIR, "slab": AIR},
        boundaries=[gfem.PEC("outer")],
        background=AIR,
    )
    with pytest.raises(ValueError, match="needs a radiating truncation"):
        _solve(model, WAVE)


def _walled(mesh, **kw):
    return gfem.Model(
        mesh,
        materials={"air": AIR, "pml": AIR},
        pml=gfem.PML("pml"),
        background=AIR,
        **kw,
    )


#: Along +z, E along x (TM at phi = 0) or y (TE).
NORMAL_TM = gfem.PlaneWave("n", theta=0.0, polarization="TM")
NORMAL_TE = gfem.PlaneWave("n", theta=0.0, polarization="TE")


def test_walls_the_incident_field_satisfies_are_accepted(meshes):
    # PEC where E is normal, PMC (by default) where H is normal
    model = _walled(meshes["walls"], boundaries=[gfem.PEC("xwall"), gfem.PEC("xfeed")])
    result = _solve(model, NORMAL_TM)
    total = result.e_field(F)
    assert _norm(result.e_field(F, part="scattered")) <= 1e-12 * _norm(total)


def test_a_pec_wall_the_incident_field_crosses_is_refused(meshes):
    model = _walled(meshes["walls"], boundaries=[gfem.PEC("xwall"), gfem.PEC("xfeed")])
    with pytest.raises(
        ValueError,
        match=r"'n' does not satisfy the PEC wall 'xwall': \|n x E_inc\| is 1\.0e\+00",
    ):
        _solve(model, NORMAL_TE)


def test_a_default_pmc_wall_the_incident_field_crosses_is_refused(meshes):
    with pytest.raises(
        ValueError,
        match=r"the exterior face 'ywall' \(PMC by default, no boundary condition "
        r"given\): eta \|n x H_inc\| is 1\.0e\+00",
    ):
        _solve(_walled(meshes["walls"]), NORMAL_TE)


@pytest.mark.parametrize(
    ("item", "match"),
    [
        (gfem.Impedance("xwall", resistance=377.0), "the exterior Impedance face"),
        (
            gfem.LumpedPort("p", "xfeed", "+y", resistance=50.0),
            "the lumped port 'p' on the exterior face 'xfeed'",
        ),
    ],
)
def test_an_exterior_lossy_or_lumped_face_is_refused(meshes, item, match):
    if isinstance(item, gfem.LumpedPort):
        model = _walled(meshes["walls"], ports=[item])
    else:
        model = _walled(meshes["walls"], boundaries=[item])
    with pytest.raises(ValueError, match=f"cannot meet {match}"):
        _solve(model, NORMAL_TM)


# --- what this version does not solve ----------------------------------------------


@pytest.mark.parametrize(
    ("background", "source", "match"),
    [
        (
            gfem.Background([AIR, gfem.Material(eps_r=2.0)], interfaces=[0.5e-6]),
            gfem.GaussianBeam("b", waist=1e-6, theta=0.0, polarization="TM"),
            "GaussianBeam 'b' is solved in a homogeneous background",
        ),
        (
            gfem.Material(tan_delta=0.1),
            gfem.GaussianBeam("b", waist=1e-6, theta=0.0, polarization="TM"),
            "GaussianBeam 'b' travels in a lossy background",
        ),
        (gfem.Material(tan_delta=0.1), WAVE, "lossy background"),
    ],
)
def test_sources_this_version_does_not_solve_are_refused(
    meshes, background, source, match
):
    model = _model(meshes["empty"]).replace(background=background)
    with pytest.raises(NotImplementedError, match=match):
        _solve(model, source)


def test_a_beam_too_tight_at_one_frequency_is_refused_before_factorization(meshes):
    # k w0 = 0.19 at the lower frequency only
    beam = gfem.GaussianBeam(
        "b", waist=0.19 / (2 * np.pi) * 1e-6, theta=0.0, polarization="TE"
    )
    problem = gfem.Driven([F, 1.5 * F], sources=beam)
    with pytest.raises(ValueError, match=r"GaussianBeam 'b' has k w0 = 0\.19"):
        gfem.solve(_model(meshes["empty"]), problem, options=OPTIONS)


def test_frequency_workers_do_not_take_incident_fields_yet(meshes):
    options = gfem.SolverOptions(order=1, frequency_workers=2)
    problem = gfem.Driven([F, 1.1 * F], sources=WAVE)
    with pytest.raises(NotImplementedError, match="frequency_workers=1"):
        gfem.solve(_model(meshes["empty"]), problem, options=options)


# --- gradients ------------------------------------------------------------------------


def _field_norm(model, problem, part="total"):
    result = gfem.solve(model, problem, options=OPTIONS)
    return jnp.sum(jnp.abs(result.e_field(F, part=part).coefficients) ** 2)


def test_node_coordinates_are_not_differentiable_with_an_incident_field(meshes):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    structure = jax.tree_util.tree_structure(model.mesh)
    problem = gfem.Driven(F, sources=WAVE, keep_fields=True)

    def objective(x):
        moved = jax.tree_util.tree_unflatten(structure, [x])
        return _field_norm(model.replace(mesh=moved), problem)

    with pytest.raises(gfem.NonDifferentiableError, match="incident field 'w'"):
        jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))


def test_a_design_touching_the_layer_is_not_differentiable(meshes):
    n = len(meshes["slab"].elements("slab"))
    model = _model(meshes["slab"], gfem.MaterialField(eps_r=np.ones(n)), region="slab")
    problem = gfem.Driven(
        F, sources=gfem.IncidentField("g", _envelope(0.2)), keep_fields=True
    )

    def objective(eps):
        return _field_norm(
            model.with_material("slab", gfem.MaterialField(eps_r=eps)), problem
        )

    with pytest.raises(
        gfem.NonDifferentiableError, match="region 'slab' meets the PML"
    ):
        jax.make_jaxpr(jax.grad(objective))(jnp.ones(n))


def test_the_incident_part_has_a_zero_gradient(meshes):
    n = len(meshes["scatterer"].elements("scat"))
    model = _model(meshes["scatterer"], gfem.MaterialField(eps_r=np.full(n, 4.0)))
    problem = gfem.Driven(F, sources=WAVE, keep_fields=True)

    def objective(eps, part):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=eps))
        return _field_norm(moved, problem, part)

    eps = jnp.full(n, 4.0)
    assert not np.asarray(jax.grad(objective)(eps, "incident")).any()
    assert np.abs(np.asarray(jax.grad(objective)(eps, "scattered"))).max() > 0


# --- a written result ---------------------------------------------------------------


def test_a_written_result_rebuilds_its_parts(meshes, tmp_path):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    job = gfem.Job(model, gfem.Driven(F, sources=WAVE, keep_fields=True), OPTIONS)
    result = job.run(tmp_path / "out")
    back = gfem.load_result(tmp_path / "out")
    for part in ("total", "scattered", "incident"):
        np.testing.assert_array_equal(
            _coefficients(back.e_field(F, "w", part=part)),
            _coefficients(result.e_field(F, "w", part=part)),
        )


def test_a_result_of_an_incident_function_cannot_be_written(meshes, tmp_path):
    job = gfem.Job(
        _model(meshes["empty"]),
        gfem.Driven(F, sources=gfem.IncidentField("f", _same_wave), keep_fields=True),
        OPTIONS,
    )
    with pytest.raises(TypeError, match="IncidentField 'f'"):
        job.run(tmp_path / "out")


def test_an_incident_field_on_a_wave_port_face_is_refused(meshes):
    model = _walled(
        meshes["walls"],
        boundaries=[gfem.PEC("xwall")],
        ports=[gfem.WavePort("o1", "xfeed")],
    )
    with pytest.raises(
        ValueError,
        match=r"reaches wave port surface 'xfeed': \|E_inc\| on the port face is "
        r"1\.0e\+00 of its peak.*drive the port and use the far field",
    ):
        _solve(model, NORMAL_TM, excite=())
