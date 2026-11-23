"""Incident fields on a layered background: the matched stack, its rules.

A unit cube (um) on a coarse structured mesh, cut by the planes z = 0.25
and z = 0.5 into a substrate, a lossy film and air, wrapped in a perfectly
matched layer whose regions are split at the same planes and carry the
stack's materials. With no scatterer the model is the background stack, so
the scattered field vanishes to round-off whatever the incidence; the rest
are the refusals of a stack.
"""

import math

import numpy as np
import pml_cases as pc
import pytest

import gfem

ALL = ("-x", "+x", "-y", "+y", "-z", "+z")
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0
UM = 1e-6
SUB = gfem.Material(eps_r=2.25)
FILM = gfem.Material(eps_r=4.0, tan_delta=0.1)
AIR = gfem.Material()
STACK = gfem.Background([SUB, FILM, AIR], interfaces=[0.25 * UM, 0.5 * UM])
#: Incidence 40 degrees from the top; from the substrate at 50 degrees, past
#: the critical angle of the film-free substrate/air pair (41.8 degrees).
TOP = math.pi - math.radians(40)
TIR = math.radians(50)


def _inside(c):
    return bool(((c > 0) & (c < 1)).all())


def _layer(z):
    return "sub" if z < 0.25 else "film" if z < 0.5 else "air"


def _stack(c):
    name = _layer(c[2])
    return name if _inside(c) else f"pml_{name}"


def _particle(c):
    if 0.25 < c[0] < 0.75 and 0.25 < c[1] < 0.75 and 0.5 < c[2] < 0.75:
        return "particle"
    return _stack(c)


def _post(c):
    """A post across the film/air plane, one region."""
    if 0.25 < c[0] < 0.75 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75:
        return "post"
    return _stack(c)


def _grounded(c):
    """No layer below: the cube's bottom z = 0 is the ground plane."""
    return _stack(c)


def _bottom(c):
    return "ground" if abs(c[2]) < 1e-9 else ""


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("layered")

    def box(name, sides, volume, **kw):
        path = pc.box(
            tmp / f"{name}.msh", sides, thickness=0.25, cells=4, volume=volume, **kw
        )
        return gfem.Mesh.from_gmsh(path)

    return {
        "stack": box("s", ALL, _stack),
        "particle": box("p", ALL, _particle),
        "post": box("q", ALL, _post),
        "grounded": box("g", (*ALL[:-2], "+z"), _grounded, surface=_bottom),
    }


MATERIALS = {
    "sub": SUB,
    "film": FILM,
    "air": AIR,
    "pml_sub": SUB,
    "pml_film": FILM,
    "pml_air": AIR,
}
PML = gfem.PML(["pml_sub", "pml_film", "pml_air"])


def _model(mesh, background=STACK, **changes):
    materials = {**MATERIALS, **changes.pop("materials", {})}
    materials = {k: v for k, v in materials.items() if k in mesh.volumes}
    return gfem.Model(
        mesh, materials=materials, pml=PML, background=background, **changes
    )


def _solve(model, *sources):
    problem = gfem.Driven(F, sources=sources, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


def _norm(field):
    return float(np.linalg.norm(np.asarray(field.coefficients)))


def _vanishes(result, name):
    total = result.e_field(F, name)
    scattered = result.e_field(F, name, part="scattered")
    assert _norm(total) > 0
    assert _norm(scattered) <= 1e-12 * _norm(total), name


# --- the matched stack: no scattered field -------------------------------------


def test_a_matched_stack_scatters_nothing_from_the_top(meshes):
    result = _solve(
        _model(meshes["stack"]),
        gfem.PlaneWave("te", theta=TOP, phi=0.3, polarization="TE"),
        gfem.PlaneWave("tm", theta=TOP, phi=0.3, polarization="TM"),
    )
    _vanishes(result, "te")
    _vanishes(result, "tm")


def test_a_matched_stack_scatters_nothing_under_total_internal_reflection(meshes):
    result = _solve(
        _model(meshes["stack"]),
        gfem.PlaneWave("te", theta=TIR, phi=0.3, polarization="TE"),
        gfem.PlaneWave("tm", theta=TIR, phi=0.3, polarization="TM"),
    )
    _vanishes(result, "te")
    _vanishes(result, "tm")


def test_a_stack_on_a_ground_plane_scatters_nothing(meshes):
    stack = gfem.Background([SUB, FILM, AIR], interfaces=STACK.interfaces, ground=0.0)
    model = _model(
        meshes["grounded"], background=stack, boundaries=[gfem.PEC("ground")]
    )
    result = _solve(
        model, gfem.PlaneWave("w", theta=TOP, phi=0.3, polarization=(0.6, 0.8j))
    )
    _vanishes(result, "w")


def test_the_incident_part_is_the_stack_field(meshes):
    # the film and the substrate change the field from the plain wave
    f = F / 4  # resolved: k h = 0.6 in the film at order 2
    wave = gfem.PlaneWave("w", theta=TOP, polarization="TE")
    result = gfem.solve(
        _model(meshes["stack"]),
        gfem.Driven(f, sources=wave, keep_fields=True),
        options=OPTIONS,
    )
    incident = result.e_field(f, "w", part="incident")
    points = np.array([[0.4, 0.5, 0.1], [0.3, 0.6, 0.4], [0.6, 0.4, 0.9]]) * UM
    from gfem._transfer_matrix import StackWave

    exact = StackWave(STACK, wave, f).electric(points)
    np.testing.assert_allclose(np.asarray(incident.at(points)), exact, atol=2e-2)


def test_a_particle_on_the_film_scatters(meshes):
    model = _model(meshes["particle"], materials={"particle": gfem.Material(eps_r=6)})
    result = _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TM"))
    total = result.e_field(F, "w")
    assert _norm(result.e_field(F, "w", part="scattered")) > 1e-2 * _norm(total)


def test_a_region_across_a_stack_plane_is_classified_per_element(meshes):
    # a film-material post through the film/air plane scatters only above
    # the plane: the same as a film-material particle in the air
    wave = gfem.PlaneWave("w", theta=TOP, phi=0.3, polarization="TM")
    post = _solve(_model(meshes["post"], materials={"post": FILM}), wave)
    particle = _solve(_model(meshes["particle"], materials={"particle": FILM}), wave)
    for part in ("total", "scattered"):
        a = np.asarray(post.e_field(F, "w", part=part).at(PROBES))
        b = np.asarray(particle.e_field(F, "w", part=part).at(PROBES))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10 * np.abs(b).max())
    scattered = np.asarray(post.e_field(F, "w", part="scattered").at(PROBES))
    assert np.abs(scattered).max() > 1e-3


PROBES = np.array([[0.1, 0.2, 0.1], [0.5, 0.5, 0.4], [0.6, 0.4, 0.6], [0.8, 0.9, 0.9]])
PROBES = PROBES * UM


# --- refusals ------------------------------------------------------------------


def test_a_tetrahedron_across_a_stack_plane_is_refused(meshes):
    stack = gfem.Background([SUB, AIR], interfaces=[0.4 * UM])
    model = _model(
        meshes["stack"],
        background=stack,
        materials={"film": SUB, "pml_film": SUB},
    )
    with pytest.raises(ValueError, match=r"stack plane z = 4e-07 m cuts .*'film'"):
        _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TE"))


def test_a_lossy_incidence_layer_is_refused(meshes):
    lossy = gfem.Material(eps_r=2.25, conductivity=1e3)
    stack = gfem.Background([lossy, FILM, AIR], interfaces=STACK.interfaces)
    model = _model(
        meshes["stack"], background=stack, materials={"sub": lossy, "pml_sub": lossy}
    )
    with pytest.raises(NotImplementedError, match=r"comes from layer 0 .* lossy"):
        _solve(model, gfem.PlaneWave("w", theta=TIR, polarization="TE"))


def test_grazing_incidence_on_a_stack_is_refused(meshes):
    with pytest.raises(ValueError, match=r"grazing"):
        _solve(
            _model(meshes["stack"]),
            gfem.PlaneWave("w", theta=math.pi / 2, polarization="TE"),
        )


def test_incidence_from_below_a_ground_plane_is_refused(meshes):
    stack = gfem.Background([SUB, FILM, AIR], interfaces=STACK.interfaces, ground=0.0)
    model = _model(
        meshes["grounded"], background=stack, boundaries=[gfem.PEC("ground")]
    )
    with pytest.raises(ValueError, match=r"from below the ground plane"):
        _solve(model, gfem.PlaneWave("w", theta=TIR, polarization="TE"))


def test_a_ground_plane_needs_a_pec_face_on_it(meshes):
    stack = gfem.Background([SUB, FILM, AIR], interfaces=STACK.interfaces, ground=0.0)
    model = _model(meshes["grounded"], background=stack)
    with pytest.raises(ValueError, match=r"ground plane at z = 0.0 m needs .*PEC"):
        _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TE"))


def test_a_ground_plane_off_the_mesh_bottom_is_refused(meshes):
    stack = gfem.Background(
        [SUB, FILM, AIR], interfaces=STACK.interfaces, ground=-0.25 * UM
    )
    model = _model(
        meshes["grounded"], background=stack, boundaries=[gfem.PEC("ground")]
    )
    with pytest.raises(ValueError, match=r"ground plane at z = -2.5e-07 m needs"):
        _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TE"))


def test_elements_below_the_ground_plane_are_refused(meshes):
    stack = gfem.Background([SUB, FILM, AIR], interfaces=STACK.interfaces, ground=0.0)
    with pytest.raises(ValueError, match=r"below the background stack.s ground plane"):
        _solve(
            _model(meshes["stack"], background=stack),
            gfem.PlaneWave("w", theta=TOP, polarization="TE"),
        )


def test_a_gaussian_beam_on_several_layers_is_refused(meshes):
    beam = gfem.GaussianBeam("b", waist=0.5 * UM, theta=TOP, polarization="TE")
    with pytest.raises(NotImplementedError, match=r"layered background.*IncidentField"):
        _solve(_model(meshes["stack"]), beam)


def test_a_layer_region_off_the_stack_is_refused_by_its_contact(meshes):
    model = _model(meshes["stack"], materials={"pml_film": AIR})
    with pytest.raises(
        ValueError,
        match=r"region 'pml_film'.*background stack's layer 1 there, "
        r"Material\(eps_r=4\.0, tan_delta=0\.1\)",
    ):
        _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TE"))


def test_cross_sections_refuse_a_lossy_stack(meshes):
    model = _model(meshes["particle"], materials={"particle": gfem.Material(eps_r=6)})
    result = _solve(model, gfem.PlaneWave("w", theta=TOP, polarization="TM"))
    with pytest.raises(NotImplementedError, match=r"lossy background \(layer 1 "):
        gfem.cross_sections(result, "w", model=model)
