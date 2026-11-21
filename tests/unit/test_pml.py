r"""`gfem.PML`: declaring a perfectly matched layer on a model.

The layer meshes are `pml_cases.box`: the unit cube "air" (micrometres)
with layer cells beyond some of its sides. The model derives the interior
box and each side's thickness from the mesh, refuses layers it cannot
stretch, and leaves models without a layer as they were. The profile
helpers are checked against the continuum PML reflection,
:math:`|R| = \exp(-2k\cos\theta\int_0^d\sigma\,\mathrm d\xi)`.
"""

import json
import logging
import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases
import pytest
import structured_meshes as sm

import gfem
from gfem import _pml
from gfem._engine.stretch import pml_tensor

AIR = gfem.Material()
LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"


def _box(tmp_path, sides=("+x",), name="box.msh", **kw):
    return gfem.Mesh.from_gmsh(pml_cases.box(tmp_path / name, sides, **kw))


def _model(mesh, pml=None, materials=None, **kw):
    pml = gfem.PML("pml") if pml is None else pml
    materials = {"air": AIR, "pml": AIR} if materials is None else materials
    return gfem.Model(mesh, materials=materials, pml=pml, **kw)


# --- the public value -------------------------------------------------------------


def test_pml_is_public_from_gfem_and_gfem_model():
    assert gfem.PML is gfem.model.PML
    assert "PML" in gfem.__all__
    assert "PML" in gfem.model.__all__


@pytest.mark.parametrize("name", ["LayerClearanceWarning", "ThinLayerWarning"])
def test_the_layer_warnings_are_public_user_warnings(name):
    cls = getattr(gfem, name)
    assert name in gfem.__all__
    assert cls is getattr(gfem.errors, name)
    assert issubclass(cls, UserWarning)


def test_a_layer_defaults_to_a_target_reflection_of_one_in_a_million():
    pml = gfem.PML("pml")
    assert pml.regions == ("pml",)
    assert pml.target_reflection == 1e-6
    assert pml.reference_frequency is None


def test_a_layer_stores_its_regions_as_a_tuple_and_its_knobs_as_floats():
    pml = gfem.PML(["a", "b"], target_reflection=1e-3, reference_frequency=2)
    assert pml.regions == ("a", "b")
    assert type(pml.reference_frequency) is float
    assert pml == gfem.PML(("a", "b"), target_reflection=1e-3, reference_frequency=2e0)


def test_the_knobs_are_keyword_only():
    with pytest.raises(TypeError):
        gfem.PML("pml", 1e-3)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        ((3,), {}, TypeError, r"PML: regions must be volume region names .* got 3"),
        ((["a", 3],), {}, TypeError, "PML: regions must be volume region names"),
        (((),), {}, ValueError, "PML: name at least one layer region"),
        ((("pml", "pml"),), {}, ValueError, "PML: region 'pml' is listed twice"),
        (("pml",), {"target_reflection": 0.0}, ValueError, "between 0 and 1"),
        (("pml",), {"target_reflection": 1.0}, ValueError, "between 0 and 1"),
        (("pml",), {"target_reflection": math.nan}, ValueError, "target_reflection"),
        (("pml",), {"reference_frequency": -1.0}, ValueError, "> 0 Hz, got -1.0"),
        (("pml",), {"reference_frequency": 0}, ValueError, "> 0 Hz"),
    ],
)
def test_a_layer_refuses_bad_values(args, kwargs, error, match):
    with pytest.raises(error, match=match):
        gfem.PML(*args, **kwargs)


@pytest.mark.parametrize("knob", ["target_reflection", "reference_frequency"])
def test_a_traced_knob_is_not_differentiable(knob):
    def f(v):
        gfem.PML("pml", **{knob: v})
        return v

    with pytest.raises(gfem.NonDifferentiableError, match=f"{knob} is a fixed value"):
        jax.grad(f)(0.5)


# --- model checks ------------------------------------------------------------------


def test_pml_takes_one_layer_value(tmp_path):
    mesh = _box(tmp_path)
    with pytest.raises(TypeError, match=r"pml takes one gfem.PML .* got list"):
        _model(mesh, pml=[gfem.PML("pml")])  # ty: ignore[invalid-argument-type]


def test_a_layer_region_missing_from_the_mesh_is_not_found(tmp_path):
    with pytest.raises(gfem.RegionNotFoundError, match=r"nope"):
        _model(_box(tmp_path), pml=gfem.PML(("pml", "nope")))


def test_a_layer_region_must_be_a_volume(tmp_path):
    with pytest.raises(
        ValueError, match=r"perfectly matched layer.*'outer' is a surface"
    ):
        _model(_box(tmp_path), pml=gfem.PML(("pml", "outer")))


def test_a_layer_region_excluded_from_electromagnetic_solves_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"perfectly matched layer 'pml'.*excluded"):
        _model(_box(tmp_path), materials={"air": AIR}, excluded=("pml",))


def test_a_layer_region_without_a_material_names_its_background(tmp_path):
    with pytest.raises(ValueError, match=r"'pml'.*takes its background material"):
        _model(_box(tmp_path), materials={"air": AIR})


@pytest.mark.parametrize(
    "background",
    [
        gfem.Material(
            eps_r=gfem.PermittivityLaw(
                eps_inf=1.0, terms=[gfem.Drude(plasma_frequency=1e15, damping=1e13)]
            )
        ),
        gfem.MaterialField.uniform("pml", AIR),
        gfem.Material(
            eps_r=(2.0, 3.0, 4.0),
            axes=((0.6, 0.8, 0.0), (-0.8, 0.6, 0.0), (0.0, 0.0, 1.0)),
        ),
    ],
    ids=["law", "material-field", "rotated"],
)
def test_a_layer_background_must_be_constant_and_axis_aligned(tmp_path, background):
    with pytest.raises(NotImplementedError, match=r"perfectly matched layer 'pml'"):
        _model(_box(tmp_path), materials={"air": AIR, "pml": background})


def test_an_axis_aligned_anisotropic_background_is_accepted(tmp_path):
    background = gfem.Material(
        eps_r=(2.0, 3.0, 4.0), axes=((0, 1, 0), (1, 0, 0), (0, 0, -1))
    )
    _model(_box(tmp_path), materials={"air": AIR, "pml": background})


def _one_interior_cell(c):
    if ((c > 0) & (c < 1)).all():
        return "pml" if (c < 0.5).all() else "air"
    return "pml"


def test_a_layer_cell_inside_the_interior_box_is_refused(tmp_path):
    mesh = _box(tmp_path, volume=_one_interior_cell)
    with pytest.raises(ValueError, match=r"'pml'.*inside the interior box"):
        _model(mesh)


def test_a_layer_beyond_no_face_stretches_along_no_axis(tmp_path):
    mesh = _box(tmp_path, sides=(), volume=_one_interior_cell)
    with pytest.raises(ValueError, match=r"perfectly matched layer.*no axis"):
        _model(mesh)


def test_a_layer_that_straddles_a_face_of_the_interior_box_is_refused(tmp_path):
    def shear(p):
        return p + np.outer(p[:, 1], (0.2, 0.0, 0.0))

    mesh = _box(tmp_path, mapping=shear)
    with pytest.raises(NotImplementedError, match=r"'pml'.*box shell"):
        _model(mesh)


def test_two_layer_regions_ending_on_two_planes_of_one_side_are_refused(tmp_path):
    def volume(c):
        if c[0] < 1:
            return "air"
        if c[1] < 0.5:
            return "pml_a"
        return "pml_b" if c[0] < 1.5 else None

    mesh = _box(tmp_path, thickness=1.0, volume=volume)
    materials = {"air": AIR, "pml_a": AIR, "pml_b": AIR}
    with pytest.raises(
        ValueError, match=r"\+x.*1 um thick \('pml_a'\).*0\.5 um \('pml_b'\)"
    ):
        _model(mesh, pml=gfem.PML(("pml_a", "pml_b")), materials=materials)


def test_one_layer_region_with_a_stepped_outer_face_is_refused(tmp_path):
    def volume(c):
        if c[0] < 1:
            return "air"
        return None if c[0] > 1.5 and c[1] > 0.5 else "pml"

    mesh = _box(tmp_path, thickness=1.0, volume=volume)
    with pytest.raises(ValueError, match=r"perfectly matched layer.*\+x.*one plane"):
        _model(mesh)


def test_absorbing_on_a_face_of_the_layer_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match=r"Absorbing on 'outer'.*perfectly matched layer.*PEC"
    ):
        _model(_box(tmp_path), boundaries=[gfem.Absorbing("outer")])


def test_absorbing_on_a_face_away_from_the_layer_is_accepted(tmp_path):
    def surface(c):
        return "top" if abs(c[2] - 1) < 1e-9 and c[0] < 1 else ""

    model = _model(_box(tmp_path, surface=surface), boundaries=[gfem.Absorbing("top")])
    assert model.boundaries == (gfem.Absorbing("top"),)


@pytest.mark.parametrize(
    "sheet",
    [gfem.Impedance("sheet", resistance=50.0), gfem.Conductor("sheet", 5.8e7)],
    ids=["impedance", "conductor"],
)
def test_a_sheet_in_the_layer_is_refused(tmp_path, sheet):
    # a sheet between two layer regions: a plane inside one region is no face
    def volume(c):
        if c[0] < 1:
            return "air"
        return "pml" if c[0] < 1.25 else "pml_out"

    def surface(c):
        return "sheet" if abs(c[0] - 1.25) < 1e-9 else ""

    mesh = _box(tmp_path, cells=4, volume=volume, surface=surface)
    with pytest.raises(NotImplementedError, match=r"'sheet'.*perfectly matched layer"):
        _model(
            mesh,
            pml=gfem.PML(("pml", "pml_out")),
            materials={"air": AIR, "pml": AIR, "pml_out": AIR},
            boundaries=[sheet],
        )


def test_a_lumped_port_touching_the_layer_is_refused(tmp_path):
    def surface(c):
        return "lp" if abs(c[0] - 1) < 1e-9 else ""

    mesh = _box(tmp_path, surface=surface)
    with pytest.raises(
        NotImplementedError, match=r"LumpedPort 'p'.*perfectly matched layer"
    ):
        _model(mesh, ports=[gfem.LumpedPort("p", "lp", "+z")])


def test_a_wave_port_normal_along_a_stretch_axis_is_refused(tmp_path):
    def surface(c):
        return "end" if abs(c[0] - 1.5) < 1e-9 else ""

    mesh = _box(tmp_path, surface=surface)
    with pytest.raises(
        NotImplementedError, match=r"WavePort 'w'.*perfectly matched layer.*x"
    ):
        _model(mesh, ports=[gfem.WavePort("w", "end")])


def test_a_wave_port_across_side_layers_is_accepted(tmp_path):
    def surface(c):
        return "end" if abs(c[0]) < 1e-9 else ""

    mesh = _box(tmp_path, sides=("-y", "+y"), surface=surface)
    model = _model(mesh, ports=[gfem.WavePort("w", "end")])
    assert model.port_names == ("w",)


def _cell(tmp_path, top_layer):
    def volume(c):
        if c[2] < 0:
            return "pml"
        return "pml" if top_layer and c[2] > 1 else "air"

    edges = (
        np.linspace(0, 1, 3),
        np.linspace(0, 1, 3),
        np.linspace(-0.5, 1.5 if top_layer else 1.0, 5 if top_layer else 4),
    )
    path = sm.write_msh22(tmp_path / "cell.msh", sm.periodic_cell(edges, volume))
    return gfem.Mesh.from_gmsh(path)


def _floquet(mesh):
    return _model(
        mesh,
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=[gfem.FloquetPort("f", "z_max", polarization="TE")],
    )


def test_a_floquet_face_touching_the_layer_is_refused(tmp_path):
    with pytest.raises(
        NotImplementedError, match=r"FloquetPort 'f'.*perfectly matched layer"
    ):
        _floquet(_cell(tmp_path, top_layer=True))


def test_a_layer_opposite_a_floquet_port_is_accepted(tmp_path):
    model = _floquet(_cell(tmp_path, top_layer=False))
    assert "-z: d = 0.5 um (pml)" in repr(model)


def test_a_stretch_along_a_periodic_axis_is_refused(tmp_path):
    def volume(c):
        return "air" if c[0] < 1 else "pml"

    edges = (np.linspace(0, 1.5, 4), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    path = sm.write_msh22(tmp_path / "cell.msh", sm.periodic_cell(edges, volume))
    with pytest.raises(
        NotImplementedError, match=r"perfectly matched layer.*periodic.*x"
    ):
        _model(
            gfem.Mesh.from_gmsh(path), boundaries=[gfem.Periodic(("x_min", "x_max"))]
        )


def test_a_layer_spanning_the_period_is_not_stretched_along_it(tmp_path):
    model = _model(
        _cell(tmp_path, top_layer=False),
        boundaries=[gfem.Periodic(("x_min", "x_max"))],
    )
    text = repr(model)
    assert "-z: d = 0.5 um (pml)" in text
    assert "x: d" not in text


# --- derived geometry --------------------------------------------------------------


def test_each_side_reports_its_thickness_and_regions(tmp_path):
    def volume(c):
        if ((c > 0) & (c < 1)).all():
            return "air"
        return "pml_low" if c[2] < 0.5 else "pml_high"

    mesh = _box(tmp_path, sides=("+x", "-y", "-z"), volume=volume)
    materials = {"air": AIR, "pml_low": AIR, "pml_high": AIR}
    model = _model(mesh, pml=gfem.PML(("pml_low", "pml_high")), materials=materials)
    lines = repr(model).splitlines()
    assert "  pml: PML(('pml_low', 'pml_high'), target_reflection=1e-06)" in lines
    assert "    +x: d = 0.5 um (pml_low, pml_high)" in lines
    assert "    -y: d = 0.5 um (pml_low, pml_high)" in lines
    assert "    -z: d = 0.5 um (pml_low)" in lines
    assert not any(line.lstrip().startswith(("-x", "+y", "+z")) for line in lines)


def test_the_repr_shows_a_pinned_reference_frequency(tmp_path):
    pml = gfem.PML("pml", target_reflection=1e-3, reference_frequency=2e14)
    text = repr(_model(_box(tmp_path), pml=pml))
    assert (
        "pml: PML(('pml',), target_reflection=0.001, reference_frequency=2e+14)" in text
    )


def test_an_excluded_conductor_crossing_a_side_layer_keeps_the_box(tmp_path):
    def volume(c):
        if 0.5 < c[1] < 1 and c[2] < 0.5:
            return "metal"
        return "air" if c[0] < 1 else "pml"

    mesh = _box(tmp_path, volume=volume)
    model = _model(mesh, excluded=("metal",))
    assert "+x: d = 0.5 um (pml)" in repr(model)
    with pytest.raises(ValueError, match=r"inside the interior box"):
        _model(mesh, materials={"air": AIR, "pml": AIR, "metal": AIR})


# --- derived models, pytree and codec ----------------------------------------------


def test_replace_adds_and_removes_the_layer(tmp_path):
    model = _model(_box(tmp_path))
    plain = model.replace(pml=None)
    assert plain.pml is None
    assert "  pml: PML" not in repr(plain)
    back = plain.replace(pml=gfem.PML("pml", target_reflection=1e-4))
    assert back.pml == gfem.PML("pml", target_reflection=1e-4)
    assert model.with_material("pml", gfem.Material(eps_r=2.0)).pml == model.pml


def test_with_material_checks_the_new_background(tmp_path):
    model = _model(_box(tmp_path))
    with pytest.raises(NotImplementedError, match=r"perfectly matched layer 'pml'"):
        model.with_material("pml", gfem.MaterialField.uniform("pml", AIR))


def test_the_layer_is_static_pytree_data(tmp_path):
    model = _model(_box(tmp_path))
    leaves, tree = jax.tree_util.tree_flatten(model)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.pml == model.pml
    assert repr(back) == repr(model)


def test_a_layer_round_trips_through_a_job_file(tmp_path):
    pml = gfem.PML(["pml"], target_reflection=1e-4, reference_frequency=3e14)
    model = _model(_box(tmp_path), pml=pml)
    path = gfem.Job(model, gfem.Driven(frequency=2e14)).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path).model
    assert back.pml == pml
    assert repr(back) == repr(model)


def test_a_model_without_a_layer_writes_no_layer_entry(tmp_path):
    model = gfem.Model(_box(tmp_path), materials={"air": AIR, "pml": AIR})
    assert model.pml is None
    path = gfem.Job(model, gfem.Driven(frequency=2e14)).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    assert set(doc["job"]["model"]) == {
        "$type",
        "mesh",
        "materials",
        "boundaries",
        "ports",
        "excluded",
    }
    assert '"PML"' not in path.read_text()
    assert gfem.Job.from_json(path).model.pml is None


@pytest.mark.bitwise
def test_a_model_without_a_layer_is_unchanged(tmp_path):
    mesh = gfem.Mesh.from_gmsh(LINE, length_unit="mm")
    z_line = 376.73 / 6
    model = gfem.Model(
        mesh,
        materials={"feed": AIR, "design": gfem.Material(eps_r=10.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
            gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
        ],
    )
    explicit = gfem.Model(
        mesh,
        materials=dict(model.materials),
        boundaries=model.boundaries,
        ports=model.ports,
        pml=None,
    )
    problem = gfem.Driven(frequency=(2e9, 5e9), excite="p1")
    options = gfem.SolverOptions(linear_solver="direct", threads=1)
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    loaded = gfem.Job.from_json(path).model
    s = [
        np.asarray(gfem.solve(m, problem, options=options).s["p2", "p1"])
        for m in (model, explicit, loaded)
    ]
    np.testing.assert_array_equal(s[0], s[1])
    np.testing.assert_array_equal(s[0], s[2])
    assert repr(explicit) == repr(model)


# --- profile -----------------------------------------------------------------------


def test_the_stretch_maximum_meets_the_target_reflection_at_normal_incidence():
    # |R| = exp(-2 k int_0^d sigma dxi), sigma = sigma_max (xi / d)^2
    k, d = 2 * math.pi / 1.5e-6, 0.4e-6
    for target in (1e-2, 1e-6, 1e-9):
        smax = _pml.sigma_max(target, k, d)
        integral = smax * d / 3
        assert math.exp(-2 * k * integral) == pytest.approx(target, rel=1e-12)
    # -3 ln(1e-6) / (2 * 1 * 1) = 20.723265836946411
    assert _pml.sigma_max(1e-6, 1.0, 1.0) == pytest.approx(20.723265836946411)


def test_the_reflection_falls_with_frequency_and_rises_with_obliquity():
    assert _pml.reflection(1e-6, 1e14, 2e14, 1.0) == pytest.approx(1e-3, rel=1e-12)
    assert _pml.reflection(1e-6, 2e14, 2e14, 1.0) == pytest.approx(1e-6, rel=1e-12)
    assert _pml.reflection(1e-6, 2e14, 2e14, 0.5) == pytest.approx(1e-3, rel=1e-12)
    assert _pml.reflection(1e-2, 3e14, 2e14, 1.0) == pytest.approx(1e-3, rel=1e-12)


def _ngsolve_box():
    from netgen.occ import Box, OCCGeometry
    from ngsolve import Mesh

    geo = OCCGeometry(Box((-0.5, -0.5, -0.5), (1.5, 1.5, 1.5)))
    return Mesh(geo.GenerateMesh(maxh=1.0))


def test_the_pml_tensor_is_the_product_of_the_axis_stretches(tmp_path):
    model = _model(_box(tmp_path, sides=("+x", "-y")))
    layers = model._layers
    k_ref, target = 2 * math.pi / 1e-6, 1e-4
    lam, lam_inv = pml_tensor(layers, k_ref, target, scale=1e-6)
    mesh = _ngsolve_box()
    # sigma_max = -3 ln(1e-4) / (2 k_ref d), d = 0.5 um: k_ref d = pi
    smax = -3 * math.log(1e-4) / (2 * math.pi)
    for point, (xi_x, xi_y) in [
        ((0.5, 0.5, 0.5), (0.0, 0.0)),
        ((1.25, 0.5, 0.5), (0.25, 0.0)),
        ((1.4, -0.2, 0.3), (0.4, 0.2)),
        ((0.3, -0.5, 1.2), (0.0, 0.5)),
    ]:
        sx = 1 - 1j * smax * (xi_x / 0.5) ** 2
        sy = 1 - 1j * smax * (xi_y / 0.5) ** 2
        expected = np.diag([sy / sx, sx / sy, sx * sy])
        mp = mesh(*point)
        got = np.array(lam(mp)).reshape(3, 3)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
        got_inv = np.array(lam_inv(mp)).reshape(3, 3)
        np.testing.assert_allclose(got_inv @ got, np.eye(3), atol=1e-12)


def test_no_layer_warning_fires_while_building_a_layered_model(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _model(_box(tmp_path))


# --- driven solves -----------------------------------------------------------------

DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
ORDER_3 = gfem.SolverOptions(order=3, linear_solver="direct")


def _line(tmp_path, pml, *, air=AIR, name="line.msh", cells=6, **kw):
    """The TEM line of `pml_cases` closed by a layer, h = 1 / ``cells`` um."""
    mesh = gfem.Mesh.from_gmsh(pml_cases.tem_line(tmp_path / name, cells=cells))
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": AIR},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=pml,
        **kw,
    )


def _driven(model, frequencies=(pml_cases.F0,), options=DIRECT, **kw):
    problem = gfem.Driven(tuple(frequencies), keep_fields=True, **kw)
    return gfem.solve(model, problem, options=options)


def test_the_layer_absorbs_what_the_port_does_not_get_back(tmp_path):
    # order 3 at h = lambda / 10: the port's 1 W counts its section's mode,
    # whose propagation constant differs from the 3D mesh's by the
    # discretization error, which caps the balance at 4.6e-6 at order 2
    # (measured 3.9e-9 here)
    model = _line(tmp_path, gfem.PML("pml", target_reflection=1e-2), cells=10)
    frequencies = (0.8 * pml_cases.F0, pml_cases.F0)
    result = _driven(model, frequencies, options=ORDER_3)
    for k, f in enumerate(frequencies):
        s11 = complex(np.asarray(result.s.values)[k, 0, 0])
        field = result.e_field(f)
        absorbed = -float(gfem.power_flow(field, "pml", model=model))
        assert absorbed == pytest.approx(1 - abs(s11) ** 2, rel=1e-8)
        # and what crosses any plane of the air reaches the layer
        crossing = -float(gfem.power_flow(field, _beyond(0.7), model=model))
        assert crossing == pytest.approx(absorbed, rel=1e-12)


def _beyond(z):
    return lambda c: c[:, 2] > z * 1e-6


def test_a_side_cutting_the_layer_reads_the_field_s_own_stretch(tmp_path):
    # inside the layer the power reads the stretched pieces, so the power
    # out of a side and out of its complement cancel only with the stretch
    # the field was solved with: the sweep's, normalized at its lowest
    # frequency, not one at the field's own frequency
    model = _line(tmp_path, gfem.PML("pml", target_reflection=1e-2))
    f = 1.3 * pml_cases.F0
    field = _driven(model, (pml_cases.F0, f)).e_field(f)
    deep = float(gfem.power_flow(field, _beyond(1.75), model=model))
    rest = float(gfem.power_flow(field, lambda c: c[:, 2] <= 1.75e-6, model=model))
    assert deep < 0
    assert rest == pytest.approx(-deep, rel=1e-10)


def _static_box(tmp_path, pml):
    """The unit cube with a +x layer, between a "top" (z = 1) and a "bottom" plate."""

    def plates(c):
        if abs(c[2]) < 1e-9:
            return "bottom"
        return "top" if abs(c[2] - 1) < 1e-9 else ""

    mesh = _box(tmp_path, ("+x",), surface=plates)
    background = gfem.Material(eps_r=2.0, conductivity=3.0)
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(eps_r=4.0, conductivity=5.0),
            "pml": background,
        },
        pml=pml,
    )


@pytest.mark.bitwise
def test_static_solves_see_the_background(tmp_path):
    layered = _static_box(tmp_path, gfem.PML("pml"))
    plain = _static_box(tmp_path, None)
    options = gfem.SolverOptions(linear_solver="direct", threads=1)
    for problem, read in [
        (gfem.Electrostatic(terminals={"t": "top"}, ground="bottom"), "capacitance"),
        (gfem.DCConduction(terminals={"t": "top"}, ground="bottom"), "conductance"),
    ]:
        got, want = (
            np.asarray(getattr(gfem.solve(m, problem, options=options), read))
            for m in (layered, plain)
        )
        np.testing.assert_array_equal(got, want)
        assert got.all()


def test_changing_the_target_reflection_changes_the_operator(tmp_path):
    from gfem._engine.driven import driven_operator
    from gfem._solve_cache import solve_key

    # the air as a material field: the layer lies in the operator's kept part
    mesh = gfem.Mesh.from_gmsh(pml_cases.tem_line(tmp_path / "line.msh", cells=6))
    air = gfem.MaterialField.uniform("air", AIR)
    models = [
        gfem.Model(
            mesh,
            materials={"air": air, "pml": AIR},
            boundaries=[gfem.PEC("pec")],
            ports=[gfem.WavePort("p1", "port")],
            pml=gfem.PML("pml", target_reflection=r0),
        )
        for r0 in (1e-2, 1e-3)
    ]
    problem = gfem.Driven((pml_cases.F0,))
    # on one mesh, one after the other: the kept part of the first stretch
    # must not serve the second
    s = [abs(complex(np.asarray(_driven(m).s.values)[0, 0, 0])) for m in models]
    fresh = _line(tmp_path, gfem.PML("pml", target_reflection=1e-3), name="b.msh")
    assert s[1] == pytest.approx(
        abs(complex(np.asarray(_driven(fresh).s.values)[0, 0, 0]))
    )
    assert s[0] != pytest.approx(s[1], rel=1e-2)
    keys = {solve_key(m, problem, DIRECT)[0] for m in models}
    assert len(keys) == 2
    digests = set()
    for m in models:
        stretch = _pml.stretch(m, [pml_cases.F0])
        op = driven_operator(m, 2, stretch=stretch)
        digests.add(tuple(p.digest for p in op.fixed.split(op.pieces)[1] if p))
    assert len(digests) == 2


def test_an_explicit_bddc_gmres_refuses_the_layer(tmp_path):
    model = _line(tmp_path, gfem.PML("pml"))
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    with pytest.raises(ValueError, match=r"bddc_gmres.*perfectly matched layer.*'pml'"):
        _driven(model, options=options)


def test_auto_picks_direct_for_a_layer_whatever_the_memory(
    tmp_path, monkeypatch, caplog
):
    from gfem._engine import driven
    from gfem._engine.driven import DrivenRule

    model = _line(tmp_path, gfem.PML("pml"))
    # no budget at all: without a layer, auto would take bddc_gmres
    monkeypatch.setattr(driven, "default_retain_memory", lambda: 0)
    caplog.set_level(logging.INFO, logger="gfem.solver")
    rule = DrivenRule(model, gfem.Driven((pml_cases.F0,)), gfem.SolverOptions())
    assert rule.linear.kind == "direct"
    (record,) = [r for r in caplog.records if hasattr(r, "linear_solver")]
    assert record.linear_solver == "direct"
    assert "perfectly matched layer" in record.getMessage()


def test_a_thin_layer_warns(tmp_path):
    model = _line(tmp_path, gfem.PML("pml"))
    # d = 0.5 um: k_ref d = 2 pi / 100 at lambda_ref = 50 um, thin above
    thin = pml_cases.C0 / 51e-6
    with pytest.warns(gfem.ThinLayerWarning, match=r"\+z side \(pml\)") as record:
        _driven(model, (thin,))
    assert Path(record[0].filename) == Path(__file__)
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.ThinLayerWarning)
        _driven(model, (pml_cases.C0 / 49e-6,))


# --- readers ------------------------------------------------------------------------


def test_electric_energy_reads_the_physical_regions_only(tmp_path):
    # the layer's tetrahedra hold the stretched field with the background
    # permittivity: no energy, so no key, and the device's own unchanged
    model = _line(tmp_path, gfem.PML("pml", target_reflection=1e-2))
    field = _driven(model).e_field(pml_cases.F0)
    got = gfem.electric_energy(field, model=model)
    whole = gfem.electric_energy(field, model=model.replace(pml=None))
    assert set(whole) == {"air", "pml"}
    assert got == {"air": pytest.approx(whole["air"], rel=1e-14)}


def _split_box(tmp_path, materials=None):
    """The +x box with air and layer cut at z = 1/2 by "mid_air" and "mid_pml"."""

    def volume(c):
        half = "low" if c[2] < 0.5 else "high"
        return f"{half}_air" if ((c > 0) & (c < 1)).all() else f"{half}_pml"

    def surface(c):
        if abs(c[2] - 0.5) < 1e-9 and 0 < c[1] < 1:
            return "mid_air" if c[0] < 1 else "mid_pml"
        return ""

    mesh = _box(tmp_path, ("+x",), volume=volume, surface=surface)
    if materials is None:
        crystal = gfem.Material(eps_r=(2.0, 3.0, 4.0))
        materials = {
            "low_air": gfem.Material(eps_r=2.0),
            "high_air": AIR,
            "low_pml": crystal,
            "high_pml": AIR,
        }
    return gfem.Model(mesh, materials=materials, pml=gfem.PML(("low_pml", "high_pml")))


def _modes(model):
    problem = gfem.Eigenmode(n_modes=2, target=1.0e14)
    return gfem.solve(model, problem, options=DIRECT)


def test_a_mode_s_participations_count_the_physical_regions_only(tmp_path):
    model = _split_box(tmp_path)
    modes = _modes(model)
    assert modes.pml == gfem.PML(("low_pml", "high_pml"))
    assert set(modes.bulk_participation) == {"low_air", "high_air"}
    # the anisotropic background of the layer gets no principal split
    assert modes.principal_participation == {}
    total = sum(np.asarray(p) for p in modes.bulk_participation.values())
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)
    # 1 J over the physical regions: the energy reader agrees
    for k, field in enumerate(modes.e_fields):
        energy = gfem.electric_energy(field, model=model)
        assert set(energy) == {"low_air", "high_air"}
        assert 2 * float(sum(energy.values())) == pytest.approx(1.0, rel=1e-10), k


def test_an_eigenmode_result_carries_its_layer_through_a_result_file(tmp_path):
    model = _split_box(tmp_path)
    problem = gfem.Eigenmode(n_modes=1, target=1.0e14)
    gfem.Job(model, problem, DIRECT).run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert loaded.pml == model.pml
    assert set(loaded.bulk_participation) == {"low_air", "high_air"}


def test_a_port_mode_carries_its_layer_through_a_result_file(tmp_path):
    model = _line(tmp_path, gfem.PML("pml"))
    problem = gfem.Driven((pml_cases.F0,))
    result = gfem.Job(model, problem, DIRECT).run(out=tmp_path / "out")
    assert result.port_modes["p1"].pml == model.pml
    assert gfem.load_result(tmp_path / "out").port_modes["p1"].pml == model.pml
    alone = gfem.solve(model, gfem.PortMode("p1", (pml_cases.F0,)), options=DIRECT)
    assert alone.pml == model.pml


def test_a_result_without_a_layer_writes_no_layer_entry(tmp_path):
    from gfem import _codec

    result = gfem.EigenmodeResult(omega=np.array([1e10 + 1e6j]), error=np.array([0.0]))
    context = _codec.Context(tmp_path)
    doc = _codec.Encoder(context).encode(result)
    assert "pml" not in json.dumps(doc)


def test_interface_participation_drops_the_faces_in_the_layer(tmp_path):
    modes = _modes(_split_box(tmp_path))
    both = gfem.interface_participation(modes, ["mid_air", "mid_pml"], 1e-9, 4.0)
    air = gfem.interface_participation(modes, "mid_air", 1e-9, 4.0)
    assert np.all(np.asarray(air.participation) > 0)
    np.testing.assert_allclose(both.participation, air.participation, rtol=1e-14)


def test_interface_participation_refuses_a_surface_all_in_the_layer(tmp_path):
    modes = _modes(_split_box(tmp_path))
    with pytest.raises(
        ValueError,
        match=r"'mid_pml'.*every face.*perfectly matched layer \('low_pml', "
        r"'high_pml'\).*physical regions",
    ):
        gfem.interface_participation(modes, "mid_pml", 1e-9, 4.0)


def _refused(subject):
    return pytest.raises(
        gfem.MeshRefinementUnsupportedError,
        match=rf"^{subject} has no path for a model with a perfectly matched "
        r"layer \(regions '.*'\): its indicators weigh the layer with its "
        r"background material.*gfem.Absorbing",
    )


@pytest.mark.parametrize(
    ("estimator", "subject"),
    [
        (gfem.FieldEstimator(), "gfem.FieldEstimator"),
        (gfem.QuantityEstimator(s=("p1", "p1")), "gfem.QuantityEstimator"),
        (
            gfem.QuantityEstimator(objective=lambda r, _: abs(r.s["p1", "p1"][0])),
            "gfem.QuantityEstimator",
        ),
    ],
)
def test_driven_error_estimates_refuse_the_layer(tmp_path, estimator, subject):
    model = _line(tmp_path, gfem.PML("pml"))
    result = _driven(model)
    with _refused(subject) as caught:
        gfem.error_estimate(result, estimator, model=model)
    assert caught.value.feature == "perfectly matched layer"


@pytest.mark.parametrize(
    ("estimator", "subject"),
    [
        (gfem.FieldEstimator(), "gfem.FieldEstimator"),
        (gfem.QuantityEstimator(omega=0), "gfem.QuantityEstimator"),
    ],
)
def test_eigenmode_error_estimates_refuse_the_layer(tmp_path, estimator, subject):
    model = _split_box(tmp_path)
    with _refused(subject):
        gfem.error_estimate(_modes(model), estimator, model=model)


@pytest.mark.parametrize(
    "problem",
    [gfem.Driven((pml_cases.F0,)), gfem.Eigenmode(n_modes=1, target=1.0e14)],
    ids=["driven", "eigenmode"],
)
def test_refine_mesh_refuses_the_layer_before_any_solve(tmp_path, monkeypatch, problem):
    from gfem import refinement

    def no_solve(*_, **__):
        raise AssertionError("solved")

    monkeypatch.setattr(refinement, "solve", no_solve, raising=False)
    model = _line(tmp_path, gfem.PML("pml"))
    name = type(problem).__name__
    with pytest.raises(
        gfem.MeshRefinementUnsupportedError,
        match=rf"^gfem.refine_mesh has no path for a gfem.{name} problem on a model "
        r"with a perfectly matched layer \(regions 'pml'\)",
    ):
        gfem.refine_mesh(model, problem)


def test_electrostatic_refinement_runs_on_a_layered_model(tmp_path):
    # the electrostatic solve sees the background, and so does its estimator
    model = _static_box(tmp_path, gfem.PML("pml"))
    problem = gfem.Electrostatic(terminals={"t": "top"}, ground="bottom")
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    # between the plates the potential is linear: met on the first mesh
    refined = gfem.refine_mesh(
        model, problem, tolerance=1e-4, max_cycles=1, options=options
    )
    assert refined.stop == "tolerance"
    assert refined.model.pml == model.pml
    result = gfem.solve(model, problem, options=options)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert np.isfinite(estimate.value)


# --- wave ports through side layers --------------------------------------------------


def _strip(tmp_path, clearance, name="strip.msh", pml="default", **kw):
    path = pml_cases.strip_guide(tmp_path / name, clearance, **kw)
    return pml_cases.strip_model(path, pml)


def _n_eff(result):
    k0 = 2 * math.pi * pml_cases.STRIP_F / pml_cases.C0
    return complex(np.asarray(result.gamma)[0]).imag / k0


def test_a_port_section_inside_side_layers_returns_the_guided_mode(tmp_path):
    # the full section at clearance 0.4 um: the layer modes crowd the shift
    # and pass the propagating filter, so only the layer share tells the
    # guided mode from them; its tail in the layer warns
    model = _strip(tmp_path, 0.4, quarter=False, length=0.15)
    problem = gfem.PortMode("p1", (pml_cases.STRIP_F,))
    with pytest.warns(gfem.LayerClearanceWarning, match=r"'p1'.*clearance"):
        result = gfem.solve(model, problem, options=DIRECT)
    assert _n_eff(result) == pytest.approx(pml_cases.STRIP_NEFF, abs=1e-3)


def test_a_wave_port_oblique_in_a_layer_is_refused(tmp_path):
    # the end face x = 0 tilted to x = 0.3 z: where it cuts the +z layer,
    # the layer is no longer a box shell around the interior box
    def surface(c):
        return "end" if abs(c[0]) < 1e-9 else ""

    def tilt(p):
        return p + np.outer(0.3 * p[:, 2], (1.0, 0.0, 0.0))

    mesh = _box(tmp_path, sides=("+z",), surface=surface, mapping=tilt)
    with pytest.raises(NotImplementedError, match=r"perfectly matched layer 'pml'"):
        _model(mesh, ports=[gfem.WavePort("w", "end")])


def _quarter_strip(tmp_path, pml="default", name="q.msh"):
    return _strip(tmp_path, 0.4, name=name, pml=pml, length=0.15)


@pytest.mark.parametrize("which", ["voltage_path", "current_path"])
def test_an_impedance_path_through_the_layer_is_refused(tmp_path, which):
    model = _quarter_strip(tmp_path)
    with pytest.warns(gfem.LayerClearanceWarning):
        mode = gfem.solve(
            model, gfem.PortMode("p1", (pml_cases.STRIP_F,)), options=DIRECT
        )
    # from the core's centre across the cladding into the +x layer (m)
    path = [(0.0, 0.0, 0.0), (1.2e-6, 0.0, 0.0)]
    with pytest.raises(
        ValueError,
        match=rf"characteristic_impedance: {which} on port 'p1' passes through "
        r"the perfectly matched layer \('pml'\)",
    ):
        gfem.characteristic_impedance(mode, **{which: path})
    # a path in the physical part stays accepted
    inside = [(0.0, 0.0, 0.0), (0.0, 0.5e-6, 0.0)]
    assert np.isfinite(gfem.characteristic_impedance(mode, voltage_path=inside)).all()


def test_the_port_section_key_follows_the_stretch(tmp_path):
    from gfem._port_section import port_section_key

    keys = set()
    for name, pml in (
        ("a", gfem.PML("pml")),
        ("b", gfem.PML("pml", target_reflection=1e-3)),
        ("c", gfem.PML("pml", reference_frequency=2 * pml_cases.STRIP_F)),
    ):
        model = _quarter_strip(tmp_path, pml, name=f"{name}.msh")
        stretch = _pml.stretch(model, [pml_cases.STRIP_F])
        keys.add(port_section_key(model, model.ports[0], stretch))
    assert len(keys) == 3
    # and the lowest solve frequency, when nothing is pinned
    model = _quarter_strip(tmp_path, name="d.msh")
    port = model.ports[0]
    keys = {
        port_section_key(model, port, _pml.stretch(model, [f]))
        for f in (pml_cases.STRIP_F, 0.9 * pml_cases.STRIP_F)
    }
    assert len(keys) == 2


# --- gradients next to a layer ----------------------------------------------------


def _moved(model, node, dz):
    """``model`` with node ``node`` moved by ``dz`` metres along z."""
    mesh = model.mesh
    x = np.asarray(mesh.nodes).copy()
    x[node, 2] += dz
    moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
    return model.replace(mesh=moved)


def _inner_face_node(model):
    """A node on the layer's inner face (z = 1.5 um) off the line's walls."""
    x = np.asarray(model.mesh.imported_nodes) * 1e6
    on = np.flatnonzero(np.abs(x[:, 2] - pml_cases.AIR_LENGTH) < 1e-9)
    return int(on[np.argmin(np.abs(x[on, 0] - 0.1) + np.abs(x[on, 1] - 0.1))])


def test_a_moved_mesh_pushing_air_past_a_stretched_face_is_refused(tmp_path):
    model = _line(tmp_path, gfem.PML("pml"), cells=10)
    moved = _moved(model, _inner_face_node(model), 0.02e-6)
    match = (
        r"^The moved mesh puts \d+ tetrahedra of region 'air' past the inner face "
        r"\+z \(z = 1\.5e-06 m\) of the perfectly matched layer"
    )
    with pytest.raises(ValueError, match=match):
        _pml.check_layer_nodes(moved)
    with pytest.raises(ValueError, match=match):
        _driven(moved)


def test_a_moved_mesh_pushing_air_past_a_stretched_face_is_refused_under_grad(
    tmp_path,
):
    model = _line(tmp_path, gfem.PML("pml"), cells=10)
    node = _inner_face_node(model)
    mesh = model.mesh

    def objective(dz):
        x = jnp.asarray(mesh.nodes).at[node, 2].add(dz)
        moved = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mesh), [x])
        result = gfem.solve(model.replace(mesh=moved), gfem.Driven(pml_cases.F0))
        return jnp.sum(jnp.abs(result.s.values) ** 2)

    with pytest.raises(
        jax.errors.JaxRuntimeError, match=r"ValueError.*past the inner face \+z"
    ):
        jax.grad(objective)(0.02e-6)


def test_a_layer_node_moved_inward_is_accepted(tmp_path):
    # the layer's background has no stretch below its inner face
    model = _line(tmp_path, gfem.PML("pml"), cells=10)
    _pml.check_layer_nodes(_moved(model, _inner_face_node(model), -0.02e-6))


def _walled_box(tmp_path):
    """The +x layer box with a "core" below x = 0.5 and named faces."""

    def volume(c):
        if c[0] > 1:
            return "pml"
        return "core" if c[0] < 0.5 else "air"

    def surface(c):
        if abs(c[0] - 0.5) < 1e-9:
            return "wall"
        if abs(c[0] - 1.5) < 1e-9:
            return "far"
        return ""

    mesh = _box(tmp_path, cells=4, volume=volume, surface=surface)
    return _model(mesh, materials={"air": AIR, "core": AIR, "pml": AIR})


def test_boundary_morph_holds_the_layer(tmp_path):
    from gfem.shape import boundary_morph

    model = _walled_box(tmp_path)
    morph = boundary_morph(model, "wall", grows="core")
    nodes = np.asarray(morph(0.05e-6 * np.ones(morph.n_params)).nodes)
    x0 = np.asarray(model.mesh.imported_nodes)
    layer = np.unique(
        model.mesh.tetrahedra[_pml.layer_tetrahedra(model.mesh, model.pml)]
    )
    moved = np.any(nodes != x0, axis=1)
    assert moved.any()
    assert not moved[layer].any()


def test_boundary_morph_of_a_surface_in_the_layer_is_refused(tmp_path):
    from gfem.shape import boundary_morph

    model = _walled_box(tmp_path)
    with pytest.raises(ValueError, match="or lies in the perfectly matched layer"):
        boundary_morph(model, "far")
