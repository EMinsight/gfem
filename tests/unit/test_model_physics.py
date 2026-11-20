"""Per-physics solve domains, thermal boundaries and their model checks.

`Model(excluded=...)` takes a tuple (electromagnetic problems only) or a
mapping with the keys "em", "thermal" and "dc_conduction".
Thermal boundaries are checked against the thermal domain.

Fixture (see test_model.py): "sub" holds a dielectric entity and a conductor
entity whose shell is "metal_xy" + "metal_z"; "air" sits on top; "air__sub"
is their interface; "sub__None" and "air__None" are the exterior.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from unit.test_cross_section_mesh import write_section

import gfem
from gfem import _codec
from gfem.errors import NonDifferentiableError

DATA = Path(__file__).parent / "data"
SI = gfem.Material(eps_r=11.7, name="si", thermal_conductivity=148.0)
AIR = gfem.Material(name="air", thermal_conductivity=0.026)
METAL = gfem.enclosed_by("metal_xy", "metal_z")


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "two_box_msh22.msh")


def _model(mesh, **kwargs):
    kwargs.setdefault("materials", {"sub": SI, "air": AIR})
    return gfem.Model(mesh, **kwargs)


# --- the excluded= mapping ------------------------------------------------------


def test_a_mapping_excludes_per_physics(mesh):
    model = _model(mesh, excluded={"em": (METAL,), "thermal": ["air"]})
    assert model.excluded == (METAL,)
    assert model.excluded_from("em") == (METAL,)
    assert model.excluded_from("thermal") == ("air",)
    assert model.excluded_from("dc_conduction") == ()


def test_a_tuple_excludes_for_electromagnetic_problems_only(mesh):
    model = _model(mesh, excluded=(METAL,))
    assert model.excluded_from("em") == (METAL,)
    assert model.excluded_from("thermal") == ()
    assert model.excluded_from("dc_conduction") == ()
    same = _model(mesh, excluded={"em": (METAL,)})
    assert same.excluded == model.excluded


def test_an_unknown_physics_key_lists_the_known_keys(mesh):
    with pytest.raises(ValueError, match=r"'heat'.*'em', 'thermal', 'dc_conduction'"):
        _model(mesh, excluded={"heat": ("air",)})
    with pytest.raises(ValueError, match="'optics'"):
        _model(mesh).excluded_from("optics")


def test_an_em_excluded_heater_may_carry_a_material(mesh):
    model = _model(mesh, excluded=("sub",))
    assert model.materials["sub"] is SI


def test_a_material_on_a_region_excluded_everywhere_raises(mesh):
    everywhere = {"em": ("sub",), "thermal": ("sub",), "dc_conduction": ("sub",)}
    with pytest.raises(ValueError, match="'sub' is excluded from every"):
        _model(mesh, excluded=everywhere)
    model = _model(mesh, excluded=everywhere, materials={"air": AIR})
    assert "sub" not in model.materials


def test_a_thermal_only_region_needs_no_material(mesh):
    model = _model(mesh, excluded=("sub",), materials={"air": AIR})
    assert "sub" not in model.materials


def test_every_volume_excluded_raises_only_when_every_physics_is_empty(mesh):
    _model(mesh, excluded=("sub", "air"), materials={})
    with pytest.raises(ValueError, match="nothing left"):
        _model(
            mesh,
            materials={},
            excluded=dict.fromkeys(("em", "thermal", "dc_conduction"), ("sub", "air")),
        )


def test_an_electromagnetic_solve_on_an_empty_domain_raises(mesh):
    model = _model(mesh, excluded=("sub", "air"), materials={})
    with pytest.raises(ValueError, match="electromagnetic"):
        gfem.solve(model, gfem.Electrostatic(terminals={"t": "metal_xy"}))


def test_replace_keeps_and_changes_the_mapping(mesh):
    model = _model(mesh, excluded={"em": (METAL,), "thermal": ("air",)})
    kept = model.replace(materials={"sub": SI, "air": AIR})
    assert kept.excluded_from("thermal") == ("air",)
    assert kept.excluded_from("em") == (METAL,)
    em_only = model.replace(excluded=("sub",))
    assert em_only.excluded_from("thermal") == ()


def test_the_pytree_keeps_the_domains(mesh):
    model = _model(mesh, excluded={"em": (METAL,), "thermal": ("air",)})
    back = jax.tree.map(lambda x: x, model)
    assert back.excluded_from("thermal") == ("air",)
    assert back.default_insulated == model.default_insulated


# --- thermal boundaries ---------------------------------------------------------


def test_thermal_conditions_hold_their_values():
    fixed = gfem.FixedTemperature("bottom", temperature=300)
    conv = gfem.Convection("top", coefficient=10, ambient=293.15)
    flux = gfem.HeatFlux("strip", flux=-5)
    assert (fixed.temperature, conv.coefficient, conv.ambient, flux.flux) == (
        300.0,
        10.0,
        293.15,
        -5.0,
    )
    with pytest.raises(TypeError):
        gfem.FixedTemperature("bottom", 300)  # ty: ignore[missing-argument, too-many-positional-arguments]


@pytest.mark.parametrize(
    "build",
    [
        lambda t: gfem.FixedTemperature("s", temperature=t),
        lambda t: gfem.Convection("s", coefficient=10.0, ambient=t),
    ],
)
@pytest.mark.parametrize("value", [0.0, -20.0])
def test_a_temperature_at_or_below_zero_kelvin_raises(build, value):
    with pytest.raises(ValueError, match=r"absolute.*kelvin.*293\.15"):
        build(value)


def test_thermal_values_are_checked():
    with pytest.raises(ValueError, match="coefficient must be > 0"):
        gfem.Convection("s", coefficient=0.0, ambient=300.0)
    with pytest.raises(ValueError, match="finite"):
        gfem.HeatFlux("s", flux=float("inf"))
    with pytest.raises(TypeError, match="one surface name"):
        gfem.HeatFlux(("a", "b"), flux=1.0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "build",
    [
        lambda v: gfem.FixedTemperature("s", temperature=v),
        lambda v: gfem.Convection("s", coefficient=v, ambient=300.0),
        lambda v: gfem.Convection("s", coefficient=10.0, ambient=v),
        lambda v: gfem.HeatFlux("s", flux=v),
    ],
)
def test_a_traced_thermal_value_raises(build):
    def f(v):
        build(v)
        return v

    with pytest.raises(NonDifferentiableError, match="fixed value"):
        jax.grad(f)(300.0)


def test_a_thermal_condition_on_a_thermal_only_surface_is_accepted(mesh):
    model = _model(
        mesh,
        excluded=("sub",),
        boundaries=[gfem.FixedTemperature("metal_xy", temperature=350.0)],
    )
    assert model.boundaries == (gfem.FixedTemperature("metal_xy", temperature=350.0),)


def test_a_thermal_condition_off_the_thermal_domain_names_it(mesh):
    with pytest.raises(ValueError, match=r"thermal solve domain.*\"thermal\""):
        _model(
            mesh,
            excluded={"thermal": ("sub",)},
            boundaries=[gfem.FixedTemperature("metal_xy", temperature=350.0)],
        )


def test_an_em_and_a_thermal_condition_share_a_surface(mesh):
    model = _model(
        mesh,
        boundaries=[
            gfem.PEC("sub__None"),
            gfem.FixedTemperature("sub__None", temperature=300.0),
        ],
    )
    assert model.default_pmc == ("air__None",)
    assert model.default_insulated == ("air__None",)
    with pytest.raises(ValueError, match="more than one thermal"):
        _model(
            mesh,
            boundaries=[
                gfem.HeatFlux("sub__None", flux=1.0),
                gfem.FixedTemperature("sub__None", temperature=300.0),
            ],
        )


def test_convection_on_an_interior_surface_raises(mesh):
    with pytest.raises(ValueError, match=r"Convection on 'air__sub'.*interior"):
        _model(
            mesh,
            boundaries=[gfem.Convection("air__sub", coefficient=10.0, ambient=300.0)],
        )


def test_fixed_temperature_and_flux_may_sit_inside(mesh):
    _model(
        mesh,
        boundaries=[
            gfem.FixedTemperature("air__sub", temperature=300.0),
            gfem.HeatFlux("metal_xy", flux=1e3),
        ],
    )


def test_touching_unequal_fixed_temperatures_raise(mesh):
    def build(t):
        return _model(
            mesh,
            boundaries=[
                gfem.FixedTemperature("metal_xy", temperature=300.0),
                gfem.FixedTemperature("metal_z", temperature=t),
            ],
        )

    build(300.0)
    with pytest.raises(ValueError, match="merge them, or separate them"):
        build(310.0)


def test_default_insulated_on_a_slab_lists_the_end_faces(tmp_path):
    slab = gfem.Mesh.from_gmsh_cross_section(write_section(tmp_path / "s.msh"))
    model = gfem.Model(
        slab,
        materials={"lower": SI, "upper": AIR},
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=10.0, ambient=300.0),
        ],
    )
    assert "front" in model.default_insulated
    assert "back" in model.default_insulated
    assert "bottom" not in model.default_insulated
    assert "top" not in model.default_insulated
    assert "interface" not in model.default_insulated


def test_repr_lines(mesh):
    model = _model(
        mesh,
        excluded={"em": (METAL,), "thermal": ("air",)},
        boundaries=[gfem.FixedTemperature("sub__None", temperature=300.0)],
    )
    text = repr(model)
    assert "excluded (electromagnetic): sub entity" in text
    assert "excluded (thermal): air entity" in text
    assert "excluded (DC conduction)" not in text
    assert "insulated by default (thermal): air__sub" in text
    assert "insulated" not in repr(_model(mesh))


def test_em_problems_ignore_thermal_conditions_in_the_pmc_default(mesh):
    model = _model(
        mesh, boundaries=[gfem.Convection("air__None", coefficient=5.0, ambient=300.0)]
    )
    assert model.default_pmc == ("sub__None", "air__None")


# --- codec ----------------------------------------------------------------------

PROBLEM = gfem.Electrostatic(terminals={"t": "metal_xy"})


def _job_doc(model, path):
    gfem.Job(model, PROBLEM).to_json(path)
    return json.loads(path.read_text())


def test_the_mapping_round_trips(mesh, tmp_path):
    model = _model(
        mesh,
        excluded={"em": (METAL,), "thermal": ("air",)},
        boundaries=[
            gfem.FixedTemperature("sub__None", temperature=300.0),
            gfem.Convection("air__sub", coefficient=10.0, ambient=293.15),
            gfem.HeatFlux("metal_z", flux=-2.0),
        ],
    )
    doc = _job_doc(model, tmp_path / "job.json")
    assert doc["job"]["model"]["excluded"]["$type"] == "Excluded"
    back = gfem.Job.from_json(tmp_path / "job.json").model
    assert back.excluded_from("em") == (METAL,)
    assert back.excluded_from("thermal") == ("air",)
    assert back.boundaries == model.boundaries


def test_an_em_only_job_is_written_as_before(mesh, tmp_path):
    tuple_doc = (tmp_path / "a.json", _model(mesh, excluded=(METAL,)))
    gfem.Job(tuple_doc[1], PROBLEM).to_json(tuple_doc[0])
    gfem.Job(_model(mesh, excluded={"em": [METAL]}), PROBLEM).to_json(
        tmp_path / "b.json"
    )
    a = tuple_doc[0].read_text()
    assert a == (tmp_path / "b.json").read_text()
    assert json.loads(a)["job"]["model"]["excluded"] == [
        {"$type": "EnclosedBy", "surfaces": ["metal_xy", "metal_z"]}
    ]


def test_an_older_reader_refuses_the_mapping_by_name(mesh, tmp_path, monkeypatch):
    model = _model(mesh, excluded={"thermal": ("air",)})
    doc = _job_doc(model, tmp_path / "job.json")
    monkeypatch.delitem(_codec._BY_TAG, "Excluded")
    decoder = _codec.Decoder(_codec.Context(tmp_path), {})
    with pytest.raises(ValueError, match=r"type 'Excluded'.*upgrade gfem"):
        decoder.decode(doc["job"]["model"]["excluded"])


# --- engine spaces per physics --------------------------------------------------


def test_engine_regions_and_spaces_follow_the_physics(mesh):
    import ngsolve as ngs

    from gfem._engine.domain import domain_key, em_region, h1_space, space_key
    from gfem._engine.mesh import to_ngsolve

    model = _model(mesh, excluded={"em": (METAL,), "thermal": ("air",)})
    built = to_ngsolve(mesh)

    def volume(physics):
        return ngs.Integrate(1, built.mesh, definedon=em_region(model, built, physics))

    # file units are um: 2 x 2 x 2 box of "sub" (1 high) and "air" (1 high)
    assert volume("em") == pytest.approx(7.6, rel=1e-12)
    assert volume("thermal") == pytest.approx(4.0, rel=1e-12)
    assert volume("dc_conduction") == pytest.approx(8.0, rel=1e-12)
    assert domain_key(model) != domain_key(model, "thermal")
    assert space_key(model, "h1", 1, fixed=()) != space_key(
        model, "h1", 1, fixed=(), physics="thermal"
    )
    thermal = h1_space(model, built, 1, physics="thermal")
    sub_nodes = np.unique(mesh.tetrahedra[mesh.elements("sub")])
    assert thermal.ndof == len(sub_nodes)
