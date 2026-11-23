"""On a 2D mesh, a feature with no 2D path raises `NotImplementedError` by name.

Each message names its remedy: a supported option, or the slab model through
`gfem.Mesh.from_gmsh_cross_section`. No 3D-only code runs on a 2D model.
"""

import re
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from strip2d import write_strip

import gfem

SLAB = "Mesh.from_gmsh_cross_section"
AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = write_strip(tmp_path_factory.mktemp("strip") / "strip.msh")
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def materials():
    return {name: gfem.Material(eps_r=2.0) for name in "abc"}


@pytest.fixture(scope="module")
def model(mesh):
    return gfem.Model(mesh, materials=materials(), boundaries=(gfem.PEC("outer"),))


def _raises(remedy: str, call) -> None:
    with pytest.raises(NotImplementedError, match="2D mesh") as info:
        call()
    assert re.search(remedy, str(info.value)), str(info.value)


# --- the 2D block of Model -----------------------------------------------------

MODEL_ITEMS = {
    "lumped port": (
        {"ports": (gfem.LumpedPort("p", "ab", "+x"),)},
        r"LumpedPort.*WavePort on a straight exterior line",
    ),
    "Floquet port": (
        {"ports": (gfem.FloquetPort("f", "outer", polarization="TE"),)},
        rf"FloquetPort.*gfem\.Periodic.*{SLAB}",
    ),
    "Periodic": (
        {"boundaries": (gfem.Periodic(("ab", "cb")),)},
        rf"Periodic.*1D grating.*{SLAB}",
    ),
    "conductor sheet": (
        {"boundaries": (gfem.Conductor("ab", 1e7),)},
        r"Conductor.*PEC.*area of its own",
    ),
    "impedance sheet": (
        {"boundaries": (gfem.Impedance("ab", resistance=50.0),)},
        r"Impedance.*PEC.*area of its own",
    ),
    "SheetField": (
        {"boundaries": (gfem.Conductor("ab", 1e7, fill=gfem.SheetField(jnp.ones(1))),)},
        rf"SheetField.*{SLAB}",
    ),
    "fixed temperature": (
        {"boundaries": (gfem.FixedTemperature("outer", temperature=300.0),)},
        rf"FixedTemperature.*{SLAB}",
    ),
    "convection": (
        {"boundaries": (gfem.Convection("outer", coefficient=10.0, ambient=300.0),)},
        rf"Convection.*{SLAB}",
    ),
    "heat flux": (
        {"boundaries": (gfem.HeatFlux("outer", flux=1.0),)},
        rf"HeatFlux.*{SLAB}",
    ),
    "enclosed_by": (
        {"excluded": (gfem.enclosed_by("ab"),)},
        r"enclosed_by.*excluded=\('name', \.\.\.\)",
    ),
    "voltage_path": (
        {"ports": (gfem.WavePort("p", "outer", voltage_path=((0, 0, 0), (1, 0, 0))),)},
        r"voltage_path.*leave it out",
    ),
    "anisotropic material": (
        {"materials": {"a": gfem.Material(eps_r=(2.0, 3.0, 4.0))}},
        rf"anisotropic.*'a'.*one value per property.*{SLAB}",
    ),
    "anisotropic material field": (
        {
            "materials": {
                "a": gfem.MaterialField(eps_r=jnp.ones((2, 3)), axes=AXES, region="a")
            }
        },
        rf"anisotropic.*'a'.*axes.*{SLAB}",
    ),
    "perfectly matched layer": (
        {"pml": gfem.PML("c")},
        rf"perfectly matched layer \(gfem\.PML on 'c'\).*gfem\.Absorbing.*{SLAB}",
    ),
}


@pytest.mark.parametrize("item", MODEL_ITEMS)
def test_a_model_on_a_2d_mesh_refuses_by_name(mesh, item):
    change, remedy = MODEL_ITEMS[item]
    mats = materials() | change.get("materials", {})
    kwargs = {k: v for k, v in change.items() if k != "materials"}
    _raises(remedy, lambda: gfem.Model(mesh, materials=mats, **kwargs))


def test_a_model_on_a_2d_mesh_builds_with_pec_and_named_exclusions(mesh):
    model = gfem.Model(
        mesh,
        materials=materials(),
        boundaries=(gfem.PEC("outer"),),
        excluded=("c",),
    )
    assert model.mesh.dim == 2
    assert model.excluded == ("c",)


# --- the solve entry -----------------------------------------------------------

F = 1e9

SOLVE_ITEMS = {
    "eigenmode": (lambda: gfem.Eigenmode(F), {}, rf"Eigenmode.*{SLAB}"),
    "electrostatic": (
        lambda: gfem.Electrostatic(["ab"]),
        {},
        rf"Electrostatic.*{SLAB}",
    ),
    "DC conduction": (
        lambda: gfem.DCConduction(["ab"]),
        {},
        rf"DCConduction.*{SLAB}",
    ),
    "steady thermal": (gfem.SteadyThermal, {}, rf"SteadyThermal.*{SLAB}"),
    "transient thermal": (
        lambda: gfem.TransientThermal(duration=1.0, steps=1),
        {},
        rf"TransientThermal.*{SLAB}",
    ),
    "wave_vector": (
        lambda: gfem.Driven(F, wave_vector=(1.0, 0.0, 0.0)),
        {},
        rf"wave_vector.*normal incidence.*{SLAB}",
    ),
    "adaptive sweep": (
        lambda: gfem.Driven(np.linspace(F, 2 * F, 30), sweep="adaptive"),
        {},
        r'adaptive sweep.*sweep="uniform"',
    ),
    "comm": (lambda: gfem.Driven(F), {"comm": object()}, r"one process"),
    "BDDC": (
        lambda: gfem.Driven(F),
        {"options": gfem.SolverOptions(linear_solver="bddc_gmres")},
        r"'bddc_gmres'.*linear_solver=\"direct\"",
    ),
    "GPU": (
        lambda: gfem.Driven(F),
        {"options": gfem.SolverOptions(device="cuda")},
        r"'cuda'.*device=\"cpu\"",
    ),
}


@pytest.mark.parametrize("item", SOLVE_ITEMS)
def test_a_solve_on_a_2d_mesh_refuses_by_name(model, item):
    problem, kwargs, remedy = SOLVE_ITEMS[item]
    _raises(remedy, lambda: gfem.solve(model, problem(), **kwargs))


def test_a_solve_on_moved_2d_nodes_refuses_by_name(mesh):
    moved = jax.tree_util.tree_map(lambda x: 1.01 * x, mesh)
    model = gfem.Model(moved, materials=materials(), boundaries=(gfem.PEC("outer"),))
    _raises(r"moved.*gmsh", lambda: gfem.solve(model, gfem.Driven(F)))


# --- measurements and other entries ----------------------------------------------


def field_on(mesh):
    return gfem.Field(
        coefficients=jnp.zeros(len(mesh.nodes), complex),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="h1",
        order=1,
        support=("a", "b", "c"),
        omega=2 * np.pi * F,
    )


def line_port_mode():
    """A port mode result whose field has a 2-component normal: a line port's."""
    return gfem.PortModeResult(
        port="p",
        mode=1,
        frequency=jnp.array([F]),
        gamma=jnp.array([1j]),
        Z_PV=jnp.array([np.nan]),
        residual=jnp.zeros(1),
        solved=jnp.array([True]),
        e_field=types.SimpleNamespace(normal=(1.0, 0.0)),
    )


ENTRIES = {
    "power_flow": (
        lambda model: gfem.power_flow(field_on(model.mesh), "a", model=model),
        r"power_flow.*S-parameters",
    ),
    "region_mean": (
        lambda model: gfem.region_mean(field_on(model.mesh), "a"),
        r"region_mean.*Field\.at",
    ),
    "element_mean": (
        lambda model: gfem.element_mean(field_on(model.mesh), "a"),
        r"element_mean.*Field\.at",
    ),
    "electric_energy": (
        lambda model: gfem.electric_energy(field_on(model.mesh), model=model),
        rf"electric_energy.*{SLAB}",
    ),
    "dissipated_power": (
        lambda model: gfem.dissipated_power(field_on(model.mesh), model=model),
        r"dissipated_power.*from_gmsh_cross_section.*thermal problem",
    ),
    "interface_participation": (
        lambda model: gfem.interface_participation(
            gfem.EigenmodeResult(
                omega=jnp.array([2j * np.pi * F]),
                error=jnp.zeros(1),
                e_fields=(field_on(model.mesh),),
                materials=model.materials,
            ),
            "ab",
            1e-9,
            4.0,
        ),
        rf"interface_participation.*{SLAB}",
    ),
    "characteristic_impedance": (
        lambda _model: gfem.characteristic_impedance(
            line_port_mode(), voltage_path=((0, 0, 0), (1, 0, 0))
        ),
        r"characteristic_impedance.*1 W/m.*no reference impedance",
    ),
    "refine_mesh": (
        lambda model: gfem.refine_mesh(model, gfem.Driven(F)),
        r"refine_mesh.*gmsh",
    ),
    "error_estimate": (
        lambda model: gfem.error_estimate(None, gfem.FieldEstimator(), model=model),
        r"error_estimate.*gmsh",
    ),
    "boundary_morph": (
        lambda model: gfem.shape.boundary_morph(model, "ab"),
        r"boundary_morph.*gmsh",
    ),
    "quality": (lambda model: gfem.shape.quality(model.mesh), r"quality.*gmsh"),
    "helmholtz_filter": (
        lambda model: gfem.design.helmholtz_filter(model.mesh, "a", radius=1e-6),
        r"helmholtz_filter.*LayerGrid",
    ),
}


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_entry_on_a_2d_mesh_refuses_by_name(model, entry):
    call, remedy = ENTRIES[entry]
    _raises(remedy, lambda: call(model))


# --- backstops behind the solve entry --------------------------------------------


def test_a_planar_port_mode_space_refuses_a_section_that_is_not_planar():
    from gfem._engine.port_mode import PortModeSpace

    with pytest.raises(TypeError, match="planar PortSection"):
        PortModeSpace(types.SimpleNamespace(), 1)


def test_a_driven_solve_on_a_2d_mesh_needs_a_polarization(mesh):
    model = gfem.Model(
        mesh, materials=materials(), ports=(gfem.WavePort("p", "outer"),)
    )
    with pytest.raises(ValueError, match=r'Driven\(polarization="Ez"\).*"Hz"'):
        gfem.solve(model, gfem.Driven(F))
