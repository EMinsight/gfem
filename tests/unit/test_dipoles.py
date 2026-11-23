"""Point dipoles: their columns, emitted power and position refusals.

Coarse boxes (order 2, h = lambda/4) in a perfectly matched layer: the
emitted power is within a few percent of the bulk power there; the
closed-form gates at 3 % live in tests/verification/test_dipoles.py.
"""

import free_space_cases as fsc
import numpy as np
import pml_cases
import pytest

import gfem

ORDER2 = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return gfem.Mesh.from_gmsh(
        fsc.dipole_box(tmp_path_factory.mktemp("dipole") / "box.msh")
    )


def _vacuum(mesh, **kwargs):
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "pml": gfem.Material()},
        pml=gfem.PML("pml"),
        **kwargs,
    )


#: A generic point inside the cube, off every face of the grid, m.
INSIDE = np.array([0.47, 0.53, 0.56]) * fsc.UM


def test_an_electric_dipole_in_vacuum_emits_its_bulk_power(box):
    p = np.array([0.0, 0.0, 1e-30])
    dipole = gfem.ElectricDipole("d", position=INSIDE, moment=p)
    result = gfem.solve(
        _vacuum(box), gfem.Driven(fsc.F0, sources=dipole), options=ORDER2
    )
    (power,) = np.asarray(result.emitted_power["d"])
    assert power / fsc.bulk_power(fsc.OMEGA0, p) == pytest.approx(1.0, abs=0.08)


# --- where a dipole may sit ---------------------------------------------------------


def _split_volume(c):
    if not ((c > 0) & (c < 1)).all():
        return "pml"
    return "air" if c[0] < 0.5 else "right"


def _split_surface(c):
    eps = 1e-9
    if abs(c[0] - 0.5) < eps and ((c > 0) & (c < 1)).all():
        # a pad off the layer (a lumped port keeps clear of it), its ring
        return "mid" if ((c[1:] > 0.25) & (c[1:] < 0.75)).all() else "ring"
    return ""


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    """The box with its cube split at x = 0.5 um into "air" and "right"."""
    path = fsc.dipole_box(
        tmp_path_factory.mktemp("split") / "split.msh",
        cells=4,
        volume=_split_volume,
        surface=_split_surface,
    )
    return gfem.Mesh.from_gmsh(path)


def _split_model(mesh, right=None, **kwargs):
    materials = {"air": gfem.Material(), "pml": gfem.Material()}
    materials["right"] = gfem.Material() if right is None else right
    return gfem.Model(mesh, materials=materials, pml=gfem.PML("pml"), **kwargs)


def _at(*point, cls=gfem.ElectricDipole):
    return cls("d", position=np.array(point) * fsc.UM, moment=(0, 0, 1))


def _solve(model, dipole):
    return gfem.solve(model, gfem.Driven(fsc.F0, sources=dipole), options=ORDER2)


LOSSY_LAW = gfem.PermittivityLaw(
    terms=[gfem.Drude(plasma_frequency=1e14, damping=1e13)]
)
MID = (0.5, 0.53, 0.56)
RIGHT = (0.73, 0.53, 0.56)


@pytest.mark.parametrize(
    ("right", "point", "match"),
    [
        (gfem.Material(eps_r=2.0), MID, "interface between regions 'air', 'right'"),
        (gfem.Material(tan_delta=1e-2), RIGHT, "region 'right', which has a loss"),
        (gfem.Material(conductivity=1.0), RIGHT, "region 'right', which has a loss"),
        (
            gfem.Material(
                eps_r=gfem.PermittivityLaw(
                    terms=[gfem.Drude(plasma_frequency=1e14, damping=1e13)]
                )
            ),
            RIGHT,
            "a lossy permittivity law",
        ),
        (gfem.Material(eps_r=(1.0, 2.0, 3.0)), RIGHT, "anisotropic"),
        (None, (-0.2, 0.53, 0.56), "perfectly matched layer region 'pml'"),
        (None, (3.0, 0.5, 0.5), r"cannot be placed: Point .* outside the mesh"),
    ],
)
def test_a_dipole_refuses_a_position_without_a_bulk_power(split, right, point, match):
    with pytest.raises(ValueError, match=f"ElectricDipole 'd'.*{match}"):
        _solve(_split_model(split, right), _at(*point))


def test_a_dipole_refuses_a_material_field(split):
    model = _split_model(split)
    field = gfem.MaterialField.uniform("right", gfem.Material(eps_r=2.0))
    with pytest.raises(ValueError, match="'right', whose material is a MaterialField"):
        _solve(model.with_material("right", field), _at(*RIGHT))


def test_a_dipole_refuses_an_excluded_volume(split):
    model = _split_model(split, excluded=["right"])
    with pytest.raises(ValueError, match="not in an excluded or conductor volume"):
        _solve(model, _at(*RIGHT))


@pytest.mark.parametrize(
    ("boundary", "match"),
    [
        (gfem.PEC("mid"), "PEC face 'mid'"),
        (gfem.PMC("mid"), "PMC face 'mid'"),
        (gfem.Impedance("mid", resistance=50.0), "Impedance face 'mid'"),
    ],
)
def test_a_dipole_refuses_a_boundary_or_sheet_face(split, boundary, match):
    model = _split_model(split, boundaries=[boundary])
    with pytest.raises(ValueError, match=f"lies on the {match}.*Move it off the face"):
        _solve(model, _at(*MID))


def test_a_dipole_refuses_a_port_face(split):
    port = gfem.LumpedPort("p1", "mid", "+z", resistance=50.0)
    model = _split_model(split, ports=[port])
    with pytest.raises(ValueError, match="lumped port face 'p1'"):
        _solve(model, _at(*MID))


def test_a_dipole_refuses_the_boundary_of_the_solve_domain(split):
    # the right half excluded: x = 0.5 bounds the solve domain
    model = _split_model(split, excluded=["right"])
    with pytest.raises(ValueError, match="boundary of the solve domain"):
        _solve(model, _at(*MID))


def test_a_dipole_on_a_face_inside_one_material_is_accepted(split):
    # "mid" separates two regions of equal materials and carries no condition
    result = _solve(_split_model(split), _at(*MID))
    assert np.asarray(result.emitted_power["d"])[0] > 0


def test_a_real_permittivity_law_is_accepted(split):
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=3 * fsc.F0, damping=0)],
    )
    result = _solve(_split_model(split, gfem.Material(eps_r=law)), _at(*RIGHT))
    assert np.asarray(result.emitted_power["d"])[0] > 0


def test_a_magnetic_dipole_refuses_like_an_electric_one(split):
    with pytest.raises(ValueError, match=r"MagneticDipole 'd'.*'pml'"):
        _solve(_split_model(split), _at(-0.2, 0.5, 0.5, cls=gfem.MagneticDipole))


# --- beside ports: port waves and the power split -----------------------------------


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    """The layer's TEM line (wave port at z = 0) with a lossy slab in its air."""
    path = pml_cases.slab_line(tmp_path_factory.mktemp("line") / "line.msh")
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "pml": gfem.Material(),
            "slab": gfem.Material(eps_r=2.0, tan_delta=0.1),
        },
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


#: Along the line's E (y), between the port and the slab, off the grid's faces.
LINE_DIPOLE = gfem.ElectricDipole(
    "d", position=np.array([0.07, 0.13, 0.24]) * fsc.UM, moment=(0, 1e-28, 0)
)


def _line_solve(model, **kwargs):
    problem = gfem.Driven(fsc.F0, keep_fields=True, **kwargs)
    return gfem.solve(model, problem, options=ORDER2)


def test_a_port_and_a_dipole_on_one_factorization_equal_their_own_solves(line):
    both = _line_solve(line, sources=LINE_DIPOLE)
    port = _line_solve(line)
    dipole = _line_solve(line, excite=(), sources=LINE_DIPOLE)
    np.testing.assert_allclose(both.s.values, port.s.values, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(both.port_waves, dipole.port_waves, rtol=1e-10)
    np.testing.assert_allclose(
        both.emitted_power["d"], dipole.emitted_power["d"], rtol=1e-10
    )
    np.testing.assert_allclose(
        both.e_field(fsc.F0, "d").coefficients,
        dipole.e_field(fsc.F0).coefficients,
        rtol=1e-10,
        atol=1e-12 * np.abs(dipole.e_field(fsc.F0).coefficients).max(),
    )
    assert dipole.s.values.shape == (1, 1, 0)
    assert both.port_waves.shape == (1, 1, 1)


def test_the_emitted_power_splits_into_ports_layer_and_losses(line):
    # T14: the discrete balance holds to round-off
    result = _line_solve(line, excite=(), sources=LINE_DIPOLE)
    field = result.e_field(fsc.F0)
    emitted = float(result.emitted_power["d"][0])
    guided = float(np.abs(result.port_wave("p1", "d")[0]) ** 2)
    absorbed = -float(gfem.power_flow(field, "slab", model=line))
    radiated = -float(gfem.power_flow(field, "pml", model=line))
    assert min(guided, absorbed, radiated) > 1e-3 * emitted
    assert guided + absorbed + radiated == pytest.approx(emitted, rel=1e-8)


def test_power_flow_refuses_a_side_whose_element_layer_holds_a_dipole(line):
    # the slab's outside layer is z in [0.4, 0.5] um: the load enters the residual
    near = gfem.ElectricDipole(
        "d", position=np.array([0.07, 0.13, 0.45]) * fsc.UM, moment=(0, 1e-28, 0)
    )
    field = _line_solve(line, excite=(), sources=near).e_field(fsc.F0)
    with pytest.raises(ValueError, match="outside this side holds the dipole 'd'"):
        gfem.power_flow(field, "slab", model=line)
    # a side holding it whole counts its emission
    assert float(gfem.power_flow(field, "pml", model=line)) < 0


def test_the_adaptive_sweep_gives_the_dipole_outputs_of_the_uniform_one(line):
    band = tuple(np.linspace(0.9, 1.1, 9) * fsc.F0)

    def solve(sweep):
        problem = gfem.Driven(band, excite=(), sources=LINE_DIPOLE, sweep=sweep)
        return gfem.solve(line, problem, options=ORDER2)

    adaptive, uniform = solve("adaptive"), solve("uniform")
    np.testing.assert_allclose(
        adaptive.emitted_power["d"], uniform.emitted_power["d"], rtol=1e-5
    )
    np.testing.assert_allclose(adaptive.port_waves, uniform.port_waves, rtol=1e-5)


def test_a_port_free_dipole_solves_on_the_adaptive_sweep(box):
    band = tuple(np.linspace(0.95, 1.05, 5) * fsc.F0)
    dipole = gfem.ElectricDipole("d", position=INSIDE, moment=(0, 0, 1e-30))
    problem = gfem.Driven(band, sources=dipole, sweep="adaptive")
    result = gfem.solve(_vacuum(box), problem, options=ORDER2)
    assert result.port_waves is None
    ratio = np.asarray(result.emitted_power["d"]) / [
        fsc.bulk_power(2 * np.pi * f, np.array([0, 0, 1e-30])) for f in band
    ]
    np.testing.assert_allclose(ratio, 1.0, atol=0.08)
