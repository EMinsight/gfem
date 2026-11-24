"""The section's shape kernel against differences of the assembled forms.

Each form is re-assembled by `PortModeSpace` on a section whose points moved
by a random in-plane displacement; the kernel's cotangent contracted with the
displacement must equal the derivative (fourth-order central difference). The
checks cover every form kind of the pencil's list, the power and coupling
forms, the voltage functional and the chain to the 3D nodes.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pml_cases
import pytest

import gfem
from gfem import _pml
from gfem._boundaries import Conductor
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_sensitivity import FormPair
from gfem._engine.port_mode_shape import (
    ShapeForm,
    chain_to_nodes,
    voltage_eulerian,
)
from gfem._port_section import port_section
from gfem._sweep.port_mode import _voltage_row, _voltage_rule

DATA = Path(__file__).parent / "data"
ORDER = 2
K0 = 0.35  # per local unit
LAW = gfem.PermittivityLaw(
    eps_inf=1.5,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=12e9, damping=1e8)],
)


def guide(mesh_name, unit, materials, boundaries, ports=None):
    mesh = gfem.Mesh.from_gmsh(DATA / mesh_name, length_unit=unit)
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries,
        ports=ports or [gfem.WavePort("p1", "port1")],
    )
    return model, port_section(model, model.ports[0])


def lossy_slab(**kwargs):
    return guide(
        "slab_waveguide.msh",
        "mm",
        {"slab": gfem.Material(eps_r=4.0, tan_delta=0.03), "air": gfem.Material()},
        [gfem.PEC("walls")],
        **kwargs,
    )


def vectors(space, seed=0):
    n = len(space.free)
    rng = np.random.default_rng(seed)

    def cplx(m):
        return rng.standard_normal(m) + 1j * rng.standard_normal(m)

    return cplx(n), cplx(n), complex(rng.standard_normal() + 0.3j), cplx


def displacement(section, seed=1):
    return np.random.default_rng(seed).standard_normal(section.points.shape)


def moved(section, w, h):
    return dataclasses.replace(section, points=section.points + h * w)


def central(value, section, w, h=5e-4):
    """Fourth-order central difference of ``value(section)`` along ``w``."""
    f = [value(moved(section, w, t * h)) for t in (2, 1, -1, -2)]
    return (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h)


def pencil_value(section, q, x, lam):
    a, b = PortModeSpace(section, ORDER).pencil.at(K0)
    return (q @ (a @ x) - lam * (q @ (b @ x))).real


def check_pencil(section, rel=1e-8):
    space = PortModeSpace(section, ORDER)
    q, x, lam, _ = vectors(space)
    w = displacement(section)
    shape = ShapeForm(space)
    shape.add_pencil(q, x, lam, space.pencil.theta(K0))
    got = float(np.sum(shape.result() * w))
    fd = central(lambda s: pencil_value(s, q, x, lam), section, w)
    assert got == pytest.approx(fd, rel=rel)


def test_a_lossy_slab_pencil_matches_the_assembled_forms():
    _, section = lossy_slab()
    check_pencil(section)


def test_a_wr90_pencil_matches_the_assembled_forms():
    _, section = guide(
        "waveguide.msh", "mm", {"air": gfem.Material(eps_r=2.0)}, [gfem.PEC("walls")]
    )
    check_pencil(section)


def test_an_absorbing_edge_pencil_matches_the_assembled_forms():
    _, section = guide(
        "waveguide.msh",
        "mm",
        {"air": gfem.Material(eps_r=2.0)},
        [gfem.Absorbing("walls")],
    )
    assert len(section.absorbing_edges)
    check_pencil(section)


def test_a_conductor_edge_pencil_matches_the_assembled_forms():
    _, section = guide(
        "cpw_section.msh",
        "um",
        {"si": gfem.Material(eps_r=11.9), "air": gfem.Material()},
        [gfem.PEC("walls"), Conductor("metal", conductivity=5.8e7)],
    )
    assert len(section.impedance_edges)
    check_pencil(section, rel=1e-8)


def test_a_law_region_pencil_matches_the_assembled_forms():
    _, section = guide(
        "slab_waveguide.msh",
        "mm",
        {"slab": gfem.Material(eps_r=LAW), "air": gfem.Material()},
        [gfem.PEC("walls")],
    )
    assert len(section.laws)
    check_pencil(section)


def test_a_pml_cut_strip_includes_the_position_term(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, quarter=False, length=0.15
    )
    strip = pml_cases.strip_model(path)
    stretch = _pml.stretch(strip, [pml_cases.STRIP_F])
    section = port_section(strip, strip.ports[0], stretch)
    assert section.layer is not None
    space = PortModeSpace(section, ORDER)
    q, x, lam, _ = vectors(space)
    w = displacement(section, 3)
    on_layer = np.unique(section.triangles[section.layer])
    inside = np.zeros(len(section.points), bool)
    inside[on_layer] = True

    def value(sec):
        return pencil_value(sec, q, x, lam)

    # nodes inside the layer alone, where the coefficient is a function of position
    w_layer = w * inside[:, None]
    shape = ShapeForm(space)
    shape.add_pencil(q, x, lam, space.pencil.theta(K0))
    cot = shape.result()
    for direction in (w, w_layer):
        got = float(np.sum(cot * direction))
        fd = central(value, section, direction, h=1e-4)
        assert got == pytest.approx(fd, rel=1e-8)


@pytest.mark.parametrize("form", ["power_g", "power_t", "coupling_g", "coupling_t"])
def test_the_normalization_and_coupling_forms_match_the_assembled_forms(form):
    _, section = lossy_slab()
    space = PortModeSpace(section, ORDER)
    _, x, _, cplx = vectors(space, 4)
    coupling = form.startswith("coupling")
    left = cplx(space.n_t if coupling else len(space.free))
    weight = 0.7 - 0.4j
    w = displacement(section, 5)
    shape = ShapeForm(space)
    shape.add_pair(FormPair(form, left, x, weight))

    def value(sec):
        sp = PortModeSpace(sec, ORDER)
        g, t = sp.coupling_forms if coupling else sp.power_forms
        f = g if form.endswith("_g") else t
        return (weight * (left @ (f @ x))).real

    got = float(np.sum(shape.result() * w))
    assert got == pytest.approx(central(value, section, w), rel=1e-8)


def test_the_gauge_forms_carry_no_node_derivative():
    _, section = lossy_slab()
    space = PortModeSpace(section, ORDER)
    _, x, _, _ = vectors(space)
    shape = ShapeForm(space)
    shape.add_pair(FormPair("gauge_n", None, x, 1.0))
    assert not shape.result().any()


def test_a_pml_cut_strip_coupling_forms_include_the_position_term(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, quarter=False, length=0.15
    )
    strip = pml_cases.strip_model(path)
    section = port_section(
        strip, strip.ports[0], _pml.stretch(strip, [pml_cases.STRIP_F])
    )
    space = PortModeSpace(section, ORDER)
    _, x, _, cplx = vectors(space, 6)
    w = displacement(section, 7)
    for form in ("coupling_t", "power_g"):
        coupling = form.startswith("coupling")
        left = cplx(space.n_t if coupling else len(space.free))
        shape = ShapeForm(space)
        shape.add_pair(FormPair(form, left, x, 0.6 + 0.2j))

        def value(sec, form=form, left=left, coupling=coupling):
            sp = PortModeSpace(sec, ORDER)
            g, t = sp.coupling_forms if coupling else sp.power_forms
            f = g if form.endswith("_g") else t
            return ((0.6 + 0.2j) * (left @ (f @ x))).real

        got = float(np.sum(shape.result() * w))
        assert got == pytest.approx(central(value, section, w, h=1e-4), rel=1e-7), form


def test_the_voltage_functional_is_differentiated_at_fixed_points():
    model, section = guide(
        "slab_waveguide.msh",
        "mm",
        {"slab": gfem.Material(eps_r=4.0, tan_delta=0.03), "air": gfem.Material()},
        [gfem.PEC("walls")],
        ports=[
            gfem.WavePort(
                "p1", "port1", voltage_path=((4e-3, 0, 0), (4e-3, 10.16e-3, 0))
            )
        ],
    )
    space = PortModeSpace(section, ORDER)
    path = np.asarray(model.ports[0].voltage_path, float)
    pts, wts = _voltage_rule(section, path, "p1")
    _, x, _, _ = vectors(space, 8)
    bar = 0.5 - 1.5j
    w = displacement(section, 9)
    full = space.full(x)
    got = float(np.sum(voltage_eulerian(space, full, pts, wts, bar) * w))

    def value(sec):
        sp = PortModeSpace(sec, ORDER)
        return (bar * (_voltage_row(sp, (pts, wts)) @ x)).real

    assert got == pytest.approx(central(value, section, w), rel=1e-8)


def test_the_cotangent_carries_the_rigid_motion_invariants():
    _, section = lossy_slab()
    space = PortModeSpace(section, ORDER)
    q, x, lam, _ = vectors(space)
    shape = ShapeForm(space)
    shape.add_pencil(q, x, lam, space.pencil.theta(K0))
    cot = shape.result()
    scale = np.abs(cot).max()
    assert np.abs(cot.sum(axis=0)).max() <= 1e-12 * scale * len(cot)
    p = section.points
    moment = np.sum(p[:, 0] * cot[:, 1] - p[:, 1] * cot[:, 0])
    assert abs(moment) <= 1e-12 * scale * np.abs(p).max() * len(cot)


def test_the_chain_puts_the_points_cotangent_on_the_face_nodes():
    model, section = lossy_slab()
    mesh = model.mesh
    rng = np.random.default_rng(2)
    cot = rng.standard_normal(section.points.shape)
    out = chain_to_nodes(section, mesh.triangles, cot, len(mesh.nodes))
    used = np.unique(mesh.triangles[section.faces])
    assert out.shape == (len(mesh.nodes), 3)
    assert not np.delete(out, used, axis=0).any()
    # a node move X -> X + d changes the points by d F2^T / s (frame held)
    d = rng.standard_normal((len(used), 3))
    d3 = np.zeros((len(mesh.nodes), 3))
    d3[used] = d
    dp = d @ section.frame[:2].T / section.scale
    assert np.sum(out * d3) == pytest.approx(np.sum(cot * dp), rel=1e-12)
