"""The port mode pencil is assembled from one list of form items.

The matrices must not move when the assembly is reorganized: each case below
compares every pencil matrix to an independent assembly written out slot by
slot (`_reference`), exactly, on sections that exercise every piece of the
pencil.
"""

import dataclasses
from pathlib import Path

import ngsolve as ngs
import numpy as np
import pml_cases
import pytest
import scipy.constants
import scipy.sparse as sp

import gfem
from gfem import _pml
from gfem._boundaries import Conductor
from gfem._engine import quadrature
from gfem._engine.port_mode import (
    ConductorScalar,
    FormItem,
    LawEdgeScalar,
    LawScalar,
    PortModeSpace,
    _admittance_groups,
    _edge_fill,
    _p0,
    law_slope,
)
from gfem._port_section import port_section

DATA = Path(__file__).parent / "data"
LAW = gfem.PermittivityLaw(
    eps_inf=1.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=12e9, damping=1e8)],
)


@pytest.fixture(scope="module")
def plate_mesh():
    return gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def slab_mesh():
    return gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")


def _guide(mesh, wall, material=None):
    return gfem.Model(
        mesh,
        materials={"air": material or gfem.Material()},
        boundaries=[wall],
        ports=[gfem.WavePort("p1", "port1")],
    )


def _plate(mesh, wall):
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0, tan_delta=0.01)},
        boundaries=[wall],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def _reference(space: PortModeSpace):  # noqa: PLR0915
    """The pencil's matrices, assembled slot by slot."""
    c, section, order = space._coef, space.section, space.order
    (et, en), (vt, vn) = space.space.TnT()
    q = quadrature.degree(order)
    dx = ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})

    def ds(name):
        return ngs.ds(name, intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)})

    sigma_scale = scipy.constants.mu_0 * scipy.constants.c * section.scale
    wt, wv = et + ngs.grad(en), vt + ngs.grad(vn)
    has_abc = len(_admittance_groups(section)) > 0
    abc_t = abc_n = None
    if has_abc:
        abc_t = c["Y_tau"] * et.Trace() * vt.Trace() * ds("absorbing.*")
        abc_n = c["Y_n"] * en.Trace() * vn.Trace() * ds("absorbing.*")
    forms_a = [
        [c["nu_nn"] * ngs.curl(et) * ngs.curl(vt) * dx],
        [(c["eps_tt"] * et) * vt * dx],
        [sigma_scale * (c["sigma_tt"] * et) * vt * dx, abc_t],
    ]
    forms_b = [
        [(c["nu_tt"] * wt) * wv * dx],
        [c["eps_nn"] * en * vn * dx],
        [sigma_scale * c["sigma_nn"] * en * vn * dx, abc_n],
    ]
    extra, slopes = [], []
    fill = _edge_fill(space.mesh, section)
    z0 = scipy.constants.mu_0 * scipy.constants.c
    for g, (item, factor) in enumerate(section.impedance_terms):
        edge_t = fill * et.Trace() * vt.Trace() * ds(f"impedance{g}")
        edge_n = fill * en.Trace() * vn.Trace() * ds(f"impedance{g}")
        if isinstance(item, Conductor):
            forms_a.append([edge_t])
            forms_b.append([edge_n])
            h = None if item.thickness is None else factor * item.thickness
            extra.append(ConductorScalar(item.conductivity, h, section.scale))
            slopes.append(space._conductor_slope(item, factor))
            continue
        if item.resistance is not None:
            forms_a[2].append(z0 / item.resistance * edge_t)
            forms_b[2].append(z0 / item.resistance * edge_n)
        if item.inductance is not None:
            w = scipy.constants.mu_0 * section.scale / item.inductance
            forms_a[0].append(w * edge_t)
            forms_b[0].append(w * edge_n)
        if item.capacitance is not None:
            w = item.capacitance / (scipy.constants.epsilon_0 * section.scale)
            forms_a[1].append(w * edge_t)
            forms_b[1].append(w * edge_n)
    index = section.law_index
    for g, (_, law) in enumerate(section.laws):
        inside = _p0(space.mesh, (np.asarray(index) == g).astype(float))
        forms_a.append([inside * et * vt * dx])
        forms_b.append([inside * en * vn * dx])
        extra.append(LawScalar(law, section.scale))
        slopes.append(law_slope(law, section.scale))
        if section.absorbing_law is not None and (section.absorbing_law == g).any():
            forms_a.append([et.Trace() * vt.Trace() * ds(f"law{g}")])
            forms_b.append([en.Trace() * vn.Trace() * ds(f"law{g}")])
            rows = np.asarray(index) == g
            mu = 1 / float(section.nu_nn[rows][0].real)
            extra.append(LawEdgeScalar(law, mu, section.scale))
            slopes.append(None)
    a = tuple(space._matrix(f) for f in forms_a)
    b = tuple(space._matrix(f) for f in forms_b)
    return a, b, extra, slopes


def _same(x: sp.csr_matrix, y: sp.csr_matrix) -> bool:
    x, y = x.tocsr(), y.tocsr()
    return (
        x.shape == y.shape
        and np.array_equal(x.indptr, y.indptr)
        and np.array_equal(x.indices, y.indices)
        and np.array_equal(x.data, y.data)
    )


def _check(section):
    space = PortModeSpace(section, 2)
    a, b, extra, slopes = _reference(space)
    pencil = space.pencil
    assert len(pencil.a) == len(a) == len(pencil.b) == len(b)
    for got, want in zip((*pencil.a, *pencil.b), (*a, *b), strict=True):
        assert _same(got, want)
    assert len(pencil.extra) == len(extra)
    assert [s is None for s in pencil.slopes] == [s is None for s in slopes]
    return space


def _sections(tmp_path, guide_mesh, plate_mesh, slab_mesh):
    out = {}
    slab = gfem.Model(
        slab_mesh,
        materials={
            "slab": gfem.Material(eps_r=4.0, tan_delta=0.02),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    out["slab"] = port_section(slab, slab.ports[0])
    wall = gfem.Conductor("walls", conductivity=5.8e7)
    out["conductor"] = port_section(
        _guide(guide_mesh, wall), gfem.WavePort("p", "port1")
    )
    air = _guide(guide_mesh, gfem.Absorbing("walls", order=2))
    out["absorbing"] = port_section(air, air.ports[0])
    law = _guide(guide_mesh, gfem.Absorbing("walls"), gfem.Material(eps_r=LAW))
    out["law"] = port_section(law, law.ports[0])
    law_pec = _guide(guide_mesh, gfem.PEC("walls"), gfem.Material(eps_r=LAW))
    out["law_pec"] = port_section(law_pec, law_pec.ports[0])
    rlc = _plate(
        plate_mesh,
        gfem.Impedance("plates", resistance=2.0, inductance=3e-12, capacitance=4e-15),
    )
    section = port_section(rlc, rlc.ports[0])
    out["impedance"] = section
    # a sheet field's fill, per edge
    fill = np.linspace(0.3, 1.0, len(section.impedance_edges))
    out["impedance_fill"] = dataclasses.replace(section, impedance_fill=fill)
    cond = _plate(plate_mesh, gfem.Conductor("plates", conductivity=1e6))
    out["conductor_plates"] = port_section(cond, cond.ports[0])
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, quarter=False, length=0.15
    )
    strip = pml_cases.strip_model(path)
    stretch = _pml.stretch(strip, [pml_cases.STRIP_F])
    out["pml"] = port_section(strip, strip.ports[0], stretch)
    return out


@pytest.fixture(scope="module")
def sections(tmp_path_factory, guide_mesh, plate_mesh, slab_mesh):
    return _sections(
        tmp_path_factory.mktemp("sections"), guide_mesh, plate_mesh, slab_mesh
    )


@pytest.mark.parametrize(
    "name",
    [
        "slab",
        "conductor",
        "absorbing",
        "law",
        "law_pec",
        "impedance",
        "impedance_fill",
        "conductor_plates",
        "pml",
    ],
)
def test_the_pencil_matrices_are_those_of_the_slot_by_slot_assembly(sections, name):
    _check(sections[name])


def test_every_item_names_its_slot_side_kind_region_and_input(sections):
    space = PortModeSpace(sections["impedance"], 2)
    items = space.form_items
    assert all(isinstance(i, FormItem) for i in items)
    assert {i.side for i in items} == {"A", "B"}
    inputs = {i.input for i in items}
    assert {("eps",), ("sigma",), ("nu",)} <= inputs
    assert ("impedance", 0, "resistance") in inputs
    assert ("impedance", 0, "inductance") in inputs
    assert ("impedance", 0, "capacitance") in inputs
    # a resistance sits in the i k0 slot, an inductance in the first, a
    # capacitance in the -k0^2 slot
    slot = {i.input[2]: i.slot for i in items if i.input[0] == "impedance"}
    assert slot == {"resistance": 2, "inductance": 0, "capacitance": 1}
    assert {i.region for i in items} == {"dx", "impedance0"}


def test_laws_and_conductors_add_a_slot_and_an_input_each(sections):
    law = PortModeSpace(sections["law"], 2)
    inputs = [i.input for i in law.form_items if i.input[0] in ("law", "absorbing_law")]
    assert ("law", 0) in inputs
    assert ("absorbing_law", 0) in inputs
    assert max(i.slot for i in law.form_items) == len(law.pencil.extra) + 2
    cond = PortModeSpace(sections["conductor_plates"], 2)
    assert {i.input for i in cond.form_items if i.slot > 2} == {("fill", 0)}


def test_the_absorbing_admittance_is_read_by_both_slots_of_one_side(sections):
    space = PortModeSpace(sections["absorbing"], 2)
    edges = [i for i in space.form_items if i.input == ("absorbing",)]
    assert {(i.side, i.kind) for i in edges} == {("A", "edge_t"), ("B", "edge_n")}
    assert {i.region for i in edges} == {"absorbing.*"}
