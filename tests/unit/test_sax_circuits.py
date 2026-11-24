r"""`gfem.SaxComponent` inside SAX circuits.

Optical: the 2D slab guide of `circuit_cases` as a SAX model: settings
discovery, the build call's solves, a reference-plane offset against SAX's
straight waveguide (an identity to round-off), and one gradient through a
gfem design setting and a SAX length together, eagerly and under `jax.jit`.

RF: the 3D get-started line as a block in series with SAX's capacitor,
against scikit-rf's cascade, and the gradients through the block's
permittivity, the reference impedance and an internal lumped capacitance.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from circuit_cases import (
    BAND,
    C0,
    OPTIONS,
    count_solves,
    scaled_function,
    slab_mesh,
)
from finite_differences import assert_gradient_matches, unit_directions

import gfem

sax = pytest.importorskip("sax")

PORTS = {"p1": "o1", "p2": "o2"}
WL = 1.5371  # um, between band frequencies


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("slab") / "slab.msh")


def _component(mesh, **kwargs):
    return gfem.SaxComponent(
        scaled_function(mesh, **kwargs),
        gfem.Driven(BAND, polarization="Ez"),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )


@pytest.fixture(scope="module")
def slab(mesh):
    return _component(mesh)


def _ports(model):
    return sax.get_ports(model())


@pytest.fixture(scope="module")
def mzi(slab):
    """An interferometer: the gfem slab in one arm, SAX's straight in the other."""
    i0, i1, o0, o1 = _ports(sax.models.coupler_ideal)
    a, b = _ports(sax.models.straight)
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "split": "coupler",
                "top": "slab",
                "bottom": "straight",
                "join": "coupler",
            },
            "connections": {
                f"split,{o0}": "top,o1",
                "top,o2": f"join,{i0}",
                f"split,{o1}": f"bottom,{a}",
                f"bottom,{b}": f"join,{i1}",
            },
            "ports": {"in": f"split,{i0}", "bar": f"join,{o0}", "cross": f"join,{o1}"},
        },
        models={
            "coupler": sax.models.coupler_ideal,
            "slab": slab,
            "straight": sax.models.straight,
        },
    )
    return circuit


def _cross_power(mzi, p, wl=WL):
    s = mzi(
        wl=wl,
        top={"scale": p[0]},
        bottom={"length": p[1], "neff": 3.0, "ng": 3.0, "wl0": 1.55},
    )
    return jnp.abs(s["in", "cross"]) ** 2


def test_sax_reads_the_components_settings(slab):
    assert sax.get_settings(slab) == slab.settings
    assert slab.settings == {"scale": 1.0, "wl": pytest.approx(C0 / BAND[2] * 1e6)}


def test_building_a_circuit_and_calling_it_at_the_defaults_solve_nothing(
    mesh, monkeypatch
):
    solves = count_solves(monkeypatch)
    component = _component(mesh)
    assert len(solves) == 1
    circuit, _ = sax.circuit(
        netlist={
            "instances": {"a": "slab", "b": "slab"},
            "connections": {"a,o2": "b,o1"},
            "ports": {"in": "a,o1", "out": "b,o2"},
        },
        models={"slab": component},
    )
    s = circuit()
    assert len(solves) == 1
    # two equal two-ports in a row, their reflections included
    one = {k: complex(v) for k, v in component().items()}
    s21 = one["o1", "o2"] ** 2 / (1 - one["o2", "o2"] * one["o1", "o1"])
    assert complex(s["in", "out"]) == pytest.approx(s21, rel=1e-12)


def test_a_port_offset_is_saxs_straight_with_the_port_modes_index(mesh, slab):
    # the plane of p2 moved inside by d: the plain slab is that component
    # followed by a straight of length d with the port mode's index
    d = 0.5e-6
    moved = _component(mesh, offset=d)
    wl = slab.settings["wl"]
    f = C0 / (wl * 1e-6)
    result = gfem.solve(scaled_function(mesh)(), gfem.Driven(f, polarization="Ez"))
    beta = np.imag(complex(np.asarray(result.port_modes["p2"].gamma)[0]))
    neff = beta / (2 * np.pi * f / C0)
    a, b = _ports(sax.models.straight)
    circuit, _ = sax.circuit(
        netlist={
            "instances": {"dev": "moved", "wg": "straight"},
            "connections": {"dev,o2": f"wg,{a}"},
            "ports": {"o1": "dev,o1", "o2": f"wg,{b}"},
        },
        models={"moved": moved, "straight": sax.models.straight},
    )
    s = circuit(wl=wl, wg={"length": d * 1e6, "neff": neff, "ng": neff, "wl0": wl})
    plain = slab(wl=wl)
    assert abs(complex(plain["o1", "o2"])) > 0.95
    for key, value in plain.items():
        assert abs(complex(s[key]) - complex(value)) < 1e-12, key


def test_one_gradient_through_a_gfem_setting_and_a_sax_length(mzi):
    p = jnp.array([1.02, 2.1])

    def objective(p):
        return _cross_power(mzi, p)

    value, grad = jax.value_and_grad(objective)(p)
    assert 0.05 < value < 0.95
    assert np.all(np.abs(grad) > 1e-3)
    directions = unit_directions(np.random.default_rng(3), p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=1e-3, grad=grad)
    jit_value, jit_grad = jax.jit(jax.value_and_grad(objective))(p)
    assert abs(jit_value - value) < 1e-12
    np.testing.assert_allclose(jit_grad, grad, rtol=1e-10, atol=0)


def test_a_wavelength_sweep_under_vmap_solves_nothing(mzi, monkeypatch):
    p = (1.0, 2.1)  # constants: an array indexed inside jax.jit would be traced
    _cross_power(mzi, p)  # warm: SAX compiles its first call
    solves = count_solves(monkeypatch)
    wls = jnp.linspace(1.5, 1.6, 41)
    power = jax.jit(jax.vmap(lambda wl: _cross_power(mzi, p, wl)))(wls)
    assert len(solves) == 0
    assert power.shape == (41,)
    assert np.all(np.isfinite(power))
    assert float(power[20]) == pytest.approx(float(_cross_power(mzi, p, wls[20])))


# --- RF ----------------------------------------------------------------------------

LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"
RF_BAND = (2e9, 5e9, 8e9)
Z_LINE = 376.73 / 6  # ohm, the line's quasi-TEM impedance, port 1's resistance


def _line(scale=1.0, capacitance=None, resistance=50.0):
    """The line between two lumped ports; port 2 is an internal capacitance if given."""
    mesh = gfem.Mesh.from_gmsh(LINE, length_unit="mm")
    p2 = (
        gfem.LumpedPort("p2", "port2", "+z", resistance=resistance)
        if capacitance is None
        else gfem.LumpedPort("p2", "port2", "+z", capacitance=capacitance)
    )
    field = gfem.MaterialField.uniform("design", gfem.Material(eps_r=4.0))
    field = dataclasses.replace(field, eps_r=4.0 * scale)
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": field},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.LumpedPort("p1", "port1", "+z", resistance=resistance), p2],
    )


@pytest.fixture(scope="module")
def rf_block():
    return gfem.SaxComponent(
        lambda *, scale=1.0: _line(scale),
        gfem.Driven(RF_BAND),
        circuit="rf",
        ports=PORTS,
        options=OPTIONS,
    )


@pytest.fixture(scope="module")
def rf_chain(rf_block):
    """The block, then SAX's series capacitor."""
    chain, _ = sax.circuit(
        netlist={
            "instances": {"dut": "line", "cap": "cap"},
            "connections": {"dut,o2": "cap,o1"},
            "ports": {"in": "dut,o1", "out": "cap,o2"},
        },
        models={"line": rf_block, "cap": sax.models.rf.capacitor},
    )
    return chain


def _skrf_cascade(scale, capacitance, z0):
    """scikit-rf's ``to_skrf`` block followed by the capacitor, at ``z0``."""
    rf = pytest.importorskip("skrf")
    solved = gfem.solve(
        _line(scale), gfem.Driven(RF_BAND, excite=("p1", "p2")), options=OPTIONS
    )
    block = solved.s.to_skrf(ports=["p1", "p2"])
    block.renormalize(z0)
    f = np.sort(RF_BAND)
    sd = sax.models.rf.capacitor(f=f, capacitance=capacitance, z0=z0)
    s = np.zeros((len(f), 2, 2), complex)
    for i, a in enumerate(("o1", "o2")):
        for j, b in enumerate(("o1", "o2")):
            s[:, j, i] = np.asarray(sd[a, b])
    cap = rf.Network(f=f, s=s, z0=z0, s_def="pseudo")
    return (block**cap).s


@pytest.mark.parametrize("z0", [50.0, 75.0])
def test_an_rf_block_with_saxs_capacitor_is_scikit_rfs_cascade(rf_chain, z0):
    scale, capacitance = 1.3, 0.4e-12
    f = np.sort(RF_BAND)
    s = rf_chain(f=f, z0=z0, dut={"scale": scale}, cap={"capacitance": capacitance})
    expected = _skrf_cascade(scale, capacitance, z0)
    got = {("in", "in"): (0, 0), ("in", "out"): (1, 0), ("out", "in"): (0, 1)}
    for key, (i, j) in got.items():
        np.testing.assert_allclose(s[key], expected[:, i, j], rtol=0, atol=1e-12)


def test_the_reference_impedance_is_no_solve_and_changes_s(rf_chain, monkeypatch):
    f = np.sort(RF_BAND)
    at_50 = rf_chain(f=f, z0=50.0)
    solves = count_solves(monkeypatch)
    at_75 = rf_chain(f=f, z0=75.0)
    assert len(solves) == 0
    assert not np.allclose(at_50["in", "out"], at_75["in", "out"], atol=1e-3)


def test_rf_gradients_over_permittivity_and_the_reference_impedance(rf_chain):
    f = jnp.asarray(np.sort(RF_BAND))

    def objective(p):
        s = rf_chain(
            f=f,
            z0=50.0 * p[1],
            dut={"scale": p[0]},
            cap={"capacitance": 1e-12},
        )
        return jnp.sum(jnp.abs(s["in", "out"]) ** 2)

    p = jnp.array([1.3, 1.1])
    grad = jax.grad(objective)(p)
    assert np.all(np.abs(grad) > 1e-4)
    directions = unit_directions(np.random.default_rng(5), p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=1e-3, grad=grad)


def test_an_unexposed_capacitance_port_has_a_differentiable_value():
    block = gfem.SaxComponent(
        lambda *, c=0.5e-12: _line(capacitance=c, resistance=Z_LINE),
        gfem.Driven(RF_BAND),
        circuit="rf",
        ports={"p1": "o1"},
        options=OPTIONS,
    )
    assert block.settings["c"] == 0.5e-12

    def objective(p):
        # the capacitance in pF, the reflection's phase against a fixed phasor
        s = block(f=5e9, c=p[0] * 1e-12)
        return jnp.real(s["o1", "o1"] * (0.3 - 0.7j))

    p = jnp.array([0.5])
    grad = jax.grad(objective)(p)
    assert abs(float(grad[0])) > 1e-4
    directions = unit_directions(np.random.default_rng(7), p.shape, 1)
    assert_gradient_matches(objective, p, directions, step=1e-3, grad=grad)
