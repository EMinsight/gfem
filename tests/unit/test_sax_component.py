"""`gfem.SaxComponent` called directly, without SAX.

A 2D slab guide (`circuit_cases`) with a design patch, wrapped as an optical
component: validation, signature and settings, the band, the memo's solve
counts and a finite-difference gradient. Where only the bookkeeping is under
test, a canned solve stands in for `gfem.solve`. SAX itself is never
imported here.
"""

import dataclasses
import inspect
import sys
import types
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
    design_eps,
    slab,
    slab_function,
    slab_mesh,
)
from finite_differences import assert_gradient_matches, unit_directions
from periodic_cases import f1, f1_mesh

import gfem

LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"
PORTS = {"p1": "o1", "p2": "o2"}
PAIRS = {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("slab") / "slab.msh")


def _template():
    return gfem.Driven(BAND, polarization="Ez")


@pytest.fixture(scope="module")
def component(mesh):
    return gfem.SaxComponent(
        slab_function(mesh),
        _template(),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )


@pytest.fixture
def solves(monkeypatch):
    return count_solves(monkeypatch)


@pytest.fixture
def canned(monkeypatch):
    """Replace the component's solves by a canned S: S_ij(f) = (i + 2j) f / f0."""
    count = []

    def fake(model, problem, **_):
        count.append(1)
        f = np.asarray(problem.frequency)
        rows = np.arange(len(model.port_names))[:, None]
        cols = np.array([model.port_names.index(p) for p in problem.excite])[None, :]
        values = (rows + 2j * cols + 1)[None] * (f / f.min())[:, None, None]
        s = gfem.NetworkParameters(
            values=values,
            z_ref=np.full((len(f), len(model.port_names)), 50.0),
            frequency=problem.frequency,
            ports=model.port_names,
            excited=problem.excite,
        )
        return types.SimpleNamespace(s=s)

    monkeypatch.setattr("gfem.circuit.solve", fake)
    return count


def _lumped_line(**ports):
    mesh = gfem.Mesh.from_gmsh(LINE, length_unit="mm")
    default = {
        "p1": gfem.LumpedPort("p1", "port1", "+z", resistance=50.0),
        "p2": gfem.LumpedPort("p2", "port2", "+z", resistance=50.0),
    }
    return gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=list({**default, **ports}.values()),
    )


# --- the signature and the object --------------------------------------------------


def test_the_signature_is_the_functions_keywords_and_the_wavelength(component):
    signature = inspect.signature(component)
    assert list(signature.parameters) == ["eps", "wl"]
    for parameter in signature.parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.return_annotation is inspect.Signature.empty
    eps = signature.parameters["eps"].default
    assert np.array_equal(eps, component.settings["eps"])
    # the band point nearest the band's midpoint (in f), as a wavelength in um
    assert signature.parameters["wl"].default == component.settings["wl"]
    assert component.settings["wl"] == pytest.approx(C0 / np.sort(BAND)[2] * 1e6)


def test_the_component_is_read_only_identity_hashed_with_a_repr(mesh, component):
    with pytest.raises(dataclasses.FrozenInstanceError):
        component.circuit = "rf"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        component.settings["wl"] = 1.0  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        component.ports["p1"] = "x"  # ty: ignore[invalid-assignment]
    assert {component: 1}[component] == 1
    assert component.circuit == "optical"
    assert dict(component.ports) == PORTS
    assert component.problem == _template()
    assert component.options is OPTIONS
    n = len(design_eps(mesh))
    text = repr(component)
    assert text.startswith(
        f"SaxComponent(circuit='optical', ports={PORTS!r}, settings={{'eps': "
        f"array({n},), 'wl': "
    )
    assert text.endswith("frequency=5 values 187.4-199.9 THz)")


@pytest.mark.usefixtures("canned")
def test_a_bare_model_is_a_component_without_design_settings(mesh):
    component = gfem.SaxComponent(
        slab(mesh, design_eps(mesh)),
        _template(),
        circuit="optical",
        ports=PORTS,
    )
    assert list(component.settings) == ["wl"]
    assert set(component()) == PAIRS


# --- refusals ----------------------------------------------------------------------


def _with_args(*args, eps=1.0):
    raise AssertionError((args, eps))


def _with_kwargs(*, eps=1.0, **kwargs):
    raise AssertionError((eps, kwargs))


def _positional_only(eps=1.0, /):
    raise AssertionError(eps)


def _no_default(*, eps):
    raise AssertionError(eps)


def _owned(*, wl=1.55):
    raise AssertionError(wl)


def _dict_default(*, eps={"core": 12.0}):  # noqa: B006
    raise AssertionError(eps)


@pytest.mark.parametrize(
    ("function", "error", "message"),
    [
        (_with_args, TypeError, r"\*args.*own keyword parameter.*def device"),
        (_with_kwargs, TypeError, r"\*\*kwargs.*own keyword parameter.*def device"),
        (_positional_only, TypeError, r"'eps' is positional-only.*drop the '/'"),
        (_no_default, TypeError, r"'eps' has no default.*give it one"),
        (_owned, ValueError, r"'wl' is a setting of the component.*rename it"),
        (_dict_default, TypeError, r"'eps' has a dictionary default.*separate"),
        (42, TypeError, r"gfem.Model or a function.*got int"),
        (lambda: 42, TypeError, r"must return a gfem.Model, got int"),
    ],
)
def test_a_model_function_saxs_settings_cannot_read_raises(function, error, message):
    with pytest.raises(error, match=message):
        gfem.SaxComponent(function, _template(), circuit="optical")


def _frequency(*, f=1e14):
    raise AssertionError(f)


def _reference(*, z0=50.0):
    raise AssertionError(z0)


@pytest.mark.parametrize(("function", "name"), [(_frequency, "f"), (_reference, "z0")])
def test_every_name_the_component_owns_is_refused_for_either_kind(function, name):
    with pytest.raises(ValueError, match=rf"'{name}' is a setting of the component"):
        gfem.SaxComponent(function, _template(), circuit="optical")


def test_the_circuit_kind_is_required_and_checked(mesh):
    function = slab_function(mesh)
    with pytest.raises(TypeError, match="circuit"):
        gfem.SaxComponent(function, _template())  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match=r"'optical' or 'rf'.*never inferred"):
        gfem.SaxComponent(
            function,
            _template(),
            circuit="photonic",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("problem", "error", "message"),
    [
        (gfem.Eigenmode(1e14), TypeError, r"gfem.Driven template.*got Eigenmode"),
        (
            gfem.Driven(BAND, polarization="Ez", excite="p1"),
            ValueError,
            r"excite=\('p1',\) differs from the exposed ports.*excite=None",
        ),
        (
            gfem.Driven(BAND, polarization="Ez", keep_fields=True),
            ValueError,
            r"keeps fields.*drop keep_fields",
        ),
        (
            gfem.Driven(  # refused before the 2D polarization is checked
                BAND,
                sources=gfem.ElectricDipole(
                    "d", position=(1e-6, 0.0, 0.0), moment=(0, 0, 1e-30)
                ),
            ),
            ValueError,
            r"has sources.*drop sources",
        ),
    ],
)
def test_a_template_the_component_cannot_use_raises(mesh, problem, error, message):
    with pytest.raises(error, match=message):
        gfem.SaxComponent(slab_function(mesh), problem, circuit="optical")


@pytest.mark.usefixtures("canned")
def test_the_template_may_name_the_exposed_ports_in_any_order(mesh):
    problem = gfem.Driven(BAND, polarization="Ez", excite=("p2", "p1"))
    component = gfem.SaxComponent(slab_function(mesh), problem, circuit="optical")
    assert component.problem is problem


@pytest.mark.parametrize(
    ("ports", "error", "message"),
    [
        ({"p3": "o1"}, ValueError, r"No port 'p3'; the ports are 'p1', 'p2'"),
        ({"p1": "o1", "p2": "o1"}, ValueError, r"'o1' is given to more than one"),
        ({"p1": "o-1"}, ValueError, r"'o-1' is not an identifier"),
        ({"p1": "o1@TE0", "p2": "o2"}, ValueError, r"every circuit port name"),
        ({}, ValueError, r"exposes no port"),
        (["p1"], TypeError, r"maps gfem port names"),
    ],
)
def test_a_bad_port_mapping_raises_listing_the_ports(mesh, ports, error, message):
    with pytest.raises(error, match=message):
        gfem.SaxComponent(
            slab_function(mesh), _template(), circuit="optical", ports=ports
        )


@pytest.mark.usefixtures("canned")
def test_a_gfem_port_name_that_is_no_identifier_needs_a_mapping(mesh):
    def device():
        model = slab(mesh, design_eps(mesh))
        p1, p2 = model.ports
        return model.replace(ports=(dataclasses.replace(p1, name="port-1"), p2))

    with pytest.raises(ValueError, match=r"'port-1'.*ports=\{'port-1': 'o1'\}"):
        gfem.SaxComponent(device, _template(), circuit="optical")
    component = gfem.SaxComponent(
        device, _template(), circuit="optical", ports={"port-1": "o1", "p2": "p2"}
    )
    assert set(component()) == {("o1", "o1"), ("o1", "p2"), ("p2", "o1"), ("p2", "p2")}


@pytest.mark.usefixtures("canned")
def test_an_exposed_wave_port_without_a_polarity_reference_raises(mesh):
    function = slab_function(mesh, polarity=False)
    with pytest.raises(ValueError, match=r"WavePort 'p1' has no polarity reference.*"):
        gfem.SaxComponent(function, _template(), circuit="optical")

    def p2_unreferenced():
        model = slab(mesh, design_eps(mesh))
        return model.replace(
            ports=(model.ports[0], dataclasses.replace(model.ports[1], polarity=None))
        )

    with pytest.raises(ValueError, match=r"WavePort 'p2'.*polarity=.*leave it out"):
        gfem.SaxComponent(p2_unreferenced, _template(), circuit="optical")
    # a port left out terminates the device: its sign cannot reach S
    component = gfem.SaxComponent(
        p2_unreferenced, _template(), circuit="optical", ports={"p1": "o1"}
    )
    assert set(component()) == {("o1", "o1")}


@pytest.mark.usefixtures("canned")
def test_lumped_and_voltage_path_ports_need_no_polarity_reference():
    path = ((0.04, 0.003, 0.0), (0.04, 0.003, 0.001))
    for model in (
        _lumped_line(),
        _lumped_line(p2=gfem.WavePort("p2", "port2", voltage_path=path)),
    ):
        component = gfem.SaxComponent(model, gfem.Driven(5e9), circuit="optical")
        assert set(component()) == {
            ("p1", "p1"),
            ("p1", "p2"),
            ("p2", "p1"),
            ("p2", "p2"),
        }


# --- RF ---------------------------------------------------------------------------


def _rf(model, **kwargs):
    return gfem.SaxComponent(model, gfem.Driven(5e9), circuit="rf", **kwargs)


@pytest.mark.usefixtures("canned")
def test_an_rf_component_has_a_frequency_and_a_reference_setting():
    component = _rf(_lumped_line())
    assert list(inspect.signature(component).parameters) == ["f", "z0"]
    assert component.settings == {"f": 5e9, "z0": 50.0}
    assert set(component()) == {
        ("p1", "p1"),
        ("p1", "p2"),
        ("p2", "p1"),
        ("p2", "p2"),
    }
    with pytest.raises(ValueError, match=r"f = 9 GHz is outside.*5 GHz"):
        component(f=9e9)
    with pytest.raises(TypeError, match="no setting 'wl'"):
        component(wl=1.5)


@pytest.mark.usefixtures("canned")
@pytest.mark.parametrize(
    ("ports", "message"),
    [
        (
            {"p2": gfem.LumpedPort("p2", "port2", "+z", capacitance=1e-12)},
            r"LumpedPort 'p2' has no resistance.*resistance=50.*leave it out of ports=",
        ),
        (
            {"p2": gfem.WavePort("p2", "port2", polarity=((0.04, 0.003, 0.0), "+z"))},
            r"WavePort 'p2' has no voltage_path.*voltage_path=.*leave it out",
        ),
    ],
)
def test_an_rf_port_without_a_reference_impedance_raises_naming_the_remedy(
    ports, message
):
    with pytest.raises(ValueError, match=message):
        _rf(_lumped_line(**ports))
    # left out of ports=, it is a load inside the device
    component = _rf(_lumped_line(**ports), ports={"p1": "o1"})
    assert set(component()) == {("o1", "o1")}


def test_an_rf_component_refuses_an_exposed_floquet_port(tmp_path):
    cell = f1(f1_mesh(tmp_path))
    with pytest.raises(ValueError, match=r"FloquetPort 'R_TE' cannot be an RF"):
        gfem.SaxComponent(cell, gfem.Driven(5e14), circuit="rf")


def test_an_rf_component_refuses_a_2d_mesh(mesh):
    with pytest.raises(ValueError, match=r"needs a 3D model.*circuit='optical'"):
        gfem.SaxComponent(slab_function(mesh), _template(), circuit="rf")


@pytest.mark.usefixtures("canned")
@pytest.mark.parametrize("z0", [50j, 50 + 1j, [50.0, 75.0], -50.0, 0.0, np.nan, "50"])
def test_a_z0_that_is_not_one_positive_real_raises_before_any_solve(z0, solves):
    component = _rf(_lumped_line())
    n = len(solves)
    with pytest.raises(ValueError, match="one real reference impedance"):
        component(z0=z0)
    assert len(solves) == n


def test_z0_is_no_solve_input_but_renormalizes_s(monkeypatch):
    count = count_solves(monkeypatch)
    component = _rf(_lumped_line())
    assert len(count) == 1
    at_50 = component()
    at_75 = component(z0=75.0)
    jax.jit(lambda z0: component(z0=z0))(100.0)
    jax.grad(lambda z0: jnp.abs(component(z0=z0)["p1", "p1"]))(60.0)
    assert len(count) == 1
    assert not np.allclose(at_50["p1", "p1"], at_75["p1", "p1"])
    assert np.allclose(component(z0=75.0)["p1", "p1"], at_75["p1", "p1"], atol=0)


def test_an_unknown_setting_raises_listing_the_settings(component):
    with pytest.raises(TypeError, match=r"no setting 'width'; its settings are 'eps'"):
        component(width=1.0)


# --- ports, band and values --------------------------------------------------------


@pytest.mark.usefixtures("canned")
def test_ports_rename_and_select_the_entries_of_s(mesh):
    full = gfem.SaxComponent(slab_function(mesh), _template(), circuit="optical")
    assert set(full()) == {("p1", "p1"), ("p1", "p2"), ("p2", "p1"), ("p2", "p2")}
    one = gfem.SaxComponent(
        slab_function(mesh), _template(), circuit="optical", ports={"p2": "out"}
    )
    (key,) = one()
    assert key == ("out", "out")
    # canned S_ij = (i + 2j + 1) f / f_min, conjugated: S_22 at the band's lowest
    wl = C0 / BAND.min() * 1e6
    assert complex(one(wl=wl)[key]) == pytest.approx(2 - 2j, rel=1e-12)


@pytest.mark.usefixtures("canned")
def test_the_default_wavelength_is_the_band_point_nearest_the_midpoint(mesh):
    f = np.array([2.0, 5.0, 1.0, 2.4]) * 1e14  # midpoint 3e14: 2.4e14 is nearest
    component = gfem.SaxComponent(
        slab_function(mesh), gfem.Driven(f, polarization="Ez"), circuit="optical"
    )
    assert component.settings["wl"] == pytest.approx(C0 / 2.4e14 * 1e6, rel=1e-15)
    assert repr(component).endswith("frequency=4 values 100-500 THz)")


def test_the_component_at_a_band_frequency_is_the_solves_s(mesh, component):
    reference = gfem.solve(
        slab_function(mesh)(), _template(), options=OPTIONS
    ).s.to_sdict(circuit="optical", ports=PORTS)
    for i, f in enumerate(BAND):
        values = component(wl=C0 / f * 1e6)
        assert set(values) == PAIRS
        for key in PAIRS:
            assert abs(complex(values[key]) - complex(reference[key][i])) < 1e-12
    # the slab's transmission, conjugated: S_21 = e^{-i beta L} reads e^{+i beta L}
    s21 = complex(component()["o1", "o2"])
    assert abs(s21) > 0.95
    assert abs(complex(reference["o1", "o2"][2]) - s21) < 1e-12


def test_a_band_in_any_order_gives_the_same_component(mesh, component):
    shuffled = gfem.SaxComponent(
        slab_function(mesh),
        gfem.Driven(BAND[[3, 0, 4, 1, 2]], polarization="Ez"),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )
    assert shuffled.settings["wl"] == component.settings["wl"]
    wl = jnp.array([1.503, 1.55, 1.597])
    a, b = shuffled(wl=wl), component(wl=wl)
    for key in PAIRS:
        assert a[key].shape == (3,)
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)


@pytest.mark.usefixtures("canned")
def test_between_band_frequencies_the_cubic_follows_a_smooth_entry(mesh):
    # canned S is linear in f, which the cubic reproduces exactly
    component = gfem.SaxComponent(slab_function(mesh), _template(), circuit="optical")
    wl = np.array([1.5013, 1.5378, 1.5991])
    f = C0 / (wl * 1e-6)
    np.testing.assert_allclose(
        component(wl=wl)["p2", "p1"], (1 - 2j) * f / BAND.min(), rtol=1e-13
    )


def test_a_wavelength_outside_the_band_raises_eagerly_and_is_nan_traced(component):
    with pytest.raises(
        ValueError,
        match=r"wl = 1.45 um is outside the component's band, 1.5-1.6 um "
        r"\(187.37-199.862 THz\); widen the frequencies of its Driven template",
    ):
        component(wl=[1.55, 1.45])
    value = jax.jit(lambda wl: component(wl=wl)["o1", "o2"])(jnp.array([1.55, 1.45]))
    assert np.isfinite(value[0])
    assert np.isnan(value[1])

    def power(wl):
        return jnp.abs(component(wl=wl)["o1", "o2"]) ** 2

    assert np.isnan(jax.grad(power)(1.45))
    assert np.isfinite(jax.grad(power)(1.55))


@pytest.mark.usefixtures("canned")
def test_a_one_frequency_band_admits_only_its_frequency(mesh):
    component = gfem.SaxComponent(
        slab_function(mesh), gfem.Driven(BAND[0], polarization="Ez"), circuit="optical"
    )
    wl = component.settings["wl"]
    assert complex(component(wl=wl)["p1", "p1"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the component's band"):
        component(wl=wl * 1.001)


# --- solves ------------------------------------------------------------------------


def test_the_memo_solves_each_distinct_design_once(mesh, solves):
    component = gfem.SaxComponent(
        slab_function(mesh),
        _template(),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )
    assert len(solves) == 1  # the constructor solves at the defaults
    first = component()
    eps = design_eps(mesh)
    component(eps=jnp.asarray(eps))  # equal values, another array
    component(wl=1.52)
    jax.jit(lambda wl: component(wl=wl))(1.53)
    jax.jit(jax.vmap(lambda wl: component(wl=wl)))(jnp.linspace(1.5, 1.6, 7))
    assert len(solves) == 1
    new = component(eps=eps * 1.01)
    assert len(solves) == 2
    again = component(eps=eps * 1.01)
    assert len(solves) == 2
    for key in PAIRS:  # a hit is the skipped solve's S, bit for bit
        assert np.array_equal(again[key], new[key])
    assert np.array_equal(component()["o1", "o2"], first["o1", "o2"])


def test_a_traced_design_solves_every_time_and_is_never_kept(mesh, solves):
    component = gfem.SaxComponent(
        slab_function(mesh),
        _template(),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )
    eps = jnp.asarray(design_eps(mesh))

    def power(eps):
        return jnp.abs(component(eps=eps)["o1", "o2"]) ** 2

    jax.grad(power)(eps)
    jax.grad(power)(eps)
    assert len(solves) == 3
    # a concrete design solved inside jax.jit gives staged S: never kept
    other = eps * 1.02
    jax.jit(lambda: component(eps=other)["o1", "o2"])()
    assert len(solves) == 4
    component(eps=other)
    assert len(solves) == 5


def test_the_memo_keeps_the_last_64_designs(mesh, canned):
    model = slab(mesh, design_eps(mesh))

    def device(*, k=0):
        assert k >= 0
        return model

    component = gfem.SaxComponent(device, _template(), circuit="optical")
    for k in range(1, 65):
        component(k=k)
    assert len(canned) == 65
    component(k=64)
    assert len(canned) == 65
    component(k=0)  # the oldest, dropped
    assert len(canned) == 66


def test_the_cache_stores_the_constructors_solve(mesh, tmp_path):
    gfem.SaxComponent(
        slab_function(mesh),
        _template(),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
        cache=tmp_path,
    )
    assert any(tmp_path.iterdir())


def test_a_component_runs_without_sax(mesh, monkeypatch):
    for name in ("sax", "klujax"):
        monkeypatch.setitem(sys.modules, name, None)  # any import of them raises
    component = gfem.SaxComponent(
        slab_function(mesh),
        _template(),
        circuit="optical",
        ports=PORTS,
        options=OPTIONS,
    )
    assert set(component()) == PAIRS
    assert set(component(eps=design_eps(mesh) * 1.01, wl=1.57)) == PAIRS


# --- gradient ----------------------------------------------------------------------


def functional(s):
    """A real functional of every S entry."""
    return (
        jnp.abs(s["o1", "o1"]) ** 2
        + 0.5 * jnp.abs(s["o1", "o2"] - 0.3j) ** 2
        + 2.0 * jnp.abs(s["o2", "o1"]) ** 2
        - jnp.abs(s["o2", "o2"]) ** 2
        + jnp.real((0.3 - 0.7j) * s["o1", "o2"])
    )


def test_the_gradient_through_the_component_matches_differences(mesh, component):
    rng = np.random.default_rng(0)
    eps = jnp.asarray(design_eps(mesh)) * rng.uniform(0.9, 1.1, len(design_eps(mesh)))

    def objective(eps):
        # an off-grid wavelength: the interpolation is inside the check
        return functional(component(eps=eps, wl=1.5371))

    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)
