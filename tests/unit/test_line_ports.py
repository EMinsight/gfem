r"""Line ports of a 2D mesh: their 1D mode, its ends, and its trace on the 2D space.

The line tutorial's triangle mesh (``docs/tutorials/data/line2d.msh``, mm):
port ``port1`` is the line x = 0, 6 mm long, in the area ``feed``, and meets
the line ``side`` at both ends. In a uniform medium of permittivity
:math:`\varepsilon` the mode is uniform (:math:`n_\text{eff}^2 =
\varepsilon`) where the ends are natural, and a half sine
(:math:`n_\text{eff}^2 = \varepsilon - (\pi / k_0 L)^2`) where they are
essential: PEC is essential in Ez and natural in Hz, PMC (the default) the
reverse.
"""

import math
import types
from pathlib import Path

import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
from scipy.constants import c, epsilon_0
from strip2d import write_strip

import gfem
from gfem._engine.line_mode import LineModeSpace
from gfem._engine.mesh import to_ngsolve
from gfem._engine.wave_ports import trace_dofs
from gfem._port_section import port_section, port_section_key
from gfem.errors import PortGeometryError

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
F = 50e9
LENGTH = 6.0  # the port's length, mm
EPS = 4.0
OPTIONS = gfem.SolverOptions(order=3)


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")


def line(mesh, *, boundaries=(), material=None, **port):
    material = material or gfem.Material(eps_r=EPS)
    return gfem.Model(
        mesh,
        materials={"feed": material, "design": gfem.Material(eps_r=EPS)},
        boundaries=boundaries,
        ports=(gfem.WavePort("p1", "port1", **port), gfem.WavePort("p2", "port2")),
    )


def n_eff(model, polarization, frequency=F):
    problem = gfem.PortMode("p1", frequency, polarization=polarization)
    return complex(np.asarray(gfem.solve(model, problem, options=OPTIONS).n_eff)[0])


def half_sine(eps=EPS, frequency=F):
    k0 = 2 * math.pi * frequency / c * 1e-3  # per mm
    return math.sqrt(eps - (math.pi / (k0 * LENGTH)) ** 2)


# --- the ends take the condition of the line that meets them -----------------------

ENDS = {
    "default PMC, Ez: natural": ({}, "Ez", math.sqrt(EPS)),
    "default PMC, Hz: essential": ({}, "Hz", half_sine()),
    "PEC line, Ez: essential": ({"boundaries": (gfem.PEC("side"),)}, "Ez", half_sine()),
    "PEC line, Hz: natural": (
        {"boundaries": (gfem.PEC("side"),)},
        "Hz",
        math.sqrt(EPS),
    ),
    "pec_edges, Ez: essential": ({"pec_edges": ("side",)}, "Ez", half_sine()),
    "pec_edges, Hz: natural": ({"pec_edges": ("side",)}, "Hz", math.sqrt(EPS)),
    "PMC line, Ez: natural": ({"boundaries": (gfem.PMC("side"),)}, "Ez", 2.0),
}


@pytest.mark.parametrize("case", ENDS)
def test_a_line_ports_ends_take_the_meeting_lines_condition(mesh, case):
    change, polarization, expected = ENDS[case]
    n = n_eff(line(mesh, **change), polarization)
    assert n.real == pytest.approx(expected, rel=1e-6)
    assert abs(n.imag) < 1e-9


def test_pec_wins_over_an_absorbing_line_at_an_end(mesh):
    model = line(mesh, boundaries=(gfem.Absorbing("side"),), pec_edges=("side",))
    assert port_section(model, model.ports[0]).end_kinds == ("pec", "pec")
    assert n_eff(model, "Ez").real == pytest.approx(half_sine(), rel=1e-6)


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_an_absorbing_end_lets_the_mode_leak(mesh, polarization):
    model = line(mesh, boundaries=(gfem.Absorbing("side", order=2),))
    assert port_section(model, model.ports[0]).end_kinds == ("absorbing",) * 2
    # attenuated: n_eff = n' - i n'' with n'' > 0
    assert n_eff(model, polarization).imag < -1e-3


# --- frequency-dependent media ------------------------------------------------------

SIGMA = 0.4 * 2 * math.pi * F * epsilon_0  # sigma / (omega eps0) = 0.4 at F
LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=4 * F, damping=0.1 * F)],
)


def law_eps(frequency):
    w, w0, g = (2 * math.pi * x for x in (frequency, 4 * F, 0.1 * F))
    return 2.0 + 2.0 * w0**2 / (w0**2 - w**2 + 1j * g * w)


# the uniform mode where the ends are natural: n_eff^2 = eps_r(omega)
MEDIA = {
    "conductor, Ez": (gfem.Material(eps_r=EPS, conductivity=SIGMA), "Ez", EPS - 0.4j),
    "conductor, Hz": (gfem.Material(eps_r=EPS, conductivity=SIGMA), "Hz", EPS - 0.4j),
    "law, Ez": (gfem.Material(eps_r=LAW), "Ez", law_eps(F)),
    "law, Hz": (gfem.Material(eps_r=LAW), "Hz", law_eps(F)),
}


@pytest.mark.parametrize("case", MEDIA)
def test_a_lossy_uniform_line_has_the_index_of_its_medium(mesh, case):
    material, polarization, eps = MEDIA[case]
    # natural ends: the default PMC in Ez, PEC in Hz
    change = {"pec_edges": ("side",)} if polarization == "Hz" else {}
    model = line(mesh, material=material, **change)
    assert n_eff(model, polarization) == pytest.approx(np.sqrt(eps), rel=1e-9)


def test_an_hz_conductive_region_puts_its_inverse_permittivity_in_the_pencil(mesh):
    model = line(mesh, material=gfem.Material(eps_r=EPS, conductivity=SIGMA))
    section = port_section(model, model.ports[0])
    hz = LineModeSpace(section, 2, "Hz")
    k0 = 2 * math.pi * F / c * section.scale
    (theta,) = hz.pencil.extra
    assert theta(k0) == pytest.approx(1 / (EPS - 0.4j), rel=1e-12)
    # Ez keeps the conductivity in its i k0 mass piece: no extra scalar
    assert LineModeSpace(section, 2, "Ez").pencil.extra == ()


# --- the result ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("polarization", "name", "unit"),
    [("Ez", "e_field", "V/m"), ("Hz", "h_field", "A/m")],
)
def test_a_line_ports_mode_field_is_its_polarizations(mesh, polarization, name, unit):
    model = line(mesh, pec_edges=("side",))
    result = gfem.solve(
        model, gfem.PortMode("p1", [F, 1.1 * F], polarization=polarization)
    )
    field = getattr(result, name)
    other = result.h_field if name == "e_field" else result.e_field
    assert other is None
    assert (field.quantity, field.unit) == (polarization, unit)
    assert field.normal == pytest.approx((1.0, 0.0))  # into the line, along +x
    values = np.asarray(field.at(np.array([[0.0, 1e-3], [0.0, 3e-3]])))
    assert values.shape == (2, 2)
    assert np.isnan(np.asarray(result.Z_PV)).all()
    with pytest.raises(ValueError, match=r"\(x, y\) pairs"):
        field.at(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="not on the line"):
        field.at(np.array([[1e-3, 1e-3]]))


def test_an_ez_line_mode_carries_1_w_per_metre(mesh):
    # P' = beta / (2 omega mu0) int |u|^2 dt for the uniform mode of index 2
    model = line(mesh)
    result = gfem.solve(
        model, gfem.PortMode("p1", F, polarization="Ez"), options=OPTIONS
    )
    u = np.asarray(result.e_field.at(np.array([[0.0, 2e-3]])))[0, 0]
    omega = 2 * math.pi * F
    beta = 2.0 * omega / c
    power = beta / (2 * omega * 4e-7 * math.pi) * abs(u) ** 2 * LENGTH * 1e-3
    assert power == pytest.approx(1.0, rel=1e-6)
    assert abs(u.imag) <= 1e-9 * abs(u)  # gauge: int w u dt real and positive
    assert u.real > 0


# --- geometry and the problem ---------------------------------------------------------


def test_a_line_port_that_is_not_straight_raises(tmp_path):
    strip = gfem.Mesh.from_gmsh(write_strip(tmp_path / "strip.msh"), length_unit="um")
    model = gfem.Model(
        strip,
        materials={n: gfem.Material(eps_r=2.0) for n in "abc"},
        ports=(gfem.WavePort("p", "outer"),),
    )
    with pytest.raises(PortGeometryError, match="not straight"):
        port_section(model, model.ports[0])


def test_a_port_mode_on_a_2d_mesh_needs_a_polarization(mesh):
    with pytest.raises(ValueError, match=r'pass PortMode\(polarization="Ez"\)'):
        gfem.solve(line(mesh), gfem.PortMode("p1", F))


def test_the_section_key_tells_line_materials_apart(mesh):
    a, b = line(mesh), line(mesh, material=gfem.Material(eps_r=5.0))
    assert port_section_key(a, a.ports[0]) == port_section_key(line(mesh), a.ports[0])
    assert port_section_key(a, a.ports[0]) != port_section_key(b, b.ports[0])


def test_an_hz_line_mode_is_refused_by_characteristic_impedance():
    mode = gfem.PortModeResult(
        port="p",
        mode=1,
        frequency=jnp.array([F]),
        gamma=jnp.array([1j]),
        Z_PV=jnp.array([np.nan]),
        residual=jnp.zeros(1),
        solved=jnp.array([True]),
        h_field=types.SimpleNamespace(normal=(1.0, 0.0)),
    )
    with pytest.raises(NotImplementedError, match="1 W/m"):
        gfem.characteristic_impedance(mode, voltage_path=((0, 0, 0), (1, 0, 0)))


# --- the trace of the 2D space on the line ------------------------------------------


@pytest.mark.parametrize("order", [3, 4])
def test_trace_dofs_carry_a_line_function_to_its_2d_trace(mesh, order):
    """A 1D function copied through `trace_dofs` is the same function on the line.

    Oracle: the 2D H1 space restricted to the port line (``definedon``), with
    the function interpolated on it. Order 3 and up has segment-interior
    functions whose sign follows the segment's orientation on each mesh: a
    mismatch would flip them.
    """
    model = line(mesh)
    section = port_section(model, model.ports[0])
    modes = LineModeSpace(section, order, "Ez")
    built = to_ngsolve(model.mesh)
    space = ngs.H1(built.mesh, order=order, complex=True)
    idx2, idx1 = trace_dofs(model, section, modes, built, space)
    assert sorted(idx1) == list(range(modes.space.ndof))

    def f(t):  # no symmetry along the line
        return ngs.sin(1.3 * t) + 0.05 * t**3

    gf1 = ngs.GridFunction(modes.space)
    gf1.Set(f(ngs.x))
    copied = ngs.GridFunction(space)
    copied.vec.FV().NumPy()[idx2] = gf1.vec.FV().NumPy()[idx1]
    port = built.mesh.Boundaries("port1")
    trace = ngs.H1(built.mesh, order=order, complex=True, definedon=port)
    oracle = ngs.GridFunction(trace)
    origin, tangent = section.origin, section.frame[0]
    t = (ngs.x - origin[0]) * tangent[0] + (ngs.y - origin[1]) * tangent[1]
    oracle.Set(f(t), definedon=port)
    # off the nodes, where the interior functions do not vanish
    ts = np.linspace(-2.9, 2.9, 41)
    xy = section.to_global(ts) / section.scale
    one = [gf1(modes.mesh(float(s))) for s in ts]
    two = [copied(built.mesh(float(x), float(y), VOL_or_BND=ngs.BND)) for x, y in xy]
    ref = [oracle(built.mesh(float(x), float(y), VOL_or_BND=ngs.BND)) for x, y in xy]
    np.testing.assert_allclose(two, one, rtol=0, atol=1e-12)
    np.testing.assert_allclose(two, ref, rtol=0, atol=1e-6)


def test_an_hz_line_modes_result_bundle_reads_back_its_field(mesh, tmp_path):
    from gfem.job import _write_result  # the writer behind Job.run's bundle

    result = gfem.solve(line(mesh), gfem.PortMode("p1", F, polarization="Hz"))
    _write_result(tmp_path / "out", result, {}, None)
    back = gfem.load_result(tmp_path / "out")
    points = np.array([[0.0, 1e-3], [0.0, 2.5e-3]])
    assert back.e_field is None
    assert back.h_field.quantity == "Hz"
    np.testing.assert_array_equal(back.h_field.at(points), result.h_field.at(points))


def test_an_ez_pec_edges_mode_vanishes_where_the_2d_field_stays_free(mesh):
    # pec_edges fixes the mode at the ends, while the PMC sides leave the 2D
    # field free there: the coupling meets the mode's zero end values only
    model = line(mesh, pec_edges=("side",))
    mode = gfem.solve(model, gfem.PortMode("p1", F, polarization="Ez"), options=OPTIONS)
    ends = np.asarray(mode.e_field.at(np.array([[0.0, 0.0], [0.0, LENGTH * 1e-3]])))
    middle = np.asarray(mode.e_field.at(np.array([[0.0, LENGTH * 0.5e-3]])))
    assert np.abs(ends).max() <= 1e-12 * np.abs(middle).max()
    result = gfem.solve(model, gfem.Driven(F, polarization="Ez"), options=OPTIONS)
    s = np.asarray(result.s.values)[0]
    assert np.isfinite(s).all()
    # passive (the 2D field's other modes take some power) and reciprocal
    assert ((np.abs(s) ** 2).sum(axis=0) <= 1 + 1e-9).all()
    assert s[0, 1] == pytest.approx(s[1, 0], abs=1e-9)
