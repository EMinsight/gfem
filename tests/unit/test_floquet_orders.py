"""Floquet ports before any solve: the order table, the helpers and the model checks.

The pure functions of `gfem._floquet` (lattice frame, diffraction orders,
DtN values, wave impedances, the modelled set and the order cap; ADRs
0147, 0152, 0153), `gfem.FloquetPort` construction, the `Model` checks of a
Floquet surface (ticket 08 Q6 of the floquet-ports spec), the
`Model.bloch_wave_vector` and `Model.propagating_orders` helpers, and the
codec. The cells are `structured_meshes.periodic_cell` boxes in um, 1.2 x
0.4 um, layered along z: substrate (eps 2.25) below z = 0.3 um, a slab
(eps 4) up to z = 0.5 um and air above, with the open faces z_min, z_max.
"""

import dataclasses
import math
import warnings

import jax
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem import _floquet as fl
from gfem._periodic import Pairing

UM = 1e-6
AX, AY = 1.2 * UM, 0.4 * UM
ETA0 = const.mu_0 * const.c
#: λ0 = 1 um.
F0 = const.c / UM
K0 = 2 * np.pi / UM
X_EDGES = np.linspace(0, 1.2, 7)
Y_EDGES = np.linspace(0, 0.4, 3)
Z_EDGES = np.array([0, 0.15, 0.3, 0.4, 0.5, 0.75, 1.0])
MATERIALS = {
    "sub": gfem.Material(eps_r=2.25),
    "slab": gfem.Material(eps_r=4.0),
    "air": gfem.Material(),
}


def _layers(c):
    return "sub" if c[2] < 0.3 else ("slab" if c[2] < 0.5 else "air")


def _mesh(tmp_path, volume=_layers, surface=None, name="cell"):
    cell = sm.periodic_cell((X_EDGES, Y_EDGES, Z_EDGES), volume, surface=surface)
    return gfem.Mesh.from_gmsh(sm.write_msh22(tmp_path / f"{name}.msh", cell))


def _pairs(*axes):
    return [gfem.Periodic((f"{a}_min", f"{a}_max")) for a in axes]


def _ports(top="z_max", bottom="z_min"):
    return [
        gfem.FloquetPort("R", top, polarization="TE"),
        gfem.FloquetPort("T", bottom, polarization="TE"),
    ]


def _model(mesh, boundaries=None, ports=None, materials=None):
    return gfem.Model(
        mesh,
        materials=MATERIALS if materials is None else materials,
        boundaries=_pairs("x", "y") if boundaries is None else boundaries,
        ports=_ports() if ports is None else ports,
    )


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return _mesh(tmp_path_factory.mktemp("floquet"))


@pytest.fixture(scope="module")
def model(mesh):
    return _model(mesh)


# --- lattice and orders -------------------------------------------------------


def test_orders_on_a_rectangular_lattice_add_the_reciprocal_vectors():
    kt = (1e6, 2e6, 0.0)
    rows = fl.order_rows(
        [(2, -1), (0, 0)], F0, kt, a1=(AX, 0, 0), a2=(0, AY, 0), eps_r=1.0
    )
    expected = np.array([1e6 + 2 * 2 * np.pi / AX, 2e6 - 2 * np.pi / AY, 0.0])
    np.testing.assert_allclose(rows.wave_vector[0, 0], expected, rtol=1e-14)
    np.testing.assert_allclose(rows.wave_vector[0, 1], kt, rtol=1e-14)


def test_orders_on_a_hexagonal_lattice_use_its_reciprocal_lattice():
    a = 0.5 * UM
    rows = fl.order_rows(
        [(1, 0), (0, 1), (1, 1)],
        F0,
        None,
        a1=(a, 0, 0),
        a2=(a / 2, a * math.sqrt(3) / 2, 0),
        eps_r=1.0,
    )
    g = 2 * np.pi / a
    b1 = g * np.array([1, -1 / math.sqrt(3), 0])
    b2 = g * np.array([0, 2 / math.sqrt(3), 0])
    np.testing.assert_allclose(
        rows.wave_vector[0], [b1, b2, b1 + b2], rtol=0, atol=1e-9 * g
    )


def test_gamma_is_i_beta_when_propagating_alpha_when_evanescent_and_zero_at_grazing():
    k = K0  # air at F0
    # kt along x of exactly |k|: the specular order grazes
    rows = fl.order_rows(
        [(0, 0), (1, 0), (-1, 0)],
        F0,
        (float(fl.wavenumber(F0, 1, 1)), 0, 0),
        a1=(AX, 0, 0),
        a2=(0, AY, 0),
        eps_r=1.0,
    )
    gamma = rows.gamma[0]
    assert gamma[0] == 0
    assert gamma[1].imag == 0
    assert gamma[1].real > 0
    np.testing.assert_allclose(
        gamma[1].real, math.sqrt((k + 2 * np.pi / AX) ** 2 - k**2), rtol=1e-12
    )
    # κ = k - 2π/AX = k / 6 along +x
    assert gamma[2].real == 0
    assert gamma[2].imag > 0
    np.testing.assert_allclose(
        gamma[2].imag, math.sqrt(k**2 - (k - 2 * np.pi / AX) ** 2), rtol=1e-12
    )
    # strictly: a grazing order does not propagate
    assert rows.propagating[0].tolist() == [False, False, True]
    assert rows.theta[0, 0] == np.pi / 2
    assert rows.phi[0, 0] == 0.0
    assert np.isnan(rows.theta[0, 1])
    assert np.isnan(rows.phi[0, 1])
    np.testing.assert_allclose(
        rows.theta[0, 2], math.asin(abs(k - 2 * np.pi / AX) / k), rtol=1e-12
    )
    assert rows.phi[0, 2] == 0.0


def test_the_frame_follows_the_pairs_and_takes_e1_at_normal_incidence():
    rows = fl.order_rows(
        [(0, 0), (0, 1)], F0, None, a1=(AX, 0, 0), a2=(0, AY, 0), eps_r=1.0
    )
    # κ = 0: TM along ê1 = x, TE along ê3 x ê1 = y
    np.testing.assert_array_equal(rows.tm[0, 0], [1, 0, 0])
    np.testing.assert_array_equal(rows.te[0, 0], [0, 1, 0])
    # κ along y: TM along y, TE along z x y = -x
    np.testing.assert_allclose(rows.tm[0, 1], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(rows.te[0, 1], [-1, 0, 0], atol=1e-15)
    # pairs listed (y, x): ê1 = y, ê3 = y x x = -z
    swapped = fl.order_rows([(0, 0)], F0, None, a1=(0, AY, 0), a2=(AX, 0, 0), eps_r=1.0)
    np.testing.assert_array_equal(swapped.tm[0, 0], [0, 1, 0])
    np.testing.assert_allclose(swapped.te[0, 0], [1, 0, 0], atol=0)


def test_the_frame_is_continuous_along_phi_zero():
    for theta in (0.0, 1e-9, 0.3):
        k = fl.wavenumber(F0, 1, 1) * math.sin(theta)
        rows = fl.order_rows(
            [(0, 0)], F0, (float(k), 0, 0), a1=(AX, 0, 0), a2=(0, AY, 0), eps_r=1.0
        )
        np.testing.assert_allclose(rows.tm[0, 0], [1, 0, 0], atol=1e-15)
        np.testing.assert_allclose(rows.te[0, 0], [0, 1, 0], atol=1e-15)


def test_dtn_values_and_the_infinite_tm_flag_at_grazing():
    # propagating gamma = 2i, k = 3, μ_r = 1: λ_TE = iβ = 2i, λ_TM = i k²/β = 4.5i
    for gamma, te, tm in ((2j, 2j, 4.5j), (2.0, 2.0, -4.5)):
        assert fl.dtn(gamma, 3.0, 1.0, "TE")[0] == pytest.approx(te)
        assert fl.dtn(gamma, 3.0, 1.0, "TM")[0] == pytest.approx(tm)
    assert fl.dtn(2j, 3.0, 2.0, "TE")[0] == pytest.approx(1j)
    lam, infinite = fl.dtn(np.array([0.0, 2j]), 3.0, 1.0, "TM")
    assert infinite.tolist() == [True, False]
    assert lam[0].real == np.inf
    assert lam[0].imag == 0
    assert not np.isnan(lam).any()
    lam_te, flag_te = fl.dtn(0.0, 3.0, 1.0, "TE")
    assert lam_te == 0
    assert not flag_te


def test_wave_impedances_are_eta_over_cos_and_eta_cos_and_nan_off_propagation():
    # k = 1, gamma = 0.5i: cos θ = 0.5; eps_r = 4: η = η0 / 2
    eta = ETA0 / 2
    assert fl.wave_impedance(0.5j, 1.0, 4.0, 1.0, "TE") == pytest.approx(2 * eta)
    assert fl.wave_impedance(0.5j, 1.0, 4.0, 1.0, "TM") == pytest.approx(eta / 2)
    for gamma in (0.0, 0.5):
        assert np.isnan(fl.wave_impedance(gamma, 1.0, 4.0, 1.0, "TE"))
        assert np.isnan(fl.wave_impedance(gamma, 1.0, 4.0, 1.0, "TM"))


# --- the modelled set --------------------------------------------------------------


def _modelled(f, kt, distance=0.2 * UM, tol=1e-4, cap=(20, 20)):
    """The modelled set of an air face on the AX x AY lattice, d from a scatterer."""
    face = fl.FloquetFace(
        surface="top",
        region="air",
        eps_r=1.0,
        mu_r=1.0,
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        a1=np.array([AX, 0.0, 0.0]),
        a2=np.array([0.0, AY, 0.0]),
        pairs=(),  # ty: ignore[invalid-argument-type]
        area=AX * AY,
        h_face=0.1 * UM,
        distance=distance,
    )
    return fl.modelled_orders(face, f, kt, tolerance=tol, cap=cap)


def _rule(f, kt, distance, tol):
    """The truncation rule on the rectangular lattice, from its closed form."""
    k = 2 * np.pi * f / const.c
    out = set()
    for m in range(-40, 41):
        for n in range(-40, 41):
            kx = kt[0] + 2 * np.pi * m / AX
            ky = kt[1] + 2 * np.pi * n / AY
            alpha = math.sqrt(max(kx**2 + ky**2 - k**2, 0))
            if math.exp(-2 * alpha * distance) > tol:
                out.add((m, n))
    return out


def test_the_modelled_set_is_the_rule_and_the_union_over_the_grid():
    kt = (0.3 * K0, 0.1 * K0, 0.0)
    for f in (F0, 1.3 * F0):
        assert set(_modelled(f, kt).orders) == _rule(f, kt, 0.2 * UM, 1e-4)
    union = _rule(F0, kt, 0.2 * UM, 1e-4) | _rule(1.3 * F0, kt, 0.2 * UM, 1e-4)
    t = _modelled([F0, 1.3 * F0], kt)
    assert set(t.orders) == union
    assert t.orders == fl.sort_orders(union)
    assert not t.cut
    # the propagating orders are in, whatever the distance
    assert {(0, 0), (-1, 0)} <= set(_modelled(F0, kt, distance=math.inf).orders)


def test_the_modelled_set_is_symmetric_at_normal_incidence_and_mirrors_under_minus_k():
    plain = set(_modelled([F0, 1.2 * F0], None).orders)
    assert plain == {(-m, -n) for m, n in plain}
    assert len(plain) > 3
    kt = np.array([0.3 * K0, 0.1 * K0, 0.0])
    forward = set(_modelled([F0, 1.2 * F0], kt).orders)
    backward = set(_modelled([F0, 1.2 * F0], -kt).orders)
    assert backward == {(-m, -n) for m, n in forward}
    # propagating only (no scatterer): the oblique set is not symmetric itself
    forward = set(_modelled(F0, kt, distance=math.inf).orders)
    backward = set(_modelled(F0, -kt, distance=math.inf).orders)
    assert forward == {(0, 0), (-1, 0)}
    assert backward == {(0, 0), (1, 0)}


def test_the_order_cap_cuts_the_rule_and_names_the_first_orders_past_it():
    b = 2 * np.pi / UM
    # pπ/(|b| h) = 2π/(2π 1e6 · 0.12e-6) = 8.33
    assert fl.order_cap((b, 0, 0), (0, b, 0), 0.12 * UM, 2) == (8, 8)
    t = _modelled(F0, None, distance=0.01 * UM, cap=(2, 1))
    assert max(abs(m) for m, _ in t.orders) == 2
    assert max(abs(n) for _, n in t.orders) == 1
    assert (3, 0) in t.cut
    # the slowest-decaying orders past the cap: (±3, 0)
    alpha = math.sqrt((3 * 2 * np.pi / AX) ** 2 - K0**2)
    assert t.alpha_cap == pytest.approx(alpha, rel=1e-12)


# --- FloquetPort ------------------------------------------------------------------


def test_a_floquet_port_defaults_to_the_specular_order_and_needs_a_polarization():
    port = gfem.FloquetPort("in", "top", polarization="TM")
    assert (port.order, port.polarization, port.offset) == ((0, 0), "TM", 0.0)
    assert gfem.FloquetPort("o", "top", order=[-1, 2], polarization="TE").order == (
        -1,
        2,
    )
    with pytest.raises(TypeError, match="polarization"):
        gfem.FloquetPort("in", "top")  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        gfem.FloquetPort("in", "top", (0, 0), "TE")  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"name": 3}, TypeError, "name must be a string"),
        ({"name": ""}, ValueError, "name must not be empty"),
        ({"surface": ("a", "b")}, TypeError, "surface must be one surface name"),
        ({"polarization": "s"}, ValueError, r"polarization must be 'TE' or 'TM'"),
        ({"order": (True, 0)}, TypeError, r"order must be two integers \(m, n\)"),
        ({"order": (1.0, 0)}, TypeError, "order must be two integers"),
        ({"order": (1, 0, 0)}, TypeError, "order must be two integers"),
        ({"order": 1}, TypeError, "order must be two integers"),
        ({"offset": -1e-6}, ValueError, "offset must be >= 0 m"),
        ({"offset": math.inf}, ValueError, "must be finite"),
    ],
)
def test_floquet_port_construction_checks(kwargs, error, match):
    args = {"name": "p", "surface": "top", "polarization": "TE", **kwargs}
    name, surface = args.pop("name"), args.pop("surface")
    with pytest.raises(error, match=match):
        gfem.FloquetPort(name, surface, **args)


def test_a_floquet_port_offset_is_static():
    def f(x):
        gfem.FloquetPort("p", "top", polarization="TE", offset=x)
        return x

    with pytest.raises(gfem.NonDifferentiableError, match=r"FloquetPort\.offset"):
        jax.grad(f)(1e-6)


# --- the face, the order table and the cap warning ------------------------------


def test_the_mesh_distance_is_the_geometric_distance_to_the_structure(model):
    table = fl.order_table(model, [F0], None, gfem.SolverOptions())
    top, bottom = table["z_max"].face, table["z_min"].face
    assert top.distance == pytest.approx(0.5 * UM, rel=1e-12)
    assert bottom.distance == pytest.approx(0.3 * UM, rel=1e-12)
    assert (top.region, top.eps_r, bottom.region, bottom.eps_r) == (
        "air",
        1.0,
        "sub",
        2.25,
    )
    np.testing.assert_allclose(top.normal, [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(bottom.normal, [0, 0, 1], atol=1e-15)
    assert top.area == pytest.approx(AX * AY, rel=1e-12)
    assert top.h_face == pytest.approx(math.hypot(0.2, 0.2) * UM, rel=1e-12)
    np.testing.assert_allclose(top.a1, (AX, 0, 0), atol=1e-12 * UM)
    np.testing.assert_allclose(top.a2, (0, AY, 0), atol=1e-12 * UM)


def test_the_order_table_holds_the_modelled_and_declared_orders(mesh):
    ports = [
        *_ports(),
        gfem.FloquetPort("far", "z_max", order=(9, 0), polarization="TM"),
    ]
    model = _model(mesh, ports=ports)
    options = gfem.SolverOptions(order=3)
    table = fl.order_table(model, [F0, 1.1 * F0], None, options)
    top = table["z_max"]
    assert top.ports == (ports[0], ports[2])
    assert top.truncation.cap == top.face.cap(3)
    assert (9, 0) in top.orders
    assert not top.modelled[top.orders.index((9, 0))]
    assert set(top.truncation.orders) == {
        o for o, m in zip(top.orders, top.modelled, strict=True) if m
    }
    assert (
        top.lam.shape
        == top.infinite.shape
        == top.impedance.shape
        == (2, len(top.orders), 2)
    )
    i = top.orders.index((0, 0))
    np.testing.assert_allclose(
        top.lam[:, i, 0], 1j * fl.wavenumber([F0, 1.1 * F0], 1, 1)
    )
    np.testing.assert_allclose(top.impedance[:, i], ETA0, rtol=1e-12)
    # the parent and a worker computing the table again get the same bits
    again = fl.order_table(model, [F0, 1.1 * F0], None, options)["z_max"]
    np.testing.assert_array_equal(again.lam, top.lam)
    assert again.orders == top.orders


def test_the_cap_warning_names_the_surface_and_the_distance_to_move_it_to(mesh):
    model = _model(mesh)
    table = fl.order_table(model, [F0], None, gfem.SolverOptions(order=1))
    cut = [s for s, t in table.items() if t.truncation.cut]
    assert cut == ["z_min"]  # 0.3 um from the slab, h = 0.28 um, order 1
    t = table["z_min"].truncation
    reach = math.log(1e4) / (2 * t.alpha_cap)
    with pytest.warns(gfem.FloquetOrderCapWarning) as record:
        fl.warn_order_cap(table, 1e-4)
    assert len(record) == 1
    text = str(record[0].message)
    assert text.startswith("Floquet surface 'z_min': its face mesh resolves")
    assert f"Move 'z_min' out to {reach:.3g} m" in text
    assert "or refine the mesh of 'z_min'" in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fl.warn_order_cap(fl.order_table(model, [F0], None, gfem.SolverOptions()), 1e-4)


# --- Model helpers ----------------------------------------------------------------


def test_bloch_wave_vector_reads_the_index_behind_the_face(model):
    theta, phi = 0.3, 0.2
    direction = np.array([math.cos(phi), math.sin(phi), 0.0])
    k = model.bloch_wave_vector("z_max", F0, theta=theta, phi=phi)
    np.testing.assert_allclose(k, K0 * math.sin(theta) * direction, rtol=1e-14)
    k = model.bloch_wave_vector("z_min", F0, theta=theta, phi=phi)
    np.testing.assert_allclose(k, 1.5 * K0 * math.sin(theta) * direction, rtol=1e-14)
    assert model.bloch_wave_vector("z_max", F0, theta=0.0) == (0.0, 0.0, 0.0)
    # works on a cell with no Floquet port yet
    bare = _model(model.mesh, ports=[])
    assert bare.bloch_wave_vector("z_max", F0, theta=0.1) == model.bloch_wave_vector(
        "z_max", F0, theta=0.1
    )


def test_bloch_wave_vector_refuses_a_traced_angle(model):
    def f(theta):
        return model.bloch_wave_vector("z_max", F0, theta=theta)[0]

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"Model\.bloch_wave_vector: theta and phi must be fixed numbers: the "
        r"incidence of a Floquet port is a constant of the solve, not a design "
        r"variable",
    ):
        jax.grad(f)(0.3)


def test_bloch_wave_vector_checks_its_numbers(model):
    with pytest.raises(ValueError, match=r"theta must be in \[-pi/2, pi/2\] rad"):
        model.bloch_wave_vector("z_max", F0, theta=2.0)
    with pytest.raises(ValueError, match="frequency must be > 0 Hz"):
        model.bloch_wave_vector("z_max", -F0, theta=0.1)
    with pytest.raises(ValueError, match="is a volume, not a surface"):
        model.bloch_wave_vector("air", F0, theta=0.1)


def test_propagating_orders_are_sorted_and_unite_the_frequencies(model):
    assert model.propagating_orders("z_max", F0) == ((0, 0), (-1, 0), (1, 0))
    kt = (0.5 * K0, 0.0, 0.0)
    # air: |0.5 + m/1.2| < 1 for m = 0, -1
    assert model.propagating_orders("z_max", F0, wave_vector=kt) == ((0, 0), (-1, 0))
    # substrate, n = 1.5: m = -2..1
    assert model.propagating_orders("z_min", F0, wave_vector=kt) == (
        (0, 0),
        (-1, 0),
        (1, 0),
        (-2, 0),
    )
    # 1.3 f0 in air adds m = -2 (|0.5 - 2/1.2| < 1.3), not m = 1 (1.33)
    assert model.propagating_orders("z_max", [F0, 1.3 * F0], wave_vector=kt) == (
        (0, 0),
        (-1, 0),
        (-2, 0),
    )
    with pytest.raises(ValueError, match="outside the span"):
        model.propagating_orders("z_max", F0, wave_vector=(0, 0, 1e6))


# --- Model-time checks of a Floquet surface ------------------------------------------


def test_floquet_ports_need_a_periodic_pair(mesh):
    with pytest.raises(
        ValueError,
        match=r"FloquetPort 'R' on surface 'z_max': the model has no gfem\.Periodic "
        r"pair, but a Floquet port's diffraction orders are those of a periodic "
        r"cell\. Pair the surfaces that meet 'z_max' with gfem\.Periodic, or end a "
        r"closed model with gfem\.WavePort or gfem\.Absorbing\.",
    ):
        _model(mesh, boundaries=[])


def test_a_pair_along_the_normal_closes_the_face(tmp_path):
    # z_max and z_min split at x = 0.6: "top"/"bot" for x < 0.6, a z pair beyond
    def surface(c):
        if c[0] < 0.6 and abs(c[2] - 1.0) < 1e-9:
            return "top"
        if c[0] < 0.6 and abs(c[2]) < 1e-9:
            return "bot"
        return ""

    mesh = _mesh(tmp_path, surface=surface)
    with pytest.raises(
        ValueError,
        match=r"FloquetPort 'R' on surface 'top': Periodic\(\('z_min', 'z_max'\)\) "
        r"translates along the normal of 'top', so 'top' is not an open face of the "
        r"cell\. Remove that pair: a Floquet port needs the cell open along its "
        r"normal\.",
    ):
        _model(
            mesh,
            boundaries=_pairs("x", "y", "z"),
            ports=[gfem.FloquetPort("R", "top", polarization="TE")],
        )


def test_rim_edges_off_the_pairs_are_refused(tmp_path):
    def surface(c):
        if abs(c[2] - 1.0) < 1e-9:
            return "top" if c[0] < 0.6 else "lid"
        return ""

    mesh = _mesh(tmp_path, surface=surface)
    with pytest.raises(
        gfem.PortGeometryError,
        match=r"FloquetPort 'R' on surface 'top': 2 edges of its rim lie on no "
        r"periodic pair's surface \(one at \(6e-07, [0-9.e-]+, 1e-06\) m\)\. A Floquet "
        r"surface spans the whole cell, its rim on the surfaces of two "
        r"gfem\.Periodic pairs: make 'top' one planar face across the cell, with no "
        r"other boundary or structure on it\.",
    ):
        _model(mesh, ports=_ports(top="top"))


def test_a_rim_on_one_pair_and_walls_is_not_implemented(mesh):
    with pytest.raises(
        NotImplementedError,
        match=r"FloquetPort 'R' on surface 'z_max': its rim lies on one periodic "
        r"pair \(Periodic\(\('x_min', 'x_max'\)\)\) and on 'y_max', 'y_min'\. In this "
        r"version of gfem a Floquet port needs a cell periodic along both directions "
        r"of its face: pair the two walls with gfem\.Periodic as well, keep the cell "
        r"one element thick across them, and give the wave vector no component along "
        r"that pair\. A 1D grating is modelled this way\.",
    ):
        _model(mesh, boundaries=_pairs("x"))


def test_rim_twins_off_the_face_are_refused(model):
    # A conforming planar face whose rim lies on two pairs always meets the twin
    # edges, so the check is fed a pairing shifted one z layer down on x_max.
    x = model.mesh.imported_nodes
    p = model._periodic[0]
    first = x[p.first_nodes]
    second = x[p.second_nodes]
    zs = np.unique(np.round(first[:, 2] / UM, 9))
    lower = {z: zs[max(i - 1, 0)] if i else zs[-1] for i, z in enumerate(zs)}
    index = {
        (round(c[1] / UM, 9), round(c[2] / UM, 9)): n
        for c, n in zip(first, p.first_nodes, strict=True)
    }
    shifted = np.array(
        [index[(round(c[1] / UM, 9), lower[round(c[2] / UM, 9)])] for c in second]
    )
    bad = dataclasses.replace(p, first_nodes=shifted)
    assert isinstance(bad, Pairing)
    with pytest.raises(
        gfem.PortGeometryError,
        match=r"FloquetPort 'R' on surface 'z_max': it meets 'x_min' along 2 edges "
        r"whose twins on 'x_max' are not edges of 'z_max', so the face is not one "
        r"period of the cell\. Mesh 'z_max' across the whole cell\.",
    ):
        fl.floquet_face(
            model,
            "z_max",
            "FloquetPort 'R' on surface 'z_max'",
            [bad, model._periodic[1]],
        )


def test_several_regions_behind_the_face_are_not_implemented(tmp_path):
    def volume(c):
        return "air2" if c[2] > 0.75 and c[0] < 0.6 else _layers(c)

    mesh = _mesh(tmp_path, volume=volume)
    with pytest.raises(
        NotImplementedError,
        match=r"FloquetPort 'R' on surface 'z_max': the elements behind 'z_max' "
        r"belong to regions '.*', '.*'\. In this version of gfem a Floquet port "
        r"needs one homogeneous region behind its face, because its orders are plane "
        r"waves in that medium\. Move the face off the structure, leaving a layer of "
        r"one material between them\.",
    ):
        _model(mesh, materials={**MATERIALS, "air2": gfem.Material()})


@pytest.mark.parametrize(
    ("material", "what"),
    [
        (gfem.Material(conductivity=1.0), "a conductivity"),
        (gfem.Material(tan_delta=1e-3), "a loss tangent"),
        (gfem.Material(eps_r=(1.0, 1.0, 2.0)), "anisotropic eps_r or mu_r"),
        (
            gfem.Material(
                eps_r=gfem.PermittivityLaw(
                    terms=[gfem.Drude(plasma_frequency=1e15, damping=1e13)]
                )
            ),
            "a permittivity law",
        ),
        ("field", "a MaterialField"),
    ],
)
def test_a_lossy_anisotropic_dispersive_or_field_half_space_is_not_implemented(
    mesh, material, what
):
    if material == "field":
        material = gfem.MaterialField.uniform("air", gfem.Material())
    with pytest.raises(
        NotImplementedError,
        match=rf"FloquetPort 'R' on surface 'z_max': region 'air' behind it has "
        rf"{what}\. In this version of gfem a Floquet port needs a lossless isotropic "
        r"material with constant real eps_r and mu_r behind its face\. Put a "
        r"lossless layer between the face and that material\.",
    ):
        _model(mesh, materials={**MATERIALS, "air": material})


def test_a_floquet_surface_takes_only_floquet_ports_each_its_own_order(mesh):
    with pytest.raises(ValueError, match="more than one electromagnetic boundary"):
        _model(
            mesh,
            boundaries=[*_pairs("x", "y"), gfem.Absorbing("z_max")],
            ports=_ports(),
        )
    with pytest.raises(ValueError, match="more than one electromagnetic boundary"):
        _model(
            mesh,
            ports=[*_ports(), gfem.WavePort("w", "z_max")],
        )
    with pytest.raises(
        ValueError,
        match=r"FloquetPorts 'R' and 'R2' on surface 'z_max' both request order "
        r"\(0, 0\) in TE; give each Floquet port on a surface a different \(order, "
        r"polarization\)\.",
    ):
        _model(
            mesh,
            ports=[*_ports(), gfem.FloquetPort("R2", "z_max", polarization="TE")],
        )
    model = _model(
        mesh,
        ports=[
            *_ports(),
            gfem.FloquetPort("R_TM", "z_max", polarization="TM"),
            gfem.FloquetPort("R1", "z_max", order=(1, 0), polarization="TE"),
        ],
    )
    assert model.port_names == ("R", "T", "R_TM", "R1")


# --- options and codec ------------------------------------------------------------


def test_the_order_tolerance_option():
    assert gfem.SolverOptions().floquet_order_tolerance == 1e-4
    with pytest.raises(ValueError, match="floquet_order_tolerance must be in"):
        gfem.SolverOptions(floquet_order_tolerance=0.0)


def test_floquet_ports_round_trip_through_a_job_file(model, tmp_path):
    ports = (
        *model.ports,
        gfem.FloquetPort("R1", "z_max", order=(1, -2), polarization="TM", offset=2e-7),
    )
    model = model.replace(ports=ports)
    options = gfem.SolverOptions(floquet_order_tolerance=1e-3)
    path = gfem.Job(model, gfem.Driven(F0), options).to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path)
    assert back.model.ports == ports
    assert back.model.ports[2].order == (1, -2)
    assert back.options == options


def test_the_warning_classes_are_exported():
    for name in (
        "UnrequestedOrderWarning",
        "ObliqueAbsorberWarning",
        "FloquetOrderCapWarning",
    ):
        assert issubclass(getattr(gfem, name), UserWarning)
        assert name in gfem.__all__
    assert "FloquetPort" in gfem.__all__
