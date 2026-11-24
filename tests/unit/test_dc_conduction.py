"""`gfem.DCConduction`, `gfem.joule_heating` and `gfem.HeatSource`.

Exact checks: a straight bar's potential is linear, so every H1
space holds it and R = L / (sigma A) to round-off with a uniform Joule
density; on a cross-section slab with terminals on its end faces the
potential is linear through the thickness, so the solve reproduces the
longitudinal model of gsim, R' = 1 / sum_i sigma_i A_i and q_i = sigma_i
(I R')^2, at any order and thickness.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem

UM = 1e-6
SIGMA = 1e6
L, W, T = 20.0, 2.0, 0.2  # bar, um
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
TERMINALS = {"a": "end_a"}
OXIDE = gfem.Material(eps_r=3.9)
METAL = gfem.Material(conductivity=SIGMA)


def _bar_mesh(tmp_path, *, island=None, name="bar.msh"):
    """A 20 x 2 x 0.2 um bar in an oxide box; contacts "end_a" (x = 0), "end_b".

    ``island``: ``None``, or the region name of a 2 x 1 x 0.2 um block
    beside the bar touching no contact (the bar's own name puts it in the
    bar's gmsh entity).
    """
    edges = (
        np.linspace(0.0, L, 11),
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, T, 0.6]),
    )
    tol = 1e-9

    def volume(c):
        if c[2] < T and 0 < c[1] < W:
            return "bar"
        if island and c[2] < T and c[1] > W + 1 and 8 < c[0] < 12 and c[1] < 4:
            return island
        return "oxide"

    def surface(c):
        in_bar = c[2] < T and 0 < c[1] < W
        if abs(c[0]) < tol and in_bar:
            return "end_a"
        if abs(c[0] - L) < tol and in_bar:
            return "end_b"
        if abs(c[2]) < tol and not in_bar:
            return "floor"
        return "box" if abs(c[2] - 0.6) < tol or abs(c[1] - 4) < tol else ""

    return gfem.Mesh.from_gmsh(
        sm.write_msh22(tmp_path / name, sm.structured(edges, volume, surface))
    )


@pytest.fixture(scope="module")
def bar_mesh(tmp_path_factory):
    return _bar_mesh(tmp_path_factory.mktemp("bar"))


def _bar(mesh, metal=METAL, **kw):
    return gfem.Model(mesh, materials={"bar": metal, "oxide": OXIDE}, **kw)


def _solve(model, order=2, ground="end_b"):
    problem = gfem.DCConduction(terminals=TERMINALS, ground=ground)
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(model, problem, options=options)


def _volumes(mesh):
    p = np.asarray(mesh.nodes)[mesh.tetrahedra]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6


# --- the straight bar ---------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_a_bar_has_r_l_over_sigma_a_and_a_uniform_joule_density(bar_mesh, order):
    dc = _solve(_bar(bar_mesh), order)
    expected = (L * UM) / (SIGMA * W * UM * T * UM)  # 50 ohm
    assert expected == pytest.approx(50.0)
    assert float(dc.resistance[0, 0]) == pytest.approx(expected, rel=1e-12)
    q = np.asarray(dc.dissipation[:, 0, 0])
    inside = np.zeros(len(q), bool)
    inside[bar_mesh.elements("bar")] = True
    # per element, the solve's round-off (1.1e-12 measured at order 2)
    np.testing.assert_allclose(q[inside], SIGMA / (L * UM) ** 2, rtol=1e-11)
    assert not q[~inside].any()
    assert dc.terminals == ("a",)
    assert dc.index("a") == 0
    assert "resistance 50 ohm" in repr(dc)


def test_joule_power_is_conserved(bar_mesh):
    dc = _solve(_bar(bar_mesh))
    heat = gfem.joule_heating(dc, current=10e-3)
    density = heat._density_on(bar_mesh)
    total = float(jnp.sum(density * _volumes(bar_mesh)))
    assert total == pytest.approx((10e-3) ** 2 * 50.0, rel=1e-12)
    assert heat.regions == ("bar",)
    assert "total 0.005 W" in repr(heat)


def test_the_three_drives_agree(bar_mesh):
    dc = _solve(_bar(bar_mesh))
    r = float(dc.resistance[0, 0])
    q1 = np.asarray(dc.dissipation[:, 0, 0])
    by_voltage = gfem.joule_heating(dc, voltage=3.0)._density_on(bar_mesh)
    np.testing.assert_allclose(by_voltage, 9.0 * q1, rtol=1e-14)
    # 10 mA through 50 ohm is 0.5 V, as is 5 mW
    by_current = gfem.joule_heating(dc, current=10e-3)._density_on(bar_mesh)
    by_power = gfem.joule_heating(dc, power=5e-3)._density_on(bar_mesh)
    np.testing.assert_allclose(by_current, (10e-3 * r) ** 2 * q1, rtol=1e-12)
    np.testing.assert_allclose(by_power, by_current, rtol=1e-12)
    by_mapping = gfem.joule_heating(dc, voltage={"a": 3.0})._density_on(bar_mesh)
    np.testing.assert_allclose(by_mapping, by_voltage, rtol=1e-14)


def test_the_drive_is_differentiable(bar_mesh):
    dc = _solve(_bar(bar_mesh))
    vol = _volumes(bar_mesh)

    def power(i):
        heat = gfem.joule_heating(dc, current=i)
        return jnp.sum(heat._density_on(bar_mesh) * vol)

    # 1.1e-12 off on a GitHub runner without PARDISO: the solve's rounding
    assert float(jax.grad(power)(0.01)) == pytest.approx(2 * 0.01 * 50.0, rel=1e-11)


def test_the_conductance_is_differentiable_in_the_conductivity(bar_mesh):
    n = len(bar_mesh.elements("bar"))
    rng = np.random.default_rng(1)
    base = SIGMA * (1 + 0.3 * rng.random(n))

    def g(sigma):
        field = gfem.MaterialField(conductivity=sigma, region="bar")
        dc = _solve(_bar(bar_mesh, metal=field), order=1)
        return dc.conductance[0, 0] * 1e3

    assert_gradient_matches(
        g,
        base,
        unit_directions(np.random.default_rng(3), n, 2),
        step=1e-3 * SIGMA,
        rel=1e-6,
    )


# --- the conduction domain ----------------------------------------------------


@pytest.mark.parametrize("island", ["island", "bar"])
def test_a_floating_conductor_carries_nothing(tmp_path, island):
    mesh = _bar_mesh(tmp_path, island=island)
    materials = {"bar": METAL, "oxide": OXIDE} | (
        {"island": gfem.Material(conductivity=3 * SIGMA)} if island == "island" else {}
    )
    dc = _solve(gfem.Model(mesh, materials=materials))
    assert float(dc.resistance[0, 0]) == pytest.approx(50.0, rel=1e-12)
    q = np.asarray(dc.dissipation[:, 0, 0])
    block = np.asarray(mesh.nodes)[mesh.tetrahedra].mean(axis=1)[:, 1] > (W + 1) * UM
    conductor = np.zeros(len(q), bool)
    conductor[mesh.elements(island)] = True
    assert (conductor & block).any()
    assert not q[conductor & block].any()
    assert dc._regions == ("bar",)


def test_a_substrate_excluded_from_dc_conduction_leaves_the_circuit(tmp_path):
    mesh = _bar_mesh(tmp_path, island="island")
    doped = gfem.Material(eps_r=11.7, conductivity=10.0)
    model = gfem.Model(
        mesh,
        materials={"bar": METAL, "oxide": OXIDE, "island": doped},
        excluded={"dc_conduction": ("island",)},
    )
    assert float(_solve(model).resistance[0, 0]) == pytest.approx(50.0, rel=1e-12)


def test_a_terminal_off_the_conductors_raises(bar_mesh):
    problem = gfem.DCConduction(terminals={"a": "box"}, ground="end_b")
    with pytest.raises(ValueError, match=r"terminal 'a' \('box'\) touches no conduct"):
        gfem.solve(_bar(bar_mesh), problem, options=DIRECT)


def test_no_conducting_volume_raises(bar_mesh):
    model = gfem.Model(bar_mesh, materials={"bar": OXIDE, "oxide": OXIDE})
    with pytest.raises(ValueError, match="no volume conducts"):
        _solve(model)


def test_a_zero_conductivity_element_raises_with_the_void_remedy(bar_mesh):
    sigma = np.full(len(bar_mesh.elements("bar")), SIGMA)
    sigma[3] = 0.0
    field = gfem.MaterialField(conductivity=sigma, region="bar")
    with pytest.raises(
        ValueError, match=r"1 element\(s\) of the conducting volume 'bar'.*void"
    ):
        _solve(_bar(bar_mesh, metal=field))


def test_touching_contacts_raise(bar_mesh):
    problem = gfem.DCConduction(terminals={"a": "end_a"}, ground=["end_b", "floor"])
    with pytest.raises(ValueError, match="touch"):
        gfem.solve(_bar(bar_mesh), problem, options=DIRECT)


def test_the_problem_validates_like_electrostatic():
    with pytest.raises(ValueError, match="DCConduction needs at least one terminal"):
        gfem.DCConduction(terminals={})
    with pytest.raises(ValueError, match="both in terminal"):
        gfem.DCConduction(terminals={"a": "x"}, ground="x")
    problem = gfem.DCConduction(["x", "y"], ground="g")
    assert problem.terminal_names == ("x", "y")
    assert (
        repr(problem)
        == "DCConduction(terminals={'x': ['x'], 'y': ['y']}, ground=['g'])"
    )


# --- drives and their refusals ------------------------------------------------


def test_joule_heating_refuses_a_driven_result():
    driven = object.__new__(gfem.DrivenResult)
    with pytest.raises(TypeError, match=r"gfem\.electromagnetic_heating\(result"):
        gfem.joule_heating(driven, current=1e-3)
    with pytest.raises(TypeError, match="DCConduction"):
        gfem.joule_heating(object(), current=1e-3)


def test_one_drive_exactly(bar_mesh):
    dc = _solve(_bar(bar_mesh))
    with pytest.raises(ValueError, match="exactly one drive"):
        gfem.joule_heating(dc)
    with pytest.raises(ValueError, match="exactly one drive"):
        gfem.joule_heating(dc, voltage=1.0, current=1.0)
    with pytest.raises(KeyError, match="not a terminal"):
        gfem.joule_heating(dc, voltage={"b": 1.0})


def test_power_per_length_on_a_3d_mesh_names_power(bar_mesh):
    dc = _solve(_bar(bar_mesh))
    with pytest.raises(ValueError, match=r"3D mesh: give the total power="):
        gfem.joule_heating(dc, power_per_length=1.0)
    with pytest.raises(ValueError, match="resistance_per_length is defined on a cross"):
        _ = dc.resistance_per_length
    source = gfem.HeatSource.uniform("bar", power_per_length=1.0)
    with pytest.raises(ValueError, match=r"3D: give the total power="):
        source._density_on(bar_mesh)


def test_power_needs_one_terminal_and_a_ground(bar_mesh):
    dc = _solve(_bar(bar_mesh), ground=())
    with pytest.raises(ValueError, match="one terminal and a ground"):
        gfem.joule_heating(dc, power=1e-3)


# --- HeatSource.uniform -------------------------------------------------------


def test_a_uniform_source_spreads_its_power_by_volume(bar_mesh):
    vol = _volumes(bar_mesh)
    source = gfem.HeatSource.uniform(["bar", "oxide"], power=2e-3)
    density = np.asarray(source._density_on(bar_mesh))
    np.testing.assert_allclose(density, 2e-3 / vol.sum(), rtol=1e-12)
    dense = gfem.HeatSource.uniform("bar", power_density=5.0)._density_on(bar_mesh)
    assert float(jnp.sum(dense * vol)) == pytest.approx(5.0 * L * W * T * UM**3)
    assert (
        repr(source) == "HeatSource(uniform, regions=['bar', 'oxide'], power=0.002 W)"
    )


def test_a_uniform_source_is_differentiable_in_its_amount(bar_mesh):
    vol = _volumes(bar_mesh)

    def total(p):
        return jnp.sum(
            gfem.HeatSource.uniform("bar", power=p)._density_on(bar_mesh) * vol
        )

    assert float(jax.grad(total)(1e-3)) == pytest.approx(1.0, rel=1e-12)


def test_a_uniform_source_checks_its_amount_and_regions(bar_mesh):
    with pytest.raises(ValueError, match="exactly one amount"):
        gfem.HeatSource.uniform("bar", power=1.0, power_density=1.0)
    with pytest.raises(ValueError, match="finite"):
        gfem.HeatSource.uniform("bar", power=math.inf)
    with pytest.raises(gfem.RegionNotFoundError):
        gfem.HeatSource.uniform("nowhere", power=1.0)._density_on(bar_mesh)
    with pytest.raises(ValueError, match="HeatFlux"):
        gfem.HeatSource.uniform("end_a", power=1.0)._density_on(bar_mesh)


def test_a_heat_source_is_a_pytree():
    source = gfem.HeatSource.uniform("bar", power=1.0)
    leaves, tree = jax.tree.flatten(source)
    assert len(leaves) == 1
    back = jax.tree.unflatten(tree, [jnp.asarray(2.0)])
    assert back.regions == ("bar",)
    assert float(back._amount) == 2.0


# --- the slab cross-section against gsim's longitudinal model -----------------

SIGMAS = {"h1": 1e6, "h2": 3e6}
#: The conductors' sections in um: x and y ranges.
BOXES = {"h1": ((1.0, 2.0), (1.0, 1.5)), "h2": ((4.0, 5.5), (1.0, 1.5))}


def _write_section(path):
    """A 6 x 3 um oxide section with conductors h1 and h2, msh 2.2 triangles."""
    xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0])
    ys = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0])
    nx = len(xs)
    nodes = [(x, y) for y in ys for x in xs]

    def n(i, j):
        return j * nx + i + 1

    tags = {"clad": 1, "h1": 2, "h2": 3, "bottom": 4}
    tris = []
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            name = next(
                (
                    k
                    for k, ((x0, x1), (y0, y1)) in BOXES.items()
                    if x0 < cx < x1 and y0 < cy < y1
                ),
                "clad",
            )
            a, b, c, d = n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)
            pair = ((a, b, c), (a, c, d)) if (i + j) % 2 else ((a, b, d), (b, c, d))
            tris += [(name, t) for t in pair]
    lines = [(n(i, 0), n(i + 1, 0)) for i in range(nx - 1)]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4"]
    out += ['1 4 "bottom"', '2 1 "clad"', '2 2 "h1"', '2 3 "h2"', "$EndPhysicalNames"]
    out += ["$Nodes", str(len(nodes))]
    out += [f"{k + 1} {float(x)!r} {float(y)!r} 0" for k, (x, y) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(tris))]
    k = 0
    for a, b in lines:
        k += 1
        out.append(f"{k} 1 2 4 4 {a} {b}")
    for name, (a, b, c) in tris:
        k += 1
        out.append(f"{k} 2 2 {tags[name]} {tags[name]} {a} {b} {c}")
    out.append("$EndElements")
    path.write_text("\n".join(out) + "\n")
    return path


@pytest.fixture(scope="module")
def section_path(tmp_path_factory):
    return _write_section(tmp_path_factory.mktemp("section") / "section.msh")


def _slab(path, thickness):
    mesh = gfem.Mesh.from_gmsh_cross_section(path, thickness=thickness)
    materials = {"clad": OXIDE} | {
        k: gfem.Material(conductivity=s) for k, s in SIGMAS.items()
    }
    model = gfem.Model(mesh, materials=materials)
    return mesh, model


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("thickness", [0.5e-6, 3e-6])
def test_a_slab_reproduces_the_longitudinal_model(section_path, order, thickness):
    mesh, model = _slab(section_path, thickness)
    problem = gfem.DCConduction(terminals={"heater": "front"}, ground="back")
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    dc = gfem.solve(model, problem, options=options)
    areas = {
        k: (x1 - x0) * (y1 - y0) * UM**2 for k, ((x0, x1), (y0, y1)) in BOXES.items()
    }
    r_per_length = 1 / sum(SIGMAS[k] * areas[k] for k in SIGMAS)
    assert float(dc.resistance_per_length[0, 0]) == pytest.approx(
        r_per_length, rel=1e-12
    )
    current = 25e-3
    heat = gfem.joule_heating(dc, current=current)
    density = np.asarray(heat._density_on(mesh))
    field = current * r_per_length  # V/m along the waveguide
    for k, sigma in SIGMAS.items():
        np.testing.assert_allclose(
            density[mesh.elements(k)], sigma * field**2, rtol=1e-12
        )
    assert not density[mesh.elements("clad")].any()
    per_length = float(np.sum(density * _volumes(mesh))) / thickness
    assert per_length == pytest.approx(current**2 * r_per_length, rel=1e-12)
    by_power = gfem.joule_heating(dc, power_per_length=per_length)._density_on(mesh)
    np.testing.assert_allclose(by_power, density, rtol=1e-12)
    assert "resistance per length" in repr(dc)
    assert "W/m" in repr(heat)


def test_a_slab_refuses_slab_totals_with_the_per_length_remedy(section_path):
    mesh, model = _slab(section_path, 1e-6)
    problem = gfem.DCConduction(terminals={"heater": "front"}, ground="back")
    dc = gfem.solve(model, problem, options=DIRECT)
    with pytest.raises(ValueError, match=r"power= .*power_per_length=P / L"):
        gfem.joule_heating(dc, power=1e-3)
    with pytest.raises(
        ValueError, match=r"voltage= .*current=V / \(resistance_per_length \* L\)"
    ):
        gfem.joule_heating(dc, voltage=1.0)
    with pytest.raises(ValueError, match=r"power= .*power_per_length=P / L"):
        gfem.HeatSource.uniform("h1", power=1e-3)._density_on(mesh)
    per_length = gfem.HeatSource.uniform("h1", power_per_length=2.0)._density_on(mesh)
    area = 1.0 * 0.5 * UM**2
    np.testing.assert_allclose(
        np.asarray(per_length)[mesh.elements("h1")], 2.0 / area, rtol=1e-12
    )


# --- gsim's femwell heater ---------------------------------------------------------

#: gsim's cross-section mesh of femwell's TiN heater (tests/verification/data).
FEMWELL_MESH = (
    Path(__file__).parents[1] / "verification" / "data" / "gsim_femwell_tin.msh"
)


def test_femwell_resistance_is_gsims_on_its_mesh():
    """A 2 um x 140 nm TiN heater, sigma 2.3e6 S/m, 320 um long: 496.894 Ohm."""
    mesh = gfem.Mesh.from_gmsh_cross_section(FEMWELL_MESH, thickness=0.5e-6)
    model = gfem.Model(
        mesh,
        materials={
            "heater": gfem.Material(conductivity=2.3e6),
            **dict.fromkeys(("core", "oxide", "si_wafer"), OXIDE),
        },
    )
    problem = gfem.DCConduction(terminals={"heater": "front"}, ground="back")
    dc = gfem.solve(model, problem, options=DIRECT)
    # gsim 5ecbb6a: 496.8944099378894 Ohm; 3e-14 measured
    resistance = float(dc.resistance_per_length[0, 0]) * 320e-6
    assert resistance == pytest.approx(496.8944099378894, rel=1e-9)
