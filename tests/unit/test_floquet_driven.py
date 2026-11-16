"""Driven solves with Floquet ports: S on coarse cells against closed forms.

Cell F1 of `periodic_cases` (an eps 4 slab on an eps 2.25 substrate, air
above, Floquet ports on both open faces) at normal incidence, coarse (p = 2),
against the tangential-E Airy form in ADR 0147's lattice frame; a bare
glass/air interface; a cell closed by PEC behind one Floquet face; the
`DrivenResult.diffraction_orders` it reports; and the `bddc_gmres` solver
against `direct`.
"""

import dataclasses
import math
import warnings

import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem

P2 = gfem.SolverOptions(order=2, threads=2)


@pytest.fixture(scope="module")
def f1_mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("f1"))


@pytest.fixture(scope="module")
def f1_result(f1_mesh):
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"])
    return gfem.solve(pc.f1(f1_mesh), problem, options=P2)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_f1_at_normal_incidence_matches_the_airy_form(f1_result, pol):
    r, t = pc.f1_airy(pol)
    s = f1_result.s
    assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_COARSE_TOL
    assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_COARSE_TOL


def test_f1_reports_the_diffraction_order_of_each_floquet_port(f1_result):
    k0 = 2 * math.pi * pc.F1_FREQUENCY / pc.C0
    for name, surface, index in (("R_TM", "z_max", 1.0), ("T_TE", "z_min", 1.5)):
        order = f1_result.diffraction_orders[name]
        assert order.port == name
        assert order.surface == surface
        assert order.order == (0, 0)
        assert order.polarization == name[-2:]
        assert order.index == pytest.approx(index)
        np.testing.assert_array_equal(order.frequency, [pc.F1_FREQUENCY])
        np.testing.assert_array_equal(order.wave_vector, np.zeros((1, 3)))
        np.testing.assert_allclose(order.gamma, [1j * k0 * index], rtol=1e-12)
        np.testing.assert_array_equal(order.propagating, [True])
        np.testing.assert_array_equal(order.theta, [0.0])
        np.testing.assert_array_equal(order.phi, [0.0])
    assert set(f1_result.diffraction_orders) == {"R_TE", "R_TM", "T_TE", "T_TM"}


def test_the_diffraction_orders_round_trip_through_a_result_bundle(f1_mesh, tmp_path):
    job = gfem.Job(pc.f1(f1_mesh), gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE"]), P2)
    result = job.run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert set(loaded.diffraction_orders) == set(result.diffraction_orders)
    for name, order in result.diffraction_orders.items():
        back = loaded.diffraction_orders[name]
        assert isinstance(back, gfem.DiffractionOrderResult)
        for field in ("port", "order", "polarization", "surface", "index"):
            assert getattr(back, field) == getattr(order, field)
        for field in ("frequency", "wave_vector", "gamma", "propagating", "theta"):
            np.testing.assert_array_equal(getattr(back, field), getattr(order, field))


def test_bddc_gmres_agrees_with_direct(f1_mesh, f1_result):
    options = gfem.SolverOptions(order=2, threads=2, linear_solver="bddc_gmres")
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R_TE", "R_TM"])
    iterative = gfem.solve(pc.f1(f1_mesh), problem, options=options)
    error = np.abs(np.asarray(iterative.s.values) - np.asarray(f1_result.s.values))
    assert error.max() <= 10 * options.tolerance


# --- a bare interface and a closed cell -------------------------------------------

N_GLASS = 1.5
#: The glass and air layers' thickness, um: each Floquet face that far from
#: the interface.
D = 0.25


def _interface_mesh(directory):
    """Glass below z = D, air above, 0.4 x 0.4 um, 3 cells per layer."""
    lateral = np.linspace(0.0, pc.F1_A, 3)
    z = np.linspace(0.0, 2 * D, 7)
    cell = sm.periodic_cell(
        (lateral, lateral, z),
        lambda c: "glass" if c[2] < D else "air",
        paired="xy",
    )
    path = sm.write_msh22(directory / "interface.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _interface_model(mesh, ports, boundaries=()):
    return gfem.Model(
        mesh,
        materials={"glass": gfem.Material(eps_r=N_GLASS**2), "air": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            *boundaries,
        ],
        ports=ports,
    )


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_a_bare_glass_air_interface_reflects_n_minus_1_over_n_plus_1(tmp_path, pol):
    ports = [
        gfem.FloquetPort("glass", "z_min", polarization=pol, offset=D * 1e-6),
        gfem.FloquetPort("air", "z_max", polarization=pol, offset=D * 1e-6),
    ]
    model = _interface_model(_interface_mesh(tmp_path), ports)
    options = gfem.SolverOptions(order=3, threads=2)
    s = gfem.solve(
        model, gfem.Driven([pc.F1_FREQUENCY], excite=["glass"]), options=options
    ).s
    # tangential E at the interface: r = (n - 1)/(n + 1) seen from the glass,
    # t = 1 + r, times sqrt(Z_glass / Z_air) = sqrt(1 / n) in power waves
    r = (N_GLASS - 1) / (N_GLASS + 1)
    assert abs(complex(s["glass", "glass"][0]) - r) <= 1e-3
    assert abs(complex(s["air", "glass"][0]) - (1 + r) / math.sqrt(N_GLASS)) <= 1e-3


def test_a_cell_closed_by_pec_reflects_all_power(tmp_path):
    ports = [
        gfem.FloquetPort(f"R_{pol}", "z_max", polarization=pol) for pol in ("TE", "TM")
    ]
    model = _interface_model(
        _interface_mesh(tmp_path), ports, boundaries=[gfem.PEC("z_min")]
    )
    options = gfem.SolverOptions(order=3, threads=2)
    s = gfem.solve(model, gfem.Driven([pc.F1_FREQUENCY]), options=options).s
    values = np.asarray(s.values)[0]
    np.testing.assert_allclose(np.abs(np.diag(values)), 1.0, atol=1e-6)


def test_exciting_an_evanescent_order_raises_before_any_solve(f1_mesh):
    ports = [
        *pc.f1_ports(),
        gfem.FloquetPort("R1", "z_max", order=(1, 0), polarization="TE"),
    ]
    with pytest.raises(
        ValueError,
        match=r"FloquetPort 'R1' \(order \(1, 0\), TE\) cannot be excited at "
        r".*excite=\['R_TE', 'R_TM', 'T_TE', 'T_TM'\]",
    ):
        gfem.solve(pc.f1(f1_mesh, ports), gfem.Driven([pc.F1_FREQUENCY]), options=P2)


# --- Bloch (oblique) incidence -------------------------------------------------------


@pytest.fixture(scope="module")
def f2_mesh(tmp_path_factory):
    return pc.f2_mesh(tmp_path_factory.mktemp("f2"))


def test_a_zero_wave_vector_is_plain_periodicity_over_the_full_s(f2_mesh):
    # the general (LU) Bloch path against the symmetric plain one: every
    # propagating order of the grating, both faces, both polarizations
    model = pc.f2(f2_mesh)

    def s(wave_vector):
        problem = gfem.Driven([pc.F1_FREQUENCY], wave_vector=wave_vector)
        return np.asarray(gfem.solve(model, problem, options=P2).s.values)[0]

    plain, zero = s(None), s((0.0, 0.0, 0.0))
    assert plain.shape == (12, 12)
    assert np.abs(plain).max() > 0.1  # a non-trivial S
    np.testing.assert_allclose(zero, plain, rtol=0, atol=1e-12)


def _oblique(model, theta, phi=0.0):
    return model.bloch_wave_vector(
        "z_max", pc.F1_FREQUENCY, theta=math.radians(theta), phi=math.radians(phi)
    )


def test_bddc_gmres_agrees_with_direct_at_oblique_incidence(f1_mesh):
    model = pc.f1(f1_mesh)
    problem = gfem.Driven(
        [pc.F1_FREQUENCY], excite=["R_TE", "R_TM"], wave_vector=_oblique(model, 20.0)
    )
    direct = gfem.solve(model, problem, options=P2)
    options = gfem.SolverOptions(order=2, threads=2, linear_solver="bddc_gmres")
    iterative = gfem.solve(model, problem, options=options)
    s = np.asarray(direct.s.values)
    assert abs(s[0, 0, 0]) > 0.1  # oblique S of the slab, not a blank
    error = np.abs(np.asarray(iterative.s.values) - s)
    assert error.max() <= 10 * options.tolerance


def test_the_transposed_system_is_the_system_at_minus_k(f1_mesh):
    # the adjoint solve under k (the transposed system, its Floquet couplings
    # conj(U) C^T U^T read off the forward vectors) against a forward solve of
    # the model at -k, whose couplings are built from the mirrored orders
    from gfem._engine.driven import DrivenRule

    model = pc.f1(f1_mesh)
    k = _oblique(model, 40.0, 30.0)
    omega = 2 * math.pi * pc.F1_FREQUENCY

    def system(wave_vector):
        problem = gfem.Driven([pc.F1_FREQUENCY], wave_vector=wave_vector)
        rule = DrivenRule(model, problem, P2)
        return rule._ports(model)[0]

    plus, minus = system(k), system(tuple(-c for c in k))
    try:
        free = np.array(list(minus.operator.space.FreeDofs()))
        rng = np.random.default_rng(0)
        loads = rng.standard_normal((free.size, 2)) + 1j * rng.standard_normal(
            (free.size, 2)
        )
        loads[~free] = 0
        adjoint = plus.solve_adjoint(0, omega, loads)
        with minus.sweep() as sweep:
            forward, _ = sweep.solve(omega, loads)
        scale = np.abs(forward[free]).max()
        # (the forward Floquet term in place of its transpose is off by 11x scale)
        np.testing.assert_allclose(
            adjoint[free], forward[free], rtol=0, atol=1e-12 * scale
        )
    finally:
        plus.free()
        minus.free()


# --- higher orders: the full S and its readers ----------------------------------------


def test_a_propagating_order_without_a_port_warns_per_surface(f2_mesh):
    # only the specular ports: the +-1 orders propagate on both faces
    model = pc.f2(f2_mesh, pc.f2_ports(((0, 0),)))
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R0,0_TE"])
    with pytest.warns(gfem.UnrequestedOrderWarning) as caught:
        gfem.solve(model, problem, options=P2)
    messages = sorted(
        str(w.message) for w in caught if w.category is gfem.UnrequestedOrderWarning
    )
    assert len(messages) == 2
    for message, surface in zip(messages, ("z_max", "z_min"), strict=True):
        assert message == (
            f"Propagating order(s) (-1, 0), (1, 0) on {surface!r} at 299.792 THz "
            "carry power that S does not report (they are absorbed, not "
            "reflected); add gfem.FloquetPort(..., order=..., polarization=...) "
            "to read them, or filter UnrequestedOrderWarning."
        )


def test_declaring_every_propagating_order_does_not_warn(f2_mesh, recwarn):
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R0,0_TE"])
    gfem.solve(pc.f2(f2_mesh), problem, options=P2)
    assert not [w for w in recwarn if w.category is gfem.UnrequestedOrderWarning]


#: A reference-plane offset far beyond the decay length of F2's evanescent
#: orders: exp(|gamma| d) overflows (|gamma| d about 8e3 for (2, 0) on top).
FAR = 1e-3


def _with_order(ports, name, surface, order, offset=0.0):
    return [
        *ports,
        gfem.FloquetPort(name, surface, order=order, polarization="TE", offset=offset),
    ]


def test_an_evanescent_row_reads_exactly_zero_even_far_off_its_plane(f2_mesh):
    ports = _with_order(pc.f2_ports(), "R2", "z_max", (2, 0), offset=FAR)
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R0,0_TE"])
    result = gfem.solve(pc.f2(f2_mesh, ports), problem, options=P2)
    s = result.s
    assert not bool(np.asarray(result.diffraction_orders["R2"].propagating)[0])
    assert complex(s["R2", "R0,0_TE"][0]) == 0
    assert np.isfinite(np.asarray(s.values)).all()
    assert np.isnan(np.asarray(s.z_ref)[0, s.ports.index("R2")])
    shifted = result.with_offsets({"R2": 2 * FAR})
    assert complex(shifted.s["R2", "R0,0_TE"][0]) == 0
    assert np.isfinite(np.asarray(shifted.s.values)).all()


def test_floquet_offsets_are_listed_and_moved_by_with_offsets(f2_mesh):
    d = 0.15e-6
    problem = gfem.Driven([pc.F1_FREQUENCY], excite=["R0,0_TE", "R0,0_TM"])
    base = pc.f2_ports()
    plain = gfem.solve(pc.f2(f2_mesh, base), problem, options=P2)
    moved = [
        dataclasses.replace(p, offset=d) if p.name in ("R1,0_TE", "T-1,0_TM") else p
        for p in base
    ]
    solved = gfem.solve(pc.f2(f2_mesh, moved), problem, options=P2)
    assert dict(plain.offsets) == {p.name: 0.0 for p in base}
    assert dict(solved.offsets)["R1,0_TE"] == d
    shifted = plain.with_offsets({"R1,0_TE": d, "T-1,0_TM": d})
    assert dict(shifted.offsets) == dict(solved.offsets)
    expected = np.asarray(solved.s.values)
    row = plain.s.ports.index("R1,0_TE")
    # the offset moved a phase (beta d about 0.6 rad for the top +1 order)
    assert abs(expected[0, row, 0] - np.asarray(plain.s.values)[0, row, 0]) > 1e-2
    np.testing.assert_allclose(
        np.asarray(shifted.s.values), expected, rtol=0, atol=1e-12
    )


def test_frequency_workers_model_the_orders_of_the_whole_grid(f2_mesh):
    # with floquet_order_tolerance 1e-2 the top face models (+-2, 0) for
    # 1.6 f0 (evanescent, close to cutoff) but not for f0 alone
    # (exp(-2 |gamma| d) = 6.5e-3 there): a worker's block on its own would
    # model fewer orders than the sweep
    frequency = [pc.F1_FREQUENCY, 1.6 * pc.F1_FREQUENCY]
    ports = pc.f2_ports(((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)))
    model = pc.f2(f2_mesh, ports)
    problem = gfem.Driven(frequency, excite=["R0,0_TE", "R0,0_TM"])

    def s(workers):
        options = gfem.SolverOptions(
            order=2, threads=2, frequency_workers=workers, floquet_order_tolerance=1e-2
        )
        return np.asarray(gfem.solve(model, problem, options=options).s.values)

    serial, parallel = s(1), s(2)
    assert np.abs(serial).max() > 0.1
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=1e-12)


#: The Bloch reciprocity gate on `direct` (`test_periodic_driven.SYMMETRY_TOL`).
SYMMETRY_TOL = 1e-12


def _mirror(name):
    """The port of the opposite order: same face and polarization, (-m, -n)."""
    side, rest = name[0], name[1:]
    orders, pol = rest.split("_")
    m, n = (int(i) for i in orders.split(","))
    return f"{side}{-m},{-n}_{pol}"


def _full_s(mesh, top, bottom, wave_vector):
    """F2 solved with every port excited: ports "R{m},0_{pol}" and "T..."."""
    ports = [
        gfem.FloquetPort(f"{side}{m},0_{pol}", surface, order=(m, 0), polarization=pol)
        for side, surface, orders in (("R", "z_max", top), ("T", "z_min", bottom))
        for m in orders
        for pol in ("TE", "TM")
    ]
    problem = gfem.Driven([pc.F1_FREQUENCY], wave_vector=wave_vector)
    result = gfem.solve(pc.f2(mesh, ports), problem, options=P2)
    return result


#: (theta, phi in degrees from the air, top orders m, bottom orders m): every
#: propagating order of F2 at that incidence.
RECIPROCITY_CASES = {
    # normal incidence: +-1 propagate on both faces
    "normal": (0.0, 0.0, (-1, 0, 1), (-1, 0, 1)),
    # conical 20/30 degrees: top (-1, 0), bottom (-2..1)
    "conical": (20.0, 30.0, (-1, 0), (-2, -1, 0, 1)),
}


@pytest.mark.parametrize("case", list(RECIPROCITY_CASES))
def test_the_full_s_obeys_bloch_reciprocity_with_the_order_signs(
    f2_mesh, recwarn, case
):
    # S_{p<-q}(k) = sigma_p sigma_q S_{-q<--p}(-k), sigma = -1 off kappa = 0
    theta, phi, top, bottom = RECIPROCITY_CASES[case]
    model = pc.f2(f2_mesh)
    k = _oblique(model, theta, phi) if theta else (0.0, 0.0, 0.0)
    plus = _full_s(f2_mesh, top, bottom, k)
    minus = _full_s(
        f2_mesh,
        tuple(-m for m in top),
        tuple(-m for m in bottom),
        tuple(-c for c in k),
    )
    assert not [w for w in recwarn if w.category is gfem.UnrequestedOrderWarning]
    names = plus.s.ports
    s_plus = np.asarray(plus.s.values)[0]
    s_minus = np.asarray(minus.s.values)[0]
    at = {name: i for i, name in enumerate(minus.s.ports)}

    def zero(name):
        return not np.any(np.asarray(plus.diffraction_orders[name].wave_vector))

    sigma = np.array([1.0 if zero(name) else -1.0 for name in names])
    mirrored = np.array(
        [[s_minus[at[_mirror(q)], at[_mirror(p)]] for q in names] for p in names]
    )
    assert np.abs(s_plus).max() > 0.1
    np.testing.assert_allclose(
        s_plus, sigma[:, None] * sigma[None, :] * mirrored, rtol=0, atol=SYMMETRY_TOL
    )
    if theta == 0:
        # the sign is pinned: without sigma the specular-to-(+-1) entries flip
        assert np.abs(s_plus - mirrored).max() > 0.1


def _grazing_frequency(model):
    """The float frequency at which F2's top (1, 0) order grazes: gamma == 0."""
    from gfem import _floquet

    f = pc.C0 / (pc.F2_A[0] * pc.UM)  # k0 = |b1| in the air, up to rounding
    for step in range(64):
        for g in (f, -f):
            trial = abs(g)
            for _ in range(step):
                trial = np.nextafter(trial, math.copysign(np.inf, g))
            table = _floquet.order_table(model, [trial], None, P2)["z_max"]
            if table.rows.gamma[0, table.orders.index((1, 0))] == 0:
                return float(trial)
    raise AssertionError("no float frequency grazes")  # pragma: no cover


def test_s_is_continuous_through_a_rayleigh_anomaly(f2_mesh):
    # the top +-1 orders at |gamma|/k = 1e-6 (propagating) and at gamma = 0
    # exactly (rows 0, the TM shift an exact constraint row): S stays finite,
    # the grazing rows fall as 0.5 sqrt(|gamma|/k), the rest barely moves
    model = pc.f2(f2_mesh)
    grazing = _grazing_frequency(model)
    near = grazing * (1 + 0.5e-12)  # |gamma|/k = 1e-6
    problem = gfem.Driven([grazing, near], excite=["R0,0_TE", "R0,0_TM"])
    result = gfem.solve(model, problem, options=P2)
    propagating = np.asarray(result.diffraction_orders["R1,0_TM"].propagating)
    np.testing.assert_array_equal(propagating, [False, True])
    s = np.asarray(result.s.values)
    assert np.isfinite(s).all()
    rows = [result.s.ports.index(f"R{m},0_{p}") for m in (-1, 1) for p in ("TE", "TM")]
    others = [i for i in range(len(result.s.ports)) if i not in rows]
    np.testing.assert_array_equal(s[0, rows], 0)
    assert np.abs(s[1, rows]).max() <= 1e-3
    np.testing.assert_allclose(s[0, others], s[1, others], rtol=0, atol=1e-5)


# --- fixed-angle sweeps ---------------------------------------------------------------


def test_a_fixed_angle_needs_a_wave_vector():
    with pytest.raises(
        ValueError,
        match=r"Driven\.wave_vector_frequency scales Driven\.wave_vector with "
        r"frequency, but wave_vector is None",
    ):
        gfem.Driven([1e14], wave_vector_frequency=1e14)


@pytest.mark.parametrize("value", [0.0, -1e14, math.inf, math.nan, "1e14", True])
def test_the_reference_frequency_is_a_positive_finite_number(value):
    with pytest.raises(
        ValueError,
        match=r"Driven\.wave_vector_frequency must be a positive finite frequency "
        r"in Hz",
    ):
        gfem.Driven([1e14], wave_vector=(1e6, 0, 0), wave_vector_frequency=value)


def test_a_traced_reference_frequency_raises():
    import jax

    def build(f):
        gfem.Driven([1e14], wave_vector=(1e6, 0, 0), wave_vector_frequency=f)
        return f

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"Driven\.wave_vector_frequency must be a fixed number",
    ):
        jax.grad(build)(1e14)


def test_the_reference_frequency_defaults_to_none_and_is_a_float():
    assert gfem.Driven([1e14]).wave_vector_frequency is None
    problem = gfem.Driven([1e14], wave_vector=(1e6, 0, 0), wave_vector_frequency=2)
    assert problem.wave_vector_frequency == 2.0
    assert isinstance(problem.wave_vector_frequency, float)


def test_a_fixed_angle_sweep_round_trips_through_a_job_file(f1_mesh, tmp_path):
    import json

    model = pc.f1(f1_mesh)
    k = _oblique(model, 40.0, 30.0)
    problem = gfem.Driven(
        [pc.F1_FREQUENCY], wave_vector=k, wave_vector_frequency=pc.F1_FREQUENCY
    )
    path = gfem.Job(model, problem, P2).to_json(tmp_path / "job.json")
    assert gfem.Job.from_json(path).problem == problem
    # a job written before the field reads None
    doc = json.loads(path.read_text())
    del doc["job"]["problem"]["wave_vector_frequency"]
    path.write_text(json.dumps(doc))
    back = gfem.Job.from_json(path).problem
    assert back.wave_vector_frequency is None
    assert back.wave_vector == problem.wave_vector


FIXED_ANGLE = [pc.F1_FREQUENCY, 1.1 * pc.F1_FREQUENCY]


def _at_explicit_k(model, k_ref, options):
    """S of single-frequency solves at k(f) = k_ref f / f0, stacked over FIXED_ANGLE."""
    rows = []
    for f in FIXED_ANGLE:
        k = tuple(c * f / pc.F1_FREQUENCY for c in k_ref)
        problem = gfem.Driven([f], excite=["R_TE", "R_TM"], wave_vector=k)
        rows.append(np.asarray(gfem.solve(model, problem, options=options).s.values))
    return np.concatenate(rows)


@pytest.fixture(scope="module")
def fixed_angle(f1_mesh):
    model = pc.f1(f1_mesh)
    k_ref = _oblique(model, 40.0, 30.0)
    problem = gfem.Driven(
        FIXED_ANGLE,
        excite=["R_TE", "R_TM"],
        wave_vector=k_ref,
        wave_vector_frequency=pc.F1_FREQUENCY,
    )
    return model, k_ref, problem, gfem.solve(model, problem, options=P2)


def test_a_fixed_angle_sweep_solves_each_frequency_at_its_own_wave_vector(fixed_angle):
    model, k_ref, _, result = fixed_angle
    swept = np.asarray(result.s.values)
    explicit = _at_explicit_k(model, k_ref, P2)
    assert np.abs(swept[0] - swept[1]).max() > 1e-2  # two different solves
    np.testing.assert_allclose(swept, explicit, rtol=0, atol=1e-12)
    # the specular order's in-plane wave vector is k(f)
    expected = [np.asarray(k_ref) * f / pc.F1_FREQUENCY for f in FIXED_ANGLE]
    np.testing.assert_allclose(
        result.diffraction_orders["R_TE"].wave_vector, expected, rtol=1e-14
    )


def test_a_fixed_angle_sweep_on_bddc_gmres_agrees_with_explicit_k(fixed_angle):
    model, k_ref, problem, _ = fixed_angle
    options = gfem.SolverOptions(order=2, threads=2, linear_solver="bddc_gmres")
    swept = np.asarray(gfem.solve(model, problem, options=options).s.values)
    error = np.abs(swept - _at_explicit_k(model, k_ref, P2))
    assert error.max() <= 10 * options.tolerance


def test_a_fixed_angle_sweep_on_frequency_workers_equals_the_serial_one(fixed_angle):
    model, _, problem, result = fixed_angle
    options = gfem.SolverOptions(order=2, threads=2, frequency_workers=2)
    parallel = np.asarray(gfem.solve(model, problem, options=options).s.values)
    np.testing.assert_allclose(
        parallel, np.asarray(result.s.values), rtol=0, atol=1e-12
    )


def test_a_fixed_angle_sweep_raises_on_the_adaptive_sweep(fixed_angle):
    model, k_ref, _, _ = fixed_angle
    problem = gfem.Driven(
        FIXED_ANGLE,
        sweep="adaptive",
        wave_vector=k_ref,
        wave_vector_frequency=pc.F1_FREQUENCY,
    )
    with pytest.raises(NotImplementedError, match="adaptive sweep cannot solve it"):
        gfem.solve(model, problem, options=P2)


def test_a_long_fixed_angle_sweep_keeps_a_fixed_wave_vector_cached(f1_mesh):
    from gfem._engine.driven import driven_operator

    model = pc.f1(f1_mesh)
    k_ref = _oblique(model, 40.0, 30.0)
    fixed = driven_operator(model, 2, wave_vector=k_ref).space
    # f0 first: its k(f) is the fixed wave vector's, nine others follow
    frequencies = [pc.F1_FREQUENCY * (1.0 + 0.01 * n) for n in range(10)]
    problem = gfem.Driven(
        frequencies,
        excite=["R_TE"],
        wave_vector=k_ref,
        wave_vector_frequency=pc.F1_FREQUENCY,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnrequestedOrderWarning)
        gfem.solve(model, problem, options=P2)
    # ten wave vectors, more than the eight phase sets of fixed ones
    assert driven_operator(model, 2, wave_vector=k_ref).space is fixed


# --- circular polarization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("surface", "travel", "handedness"),
    [("z_max", -1.0, "right"), ("z_min", +1.0, "left")],
)
def test_tm_plus_i_te_is_right_handed_from_the_top_and_left_from_the_bottom(
    tmp_path, surface, travel, handedness
):
    # The column combination (S[:, TM] + i S[:, TE]) / sqrt 2 is the response to
    # the incident field E_TM + i E_TE. In an empty cell, read that field at a
    # point and watch Re(E exp(+i w t)) turn: from w t = 0 to a quarter period
    # it turns about E(0) x E(T/4), which is along the travel direction for a
    # right-handed wave (IEEE: thumb along the propagation).
    ports = [
        gfem.FloquetPort(f"{side}_{pol}", face, polarization=pol)
        for side, face in (("top", "z_max"), ("bottom", "z_min"))
        for pol in ("TE", "TM")
    ]
    mesh = _interface_mesh(tmp_path)
    empty = gfem.Model(
        mesh,
        materials={"glass": gfem.Material(), "air": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=ports,
    )
    side = "top" if surface == "z_max" else "bottom"
    f = pc.F1_FREQUENCY
    problem = gfem.Driven([f], excite=[f"{side}_TE", f"{side}_TM"], keep_fields=f)
    result = gfem.solve(empty, problem, options=gfem.SolverOptions(order=3, threads=2))
    point = np.array([[0.5 * pc.F1_A, 0.5 * pc.F1_A, D]]) * 1e-6

    def incident(pol):
        return np.asarray(result.e_field(f, excite=f"{side}_{pol}").at(point))[0]

    e = incident("TM") + 1j * incident("TE")
    now, quarter = np.real(e), np.real(1j * e)  # Re(E exp(i w t)) at w t = 0, pi/2
    turn = np.cross(now, quarter)
    # circular: two equal, orthogonal, transverse components (to the mesh:
    # |E_z| / |E| 1.1e-3 measured)
    assert abs(e[2]) <= 1e-2 * np.linalg.norm(e)
    np.testing.assert_allclose(abs(turn[2]), np.vdot(e, e).real / 2, rtol=1e-2)
    sense = np.sign(turn[2] * travel)
    assert sense == (1.0 if handedness == "right" else -1.0)
