"""Cell outputs of a periodic cell: `floquet_ports`, `efficiencies`, `power_balance`.

Identities on coarse F1 cells (an eps 4 slab on an eps 2.25 substrate under air,
Floquet ports on both faces) at p = 2, and on the bare interface closed by PEC
at p = 3, all with the direct solver. The checks are exact identities of the
solved S (no comparison with a closed form), so they hold on a coarse mesh; the
absorptance bounds of the lossless cell measure the mesh's power defect.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem import diffraction as dif

OPTS = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
TOP, BOT = "z_max", "z_min"
F0 = pc.F1_FREQUENCY
#: Identities of exact arithmetic on S: round-off only (measured about 2e-16).
EXACT = 1e-14
#: |A| of the lossless F1 at oblique incidence on this coarse mesh at p = 2: the
#: discretization's power defect, not round-off (1.2e-3 measured at 40°/30°,
#: 8.1e-4 at 30° and 1.6 f0; 6.0e-7 on `F1_FINE_CELLS` at p = 3). At normal
#: incidence the coarse defect is within `F1_POWER_TOL` (3.7e-5 measured). A
#: dropped row or a wrong weight moves A by F1's smallest reflectance (3.5e-2)
#: or more.
OBLIQUE_POWER_TOL = 5e-3
POLARIZATIONS = ("TE", "TM", "RCP", "LCP", "unpolarized", (0.6, 0.8j))


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("outputs"))


@pytest.fixture(scope="module")
def model(mesh):
    return pc.f1(mesh)


@pytest.fixture(scope="module")
def normal(model):
    """Normal incidence, two frequencies, every specular port excited."""
    return gfem.solve(model, gfem.Driven([F0, 1.1 * F0]), options=OPTS)


@pytest.fixture(scope="module")
def oblique(model):
    k = model.bloch_wave_vector(TOP, F0, theta=pc.F1_THETA, phi=pc.F1_PHI)
    problem = gfem.Driven([F0, 1.1 * F0], wave_vector=k, wave_vector_frequency=F0)
    return gfem.solve(model, problem, options=OPTS)


@pytest.fixture(scope="module")
def pec(tmp_path_factory):
    ports = [gfem.FloquetPort(f"R_{p}", TOP, polarization=p) for p in ("TE", "TM")]
    mesh = pc.interface_mesh(tmp_path_factory.mktemp("pec"))
    model = pc.interface_model(mesh, ports, boundaries=[gfem.PEC(BOT)])
    options = gfem.SolverOptions(order=3, threads=2, linear_solver="direct")
    return model, gfem.solve(model, gfem.Driven([F0]), options=options)


@pytest.fixture(scope="module")
def partial(model):
    """F1 without the port ``T_TM``: order (0, 0) propagates on z_min in TM unread."""
    partial = model.replace(ports=[p for p in model.ports if p.name != "T_TM"])
    with pytest.warns(gfem.UnrequestedOrderWarning):
        result = gfem.solve(partial, gfem.Driven([F0]), options=OPTS)
    return partial, result


@pytest.fixture(params=["normal", "oblique"])
def cell(request, model):
    return request.getfixturevalue(request.param), model


def _te_column(result, surface):
    """S's column of the TE port of the face ``surface``."""
    s = result.s
    return s.values[:, :, s.excited.index(("R" if surface == TOP else "T") + "_TE")]


# --- C1: efficiency rows ---------------------------------------------------------


@pytest.mark.parametrize("surface", [TOP, BOT])
def test_c1_circular_rows_are_the_circular_combinations_of_the_outgoing_waves(
    cell, surface
):
    result, model = cell
    eff = dif.efficiencies(
        result, surface, polarization="TE", model=model, basis="circular"
    )
    col = _te_column(result, surface)
    ports = result.s.ports
    for (s, order, pol), row in zip(eff.rows, eff.values.T, strict=True):
        h = 1 if s == TOP else -1
        pre = "R" if s == TOP else "T"
        tm, te = col[:, ports.index(f"{pre}_TM")], col[:, ports.index(f"{pre}_TE")]
        sign = {"RCP": 1, "LCP": -1}[pol]
        ref = jnp.abs((tm + sign * 1j * h * te) / math.sqrt(2)) ** 2
        np.testing.assert_allclose(row, ref, rtol=0, atol=EXACT)
        assert order == (0, 0)
    assert [r[2] for r in eff.rows] == ["RCP", "LCP"] * 2


@pytest.mark.parametrize("surface", [TOP, BOT])
def test_c1_rcp_incidence_is_the_documented_combination(normal, model, surface):
    sign = 1 if surface == TOP else -1
    pre = "R" if surface == TOP else "T"
    s = normal.s
    tm = s.values[:, :, s.excited.index(f"{pre}_TM")]
    te = s.values[:, :, s.excited.index(f"{pre}_TE")]
    for pol, c in (("RCP", sign), ("LCP", -sign)):
        eff = dif.efficiencies(normal, surface, polarization=pol, model=model)
        col = (tm + c * 1j * te) / math.sqrt(2)
        np.testing.assert_allclose(eff.values, jnp.abs(col) ** 2, rtol=0, atol=EXACT)
        assert eff.incidence == (surface, (0, 0), pol)


def test_c1_rows_and_lookup(normal, model):
    eff = dif.efficiencies(normal, TOP, polarization="TE", model=model)
    assert eff.rows == (
        (TOP, (0, 0), "TE"),
        (TOP, (0, 0), "TM"),
        (BOT, (0, 0), "TE"),
        (BOT, (0, 0), "TM"),
    )
    assert eff.values.shape == (2, 4)
    np.testing.assert_array_equal(eff[TOP, (0, 0), "TM"], eff.values[:, 1])
    np.testing.assert_array_equal(eff[BOT, (0, 0)], eff.values[:, 2] + eff.values[:, 3])
    np.testing.assert_array_equal(eff.frequency, [F0, 1.1 * F0])


# --- C2: power balance -----------------------------------------------------------


@pytest.mark.parametrize("surface", [TOP, BOT])
@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_c2_the_power_balance_closes_from_both_faces(cell, surface, polarization):
    result, model = cell
    bal = dif.power_balance(result, surface, polarization=polarization, model=model)
    total = bal.reflectance + bal.transmittance + bal.absorptance + bal.coupled
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=EXACT)
    np.testing.assert_array_equal(bal.coupled, 0.0)
    eff = dif.efficiencies(result, surface, polarization=polarization, model=model)
    on = [i for i, r in enumerate(eff.rows) if r[0] == surface]
    off = [i for i, r in enumerate(eff.rows) if r[0] != surface]
    np.testing.assert_allclose(
        bal.reflectance, eff.values[:, on].sum(axis=1), rtol=0, atol=EXACT
    )
    np.testing.assert_allclose(
        bal.transmittance, eff.values[:, off].sum(axis=1), rtol=0, atol=EXACT
    )
    assert float(jnp.abs(bal.absorptance).max()) <= OBLIQUE_POWER_TOL


@pytest.mark.parametrize("surface", [TOP, BOT])
def test_c2_a_lossless_cell_at_normal_incidence_absorbs_nothing(normal, model, surface):
    for pol in POLARIZATIONS:
        bal = dif.power_balance(normal, surface, polarization=pol, model=model)
        assert float(jnp.abs(bal.absorptance).max()) <= pc.F1_POWER_TOL


def test_c2_a_jones_pair_is_normalized(normal, model):
    for fn in (dif.efficiencies, dif.power_balance):
        a, b = (
            fn(normal, BOT, polarization=pair, model=model)
            for pair in ((0.6, 0.8j), (3, 4j))
        )
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=0, atol=EXACT)
        assert a.incidence[:2] == b.incidence[:2] == (BOT, (0, 0))
        np.testing.assert_allclose(b.incidence[2], (0.6, 0.8j), rtol=0, atol=EXACT)


@pytest.mark.parametrize("surface", [TOP, BOT])
def test_c2_unpolarized_is_the_mean_of_te_tm_and_of_rcp_lcp(cell, surface):
    result, model = cell

    def bal(pol):
        return dif.power_balance(result, surface, polarization=pol, model=model)

    un = bal("unpolarized")
    for a, b in (("TE", "TM"), ("RCP", "LCP")):
        for field in ("reflectance", "transmittance", "absorptance"):
            mean = (getattr(bal(a), field) + getattr(bal(b), field)) / 2
            np.testing.assert_allclose(getattr(un, field), mean, rtol=0, atol=EXACT)


def test_c2_a_pec_backed_cell_reflects_everything(pec):
    model, result = pec
    for pol in ("TE", "TM", "RCP", "unpolarized"):
        bal = dif.power_balance(result, TOP, polarization=pol, model=model)
        np.testing.assert_allclose(bal.reflectance, 1.0, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(bal.transmittance, 0.0)


# --- C7: completeness ------------------------------------------------------------


def test_c7_a_missing_polarization_refuses_the_balance_but_not_the_efficiencies(
    partial,
):
    partial, result = partial
    with pytest.raises(
        ValueError,
        match=re.escape(
            "power_balance: Diffraction order (0, 0) TM on 'z_min' at 299.792 THz "
            "propagates without a Floquet port: the surface absorbs it, so its "
            "power is missing from S and 1 - R - T would count it as absorbed. "
            "Add a gfem.FloquetPort for each (gfem.diffraction.floquet_ports "
            "declares every propagating order in TE and TM), or read the declared "
            "orders with gfem.diffraction.efficiencies."
        ),
    ):
        dif.power_balance(result, TOP, polarization="TE", model=partial)
    eff = dif.efficiencies(result, TOP, polarization="TE", model=partial)
    assert eff.rows == (
        (TOP, (0, 0), "TE"),
        (TOP, (0, 0), "TM"),
        (BOT, (0, 0), "TE"),
    )


@pytest.mark.filterwarnings("ignore::gfem.UnrequestedOrderWarning")
def test_c7_an_undeclared_order_that_opens_in_the_band_is_listed(model):
    ports = dif.floquet_ports(model, TOP, [F0]) + dif.floquet_ports(model, BOT, [F0])
    cell = model.replace(ports=ports)
    result = gfem.solve(cell, gfem.Driven([F0, 3 * F0]), options=OPTS)
    with pytest.raises(
        ValueError, match=r"Diffraction order \(-1, 0\) TE and TM on 'z_max' at 899"
    ):
        dif.power_balance(result, TOP, polarization="TE", model=cell)


# --- C8: jit ----------------------------------------------------------------------


def _slab(model, eps):
    """``model`` with the slab's eps_r ``eps`` on every element, traceable."""
    n = len(model.mesh.elements("slab"))
    return model.with_material("slab", gfem.MaterialField(eps_r=jnp.full(n, eps)))


def test_c8_power_balance_under_jit_equals_eager(model):
    problem = gfem.Driven([F0, 1.1 * F0])

    def balance(eps):
        result = gfem.solve(_slab(model, eps), problem, options=OPTS)
        return dif.power_balance(result, BOT, polarization="RCP", model=model)

    jitted, eager = jax.jit(balance)(pc.F1_EPS_SLAB), balance(pc.F1_EPS_SLAB)
    for x, y in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager), strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=EXACT)
    assert jitted.incidence == eager.incidence
    np.testing.assert_array_equal(jitted.frequency, eager.frequency)


def test_c8_a_result_entering_jit_serves_efficiencies_but_not_power_balance(
    normal, model
):
    def eff(result):
        return dif.efficiencies(result, TOP, polarization="RCP", model=model)

    out = jax.jit(eff)(normal)
    np.testing.assert_allclose(out.values, eff(normal).values, rtol=0, atol=EXACT)
    assert out.rows == eff(normal).rows

    def bal(result):
        return dif.power_balance(
            result, TOP, polarization="TE", model=model
        ).absorptance

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"power_balance: the result's diffraction orders are traced by JAX "
        r"\(the result entered jax\.jit or jax\.vmap as an argument\)",
    ):
        jax.jit(bal)(normal)


def test_c8_the_outputs_differentiate_through_s(model):
    problem = gfem.Driven([F0])

    def loss(eps):
        result = gfem.solve(_slab(model, eps), problem, options=OPTS)
        return dif.power_balance(
            result, TOP, polarization="RCP", model=model
        ).reflectance[0]

    h = 1e-4
    eps = pc.F1_EPS_SLAB
    fd = (loss(eps + h) - loss(eps - h)) / (2 * h)
    assert abs(float(jax.grad(loss)(eps)) - float(fd)) <= 1e-6 * abs(float(fd))


# --- C9: sequences and messages ----------------------------------------------------


def test_c9_a_sequence_is_the_stack_of_the_per_result_outputs(normal, oblique, model):
    both = [normal, oblique]
    eff = dif.efficiencies(both, TOP, polarization="RCP", model=model)
    bal = dif.power_balance(both, BOT, polarization=(0.6, 0.8j), model=model)
    for i, r in enumerate(both):
        one = dif.efficiencies(r, TOP, polarization="RCP", model=model)
        np.testing.assert_array_equal(eff.values[i], one.values)
        one = dif.power_balance(r, BOT, polarization=(0.6, 0.8j), model=model)
        for field in ("reflectance", "transmittance", "coupled", "absorptance"):
            np.testing.assert_array_equal(getattr(bal, field)[i], getattr(one, field))
    assert eff.values.shape == (2, 2, 4)
    assert bal.reflectance.shape == (2, 2)
    assert bal.incidence == (BOT, (0, 0), (0.6 + 0j, 0.8j))
    assert dif.efficiencies(
        [normal], TOP, polarization="TE", model=model
    ).values.shape == (1, 2, 4)


@pytest.fixture(scope="module")
def other_frequency(model):
    return gfem.solve(model, gfem.Driven([F0]), options=OPTS)


@pytest.fixture(scope="module")
def one_excited(model):
    return gfem.solve(model, gfem.Driven([F0], excite=["R_TE"]), options=OPTS)


@pytest.mark.parametrize("name", ["efficiencies", "power_balance"])
def test_c9_the_messages_of_the_inputs(
    name, normal, other_frequency, one_excited, model
):
    fn = getattr(dif, name)

    def call(result=normal, surface=TOP, **kw):
        kw = {"polarization": "TE", "model": model} | kw
        return fn(result, surface, **kw)

    def raises(error, message, **kw):
        with pytest.raises(error, match=re.escape(message)):
            call(**kw)

    raises(
        TypeError,
        f"{name} needs a gfem.DrivenResult or a sequence of them (an angle sweep), "
        "got int.",
        result=3,
    )
    raises(TypeError, f"{name} needs model= a gfem.Model, got str.", model="cell")
    wrong = model.replace(
        ports=[
            gfem.FloquetPort("R_TE", TOP, order=(0, 0), polarization="TM"),
            gfem.FloquetPort("R_TM", TOP, order=(0, 0), polarization="TE"),
            *[p for p in model.ports if p.name not in ("R_TE", "R_TM")],
        ]
    )
    raises(
        ValueError,
        f"{name}: model= is not the model the result was solved with: Floquet "
        "port 'R_TE' reads order (0, 0) in TE on 'z_max' in the result, but "
        "order (0, 0) in TM on 'z_max' in the model. Pass the model given to "
        "gfem.solve.",
        model=wrong,
    )
    raises(
        ValueError,
        f"{name}: model= is not the model the result was solved with: Floquet "
        "port 'T_TM' reads order (0, 0) in TM on 'z_min' in the result, but the "
        "model has no such Floquet port. Pass the model given to gfem.solve.",
        model=model.replace(ports=[p for p in model.ports if p.name != "T_TM"]),
    )
    raises(
        ValueError,
        f"{name}: 'nowhere' carries no Floquet port of the result; its Floquet "
        "surfaces are 'z_max', 'z_min'.",
        surface="nowhere",
    )
    raises(
        ValueError,
        f"{name}: no Floquet port reads order (1, 0) in TE on 'z_max'. Add "
        "gfem.FloquetPort(name, 'z_max', order=(1, 0), polarization='TE') to "
        "the model.",
        order=(1, 0),
    )
    raises(
        ValueError,
        f"{name}: no Floquet port reads order (1, 0) in TM on 'z_max' "
        "(polarization='RCP' combines the order's TM and TE columns). Add "
        "gfem.FloquetPort(name, 'z_max', order=(1, 0), polarization='TM') to "
        "the model.",
        order=(1, 0),
        polarization="RCP",
    )
    raises(
        ValueError,
        f"{name}: Floquet port 'R_TM' (order (0, 0), TM on 'z_max') was not "
        "excited, so S has no column for it. Solve with Driven(..., "
        "excite=None) or add 'R_TM' to Driven.excite.",
        result=one_excited,
        polarization="RCP",
    )
    for bad in ("circular", "other", None):
        raises(
            ValueError,
            f"{name}: polarization must be 'TE', 'TM', 'RCP', 'LCP', "
            f"'unpolarized' or a Jones pair (a_TM, a_TE), got {bad!r}.",
            polarization=bad,
        )
    for pair in ((0, 0), (1, float("nan")), (1, np.inf)):
        raises(
            ValueError,
            f"{name}: the Jones pair {pair!r} must be two finite numbers, not "
            "both zero.",
            polarization=pair,
        )
    raises(
        TypeError,
        f"{name}: order must be two integers (m, n), got (0.5, 0).",
        order=(0.5, 0),
    )
    raises(
        ValueError,
        f"{name}: the results of a sequence must share their ports, excited "
        "ports, frequencies and Floquet ports (one angle sweep); result 1 "
        "differs from result 0 in its excited ports.",
        result=[normal, one_excited],
    )
    raises(
        ValueError,
        f"{name}: the results of a sequence must share their ports, excited "
        "ports, frequencies and Floquet ports (one angle sweep); result 1 "
        "differs from result 0 in its frequencies.",
        result=[normal, other_frequency],
    )
    raises(
        ValueError,
        f"{name}: the results of a sequence must share their ports, excited "
        "ports, frequencies and Floquet ports (one angle sweep); result 1 "
        "differs from result 0 in its Floquet ports.",
        result=[normal, normal.with_offsets({"R_TM": 1e-7})],
    )
    raises(ValueError, f"{name}: the sequence of results is empty.", result=[])
    raises(
        TypeError,
        f"{name} needs a gfem.DrivenResult or a sequence of them (an angle sweep), "
        "got str.",
        result=[normal, "z_max"],
    )


def test_c9_a_basis_other_than_linear_or_circular_is_refused(normal, model):
    with pytest.raises(
        ValueError,
        match=re.escape("efficiencies: basis must be 'linear' or 'circular', got 'x'."),
    ):
        dif.efficiencies(normal, TOP, polarization="TE", model=model, basis="x")


def test_c9_a_circular_basis_needs_both_polarizations_of_every_order(partial):
    partial, result = partial
    with pytest.raises(
        ValueError,
        match=re.escape(
            "efficiencies: no Floquet port reads order (0, 0) in TM on 'z_min' "
            "(basis='circular' needs both polarizations of every order). Add "
            "gfem.FloquetPort(name, 'z_min', order=(0, 0), polarization='TM') to "
            "the model."
        ),
    ):
        dif.efficiencies(
            result, TOP, polarization="TE", model=partial, basis="circular"
        )


def test_c9_the_two_polarizations_of_a_coherent_combination_share_an_offset(
    normal, model
):
    result, cell = normal.with_offsets({"R_TM": 1e-7}), model
    message = (
        "Floquet ports 'R_TM' and 'R_TE' read order (0, 0) on 'z_max' at "
        "different offsets (1e-07 m and 0 m), so their relative phase mixes two "
        "reference planes. Give both the same FloquetPort(offset=...), or align "
        "them with result.with_offsets(...)."
    )
    for pol in ("RCP", (0.6, 0.8j)):
        with pytest.raises(ValueError, match=re.escape(message)):
            dif.efficiencies(result, TOP, polarization=pol, model=cell)
    with pytest.raises(ValueError, match=re.escape(message)):
        dif.efficiencies(result, TOP, polarization="TE", model=cell, basis="circular")
    # a single column, or the incoherent mean, accepts two offsets
    dif.efficiencies(result, TOP, polarization="TE", model=cell)
    dif.efficiencies(result, TOP, polarization="unpolarized", model=cell)


def test_c9_a_result_without_floquet_ports_is_refused(tmp_path):
    mesh = pc.c1_mesh(tmp_path)
    model = pc.c1(mesh)
    result = gfem.solve(
        model, gfem.Driven(frequency=pc.C1_FREQUENCY, wave_vector=pc.C1_K), options=OPTS
    )
    for name in ("efficiencies", "power_balance"):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{name}: the result has no Floquet port; {name} reads the "
                "gfem.FloquetPort rows of S of a periodic cell."
            ),
        ):
            getattr(dif, name)(result, "x", polarization="TE", model=model)


def test_c9_a_traced_jones_pair_raises(normal, model):
    def f(a):
        return dif.efficiencies(
            normal, TOP, polarization=(a, 1j), model=model
        ).values.sum()

    for transform in (jax.grad, jax.jit):
        with pytest.raises(
            gfem.NonDifferentiableError,
            match=r"efficiencies: polarization must be fixed numbers",
        ):
            transform(f)(1.0)


def test_c9_efficiency_keys_raise_named_errors(normal, model, partial):
    eff = dif.efficiencies(normal, TOP, polarization="TE", model=model)
    rows = ", ".join(map(repr, eff.rows))
    with pytest.raises(
        KeyError,
        match=re.escape(
            f"No efficiency row ('z_max', (1, 0), 'TE'); the rows are {rows}."
        ),
    ):
        eff[TOP, (1, 0), "TE"]
    with pytest.raises(KeyError, match="No efficiency row"):
        eff["z_max"]
    partial, result = partial
    eff = dif.efficiencies(result, TOP, polarization="TE", model=partial)
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Order (0, 0) on 'z_min' has no TM row, so its total over both "
            "polarizations is not known; read eff['z_min', (0, 0), 'TE'] or add "
            "the missing Floquet port."
        ),
    ):
        eff[BOT, (0, 0)]


# --- floquet_ports -----------------------------------------------------------------


def test_floquet_ports_names_order_and_polarizations(model):
    ports = dif.floquet_ports(model, TOP, [F0, 3 * F0])
    assert [p.name for p in ports[:6]] == [
        "z_max (0, 0) TE",
        "z_max (0, 0) TM",
        "z_max (-1, 0) TE",
        "z_max (-1, 0) TM",
        "z_max (0, -1) TE",
        "z_max (0, -1) TM",
    ]
    assert all(isinstance(p, gfem.FloquetPort) and p.surface == TOP for p in ports)
    orders = [p.order for p in ports[::2]]
    assert orders == list(model.propagating_orders(TOP, [F0, 3 * F0]))
    assert [p.polarization for p in ports[:2]] == ["TE", "TM"]
    assert {p.offset for p in ports} == {0.0}
    assert dif.floquet_ports(model, TOP, F0, offset=2e-7)[0].offset == 2e-7
    assert len(dif.floquet_ports(model, TOP, F0)) == 2


def test_floquet_ports_declare_the_union_over_wave_vectors(model):
    f = 1.4 * F0
    ks = [
        model.bloch_wave_vector(TOP, f, theta=t, phi=0.0)
        for t in (0.0, math.radians(35.0), math.radians(60.0))
    ]
    union = set()
    for k in ks:
        union |= set(model.propagating_orders(TOP, f, wave_vector=k))
    ports = dif.floquet_ports(model, TOP, f, wave_vector=np.array(ks))
    assert {p.order for p in ports} == union
    assert len({p.order for p in ports}) > len(model.propagating_orders(TOP, f))
    one = dif.floquet_ports(model, TOP, f, wave_vector=ks[1])
    assert {p.order for p in one} == set(
        model.propagating_orders(TOP, f, wave_vector=ks[1])
    )


def test_floquet_ports_scale_a_wave_vector_with_the_frequency(model):
    k = model.bloch_wave_vector(TOP, F0, theta=math.radians(50.0), phi=0.0)
    fixed = dif.floquet_ports(model, TOP, [F0, 1.6 * F0], wave_vector=k)
    scaled = dif.floquet_ports(
        model, TOP, [F0, 1.6 * F0], wave_vector=k, wave_vector_frequency=F0
    )
    expected = set(model.propagating_orders(TOP, [F0], wave_vector=k))
    expected |= set(
        model.propagating_orders(TOP, [1.6 * F0], wave_vector=tuple(1.6 * c for c in k))
    )
    assert {p.order for p in scaled} == expected
    assert {p.order for p in fixed} != expected


def test_floquet_ports_make_a_complete_declaration(model):
    k = model.bloch_wave_vector(TOP, F0, theta=math.radians(30.0), phi=0.4)
    f = [F0, 1.6 * F0]
    ports = [
        p
        for s in (TOP, BOT)
        for p in dif.floquet_ports(model, s, f, wave_vector=k, wave_vector_frequency=F0)
    ]
    assert len(ports) > 4  # orders beyond the specular one open at 1.6 f0
    cell = model.replace(ports=ports)
    problem = gfem.Driven(
        f,
        wave_vector=k,
        wave_vector_frequency=F0,
        excite=["z_max (0, 0) TE", "z_max (0, 0) TM"],
    )
    result = gfem.solve(cell, problem, options=OPTS)
    bal = dif.power_balance(result, TOP, polarization="unpolarized", model=cell)
    np.testing.assert_allclose(
        bal.reflectance + bal.transmittance + bal.absorptance, 1.0, rtol=0, atol=EXACT
    )
    assert float(jnp.abs(bal.absorptance).max()) <= OBLIQUE_POWER_TOL


def test_floquet_ports_messages(model):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "floquet_ports: wave_vector_frequency scales wave_vector, but "
            "wave_vector is None."
        ),
    ):
        dif.floquet_ports(model, TOP, F0, wave_vector_frequency=F0)
    for bad in (np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2, 3))):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "floquet_ports: wave_vector must be one 3-vector or a sequence of "
                f"them, shape (n, 3), in rad/m, got shape {bad.shape}."
            ),
        ):
            dif.floquet_ports(model, TOP, F0, wave_vector=bad)
    for f0 in (0.0, -F0, "f0", math.inf):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "floquet_ports: wave_vector_frequency must be a positive finite "
                "frequency in Hz, or None for the same wave vector at every "
                f"frequency; got {f0!r}."
            ),
        ):
            dif.floquet_ports(
                model, TOP, F0, wave_vector=(0, 0, 0), wave_vector_frequency=f0
            )
    with pytest.raises(gfem.errors.RegionNotFoundError, match="nowhere"):
        dif.floquet_ports(model, "nowhere", F0)
    with pytest.raises(gfem.NonDifferentiableError, match="fixed"):
        jax.grad(lambda f: float(len(dif.floquet_ports(model, TOP, f))))(F0)
