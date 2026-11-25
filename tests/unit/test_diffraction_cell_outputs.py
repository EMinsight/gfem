"""Cell outputs of a periodic cell: ports, efficiencies, power balance, Jones.

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
import structured_meshes as sm
from polarization_reference import circular_unitary, mueller_measured

import gfem
from gfem import diffraction as dif

OPTS = gfem.SolverOptions(order=2, threads=2, linear_solver="direct")
TOP, BOT = "z_max", "z_min"
F0 = pc.F1_FREQUENCY
#: Identities of exact arithmetic on S: round-off only (measured about 2e-16).
EXACT = 1e-14
EPS = float(np.finfo(float).eps)
# jax.jit and eager evaluation of the same expression may round differently: XLA
# fuses, reorders and contracts into fused multiply-adds, so each operation can
# land one unit in the last place away. For a pure-JAX output built from S, an
# entry that is a sum of n products of k factors of modulus at most one is then
# off by at most about n * k * eps; the bounds below take four times that, for the
# moduli and square roots around the sums.
JIT_MARGIN = 4
# |a S_TM + b S_TE|^2: two products of two factors, and the square doubles it.
EFFICIENCY_JIT = JIT_MARGIN * 2 * (2 * 2) * EPS
# A (J (x) J*) A^-1: sixteen products of four factors.
MUELLER_JIT = JIT_MARGIN * (16 * 4) * EPS
# A solve inside jax.jit assembles the operator from traced material values, so
# its entries can differ from the eager ones in the last bit, and the direct
# solve amplifies that by the operator's condition number. Every balance term is a
# fraction of the unit incident power, so the bound is absolute on that scale (the
# absorptance of this lossless cell is itself near zero). The jit-to-eager
# difference measured in CI was 1.4e-14 (about 76 units in the last place of a
# 0.83 reflectance); 1e-12 leaves a factor of about 70, while any real difference
# (a dropped row, a wrong weight or sign) moves the balance by at least 3.5e-2,
# F1's smallest reflectance.
SOLVE_JIT_TOL = 1e-12
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


# --- efficiency rows ---------------------------------------------------------


@pytest.mark.parametrize("surface", [TOP, BOT])
def test_circular_rows_are_the_circular_combinations_of_the_outgoing_waves(
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
def test_rcp_incidence_is_the_documented_combination(normal, model, surface):
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


def test_rows_and_lookup(normal, model):
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


# --- power balance -----------------------------------------------------------


@pytest.mark.parametrize("surface", [TOP, BOT])
@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_the_power_balance_closes_from_both_faces(cell, surface, polarization):
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
def test_a_lossless_cell_at_normal_incidence_absorbs_nothing(normal, model, surface):
    for pol in POLARIZATIONS:
        bal = dif.power_balance(normal, surface, polarization=pol, model=model)
        assert float(jnp.abs(bal.absorptance).max()) <= pc.F1_POWER_TOL


def test_a_jones_pair_is_normalized(normal, model):
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
def test_unpolarized_is_the_mean_of_te_tm_and_of_rcp_lcp(cell, surface):
    result, model = cell

    def bal(pol):
        return dif.power_balance(result, surface, polarization=pol, model=model)

    un = bal("unpolarized")
    for a, b in (("TE", "TM"), ("RCP", "LCP")):
        for field in ("reflectance", "transmittance", "absorptance"):
            mean = (getattr(bal(a), field) + getattr(bal(b), field)) / 2
            np.testing.assert_allclose(getattr(un, field), mean, rtol=0, atol=EXACT)


def test_a_pec_backed_cell_reflects_everything(pec):
    model, result = pec
    for pol in ("TE", "TM", "RCP", "unpolarized"):
        bal = dif.power_balance(result, TOP, polarization=pol, model=model)
        np.testing.assert_allclose(bal.reflectance, 1.0, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(bal.transmittance, 0.0)


# --- completeness ------------------------------------------------------------


def test_a_missing_polarization_refuses_the_balance_but_not_the_efficiencies(
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
def test_an_undeclared_order_that_opens_in_the_band_is_listed(model):
    ports = dif.floquet_ports(model, TOP, [F0]) + dif.floquet_ports(model, BOT, [F0])
    cell = model.replace(ports=ports)
    result = gfem.solve(cell, gfem.Driven([F0, 3 * F0]), options=OPTS)
    with pytest.raises(
        ValueError, match=r"Diffraction order \(-1, 0\) TE and TM on 'z_max' at 899"
    ):
        dif.power_balance(result, TOP, polarization="TE", model=cell)


# --- jit ----------------------------------------------------------------------


def _slab(model, eps):
    """``model`` with the slab's eps_r ``eps`` on every element, traceable."""
    n = len(model.mesh.elements("slab"))
    return model.with_material("slab", gfem.MaterialField(eps_r=jnp.full(n, eps)))


def test_power_balance_under_jit_equals_eager(model):
    problem = gfem.Driven([F0, 1.1 * F0])

    def balance(eps):
        result = gfem.solve(_slab(model, eps), problem, options=OPTS)
        return dif.power_balance(result, BOT, polarization="RCP", model=model)

    jitted, eager = jax.jit(balance)(pc.F1_EPS_SLAB), balance(pc.F1_EPS_SLAB)
    for x, y in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager), strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=SOLVE_JIT_TOL)
    assert jitted.incidence == eager.incidence
    np.testing.assert_array_equal(jitted.frequency, eager.frequency)


def test_a_result_entering_jit_serves_efficiencies_but_not_power_balance(normal, model):
    def eff(result):
        return dif.efficiencies(result, TOP, polarization="RCP", model=model)

    out = jax.jit(eff)(normal)
    np.testing.assert_allclose(
        out.values, eff(normal).values, rtol=0, atol=EFFICIENCY_JIT
    )
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


def test_the_outputs_differentiate_through_s(model):
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


# --- sequences and messages ----------------------------------------------------


def test_a_sequence_is_the_stack_of_the_per_result_outputs(normal, oblique, model):
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
def test_the_messages_of_the_inputs(name, normal, other_frequency, one_excited, model):
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


def test_a_basis_other_than_linear_or_circular_is_refused(normal, model):
    with pytest.raises(
        ValueError,
        match=re.escape("efficiencies: basis must be 'linear' or 'circular', got 'x'."),
    ):
        dif.efficiencies(normal, TOP, polarization="TE", model=model, basis="x")


def test_a_circular_basis_needs_both_polarizations_of_every_order(partial):
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


def test_the_two_polarizations_of_a_coherent_combination_share_an_offset(normal, model):
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


def test_a_result_without_floquet_ports_is_refused(tmp_path):
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


def test_a_traced_jones_pair_raises(normal, model):
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


def test_efficiency_keys_raise_named_errors(normal, model, partial):
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


# --- Jones, circular and Mueller ---------------------------------------------------

#: The surfaces and the (h_out, h_in) of the four blocks of the two-face cell.
BLOCKS = (
    (TOP, TOP, 1, -1),  # reflection from the top
    (BOT, TOP, -1, -1),  # transmission downward
    (TOP, BOT, 1, 1),  # transmission upward
    (BOT, BOT, -1, 1),  # reflection from the bottom
)
SURFACES = tuple(b[:2] for b in BLOCKS)
PAULI_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize(("out", "inc", "h_out", "h_in"), BLOCKS)
def test_jones_circular_columns_are_the_documented_combinations(
    cell, out, inc, h_out, h_in
):
    result, model = cell
    j = dif.jones(result, out, inc, model=model)
    assert j.direction == (h_out, h_in)
    assert j.outgoing == (out, (0, 0))
    assert j.incident == (inc, (0, 0))
    s = result.s
    pre_out, pre_in = ("R" if out == TOP else "T"), ("R" if inc == TOP else "T")
    col = {p: s.values[:, :, s.excited.index(f"{pre_in}_{p}")] for p in ("TM", "TE")}
    row = {p: s.ports.index(f"{pre_out}_{p}") for p in ("TM", "TE")}
    circ = j.circular
    for c, sign in ((0, 1), (1, -1)):
        # RCP/LCP incident: (S_TM + sign * i * (-h_in) ... ) from the top face
        # (h_in = -1) the documented (S_TM + i S_TE)/sqrt2, from the bottom
        # the conjugate combination
        mix = (col["TM"] - sign * 1j * h_in * col["TE"]) / math.sqrt(2)
        b_tm, b_te = mix[:, row["TM"]], mix[:, row["TE"]]
        right = (b_tm + 1j * h_out * b_te) / math.sqrt(2)
        left = (b_tm - 1j * h_out * b_te) / math.sqrt(2)
        np.testing.assert_allclose(circ[:, 0, c], right, rtol=0, atol=EXACT)
        np.testing.assert_allclose(circ[:, 1, c], left, rtol=0, atol=EXACT)


@pytest.mark.parametrize(("out", "inc"), SURFACES)
def test_jones_values_are_the_tm_te_block_of_s(cell, out, inc):
    result, model = cell
    j = dif.jones(result, out, inc, model=model)
    pre_out, pre_in = ("R" if out == TOP else "T"), ("R" if inc == TOP else "T")
    for a, p in enumerate(("TM", "TE")):
        for b, q in enumerate(("TM", "TE")):
            np.testing.assert_array_equal(
                j.values[:, a, b], result.s[f"{pre_out}_{p}", f"{pre_in}_{q}"]
            )
    np.testing.assert_array_equal(j.frequency, [F0, 1.1 * F0])
    assert j.values.shape == (2, 2, 2)


@pytest.mark.parametrize(("out", "inc", "h_out", "h_in"), BLOCKS)
def test_mueller_is_the_stokes_measured_matrix(cell, out, inc, h_out, h_in):
    result, model = cell
    j = dif.jones(result, out, inc, model=model)
    m = np.asarray(j.mueller)
    values = np.asarray(j.values)
    assert m.shape == (2, 4, 4)
    assert m.dtype.kind == "f"
    for i in range(values.shape[0]):
        np.testing.assert_allclose(
            m[i], mueller_measured(values[i], h_out, h_in), rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            m[i, 0, 0], 0.5 * np.sum(np.abs(values[i]) ** 2), rtol=0, atol=1e-13
        )


@pytest.mark.parametrize(("out", "inc"), SURFACES)
def test_the_third_stokes_parameter_is_the_circular_power_difference(cell, out, inc):
    result, model = cell
    j = dif.jones(result, out, inc, model=model)
    circ, m = np.asarray(j.circular), np.asarray(j.mueller)
    for c, s_in in ((0, [1.0, 0, 0, 1.0]), (1, [1.0, 0, 0, -1.0])):
        out_stokes = m @ np.asarray(s_in)
        b = circ[:, :, c]
        np.testing.assert_allclose(
            out_stokes[:, 3],
            np.abs(b[:, 0]) ** 2 - np.abs(b[:, 1]) ** 2,
            rtol=0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            out_stokes[:, 0], np.sum(np.abs(b) ** 2, axis=1), rtol=0, atol=1e-13
        )


@pytest.fixture(scope="module", params=[0.0, 30.0], ids=["normal", "30deg"])
def pec_jones(request, tmp_path_factory):
    """Air above a PEC mirror, the Floquet face de-embedded onto the mirror."""
    depth = 2 * pc.D * pc.UM
    ports = [
        gfem.FloquetPort(f"R_{p}", TOP, polarization=p, offset=depth)
        for p in ("TE", "TM")
    ]
    # three lateral cells: the oblique plane wave's cross-polarized leakage is
    # 1.4e-5 in M with two (1.5e-6 with three, measured), above the 1e-5 gate
    lateral = np.linspace(0.0, pc.F1_A, 4)
    cell = sm.periodic_cell(
        (lateral, lateral, np.linspace(0.0, 2 * pc.D, 7)),
        lambda _: "air",
        paired="xy",
    )
    path = sm.write_msh22(tmp_path_factory.mktemp("mirror") / "mirror.msh", cell)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"air": gfem.Material()},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC(BOT),
        ],
        ports=ports,
    )
    k = None
    if request.param:
        theta = math.radians(request.param)
        k = model.bloch_wave_vector(TOP, F0, theta=theta, phi=pc.F1_PHI)
    options = gfem.SolverOptions(order=3, threads=2, linear_solver="direct")
    return model, gfem.solve(model, gfem.Driven([F0], wave_vector=k), options=options)


def test_a_pec_mirror_reflects_with_minus_one_and_swaps_the_handedness(pec_jones):
    model, result = pec_jones
    j = dif.jones(result, TOP, TOP, model=model)
    assert j.direction == (1, -1)
    np.testing.assert_allclose(j.values[0], -np.eye(2), rtol=0, atol=1e-5)
    np.testing.assert_allclose(j.circular[0], -PAULI_SWAP, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        j.mueller[0], np.diag([1.0, 1.0, -1.0, -1.0]), rtol=0, atol=1e-5
    )


@pytest.fixture(scope="module", params=["normal", "oblique"])
def empty(request, mesh):
    """F1's mesh with air everywhere: only the propagation between the faces."""
    materials = {n: gfem.Material() for n in ("sub", "slab", "air")}
    model = pc.f1(mesh).replace(materials=materials)
    k = None
    if request.param == "oblique":
        k = model.bloch_wave_vector(TOP, F0, theta=pc.F1_THETA, phi=pc.F1_PHI)
    problem = gfem.Driven([F0], wave_vector=k)
    return model, gfem.solve(model, problem, options=OPTS), k


def test_an_empty_cell_transmits_the_phase_and_keeps_the_handedness(empty):
    model, result, k = empty
    length = (2 * pc.F1_D + pc.F1_H) * pc.UM
    k0 = 2 * math.pi * F0 / pc.C0
    kz = k0 if k is None else math.sqrt(k0**2 - k[0] ** 2 - k[1] ** 2)
    phase = np.exp(-1j * kz * length)
    for out, inc in ((BOT, TOP), (TOP, BOT)):
        j = dif.jones(result, out, inc, model=model)
        tol = pc.F1_COARSE_TOL
        np.testing.assert_allclose(j.values[0], phase * np.eye(2), rtol=0, atol=tol)
        circ = np.asarray(j.circular[0])
        np.testing.assert_allclose(circ, np.diag(np.diag(circ)), rtol=0, atol=tol)
        np.testing.assert_allclose(circ, phase * np.eye(2), rtol=0, atol=tol)
        np.testing.assert_allclose(j.mueller[0], np.eye(4), rtol=0, atol=tol)
    r = dif.jones(result, TOP, TOP, model=model)
    assert float(jnp.abs(r.values).max()) <= pc.F1_COARSE_TOL


@pytest.fixture(scope="module")
def plate(mesh):
    """The wave plate at 30 degrees under the conical 20/30 incidence."""
    model = pc.waveplate(mesh, math.radians(30.0))
    k = model.bloch_wave_vector(TOP, F0, theta=math.radians(20.0), phi=math.radians(30))
    return model, gfem.solve(model, gfem.Driven([F0], wave_vector=k), options=OPTS)


def test_a_uniaxial_slab_is_admitted_and_conserves_power_on_a_coarse_mesh(plate):
    model, result = plate
    values = np.asarray(result.s.values)[0]
    defect = np.abs(1 - np.sum(np.abs(values) ** 2, axis=0))
    assert defect.max() <= 1e-3
    for surface in (TOP, BOT):
        for pol in ("TE", "TM", "RCP", "unpolarized"):
            bal = dif.power_balance(result, surface, polarization=pol, model=model)
            assert float(jnp.abs(bal.absorptance).max()) <= 1e-3


def test_the_wave_plate_converts_handedness_in_the_circular_view(plate):
    model, result = plate
    j = dif.jones(result, BOT, TOP, model=model)
    circ = np.asarray(j.circular[0])
    # an anisotropic plate mixes RCP and LCP: the off-diagonal circular
    # entries are not small
    assert abs(circ[1, 0]) > 0.1
    np.testing.assert_allclose(
        np.sum(np.abs(circ) ** 2), np.sum(np.abs(np.asarray(j.values[0])) ** 2)
    )


# --- jones: other behaviour -----------------------------------------------------------


def test_jones_is_a_jax_function_of_s_and_a_result_enters_jit(normal, model):
    def circ(result):
        return dif.jones(result, BOT, TOP, model=model).circular

    # Each circular entry is a sum of four products of three factors of modulus
    # at most one (U^H, S, U).
    np.testing.assert_allclose(
        jax.jit(circ)(normal), circ(normal), rtol=0, atol=JIT_MARGIN * (4 * 3) * EPS
    )

    def mueller(result):
        return dif.jones(result, TOP, TOP, model=model).mueller

    np.testing.assert_allclose(
        jax.jit(mueller)(normal), mueller(normal), rtol=0, atol=MUELLER_JIT
    )


def test_jones_is_a_pytree_with_static_labels(normal, model):
    j = dif.jones(normal, BOT, TOP, model=model)
    leaves, tree = jax.tree.flatten(j)
    assert len(leaves) == 1
    again = jax.tree.unflatten(tree, leaves)
    assert again.direction == (-1, -1)
    assert again.outgoing == (BOT, (0, 0))
    np.testing.assert_array_equal(again.values, j.values)


def test_jones_a_sequence_is_the_stack_of_the_per_result_matrices(
    normal, oblique, model
):
    both = [normal, oblique]
    j = dif.jones(both, BOT, TOP, model=model)
    assert j.values.shape == (2, 2, 2, 2)
    assert j.circular.shape == (2, 2, 2, 2)
    assert j.mueller.shape == (2, 2, 4, 4)
    for i, r in enumerate(both):
        one = dif.jones(r, BOT, TOP, model=model)
        np.testing.assert_array_equal(j.values[i], one.values)
        np.testing.assert_allclose(j.circular[i], one.circular, rtol=0, atol=EXACT)
        np.testing.assert_allclose(j.mueller[i], one.mueller, rtol=0, atol=EXACT)


@pytest.fixture(scope="module")
def grating(tmp_path_factory):
    """F2, the lamellar grating, with ports of the orders 0 and -1, 1 on both faces."""
    mesh = pc.f2_mesh(tmp_path_factory.mktemp("grating"))
    return pc.f2(mesh)


def test_jones_an_evanescent_order_reads_zero_without_nan(grating):
    # the orders +-1 propagate in the air at f0 and are evanescent at 0.8 f0
    excite = ["R0,0_TE", "R0,0_TM"]
    result = gfem.solve(
        grating, gfem.Driven([F0, 0.8 * F0], excite=excite), options=OPTS
    )
    j = dif.jones(result, TOP, TOP, model=grating, order=(1, 0))
    values = np.asarray(j.values)
    np.testing.assert_array_equal(values[1], 0.0)
    assert np.abs(values[0]).max() > 1e-2
    for view in (j.values, j.circular, j.mueller):
        assert np.isfinite(np.asarray(view)).all()
    np.testing.assert_array_equal(np.asarray(j.mueller)[1], 0.0)
    assert j.outgoing == (TOP, (1, 0))
    assert j.incident == (TOP, (0, 0))
    # into the substrate the order propagates at both frequencies
    inside = dif.jones(result, BOT, TOP, model=grating, order=(1, 0))
    assert (np.abs(np.asarray(inside.values)).max(axis=(1, 2)) > 1e-2).all()


def test_jones_reads_an_incident_order_other_than_the_specular_one(grating):
    result = gfem.solve(grating, gfem.Driven([F0]), options=OPTS)
    j = dif.jones(result, TOP, TOP, model=grating, order=(0, 0), incident_order=(1, 0))
    assert j.incident == (TOP, (1, 0))
    assert j.outgoing == (TOP, (0, 0))
    s = result.s
    for a, p in enumerate(("TM", "TE")):
        for b, q in enumerate(("TM", "TE")):
            np.testing.assert_array_equal(
                j.values[:, a, b], s[f"R0,0_{p}", f"R1,0_{q}"]
            )
    assert np.abs(np.asarray(j.values)).max() > 1e-2


def test_jones_circular_and_mueller_follow_the_direction_signs():
    values = np.arange(1, 9, dtype=complex).reshape(2, 2, 2) * (1 + 0.5j)
    base = dict(frequency=[F0, 2 * F0], outgoing=(TOP, (0, 0)), incident=(BOT, (0, 0)))
    for h_out, h_in in ((1, -1), (-1, -1), (1, 1), (-1, 1)):
        j = dif.Jones(values=values, direction=(h_out, h_in), **base)
        for i in range(2):
            ref = circular_unitary(h_out).conj().T @ values[i] @ circular_unitary(h_in)
            np.testing.assert_allclose(j.circular[i], ref, rtol=0, atol=EXACT)
            assert j.mueller[i].shape == (4, 4)
    j = dif.Jones(values=jnp.asarray(values), direction=(1, -1), **base)
    assert j.direction == (1, -1)
    assert isinstance(j.frequency, np.ndarray)


def test_jones_the_constructor_refuses_what_it_cannot_read():
    ok = dict(
        frequency=[F0],
        outgoing=(TOP, (0, 0)),
        incident=(TOP, (0, 0)),
        direction=(1, -1),
    )
    good = np.zeros((1, 2, 2), complex)
    for shape in ((2, 2), (1, 2, 3), (1, 3, 3), (1, 2, 2, 1), (4,)):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Jones: values must have shape (..., n_freq, 2, 2), got {shape}."
            ),
        ):
            dif.Jones(values=np.zeros(shape, complex), **ok)
    for bad in ((1, 0), (2, 1), (1,), (1, -1, 1), "ab", 1, (True, -1), (0.5, -1)):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Jones: direction must be two signs (h_out, h_in), each +1 or -1, "
                f"got {bad!r}."
            ),
        ):
            dif.Jones(values=good, **{**ok, "direction": bad})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Jones: frequency must hold 1 frequencies, one per entry of values' "
            "frequency axis, got 2."
        ),
    ):
        dif.Jones(values=good, **{**ok, "frequency": [F0, 2 * F0]})
    # a stacked angle axis: the frequency axis is the third from last
    dif.Jones(values=np.zeros((3, 1, 2, 2), complex), **ok)
    with pytest.raises(ValueError, match="Jones: frequency must hold 2 frequencies"):
        dif.Jones(values=np.zeros((1, 2, 2, 2), complex), **ok)


def test_jones_validates_inside_a_transformation_without_failing_on_tracers():
    def f(v):
        j = dif.Jones(
            values=v,
            frequency=[F0],
            outgoing=(TOP, (0, 0)),
            incident=(TOP, (0, 0)),
            direction=(1, -1),
        )
        return jnp.sum(j.mueller)

    assert np.isfinite(float(jax.jit(f)(jnp.ones((1, 2, 2), complex))))


def test_jones_messages_of_the_inputs(normal, one_excited, model, partial):
    def raises(error, message, result=normal, out=BOT, inc=TOP, **kw):
        kw = {"model": model} | kw
        with pytest.raises(error, match=re.escape(message)):
            dif.jones(result, out, inc, **kw)

    raises(
        TypeError,
        "jones needs a gfem.DrivenResult or a sequence of them (an angle sweep), "
        "got int.",
        result=3,
    )
    raises(TypeError, "jones needs model= a gfem.Model, got str.", model="cell")
    raises(
        ValueError,
        "jones: 'nowhere' carries no Floquet port of the result; its Floquet "
        "surfaces are 'z_max', 'z_min'.",
        out="nowhere",
    )
    raises(
        ValueError,
        "jones: 'nowhere' carries no Floquet port of the result; its Floquet "
        "surfaces are 'z_max', 'z_min'.",
        inc="nowhere",
    )
    raises(
        ValueError,
        "jones: no Floquet port reads order (1, 0) in TM on 'z_min' (a Jones "
        "matrix needs both polarizations of each order). Add "
        "gfem.FloquetPort(name, 'z_min', order=(1, 0), polarization='TM') to "
        "the model.",
        order=(1, 0),
    )
    raises(
        ValueError,
        "jones: no Floquet port reads order (0, 1) in TM on 'z_max' (a Jones "
        "matrix needs both polarizations of each order). Add "
        "gfem.FloquetPort(name, 'z_max', order=(0, 1), polarization='TM') to "
        "the model.",
        incident_order=(0, 1),
    )
    partial_model, partial_result = partial
    raises(
        ValueError,
        "jones: no Floquet port reads order (0, 0) in TM on 'z_min' (a Jones "
        "matrix needs both polarizations of each order). Add "
        "gfem.FloquetPort(name, 'z_min', order=(0, 0), polarization='TM') to "
        "the model.",
        result=partial_result,
        model=partial_model,
    )
    raises(
        ValueError,
        "jones: Floquet port 'R_TM' (order (0, 0), TM on 'z_max') was not "
        "excited, so S has no column for it. Solve with Driven(..., "
        "excite=None) or add 'R_TM' to Driven.excite.",
        result=one_excited,
    )
    raises(
        TypeError,
        "jones: order must be two integers (m, n), got (0.5, 0).",
        order=(0.5, 0),
    )
    raises(
        TypeError,
        "jones: incident_order must be two integers (m, n), got 'ab'.",
        incident_order="ab",
    )
    raises(ValueError, "jones: the sequence of results is empty.", result=[])


def test_jones_two_offsets_of_one_order_are_refused_for_either_side(normal, model):
    for moved, out, inc in (("R_TM", TOP, BOT), ("T_TM", TOP, BOT), ("R_TM", BOT, TOP)):
        result = normal.with_offsets({moved: 1e-7})
        a, b = (moved, moved.replace("TM", "TE"))
        surface = BOT if moved == "T_TM" else TOP
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"jones: Floquet ports {a!r} and {b!r} read order (0, 0) on "
                f"{surface!r} at different offsets (1e-07 m and 0 m), so their "
                "relative phase mixes two reference planes. Give both the same "
                "FloquetPort(offset=...), or align them with result.with_offsets(...)."
            ),
        ):
            dif.jones(result, out, inc, model=model)


# --- power balance of a fed cell ----------------------------------------------------


@pytest.fixture(scope="module")
def fed(tmp_path_factory):
    """A feed sheet in front of a PEC, radiating into one Floquet face.

    The lumped port is a sheet across the whole cell (periodic along y and z),
    so the cell has one Floquet face and one lumped port; the port reads
    power wherever the feed's polarization (z) meets it.
    """
    a = pc.F1_A
    edges = (np.linspace(0.0, 0.5, 6), np.linspace(0.0, a, 3), np.linspace(0.0, a, 3))
    raw = sm.periodic_cell(edges, sheets=(("feed", 0.2),), paired="yz")
    path = sm.write_msh22(tmp_path_factory.mktemp("fed") / "fed.msh", raw)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("y_min", "y_max")),
            gfem.Periodic(("z_min", "z_max")),
            gfem.PEC("x_min"),
        ],
        ports=[
            gfem.LumpedPort("feed", "feed", "+z", impedance=50.0),
            gfem.FloquetPort("TE", "x_max", polarization="TE"),
            gfem.FloquetPort("TM", "x_max", polarization="TM"),
        ],
    )
    return model, gfem.solve(model, gfem.Driven([F0]), options=OPTS)


@pytest.mark.parametrize("polarization", ["TE", "TM", "RCP", "LCP", "unpolarized"])
def test_a_fed_cell_counts_the_power_into_its_feed(fed, polarization):
    model, result = fed
    bal = dif.power_balance(result, "x_max", polarization=polarization, model=model)
    total = bal.reflectance + bal.transmittance + bal.coupled + bal.absorptance
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-12)
    feed = np.asarray(result.s.values)[0, result.s.ports.index("feed")]
    te, tm = (abs(feed[result.s.excited.index(p)]) ** 2 for p in ("TE", "TM"))
    if polarization in ("TE", "TM"):
        expected = {"TE": te, "TM": tm}[polarization]
    else:  # RCP + LCP = TE + TM: the circular pair is a rotation of the linear one
        expected = (te + tm) / 2
    if polarization in ("RCP", "LCP"):
        other = "LCP" if polarization == "RCP" else "RCP"
        pair = dif.power_balance(result, "x_max", polarization=other, model=model)
        np.testing.assert_allclose(
            bal.coupled + pair.coupled, te + tm, rtol=0, atol=1e-12
        )
    else:
        np.testing.assert_allclose(bal.coupled, [expected], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(bal.transmittance, 0.0)


def test_the_feed_of_a_fed_cell_takes_power_in_one_polarization(fed):
    model, result = fed
    bal = dif.power_balance(result, "x_max", polarization="TE", model=model)
    assert float(bal.coupled[0]) > 0.1
    assert float(bal.absorptance[0]) <= pc.F1_POWER_TOL
