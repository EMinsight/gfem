"""The cell response: evaluation, validation, codec, and the build on a ridge."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem import diffraction as dif
from gfem._diffraction.outputs import Jones, jones_block

PORTS = ("sub TM", "sub TE", "air TM", "air TE")
EXCITED = ("sub TM", "sub TE")
GRID = np.array([0.17, 0.2, 0.26, 0.27, 0.35, 0.4])
FREQ = np.array([2.0e14, 3.0e14])


def _cubic(p):
    """A complex cubic in p per entry of (n_f, ports, excited) and its derivative."""
    shape = (len(FREQ), len(PORTS), len(EXCITED))
    a = (np.arange(np.prod(shape)).reshape(shape) + 1.0) / 10
    c = [a * (1 + 0.5j), a[::-1] * (0.3 - 1j), 0.7 * a * (2 - 0.1j), a * (0.2 + 0.4j)]
    p = jnp.asarray(p)
    val = c[0] + c[1] * p + c[2] * p**2 + c[3] * p**3
    return val, c[1] + 2 * c[2] * p + 3 * c[3] * p**2


def _response(grid=GRID, **over):
    vals = np.stack([np.asarray(_cubic(p)[0]) for p in grid])
    ders = np.stack([np.asarray(_cubic(p)[1]) for p in grid])
    orders = {
        name: gfem.DiffractionOrderResult(
            port=name,
            frequency=FREQ,
            order=(0, 0),
            polarization=name.split()[1],
            surface="z_min" if name.startswith("sub") else "z_max",
            index=1.5,
            wave_vector=np.zeros((2, 3)),
            gamma=1j * np.ones(2),
            propagating=np.ones(2, bool),
            theta=np.zeros(2),
            phi=np.zeros(2),
        )
        for name in PORTS
    }
    lat = ((5e-7, 0.0, 0.0), (0.0, 1.6e-7, 0.0))
    kw = {
        "parameter": grid,
        "frequency": FREQ,
        "values": vals,
        "derivative": ders,
        "z_ref": np.ones((2, 4)),
        "ports": PORTS,
        "excited": EXCITED,
        "diffraction_orders": orders,
        "lattice": {"z_min": lat, "z_max": lat},
        "direction": {"z_min": -1, "z_max": 1},
        "wave_vector": None,
        "offsets": {"sub TM": 3e-7, "sub TE": 3e-7, "air TM": 3.5e-7},
    }
    kw.update(over)
    return dif.CellResponse(**kw)


def test_r5_the_cubic_is_reproduced_exactly_on_a_non_uniform_grid():
    r = _response()
    for p in np.linspace(GRID[0], GRID[-1], 50):
        np.testing.assert_allclose(r(p).values, _cubic(p)[0], rtol=0, atol=1e-13)
    for i, p in enumerate(GRID):
        np.testing.assert_array_equal(r(p).values, r.values[i])
    assert r(0.3).ports == PORTS
    assert r(0.3).excited == EXCITED


def test_r5_the_gradient_at_the_knots_is_the_stored_derivative_and_is_continuous():
    r = _response()

    def d(p):
        re = jax.jacfwd(lambda q: r(q).values.real)(p)
        return re + 1j * jax.jacfwd(lambda q: r(q).values.imag)(p)

    for i, p in enumerate(GRID):
        np.testing.assert_allclose(d(p), r.derivative[i], rtol=0, atol=1e-13)
        eps = 1e-9
        if 0 < i < len(GRID) - 1:
            np.testing.assert_allclose(d(p - eps), d(p + eps), rtol=0, atol=1e-7)
    # reverse mode, scalar objective
    g = jax.grad(lambda q: jnp.sum(r(q).values.real))(GRID[2])
    np.testing.assert_allclose(g, np.sum(r.derivative[2].real), atol=1e-13)


def test_r6_out_of_range_raises_when_concrete_and_is_nan_when_traced():
    r = _response()
    with pytest.raises(ValueError, match=r"1 of 1 parameter values lie outside"):
        r(0.5)
    with pytest.raises(ValueError, match="does not extrapolate"):
        r(0.1)

    def f(p):
        return r(p).values[0, 0, 0]

    assert np.isnan(jax.jit(f)(0.5))
    assert np.isnan(jax.vmap(f)(jnp.array([0.3, 0.5]))[1])
    assert np.isfinite(jax.vmap(f)(jnp.array([0.3, 0.5]))[0])
    assert np.isnan(jax.grad(lambda p: f(p).real)(0.5))
    assert np.isnan(jax.jit(jax.grad(lambda p: f(p).real))(0.1))
    assert np.isfinite(jax.jit(jax.grad(lambda p: f(p).real))(0.3))
    with pytest.raises(ValueError, match="scalar"):
        r(np.array([0.2, 0.3]))


def test_r6_json_round_trip_is_bitwise(tmp_path):
    r = _response()
    path = r.to_json(tmp_path / "sub" / "r.json")
    back = dif.CellResponse.from_json(path)
    for name in ("values", "derivative"):
        np.testing.assert_array_equal(getattr(back, name), getattr(r, name))
    for name in ("parameter", "frequency", "z_ref"):
        np.testing.assert_array_equal(getattr(back, name), getattr(r, name))
    assert back.ports == r.ports
    assert back.excited == r.excited
    assert back.lattice == r.lattice
    assert back.direction == r.direction
    assert back.wave_vector is None
    assert (
        back.offsets
        == r.offsets
        == {
            "sub TM": 3e-7,
            "sub TE": 3e-7,
            "air TM": 3.5e-7,
            "air TE": 0.0,
        }
    )
    assert set(back.diffraction_orders) == set(r.diffraction_orders)
    d0, d1 = r.diffraction_orders["sub TE"], back.diffraction_orders["sub TE"]
    assert (d1.order, d1.surface, d1.polarization, d1.index) == (
        d0.order,
        d0.surface,
        d0.polarization,
        d0.index,
    )
    np.testing.assert_array_equal(d1.gamma, d0.gamma)
    np.testing.assert_array_equal(back(0.3).values, r(0.3).values)
    with pytest.raises(ValueError, match="not a"):
        gfem.Job.from_json(path)


def test_r6_the_block_extractor_on_the_response_is_the_jones_of_the_stored_values():
    r = _response()
    out, inc = ("sub TM", "sub TE"), ("sub TM", "sub TE")
    p = GRID[3]
    block = jones_block(r(p), out, inc)
    ref = Jones(
        values=r.values[3][:, :2, :2],
        frequency=FREQ,
        outgoing=("z_min", (0, 0)),
        incident=("z_min", (0, 0)),
        direction=(r.direction["z_min"], -r.direction["z_min"]),
    )
    np.testing.assert_array_equal(block, ref.values)
    block = jones_block(r(p), ("air TM", "air TE"), inc)
    np.testing.assert_array_equal(block, r.values[3][:, 2:, :])


def test_constructor_refusals():
    with pytest.raises(
        ValueError, match=r"CellResponse: parameter must be a flat, strictly increasing"
    ):
        _response(parameter=GRID[::-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        _response(parameter=np.array([0.2, 0.2, 0.3, 0.4, 0.5, 0.6]))
    with pytest.raises(ValueError, match="strictly increasing"):
        _response(parameter=np.array([0.2, np.nan, 0.3, 0.4, 0.5, 0.6]))
    with pytest.raises(
        ValueError,
        match=(
            r"CellResponse: values and derivative must both have shape "
            r"\(n_parameter, n_frequency, n_ports, n_excited\) = \(6, 2, 4, 2\), "
            r"got \(6, 2, 4, 2\) and \(5, 2, 4, 2\)"
        ),
    ):
        _response(derivative=_response().derivative[:5])
    with pytest.raises(ValueError, match="values and derivative must both have shape"):
        _response(values=_response().values[..., :1])
    r = _response()
    orders = dict(r.diffraction_orders)
    del orders["air TE"]
    with pytest.raises(
        ValueError,
        match=r"CellResponse: diffraction_orders lacks Floquet port\(s\) \['air TE'\]",
    ):
        _response(diffraction_orders=orders)
    with pytest.raises(ValueError, match="direction must map each Floquet surface to"):
        _response(direction={"z_min": -1, "z_max": 2})
    with pytest.raises(ValueError, match="direction must map each Floquet surface to"):
        _response(direction={"z_min": -1})
    with pytest.raises(ValueError, match=r"frequency must hold 2 frequencies > 0 Hz"):
        _response(frequency=np.array([1e14]))
    with pytest.raises(ValueError, match=r"frequency must hold 2 frequencies > 0 Hz"):
        _response(frequency=np.array([1e14, -1.0]))
    bad = {"z_min": ((1.0, 0.0), (0.0, 1.0, 0.0)), "z_max": r.lattice["z_max"]}
    with pytest.raises(
        ValueError,
        match=r"lattice vectors of 'z_min' must be two finite 3-vectors in m",
    ):
        _response(lattice=bad)
    nan = {"z_min": r.lattice["z_min"], "z_max": ((np.nan, 0, 0), (0, 1, 0))}
    with pytest.raises(ValueError, match="lattice vectors of 'z_max'"):
        _response(lattice=nan)
    with pytest.raises(ValueError, match="wave_vector must be"):
        _response(wave_vector=(1.0, 2.0))
    for offsets in ({"top": 0.0}, {"sub TE": -1e-7}, {"sub TE": np.nan}, None):
        with pytest.raises(
            ValueError,
            match=r"CellResponse: offsets must map ports of the response to finite "
            r"distances >= 0 m",
        ):
            _response(offsets=offsets)
    assert _response(offsets={}).offsets == dict.fromkeys(PORTS, 0.0)
    with pytest.raises(gfem.NonDifferentiableError, match="offsets must be fixed"):
        jax.jit(lambda d: dataclasses.replace(r, offsets={"sub TE": d}).values)(1e-7)


def test_a_one_sample_response_evaluates_only_at_its_sample():
    one = _response(grid=GRID[:1])
    np.testing.assert_array_equal(one(GRID[0]).values, one.values[0])
    with pytest.raises(ValueError, match="outside the sampled range"):
        one(GRID[0] + 0.01)
    g = jax.grad(lambda p: jnp.sum(one(p).values.real))(GRID[0])
    np.testing.assert_allclose(g, np.sum(one.derivative[0].real), atol=1e-13)


def test_the_response_is_a_pytree_differentiable_in_its_arrays():
    r = _response()

    def objective(vals):
        return jnp.sum(jnp.abs(dataclasses.replace(r, values=vals)(0.3).values) ** 2)

    g = jax.grad(objective)(r.values)
    assert g.shape == r.values.shape
    assert np.all(np.isfinite(np.asarray(g)))
    leaves = jax.tree.leaves(r)
    assert len(leaves) == 2
    assert jax.jit(lambda resp, p: resp(p).values)(r, 0.3).shape == (2, 4, 2)


# --- the build, on the ridge -------------------------------------------------------

import periodic_cases as pc  # noqa: E402

UM = 1e-6
EXCITE = ["sub TE", "sub TM"]
W0 = pc.RIDGE_W0
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=2)


@pytest.fixture(scope="module")
def ridge_cell(tmp_path_factory):
    mesh = pc.ridge_mesh(tmp_path_factory.mktemp("ridge"))
    base = pc.ridge(mesh)
    morph = gfem.shape.boundary_morph(base, "sidewall", grows="atom")
    wm = jnp.asarray(pc.ridge_width_map(morph))

    def make_cell(p, dp):
        return base.replace(mesh=morph((p - W0 + dp) * UM * wm))

    return make_cell


@pytest.fixture
def solves(monkeypatch):
    count = []
    real = gfem.solve

    def counted(*args, **kwargs):
        count.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gfem, "solve", counted)
    return count


def _build(make_cell, samples, **kw):
    return dif.cell_response(
        make_cell,
        samples,
        frequency=[pc.RIDGE_FREQUENCY],
        excite=EXCITE,
        options=OPTIONS,
        **kw,
    )


def test_r7_one_sample_equals_the_solve_and_a_rerun_solves_nothing(
    ridge_cell, tmp_path, solves
):
    make_cell = ridge_cell
    out = tmp_path / "ridge.json"
    response = _build(make_cell, [W0], out=out)
    assert len(solves) == 1
    result = gfem.solve(
        make_cell(W0, 0.0),
        gfem.Driven([pc.RIDGE_FREQUENCY], excite=EXCITE),
        options=OPTIONS,
    )
    ref = result.s
    assert response.ports == ref.ports == ("sub TE", "sub TM", "air TE", "air TM")
    assert response.excited == tuple(EXCITE)
    np.testing.assert_allclose(response(W0).values, ref.values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(response.z_ref, np.asarray(ref.z_ref), rtol=0, atol=0)
    assert response.direction == {"z_min": -1, "z_max": 1}
    assert response.wave_vector is None
    assert response.offsets == {n: dict(result.offsets)[n] for n in response.ports}
    assert response.offsets["air TE"] == pytest.approx(pc.RIDGE_AIR * UM)
    assert response.diffraction_orders["sub TE"].frequency.shape == (1,)
    # the block of the response at its sample is the Jones of the solve behind it
    pair = ("sub TM", "sub TE")
    np.testing.assert_allclose(
        jones_block(response(W0), pair, pair),
        dif.jones(result, "z_min", "z_min", model=make_cell(W0, 0.0)).values,
        rtol=0,
        atol=1e-12,
    )
    assert np.abs(response.derivative).max() > 0
    solves.clear()
    again = _build(make_cell, [W0], out=out)
    assert solves == []
    np.testing.assert_array_equal(again.values, response.values)
    np.testing.assert_array_equal(again.derivative, response.derivative)
    # a longer grid resumes: one new solve
    more = _build(make_cell, [W0, W0 + 0.01], out=out)
    assert len(solves) == 1
    np.testing.assert_array_equal(more.values[0], response.values[0])
    assert dif.CellResponse.from_json(out).parameter.tolist() == [W0, W0 + 0.01]

    def moved_planes(p, dp):
        model = make_cell(p, dp)
        ports = [dataclasses.replace(q, offset=2 * q.offset) for q in model.ports]
        return model.replace(ports=ports)

    solves.clear()
    with pytest.raises(
        ValueError, match=r"holds a cell response with other port offsets"
    ):
        _build(moved_planes, [W0, W0 + 0.01], out=out)
    assert solves == []


def test_r7_a_sample_that_changes_the_substrate_medium_raises_cr5(ridge_cell, solves):
    make_cell = ridge_cell

    def changing(p, dp):
        model = make_cell(p, dp)
        return model.with_material("substrate", gfem.Material(eps_r=2.1 + (p - W0)))

    with pytest.raises(
        ValueError, match=r"differs from the cell at 0\.25 in its half-space media"
    ):
        _build(changing, [W0, W0 + 0.01])
    assert len(solves) == 1  # sample 0; the check precedes sample 1's solve


def test_r7_a_traced_design_or_sample_raises_before_any_solve(ridge_cell, solves):
    make_cell = ridge_cell
    with pytest.raises(gfem.NonDifferentiableError, match="cell's design is traced"):
        jax.jit(lambda p: _build(make_cell, [p]).values)(W0)

    def traced_design(eps):
        def make(p, dp):
            model = make_cell(p, dp)
            n = len(model.mesh.elements("atom"))
            return model.with_material(
                "atom", gfem.MaterialField(eps_r=eps * jnp.ones(n))
            )

        return _build(make, [W0]).values

    with pytest.raises(
        gfem.NonDifferentiableError, match="second derivatives of the solve"
    ):
        jax.grad(lambda e: jnp.sum(traced_design(e).real))(5.0)
    assert solves == []


def test_r7_input_refusals(ridge_cell, tmp_path, solves):
    make_cell = ridge_cell
    with pytest.raises(
        TypeError,
        match=r"cell_response needs make_cell, a function \(p, dp\) -> gfem.Model, got",
    ):
        _build(3, [W0])
    with pytest.raises(
        ValueError,
        match=r"cell_response: parameter must be a flat, strictly increasing",
    ):
        _build(make_cell, [0.3, 0.2])
    with pytest.raises(
        TypeError, match=r"make_cell\(0\.25, dp\) returned str, not a gfem.Model"
    ):
        _build(lambda _p, _dp: "x", [W0])
    with pytest.raises(
        ValueError, match=r"cell_response: the cell has no Floquet port"
    ):
        _build(lambda p, dp: make_cell(p, dp).replace(ports=[]), [W0])
    for bad in ([], ["nope"], "sub TE"):
        with pytest.raises(
            ValueError, match="excite must name at least one port of the cell"
        ):
            dif.cell_response(
                make_cell, [W0], frequency=[pc.RIDGE_FREQUENCY], excite=bad
            )
    # CR6: an out file with other samples
    out = tmp_path / "other.json"
    _response(grid=GRID).to_json(out)
    with pytest.raises(
        ValueError,
        match=r"holds a cell response with other (parameter values|frequencies)",
    ):
        _build(make_cell, [W0], out=out)
    assert solves == []
