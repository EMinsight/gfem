"""Eigenmode solves on periodic models: fields, phase, refinement and refusals.

A periodic eigen field is expanded to the unpaired space and obeys the twin
relation of its pairs, E(r + a) = exp(-i k·a) E(r); a Bloch mode's largest
solve-space coefficient is real and positive. A lossy Bloch cell is refined
two-sided: its left eigenvector solves T(k)^T y = 0, which conj(x) does not.
Plain periodicity solves on every linear solver, and Bloch at k = 0 gives
the plain modes; a Bloch solve refuses every linear solver but ``direct``
before assembly.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._engine import cache
from gfem._engine.driven import driven_operator
from gfem._engine.fields import fe_space
from gfem._sign import largest_entry

MM = pc.MM


@pytest.fixture(scope="module")
def c2_mesh(tmp_path_factory):
    return pc.c2_mesh(tmp_path_factory.mktemp("c2"), cells=(4, 4, 1))


@pytest.fixture(scope="module")
def c3_mesh(tmp_path_factory):
    return pc.c3_mesh(tmp_path_factory.mktemp("c3"))


def _bloch_c2():
    target = 0.9 * pc.C0 * np.linalg.norm(pc.C2_K) / (2 * math.pi * math.sqrt(2))
    return gfem.Eigenmode(target=target, wave_vector=pc.C2_K)


def test_bloch_eigen_fields_live_on_the_unpaired_space(c2_mesh):
    result = gfem.solve(pc.c2(c2_mesh), _bloch_c2())
    (field,) = result.e_fields
    _, space, _ = fe_space(c2_mesh, ("cell",), (), "hcurl", 2)
    assert field.coefficients.shape == (space.ndof,)
    # twin points of the x pair: tangential components advance by the phase
    y = np.array([0.9, 3.1, 5.2]) * MM
    z = np.array([0.2, 0.5, 0.8]) * MM
    first = np.stack([np.zeros(3), y, z], axis=1)
    a = pc.C2_AX * MM
    second = first + np.array([a, 0, 0])
    e1, e2 = np.asarray(field.at(first)), np.asarray(field.at(second))
    scale = np.abs(e1).max()
    assert scale > 0
    phase = np.exp(-1j * pc.C2_K[0] * a)
    np.testing.assert_allclose(e2[:, 1:], phase * e1[:, 1:], rtol=0, atol=1e-8 * scale)


def test_bloch_mode_phase_fixes_its_largest_solve_coefficient(c2_mesh):
    model = pc.c2(c2_mesh)
    options = gfem.SolverOptions()
    rule = eigen_rule.EigenmodeRule(model, _bloch_c2(), options)
    outputs, entry = rule.forward(model, keep=True)
    x = entry.fields["x"][:, 0]
    largest = x[largest_entry(x)]
    assert largest.imag == 0
    assert largest.real > 0
    op = driven_operator(model, options.order, wave_vector=pc.C2_K)
    np.testing.assert_allclose(
        outputs["e_field"][:, 0], op.expansion.matrix @ x, rtol=1e-14
    )
    # scaled to the mode energy: the bulk participations sum to one
    assert outputs["bulk"].sum() == pytest.approx(1.0, rel=1e-12)


def test_lossy_bloch_mode_is_refined_two_sided(c3_mesh):
    # one lossy layer: T(k) is neither symmetric nor Hermitian (nor a
    # Hermitian K and a uniformly lossy M, whose k^2 is real and whose left
    # eigenvector is conj(x)), so conj(x) is not the left eigenvector; the
    # refinement's y is
    low = gfem.Material(eps_r=pc.C3_EPS_LOW, tan_delta=pc.C3_TAN_DELTA)
    model = pc.c3(c3_mesh).with_material("low", low)
    k = 0.5 * math.pi / (pc.C3_A * MM)
    problem = gfem.Eigenmode(target=2.5e9, wave_vector=(k, 0.0, 0.0))
    options = gfem.SolverOptions(order=pc.C3_ORDER)
    rule = eigen_rule.EigenmodeRule(model, problem, options)
    _, entry = rule.forward(model, keep=True)
    x, y, kk = entry.fields["x"][:, 0], entry.fields["y"][:, 0], entry.fields["k"][0]
    op = driven_operator(model, options.order, wave_vector=problem.wave_vector)
    eop = rule.operator(op, kk.real)
    try:
        assert eop.backward_error(kk, x) <= options.eigen_tolerance
        assert eop.backward_error(kk, y, transpose=True) <= options.eigen_tolerance
        assert eop.backward_error(kk, np.conj(x), transpose=True) > 1e3 * (
            options.eigen_tolerance
        )
    finally:
        eop.free_handle()


def test_a_lossy_bloch_cell_returns_its_nearest_band(tmp_path):
    # loss in the rod only: the seed is not exactly divergence-free, and
    # refined at a target far below it, it would drift to the static fields
    mesh = pc.rods_mesh(tmp_path)
    k = (0.5 * math.pi / (pc.ROD_A * MM), 0.0, 0.0)
    problem = gfem.Eigenmode(pc.rods_frequency(0.02), n_modes=1, wave_vector=k)

    def omega(tan_delta):
        model = pc.rods(mesh, tan_delta=tan_delta)
        return np.asarray(gfem.solve(model, problem).omega)

    np.testing.assert_allclose(omega(0.01).real, omega(0.0).real, rtol=1e-3)


def test_a_discretization_keeps_the_spaces_of_eight_bloch_phase_sets(c2_mesh):
    # a band path: every wave vector adds its spaces to the discretization
    model = pc.c2(c2_mesh)
    disc = cache.discretization(model.mesh, model._periodic)
    kx = math.pi / (pc.C2_AX * MM)

    def omega(n):
        k = (0.1 * (n + 1) * kx, 0.0, 0.0)
        target = 0.9 * pc.C0 * k[0] / (2 * math.pi * math.sqrt(2))
        return np.asarray(
            gfem.solve(model, gfem.Eigenmode(target, wave_vector=k)).omega
        )

    first, keys = omega(0), []
    for n in range(1, 10):
        omega(n)
        keys.append(len(disc.spaces))
        assert disc.phase_sets <= 8
    # wave vectors 8, 9 and 10 each replace the oldest phase set
    assert disc.phase_sets == 8
    assert keys[6] == keys[7] == keys[8]
    # the first one's spaces are gone, and built again
    np.testing.assert_allclose(omega(0), first, rtol=1e-10)
    assert len(disc.spaces) == keys[-1]


@pytest.mark.parametrize("linear_solver", ["direct", "bddc_gmres"])
def test_plain_periodic_eigen_solves_on_every_linear_solver(c3_mesh, linear_solver):
    problem = gfem.Eigenmode(target=pc.C3_PLAIN_TARGET, n_modes=2)
    options = gfem.SolverOptions(order=pc.C3_ORDER, linear_solver=linear_solver)
    result = gfem.solve(pc.c3(c3_mesh), problem, options=options)
    f = np.asarray(result.omega) / (2 * math.pi)
    exact = [pc.tmm_frequency(0.0, complex(v)) for v in f]
    np.testing.assert_allclose(f, exact, rtol=pc.C3_FREQ_RTOL)


def test_bloch_eigen_at_k_zero_is_plain_periodicity(c3_mesh):
    model = pc.c3(c3_mesh, pc.C3_TAN_DELTA)
    options = gfem.SolverOptions(order=pc.C3_ORDER)

    def omega(wave_vector):
        problem = gfem.Eigenmode(
            target=pc.C3_PLAIN_TARGET, n_modes=2, wave_vector=wave_vector
        )
        return np.asarray(gfem.solve(model, problem, options=options).omega)

    np.testing.assert_allclose(omega((0.0, 0.0, 0.0)), omega(None), rtol=1e-10)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        (
            {"linear_solver": "bddc_gmres"},
            ValueError,
            (
                r"Eigenmode\.wave_vector sets a Bloch phase, so the operator is "
                r"not symmetric, and linear_solver='bddc_gmres' cannot solve the "
                r"transposed system .*: use linear_solver='direct'\.$"
            ),
        ),
        (
            {"linear_solver": "direct", "device": "cuda"},
            NotImplementedError,
            "device='cpu'",
        ),
    ],
    ids=["bddc_gmres", "cuda"],
)
def test_bloch_eigen_refusals_come_before_assembly(
    c2_mesh, monkeypatch, options, error, message
):
    def assembled(*_, **__):
        raise AssertionError("assembled before the refusal")

    monkeypatch.setattr(eigen_rule, "driven_operator", assembled)
    with pytest.raises(error, match=message):
        gfem.solve(pc.c2(c2_mesh), _bloch_c2(), options=gfem.SolverOptions(**options))


def test_v_g_is_none_without_a_wave_vector_and_a_complex_vector_with_one(
    c2_mesh, c3_mesh
):
    # the lowest Bloch band of the empty C2 cell: a plane wave along k + G
    result = gfem.solve(pc.c2(c2_mesh), _bloch_c2())
    v = np.asarray(result.v_g)
    assert v.shape == (1, 3)
    assert v.dtype == np.complex128
    # lossless: snapped real, and nothing outside the span of the translations
    assert not v.imag.any()
    assert v[0, 2] == 0
    assert np.isfinite(v).all()
    # the same solve under jit-free eager use gives the dataset a "m/s" variable
    xarray = pytest.importorskip("xarray")
    assert isinstance(result.to_xarray(), xarray.Dataset)
    ds = result.to_xarray()
    assert ds["v_g"].dims == ("mode", "axis")
    assert ds["v_g"].attrs["units"] == "m/s"
    # plain periodicity: no wave vector, no group velocity, no variable
    plain = gfem.solve(
        pc.c3(c3_mesh),
        gfem.Eigenmode(target=pc.C3_PLAIN_TARGET, n_modes=2),
        options=gfem.SolverOptions(order=pc.C3_ORDER),
    )
    assert plain.v_g is None
    assert "v_g" not in plain.to_xarray()


def test_the_codec_writes_v_g_only_when_there_is_one():
    import gfem.job  # registers the gfem classes
    from gfem import _codec

    context = _codec.Context(__import__("pathlib").Path())

    def round_trip(result):
        encoder = _codec.Encoder(context)
        doc = encoder.encode(result)
        return doc, _codec.Decoder(context, encoder.arrays).decode(doc)

    plain = gfem.EigenmodeResult(omega=np.array([1e10 + 0j]), error=np.array([1e-12]))
    doc, back = round_trip(plain)
    assert "v_g" not in str(doc)
    assert back.v_g is None
    with_v = gfem.EigenmodeResult(
        omega=np.array([1e10 + 0j]),
        error=np.array([1e-12]),
        v_g=np.array([[1.5e8, 0.0, 0.0]], dtype=complex),
    )
    _, back = round_trip(with_v)
    np.testing.assert_array_equal(back.v_g, with_v.v_g)


def test_v_g_at_order_3_on_two_pairs_matches_the_empty_lattice(c2_mesh):
    # some rows of P mix two lattice vectors at order 3 with two pairs: the
    # derivative of P is taken entry by entry
    periods = (pc.C2_AX * MM, pc.C2_AY * MM, None)
    exact = pc.empty_lattice(pc.C2_K, periods, pc.C2_EPS, n=1)[0]
    problem = gfem.Eigenmode(target=0.9 * exact, n_modes=3, wave_vector=pc.C2_K)
    result = gfem.solve(pc.c2(c2_mesh), problem, options=gfem.SolverOptions(order=3))
    v = np.asarray(result.v_g)
    assert np.isfinite(v).all()
    k = np.asarray(pc.C2_K)
    want = pc.C0 / math.sqrt(pc.C2_EPS) * k / np.linalg.norm(k)
    band = int(np.argmin(np.asarray(result.omega).real))
    assert np.abs(v[band] - want).max() <= 1e-6 * pc.C0
