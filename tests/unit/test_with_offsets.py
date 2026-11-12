"""`DrivenResult.with_offsets`: wave-port reference planes moved after the solve.

A lossy two-layer line (complex propagation constants) solved once with its
reference-plane offsets and once without: shifting the second result to the
first one's planes gives the first one's S, from the stored port modes alone.
"""

import jax
import numpy as np
import pytest
from gradient_cases import WAVE_OFFSETS, wave_line, wave_mesh

import gfem
from gfem.job import _write_result

FREQUENCIES = (20e9, 25e9)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
D1, D2 = WAVE_OFFSETS["lossy"]
PLANES = {"p1": D1, "p2": D2}


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return wave_mesh(tmp_path_factory.mktemp("wave"))


def _zero(model):
    return model.replace(ports=[gfem.WavePort(p.name, p.surface) for p in model.ports])


def _solve(model, excite=None):
    return gfem.solve(model, gfem.Driven(FREQUENCIES, excite=excite), options=OPTIONS)


@pytest.fixture(scope="module")
def zero(mesh):
    return _solve(_zero(wave_line(mesh, "lossy")))


def _difference(a, b):
    return np.abs(np.asarray(a.s.values) - np.asarray(b.s.values)).max()


@pytest.mark.parametrize("excite", [None, "p2"], ids=["all", "one"])
def test_shifted_s_is_the_s_of_a_solve_with_those_offsets(mesh, excite):
    model = wave_line(mesh, "lossy")
    solved = _solve(model, excite)
    plain = _solve(_zero(model), excite)
    assert _difference(solved, plain) > 0.1  # the offsets move S
    assert abs(np.asarray(plain.port_modes["p1"].gamma).real).min() > 1.0  # lossy
    shifted = plain.with_offsets(PLANES)
    assert _difference(shifted, solved) <= 1e-12
    assert shifted.offsets == solved.offsets
    np.testing.assert_array_equal(shifted.s.z_ref, plain.s.z_ref)
    assert shifted.port_modes == plain.port_modes
    assert shifted.sweep is plain.sweep
    assert dict(plain.offsets) == {"p1": 0.0, "p2": 0.0}  # a new result


def test_offsets_are_absolute(zero):
    shifted = zero.with_offsets(PLANES)
    assert _difference(shifted.with_offsets(PLANES), shifted) == 0.0
    back = shifted.with_offsets({"p1": 0.0, "p2": 0.0})
    assert _difference(back, zero) <= 1e-13
    assert back.offsets == zero.offsets
    # a port left out keeps its plane
    one = zero.with_offsets({"p2": D2})
    assert dict(one.offsets) == {"p1": 0.0, "p2": D2}
    assert _difference(one.with_offsets({"p1": D1}), shifted) <= 1e-13
    np.testing.assert_array_equal(one.s["p1", "p1"], zero.s["p1", "p1"])


def test_a_loaded_result_shifts_the_same(mesh, zero, tmp_path):
    job = gfem.Job(_zero(wave_line(mesh, "lossy")), gfem.Driven(FREQUENCIES), OPTIONS)
    job.run(out=tmp_path / "zero")
    loaded = gfem.load_result(tmp_path / "zero").with_offsets(PLANES)
    shifted = zero.with_offsets(PLANES)
    assert _difference(loaded, shifted) <= 1e-12
    # and the shifted result's offsets round-trip through a bundle
    _write_result(tmp_path / "shifted", shifted, {}, None)
    again = gfem.load_result(tmp_path / "shifted")
    assert again.offsets == shifted.offsets == (("p1", D1), ("p2", D2))
    np.testing.assert_array_equal(again.s.values, shifted.s.values)
    assert _difference(again.with_offsets({"p1": 0.0, "p2": 0.0}), zero) <= 1e-13


def test_with_offsets_names_what_it_refuses(mesh, zero):
    mixed = _solve(wave_line(mesh, "mixed"))
    assert mixed.with_offsets({"p1": 0.0}).offsets == (("p1", 0.0),)
    with pytest.raises(ValueError, match="'p2' is not a wave port"):
        mixed.with_offsets({"p2": 1e-4})
    with pytest.raises(KeyError, match="No port 'p3'"):
        zero.with_offsets({"p3": 1e-4})
    for bad in (-1e-4, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="'p1'"):
            zero.with_offsets({"p1": bad})
    with pytest.raises(gfem.NonDifferentiableError, match="offset"):
        jax.grad(lambda d: abs(zero.with_offsets({"p1": d}).s["p2", "p1"][0]))(1e-4)
