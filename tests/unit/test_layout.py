"""One process's layout: each product is today's NumPy expression, bit for bit."""

import numpy as np
import pytest

from gfem._linalg.layout import ONE_PROCESS


def _columns(rng, n, k, *, complex_valued):
    values = rng.standard_normal((n, k))
    if complex_valued:
        values = values + 1j * rng.standard_normal((n, k))
    return values


@pytest.mark.parametrize("complex_valued", [True, False])
def test_state_products_are_numpys_on_one_process(complex_valued):
    rng = np.random.default_rng(7)
    A = _columns(rng, 101, 3, complex_valued=complex_valued)
    B = _columns(rng, 101, 2, complex_valued=complex_valued)
    a, b = A[:, 0], B[:, 1]
    inner = ONE_PROCESS.state_inner(a, b)
    assert inner.tobytes() == np.vdot(a, b).tobytes()
    assert ONE_PROCESS.state_gram(A, B).tobytes() == (A.conj().T @ B).tobytes()
    assert ONE_PROCESS.state_norm(a) == float(np.linalg.norm(a))


@pytest.mark.parametrize("complex_valued", [True, False])
def test_state_qr_is_numpys_qr_on_one_process(complex_valued):
    rng = np.random.default_rng(8)
    A = _columns(rng, 64, 3, complex_valued=complex_valued)
    assert ONE_PROCESS.state_qr(A).tobytes() == np.linalg.qr(A)[0].tobytes()


def test_held_by_any_is_the_mask_on_one_process():
    mask = np.array([True, False, False, True])
    held = ONE_PROCESS.held_by_any(mask)
    np.testing.assert_array_equal(held, mask)
    assert held.dtype == bool
