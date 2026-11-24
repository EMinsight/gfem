"""Fit a tabulated permittivity to a passive law with `gfem.fit_permittivity`.

The fit takes frequencies in Hz and either the refractive index :math:`n` with
its extinction coefficient :math:`k \\ge 0`, or a complex permittivity in gfem's
:math:`e^{+i\\omega t}` convention, and returns a `gfem.PermittivityFit`: a
passive `gfem.PermittivityLaw` with its largest relative errors in
:math:`\\varepsilon`, :math:`n` and :math:`k`. These tests fit synthetic data
drawn from known laws; real tables are in the verification tier.
"""

import dataclasses
import os
import subprocess
import sys
import warnings

import jax
import numpy as np
import pytest

import gfem
import gfem.model

THZ = 1e12


def _eps(f, eps_inf, drude=None, lorentz=()):
    """A law's permittivity, written out independently of gfem."""
    e = np.full(f.shape, eps_inf, dtype=complex)
    if drude is not None:
        fp, g = drude
        e += -(fp**2) / (f * (f - 1j * g))
    for d, f0, g in lorentz:
        e += d * f0**2 / (f0**2 - f**2 + 1j * g * f)
    return e


def _nk(eps):
    """The tabulated index: :math:`n - ik = \\sqrt{\\varepsilon}` with k >= 0."""
    s = np.sqrt(eps)
    s = np.where(s.imag > 0, -s, s)
    return s.real, -s.imag


# --- a lossy Lorentz law, 30 points ------------------------------------------------

LOSSY_F = np.linspace(100, 500, 30) * THZ
LOSSY = (2.0, None, [(3.0, 300 * THZ, 30 * THZ)])


@pytest.fixture(scope="module")
def lossy_fit():
    return gfem.fit_permittivity(LOSSY_F, eps=_eps(LOSSY_F, *LOSSY), tolerance=1e-10)


def test_a_lossy_lorentz_law_is_recovered(lossy_fit):
    law = lossy_fit.law
    assert len(law.terms) == 1
    (term,) = law.terms
    assert isinstance(term, gfem.Lorentz)
    np.testing.assert_allclose(law.eps_inf, 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        [term.delta_eps, term.resonance_frequency, term.damping],
        [3.0, 300 * THZ, 30 * THZ],
        rtol=1e-6,
    )
    assert lossy_fit.eps_error < 1e-8
    assert law.frequency_range == (100 * THZ, 500 * THZ)


# --- a Drude + Lorentz metal from n and k ------------------------------------------

METAL_F = np.linspace(180, 750, 40) * THZ
METAL = (1.0, (2000 * THZ, 20 * THZ), [(1.5, 700 * THZ, 150 * THZ)])


@pytest.fixture(scope="module")
def metal_fit():
    n, k = _nk(_eps(METAL_F, *METAL))
    return gfem.fit_permittivity(METAL_F, n=n, k=k, drude=True, tolerance=1e-10)


def test_a_drude_lorentz_metal_is_recovered_from_n_and_k(metal_fit):
    law = metal_fit.law
    drude, lorentz = law.terms
    assert isinstance(drude, gfem.Drude)
    assert isinstance(lorentz, gfem.Lorentz)
    np.testing.assert_allclose(law.eps_inf, 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        [drude.plasma_frequency, drude.damping], [2000 * THZ, 20 * THZ], rtol=1e-6
    )
    np.testing.assert_allclose(
        [lorentz.delta_eps, lorentz.resonance_frequency, lorentz.damping],
        [1.5, 700 * THZ, 150 * THZ],
        rtol=1e-6,
    )
    assert metal_fit.eps_error < 1e-8
    assert metal_fit.n_error < 1e-6
    assert metal_fit.k_error < 1e-6


def test_eps_input_gives_the_same_law_as_n_and_k(metal_fit):
    n, k = _nk(_eps(METAL_F, *METAL))
    eps = (n - 1j * k) ** 2
    fit = gfem.fit_permittivity(METAL_F, eps=eps, drude=True, tolerance=1e-10)
    assert fit.law == metal_fit.law


# --- a transparent table -----------------------------------------------------------

#: A silica-like law with its resonance above the data, at 20 points.
CLEAR_F = np.linspace(150, 600, 20) * THZ
CLEAR = (1.0, None, [(1.1, 2400 * THZ, 0.0)])


@pytest.fixture(scope="module")
def clear_fit():
    n, _ = _nk(_eps(CLEAR_F, *CLEAR))
    return gfem.fit_permittivity(CLEAR_F, n=n, tolerance=1e-10)


def test_a_transparent_table_gives_a_lossless_law(clear_fit):
    law = clear_fit.law
    assert all(t.damping == 0 for t in law.terms)
    assert clear_fit.k_error == 0.0
    dense = np.geomspace(CLEAR_F[0] / 1e3, CLEAR_F[-1] * 1e3, 20000)
    dense = dense[np.abs(dense / 2400e12 - 1) > 1e-6]
    assert np.all(np.asarray(law(dense)).imag == 0)
    assert clear_fit.eps_error < 1e-8
    (term,) = law.terms
    np.testing.assert_allclose(
        [law.eps_inf, term.delta_eps, term.resonance_frequency],
        [1.0, 1.1, 2400 * THZ],
        rtol=1e-6,
    )


def test_order_and_repeat_do_not_change_the_law(lossy_fit):
    eps = _eps(LOSSY_F, *LOSSY)
    again = gfem.fit_permittivity(LOSSY_F, eps=eps, tolerance=1e-10)
    reversed_ = gfem.fit_permittivity(LOSSY_F[::-1], eps=eps[::-1], tolerance=1e-10)
    assert again.law == lossy_fit.law
    assert reversed_.law == lossy_fit.law


def test_every_fitted_law_is_passive_on_a_dense_grid(lossy_fit, metal_fit, clear_fit):
    for fit in (lossy_fit, metal_fit, clear_fit):
        lo, hi = fit.law.frequency_range
        dense = np.geomspace(lo / 1e3, hi * 1e3, 20000)
        assert np.all(np.asarray(fit.law(dense)).imag <= 0)


# --- the record --------------------------------------------------------------------


def test_the_repr_reports_terms_errors_and_range(metal_fit, clear_fit):
    text = repr(metal_fit)
    assert text.startswith("PermittivityFit(Drude + 1 Lorentz;")
    for part in ("eps ", "n ", "k ", " %", "1.8e+14-7.5e+14 Hz"):
        assert part in text
    assert "1 Lorentz" in repr(clear_fit)
    assert "k 0 %" in repr(clear_fit)


def test_the_record_is_frozen_and_the_law_differentiates(lossy_fit):
    with pytest.raises(dataclasses.FrozenInstanceError):
        lossy_fit.eps_error = 0.0  # type: ignore[misc]
    grad = jax.grad(lambda law: law(300 * THZ).imag)(lossy_fit.law)
    assert isinstance(grad, gfem.PermittivityLaw)
    assert grad.frequency_range == lossy_fit.law.frequency_range
    assert grad.terms[0].damping != 0


# --- warnings ----------------------------------------------------------------------

#: Two weak resonances: Re eps stays > 0, so the remedies do not suggest a metal.
TWO = (2.0, None, [(0.5, 200 * THZ, 40 * THZ), (0.3, 450 * THZ, 60 * THZ)])


def test_too_few_terms_warns_and_returns_the_best():
    eps = _eps(LOSSY_F, *TWO)
    with pytest.warns(gfem.PermittivityFitWarning) as record:
        fit = gfem.fit_permittivity(LOSSY_F, eps=eps, max_lorentz_terms=1)
    assert len(record) == 1
    message = str(record[0].message)
    assert len(fit.law.terms) == 1
    assert fit.eps_error > 1e-2
    assert f"{100 * fit.eps_error:.2g} %" in message
    assert "1 Lorentz" in message
    assert "max_lorentz_terms" in message
    assert "narrow the band" in message
    assert "raise tolerance" in message
    assert "drude=True" not in message  # Re eps > 0: not a metal


def test_loss_where_the_table_has_none_warns():
    n, k = _nk(_eps(LOSSY_F, *LOSSY))
    k = np.where(LOSSY_F < 200 * THZ, 0.0, k)
    with pytest.warns(gfem.PermittivityFitWarning) as record:
        fit = gfem.fit_permittivity(LOSSY_F, n=n, k=k)
    assert len(record) == 1
    assert fit.k_error == np.inf
    assert "inf (loss where the data have k = 0)" in repr(fit)
    message = str(record[0].message)
    assert "where the data have k = 0" in message
    assert "absorption edge" in message
    assert "without k" in message


def test_a_fit_within_tolerance_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gfem.fit_permittivity(LOSSY_F, eps=_eps(LOSSY_F, *LOSSY))


# --- refusals ----------------------------------------------------------------------

F8 = np.linspace(100, 500, 8) * THZ
N8 = np.linspace(1.5, 1.6, 8)
K8 = np.linspace(0.1, 0.2, 8)
E8 = (N8 - 1j * K8) ** 2


def _with(arr, i, value):
    out = np.array(arr, dtype=np.result_type(arr, value))
    out[i] = value
    return out


REFUSALS = {
    "neither": (TypeError, "not both", (F8,), {}),
    "both": (TypeError, "not both", (F8,), {"n": N8, "eps": E8}),
    "k with eps": (TypeError, "k goes with n", (F8,), {"k": K8, "eps": E8}),
    "k alone": (TypeError, "k goes with n", (F8,), {"k": K8}),
    "2-D": (ValueError, r"n \(2, 4\)", (F8,), {"n": N8.reshape(2, 4)}),
    "lengths": (ValueError, r"k \(7,\)", (F8,), {"n": N8, "k": K8[:7]}),
    "complex n": (ValueError, "n must be real", (F8,), {"n": N8 + 0j}),
    "complex frequency": (ValueError, "frequency must be real", (F8 + 0j,), {"n": N8}),
    "text eps": (ValueError, "eps must be numbers", (F8,), {"eps": ["a"] * 8}),
    "nan": (
        ValueError,
        r"k has 1 NaN.*index 3.*np.isfinite",
        (F8,),
        {"n": N8, "k": _with(K8, 3, np.nan)},
    ),
    "zero frequency": (ValueError, "must be > 0 Hz", (_with(F8, 0, 0.0),), {"n": N8}),
    "duplicate": (
        ValueError,
        "appears more than once; average or drop",
        (_with(F8, 1, F8[0]),),
        {"n": N8},
    ),
    "negative n": (ValueError, "negative index", (F8,), {"n": _with(N8, 2, -1.0)}),
    "some gain": (
        ValueError,
        r"is gain.*k=np.clip\(k, 0, None\)",
        (F8,),
        {"n": N8, "k": _with(K8, 2, -1e-6)},
    ),
    "sign flipped": (ValueError, "pass k=-k", (F8,), {"n": N8, "k": -K8}),
    "e^{-iwt} eps": (ValueError, r"np.conj\(eps\)", (F8,), {"eps": np.conj(E8)}),
    "eps gain": (
        ValueError,
        r"is gain.*np.minimum\(eps.imag, 0\)",
        (F8,),
        {"eps": _with(E8, 2, 2.0 + 1e-6j)},
    ),
    "eps zero": (
        ValueError,
        "undefined where eps = 0",
        (F8,),
        {"eps": _with(E8, 2, 0)},
    ),
    "n = k = 0": (
        ValueError,
        "undefined where eps = 0",
        (F8,),
        {"n": _with(N8, 2, 0.0), "k": _with(K8, 2, 0.0)},
    ),
    "too few damped": (
        ValueError,
        r"at least 8 data points for one Lorentz term \(twice its 4",
        (F8[:7],),
        {"n": N8[:7], "k": K8[:7]},
    ),
    "too few lossless": (
        ValueError,
        "at least 6 data points",
        (F8[:5],),
        {"n": N8[:5]},
    ),
    "too few damped Drude": (
        ValueError,
        r"at least 6 data points for a Drude term \(twice its 3",
        (F8[:5],),
        {"n": N8[:5], "k": K8[:5], "drude": True},
    ),
    "too few lossless Drude": (
        ValueError,
        "at least 4 data points for a Drude term",
        (F8[:3],),
        {"n": N8[:3], "drude": True},
    ),
    "empty": (ValueError, "at least 6 data points", (np.zeros(0),), {"n": np.zeros(0)}),
    "terms above the cap": (
        ValueError,
        "max_lorentz_terms=2 needs at least 14 points, got 8: use max_lorentz_terms<=1",
        (F8,),
        {"n": N8, "k": K8, "max_lorentz_terms": 2},
    ),
    "terms negative": (
        ValueError,
        "max_lorentz_terms",
        (F8,),
        {"n": N8, "max_lorentz_terms": -1},
    ),
    "terms float": (
        ValueError,
        "max_lorentz_terms",
        (F8,),
        {"n": N8, "max_lorentz_terms": 1.0},
    ),
    "terms bool": (
        ValueError,
        "max_lorentz_terms",
        (F8,),
        {"n": N8, "max_lorentz_terms": True},
    ),
    "no term": (
        ValueError,
        "the law would have no term",
        (F8,),
        {"n": N8, "max_lorentz_terms": 0},
    ),
    "tolerance 0": (
        ValueError,
        "tolerance must be > 0",
        (F8,),
        {"n": N8, "tolerance": 0},
    ),
    "tolerance nan": (
        ValueError,
        "tolerance must be > 0",
        (F8,),
        {"n": N8, "tolerance": np.nan},
    ),
    "drude not bool": (
        TypeError,
        "drude takes True or False",
        (F8,),
        {"n": N8, "drude": 1},
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_unusable_input_is_refused_before_fitting(case, monkeypatch):
    error, match, args, kwargs = REFUSALS[case]
    copies = [np.array(a, copy=True) for a in args] + [
        np.array(v, copy=True) for v in kwargs.values() if not np.isscalar(v)
    ]

    def no_fit(*_, **__):
        raise AssertionError("fitting started")

    monkeypatch.setattr("gfem._permittivity_fit._greedy", no_fit)
    with pytest.raises(error, match=match) as info:
        gfem.fit_permittivity(*args, **kwargs)
    assert str(info.value).startswith("fit_permittivity")
    given = list(args) + [v for v in kwargs.values() if not np.isscalar(v)]
    for before, after in zip(copies, given, strict=True):
        np.testing.assert_array_equal(np.asarray(after), before)


def test_a_traced_input_is_refused():
    eps = _eps(LOSSY_F, *LOSSY)
    with pytest.raises(gfem.NonDifferentiableError, match="not differentiable"):
        jax.grad(lambda x: gfem.fit_permittivity(LOSSY_F, eps=eps * x).eps_error)(1.0)
    with pytest.raises(gfem.NonDifferentiableError, match=r"outside jax\.grad/jit"):
        jax.jit(lambda n: gfem.fit_permittivity(LOSSY_F, n=n).eps_error)(eps.real)
    with pytest.raises(gfem.NonDifferentiableError):
        jax.jit(lambda f: gfem.fit_permittivity(f, eps=eps).eps_error)(LOSSY_F)
    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(
            lambda t: gfem.fit_permittivity(LOSSY_F, eps=eps, tolerance=t).eps_error
        )(1e-2)


def test_the_point_cap_counts_per_mode():
    f = np.linspace(100, 500, 8) * THZ
    # a broad resonance: the 8 points' linewidth floor is 2 x 57 THz
    n, k = _nk(_eps(f, 2.0, None, [(1.0, 300 * THZ, 150 * THZ)]))
    with pytest.raises(ValueError, match="at least 8"):
        gfem.fit_permittivity(f[:7], n=n[:7], k=k[:7])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(gfem.fit_permittivity(f, n=n, k=k).law.terms) == 1
    n_clear, _ = _nk(_eps(f, *CLEAR))
    with pytest.raises(ValueError, match="at least 6"):
        gfem.fit_permittivity(f[:5], n=n_clear[:5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(gfem.fit_permittivity(f[:6], n=n_clear[:6]).law.terms) == 1


# --- the API -----------------------------------------------------------------------


def test_the_fit_names_sit_beside_the_law_in_the_api():
    assert {"fit_permittivity", "PermittivityFit"} <= set(gfem.model.__all__)
    assert {"fit_permittivity", "PermittivityFit", "PermittivityFitWarning"} <= set(
        gfem.__all__
    )
    assert gfem.fit_permittivity is gfem.model.fit_permittivity
    for warning in (gfem.PermittivityFitWarning, gfem.FittedRangeWarning):
        assert warning.__module__ == "gfem.errors"
        assert issubclass(warning, UserWarning)


def test_importing_gfem_leaves_scipy_optimize_unloaded():
    code = "import gfem, sys; assert 'scipy.optimize' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
