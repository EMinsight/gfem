"""`gfem.fit_permittivity` on measured tables.

The tables are CC0 excerpts of the refractiveindex.info database (Polyanskiy,
Sci. Data 11, 94 (2024)), each citing its paper in its header: McPeak et al.
2015 gold and aluminium films, Li 1980 silicon at 293 K and Schinke et al.
2015 silicon. The numbers they check (about 1 % in eps with a Drude term and
four Lorentz terms on a dense metal table, an n error of 13-30 % near gold's
index minimum, a k error far above the eps error across silicon's absorption
edge) are the ones the fit's docstring states.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c, e, h

import gfem

ROOT = Path(__file__).parents[2]
DATA = Path(__file__).parent / "data"
EV = e / h  # Hz per eV


def _table(path, low_um=0.0, high_um=np.inf):
    """Frequency (Hz) and the table's other columns, over a band in µm."""
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    rows = rows[(rows[:, 0] >= low_um - 1e-9) & (rows[:, 0] <= high_um + 1e-9)]
    return c / (rows[:, 0] * 1e-6), rows[:, 1:].T


def _quiet_fit(frequency, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return gfem.fit_permittivity(frequency, **kwargs)


def _passive(law):
    lo, hi = law.frequency_range
    dense = np.geomspace(lo / 1e3, hi * 1e3, 20000)
    return bool(np.all(np.asarray(law(dense)).imag <= 0))


def test_gold_film_table():
    f, (n, k) = _table(ROOT / "docs/tutorials/data/gold_mcpeak2015.csv", 0.4, 1.7)
    fit = _quiet_fit(f, n=n, k=k, drude=True)
    assert fit.eps_error <= 1e-2
    assert fit.k_error <= 1e-2
    # a metal's loss: Im eps = -2nk, so the n error is that of its absorption
    assert 0.1 < fit.n_error < 0.3
    assert isinstance(fit.law.terms[0], gfem.Drude)
    assert sum(isinstance(t, gfem.Lorentz) for t in fit.law.terms) <= 4
    assert fit.law.frequency_range == (f.min(), f.max())
    assert _passive(fit.law)


def test_aluminium_film_table():
    f, (n, k) = _table(DATA / "al_mcpeak2015.csv", 0.4, 1.7)
    fit = _quiet_fit(f, n=n, k=k, drude=True)
    assert fit.eps_error <= 1e-2
    assert _passive(fit.law)


def test_silicon_transparent_table_matches_the_library():
    materials = pytest.importorskip("gfem.materials")
    f, (n,) = _table(DATA / "si_li1980_293k.csv")
    fit = _quiet_fit(f, n=n)
    assert fit.k_error == 0.0
    assert all(t.damping == 0 for t in fit.law.terms)
    lo, hi = fit.law.frequency_range
    assert all(not lo <= t.resonance_frequency <= hi for t in fit.law.terms)
    library = materials.silicon().eps_r
    n_fit = np.sqrt(np.asarray(fit.law(f))).real
    n_library = np.sqrt(np.asarray(library(f))).real
    assert np.max(np.abs(n_fit - n_library)) <= 2e-3


def test_an_absorption_edge_warns_on_k():
    f, (n, k) = _table(DATA / "si_schinke2015.csv", 0.4, 1.0)
    with pytest.warns(
        gfem.PermittivityFitWarning, match="the law's k is off"
    ) as record:
        fit = gfem.fit_permittivity(f, n=n, k=k)
    assert len(record) == 1
    assert fit.eps_error <= 1e-2
    assert fit.k_error > 1
    assert "absorption edge" in str(record[0].message)


def test_rakic_gold_is_recovered_from_its_own_samples():
    materials = pytest.importorskip("gfem.materials")
    law = materials.gold().eps_r
    f = np.geomspace(0.1, 6, 200) * EV
    fit = gfem.fit_permittivity(f, eps=np.asarray(law(f)), drude=True, tolerance=1e-3)
    assert fit.eps_error <= 1e-3
    assert _passive(fit.law)
