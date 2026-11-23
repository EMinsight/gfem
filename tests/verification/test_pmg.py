r"""The p-multigrid's iterations on models that need each of its mechanisms.

``pmg_gmres`` keeps its GMRES iterations flat because its direct level
takes what the vertex-star smoother cannot: the Whitney DOFs, the closure of
the perfectly matched layer's cells and of poor cells. Each case below
needs one of these, and bounds the iterations per column at the count
measured on one or two threads plus 2. The bounds sit far below the counts
measured on one thread with the mechanism removed: 230-232 iterations on
the strip guide with its layer smoothed, 40-42 on the SOI straight with its
thin cells smoothed, 96-110 on the straight and 59-60 at order 3 with no
direct level at all; with no reuse the sweep builds three times.
Every case also checks S against ``direct`` to $10^{-5}$ (measured
$2 \cdot 10^{-7}$ to $2 \cdot 10^{-6}$).

- The strip guide of `pml_cases` between side layers (59.5k DOF): 8
  iterations measured.
- The SOI straight of `soi_cases`, 1 cladding wavelength wide (54k DOF): 8;
  with a 1 nm cell layer along its outer y faces (63k DOF): 9; at order 3,
  0.5 wavelength wide (50k DOF): 10.
- That straight swept over $f_0 (1, 1.01, 1.02)$ (its first frequency is
  the fresh case): one build serves the three frequencies. A reused V-cycle
  solves to a tolerance 10 times tighter, so its columns are bounded by the
  fresh count scaled to that tolerance, plus 2 (measured 9-10).
"""

import math

import numpy as np
import pml_cases as pc
import soi_cases
from preconditioners import count_builds, count_iterations

import gfem

#: S against direct.
ATOL = 1e-5


def _solve(model, problem, monkeypatch, order=2):
    """S of ``pmg_gmres`` and of ``direct``, and the p-multigrid's iterations."""
    iterations = count_iterations(monkeypatch)
    s = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(order=order, linear_solver="pmg_gmres"),
    ).s.values
    direct = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(order=order, linear_solver="direct"),
    ).s.values
    np.testing.assert_allclose(np.asarray(s), np.asarray(direct), rtol=0, atol=ATOL)
    return list(iterations)


def _soi(tmp_path, scale, **kw):
    return soi_cases.model(soi_cases.straight(tmp_path / "soi.msh", scale, **kw))


SOI = gfem.Driven(frequency=soi_cases.F0, excite=None)


def test_a_strip_guide_between_layers_takes_the_layer_on_the_direct_level(
    tmp_path, monkeypatch
):
    path = pc.strip_guide(tmp_path / "strip.msh", 1.0, length=1.0, clad_h=0.15)
    iterations = _solve(pc.strip_model(path), gfem.Driven((pc.STRIP_F,)), monkeypatch)
    assert iterations
    assert max(iterations) <= 10


def test_an_soi_straight_with_a_nanometre_cell_layer_takes_it_on_the_direct_level(
    tmp_path, monkeypatch
):
    # a grid line 1 nm inside each outer y face: a layer of cells of aspect
    # 100 and more, whose smoothing alone stalls GMRES
    layer = soi_cases.half_width(1.0) - 0.001
    iterations = _solve(_soi(tmp_path, 1.0, lines=(layer,)), SOI, monkeypatch)
    assert iterations
    assert max(iterations) <= 11


def test_an_soi_straight_at_order_3(tmp_path, monkeypatch):
    iterations = _solve(_soi(tmp_path, 0.5), SOI, monkeypatch, order=3)
    assert iterations
    assert max(iterations) <= 12


def test_an_soi_straight_swept_at_one_percent_builds_once(tmp_path, monkeypatch):
    model = _soi(tmp_path, 1.0)
    builds = count_builds(monkeypatch, "pmg_gmres")
    f0 = soi_cases.F0
    problem = gfem.Driven(frequency=(f0, 1.01 * f0, 1.02 * f0), excite=None)
    iterations = _solve(model, problem, monkeypatch)
    assert len(builds) == 1
    # each frequency's columns in turn, the first frequency's on a fresh build
    columns, rest = divmod(len(iterations), 3)
    assert columns > 0
    assert rest == 0
    fresh = max(iterations[:columns])
    assert fresh <= 10
    # a reused V-cycle solves to a tolerance 10 times tighter: 7 digits for
    # the fresh 6, in 7/6 the iterations (measured 9-10 against 8)
    assert max(iterations[columns:]) <= math.ceil(fresh * 7 / 6) + 2
