"""`gfem.release_memory`: drop what gfem keeps between solves, results unchanged.

After it no solve entry, pooled solver handle or cached discretization is
left. A gradient whose entry was released is recomputed from the design, and
the same solve again gives the same S.
"""

import pathlib
import subprocess
import sys

import jax
import numpy as np
import pytest
from gradient_cases import layered_line, line_design, s_functional, with_fields

import gfem
from gfem._engine import cache

PROBLEM = gfem.Driven(frequency=24e9)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
DOCS = pathlib.Path(__file__).parents[2] / "docs"


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


@pytest.fixture
def empty_caches():
    gfem.release_memory()
    yield
    gfem.release_memory()


def _kept() -> tuple[int, int, int, int]:
    entries = cache.ENTRIES
    return len(entries), entries.pooled, entries.nbytes, len(cache._DISCRETIZATIONS)


@pytest.mark.usefixtures("empty_caches")
def test_release_memory_frees_everything_kept_and_changes_no_result(line):
    values, _ = line_design(line, seed=3)

    def s_of(eps):
        model = with_fields(line, {**values, "eps_r": eps})
        return gfem.solve(model, PROBLEM, options=OPTIONS).s.values

    def objective(eps):
        return s_functional(s_of(eps))

    eps = values["eps_r"]
    _, pullback = jax.vjp(objective, eps)
    (gradient,) = pullback(1.0)
    s = np.asarray(s_of(eps))
    entries, pooled, nbytes, discretizations = _kept()
    assert entries > 0 or pooled > 0
    assert nbytes > 0
    assert discretizations > 0

    assert gfem.release_memory() is None
    assert _kept() == (0, 0, 0, 0)

    # the entry is gone: the gradient is recomputed from the design
    (again,) = pullback(1.0)
    error = np.linalg.norm(again - gradient) / np.linalg.norm(gradient)
    assert error <= 1e-10
    assert np.abs(np.asarray(s_of(eps)) - s).max() <= 1e-12 * np.abs(s).max()


def test_release_memory_twice_or_before_any_solve_does_nothing():
    gfem.release_memory()
    gfem.release_memory()
    assert _kept() == (0, 0, 0, 0)


def test_importing_gfem_and_releasing_nothing_loads_no_engine():
    # a fresh process: `import gfem` must stay light, and so does a release
    # with nothing kept
    code = (
        "import sys, gfem\n"
        "assert 'ngsolve' not in sys.modules\n"
        "gfem.release_memory()\n"
        "print('ngsolve' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_release_memory_is_public_and_in_the_recipes():
    assert "release_memory" in gfem.__all__
    for recipe in ("s-parameter-gradient.md", "capacitance-gradient.md"):
        text = (DOCS / "recipes" / recipe).read_text(encoding="utf-8")
        assert "gfem.release_memory" in text
