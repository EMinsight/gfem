"""A permittivity law on the GPU: run with ``pytest -m cuda``.

The GPU's only path is the summed matrix, which the law's
susceptibility piece joins like any other piece: driven S of the strip line with
the `poles` law on its core (`gradient_cases.dispersive_line`, Re eps < 0 at its
first frequency) on ``device="cuda"`` equals the CPU direct solver's.
"""

import gradient_cases as cases
import numpy as np

import gfem


def test_a_law_models_s_on_the_gpu_matches_the_direct_solver(tmp_path):
    model = cases.dispersive_line(cases.dispersive_mesh(tmp_path), cases.LAWS["poles"])
    problem = gfem.Driven(frequency=list(cases.LAW_FREQUENCIES["poles"]))
    cpu = gfem.SolverOptions(order=2, linear_solver="direct")
    gpu = gfem.SolverOptions(order=2, linear_solver="direct", device="cuda")
    want = np.asarray(gfem.solve(model, problem, options=cpu).s.values)
    got = np.asarray(gfem.solve(model, problem, options=gpu).s.values)
    error = np.abs(got - want).max() / np.abs(want).max()
    assert error < 1e-10, f"S off by {error:.3g} relative"
