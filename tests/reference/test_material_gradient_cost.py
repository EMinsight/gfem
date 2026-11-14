"""Cost of a material-field gradient on the driven timing case.

On ``driven/r8`` (the IHP coplanar line at gsim defaults, order 2) every
dielectric region carries a per-element `MaterialField`, and the objective
is a weighted sum of |S|^2 at one frequency with both ports excited. The
backward of `jax.vjp` (S cotangents: no solve, one element pass per field
pair) must take at most a tenth of the forward wall time, both warm (as in
every iteration of an optimization after the first).

Each timed evaluation is a new design, one optimizer step (1 %) from the
last: a BDDC solve starts from the previous design's solution (ADR 0126), so
timing the design just solved again measures a forward of almost no
iterations. The backward is the fastest of three runs of the same pullback.
"""

import importlib.util
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

THREADS = 4
FREQUENCY = 50e9
#: The backward wall per forward wall the topology milestone allows.
MAX_RATIO = 0.1


@pytest.fixture(scope="module")
def reference(reference_dir):
    return reference_dir / "driven"


def _driven_model(case: Path) -> gfem.Model:
    """The Palace case as gsim builds it (shared with test_driven.py)."""
    path = Path(__file__).with_name("test_driven.py")
    spec = importlib.util.spec_from_file_location("_reference_driven", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model(case)


def test_material_gradient_backward_is_a_tenth_of_the_forward(reference):
    model = _driven_model(reference / "r8")
    fields = {}
    for region, material in model.materials.items():
        n = len(model.mesh.elements(region))
        fields[region] = {
            "eps_r": jnp.full(n, material.eps_r),
            "tan_delta": jnp.full(n, material.tan_delta),
        }
    problem = gfem.Driven(frequency=FREQUENCY)
    options = gfem.SolverOptions(threads=THREADS, linear_solver="bddc_gmres")
    weights = jnp.array([[1.0, 0.5], [2.0, -1.0]])

    def objective(values):
        designed = model
        for region, props in values.items():
            designed = designed.with_material(region, gfem.MaterialField(**props))
        s = gfem.solve(designed, problem, options=options).s.values[0]
        return jnp.sum(weights * jnp.abs(s) ** 2)

    def step(k):  # the design k optimizer steps on: every value 1 % per step
        return jax.tree.map(lambda v: v * (1 + 0.01 * k), fields)

    # warm up: the discretization cache, and JAX's compilation of the backward
    value, pullback = jax.vjp(objective, step(0))
    jax.block_until_ready(pullback(jnp.ones_like(value)))
    start = time.perf_counter()
    jax.block_until_ready(objective(step(1)))
    forward = time.perf_counter() - start

    start = time.perf_counter()
    value, pullback = jax.vjp(objective, step(2))
    jax.block_until_ready(value)
    forward_kept = time.perf_counter() - start
    backward = math.inf
    for _ in range(3):
        start = time.perf_counter()
        (grad,) = pullback(jnp.ones_like(value))
        jax.block_until_ready(grad)
        backward = min(backward, time.perf_counter() - start)

    n_elements = sum(len(v["eps_r"]) for v in fields.values())
    print(  # noqa: T201
        f"r8 material gradient ({len(fields)} regions, {n_elements} elements): "
        f"forward {forward:.2f} s, forward with kept fields {forward_kept:.2f} s, "
        f"backward {backward:.3f} s = {backward / forward:.4f}x forward; "
        f"load {os.getloadavg()}"
    )
    assert all(np.isfinite(np.asarray(g["eps_r"])).all() for g in grad.values())
    assert any(np.abs(np.asarray(g["eps_r"])).max() > 0 for g in grad.values())
    assert backward <= MAX_RATIO * forward
