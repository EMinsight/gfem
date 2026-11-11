"""Gradients on a refined model: the stage loop of the refinement recipe.

Refine between optimization stages on concrete values, carry the per-element
design with `MeshRefinement.transfer`, rebuild the morph on the refined mesh,
then differentiate `gfem.solve` on `refinement.model`. A refined mesh reaches
the solve by new code (built from arrays, imported nodes re-based, the
transfer gather on traced values), so its gradient is checked against finite
differences rather than assumed.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import capacitor

import gfem
from gfem.shape import boundary_morph

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
PROBLEM = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    """The capacitor with a per-element oxide design, refined twice."""
    model = capacitor(tmp_path_factory.mktemp("plate"))
    n = len(model.mesh.elements("oxide"))
    eps = np.random.default_rng(11).uniform(3.0, 5.0, n)
    designed = model.replace(
        materials={**model.materials, "oxide": gfem.MaterialField(eps_r=eps)}
    )
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            designed, PROBLEM, tolerance=1e-4, max_cycles=2, options=OPTIONS
        )
    assert len(ref.cycles) == 3
    assert len(ref.model.mesh.elements("oxide")) > n  # the transfer gathers
    return ref, jnp.asarray(eps)


def _capacitance(model) -> jnp.ndarray:
    return gfem.solve(model, PROBLEM, options=OPTIONS).capacitance[0, 0] * 1e12


def test_a_transferred_design_gradient_matches_fd(stage):
    ref, eps = stage

    def objective(p):
        field = gfem.MaterialField(eps_r=ref.transfer(p, "oxide"))
        return _capacitance(
            ref.model.replace(materials={**ref.model.materials, "oxide": field})
        )

    rng = np.random.default_rng(12)
    assert_gradient_matches(
        objective, eps, unit_directions(rng, eps.shape, 2), step=1e-2
    )


def test_a_morph_rebuilt_on_the_refined_mesh_gradient_matches_fd(stage):
    ref, _ = stage
    morph = boundary_morph(ref.model, "oxide__nitride", grows="oxide")

    def objective(t):
        return _capacitance(ref.model.replace(mesh=morph(t)))

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(13)
    assert_gradient_matches(objective, t, unit_directions(rng, t.shape, 2), step=2e-6)
