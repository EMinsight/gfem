"""The metal-bend tutorial's 2D model against a fine mesh and against the slab.

The case is `metal_bend_reference`'s: the tutorial's bend, its design square
an index blend of the cladding and a gold-like Drude metal, driven from p1 at
1.50, 1.55 and 1.60 um.

- The 45 degree gold mirror on the tutorial mesh (``bend2d.msh``, 0.1 um
  cells) at element order 3 gives the S of the same 2D model on 0.0125 um
  cells at order 2, computed once and frozen in the reference data as
  ``metal-bend/fine.json`` (``python tools/metal_bend_reference.py``). The
  pixels are read at the same quadrature points on both meshes, so the gap
  is the field's discretization error.
- The band objective's gradient with respect to the pixels, through the
  tutorial's chain (mirror about y = x, Helmholtz filter, projection at
  beta = 4) at the noisy mirror of the metal-bend verification test, points
  the same way on the 2D model and on the slab one cell thick between PEC
  plates. The two read the pixels differently (several points per triangle,
  one per tetrahedron), so their gradients are close, not equal.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import metal_bend_reference as mb
import numpy as np
import pytest

from gfem import design

#: |dS| of the tutorial mesh at order 3 against the fine value, each entry.
FINE_TOLERANCE = 3e-3
#: Least cosine of the angle between the 2D and the slab pixel gradients.
MIN_COSINE = 0.999
BETA = 4.0


@pytest.fixture(scope="module")
def fine(reference_dir):
    path = reference_dir / mb.FINE_FILE
    if not path.exists():
        pytest.skip(f"{path} missing (run tools/metal_bend_reference.py)")
    return json.loads(path.read_text())


def _complex(pairs):
    return np.array([complex(*p) for p in pairs])


def test_mirror_on_the_tutorial_mesh_matches_the_fine_mesh(fine):
    np.testing.assert_allclose(fine["frequencies"], mb.FREQUENCIES, rtol=1e-12)
    model, grid = mb.case(mb.TUTORIAL_MESH)
    s21, s11 = mb.band_s(model, grid, mb.mirror(grid), order=3)
    np.testing.assert_allclose(s21, _complex(fine["s21"]), rtol=0, atol=FINE_TOLERANCE)
    np.testing.assert_allclose(s11, _complex(fine["s11"]), rtol=0, atol=FINE_TOLERANCE)


def _pixel_gradient(model, grid, problem):
    blur = design.helmholtz_filter(grid, radius=0.3 * mb.um)

    def missed(rho):
        density = design.project(blur(0.5 * (rho + rho.T)), BETA)
        s21 = mb.solve(model, grid, density, problem).s["p2", "p1"]
        return jnp.mean(1 - jnp.abs(s21) ** 2)

    return np.asarray(jax.grad(missed)(jnp.asarray(mb.noisy_mirror(grid))))


def test_2d_and_slab_pixel_gradients_point_the_same_way():
    model, grid = mb.case(mb.TUTORIAL_MESH)
    slab, slab_grid = mb.case(mb.SLAB_MESH, slab=True)
    np.testing.assert_allclose(slab_grid.x, grid.x)
    np.testing.assert_allclose(slab_grid.y, grid.y)
    g = _pixel_gradient(model, grid, mb.problem())
    g_slab = _pixel_gradient(slab, slab_grid, mb.problem(slab=True))
    cosine = float(g.ravel() @ g_slab.ravel()) / (
        np.linalg.norm(g) * np.linalg.norm(g_slab)
    )
    assert cosine >= MIN_COSINE, f"cosine {cosine:.5f}"


def test_the_fine_mesh_is_the_tutorial_section_refined(tmp_path):
    path = mb.write_fine_mesh(tmp_path, h=0.1)
    assert Path(path).read_bytes() == mb.TUTORIAL_MESH.read_bytes()
