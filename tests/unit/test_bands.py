r"""The reciprocal lattice and the wave vector path of a periodic cell.

`Model.reciprocal_lattice` is dual to the periodic translations,
$\mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}$. `gfem.wave_vector_path`
recognises the lattice from the translations, places its high-symmetry
points (the hexagonal K as the zone corner, whatever the basis angle) and
samples straight segments between them with exactly ``n_points`` wave
vectors, every high-symmetry point among them. Geometry only: nothing here
solves.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest

import gfem

LATTICES = ("line", "rect", "square", "cubic", "hex60", "hex120", "oblique75")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """One model per lattice case, built lazily."""
    built = {}
    builders = {
        "line": lambda d: pc.c3(pc.c3_mesh(d)),
        "rect": lambda d: pc.c2(pc.c2_mesh(d, cells=(2, 2, 1))),
        "square": lambda d: pc.c2(pc.c2_square_mesh(d, cells=(2, 2, 1))),
        "cubic": lambda d: pc.c4(pc.c4_mesh(d, cells=(2, 2, 2))),
        "hex60": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 60)),
        "hex120": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 120)),
        "oblique75": lambda d: pc.sheared_cell(pc.sheared_cell_mesh(d, 75)),
    }

    def get(name):
        if name not in built:
            built[name] = builders[name](tmp_path_factory.mktemp(name))
        return built[name]

    return get


@pytest.mark.parametrize("case", LATTICES)
def test_reciprocal_lattice_is_dual_to_the_translations(models, case):
    model = models(case)
    a = np.asarray(model.periodic_translations)
    b = np.asarray(model.reciprocal_lattice)
    assert b.shape == a.shape
    np.testing.assert_allclose(
        b @ a.T, 2 * math.pi * np.eye(len(a)), rtol=0, atol=1e-12
    )
    # inside the span of the translations
    basis, _ = np.linalg.qr(a.T)
    np.testing.assert_allclose(basis @ (basis.T @ b.T), b.T, rtol=0, atol=1e-9)


def test_reciprocal_lattice_is_empty_without_a_pair(tmp_path):
    mesh = pc.c2_mesh(tmp_path, cells=(2, 2, 1))
    model = gfem.Model(mesh, materials={"cell": gfem.Material()})
    assert model.reciprocal_lattice == ()
