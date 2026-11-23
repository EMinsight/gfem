r"""Material gradients under a plane wave, against finite differences.

The load of an incident column, :math:`A_\mathrm{bg}(\chi\Pi w)`, reads the
background's material, never the scatterer's, so the existing material
kernel on the pair :math:`(-\lambda, u_\mathrm{total})` is the whole
gradient. A dielectric cube (a `gfem.MaterialField`) in an air box wrapped in
a perfectly matched layer, with a lumped port on its +x face, at order 2; the
objective reads a plane wave's port wave, its total field at a point, and
its scattered part, against fourth-order differences to :math:`10^{-6}`.
"""

import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
from finite_differences import assert_gradient_matches, unit_directions

import gfem

pytestmark = pytest.mark.verification

AIR = gfem.Material()
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0 / 2
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization=(1.0, 0.5j))
PROBE = np.array([[0.85, 0.4, 0.6]]) * 1e-6


def _volume(c):
    if ((c > 0.25) & (c < 0.75)).all():
        return "scat"
    return "air" if ((c > 0) & (c < 1)).all() else "pml"


def _feed(c):
    on = abs(c[0] - 0.75) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75
    return "feed" if on else ""


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = pc.box(
        tmp_path_factory.mktemp("incident") / "box.msh",
        ("-x", "+x", "-y", "+y", "-z", "+z"),
        cells=4,
        volume=_volume,
        surface=_feed,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("scat"))
    return gfem.Model(
        mesh,
        materials={
            "air": AIR,
            "pml": AIR,
            "scat": gfem.MaterialField(eps_r=np.full(n, 3.0)),
        },
        ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
        pml=gfem.PML("pml"),
        background=AIR,
    )


def _objective(model):
    problem = gfem.Driven(F, excite=(), sources=WAVE, keep_fields=True)

    def objective(eps):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(moved, problem, options=OPTIONS)
        wave = jnp.abs(result.port_wave("p1", "w")[0]) ** 2
        total = jnp.sum(jnp.abs(result.e_field(F).at(PROBE)) ** 2)
        scattered = result.e_field(F, part="scattered")
        return wave + total + 1e-2 * jnp.sum(jnp.abs(scattered.coefficients) ** 2)

    return objective


def test_a_plane_wave_objective_by_the_scatterer_permittivity(model):
    eps = jnp.asarray(model.materials["scat"].eps_r)
    directions = unit_directions(np.random.default_rng(3), eps.shape, 2)
    assert_gradient_matches(_objective(model), eps, directions, step=1e-2)
