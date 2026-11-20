"""The flagship: the dispersive shift of a 3D transmon and its pad-gap gradient.

A transmon in a PEC box cavity (lengths in mm): two void pads with PEC faces
on a silicon chip, joined by a 7.5 nH junction port; the box's TE101 mode is
the readout. Mesh ``docs/tutorials/data/transmon.msh`` (52 810 DOF at order
2). The pad gap (300 um) is changed by a boundary morph of the facing pad
edges, each moved by half the change, with the stub tips, the walls and the
junction port held.

``jax.grad`` of the diagonalized dispersive shift ``chi`` between the qubit
and the readout mode runs the mode-field adjoint through the
eigenmode solve, the junction participation and `gfem.quantum.dressed_modes`.
It is checked against a fourth-order difference of plain solves with a 5 um
step. The expected values were measured on the polished
forward at the default ``eigen_tolerance``.

Reference tier: six forwards of 20-26 s each and about 1.05 GiB.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem.shape import boundary_morph

pytestmark = pytest.mark.reference

MESH = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "transmon.msh"
L_J = 7.5e-9  # H
PROBLEM = gfem.Eigenmode(target=5.5e9, n_modes=2)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
STEP = 5e-6  # m, the finite-difference step in the gap
MHZ = 2 * np.pi * 1e6  # rad/s per MHz


@pytest.fixture(scope="module")
def device():
    """The transmon model and the pad gap change (m) -> moved mesh."""
    mesh = gfem.Mesh.from_gmsh(MESH, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"vacuum": gfem.Material(), "chip": gfem.Material(eps_r=11.7)},
        boundaries=[
            gfem.PEC(s) for s in ("walls", "pad", "pad_edge", "stub_wall", "stub_tip")
        ],
        ports=[gfem.LumpedPort("jj", "junction", "+x", inductance=L_J)],
    )
    morph = boundary_morph(model, "pad_edge", fixed=("stub_tip", "walls"))
    ones = jnp.ones(morph.n_params)

    def moved(dg):
        # a positive t moves the edge into the domain (grows the pad), so the
        # gap widens by dg when each facing edge moves by -dg / 2
        return model.replace(mesh=morph(-dg / 2 * ones))

    return model, moved


def _chi(model):
    """Diagonalized chi (rad/s) and the solve and dressed modes as aux."""
    result = gfem.solve(model, PROBLEM, options=OPTIONS)
    dressed = gfem.quantum.dressed_modes(
        result.omega, result.participation["jj"], inductance=L_J
    )
    # dispersive_shift is symmetric, so the mode order does not matter here
    return dressed.dispersive_shift[0, 1], (result, dressed)


@pytest.fixture(scope="module")
def at_300um(device):
    _, moved = device
    (chi, (result, dressed)), grad = jax.value_and_grad(
        lambda dg: _chi(moved(dg)), has_aux=True
    )(0.0)
    return chi, grad, result, dressed


def test_values_at_300um(at_300um):
    chi, _, result, dressed = at_300um
    omega = np.real(np.asarray(result.omega))
    order = np.argsort(omega)  # qubit first
    p = np.asarray(result.participation["jj"])[order]
    np.testing.assert_allclose(
        omega[order] / (2 * np.pi * 1e9), [5.642669513, 6.794556202], rtol=1e-7
    )
    np.testing.assert_allclose(p, [-0.9664951294, -0.008274175605], rtol=1e-7)
    np.testing.assert_allclose(float(chi) / MHZ, -2.4959685, rtol=1e-6)
    first = gfem.quantum.dressed_modes(
        result.omega, result.participation["jj"], inductance=L_J, method="first_order"
    )
    np.testing.assert_allclose(
        float(first.dispersive_shift[0, 1]) / MHZ, -3.5168664, rtol=1e-6
    )
    np.testing.assert_allclose(
        float(np.asarray(dressed.anharmonicity)[order[0]]) / MHZ,
        -185.33148,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(dressed.omega)[order] / (2 * np.pi * 1e9),
        [5.4644573, 6.7929680],
        rtol=1e-7,
    )
    np.testing.assert_allclose(float(dressed.overlap), 0.993827, atol=1e-5)


def test_chi_gradient_in_the_pad_gap(at_300um):
    _, grad, _, _ = at_300um
    # MHz per um; a frequency-only chain would give the wrong sign
    np.testing.assert_allclose(float(grad) / MHZ * 1e-6, -1.109326619e-2, rtol=1e-6)


def test_chi_gradient_matches_a_fourth_order_difference(device, at_300um):
    _, moved = device
    _, grad, _, _ = at_300um
    chi = {s: float(_chi(moved(s * STEP))[0]) for s in (-2, -1, 1, 2)}
    fd4 = (8 * (chi[1] - chi[-1]) - (chi[2] - chi[-2])) / (12 * STEP)
    # measured 7.4e-9
    np.testing.assert_allclose(float(grad), fd4, rtol=1e-6)
