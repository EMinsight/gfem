"""The mode weights an optical shift reads on a wave port's section.

A silicon core in oxide on a small cross-section slab. The scalar weight
per section triangle is the thermo-optic one,
:math:`w_t = \\tfrac{c\\varepsilon_0}{4} 2 n_{0,t} \\int_t |E_0|^2 \\, dA`;
the tensor weight
:math:`W_t = \\tfrac{c\\varepsilon_0}{4} \\mathrm{Re} \\int_t E_0 E_0^H \\, dA`
is real symmetric in the global frame with trace
:math:`w_t / (2 n_{0,t})` (both read from one mode: two solves agree only to
the eigensolver's threaded scatter), and a lossless mode's transverse and
normal components are in quadrature, so its transverse-normal entries vanish.
"""

import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem
from gfem._engine.thermo_optic import section_tensor_weights, section_weights
from gfem._optical_shift import _mode_weights, _section, _wave_port
from gfem._sweep.port_mode import PortModeRule
from gfem.options import SolverOptions

FREQUENCY = scipy.constants.c / 1.55e-6
N_SI, N_OX = 3.4777, 1.444
RECTS = [
    (-1.75, -1.0, 1.75, 1.0, "clad", None),
    (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
]


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    path = tmp_path_factory.mktemp("optical_shift") / "section.msh"
    mesh = tc.section(path, RECTS, 0.3, [])
    model = gfem.Model(
        mesh,
        materials={
            "clad": gfem.Material(eps_r=N_OX**2),
            "core": gfem.Material(eps_r=N_SI**2),
        },
        ports=[gfem.WavePort("p", "front")],
    )
    faces, tets = _section(model, _wave_port(model, "p"))
    _, _, tensor, _ = _mode_weights(model, "p", FREQUENCY, None, faces, tensor=True)
    corners = np.asarray(mesh.nodes)[np.asarray(mesh.triangles)[faces[0]]]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    return model, faces, tets, np.asarray(tensor), normal / np.linalg.norm(normal)


def test_the_tensor_weight_is_real_symmetric_per_triangle(section):
    _, faces, _, tensor, _ = section
    assert tensor.shape == (len(faces), 3, 3)
    assert tensor.dtype == np.float64
    np.testing.assert_array_equal(tensor, np.swapaxes(tensor, 1, 2))


def test_the_trace_of_the_tensor_weight_is_the_scalar_weight(section):
    model, faces, tets, _, _ = section
    rule = PortModeRule(model, gfem.PortMode("p", FREQUENCY), SolverOptions())
    sweep = rule.forward(model, keep=False)[0]["sweep"]
    coefficients = sweep.basis @ sweep.coords[0]
    scalar = section_weights(sweep.space, coefficients)
    tensor = section_tensor_weights(sweep.space, coefficients)
    behind = dict(zip(faces, tets, strict=True))
    core = set(model.mesh.elements("core"))
    n0 = np.array(
        [N_SI if behind[f] in core else N_OX for f in sweep.space.section.faces]
    )
    trace = np.trace(tensor, axis1=1, axis2=2)
    np.testing.assert_allclose(2 * n0 * trace, scalar, rtol=1e-12, atol=0)


def test_the_transverse_normal_entries_vanish(section):
    _, _, _, tensor, normal = section
    across = np.eye(3) - np.outer(normal, normal)
    tn = np.einsum("ij,tjk,k->ti", across, tensor, normal)
    tt = np.einsum("ij,tjk,kl->til", across, tensor, across)
    assert np.max(np.abs(tn)) < 1e-12 * np.max(np.abs(tt))
