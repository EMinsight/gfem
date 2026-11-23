r"""The Gaussian beam in a discrete solve.

Its moment interpolant in a box of air leaves a discrete Maxwell residual
:math:`\lVert A\,\Pi w\rVert / \lVert k^2 M\,\Pi w\rVert` on the interior
DOFs that falls under refinement in :math:`h` and in :math:`p`: the beam is
an exact solution, so only the interpolation error remains (a paraxial
beam stalls near 16 % at :math:`w_0 = \lambda`). Its cross sections are
checked against a plane wave's in `test_free_space_mie`, their gradient in
`test_free_space_gradient`.
"""

import math

import numpy as np
import pml_cases as pc
import pytest

import gfem
from gfem._engine.driven import driven_operator
from gfem._engine.moments import moment_interpolant
from gfem._incident import values

pytestmark = pytest.mark.verification


def _residual(tmp_path, cells, order):
    path = pc.box(tmp_path / f"box{cells}.msh", (), cells=cells)
    air = gfem.Material()
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"air": air},
        boundaries=[gfem.PEC("outer")],
        background=air,
    )
    beam = gfem.GaussianBeam(
        "b",
        waist=0.6e-6,
        focus=(0.5e-6, 0.5e-6, 0.5e-6),
        theta=0.7,
        phi=0.3,
        polarization="TM",
    )
    interpolant = moment_interpolant(model, order)
    w = interpolant(values(beam, model.background, interpolant.points, pc.F0))
    op = driven_operator(model, order)
    omega = 2 * math.pi * pc.F0
    # the PEC walls fix the boundary DOFs: the free ones are the interior's
    free = np.array(list(op.space.FreeDofs()), bool)
    total = sum(
        complex(p.theta(omega)) * op.apply(k, w) for k, p in enumerate(op.pieces)
    )
    (mass,) = (k for k, p in enumerate(op.pieces) if p.name == "mass")
    reference = complex(op.pieces[mass].theta(omega)) * op.apply(mass, w)
    return np.linalg.norm(total[free]) / np.linalg.norm(reference[free])


def test_the_discrete_residual_falls_with_h_and_p(tmp_path):
    coarse = _residual(tmp_path, 4, 2)
    fine = _residual(tmp_path, 8, 2)
    higher = _residual(tmp_path, 4, 3)
    # measured 2.6e-2, 8.4e-3 at h / 2 (3.1x), 1.6e-3 at p3 (16x)
    assert fine < coarse / 2.5
    assert higher < coarse / 8
