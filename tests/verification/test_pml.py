r"""Perfectly matched layers against their closed-form reflection.

A PEC-backed layer of target reflection :math:`R_0` reflects
:math:`|R| = R_0^{n f \cos\theta / f_\text{ref}}` at the frequency
:math:`f`, the angle :math:`\theta` from its normal and in a background of
index :math:`n` (`gfem._pml.reflection`). The cases end a straight guide
with a layer and read :math:`|S_{11}|` at a wave port on the far end, which
is exact for the guide's mode, so :math:`|S_{11}| = |R|`:

- a TEM line (PEC y walls, PMC x walls): normal incidence;
- a TE10 guide: two plane waves at :math:`\cos\theta = \beta_{10}/k_0`.

Order 3 on z-layered structured meshes, :math:`h = \lambda/10`
(:math:`\lambda/14` for TE10), where the reflection floor is about
:math:`10^{-6}`; each closed form sits at least 100 times above its floor
and is asserted to 1 % (`pml_cases`).
"""

import math

import numpy as np
import pml_cases as pc
import pytest

import gfem
from gfem import _pml

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


def _s11(path, frequencies, pml, eps_r=1.0):
    mesh = gfem.Mesh.from_gmsh(path)
    material = gfem.Material(eps_r=eps_r)
    model = gfem.Model(
        mesh,
        materials={"air": material, "pml": material},
        boundaries=[gfem.PEC("pec")],
        ports=[gfem.WavePort("p1", "port")],
        pml=pml,
    )
    result = gfem.solve(model, gfem.Driven(tuple(frequencies)), options=OPTIONS)
    return np.abs(np.asarray(result.s.values)[:, 0, 0])


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return pc.tem_line(tmp_path_factory.mktemp("pml") / "line.msh")


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    return pc.te10_guide(tmp_path_factory.mktemp("pml") / "guide.msh")


def _close(measured, expected):
    assert abs(measured - expected) <= pc.REL * expected, (measured, expected)


def test_a_layer_behind_a_tem_line_reflects_its_target(line):
    (s11,) = _s11(line, [pc.F0], gfem.PML("pml", target_reflection=1e-2))
    _close(s11, 1e-2)


def test_the_lowest_solve_frequency_is_the_reference(line):
    s11 = _s11(line, [pc.F0, 1.5 * pc.F0], gfem.PML("pml", target_reflection=1e-3))
    _close(s11[0], 1e-3)
    # 10^-4.5 sits only 30 times above the floor: an absolute bound of 3 floors
    assert abs(s11[1] - _pml.reflection(1e-3, 1.5, 1.0, 1.0)) <= 3 * pc.FLOOR


def test_a_pinned_reference_frequency_sets_the_reflection(line):
    pml = gfem.PML("pml", reference_frequency=2 * pc.F0)
    (s11,) = _s11(line, [pc.F0], pml)
    _close(s11, 1e-3)


def test_a_denser_background_reflects_less(line):
    # the stretch is normalized by the vacuum wavenumber: R0^n in index n
    (s11,) = _s11(line, [pc.F0], gfem.PML("pml", target_reflection=1e-2), eps_r=2.25)
    _close(s11, 1e-3)


def _cos_te10(wavelength):
    return math.sqrt(1 - (wavelength / (2 * pc.GUIDE_SECTION[0])) ** 2)


def test_a_te10_guide_ended_by_a_layer(guide):
    (s11,) = _s11(guide, [pc.F0], gfem.PML("pml", target_reflection=1e-3))
    expected = _pml.reflection(1e-3, 1.0, 1.0, _cos_te10(1.0))
    assert expected == pytest.approx(5.807e-3, rel=1e-3)
    _close(s11, expected)


def test_a_guide_near_cutoff_reflects_more(guide):
    (s11,) = _s11(guide, [pc.F0 / 1.4], gfem.PML("pml"))
    expected = _pml.reflection(1e-6, 1.0, 1.0, _cos_te10(1.4))
    assert expected == pytest.approx(7.013e-3, rel=1e-3)
    _close(s11, expected)


def test_a_reference_frequency_at_beta_restores_the_target(guide):
    # f_ref where k0 = beta: the layer meets R0 = 1e-6 again, below the floor
    f = pc.F0 / 1.4
    (s11,) = _s11(guide, [f], gfem.PML("pml", reference_frequency=f * _cos_te10(1.4)))
    assert s11 <= pc.GUIDE_FLOOR
