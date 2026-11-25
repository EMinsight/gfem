"""The effective-medium retrieval on FEM slabs: isotropic, magnetic, gradient.

F1 with ``eps_sub = 1`` (air / eps 4 slab / air; `periodic_cases.f1`), Floquet
ports offset by ``F1_D`` onto the slab faces so the reference planes are the
slab faces, p = 3 on `F1_FINE_CELLS`, direct solver. The retrieval
`gfem.diffraction.effective_medium` must return the material that was meshed:
eps 4 and mu 1 for the isotropic slab; eps = (2, 3), mu = (1.2, 1.5) along
(x, y) for the anisotropic magnetic one, which catches an axis mix-up or an
eps <-> mu swap; and its derivative with respect to the slab permittivity must
match a finite difference of the whole solve.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
from finite_differences import assert_gradient_matches

import gfem
from gfem import diffraction as dif

TOP, BOT = "z_max", "z_min"
F0 = pc.F1_FREQUENCY
D = pc.F1_D * pc.UM
K0D = 2 * math.pi * F0 / pc.C0 * pc.F1_H * pc.UM
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


def _ports():
    return [dataclasses.replace(p, offset=D) for p in pc.f1_ports()]


def airy_on_the_slab(pol):
    """Airy (S11, S21) of the slab between planes on its faces (offsets D)."""
    layers = [(1.0, 0.0), (pc.F1_EPS_SLAB, pc.F1_H * pc.UM), (1.0, 0.0)]
    return pc.airy(F0, 0.0, layers, pol)


def _model(mesh, slab):
    return pc.f1(mesh, ports=_ports(), eps_sub=1.0).with_material("slab", slab)


def _solve(model):
    return gfem.solve(model, gfem.Driven([F0]), options=OPTIONS)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return pc.f1_mesh(tmp_path_factory.mktemp("em"), pc.F1_FINE_CELLS)


def slab_s(eps, mu, k0d):
    """Closed-form (S11, S21) of an isotropic slab in vacuum, e^{+i w t}.

    Wave number n = sqrt(eps mu) with Im n <= 0, impedance Z/Z0 = sqrt(mu/eps),
    interface reflection G = (z - 1)/(z + 1), T = e^{-i n k0 d}:
    S11 = G (1 - T^2) / (1 - G^2 T^2), S21 = T (1 - G^2) / (1 - G^2 T^2).
    """
    n = np.sqrt(complex(eps * mu))
    if n.imag > 0:
        n = -n
    z = np.sqrt(complex(mu / eps))
    z = z if z.real >= 0 else -z
    g = (z - 1) / (z + 1)
    t = np.exp(-1j * n * k0d)
    den = 1 - g**2 * t**2
    return g * (1 - t**2) / den, t * (1 - g**2) / den


def _l1_rows(s11, s21, k0d, branch):
    """|J| of (Re eps, Im eps, Re mu, Im mu) in Re, Im of (S11, S21), shape (4, 4)."""
    from gfem._diffraction.effective import retrieve

    x = jnp.asarray([s11.real, s11.imag, s21.real, s21.imag])

    def out(x):
        e, m, _, _ = retrieve(
            jnp.asarray([x[0] + 1j * x[1]]),
            jnp.asarray([x[2] + 1j * x[3]]),
            jnp.asarray([k0d]),
            1.0,
            1.0,
            branch,
        )
        return jnp.stack([e[0].real, e[0].imag, m[0].real, m[0].imag])

    return np.abs(np.asarray(jax.jacobian(out)(x)))


def _jacobian_tolerance(reference, measured, k0d, branch, bound):
    """The bound of the error of (eps, mu), as complex moduli, from |S - reference|.

    ``bound`` is the documented |S - reference| per entry, so each real and
    imaginary part of S11 and S21 is off by at most that. The retrieved value
    moves by the Jacobian of `retrieve` integrated along the segment from the
    ``reference`` S to the ``measured`` one; the l1 row sum of |J| times
    ``bound`` bounds it with the largest |J| on the segment. J is smooth along
    the segment (it moves by the second-order remainder, relative size
    ``bound`` times the curvature of `retrieve`, far below one), so the larger
    of its two endpoint values stands for it.
    """
    rows = np.maximum(
        _l1_rows(*reference, k0d, branch), _l1_rows(*measured, k0d, branch)
    ).sum(axis=1)
    err = bound * rows
    return float(np.hypot(err[0], err[1])), float(np.hypot(err[2], err[3]))


def test_isotropic_slab_reads_eps_four_and_mu_one(mesh):
    model = _model(mesh, gfem.Material(eps_r=pc.F1_EPS_SLAB))
    result = _solve(model)
    # n k0 d = 3.77 rad > pi: the principal phase has wrapped, second sheet
    for pol in ("TE", "TM"):
        r, t = airy_on_the_slab(pol)
        s = result.s
        assert abs(complex(s[f"R_{pol}", f"R_{pol}"][0]) - r) <= pc.F1_S_TOL
        assert abs(complex(s[f"T_{pol}", f"R_{pol}"][0]) - t) <= pc.F1_S_TOL
    em = dif.effective_medium(result, TOP, BOT, model=model, branch=1)
    measured = (
        complex(result.s["R_TM", "R_TM"][0]),
        complex(result.s["T_TM", "R_TM"][0]),
    )
    tol_e, tol_m = _jacobian_tolerance(
        airy_on_the_slab("TM"), measured, K0D, 1, pc.F1_S_TOL
    )
    assert np.all(np.abs(np.asarray(em.permittivity[0]) - 4.0) <= tol_e)
    assert np.all(np.abs(np.asarray(em.permeability[0]) - 1.0) <= tol_m)


EPS_ANISO = (2.0, 3.0, 2.0)
MU_ANISO = (1.2, 1.5, 1.0)
#: Lattice axes: column 0 has E along e1 = +x, so eps_11 = eps_x and mu_22 = mu_y
#: set its slab (n^2 = eps_x mu_y, Z = sqrt(mu_y / eps_x)); column 1 pairs
#: eps_22 = eps_y with mu_11 = mu_x.
COLUMNS = {0: (EPS_ANISO[0], MU_ANISO[1]), 1: (EPS_ANISO[1], MU_ANISO[0])}


def _branch(n):
    """Phase sheet of n k0 d at one frequency: the principal phase wraps at pi."""
    return math.floor((n.real * K0D + math.pi) / (2 * math.pi))


def test_anisotropic_magnetic_slab_reads_each_axis_and_each_field(mesh):
    model = _model(mesh, gfem.Material(eps_r=EPS_ANISO, mu_r=MU_ANISO))
    result = _solve(model)
    refl = np.asarray(dif.jones(result, TOP, TOP, model=model).values)[0]
    trans = np.asarray(dif.jones(result, BOT, TOP, model=model).values)[0]
    closed = {i: slab_s(*COLUMNS[i], K0D) for i in (0, 1)}
    branch = tuple(_branch(np.sqrt(complex(e * m))) for e, m in COLUMNS.values())
    # F1_S_TOL was measured on eps 4; the magnetic slab (n 1.73 and 1.90, other
    # impedances) is asserted against the same bound first (measured here:
    # 1e-7 to 3e-7 per entry, so the bound is not loosened)
    for i in (0, 1):
        assert abs(refl[i, i] - closed[i][0]) <= pc.F1_S_TOL
        assert abs(trans[i, i] - closed[i][1]) <= pc.F1_S_TOL
    em = dif.effective_medium(result, TOP, BOT, model=model, branch=branch)
    eps = np.asarray(em.permittivity[0])
    mu = np.asarray(em.permeability[0])  # (mu_11, mu_22)
    for i, mu_index in ((0, 1), (1, 0)):  # column i reads eps_ii and mu_jj, j != i
        measured = (complex(refl[i, i]), complex(trans[i, i]))
        tol_e, tol_m = _jacobian_tolerance(
            closed[i], measured, K0D, branch[i], pc.F1_S_TOL
        )
        assert abs(eps[i] - COLUMNS[i][0]) <= tol_e
        assert abs(mu[mu_index] - COLUMNS[i][1]) <= tol_m


#: The finite-difference step in the slab permittivity. The retrieved eps_11
#: equals the slab permittivity up to the FEM error of S, so the objective is
#: a straight line plus a smooth discretization term; its fifth derivative is
#: set by the phase n k0 d (2 rad per unit of sqrt(eps), so about 0.5 rad per
#: unit of eps), and the fourth-order truncation h^4 f^(5) / 30 is below
#: 1e-8 at h = 0.05 (and 1e-6 would already be the gate). The round-off of a
#: direct solve, about 1e-14 relative, amplified by the retrieval's
#: conditioning (order 10 at this slab) and divided by h, is below 1e-11.
#: Measured: gradient 0.9999988 (the FEM error of S), difference from the
#: fourth-order quotient 1.7e-12, 1.4e-12, 1.0e-12 at h = 0.02, 0.05, 0.1.
STEP = 0.05


def test_gradient_of_the_retrieved_permittivity_matches_finite_differences(mesh):
    n = len(mesh.elements("slab"))

    def objective(eps):  # one scalar permittivity on every element of the slab
        model = _model(mesh, gfem.MaterialField(eps_r=eps * jnp.ones(n)))
        result = _solve(model)
        em = dif.effective_medium(result, TOP, BOT, model=model, branch=1)
        return jnp.real(em.permittivity[0, 0])

    eps = jnp.asarray(pc.F1_EPS_SLAB)
    grad = jax.grad(objective)(eps)
    # d eps_11 / d eps_slab is 1 up to the derivative of the FEM error of S.
    # That error is bounded by F1_S_TOL and analytic in the slab permittivity
    # on the disc of radius R about 4, R the distance to the nearest
    # half-wave resonance of the slab (n k0 d = pi at eps = (pi / K0D)^2, where
    # S11 -> 0 and the retrieval is singular); Cauchy's estimate bounds its
    # derivative by F1_S_TOL / R, and the gradient error by the l1 row of the
    # Jacobian of Re eps_11 with respect to S times that.
    radius = abs(pc.F1_EPS_SLAB - (math.pi / K0D) ** 2)
    ref = airy_on_the_slab("TM")
    row = _l1_rows(*ref, K0D, 1)[0].sum()
    assert abs(float(grad) - 1.0) < row * pc.F1_S_TOL / radius
    assert_gradient_matches(
        objective, eps, [np.asarray(1.0)], step=STEP, rel=1e-6, grad=grad
    )
