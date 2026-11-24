"""The multipole entries on ranks: one process's values and gradient.

The dielectric cube of `free_space_cases.scatterer_box` at 8 cells per um (a
two-cell perfectly matched layer, a shell of air between it and the cube), as
the portless model of `multipole_cases.portless_scatterer_model`, under a
plane wave on `RANKS` ranks against pytest's one process:

- `gfem.multipoles` (the coefficients and the partial cross sections) and
  `gfem.current_moments` are collective: every rank gets the one-process
  values to `GATE`;
- the gradient of a magnetic dipole cross section by the cube's per-element
  permittivity is the one process's to `GRADIENT_GATE`.
"""

import free_space_cases as fs
import jax
import jax.numpy as jnp
import multipole_cases as mc
import numpy as np
import pytest
from ranks import on_ranks

import gfem
from gfem import _petsc

#: NGSolve's rank 0 holds no element: two more ranks share a partition face.
RANKS = 3
#: Relative to the largest one-process value: direct solves of one system
#: split over ranks. Measured worst 2.9e-13; kept well above round-off because
#: the reduction order differs between rank counts and machines.
GATE = 1e-10
#: Relative to the largest one-process entry of the gradient. Measured 2.7e-13.
GRADIENT_GATE = 1e-9
F = fs.F0 / 2
WAVE = gfem.PlaneWave("w", theta=0.4, phi=0.3, polarization="TM")
ORIGIN = (0.5e-6, 0.5e-6, 0.5e-6)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _model(path, *, field=False):
    return mc.portless_scatterer_model(path, field=field)


def _solved(model):
    problem = gfem.Driven(F, sources=WAVE, keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


def _values(path):
    """Every value of the two entries, on this process."""
    model = _model(path)
    result = _solved(model)
    mp = gfem.multipoles(result, "w", model=model, origin=ORIGIN)
    cm = gfem.current_moments(result, "w", model=model, origin=ORIGIN)
    return {
        "alpha": np.asarray(mp.alpha),
        "beta": np.asarray(mp.beta),
        "electric dipole": np.asarray(mp.electric_dipole),
        "magnetic dipole": np.asarray(mp.magnetic_dipole),
        "scattering": np.asarray(mp.scattering),
        "electric scattering": np.asarray(mp.electric_scattering),
        "magnetic scattering": np.asarray(mp.magnetic_scattering),
        "long-wavelength dipole": np.asarray(cm.long_wavelength_dipole),
        "toroidal dipole": np.asarray(cm.toroidal_dipole),
    }


def _gradient(path):
    model = _model(path, field=True)
    eps = jnp.full(len(model.mesh.elements("scat")), mc.CUBE_EPS)

    def objective(e):
        moved = model.with_material("scat", gfem.MaterialField(eps_r=e))
        result = _solved(moved)
        m = gfem.multipoles(result, "w", model=moved, origin=ORIGIN)
        return 1e12 * m.magnetic_scattering[0, 0]

    value, grad = jax.value_and_grad(objective)(eps)
    return np.asarray(value), np.asarray(grad)


def _on_ranks_values(path):
    return _values(path)


def _on_ranks_gradient(path):
    return _gradient(path)


@pytest.fixture
def fine_box(tmp_path):
    """The scatterer box at 8 cells per um, a two-cell layer."""
    return str(fs.scatterer_box(tmp_path / "fine.msh", cells=8, thickness=0.25))


def _relative(a, b):
    return np.abs(np.asarray(a) - b).max() / np.abs(b).max()


@needs_stack
def test_m1_the_multipoles_and_current_moments_across_ranks_are_one_process_s(
    fine_box,
):
    got, *_ = on_ranks(_on_ranks_values, fine_box, ranks=RANKS, limit=900)
    one = _values(fine_box)
    for name, value in one.items():
        assert np.abs(value).max() > 0, name
        error = _relative(got[name], value)
        print(f"ranks={RANKS} {name} error={error:.3g}")  # noqa: T201
        assert error < GATE, f"{name} off by {error:.3g} relative"


@needs_stack
def test_g6_the_multipole_gradient_across_ranks_is_one_process_s(fine_box):
    (value, grad), *_ = on_ranks(_on_ranks_gradient, fine_box, ranks=RANKS, limit=1500)
    value_one, grad_one = _gradient(fine_box)
    assert np.abs(grad_one).max() > 0
    assert _relative(value, value_one) < GATE
    error = _relative(grad, grad_one)
    print(f"ranks={RANKS} multipole gradient error={error:.3g}")  # noqa: T201
    assert error < GRADIENT_GATE, f"the gradient off by {error:.3g} relative"
