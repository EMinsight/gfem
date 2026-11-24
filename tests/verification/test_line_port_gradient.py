r"""Line ports of 2D models differentiate material values at the port line.

A 2D layered guide (``tests/line_port_cases.py``) with a core of
:math:`\varepsilon_r = 4` at ``port1`` and 3 at ``port2``, in Ez and Hz,
lossless and lossy. Every gradient is checked against fourth-order central
differences of gfem's own solves at 1e-6:

- the port mode's :math:`\gamma` and normalized profile at two frequencies,
- a driven S functional for a design at the excited port and at the
  measuring one,
- a per-element field, and an absorbing end of the line.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from line_port_cases import FREQUENCY, guide, write_guide

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
FREQUENCIES = [FREQUENCY, 1.025 * FREQUENCY]
E1, E2 = 4.0, 3.0
STEP = 1e-2


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = write_guide(tmp_path_factory.mktemp("guide") / "guide.msh")
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def functional(s):
    """A real functional of every S entry, S (2, 2)."""
    weights = jnp.array([[1.0, 0.5], [2.0, -1.0]])
    return jnp.sum(weights * jnp.abs(s) ** 2) + jnp.real((0.3 - 0.7j) * s[1, 0])


def check(objective, p, seed):
    """FD along the (scalar) design and along its imaginary direction is not
    needed: the design is one real number."""
    del seed
    assert_gradient_matches(objective, p, [np.array(1.0)], step=STEP)


CASES = [
    ("Ez", 0.0),
    ("Ez", 0.05),
    ("Hz", 0.0),
    ("Hz", 0.05),
]
IDS = ["Ez", "Ez-lossy", "Hz", "Hz-lossy"]
PROFILE = np.linspace(0.2e-3, 3.8e-3, 7)


@pytest.mark.parametrize(("polarization", "tan_delta"), CASES, ids=IDS)
def test_a_line_port_modes_gamma_and_profile_differentiate(
    mesh, polarization, tan_delta
):
    points = np.stack([np.zeros_like(PROFILE), PROFILE], axis=1)
    rng = np.random.default_rng(5)
    weights = jnp.asarray(rng.standard_normal((2, len(PROFILE))))

    def objective(p):
        model = guide(mesh, p, E2, tan_delta=tan_delta)
        mode = gfem.solve(
            model,
            gfem.PortMode("p1", FREQUENCIES, polarization=polarization),
            options=OPTIONS,
        )
        field = mode.h_field if polarization == "Hz" else mode.e_field
        u = field.at(points)
        return (
            jnp.real(jnp.sum(mode.gamma * (1.0 + 0.5j)))
            + jnp.real(jnp.sum(weights * u))
            + jnp.imag(jnp.sum(weights * u**2))
        )

    check(objective, jnp.asarray(E1), 1)


@pytest.mark.parametrize("region", ["core1", "core2"])
@pytest.mark.parametrize(("polarization", "tan_delta"), CASES, ids=IDS)
def test_a_driven_s_functional_differentiates_at_either_port(
    mesh, polarization, tan_delta, region
):
    def objective(p):
        e1, e2 = (p, E2) if region == "core1" else (E1, p)
        model = guide(mesh, e1, e2, tan_delta=tan_delta)
        problem = gfem.Driven(FREQUENCY, polarization=polarization)
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    check(objective, jnp.asarray(E1 if region == "core1" else E2), 2)


def test_a_per_element_field_at_the_line_port_differentiates(mesh):
    n = len(mesh.elements("core1"))
    rng = np.random.default_rng(3)
    p = jnp.asarray(E1 * rng.uniform(0.9, 1.1, n))

    def objective(p):
        model = guide(mesh, p, E2, tan_delta=0.02)
        problem = gfem.Driven(FREQUENCY, polarization="Hz")
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    assert_gradient_matches(objective, p, unit_directions(rng, p.shape, 2), step=STEP)


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_an_absorbing_end_of_the_line_differentiates(mesh, polarization):
    # the design is the cladding, which holds both ends of each port line:
    # the admittance of an absorbing end reads its permittivity
    n = len(mesh.elements("clad"))
    rng = np.random.default_rng(4)
    p = jnp.asarray(1.5 * rng.uniform(0.9, 1.1, n))

    def objective(p):
        model = guide(
            mesh, E1, E2, tan_delta=0.02, boundaries=[gfem.Absorbing("side")], clad=p
        )
        problem = gfem.Driven(FREQUENCY, polarization=polarization)
        return functional(gfem.solve(model, problem, options=OPTIONS).s.values[0])

    assert_gradient_matches(objective, p, unit_directions(rng, p.shape, 2), step=STEP)
