r"""Gradients of an eigenmode with wave ports against exact values.

Case A (a short, :math:`\varepsilon_r = 4` for 10 mm, 5 mm of air up to a
WR-90 TE10 port) has an exact complex root, a transmission-line equation
along the guide, so the derivative of the root by the layer's permittivity
is exact too (a fourth-order difference of exact roots, step
:math:`10^{-6}`). The layer does not touch the port face: the port term is
held, and the gradient carries it through :math:`T'(k)` alone. Q is
:math:`\mathrm{Re}\,\omega / 2\,\mathrm{Im}\,\omega`.

B3 (two TE10 ports on separate surfaces, three windows) checks the coupling
Q's gradient where both ports carry power, against central differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import WAVE_A, WAVE_A_ROOT, wave_guide, wave_resonator

import gfem

#: d f / d eps_r of the layer (GHz) and d Q / d eps_r, exact.
DF_DEPS = -1.4557073690 - 0.3555345908j
DQ_DEPS = 1.15208486
#: B3: windows of eps_r = 10 between two TE10 ports; exact root, GHz.
WAVE_B3 = (
    ("air_l", 1.0, 3.0),
    ("window_1", 10.0, 2.4),
    ("air_1", 1.0, 20.0),
    ("window_2", 10.0, 2.4),
    ("air_2", 1.0, 9.9),
    ("window_3", 10.0, 2.4),
    ("air", 1.0, 3.0),
)
ROOT_B3 = 9.974259423230 + 0.096704835502j
GHZ = 2 * np.pi * 1e9


def _nearest(omega, root) -> int:
    return int(np.argmin(np.abs(np.asarray(omega) / GHZ - root)))


def test_case_a_frequency_and_q_gradients_match_the_exact_derivative(tmp_path):
    mesh = wave_guide(tmp_path, WAVE_A)
    problem = gfem.Eigenmode(target=11.67e9, n_modes=3)
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    ports = [gfem.WavePort("p1", "port")]

    def omega(eps):
        design = {"dielectric": gfem.MaterialField(eps_r=eps)}
        result = gfem.solve(
            wave_resonator(mesh, WAVE_A, design, ports), problem, options=options
        )
        return result.omega

    eps = jnp.asarray(4.0)
    j = _nearest(omega(eps), WAVE_A_ROOT)
    d_re = jax.grad(lambda e: jnp.real(omega(e)[j]))(eps)
    d_im = jax.grad(lambda e: jnp.imag(omega(e)[j]))(eps)
    df = complex(d_re + 1j * d_im) / GHZ
    d_q = float(
        jax.grad(lambda e: jnp.real(omega(e)[j]) / (2 * jnp.imag(omega(e)[j])))(eps)
    )
    assert abs(df - DF_DEPS) <= 1e-4 * abs(DF_DEPS)
    assert abs(d_q - DQ_DEPS) <= 1e-4 * abs(DQ_DEPS)


def test_two_ports_coupling_q_gradients_match_central_differences(tmp_path):
    mesh = wave_guide(tmp_path, WAVE_B3, left_port=True)
    problem = gfem.Eigenmode(target=9.97e9, n_modes=3)
    options = gfem.SolverOptions(order=2, linear_solver="direct", eigen_tolerance=1e-12)
    ports = [gfem.WavePort("left", "portl"), gfem.WavePort("right", "port")]

    def solve(eps):
        design = {"air_1": gfem.MaterialField(eps_r=eps)}
        return gfem.solve(
            wave_resonator(mesh, WAVE_B3, design, ports), problem, options=options
        )

    eps = jnp.asarray(np.linspace(1.0, 1.2, len(mesh.elements("air_1"))))
    j = _nearest(solve(eps).omega, ROOT_B3)

    def objective(eps):
        q_port = solve(eps).q_port
        return 30.0 / q_port["left"][j] + 300.0 / q_port["right"][j]

    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)
