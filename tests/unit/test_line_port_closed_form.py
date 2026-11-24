r"""A uniform line port's mode has the closed-form derivative of its index.

With natural ends the mode of a uniform line in Ez is constant along it,
:math:`\gamma = ik_0\sqrt{\varepsilon_r}` exactly on the discrete line (a
constant is in every H1 space), so

.. math::

    \frac{\mathrm{d}\gamma}{\mathrm{d}\varepsilon_r}
    = \frac{ik_0}{2\sqrt{\varepsilon_r}}.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from line_port_cases import FREQUENCY, guide, write_guide
from scipy.constants import c

import gfem


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = write_guide(tmp_path_factory.mktemp("guide") / "guide.msh", nx=4, ny=4)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def test_the_derivative_of_gamma_by_a_uniform_permittivity(mesh):
    eps = 2.5
    k0 = 2 * np.pi * FREQUENCY / c
    sizes = [len(mesh.elements(r)) for r in ("clad", "core1", "core2")]

    def gamma(p):
        fields = [jnp.full(n, p) for n in sizes]
        model = guide(mesh, fields[1], fields[2], clad=fields[0], boundaries=[])
        mode = gfem.solve(model, gfem.PortMode("p1", [FREQUENCY], polarization="Ez"))
        return mode.gamma[0]

    value = gamma(jnp.asarray(eps))
    np.testing.assert_allclose(value, 1j * k0 * np.sqrt(eps), rtol=1e-10)
    # the derivative of the imaginary part (JAX needs a real output)
    slope = jax.grad(lambda p: jnp.imag(gamma(p)))(jnp.asarray(eps))
    np.testing.assert_allclose(slope, k0 / (2 * np.sqrt(eps)), rtol=1e-9)
