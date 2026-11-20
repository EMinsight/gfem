r"""V5: a line port's mode on the 2D bend against the symmetric slab's closed form.

The bend tutorial's triangle mesh (``docs/tutorials/data/bend2d.msh``): port
``port1`` crosses the input arm, a core of index 2.25 and width 0.55 um in a
cladding of index 1.444, with perfectly conducting ends (``pec_edges``). Its
fundamental mode in each polarization matches the even mode of the infinite
symmetric slab,

.. math::

    \tan\frac{\kappa w}{2} = r\,\frac{\alpha}{\kappa},\quad
    \kappa = k_0\sqrt{n_1^2 - n_\text{eff}^2},\quad
    \alpha = k_0\sqrt{n_\text{eff}^2 - n_2^2},

with :math:`r = 1` for Ez (TE0) and :math:`r = n_1^2/n_2^2` for Hz (TM0). The
port's ends sit 0.825 um from the core, where the mode has decayed below
:math:`10^{-4}` of its effective index.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c
from scipy.optimize import brentq

import gfem

MESH = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "bend2d.msh"
um = 1e-6
WAVELENGTHS = (1.50, 1.55, 1.60)
FREQUENCIES = [c / (w * um) for w in WAVELENGTHS]
N_CORE, N_CLADDING, WIDTH = 2.25, 1.444, 0.55


def slab_index(wavelength: float, polarization: str) -> float:
    """The even fundamental mode of the symmetric slab (lengths in um)."""
    k0 = 2 * np.pi / wavelength
    ratio = 1.0 if polarization == "Ez" else N_CORE**2 / N_CLADDING**2

    def mismatch(n):
        kappa = k0 * np.sqrt(N_CORE**2 - n**2)
        alpha = k0 * np.sqrt(n**2 - N_CLADDING**2)
        return np.tan(kappa * WIDTH / 2) - ratio * alpha / kappa

    lowest = max(np.sqrt(N_CORE**2 - (wavelength / (2 * WIDTH)) ** 2), N_CLADDING)
    return brentq(mismatch, lowest + 1e-9, N_CORE - 1e-9)


@pytest.fixture(scope="module")
def bend():
    mesh = gfem.Mesh.from_gmsh(MESH, length_unit="um")
    cladding = gfem.Material(eps_r=N_CLADDING**2)
    return gfem.Model(
        mesh,
        materials={
            "cladding": cladding,
            "design": cladding,
            "core": gfem.Material(eps_r=N_CORE**2),
        },
        boundaries=[gfem.Absorbing("outer", order=2)],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_v5_a_line_ports_mode_has_the_slab_index(bend, polarization):
    mode = gfem.solve(bend, gfem.PortMode("p1", FREQUENCIES, polarization=polarization))
    expected = [slab_index(w, polarization) for w in WAVELENGTHS]
    n_eff = np.asarray(mode.n_eff)
    np.testing.assert_allclose(n_eff.real, expected, rtol=0, atol=1e-4)
    np.testing.assert_allclose(n_eff.imag, 0.0, atol=1e-9)
