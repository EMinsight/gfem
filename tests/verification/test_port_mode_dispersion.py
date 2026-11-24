r"""Group index and dispersion of a Sellmeier silicon slab converge to closed forms.

A 220 nm silicon slab in silica at 1550 nm, crossed by a line port with
perfectly conducting ends 3.0 um from the core. Both materials are Sellmeier
laws, written as lossless Lorentz terms. The fundamental TE (Ez) and TM (Hz)
modes of the symmetric slab have a closed-form dispersion relation, and its
exact differentiation to third order gives the literals below (40-digit
arithmetic, rounded here to 15 digits). The group index is the energy-velocity
value, with the energy permittivity :math:`\partial(\omega\varepsilon)/\partial\omega`.
The element order refines the mesh's solution towards them.
"""

import itertools

import numpy as np
import pytest
from scipy.constants import c
from slab2d import write_slab

import gfem

ORDERS = (2, 3, 4, 5)
FREQUENCY = c / 1.55e-6

# (B_i, resonance wavelength in um) of the Sellmeier terms.
SILICA = [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)]
SILICON = [
    (10.6684293, 0.301516485),
    (0.0030434748, 1.13475115),
    (1.54133408, 1104.0),
]

# n_eff, n_g, beta_2 (ps^2/km), beta_3 (ps^3/km), D (ps/(nm km)).
LITERALS = {
    "Ez": (
        2.8494894674291,
        3.70669351037895,
        1874.25555506947,
        -1.30187817871025,
        -1469.48778932109,
    ),
    "Hz": (
        2.05461854735691,
        3.97575382401877,
        13081.5033305684,
        -55.2498156758791,
        -10256.3971910016,
    ),
}
PS2_KM, PS3_KM, D_UNIT = 1e-27, 1e-39, 1e-6  # to s^2/m, s^3/m, s/m^2
GATES = {"n_eff": 1e-12, "n_g": 1e-12, "beta_2": 1e-10, "beta_3": 1e-9, "D": 1e-10}


def _sellmeier(terms):
    return gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[
            gfem.Lorentz(delta_eps=b, resonance_frequency=c / (r * 1e-6), damping=0.0)
            for b, r in terms
        ],
    )


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=3.0,
        h_core=0.0275,
        h_clad=0.05,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(eps_r=_sellmeier(SILICON)),
            "clad": gfem.Material(eps_r=_sellmeier(SILICA)),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )


def _errors(result, polarization):
    n_eff, n_g, beta_2, beta_3, d = LITERALS[polarization]
    got = {
        "n_eff": np.asarray(result.n_eff)[0],
        "n_g": np.asarray(result.n_g)[0],
        "beta_2": np.asarray(result.beta_2)[0] / PS2_KM,
        "beta_3": np.asarray(result.beta_3)[0] / PS3_KM,
        "D": np.asarray(result.D)[0] / D_UNIT,
    }
    want = {"n_eff": n_eff, "n_g": n_g, "beta_2": beta_2, "beta_3": beta_3, "D": d}
    return {k: abs(got[k] - want[k]) / abs(want[k]) for k in want}, got


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_dispersion_converges_to_the_closed_form_with_order(model, polarization):
    errors, imag = {}, []
    for p in ORDERS:
        result = gfem.solve(
            model,
            gfem.PortMode("p1", FREQUENCY, polarization=polarization),
            options=gfem.SolverOptions(order=p),
        )
        errors[p], got = _errors(result, polarization)
        imag.append(
            max(abs(v.imag) / abs(v.real) for v in got.values() if v is not None)
        )
    for name, gate in GATES.items():
        series = [errors[p][name] for p in ORDERS]
        assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(series)), (name, series)
        assert series[-1] < gate, (name, series)
    assert max(imag) < 1e-12, imag
