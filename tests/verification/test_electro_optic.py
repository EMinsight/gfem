r"""The first-order electro-optic shift against closed forms and mode re-solves.

- A layered slab (oxide, a 0.6 um lithium niobate film, oxide) between
  electrode edges 8 um apart: a single static permittivity gives the
  uniform lateral field :math:`E_x = -V/d`. PEC on the electrode edges keeps
  the TE mode the 1D slab's, so the shift is the closed form
  :math:`\delta n_\mathrm{eff} = \Delta\varepsilon\,\Gamma/(2 n_\mathrm{eff})`
  with :math:`\Gamma` the film's share of :math:`\int |E|^2` and
  :math:`n_\mathrm{eff}` from the exact 1D TE slab: for x-cut,
  :math:`\Delta\varepsilon = -n_e^4 r_{33} E_x`; with the crystal turned so
  that its y axis is along the field, :math:`-n_o^4 r_{22} E_x`. Within
  :math:`2\times10^{-4}` (:math:`5\times10^{-5}` measured).
- A ridge between electrodes: the crystal-frame diagonal part of the
  change, re-solved per element at :math:`\pm 10` V, against the first
  order of the tensor's rows xx, yy, zz. Within :math:`10^{-6}`
  (:math:`5\times10^{-8}` measured).
- An L-shaped core under a uniform vertical field with :math:`r_{51}`
  only: the change is an in-plane off-diagonal entry; the perturbed
  permittivity, diagonalized into principal values on axes turned in the
  section plane, re-solved at :math:`\pm 10` V. Within :math:`10^{-6}`
  (:math:`6\times10^{-8}` measured). A mirror-symmetric core would give a
  vanishing :math:`r_{51}` shift.
- The ridge's static field solved on a second mesh of the same geometry,
  read across meshes. Within :math:`10^{-3}` (:math:`5.5\times10^{-4}`
  measured).

Central differences: a one-sided re-solve carries the second order
(:math:`10^{-5}`).
"""

import electro_optic_cases as eo
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import gfem

K0 = 2 * np.pi / eo.WAVELENGTH
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
#: Crystal y along the field (global x), crystal z up, crystal x along the guide.
TURNED = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
#: The ridge's TE mode on the full section: two slab modes guided under the
#: electrodes' shells and two film modes sit above it.
RIDGE_MODE = 5
VOLTS = 10.0


def _te_slab(eps_film):
    """The 1D TE slab's effective index and the film's share of the field energy."""
    d, n_ox = eo.SLAB_FILM * 1e-6, eo.N_OX

    def kappa(n):
        return K0 * np.sqrt(eps_film - n * n)

    def gamma(n):
        return K0 * np.sqrt(n * n - n_ox * n_ox)

    def even(n):
        return np.tan(kappa(n) * d / 2) - gamma(n) / kappa(n)

    lowest = np.sqrt(max(eps_film - (np.pi / (K0 * d)) ** 2, n_ox**2)) + 1e-9
    n = brentq(even, lowest, np.sqrt(eps_film) - 1e-12, xtol=1e-15)
    film = quad(lambda x: np.cos(kappa(n) * x) ** 2, -d / 2, d / 2)[0]
    cladding = np.cos(kappa(n) * d / 2) ** 2 / gamma(n)
    return n, film / (film + cladding)


def _shift(model, potential, **kw):
    return gfem.electro_optic_shift(
        model, potential, port="p", frequency=eo.FREQUENCY, **kw
    )


def _n_eff(model, mode=1):
    port_mode = gfem.solve(model, gfem.PortMode("p", eo.FREQUENCY, mode=mode))
    return float(np.real(port_mode.n_eff[0]))


# --- layered slab ---------------------------------------------------------------


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    return eo.layered_slab(tmp_path_factory.mktemp("slab") / "slab.msh")


@pytest.mark.parametrize(
    ("axes", "n_film", "entry"),
    [(eo.X_CUT, eo.N_E, eo.R33), (TURNED, eo.N_O, eo.R22)],
    ids=["x-cut-r33", "turned-r22"],
)
def test_a_slab_matches_the_1d_closed_form(slab, axes, n_film, entry):
    static = gfem.Model(
        slab,
        materials={r: gfem.Material(eps_r=3.9) for r in ("below", "film", "above")},
    )
    drive = gfem.Electrostatic(terminals={"r": ["right"]}, ground=["left"])
    potential = gfem.solve(static, drive, options=OPTIONS).potential["r"]
    oxide = gfem.Material(eps_r=eo.N_OX**2)
    film = gfem.Material(eps_r=eo.LN_OPTICAL, axes=axes, electro_optic_tensor=eo.R_LN)
    optical = gfem.Model(
        slab,
        materials={"below": oxide, "film": film, "above": oxide},
        boundaries=[gfem.PEC("left"), gfem.PEC("right")],
        ports=[gfem.WavePort("p", "front")],
    )
    shift = _shift(optical, potential)
    n, share = _te_slab(n_film**2)
    field = -1 / (2 * eo.SLAB_HALF * 1e-6)  # V/m at 1 V
    closed = -(n_film**4) * entry * field * share / (2 * n)
    np.testing.assert_allclose(shift.n_eff, n, rtol=1e-4)
    np.testing.assert_allclose(shift.delta_n_eff_per_volt, closed, rtol=2e-4)


# --- L-shaped core, r51 ----------------------------------------------------------


def test_the_r51_shift_on_an_l_core_matches_a_re_solve(tmp_path):
    mesh = eo.l_core(tmp_path / "lcore.msh", 0.3, 0.06)
    regions = ("core", "foot", "clad")
    static = gfem.Model(mesh, materials={r: gfem.Material(eps_r=3.9) for r in regions})
    drive = gfem.Electrostatic(terminals={"t": ["top"]}, ground=["bottom"])
    potential = gfem.solve(static, drive, options=OPTIONS).potential["t"]

    def optical(crystal):
        oxide = gfem.Material(eps_r=eo.N_OX**2)
        return gfem.Model(
            mesh,
            materials={"clad": oxide, "core": crystal, "foot": crystal},
            ports=[gfem.WavePort("p", "front")],
        )

    r51 = eo.tensor_3m(r13=0.0, r22=0.0, r33=0.0)
    first = _shift(
        optical(
            gfem.Material(eps_r=eo.LN_OPTICAL, axes=eo.X_CUT, electro_optic_tensor=r51)
        ),
        potential,
    )
    # E = -1/GAP along global y = crystal x: the crystal's xz entry,
    # -eps_x eps_z r51 E, sits at the global (x, y) entry; the section's
    # x, y principal values are n_e^2, n_o^2
    off = eo.N_O**2 * eo.N_E**2 * eo.R51 / eo.GAP
    n_eff = []
    for volts in (VOLTS, -VOLTS):
        eps = np.diag([eo.N_E**2, eo.N_O**2, eo.N_O**2])
        eps[0, 1] = eps[1, 0] = volts * off
        values, vectors = np.linalg.eigh(eps)
        axes = vectors.T * np.sign(np.linalg.det(vectors))
        crystal = gfem.Material(eps_r=tuple(values), axes=tuple(map(tuple, axes)))
        n_eff.append(_n_eff(optical(crystal)))
    resolved = (n_eff[0] - n_eff[1]) / (2 * VOLTS)
    # the in-plane term is not small: n_eff moves by -3e-5 per volt
    assert abs(float(first.delta_n_eff_per_volt)) > 1e-5
    np.testing.assert_allclose(first.delta_n_eff_per_volt, resolved, rtol=1e-6)


# --- ridge between electrodes ------------------------------------------------------


def _ridge_static(mesh):
    crystal = gfem.Material(eps_r=eo.LN_STATIC, axes=eo.X_CUT)
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "box": gfem.Material(eps_r=3.9),
            "film": crystal,
            "ridge": crystal,
            "signal": gfem.Material(),
            "ground": gfem.Material(),
        },
        excluded={"em": ["signal", "ground"]},
    )


def _ridge_optical(mesh, film, ridge=None):
    return gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "box": gfem.Material(eps_r=eo.N_OX**2),
            "film": film,
            "ridge": film if ridge is None else ridge,
            "signal": gfem.Material(),
            "ground": gfem.Material(),
        },
        ports=[gfem.WavePort("p", "front")],
        excluded={"em": ["signal", "ground"]},
    )


@pytest.fixture(scope="module")
def ridge(tmp_path_factory):
    return eo.tfln_ridge(tmp_path_factory.mktemp("ridge") / "ridge.msh")


@pytest.fixture(scope="module")
def ridge_potential(ridge):
    solved = gfem.solve(_ridge_static(ridge), eo.RIDGE_DRIVE, options=OPTIONS)
    return solved.potential["signal"]


def test_the_ridge_s_diagonal_shift_matches_a_re_solve(ridge, ridge_potential):
    from gfem.electro_optic import _static_field

    # rows xx, yy, zz: the change is diagonal in the crystal frame, so the
    # perturbed crystal keeps its axes with new principal values per element
    diagonal = eo.tensor_3m()
    diagonal[3:] = 0.0
    crystal = gfem.Material(
        eps_r=eo.LN_OPTICAL, axes=eo.X_CUT, electro_optic_tensor=diagonal
    )
    first = _shift(_ridge_optical(ridge, crystal), ridge_potential, mode=RIDGE_MODE)
    assert 1.83 < float(first.n_eff) < 1.90

    eps = np.asarray(eo.LN_OPTICAL)
    change = {}
    for region in ("film", "ridge"):
        tets = ridge.elements(region)
        field = np.asarray(_static_field(ridge_potential, ridge, tets, "test"))
        in_crystal = field @ np.asarray(eo.X_CUT).T
        change[region] = -(eps**2) * (in_crystal @ diagonal[:3].T)
    n_eff = []
    for volts in (VOLTS, -VOLTS):
        film, ridge_ = (
            gfem.MaterialField(eps_r=eps + volts * change[r], axes=eo.X_CUT)
            for r in ("film", "ridge")
        )
        n_eff.append(_n_eff(_ridge_optical(ridge, film, ridge_), RIDGE_MODE))
    resolved = (n_eff[0] - n_eff[1]) / (2 * VOLTS)
    np.testing.assert_allclose(first.delta_n_eff_per_volt, resolved, rtol=1e-6)


def test_the_ridge_s_field_read_across_meshes_matches_its_own_mesh(
    tmp_path, ridge, ridge_potential
):
    crystal = gfem.Material(
        eps_r=eo.LN_OPTICAL, axes=eo.X_CUT, electro_optic_tensor=eo.R_LN
    )
    optical = _ridge_optical(ridge, crystal)
    own = _shift(optical, ridge_potential, mode=RIDGE_MODE)
    other = eo.tfln_ridge(tmp_path / "other.msh", factor=0.93)
    solved = gfem.solve(_ridge_static(other), eo.RIDGE_DRIVE, options=OPTIONS)
    across = _shift(optical, solved.potential["signal"], mode=RIDGE_MODE)
    np.testing.assert_allclose(
        across.delta_n_eff_per_volt, own.delta_n_eff_per_volt, rtol=1e-3
    )
