r"""Jones, circular and Mueller views of a cell against closed forms, at p = 3.

The Fresnel slab F1 (`periodic_cases.f1`) at normal incidence, at conical
incidence (40 degrees, azimuth 30) and as the Brewster cell, against the
tangential-E Airy form through `gfem.diffraction`: the Jones blocks of both
faces, the efficiencies and the power balance, and the Stokes parameters of
the light the Brewster cell reflects. The uniaxial wave plate
(`periodic_cases.waveplate`, a quarter-wave plate at its optic axis angle
psi = 0 and 30 degrees) against $R(\psi)\,\mathrm{diag}(\text{Airy}_e,
\text{Airy}_o)\,R(\psi)^T$: the Jones, circular and Mueller views, the
handedness of 45-degree linear light and the Pancharatnam-Berry phases of
the circular entries. Meshes at h about lambda/10, the direct solver.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest
from polarization_reference import circular_unitary, mueller_measured, rotation

import gfem
from gfem import diffraction as dif

TOP, BOT = "z_max", "z_min"
F0 = pc.F1_FREQUENCY
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
#: Efficiencies against the Airy form (the |S| error 1.8e-6 at most, twice
#: over, measured 5e-6 on the cross-polarized Brewster rows).
EFFICIENCY_TOL = 1e-5
#: |S|^2-level cross-polarized parts of the Mueller matrix of the plate
#: (4.6e-7 measured, ten times under the gate).
MUELLER_TOL = 1e-5
#: The efficiencies of the 45-degree linear light of the wave plate (closed
#: form, `test_v2_...`), 1e-5 against FEM (2e-7 measured).
HANDEDNESS_TOL = 1e-5
#: The Pancharatnam-Berry phases, rad (4e-7 measured).
PHASE_TOL = 1e-4
#: Reciprocity of the plate's Jones blocks (3e-15 measured).
RECIPROCITY_TOL = 1e-10
#: (outgoing, incident, h_out, h_in) of the four blocks of a two-face cell.
BLOCKS = (
    (TOP, TOP, 1, -1),
    (BOT, TOP, -1, -1),
    (TOP, BOT, 1, 1),
    (BOT, BOT, -1, 1),
)


def _solve(model, k):
    problem = gfem.Driven([F0], wave_vector=k)
    return gfem.solve(model, problem, options=OPTIONS)


# --- F1 and the Brewster cell: Jones blocks against Airy ------------------------


def _f1(tmp_path_factory, cells, eps_sub, theta, phi):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("f1"), cells)
    model = pc.f1(mesh, eps_sub=eps_sub)
    k = None
    kt = 0.0
    if theta:
        k = model.bloch_wave_vector(TOP, F0, theta=theta, phi=phi)
        kt = float(np.linalg.norm(k))
    return model, _solve(model, k), kt, eps_sub


@pytest.fixture(scope="module")
def normal(tmp_path_factory):
    return _f1(tmp_path_factory, pc.F1_FINE_CELLS, pc.F1_EPS_SUB, 0.0, 0.0)


@pytest.fixture(scope="module")
def conical(tmp_path_factory):
    return _f1(
        tmp_path_factory, pc.F1_FINE_CELLS, pc.F1_EPS_SUB, pc.F1_THETA, pc.F1_PHI
    )


@pytest.fixture(scope="module")
def brewster(tmp_path_factory):
    return _f1(
        tmp_path_factory,
        pc.BREWSTER_CELLS,
        pc.BREWSTER_EPS_SUB,
        pc.BREWSTER_THETA,
        0.0,
    )


@pytest.fixture(params=["normal", "conical", "brewster"])
def case(request):
    return request.getfixturevalue(request.param)


def _airy_blocks(kt, eps_sub):
    """The closed-form Jones blocks of F1, ``{(out, inc): diag(TM, TE)}``."""
    d, slab = (pc.F1_D * pc.UM), (pc.F1_EPS_SLAB, pc.F1_H * pc.UM)
    stacks = {
        TOP: [(1.0, d), slab, (eps_sub, d)],
        BOT: [(eps_sub, d), slab, (1.0, d)],
    }
    blocks = {}
    for inc, layers in stacks.items():
        other = BOT if inc == TOP else TOP
        pair = {p: pc.airy(F0, kt, layers, p) for p in ("TM", "TE")}
        blocks[inc, inc] = np.diag([pair["TM"][0], pair["TE"][0]])
        blocks[other, inc] = np.diag([pair["TM"][1], pair["TE"][1]])
    return blocks


@pytest.mark.parametrize(("out", "inc", "h_out", "h_in"), BLOCKS)
def test_v1_the_jones_blocks_match_airy(case, out, inc, h_out, h_in):
    model, result, kt, eps_sub = case
    ref = _airy_blocks(kt, eps_sub)[out, inc]
    j = dif.jones(result, out, inc, model=model)
    assert j.direction == (h_out, h_in)
    got = np.asarray(j.values)[0]
    for i in range(2):
        assert abs(got[i, i] - ref[i, i]) <= pc.F1_S_TOL
    assert abs(got[0, 1]) <= pc.F1_CROSS_TOL
    assert abs(got[1, 0]) <= pc.F1_CROSS_TOL
    circ = circular_unitary(h_out).conj().T @ ref @ circular_unitary(h_in)
    assert np.abs(np.asarray(j.circular)[0] - circ).max() <= pc.F1_S_TOL


def _expected_rows(ref, surface, pol):
    """The closed-form linear and circular rows of the efficiencies of ``pol``."""
    h_in = -1 if surface == TOP else 1
    pairs = {
        "TE": [np.array([0.0, 1.0])],
        "TM": [np.array([1.0, 0.0])],
        "RCP": [circular_unitary(h_in)[:, 0]],
        "unpolarized": [np.array([0.0, 1.0]), np.array([1.0, 0.0])],
    }[pol]
    rows = {}
    for out in (TOP, BOT):
        h_out = 1 if out == TOP else -1
        for a in pairs:
            b = ref[out, surface] @ a
            lin = np.abs(b) ** 2
            circ = np.abs(circular_unitary(h_out).conj().T @ b) ** 2
            for name, v in (("TM", lin[0]), ("TE", lin[1])):
                rows[out, name] = rows.get((out, name), 0.0) + v / len(pairs)
            for name, v in (("RCP", circ[0]), ("LCP", circ[1])):
                rows[out, name] = rows.get((out, name), 0.0) + v / len(pairs)
    return rows


@pytest.mark.parametrize("surface", [TOP, BOT])
@pytest.mark.parametrize("pol", ["TE", "TM", "RCP", "unpolarized"])
def test_v1_the_efficiencies_and_the_balance_match_airy(case, surface, pol):
    model, result, kt, eps_sub = case
    rows = _expected_rows(_airy_blocks(kt, eps_sub), surface, pol)
    for basis in ("linear", "circular"):
        eff = dif.efficiencies(
            result, surface, polarization=pol, model=model, basis=basis
        )
        for (out, order, name), value in zip(
            eff.rows, np.asarray(eff.values[0]), strict=True
        ):
            assert order == (0, 0)
            assert abs(value - rows[out, name]) <= EFFICIENCY_TOL
    bal = dif.power_balance(result, surface, polarization=pol, model=model)
    assert float(abs(bal.absorptance[0])) <= pc.F1_POWER_TOL
    other = BOT if surface == TOP else TOP
    for out, value in ((surface, bal.reflectance), (other, bal.transmittance)):
        expected = rows[out, "TM"] + rows[out, "TE"]
        assert abs(float(value[0]) - expected) <= EFFICIENCY_TOL


def test_v1_the_brewster_cell_reflects_no_tm_and_rcp_reflects_pure_te(brewster):
    model, result, _, _ = brewster
    tm = dif.power_balance(result, TOP, polarization="TM", model=model)
    assert float(tm.reflectance[0]) <= pc.BREWSTER_TOL**2
    mueller = np.asarray(dif.jones(result, TOP, TOP, model=model).mueller)[0]
    stokes = mueller @ np.array([1.0, 0.0, 0.0, 1.0])
    # TE light is the phi component alone: S1 = -S0, S2 = S3 = 0
    assert abs(stokes[3] / stokes[0]) <= 1e-4
    assert abs(stokes[1] / stokes[0] + 1) <= 1e-4
    assert abs(stokes[2] / stokes[0]) <= 1e-4
    assert stokes[0] > 0.07  # TE does reflect: half of its 0.155 for RCP


# --- the wave plate ---------------------------------------------------------------


def _plate_airy(eps):
    return pc.airy(
        F0,
        0.0,
        [(1.0, pc.F1_D * pc.UM), (eps, pc.F1_H * pc.UM), (1.0, pc.F1_D * pc.UM)],
        "TE",
    )


@pytest.fixture(scope="module", params=[0.0, 30.0])
def plate(request, tmp_path_factory):
    psi = math.radians(request.param)
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("plate"), pc.F1_FINE_CELLS)
    model = pc.waveplate(mesh, psi)
    return model, _solve(model, None), psi


def _plate_blocks(psi):
    r"""``(J^r, J^t)`` of the plate.

    Both are $R\,\mathrm{diag}(\text{Airy}_e, \text{Airy}_o)\,R^T$.
    """
    (r_e, t_e), (r_o, t_o) = (
        _plate_airy(e) for e in (pc.WAVEPLATE_EPS_E, pc.WAVEPLATE_EPS_O)
    )
    rot = rotation(psi)
    return (
        rot @ np.diag([r_e, r_o]) @ rot.T,
        rot @ np.diag([t_e, t_o]) @ rot.T,
    )


@pytest.mark.parametrize(("out", "inc", "h_out", "h_in"), BLOCKS)
def test_v2_the_plate_jones_circular_and_mueller_match_the_rotated_airy(
    plate, out, inc, h_out, h_in
):
    model, result, psi = plate
    j_r, j_t = _plate_blocks(psi)
    ref = j_r if out == inc else j_t
    j = dif.jones(result, out, inc, model=model)
    assert np.abs(np.asarray(j.values)[0] - ref).max() <= pc.F1_S_TOL
    circ = circular_unitary(h_out).conj().T @ ref @ circular_unitary(h_in)
    assert np.abs(np.asarray(j.circular)[0] - circ).max() <= pc.F1_S_TOL
    mueller = mueller_measured(ref, h_out, h_in)
    assert np.abs(np.asarray(j.mueller)[0] - mueller).max() <= MUELLER_TOL


def test_v2_the_plate_is_reciprocal(plate):
    model, result, _ = plate
    down = np.asarray(dif.jones(result, BOT, TOP, model=model).values)[0]
    up = np.asarray(dif.jones(result, TOP, BOT, model=model).values)[0]
    assert np.abs(up - down.T).max() <= RECIPROCITY_TOL
    for surface in (TOP, BOT):
        r = np.asarray(dif.jones(result, surface, surface, model=model).values)[0]
        assert np.abs(r - r.T).max() <= RECIPROCITY_TOL


def test_v2_45_degree_linear_light_leaves_with_the_handedness_of_the_plate(plate):
    model, result, psi = plate
    _, j_t = _plate_blocks(psi)
    # 45 degrees from the optic axis, in the lab frame
    lab = rotation(psi) @ np.array([1.0, 1.0]) / math.sqrt(2)
    t_e, t_o = _plate_airy(pc.WAVEPLATE_EPS_E)[1], _plate_airy(pc.WAVEPLATE_EPS_O)[1]
    # along the axis the plate delays e by -pi/2: the output (t_e, t_o)/sqrt2 in the
    # axis frame; going down (h = -1) it is right-handed by (t_e - i t_o)/2
    closed = {
        TOP: (abs(t_e - 1j * t_o) ** 2 / 4, abs(t_e + 1j * t_o) ** 2 / 4),
    }
    closed[BOT] = closed[TOP][::-1]
    assert abs(closed[TOP][0] - 0.77796) <= 1e-5
    assert abs(closed[TOP][1] - 0.01205) <= 1e-5
    for inc, out in ((TOP, BOT), (BOT, TOP)):
        eff = dif.efficiencies(
            result, inc, polarization=tuple(lab), model=model, basis="circular"
        )
        got = {name: float(eff[out, (0, 0), name][0]) for name in ("RCP", "LCP")}
        assert abs(got["RCP"] - closed[inc][0]) <= HANDEDNESS_TOL
        assert abs(got["LCP"] - closed[inc][1]) <= HANDEDNESS_TOL
        # the same from the closed-form Jones matrix
        h_out = 1 if out == TOP else -1
        b = circular_unitary(h_out).conj().T @ (j_t @ lab)
        assert abs(abs(b[0]) ** 2 - got["RCP"]) <= HANDEDNESS_TOL


def test_v2_the_circular_entries_carry_the_pancharatnam_berry_phase(tmp_path_factory):
    mesh = pc.f1_mesh(tmp_path_factory.mktemp("pb"), pc.F1_FINE_CELLS)
    psi = math.radians(30.0)
    circ = {}
    for angle in (0.0, psi):
        model = pc.waveplate(mesh, angle)
        result = _solve(model, None)
        circ[angle] = {
            (out, inc): np.asarray(dif.jones(result, out, inc, model=model).circular)[0]
            for out, inc, _, _ in BLOCKS
        }

    def phase(out, inc, row, col):
        ratio = circ[psi][out, inc][row, col] / circ[0.0][out, inc][row, col]
        return float(np.angle(ratio))

    # rows and columns (RCP, LCP): [1, 0] is LCP <- RCP
    for out, inc, row, col, sign in (
        (BOT, TOP, 1, 0, +2),  # from the top: +2 psi
        (BOT, TOP, 0, 1, -2),
        (TOP, BOT, 1, 0, -2),  # from the bottom: -2 psi
        (TOP, BOT, 0, 1, +2),
        (TOP, TOP, 0, 0, +2),  # reflected RR
        (TOP, TOP, 1, 1, -2),
    ):
        assert abs(phase(out, inc, row, col) - sign * psi) <= PHASE_TOL
    # the entries that keep the handedness of the plate's transmission carry none
    assert abs(phase(BOT, TOP, 0, 0)) <= PHASE_TOL
    assert abs(phase(BOT, TOP, 1, 1)) <= PHASE_TOL
