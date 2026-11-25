"""Periodic eigenmode solves, plain and Bloch, against closed forms.

- C2, an empty parallel-plate cell periodic along x and y: its 4 lowest
  Bloch bands are the empty lattice, real (the real snap of a lossless,
  Hermitian cell), and the lowest band's field advances by exp(-i k·Δr)
  between two points, which pins the sign of the Bloch phase.
  The same bands on an unstructured gmsh mesh tied by `setPeriodic`.
- C3, a layered (Bragg) line: its bands are the transfer-matrix roots,
  lossless (real) and lossy (complex), at both edges of the gap k = π/a, at
  π/2a, and plain at Γ.
- C4, an empty cube periodic along x, y and z (no other boundary): the
  lowest band is a doublet at c|k| / 2π.
- C1, the Bloch line loaded by its two 50 ohm ports: a damped Bloch problem
  (its left eigenvectors are not conj(x): the two-sided refinement) whose
  resonance is the root of det(Z(f, k) + R) = 0.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest
import scipy.optimize

import gfem

# --- C2: empty parallel-plate cell -------------------------------------------------


@pytest.fixture(scope="module")
def c2_modes(tmp_path_factory):
    mesh = pc.c2_mesh(tmp_path_factory.mktemp("c2"))
    exact = _c2_exact()
    problem = gfem.Eigenmode(target=0.9 * exact[0], n_modes=4, wave_vector=pc.C2_K)
    options = gfem.SolverOptions(order=pc.C2_ORDER)
    return gfem.solve(pc.c2(mesh), problem, options=options)


def _c2_exact():
    periods = (pc.C2_AX * pc.MM, pc.C2_AY * pc.MM, None)
    return pc.empty_lattice(pc.C2_K, periods, pc.C2_EPS)


def test_empty_plate_cell_bands_match_the_empty_lattice(c2_modes):
    f = np.asarray(c2_modes.omega).real / (2 * math.pi)
    np.testing.assert_allclose(np.sort(f), _c2_exact(), rtol=pc.C2_BAND_RTOL)


def test_empty_plate_cell_bands_are_real(c2_modes):
    # lossless and undamped: K and M are Hermitian, so the snap is exact
    assert not np.asarray(c2_modes.omega).imag.any()


def test_empty_plate_cell_field_pins_the_sign_of_the_phase(c2_modes):
    # the lowest band is G = 0: E = E0 exp(-i k·r) z
    f = np.asarray(c2_modes.omega).real / (2 * math.pi)
    j = int(np.argmin(np.abs(f - _c2_exact()[0])))
    e = np.asarray(c2_modes.e_fields[j].at(np.array(pc.C2_PROBES)))
    want = np.exp(-1j * np.dot(pc.C2_K, np.subtract(pc.C2_PROBES[1], pc.C2_PROBES[0])))
    assert abs(e[1, 2] / e[0, 2] - want) <= pc.C2_PHASE_TOL


def test_a_gmsh_set_periodic_cell_has_the_empty_lattice_bands():
    # C2 on an unstructured gmsh mesh whose pairs were tied by setPeriodic
    exact = _c2_exact()
    problem = gfem.Eigenmode(target=0.9 * exact[0], n_modes=4, wave_vector=pc.C2_K)
    options = gfem.SolverOptions(order=pc.C2_ORDER)
    modes = gfem.solve(pc.c2(pc.c2_gmsh_mesh()), problem, options=options)
    omega = np.asarray(modes.omega)
    f = np.sort(omega.real) / (2 * math.pi)
    np.testing.assert_allclose(f, exact, rtol=pc.C2_BAND_RTOL)  # 2.4e-5 measured
    assert not omega.imag.any()


def _c2_velocities(n: int = 4, k=pc.C2_K) -> tuple[np.ndarray, np.ndarray]:
    """The ``n`` lowest empty-lattice bands (Hz) and their velocities (m/s).

    The velocity of a band is (c / n_r) (k + G) / |k + G|.
    """
    periods = (pc.C2_AX * pc.MM, pc.C2_AY * pc.MM)
    out = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            q = np.array(
                [
                    k[0] + 2 * math.pi * i / periods[0],
                    k[1] + 2 * math.pi * j / periods[1],
                    0,
                ]
            )
            out.append((np.linalg.norm(q), q))
    out.sort(key=lambda item: item[0])
    f = np.array([c * pc.C0 / (2 * math.pi * math.sqrt(pc.C2_EPS)) for c, _ in out[:n]])
    v = np.array([pc.C0 / math.sqrt(pc.C2_EPS) * q / c for c, q in out[:n]])
    return f, v


def test_empty_plate_cell_group_velocity(c2_modes):
    # the full vector of each band: along k + G at the speed c / n
    _, v = _c2_velocities()
    omega = np.asarray(c2_modes.omega).real
    order = np.argsort(omega)
    got = np.asarray(c2_modes.v_g)[order]
    assert np.isfinite(got).all()
    assert not got.imag.any()
    assert np.abs(got - v).max() <= pc.C2_VG_TOL * pc.C0  # 6e-4 c measured


def test_a_group_cut_by_n_modes_has_nan_group_velocity():
    # k along x on the unstructured cell: the (0, +-1) doublet is the fourth
    # and fifth band, and four modes cut it
    k = (0.4 * math.pi / (pc.C2_AX * pc.MM), 0.0, 0.0)
    f, v = _c2_velocities(5, k)
    problem = gfem.Eigenmode(target=0.9 * f[0], n_modes=4, wave_vector=k)
    options = gfem.SolverOptions(order=pc.C2_ORDER)
    result = gfem.solve(pc.c2(pc.c2_gmsh_mesh()), problem, options=options)
    order = np.argsort(np.asarray(result.omega).real)
    got = np.asarray(result.v_g)[order]
    assert np.isnan(got[3]).all()  # the lone member of a cut doublet
    assert np.abs(got[:3] - v[:3]).max() <= 1e-3 * pc.C0  # 8e-5 c measured


# --- C3: Bragg line --------------------------------------------------------------


@pytest.fixture(scope="module")
def c3_mesh(tmp_path_factory):
    return pc.c3_mesh(tmp_path_factory.mktemp("c3"))


def _tmm_bands(
    k: float, tan_delta: float | tuple[float, float], f_max: float = 20e9
) -> np.ndarray:
    """The C3 TEM band frequencies (Hz) at k up to ``f_max``, independently.

    The lossless roots bracketed on a fine grid (no seed from the solve),
    each continued to the lossy root by `periodic_cases.tmm_frequency`.
    """
    a, d1 = pc.C3_A * pc.MM, pc.C3_D1 * pc.MM
    n1, n2 = math.sqrt(pc.C3_EPS_LOW), math.sqrt(pc.C3_EPS_HIGH)

    def residual(f):
        k0 = 2 * math.pi * f / pc.C0
        return (
            math.cos(n1 * k0 * d1) * math.cos(n2 * k0 * (a - d1))
            - 0.5
            * (n1 / n2 + n2 / n1)
            * math.sin(n1 * k0 * d1)
            * math.sin(n2 * k0 * (a - d1))
            - math.cos(k * a)
        )

    grid = np.linspace(1e7, f_max, 4000)
    values = [residual(f) for f in grid]
    roots = [
        scipy.optimize.brentq(residual, grid[i], grid[i + 1], xtol=1e-6)
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    if tan_delta:
        roots = [pc.tmm_frequency(k, complex(f), tan_delta) for f in roots]
    return np.array(roots, dtype=complex)


#: (k as a fraction of π/a, tan δ): both gap edges at the zone boundary, a
#: point inside the zone, lossless and lossy; None is plain periodicity (Γ).
C3_POINTS = [
    (1.0, 0.0),
    (0.5, 0.0),
    (1.0, pc.C3_TAN_DELTA),
    (0.5, pc.C3_TAN_DELTA),
    (None, 0.0),
    (None, pc.C3_TAN_DELTA),
]


@pytest.mark.parametrize(
    ("fraction", "tan_delta"),
    C3_POINTS,
    ids=[f"{'plain' if f is None else f'{f}pi/a'}-tan{t}" for f, t in C3_POINTS],
)
def test_bragg_line_bands_match_the_transfer_matrix(c3_mesh, fraction, tan_delta):
    a = pc.C3_A * pc.MM
    if fraction is None:
        k, wave_vector, target = 0.0, None, pc.C3_PLAIN_TARGET
    else:
        k = fraction * math.pi / a
        wave_vector = (k, 0.0, 0.0)
        # below every band: c k / (2 pi n_max) bounds the lowest one
        target = 0.9 * pc.C0 * k / (2 * math.pi * math.sqrt(pc.C3_EPS_HIGH))
    problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=wave_vector)
    options = gfem.SolverOptions(order=pc.C3_ORDER)
    result = gfem.solve(pc.c3(c3_mesh, tan_delta), problem, options=options)
    f = np.asarray(result.omega) / (2 * math.pi)
    bands = _tmm_bands(k, tan_delta)
    for value in f:
        nearest = bands[np.argmin(np.abs(bands - value))]
        assert abs(value - nearest) <= pc.C3_FREQ_RTOL * abs(nearest)
    if fraction is not None:
        # the two lowest bands: both edges of the gap at k = π/a
        np.testing.assert_allclose(
            np.sort(f.real), bands[:2].real, rtol=pc.C3_FREQ_RTOL
        )
    if tan_delta == 0:
        assert not np.asarray(result.omega).imag.any()
    else:
        assert (np.asarray(result.omega).imag > 0).all()


#: tan delta of the C3 layers (low, high): loss in the high layer only, where the
#: left eigenvector is not the conjugate of the mode.
C3_ONE_LAYER_LOSS = (0.0, 0.05)


@pytest.mark.parametrize(
    "tan_delta", [0.0, C3_ONE_LAYER_LOSS], ids=["lossless", "loss"]
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_bragg_line_band_group_velocity_matches_the_transfer_matrix(
    c3_mesh, fraction, tan_delta
):
    a = pc.C3_A * pc.MM
    k = fraction * math.pi / a
    target = 0.9 * pc.C0 * k / (2 * math.pi * math.sqrt(pc.C3_EPS_HIGH))
    problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=(k, 0.0, 0.0))
    options = gfem.SolverOptions(order=pc.C3_ORDER)
    result = gfem.solve(pc.c3(c3_mesh, tan_delta), problem, options=options)
    bands = _tmm_bands(k, tan_delta)
    omega = np.asarray(result.omega)
    v = np.asarray(result.v_g)
    for j in range(2):
        root = bands[np.argmin(np.abs(bands - omega[j] / (2 * math.pi)))]
        want = pc.tmm_group_velocity(k, root, tan_delta)
        # measured 1.8e-7 c lossless, 1.5e-7 c lossy (the conjugate of the mode
        # in place of its left eigenvector would be 9e-4 c off)
        assert abs(v[j, 0] - want) <= pc.C3_VG_TOL * pc.C0
        assert v[j, 1] == 0
        assert v[j, 2] == 0
    if tan_delta == 0:
        assert not v.imag.any()
    else:
        assert v.imag.any()


def test_bragg_line_group_velocity_limits(c3_mesh):
    a = pc.C3_A * pc.MM
    mean_eps = (
        pc.C3_EPS_LOW * pc.C3_D1 + pc.C3_EPS_HIGH * (pc.C3_A - pc.C3_D1)
    ) / pc.C3_A
    options = gfem.SolverOptions(order=pc.C3_ORDER)

    def solve(fraction):
        k = fraction * math.pi / a
        target = 0.9 * pc.C0 * k / (2 * math.pi * math.sqrt(pc.C3_EPS_HIGH))
        problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=(k, 0.0, 0.0))
        result = gfem.solve(pc.c3(c3_mesh), problem, options=options)
        order = np.argsort(np.asarray(result.omega).real)
        return np.asarray(result.v_g)[order, 0].real

    # long wavelengths: the band is the effective medium's light line
    low = solve(0.02)
    assert abs(low[0] / (pc.C0 / math.sqrt(mean_eps)) - 1) <= 1e-4  # 3.6e-5: dispersion
    assert low[1] < 0  # the second band falls from the gap's upper edge to Γ
    # the zone edge: both gap edges are flat
    assert np.abs(solve(1.0)).max() <= 1e-8 * pc.C0


# --- C4: empty cube ---------------------------------------------------------------


def test_empty_cube_lowest_band_is_the_light_line_doublet(tmp_path):
    mesh = pc.c4_mesh(tmp_path)
    exact = pc.C0 * np.linalg.norm(pc.C4_K) / (2 * math.pi)
    problem = gfem.Eigenmode(target=0.9 * exact, n_modes=2, wave_vector=pc.C4_K)
    options = gfem.SolverOptions(order=pc.C4_ORDER)
    result = gfem.solve(pc.c4(mesh), problem, options=options)
    f = np.asarray(result.omega).real / (2 * math.pi)
    np.testing.assert_allclose(f, exact, rtol=pc.C4_RTOL)
    assert not np.asarray(result.omega).imag.any()
    # a symmetry doublet, split by the mesh: one shared group velocity along k
    v = np.asarray(result.v_g)
    k = np.asarray(pc.C4_K)
    assert np.isfinite(v).all()
    assert np.abs(v - pc.C0 * k / np.linalg.norm(k)).max() <= 1e-3 * pc.C0


# --- C1 loaded by its ports: a damped Bloch cell ------------------------------------


def test_port_loaded_bloch_line_resonance_matches_the_closed_form(tmp_path):
    mesh = pc.c1_mesh(tmp_path)
    problem = gfem.Eigenmode(target=pc.C1_EIGEN_TARGET, wave_vector=pc.C1_K)
    options = gfem.SolverOptions(order=pc.C1_ORDER)
    result = gfem.solve(pc.c1(mesh), problem, options=options)
    f = complex(np.asarray(result.omega)[0]) / (2 * math.pi)
    # seeded at the target, not at the solve's value
    exact = pc.loaded_line_frequency(pc.C1_K[0], complex(pc.C1_EIGEN_TARGET, 2e9))
    assert abs(f - exact) <= pc.C1_EIGEN_RTOL * abs(exact)


# --- the group velocity of a damped cell ------------------------------------------


def test_band_group_velocity_equals_central_differences_on_a_damped_cell(tmp_path):
    # eps, sigma and a resistive sheet: T' is nonlinear in k and omega complex
    mesh = pc.eigen_cell_mesh(tmp_path)
    design = pc.eigen_cell_design(mesh)
    model = pc.eigen_cell(mesh, design["eps"], design["sigma"], design["fill"])
    options = gfem.SolverOptions(order=2)

    def solve(k):
        problem = gfem.Eigenmode(target=pc.EG_BLOCH_TARGET, wave_vector=k)
        return gfem.solve(model, problem, options=options)

    result = solve(pc.EG_K)
    v = np.asarray(result.v_g)[0]
    assert v[2] == 0
    delta = 1e-4 * math.pi / (pc.EG_A * pc.MM)
    for axis in (0, 1):
        plus, minus = (list(pc.EG_K), list(pc.EG_K))
        plus[axis] += delta
        minus[axis] -= delta
        fd = (
            complex(np.asarray(solve(tuple(plus)).omega)[0])
            - complex(np.asarray(solve(tuple(minus)).omega)[0])
        ) / (2 * delta)
        assert abs(v[axis] - fd) <= 1e-5 * abs(fd)
