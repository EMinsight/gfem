"""The quantity estimator's effectivity, on singular and smooth known answers.

A `gfem.QuantityEstimator` estimates how far a capacitance entry, an S entry
or a mode's omega is from its value on the same geometry with an infinitely
fine mesh. Its effectivity is the estimate divided by the error against a
known answer; every listed estimate must lie in its case's range of
`EFFECTIVITY_BY_CASE`, and `EFFECTIVITY` (the lowest low and the highest
high) is the range the docs quote. Order 2, direct solves.

Cases and references (none recomputed by the tests but the closed forms):

- square coax quarter, singular: C over `refine_mesh` cycles 0-8 against
  `refinement_cases.COAX_C` (Richardson);
- circular coax quarter, nodes on the circles, k = 1, 2, 4, 8: C against an
  order-5 solve on the same mesh (the error on its own facets); against the
  exact log formula the faceting adds about 39 % of the error, which the
  estimate does not see (asserted at most 0.95);
- step line, lumped ports, singular: S21 at 3 and 8 GHz on uniform n = 2, 4
  and over `refine_mesh` cycles 0-4 (one frequency per run) against
  `refinement_cases.STEP_S`;
- step line, wave ports: S21 at 3 GHz on n = 4 against an order-3 graded
  reference (`STEP_WAVE_S21`);
- parallel-plate line between 50 and 30 ohm lumped ports, eps_r 4: S11, S21
  and S22 at 3, 8, 15, 25 GHz on n = 4, 8, 16 cells along the line, against
  its ABCD closed form;
- L cavity, singular: f1 over `refine_mesh` cycles 0-8 against
  `refinement_cases.CAVITY_F1` (Trefethen and Betcke);
- rectangular PEC box 1 x 0.7 x 0.4 mm: modes 110, 210, 101, 310 on
  h = 0.2 / m mm, m = 2, 3, against the closed form;
- anisotropic coax (`refinement_cases.anisotropic_coax_model`), both fills,
  k = 1, 2, 4, 8: C against an order-5 solve on the same mesh;
- sheet (`refinement_cases.sheet_model`), the only case cut along an
  interior sheet: C_aa and C_bb on uniform n = 2, 4, 8, 16 and C_aa over
  `refine_mesh` cycles 0-8, against `refinement_cases.SHEET_C`;
- the parallel-plate line filled with mu_r 2 ("permeability line") or with
  diagonal and z-rotated anisotropic eps_r and mu_r ("anisotropic line"),
  `LINE_FILLS`, on n = 4 and 16 (never 8, an error crossing), against the
  TEM closed form; the fill rotated about all three axes against an order-4
  solve on the same mesh;
- the box with mu_r 3 ("permeability box") or diagonal anisotropic eps_r
  and mu_r ("anisotropic box"), `refinement_cases.ANISOTROPIC_BOX`, the same
  modes on m = 2, 3 against the closed form; the fill rotated about all three
  axes, and an eps_r 3 layer under a mu_r 2.5 layer, each of their four
  lowest modes targeted on its own on m = 2, against
  `refinement_cases.ANISOTROPIC_BOX_F` (order 4, same mesh).

Measured (ostau, 1 thread):

    square coax           1.17 - 1.57
    circular coax         1.43 - 1.47   (faceting: 0.88 - 0.89)
    step line, lumped     1.65 - 1.81
    step line, wave       1.77
    parallel-plate line   1.65 - 15.75
    L cavity              2.87 - 4.08
    rectangular box       7.05 - 10.6
    anisotropic coax      1.60 - 2.39
    sheet                 1.22 - 1.69
    permeability line     1.40 - 4.57
    anisotropic line      1.37 - 5.60   (rotated about x, y, z: 2.16 - 15.6)
    permeability box      7.05 - 10.6   (layers: 7.43 - 9.65)
    anisotropic box       5.08 - 11.1   (rotated about x, y, z: 6.05 - 9.89)
    order 3: line 2.87 - 18.1, box (m = 1) 4.69 - 5.39

The higher ratios come from the fields with no sharp metal edge (the uniform
line and the empty box), where the curl and mass terms of the error identity
nearly cancel and the estimate, which never lets them cancel, overestimates.
The line's top is at 15 GHz on n = 8, where its S error crosses near zero.
"""

import math

import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")

#: Asserted effectivity range per case (measured, with margin).
EFFECTIVITY_BY_CASE = {
    "square coax": (0.9, 2.0),
    "circular coax": (0.9, 2.0),
    "step line, lumped ports": (0.9, 2.5),
    "step line, wave ports": (0.9, 2.5),
    "parallel-plate line": (0.9, 17.0),
    "L cavity": (0.9, 5.0),
    "rectangular box": (0.9, 12.0),
    "anisotropic coax": (0.9, 2.5),
    "sheet": (0.9, 2.5),
    "permeability line": (0.9, 17.0),
    "anisotropic line": (0.9, 17.0),
    "permeability box": (0.9, 12.0),
    "anisotropic box": (0.9, 12.0),
}

#: The range the docs quote: the lowest low and the highest high.
EFFECTIVITY = (
    min(low for low, _ in EFFECTIVITY_BY_CASE.values()),
    max(high for _, high in EFFECTIVITY_BY_CASE.values()),
)


def _assert_in(case, estimate, error):
    low, high = EFFECTIVITY_BY_CASE[case]
    assert low <= estimate / abs(error) <= high, (case, estimate / abs(error))


def test_the_estimate_leaves_out_the_linear_solve_so_a_tight_tolerance_raises(
    tmp_path,
):
    # The estimate reads the discrete field as if solved exactly: refine_mesh
    # refuses a tolerance below 10 x SolverOptions.tolerance (1e-6 default).
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    with pytest.raises(ValueError, match="tolerance"):
        gfem.refine_mesh(
            rc.coax_model(tmp_path, 2),
            rc.COAX_PROBLEM,
            estimator=estimator,
            tolerance=5e-6,
        )


# --- square coax: capacitance, singular --------------------------------------


def test_the_square_coax_capacitance(tmp_path):
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            rc.coax_model(tmp_path, 2),
            rc.COAX_PROBLEM,
            estimator=estimator,
            tolerance=1e-5,
            max_cycles=8,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 9
    scale = const.epsilon_0 * 0.5e-6 / 4
    for cycle in ref.cycles:
        error = (cycle.value / scale - rc.COAX_C) / rc.COAX_C
        assert error > 0  # C_h >= C on the meshed geometry
        _assert_in("square coax", cycle.estimate, error)


# --- circular coax: capacitance, smooth, faceted ------------------------------

#: The exact capacitance of the quarter circular coax (a = 1, b = 2, 0.5 long, um).
CIRCULAR_C = math.pi * const.epsilon_0 * 0.5e-6 / (2 * math.log(2.0))


def _circular_coax(directory, k):
    path = sm.write_msh22(
        directory / "coax.msh", sm.coax_quarter(1.0, 2.0, 0.5, n_r=2 * k, n_theta=4 * k)
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    return gfem.Model(mesh, materials={"dielectric": gfem.Material()})


@pytest.fixture(scope="module")
def circular_coax(tmp_path_factory):
    """Per k: order-2 C, its estimate, and order-5 C on the same mesh."""
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    out = {}
    for k in (1, 2, 4, 8):
        model = _circular_coax(tmp_path_factory.mktemp(f"coax{k}"), k)
        result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
        estimate = gfem.error_estimate(result, estimator, model=model)
        fine = gfem.solve(
            model,
            rc.COAX_PROBLEM,
            options=gfem.SolverOptions(order=5, linear_solver="direct"),
        )
        out[k] = (
            float(result.capacitance[0, 0]),
            estimate.value,
            float(fine.capacitance[0, 0]),
        )
    return out


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_circular_coax_capacitance_against_its_own_facets(circular_coax, k):
    c, estimate, c5 = circular_coax[k]
    _assert_in("circular coax", estimate, (c - c5) / c5)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_the_estimate_does_not_see_the_faceting(circular_coax, k):
    # Against the true circle the error includes the faceting, which mesh
    # refinement never removes; the docs say the estimate leaves it out.
    c, estimate, _ = circular_coax[k]
    assert estimate / abs((c - CIRCULAR_C) / CIRCULAR_C) <= 0.95


# --- anisotropic coax: capacitance, a rotated tensor -------------------------


@pytest.mark.parametrize("fill", list(rc.ANISOTROPIC_COAX))
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_anisotropic_coax_capacitance(tmp_path, k, fill):
    model = rc.anisotropic_coax_model(tmp_path, k, fill)
    result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    estimate = gfem.error_estimate(result, estimator, model=model).value
    fine = gfem.solve(
        model,
        rc.COAX_PROBLEM,
        options=gfem.SolverOptions(order=5, linear_solver="direct"),
    )
    c, c5 = float(result.capacitance[0, 0]), float(fine.capacitance[0, 0])
    _assert_in("anisotropic coax", estimate, (c - c5) / c5)


# --- sheet: capacitance, cut along an interior sheet --------------------------


def _sheet_error(capacitance, i):
    return (float(np.real(capacitance)) - rc.SHEET_C[i][i]) / rc.SHEET_C[i][i]


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_sheet_capacitance_on_uniform_meshes(tmp_path, n):
    model = rc.sheet_model(tmp_path, n)
    result = gfem.solve(model, rc.SHEET_PROBLEM, options=OPTIONS)
    for i, terminal in enumerate(("a", "b")):
        estimator = gfem.QuantityEstimator(capacitance=(terminal, terminal))
        estimate = gfem.error_estimate(result, estimator, model=model)
        error = _sheet_error(result.capacitance[i, i], i)
        assert error > 0  # C_h >= C
        _assert_in("sheet", estimate.value, error)


def test_the_sheet_capacitance_under_mesh_refinement(tmp_path):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            rc.sheet_model(tmp_path, 2),
            rc.SHEET_PROBLEM,
            estimator=gfem.QuantityEstimator(capacitance=("a", "a")),
            tolerance=1e-4,
            max_cycles=8,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 9
    for cycle in ref.cycles:
        _assert_in("sheet", cycle.estimate, _sheet_error(cycle.value, 0))


# --- step line: S21, singular ------------------------------------------------

S21 = gfem.QuantityEstimator(s=("p2", "p1"))


def _step_errors(s21_by_frequency, estimates):
    for f, s21, estimate in zip(rc.STEP_S, s21_by_frequency, estimates, strict=True):
        _assert_in("step line, lumped ports", estimate, s21 - rc.STEP_S[f]["S21"])


@pytest.mark.parametrize("n", [2, 4])
def test_the_step_line_s21_on_uniform_meshes(tmp_path, n):
    model = rc.step_model(tmp_path, n)
    problem = gfem.Driven(list(rc.STEP_S), keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, S21, model=model)
    assert len(estimate.values) == len(rc.STEP_S)
    _step_errors(np.asarray(result.s["p2", "p1"]), estimate.values)


@pytest.mark.parametrize("frequency", list(rc.STEP_S))
def test_the_step_line_s21_under_mesh_refinement(tmp_path, frequency):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            rc.step_model(tmp_path, 2),
            rc.step_problem([frequency]),
            estimator=S21,
            tolerance=1e-5,
            max_cycles=4,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 5
    for cycle in ref.cycles:
        (s21,) = cycle.value
        error = s21 - rc.STEP_S[frequency]["S21"]
        _assert_in("step line, lumped ports", cycle.estimate, error)


#: The step line's S21 at 3 GHz with wave ports: order 3 on the n = 16 mesh
#: graded 3 towards the edge, 1.3e-6 from the n = 8 one.
STEP_WAVE_S21 = 0.2811336 - 0.8998640j


def test_the_step_line_s21_with_wave_ports(tmp_path):
    lumped = rc.step_model(tmp_path, 4)
    model = gfem.Model(
        lumped.mesh,
        materials=lumped.materials,
        boundaries=lumped.boundaries,
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    problem = gfem.Driven([3e9], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, S21, model=model)
    error = complex(result.s["p2", "p1"][0]) - STEP_WAVE_S21
    _assert_in("step line, wave ports", estimate.value, error)


# --- parallel-plate line: S, smooth ------------------------------------------

_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3, "eps_r": 4.0}
_R1, _R2 = 50.0, 30.0
LINE_FREQUENCIES = (3e9, 8e9, 15e9, 25e9)


def _line_s(f, eps_r=_LINE["eps_r"], mu_r=1.0):
    """The line's S matrix between 50 and 30 ohm ports, from its ABCD matrix.

    ``eps_r`` and ``mu_r`` are the fill's entries the TEM wave sees: eps_r
    along the plates' normal z, mu_r along the width y.
    """
    beta = 2 * math.pi * f / const.c * math.sqrt(eps_r * mu_r)
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * math.sqrt(mu_r / eps_r) * _LINE["height"] / _LINE["width"]
    c, s = math.cos(beta * _LINE["length"]), math.sin(beta * _LINE["length"])
    a, b, cc, d = c, 1j * z0 * s, 1j * s / z0, c
    den = a * _R2 + b + cc * _R1 * _R2 + d * _R1
    k = math.sqrt(_R1 * _R2)
    s11 = (a * _R2 + b - cc * _R1 * _R2 - d * _R1) / den
    s22 = (-a * _R2 + b - cc * _R1 * _R2 + d * _R1) / den
    return np.array([[s11, 2 * k * (a * d - b * cc) / den], [2 * k / den, s22]])


def _line(directory, n, order=2, material=None):
    path = sm.write_msh22(
        directory / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    material = material or gfem.Material(eps_r=_LINE["eps_r"])
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": material},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    problem = gfem.Driven(list(LINE_FREQUENCIES), keep_fields=True)
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return model, gfem.solve(model, problem, options=options)


def _line_effectivities(model, result, exact=None):
    """(estimate, error) of S11, S21 and S22 at each frequency.

    ``exact``: the reference S, shape (frequencies, 2, 2); by default the
    closed form of the eps_r 4 line.
    """
    out = []
    if exact is None:
        exact = np.array([_line_s(f) for f in LINE_FREQUENCIES])
    ports = ("p1", "p2")
    for to, source in (("p1", "p1"), ("p2", "p1"), ("p2", "p2")):
        estimator = gfem.QuantityEstimator(s=(to, source))
        estimate = gfem.error_estimate(result, estimator, model=model)
        reference = exact[:, ports.index(to), ports.index(source)]
        error = np.asarray(result.s[to, source]) - reference
        out += list(zip(estimate.values, error, strict=True))
    return out


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_parallel_plate_line_s(tmp_path, n):
    for estimate, error in _line_effectivities(*_line(tmp_path, n)):
        _assert_in("parallel-plate line", estimate, error)


# --- magnetic and anisotropic lines: S, smooth --------------------------------

#: The line's magnetic and anisotropic fills: (material, eps_zz, mu_yy), the
#: entries its TEM wave sees, or None where the fill has no closed form.
#: "rotz" turns eps_r about z, the plates' normal, and keeps mu_r isotropic in
#: the plates' plane: a turned in-plane mu_r would put H_x on the open sides,
#: and the wave would no longer be TEM. "rot3" turns both about all three axes.
LINE_FILLS = {
    "mu": (gfem.Material(eps_r=2.0, mu_r=2.0), 2.0, 2.0),
    "diag": (gfem.Material(eps_r=(3.0, 5.0, 2.0), mu_r=(1.5, 2.0, 2.5)), 2.0, 2.0),
    "rotz": (
        gfem.Material(
            eps_r=(3.0, 5.0, 2.0),
            mu_r=(2.0, 2.0, 2.5),
            axes=tuple(map(tuple, rc.rotation("z", 30).T)),
        ),
        2.0,
        2.0,
    ),
    "rot3": (
        gfem.Material(
            eps_r=(3.0, 5.0, 2.0),
            mu_r=(1.5, 2.5, 2.0),
            axes=tuple(map(tuple, rc.ROT3.T)),
        ),
        None,
        None,
    ),
}


# n = 8 is left out: there the error of S crosses near zero at 8 and 15 GHz
# (effectivity up to 165), a property of the uniform line, not of its fill.
@pytest.mark.parametrize("n", [4, 16])
def test_the_permeability_line_s(tmp_path, n):
    material, eps, mu = LINE_FILLS["mu"]
    exact = np.array([_line_s(f, eps, mu) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, error in _line_effectivities(model, result, exact):
        _assert_in("permeability line", estimate, error)


@pytest.mark.parametrize("fill", ["diag", "rotz"])
@pytest.mark.parametrize("n", [4, 16])
def test_the_anisotropic_line_s(tmp_path, n, fill):
    material, eps, mu = LINE_FILLS[fill]
    exact = np.array([_line_s(f, eps, mu) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, error in _line_effectivities(model, result, exact):
        _assert_in("anisotropic line", estimate, error)


@pytest.mark.parametrize("n", [4, 16])
def test_the_rotated_anisotropic_line_s_against_order_4(tmp_path, n):
    # No closed form: the reference is order 4 on the same mesh, whose own
    # error (|S4 - S6|) is 5 to 10 % of |S2 - S4| on n = 4 and 16.
    material, _, _ = LINE_FILLS["rot3"]
    model, result = _line(tmp_path, n, material=material)
    fine = gfem.solve(
        model,
        gfem.Driven(list(LINE_FREQUENCIES)),
        options=gfem.SolverOptions(order=4, linear_solver="direct"),
    )
    exact = np.asarray(fine.s.values)  # (frequencies, to, source), p1 then p2
    for estimate, error in _line_effectivities(model, result, exact):
        _assert_in("anisotropic line", estimate, error)


# --- L cavity: omega, singular -----------------------------------------------


def test_the_l_cavity_frequency_under_mesh_refinement(tmp_path):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            rc.cavity_model(tmp_path, 2),
            rc.CAVITY_PROBLEM,
            estimator=gfem.QuantityEstimator(omega=0),
            tolerance=1e-7,
            max_cycles=8,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 9
    for cycle in ref.cycles:
        f = cycle.value.real / (2 * math.pi)
        _assert_in("L cavity", cycle.estimate, f / rc.CAVITY_F1 - 1)


# --- rectangular box: omega, smooth ------------------------------------------

BOX = (1e-3, 0.7e-3, 0.4e-3)
BOX_MODES = ((1, 1, 0), (2, 1, 0), (1, 0, 1), (3, 1, 0))


def _box(directory, m, order=2):
    """Per mode of `BOX_MODES`: (frequency, its estimate, the exact one)."""
    cells = sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m))
    path = sm.write_msh22(directory / "box.msh", cells)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    estimator = gfem.QuantityEstimator(omega=0)
    out = []
    for mode in BOX_MODES:
        k = math.pi * math.sqrt(
            sum((i / a) ** 2 for i, a in zip(mode, BOX, strict=True))
        )
        exact = const.c * k / (2 * math.pi)
        problem = gfem.Eigenmode(target=exact, n_modes=1)
        result = gfem.solve(model, problem, options=options)
        estimate = gfem.error_estimate(result, estimator, model=model)
        f = float(np.real(np.asarray(result.frequency)[0]))
        out.append((f, estimate.value, exact))
    return out


@pytest.mark.parametrize("m", [2, 3])
def test_the_rectangular_box_frequencies(tmp_path, m):
    for f, estimate, exact in _box(tmp_path, m):
        _assert_in("rectangular box", estimate, f / exact - 1)


# --- magnetic and anisotropic boxes: omega, smooth -----------------------------


def _diagonal_box_frequency(fill, mode):
    """The closed-form frequency (Hz) of a mode with one zero index.

    With eps_r and mu_r diagonal and E along the axis d of the zero index,
    k0^2 eps_dd = sum over i != d of (n_i pi / L_i)^2 / mu_jj, j the third axis.
    """
    material = rc.ANISOTROPIC_BOX[fill]["air"]
    eps = np.broadcast_to(np.asarray(material.eps_r, float), (3,))
    mu = np.broadcast_to(np.asarray(material.mu_r, float), (3,))
    d = mode.index(0)
    k2 = sum(
        (mode[i] * math.pi / BOX[i]) ** 2 / mu[3 - i - d] for i in range(3) if i != d
    )
    return const.c * math.sqrt(k2 / eps[d]) / (2 * math.pi)


def _filled_box_mode(model, target):
    """(frequency, its estimate) of the mode nearest ``target``, order 2."""
    result = gfem.solve(
        model, gfem.Eigenmode(target=target, n_modes=1), options=OPTIONS
    )
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    return float(np.real(np.asarray(result.frequency)[0])), estimate.value


@pytest.mark.parametrize("mode", BOX_MODES)
@pytest.mark.parametrize("m", [2, 3])
def test_the_permeability_box_frequency(tmp_path, m, mode):
    exact = _diagonal_box_frequency("iso_mu", mode)
    model = rc.anisotropic_box_model(tmp_path, m, "iso_mu")
    f, estimate = _filled_box_mode(model, exact)
    _assert_in("permeability box", estimate, f / exact - 1)


@pytest.mark.parametrize("mode", BOX_MODES)
@pytest.mark.parametrize("m", [2, 3])
def test_the_anisotropic_box_frequency(tmp_path, m, mode):
    exact = _diagonal_box_frequency("diag", mode)
    model = rc.anisotropic_box_model(tmp_path, m, "diag")
    f, estimate = _filled_box_mode(model, exact)
    _assert_in("anisotropic box", estimate, f / exact - 1)


# The rotated fill and the two layers have no closed form: each of their four
# lowest modes is targeted at its stored order-4 frequency on the same mesh.
@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize(
    ("fill", "case"),
    [("rot3", "anisotropic box"), ("layer_mu", "permeability box")],
)
def test_the_rotated_anisotropic_and_layered_box_frequency(tmp_path, fill, case, mode):
    reference = rc.ANISOTROPIC_BOX_F[fill][mode]
    model = rc.anisotropic_box_model(tmp_path, 2, fill)
    f, estimate = _filled_box_mode(model, reference)
    _assert_in(case, estimate, f / reference - 1)


# --- order 3 -----------------------------------------------------------------


# Expected out of range: at order 3 the parallel-plate line measured 2.9 to
# 18.1 (n = 4, 8; the top at 8 GHz, n = 8), above `EFFECTIVITY`, so the docs
# claim the range for order 2 only. These ranges are not part of
# `EFFECTIVITY`; they keep the order-3 behaviour from drifting unseen.
#: Asserted order-3 effectivity range per case (measured, with margin).
ORDER_3_EFFECTIVITY_BY_CASE = {
    "parallel-plate line": (0.9, 25.0),  # measured 2.87-18.1
    "rectangular box": (0.9, 8.0),  # measured 4.69-5.39 (m = 1)
}


def _assert_order_3(case, estimate, error):
    low, high = ORDER_3_EFFECTIVITY_BY_CASE[case]
    assert low <= estimate / abs(error) <= high, (case, estimate / abs(error))


@pytest.mark.parametrize("n", [4, 8])
def test_the_parallel_plate_line_s_at_order_3(tmp_path, n):
    for estimate, error in _line_effectivities(*_line(tmp_path, n, order=3)):
        _assert_order_3("parallel-plate line", estimate, error)


def test_the_rectangular_box_frequencies_at_order_3(tmp_path):
    for f, estimate, exact in _box(tmp_path, 1, order=3):
        _assert_order_3("rectangular box", estimate, f / exact - 1)
