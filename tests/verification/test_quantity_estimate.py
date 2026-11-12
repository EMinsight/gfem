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
- step line, wave ports de-embedded by 1 and 2 mm: the same S21 times the
  exact TEM phase of the offsets (the estimate is the one without offsets);
- block guide (`refinement_cases.block_guide_model`), TE10 and TE20 on each
  of two shared port surfaces: six S entries at 170 GHz on n = 4, 8 against
  `refinement_cases.BLOCK_GUIDE_S` (order 4 on n = 4); n = 2, under two
  cells per wavelength in the block, is pre-asymptotic and not run;
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
- sheet, off-diagonal: C_ab on uniform n = 2, 4, 8, 16 against
  `refinement_cases.SHEET_C`;
- two bars (`refinement_cases.bars_model`), off-diagonal C_ab against
  `refinement_cases.BARS_C_AB` (Richardson): "screen" (a grounded fin
  between the bars, where the entry's terms change sign) over
  `refine_mesh` cycles 0-8, "near" on uniform n = 2, 4, 8.

Measured (ostau, 1 thread):

    square coax           1.17 - 1.57
    circular coax         1.43 - 1.47   (faceting: 0.88 - 0.89)
    step line, lumped     1.65 - 1.81
    step line, wave       1.77
    block guide           3.34 - 8.58
    parallel-plate line   1.65 - 15.75
    L cavity              2.87 - 4.08
    rectangular box       7.05 - 10.6
    anisotropic coax      1.60 - 2.39
    sheet                 1.22 - 1.69
    sheet, C_ab           1.79 - 2.26
    two bars, screen      3.70 - 9.55
    two bars, near        1.66 - 1.71
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
from gfem._engine import estimate as kernels

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")

#: Asserted effectivity range per case (measured, with margin).
EFFECTIVITY_BY_CASE = {
    "square coax": (0.9, 2.0),
    "circular coax": (0.9, 2.0),
    "step line, lumped ports": (0.9, 2.5),
    "step line, wave ports": (0.9, 2.5),
    "shared port surface": (0.9, 12.0),
    "parallel-plate line": (0.9, 17.0),
    "L cavity": (0.9, 5.0),
    "rectangular box": (0.9, 12.0),
    "anisotropic coax": (0.9, 2.5),
    "sheet": (0.9, 2.5),
    "screen": (0.9, 12.0),
    "near": (0.9, 2.0),
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
    # QuantityEstimator still refuses an anisotropic permittivity: the
    # estimate is the kernel's, as error_estimate would report it.
    model = rc.anisotropic_coax_model(tmp_path, k, fill)
    result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
    estimate, _ = kernels.capacitance(model, result, "inner", "inner")
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


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_sheet_off_diagonal_capacitance_on_uniform_meshes(tmp_path, n):
    model = rc.sheet_model(tmp_path, n)
    result = gfem.solve(model, rc.SHEET_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    reference = rc.SHEET_C[0][1]
    error = (float(np.real(result.capacitance[0, 1])) - reference) / reference
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


# --- two bars: off-diagonal capacitance --------------------------------------


def _bars_error(case, capacitance):
    reference = rc.BARS_C_AB[case]
    return (float(np.real(capacitance)) - reference) / reference


def test_the_screened_off_diagonal_capacitance_under_mesh_refinement(tmp_path):
    # Two bars partly screened by a grounded fin: the entry's terms change
    # sign around the fin, and the estimate never lets them cancel.
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            rc.bars_model(tmp_path, "screen", 2),
            rc.BARS_PROBLEM,
            estimator=gfem.QuantityEstimator(capacitance=("a", "b")),
            tolerance=1e-4,
            max_cycles=8,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 9
    for cycle in ref.cycles:
        _assert_in("screen", cycle.estimate, _bars_error("screen", cycle.value))


@pytest.mark.parametrize("n", [2, 4, 8])
def test_the_near_off_diagonal_capacitance_on_uniform_meshes(tmp_path, n):
    model = rc.bars_model(tmp_path, "near", n)
    result = gfem.solve(model, rc.BARS_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    _assert_in("near", estimate.value, _bars_error("near", result.capacitance[0, 1]))


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


def test_the_step_line_s21_with_wave_ports_at_an_offset(tmp_path):
    # The offsets move the reference planes 1 and 2 mm into the line: the
    # TEM port's beta is exact, so the reference is STEP_WAVE_S21 times
    # exp(i beta (1 + 2) mm).
    offsets = (1e-3, 2e-3)
    lumped = rc.step_model(tmp_path, 4)
    model = gfem.Model(
        lumped.mesh,
        materials=lumped.materials,
        boundaries=lumped.boundaries,
        ports=[
            gfem.WavePort("p1", "port1", offset=offsets[0]),
            gfem.WavePort("p2", "port2", offset=offsets[1]),
        ],
    )
    problem = gfem.Driven([3e9], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, S21, model=model)
    beta = 2 * math.pi * 3e9 / const.c * math.sqrt(4.0)
    reference = STEP_WAVE_S21 * np.exp(1j * beta * sum(offsets))
    error = complex(result.s["p2", "p1"][0]) - reference
    _assert_in("step line, wave ports", estimate.value, error)


# --- block guide: S, two modes on each shared port surface -------------------


@pytest.mark.parametrize("n", [4, 8])
def test_the_block_guide_s_on_shared_port_surfaces(tmp_path, n):
    model = rc.block_guide_model(tmp_path, n)
    problem = gfem.Driven([rc.BLOCK_GUIDE_FREQUENCY], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    for (to, source), reference in rc.BLOCK_GUIDE_S.items():
        estimator = gfem.QuantityEstimator(s=(to, source))
        estimate = gfem.error_estimate(result, estimator, model=model)
        error = complex(result.s[to, source][0]) - reference
        _assert_in("shared port surface", estimate.value, error)


# --- parallel-plate line: S, smooth ------------------------------------------

_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3, "eps_r": 4.0}
_R1, _R2 = 50.0, 30.0
LINE_FREQUENCIES = (3e9, 8e9, 15e9, 25e9)


def _line_s(f):
    """The line's S matrix between 50 and 30 ohm ports, from its ABCD matrix."""
    beta = 2 * math.pi * f / const.c * math.sqrt(_LINE["eps_r"])
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * _LINE["height"] / (_LINE["width"] * math.sqrt(_LINE["eps_r"]))
    c, s = math.cos(beta * _LINE["length"]), math.sin(beta * _LINE["length"])
    a, b, cc, d = c, 1j * z0 * s, 1j * s / z0, c
    den = a * _R2 + b + cc * _R1 * _R2 + d * _R1
    k = math.sqrt(_R1 * _R2)
    s11 = (a * _R2 + b - cc * _R1 * _R2 - d * _R1) / den
    s22 = (-a * _R2 + b - cc * _R1 * _R2 + d * _R1) / den
    return np.array([[s11, 2 * k * (a * d - b * cc) / den], [2 * k / den, s22]])


def _line(directory, n, order=2):
    path = sm.write_msh22(
        directory / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=_LINE["eps_r"])},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    problem = gfem.Driven(list(LINE_FREQUENCIES), keep_fields=True)
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return model, gfem.solve(model, problem, options=options)


def _line_effectivities(model, result):
    """(estimate, error) of S11, S21 and S22 at each frequency."""
    out = []
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


# --- order 3 -----------------------------------------------------------------


# A drift fence, not a claim: the docs claim `EFFECTIVITY` at order 2 only,
# and these ranges are not part of it. At order 3 only the parallel-plate
# line leaves the order-2 range, at 18.1 on n = 8 (S22, 8 GHz), where order 2
# also peaks (14.7): the same smooth-field overestimation as at order 2,
# curl and mass terms the estimate never lets cancel, not an order-3 effect.
# The box at order 3 lies inside its order-2 range. Recovering into degree p
# instead of p - 1 (underestimates singular cases, down to 0.51) and a finer
# quadrature (identical to every printed digit) were measured and rejected.
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
