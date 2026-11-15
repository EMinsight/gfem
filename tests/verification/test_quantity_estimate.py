"""The quantity estimator's effectivity, on singular and smooth known answers.

A `gfem.QuantityEstimator` estimates how far a capacitance entry, an S entry
or a mode's omega is from its value on the same geometry with an infinitely
fine mesh. Its effectivity is the estimate divided by the error against a
known answer; every listed estimate must lie in its case's range of
`EFFECTIVITY_BY_KIND`, keyed by the estimate's kind (`ErrorEstimate.kind`),
and each kind's span is the range the docs quote for it. Order 2, direct
solves.

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
- lossy guide (`refinement_cases.lossy_guide_model`), tan_delta 0.05 and
  0.3, wave ports de-embedded by 0 and 1 mm: all four S entries at 170 GHz
  on n = 4, 8 against `refinement_cases.LOSSY_GUIDE_S` (order 4 on n = 4),
  the slab's and step's tan_delta 0.3 thru entries on n = 8 left out (their
  reference is not within 20 % of their error). "uniform" (TE10, parallel:
  the free dual), "slab" and "step" (a hybrid lossy port mode: the adjoint
  of port ``to``'s S functional);
- parallel-plate line between 50 and 30 ohm lumped ports, eps_r 4: S11, S21
  and S22 at 3, 8, 15, 25 GHz on n = 4, 8, 16 cells along the line, against
  its ABCD closed form; the same with a lossy fill (tan_delta 0.3, or
  0.1 S/m), against the closed form with complex eps_r ("lossy line");
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
  `refinement_cases.ANISOTROPIC_BOX_F` (order 4, same mesh);
- sheet, off-diagonal: C_ab on uniform n = 2, 4, 8, 16 against
  `refinement_cases.SHEET_C`;
- two bars (`refinement_cases.bars_model`), off-diagonal C_ab against
  `refinement_cases.BARS_C_AB` (Richardson): "screen" (a grounded fin
  between the bars, where the entry's terms change sign) over
  `refine_mesh` cycles 0-8, "near" on uniform n = 2, 4, 8;
- lossy box: the rectangular box filled with eps_r 4, tan_delta 0.1 and
  5 S/m (as a `Material` and as a `MaterialField` with complex eps_r), its
  four modes on m = 2, against the fill's closed form (complex omega);
- lossy L cavity: the L cavity filled with eps_r 4, tan_delta 0.1, complex
  omega over `refine_mesh` cycles 0-7 against the exact scaling of
  `refinement_cases.CAVITY_LAMBDA1`;
- lossy walls: the air box with an impedance (377 ohm/sq), first-order
  absorbing or conductor (1e5 S/m) top wall, mode 110 on m = 2, 3, against
  `refinement_cases.LOSSY_WALL_OMEGA` (Richardson);
- resistive line: the parallel-plate line closed by its 50 and 30 ohm
  lumped ports as a resonator, first two resonances on n = 4, 8, 16, 32,
  against its transmission-line closed form.

The eigen terms ADR 0138 lifted, and the lossy magnetic and anisotropic
fills of the line:

- dispersive box: the rectangular box filled with a Drude (lossy and
  lossless) or Lorentz law at Re eps > 0, modes 110 and 101 on m = 2,
  against the root of f^2 eps(f) = K^2;
- layered box: a 0.2 mm Drude dielectric layer, or that law frozen at
  217.4 GHz, under air, three modes on m = 2, against
  `refinement_cases.LAYERED_BOX_OMEGA` (order 4, same mesh);
- second-order absorbing walls: the air box with `Absorbing(order=2)` side
  walls, the modes continuing 110, 210 and 120 on m = 1, 2, against
  `refinement_cases.ABSORBING_BOX_OMEGA` (order 4, same mesh; six modes
  solved, the nearest the reference estimated);
- lossy permeability box: eps_r 2, mu_r 2 (1 - 0.1i), modes 110 and 101 on
  m = 2, against omega = c k0 / sqrt(eps mu); lossy permeability layers:
  eps_r 3 under mu_r 2.5 (1 - 0.2i), three modes on m = 2, against
  `refinement_cases.LOSSY_BOX_OMEGA`;
- lossy anisotropic box ("anisotropic box"): the diagonal fill with
  tan_delta 0.1, conductivity (5, 10, 20) S/m or per-axis complex eps_r,
  against the closed form of the nearest mode with one zero index, on m = 1
  (110, 101) and m = 2 (all four targets); the per-axis loss along `ROT3`
  against `refinement_cases.LOSSY_BOX_OMEGA`;
- lossy permeability line and lossy anisotropic line: the parallel-plate
  line filled with eps_r 2 and mu_r 2 (1 - 0.1i), the diagonal fill with
  tan_delta 0.1, or per-axis complex eps_r turned about z, against the TEM
  closed form on n = 4 and 16; the per-axis loss turned about all three
  axes against order 4 on the same mesh (rows whose order-3 to order-4 gap
  exceeds 20 % of their error left out);
- not claimed: the plasmon-bound mode of a Drude floor (Re eps -1.73), which
  raises, measured by the private kernel against its LSM closed form
  (`PLASMON_WITNESS`).

Measured (ostau, 1 thread):

    square coax           1.17 - 1.57
    circular coax         1.43 - 1.47   (faceting: 0.88 - 0.89)
    step line, lumped     1.65 - 1.81
    step line, wave       1.77
    block guide           3.34 - 8.58
    lossy uniform guide   2.24 - 14.0
    lossy slab guide      1.10 - 14.0
    lossy step guide      1.19 - 17.1
    parallel-plate line   1.65 - 15.75
    lossy line            1.38 - 15.7   (tan_delta 1.38 - 15.5, sigma 1.56 - 15.7)
    L cavity              2.87 - 4.08
    rectangular box       7.05 - 10.6
    anisotropic coax      1.60 - 2.39
    sheet                 1.22 - 1.69
    permeability line     1.40 - 4.57
    anisotropic line      1.37 - 5.60   (rotated about x, y, z: 2.16 - 15.6)
    permeability box      7.05 - 10.6   (layers: 7.43 - 9.65)
    anisotropic box       5.08 - 11.1   (rotated about x, y, z: 6.05 - 9.89)
    sheet, C_ab           1.79 - 2.26
    two bars, screen      3.70 - 9.55
    two bars, near        1.66 - 1.71
    lossy box             7.18 - 10.6   (both fills)
    lossy L cavity        2.87 - 4.08
    lossy walls           4.34 - 9.17
    resistive line        1.47 - 7.44
    dispersive box        7.18 - 8.63
    layered box           6.14 - 21.2   (law 7.50 - 20.5, constant 6.14 - 21.2)
    second-order absorbing walls  1.17 - 4.18
    lossy permeability box  7.18 - 8.63  (layers: 7.25 - 8.39)
    lossy anisotropic box 4.73 - 13.9   (rotated about x, y, z: 5.85 - 8.36)
    lossy permeability line  1.46 - 4.56
    lossy anisotropic line   1.39 - 7.03  (rotated about x, y, z: 2.23 - 3.14)
    plasmon-bound mode (refused)  0.96
    order 3: line 3.10 - 19.6, box (m = 1) 4.69 - 5.39
    order 4: L cavity (n = 2, 4) 1.80, 1.73

The higher ratios come from the fields with no sharp metal edge (the uniform
line and the empty box), where the curl and mass terms of the error identity
nearly cancel and the estimate, which never lets them cancel, overestimates.
The line's top is at 15 GHz on n = 8, where its S error crosses near zero.
Three ways to cut that overestimation were measured and rejected, as each
underestimates somewhere or rests on a constant fitted to one case: the
signed sum floored at a fraction alpha of the magnitude, sums per region,
and a degree-p recovery.
"""

import cmath
import math

import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")

#: Asserted effectivity at order 2, per estimate kind (`ErrorEstimate.kind`),
#: then per case (measured, with margin). The docs quote each kind's span, and
#: the span of the three entry kinds as the quantity estimator's range.
EFFECTIVITY_BY_KIND = {
    "capacitance": {
        "square coax": (0.9, 2.0),
        "circular coax": (0.9, 2.0),
        "anisotropic coax": (0.9, 2.5),
        "sheet": (0.9, 2.5),
        "screen": (0.9, 12.0),
        "near": (0.9, 2.0),
    },
    "s": {
        "step line, lumped ports": (0.9, 2.5),
        "step line, wave ports": (0.9, 2.5),
        "shared port surface": (0.9, 12.0),
        "parallel-plate line": (0.9, 17.0),
        "permeability line": (0.9, 17.0),
        "anisotropic line": (0.9, 17.0),
        "lossy uniform guide": (0.9, 20.0),
        "lossy slab guide": (0.9, 20.0),
        "lossy step guide": (0.9, 20.0),
        "lossy permeability line": (0.9, 6.0),
        "lossy anisotropic line": (0.9, 9.0),
    },
    "omega": {
        "L cavity": (0.9, 5.0),
        "rectangular box": (0.9, 12.0),
        "permeability box": (0.9, 12.0),
        "anisotropic box": (0.9, 15.0),
        "lossy box": (0.9, 12.0),
        "lossy L cavity": (0.9, 5.0),
        "lossy walls": (0.9, 12.0),
        "resistive line": (0.9, 8.0),
        "dispersive box": (0.9, 12.0),
        "layered box": (0.9, 24.0),
        "second-order absorbing walls": (0.9, 5.0),
        "lossy permeability box": (0.9, 12.0),
        "lossy permeability layers": (0.9, 12.0),
    },
}

#: Cases asserted >= 0.9 whose high end the docs quote on their own (Limits)
#: instead of widening their kind's range: name -> (kind, (low, high)).
OVERESTIMATES: dict[str, tuple[str, tuple[float, float]]] = {}

#: Configurations that are not claimed, measured bands that reach below 0.9:
#: name -> (kind, (low, high)).
WITNESSES: dict[str, tuple[str, tuple[float, float]]] = {}


def _assert_in(kind, case, estimate, error, value=None):
    """``value`` (default ``estimate.value``) over ``|error|`` in the case's range.

    ``estimate`` is the `gfem.ErrorEstimate` the value comes from: its kind
    must be ``kind``, the key of the case's range.
    """
    assert estimate.kind == kind, (case, estimate.kind)
    low, high = EFFECTIVITY_BY_KIND[kind][case]
    value = estimate.value if value is None else value
    assert low <= value / abs(error) <= high, (case, value / abs(error))


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
        _assert_in("capacitance", "square coax", ref.estimate, error, cycle.estimate)


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
            estimate,
            float(fine.capacitance[0, 0]),
        )
    return out


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_circular_coax_capacitance_against_its_own_facets(circular_coax, k):
    c, estimate, c5 = circular_coax[k]
    _assert_in("capacitance", "circular coax", estimate, (c - c5) / c5)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_the_estimate_does_not_see_the_faceting(circular_coax, k):
    # Against the true circle the error includes the faceting, which mesh
    # refinement never removes; the docs say the estimate leaves it out.
    c, estimate, _ = circular_coax[k]
    assert estimate.value / abs((c - CIRCULAR_C) / CIRCULAR_C) <= 0.95


# --- anisotropic coax: capacitance, a rotated tensor -------------------------


@pytest.mark.parametrize("fill", list(rc.ANISOTROPIC_COAX))
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_anisotropic_coax_capacitance(tmp_path, k, fill):
    model = rc.anisotropic_coax_model(tmp_path, k, fill)
    result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("inner", "inner"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    fine = gfem.solve(
        model,
        rc.COAX_PROBLEM,
        options=gfem.SolverOptions(order=5, linear_solver="direct"),
    )
    c, c5 = float(result.capacitance[0, 0]), float(fine.capacitance[0, 0])
    _assert_in("capacitance", "anisotropic coax", estimate, (c - c5) / c5)


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
        _assert_in("capacitance", "sheet", estimate, error)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_sheet_off_diagonal_capacitance_on_uniform_meshes(tmp_path, n):
    model = rc.sheet_model(tmp_path, n)
    result = gfem.solve(model, rc.SHEET_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    reference = rc.SHEET_C[0][1]
    error = (float(np.real(result.capacitance[0, 1])) - reference) / reference
    _assert_in("capacitance", "sheet", estimate, error)


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
        error = _sheet_error(cycle.value, 0)
        _assert_in("capacitance", "sheet", ref.estimate, error, cycle.estimate)


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
        error = _bars_error("screen", cycle.value)
        _assert_in("capacitance", "screen", ref.estimate, error, cycle.estimate)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_the_near_off_diagonal_capacitance_on_uniform_meshes(tmp_path, n):
    model = rc.bars_model(tmp_path, "near", n)
    result = gfem.solve(model, rc.BARS_PROBLEM, options=OPTIONS)
    estimator = gfem.QuantityEstimator(capacitance=("a", "b"))
    estimate = gfem.error_estimate(result, estimator, model=model)
    error = _bars_error("near", result.capacitance[0, 1])
    _assert_in("capacitance", "near", estimate, error)


# --- step line: S21, singular ------------------------------------------------

S21 = gfem.QuantityEstimator(s=("p2", "p1"))


def _step_errors(s21_by_frequency, estimate):
    values = estimate.values
    for f, s21, value in zip(rc.STEP_S, s21_by_frequency, values, strict=True):
        error = s21 - rc.STEP_S[f]["S21"]
        _assert_in("s", "step line, lumped ports", estimate, error, value)


@pytest.mark.parametrize("n", [2, 4])
def test_the_step_line_s21_on_uniform_meshes(tmp_path, n):
    model = rc.step_model(tmp_path, n)
    problem = gfem.Driven(list(rc.STEP_S), keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, S21, model=model)
    assert len(estimate.values) == len(rc.STEP_S)
    _step_errors(np.asarray(result.s["p2", "p1"]), estimate)


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
        _assert_in("s", "step line, lumped ports", ref.estimate, error, cycle.estimate)


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
    _assert_in("s", "step line, wave ports", estimate, error)


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
    _assert_in("s", "step line, wave ports", estimate, error)


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
        _assert_in("s", "shared port surface", estimate, error)


# --- lossy guide: S on lossy wave ports --------------------------------------

#: Entries (to, from) of the lossy guide: thru first, then reflections.
LOSSY_GUIDE_ENTRIES = (("p2", "p1"), ("p1", "p2"), ("p1", "p1"), ("p2", "p2"))


def _lossy_guide(tmp_path, kind, n, tan_delta, offset):
    """Assert every entry of the lossy guide ``kind`` against `LOSSY_GUIDE_S`.

    The tan_delta 0.3 thru entries of the slab and step at n = 8 are left
    out: their reference is within 20 % of their error only at n = 4.
    """
    model = rc.lossy_guide_model(tmp_path, n, kind, tan_delta, offset * 1e-3)
    problem = gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    references = rc.LOSSY_GUIDE_S[kind, tan_delta, offset]
    for to, source in LOSSY_GUIDE_ENTRIES:
        if kind != "uniform" and tan_delta == 0.3 and n == 8 and to != source:
            continue
        estimator = gfem.QuantityEstimator(s=(to, source))
        estimate = gfem.error_estimate(result, estimator, model=model)
        error = complex(result.s[to, source][0]) - references[to, source]
        _assert_in("s", f"lossy {kind} guide", estimate, error)


_LOSSY_GUIDE_CASES = pytest.mark.parametrize(
    ("n", "tan_delta", "offset"),
    [(n, t, o) for n in (4, 8) for t in (0.05, 0.3) for o in (0, 1)],
)


@_LOSSY_GUIDE_CASES
def test_the_lossy_guide_s_with_a_parallel_port_mode(tmp_path, n, tan_delta, offset):
    _lossy_guide(tmp_path, "uniform", n, tan_delta, offset)


@_LOSSY_GUIDE_CASES
def test_the_lossy_slab_guide_s_with_an_adjoint(tmp_path, n, tan_delta, offset):
    _lossy_guide(tmp_path, "slab", n, tan_delta, offset)


@_LOSSY_GUIDE_CASES
def test_the_lossy_step_guide_s_with_an_adjoint(tmp_path, n, tan_delta, offset):
    _lossy_guide(tmp_path, "step", n, tan_delta, offset)


# --- parallel-plate line: S, smooth ------------------------------------------

_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3, "eps_r": 4.0}
_R1, _R2 = 50.0, 30.0
LINE_FREQUENCIES = (3e9, 8e9, 15e9, 25e9)


def _line_s(f, eps_r=_LINE["eps_r"], mu_r=1.0):
    """The line's S matrix between 50 and 30 ohm ports, from its ABCD matrix.

    ``eps_r`` and ``mu_r`` are the fill's entries the TEM wave sees: eps_r
    along the plates' normal z, mu_r along the width y; a complex eps_r
    (``eps_r (1 - i tan_delta) - i sigma / (omega eps0)``) is a lossy fill.
    """
    beta = 2 * math.pi * f / const.c * cmath.sqrt(eps_r * mu_r)
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * cmath.sqrt(mu_r / eps_r) * _LINE["height"] / _LINE["width"]
    c, s = cmath.cos(beta * _LINE["length"]), cmath.sin(beta * _LINE["length"])
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
    """(estimate, value, error) of S11, S21 and S22 at each frequency.

    ``estimate`` is the entry's `gfem.ErrorEstimate`, ``value`` its value at
    the frequency.

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
        out += [
            (estimate, value, e)
            for value, e in zip(estimate.values, error, strict=True)
        ]
    return out


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_parallel_plate_line_s(tmp_path, n):
    for estimate, value, error in _line_effectivities(*_line(tmp_path, n)):
        _assert_in("s", "parallel-plate line", estimate, error, value)


#: The line's lossy fills: (material, its complex eps_r at a frequency).
LOSSY_LINE_FILLS = {
    "tan_delta": (
        gfem.Material(eps_r=_LINE["eps_r"], tan_delta=0.3),
        lambda _: _LINE["eps_r"] * (1 - 0.3j),
    ),
    "sigma": (
        gfem.Material(eps_r=_LINE["eps_r"], conductivity=0.1),
        lambda f: _LINE["eps_r"] - 0.1j / (2 * math.pi * f * const.epsilon_0),
    ),
}


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("fill", ["tan_delta", "sigma"])
def test_the_lossy_line_s(tmp_path, n, fill):
    # Lumped ports on a lossy fill: the S estimate's mass term recovers
    # eps_eff E, eps_eff = eps_r (1 - i tan_delta) - i sigma / (omega eps0).
    material, eps = LOSSY_LINE_FILLS[fill]
    exact = np.array([_line_s(f, eps(f)) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "parallel-plate line", estimate, error, value)


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
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "permeability line", estimate, error, value)


@pytest.mark.parametrize("fill", ["diag", "rotz"])
@pytest.mark.parametrize("n", [4, 16])
def test_the_anisotropic_line_s(tmp_path, n, fill):
    material, eps, mu = LINE_FILLS[fill]
    exact = np.array([_line_s(f, eps, mu) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "anisotropic line", estimate, error, value)


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
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "anisotropic line", estimate, error, value)


# --- lossy magnetic and anisotropic lines: S (ADR 0138) ----------------------

_LOSSY_AXES = np.array([3.0, 5.0, 2.0]) * (1 - 1j * np.array([0.05, 0.1, 0.3]))
#: The line's lossy magnetic and anisotropic fills: (material, eps_zz, mu_yy),
#: the entries its TEM wave sees, or None where the fill has no closed form.
#: "mu": a complex mu_r; "diag": diagonal eps_r and mu_r with tan_delta 0.1;
#: "rotz": per-axis complex eps_r (tan_delta 0.05, 0.1, 0.3) turned about z;
#: "rot3": that loss with mu_r turned about all three axes.
LOSSY_LINE_TENSORS = {
    "mu": (gfem.MaterialField(eps_r=2.0, mu_r=2.0 * (1 - 0.1j)), 2.0, 2.0 * (1 - 0.1j)),
    "diag": (
        gfem.Material(eps_r=(3.0, 5.0, 2.0), mu_r=(1.5, 2.0, 2.5), tan_delta=0.1),
        2.0 * (1 - 0.1j),
        2.0,
    ),
    "rotz": (
        gfem.MaterialField(
            eps_r=_LOSSY_AXES,
            mu_r=(2.0, 2.0, 2.5),
            axes=tuple(map(tuple, rc.rotation("z", 30).T)),
        ),
        2.0 * (1 - 0.3j),
        2.0,
    ),
    "rot3": (
        gfem.MaterialField(
            eps_r=_LOSSY_AXES, mu_r=(1.5, 2.5, 2.0), axes=tuple(map(tuple, rc.ROT3.T))
        ),
        None,
        None,
    ),
}


@pytest.mark.parametrize("n", [4, 16])
def test_the_lossy_permeability_line_s(tmp_path, n):
    # The recovered H is mu^-1 curl E with the complex mu (ADR 0138).
    material, eps, mu = LOSSY_LINE_TENSORS["mu"]
    exact = np.array([_line_s(f, eps, mu) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "lossy permeability line", estimate, error, value)


@pytest.mark.parametrize("fill", ["diag", "rotz"])
@pytest.mark.parametrize("n", [4, 16])
def test_the_lossy_anisotropic_line_s(tmp_path, n, fill):
    material, eps, mu = LOSSY_LINE_TENSORS[fill]
    exact = np.array([_line_s(f, eps, mu) for f in LINE_FREQUENCIES])
    model, result = _line(tmp_path, n, material=material)
    for estimate, value, error in _line_effectivities(model, result, exact):
        _assert_in("s", "lossy anisotropic line", estimate, error, value)


@pytest.mark.parametrize("n", [4, 16])
def test_the_rotated_lossy_anisotropic_line_s_against_order_4(tmp_path, n):
    # No closed form: order 4 on the same mesh. Rows whose reference is not
    # within 20 % of their error (|S3 - S4| against |S2 - S4|) are left out.
    material, _, _ = LOSSY_LINE_TENSORS["rot3"]
    model, result = _line(tmp_path, n, material=material)
    exact, middle = (
        np.asarray(
            gfem.solve(
                model,
                gfem.Driven(list(LINE_FREQUENCIES)),
                options=gfem.SolverOptions(order=order, linear_solver="direct"),
            ).s.values
        )
        for order in (4, 3)
    )
    uncertainty = np.abs(middle - exact)
    ports = ("p1", "p2")
    kept = 0
    for to, source in (("p1", "p1"), ("p2", "p1"), ("p2", "p2")):
        estimator = gfem.QuantityEstimator(s=(to, source))
        estimate = gfem.error_estimate(result, estimator, model=model)
        i, j = ports.index(to), ports.index(source)
        error = np.asarray(result.s[to, source]) - exact[:, i, j]
        for value, e, u in zip(
            estimate.values, error, uncertainty[:, i, j], strict=True
        ):
            if u <= 0.2 * abs(e):
                kept += 1
                _assert_in("s", "lossy anisotropic line", estimate, e, value)
    assert kept >= 3


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
        error = f / rc.CAVITY_F1 - 1
        _assert_in("omega", "L cavity", ref.estimate, error, cycle.estimate)


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
        out.append((f, estimate, exact))
    return out


@pytest.mark.parametrize("m", [2, 3])
def test_the_rectangular_box_frequencies(tmp_path, m):
    for f, estimate, exact in _box(tmp_path, m):
        _assert_in("omega", "rectangular box", estimate, f / exact - 1)


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
    """(frequency, its `gfem.ErrorEstimate`) of the mode nearest ``target``, order 2."""
    result = gfem.solve(
        model, gfem.Eigenmode(target=target, n_modes=1), options=OPTIONS
    )
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    return float(np.real(np.asarray(result.frequency)[0])), estimate


@pytest.mark.parametrize("mode", BOX_MODES)
@pytest.mark.parametrize("m", [2, 3])
def test_the_permeability_box_frequency(tmp_path, m, mode):
    exact = _diagonal_box_frequency("iso_mu", mode)
    model = rc.anisotropic_box_model(tmp_path, m, "iso_mu")
    f, estimate = _filled_box_mode(model, exact)
    _assert_in("omega", "permeability box", estimate, f / exact - 1)


@pytest.mark.parametrize("mode", BOX_MODES)
@pytest.mark.parametrize("m", [2, 3])
def test_the_anisotropic_box_frequency(tmp_path, m, mode):
    exact = _diagonal_box_frequency("diag", mode)
    model = rc.anisotropic_box_model(tmp_path, m, "diag")
    f, estimate = _filled_box_mode(model, exact)
    _assert_in("omega", "anisotropic box", estimate, f / exact - 1)


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
    _assert_in("omega", case, estimate, f / reference - 1)


# --- lossy eigenmodes: complex omega ------------------------------------------

_ETA0 = const.mu_0 * const.c
#: The lossy fill: eps_r 4, tan_delta 0.1, conductivity 5 S/m.
_LOSSY_EPS, _LOSSY_TAN, _LOSSY_SIGMA = 4.0, 0.1, 5.0
LOSSY_FILLS = {
    "material": gfem.Material(
        eps_r=_LOSSY_EPS, tan_delta=_LOSSY_TAN, conductivity=_LOSSY_SIGMA
    ),
    "complex eps_r": gfem.MaterialField(
        eps_r=_LOSSY_EPS * (1 - 1j * _LOSSY_TAN), conductivity=_LOSSY_SIGMA
    ),
}


def _lossy_omega(k0_squared, eps_c, sigma, near):
    """The root of eps_c k^2 - i eta0 sigma k - k0^2 = 0 nearest ``near``, rad/s.

    ``k0_squared`` is the lossless Dirichlet eigenvalue, rad^2/m^2: a cavity
    uniformly filled with eps_c and sigma resonates where the fill's
    eps_c - i eta0 sigma / k times k^2 equals it.
    """
    roots = np.roots([eps_c, -1j * _ETA0 * sigma, -k0_squared]) * const.c
    return complex(roots[np.argmin(abs(roots - near))])


def _box_model(directory, m, material, top=None):
    cells = sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m))
    path = sm.write_msh22(directory / "box.msh", cells)
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": material},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("side"), top or gfem.PEC("top")],
    )


def _omega_and_estimate(model, target):
    result = gfem.solve(
        model, gfem.Eigenmode(target=target, n_modes=1), options=OPTIONS
    )
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    return complex(np.asarray(result.omega)[0]), estimate


@pytest.mark.parametrize("fill", list(LOSSY_FILLS))
def test_the_lossy_box_frequencies(tmp_path, fill):
    # The rectangular box filled uniformly: every mode keeps its field and
    # its omega solves the fill's quadratic, Q from 4.5 to 7.1.
    model = _box_model(tmp_path, 2, LOSSY_FILLS[fill])
    eps_c = _LOSSY_EPS * (1 - 1j * _LOSSY_TAN)
    for mode in BOX_MODES:
        k0_squared = math.pi**2 * sum(
            (i / a) ** 2 for i, a in zip(mode, BOX, strict=True)
        )
        target = const.c * math.sqrt(k0_squared / _LOSSY_EPS) / (2 * math.pi)
        omega, estimate = _omega_and_estimate(model, target)
        exact = _lossy_omega(k0_squared, eps_c, _LOSSY_SIGMA, omega)
        _assert_in("omega", "lossy box", estimate, (omega - exact) / abs(exact))


def test_the_lossy_l_cavity_frequency_under_mesh_refinement(tmp_path):
    lossless = rc.cavity_model(tmp_path, 2)
    fill = gfem.Material(eps_r=_LOSSY_EPS, tan_delta=_LOSSY_TAN)
    model = gfem.Model(
        lossless.mesh, materials={"air": fill}, boundaries=lossless.boundaries
    )
    k0_squared = rc.CAVITY_LAMBDA1 * 1e6  # mm^-2 to m^-2
    eps_c = _LOSSY_EPS * (1 - 1j * _LOSSY_TAN)
    exact = _lossy_omega(k0_squared, eps_c, 0.0, 2 * math.pi * rc.CAVITY_F1)
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            model,
            gfem.Eigenmode(target=exact.real / (2 * math.pi), n_modes=1),
            estimator=gfem.QuantityEstimator(omega=0),
            tolerance=1e-7,
            max_cycles=7,
            options=OPTIONS,
        )
    assert len(ref.cycles) == 8
    for cycle in ref.cycles:
        error = (cycle.value - exact) / abs(exact)
        _assert_in("omega", "lossy L cavity", ref.estimate, error, cycle.estimate)


LOSSY_WALLS = {
    "impedance": gfem.Impedance("top", resistance=377.0),
    "absorbing": gfem.Absorbing("top"),
    "conductor": gfem.Conductor("top", conductivity=1e5),
}


@pytest.mark.parametrize("wall", list(LOSSY_WALLS))
@pytest.mark.parametrize("m", [2, 3])
def test_the_box_with_a_lossy_wall(tmp_path, m, wall):
    # Air box, PEC but for its top wall; mode 110, Q 4.0 (impedance,
    # absorbing) and 254 (conductor). The loss is on the boundary only, where
    # the estimate has no term: it enters through T'(k) alone.
    model = _box_model(tmp_path, m, gfem.Material(), LOSSY_WALLS[wall])
    target = const.c * math.sqrt(1 / BOX[0] ** 2 + 1 / BOX[1] ** 2) / 2
    omega, estimate = _omega_and_estimate(model, target)
    reference = rc.LOSSY_WALL_OMEGA[wall]
    error = (omega - reference) / abs(reference)
    _assert_in("omega", "lossy walls", estimate, error)


#: The parallel-plate line closed by its 50 and 30 ohm lumped ports, PEC
#: plates: a resonator whose only loss is the two resistors (Q 1.8 and 3.4).
_RESONANCES = (7.5e9, 15e9)


def _line_resonance(near):
    """The resonance nearest ``near``, rad/s: Gamma1 Gamma2 e^{-2j beta L} = 1."""
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * _LINE["height"] / (_LINE["width"] * math.sqrt(_LINE["eps_r"]))
    gammas = ((_R1 - z0) / (_R1 + z0)) * ((_R2 - z0) / (_R2 + z0))
    speed = const.c / math.sqrt(_LINE["eps_r"])
    # beta L = pi n - i ln(Gamma1 Gamma2) / 2, omega = beta speed
    n = round(near.real * _LINE["length"] / (speed * math.pi))
    return complex((math.pi * n - 0.5j * np.log(gammas)) * speed / _LINE["length"])


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_the_resistive_line_resonances(tmp_path, n):
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=_LINE["eps_r"])},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=_R1),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    for target in _RESONANCES:
        omega, estimate = _omega_and_estimate(model, target)
        exact = _line_resonance(omega)
        _assert_in("omega", "resistive line", estimate, (omega - exact) / abs(exact))


# --- lifted eigen terms (ADR 0138): laws, second-order walls, lossy mu, loss --
# --- inside an anisotropic material --------------------------------------------


def _nearest_mode(model, target, near, n_modes=1):
    """(omega, its omega `gfem.ErrorEstimate`) of the solved mode nearest ``near``.

    ``target`` (Hz) seeds the solve; ``near`` (rad/s) picks among its modes.
    """
    problem = gfem.Eigenmode(target=target, n_modes=n_modes)
    result = gfem.solve(model, problem, options=OPTIONS)
    omegas = np.asarray(result.omega, complex)
    i = int(np.argmin(abs(omegas - near)))
    estimator = gfem.QuantityEstimator(omega=i)
    return complex(omegas[i]), gfem.error_estimate(result, estimator, model=model)


def _nearest_root(roots, near):
    roots = np.asarray(roots, complex)
    return complex(roots[np.argmin(abs(roots - near))])


def _k0_squared(mode):
    """The empty PEC box's k0^2 of ``mode``, 1/m^2."""
    return sum((n * math.pi / a) ** 2 for n, a in zip(mode, BOX, strict=True))


#: Every mode index of the box up to (4, 4, 3) with at most one zero: the
#: closed forms of a filled box match the solved mode to the nearest root
#: over all of them, so a mode found in place of the targeted one is still
#: checked against its own root.
_ALL_MODES = [
    (a, b, c)
    for a in range(5)
    for b in range(5)
    for c in range(4)
    if sum(x == 0 for x in (a, b, c)) <= 1
]


def _drude_roots(eps_inf, fp, gamma):
    """f^2 eps(f) = K^2 for a Drude fill, every box mode's roots in rad/s."""
    out = []
    for mode in _ALL_MODES:
        kk = (const.c * math.sqrt(_k0_squared(mode)) / (2 * math.pi)) ** 2
        coefficients = [eps_inf, -1j * gamma * eps_inf, -(kk + fp**2), 1j * gamma * kk]
        out += list(2 * math.pi * np.roots(coefficients))
    return out


def _lorentz_roots(eps_inf, delta, f0, gamma):
    """f^2 eps(f) = K^2 for a Lorentz fill, every box mode's roots in rad/s."""
    out = []
    for mode in _ALL_MODES:
        kk = (const.c * math.sqrt(_k0_squared(mode)) / (2 * math.pi)) ** 2
        coefficients = [
            -eps_inf,
            1j * gamma * eps_inf,
            (eps_inf + delta) * f0**2 + kk,
            -1j * gamma * kk,
            -kk * f0**2,
        ]
        out += list(2 * math.pi * np.roots(coefficients))
    return out


#: The dispersive fills of the box: (law, its closed-form roots, the static
#: eps_r that seeds each mode's target).
DISPERSIVE_FILLS = {
    "drude, lossy": (
        gfem.PermittivityLaw(
            eps_inf=4.0, terms=[gfem.Drude(plasma_frequency=300e9, damping=30e9)]
        ),
        lambda: _drude_roots(4.0, 300e9, 30e9),
    ),
    "drude, lossless": (
        gfem.PermittivityLaw(
            eps_inf=4.0, terms=[gfem.Drude(plasma_frequency=300e9, damping=0.0)]
        ),
        lambda: _drude_roots(4.0, 300e9, 0.0),
    ),
    "lorentz": (
        gfem.PermittivityLaw(
            eps_inf=2.0,
            terms=[
                gfem.Lorentz(delta_eps=2.0, resonance_frequency=600e9, damping=50e9)
            ],
        ),
        lambda: _lorentz_roots(2.0, 2.0, 600e9, 50e9),
    ),
}


def _dispersive_target(fill, mode):
    """The undamped estimate of ``mode``'s frequency in ``fill``, Hz."""
    kk = (const.c * math.sqrt(_k0_squared(mode)) / (2 * math.pi)) ** 2
    if fill == "lorentz":
        return math.sqrt(kk / 4.0)
    return math.sqrt((kk + 300e9**2) / 4.0)


@pytest.mark.parametrize("fill", list(DISPERSIVE_FILLS))
def test_the_dispersive_box_frequencies(tmp_path, fill):
    # The box filled with a law at Re eps > 0: its omega solves
    # f^2 eps(f) = K^2 (Q 12 to 106 on the lossy fills).
    law, roots = DISPERSIVE_FILLS[fill]
    model = _box_model(tmp_path, 2, gfem.Material(eps_r=law))
    for mode in ((1, 1, 0), (1, 0, 1)):
        target = _dispersive_target(fill, mode)
        omega, estimate = _nearest_mode(model, target, 2 * math.pi * target)
        exact = _nearest_root(roots(), omega)
        _assert_in("omega", "dispersive box", estimate, abs(omega - exact) / abs(exact))


#: The layered boxes' lower layer: a Drude law at Re eps > 0, and that law
#: frozen at 217.4 GHz (`refinement_cases.LAYERED_BOX_OMEGA`).
_LAYER_LAW = gfem.PermittivityLaw(
    eps_inf=4.0, terms=[gfem.Drude(plasma_frequency=300e9, damping=30e9)]
)
LAYERED_FILLS = {
    "law": gfem.Material(eps_r=_LAYER_LAW),
    "constant": gfem.MaterialField(
        eps_r=4.0 - 300e9**2 / (217.4e9 * (217.4e9 - 30e9j))
    ),
}


def _layered_box(directory, lower):
    cells = sm.plate(1.0, 0.7, (("lower", 0.2), ("air", 0.2)), (10, 8, 2))
    path = sm.write_msh22(directory / "box.msh", cells)
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"lower": lower, "air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )


@pytest.mark.parametrize("fill", list(LAYERED_FILLS))
def test_the_layered_box_frequencies(tmp_path, fill):
    # A lossy layer under air, against order 4 on the same mesh. The omega
    # estimate reaches about 21 here with or without the law: the layered
    # cavity's own overestimate.
    model = _layered_box(tmp_path, LAYERED_FILLS[fill])
    for target, reference in rc.LAYERED_BOX_OMEGA[fill].items():
        omega, estimate = _nearest_mode(model, target, reference)
        error = abs(omega - reference) / abs(reference)
        assert error > 5 * rc.LAYERED_BOX_BOUND[fill][target]
        _assert_in("omega", "layered box", estimate, error)


_BOX_PEC_MODES = ((1, 1, 0), (2, 1, 0), (1, 2, 0))


def _absorbing_box(directory, m):
    cells = sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m))
    path = sm.write_msh22(directory / "box.msh", cells)
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
    )


@pytest.mark.parametrize("m", [1, 2])
def test_the_box_with_second_order_absorbing_walls(tmp_path, m):
    # Q 1.7 to 4. Six modes are solved and the one nearest the stored order-4
    # reference is estimated: a single-mode seed can return a spurious mode.
    # (On the top wall, the second-order term is inert for E normal to it.)
    model = _absorbing_box(tmp_path, m)
    for mode in _BOX_PEC_MODES:
        target = const.c * math.sqrt(_k0_squared(mode)) / (2 * math.pi)
        reference = rc.ABSORBING_BOX_OMEGA[m, mode]
        omega, estimate = _nearest_mode(model, target, reference, n_modes=6)
        error = abs(omega - reference) / abs(reference)
        assert abs(omega - reference) < 0.05 * abs(reference)  # the same mode
        assert error > 5 * rc.ABSORBING_BOX_BOUND[m, mode]
        _assert_in("omega", "second-order absorbing walls", estimate, error)


_LOSSY_MU = 2.0 * (1 - 0.1j)


@pytest.mark.parametrize("m", [2])
def test_the_lossy_permeability_box_frequencies(tmp_path, m):
    # eps_r 2, mu_r 2 (1 - 0.1i): omega = c k0 / sqrt(eps mu), Q 10. The
    # recovered H is mu^-1 curl E with the complex mu.
    model = rc.anisotropic_box_model(tmp_path, m, "lossy_mu")
    for mode in ((1, 1, 0), (1, 0, 1)):
        k0 = math.sqrt(_k0_squared(mode))
        target = const.c * k0 / math.sqrt(2.0 * _LOSSY_MU.real) / (2 * math.pi)
        omega, estimate = _nearest_mode(model, target, 2 * math.pi * target)
        w = const.c * k0 / cmath.sqrt(2.0 * _LOSSY_MU)
        exact = _nearest_root([w, -w, w.conjugate(), -w.conjugate()], omega)
        error = abs(omega - exact) / abs(exact)
        _assert_in("omega", "lossy permeability box", estimate, error)


@pytest.mark.parametrize("fill", ["lossy_layer_mu", "lossy_rot3"])
def test_the_lossy_layered_and_rotated_box_frequencies(tmp_path, fill):
    # No closed form: against order 4 on the same mesh. A real-mu recovery
    # measured up to 40.9 on the layers.
    case = {
        "lossy_layer_mu": "lossy permeability layers",
        "lossy_rot3": "anisotropic box",
    }
    model = rc.anisotropic_box_model(tmp_path, 2, fill)
    for target, reference in rc.LOSSY_BOX_OMEGA[fill].items():
        omega, estimate = _nearest_mode(model, target, reference)
        error = abs(omega - reference) / abs(reference)
        assert error > 5 * rc.LOSSY_BOX_BOUND[fill][target]
        _assert_in("omega", case[fill], estimate, error)


_DIAG_EPS = np.array([2.0, 3.0, 4.0])
_DIAG_MU = np.array([1.5, 2.0, 2.5])
#: The lossy diagonal fills: (complex eps_r, conductivity) per axis.
LOSSY_DIAGONAL = {
    "lossy_diag_tan": (_DIAG_EPS * (1 - 0.1j), np.zeros(3)),
    "lossy_diag_sigma": (_DIAG_EPS.astype(complex), np.array([5.0, 10.0, 20.0])),
    "lossy_diag_complex": (rc.LOSSY_EPS_DIAG, np.zeros(3)),
}


def _diagonal_k2(mode):
    """sum over i != d of (n_i pi / L_i)^2 / mu_jj, d the axis of the zero index."""
    d = mode.index(0)
    return sum(
        (mode[i] * math.pi / BOX[i]) ** 2 / _DIAG_MU[3 - i - d]
        for i in range(3)
        if i != d
    )


def _lossy_diagonal_roots(fill):
    """The closed-form omegas (rad/s) of every box mode with one zero index.

    With E along the axis d of the zero index, k^2 eps_eff,dd = `_diagonal_k2`,
    eps_eff = eps_c - i eta0 sigma / k. A target can return a neighbouring
    mode, so the solved mode is matched to the nearest root of them all.
    """
    eps_c, sigma = LOSSY_DIAGONAL[fill]
    out = []
    for mode in _ALL_MODES:
        if mode.count(0) != 1:
            continue
        d = mode.index(0)
        roots = np.roots([eps_c[d], -1j * _ETA0 * sigma[d], -_diagonal_k2(mode)])
        out += list(roots * const.c)
    return out


@pytest.mark.parametrize(
    ("m", "modes"), [(1, ((1, 1, 0), (1, 0, 1))), (2, BOX_MODES)], ids=["m1", "m2"]
)
@pytest.mark.parametrize("fill", list(LOSSY_DIAGONAL))
def test_the_lossy_anisotropic_box_frequencies(tmp_path, fill, m, modes):
    # Loss inside a diagonal tensor fill, Q 1.1 to 10: the modulus weights of
    # the complex tensor eps_eff are exact (ADR 0138).
    model = rc.anisotropic_box_model(tmp_path, m, fill)
    roots = _lossy_diagonal_roots(fill)
    for mode in modes:
        d = mode.index(0)
        target = const.c * math.sqrt(_diagonal_k2(mode) / _DIAG_EPS[d]) / (2 * math.pi)
        omega, estimate = _nearest_mode(model, target, 2 * math.pi * target)
        exact = _nearest_root(roots, omega)
        error = abs(omega - exact) / abs(exact)
        assert error < 1e-3  # a mode with one zero index
        _assert_in("omega", "anisotropic box", estimate, error)


def test_the_plasmon_bound_mode_measures_below_the_floor(tmp_path):
    # Not claimed: why a mode bound to a law region at Re eps < 0 raises
    # (ADR 0138). The private kernel, past the refusal, estimates the LSM
    # (2, 1) mode of the Drude floor at about 0.96 of its error on m = 2,
    # falling towards 0.81 as Re eps approaches -1 and the mesh coarsens.
    from gfem._engine import estimate as kernels

    floor = gfem.PermittivityLaw(
        terms=[gfem.Drude(plasma_frequency=350e9, damping=5e9)]
    )
    cells = sm.plate(1.0, 0.7, (("metal", 0.2), ("air", 0.2)), (10, 8, 2))
    path = sm.write_msh22(tmp_path / "box.msh", cells)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"metal": gfem.Material(eps_r=floor), "air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    problem = gfem.Eigenmode(target=200e9, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    with pytest.raises(gfem.MeshRefinementUnsupportedError):
        gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    omega = complex(np.asarray(result.omega)[0])
    error = abs(omega - rc.PLASMON_OMEGA) / abs(rc.PLASMON_OMEGA)
    assert error < 1e-2  # the LSM (2, 1) mode
    value = kernels.omega(model, result, 0).value
    assert PLASMON_WITNESS[0] <= value / error <= PLASMON_WITNESS[1]


#: The refused plasmon-bound mode's omega effectivity (measured 0.959 on
#: m = 2): a band below the floor's reach, not a claim.
PLASMON_WITNESS = (0.93, 0.99)


# --- order 3 -----------------------------------------------------------------


# A drift fence, not a claim: the docs claim `EFFECTIVITY_BY_KIND` at order 2 only,
# and these ranges are not part of it. At order 3 only the parallel-plate
# line leaves the order-2 range, at 19.6 on n = 8 (S22, 8 GHz), the mesh where
# order 2 also peaks (15.75): the same smooth-field overestimation as at
# order 2, curl and mass terms the estimate never lets cancel, not an order-3
# effect. The box at order 3 lies inside its order-2 range. Recovering into
# degree p instead of p - 1 (underestimates singular benchmark cases, down to
# 0.51, measured before the natural-surface constraint) and a finer
# quadrature (identical to every printed digit) were measured and rejected.
#: Asserted order-3 effectivity range per case (measured, with margin).
ORDER_3_EFFECTIVITY_BY_CASE = {
    "parallel-plate line": (0.9, 25.0),  # measured 3.10-19.6
    "rectangular box": (0.9, 8.0),  # measured 4.69-5.39 (m = 1)
}


def _assert_order_3(kind, case, estimate, error, value=None):
    """As `_assert_in`, against the order-3 fence."""
    assert estimate.kind == kind, (case, estimate.kind)
    low, high = ORDER_3_EFFECTIVITY_BY_CASE[case]
    value = estimate.value if value is None else value
    assert low <= value / abs(error) <= high, (case, value / abs(error))


@pytest.mark.parametrize("n", [4, 8])
def test_the_parallel_plate_line_s_at_order_3(tmp_path, n):
    for estimate, value, error in _line_effectivities(*_line(tmp_path, n, order=3)):
        _assert_order_3("s", "parallel-plate line", estimate, error, value)


def test_the_rectangular_box_frequencies_at_order_3(tmp_path):
    for f, estimate, exact in _box(tmp_path, 1, order=3):
        _assert_order_3("omega", "rectangular box", estimate, f / exact - 1)


# --- order 4 -----------------------------------------------------------------


#: The order-4 drift fence on the L cavity's omega (measured 1.80, 1.73): the
#: order-3 fence's bounds. Not a claim: order 4 and above run, unverified
#: beyond this one case.
ORDER_4_EFFECTIVITY = (0.9, 25.0)


@pytest.mark.parametrize("n", [2, 4])
def test_the_l_cavity_frequency_at_order_4(tmp_path, n):
    model = rc.cavity_model(tmp_path, n)
    options = gfem.SolverOptions(order=4, linear_solver="direct")
    result = gfem.solve(model, rc.CAVITY_PROBLEM, options=options)
    estimate = gfem.error_estimate(result, gfem.QuantityEstimator(omega=0), model=model)
    error = abs(rc.cavity_frequency(result) / rc.CAVITY_F1 - 1)
    low, high = ORDER_4_EFFECTIVITY
    assert estimate.kind == "omega"
    assert low < estimate.value / error < high
