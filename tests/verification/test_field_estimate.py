"""The field estimator and mesh refinement, on singular and smooth cases.

Square coax (electrostatic, singular). For a capacitance with one terminal
at 1 V, C_h - C = a(e, e) exactly, so the true relative energy error of the
potential is sqrt((C_h - C) / C), with C the hard-coded reference
`refinement_cases.COAX_C`. The field estimator's effectivity (estimate over
that error) measured 1.25, 1.30, 1.32 and 1.33 under uniform refinement,
n = 2 to 16. Mesh refinement driven by it, run until the next mesh would
pass a quarter of the uniform n = 16 tetrahedra, ends on 936 tetrahedra and
1578 DOF with a capacitance error of 4.4e-4, below the uniform n = 16 error
of 5.1e-4 on 9603 DOF.

Step line (driven) and L cavity (eigenmode), singular. The cavity's true
relative energy error is sqrt(|lambda_h - lambda| / lambda); the step line
has none, so its effectivity (the root mean square of the per-frequency
estimates) is against the proxy sqrt(rms |S21 - S21_ref|) over 3 and 8 GHz,
and only its flatness is evidence. Mesh refinement is compared with the
uniform split of each tetrahedron into 8 (`refinement_cases.STEP_SPLIT`,
`CAVITY_SPLIT`) at the adaptive run's final DOF.

Smooth cases, against a closed-form field or capacitance: a parallel-plate
line between 50 and 30 ohm lumped ports, whose field is the transmission
line's TEM wave; the first mode of a rectangular PEC box; a circular coax
quarter, against an order-5 solve on the same mesh (its error on its own
facets) and against the exact log formula, where the faceting, which mesh
refinement never removes and the estimate does not see, counts in the true
error. The true error of a field is its distance to the exact one in the
energy norm, || curl e ||^2 + k^2 || e ||^2_eps, relative to the field's, by
a Gauss rule on each tetrahedron.

Anisotropic coax (`refinement_cases.anisotropic_coax_model`), electrostatic:
the circular coax quarter filled with a rotated tensor eps_r, against an
order-5 solve on the same mesh. The recovery's constraint on the natural
surfaces (ADR 0087) is what keeps this estimate above the error: without it
the estimate measured 0.45 times the error.

Lossy cases, in the norm || curl e ||^2 + |k|^2 || e ||^2_{|eps_eff|} (the
lossless norm without loss): the parallel-plate line with tan_delta 0.3 or
10 S/m in its filling, against the closed-form TEM wave with complex beta and
Z0; and the PEC box in two 0.2 mm layers, the upper one lossy (tan_delta 0.5
or 30 S/m), whose mode puts E normal to the loss jump, against an order p + 2
solve on the same mesh. Loss on a wall only: the air box, PEC but for an
impedance (377 ohm/sq) or first-order absorbing top wall, whose mode has a
complex omega (|k| weighs the mass term), against an order p + 2 solve on
the same mesh. A lumped port across a loss jump (the line in two
layers, the upper lossy) is a witness: its effectivity is asserted in its
measured band `LOSS_JUMP_WITNESS`, below the quoted range.

Each case's effectivity must lie in its range of `FIELD_EFFECTIVITY_BY_CASE`
at order 2, and of `FIELD_EFFECTIVITY_ORDER_3_BY_CASE` at order 3 (the
circular coax against an order-6 solve there); `FIELD_EFFECTIVITY` (the
lowest low and the highest high over both) is the range the docs quote. The
step line's proxy ratio is not an effectivity and is asserted on its own.
At order 1 both estimators refuse: there the L cavity's frequency, on its one
cell across the thickness, moved away from the exact one under refinement,
and the estimate measured 0.77 times the error.

Measured (order 2, direct, ostau, 1 thread):

    step line, uniform n = 2, 4, 8, 16   0.891, 0.894, 0.894, 0.893 (proxy)
    L cavity, uniform n = 4, 8, 16       1.139, 1.037, 0.995
    step line, refined to 9099 tets     S21(3 GHz) error 2.5e-5 on 60064 DOF,
                                        8.3 x below the uniform split
    L cavity, refined to 6324 tets      f1 error 5.2e-5 on 43378 DOF,
                                        16 x below the uniform split
    parallel-plate line, n = 4, 8, 16   1.16 to 1.25 (3 and 8 GHz)
    PEC box TM110, h = 0.2/m mm,
    m = 1, 2, 3                         1.20, 1.22, 1.22
    circular coax, k = 2, 4, 8          1.21, 1.21, 1.20 (own facets)
                                        0.94, 0.94, 0.94 (exact circle)
    anisotropic coax, k = 1, 2, 4, 8    1.28, 1.39, 1.48, 1.54 (rotz)
                                        1.27, 1.38, 1.47, 1.51 (rot3)
    lossy line, tan_delta 0.3, n = 4, 16  1.17, 1.25; 1.16, 1.17 (3, 8 GHz)
    lossy line, 10 S/m, n = 8, 32       1.26, 1.25; 1.06, 1.06 (0.62 and
                                        0.73 at n = 32 recovering eps_r E)
    two-layer box, m = 1, 2             1.20, 1.19 (tan_delta 0.5)
                                        1.18, 1.15 (30 S/m)
    lossy-wall box, m = 1, 2            1.10, 1.10 (impedance and absorbing)
    loss-jump port line, order 2,
    order-4 reference, n = 4, 8, 12     0.93, 0.95, 0.96 (1 S/m, 3 GHz)

Measured (order 3, direct, ostau, 1 thread):

    square coax, n = 2, 4, 8            1.25, 1.29, 1.31
    step line, n = 2, 4, 8              0.881, 0.886, 0.886 (proxy)
    L cavity, n = 4, 8                  0.940, 0.923
    parallel-plate line, n = 4, 16      1.83, 1.84; 1.57, 1.63 (3 and 8 GHz)
    PEC box TM110, m = 1, 2             1.56, 1.70
    circular coax, k = 2, 4             1.38, 1.39 (own facets, order 6)
                                        0.69, 0.70 (exact circle)
    lossy line, tan_delta 0.3, n = 4, 16  1.83, 1.80; 1.56, 1.63 (3, 8 GHz)
    lossy line, 10 S/m, n = 4, 16       1.61, 1.50; 1.59, 1.55
    two-layer box, m = 1                1.40 (tan_delta 0.5), 1.21 (30 S/m)
    lossy-wall box, m = 1               1.33 (impedance and absorbing)
    loss-jump port line, n = 4,
    order-7 reference                   0.88, 0.89 (tan_delta 0.5)
                                        0.85, 0.89 (1 S/m)
"""

import cmath
import math

import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm
from scipy.special import roots_jacobi

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
ORDER_3 = gfem.SolverOptions(order=3, linear_solver="direct")

#: Asserted field effectivity range per case (measured, with margin).
FIELD_EFFECTIVITY_BY_CASE = {
    "square coax": (0.9, 1.6),
    "circular coax": (0.9, 1.6),
    "L cavity": (0.9, 1.5),
    "parallel-plate line": (0.9, 1.6),
    "rectangular box": (0.9, 1.6),
    "anisotropic coax": (0.9, 1.8),
    "lossy parallel-plate line": (0.9, 1.6),
    "two-layer box": (0.9, 1.6),
    "lossy-wall box": (0.9, 1.6),
}

#: Asserted field effectivity range per case at order 3 (measured, with margin).
FIELD_EFFECTIVITY_ORDER_3_BY_CASE = {
    "square coax": (0.9, 1.6),
    "circular coax": (0.9, 1.6),
    "L cavity": (0.9, 1.5),
    "parallel-plate line": (0.9, 2.0),
    "rectangular box": (0.9, 2.0),
    "lossy parallel-plate line": (0.9, 2.0),
    "two-layer box": (0.9, 1.6),
    "lossy-wall box": (0.9, 1.6),
}

#: The range the docs quote: the lowest low and the highest high, both orders.
FIELD_EFFECTIVITY = (
    min(
        low
        for ranges in (FIELD_EFFECTIVITY_BY_CASE, FIELD_EFFECTIVITY_ORDER_3_BY_CASE)
        for low, _ in ranges.values()
    ),
    max(
        high
        for ranges in (FIELD_EFFECTIVITY_BY_CASE, FIELD_EFFECTIVITY_ORDER_3_BY_CASE)
        for _, high in ranges.values()
    ),
)


def _assert_in(case, estimate, error, order=2):
    ranges = (
        FIELD_EFFECTIVITY_BY_CASE if order == 2 else FIELD_EFFECTIVITY_ORDER_3_BY_CASE
    )
    low, high = ranges[case]
    assert low <= estimate / error <= high, (case, order, estimate / error)


def _energy_error(result: gfem.ElectrostaticResult) -> float:
    return math.sqrt(abs(rc.coax_capacitance(result) - rc.COAX_C) / rc.COAX_C)


def _dofs(result: gfem.ElectrostaticResult) -> int:
    return len(result.potential["inner"].coefficients)


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    """Uniform order-2 solves and their field estimates, n = 2, 4, 8, 16."""
    out = {}
    for n in (2, 4, 8, 16):
        model = rc.coax_model(tmp_path_factory.mktemp(f"coax{n}"), n)
        result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
        estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
        out[n] = (model, result, estimate)
    return out


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_field_estimator_effectivity_is_near_one_on_uniform_meshes(uniform, n):
    _, result, estimate = uniform[n]
    _assert_in("square coax", estimate.value, _energy_error(result))


@pytest.mark.parametrize("n", [2, 4, 8])
def test_the_square_coax_effectivity_at_order_3(tmp_path, n):
    model = rc.coax_model(tmp_path, n)
    result = gfem.solve(model, rc.COAX_PROBLEM, options=ORDER_3)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    _assert_in("square coax", estimate.value, _energy_error(result), order=3)


def test_mesh_refinement_reaches_the_uniform_error_with_a_fifth_of_its_dofs(uniform):
    fine_model, fine, _ = uniform[16]
    target = abs(rc.coax_capacitance(fine) - rc.COAX_C) / rc.COAX_C
    coarse, _, _ = uniform[2]
    # run until the next mesh would pass a quarter of the uniform tetrahedra
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            coarse,
            rc.COAX_PROBLEM,
            tolerance=1e-3,
            max_cycles=30,
            max_tetrahedra=len(fine_model.mesh.tetrahedra) // 4,
            options=OPTIONS,
        )
    assert ref.stop == "max_tetrahedra"
    error = abs(rc.coax_capacitance(ref.result) - rc.COAX_C) / rc.COAX_C
    assert error <= target
    assert ref.cycles[-1].dofs == _dofs(ref.result)
    assert ref.cycles[-1].dofs < _dofs(fine) / 5


# --- step line and L cavity --------------------------------------------------


def _step_proxy(result: gfem.DrivenResult) -> float:
    s21 = rc.step_s(result)[:, 1, 0]
    ref = np.array([rc.STEP_S[f]["S21"] for f in rc.STEP_S])
    return math.sqrt(math.sqrt(np.mean(np.abs(s21 - ref) ** 2)))


def _cavity_error(result: gfem.EigenmodeResult) -> float:
    return math.sqrt(abs((rc.cavity_frequency(result) / rc.CAVITY_F1) ** 2 - 1))


def _rms(estimate: gfem.ErrorEstimate) -> float:
    return math.sqrt(np.mean(np.square(estimate.values)))


@pytest.fixture(scope="module")
def step_uniform(tmp_path_factory):
    """Uniform solves of the step line and their field estimates, n = 2 to 16."""
    out = {}
    problem = gfem.Driven(list(rc.STEP_S), keep_fields=True)
    for n in (2, 4, 8, 16):
        model = rc.step_model(tmp_path_factory.mktemp(f"step{n}"), n)
        result = gfem.solve(model, problem, options=OPTIONS)
        estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
        out[n] = (result, estimate)
    return out


@pytest.fixture(scope="module")
def cavity_uniform(tmp_path_factory):
    """Uniform solves of the L cavity and their field estimates, n = 4 to 16."""
    out = {}
    for n in (4, 8, 16):
        model = rc.cavity_model(tmp_path_factory.mktemp(f"cavity{n}"), n)
        result = gfem.solve(model, rc.CAVITY_PROBLEM, options=OPTIONS)
        estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
        out[n] = (result, estimate)
    return out


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_the_step_line_effectivity_is_flat(step_uniform, n):
    result, estimate = step_uniform[n]
    assert len(estimate.values) == len(rc.STEP_S)
    assert 0.7 < _rms(estimate) / _step_proxy(result) < 1.2


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_cavity_effectivity_is_near_one(cavity_uniform, n):
    result, estimate = cavity_uniform[n]
    _assert_in("L cavity", estimate.value, _cavity_error(result))


@pytest.mark.parametrize("n", [2, 4, 8])
def test_the_step_line_effectivity_is_flat_at_order_3(tmp_path, n):
    problem = gfem.Driven(list(rc.STEP_S), keep_fields=True)
    model = rc.step_model(tmp_path, n)
    result = gfem.solve(model, problem, options=ORDER_3)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    assert 0.7 < _rms(estimate) / _step_proxy(result) < 1.2


@pytest.mark.parametrize("n", [4, 8])
def test_the_cavity_effectivity_at_order_3(tmp_path, n):
    model = rc.cavity_model(tmp_path, n)
    result = gfem.solve(model, rc.CAVITY_PROBLEM, options=ORDER_3)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    _assert_in("L cavity", estimate.value, _cavity_error(result), order=3)


def _adaptive(model, problem, max_tetrahedra):
    with pytest.warns(gfem.MeshRefinementNotConvergedWarning):
        ref = gfem.refine_mesh(
            model,
            problem,
            tolerance=1e-3,
            max_cycles=40,
            max_tetrahedra=max_tetrahedra,
            options=OPTIONS,
        )
    assert ref.stop == "max_tetrahedra"
    return ref


def test_mesh_refinement_beats_the_uniform_split_on_the_step_line(tmp_path):
    ref = _adaptive(rc.step_model(tmp_path, 2), rc.step_problem(), 10_000)
    error = abs(rc.step_s(ref.result)[0, 1, 0] - rc.STEP_S[3e9]["S21"])
    uniform = rc.uniform_split_error(rc.STEP_SPLIT, ref.cycles[-1].dofs)
    assert uniform / error >= 5


def test_mesh_refinement_beats_the_uniform_split_on_the_cavity(tmp_path):
    ref = _adaptive(rc.cavity_model(tmp_path, 2), rc.CAVITY_PROBLEM, 7_000)
    error = abs(rc.cavity_frequency(ref.result) / rc.CAVITY_F1 - 1)
    uniform = rc.uniform_split_error(rc.CAVITY_SPLIT, ref.cycles[-1].dofs)
    assert uniform / error >= 10


# --- smooth cases ------------------------------------------------------------


def _gauss_points(mesh: gfem.Mesh, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Points and weights of a collapsed Gauss rule (exact to degree 2n - 1)."""
    (a, wa), (b, wb), (c, wc) = (roots_jacobi(n, alpha, 0) for alpha in (2, 1, 0))
    u, v, w = np.meshgrid((a + 1) / 2, (b + 1) / 2, (c + 1) / 2, indexing="ij")
    weights = np.einsum("i,j,k->ijk", wa / 8, wb / 4, wc / 2).ravel()
    reference = np.stack(
        [u.ravel(), ((1 - u) * v).ravel(), ((1 - u) * (1 - v) * w).ravel()], axis=1
    )
    corners = np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)]
    edges = (corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)
    volume = np.abs(np.linalg.det(edges))
    points = corners[:, None, 0] + np.einsum("tij,qj->tqi", edges, reference)
    return points.reshape(-1, 3), (volume[:, None] * weights).ravel()


def _relative_energy_error(field, exact, exact_curl, eps, *, fit=False):
    """|| E_h - E ||_energy / || E_h ||_energy; ``fit`` scales E to E_h first.

    The energy norm is || curl e ||^2 + |k|^2 || e ||^2_{|eps_eff|}, with the
    field's k (complex for a lossy mode) and ``eps`` the relative eps_eff: a
    number, or a function of the points and k (in 1/m).
    """
    x, w = _gauss_points(field.mesh)
    omega = complex(np.asarray(field.omega))
    k = omega / const.c
    weight = abs(k) ** 2 * np.abs(eps(x, k) if callable(eps) else eps)
    e, curl = np.asarray(field.at(x)), -1j * omega * np.asarray(field.b().at(x))
    e0, curl0 = exact(x), exact_curl(x)

    def inner(a, ca, b, cb):
        dot = np.sum(ca * cb.conj(), axis=1) + weight * np.sum(a * b.conj(), axis=1)
        return np.sum(w * dot)

    if fit:
        alpha = inner(e, curl, e0, curl0) / inner(e0, curl0, e0, curl0)
        e0, curl0 = alpha * e0, alpha * curl0
    error = inner(e - e0, curl - curl0, e - e0, curl - curl0).real
    return math.sqrt(error / inner(e, curl, e, curl).real)


_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3, "eps_r": 4.0}
_R1, _R2 = 50.0, 30.0


def _eps_eff(eps_r, tan_delta, sigma, k):
    """eps_r (1 - i tan_delta) - i eta0 sigma / k, k in 1/m."""
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    return eps_r * (1 - 1j * tan_delta) - 1j * eta0 * sigma / k


def _line_wave(f, eps=_LINE["eps_r"]):
    """The line's exact E_z and (curl E)_y for 1 W incident at p1, functions of x.

    ``eps`` is the filling's eps_eff at f; complex beta and Z0 when lossy.
    """
    beta = 2 * math.pi * f / const.c * cmath.sqrt(eps)
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * _LINE["height"] / (_LINE["width"] * cmath.sqrt(eps))
    c, s = cmath.cos(beta * _LINE["length"]), cmath.sin(beta * _LINE["length"])
    a, b, cc = c, 1j * z0 * s, 1j * s / z0  # ABCD of the line, D = A
    s11 = (a * _R2 + b - cc * _R1 * _R2 - a * _R1) / (
        a * _R2 + b + cc * _R1 * _R2 + a * _R1
    )
    incident = math.sqrt(2)  # the peak power wave of 1 W
    v1 = math.sqrt(_R1) * incident * (1 + s11)
    i1 = incident * (1 - s11) / math.sqrt(_R1)
    h = _LINE["height"]

    def exact(x):
        out = np.zeros((len(x), 3), complex)
        bx = beta * x[:, 0]
        out[:, 2] = (v1 * np.cos(bx) - 1j * z0 * i1 * np.sin(bx)) / h
        return out

    def exact_curl(x):
        out = np.zeros((len(x), 3), complex)
        bx = beta * x[:, 0]
        out[:, 1] = beta * (v1 * np.sin(bx) + 1j * z0 * i1 * np.cos(bx)) / h
        return out

    return exact, exact_curl


def _model(tmp_path, cells, unit, **kwargs):
    path = sm.write_msh22(tmp_path / "case.msh", cells)
    return gfem.Model(gfem.Mesh.from_gmsh(path, length_unit=unit), **kwargs)


def _line_model(tmp_path, cells, materials):
    return _model(
        tmp_path,
        cells,
        "mm",
        materials=materials,
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )


_LINE_FREQUENCIES = (3e9, 8e9)


def _line_estimate(model, options):
    """The line solved at 3 and 8 GHz, exciting p1, and its field estimate."""
    problem = gfem.Driven(list(_LINE_FREQUENCIES), excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=options)
    return result, gfem.error_estimate(result, gfem.FieldEstimator(), model=model)


def _parallel_plate_line(
    tmp_path, n, options, *, tan_delta=0.0, sigma=0.0, case="parallel-plate line"
):
    filling = gfem.Material(
        eps_r=_LINE["eps_r"], tan_delta=tan_delta, conductivity=sigma
    )
    cells = sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    model = _line_model(tmp_path, cells, {"dielectric": filling})
    result, estimate = _line_estimate(model, options)
    for f, value in zip(_LINE_FREQUENCIES, estimate.values, strict=True):
        eps = _eps_eff(_LINE["eps_r"], tan_delta, sigma, 2 * math.pi * f / const.c)
        field = result.e_field(f, "p1")
        true = _relative_energy_error(field, *_line_wave(f, eps), eps)
        _assert_in(case, value, true, order=options.order)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_field_effectivity_on_a_parallel_plate_line(tmp_path, n):
    _parallel_plate_line(tmp_path, n, OPTIONS)


@pytest.mark.parametrize("n", [4, 16])
def test_the_field_effectivity_on_a_parallel_plate_line_at_order_3(tmp_path, n):
    _parallel_plate_line(tmp_path, n, ORDER_3)


def test_the_field_estimate_of_a_lossless_line_is_unchanged(tmp_path):
    # Measured before the lossy flux (ADR 0087, ticket 11 amendment): without
    # loss eps_eff = eps_r and |k| = k, so the estimate is the same number.
    cells = sm.strip_line(10.0, 2.0, 1.0, (4, 1, 1))
    filling = gfem.Material(eps_r=_LINE["eps_r"])
    model = _line_model(tmp_path, cells, {"dielectric": filling})
    _, estimate = _line_estimate(model, OPTIONS)
    assert estimate.values == pytest.approx(
        (0.0026562524133877095, 0.019399239186158542), rel=1e-12
    )
    assert np.sum(estimate.indicators**2) == pytest.approx(
        0.00019169307894270854, rel=1e-12
    )


#: The lossy line's filling loss: tan_delta 0.3, or 10 S/m.
LINE_LOSS = {"tan": {"tan_delta": 0.3}, "sigma": {"sigma": 10.0}}


@pytest.mark.parametrize(
    ("loss", "n"), [("tan", 4), ("tan", 16), ("sigma", 8), ("sigma", 32)]
)
def test_the_field_effectivity_on_a_lossy_parallel_plate_line(tmp_path, loss, n):
    # At 10 S/m and n = 32 the recovery of eps_r E measured 0.62.
    _parallel_plate_line(
        tmp_path, n, OPTIONS, case="lossy parallel-plate line", **LINE_LOSS[loss]
    )


@pytest.mark.parametrize("n", [4, 16])
@pytest.mark.parametrize("loss", list(LINE_LOSS))
def test_the_field_effectivity_on_a_lossy_parallel_plate_line_at_order_3(
    tmp_path, loss, n
):
    _parallel_plate_line(
        tmp_path, n, ORDER_3, case="lossy parallel-plate line", **LINE_LOSS[loss]
    )


def _pec_box_mode(tmp_path, m, options):
    a, b = 1e-3, 0.7e-3  # the box is 0.4 mm high; TM110 does not vary along z
    model = _model(
        tmp_path,
        sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m)),
        "mm",
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    f110 = const.c / 2 * math.hypot(1 / a, 1 / b)
    result = gfem.solve(model, gfem.Eigenmode(target=f110, n_modes=1), options=options)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)

    def exact(x):
        out = np.zeros((len(x), 3), complex)
        out[:, 2] = np.sin(np.pi * x[:, 0] / a) * np.sin(np.pi * x[:, 1] / b)
        return out

    def exact_curl(x):
        sx, cx = np.sin(np.pi * x[:, 0] / a), np.cos(np.pi * x[:, 0] / a)
        sy, cy = np.sin(np.pi * x[:, 1] / b), np.cos(np.pi * x[:, 1] / b)
        out = np.zeros((len(x), 3), complex)
        out[:, 0] = np.pi / b * sx * cy
        out[:, 1] = -np.pi / a * cx * sy
        return out

    true = _relative_energy_error(result.e_fields[0], exact, exact_curl, 1.0, fit=True)
    _assert_in("rectangular box", estimate.value, true, order=options.order)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_field_effectivity_on_a_pec_box_mode(tmp_path, m):
    _pec_box_mode(tmp_path, m, OPTIONS)


@pytest.mark.parametrize("m", [1, 2])
def test_the_field_effectivity_on_a_pec_box_mode_at_order_3(tmp_path, m):
    _pec_box_mode(tmp_path, m, ORDER_3)


# --- loss jumps --------------------------------------------------------------

#: The lossy upper layer of the two-layer box and the layered line.
UPPER_LOSS = {"tan": {"tan_delta": 0.5}, "sigma": {"conductivity": 30.0}}


def _layers_eps(lower, upper, z):
    """eps_eff of two `Material` layers, ``upper`` above height ``z`` (m)."""

    def eps(x, k):
        values = [
            _eps_eff(m.eps_r, m.tan_delta, m.conductivity, k) for m in (lower, upper)
        ]
        return np.where(x[:, 2] > z, values[1], values[0])

    return eps


def _two_layer_box(tmp_path, m, options, loss):
    # A 1 x 0.7 x 0.4 mm PEC box of two 0.2 mm layers stacked in z; the
    # TM110-like mode puts E normal to the interface, where only eps_eff E
    # has a continuous normal component.
    lower = gfem.Material(eps_r=4.0)
    upper = gfem.Material(eps_r=4.0, **loss)
    model = _model(
        tmp_path,
        sm.plate(1.0, 0.7, (("lower", 0.2), ("upper", 0.2)), (5 * m, 4 * m, m)),
        "mm",
        materials={"lower": lower, "upper": upper},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    f110 = const.c / 2 * math.hypot(1 / 1e-3, 1 / 0.7e-3) / 2  # eps_r = 4
    problem = gfem.Eigenmode(target=f110, n_modes=1)
    result = gfem.solve(model, problem, options=options)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    reference = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(order=options.order + 2, linear_solver="direct"),
    ).e_fields[0]
    omega = complex(np.asarray(reference.omega))
    true = _relative_energy_error(
        result.e_fields[0],
        lambda x: np.asarray(reference.at(x)),
        lambda x: -1j * omega * np.asarray(reference.b().at(x)),
        _layers_eps(lower, upper, 0.2e-3),
        fit=True,
    )
    _assert_in("two-layer box", estimate.value, true, order=options.order)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("loss", list(UPPER_LOSS))
def test_the_field_effectivity_on_a_two_layer_box(tmp_path, loss, m):
    # Recovering eps_r E measured 1.97 and 4.91 (tan_delta 0.5), 2.86 and
    # 7.86 (30 S/m) here, growing as the mesh is refined.
    _two_layer_box(tmp_path, m, OPTIONS, UPPER_LOSS[loss])


@pytest.mark.parametrize("loss", list(UPPER_LOSS))
def test_the_field_effectivity_on_a_two_layer_box_at_order_3(tmp_path, loss):
    _two_layer_box(tmp_path, 1, ORDER_3, UPPER_LOSS[loss])


#: The lossy top walls of the air box: loss on the boundary only.
LOSSY_WALLS = {
    "impedance": gfem.Impedance("top", resistance=377.0),
    "absorbing": gfem.Absorbing("top"),
}


def _lossy_wall_box(tmp_path, m, options, wall):
    # The 1 x 0.7 x 0.4 mm air box of the omega estimate's lossy-wall case,
    # Q about 4: the volume is lossless, the mode's omega complex.
    model = _model(
        tmp_path,
        sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m)),
        "mm",
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("side"), LOSSY_WALLS[wall]],
    )
    f110 = const.c / 2 * math.hypot(1 / 1e-3, 1 / 0.7e-3)
    problem = gfem.Eigenmode(target=f110, n_modes=1)
    result = gfem.solve(model, problem, options=options)
    assert abs(complex(np.asarray(result.omega)[0]).imag) > 0
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    reference = gfem.solve(
        model,
        problem,
        options=gfem.SolverOptions(order=options.order + 2, linear_solver="direct"),
    ).e_fields[0]
    omega = complex(np.asarray(reference.omega))
    true = _relative_energy_error(
        result.e_fields[0],
        lambda x: np.asarray(reference.at(x)),
        lambda x: -1j * omega * np.asarray(reference.b().at(x)),
        1.0,
        fit=True,
    )
    _assert_in("lossy-wall box", estimate.value, true, order=options.order)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("wall", list(LOSSY_WALLS))
def test_the_field_effectivity_on_a_lossy_wall_box(tmp_path, wall, m):
    _lossy_wall_box(tmp_path, m, OPTIONS, wall)


@pytest.mark.parametrize("wall", list(LOSSY_WALLS))
def test_the_field_effectivity_on_a_lossy_wall_box_at_order_3(tmp_path, wall):
    _lossy_wall_box(tmp_path, 1, ORDER_3, wall)


#: The measured field effectivity of a lumped port across a loss jump: below
#: the quoted range, so that configuration is not claimed (recipe Limits).
LOSS_JUMP_WITNESS = (0.84, 0.90)


@pytest.mark.parametrize(
    "loss", [{"tan_delta": 0.5}, {"conductivity": 1.0}], ids=["tan", "sigma"]
)
def test_the_field_estimate_under_a_lumped_port_across_a_loss_jump(tmp_path, loss):
    # The parallel-plate line in two 0.5 mm layers, the upper one lossy, its
    # lumped ports spanning both: the port face meets the interface in a
    # singularity. At order 3 on 48 tetrahedra, against an order-7 solve on
    # the same mesh, the estimate stays below the error (0.85-0.89); it rises
    # under refinement (order 2: 0.93, 0.95, 0.96 at n = 4, 8, 12, 3 GHz).
    lower = gfem.Material(eps_r=_LINE["eps_r"])
    upper = gfem.Material(eps_r=_LINE["eps_r"], **loss)
    cells = sm.strip_line(
        10.0, 2.0, 1.0, (4, 1, 1), layers=(("lower", 0.5), ("upper", 0.5))
    )
    model = _line_model(tmp_path, cells, {"lower": lower, "upper": upper})
    result, estimate = _line_estimate(model, ORDER_3)
    reference = gfem.solve(
        model,
        gfem.Driven(list(_LINE_FREQUENCIES), excite="p1", keep_fields=True),
        options=gfem.SolverOptions(order=7, linear_solver="direct"),
    )
    low, high = LOSS_JUMP_WITNESS
    for f, value in zip(_LINE_FREQUENCIES, estimate.values, strict=True):
        exact = reference.e_field(f, "p1")
        omega = complex(np.asarray(exact.omega))
        true = _relative_energy_error(
            result.e_field(f, "p1"),
            lambda x, exact=exact: np.asarray(exact.at(x)),
            lambda x, exact=exact, omega=omega: (
                -1j * omega * np.asarray(exact.b().at(x))
            ),
            _layers_eps(lower, upper, 0.5e-3),
        )
        assert low <= value / true <= high, (f, value / true)


ORDER_5 = gfem.SolverOptions(order=5, linear_solver="direct")
ORDER_6 = gfem.SolverOptions(order=6, linear_solver="direct")
_CIRCLE_C = math.pi * const.epsilon_0 * 0.5e-6 / (2 * math.log(2.0))  # a quarter


def _coax_case(model, options=OPTIONS, reference=ORDER_5):
    """(C, its field estimate, C of a higher-order solve on the same mesh)."""
    result = gfem.solve(model, rc.COAX_PROBLEM, options=options)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    fine = gfem.solve(model, rc.COAX_PROBLEM, options=reference)
    c, c5 = float(result.capacitance[0, 0]), float(fine.capacitance[0, 0])
    return c, estimate.value, c5


def _circular_coax_model(directory, k):
    return _model(
        directory,
        sm.coax_quarter(1.0, 2.0, 0.5, n_r=2 * k, n_theta=4 * k),
        "um",
        materials={"dielectric": gfem.Material()},
    )


@pytest.fixture(scope="module")
def circular_coax(tmp_path_factory):
    """Per k: the circular coax quarter's `_coax_case`."""
    return {
        k: _coax_case(_circular_coax_model(tmp_path_factory.mktemp(f"circular{k}"), k))
        for k in (2, 4, 8)
    }


@pytest.fixture(scope="module")
def circular_coax_order_3(tmp_path_factory):
    """Per k: the order-3 `_coax_case`, against an order-6 solve."""
    out = {}
    for k in (2, 4):
        model = _circular_coax_model(tmp_path_factory.mktemp(f"circular3_{k}"), k)
        out[k] = _coax_case(model, ORDER_3, ORDER_6)
    return out


@pytest.mark.parametrize("k", [2, 4, 8])
def test_the_field_effectivity_on_a_circular_coax(circular_coax, k):
    c, estimate, c5 = circular_coax[k]
    _assert_in("circular coax", estimate, math.sqrt((c - c5) / c5))


@pytest.mark.parametrize("k", [2, 4, 8])
def test_the_field_estimate_does_not_see_the_faceting(circular_coax, k):
    # Against the true circle the error includes the faceting, which mesh
    # refinement never removes: the estimate stays below that error.
    c, estimate, _ = circular_coax[k]
    assert estimate / math.sqrt((c - _CIRCLE_C) / _CIRCLE_C) < 1.0


@pytest.mark.parametrize("k", [2, 4])
def test_the_field_effectivity_on_a_circular_coax_at_order_3(circular_coax_order_3, k):
    c, estimate, c6 = circular_coax_order_3[k]
    _assert_in("circular coax", estimate, math.sqrt((c - c6) / c6), order=3)


@pytest.mark.parametrize("k", [2, 4])
def test_the_field_estimate_does_not_see_the_faceting_at_order_3(
    circular_coax_order_3, k
):
    c, estimate, _ = circular_coax_order_3[k]
    assert estimate / math.sqrt((c - _CIRCLE_C) / _CIRCLE_C) < 1.0


@pytest.mark.parametrize("fill", list(rc.ANISOTROPIC_COAX))
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_field_effectivity_on_an_anisotropic_coax(tmp_path, k, fill):
    c, estimate, c5 = _coax_case(rc.anisotropic_coax_model(tmp_path, k, fill))
    _assert_in("anisotropic coax", estimate, math.sqrt(abs(c - c5) / c5))
