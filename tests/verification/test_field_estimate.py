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

Each case's effectivity must lie in its range of `FIELD_EFFECTIVITY_BY_CASE`;
`FIELD_EFFECTIVITY` (the lowest low and the highest high) is the range the
docs quote. The step line's proxy ratio is not an effectivity and is
asserted on its own.

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
"""

import math

import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm
from scipy.special import roots_jacobi

import gfem

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")

#: Asserted field effectivity range per case (measured, with margin).
FIELD_EFFECTIVITY_BY_CASE = {
    "square coax": (0.9, 1.6),
    "circular coax": (0.9, 1.6),
    "L cavity": (0.9, 1.5),
    "parallel-plate line": (0.9, 1.6),
    "rectangular box": (0.9, 1.6),
    "anisotropic coax": (0.9, 1.8),
}

#: The range the docs quote: the lowest low and the highest high.
FIELD_EFFECTIVITY = (
    min(low for low, _ in FIELD_EFFECTIVITY_BY_CASE.values()),
    max(high for _, high in FIELD_EFFECTIVITY_BY_CASE.values()),
)


def _assert_in(case, estimate, error):
    low, high = FIELD_EFFECTIVITY_BY_CASE[case]
    assert low <= estimate / error <= high, (case, estimate / error)


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


def _relative_energy_error(field, exact, exact_curl, eps_r, *, fit=False):
    """|| E_h - E ||_energy / || E_h ||_energy; ``fit`` scales E to E_h first."""
    x, w = _gauss_points(field.mesh)
    omega = complex(np.asarray(field.omega))
    k2 = (omega.real / const.c) ** 2
    e, curl = np.asarray(field.at(x)), -1j * omega * np.asarray(field.b().at(x))
    e0, curl0 = exact(x), exact_curl(x)

    def inner(a, ca, b, cb):
        dot = np.sum(ca * cb.conj(), axis=1) + k2 * eps_r * np.sum(a * b.conj(), axis=1)
        return np.sum(w * dot)

    if fit:
        alpha = inner(e, curl, e0, curl0) / inner(e0, curl0, e0, curl0)
        e0, curl0 = alpha * e0, alpha * curl0
    error = inner(e - e0, curl - curl0, e - e0, curl - curl0).real
    return math.sqrt(error / inner(e, curl, e, curl).real)


_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3, "eps_r": 4.0}
_R1, _R2 = 50.0, 30.0


def _line_wave(f):
    """The line's exact E_z and (curl E)_y for 1 W incident at p1, functions of x."""
    beta = 2 * math.pi * f / const.c * math.sqrt(_LINE["eps_r"])
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * _LINE["height"] / (_LINE["width"] * math.sqrt(_LINE["eps_r"]))
    c, s = math.cos(beta * _LINE["length"]), math.sin(beta * _LINE["length"])
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


@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_field_effectivity_on_a_parallel_plate_line(tmp_path, n):
    model = _model(
        tmp_path,
        sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1)),
        "mm",
        materials={"dielectric": gfem.Material(eps_r=_LINE["eps_r"])},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    frequencies = (3e9, 8e9)
    problem = gfem.Driven(list(frequencies), excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    for f, value in zip(frequencies, estimate.values, strict=True):
        field = result.e_field(f, "p1")
        true = _relative_energy_error(field, *_line_wave(f), _LINE["eps_r"])
        _assert_in("parallel-plate line", value, true)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_field_effectivity_on_a_pec_box_mode(tmp_path, m):
    a, b = 1e-3, 0.7e-3  # the box is 0.4 mm high; TM110 does not vary along z
    model = _model(
        tmp_path,
        sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m)),
        "mm",
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    f110 = const.c / 2 * math.hypot(1 / a, 1 / b)
    result = gfem.solve(model, gfem.Eigenmode(target=f110, n_modes=1), options=OPTIONS)
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
    _assert_in("rectangular box", estimate.value, true)


ORDER_5 = gfem.SolverOptions(order=5, linear_solver="direct")


def _coax_case(model):
    """(C, its field estimate, C of an order-5 solve on the same mesh)."""
    result = gfem.solve(model, rc.COAX_PROBLEM, options=OPTIONS)
    estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
    fine = gfem.solve(model, rc.COAX_PROBLEM, options=ORDER_5)
    c, c5 = float(result.capacitance[0, 0]), float(fine.capacitance[0, 0])
    return c, estimate.value, c5


@pytest.fixture(scope="module")
def circular_coax(tmp_path_factory):
    """Per k: the circular coax quarter's `_coax_case`."""
    out = {}
    for k in (2, 4, 8):
        directory = tmp_path_factory.mktemp(f"circular{k}")
        model = _model(
            directory,
            sm.coax_quarter(1.0, 2.0, 0.5, n_r=2 * k, n_theta=4 * k),
            "um",
            materials={"dielectric": gfem.Material()},
        )
        out[k] = _coax_case(model)
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
    exact = math.pi * const.epsilon_0 * 0.5e-6 / (2 * math.log(2.0))  # a quarter
    assert estimate / math.sqrt((c - exact) / exact) < 1.0


@pytest.mark.parametrize("fill", list(rc.ANISOTROPIC_COAX))
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_field_effectivity_on_an_anisotropic_coax(tmp_path, k, fill):
    c, estimate, c5 = _coax_case(rc.anisotropic_coax_model(tmp_path, k, fill))
    _assert_in("anisotropic coax", estimate, math.sqrt(abs(c - c5) / c5))
