"""The objective estimates' effectivity, on known answers (ADRs 0129, 0143).

A `gfem.QuantityEstimator(objective=J)` estimates how far an objective
``J(result, model)`` of a solved result is from its value on the same
geometry with an infinitely fine mesh. Its effectivity is the estimate over
``|J(S_h) - J(S_ref)|``, with ``J`` evaluated on the result whose S is
replaced by the reference S; every estimate must lie in its case's range of
`EFFECTIVITY_BY_KIND`, keyed by the estimate's kind (`ErrorEstimate.kind`).
Order 2, direct solves.

A row is not asserted where the reference's own uncertainty, carried into
J (sum |ct| bound plus the remainder of the bound), exceeds 20 % of the
error, or where the error is below 1e-12 (the rule of the S cases).

Objectives of S (kind ``objective of s``, the C form of ADR 0129):

- lossy line: the parallel-plate line between 50 and 30 ohm lumped ports,
  eps_r 4 with tan_delta 0.3, at 3, 8, 15, 25 GHz on n = 4, 8, 16, against
  its ABCD closed form: |S21|^2 at each frequency, the band mean of
  1 - |S21|^2, the mean of |S11|^2, |S11|^2 at each frequency, the sheet
  objective mean((|S21|^2 - 1/2)^2 + |S11|^2), the absorbed power
  mean(1 - |S11|^2 - |S21|^2), mean |S11|^2 + |S22|^2 and mean |S21|^2 +
  |S12|^2;
- block guide (`refinement_cases.block_guide_model`, n = 4; n = 8 in the
  reference tier): the conversion |S(b2, a1)|^2, its total with
  |S(a2, a1)|^2, |S(b1, a1)|^2, 1 - |S(b1, a1)|^2 - |S(b2, a2)|^2,
  |S(a1, a1)|^2 + |S(a2, a2)|^2 and a signed weighted sum over six entries
  of two sources, against `refinement_cases.BLOCK_GUIDE_S`;
- lossy guides (`refinement_cases.lossy_guide_model`, uniform, slab and
  step, tan_delta 0.05 and 0.3, wave ports de-embedded by 0 and 1 mm,
  n = 4): |S21|^2, |S11|^2, the absorbed power 1 - |S11|^2 - |S21|^2,
  (|S21|^2 + |S12|^2) / 2 and |S11|^2 + |S22|^2, against
  `refinement_cases.LOSSY_GUIDE_S`; the slab and step read a hybrid lossy
  port mode, whose dual is the solved adjoint;
- deep metal (`contrast_cases.metal_floor_model`, wedge and flat at Re eps
  -29.7, n = 1, 2): |S21|^2 against `contrast_cases.METAL_FLOOR_S`.

Objectives of omega (kind ``objective of omega``, ADR 0131 amendment):
free duals, the Re rule per mode read plus the Hessian remainder, against
exact omegas:

- L cavity (`refinement_cases.cavity_model`, eps_r 4, lossless and 5 S/m,
  n = 2, 4, 8): the frequency in GHz, (f/f0 - 1)^2 off target (f0 0.99 of
  the exact f) and on target (f0 the solved f), and 1/q;
- lumped-R line (50 and 30 ohm ports, n = 4, 16, 64): 1/q, and the
  detuning off (0.99 f_h) and on target;
- two-mode box (eps_r 4, 0.5 S/m, modes 110 and 210, m = 1, 2): the ratio
  f2/f1 off target, the split in GHz and the sum of 1/q.

Objectives of participation (kind ``objective of participation``, ADR
0142): the mode-field dual with its flux equilibrated, plus the remainder
R2, on a 1 x 0.7 x 0.2 mm box at 140 GHz with eps_r 4 in a region:

- slab (0.3 < x < 0.6 across the section, m = 1, 2): PEC, tan_delta 0.05,
  and an `Impedance` end wall of 50 and 377 ohm/sq, against the exact E_z
  modes (a transcendental equation);
- block (0.3-0.6 x 0.2-0.5 x 0-0.1 mm, 3D corners, m = 2, 3): PEC and a
  377 ohm/sq `Impedance` top, against `BLOCK_PARTICIPATION` (Richardson;
  rows within 5 times its uncertainty are not asserted).

Overestimates (`OVERESTIMATES`), asserted >= 0.9 and quoted on their own:
an objective at its target, (|S|^2 - |S_h|^2)^2 and |S - S_h|^2 on the
first entry of the lossy line and of the block guide, whose cotangent is
zero, so that only the Hessian remainder is left; several modes at their
target, (f2/f1 - its solved value)^2 on the two-mode box; participation
times frequency, p f on the slab, whose two errors cancel in J.

A robust max over frequencies or entries is not estimated: it branches on
what it reads and raises ``"non-smooth objective"``
(`tests/unit/test_refine_objective.py`).

Measured (ostau, 2 threads):

    lossy line            1.90 - 38.2
    block guide           7.82 - 42.3   (n = 4)
    lossy guides          2.08 - 175    (uniform 0.05: 175, 0.3: 2.45 - 26.8;
                                         slab 2.21 - 49.4; step 2.08 - 120)
    deep metal            2.53 - 4.77
    at its target         3.56 - 53.7   (lossy line n = 4, block guide; on
                                         the finer lines the error is under
                                         1e-12)
    L cavity (omega)      2.15 - 14.4
    lumped-R line         1.04 - 36.8   (1.04: detuning at n = 64)
    two-mode box          8.53 - 16.0
    modes at target       162 - 176

The lossy guide's slab at tan_delta 0.05 without offset is not run: its
reference uncertainty exceeds 20 % of every row's error. The top, 175, is
the near-lossless uniform guide's |S21|^2, whose error is second order and
mostly a phase error of S21 that the magnitude form cannot see (ADR 0129).
"""

import cmath
import dataclasses
import math

import contrast_cases as cc
import jax.numpy as jnp
import numpy as np
import pytest
import refinement_cases as rc
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem import _objective

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")

#: Asserted effectivity at order 2, per estimate kind (`ErrorEstimate.kind`),
#: then per case (measured, with margin). The docs quote each kind's span.
EFFECTIVITY_BY_KIND = {
    "objective of s": {
        "lossy line": (0.9, 45.0),
        "block guide": (0.9, 50.0),
        "lossy guides": (0.9, 200.0),
        "deep metal": (0.9, 8.0),
    },
    "objective of omega": {
        "L cavity": (0.9, 40.0),
        "lumped-R line": (0.9, 40.0),
        "two-mode box": (0.9, 40.0),
    },
    "objective of participation": {
        "slab": (0.9, 6.0),
        "block": (0.9, 6.0),
    },
}

#: Cases asserted >= 0.9 whose high end the docs quote on its own (Limits)
#: instead of widening their kind's range: name -> (kind, (low, high)).
OVERESTIMATES = {
    "objective at its target": ("objective of s", (0.9, 2e5)),
    "several modes at their target": ("objective of omega", (0.9, 200.0)),
    "participation times frequency": ("objective of participation", (0.9, 1.5e3)),
}

#: Configurations that are not claimed: name -> (kind, (low, high)).
WITNESSES: dict[str, tuple[str, tuple[float, float]]] = {}


def _a2(s):
    """|s|^2, smooth at 0."""
    return jnp.real(s) ** 2 + jnp.imag(s) ** 2


def _with_s(result, values):
    """``result`` with its S values replaced (the reference's)."""
    return dataclasses.replace(
        result, s=dataclasses.replace(result.s, values=jnp.asarray(values))
    )


def _effectivity(model, result, reference, bound, objective):
    """The estimate and its effectivity, or ``None`` for a row not asserted.

    ``reference`` is the reference S, shaped as ``result.s.values`` (NaN
    where there is none); ``bound`` its uncertainty per entry.
    """
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    error = float(objective(result, model)) - float(
        objective(_with_s(result, reference), model)
    )
    assert math.isfinite(error)
    classified = _objective.classify(objective, result, model)
    ct = np.array([abs(r.cotangent) for r in classified.reads])
    uncertainty = bound * ct.sum() + 0.5 * classified.curvature.sum() * bound**2
    if abs(error) < 1e-12 or uncertainty > 0.2 * abs(error):
        return estimate, None
    return estimate, estimate.value / abs(error)


def _assert_rows(kind, case, rows):
    ratios = [ratio for _, ratio in rows if ratio is not None]
    assert ratios, case  # at least one row asserted
    low, high = EFFECTIVITY_BY_KIND[kind][case]
    for estimate, ratio in rows:
        assert estimate.kind == kind, (case, estimate.kind)
        if ratio is not None:
            assert low <= ratio <= high, (case, ratio)


def _assert_overestimate(name, rows):
    kind, (low, high) = OVERESTIMATES[name]
    for estimate, ratio in rows:
        assert estimate.kind == kind
        if ratio is not None:
            assert low <= ratio <= high, (name, ratio)


def _s(result, to, source, f=None):
    s = result.s[to, source]
    return s if f is None else s[f]


def _on_target(result, to, source):
    """The two objectives at their target on ``s[to, source]`` at frequency 0."""
    s_h = complex(np.asarray(result.s[to, source])[0])

    def distance(r, _model):
        return _a2(_s(r, to, source, 0) - s_h)

    def power(r, _model):
        return (_a2(_s(r, to, source, 0)) - abs(s_h) ** 2) ** 2

    return [distance, power]


# --- lossy line -----------------------------------------------------------------

_LINE = {"length": 10e-3, "height": 1e-3, "width": 2e-3}
_R1, _R2 = 50.0, 30.0
LINE_FREQUENCIES = (3e9, 8e9, 15e9, 25e9)
_LOSSY_EPS = 4.0 * (1 - 0.3j)


def _line_s(f, eps_r):
    """The line's S between its 50 and 30 ohm ports, from its ABCD matrix."""
    beta = 2 * math.pi * f / const.c * cmath.sqrt(eps_r)
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 / cmath.sqrt(eps_r) * _LINE["height"] / _LINE["width"]
    c, s = cmath.cos(beta * _LINE["length"]), cmath.sin(beta * _LINE["length"])
    a, b, cc_, d = c, 1j * z0 * s, 1j * s / z0, c
    den = a * _R2 + b + cc_ * _R1 * _R2 + d * _R1
    k = math.sqrt(_R1 * _R2)
    s11 = (a * _R2 + b - cc_ * _R1 * _R2 - d * _R1) / den
    s22 = (-a * _R2 + b - cc_ * _R1 * _R2 + d * _R1) / den
    return np.array([[s11, 2 * k * (a * d - b * cc_) / den], [2 * k / den, s22]])


def _lossy_line(directory, n):
    path = sm.write_msh22(
        directory / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=0.3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    problem = gfem.Driven(list(LINE_FREQUENCIES), keep_fields=True)
    return model, gfem.solve(model, problem, options=OPTIONS)


def _per_frequency(to, source):
    return [(lambda r, _m, f=f: _a2(_s(r, to, source, f))) for f in range(4)]


def _line_objectives():
    s21, s11 = ("p2", "p1"), ("p1", "p1")
    return [
        *_per_frequency(*s21),
        lambda r, _m: jnp.mean(1 - _a2(_s(r, *s21))),
        lambda r, _m: jnp.mean(_a2(_s(r, *s11))),
        *_per_frequency(*s11),
        lambda r, _m: jnp.mean((_a2(_s(r, *s21)) - 0.5) ** 2 + _a2(_s(r, *s11))),
        lambda r, _m: jnp.mean(1 - _a2(_s(r, *s11)) - _a2(_s(r, *s21))),
        lambda r, _m: jnp.mean(_a2(_s(r, *s11)) + _a2(_s(r, "p2", "p2"))),
        lambda r, _m: jnp.mean(_a2(_s(r, *s21)) + _a2(_s(r, "p1", "p2"))),
    ]


@pytest.fixture(scope="module", params=[4, 8, 16])
def lossy_line(request, tmp_path_factory):
    model, result = _lossy_line(tmp_path_factory.mktemp("line"), request.param)
    reference = np.array([_line_s(f, _LOSSY_EPS) for f in LINE_FREQUENCIES])
    return model, result, reference


def test_objective_of_s_on_the_lossy_line(lossy_line):
    model, result, reference = lossy_line
    rows = [
        _effectivity(model, result, reference, 1e-12, objective)
        for objective in _line_objectives()
    ]
    _assert_rows("objective of s", "lossy line", rows)


def test_objective_of_s_at_its_target_on_the_lossy_line(lossy_line):
    model, result, reference = lossy_line
    rows = [
        _effectivity(model, result, reference, 1e-12, objective)
        for objective in _on_target(result, "p2", "p1")
    ]
    _assert_overestimate("objective at its target", rows)


# --- block guide ------------------------------------------------------------------

_BLOCK_WEIGHTS = (1.0, 0.5, 2.0, -1.0, 1.0, 0.3)
_BLOCK_ENTRIES = (
    ("b1", "a1"),
    ("b2", "a2"),
    ("b2", "a1"),
    ("a2", "a1"),
    ("a1", "a1"),
    ("a2", "a2"),
)


def _block_objectives():
    def entry(r, to, source):
        return _s(r, to, source, 0)

    return [
        lambda r, _m: _a2(entry(r, "b2", "a1")),
        lambda r, _m: _a2(entry(r, "b2", "a1")) + _a2(entry(r, "a2", "a1")),
        lambda r, _m: _a2(entry(r, "b1", "a1")),
        lambda r, _m: 1 - _a2(entry(r, "b1", "a1")) - _a2(entry(r, "b2", "a2")),
        lambda r, _m: _a2(entry(r, "a1", "a1")) + _a2(entry(r, "a2", "a2")),
        lambda r, _m: sum(
            w * _a2(entry(r, *e))
            for w, e in zip(_BLOCK_WEIGHTS, _BLOCK_ENTRIES, strict=True)
        ),
    ]


def _reference(result, entries):
    """The result's S shape filled from ``entries[(to, from)]``, NaN elsewhere."""
    s = result.s
    out = np.full(np.shape(s.values), np.nan + 0j)
    for (to, source), value in entries.items():
        out[0, s.ports.index(to), s.excited.index(source)] = value
    return out


def _block_guide(directory, n):
    model = rc.block_guide_model(directory, n)
    problem = gfem.Driven([rc.BLOCK_GUIDE_FREQUENCY], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    return model, result, _reference(result, rc.BLOCK_GUIDE_S)


@pytest.fixture(scope="module")
def block_guide(tmp_path_factory):
    return _block_guide(tmp_path_factory.mktemp("block"), 4)


def _block_rows(model, result, reference):
    return [
        _effectivity(model, result, reference, 6.4e-5, objective)
        for objective in _block_objectives()
    ]


def test_objective_of_s_on_the_block_guide(block_guide):
    _assert_rows("objective of s", "block guide", _block_rows(*block_guide))


@pytest.mark.reference
def test_objective_of_s_on_the_block_guide_at_n_8(tmp_path):
    rows = _block_rows(*_block_guide(tmp_path, 8))
    _assert_rows("objective of s", "block guide", rows)


def test_objective_of_s_at_its_target_on_the_block_guide(block_guide):
    model, result, reference = block_guide
    rows = [
        _effectivity(model, result, reference, 6.4e-5, objective)
        for objective in _on_target(result, "b2", "a1")
    ]
    _assert_overestimate("objective at its target", rows)


# --- lossy guides -----------------------------------------------------------------


def _guide_objectives():
    def entry(r, to, source):
        return _s(r, to, source, 0)

    return [
        lambda r, _m: _a2(entry(r, "p2", "p1")),
        lambda r, _m: _a2(entry(r, "p1", "p1")),
        lambda r, _m: 1 - _a2(entry(r, "p1", "p1")) - _a2(entry(r, "p2", "p1")),
        lambda r, _m: (_a2(entry(r, "p2", "p1")) + _a2(entry(r, "p1", "p2"))) / 2,
        lambda r, _m: _a2(entry(r, "p1", "p1")) + _a2(entry(r, "p2", "p2")),
    ]


@pytest.mark.parametrize(
    ("kind", "tan_delta", "offset"),
    [
        (kind, tan_delta, offset)
        for kind in ("uniform", "slab", "step")
        for tan_delta in (0.05, 0.3)
        for offset in (0, 1)
        if (kind, tan_delta, offset) != ("slab", 0.05, 0)
    ],
)
def test_objective_of_s_on_the_lossy_guides(tmp_path, kind, tan_delta, offset):
    model = rc.lossy_guide_model(tmp_path, 4, kind, tan_delta, offset * 1e-3)
    problem = gfem.Driven([rc.LOSSY_GUIDE_FREQUENCY], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    reference = _reference(result, rc.LOSSY_GUIDE_S[kind, tan_delta, offset])
    bound = 1e-6 if kind == "uniform" else 2e-4
    rows = [
        _effectivity(model, result, reference, bound, objective)
        for objective in _guide_objectives()
    ]
    _assert_rows("objective of s", "lossy guides", rows)


# --- deep metal ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("geometry", ["wedge", "flat"])
def test_objective_of_s_on_deep_metal(tmp_path, geometry, n):
    model = cc.metal_floor_model(tmp_path, geometry, n)
    problem = gfem.Driven([cc.METAL_DEEP], keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    reference = _reference(result, cc.METAL_FLOOR_S[geometry, n, cc.METAL_DEEP])
    row = _effectivity(
        model, result, reference, 0.0, lambda r, _m: _a2(_s(r, "p2", "p1", 0))
    )
    _assert_rows("objective of s", "deep metal", [row])


# --- objectives of omega ------------------------------------------------------------

_ETA0 = const.mu_0 * const.c


def _lossy_omega(k0_squared, eps_c, sigma, near):
    """The root of eps_c k^2 - i eta0 sigma k - k0^2 = 0 nearest ``near``, rad/s."""
    roots = np.roots([eps_c, -1j * _ETA0 * sigma, -k0_squared]) * const.c
    return complex(roots[np.argmin(abs(roots - near))])


def _inverse_q(omega):
    return 2 * jnp.abs(jnp.imag(omega)) / jnp.abs(omega)


def _hz(omega):
    return jnp.real(omega) / (2 * jnp.pi)


def _omega_rows(model, result, exact, objectives):
    """Per objective, its estimate and effectivity against the exact omegas.

    ``exact`` maps a mode index of ``result`` to its exact omega; the error
    is J on the result minus J with those omegas put in.
    """
    omega = np.asarray(result.omega).copy()
    for mode, value in exact.items():
        omega[mode] = value
    reference = dataclasses.replace(result, omega=jnp.asarray(omega))
    rows = []
    for objective in objectives:
        estimate = gfem.error_estimate(
            result, gfem.QuantityEstimator(objective=objective), model=model
        )
        error = float(objective(result, model)) - float(objective(reference, model))
        assert error != 0
        rows.append((estimate, estimate.value / abs(error)))
    return rows


def _detunings(f0, f_h, mode=0):
    """(f/f0 - 1)^2 off target (f0) and on target (the solved f_h)."""

    def off(r, _model):
        return (_hz(r.omega[mode]) / f0 - 1) ** 2

    def on(r, _model):
        return (_hz(r.omega[mode]) / f_h - 1) ** 2

    return [off, on]


@pytest.mark.parametrize("sigma", [0.0, 5.0])
@pytest.mark.parametrize("n", [2, 4, 8])
def test_objective_of_omega_on_the_l_cavity(tmp_path, n, sigma):
    # Singular corner, eps_r 4, lossless or 5 S/m: exact omega (the fill's
    # scaling of the L cavity's eigenvalue). Frequency, detuning off and on
    # target, and 1/q on the lossy fill.
    lossless = rc.cavity_model(tmp_path, n)
    fill = gfem.Material(eps_r=4.0, conductivity=sigma)
    model = gfem.Model(
        lossless.mesh, materials={"air": fill}, boundaries=lossless.boundaries
    )
    near = 2 * math.pi * rc.CAVITY_F1 / 2
    exact = _lossy_omega(rc.CAVITY_LAMBDA1 * 1e6, 4.0, sigma, near)
    problem = gfem.Eigenmode(target=exact.real / (2 * math.pi), n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    f_h = float(_hz(np.asarray(result.omega)[0]))
    objectives = [
        lambda r, _m: _hz(r.omega[0]) * 1e-9,
        *_detunings(0.99 * exact.real / (2 * math.pi), f_h),
    ]
    if sigma:
        objectives.append(lambda r, _m: _inverse_q(r.omega[0]))
    rows = _omega_rows(model, result, {0: exact}, objectives)
    _assert_rows("objective of omega", "L cavity", rows)


#: The parallel-plate line closed by 50 and 30 ohm lumped ports (Q 1.8).
_LINE_EPS = 4.0


def _line_resonance(near):
    """The resonance nearest ``near``, rad/s: Gamma1 Gamma2 e^{-2j beta L} = 1."""
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 * _LINE["height"] / (_LINE["width"] * math.sqrt(_LINE_EPS))
    gammas = ((_R1 - z0) / (_R1 + z0)) * ((_R2 - z0) / (_R2 + z0))
    speed = const.c / math.sqrt(_LINE_EPS)
    n = round(near.real * _LINE["length"] / (speed * math.pi))
    return complex((math.pi * n - 0.5j * np.log(gammas)) * speed / _LINE["length"])


@pytest.mark.parametrize("n", [4, 16, 64])
def test_objective_of_omega_on_the_lumped_resistor_line(tmp_path, n):
    # The only loss is in the two port resistors: closed-form omega. 1/q,
    # detuning off (0.99 f_h) and on target.
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=_LINE_EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=_R1),
            gfem.LumpedPort("p2", "port2", "+z", resistance=_R2),
        ],
    )
    problem = gfem.Eigenmode(target=7.5e9, n_modes=1)
    result = gfem.solve(model, problem, options=OPTIONS)
    omega = complex(np.asarray(result.omega)[0])
    f_h = omega.real / (2 * math.pi)
    objectives = [
        lambda r, _m: _inverse_q(r.omega[0]),
        *_detunings(0.99 * f_h, f_h),
    ]
    rows = _omega_rows(model, result, {0: _line_resonance(omega)}, objectives)
    _assert_rows("objective of omega", "lumped-R line", rows)


_BOX = (1e-3, 0.7e-3, 0.4e-3)


def _two_mode_box(directory, m):
    """The box filled with eps_r 4, 0.5 S/m: its modes 110 and 210, exact."""
    cells = sm.plate(1.0, 0.7, (("air", 0.4),), (5 * m, 4 * m, 2 * m))
    path = sm.write_msh22(directory / "box.msh", cells)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"air": gfem.Material(eps_r=4.0, conductivity=0.5)},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top", "side")],
    )
    exact = []
    for mode in ((1, 1, 0), (2, 1, 0)):
        k0_squared = math.pi**2 * sum(
            (i / a) ** 2 for i, a in zip(mode, _BOX, strict=True)
        )
        near = const.c * math.sqrt(k0_squared / 4.0)
        exact.append(_lossy_omega(k0_squared, 4.0, 0.5, near))
    problem = gfem.Eigenmode(target=exact[0].real / (2 * math.pi), n_modes=2)
    result = gfem.solve(model, problem, options=OPTIONS)
    omega = np.asarray(result.omega)
    by_mode = {
        i: exact[int(np.argmin(abs(np.asarray(exact) - omega[i])))] for i in range(2)
    }
    return model, result, by_mode


def _ratio(r):
    return jnp.real(r.omega[1]) / jnp.real(r.omega[0])


@pytest.mark.parametrize("m", [1, 2])
def test_objective_of_omega_on_two_modes_of_a_box(tmp_path, m):
    # Two modes read together, each followed on its own track: the ratio
    # f2/f1 off target, the split in GHz and the sum of 1/q.
    model, result, exact = _two_mode_box(tmp_path, m)
    ratio_h = float(_ratio(result))
    objectives = [
        lambda r, _m: (_ratio(r) - 1.02 * ratio_h) ** 2,
        lambda r, _m: (_hz(r.omega[1]) - _hz(r.omega[0])) * 1e-9,
        lambda r, _m: _inverse_q(r.omega[0]) + _inverse_q(r.omega[1]),
    ]
    rows = _omega_rows(model, result, exact, objectives)
    _assert_rows("objective of omega", "two-mode box", rows)


@pytest.mark.parametrize("m", [1, 2])
def test_objective_of_omega_with_two_modes_at_their_target(tmp_path, m):
    # (f2/f1 - its solved value)^2: zero cotangent, the remainder alone
    model, result, exact = _two_mode_box(tmp_path, m)
    ratio_h = float(_ratio(result))
    rows = _omega_rows(model, result, exact, [lambda r, _m: (_ratio(r) - ratio_h) ** 2])
    _assert_overestimate("several modes at their target", rows)


# --- objectives of participation -----------------------------------------------------

#: The participation boxes (mm): 1 x 0.7 x 0.2, eps_r 4 in "block".
_PBOX = (1.0, 0.7, 0.2)
_PEPS = 4.0
_PTARGET = 140e9
#: The block's extent per axis, mm: the slab spans the section.
_SLAB = ((0.3, 0.6), (-1.0, 2.0), (-1.0, 2.0))
_BLOCK = ((0.3, 0.6), (0.2, 0.5), (0.0, 0.1))
#: The block's participation of its two modes by Richardson extrapolation
#: of order-2 solves on m = 2, 3, 4 (fitted rate 1.58 to 1.65), with its
#: uncertainty (the larger difference to the rate-4/3 extrapolation and to
#: the m = 1, 2, 3 fit), by variant and mode: (p, uncertainty).
BLOCK_PARTICIPATION = {
    ("pec", 0): (0.19092525691418702, 1.7021294958671773e-4),
    ("pec", 1): (0.1612304796735319, 1.0444788009938977e-4),
    ("impedance", 0): (0.10020139077778502, 1.1157790198222661e-4),
}


def _participation_box(directory, m, block, *, top=None, end=None, tan_delta=0.0):
    """The box with eps_r 4 in ``block``, 0.1 / m mm cells, PEC but ``top``/``end``."""
    edges = tuple(np.linspace(0, a, round(a * 10 * m) + 1) for a in _PBOX)

    def volume(c):
        inside = all(lo < v < hi for v, (lo, hi) in zip(c, block, strict=True))
        return "block" if inside else "air"

    def surface(c):
        tol = 1e-9
        if abs(c[2] - _PBOX[2]) < tol:
            return "top"
        if abs(c[0] - _PBOX[0]) < tol:
            return "end"
        on = (
            abs(c[0]) < tol,
            abs(c[1]) < tol,
            abs(c[1] - _PBOX[1]) < tol,
            abs(c[2]) < tol,
        )
        return "walls" if any(on) else ""

    path = sm.write_msh22(directory / "pbox.msh", sm.structured(edges, volume, surface))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "air": gfem.Material(),
            "block": gfem.Material(eps_r=_PEPS, tan_delta=tan_delta),
        },
        boundaries=[gfem.PEC("walls"), top or gfem.PEC("top"), end or gfem.PEC("end")],
    )


def _slab_exact(k_h, eps_c, resistance=None):
    """The slab box's E_z mode nearest ``k_h`` (mm^-1): (k, participation).

    E_z = X(x) sin(beta y), beta = n pi / 0.7; X is sin(q1 x) in the air
    before the slab, a cosine-sine pair in it, and closes at x = 1 on PEC or
    on the Robin condition of an `Impedance` of ``resistance`` ohm/sq.
    """
    from scipy import integrate, optimize

    a, b, _ = _PBOX
    x1, x2 = _SLAB[0]

    def solve(n):
        beta = n * math.pi / b

        def delta(k):
            if resistance is None:
                return 0.0
            omega = k * const.c / 1e-3
            return -resistance / (1j * omega * const.mu_0) / 1e-3

        def right(k, q1, x):
            s, d = a - x, delta(k)
            return (
                np.sin(q1 * s) - d * q1 * np.cos(q1 * s),
                -q1 * (np.cos(q1 * s) + d * q1 * np.sin(q1 * s)),
            )

        def parts(k):
            q1 = np.sqrt(complex(k * k - beta * beta))
            q2 = np.sqrt(complex(eps_c * k * k - beta * beta))
            c1, c2 = np.sin(q1 * x1), q1 * np.cos(q1 * x1) / q2
            w = x2 - x1
            x_2 = c1 * np.cos(q2 * w) + c2 * np.sin(q2 * w)
            dx_2 = -c1 * q2 * np.sin(q2 * w) + c2 * q2 * np.cos(q2 * w)
            return q1, q2, c1, c2, x_2, dx_2

        def f(k):
            q1, _, _, _, x_2, dx_2 = parts(k)
            xr, dxr = right(k, q1, x2)
            return x_2 * dxr - dx_2 * xr

        start = complex(k_h) * (1 if complex(k_h).imag else 1 + 1e-6j)
        k = complex(optimize.newton(f, start, tol=1e-15, maxiter=200))
        q1, q2, c1, c2, x_2, _ = (complex(v) for v in parts(k))
        closing = x_2 / right(k, q1, x2)[0]

        def x_of(x):
            if x <= x1:
                return np.sin(q1 * x)
            if x <= x2:
                return c1 * np.cos(q2 * (x - x1)) + c2 * np.sin(q2 * (x - x1))
            return closing * right(k, q1, x)[0]

        def energy(lo, hi):
            return integrate.quad(
                lambda x: abs(x_of(x)) ** 2, lo, hi, limit=200, epsrel=1e-13
            )[0]

        inside = energy(x1, x2) * _PEPS
        outside = energy(0, x1) + energy(x2, a)
        return k, inside / (inside + outside), abs(abs(k) - beta) < 1e-6

    best = None
    for n in (1, 2, 3):
        try:
            k, p, spurious = solve(n)
        except (RuntimeError, ZeroDivisionError):
            continue
        if not spurious and (best is None or abs(k - k_h) < abs(best[0] - k_h)):
            best = (k, p)
    return best


def _participation(mode):
    def objective(r, _model):
        return r.bulk_participation["block"][mode]

    return objective


def _participation_row(model, result, objective, reference, uncertainty):
    """An estimate and its effectivity, ``None`` past the 20 % rule."""
    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    error = float(objective(result, model)) - reference
    if uncertainty > 0.2 * abs(error):
        return estimate, None
    return estimate, estimate.value / abs(error)


_SLAB_WALLS = {
    "pec": {},
    "tan_delta": {"tan_delta": 0.05},
    "impedance 50": {"end": gfem.Impedance("end", resistance=50.0)},
    "impedance 377": {"end": gfem.Impedance("end", resistance=377.0)},
}


def _slab(directory, m, variant):
    model = _participation_box(directory, m, _SLAB, **_SLAB_WALLS[variant])
    problem = gfem.Eigenmode(target=_PTARGET, n_modes=2)
    result = gfem.solve(model, problem, options=OPTIONS)
    end = _SLAB_WALLS[variant].get("end")
    eps_c = _PEPS * (1 - 1j * _SLAB_WALLS[variant].get("tan_delta", 0.0))
    scale = 1e-3 / const.c
    exact = [
        _slab_exact(w * scale, eps_c, None if end is None else end.resistance)
        for w in np.asarray(result.omega)
    ]
    return model, result, exact


def _participation_times_frequency(mode):
    def objective(r, _model):
        return r.bulk_participation["block"][mode] * _hz(r.omega[mode]) * 1e-11

    return objective


@pytest.mark.parametrize("variant", list(_SLAB_WALLS))
@pytest.mark.parametrize("m", [1, 2])
def test_objective_of_participation_on_the_slab(tmp_path, m, variant):
    # The slab spans the box's section: exact E_z modes (tan_delta and the
    # Robin condition of an Impedance end wall included). Both modes'
    # participation; no interior face crosses E's normal component.
    model, result, exact = _slab(tmp_path, m, variant)
    rows = [
        _participation_row(model, result, _participation(j), exact[j][1], 0.0)
        for j in range(2)
    ]
    _assert_rows("objective of participation", "slab", rows)


@pytest.mark.parametrize("m", [1, 2])
def test_objective_of_participation_times_frequency_on_the_slab(tmp_path, m):
    # p f mixes the two kinds: their errors cancel in J, which the magnitude
    # form cannot see (ADR 0142), so it is quoted on its own
    model, result, exact = _slab(tmp_path, m, "pec")
    rows = []
    for j in range(2):
        k, p = exact[j]
        f = (k * const.c / 1e-3).real / (2 * math.pi) * 1e-11
        objective = _participation_times_frequency(j)
        rows.append(_participation_row(model, result, objective, p * f, 0.0))
    _assert_overestimate("participation times frequency", rows)


@pytest.mark.parametrize("variant", ["pec", "impedance"])
@pytest.mark.parametrize("m", [2, 3])
def test_objective_of_participation_on_the_block(tmp_path, m, variant):
    # A block with 3D corners whose top face crosses E's normal component:
    # the dual flux must be equilibrated with its load (ADR 0142). The
    # Impedance variant holds a 377 ohm/sq top (a very lossy mode).
    top = gfem.Impedance("top", resistance=377.0) if variant == "impedance" else None
    model = _participation_box(tmp_path, m, _BLOCK, top=top)
    problem = gfem.Eigenmode(target=_PTARGET, n_modes=2)
    result = gfem.solve(model, problem, options=OPTIONS)
    rows = [
        _participation_row(model, result, _participation(j), *BLOCK_PARTICIPATION[key])
        for j in range(2)
        if (key := (variant, j)) in BLOCK_PARTICIPATION
    ]
    _assert_rows("objective of participation", "block", rows)
