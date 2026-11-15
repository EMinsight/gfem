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

Overestimates (`OVERESTIMATES`), asserted >= 0.9 and quoted on their own:
an objective at its target, (|S|^2 - |S_h|^2)^2 and |S - S_h|^2 on the
first entry of the lossy line and of the block guide, whose cotangent is
zero, so that only the Hessian remainder is left.

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

The lossy guide's slab at tan_delta 0.05 without offset is not run: its
reference uncertainty exceeds 20 % of every row's error. The top, 175, is
the near-lossless uniform guide's |S21|^2, whose error is second order and
mostly a phase error of S21 that the magnitude form cannot see (ADR 0129).

Electrostatic objectives (ADR 0141), on the split two bars
(`refinement_cases.bars_split_model`, near and screen, n = 4, 8) against
`refinement_cases.BARS_SPLIT_REFERENCE`, and the coax (n = 2, 4, 8)
against `refinement_cases.COAX_C`; the effectivity is the estimate over
``|J(result) - J(reference values)|``:

- ``objective of capacitance``: C_ab, the mutual capacitances to ground,
  the inverse's (a, a) and (a, b), C_ab / C_aa and a sum of squares on the
  bars; C, 1/C, C^2 and (C - C_h)^2 (at its target: the remainder alone)
  on the coax;
- ``objective of electrostatic energy``: each excitation's energy in each
  region (the stationary row, a region holding its own bar, is carried by
  its remainder);
- ``objective of potential means``: each potential's mean over each
  region, and (several reads within the kind) the mean over "left" minus
  the mean over "right", measured here first.

Measured (ostau, 2 threads):

    capacitance, bars     1.65 - 6.30
    capacitance, coax     1.58 - 3.03   (on target 2.50 - 3.03)
    energy, bars          2.28 - 5.52
    means, bars           1.71 - 15.3
    means, two regions    1.65 - 2.18
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
    "objective of capacitance": {"bars": (0.9, 8.0), "coax": (0.9, 8.0)},
    "objective of electrostatic energy": {"bars": (0.9, 7.0)},
    "objective of potential means": {
        "bars": (0.9, 18.0),
        "bars, two regions": (0.9, 18.0),
    },
}

#: Cases asserted >= 0.9 whose high end the docs quote on its own (Limits)
#: instead of widening their kind's range: name -> (kind, (low, high)).
OVERESTIMATES = {
    "objective at its target": ("objective of s", (0.9, 2e5)),
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


# --- electrostatic objectives (ADR 0141) ------------------------------------------
#
# Each objective is written once on a reader of the quantities it needs, and
# evaluated on the result (the objective gfem classifies) and on the
# reference values (the infinitely fine mesh's).

ELECTROSTATIC = gfem.SolverOptions(order=2, linear_solver="direct")


class _Solved:
    """Reads a solved electrostatic result: C, energies and potential means."""

    def __init__(self, result):
        self.result = result

    def c(self):
        return self.result.capacitance

    def energy(self, region, terminal):
        return self.result.energy[region][self.result.terminals.index(terminal)]

    def mean(self, region, terminal):
        return gfem.region_mean(self.result.potential[terminal], region)


class _Reference:
    """The same reads on a case's reference values."""

    def __init__(self, reference, terminals):
        self.reference = reference
        self.terminals = terminals

    def c(self):
        return jnp.asarray(self.reference["C"])

    def energy(self, region, terminal):
        return self.reference["energy"][region][self.terminals.index(terminal)]

    def mean(self, region, terminal):
        return self.reference["mean"][region][self.terminals.index(terminal)]


def _electrostatic_effectivity(model, result, reference, relative, g):
    """``g``'s estimate and effectivity on ``result``; ``None`` if not asserted.

    ``g(reader)`` is the objective; ``relative`` bounds every reference
    value relative to it, carried into J by its cotangents and curvature.
    """

    def objective(r, _model):
        return g(_Solved(r))

    estimate = gfem.error_estimate(
        result, gfem.QuantityEstimator(objective=objective), model=model
    )
    exact = float(g(_Reference(reference, list(result.terminals))))
    error = float(objective(result, model)) - exact
    classified = _objective.classify(objective, result, model)
    bounds = np.array([relative * abs(r.value) for r in classified.reads])
    ct = np.array([abs(r.cotangent) for r in classified.reads])
    uncertainty = ct @ bounds + 0.5 * bounds @ classified.curvature @ bounds
    if error == 0 or uncertainty > 0.2 * abs(error):
        return estimate, None
    return estimate, estimate.value / abs(error)


def _capacitance_objectives():
    return [
        lambda q: q.c()[0, 1],
        lambda q: q.c().sum(axis=1)[0],
        lambda q: q.c().sum(axis=1)[1],
        lambda q: jnp.linalg.inv(q.c())[0, 0],
        lambda q: jnp.linalg.inv(q.c())[0, 1],
        lambda q: q.c()[0, 1] / q.c()[0, 0],
        lambda q: (q.c()[0, 0] ** 2 + q.c()[0, 1] ** 2) / 1e-34,
    ]


def _energy_objectives():
    return [
        (lambda q, r=r, t=t: q.energy(r, t))
        for r in ("left", "right")
        for t in ("a", "b")
    ]


def _mean_objectives():
    return [
        (lambda q, r=r, t=t: q.mean(r, t))
        for r in ("left", "right")
        for t in ("a", "b")
    ]


@pytest.fixture(
    scope="module", params=[(c, n) for c in ("near", "screen") for n in (4, 8)]
)
def split_bars(request, tmp_path_factory):
    case, n = request.param
    model = rc.bars_split_model(tmp_path_factory.mktemp("bars"), case, n)
    result = gfem.solve(model, rc.BARS_PROBLEM, options=ELECTROSTATIC)
    return model, result, rc.BARS_SPLIT_REFERENCE[case]


def _bars_rows(split_bars, objectives):
    model, result, reference = split_bars
    return [
        _electrostatic_effectivity(model, result, reference, rc.BARS_SPLIT_BOUND, g)
        for g in objectives
    ]


def test_objective_of_capacitance_on_the_bars(split_bars):
    rows = _bars_rows(split_bars, _capacitance_objectives())
    _assert_rows("objective of capacitance", "bars", rows)


def test_objective_of_capacitance_on_the_coax(tmp_path):
    rows = []
    c_ref = rc.COAX_C * const.epsilon_0 * rc._QUARTER
    for n in (2, 4, 8):
        model = rc.coax_model(tmp_path, n)
        result = gfem.solve(model, rc.COAX_PROBLEM, options=ELECTROSTATIC)
        c_h = float(result.capacitance[0, 0])
        reference = {"C": ((c_ref,),)}
        objectives = [
            lambda q: q.c()[0, 0],
            lambda q: 1 / q.c()[0, 0],
            lambda q: (q.c()[0, 0] / 1e-17) ** 2,
            lambda q, c_h=c_h: ((q.c()[0, 0] - c_h) / 1e-17) ** 2,
        ]
        relative = rc.COAX_C_BOUND / rc.COAX_C
        rows += [
            _electrostatic_effectivity(model, result, reference, relative, g)
            for g in objectives
        ]
    _assert_rows("objective of capacitance", "coax", rows)


def test_objective_of_electrostatic_energy_on_the_bars(split_bars):
    rows = _bars_rows(split_bars, _energy_objectives())
    _assert_rows("objective of electrostatic energy", "bars", rows)


def test_objective_of_potential_means_on_the_bars(split_bars):
    rows = _bars_rows(split_bars, _mean_objectives())
    _assert_rows("objective of potential means", "bars", rows)


def test_objective_of_potential_means_over_two_regions(split_bars):
    """Several reads within the kind: mean of phi_a over "left" minus "right"."""
    rows = _bars_rows(
        split_bars,
        [
            lambda q: q.mean("left", "a") - q.mean("right", "a"),
            lambda q: q.mean("left", "b") - q.mean("right", "b"),
        ],
    )
    _assert_rows("objective of potential means", "bars, two regions", rows)
