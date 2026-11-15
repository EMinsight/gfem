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

Objectives of S on lossless models with lumped ports (kind ``objective of
s, lossless rewrite``, ADR 0135):

- step line (`refinement_cases.step_model`, 3 and 8 GHz, n = 2, 4; both
  ports excited, and p1 alone): |S21|^2 at each frequency, the band mean of
  1 - |S21|^2, the mean of |S11|^2, |S11|^2 at each frequency and the sheet
  objective, against `refinement_cases.STEP_S` (p1 alone: the S21 ones);
- the 50/30 line, lossless (eps_r 4), n = 4, 8, 16: the lossy line's
  objectives against the same closed form;
- matched and near-matched lines: both ports at Z0 = 94.17 ohm and at 0.9
  and 1.3 Z0, n = 4, 8 (1.05 and 1.1 Z0, and n = 16, in the reference
  tier): the transmission and mixed objectives, |S21|^2 at each frequency,
  the band mean of 1 - |S21|^2, the sheet objective, the absorbed power and
  mean |S21|^2 + |S12|^2.

Every row of the rewrite also passes the inner gate: the rewritten
objective's own estimate C(J_lam) against its own error err(J) - lam . D,
asserted >= 0.9 (not quoted), so the computed defect does not prop it up.

Overestimates (`OVERESTIMATES`), asserted >= 0.9 and quoted on their own:

- an objective at its target, (|S|^2 - |S_h|^2)^2 and |S - S_h|^2 on the
  first entry of the lossy line and of the block guide, whose cotangent is
  zero, so that only the Hessian remainder is left;
- a reflection objective near a reflection zero: the mean of |S11|^2, |S11|^2
  at each frequency and mean |S11|^2 + |S22|^2 on the matched and the 0.9 Z0
  lines (n = 4, 8), where dS11 is mostly orthogonal to a small S11 and no
  column constraint removes it.

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

    lossless rewrite:
    step line             1.76 - 10.3
    50/30 line            1.32 - 24.1
    matched and near      1.00 - 496    (n = 4, 8)
    inner gate            1.19 and up
    reflection near zero  1.19 - 5.63e3 (matched n = 4, 15 GHz: 5.63e3)

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

Objectives of driven fields (ADRs 0132, 0133, 0144), each reading port
p1's excitation through ``-gfem.power_flow`` over a lossy side (kind
``objective of absorbed power``) or ``gfem.region_mean`` (Re of its E_z
and |mean E|^2: ``objective of field means``), against the stored
references of `field_objective_cases`; the dual's flux is recovered
equilibrated with its load:

- block guide (n = 2, 3, 4): the power absorbed in the lossy block and in
  its corner octant, which touch no wall (3D dielectric corners); means
  over the octant and the block, and over two air regions, one sharing the
  block's edge;
- stepped line (n = 4, 6, 8): the power absorbed in a lossy layer and in its
  middle; means over a box touching the re-entrant edge, whose top face
  crosses E_z inside the domain (the case the plain recovered dual flux
  underestimates);
- half-width guide (n = 2, 4) and edge-box line (n = 4, 8): full-height
  regions, the absorbed power and the means over air regions and over a box
  at the edge.

Measured (ostau, 2 threads):

    absorbed, block guide        3.63 - 8.51
    absorbed, stepped line       1.45 - 1.54
    absorbed, half-width guide   20.0 - 24.7   (|sum c| and k0^2 |sum m|
                                                cancel in the signed sum)
    absorbed, edge-box line      4.77 - 5.88
    means, block guide           2.49 - 4.28
    means, block guide, air      5.77 - 55.9   (the same cancellation)
    means, stepped line          1.55 - 2.11
    means, half-width guide      5.24 - 8.01
    means, edge-box line         2.64 - 10.7
"""

import cmath
import dataclasses
import math

import contrast_cases as cc
import field_objective_cases as foc
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
    "objective of s, lossless rewrite": {
        "step line": (0.9, 12.0),
        "50/30 line": (0.9, 28.0),
        "matched and near-matched lines": (0.9, 600.0),
    },
    "objective of capacitance": {"bars": (0.9, 8.0), "coax": (0.9, 8.0)},
    "objective of electrostatic energy": {"bars": (0.9, 7.0)},
    "objective of potential means": {
        "bars": (0.9, 18.0),
        "bars, two regions": (0.9, 18.0),
    },
    "objective of absorbed power": {
        "block guide": (0.9, 10.0),
        "stepped line": (0.9, 10.0),
        "half-width guide": (0.9, 30.0),
        "edge-box line": (0.9, 10.0),
    },
    "objective of field means": {
        "block guide": (0.9, 60.0),
        "block guide, air": (0.9, 60.0),
        "stepped line": (0.9, 60.0),
        "half-width guide": (0.9, 60.0),
        "edge-box line": (0.9, 60.0),
    },
}

#: Cases asserted >= 0.9 whose high end the docs quote on its own (Limits)
#: instead of widening their kind's range: name -> (kind, (low, high)).
OVERESTIMATES = {
    "objective at its target": ("objective of s", (0.9, 2e5)),
    "reflection objective near a reflection zero": (
        "objective of s, lossless rewrite",
        (0.9, 7e3),
    ),
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


def _measured(model, result, reference, bound, objective):
    """The estimate, the true error and the reference's uncertainty in J.

    ``reference`` is the reference S, shaped as ``result.s.values`` (NaN
    where there is none); ``bound`` its uncertainty per entry, carried into
    J as sum |ct| bound plus the remainder of the bound.
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
    return estimate, error, uncertainty


def _ratio(estimate, error, uncertainty):
    """The effectivity, or ``None`` for a row not asserted."""
    if abs(error) < 1e-12 or uncertainty > 0.2 * abs(error):
        return None
    return estimate.value / abs(error)


def _effectivity(model, result, reference, bound, objective):
    """The estimate and its effectivity, or ``None`` for a row not asserted."""
    estimate, error, uncertainty = _measured(model, result, reference, bound, objective)
    return estimate, _ratio(estimate, error, uncertainty)


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


def _line_s(f, eps_r, r1=_R1, r2=_R2):
    """The line's S between its ``r1`` and ``r2`` ports, from its ABCD matrix."""
    beta = 2 * math.pi * f / const.c * cmath.sqrt(eps_r)
    eta0 = math.sqrt(const.mu_0 / const.epsilon_0)
    z0 = eta0 / cmath.sqrt(eps_r) * _LINE["height"] / _LINE["width"]
    c, s = cmath.cos(beta * _LINE["length"]), cmath.sin(beta * _LINE["length"])
    a, b, cc_, d = c, 1j * z0 * s, 1j * s / z0, c
    den = a * r2 + b + cc_ * r1 * r2 + d * r1
    k = math.sqrt(r1 * r2)
    s11 = (a * r2 + b - cc_ * r1 * r2 - d * r1) / den
    s22 = (-a * r2 + b - cc_ * r1 * r2 + d * r1) / den
    return np.array([[s11, 2 * k * (a * d - b * cc_) / den], [2 * k / den, s22]])


def _lossy_line(directory, n, tan_delta=0.3, r1=_R1, r2=_R2):
    path = sm.write_msh22(
        directory / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (n, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=tan_delta)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=r1),
            gfem.LumpedPort("p2", "port2", "+z", resistance=r2),
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


# --- the lossless rewrite (ADR 0135) ----------------------------------------------

#: The line's characteristic impedance, eta0 / sqrt(4) x height / width.
_Z0 = math.sqrt(const.mu_0 / const.epsilon_0) / 2 * _LINE["height"] / _LINE["width"]


def _inner(estimate, error, uncertainty):
    """The rewritten objective's own effectivity, C(J_lam) / |err(J) - lam . D|.

    ``None`` where that error is below 1e-12 or within 5 times the
    reference's uncertainty carried into J.
    """
    defect = estimate._defect
    inner = error - defect
    if abs(inner) < 1e-12 or abs(inner) < 5 * uncertainty:
        return None
    return (estimate.value - abs(defect)) / abs(inner)


def _rewrite_rows(model, result, reference, bound, objectives):
    """(estimate, ratio) per objective, and the inner gate (>= 0.9) on each."""
    rows = []
    for objective in objectives:
        measured = _measured(model, result, reference, bound, objective)
        inner = _inner(*measured)
        assert inner is None or inner >= 0.9, inner
        rows.append((measured[0], _ratio(*measured)))
    return rows


_REWRITE = "objective of s, lossless rewrite"

# step line: 3 and 8 GHz, every objective of ticket 03 but the robust max


def _step_objectives(reflection=True):
    s21, s11 = ("p2", "p1"), ("p1", "p1")
    out = [
        *[(lambda r, _m, f=f: _a2(_s(r, *s21, f))) for f in range(2)],
        lambda r, _m: jnp.mean(1 - _a2(_s(r, *s21))),
    ]
    if reflection:
        out += [
            lambda r, _m: jnp.mean(_a2(_s(r, *s11))),
            *[(lambda r, _m, f=f: _a2(_s(r, *s11, f))) for f in range(2)],
            lambda r, _m: jnp.mean((_a2(_s(r, *s21)) - 0.5) ** 2 + _a2(_s(r, *s11))),
        ]
    return out


def _step_reference(result):
    out = np.full(np.shape(result.s.values), np.nan + 0j)
    for f, value in enumerate(rc.STEP_S.values()):
        out[f, 1, 0], out[f, 0, 0] = value["S21"], value["S11"]
    return out


#: The step reference's largest uncertainty per entry (`STEP_S`).
_STEP_BOUND = 1.4e-6


@pytest.mark.parametrize("excite", [None, ["p1"]])
@pytest.mark.parametrize("n", [2, 4])
def test_objective_of_s_lossless_rewrite_on_the_step_line(tmp_path, n, excite):
    model = rc.step_model(tmp_path, n)
    problem = gfem.Driven(list(rc.STEP_S), excite=excite, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    rows = _rewrite_rows(
        model,
        result,
        _step_reference(result),
        _STEP_BOUND,
        _step_objectives(reflection=excite is None),
    )
    _assert_rows(_REWRITE, "step line", rows)


# the 50/30 line, lossless


@pytest.mark.parametrize("n", [4, 8, 16])
def test_objective_of_s_lossless_rewrite_on_the_50_30_line(tmp_path, n):
    model, result = _lossy_line(tmp_path, n, tan_delta=0.0)
    reference = np.array([_line_s(f, 4.0) for f in LINE_FREQUENCIES])
    rows = _rewrite_rows(model, result, reference, 1e-12, _line_objectives())
    _assert_rows(_REWRITE, "50/30 line", rows)


# matched (both ports at Z0) and near-matched lines (0.9, 1.05, 1.1, 1.3 Z0)


def _transmission_objectives():
    s21, s11 = ("p2", "p1"), ("p1", "p1")
    return [
        *_per_frequency(*s21),
        lambda r, _m: jnp.mean(1 - _a2(_s(r, *s21))),
        lambda r, _m: jnp.mean((_a2(_s(r, *s21)) - 0.5) ** 2 + _a2(_s(r, *s11))),
        lambda r, _m: jnp.mean(1 - _a2(_s(r, *s11)) - _a2(_s(r, *s21))),
        lambda r, _m: jnp.mean(_a2(_s(r, *s21)) + _a2(_s(r, "p1", "p2"))),
    ]


def _reflection_objectives():
    s11 = ("p1", "p1")
    return [
        lambda r, _m: jnp.mean(_a2(_s(r, *s11))),
        *_per_frequency(*s11),
        lambda r, _m: jnp.mean(_a2(_s(r, *s11)) + _a2(_s(r, "p2", "p2"))),
    ]


def _near_line(directory, n, ratio):
    r = ratio * _Z0
    model, result = _lossy_line(directory, n, tan_delta=0.0, r1=r, r2=r)
    reference = np.array([_line_s(f, 4.0, r, r) for f in LINE_FREQUENCIES])
    return model, result, reference


@pytest.fixture(scope="module", params=[(1.0, 4), (1.0, 8), (0.9, 4), (0.9, 8)])
def near_line(request, tmp_path_factory):
    ratio, n = request.param
    return _near_line(tmp_path_factory.mktemp("near"), n, ratio)


@pytest.mark.parametrize("ratio", [1.3])
@pytest.mark.parametrize("n", [4, 8])
def test_objective_of_s_lossless_rewrite_on_a_near_matched_line(tmp_path, ratio, n):
    model, result, reference = _near_line(tmp_path, n, ratio)
    rows = _rewrite_rows(model, result, reference, 1e-12, _transmission_objectives())
    _assert_rows(_REWRITE, "matched and near-matched lines", rows)


def test_objective_of_s_lossless_rewrite_on_the_matched_lines(near_line):
    model, result, reference = near_line
    rows = _rewrite_rows(model, result, reference, 1e-12, _transmission_objectives())
    _assert_rows(_REWRITE, "matched and near-matched lines", rows)


@pytest.mark.reference
@pytest.mark.parametrize(
    ("ratio", "n"),
    [(1.05, 4), (1.05, 8), (1.1, 4), (1.1, 8), *[(r, 16) for r in (1.0, 0.9, 1.3)]],
)
def test_objective_of_s_lossless_rewrite_on_more_near_matched_lines(tmp_path, ratio, n):
    model, result, reference = _near_line(tmp_path, n, ratio)
    rows = _rewrite_rows(model, result, reference, 1e-12, _transmission_objectives())
    _assert_rows(_REWRITE, "matched and near-matched lines", rows)


def test_objective_of_s_lossless_rewrite_of_a_reflection_near_its_zero(near_line):
    model, result, reference = near_line
    rows = _rewrite_rows(model, result, reference, 1e-12, _reflection_objectives())
    _assert_overestimate("reflection objective near a reflection zero", rows)


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


# --- objectives of driven fields (ADRs 0132, 0133, 0144) ---------------------------
#
# Each objective reads the excitation of port p1 at the case's frequency,
# through gfem.power_flow (absorbed power: minus the power leaving a lossy
# side) or gfem.region_mean; the effectivity is the estimate over
# |J(result) - J_ref|, J_ref a stored reference (`field_objective_cases`).


def _field(result):
    return result.e_fields[0]


def _absorbed(regions):
    return lambda r, m: -gfem.power_flow(_field(r), list(regions), model=m)


def _mean_z(regions):
    return lambda r, _m: jnp.real(gfem.region_mean(_field(r), list(regions))[2])


def _mean_squared(regions):
    return lambda r, _m: jnp.sum(_a2(gfem.region_mean(_field(r), list(regions))))


_FIELD_OBJECTIVES = {"abs": _absorbed, "mz": _mean_z, "m2": _mean_squared}


def _field_rows(model, result, references, objectives):
    """Per objective name: its estimate and effectivity (``None``: not asserted).

    ``objectives[name]`` is the objective; ``references[name]`` its
    reference value and uncertainty.
    """
    rows = {}
    for name, objective in objectives.items():
        estimate = gfem.error_estimate(
            result, gfem.QuantityEstimator(objective=objective), model=model
        )
        reference, uncertainty = references[name]
        error = float(objective(result, model)) - reference
        ratio = None if uncertainty > 0.2 * abs(error) else estimate.value / abs(error)
        rows[name] = (estimate, ratio)
    return rows


def _named_objectives(names, regions):
    return {
        name: _FIELD_OBJECTIVES[name.split("_")[0]](regions[name.split("_", 1)[1]])
        for name in names
    }


def _solve_field_case(model, frequency):
    problem = gfem.Driven([frequency], excite=["p1"], keep_fields=True)
    return gfem.solve(model, problem, options=OPTIONS)


@pytest.fixture(scope="module", params=[2, 3, 4])
def block_guide_fields(request, tmp_path_factory):
    model = foc.block_guide_model(tmp_path_factory.mktemp("bguide"), request.param)
    result = _solve_field_case(model, foc.BLOCK_GUIDE_F)
    objectives = _named_objectives(foc.BLOCK_GUIDE_J, foc.BLOCK_GUIDE_REGIONS)
    return _field_rows(model, result, foc.BLOCK_GUIDE_J, objectives)


@pytest.fixture(scope="module", params=[4, 6, 8])
def stepped_line_fields(request, tmp_path_factory):
    model = foc.stepped_line_model(tmp_path_factory.mktemp("sline"), request.param)
    result = _solve_field_case(model, foc.STEPPED_LINE_F)
    objectives = _named_objectives(foc.STEPPED_LINE_J, foc.STEPPED_LINE_REGIONS)
    return _field_rows(model, result, foc.STEPPED_LINE_J, objectives)


def _pick(rows, prefixes, names=None):
    return [
        row
        for name, row in rows.items()
        if name.split("_")[0] in prefixes and (names is None or name in names)
    ]


def test_objective_of_absorbed_power_on_the_block_guide(block_guide_fields):
    rows = _pick(block_guide_fields, ("abs",))
    _assert_rows("objective of absorbed power", "block guide", rows)


def test_objective_of_absorbed_power_on_the_stepped_line(stepped_line_fields):
    rows = _pick(stepped_line_fields, ("abs",))
    _assert_rows("objective of absorbed power", "stepped line", rows)


def test_objective_of_field_means_in_the_block_on_the_block_guide(
    block_guide_fields,
):
    rows = _pick(block_guide_fields, ("mz", "m2"), ("mz_sub", "m2_sub", "mz_blk"))
    _assert_rows("objective of field means", "block guide", rows)


def test_objective_of_field_means_in_air_on_the_block_guide(block_guide_fields):
    names = ("mz_far", "m2_far", "mz_edge", "m2_edge")
    rows = _pick(block_guide_fields, ("mz", "m2"), names)
    _assert_rows("objective of field means", "block guide, air", rows)


def test_objective_of_field_means_at_the_edge_of_the_stepped_line(
    stepped_line_fields,
):
    rows = _pick(stepped_line_fields, ("mz", "m2"))
    _assert_rows("objective of field means", "stepped line", rows)


@pytest.mark.parametrize("n", [2, 4])
def test_absorbed_power_and_field_means_on_the_half_width_guide(tmp_path, n):
    model = foc.half_width_guide_model(tmp_path, n)
    result = _solve_field_case(model, foc.HALF_WIDTH_GUIDE_F)
    regions = {"far": ("rfar",), "edge": ("redge",)}
    objectives = {"absorbed": _absorbed(("block",))} | _named_objectives(
        ("mz_far", "m2_far", "mz_edge", "m2_edge"), regions
    )
    rows = _field_rows(model, result, foc.HALF_WIDTH_GUIDE_J[n], objectives)
    absorbed = rows.pop("absorbed")
    _assert_rows("objective of absorbed power", "half-width guide", [absorbed])
    _assert_rows("objective of field means", "half-width guide", list(rows.values()))


@pytest.mark.parametrize("n", [4, 8])
def test_absorbed_power_and_field_means_on_the_edge_box_line(tmp_path, n):
    model = foc.edge_box_line_model(tmp_path, n)
    result = _solve_field_case(model, foc.EDGE_BOX_LINE_F)
    objectives = {
        "absorbed": _absorbed(("lossy",)),
        "mz": _mean_z(("rbox",)),
        "m2": _mean_squared(("rbox",)),
    }
    rows = _field_rows(model, result, foc.EDGE_BOX_LINE_J, objectives)
    absorbed = rows.pop("absorbed")
    _assert_rows("objective of absorbed power", "edge-box line", [absorbed])
    _assert_rows("objective of field means", "edge-box line", list(rows.values()))
