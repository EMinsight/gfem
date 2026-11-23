r"""Scoring: accuracy against the converged reference, same-code error.

``close`` scores each ``ok`` record once; a scored record is never rescored silently
(the report's ``--rescore`` recomputes beside it).

- The reference is the case directory's ``reference.json``. It is refused (``accuracy:
  null``, check ``reference_matches: fail``) when its ``physics_sha256`` differs from
  the case's, the run's mesh is not in its ``meshes``, the case file changed since
  the run (``definition_sha256``), or the file changed since the run named it.
  Without a file, or with ``method: "none"``, ``accuracy`` is null with reason
  ``no_reference``.
- S (driven): :math:`\max |S_{ij} - S_{\mathrm{ref},ij}|` over the entries and
  frequencies, linear complex, frequencies matched by exact Hz; ``per_item`` is the
  per-frequency maximum.
- Eigenfrequency: :math:`\max_k |\mathrm{Re}\,f_k - f_{\mathrm{ref},k}| /
  f_{\mathrm{ref},k}`, the returned modes at or above the target sorted by real part
  and paired k-th to k-th; fewer modes than the reference fail ``complete``. For
  lossless references :math:`|\mathrm{Im}\,f|/|f|` is kept as an observation.
- ``resolved`` when the error is at least ten times the reference's bound;
  ``accuracy_plausible`` passes when the error is at most the ceiling of the run's
  order (``skipped`` without one). The error itself is never gated.
- Same-code error (solver error, never accuracy): the same metric against the same
  code's direct run of the case in the campaign, on the same stack, solver version
  and order (the adapter's preferred direct kind, else another direct kind; at its
  tightest tolerance), null without one.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from benchmarks.adapters import ADAPTERS
from benchmarks.harness import case as case_mod
from benchmarks.harness import records
from benchmarks.references import document as refdoc

NO_REFERENCE = "no_reference"
S_METRIC = "max_abs_dS"
F_METRIC = "max_rel_dRe_f"
SOLVER_ERROR = "solver error"


class IncompleteError(ValueError):
    """Outputs that do not cover what the reference holds."""


# --- metrics ------------------------------------------------------------------------


def _paired_modes(case: case_mod.Case, outputs: Mapping[str, Any]) -> list[complex]:
    target = case.problem.target_hz or 0.0
    modes = [complex(re, im) for re, im in outputs.get("f_hz") or []]
    return sorted((m for m in modes if m.real >= target), key=lambda m: m.real)


def eigen_error(
    case: case_mod.Case, outputs: Mapping[str, Any], reference: Sequence[float]
) -> tuple[float, list[float], list[float]]:
    """``(max, per mode, |Im f|/|f| per mode)`` of the paired modes."""
    modes = _paired_modes(case, outputs)
    if len(modes) < len(reference):
        raise IncompleteError(
            f"too_few_modes: {len(modes)} at or above the target, "
            f"the reference has {len(reference)}"
        )
    per = [abs(m.real - f) / f for m, f in zip(modes, reference, strict=False)]
    im = [abs(m.imag) / abs(m) if m else 0.0 for m in modes[: len(reference)]]
    return max(per), per, im


def _s_at(outputs: Mapping[str, Any], f: float, key: str) -> complex:
    have = list(outputs.get("frequencies_hz") or [])
    if f not in have:
        raise IncompleteError(f"no output at {f} Hz")
    try:
        re, im = (outputs.get("s") or {})[key][have.index(f)]
    except (KeyError, IndexError) as error:
        raise IncompleteError(f"no S entry {key} at {f} Hz") from error
    return complex(re, im)


def s_error(
    outputs: Mapping[str, Any],
    frequencies: Sequence[float],
    values: Mapping[str, Sequence[Sequence[float]]],
    keys: Sequence[str],
) -> tuple[float, list[float]]:
    r"""``(max, per frequency)`` of :math:`|S - S_\mathrm{ref}|` over ``keys``."""
    per = []
    for i, f in enumerate(frequencies):
        worst = 0.0
        for key in keys:
            if key not in values:
                raise IncompleteError(f"the reference has no S entry {key}")
            ref = complex(*values[key][i])
            worst = max(worst, abs(_s_at(outputs, f, key) - ref))
        per.append(worst)
    return max(per), per


def _keys(case: case_mod.Case) -> list[str]:
    return [f"{to},{frm}" for to, frm in case.quantities.s]


def error(
    case: case_mod.Case, outputs: Mapping[str, Any], reference: Mapping[str, Any]
) -> tuple[str, float, list[float], dict[str, Any], dict[str, Any]]:
    """``(metric, value, per_item, observations, quantity)`` against a reference."""
    quantities = reference.get("quantities") or {}
    if case.problem.kind == "eigen":
        q = quantities.get("f_hz")
        if q is None:
            raise IncompleteError("the reference has no f_hz")
        value, per, im = eigen_error(case, outputs, q["values"])
        return F_METRIC, value, per, {"im_over_abs": im}, q
    q = quantities.get("s")
    if q is None:
        raise IncompleteError("the reference has no s")
    value, per = s_error(outputs, q["frequencies_hz"], q["values"], _keys(case))
    return S_METRIC, value, per, {}, q


# --- one record ---------------------------------------------------------------------


def mismatch(
    record: records.SuiteRecord,
    case: case_mod.Case,
    reference: Mapping[str, Any],
    reference_sha256: str | None = None,
) -> list[str]:
    """Why ``reference`` may not score ``record`` (empty when it may)."""
    found = []
    if reference.get("format") != refdoc.FORMAT:
        found.append(f"reference format {reference.get('format')!r}")
    if reference.get("physics_sha256") != case_mod.physics_sha256(case):
        found.append("physics_sha256 differs from the case's")
    mesh = record.case.mesh_sha256 or case.mesh.sha256
    if mesh not in (reference.get("meshes") or {}):
        found.append(f"mesh {mesh[:12]} is not in the reference's meshes")
    ran = record.case.definition_sha256
    if ran is not None and ran != case_mod.definition_sha256(case):
        found.append("the case file changed since the run (definition_sha256)")
    named = record.case.reference
    if (
        named is not None
        and reference_sha256 is not None
        and named.sha256 != reference_sha256
    ):
        found.append(f"reference.json changed since the run named {named.id}")
    return found


def _null(record: records.SuiteRecord, reason: str) -> None:
    record.accuracy = None
    record.extra["accuracy_reason"] = reason
    record.checks.setdefault("accuracy_plausible", "skipped")


def score(
    record: records.SuiteRecord,
    case: case_mod.Case,
    reference: Mapping[str, Any] | None,
    reference_sha256: str | None = None,
) -> None:
    """Fill ``accuracy`` and checks ``reference_matches``, ``accuracy_plausible``."""
    if record.status != "ok":
        return
    if reference is None:
        record.checks["reference_matches"] = "skipped"
        _null(record, NO_REFERENCE)
        return
    problems = mismatch(record, case, reference, reference_sha256)
    if problems:
        record.checks["reference_matches"] = "fail"
        _null(record, "reference_mismatch: " + "; ".join(problems))
        return
    record.checks["reference_matches"] = "pass"
    if reference.get("method") == "none":
        _null(record, NO_REFERENCE)
        return
    try:
        metric, value, per, seen, q = error(case, record.outputs, reference)
    except IncompleteError as why:
        if str(why).startswith("too_few_modes"):
            record.checks["complete"] = "fail"
        _null(record, str(why))
        return
    bound = float(q["bound"])
    record.accuracy = records.Accuracy(
        metric=metric,
        value=value,
        per_item=per,
        bound=bound,
        resolved=value >= 10 * bound,
        extra=seen,
    )
    record.extra.pop("accuracy_reason", None)
    ceiling = (reference.get("ceiling") or {}).get(str(case.order))
    record.checks["accuracy_plausible"] = (
        "skipped" if ceiling is None else "pass" if value <= ceiling else "fail"
    )


# --- same-code error -----------------------------------------------------------------


def _code_kind(solver: str) -> tuple[str, str]:
    code, _, kind = solver.partition(":")
    return code, kind


def _is_direct(solver: str) -> bool:
    code, kind = _code_kind(solver)
    adapter = ADAPTERS.get(code)
    return (
        adapter is not None
        and kind in adapter.kinds
        and (adapter.kinds[kind].klass == "direct")
    )


def _anchor(rows: Sequence[records.SuiteRecord]) -> records.SuiteRecord | None:
    direct = [r for r in rows if r.status == "ok" and _is_direct(r.run.solver)]
    if not direct:
        return None

    def key(r: records.SuiteRecord) -> tuple[bool, float, str]:
        code, kind = _code_kind(r.run.solver)
        preferred = ADAPTERS[code].direct_kind if code in ADAPTERS else None
        return (preferred != kind, r.run.tolerance or 0.0, r.name)

    return min(direct, key=key)


def _against(
    case: case_mod.Case, outputs: Mapping[str, Any], anchor: Mapping[str, Any]
) -> tuple[str, float, list[float]]:
    if case.problem.kind == "eigen":
        n = case.quantities.modes or len(_paired_modes(case, anchor))
        ref = [m.real for m in _paired_modes(case, anchor)[:n]]
        if len(ref) < n:
            raise IncompleteError("the direct run has too few modes")
        value, per, _ = eigen_error(case, outputs, ref)
        return F_METRIC, value, per
    freqs = list(case.problem.frequencies())
    keys = _keys(case)
    values = {k: [_pair(_s_at(anchor, f, k)) for f in freqs] for k in keys}
    value, per = s_error(outputs, freqs, values, keys)
    return S_METRIC, value, per


def _pair(z: complex) -> list[float]:
    return [z.real, z.imag]


def _group(r: records.SuiteRecord) -> tuple[Any, ...]:
    """Records that share a same-code anchor: case, code, stack, version, order."""
    return (
        r.run.case,
        _code_kind(r.run.solver)[0],
        r.run.stack,
        r.provenance.solver.get("version"),
        r.case.order,
    )


def same_code(
    rows: Iterable[records.SuiteRecord], cases: Mapping[str, case_mod.Case]
) -> list[records.SuiteRecord]:
    """Fill ``same_code_error`` of ok records still without one; return those set."""
    groups: dict[tuple[Any, ...], list[records.SuiteRecord]] = {}
    for r in rows:
        groups.setdefault(_group(r), []).append(r)
    changed = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        name = key[0]
        anchor = _anchor(members)
        case = cases.get(name)
        if anchor is None or case is None:
            continue
        for r in members:
            if r.status != "ok" or r.same_code_error is not None:
                continue
            try:
                metric, value, per = _against(case, r.outputs, anchor.outputs)
            except IncompleteError:
                continue
            r.same_code_error = {
                "metric": metric,
                "value": value,
                "per_item": per,
                "against": anchor.name,
                "label": SOLVER_ERROR,
            }
            changed.append(r)
    return changed


# --- a campaign ----------------------------------------------------------------------


def load_reference(case: case_mod.Case) -> tuple[dict[str, Any] | None, str | None]:
    """The case's ``reference.json`` and its sha256, if there is one.

    The file is the case directory's own, or that of the sibling case its
    ``[reference] case`` key names (:func:`case_mod.reference_directory`).
    """
    directory = case_mod.reference_directory(case)
    if directory is None:
        return None, None
    path = directory / refdoc.FILE
    reference = refdoc.load(path)
    return reference, records.file_sha256(path) if reference is not None else None


def score_campaign(
    rows: Sequence[records.SuiteRecord],
    case_of: Callable[[str], case_mod.Case],
) -> list[records.SuiteRecord]:
    """Score the campaign's unscored ok records; return every record changed."""
    cases: dict[str, case_mod.Case] = {}
    for r in rows:
        if r.run.case not in cases:
            try:
                cases[r.run.case] = case_of(r.run.case)
            except case_mod.CaseError:
                continue
    changed: dict[str, records.SuiteRecord] = {}
    for r in rows:
        case = cases.get(r.run.case)
        if r.status != "ok" or case is None or "reference_matches" in r.checks:
            continue
        reference, sha = load_reference(case)
        score(r, case, reference, sha)
        changed[r.name] = r
    for r in same_code(rows, cases):
        changed[r.name] = r
    return [r for r in rows if r.name in changed]
