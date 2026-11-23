"""`benchmarks.harness.score`: accuracy against a converged reference.

The references here are synthetic fixtures built on the committed cases' physics, never
the committed ``reference.json`` files (those are no Baseline).
"""

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.harness import case as case_mod
from benchmarks.harness import config, records, score

BRAGG_F = [5.0e9, 7.0e9]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the benchmark harness runs on Linux and macOS",
)


def _case(name: str) -> case_mod.Case:
    return case_mod.load_case(config.CASES_DIR / name)


def _reference(c: case_mod.Case, quantities: dict, *, method="closed_form", **kw):
    doc = {
        "format": "suite-reference/1",
        "id": f"{c.name}-20261117-0000000",
        "family": c.family,
        "computed_utc": "2026-11-17T00:00:00Z",
        "case": {
            "name": c.name,
            "definition_sha256": case_mod.definition_sha256(c),
            "mesh_sha256": c.mesh.sha256,
        },
        "physics_sha256": case_mod.physics_sha256(c),
        "meshes": {c.mesh.sha256: {"level": 0, "dof": {str(c.order): 1}}},
        "method": method,
        "quantities": quantities,
        "ceiling": {},
        "same_mesh_error": {},
        "cross_checks": [],
        "inputs": {},
        "provenance": {},
        "runs": None,
        "cost_core_h": 0.0,
    }
    doc.update(kw)
    return doc


def _bragg_reference(**kw):
    q = {
        "f_hz": {
            "values": BRAGG_F,
            "bound": 1e-12,
            "bound_kind": "relative",
            "metric": "max_rel_dRe_f",
        }
    }
    return _reference(_case("bragg"), q, **kw)


def _record(c: case_mod.Case, outputs: dict, *, solver="gfem:direct", name="r1"):
    return records.SuiteRecord(
        campaign="c",
        name=f"{c.name}/{name}",
        run=records.Run(
            case=c.name,
            solver=solver,
            stack="wheels",
            layout=records.Layout("default", 1, 4),
            core_budget=4,
            repeat=1,
        ),
        status="ok",
        case=records.CaseFacts(
            definition_sha256=case_mod.definition_sha256(c),
            mesh_sha256=c.mesh.sha256,
            order=c.order,
        ),
        outputs=outputs,
    )


def _modes(*freqs: complex) -> dict:
    return {"f_hz": [[f.real, f.imag] for f in freqs], "q": [None] * len(freqs)}


# --- eigen ---------------------------------------------------------------------------


def test_eigen_error_pairs_modes_in_ascending_real_part_above_the_target():
    c = _case("bragg")
    below = c.problem.target_hz * 0.5
    r = _record(c, _modes(7.0e9 * (1 + 2e-6), below, 5.0e9 * (1 - 1e-6) + 3j))
    score.score(r, c, _bragg_reference())
    assert r.accuracy is not None
    assert r.accuracy.metric == "max_rel_dRe_f"
    assert r.accuracy.per_item == pytest.approx([1e-6, 2e-6])
    assert r.accuracy.value == pytest.approx(2e-6)
    assert r.accuracy.bound == 1e-12
    assert r.accuracy.resolved is True
    assert r.accuracy.extra["im_over_abs"][0] == pytest.approx(3 / 5.0e9, rel=1e-5)
    assert r.checks["reference_matches"] == "pass"


def test_too_few_modes_fails_complete_and_leaves_no_accuracy():
    c = _case("bragg")
    r = _record(c, _modes(5.0e9, 3.0e9))  # one mode at or above the target
    score.score(r, c, _bragg_reference())
    assert r.accuracy is None
    assert r.checks["complete"] == "fail"
    assert r.extra["accuracy_reason"].startswith("too_few_modes")


def test_an_error_below_ten_bounds_is_unresolved():
    c = _case("bragg")
    ref = _bragg_reference()
    ref["quantities"]["f_hz"]["bound"] = 1e-6
    r = _record(c, _modes(5.0e9 * (1 + 5e-6), 7.0e9))
    score.score(r, c, ref)
    assert r.accuracy.resolved is False


def test_plausibility_ceiling_by_order():
    c = _case("bragg")
    ref = _bragg_reference(ceiling={"3": 1e-5})
    ok = _record(c, _modes(5.0e9 * (1 + 1e-6), 7.0e9))
    wrong = _record(c, _modes(5.0e9 * 1.01, 7.0e9))
    score.score(ok, c, ref)
    score.score(wrong, c, ref)
    assert ok.checks["accuracy_plausible"] == "pass"
    assert wrong.checks["accuracy_plausible"] == "fail"
    other = _record(c, _modes(5.0e9, 7.0e9))
    score.score(other, c, _bragg_reference(ceiling={"2": 1e-5}))
    assert other.checks["accuracy_plausible"] == "skipped"


# --- S ------------------------------------------------------------------------------


def _s_outputs(c: case_mod.Case, value) -> dict:
    freqs = list(c.problem.frequencies())
    keys = [f"{t},{f}" for t, f in c.quantities.s]
    return {
        "frequencies_hz": freqs,
        "s": {
            k: [[value(i, k).real, value(i, k).imag] for i in range(len(freqs))]
            for k in keys
        },
    }


def _ms_reference(**kw):
    c = _case("microstrip")
    out = _s_outputs(c, lambda i, _k: complex(0.1 * i, -0.5))
    q = {
        "s": {
            "frequencies_hz": out["frequencies_hz"],
            "values": out["s"],
            "bound": 1e-6,
            "bound_kind": "absolute",
            "metric": "max_abs_dS",
        }
    }
    return _reference(c, q, method="richardson", **kw)


def test_s_metric_is_the_max_complex_difference_over_entries_and_frequencies():
    c = _case("microstrip")

    def run(i, k):
        bump = 3e-4j if (i, k) == (2, "p2,p1") else 1e-4
        return complex(0.1 * i, -0.5) + bump

    r = _record(c, _s_outputs(c, run))
    score.score(r, c, _ms_reference())
    assert r.accuracy.metric == "max_abs_dS"
    assert r.accuracy.value == pytest.approx(3e-4)
    assert r.accuracy.per_item == pytest.approx([1e-4, 1e-4, 3e-4, 1e-4, 1e-4])
    assert r.accuracy.resolved is True


def test_s_frequencies_are_matched_by_exact_hz():
    c = _case("microstrip")
    outputs = _s_outputs(c, lambda i, _k: complex(0.1 * i, -0.5))
    outputs["frequencies_hz"][0] *= 1 + 1e-12
    r = _record(c, outputs)
    score.score(r, c, _ms_reference())
    assert r.accuracy is None
    assert "1000000000.0" in r.extra["accuracy_reason"]


# --- refusals -----------------------------------------------------------------------


def test_physics_hash_mismatch_refuses_to_score():
    c = _case("bragg")
    ref = _bragg_reference(physics_sha256="0" * 64)
    r = _record(c, _modes(5.0e9, 7.0e9))
    score.score(r, c, ref)
    assert r.accuracy is None
    assert r.checks["reference_matches"] == "fail"
    assert r.checks["accuracy_plausible"] == "skipped"
    assert "physics" in r.extra["accuracy_reason"]


def test_mesh_not_in_the_reference_refuses_to_score():
    c = _case("bragg")
    ref = _bragg_reference(meshes={"f" * 64: {"level": 1, "dof": {}}})
    r = _record(c, _modes(5.0e9, 7.0e9))
    score.score(r, c, ref)
    assert r.checks["reference_matches"] == "fail"
    assert "mesh" in r.extra["accuracy_reason"]


def test_a_case_changed_since_the_run_refuses_to_score():
    c = _case("bragg")
    r = _record(c, _modes(5.0e9, 7.0e9))
    r.case.definition_sha256 = "e" * 64
    score.score(r, c, _bragg_reference())
    assert r.checks["reference_matches"] == "fail"


def test_no_reference_file_or_method_none_is_null_with_no_reference():
    c = _case("r10M_void")
    r = _record(c, _s_outputs(c, lambda _i, _k: 0j))
    score.score(r, c, None)
    assert r.accuracy is None
    assert r.extra["accuracy_reason"] == "no_reference"
    assert r.checks["reference_matches"] == "skipped"
    none = _reference(c, {}, method="none", reason="no_reference: too big")
    r2 = _record(c, _s_outputs(c, lambda _i, _k: 0j))
    score.score(r2, c, none)
    assert r2.accuracy is None
    assert r2.extra["accuracy_reason"] == "no_reference"
    assert r2.checks["reference_matches"] == "pass"


def test_records_that_did_not_run_ok_are_left_alone():
    c = _case("bragg")
    r = _record(c, {})
    r.status = "failed"
    score.score(r, c, _bragg_reference())
    assert "reference_matches" not in r.checks


# --- same-code error ----------------------------------------------------------------


def test_same_code_error_is_against_the_codes_direct_run():
    c = _case("bragg")
    direct = _record(c, _modes(5.0e9, 7.0e9), name="gfem-direct")
    bddc = _record(c, _modes(5.0e9 * (1 + 1e-8), 7.0e9), solver="gfem:bddc", name="b")
    bddc.run.tolerance = 1e-8
    palace = _record(c, _modes(5.1e9, 7.0e9), solver="palace:pmg", name="p")
    score.same_code([direct, bddc, palace], {c.name: c})
    assert bddc.same_code_error["value"] == pytest.approx(1e-8)
    assert bddc.same_code_error["against"] == direct.name
    assert bddc.same_code_error["label"] == "solver error"
    assert direct.same_code_error["value"] == 0.0
    assert palace.same_code_error is None  # no Palace direct run in the campaign


def test_same_code_error_prefers_pardiso_and_the_tightest_palace_direct():
    c = _case("microstrip")
    base = _s_outputs(c, lambda i, _k: complex(0.1 * i, -0.5))
    shifted = _s_outputs(c, lambda i, _k: complex(0.1 * i, -0.5) + 1e-9)
    mumps = _record(c, shifted, solver="gfem:petsc-mumps", name="a-mumps")
    pardiso = _record(c, base, name="z-direct")
    bddc = _record(c, base, solver="gfem:bddc", name="bddc")
    score.same_code([mumps, pardiso, bddc], {c.name: c})
    assert bddc.same_code_error["against"] == pardiso.name
    assert mumps.same_code_error["value"] == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("field", "other"),
    [("stack", "native-oss"), ("version", "0.18.1"), ("order", 5)],
)
def test_same_code_anchor_is_per_stack_solver_version_and_order(field, other):
    c = _case("bragg")
    direct = _record(c, _modes(5.0e9, 7.0e9), solver="palace:mumps", name="d")
    elsewhere = _record(c, _modes(5.0e9, 7.0e9), solver="palace:mumps", name="e")
    pmg = _record(c, _modes(5.0e9 * (1 + 1e-8), 7.0e9), solver="palace:pmg", name="p")
    for r in (direct, elsewhere, pmg):
        r.provenance.solver = {"code": "palace", "version": "0.19.0"}
    if field == "stack":
        elsewhere.run.stack = pmg.run.stack = other
    elif field == "version":
        elsewhere.provenance.solver["version"] = other
        pmg.provenance.solver["version"] = other
    else:
        elsewhere.case.order = pmg.case.order = other
    elsewhere.outputs = _modes(5.0e9 * (1 + 1e-9), 7.0e9)
    score.same_code([direct, elsewhere, pmg], {c.name: c})
    assert pmg.same_code_error["against"] == elsewhere.name
    assert direct.same_code_error["against"] == direct.name


def test_same_code_anchor_needs_a_direct_run_in_the_same_group():
    c = _case("bragg")
    direct = _record(c, _modes(5.0e9, 7.0e9), name="d")
    bddc = _record(c, _modes(5.0e9, 7.0e9), solver="gfem:bddc", name="b")
    bddc.run.stack = "native-oss"
    score.same_code([direct, bddc], {c.name: c})
    assert bddc.same_code_error is None


# --- the campaign ---------------------------------------------------------------------


def _cases_with_reference(tmp_path: Path, name: str, doc: dict) -> Path:
    target = tmp_path / name
    shutil.copytree(config.CASES_DIR / name, target)
    (target / "reference.json").write_text(json.dumps(doc))
    return tmp_path


def test_score_campaign_reads_each_cases_reference_and_scores_once(tmp_path):
    cases = _cases_with_reference(tmp_path, "bragg", _bragg_reference())
    c = case_mod.load_case(cases / "bragg")
    r = _record(c, _modes(5.0e9 * (1 + 1e-6), 7.0e9))
    changed = score.score_campaign([r], lambda name: case_mod.load_case(cases / name))
    assert changed == [r]
    assert r.accuracy.value == pytest.approx(1e-6)
    again = copy.deepcopy(r)
    again.accuracy = replace(r.accuracy, value=9.0)
    assert score.score_campaign([again], lambda n: case_mod.load_case(cases / n)) == []
    assert again.accuracy.value == 9.0  # never rescored silently


def test_a_reference_file_changed_since_the_run_refuses_to_score(tmp_path):
    cases = _cases_with_reference(tmp_path, "bragg", _bragg_reference())
    c = case_mod.load_case(cases / "bragg")
    r = _record(c, _modes(5.0e9, 7.0e9))
    r.case.reference = records.Reference(id="bragg-x", sha256="0" * 64)
    score.score_campaign([r], lambda name: case_mod.load_case(cases / name))
    assert r.checks["reference_matches"] == "fail"
    assert "reference.json" in r.extra["accuracy_reason"]


def test_microstrip_p3_scores_against_the_microstrip_reference_with_no_copy(tmp_path):
    cases = _cases_with_reference(tmp_path, "microstrip", _ms_reference())
    shutil.copytree(config.CASES_DIR / "microstrip_p3", cases / "microstrip_p3")
    assert not (cases / "microstrip_p3" / "reference.json").exists()
    p3 = case_mod.load_case(cases / "microstrip_p3")
    assert p3.order == 3
    assert p3.mesh.sha256 == _case("microstrip").mesh.sha256
    doc, sha = score.load_reference(p3)
    assert doc is not None
    assert doc["id"] == "microstrip-20261117-0000000"
    assert sha == records.file_sha256(cases / "microstrip" / "reference.json")
    r = _record(p3, _s_outputs(p3, lambda i, _k: complex(0.1 * i, -0.5) + 2e-4))
    score.score_campaign([r], lambda name: case_mod.load_case(cases / name))
    assert r.checks["reference_matches"] == "pass"
    assert r.extra.get("accuracy_reason") != "no_reference"
    assert r.accuracy is not None
    assert r.accuracy.value == pytest.approx(2e-4)
