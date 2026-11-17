"""`benchmarks.references`: closed forms, committed references, microstrip ladder."""

import cmath
import json
import shutil
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from benchmarks.__main__ import main
from benchmarks.harness import case, config
from benchmarks.references import closed, microstrip
from benchmarks.references import document as refdoc

# Probe 08 exact.out: mpmath at 40 digits, printed by nstr(x, 20) (trailing 0 dropped).
LCAVITY_F1 = "148140269652.0220093"
BRAGG_MODES = ("5185529649.204525515", "7206816174.7901555042")


def _sig(value: Decimal, digits: int) -> str:
    return f"{value:.{digits}g}"


# --- closed forms ------------------------------------------------------------------


def test_lcavity_f1_equals_the_40_digit_value():
    assert _sig(closed.lcavity_f1(), 19) == LCAVITY_F1


def test_bragg_band_edges_equal_the_40_digit_tmm_roots():
    low, high = closed.bragg_modes()
    assert _sig(low, 19) == BRAGG_MODES[0]
    assert _sig(high, 20) == BRAGG_MODES[1]


def test_closed_forms_carry_40_significant_digits():
    (value,) = closed.document(case.load_case(config.CASES_DIR / "lcavity"))[
        "quantities"
    ]["f_hz"]["values_decimal"]
    assert len(value.replace(".", "").lstrip("0")) == 40


# --- case hashes ---------------------------------------------------------------------


def _lcavity() -> case.Case:
    return case.load_case(config.CASES_DIR / "lcavity")


def test_definition_and_physics_hashes_ignore_the_est_s_hints():
    c = _lcavity()
    hinted = replace(c, est_s={"gfem:direct": 1.0})
    assert case.definition_sha256(hinted) == case.definition_sha256(c)
    assert case.physics_sha256(hinted) == case.physics_sha256(c)


def test_physics_hash_ignores_mesh_order_name_and_size_but_not_the_problem():
    c = _lcavity()
    sibling = replace(
        c,
        name="other",
        order=3,
        size="large",
        mesh=replace(c.mesh, sha256="0" * 64, file=None),
    )
    assert case.physics_sha256(sibling) == case.physics_sha256(c)
    assert case.definition_sha256(sibling) != case.definition_sha256(c)
    moved = replace(c, problem=replace(c.problem, target_hz=1e9))
    assert case.physics_sha256(moved) != case.physics_sha256(c)


# --- committed references ------------------------------------------------------------

COMMITTED = ("lcavity", "bragg", "r10M_void")


@pytest.mark.parametrize("name", COMMITTED)
def test_committed_reference_is_valid_and_matches_the_current_pins(name):
    c = case.load_case(config.CASES_DIR / name)
    doc = json.loads((c.directory / "reference.json").read_text())
    assert refdoc.problems(doc) == []
    assert doc["case"]["definition_sha256"] == case.definition_sha256(c)
    assert doc["case"]["mesh_sha256"] == c.mesh.sha256
    assert doc["physics_sha256"] == case.physics_sha256(c)
    assert str(c.order) in doc["meshes"][c.mesh.sha256]["dof"]
    assert doc["id"].startswith(f"{name}-")


def test_r10m_void_has_no_reference_but_pins_its_dof():
    doc = json.loads((config.CASES_DIR / "r10M_void" / "reference.json").read_text())
    assert doc["method"] == "none"
    assert doc["quantities"] == {}
    pinned = doc["meshes"][doc["case"]["mesh_sha256"]]["dof"]["2"]
    assert pinned == 10_248_856


def test_schema_check_names_missing_and_unknown_fields():
    doc = closed.document(_lcavity())
    assert refdoc.problems(doc) == []
    broken = {k: v for k, v in doc.items() if k != "physics_sha256"}
    broken["format"] = "suite-reference/2"
    found = refdoc.problems(broken)
    assert any("physics_sha256" in p for p in found)
    assert any("format" in p for p in found)


def _copy_case(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(config.CASES_DIR / name, target)
    return tmp_path


def test_reference_command_rewrites_an_unchanged_reference_byte_identically(
    tmp_path, capsys
):
    cases = _copy_case(tmp_path, "lcavity")
    path = cases / "lcavity" / "reference.json"
    before = path.read_bytes()
    assert main(["reference", "lcavity", "--cases", str(cases)]) == 0
    assert path.read_bytes() == before
    assert "unchanged" in capsys.readouterr().out


def test_reference_command_gives_a_changed_reference_a_new_identity(tmp_path):
    cases = _copy_case(tmp_path, "lcavity")
    path = cases / "lcavity" / "reference.json"
    doc = json.loads(path.read_text())
    doc["quantities"]["f_hz"]["values"] = [1.0]
    doc["computed_utc"] = "2000-01-01T00:00:00Z"
    path.write_text(json.dumps(doc))
    assert main(["reference", "lcavity", "--cases", str(cases)]) == 0
    new = json.loads(path.read_text())
    assert new["computed_utc"] != "2000-01-01T00:00:00Z"
    assert new["quantities"]["f_hz"]["values"] != [1.0]


def test_reference_command_keeps_dof_pins_added_by_hand(tmp_path):
    cases = _copy_case(tmp_path, "bragg")
    path = cases / "bragg" / "reference.json"
    doc = json.loads(path.read_text())
    sha = doc["case"]["mesh_sha256"]
    doc["meshes"][sha]["dof"]["2"] = 777
    path.write_text(json.dumps(doc))
    assert main(["reference", "bragg", "--cases", str(cases)]) == 0
    assert json.loads(path.read_text())["meshes"][sha]["dof"]["2"] == 777


def test_reference_command_refuses_microstrip_without_a_step(tmp_path, capsys):
    assert main(["reference", "microstrip", "--cases", str(tmp_path)]) == 2
    assert "--plan" in capsys.readouterr().err


# --- microstrip ladder ---------------------------------------------------------------


def _ladder(alpha: float, *, l1p3_offset=0.0, algebraic=0.0):
    """Synthetic rungs S_L = S* + C h^alpha at two frequencies, two entries."""
    exact = [[0.5 + 0.1j, -0.2 + 0.7j], [0.3 - 0.4j, 0.9 + 0.0j]]
    c = [[1e-2 + 2e-3j, -4e-3j], [3e-3, 2e-3 + 1e-3j]]

    def level(n):
        h = 2.0**-n
        return [
            [e + k * h**alpha for e, k in zip(er, cr, strict=True)]
            for er, cr in zip(exact, c, strict=True)
        ]

    l0, l1, l2 = level(0), level(1), level(2)
    l1p3 = [[e + l1p3_offset for e in row] for row in exact]
    l1_iter = [[e + algebraic for e in row] for row in l1]
    return exact, l0, l1, l2, l1p3, l1_iter


def test_richardson_reproduces_the_rate_and_the_exact_value():
    exact, l0, l1, l2, l1p3, l1_iter = _ladder(2.0)
    out = microstrip.extrapolate(l0, l1, l2, l1p3=l1p3, l1_iter=l1_iter, l1_direct=l1)
    assert out.alpha == pytest.approx([2.0, 2.0])
    for got, want in zip(out.values, exact, strict=True):
        assert got == pytest.approx(want, abs=1e-14)
    assert out.terms["finer_order"] == pytest.approx([0.0, 0.0], abs=1e-14)
    assert out.terms["algebraic"] == [0.0, 0.0]


def test_rate_is_clipped_to_1_and_4():
    *_, l0, l1, l2, l1p3, _ = _ladder(5.0)
    assert microstrip.extrapolate(l0, l1, l2, l1p3=l1p3).alpha == [4.0, 4.0]
    *_, l0, l1, l2, l1p3, _ = _ladder(0.5)
    out = microstrip.extrapolate(l0, l1, l2, l1p3=l1p3)
    assert out.alpha == [1.0, 1.0]
    assert out.raw_alpha == pytest.approx([0.5, 0.5])


def test_bound_terms_are_rate_sensitivity_finer_order_and_algebraic():
    _, l0, l1, l2, l1p3, l1_iter = _ladder(2.0, l1p3_offset=3e-5, algebraic=2e-6)
    out = microstrip.extrapolate(l0, l1, l2, l1p3=l1p3, l1_iter=l1_iter, l1_direct=l1)
    # rate sensitivity: the S_ref of alpha 1.5 and 2.5, by hand at frequency 0
    worst = 0.0
    for e in range(2):
        d = l2[0][e] - l1[0][e]
        ref = l2[0][e] + d / 3
        for a in (1.5, 2.5):
            worst = max(worst, abs(l2[0][e] + d / (2**a - 1) - ref))
    assert out.terms["rate"][0] == pytest.approx(worst)
    assert out.terms["finer_order"] == pytest.approx([3e-5, 3e-5])
    assert out.terms["algebraic"] == pytest.approx([2e-6, 2e-6])
    per_f = [max(t[f] for t in out.terms.values()) for f in range(2)]
    assert out.bound_per_frequency == pytest.approx(per_f)
    assert out.bound == pytest.approx(max(per_f))


def test_pilot_rate_estimates_alpha_from_the_finer_order():
    exact, l0, l1, _, _, _ = _ladder(2.0)
    est = microstrip.pilot_rate(l0, l1, exact)
    assert est == pytest.approx([2.0, 2.0])


def test_definition_offset_removes_the_part_that_shrinks_with_h():
    alpha = [2.0, 2.0]
    delta = [[1e-5, 0.0], [0.0, -2e-5j]]
    d0 = [[d + 4e-3 for d in row] for row in delta]
    d1 = [[d + 1e-3 for d in row] for row in delta]
    out = microstrip.definition_offset(d0, d1, alpha)
    assert out["delta_per_frequency"] == pytest.approx([1e-5, 2e-5])
    assert out["delta"] == pytest.approx(2e-5)
    assert out["d_L1"] == pytest.approx(1e-3 + 1e-5)


def test_ladder_rung_cases_share_the_microstrip_physics():
    ms = case.load_case(config.CASES_DIR / "microstrip")
    rungs = {
        name: case.load_case(microstrip.RUNGS_DIR / name) for name in microstrip.RUNGS
    }
    assert {r.family for r in rungs.values()} == {"microstrip"}
    assert {case.physics_sha256(r) for r in rungs.values()} == {case.physics_sha256(ms)}
    assert rungs["microstrip_L0_p2"].mesh.sha256 == ms.mesh.sha256
    assert {r.order for r in rungs.values()} == {1, 2, 3}


def test_plan_stages_name_committed_campaign_files():
    for stage in ("pilot", "l2", "palace-0.18.1.post5", "palace-0.19.0"):
        path = microstrip.campaign_file(stage)
        loaded = config.load_campaign(path)
        assert set(loaded.cases) <= set(microstrip.RUNGS)
    with pytest.raises(config.ConfigError, match="pilot"):
        microstrip.campaign_file("nope")


def test_l2_stage_plans_no_direct_run_on_the_finest_rung():
    loaded = config.load_campaign(microstrip.campaign_file("l2"))
    assert loaded.solvers == ("gfem:bddc",)
    assert loaded.same_code_direct is False
    assert loaded.anchor_tolerance == pytest.approx(1e-11)


def _s_record(name, case_name, solver, s, *, campaign="c1", tol=None, ndof=1, sha="m"):
    freqs = [1e9, 2e9]
    entries = ("p1,p1", "p2,p1")
    return {
        "schema": "suite-record/1",
        "campaign": campaign,
        "name": f"{case_name}/{name}",
        "run": {
            "case": case_name,
            "solver": solver,
            "stack": "wheels",
            "layout": {"asked": "default", "ranks": 1, "threads": 4},
            "core_budget": 4,
            "repeat": 1,
            "tolerance": tol,
        },
        "status": "ok",
        "case": {"mesh_sha256": sha, "order": int(case_name[-1]), "ndof": ndof},
        "outputs": {
            "frequencies_hz": freqs,
            "s": {
                e: [[s[f][i].real, s[f][i].imag] for f in range(2)]
                for i, e in enumerate(entries)
            },
        },
    }


def test_from_records_builds_a_reference_with_offset_and_floors():
    exact, l0, l1, l2, l1p3, _ = _ladder(2.0, l1p3_offset=1e-6)
    p1 = [[v + 0.05 for v in row] for row in exact]
    rows = [
        _s_record("a", "microstrip_L0_p1", "gfem:direct", p1, sha="L0", ndof=10),
        _s_record("b", "microstrip_L0_p2", "gfem:direct", l0, sha="L0", ndof=20),
        _s_record("c", "microstrip_L0_p3", "gfem:direct", l0, sha="L0", ndof=30),
        _s_record("d", "microstrip_L1_p2", "gfem:direct", l1, sha="L1", ndof=40),
        _s_record("e", "microstrip_L1_p3", "gfem:direct", l1p3, sha="L1", ndof=50),
        _s_record(
            "f", "microstrip_L1_p2", "gfem:bddc", l1, campaign="c2", tol=1e-11, sha="L1"
        ),
        _s_record(
            "g", "microstrip_L2_p2", "gfem:bddc", l2, campaign="c2", tol=1e-11, sha="L2"
        ),
        _s_record("h", "microstrip_L0_p2", "palace:mumps", l0, campaign="c3", sha="L0"),
        _s_record("i", "microstrip_L1_p2", "palace:mumps", l1, campaign="c3", sha="L1"),
    ]
    ladder = microstrip.Ladder.from_records(
        rows,
        frequencies=[1e9, 2e9],
        entries=("p1,p1", "p2,p1"),
        palace_versions={"c3": "0.19.0"},
    )
    out = ladder.extrapolate()
    assert out.alpha == pytest.approx([2.0, 2.0])
    floors = ladder.same_mesh_error(out.values)
    assert floors["1"] == pytest.approx(0.05)
    (check,) = ladder.cross_checks(out.alpha)
    assert check["version"] == "0.19.0"
    assert check["delta"] == pytest.approx(0.0, abs=1e-14)
    assert ladder.meshes() == {
        "L0": {"level": 0, "dof": {"1": 10, "2": 20, "3": 30}},
        "L1": {"level": 1, "dof": {"2": 40, "3": 50}},
        "L2": {"level": 2, "dof": {"2": 1}},
    }


def test_from_records_without_l2_is_a_pilot():
    exact, l0, l1, *_ = _ladder(2.0)
    rows = [
        _s_record("b", "microstrip_L0_p2", "gfem:direct", l0),
        _s_record("d", "microstrip_L1_p2", "gfem:direct", l1),
        _s_record("e", "microstrip_L1_p3", "gfem:direct", exact),
    ]
    ladder = microstrip.Ladder.from_records(
        rows, frequencies=[1e9, 2e9], entries=("p1,p1", "p2,p1")
    )
    assert not ladder.complete
    assert ladder.pilot_rate() == pytest.approx([2.0, 2.0])


def test_complex_helpers_are_plain_python():
    assert microstrip.as_complex([[1.0, -2.0]]) == [complex(1, -2)]
    assert cmath.isclose(microstrip.as_complex([[0.0, 1.0]])[0], 1j)


# --- the command's ladder steps ------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_SITE = REPO / "benchmarks" / "site.example.toml"
PLAN = ("reference", "microstrip", "--plan")
MANIFEST = Path(__file__).resolve().parent / "data" / "bench" / "campaign.example.json"


def _planned(rundir: Path) -> list[dict]:
    return [
        json.loads(line)
        for steps in (rundir / "jobs").glob("*.steps")
        for line in steps.read_text().splitlines()
    ]


def test_plan_pilot_writes_a_normal_campaign_of_the_five_direct_rungs(tmp_path):
    out = tmp_path / "pilot"
    code = main([*PLAN, "pilot", "--site", str(EXAMPLE_SITE), "--out", str(out)])
    assert code == 0
    rows = _planned(out)
    assert sorted(r["case"] for r in rows) == sorted(
        set(microstrip.RUNGS) - {"microstrip_L2_p2"}
    )
    assert {r["solver"] for r in rows} == {"gfem:direct"}
    assert (out / "campaign.json").is_file()


def test_plan_l2_adds_no_direct_run_on_the_finest_rung(tmp_path):
    out = tmp_path / "l2"
    code = main([*PLAN, "l2", "--site", str(EXAMPLE_SITE), "--out", str(out)])
    assert code == 0
    rows = _planned(out)
    assert {(r["case"], r["solver"]) for r in rows} == {
        ("microstrip_L1_p2", "gfem:bddc"),
        ("microstrip_L2_p2", "gfem:bddc"),
    }


def _rundir(tmp_path: Path, cid: str, rows: list[dict], palace=None) -> Path:
    rundir = tmp_path / cid
    manifest = json.loads(MANIFEST.read_text())
    manifest["id"] = cid
    manifest["pins"]["solvers"] = {"gfem": "0.1.0"}
    if palace:
        manifest["pins"]["solvers"]["palace"] = palace
    manifest["jobs"][0]["billed_core_h"] = 10.0
    (rundir / "records").mkdir(parents=True)
    (rundir / "campaign.json").write_text(json.dumps(manifest))
    for row in rows:
        row["campaign"] = cid
        path = rundir / "records" / f"{row['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
    return rundir


def _ms_cases(tmp_path: Path) -> tuple[Path, list[float], tuple[str, ...]]:
    cases = _copy_case(tmp_path / "cases", "microstrip")
    ms = case.load_case(cases / "microstrip")
    return (
        cases,
        list(ms.problem.frequencies()),
        tuple(f"{t},{f}" for t, f in ms.quantities.s),
    )


def _full(n_f: int, n_e: int, h: float, *, offset=0.0) -> list[list[complex]]:
    return [
        [
            complex(0.1 * e + 0.01 * f, -0.2) + (1e-2 + 3e-3j) * h**2 + offset
            for e in range(n_e)
        ]
        for f in range(n_f)
    ]


def _ms_record(name, rung, solver, s, *, freqs, entries, tol=None, sha="m", ndof=1):
    row = _s_record(name, rung, solver, [[0j, 0j]] * 2, tol=tol, sha=sha, ndof=ndof)
    row["outputs"] = {
        "frequencies_hz": freqs,
        "s": {
            e: [[s[f][i].real, s[f][i].imag] for f in range(len(freqs))]
            for i, e in enumerate(entries)
        },
    }
    return row


def test_from_rundirs_writes_the_microstrip_reference(tmp_path, capsys):
    cases, freqs, entries = _ms_cases(tmp_path)
    ms = case.load_case(cases / "microstrip")
    n_f, n_e = len(freqs), len(entries)
    L0, L1, L2 = ms.mesh.sha256, "1" * 64, "2" * 64

    def rec(name, rung, solver, h, sha, **kw):
        return _ms_record(
            name,
            rung,
            solver,
            _full(n_f, n_e, h, **kw.pop("off", {})),
            freqs=freqs,
            entries=entries,
            sha=sha,
            **kw,
        )

    pilot = _rundir(
        tmp_path,
        "p",
        [
            rec("a", "microstrip_L0_p1", "gfem:direct", 1.0, L0, off={"offset": 0.05}),
            rec("b", "microstrip_L0_p2", "gfem:direct", 1.0, L0, ndof=315870),
            rec("c", "microstrip_L0_p3", "gfem:direct", 1.0, L0, off={"offset": 1e-3}),
            rec("d", "microstrip_L1_p2", "gfem:direct", 0.5, L1),
            rec("e", "microstrip_L1_p3", "gfem:direct", 0.0, L1, off={"offset": 1e-7}),
        ],
    )
    assert (
        main(["reference", "microstrip", "--from", str(pilot), "--cases", str(cases)])
        == 0
    )
    assert "go: plan the l2 stage" in capsys.readouterr().out
    assert not (cases / "microstrip" / "reference.json").exists()
    l2 = _rundir(
        tmp_path,
        "l",
        [
            rec("f", "microstrip_L1_p2", "gfem:bddc", 0.5, L1, tol=1e-11),
            rec("g", "microstrip_L2_p2", "gfem:bddc", 0.25, L2, tol=1e-11),
        ],
    )
    pal = _rundir(
        tmp_path,
        "q",
        [
            rec("h", "microstrip_L0_p2", "palace:mumps", 1.0, L0, off={"offset": 2e-5}),
            rec("i", "microstrip_L1_p2", "palace:mumps", 0.5, L1, off={"offset": 2e-5}),
        ],
        palace="0.19.0",
    )
    args = ["reference", "microstrip", "--from", str(pilot), str(l2), str(pal)]
    assert main([*args, "--cases", str(cases)]) == 0
    doc = json.loads((cases / "microstrip" / "reference.json").read_text())
    assert refdoc.problems(doc) == []
    assert doc["method"] == "richardson"
    assert doc["physics_sha256"] == case.physics_sha256(ms)
    assert doc["meshes"][L0]["dof"]["2"] == 315870
    terms = doc["provenance"]["rate"]["terms"]
    assert doc["provenance"]["rate"]["alpha"] == pytest.approx([2.0] * n_f)
    assert doc["quantities"]["s"]["bound"] == max(max(t) for t in terms.values())
    assert max(terms["finer_order"]) == pytest.approx(1e-7, rel=1e-6)
    assert doc["same_mesh_error"]["2"] == pytest.approx(abs(1e-2 + 3e-3j))
    assert doc["ceiling"]["3"] == pytest.approx(10 * doc["same_mesh_error"]["3"])
    assert doc["cross_checks"][0]["version"] == "0.19.0"
    assert doc["cross_checks"][0]["delta"] == pytest.approx(2e-5, rel=1e-6)
    assert doc["cost_core_h"] == pytest.approx(30.0)
    assert doc["target"]["met"] is False  # the rate sensitivity is about 4e-4
    assert "unresolved" in capsys.readouterr().out
    assert len(doc["provenance"]["records"]) == 9


def test_ship_sends_the_external_ladder_meshes_once_each():
    from benchmarks.harness import ship

    found = ship.external_cases(config.CASES_DIR, config.REFERENCES_DIR)
    names = [c.name for c in found]
    assert "r10M_void" in names
    assert "microstrip_L2_p2" in names
    assert len({c.mesh.sha256 for c in found}) == len(found)
    assert not any(n.startswith("microstrip_L0") for n in names)
