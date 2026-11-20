"""`tools/scale_tables.py`: run records to the scaling page's committed snippets."""

import json
import os
from pathlib import Path

import pytest
import scale_tables

COMMIT = "9a57680c0db3f8bbf179aff83db9a9e0805efcc3"


def _record(name, *, nodes, seconds, dofs=2_800_000, **extra):
    """A finished run on 28 ranks per node at 4 threads, the fields a table reads."""
    ranks = 28 * nodes
    return {
        "name": name,
        "case": f"/scratch/meshes/{extra.pop('mesh', 'm_void')}",
        "solver": extra.pop("solver", "bddc_gmres"),
        "options": {"tolerance": 1e-8, "order": 2},
        "frequencies_ghz": [50.0],
        "commit": extra.pop("commit", COMMIT),
        "job": extra.pop("job", "100"),
        "node_count": nodes,
        "ranks": ranks,
        "threads_read": 4,
        "dofs": {
            "owned_sum": dofs,
            "owned_mean": dofs / ranks,
            "owned_min": dofs / ranks - 1000,
            "owned_max": dofs / ranks + 1000,
        },
        "phases": {
            "solve": {"max_s": seconds},
            "gmres": {"max_s": seconds / 2},
            "coarse_apply": {"max_s": seconds / 4},
        },
        "krylov": {"vector_share": 0.1, "reduction_wait_s": 1.0},
        "coarse": [{"dofs": 9000, "ranks": 2}],
        "iterations": {"iterations": [[40, 41]]},
        "memory": {"peak_rss_largest_node_gib": 50.0},
        "ranks_bitwise": True,
        "checks": extra.pop("checks", []),
        **extra,
    }


def _stub(name, *, nodes, state, job, **extra):
    """A run that wrote nothing itself, as the job script closes it."""
    return {
        "name": name,
        "failed": True,
        "mode": "failed",
        "state": state,
        "case": "/scratch/meshes/m_void",
        "solver": "bddc_gmres",
        "node_count": nodes,
        "ranks": 28 * nodes,
        "threads_read": 4,
        "commit": COMMIT,
        "job": job,
        "last_phase": "gmres",
        **extra,
    }


def _write(directory: Path, *records: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for r in records:
        (directory / f"{r['name']}.json").write_text(json.dumps(r))
    return directory


LADDER = {
    "strong": {
        "kind": "ladder",
        "records": ["s_1_r*", "s_2_r*"],
        "base": 1,
    }
}


@pytest.fixture
def ladder(tmp_path):
    return _write(
        tmp_path / "records",
        _record("s_1_r1", nodes=1, seconds=100.0, job="11"),
        _record("s_1_r2", nodes=1, seconds=120.0, job="11"),
        _record("s_1_r3", nodes=1, seconds=110.0, job="11"),
        _record("s_2_r1", nodes=2, seconds=50.0, job="12"),
        _record("s_2_r2", nodes=2, seconds=70.0, job="12"),
        _stub("s_2_r3", nodes=2, state="TIMEOUT", job="12"),
    )


def _table_rows(text: str) -> list[list[str]]:
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    return [[c.strip() for c in row.strip("|").split("|")] for row in rows[1:]]


def test_a_ladder_gives_each_rung_its_median_dof_per_rank_and_tolerance(
    ladder, tmp_path
):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rows = _table_rows(path.read_text())
    one, two = rows[0], rows[1]
    assert one[:2] == ["1", "28 x 4"]
    assert "100 k" in one  # 2.8 M DOF on 28 ranks
    assert "1e-08" in one
    assert "110" in one  # the median of 100, 110, 120
    assert "100-120" in one
    assert two[:2] == ["2", "56 x 4"]
    assert "50 k" in two
    assert "60" in two  # the median of the two that finished
    assert "0.92" in two  # 110 / (60 x 2), against 1 node


def test_a_failed_repeat_is_a_row_of_its_own_with_its_state(ladder, tmp_path):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rows = _table_rows(path.read_text())
    assert len(rows) == 3
    failed = rows[2]
    assert failed[:2] == ["2", "56 x 4"]
    assert "TIMEOUT" in " ".join(failed)
    assert "50 k" in failed  # the mesh's DOF, from the rung's finished runs


def test_a_record_of_another_commit_is_refused(ladder, tmp_path):
    _write(ladder, _record("s_2_r2", nodes=2, seconds=70.0, commit="0" * 40))
    with pytest.raises(SystemExit, match="s_2_r2"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=LADDER
        )


def test_a_missing_record_is_refused(ladder, tmp_path):
    spec = {"strong": {**LADDER["strong"], "records": ["s_1_r*", "s_4_r*"]}}
    with pytest.raises(SystemExit, match="s_4_r"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=spec
        )


def test_a_failed_agreement_gate_is_refused(ladder, tmp_path):
    check = {"gate": 1e-9, "differences": {"s": 2e-9}, "passed": False}
    _write(ladder, _record("s_1_r1", nodes=1, seconds=100.0, checks=[check]))
    with pytest.raises(SystemExit, match="s_1_r1"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=LADDER
        )


def test_the_header_names_the_commit_and_changes_with_any_record(ladder, tmp_path):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "a", commit="9a57680", spec=LADDER
    )
    first = path.read_text().splitlines()[0]
    assert first.startswith("%")
    assert "9a57680" in first
    _write(ladder, _record("s_1_r1", nodes=1, seconds=100.5, job="11"))
    (again,) = scale_tables.write_snippets(
        ladder, tmp_path / "b", commit="9a57680", spec=LADDER
    )
    assert again.read_text().splitlines()[0] != first


def test_a_superseded_record_is_not_read(ladder, tmp_path):
    old = _record("s_1_r1", nodes=1, seconds=999.0, commit="0" * 40)
    (ladder / "s_1_r1.superseded-1.json").write_text(json.dumps(old))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    assert "999" not in path.read_text()


def test_a_table_names_the_machine_and_nodes_its_runs_name(ladder, tmp_path):
    for path in ladder.glob("*.json"):
        record = json.loads(path.read_text())
        record.update(machine="Some Cluster", constraint="highmem")
        path.write_text(json.dumps(record))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    where = path.read_text().splitlines()[2]
    assert "Some Cluster" in where
    assert "highmem" in where


def test_what_renders_names_no_commit_and_no_job(ladder, tmp_path):
    for path in ladder.glob("*.json"):
        record = json.loads(path.read_text())
        record.update(machine="Some Cluster", job="48598488")
        path.write_text(json.dumps(record))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rendered = "\n".join(
        line for line in path.read_text().splitlines() if not line.startswith("%")
    )
    assert "9a57680" not in rendered
    assert "48598488" not in rendered
    assert "Job" not in rendered


DATA = Path(__file__).parent / "data" / "scale_tables"
#: The first scaling set's tables, committed: never regenerated by another set.
SCALING = Path(__file__).parents[2] / "docs" / "project" / "_scaling"
#: The scaling record's directory (records, step logs, mesh checksums), if at hand.
RECORD = os.environ.get("GFEM_SCALING_RECORD")


def _body(path: Path) -> str:
    """A snippet without its header line, which names the records manifest."""
    return path.read_text().split("\n", 2)[2]


def test_the_first_sets_kinds_render_as_they_did(tmp_path):
    records = _write(
        tmp_path / "records",
        _record("s_1_r1", nodes=1, seconds=100.0, job="11"),
        _record("s_1_r2", nodes=1, seconds=120.0, job="11"),
        _record("s_2_r1", nodes=2, seconds=50.0, job="12"),
        _stub("s_2_r3", nodes=2, state="TIMEOUT", job="12"),
    )
    spec = {
        "strong": {"kind": "ladder", "records": ["s_*"], "base": 1},
        "phases": {"kind": "phases", "records": ["s_*"]},
    }
    for path in scale_tables.write_snippets(
        records, tmp_path / "out", commit="9a57680", spec=spec
    ):
        assert _body(path) == (DATA / f"campaign-{path.name}").read_text()


@pytest.mark.skipif(RECORD is None, reason="GFEM_SCALING_RECORD names no record")
def test_the_committed_snippets_regenerate_byte_for_byte(tmp_path):
    record = Path(str(RECORD))
    written = scale_tables.write_snippets(
        record / "records",
        tmp_path,
        commit="9a57680",
        logs=record / "logs" / "steps",
        mesh_md5=[record / "meshes" / "md5.txt", record / "meshes" / "splits.md5"],
    )
    assert sorted(p.name for p in written) == sorted(
        p.name for p in SCALING.glob("*.md")
    )
    for path in written:
        assert path.read_bytes() == (SCALING / path.name).read_bytes(), path.name


SECOND = "c0ffee0" + "0" * 33


def _design(i, *, solve, ordering, ranks, kept, memory, apply=(100.0, 200)):
    """One design of a loop: its phases since the last, its set-up, its memory."""
    seconds, calls = apply
    name = "solve" if i == 0 else f"solve_{i}"
    phases = {
        name: {"max_s": solve, "calls": 1},
        "coarse_setup": {"max_s": solve / 3, "calls": 1},
        "coarse_apply": {"max_s": seconds, "calls": calls},
        "gmres": {"max_s": solve / 2, "calls": 2},
        "gmres_reduction": {"max_s": solve / 20, "calls": 400},
    }
    coarse = {
        "design": i,
        "dofs": 9_000_000,
        "ordering": ordering,
        "reduction": 4,
        "ranks": ranks,
        "columns": 2,
        "seen": i > 0,
        "kept": kept,
        "icntl14": 60,
        "retries": 0,
        "setup_s": solve / 3,
        "infog29": 7_900_000_000,
        "infog7": 5 if ordering == "metis" else 3,
        "info22_mb_node_max": 90_000,
        "factorization_s": 17.0,
        "entries": 7_900_000_000 + i * 100_000_000,
        **({} if kept else {"analysis_s": 400.0}),
    }
    design = {
        "design": i,
        "phases": phases,
        "krylov": {"vector_share": 0.1, "reduction_wait_s": solve / 20},
        "memory": {"peak_rss_largest_node_gib": memory},
    }
    return design, coarse


def _loop(name, *, nodes, designs, mesh="B2_void_split"):
    """A ``--solves K`` record: design 0 is also the run's one-frequency solve."""
    made = [_design(i, **d) for i, d in enumerate(designs)]
    record = _record(
        name,
        nodes=nodes,
        seconds=designs[0]["solve"],
        mesh=mesh,
        commit=SECOND,
        dofs=108_570_046,
    )
    record["designs"] = [d for d, _ in made]
    record["coarse"] = [c for _, c in made]
    record["phases"] = {
        k: {"max_s": sum(d["phases"].get(k, {}).get("max_s", 0.0) for d, _ in made)}
        for k in ("coarse_setup", "coarse_apply", "gmres")
    } | {
        f"solve_{i}" if i else "solve": {"max_s": d["solve"]}
        for i, d in enumerate(designs)
    }
    record["iterations"] = {
        ("iterations" if i == 0 else f"iterations_{i}"): [[64, 65 + i]]
        for i in range(len(designs))
    }
    record["memory"] = {"peak_rss_largest_node_gib": designs[-1]["memory"]}
    record["mumps_printed"] = [{"entries": c["entries"]} for _, c in made]
    return record


def _single(name, *, nodes, solve, memory, mesh="B2_void_split"):
    """A one-frequency record of the second set: one design, PT-Scotch."""
    return _loop(
        name,
        nodes=nodes,
        mesh=mesh,
        designs=[
            {
                "solve": solve,
                "ordering": "pt-scotch",
                "ranks": 56,
                "kept": False,
                "memory": memory,
                "apply": (90.0, 200),
            }
        ],
    )


@pytest.fixture
def coarse_set(tmp_path):
    pts, metis = "pt-scotch", "metis"
    return _write(
        tmp_path / "coarse",
        _loop(
            "c2_bddc_8_r1",
            nodes=8,
            designs=[
                {
                    "solve": 441.0,
                    "ordering": pts,
                    "ranks": 56,
                    "kept": False,
                    "memory": 60.0,
                },
                {
                    "solve": 290.0,
                    "ordering": pts,
                    "ranks": 56,
                    "kept": True,
                    "memory": 70.0,
                },
                {
                    "solve": 288.0,
                    "ordering": pts,
                    "ranks": 56,
                    "kept": True,
                    "memory": 72.0,
                },
            ],
        ),
        _single("c2_bddc_8_r2", nodes=8, solve=421.0, memory=61.0),
        _single(
            "c4_bddc_8_r1", nodes=8, solve=180.0, memory=20.0, mesh="r0.5_void_split"
        ),
        _loop(
            "c2_bddc_16_r1",
            nodes=16,
            designs=[
                {
                    "solve": 450.0,
                    "ordering": pts,
                    "ranks": 56,
                    "kept": False,
                    "memory": 40.0,
                },
                {
                    "solve": 560.0,
                    "ordering": metis,
                    "ranks": 112,
                    "kept": False,
                    "memory": 45.0,
                    "apply": (109.0, 218),
                },
                {
                    "solve": 136.0,
                    "ordering": metis,
                    "ranks": 112,
                    "kept": True,
                    "memory": 46.0,
                    "apply": (109.0, 218),
                },
            ],
        ),
        _single("c2_bddc_16_r2", nodes=16, solve=430.0, memory=41.0),
        _single(
            "c4_bddc_16_r1", nodes=16, solve=210.0, memory=12.0, mesh="r0.5_void_split"
        ),
    )


def _column(text: str, title: str) -> int:
    head = next(line for line in text.splitlines() if line.startswith("| "))
    return [c.strip() for c in head.strip("|").split("|")].index(title)


def test_the_second_set_writes_its_three_snippets(coarse_set, tmp_path):
    written = scale_tables.write_snippets(
        coarse_set, tmp_path / "out", commit="c0ffee0", spec=scale_tables.SETS["coarse"]
    )
    assert sorted(p.name for p in written) == [
        "coarse-design-loop.md",
        "coarse-phases.md",
        "coarse-strong-108m.md",
    ]


def test_the_second_sets_ladder_reads_design_zero_and_names_the_coarse_configuration(
    coarse_set, tmp_path
):
    scale_tables.write_snippets(
        coarse_set, tmp_path / "out", commit="c0ffee0", spec=scale_tables.SETS["coarse"]
    )
    text = (tmp_path / "out" / "coarse-strong-108m.md").read_text()
    rows = _table_rows(text)
    b2_8 = next(r for r in rows if r[0] == "8" and r[3] == "B2_void_split")
    assert b2_8[_column(text, "Repeats")] == "2"
    assert b2_8[_column(text, "`gfem.solve` median, s")] == "431"  # 441 and 421
    assert b2_8[_column(text, "Range, s")] == "421-441"
    assert b2_8[_column(text, "Largest node, GiB")] == "61"  # not the loop's 72
    assert b2_8[_column(text, "Coarse apply median, s")] == "95"  # 100 and 90
    assert b2_8[_column(text, "Coarse ordering")] == "PT-Scotch"
    assert b2_8[_column(text, "Coarse ranks")] == "56"
    assert b2_8[_column(text, "Efficiency against 8 nodes")] == "base"
    b2_16 = next(r for r in rows if r[0] == "16" and r[3] == "B2_void_split")
    assert b2_16[_column(text, "Efficiency against 8 nodes")] == "0.49"  # 431/(440x2)


def test_the_design_loop_has_a_row_per_design(coarse_set, tmp_path):
    scale_tables.write_snippets(
        coarse_set, tmp_path / "out", commit="c0ffee0", spec=scale_tables.SETS["coarse"]
    )
    text = (tmp_path / "out" / "coarse-design-loop.md").read_text()
    rows = _table_rows(text)
    assert [(r[0], r[1]) for r in rows] == [
        (n, d) for n in ("8", "16") for d in ("0", "1", "2")
    ]
    later = rows[5]  # 16 nodes, design 2

    def cell(title):
        return later[_column(text, title)]

    assert cell("Coarse ordering") == "METIS"
    assert cell("Coarse ranks") == "112"
    assert cell("Analysis kept") == "yes"
    assert cell("Coarse analysis, s") == "-"
    assert cell("Coarse factorization, s") == "17"
    assert cell("Coarse apply per call, s") == "0.50"  # 109 s over 218 calls
    assert cell("`gfem.solve`, s") == "136"
    assert cell("Running total, s") == "1146"  # 450 + 560 + 136
    assert cell("Reduction wait in `gfem.solve`") == "5.0%"
    assert cell("Iterations, ports 1/2") == "64/67"
    assert cell("Largest node after this design, GiB") == "46"
    first = rows[3]
    assert first[_column(text, "Analysis kept")] == "no"
    assert first[_column(text, "Coarse analysis, s")] == "400"
    assert "$1 + 10^{-3} i$" in text


def test_the_second_sets_phases_read_design_zero(coarse_set, tmp_path):
    scale_tables.write_snippets(
        coarse_set, tmp_path / "out", commit="c0ffee0", spec=scale_tables.SETS["coarse"]
    )
    text = (tmp_path / "out" / "coarse-phases.md").read_text()
    rows = _table_rows(text)
    b2_8 = next(r for r in rows if r[1] == "8" and r[0] == "B2_void_split")
    assert b2_8[_column(text, "`gfem.solve`, s")] == "431"
    assert b2_8[_column(text, "Coarse setup, s")] == "144"  # 147 and 140.33
    assert b2_8[_column(text, "Coarse ranks")] == "56"
    b2_16 = next(r for r in rows if r[1] == "16" and r[0] == "B2_void_split")
    assert b2_16[_column(text, "Coarse factor entries")] == "7.9e+09"  # not design 1's


def test_the_second_sets_ladder_keeps_each_meshs_rows_together(coarse_set, tmp_path):
    scale_tables.write_snippets(
        coarse_set, tmp_path / "out", commit="c0ffee0", spec=scale_tables.SETS["coarse"]
    )
    rows = _table_rows((tmp_path / "out" / "coarse-strong-108m.md").read_text())
    assert [(r[3], r[0]) for r in rows] == [
        ("B2_void_split", "8"),
        ("B2_void_split", "16"),
        ("r0.5_void_split", "8"),
        ("r0.5_void_split", "16"),
    ]


def test_the_set_defaults_to_the_first(tmp_path, ladder, monkeypatch):
    assert scale_tables.SETS["campaign"] is scale_tables.CAMPAIGN
    seen = {}
    monkeypatch.setattr(
        scale_tables, "write_snippets", lambda *_, **k: seen.update(k) or []
    )
    monkeypatch.setattr(scale_tables, "manifest", lambda _: "")
    scale_tables.main([str(ladder), str(tmp_path), "--commit", "9a57680"])
    assert seen["spec"] is scale_tables.CAMPAIGN
    scale_tables.main(
        [str(ladder), str(tmp_path), "--commit", "9a57680", "--set", "coarse"]
    )
    assert seen["spec"] is scale_tables.SETS["coarse"]
