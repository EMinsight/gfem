"""Duration-balanced shards: the partition function and the conftest hook."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import shards

ROOT = Path(__file__).resolve().parents[2]


def test_junit_key_function():
    assert (
        shards.junit_key("tests/unit/test_a.py::test_f") == "tests.unit.test_a::test_f"
    )


def test_junit_key_class_and_param_ids():
    nodeid = "tests/verification/test_b.py::TestC::test_g[a-1/2]"
    assert shards.junit_key(nodeid) == "tests.verification.test_b.TestC::test_g[a-1/2]"


def test_junit_key_param_id_with_double_colon():
    nodeid = "tests/a/test_b.py::TestC::test_g[x::y/z-1]"
    assert shards.junit_key(nodeid) == "tests.a.test_b.TestC::test_g[x::y/z-1]"
    assert shards.junit_key("tests/a/test_b.py::test_g[p::q]") == (
        "tests.a.test_b::test_g[p::q]"
    )


def test_assign_covers_every_test_once_and_is_deterministic():
    costs = {f"t{i}": float(i % 7) for i in range(50)}
    first = shards.assign(costs, 4)
    assert set(first) == set(costs)
    assert set(first.values()) <= set(range(4))
    assert shards.assign(dict(reversed(list(costs.items()))), 4) == first


def test_assign_balances_a_known_case():
    costs = {"a": 8.0, "b": 7.0, "c": 6.0, "d": 5.0, "e": 4.0}
    owner = shards.assign(costs, 2)
    # LPT: a->0, b->1, c->1 (7<8), d->0 (8<13), e->0 (13 vs 13: tie goes to 0)
    assert owner == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 0}


def test_assign_ties_go_to_the_lowest_shard_and_lowest_nodeid_first():
    owner = shards.assign({"b": 1.0, "a": 1.0, "c": 1.0}, 2)
    assert owner == {"a": 0, "b": 1, "c": 0}


def test_unknown_tests_cost_the_median():
    known = {"m.a::f": 1.0, "m.a::g": 3.0, "m.a::h": 100.0}
    costs = shards.costs_of(["m/a.py::f", "m/a.py::new"], known)
    assert costs == {"m/a.py::f": 1.0, "m/a.py::new": 3.0}


def test_default_ignores_recorded_zeros():
    known = {f"m.a::z{i}": 0.0 for i in range(8)} | {"m.a::p": 2.0, "m.a::q": 4.0}
    assert shards.costs_of(["m/a.py::new"], known) == {"m/a.py::new": 3.0}
    assert shards.costs_of(["m/a.py::new"], {"m.a::z": 0.0}) == {"m/a.py::new": 1.0}


def test_zero_cost_tests_spread_evenly():
    owner = shards.assign({f"t{i:02d}": 0.0 for i in range(20)}, 4)
    assert sorted(list(owner.values()).count(s) for s in range(4)) == [5] * 4


def test_no_file_costs_one_second(tmp_path):
    assert shards.load(tmp_path / "missing.json") == {}
    assert shards.costs_of(["m/a.py::f"], {}) == {"m/a.py::f": 1.0}


def test_load_reads_the_file(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"m.a::f": 2.5}))
    assert shards.load(path) == {"m.a::f": 2.5}


def test_committed_durations_are_sorted_and_nonnegative():
    text = (ROOT / "tests" / "durations.json").read_text()
    data = json.loads(text)
    assert text == json.dumps(data, indent=0, sort_keys=True) + "\n"
    assert all(value >= 0 for value in data.values())


def test_conftest_shards_partition_a_collection():
    """The shards of one file's collection are disjoint and complete."""
    target = "tests/unit/test_shards.py"

    def collect(*extra):
        run = subprocess.run(
            [sys.executable, "-m", "pytest", target, "--collect-only", "-q", *extra],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert run.returncode in (0, 5), run.stderr  # 5: this shard is empty
        return {line for line in run.stdout.splitlines() if "::" in line}

    whole = collect()
    parts = [collect("--shard", f"{i}/3") for i in range(3)]
    assert set().union(*parts) == whole
    assert sum(len(p) for p in parts) == len(whole)


@pytest.mark.parametrize("bad", ["3", "a/b", "3/3", "-1/2"])
def test_bad_shard_is_a_usage_error(bad):
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", f"--shard={bad}"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 4
    assert "--shard" in run.stderr


def test_recorded_zeros_cost_the_floor_and_spread():
    known = {f"m.a::z{i}": 0.0 for i in range(40)} | {"m.a::big": 10.0}
    ids = [f"m/a.py::z{i}" for i in range(40)] + ["m/a.py::big"]
    costs = shards.costs_of(ids, known)
    assert costs["m/a.py::z0"] == shards.MIN_SECONDS
    owner = shards.assign(costs, 4)
    counts = [sum(1 for s in owner.values() if s == k) for k in range(4)]
    assert counts[0] == 1  # the long test's shard takes nothing else
    assert sorted(counts[1:]) == [13, 13, 14]
