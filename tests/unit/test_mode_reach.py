"""Mode-field reach without a mesh: the options' conditions and per-mode isolation.

`blocker` names the first condition that leaves mode fields out of reach;
`ModeReach` says which found modes are isolated, built from live modes
(`ModeReach.of`) or from an entry's fields (`ModeReach.from_fields`).
"""

import dataclasses

import numpy as np
import pytest

import gfem
from gfem._eigen.reach import FIELD_TOLERANCE, GapSet, ModeReach, blocker, threshold
from gfem._eigen.refine import Refined
from gfem._eigen.search import ModeSearch
from gfem._linalg.layout import ONE_PROCESS


def test_blocker_names_the_first_condition_in_order():
    loose = gfem.SolverOptions(eigen_tolerance=1e-6)
    assert (
        blocker(loose, "iterative", wave_vector=(1.0, 0.0, 0.0), periodic=True)
        == "wave_vector"
    )
    assert blocker(loose, "iterative", wave_vector=None, periodic=True) == "periodic"
    cuda = gfem.SolverOptions(device="cuda", eigen_tolerance=1e-6)
    assert blocker(cuda, "iterative", wave_vector=None, periodic=False) == "device"
    assert blocker(loose, "iterative", wave_vector=None, periodic=False) == "linear"
    assert blocker(loose, "direct", wave_vector=None, periodic=False) == "tolerance"
    default = gfem.SolverOptions()
    assert default.eigen_tolerance <= FIELD_TOLERANCE
    assert blocker(default, "direct", wave_vector=None, periodic=False) is None


def test_the_polish_subset_ignores_a_loose_eigen_tolerance():
    # the forward polishes at any tolerance; only mode-field gradients need it
    loose = gfem.SolverOptions(eigen_tolerance=1e-6)
    assert (
        blocker(loose, "direct", wave_vector=None, periodic=False, tolerance=False)
        is None
    )
    assert (
        blocker(loose, "iterative", wave_vector=None, periodic=False, tolerance=False)
        == "linear"
    )


def test_threshold_is_the_isolation_gap_or_cluster_rtol_if_larger():
    assert threshold(1e-6) == 1e-4
    assert threshold(1e-3) == 1e-3


def test_a_failed_polish_a_cluster_and_a_cut_cluster_are_not_isolated():
    # modes 0, 1 a cluster; 2 alone but polish failed; 3 alone, its cluster
    # reaches the guard; 4 isolated
    ks = [10.0, 10.0 * (1 + 1e-8), 12.0, 14.0, 16.0]
    guard = 14.0 * (1 + 1e-8)
    gaps = GapSet(np.array(ks, dtype=complex), guard, np.array([], dtype=complex))
    failed = np.array([False, False, True, False, False])
    reach = ModeReach.of(ks, guard, gaps, failed, 1e-6)
    np.testing.assert_array_equal(reach.labels, [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(reach.incomplete, [False, False, False, True, False])
    assert reach.gap[2] == 0.0
    assert reach.gap[4] == pytest.approx(2 / 16)
    np.testing.assert_array_equal(reach.isolated, [False, False, False, False, True])
    assert reach.threshold == 1e-4
    assert reach.labels.dtype == np.int64
    assert reach.incomplete.dtype == bool
    assert reach.polish_failed.dtype == bool
    assert reach.gap.dtype == np.float64


def test_from_fields_reads_back_what_fields_stores():
    ks = [10.0, 12.0]
    gaps = GapSet(np.array(ks, dtype=complex), None, np.array([], dtype=complex))
    reach = ModeReach.of(ks, None, gaps, np.array([True, False]), 1e-6)
    fields = reach.fields()
    assert fields["cluster"] is reach.labels  # the arrays themselves
    back = ModeReach.from_fields({"k": np.array(ks), **fields}, 1e-6)
    for name in ("labels", "incomplete", "gap", "polish_failed"):
        np.testing.assert_array_equal(getattr(back, name), getattr(reach, name))
    np.testing.assert_array_equal(back.isolated, reach.isolated)
    # an entry without polish_failed had no polish fail
    del fields["polish_failed"]
    old = ModeReach.from_fields({"k": np.array(ks), **fields}, 1e-6)
    np.testing.assert_array_equal(old.polish_failed, [False, False])
    assert old.polish_failed.dtype == bool


@dataclasses.dataclass
class _Lossless:
    """A real (lossless, undamped) stub operator: its ``k`` snap real."""

    shift: complex
    nonpolynomial: tuple = ()
    layout = ONE_PROCESS
    seed_damped = False
    general = False
    free = np.ones(4, dtype=bool)

    def reshift(self, k, **_):
        self.shift = k

    def parts(self, x):
        return x, None, x


def test_polishing_reads_the_unsnapped_gap_the_entry_the_snapped_one():
    # mode 0 carries an imaginary part the snap removes: unsnapped, the two
    # modes are 2.06e-4 apart (isolated, both polished); snapped, 5e-5
    # (below the threshold 1e-4, both recorded not isolated)
    ks = [10.0 + 2e-3j, 10.0005 + 0j]
    basis = np.eye(4, dtype=complex)
    polished = []

    def polish(_op, mode, gap, *, index):
        polished.append((index, gap))
        return mode

    options = gfem.SolverOptions()
    search = ModeSearch(
        n_modes=2,
        target=1e9,
        eigen_tolerance=options.eigen_tolerance,
        max_refine_iterations=options.max_refine_iterations,
        cluster_rtol=options.cluster_rtol,
        laws=False,
        eigen_solver=lambda *_, **__: (np.array(ks), basis[:, :2]),
        crowding=lambda: None,
        refine=lambda _op, seed, x, **_: Refined(complex(seed), x, 0.0, 1, True),
        polish=polish,
    )
    found = search.find(_Lossless(10.0), None, 10.0, guard=False, polish_isolated=True)
    unsnapped = abs(ks[0] - ks[1]) / max(abs(ks[0]), abs(ks[1]))
    assert unsnapped > threshold(options.cluster_rtol)
    assert [i for i, _ in polished] == [0, 1]
    assert [gap for _, gap in polished] == pytest.approx([unsnapped, unsnapped])
    assert found.ks == [10.0, 10.0005]
    assert found.reach.gap == pytest.approx([0.0005 / 10.0005] * 2)
    np.testing.assert_array_equal(found.reach.isolated, [False, False])
    np.testing.assert_array_equal(found.polish_failed, [False, False])
    assert found.labels is found.reach.labels
